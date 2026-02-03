"""Train a minimal T2V diffusion model on cached latents."""

from __future__ import annotations

import argparse
import copy
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# * Silence HuggingFace tokenizers fork warning (common with DataLoader workers / torchrun).
# * Users can override by explicitly setting TOKENIZERS_PARALLELISM in the environment.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from mmdit.config import load_yaml_config
from mmdit.data.latents_cache import BucketBatchSampler, LatentsCacheDataset, collate_latents
from mmdit.diffusion.flow_matching import FlowMatchingConfig, make_xt_and_vt, sample_t, time_embedding_value
from mmdit.diffusion.sampling import DdimConfig, FlowOdeConfig, ddim_sample_eps, flow_ode_sample, make_ddim_timesteps
from mmdit.diffusion.schedule import add_noise, make_schedule
from mmdit.lora import LoRAConfig, apply_lora_
from mmdit.models.video_mmdit import VideoMMDiT, VideoMMDiTConfig
from mmdit.text_encoders.hash_encoder import HashTextEncoder, HashTextEncoderConfig
from mmdit.text_encoders.hf_encoder import HfTextEncoder, HfTextEncoderConfig
from mmdit.tokenizers.vtp import VTPTokenizer, VTPTokenizerConfig
from mmdit.training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from mmdit.utils import (
    autocast_dtype,
    barrier,
    get_rank,
    is_main_process,
    local_rank_from_env,
    maybe_init_distributed,
    select_device,
    set_global_seed,
)


@torch.no_grad()
def _update_ema_(ema_model: torch.nn.Module, model: torch.nn.Module, *, decay: float) -> None:
    """Updates `ema_model` in-place using exponential moving average.

    EMA is applied to all floating-point tensors in the state_dict (params + buffers).
    Non-floating tensors (e.g., indices) are copied verbatim.
    """

    d = float(decay)
    if not (0.0 <= d < 1.0):
        raise ValueError(f"ema.decay must be in [0, 1), got {decay}")
    ema_sd = ema_model.state_dict()
    model_sd = model.state_dict()
    for k, v in ema_sd.items():
        src = model_sd[k]
        if torch.is_floating_point(v):
            v.mul_(d).add_(src, alpha=1.0 - d)
        else:
            v.copy_(src)


def _build_text_encoder(cfg: dict[str, Any], *, device: torch.device) -> tuple[torch.nn.Module, int]:
    enc_type = str(cfg.get("type", "hash")).lower()
    if enc_type == "hash":
        enc_cfg = HashTextEncoderConfig(
            dim=int(cfg.get("dim", 512)),
            max_length=int(cfg.get("max_length", 128)),
            vocab_size=int(cfg.get("vocab_size", 65536)),
            trainable=bool(cfg.get("trainable", False)),
        )
        enc = HashTextEncoder(enc_cfg).to(device)
        return enc, int(enc_cfg.dim)
    if enc_type == "hf":
        name_or_path = str(cfg.get("name_or_path", "")).strip()
        if not name_or_path:
            raise ValueError("text_encoder.name_or_path must be set for type=hf.")
        enc = HfTextEncoder(
            HfTextEncoderConfig(
                name_or_path=name_or_path,
                max_length=int(cfg.get("max_length", 77)),
                freeze=bool(cfg.get("freeze", True)),
                use_processor=bool(cfg.get("use_processor", False)),
                trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            ),
            device=device,
        )
        return enc, int(enc.dim)
    raise ValueError(f"Unsupported text_encoder.type: '{enc_type}'")


def _save_sample_grid(
    out_path: Path,
    *,
    images: torch.Tensor,
    captions: list[str] | None = None,
    max_frames: int = 8,
) -> None:
    """Saves a grid/filmstrip image.

    Args:
        out_path: Destination PNG path.
        images: (B, F, 3, H, W) in [0, 1].
        captions: Optional list of prompt strings (length B). When provided, each
          prompt is drawn above its corresponding filmstrip.
        max_frames: Max frames to include per sample (filmstrip width).
    """

    try:
        from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-error
    except (ImportError, OSError) as e:
        raise RuntimeError("Pillow is required to save sample images.") from e

    if images.ndim != 5:
        raise ValueError(f"Expected images (B,F,3,H,W), got {tuple(images.shape)}")
    b, f, c, h, w = images.shape
    if int(c) != 3:
        raise ValueError("Expected RGB images (C=3).")

    def _to_pil_rgb(x: torch.Tensor) -> Image.Image:
        if x.ndim != 3 or int(x.shape[0]) != 3:
            raise ValueError(f"Expected tensor (3,H,W), got {tuple(x.shape)}")
        x = x.detach().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8).cpu()
        arr = x.permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(arr)

    def _load_font(*, size: int) -> ImageFont.ImageFont:
        # * Try common system fonts first so Unicode prompts render when available.
        candidates = [
            "usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for p in candidates:
            try:
                for candidate in (Path(p), Path("/") / p):
                    if candidate.is_file():
                        return ImageFont.truetype(str(candidate), size=int(size))
            except Exception:  # noqa: BLE001
                pass
        return ImageFont.load_default()

    def _wrap_text_to_width(
        text: str,
        *,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.ImageFont,
        max_width_px: int,
        max_lines: int,
    ) -> list[str]:
        t = " ".join(str(text).replace("\n", " ").split()).strip()
        if not t:
            return []

        # * Word-wrap for space-separated languages; char-wrap otherwise.
        use_words = (" " in t)
        units = t.split(" ") if use_words else list(t)
        joiner = " " if use_words else ""

        lines: list[str] = []
        cur = ""
        for u in units:
            cand = (cur + joiner + u) if cur else u
            try:
                width_px = int(draw.textlength(cand, font=font))
            except Exception:  # noqa: BLE001
                width_px = int(font.getsize(cand)[0])  # type: ignore[attr-defined]
            if width_px <= int(max_width_px):
                cur = cand
                continue
            if cur:
                lines.append(cur)
                if len(lines) >= int(max_lines):
                    break
            cur = u

        if len(lines) < int(max_lines) and cur:
            lines.append(cur)

        if len(lines) > int(max_lines):
            lines = lines[: int(max_lines)]
        if len(lines) == int(max_lines) and units:
            # * Indicate truncation when the prompt is longer than the allocated lines.
            if use_words and (" ".join(lines) != t):
                lines[-1] = (lines[-1] + " ...") if lines[-1] else "..."
            elif (not use_words) and ("".join(lines) != t):
                lines[-1] = (lines[-1] + " ...") if lines[-1] else "..."
        return lines

    f_show = min(int(f), int(max_frames))
    strips: list[Image.Image] = []
    for i in range(int(b)):
        # (F,3,H,W) -> (3,H,W*F)
        strip_t = torch.cat([images[i, j] for j in range(f_show)], dim=2)
        strip_img = _to_pil_rgb(strip_t)

        caption = captions[i] if captions is not None and i < len(captions) else ""
        if caption:
            font = _load_font(size=16)
            pad = 6
            draw = ImageDraw.Draw(strip_img)
            lines = _wrap_text_to_width(
                caption,
                draw=draw,
                font=font,
                max_width_px=int(strip_img.width - 2 * pad),
                max_lines=4,
            )
            if lines:
                try:
                    bbox = font.getbbox("Ag")  # type: ignore[attr-defined]
                    line_h = int(bbox[3] - bbox[1])
                except Exception:  # noqa: BLE001
                    line_h = int(font.getsize("Ag")[1])  # type: ignore[attr-defined]
                band_h = int(line_h * len(lines) + 2 * pad)
                band = Image.new("RGB", (strip_img.width, band_h), color=(255, 255, 255))
                band_draw = ImageDraw.Draw(band)
                band_draw.multiline_text((pad, pad), "\n".join(lines), fill=(0, 0, 0), font=font)
                combined = Image.new("RGB", (strip_img.width, band_h + strip_img.height), color=(0, 0, 0))
                combined.paste(band, (0, 0))
                combined.paste(strip_img, (0, band_h))
                strip_img = combined

        strips.append(strip_img)

    # * Stack vertically with small padding to mimic torchvision's default.
    pad = 2
    out_w = max(img.width for img in strips)
    out_h = pad + sum(img.height + pad for img in strips)
    canvas = Image.new("RGB", (out_w + 2 * pad, out_h), color=(0, 0, 0))
    y = pad
    for img in strips:
        x = pad + (out_w - img.width) // 2
        canvas.paste(img, (int(x), int(y)))
        y += img.height + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train T2V diffusion on cached VTP latents.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--train_cache_dir", type=str, required=True, help="Directory of cached .pt files.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    maybe_init_distributed()
    set_global_seed(int(cfg.get("seed", 42)) + int(get_rank()))
    local_rank = local_rank_from_env(0)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    # * Config sections
    video_cfg = cfg.get("video", {}) or {}
    tok_cfg = cfg.get("tokenizer", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    diff_cfg = cfg.get("diffusion", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    lora_cfg = cfg.get("lora", {}) or {}
    tread_cfg = cfg.get("tread", {}) or {}

    num_frames = int(video_cfg.get("num_frames", 16))
    frame_stride = int(video_cfg.get("frame_stride", 1))

    latent_d = int(tok_cfg.get("latent_d", 64))
    latent_f = int(tok_cfg.get("latent_f", 16))
    latent_scale = float(tok_cfg.get("latent_scale", 1.0) or 1.0)
    if latent_scale <= 0:
        raise ValueError(f"tokenizer.latent_scale must be > 0, got {latent_scale}")
    width = int(video_cfg.get("width", 256))
    height = int(video_cfg.get("height", 256))
    buck_cfg = cfg.get("bucketing", {}) or {}
    bucketing_enabled = bool(buck_cfg.get("enabled", False))

    # * Dataset
    ds = LatentsCacheDataset(
        args.train_cache_dir,
        num_frames=num_frames,
        frame_stride=frame_stride,
        cfg_dropout_prob=float(train_cfg.get("cfg_dropout_prob", 0.0)),
    )

    batch_size = int(train_cfg.get("batch_size", 1))
    if (not bucketing_enabled) and ds.num_buckets > 1 and batch_size > 1:
        raise ValueError(
            "Multi-bucket cache detected but bucketing is disabled.\n"
            f"train_cache_dir={args.train_cache_dir}\n"
            f"num_buckets={ds.num_buckets}\n"
            "Fix options:\n"
            "  1) Set bucketing.enabled=true in the YAML config (recommended), or\n"
            "  2) Use train.batch_size=1 (slow), or\n"
            "  3) Re-cache into a single fixed resolution."
        )

    epoch_sampler = None
    if bucketing_enabled:
        rank = int(get_rank())
        world_size = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
        batch_sampler = BucketBatchSampler(
            ds.bucket_to_indices,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            seed=int(cfg.get("seed", 42)),
            rank=rank,
            world_size=world_size,
        )
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_latents,
        )
        epoch_sampler = batch_sampler
    else:
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=(device.type == "cuda"),
            drop_last=True,
            collate_fn=collate_latents,
        )
        epoch_sampler = sampler

    if len(loader) <= 0:
        raise ValueError(
            "Training DataLoader has 0 batches. This usually means your cache is too small for the configured "
            "batch_size/drop_last/world_size.\n"
            f"train_cache_dir={args.train_cache_dir}\n"
            f"batch_size={batch_size}\n"
            f"drop_last=True\n"
            f"num_files={len(ds)}"
        )

    # * Text encoder
    text_enc, text_dim = _build_text_encoder(cfg.get("text_encoder", {}) or {}, device=device)

    # * Model
    net = VideoMMDiT(
        VideoMMDiTConfig(
            latent_d=latent_d,
            model_dim=int(model_cfg.get("dim", 512)),
            depth=int(model_cfg.get("depth", 8)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            window_size=int(model_cfg.get("window_size", 8)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            text_dim=int(text_dim),
        )
    ).to(device)
    # * Optional: enable TREAD-style token routing (training acceleration).
    if hasattr(net, "configure_tread"):
        net.configure_tread(tread_cfg)  # type: ignore[attr-defined]

    # * LoRA (optional)
    lora_applied = apply_lora_(
        net,
        LoRAConfig(
            enabled=bool(lora_cfg.get("enabled", False)),
            rank=int(lora_cfg.get("rank", 16)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=[str(x) for x in (lora_cfg.get("target_modules", []) or [])],
            train_lora_only=bool(lora_cfg.get("train_lora_only", True)),
        ),
    )
    if is_main_process() and lora_applied:
        print(f"[lora] wrapped {lora_applied} Linear layers")

    ddp_model: torch.nn.Module = net
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ddp_model = DDP(net, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

    # * Optimizer
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        betas=(0.9, 0.95),
    )

    # * Resume
    exp_name = str(cfg.get("experiment_name", "t2v_mmdit"))
    out_root = Path(str(cfg.get("output_dir", "runs")))
    exp_dir = out_root / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    if args.resume:
        latest = find_latest_checkpoint(ckpt_dir, prefix="step")
        if latest is not None:
            state = load_checkpoint(latest, model=ddp_model, optimizer=optim, map_location=device)
            global_step = state.global_step
            if is_main_process():
                print(f"[resume] {latest} (global_step={global_step})")

    # * Diffusion schedule
    sched = make_schedule(int(diff_cfg.get("num_timesteps", 1000)), schedule=str(diff_cfg.get("schedule", "cosine")))
    num_timesteps = int(sched.betas.shape[0])

    # * EMA (optional): keeps a shadow copy of the model for better sampling quality.
    ema_cfg = cfg.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.9999))
    ema_update_every = int(ema_cfg.get("update_every", 1))
    ema_start_step = int(ema_cfg.get("start_step", 0))
    ema_use_for_sampling = bool(ema_cfg.get("use_for_sampling", True))
    ema_model = None
    if ema_enabled:
        if ema_update_every <= 0:
            raise ValueError("ema.update_every must be > 0.")
        if ema_start_step < 0:
            raise ValueError("ema.start_step must be >= 0.")
        if not (0.0 <= float(ema_decay) < 1.0):
            raise ValueError(f"ema.decay must be in [0, 1), got {ema_decay}")
        ema_model = copy.deepcopy(net).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # * Precision
    dtype = autocast_dtype(str(train_cfg.get("precision", "bf16")))
    amp_enabled = (dtype is not None) and device.type == "cuda"

    # * Train loop
    max_steps = int(train_cfg.get("max_steps", 1000))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every_steps", 50))
    save_every = int(train_cfg.get("save_every_steps", 500))

    # * Sampling during training (rank0 only).
    sample_every = int(train_cfg.get("sample_every_steps", 0) or 0)
    sample_num = int(train_cfg.get("sample_num_images", 4) or 4)
    sample_steps = int(train_cfg.get("sample_num_steps", 50) or 50)
    sample_eta = float(train_cfg.get("sample_eta", 0.0) or 0.0)
    sample_cfg_scale = float(train_cfg.get("sample_cfg_scale", 5.0) or 5.0)
    sample_seed = train_cfg.get("sample_seed", None)
    sample_prompts = train_cfg.get("sample_prompts", None)

    # * Training objective: "ddpm_eps" (default) or "flow_matching".
    objective = str(train_cfg.get("objective", "ddpm_eps")).lower()
    fm_cfg_raw = cfg.get("flow_matching", {}) or {}
    fm_cfg = FlowMatchingConfig(
        path=str(fm_cfg_raw.get("path", "linear")),
        t_min=float(fm_cfg_raw.get("t_min", 0.0)),
        t_max=float(fm_cfg_raw.get("t_max", 1.0)),
        time_scale=float(fm_cfg_raw.get("time_scale", float(num_timesteps - 1))),
    )

    sample_vtp = None
    sample_ref_latents = None
    if is_main_process() and sample_every > 0:
        # * Load VTP only for decoding sampled latents.
        name_or_path = str(tok_cfg.get("name_or_path", "")).strip()
        repo_path = tok_cfg.get("repo_path", None)
        if not name_or_path:
            raise ValueError("tokenizer.name_or_path must be set to decode samples.")
        sample_vtp = VTPTokenizer(
            VTPTokenizerConfig(
                name_or_path=name_or_path,
                repo_path=repo_path,
                latent_f=latent_f,
                latent_d=latent_d,
            ),
            device=device,
        )

        # * Prepare sampling captions/prompts.
        if sample_prompts is None:
            sample_ds = LatentsCacheDataset(
                args.train_cache_dir,
                num_frames=num_frames,
                frame_stride=frame_stride,
                cfg_dropout_prob=0.0,
            )
            if bucketing_enabled and sample_ds.num_buckets > 1:
                # * Pick a single bucket deterministically so the reference batch has a consistent H/W.
                sample_batch_sampler = BucketBatchSampler(
                    sample_ds.bucket_to_indices,
                    batch_size=sample_num,
                    shuffle=False,
                    drop_last=False,
                    seed=int(cfg.get("seed", 42)),
                    rank=0,
                    world_size=1,
                )
                sample_loader = DataLoader(
                    sample_ds,
                    batch_sampler=sample_batch_sampler,
                    num_workers=0,
                    collate_fn=collate_latents,
                )
            else:
                sample_loader = DataLoader(
                    sample_ds,
                    batch_size=sample_num,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_latents,
                )
            _lat, _caps = next(iter(sample_loader))
            sample_ref_latents = _lat  # keep on CPU, move to device only when needed
            sample_prompts = _caps
        if not isinstance(sample_prompts, list):
            raise ValueError("train.sample_prompts must be a YAML list of strings, or omit it to sample from dataset.")
        sample_prompts = [str(x) for x in sample_prompts][:sample_num]

    ddp_model.train()
    optim.zero_grad(set_to_none=True)

    accum = 0
    while global_step < max_steps:
        if epoch_sampler is not None and hasattr(epoch_sampler, "set_epoch"):
            epoch_sampler.set_epoch(global_step)
        for latents, captions in loader:
            # latents: (B,F,H,W,D)
            latents = latents.to(device, non_blocking=True).float() * float(latent_scale)
            b = int(latents.shape[0])

            if objective in ("ddpm", "ddpm_eps", "diffusion", "eps"):
                t = torch.randint(0, num_timesteps, (b,), device=device, dtype=torch.long)
                noise = torch.randn_like(latents)
                x_t = add_noise(latents, noise, t, sched)
                target = noise
                t_embed = t
            elif objective in ("flow", "flow_matching", "fm", "rectified_flow"):
                t_cont = sample_t(b, cfg=fm_cfg, device=device, dtype=torch.float32)
                x1 = torch.randn_like(latents)
                x_t, target = make_xt_and_vt(latents, x1, t_cont, cfg=fm_cfg)
                t_embed = time_embedding_value(t_cont, cfg=fm_cfg)
            else:
                raise ValueError(f"Unsupported train.objective: '{objective}'")

            text_tokens, text_mask = text_enc.encode(captions)
            text_tokens = text_tokens.to(device, non_blocking=True)
            text_mask = text_mask.to(device, non_blocking=True)

            sync = (accum + 1) >= grad_accum_steps
            ddp_no_sync = (
                hasattr(ddp_model, "no_sync") and torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            ctx = ddp_model.no_sync() if (ddp_no_sync and not sync) else nullcontext()

            with ctx:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
                    pred = ddp_model(x_t, timesteps=t_embed, text_tokens=text_tokens, text_mask=text_mask)
                    loss = F.mse_loss(pred, target)
                    loss = loss / float(grad_accum_steps)
                loss.backward()

            accum += 1
            if accum >= grad_accum_steps:
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                if ema_model is not None and global_step >= ema_start_step and (global_step % ema_update_every == 0):
                    _update_ema_(ema_model, net, decay=ema_decay)

                if is_main_process() and (log_every > 0) and (global_step % log_every == 0):
                    print(f"[train] step={global_step}/{max_steps} loss={loss.item()*grad_accum_steps:.6f}")

                if is_main_process() and (save_every > 0) and (global_step % save_every == 0):
                    save_checkpoint(
                        ckpt_dir / f"step_{global_step:08d}.pt",
                        model=ddp_model,
                        optimizer=optim,
                        global_step=global_step,
                        extra={"config": cfg},
                    )

                # -----------------------
                # Samples (rank0 only)
                # -----------------------
                if is_main_process() and sample_every > 0 and (global_step % sample_every == 0):
                    assert sample_vtp is not None
                    # * Use the *unwrapped* model to avoid DDP collectives during sampling.
                    # * EMA can lag heavily early (e.g., decay=0.9999). Use it for sampling only after start_step.
                    use_ema_for_sampling = (
                        (ema_model is not None)
                        and bool(ema_use_for_sampling)
                        and (global_step >= int(ema_start_step))
                        and (global_step > 0)
                    )
                    sample_model = ema_model if use_ema_for_sampling else net
                    sample_model.eval()
                    with torch.no_grad():
                        # * Build text tokens.
                        tokens, mask = text_enc.encode(sample_prompts)
                        tokens = tokens.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        if sample_cfg_scale > 1.0:
                            uncond_tokens, uncond_mask = text_enc.encode([""] * len(sample_prompts))
                            uncond_tokens = uncond_tokens.to(device, non_blocking=True)
                            uncond_mask = uncond_mask.to(device, non_blocking=True)
                        else:
                            uncond_tokens, uncond_mask = None, None

                        # * Determine latent shape.
                        # * When bucketing is enabled, follow the denoise-debug reference bucket so sample and
                        # * denoise images share the same base aspect ratio.
                        if bucketing_enabled and sample_ref_latents is not None:
                            h_lat = int(sample_ref_latents.shape[2])
                            w_lat = int(sample_ref_latents.shape[3])
                        else:
                            h_lat = int(height) // int(latent_f)
                            w_lat = int(width) // int(latent_f)
                        shape = (len(sample_prompts), num_frames, h_lat, w_lat, latent_d)

                        # * Always sample in float32 for stability; decoding handles autocast internally.
                        if objective in ("ddpm", "ddpm_eps", "diffusion", "eps"):
                            ddim = DdimConfig(num_steps=sample_steps, eta=sample_eta)
                            dd_steps = make_ddim_timesteps(num_timesteps, sample_steps)
                            latents = ddim_sample_eps(
                                sample_model,
                                shape=shape,
                                timesteps=dd_steps,
                                schedule=sched,
                                text_tokens=tokens,
                                text_mask=mask,
                                uncond_text_tokens=uncond_tokens,
                                uncond_text_mask=uncond_mask,
                                cfg_scale=sample_cfg_scale,
                                ddim=ddim,
                                device=device,
                                dtype=torch.float32,
                                seed=int(sample_seed) if sample_seed is not None else None,
                            )
                        else:
                            ode = FlowOdeConfig(
                                num_steps=sample_steps,
                                solver=str(train_cfg.get("sample_solver", "euler")),
                            )
                            latents = flow_ode_sample(
                                sample_model,
                                shape=shape,
                                flow_cfg=fm_cfg,
                                ode_cfg=ode,
                                text_tokens=tokens,
                                text_mask=mask,
                                uncond_text_tokens=uncond_tokens,
                                uncond_text_mask=uncond_mask,
                                cfg_scale=sample_cfg_scale,
                                device=device,
                                dtype=torch.float32,
                                seed=int(sample_seed) if sample_seed is not None else None,
                            )
                        # * Decode resolution is inferred from the latent grid shape to support bucketing.
                        dec_h = int(latents.shape[2]) * int(latent_f)
                        dec_w = int(latents.shape[3]) * int(latent_f)
                        # * VTP expects latents in the original tokenizer scale.
                        images = sample_vtp.decode(latents / float(latent_scale), width=dec_w, height=dec_h)  # (B,F,3,H,W)
                        sample_images = images
                        _save_sample_grid(
                            exp_dir / "samples" / f"step_{global_step:08d}.png",
                            images=images,
                            captions=sample_prompts,
                            max_frames=8,
                        )

                        # * Optional denoise-debug: does the model reconstruct x0 from a noisy x_t?
                        if bool(train_cfg.get("sample_save_denoise_debug", False)) and sample_ref_latents is not None:
                            x0 = sample_ref_latents.to(device, non_blocking=True).float() * float(latent_scale)
                            b0 = int(x0.shape[0])

                            if objective in ("ddpm", "ddpm_eps", "diffusion", "eps"):
                                t_debug = int(train_cfg.get("sample_denoise_t", 500))
                                t_debug = max(0, min(t_debug, num_timesteps - 1))
                                t0 = torch.full((b0,), t_debug, device=device, dtype=torch.long)
                                g = torch.Generator(device=device)
                                if sample_seed is not None:
                                    g.manual_seed(int(sample_seed))
                                # * Some PyTorch builds do not support `generator` in randn_like().
                                noise0 = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=g)
                                x_t0 = add_noise(x0, noise0, t0, sched)
                                eps0 = sample_model(x_t0, timesteps=t0, text_tokens=tokens[:b0], text_mask=mask[:b0])
                                a = sched.sqrt_alpha_cumprod.to(device=device, dtype=x0.dtype)[t0].view(
                                    b0, *([1] * (x0.ndim - 1))
                                )
                                s = sched.sqrt_one_minus_alpha_cumprod.to(device=device, dtype=x0.dtype)[t0].view(
                                    b0, *([1] * (x0.ndim - 1))
                                )
                                x0_hat = (x_t0 - (s * eps0)) / (a + 1e-8)
                            else:
                                # * Flow matching debug: predict velocity and reconstruct x0_hat.
                                t_debug = float(train_cfg.get("sample_denoise_t", 0.5))
                                t_debug = max(float(fm_cfg.t_min), min(float(t_debug), float(fm_cfg.t_max)))
                                t0 = torch.full((b0,), t_debug, device=device, dtype=torch.float32)
                                g = torch.Generator(device=device)
                                if sample_seed is not None:
                                    g.manual_seed(int(sample_seed))
                                x1 = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype, generator=g)
                                x_t0, v_target = make_xt_and_vt(x0, x1, t0, cfg=fm_cfg)
                                v_pred = sample_model(
                                    x_t0,
                                    timesteps=time_embedding_value(t0, cfg=fm_cfg),
                                    text_tokens=tokens[:b0],
                                    text_mask=mask[:b0],
                                )
                                # * Stable reconstruction from (x_t, v_t) without using x1:
                                # *   x_t = alpha*x0 + sigma*x1
                                # *   v_t = alpha_dot*x0 + sigma_dot*x1
                                # * => x0 = (sigma_dot*x_t - sigma*v_t) / (sigma_dot*alpha - sigma*alpha_dot)
                                # * This stays well-conditioned for cosine paths (denominator is constant pi/2).
                                from mmdit.diffusion.flow_matching import path_coeffs

                                alpha, sigma, a_dot, s_dot = path_coeffs(t0, ndim=x0.ndim, path=fm_cfg.path)
                                denom = (s_dot * alpha) - (sigma * a_dot)
                                x0_hat = ((s_dot * x_t0) - (sigma * v_pred)) / (denom + 1e-8)
                            # * Decode resolution is inferred from the latent grid shape to support bucketing.
                            dec_h = int(x0.shape[2]) * int(latent_f)
                            dec_w = int(x0.shape[3]) * int(latent_f)
                            gt_img = sample_vtp.decode(x0 / float(latent_scale), width=dec_w, height=dec_h)
                            pred_img = sample_vtp.decode(x0_hat / float(latent_scale), width=dec_w, height=dec_h)

                            # * Optional visualization: compare the cached reference reconstruction (GT) against
                            # * the model's full sampling result for the same prompts.
                            if "sample_images" in locals() and sample_images is not None:
                                b_cmp = min(int(gt_img.shape[0]), int(sample_images.shape[0]))
                                if b_cmp > 0:
                                    ref_vs_sample = torch.cat([gt_img[:b_cmp], sample_images[:b_cmp]], dim=1)  # (B,2,3,H,W)
                                    _save_sample_grid(
                                        exp_dir / "samples" / f"ref_vs_sample_step_{global_step:08d}.png",
                                        images=ref_vs_sample,
                                        captions=sample_prompts[:b_cmp],
                                        max_frames=2,
                                    )

                            compare = torch.cat([gt_img, pred_img], dim=1)  # (B, 2F, 3, H, W)
                            _save_sample_grid(
                                exp_dir / "samples" / f"denoise_step_{global_step:08d}.png",
                                images=compare,
                                captions=sample_prompts,
                                max_frames=2,
                            )
                    net.train()

                if global_step >= max_steps:
                    break

        barrier()


if __name__ == "__main__":
    main()


