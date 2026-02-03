"""Caches VTP reconstruction latents for a frames-on-disk dataset.

Run:
  python -m mmdit.scripts.cache_vtp_latents --config configs/...yaml --meta_jsonl ... --out_dir ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
from tqdm import tqdm

from mmdit.config import ensure_not_placeholder, load_yaml_config
from mmdit.data.bucketing import bucket_dir_name, parse_bucket_config, select_bucket
from mmdit.data.frames import list_frame_files, load_frames_tensor
from mmdit.data.jsonl import read_metadata_jsonl
from mmdit.tokenizers.vtp import VTPTokenizer, VTPTokenizerConfig
from mmdit.utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache VTP latents for T2V training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="train", help="Split name (informational).")
    parser.add_argument("--meta_jsonl", type=str, required=True, help="Path to metadata JSONL.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for cached .pt files.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optionally cap frames per clip (e.g., 80).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    video_cfg = cfg.get("video", {}) or {}
    tok_cfg = cfg.get("tokenizer", {}) or {}

    width = int(video_cfg["width"])
    height = int(video_cfg["height"])

    name_or_path = str(tok_cfg.get("name_or_path", ""))
    ensure_not_placeholder(name_or_path, "tokenizer.name_or_path")

    device = select_device(args.device)
    latent_f = int(tok_cfg.get("latent_f", 16))
    tokenizer = VTPTokenizer(
        VTPTokenizerConfig(
            name_or_path=name_or_path,
            latent_f=latent_f,
            latent_d=int(tok_cfg.get("latent_d", 64)),
            repo_path=tok_cfg.get("repo_path", None),
        ),
        device=device,
    )

    bucketing_enabled, buckets, bucket_strategy = parse_bucket_config(
        cfg,
        default_width=width,
        default_height=height,
        latent_f=latent_f,
    )

    samples = read_metadata_jsonl(args.meta_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = {p.stem for p in out_dir.rglob("*.pt")}

    for s in tqdm(samples, desc=f"cache[{args.split}]", unit="clip"):
        if s.sample_id in existing_ids:
            continue

        frame_files = list_frame_files(s.frames_dir)
        if args.max_frames is not None:
            frame_files = frame_files[: int(args.max_frames)]
        frame_indices = list(range(len(frame_files)))

        enc_w, enc_h = width, height
        out_path = out_dir / f"{s.sample_id}.pt"
        if bucketing_enabled:
            # * Use the first frame to infer the source aspect ratio.
            first = frame_files[0]
            img0 = Image.open(first)
            b = select_bucket(
                input_width=int(img0.size[0]),
                input_height=int(img0.size[1]),
                buckets=buckets,
                strategy=bucket_strategy,
            )
            enc_w, enc_h = int(b.width), int(b.height)
            out_path = out_dir / bucket_dir_name(b) / f"{s.sample_id}.pt"

        frames = load_frames_tensor(
            s.frames_dir,
            frame_indices=frame_indices,
            width=enc_w,
            height=enc_h,
            normalize="imagenet",
        )  # (F,3,H,W)
        frames = frames.unsqueeze(0)  # (1,F,3,H,W)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        latents = tokenizer.encode(frames, width=enc_w, height=enc_h)  # (1,F,H_lat,W_lat,D_lat)
        payload = {
            "latents": latents[0].detach().cpu().to(torch.float16),
            "caption": s.caption,
            "width": int(enc_w),
            "height": int(enc_h),
        }
        torch.save(payload, out_path)
        existing_ids.add(s.sample_id)


if __name__ == "__main__":
    main()


