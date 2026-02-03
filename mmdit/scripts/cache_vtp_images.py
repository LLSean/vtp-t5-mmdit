"""Caches VTP reconstruction latents for an image+caption JSONL dataset.

Run:
  python3 -m mmdit.scripts.cache_vtp_images --config configs/...yaml --meta_jsonl ... --out_dir ...

JSONL line format:
  {"id":"img_000001","image_path":".../000001.png","caption":"a cat"}
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

from PIL import Image
import torch
from tqdm import tqdm

from mmdit.config import ensure_not_placeholder, load_yaml_config
from mmdit.data.bucketing import bucket_dir_name, parse_bucket_config, select_bucket
from mmdit.data.frames import imagenet_normalize, pil_to_tensor, resize_and_crop, resize_and_center_crop
from mmdit.data.jsonl import read_image_metadata_jsonl
from mmdit.tokenizers.vtp import VTPTokenizer, VTPTokenizerConfig
from mmdit.utils import select_device


def _stable_int_seed(*parts: str) -> int:
    """Returns a stable 32-bit seed from string parts."""

    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache VTP latents for T2I training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="train", help="Split name (informational).")
    parser.add_argument("--meta_jsonl", type=str, required=True, help="Path to image metadata JSONL.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for cached .pt files.")
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
    tokenizer = VTPTokenizer(
        VTPTokenizerConfig(
            name_or_path=name_or_path,
            latent_f=int(tok_cfg.get("latent_f", 16)),
            latent_d=int(tok_cfg.get("latent_d", 64)),
            repo_path=tok_cfg.get("repo_path", None),
        ),
        device=device,
    )
    latent_f = int(tok_cfg.get("latent_f", 16))

    bucketing_enabled, buckets, bucket_strategy = parse_bucket_config(
        cfg,
        default_width=width,
        default_height=height,
        latent_f=latent_f,
    )

    pre_cfg = cfg.get("preprocess", {}) or {}
    crop_mode = str(pre_cfg.get("crop", "center")).strip().lower()
    shift_fraction = float(pre_cfg.get("shift_fraction", 1.0))
    num_crops = int(pre_cfg.get("num_crops", 1))
    crop_seed = int(pre_cfg.get("seed", cfg.get("seed", 0) or 0))
    if num_crops <= 0:
        raise ValueError("preprocess.num_crops must be > 0.")

    samples = read_image_metadata_jsonl(args.meta_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # * Avoid silent duplication when switching bucketing configs.
    # * If a sample_id already exists anywhere under out_dir, skip caching it.
    existing_ids = {p.stem for p in out_dir.rglob("*.pt")}

    for s in tqdm(samples, desc=f"cache_img[{args.split}]", unit="img"):
        img = Image.open(s.image_path)
        if bucketing_enabled:
            b = select_bucket(
                input_width=int(img.size[0]),
                input_height=int(img.size[1]),
                buckets=buckets,
                strategy=bucket_strategy,
            )
            enc_w, enc_h = int(b.width), int(b.height)
            bucket_subdir = bucket_dir_name(b)
        else:
            enc_w, enc_h = width, height
            bucket_subdir = ""

        for crop_idx in range(num_crops):
            if num_crops == 1:
                sid = s.sample_id
            else:
                sid = f"{s.sample_id}__c{crop_idx:02d}"
            if sid in existing_ids:
                continue

            if bucket_subdir:
                out_path = out_dir / bucket_subdir / f"{sid}.pt"
            else:
                out_path = out_dir / f"{sid}.pt"

            if crop_mode == "center":
                img_crop = resize_and_center_crop(img, width=enc_w, height=enc_h)
            else:
                seed = crop_seed ^ _stable_int_seed(str(s.sample_id), str(crop_idx))
                rng = random.Random(int(seed))
                img_crop = resize_and_crop(
                    img,
                    width=enc_w,
                    height=enc_h,
                    crop="shifted" if crop_mode == "shifted" else crop_mode,
                    rng=rng,
                    shift_fraction=shift_fraction,
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            x = pil_to_tensor(img_crop)  # [0, 1]
            x = imagenet_normalize(x)  # VTP expects ImageNet normalization
            frames = x.unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

            latents = tokenizer.encode(frames, width=enc_w, height=enc_h)  # (1,1,H_lat,W_lat,D_lat)
            payload = {
                "latents": latents[0].detach().cpu().to(torch.float16),  # (1,H_lat,W_lat,D_lat)
                "caption": s.caption,
                "width": int(enc_w),
                "height": int(enc_h),
            }
            torch.save(payload, out_path)
            existing_ids.add(sid)


if __name__ == "__main__":
    main()


