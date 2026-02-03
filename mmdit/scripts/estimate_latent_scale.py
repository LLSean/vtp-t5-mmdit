"""Estimates a recommended `tokenizer.latent_scale` for cached VTP latents.

Why:
- VTP reconstruction latents often have std << 1.0 (e.g., ~0.16).
- Diffusion / flow training usually injects noise ~ N(0, 1).
- Scaling the data latents to std≈1 makes the optimization problem better
  conditioned and often improves convergence and sample quality.

This script scans a cache directory produced by:
- `mmdit.scripts.cache_vtp_images` (T2I) or
- `mmdit.scripts.cache_vtp_latents` (T2V)

It supports bucketed caches laid out as:
  cache_dir/<WxH>/<id>.pt
as well as flat caches:
  cache_dir/<id>.pt

Example:
  python3 -m mmdit.scripts.estimate_latent_scale \
    --cache_dir cache/danbooru_surtr_solo_100k_t2i_latents \
    --target_std 1.0 \
    --max_files 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm


@dataclass
class _Moments:
    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        x = x.to(dtype=torch.float32)
        self.n += int(x.numel())
        self.sum += float(x.sum(dtype=torch.float64).item())
        self.sumsq += float((x * x).sum(dtype=torch.float64).item())

    def mean(self) -> float:
        return self.sum / float(self.n) if self.n > 0 else float("nan")

    def std(self) -> float:
        if self.n <= 0:
            return float("nan")
        mean = self.mean()
        var = (self.sumsq / float(self.n)) - (mean * mean)
        return math.sqrt(max(0.0, float(var)))


def _bucket_key(cache_dir: Path, p: Path) -> str:
    rel = p.relative_to(cache_dir)
    if len(rel.parts) > 1:
        return str(rel.parts[0])
    return "__root__"


def _load_latents(path: Path) -> torch.Tensor:
    payload: dict[str, Any] = torch.load(path, map_location="cpu")
    latents = payload.get("latents", None)
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Invalid cache file (missing tensor 'latents'): {path}")
    if latents.ndim != 4:
        raise ValueError(f"Expected latents with shape (F,H,W,D), got {tuple(latents.shape)} in {path}")
    return latents


def _select_files(
    bucket_to_files: dict[str, list[Path]],
    *,
    seed: int,
    samples_per_bucket: int | None,
    max_files: int,
) -> list[Path]:
    rng = random.Random(int(seed))
    chosen: list[Path] = []

    if samples_per_bucket is not None:
        k = int(samples_per_bucket)
        if k <= 0:
            raise ValueError("samples_per_bucket must be > 0 when provided.")
        for files in bucket_to_files.values():
            if not files:
                continue
            if len(files) <= k:
                chosen.extend(files)
            else:
                chosen.extend(rng.sample(files, k))
    else:
        for files in bucket_to_files.values():
            chosen.extend(files)

    if int(max_files) == 0:
        return sorted(chosen)

    m = int(max_files)
    if m < 0:
        raise ValueError("max_files must be >= 0 (0 means all selected).")
    if len(chosen) <= m:
        return sorted(chosen)
    return sorted(rng.sample(chosen, m))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    q = float(q)
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def _print_bucket_counts(bucket_to_files: dict[str, list[Path]]) -> None:
    items = sorted(bucket_to_files.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    total = sum(len(v) for _, v in items)
    print(f"[cache] buckets={len(items)} total_files={total}")
    for k, files in items:
        print(f"  - {k}: {len(files)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a recommended tokenizer.latent_scale from cached latents.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory containing *.pt files.")
    parser.add_argument(
        "--target_std",
        type=float,
        default=1.0,
        help="Target latent std after scaling (default: 1.0). Recommended: 1.0.",
    )
    parser.add_argument(
        "--samples_per_bucket",
        type=int,
        default=None,
        help="If set, randomly samples up to K files per bucket for faster estimation.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum number of files to process after bucketing sampling (0 means all selected).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument(
        "--scale_stat",
        type=str,
        default="global",
        choices=["global", "median_sample"],
        help="Which statistic to use for the recommended scale (default: global).",
    )
    parser.add_argument(
        "--print_yaml",
        action="store_true",
        help="Print a YAML snippet for tokenizer.latent_scale.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")

    all_files = sorted(cache_dir.rglob("*.pt"))
    if not all_files:
        raise FileNotFoundError(f"No .pt files found under: {cache_dir}")

    bucket_to_files: dict[str, list[Path]] = defaultdict(list)
    for p in all_files:
        bucket_to_files[_bucket_key(cache_dir, p)].append(p)

    _print_bucket_counts(bucket_to_files)

    chosen = _select_files(
        bucket_to_files,
        seed=int(args.seed),
        samples_per_bucket=args.samples_per_bucket,
        max_files=int(args.max_files),
    )
    print(f"[select] selected_files={len(chosen)} (samples_per_bucket={args.samples_per_bucket}, max_files={args.max_files})")

    global_m = _Moments()
    bucket_m: dict[str, _Moments] = defaultdict(_Moments)
    sample_stds: list[float] = []

    for p in tqdm(chosen, desc="estimate_latent_scale", unit="file"):
        latents = _load_latents(p)
        global_m.update(latents)
        bucket_m[_bucket_key(cache_dir, p)].update(latents)
        sample_stds.append(float(latents.float().std(unbiased=False).item()))

    g_mean = global_m.mean()
    g_std = global_m.std()
    if not (g_std > 0.0):
        raise ValueError(f"Invalid global std computed: {g_std}")

    print("[global]")
    print(f"  mean={g_mean:.6f}")
    print(f"  std ={g_std:.6f}")

    print("[per-sample std]")
    print(f"  mean={sum(sample_stds)/len(sample_stds):.6f}")
    print(f"  p05 ={_percentile(sample_stds, 5):.6f}")
    print(f"  p50 ={_percentile(sample_stds, 50):.6f}")
    print(f"  p95 ={_percentile(sample_stds, 95):.6f}")
    print(f"  min ={min(sample_stds):.6f}")
    print(f"  max ={max(sample_stds):.6f}")

    print("[per-bucket std]")
    for k in sorted(bucket_m.keys()):
        m = bucket_m[k]
        print(f"  - {k}: std={m.std():.6f} mean={m.mean():.6f} files={len(bucket_to_files.get(k, []))}")

    target_std = float(args.target_std)
    if target_std <= 0:
        raise ValueError("target_std must be > 0.")

    scale_global = target_std / g_std
    scale_median = target_std / _percentile(sample_stds, 50)

    print("[recommended latent_scale]")
    print(f"  from_global_std:  {scale_global:.6f}")
    print(f"  from_median_std:  {scale_median:.6f}")
    if str(args.scale_stat).lower() == "median_sample":
        scale = scale_median
    else:
        scale = scale_global
    print(f"  chosen({args.scale_stat}): {scale:.6f}")

    if bool(args.print_yaml):
        print("\n# Put this under tokenizer: in your YAML config")
        print(f"latent_scale: {scale:.6f}")


if __name__ == "__main__":
    main()


