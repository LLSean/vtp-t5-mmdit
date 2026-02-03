"""Extracts a small subset of edges2shoes parquet data into images + train_images.jsonl.

This script is intended for **smoke testing** the T2I pipeline before you have
your own (anime) dataset ready.

Dataset expected (local HF parquet layout):
  ~/pix2pix/pix2pix/edges2shoes/
    dataset_infos.json
    data/train-*.parquet

Each parquet row contains:
  - imageA: struct<bytes: binary, path: string>  (edges)
  - imageB: struct<bytes: binary, path: string>  (shoes)

Run:
  python3 -m mmdit.scripts.prepare_edges2shoes_t2i --max_samples 512
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Optional

from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm


def _decode_image_struct(value: dict) -> Image.Image:
    """Decodes a HF-style Image struct {'bytes': ..., 'path': ...} to PIL.Image."""

    raw = value.get("bytes")
    path = value.get("path")
    if raw is not None:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if path:
        return Image.open(path).convert("RGB")
    raise ValueError("Image struct has neither 'bytes' nor 'path'.")


def _iter_parquet_images(parquet_path: Path, *, column: str, batch_size: int):
    pf = pq.ParquetFile(str(parquet_path))
    for rb in pf.iter_batches(batch_size=int(batch_size), columns=[str(column)]):
        arr = rb.column(0)
        for i in range(rb.num_rows):
            v = arr[i].as_py()
            if not isinstance(v, dict):
                raise TypeError(f"Expected dict struct for image column '{column}', got {type(v).__name__}")
            yield v


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small T2I dataset from edges2shoes parquet.")
    parser.add_argument(
        "--edges2shoes_root",
        type=str,
        default=str(Path("~/pix2pix/pix2pix/edges2shoes").expanduser()),
        help="edges2shoes dataset root containing dataset_infos.json and data/*.parquet",
    )
    parser.add_argument("--split", type=str, default="train", help="Split name (train/val/test).")
    parser.add_argument(
        "--column",
        type=str,
        default="imageB",
        choices=["imageA", "imageB"],
        help="Which image column to export. imageB is the photo target; imageA is the edge sketch.",
    )
    parser.add_argument("--max_samples", type=int, default=512, help="Number of samples to extract.")
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Optional caption to use for all samples. If omitted, uses a simple default based on the column.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="data/edges2shoes_t2i",
        help="Output root. Will create images/ and train_images.jsonl under it.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Parquet batch size (decode streaming).")
    args = parser.parse_args()

    root = Path(args.edges2shoes_root).expanduser().resolve()
    data_dir = root / "data"
    if not (root / "dataset_infos.json").is_file():
        raise FileNotFoundError(f"dataset_infos.json not found under: {root}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data/ dir not found under: {root}")

    parquet_files = sorted(data_dir.glob(f"{args.split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found: {data_dir}/{args.split}-*.parquet")

    out_root = Path(args.out_root).expanduser().resolve()
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "train_images.jsonl"

    cap = args.caption
    if cap is None:
        cap = "a shoe" if args.column == "imageB" else "a sketch of shoes"

    max_samples = int(args.max_samples)
    if max_samples <= 0:
        raise ValueError("--max_samples must be > 0")

    written = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for pq_path in tqdm(parquet_files, desc="parquet", unit="file"):
            for v in _iter_parquet_images(pq_path, column=args.column, batch_size=int(args.batch_size)):
                img = _decode_image_struct(v)
                sample_id = f"edges2shoes_{args.split}_{written:06d}"
                out_img = images_dir / f"{sample_id}.png"
                img.save(out_img)

                rec = {"id": sample_id, "image_path": str(out_img), "caption": cap}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

                if written >= max_samples:
                    break
            if written >= max_samples:
                break

    print(f"[done] wrote {written} samples")
    print(f"[images] {images_dir}")
    print(f"[jsonl]  {jsonl_path}")


if __name__ == "__main__":
    main()


