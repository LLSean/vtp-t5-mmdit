from __future__ import annotations

from pathlib import Path

import torch

from mmdit.data.latents_cache import LatentsCacheDataset, _sample_frame_indices, collate_latents


def test_sample_frame_indices_basic() -> None:
    idx = _sample_frame_indices(total_frames=100, num_frames=16, stride=5)
    assert len(idx) == 16
    assert idx[0] >= 0
    assert idx[-1] < 100
    # * Strided
    assert all((idx[i + 1] - idx[i]) == 5 for i in range(len(idx) - 1))


def test_latents_cache_dataset_shapes(tmp_path: Path) -> None:
    h, w, d = 16, 28, 64
    for i in range(5):
        latents = torch.randn(100, h, w, d, dtype=torch.float16)
        torch.save({"latents": latents, "caption": f"cap {i}"}, tmp_path / f"clip_{i:03d}.pt")

    ds = LatentsCacheDataset(tmp_path, num_frames=16, frame_stride=5, cfg_dropout_prob=0.0)
    sample = ds[0]
    assert sample.latents.shape == (16, h, w, d)
    assert isinstance(sample.caption, str)

    batch_latents, batch_caps = collate_latents([ds[0], ds[1]])
    assert batch_latents.shape == (2, 16, h, w, d)
    assert len(batch_caps) == 2


