from __future__ import annotations

import pytest
import torch

from mmdit.data.latents_cache import BucketBatchSampler, LatentsCacheDataset


def _write_cache_pt(path: torch.Tensor, out_path) -> None:
    payload = {"latents": path, "caption": "x"}
    torch.save(payload, out_path)


def test_latents_cache_dataset_recurses_and_reports_buckets(tmp_path) -> None:
    # bucket A: 256x256 -> (H_lat,W_lat)=(16,16) with latent_f=16
    d1 = tmp_path / "256x256"
    d1.mkdir(parents=True, exist_ok=True)
    _write_cache_pt(torch.zeros((1, 16, 16, 64), dtype=torch.float16), d1 / "a.pt")

    # bucket B: 320x256 -> (H_lat,W_lat)=(16,20)
    d2 = tmp_path / "320x256"
    d2.mkdir(parents=True, exist_ok=True)
    _write_cache_pt(torch.zeros((1, 16, 20, 64), dtype=torch.float16), d2 / "b.pt")

    ds = LatentsCacheDataset(tmp_path, num_frames=1, frame_stride=1, cfg_dropout_prob=0.0)
    assert len(ds) == 2
    assert ds.num_buckets == 2
    assert set(ds.bucket_to_indices.keys()) == {"256x256", "320x256"}


def test_latents_cache_dataset_rejects_duplicate_ids_across_buckets(tmp_path) -> None:
    d1 = tmp_path / "256x256"
    d1.mkdir(parents=True, exist_ok=True)
    _write_cache_pt(torch.zeros((1, 16, 16, 64), dtype=torch.float16), d1 / "dup.pt")

    d2 = tmp_path / "320x256"
    d2.mkdir(parents=True, exist_ok=True)
    _write_cache_pt(torch.zeros((1, 16, 20, 64), dtype=torch.float16), d2 / "dup.pt")

    with pytest.raises(ValueError, match=r"Duplicate sample IDs"):
        _ = LatentsCacheDataset(tmp_path, num_frames=1, frame_stride=1, cfg_dropout_prob=0.0)


def test_bucket_batch_sampler_groups_indices_by_bucket(tmp_path) -> None:
    d1 = tmp_path / "256x256"
    d1.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        _write_cache_pt(torch.zeros((1, 16, 16, 64), dtype=torch.float16), d1 / f"a_{i}.pt")

    d2 = tmp_path / "320x256"
    d2.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        _write_cache_pt(torch.zeros((1, 16, 20, 64), dtype=torch.float16), d2 / f"b_{i}.pt")

    ds = LatentsCacheDataset(tmp_path, num_frames=1, frame_stride=1, cfg_dropout_prob=0.0)
    sampler = BucketBatchSampler(
        ds.bucket_to_indices,
        batch_size=2,
        shuffle=True,
        drop_last=False,
        seed=123,
        rank=0,
        world_size=1,
    )
    sampler.set_epoch(0)
    for batch in sampler:
        keys = {ds.bucket_key(i) for i in batch}
        assert len(keys) == 1


def test_bucket_batch_sampler_distributed_split_is_disjoint(tmp_path) -> None:
    d1 = tmp_path / "256x256"
    d1.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        _write_cache_pt(torch.zeros((1, 16, 16, 64), dtype=torch.float16), d1 / f"a_{i}.pt")

    d2 = tmp_path / "320x256"
    d2.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        _write_cache_pt(torch.zeros((1, 16, 20, 64), dtype=torch.float16), d2 / f"b_{i}.pt")

    ds = LatentsCacheDataset(tmp_path, num_frames=1, frame_stride=1, cfg_dropout_prob=0.0)
    s0 = BucketBatchSampler(
        ds.bucket_to_indices,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        seed=999,
        rank=0,
        world_size=2,
    )
    s1 = BucketBatchSampler(
        ds.bucket_to_indices,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        seed=999,
        rank=1,
        world_size=2,
    )
    s0.set_epoch(7)
    s1.set_epoch(7)

    batches0 = list(s0)
    batches1 = list(s1)
    assert len(batches0) == len(batches1) == len(s0) == len(s1)

    flat0 = [i for b in batches0 for i in b]
    flat1 = [i for b in batches1 for i in b]
    assert set(flat0).isdisjoint(set(flat1))


