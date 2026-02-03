from __future__ import annotations

from pathlib import Path

import pytest

from mmdit.data.jsonl import read_image_metadata_jsonl, read_metadata_jsonl


def test_read_video_metadata_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "video.jsonl"
    p.write_text('{"id":"a","frames_dir":"x/y","caption":"cap"}\n', encoding="utf-8")
    out = read_metadata_jsonl(p)
    assert len(out) == 1
    assert out[0].sample_id == "a"
    assert out[0].caption == "cap"


def test_read_image_metadata_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "img.jsonl"
    p.write_text('{"id":"a","image_path":"x/y.png","caption":"cap"}\n', encoding="utf-8")
    out = read_image_metadata_jsonl(p)
    assert len(out) == 1
    assert out[0].sample_id == "a"
    assert out[0].caption == "cap"


def test_read_image_metadata_jsonl_missing_field(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"a","caption":"cap"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        _ = read_image_metadata_jsonl(p)


