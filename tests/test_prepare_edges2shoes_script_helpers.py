from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mmdit.scripts.prepare_edges2shoes_t2i import _decode_image_struct, _iter_parquet_images


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (8, 8), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_image_struct_bytes() -> None:
    img = _decode_image_struct({"bytes": _png_bytes((255, 0, 0)), "path": None})
    assert img.size == (8, 8)


def test_iter_parquet_images_reads_structs(tmp_path: Path) -> None:
    struct_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    arr = pa.array(
        [
            {"bytes": _png_bytes((0, 255, 0)), "path": None},
            {"bytes": _png_bytes((0, 0, 255)), "path": None},
        ],
        type=struct_type,
    )
    table = pa.table({"imageB": arr})
    p = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(table, p)

    rows = list(_iter_parquet_images(p, column="imageB", batch_size=1))
    assert len(rows) == 2
    assert isinstance(rows[0], dict)
    assert "bytes" in rows[0]


