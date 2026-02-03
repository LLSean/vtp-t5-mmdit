from __future__ import annotations

import random

from PIL import Image
import numpy as np

from mmdit.data.frames import resize_and_crop


def _make_xy_image(w: int, h: int) -> Image.Image:
    # * Encodes x/y position in RGB so different crops produce different pixels.
    xs = np.tile(np.arange(w, dtype=np.uint16)[None, :], (h, 1))
    ys = np.tile(np.arange(h, dtype=np.uint16)[:, None], (1, w))
    r = (xs % 256).astype(np.uint8)
    g = (ys % 256).astype(np.uint8)
    b = ((xs + ys) % 256).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def test_resize_and_crop_center_output_size() -> None:
    img = _make_xy_image(640, 360)
    out = resize_and_crop(img, width=256, height=256, crop="center")
    assert out.size == (256, 256)


def test_resize_and_crop_shifted_is_deterministic_with_seed() -> None:
    img = _make_xy_image(640, 360)
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    out1 = resize_and_crop(img, width=256, height=256, crop="shifted", rng=rng1, shift_fraction=1.0)
    out2 = resize_and_crop(img, width=256, height=256, crop="shifted", rng=rng2, shift_fraction=1.0)
    assert out1.tobytes() == out2.tobytes()


def test_resize_and_crop_shifted_changes_with_different_seed() -> None:
    img = _make_xy_image(640, 360)
    out1 = resize_and_crop(img, width=256, height=256, crop="shifted", rng=random.Random(1), shift_fraction=1.0)
    out2 = resize_and_crop(img, width=256, height=256, crop="shifted", rng=random.Random(2), shift_fraction=1.0)
    assert out1.tobytes() != out2.tobytes()

