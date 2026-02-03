from __future__ import annotations

import torch

from mmdit.models.video_mmdit import VideoMMDiT, VideoMMDiTConfig


def test_video_mmdit_forward_shape_cpu() -> None:
    b, f, h, w = 2, 16, 16, 28
    d_lat = 64
    d_txt = 32
    model = VideoMMDiT(
        VideoMMDiTConfig(
            latent_d=d_lat,
            model_dim=64,
            depth=2,
            num_heads=4,
            window_size=8,
            mlp_ratio=2.0,
            dropout=0.0,
            text_dim=d_txt,
        )
    )

    x = torch.randn(b, f, h, w, d_lat)
    t = torch.randint(0, 1000, (b,), dtype=torch.long)
    text = torch.randn(b, 8, d_txt)
    mask = torch.ones(b, 8, dtype=torch.bool)
    out = model(x, timesteps=t, text_tokens=text, text_mask=mask)
    assert out.shape == x.shape


def test_video_mmdit_forward_accepts_bfloat16_text_tokens_cpu() -> None:
    b, f, h, w = 2, 4, 8, 8
    d_lat = 32
    d_txt = 16
    model = VideoMMDiT(
        VideoMMDiTConfig(
            latent_d=d_lat,
            model_dim=32,
            depth=1,
            num_heads=4,
            window_size=4,
            mlp_ratio=2.0,
            dropout=0.0,
            text_dim=d_txt,
        )
    )

    x = torch.randn(b, f, h, w, d_lat, dtype=torch.float32)
    t = torch.randint(0, 1000, (b,), dtype=torch.long)
    # * Regression: allow text tokens produced under BF16 autocast to be fed into a FP32 model.
    text = torch.randn(b, 8, d_txt, dtype=torch.bfloat16)
    mask = torch.ones(b, 8, dtype=torch.bool)
    out = model(x, timesteps=t, text_tokens=text, text_mask=mask)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

