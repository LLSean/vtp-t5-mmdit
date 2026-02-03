from __future__ import annotations

import torch

from mmdit.models.video_mmdit import VideoMMDiT, VideoMMDiTConfig


def test_video_mmdit_forward_with_tread_enabled_cpu() -> None:
    b, f, h, w = 2, 1, 16, 16
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
    model.train()
    model.configure_tread({"enabled": True, "selection_rate": 0.5, "apply_to": ["cross", "mlp"], "mode": "rotate"})

    x = torch.randn(b, f, h, w, d_lat)
    t = torch.randint(0, 1000, (b,), dtype=torch.long)
    text = torch.randn(b, 8, d_txt)
    mask = torch.ones(b, 8, dtype=torch.bool)

    out = model(x, timesteps=t, text_tokens=text, text_mask=mask)
    assert out.shape == x.shape


def test_video_mmdit_backward_with_tread_enabled_cpu() -> None:
    b, f, h, w = 2, 1, 16, 16
    d_lat = 32
    d_txt = 16

    model = VideoMMDiT(
        VideoMMDiTConfig(
            latent_d=d_lat,
            model_dim=32,
            depth=2,
            num_heads=4,
            window_size=8,
            mlp_ratio=2.0,
            dropout=0.0,
            text_dim=d_txt,
        )
    )
    model.train()
    model.configure_tread({"enabled": True, "selection_rate": 0.5, "apply_to": ["cross", "mlp"], "mode": "rotate"})

    x = torch.randn(b, f, h, w, d_lat, requires_grad=True)
    t = torch.randint(0, 1000, (b,), dtype=torch.long)
    text = torch.randn(b, 8, d_txt)
    mask = torch.ones(b, 8, dtype=torch.bool)

    out = model(x, timesteps=t, text_tokens=text, text_mask=mask)
    loss = out.square().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    for g in grads:
        if g is None:
            continue
        assert torch.isfinite(g).all().item()

