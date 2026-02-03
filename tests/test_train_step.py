from __future__ import annotations

import torch
from torch.nn import functional as F

from mmdit.diffusion.schedule import add_noise, make_schedule
from mmdit.models.video_mmdit import VideoMMDiT, VideoMMDiTConfig


def test_single_train_step_cpu() -> None:
    b, f, h, w = 2, 8, 8, 8
    d_lat = 32
    d_txt = 16

    model = VideoMMDiT(
        VideoMMDiTConfig(
            latent_d=d_lat,
            model_dim=64,
            depth=2,
            num_heads=4,
            window_size=4,
            mlp_ratio=2.0,
            dropout=0.0,
            text_dim=d_txt,
        )
    )
    sched = make_schedule(100, schedule="cosine")

    x0 = torch.randn(b, f, h, w, d_lat)
    t = torch.randint(0, 100, (b,), dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, noise, t, sched)

    text = torch.randn(b, 4, d_txt)
    mask = torch.ones(b, 4, dtype=torch.bool)

    pred = model(x_t, timesteps=t, text_tokens=text, text_mask=mask)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    # * Avoid torch.optim in this test because some environments may have a local
    # * `transformers/` folder shadowing HuggingFace transformers, which can
    # * break torch._dynamo imports during optimizer construction.
    lr = 1e-3
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            p.add_(p.grad, alpha=-lr)
        model.zero_grad(set_to_none=True)

    assert torch.isfinite(loss).item()


