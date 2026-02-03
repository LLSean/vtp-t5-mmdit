from __future__ import annotations

import torch
from torch import nn

from mmdit.diffusion.sampling import DdimConfig, ddim_sample_eps, make_ddim_timesteps
from mmdit.diffusion.sampling import FlowOdeConfig, flow_ode_sample
from mmdit.diffusion.schedule import make_schedule
from mmdit.diffusion.flow_matching import FlowMatchingConfig
from mmdit.models.video_mmdit import VideoMMDiT, VideoMMDiTConfig


class _ZeroEps(nn.Module):
    def forward(self, x: torch.Tensor, *, timesteps: torch.Tensor, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros_like(x)


def test_ddim_sample_eps_shape_and_finite_cpu() -> None:
    sched = make_schedule(20, schedule="cosine")
    timesteps = make_ddim_timesteps(20, 6)

    model = _ZeroEps()
    b, f, h, w, d = 2, 1, 4, 4, 8
    text = torch.randn(b, 4, 16)
    mask = torch.ones(b, 4, dtype=torch.bool)
    out = ddim_sample_eps(
        model,
        shape=(b, f, h, w, d),
        timesteps=timesteps,
        schedule=sched,
        text_tokens=text,
        text_mask=mask,
        uncond_text_tokens=None,
        uncond_text_mask=None,
        cfg_scale=1.0,
        ddim=DdimConfig(num_steps=6, eta=0.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=123,
    )
    assert out.shape == (b, f, h, w, d)
    assert torch.isfinite(out).all().item()


def test_flow_ode_sample_shape_and_finite_cpu() -> None:
    cfg = FlowMatchingConfig(path="linear", t_min=0.0, t_max=1.0, time_scale=999.0)
    ode = FlowOdeConfig(num_steps=6, solver="euler")
    model = _ZeroEps()
    b, f, h, w, d = 2, 1, 4, 4, 8
    text = torch.randn(b, 4, 16)
    mask = torch.ones(b, 4, dtype=torch.bool)
    out = flow_ode_sample(
        model,
        shape=(b, f, h, w, d),
        flow_cfg=cfg,
        ode_cfg=ode,
        text_tokens=text,
        text_mask=mask,
        uncond_text_tokens=None,
        uncond_text_mask=None,
        cfg_scale=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=123,
    )
    assert out.shape == (b, f, h, w, d)
    assert torch.isfinite(out).all().item()


def test_flow_ode_sample_accepts_bfloat16_text_tokens_cpu() -> None:
    cfg = FlowMatchingConfig(path="linear", t_min=0.0, t_max=1.0, time_scale=999.0)
    ode = FlowOdeConfig(num_steps=4, solver="euler")

    b, f, h, w = 2, 1, 4, 4
    d_lat = 8
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

    # * Regression: sampling runs in float32, but HF encoders may emit BF16 tokens.
    text = torch.randn(b, 4, d_txt, dtype=torch.bfloat16)
    mask = torch.ones(b, 4, dtype=torch.bool)
    uncond_text = torch.randn(b, 4, d_txt, dtype=torch.bfloat16)
    uncond_mask = torch.ones(b, 4, dtype=torch.bool)

    out = flow_ode_sample(
        model,
        shape=(b, f, h, w, d_lat),
        flow_cfg=cfg,
        ode_cfg=ode,
        text_tokens=text,
        text_mask=mask,
        uncond_text_tokens=uncond_text,
        uncond_text_mask=uncond_mask,
        cfg_scale=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=123,
    )
    assert out.shape == (b, f, h, w, d_lat)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all().item()


