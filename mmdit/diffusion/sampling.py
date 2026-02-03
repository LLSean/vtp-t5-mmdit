"""Sampling utilities (DDIM).

This module intentionally implements only what we need for "sample during training":
- eps-pred models
- optional classifier-free guidance (CFG)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from mmdit.diffusion.schedule import DiffusionSchedule
from mmdit.diffusion.flow_matching import FlowMatchingConfig, time_embedding_value


@dataclass(frozen=True)
class DdimConfig:
    num_steps: int
    eta: float = 0.0


@dataclass(frozen=True)
class FlowOdeConfig:
    num_steps: int
    solver: str = "euler"  # "euler" | "heun"


def make_ddim_timesteps(num_timesteps: int, num_steps: int) -> torch.Tensor:
    """Creates a monotonic increasing DDIM timestep subset (int64)."""

    t = int(num_timesteps)
    n = int(num_steps)
    if n <= 1:
        raise ValueError("num_steps must be > 1")
    if t <= 1:
        raise ValueError("num_timesteps must be > 1")
    # * Linearly spaced indices in [0, T-1].
    steps = torch.linspace(0, t - 1, n, dtype=torch.float64)
    return steps.round().to(torch.long).unique(sorted=True)


def _extract(x: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Extracts per-batch scalars and reshapes to broadcast over x."""

    b = int(t.shape[0])
    out = x.to(device=t.device)[t].view(b, *([1] * (ndim - 1)))
    return out


@torch.no_grad()
def ddim_sample_eps(
    model,
    *,
    shape: tuple[int, ...],
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    text_tokens: torch.Tensor,
    text_mask: Optional[torch.Tensor],
    uncond_text_tokens: Optional[torch.Tensor] = None,
    uncond_text_mask: Optional[torch.Tensor] = None,
    cfg_scale: float,
    ddim: DdimConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """DDIM sampling for eps-pred diffusion models.

    Args:
        model: Callable like model(x, timesteps=t, text_tokens=..., text_mask=...) -> eps
        shape: Output latent shape (B, F, H, W, D).
        timesteps: A 1D int64 tensor of DDIM steps (monotonic increasing).
        schedule: Precomputed diffusion schedule.
        text_tokens: (B, T, D_text) tokens.
        text_mask: Optional (B, T) bool.
        cfg_scale: CFG guidance scale (>=1 enables guidance).
        ddim: DDIM config (num_steps, eta).
        device: Sampling device.
        dtype: Sampling dtype (float32 recommended).
        seed: Optional RNG seed for deterministic sampling.

    Returns:
        Sampled latents with shape `shape`.
    """

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    else:
        g = None

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be 1D.")
    if timesteps.dtype != torch.long:
        timesteps = timesteps.to(torch.long)

    b = int(shape[0])
    if int(text_tokens.shape[0]) != b:
        raise ValueError("text_tokens batch size mismatch.")
    if text_mask is not None and int(text_mask.shape[0]) != b:
        raise ValueError("text_mask batch size mismatch.")

    x = torch.randn(shape, device=device, dtype=dtype, generator=g)
    alpha_bar = schedule.alpha_cumprod.to(device=device, dtype=dtype)

    # * Iterate from large t to small t.
    steps = timesteps.to(device=device)
    for i in reversed(range(int(steps.shape[0]))):
        t = steps[i].expand(b)
        a_t = _extract(alpha_bar, t, x.ndim)  # (B,1,1,1,1)
        sqrt_a_t = torch.sqrt(a_t)
        sqrt_one_minus_a_t = torch.sqrt(1.0 - a_t)

        # * eps prediction (optionally with CFG).
        if float(cfg_scale) <= 1.0:
            eps = model(x, timesteps=t, text_tokens=text_tokens, text_mask=text_mask)
        else:
            if uncond_text_tokens is None:
                raise ValueError(
                    "CFG sampling requires `uncond_text_tokens` (i.e., encode empty prompts with the SAME text encoder). "
                    "Either pass uncond_text_tokens/uncond_text_mask or set cfg_scale <= 1."
                )
            if uncond_text_tokens.shape != text_tokens.shape:
                raise ValueError("uncond_text_tokens must have the same shape as text_tokens.")
            if text_mask is None:
                if uncond_text_mask is not None:
                    raise ValueError("uncond_text_mask must be None when text_mask is None.")
            else:
                if uncond_text_mask is None:
                    raise ValueError("uncond_text_mask must be provided when text_mask is provided.")
                if uncond_text_mask.shape != text_mask.shape:
                    raise ValueError("uncond_text_mask must have the same shape as text_mask.")

            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            tok_in = torch.cat([uncond_text_tokens, text_tokens], dim=0)
            if text_mask is None:
                mask_in = None
            else:
                mask_in = torch.cat([uncond_text_mask, text_mask], dim=0)
            eps_all = model(x_in, timesteps=t_in, text_tokens=tok_in, text_mask=mask_in)
            eps_uncond, eps_cond = eps_all.chunk(2, dim=0)
            eps = eps_uncond + float(cfg_scale) * (eps_cond - eps_uncond)

        x0 = (x - (sqrt_one_minus_a_t * eps)) / (sqrt_a_t + 1e-8)

        if i == 0:
            x = x0
            break

        t_prev = steps[i - 1].expand(b)
        a_prev = _extract(alpha_bar, t_prev, x.ndim)

        # * DDIM update:
        #   x_{t_prev} = sqrt(a_prev)*x0 + sqrt(1-a_prev - sigma^2)*eps + sigma*z
        eta = float(ddim.eta)
        if eta < 0:
            raise ValueError("eta must be >= 0")
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - (a_t / (a_prev + 1e-8)))
        if eta > 0:
            # * Some PyTorch builds do not support `generator` in randn_like().
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
        else:
            noise = torch.zeros_like(x)
        dir_term = torch.sqrt(torch.clamp(1 - a_prev - sigma**2, min=0.0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_term + sigma * noise

    return x


@torch.no_grad()
def flow_ode_sample(
    model,
    *,
    shape: tuple[int, ...],
    flow_cfg: FlowMatchingConfig,
    ode_cfg: FlowOdeConfig,
    text_tokens: torch.Tensor,
    text_mask: Optional[torch.Tensor],
    uncond_text_tokens: Optional[torch.Tensor],
    uncond_text_mask: Optional[torch.Tensor],
    cfg_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """ODE sampling for flow-matching models (predicting velocity dx/dt).

    Integrates from t=1 (noise) -> t=0 (data) using Euler or Heun.
    """

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
    else:
        g = None

    b = int(shape[0])
    if int(text_tokens.shape[0]) != b:
        raise ValueError("text_tokens batch size mismatch.")
    if text_mask is not None and int(text_mask.shape[0]) != b:
        raise ValueError("text_mask batch size mismatch.")

    n = int(ode_cfg.num_steps)
    if n <= 1:
        raise ValueError("ode_cfg.num_steps must be > 1")

    solver = str(ode_cfg.solver).lower()
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported flow ODE solver: '{ode_cfg.solver}'")

    x = torch.randn(shape, device=device, dtype=dtype, generator=g)
    t_grid = torch.linspace(1.0, 0.0, n, device=device, dtype=dtype)

    def _model_v(x_in: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        t_vec = t_scalar.expand(b).to(device=device, dtype=torch.float32)
        t_embed = time_embedding_value(t_vec, cfg=flow_cfg)
        if float(cfg_scale) <= 1.0:
            return model(x_in, timesteps=t_embed, text_tokens=text_tokens, text_mask=text_mask)

        if uncond_text_tokens is None:
            raise ValueError("CFG sampling requires uncond_text_tokens.")
        x_cat = torch.cat([x_in, x_in], dim=0)
        t_cat = torch.cat([t_embed, t_embed], dim=0)
        tok_cat = torch.cat([uncond_text_tokens, text_tokens], dim=0)
        if text_mask is None:
            mask_cat = None
        else:
            if uncond_text_mask is None:
                raise ValueError("CFG sampling requires uncond_text_mask when text_mask is provided.")
            mask_cat = torch.cat([uncond_text_mask, text_mask], dim=0)
        v_all = model(x_cat, timesteps=t_cat, text_tokens=tok_cat, text_mask=mask_cat)
        v_uncond, v_cond = v_all.chunk(2, dim=0)
        return v_uncond + float(cfg_scale) * (v_cond - v_uncond)

    for i in range(n - 1):
        t_i = t_grid[i]
        t_next = t_grid[i + 1]
        dt = (t_i - t_next).to(dtype=dtype)  # positive scalar
        v_i = _model_v(x, t_i)

        if solver == "euler":
            x = x - v_i * dt
        else:
            # Heun: predictor-corrector
            x_euler = x - v_i * dt
            v_next = _model_v(x_euler, t_next)
            x = x - 0.5 * (v_i + v_next) * dt

    return x

