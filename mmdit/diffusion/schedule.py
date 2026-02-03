"""DDPM-style discrete diffusion schedule utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def cosine_beta_schedule(num_timesteps: int, *, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal).

    Returns:
        betas: (T,) float64 tensor in (0, 1).
    """

    t = int(num_timesteps)
    steps = t + 1
    x = torch.linspace(0, t, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / t) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999)


def linear_beta_schedule(num_timesteps: int, *, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear beta schedule (classic DDPM baseline)."""

    return torch.linspace(float(beta_start), float(beta_end), int(num_timesteps), dtype=torch.float64)


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor  # (T,) float64
    alphas: torch.Tensor  # (T,) float64
    alpha_cumprod: torch.Tensor  # (T,) float64
    sqrt_alpha_cumprod: torch.Tensor  # (T,) float64
    sqrt_one_minus_alpha_cumprod: torch.Tensor  # (T,) float64


def make_schedule(num_timesteps: int, *, schedule: str) -> DiffusionSchedule:
    """Builds a diffusion schedule."""

    name = str(schedule).lower()
    if name == "cosine":
        betas = cosine_beta_schedule(num_timesteps)
    elif name == "linear":
        betas = linear_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unsupported schedule: '{schedule}'")

    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_cumprod=alpha_cumprod,
        sqrt_alpha_cumprod=torch.sqrt(alpha_cumprod),
        sqrt_one_minus_alpha_cumprod=torch.sqrt(1.0 - alpha_cumprod),
    )


def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, sched: DiffusionSchedule) -> torch.Tensor:
    """Adds noise to x0 at timesteps t.

    Args:
        x0: (B, ...) clean latents.
        noise: (B, ...) gaussian noise.
        t: (B,) int64 timesteps.
        sched: Precomputed schedule.
    """

    if x0.shape != noise.shape:
        raise ValueError("x0 and noise must have the same shape.")
    if t.ndim != 1 or t.shape[0] != x0.shape[0]:
        raise ValueError("t must be (B,).")

    b = x0.shape[0]
    a = sched.sqrt_alpha_cumprod.to(device=x0.device, dtype=x0.dtype)[t].view(b, *([1] * (x0.ndim - 1)))
    s = sched.sqrt_one_minus_alpha_cumprod.to(device=x0.device, dtype=x0.dtype)[t].view(b, *([1] * (x0.ndim - 1)))
    return (a * x0) + (s * noise)


