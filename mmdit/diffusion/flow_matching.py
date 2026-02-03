"""Flow matching (rectified flow) utilities.

This implements a simple, practical form of flow matching used in modern
diffusion/flow generative models:

- Sample data x0 and noise x1 ~ N(0, I)
- Sample continuous time t ~ Uniform(t_min, t_max) in [0, 1]
- Construct an interpolant x_t = alpha(t) * x0 + sigma(t) * x1
- Train the model to predict the velocity:
    v_t = d/dt x_t = alpha_dot(t) * x0 + sigma_dot(t) * x1

We support two common paths:
- linear:  alpha(t)=1-t, sigma(t)=t
- cosine:  alpha(t)=cos(pi/2 * t), sigma(t)=sin(pi/2 * t)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class FlowMatchingConfig:
    path: str = "linear"  # "linear" | "cosine"
    t_min: float = 0.0
    t_max: float = 1.0
    time_scale: float = 1000.0  # maps t in [0,1] -> embedding scale (e.g. 0..1000)


def sample_t(batch_size: int, *, cfg: FlowMatchingConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Samples continuous t in [t_min, t_max]. Returns (B,) tensor."""

    b = int(batch_size)
    t_min = float(cfg.t_min)
    t_max = float(cfg.t_max)
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError(f"Expected 0 <= t_min < t_max <= 1, got t_min={t_min}, t_max={t_max}")
    t = torch.rand((b,), device=device, dtype=dtype) * (t_max - t_min) + t_min
    return t


def _broadcast(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """(B,) -> (B,1,1,...) for broadcasting with an ndim tensor."""

    return t.view(int(t.shape[0]), *([1] * (int(ndim) - 1)))


def path_coeffs(t: torch.Tensor, *, ndim: int, path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (alpha, sigma, alpha_dot, sigma_dot) broadcastable to x."""

    name = str(path).lower()
    tt = _broadcast(t, ndim)
    if name == "linear":
        alpha = 1.0 - tt
        sigma = tt
        alpha_dot = -torch.ones_like(tt)
        sigma_dot = torch.ones_like(tt)
        return alpha, sigma, alpha_dot, sigma_dot
    if name == "cosine":
        theta = (math.pi / 2.0) * tt
        alpha = torch.cos(theta)
        sigma = torch.sin(theta)
        alpha_dot = -(math.pi / 2.0) * torch.sin(theta)
        sigma_dot = (math.pi / 2.0) * torch.cos(theta)
        return alpha, sigma, alpha_dot, sigma_dot
    raise ValueError(f"Unsupported flow_matching.path: '{path}'")


def make_xt_and_vt(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    *,
    cfg: FlowMatchingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds x_t and target v_t for flow matching."""

    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have the same shape.")
    if t.ndim != 1 or int(t.shape[0]) != int(x0.shape[0]):
        raise ValueError("t must be (B,).")

    alpha, sigma, alpha_dot, sigma_dot = path_coeffs(t, ndim=x0.ndim, path=cfg.path)
    x_t = (alpha * x0) + (sigma * x1)
    v_t = (alpha_dot * x0) + (sigma_dot * x1)
    return x_t, v_t


def time_embedding_value(t: torch.Tensor, *, cfg: FlowMatchingConfig) -> torch.Tensor:
    """Maps t in [0,1] to the value fed into the timestep embedder."""

    return t * float(cfg.time_scale)


