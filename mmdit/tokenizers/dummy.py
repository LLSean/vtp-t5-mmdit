"""A tiny dummy tokenizer used for unit tests (no external weights)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DummyTokenizerConfig:
    latent_f: int = 16
    latent_d: int = 64


class DummyTokenizer(nn.Module):
    """Downsamples RGB frames and projects to `latent_d` channels.

    This intentionally mimics the *shape* of a VTP-style tokenizer but is not a
    meaningful tokenizer for generation quality.
    """

    def __init__(self, cfg: DummyTokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(3, int(cfg.latent_d), kernel_size=1, stride=1, padding=0)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected frames shape (B,F,3,H,W), got {tuple(frames.shape)}")
        b, f, c, h, w = frames.shape
        if c != 3:
            raise ValueError("DummyTokenizer expects RGB frames (C=3).")
        if int(h) != int(height) or int(w) != int(width):
            raise ValueError("Frame size mismatch.")

        x = frames.reshape(b * f, c, h, w)
        # * Downsample to latent grid (H/latent_f, W/latent_f).
        x = F.interpolate(x, scale_factor=1.0 / float(self.cfg.latent_f), mode="bilinear", align_corners=False)
        x = self.proj(x)  # (B*F, D, H_lat, W_lat)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B*F, H_lat, W_lat, D)
        h_lat, w_lat = x.shape[1], x.shape[2]
        return x.view(b, f, h_lat, w_lat, int(self.cfg.latent_d))


