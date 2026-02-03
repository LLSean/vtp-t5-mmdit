"""Local window self-attention blocks (Swin-style).

This is adapted from the existing pix2pix SLA implementation in the same
workspace, but duplicated here to keep this project self-contained.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partitions feature maps into non-overlapping windows.

    Args:
        x: Tensor in NHWC format with shape (B, H, W, C).
        window_size: Window size.

    Returns:
        Windows with shape (B * num_windows, window_size, window_size, C).
    """

    b, h, w, c = x.shape
    x = x.view(
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverses window partition back to feature maps.

    Args:
        windows: Tensor with shape (B * num_windows, window_size, window_size, C).
        window_size: Window size.
        h: Height (padded).
        w: Width (padded).

    Returns:
        Reconstructed tensor in NHWC with shape (B, H, W, C).
    """

    num_windows_per_image = (h // window_size) * (w // window_size)
    b = int(windows.shape[0] // num_windows_per_image)
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        # * Separate projections make it easy to apply LoRA or freeze Q/K.
        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(self.dim, self.dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # * Relative position bias.
        num_rel = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_rel, self.num_heads))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Ws, Ws)
        coords_flatten = torch.flatten(coords, 1)  # (2, Ws*Ws)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tokens with shape (B_, N, C).
            mask: Optional attention mask with shape (nW, N, N).

        Returns:
            Output tokens with shape (B_, N, C).
        """

        b_, n, c = x.shape
        q = self.q(x).view(b_, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).view(b_, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).view(b_, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)

        rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(n, n, -1).permute(2, 0, 1).contiguous()  # (heads, N, N)
        attn = attn + rel_bias.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(b_, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SpatialWindowAttention(nn.Module):
    """Shifted-window attention on a 2D grid (no internal LayerNorm/MLP).

    This module is designed to be used inside an AdaLN-conditioned block.
    """

    def __init__(
        self,
        *,
        dim: int,
        window_size: int,
        num_heads: int,
        shift_size: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if self.shift_size >= self.window_size:
            raise ValueError("shift_size must be smaller than window_size.")

        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=int(num_heads),
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def _build_attn_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, h, w, 1), device=device)  # (1, H, W, 1)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, hs, ws_, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)  # (nW, Ws, Ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies window attention.

        Args:
            x: NHWC tensor with shape (B, H, W, C).

        Returns:
            NHWC tensor with shape (B, H, W, C).
        """

        if x.ndim != 4:
            raise ValueError(f"Expected NHWC tensor, got {tuple(x.shape)}")
        b, h, w, c = x.shape
        ws = self.window_size

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._build_attn_mask(hp, wp, x.device)
        else:
            shifted = x
            attn_mask = None

        windows = window_partition(shifted, ws).view(-1, ws * ws, c)  # (B*nW, N, C)
        windows = self.attn(windows, mask=attn_mask)
        windows = windows.view(-1, ws, ws, c)

        shifted = window_reverse(windows, ws, hp, wp)
        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        x = x[:, :h, :w, :].contiguous()
        return x


