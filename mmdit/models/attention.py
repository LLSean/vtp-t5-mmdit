"""Attention layers implemented with explicit Linear modules (LoRA-friendly)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

_SDPA_FALLBACK_WARNED = False


def _shape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, N, D) -> (B, heads, N, head_dim)"""

    b, n, d = x.shape
    h = int(num_heads)
    if d % h != 0:
        raise ValueError(f"dim ({d}) must be divisible by num_heads ({h})")
    hd = d // h
    return x.view(b, n, h, hd).transpose(1, 2).contiguous()


def _unshape_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, heads, N, head_dim) -> (B, N, D)"""

    b, h, n, hd = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * hd)


class SelfAttention(nn.Module):
    """Multi-head self-attention with explicit Q/K/V projections."""

    def __init__(self, *, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.q = nn.Linear(self.dim, self.dim, bias=True)
        self.k = nn.Linear(self.dim, self.dim, bias=True)
        self.v = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """

        if x.ndim != 3:
            raise ValueError(f"Expected x (B,N,D), got {tuple(x.shape)}")

        # * Fast path: for N==1, attention reduces to the V projection (no need for SDPA).
        # * This avoids some SDPA kernel edge cases on certain CUDA builds.
        if int(x.shape[1]) == 1:
            return self.proj(self.v(x))

        q = _shape_heads(self.q(x), self.num_heads)
        k = _shape_heads(self.k(x), self.num_heads)
        v = _shape_heads(self.v(x), self.num_heads)

        # * Prefer SDPA on CUDA for memory efficiency. Use an explicit softmax path on CPU
        # * to avoid backend-specific numerical quirks.
        if x.device.type == "cuda":
            try:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training and self.dropout > 0 else 0.0,
                    is_causal=False,
                )
            except Exception as e:  # noqa: BLE001
                # * Fallback: some CUDA SDPA kernels can fail with "invalid configuration argument".
                global _SDPA_FALLBACK_WARNED
                if not _SDPA_FALLBACK_WARNED:
                    _SDPA_FALLBACK_WARNED = True
                    print(f"[warn] SDPA failed on CUDA; falling back to math attention. err={type(e).__name__}: {e}")
                hd = q.shape[-1]
                scale = 1.0 / float(hd) ** 0.5
                attn = (q * scale) @ k.transpose(-2, -1)
                attn = torch.softmax(attn, dim=-1)
                if self.training and self.dropout > 0:
                    attn = F.dropout(attn, p=self.dropout)
                out = attn @ v
        else:
            hd = q.shape[-1]
            scale = 1.0 / float(hd) ** 0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v
        out = _unshape_heads(out)
        out = self.proj(out)
        return out


class CrossAttention(nn.Module):
    """Multi-head cross-attention with explicit Q/K/V projections."""

    def __init__(self, *, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.q = nn.Linear(self.dim, self.dim, bias=True)
        self.k = nn.Linear(self.dim, self.dim, bias=True)
        self.v = nn.Linear(self.dim, self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, x: torch.Tensor, *, context: torch.Tensor, context_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Args:
            x: (B, N, D)
            context: (B, T, D)
            context_mask: Optional (B, T) bool, True for valid tokens
        Returns:
            (B, N, D)
        """

        if x.ndim != 3:
            raise ValueError(f"Expected x (B,N,D), got {tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(f"Expected context (B,T,D), got {tuple(context.shape)}")
        if x.shape[0] != context.shape[0]:
            raise ValueError("Batch size mismatch between x and context.")

        q = _shape_heads(self.q(x), self.num_heads)
        k = _shape_heads(self.k(context), self.num_heads)
        v = _shape_heads(self.v(context), self.num_heads)

        attn_mask = None
        if context_mask is not None:
            if context_mask.ndim != 2:
                raise ValueError("context_mask must be (B, T).")
            # * SDPA bool mask: True means "masked out".
            attn_mask = (~context_mask).to(torch.bool)[:, None, None, :]  # (B,1,1,T), broadcastable

        if x.device.type == "cuda":
            try:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training and self.dropout > 0 else 0.0,
                    is_causal=False,
                )
            except Exception as e:  # noqa: BLE001
                global _SDPA_FALLBACK_WARNED
                if not _SDPA_FALLBACK_WARNED:
                    _SDPA_FALLBACK_WARNED = True
                    print(f"[warn] SDPA failed on CUDA; falling back to math attention. err={type(e).__name__}: {e}")
                hd = q.shape[-1]
                scale = 1.0 / float(hd) ** 0.5
                attn = (q * scale) @ k.transpose(-2, -1)
                if attn_mask is not None:
                    attn = attn.masked_fill(attn_mask, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                if self.training and self.dropout > 0:
                    attn = F.dropout(attn, p=self.dropout)
                out = attn @ v
        else:
            hd = q.shape[-1]
            scale = 1.0 / float(hd) ** 0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v
        out = _unshape_heads(out)
        out = self.proj(out)
        return out


