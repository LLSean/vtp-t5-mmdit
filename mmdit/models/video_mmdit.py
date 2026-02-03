"""A minimal MMDiT-style video denoiser (factorized space + time attention).

Design goals:
- Scalability to longer clips via factorized attention:
  - spatial: local window attention (Swin-style)
  - temporal: self-attention along frames for each spatial location
- Diffusion timestep conditioning via AdaLN-Zero (DiT-style)
- Text conditioning via cross-attention
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from mmdit.models.attention import CrossAttention, SelfAttention
from mmdit.models.window_attention import SpatialWindowAttention


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (DiT/DDPM-style)."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1D tensor.")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Maps diffusion timesteps to a model-dim conditioning vector."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim or (4 * d))
        self.mlp = nn.Sequential(
            nn.Linear(d, h),
            nn.SiLU(),
            nn.Linear(h, d),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_timestep_embedding(t, self.mlp[0].in_features))


class Mlp(nn.Module):
    """Transformer MLP."""

    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        h = int(d * float(mlp_ratio))
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(h, d)
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class VideoDiTBlock(nn.Module):
    """A single factorized video transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        window_size: int,
        shift: bool,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        ws = int(window_size)
        ss = ws // 2 if bool(shift) else 0

        self.norm_spatial = nn.LayerNorm(d, elementwise_affine=False)
        self.spatial_attn = SpatialWindowAttention(
            dim=d,
            window_size=ws,
            num_heads=h,
            shift_size=ss,
            attn_drop=float(dropout),
            proj_drop=float(dropout),
        )

        self.norm_temporal = nn.LayerNorm(d, elementwise_affine=False)
        self.temporal_attn = SelfAttention(dim=d, num_heads=h, dropout=float(dropout))

        self.norm_cross = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = CrossAttention(dim=d, num_heads=h, dropout=float(dropout))

        self.norm_mlp = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = Mlp(dim=d, mlp_ratio=mlp_ratio, dropout=dropout)

        # * AdaLN-Zero modulation: (shift, scale, gate) for 4 sublayers.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 12 * d))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cond: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        tread_idx_hw: Optional[torch.Tensor] = None,
        tread_apply: Optional[tuple[str, ...]] = None,
        tread_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, F, H, W, D)
            cond: (B, D) timestep embedding
            text: (B, T, D) projected text tokens
            text_mask: Optional (B, T) bool, True for valid tokens.
        """

        if x.ndim != 5:
            raise ValueError(f"Expected x (B,F,H,W,D), got {tuple(x.shape)}")
        b, f, h, w, d = x.shape
        n_hw = int(h) * int(w)
        use_tread = tread_idx_hw is not None and tread_apply is not None and bool(tread_apply)
        if use_tread:
            if tread_idx_hw.ndim != 1:
                raise ValueError(f"tread_idx_hw must be 1D, got {tuple(tread_idx_hw.shape)}")
            if tread_idx_hw.numel() <= 0:
                use_tread = False
            else:
                tread_idx_hw = tread_idx_hw.to(device=x.device, dtype=torch.long)
                # * Expand HW indices to token indices across frames.
                offs = (torch.arange(int(f), device=x.device, dtype=torch.long) * int(n_hw)).view(int(f), 1)
                idx_token = (offs + tread_idx_hw.view(1, -1)).reshape(-1)  # (F*K,)
        else:
            idx_token = None
        ts = float(tread_scale)
        if ts <= 0:
            raise ValueError(f"tread_scale must be > 0, got {tread_scale}")

        m = self.adaLN_modulation(cond).view(b, 12, d)
        shift_s, scale_s, gate_s = m[:, 0], m[:, 1], m[:, 2]
        shift_t, scale_t, gate_t = m[:, 3], m[:, 4], m[:, 5]
        shift_c, scale_c, gate_c = m[:, 6], m[:, 7], m[:, 8]
        shift_m, scale_m, gate_m = m[:, 9], m[:, 10], m[:, 11]

        # ---------------------
        # (1) Spatial attention (per frame)
        # ---------------------
        xs = self.norm_spatial(x)
        xs = _modulate(xs, shift_s.view(b, 1, 1, 1, d), scale_s.view(b, 1, 1, 1, d))
        xs = xs.view(b * f, h, w, d)
        xs = self.spatial_attn(xs)
        xs = xs.view(b, f, h, w, d)
        x = x + gate_s.view(b, 1, 1, 1, d) * xs

        # ---------------------
        # (2) Temporal attention (per spatial location)
        # ---------------------
        xt = self.norm_temporal(x)
        xt = _modulate(xt, shift_t.view(b, 1, 1, 1, d), scale_t.view(b, 1, 1, 1, d))
        if use_tread and ("temporal" in tread_apply):
            # * Route only selected spatial locations through temporal attention.
            xt_hw = xt.permute(0, 2, 3, 1, 4).contiguous().view(b, n_hw, f, d)  # (B, HW, F, D)
            xt_sel = xt_hw.index_select(1, tread_idx_hw).reshape(b * int(tread_idx_hw.numel()), f, d)  # (B*K, F, D)
            xt_sel = self.temporal_attn(xt_sel).view(b, int(tread_idx_hw.numel()), f, d)

            x_hw = x.permute(0, 2, 3, 1, 4).contiguous().view(b, n_hw, f, d)
            x_hw_out = x_hw.clone()
            x_hw_out[:, tread_idx_hw, :, :] = x_hw_out[:, tread_idx_hw, :, :] + ((gate_t * ts).view(b, 1, 1, d) * xt_sel)
            x = x_hw_out.view(b, h, w, f, d).permute(0, 3, 1, 2, 4).contiguous()
        else:
            xt = xt.permute(0, 2, 3, 1, 4).contiguous().view(b * h * w, f, d)  # (B*H*W, F, D)
            xt = self.temporal_attn(xt)
            xt = xt.view(b, h, w, f, d).permute(0, 3, 1, 2, 4).contiguous()
            x = x + gate_t.view(b, 1, 1, 1, d) * xt

        # ---------------------
        # (3) Cross-attention to text (global)
        # ---------------------
        xc = self.norm_cross(x)
        xc = _modulate(xc, shift_c.view(b, 1, 1, 1, d), scale_c.view(b, 1, 1, 1, d))
        if use_tread and ("cross" in tread_apply):
            assert idx_token is not None
            q = xc.view(b, f * h * w, d).index_select(1, idx_token)
            out = self.cross_attn(q, context=text, context_mask=text_mask)  # (B, F*K, D)
            x_flat = x.view(b, f * h * w, d)
            x_out = x_flat.clone()
            x_out[:, idx_token, :] = x_out[:, idx_token, :] + ((gate_c * ts).view(b, 1, d) * out)
            x = x_out.view(b, f, h, w, d)
        else:
            q = xc.view(b, f * h * w, d)
            out = self.cross_attn(q, context=text, context_mask=text_mask)
            out = out.view(b, f, h, w, d)
            x = x + gate_c.view(b, 1, 1, 1, d) * out

        # ---------------------
        # (4) MLP
        # ---------------------
        xm = self.norm_mlp(x)
        xm = _modulate(xm, shift_m.view(b, 1, 1, 1, d), scale_m.view(b, 1, 1, 1, d))
        if use_tread and ("mlp" in tread_apply):
            assert idx_token is not None
            xm_sel = xm.view(b, f * h * w, d).index_select(1, idx_token)
            n_sel = int(idx_token.numel())
            xm_out = self.mlp(xm_sel.reshape(b * n_sel, d)).view(b, n_sel, d)
            x_flat = x.view(b, f * h * w, d)
            x_out = x_flat.clone()
            x_out[:, idx_token, :] = x_out[:, idx_token, :] + ((gate_m * ts).view(b, 1, d) * xm_out)
            x = x_out.view(b, f, h, w, d)
        else:
            xm = self.mlp(xm.view(b * f * h * w, d)).view(b, f, h, w, d)
            x = x + gate_m.view(b, 1, 1, 1, d) * xm

        return x


@dataclass(frozen=True)
class VideoMMDiTConfig:
    latent_d: int
    model_dim: int
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float
    dropout: float
    text_dim: int


class VideoMMDiT(nn.Module):
    """Video diffusion denoiser operating on VTP-style latents."""

    def __init__(self, cfg: VideoMMDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(int(cfg.latent_d), int(cfg.model_dim))
        self.text_proj = nn.Linear(int(cfg.text_dim), int(cfg.model_dim))
        self.time_embed = TimestepEmbedder(int(cfg.model_dim))

        blocks: list[nn.Module] = []
        for i in range(int(cfg.depth)):
            blocks.append(
                VideoDiTBlock(
                    dim=int(cfg.model_dim),
                    num_heads=int(cfg.num_heads),
                    window_size=int(cfg.window_size),
                    shift=bool(i % 2 == 1),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.LayerNorm(int(cfg.model_dim))
        self.out_proj = nn.Linear(int(cfg.model_dim), int(cfg.latent_d))

        # * TREAD-style token routing (training-time acceleration).
        # * This is configured via `configure_tread()` and does not affect the model state_dict.
        self._tread_enabled = False
        self._tread_selection_rate = 1.0
        self._tread_apply: tuple[str, ...] = ("cross", "mlp")
        self._tread_mode = "rotate"  # "rotate" | "random"
        self._tread_rescale = False

    def configure_tread(self, cfg: dict[str, Any]) -> None:
        """Configures TREAD-style token routing.

        Expected YAML shape:
          tread:
            enabled: true
            selection_rate: 0.5
            apply_to: [cross, mlp]  # optional: temporal
            mode: rotate
        """

        c = cfg or {}
        self._tread_enabled = bool(c.get("enabled", False))
        self._tread_selection_rate = float(c.get("selection_rate", 1.0))
        if self._tread_selection_rate <= 0.0 or self._tread_selection_rate > 1.0:
            raise ValueError(f"tread.selection_rate must be in (0, 1], got {self._tread_selection_rate}")
        mode = str(c.get("mode", "rotate")).strip().lower()
        if mode not in ("rotate", "random"):
            raise ValueError(f"Unsupported tread.mode: '{mode}' (expected 'rotate' or 'random')")
        self._tread_mode = mode
        apply_to = c.get("apply_to", None)
        if apply_to is None:
            apply = ("cross", "mlp")
        else:
            if not isinstance(apply_to, list):
                raise TypeError("tread.apply_to must be a YAML list of strings.")
            apply = tuple(str(x).strip().lower() for x in apply_to)
        allowed = {"temporal", "cross", "mlp"}
        if any(a not in allowed for a in apply):
            raise ValueError(f"tread.apply_to contains unsupported items: {apply} (allowed={sorted(allowed)})")
        self._tread_apply = apply
        self._tread_rescale = bool(c.get("rescale", False))

    def forward(
        self,
        x: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: Noisy latents with shape (B, F, H, W, D_lat).
            timesteps: (B,) int64
            text_tokens: (B, T, D_text)
            text_mask: Optional (B, T) bool

        Returns:
            Predicted noise with shape (B, F, H, W, D_lat).
        """

        if x.ndim != 5:
            raise ValueError(f"Expected x (B,F,H,W,D_lat), got {tuple(x.shape)}")
        b, f, h, w, d_lat = x.shape
        if int(d_lat) != int(self.cfg.latent_d):
            raise ValueError(f"latent_d mismatch: got {d_lat}, expected {self.cfg.latent_d}")

        # * Token projection.
        x = self.in_proj(x)

        # * Conditioning.
        cond = self.time_embed(timesteps)
        # * Normalize conditioning dtypes to match the latent stream.
        # * This keeps sampling in float32 stable even when the text encoder runs under BF16 autocast.
        if cond.dtype != x.dtype:
            cond = cond.to(dtype=x.dtype)
        if text_tokens.dtype != x.dtype:
            text_tokens = text_tokens.to(dtype=x.dtype)
        text = self.text_proj(text_tokens)

        tread_on = bool(self._tread_enabled) and self.training and float(self._tread_selection_rate) < 1.0
        perm_hw = None
        k_hw = None
        tread_scale = 1.0
        if tread_on:
            n_hw = int(h) * int(w)
            k_hw = max(1, int(round(float(self._tread_selection_rate) * float(n_hw))))
            perm_hw = torch.randperm(n_hw, device=x.device)
            if bool(self._tread_rescale):
                tread_scale = 1.0 / float(self._tread_selection_rate)

        for i, block in enumerate(self.blocks):
            idx_hw = None
            if tread_on and perm_hw is not None and k_hw is not None:
                n_hw = int(h) * int(w)
                if self._tread_mode == "random":
                    idx_hw = perm_hw[:k_hw]
                else:
                    # * Rotate through a fixed permutation so tokens take different routes across depth.
                    start = (int(i) * int(k_hw)) % int(n_hw)
                    end = start + int(k_hw)
                    if end <= n_hw:
                        idx_hw = perm_hw[start:end]
                    else:
                        idx_hw = torch.cat([perm_hw[start:], perm_hw[: end - n_hw]], dim=0)

            if tread_on:
                x = block(
                    x,
                    cond=cond,
                    text=text,
                    text_mask=text_mask,
                    tread_idx_hw=idx_hw,
                    tread_apply=self._tread_apply,
                    tread_scale=tread_scale,
                )
            else:
                x = block(x, cond=cond, text=text, text_mask=text_mask)

        x = self.out_norm(x)
        x = self.out_proj(x)
        return x


