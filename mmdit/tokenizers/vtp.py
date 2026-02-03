"""VTP tokenizer adapter.

This module intentionally wraps only the few APIs needed by this project:

- encode frames -> latents
- (optional) decode latents -> frames

Reference: https://github.com/MiniMax-AI/VTP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class VTPTokenizerConfig:
    name_or_path: str
    latent_f: int = 16
    latent_d: int = 64
    repo_path: Optional[str] = None


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Inverts ImageNet normalization back to [0, 1] range."""

    if x.ndim != 4 or int(x.shape[1]) != 3:
        raise ValueError(f"Expected tensor (B,3,H,W), got {tuple(x.shape)}")
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std) + mean


class VTPTokenizer:
    """A thin wrapper around VTPModel for reconstruction latents."""

    def __init__(self, cfg: VTPTokenizerConfig, *, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        if cfg.repo_path:
            repo = Path(str(cfg.repo_path))
            if not repo.exists():
                raise FileNotFoundError(f"VTP repo_path does not exist: {repo}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

        try:
            from vtp.models.vtp_hf import VTPModel  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "VTP cannot be imported.\n"
                "Fix options:\n"
                "  1) Download/clone VTP and add its repo root to PYTHONPATH, e.g.\n"
                "       export PYTHONPATH=./VTP:$PYTHONPATH\n"
                "  2) Or set tokenizer.repo_path in your YAML config.\n"
                "Also install VTP deps:\n"
                "  python3 -m pip install -r ./VTP/requirements.txt\n"
                "Reference: https://github.com/MiniMax-AI/VTP"
            ) from e

        # * Most VTP checkpoints are HuggingFace-style.
        if not hasattr(VTPModel, "from_pretrained"):
            raise RuntimeError("Unsupported VTPModel API: missing from_pretrained().")

        self.model = VTPModel.from_pretrained(cfg.name_or_path).to(device)  # type: ignore[attr-defined]
        self.model.eval()

    @torch.no_grad()
    def encode(self, frames: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
        """Encodes frames into reconstruction latents.

        Args:
            frames: Tensor in shape (B, F, 3, H, W), ImageNet-normalized.
            width: Original frame width (after crop/resize).
            height: Original frame height (after crop/resize).

        Returns:
            Latents in canonical shape (B, F, H_lat, W_lat, D_lat).
        """

        if frames.ndim != 5:
            raise ValueError(f"Expected frames shape (B,F,3,H,W), got {tuple(frames.shape)}")
        b, f, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"Expected RGB frames (C=3), got C={c}")
        if int(h) != int(height) or int(w) != int(width):
            raise ValueError(f"Frame size mismatch: got {(w, h)}, expected {(width, height)}")

        frames = frames.to(self.device, non_blocking=True)
        x = frames.reshape(b * f, c, h, w)

        # * VTP expects autocast for speed on CUDA.
        autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            latents = self.model.get_reconstruction_latents(x)  # type: ignore[attr-defined]

        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Unexpected latents type from VTP: {type(latents).__name__}")

        h_lat = int(height) // int(self.cfg.latent_f)
        w_lat = int(width) // int(self.cfg.latent_f)
        d_lat = int(self.cfg.latent_d)

        if latents.ndim == 4:
            # (B*F, D, H_lat, W_lat) -> (B*F, H_lat, W_lat, D)
            if int(latents.shape[1]) != d_lat:
                raise ValueError(f"Unexpected latent dim: got {latents.shape[1]}, expected {d_lat}")
            latents = latents.permute(0, 2, 3, 1).contiguous()
        elif latents.ndim == 3:
            # (B*F, N, D) -> (B*F, H_lat, W_lat, D)
            if int(latents.shape[-1]) != d_lat:
                raise ValueError(f"Unexpected latent dim: got {latents.shape[-1]}, expected {d_lat}")
            n = int(latents.shape[1])
            if n != h_lat * w_lat:
                raise ValueError(f"Unexpected token count: got {n}, expected {h_lat*w_lat} (H_lat={h_lat},W_lat={w_lat})")
            latents = latents.view(b * f, h_lat, w_lat, d_lat).contiguous()
        else:
            raise ValueError(f"Unsupported latents ndim from VTP: {latents.ndim}")

        latents = latents.view(b, f, h_lat, w_lat, d_lat)
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
        """Decodes reconstruction latents back to images.

        Args:
            latents: (B, F, H_lat, W_lat, D_lat)
            width: Target output width.
            height: Target output height.

        Returns:
            Images in [0, 1] with shape (B, F, 3, H, W).
        """

        if latents.ndim != 5:
            raise ValueError(f"Expected latents (B,F,H_lat,W_lat,D), got {tuple(latents.shape)}")
        b, f, h_lat, w_lat, d = latents.shape
        if int(d) != int(self.cfg.latent_d):
            raise ValueError(f"latent_d mismatch: got {d}, expected {self.cfg.latent_d}")

        # (B*F, D, H_lat, W_lat)
        z = latents.to(self.device, non_blocking=True).reshape(b * f, h_lat, w_lat, d).permute(0, 3, 1, 2).contiguous()

        autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.device.type == "cuda"):
            recon = self.model.get_latents_decoded_images(z)  # type: ignore[attr-defined]

        if not isinstance(recon, torch.Tensor):
            raise TypeError(f"Unexpected recon type from VTP: {type(recon).__name__}")
        if recon.ndim != 4 or int(recon.shape[1]) != 3:
            raise ValueError(f"Unexpected recon shape from VTP: {tuple(recon.shape)}")

        # * VTP returns ImageNet-normalized images; invert to [0, 1].
        recon = denormalize_imagenet(recon).clamp(0.0, 1.0)
        recon = recon.view(b, f, 3, int(height), int(width))
        return recon


