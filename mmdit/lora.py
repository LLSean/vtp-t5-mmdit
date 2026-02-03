"""LoRA utilities (minimal, dependency-free).

This implementation focuses on training-time correctness and simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    train_lora_only: bool = True


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear.

    Forward: y = W x + scale * B(A(x))
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base module.")
        r = int(rank)
        if r <= 0:
            raise ValueError("rank must be > 0")

        self.base = base
        self.rank = r
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r)
        self.drop = nn.Dropout(float(dropout))

        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)

        # * Initialization: A ~ N(0, 0.01), B = 0 (start as no-op).
        nn.init.normal_(self.lora_a.weight, std=0.01)
        nn.init.zeros_(self.lora_b.weight)

        # * Freeze base weights by default (LoRA fine-tuning).
        for p in self.base.parameters():
            p.requires_grad = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, rank: int, alpha: int, dropout: float) -> "LoRALinear":
        return cls(linear, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.scale * self.lora_b(self.lora_a(self.drop(x))))


def apply_lora_(module: nn.Module, cfg: LoRAConfig) -> int:
    """In-place LoRA injection. Returns number of wrapped Linear layers."""

    if not bool(cfg.enabled):
        return 0

    targets = set(str(t) for t in cfg.target_modules)
    count = 0

    def _rec(parent: nn.Module) -> None:
        nonlocal count
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(
                    parent,
                    name,
                    LoRALinear.from_linear(child, rank=int(cfg.rank), alpha=int(cfg.alpha), dropout=float(cfg.dropout)),
                )
                count += 1
            else:
                _rec(child)

    _rec(module)
    if bool(cfg.train_lora_only):
        mark_only_lora_as_trainable_(module)
    return count


def mark_only_lora_as_trainable_(module: nn.Module) -> None:
    """Freezes all params except LoRA adapters."""

    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_a.parameters():
                p.requires_grad = True
            for p in m.lora_b.parameters():
                p.requires_grad = True


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Extracts only LoRA parameters from a module."""

    out: dict[str, torch.Tensor] = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            out[f"{name}.lora_a.weight"] = m.lora_a.weight.detach().cpu()
            out[f"{name}.lora_b.weight"] = m.lora_b.weight.detach().cpu()
    return out


