"""Checkpoint utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class CheckpointState:
    global_step: int
    path: Path


def find_latest_checkpoint(directory: str | Path, prefix: str = "step") -> Optional[Path]:
    """Finds the latest checkpoint in a directory."""

    d = Path(directory)
    if not d.is_dir():
        return None
    candidates = sorted(d.glob(f"{prefix}_*.pt"))
    return candidates[-1] if candidates else None


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "global_step": int(global_step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device | None = None,
) -> CheckpointState:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return CheckpointState(global_step=int(payload.get("global_step", 0)), path=Path(path))


