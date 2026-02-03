"""Small utilities shared across modules."""

from __future__ import annotations

import os
import random
from typing import Optional

import torch


def set_global_seed(seed: int) -> None:
    """Sets RNG seeds for Python and PyTorch."""

    s = int(seed)
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_rank() -> int:
    """Returns process rank (0 if not in distributed mode)."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def local_rank_from_env(default: int = 0) -> int:
    """Returns LOCAL_RANK if available (torchrun), else default."""

    v = os.environ.get("LOCAL_RANK")
    if v is None:
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def maybe_init_distributed(backend: str = "nccl") -> None:
    """Initializes torch.distributed if launched with torchrun."""

    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    torch.distributed.init_process_group(backend=backend)


def select_device(device: str) -> torch.device:
    """Selects a device.

    Args:
        device: One of "auto", "cpu", "cuda".
    """

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def autocast_dtype(precision: str) -> Optional[torch.dtype]:
    """Returns autocast dtype for the given precision string."""

    p = str(precision).lower()
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16"):
        return torch.float16
    if p in ("fp32", "float32", "no", "none"):
        return None
    raise ValueError(f"Unsupported precision: '{precision}'")


