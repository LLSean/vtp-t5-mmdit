"""Config utilities for this project.

This project intentionally uses a simple YAML config (dict-like) to keep the
scaffold easy to modify during early research iterations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Loads a YAML config file.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed config mapping.

    Raises:
        ValueError: If the YAML does not parse to a mapping.
    """

    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (top-level dict).")
    return cfg


def ensure_not_placeholder(value: str, field_name: str) -> None:
    """Raises if a config field is unset or still uses a placeholder."""

    if not value:
        raise ValueError(f"Config field '{field_name}' must be set (empty).")
    if "PATH_TO_" in value:
        raise ValueError(f"Config field '{field_name}' must be set (placeholder: '{value}').")


def get_required(cfg: dict[str, Any], key: str, *, expected_type: type) -> Any:
    """Reads a required config entry with type checking."""

    if key not in cfg:
        raise KeyError(f"Missing required config key: '{key}'")
    value = cfg[key]
    if not isinstance(value, expected_type):
        raise TypeError(f"Config key '{key}' must be {expected_type.__name__}, got: {type(value).__name__}")
    return value


