# src/utils/config.py

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Shallow merge — override wins."""
    merged = base.copy()
    merged.update(override)
    return merged
