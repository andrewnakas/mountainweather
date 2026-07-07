"""Central config loader.

All tunables live in ``configs/*.yaml`` so the code stays declarative and the same
scripts run for a smoke test (10 stations, 1 week) or the full backfill just by
pointing at a different config. ``load_configs()`` returns a single namespace-like
dict; access with ``cfg["region"]``, ``cfg["variables"]``, etc.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/mtnwx/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_CONFIG_FILES = {
    "region": "region.yaml",
    "variables": "variables.yaml",
    "predictors": "predictors.yaml",
    "model": "model.yaml",
    "hub": "hub.yaml",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def load_configs(config_dir: str | None = None) -> dict[str, Any]:
    """Load and cache all config YAMLs into one dict keyed by config name."""
    base = Path(config_dir) if config_dir else CONFIG_DIR
    out: dict[str, Any] = {}
    for key, fname in _CONFIG_FILES.items():
        path = base / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing config: {path}")
        out[key] = _load_yaml(path)
    return out


def data_dir() -> Path:
    """Local scratch/data directory (git-ignored)."""
    d = REPO_ROOT / "data"
    d.mkdir(exist_ok=True)
    return d
