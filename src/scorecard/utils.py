"""Shared file and configuration helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """Return a path relative to the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, returning an empty dict if no config is present."""
    config_path = Path(path) if path is not None else project_path("configs", "config.yaml")
    if not config_path.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load config.yaml. Install package requirements.") from exc

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
