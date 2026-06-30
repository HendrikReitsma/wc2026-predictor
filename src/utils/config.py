from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIG_DIR


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (CONFIG_DIR / "config.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file {path} did not contain a top-level mapping.")
    return loaded


def config_value(*keys: str, default: Any | None = None, config_path: Path | None = None) -> Any:
    config: Any = load_config(config_path)
    for key in keys:
        if not isinstance(config, dict) or key not in config:
            return default
        config = config[key]
    return config
