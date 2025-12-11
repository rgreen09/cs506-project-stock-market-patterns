import os
from functools import lru_cache
from typing import Any, Dict

import yaml


CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "patterns.yaml",
)


@lru_cache(maxsize=1)
def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load YAML configuration for the pattern pipeline.

    Uses an LRU cache so callers can simply import and call without
    worrying about performance.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}

