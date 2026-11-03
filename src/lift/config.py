"""Load the YAML configs. Values are plain dicts; ``get`` reads dotted paths and fails loudly."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from lift.io import sha256_obj
from lift.paths import CONFIG_DIR

_MISSING = object()


@cache
def load(name: str = "project") -> dict:
    """Load ``configs/{name}.yaml``. Cached: configs do not change inside a run."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def get(dotted: str, default: Any = _MISSING, *, name: str = "project") -> Any:
    """``get("generation.max_new_tokens")`` -> 256. Raises KeyError unless a default is given."""
    node: Any = load(name)
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif default is not _MISSING:
            return default
        else:
            raise KeyError(f"{name}.yaml has no '{dotted}' (stopped at '{part}')")
    return node


def digest(*names: str) -> str:
    """Digest of the named configs, recorded in every run manifest."""
    names = names or ("project", "train", "repairs")
    return sha256_obj({n: load(n) for n in names})


def rubric(path: str) -> str:
    """Read a frozen judge rubric (path relative to the repo root)."""
    return (CONFIG_DIR.parent / Path(path)).read_text(encoding="utf-8")
