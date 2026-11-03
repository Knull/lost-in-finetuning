"""Deterministic dev/final (and fit) splits by prompt id, identical in every language.

A split is computed once per prompt set from the sorted pids, the project split seed and the
locked sizes, then written to data/splits/{name}.json and committed. Every stage reads the
committed file; nothing recomputes splits on the fly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import date

import numpy as np

from lift.paths import SPLITS_DIR


def assign_splits(pids: Iterable[str], sizes: Mapping[str, int], seed: int) -> dict[str, str]:
    """Map each pid to a split name. ``sizes`` must sum to the number of pids exactly.

    The order of ``sizes`` matters (first name takes the first slice of the permutation), so pass
    the dict straight from the config.
    """
    pids = list(pids)
    unique = sorted(set(pids))
    if len(unique) != len(pids):
        raise ValueError("duplicate pids")
    total = sum(sizes.values())
    if total != len(unique):
        raise ValueError(f"sizes sum to {total}, but there are {len(unique)} pids")
    order = np.random.default_rng(seed).permutation(len(unique))
    out: dict[str, str] = {}
    start = 0
    for name, n in sizes.items():
        for j in order[start : start + n]:
            out[unique[int(j)]] = name
        start += n
    return out


def save_split(name: str, mapping: Mapping[str, str], *, seed: int, sizes: Mapping[str, int],
               overwrite: bool = False) -> None:
    path = SPLITS_DIR / f"{name}.json"
    if path.exists() and not overwrite:
        existing = load_split(name)
        if existing != dict(mapping):
            raise FileExistsError(f"{path} exists with different contents; splits are frozen")
        return
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "name": name,
        "seed": seed,
        "sizes": dict(sizes),
        "created": date.today().isoformat(),
        "n": len(mapping),
        "assignments": dict(sorted(mapping.items())),
    }
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_split(name: str) -> dict[str, str]:
    path = SPLITS_DIR / f"{name}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    return dict(doc["assignments"])


def check_disjoint(a: Iterable[str], b: Iterable[str], what: str = "sets") -> None:
    """Raise if two id collections overlap, e.g. utility held-out ids vs training ids."""
    overlap = set(a) & set(b)
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"{what} overlap on {len(overlap)} ids, e.g. {sample}")
