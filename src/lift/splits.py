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


def assign_splits(pids: Iterable[str], sizes: Mapping[str, int], seed: int,
                  groups: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map each pid to a split name. ``sizes`` must sum to the number of pids exactly.

    The order of ``sizes`` matters (first name takes the first slice of the permutation), so pass
    the dict straight from the config.

    ``groups`` maps pids to a group key; pids sharing a key always land in the same split. The
    build uses it for prompts whose English text is identical (PolyRefuse's harmful test repeats
    10 prompts), so the same prompt can never sit in both dev and final. Groups are permuted with
    the seed and each split, in config order, takes the next groups that still fit, so the sizes
    stay exact. Without ``groups`` the assignment is the plain permutation, unchanged.
    """
    pids = list(pids)
    unique = sorted(set(pids))
    if len(unique) != len(pids):
        raise ValueError("duplicate pids")
    total = sum(sizes.values())
    if total != len(unique):
        raise ValueError(f"sizes sum to {total}, but there are {len(unique)} pids")
    out: dict[str, str] = {}
    if groups is None:
        order = np.random.default_rng(seed).permutation(len(unique))
        start = 0
        for name, n in sizes.items():
            for j in order[start : start + n]:
                out[unique[int(j)]] = name
            start += n
        return out

    members: dict[str, list[str]] = {}
    for pid in unique:
        members.setdefault(str(groups.get(pid, pid)), []).append(pid)
    keys = sorted(members)
    pending = [keys[int(j)] for j in np.random.default_rng(seed).permutation(len(keys))]
    for name, n in sizes.items():
        room, rest = n, []
        for key in pending:
            if len(members[key]) <= room:
                room -= len(members[key])
                out.update(dict.fromkeys(members[key], name))
            else:
                rest.append(key)
        if room:
            raise ValueError(f"cannot fill split {name!r} exactly with whole groups ({room} short)")
        pending = rest
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
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


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
