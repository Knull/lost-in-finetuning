"""JSONL and hashing helpers shared by every stage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def stable_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no spaces, unicode kept. Used for every digest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(stable_json(obj))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad JSON line") from e


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write atomically (temp file + rename) so a killed job never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    n = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(stable_json(row) + "\n")
                n += 1
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return n


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Append and flush line by line. Resumable stages use this plus ``existing_keys``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(stable_json(row) + "\n")
            f.flush()
            n += 1
    return n


def existing_keys(path: str | Path, field: str = "key") -> set[str]:
    """Keys already written to a JSONL file. A truncated last line is ignored, not fatal."""
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                keys.add(json.loads(line)[field])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys
