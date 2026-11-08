"""JSONL and hashing helpers shared by every stage.

Files are always written with LF line endings, also on Windows, so hashes and byte-for-byte
comparisons agree across machines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


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
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(stable_json(row) + "\n")
                n += 1
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return n


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Append and flush line by line. Resumable stages use this plus ``existing_keys``.

    A last line that a killed append cut short (it has no newline) is closed off first, so a
    resumed record never lands on it: it is completed with its newline if it parses as JSON (then
    ``existing_keys`` counted it as written) and truncated away otherwise (then ``existing_keys``
    ignored it, so its record is written again).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _close_cut_line(path)
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(stable_json(row) + "\n")
            f.flush()
            n += 1
    return n


def _close_cut_line(path: Path) -> None:
    """Complete or drop a last line without its newline (see ``append_jsonl``). A file that ends in
    a newline costs one 1-byte read. Only the tail is touched (one write or an in-place truncate),
    so a kill during the repair cannot lose a complete row."""
    try:
        f = open(path, "r+b")
    except FileNotFoundError:
        return
    with f:
        end = f.seek(0, os.SEEK_END)
        if end == 0:
            return
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return
        start = end
        while start > 0:                                 # back to the byte after the last newline
            step = min(start, 1 << 16)
            f.seek(start - step)
            newline = f.read(step).rfind(b"\n")
            start -= step
            if newline >= 0:
                start += newline + 1
                break
        f.seek(start)
        try:
            json.loads(f.read(end - start).decode("utf-8"))
            complete = True
        except ValueError:                               # JSONDecodeError and UnicodeDecodeError
            complete = False
        if complete:
            f.seek(end)
            f.write(b"\n")
        else:
            f.truncate(start)
    log.warning("%s: closed off a cut last line (%d bytes %s)", path, end - start,
                "completed" if complete else "dropped")


def existing_keys(path: str | Path, field: str = "key") -> set[str]:
    """Keys already written to a JSONL file. A line that is not a JSON object holding ``field`` is
    ignored, not fatal: a truncated last line (also one cut inside a multi-byte character), a blank
    line, a list, string, number or null, an object without ``field``, or one whose ``field`` holds a
    list or an object (no key is unhashable)."""
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[str] = set()
    with open(path, "rb") as f:
        for line in f:
            try:
                row = json.loads(line.decode("utf-8"))
            except ValueError:                           # JSONDecodeError and UnicodeDecodeError
                continue
            if isinstance(row, dict) and field in row and not isinstance(row[field], (list, dict)):
                keys.add(row[field])
    return keys
