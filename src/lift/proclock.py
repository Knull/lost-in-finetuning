"""Machine-wide locks: one lift process at a time writes what a lock guards (decisions item 101).

On Nov 12 the laptop half of the core finals ran in two windows at once. Both processes appended the same
second-judge judgements to the same files (each file then holds two records per key, which every reader
refuses), sent three Message Batches twice and interleaved two appends to one call-cache line. The
frontier judge now holds the lock beside its call cache for as long as it lives (``lift.judge.frontier``):
a second process that builds a judge on the same cache is refused before it reads or writes anything, and
the refusal names the holder.

The lock is the OS's (``msvcrt.locking`` on Windows, ``flock`` elsewhere), so it goes with its process, a
crash included, and leaves nothing to clean up. Within one process it is re-entrant, counted by holder:
``lift final --judge-only`` and ``lift sweep`` run several lift commands in one process, each building its
own judge. The lock file holds a note (the command, its pid and start) for a refused process to read; lift
sync never carries *.lock files.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

LOCK_BYTE = 1 << 20  # the byte locked on Windows: past the note, which a refused process reads


class Busy(RuntimeError):
    """Another process holds the lock. ``holder`` names it: its command, pid and start, from its note."""

    def __init__(self, path: Path, holder: str) -> None:
        super().__init__(f"{path} is held by {holder}")
        self.path, self.holder = path, holder


_held: dict[str, list[int]] = {}       # resolved lock path -> [fd, holders in this process]
_guard = threading.Lock()


def _os_lock(fd: int) -> None:
    """Lock ``fd`` without waiting: an OSError when another open handle holds the lock."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, LOCK_BYTE, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _os_unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, LOCK_BYTE, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _key(path: str | Path) -> str:
    return str(Path(path).resolve())


def _command() -> str:
    """What this process runs, for the note: ``lift judge`` (the subcommand, never an option)."""
    argv = sys.argv or [""]
    if argv[0] in ("", "-c"):
        return "python"
    script = Path(argv[0])
    name = script.parent.name if script.stem == "__main__" else script.stem   # python -m lift: "lift"
    sub = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else ""
    return f"{name} {sub}".strip()


def _note(fd: int) -> str:
    """Who holds the lock, from the note its holder wrote."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        note = json.loads(os.read(fd, 4096).decode("utf-8"))
        return f"`{note['command']}` (pid {int(note['pid'])}, since {note['since']})"
    except (OSError, ValueError, KeyError, TypeError):
        return "another process (its note cannot be read)"


def acquire(path: str | Path) -> None:
    """Hold the lock at ``path`` for this process: once more when this process holds it already, else
    without waiting, Busy when another process holds it."""
    path, key = Path(path), _key(path)
    with _guard:
        entry = _held.get(key)
        if entry is not None:
            entry[1] += 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            _os_lock(fd)
        except OSError:
            holder = _note(fd)
            os.close(fd)
            raise Busy(path, holder) from None
        try:
            note = {"command": _command(), "pid": os.getpid(),
                    "since": datetime.now(UTC).isoformat(timespec="seconds")}
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(note).encode("utf-8"))
        except OSError:
            pass  # the lock holds without its note
        _held[key] = [fd, 1]


def release(path: str | Path) -> None:
    """Let go of one hold of ``path`` by this process; the last one unlocks it. Nothing when not held."""
    key = _key(path)
    with _guard:
        entry = _held.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] > 0:
            return
        del _held[key]
    fd = entry[0]
    try:
        os.ftruncate(fd, 0)
        _os_unlock(fd)
    except OSError:
        pass  # closing the file below lets go of it all the same
    finally:
        os.close(fd)


def held(path: str | Path) -> bool:
    """True while this process holds the lock at ``path``."""
    return _key(path) in _held
