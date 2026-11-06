"""``lift publish preflight``: may this git repo be made public?

It fails if any tracked file, or any file in the git history (every blob reachable from any ref,
since making a repo public publishes its history too), holds:

- a secret or token (HF, GitHub, Anthropic, OpenAI-style, Google API, AWS, Kaggle, a private key);
- a raw completion: a JSON line shaped like a GenerationRecord (key, variant, output) or an AuditKIT
  record that keeps outputs, or any 60-character run of an output that a judge or human labelled
  harmful in this machine's outputs/ (whitespace-normalised);
- a path that must never be committed: outputs/, data/ outside data/splits/, validation sheets, labels
  and keys, model weights, .env, *.token, kaggle.json.

A finding names the file, line and kind, never the matched text. A line holding ``preflight: allow``
is skipped (for a documented fake in a test). When it passes it prints the steps to make the repo public.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from lift import paths

ALLOW_MARK = "preflight: allow"
SECRETS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Kaggle API key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
}
FORBIDDEN_PATHS = (
    "outputs/*",
    "validation/sheets/*",
    "validation/labels/*",
    "validation/keys/*",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.gguf",
    ".env",
    "*/.env",
    "*.token",
    "kaggle.json",
    "*/kaggle.json",
)
WINDOW = 60
MAX_BYTES = 20 << 20  # larger blobs are only path-checked


@dataclass(frozen=True)
class Finding:
    where: str  # "tree" or "history <short sha>"
    path: str
    line: int
    kind: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.kind}: {loc} ({self.where})"


def _norm(text: str) -> str:
    return " ".join(text.split())


def harmful_windows() -> set[str]:
    """60-character windows (start and middle) of every output labelled harmful in this machine's outputs/."""
    from lift.publish.datasets import harmful_record

    _, texts = harmful_record()
    out = set()
    for t in texts:
        n = _norm(t)
        if len(n) >= WINDOW:
            out.add(n[:WINDOW])
            mid = (len(n) - WINDOW) // 2
            out.add(n[mid : mid + WINDOW])
    return out


def forbidden_path(path: str) -> bool:
    if path.startswith("data/") and not path.startswith("data/splits/"):
        return True
    return any(fnmatch(path, pat) for pat in FORBIDDEN_PATHS)


# An AuditKIT record line that keeps an output (a redacted record stores "[redacted sha256:...]").
_KEPT_OUTPUT = re.compile(r'"actual_output"\s*:\s*"(?!\[redacted)[^"]')


def _raw_record(line: str) -> bool:
    s = line.strip()
    if _KEPT_OUTPUT.search(s):
        return True
    if not (s.startswith("{") and '"output"' in s):
        return False
    try:
        row = json.loads(s)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(row, dict)
        and {"key", "variant", "output"} <= row.keys()
        and bool(str(row.get("output") or "").strip())
    )


def allowed_lines(text: str) -> set[str]:
    """The lines a file marks as documented fakes, without the mark, so the same line in an older
    commit (written before it was marked) is allowed too."""
    out = set()
    for line in text.splitlines():
        if ALLOW_MARK in line:
            out.add(line.split("#")[0].strip() if "#" in line else line.replace(ALLOW_MARK, "").strip())
    return out


def scan_text(
    text: str, path: str, where: str, windows: set[str], allowed: set[str] | frozenset = frozenset()
) -> list[Finding]:
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line or line.strip() in allowed:
            continue
        for kind, pat in SECRETS.items():
            if pat.search(line):
                found.append(Finding(where, path, i, kind))
        if _raw_record(line):
            found.append(Finding(where, path, i, "raw completion record"))
    if windows:
        flat = _norm(text)
        for start in range(0, max(0, len(flat) - WINDOW + 1)):
            if flat[start : start + WINDOW] in windows:
                found.append(Finding(where, path, 0, "text of an output labelled harmful"))
                break
    return found


def _git(*args: str, cwd: Path, input: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, input=input, capture_output=True, check=True).stdout


def tracked_files(repo: Path) -> list[str]:
    return [p for p in _git("ls-files", "-z", cwd=repo).decode("utf-8").split("\0") if p]


def _history_blobs(repo: Path) -> dict[str, str]:
    """{blob sha: one path it had} for every blob reachable from any ref."""
    out: dict[str, str] = {}
    for line in _git("rev-list", "--all", "--objects", cwd=repo).decode("utf-8", "replace").splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in out:
            out[sha] = path
    return out


def _cat_blobs(repo: Path, shas: Iterable[str]) -> Iterator[tuple[str, str, bytes | None]]:
    shas = list(shas)
    if not shas:
        return
    data = _git("cat-file", "--batch", cwd=repo, input=("\n".join(shas) + "\n").encode())
    pos = 0
    while pos < len(data):
        end = data.index(b"\n", pos)
        sha, kind, *rest = data[pos:end].decode().split()
        if kind == "missing":
            pos = end + 1
            continue
        size = int(rest[0])
        body = data[end + 1 : end + 1 + size]
        pos = end + 1 + size + 1
        yield sha, kind, (body if kind == "blob" else None)


def _decode(body: bytes) -> str | None:
    if b"\0" in body[:8192] or len(body) > MAX_BYTES:
        return None
    return body.decode("utf-8", "replace")


def run(repo: Path | None = None, *, history: bool = True, windows: set[str] | None = None) -> list[Finding]:
    repo = Path(repo or paths.REPO_ROOT)
    windows = harmful_windows() if windows is None else windows
    found: list[Finding] = []
    allowed: set[str] = set()
    for path in tracked_files(repo):
        if forbidden_path(path):
            found.append(Finding("tree", path, 0, "file that must never be committed"))
        f = repo / path
        if f.is_file():
            text = _decode(f.read_bytes())
            if text is not None:
                found += scan_text(text, path, "tree", windows)
                allowed |= allowed_lines(text)
    if history:
        seen = {(x.path, x.kind) for x in found}
        blobs = _history_blobs(repo)
        for sha, kind, body in _cat_blobs(repo, blobs):
            if kind != "blob" or body is None:
                continue
            path = blobs[sha]
            if forbidden_path(path) and (path, "file that must never be committed") not in seen:
                found.append(Finding(f"history {sha[:10]}", path, 0, "file that must never be committed"))
            text = _decode(body)
            if text is None:
                continue
            for x in scan_text(text, blobs[sha], f"history {sha[:10]}", windows, allowed):
                if (x.path, x.kind) not in seen:
                    found.append(x)
    return found


PUBLIC_STEPS = """\
Steps to make the GitHub repo public (a person runs them):
  1. lift publish preflight            (this check: it must pass on main, history included)
  2. Check that the runs repo is still private: lift sync check (it refuses a runs repo that is not private)
  3. Check the frozen recipe tag is pushed: git tag --list "recipe-frozen-*"; git push --tags
  4. gh repo edit <owner>/<repo> --visibility public --accept-visibility-change-consequences
  5. Open the repo logged out and check the README, LICENSE and notebooks/acceptance_test.ipynb render.
"""
