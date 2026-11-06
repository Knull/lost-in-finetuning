"""``lift publish preflight``: may this git repo be made public? Also the text scan every public upload
goes through (``scan_folder``, called by ``lift.publish.hub.check_ready`` and the logs dataset build).

It fails if any tracked file, or any file in the git history (every blob reachable from any ref,
since making a repo public publishes its history too), holds:

- a secret or token (HF, GitHub, Anthropic, OpenAI-style, Google API and OAuth, AWS, Kaggle, a private
  key);
- a raw completion: a JSON line shaped like a GenerationRecord (key, variant, output) or an AuditKIT
  record that keeps outputs, or any 60-character window (taken every 30 characters, so any run of 90
  characters or more is caught) of an output that a judge or human labelled harmful in this machine's
  outputs/ (whitespace-normalised; JSON, JSONL, notebook and CSV files are also scanned as their decoded
  string values, so an escaped text is found too);
- a notebook committed with outputs or execution counts (a Colab run saved with its results);
- a path that must never be committed: outputs/, data/ outside data/splits/, validation sheets, labels
  and keys, model weights, .env, *.token, kaggle.json.

Every text blob is scanned, however large (a big one in chunks); only binary blobs are path-checked.
The text check needs this machine's outputs: ``load_windows`` refuses when there are no judgements here,
or no output labelled harmful whose text is here (``lift sync pull`` first), so the check is never a
silent no-op. GitHub publishes every branch, tag and pull-request ref of the remote, not only the local
ones: the CLI fetches them all first (``fetch_public_refs``), and the steps list what a person must still
read by hand (PR and issue text, Actions logs).

A finding names the file, line and kind, never the matched text. A line holding ``preflight: allow``
is skipped (for a documented fake in a test). When it passes it prints the steps to make the repo public.
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from lift import paths
from lift.publish import PublishError

ALLOW_MARK = "preflight: allow"
SECRETS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "Google OAuth token": re.compile(r"\bya29\.[0-9A-Za-z_-]{20,}"),
    "Google OAuth client secret": re.compile(r"\bGOCSPX-[0-9A-Za-z_-]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Kaggle API key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "Kaggle API key (KAGGLE_KEY)": re.compile(r"\bKAGGLE_KEY\b[\"']?\s*[:=]\s*[\"']?[0-9a-f]{32}\b"),
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
STRIDE = 30  # windows start every STRIDE characters, so any run of WINDOW + STRIDE characters holds one
CHUNK_CHARS = 8 << 20  # a longer text is scanned a chunk at a time (windows_in), never skipped
HARMFUL_TEXT = "text of an output labelled harmful"
NOTEBOOK_OUTPUTS = "notebook with outputs or execution counts"
JSON_SUFFIXES = (".json", ".jsonl", ".ipynb")


@dataclass(frozen=True)
class Finding:
    where: str  # "tree", "history <short sha>" or "staged"
    path: str
    line: int
    kind: str

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.kind}: {loc} ({self.where})"


def _norm(text: str) -> str:
    return " ".join(text.split())


def windows_of(texts: Iterable[str]) -> set[str]:
    """The WINDOW-character windows of each text (whitespace-normalised) starting every STRIDE characters,
    plus its last window: a copy of any WINDOW + STRIDE characters of a text holds one of them whole."""
    out: set[str] = set()
    for t in texts:
        n = _norm(t)
        if len(n) < WINDOW:
            continue
        out.update(n[s : s + WINDOW] for s in range(0, len(n) - WINDOW + 1, STRIDE))
        out.add(n[-WINDOW:])
    return out


def has_window(text: str, windows: set[str] | frozenset) -> bool:
    """Whether ``text`` (whitespace-normalised) holds one of ``windows`` anywhere."""
    return _in_flat(_norm(text), windows)


def _in_flat(flat: str, windows: set[str] | frozenset) -> bool:
    return bool(windows) and any(flat[s : s + WINDOW] in windows for s in range(0, len(flat) - WINDOW + 1))


def harmful_windows() -> set[str]:
    """The windows of every output labelled harmful in this machine's outputs/ (may be empty: the
    publish paths use ``load_windows``, which refuses an empty set)."""
    from lift.publish.datasets import harmful_record

    _, texts = harmful_record()
    return windows_of(texts)


def load_windows() -> tuple[set[str], int]:
    """(windows, number of harmful outputs they come from), or PublishError when this machine cannot say
    which texts are harmful: no judgements here, outputs labelled harmful whose generations are not
    here, or no window at all. A check against an empty set would pass anything."""
    from lift.publish.datasets import harmful_outputs, judgement_files

    pull = "run `lift sync pull` first (outputs/ and validation/ come only from the private runs repo)"
    files = judgement_files()
    if not files:
        raise PublishError(
            f"no harmful outputs found on this machine (no judgements in {paths.judgements_dir().name}/), so "
            f"the text check would be empty: {pull}"
        )
    keys, texts, missing = harmful_outputs()
    if missing:
        raise PublishError(
            f"{len(missing)} of {len(keys)} outputs labelled harmful have no generation on this machine, so "
            f"their text cannot be checked: {pull}"
        )
    windows = windows_of(texts)
    if not windows:
        raise PublishError(
            f"no harmful outputs found on this machine ({len(files)} judgement files, but no output labelled "
            f"harmful of {WINDOW} characters or more), so the text check would be empty: {pull}"
        )
    return windows, len(texts)


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


def _strings(value: object) -> Iterator[str]:
    """Every string value in a decoded JSON document (keys left out)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def decoded_text(text: str, path: str) -> str | None:
    """A JSON, JSONL or notebook file's string values (a CSV file's cells), one per line, so that an
    escaped text (\\n, \\uXXXX, doubled quotes) is scanned as it reads. None for other files, or a file
    that does not parse."""
    try:
        if path.endswith((".json", ".ipynb")):
            return "\n".join(_strings(json.loads(text)))
        if path.endswith(".jsonl"):
            out = []
            for line in text.splitlines():
                if line.strip():
                    try:
                        out.extend(_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        out.append(line)
            return "\n".join(out)
        if path.endswith(".csv"):
            return "\n".join(cell for row in csv.reader(io.StringIO(text)) for cell in row)
    except (json.JSONDecodeError, csv.Error, RecursionError):
        return None
    return None


def windows_in(text: str, windows: set[str] | frozenset) -> bool:
    """``has_window`` for a text of any size: one longer than CHUNK_CHARS is normalised and scanned a
    chunk at a time, each chunk led by the last WINDOW - 1 normalised characters before it, so a window
    across a cut is still found (the result equals ``has_window`` on the whole text)."""
    if len(text) <= CHUNK_CHARS:
        return has_window(text, windows)
    if not windows:
        return False
    carry, space_before = "", False
    for start in range(0, len(text), CHUNK_CHARS):
        chunk = text[start : start + CHUNK_CHARS]
        flat = _norm(chunk)
        if not flat:
            space_before = True
            continue
        sep = " " if carry and (space_before or chunk[0].isspace()) else ""
        joined = carry + sep + flat
        if _in_flat(joined, windows):
            return True
        carry, space_before = joined[-(WINDOW - 1) :], chunk[-1].isspace()
    return False


def notebook_has_outputs(text: str) -> bool:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return False
    cells = doc.get("cells") if isinstance(doc, dict) else None
    return any(
        isinstance(c, dict) and (c.get("outputs") or c.get("execution_count") is not None)
        for c in cells or []
    )


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
    if path.endswith(".ipynb") and notebook_has_outputs(text):
        found.append(Finding(where, path, 0, NOTEBOOK_OUTPUTS))
    if windows:
        decoded = decoded_text(text, path)
        for body in (text, decoded) if decoded is not None else (text,):
            if windows_in(body, windows):
                found.append(Finding(where, path, 0, HARMFUL_TEXT))
                break
    return found


def auditkit_problem(doc: object) -> str | None:
    """Why a public AuditKIT record is not redacted (None when it is): ``metadata.lift.text`` must be
    "redacted" and every ``actual_output`` a ``[redacted sha256:...]`` stand-in."""
    meta = doc.get("metadata") if isinstance(doc, dict) else None
    lift_meta = meta.get("lift") if isinstance(meta, dict) else None
    if not isinstance(lift_meta, dict) or lift_meta.get("text") != "redacted":
        return 'its metadata.lift.text is not "redacted"'

    def outputs(value: object) -> Iterator[object]:
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "actual_output":
                    yield v
                else:
                    yield from outputs(v)
        elif isinstance(value, list):
            for v in value:
                yield from outputs(v)

    kept = sum(1 for v in outputs(doc) if v is not None and not str(v).startswith("[redacted"))
    return f"{kept} actual_output values are not redacted" if kept else None


def _binary(head: bytes) -> bool:
    return b"\0" in head[:8192]


def scan_folder(folder: Path, windows: set[str], *, where: str = "staged") -> list[Finding]:
    """Scan every text file of a staged upload as the preflight scans a tracked file (secrets, raw
    completion records, notebooks with outputs, harmful-output windows), and every AuditKIT record
    (a .json under an ``auditkit`` folder) for redaction. Binary files (weights) are skipped."""
    found: list[Finding] = []
    for f in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        rel = f.relative_to(folder).as_posix()
        with open(f, "rb") as fh:
            if _binary(fh.read(8192)):
                continue
        text = f.read_bytes().decode("utf-8", "replace")
        found += scan_text(text, rel, where, windows)
        if "auditkit" in rel.split("/")[:-1] and rel.endswith(".json"):
            try:
                problem = auditkit_problem(json.loads(text))
            except json.JSONDecodeError:
                problem = "it is not valid JSON"
            if problem:
                found.append(Finding(where, rel, 0, f"AuditKIT record that is not redacted ({problem})"))
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
    """A blob's text, or None for a binary blob (only path-checked). Size never skips a text blob."""
    if _binary(body):
        return None
    return body.decode("utf-8", "replace")


FETCH_REFSPECS = ("+refs/heads/*:refs/remotes/{remote}/*", "+refs/pull/*/head:refs/remotes/pr/*")


def fetch_public_refs(repo: Path | None = None, remote: str = "origin") -> bool:
    """Fetch every branch, tag and pull-request head of ``remote`` (what GitHub publishes with the repo,
    deleted branches' PR refs included), so the history scan sees them. False when the repo has no such
    remote; PublishError when the fetch fails (git's own message is not shown: it can name a URL with
    credentials)."""
    repo = Path(repo or paths.REPO_ROOT)
    if remote not in _git("remote", cwd=repo).decode().split():
        return False
    specs = [spec.format(remote=remote) for spec in FETCH_REFSPECS]
    try:
        _git("fetch", "--quiet", "--tags", "--no-write-fetch-head", remote, *specs, cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise PublishError(
            f"could not fetch every branch, tag and pull-request ref from {remote} (git exit "
            f"{exc.returncode}); they go public with the repo, so they must be scanned: check the network "
            "and credentials and run again (--no-fetch scans local refs only: not enough to go public)"
        ) from None
    return True


def run(repo: Path | None = None, *, history: bool = True, windows: set[str] | None = None) -> list[Finding]:
    """Every finding in the tracked files (and the history of every ref). ``windows`` None loads them with
    ``load_windows``, which refuses a machine that cannot say which texts are harmful."""
    repo = Path(repo or paths.REPO_ROOT)
    windows = load_windows()[0] if windows is None else windows
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
  1. lift sync pull, then lift publish preflight   (it must pass on main, history included: it fetches every
     branch, tag and pull-request ref of origin first, and checks against this machine's harmful outputs)
  2. Check that the runs repo is still private: lift sync check (it refuses a runs repo that is not private)
  3. Check the frozen recipe tag is pushed: git tag --list "recipe-frozen-*"; git push --tags
  4. Read by hand what GitHub also publishes and this check cannot see: every PR and issue title, body and
     comment (gh pr list --state all; gh issue list --state all) and the Actions logs (gh run list). Delete
     any that quotes a completion (gh run delete for a run log). A finding in a refs/pull/* ref stays
     public after its branch is deleted: only GitHub Support can purge it, or publish a fresh repo.
  5. gh repo edit <owner>/<repo> --visibility public --accept-visibility-change-consequences
  6. Open the repo logged out and check the README, LICENSE and notebooks/acceptance_test.ipynb render.
"""
