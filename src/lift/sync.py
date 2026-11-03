"""Move data, outputs and validation files between machines through the PRIVATE HF dataset repo
named by ``LIFT_RUNS_REPO`` (configs/project.yaml ``storage.runs_repo_env``). Implemented in P02.

The repo holds raw harmful completions, unrepaired adapters and human labels, so every command
first checks ``dataset_info(repo).private is True`` and refuses otherwise (False, or unknown).

Synced trees, as canonical POSIX repo prefix -> local dir (from lift.paths at call time, so the
LIFT_DATA_DIR, LIFT_OUTPUT_DIR and LIFT_VALIDATION_DIR overrides apply): data/prompts, data/sft,
data/interim/translations, outputs, validation/sheets, validation/keys, validation/labels.

Every file is compared by content with the remote tree, listed once and pinned to one commit (git
blob sha1 for regular entries, sha256 for LFS/xet entries, as huggingface_hub does), and with the
state file ``outputs/.sync-state.json`` (path -> content hashes at the last successful push or
pull; never pushed), which says which side changed since:

- missing on one side: copied; the same on both sides: skipped;
- changed on one side only: that side wins. Pull updates a local copy nobody edited and keeps a
  local edit; push uploads a local edit and never puts an unchanged (stale) copy over a newer one;
- changed on both sides, or no state and different: append-only JSONL under outputs/generations,
  outputs/judgements and outputs/judge_cache is merged by "key" (``merge_jsonl_by_key``: local
  lines in order, then the remote lines whose key is new). Anything else is a CONFLICT, and so is a
  merge that would have to pick between two different records of one key or would break a rule
  the file's readers enforce: the local file is kept and reported; push also uploads the local
  copy under ``conflicts/<utc-timestamp>/<path>`` so it is not lost. ``force`` lets the remote copy
  win on pull (the replaced local copy is kept under ``outputs/.sync-backup/<utc-timestamp>/<path>``,
  never pushed) and the local copy win on push (the Hub keeps the old one in its history).

Push sends exactly the files it planned (new, updated, merged, forced) and the conflict copies,
with ``create_commit``. Never upload_folder: it re-sends files nobody changed and guards only the
first of its commits, so a later one can revert another machine's push. Every commit carries
``parent_commit``: the listed commit, then each part's own commit when a push is split
(COMMIT_FILES, COMMIT_BYTES), and the tree at that commit is listed again before the next part. A
commit landing meanwhile (HTTP 412, or a tree that is not the plan) makes the push plan again.

Nothing is deleted on either side. Downloads go to a staging dir outside the synced trees, are
checked against the tree hash (hf_hub_download silently returns stale local files when the Hub is
unreachable) and are moved into place atomically. Logs carry counts and paths, never contents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from lift import config, paths
from lift.io import stable_json, write_jsonl

log = logging.getLogger(__name__)

REPO_TYPE = "dataset"
STATE_FILE = ".sync-state.json"     # in paths.output_dir(); never pushed
STATE_VERSION = 1
BACKUP_DIR = ".sync-backup"         # in paths.output_dir(): local copies a forced pull replaced; never pushed
CONFLICTS_DIR = "conflicts"
PUSH_ATTEMPTS = 3
DOWNLOAD_WORKERS = 8
# One commit holds at most this many files and bytes (a bigger file goes alone), so the inline base64
# payload of regular files stays well under the Hub's 1 GB commit limit.
COMMIT_FILES = 200
COMMIT_BYTES = 512 << 20
# Append-only JSONL whose records carry a unique "key", merged by key when both sides changed. The
# kind says how strictly: generations and judgements feed every reported number, judge_cache is a
# call cache.
MERGE_KINDS = {"outputs/generations/": "generations", "outputs/judgements/": "judgements",
               "outputs/judge_cache/": "judge_cache"}
# How a GenerationRecord was run rather than what it holds: two records of one key that differ only
# here are the same record (wall time, batching, GPU and library versions vary; the key pins the rest).
RUN_FIELDS = ("seconds",)
RUN_META = ("gpu", "device", "batch_size", "n_generated", "torch", "transformers")
# Never synced. Matched like huggingface_hub matches them (fnmatchcase, `*` crosses `/`) on the path
# relative to the synced folder. CommitOperationAdd refuses a `.cache` or `.git` component, and
# hf_hub_download leaves `.cache/huggingface` metadata behind.
IGNORE_PATTERNS = (
    ".cache", ".cache/*", "*/.cache", "**/.cache/**", ".git", ".git/*", "*/.git", "**/.git/**",
    "__pycache__/*", "**/__pycache__/**", "*.lock", "*.tmp", "*.incomplete", "*.pyc",
    ".DS_Store", "**/.DS_Store", STATE_FILE, BACKUP_DIR, f"{BACKUP_DIR}/*",
)

# What happened to a file (for status: what would happen).
NEW, UPDATE, SAME, MERGE, FORCE, CONFLICT = "new", "update", "same", "merge", "force", "conflict"
KEEP = "keep-local"   # pull: edited here, unchanged on the Hub; kept until pushed
STALE = "stale"       # push: newer on the Hub, unchanged here; not uploaded
ACTIONS = (NEW, UPDATE, MERGE, FORCE, SAME, KEEP, STALE, CONFLICT)
_SENT = (NEW, UPDATE, MERGE, FORCE)   # push: these go up
_SHA = re.compile(r"[0-9a-f]{40}")


class SyncError(RuntimeError):
    """Sync cannot or must not go ahead. The message says what to do next; it never holds a token."""


class _RemoteMoved(Exception):
    """The runs repo got a commit this push did not plan against (HTTP 412 on parent_commit, or a
    tree between two parts that is not the plan)."""


# ---------------------------------------------------------------------------- content


@dataclass(frozen=True)
class Digest:
    """A file's content as both hashes the Hub uses: sha256 (LFS/xet entries), git blob sha1."""

    sha256: str
    git: str

    def matches(self, entry: Any) -> bool:
        algo, value = remote_hash(entry)
        return (self.sha256 if algo == "sha256" else self.git) == value


def remote_hash(entry: Any) -> tuple[str, str]:
    """``("sha256", hex)`` for an LFS/xet tree entry, else ``("git", blob sha1)``: the rule
    huggingface_hub's own checksum verification uses."""
    lfs = getattr(entry, "lfs", None)
    sha256 = getattr(lfs, "sha256", None)
    if sha256 is None and isinstance(lfs, dict):
        sha256 = lfs.get("sha256")
    if sha256:
        return "sha256", str(sha256).lower()
    return "git", str(entry.blob_id).lower()


def file_digest(path: Path, chunk: int = 1 << 20) -> Digest:
    """Both hashes in one pass; the same values as huggingface_hub.utils.sha (sha_fileobj, git_hash)."""
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git = hashlib.sha1(b"blob %d\0" % size, usedforsecurity=False)
    seen = 0
    with open(path, "rb") as f:
        while block := f.read(chunk):
            sha256.update(block)
            git.update(block)
            seen += len(block)
    if seen != size:
        raise SyncError(f"{path} changed while it was being hashed: stop whatever writes it, then rerun")
    return Digest(sha256.hexdigest(), git.hexdigest())


# ---------------------------------------------------------------------------- merging JSONL


def merge_kind(path: str) -> str | None:
    """"generations", "judgements" or "judge_cache" for an append-only JSONL merged by key, else None."""
    if not path.endswith(".jsonl"):
        return None
    return next((kind for prefix, kind in MERGE_KINDS.items() if path.startswith(prefix)), None)


def is_mergeable(path: str) -> bool:
    return merge_kind(path) is not None


@dataclass(frozen=True)
class Merged:
    rows: list[dict]
    added: int      # remote records whose key was new
    dropped: int    # judge_cache lines left out (not a JSON object with a string "key"), both sides


def merge_jsonl_by_key(local: Path, remote: Path, *, kind: str) -> Merged:
    """Local records in order, then the remote records whose "key" is new.

    Generations and judgements: every line must be a JSON object with a string "key", a key on both
    sides must hold the same record (for generations, RUN_FIELDS and RUN_META aside) and the result
    must keep the rules its readers enforce (``check_merged``); otherwise ValueError, and the caller
    reports a conflict instead of picking a side. judge_cache, a call cache, is read like its reader
    reads it: a line that is not a JSON object with a string "key" is dropped (and counted), and a
    key on both sides keeps the local row.
    """
    lenient = kind == "judge_cache"
    local_rows, dropped_here = _keyed_rows(local, "the local copy", lenient=lenient)
    remote_rows, dropped_there = _keyed_rows(remote, "the Hub's copy", lenient=lenient)
    seen: dict[str, dict] = {}
    for row in local_rows:
        seen.setdefault(row["key"], row)
    added = []
    for row in remote_rows:
        mine = seen.get(row["key"])
        if mine is None:
            seen[row["key"]] = row
            added.append(row)
        elif not lenient and _comparable(kind, mine) != _comparable(kind, row):
            raise ValueError(f"{local.name}: key {row['key'][:12]} holds different records here and on "
                             "the Hub")
    rows = local_rows + added
    if not lenient:
        check_merged(kind, rows, local.name)
    return Merged(rows, len(added), dropped_here + dropped_there)


def check_merged(kind: str, rows: Sequence[dict], name: str) -> None:
    """ValueError when ``rows`` break a rule the readers enforce on a whole file (generate's
    read_generations and resume, the judge pipeline's load_judgements), as a merge of two valid
    files can: one record per key and per prompt ((set, lang, pid) for generations, pid for
    judgements), one variant/set/lang/split cell, and one meta.run_config and meta.dtype
    (generations; a record without one counts as a value of its own) or one judge id (judgements)."""
    per_key: Counter = Counter()
    per_prompt: Counter = Counter()
    single: dict[str, set[str]] = {}
    for row in rows:
        meta = row["meta"] if isinstance(row.get("meta"), dict) else {}
        one: dict[str, Any] = {"variant/set/lang/split": [row.get(f) for f in ("variant", "set", "lang",
                                                                                "split")]}
        if kind == "generations":
            prompt: Any = [row.get("set"), row.get("lang"), row.get("pid")]
            one.update({"meta.run_config": meta.get("run_config"), "meta.dtype": meta.get("dtype")})
        else:
            prompt = row.get("pid")
            one["judge"] = row.get("judge")
        per_key[row["key"]] += 1
        per_prompt[stable_json(prompt)] += 1
        for what, value in one.items():
            single.setdefault(what, set()).add(stable_json(value))
    problems = [f"{n} {what}(s) with more than one record"
                for what, counts in (("key", per_key), ("prompt", per_prompt))
                if (n := sum(c > 1 for c in counts.values()))]
    problems += [f"{len(values)} different {what} values"
                 for what, values in single.items() if len(values) > 1]
    if problems:
        raise ValueError(f"{name}: merged, it would break its readers' rules ({'; '.join(problems)})")


def _comparable(kind: str, row: dict) -> str:
    """``row`` as canonical JSON (so a NaN equals itself), for generations without what says how
    the record was run (RUN_FIELDS, RUN_META)."""
    if kind == "generations":
        row = {k: v for k, v in row.items() if k not in RUN_FIELDS}
        if isinstance(row.get("meta"), dict):
            row["meta"] = {k: v for k, v in row["meta"].items() if k not in RUN_META}
    return stable_json(row)


def _keyed_rows(path: Path, side: str, *, lenient: bool) -> tuple[list[dict], int]:
    """The records of a JSONL file and how many lines were dropped: ``lenient`` drops a line that is
    not a JSON object with a string "key", otherwise such a line raises ValueError."""
    rows, dropped = [], 0
    with open(path, "rb") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:   # also a line cut in the middle of a UTF-8 sequence
                row = None
            if isinstance(row, dict) and isinstance(row.get("key"), str):
                rows.append(row)
            elif lenient:
                dropped += 1
            else:
                raise ValueError(f"{path.name}, {side}: line {i} is not a JSON object with a string 'key'")
    return rows, dropped


# ---------------------------------------------------------------------------- what is synced


def targets() -> dict[str, Path]:
    """Canonical POSIX repo prefix -> local dir, resolved now so the LIFT_*_DIR overrides apply."""
    validation = paths.validation_dir()
    return {
        "data/prompts": paths.prompts_dir(),
        "data/sft": paths.sft_dir(),
        "data/interim/translations": paths.interim_dir() / "translations",
        "outputs": paths.output_dir(),
        "validation/sheets": validation / "sheets",
        "validation/keys": validation / "keys",
        "validation/labels": validation / "labels",
    }


def select(subdirs: Iterable[str] = ()) -> list[tuple[str, Path]]:
    """(repo prefix, local dir) pairs to sync, all of them when ``subdirs`` is empty.

    Each entry is a canonical prefix, a folder under one (``outputs/generations``; backslashes are
    accepted) or a parent of some (``data``, ``validation``). ``..``, anything outside the synced
    trees and anything never synced (IGNORE_PATTERNS: caches, the sync state, outputs/.sync-backup)
    are refused. A folder inside another selected one adds nothing.
    """
    table = targets()
    chosen: dict[str, Path] = {}
    for raw in subdirs:
        norm = str(raw).replace("\\", "/").strip().strip("/")
        parts = norm.split("/")
        if not norm or any(part in ("", ".", "..") for part in parts):
            raise SyncError(f"bad sync path {raw!r}: give a repo path such as outputs/generations")
        base = max((b for b in table if norm == b or norm.startswith(b + "/")), key=len, default=None)
        if base is not None:
            if norm != base and _ignored(norm[len(base) + 1:]):
                raise SyncError(f"{raw!r} is never synced: caches, temp files, the sync state and "
                                f"{BACKUP_DIR} stay on this machine")
            chosen[norm] = table[base].joinpath(*parts[base.count("/") + 1:])
            continue
        below = [b for b in table if b.startswith(norm + "/")]
        if not below:
            raise SyncError(f"{raw!r} is not synced; choose from {', '.join(table)} or a folder under one")
        chosen.update((b, table[b]) for b in below)
    if not chosen:
        chosen = table
    nested = {p for p in chosen for q in chosen if p != q and p.startswith(q + "/")}
    return sorted((p, d) for p, d in chosen.items() if p not in nested)


def _ignored(rel: str) -> bool:
    return any(fnmatchcase(rel, pattern) for pattern in IGNORE_PATTERNS)


def _local_files(folder: Path) -> list[tuple[str, Path]]:
    """(path relative to ``folder``, file) for every file that would be pushed, sorted."""
    if not folder.is_dir():
        return []
    found = ((p.relative_to(folder).as_posix(), p) for p in folder.rglob("*") if p.is_file())
    return sorted((rel, p) for rel, p in found if not _ignored(rel))


def _local_path(folder: Path, path: str, rel: str) -> Path:
    """Where a remote file lands; refuses paths that would leave ``folder``."""
    parts = rel.split("/")
    dest = folder.joinpath(*parts)
    if any(part in ("", ".", "..") or "\\" in part for part in parts) or not dest.is_relative_to(folder):
        raise SyncError(f"remote path {path!r} is not a safe relative path; remove it from the runs repo")
    return dest


def _owner(path: str, prefixes: Sequence[str]) -> str:
    return next(p for p in prefixes if path.startswith(p + "/"))


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------- the Hub


def runs_repo() -> str:
    """The runs repo id, from the env var named in configs/project.yaml (storage.runs_repo_env)."""
    var = config.get("storage.runs_repo_env")
    repo = os.environ.get(var, "").strip()
    if not repo:
        raise SyncError(f"{var} is not set: export {var}=<hf-user>/lift-runs, a PRIVATE Hugging Face "
                        "dataset repo (hf repos create lift-runs --repo-type dataset --private)")
    owner, _, name = repo.partition("/")
    if not owner or not name or "/" in name:
        raise SyncError(f"{var}={repo!r} is not a repo id of the form <owner>/<name>")
    return repo


def _require_token(api: Any = None) -> None:
    from huggingface_hub import get_token

    if getattr(api, "token", None) or get_token() is not None:
        return
    raise SyncError("no Hugging Face token: set HF_TOKEN (a WRITE token for the runs repo to push) "
                    "or run `hf auth login`")


def make_api() -> Any:
    """An HfApi using the configured token. Refuses when there is none (its value is never shown)."""
    from huggingface_hub import HfApi

    _require_token()
    return HfApi()


def _status_code(error: BaseException) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


@contextmanager
def _hub_errors(what: str, repo: str) -> Iterator[None]:
    """Turn huggingface_hub errors into a SyncError that says what to do (never with the token)."""
    from huggingface_hub.errors import (
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )

    try:
        yield
    except RepositoryNotFoundError as e:  # also GatedRepoError; 401 without a token, 404 with one
        raise SyncError(f"{what}: runs repo {repo!r} was not found or is not readable with this token "
                        f"(HTTP {_status_code(e)}). Create it as a PRIVATE dataset repo and use a token "
                        "that can read it (and write it, to push)") from e
    except HfHubHTTPError as e:  # an invalid token is a plain 401 here, not RepositoryNotFoundError
        code = _status_code(e)
        if code == 412:
            raise _RemoteMoved(what) from e
        if code == 401:
            raise SyncError(f"{what}: the Hugging Face token was rejected (HTTP 401): check HF_TOKEN") from e
        if code == 403:
            raise SyncError(f"{what}: the token may not do this (HTTP 403); pushing needs a WRITE token "
                            "for the runs repo") from e
        raise SyncError(f"{what}: the Hub answered HTTP {code}; rerun (sync compares content, so it "
                        "resumes)") from e
    except OfflineModeIsEnabled as e:
        raise SyncError(f"{what}: huggingface_hub is in offline mode (HF_HUB_OFFLINE is set)") from e
    except LocalEntryNotFoundError as e:  # what hf_hub_download raises when the Hub cannot be reached
        raise SyncError(f"{what}: could not reach the Hub; check the connection and rerun") from e
    except HFValidationError as e:
        raise SyncError(f"{what}: {repo!r} is not a valid Hugging Face repo id") from e


def require_private(api: Any, repo: str) -> Any:
    """The repo's DatasetInfo, or SyncError unless the Hub confirms that the repo is private."""
    with _hub_errors("checking the runs repo", repo):
        info = api.dataset_info(repo, expand=["private", "sha"])
    if info.private is not True:  # False is public; None means the Hub did not say
        raise SyncError(f"refusing to sync: dataset repo {repo!r} is not private (private={info.private!r}). "
                        "It holds raw harmful completions, unrepaired adapters and human labels: make it "
                        f"private or point {config.get('storage.runs_repo_env')} at a private repo")
    if not isinstance(info.sha, str) or not _SHA.fullmatch(info.sha):
        raise SyncError(f"the Hub returned no commit sha for {repo!r}")
    return info


@dataclass(frozen=True)
class _Remote:
    api: Any
    repo: str
    sha: str   # the commit every read of this run is pinned to


def _connect(api: Any) -> _Remote:
    from huggingface_hub import constants
    from huggingface_hub.utils import disable_progress_bars

    repo = runs_repo()
    if getattr(constants, "HF_DEBUG", False):
        raise SyncError("HF_DEBUG is set: huggingface_hub would log request bodies (file contents) into "
                        "this log. Unset it to sync")
    if api is None:
        api = make_api()
    else:
        _require_token(api)
    disable_progress_bars()
    return _Remote(api, repo, require_private(api, repo).sha)


def _remote_tree(remote: _Remote, revision: str | None = None) -> dict[str, Any]:
    """Every file of the repo at one commit (the pinned one by default; folders dropped), from a
    single listing."""
    with _hub_errors("listing the runs repo", remote.repo):
        entries = list(remote.api.list_repo_tree(remote.repo, recursive=True, revision=revision or remote.sha,
                                                 repo_type=REPO_TYPE))
    return {e.path: e for e in entries if getattr(e, "blob_id", None) is not None}


@contextmanager
def _staging() -> Iterator[Path]:
    """Scratch space outside every synced tree, so hf_hub_download's .cache never lands there."""
    with tempfile.TemporaryDirectory(prefix="lift-sync-", ignore_cleanup_errors=True) as tmp:
        yield Path(tmp)


def _download(remote: _Remote, item: _Item, staging: Path) -> tuple[Path, Digest]:
    """Fetch one file at the pinned commit into ``staging`` and check it against its tree entry."""
    with _hub_errors(f"downloading {item.path}", remote.repo):
        remote.api.hf_hub_download(repo_id=remote.repo, filename=item.path, repo_type=REPO_TYPE,
                                   revision=remote.sha, local_dir=str(staging))
    staged = staging.joinpath(*item.path.split("/"))
    if not staged.is_file():
        raise SyncError(f"downloading {item.path} left no file; rerun")
    digest = file_digest(staged)
    if not digest.matches(item.entry):
        algo, want = remote_hash(item.entry)
        raise SyncError(f"downloaded {item.path} does not match the runs repo at {remote.sha[:10]} "
                        f"({algo} {want[:12]}): the Hub may be unreachable (huggingface_hub then returns "
                        "stale files). The local copy was left alone; rerun")
    return staged, digest


def _install(staged: Path, dest: Path) -> None:
    """Move a verified file into place atomically: a rename, or copy + rename across filesystems."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staged, dest)
        return
    except OSError:
        pass  # staging on another filesystem (e.g. /tmp vs /kaggle/working); a real error recurs below
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name, suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(staged, tmp)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    staged.unlink(missing_ok=True)


def _run_parallel(work: Callable[[_Item], Any], items: list[_Item],
                  done: Callable[[_Item, Any], None]) -> None:
    """``work`` on DOWNLOAD_WORKERS threads, ``done`` in this thread; stops at the first error."""
    if not items:
        return
    pool = ThreadPoolExecutor(max_workers=min(DOWNLOAD_WORKERS, len(items)), thread_name_prefix="lift-sync")
    try:
        futures = {pool.submit(work, item): item for item in items}
        for future in as_completed(futures):
            done(futures[future], future.result())
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------- state and plans


class _State:
    """outputs/.sync-state.json: canonical path -> Digest at the last successful push or pull."""

    def __init__(self, path: Path, repo: str, files: dict[str, Digest]):
        self.path, self.repo, self.files, self.dirty = path, repo, files, False

    @classmethod
    def load(cls, repo: str) -> _State:
        path = paths.output_dir() / STATE_FILE
        if not path.is_file():
            return cls(path, repo, {})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            files = {p: Digest(sha256=d["sha256"], git=d["git"]) for p, d in data["files"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise SyncError(f"cannot read the sync state {path} ({type(e).__name__}). Delete it to start "
                            "over: every file that differs from the Hub then counts as changed on both "
                            "sides") from e
        if data.get("version") != STATE_VERSION or data.get("repo") != repo:
            log.warning("sync state %s belongs to %s (version %s), not %s: ignoring it",
                        path, data.get("repo"), data.get("version"), repo)
            return cls(path, repo, {})
        return cls(path, repo, files)

    def get(self, path: str) -> Digest | None:
        return self.files.get(path)

    def set(self, path: str, digest: Digest) -> None:
        if self.files.get(path) != digest:
            self.files[path] = digest
            self.dirty = True

    def save(self) -> None:
        """Atomic write (temp file + rename), LF, sorted: a killed sync never leaves half a state."""
        if not self.dirty:
            return
        payload = {"version": STATE_VERSION, "repo": self.repo,
                   "files": {p: {"sha256": d.sha256, "git": d.git} for p, d in sorted(self.files.items())}}
        paths.ensure(self.path.parent)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=STATE_FILE, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.dirty = False


@dataclass
class _Item:
    path: str                  # canonical repo path
    prefix: str                # the selected prefix it falls under
    local: Path
    entry: Any | None          # remote tree entry; None when the file is only local
    digest: Digest | None      # local content; None when the file is only remote
    action: str = ""
    backup: Path | None = None   # pull --force: where the replaced local copy was kept


def _decide(item: _Item, prev: Digest | None, *, pull: bool, force: bool) -> str:
    """The per-file rule of the module docstring: compare with the Hub, then with the last sync."""
    if item.digest is None or item.entry is None:
        return NEW
    if item.digest.matches(item.entry):
        return SAME
    local_changed = prev is None or prev.sha256 != item.digest.sha256
    remote_changed = prev is None or not prev.matches(item.entry)
    if not local_changed:
        return UPDATE if pull else STALE
    if not remote_changed:
        return KEEP if pull else UPDATE
    if is_mergeable(item.path):
        return MERGE
    return FORCE if force else CONFLICT


def _digest_memo() -> Callable[[Path], Digest]:
    memo: dict[Path, Digest] = {}

    def digest(path: Path) -> Digest:
        if path not in memo:
            memo[path] = file_digest(path)
        return memo[path]

    return digest


def _plan_pull(selected: list[tuple[str, Path]], tree: dict[str, Any], state: _State, *, force: bool,
               digest: Callable[[Path], Digest]) -> list[_Item]:
    items = []
    for prefix, folder in selected:
        for path in sorted(p for p in tree if p.startswith(prefix + "/")):
            rel = path[len(prefix) + 1:]
            if _ignored(rel):
                continue
            local = _local_path(folder, path, rel)
            item = _Item(path, prefix, local, tree[path], digest(local) if local.is_file() else None)
            item.action = _decide(item, state.get(path), pull=True, force=force)
            items.append(item)
    return items


def _plan_push(selected: list[tuple[str, Path]], tree: dict[str, Any], state: _State, *, force: bool,
               digest: Callable[[Path], Digest]) -> list[_Item]:
    items = []
    for prefix, folder in selected:
        for rel, local in _local_files(folder):
            path = f"{prefix}/{rel}"
            item = _Item(path, prefix, local, tree.get(path), digest(local))
            item.action = _decide(item, state.get(path), pull=False, force=force)
            items.append(item)
    return items


def _apply_merge(item: _Item, merged: Merged, direction: str) -> bool:
    """Write a merge into the local file when it changes it; True if it did."""
    if merged.dropped:
        log.warning("sync %s: dropped %d unreadable line(s) of %s (lines its reader skips too)",
                    direction, merged.dropped, item.path)
    log.info("sync %s: merged %d remote record(s) into %s", direction, merged.added, item.path)
    if not (merged.added or merged.dropped):
        return False
    write_jsonl(item.local, merged.rows)
    return True


# ---------------------------------------------------------------------------- reports


@dataclass
class SyncReport:
    """What a pull or push did (``dry_run``: would do), per file, keyed by canonical repo path."""

    direction: str                      # "pull" | "push"
    repo: str
    revision: str                       # the remote commit the comparison was made against
    prefixes: list[str]
    files: dict[str, str] = field(default_factory=dict)            # path -> one of ACTIONS
    dry_run: bool = False
    commits: list[str] = field(default_factory=list)               # push: commits made, in order
    conflict_copies: dict[str, str] = field(default_factory=dict)  # push: path -> copy under conflicts/
    backups: dict[str, str] = field(default_factory=dict)          # pull --force: path -> replaced local copy
    backup_dir: str = ""                                           # pull --force: the folder holding them

    def paths(self, action: str) -> list[str]:
        return sorted(p for p, a in self.files.items() if a == action)

    @property
    def conflicts(self) -> list[str]:
        return self.paths(CONFLICT)

    @property
    def ok(self) -> bool:
        return not self.conflicts

    def counts(self) -> dict[str, Counter]:
        """prefix -> action -> number of files."""
        out: dict[str, Counter] = {p: Counter() for p in self.prefixes}
        for path, action in self.files.items():
            out[_owner(path, self.prefixes)][action] += 1
        return out

    def summary(self) -> list[str]:
        """Lines for the CLI and job logs: counts and paths, never contents."""
        head = f"{self.direction}{' (dry run)' if self.dry_run else ''} {self.repo}@{self.revision[:10]}"
        if self.commits:
            head += f" -> {self.commits[-1][:10]}"
        lines = [head]
        for prefix, counts in self.counts().items():
            if counts:
                lines.append(f"  {prefix}: " + ", ".join(f"{counts[a]} {a}" for a in ACTIONS if counts[a]))
        for path in self.conflicts:
            copy = self.conflict_copies.get(path)
            lines.append(f"  CONFLICT {path}" + (f" (local copy on the Hub at {copy})" if copy else ""))
        would = "would be " if self.dry_run else ""
        if kept := self.paths(KEEP):
            lines.append(f"  {len(kept)} local edit(s) {would}kept, not on the Hub yet: run lift sync push")
        if stale := self.paths(STALE):
            lines.append(f"  {len(stale)} file(s) newer on the Hub and unchanged here {would}left out: "
                         "run lift sync pull")
        if self.backups:
            lines.append(f"  {len(self.backups)} local copy(ies) replaced by the Hub's are kept under "
                         f"{self.backup_dir}")
        if self.conflicts:
            fix = (f"lift sync pull --force takes the Hub's copy (the local one is kept under "
                   f"outputs/{BACKUP_DIR})" if self.direction == "pull"
                   else "lift sync push --force uploads the local copy over the Hub's")
            lines.append(f"  {len(self.conflicts)} conflict(s): changed here and on the Hub since the last "
                         f"sync; the local copy {would or 'was '}kept. Resolve by hand, or {fix}")
        return lines


def _log(report: SyncReport) -> None:
    totals = Counter(report.files.values())
    log.info("sync %s %s@%s: %s", report.direction, report.repo, report.revision[:10],
             ", ".join(f"{totals[a]} {a}" for a in ACTIONS if totals[a]) or "nothing to do")


# ---------------------------------------------------------------------------- pull


def pull(subdirs: Sequence[str] = (), *, force: bool = False, api: Any = None) -> SyncReport:
    """Download new and changed files of the runs repo, pinned to its current commit, without ever
    overwriting a local edit (rules in the module docstring). Conflicts leave ``report.ok`` False;
    ``force`` lets the remote copy win them instead, keeping each replaced local copy under
    outputs/.sync-backup/<utc-timestamp>/<path> (``report.backups``)."""
    selected = select(subdirs)
    remote = _connect(api)
    tree = _remote_tree(remote)
    state = _State.load(remote.repo)
    items = _plan_pull(selected, tree, state, force=force, digest=_digest_memo())
    report = SyncReport("pull", remote.repo, remote.sha, [p for p, _ in selected])
    backups = _unused_dir(paths.output_dir() / BACKUP_DIR, _stamp())
    try:
        for item in items:
            if item.action == SAME:
                state.set(item.path, item.digest)
        fetch = [i for i in items if i.action in (NEW, UPDATE, FORCE, MERGE)]
        with _staging() as staging:
            _run_parallel(lambda item: _fetch(remote, item, staging, backups), fetch,
                          lambda item, result: _pulled(item, result, state, force=force, backups=backups))
    finally:
        state.save()   # whatever finished is recorded, also when a download fails
    report.files = {i.path: i.action for i in items}
    report.backups = {i.path: str(i.backup) for i in items if i.backup is not None}
    report.backup_dir = str(backups) if report.backups else ""
    _log(report)
    return report


def _fetch(remote: _Remote, item: _Item, staging: Path, backups: Path) -> tuple[Path, Digest]:
    staged, digest = _download(remote, item, staging)
    if item.action == FORCE:
        _backup(item, backups)
    if item.action != MERGE:
        _install(staged, item.local)
    return staged, digest


def _unused_dir(parent: Path, name: str) -> Path:
    """``parent/name``, or ``parent/name-<n>`` when that exists: a backup never replaces another."""
    path, n = parent / name, 0
    while path.exists():
        n += 1
        path = parent / f"{name}-{n}"
    return path


def _backup(item: _Item, root: Path) -> None:
    """Copy the local file a forced pull is about to replace to ``root/<path>``."""
    dest = root.joinpath(*item.path.split("/"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(item.local, dest)
    item.backup = dest
    log.warning("sync pull --force: the local copy of %s is kept at %s", item.path, dest)


def _pulled(item: _Item, result: tuple[Path, Digest], state: _State, *, force: bool, backups: Path) -> None:
    """Record a verified download (already in place), or merge it into the local JSONL first."""
    staged, digest = result
    if item.action == MERGE:
        try:
            merged = merge_jsonl_by_key(item.local, staged, kind=merge_kind(item.path))
        except ValueError as e:
            log.warning("sync pull: cannot merge %s by key (%s)", item.path, e)
            if not force:
                item.action = CONFLICT
                return
            _backup(item, backups)
            _install(staged, item.local)
            item.action = FORCE
        else:
            _apply_merge(item, merged, "pull")
    # The remote version is now in (or merged into) the local file. After a merge the local file
    # still differs from it, so its extra lines count as a local change and the next push sends them.
    state.set(item.path, digest)


# ---------------------------------------------------------------------------- push


@dataclass(frozen=True)
class _Upload:
    path: str              # where it goes in the runs repo
    local: Path
    digest: Digest
    item: _Item | None     # None for a conflict copy: the conflicting file itself stays as it is


def push(subdirs: Sequence[str] = (), *, force: bool = False, api: Any = None) -> SyncReport:
    """Upload new and changed files to the runs repo in commits guarded by parent_commit (rules in
    the module docstring). A file changed on both sides is merged (append-only JSONL) or, without
    ``force``, left alone on the Hub and uploaded under conflicts/<utc-timestamp>/<path>; conflicts
    leave ``report.ok`` False. If another machine commits meanwhile, the push is planned again."""
    selected = select(subdirs)
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        try:
            return _push_once(selected, force=force, api=api)
        except _RemoteMoved as moved:
            log.warning("sync push: the runs repo moved on while %s (another machine pushed); "
                        "planning again (%d/%d)", moved, attempt, PUSH_ATTEMPTS)
    raise SyncError(f"the runs repo kept changing during push ({PUSH_ATTEMPTS} attempts): another machine "
                    "is pushing. Rerun lift sync push")


def _push_once(selected: list[tuple[str, Path]], *, force: bool, api: Any) -> SyncReport:
    remote = _connect(api)
    tree = _remote_tree(remote)
    state = _State.load(remote.repo)
    items = _plan_push(selected, tree, state, force=force, digest=_digest_memo())
    report = SyncReport("push", remote.repo, remote.sha, [p for p, _ in selected])
    try:
        for item in items:
            if item.action == SAME:
                state.set(item.path, item.digest)
        with _staging() as staging:
            for item in items:
                if item.action == MERGE:
                    _merge_for_push(remote, item, staging, force=force)
        stamp = _stamp()
        report.conflict_copies, copies = _conflict_copies([i for i in items if i.action == CONFLICT], tree,
                                                          stamp)
        uploads = [_Upload(i.path, i.local, i.digest, i) for i in items if i.action in _SENT]
        _commit(remote, tree, sorted(uploads + copies, key=lambda u: u.path), state, report)
        if copies:
            log.warning("sync push: %d conflicting local file(s) uploaded under %s/%s",
                        len(copies), CONFLICTS_DIR, stamp)
    finally:
        state.save()   # parts already committed are recorded, also when a later one fails
    report.files = {i.path: i.action for i in items}
    _log(report)
    return report


def _merge_for_push(remote: _Remote, item: _Item, staging: Path, *, force: bool) -> None:
    """Fold the remote records of a JSONL changed on both sides into the local file before upload."""
    staged, _ = _download(remote, item, staging)
    try:
        merged = merge_jsonl_by_key(item.local, staged, kind=merge_kind(item.path))
    except ValueError as e:
        log.warning("sync push: cannot merge %s by key (%s)", item.path, e)
        item.action = FORCE if force else CONFLICT
        return
    if _apply_merge(item, merged, "push"):
        item.digest = file_digest(item.local)


def _conflict_copies(items: list[_Item], tree: dict[str, Any],
                     stamp: str) -> tuple[dict[str, str], list[_Upload]]:
    """Where each conflicting local file is kept on the Hub: an identical copy already under
    conflicts/, or a new conflicts/<stamp>/<path> (``<stamp>-<n>`` when that path is taken, so a copy
    never replaces another one), returned as an upload."""
    existing: dict[str, list[tuple[str, Any]]] = {}
    for path, entry in tree.items():
        parts = path.split("/", 2)
        if len(parts) == 3 and parts[0] == CONFLICTS_DIR:
            existing.setdefault(parts[2], []).append((path, entry))
    kept: dict[str, str] = {}
    uploads: list[_Upload] = []
    for item in items:
        copy = next((p for p, e in existing.get(item.path, []) if item.digest.matches(e)), None)
        if copy is None:
            copy, n = f"{CONFLICTS_DIR}/{stamp}/{item.path}", 0
            while copy in tree:
                n += 1
                copy = f"{CONFLICTS_DIR}/{stamp}-{n}/{item.path}"
            uploads.append(_Upload(copy, item.local, item.digest, None))
        kept[item.path] = copy
    return kept, uploads


def _parts(uploads: Sequence[_Upload]) -> list[list[_Upload]]:
    """Consecutive runs of at most COMMIT_FILES files and COMMIT_BYTES bytes (a bigger file goes alone)."""
    parts: list[list[_Upload]] = []
    size = 0
    for upload in uploads:
        n = upload.local.stat().st_size
        if not parts or len(parts[-1]) >= COMMIT_FILES or size + n > COMMIT_BYTES:
            parts.append([])
            size = 0
        parts[-1].append(upload)
        size += n
    return parts


def _commit(remote: _Remote, tree: dict[str, Any], uploads: Sequence[_Upload], state: _State,
            report: SyncReport) -> None:
    """Send ``uploads`` with create_commit, part by part, each guarded by parent_commit: the listed
    commit first, then the previous part's commit. Before every later part the tree at that commit is
    listed and must be the plan, because huggingface_hub makes no commit when every file of a part is
    already on the Hub and then returns the CURRENT head unchecked."""
    from huggingface_hub import CommitOperationAdd

    parts = _parts(uploads)
    parent = remote.sha
    written: dict[str, Digest] = {}
    for n, part in enumerate(parts, 1):
        what = f"committing part {n}/{len(parts)}" if len(parts) > 1 else "committing"
        if written:
            _check_tree(remote, parent, tree, written, what)
        ops = [CommitOperationAdd(path_in_repo=u.path, path_or_fileobj=str(u.local)) for u in part]
        where = sorted({u.item.prefix if u.item is not None else CONFLICTS_DIR for u in part})
        message = f"lift sync push {', '.join(where)}"
        if len(parts) > 1:
            message += f" (part {n}/{len(parts)})"
        with _hub_errors(what, remote.repo):
            info = remote.api.create_commit(repo_id=remote.repo, repo_type=REPO_TYPE, operations=ops,
                                            commit_message=message, parent_commit=parent)
        oid = getattr(info, "oid", None)
        if not isinstance(oid, str) or not _SHA.fullmatch(oid):
            raise SyncError(f"{what}: the Hub returned no commit sha; rerun lift sync push")
        if oid != parent:
            report.commits.append(oid)
        parent = oid
        for u in part:
            written[u.path] = u.digest
            if u.item is not None:
                state.set(u.item.path, u.item.digest)


def _check_tree(remote: _Remote, revision: str, tree: dict[str, Any], written: dict[str, Digest],
                what: str) -> None:
    """Raise _RemoteMoved unless the repo at ``revision`` is exactly the plan: the listed tree plus the
    files this push wrote."""
    now = _remote_tree(remote, revision)
    for path in sorted(set(now) | set(tree) | set(written)):
        entry, old = now.get(path), tree.get(path)
        if path in written:
            planned = entry is not None and written[path].matches(entry)
        else:
            planned = (entry is None and old is None) or (
                entry is not None and old is not None and remote_hash(entry) == remote_hash(old))
        if not planned:
            raise _RemoteMoved(f"{what} ({path} is not as planned)")


# ---------------------------------------------------------------------------- status


def status(subdirs: Sequence[str] = (), *, api: Any = None) -> dict[str, SyncReport]:
    """Dry run: {"pull": ..., "push": ...}, what each would do now. Downloads, uploads and writes
    nothing (merges are planned, not tried)."""
    selected = select(subdirs)
    remote = _connect(api)
    tree = _remote_tree(remote)
    state = _State.load(remote.repo)
    digest = _digest_memo()
    prefixes = [p for p, _ in selected]
    reports = {}
    for direction, plan in (("pull", _plan_pull), ("push", _plan_push)):
        items = plan(selected, tree, state, force=False, digest=digest)
        reports[direction] = SyncReport(direction, remote.repo, remote.sha, prefixes,
                                        {i.path: i.action for i in items}, dry_run=True)
    return reports
