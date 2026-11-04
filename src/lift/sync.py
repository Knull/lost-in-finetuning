"""Move data, outputs and validation files between machines through the PRIVATE HF dataset repo
named by ``LIFT_RUNS_REPO`` (configs/project.yaml ``storage.runs_repo_env``). Implemented in P02.

The repo holds raw harmful completions, unrepaired adapters and human labels, so every command
first checks ``dataset_info(repo).private is True`` and refuses otherwise (False, or unknown).

Synced trees, as canonical POSIX repo prefix -> local dir (from lift.paths at call time, so the
LIFT_DATA_DIR, LIFT_OUTPUT_DIR and LIFT_VALIDATION_DIR overrides apply): data/prompts, data/sft,
data/interim/translations, outputs, validation/sheets, validation/keys, validation/labels. Caches, temp
files, the sync state, outputs/.sync-backup and outputs/.aside (the files ``lift restart`` set aside)
are never synced (IGNORE_PATTERNS).

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
  never pushed) and the local copy win on push (the Hub keeps the old one in its history);
- judgements and AuditKIT records pair with their cell, checked by content: a cell whose files do not
  belong together is one every reader rejects and no sync repairs, and P08 publishes the records.
  A judgement pairs with a generations file when its key is there, with the same pid, and its
  output_sha is the sha256 of the output now under that key (judge_file's resume and load_records
  enforce the same rule; a judgement without output_sha never pairs). A non-empty AuditKIT record
  (outputs/metrics/auditkit/{judge}/{variant}/{cell}[-{dtype}][.full].json) pairs when its
  metadata.lift keys_sha256 and labels_sha256 are those of its judge's judgements file of the cell,
  which pairs in turn; an empty one (retired by lift restart) always does.
- push: a judgements file goes up only with the generations it was made from, and a record only with
  those and its judge's judgements: one whose sources will not be this machine's copy on the Hub after
  the push (a conflict, stale, or left out of the push and different) is held back. So is one that
  does not pair with them, a merge included (the merge is then not written; when only the Hub's
  judgements would not pair, it is a plain conflict, which --force settles by sending the local copy
  as it is). Held back means a CONFLICT, never merged or sent to its path, and --force sends nothing
  that does not pair; the summary names the ``lift restart`` line that sets it aside (``unpaired``). A
  push that replaces a cell's generations, or a judgements file, other than by appending reports the
  cell's judgements files and records on the Hub that it does not send and that no longer pair
  (``unpaired_on_hub``, fetched unless the copy here is the Hub's).
- pull: generations come first. A judgements file or record of a cell whose generations conflict is
  not taken from the Hub (it goes with the Hub's generations); a merge of judgements that would not
  pair with the generations here is not written; and every judgements file and record of a cell the
  pull changed that no longer pairs with the generations here is reported (``unpaired``).

Push sends exactly the files it planned (new, updated, merged, forced) and the conflict copies,
with ``create_commit``. Never upload_folder: it re-sends files nobody changed and guards only the
first of its commits, so a later one can revert another machine's push. Every commit carries
``parent_commit``: the listed commit, then each part's own commit when a push is split
(COMMIT_FILES, COMMIT_BYTES), and the tree at that commit is listed again before the next part. A
commit landing meanwhile (HTTP 412, or a tree that is not the plan) makes the push plan again.

Nothing is deleted on either side. Downloads go to a staging dir outside the synced trees, are
checked against the tree hash (hf_hub_download silently returns stale local files when the Hub is
unreachable) and are moved into place atomically. Logs carry counts and paths, never contents.

Sync must not run while a lift command on the same machine is writing outputs. As a guard, a local
file is hashed again right before a download replaces it or a merge is written into it; one that
changed since planning is left as it is and reported as a conflict (``SyncReport.busy``, without a
conflict copy), which a rerun settles once nothing writes it.

Errors are SyncError. ``permanent`` marks the ones no rerun can fix, so runners stop instead of
retrying: no token, or one the Hub rejects (401) or refuses (403); the runs repo missing (404),
disabled or not private; LIFT_RUNS_REPO unset or malformed; HF_DEBUG or HF_HUB_OFFLINE set; a bad
path; an unreadable sync state. Other HTTP answers, network errors, a download that does not match
and a repo that keeps moving are transient. ``preflight`` checks the permanent ones, and whether the
token may write, before a job spends GPU time. An error says what is wrong with LIFT_RUNS_REPO but
never shows its value, which is named only once the Hub has answered for it as a repo: a token pasted
there by mistake would otherwise end up in a job log that is kept (Kaggle saves every printed line).
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
from lift.io import sha256_obj, sha256_text, stable_json, write_jsonl

log = logging.getLogger(__name__)

REPO_TYPE = "dataset"
STATE_FILE = ".sync-state.json"     # in paths.output_dir(); never pushed
STATE_VERSION = 1
BACKUP_DIR = ".sync-backup"         # in paths.output_dir(): local copies a forced pull replaced; never pushed
ASIDE_DIR = ".aside"                # in paths.output_dir(): the files lift restart set aside; never synced
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
# A merged file must still hold one stack (check_merged), as generate and the readers require.
RUN_FIELDS = ("seconds",)
STACK_FIELDS = ("torch", "transformers", "gpu_capability")      # as lift.models.generate.STACK_FIELDS
RUN_META = ("gpu", "device", "batch_size", "n_generated", *STACK_FIELDS, "stack")
# Never synced. Matched like huggingface_hub matches them (fnmatchcase, `*` crosses `/`) on the path
# relative to the synced folder. CommitOperationAdd refuses a `.cache` or `.git` component, and
# hf_hub_download leaves `.cache/huggingface` metadata behind.
IGNORE_PATTERNS = (
    ".cache", ".cache/*", "*/.cache", "**/.cache/**", ".git", ".git/*", "*/.git", "**/.git/**",
    "__pycache__/*", "**/__pycache__/**", "*.lock", "*.tmp", "*.incomplete", "*.pyc",
    ".DS_Store", "**/.DS_Store", STATE_FILE, BACKUP_DIR, f"{BACKUP_DIR}/*", ASIDE_DIR, f"{ASIDE_DIR}/*",
)
# AuditKIT record names (lift.audit.auditkit_path): {set}_{lang}_{split}, then -{dtype} for a precision
# other than generation.dtype, then .full for the private record that keeps the outputs.
_RECORD_NAME = re.compile(r"(?P<cell>.+_[^_.-]+)(?:-[^_.-]+)?(?:\.full)?\.json")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_NOT_SHOWN = "its value is not shown, in case a token was pasted there"

# What happened to a file (for status: what would happen).
NEW, UPDATE, SAME, MERGE, FORCE, CONFLICT = "new", "update", "same", "merge", "force", "conflict"
KEEP = "keep-local"   # pull: edited here, unchanged on the Hub; kept until pushed
STALE = "stale"       # push: newer on the Hub, unchanged here; not uploaded
ACTIONS = (NEW, UPDATE, MERGE, FORCE, SAME, KEEP, STALE, CONFLICT)
_SENT = (NEW, UPDATE, MERGE, FORCE)   # push: these go up
_SHA = re.compile(r"[0-9a-f]{40}")
# whoami: the fine-grained permission that writes a repo, and the scope types preflight can place
_REPO_WRITE = "repo.write"
_SCOPES = ("user", "org", "model", "dataset", "space")


class SyncError(RuntimeError):
    """Sync cannot or must not go ahead. The message says what to do next; it never holds a token.
    ``permanent``: no rerun can fix it (``lift sync`` exits 2 and runners stop instead of retrying);
    otherwise a rerun may, and resumes, since sync compares content."""

    def __init__(self, message: str, *, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


class _RemoteMoved(SyncError):
    """The runs repo got a commit this push did not plan against (HTTP 412 on parent_commit, or a
    tree between two parts that is not the plan). Push plans again; anywhere else a rerun does."""

    def __init__(self, what: str):
        super().__init__(f"{what}: the runs repo changed meanwhile (another machine pushed); rerun")
        self.what = what


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
    read_generations and resume, the judge pipeline's load_records), as a merge of two valid files
    can: one record per key and per prompt ((set, lang, pid) for generations, pid for judgements),
    one variant/set/lang/split cell, and for generations one meta.run_config, meta.dtype and stack (a
    record without one counts as a value of its own), for judgements one judge id and judge stack
    (native.stack; an API judge has none). Two machines on different stacks therefore conflict
    instead of pushing a cell that every reader refuses."""
    per_key: Counter = Counter()
    per_prompt: Counter = Counter()
    single: dict[str, set[str]] = {}
    for row in rows:
        meta = row["meta"] if isinstance(row.get("meta"), dict) else {}
        one: dict[str, Any] = {"variant/set/lang/split": [row.get(f) for f in ("variant", "set", "lang",
                                                                                "split")]}
        if kind == "generations":
            prompt: Any = [row.get("set"), row.get("lang"), row.get("pid")]
            one.update({"meta.run_config": meta.get("run_config"), "meta.dtype": meta.get("dtype"),
                        "stack": record_stack(meta)})
        else:
            prompt = row.get("pid")
            native = row["native"] if isinstance(row.get("native"), dict) else {}
            one.update({"judge": row.get("judge"), "judge stack": native.get("stack")})
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


def record_stack(meta: dict) -> str:
    """A generation's stack as lift.models.generate.record_stack reads it: meta.stack, else (an older
    record) the digest of its torch, transformers and GPU class. Kept here so sync needs no torch."""
    return str(meta.get("stack") or sha256_obj({name: meta.get(name) for name in STACK_FIELDS})[:16])


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
    trees and anything never synced (IGNORE_PATTERNS: caches, the sync state, outputs/.sync-backup,
    outputs/.aside) are refused. A folder inside another selected one adds nothing.
    """
    table = targets()
    chosen: dict[str, Path] = {}
    for raw in subdirs:
        norm = str(raw).replace("\\", "/").strip().strip("/")
        parts = norm.split("/")
        if not norm or any(part in ("", ".", "..") for part in parts):
            raise SyncError(f"bad sync path {raw!r}: give a repo path such as outputs/generations",
                            permanent=True)
        base = max((b for b in table if norm == b or norm.startswith(b + "/")), key=len, default=None)
        if base is not None:
            if norm != base and _ignored(norm[len(base) + 1:]):
                raise SyncError(f"{raw!r} is never synced: caches, temp files, the sync state, "
                                f"{BACKUP_DIR} and {ASIDE_DIR} stay on this machine", permanent=True)
            chosen[norm] = table[base].joinpath(*parts[base.count("/") + 1:])
            continue
        below = [b for b in table if b.startswith(norm + "/")]
        if not below:
            raise SyncError(f"{raw!r} is not synced; choose from {', '.join(table)} or a folder under one",
                            permanent=True)
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
        raise SyncError(f"remote path {path!r} is not a safe relative path; remove it from the runs repo",
                        permanent=True)
    return dest


def _owner(path: str, prefixes: Sequence[str]) -> str:
    return next(p for p in prefixes if path.startswith(p + "/"))


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------- the Hub


def runs_repo() -> str:
    """The runs repo id, from the env var named in configs/project.yaml (storage.runs_repo_env). A
    malformed value is refused with what is wrong with it, never the value (module docstring)."""
    var = _repo_env()
    repo = os.environ.get(var, "").strip()
    if not repo:
        raise SyncError(f"{var} is not set: export {var}=<hf-user>/lift-runs, a PRIVATE Hugging Face "
                        "dataset repo (hf repos create lift-runs --repo-type dataset --private)",
                        permanent=True)
    owner, slash, name = repo.partition("/")
    problem = ("has no '/'" if not slash else "has more than one '/'" if "/" in name
               else "has an empty owner or name" if not owner or not name else "")
    if problem:
        raise SyncError(f"{var} is not a repo id of the form <owner>/<name>: it {problem} ({_NOT_SHOWN})",
                        permanent=True)
    return repo


def _repo_env() -> str:
    return config.get("storage.runs_repo_env")


def _require_token(api: Any = None) -> None:
    from huggingface_hub import get_token

    if getattr(api, "token", None) or get_token() is not None:
        return
    raise SyncError("no Hugging Face token: set HF_TOKEN (a WRITE token for the runs repo to push) "
                    "or run `hf auth login`", permanent=True)


def make_api() -> Any:
    """An HfApi using the configured token. Refuses when there is none (its value is never shown)."""
    from huggingface_hub import HfApi

    _require_token()
    return HfApi()


def _status_code(error: BaseException) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


@contextmanager
def _hub_errors(what: str) -> Iterator[None]:
    """Turn huggingface_hub errors into a SyncError that says what to do and whether a rerun can help: a
    token rejected or refused, a repo missing or disabled, offline mode and a bad repo id are permanent;
    any other HTTP answer and network errors are not. The message holds neither the token nor the runs
    repo id, which the Hub may not have answered for yet (module docstring); HFValidationError's own text
    repeats that id, so it is not chained."""
    from huggingface_hub.errors import (
        DisabledRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
    )

    network = (ConnectionError, TimeoutError, HfHubHTTPError.__bases__[0])   # [0]: the HTTP client's error
    try:
        yield
    except RepositoryNotFoundError as e:  # also GatedRepoError; 401 without a token, 404 with one
        var = _repo_env()
        raise SyncError(f"{what}: the repo {var} names was not found or is not readable with this token "
                        f"(HTTP {_status_code(e)}). Check {var}, create the repo as a PRIVATE dataset repo "
                        "and use a token that can read it (and write it, to push)", permanent=True) from e
    except DisabledRepoError as e:
        raise SyncError(f"{what}: the repo {_repo_env()} names is disabled on the Hub", permanent=True) from e
    except HfHubHTTPError as e:  # an invalid token is a plain 401 here, not RepositoryNotFoundError
        code = _status_code(e)
        if code == 412:
            raise _RemoteMoved(what) from e
        if code == 401:
            raise SyncError(f"{what}: the Hugging Face token was rejected (HTTP 401): check HF_TOKEN",
                            permanent=True) from e
        if code == 403:
            raise SyncError(f"{what}: the token may not do this (HTTP 403); pushing needs a WRITE token "
                            "for the runs repo", permanent=True) from e
        raise SyncError(f"{what}: the Hub answered HTTP {code}; rerun (sync compares content, so it "
                        "resumes)") from e
    except OfflineModeIsEnabled as e:
        raise SyncError(f"{what}: huggingface_hub is in offline mode (HF_HUB_OFFLINE is set)",
                        permanent=True) from e
    except LocalEntryNotFoundError as e:  # what hf_hub_download raises when the Hub cannot be reached
        raise SyncError(f"{what}: could not reach the Hub; check the connection and rerun") from e
    except HFValidationError:
        raise SyncError(f"{what}: {_repo_env()} is not a valid Hugging Face repo id: <owner>/<name>, each "
                        "at most 96 letters, digits, '-', '_' and '.', neither starting nor ending with '-' "
                        f"or '.', without '--', '..' or a final '.git' ({_NOT_SHOWN})",
                        permanent=True) from None
    except network as e:   # a refused or dropped connection, a timeout
        raise SyncError(f"{what}: could not reach the Hub ({type(e).__name__}); check the connection "
                        "and rerun") from e


def require_private(api: Any, repo: str) -> Any:
    """The repo's DatasetInfo, or SyncError unless the Hub confirms that the repo is private."""
    with _hub_errors("checking the runs repo"):
        info = api.dataset_info(repo, expand=["private", "sha"])
    if info.private is not True:  # False is public; None means the Hub did not say
        raise SyncError(f"refusing to sync: dataset repo {repo!r} is not private (private={info.private!r}). "
                        "It holds raw harmful completions, unrepaired adapters and human labels: make it "
                        f"private or point {_repo_env()} at a private repo",
                        permanent=True)
    if not isinstance(info.sha, str) or not _SHA.fullmatch(info.sha):
        raise SyncError(f"the Hub returned no commit sha for {repo!r}; rerun")
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
                        "this log. Unset it to sync", permanent=True)
    if api is None:
        api = make_api()
    else:
        _require_token(api)
    disable_progress_bars()
    return _Remote(api, repo, require_private(api, repo).sha)


def preflight(*, write: bool = True, api: Any = None) -> None:
    """What a job checks before it spends GPU time: LIFT_RUNS_REPO and a token set, HF_DEBUG unset, the
    runs repo there and private (``private is True``) and, with ``write``, a token that may write it.
    Write access is refused only when HfApi.whoami() shows it (``_cannot_write``); whatever that answer
    cannot tell passes, and the push decides. SyncError, ``permanent`` for all of these and not for a
    Hub or network error."""
    remote = _connect(api)
    if not write:
        return
    with _hub_errors("checking what the token may do"):
        me = remote.api.whoami()
    if why := _cannot_write(me, remote.repo):
        raise SyncError(f"the token cannot write {remote.repo!r} ({why}): pushing needs a WRITE token for "
                        "the runs repo", permanent=True)


def _cannot_write(me: Any, repo: str) -> str | None:
    """Why HfApi.whoami()'s answer shows that the token cannot write ``repo``: a read token, or a
    fine-grained one none of whose scoped permissions grants repo.write on the repo or its owner. None
    when it can, or when the answer does not say (another kind of token, an unexpected shape, a scope
    this cannot place, such as a resource group)."""
    auth = me.get("auth") if isinstance(me, dict) else None
    token = auth.get("accessToken") if isinstance(auth, dict) else None
    role = token.get("role") if isinstance(token, dict) else None
    if role == "read":
        return "it is a read token"
    fine = token.get("fineGrained") if role == "fineGrained" else None
    scoped, broad = (fine.get("scoped"), fine.get("global", [])) if isinstance(fine, dict) else (None, None)
    if not isinstance(scoped, list) or not isinstance(broad, list) or _REPO_WRITE in broad:
        return None
    owner = repo.split("/")[0].lower()
    covering = {("dataset", repo.lower()), ("user", owner), ("org", owner)}   # caseless: refuse only if sure
    for scope in scoped:
        entity = scope.get("entity") if isinstance(scope, dict) else None
        perms = scope.get("permissions") if isinstance(scope, dict) else None
        if not isinstance(entity, dict) or not isinstance(perms, list):
            return None
        if _REPO_WRITE not in perms:
            continue
        kind, name = entity.get("type"), str(entity.get("name", "")).lower()
        if kind not in _SCOPES or (kind, name) in covering:
            return None
    return "it is a fine-grained token without repo.write on the repo or its owner"


def _remote_tree(remote: _Remote, revision: str | None = None) -> dict[str, Any]:
    """Every file of the repo at one commit (the pinned one by default; folders dropped), from a
    single listing."""
    with _hub_errors("listing the runs repo"):
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
    with _hub_errors(f"downloading {item.path}"):
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
                            "sides", permanent=True) from e
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
    busy: bool = False           # the local file changed while sync ran: left as it is (a CONFLICT)


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


def _judged_from(path: str) -> tuple[str, ...]:
    """The generations file a judgements file was made from (outputs/judgements/{judge}/{variant}/{cell}
    -> outputs/generations/{variant}/{cell}); () for any other path."""
    parts = path.split("/")
    if len(parts) == 5 and parts[:2] == ["outputs", "judgements"] and path.endswith(".jsonl"):
        return (f"outputs/generations/{parts[3]}/{parts[4]}",)
    return ()


def _recorded_from(path: str) -> tuple[str, ...]:
    """The files an AuditKIT record was made from: outputs/metrics/auditkit/{judge}/{variant}/{name}, with
    the name {cell}[-{dtype}][.full].json (_RECORD_NAME), -> outputs/generations/{variant}/{cell}.jsonl
    and outputs/judgements/{judge}/{variant}/{cell}.jsonl; () for any other path."""
    parts = path.split("/")
    if len(parts) != 6 or parts[:3] != ["outputs", "metrics", "auditkit"]:
        return ()
    if (name := _RECORD_NAME.fullmatch(parts[5])) is None:
        return ()
    judge, variant, cell = parts[3], parts[4], name["cell"]
    return f"outputs/generations/{variant}/{cell}.jsonl", f"outputs/judgements/{judge}/{variant}/{cell}.jsonl"


def _hold_back(items: list[_Item], tree: dict[str, Any], digest: Callable[[Path], Digest],
               made_from: Callable[[str], tuple[str, ...]]) -> dict[str, str]:
    """A file goes up only with the files it was made from (``made_from``): one the push would send (or
    merge) becomes a CONFLICT, kept under conflicts/ like any other, unless the Hub will then hold this
    machine's copy of each of them. Otherwise the Hub would pair one machine's judgements, or AuditKIT
    record, with another machine's generations or judgements. A retired record goes up regardless: it
    counts nothing. Returns path -> the first file it waits for."""
    planned = {i.path: i for i in items}
    held = {}
    for item in items:
        if item.action not in _SENT or _retired(item):
            continue
        waits = [p for p in made_from(item.path) if not _same_after_push(p, planned, tree, digest)]
        if waits:
            item.action = CONFLICT
            held[item.path] = waits[0]
    return held


def _same_after_push(path: str, planned: dict[str, _Item], tree: dict[str, Any],
                     digest: Callable[[Path], Digest]) -> bool:
    """Whether the Hub will hold this machine's copy of ``path`` once the push is done (neither side
    having one counts as the same). A planned file must not be held back (STALE, CONFLICT); one left
    out of the push must already match the Hub's."""
    item = planned.get(path)
    if item is not None:
        return item.action not in (STALE, CONFLICT)
    local, entry = paths.output_dir().joinpath(*path.split("/")[1:]), tree.get(path)
    if not local.is_file():
        return entry is None
    return entry is not None and digest(local).matches(entry)


def _retired(item: _Item) -> bool:
    """An AuditKIT record ``lift restart`` emptied: readers and P08 skip it, so it may always go up."""
    return bool(_recorded_from(item.path)) and item.digest is not None and item.digest.sha256 == _EMPTY_SHA256


# ---------------------------------------------------------------------------- pairing


def _cell(path: str) -> tuple[str, str] | None:
    """(variant, "{set}_{lang}_{split}") of a cell's generations file, judgements file or AuditKIT
    record; None for any other path."""
    sources = (path,) if merge_kind(path) == "generations" else _judged_from(path) or _recorded_from(path)
    parts = sources[0].split("/") if sources else []
    if len(parts) != 4:
        return None
    name = parts[3].removesuffix(".jsonl")
    pieces = name.rsplit("_", 2)
    return (parts[2], name) if len(pieces) == 3 and all(pieces) else None


def _generations_of(cell: tuple[str, str]) -> str:
    return f"outputs/generations/{cell[0]}/{cell[1]}.jsonl"


def _judge_dir_of(path: str) -> str:
    """The judge directory of a judgements file (outputs/judgements/{judge}/...) or an AuditKIT record
    (outputs/metrics/auditkit/{judge}/...)."""
    parts = path.split("/")
    return parts[2] if parts[1] == "judgements" else parts[3]


def _rows(data: bytes) -> list[dict] | None:
    """The records of a JSONL file as its readers see them: blank lines skipped, a last line without its
    newline kept if it parses and dropped if it was cut short (as repair_tail does). None when any other
    line is not a JSON object: nothing in such a file can be trusted to pair."""
    lines = data.split(b"\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:   # also invalid UTF-8
            row = None
        if isinstance(row, dict):
            rows.append(row)
        elif i < len(lines) - 1:
            return None
    return rows


@dataclass(frozen=True)
class _Outputs:
    """What pairing needs of a cell's generations file: key -> (pid, sha256 of its output), and the
    precision it was made in (the first record's meta.dtype), which the cell's lift restart line names."""

    by_key: dict[str, tuple[Any, str | None]]
    dtype: str | None = None


def _outputs(data: bytes | None) -> _Outputs:
    """The _Outputs of a generations file's bytes. No file, or one that is not JSONL, holds none."""
    by_key: dict[str, tuple[Any, str | None]] = {}
    dtype = None
    for row in (_rows(data) if data else None) or ():
        key, output, meta = row.get("key"), row.get("output"), row.get("meta")
        if isinstance(key, str):
            by_key[key] = (row.get("pid"), sha256_text(output) if isinstance(output, str) else None)
        if dtype is None and isinstance(meta, dict) and isinstance(meta.get("dtype"), str):
            dtype = meta["dtype"]
    return _Outputs(by_key, dtype)


def _pairs(judgements: Sequence[dict] | None, gens: _Outputs) -> bool:
    """Whether every judgement is of a generation in ``gens``: its key there, with the same pid, and its
    output_sha the sha256 of the output now under the key. A judgement without output_sha never pairs,
    nor does anything of a file that is not JSONL (None); a file without judgements always does."""
    if judgements is None:
        return False
    for row in judgements:
        key, sha = row.get("key"), row.get("output_sha")
        gen = gens.by_key.get(key) if isinstance(key, str) else None
        if gen is None or gen[0] != row.get("pid") or not isinstance(sha, str) or sha != gen[1]:
            return False
    return True


def _counts(record: bytes, judgements: Sequence[dict]) -> bool:
    """Whether an AuditKIT record counts exactly ``judgements``: its metadata.lift keys_sha256 and
    labels_sha256 are sha256_obj({pid: key}) and sha256_obj({pid: label}) over them, as lift.audit writes
    both. A record that does not parse, or names neither, counts nothing."""
    try:
        meta = json.loads(record)["metadata"]["lift"]
    except (ValueError, KeyError, TypeError):
        return False
    if not isinstance(meta, dict) or not all(isinstance(row.get("pid"), str) for row in judgements):
        return False
    keys = sha256_obj({row["pid"]: row.get("key") for row in judgements})
    labels = sha256_obj({row["pid"]: row.get("label") for row in judgements})
    return meta.get("keys_sha256") == keys and meta.get("labels_sha256") == labels


def _record_fault(record: bytes, judgements: bytes | None, gens: _Outputs) -> str | None:
    """Why an AuditKIT record may not stand beside its cell: "judgements" when it does not count exactly
    ``judgements``, its judge's judgements file of the cell (no such file counts nothing), "generations"
    when that file does not pair with ``gens``; None when it may, and for a retired (empty) record."""
    if not record:
        return None
    rows = _rows(judgements) if judgements is not None else None
    if rows is None or not _counts(record, rows):
        return "judgements"
    return None if _pairs(rows, gens) else "generations"


def _local_bytes(path: str) -> bytes | None:
    """This machine's copy of the canonical ``path`` (under outputs/), None when it has none."""
    local = paths.output_dir().joinpath(*path.split("/")[1:])
    return local.read_bytes() if local.is_file() else None


def _restart_line(cell: tuple[str, str], judge_dir: str | None, gens: _Outputs) -> str:
    """The ``lift restart`` line that sets aside a file of ``cell`` which does not pair with it: --judge for
    the role whose judgements directory is ``judge_dir`` (lift.judge.pipeline.judge_dir), the whole cell
    when no role's is, and the precision the cell's generations were made in, which lift restart must be
    given. Imported here: sync runs without torch."""
    from lift.judge.pipeline import ROLES
    from lift.judge.pipeline import judge_dir as role_dir
    from lift.models.generate import restart_command

    def owns(role: str) -> bool:
        try:
            return role_dir(role) == judge_dir
        except (KeyError, ValueError):     # an entry that names no judge kind
            return False

    role = next((r for r in ROLES if judge_dir is not None and owns(r)), None)
    variant, name = cell
    set_name, lang, split = name.rsplit("_", 2)
    return restart_command(variant, set_name, lang, split, judge=role, dtype=gens.dtype)


def _unpair(item: _Item, report: SyncReport, source: str, gens: _Outputs | None = None) -> None:
    """``item`` does not pair with ``source`` (its generations or judgements): a CONFLICT whose summary
    names the restart line. ``gens``: the cell's generations here, read when not given."""
    cell = _cell(item.path)
    if gens is None:
        gens = _outputs(_local_bytes(_generations_of(cell)))
    item.action = CONFLICT
    report.held[item.path] = source
    report.unpaired[item.path] = _restart_line(cell, _judge_dir_of(item.path), gens)


def _merge_refused(item: _Item, merged: Sequence[dict]) -> str | None:
    """Why a merge of a judgements file must not be written: "local" when this machine's own judgements do
    not pair with the cell's generations here, "hub" when only the Hub's would not, None when every
    merged judgement pairs (and for any other file). Push asks once its location rule made the
    generations here the ones the Hub will hold; pull, once it took the Hub's generations."""
    cell = _cell(item.path)
    if merge_kind(item.path) != "judgements" or cell is None:
        return None
    gens = _outputs(_local_bytes(_generations_of(cell)))
    if _pairs(merged, gens):
        return None
    return "hub" if _pairs(_rows(item.local.read_bytes()), gens) else "local"


def _hold_unpaired(items: list[_Item], report: SyncReport, *, records: bool) -> None:
    """Push: a judgements file (``records`` False) or an AuditKIT record that would go up must pair with its
    cell as the Hub will then hold it, which _hold_back made this machine's copy. Otherwise it is held
    back (_unpair), also with --force. A record held back with judgements that do not pair names their
    restart line."""
    gens: dict[tuple[str, str], _Outputs] = {}
    for item in items:
        cell = _cell(item.path)
        made_from = _recorded_from(item.path) if records else _judged_from(item.path)
        if cell is None or not made_from:
            continue
        if records and (source := report.held.get(item.path)) in report.unpaired:
            report.held[item.path] = report.held[source]     # the generations its judgements do not pair with
            report.unpaired[item.path] = report.unpaired[source]
            continue
        if item.action not in _SENT:
            continue
        if cell not in gens:
            gens[cell] = _outputs(_local_bytes(_generations_of(cell)))
        data = item.local.read_bytes()
        if records and (fault := _record_fault(data, _local_bytes(made_from[1]), gens[cell])):
            _unpair(item, report, made_from[1] if fault == "judgements" else made_from[0], gens[cell])
        elif not records and not _pairs(_rows(data), gens[cell]):
            _unpair(item, report, made_from[0], gens[cell])


def _extends(local: Path, entry: Any) -> bool:
    """Whether ``local`` is the Hub's copy (``entry``) with lines appended: its first entry.size bytes hash as
    the entry does. Sending it keeps every record the Hub had, so whatever paired with them still does."""
    size = getattr(entry, "size", None)
    if not isinstance(size, int) or local.stat().st_size < size:
        return False
    algo, want = remote_hash(entry)
    h = hashlib.sha256() if algo == "sha256" else hashlib.sha1(b"blob %d\0" % size, usedforsecurity=False)
    left = size
    with open(local, "rb") as f:
        while left and (block := f.read(min(left, 1 << 20))):
            h.update(block)
            left -= len(block)
    return not left and h.hexdigest() == want


def _unpaired_on_hub(remote: _Remote, items: list[_Item], tree: dict[str, Any], staging: Path,
                     digest: Callable[[Path], Digest]) -> dict[str, str]:
    """Push: the judgements files and AuditKIT records on the Hub that this push does not send and that no
    longer pair once it replaces their cell's generations, or a record's judgements, other than by
    appending (path -> restart line). Each is fetched unless the copy here is the Hub's; a push that only
    appends fetches nothing."""
    sent = {i.path: i for i in items if i.action in _SENT}
    replaced = {p for p, i in sent.items() if i.action in (UPDATE, FORCE) and i.entry is not None
                and merge_kind(p) in ("generations", "judgements") and _cell(p) is not None
                and not _extends(i.local, i.entry)}
    cells = {_cell(p) for p in replaced if merge_kind(p) == "generations"}

    def hub(path: str) -> bytes | None:
        """What the Hub will hold at ``path`` once this push is done."""
        if path in sent:
            return sent[path].local.read_bytes()
        if (entry := tree.get(path)) is None:
            return None
        local = paths.output_dir().joinpath(*path.split("/")[1:])
        if local.is_file() and digest(local).matches(entry):
            return local.read_bytes()
        return _download(remote, _Item(path, "outputs", local, entry, None), staging)[0].read_bytes()

    gens: dict[tuple[str, str], _Outputs] = {}
    found = {}
    for path in sorted(tree):
        cell, (_, jud) = _cell(path), _recorded_from(path) or (None, None)
        if path in sent or cell is None or merge_kind(path) == "generations" or not (
                cell in cells or jud in replaced):
            continue
        if cell not in gens:
            gens[cell] = _outputs(hub(_generations_of(cell)))
        data = hub(path)
        if (_record_fault(data, hub(jud), gens[cell]) if jud else not _pairs(_rows(data), gens[cell])):
            found[path] = _restart_line(cell, _judge_dir_of(path), gens[cell])
    return found


def _wait_for_generations(items: list[_Item]) -> dict[str, str]:
    """Pull: a judgements file or AuditKIT record of a cell whose generations conflict is not taken from the
    Hub, where it goes with the Hub's generations, not with the ones kept here: it stays as it is, a
    CONFLICT. Returns path -> those generations."""
    conflicting = {i.path for i in items if merge_kind(i.path) == "generations" and i.action == CONFLICT}
    held = {}
    for item in items:
        sources = _judged_from(item.path) or _recorded_from(item.path)
        if sources and sources[0] in conflicting and item.action in (NEW, UPDATE, MERGE, FORCE):
            item.action = CONFLICT
            held[item.path] = sources[0]
    return held


def _cell_files_here(cell: tuple[str, str]) -> list[str]:
    """The canonical paths of this machine's judgements files and AuditKIT records of ``cell``."""
    variant, name = cell
    out = paths.output_dir()
    found = [f"outputs/judgements/{d.name}/{variant}/{name}.jsonl" for d in _subdirs(out / "judgements")
             if (d / variant / f"{name}.jsonl").is_file()]
    for d in _subdirs(out / "metrics" / "auditkit"):
        folder = d / variant
        for f in sorted(folder.iterdir()) if folder.is_dir() else ():
            path = f"outputs/metrics/auditkit/{d.name}/{variant}/{f.name}"
            if f.is_file() and _cell(path) == cell:
                found.append(path)
    return found


def _subdirs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


def _unpaired_here(items: list[_Item], report: SyncReport) -> dict[str, str]:
    """Pull: the judgements files and AuditKIT records here of every cell this pull changed that no longer
    pair with the cell's generations here (path -> restart line). A cell without generations here (a pull
    of judgements alone) is left out."""
    changed = {cell for i in items if i.action in (NEW, UPDATE, FORCE, MERGE) and (cell := _cell(i.path))}
    found = {}
    for cell in sorted(changed):
        if (gen := _local_bytes(_generations_of(cell))) is None:
            continue
        gens = _outputs(gen)
        for path in _cell_files_here(cell):
            data, (_, jud) = _local_bytes(path), _recorded_from(path) or (None, None)
            if path in report.unpaired or data is None:
                continue
            if (_record_fault(data, _local_bytes(jud), gens) if jud else not _pairs(_rows(data), gens)):
                found[path] = _restart_line(cell, _judge_dir_of(path), gens)
    return found


def _changed_since_plan(item: _Item) -> bool:
    """True when the local file is no longer what the plan saw (created, deleted or written since):
    something on this machine is writing it, and replacing it now would lose those writes."""
    if item.digest is None:
        return item.local.exists()
    try:
        return file_digest(item.local) != item.digest
    except (OSError, SyncError):   # deleted, or written while being hashed
        return True


def _leave_busy(item: _Item, direction: str) -> None:
    log.warning("sync %s: %s changed on this machine while sync ran; left as it is", direction, item.path)
    item.action, item.busy = CONFLICT, True


def _apply_merge(item: _Item, merged: Merged, direction: str) -> bool:
    """Write a merge into the local file when it changes it; True if it did. A local file that changed
    since planning is left as it is instead (a conflict): the lines added meanwhile are not in the
    merge, and writing it would drop them."""
    if not (merged.added or merged.dropped):
        log.info("sync %s: %s already holds every remote record", direction, item.path)
        return False
    if _changed_since_plan(item):
        _leave_busy(item, direction)
        return False
    if merged.dropped:
        log.warning("sync %s: dropped %d unreadable line(s) of %s (lines its reader skips too)",
                    direction, merged.dropped, item.path)
    write_jsonl(item.local, merged.rows)
    log.info("sync %s: merged %d remote record(s) into %s", direction, merged.added, item.path)
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
    # Held back (push) or not taken (pull) -> the file it waits for, or does not pair with.
    held: dict[str, str] = field(default_factory=dict)
    # Judgements files and AuditKIT records here that do not pair with their cell -> the lift restart line
    # that sets them aside: held back by push, left as they are or reported by pull.
    unpaired: dict[str, str] = field(default_factory=dict)
    # push: the same on the Hub, left there by this push (it does not send them).
    unpaired_on_hub: dict[str, str] = field(default_factory=dict)
    busy: list[str] = field(default_factory=list)        # changed here while sync ran: left as they are

    def paths(self, action: str) -> list[str]:
        return sorted(p for p, a in self.files.items() if a == action)

    @property
    def conflicts(self) -> list[str]:
        return self.paths(CONFLICT)

    @property
    def ok(self) -> bool:
        return not (self.conflicts or self.unpaired or self.unpaired_on_hub)

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
        lines += [f"  CONFLICT {path}{self._notes(path)}" for path in self.conflicts]
        lines += [f"  UNPAIRED {path}" for path in sorted(self.unpaired) if self.files.get(path) != CONFLICT]
        lines += [f"  UNPAIRED {path} (on the Hub)" for path in sorted(self.unpaired_on_hub)]
        would = "would be " if self.dry_run else ""
        if kept := self.paths(KEEP):
            lines.append(f"  {len(kept)} local edit(s) {would}kept, not on the Hub yet: run lift sync push")
        if stale := self.paths(STALE):
            lines.append(f"  {len(stale)} file(s) newer on the Hub and unchanged here {would}left out: "
                         "run lift sync pull")
        if self.backups:
            lines.append(f"  {len(self.backups)} local copy(ies) replaced by the Hub's are kept under "
                         f"{self.backup_dir}")
        if plain := [p for p in self.conflicts if p not in self.held and p not in self.busy]:
            fix = (f"lift sync pull --force takes the Hub's copy (the local one is kept under "
                   f"outputs/{BACKUP_DIR})" if self.direction == "pull"
                   else "lift sync push --force uploads the local copy over the Hub's")
            lines.append(f"  {len(plain)} conflict(s): changed here and on the Hub since the last "
                         f"sync; the local copy {would or 'was '}kept. Resolve by hand, or {fix}")
        lines += self._pairing(would)
        if self.busy:
            lines.append(f"  {len(self.busy)} file(s) changed on this machine while sync ran and were "
                         "left as they are: stop whatever writes them, then rerun lift sync "
                         f"{self.direction}")
        return lines

    def _notes(self, path: str) -> str:
        """Why a conflicting file stayed as it is, and where its copy went."""
        notes = []
        if source := self.held.get(path):
            if path in self.unpaired:
                notes.append(f"does not pair with {source}")
            else:
                notes.append(f"held back with {source}" if self.direction == "push"
                             else f"not taken while {source} conflicts")
        if path in self.busy:
            notes.append("changed here while sync ran")
        if copy := self.conflict_copies.get(path):
            notes.append(f"local copy on the Hub at {copy}")
        return f" ({'; '.join(notes)})" if notes else ""

    def _pairing(self, would: str) -> list[str]:
        """What to do about the judgements and records held back or not taken, and those that do not pair."""
        lines = []
        waiting = [p for p in self.held if p not in self.unpaired]
        kinds = (("judgements file(s)", "generations", _judged_from),
                 ("AuditKIT record(s)", "generations and judgements", _recorded_from))
        for what, sources, made_from in kinds:
            if not (n := sum(bool(made_from(p)) for p in waiting)):
                continue
            if self.direction == "push":
                lines.append(f"  {n} {what} {would}held back with the {sources} they were made from: settle "
                             "those first (pull, or resolve their conflict), then push again")
            else:
                lines.append(f"  {n} {what} {would}kept as they are while their cell's generations conflict: "
                             "settle that conflict first, then pull again")
        for line, found in _by_line(self.unpaired):
            lines.append(f"  {_kinds(found)} here do not pair with their cell's generations: set them aside "
                         f"with `{line}` (on a GPU job, a line right after lift sync pull), then judge the "
                         "cell again")
        for line, found in _by_line(self.unpaired_on_hub):
            lines.append(f"  {_kinds(found)} on the Hub no longer pair with the cell this push sent: run "
                         f"lift sync pull, then `{line}`, judge the cell again and push")
        return lines


def _by_line(unpaired: dict[str, str]) -> list[tuple[str, list[str]]]:
    """(restart line, the paths it sets aside), by line."""
    grouped: dict[str, list[str]] = {}
    for path, line in sorted(unpaired.items()):
        grouped.setdefault(line, []).append(path)
    return sorted(grouped.items())


def _kinds(found: Sequence[str]) -> str:
    """"2 judgements file(s) and 1 AuditKIT record(s)"."""
    records = sum(bool(_recorded_from(p)) for p in found)
    return " and ".join(f"{n} {what}" for n, what in ((len(found) - records, "judgements file(s)"),
                                                     (records, "AuditKIT record(s)")) if n)


def _log(report: SyncReport) -> None:
    totals = Counter(report.files.values())
    log.info("sync %s %s@%s: %s", report.direction, report.repo, report.revision[:10],
             ", ".join(f"{totals[a]} {a}" for a in ACTIONS if totals[a]) or "nothing to do")


# ---------------------------------------------------------------------------- pull


def pull(subdirs: Sequence[str] = (), *, force: bool = False, api: Any = None) -> SyncReport:
    """Download new and changed files of the runs repo, pinned to its current commit, without ever
    overwriting a local edit (rules in the module docstring). Generations come first: the judgements
    and AuditKIT records of a cell whose generations conflict are not taken, and a merge of judgements
    that would not pair with the generations here is not written. Conflicts, and files of a changed cell
    that no longer pair (``report.unpaired``), leave ``report.ok`` False; ``force`` lets the remote copy
    win conflicts instead, keeping each replaced local copy under outputs/.sync-backup/<utc-timestamp>/<path>
    (``report.backups``)."""
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
            def work(item: _Item) -> tuple[Path, Digest]:
                return _fetch(remote, item, staging, backups)

            def done(item: _Item, result: tuple[Path, Digest]) -> None:
                _pulled(item, result, state, report, force=force, backups=backups)

            later = [i for i in fetch if merge_kind(i.path) != "generations"]
            _run_parallel(work, [i for i in fetch if merge_kind(i.path) == "generations"], done)
            report.held = _wait_for_generations(items)
            _run_parallel(work, [i for i in later if i.action != CONFLICT], done)
    finally:
        state.save()   # whatever finished is recorded, also when a download fails
    report.files = {i.path: i.action for i in items}
    report.backups = {i.path: str(i.backup) for i in items if i.backup is not None}
    report.backup_dir = str(backups) if report.backups else ""
    report.busy = sorted(i.path for i in items if i.busy)
    report.unpaired |= _unpaired_here(items, report)
    _log(report)
    return report


def _fetch(remote: _Remote, item: _Item, staging: Path, backups: Path) -> tuple[Path, Digest]:
    staged, digest = _download(remote, item, staging)
    if item.action != MERGE:   # a merge is made in the main thread (_pulled)
        _replace_local(item, staged, backups)
    return staged, digest


def _replace_local(item: _Item, staged: Path, backups: Path) -> None:
    """Move a verified download over the local file (a forced pull keeps that file first), unless the
    local file changed since planning: then it stays as it is, a conflict."""
    if _changed_since_plan(item):
        _leave_busy(item, "pull")
        return
    if item.action == FORCE:
        _backup(item, backups)
    _install(staged, item.local)


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


def _pulled(item: _Item, result: tuple[Path, Digest], state: _State, report: SyncReport, *, force: bool,
            backups: Path) -> None:
    """Record a verified download (already in place), or merge it into the local JSONL first. A merge that
    cannot be made, or of judgements that would not pair with the generations here, is not written: the
    Hub's copy replaces the local one with ``force``, else the local one stays (a conflict, which
    records nothing)."""
    staged, digest = result
    if item.action == MERGE:
        refused: str | None
        try:
            merged = merge_jsonl_by_key(item.local, staged, kind=merge_kind(item.path))
        except ValueError as e:
            log.warning("sync pull: cannot merge %s by key (%s)", item.path, e)
            refused = "merge"
        else:
            if refused := _merge_refused(item, merged.rows):
                log.warning("sync pull: merged, %s would hold judgements that do not pair with the cell's "
                            "generations here; not merged", item.path)
        if refused is None:
            _apply_merge(item, merged, "pull")
        elif force:
            item.action = FORCE
            _replace_local(item, staged, backups)
        elif refused == "local":
            _unpair(item, report, _generations_of(_cell(item.path)))
        else:
            item.action = CONFLICT
    if item.action == CONFLICT:
        return
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
    ``force``, left alone on the Hub and uploaded under conflicts/<utc-timestamp>/<path>, and so are
    the judgements and AuditKIT records held back (``force`` never sends one that does not pair). Conflicts,
    and judgements or records on the Hub this push leaves unpaired (``report.unpaired_on_hub``), leave
    ``report.ok`` False. If another machine commits meanwhile, the push is planned again."""
    selected = select(subdirs)
    for attempt in range(1, PUSH_ATTEMPTS + 1):
        try:
            return _push_once(selected, force=force, api=api)
        except _RemoteMoved as moved:
            log.warning("sync push: the runs repo moved on while %s (another machine pushed); "
                        "planning again (%d/%d)", moved.what, attempt, PUSH_ATTEMPTS)
    raise SyncError(f"the runs repo kept changing during push ({PUSH_ATTEMPTS} attempts): another machine "
                    "is pushing. Rerun lift sync push")


def _push_once(selected: list[tuple[str, Path]], *, force: bool, api: Any) -> SyncReport:
    remote = _connect(api)
    tree = _remote_tree(remote)
    state = _State.load(remote.repo)
    digest = _digest_memo()
    items = _plan_push(selected, tree, state, force=force, digest=digest)
    report = SyncReport("push", remote.repo, remote.sha, [p for p, _ in selected])
    gens = [i for i in items if merge_kind(i.path) == "generations"]
    rest = [i for i in items if merge_kind(i.path) != "generations"]
    try:
        for item in items:
            if item.action == SAME:
                state.set(item.path, item.digest)
        with _staging() as staging:
            # Generations first: how they end decides which judgements may go up, or be merged at all.
            for item in gens:
                _merge_for_push(remote, item, staging, report, force=force)
            report.held = _hold_back(items, tree, digest, _judged_from)
            for item in rest:
                _merge_for_push(remote, item, staging, report, force=force)
            _hold_unpaired(items, report, records=False)
            # AuditKIT records last: they follow their judgements too, whose merge may just have failed.
            report.held |= _hold_back(items, tree, digest, _recorded_from)
            _hold_unpaired(items, report, records=True)
            report.unpaired_on_hub = _unpaired_on_hub(remote, items, tree, staging, digest)
        stamp = _stamp()
        aside = [i for i in items if i.action == CONFLICT and not i.busy]   # a busy file stays here only
        report.conflict_copies, copies = _conflict_copies(aside, tree, stamp)
        uploads = [_Upload(i.path, i.local, i.digest, i) for i in items if i.action in _SENT]
        _commit(remote, tree, sorted(uploads + copies, key=lambda u: u.path), state, report)
        if copies:
            log.warning("sync push: %d conflicting local file(s) uploaded under %s/%s",
                        len(copies), CONFLICTS_DIR, stamp)
    finally:
        state.save()   # parts already committed are recorded, also when a later one fails
    report.files = {i.path: i.action for i in items}
    report.busy = sorted(i.path for i in items if i.busy)
    _log(report)
    return report


def _merge_for_push(remote: _Remote, item: _Item, staging: Path, report: SyncReport, *, force: bool) -> None:
    """Fold the remote records of a JSONL changed on both sides into the local file before upload. A merge
    of judgements that would not pair with the cell's generations is not written: held back when this
    machine's own judgements do not pair (_unpair), a conflict when only the Hub's would not (``force``
    then sends the local copy as it is)."""
    if item.action != MERGE:
        return
    staged, _ = _download(remote, item, staging)
    try:
        merged = merge_jsonl_by_key(item.local, staged, kind=merge_kind(item.path))
    except ValueError as e:
        if _changed_since_plan(item):   # e.g. a line still being written: neither forced nor kept aside
            _leave_busy(item, "push")
            return
        log.warning("sync push: cannot merge %s by key (%s)", item.path, e)
        item.action = FORCE if force else CONFLICT
        return
    if refused := _merge_refused(item, merged.rows):
        whose = "this machine's" if refused == "local" else "the Hub's"
        log.warning("sync push: merged, %s would hold judgements (%s) that do not pair with the cell's "
                    "generations; not merged", item.path, whose)
        if refused == "local":
            _unpair(item, report, _generations_of(_cell(item.path)))
        else:
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
        with _hub_errors(what):
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
    nothing: merges are planned, not tried (a planned merge counts as going up when push decides which
    judgements and AuditKIT records to hold back, and its judgements are checked as they stand here),
    and the Hub's own judgements and records that a push replacing generations would leave unpaired,
    which takes their contents, are not looked at."""
    selected = select(subdirs)
    remote = _connect(api)
    tree = _remote_tree(remote)
    state = _State.load(remote.repo)
    digest = _digest_memo()
    prefixes = [p for p, _ in selected]
    reports = {}
    for direction, plan in (("pull", _plan_pull), ("push", _plan_push)):
        items = plan(selected, tree, state, force=False, digest=digest)
        report = SyncReport(direction, remote.repo, remote.sha, prefixes, dry_run=True)
        if direction == "push":                                 # in push's order
            report.held = _hold_back(items, tree, digest, _judged_from)
            _hold_unpaired(items, report, records=False)
            report.held |= _hold_back(items, tree, digest, _recorded_from)
            _hold_unpaired(items, report, records=True)
        report.files = {i.path: i.action for i in items}
        reports[direction] = report
    return reports
