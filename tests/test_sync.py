"""lift.sync and ``lift sync`` against an in-memory fake of the HfApi calls sync makes, and against the
REAL huggingface_hub client talking to the same fake over HTTP (httpx2.MockTransport).

Offline: an autouse fixture puts huggingface_hub in offline mode and sets a dummy HF_TOKEN, so a call
that misses the fake fails fast instead of reaching the real Hub with this machine's credentials. The
HTTP tests lift offline mode only once every request goes to the mock transport.
Files hold benign placeholder text only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from lift import paths, sync
from lift.io import read_jsonl, stable_json

REPO = "tester/lift-runs"
GEN = "outputs/generations/stock/harmful_en_dev.jsonl"
JUD = "outputs/judgements/qwen3guard/stock/harmful_en_dev.jsonl"
CACHE = "outputs/judge_cache/frontier/openai__judge.jsonl"
LABELS = "validation/labels/hi_r1.csv"
ONE_PER_TARGET = (
    "data/prompts/harmful_en.jsonl",
    "data/sft/hi/manifest.json",
    "data/interim/translations/hi.jsonl",
    GEN,
    "validation/sheets/round1_hi.csv",
    "validation/keys/round1.json",
    LABELS,
)


@pytest.fixture(autouse=True)
def offline_hub(monkeypatch):
    """No test can reach the real Hub or use this machine's token."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)   # read at call time: unmocked calls raise
    monkeypatch.setattr(hf_constants, "HF_DEBUG", False)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_not_a_real_token")
    monkeypatch.setenv("LIFT_RUNS_REPO", REPO)


def hub_error(cls_name: str, status: int, method: str = "GET") -> Exception:
    httpx2 = pytest.importorskip("httpx2")
    from huggingface_hub import errors

    request = httpx2.Request(method, f"https://huggingface.co/api/datasets/{REPO}")
    response = httpx2.Response(status, request=request)
    return getattr(errors, cls_name)(f"{status} test error", response=response)


def whoami_answer(**access_token) -> dict:
    """HfApi.whoami()'s answer (the Hub's /api/whoami-v2) for a token with these accessToken fields."""
    return {"type": "user", "id": "0", "name": "tester", "orgs": [],
            "auth": {"type": "access_token", "accessToken": {"displayName": "test", **access_token}}}


def fine_grained(*scopes: tuple[str, str, list[str]]) -> dict:
    """A fine-grained token's accessToken fields; each scope is (entity type, entity name, permissions)."""
    scoped = [{"entity": {"_id": str(i), "type": kind, "name": name}, "permissions": perms}
              for i, (kind, name, perms) in enumerate(scopes)]
    return {"role": "fineGrained", "fineGrained": {"canReadGatedRepos": True, "global": [], "scoped": scoped}}


class FakeHub:
    """Stand-in for the HfApi methods lift.sync calls. Every commit is a snapshot keyed by its sha, so a
    read pinned to a revision sees exactly that commit. ``*.safetensors`` files are LFS entries.

    create_commit behaves like huggingface_hub 2.4's: operations whose bytes the head already holds are
    dropped; when nothing is left no commit is made and the CURRENT head's sha comes back without any
    parent_commit check; otherwise parent_commit must be the head (HTTP 412, as the Hub answers).
    """

    token = None

    def __init__(self, files: dict[str, bytes] | None = None, *, private: bool | None = True):
        self.private = private
        self.me: dict | None = whoami_answer(role="write")   # what whoami answers; None: HTTP 401
        self.snapshots: dict[str, dict[str, bytes]] = {}
        self.head = self._commit(dict(files or {}))
        self.calls: list[tuple[str, dict]] = []
        self.corrupt: set[str] = set()              # downloads of these come back with stale bytes
        self.before_commit: Callable[[FakeHub], None] | None = None   # runs once, before the next commit
        self.after_commit: Callable[[FakeHub], None] | None = None    # runs once, after the next commit lands
        self.after_list: Callable[[FakeHub], None] | None = None      # runs once, after the next listing

    def _commit(self, files: dict[str, bytes]) -> str:
        sha = hashlib.sha1(f"commit {len(self.snapshots)}".encode()).hexdigest()
        self.snapshots[sha] = files
        return sha

    @property
    def files(self) -> dict[str, bytes]:
        return self.snapshots[self.head]

    def put(self, files: dict[str, bytes]) -> str:
        """Another machine commits ``files``."""
        self.head = self._commit({**self.files, **files})
        return self.head

    def called(self, name: str) -> list[dict]:
        return [kwargs for n, kwargs in self.calls if n == name]

    def hook(self, name: str) -> None:
        hook = getattr(self, name)
        if hook is not None:
            setattr(self, name, None)
            hook(self)

    def commit(self, changed: dict[str, bytes], *, parent: str | None) -> str:
        """The Hub's commit endpoint: HTTP 412 unless ``parent`` is None or the head."""
        if parent is not None and parent != self.head:
            raise hub_error("HfHubHTTPError", 412, "POST")
        self.head = ours = self._commit({**self.files, **changed})
        self.hook("after_commit")
        return ours

    @staticmethod
    def entry(path: str, data: bytes):
        from huggingface_hub.hf_api import RepoFile
        from huggingface_hub.utils.sha import git_hash

        if path.endswith(".safetensors"):
            sha256 = hashlib.sha256(data).hexdigest()
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {len(data)}\n"
            return RepoFile(path=path, size=len(data), oid=git_hash(pointer.encode()),
                            lfs={"size": len(data), "oid": sha256, "pointerSize": len(pointer)})
        return RepoFile(path=path, size=len(data), oid=git_hash(data))

    def dataset_info(self, repo_id, *, expand=None, **kwargs):
        self.calls.append(("dataset_info", {"repo_id": repo_id, "expand": expand, **kwargs}))
        return SimpleNamespace(id=repo_id, private=self.private, sha=self.head)

    def whoami(self, token=None, **kwargs):
        self.calls.append(("whoami", {}))
        if self.me is None:
            raise hub_error("HfHubHTTPError", 401)
        return self.me

    def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, revision=None, repo_type=None,
                       **kwargs):
        from huggingface_hub.hf_api import RepoFolder

        self.calls.append(("list_repo_tree", {"repo_id": repo_id, "path_in_repo": path_in_repo,
                                              "recursive": recursive, "revision": revision,
                                              "repo_type": repo_type}))
        snapshot = self.snapshots[revision]
        folders = {"/".join(p.split("/")[:i]) for p in snapshot for i in range(1, p.count("/") + 1)}
        entries = [RepoFolder(path=f, oid="0" * 40) for f in sorted(folders)]
        entries += [self.entry(p, data) for p, data in sorted(snapshot.items())]
        self.hook("after_list")
        return iter(entries)

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, revision=None, local_dir=None, **kwargs):
        self.calls.append(("hf_hub_download", {"repo_id": repo_id, "filename": filename,
                                               "repo_type": repo_type, "revision": revision,
                                               "local_dir": local_dir}))
        data = self.snapshots[revision][filename]
        if filename in self.corrupt:   # what a silent fallback to an old local file looks like
            data = b"stale copy " + data
        dest = Path(local_dir, *filename.split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        meta = Path(local_dir, ".cache", "huggingface", "download", *filename.split("/"))
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.with_name(meta.name + ".metadata").write_text(f"{revision}\netag\n0\n")
        return str(dest)

    def create_commit(self, repo_id, operations, *, commit_message, repo_type=None, parent_commit=None,
                      **kwargs):
        from huggingface_hub import CommitOperationAdd

        ops = list(operations)
        self.calls.append(("create_commit", {"repo_id": repo_id, "repo_type": repo_type,
                                             "paths": [op.path_in_repo for op in ops],
                                             "sources": [op.path_or_fileobj for op in ops],
                                             "parent_commit": parent_commit, "commit_message": commit_message,
                                             "extra": kwargs}))
        assert all(isinstance(op, CommitOperationAdd) for op in ops)     # sync never deletes or copies
        self.hook("before_commit")
        changed = {}
        for op in ops:
            with op.as_file() as f:
                data = f.read()
            if self.files.get(op.path_in_repo) != data:
                changed[op.path_in_repo] = data
        if not changed:
            return SimpleNamespace(oid=self.head)
        return SimpleNamespace(oid=self.commit(changed, parent=parent_commit))

    def upload_folder(self, **kwargs):
        raise AssertionError("lift.sync must not call upload_folder: huggingface_hub 2.4 re-sends unchanged "
                             "files and guards only the first of its commits")


class HttpFront:
    """A FakeHub served over HTTP, so the REAL huggingface_hub client runs against it offline: dataset
    info, tree listing, resolve (downloads), preupload and commit. Regular files only; the commit
    endpoint enforces parentCommit with HTTP 412, like the Hub."""

    def __init__(self, hub: FakeHub):
        self.hub = hub
        self.commits: list[dict] = []     # every commit request: summary, parent, paths, status
        self.before_preupload: Callable[[FakeHub], None] | None = None   # runs once
        self.refuse_writes: int | None = None   # preupload and commit answer this (403: a read-only token)
        self.down = False                       # every request fails to connect

    def _sha(self, rev: str) -> str:
        return self.hub.head if rev == "main" else rev

    @staticmethod
    def _error(request, status: int, message: str, code: str | None = None):
        import httpx2

        headers = {"X-Error-Message": message, **({"X-Error-Code": code} if code else {})}
        return httpx2.Response(status, json={"error": message}, headers=headers, request=request)

    def handle(self, request):
        import base64

        import httpx2
        from huggingface_hub.utils.sha import git_hash

        hub, method, path = self.hub, request.method, request.url.path
        api, resolve = f"/api/datasets/{REPO}", f"/datasets/{REPO}/resolve/"
        if self.down:
            raise httpx2.ConnectError("connection refused", request=request)
        if method == "GET" and path == "/api/whoami-v2":
            if hub.me is None:
                return self._error(request, 401, "Invalid credentials in Authorization header")
            return httpx2.Response(200, json=hub.me, request=request)
        writes = (f"{api}/preupload/", f"{api}/commit/")
        if self.refuse_writes and method == "POST" and path.startswith(writes):
            return self._error(request, self.refuse_writes, "You don't have the rights to write to this repo")
        if path.startswith(resolve):
            rev, _, filename = path[len(resolve):].partition("/")
            data = hub.snapshots[self._sha(rev)].get(filename)
            if data is None:
                return self._error(request, 404, "Entry not found", "EntryNotFound")
            headers = {"X-Repo-Commit": self._sha(rev), "ETag": f'"{git_hash(data)}"',
                       "Content-Length": str(len(data))}
            if method == "HEAD":
                return httpx2.Response(200, headers=headers, request=request)
            return httpx2.Response(200, headers=headers, content=data, request=request)
        if not path.startswith(api):
            return self._error(request, 404, "Repository not found", "RepoNotFound")
        rest = path[len(api):]
        if method == "GET" and (rest == "" or rest.startswith("/revision/")):
            info = {"id": REPO, "sha": self._sha(rest[len("/revision/"):] if rest else "main"),
                    "private": hub.private}
            return httpx2.Response(200, json=info, request=request)
        if method == "GET" and rest.startswith("/tree/"):
            snapshot = hub.snapshots[self._sha(rest[len("/tree/"):].split("/")[0])]
            entries = [{"type": "file", "path": p, "size": len(d), "oid": git_hash(d)}
                       for p, d in sorted(snapshot.items())]
            return httpx2.Response(200, json=entries, request=request)
        if method == "POST" and rest.startswith("/preupload/"):
            if self.before_preupload is not None:
                hook, self.before_preupload = self.before_preupload, None
                hook(hub)
            files = [{"path": f["path"], "uploadMode": "regular", "shouldIgnore": False,
                      "oid": git_hash(hub.files[f["path"]]) if f["path"] in hub.files else None}
                     for f in json.loads(request.content)["files"]]
            return httpx2.Response(200, json={"files": files}, request=request)
        if method == "POST" and rest.startswith("/commit/"):
            items = [json.loads(line) for line in request.content.splitlines() if line.strip()]
            header, changed = items[0]["value"], {}
            for item in items[1:]:
                assert item["key"] == "file", item["key"]          # regular files; sync never deletes
                changed[item["value"]["path"]] = base64.b64decode(item["value"]["content"])
            hub.hook("before_commit")
            parent = header.get("parentCommit")
            ok = parent is None or parent == hub.head
            self.commits.append({"summary": header["summary"], "parent": parent, "paths": sorted(changed),
                                 "status": 200 if ok else 412})
            if not ok:
                return self._error(request, 412, "A commit has happened since. Please refresh and try again.")
            sha = hub.commit(changed, parent=parent)
            return httpx2.Response(200, json={"commitUrl": f"https://huggingface.co/datasets/{REPO}/commit/{sha}",
                                              "commitOid": sha}, request=request)
        return self._error(request, 400, f"unhandled {method} {path}")


@pytest.fixture
def http_front(monkeypatch):
    """An empty FakeHub behind the real huggingface_hub client; every request goes to a mock transport."""
    httpx2 = pytest.importorskip("httpx2")
    import huggingface_hub.constants as hf_constants
    from huggingface_hub.utils import set_client_factory
    from huggingface_hub.utils._http import default_client_factory, hf_request_event_hook

    front = HttpFront(FakeHub())
    set_client_factory(lambda: httpx2.Client(transport=httpx2.MockTransport(front.handle),
                                             event_hooks={"request": [hf_request_event_hook]},
                                             follow_redirects=True))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    try:
        yield front
    finally:
        set_client_factory(default_client_factory)


def local(path: str) -> Path:
    """The local file a canonical repo path maps to, under the current LIFT_*_DIR settings."""
    table = sync.targets()
    prefix = max((p for p in table if path.startswith(p + "/")), key=len)
    return table[prefix].joinpath(*path[len(prefix) + 1:].split("/"))


def write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def append(path: Path, data: bytes) -> None:
    with open(path, "ab") as f:
        f.write(data)


def lines(*rows: dict) -> bytes:
    return "".join(stable_json(r) + "\n" for r in rows).encode()


def row(key: str, **fields) -> dict:
    """A record every merge rule accepts: one variant/set/lang/split cell, its own pid, one run_config."""
    return {"key": key, "variant": "stock", "set": "harmful", "lang": "en", "split": "dev",
            "pid": f"pid-{key}", "output": f"placeholder {key}",
            "meta": {"dtype": "bf16", "run_config": "rc-bf16"}, **fields}


def records(*keys: str) -> bytes:
    return lines(*(row(k) for k in keys))


def gen_row(key: str, pid: str, *, dtype: str = "bf16", **fields) -> dict:
    """A GenerationRecord as lift.models.generate writes it."""
    return {"key": key, "variant": "stock", "pid": pid, "lang": "en", "set": "harmful", "split": "dev",
            "prompt_sha": "0" * 64, "output": "placeholder", "n_prompt_tokens": 10, "n_new_tokens": 20,
            "hit_cap": False, "seconds": 0.5,
            "meta": {"dtype": dtype, "gpu": "NVIDIA A100", "device": "cuda:0", "batch_size": 16,
                     "n_generated": 21, "run_config": f"rc-{dtype}"}, **fields}


def jud_row(key: str, pid: str, label: str, **fields) -> dict:
    """A JudgementRecord as lift.judge.pipeline writes it."""
    return {"key": key, "variant": "stock", "pid": pid, "lang": "en", "set": "harmful", "split": "dev",
            "judge": "Qwen/Qwen3Guard-Gen-4B@" + "a" * 40 + "#bf16#strict", "judge_label": label,
            "label": label, "native": {}, "raw": "",
            "flags": {"too_short": False, "repetitive": False, "garbled": False, "wrong_lang": False,
                      "code_mixed": False, "lid_top": []}, **fields}


def keys_of(path: Path) -> list[str]:
    return [r["key"] for r in read_jsonl(path)]


def keys_of_bytes(data: bytes) -> list[str]:
    return [json.loads(line)["key"] for line in data.decode().splitlines() if line.strip()]


def machine(root: Path, monkeypatch) -> Path:
    """Point the LIFT_* dirs at ``root``: one simulated machine with its own files and sync state."""
    for sub, var in (("data", "LIFT_DATA_DIR"), ("outputs", "LIFT_OUTPUT_DIR"),
                     ("validation", "LIFT_VALIDATION_DIR")):
        (root / sub).mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(var, str(root / sub))
    return root.resolve()


def state_file() -> Path:
    return paths.output_dir() / sync.STATE_FILE


def judged(gens: list[dict], label: str = "refusal", **fields) -> bytes:
    """One judgement per generation, each labelled ``label``, as lift.judge.pipeline writes them."""
    return lines(*(jud_row(g["key"], g["pid"], label, **fields) for g in gens))


def record_path(*, judge: str = "qwen3guard", dtype: str | None = None, text: str = "redacted") -> str:
    """The canonical repo path of the stock harmful_en_dev cell's AuditKIT record, as lift.audit names it."""
    from lift.audit import auditkit_path

    path = auditkit_path(judge, "stock", "harmful", "en", "dev", text=text, dtype=dtype)
    return "outputs/" + path.relative_to(paths.output_dir()).as_posix()


def audit_record(gens: list[dict], refusal_rate: float) -> bytes:
    """Stands in for audit.record_safety's record of a cell: its keys and a rate, no text."""
    return (json.dumps({"metadata": {"lift": {"refusal_rate": refusal_rate,
                                              "keys": [g["key"] for g in gens]}}}) + "\n").encode()


REAL_MAKE_API = sync.make_api


def serve(monkeypatch, hub: FakeHub) -> None:
    """Commands called without ``api`` (the CLI) get ``hub``, after the token check make_api makes."""
    def make_api():
        sync._require_token()
        return hub

    monkeypatch.setattr(sync, "make_api", make_api)


def raising(error: Callable[[], Exception]) -> Callable:
    def fail(*args, **kwargs):
        raise error()

    return fail


# ---------------------------------------------------------------------------- refusals and errors


@pytest.mark.parametrize("private", [False, None])
@pytest.mark.parametrize("command", ["push", "pull", "status"])
def test_refuses_a_repo_that_is_not_private(command, private):
    hub = FakeHub({GEN: records("a")}, private=private)
    write(local(GEN), records("b"))
    with pytest.raises(sync.SyncError, match="not private"):
        getattr(sync, command)(api=hub)
    assert [name for name, _ in hub.calls] == ["dataset_info"]       # nothing listed, sent or fetched
    assert hub.called("dataset_info")[0] == {"repo_id": REPO, "expand": ["private", "sha"]}
    assert keys_of(local(GEN)) == ["b"] and not state_file().exists()


def test_needs_the_runs_repo_env(monkeypatch):
    hub = FakeHub()
    monkeypatch.delenv("LIFT_RUNS_REPO")
    for command in (sync.push, sync.pull, sync.status):
        with pytest.raises(sync.SyncError, match="LIFT_RUNS_REPO is not set"):
            command(api=hub)
    for value, problem in (("lift-runs", "has no '/'"), ("tester/lift/runs", "has more than one '/'"),
                           ("/lift-runs", "an empty owner or name"), ("tester/", "an empty owner or name")):
        monkeypatch.setenv("LIFT_RUNS_REPO", value)
        with pytest.raises(sync.SyncError, match="<owner>/<name>") as info:
            sync.pull(api=hub)
        assert problem in str(info.value) and value not in str(info.value) and info.value.permanent
    assert hub.calls == []


def test_needs_a_token(monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(tmp_path / "no-token"))   # hide the token file
    hub = FakeHub()
    with pytest.raises(sync.SyncError, match="no Hugging Face token"):
        sync.push()
    with pytest.raises(sync.SyncError, match="no Hugging Face token"):
        sync.pull(api=hub)
    assert hub.calls == []


@pytest.mark.parametrize("cls_name,status,message,permanent", [
    ("RepositoryNotFoundError", 404, "not found or is not readable", True),
    ("RepositoryNotFoundError", 401, "not found or is not readable", True),
    ("GatedRepoError", 403, "not found or is not readable", True),
    ("HfHubHTTPError", 401, "token was rejected", True),
    ("HfHubHTTPError", 403, "WRITE token", True),
    ("DisabledRepoError", 403, "disabled on the Hub", True),
    ("HfHubHTTPError", 503, "HTTP 503", False),
    ("HfHubHTTPError", 429, "HTTP 429", False),
    ("HfHubHTTPError", 412, "changed meanwhile", False),     # outside a push, a rerun plans again
])
def test_hub_errors_become_actionable_sync_errors(cls_name, status, message, permanent):
    hub = FakeHub()
    hub.dataset_info = raising(lambda: hub_error(cls_name, status))
    with pytest.raises(sync.SyncError, match=message) as info:
        sync.pull(api=hub)
    assert info.value.permanent is permanent and "hf_dummy" not in str(info.value)


def test_an_unmocked_call_cannot_reach_the_hub():
    with pytest.raises(sync.SyncError, match="offline mode"):
        sync.pull()


def test_refuses_to_run_with_hf_debug(monkeypatch):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_DEBUG", True)
    hub = FakeHub()
    with pytest.raises(sync.SyncError, match="HF_DEBUG"):
        sync.push(api=hub)
    assert hub.calls == []


# ---------------------------------------------------------------------------- permanent errors and preflight


def test_a_sync_error_is_transient_unless_marked_permanent():
    assert sync.SyncError("rerun").permanent is False
    assert sync.SyncError("fix it", permanent=True).permanent is True


def _token_unset(hub, monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_OIDC_RESOURCE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(tmp_path / "no-token"))   # hide the token file


def _hf_debug(hub, monkeypatch, tmp_path):
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_DEBUG", True)


def _real_client(hub, monkeypatch, tmp_path):
    """The CLI gets the real HfApi (offline here) instead of ``hub``."""
    monkeypatch.setattr(sync, "make_api", REAL_MAKE_API)


def _keeps_moving(hub, monkeypatch, tmp_path):
    commit = hub.create_commit

    def overtaken(*args, **kwargs):
        hub.put({"outputs/other.jsonl": records(str(len(hub.snapshots)))})
        return commit(*args, **kwargs)

    hub.create_commit = overtaken


def _answer(method: str, error: Callable[[], Exception]) -> Callable:
    return lambda hub, monkeypatch, tmp_path: setattr(hub, method, raising(error))


def _http(cls_name: str, status: int, method: str = "GET") -> Callable[[], Exception]:
    return lambda: hub_error(cls_name, status, method)


def _network(name: str) -> Callable[[], Exception]:
    return lambda: getattr(pytest.importorskip("httpx2"), name)("placeholder network error")


def _unreachable() -> Exception:
    from huggingface_hub.errors import LocalEntryNotFoundError

    return LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")


PERMANENT, TRANSIENT = 2, 1
EXIT_CASES = [
    pytest.param(["pull"], _answer("dataset_info", _http("RepositoryNotFoundError", 404)), PERMANENT,
                 id="repo-missing-404"),
    pytest.param(["status"], _answer("dataset_info", _http("RepositoryNotFoundError", 401)), PERMANENT,
                 id="repo-unseen-401"),
    pytest.param(["pull"], _answer("dataset_info", _http("GatedRepoError", 403)), PERMANENT, id="gated-403"),
    pytest.param(["pull"], _answer("dataset_info", _http("HfHubHTTPError", 401)), PERMANENT,
                 id="token-rejected-401"),
    pytest.param(["push"], _answer("create_commit", _http("HfHubHTTPError", 403, "POST")), PERMANENT,
                 id="read-only-token-403"),
    pytest.param(["push"], lambda hub, *_: setattr(hub, "private", False), PERMANENT, id="public"),
    pytest.param(["pull"], lambda hub, *_: setattr(hub, "private", None), PERMANENT, id="privacy-unknown"),
    pytest.param(["push"], _token_unset, PERMANENT, id="token-unset"),
    pytest.param(["pull"], lambda hub, mp, _: mp.delenv("LIFT_RUNS_REPO"), PERMANENT, id="runs-repo-unset"),
    pytest.param(["pull"], lambda hub, mp, _: mp.setenv("LIFT_RUNS_REPO", "lift-runs"), PERMANENT,
                 id="runs-repo-malformed"),
    pytest.param(["push"], _hf_debug, PERMANENT, id="hf-debug"),
    pytest.param(["pull"], _real_client, PERMANENT, id="offline-mode"),
    pytest.param(["pull", "--only", "outputs/../src"], None, PERMANENT, id="bad-path"),
    pytest.param(["push", "--only", "outputs/.sync-backup"], None, PERMANENT, id="never-synced-path"),
    pytest.param(["pull"], lambda *_: write(state_file(), "{not json"), PERMANENT, id="unreadable-state"),
    pytest.param(["pull"], lambda hub, *_: hub.put({"outputs/../evil.txt": b"x"}), PERMANENT,
                 id="unsafe-remote-path"),
    pytest.param(["push"], _answer("dataset_info", _http("HfHubHTTPError", 503)), TRANSIENT,
                 id="hub-down-503"),
    pytest.param(["pull"], _answer("list_repo_tree", _http("HfHubHTTPError", 429)), TRANSIENT,
                 id="rate-limited-429"),
    pytest.param(["pull"], _answer("list_repo_tree", _http("HfHubHTTPError", 412)), TRANSIENT,
                 id="moved-412-outside-push"),
    pytest.param(["pull"], _answer("dataset_info", _network("ConnectError")), TRANSIENT,
                 id="connection-refused"),
    pytest.param(["push"], _answer("create_commit", _network("ReadTimeout")), TRANSIENT, id="read-timeout"),
    pytest.param(["pull"], _answer("hf_hub_download", _unreachable), TRANSIENT, id="download-unreachable"),
    pytest.param(["pull"], lambda hub, *_: setattr(hub, "corrupt", {LABELS}), TRANSIENT, id="download-stale"),
    pytest.param(["push"], _keeps_moving, TRANSIENT, id="keeps-moving-412"),
    pytest.param(["push"], lambda hub, *_: setattr(hub, "create_commit", lambda *a, **k: SimpleNamespace()),
                 TRANSIENT, id="no-commit-sha"),
]


@pytest.mark.parametrize("argv,setup,code", EXIT_CASES)
def test_the_exit_code_says_whether_a_rerun_can_help(argv, setup, code, monkeypatch, tmp_path, capsys):
    """2 for what no rerun fixes, so runners stop at once instead of idling the GPU on retries; 1 for
    a transient or unknown error, which a rerun resumes."""
    hub = FakeHub({LABELS: b"v1\n"})
    write(local(GEN), records("a"))
    serve(monkeypatch, hub)
    if setup is not None:
        setup(hub, monkeypatch, tmp_path)
    assert run_cli(["sync", *argv]) == code
    err = capsys.readouterr().err
    assert err.startswith(f"lift sync {argv[0]}: ") and "Traceback" not in err and "hf_dummy" not in err


CAN_WRITE = [
    pytest.param(whoami_answer(role="write"), id="write-token"),
    pytest.param(whoami_answer(**fine_grained(("user", "tester", ["repo.content.read", "repo.write"]))),
                 id="fine-grained-owner"),
    pytest.param(whoami_answer(**fine_grained(("dataset", REPO, ["repo.content.read", "repo.write"]))),
                 id="fine-grained-repo"),
    pytest.param(whoami_answer(**fine_grained(("dataset", REPO.title(), ["repo.write"]))), id="other-case"),
    pytest.param(whoami_answer(**fine_grained(("org", "tester", ["repo.write"]))), id="fine-grained-org"),
    pytest.param(whoami_answer(**fine_grained(("resource-group", "rg", ["repo.write"]))),
                 id="scope-it-cannot-place"),
    pytest.param(whoami_answer(role="fineGrained"), id="no-permissions-shown"),
    pytest.param(whoami_answer(role="fineGrained", fineGrained={"scoped": [], "global": 7}), id="odd-global"),
    pytest.param(whoami_answer(role="fineGrained", fineGrained={"scoped": [{"entity": "x"}]}),
                 id="odd-scope"),
    pytest.param(whoami_answer(role="fineGrained", fineGrained={"scoped": [], "global": ["repo.write"]}),
                 id="global-write"),
    pytest.param(whoami_answer(), id="no-role"),
    pytest.param({}, id="empty-answer"),
]
CANNOT_WRITE = [
    pytest.param(whoami_answer(role="read"), id="read-token"),
    pytest.param(whoami_answer(**fine_grained(("user", "tester", ["repo.content.read"]))),
                 id="read-only-scope"),
    pytest.param(whoami_answer(**fine_grained(("dataset", "tester/other-runs", ["repo.write"]),
                                              ("model", REPO, ["repo.write"]),
                                              ("user", "someone-else", ["repo.write"]))),
                 id="write-elsewhere"),
    pytest.param(whoami_answer(**fine_grained()), id="no-scopes"),
]


@pytest.mark.parametrize("answer", CAN_WRITE)
def test_preflight_passes_a_token_whoami_does_not_show_as_read_only(answer):
    hub = FakeHub()
    hub.me = answer
    sync.preflight(api=hub)
    assert [name for name, _ in hub.calls] == ["dataset_info", "whoami"]     # nothing listed, fetched or sent


@pytest.mark.parametrize("answer", CANNOT_WRITE)
def test_preflight_refuses_a_token_that_cannot_write(answer):
    """A read token is enough for the gated models and for `lift sync pull`, so a job starts normally and
    fails only at its first push, hours of GPU later. The preflight stops it before any GPU line."""
    hub = FakeHub()
    hub.me = answer
    with pytest.raises(sync.SyncError, match="WRITE token") as info:
        sync.preflight(api=hub)
    assert info.value.permanent and "hf_dummy" not in str(info.value)
    sync.preflight(write=False, api=hub)                  # reading is all a pull needs
    assert [name for name, _ in hub.calls] == ["dataset_info", "whoami", "dataset_info"]


@pytest.mark.parametrize("setup,permanent", [
    pytest.param(lambda hub, *_: setattr(hub, "private", False), True, id="public"),
    pytest.param(lambda hub, *_: setattr(hub, "private", None), True, id="privacy-unknown"),
    pytest.param(_answer("dataset_info", _http("RepositoryNotFoundError", 404)), True, id="repo-missing"),
    pytest.param(lambda hub, *_: setattr(hub, "me", None), True, id="token-rejected"),
    pytest.param(_token_unset, True, id="token-unset"),
    pytest.param(lambda hub, mp, _: mp.delenv("LIFT_RUNS_REPO"), True, id="runs-repo-unset"),
    pytest.param(_hf_debug, True, id="hf-debug"),
    pytest.param(_answer("whoami", _http("HfHubHTTPError", 429)), False, id="whoami-rate-limited"),
    pytest.param(_answer("whoami", _network("ConnectError")), False, id="whoami-unreachable"),
    pytest.param(_answer("dataset_info", _http("HfHubHTTPError", 502)), False, id="hub-down"),
])
def test_preflight_errors_say_whether_a_rerun_can_help(setup, permanent, monkeypatch, tmp_path):
    hub = FakeHub()
    setup(hub, monkeypatch, tmp_path)
    with pytest.raises(sync.SyncError) as info:
        sync.preflight(api=hub)
    assert info.value.permanent is permanent and "hf_dummy" not in str(info.value)
    assert {name for name, _ in hub.calls} <= {"dataset_info", "whoami"}


PASTED_TOKEN = "hf_" + "Q" * 34       # shaped like a user access token; not a real one
PASTED = [
    pytest.param(PASTED_TOKEN, None, id="the-token"),
    pytest.param(f"tester/lift-runs/{PASTED_TOKEN}", None, id="two-slashes"),
    pytest.param(f"{PASTED_TOKEN}/lift-runs", _answer("dataset_info", _http("RepositoryNotFoundError", 404)),
                 id="token-as-owner-404"),
    pytest.param(f"tester/{PASTED_TOKEN}", _answer("dataset_info", _http("GatedRepoError", 403)),
                 id="token-as-name-403"),
    pytest.param(f"tester/{PASTED_TOKEN} lift-runs", _real_client,     # HFValidationError, the value inside
                 id="invalid-repo-id"),
]


@pytest.mark.parametrize("argv", [["push"], ["pull"], ["status"], ["check"], ["check", "--write"]])
@pytest.mark.parametrize("value,setup", PASTED)
def test_a_token_pasted_into_lift_runs_repo_is_never_shown(argv, value, setup, monkeypatch, tmp_path, capsys,
                                                           caplog):
    """Kaggle reads HF_TOKEN and LIFT_RUNS_REPO from its secrets, so a token can land in the wrong slot.
    Its runner masks nothing and Kaggle keeps every printed line with the notebook version. No sync
    command shows the value: errors say what is wrong with it instead, and a value is named only once
    the Hub has answered for it as a repo."""
    hub = FakeHub()
    serve(monkeypatch, hub)
    monkeypatch.setenv("LIFT_RUNS_REPO", value)
    if setup is not None:
        setup(hub, monkeypatch, tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert run_cli(["sync", *argv]) == 2
    out, err = capsys.readouterr()
    shown = out + err + caplog.text
    assert PASTED_TOKEN[:12] not in shown and "QQQQ" not in shown
    assert err.startswith(f"lift sync {argv[0]}: ") and "LIFT_RUNS_REPO" in err and "Traceback" not in err


# ---------------------------------------------------------------------------- selection


def test_select_maps_repo_paths_to_local_dirs():
    out = paths.output_dir()
    assert sync.select(["outputs/generations"]) == [("outputs/generations", out / "generations")]
    assert sync.select(["outputs\\adapters\\hi\\"]) == [("outputs/adapters/hi", out / "adapters" / "hi")]
    assert [p for p, _ in sync.select(["validation"])] == ["validation/keys", "validation/labels",
                                                          "validation/sheets"]
    assert [p for p, _ in sync.select(["data", "data/sft/hi"])] == ["data/interim/translations",
                                                                   "data/prompts", "data/sft"]
    assert [p for p, _ in sync.select(["outputs/generations", "outputs"])] == ["outputs"]
    assert sync.select() == sorted(sync.targets().items())


@pytest.mark.parametrize("bad", ["outputs/../src", "..", "outputs/./x", "outputs//x", "", "/", "src/lift",
                                 "data/raw", "data/interim/curator", "Outputs", "outputs/.sync-backup",
                                 "outputs/.sync-backup/20260101T000000Z", "outputs/.cache", "outputs/.aside",
                                 "outputs/.aside/20260101T000000Z/generations"])
def test_select_refuses_paths_outside_the_synced_trees(bad):
    with pytest.raises(sync.SyncError):
        sync.select([bad])


def test_only_moves_the_selected_paths():
    hub = FakeHub({GEN: records("a"), LABELS: b"id,label\n"})
    assert list(sync.pull(["outputs/generations"], api=hub).files) == [GEN]
    assert not local(LABELS).exists()
    write(local("validation/keys/round1.json"), "{}")
    write(local("outputs/metrics/gate0.json"), "{}")
    sync.push(["validation\\keys"], api=hub)
    assert [c["paths"] for c in hub.called("create_commit")] == [["validation/keys/round1.json"]]
    assert "outputs/metrics/gate0.json" not in hub.files


# ---------------------------------------------------------------------------- push


@pytest.mark.parametrize("layout", ["default", "overrides"])
def test_push_maps_every_target_to_its_posix_prefix(layout, tmp_path, monkeypatch):
    if layout == "overrides":   # dirs whose names say nothing about the repo layout (e.g. on Kaggle)
        monkeypatch.setenv("LIFT_DATA_DIR", str(tmp_path / "kaggle" / "d"))
        monkeypatch.setenv("LIFT_OUTPUT_DIR", str(tmp_path / "kaggle" / "o"))
        monkeypatch.setenv("LIFT_VALIDATION_DIR", str(tmp_path / "elsewhere" / "v"))
    for path in ONE_PER_TARGET:
        write(local(path), f"placeholder {path}\n")
    write(paths.raw_dir() / "polyrefuse" / "x.json", "{}")          # data/ outside the targets
    write(paths.interim_dir() / "curator" / "x.json", "{}")
    hub = FakeHub()
    report = sync.push(api=hub)
    (commit,) = hub.called("create_commit")
    assert commit["repo_id"] == REPO and commit["repo_type"] == "dataset" and commit["extra"] == {}
    assert commit["paths"] == sorted(ONE_PER_TARGET)
    assert [Path(s) for s in commit["sources"]] == [local(p) for p in sorted(ONE_PER_TARGET)]
    assert commit["parent_commit"] == report.revision
    assert sorted(hub.files) == sorted(ONE_PER_TARGET)
    assert report.ok and report.files == {p: "new" for p in ONE_PER_TARGET}
    assert report.commits == [hub.head]


def test_push_leaves_caches_temp_files_and_the_state_file_behind():
    out = paths.output_dir()
    write(out / "artifacts" / "v.json", "{}")
    for junk in (".cache/huggingface/download/x.metadata", "artifacts/.cache/torch/k.bin", ".git/HEAD",
                 "generations/stock/a.jsonl123.tmp", "x.lock", "y.incomplete", "pkg/__pycache__/m.pyc",
                 ".DS_Store", "metrics/.DS_Store", ".sync-backup/20260101T000000Z/validation/labels/x.csv",
                 ".aside/20260101T000000Z/generations/stock/harmful_en_dev.jsonl"):
        write(out / junk, "junk")
    hub = FakeHub()
    assert sync.push(api=hub).files == {"outputs/artifacts/v.json": "new"}
    assert sorted(hub.files) == ["outputs/artifacts/v.json"] and state_file().is_file()
    assert sync.push(api=hub).files == {"outputs/artifacts/v.json": "same"}   # the state file stays local
    assert len(hub.called("create_commit")) == 1                              # nothing new: no call


def test_files_lift_restart_set_aside_are_never_synced():
    """lift restart moves a stale cell's files to outputs/.aside/<stamp>/<path> and leaves each one empty
    in place. The next push replaces the Hub's copies with the emptied files (a change made here only);
    the files set aside stay on this machine, both ways."""
    from lift.models.generate import ASIDE_DIR

    assert sync.ASIDE_DIR == ASIDE_DIR == ".aside"
    hub = FakeHub({GEN: records("a"), JUD: records("a")})
    sync.pull(api=hub)
    aside = paths.output_dir() / ".aside" / "20261104T000000Z"
    for path in (GEN, JUD):
        write(aside.joinpath(*path.split("/")[1:]), local(path).read_bytes())
        write(local(path), b"")
    assert sync.status(api=hub)["push"].files == {GEN: "update", JUD: "update"}
    assert sync.push(api=hub).files == {GEN: "update", JUD: "update"}
    assert hub.files[GEN] == hub.files[JUD] == b"" and not [p for p in hub.files if ".aside" in p]
    hub.put({"outputs/.aside/20261104T000000Z/generations/stock/x.jsonl": records("b")})
    assert sync.pull(api=hub).files == {GEN: "same", JUD: "same"}
    assert not aside.joinpath("generations", "stock", "x.jsonl").exists()
    with pytest.raises(sync.SyncError, match="never synced"):
        sync.select(["outputs/.aside"])


def test_push_skips_missing_and_empty_dirs():
    (paths.validation_dir() / "labels").mkdir(parents=True)
    write(paths.validation_dir() / "keys" / ".cache" / "x", "junk")
    write(local("outputs/a.jsonl"), "x\n")
    hub = FakeHub()
    sync.push(api=hub)
    assert [c["paths"] for c in hub.called("create_commit")] == [["outputs/a.jsonl"]]


def test_push_uploads_a_local_edit():
    hub = FakeHub({LABELS: b"id,label\n1,refusal\n"})
    sync.pull(api=hub)
    write(local(LABELS), "id,label\n1,benign\n")
    assert sync.push(api=hub).files == {LABELS: "update"}
    assert hub.files[LABELS] == b"id,label\n1,benign\n"
    assert sync.pull(api=hub).files == {LABELS: "same"}


def test_push_never_puts_a_stale_copy_over_a_newer_one():
    hub = FakeHub({GEN: records("a")})
    sync.pull(api=hub)
    hub.put({GEN: records("a", "b")})          # another machine appended and pushed
    report = sync.push(api=hub)
    assert report.files == {GEN: "stale"} and report.ok
    assert sync.push(api=hub, force=True).files == {GEN: "stale"}     # force settles conflicts only
    assert hub.files[GEN] == records("a", "b") and not hub.called("create_commit")


def test_push_conflict_keeps_the_hub_copy_and_uploads_the_local_one_aside():
    hub = FakeHub({LABELS: b"v1\n"})
    sync.pull(api=hub)
    hub.put({LABELS: b"v2 from another machine\n"})
    write(local(LABELS), "v1 edited here\n")
    write(local("validation/labels/other.csv"), "new\n")
    report = sync.push(api=hub)
    assert not report.ok and report.conflicts == [LABELS]
    assert report.files["validation/labels/other.csv"] == "new"
    assert hub.files[LABELS] == b"v2 from another machine\n"
    copy = report.conflict_copies[LABELS]
    assert re.fullmatch(r"conflicts/\d{8}T\d{6}Z/validation/labels/hi_r1\.csv", copy)
    assert hub.files[copy] == b"v1 edited here\n" and local(LABELS).read_bytes() == b"v1 edited here\n"
    (commit,) = hub.called("create_commit")
    assert commit["paths"] == [copy, "validation/labels/other.csv"]     # never the conflicting file itself
    again = sync.push(api=hub)                 # an identical copy is not uploaded twice
    assert again.conflict_copies == {LABELS: copy}
    assert sum(p.startswith("conflicts/") for p in hub.files) == 1
    forced = sync.push(api=hub, force=True)
    assert forced.ok and forced.files[LABELS] == "force" and hub.files[LABELS] == b"v1 edited here\n"
    assert sync.pull(api=hub).files[LABELS] == "same"


def test_a_conflict_copy_never_replaces_another_one(monkeypatch):
    """Another machine kept its own copy of the file aside in the same second: ours gets a new folder."""
    stamp = "20260101T000000Z"
    theirs = f"conflicts/{stamp}/{LABELS}"
    hub = FakeHub({LABELS: b"v1\n", theirs: b"their copy\n"})
    sync.pull(api=hub)
    hub.put({LABELS: b"v2 from another machine\n"})
    write(local(LABELS), "v1 edited here\n")
    monkeypatch.setattr(sync, "_stamp", lambda: stamp)
    report = sync.push(api=hub)
    ours = f"conflicts/{stamp}-1/{LABELS}"
    assert report.conflict_copies == {LABELS: ours}
    assert hub.files[theirs] == b"their copy\n" and hub.files[ours] == b"v1 edited here\n"


def test_pull_force_never_replaces_an_earlier_backup(monkeypatch):
    stamp = "20260101T000000Z"
    earlier = write(paths.output_dir() / ".sync-backup" / stamp / "validation" / "labels" / "hi_r1.csv",
                    "earlier\n")
    hub = FakeHub({LABELS: b"v1\n"})
    sync.pull(api=hub)
    write(local(LABELS), "edited here\n")
    hub.put({LABELS: b"v2\n"})
    monkeypatch.setattr(sync, "_stamp", lambda: stamp)
    forced = sync.pull(api=hub, force=True)
    assert earlier.read_text() == "earlier\n"
    assert Path(forced.backups[LABELS]).read_text() == "edited here\n"
    assert Path(forced.backup_dir).name == f"{stamp}-1"


def test_push_merges_jsonl_changed_on_both_sides_by_key():
    hub = FakeHub({GEN: records("a", "b")})
    sync.pull(api=hub)
    hub.put({GEN: records("a", "b", "r1")})
    append(local(GEN), records("l1", "l2"))
    report = sync.push(api=hub)
    assert report.files == {GEN: "merge"} and report.ok
    assert keys_of(local(GEN)) == ["a", "b", "l1", "l2", "r1"]
    assert hub.files[GEN] == local(GEN).read_bytes()
    assert sync.status(api=hub)["pull"].files == {GEN: "same"}


def test_push_plans_again_when_another_machine_pushes_meanwhile():
    hub = FakeHub({GEN: records("a")})
    sync.pull(api=hub)
    append(local(GEN), records("l1"))
    hub.before_commit = lambda h: h.put({GEN: records("a", "r1")})   # lands between our listing and commit
    report = sync.push(api=hub)
    assert report.files == {GEN: "merge"}
    assert keys_of(local(GEN)) == ["a", "l1", "r1"] and hub.files[GEN] == local(GEN).read_bytes()
    first, second = (c["parent_commit"] for c in hub.called("create_commit"))
    assert first != second


def test_push_gives_up_when_the_repo_keeps_moving():
    hub = FakeHub()
    write(local(GEN), records("a"))
    commit = hub.create_commit

    def busy(*args, **kwargs):
        hub.put({"outputs/other.jsonl": records(str(len(hub.snapshots)))})
        return commit(*args, **kwargs)

    hub.create_commit = busy
    with pytest.raises(sync.SyncError, match="kept changing"):
        sync.push(api=hub)
    assert len(hub.called("create_commit")) == sync.PUSH_ATTEMPTS and GEN not in hub.files


# ---------------------------------------------------------------------------- push: guarded commits


def test_push_never_reverts_a_concurrent_push_between_its_commits(tmp_path, monkeypatch):
    """A push of many files goes up in several commits. Another machine's push landing between two of
    them must survive: upload_folder re-sends files it never changed and guards only its first commit,
    so its later commits put this machine's older copy back. Every commit must be guarded instead."""
    gens = {f"outputs/generations/stock/f{i}.jsonl": records(f"k{i}") for i in range(6)}
    hub = FakeHub({**gens, JUD: records("j1")})
    monkeypatch.setattr(sync, "COMMIT_FILES", 1)
    a, b = tmp_path / "A", tmp_path / "B"
    machine(b, monkeypatch)
    sync.pull(api=hub)
    append(local(JUD), records("jB"))                                 # B judged more, pushes mid-way
    machine(a, monkeypatch)
    sync.pull(api=hub)
    for i in (0, 5):
        append(local(f"outputs/generations/stock/f{i}.jsonl"), records(f"a{i}"))

    def b_pushes(h):
        machine(b, monkeypatch)
        assert sync.push(api=h).files[JUD] == "update"
        machine(a, monkeypatch)

    hub.after_commit = b_pushes
    report = sync.push(api=hub)
    assert keys_of_bytes(hub.files[JUD]) == ["j1", "jB"], "A's push reverted B's judgements"
    assert report.ok and report.files[JUD] == "stale"
    for i in (0, 5):
        path = f"outputs/generations/stock/f{i}.jsonl"
        assert hub.files[path] == local(path).read_bytes()
    assert all(c["parent_commit"] for c in hub.called("create_commit"))
    machine(b, monkeypatch)
    assert sync.pull(api=hub).files[JUD] == "same" and keys_of(local(JUD)) == ["j1", "jB"]


def test_push_commits_exactly_the_planned_files_in_guarded_parts(monkeypatch):
    """Only new, updated, merged and forced files (and conflict copies) go up, never one that is the
    same or newer on the Hub. A push split into parts chains parent_commit through each part's own
    commit, and lists that commit before sending the next part."""
    names = [f"outputs/generations/stock/f{i}.jsonl" for i in range(6)]
    hub = FakeHub({p: records(f"k{i}") for i, p in enumerate(names)})
    sync.pull(api=hub)
    for p in names[:3]:
        append(local(p), records("here"))
    write(local("outputs/metrics/new.json"), "{}")
    hub.put({names[5]: records("k5", "theirs")})                      # newer on the Hub: stale here
    monkeypatch.setattr(sync, "COMMIT_FILES", 2)
    listed = len(hub.called("list_repo_tree"))
    report = sync.push(api=hub)
    commits = hub.called("create_commit")
    assert [c["paths"] for c in commits] == [names[:2], [names[2], "outputs/metrics/new.json"]]
    assert [c["parent_commit"] for c in commits] == [report.revision, report.commits[0]]
    assert [c["commit_message"] for c in commits] == ["lift sync push outputs (part 1/2)",
                                                      "lift sync push outputs (part 2/2)"]
    assert report.commits == [commits[1]["parent_commit"], hub.head]
    assert [c["revision"] for c in hub.called("list_repo_tree")[listed:]] == [report.revision,
                                                                              report.commits[0]]
    assert report.files[names[5]] == "stale" and hub.files[names[5]] == records("k5", "theirs")
    assert all(hub.files[p] == local(p).read_bytes() for p in names[:3])


def test_a_part_with_nothing_new_does_not_end_the_guard(monkeypatch):
    """When every file of a commit is already on the Hub, huggingface_hub makes no commit and returns
    the CURRENT head without checking parent_commit. The next part must not take that head on trust:
    here another machine pushed the same generations and its own labels, which must survive."""
    hub = FakeHub({GEN: records("a"), LABELS: b"v1\n"})
    monkeypatch.setattr(sync, "COMMIT_FILES", 1)
    sync.pull(api=hub)
    append(local(GEN), records("b"))
    write(local(LABELS), "v1 edited here\n")
    hub.before_commit = lambda h: h.put({GEN: local(GEN).read_bytes(), LABELS: b"v2 from another machine\n"})
    report = sync.push(api=hub)
    assert hub.files[LABELS] == b"v2 from another machine\n", "the push overwrote a change it never saw"
    assert report.files == {GEN: "same", LABELS: "conflict"}
    assert hub.files[report.conflict_copies[LABELS]] == b"v1 edited here\n"


def test_parts_hold_at_most_commit_files_and_commit_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "COMMIT_FILES", 3)
    monkeypatch.setattr(sync, "COMMIT_BYTES", 100)
    uploads = []
    for i, size in enumerate([40, 40, 40, 150, 10, 10, 10, 10]):
        f = write(tmp_path / f"f{i}", b"x" * size)
        uploads.append(sync._Upload(f"outputs/f{i}", f, sync.file_digest(f), None))
    parts = [[u.path.split("/")[1] for u in part] for part in sync._parts(uploads)]
    assert parts == [["f0", "f1"], ["f2"], ["f3"], ["f4", "f5", "f6"], ["f7"]]


def test_real_client_push_merges_a_push_landing_between_its_parts(http_front, monkeypatch):
    """The REAL huggingface_hub client, HTTP faked: every commit request carries parentCommit, nothing
    unchanged is sent, and another machine's commit between two parts gets HTTP 412, so the push plans
    again and merges instead of overwriting."""
    from huggingface_hub import HfApi

    hub = http_front.hub
    hub.put({**{f"outputs/generations/stock/f{i}.jsonl": records(f"k{i}") for i in range(4)},
             CACHE: records("c1")})
    sync.pull(api=HfApi())
    append(local("outputs/generations/stock/f0.jsonl"), records("a0"))
    append(local(CACHE), records("cA"))
    monkeypatch.setattr(sync, "COMMIT_FILES", 1)
    hub.after_commit = lambda h: h.put({CACHE: records("c1", "cB")})     # lands after our first part
    report = sync.push(api=HfApi())
    assert keys_of_bytes(hub.files[CACHE]) == ["c1", "cA", "cB"]
    assert report.ok and report.files[CACHE] == "merge"
    assert [(c["paths"], c["status"]) for c in http_front.commits] == [
        (["outputs/generations/stock/f0.jsonl"], 200), ([CACHE], 412), ([CACHE], 200)]
    assert all(c["parent"] for c in http_front.commits)
    assert sync.pull(api=HfApi()).files[CACHE] == "same"


def test_real_client_part_with_nothing_new_does_not_end_the_guard(http_front, monkeypatch):
    from huggingface_hub import HfApi

    hub = http_front.hub
    hub.put({GEN: records("a"), LABELS: b"v1\n"})
    sync.pull(api=HfApi())
    append(local(GEN), records("b"))
    write(local(LABELS), "v1 edited here\n")
    monkeypatch.setattr(sync, "COMMIT_FILES", 1)
    http_front.before_preupload = lambda h: h.put({GEN: local(GEN).read_bytes(),
                                                   LABELS: b"v2 from another machine\n"})
    report = sync.push(api=HfApi())
    assert hub.files[LABELS] == b"v2 from another machine\n"
    assert report.files == {GEN: "same", LABELS: "conflict"}
    copy = report.conflict_copies[LABELS]
    assert hub.files[copy] == b"v1 edited here\n"
    assert [c["paths"] for c in http_front.commits] == [[copy]]        # nothing else was ever sent


def test_real_client_errors_say_whether_a_rerun_can_help(http_front):
    from huggingface_hub import HfApi

    hub = http_front.hub
    write(local(GEN), records("a"))
    http_front.refuse_writes = 403                                     # a read-only token
    with pytest.raises(sync.SyncError, match="WRITE token") as info:
        sync.push(api=HfApi())
    assert info.value.permanent and GEN not in hub.files
    http_front.refuse_writes, http_front.down = None, True            # the connection drops
    with pytest.raises(sync.SyncError, match="could not reach the Hub") as info:
        sync.push(api=HfApi())
    assert not info.value.permanent


def test_real_client_preflight_reads_whoami(http_front):
    from huggingface_hub import HfApi

    hub = http_front.hub
    hub.me = whoami_answer(role="read")
    with pytest.raises(sync.SyncError, match="WRITE token") as info:
        sync.preflight(api=HfApi())
    assert info.value.permanent
    hub.me = whoami_answer(**fine_grained(("dataset", REPO, ["repo.content.read", "repo.write"])))
    sync.preflight(api=HfApi())
    hub.me = None                                                      # the Hub rejects the token
    with pytest.raises(sync.SyncError, match="HTTP 401") as info:
        sync.preflight(api=HfApi())
    assert info.value.permanent


# ---------------------------------------------------------------------------- judgements follow generations


FP16_JUDGE = "Qwen/Qwen3Guard-Gen-4B@" + "a" * 40 + "#fp16#strict"


def test_judgements_never_go_up_without_their_generations(tmp_path, monkeypatch):
    """A Colab bf16 job and a Kaggle fp16 job of one variant write the same paths. Kaggle's generations
    conflict with the ones Colab pushed first, so its judgements of them must not go up either: the Hub
    would pair Colab's generations with Kaggle's judgements, a cell every reader rejects and no sync can
    repair. They are a conflict too (kept under conflicts/), and Colab's own judgements then go up."""
    hub = FakeHub()
    machine(tmp_path / "colab", monkeypatch)
    g_bf16 = [gen_row("kb1", "p1"), gen_row("kb2", "p2")]
    write(local(GEN), lines(*g_bf16))
    assert sync.push(api=hub).files == {GEN: "new"}
    machine(tmp_path / "kaggle", monkeypatch)                          # pulled before Colab's push
    g_fp16 = [gen_row("kf1", "p1", dtype="fp16"), gen_row("kf2", "p2", dtype="fp16")]
    write(local(GEN), lines(*g_fp16))
    write(local(JUD), judged(g_fp16, judge=FP16_JUDGE))
    report = sync.push(api=hub)
    assert report.files == {GEN: "conflict", JUD: "conflict"} and not report.ok
    assert report.held == {JUD: GEN} and JUD not in hub.files
    assert hub.files[report.conflict_copies[JUD]] == local(JUD).read_bytes()
    assert f"  CONFLICT {JUD} (held back with {GEN}; local copy on the Hub at " \
           f"{report.conflict_copies[JUD]})" in report.summary()
    assert any("1 judgements file(s) held back" in line for line in report.summary())
    machine(tmp_path / "colab", monkeypatch)
    write(local(JUD), judged(g_bf16))
    assert sync.push(api=hub).files == {GEN: "same", JUD: "new"}
    assert keys_of_bytes(hub.files[JUD]) == keys_of_bytes(hub.files[GEN]) == ["kb1", "kb2"]


def test_judgements_held_back_are_not_merged_here_either(tmp_path, monkeypatch):
    """Held back, a judgements file is not merged with the Hub's either: that would put judgements of
    generations this machine does not have into its own cell, and upload them."""
    first = gen_row("k1", "p1")
    hub = FakeHub({GEN: lines(first), JUD: judged([first])})
    a, b = tmp_path / "A", tmp_path / "B"
    machine(a, monkeypatch)
    sync.pull(api=hub)
    machine(b, monkeypatch)
    sync.pull(api=hub)
    append(local(GEN), lines(gen_row("k2", "p2")))
    append(local(JUD), judged([gen_row("k2", "p2")]))
    assert sync.push(api=hub).files == {GEN: "update", JUD: "update"}
    machine(a, monkeypatch)
    other = gen_row("k3", "p3", dtype="fp16")          # another run configuration: generations cannot merge
    append(local(GEN), lines(other))
    append(local(JUD), judged([other]))                # same judge, new prompt: these alone would merge
    before = local(JUD).read_bytes()
    report = sync.push(api=hub)
    assert report.files == {GEN: "conflict", JUD: "conflict"} and report.held == {JUD: GEN}
    assert local(JUD).read_bytes() == before and keys_of_bytes(hub.files[JUD]) == ["k1", "k2"]


def test_judgements_wait_for_generations_that_are_newer_on_the_hub(tmp_path, monkeypatch):
    first = gen_row("k1", "p1")
    hub = FakeHub({GEN: lines(first)})
    a, b = tmp_path / "A", tmp_path / "B"
    machine(b, monkeypatch)
    sync.pull(api=hub)
    machine(a, monkeypatch)
    sync.pull(api=hub)
    write(local(JUD), judged([first]))
    machine(b, monkeypatch)
    append(local(GEN), lines(gen_row("k2", "p2")))
    sync.push(api=hub)
    machine(a, monkeypatch)
    planned = sync.status(api=hub)["push"]
    assert planned.files == {GEN: "stale", JUD: "conflict"} and planned.held == {JUD: GEN}
    report = sync.push(api=hub)
    assert report.files == {GEN: "stale", JUD: "conflict"} and JUD not in hub.files
    assert sync.pull(api=hub).files == {GEN: "update"}
    assert sync.push(api=hub).files == {GEN: "same", JUD: "new"}
    assert keys_of_bytes(hub.files[JUD]) == ["k1"] and keys_of_bytes(hub.files[GEN]) == ["k1", "k2"]


def test_judgements_pushed_alone_need_the_same_generations_on_the_hub():
    first = gen_row("k1", "p1")
    hub = FakeHub({GEN: lines(first)})
    sync.pull(api=hub)
    write(local(JUD), judged([first]))
    append(local(GEN), lines(gen_row("k2", "p2")))                   # generated more, not pushed yet
    report = sync.push(["outputs/judgements"], api=hub)
    assert report.files == {JUD: "conflict"} and report.held == {JUD: GEN} and JUD not in hub.files
    write(local(GEN), lines(first))                                    # the Hub's copy again
    assert sync.push(["outputs/judgements"], api=hub).files == {JUD: "new"}
    append(local(JUD), judged([gen_row("k2", "p2")]))
    local(GEN).unlink()                                     # the Hub has generations, this machine none
    assert sync.push(["outputs/judgements"], api=hub).files == {JUD: "conflict"}
    lone = "outputs/judgements/qwen3guard/adapter-hi/harmful_en_dev.jsonl"
    write(local(lone), judged([gen_row("k9", "p9")]))      # no generations anywhere: nothing to pair
    assert sync.push(["outputs/judgements/qwen3guard/adapter-hi"], api=hub).files == {lone: "new"}


# ---------------------------------------------------------------------------- AuditKIT records follow a cell


def test_auditkit_records_never_go_up_without_their_cell(tmp_path, monkeypatch):
    """lift judge and lift run-dev write an AuditKIT record of every judged cell. VM A pushed a cell's
    generations (job.py's checkpoint); VM B, on another GPU class, generated, judged and recorded the
    same cell. B's generations conflict and its judgements are held back, so its record must be too:
    the Hub would otherwise hold B's numbers beside A's cell, and P08 publishes those records. A's own
    record then goes up with A's judgements."""
    hub = FakeHub()
    rec = record_path()
    machine(tmp_path / "A", monkeypatch)
    g_a = [gen_row("k1", "p1", output="A1"), gen_row("k2", "p2", output="A2")]
    write(local(GEN), lines(*g_a))
    assert sync.push(api=hub).files == {GEN: "new"}
    machine(tmp_path / "B", monkeypatch)
    g_b = [gen_row("k1", "p1", output="B1"), gen_row("k2", "p2", output="B2")]
    write(local(GEN), lines(*g_b))
    write(local(JUD), judged(g_b, "harmful"))
    write(local(rec), audit_record(g_b, 0.0))
    report = sync.push(api=hub)
    assert report.files == {GEN: "conflict", JUD: "conflict", rec: "conflict"}
    assert report.held == {JUD: GEN, rec: GEN} and rec not in hub.files
    copy = report.conflict_copies[rec]
    assert hub.files[copy] == local(rec).read_bytes()
    assert f"  CONFLICT {rec} (held back with {GEN}; local copy on the Hub at {copy})" in report.summary()
    assert any("1 AuditKIT record(s) held back" in line for line in report.summary())
    assert any("1 judgements file(s) held back" in line for line in report.summary())
    machine(tmp_path / "A", monkeypatch)
    write(local(JUD), judged(g_a))
    write(local(rec), audit_record(g_a, 1.0))
    assert sync.push(api=hub).files == {GEN: "same", JUD: "new", rec: "new"}
    assert hub.files[rec] == local(rec).read_bytes()


def test_a_precision_record_follows_its_cell(tmp_path, monkeypatch):
    """Kaggle's fp16 cell conflicts with Colab's bf16 one: its licensed -fp16 record, and the private
    .full one beside it, stay off the canonical paths with the cell."""
    hub = FakeHub()
    machine(tmp_path / "colab", monkeypatch)
    write(local(GEN), lines(gen_row("kb1", "p1"), gen_row("kb2", "p2")))
    sync.push(api=hub)
    machine(tmp_path / "kaggle", monkeypatch)
    g_fp16 = [gen_row("kf1", "p1", dtype="fp16"), gen_row("kf2", "p2", dtype="fp16")]
    recs = [record_path(dtype="fp16"), record_path(dtype="fp16", text="full")]
    assert [r.rsplit("/", 1)[1] for r in recs] == ["harmful_en_dev-fp16.json",
                                                   "harmful_en_dev-fp16.full.json"]
    write(local(GEN), lines(*g_fp16))
    write(local(JUD), judged(g_fp16, judge=FP16_JUDGE))
    for rec in recs:
        write(local(rec), audit_record(g_fp16, 1.0))
    report = sync.push(api=hub)
    assert report.files == {GEN: "conflict", JUD: "conflict", **{rec: "conflict" for rec in recs}}
    assert report.held == {JUD: GEN, **{rec: GEN for rec in recs}}
    assert sorted(p for p in hub.files if p.startswith("outputs/")) == [GEN]


def test_auditkit_records_never_go_up_beside_other_judgements(tmp_path, monkeypatch):
    """Both machines hold the same generations. A pushed its judgements before its record was written;
    B judged the same cell on another judge stack and recorded it. B's judgements cannot merge with
    A's (one key, two records), which push finds out only when it tries, after judgements were held
    back: B's record must still stay off the Hub, which holds A's judgements."""
    g = [gen_row("k1", "p1"), gen_row("k2", "p2")]
    hub = FakeHub({GEN: lines(*g)})
    rec = record_path()
    a, b = tmp_path / "A", tmp_path / "B"
    machine(b, monkeypatch)
    sync.pull(api=hub)
    machine(a, monkeypatch)
    sync.pull(api=hub)
    write(local(JUD), judged(g))
    assert sync.push(api=hub).files == {GEN: "same", JUD: "new"}
    machine(b, monkeypatch)
    write(local(JUD), judged(g, "harmful"))
    write(local(rec), audit_record(g, 0.0))
    assert sync.status(api=hub)["push"].files == {GEN: "same", JUD: "merge", rec: "new"}   # not tried yet
    report = sync.push(api=hub)
    assert report.files == {GEN: "same", JUD: "conflict", rec: "conflict"} and report.held == {rec: JUD}
    assert rec not in hub.files and hub.files[report.conflict_copies[rec]] == local(rec).read_bytes()


def test_auditkit_records_pushed_alone_need_their_cell_on_the_hub():
    g = [gen_row("k1", "p1")]
    hub = FakeHub({GEN: lines(*g)})
    sync.pull(api=hub)
    rec, full = record_path(), record_path(text="full")
    write(local(JUD), judged(g))
    for path in (rec, full):
        write(local(path), audit_record(g, 1.0))
    planned = sync.status(["outputs/metrics"], api=hub)["push"]
    assert planned.files == {rec: "conflict", full: "conflict"} and planned.held == {rec: JUD, full: JUD}
    report = sync.push(["outputs/metrics"], api=hub)
    assert report.held == {rec: JUD, full: JUD} and rec not in hub.files and full not in hub.files
    assert sync.push(["outputs/judgements"], api=hub).files == {JUD: "new"}
    assert sync.push(["outputs/metrics"], api=hub).files == {rec: "new", full: "new"}
    hub.put({JUD: judged(g, "harmful")})                      # another machine re-judged the cell
    write(local(rec), audit_record(g, 0.5))                   # a record of the old judgements, rewritten
    report = sync.push(api=hub)
    assert report.files[JUD] == "stale" and report.files[rec] == "conflict" and report.held == {rec: JUD}
    other = record_path(judge="frontier")                     # the second judge's: no judgements anywhere
    write(local(other), audit_record(g, 1.0))
    assert sync.push(["outputs/metrics/auditkit/frontier"], api=hub).files == {other: "new"}


@pytest.mark.parametrize("path,made_from", [
    ("outputs/metrics/auditkit/qwen3guard/stock/harmful_en_dev.json", (GEN, JUD)),
    ("outputs/metrics/auditkit/qwen3guard/stock/harmful_en_dev.full.json", (GEN, JUD)),
    ("outputs/metrics/auditkit/qwen3guard/stock/harmful_en_dev-fp16.json", (GEN, JUD)),
    ("outputs/metrics/auditkit/qwen3guard/stock/harmful_en_dev-fp16.full.json", (GEN, JUD)),
    ("outputs/metrics/auditkit/frontier/adapter-hi/harmless_zh_final.json",
     ("outputs/generations/adapter-hi/harmless_zh_final.jsonl",
      "outputs/judgements/frontier/adapter-hi/harmless_zh_final.jsonl")),
    ("outputs/metrics/auditkit/qwen3guard/stock/notes.json", ()),
    ("outputs/metrics/auditkit/qwen3guard/stock/harmful_en_dev.md", ()),
    ("outputs/metrics/gate0.json", ()),
    (JUD, ()),
])
def test_an_auditkit_record_maps_to_its_cell(path, made_from):
    """Every name lift.audit.auditkit_path gives ({set}_{lang}_{split}, then -{dtype} and .full)."""
    assert sync._recorded_from(path) == made_from


# ---------------------------------------------------------------------------- pull


def test_pull_brings_new_files_into_every_target_and_nothing_else(isolated_dirs):
    files = {path: f"placeholder {path}\n".encode() for path in ONE_PER_TARGET}
    files["outputs/adapters/hi-r1/adapter_model.safetensors"] = bytes(range(256)) * 4
    files["conflicts/20260101T000000Z/validation/labels/x.csv"] = b"aside\n"
    files["README.md"] = b"# runs\n"
    hub = FakeHub(files)
    report = sync.pull(api=hub)
    synced = {p: data for p, data in files.items() if p.startswith(tuple(sync.targets()))}
    assert report.ok and report.files == {p: "new" for p in synced}
    assert all(local(p).read_bytes() == data for p, data in synced.items())
    root = isolated_dirs.resolve()
    assert not [p for p in root.rglob("*") if {".cache", "conflicts", "README.md"} & set(p.parts)]
    staging = {Path(c["local_dir"]).resolve() for c in hub.called("hf_hub_download")}
    assert not any(s.is_relative_to(root) for s in staging)                    # staged outside the tree
    assert {c["revision"] for c in hub.called("hf_hub_download")} == {hub.head}
    assert state_file().is_file()


def test_pull_reads_one_pinned_commit():
    hub = FakeHub({GEN: records("a")})
    pinned = hub.head
    hub.after_list = lambda h: h.put({GEN: records("a", "late")})   # a push lands in the middle of the pull
    report = sync.pull(api=hub)
    assert report.revision == pinned and keys_of(local(GEN)) == ["a"]
    assert {c["revision"] for c in hub.called("list_repo_tree") + hub.called("hf_hub_download")} == {pinned}
    assert sync.pull(api=hub).files == {GEN: "update"}


def test_pull_skips_identical_files():
    adapter = "outputs/adapters/hi-r1/adapter_model.safetensors"
    hub = FakeHub({GEN: records("a"), adapter: b"\x00" * 64})
    sync.pull(api=hub)
    fetched = len(hub.called("hf_hub_download"))
    assert sync.pull(api=hub).files == {GEN: "same", adapter: "same"}
    assert len(hub.called("hf_hub_download")) == fetched


def test_pull_updates_a_local_copy_nobody_edited_and_deletes_nothing():
    hub = FakeHub({GEN: records("a")})
    sync.pull(api=hub)
    write(local("outputs/generations/stock/local_only.jsonl"), records("x"))
    hub.put({GEN: records("a", "b")})
    assert sync.pull(api=hub).files == {GEN: "update"}
    assert keys_of(local(GEN)) == ["a", "b"]
    assert keys_of(local("outputs/generations/stock/local_only.jsonl")) == ["x"]


def test_pull_keeps_a_local_edit_when_the_hub_did_not_change():
    hub = FakeHub({LABELS: b"v1\n"})
    sync.pull(api=hub)
    write(local(LABELS), "edited here\n")
    report = sync.pull(api=hub)
    assert report.files == {LABELS: "keep-local"} and report.ok
    assert local(LABELS).read_text() == "edited here\n" and len(hub.called("hf_hub_download")) == 1


def test_pull_reports_a_conflict_and_force_takes_the_hub_copy():
    hub = FakeHub({LABELS: b"v1\n"})
    sync.pull(api=hub)
    write(local(LABELS), "edited here\n")
    hub.put({LABELS: b"v2 from another machine\n"})
    report = sync.pull(api=hub)
    assert report.files == {LABELS: "conflict"} and not report.ok
    assert local(LABELS).read_text() == "edited here\n"
    forced = sync.pull(api=hub, force=True)
    assert forced.files == {LABELS: "force"} and forced.ok
    assert local(LABELS).read_bytes() == b"v2 from another machine\n"
    assert sync.pull(api=hub).files == {LABELS: "same"}


def test_pull_force_keeps_the_local_copy_it_replaces():
    """An annotator's unpushed labels conflict with the Hub. The summary offers pull --force, which must
    not leave the human labels nowhere: the local copy goes to outputs/.sync-backup/<stamp>/<path>,
    which is never pushed."""
    hub = FakeHub({LABELS: b"id,label\n1,refusal\n"})
    sync.pull(api=hub)
    write(local(LABELS), "id,label\n1,refusal\n2,harmful\n")
    hub.put({LABELS: b"id,label\n1,benign\n"})
    assert any("pull --force" in line for line in sync.pull(api=hub).summary())
    forced = sync.pull(api=hub, force=True)
    assert forced.files == {LABELS: "force"} and local(LABELS).read_bytes() == b"id,label\n1,benign\n"
    backup = Path(forced.backups[LABELS])
    assert backup.read_bytes() == b"id,label\n1,refusal\n2,harmful\n"
    stamp = backup.relative_to(paths.output_dir() / ".sync-backup").parts[0]
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    assert backup == paths.output_dir() / ".sync-backup" / stamp / "validation" / "labels" / "hi_r1.csv"
    assert any(str(backup.parents[2]) in line for line in forced.summary())
    assert sync.push(api=hub).files == {LABELS: "same"}
    assert not [p for p in hub.files if ".sync-backup" in p]
    assert sync.status(api=hub)["push"].files == {LABELS: "same"}


def test_pull_force_keeps_a_jsonl_it_could_not_merge():
    hub = FakeHub({GEN: records("a")})
    sync.pull(api=hub)
    append(local(GEN), b'{"key": "l1", "output": "trunc')            # half-written: cannot merge
    hub.put({GEN: records("a", "r1")})
    before = local(GEN).read_bytes()
    forced = sync.pull(api=hub, force=True)
    assert forced.files == {GEN: "force"} and keys_of(local(GEN)) == ["a", "r1"]
    assert Path(forced.backups[GEN]).read_bytes() == before


def test_without_a_sync_state_a_differing_file_is_a_conflict_both_ways():
    write(local(LABELS), "made here before any sync\n")
    hub = FakeHub({LABELS: b"on the hub\n"})
    assert sync.pull(api=hub).files == {LABELS: "conflict"}
    assert sync.push(api=hub).files == {LABELS: "conflict"}
    assert hub.files[LABELS] == b"on the hub\n"
    assert local(LABELS).read_text() == "made here before any sync\n"


def test_pull_merges_jsonl_changed_on_both_sides_by_key():
    hub = FakeHub({GEN: records("a", "b")})
    sync.pull(api=hub)
    hub.put({GEN: records("a", "b", "r1", "r2")})
    append(local(GEN), records("l1"))
    report = sync.pull(api=hub)
    assert report.files == {GEN: "merge"} and report.ok
    assert keys_of(local(GEN)) == ["a", "b", "l1", "r1", "r2"]
    assert sync.push(api=hub).files == {GEN: "update"}       # the local-only line still goes up
    assert hub.files[GEN] == local(GEN).read_bytes()
    assert sync.pull(api=hub).files == {GEN: "same"}


@pytest.mark.parametrize("path,extra", [
    (GEN, b'{"key": "l1", "output": "trunc'),                                  # half-written line
    (JUD, b'{"id": "no key"}\n'),                                              # a record without a key
])
def test_pull_does_not_merge_jsonl_it_cannot_parse(path, extra):
    hub = FakeHub({path: records("a")})
    sync.pull(api=hub)
    hub.put({path: records("a", "r1")})
    append(local(path), extra)
    before = local(path).read_bytes()
    report = sync.pull(api=hub)
    assert report.files == {path: "conflict"} and local(path).read_bytes() == before


def test_a_download_that_does_not_match_the_tree_raises_and_overwrites_nothing():
    hub = FakeHub({GEN: records("a")})
    sync.pull(api=hub)
    hub.put({GEN: records("a", "b")})
    hub.corrupt = {GEN}
    with pytest.raises(sync.SyncError, match="does not match"):
        sync.pull(api=hub)
    assert keys_of(local(GEN)) == ["a"]
    hub.corrupt = set()
    assert sync.pull(api=hub).files == {GEN: "update"}


def test_a_download_that_cannot_reach_the_hub_raises_a_sync_error(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    hub = FakeHub({GEN: records("a")})

    def unreachable(*args, **kwargs):   # hf_hub_download on a dropped connection with no local file
        raise LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")

    monkeypatch.setattr(hub, "hf_hub_download", unreachable)
    with pytest.raises(sync.SyncError, match="could not reach the Hub"):
        sync.pull(api=hub)
    assert not local(GEN).exists()


def test_a_download_that_leaves_no_file_raises(monkeypatch):
    hub = FakeHub({GEN: records("a")})
    monkeypatch.setattr(hub, "hf_hub_download", lambda *a, **k: str(Path(k["local_dir"]) / "nothing"))
    with pytest.raises(sync.SyncError, match="left no file"):
        sync.pull(api=hub)
    assert not local(GEN).exists()


def test_lfs_entries_are_compared_by_sha256():
    adapter = "outputs/adapters/hi-r1/adapter_model.safetensors"
    hub = FakeHub({adapter: b"\x01" * 1000})
    assert sync.remote_hash(hub.entry(adapter, b"\x01" * 1000))[0] == "sha256"
    sync.pull(api=hub)
    assert sync.pull(api=hub).files == {adapter: "same"}
    write(local(adapter), b"\x02" * 1000)
    assert sync.push(api=hub).files == {adapter: "update"} and hub.files[adapter] == b"\x02" * 1000


def test_round_trip_between_two_machines(tmp_path, monkeypatch):
    hub = FakeHub()
    machine(tmp_path / "colab", monkeypatch)
    write(local(GEN), records("a"))
    write(local("data/prompts/harmful_hi.jsonl"), "x\n")
    sync.push(api=hub)
    machine(tmp_path / "laptop", monkeypatch)
    assert set(sync.pull(api=hub).files.values()) == {"new"}
    write(local(LABELS), "labels\n")
    sync.push(api=hub)
    machine(tmp_path / "colab", monkeypatch)
    assert sync.pull(api=hub).files == {GEN: "same", "data/prompts/harmful_hi.jsonl": "same", LABELS: "new"}


@pytest.mark.parametrize("path", ["outputs/../evil.txt", r"outputs/a\..\..\evil.txt"])
def test_pull_refuses_remote_paths_that_leave_the_target(path, isolated_dirs):
    hub = FakeHub({path: b"x"})
    with pytest.raises(sync.SyncError, match="not a safe relative path"):
        sync.pull(api=hub)
    assert not hub.called("hf_hub_download")
    assert not [p for p in isolated_dirs.parent.rglob("evil.txt")]


def test_a_sync_state_from_another_repo_is_ignored(monkeypatch):
    sync.pull(api=FakeHub({LABELS: b"v1\n"}))
    monkeypatch.setenv("LIFT_RUNS_REPO", "tester/other-runs")
    other = FakeHub({LABELS: b"another repo's copy\n"})
    assert sync.pull(api=other).files == {LABELS: "conflict"}       # not "update": nothing is known
    assert local(LABELS).read_bytes() == b"v1\n"


def test_an_unreadable_sync_state_stops_the_sync():
    write(state_file(), "{not json")
    with pytest.raises(sync.SyncError, match="cannot read the sync state"):
        sync.pull(api=FakeHub({LABELS: b"v1\n"}))
    assert not local(LABELS).exists()


def test_install_copies_across_filesystems(tmp_path, monkeypatch):
    staged = write(tmp_path / "staging" / "f.jsonl", "x\n")
    dest = tmp_path / "tree" / "sub" / "f.jsonl"
    real_replace = os.replace

    def replace(src, dst):   # renaming out of staging fails as it does from /tmp to /kaggle/working
        if Path(src) == staged:
            raise OSError(18, "Invalid cross-device link")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    sync._install(staged, dest)
    assert dest.read_text() == "x\n" and not staged.exists()
    assert [p.name for p in dest.parent.iterdir()] == ["f.jsonl"]          # no temp file left behind


# ---------------------------------------------------------------------------- files written while sync runs


def while_downloading(hub: FakeHub, then: Callable[[], None]) -> None:
    """Run ``then`` after every download, as a process on this machine writing meanwhile would."""
    download = hub.hf_hub_download

    def download_then(*args, **kwargs):
        result = download(*args, **kwargs)
        then()
        return result

    hub.hf_hub_download = download_then


def test_pull_never_replaces_a_file_written_while_it_downloads():
    """Pull plans "update" because nobody edited the local copy, then downloads. Rows a judge on this
    machine appends meanwhile must not be lost to the replace: the file stays as it is, reported as a
    conflict, and the next pull merges it."""
    hub = FakeHub()
    first = jud_row("k1", "p1", "refusal")
    write(local(JUD), lines(first))
    sync.push(api=hub)
    hub.put({JUD: lines(first, jud_row("k2", "p2", "benign"))})       # another machine judged k2
    while_downloading(hub, lambda: append(local(JUD), lines(jud_row("k3", "p3", "harmful"))))
    report = sync.pull(api=hub)
    assert report.files == {JUD: "conflict"} and report.busy == [JUD] and not report.ok
    assert keys_of(local(JUD)) == ["k1", "k3"]
    assert f"  CONFLICT {JUD} (changed here while sync ran)" in report.summary()
    assert any("1 file(s) changed on this machine while sync ran" in line for line in report.summary())
    assert sync.pull(api=FakeHub(hub.files)).files == {JUD: "merge"}  # nothing writes now
    assert keys_of(local(JUD)) == ["k1", "k3", "k2"]


def test_pull_never_replaces_a_file_created_while_it_downloads():
    hub = FakeHub({LABELS: b"from the hub\n"})
    while_downloading(hub, lambda: write(local(LABELS), "saved here meanwhile\n"))
    report = sync.pull(api=hub)
    assert report.files == {LABELS: "conflict"} and report.busy == [LABELS]
    assert local(LABELS).read_text() == "saved here meanwhile\n"
    assert sync.pull(api=FakeHub(hub.files)).files == {LABELS: "conflict"}   # nothing was recorded as synced


def test_pull_force_leaves_a_file_written_while_it_downloads():
    hub = FakeHub({LABELS: b"v1\n"})
    sync.pull(api=hub)
    write(local(LABELS), "edited here\n")
    hub.put({LABELS: b"v2\n"})
    while_downloading(hub, lambda: append(local(LABELS), b"more\n"))
    forced = sync.pull(api=hub, force=True)
    assert forced.files == {LABELS: "conflict"} and forced.busy == [LABELS] and not forced.backups
    assert local(LABELS).read_text() == "edited here\nmore\n"


@pytest.mark.parametrize("direction", ["pull", "push"])
def test_a_merge_is_never_written_over_lines_added_meanwhile(direction, monkeypatch):
    hub = FakeHub({JUD: lines(jud_row("k1", "p1", "refusal"))})
    sync.pull(api=hub)
    hub.put({JUD: lines(jud_row("k1", "p1", "refusal"), jud_row("k2", "p2", "benign"))})
    append(local(JUD), lines(jud_row("k3", "p3", "harmful")))
    merge = sync.merge_jsonl_by_key

    def merge_while_the_judge_appends(mine, theirs, *, kind):
        merged = merge(mine, theirs, kind=kind)
        append(mine, lines(jud_row("k4", "p4", "benign")))            # after the read, before the write
        return merged

    monkeypatch.setattr(sync, "merge_jsonl_by_key", merge_while_the_judge_appends)
    on_the_hub = hub.files[JUD]
    report = getattr(sync, direction)(api=hub)
    assert report.files == {JUD: "conflict"} and report.busy == [JUD]
    assert keys_of(local(JUD)) == ["k1", "k3", "k4"] and hub.files[JUD] == on_the_hub
    assert not report.conflict_copies and not [p for p in hub.files if p.startswith("conflicts/")]
    monkeypatch.setattr(sync, "merge_jsonl_by_key", merge)
    assert getattr(sync, direction)(api=hub).files == {JUD: "merge"}
    assert keys_of(local(JUD)) == ["k1", "k3", "k4", "k2"]


# ---------------------------------------------------------------------------- merge rules


def test_merge_never_reverts_a_rejudged_file_to_older_records(tmp_path, monkeypatch):
    """A re-judge after a fix keeps the keys and changes labels. A machine that still holds the old file
    plus a change of its own must not merge its old labels back in, by pull or by push."""
    hub = FakeHub({JUD: lines(jud_row("k1", "p1", "harmful"), jud_row("k2", "p2", "refusal"))})
    a, b = tmp_path / "A", tmp_path / "B"
    machine(a, monkeypatch)
    sync.pull(api=hub)
    append(local(JUD), lines(jud_row("k3", "p3", "benign")))             # A judged one more key
    machine(b, monkeypatch)
    sync.pull(api=hub)
    write(local(JUD), lines(jud_row("k1", "p1", "refusal"), jud_row("k2", "p2", "refusal"),
                            jud_row("k3", "p3", "benign")))             # B re-judged after a parser fix
    assert sync.push(api=hub).files == {JUD: "update"}
    rejudged = hub.files[JUD]
    machine(a, monkeypatch)
    old = local(JUD).read_bytes()
    pulled = sync.pull(api=hub)
    assert pulled.files == {JUD: "conflict"} and local(JUD).read_bytes() == old
    pushed = sync.push(api=hub)
    assert pushed.files == {JUD: "conflict"} and hub.files[JUD] == rejudged
    assert hub.files[pushed.conflict_copies[JUD]] == old


def test_merge_compares_generations_without_their_run_fields(tmp_path):
    first = gen_row("k1", "p1")
    rerun = {**first, "seconds": 9.25, "meta": {**first["meta"], "gpu": "NVIDIA L4", "device": "cuda:1",
                                                 "batch_size": 3, "n_generated": 22, "torch": "2.15.0",
                                                 "transformers": "5.22.1"}}
    mine = write(tmp_path / "local.jsonl", lines(first, gen_row("k2", "p2")))
    theirs = write(tmp_path / "remote.jsonl", lines(rerun, gen_row("k3", "p3")))
    merged = sync.merge_jsonl_by_key(mine, theirs, kind="generations")
    assert merged.rows == [first, gen_row("k2", "p2"), gen_row("k3", "p3")] and merged.added == 1
    for changed in ({"output": "another output"}, {"hit_cap": True},
                    {"meta": {**first["meta"], "dtype": "fp16"}}):
        theirs = write(tmp_path / "theirs.jsonl", lines({**first, **changed}))
        with pytest.raises(ValueError, match="different records"):
            sync.merge_jsonl_by_key(mine, theirs, kind="generations")
    odd = write(tmp_path / "nan.jsonl", lines({**first, "meta": {**first["meta"], "score": float("nan")}}))
    assert sync.merge_jsonl_by_key(odd, odd, kind="generations").added == 0     # NaN equals itself here


def test_merge_never_writes_two_records_for_one_prompt(tmp_path, monkeypatch):
    """The dtype is in the key, not the path: a Kaggle fp16 run and a Colab bf16 run of one variant write
    the same file with different keys for the same pids. Merging them would push a file that every
    reader rejects (two records per pid, two run configurations), so it is a conflict."""
    hub = FakeHub()
    kaggle, colab = tmp_path / "kaggle", tmp_path / "colab"
    machine(kaggle, monkeypatch)
    sync.pull(api=hub)
    machine(colab, monkeypatch)
    write(local(GEN), lines(gen_row("k-bf16", "p1")))
    sync.push(api=hub)
    machine(kaggle, monkeypatch)
    write(local(GEN), lines(gen_row("k-fp16", "p1", dtype="fp16")))
    report = sync.push(api=hub)
    assert report.files == {GEN: "conflict"}
    assert hub.files[GEN] == lines(gen_row("k-bf16", "p1")) and keys_of(local(GEN)) == ["k-fp16"]
    assert sync.pull(api=hub).files == {GEN: "conflict"} and keys_of(local(GEN)) == ["k-fp16"]


@pytest.mark.parametrize("kind,mine,theirs,problem", [
    ("generations", [gen_row("k1", "p1")], [gen_row("k2", "p1")], "prompt"),
    ("generations", [gen_row("k1", "p1")], [gen_row("k2", "p2", dtype="fp16")], "run_config"),
    ("generations", [gen_row("k1", "p1")], [{**gen_row("k2", "p2"), "meta": {"dtype": "bf16"}}],
     "run_config"),                                                    # an older record without one
    ("generations", [gen_row("k1", "p1"), gen_row("k1", "p1")], [gen_row("k2", "p2")], "key"),
    ("generations", [gen_row("k1", "p1")], [{**gen_row("k2", "p2"), "variant": "adapter-hi"}],
     "variant/set/lang/split"),
    ("judgements", [jud_row("k1", "p1", "refusal")], [jud_row("k2", "p1", "refusal")], "prompt"),
    ("judgements", [jud_row("k1", "p1", "refusal")],
     [jud_row("k2", "p2", "refusal", judge="other@" + "b" * 40)], "judge"),
])
def test_merge_refuses_a_file_its_readers_would_reject(tmp_path, kind, mine, theirs, problem):
    left = write(tmp_path / "local.jsonl", lines(*mine))
    right = write(tmp_path / "remote.jsonl", lines(*theirs))
    with pytest.raises(ValueError, match=re.escape(problem)):
        sync.merge_jsonl_by_key(left, right, kind=kind)


A100 = {"torch": "2.15.0", "transformers": "5.22.1", "gpu_capability": "8.0"}
L4 = {**A100, "gpu_capability": "8.9"}


def on(stack: dict, row: dict) -> dict:
    """``row`` as generated on ``stack`` (meta carries the fields and their digest, as generate writes)."""
    from lift.models.generate import stack_digest

    return {**row, "meta": {**row["meta"], **stack, "stack": stack_digest(stack)}}


def test_merge_never_mixes_two_stacks_in_one_cell(tmp_path):
    """run_config holds only the inputs, so two halves of a cell generated on an A100 and an L4 share it;
    readers refuse a cell on two stacks, so the merge is a conflict, not a wedged cell on the Hub. The
    same record of one key from two stacks is still one record."""
    mine = write(tmp_path / "local.jsonl", lines(on(A100, gen_row("k1", "p1"))))
    theirs = write(tmp_path / "remote.jsonl", lines(on(L4, gen_row("k2", "p2"))))
    with pytest.raises(ValueError, match="2 different stack values"):
        sync.merge_jsonl_by_key(mine, theirs, kind="generations")
    same = write(tmp_path / "same.jsonl", lines(on(L4, gen_row("k1", "p1"))))
    merged = sync.merge_jsonl_by_key(mine, same, kind="generations")
    assert merged.added == 0 and merged.rows == [on(A100, gen_row("k1", "p1"))]


def test_merge_never_mixes_two_judge_stacks_in_one_file(tmp_path):
    mine = write(tmp_path / "local.jsonl", lines(jud_row("k1", "p1", "refusal", native={"stack": A100})))
    theirs = write(tmp_path / "remote.jsonl", lines(jud_row("k2", "p2", "refusal", native={"stack": L4})))
    with pytest.raises(ValueError, match="2 different judge stack values"):
        sync.merge_jsonl_by_key(mine, theirs, kind="judgements")
    api = write(tmp_path / "api.jsonl", lines(jud_row("k3", "p3", "refusal")))       # an API judge: no stack
    assert sync.merge_jsonl_by_key(api, api, kind="judgements").added == 0


@pytest.mark.parametrize("meta", [{**A100}, {**A100, "stack": "f" * 16}, {"dtype": "bf16"}])
def test_sync_reads_a_stack_as_generate_does(meta):
    from lift.models.generate import STACK_FIELDS, record_stack

    assert sync.STACK_FIELDS == STACK_FIELDS and sync.record_stack(meta) == record_stack(meta)


def test_judge_cache_keeps_the_local_row_for_a_key_on_both_sides():
    hub = FakeHub({CACHE: records("c1")})
    sync.pull(api=hub)
    append(local(CACHE), lines(row("c2", text="answer made here")))
    hub.put({CACHE: records("c1") + lines(row("c2", text="answer made there"), row("c3"))})
    report = sync.pull(api=hub)
    assert report.files == {CACHE: "merge"} and report.ok
    rows = list(read_jsonl(local(CACHE)))
    assert [r["key"] for r in rows] == ["c1", "c2", "c3"] and rows[1]["text"] == "answer made here"


def test_judge_cache_with_an_unreadable_line_still_merges(caplog):
    """The frontier judge's call cache skips a line a crash cut short (the next append is glued onto
    it) and a line without a key. Sync drops them too, and counts them, instead of turning every later
    push and pull of the shared cache into a conflict."""
    hub = FakeHub({CACHE: records("c1")})
    sync.pull(api=hub)
    append(local(CACHE), b'{"key": "c2", "text": "trunc')     # killed mid-append...
    append(local(CACHE), records("c3", "c4"))                  # ...and the next run's first line is glued on
    append(local(CACHE), b'{"no": "key"}\n')
    hub.put({CACHE: records("c1", "cB")})
    with caplog.at_level(logging.INFO, logger="lift.sync"):
        report = sync.push(api=hub)
    assert report.ok and report.files == {CACHE: "merge"}
    assert keys_of_bytes(hub.files[CACHE]) == ["c1", "c4", "cB"]
    assert local(CACHE).read_bytes() == hub.files[CACHE]
    assert "dropped 2 unreadable line(s)" in caplog.text
    assert sync.push(api=hub).files == {CACHE: "same"}


def test_merge_jsonl_by_key_keeps_local_order_then_new_remote_keys(tmp_path):
    mine = write(tmp_path / "local.jsonl", records("b", "a", "c"))
    theirs = write(tmp_path / "remote.jsonl", records("a", "d", "b", "e", "d"))
    merged = sync.merge_jsonl_by_key(mine, theirs, kind="generations")
    assert [r["key"] for r in merged.rows] == ["b", "a", "c", "d", "e"] and merged.added == 2
    with pytest.raises(ValueError, match="string 'key'"):
        sync.merge_jsonl_by_key(mine, write(tmp_path / "nokey.jsonl", '{"id": 1}\n'), kind="generations")


def test_merge_kind_follows_the_path():
    assert [sync.merge_kind(p) for p in (GEN, JUD, CACHE, "outputs/generations/stock/x.json",
                                         "outputs/metrics/x.jsonl", LABELS)] == [
        "generations", "judgements", "judge_cache", None, None, None]


# ---------------------------------------------------------------------------- status and helpers


def test_status_is_a_dry_run_of_both_directions():
    hub = FakeHub({GEN: records("a"), LABELS: b"v1\n"})
    sync.pull(api=hub)
    hub.put({GEN: records("a", "r1"), "outputs/remote_only.jsonl": records("x")})
    write(local(LABELS), "edited here\n")
    write(local("outputs/local_only.jsonl"), "x\n")
    state, calls = state_file().read_bytes(), len(hub.calls)
    reports = sync.status(api=hub)
    assert reports["pull"].files == {GEN: "update", LABELS: "keep-local", "outputs/remote_only.jsonl": "new"}
    assert reports["push"].files == {GEN: "stale", LABELS: "update", "outputs/local_only.jsonl": "new"}
    assert all(r.dry_run and r.ok for r in reports.values())
    assert [name for name, _ in hub.calls[calls:]] == ["dataset_info", "list_repo_tree"]
    assert state_file().read_bytes() == state and keys_of(local(GEN)) == ["a"]


def test_file_digest_matches_the_huggingface_hub_helpers(tmp_path):
    from huggingface_hub.utils.sha import git_hash, sha_fileobj

    for data in (b"", b"placeholder\n", bytes(range(256)) * 9000):
        path = write(tmp_path / "f.bin", data)
        digest = sync.file_digest(path, chunk=4096)
        with open(path, "rb") as f:
            assert digest.sha256 == sha_fileobj(f).hex()
        assert digest.git == git_hash(data)


def test_importing_sync_does_not_import_huggingface_hub():
    code = "import sys, lift.sync, lift.commands.sync; print('huggingface_hub' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------- CLI


def run_cli(argv: list[str]) -> int:
    """``lift <argv>`` with only the sync command registered, as lift.cli.main runs it."""
    from lift.commands import sync as sync_command

    parser = argparse.ArgumentParser(prog="lift")
    sync_command.register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


@pytest.mark.parametrize("argv", [["sync", "--help"], ["sync", "push", "--help"], ["sync", "pull", "--help"],
                                  ["sync", "status", "--help"], ["sync", "check", "--help"]])
def test_cli_help(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_cli(argv)
    assert exit_info.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    if argv[1] == "check":
        assert "--write" in out and "--only" not in out and "--force" not in out
        assert "exit codes: 0 ok; 1 a transient" in out and "2 a problem no rerun fixes" in out
        return
    if len(argv) == 2:
        assert all(word in out for word in ("push", "pull", "status", "check"))
    else:
        assert "--only" in out and ("--force" in out) == (argv[1] != "status")
    assert "exit codes: 0 done; 3 done, but some files conflicted" in out
    assert "1 a transient or unknown error" in out and "2 a problem no rerun fixes" in out


def test_lift_cli_registers_sync(capsys):
    from lift.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["sync", "--help"])
    assert exit_info.value.code == 0 and "push" in capsys.readouterr().out


def test_cli_exit_codes(monkeypatch, capsys):
    """0 done; 3 done, but some files conflicted (runners report it and never retry); 1 a transient
    error, so not everything was transferred and a rerun is safe; 2 a problem no rerun fixes."""
    hub = FakeHub({LABELS: b"v1\n", GEN: records("a")})
    serve(monkeypatch, hub)
    assert run_cli(["sync", "pull"]) == 0
    assert "outputs: 1 new" in capsys.readouterr().out
    write(local(LABELS), "edited here\n")
    hub.put({LABELS: b"v2\n"})
    assert run_cli(["sync", "status"]) == 3
    assert run_cli(["sync", "pull"]) == 3
    assert run_cli(["sync", "push", "--only", "validation/labels"]) == 3
    out = capsys.readouterr().out
    assert "CONFLICT validation/labels/hi_r1.csv" in out and "edited here" not in out
    assert run_cli(["sync", "pull", "--force"]) == 0
    assert ".sync-backup" in capsys.readouterr().out
    hub.put({GEN: records("a", "b")})
    hub.corrupt = {GEN}
    assert run_cli(["sync", "pull"]) == 1                             # not everything arrived: retry
    assert "does not match" in capsys.readouterr().err
    hub.private = False
    assert run_cli(["sync", "push"]) == 2                             # no rerun fixes it
    err = capsys.readouterr().err
    assert "not private" in err and "Traceback" not in err
    assert run_cli(["sync", "pull", "--only", "outputs/../src"]) == 2


def test_cli_check_runs_the_preflight(monkeypatch, capsys):
    """`lift sync check [--write]`: 0 ok, 1 transient, 2 permanent. It only asks; nothing moves."""
    hub = FakeHub({LABELS: b"v1\n"})
    serve(monkeypatch, hub)
    hub.me = whoami_answer(role="read")
    assert run_cli(["sync", "check"]) == 0                            # reading is all a pull needs
    assert REPO in capsys.readouterr().out
    assert run_cli(["sync", "check", "--write"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("lift sync check: ") and "WRITE token" in err and "hf_dummy" not in err
    hub.me = whoami_answer(role="write")
    assert run_cli(["sync", "check", "--write"]) == 0
    hub.whoami = raising(_http("HfHubHTTPError", 503))
    assert run_cli(["sync", "check", "--write"]) == 1
    hub.private = False
    assert run_cli(["sync", "check"]) == 2
    assert {name for name, _ in hub.calls} == {"dataset_info", "whoami"} and not local(LABELS).exists()
