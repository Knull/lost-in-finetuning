"""``lift remote``: run lift lines on a Colab GPU VM through the Colab CLI 0.7.5, then stop the VM.

Why this shape (the CLI's source was read and its behaviour tested on Windows against fakes;
colab/README.md has the operator's view):

- ``colab run`` tears the VM down on any laptop-side hiccup, and ``--env`` values, like any exec'd
  code, land in plaintext in the CLI's local history and colab.log. So the flow is ``colab new``,
  then ``colab upload`` of a temp secrets file (deleted here in ``finally``, and on the VM by
  colab/job.py before anything else), then ``colab exec -f colab/job.py`` as a background process
  whose ``--env`` carries only the job spec.
- Before ``colab new``, lift.sync.preflight checks from here that the runs repo exists, is private
  and takes this token's writes: a read-only token would otherwise be found by the job's first push,
  after the GPU work.
- ``colab exec`` exits 0 even when the code fails, so the job's status comes from the
  "@@LIFT_DONE {json}" line that job.py appends to /content/lift_job.log, fetched every ``--poll``
  seconds with ``colab download`` (the contents API needs no long-lived websocket). When the log
  cannot be written, job.py prints that line instead, and the exec client's output carries it here.
- The job must get to push its outputs. On Ctrl+C or at ``--max-hours``, lift remote uploads
  /content/lift_job.stop and waits up to STOP_WAIT for the job's last line, longer (up to PUSH_WAIT)
  while the job's log shows a push still running: job.py stops the running line and still runs its
  final push. When that push fails (but not with lift sync's exit 2, which no retry fixes), or the job
  ends with no last line after running lines, lift remote tries a push once more itself (job.py in
  push mode) before it stops the VM.
- ``colab stop`` exits 0 for a name the CLI has forgotten, so ``finally`` stops the VM and then checks
  that ``colab sessions`` no longer lists its endpoint, which is saved to
  outputs/manifests/remote-<name>.json right after ``colab new``. Without an endpoint, a VM the CLI
  has no record of ("[?]") may be this one, so nothing is called released.
- Every colab call gets list argv, stdin=DEVNULL (the CLI would otherwise wait on its login prompt),
  PYTHONUTF8=1 (it otherwise uses cp1252 on Windows), an explicit timeout (its HTTP calls have none)
  and its own process group, so only lift remote sees a Ctrl+C. Printing can never stop the release.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

from lift.paths import REPO_ROOT, manifests_dir, output_dir

GPUS = ("A100", "L4", "T4")
DEFAULT_REPO = "Knull/lost-in-finetuning"
DEFAULT_REF = "main"
DEFAULT_EXTRAS = "gpu,data,eval"
JOB_FILE = REPO_ROOT / "colab" / "job.py"
# Remote paths without a leading slash: the CLI strips it anyway, and Git Bash rewrites "/content/...".
REMOTE_LOG = "content/lift_job.log"
REMOTE_SECRETS = "content/.lift_secrets.json"
REMOTE_STOP = "content/lift_job.stop"
DONE_MARKER = "@@LIFT_DONE "
TIMEOUT_CAP = 4_294_967  # threading.TIMEOUT_MAX on Windows: a larger --timeout overflows
TIMEOUTS = {"new": 900.0, "upload": 300.0, "download": 300.0, "stop": 180.0, "sessions": 120.0}
SECRET_VARS = ("HF_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
OPTIONAL_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LIFT_JUDGE_MAX_USD")
MIN_SECRET_LEN = 8  # as in colab/job.py: shorter values (a spend cap) are settings, not secrets
START_TIMEOUT = 900.0  # no job log this long after launch means the job never started
STALE_AFTER = 1200.0  # job.py logs a heartbeat every 5 minutes, so 20 silent minutes mean it is dead
STOP_WAIT = 900.0  # after a stop request, how long the job gets for its final push and last line
# colab/job.py's push schedule (a test keeps these equal): it kills a push running PUSH_LIMIT_SECONDS
# (SIGKILL KILL_GRACE_SECONDS after its SIGINT) and waits PUSH_DELAYS between the tries.
JOB_PUSH_LIMIT, JOB_KILL_GRACE, JOB_PUSH_DELAYS = 1800, 30, (60, 180, 360)
# After a stop request, how long a job whose log shows a push still running keeps its VM (55 minutes):
# one push stalled until job.py kills it, the waits of its retry schedule, and a margin for the kill
# grace and the other tries.
PUSH_WAIT = JOB_PUSH_LIMIT + sum(JOB_PUSH_DELAYS) + 900.0
RESCUE_WAIT = 2400.0  # the push retry from here: job.py kills a push after 30 minutes
JOB_MARGIN = 1800.0  # job.py stops itself this long before --max-hours (a quarter of a shorter cap)
MAX_SILENT_MINUTES = 45.0  # job.py kills a command that prints nothing for this long
PUSHED = (0, 3)  # lift sync push: done, or done with the conflicting copies kept under conflicts/
SYNC_REFUSED = 2  # lift sync: a problem no retry can fix (the token, the runs repo); never retried
SETUP_STEPS = ("spec", "secrets", "python", "fetch", "install", "sync-check")  # job.py's, before any line
# The commands that refuse a revision that is not a commit sha (lift.commands.run.require_pinned), and
# the config entry of each judge role; pinned_entries says which entries each line needs.
PINNED_COMMANDS = ("generate", "judge", "run-dev")
JUDGE_ENTRIES = {"primary": "judges.safety_primary", "second": "judges.safety_second"}
SHA_RE = re.compile(r"[0-9a-f]{40}")
_NO_REVISION = object()  # config.get's default: an entry without a revision key has nothing to pin
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_SETUP = 0, 1, 2, 3
EXIT_NOT_RELEASED, EXIT_NOT_PUSHED, EXIT_INTERRUPTED = 4, 5, 130
EXIT_CODES = """\
exit codes:
  0    the job succeeded, its outputs are pushed and the VM is released (a push that hit
       conflicts counts: the VM's copies are under conflicts/ in the runs repo, and it says so)
  1    a job line failed, or the job's status is unknown; or the runs repo check met a transient
       error (nothing was created: try again)
  2    lift remote refused to start: bad options, no login, a missing secret, or a runs repo that
       is missing, not private or not writable with this token (nothing was created)
  3    the job could not be set up: colab new or the secrets upload failed, or job.py's spec,
       secrets, python, fetch, pip or sync-check step; or a lift sync call on the VM exited 2 (a
       problem no retry can fix) before anything was lost
  4    the VM could not be confirmed released and may still be billing (this overrides every other
       code; lift remote stop says 0 or 4)
  5    the job ended but its outputs could not be pushed: its final push failed, and so did the
       retry from this laptop (none is tried for a push that exited 2)
  130  interrupted (Ctrl+C)
"""
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}")
ENDPOINT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
REF_RE = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./-]*")
SESSION_LINE = re.compile(r"^\[(?P<name>[^\]\n]*)\]\s+(?P<endpoint>\S+)\s+\|", re.MULTILINE)
# A failed upload or download prints the request URL, which carries the VM's proxy token.
URL_TOKEN = re.compile(r"(token)=[^&\s'\"]+", re.IGNORECASE)
# job.py's own lines in its log: "[lift-job] HH:MM:SS <message>".
JOB_NOTE = re.compile(rb"^\[lift-job\] \d\d:\d\d:\d\d (.*)$", re.MULTILINE)
# Its messages while a push runs or waits for its next try: a try starting, a heartbeat whose step is
# a push or that times the running push, the wait before the next try.
PUSH_NOTE = re.compile(r"^lift sync push\b|^alive [\d:]+: [^,]*\bpush\b|the push has run for|"
                       r"^the push failed .*trying again in")
LINE_NOTE = re.compile(r"^\$ lift (?!sync\b)\S")  # a job line other than lift sync starting


class RemoteError(RuntimeError):
    """A problem to report without a traceback. Code 2: bad input, missing login or secrets."""

    def __init__(self, message: str, code: int = EXIT_REFUSED):
        super().__init__(message)
        self.code = code


# --- the outside world ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    """A colab call's exit code (None when it timed out and was killed) and its stdout+stderr."""

    rc: int | None
    out: str


class Handle(Protocol):
    """A background process; subprocess.Popen fits."""

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class Runner(Protocol):
    def run(self, argv: list[str], *, timeout: float) -> Result: ...

    def spawn(self, argv: list[str], *, log_path: Path) -> Handle: ...


def colab_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The env of every colab call: UTF-8 I/O (on Windows the CLI otherwise reads job.py and writes its
    output as cp1252, and crashes on Hindi), a 60 s kernel request timeout instead of 10 s for a fresh
    VM, and none of our secrets (the CLI needs none)."""
    env = dict(os.environ if environ is None else environ)
    for var in SECRET_VARS:
        env.pop(var, None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("REQUEST_TIMEOUT", "60")
    return env


def own_group() -> dict[str, Any]:
    """Popen kwargs that keep the console's Ctrl+C away from a child process."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class SubprocessRunner:
    """Runs the real colab executable: list argv, stdin=DEVNULL, the UTF-8 env, an explicit timeout,
    its own process group. Killing colab.exe also kills its python child (uv's launcher). A CLI that
    cannot start (being reinstalled with `uv tool install ... --force`) is a failed call to ``run``,
    never an exception: polling goes on, and the VM is still stopped. ``spawn`` raises the OSError,
    which its callers count as a failed start (the job's setup, or the push retry)."""

    def __init__(self, environ: Mapping[str, str] | None = None, *, run: Callable[..., Any] | None = None,
                 popen: Callable[..., Any] | None = None):
        self.env = colab_env(environ)
        self._run = run or subprocess.run
        self._popen = popen or subprocess.Popen

    def run(self, argv: list[str], *, timeout: float) -> Result:
        try:
            done = self._run(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=self.env, timeout=timeout, encoding="utf-8",
                             errors="replace", **own_group())
        except subprocess.TimeoutExpired:
            return Result(None, f"timed out after {timeout:g} s and was killed")
        except OSError as e:
            return Result(None, f"could not start the Colab CLI: {type(e).__name__}: {e}")
        return Result(done.returncode, done.stdout or "")

    def spawn(self, argv: list[str], *, log_path: Path) -> Handle:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log:  # the child gets its own handle to the file
            return self._popen(list(argv), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               env=self.env, **own_group())


@dataclass(frozen=True)
class Colab:
    """The CLI's executable and the runner that calls it."""

    exe: str
    runner: Runner

    def argv(self, *args: str) -> list[str]:
        return [self.exe, *args]

    def call(self, *args: str, timeout: float | None = None) -> Result:
        return self.runner.run(self.argv(*args), timeout=timeout or TIMEOUTS[args[0]])

    def spawn(self, *args: str, log_path: Path) -> Handle:
        return self.runner.spawn(self.argv(*args), log_path=log_path)


def colab_config_dir() -> Path:
    """The CLI's state (sessions.json, token.json, history/, colab.log). ``~`` is USERPROFILE on
    Windows."""
    return Path(os.path.expanduser("~")) / ".config" / "colab-cli"


def find_colab() -> str | None:
    """``colab`` on PATH, else where ``uv tool install`` puts it (~/.local/bin, not on PATH on Windows)."""
    found = shutil.which("colab")
    if found:
        return found
    bin_dir = Path(os.path.expanduser("~")) / ".local" / "bin"
    for candidate in (bin_dir / "colab.exe", bin_dir / "colab"):
        if candidate.is_file():
            return str(candidate)
    return None


def cached_hf_token() -> str | None:
    """The token ``hf auth login`` stored, used when HF_TOKEN is unset. Never printed."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


class SafeOut:
    """The console as lift remote writes to it: a write error (``lift remote ... | head``, a closed
    console) must never stop the VM release, so after the first OSError every write is dropped. A
    character the console cannot encode is replaced instead."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.broken = False

    def write(self, text: str) -> int:
        if not self.broken:
            try:
                self.stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(self.stream, "encoding", None) or "ascii"
                self._try(lambda: self.stream.write(text.encode(encoding, "replace").decode(encoding)))
            except (OSError, ValueError):
                self.broken = True
        return len(text)

    def flush(self) -> None:
        if not self.broken:
            self._try(self.stream.flush)

    def _try(self, action: Callable[[], Any]) -> None:
        try:
            action()
        except (OSError, ValueError):
            self.broken = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


@dataclass
class Context:
    """Everything lift remote touches outside this process. Tests pass fakes; None means the real one."""

    runner: Runner | None = None
    environ: Mapping[str, str] | None = None
    out: TextIO | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    hf_token: Callable[[], str | None] | None = None
    colab_exe: str | None = None
    config_dir: Path | None = None

    def __post_init__(self) -> None:
        self.environ = dict(os.environ if self.environ is None else self.environ)
        self.out = self.out or sys.stdout
        self.hf_token = self.hf_token or cached_hf_token
        self.config_dir = self.config_dir or colab_config_dir()
        self.colab_exe = self.colab_exe or find_colab()
        self.runner = self.runner or SubprocessRunner(self.environ)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "out" and value is not None and not isinstance(value, SafeOut):
            value = SafeOut(value)
        super().__setattr__(name, value)


# --- the job --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    """The options of ``lift remote --gpu ... --job ...``."""

    gpu: str
    job: str
    ref: str = DEFAULT_REF
    repo: str = DEFAULT_REPO
    high_mem: bool = False
    max_hours: float = 12.0
    poll: float = 45.0
    name: str | None = None
    dry_run: bool = False
    max_silent_minutes: float = MAX_SILENT_MINUTES


@dataclass(frozen=True)
class JobSpec:
    """What colab/job.py runs. It travels base64-encoded in LIFT_JOB_B64, which the CLI writes into its
    local history and colab.log, so it never holds a secret. ``deadline`` (epoch seconds) is when the
    job stops itself; mode "push" with an ``attempt`` id asks job.py for one more push only."""

    repo: str
    ref: str
    lines: tuple[str, ...]
    job_id: str
    extras: str = DEFAULT_EXTRAS
    deadline: float | None = None
    max_silent_minutes: float = MAX_SILENT_MINUTES
    mode: str = "job"
    attempt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"repo": self.repo, "ref": self.ref, "lines": list(self.lines), "job_id": self.job_id,
                "extras": self.extras, "deadline": self.deadline,
                "max_silent_minutes": self.max_silent_minutes, "mode": self.mode, "attempt": self.attempt}

    def encode(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return base64.b64encode(text.encode("utf-8")).decode("ascii")


def validate_gpu(gpu: str | None) -> str:
    """A100, L4 or T4, case-insensitive. Checked here because the CLI silently turns an unknown
    ``--gpu`` into an A100."""
    value = (gpu or "").strip().upper()
    if value not in GPUS:
        raise RemoteError(f"--gpu must be one of {', '.join(GPUS)}, got {gpu!r} (the Colab CLI would "
                          "quietly turn an unknown value into an A100)")
    return value


def parse_job(text: str) -> list[str]:
    """``--job`` text -> lift lines. Lines split on newlines and on unquoted ";"; blank lines and "#"
    comments are dropped; each line must be a lift command, since job.py runs ``python -m lift ...``."""
    lines: list[str] = []
    for raw in (text or "").splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        lexer = shlex.shlex(raw, posix=True, punctuation_chars=";")
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            tokens = list(lexer)
        except ValueError as e:
            raise RemoteError(f"cannot parse the job line {raw.strip()!r}: {e}") from None
        words: list[str] = []
        for token in [*tokens, ";"]:
            if set(token) == {";"}:
                if words:
                    lines.append(_lift_line(words))
                words = []
            else:
                words.append(token)
    if not lines:
        raise RemoteError("--job has no lift lines")
    return lines


def _lift_line(words: list[str]) -> str:
    if len(words) < 2 or words[0] != "lift":
        raise RemoteError(f"every job line must be a lift command, got {shlex.join(words)!r}")
    if words[1] == "remote":
        raise RemoteError("a remote job cannot run lift remote")
    return shlex.join(words)


def check_repo_ref(repo: str, ref: str) -> None:
    if not REPO_RE.fullmatch(repo or ""):
        raise RemoteError(f"--repo must be owner/name, got {repo!r}")
    if not REF_RE.fullmatch(ref or "") or ".." in ref:
        raise RemoteError(f"--ref must be a branch, a tag or a full commit sha, got {ref!r}")


def check_limits(max_hours: float, poll: float, max_silent_minutes: float = MAX_SILENT_MINUTES) -> None:
    if not 0 < max_hours <= 24:
        raise RemoteError(f"--max-hours must be more than 0 and at most 24, got {max_hours:g}")
    if poll < 5:
        raise RemoteError(f"--poll must be at least 5 seconds, got {poll:g}")
    if not 0 < max_silent_minutes <= 24 * 60:
        raise RemoteError(f"--max-silent-minutes must be more than 0 and at most 1440, got "
                          f"{max_silent_minutes:g}")


def check_name(name: str) -> None:
    """Session names become file names (the CLI's history/<name>.jsonl, our manifest and logs)."""
    if not NAME_RE.fullmatch(name or ""):
        raise RemoteError(f"bad session name {name!r}: use letters, digits, '.', '_' and '-' (63 at most)")


def new_name(now: datetime | None = None) -> str:
    """lift-YYYYmmdd-HHMMSS-xxxx: unique, and safe as a file name."""
    return f"lift-{(now or datetime.now()):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def job_margin(max_hours: float) -> float:
    """How long before --max-hours job.py stops itself, so its final push fits: 30 minutes, or a
    quarter of a shorter cap."""
    return min(JOB_MARGIN, max_hours * 3600 / 4)


def exec_timeout(max_hours: float) -> int:
    """``colab exec --timeout``: past --max-hours and the longest wait for a stopped job (the VM is
    stopped by then), capped for Windows. Its expiry never stops the job; it only ends the client's
    wait."""
    return min(int(max_hours * 3600 + PUSH_WAIT) + 900, TIMEOUT_CAP)


def new_args(name: str, gpu: str, high_mem: bool) -> list[str]:
    return ["new", "-s", name, "--gpu", gpu, *(["--high-mem"] if high_mem else [])]


def upload_args(name: str, local: str, remote: str = REMOTE_SECRETS) -> list[str]:
    return ["upload", "-s", name, local, remote]


def exec_args(name: str, spec: JobSpec, timeout: int) -> list[str]:
    return ["exec", "-s", name, "-f", str(JOB_FILE), "--timeout", str(timeout),
            "--env", f"LIFT_JOB_B64={spec.encode()}"]


def download_args(name: str) -> list[str]:
    return ["download", "-s", name, REMOTE_LOG, str(log_path(name))]


# --- secrets and masking --------------------------------------------------------------------------


def basic_auth(token: str) -> str:
    """The credential job.py hands git: base64 of "x-access-token:<token>". Masked like the token."""
    return base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")


class Masker:
    """Replace each secret value (8+ characters) and the git credential built from GITHUB_TOKEN with
    "***". job.py masks the remote log the same way; this is the second line of defence."""

    def __init__(self, secrets: Mapping[str, str]):
        values = [value for value in secrets.values() if len(value) >= MIN_SECRET_LEN]
        if secrets.get("GITHUB_TOKEN"):
            values.append(basic_auth(secrets["GITHUB_TOKEN"]))
        self.values = sorted(set(values), key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for value in self.values:
            text = text.replace(value, "***")
        return text


def gather_secrets(ctx: Context) -> tuple[dict[str, str], list[str]]:
    """The values the VM needs, from this shell, and the names of the required ones that are missing.
    HF_TOKEN may also come from ``hf auth login``. None of them is ever printed."""
    from lift import config

    env = ctx.environ
    runs_var = config.get("storage.runs_repo_env", "LIFT_RUNS_REPO")
    values: dict[str, str] = {}
    missing: list[str] = []
    for var in ("GITHUB_TOKEN", runs_var):
        if env.get(var):
            values[var] = env[var]
        else:
            missing.append(var)
    hf = env.get("HF_TOKEN") or ctx.hf_token()
    if hf:
        values["HF_TOKEN"] = hf
    else:
        missing.append("HF_TOKEN")
    for var in OPTIONAL_VARS:
        if env.get(var):
            values[var] = env[var]
    return values, missing


def missing_message(missing: list[str]) -> str:
    example = "$env:NAME = '...'" if os.name == "nt" else "export NAME=..."
    return (f"missing {', '.join(missing)}: set them in this shell first ({example}); HF_TOKEN can also "
            "come from `hf auth login`")


def scrub(text: str, masker: Masker | None = None) -> str:
    """CLI output made safe to print: our secrets masked, and the VM's proxy token, which a failed
    upload or download prints inside the request URL."""
    return URL_TOKEN.sub(r"\1=***", masker(text) if masker else text)


def tail(text: str, masker: Masker | None = None, lines: int = 4, chars: int = 400) -> str:
    """The end of a CLI's output on one line, scrubbed, for a message."""
    end = " | ".join(line.strip() for line in scrub(text, masker).strip().splitlines()[-lines:])
    return end[-chars:] or "(no output)"


# --- small helpers --------------------------------------------------------------------------------


def say(ctx: Context, message: str) -> None:
    ctx.out.write(f"[lift remote] {message}\n")
    ctx.out.flush()


def shell_command(exe: str | None, *args: str) -> str:
    """How a person runs the CLI by hand: PowerShell's call operator on Windows."""
    if os.name == "nt":
        exe = exe or r"$env:USERPROFILE\.local\bin\colab.exe"
        return f'& "{exe}" {" ".join(args)}'
    return shlex.join([exe or "colab", *args])


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def log_path(name: str) -> Path:
    """The job log as last downloaded (outputs/logs/remote-{job_id}.log)."""
    return output_dir() / "logs" / f"remote-{name}.log"


def exec_log_path(name: str) -> Path:
    """What the background exec clients printed: the CLI's own lines and job.py's heartbeats."""
    return output_dir() / "logs" / f"remote-{name}.exec.log"


def manifest_path(name: str) -> Path:
    return manifests_dir() / f"remote-{name}.json"


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, path)


def load_manifest(name: str) -> dict[str, Any] | None:
    path = manifest_path(name)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RemoteError(f"{path} is not a JSON object")
    return data


def read_endpoint(config_dir: Path, name: str) -> str | None:
    """The VM endpoint the CLI recorded for ``name`` in sessions.json, or None. Read only; the record's
    proxy token is never read out."""
    path = config_dir / "sessions.json"
    for _ in range(3):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            records = json.loads(text) if text.strip() else {}
        except ValueError:  # caught while the CLI rewrites the file: read again
            time.sleep(0.5)
            continue
        record = records.get(name) if isinstance(records, dict) else None
        endpoint = record.get("endpoint") if isinstance(record, dict) else None
        return endpoint if isinstance(endpoint, str) and endpoint else None
    return None


def login_hint(ctx: Context) -> str:
    where = "PowerShell window" if os.name == "nt" else "terminal"
    return (f"the Colab CLI is not logged in ({ctx.config_dir / 'token.json'} is missing). Log in once, "
            f"in an interactive {where}: it prints a URL; sign in with the Google AI Pro account and "
            f"paste the code back:\n      {shell_command(ctx.colab_exe, 'sessions')}")


def preflight(ctx: Context, *, need_job: bool = True) -> list[str]:
    """What stops a run before any call: no CLI, no login, no job file."""
    problems = []
    if not ctx.colab_exe:
        problems.append("the Colab CLI is not installed: uv tool install google-colab-cli==0.7.5")
    if not (ctx.config_dir / "token.json").is_file():
        problems.append(login_hint(ctx))
    if need_job and not JOB_FILE.is_file():
        problems.append(f"{JOB_FILE} is missing")
    return problems


def ready_colab(ctx: Context) -> Colab:
    problems = preflight(ctx, need_job=False)
    if problems:
        raise RemoteError("cannot reach Colab:\n  - " + "\n  - ".join(problems))
    return Colab(ctx.colab_exe, ctx.runner)


@contextmanager
def _sigint_ignored() -> Iterator[None]:
    """Hold off Ctrl+C while the VM is being stopped: a second Ctrl+C would otherwise cut the cleanup
    short and leave the VM burning units. Every colab call is bounded by its timeout."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


# --- the job's log and status ---------------------------------------------------------------------

Accept = Callable[[dict[str, Any]], bool]  # picks the @@LIFT_DONE line to wait for


def is_job_status(status: dict[str, Any]) -> bool:
    """The job's own @@LIFT_DONE line, not the line of a push retry."""
    return status.get("mode", "job") == "job"


def find_status(data: bytes, accept: Accept = is_job_status) -> dict[str, Any] | None:
    """The status on the last "@@LIFT_DONE {json}" line (at a line start) that ``accept`` takes, or
    None while the job runs or the line is still half-written."""
    marker = DONE_MARKER.encode()
    for line in reversed(data.split(b"\n")):
        if not line.startswith(marker):
            continue
        try:
            status = json.loads(line[len(marker):])
        except ValueError:
            continue
        if isinstance(status, dict) and accept(status):
            return status
    return None


def installed(status: dict[str, Any]) -> bool:
    """Whether lift got installed on the VM (a status without the field: no setup step failed)."""
    value = status.get("installed")
    return status.get("failed_step") not in SETUP_STEPS if value is None else bool(value)


def push_failed(status: dict[str, Any]) -> bool:
    """Lift got installed, a line other than lift sync ran (job.py's "ran"; a status without it: maybe),
    and the final push did not end 0 or 3 (a missing rc included): the outputs since the last good
    push exist only on the VM."""
    return installed(status) and status.get("ran") != 0 and status.get("push_rc") not in PUSHED


def rescuable(status: dict[str, Any]) -> bool:
    """A failed final push that one more push from here may fix: not one that exited 2."""
    return push_failed(status) and status.get("push_rc") != SYNC_REFUSED


def lines_ok(status: dict[str, Any]) -> bool:
    """Every line ran and succeeded. job.py turns a failed push into the job's rc, so a status whose
    only failure is the push counts too: the push is judged on its own."""
    if status.get("failed_step") is not None or status.get("failed_line") is not None:
        return False
    if status.get("stopped") or status.get("sync_refused"):
        return False
    rc = status.get("rc")
    return (rc == 0 and not isinstance(rc, bool)) or push_failed(status)


def exit_code(status: dict[str, Any] | None, *, outcome: str = "done", released: bool | None = True,
              rescue: dict[str, Any] | None = None) -> int:
    """lift remote's exit code (EXIT_CODES): 4 while the VM may run, then 5 when the outputs could not
    be pushed (``rescue`` is the push retry's status), 130 after Ctrl+C, 3 when the job could not be
    set up or lift sync refused it (exit 2), 1 when a line failed or the status is unknown, else 0."""
    if released is not True:
        return EXIT_NOT_RELEASED
    rescued = rescue is not None and rescue.get("push_rc") in PUSHED
    if status is not None and push_failed(status) and not rescued:
        return EXIT_NOT_PUSHED
    if outcome == "interrupted":
        return EXIT_INTERRUPTED
    if outcome == "colab new failed" or (status is not None and (
            status.get("failed_step") in SETUP_STEPS or status.get("sync_refused"))):
        return EXIT_SETUP
    return EXIT_OK if status is not None and lines_ok(status) else EXIT_FAILED


def final_code(status: dict[str, Any] | None, *, outcome: str, released: bool | None,
               rescue: dict[str, Any] | None, error_code: int | None = None) -> int:
    """exit_code, then the code of the error that ended the run, unless 4 or 5 overrides it."""
    code = exit_code(status, outcome=outcome, released=released, rescue=rescue)
    if error_code is not None and code not in (EXIT_NOT_RELEASED, EXIT_NOT_PUSHED):
        code = error_code
    return code


def describe(status: dict[str, Any]) -> str:
    parts = [f"job finished: rc {status.get('rc')}"]
    if status.get("stopped"):
        parts.append(f"stopped ({status['stopped']})")
    if status.get("failed_step"):
        parts.append(f"failed at {status['failed_step']}")
    if isinstance(status.get("failed_line"), int):
        parts.append(f"line {status['failed_line'] + 1} failed")
    if status.get("sync_refused"):
        parts.append(f"lift sync exited 2 at {status['sync_refused']}: a problem no retry can fix (the log "
                     "says which)")
    if status.get("killed") and status.get("killed") != status.get("stopped"):
        parts.append(f"killed: {status['killed']}")
    if status.get("ran") == 0 and installed(status):
        parts.append("no line but lift sync ran, so there was nothing new to push")
    parts.append(f"final push rc {status.get('push_rc')}")
    if status.get("log_dropped"):
        parts.append(f"{status['log_dropped']} bytes of its log could not be written")
    if status.get("checkpoint_push_failures"):
        parts.append(f"{status['checkpoint_push_failures']} checkpoint push(es) failed")
    if status.get("push_conflicts"):
        parts.append(f"{status['push_conflicts']} push(es) hit conflicts: the VM's copies are under "
                     "conflicts/ in the runs repo")
    total = (status.get("seconds") or {}).get("total")
    if isinstance(total, int | float):
        parts.append(f"{hms(total)} on the VM")
    return ", ".join(parts)


class LogPrinter:
    """Print the part of the downloaded log not printed yet: whole lines only (a half-written line or
    UTF-8 character waits for the next download), masked."""

    def __init__(self, out: TextIO, masker: Masker):
        self.out = out
        self.masker = masker
        self.seen = 0

    def feed(self, data: bytes, *, final: bool = False) -> None:
        if len(data) < self.seen:
            self.out.write("[lift remote] the remote log got shorter; printing it again\n")
            self.seen = 0
        end = len(data) if final else data.rfind(b"\n") + 1
        if end > self.seen:
            self.out.write(self.masker(data[self.seen:end].decode("utf-8", "replace")))
            if final and not data.endswith(b"\n"):
                self.out.write("\n")
            self.out.flush()
            self.seen = end


def job_pushing(data: bytes) -> bool:
    """Whether job.py's last word in its log (a job.py line, or the job's last line) is about a push
    that runs or waits for its next try. Its heartbeat comes every 5 minutes and names the step, so a
    push that ended since is followed by another job.py line: the next try, or the last line."""
    marker = DONE_MARKER.encode()
    for line in reversed(data.split(b"\n")):
        if line.startswith(marker):
            return False
        note = JOB_NOTE.match(line)
        if note:
            return PUSH_NOTE.search(note.group(1).decode("utf-8", "replace")) is not None
    return False


def job_lines_ran(data: bytes) -> bool:
    """Whether job.py's log shows a line other than a lift sync line starting: only those leave new
    files on the VM."""
    return any(LINE_NOTE.match(note.decode("utf-8", "replace")) for note in JOB_NOTE.findall(data))


def session_gone(out: str) -> bool:
    """The CLI's words for a session it no longer has (a pruned record, or a 404/401 from the VM)."""
    return "appears to be lost" in out or re.search(r"Session '[^']*' not found", out) is not None


def show_exec_tail(ctx: Context, name: str, masker: Masker, lines: int = 20) -> None:
    path = exec_log_path(name)
    if not path.is_file():
        return
    say(ctx, f"the last lines of {path}:")
    for line in path.read_bytes().decode("utf-8", "replace").splitlines()[-lines:]:
        ctx.out.write(f"    {scrub(line, masker)}\n")
    ctx.out.flush()


class Follower:
    """Follow the job log: download it every ``poll`` seconds, print what is new, and look for an
    @@LIFT_DONE line that ``accept`` takes. One instance per run, so a later wait (after a stop
    request, or for the push retry) goes on where the last one left off.

    ``follow`` returns (outcome, status): ("done", the status), or ("timeout" | "lost" | "no-start" |
    "no-marker" | "stale", None). "lost": the VM is gone; "no-start": no job log ever appeared;
    "no-marker": the job cell ended without writing its last line; "stale": the log stopped growing
    although job.py logs a heartbeat every 5 minutes. A non-zero exit of the exec client alone means
    only that its connection dropped: the job runs on, so polling goes on. The last line also counts
    when it reached only the exec client's output (job.py prints it there when its log cannot take
    it).
    """

    def __init__(self, colab: Colab, name: str, endpoint: str | None, *, ctx: Context, masker: Masker):
        self.colab, self.name, self.endpoint, self.ctx, self.masker = colab, name, endpoint, ctx, masker
        self.local = log_path(name)
        self.local.parent.mkdir(parents=True, exist_ok=True)
        self.printer = LogPrinter(ctx.out, masker)
        self.seen_log = False
        self.size = -1
        self.data = b""  # the log as last downloaded
        self.restart(None)

    def pushing(self) -> bool:
        return job_pushing(self.data)

    def lines_ran(self) -> bool:
        return job_lines_ran(self.data)

    def restart(self, handle: Handle | None, accept: Accept = is_job_status) -> None:
        """Wait from now on for a new last line, written by the cell behind ``handle``."""
        self.handle, self.accept = handle, accept
        self.exec_rc: int | None = None
        self.grace = self.failures = 0
        self.started = self.grown_at = self.ctx.clock()

    def follow(self, deadline: float, poll: float) -> tuple[str, dict[str, Any] | None]:
        """Poll until a last line is found or ``deadline``; then read the log once more, since the job
        may have finished while nobody looked."""
        check_every = max(1, round(600 / poll))  # `colab sessions` about every 10 minutes
        polls = 0
        while self.ctx.clock() < deadline:
            self.ctx.sleep(poll)
            polls += 1
            outcome = (self._download() or self._exec_status() or self._check_exec()
                       or self._check_vm(polls % check_every == 0))
            if outcome:
                return outcome
        return self._download() or self._exec_status() or ("timeout", None)

    def _download(self) -> tuple[str, dict[str, Any] | None] | None:
        res = self.colab.call(*download_args(self.name))
        if res.rc == 0:
            self.failures = 0
            self.seen_log = True
            data = self.data = self.local.read_bytes()
            if len(data) > self.size:
                self.size, self.grown_at = len(data), self.ctx.clock()
            status = find_status(data, self.accept)
            self.printer.feed(data, final=status is not None)
            if status is not None:
                return "done", status
            found = self._exec_status()
            if found:
                return found
            if self.ctx.clock() - self.grown_at > STALE_AFTER:
                say(self.ctx, f"the job log has not grown for {STALE_AFTER / 60:.0f} minutes, although "
                              "job.py logs a heartbeat every 5: the job is dead")
                show_exec_tail(self.ctx, self.name, self.masker)
                return "stale", None
        elif "File or directory not found" in res.out:
            self.failures = 0  # job.py has not created its log yet
        elif session_gone(res.out):
            say(self.ctx, f"the Colab CLI no longer has session {self.name}: {tail(res.out, self.masker)}")
            return "lost", None
        else:
            self.failures += 1
            say(self.ctx, f"log download failed ({self.failures} in a row, rc {res.rc}): "
                          f"{tail(res.out, self.masker)}")
            if self.failures % 3 == 0 and is_listed(self.colab, self.name, self.endpoint, self.ctx) is False:
                say(self.ctx, "the VM is no longer listed by `colab sessions`")
                return "lost", None
        return None

    def _exec_status(self) -> tuple[str, dict[str, Any]] | None:
        """The last line from the exec clients' output, where job.py prints it when its log cannot take
        it (a full disk). Only job.py's own messages reach that output, never the commands'."""
        try:
            data = exec_log_path(self.name).read_bytes()
        except OSError:
            return None
        status = find_status(data, self.accept)
        if status is None:
            return None
        say(self.ctx, "the job's last line came through the exec client's output: its log could not take it")
        return "done", status

    def _check_exec(self) -> tuple[str, None] | None:
        if self.handle is not None and self.exec_rc is None:
            self.exec_rc = self.handle.poll()
            if self.exec_rc == 0:
                say(self.ctx, "the job cell has ended (the exec client exited 0)")
            elif self.exec_rc is not None:
                say(self.ctx, f"the exec client exited with rc {self.exec_rc} (connection lost?); the job "
                              "may still run, so the log is still polled")
        if self.exec_rc is not None and (self.exec_rc == 0 or not self.seen_log):
            self.grace += 1
            if self.grace > 2:
                say(self.ctx, "the job cell ended without its @@LIFT_DONE line" if self.seen_log
                    else "the job never started: no job log appeared before the exec client exited")
                show_exec_tail(self.ctx, self.name, self.masker)
                return ("no-marker" if self.seen_log else "no-start"), None
        if not self.seen_log and self.ctx.clock() - self.started > START_TIMEOUT:
            say(self.ctx, f"no job log after {START_TIMEOUT / 60:.0f} minutes: the job never started")
            show_exec_tail(self.ctx, self.name, self.masker)
            return "no-start", None
        return None

    def _check_vm(self, now: bool) -> tuple[str, None] | None:
        if not now:
            return None
        listed = is_listed(self.colab, self.name, self.endpoint, self.ctx)
        if listed is False:
            say(self.ctx, "the VM is no longer listed by `colab sessions` (reclaimed, or stopped elsewhere)")
            return "lost", None
        say(self.ctx, f"{hms(self.ctx.clock() - self.started)} since launch; the VM is "
                      f"{'listed' if listed else 'unchecked (colab sessions failed)'}")
        return None


def send_stop(colab: Colab, name: str, reason: str, ctx: Context, masker: Masker) -> bool:
    """Upload content/lift_job.stop, which job.py looks for every few seconds. Returns whether it went
    through."""
    fd, tmp = tempfile.mkstemp(prefix="lift-stop-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as f:
            f.write(reason + "\n")
        for attempt in (1, 2):
            res = colab.call(*upload_args(name, tmp, REMOTE_STOP))
            if res.rc == 0:
                return True
            say(ctx, f"the stop request did not go up (try {attempt}/2, rc {res.rc}): "
                     f"{tail(res.out, masker)}")
        return False
    finally:
        Path(tmp).unlink(missing_ok=True)


def wind_down(colab: Colab, follower: Follower, reason: str, ctx: Context, masker: Masker,
              poll: float) -> tuple[str, dict[str, Any] | None]:
    """Ask the job to stop, then wait up to STOP_WAIT for its last line: job.py stops the running line
    (never a push), skips the rest and runs its final push. While its log then shows a push still
    running (job.py kills a stalled one only after 30 minutes, and tries it again), wait on, up to
    PUSH_WAIT after the request. A KeyboardInterrupt here, a second Ctrl+C, is the caller's: it stops
    the VM at once."""
    if not send_stop(colab, follower.name, reason, ctx, masker):
        say(ctx, "could not ask the job to stop: the VM is stopped now")
        return "stop-failed", None
    asked, poll = ctx.clock(), min(poll, 30.0)
    say(ctx, f"asked the job to stop ({reason}); waiting up to {STOP_WAIT / 60:.0f} minutes for its final "
             f"push, up to {PUSH_WAIT / 60:.0f} while its log shows a push running")
    outcome, status = follower.follow(asked + STOP_WAIT, poll)
    if outcome == "timeout" and follower.pushing():
        say(ctx, f"the job's log shows a push still running: waiting for it, up to {PUSH_WAIT / 60:.0f} "
                 "minutes after the stop request")
        outcome, status = follower.follow(asked + PUSH_WAIT, poll)
    if outcome == "timeout":
        say(ctx, f"no last line from the job within {hms(ctx.clock() - asked)} of the stop request")
    return outcome, status


def rescue_push(colab: Colab, follower: Follower, spec: JobSpec, secrets: Mapping[str, str], ctx: Context,
                masker: Masker, poll: float, handles: list[Handle],
                why: str = "the job's final push failed") -> dict[str, Any]:
    """The one push retry from here, after the job's own final push failed (or the job ended with no
    last line after running lines): upload the secrets again and run colab/job.py in push mode, which
    reads and deletes them, runs ``lift sync push`` once in /content/lift and appends its own
    @@LIFT_DONE line, tagged with the attempt id. Returns that line's status (push_rc None and an error
    when it never came, or when the exec client could not start: a failed retry, never an exception)."""
    attempt = uuid.uuid4().hex[:8]
    say(ctx, f"{why}: trying a push once more from here (push retry {attempt})")
    try:
        upload_secrets(colab, follower.name, secrets, ctx, masker)
    except RemoteError as e:
        say(ctx, f"the push retry cannot start: {e}")
        return {"attempt": attempt, "push_rc": None, "error": str(e)}
    push_spec = replace(spec, lines=(), deadline=None, mode="push", attempt=attempt)
    try:
        handle = colab.spawn(*exec_args(follower.name, push_spec, int(RESCUE_WAIT) + 900),
                             log_path=exec_log_path(follower.name))
    except OSError as e:  # colab.exe being replaced (uv tool install ... --force), or a locked exec log
        error = f"colab exec could not start: {error_text(e, masker)}"
        say(ctx, f"the push retry cannot start: {error}")
        return {"attempt": attempt, "push_rc": None, "error": error}
    handles.append(handle)
    follower.restart(handle, accept=lambda s: s.get("mode") == "push" and s.get("attempt") == attempt)
    outcome, status = follower.follow(ctx.clock() + RESCUE_WAIT, poll)
    end_exec(handle, finished=outcome == "done")
    if status is None:
        say(ctx, f"the push retry gave no answer ({outcome})")
        return {"attempt": attempt, "push_rc": None, "error": outcome}
    say(ctx, f"push retry {attempt}: push rc {status.get('push_rc')}")
    return status


def lost_alarm(ctx: Context, name: str, status: dict[str, Any], rescue: dict[str, Any] | None) -> None:
    """The alarm for exit 5. ``rescue`` is the push retry's status: None when none was tried, or when
    an error cut it short."""
    bar = "!" * 78
    if status.get("push_rc") == SYNC_REFUSED:
        how = ("  the job's final push exited 2, a problem no retry can fix (the log says which: a token "
               "that cannot write, the runs repo missing or public, ...), so none was tried; the VM is "
               "stopped.")
        fix = "  Fix that first (lift sync check --write), then rerun the job: it resumes by cache key."
    else:
        retry = "no retry from here pushed them" if rescue is None else "so did the retry from this laptop"
        how = f"  the job's final push failed, {retry}, and the VM is stopped."
        fix = "  Rerun the job: it resumes by cache key."
    for line in (bar, f"THE OUTPUTS OF {name} SINCE ITS LAST SUCCESSFUL PUSH ARE LOST", how,
                 "  The runs repo holds what the earlier pushes sent.", fix, bar):
        say(ctx, line)


def kept_on_vm(ctx: Context, name: str, status: dict[str, Any], rescue: dict[str, Any] | None) -> None:
    """watch --no-stop after a failed final push: the outputs are still on the VM it leaves running."""
    say(ctx, "the job's final push failed: the outputs since its last successful push are still on the VM, "
             "which keeps running (--no-stop)")
    if status.get("push_rc") == SYNC_REFUSED:
        say(ctx, "  the push exited 2, a problem no retry can fix (the log says which), so lift remote does "
                 f"not retry it; lift remote stop {name} releases the VM, and the outputs with it")
    elif rescue is not None:
        say(ctx, f"  the push retry from this laptop failed already; lift remote stop {name} releases the "
                 "VM, and the outputs with it")
    else:
        say(ctx, f"  lift remote watch {name} retries the push from this laptop, then stops the VM; lift "
                 f"remote stop {name} releases the VM without that retry, and the outputs with it")


# --- stopping -------------------------------------------------------------------------------------


def list_sessions(colab: Colab, ctx: Context, tries: int = 2) -> list[tuple[str, str]] | None:
    """(name, endpoint) for each VM ``colab sessions`` (which asks the server) lists, name "?" for one
    the CLI has no record of; None when the call keeps failing."""
    for attempt in range(1, tries + 1):
        res = colab.call("sessions")
        if res.rc == 0:
            return [(m["name"], m["endpoint"]) for m in SESSION_LINE.finditer(res.out)]
        if attempt < tries:
            ctx.sleep(10)
    return None


def unknown_endpoints(sessions: list[tuple[str, str]] | None) -> set[str] | None:
    """The endpoints of the VMs the CLI has no record of ("[?]" lines)."""
    return None if sessions is None else {endpoint for name, endpoint in sessions if name == "?"}


def listed_in(sessions: list[tuple[str, str]] | None, name: str, endpoint: str | None,
              ignore: frozenset[str] = frozenset()) -> bool | None:
    """Whether the VM is in the listing: by endpoint when known, else by name. Without an endpoint, a
    "[?]" VM (outside ``ignore``) may be this one, so the answer is None, as when the listing failed."""
    if sessions is None:
        return None
    if endpoint:
        return any(ep == endpoint for _, ep in sessions)
    if any(n == name for n, _ in sessions):
        return True
    if any(n == "?" and ep not in ignore for n, ep in sessions):
        return None
    return False


def is_listed(colab: Colab, name: str, endpoint: str | None, ctx: Context, tries: int = 2) -> bool | None:
    return listed_in(list_sessions(colab, ctx, tries), name, endpoint)


def force_stop(colab: Colab, name: str, endpoint: str) -> Result:
    """Stop a VM the CLI has no record of. A throwaway sessions file holding only the endpoint makes
    ``colab --config FILE stop`` look the VM up on the server and unassign it. The CLI then writes a
    live proxy token into that file, so it is deleted at once."""
    fd, tmp = tempfile.mkstemp(prefix="lift-colab-", suffix=".json")
    try:
        record = {"name": name, "token": "none", "url": "http://invalid", "endpoint": endpoint,
                  "token_expires_at": None}
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({name: record}, f)
        return colab.call("--config", tmp, "stop", "-s", name, timeout=TIMEOUTS["stop"])
    finally:
        for path in (tmp, tmp + ".lock"):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


def alarm(ctx: Context, name: str, endpoint: str | None, *, checked: bool, unknown: list[str] = ()) -> None:
    bar = "!" * 78
    lines = [bar, f"THE VM {name} MAY STILL BE RUNNING AND BURNING COMPUTE UNITS"
             + ("" if checked else " (colab sessions could not be checked)")]
    for other in unknown:
        lines += [f"  `colab sessions` lists {other}, a VM the CLI has no record of: maybe this one.",
                  f"  stop it: lift remote stop {name} --endpoint {other}"]
    lines += [f"  retry:   lift remote stop {name}",
              f"  check:   {shell_command(ctx.colab_exe, 'sessions')}"
              + (f"   (look for {endpoint})" if endpoint else ""),
              "  browser: colab.research.google.com > Runtime > Manage sessions > Terminate",
              bar]
    for line in lines:
        say(ctx, line)


def release(colab: Colab, name: str, endpoint: str | None, ctx: Context, attempts: int = 3,
            ignore: frozenset[str] = frozenset()) -> bool | None:
    """Stop the VM, then check that ``colab sessions`` no longer lists it. True: gone; False: still
    listed; None: cannot tell (the listing failed, or with no endpoint known it shows a VM the CLI has
    no record of, outside ``ignore``). ``colab stop`` alone proves nothing: it exits 0 for a name the
    CLI has forgotten, and its teardown swallows errors."""
    say(ctx, f"stopping the VM {name} (Ctrl+C is held off until this is done)")
    for attempt in range(1, attempts + 1):
        res = colab.call("stop", "-s", name)
        if res.rc == 0:
            break
        say(ctx, f"colab stop failed (try {attempt}/{attempts}, rc {res.rc}): {tail(res.out)}")
        if attempt < attempts:
            ctx.sleep(10 * attempt)
    sessions = list_sessions(colab, ctx)
    if endpoint is None and sessions:
        endpoint = next((ep for n, ep in sessions if n == name), None)  # the CLI still lists it by name
    listed = listed_in(sessions, name, endpoint, ignore)
    if listed and endpoint:
        say(ctx, "still listed: stopping it by its endpoint")
        res = force_stop(colab, name, endpoint)
        if res.rc != 0:
            say(ctx, f"stopping by endpoint failed (rc {res.rc}): {tail(res.out)}")
        sessions = list_sessions(colab, ctx)
        listed = listed_in(sessions, name, endpoint, ignore)
    if listed is False:
        say(ctx, "VM released: `colab sessions` no longer lists it" if endpoint
            else "no VM of this job is listed by `colab sessions`")
        return True
    unknown = [] if endpoint else sorted((unknown_endpoints(sessions) or set()) - ignore)
    alarm(ctx, name, endpoint, checked=sessions is not None, unknown=unknown)
    return None if listed is None else False


def end_exec(handle: Handle, *, finished: bool) -> None:
    """Make sure a background exec client is gone before the VM is stopped. It exits by itself once
    the job cell ends; otherwise it would wait out its --timeout against a dead VM."""
    if finished:
        try:
            handle.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            pass
    if handle.poll() is None:
        handle.kill()
        try:
            handle.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def upload_secrets(colab: Colab, name: str, secrets: Mapping[str, str], ctx: Context, masker: Masker,
                   attempts: int = 4) -> None:
    """Send the secrets as a file: the one channel that stays out of the CLI's history and colab.log.
    The local temp file is deleted in ``finally``; job.py deletes the remote copy before anything
    else. Retries also cover a fresh VM that does not answer at once."""
    fd, tmp = tempfile.mkstemp(prefix="lift-secrets-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(secrets), f)
        for attempt in range(1, attempts + 1):
            res = colab.call(*upload_args(name, tmp))
            if res.rc == 0:
                say(ctx, f"secrets uploaded: {', '.join(sorted(secrets))}")
                return
            say(ctx, f"secrets upload failed (try {attempt}/{attempts}, rc {res.rc}): "
                     f"{tail(res.out, masker)}")
            if attempt < attempts:
                ctx.sleep(15 * 2 ** (attempt - 1))
        raise RemoteError(f"could not upload the secrets in {attempts} tries", code=EXIT_SETUP)
    finally:
        Path(tmp).unlink(missing_ok=True)


# --- the commands ---------------------------------------------------------------------------------


def launch(opts: Launch, ctx: Context | None = None) -> int:
    """``lift remote --gpu G --job '...'``: run the job on a fresh VM, stream its log, stop the VM.

    Returns an exit code from EXIT_CODES. Bad options, a missing login or a missing secret raise
    RemoteError (code 2) before anything is called.
    """
    ctx = ctx or Context()
    gpu = validate_gpu(opts.gpu)
    lines = parse_job(opts.job)
    check_repo_ref(opts.repo, opts.ref)
    check_limits(opts.max_hours, opts.poll, opts.max_silent_minutes)
    name = opts.name or new_name()
    check_name(name)
    if read_endpoint(ctx.config_dir, name) is not None:
        raise RemoteError(f"the Colab CLI already has a session named {name}: reusing a name orphans its VM")
    if manifest_path(name).exists():
        raise RemoteError(f"{manifest_path(name)} exists: pick another --name")
    spec = JobSpec(opts.repo, opts.ref, tuple(lines), name, max_silent_minutes=opts.max_silent_minutes)
    secrets, missing = gather_secrets(ctx)
    problems = preflight(ctx) + ([missing_message(missing)] if missing else [])
    colab = Colab(ctx.colab_exe or "colab", ctx.runner)
    warning = unpinned_warning(lines)
    if warning:
        say(ctx, warning)
    if opts.dry_run:
        print_plan(ctx, colab, opts, gpu, spec, secrets, problems)
        return EXIT_REFUSED if problems else EXIT_OK
    if problems:
        raise RemoteError("cannot start:\n  - " + "\n  - ".join(problems))
    check_runs_repo(ctx, Masker(secrets))
    if not any(shlex.split(line)[:3] == ["lift", "sync", "pull"] for line in lines):
        say(ctx, "note: no `lift sync pull` line, so the VM has no data/ or outputs/ from the runs repo")
    return _run(colab, ctx, opts, gpu, spec, secrets)


def unpinned_warning(lines: list[str]) -> str | None:
    """A warning when a line's command would refuse a revision in this laptop's configs/project.yaml
    that is not a commit sha (pinned_entries says which it checks): on the VM that refusal comes only
    after the setup. Only a warning: the VM reads the ref's copy of the file, which may differ. An
    entry without a revision key is skipped."""
    from lift import config

    found = []
    try:
        for i, line in enumerate(lines, 1):
            loose = []
            for entry in pinned_entries(line):
                rev = config.get(entry + ".revision", _NO_REVISION)
                if rev is not _NO_REVISION and not SHA_RE.fullmatch(str(rev)):
                    loose.append(f"{entry}.revision = {rev!r}")
            if loose:
                found.append(f"line {i} (lift {shlex.split(line)[1]}): {', '.join(loose)}")
    except Exception as e:  # advice only: it never blocks a launch
        return f"warning: could not read configs/project.yaml to check the pins ({type(e).__name__})"
    if not found:
        return None
    return ("warning: unpinned revisions in this laptop's configs/project.yaml, which these lines refuse: "
            + "; ".join(found) + ". Pin the shas Gate 0 printed (gate0.json, 'revisions') with a dated "
            "docs/decisions.md entry, commit, and push to --ref: the VM reads the ref's copy, not this one.")


def pinned_entries(line: str) -> list[str]:
    """The config entries whose revision a job line's command refuses unless it is a commit sha, as its
    require_pinned call checks (lift.commands.run): models.global for generate and run-dev, and the
    entry of the judge the line runs (lift judge's --judge, primary by default; run-dev judges with the
    primary) for judge and run-dev, only when that judge is local (Qwen3Guard): a frontier judge has no
    revision. A line that does not parse needs none: lift refuses it before any pin check."""
    from lift.judge.pipeline import judge_dir

    args = parse_line(line)
    if args is None:
        return []
    entries = [] if args.command == "judge" else ["models.global"]
    if args.command != "generate":
        role = args.judge if args.command == "judge" else "primary"
        if judge_dir(role) == "qwen3guard":
            entries.append(JUDGE_ENTRIES[role])
    return entries


def parse_line(line: str) -> argparse.Namespace | None:
    """A generate, judge or run-dev line as lift.commands.run's own parser reads it (defaults, ``=`` and
    abbreviations as on the VM), or None for any other line or one that does not parse. argparse's own
    output (usage, help) is swallowed."""
    words = shlex.split(line)[1:]
    if not words or words[0] not in PINNED_COMMANDS:
        return None
    from lift.commands import run

    parser = argparse.ArgumentParser(prog="lift")
    run.register(parser.add_subparsers(dest="command"))
    sink = io.StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            return parser.parse_args(words)
    except SystemExit:
        return None


def check_runs_repo(ctx: Context, masker: Masker) -> None:
    """lift.sync.preflight(write=True), from here and before any VM exists: the runs repo exists, is
    private and takes this token's writes. A problem no retry can fix refuses (2); a transient error
    exits 1. The message is lift sync's, which never holds a token, masked all the same."""
    from lift import sync

    say(ctx, "checking the runs repo: it exists, is private and takes this token's writes")
    try:
        sync.preflight(write=True)
    except ImportError as e:
        raise RemoteError(f"cannot check the runs repo: {e.name or 'huggingface_hub'} is not installed in "
                          "this venv (pip install -e \".[sync]\")") from None
    except sync.SyncError as e:
        permanent = bool(getattr(e, "permanent", False))
        what = "the runs repo check failed" + ("" if permanent else " (a transient error: try again)")
        code = EXIT_REFUSED if permanent else EXIT_FAILED
        raise RemoteError(f"{what}: {masker(str(e))}", code=code) from None


def print_plan(ctx: Context, colab: Colab, opts: Launch, gpu: str, spec: JobSpec, secrets: Mapping[str, str],
               problems: list[str]) -> None:
    masker = Masker(secrets)
    name = spec.job_id
    say(ctx, "dry run: nothing is called")
    say(ctx, f"session {name}: Colab {gpu}{' high-mem' if opts.high_mem else ''}, {spec.repo}@{spec.ref}, "
             f"stopped after {opts.max_hours:g} h at most")
    for i, line in enumerate(spec.lines, 1):
        say(ctx, f"  line {i}: {line}")
    shown = ", ".join(f"{var}=***" for var in sorted(secrets)) or "none"
    say(ctx, f"secrets, sent as a temp file that is deleted on both sides: {shown}")
    say(ctx, "  0. check the runs repo as `lift sync check --write` does: it exists, is private and takes "
             "this token's writes")
    steps = ((["sessions"], "   (VMs the CLI has no record of, before this one exists)"),
             (new_args(name, gpu, opts.high_mem), ""),
             (upload_args(name, "<temp secrets file>"), ""),
             (exec_args(name, spec, exec_timeout(opts.max_hours)), "   (in the background)"),
             (download_args(name), f"   (every {opts.poll:g} s until the @@LIFT_DONE line)"),
             (["stop", "-s", name], ""),
             (["sessions"], "   (checks that the VM is gone)"))
    call = "& " if os.name == "nt" else ""  # PowerShell runs a quoted path only with its call operator
    for i, (args, note) in enumerate(steps, 1):
        say(ctx, masker(f"  {i}. {call}{shlex.join(colab.argv(*args))}{note}"))
    for problem in problems:
        say(ctx, f"problem: {problem}")


@dataclass
class RunState:
    """What one launch knows so far; the manifest and the exit code are built from it."""

    outcome: str = "not started"
    status: dict[str, Any] | None = None
    endpoint: str | None = None
    new_called: bool = False
    before: frozenset[str] = frozenset()  # "[?]" VMs listed before colab new: not ours
    orphans: list[str] = field(default_factory=list)
    handles: list[Handle] = field(default_factory=list)
    rescue: dict[str, Any] | None = None
    error_code: int | None = None
    forced: bool = False


def _run(colab: Colab, ctx: Context, opts: Launch, gpu: str, spec: JobSpec, secrets: dict[str, str]) -> int:
    name = spec.job_id
    masker = Masker(secrets)
    started = ctx.clock()
    spec = replace(spec, deadline=started + opts.max_hours * 3600 - job_margin(opts.max_hours))
    path = manifest_path(name)
    manifest: dict[str, Any] = {
        "run_id": name, "stage": "remote", "name": name, "gpu": gpu, "high_mem": opts.high_mem,
        "repo": spec.repo, "ref": spec.ref, "lines": list(spec.lines), "max_hours": opts.max_hours,
        "started": iso(started), "started_epoch": started, "endpoint": None,
        "secrets_sent": sorted(secrets), "log": str(log_path(name)), "exec_log": str(exec_log_path(name)),
    }
    say(ctx, f"job {name}: {len(spec.lines)} line(s) on a Colab {gpu}, {spec.repo}@{spec.ref}; the VM is "
             f"stopped when the job ends, on Ctrl+C, or after {opts.max_hours:g} h")
    state = RunState()
    try:
        _drive(colab, ctx, opts, gpu, spec, secrets, masker, started, manifest, state)
    except KeyboardInterrupt:
        state.outcome, state.forced = "interrupted", True
    except RemoteError as e:
        state.outcome, state.error_code = f"error: {e}", e.code
    except Exception as e:  # a bug, or a failure nobody foresaw: the VM is released all the same
        state.outcome = f"error: {error_text(e, masker)}"
        state.error_code = EXIT_FAILED if state.handles else EXIT_SETUP
    finally:
        released = _clean_up(colab, ctx, name, state)  # before any closing message: printing cannot stop it
        code = final_code(state.status, outcome=state.outcome, released=released, rescue=state.rescue,
                          error_code=state.error_code)
        if state.new_called:
            end = ctx.clock()
            manifest.update(endpoint=state.endpoint, finished=iso(end), wall_seconds=round(end - started, 1),
                            outcome=state.outcome, rc=code, error_code=state.error_code, job=state.status,
                            vm_released=released, rescue_push=state.rescue)
            with _sigint_ignored():
                try:
                    write_manifest(path, manifest)
                except OSError as e:
                    say(ctx, f"could not write {path}: {type(e).__name__}")
    if state.forced:
        say(ctx, "interrupted: the VM was stopped at once, so the outputs since the last successful push "
                 "are lost")
    elif state.error_code is not None:
        say(ctx, state.outcome)
    elif state.status is None and state.rescue is not None and state.rescue.get("push_rc") in PUSHED:
        say(ctx, f"the job never reported its end ({state.outcome}), but the push retry from here pushed "
                 "what was on the VM")
    elif state.status is None and state.handles:
        say(ctx, f"the job never reported its end ({state.outcome}), so the outputs since its last "
                 "successful push may be lost")
    if code == EXIT_NOT_PUSHED:
        lost_alarm(ctx, name, state.status or {}, state.rescue)
    say(ctx, f"exit code {code}; log {log_path(name)}")
    return code


def error_text(error: BaseException, masker: Masker) -> str:
    """An unexpected error for a message: its type and its text, scrubbed, on one line."""
    name, text = type(error).__name__, str(error).strip()
    return f"{name}: {tail(text, masker, lines=2, chars=300)}" if text else name


def _drive(colab: Colab, ctx: Context, opts: Launch, gpu: str, spec: JobSpec, secrets: dict[str, str],
           masker: Masker, started: float, manifest: dict[str, Any], state: RunState) -> None:
    """Everything between the first and the last colab call; _run stops the VM whatever happens here."""
    name = spec.job_id
    path = manifest_path(name)
    before = unknown_endpoints(list_sessions(colab, ctx))
    state.before = frozenset(before or ())
    if before:
        say(ctx, f"note: `colab sessions` lists {len(before)} VM(s) the CLI has no record of, billed until "
                 f"stopped: {', '.join(sorted(before))}")
    say(ctx, "creating the VM (colab new)")
    state.new_called = True
    res = colab.call(*new_args(name, gpu, opts.high_mem))
    state.endpoint = read_endpoint(ctx.config_dir, name)
    if res.rc != 0 or state.endpoint is None:
        state.outcome = "colab new failed"
        say(ctx, f"colab new failed (rc {res.rc}): {tail(res.out, masker)}")
        if state.endpoint is None:
            after = unknown_endpoints(list_sessions(colab, ctx))
            state.orphans = sorted((after or set()) - state.before)
        manifest["orphan_endpoints"] = state.orphans
        write_manifest(path, manifest)
        return
    manifest["endpoint"] = state.endpoint
    write_manifest(path, manifest)
    say(ctx, f"VM ready (endpoint {state.endpoint}); manifest {path}")
    upload_secrets(colab, name, secrets, ctx, masker)
    follower = Follower(colab, name, state.endpoint, ctx=ctx, masker=masker)
    handle = colab.spawn(*exec_args(name, spec, exec_timeout(opts.max_hours)), log_path=exec_log_path(name))
    state.handles.append(handle)
    follower.restart(handle)
    say(ctx, f"job started; its log is fetched to {log_path(name)} every {opts.poll:g} s. Ctrl+C asks the "
             "job to stop and push its outputs, then stops the VM")
    try:
        outcome, state.status = follower.follow(started + opts.max_hours * 3600, opts.poll)
        state.outcome = outcome
    except KeyboardInterrupt:
        state.outcome = "interrupted"
        say(ctx, f"Ctrl+C: asking the job to stop and push (up to {STOP_WAIT / 60:.0f} minutes, longer "
                 "while a push runs). Ctrl+C again stops the VM at once; the outputs since the last "
                 "successful push are then lost")
        outcome, state.status = wind_down(colab, follower, "ctrl-c", ctx, masker, opts.poll)
    except Exception as e:  # a bug, or a failure nobody foresaw: the job still gets to push
        state.outcome = f"error: {error_text(e, masker)}"
        say(ctx, f"following the job failed ({state.outcome[7:]}): asking it to stop and push before the VM "
                 "is stopped")
        outcome, state.status = wind_down(colab, follower, "error", ctx, masker, opts.poll)
    else:
        if outcome == "timeout":  # a Ctrl+C during this wait stops the VM at once
            say(ctx, "--max-hours reached without the job's last line")
            state.outcome = "max-hours"
            outcome, state.status = wind_down(colab, follower, "max-hours", ctx, masker, opts.poll)
    if state.status is not None:
        say(ctx, describe(state.status))
    why = retry_reason(state.status, outcome, follower)
    if why:
        end_exec(handle, finished=state.status is not None)
        state.rescue = rescue_push(colab, follower, spec, secrets, ctx, masker, opts.poll, state.handles,
                                   why=why)
    elif state.status is not None and push_failed(state.status):
        say(ctx, "the job's final push exited 2, a problem no retry can fix: no retry from here")


def _clean_up(colab: Colab, ctx: Context, name: str, state: RunState) -> bool | None:
    """End every exec client, then stop the VM and check it is gone. Never raises: whatever happened
    before, the VM must be released. True also when no VM was ever asked for."""
    with _sigint_ignored():
        for handle in state.handles:
            try:
                end_exec(handle, finished=state.outcome == "done")
            except Exception:  # a client that cannot be ended must not keep the VM
                pass
        if not state.new_called:
            return True
        try:
            state.endpoint = state.endpoint or read_endpoint(ctx.config_dir, name)  # new saved a late record
            return release(colab, name, state.endpoint, ctx, ignore=state.before)
        except Exception as e:
            say(ctx, f"stopping the VM failed: {type(e).__name__}")
            alarm(ctx, name, state.endpoint, checked=False)
            return None


def watch(name: str, *, poll: float = 45.0, max_hours: float | None = None, stop: bool = True,
          ctx: Context | None = None) -> int:
    """``lift remote watch NAME``: re-attach to a job started from this machine (after a sleep or a
    crash), print its log, and stop the VM when the job ends (unless ``stop`` is False), after one
    push retry when its final push failed. Ctrl+C stops the watching only. An unexpected error is
    handled as in launch: a job that may still run is asked to stop and push, the VM is still
    released (unless ``stop`` is False), and 4 and 5 still reach the exit code. ``max_hours`` counts
    from now; by default the launch's cap still holds. A manifest that says the VM is released only
    gets its log printed again, and a recorded job status is never replaced by a weaker outcome."""
    ctx = ctx or Context()
    check_name(name)
    manifest = load_manifest(name)
    if manifest is None:
        raise RemoteError(f"no {manifest_path(name)}: watch needs the manifest of a job started here")
    check_limits(12.0 if max_hours is None else max_hours, poll)
    secrets, missing = gather_secrets(ctx)
    masker = Masker(secrets)
    if manifest.get("vm_released") is True:
        return replay(ctx, name, manifest, masker)
    colab = ready_colab(ctx)
    if max_hours is not None:
        deadline = ctx.clock() + max_hours * 3600
    else:
        start = float(manifest.get("started_epoch", ctx.clock()))
        deadline = start + float(manifest.get("max_hours", 12)) * 3600
    state = RunState(outcome="watching", endpoint=manifest.get("endpoint"), new_called=True)
    say(ctx, f"watching {name} (endpoint {state.endpoint}); Ctrl+C stops the watching, not the VM")
    try:
        _follow_watched(colab, ctx, name, manifest, secrets, missing, masker, deadline, poll, stop, state)
    except KeyboardInterrupt:
        for handle in state.handles:
            end_exec(handle, finished=False)
        say(ctx, f"stopped watching; the VM keeps running (lift remote watch|stop {name})")
        return EXIT_INTERRUPTED
    except Exception as e:  # a bug, or a failure nobody foresaw: the VM is released all the same
        state.outcome, state.error_code = f"error: {error_text(e, masker)}", EXIT_FAILED
    return _end_watch(colab, ctx, name, manifest, stop, state)


def _follow_watched(colab: Colab, ctx: Context, name: str, manifest: dict[str, Any], secrets: dict[str, str],
                    missing: list[str], masker: Masker, deadline: float, poll: float, stop: bool,
                    state: RunState) -> None:
    """watch's part between its first and its last colab call, kept in ``state`` as it goes: watch
    stops the VM whatever happens here. An unexpected error while following asks the job, which may
    still run, to stop and push first, as launch does; with ``stop`` False it ends the watching."""
    follower = Follower(colab, name, state.endpoint, ctx=ctx, masker=masker)
    try:
        ended, state.status = follower.follow(deadline, poll)
    except Exception as e:
        state.outcome = f"error: {error_text(e, masker)}"
        if not stop:
            raise
        say(ctx, f"following the job failed ({state.outcome[7:]}): asking it to stop and push before the VM "
                 "is stopped")
        ended, state.status = wind_down(colab, follower, "error", ctx, masker, poll)
    else:
        state.outcome = ended
        if ended == "timeout":
            state.outcome = "max-hours"
            if stop:
                ended, state.status = wind_down(colab, follower, "max-hours", ctx, masker, poll)
    recorded = manifest.get("job")
    if state.status is None and isinstance(recorded, dict):
        say(ctx, f"no last line now ({state.outcome}); keeping the job status recorded before")
        state.outcome, state.status = manifest.get("outcome") or "done", recorded
    if state.status is not None:
        say(ctx, describe(state.status))
    retried = manifest.get("rescue_push") is not None  # an earlier watch or launch tried it already
    why = retry_reason(state.status, ended, follower) if stop and not retried else None
    if why and missing:
        say(ctx, f"the push cannot be retried from here: {missing_message(missing)}")
    elif why:
        spec = JobSpec(manifest.get("repo") or DEFAULT_REPO, manifest.get("ref") or DEFAULT_REF, (), name)
        state.rescue = rescue_push(colab, follower, spec, secrets, ctx, masker, poll, state.handles, why=why)
    elif state.status is not None and stop and push_failed(state.status) and not rescuable(state.status):
        say(ctx, "the job's final push exited 2, a problem no retry can fix: no retry from here")


def _end_watch(colab: Colab, ctx: Context, name: str, manifest: dict[str, Any], stop: bool,
               state: RunState) -> int:
    """Stop the VM (unless ``stop`` is False), then record the run and return its code. A recorded job
    status outlives an error that cut the watching short, and so does a recorded error code."""
    recorded = manifest.get("job")
    if state.status is None and isinstance(recorded, dict):
        state.status = recorded
    if state.rescue is not None:
        manifest["rescue_push"] = state.rescue
    error_code = state.error_code if state.error_code is not None else manifest.get("error_code")
    manifest.update(outcome=state.outcome, job=state.status, error_code=error_code, watched=iso(ctx.clock()))
    released: bool | None = True  # for the code: --no-stop keeps the VM on purpose
    if stop:
        released = _clean_up(colab, ctx, name, state)  # before any closing message: printing cannot stop it
        manifest.update(vm_released=released, finished=iso(ctx.clock()),
                        endpoint=manifest.get("endpoint") or state.endpoint)
        if released is True and isinstance(manifest.get("started_epoch"), int | float):
            manifest["wall_seconds"] = round(ctx.clock() - manifest["started_epoch"], 1)  # released here
    else:
        say(ctx, f"--no-stop: the VM keeps running; lift remote stop {name} releases it")
    code = final_code(state.status, outcome=state.outcome, released=released,
                      rescue=manifest.get("rescue_push"), error_code=error_code)
    kept = not stop and code == EXIT_NOT_PUSHED  # not lost: they are on the VM it leaves running
    if kept:
        code = EXIT_FAILED
    manifest["rc"] = code
    with _sigint_ignored():
        try:
            write_manifest(manifest_path(name), manifest)
        except OSError as e:
            say(ctx, f"could not write {manifest_path(name)}: {type(e).__name__}")
    if state.error_code is not None:
        say(ctx, state.outcome)
    elif state.status is None and stop:
        pushed = state.rescue is not None and state.rescue.get("push_rc") in PUSHED
        say(ctx, f"the job never reported its end ({state.outcome}), " + (
            "but the push retry from here pushed what was on the VM" if pushed
            else "so the outputs since its last successful push may be lost"))
    if kept:
        kept_on_vm(ctx, name, state.status or {}, manifest.get("rescue_push"))
    elif code == EXIT_NOT_PUSHED:
        lost_alarm(ctx, name, state.status or {}, manifest.get("rescue_push"))
    return code


def retry_reason(status: dict[str, Any] | None, outcome: str, follower: Follower) -> str | None:
    """Why one more push from here is worth trying, or None: the job's final push failed (not with an
    exit 2), or the job ended with no last line (its cell gone, or its log dead) after running lines."""
    if status is not None:
        return "the job's final push failed" if rescuable(status) else None
    if outcome in ("no-marker", "stale") and follower.lines_ran():
        return f"the job never reported its end ({outcome}) after running lines"
    return None


def replay(ctx: Context, name: str, manifest: dict[str, Any], masker: Masker) -> int:
    """watch on a released VM: print the log as last downloaded, touch nothing, return the recorded
    exit code."""
    say(ctx, f"the VM of {name} was released already ({manifest.get('finished') or 'time unknown'}); "
             "this is its log as last downloaded")
    path = log_path(name)
    if path.is_file():
        LogPrinter(ctx.out, masker).feed(path.read_bytes(), final=True)
    else:
        say(ctx, f"no local copy of the log at {path}")
    if isinstance(manifest.get("job"), dict):
        say(ctx, describe(manifest["job"]))
    rc = manifest.get("rc")
    return rc if isinstance(rc, int) and not isinstance(rc, bool) else EXIT_FAILED


def stop(name: str, *, endpoint: str | None = None, ctx: Context | None = None) -> int:
    """``lift remote stop NAME [--endpoint E]``: stop the VM and check it is gone, by its endpoint (given,
    or recorded in the manifest) when the CLI has lost its record. Without an endpoint, a VM the CLI
    has no record of may be this one, so it is never called released. Returns 0 when released, else 4.
    Once the release is confirmed, the recorded code becomes the run's own: a 4 is answered, and a job
    whose final push failed (its outputs still on the VM after watch --no-stop) gets 5 and the alarm
    that its outputs are lost."""
    ctx = ctx or Context()
    check_name(name)
    if endpoint is not None and not ENDPOINT_RE.fullmatch(endpoint):
        raise RemoteError(f"bad --endpoint {endpoint!r}: copy it from a line of `colab sessions`")
    manifest = load_manifest(name)
    endpoint = endpoint or (manifest.get("endpoint") if manifest else None)
    if endpoint is None:
        say(ctx, f"no endpoint for {name} (no manifest, or colab new never reported one): stopping by name "
                 "only")
    colab = ready_colab(ctx)
    with _sigint_ignored():
        released = release(colab, name, endpoint, ctx)
    if manifest is not None:
        manifest.update(vm_released=released, stopped=iso(ctx.clock()))
        job = manifest.get("job")
        if released is True and (manifest.get("rc") == EXIT_NOT_RELEASED or isinstance(job, dict)):
            manifest["rc"] = final_code(job, outcome=manifest.get("outcome") or "done", released=True,
                                        rescue=manifest.get("rescue_push"),
                                        error_code=manifest.get("error_code"))
        if endpoint and not manifest.get("endpoint"):
            manifest["endpoint"] = endpoint
        write_manifest(manifest_path(name), manifest)
        if released is True and manifest.get("rc") == EXIT_NOT_PUSHED:
            lost_alarm(ctx, name, job if isinstance(job, dict) else {}, manifest.get("rescue_push"))
    return EXIT_OK if released else EXIT_NOT_RELEASED
