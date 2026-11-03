# colab/job.py: one lift job, run INSIDE a Colab kernel. `lift remote` (src/lift/remote.py) sends it
# with `colab exec -f colab/job.py --env LIFT_JOB_B64=<spec>`; colab/README.md explains the flow.
"""Run a lift job as one Colab kernel cell and report through files, never through an exit status.

The kernel context sets the rules. The file never reads sys.argv (the kernel's own) and never calls
sys.exit, and it has no `from __future__` import (the CLI prepends code to the cell). `colab exec`
exits 0 whatever the code does, so the result goes to /content/lift_job.status.json and to a final
"@@LIFT_DONE {json}" line in /content/lift_job.log, which `lift remote` polls with `colab download`.
It is ASCII only (`colab exec -f` reads it in the laptop's locale encoding) and stdlib only, and every
path and subprocess call can be replaced, so the tests run it on fakes.

Order: take the secrets file that `lift remote` uploaded and delete it first thing; refuse a second
run on the same VM; check Python >= 3.12; fetch repo@ref with the GitHub token only in an http
extraheader given through `git -c`; install ".[gpu,data,eval]" with the kernel's own python and
uninstall torchao; run each line as `python -m lift ...`, stopping at the first failure, with a
`lift sync push` checkpoint after every line except sync lines and the last one; then, in `finally`,
run the final `lift sync push` and write the status and the marker. A push that exits 1 (an error) or
hangs is tried again, 4 tries in about 10 minutes; one that exits 3 (done, the conflicting copies kept
under conflicts/ in the runs repo) is not. Everything the commands print goes to the log with every
secret value replaced by "***".

A monitor thread looks every few seconds for /content/lift_job.stop (uploaded by `lift remote` on
Ctrl+C or at --max-hours) and for the spec's deadline: either stops the running command (not a push)
and skips the remaining lines, so the final push still runs. It also kills a command silent for the
spec's max_silent_minutes and a push running past PUSH_LIMIT_SECONDS, which a stalled network read
would otherwise hold forever. Every few minutes it logs a heartbeat that says how long the command
has been silent, to the kernel's output and to the log, so `lift remote` can tell a quiet job from a
dead one. In push mode (spec "mode": "push") the file instead runs one more `lift sync push` for a
job whose final push failed, and ends with its own @@LIFT_DONE line tagged with the attempt id.
"""

import base64
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = "/content"
SECRETS_FILE = ".lift_secrets.json"
STARTED_FILE = "lift_job.started"
STOP_FILE = "lift_job.stop"
LOG_FILE = "lift_job.log"
STATUS_FILE = "lift_job.status.json"
REPO_DIR = "lift"
DONE_MARKER = "@@LIFT_DONE "
MIN_PYTHON = (3, 12)
HEARTBEAT_SECONDS = 300
TICK_SECONDS = 5  # how often the monitor looks for the stop file, the deadline and a stuck command
MAX_SILENT_MINUTES = 45  # the spec's default: a command silent this long is hung
PUSH_LIMIT_SECONDS = 1800  # huggingface_hub sets no HTTP timeout, so a stalled push would never end
PUSH_DELAYS = (60, 180, 360)  # waits between the 4 tries of a push: about 10 minutes in all
KILL_GRACE_SECONDS = 30  # SIGINT first, so the command can close its files; SIGKILL after this
PUSH_RETRY_RC = 1  # lift sync push: an error, and trying again loses nothing (push compares content)
PUSH_CONFLICT_RC = 3  # lift sync push: done, the conflicting local copies kept under conflicts/
PUSHED = (0, PUSH_CONFLICT_RC)
SETUP_FAILED_RC = 3  # lift remote's exit code for "the job could not be set up on the VM"
STOPPED_RC = 1  # a stop request before a line started: the job did not finish
MIN_SECRET_LEN = 8  # shorter values (a spend cap such as "20") are settings, not secrets: not masked
# Used only when there is no secrets file (job.py started by hand in a notebook).
ENV_FALLBACK = ("HF_TOKEN", "GITHUB_TOKEN", "LIFT_RUNS_REPO", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "LIFT_JUDGE_MAX_USD")
# git prints request headers, the token header included, when any of these is set.
GIT_TRACE_VARS = ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_TRACE_PACKET", "GIT_CURL_VERBOSE", "GIT_TRACE2",
                  "GIT_TRACE2_EVENT", "GIT_TRACE2_PERF")
SPEC_DEFAULTS = {"deadline": None, "max_silent_minutes": MAX_SILENT_MINUTES, "mode": "job", "attempt": None}
_KILL = getattr(signal, "SIGKILL", None)  # None on Windows, where only the test fakes run

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REF = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./-]*$")
_EXTRAS = re.compile(r"^[a-z0-9_-]+(,[a-z0-9_-]+)*$")
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decode_spec(b64):
    """LIFT_JOB_B64 -> {"repo", "ref", "lines", "job_id", "extras", "deadline", "max_silent_minutes",
    "mode", "attempt"}, the last four filled with their defaults. Raises ValueError saying why not."""
    if not b64:
        raise ValueError("LIFT_JOB_B64 is not set")
    try:
        spec = json.loads(base64.b64decode(b64, validate=True).decode("utf-8"))
    except (ValueError, TypeError) as e:  # binascii.Error and UnicodeDecodeError are ValueErrors
        raise ValueError(f"LIFT_JOB_B64 is not base64 JSON ({type(e).__name__})") from None
    if not isinstance(spec, dict):
        raise ValueError("the job spec is not a JSON object")
    missing = [key for key in ("repo", "ref", "lines", "job_id", "extras") if key not in spec]
    if missing:
        raise ValueError(f"the job spec lacks {missing}")
    spec = {**SPEC_DEFAULTS, **spec}
    for key, pattern in (("repo", _REPO), ("ref", _REF), ("job_id", _JOB_ID), ("extras", _EXTRAS)):
        value = spec[key]
        if not isinstance(value, str) or not pattern.match(value) or (key == "ref" and ".." in value):
            raise ValueError(f"bad {key} in the job spec: {value!r}")
    lines = spec["lines"]
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise ValueError("the job spec's lines are not a list of strings")
    for i, line in enumerate(lines):
        words = shlex.split(line)
        if len(words) < 2 or words[0] != "lift":
            raise ValueError(f"job line {i + 1} is not a lift command: {line!r}")
    if spec["deadline"] is not None and not _number(spec["deadline"]):
        raise ValueError(f"bad deadline in the job spec: {spec['deadline']!r}")
    if not _number(spec["max_silent_minutes"]) or spec["max_silent_minutes"] <= 0:
        raise ValueError(f"bad max_silent_minutes in the job spec: {spec['max_silent_minutes']!r}")
    if spec["mode"] not in ("job", "push"):
        raise ValueError(f"bad mode in the job spec: {spec['mode']!r}")
    attempt = spec["attempt"]
    if spec["mode"] == "push" and not (isinstance(attempt, str) and _JOB_ID.match(attempt)):
        raise ValueError(f"bad attempt in the job spec: {attempt!r}")
    return spec


def plan_lines(lines):
    """Drop trailing plain `lift sync push` lines: the push in `finally` runs anyway."""
    lines = list(lines)
    while lines and shlex.split(lines[-1]) == ["lift", "sync", "push"]:
        lines.pop()
    return lines


def basic_auth(token):
    """The credential git sends for GitHub: base64 of "x-access-token:<token>"."""
    return base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")


def mask_values(secrets):
    """Byte strings to hide in the log, longest first: each secret value of MIN_SECRET_LEN+ characters
    and the git credential built from GITHUB_TOKEN."""
    values = [value for value in secrets.values() if len(value) >= MIN_SECRET_LEN]
    if secrets.get("GITHUB_TOKEN"):
        values.append(basic_auth(secrets["GITHUB_TOKEN"]))
    return sorted({value.encode("utf-8") for value in values}, key=len, reverse=True)


def mask(data, secrets):
    for secret in secrets:
        data = data.replace(secret, b"***")
    return data


class Masker:
    """Copy a byte stream to `write` with every secret replaced by "***".

    Text is released up to the last line break (a secret never contains one). A run of more than
    `hold` bytes without a break is released except for its last len(longest secret) - 1 bytes, which
    could be the start of a secret that is still arriving.
    """

    def __init__(self, secrets, write, hold=1 << 16):
        self.secrets = list(secrets)
        self.keep = max([len(secret) for secret in self.secrets] + [1]) - 1
        self.write = write
        self.hold = hold
        self.buf = b""

    def feed(self, chunk):
        self.buf += chunk
        cut = max(self.buf.rfind(b"\n"), self.buf.rfind(b"\r")) + 1
        if cut:
            self.write(mask(self.buf[:cut], self.secrets))
            self.buf = self.buf[cut:]
        if len(self.buf) > self.hold:
            masked = mask(self.buf, self.secrets)
            cut = max(0, len(masked) - self.keep)
            self.write(masked[:cut])
            self.buf = masked[cut:]

    def close(self):
        if self.buf:
            self.write(mask(self.buf, self.secrets))
            self.buf = b""


class JobFailed(Exception):
    """A setup step failed. `step` goes to the status; `reason` to the log (it never holds a secret).
    The default rc is lift remote's "could not set up" code, never its "refused to start" (2)."""

    def __init__(self, step, reason, rc=SETUP_FAILED_RC):
        super().__init__(reason)
        self.step = step
        self.reason = reason
        self.rc = rc


class Running:
    """The command _stream is running, as the monitor thread sees it."""

    def __init__(self, proc, kind, now):
        self.proc = proc
        self.kind = kind  # "setup", "line" or "push"
        self.started = now
        self.last_output = now
        self.killed = None  # why the monitor stopped it
        self.killed_at = None


def _new_group():
    """Popen kwargs that give a command its own process group, so the whole group can be signalled."""
    return {"start_new_session": True} if os.name == "posix" else {}


def _signal(proc, sig):
    """Send sig to the command's process group (its children may hold the log pipe open), or to the
    command alone where there are no groups (and to test fakes, which have no pid)."""
    try:
        pid = getattr(proc, "pid", None)
        if os.name == "posix" and isinstance(pid, int):
            os.killpg(pid, sig)
        elif sig is _KILL:
            proc.kill()
        else:
            proc.send_signal(sig)
    except (OSError, ValueError):  # it ended meanwhile
        pass


class Job:
    """One run of the job. Every argument replaces a piece of the outside world for tests."""

    def __init__(self, root=ROOT, environ=None, popen=None, python=None, version_info=None,
                 heartbeat=HEARTBEAT_SECONDS, out=None, clock=None, sleep=None, push_limit=None):
        self.root = root
        self.environ = dict(os.environ if environ is None else environ)
        self.popen = popen or subprocess.Popen
        self.python = python or sys.executable
        self.version_info = tuple(sys.version_info[:3] if version_info is None else version_info)
        self.heartbeat = heartbeat
        self.out = out or sys.stdout
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self.push_limit = PUSH_LIMIT_SECONDS if push_limit is None else push_limit
        self.max_silent = MAX_SILENT_MINUTES * 60
        self.deadline = None
        self.repo = os.path.join(root, REPO_DIR)
        self.secrets = {}
        self.masks = []
        self.lines = []
        self.step = "starting"
        self.installed = False
        self.stoppable = True  # push mode ignores the stop file and the deadline: pushing is all it does
        self.stop_reason = None
        self.killed = None  # why the monitor stopped the last command, if it did
        self._running = None
        self._log = None
        self._last = b"\n"
        self._lock = threading.Lock()  # the monitor thread writes to the log too
        self._stop_lock = threading.Lock()

    def path(self, name):
        return os.path.join(self.root, name)

    def run(self):
        """Run the job once, or in push mode one more push. Returns the status dict, or None when this
        VM already ran a job."""
        self.secrets, problem, source = self._take_secrets()
        self.masks = mask_values(self.secrets)
        try:
            spec, spec_error = decode_spec(self.environ.get("LIFT_JOB_B64", "")), None
        except ValueError as e:
            spec, spec_error = None, str(e)
        if spec is not None:
            self.deadline, self.max_silent = spec["deadline"], spec["max_silent_minutes"] * 60
            if spec["mode"] == "push":
                return self._push_only(spec, problem)
        if not self._claim():
            self.say(f"refused: {self.path(STARTED_FILE)} exists, so a job already ran on this VM")
            return None
        t0 = self.clock()
        status = {"mode": "job", "job_id": None, "rc": None, "failed_line": None, "failed_step": None,
                  "push_rc": None, "installed": False, "stopped": None, "killed": None,
                  "checkpoint_push_failures": 0, "push_conflicts": 0, "commit": None, "gpu": None,
                  "python": ".".join(str(part) for part in self.version_info), "started": _iso(t0),
                  "seconds": {"setup": None, "lines": [], "push": None}}
        self._log = open(self.path(LOG_FILE), "ab")
        stop_monitor = self._start_monitor(t0)
        self.say(f"job started; log {self.path(LOG_FILE)}")
        try:
            self._setup(status, spec, spec_error, problem, source, t0)
            self._run_lines(status)
        except JobFailed as e:
            status["rc"], status["failed_step"], status["killed"] = e.rc, e.step, self.killed
            if self.killed and self.killed == self.stop_reason:
                status["stopped"] = self.stop_reason
            self.note(f"FAILED at {e.step}: {e.reason}")
        except BaseException as e:  # an interrupt or a bug in this file: still push and report
            status["rc"], status["failed_step"] = 1, f"job.py: {type(e).__name__}"
            self.note("job.py stopped on an exception:\n" + traceback.format_exc())
        finally:
            try:
                self._final_push(status)
            finally:
                stop_monitor.set()
                self._finish(status, t0)
        return status

    # -- steps --------------------------------------------------------------------------------------

    def _take_secrets(self):
        """Read the secrets file and delete it before anything else. Returns (values, problem, source)."""
        path = self.path(SECRETS_FILE)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            values = {key: self.environ[key] for key in ENV_FALLBACK if self.environ.get(key)}
            return values, None, "the kernel environment (no secrets file)"
        except OSError as e:
            return {}, f"cannot read the secrets file ({type(e).__name__})", "file"
        finally:
            _remove(path)
        try:
            values = json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}, "the secrets file is not UTF-8 JSON", "file"
        if not isinstance(values, dict) or not all(
                isinstance(key, str) and _NAME.match(key) and isinstance(value, str)
                for key, value in values.items()):
            return {}, "the secrets file does not map variable names to strings", "file"
        return values, None, "the uploaded file, now deleted"

    def _claim(self):
        """Create the started marker, or return False if it exists: a retried or queued exec must
        not run the job twice."""
        try:
            fd = os.open(self.path(STARTED_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, (_iso(self.clock()) + "\n").encode("ascii"))
        os.close(fd)
        return True

    def _setup(self, status, spec, spec_error, problem, source, t0):
        self.step = "setup"
        if spec_error:
            raise JobFailed("spec", spec_error)
        status["job_id"] = spec["job_id"]
        self.lines = plan_lines(spec["lines"])
        self.note(f"job {spec['job_id']}: {len(self.lines)} lift line(s) from {spec['repo']}@{spec['ref']}, "
                  f"Python {status['python']}")
        if self.deadline is not None:
            self.note(f"the job stops itself at {_iso(self.deadline)}, so its final push fits before "
                      "--max-hours")
        if problem:
            raise JobFailed("secrets", problem)
        self.note(f"secrets: {', '.join(sorted(self.secrets)) or 'none'}, from {source}")
        if self.version_info < MIN_PYTHON:
            raise JobFailed("python", f"Python {status['python']} is too old: SafeTune needs 3.12 or newer")
        status["gpu"] = self._gpu()
        status["commit"] = self._fetch(spec["repo"], spec["ref"])
        self._install(spec["extras"])
        status["seconds"]["setup"] = round(self.clock() - t0, 1)

    def _gpu(self):
        """The GPU this VM really has, for the record (not fatal: a CPU runtime has none)."""
        sink = []
        rc = self._stream(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          env=self._env(secrets=False), sink=sink)
        text = b"".join(sink).decode("utf-8", "replace").strip()
        return text if rc == 0 and text else None

    def _fetch(self, repo, ref):
        """Fetch repo@ref (branch, tag or full sha) into /content/lift. The token travels only as an
        http extraheader given with `git -c`: never in a URL or .git/config, and argv is never logged."""
        self.step = "fetch"
        if os.path.isdir(self.repo) and os.listdir(self.repo):
            raise JobFailed("fetch", f"{self.repo} already exists and is not empty")
        env = self._env(secrets=False)
        token = self.secrets.get("GITHUB_TOKEN")
        header = "http.https://github.com/.extraheader=AUTHORIZATION: basic "
        auth = ["-c", header + basic_auth(token)] if token else []
        self.note(f"fetching {repo}@{ref}")
        steps = (("git init", ["git", "init", "-q", self.repo]),
                 ("git remote add", ["git", "-C", self.repo, "remote", "add", "origin",
                                     f"https://github.com/{repo}.git"]),
                 ("git fetch", ["git", *auth, "-C", self.repo, "fetch", "-q", "--depth", "1", "origin", ref]),
                 ("git checkout", ["git", "-C", self.repo, "checkout", "-q", "--detach", "FETCH_HEAD"]))
        for what, argv in steps:
            rc = self._stream(argv, env=env)
            if rc != 0 or self.killed:
                raise JobFailed("fetch", f"{what} failed (rc {rc}){self._why()}", rc or 1)
        sink = []
        self._stream(["git", "-C", self.repo, "rev-parse", "HEAD"], env=env, sink=sink)
        return b"".join(sink).decode("ascii", "replace").strip() or None

    def _install(self, extras):
        self.step = "install"
        env = self._env(secrets=False)
        target = f"{self.repo}[{extras}]"
        self.note(f"pip install -e {target}")
        rc = self._stream([self.python, "-m", "pip", "install", "-q", "-e", target], cwd=self.repo, env=env)
        if rc != 0 or self.killed:
            raise JobFailed("install", f"pip install failed (rc {rc}){self._why()}", rc or 1)
        self.installed = True
        # torchao on some images breaks SafeTune/TRL imports; pip exits 0 when it is not installed.
        rc = self._stream([self.python, "-m", "pip", "uninstall", "-y", "-q", "torchao"], env=env)
        if rc != 0:
            self.note(f"pip uninstall torchao returned rc {rc}; continuing")

    def _run_lines(self, status):
        n = len(self.lines)
        env = self._env(secrets=True)
        for i, line in enumerate(self.lines):
            if self._stopping():
                status["rc"], status["stopped"] = STOPPED_RC, self.stop_reason
                self.note(f"{self.stop_reason}: lines {i + 1}-{n} are skipped")
                return
            words = shlex.split(line)
            self.step = f"line {i + 1}/{n}"
            self.note(f"$ {line}   (line {i + 1}/{n})")
            t = self.clock()
            rc = self._stream(self._lift(words[1:]), cwd=self.repo, env=env, kind="line")
            status["seconds"]["lines"].append(round(self.clock() - t, 1))
            if rc != 0 or self.killed:
                status["rc"], status["failed_line"], status["killed"] = rc or 1, i, self.killed
                if self.killed and self.killed == self.stop_reason:
                    status["stopped"] = self.stop_reason
                self.note(f"line {i + 1}/{n} failed (rc {rc}){self._why()}; the remaining lines are skipped")
                return
            self.note(f"line {i + 1}/{n} ok in {_hms(self.clock() - t)}")
            if i < n - 1 and words[1] != "sync":
                self.step = f"checkpoint push after line {i + 1}/{n}"
                if self._push("checkpoint", status) not in PUSHED:
                    status["checkpoint_push_failures"] += 1
                    self.note("checkpoint push failed; continuing, the final push sends everything")
        status["rc"] = 0

    def _push(self, what, status=None, tries=None):
        """`lift sync push`, tried again after PUSH_DELAYS when it exits 1 (an error; push compares
        content, so trying again loses nothing) or was killed for running too long. A 3 (done, the
        conflicting copies kept under conflicts/ in the runs repo) is final. Returns the last rc."""
        delays = PUSH_DELAYS[:len(PUSH_DELAYS) if tries is None else tries - 1]
        env = self._env(secrets=True)
        rc = None
        for attempt, delay in enumerate((*delays, None), 1):
            self.note(f"lift sync push ({what}, try {attempt}/{len(delays) + 1})")
            rc = self._stream(self._lift(["sync", "push"]), cwd=self.repo, env=env, kind="push")
            if rc == 0 and not self.killed:
                return 0
            if rc == PUSH_CONFLICT_RC:
                if status is not None:
                    status["push_conflicts"] += 1
                self.note("the push hit conflicts: the VM's copies are under conflicts/ in the runs repo")
                return rc
            retry = rc == PUSH_RETRY_RC or self.killed is not None
            if not retry or delay is None or (what == "checkpoint" and self._stopping()):
                return rc or 1
            self.note(f"the push failed (rc {rc}){self._why()}; trying again in {delay} s")
            self.sleep(delay)
        return rc

    def _final_push(self, status):
        """`lift sync push` whatever happened before, once lift is installed. From then on anything but
        a push that exits 0 or 3, an interrupt included, is a failed push and fails the job: the
        outputs since the last good push exist only on this VM."""
        status["installed"] = self.installed
        if not self.installed:
            self.note("no final push: setup did not finish, so lift is not installed and nothing ran")
        else:
            self.step = "final push"
            t = self.clock()
            try:
                status["push_rc"] = self._push("final", status)
            except BaseException as e:  # a kernel interrupt mid-push: recorded, never taken for a success
                self.note(f"the final push stopped on {type(e).__name__}")
            status["seconds"]["push"] = round(self.clock() - t, 1)
            if status["push_rc"] not in PUSHED:
                self.note(f"FINAL PUSH FAILED (rc {status['push_rc']}): outputs since the last good push "
                          "exist only on this VM")
        if status["rc"] is None:
            status["rc"] = 1
        if self.installed and status["rc"] == 0 and status["push_rc"] not in PUSHED:
            status["rc"] = status["push_rc"] or 1

    def _push_only(self, spec, problem):
        """Push mode, sent by lift remote when the job's final push failed: read (and delete) the
        secrets it uploaded again, run `lift sync push` once in /content/lift, and append an
        @@LIFT_DONE line carrying the mode and the attempt id, so it is never taken for the job's."""
        self.stoppable = False
        t0 = self.clock()
        status = {"mode": "push", "attempt": spec["attempt"], "job_id": spec["job_id"], "push_rc": None,
                  "error": None, "started": _iso(t0), "seconds": {}}
        self._log = open(self.path(LOG_FILE), "ab")
        stop_monitor = self._start_monitor(t0)
        self.step = "push retry"
        self.note(f"push retry {spec['attempt']} for job {spec['job_id']}, sent by lift remote")
        try:
            if problem:
                status["error"] = f"secrets: {problem}"
                self.note(f"no push: {problem}")
            else:
                status["push_rc"] = self._push("retry", tries=1)
        except BaseException as e:
            status["error"] = f"job.py: {type(e).__name__}"
            self.note("the push retry stopped on an exception:\n" + traceback.format_exc())
        finally:
            stop_monitor.set()
            self._finish(status, t0, status_file=False)
        return status

    def _finish(self, status, t0, status_file=True):
        end = self.clock()
        status["finished"] = _iso(end)
        status["seconds"]["total"] = round(end - t0, 1)
        text = json.dumps(status, sort_keys=True)
        if status_file:
            try:
                tmp = self.path(STATUS_FILE + ".tmp")
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text + "\n")
                os.replace(tmp, self.path(STATUS_FILE))
            except OSError as e:
                self.note(f"could not write {STATUS_FILE}: {type(e).__name__}")
        with self._lock:
            log, self._log = self._log, None
            log.write((b"" if self._last == b"\n" else b"\n") + (DONE_MARKER + text + "\n").encode("utf-8"))
            log.close()

    # -- the monitor --------------------------------------------------------------------------------

    def _start_monitor(self, t0):
        """Start the monitor thread (see the module docstring). Returns the event that ends it."""
        done = threading.Event()
        tick = min(TICK_SECONDS, self.heartbeat) if self.heartbeat else TICK_SECONDS

        def watch():
            next_beat = t0 + self.heartbeat if self.heartbeat else None
            while not done.wait(tick):
                now = self.clock()
                try:
                    self._check(now)
                    if next_beat is not None and now >= next_beat:
                        next_beat = now + self.heartbeat
                        self._beat(now, t0)
                except Exception as e:  # one bad tick must not end the monitor
                    self.note(f"monitor: {type(e).__name__}")

        threading.Thread(target=watch, name="lift-job-monitor", daemon=True).start()
        return done

    def _check(self, now):
        """One tick: notice a stop request, SIGINT a command that must not go on, and SIGKILL one that
        ignored its SIGINT for KILL_GRACE_SECONDS."""
        self._stopping()
        with self._lock:
            run = self._running
        if run is None or run.proc.poll() is not None:
            return
        if run.killed is None:
            why = self._why_stop(run, now)
            if why:
                run.killed, run.killed_at = why, now
                self.note(f"stopping the {run.kind} command: {why}")
                _signal(run.proc, signal.SIGINT)
        elif now - run.killed_at >= KILL_GRACE_SECONDS:
            _signal(run.proc, _KILL)

    def _why_stop(self, run, now):
        """Why the running command must stop now, or None. A push is never stopped by a stop request
        or the deadline: pushing is what both are for."""
        if run.kind == "push":
            if now - run.started > self.push_limit:
                return f"the push has run for {_hms(now - run.started)}"
            return None
        if self.stop_reason:
            return self.stop_reason
        if now - run.last_output > self.max_silent:
            return f"no output for {_hms(now - run.last_output)}"
        return None

    def _stopping(self):
        """The reason to stop the job (the stop file, or the spec's deadline), or None. Checked by the
        monitor and before each line."""
        with self._stop_lock:
            if self.stop_reason is not None or not self.stoppable:
                return self.stop_reason
            try:
                with open(self.path(STOP_FILE), "rb") as f:
                    text = f.read(64).decode("ascii", "replace").lower()
                reason = "stop file: " + (re.sub(r"[^a-z0-9 -]", "", text).strip() or "no reason given")
            except OSError:
                deadline = self.deadline is not None and self.clock() >= self.deadline
                reason = "deadline reached" if deadline else None
            if reason is None:
                return None
            self.stop_reason = reason
        self.note(f"{reason}: stopping the job; its final push still runs")
        return reason

    def _beat(self, now, t0):
        with self._lock:
            run = self._running
        if run is None:
            detail = ""
        elif run.kind == "push":
            detail = f", the push has run for {_hms(now - run.started)}"
        else:
            detail = f", {_hms(now - run.last_output)} since its last output"
        message = f"alive {_hms(now - t0)}: {self.step}{detail}"
        self.say(message)
        self.note(message)  # a log that stops growing tells lift remote the job died

    # -- plumbing -----------------------------------------------------------------------------------

    def _why(self):
        return f"; killed: {self.killed}" if self.killed else ""

    def _lift(self, args):
        return [self.python, "-m", "lift", *args]

    def _env(self, secrets):
        """The env for a child process. Secrets go to lift only, never to git or pip (pip runs
        third-party build code); trace variables that would print the token header are dropped."""
        env = dict(self.environ)
        for key in ("LIFT_JOB_B64", *GIT_TRACE_VARS, *self.secrets):
            env.pop(key, None)
        env.update({"PYTHONUNBUFFERED": "1", "GIT_TERMINAL_PROMPT": "0", "MPLBACKEND": "Agg"})
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TQDM_MININTERVAL", "30")
        if secrets:
            env.update(self.secrets)
        return env

    def _stream(self, argv, cwd=None, env=None, sink=None, kind="setup"):
        """Run argv to the end in its own process group, copying its stdout and stderr into the log,
        masked. Returns the exit code (128 + N when killed by signal N); self.killed says why the
        monitor stopped it, if it did. argv itself is never logged."""

        def write(data):
            self._write(data)
            if sink is not None:
                sink.append(data)

        self.killed = None
        try:
            proc = self.popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, **_new_group())
        except OSError as e:
            self.note(f"could not start {os.path.basename(argv[0])}: {type(e).__name__}")
            return 127
        run = Running(proc, kind, self.clock())
        with self._lock:
            self._running = run
        masker = Masker(self.masks, write)
        read = getattr(proc.stdout, "read1", proc.stdout.read)
        try:
            while True:
                chunk = read(1 << 16)
                if not chunk:
                    break
                run.last_output = self.clock()
                masker.feed(chunk)
            masker.close()
            proc.stdout.close()
            rc = proc.wait()
        except BaseException:
            _signal(proc, _KILL)
            proc.wait()
            raise
        finally:
            with self._lock:
                self._running = None
            self.killed = run.killed
        return rc if rc >= 0 else 128 - rc

    def _write(self, data):
        with self._lock:
            if data and self._log is not None:
                self._log.write(data)
                self._log.flush()
                self._last = data[-1:]

    def note(self, message):
        """One line in the job log, masked."""
        line = f"[lift-job] {time.strftime('%H:%M:%S')} {message}\n"
        self._write(mask(line.encode("utf-8", "replace"), self.masks))

    def say(self, message):
        """One line on the kernel's stdout, which `colab exec` relays to the laptop: job.py's own
        messages only, never the commands' output."""
        print(f"[lift-job] {time.strftime('%H:%M:%S')} {message}", file=self.out, flush=True)


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _hms(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main(**kwargs):
    """Kernel entry point: run the job and print a one-line summary. Never raises into the cell."""
    job = Job(**kwargs)
    try:
        status = job.run()
    except BaseException:
        trace = mask(traceback.format_exc().encode("utf-8", "replace"), job.masks)
        job.say("job.py crashed:\n" + trace.decode("utf-8", "replace"))
        return None
    if status is not None and status["mode"] == "push":
        job.say(f"push retry finished: push rc {status['push_rc']}")
    elif status is not None:
        job.say(f"finished: rc {status['rc']}, final push rc {status['push_rc']}")
    return status


if __name__ == "__main__":
    main()
