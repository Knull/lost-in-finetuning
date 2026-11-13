"""``lift remote`` and colab/job.py, on fakes: no colab.exe, no network, no real credential.

job.py runs with a temp dir standing in for /content and a fake Popen. lift remote runs against a fake
Colab CLI (its sessions.json, the server's VM list, uploads, a scripted job log) and a fake clock.
Every "secret" below is a made-up string.
"""

from __future__ import annotations

import ast
import base64
import errno
import importlib.util
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from lift import remote
from lift.cli import build_parser, main
from lift.paths import REPO_ROOT

JOB_PATH = REPO_ROOT / "colab" / "job.py"
README_PATH = REPO_ROOT / "colab" / "README.md"
PY = "/usr/bin/python3"
EXE = "C:/fake bin/colab.exe"
GH = "ghp_fake0000github0000token0000"
HF = "hf_fake0000hugging0000face00000"
OAI = "sk-fake0000openai0000key0000000"
RUNS = "someone/lift-runs-private"
SECRETS = {"GITHUB_TOKEN": GH, "HF_TOKEN": HF, "LIFT_RUNS_REPO": RUNS, "OPENAI_API_KEY": OAI}
BASIC = base64.b64encode(f"x-access-token:{GH}".encode()).decode()
SHA = "0123456789abcdef0123456789abcdef01234567"
STOP_FILE = "content/lift_job.stop"
REAL_RUN = remote.SubprocessRunner.run
REAL_SPAWN = remote.SubprocessRunner.spawn


@pytest.fixture(autouse=True)
def no_real_world(monkeypatch, tmp_path):
    """No test may reach colab.exe, the HF token file or the real user profile."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test reached the real Colab CLI")

    monkeypatch.setattr(remote.SubprocessRunner, "run", refuse)
    monkeypatch.setattr(remote.SubprocessRunner, "spawn", refuse)
    monkeypatch.setattr(remote, "cached_hf_token", lambda: None)
    home = tmp_path / "home"
    home.mkdir()
    for var in ("USERPROFILE", "HOME"):
        monkeypatch.setenv(var, str(home))
    for var in (*SECRETS, "ANTHROPIC_API_KEY", "LIFT_JUDGE_MAX_USD"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def runs_repo_check(monkeypatch):
    """lift remote checks the runs repo with lift.sync.preflight(write=True) before `colab new`. No test
    reaches the Hub, so it is stubbed: it records each call's ``write`` and raises what a test puts in
    ``errors``."""
    from lift import sync

    calls: list[bool] = []
    errors: list[Exception] = []

    def preflight(*, write=True):
        calls.append(write)
        if errors:
            raise errors.pop(0)

    monkeypatch.setattr(sync, "preflight", preflight, raising=False)
    return types.SimpleNamespace(calls=calls, errors=errors)


def sync_error(message: str, *, permanent: bool) -> Exception:
    from lift import sync

    error = sync.SyncError(message)
    error.permanent = permanent
    return error


@pytest.fixture(scope="module")
def job():
    spec = importlib.util.spec_from_file_location("lift_colab_job", JOB_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- colab/job.py ---------------------------------------------------------------------------------


class FakeProc:
    def __init__(self, output: bytes, rc: int):
        self.stdout = io.BytesIO(output)
        self.rc = rc

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        return self.rc

    def send_signal(self, sig):
        pass

    def kill(self):
        pass


class HangingProc:
    """A command that prints ``output`` and then nothing until it is signalled. After 10 s it gives up
    and ends with rc 0, so a missing fix fails the test instead of hanging the suite."""

    def __init__(self, output: bytes = b""):
        self.chunks = [output] if output else []
        self.signals: list = []
        self.ended = threading.Event()
        self.stdout = self

    def read(self, size=-1):
        if self.chunks:
            return self.chunks.pop(0)
        self.ended.wait(10)
        return b""

    def close(self):
        pass

    def poll(self):
        return self.wait() if self.ended.is_set() else None

    def wait(self, timeout=None):
        self.ended.wait(10)
        return -int(signal.SIGINT) if self.signals else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self.ended.set()

    def kill(self):
        self.signals.append("kill")
        self.ended.set()


class WaitingProc(FakeProc):
    """A command that prints ``output`` only once ``ready()`` holds (5 s at most), then ends with rc 0."""

    def __init__(self, output: bytes, ready):
        super().__init__(b"", 0)
        self.data = [output]
        self.ready = ready
        self.stdout = self

    def read(self, size=-1):
        deadline = time.monotonic() + 5
        while not self.ready() and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.data.pop(0) if self.data else b""

    def close(self):
        pass


def command_key(argv: list[str]) -> str:
    if argv[0] == "git":
        return "git " + next(a for a in argv[1:] if a in {"init", "remote", "fetch", "checkout", "rev-parse"})
    if argv[0] == "nvidia-smi":
        return "nvidia-smi"
    assert argv[:2] == [PY, "-m"], argv
    if argv[2:] == ["lift", "sync", "--help"]:
        return "lift-check"  # job.py's setup check, kept apart from the job's own lift commands
    return f"{argv[2]} {' '.join(argv[3:])}" if argv[2] == "lift" else f"pip {argv[3]}"


class FakeVM:
    """subprocess.Popen as job.py sees it on the VM: records each call and answers by command. An rc
    given as a list is used up call by call (its last value repeats); ``procs`` hands out special
    processes, one per call, before the plain ones. The checkout brings the ref's constraints.txt
    unless ``constraints`` is False (a ref older than it)."""

    def __init__(self, root: Path, *, rcs=None, outputs=None, hook=None, procs=None, constraints=True):
        self.root = root
        self.rcs = {key: (list(rc) if isinstance(rc, list) else rc) for key, rc in (rcs or {}).items()}
        self.outputs = {"git rev-parse": f"{SHA}\n".encode(),
                        "nvidia-smi": b"NVIDIA A100-SXM4-40GB, 40960 MiB\n", **(outputs or {})}
        self.hook = hook
        self.procs = {key: list(queue) for key, queue in (procs or {}).items()}
        self.constraints = constraints
        self.calls: list[dict] = []
        self.secrets_file_at_first_call = None

    def __call__(self, argv, cwd=None, env=None, stdin=None, stdout=None, stderr=None, **kwargs):
        if not self.calls:
            self.secrets_file_at_first_call = (self.root / ".lift_secrets.json").exists()
        assert stdin is subprocess.DEVNULL and stdout is subprocess.PIPE and stderr is subprocess.STDOUT
        key = command_key(argv)
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env), "key": key, "kwargs": kwargs})
        if key == "git checkout" and self.constraints:
            (self.root / "lift").mkdir(exist_ok=True)
            (self.root / "lift" / "constraints.txt").write_text("transformers==5.22.1\n", encoding="utf-8")
        if self.hook:
            self.hook(key)
        if self.procs.get(key):
            return self.procs[key].pop(0)
        rc = self.rcs.get(key, 0)
        if isinstance(rc, list):
            rc = rc.pop(0) if len(rc) > 1 else rc[0]
        return FakeProc(self.outputs.get(key, b"ok\n"), rc)

    def keys(self, prefix: str = "") -> list[str]:
        return [c["key"] for c in self.calls if c["key"].startswith(prefix)]


def job_env(lines, spec_extra=None) -> dict:
    """The kernel environment `lift remote` gives job.py: the job spec, base64 JSON."""
    spec = {**remote.JobSpec("Knull/lost-in-finetuning", "main", tuple(lines), "job-1").to_dict(),
            **(spec_extra or {})}
    return {"LIFT_JOB_B64": base64.b64encode(json.dumps(spec).encode()).decode(), "PATH": "/usr/bin"}


def run_job(job, root: Path, lines, *, vm=None, secrets=SECRETS, environ=None, version=(3, 12, 4),
            out=None, heartbeat=0, spec_extra=None, **job_kwargs):
    root.mkdir(exist_ok=True)
    if secrets is not None:
        (root / ".lift_secrets.json").write_text(json.dumps(secrets), encoding="utf-8")
    vm = vm or FakeVM(root)
    out = out or io.StringIO()
    env = {**job_env(lines, spec_extra), **(environ or {})}
    status = job.main(root=str(root), environ=env, popen=vm, python=PY, version_info=version,
                      heartbeat=heartbeat, out=out, **job_kwargs)
    return status, vm, out.getvalue()


def log_bytes(root: Path) -> bytes:
    return (root / "lift_job.log").read_bytes()


def last_marker(root: Path) -> dict:
    line = log_bytes(root).splitlines()[-1]
    assert line.startswith(b"@@LIFT_DONE ")
    return json.loads(line[len(b"@@LIFT_DONE "):])


def test_job_file_is_ascii_and_safe_as_a_kernel_cell():
    data = JOB_PATH.read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "__future__"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "sys":
            assert node.attr not in {"argv", "exit"}
    # The CLI prepends its own lines to the cell; the file must still compile after them.
    compile("import sys, warnings\nsys.argv = ['x']\n__name__ = '__main__'\n" + text, "<cell>", "exec")


def test_job_spec_round_trips(job):
    lines = remote.parse_job('lift sync pull; lift judge --keys "a b;c"\nlift info')
    spec = remote.JobSpec("Knull/lost-in-finetuning", "p02-harness", tuple(lines), "lift-20261104-0915-ab12",
                          deadline=1_800_041_400.0, max_silent_minutes=30.0)
    decoded = job.decode_spec(spec.encode())
    assert decoded == spec.to_dict()
    assert decoded["deadline"] == 1_800_041_400.0 and decoded["mode"] == "job" and decoded["attempt"] is None
    assert [shlex.split(line) for line in decoded["lines"]] == [
        ["lift", "sync", "pull"], ["lift", "judge", "--keys", "a b;c"], ["lift", "info"]]
    push = remote.JobSpec("a/b", "main", (), "lift-x", mode="push", attempt="ab12cd34")
    assert job.decode_spec(push.encode())["attempt"] == "ab12cd34"


@pytest.mark.parametrize("extra", [{"deadline": "soon"}, {"max_silent_minutes": 0}, {"mode": "shell"},
                                   {"mode": "push"}, {"mode": "push", "attempt": "../x"}])
def test_job_spec_refuses_bad_optional_fields(job, extra):
    spec = {**remote.JobSpec("a/b", "main", ("lift info",), "j").to_dict(), **extra}
    with pytest.raises(ValueError):
        job.decode_spec(base64.b64encode(json.dumps(spec).encode()).decode())


def test_job_runs_lines_in_order_with_checkpoints_and_one_final_push(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    vm = FakeVM(root, outputs={"lift smoke": f"loaded with {HF} and {GH} and {BASIC}\n".encode()})
    status, vm, out = run_job(job, root, ["lift sync pull", "lift smoke", "lift run-dev --variant stock",
                                          "lift sync push"], vm=vm)

    assert status["rc"] == 0 and status["push_rc"] == 0 and status["installed"] is True
    assert status["failed_line"] is None and status["failed_step"] is None and status["stopped"] is None
    assert status["checkpoint_push_failures"] == 0 and status["push_conflicts"] == 0
    assert status["commit"] == SHA and status["gpu"].startswith("NVIDIA A100")
    # The secrets file was gone before the first subprocess, and the started marker is there.
    assert vm.secrets_file_at_first_call is False
    assert not (root / ".lift_secrets.json").exists() and (root / "lift_job.started").exists()
    assert vm.keys()[:9] == ["nvidia-smi", "git init", "git remote", "git fetch", "git checkout",
                             "git rev-parse", "git fetch", "pip install", "pip uninstall"]
    # A checkpoint push after each line except sync lines and the last; the trailing push is the final one.
    assert vm.keys("lift ") == ["lift sync pull", "lift smoke", "lift sync push",
                                "lift run-dev --variant stock", "lift sync push"]
    repo = os.path.join(str(root), "lift")
    pip = next(c for c in vm.calls if c["key"] == "pip install")
    constraints = os.path.join(repo, "constraints.txt")
    assert pip["argv"] == [PY, "-m", "pip", "install", "-q", "-c", constraints, "-e",
                           repo + "[gpu,data,eval]"]
    assert next(c for c in vm.calls if c["key"] == "pip uninstall")["argv"][-1] == "torchao"
    assert all(c["cwd"] == repo for c in vm.calls if c["key"].startswith("lift "))

    # The token reaches git only as an extraheader, and no raw secret is in any argv.
    fetch, tags = [c for c in vm.calls if c["key"] == "git fetch"]
    assert fetch["argv"][1:3] == ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {BASIC}"]
    assert fetch["argv"][-2:] == ["origin", "main"]
    # Then the freeze tags the --depth 1 fetch leaves out (lift publish reads the recipe a tag froze).
    assert tags["argv"][1:3] == fetch["argv"][1:3]
    refspec = "+refs/tags/recipe-frozen-*:refs/tags/recipe-frozen-*"
    assert tags["argv"][-3:] == ["--no-tags", "origin", refspec]
    for call in vm.calls:
        assert not any(secret in arg for arg in call["argv"] for secret in SECRETS.values())
    # Secrets reach lift only: never git or pip, which also get no terminal prompt.
    for call in vm.calls:
        lift = call["key"].startswith("lift ")
        assert (call["env"].get("HF_TOKEN") == HF) is lift
        assert (call["env"].get("GITHUB_TOKEN") == GH) is lift
        assert call["env"]["GIT_TERMINAL_PROMPT"] == "0" and "LIFT_JOB_B64" not in call["env"]

    log = log_bytes(root)
    for secret in (*SECRETS.values(), BASIC):
        assert secret.encode() not in log
    assert b"loaded with *** and *** and ***" in log
    on_disk = json.loads((root / "lift_job.status.json").read_text(encoding="utf-8"))
    assert on_disk == status
    assert log.splitlines()[-1] == b"@@LIFT_DONE " + json.dumps(status, sort_keys=True).encode()
    # The kernel's own output is a summary, never the log.
    assert "loaded with" not in out and "finished: rc 0" in out
    assert not any(secret in out for secret in SECRETS.values())


def test_job_stops_at_the_first_failure_and_still_pushes_once(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift smoke", "lift info"],
                            vm=FakeVM(root, rcs={"lift smoke": 3}))
    assert status["rc"] == 3 and status["failed_line"] == 1 and status["push_rc"] == 0
    assert vm.keys("lift ") == ["lift sync pull", "lift smoke", "lift sync push"]
    assert len(status["seconds"]["lines"]) == 2
    assert b"@@LIFT_DONE " in log_bytes(root)


def test_job_pushes_once_when_the_last_line_is_a_push(job, tmp_path):
    root = tmp_path / "content"
    status, vm, _ = run_job(job, root, ["lift info", "lift sync push"])
    assert vm.keys("lift ") == ["lift info", "lift sync push"]
    assert status["rc"] == 0


def test_job_fails_when_the_final_push_fails(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    # rc 5 is not lift sync push's "error, try again" code (1), so it is not tried again
    status, vm, _ = run_job(job, root, ["lift info"], vm=FakeVM(root, rcs={"lift sync push": 5}))
    assert vm.keys("lift ") == ["lift info", "lift sync push"]
    assert status["rc"] == 5 and status["push_rc"] == 5 and status["failed_line"] is None
    assert b"FINAL PUSH FAILED" in log_bytes(root)


# Review #10: the final push was tried once, so one transient Hub error lost the job's outputs.
def test_job_retries_a_push_that_ends_1_but_never_one_that_conflicts(job, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "PUSH_DELAYS", (0.001, 0.001, 0.001), raising=False)
    # The final push of the canonical dev eval fails once (a Hub 5xx), then works: the outputs get there.
    root = tmp_path / "flaky"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift run-dev --variant stock"],
                            vm=FakeVM(root, rcs={"lift sync push": [1, 0]}))
    assert vm.keys("lift ") == ["lift sync pull", "lift run-dev --variant stock", "lift sync push",
                                "lift sync push"]
    assert status["rc"] == 0 and status["push_rc"] == 0
    # Checkpoint pushes are tried again too.
    root = tmp_path / "checkpoint"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift smoke", "lift info"],
                            vm=FakeVM(root, rcs={"lift sync push": [1, 0]}))
    assert vm.keys("lift ") == ["lift smoke", "lift sync push", "lift sync push", "lift info",
                                "lift sync push"]
    assert status["checkpoint_push_failures"] == 0 and status["rc"] == 0
    # A push that keeps failing: four tries, then the job fails and says the outputs are only on the VM.
    root = tmp_path / "down"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift info"], vm=FakeVM(root, rcs={"lift sync push": [1]}))
    assert vm.keys("lift sync push") == ["lift sync push"] * 4
    assert status["rc"] == 1 and status["push_rc"] == 1
    assert b"FINAL PUSH FAILED" in log_bytes(root)
    # Exit 3 (done; the conflicting copies went under conflicts/ in the runs repo) is final and not a failure.
    root = tmp_path / "conflict"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift smoke", "lift info"],
                            vm=FakeVM(root, rcs={"lift sync push": [3]}))
    assert vm.keys("lift ") == ["lift smoke", "lift sync push", "lift info", "lift sync push"]
    assert status["rc"] == 0 and status["push_rc"] == 3 and status["push_conflicts"] == 2
    assert status["checkpoint_push_failures"] == 0


def test_push_backoff_is_four_tries_in_about_ten_minutes(job):
    assert len(job.PUSH_DELAYS) == 3 and 540 <= sum(job.PUSH_DELAYS) <= 660


# Review #30: an exception during the final push left rc 0 on the @@LIFT_DONE line.
def test_an_interrupted_final_push_is_a_failed_push(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()

    def interrupt_the_push(key):
        if key == "lift sync push":
            raise KeyboardInterrupt  # e.g. a kernel interrupt from a browser attached with `colab url`

    status, vm, out = run_job(job, root, ["lift run-dev --variant stock"],
                              vm=FakeVM(root, hook=interrupt_the_push))
    marker = last_marker(root)
    assert marker == status and marker["installed"] is True
    assert marker["push_rc"] is None and marker["rc"] == 1
    assert b"FINAL PUSH FAILED" in log_bytes(root) and "finished: rc 1" in out


def test_job_setup_failure_runs_no_line_and_no_push(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift info"], vm=FakeVM(root, rcs={"git fetch": 128}))
    assert status["rc"] == 128 and status["failed_step"] == "fetch" and status["push_rc"] is None
    assert status["installed"] is False
    assert vm.keys("pip") == [] and vm.keys("lift ") == []
    log = log_bytes(root)
    assert b"git fetch failed (rc 128)" in log and GH.encode() not in log and BASIC.encode() not in log
    assert last_marker(root) == status


def test_a_repo_without_freeze_tags_still_runs_its_lines(job, tmp_path):
    """The tags fetch fails when no recipe-frozen-* tag exists yet (a glob that matches nothing): noted,
    not fatal, since only a line that needs a tag can say whether it matters."""
    root = tmp_path / "content"
    root.mkdir()
    status, vm, _ = run_job(job, root, ["lift info"], vm=FakeVM(root, rcs={"git fetch": [0, 1]}))
    assert status["rc"] == 0 and status["failed_step"] is None and status["installed"] is True
    assert vm.keys("git fetch") == ["git fetch", "git fetch"]
    assert vm.keys("lift ") == ["lift info", "lift sync push"]  # the line, then the final push
    assert b"no recipe-frozen-* tag fetched (rc 1)" in log_bytes(root)


# Review #62: JobFailed's default code was 2, the code that means "lift remote refused to start".
def test_job_refuses_an_old_python_before_any_subprocess(job, tmp_path):
    root = tmp_path / "content"
    status, vm, _ = run_job(job, root, ["lift info"], version=(3, 11, 9))
    assert status["failed_step"] == "python" and status["rc"] == 3
    assert vm.calls == [] and not (root / ".lift_secrets.json").exists()
    assert b"@@LIFT_DONE " in log_bytes(root)


def test_job_refuses_a_non_lift_line(job, tmp_path):
    root = tmp_path / "content"
    spec = {"repo": "a/b", "ref": "main", "lines": ["rm -rf /"], "job_id": "j", "extras": "gpu"}
    bad = base64.b64encode(json.dumps(spec).encode()).decode()
    status, vm, _ = run_job(job, root, [], environ={"LIFT_JOB_B64": bad})
    assert status["failed_step"] == "spec" and status["rc"] == 3 and vm.calls == []


def test_job_refuses_a_second_run_and_still_deletes_the_secrets(job, tmp_path):
    root = tmp_path / "content"
    first, _, _ = run_job(job, root, ["lift info"])
    assert first["rc"] == 0
    second, vm, out = run_job(job, root, ["lift info"])
    assert second is None and vm.calls == [] and "refused" in out
    assert not (root / ".lift_secrets.json").exists()
    assert log_bytes(root).count(b"@@LIFT_DONE ") == 1


def test_job_push_mode_runs_one_push_for_lift_remote(job, tmp_path, monkeypatch):
    """lift remote's retry of a failed final push: job.py in push mode, after the job's own run."""
    monkeypatch.setattr(job, "PUSH_DELAYS", (0.001, 0.001, 0.001), raising=False)
    root = tmp_path / "content"
    root.mkdir()
    first, _, _ = run_job(job, root, ["lift info"], vm=FakeVM(root, rcs={"lift sync push": [1]}))
    assert first["push_rc"] == 1
    job_status = (root / "lift_job.status.json").read_bytes()

    second, vm, out = run_job(job, root, [], spec_extra={"mode": "push", "attempt": "ab12cd34"})
    assert vm.keys() == ["lift sync push"] and vm.secrets_file_at_first_call is False
    assert vm.calls[0]["env"]["HF_TOKEN"] == HF and vm.calls[0]["cwd"] == os.path.join(str(root), "lift")
    assert second["mode"] == "push" and second["attempt"] == "ab12cd34" and second["push_rc"] == 0
    assert last_marker(root) == second and "push rc 0" in out
    # The job's own status stays as it was, and lift remote still finds the job's line by default.
    assert (root / "lift_job.status.json").read_bytes() == job_status
    log = log_bytes(root)
    assert remote.find_status(log) == first
    assert remote.find_status(log, accept=lambda s: s.get("attempt") == "ab12cd34") == second


# Review #33: Ctrl+C and --max-hours tore the VM down with no final push.
def test_job_stops_the_running_line_on_the_stop_file_then_pushes(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    hang = HangingProc(b"loading\n")

    def ask_to_stop(key):
        if key == "lift run-dev --variant stock":  # whole: the monitor polls, and must never read it empty
            (root / "lift_job.stop.tmp").write_text("ctrl-c\n", encoding="ascii")
            os.replace(root / "lift_job.stop.tmp", root / "lift_job.stop")

    vm = FakeVM(root, hook=ask_to_stop, procs={"lift run-dev --variant stock": [hang]})
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift run-dev --variant stock", "lift info"], vm=vm,
                            heartbeat=0.01)
    assert hang.signals == [signal.SIGINT]
    assert vm.keys("lift ") == ["lift sync pull", "lift run-dev --variant stock", "lift sync push"]
    assert status["stopped"] == "stop file: ctrl-c" and status["failed_line"] == 1 and status["rc"] == 130
    assert status["push_rc"] == 0 and last_marker(root) == status


def test_job_stops_itself_at_its_deadline_then_pushes(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    hang = HangingProc(b"generating\n")
    late = []  # the VM's clock jumps two hours once the line starts: past the deadline, mid-line

    def clock():
        return time.time() + (7200 if late else 0)

    vm = FakeVM(root, procs={"lift run-dev --variant stock": [hang]},
                hook=lambda key: late.append(key) if key.startswith("lift run-dev") else None)
    status, vm, _ = run_job(job, root, ["lift run-dev --variant stock", "lift info"], vm=vm, heartbeat=0.01,
                            spec_extra={"deadline": time.time() + 3600}, clock=clock)
    assert hang.signals == [signal.SIGINT]
    assert vm.keys("lift ") == ["lift run-dev --variant stock", "lift sync push"]
    assert status["stopped"] == "deadline reached" and status["failed_line"] == 0 and status["push_rc"] == 0


# Review #34: a hung line or a stalled push was kept "alive" by the heartbeats until --max-hours.
def test_job_kills_a_line_that_stays_silent_too_long(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    hang = HangingProc(b"loading\n")
    vm = FakeVM(root, procs={"lift run-dev --variant stock": [hang]})
    status, vm, _ = run_job(job, root, ["lift run-dev --variant stock", "lift info"], vm=vm, heartbeat=0.01,
                            spec_extra={"max_silent_minutes": 0.005})
    assert hang.signals == [signal.SIGINT]
    assert status["failed_line"] == 0 and status["rc"] == 130 and status["stopped"] is None
    assert status["killed"].startswith("no output for")
    assert vm.keys("lift ") == ["lift run-dev --variant stock", "lift sync push"] and status["push_rc"] == 0
    assert b"no output for" in log_bytes(root)


def test_job_kills_a_hung_push_and_tries_again(job, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "PUSH_LIMIT_SECONDS", 0.3, raising=False)
    monkeypatch.setattr(job, "PUSH_DELAYS", (0.001, 0.001, 0.001), raising=False)
    root = tmp_path / "content"
    root.mkdir()
    hang = HangingProc()  # a stalled HTTPS read: huggingface_hub sets no timeout
    vm = FakeVM(root, procs={"lift sync push": [hang]})
    status, vm, _ = run_job(job, root, ["lift info"], vm=vm, heartbeat=0.01)
    assert hang.signals == [signal.SIGINT]
    assert vm.keys("lift ") == ["lift info", "lift sync push", "lift sync push"]
    assert status["push_rc"] == 0 and status["rc"] == 0


class LeaderGoneProc:
    """A line whose lift process (the group leader) printed ``output`` and exited with ``rc``, while a
    child it started (same process group, same stdout) stays alive and silent with the pipe open. poll()
    and wait() give the leader's rc; read() gives the output, then blocks until the group is signalled
    (which ends the child and closes the pipe). After 10 s it gives up (``timed_out``), so a missing fix
    fails the test instead of hanging the suite."""

    def __init__(self, output: bytes = b"generating\n", rc: int = 0):
        self.chunks = [output]
        self.rc = rc
        self.signals: list = []
        self.released = threading.Event()
        self.timed_out = False
        self.stdout = self

    def read(self, size=-1):
        if self.chunks:
            return self.chunks.pop(0)
        if not self.released.wait(10):
            self.timed_out = True
        return b""

    def close(self):
        pass

    def poll(self):
        return self.rc  # the leader is gone

    def wait(self, timeout=None):
        return self.rc

    def send_signal(self, sig):  # job.py signals the group (os.killpg) on a VM, the fake here
        self.signals.append(sig)
        self.released.set()

    def kill(self):
        self.signals.append("kill")
        self.released.set()


class SigintSurvivorProc(LeaderGoneProc):
    """A line still running and silent. SIGINT ends its leader (poll() then gives -SIGINT) but not its
    child, which ignores SIGINT and holds the pipe open until SIGKILL (kill())."""

    def __init__(self):
        super().__init__()
        self.leader_rc = None

    def poll(self):
        return self.leader_rc

    def wait(self, timeout=None):
        return 0 if self.leader_rc is None else self.leader_rc

    def send_signal(self, sig):
        self.signals.append(sig)
        if sig == signal.SIGINT:
            self.leader_rc = -int(signal.SIGINT)  # the leader dies of it; the child goes on, pipe open


# R1-30: the monitor stopped policing a command once its leader exited (poll() not None), while _stream
# went on reading the pipe a child still held: no silence limit, stop request or SIGKILL reached the group,
# so the line, the next lines and the final push waited on the child (on a VM, until --max-hours).
def test_job_kills_a_silent_child_holding_the_pipe_after_its_leader_exited(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    proc = LeaderGoneProc()
    vm = FakeVM(root, procs={"lift run-dev --variant stock": [proc]})
    status, vm, _ = run_job(job, root, ["lift run-dev --variant stock", "lift info"], vm=vm, heartbeat=0.01,
                            spec_extra={"max_silent_minutes": 0.005})  # silent 0.3 s: hung
    assert proc.signals == [signal.SIGINT] and not proc.timed_out
    assert status["failed_line"] == 0 and status["rc"] == 1 and status["killed"].startswith("no output for")
    assert vm.keys("lift ") == ["lift run-dev --variant stock", "lift sync push"] and status["push_rc"] == 0


def test_job_stops_a_line_whose_leader_exited_but_whose_child_holds_the_pipe(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    proc = LeaderGoneProc()

    def ask_to_stop(key):  # lift remote uploads the stop file while the line runs
        if key == "lift run-dev --variant stock":  # whole: the monitor polls, and must never read it empty
            (root / "lift_job.stop.tmp").write_text("ctrl-c\n", encoding="ascii")
            os.replace(root / "lift_job.stop.tmp", root / "lift_job.stop")

    vm = FakeVM(root, hook=ask_to_stop, procs={"lift run-dev --variant stock": [proc]})
    status, vm, _ = run_job(job, root, ["lift run-dev --variant stock", "lift info"], vm=vm, heartbeat=0.01)
    assert proc.signals == [signal.SIGINT] and not proc.timed_out
    assert status["stopped"] == "stop file: ctrl-c" and status["failed_line"] == 0
    assert vm.keys("lift ") == ["lift run-dev --variant stock", "lift sync push"] and status["push_rc"] == 0


def test_job_sigkills_a_child_that_outlives_the_sigint_of_its_leader(job, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "KILL_GRACE_SECONDS", 0.2, raising=False)
    root = tmp_path / "content"
    root.mkdir()
    proc = SigintSurvivorProc()
    vm = FakeVM(root, procs={"lift run-dev --variant stock": [proc]})
    status, vm, _ = run_job(job, root, ["lift run-dev --variant stock", "lift info"], vm=vm, heartbeat=0.01,
                            spec_extra={"max_silent_minutes": 0.005})
    # SIGINT while the leader ran, then SIGKILL after the grace, though the leader had died of the SIGINT.
    assert proc.signals[0] == signal.SIGINT and set(proc.signals[1:]) == {"kill"} and not proc.timed_out
    assert status["failed_line"] == 0 and status["rc"] == 130 and status["killed"].startswith("no output for")
    assert vm.keys("lift ") == ["lift run-dev --variant stock", "lift sync push"] and status["push_rc"] == 0


def test_job_heartbeat_reaches_the_kernel_and_the_log_but_the_log_stays_off_the_kernel(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    out = io.StringIO()
    slow = WaitingProc(b"verbose lift output\n", lambda: "since its last output" in out.getvalue())
    vm = FakeVM(root, procs={"lift info": [slow]})
    status, _, text = run_job(job, root, ["lift info"], vm=vm, out=out, heartbeat=0.02)
    assert status["rc"] == 0 and "alive" in text and "line 1/1" in text
    log = log_bytes(root)
    assert b"alive" in log and b"verbose lift output" in log and "verbose lift output" not in text
    assert b"since its last output" in log  # the heartbeat tells a silent command from a busy one
    assert log.splitlines()[-1].startswith(b"@@LIFT_DONE ")


def test_masker_catches_secrets_split_across_chunks(job):
    written = []
    masker = job.Masker([b"SECRETVALUE123"], written.append, hold=8)
    for chunk in (b"abc SECRET", b"VAL", b"UE123 def", b" and SECRETVALUE123", b"\n", b"xxxxxxxxxxSECRE",
                  b"TVALUE123\n"):
        masker.feed(chunk)
    masker.close()
    assert b"".join(written) == b"abc *** def and ***\nxxxxxxxxxx***\n"
    assert all(b"SECRE" not in piece for piece in written)


# Review 2, R1: a ref without `lift sync` (main before P02 merged) ran its lines and reported the
# outputs of a job that never ran as lost.
def test_job_refuses_a_ref_without_lift_sync_before_any_line(job, tmp_path):
    root = tmp_path / "content"
    usage = b"usage: lift [-h] {info} ...\nlift: error: argument command: invalid choice: 'sync'\n"
    vm = FakeVM(root, rcs={"lift-check": 2}, outputs={"lift-check": usage})
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift smoke"], vm=vm)
    assert status["failed_step"] == "sync-check" and status["rc"] == 3 and status["installed"] is False
    assert vm.keys("lift ") == [] and status["push_rc"] is None
    log = log_bytes(root)
    assert b"lift sync" in log and b"P01 and P02" in log and b"invalid choice" in log
    assert remote.exit_code(status) == remote.EXIT_SETUP
    # The check runs right after the install, before any line.
    root = tmp_path / "ok"
    status, vm, _ = run_job(job, root, ["lift info"])
    keys = vm.keys()
    assert keys.index("lift-check") == keys.index("pip uninstall") + 1 < keys.index("lift info")
    assert status["rc"] == 0 and status["installed"] is True


# Contract 7: the GPU runtimes install with the ref's constraints.txt. Review 3, V14: a ref pushed
# without the (then untracked) file was blamed on a missing P01/P02 first.
def test_job_refuses_a_ref_without_constraints(job, tmp_path):
    root = tmp_path / "content"
    status, vm, _ = run_job(job, root, ["lift info"], vm=FakeVM(root, constraints=False))
    assert status["failed_step"] == "install" and status["rc"] == 3 and status["installed"] is False
    assert vm.keys("pip") == [] and vm.keys("lift ") == []
    log = log_bytes(root)
    assert b"commit constraints.txt to the ref" in log and b"P01 and P02" in log
    assert log.index(b"commit constraints.txt") < log.index(b"P01 and P02")


# Contract 1 (review 2, R25/R34): lift sync exits 2 for a problem no retry can fix; job.py tried it again
# like a transient error, idling the GPU for 10 minutes per push.
def test_job_never_retries_a_sync_exit_2_and_ends_the_job(job, tmp_path):
    root = tmp_path / "content"
    naps = []
    vm = FakeVM(root, rcs={"lift sync push": [2]})
    status, vm, _ = run_job(job, root, ["lift smoke", "lift info"], vm=vm, sleep=naps.append)
    # The checkpoint push after line 1 exits 2: no second try, no next line; the final push is tried once.
    assert vm.keys("lift ") == ["lift smoke", "lift sync push", "lift sync push"] and naps == []
    assert status["sync_refused"] == "the checkpoint push after line 1/2"
    assert status["rc"] == 2 and status["push_rc"] == 2 and status["failed_line"] is None
    assert b"no retry can fix" in log_bytes(root)
    # The smoke outputs exist only on the VM, and lift remote must not retry that push either.
    assert remote.push_failed(status) and remote.exit_code(status) == remote.EXIT_NOT_PUSHED


def test_a_pull_line_is_tried_again_on_1_and_a_2_ends_the_job_at_once(job, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "PUSH_DELAYS", (0.001, 0.001, 0.001), raising=False)
    root = tmp_path / "flaky"
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift smoke"],
                            vm=FakeVM(root, rcs={"lift sync pull": [1, 0]}))
    assert vm.keys("lift ") == ["lift sync pull", "lift sync pull", "lift smoke", "lift sync push"]
    assert status["rc"] == 0 and status["failed_line"] is None
    root = tmp_path / "refused"
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift smoke"],
                            vm=FakeVM(root, rcs={"lift sync pull": [2]}))
    # Not tried again, no next line, and no final push: only lift sync ran, so nothing new is on the VM.
    assert vm.keys("lift ") == ["lift sync pull"]
    assert status["rc"] == 2 and status["failed_line"] == 0 and status["ran"] == 0
    assert status["sync_refused"] == "line 1/2 (lift sync pull)" and status["push_rc"] is None
    # Not "outputs lost": the runs repo refused before anything ran.
    assert not remote.push_failed(status) and remote.exit_code(status) == remote.EXIT_SETUP


# Review 3, V11: a pull killed for silence (a stalled Hub read) was tried 3 more times: about 3 hours of
# an idle A100 at the defaults before the job failed, instead of one --max-silent-minutes.
def test_a_pull_killed_as_hung_ends_the_job_without_another_try(job, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "PUSH_DELAYS", (0.001, 0.001, 0.001), raising=False)
    root = tmp_path / "content"
    hangs = [HangingProc(b"pulling\n"), HangingProc(b"pulling\n")]
    vm = FakeVM(root, procs={"lift sync pull": hangs})
    status, vm, _ = run_job(job, root, ["lift sync pull", "lift run-dev --variant stock"], vm=vm,
                            heartbeat=0.01, spec_extra={"max_silent_minutes": 0.005})
    assert vm.keys("lift ") == ["lift sync pull"] and hangs[0].signals == [signal.SIGINT]
    assert status["failed_line"] == 0 and status["killed"].startswith("no output for")
    assert status["rc"] == 130 and status["ran"] == 0 and status["push_rc"] is None
    assert b"trying again" not in log_bytes(root)


# Review 2, R16: a stop request during the wait before another checkpoint try did not cut the wait (up
# to 360 s), and another checkpoint try ran before the final push.
def test_a_stop_request_cuts_a_checkpoint_retry_wait_and_ends_its_tries(job, tmp_path):
    root = tmp_path / "content"
    clock = FakeClock(time.time())
    naps = []

    def sleep(seconds):
        naps.append(seconds)
        clock.sleep(seconds)
        (root / "lift_job.stop").write_text("ctrl-c\n", encoding="ascii")  # Ctrl+C on the laptop now

    vm = FakeVM(root, rcs={"lift sync push": [1, 0]})
    status, vm, _ = run_job(job, root, ["lift smoke", "lift info"], vm=vm, sleep=sleep, clock=clock)
    assert naps == [job.TICK_SECONDS]  # the stop request is seen at the next look, not after 60 s
    assert vm.keys("lift ") == ["lift smoke", "lift sync push", "lift sync push"]  # no 2nd checkpoint try
    assert status["stopped"] == "stop file: ctrl-c" and status["checkpoint_push_failures"] == 1
    assert status["push_rc"] == 0


def test_a_stop_request_never_cuts_the_final_push_wait(job, tmp_path):
    root = tmp_path / "content"
    naps = []
    def hook(key):  # the stop request comes while the only line runs
        if key == "lift info":
            (root / "lift_job.stop").write_text("ctrl-c\n", encoding="ascii")

    vm = FakeVM(root, rcs={"lift sync push": [1, 0]}, hook=hook)
    status, vm, _ = run_job(job, root, ["lift info"], vm=vm, sleep=naps.append)
    assert naps == [job.PUSH_DELAYS[0]] and vm.keys("lift sync push") == ["lift sync push"] * 2
    assert status["push_rc"] == 0


class FullDiskLog:
    """The job log once the VM's disk is full: every write fails with ENOSPC."""

    def write(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    def flush(self):
        raise OSError(errno.ENOSPC, "No space left on device")

    def close(self):
        pass


# Review 2, R19: a failing log write escaped every handler, so the final push never ran and no last
# line was written.
def test_a_full_disk_still_gets_the_final_push_and_the_last_line_out(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    (root / ".lift_secrets.json").write_text(json.dumps(SECRETS), encoding="utf-8")
    vm = FakeVM(root)
    out = io.StringIO()
    env = job_env(["lift sync pull", "lift run-dev --variant stock"])
    run = job.Job(root=str(root), environ=env, popen=vm, python=PY, version_info=(3, 12, 4), heartbeat=0,
                  out=out)
    vm.hook = lambda key: setattr(run, "_log", FullDiskLog()) if key.startswith("lift run-dev") else None
    status = run.run()
    assert vm.keys("lift ") == ["lift sync pull", "lift run-dev --variant stock", "lift sync push"]
    assert status["rc"] == 0 and status["push_rc"] == 0 and status["log_dropped"] > 0
    assert b"@@LIFT_DONE" not in log_bytes(root)
    text = out.getvalue()
    assert text.count("cannot be written") == 1
    # The last line went to the kernel's output, which `colab exec` relays to the laptop's exec log.
    assert remote.find_status(text.encode()) == status
    assert not any(secret in text for secret in SECRETS.values())


# Review 2, R32: a `lift sync push` line mid-job ran as a plain line: an exit 1 was not tried again and
# an exit 3 (done, with conflicts) stopped the job.
def test_an_explicit_push_line_runs_as_a_push_and_never_stops_the_job(job, tmp_path, monkeypatch):
    monkeypatch.setattr(job, "PUSH_DELAYS", (0.001, 0.001, 0.001), raising=False)
    lines = ["lift sync pull", "lift generate --variant stock --split dev", "lift sync push",
             "lift judge --variant stock --split dev"]
    root = tmp_path / "flaky"  # pushes: the checkpoint (0), the line (1, then 0), the final one (0)
    status, vm, _ = run_job(job, root, lines, vm=FakeVM(root, rcs={"lift sync push": [0, 1, 0]}))
    assert vm.keys("lift ") == ["lift sync pull", "lift generate --variant stock --split dev",
                                "lift sync push", "lift sync push", "lift sync push",
                                "lift judge --variant stock --split dev", "lift sync push"]
    assert status["rc"] == 0 and status["failed_line"] is None and status["checkpoint_push_failures"] == 0
    root = tmp_path / "conflict"
    status, vm, _ = run_job(job, root, lines, vm=FakeVM(root, rcs={"lift sync push": [0, 3, 0]}))
    assert "lift judge --variant stock --split dev" in vm.keys("lift ")
    assert status["rc"] == 0 and status["failed_line"] is None and status["push_conflicts"] == 1
    root = tmp_path / "down"  # a push line that keeps failing is reported, like a checkpoint push
    status, vm, _ = run_job(job, root, lines, vm=FakeVM(root, rcs={"lift sync push": [0, 1, 1, 1, 1, 0]}))
    assert "lift judge --variant stock --split dev" in vm.keys("lift ")
    assert status["rc"] == 0 and status["checkpoint_push_failures"] == 1


SWEEP = ["lift sync pull", "lift run-dev --variant stock",
         "lift run-dev --variant wiseft-l0.50-adapter-aya-std-s1",
         "lift run-dev --variant resta-a0.50-adapter-aya-std-s1"]
SKIPPED = (b"1 cells were skipped (see the errors above), each with its remedy:\n"
           b"  harmful_hi_dev: lift restart --variant stock --split dev --sets harmful --langs hi\n")


# Review 4, W8 and contract G: generate, judge and run-dev exit 3 for "done, but some cells were skipped".
# job.py ended the job at such a line, so one stale cell of stock skipped every repair variant after it.
def test_a_line_that_skipped_cells_is_reported_and_the_next_lines_run(job, tmp_path):
    root = tmp_path / "content"
    root.mkdir()
    vm = FakeVM(root, rcs={SWEEP[1]: 3}, outputs={SWEEP[1]: SKIPPED})
    status, vm, out = run_job(job, root, SWEEP, vm=vm)
    assert vm.keys("lift ") == [SWEEP[0], SWEEP[1], "lift sync push", SWEEP[2], "lift sync push", SWEEP[3],
                                "lift sync push"]
    assert status["partial_lines"] == [1] and status["failed_line"] is None and status["stopped"] is None
    assert status["rc"] == 3 and status["push_rc"] == 0 and status["checkpoint_push_failures"] == 0
    log = log_bytes(root)
    assert SKIPPED in log and b"line 2/4 exited 3" in log and b"the job fails at the end" in log
    assert "1 line(s) exited 3" in out
    # lift remote: a job line failed (1), also when the push had to be retried from the laptop.
    assert remote.exit_code(status) == 1 and "line 2 exited 3" in remote.describe(status)
    assert remote.exit_code({**status, "push_rc": 1}, rescue={"push_rc": 0}) == 1
    # A line that fails after it still ends the job, and so does exit 3 of any other command.
    root = tmp_path / "then-fails"
    lines = ["lift generate --variant stock --split dev", "lift judge --variant stock --split dev",
             "lift info"]
    status, vm, _ = run_job(job, root, lines, vm=FakeVM(root, rcs={lines[0]: 3, lines[1]: 1}))
    assert vm.keys("lift ") == [lines[0], "lift sync push", lines[1], "lift sync push"]
    assert status["partial_lines"] == [0] and status["failed_line"] == 1 and status["rc"] == 1
    root = tmp_path / "smoke"
    status, vm, _ = run_job(job, root, ["lift smoke", "lift info"], vm=FakeVM(root, rcs={"lift smoke": 3}))
    assert vm.keys("lift ") == ["lift smoke", "lift sync push"]
    assert status["partial_lines"] == [] and status["failed_line"] == 0 and status["rc"] == 3


# P03: lift utility, belebele and heldout-loss exit 3 as generate, judge and run-dev do: done, but some
# cells were skipped (a stale scores file) or left for a rerun (lift utility --no-wait while its Message
# Batch runs).
P03_PARTIAL = ["lift utility --variant adapter-aya-std-s1 --split dev --no-wait",
               "lift belebele --variant adapter-aya-std-s1 --split dev --langs hi,en",
               "lift heldout-loss --variant adapter-aya-std-s1"]
PARTIAL_COMMANDS = ("generate", "judge", "run-dev", "utility", "belebele", "heldout-loss", "final")


def readme_partial() -> tuple[str, ...]:
    """The PARTIAL tuple of colab/README.md's notebook fallback cell."""
    tree = ast.parse(notebook_cell('JOB = """'))
    (value,) = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and [getattr(t, "id", None) for t in node.targets] == ["PARTIAL"]]
    return ast.literal_eval(value)


def kaggle_cell_commands() -> tuple[str, ...]:
    """CELL_COMMANDS of kaggle/runner.ipynb's job cell."""
    cells = json.loads((REPO_ROOT / "kaggle" / "runner.ipynb").read_text(encoding="utf-8"))["cells"]
    (source,) = ["".join(c["source"]) for c in cells if c.get("id") == "job"]
    (value,) = [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                and [getattr(t, "id", None) for t in node.targets] == ["CELL_COMMANDS"]]
    return ast.literal_eval(value)


def test_the_runners_agree_on_which_commands_exit_3_and_go_on(job):
    """The list lives in four places a job reads it from (job.py, the README's notebook fallback, the Kaggle
    runner) or an operator does (lift remote's exit codes, the README's table): all name the same commands.
    lift train and lift gate never exit 3, so they are in none."""
    assert job.PARTIAL_COMMANDS == readme_partial() == kaggle_cell_commands() == PARTIAL_COMMANDS
    named = "a generate, judge, run-dev, utility, belebele, heldout-loss or final line exited 3"
    assert named in " ".join(remote.EXIT_CODES.split())
    row = next(line for line in readme().splitlines() if line.startswith("| 1 |"))
    assert named in row.lower()


# The lines a T4 runs with --dtype fp16, as kaggle/README.md's "fp16 everywhere" bullet names them.
T4_FP16 = ("generate", "judge", "run-dev", "train", "utility --generate-only", "belebele", "heldout-loss")
# A command in backticks, with or without "lift" ("`lift utility --generate-only`", "`run-dev`"); not a flag.
LIFT_COMMAND = re.compile(r"`(?:lift )?([a-z][a-z-]*(?: --[a-z-]+)*)`")


def sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()))


def kaggle_about() -> str:
    """The first (markdown) cell of kaggle/runner.ipynb, which a Kaggle user reads while filling JOB."""
    cells = json.loads((REPO_ROOT / "kaggle" / "runner.ipynb").read_text(encoding="utf-8"))["cells"]
    (source,) = ["".join(c["source"]) for c in cells if c.get("id") == "about"]
    return source


# Review C14: after P03, CLAUDE.md (the brief every session reads) still named only generate, judge and
# run-dev as the commands whose exit 3 the runners go past, and the Kaggle runner's T4 rule gave
# --dtype fp16 to their lines alone, while kaggle/README.md gives it to train, utility --generate-only,
# belebele and heldout-loss lines too: a T4 user following the notebook lost the job to a refused line.
def test_the_docs_name_every_command_that_exits_3_and_every_t4_model_line(job):
    brief = sentences((REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
    (threes,) = [s for s in brief if re.search(r"\bexit 3\b", s) and "runners go on" in s]
    for cmd in job.PARTIAL_COMMANDS:
        assert re.search(rf"`(lift )?{cmd}`", threes), cmd
    assert "`lift train` and `lift gate` never exit 3" in threes
    (t4,) = [s for s in brief if "T4" in s and "`--dtype fp16`" in s]
    for cmd in T4_FP16:
        assert re.search(rf"`(lift )?{cmd}`", t4), cmd
    readme = (REPO_ROOT / "kaggle" / "README.md").read_text(encoding="utf-8")
    bullet = " ".join(readme.split("**fp16 everywhere.**", 1)[1].split("\n- **", 1)[0].split())
    rule = " ".join(kaggle_about().split("**The T4 rule.**", 1)[1].split("\n\n", 1)[0].split())
    assert set(T4_FP16) <= set(LIFT_COMMAND.findall(bullet)) <= set(LIFT_COMMAND.findall(rule))
    assert "`--dtype fp16`" in rule
    (train,) = re.findall(r"`lift train` \(([^)]*)\)", bullet)
    for flags in re.findall(r"`(--[^`]+)`", train):  # the T4's own training flags
        assert flags in rule, flags


@pytest.mark.parametrize("line", P03_PARTIAL)
def test_a_p03_line_that_exits_3_is_reported_and_the_next_lines_run(job, tmp_path, line):
    root = tmp_path / "content"
    lines = ["lift sync pull", line, "lift info"]
    status, vm, out = run_job(job, root, lines, vm=FakeVM(root, rcs={line: 3}))
    assert vm.keys("lift ") == [lines[0], line, "lift sync push", "lift info", "lift sync push"]
    assert status["partial_lines"] == [1] and status["failed_line"] is None and status["rc"] == 3
    log = log_bytes(root)
    assert b"line 2/3 exited 3" in log and b"left for a rerun" in log and b"the job fails at the end" in log
    assert remote.exit_code(status) == 1 and "skipped or left for a rerun" in remote.describe(status)


@pytest.mark.parametrize("line", ["lift train --candidate aya --recipe standard --seed 1", "lift gate 1"])
def test_an_exit_3_of_lift_train_or_gate_ends_the_job(job, tmp_path, line):
    root = tmp_path / "content"
    status, vm, _ = run_job(job, root, [line, "lift info"], vm=FakeVM(root, rcs={line: 3}))
    assert vm.keys("lift ") == [line, "lift sync push"]
    assert status["partial_lines"] == [] and status["failed_line"] == 0 and status["rc"] == 3


# --- lift remote ----------------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1_800_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeHandle:
    """The background `colab exec` client."""

    def __init__(self):
        self.rc = None
        self.killed = False

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        if self.rc is None:
            raise subprocess.TimeoutExpired("colab exec", timeout)
        return self.rc

    def kill(self):
        self.killed = True
        self.rc = 1


def push_marker(attempt: str, rc: int) -> bytes:
    status = {"mode": "push", "attempt": attempt, "job_id": "x", "push_rc": rc, "error": None}
    line = b"@@LIFT_DONE " + json.dumps(status, sort_keys=True).encode() + b"\n"
    return b"[lift-job] 11:00:00 push retry\n" + line


class FakeColab:
    """The Colab CLI and its server as lift remote sees them: sessions.json, the VM list, uploads, and
    a scripted job log (one entry per download: bytes, None for "not there yet", or a Result). A
    push-mode exec (lift remote's retry of a failed final push) appends its line with ``rescue_rc``."""

    def __init__(self, config_dir: Path, logs=(), *, upload_rc=0, stop_rc=0, on_download=None, on_upload=None,
                 new_mode="ok", rescue_rc=None):
        self.config_dir = config_dir
        self.logs = list(logs) or [None]
        self.upload_rc = upload_rc
        self.stop_rc = stop_rc
        self.on_download = on_download
        self.on_upload = on_upload
        self.new_mode = new_mode
        self.rescue_rc = rescue_rc
        self.server: dict[str, str] = {}  # endpoint -> session name
        self.calls: list[str] = []
        self.argvs: list[list[str]] = []
        self.uploads: list[dict] = []
        self.config_files: list[str] = []
        self.specs: list[dict] = []
        self.rescues: list[dict] = []
        self.handle = FakeHandle()

    def records(self) -> dict:
        path = self.config_dir / "sessions.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save(self, records: dict) -> None:
        (self.config_dir / "sessions.json").write_text(json.dumps(records), encoding="utf-8")

    def run(self, argv, *, timeout):
        assert argv[0] == EXE and timeout > 0
        self.argvs.append(list(argv))
        args, config = list(argv[1:]), None
        if args[0] == "--config":
            config, args = args[1], args[2:]
            self.config_files.append(config)
        self.calls.append(args[0])
        return getattr(self, "do_" + args[0])(args, config)

    def spawn(self, argv, *, log_path):
        assert argv[0] == EXE and argv[1] == "exec"
        self.argvs.append(list(argv))
        self.calls.append("exec")
        spec = json.loads(base64.b64decode(argv[argv.index("--env") + 1].split("=", 1)[1]))
        self.specs.append(spec)
        if spec.get("mode") == "push":
            self.rescues.append(spec)
            return FakeHandle()
        return self.handle

    def do_new(self, args, config):
        name = args[args.index("-s") + 1]
        endpoint = f"gpu-a100-s-{name}"
        self.server[endpoint] = name
        if self.new_mode == "orphan":  # the CLI timed out after the server assigned the VM: no record
            return remote.Result(None, "timed out after 900 s and was killed")
        records = self.records()
        records[name] = {"name": name, "token": "proxy-token-never-read", "url": "https://vm.invalid",
                         "endpoint": endpoint}
        self.save(records)
        return remote.Result(0, f"[colab] Creating session '{name}'...\n[colab] Session READY.\n")

    def do_upload(self, args, config):
        local, target = args[3], args[4]
        self.uploads.append({"local": local, "remote": target, "data": Path(local).read_bytes()})
        if self.upload_rc:
            return remote.Result(self.upload_rc, "[colab] Upload failed: 502 Server Error for url: https://vm.inv"
                                 f"alid/api/contents/{target}?authuser=0&colab-runtime-proxy-token=proxyTOKEN123")
        if self.on_upload:
            self.on_upload(target)
        return remote.Result(0, f"[colab] Uploaded '{local}' to '{target}'\n")

    def do_download(self, args, config):
        if self.on_download:
            self.on_download(self.calls.count("download"))
        step = self.logs.pop(0) if len(self.logs) > 1 else self.logs[0]
        if step is None:
            return remote.Result(1, f"[colab] Download failed: File or directory not found: {args[3]}\n")
        if isinstance(step, remote.Result):
            return step
        if self.rescue_rc is not None:
            step += b"".join(push_marker(spec["attempt"], self.rescue_rc) for spec in self.rescues)
        Path(args[4]).write_bytes(step)
        return remote.Result(0, f"[colab] Downloaded '{args[3]}' to '{args[4]}'\n")

    def do_stop(self, args, config):
        name = args[args.index("-s") + 1]
        if self.stop_rc:
            return remote.Result(self.stop_rc, "Traceback (most recent call last):\n  unassign failed\n")
        records = json.loads(Path(config).read_text(encoding="utf-8")) if config else self.records()
        if name not in records:
            return remote.Result(0, f"[colab] Session '{name}' not found.\n")
        self.server.pop(records[name]["endpoint"], None)
        if config is None:
            records.pop(name)
            self.save(records)
        return remote.Result(0, f"[colab] Stopping session '{name}'...\n[colab] Session terminated.\n")

    def do_sessions(self, args, config):
        names = {record["endpoint"]: name for name, record in self.records().items()}
        lines = [f"[{names.get(ep, '?')}] {ep} | Hardware: A100 | Shape: Standard | Variant: GPU"
                 for ep in self.server]
        if not lines:
            return remote.Result(0, "[colab] No active sessions found on server.\n")
        return remote.Result(0, "\n".join(lines) + "\n")


def setup(tmp_path, logs=(), *, environ=None, logged_in=True, hf_token=None, **fake_kwargs):
    config_dir = tmp_path / "colab-cli"
    config_dir.mkdir(parents=True, exist_ok=True)
    if logged_in:
        (config_dir / "token.json").write_text("{}", encoding="utf-8")
    fake = FakeColab(config_dir, logs, **fake_kwargs)
    clock = FakeClock()
    ctx = remote.Context(runner=fake, environ=dict(SECRETS) if environ is None else environ,
                         out=io.StringIO(), sleep=clock.sleep, clock=clock,
                         hf_token=hf_token or (lambda: None), colab_exe=EXE, config_dir=config_dir)
    return fake, ctx


def done_log(status: dict, body: bytes = b"") -> bytes:
    return LOG_START + body + b"@@LIFT_DONE " + json.dumps(status, sort_keys=True).encode() + b"\n"


def manifest(name: str) -> dict:
    return json.loads(remote.manifest_path(name).read_text(encoding="utf-8"))


LOG_START = b"[lift-job] 10:00:00 job started\n"
OK = {"rc": 0, "push_rc": 0, "failed_line": None, "failed_step": None, "seconds": {"total": 125.0}}
PUSH_FAILED = {**OK, "rc": 1, "push_rc": 1, "installed": True}


def test_launch_runs_new_upload_exec_poll_stop_sessions(tmp_path):
    body = f"progress with {HF}\n".encode()
    logs = [None, LOG_START, LOG_START + body + b"partial", done_log(OK, body + b"partial line\n")]
    fake, ctx = setup(tmp_path, logs)
    fake.on_download = lambda n: setattr(fake.handle, "rc", 0) if n == 4 else None
    rc = remote.launch(remote.Launch(gpu="a100", job="lift sync pull; lift smoke", name="lift-t1"), ctx)

    assert rc == 0
    # `colab sessions` first: a VM the CLI has no record of ([?]) that appears later is then ours
    assert fake.calls == ["sessions", "new", "upload", "exec", "download", "download", "download", "download",
                          "stop", "sessions"]
    assert fake.argvs[1] == [EXE, "new", "-s", "lift-t1", "--gpu", "A100"]
    exec_argv = fake.argvs[3]
    assert exec_argv[1:8] == ["exec", "-s", "lift-t1", "-f", str(remote.JOB_FILE), "--timeout",
                              str(12 * 3600 + int(remote.STOP_WAIT + remote.PUSH_WAIT) + 900)]  # longest wait
    assert exec_argv[8] == "--env" and exec_argv[9].startswith("LIFT_JOB_B64=")
    spec = json.loads(base64.b64decode(exec_argv[9].split("=", 1)[1]))
    # The job stops itself 30 minutes before --max-hours, so its final push fits.
    assert spec == {"repo": "Knull/lost-in-finetuning", "ref": "main", "job_id": "lift-t1",
                    "lines": ["lift sync pull", "lift smoke"], "extras": "gpu,data,eval",
                    "deadline": 1_800_000_000.0 + 12 * 3600 - 1800, "max_silent_minutes": 45.0,
                    "mode": "job", "attempt": None}
    assert fake.argvs[4][1:] == ["download", "-s", "lift-t1", "content/lift_job.log",
                                 str(remote.log_path("lift-t1"))]
    for argv in fake.argvs:
        assert not any(secret in arg for arg in argv for secret in SECRETS.values())
    # The secrets went as a file, which is gone now; the exec client ended by itself.
    (upload,) = fake.uploads
    assert upload["remote"] == "content/.lift_secrets.json" and json.loads(upload["data"]) == SECRETS
    assert not Path(upload["local"]).exists()
    assert not fake.handle.killed and fake.server == {}

    text = ctx.out.getvalue()
    assert not any(secret in text for secret in SECRETS.values())
    assert "progress with ***" in text and text.count("partial") == 1 and "partial line" in text
    assert "job finished: rc 0" in text and "VM released" in text
    assert remote.log_path("lift-t1").read_bytes() == logs[-1]
    record = manifest("lift-t1")
    assert record["endpoint"] == "gpu-a100-s-lift-t1" and record["gpu"] == "A100"
    assert record["rc"] == 0 and record["outcome"] == "done" and record["vm_released"] is True
    assert record["job"]["push_rc"] == 0 and record["secrets_sent"] == sorted(SECRETS)
    assert not any(secret in remote.manifest_path("lift-t1").read_text(encoding="utf-8")
                   for secret in SECRETS.values())


def test_a_failed_line_exits_1_and_says_which(tmp_path):
    fake, ctx = setup(tmp_path, [done_log({"rc": 3, "push_rc": 0, "failed_line": 1})])
    assert remote.launch(remote.Launch(gpu="T4", job="lift info", name="lift-t2"), ctx) == 1
    assert manifest("lift-t2")["rc"] == 1 and manifest("lift-t2")["job"]["rc"] == 3 and fake.handle.killed
    assert "line 2 failed" in ctx.out.getvalue()


# Contract G: a job whose generate, judge or run-dev line exited 3 (cells skipped) ran its other lines,
# and lift remote exits 1 for it, also when the push retry from here sent its outputs.
def test_a_job_whose_lines_skipped_cells_exits_1(tmp_path):
    partial = {**OK, "rc": 3, "partial_lines": [1], "installed": True, "ran": 3}
    job_text = "; ".join(SWEEP[:3])
    fake, ctx = setup(tmp_path, [done_log(partial)])
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job=job_text, name="lift-p3"), ctx) == 1
    text = ctx.out.getvalue()
    assert manifest("lift-p3")["rc"] == 1 and "line 2 exited 3" in text
    fake, ctx = setup(tmp_path / "rescued", [done_log({**partial, "push_rc": 1})], rescue_rc=0)
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job=job_text, name="lift-p4"), ctx) == 1
    assert len(fake.rescues) == 1 and manifest("lift-p4")["rescue_push"]["push_rc"] == 0


@pytest.mark.parametrize("gpu", ["a10g", "H100", "cpu", "", None])
def test_unknown_gpu_is_refused_before_any_call(tmp_path, gpu):
    fake, ctx = setup(tmp_path)
    with pytest.raises(remote.RemoteError, match="A100") as err:
        remote.launch(remote.Launch(gpu=gpu, job="lift info"), ctx)
    assert err.value.code == 2 and fake.calls == []


def test_validate_gpu_normalises():
    assert remote.validate_gpu(" l4 ") == "L4" and remote.validate_gpu("a100") == "A100"


def test_missing_login_prints_the_login_command(tmp_path):
    fake, ctx = setup(tmp_path, logged_in=False)
    with pytest.raises(remote.RemoteError) as err:
        remote.launch(remote.Launch(gpu="A100", job="lift info"), ctx)
    assert err.value.code == 2 and "token.json" in str(err.value)
    assert remote.shell_command(EXE, "sessions") in str(err.value)
    assert fake.calls == []


def test_missing_secrets_are_refused(tmp_path):
    fake, ctx = setup(tmp_path, environ={"HF_TOKEN": HF})
    with pytest.raises(remote.RemoteError) as err:
        remote.launch(remote.Launch(gpu="A100", job="lift info"), ctx)
    assert "GITHUB_TOKEN" in str(err.value) and "LIFT_RUNS_REPO" in str(err.value)
    assert HF not in str(err.value) and fake.calls == []
    fake, ctx = setup(tmp_path, environ={"GITHUB_TOKEN": GH, "LIFT_RUNS_REPO": RUNS})
    with pytest.raises(remote.RemoteError, match="HF_TOKEN"):
        remote.launch(remote.Launch(gpu="A100", job="lift info"), ctx)


def test_hf_token_falls_back_to_the_cached_login_only_when_unset(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)], environ={"GITHUB_TOKEN": GH, "LIFT_RUNS_REPO": RUNS},
                      hf_token=lambda: HF)
    assert remote.launch(remote.Launch(gpu="L4", job="lift info", name="lift-t3"), ctx) == 0
    assert json.loads(fake.uploads[0]["data"])["HF_TOKEN"] == HF

    def must_not_read():
        raise AssertionError("the token file must not be read when HF_TOKEN is set")

    other = tmp_path / "other"
    other.mkdir()
    _, ctx = setup(other, environ=dict(SECRETS), hf_token=must_not_read)
    secrets, missing = remote.gather_secrets(ctx)
    assert secrets["HF_TOKEN"] == HF and missing == []


def test_upload_failure_deletes_the_temp_file_and_stops_the_vm(tmp_path):
    fake, ctx = setup(tmp_path, upload_rc=1)
    rc = remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t4"), ctx)
    assert rc == 3  # the job could not be set up
    assert fake.calls == ["sessions", "new", "upload", "upload", "upload", "upload", "stop", "sessions"]
    assert len(fake.uploads) == 4 and not any(Path(u["local"]).exists() for u in fake.uploads)
    text = ctx.out.getvalue()
    assert "proxyTOKEN123" not in text and "token=***" in text
    assert manifest("lift-t4")["vm_released"] is True and manifest("lift-t4")["outcome"].startswith("error")


# Review 2, R30: an unexpected error while following stopped the VM at once, with no stop request (so
# no final push), and escaped as a traceback (exit 1) even when the VM could not be released (4).
def test_an_unexpected_error_asks_the_job_to_stop_and_push_then_returns_a_code(tmp_path):
    def explode(n):
        if n == 2:
            raise RuntimeError("a bug while reading the log")

    fake, ctx = setup(tmp_path, [None, LOG_START], on_download=explode)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t5"), ctx) == 1
    assert [u["remote"] for u in fake.uploads] == ["content/.lift_secrets.json", STOP_FILE]
    assert fake.calls[-2:] == ["stop", "sessions"] and fake.handle.killed and fake.server == {}
    record = manifest("lift-t5")
    assert record["vm_released"] is True and record["outcome"].startswith("error: RuntimeError")
    assert record["rc"] == 1


def test_an_unexpected_error_still_exits_4_when_the_vm_cannot_be_released(tmp_path, monkeypatch):
    def explode(n):
        if n == 2:
            raise RuntimeError("a bug while reading the log")

    fake, ctx = setup(tmp_path, [None, LOG_START], on_download=explode, stop_rc=1)
    monkeypatch.setattr(remote, "Context", lambda *args, **kwargs: ctx)
    assert main(["remote", "--gpu", "A100", "--job", "lift info", "--name", "lift-t5b"]) == 4
    assert fake.server and manifest("lift-t5b")["rc"] == 4
    assert "MAY STILL BE RUNNING" in ctx.out.getvalue()


def test_a_colab_cli_that_cannot_start_is_a_failed_call():
    # `uv tool install google-colab-cli==<v> --force` replaces colab.exe while a job runs.
    def missing(argv, **kwargs):
        raise FileNotFoundError(2, "The system cannot find the file specified", argv[0])

    res = REAL_RUN(remote.SubprocessRunner({}, run=missing), [EXE, "download"], timeout=5)
    assert res.rc is None and "FileNotFoundError" in res.out


def answer_the_stop_file(fake: FakeColab, status: dict):
    """on_upload: once the stop request is up, the job's log ends with ``status``."""

    def on_upload(target):
        if target == STOP_FILE:
            fake.logs = [done_log(status)]

    return on_upload


def interrupting_sleep(ctx, *at: int):
    """ctx.sleep that advances the fake clock and raises KeyboardInterrupt on the given calls."""
    naps: list[float] = []
    clock = ctx.clock

    def sleep(seconds):
        naps.append(seconds)
        clock.now += seconds
        if len(naps) in at:
            raise KeyboardInterrupt

    return sleep


def ctrl_c_sleep(ctx, *at: int):
    """ctx.sleep that advances the fake clock and presses Ctrl+C on the given calls: a real SIGINT, which
    the default handler turns into a KeyboardInterrupt, as at the console."""
    naps: list[float] = []
    clock = ctx.clock

    def sleep(seconds):
        naps.append(seconds)
        clock.now += seconds
        if len(naps) in at:
            signal.raise_signal(signal.SIGINT)

    return sleep


def test_ctrl_c_before_the_job_line_still_stops_the_vm(tmp_path):
    fake, ctx = setup(tmp_path)
    ctx.sleep = interrupting_sleep(ctx, 2)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t6"), ctx) == 130
    assert fake.calls[-2:] == ["stop", "sessions"] and fake.handle.killed and fake.server == {}
    assert manifest("lift-t6")["outcome"] == "interrupted"
    # The stop request went up, but no job log ever appeared: the operator is told nothing was lost.
    assert fake.uploads[-1]["remote"] == STOP_FILE
    text = ctx.out.getvalue()
    assert "the job never started" in text and "nothing was lost" in text and "may be lost" not in text


# Review 4, W13: the closing message said outputs were lost when no job line could have run.
def test_a_stop_before_any_job_line_could_run_says_nothing_was_lost(tmp_path):
    # Ctrl+C while the secrets go up to a fresh VM that does not answer yet: no job was started.
    fake, ctx = setup(tmp_path, upload_rc=1)
    ctx.sleep = interrupting_sleep(ctx, 1)
    opts = remote.Launch(gpu="A100", job="lift sync pull; lift smoke", name="lift-cs")
    assert remote.launch(opts, ctx) == 130
    text = ctx.out.getvalue()
    assert "exec" not in fake.calls and fake.server == {}
    assert "before the job was started" in text and "nothing was lost" in text
    assert "are lost" not in text and "may be lost" not in text
    # Ctrl+C twice after the exec, before job.py wrote its log: no line can have run either.
    fake, ctx = setup(tmp_path / "twice", [None])
    ctx.sleep = interrupting_sleep(ctx, 2, 3)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-cs2"), ctx) == 130
    text = ctx.out.getvalue()
    assert fake.server == {} and "before the job's log appeared" in text and "nothing was lost" in text
    assert "are lost" not in text and "may be lost" not in text
    # The exec client failed before the job's log appeared: the job never started.
    fake, ctx = setup(tmp_path / "no-start", [None])
    fake.handle.rc = 1
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-cs3"), ctx) == 1
    text = ctx.out.getvalue()
    assert manifest("lift-cs3")["outcome"] == "no-start"
    assert "the job never started" in text and "nothing was lost" in text and "may be lost" not in text
    # Once the log was seen, a line can have run: the warning stays.
    fake, ctx = setup(tmp_path / "seen", [LOG_START])
    ctx.sleep = interrupting_sleep(ctx, 2, 3)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-cs4"), ctx) == 130
    text = ctx.out.getvalue()
    assert "since the last successful push are lost" in text
    # So it does when nobody looked for the log within minutes of the start (a long --poll), and when
    # job.py ran without a log it could not open (a full disk): it says so on the exec client's output.
    fake, ctx = setup(tmp_path / "late", [None])
    ctx.sleep = interrupting_sleep(ctx, 1, 2)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-cs5", poll=600), ctx) == 130
    text = ctx.out.getvalue()
    assert "since the last successful push are lost" in text
    fake, ctx = setup(tmp_path / "no-log", [None])
    fake.handle.rc = 0
    exec_log = remote.exec_log_path("lift-cs6")
    exec_log.parent.mkdir(parents=True, exist_ok=True)
    exec_log.write_bytes(b"[lift-job] 10:00:00 cannot open /content/lift_job.log (OSError): job.py goes on\n")
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-cs6"), ctx) == 1
    text = ctx.out.getvalue()
    assert "may be lost" in text and "nothing was lost" not in text


# Review #33: Ctrl+C unassigned the VM at once, so job.py never ran its final push.
def test_ctrl_c_asks_the_job_to_stop_and_push_before_the_vm_is_stopped(tmp_path):
    stopped = {**OK, "rc": 130, "failed_line": 1, "stopped": "stop file: ctrl-c", "installed": True}
    fake, ctx = setup(tmp_path, [LOG_START])
    fake.on_upload = answer_the_stop_file(fake, stopped)
    ctx.sleep = interrupting_sleep(ctx, 3)
    rc = remote.launch(remote.Launch(gpu="A100", job="lift sync pull; lift run-dev --variant stock",
                                     name="lift-c1"), ctx)
    assert rc == 130
    # The stop request went up, the log was read until the job's last line, and only then the VM stopped.
    assert [u["remote"] for u in fake.uploads] == ["content/.lift_secrets.json", STOP_FILE]
    assert fake.uploads[1]["data"] == b"ctrl-c\n"
    i = fake.calls.index("stop")
    assert fake.calls[i - 2:] == ["upload", "download", "stop", "sessions"] and fake.server == {}
    record = manifest("lift-c1")
    assert record["outcome"] == "interrupted" and record["job"]["stopped"] == "stop file: ctrl-c"
    assert "Ctrl+C again" in ctx.out.getvalue()


def test_a_second_ctrl_c_stops_the_vm_at_once_and_says_what_is_lost(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START])
    ctx.sleep = interrupting_sleep(ctx, 3, 5)
    rc = remote.launch(remote.Launch(gpu="A100", job="lift sync pull; lift run-dev --variant stock",
                                     name="lift-c2"), ctx)
    assert rc == 130 and fake.server == {}
    # One poll after the stop request, the second Ctrl+C: the VM is stopped without waiting for the job.
    assert fake.calls[-4:] == ["upload", "download", "stop", "sessions"]
    assert [u["remote"] for u in fake.uploads] == ["content/.lift_secrets.json", STOP_FILE]
    assert "since the last successful push are lost" in ctx.out.getvalue()
    assert manifest("lift-c2")["outcome"] == "interrupted" and manifest("lift-c2")["job"] is None


def test_exec_ending_without_the_marker_is_a_failure(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START])
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t7"), ctx) == 1
    assert fake.calls.count("download") == 3 and manifest("lift-t7")["outcome"] == "no-marker"


def test_exec_failing_before_the_log_appears_is_a_failure(tmp_path):
    fake, ctx = setup(tmp_path, [None])
    fake.handle.rc = 1
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t8"), ctx) == 1
    assert manifest("lift-t8")["outcome"] == "no-start" and fake.calls[-2:] == ["stop", "sessions"]


def test_exec_client_dropping_mid_job_keeps_polling(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START, LOG_START, LOG_START, LOG_START, done_log(OK)])
    fake.handle.rc = 1  # the websocket dropped; the job runs on
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t9"), ctx) == 0
    assert fake.calls.count("download") == 5


def test_a_log_that_stops_growing_means_a_dead_job(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START])  # job.py logs a heartbeat every 5 minutes; this log never grows
    opts = remote.Launch(gpu="A100", job="lift info", name="lift-t12", poll=60)
    assert remote.launch(opts, ctx) == 1
    assert manifest("lift-t12")["outcome"] == "stale" and fake.calls[-2:] == ["stop", "sessions"]
    assert fake.calls.count("download") == 22 and fake.handle.killed


def test_max_hours_asks_the_job_to_stop_and_push_and_the_job_knows_its_deadline(tmp_path):
    stopped = {**OK, "rc": 130, "failed_line": 0, "stopped": "deadline reached", "installed": True}
    fake, ctx = setup(tmp_path, [LOG_START])
    fake.on_upload = answer_the_stop_file(fake, stopped)
    opts = remote.Launch(gpu="A100", job="lift run-dev --variant stock", name="lift-t10", max_hours=0.25,
                         poll=60)
    assert remote.launch(opts, ctx) == 1
    # The job's own deadline: a quarter of a short cap before it (30 minutes for long ones).
    assert fake.specs[0]["deadline"] == 1_800_000_000.0 + 900 - 225
    assert [u["remote"] for u in fake.uploads] == ["content/.lift_secrets.json", STOP_FILE]
    assert fake.uploads[1]["data"] == b"max-hours\n" and fake.calls[-2:] == ["stop", "sessions"]
    record = manifest("lift-t10")
    assert record["outcome"] == "max-hours" and record["job"]["stopped"] == "deadline reached"


def test_max_hours_without_an_answer_stops_the_vm_after_the_wait(tmp_path):
    fake, ctx = setup(tmp_path, [None])
    opts = remote.Launch(gpu="A100", job="lift info", name="lift-t13", max_hours=0.05, poll=60)
    assert remote.launch(opts, ctx) == 1
    assert fake.calls[:7] == ["sessions", "new", "upload", "exec", "download", "download", "download"]
    assert fake.calls[7:9] == ["download", "upload"]  # one last look, then the stop request
    assert fake.calls[-2:] == ["stop", "sessions"] and fake.server == {}
    assert manifest("lift-t13")["outcome"] == "max-hours"
    # No job log ever appeared (review 4, W13): nothing was lost, and the closing message says so.
    text = ctx.out.getvalue()
    assert "the job never started" in text and "nothing was lost" in text and "may be lost" not in text


# Review #35: past the cap, the log was never read, so a finished job was reported as max-hours.
def test_the_log_is_read_once_more_when_max_hours_is_reached(tmp_path):
    fake, ctx = setup(tmp_path, [None, None, None, done_log(OK)])
    opts = remote.Launch(gpu="A100", job="lift info", name="lift-cap", max_hours=0.05, poll=60)
    assert remote.launch(opts, ctx) == 0
    assert fake.calls.count("download") == 4 and manifest("lift-cap")["outcome"] == "done"
    assert STOP_FILE not in [u["remote"] for u in fake.uploads]


def test_a_vanished_vm_ends_the_job(tmp_path):
    failed = remote.Result(1, "[colab] Download failed: Max retries exceeded, url: /api/contents/x?token=abc")
    fake, ctx = setup(tmp_path, [failed])
    fake.on_download = lambda n: fake.server.clear()
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-t11"), ctx) == 1
    assert fake.calls == ["sessions", "new", "upload", "exec", "download", "download", "download", "sessions",
                          "stop", "sessions"]
    assert manifest("lift-t11")["outcome"] == "lost" and "token=abc" not in ctx.out.getvalue()


def test_a_reused_name_is_refused(tmp_path):
    fake, ctx = setup(tmp_path)
    fake.do_new(["new", "-s", "lift-dup"], None)
    with pytest.raises(remote.RemoteError, match="orphans"):
        remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-dup"), ctx)
    with pytest.raises(remote.RemoteError, match="session name"):
        remote.launch(remote.Launch(gpu="A100", job="lift info", name="../evil"), ctx)


# Review #10 and #30: after a failed final push, lift remote stopped the VM at once.
def test_a_failed_final_push_is_retried_once_from_here_before_the_vm_is_stopped(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], rescue_rc=0)
    fake.handle.rc = 0
    rc = remote.launch(remote.Launch(gpu="A100", job="lift sync pull; lift run-dev --variant stock",
                                     name="lift-r1"), ctx)
    assert rc == 0
    i = fake.calls.index("stop")
    assert fake.calls[:i].count("upload") == 2 and fake.calls[:i].count("exec") == 2
    (rescue,) = fake.rescues  # job.py in push mode: it reads and deletes the secrets, pushes once
    assert rescue["mode"] == "push" and rescue["lines"] == [] and rescue["job_id"] == "lift-r1"
    assert fake.uploads[1]["remote"] == "content/.lift_secrets.json"
    assert json.loads(fake.uploads[1]["data"]) == SECRETS and not Path(fake.uploads[1]["local"]).exists()
    record = manifest("lift-r1")
    assert record["rescue_push"]["push_rc"] == 0 and record["rc"] == 0 and fake.server == {}


def test_outputs_that_cannot_be_pushed_exit_5_with_a_loud_warning(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], rescue_rc=1)
    fake.handle.rc = 0
    rc = remote.launch(remote.Launch(gpu="A100", job="lift sync pull; lift run-dev --variant stock",
                                     name="lift-r2"), ctx)
    assert rc == 5 and len(fake.rescues) == 1 and fake.server == {}
    text = ctx.out.getvalue()
    assert "SINCE ITS LAST SUCCESSFUL PUSH ARE LOST" in text
    assert manifest("lift-r2")["rc"] == 5 and manifest("lift-r2")["rescue_push"]["push_rc"] == 1


def test_a_missing_final_push_rc_counts_as_a_failed_push(tmp_path):
    # The line of a job.py whose final push was interrupted: rc 0, but nothing pushed.
    status = {**OK, "push_rc": None}
    fake, ctx = setup(tmp_path, [done_log(status)], rescue_rc=0)
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-r3"), ctx) == 0
    assert len(fake.rescues) == 1
    (tmp_path / "again").mkdir()
    fake, ctx = setup(tmp_path / "again", [done_log(status)])  # this time the retry never answers
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-r4"), ctx) == 5


# Review #62: a setup failure on the VM exited 2, the code for "lift remote refused to start".
@pytest.mark.parametrize("status", [
    {"rc": 2, "push_rc": None, "failed_line": None, "failed_step": "python", "installed": False},
    {"rc": 128, "push_rc": None, "failed_line": None, "failed_step": "fetch", "installed": False},
])
def test_a_setup_failure_on_the_vm_exits_3(tmp_path, status):
    fake, ctx = setup(tmp_path, [done_log(status)])
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-s1"), ctx) == 3
    assert fake.rescues == [] and fake.calls[:4] == ["sessions", "new", "upload", "exec"]


# Review #61: lift remote exited 0 while the VM it could not release kept billing.
def test_a_vm_that_cannot_be_released_exits_4(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)], stop_rc=1)
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-u1"), ctx) == 4
    assert fake.server and manifest("lift-u1")["vm_released"] is False
    assert manifest("lift-u1")["rc"] == 4 and "MAY STILL BE RUNNING" in ctx.out.getvalue()


# Review #32: a stdout write error (`lift remote ... | head`) skipped the VM release.
class ClosedAfter:
    """stdout of ``lift remote ... | head -n N``: after N writes the reader is gone."""

    def __init__(self, n):
        self.n = n
        self.text = []

    def write(self, s):
        if self.n <= 0:
            raise BrokenPipeError(32, "Broken pipe")
        self.n -= 1
        self.text.append(s)
        return len(s)

    def flush(self):
        if self.n <= 0:
            raise BrokenPipeError(32, "Broken pipe")


def test_a_broken_stdout_never_keeps_the_vm_running(tmp_path):
    body = b"".join(b"[lift-job] 10:00:%02d progress\n" % i for i in range(40))
    fake, ctx = setup(tmp_path, [LOG_START + body, done_log(OK, body)])
    fake.handle.rc = 0
    ctx.out = ClosedAfter(8)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-p1"), ctx) == 0
    assert fake.calls[-2:] == ["stop", "sessions"] and fake.server == {}
    assert manifest("lift-p1")["vm_released"] is True


# Contract 1 (review 2, R25/R34): nothing checked the runs repo before paying for a VM, so a read-only
# token was found only by the job's first push, hours later.
def test_launch_checks_the_runs_repo_before_any_vm(tmp_path, runs_repo_check, capsys, monkeypatch):
    fake, ctx = setup(tmp_path, [done_log(OK)])
    monkeypatch.setattr(remote, "Context", lambda *args, **kwargs: ctx)
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-pf"), ctx) == 0
    assert runs_repo_check.calls == [True] and fake.calls[0] == "sessions"
    # A problem no retry can fix refuses (2); a transient error exits 1; both before any VM exists.
    for permanent, code in ((True, 2), (False, 1)):
        runs_repo_check.errors.append(sync_error(f"the runs repo {RUNS} is not private", permanent=permanent))
        fake.calls.clear()
        with pytest.raises(remote.RemoteError) as err:
            remote.launch(remote.Launch(gpu="A100", job="lift info"), ctx)
        assert err.value.code == code and fake.calls == [] and "runs repo" in str(err.value)
        assert RUNS not in str(err.value)  # masked like every other value sent to the VM
    runs_repo_check.errors.append(sync_error("HTTP 403: this token cannot write", permanent=True))
    assert main(["remote", "--gpu", "A100", "--job", "lift info", "--name", "lift-pf2"]) == 2
    assert "cannot write" in capsys.readouterr().err
    runs_repo_check.errors.append(ModuleNotFoundError("no huggingface_hub", name="huggingface_hub"))
    with pytest.raises(remote.RemoteError, match=r"\.\[sync\]") as err:
        remote.launch(remote.Launch(gpu="A100", job="lift info"), ctx)
    assert err.value.code == 2
    # --dry-run calls nothing: it says the runs repo would be checked.
    calls = len(runs_repo_check.calls)
    opts = remote.Launch(gpu="A100", job="lift info", name="lift-pf3", dry_run=True)
    assert remote.launch(opts, ctx) == 0 and len(runs_repo_check.calls) == calls
    assert "lift sync check --write" in ctx.out.getvalue()


def replace_opts(opts: remote.Launch, **changes) -> remote.Launch:
    from dataclasses import replace

    return replace(opts, **changes)


# Review 2, R6: the dev eval was launched with configs/project.yaml still unpinned, and refused on the
# VM after about 10 A100-minutes.
def test_eval_lines_warn_while_this_laptops_revisions_are_unpinned(tmp_path, monkeypatch):
    from lift import config

    fake, ctx = setup(tmp_path)
    opts = remote.Launch(gpu="A100", job="lift sync pull; lift run-dev --variant stock", name="lift-pin",
                         dry_run=True)
    assert remote.launch(opts, ctx) == 0  # a warning, not a refusal
    text = ctx.out.getvalue()
    assert "unpinned" in text and "models.global.revision = 'main'" in text and "ref's copy" in text
    # Gate 0 itself needs no pin.
    ctx.out = io.StringIO()
    assert remote.launch(replace_opts(opts, job="lift sync pull; lift smoke", name="lift-pin2"), ctx) == 0
    assert "unpinned" not in ctx.out.getvalue()
    # A real launch warns too, and runs.
    fake, ctx = setup(tmp_path / "real", [done_log(OK)])
    fake.handle.rc = 0
    run = remote.Launch(gpu="A100", job="lift judge --variant stock --split dev", name="lift-pin3")
    assert remote.launch(run, ctx) == 0 and "unpinned" in ctx.out.getvalue()
    # Pinned to the shas Gate 0 printed: no warning.
    view = config.load("project")
    for entry in (view["models"]["global"], view["models"]["base"], view["judges"]["safety_primary"]):
        entry["revision"] = SHA
    ctx.out = io.StringIO()
    assert remote.launch(replace_opts(opts, name="lift-pin4"), ctx) == 0
    assert "unpinned" not in ctx.out.getvalue()
    # Advice only: revisions it cannot read never block a launch.
    real_get = config.get

    def unreadable(dotted, *args, **kwargs):
        if dotted.endswith(".revision"):
            raise ValueError("a malformed entry")
        return real_get(dotted, *args, **kwargs)

    monkeypatch.setattr(config, "get", unreadable)
    assert remote.launch(replace_opts(opts, name="lift-pin5"), ctx) == 0
    assert "could not read configs/project.yaml" in ctx.out.getvalue()


# Review 3, V13/V18: the warning checked a fixed list of entries. After P04's judge swap it named the
# frontier primary (no revision) on every eval line, and never the unpinned local second judge.
def test_the_pin_warning_checks_what_each_lines_command_refuses():
    from lift import config

    view = config.load("project")  # every revision "main"; Qwen3Guard is the primary judge
    warn = remote.unpinned_warning
    generate = warn(["lift sync pull", "lift generate --variant stock --split dev"])
    assert "line 2 (lift generate): models.global.revision = 'main'" in generate
    assert "judges." not in generate and "models.base" not in generate
    judge = warn(["lift judge --variant stock --split dev"])  # lift judge pins only its judge
    assert "judges.safety_primary.revision = 'main'" in judge and "models.global" not in judge
    both = warn(["lift sync pull", "lift run-dev --variant stock"])
    assert "models.global.revision" in both and "judges.safety_primary.revision" in both
    assert warn(["lift judge --judge=second --variant stock --split dev"]) is None  # frontier: no revision
    assert warn(["lift run-dev --variant stock --sets"]) is None  # refused by argparse, before any pin
    # After P04's swap: the frontier judge is primary, Qwen3Guard the second judge.
    qwen = view["judges"]["safety_primary"]
    view["judges"]["safety_primary"] = {"provider": "anthropic", "model": "claude-sonnet-5-5",
                                        "rubric": "configs/rubrics/safety_v1.md"}
    view["judges"]["safety_second"] = qwen
    view["models"]["global"]["revision"] = SHA
    assert warn(["lift sync pull", "lift run-dev --variant stock"]) is None
    second = warn(["lift sync pull", "lift judge --variant stock --split dev --judge second"])
    assert "line 2 (lift judge): judges.safety_second.revision = 'main'" in second
    assert "safety_primary" not in second
    qwen["revision"] = SHA
    assert warn(["lift judge --judge second --variant stock --split dev"]) is None
    # An entry without a revision key is skipped, never an error.
    del view["models"]["global"]["revision"]
    assert warn(["lift generate --variant stock --split dev"]) is None


# P03: the pin warning also covers the commands that load Global: lift train, utility (not --judge-only,
# which only judges generations made before, with either judge), belebele and heldout-loss. lift gate runs
# on the laptop and loads no model. Review C13: this was checked against hand-written stand-ins for their
# parsers, so a renamed flag, a new required option or another dest for --judge-only (pinned_entries
# reads args.judge_only) kept it green while lift remote warned about the wrong lines. Every line here
# goes through lift's own parser (lift.cli.build_parser, as parse_line reads it), the real P03 command
# modules included.
def test_the_pin_warning_covers_the_p03_commands_that_load_global():
    from lift import config

    warn = remote.unpinned_warning
    lines = ["lift train --candidate aya --recipe standard --seed 1 --per-device-batch 4",
             "lift train --candidate aya --recipe standard --seed 2 --dtype fp16 --per-device-batch 1 "
             "--parity-dtype fp16",
             "lift utility --variant adapter-aya-std-s1 --split dev",
             "lift utility --variant adapter-aya-std-s1 --split dev --judge fallback",
             "lift utility --variant adapter-aya-std-s1 --split dev --generate-only --dtype fp16",
             "lift belebele --variant stock --split dev --langs hi,en",
             "lift belebele --variant adapter-aya-std-s1 --split dev --langs hi en --dtype fp16",
             "lift heldout-loss --variant stock",
             "lift heldout-loss --variant adapter-aya-std-s1 --split dev --restart"]
    modules = {"train": "train", "utility": "utility", "belebele": "loglik", "heldout-loss": "loglik"}
    for line in lines:  # read by the command modules themselves, not by a stand-in
        args = remote.parse_line(line)
        assert args is not None and args.func.__module__ == f"lift.commands.{modules[args.command]}", line
    found = warn(["lift sync pull", *lines])  # unpinned: the autouse fixture serves every revision "main"
    for i, line in enumerate(lines, 2):
        assert f"line {i} (lift {shlex.split(line)[1]}): models.global.revision = 'main'" in found
    assert "judges." not in found and "models.base" not in found
    judging = ["lift utility --variant adapter-aya-std-s1 --split dev --judge-only",
               "lift utility --variant adapter-aya-std-s1 --split dev --judge fallback --judge-only",
               "lift utility --variant adapter-aya-std-s1 --split dev --judge-only --judge=frontier "
               "--no-wait"]
    for line in judging:  # parsed, so a renamed --judge-only fails here rather than reading as refused
        args = remote.parse_line(line)
        assert args is not None and getattr(args, "judge_only", None) is True, line
        assert warn([line]) is None, line
    refused = ["lift train --candidate aya --seed 1",                       # --recipe is required
               "lift utility --variant adapter-aya-std-s1 --split dev --generate-only --judge-only",
               "lift belebele --variant stock",                             # --split and --langs too
               "lift heldout-loss --variant stock --langs hi"]              # Hindi only: no --langs
    for line in refused:  # lift refuses them before any pin check: nothing to warn about
        assert remote.parse_line(line) is None and warn([line]) is None, line
    for line in ("lift gate 1", "lift gate 2 --candidate adapter-aya-std-s1",
                 "lift register-adapter adapter-aya-std-s1"):
        assert warn([line]) is None, line
    # Pinned to a sha: no warning, though the safety judges stay "main" (no P03 line runs one).
    config.load("project")["models"]["global"]["revision"] = SHA
    assert warn(["lift sync pull", *lines, *judging]) is None


# Contract 1: a final push that exited 2 (a problem no retry can fix) is never retried from here either.
def test_lift_remote_never_retries_a_final_push_that_exited_2(tmp_path):
    refused = {**PUSH_FAILED, "rc": 2, "push_rc": 2, "ran": 1, "sync_refused": "the final push"}
    fake, ctx = setup(tmp_path, [done_log(refused)], rescue_rc=0)
    fake.handle.rc = 0
    rc = remote.launch(remote.Launch(gpu="A100", job="lift run-dev --variant stock", name="lift-x2"), ctx)
    assert rc == 5 and fake.rescues == [] and fake.server == {}
    text = ctx.out.getvalue()
    assert "ARE LOST" in text and "exited 2" in text and "so did the retry" not in text


# Review 2, R1: a push that exits 2 is never "outputs lost" when no line ran.
def test_a_sync_refusal_before_any_line_is_a_setup_failure(tmp_path):
    status = {"rc": 2, "push_rc": None, "failed_line": 0, "failed_step": None, "installed": True, "ran": 0,
              "sync_refused": "line 1/2 (lift sync pull)"}
    fake, ctx = setup(tmp_path, [done_log(status)])
    fake.handle.rc = 0
    rc = remote.launch(remote.Launch(gpu="A100", job="lift sync pull; lift smoke", name="lift-x3"), ctx)
    assert rc == 3 and fake.rescues == [] and "ARE LOST" not in ctx.out.getvalue()
    assert "no retry can fix" in ctx.out.getvalue()
    check = {"rc": 3, "push_rc": None, "failed_line": None, "failed_step": "sync-check", "installed": False}
    assert remote.exit_code(check) == remote.EXIT_SETUP


def clock_text(seconds: float) -> str:
    return remote.hms(seconds).rjust(8, "0")


def job_note(t: float, message: str) -> bytes:
    """One of job.py's own log lines, as it writes them, t seconds past 10:00:00."""
    return f"[lift-job] {clock_text(36000 + t)} {message}\n".encode()


def pushing_log(minutes: float, *, done_at: float | None = None, status: dict | None = None) -> bytes:
    """The job log ``minutes`` after a stop request, its final push stalled: job.py's stop note, the first
    try, then a heartbeat every 5 minutes that says the push runs. At ``done_at`` the job's last line."""
    data = LOG_START + job_note(1, "stop file: ctrl-c: stopping the job; its final push still runs")
    data += job_note(30, "lift sync push (final, try 1/4)")
    for i in range(1, int(minutes // 5) + 1):
        alive, pushing = remote.hms(3 * 3600 + 300 * i), remote.hms(300 * i)
        data += job_note(300 * i + 30, f"alive {alive}: final push, the push has run for {pushing}")
    if done_at is not None and minutes >= done_at:
        data += b"@@LIFT_DONE " + json.dumps(status, sort_keys=True).encode() + b"\n"
    return data


def stop_and_push_slowly(fake: FakeColab, ctx: remote.Context, **log) -> dict:
    """After the stop request, every download returns ``pushing_log`` for the minutes since then. Returns
    a dict that gets "at": the stop request's time."""
    asked: dict = {}

    def on_upload(target):
        if target == STOP_FILE:
            asked["at"] = ctx.clock.now

    def on_download(n):
        if "at" in asked:
            fake.logs = [pushing_log((ctx.clock.now - asked["at"]) / 60, **log)]

    fake.on_upload, fake.on_download = on_upload, on_download
    return asked


# Review 2, R16: after Ctrl+C lift remote stopped the VM after STOP_WAIT (15 minutes), while job.py's own
# final push (it kills a stalled try at 30 minutes, then tries again) was still running.
def test_ctrl_c_waits_while_the_log_shows_the_final_push_running(tmp_path):
    stopped = {**OK, "rc": 130, "failed_line": 1, "stopped": "stop file: ctrl-c", "installed": True}
    fake, ctx = setup(tmp_path, [LOG_START])
    ctx.sleep = interrupting_sleep(ctx, 3)
    asked = stop_and_push_slowly(fake, ctx, done_at=31, status=stopped)
    rc = remote.launch(remote.Launch(gpu="A100", job="lift sync pull; lift run-dev --variant stock",
                                     name="lift-pw"), ctx)
    assert rc == 130 and fake.server == {} and fake.rescues == []
    assert ctx.clock.now - asked["at"] >= 31 * 60  # the VM was stopped only after the job's last line
    assert manifest("lift-pw")["job"] == stopped and "may be lost" not in ctx.out.getvalue()


def test_the_wait_for_a_running_push_has_a_bound(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START])
    ctx.sleep = interrupting_sleep(ctx, 3)
    asked = stop_and_push_slowly(fake, ctx)  # the push never ends
    rc = remote.launch(remote.Launch(gpu="A100", job="lift run-dev --variant stock", name="lift-pb"), ctx)
    assert remote.STOP_WAIT < ctx.clock.now - asked["at"] <= remote.PUSH_WAIT + 60
    assert rc == 130 and fake.server == {} and manifest("lift-pb")["job"] is None
    assert "may be lost" in ctx.out.getvalue()


def deadline_log(t0: float, now: float, done_at: float, status: dict) -> bytes:
    """The job log at ``now`` under a 1-hour cap from ``t0``: a line runs; the job stops itself at its
    own deadline (t0 + 2700) and runs its final push, a heartbeat every 5 minutes, until ``done_at``."""
    data = LOG_START + job_note(5, "$ lift run-dev --variant stock   (line 1/1)")
    for t in range(300, 2700, 300):
        if t0 + t <= now:
            data += job_note(t, f"alive {remote.hms(t)}: line 1/1, 0:00:10 since its last output")
    if t0 + 2700 <= now:
        data += job_note(2700, "deadline reached: stopping the job; its final push still runs")
        data += job_note(2730, "lift sync push (final, try 1/4)")
    for t in range(3000, int(done_at - t0) + 1, 300):
        if t0 + t <= now:
            pushing = remote.hms(t - 2730)
            data += job_note(t, f"alive {remote.hms(t)}: final push, the push has run for {pushing}")
    if now >= done_at:
        data += b"@@LIFT_DONE " + json.dumps(status, sort_keys=True).encode() + b"\n"
    return data


def failing_stop_uploads(fake: FakeColab, ctx: remote.Context, fail: int, then=None) -> list[float]:
    """The VM's proxy fails the first ``fail`` stop-file uploads (all when ``fail`` is None); the next
    one goes up and calls ``then``. Returns the time of every stop-file try."""
    tries: list[float] = []
    upload = fake.do_upload

    def do_upload(args, config):
        if args[4] != STOP_FILE:
            return upload(args, config)
        tries.append(ctx.clock())
        if fail is None or len(tries) <= fail:
            return remote.Result(1, "[colab] Upload failed: 502 Server Error for url: https://vm.invalid/x")
        if then:
            then()
        return upload(args, config)

    fake.do_upload = do_upload
    return tries


# Review 4, W5: a stop request whose upload failed twice in a row (no wait between the tries) skipped the
# whole wait and stopped the VM at once, even at --max-hours, where the job had stopped itself at its
# own deadline and was running its final push.
def test_a_stop_request_that_does_not_go_up_never_cuts_the_wait(tmp_path):
    stopped = {**OK, "rc": 130, "failed_line": 0, "stopped": "deadline reached", "installed": True, "ran": 1}
    fake, ctx = setup(tmp_path, [LOG_START])
    t0 = ctx.clock()
    done_at = t0 + 3600 + 300  # the job's final push ends 5 minutes after the cap
    tries = failing_stop_uploads(fake, ctx, None)
    fake.on_download = lambda n: setattr(fake, "logs", [deadline_log(t0, ctx.clock(), done_at, stopped)])
    opts = remote.Launch(gpu="A100", job="lift run-dev --variant stock", name="lift-sf", max_hours=1.0,
                         poll=60)
    assert remote.launch(opts, ctx) == 1
    assert ctx.clock() >= done_at and fake.server == {}  # the VM was stopped after the job's last line
    record = manifest("lift-sf")
    assert record["job"] == stopped and record["outcome"] == "max-hours" and record["rc"] == 1
    # The first tries 15 and 30 s apart, then one every STOP_RETRY_EVERY while lift remote waited.
    gaps = [round(b - a) for a, b in zip(tries, tries[1:], strict=False)]
    assert len(tries) > remote.STOP_TRIES and gaps[:2] == [15, 30]
    assert all(gap >= remote.STOP_RETRY_EVERY for gap in gaps[2:])
    text = ctx.out.getvalue()
    assert "the VM is stopped now" not in text and "may be lost" not in text


def test_ctrl_c_tries_the_stop_request_again_while_it_waits(tmp_path):
    stopped = {**OK, "rc": 130, "failed_line": 1, "stopped": "stop file: ctrl-c", "installed": True}
    running = LOG_START + job_note(5, "$ lift run-dev --variant stock   (line 2/2)")
    fake, ctx = setup(tmp_path, [running])
    ctx.sleep = interrupting_sleep(ctx, 3)
    answered = [done_log(stopped, running[len(LOG_START):])]
    tries = failing_stop_uploads(fake, ctx, remote.STOP_TRIES, then=lambda: setattr(fake, "logs", answered))
    job_text = "lift sync pull; lift run-dev --variant stock"
    assert remote.launch(remote.Launch(gpu="A100", job=job_text, name="lift-sr"), ctx) == 130
    assert len(tries) == remote.STOP_TRIES + 1 and tries[-1] - tries[-2] >= remote.STOP_RETRY_EVERY
    assert manifest("lift-sr")["job"] == stopped and fake.server == {}
    text = ctx.out.getvalue()
    assert "the stop request went up" in text and "are lost" not in text and "may be lost" not in text
    # A second Ctrl+C, here in the wait before the stop request's second try, stops the VM at once.
    fake, ctx = setup(tmp_path / "twice", [running])
    ctx.sleep = interrupting_sleep(ctx, 3, 4)
    tries = failing_stop_uploads(fake, ctx, None)
    assert remote.launch(remote.Launch(gpu="A100", job=job_text, name="lift-sr2"), ctx) == 130
    assert len(tries) == 1 and fake.calls[-3:] == ["upload", "stop", "sessions"] and fake.server == {}
    text = ctx.out.getvalue()
    assert "since the last successful push are lost" in text


def retry_pushes_at(fake: FakeColab, n: int) -> None:
    """The push retry's last line (push rc 0) shows from the n-th download of the log on."""
    fake.rescue_rc = None
    fake.on_download = lambda count: setattr(fake, "rescue_rc", 0) if count >= n else None


# Review 5, X3: a first Ctrl+C during the push retry from here, or during the wait for the job's final push
# at --max-hours or after an error, stopped the VM at once and the push with it, although lift remote had
# said that Ctrl+C asks the job to stop and push and that only a second one stops the VM at once.
def test_a_first_ctrl_c_during_the_push_retry_lets_it_go_on(tmp_path):
    failed = done_log({**PUSH_FAILED, "ran": 1})
    fake, ctx = setup(tmp_path, [failed])
    fake.handle.rc = 0
    retry_pushes_at(fake, 3)
    ctx.sleep = ctrl_c_sleep(ctx, 2)  # sleep 1: the job's poll; sleep 2: the push retry's first poll
    before = signal.getsignal(signal.SIGINT)
    opts = remote.Launch(gpu="A100", job="lift run-dev --variant stock", name="lift-x3r")
    assert remote.launch(opts, ctx) == 0
    text = ctx.out.getvalue()
    assert len(fake.rescues) == 1 and fake.server == {}
    assert manifest("lift-x3r")["rescue_push"]["push_rc"] == 0
    assert "the push retry goes on" in text and "Ctrl+C again stops the VM at once" in text
    assert "ARE LOST" not in text and signal.getsignal(signal.SIGINT) is before
    # The second Ctrl+C stops the VM at once.
    fake, ctx = setup(tmp_path / "twice", [failed])
    fake.handle.rc = 0
    retry_pushes_at(fake, 4)
    ctx.sleep = ctrl_c_sleep(ctx, 2, 3)
    assert remote.launch(replace_opts(opts, name="lift-x3s"), ctx) == 5
    text = ctx.out.getvalue()
    assert fake.server == {} and "stopped at once" in text and "ARE LOST" in text
    assert signal.getsignal(signal.SIGINT) is before


def test_a_first_ctrl_c_in_the_wait_at_max_hours_or_after_an_error_lets_the_final_push_end(tmp_path):
    stopped = {**OK, "rc": 1, "failed_line": 0, "stopped": "deadline reached", "installed": True, "ran": 1}
    before = signal.getsignal(signal.SIGINT)
    for name, presses in (("lift-x3m", (62,)), ("lift-x3n", (62, 64))):
        fake, ctx = setup(tmp_path / name, [b""])
        t0 = ctx.clock()
        done_at = t0 + 3600 + 600  # stopped at its own deadline, its final push ends 10 minutes past the cap
        fake.on_download = lambda n, fake=fake, ctx=ctx, t0=t0, done_at=done_at: setattr(
            fake, "logs", [deadline_log(t0, ctx.clock(), done_at, stopped)])
        ctx.sleep = ctrl_c_sleep(ctx, *presses)  # 60 polls to the cap, then the wind-down's polls
        opts = remote.Launch(gpu="A100", job="lift run-dev --variant stock", name=name, max_hours=1.0,
                             poll=60)
        rc = remote.launch(opts, ctx)
        text = ctx.out.getvalue()
        assert fake.server == {} and signal.getsignal(signal.SIGINT) is before
        if len(presses) == 1:
            assert rc == 1 and ctx.clock() >= done_at and manifest(name)["job"] == stopped
            assert manifest(name)["outcome"] == "max-hours" and "are lost" not in text
            assert "goes on" in text and "Ctrl+C again stops the VM at once" in text
        else:
            assert rc == 130 and ctx.clock() < done_at and "since the last successful push are lost" in text
    # After an unexpected error the job is asked to stop and push the same way; a first Ctrl+C lets it.
    answered = {**OK, "rc": 130, "failed_line": 1, "stopped": "stop file: error", "installed": True}

    def explode(n):
        if n == 2:
            raise RuntimeError("a bug while reading the log")

    fake, ctx = setup(tmp_path / "error", [None, LOG_START], on_download=explode)
    fake.on_upload = answer_the_stop_file(fake, answered)
    ctx.sleep = ctrl_c_sleep(ctx, 3)  # the wind-down's first poll
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-x3e"), ctx) == 1
    record = manifest("lift-x3e")
    assert record["job"] == answered and record["outcome"].startswith("error: RuntimeError")
    assert "goes on" in ctx.out.getvalue() and fake.server == {}


def test_lift_remote_knows_job_py_push_schedule(job):
    assert remote.JOB_PUSH_LIMIT == job.PUSH_LIMIT_SECONDS and remote.JOB_KILL_GRACE == job.KILL_GRACE_SECONDS
    assert remote.JOB_PUSH_DELAYS == job.PUSH_DELAYS
    one_stalled_try = job.PUSH_LIMIT_SECONDS + job.KILL_GRACE_SECONDS + sum(job.PUSH_DELAYS)
    assert remote.STOP_WAIT < one_stalled_try and one_stalled_try + 600 <= remote.PUSH_WAIT == 3300
    # The longest wait after the cap: a stop request that went up only at the end of STOP_WAIT, then
    # PUSH_WAIT from there (review 4, W5). The exec client must outlast it.
    assert remote.exec_timeout(12) > 12 * 3600 + remote.STOP_WAIT + remote.PUSH_WAIT


def test_job_pushing_reads_what_job_py_logs(job, tmp_path):
    root = tmp_path / "content"
    seen = []

    def heartbeat_logged():
        log = log_bytes(root)
        if b"the push has run for" in log:
            seen.append(log)
        return bool(seen)

    vm = FakeVM(root, procs={"lift sync push": [WaitingProc(b"pushed\n", heartbeat_logged)]})
    status, _, _ = run_job(job, root, ["lift run-dev --variant stock"], vm=vm, heartbeat=0.02)
    assert status["push_rc"] == 0 and seen and remote.job_pushing(seen[0])
    log = log_bytes(root)
    assert not remote.job_pushing(log)  # the job's last line ends it
    assert remote.job_lines_ran(log)
    assert not remote.job_pushing(log[:log.index(b"line 1/1 ok")])
    assert not remote.job_lines_ran(LOG_START + job_note(5, "$ lift sync pull   (line 1/2)"))


# Review 2, R16: a job that ended without its last line got no push retry, although its log showed
# lines had run.
def test_a_job_that_ended_without_its_last_line_gets_a_push_retry(tmp_path):
    ran = LOG_START + job_note(5, "$ lift run-dev --variant stock   (line 1/1)")
    fake, ctx = setup(tmp_path, [ran], rescue_rc=0)
    fake.handle.rc = 0  # the job cell ended (job.py crashed) without its last line
    opts = remote.Launch(gpu="A100", job="lift run-dev --variant stock", name="lift-nm")
    assert remote.launch(opts, ctx) == 1
    assert len(fake.rescues) == 1 and manifest("lift-nm")["rescue_push"]["push_rc"] == 0
    text = ctx.out.getvalue()
    assert "may be lost" not in text and "the push retry from here pushed" in text
    # Only lift sync lines ran: nothing new is on the VM, so no retry.
    synced = LOG_START + job_note(5, "$ lift sync pull   (line 1/2)")
    fake, ctx = setup(tmp_path / "sync-only", [synced], rescue_rc=0)
    fake.handle.rc = 0
    opts = remote.Launch(gpu="A100", job="lift sync pull; lift info", name="lift-nm2")
    assert remote.launch(opts, ctx) == 1
    assert fake.rescues == []


# Review 2, R19: with the VM's disk full, job.py's last line can reach the laptop only through the exec
# client's output.
def test_the_last_line_counts_when_it_came_only_through_the_exec_output(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START])

    def on_download(n):
        if n == 3:
            with open(remote.exec_log_path("lift-xo"), "ab") as f:
                f.write(b"[lift-job] 10:00:09 the job log cannot be written (OSError): ...\n")
                f.write(b"@@LIFT_DONE " + json.dumps(OK, sort_keys=True).encode() + b"\n")

    fake.on_download = on_download
    remote.exec_log_path("lift-xo").parent.mkdir(parents=True, exist_ok=True)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-xo"), ctx) == 0
    assert manifest("lift-xo")["job"] == OK and manifest("lift-xo")["outcome"] == "done"


# Review 2, R20: `watch --no-stop` called the outputs on a VM it leaves running lost, and exited 5.
def test_watch_no_stop_never_calls_outputs_on_the_running_vm_lost(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)])
    started_vm(fake, ctx, "lift-ns")
    assert remote.watch("lift-ns", stop=False, ctx=ctx) == 6
    text = ctx.out.getvalue()
    assert fake.server and fake.rescues == [] and "ARE LOST" not in text
    assert "still on" in text and "lift remote watch lift-ns" in text
    assert manifest("lift-ns")["rc"] == 6
    # It says what that retry needs in its shell (review 4, W1).
    assert "HF_TOKEN" in text and "LIFT_RUNS_REPO" in text


# Review 4, W1: watch from a new shell (README step 6 sets the secrets per PowerShell session) found the
# job's final push failed, said to set the secrets "first", then released the VM holding the outputs.
# It also counted GITHUB_TOKEN as needed, though the push retry never fetches.
@pytest.mark.parametrize("environ, cached, lacking", [
    ({}, HF, "LIFT_RUNS_REPO"),                  # a fresh shell; HF_TOKEN from `hf auth login`
    ({"LIFT_RUNS_REPO": RUNS}, None, "HF_TOKEN"),
], ids=["fresh-shell", "no-hf-token"])
def test_watch_keeps_the_vm_when_this_shell_cannot_retry_the_push(tmp_path, environ, cached, lacking):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], environ=environ, rescue_rc=0,
                      hf_token=lambda: cached)
    started_vm(fake, ctx, "lift-ws")
    assert remote.watch("lift-ws", ctx=ctx) == 6
    text = ctx.out.getvalue()
    assert fake.server and fake.rescues == [] and "stop" not in fake.calls and "ARE LOST" not in text
    assert f"missing {lacking}:" in text and "GITHUB_TOKEN" not in text
    assert text.index("a push retry") < text.index("job finished")  # said when watch starts
    assert "still on the VM" in text and "lift remote stop lift-ws" in text
    assert "lift remote watch lift-ws retries the push" in text
    record = manifest("lift-ws")
    assert record["rc"] == 6 and record.get("vm_released") is not True and record["job"] == PUSH_FAILED
    # From a shell with HF_TOKEN and LIFT_RUNS_REPO (no GITHUB_TOKEN: push mode never fetches) it retries.
    ctx.environ = {"HF_TOKEN": HF, "LIFT_RUNS_REPO": RUNS}
    ctx.out = io.StringIO()
    assert remote.watch("lift-ws", ctx=ctx) == 0
    assert len(fake.rescues) == 1 and fake.server == {} and manifest("lift-ws")["rescue_push"]["push_rc"] == 0
    sent = [json.loads(u["data"]) for u in fake.uploads if u["remote"] == "content/.lift_secrets.json"]
    assert sent == [{"HF_TOKEN": HF, "LIFT_RUNS_REPO": RUNS}]


def test_watch_keeps_the_vm_for_an_unreported_end_this_shell_cannot_push(tmp_path):
    ran = LOG_START + job_note(5, "$ lift run-dev --variant stock   (line 1/1)")
    fake, ctx = setup(tmp_path, [ran], environ={}, rescue_rc=0)
    started_vm(fake, ctx, "lift-wn")
    assert remote.watch("lift-wn", ctx=ctx) == 6  # the log stops growing: stale, after lines ran
    text = ctx.out.getvalue()
    assert fake.server and fake.rescues == [] and "stop" not in fake.calls
    assert "may be only on the VM" in text and "missing LIFT_RUNS_REPO, HF_TOKEN:" in text
    record = manifest("lift-wn")
    assert record["outcome"] == "stale" and record["rc"] == 6 and record.get("vm_released") is not True


def test_watch_never_retries_a_final_push_that_exited_2(tmp_path):
    refused = {**PUSH_FAILED, "rc": 2, "push_rc": 2, "ran": 1, "sync_refused": "the final push"}
    fake, ctx = setup(tmp_path, [done_log(refused)], rescue_rc=0)
    started_vm(fake, ctx, "lift-w2x")
    assert remote.watch("lift-w2x", ctx=ctx) == 5 and fake.rescues == [] and fake.server == {}
    assert "exited 2" in ctx.out.getvalue() and "so did the retry" not in ctx.out.getvalue()
    # --no-stop: the outputs are still on the VM, and no command would push them.
    fake, ctx = setup(tmp_path / "keep", [done_log(refused)], rescue_rc=0)
    started_vm(fake, ctx, "lift-w2y")
    assert remote.watch("lift-w2y", stop=False, ctx=ctx) == 6 and fake.rescues == [] and fake.server
    text = ctx.out.getvalue()
    assert "still on the VM" in text and "lift remote stop lift-w2y" in text and "ARE LOST" not in text


# Review 5, X1 and contract O: watch retried the push with this shell's token and runs repo unchecked. A
# read token (`hf auth login` caches the one that downloads the models), a revoked one or a typo in
# LIFT_RUNS_REPO made the retry exit 2, and watch then released the VM that held the only copy (exit 5).
def test_watch_checks_the_runs_repo_before_a_push_retry_and_keeps_the_vm_when_it_cannot_push(
        tmp_path, runs_repo_check):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], environ={"LIFT_RUNS_REPO": RUNS}, rescue_rc=0,
                      hf_token=lambda: HF)
    started_vm(fake, ctx, "lift-wt")
    runs_repo_check.errors.append(sync_error("the token cannot write the runs repo (it is a read token): "
                                             "pushing needs a WRITE token for the runs repo", permanent=True))
    assert remote.watch("lift-wt", ctx=ctx) == 6
    text = ctx.out.getvalue()
    assert runs_repo_check.calls == [True] and fake.rescues == [] and fake.server and "stop" not in fake.calls
    assert "read token" in text and "`hf auth login` with a WRITE token" in text and "ARE LOST" not in text
    assert "lift remote watch lift-wt retries the push" in text and "lift remote stop lift-wt" in text
    record = manifest("lift-wt")
    assert record["rc"] == 6 and record.get("vm_released") is not True and record.get("rescue_push") is None
    # A transient error keeps it too: nothing is lost by trying again later.
    ctx.out = io.StringIO()
    runs_repo_check.errors.append(sync_error("could not reach the Hub (ConnectionError)", permanent=False))
    assert remote.watch("lift-wt", ctx=ctx) == 6 and fake.rescues == [] and fake.server
    assert "try again" in ctx.out.getvalue()
    # The check passes, but the retry's push exits 2 (what the check cannot see): the VM is kept, and the
    # retry is not recorded as tried, so the next watch tries again.
    ctx.out = io.StringIO()
    fake.rescue_rc = 2
    assert remote.watch("lift-wt", ctx=ctx) == 6
    assert len(fake.rescues) == 1 and fake.server and "stop" not in fake.calls
    assert "exited 2" in ctx.out.getvalue() and manifest("lift-wt").get("rescue_push") is None
    # From a shell whose token can push, the next watch pushes them and releases the VM.
    ctx.out = io.StringIO()
    fake.rescue_rc = 0
    assert remote.watch("lift-wt", ctx=ctx) == 0
    assert len(fake.rescues) == 2 and fake.server == {} and manifest("lift-wt")["rescue_push"]["push_rc"] == 0
    assert runs_repo_check.calls == [True] * 4


# Review 6, Y2: watch's runs repo check before a push retry had no bound (huggingface_hub's client sets no
# timeout). A Hub reply that stalled held watch in it for ever while the VM billed: no exit code, no banner,
# and on Windows no Ctrl+C either.
def test_watch_keeps_the_vm_when_the_runs_repo_check_never_answers(tmp_path, monkeypatch):
    from lift import sync

    stalled = threading.Event()  # the check answers only when the test ends
    monkeypatch.setattr(sync, "preflight", lambda *, write=True: stalled.wait(60), raising=False)
    monkeypatch.setattr(remote, "RUNS_REPO_CHECK_WAIT", 0.2)
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], rescue_rc=0)
    started_vm(fake, ctx, "lift-hang")
    result: list[int] = []
    watcher = threading.Thread(target=lambda: result.append(remote.watch("lift-hang", ctx=ctx)), daemon=True)
    try:
        watcher.start()
        watcher.join(10)
        assert not watcher.is_alive(), "watch still waits for the runs repo check"
        text = ctx.out.getvalue()
        assert result == [6] and "KEEPS RUNNING" in text and "no answer within 0.2 s" in text
        assert fake.server and fake.rescues == [] and "upload" not in fake.calls and "stop" not in fake.calls
        record = manifest("lift-hang")
        assert record["rc"] == 6 and record.get("vm_released") is not True and "rescue_push" not in record
        # launch has the same bound: no answer is a transient error (1), before any VM exists.
        fake.calls.clear()
        with pytest.raises(remote.RemoteError, match="no answer within") as err:
            remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-hang2"), ctx)
        assert err.value.code == 1 and fake.calls == []
    finally:
        stalled.set()
        watcher.join(10)


# Review 5, X13: watch that keeps the VM on purpose, billing, with outputs no push sent, exited 1 (also the
# code of a failed line whose VM is released) with no banner, so a caller that looks for 4 left it billing.
# 4's advice (lift remote stop) would throw the outputs away: the keep has a code of its own.
def test_watch_that_keeps_the_vm_for_its_outputs_exits_6_with_a_banner(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], environ={}, hf_token=lambda: HF)
    started_vm(fake, ctx, "lift-kb")
    assert remote.watch("lift-kb", ctx=ctx) == 6 == remote.EXIT_KEPT
    text = ctx.out.getvalue()
    assert fake.server and "!" * 78 in text and "KEEPS RUNNING" in text and "MAY STILL BE RUNNING" not in text
    assert manifest("lift-kb")["rc"] == 6
    # lift remote stop then gives the run its own code: the outputs went with the VM (5).
    ctx.out = io.StringIO()
    assert remote.stop("lift-kb", ctx=ctx) == 0 and fake.server == {}
    assert manifest("lift-kb")["rc"] == 5 and "ARE LOST" in ctx.out.getvalue()
    # A job that never reported its end after running lines, kept for the same reason: 6, then 1 once stopped.
    ran = LOG_START + job_note(5, "$ lift run-dev --variant stock   (line 1/1)")
    fake, ctx = setup(tmp_path / "unreported", [ran], environ={})
    started_vm(fake, ctx, "lift-ku")
    assert remote.watch("lift-ku", ctx=ctx) == 6 and fake.server and "KEEPS RUNNING" in ctx.out.getvalue()
    assert remote.stop("lift-ku", ctx=ctx) == 0 and manifest("lift-ku")["rc"] == 1
    # The README's advice for scripts and Claude Code names the code.
    advice = " ".join(readme().split("**From Claude Code.**", 1)[1].split("\n\n", 1)[0].split())
    assert "a 4 means" in advice and "a 6 means" in advice and "lift remote watch" in advice


# Review 5, X14: watch kept a VM that the Colab CLI reports lost, for a push retry that no upload or exec can
# reach, said its outputs were still on it, and never ran release()'s check and stop by endpoint.
@pytest.mark.parametrize("environ", [{}, SECRETS], ids=["no-secrets", "secrets"])
@pytest.mark.parametrize("gone", ["reclaimed", "pruned"])
def test_watch_never_keeps_or_retries_a_vm_the_colab_cli_has_lost(tmp_path, gone, environ, runs_repo_check):
    name = "lift-gone"
    fake, ctx = setup(tmp_path, environ=dict(environ), rescue_rc=0, hf_token=lambda: HF)
    started_vm(fake, ctx, name, job=PUSH_FAILED, outcome="done", rc=6)  # kept earlier for a push retry
    if gone == "reclaimed":
        fake.server.clear()
        fake.logs = [remote.Result(1, f"[colab] Session '{name}' appears to be lost (404/401).\n")]
    else:  # the CLI pruned its record; the server still runs the VM
        fake.save({})
        fake.logs = [remote.Result(1, f"[colab] Session '{name}' not found.\n")]
    assert remote.watch(name, ctx=ctx) == 5
    text = ctx.out.getvalue()
    assert fake.server == {} and fake.rescues == [] and "upload" not in fake.calls
    assert runs_repo_check.calls == []
    assert "no push retry can reach it" in text and "ARE LOST" in text
    assert "KEEPS RUNNING" not in text and "still on the VM" not in text
    record = manifest(name)
    assert record["rc"] == 5 and record["vm_released"] is True
    if gone == "pruned":  # release() found it by its endpoint, and stopped it there
        assert fake.config_files and fake.calls[-2:] == ["stop", "sessions"]


# Review 2, R31: watch replayed exit 4 after `lift remote stop` had confirmed the release.
def test_watch_after_a_confirmed_stop_returns_the_runs_own_code(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)], stop_rc=1)
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-w4"), ctx) == 4
    fake.stop_rc = 0
    assert remote.stop("lift-w4", ctx=ctx) == 0 and fake.server == {}
    assert manifest("lift-w4")["rc"] == 0
    assert remote.watch("lift-w4", ctx=ctx) == 0
    # A refusal's own code comes back too: here the secrets never went up (3).
    fake, ctx = setup(tmp_path / "upload", upload_rc=1, stop_rc=1)
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-w5"), ctx) == 4
    fake.stop_rc = 0
    assert remote.stop("lift-w5", ctx=ctx) == 0 and manifest("lift-w5")["rc"] == 3


def cannot_start(argv, *, log_path):
    """FakeColab.spawn while colab.exe is being replaced (`uv tool install ... --force`)."""
    raise FileNotFoundError(2, "The system cannot find the file specified", argv[0])


# Review 3, V3: an unexpected error in watch escaped before the release: no `colab stop`, no alarm, a
# traceback (exit 1), and the VM billing on.
def test_watch_releases_the_vm_when_the_push_retry_cannot_start(tmp_path, monkeypatch):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], rescue_rc=0)
    started_vm(fake, ctx, "lift-wx")
    fake.spawn = cannot_start
    monkeypatch.setattr(remote, "Context", lambda *args, **kwargs: ctx)
    assert main(["remote", "watch", "lift-wx"]) == 5
    assert fake.server == {} and fake.calls[-2:] == ["stop", "sessions"]
    record = manifest("lift-wx")
    assert record["rc"] == 5 and record["vm_released"] is True
    assert record["rescue_push"]["push_rc"] is None and "FileNotFoundError" in record["rescue_push"]["error"]
    text = ctx.out.getvalue()
    assert "push retry cannot start" in text and "ARE LOST" in text and "so did the retry" in text


def test_watch_asks_the_job_to_stop_and_push_after_an_unexpected_error(tmp_path):
    stopped = {**OK, "rc": 130, "failed_line": 1, "stopped": "stop file: error", "installed": True}
    fake, ctx = setup(tmp_path, [LOG_START])
    started_vm(fake, ctx, "lift-we")
    fake.on_upload = answer_the_stop_file(fake, stopped)

    def locked(n):  # e.g. a virus scanner holding the fresh copy of the log
        if n == 2:
            raise PermissionError(13, "The process cannot access the file", "remote-lift-we.log")

    fake.on_download = locked
    assert remote.watch("lift-we", ctx=ctx) == 1
    assert [u["remote"] for u in fake.uploads] == [STOP_FILE] and fake.uploads[0]["data"] == b"error\n"
    assert fake.server == {} and fake.calls[-2:] == ["stop", "sessions"]
    record = manifest("lift-we")
    assert record["outcome"].startswith("error: PermissionError") and record["job"] == stopped
    assert record["rc"] == 1 and record["vm_released"] is True


def test_watch_after_an_unexpected_error_still_exits_4_when_the_vm_stays(tmp_path, monkeypatch):
    fake, ctx = setup(tmp_path, [LOG_START], stop_rc=1)
    started_vm(fake, ctx, "lift-w4x")

    def broken(n):  # every download from the second on: the wait for the stopped job fails too
        if n >= 2:
            raise RuntimeError("a bug while reading the log")

    fake.on_download = broken
    monkeypatch.setattr(remote, "Context", lambda *args, **kwargs: ctx)
    assert main(["remote", "watch", "lift-w4x"]) == 4
    assert fake.server and "MAY STILL BE RUNNING" in ctx.out.getvalue()
    record = manifest("lift-w4x")
    assert record["rc"] == 4 and record["outcome"].startswith("error: RuntimeError")
    assert record["vm_released"] is False
    # --no-stop: the error ends the watching only; the VM and the recorded job status stay.
    fake, ctx = setup(tmp_path / "keep", [LOG_START], on_download=broken)
    started_vm(fake, ctx, "lift-w4k", job=OK, outcome="done")
    assert remote.watch("lift-w4k", stop=False, ctx=ctx) == 1
    assert fake.server and STOP_FILE not in [u["remote"] for u in fake.uploads]
    record = manifest("lift-w4k")
    assert record["job"] == OK and record["rc"] == 1 and record["outcome"].startswith("error: RuntimeError")


# Review 3, V12: `lift remote stop` after `watch --no-stop` released the outputs still on the VM with no
# alarm, and the record kept saying 1.
def test_stop_after_watch_no_stop_reports_the_outputs_it_releases_as_lost(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], rescue_rc=0)
    started_vm(fake, ctx, "lift-nss")
    assert remote.watch("lift-nss", stop=False, ctx=ctx) == 6 and fake.server
    assert "lift remote stop lift-nss releases the VM without that retry" in ctx.out.getvalue()
    ctx.out = io.StringIO()
    assert remote.stop("lift-nss", ctx=ctx) == 0 and fake.server == {} and fake.rescues == []
    text = ctx.out.getvalue()
    assert "ARE LOST" in text and "no retry from here pushed them" in text
    assert manifest("lift-nss")["rc"] == 5
    ctx.out = io.StringIO()
    assert remote.watch("lift-nss", ctx=ctx) == 5
    # A run whose outputs were pushed keeps its code, and no alarm.
    fake, ctx = setup(tmp_path / "ok", [done_log(OK)])
    started_vm(fake, ctx, "lift-nso")
    assert remote.watch("lift-nso", stop=False, ctx=ctx) == 0
    ctx.out = io.StringIO()
    assert remote.stop("lift-nso", ctx=ctx) == 0 and manifest("lift-nso")["rc"] == 0
    assert "ARE LOST" not in ctx.out.getvalue()


def test_parse_job():
    text = "lift sync pull; lift smoke\n# a comment\n\nlift run-dev --variant stock;"
    assert remote.parse_job(text) == ["lift sync pull", "lift smoke", "lift run-dev --variant stock"]
    assert remote.parse_job('lift judge --keys "a;b c"') == ["lift judge --keys 'a;b c'"]
    for bad in ("", "  ", "python -m lift info", "lift", "lift info; rm -rf /", "lift remote --gpu T4",
                'lift info "unclosed'):
        with pytest.raises(remote.RemoteError):
            remote.parse_job(bad)


def test_find_status_needs_a_whole_line_at_a_line_start():
    assert remote.find_status(b"x\n@@LIFT_DONE {\"rc\": 4}\n") == {"rc": 4}
    assert remote.find_status(b"@@LIFT_DONE {\"rc\": 0}") == {"rc": 0}
    assert remote.find_status(b"x\n@@LIFT_DONE {\"rc\": 0") is None
    assert remote.find_status(b"echo @@LIFT_DONE {\"rc\": 0}\n") is None
    # A push retry's line never stands for the job's.
    both = b"@@LIFT_DONE {\"rc\": 1}\n" + push_marker("ab12cd34", 0)
    assert remote.find_status(both) == {"rc": 1}
    assert remote.find_status(both, accept=lambda s: s.get("attempt") == "ab12cd34")["push_rc"] == 0


def test_exit_codes():
    code = remote.exit_code
    assert code({"rc": 0, "push_rc": 0}) == 0 and code(None) == 1
    assert code({"rc": 300, "failed_line": 0, "push_rc": 0}) == 1 and code({"rc": True, "push_rc": 0}) == 1
    assert code({"rc": 0, "push_rc": 3}) == 0  # conflicts: pushed, the copies kept under conflicts/
    assert code({"rc": 0, "push_rc": 0}, released=None) == 4
    assert code({"rc": 0, "push_rc": 0}, released=False) == 4
    assert code(PUSH_FAILED) == 5 and code(PUSH_FAILED, rescue={"push_rc": 0}) == 0
    assert code(PUSH_FAILED, outcome="interrupted") == 5 and code(None, outcome="interrupted") == 130
    assert code({"rc": 3, "failed_step": "python", "installed": False}) == 3
    assert code(None, outcome="colab new failed") == 3 and code(PUSH_FAILED, released=False) == 4


def started_vm(fake: FakeColab, ctx: remote.Context, name: str, **fields) -> None:
    """A VM that an earlier lift remote started: the CLI's record, the server's VM, our manifest."""
    fake.do_new(["new", "-s", name], None)
    record = {"name": name, "endpoint": f"gpu-a100-s-{name}", "started_epoch": ctx.clock(), "max_hours": 12,
              "repo": "Knull/lost-in-finetuning", "ref": "main"}
    remote.write_manifest(remote.manifest_path(name), {**record, **fields})


def test_watch_reattaches_and_stops_the_vm(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START, done_log(OK)])
    started_vm(fake, ctx, "lift-old")
    assert remote.watch("lift-old", ctx=ctx) == 0
    assert fake.calls == ["download", "download", "stop", "sessions"]
    record = manifest("lift-old")
    assert record["vm_released"] is True and record["rc"] == 0 and record["outcome"] == "done"


def test_watch_no_stop_leaves_the_vm(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)])
    started_vm(fake, ctx, "lift-keep")
    assert remote.watch("lift-keep", stop=False, ctx=ctx) == 0
    assert fake.calls == ["download"] and fake.server


def test_watch_retries_a_failed_final_push_before_stopping(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)], rescue_rc=0)
    started_vm(fake, ctx, "lift-wr")
    assert remote.watch("lift-wr", ctx=ctx) == 0
    assert len(fake.rescues) == 1 and fake.calls[-2:] == ["stop", "sessions"]
    assert manifest("lift-wr")["rescue_push"]["push_rc"] == 0


# Review #35: watch past the cap never read the log.
def test_watch_past_the_cap_still_reads_the_log_once(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)])
    started_vm(fake, ctx, "lift-late", started_epoch=ctx.clock() - 13 * 3600)
    assert remote.watch("lift-late", ctx=ctx) == 0
    assert fake.calls == ["download", "stop", "sessions"]
    record = manifest("lift-late")
    assert record["outcome"] == "done" and record["rc"] == 0 and record["job"] == OK


def test_watch_past_the_cap_asks_a_running_job_to_stop(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START])
    started_vm(fake, ctx, "lift-late2", started_epoch=ctx.clock() - 13 * 3600)
    assert remote.watch("lift-late2", ctx=ctx) == 1
    assert fake.calls[:2] == ["download", "upload"] and fake.uploads[0]["remote"] == STOP_FILE
    assert fake.calls[-2:] == ["stop", "sessions"] and manifest("lift-late2")["outcome"] == "max-hours"


# Review #36: watch after the release rewrote a good manifest as "lost" and inflated wall_seconds.
def test_watch_after_the_vm_was_released_only_replays_the_log(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK, b"[lift-job] 10:01:00 line 1/1 ok\n")])
    fake.handle.rc = 0
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-f12"), ctx) == 0
    before = manifest("lift-f12")
    fake.calls.clear()
    fake.logs = [remote.Result(1, "[colab] Session 'lift-f12' not found.\n")]
    ctx.clock.now += 14 * 3600
    ctx.out = io.StringIO()
    assert remote.watch("lift-f12", ctx=ctx, max_hours=1) == 0
    assert manifest("lift-f12") == before and fake.calls == []
    assert "line 1/1 ok" in ctx.out.getvalue()


def test_watch_never_replaces_a_recorded_status_with_a_weaker_one(tmp_path):
    lost = remote.Result(1, "[colab] Session 'lift-w2' appears to be lost (404/401).\n")
    fake, ctx = setup(tmp_path, [lost])
    started = ctx.clock()
    started_vm(fake, ctx, "lift-w2", outcome="done", rc=4, job=OK, vm_released=None, wall_seconds=7200.0)
    ctx.clock.now += 3 * 3600
    assert remote.watch("lift-w2", ctx=ctx) == 0
    record = manifest("lift-w2")
    assert record["outcome"] == "done" and record["job"] == OK and record["rc"] == 0
    # This call released the VM, so the billed time runs to now.
    assert record["vm_released"] is True and record["wall_seconds"] == ctx.clock() - started > 3 * 3600


# Review 4, W12: watch run beside a still-alive launch (the README's after-sleep advice) wrote over the
# launch's record and raised false "may be lost" and "ARE LOST" alarms. One lift remote process at a time
# follows a job now; a second one is refused with the holder's pid.
def test_watch_refuses_while_another_lift_remote_follows_the_job(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)])
    started_vm(fake, ctx, "lift-lk")
    before = manifest("lift-lk")
    with remote.follower_lock("lift-lk", "launch", ctx) as holder:
        assert holder is None
        with pytest.raises(remote.RemoteError) as err:
            remote.watch("lift-lk", ctx=ctx)
    message = str(err.value)
    assert err.value.code == 2 and f"pid {os.getpid()}" in message and "launch" in message
    assert fake.calls == [] and manifest("lift-lk") == before
    assert remote.watch("lift-lk", ctx=ctx) == 0  # once that process is done, watch follows the job


def test_launch_holds_the_jobs_lock_while_it_follows(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START, done_log(OK)])
    seen = []

    def look(n):
        with remote.follower_lock("lift-lh", "watch", ctx) as holder:
            seen.append(holder)

    fake.on_download = look
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-lh"), ctx) == 0
    assert seen and all(h and f"pid {os.getpid()}" in h and "launch" in h for h in seen)
    with remote.follower_lock("lift-lh", "watch", ctx) as holder:
        assert holder is None


LOCK_HOLDER = """\
import os, sys
from lift import remote
ctx = remote.Context(runner=object(), out=sys.stderr, colab_exe="colab", config_dir=remote.Path("."))
with remote.follower_lock(sys.argv[1], "watch", ctx) as holder:
    print("held" if holder is None else "busy", os.getpid(), flush=True)
    sys.stdin.readline()
"""


def test_the_jobs_lock_holds_across_processes_and_ends_with_its_holder(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(OK)])
    started_vm(fake, ctx, "lift-lx")
    child = subprocess.Popen([sys.executable, "-c", LOCK_HOLDER, "lift-lx"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        said, pid = child.stdout.readline().split()  # its own pid: a venv's python.exe is a launcher
        assert said == "held"
        with pytest.raises(remote.RemoteError) as err:
            remote.watch("lift-lx", ctx=ctx)
        assert err.value.code == 2 and f"pid {pid}" in str(err.value) and fake.calls == []
    finally:
        child.kill()  # no cleanup, as when that lift remote crashed
        child.wait(timeout=60)
    deadline = time.monotonic() + 10
    while True:  # the OS ends a dead process's lock (on Windows, within moments)
        with remote.follower_lock("lift-lx", "watch", ctx) as holder:
            if holder is None:
                break
        assert time.monotonic() < deadline, holder
        time.sleep(0.05)
    assert remote.watch("lift-lx", ctx=ctx) == 0


# W12, the record itself: whatever another process recorded meanwhile (a follower that held no lock, or
# `lift remote stop`) is read again before the retry decision and the last write, and kept.
def test_watch_keeps_what_another_process_recorded_meanwhile(tmp_path):
    fake, ctx = setup(tmp_path, [LOG_START + job_note(5, "$ lift smoke   (line 1/1)")])
    started_vm(fake, ctx, "lift-cc")

    def launch_finishes(n):  # between two polls the other follower saw the last line and released the VM
        if n == 2:
            record = remote.load_manifest("lift-cc")
            record.update(outcome="done", rc=0, job=OK, vm_released=True, wall_seconds=125.0)
            remote.write_manifest(remote.manifest_path("lift-cc"), record)
            fake.server.clear()
            fake.save({})
            fake.logs = [remote.Result(1, "[colab] Session 'lift-cc' not found.\n")]

    fake.on_download = launch_finishes
    assert remote.watch("lift-cc", ctx=ctx) == 0
    text = ctx.out.getvalue()
    assert "may be lost" not in text and "stop" not in fake.calls
    record = manifest("lift-cc")
    assert record["outcome"] == "done" and record["job"] == OK and record["rc"] == 0
    assert record["vm_released"] is True and record["wall_seconds"] == 125.0


def test_watch_keeps_a_push_retry_that_another_process_made(tmp_path):
    fake, ctx = setup(tmp_path, [done_log(PUSH_FAILED)])
    started_vm(fake, ctx, "lift-cd")
    theirs = {"mode": "push", "attempt": "launch01", "push_rc": 0}

    def launch_finishes(n):  # while watch's retry waits, the other follower's retry pushed and it released
        if n == 2:
            record = remote.load_manifest("lift-cd")
            record.update(outcome="done", rc=0, job=PUSH_FAILED, vm_released=True, rescue_push=theirs)
            remote.write_manifest(remote.manifest_path("lift-cd"), record)
            fake.server.clear()
            fake.save({})
            fake.logs = [remote.Result(1, "[colab] Session 'lift-cd' not found.\n")]

    fake.on_download = launch_finishes
    assert remote.watch("lift-cd", ctx=ctx) == 0
    text = ctx.out.getvalue()
    assert "ARE LOST" not in text
    record = manifest("lift-cd")
    assert record["rc"] == 0 and record["rescue_push"] == theirs and record["vm_released"] is True
    # The retry decision reads the record again too: a retry that pushed already is not tried again.
    fake, ctx = setup(tmp_path / "before", [LOG_START, done_log(PUSH_FAILED)], rescue_rc=1)
    started_vm(fake, ctx, "lift-ce")

    def pushed_meanwhile(n):
        if n == 2:
            record = remote.load_manifest("lift-ce")
            remote.write_manifest(remote.manifest_path("lift-ce"), {**record, "rescue_push": theirs})

    fake.on_download = pushed_meanwhile
    assert remote.watch("lift-ce", ctx=ctx) == 0
    assert fake.rescues == [] and fake.server == {} and manifest("lift-ce")["rescue_push"] == theirs


def test_stop_by_endpoint_when_the_cli_lost_its_record(tmp_path):
    fake, ctx = setup(tmp_path)
    started_vm(fake, ctx, "lift-orphan")
    fake.save({})  # the CLI pruned its record; the VM still runs
    assert remote.stop("lift-orphan", ctx=ctx) == 0
    assert fake.calls == ["stop", "sessions", "stop", "sessions"]
    assert fake.argvs[2][1:3] == ["--config", fake.config_files[0]]
    assert not Path(fake.config_files[0]).exists() and fake.server == {}
    assert manifest("lift-orphan")["vm_released"] is True


def test_stop_that_cannot_release_raises_the_alarm(tmp_path):
    fake, ctx = setup(tmp_path, stop_rc=1)
    started_vm(fake, ctx, "lift-stuck")
    assert remote.stop("lift-stuck", ctx=ctx) == 4
    assert fake.calls == ["stop", "stop", "stop", "sessions", "stop", "sessions"]
    text = ctx.out.getvalue()
    assert "MAY STILL BE RUNNING" in text and "lift remote stop lift-stuck" in text
    assert manifest("lift-stuck")["vm_released"] is False


# Review #31: without an endpoint, a VM the CLI had forgotten ([?] in `colab sessions`) was called released.
def test_stop_without_a_manifest_never_calls_an_unknown_vm_released(tmp_path):
    fake, ctx = setup(tmp_path)
    fake.do_new(["new", "-s", "lift-orph"], None)
    fake.save({})  # the CLI pruned its record, and the job was launched from another worktree
    assert remote.stop("lift-orph", ctx=ctx) == 4
    text = ctx.out.getvalue()
    assert "VM released" not in text and "MAY STILL BE RUNNING" in text
    assert "lift remote stop lift-orph --endpoint gpu-a100-s-lift-orph" in text
    assert fake.server  # nothing is stopped on a guess
    # The operator names the endpoint the alarm listed.
    ctx.out = io.StringIO()
    assert remote.stop("lift-orph", endpoint="gpu-a100-s-lift-orph", ctx=ctx) == 0 and fake.server == {}
    with pytest.raises(remote.RemoteError, match="endpoint"):
        remote.stop("lift-orph", endpoint="bad endpoint; rm", ctx=ctx)


def test_a_failed_colab_new_records_the_new_unknown_vm_and_raises_the_alarm(tmp_path):
    fake, ctx = setup(tmp_path, new_mode="orphan")
    older = "gpu-a100-s-from-last-week"
    fake.server[older] = "?"  # an orphan from before this launch: listed, but not ours
    rc = remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-n1"), ctx)
    assert rc == 4
    record = manifest("lift-n1")
    assert record["outcome"] == "colab new failed" and record["orphan_endpoints"] == ["gpu-a100-s-lift-n1"]
    assert record["endpoint"] is None and record["vm_released"] is None
    text = ctx.out.getvalue()
    assert "lift remote stop lift-n1 --endpoint gpu-a100-s-lift-n1" in text
    assert f"--endpoint {older}" not in text and "VM released" not in text
    assert "upload" not in fake.calls and "exec" not in fake.calls


def test_a_failed_colab_new_without_a_vm_exits_3(tmp_path):
    fake, ctx = setup(tmp_path)
    fake.do_new = lambda args, config: remote.Result(1, "[colab] Allocation refused (precondition failed).")
    assert remote.launch(remote.Launch(gpu="A100", job="lift info", name="lift-n2"), ctx) == 3
    assert manifest("lift-n2")["orphan_endpoints"] == [] and manifest("lift-n2")["vm_released"] is True


def test_subprocess_runner_contract(tmp_path):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["run"] = (argv, kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="[colab] Session READY.\n")

    def fake_popen(argv, **kwargs):
        seen["popen"] = (argv, kwargs)
        return "handle"

    runner = remote.SubprocessRunner({"PATH": "x", "HF_TOKEN": HF, "GITHUB_TOKEN": GH}, run=fake_run,
                                     popen=fake_popen)
    assert REAL_RUN(runner, [EXE, "sessions"], timeout=120) == remote.Result(0, "[colab] Session READY.\n")
    argv, kwargs = seen["run"]
    assert argv == [EXE, "sessions"] and kwargs["stdin"] is subprocess.DEVNULL and kwargs["timeout"] == 120
    assert kwargs["stderr"] is subprocess.STDOUT and kwargs["encoding"] == "utf-8"
    assert kwargs["env"]["PYTHONUTF8"] == "1" and kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert "HF_TOKEN" not in kwargs["env"] and "GITHUB_TOKEN" not in kwargs["env"]
    assert REAL_SPAWN(runner, [EXE, "exec"], log_path=tmp_path / "logs" / "x.exec.log") == "handle"
    argv, kwargs = seen["popen"]
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["env"]["PYTHONUTF8"] == "1" and (tmp_path / "logs" / "x.exec.log").exists()

    def too_slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    assert REAL_RUN(remote.SubprocessRunner({}, run=too_slow), [EXE, "download"], timeout=5).rc is None


def test_cli_help_works(capsys):
    for argv in (["remote", "--help"], ["remote", "watch", "--help"], ["remote", "stop", "--help"]):
        with pytest.raises(SystemExit) as stop:
            build_parser().parse_args(argv)
        assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "--gpu" in text and "--max-hours" in text and "--no-stop" in text and "--endpoint" in text
    assert "--max-silent-minutes" in text
    # The exit codes are documented where a script author looks.
    for code in ("0 ", "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "130 "):
        assert re.search(rf"^\s+{code}", text, flags=re.MULTILINE), code


def test_cli_rejects_bad_options(capsys):
    assert main(["remote", "--gpu", "a10g", "--job", "lift info"]) == 2
    assert "A100" in capsys.readouterr().err
    assert main(["remote", "--job", "lift info"]) == 2
    assert "--gpu and --job are required" in capsys.readouterr().err


def test_cli_passes_the_new_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(remote, "stop", lambda name, **kwargs: seen.update(stop=(name, kwargs)) or 0)
    monkeypatch.setattr(remote, "launch", lambda opts: seen.update(launch=opts) or 0)
    assert main(["remote", "stop", "lift-orph", "--endpoint", "gpu-a100-s-x"]) == 0
    assert seen["stop"] == ("lift-orph", {"endpoint": "gpu-a100-s-x"})
    assert main(["remote", "--gpu", "A100", "--job", "lift info", "--max-silent-minutes", "90"]) == 0
    assert seen["launch"].max_silent_minutes == 90


def test_cli_dry_run(monkeypatch, tmp_path, capsys):
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "token.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(remote, "colab_config_dir", lambda: config_dir)
    monkeypatch.setattr(remote, "find_colab", lambda: EXE)
    for var, value in SECRETS.items():
        monkeypatch.setenv(var, value)
    argv = ["remote", "--gpu", "l4", "--job", "lift sync pull; lift info", "--dry-run", "--name", "lift-cli"]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "new -s lift-cli --gpu L4" in text and not any(secret in text for secret in SECRETS.values())


def test_dry_run_masks_secrets_and_calls_nothing(tmp_path):
    fake, ctx = setup(tmp_path)
    opts = remote.Launch(gpu="T4", job="lift sync pull; lift info", name="lift-dry", dry_run=True)
    assert remote.launch(opts, ctx) == 0
    assert fake.calls == [] and not remote.manifest_path("lift-dry").exists()
    text = ctx.out.getvalue()
    assert not any(secret in text for secret in SECRETS.values())
    assert "GITHUB_TOKEN=***" in text and "HF_TOKEN=***" in text and "<temp secrets file>" in text
    assert "new -s lift-dry --gpu T4" in text and "LIFT_JOB_B64=" in text
    _, ctx = setup(tmp_path, logged_in=False)
    (tmp_path / "colab-cli" / "token.json").unlink()
    assert remote.launch(opts, ctx) == 2 and "not logged in" in ctx.out.getvalue()


def test_sigint_is_held_off_only_during_cleanup():
    before = signal.getsignal(signal.SIGINT)
    with remote._sigint_ignored():
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) is before


def test_threads_do_not_touch_signals():
    errors = []

    def body():
        try:
            with remote._sigint_ignored():
                pass
        except Exception as e:  # signal.signal raises ValueError outside the main thread
            errors.append(e)

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    assert errors == []


# --- the notebook fallback in colab/README.md -----------------------------------------------------


def notebook_cell(marker: str) -> str:
    cells = re.findall(r"```python\n(.*?)```", README_PATH.read_text(encoding="utf-8"), flags=re.S)
    (cell,) = [cell for cell in cells if marker in cell]
    return cell


def run_notebook_job(monkeypatch, tmp_path, lines, rcs=None, ran=None):
    """Run the README's job cell with a fake subprocess.run; returns the lift commands it ran (also
    appended to ``ran``, which survives the cell's final raise)."""
    cell = notebook_cell('JOB = """')
    cell = re.sub(r'JOB = """.*?"""', lambda m: 'JOB = """\n' + "\n".join(lines) + '\n"""', cell, count=1,
                  flags=re.S)
    assert '"/content/lift_job.log"' in cell
    cell = cell.replace('"/content/lift_job.log"', repr(str(tmp_path / "job.log")))
    queues = {key: list(value) for key, value in (rcs or {}).items()}
    env = {"HF_TOKEN": HF, "PATH": "x"}
    ran = [] if ran is None else ran

    def fake_run(argv, **kwargs):
        assert argv[:3] == [sys.executable, "-m", "lift"] and kwargs["env"] is env
        key = " ".join(["lift", *argv[3:]])
        ran.append(key)
        queue = queues.get(key, [0])
        return subprocess.CompletedProcess(argv, queue.pop(0) if len(queue) > 1 else queue[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    scope = {"__name__": "__main__", "REPO": str(tmp_path), "ENV": env}
    exec(compile(cell, "README:job cell", "exec"), scope)
    return ran


NOTEBOOK_LINES = ["lift sync pull", "lift generate --variant stock --split dev",
                  "lift judge --variant stock --split dev"]


# Review #37: the notebook fallback pushed only at the very end, and never failed loudly.
def test_notebook_fallback_checkpoints_like_job_py(monkeypatch, tmp_path):
    lines = NOTEBOOK_LINES
    assert run_notebook_job(monkeypatch, tmp_path, lines) == [
        "lift sync pull", "lift generate --variant stock --split dev", "lift sync push",
        "lift judge --variant stock --split dev", "lift sync push"]
    # A checkpoint push that keeps failing is reported, not fatal; the final push sends everything.
    ran = run_notebook_job(monkeypatch, tmp_path, lines, {"lift sync push": [1, 1, 1, 1, 0]})
    assert ran.count("lift sync push") == 5 and ran[-2:] == ["lift judge --variant stock --split dev",
                                                             "lift sync push"]


def test_notebook_fallback_raises_at_the_end_after_its_final_push(monkeypatch, tmp_path):
    lines = NOTEBOOK_LINES
    with pytest.raises(RuntimeError, match="generate"):
        run_notebook_job(monkeypatch, tmp_path, lines, {"lift generate --variant stock --split dev": [2]})
    with pytest.raises(RuntimeError, match="push"):
        run_notebook_job(monkeypatch, tmp_path, lines[:2], {"lift sync push": [1]})


def test_notebook_setup_cell_keeps_the_secrets_from_git_and_pip(monkeypatch):
    cell = notebook_cell("userdata")

    def get(name):
        if name not in SECRETS:
            raise KeyError(name)
        return SECRETS[name]

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=get)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    scope = {"__name__": "__main__"}
    exec(compile(cell, "README:setup cell", "exec"), scope)
    assert [argv[0] for argv, _ in calls][:4] == ["git"] * 4
    for argv, kwargs in calls:
        assert "env" not in kwargs or not any(value in kwargs["env"].values() for value in SECRETS.values())
        assert not any(secret in arg for arg in argv for secret in (GH, HF, OAI))
    assert all(scope["ENV"][name] == value for name, value in SECRETS.items())
    assert not any(os.environ.get(name) == value for name, value in SECRETS.items())


# Review 2, R32 and contract 1, in the notebook fallback: sync lines run like job.py runs them.
def test_notebook_fallback_runs_sync_lines_like_job_py(monkeypatch, tmp_path):
    lines = ["lift sync pull", "lift generate --variant stock --split dev", "lift sync push",
             "lift judge --variant stock --split dev"]
    # A pull that exits 1 is tried again; a push line is tried again and never stops the job.
    rcs = {"lift sync pull": [1, 0], "lift sync push": [0, 1, 0]}
    ran = run_notebook_job(monkeypatch, tmp_path, lines, rcs)
    assert ran == ["lift sync pull", "lift sync pull", "lift generate --variant stock --split dev",
                   "lift sync push", "lift sync push", "lift sync push",
                   "lift judge --variant stock --split dev", "lift sync push"]
    ran = run_notebook_job(monkeypatch, tmp_path, lines, {"lift sync push": [0, 3, 0]})  # conflicts: reported
    assert "lift judge --variant stock --split dev" in ran
    # A sync call that exits 2 is never tried again and fails the job at once. Only the pull ran, so
    # there is nothing to push.
    ran = []
    with pytest.raises(RuntimeError, match="no retry can fix"):
        run_notebook_job(monkeypatch, tmp_path, lines, {"lift sync pull": [2]}, ran=ran)
    assert ran == ["lift sync pull"]
    ran = []
    with pytest.raises(RuntimeError, match="no retry can fix"):
        run_notebook_job(monkeypatch, tmp_path, lines, {"lift sync push": [2]}, ran=ran)
    # The checkpoint push exits 2: the job ends there, and the final push is tried once.
    assert ran == ["lift sync pull", "lift generate --variant stock --split dev", "lift sync push",
                   "lift sync push"]


# Contract G in the notebook fallback: a generate, judge or run-dev line that exits 3 (done, but some
# cells were skipped) is reported, the next lines run, and the cell raises at the end, as job.py does.
def test_notebook_fallback_goes_on_past_a_line_that_skipped_cells(monkeypatch, tmp_path):
    ran = []
    with pytest.raises(RuntimeError, match="rc 3"):
        run_notebook_job(monkeypatch, tmp_path, SWEEP[:3], {SWEEP[1]: [3]}, ran=ran)
    assert ran == [SWEEP[0], SWEEP[1], "lift sync push", SWEEP[2], "lift sync push"]
    ran = []  # exit 3 of another command still ends the job
    with pytest.raises(RuntimeError, match="rc 3"):
        run_notebook_job(monkeypatch, tmp_path, ["lift smoke", "lift info"], {"lift smoke": [3]}, ran=ran)
    assert ran == ["lift smoke", "lift sync push"]


@pytest.mark.parametrize("line", P03_PARTIAL)
def test_notebook_fallback_goes_on_past_a_p03_line_that_exits_3(monkeypatch, tmp_path, line):
    ran = []
    with pytest.raises(RuntimeError, match=r"rc 3: some cells were skipped or left for a rerun"):
        run_notebook_job(monkeypatch, tmp_path, ["lift sync pull", line, "lift info"], {line: [3]}, ran=ran)
    assert ran == ["lift sync pull", line, "lift sync push", "lift info", "lift sync push"]
    ran = []  # lift train never exits 3: its 3 ends the job
    train = "lift train --candidate aya --recipe standard --seed 1"
    with pytest.raises(RuntimeError, match="rc 3"):
        run_notebook_job(monkeypatch, tmp_path, [train, "lift info"], {train: [3]}, ran=ran)
    assert ran == [train, "lift sync push"]


# Contract 7: the notebook installs like the other GPU runtimes, with the ref's constraints.txt.
def test_notebook_setup_cell_installs_with_the_constraints(monkeypatch):
    cell = notebook_cell("userdata")
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: SECRETS[name])
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: calls.append(list(argv))
                        or subprocess.CompletedProcess(argv, 0))
    scope = {"__name__": "__main__"}
    exec(compile(cell, "README:setup cell", "exec"), scope)
    (pip,) = [argv for argv in calls if argv[1:4] == ["-m", "pip", "install"]]
    i = pip.index("-c")
    repo = scope["REPO"]
    assert pip[i + 1] == f"{repo}/constraints.txt" and pip[-2:] == ["-e", f"{repo}[gpu,data,eval]"]


# --- colab/README.md ------------------------------------------------------------------------------


def readme() -> str:
    return README_PATH.read_text(encoding="utf-8")


# Review 2, R1 and contract 6: "use --ref p02-harness" pointed GPU jobs at a ref whose P01 loaders are stubs.
def test_readme_sends_gpu_jobs_to_a_ref_with_p01_and_p02():
    text = readme()
    assert "--ref p02-harness" not in text and "p01-data" in text and "P01 and P02" in text
    assert "lift sync check --write" in text


# Review 2, R6: no runner doc showed the pin step between Gate 0 and the dev eval.
def test_the_pin_step_sits_between_gate0_and_the_dev_eval():
    from lift.commands import remote as remote_cmd

    section = readme().split("## Running a job", 1)[1].split("What it does", 1)[0]
    for text in (section, remote_cmd.EXAMPLES):
        between = text[text.index("lift smoke"):text.index("lift run-dev --variant stock")]
        assert "configs/project.yaml" in between and "decisions.md" in between, text
        assert "commit" in between and "push" in between and "--ref" in between, text


# Review 2, R9 and contract 5: the docs never said what gate0.json, gate0-fp16.json and -partial mean.
def test_readme_explains_the_gate0_report_names():
    text = readme()
    for name in ("gate0.json", "gate0-fp16.json", "gate0-partial.json", "gate0-fp16-partial.json"):
        assert name in text, name
    assert "--skip" in text and "not_run" in text


def test_the_readme_exit_code_table_matches_lift_remote_help():
    table = re.findall(r"^\| (\d+) \|", readme(), flags=re.MULTILINE)
    codes = re.findall(r"^  (\d+) ", remote.EXIT_CODES, flags=re.MULTILINE)
    assert table == codes == ["0", "1", "2", "3", "4", "5", "6", "130"]


# Review 3, V19 and contract E: the README said gate0-fp16.json alone licenses T4 numbers.
def test_readme_says_what_an_fp16_licence_needs():
    row = next(line for line in readme().splitlines() if line.startswith("| `gate0-fp16.json`"))
    for words in ("PASS", "`gate0.json`", "same Global revision", "at least 200 prompts"):
        assert words in row, words


# Review 3, V11: what happens to a silent sync line.
def test_readme_says_a_hung_pull_is_not_tried_again():
    section = readme().split("**Hung commands.**", 1)[1].split("\n- **", 1)[0].split("\n\n", 1)[0]
    assert "`lift sync pull`" in section and "not tried again" in section


# Contract A: a line refused for a stale cell names its `lift restart` line, which goes right after the pull.
def test_readme_says_how_to_restart_a_stale_cell():
    text = readme()
    assert "lift restart --variant" in text and "outputs/.aside/" in text and "--judge second" in text
    assert re.search(r'--job "lift sync pull; lift restart [^";]+; lift run-dev [^";]+"', text)


# Review 4 (W1, W5, W8, W12, W13) and contracts G, H, I and L, where the operator reads them.
def test_readme_covers_skipped_cells_the_stop_request_and_watch():
    text = readme()

    def section(start: str, end: str) -> str:  # one line of words, however the README wraps them
        return " ".join(text.split(start, 1)[1].split(end, 1)[0].split())

    skipped = section("**Skipped cells (exit 3).**", "\nNotes:")
    for words in ("exits 3", "the next lines run", "exits 1", "--dtype", "--force", "another precision",
                  "never a restart", "judge", "AuditKIT record", "A100", "T4 licence"):
        assert words in skipped, words
    ctrl_c = section("- **Ctrl+C**", "\n- **")
    assert "cannot be uploaded" in ctrl_c and "tried again" in ctrl_c and "nothing was lost" in ctrl_c
    after_sleep = section("**After a laptop sleep or crash.**", "\n- **")
    for words in ("still running", "pid", "HF_TOKEN", "LIFT_RUNS_REPO", "GITHUB_TOKEN", "keeps the VM"):
        assert words in after_sleep, words
    row = next(line for line in text.splitlines() if line.startswith("| 1 |"))
    assert "exited 3" in row
    for words in ("exited 3", "another lift remote"):
        assert words in " ".join(remote.EXIT_CODES.split()), words
    # Review 5: watch checks this shell's token before its push retry, and keeps the VM when it cannot
    # push (X1, X13): `hf auth login` must hold a write token. A first Ctrl+C at the cap or during the push
    # retry lets the wait go on (X3), here and in --help.
    for words in ("WRITE token", "lift sync check --write", "exits 2", "exit 6", "lost"):
        assert words in after_sleep, words
    assert "`hf auth login` once, with a WRITE token" in text
    cap = section("- **`--max-hours`.**", "\n- **")
    from lift.commands import remote as remote_cmd

    for words_in in (ctrl_c, cap, " ".join(remote_cmd.EXAMPLES.split())):
        assert "first Ctrl+C" in words_in and "goes on" in words_in, words_in
    assert "push retry" in ctrl_c and "push retry" in " ".join(remote_cmd.EXAMPLES.split())


# R1-07: CLAUDE.md, Secrets: "Set them with a hidden prompt, never typed or pasted on a command line or
# into a chat." Step 6 and .env.example typed each secret as a literal, lift remote's missing-secret text
# said `$env:NAME = '...'` (`export NAME=...`), and the Claude Code advice had a session set what the
# exit-6 banner names in its own command line, which only a value pasted into the chat could do.
SECRET_NAME = r"(?:[A-Z][A-Z0-9_]*_(?:TOKEN|API_KEY)|NAME)\b"  # as CLAUDE.md lists them, or the placeholder
# A quoted literal assigned in PowerShell, or a literal exported in a POSIX shell. A hidden prompt, a
# variable or a subexpression is not a literal.
TYPED_SECRET = re.compile(rf"\$env:(?P<ps>{SECRET_NAME})\s*=\s*[\"'](?!\$\()"
                          rf"|\bexport\s+(?P<sh>{SECRET_NAME})=(?:[\"'](?!\$)|[^\s\"'$])")
HIDDEN_PROMPT = re.compile(r"Read-Host\b[^\n]*-AsSecureString|\bread -r?s\b")


def typed_secrets(text: str) -> list[str]:
    """Where a secret is typed on a command line: the line number and the secret's name, never a value."""
    return [f"line {text.count(chr(10), 0, m.start()) + 1}: {m.group('ps') or m.group('sh')}"
            for m in TYPED_SECRET.finditer(text)]


def hidden_prompt(name: str) -> str:
    """Windows PowerShell 5.1's hidden prompt for one secret, as the runbooks give it."""
    return f"[Net.NetworkCredential]::new('', (Read-Host -AsSecureString '{name}')).Password"


# R2-19: for Claude Code sessions, step 6 and .env.example kept each secret at user scope, the runs repo's
# WRITE token included, so every process started afterwards inherited it, the third-party build code of a
# local pip install too (colab/job.py Job._env: secrets go to lift only, never to git or pip). The operator
# now stores each one encrypted for their Windows account (Export-Clixml: DPAPI), and a session loads it
# only in the command line that runs lift. LIFT_RUNS_REPO and LIFT_JUDGE_MAX_USD are settings.
STORED = '"$env:USERPROFILE\\.lift\\{name}.xml"'
USER_SCOPE = re.compile(r"\[Environment\]::SetEnvironmentVariable\(\s*'(?P<name>[A-Z][A-Z0-9_]*)'\s*,\s*"
                        r"(?P<value>[^\n]+?)\s*,\s*'User'\s*\)")


def stored(name: str) -> str:
    """How the operator stores one secret for Claude Code sessions, in their own window."""
    return f"Read-Host -AsSecureString '{name}' | Export-Clixml {STORED.format(name=name)}"


def loaded(name: str) -> str:
    """How a session loads one stored secret, in the command line that runs lift."""
    return (f"$env:{name} = [Net.NetworkCredential]::new('', (Import-Clixml {STORED.format(name=name)}))"
            ".Password")


@pytest.mark.parametrize("path", ["colab/README.md", "kaggle/README.md", ".env.example"])
def test_no_runbook_types_a_secret_on_a_command_line(path):
    assert typed_secrets((REPO_ROOT / path).read_text(encoding="utf-8")) == [], path


@pytest.mark.parametrize("path", ["colab/README.md", "kaggle/README.md", ".env.example"])
def test_no_runbook_keeps_a_secret_at_user_scope(path):
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    kept = sorted({m["name"] for m in USER_SCOPE.finditer(text)       # `$null` removes one instead
                   if re.fullmatch(SECRET_NAME, m["name"]) and m["value"].strip() != "$null"})
    assert kept == [], (f"{path} keeps {', '.join(kept)} at user scope, so every process started afterwards "
                        "inherits them, the build code of a local pip install included")


def test_the_colab_setup_step_sets_each_secret_with_a_hidden_prompt():
    step = readme().split("**Set the secrets**", 1)[1].split("\n7. **", 1)[0]
    assert typed_secrets(step) == []
    for name in ("GITHUB_TOKEN", "HF_TOKEN", "OPENAI_API_KEY"):
        assert f"$env:{name} = {hidden_prompt(name)}" in step, name
        # The one-time form for Claude Code sessions, typed in the user's own window (R2-19).
        assert stored(name) in step, name
    assert 'New-Item -ItemType Directory -Force "$env:USERPROFILE\\.lift"' in step
    words = " ".join(step.split())
    # Why not at user scope, and how to remove a secret an earlier step 6 put there, or a stored one.
    assert "pip install" in words and "inherit" in words
    assert "[Environment]::SetEnvironmentVariable('HF_TOKEN', $null, 'User')" in words
    assert f"Remove-Item {STORED.format(name='HF_TOKEN')}" in words


@pytest.mark.parametrize("os_name", ["nt", "posix"], ids=["powershell", "posix-shell"])
def test_lift_remote_never_says_to_type_a_missing_secret(monkeypatch, os_name):
    ctx = types.SimpleNamespace(out=io.StringIO())
    with monkeypatch.context() as mp:  # missing_message picks its example by os.name
        mp.setattr(remote.os, "name", os_name)
        launch = remote.missing_message(["GITHUB_TOKEN", "LIFT_RUNS_REPO", "HF_TOKEN"])  # a launch's exit 2
        blocked = remote.retry_blocker(ctx, None, ["HF_TOKEN"])  # watch's push retry, from a shell without it
        remote.kept_on_vm(ctx, "lift-r107", None, None, blocked)  # the exit-6 banner names what to set
    banner = ctx.out.getvalue()
    assert "KEEPS RUNNING" in banner and "missing HF_TOKEN:" in banner and "missing GITHUB_TOKEN," in launch
    for text in (launch, banner):
        assert typed_secrets(text) == [] and HIDDEN_PROMPT.search(text), text
        assert "`hf auth login` with a WRITE token" in text
    example = f"$env:NAME = {hidden_prompt('NAME')}" if os_name == "nt" else "read -rs NAME; export NAME"
    assert example in launch
    # R2-19: in a Claude Code session, a secret step 6 stored is loaded in the command line that runs lift.
    assert (f"{loaded('NAME')}; lift remote" in launch) == (os_name == "nt")


def test_the_claude_code_advice_never_routes_a_secret_through_the_chat():
    advice = " ".join(readme().split("**From Claude Code.**", 1)[1].split("\n\n", 1)[0].split())
    assert "a 6 means" in advice and "lift remote watch" in advice
    assert "never asks for or accepts a secret in the chat" in advice
    # It loads each secret a lift command needs from its stored file, in the command line that runs that
    # command, never from the user environment (R2-19).
    assert f"{loaded('HF_TOKEN')}; lift remote" in advice and "GetEnvironmentVariable" not in advice
    assert "own terminal" in advice and "set what its banner names in the same command line" not in advice


# W3, on R1-08's restart paragraphs (decisions item 63). After a whole-cell restart, when another machine
# pushed only some of the files it emptied (one judge's judgements, say), the push sends the others empty, the
# generations among them, and keeps only the changed copies: not "the runs repo's copies". After a --judge
# restart, when that machine also judged the generations it pushed, the pull the summary names reports the
# emptied-file conflict, and only `lift sync push --force` settles it: a plain push reports it again.
@pytest.mark.parametrize("path", ["colab/README.md", "kaggle/README.md"])
def test_the_restart_paragraph_says_what_the_next_push_keeps_and_what_settles_it(path):
    text = " ".join((REPO_ROOT / path).read_text(encoding="utf-8").split())
    after = text[text.index("checkpoint push after it", text.index("`lift restart` moves the cell's")):]
    note = after[:after.index("), and ") + 1]           # the parenthetical on what that push does
    assert "keeps the runs repo's copies" not in note, note
    assert "keeps the runs repo's copy of each file that changed there" in note, note
    assert "`lift sync push --force` settles by completing the restart" in note, note
    then = note.split("`lift sync pull`, then `lift sync push`", 1)[1]
    assert ("if the emptied file also changed there, that pull reports the emptied-file conflict, which "
            "`lift sync push --force` settles") in then, note
