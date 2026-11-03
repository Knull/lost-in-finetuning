"""kaggle/runner.ipynb and kaggle/bootstrap.sh, checked offline.

The notebook's two code cells run here for real, in one namespace as in a Kaggle kernel, with Kaggle's
secrets client, git, bash and the filesystem calls replaced by fakes: no network, no real token,
nothing written outside tmp. The bootstrap runs under a real bash (Git Bash on Windows) with fake
python, pip, uv and nvidia-smi on PATH; its heredoc checks run on this interpreter.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import pytest

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle"
NOTEBOOK = KAGGLE / "runner.ipynb"
BOOTSTRAP = KAGGLE / "bootstrap.sh"
WORKING = PurePosixPath("/kaggle/working")
ROOT = "/kaggle/temp/lift"
LIFT_PY = "/kaggle/temp/lift/venv/bin/python"
LIFT_DIRS = {"LIFT_DATA_DIR": "data", "LIFT_OUTPUT_DIR": "outputs", "LIFT_VALIDATION_DIR": "validation"}
# Dummies shaped like the real secrets. None may ever reach a URL, a command line or printed output.
SECRETS = {"HF_TOKEN": "hf_dummy0token0for0tests", "GITHUB_TOKEN": "github_pat_dummy0token0for0tests",
           "LIFT_RUNS_REPO": "someone/lift-runs", "ANTHROPIC_API_KEY": "sk-ant-dummy0key0for0tests"}
UNATTACHED = ("OPENAI_API_KEY", "LIFT_JUDGE_MAX_USD")
# Kaggle's secrets client authenticates with these (kaggle_web_client.py in Kaggle/docker-python), so
# any process that inherits them can fetch every attached secret.
KAGGLE_TOKENS = {"KAGGLE_USER_SECRETS_TOKEN": "eyJdummy0jwt0for0tests",
                 "KAGGLE_IAP_TOKEN": "dummy0iap0for0tests"}
HIDDEN = (*SECRETS, *UNATTACHED, *KAGGLE_TOKENS)  # no child process inherits these
PUSH = "lift sync push"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    return ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"]


def _inside_working(path: str) -> bool:
    p = PurePosixPath(path)
    return p == WORKING or WORKING in p.parents


def _auth(token: str) -> str:
    return base64.b64encode(f"x-access-token:{token}".encode()).decode()


def _hidden(env: dict[str, str]) -> dict[str, str]:
    """The HIDDEN entries of a child's environment, each value replaced by the name of the dummy it
    equals ("?" for anything else). Tests assert on this, never on an environment itself: pytest
    would print the environment's values when the assert fails."""
    dummies = {value: name for name, value in {**SECRETS, **KAGGLE_TOKENS}.items()}
    return {name: dummies.get(value, "?") for name, value in env.items() if name in HIDDEN}


AS_DUMMIES = {name: name for name in SECRETS}  # what _hidden returns for an environment holding SECRETS


# --- the notebook file ----------------------------------------------------------------------------------


def test_notebook_is_clean_nbformat_4():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code"]
    assert len({c["id"] for c in nb["cells"]}) == 3
    for cell in nb["cells"]:
        assert all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_notebook_validates_against_the_nbformat_schema():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.reads(NOTEBOOK.read_text(encoding="utf-8"), as_version=4))


def test_no_cell_writes_credentials_into_a_url():
    for source in _code_cells():
        assert not re.search(r"\w+://[^\s/'\"]*@", source)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()))


def test_docs_say_a_push_never_stops_the_job_and_a_conflict_exits_3():
    """kaggle/README.md said in one place that a push conflict stops the job and in another that a
    failed push does not. The runner never stops for a push, and a conflict is exit code 3."""
    about = "".join(_notebook()["cells"][0]["source"])
    for name, text in (("kaggle/README.md", (KAGGLE / "README.md").read_text(encoding="utf-8")),
                       ("the notebook's first cell", about)):
        stops = [s for s in _sentences(text)
                 if "push" in s and re.search(r"(?<!not )(?<!never )\bstops? the job", s)]
        assert stops == [], name
        assert re.search(r"\bexits? 3\b", text), name


# --- cell 1: secrets, paths, clone, bootstrap -----------------------------------------------------------


class _FakeSecrets:
    def __init__(self, values: dict[str, str]):
        self.values = values

    def get_secret(self, name: str) -> str:
        if name not in self.values:
            raise ConnectionError(f"No user secrets exist for label {name}")  # Kaggle raises a BackendError
        return self.values[name]


@pytest.fixture
def setup_cell(monkeypatch):
    """Runs cell 1 against fakes. ``state`` keeps each call as (command, kwargs, the environment the
    child gets), the cell's namespace and os.environ as the cell left it, even when the cell raises."""
    state = SimpleNamespace(calls=[], removed=[], env={}, ns={})

    def run(secrets: dict[str, str], *, environ: dict[str, str] | None = None, clone_rc: int = 0,
            clone_stderr: str = "", bootstrap_rc: int = 0):
        def fake_run(cmd, **kwargs):
            env = kwargs.get("env")
            state.calls.append((list(cmd), kwargs, dict(os.environ if env is None else env)))
            if cmd[:2] == ["git", "clone"]:
                return subprocess.CompletedProcess(cmd, clone_rc, stdout="", stderr=clone_stderr)
            return subprocess.CompletedProcess(cmd, bootstrap_rc)

        kaggle_secrets = types.ModuleType("kaggle_secrets")
        kaggle_secrets.UserSecretsClient = lambda: _FakeSecrets(secrets)
        monkeypatch.setitem(sys.modules, "kaggle_secrets", kaggle_secrets)
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(shutil, "rmtree", lambda path, **kwargs: state.removed.append(path))
        state.ns = {"__name__": "__main__"}
        with mock.patch.dict(os.environ):
            for name in (*HIDDEN, *LIFT_DIRS):
                os.environ.pop(name, None)
            os.environ.update(environ or {})
            try:
                exec(compile(_code_cells()[0], "runner.ipynb:setup", "exec"), state.ns)
            finally:
                state.env = dict(os.environ)

    state.run = run
    return state


def test_setup_cell_moves_every_lift_dir_out_of_kaggle_working(setup_cell):
    setup_cell.run(SECRETS)
    for var, leaf in LIFT_DIRS.items():
        path = setup_cell.env[var]
        assert PurePosixPath(path).is_absolute() and not _inside_working(path)
        assert PurePosixPath(path).name == leaf  # lift sync pull needs the canonical leaf name
    (clone_cmd, _, _), (_, bootstrap_kwargs, _) = setup_cell.calls
    assert not _inside_working(clone_cmd[-1]) and not _inside_working(bootstrap_kwargs["cwd"])


def test_setup_cell_loads_attached_secrets_and_skips_the_rest_silently(setup_cell, capsys):
    # A token pasted with a newline, and an optional secret attached empty.
    setup_cell.run({**SECRETS, "GITHUB_TOKEN": SECRETS["GITHUB_TOKEN"] + "\n", "LIFT_JUDGE_MAX_USD": " "})
    assert setup_cell.ns["SECRETS"] == SECRETS
    (_, _, clone_env), _ = setup_cell.calls
    assert clone_env["GIT_CONFIG_VALUE_0"].endswith(_auth(SECRETS["GITHUB_TOKEN"]))
    printed = "".join(capsys.readouterr())
    assert not [name for name, value in SECRETS.items() if value in printed]


def test_setup_cell_keeps_the_secrets_away_from_git_bootstrap_and_os_environ(setup_cell):
    # pip and uv run third-party build code, so neither the secrets nor Kaggle's tokens for fetching
    # them may reach git or bootstrap.sh, not even a copy an older cell 1 left in os.environ.
    stale = {"OPENAI_API_KEY": "sk-dummy0key0left0in0the0kernel"}
    setup_cell.run(SECRETS, environ={**KAGGLE_TOKENS, **stale})
    values = [*SECRETS.values(), *KAGGLE_TOKENS.values(), *stale.values()]
    assert [cmd[0] for cmd, _, _ in setup_cell.calls] == ["git", "bash"]
    for cmd, _, env in setup_cell.calls:
        names = sorted(_hidden(env))
        leaked = [value for value in values if any(value in text for text in env.values())]
        assert names == [] and leaked == [], cmd[0]
    kept = sorted(set(SECRETS) & set(setup_cell.env))
    assert kept == []  # never copied into the kernel's environment
    assert setup_cell.env["KAGGLE_USER_SECRETS_TOKEN"] == KAGGLE_TOKENS["KAGGLE_USER_SECRETS_TOKEN"]  # rerun
    assert setup_cell.ns["SECRETS"] == SECRETS
    assert sorted(setup_cell.ns["HIDDEN"]) == sorted(HIDDEN)  # bootstrap.sh unsets the same names


def test_setup_cell_clones_with_the_token_only_in_a_header(setup_cell):
    setup_cell.run(SECRETS)
    token = SECRETS["GITHUB_TOKEN"]
    (clone_cmd, clone_kwargs, env), (bootstrap_cmd, _, _) = setup_cell.calls
    assert clone_cmd[:2] == ["git", "clone"]
    url = clone_cmd[-2]
    assert url.startswith("https://github.com/") and "@" not in url
    assert not [arg for arg in clone_cmd if token in arg or _auth(token) in arg]
    assert clone_kwargs["env"] is not None  # given explicitly, not inherited
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == f"AUTHORIZATION: basic {_auth(token)}"
    assert not [name for name in setup_cell.env if name.startswith("GIT_CONFIG")]  # the header is not kept
    repo = clone_cmd[-1]
    assert setup_cell.removed == [repo]  # a re-run re-clones; nothing else is deleted
    assert bootstrap_cmd == ["bash", f"{repo}/kaggle/bootstrap.sh"]


def test_setup_cell_stops_before_cloning_when_a_required_secret_is_missing(setup_cell):
    secrets = {name: value for name, value in SECRETS.items() if name != "GITHUB_TOKEN"}
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN") as err:
        setup_cell.run(secrets)
    assert setup_cell.calls == []
    assert not [value for value in secrets.values() if value in str(err.value)]


def test_a_failed_clone_reports_the_git_error_with_the_token_redacted(setup_cell):
    token = SECRETS["GITHUB_TOKEN"]
    stderr = (f"fatal: unable to access 'https://x-access-token:{token}@github.com/x.git/'\n"
              f"header: AUTHORIZATION: basic {_auth(token)}\n")
    with pytest.raises(RuntimeError) as err:
        setup_cell.run(SECRETS, clone_rc=128, clone_stderr=stderr)
    message = str(err.value)
    assert "exit 128" in message and "[redacted]" in message
    assert token not in message and _auth(token) not in message
    assert err.value.__context__ is None and err.value.__cause__ is None  # no chained error with args
    assert [cmd[:2] for cmd, _, _ in setup_cell.calls] == [["git", "clone"]]  # bootstrap never ran


def test_a_failed_bootstrap_fails_the_setup_cell(setup_cell):
    with pytest.raises(RuntimeError, match="bootstrap.sh failed"):
        setup_cell.run(SECRETS, bootstrap_rc=1)


# --- cell 2: the job runner -----------------------------------------------------------------------------


class _FakeRun:
    """Stands in for subprocess.run. ``outcomes`` maps a line ("lift sync push") to an exit code, a
    list of exit codes (consumed in order, then 0) or an exception to raise."""

    def __init__(self, outcomes: dict | None = None):
        self.outcomes = outcomes or {}
        self.lines: list[str] = []
        self.kwargs: list[dict] = []

    def __call__(self, cmd, **kwargs):
        lift = cmd[:3] == [LIFT_PY, "-m", "lift"]
        line = " ".join(["lift", *cmd[3:]] if lift else cmd)
        self.lines.append(line)
        self.kwargs.append(kwargs)
        outcome = self.outcomes.get(line, 0)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else 0
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(cmd, outcome)


def _fake_open(path, *args, **kwargs):
    assert path == "/kaggle/working/lift_env"
    return io.StringIO(f"LIFT_PY={LIFT_PY}\n")


def _no_sleep(seconds):
    raise AssertionError(f"the job would sleep {seconds} s for real: pass sleep=")


@pytest.fixture
def job_cell(setup_cell, monkeypatch, capsys):
    """Runs cell 1, then cell 2 as committed (its template JOB) in the same namespace, as a kernel
    does, against a fake subprocess.run. Returns ``run(job, fake, sleep=...)``, which calls cell 2's
    run_job as the cell does, the fake and what cell 2 printed."""
    for name in HIDDEN:  # no real secret of this machine may reach a fake child or a failure message
        monkeypatch.delenv(name, raising=False)
    setup_cell.run(SECRETS)
    for name, path in setup_cell.ns["LIFT_DIRS"].items():  # cell 1's os.environ changes, which the
        monkeypatch.setenv(name, path)                    # setup_cell fixture rolled back
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    capsys.readouterr()
    namespace = {**setup_cell.ns, "open": _fake_open}
    with mock.patch.object(time, "sleep", _no_sleep):  # becomes run_job's default: no test really sleeps
        exec(compile(_code_cells()[1], "runner.ipynb:job", "exec"), namespace)

    def run(job, fake, sleep=_no_sleep):
        return namespace["run_job"](job, LIFT_PY, cwd=ROOT, secrets=SECRETS, run=fake, sleep=sleep)

    return SimpleNamespace(run=run, fake=fake, printed=capsys.readouterr().out)


def test_job_cell_refuses_to_run_before_the_setup_cell(monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.delenv("LIFT_OUTPUT_DIR", raising=False)  # a restarted kernel has lost cell 1's env
    with pytest.raises(RuntimeError, match="setup cell"):
        exec(compile(_code_cells()[1], "runner.ipynb:job", "exec"),
             {"__name__": "__main__", "open": _fake_open})
    assert fake.lines == []


def test_job_cell_refuses_when_the_kernel_lost_the_secrets(monkeypatch, tmp_path):
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("LIFT_OUTPUT_DIR", str(tmp_path / "outputs"))  # %reset clears SECRETS, not os.environ
    with pytest.raises(RuntimeError, match="setup cell"):
        exec(compile(_code_cells()[1], "runner.ipynb:job", "exec"),
             {"__name__": "__main__", "ROOT": ROOT, "open": _fake_open})
    assert fake.lines == []


def test_job_cell_runs_the_template_then_pushes_after_each_line_and_at_the_end(job_cell):
    assert job_cell.fake.lines == ["lift sync pull", "lift info", PUSH, PUSH]
    for kwargs in job_cell.fake.kwargs:
        hidden = _hidden(kwargs["env"])
        assert kwargs["cwd"] == ROOT and kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        assert hidden == AS_DUMMIES  # the lift commands get cell 1's SECRETS
    assert "job OK" in job_cell.printed


def test_only_the_lift_commands_get_the_secrets(job_cell, monkeypatch):
    for name, value in KAGGLE_TOKENS.items():  # a Kaggle kernel has them; no child may
        monkeypatch.setenv(name, value)
    fake = _FakeRun()
    job_cell.run("nvidia-smi -L\nlift info", fake)
    assert fake.lines == ["nvidia-smi -L", PUSH, "lift info", PUSH, PUSH]
    for line, kwargs in zip(fake.lines, fake.kwargs, strict=True):
        hidden = _hidden(kwargs["env"])
        assert hidden == (AS_DUMMIES if line.startswith("lift ") else {}), line


def test_job_stops_at_the_first_failure_and_still_pushes(job_cell):
    job = """
    lift sync pull
    # a comment
    lift generate --variant stock --split dev --dtype fp16
    lift judge --variant stock --split dev --dtype fp16
    """
    fake = _FakeRun({"lift generate --variant stock --split dev --dtype fp16": 1})
    with pytest.raises(RuntimeError, match="job FAILED: 'lift generate .*' exited 1"):
        job_cell.run(job, fake)
    assert fake.lines == ["lift sync pull", "lift generate --variant stock --split dev --dtype fp16", PUSH]


def test_a_pull_conflict_stops_the_job_before_any_gpu_work(job_cell):
    fake = _FakeRun({"lift sync pull": 3})
    with pytest.raises(RuntimeError, match="'lift sync pull' exited 3"):
        job_cell.run("lift sync pull\nlift run-dev --variant stock --dtype fp16", fake)
    assert fake.lines == ["lift sync pull", PUSH]


def test_job_pushes_even_after_an_interrupt(job_cell):
    fake = _FakeRun({"lift run-dev --variant stock --dtype fp16": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        job_cell.run("lift run-dev --variant stock --dtype fp16\nlift info", fake)
    assert fake.lines == ["lift run-dev --variant stock --dtype fp16", PUSH]


def test_a_push_that_exits_1_is_retried_with_backoff(job_cell, capsys):
    # Exit 1: not everything was sent (a Hub 502, a reset connection), and a push is safe to repeat.
    fake, waits = _FakeRun({PUSH: [1, 1, 0]}), []
    job_cell.run("lift sync pull", fake, sleep=waits.append)
    assert fake.lines == ["lift sync pull", PUSH, PUSH, PUSH]
    assert waits == [60, 120]
    assert "job OK" in capsys.readouterr().out


def test_a_push_failing_three_times_is_reported_and_the_job_goes_on(job_cell):
    # As in colab/job.py, a failed checkpoint push does not stop the GPU work: the next push sends
    # whatever is missing, and the job still fails at the end so the person sees it.
    fake, waits = _FakeRun({PUSH: [1, 1, 1]}), []
    with pytest.raises(RuntimeError, match="checkpoint push after 'lift info' exited 1 on all 3 attempts"):
        job_cell.run("lift info\nlift smoke --compare-dtype fp16", fake, sleep=waits.append)
    assert fake.lines == ["lift info", PUSH, PUSH, PUSH, "lift smoke --compare-dtype fp16", PUSH, PUSH]
    assert waits == [60, 120]


def test_a_push_conflict_is_reported_never_retried_and_the_job_goes_on(job_cell):
    # Exit 3: done, but the conflicting local copies went under conflicts/ in the runs repo.
    fake = _FakeRun({PUSH: [3]})
    with pytest.raises(RuntimeError, match=r"checkpoint push after 'lift info' exited 3 \(conflicts"):
        job_cell.run("lift info\nlift smoke --compare-dtype fp16", fake)
    assert fake.lines == ["lift info", PUSH, "lift smoke --compare-dtype fp16", PUSH, PUSH]


def test_a_failed_final_push_fails_the_job_and_only_exit_1_is_retried(job_cell):
    fake = _FakeRun({PUSH: [0, 2]})
    with pytest.raises(RuntimeError, match="final push exited 2"):
        job_cell.run("lift info", fake)
    assert fake.lines == ["lift info", PUSH, PUSH]


def test_a_push_line_in_the_job_is_retried_and_never_stops_the_job(job_cell):
    line = "lift sync push --only outputs"
    fake, waits = _FakeRun({line: [1, 3]}), []
    with pytest.raises(RuntimeError, match=re.escape(f"{line!r} exited 3")):
        job_cell.run(f"{line}\nlift info", fake, sleep=waits.append)
    assert fake.lines == [line, line, "lift info", PUSH, PUSH]
    assert waits == [60]


def test_an_interrupt_while_waiting_to_retry_still_pushes_at_the_end(job_cell):
    def interrupt(seconds):
        raise KeyboardInterrupt

    fake = _FakeRun({PUSH: [1]})
    with pytest.raises(KeyboardInterrupt):
        job_cell.run("lift info\nlift smoke --compare-dtype fp16", fake, sleep=interrupt)
    assert fake.lines == ["lift info", PUSH, PUSH]


def test_job_runs_non_lift_lines_verbatim_and_adds_no_checkpoint_after_sync_lines(job_cell):
    fake = _FakeRun()
    job_cell.run("nvidia-smi -L\nlift sync push\n\n# done", fake)
    # nvidia-smi, its checkpoint push, the job's own push line (no checkpoint after it), the final push
    assert fake.lines == ["nvidia-smi -L", PUSH, PUSH, PUSH]


# --- bootstrap.sh ---------------------------------------------------------------------------------------


def _bash() -> str | None:
    """A real bash. On Windows that is Git Bash: the bash.exe on PATH there is usually WSL's launcher."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    for root in Path(git).resolve().parents if git else ():
        if (root / "bin" / "bash.exe").is_file() and (root / "usr" / "bin").is_dir():
            return str(root / "bin" / "bash.exe")
    return None


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash (on Windows: Git Bash) found")

FAKE_PYTHON = """#!/usr/bin/env bash
# Fake interpreter: logs each call (and any watched variable it can see), fakes pip, uv and
# `lift info`, and hands the rest (the bootstrap's -c checks and heredoc scripts) to the real one.
role=image
case "$0" in */venv/bin/python) role=venv ;; esac
echo "$role: $*" >> "$FAKE_LOG"
for name in ${FAKE_WATCH:-}; do
  if [ -n "${!name:-}" ]; then echo "$role: $* sees $name" >> "$FAKE_SEEN"; fi
done
case "$role $*" in
  "venv -m pip"*) echo "$0: No module named pip" >&2; exit 1 ;;  # like a real uv venv
esac
case "$*" in
  *version_info*) exit "$FAKE_VERSION_RC" ;;
  "-m pip uninstall"* | "-m uv pip uninstall"*) rm -rf "$FAKE_SITE/torchao"; exit 0 ;;
  "-m uv venv"*) mkdir -p "${!#}/bin" && cp "$0" "${!#}/bin/python"; exit 0 ;;
  "-m uv pip install"*) echo "UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-}" >> "$FAKE_LOG"; exit 0 ;;
  "-m pip"* | "-m uv"*) exit 0 ;;
  "-m lift info") echo '{"fake": "lift info"}'; exit 0 ;;
esac
exec "$REAL_PYTHON" "$@"
"""
FAKE_NVIDIA_SMI = '#!/usr/bin/env bash\nexit "$FAKE_GPU_RC"\n'
FAKE_TORCH = """__version__ = "0.0+fake"
class version:
    cuda = "99.9"
class cuda:
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def device_count():
        return 0
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


def _run_bootstrap(tmp_path: Path, *, python_too_old: bool, gpu: bool = False, fake_torch: bool = False,
                   watch: dict[str, str] | None = None):
    """Run a copy of bootstrap.sh (so its repo is tmp/repo) with the fakes first on PATH. ``watch``
    variables are put in its environment, and every fake python call that can see one is logged."""
    script = _write(tmp_path / "repo" / "kaggle" / "bootstrap.sh", BOOTSTRAP.read_text(encoding="utf-8"))
    _write(tmp_path / "bin" / "python", FAKE_PYTHON)
    _write(tmp_path / "bin" / "nvidia-smi", FAKE_NVIDIA_SMI)
    site = tmp_path / "site"
    _write(site / "torchao" / "__init__.py", "")
    if fake_torch:
        _write(site / "torch" / "__init__.py", FAKE_TORCH)
    env = {name: value for name, value in os.environ.items() if name != "UV_TORCH_BACKEND"}
    env.update(watch or {})
    env.update({
        "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(site),
        "REAL_PYTHON": Path(sys.executable).as_posix(),
        "FAKE_LOG": (tmp_path / "calls.log").as_posix(),
        "FAKE_WATCH": " ".join(watch or ()),
        "FAKE_SEEN": (tmp_path / "seen.log").as_posix(),
        "FAKE_SITE": site.as_posix(),
        "FAKE_VERSION_RC": "1" if python_too_old else "0",
        "FAKE_GPU_RC": "0" if gpu else "1",
        "LIFT_VENV_DIR": (tmp_path / "venv").as_posix(),
        "LIFT_ENV_FILE": (tmp_path / "lift_env").as_posix(),
    })
    result = subprocess.run([BASH, script.as_posix()], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=180)
    log = (tmp_path / "calls.log").read_text(encoding="utf-8").splitlines()
    seen = tmp_path / "seen.log"
    lift_env = tmp_path / "lift_env"
    return SimpleNamespace(rc=result.returncode, stdout=result.stdout, stderr=result.stderr, log=log,
                           seen=seen.read_text(encoding="utf-8").splitlines() if seen.exists() else [],
                           lift_env=lift_env.read_text(encoding="utf-8") if lift_env.exists() else None,
                           torchao_left=(site / "torchao").exists())


@needs_bash
def test_bootstrap_has_valid_bash_syntax():
    result = subprocess.run([BASH, "-n"], input=BOOTSTRAP.read_bytes(), capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_bootstrap_venv_branch_never_uses_pip_inside_the_uv_venv():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    venv_branch = re.search(r"\nelse\n(.*?)\nfi\n", text, flags=re.S).group(1)
    assert "uv pip uninstall" in venv_branch and "-m pip uninstall" not in venv_branch
    assert 'EXTRAS="gpu,data,eval"' in text and "|| true" not in text
    assert "set -euo pipefail" in text


@needs_bash
def test_bootstrap_in_a_uv_venv_installs_and_removes_torchao_through_uv(tmp_path):
    r = _run_bootstrap(tmp_path, python_too_old=True)
    assert r.rc == 0, r.stderr
    venv_python = f"{(tmp_path / 'venv').as_posix()}/bin/python"
    assert r.lift_env == f"LIFT_PY={venv_python}\n"
    installs = [line for line in r.log if line.startswith("image: -m uv pip install")]
    assert len(installs) == 1 and f"--python {venv_python} -e " in installs[0]
    assert installs[0].endswith("/repo[gpu,data,eval]")
    assert "UV_TORCH_BACKEND=auto" in r.log
    assert f"image: -m uv pip uninstall -q --python {venv_python} torchao" in r.log
    assert not [line for line in r.log if line.startswith("venv: -m pip")]  # a uv venv has no pip
    assert not r.torchao_left
    assert r.log[-1] == "venv: -m lift info"


@needs_bash
def test_bootstrap_on_a_new_enough_image_uses_the_image_pip(tmp_path):
    r = _run_bootstrap(tmp_path, python_too_old=False)
    assert r.rc == 0, r.stderr
    assert r.lift_env.startswith("LIFT_PY=") and r.lift_env.rstrip("\n").endswith("/bin/python")
    assert "image: -m pip install -q -e " in "\n".join(r.log)
    assert [line for line in r.log if line.endswith("[gpu,data,eval]")]
    assert "image: -m pip uninstall -y -q torchao" in r.log
    assert not [line for line in r.log if "-m uv" in line]
    assert not r.torchao_left


@needs_bash
@pytest.mark.parametrize("python_too_old", [True, False])
def test_bootstrap_hides_every_secret_from_pip_uv_and_lift(tmp_path, python_too_old):
    # Cell 1 passes none, but another caller might; pip and uv run third-party build code.
    watch = {**SECRETS, **dict.fromkeys(UNATTACHED, "dummy0value0for0tests"), **KAGGLE_TOKENS}
    assert sorted(watch) == sorted(HIDDEN)
    r = _run_bootstrap(tmp_path, python_too_old=python_too_old, watch=watch)
    assert r.rc == 0, r.stderr
    assert [line for line in r.log if "-m lift info" in line]
    assert r.seen == []


@needs_bash
def test_bootstrap_stops_when_torch_cannot_use_the_gpu(tmp_path):
    r = _run_bootstrap(tmp_path, python_too_old=True, gpu=True, fake_torch=True)
    assert r.rc != 0
    assert "torch cannot use the GPU" in r.stderr
    assert r.lift_env is None  # no interpreter is handed to the job cell


def _heredoc(marker: str) -> str:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", BOOTSTRAP.read_text(encoding="utf-8"), flags=re.S)
    (block,) = [b for b in blocks if marker in b]
    return block


@pytest.mark.parametrize("output_dir, refused", [("/kaggle/working/outputs", True),
                                                 ("/kaggle/working", True),
                                                 ("/kaggle/temp/lift/outputs", False)])
def test_bootstrap_refuses_lift_dirs_inside_kaggle_working(output_dir, refused):
    env = {**os.environ, "LIFT_OUTPUT_DIR": output_dir}
    result = subprocess.run([sys.executable, "-c", _heredoc("/kaggle/working")], env=env,
                            capture_output=True, text=True, timeout=60)
    assert (result.returncode != 0) == refused, result.stderr
    assert ("refusing" in result.stderr) == refused
