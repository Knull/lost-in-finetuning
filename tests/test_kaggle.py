"""kaggle/runner.ipynb and kaggle/bootstrap.sh, checked offline.

The notebook's two code cells run here for real, in one namespace as in a Kaggle kernel, with Kaggle's
secrets client, git, bash, the filesystem calls, subprocess.Popen and os.killpg replaced by fakes: no
network, no real token, no real process started or signalled, nothing written outside tmp. The
exception is the Linux-only tests of the PDEATHSIG shim, which start and kill small local processes
(sh, cat, sleep and a stand-in kernel), never lift. The bootstrap runs under a real bash (Git Bash on
Windows) with fake python, pip, uv and nvidia-smi on PATH; its heredoc checks run on this interpreter.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import errno
import io
import itertools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
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
CHECK = "lift sync check --write"  # cell 2's preflight, before the first line
KILL = getattr(signal, "SIGKILL", signal.SIGTERM)  # as in cell 2: Windows has no SIGKILL


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    return ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"]


def _about() -> str:
    return "".join(_notebook()["cells"][0]["source"])


def _readme() -> str:
    return (KAGGLE / "README.md").read_text(encoding="utf-8")


def _t4_section() -> str:
    return _readme().split("## Jobs on a T4 x2", 1)[1].split("\n## ", 1)[0]


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


# --- the notebook file and the docs ---------------------------------------------------------------------


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


def _items(text: str) -> list[str]:
    """The paragraphs and list items of a markdown text, each on one line."""
    return [" ".join(part.split()) for part in re.split(r"\n\s*\n|\n\s*(?:-|\d+\.) ", text) if part.strip()]


def _docs() -> tuple[tuple[str, str], ...]:
    return ("kaggle/README.md", _readme()), ("the notebook's first cell", _about())


def _code_blocks(text: str) -> list[list[str]]:
    """The fenced code blocks of a markdown text, each as its stripped, non-empty lines."""
    blocks, inside = [], False
    for line in (line.strip() for line in text.splitlines()):
        if line.startswith("```"):
            inside = not inside
            if inside:
                blocks.append([])
        elif inside and line:
            blocks[-1].append(line)
    return blocks


def test_docs_state_the_sync_exit_codes_as_the_runner_acts_on_them():
    """A push never stops the job, except an exit 2: a configuration problem that no retry fixes, which
    ends the job at once. An exit 3 (conflicts) is reported and never retried."""
    for name, text in _docs():
        stops = [s for s in _sentences(text)
                 if "push" in s and re.search(r"(?<!not )(?<!never )\bstops? the job", s)]
        assert stops == [], name
        assert re.search(r"\bexits? 3\b", text), name
        assert [s for s in _sentences(text) if re.search(r"\bexits? 2\b", s) and "configuration" in s], name
        assert CHECK in text, name  # the preflight


def _constants(source: str, *names: str) -> dict:
    """Module-level constants of a cell, read without running it."""
    found = {target.id: ast.literal_eval(node.value) for node in ast.parse(source).body
             if isinstance(node, ast.Assign) for target in node.targets
             if isinstance(target, ast.Name) and target.id in names}
    assert sorted(found) == sorted(names), "cell 2 lacks some of its watchdog limits"
    return found


def test_docs_state_the_watchdog_limits_cell_2_sets():
    limits = _constants(_code_cells()[1], "MAX_SILENT_MINUTES", "PUSH_LIMIT_MINUTES", "KILL_GRACE_SECONDS",
                        "RETRY_WAITS")
    for name, text in _docs():
        text = " ".join(text.split())
        assert f"{limits['MAX_SILENT_MINUTES']} minutes" in text, name
        assert f"{limits['PUSH_LIMIT_MINUTES']} minutes" in text, name
        assert f"{len(limits['RETRY_WAITS']) + 1} attempts" in text, name
    readme = " ".join(_readme().split())
    assert f"{limits['KILL_GRACE_SECONDS']} seconds" in readme and "SIGKILL" in readme
    assert limits["MAX_SILENT_MINUTES"] == 45 and limits["PUSH_LIMIT_MINUTES"] == 30  # colab/job.py's


def test_docs_name_a_branch_that_can_run_jobs_and_put_the_pin_step_before_the_dev_eval():
    """p02-harness alone has P01's prompt loader and LangID as stubs, and main has no lift sync before
    P02 merges, so BRANCH must hold both. generate, judge and run-dev refuse unpinned revisions and
    read the clone's configs/project.yaml, so the pin step (a commit to BRANCH, then cell 1 again) sits
    between Gate 0 and the dev eval."""
    for name, text in _docs():
        text = " ".join(text.split())
        assert "p01-data" in text and "p02-harness" in text and "BRANCH" in text, name
    text = " ".join(_t4_section().split())
    order = [text.find(s) for s in ("Gate 0", "configs/project.yaml", "docs/decisions.md", "rerun cell 1",
                                    "lift run-dev --variant stock --dtype fp16")]
    assert -1 not in order and order == sorted(order)


def test_readme_says_what_each_fp16_gate0_record_means():
    """gate0-fp16.json failing with "fp16 failed" means T4 numbers are never reported. A
    gate0-fp16-partial.json means no valid comparison was made (the bf16 reference is not in the runs
    repo, say, or one GPU had no room for the judge): it decides nothing, and the bf16 Gate 0's outputs
    are pulled or re-run before trying again."""
    items = _items(_readme())
    partial = [item for item in items if item.startswith("`gate0-fp16-partial.json`")]
    assert len(partial) == 1, "no list item says what gate0-fp16-partial.json means"
    assert "bf16 Gate 0" in partial[0] and "never reported" not in partial[0]
    assert re.search(r"\b(pull|re-run)\b", partial[0])
    never = [item for item in items if "never reported" in item]
    assert never and all(item.startswith("`gate0-fp16.json`") for item in never)


def test_readme_keys_example_is_a_file_p04_writes():
    """P04 writes validation/keys/second_judge/{variant}_{split}.txt (docs/decisions.md, 31): an example
    naming any other file would send lift judge --keys to a FileNotFoundError."""
    keys = re.findall(r"validation/keys/[^\s`)]+", _readme())
    assert keys
    for key in keys:
        assert re.fullmatch(r"validation/keys/second_judge/[a-z0-9_.-]+_(dev|final)\.txt", key), key


def test_docs_say_the_final_push_still_runs_after_an_exit_2_from_a_job_sync_line():
    """Review 3, V21, and review 4, W16: only the preflight's or a plain lift sync push's exit 2 means that
    the final push would fail the same way. One from another lift sync line in JOB (a pull, status, a
    check, a push with more words) may come from the line's own arguments, so the final push still runs."""
    for name, text in _docs():
        items = [item for item in _items(text) if re.search(r"\bexits? 2\b", item)]
        assert [item for item in items if "final push still runs" in item
                and "plain `lift sync push`" in item and "push with more words" in item], name
        assert not re.search(r"\b(the check's or a push's|a push's exit 2)", " ".join(text.split())), name


def test_readme_says_commands_die_with_the_kernel_and_to_interrupt_cell_2_before_a_restart():
    """Review 3, V23: a command in its own process group outlived a forced kernel restart or shutdown,
    holding the GPUs. On Linux the shim ties it to the kernel; the interrupt stays the way to stop a job,
    since it runs the final push."""
    for name, text in _docs():
        assert re.search(r"[Ii]nterrupt cell 2 before restarting the kernel", " ".join(text.split())), name
    assert "PR_SET_PDEATHSIG" in _readme()


def test_readme_recovers_a_stale_cell_with_lift_restart_right_after_the_pull():
    """Review 3, V22: a stale cell's file moved aside by hand came back with the job's own lift sync pull,
    so the same refusal followed on every rerun. lift restart, as the JOB line right after lift sync
    pull, sets the cell's files aside and leaves them empty, so the push after it replaces the runs
    repo's copies (contract A). On the T4 it takes --dtype fp16, as lift prints it there (contract H):
    with the default, bf16, it would refuse the fp16 cell."""
    readme = _readme()
    assert not re.search(r"move [^.]*aside and run the job again", " ".join(readme.split()))
    blocks = [block for block in _code_blocks(readme) if any(s.startswith("lift restart") for s in block)]
    assert blocks, "no JOB example with a lift restart line"
    for lines in blocks:
        (i,) = [i for i, line in enumerate(lines) if line.startswith("lift restart")]  # one per example
        assert i > 0 and lines[i - 1] == "lift sync pull"
        assert re.fullmatch(r"lift restart --variant \S+ --split (dev|final) --sets( [\w-]+)+"
                            r" --langs( [a-z]{2})+( --judge (primary|second))? --dtype fp16", lines[i])


def test_readme_says_what_lift_restart_sets_aside_and_what_refills_it():
    """Contract H: a restart also empties the cell's AuditKIT records (an empty one is retired), and a
    whole-cell restart empties the other judges' judgements too, which the line that failed does not
    refill: lift restart's log names the line that does (the second judge's with P04's --keys list)."""
    items = [item for item in _items(_readme()) if item.startswith("`lift restart` moves")]
    assert len(items) == 1
    assert "AuditKIT records" in items[0] and "retired" in items[0]
    assert "--judge second" in items[0] and "--keys" in items[0]


def test_docs_say_the_fp16_licence_needs_both_gate0_records():
    """Review 3, V25/V19: check_dtype_licence reads gate0-fp16.json and the bf16 gate0.json. Both must
    pass for the Global revision that made the cell, with at least 200 prompts compared (contract E)."""
    passing = [item for item in _items(_readme()) if item.startswith("`gate0-fp16.json`, PASS")]
    assert len(passing) == 1
    assert "`gate0.json`" in passing[0] and "revision" in passing[0] and "200" in passing[0]
    for name, text in _docs():
        counts = [s for s in _sentences(text) if re.search(r"\bnumbers? counts? only\b", s)]
        assert counts and all("bf16 Gate 0" in s for s in counts), name


def test_docs_say_only_a_push_is_tried_again_after_the_push_limit():
    """Review 4, W18: the first cell said that any lift sync command running past PUSH_LIMIT_MINUTES is
    retried. Only a push is: a stopped pull, check or status ends the job (the final push still runs).
    "About 3 minutes" holds for a command that exits 1."""
    limit = _constants(_code_cells()[1], "PUSH_LIMIT_MINUTES")["PUSH_LIMIT_MINUTES"]
    for name, text in _docs():
        items = [item for item in _items(text) if f"{limit} minutes" in item]
        claims = [s for item in items for s in _sentences(item)
                  if f"{limit} minutes" in s and re.search(r"\b(retried|tried again)\b", s)]
        assert all("push" in s and "pull" in s for s in claims), name
        assert [item for item in items if re.search(r"stopped pull[^.]*\bends the job", item)], name
        assert all("exits 1" in s for s in _sentences(text) if "about 3 minutes" in s), name


def test_docs_say_a_kernel_that_ran_no_line_gives_a_stalled_final_push_one_try():
    """Review 4, W17: in a kernel where no line but lift sync has run, the final push has nothing of the
    kernel's to send, so a stalled one is not tried again."""
    for name, text in _docs():
        assert [s for s in _sentences(text) if "final push" in s and "one try" in s], name


def test_docs_say_a_line_that_skipped_cells_exits_3_and_the_next_lines_run():
    """Review 4, W8 (contract G): lift generate, judge and run-dev exit 3 when they are done but skipped
    some cells, each listed with its remedy. The next lines run, and the cell raises at the end."""
    for name, text in _docs():
        items = [item for item in _items(text) if re.search(r"\bexits? 3\b", item) and "run-dev" in item]
        assert [item for item in items if "next lines" in item and "remedy" in item], name
    readme = " ".join(_readme().split())
    assert not re.search(r"exits? 1 (and print|with no AuditKIT record)|A line exits 1 and lists", readme)


def test_readme_says_fp16_lines_run_only_on_a_gpu_without_native_bf16():
    """Contract L: fp16 runs exist only for the T4 licence, so lift generate, judge and run-dev refuse a
    --dtype other than bf16 on a GPU with native bf16: the T4's lines cannot be reused on Colab as they
    are."""
    sentences = [s for s in _sentences(_t4_section()) if re.search(r"\bwith native bf16\b", s)]
    assert sentences and all(re.search(r"\brefuses?\b", s) and "--dtype" in s for s in sentences)


def test_docs_leave_the_bf16_cells_of_a_variant_colab_started_out_of_an_fp16_run():
    """Review 4, W3 (the T4 side): output paths hold no dtype, so an fp16 run over a variant Colab started
    meets its bf16 cells, the reported ones. They are left out with --sets and --langs, and never
    restarted from the T4: lift restart --dtype fp16 refuses them unless --force, which would discard
    them (contract H)."""
    items = [item for item in _items(_t4_section()) if "bf16 cells" in item]
    assert len(items) == 1
    item = items[0]
    assert "`--sets`" in item and "`--langs`" in item and re.search(r"\b[Nn]ever restart", item)
    assert "lift restart --dtype fp16" in item and "`--force`" in item and "discard" in item
    about = [s for s in _sentences(_about()) if "bf16 cells" in s]
    assert about and all("--sets" in s and re.search(r"\bnever restart", s) for s in about)


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

_REAL_RUN = subprocess.run
_REAL_SLEEP = time.sleep
_PIDS = itertools.count(4000)


class _FakeProc:
    """One command as Popen starts it. It prints ``output`` (bytes, or a list of chunks ``pace`` s
    apart) and then ends with ``rc``. With ``hang`` it keeps running until signalled: by SIGINT, or with
    ``stubborn`` only by KILL. It gives up after 10 s, so a missing watchdog fails a test instead of
    hanging the suite. With ``interrupt`` its first wait raises KeyboardInterrupt, as interrupting the
    cell does. It also serves subprocess.run (communicate, with), so a cell 2 without the watchdog runs
    on these fakes and fails the watchdog tests on their asserts."""

    def __init__(self, argv, line, *, rc=0, output=b"", pace=0.0, hang=False, stubborn=False,
                 interrupt=False, straggler=False):
        self.args, self.line, self.pid, self.rc = argv, line, next(_PIDS), rc
        self.pace, self.hang, self.stubborn, self.interrupt = pace, hang, stubborn, interrupt
        self.straggler = straggler  # a child of it stays in its process group after it ends
        self.chunks = list(output) if isinstance(output, list) else [output]
        self.signals: list = []
        self.via_group: list = []  # the signals that came through os.killpg, to its process group
        self.ended = threading.Event()
        if not hang and not pace:
            self.ended.set()
        self.stdout = self

    def read1(self, size=-1):
        while self.chunks:
            if self.pace:
                _REAL_SLEEP(self.pace)  # time.sleep itself is _no_sleep while cell 2's template runs
            chunk = self.chunks.pop(0)
            if chunk:
                return chunk
        if not self.hang:
            self.ended.set()  # the output is done, and so is the command
        self.ended.wait(10)
        return b""

    read = read1

    def close(self):
        pass

    def deliver(self, sig, *, group=False):
        self.signals.append(sig)
        if group:
            self.via_group.append(sig)
        if sig != signal.SIGINT or not self.stubborn:
            self.ended.set()

    def _code(self):
        return -int(self.signals[-1]) if self.signals else self.rc

    def poll(self):
        return self._code() if self.ended.is_set() else None

    def wait(self, timeout=None):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt
        if not self.ended.wait(10 if timeout is None else timeout) and timeout is not None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self._code()

    def send_signal(self, sig):
        self.deliver(sig)

    def kill(self):
        self.deliver(KILL)

    def communicate(self, input=None, timeout=None):
        self.wait(timeout)
        return None, None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.ended.wait(10)


class _FakeHost:
    """subprocess.Popen and os.killpg as cell 2 sees them. ``outcomes`` maps a line ("lift sync push")
    to what a run of it does: an exit code, or a dict of _FakeProc options; a list is used up run by
    run (then exit 0). ``argvs`` keeps each argv as given; a line is read from the command inside the
    PDEATHSIG shim, if any."""

    def __init__(self):
        self.reset()

    def reset(self, outcomes: dict | None = None):
        self.outcomes = {line: list(o) if isinstance(o, list) else o for line, o in (outcomes or {}).items()}
        self.lines: list[str] = []
        self.argvs: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.procs: list[_FakeProc] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.argvs.append(argv)
        if argv[1:3] == ["-S", "-c"]:  # python -S -c PDEATHSIG_SHIM <kernel pid> <the command>
            argv = argv[5:]
        line = " ".join(["lift", *argv[3:]] if argv[:3] == [LIFT_PY, "-m", "lift"] else argv)
        outcome = self.outcomes.get(line, 0)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else 0
        proc = _FakeProc(argv, line, **(outcome if isinstance(outcome, dict) else {"rc": outcome}))
        self.lines.append(line)
        self.kwargs.append(kwargs)
        self.procs.append(proc)
        return proc

    def proc(self, line: str) -> _FakeProc:
        return next(p for p in self.procs if p.line == line)

    def killpg(self, pgid, sig):
        proc = next(p for p in self.procs if p.pid == pgid)
        if proc.ended.is_set() and not proc.straggler:
            raise ProcessLookupError(errno.ESRCH, "No such process")
        if sig == KILL:
            proc.straggler = False
        proc.deliver(sig, group=True)


def _fake_open(path, *args, **kwargs):
    assert path == "/kaggle/working/lift_env"
    return io.StringIO(f"LIFT_PY={LIFT_PY}\n")


def _no_sleep(seconds):
    raise AssertionError(f"the job would sleep {seconds} s for real: pass sleep=")


@pytest.fixture
def job_cell(setup_cell, monkeypatch, capsys):
    """Runs cell 1, then cell 2 as committed (its template JOB) in the same namespace, as a kernel
    does, with subprocess.Popen and os.killpg served by a _FakeHost. Returns ``run(job, outcomes,
    sleep=...)``, which calls cell 2's run_job as the cell does and returns the host; also the host,
    the template run's commands and output, and the cell's namespace."""
    for name in HIDDEN:  # no real secret of this machine may reach a fake child or a failure message
        monkeypatch.delenv(name, raising=False)
    setup_cell.run(SECRETS)
    for name, path in setup_cell.ns["LIFT_DIRS"].items():  # cell 1's os.environ changes, which the
        monkeypatch.setenv(name, path)                    # setup_cell fixture rolled back
    host = _FakeHost()
    monkeypatch.setattr(subprocess, "run", _REAL_RUN)  # cell 1's fake is done; run starts a Popen too
    monkeypatch.setattr(subprocess, "Popen", host)
    monkeypatch.setattr(os, "killpg", host.killpg, raising=False)  # never a real process group
    capsys.readouterr()
    namespace = {**setup_cell.ns, "open": _fake_open}
    with mock.patch.object(time, "sleep", _no_sleep):  # becomes run_job's default: no test really sleeps
        exec(compile(_code_cells()[1], "runner.ipynb:job", "exec"), namespace)
    template = SimpleNamespace(lines=list(host.lines), kwargs=list(host.kwargs))

    def run(job, outcomes=None, sleep=_no_sleep):
        host.reset(outcomes)
        namespace["run_job"](job, LIFT_PY, cwd=ROOT, secrets=SECRETS, sleep=sleep)
        return host

    return SimpleNamespace(run=run, host=host, ns=namespace, template=template,
                           printed=capsys.readouterr().out)


FAST_LIMITS = {"MAX_SILENT_MINUTES": 0.25 / 60, "PUSH_LIMIT_MINUTES": 0.25 / 60, "KILL_GRACE_SECONDS": 0.25,
               "TICK_SECONDS": 0.01}


@pytest.fixture
def fast_watchdog(job_cell):
    """Cell 2's watchdog limits cut to a quarter of a second. The time is real: the fakes block."""
    job_cell.ns.update(FAST_LIMITS)
    return job_cell


@pytest.fixture
def kernel(setup_cell, job_cell):
    """A fresh kernel where only cell 1 has run. Each ``run(job, outcomes, sleep=...)`` reruns cell 2 in
    it, as a person does for each job: its definitions run again, then its run_job, with the watchdog
    limits cut as in fast_watchdog. The Popen and killpg fakes are job_cell's."""
    namespace = {**setup_cell.ns, "open": _fake_open}
    source = _code_cells()[1]
    definitions = compile(source[:source.index('if "SECRETS" not in globals()')], "runner.ipynb:job", "exec")

    def run(job, outcomes=None, sleep=_no_sleep):
        exec(definitions, namespace)
        namespace.update(FAST_LIMITS)
        job_cell.host.reset(outcomes)
        namespace["run_job"](job, LIFT_PY, cwd=ROOT, secrets=SECRETS, sleep=sleep)
        return job_cell.host

    return SimpleNamespace(run=run, host=job_cell.host, ns=namespace)


def test_job_cell_refuses_to_run_before_the_setup_cell(monkeypatch):
    host = _FakeHost()
    monkeypatch.setattr(subprocess, "Popen", host)
    monkeypatch.delenv("LIFT_OUTPUT_DIR", raising=False)  # a restarted kernel has lost cell 1's env
    with pytest.raises(RuntimeError, match="setup cell"):
        exec(compile(_code_cells()[1], "runner.ipynb:job", "exec"),
             {"__name__": "__main__", "open": _fake_open})
    assert host.lines == []


def test_job_cell_refuses_when_the_kernel_lost_the_secrets(monkeypatch, tmp_path):
    host = _FakeHost()
    monkeypatch.setattr(subprocess, "Popen", host)
    monkeypatch.setenv("LIFT_OUTPUT_DIR", str(tmp_path / "outputs"))  # %reset clears SECRETS, not os.environ
    with pytest.raises(RuntimeError, match="setup cell"):
        exec(compile(_code_cells()[1], "runner.ipynb:job", "exec"),
             {"__name__": "__main__", "ROOT": ROOT, "open": _fake_open})
    assert host.lines == []


def test_job_cell_runs_the_template_then_pushes_after_each_line_and_at_the_end(job_cell):
    assert job_cell.template.lines == [CHECK, "lift sync pull", "lift info", PUSH, PUSH]
    for kwargs in job_cell.template.kwargs:
        hidden = _hidden(kwargs["env"])
        assert kwargs["cwd"] == ROOT and kwargs["env"]["PYTHONUNBUFFERED"] == "1"
        assert hidden == AS_DUMMIES  # the lift commands get cell 1's SECRETS
    assert "job OK" in job_cell.printed


def test_only_the_lift_commands_get_the_secrets(job_cell, monkeypatch):
    for name, value in KAGGLE_TOKENS.items():  # a Kaggle kernel has them; no child may
        monkeypatch.setenv(name, value)
    host = job_cell.run("nvidia-smi -L\nlift info")
    assert host.lines == [CHECK, "nvidia-smi -L", PUSH, "lift info", PUSH, PUSH]
    for line, kwargs in zip(host.lines, host.kwargs, strict=True):
        hidden = _hidden(kwargs["env"])
        assert hidden == (AS_DUMMIES if line.startswith("lift ") else {}), line


def test_job_stops_at_the_first_failure_and_still_pushes(job_cell):
    job = """
    lift sync pull
    # a comment
    lift generate --variant stock --split dev --dtype fp16
    lift judge --variant stock --split dev --dtype fp16
    """
    line = "lift generate --variant stock --split dev --dtype fp16"
    with pytest.raises(RuntimeError, match="job FAILED: 'lift generate .*' exited 1"):
        job_cell.run(job, {line: 1})
    assert job_cell.host.lines == [CHECK, "lift sync pull", line, PUSH]


def test_a_pull_conflict_stops_the_job_before_any_gpu_work(job_cell):
    with pytest.raises(RuntimeError, match="'lift sync pull' exited 3"):
        job_cell.run("lift sync pull\nlift run-dev --variant stock --dtype fp16", {"lift sync pull": 3})
    assert job_cell.host.lines == [CHECK, "lift sync pull", PUSH]


def test_an_interrupt_stops_the_running_command_and_still_pushes(job_cell):
    line = "lift run-dev --variant stock --dtype fp16"
    with pytest.raises(KeyboardInterrupt):
        job_cell.run(f"{line}\nlift info", {line: {"hang": True, "interrupt": True}})
    host = job_cell.host
    assert host.proc(line).via_group == [signal.SIGINT]  # lift may close its files first
    assert host.lines == [CHECK, line, PUSH]


def test_a_push_that_exits_1_is_retried_with_backoff(job_cell, capsys):
    # Exit 1: not everything was sent (a Hub 502, a reset connection), and a push is safe to repeat.
    waits = []
    host = job_cell.run("lift sync pull", {PUSH: [1, 1, 0]}, sleep=waits.append)
    assert host.lines == [CHECK, "lift sync pull", PUSH, PUSH, PUSH]
    assert waits == [60, 120]
    assert "job OK" in capsys.readouterr().out


def test_a_push_failing_three_times_is_reported_and_the_job_goes_on(job_cell):
    # As in colab/job.py, a failed checkpoint push does not stop the GPU work: the next push sends
    # whatever is missing, and the job still fails at the end so the person sees it.
    waits = []
    with pytest.raises(RuntimeError, match=r"checkpoint push after 'lift info' failed on all 3 attempts "
                                           r"\(the last exited 1\)"):
        job_cell.run("lift info\nlift smoke --compare-dtype fp16", {PUSH: [1, 1, 1]}, sleep=waits.append)
    assert job_cell.host.lines == [CHECK, "lift info", PUSH, PUSH, PUSH, "lift smoke --compare-dtype fp16",
                                   PUSH, PUSH]
    assert waits == [60, 120]


def test_a_push_conflict_is_reported_never_retried_and_the_job_goes_on(job_cell):
    # Exit 3: done, but the conflicting local copies went under conflicts/ in the runs repo.
    with pytest.raises(RuntimeError, match=r"checkpoint push after 'lift info' exited 3 \(conflicts"):
        job_cell.run("lift info\nlift smoke --compare-dtype fp16", {PUSH: [3]})
    assert job_cell.host.lines == [CHECK, "lift info", PUSH, "lift smoke --compare-dtype fp16", PUSH, PUSH]


def test_a_failed_final_push_fails_the_job_and_only_exit_1_is_retried(job_cell):
    # 137: the push was killed (the kernel's OOM killer, say); only lift sync's own exit 1 is retried.
    with pytest.raises(RuntimeError, match="final push exited 137"):
        job_cell.run("lift info", {PUSH: [0, 137]})
    assert job_cell.host.lines == [CHECK, "lift info", PUSH, PUSH]


def test_a_push_line_in_the_job_is_retried_and_never_stops_the_job(job_cell):
    line = "lift sync push --only outputs"
    waits = []
    with pytest.raises(RuntimeError, match=re.escape(f"{line!r} exited 3")):
        job_cell.run(f"{line}\nlift info", {line: [1, 3]}, sleep=waits.append)
    assert job_cell.host.lines == [CHECK, line, line, "lift info", PUSH, PUSH]
    assert waits == [60]


def test_an_interrupt_while_waiting_to_retry_still_pushes_at_the_end(job_cell):
    def interrupt(seconds):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        job_cell.run("lift info\nlift smoke --compare-dtype fp16", {PUSH: [1]}, sleep=interrupt)
    assert job_cell.host.lines == [CHECK, "lift info", PUSH, PUSH]


def test_job_runs_non_lift_lines_verbatim_and_adds_no_checkpoint_after_sync_lines(job_cell):
    host = job_cell.run("nvidia-smi -L\nlift sync push\n\n# done")
    # the check, nvidia-smi, its checkpoint push, the job's own push line (no checkpoint after it), the
    # final push
    assert host.lines == [CHECK, "nvidia-smi -L", PUSH, PUSH, PUSH]


# Contract: lift sync exits 1 for an error a retry may fix and 2 for one no retry fixes.


@pytest.mark.parametrize("outcomes, ran", [
    ({CHECK: 2}, [CHECK]),
    ({PUSH: 2}, [CHECK, "lift sync pull", "lift info", PUSH]),  # the checkpoint push after lift info
    ({PUSH: [0, 2]}, [CHECK, "lift sync pull", "lift info", PUSH, PUSH]),  # the job's own plain push line
])
def test_an_exit_2_of_the_preflight_or_a_plain_push_ends_the_job_at_once(job_cell, outcomes, ran):
    """Exit 2: HTTP 401 or 403 (a read token), a runs repo missing or public, a missing secret, a BRANCH
    without lift sync. No retry fixes it, so nothing is retried (no sleep is allowed here), no further
    line runs, and after the preflight's or a plain lift sync push's exit 2 there is no final push
    either: the final push is that very command, and would fail the same way."""
    job = "\n".join(["lift sync pull", "lift info", PUSH, "lift run-dev --variant stock --dtype fp16"])
    with pytest.raises(RuntimeError, match="exited 2: a configuration problem .*The job ends here, with no "
                                           "further push"):
        job_cell.run(job, outcomes)
    assert job_cell.host.lines == ran


@pytest.mark.parametrize("failing", ["lift sync status # what is left to push?",  # the comment's words
                                     "lift sync pull --only outputs/no-such-dir",
                                     "lift sync check",
                                     "lift sync pull",
                                     "lift sync push --only output/metrics",  # review 4, W16: a typo
                                     "lift sync push # push it",
                                     "lift sync push && lift info"])
def test_an_exit_2_from_a_job_sync_line_but_a_plain_push_still_tries_the_final_push(job_cell, failing):
    """Review 3, V21, and review 4, W16: such an exit 2 may come from the line's own arguments (an inline
    comment, a typo, an --only path that matches nothing, a shell's &&), a push line's included, and then
    a plain push still works. Here the checkpoint push after run-dev failed on all 3 attempts (a Hub
    blip): with no final push, run-dev's outputs would end with the session. No further line runs; an
    exit 2 returns at once, so trying costs seconds."""
    line = "lift run-dev --variant stock --dtype fp16"
    outcome = [0, 2] if failing == "lift sync pull" else 2  # the job's first line pulls too
    refused = re.escape(f"{failing!r} exited 2: a configuration problem")
    with pytest.raises(RuntimeError, match=refused) as err:
        job_cell.run(f"lift sync pull\n{line}\n{failing}\nlift info", {PUSH: [1, 1, 1], failing: outcome},
                     sleep=lambda seconds: None)
    assert job_cell.host.lines == [CHECK, "lift sync pull", line, PUSH, PUSH, PUSH, failing, PUSH]
    assert "the final push still runs" in str(err.value)


# Review 4, W8 (contract G): lift generate, judge and run-dev exit 3 when they are done but skipped some
# cells (stale, made in another precision, judged by another judge) or had an AuditKIT record refused,
# each listed with its remedy. Their exit 1 used to end the job there, so one stale cell of stock kept a
# sweep's later variants, which do not depend on it, from running in that session.


@pytest.mark.parametrize("first, then", [
    ("lift run-dev --variant stock --dtype fp16",
     "lift run-dev --variant wiseft-l0.50-adapter-aya-std-s1 --dtype fp16"),
    ("lift generate --variant stock --split dev --dtype fp16 --device cuda:0",
     "lift judge --variant stock --split dev --dtype fp16 --device cuda:1"),
    ("lift judge --variant stock --split dev --dtype fp16 --device cuda:1",
     "lift run-dev --variant stock --dtype fp16"),
])
def test_a_line_that_skipped_cells_is_reported_pushed_and_the_next_lines_run(job_cell, first, then):
    with pytest.raises(RuntimeError, match=re.escape(f"job FAILED: {first!r} exited 3: done, but")):
        job_cell.run(f"lift sync pull\n{first}\n{then}", {first: 3})
    assert job_cell.host.lines == [CHECK, "lift sync pull", first, PUSH, then, PUSH, PUSH]


@pytest.mark.parametrize("line", [
    "nvidia-smi -L",
    "lift smoke --compare-dtype fp16",
    "lift restart --variant stock --split dev --sets harmful --langs hi --dtype fp16",
])
def test_an_exit_3_of_any_other_line_still_stops_the_job(job_cell, line):
    """Only lift generate, judge and run-dev define an exit 3; another program's 3 is an error."""
    with pytest.raises(RuntimeError, match=re.escape(f"job FAILED: {line!r} exited 3")) as err:
        job_cell.run(f"{line}\nlift info", {line: 3})
    assert job_cell.host.lines == [CHECK, line, PUSH]
    assert "done, but" not in str(err.value)


def test_a_pull_that_exits_1_is_retried_like_a_push(job_cell):
    waits = []
    host = job_cell.run("lift sync pull\nlift info", {"lift sync pull": [1, 1, 0]}, sleep=waits.append)
    pulls = ["lift sync pull"] * 3
    assert host.lines == [CHECK, *pulls, "lift info", PUSH, PUSH]
    assert waits == [60, 120]


def test_a_pull_failing_on_every_attempt_stops_the_job(job_cell):
    waits = []
    with pytest.raises(RuntimeError, match="'lift sync pull' failed on all 3 attempts"):
        job_cell.run("lift sync pull\nlift run-dev --variant stock --dtype fp16", {"lift sync pull": 1},
                     sleep=waits.append)
    assert job_cell.host.lines == [CHECK, "lift sync pull", "lift sync pull", "lift sync pull", PUSH]


def test_the_preflight_runs_before_any_line_and_a_failing_one_runs_none(job_cell):
    """A read-only token used to surface at the first push, after the GPU work. The check is retried
    like a push when it exits 1; if it never succeeds, no line runs (the final push still tries to send
    what an earlier job left)."""
    waits = []
    with pytest.raises(RuntimeError, match=re.escape(f"preflight {CHECK!r} failed on all 3 attempts")):
        job_cell.run("lift run-dev --variant stock --dtype fp16", {CHECK: 1}, sleep=waits.append)
    assert job_cell.host.lines == [CHECK, CHECK, CHECK, PUSH]
    assert waits == [60, 120]


# Review R18: nothing stopped a hung line or a stalled push, so an unattended committed run held the T4
# x2 session, and the weekly quota, until Kaggle's 12-hour limit, which also skips the final push.


def test_a_stalled_push_is_stopped_after_the_push_limit_and_tried_again(fast_watchdog, capsys):
    """huggingface_hub sets no HTTP timeout: one stalled read in a push would never end."""
    waits = []
    line = "lift run-dev --variant stock --dtype fp16"
    host = fast_watchdog.run(f"lift sync pull\n{line}", {PUSH: [{"hang": True}]}, sleep=waits.append)
    stalled = host.proc(PUSH)
    assert stalled.via_group == [signal.SIGINT]
    # the checkpoint push after run-dev: stopped, then tried again; then the final push
    assert host.lines == [CHECK, "lift sync pull", line, PUSH, PUSH, PUSH]
    assert waits == [60]
    printed = capsys.readouterr().out
    assert "has run for" in printed and "job OK" in printed


def test_a_stalled_pull_is_stopped_once_and_not_tried_again(fast_watchdog, capsys):
    """Like colab/job.py (review 3, V11): a pull the watchdog stopped already held the GPUs for the push
    limit, and another try would most likely stall too. The job stops; the final push still runs."""
    with pytest.raises(RuntimeError, match="'lift sync pull' was stopped and is not tried again"):
        fast_watchdog.run("lift sync pull\nlift info", {"lift sync pull": {"hang": True}}, sleep=[].append)
    host = fast_watchdog.host
    assert host.proc("lift sync pull").via_group == [signal.SIGINT]
    assert host.lines == [CHECK, "lift sync pull", PUSH]


# Review 4, W17: a committed run whose check or first pull stalled ran no line, yet its final push was
# stopped and tried again on every stall: about 2 h of the T4 x2 session for a kernel with nothing to push.


@pytest.mark.parametrize("stalled", [CHECK, "lift sync pull"])
def test_a_kernel_that_ran_no_line_gives_a_stalled_final_push_one_try(kernel, stalled):
    waits = []
    outcomes = {stalled: {"hang": True}, PUSH: [{"hang": True}] * 3}
    with pytest.raises(RuntimeError, match="the final push was stopped and is not tried again") as err:
        kernel.run("lift sync pull\nlift run-dev --variant stock --dtype fp16", outcomes, sleep=waits.append)
    head = [CHECK] if stalled == CHECK else [CHECK, "lift sync pull"]
    assert kernel.host.lines == [*head, PUSH] and waits == []
    assert re.search(f"{re.escape(repr(stalled))}.* was stopped and is not tried again", str(err.value))


def test_a_kernel_that_ran_no_line_still_retries_a_final_push_that_exits_1(kernel):
    waits = []
    host = kernel.run("lift sync pull", {PUSH: [1, 0]}, sleep=waits.append)
    assert host.lines == [CHECK, "lift sync pull", PUSH, PUSH] and waits == [60]


@pytest.mark.parametrize("rc", [0, 1])
def test_after_a_job_that_ran_a_line_a_stalled_final_push_is_tried_again(kernel, rc):
    """The kernel keeps what an earlier job did when cell 2 runs again: that job's line, even a failed
    one, may have left outputs its pushes did not send."""
    first = "lift run-dev --variant stock --dtype fp16"
    with pytest.raises(RuntimeError, match="job FAILED") if rc else contextlib.nullcontext():
        kernel.run(first, {first: rc})
    waits = []
    stopped = re.escape(f"preflight {CHECK!r} was stopped and is not tried again")
    with pytest.raises(RuntimeError, match=stopped):
        kernel.run("lift sync pull\nlift info", {CHECK: {"hang": True}, PUSH: [{"hang": True}] * 3},
                   sleep=waits.append)
    assert kernel.host.lines == [CHECK, PUSH, PUSH, PUSH] and waits == [60, 120]


def test_the_attached_secrets_never_reach_the_saved_output(job_cell, capsys):
    """Kaggle saves the printed output with every version: a secret a command prints, even one split
    across two reads, is printed as ***; short values (a spend cap) are left alone."""
    token = SECRETS["HF_TOKEN"].encode()
    output = [b"token " + token[:6], token[6:] + b" ok\r", b"cap 150 " + SECRETS["GITHUB_TOKEN"].encode()]
    job_cell.run("lift info", {"lift info": {"output": output}})
    printed = capsys.readouterr().out
    assert "token *** ok\r" in printed and "cap 150 ***" in printed
    assert not any(value in printed for value in SECRETS.values())


def test_a_secret_across_the_megabyte_cut_is_masked_too(job_cell, capsys):
    """A megabyte without a line end is printed rather than held, minus the tail a secret could still be
    starting in: one that straddles that cut comes out masked."""
    token = SECRETS["HF_TOKEN"].encode()
    output = [b"x" * (1 << 20) + token[:6], token[6:] + b" ok\n"]
    job_cell.run("lift info", {"lift info": {"output": output}})
    printed = capsys.readouterr().out
    assert "x*** ok\n" in printed and SECRETS["HF_TOKEN"] not in printed
    assert printed.count("x") >= 1 << 20                            # nothing of the output is lost


def test_a_line_silent_past_the_limit_is_stopped_and_fails_then_the_outputs_are_pushed(fast_watchdog):
    line = "lift run-dev --variant stock --dtype fp16"
    with pytest.raises(RuntimeError, match=re.escape(f"{line!r} was stopped: no output for")):
        fast_watchdog.run(f"{line}\nlift info", {line: {"hang": True, "output": b"loading\n"}})
    host = fast_watchdog.host
    assert host.proc(line).via_group == [signal.SIGINT]
    assert host.lines == [CHECK, line, PUSH]


def test_a_command_that_ignores_sigint_gets_sigkill_after_the_grace(fast_watchdog):
    line = "lift generate --variant stock --split dev --dtype fp16"
    with pytest.raises(RuntimeError, match="was stopped: no output for"):
        fast_watchdog.run(line, {line: {"hang": True, "stubborn": True}})
    assert fast_watchdog.host.proc(line).via_group == [signal.SIGINT, KILL]
    assert fast_watchdog.ns["KILL"] == KILL


def test_nothing_of_a_stopped_commands_group_outlives_it(fast_watchdog):
    # A worker left behind by a stopped lift line would keep its GPU memory from the next line.
    line = "lift run-dev --variant stock --dtype fp16"
    with pytest.raises(RuntimeError, match="was stopped"):
        fast_watchdog.run(line, {line: {"hang": True, "straggler": True}})
    assert fast_watchdog.host.proc(line).via_group == [signal.SIGINT, KILL]


def test_a_line_that_keeps_printing_is_never_stopped(fast_watchdog):
    line = "lift run-dev --variant stock --dtype fp16"
    host = fast_watchdog.run(line, {line: {"output": [b"batch done\n"] * 20, "pace": 0.02}})  # 0.4 s
    assert host.proc(line).signals == []


def test_each_command_runs_in_its_own_process_group_with_its_output_in_the_cell(job_cell, capsys):
    word = "हिन्दी".encode()  # a character split across two reads must survive
    output = [b"lift info says ", word[:4], word[4:] + b"\n"]
    host = job_cell.run("lift info", {"lift info": {"output": output}})
    for line, kwargs in zip(host.lines, host.kwargs, strict=True):
        assert kwargs.get("start_new_session") is True, line  # so the watchdog can signal the whole group
        assert kwargs.get("stdout") is subprocess.PIPE and kwargs.get("stderr") is subprocess.STDOUT, line
        assert kwargs.get("stdin") is subprocess.DEVNULL, line
    assert "lift info says हिन्दी\n" in capsys.readouterr().out


# Review 3, V23: start_new_session takes a command out of the kernel's process group, which a forced kernel
# restart or shutdown kills, so a running line went on as an orphan, holding the GPUs, while the person
# reran the cells. On Linux every command now starts through a PR_SET_PDEATHSIG shim.


def _cell_2_definitions() -> dict:
    """Cell 2's constants and functions, without its last lines, which need cell 1 and run the job."""
    source = _code_cells()[1]
    source = source[:source.index('if "SECRETS" not in globals()')]
    namespace = {"__name__": "__main__", "child_env": lambda extra=(): {**os.environ, **dict(extra)}}
    exec(compile(source, "runner.ipynb:job", "exec"), namespace)
    return namespace


def test_on_linux_every_command_starts_through_the_pdeathsig_shim(job_cell, monkeypatch):
    monkeypatch.setitem(job_cell.ns, "PDEATHSIG", True)  # as on Kaggle's Linux kernel
    monkeypatch.setitem(job_cell.ns, "which", lambda cmd, path=None: cmd)
    host = job_cell.run("nvidia-smi -L\nlift info")
    assert host.lines == [CHECK, "nvidia-smi -L", PUSH, "lift info", PUSH, PUSH]
    shim = [sys.executable, "-S", "-c", job_cell.ns.get("PDEATHSIG_SHIM"), str(os.getpid())]
    for argv, kwargs in zip(host.argvs, host.kwargs, strict=True):
        assert argv[:5] == shim
        assert kwargs["start_new_session"] is True and "preexec_fn" not in kwargs  # the watchdog's group
    assert host.argvs[3][5:] == [LIFT_PY, "-m", "lift", "info"]  # the command itself, as without the shim


def test_commands_start_as_they_are_without_prctl_or_when_they_cannot_be_found(job_cell, monkeypatch):
    """Off Linux there is no prctl. A command that cannot be found is not wrapped either, so Popen
    raises for it in the kernel as it always did."""
    assert job_cell.ns["PDEATHSIG"] is sys.platform.startswith("linux")
    monkeypatch.setitem(job_cell.ns, "PDEATHSIG", False)
    assert job_cell.run("lift info").argvs[1] == [LIFT_PY, "-m", "lift", "info"]
    monkeypatch.setitem(job_cell.ns, "PDEATHSIG", True)
    monkeypatch.setitem(job_cell.ns, "which", lambda cmd, path=None: None)  # a typo'd command
    assert job_cell.run("nvidia-smii -L").argvs[1] == ["nvidia-smii", "-L"]


class _Killed(BaseException):
    """The end of a process that sent itself SIGKILL, as the shim tests' fake os.kill plays it."""


def _run_shim(monkeypatch, parent: int, command: list[str], *, sigint=None, prctl=True) -> list[tuple]:
    """Run PDEATHSIG_SHIM in this process as `python -S -c SHIM <parent> <command...>` would, with
    ctypes, signal.signal, signal.getsignal, os.getppid (4321), os.getpid (5555), os.kill and
    os.execvp faked. ``sigint`` is the SIGINT handler Python's startup left (default_int_handler).
    Returns the calls in order."""
    shim = _cell_2_definitions()["PDEATHSIG_SHIM"]
    calls = []

    def kill(pid, sig):
        calls.append(("kill", pid, sig))
        raise _Killed

    def fake_prctl(*args):
        calls.append(("prctl", *[getattr(arg, "value", arg) for arg in args]))  # c_ulong or int
        return 0

    libc = SimpleNamespace(prctl=fake_prctl) if prctl else SimpleNamespace()  # no prctl: AttributeError
    ctypes = types.ModuleType("ctypes")
    ctypes.CDLL = lambda name: libc
    ctypes.c_ulong = lambda value: SimpleNamespace(value=value)
    startup = signal.default_int_handler if sigint is None else sigint
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "ctypes", ctypes)
        m.setattr(sys, "path", list(sys.path))
        m.setattr(sys, "argv", ["-c", str(parent), *command])
        for name, number in (("SIGKILL", 9), ("SIGPIPE", 13), ("SIGXFSZ", 25)):  # Windows lacks them
            m.setattr(signal, name, getattr(signal, name, number), raising=False)
        m.setattr(signal, "getsignal", lambda sig: startup if sig == signal.SIGINT else signal.SIG_IGN)
        m.setattr(signal, "signal", lambda sig, handler: calls.append(("signal", int(sig), handler)))
        m.setattr(os, "getppid", lambda: 4321)
        m.setattr(os, "getpid", lambda: 5555)
        m.setattr(os, "kill", kill)
        m.setattr(os, "execvp", lambda file, args: calls.append(("execvp", file, list(args))))
        try:
            exec(compile(shim, "PDEATHSIG_SHIM", "exec"), {"__name__": "__main__"})
        except _Killed:
            pass
    return calls


def test_the_shim_ties_the_command_to_the_kernel_then_execs_it_unchanged(monkeypatch):
    command = [LIFT_PY, "-m", "lift", "run-dev", "--variant", "stock", "--dtype", "fp16"]
    calls = _run_shim(monkeypatch, 4321, command)
    assert calls[0] == ("prctl", 1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    assert calls[-1] == ("execvp", LIFT_PY, command)  # the argv as given; the environment is untouched
    # What Python's own startup changed goes back to what subprocess hands a command: SIGINT and
    # (restore_signals) SIGPIPE and SIGXFSZ at their defaults.
    numbers = [int(getattr(signal, name, number)) for name, number in (("SIGPIPE", 13), ("SIGXFSZ", 25))]
    restored = {call[1:] for call in calls if call[0] == "signal"}
    assert restored == {(number, signal.SIG_DFL) for number in (int(signal.SIGINT), *numbers)}
    assert not [call for call in calls if call[0] == "kill"]


def test_the_shim_kills_the_command_when_the_kernel_died_before_prctl(monkeypatch):
    # The shim's parent is 4321 now, not the kernel that started it (1234): the death signal will never
    # come, so it ends as if it had.
    calls = _run_shim(monkeypatch, 1234, [LIFT_PY, "-m", "lift", "info"])
    assert calls[0][0] == "prctl" and calls[-1] == ("kill", 5555, getattr(signal, "SIGKILL", 9))
    assert not [call for call in calls if call[0] == "execvp"]


def test_the_shim_keeps_an_ignored_sigint_and_runs_the_command_without_prctl(monkeypatch):
    calls = _run_shim(monkeypatch, 4321, ["nvidia-smi"], sigint=signal.SIG_IGN, prctl=False)
    assert not [call for call in calls if call[0] == "prctl" or call[:2] == ("signal", int(signal.SIGINT))]
    assert calls[-1] == ("execvp", "nvidia-smi", ["nvidia-smi"])


needs_linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux's")

# Plays a Jupyter kernel: runs cell 2's run_command on a long line, which writes its pid and sleeps.
_KERNEL = """
import json, os, shlex, subprocess, sys, time
notebook, pidfile, shim = sys.argv[1], shlex.quote(sys.argv[2]), sys.argv[3] == "1"
source = "".join(json.load(open(notebook, encoding="utf-8"))["cells"][2]["source"])
ns = {"__name__": "kernel", "child_env": lambda extra=(): dict(os.environ)}
exec(source[:source.index('if "SECRETS" not in globals()')], ns)
ns["PDEATHSIG"] = shim
line = ["sh", "-c", f"echo $$ > {pidfile}.tmp && mv {pidfile}.tmp {pidfile} && exec sleep 120"]
ns["run_command"](line, "a long line", sync=False, cwd=os.path.dirname(sys.argv[2]), env=dict(os.environ),
                  popen=subprocess.Popen, clock=time.monotonic)
"""


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@needs_linux
@pytest.mark.parametrize("shim", [True, False])
def test_a_running_command_dies_with_the_kernel_only_through_the_shim(tmp_path, shim):
    """jupyter_client starts a kernel in its own session and, at a forced restart or shutdown, kills the
    kernel's process group. A command in its own group survived that until the shim tied it to the
    kernel (the case without the shim shows the test reaches the orphan)."""
    pidfile = tmp_path / "line.pid"
    script = tmp_path / "kernel.py"
    script.write_text(_KERNEL, encoding="utf-8")
    kernel = subprocess.Popen([sys.executable, str(script), str(NOTEBOOK), str(pidfile), str(int(shim))],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    line_pid = None
    try:
        deadline = time.monotonic() + 60
        while not pidfile.exists():
            assert kernel.poll() is None and time.monotonic() < deadline, "the kernel never started the line"
            time.sleep(0.05)
        line_pid = int(pidfile.read_text(encoding="utf-8"))
        os.killpg(kernel.pid, signal.SIGKILL)
        kernel.wait(30)
        deadline = time.monotonic() + (10 if shim else 1)
        while _alive(line_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _alive(line_pid) is not shim
    finally:
        for kill, pid in ((os.killpg, kernel.pid), (os.kill, line_pid)):
            if pid is not None:
                try:
                    kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@needs_linux
@pytest.mark.parametrize("argv, rc", [
    (["sh", "-c", 'printf "%s|" "$0" "$@" "$PROBE"; exit 7', "zero", "one two", ""], 7),
    (["sh", "-c", "echo ending; kill -TERM $$"], 128 + signal.SIGTERM),
    (["cat", "/proc/self/status"], 0),  # the signal mask and dispositions the command starts with
    (["sha256sum", "/proc/self/environ"], 0),  # its whole environment, of which no value is printed
])
def test_the_real_shim_changes_nothing_a_command_sees(tmp_path, capsys, argv, rc):
    ns = _cell_2_definitions()
    env = {**os.environ, "PROBE": "a b"}
    assert ns["PDEATHSIG"] is True and ns["tied_to_kernel"](argv, env)[:2] == [sys.executable, "-S"]
    seen = {}
    for shim in (True, False):
        ns["PDEATHSIG"] = shim
        got = ns["run_command"](argv, "probe", sync=False, cwd=tmp_path, env=env, popen=subprocess.Popen,
                                clock=time.monotonic)
        out = capsys.readouterr().out
        if argv[0] == "cat":
            out = [line for line in out.splitlines() if re.match(r"Sig(Pnd|Blk|Ign|Cgt):", line)]
        seen[shim] = got, out
    assert seen[True] == seen[False]
    assert seen[True][0] == (rc, None) and seen[True][1]


@needs_linux
def test_the_real_shim_ends_a_command_whose_kernel_is_already_gone(tmp_path):
    marker = tmp_path / "ran"
    shim = _cell_2_definitions()["PDEATHSIG_SHIM"]
    # Its parent is this process, not "pid 1": as if the kernel had died before prctl.
    command = f"touch {shlex.quote(str(marker))}"
    result = subprocess.run([sys.executable, "-S", "-c", shim, "1", "sh", "-c", command], timeout=60)
    assert result.returncode == -signal.SIGKILL and not marker.exists()


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
# An install line with the constraints: "... -c <repo>/constraints.txt -e <repo>[gpu,data,eval]".
CONSTRAINED = re.compile(r" -c (?P<repo>.+)/constraints\.txt -e (?P=repo)\[gpu,data,eval\]$")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    return path


def _run_bootstrap(tmp_path: Path, *, python_too_old: bool, gpu: bool = False, fake_torch: bool = False,
                   watch: dict[str, str] | None = None, constraints: bool = True):
    """Run a copy of bootstrap.sh (so its repo is tmp/repo, with a constraints.txt unless told not to)
    with the fakes first on PATH. ``watch`` variables are put in its environment, and every fake python
    call that can see one is logged."""
    script = _write(tmp_path / "repo" / "kaggle" / "bootstrap.sh", BOOTSTRAP.read_text(encoding="utf-8"))
    if constraints:
        _write(tmp_path / "repo" / "constraints.txt", "transformers==0.0.0\n")
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
    log_file = tmp_path / "calls.log"
    log = log_file.read_text(encoding="utf-8").splitlines() if log_file.exists() else []
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


def test_bootstrap_installs_both_ways_with_the_constraints():
    # Contract: every GPU runtime installs with -c constraints.txt (the repo root), pip and uv alike.
    text = BOOTSTRAP.read_text(encoding="utf-8")
    installs = re.findall(r"^.*-m (?:uv )?pip install .*$", text, flags=re.M)
    lift_installs = [line for line in installs if "${REPO}[${EXTRAS}]" in line]
    assert len(lift_installs) == 2
    assert all('-c "$CONSTRAINTS"' in line for line in lift_installs)
    assert 'CONSTRAINTS="${REPO}/constraints.txt"' in text


@needs_bash
def test_bootstrap_in_a_uv_venv_installs_and_removes_torchao_through_uv(tmp_path):
    r = _run_bootstrap(tmp_path, python_too_old=True)
    assert r.rc == 0, r.stderr
    venv_python = f"{(tmp_path / 'venv').as_posix()}/bin/python"
    assert r.lift_env == f"LIFT_PY={venv_python}\n"
    installs = [line for line in r.log if line.startswith("image: -m uv pip install")]
    assert len(installs) == 1 and f"--python {venv_python} -c " in installs[0]
    assert CONSTRAINED.search(installs[0]) and installs[0].endswith("/repo[gpu,data,eval]")
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
    installs = [line for line in r.log if line.startswith("image: -m pip install -q -c ")]
    assert len(installs) == 1 and CONSTRAINED.search(installs[0])
    assert "image: -m pip uninstall -y -q torchao" in r.log
    assert not [line for line in r.log if "-m uv" in line]
    assert not r.torchao_left


@needs_bash
@pytest.mark.parametrize("python_too_old", [True, False])
def test_bootstrap_stops_on_a_ref_without_constraints_txt(tmp_path, python_too_old):
    # A ref without the file predates P02: it cannot run the jobs either (kaggle/README.md, BRANCH).
    r = _run_bootstrap(tmp_path, python_too_old=python_too_old, constraints=False)
    assert r.rc != 0
    assert "constraints.txt" in r.stderr and "BRANCH" in r.stderr
    assert not [line for line in r.log if " install" in line]  # nothing was installed
    assert r.lift_env is None


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
