"""lift publish preflight on throwaway git repos: secrets, raw completions and forbidden paths, in the
tracked files and in the history. Fake secrets are assembled at run time, so no literal one is in this
file (the preflight itself would flag it)."""

from __future__ import annotations

import json
import subprocess

import pytest

from lift import cli
from lift.publish import preflight

FAKE_HF = "hf_" + "Ab3" * 12
FAKE_GITHUB = "ghp_" + "x1Y" * 13


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@example.org", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    (r / "README.md").write_text("# clean\n")
    (r / "data" / "splits").mkdir(parents=True)
    (r / "data" / "splits" / "harmful.json").write_text("{}\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    return r


def kinds(findings):
    return sorted({f.kind for f in findings})


def test_clean_repo_passes(repo):
    assert preflight.run(repo, windows=set()) == []


def test_secret_in_tree_and_history(repo):
    (repo / "notes.txt").write_text(f"token = {FAKE_HF}\nother = {FAKE_GITHUB}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "oops")
    found = preflight.run(repo, windows=set())
    assert kinds(found) == ["GitHub token", "Hugging Face token"]
    assert all(FAKE_HF not in str(f) for f in found)  # the match is never shown
    (repo / "notes.txt").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "remove")
    assert preflight.run(repo, history=False, windows=set()) == []
    history = preflight.run(repo, windows=set())
    assert history and all(f.where.startswith("history") for f in history)


def test_allow_mark_skips_a_documented_fake(repo):
    (repo / "t.py").write_text(f'FAKE = "{FAKE_HF}"  # {preflight.ALLOW_MARK}\n')
    git(repo, "add", "-A")
    assert preflight.run(repo, history=False, windows=set()) == []


def test_raw_completion_records_and_paths(repo):
    row = {"key": "k", "variant": "adapter-aya-std-s1", "pid": "p", "output": "some completion"}
    (repo / "dump.jsonl").write_text(json.dumps(row) + "\n")
    audit = {"actual_output": "full text of an answer"}  # preflight: allow (fake)
    (repo / "record.json").write_text(json.dumps(audit, indent=1))
    (repo / "outputs").mkdir()
    (repo / "outputs" / "x.txt").write_text("hello\n")
    (repo / "data" / "prompts.jsonl").write_text("{}\n")
    git(repo, "add", "-f", "-A")
    found = preflight.run(repo, history=False, windows=set())
    assert kinds(found) == ["file that must never be committed", "raw completion record"]
    paths = {f.path for f in found}
    assert {"dump.jsonl", "record.json", "outputs/x.txt", "data/prompts.jsonl"} <= paths


def test_redacted_records_are_fine(repo):
    (repo / "record.json").write_text(json.dumps({"actual_output": "[redacted sha256:ab]"}))
    git(repo, "add", "-A")
    assert preflight.run(repo, history=False, windows=set()) == []


def test_text_of_a_harmful_output_is_found(repo):
    harmful = "word " * 30
    windows = {preflight._norm(harmful)[: preflight.WINDOW]}
    (repo / "pasted.md").write_text("an issue quoting\n" + harmful.replace(" ", "\n  ") + "\n")
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False, windows=windows)) == [
        "text of an output labelled harmful"
    ]


def test_this_repo_passes_and_cli_prints_the_steps(capsys, monkeypatch):
    monkeypatch.setattr(preflight, "harmful_windows", lambda: set())
    assert cli.main(["publish", "preflight", "--no-history"]) == 0
    assert "gh repo edit" in capsys.readouterr().out


def test_a_fake_marked_later_is_allowed_in_the_history_too(repo):
    line = f'FAKE = "{FAKE_HF}"'
    (repo / "t.py").write_text(line + "\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fake")
    assert kinds(preflight.run(repo, windows=set())) == ["Hugging Face token"]
    (repo / "t.py").write_text(f"{line}  # {preflight.ALLOW_MARK}\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "mark")
    assert preflight.run(repo, windows=set()) == []
