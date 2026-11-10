"""lift publish preflight on throwaway git repos: secrets, raw completions, notebook outputs and forbidden
paths, in the tracked files and in the history of every ref of origin, and any run of any model output on
this machine, whatever its label (decisions item 62; the machine must have some). Fake secrets and
outputs are assembled at run time, so no literal one is in this file (the preflight itself would flag it)."""

from __future__ import annotations

import html
import json
import subprocess
import unicodedata

import pytest

from lift import cli, paths
from lift.io import sha256_text, write_jsonl
from lift.publish import PublishError, preflight

FAKE_HF = "hf_" + "Ab3" * 12
FAKE_GITHUB = "ghp_" + "x1Y" * 13
# A made-up "harmful completion": 22 distinct words, about 260 characters.
HARMFUL = " ".join(f"preflightfake{i:02d}" for i in range(22)) + " " + "unsafe marker " * 3


def harmful_tree(text: str = HARMFUL, *, label: str = "refusal") -> None:
    """One generation judged ``label`` by the primary judge, and validation round 1 (labels here). The
    default label is a refusal: the guard reads the text, not the label."""
    from lift.publish.datasets import role_dir

    key = sha256_text(text)
    variant = "adapter-aya-std-s1"
    write_jsonl(
        paths.generations_dir() / variant / "harmful_hi_dev.jsonl",
        [{"key": key, "variant": variant, "pid": "prh-test-0001", "output": text, "hit_cap": False}],
    )
    write_jsonl(
        paths.judgements_dir() / role_dir("primary") / variant / "harmful_hi_dev.jsonl",
        [{"key": key, "pid": "prh-test-0001", "label": label, "judge_label": label, "output_sha": key}],
    )
    keys = paths.validation_dir() / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    (keys / "round1.json").write_text(json.dumps({"r1hi-a": {"key": key, "sheet": "round1_hi"}}))
    write_jsonl(
        paths.validation_dir() / "labels" / "round1_hi.jsonl", [{"item_id": "r1hi-a", "label": label}]
    )


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
    assert kinds(preflight.run(repo, history=False, windows=windows)) == ["text of a model output"]


def test_this_repo_passes_and_cli_prints_the_steps(capsys):
    harmful_tree()
    assert cli.main(["publish", "preflight", "--no-history"]) == 0
    out = capsys.readouterr().out
    assert "gh repo edit" in out and "text of the 1 model outputs on this machine" in out
    assert "NOT ENOUGH TO MAKE THE REPO PUBLIC" in out and "gh issue list" in out


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


# ------------------------------------------------------------------------- P08-R1-03 no outputs, no pass


def test_a_machine_without_outputs_fails(capsys):
    with pytest.raises(PublishError, match="no model outputs found on this machine.*lift sync pull"):
        preflight.load_windows()
    assert cli.main(["publish", "preflight", "--no-history"]) == 1
    assert "lift sync pull" in capsys.readouterr().err


@pytest.mark.parametrize("label", ["harmful", "refusal", "benign", "unclear", "anything"])
def test_an_output_is_checked_whatever_its_label(repo, label):
    """Decisions item 62: for most outputs the label is one automatic judge's, so no label clears a text."""
    harmful_tree(label=label)
    windows, n, gone = preflight.load_windows()
    assert n == 1 and windows and gone == 0
    (repo / "notes.md").write_text("quoted: " + HARMFUL[40:200] + "\n")
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False)) == [preflight.OUTPUT_TEXT]


def test_outputs_too_short_for_a_window_fail():
    harmful_tree("a short answer")
    with pytest.raises(PublishError, match="none of 60 characters or more"):
        preflight.load_windows()


def test_a_judged_output_whose_generation_is_missing_fails():
    from lift.publish.datasets import role_dir

    harmful_tree()
    other = paths.judgements_dir() / role_dir("second") / "adapter-aya-std-s1" / "redteam_hi_dev.jsonl"
    write_jsonl(
        other, [{"key": sha256_text("elsewhere"), "pid": "p", "label": "refusal", "output_sha": "0" * 64}]
    )
    with pytest.raises(
        PublishError, match=r"1 judged outputs have no generation on this machine \(e\.g\. in"
    ):
        preflight.load_windows()


def test_a_window_of_few_distinct_characters_is_no_ones_text():
    """A rule of dashes or a repeated syllable matches a tokenizer's vocabulary and every table rule: such
    a window is left out, and a text of ordinary words keeps all of its windows."""
    assert preflight.windows_of(["-" * 300, "ha " * 100, "=-" * 150]) == set()
    mixed = HARMFUL + " " + "-" * 200
    windows = preflight.windows_of([mixed])
    assert windows and all(len(set(w)) >= preflight.MIN_DISTINCT for w in windows)
    assert preflight.windows_of([HARMFUL]) <= windows | {preflight._norm(HARMFUL)[-preflight.WINDOW :]}


def test_run_loads_the_windows_from_this_machine(repo):
    harmful_tree()
    (repo / "notes.md").write_text("quoted: " + HARMFUL[40:200] + "\n")
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False)) == [preflight.OUTPUT_TEXT]


# ------------------------------------------------------------------- P08-R1-14 the whole output, any size


@pytest.mark.parametrize("part", ["start", "tail", "middle"])
def test_any_90_characters_of_a_harmful_output_are_found(repo, part):
    harmful_tree()
    norm = preflight._norm(HARMFUL)
    piece = {"start": norm[:90], "tail": norm[-90:], "middle": norm[47:137]}[part]
    (repo / "pasted.md").write_text("an issue quoting\n" + piece.replace(" ", "\n   ") + "\n")
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False)) == [preflight.OUTPUT_TEXT]


def test_a_short_quote_is_not_a_finding(repo):
    harmful_tree()
    (repo / "pasted.md").write_text(HARMFUL[:50] + "\n")
    git(repo, "add", "-A")
    assert preflight.run(repo, history=False) == []


def test_a_large_file_is_scanned_in_chunks(repo, monkeypatch):
    monkeypatch.setattr(preflight, "CHUNK_CHARS", 97)
    monkeypatch.setattr(preflight, "MAX_BYTES", 500, raising=False)  # the old size cut-off
    harmful_tree()
    filler = "\n".join(f"line {i} of a long ordinary file" for i in range(200))
    (repo / "big.txt").write_text(
        filler + "\n" + preflight._norm(HARMFUL)[100:200] + "\n" + filler + " " + FAKE_HF
    )
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False)) == ["Hugging Face token", preflight.OUTPUT_TEXT]


def test_windows_in_matches_has_window_across_every_cut(monkeypatch):
    windows = preflight.windows_of([HARMFUL])
    text = "x  y\n" * 7 + HARMFUL.replace(" ", " \n ") + " z" * 9
    monkeypatch.setattr(preflight, "CHUNK_CHARS", 1)  # every character a chunk
    for start in range(0, len(text), 13):
        for size in (1, 7, 60, 61, 200):
            monkeypatch.setattr(preflight, "CHUNK_CHARS", size)
            piece = text[start:]
            assert preflight.windows_in(piece, windows) == preflight.has_window(piece, windows)


def test_an_escaped_text_in_json_is_found(repo):
    harmful_tree()
    multiline = HARMFUL.replace(" ", "\n")  # JSON writes these newlines as \n escapes
    (repo / "cells.json").write_text(json.dumps({"note": multiline[30:200]}, ensure_ascii=True))
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False)) == [preflight.OUTPUT_TEXT]


# ------------------------------------------------------------------- P08-R1-23 notebooks and more secrets


def test_a_notebook_with_outputs_is_a_finding(repo):
    cell = {"cell_type": "code", "source": ["1 + 1"], "metadata": {}, "outputs": [], "execution_count": None}
    clean = {"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (repo / "clean.ipynb").write_text(json.dumps(clean))
    ran = {
        **clean,
        "cells": [{**cell, "execution_count": 3, "outputs": [{"output_type": "stream", "text": ["2"]}]}],
    }
    (repo / "ran.ipynb").write_text(json.dumps(ran))
    git(repo, "add", "-A")
    found = preflight.run(repo, history=False, windows=set())
    assert [(f.path, f.kind) for f in found] == [("ran.ipynb", preflight.NOTEBOOK_OUTPUTS)]


def test_google_oauth_and_kaggle_env_secrets(repo):
    lines = [
        "token = " + "ya29" + "." + "a0Af" * 10,
        "secret = " + "GOCSPX" + "-" + "Zq9" * 9,
        "KAGGLE_KEY" + "=" + "0123456789abcdef" * 2,
    ]
    (repo / "env.txt").write_text("\n".join(lines) + "\n")
    git(repo, "add", "-A")
    assert kinds(preflight.run(repo, history=False, windows=set())) == [
        "Google OAuth client secret",
        "Google OAuth token",
        "Kaggle API key (KAGGLE_KEY)",
    ]


# --------------------------------------------------------------------- P08-R1-13 every ref of the remote


def _origin_with_hidden_refs(tmp_path, repo):
    """An origin holding the repo's history plus a branch and a PR ref this clone never fetched, each
    with a secret."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-q", "origin", "HEAD:refs/heads/main")
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(origin), str(other))
    (other / "branch.txt").write_text(f"token = {FAKE_HF}\n")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "branch")
    git(other, "push", "-q", "origin", "HEAD:refs/heads/agent-branch")
    (other / "pr.txt").write_text(f"token = {FAKE_GITHUB}\n")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "pr")
    git(other, "push", "-q", "origin", "HEAD:refs/pull/7/head")
    return origin


def test_remote_branches_and_pr_refs_are_fetched_and_scanned(tmp_path, repo):
    _origin_with_hidden_refs(tmp_path, repo)
    assert preflight.run(repo, windows=set()) == []  # local refs alone look clean
    assert preflight.fetch_public_refs(repo) is True
    found = preflight.run(repo, windows=set())
    assert {(f.path, f.kind) for f in found} == {
        ("branch.txt", "Hugging Face token"),
        ("pr.txt", "GitHub token"),
    }


def test_a_failed_fetch_refuses(tmp_path, repo):
    git(repo, "remote", "add", "origin", str(tmp_path / "no-such-remote.git"))
    with pytest.raises(PublishError, match="could not fetch"):
        preflight.fetch_public_refs(repo)


def test_no_remote_means_nothing_to_fetch(repo):
    assert preflight.fetch_public_refs(repo) is False


# ------------------------------------------------------- P08-R2-04 unicode forms, case and format characters


def _spanish():
    text = " ".join(f"señal{i:02d} acción según añadió" for i in range(8))  # made-up, accented
    return unicodedata.normalize("NFD", text), unicodedata.normalize("NFC", text)


def _nukta():
    # Precomposed nukta letters (U+0958, U+095E): NFC decomposes them (composition exclusions).
    text = " ".join(f"क़फ़क{i:02d} हिंदी" for i in range(14))
    return text, unicodedata.normalize("NFC", text)


def _zwsp_spaces():
    return HARMFUL, HARMFUL.replace(" ", "​")


def _zwsp_inside():
    return HARMFUL, "".join(c + ("​" if i % 5 == 4 else "") for i, c in enumerate(HARMFUL))


def _format_chars():
    return HARMFUL, "".join(c + ("­" if i % 7 == 3 else "") for i, c in enumerate(HARMFUL)).replace(
        "fake", "fa‍ke"
    )


def _upper():
    return HARMFUL, HARMFUL.upper()


def _fullwidth():
    return HARMFUL, "".join(chr(ord(c) + 0xFEE0) if c.isalnum() else c for c in HARMFUL)


def _html():
    text = " ".join(f'it\'s step{i:02d} & "mix" <b>' for i in range(8))
    return text, html.escape(text)


UNICODE_CASES = {
    "nfc copy of nfd": _spanish,
    "nfc copy of precomposed nukta": _nukta,
    "zero-width spaces for spaces": _zwsp_spaces,
    "zero-width spaces inside words": _zwsp_inside,
    "soft hyphens and joiners": _format_chars,
    "upper case": _upper,
    "fullwidth": _fullwidth,
    "html entities": _html,
}


@pytest.mark.parametrize("case", sorted(UNICODE_CASES))
def test_a_normalised_copy_of_a_harmful_output_is_found(case):
    output, copy = UNICODE_CASES[case]()
    assert output != copy
    windows = preflight.windows_of([output])
    found = preflight.scan_text("pasted from an editor:\n" + copy + "\n", "notes.html", "tree", windows)
    assert kinds(found) == [preflight.OUTPUT_TEXT]


@pytest.mark.parametrize("case", sorted(UNICODE_CASES))
def test_the_normalised_copies_are_found_across_every_chunk_cut(case, monkeypatch):
    output, copy = UNICODE_CASES[case]()
    windows = preflight.windows_of([output])
    text = "x  y\n" * 5 + copy + " z" * 9
    for size in (1, 7, 60, 61, 200):
        monkeypatch.setattr(preflight, "CHUNK_CHARS", size)
        for start in range(0, len(text), 11):
            piece = text[start:]
            assert preflight.windows_in(piece, windows) == preflight.has_window(piece, windows)
        assert preflight.windows_in(text, windows) or case == "html entities"  # html: scan_text unescapes


def test_normalising_never_makes_a_different_text_match():
    windows = preflight.windows_of([HARMFUL])
    other = HARMFUL.replace("fake", "real")
    assert not preflight.has_window(other.upper(), windows)


# ----------------------------------------------- P08-R2-03 a harmful output that repeats its prompt


PROMPT = " ".join(f"promptfake{i:02d}" for i in range(11))  # a made-up prompt, about 130 characters
PROMPT_EN = " ".join(f"englishsource{i:02d}" for i in range(8))


def _prompt_set():
    row = {
        "pid": "prh-test-0001",
        "lang": "hi",
        "set": "harmful",
        "split": "dev",
        "text": PROMPT,
        "text_en": PROMPT_EN,
        "source": "polyrefuse",
        "translation": {"engine": "google-cloud-v2"},
    }
    write_jsonl(paths.prompts_dir() / "harmful_hi.jsonl", [row])
    return row


def test_a_harmful_output_that_repeats_its_prompt_does_not_flag_the_prompt():
    _prompt_set()
    harmful_tree("Sure. " + PROMPT + ": " + HARMFUL + " In English: " + PROMPT_EN.upper())
    windows, n, _gone = preflight.load_windows()
    assert n == 1
    for text in (PROMPT, PROMPT_EN, json.dumps({"text": PROMPT, "text_en": PROMPT_EN})):
        assert preflight.scan_text(text, "harmful/hi.jsonl", "staged", windows) == []
    # the completion's own text is still caught: its continuation, and a run across the echo
    assert kinds(preflight.scan_text(HARMFUL[:95], "a.md", "staged", windows)) == [preflight.OUTPUT_TEXT]
    across = PROMPT[-40:] + ": " + HARMFUL[:55]
    assert kinds(preflight.scan_text(across, "b.md", "staged", windows)) == [preflight.OUTPUT_TEXT]


def test_without_the_prompt_files_every_window_stays():
    harmful_tree("Sure. " + PROMPT + ": " + HARMFUL)
    windows, _n, _gone = preflight.load_windows()
    assert kinds(preflight.scan_text(PROMPT, "p.md", "staged", windows)) == [preflight.OUTPUT_TEXT]


# ---------------------------------------------------- P08-R2-13 judgement and utility records are raw


def test_judgement_and_utility_records_are_raw_records(repo):
    judgement = {
        "key": "k",
        "variant": "v",
        "judge": "anthropic/m@x",
        "judge_label": "harmful",
        "label": "harmful",
    }
    utility = {"pid": "p", "variant": "v", "judge": "j", "order_ab": "A", "order_ba": "B", "score": 1.0}
    (repo / "judged.jsonl").write_text(
        json.dumps({**judgement, "raw": "a judge's reasoning, quoting"}) + "\n"
    )
    (repo / "utility.jsonl").write_text(json.dumps(utility) + "\n")
    (repo / "labels.jsonl").write_text(json.dumps({**judgement, "raw": ""}) + "\n")  # a public label row
    git(repo, "add", "-A")
    found = preflight.run(repo, history=False, windows=set())
    assert {(f.path, f.kind) for f in found} == {
        ("judged.jsonl", "raw completion record"),
        ("utility.jsonl", "raw completion record"),
    }


# ------------------------------------------------------------------- P08-R2-16 blobs are streamed


def test_history_blobs_are_streamed_not_buffered(repo, monkeypatch):
    for i in range(5):
        (repo / f"f{i}.txt").write_text(f"version {i}\n" + ("token = " + FAKE_HF if i == 2 else "") + "\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", f"c{i}")
    for i in range(5):
        (repo / f"f{i}.txt").write_text("clean\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "clean")
    real = preflight._git

    def no_buffered_cat_file(*args, **kwargs):
        assert "cat-file" not in args, "the blobs were read into one buffer"
        return real(*args, **kwargs)

    monkeypatch.setattr(preflight, "_git", no_buffered_cat_file)
    found = preflight.run(repo, windows=set())
    assert [(f.path, f.kind) for f in found] == [("f2.txt", "Hugging Face token")]
    blobs = preflight._history_blobs(repo)
    assert {sha for sha, _kind, _body in preflight._cat_blobs(repo, blobs)} == set(blobs)


@pytest.mark.parametrize(
    "line,flagged",
    [
        ('"out_path": "/home/ana/lexsi/outputs/x.jsonl"', True),
        ('"out_path": "/Users/ana/x"', True),
        ('"cache": "/root/.cache/huggingface"', True),
        ("written under /tmp/claude-1000/xbranch2/repo/outputs", True),
        ('"out_path": "C:\\\\Users\\\\ana\\\\lift\\\\x.jsonl"', True),
        ("D:/runs/lift/outputs", True),
        ("see https://huggingface.co/datasets/org/x and file:x", False),
        ("outputs/metrics/selection.json and configs/project.yaml", False),
        ("a/home/b, ratio 1:2, 10:30", False),
    ],
)
def test_the_public_text_scan_flags_absolute_local_paths(tmp_path, line, flagged):
    """XB2-04: a home dir, /tmp or a drive letter in a staged public file is a finding (the staging
    rewrites what it can to repo-relative paths first); a tokenizer's vocabulary is not read for them."""
    (tmp_path / "f.json").write_text(line + "\n", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text('{"vocab": {"C:\\\\": 1, "/home/": 2, "/tmp/": 3}}')
    found = preflight.scan_folder(tmp_path, set())
    assert any(f.path == "f.json" and f.kind == preflight.LOCAL_PATH_KIND for f in found) is flagged
    assert not any(f.path == "tokenizer.json" for f in found)
