"""``lift data ...``: the parser, its help, and every command run on the fixtures (BuildContext.default
patched to the fake context, data and splits in tmp; the repo's data/ is checked untouched)."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from fake_build import BlockingTranslator, fixture_ctx, isolate, read_sheet, repo_state, write_sheet

from lift.cli import build_parser, main
from lift.commands import data as data_command
from lift.data import build


@pytest.fixture(autouse=True, scope="module")
def repo_untouched():
    before = repo_state()
    yield
    assert repo_state() == before, "a test wrote into the repo (data/, outputs/manifests, validation/sheets)"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    return isolate(monkeypatch, tmp_path)


def use_fixtures(monkeypatch, **fixed):
    """Every ``BuildContext.default(**overrides)`` the commands make becomes the fake context."""
    def default(cls, **overrides):
        return fixture_ctx(**{**fixed, **overrides})

    monkeypatch.setattr(build.BuildContext, "default", classmethod(default))


def test_lift_help_lists_data(capsys):
    assert "data" in build_parser().format_help()
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "data" in capsys.readouterr().out


@pytest.mark.parametrize(("argv", "shown"), [
    (["data"], ["build", "stats", "spotcheck-sheets", "apply-spotchecks"]),
    (["data", "build"], ["--only", "--langs", "--skip-translation", "--cache-only", "--retranslate-failed",
                         "--fallback-engine", "hard_benign", "redteam"]),
    (["data", "stats"], ["--json"]),
    (["data", "spotcheck-sheets"], ["--langs", "--n", "--overwrite"]),
    (["data", "apply-spotchecks"], ["--file", "--dry-run"]),
])
def test_every_data_command_has_help(argv, shown, capsys):
    with pytest.raises(SystemExit) as e:
        main([*argv, "--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0
    assert [s for s in shown if s not in out] == []


@pytest.mark.parametrize("argv", [
    ["data"],
    ["data", "build", "--cache-only", "--retranslate-failed"],
    ["data", "build", "--only", "toxic"],
    ["data", "build", "--langs", "fr"],
    ["data", "build", "--fallback-engine", "bing"],
    ["data", "spotcheck-sheets", "--langs", "en"],
    ["data", "apply-spotchecks"],
])
def test_bad_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2


def test_the_command_sets_are_the_build_sets():
    assert data_command.SETS == build.SETS


def test_the_parser_imports_no_data_code():
    """``lift --help`` must work with only the core install: registering parsers loads no data module."""
    heavy = ["curatorkit", "datasets", "deep_translator", "fasttext", "huggingface_hub", "torch",
             "transformers"]
    code = ("import sys; from lift.cli import build_parser; build_parser(); "
            "print(sorted(m for m in sys.modules "
            f"if m.startswith('lift.data') or m.split('.')[0] in {heavy!r}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_build_exits_0_when_complete_and_3_when_a_file_is_skipped(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch)
    assert main(["data", "build"]) == 0
    out = capsys.readouterr().out
    assert "complete;" in out and "sft: 5 rows per candidate" in out
    assert main(["data", "build", "--only", "hard_benign", "--skip-translation"]) == 3
    out = capsys.readouterr().out
    assert "skipped hard_benign_hi: needs translation (--skip-translation)" in out and "INCOMPLETE" in out
    nothing_to_translate = ["--only", "belebele", "redteam", "--skip-translation"]
    assert main(["data", "build", *nothing_to_translate]) == 0


def test_build_exits_3_when_translation_is_blocked(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch, translator=BlockingTranslator())
    assert main(["data", "build", "--only", "harmful", "--langs", "en", "hi"]) == 3
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "skipped harmful_hi: translation blocked" in out
    assert (dirs.data / "prompts" / "harmful_en.jsonl").exists()


def test_build_exits_3_when_cache_only_misses_items(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch)
    assert main(["data", "build", "--cache-only", "--only", "hard_benign", "--langs", "en", "hi"]) == 3
    assert "a rerun can translate are missing from: hard_benign_hi" in capsys.readouterr().out


def test_stats_sheets_and_apply_on_a_built_fixture(dirs, monkeypatch, capsys, caplog):
    use_fixtures(monkeypatch)
    assert main(["data", "build"]) == 0
    capsys.readouterr()
    assert main(["data", "stats"]) == 0
    assert "harmful_hi" in capsys.readouterr().out
    assert main(["data", "stats", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["files"]["harmful_hi"]["records"] == 10

    assert main(["data", "spotcheck-sheets", "--langs", "hi", "--n", "1"]) == 0
    written = [Path(line) for line in capsys.readouterr().out.splitlines()]
    assert [p.name for p in written] == ["spotcheck_hi.csv", "spotcheck_hi_failures.csv"]
    assert all(p.parent == dirs.sheets for p in written)
    caplog.set_level(logging.ERROR)
    assert main(["data", "spotcheck-sheets", "--langs", "hi"]) == 1             # readers may be filling them
    assert "already exist" in caplog.text
    assert main(["data", "spotcheck-sheets", "--langs", "hi", "--n", "1", "--overwrite"]) == 0
    capsys.readouterr()

    sheet = dirs.sheets / "spotcheck_hi.csv"
    rows = read_sheet(sheet)
    rows[0]["verdict"] = "ok"
    write_sheet(sheet, rows)
    assert main(["data", "apply-spotchecks", "--file", str(sheet), "--dry-run"]) == 0
    assert "dry run: 1 verdicts checked, nothing written" in capsys.readouterr().out
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "harmful_hi: ok 1" in out and "1 verdicts applied; rebuilt harmful_hi" in out

    rows[1]["verdict"] = "perhaps"
    write_sheet(sheet, rows)
    caplog.clear()
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 1
    assert "verdict must be ok, fixed or bad" in caplog.text and rows[1]["translation"] not in caplog.text
