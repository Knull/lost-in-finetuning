"""``lift data ...``: the parser, its help, and every command run on the fixtures (BuildContext.default
patched to the fake context, data and splits in tmp; the repo's data/ is checked untouched)."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
from datafakes import FakeTranslator
from fake_build import BlockingTranslator, fixture_ctx, isolate, read_sheet, repo_state, write_sheet

from lift.cli import build_parser, main
from lift.commands import data as data_command
from lift.data import build, translate


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


def test_build_exits_0_when_built_including_files_skip_translation_left_out(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch)
    assert main(["data", "build"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("complete; summary in ")
    assert "sft: rows per variant (target 5, no equal-size rule)" in out
    # the laptop path while Google is blocked: nothing a rerun could add is missing
    assert main(["data", "build", "--only", "hard_benign", "--skip-translation"]) == 0
    out = capsys.readouterr().out
    assert "skipped hard_benign_hi: needs translation (--skip-translation)" in out and "INCOMPLETE" not in out
    assert out.splitlines()[-1].startswith("built; 5 files left out by --skip-translation (run without it to "
                                           "translate them); summary in ")
    nothing_to_translate = ["--only", "belebele", "redteam", "--skip-translation"]
    assert main(["data", "build", *nothing_to_translate]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("complete; ")


def test_build_exits_3_when_translation_is_blocked(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch, translator=BlockingTranslator())
    assert main(["data", "build", "--only", "harmful", "--langs", "en", "hi"]) == 3
    out = capsys.readouterr().out
    assert "skipped harmful_hi: translation blocked" in out
    assert "translation BLOCKED (Google refused this IP, HTTP 429): wait 30 minutes or more" in out
    assert "blocked: fake http_429 [stopped at item 0 of 10]" in out
    assert out.splitlines()[-1].startswith("INCOMPLETE: rerun to resume from the cache; ")
    assert (dirs.data / "prompts" / "harmful_en.jsonl").exists()


def test_build_exits_3_and_says_check_the_network_after_a_streak(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch, translator=BlockingTranslator(kind="streak"))
    assert main(["data", "build", "--only", "harmful", "--langs", "en", "hi"]) == 3
    out = capsys.readouterr().out
    assert ("translation STOPPED: 10 items in a row failed with network or service errors (not an IP block): "
            "check the network, then rerun; the cache keeps everything done") in out
    assert "refused this IP" not in out and "30 minutes" not in out and "blocked: fake streak" in out
    assert out.splitlines()[-1].startswith("INCOMPLETE: rerun to resume from the cache; ")


def test_build_exits_1_when_the_nllb_packages_are_missing(dirs, monkeypatch, capsys, caplog):
    def missing():
        raise ImportError("--fallback-engine nllb needs torch and transformers, and torch does not import "
                          "(fake). Install both (pip install -e \".[gpu]\" brings everything)")

    monkeypatch.setattr(translate, "check_nllb", missing)
    use_fixtures(monkeypatch)
    caplog.set_level(logging.ERROR)
    nllb = ["--only", "hard_benign", "--langs", "en", "hi", "--fallback-engine", "nllb"]
    assert main(["data", "build", *nllb]) == 1
    assert "needs torch and transformers" in caplog.text and capsys.readouterr().out == ""
    assert "installs the data packages" not in caplog.text            # its own install hint, no other
    assert not dirs.data.exists()

    def no_translator(cfg):                               # deep-translator missing: raised on the first call
        raise ModuleNotFoundError("No module named 'deep_translator'", name="deep_translator")

    monkeypatch.setattr(translate, "google_backend", no_translator)
    use_fixtures(monkeypatch, translator=None)
    caplog.clear()
    assert main(["data", "build", "--only", "harmful", "--langs", "en", "hi"]) == 1
    assert ("No module named 'deep_translator' (pip install -e \".[data]\" installs the data "
            "packages)") in caplog.text


def test_build_exits_3_when_cache_only_misses_items(dirs, monkeypatch, capsys):
    use_fixtures(monkeypatch)
    assert main(["data", "build", "--cache-only", "--only", "hard_benign", "--langs", "en", "hi"]) == 3
    assert "a rerun can translate are missing from: hard_benign_hi" in capsys.readouterr().out


def test_build_exits_0_and_lists_a_cell_short_of_its_frozen_split(dirs, monkeypatch, capsys):
    # R1-01: a frozen pid an open failure (an empty output) leaves out: listed with its cell, not complete;
    # a rerun cannot add it, so the exit stays 0
    use_fixtures(monkeypatch)
    assert main(["data", "build", "--only", "hard_benign", "--langs", "en"]) == 0
    first = build.load_prompts("hard_benign", "en")[0]
    use_fixtures(monkeypatch, translator=FakeTranslator(empty_for=[first.text]))
    capsys.readouterr()
    assert main(["data", "build", "--only", "hard_benign", "--langs", "en", "hi"]) == 0
    out = capsys.readouterr().out
    assert (f"hard_benign_hi is short of its frozen split ({first.split} 1), awaiting a reader's verdict on "
            f"the failures sheet: {first.pid}") in out.splitlines()
    assert out.splitlines()[-1].startswith("built; 1 fit/dev/final items await a reader's verdict (failures "
                                           "sheet: lift data spotcheck-sheets, then lift data "
                                           "apply-spotchecks); summary in ")
    assert main(["data", "stats"]) == 0
    assert f"  short of their frozen split: hard_benign_hi ({first.split} 1)" in capsys.readouterr().out


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
    assert "dry run: 1 verdicts checked (1 new, 0 unchanged), nothing written" in capsys.readouterr().out
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "harmful_hi: ok 1, fixed 0 [], bad 0 [], unchanged 0, unreviewed 0" in out
    assert "rebuilt harmful_hi: 10 records, 0 open failures, 0 errors" in out
    assert out.splitlines()[-1] == "1 verdicts applied, 0 unchanged; rebuilt harmful_hi"
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 0            # the same sheet again
    out = capsys.readouterr().out
    assert "harmful_hi: ok 0, fixed 0 [], bad 0 [], unchanged 1" in out
    assert out.splitlines()[-1] == "0 verdicts applied, 1 unchanged; rebuilt nothing"

    rows[1]["verdict"] = "perhaps"
    write_sheet(sheet, rows)
    caplog.clear()
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 1
    assert "verdict must be ok, fixed or bad" in caplog.text and rows[1]["translation"] not in caplog.text


def test_spotcheck_sheets_are_written_and_read_under_the_validation_dir(dirs, monkeypatch, capsys, tmp_path):
    # R1 post-merge item 2: a remote runtime moves validation/ with LIFT_VALIDATION_DIR, and P04's sheets
    # and lift sync follow it. So do P01's: written there, and a filled sheet found there by the name a
    # job line gives it (run from the clone, where validation/ is not that tree).
    use_fixtures(monkeypatch)
    moved = tmp_path / "moved" / "validation"
    monkeypatch.setenv("LIFT_VALIDATION_DIR", str(moved))
    assert main(["data", "build"]) == 0
    capsys.readouterr()
    assert main(["data", "spotcheck-sheets", "--langs", "hi", "--n", "1"]) == 0
    written = [Path(line) for line in capsys.readouterr().out.splitlines()]
    assert [p.name for p in written] == ["spotcheck_hi.csv", "spotcheck_hi_failures.csv"]
    assert all(p.parent == moved / "sheets" and p.is_file() for p in written)
    sheet = moved / "sheets" / "spotcheck_hi.csv"
    rows = read_sheet(sheet)
    rows[0]["verdict"] = "ok"
    write_sheet(sheet, rows)
    clone = tmp_path / "clone"
    clone.mkdir()
    monkeypatch.chdir(clone)
    names = ("spotcheck_hi.csv", "validation/sheets/spotcheck_hi.csv", r"validation\sheets\spotcheck_hi.csv")
    for name in names:
        assert main(["data", "apply-spotchecks", "--file", name, "--dry-run"]) == 0, name
        assert "dry run: 1 verdicts checked (1 new" in capsys.readouterr().out
    # A file that exists where it is named is read from there, as before.
    (clone / "spotcheck_hi.csv").write_bytes(sheet.read_bytes())
    sheet.unlink()
    assert main(["data", "apply-spotchecks", "--file", "spotcheck_hi.csv", "--dry-run"]) == 0
    assert main(["data", "apply-spotchecks", "--file", "nowhere.csv", "--dry-run"]) == 1


def test_spotcheck_sheets_exits_1_before_a_build(dirs, monkeypatch, capsys, caplog):
    use_fixtures(monkeypatch)
    caplog.set_level(logging.ERROR)
    assert main(["data", "spotcheck-sheets", "--langs", "hi"]) == 1
    assert "lift data build" in caplog.text and capsys.readouterr().out == ""
    assert not dirs.sheets.exists()


def sheet_rows(records, verdict):
    return [{"item_id": f"hi-{r.pid}", "set": r.set, "english": r.text_en, "translation": r.text,
             "verdict": verdict, "corrected": "", "note": ""} for r in records]


def test_apply_exits_3_when_the_rebuild_loses_records_and_1_without_the_cache(dirs, monkeypatch, capsys,
                                                                              caplog):
    use_fixtures(monkeypatch)
    assert main(["data", "build", "--only", "harmful", "--langs", "en", "hi"]) == 0
    sheet = dirs.sheets / "sheet.csv"
    sheet.parent.mkdir(parents=True)
    records = build.load_prompts("harmful", "hi")
    write_sheet(sheet, sheet_rows(records[:2], "ok"))
    victim = records[5].pid
    real = build._translate

    def losing(run, todo, lang, engine, **kw):
        done, failures = real(run, todo, lang, engine, **kw)
        return [r for r in done if r.pid != victim], failures

    monkeypatch.setattr(build, "_translate", losing)
    capsys.readouterr()
    caplog.set_level(logging.ERROR)
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 3
    out = capsys.readouterr().out
    assert f"LOST from harmful_hi: 1 records ['{victim}']" in out and "rebuilt harmful_hi: 9 records" in out
    # no rerun is promised: the cache check rules out a missing row, so the cause is in the reports
    assert out.splitlines()[-1] == ("INCOMPLETE: the rebuild lost records (listed above); the verdicts are "
                                    "stored. The files' reports (translation_failures) and the not applied "
                                    "lines say why")
    assert "harmful_hi: 1 records were in the file before the rebuild" in caplog.text
    assert "lift data build" not in caplog.text

    monkeypatch.setattr(build, "_translate", real)
    (dirs.data / "interim" / "translations" / "google" / "hi.jsonl").unlink()
    write_sheet(sheet, sheet_rows(records[:1], "bad"))
    caplog.clear()
    assert main(["data", "apply-spotchecks", "--file", str(sheet), "--dry-run"]) == 1
    assert "lacks the engine output of items in the files this sheet has verdicts on" in caplog.text
    assert "harmful_hi: 9 items (e.g." in caplog.text and capsys.readouterr().out == ""


def test_apply_exits_1_when_glotlid_cannot_load_and_stores_nothing(dirs, monkeypatch, capsys, caplog):
    use_fixtures(monkeypatch)
    assert main(["data", "build", "--only", "harmful", "--langs", "en", "hi"]) == 0
    sheet = dirs.sheets / "sheet.csv"
    sheet.parent.mkdir(parents=True)
    write_sheet(sheet, sheet_rows(build.load_prompts("harmful", "hi")[:1], "ok"))

    def no_fasttext(cfg=None):
        raise ModuleNotFoundError("No module named 'fasttext'", name="fasttext")

    monkeypatch.setattr("lift.lang.LangID", no_fasttext)
    use_fixtures(monkeypatch, lid=None)
    capsys.readouterr()
    caplog.set_level(logging.ERROR)
    assert main(["data", "apply-spotchecks", "--file", str(sheet)]) == 1
    assert "No module named 'fasttext'" in caplog.text and capsys.readouterr().out == ""
    assert not (dirs.data / "interim" / "translations" / "spotchecks.jsonl").exists()
