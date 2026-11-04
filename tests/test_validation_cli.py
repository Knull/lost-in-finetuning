"""``lift validate ...`` through the real entry point (lift.cli.main) on fake worlds (P04).

The help of every step; round 1 from the blind sheets to the judge decision; the second-judge
sample and round 2; the utility check; the decision when the frontier judge wins; and the errors a
person running the commands can hit, which must end in one "error:" line, never a traceback.
Nothing a command prints may hold a prompt or a response (CLAUDE.md rule 6), and no command may
edit configs/project.yaml or docs/decisions.md: a person makes that change.
"""

from __future__ import annotations

import copy
import csv
import os
import re
import textwrap
from datetime import date
from functools import partial
from pathlib import Path

import pytest
import yaml
from validation_fakes import (
    FRONTIER_ID,
    LANGS,
    World,
    build_world,
    fill_sheet,
    read_csv,
    read_round,
    truth,
    use_world,
    write_generations,
    write_judgements,
    write_verdicts,
)

from lift import config, paths
from lift.cli import build_parser, main
from lift.io import read_jsonl, sha256_file
from lift.validation.store import read_json

CANDIDATE = "adapter-aya-std-s1"
REPAIRED = "wiseft-l0.50-adapter-aya-std-s1"
ROUND1 = {"dev_variants": ("stock", CANDIDATE), "final_variants": ()}
ROUND2 = {"dev_variants": ("stock", CANDIDATE, REPAIRED), "final_variants": ("stock", CANDIDATE, REPAIRED)}
NO_GIT = "0" * 40

# The options each step documents (its module docstring, the P04 prompt, and its own flags).
STEP_OPTIONS = {
    "sample": ("--round", "--seed", "--variants", "--hindi-overlap", "--no-hindi-overlap", "--check",
               "--force", "--no-prompt-check"),
    "import": ("--round", "--lang", "--file", "--annotator"),
    "score": ("--round", "--qwen3guard-judge", "--frontier-judge"),
    "decide-judge": ("--dry-run", "--allow-incomplete", "--force", "--qwen3guard-judge", "--frontier-judge"),
    "second-judge-sample": ("--fraction", "--seed", "--no-validated", "--no-prompt-check"),
    "utility-sheet": ("--variant", "--split", "--seed", "--force", "--no-prompt-check"),
    "utility-import": ("--file", "--annotator"),
    "utility-score": ("--verdicts", "--judge-id"),
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Writes go under tmp_path even in a test without a world; git is not asked for its sha."""
    for var, name in (("LIFT_DATA_DIR", "data"), ("LIFT_OUTPUT_DIR", "outputs"),
                      ("LIFT_VALIDATION_DIR", "validation")):
        monkeypatch.setenv(var, str(tmp_path / "nowhere" / name))
    monkeypatch.setattr("lift.manifest.git_sha", lambda: NO_GIT)


def cli(capsys, *argv) -> tuple[int, str]:
    """``lift validate *argv``: the exit code and everything it printed."""
    code = main(["validate", *map(str, argv)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def ok(capsys, printed: list[str], *argv) -> str:
    """``lift validate *argv``, which must succeed; what it printed is kept for the privacy check."""
    code, out = cli(capsys, *argv)
    printed.append(out)
    assert code == 0, out
    return out


def assert_error(code: int, printed: str) -> None:
    assert code == 1, printed
    assert any(line.startswith("error:") for line in printed.splitlines()), printed
    assert "Traceback" not in printed


def assert_private(world: World, *texts: str) -> None:
    """No prompt or response of the world in ``texts``, whole or cut short."""
    text = "\n".join(texts)
    secret = {g.output for recs in world.generations.values() for g in recs}
    for path in (world.data / "prompts").glob("*.jsonl"):
        secret |= {row["text"] for row in read_jsonl(path)}
    leaked = sum(s in text for s in secret)
    assert not leaked, f"{leaked} prompt or response texts were printed"
    assert not re.search(r"\b(question|answer) [0-9a-f]{6}", text)


def metrics_text(world: World) -> str:
    root = world.outputs / "metrics"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(root.rglob("*")) if p.is_file())


def tree(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under ``root`` with its size and modification time."""
    found, folders = {}, [root]
    while folders:
        try:
            with os.scandir(folders.pop()) as entries:
                for entry in entries:
                    if entry.is_dir():
                        folders.append(Path(entry.path))
                    else:
                        found[entry.path] = (entry.stat().st_size, entry.stat().st_mtime_ns)
        except FileNotFoundError:
            continue
    return found


def real_repo_files() -> dict[str, tuple[int, str]]:
    """The repo's locked config and decisions log, and its own validation/ and outputs/metrics/."""
    root = paths.REPO_ROOT
    files = [root / "configs" / "project.yaml", root / "docs" / "decisions.md"]
    for folder in (root / "validation", root / "outputs" / "metrics"):
        if folder.is_dir():
            files += [p for p in folder.rglob("*") if p.is_file()]
    return {str(p): (p.stat().st_mtime_ns, sha256_file(p)) for p in files}


def sheet_path(world: World, round_no: int, lang: str, slot: str = "") -> Path:
    return world.validation / "sheets" / f"validation_round{round_no}_{lang}{slot}.csv"


def sheet_rows(path: Path) -> list[dict[str, str]]:
    """The data rows of a sheet as written: a comment line, the column names, then the items."""
    rows = read_csv(path)
    return [dict(zip(rows[1], row, strict=True)) for row in rows[2:]]


def write_csv(path: Path, rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    return path


def true_labels(world: World, round_no: int) -> dict[str, str]:
    """What perfect readers write: item_id -> the true label of the item's generation."""
    doc = read_round(world.validation, round_no)
    return {it["item_id"]: truth(it["key"]) for it in doc["items"]}


def import_round(world: World, capsys, round_no: int, sent: Path) -> str:
    """Every main sheet of the round filled in by a perfect reader and imported."""
    sent.mkdir(exist_ok=True)
    labels = true_labels(world, round_no)
    printed = ""
    for lang in LANGS:
        src = sheet_path(world, round_no, lang)
        code, out = cli(capsys, "import", "--round", round_no, "--lang", lang,
                        "--file", fill_sheet(src, sent / src.name, labels))
        assert code == 0, out
        printed += out
    return printed


def patch_project(monkeypatch, changes: dict) -> None:
    """configs/project.yaml as if the ``changes`` (dotted key -> value) were made in it."""
    real_load = config.load
    project = copy.deepcopy(real_load("project"))
    for dotted, value in changes.items():
        *parents, last = dotted.split(".")
        node = project
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value

    def load(name: str = "project") -> dict:
        return project if name == "project" else real_load(name)

    monkeypatch.setattr(config, "load", load)


def judges_blocks(text: str) -> list[dict]:
    """The ``judges`` mapping of every YAML block printed in ``text`` (indented or not)."""
    lines = text.splitlines()
    found = []
    for i, line in enumerate(lines):
        if line.strip() != "judges:":
            continue
        indent = len(line) - len(line.lstrip())
        end = i + 1
        while end < len(lines) and lines[end].strip() and len(lines[end]) - len(lines[end].lstrip()) > indent:
            end += 1
        try:
            doc = yaml.safe_load(textwrap.dedent("\n".join(lines[i:end])))
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict) and isinstance(doc.get("judges"), dict):
            found.append(doc["judges"])
    return found


# --- help and usage ---------------------------------------------------------------------------------


def test_validate_help_lists_every_step(capsys):
    assert "validate" in build_parser().format_help()
    with pytest.raises(SystemExit) as exit_:
        main(["validate", "--help"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    for step in STEP_OPTIONS:
        assert step in out


@pytest.mark.parametrize("step", list(STEP_OPTIONS))
def test_the_help_of_each_step_lists_its_options(step, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["validate", step, "--help"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    missing = [option for option in STEP_OPTIONS[step] if option not in out]
    assert not missing


@pytest.mark.parametrize("argv", [
    ["validate"],                                                          # no step
    ["validate", "sample"],                                                # --round is required
    ["validate", "sample", "--round", "3"],                                # there are rounds 1 and 2
    ["validate", "import", "--round", "1", "--lang", "fr", "--file", "x.csv"],   # not a validated language
    ["validate", "import", "--round", "1", "--lang", "hi"],               # --file is required
    ["validate", "utility-sheet"],                                         # --variant is required
], ids=["no step", "no round", "round 3", "lang fr", "no file", "no variant"])
def test_usage_errors_stop_before_anything_runs(argv, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(argv)
    assert exit_.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err and "Traceback" not in err
    assert tree(tmp_path) == {}


# --- end to end -------------------------------------------------------------------------------------


def test_round_1_from_the_blind_sheets_to_the_judge_decision(tmp_path, monkeypatch, capsys):
    world = use_world(monkeypatch, build_world(tmp_path / "world", utility=False, **ROUND1))
    sent = tmp_path / "sent"
    sent.mkdir()
    real = real_repo_files()
    printed: list[str] = []
    run = partial(ok, capsys, printed)

    before = tree(world.root)
    run("sample", "--round", 1, "--check")
    assert tree(world.root) == before                    # a dry run writes nothing

    out = run("sample", "--round", 1)
    assert (world.validation / "keys" / "round1.json").is_file()
    for lang in LANGS:                                   # one sheet per language, and the operator
        assert sheet_path(world, 1, lang).is_file()      # is told which file is which
        assert sheet_path(world, 1, lang).name in out
    drawn_once = tree(world.root)
    run("sample", "--round", 1)                          # the same inputs again: nothing changes
    assert tree(world.root) == drawn_once

    # The readers send their sheets back. The Hindi reader skips one item on purpose, the Spanish
    # sheet comes back as a Google Sheets download, and the Chinese reader edited a response cell.
    labels = true_labels(world, 1)
    skipped = sheet_rows(sheet_path(world, 1, "hi"))[0]["item_id"]
    edited = sheet_rows(sheet_path(world, 1, "zh"))[0]["item_id"]
    imported = {}
    for lang in LANGS:
        src = sheet_path(world, 1, lang)
        dst = fill_sheet(src, sent / src.name, {**labels, skipped: ""}, notes={skipped: "skipped"},
                         google_export=lang == "es")
        if lang == "zh":
            rows = read_csv(dst)
            col = rows[1].index("response")
            next(r for r in rows[2:] if r[0] == edited)[col] += " (edited)"
            write_csv(dst, rows)
        imported[lang] = run("import", "--round", 1, "--lang", lang, "--file", dst)
    hi = list(read_jsonl(world.validation / "labels" / "round1_hi.jsonl"))
    assert len(hi) == 59 and skipped not in {r["item_id"] for r in hi}
    assert "1 skipped" in imported["hi"]                 # excluded, and counted
    assert edited in imported["zh"]                      # the edited row is named by its id
    for lang in ("zh", "es", "en"):
        assert len(list(read_jsonl(world.validation / "labels" / f"round1_{lang}.jsonl"))) == 40

    out = run("score", "--round", 1)
    metrics = world.outputs / "metrics" / "validation"
    scored = read_json(metrics / "round1.json")
    assert (metrics / "round1.md").is_file()
    assert set(scored["judges"]) == {"qwen3guard_strict", "qwen3guard_loose", "frontier"}
    for view, judge in scored["judges"].items():
        assert view in out
        assert set(judge["langs"]) == set(LANGS)
    strict = scored["judges"]["qwen3guard_strict"]["pooled"]
    assert (strict["n"], strict["agreement"]) == (179, 1.0)
    assert read_json(metrics / "gate1.json")["status"] == "pass"

    decision = world.outputs / "metrics" / "judge_decision.json"
    run("decide-judge", "--dry-run")
    assert not decision.exists()
    run("decide-judge")
    decided = read_json(decision)
    assert decided["n"] == 179 and decided["switch_primary"] is False
    assert real_repo_files() == real
    assert_private(world, *printed, metrics_text(world))


def test_second_judge_sample_and_round_2_after_round_1(tmp_path, monkeypatch, capsys):
    world = use_world(monkeypatch, build_world(tmp_path / "world", judgements=False, utility=False, **ROUND2))
    printed: list[str] = []
    run = partial(ok, capsys, printed)

    run("sample", "--round", 1)
    printed.append(import_round(world, capsys, 1, tmp_path / "sent"))

    run("second-judge-sample")
    lists = world.validation / "keys" / "second_judge"
    random_keys = set((lists / "random.txt").read_text(encoding="utf-8").split())
    round1 = {it["key"] for it in read_round(world.validation, 1)["items"]}
    assert round1 <= set((lists / "all.txt").read_text(encoding="utf-8").split())     # every validated item
    outputs = [g for recs in world.generations.values() for g in recs]
    assert len(random_keys) == len(outputs) // 10                                       # plus a 10% sample
    assert random_keys <= {g.key for g in outputs}

    run("sample", "--round", 2)
    for lang in LANGS:
        assert sheet_path(world, 2, lang).is_file()

    run("second-judge-sample")                         # again, now that round 2 is drawn
    round2 = {it["key"] for it in read_round(world.validation, 2)["items"]}
    assert round1 | round2 <= set((lists / "all.txt").read_text(encoding="utf-8").split())
    assert set((lists / "random.txt").read_text(encoding="utf-8").split()) == random_keys   # seeded

    # Every round drawn so far, pooled. No judge has run in this world, so the agreement is pending.
    run("score")
    scored = read_json(world.outputs / "metrics" / "validation" / "all.json")
    assert scored["rounds"] == [1, 2]
    assert scored["judges"]["qwen3guard_strict"]["pooled"]["n"] == 180    # round 2 is not labelled yet
    assert_private(world, *printed, metrics_text(world))


def test_utility_sheet_import_and_score(tmp_path, monkeypatch, capsys):
    # Judgement files sit under outputs/ too: the verdict search must not take them for verdicts.
    world = use_world(monkeypatch, build_world(tmp_path / "world", **ROUND1))
    sent = tmp_path / "sent"
    sent.mkdir()
    printed: list[str] = []
    run = partial(ok, capsys, printed)

    out = run("utility-sheet", "--variant", CANDIDATE)
    sheet = world.validation / "sheets" / "validation_utility_hi.csv"
    assert sheet.name in out and len(sheet_rows(sheet)) == 60     # 60 of the 70 dev pairs

    # The reader prefers the candidate, calls a tie or prefers stock, in turn; the judge agrees
    # with the reader on all but the first 6 pairs.
    items = {it["item_id"]: it for it in read_json(world.validation / "keys" / "utility.json")["items"]}
    prefs, scores = {}, {}
    for i, item_id in enumerate(sorted(items)):
        it = items[item_id]
        outcome = ("win", "tie", "loss")[i % 3]
        stock_side = "B" if it["candidate_is"] == "A" else "A"
        prefs[item_id] = {"win": it["candidate_is"], "tie": "tie", "loss": stock_side}[outcome]
        judged = outcome if i >= 6 else {"win": "loss", "tie": "win", "loss": "win"}[outcome]
        scores[it["pid"]] = {"win": 1.0, "tie": 0.5, "loss": 0.0}[judged]
    filled = fill_sheet(sheet, sent / sheet.name, prefs, column="preference")
    run("utility-import", "--file", filled)
    assert len(list(read_jsonl(world.validation / "labels" / "utility_hi.jsonl"))) == 60

    code, out = cli(capsys, "utility-score", "--verdicts", tmp_path / "no_such_verdicts.jsonl")
    printed.append(out)
    assert_error(code, out)
    write_verdicts(world, CANDIDATE, scores)
    run("utility-score")
    scored = read_json(world.outputs / "metrics" / "validation" / "utility.json")
    assert scored["n_judged"] == 60
    assert scored["agreement"] == pytest.approx(54 / 60)
    assert_private(world, *printed, metrics_text(world))


# --- the judge decision when the frontier judge wins ------------------------------------------------


@pytest.mark.parametrize("qwen3guard_wrong, switch", [(35, False), (36, True), (37, True)],
                         ids=["4.4 points", "exactly 5 points", "5.6 points"])
def test_decide_judge_prints_the_change_and_edits_nothing(qwen3guard_wrong, switch, tmp_path, monkeypatch,
                                                          capsys):
    world = use_world(monkeypatch, build_world(tmp_path / "world", judgements=False, utility=False, **ROUND1))
    real = real_repo_files()
    assert cli(capsys, "sample", "--round", 1)[0] == 0
    # 180 items. The frontier judge is wrong on 27 (85% agreement); Qwen3Guard on 35, 36 or 37, so
    # the frontier judge is 8, 9 or 10 items (4.4, 5.0 or 5.6 points) ahead. 0.85 - 0.80 is below
    # 0.05 in floating point: exactly 5 points must still switch.
    keys = sorted(it["key"] for it in read_round(world.validation, 1)["items"])
    assert len(keys) == 180
    world.frontier_flip = set(keys[:27])
    world.primary_flip = set(keys[27:27 + qwen3guard_wrong])
    write_judgements(world)
    printed = [import_round(world, capsys, 1, tmp_path / "sent")]

    decision = world.outputs / "metrics" / "judge_decision.json"
    code, dry = cli(capsys, "decide-judge", "--dry-run")
    assert code == 0 and not decision.exists()
    code, out = cli(capsys, "decide-judge")
    assert code == 0
    printed += [dry, out]
    decided = read_json(decision)
    assert decided["n"] == 180 and decided["switch_primary"] is switch

    project = paths.REPO_ROOT / "configs" / "project.yaml"
    judges = yaml.safe_load(project.read_text(encoding="utf-8"))["judges"]
    blocks = judges_blocks(out)
    if switch:
        # The exact config change: the primary and second judge blocks swap.
        assert any(b.get("safety_primary") == judges["safety_second"]
                   and b.get("safety_second") == judges["safety_primary"] for b in blocks), out
        # The docs/decisions.md entry: dated like the log's other headings, numbered after its last
        # entry, and naming the judge that becomes primary.
        d = date.today()
        headings = [line for line in out.splitlines() if line.startswith("## ")]
        assert headings and headings[0].startswith(f"## {d:%b} {d.day}, {d.year}")
        entry = out[out.index(headings[0]):]
        log = (paths.REPO_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
        number = max(int(n) for n in re.findall(r"^(\d+)\. \*\*", log, re.M)) + 1
        assert f"{number}. **" in entry and FRONTIER_ID in entry
    else:
        assert not blocks
        assert not [line for line in out.splitlines() if line.startswith("## ")]

    code, out = cli(capsys, "score", "--round", 1)       # the config, not the decision, names the primary
    assert code == 0
    printed.append(out)
    scored = read_json(world.outputs / "metrics" / "validation" / "round1.json")
    assert scored["primary_view"] == "qwen3guard_strict"
    assert scored["decision"]["recorded"] == ("frontier" if switch else "qwen3guard")
    assert scored["decision"]["applied"] is not switch
    assert real_repo_files() == real                    # a person edits the config and the log
    assert_private(world, *printed, metrics_text(world))


# --- the optional Hindi overlap -------------------------------------------------------------------


def test_the_hindi_overlap_follows_the_config_flag_unless_the_option_overrides_it(tmp_path, monkeypatch,
                                                                                  capsys):
    patch_project(monkeypatch, {"validation.hindi_overlap": True})
    world = use_world(monkeypatch, build_world(tmp_path / "world", judgements=False, utility=False, **ROUND1))
    code, out = cli(capsys, "sample", "--round", 1)
    assert code == 0
    main_hi, overlap = sheet_path(world, 1, "hi"), sheet_path(world, 1, "hi", "_b")
    assert overlap.is_file() and overlap.name in out
    shown = {(r["prompt"], r["response"]): r["item_id"] for r in sheet_rows(main_hi)}
    second = sheet_rows(overlap)
    assert len(second) == 20
    assert all((r["prompt"], r["response"]) in shown for r in second)       # the same items...
    assert not {r["item_id"] for r in second} & set(shown.values())         # ...under their own ids

    # Both Hindi readers import with --lang hi; the ids tell the two sheets apart.
    labels = true_labels(world, 1)
    printed = [out]
    for src in (main_hi, overlap):
        code, out = cli(capsys, "import", "--round", 1, "--lang", "hi",
                        "--file", fill_sheet(src, tmp_path / src.name, labels))
        assert code == 0, out
        printed.append(out)
    second_reader = list(read_jsonl(world.validation / "labels" / "round1_hi_b.jsonl"))
    assert len(second_reader) == 20 and {r["annotator"] for r in second_reader} == {"hi-b"}
    code, out = cli(capsys, "score", "--round", 1)
    assert code == 0
    printed.append(out)
    agreement = read_json(world.outputs / "metrics" / "validation" / "round1.json")["inter_annotator"]["hi"]
    assert agreement["n"] == 20 and agreement["agreement"] == 1.0
    assert_private(world, *printed)

    other = use_world(monkeypatch, build_world(tmp_path / "other", judgements=False, utility=False, **ROUND1))
    code, out = cli(capsys, "sample", "--round", 1, "--no-hindi-overlap")
    assert code == 0, out
    assert sheet_path(other, 1, "hi").is_file() and not sheet_path(other, 1, "hi", "_b").exists()


# --- errors -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("problem", ["unfinished run", "too few prompts", "round 2 before round 1"])
def test_check_returns_1_and_sample_fails_when_the_round_cannot_be_drawn(problem, tmp_path, monkeypatch,
                                                                          capsys):
    round_no = 2 if problem == "round 2 before round 1" else 1
    world = build_world(tmp_path / "world", n_dev=20 if problem == "too few prompts" else 40,
                        judgements=False, utility=False, **(ROUND2 if round_no == 2 else ROUND1))
    use_world(monkeypatch, world)
    if problem == "unfinished run":       # the adapter has answered 30 of its 40 Hindi harmful prompts
        recs = world.generations[(CANDIDATE, "harmful", "hi", "dev")]
        write_generations(world, CANDIDATE, "harmful", "hi", "dev", recs[:30])
    before = tree(world.root)
    code, out = cli(capsys, "sample", "--round", round_no, "--check")
    assert code == 1 and "Traceback" not in out
    if problem == "unfinished run":
        assert f"{CANDIDATE}/harmful_hi_dev" in out
    code, out = cli(capsys, "sample", "--round", round_no)
    assert_error(code, out)
    assert tree(world.root) == before


@pytest.mark.parametrize("argv", [
    ["sample", "--round", "1"],
    ["import", "--round", "1", "--lang", "hi", "--file", "{tmp}/filled.csv"],
    ["score"],
    ["decide-judge"],
    ["second-judge-sample"],
    ["utility-sheet", "--variant", CANDIDATE],
    ["utility-import", "--file", "{tmp}/filled.csv"],
    ["utility-score"],
], ids=lambda argv: argv[0])
def test_a_step_run_before_its_inputs_exist_fails_cleanly(argv, tmp_path, capsys):
    before = tree(tmp_path)                              # nothing at all: no data, outputs or keys
    code, out = cli(capsys, *[a.format(tmp=tmp_path) for a in argv])
    assert_error(code, out)
    assert tree(tmp_path) == before


@pytest.fixture(scope="module")
def drawn(tmp_path_factory) -> World:
    """Round 1 drawn and the utility sheet made. Shared by the error tests below, each of which
    checks that its failing command leaves this world exactly as it was."""
    world = build_world(tmp_path_factory.mktemp("drawn") / "world", judgements=False, **ROUND1)
    with pytest.MonkeyPatch.context() as mp:
        use_world(mp, world)
        mp.setattr("lift.manifest.git_sha", lambda: NO_GIT)
        for argv in (["sample", "--round", "1"], ["utility-sheet", "--variant", CANDIDATE]):
            assert main(["validate", *argv]) == 0
    return world


def _bad_label(header: list[str], rows: list[list[str]]) -> str:
    rows[0][header.index("label")] = "maybe"
    return "maybe"


def _unknown_id(header: list[str], rows: list[list[str]]) -> str:
    rows[0][header.index("item_id")] = "r1hi-zzzzzz"
    return "r1hi-zzzzzz"


def _repeated_id(header: list[str], rows: list[list[str]]) -> str:
    rows.append(list(rows[0]))
    return rows[0][header.index("item_id")]


@pytest.mark.parametrize("spoil", [_bad_label, _unknown_id, _repeated_id],
                         ids=["bad label", "unknown id", "repeated id"])
def test_a_bad_sheet_is_reported_and_nothing_is_imported(spoil, drawn, tmp_path, monkeypatch, capsys):
    use_world(monkeypatch, drawn)
    rows = read_csv(sheet_path(drawn, 1, "hi"))
    header, items = rows[1], [list(r) for r in rows[2:]]
    for row in items:
        row[header.index("label")] = "benign"
    named = spoil(header, items)
    filled = write_csv(tmp_path / "filled.csv", [rows[0], header, *items])
    before = tree(drawn.root)
    code, out = cli(capsys, "import", "--round", 1, "--lang", "hi", "--file", filled)
    assert_error(code, out)
    assert named in out
    assert tree(drawn.root) == before
    assert_private(drawn, out)


@pytest.mark.parametrize("argv, named", [
    (["import", "--round", "1", "--lang", "hi", "--file", "{tmp}/missing.csv"], "missing.csv"),
    (["utility-import", "--file", "{tmp}/missing.csv"], "missing.csv"),
    (["import", "--round", "1", "--lang", "zh", "--file", "{sheets}/validation_round1_hi.csv"],
     "validation_round1_hi.csv"),
    (["sample", "--round", "1", "--variants", "adapter-nope-std-s1"], "adapter-nope-std-s1"),
    (["sample", "--round", "1", "--seed", "7"], "--force"),
    (["utility-sheet", "--variant", "adapter-nope-std-s1"], "adapter-nope-std-s1"),
    (["utility-sheet", "--variant", "stock"], "stock"),
    (["decide-judge"], None),
    (["utility-score"], None),
], ids=["sheet file missing", "utility file missing", "hi sheet imported as zh", "unknown variant to sample",
        "another seed over a drawn round", "unknown utility candidate", "stock as the utility candidate",
        "decide before labels are imported", "utility score before the import"])
def test_errors_exit_1_with_an_error_line_and_no_traceback(argv, named, drawn, tmp_path, monkeypatch, capsys):
    use_world(monkeypatch, drawn)
    before = tree(drawn.root)
    code, out = cli(capsys, *[a.format(tmp=tmp_path, sheets=drawn.validation / "sheets") for a in argv])
    assert_error(code, out)
    if named:
        assert named in out
    assert tree(drawn.root) == before
    assert_private(drawn, out)


def test_a_generations_file_cut_short_by_a_killed_job_is_reported_not_a_traceback(tmp_path, monkeypatch,
                                                                                   capsys):
    # lift.io.append_jsonl writes resumable runs line by line, and lift.io.existing_keys skips a
    # truncated last line: a session that died mid-write leaves exactly this file behind.
    world = use_world(monkeypatch, build_world(tmp_path / "world", judgements=False, utility=False, **ROUND1))
    path = world.outputs / "generations" / CANDIDATE / "harmful_hi_dev.jsonl"
    path.write_bytes(path.read_bytes()[:-40])
    before = tree(world.root)
    code, out = cli(capsys, "sample", "--round", 1, "--check")
    assert code == 1 and "Traceback" not in out
    code, out = cli(capsys, "sample", "--round", 1)
    assert_error(code, out)
    assert tree(world.root) == before


def test_a_verdicts_file_that_is_not_jsonl_is_an_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    world = use_world(monkeypatch, build_world(tmp_path / "world", judgements=False, **ROUND1))
    assert cli(capsys, "utility-sheet", "--variant", CANDIDATE)[0] == 0
    sheet = world.validation / "sheets" / "validation_utility_hi.csv"
    filled = fill_sheet(sheet, tmp_path / "filled.csv", {r["item_id"]: "tie" for r in sheet_rows(sheet)},
                        column="preference")
    assert cli(capsys, "utility-import", "--file", filled)[0] == 0
    before = tree(world.root)
    code, out = cli(capsys, "utility-score", "--verdicts", filled)    # the sheet passed by mistake
    assert_error(code, out)
    assert "filled.csv" in out
    assert tree(world.root) == before
