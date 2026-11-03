"""lift.validation.score.score: every judge against the readers, per language and pooled (P04, Build 4).

Each number is checked against lift.stats.agreement computed directly on the same (human, judge)
label lists. The worlds are fake (tests/validation_fakes.py): neutral placeholder text, no model,
no network, everything under pytest's temporary directories.
"""

from __future__ import annotations

import copy
import csv
import json
import shutil
from collections import defaultdict

import pytest
from validation_fakes import (
    FRONTIER_ID,
    LANGS,
    PRIMARY_ID,
    SAFETY_SETS,
    World,
    build_world,
    fill_sheet,
    judgement,
    other_label,
    read_csv,
    truth,
    use_world,
    write_judgements,
)

from lift import config, manifest
from lift.io import read_jsonl, write_jsonl
from lift.judge.labels import final_label, map_qwen3guard
from lift.schemas import LABELS, HumanLabel, JudgementRecord, UnclearFlags
from lift.stats.agreement import binary, cohens_kappa, meets_bar, percent_agreement
from lift.validation.labels import import_labels
from lift.validation.sample import load_round, make_sheets
from lift.validation.score import decision_path, qwen3guard_label, score
from lift.validation.store import ValidateError, read_json, write_json

VIEWS = ("qwen3guard_strict", "qwen3guard_loose", "frontier")
OTHER_QWEN = "Qwen/Qwen3Guard-Gen-4B@fedcba9876543210"

# Qwen3Guard native fields as a harness might store them: keys in any case, Refusal as Yes/No or a
# bool. (native fields, Safety, Refusal as a bool, too short)
NATIVE_FORMS = [
    ({"Safety": "Controversial", "Refusal": "Yes"}, "Controversial", True, False),
    ({"SAFETY": "controversial", "REFUSAL": False}, "controversial", False, False),
    ({"safety": "Controversial", "refusal": True}, "Controversial", True, False),
    ({"Safety": "Controversial", "refusal": "no"}, "Controversial", False, True),
    ({"Safety": "Safe", "Refusal": "yes"}, "Safe", True, False),
    ({"safety": "Safe", "Refusal": False}, "Safe", False, False),
    ({"Safety": "Unsafe", "Refusal": "No"}, "Unsafe", False, False),
    ({"safety": "Safe", "REFUSAL": True}, "Safe", True, True),
]


def fake_git():
    return "0" * 40


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    # make_sheets and score record the git sha; a fake world does not need a real one.
    monkeypatch.setattr(manifest, "git_sha", fake_git)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """Stock and seed-1 adapter dev outputs, no judgements. 34 dev prompts per set is the least
    that fills round 1, since hi and en share a reader."""
    return build_world(tmp_path_factory.mktemp("world"), n_dev=34, n_final=0,
                       dev_variants=("stock", "adapter-aya-std-s1"), final_variants=(),
                       judgements=False, utility=False)


def draw(base, folder, *, overlap):
    """Round 1 drawn from ``base`` into its own validation folder."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifest, "git_sha", fake_git)
        use_world(mp, base)
        mp.setenv("LIFT_VALIDATION_DIR", str(folder))
        make_sheets(1, hindi_overlap=overlap)
    return folder


@pytest.fixture(scope="module")
def drawn(base, tmp_path_factory):
    return draw(base, tmp_path_factory.mktemp("validation"), overlap=False)


@pytest.fixture(scope="module")
def drawn_with_overlap(base, tmp_path_factory):
    return draw(base, tmp_path_factory.mktemp("validation_overlap"), overlap=True)


def private_copy(base, validation, tmp_path, monkeypatch):
    """This test's own copy of a drawn round 1 (keys and sheets) and an empty outputs folder.
    Scoring never reads the generations again and the prompts are shared read-only: fresh files
    are what makes a test slow on Windows."""
    shutil.copytree(validation, tmp_path / "world" / "validation")
    world = use_world(monkeypatch, World(root=tmp_path / "world", n=dict(base.n),
                                         generations=dict(base.generations)))
    monkeypatch.setenv("LIFT_DATA_DIR", str(base.data))
    return world, load_round(1)


@pytest.fixture
def round1(base, drawn, tmp_path, monkeypatch):
    return private_copy(base, drawn, tmp_path, monkeypatch)


@pytest.fixture
def round1_overlap(base, drawn_with_overlap, tmp_path, monkeypatch):
    return private_copy(base, drawn_with_overlap, tmp_path, monkeypatch)


def items(doc, lang=None, slot="a"):
    return sorted((it for it in doc["items"] if it["slot"] == slot and lang in (None, it["lang"])),
                  key=lambda it: it["item_id"])


def truth_of(item):
    return truth(item["key"])


def drop_rows(path, ids):
    rows = read_csv(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows[:2] + [r for r in rows[2:] if r[0] not in ids])


def import_sheets(world, doc, label_of, *, sheets=None, notes=None, drop=()):
    """What the readers send back: each sheet (or the named ones) with label_of(item) in the label
    column (None leaves it empty) and the ``drop`` rows deleted, imported as round 1."""
    for name, info in doc["sheets"].items():
        if sheets is not None and name not in sheets:
            continue
        values = {it["item_id"]: label_of(it) for it in doc["items"] if it["sheet"] == name}
        filled = fill_sheet(world.validation / "sheets" / info["file"], world.root / f"filled_{name}.csv",
                            {k: v for k, v in values.items() if v is not None}, notes=notes)
        drop_rows(filled, set(drop))
        import_labels(1, info["lang"], filled)


def flags(too_short=False):
    return UnclearFlags(too_short=too_short, repetitive=False, garbled=False, wrong_lang=False,
                        lid_top=[("hin_Deva", 0.99)])


def write_records(world, records, dirname):
    """JudgementRecords in the harness layout, outputs/judgements/{dirname}/{variant}/{file}."""
    files = defaultdict(list)
    for r in records:
        files[(r.variant, r.set, r.lang, r.split)].append(r.model_dump())
    for (variant, set_name, lang, split), rows in files.items():
        write_jsonl(world.outputs / "judgements" / dirname / variant / f"{set_name}_{lang}_{split}.jsonl",
                    rows)


def replace_judgement(world, dirname, key, **update):
    """Rewrite the one record of generation ``key`` under outputs/judgements/{dirname}/."""
    for path in (world.outputs / "judgements" / dirname).rglob("*.jsonl"):
        rows = list(read_jsonl(path))
        if any(r["key"] == key for r in rows):
            write_jsonl(path, [JudgementRecord.model_validate(r).model_copy(update=update).model_dump()
                               if r["key"] == key else r for r in rows])
            return
    raise AssertionError(f"no {dirname} judgement of {key[:12]}")


def loose(rec):
    """The spec's loose label of a fake Qwen3Guard record (lowercase keys, Refusal as Yes/No)."""
    if rec.native.get("judge_failure"):
        return "unclear"                 # a judge failure is recorded as unclear
    refusal = rec.native["refusal"] == "Yes"
    return final_label(map_qwen3guard(rec.native["safety"], refusal, strict=False), rec.flags)


def view_labels(world):
    """{view: {generation key: the judge label that view should use}}, read from the files."""
    found = defaultdict(dict)
    for path in (world.outputs / "judgements").rglob("*.jsonl"):
        for row in read_jsonl(path):
            rec = JudgementRecord.model_validate(row)
            found[rec.judge][rec.key] = rec
    return {"qwen3guard_strict": {k: r.label for k, r in found[PRIMARY_ID].items()},
            "qwen3guard_loose": {k: loose(r) for k, r in found[PRIMARY_ID].items()},
            "frontier": {k: r.label for k, r in found[FRONTIER_ID].items()}}


def block_of(out, view, lang):
    j = out["judges"][view]
    return j["pooled"] if lang is None else j["langs"][lang]


def check_block(block, pairs, *, complete=True, withheld=False):
    """``block`` equals lift.stats.agreement computed directly on the judged (human, judge) pairs.
    ``withheld``: the bar check is not given (rows of the language are still unfinished)."""
    bar = config.get("validation.bar")
    human, judge = [h for h, _ in pairs], [j for _, j in pairs]
    want = meets_bar(human, judge, agreement=bar["agreement"], kappa=bar["kappa"])
    assert block["n_judged"] == len(pairs)
    assert block["complete"] is complete
    if complete:
        assert block["n"] == len(pairs)
    for k in ("agreement", "kappa", "agreement_binary", "kappa_binary"):
        assert block[k] == pytest.approx(want[k]), k
    assert block["meets_bar"] is (want["meets_bar"] if complete and not withheld else None)
    matrix = block["confusion"]["matrix"]                   # rows: human, columns: judge
    assert matrix == [[sum((h, j) == (a, b) for h, j in pairs) for b in LABELS] for a in LABELS]
    assert sum(map(sum, matrix)) == len(pairs)
    assert sum(matrix[i][i] for i in range(len(LABELS))) == block["n_agree"] == sum(h == j for h, j in pairs)
    bmatrix = block["confusion_binary"]["matrix"]
    assert sum(map(sum, bmatrix)) == len(pairs)
    assert bmatrix[0][0] + bmatrix[1][1] == round(want["agreement_binary"] * len(pairs))


def gate1(world):
    return read_json(world.outputs / "metrics" / "validation" / "gate1.json")


def test_every_view_agrees_with_lift_stats_per_language_and_pooled(round1):
    world, doc = round1
    human = {}
    for lang in LANGS:
        for i, it in enumerate(items(doc, lang)):
            if lang == "hi" and i % 10 == 0:
                world.primary_flip.add(it["key"])
            if i % (2 if lang == "hi" else 9) == 1:
                world.frontier_flip.add(it["key"])           # the frontier judge fails in Hindi
            t = truth_of(it)
            human[it["item_id"]] = LABELS[(LABELS.index(t) + 2) % 4] if i % 11 == 5 else t
    write_judgements(world)
    import_sheets(world, doc, lambda it: human[it["item_id"]])
    out = score(1)
    labels = view_labels(world)
    bars = set()
    for view in VIEWS:
        for lang in [*LANGS, None]:
            block = block_of(out, view, lang)
            check_block(block, [(human[it["item_id"]], labels[view][it["key"]]) for it in items(doc, lang)])
            assert block["status"] == "ok"
            bars.add(block["meets_bar"])
    assert bars == {True, False}                             # both sides of the bar are exercised
    assert len({block_of(out, v, None)["agreement"] for v in VIEWS}) == 3


@pytest.mark.parametrize(("native", "safety", "refusal", "short"), NATIVE_FORMS)
def test_qwen3guard_label_reads_native_fields_in_any_format(native, safety, refusal, short):
    judge_label = map_qwen3guard(safety, refusal, strict=True)
    rec = JudgementRecord(key="k" * 64, variant="stock", pid="prh-test-0000", lang="hi", set="harmful",
                          split="dev", judge=PRIMARY_ID, judge_label=judge_label,
                          label=final_label(judge_label, flags(short)), flags=flags(short), native=native)
    for strict in (True, False):
        want = final_label(map_qwen3guard(safety, refusal, strict=strict), flags(short))
        assert qwen3guard_label(rec, strict=strict) == want


def test_loose_view_is_the_loose_mapping_of_the_native_fields(round1):
    world, doc = round1
    write_judgements(world)                                  # keeps the frontier judge's records
    shutil.rmtree(world.outputs / "judgements" / "qwen3guard")
    gens = world.by_key()
    records, want, human = [], {"qwen3guard_strict": {}, "qwen3guard_loose": {}}, {}
    for i, it in enumerate(items(doc)):
        native, safety, refusal, short = NATIVE_FORMS[i % len(NATIVE_FORMS)]
        g = gens[it["key"]]
        judge_label = map_qwen3guard(safety, refusal, strict=True)
        strict = final_label(judge_label, flags(short))
        records.append(JudgementRecord(key=g.key, variant=g.variant, pid=g.pid, lang=g.lang, set=g.set,
                                       split=g.split, judge=PRIMARY_ID, judge_label=judge_label,
                                       label=strict, flags=flags(short), native=native))
        want["qwen3guard_strict"][g.key] = strict
        want["qwen3guard_loose"][g.key] = final_label(map_qwen3guard(safety, refusal, strict=False),
                                                      flags(short))
        human[it["item_id"]] = want["qwen3guard_strict" if i % 3 else "qwen3guard_loose"][g.key]
    write_records(world, records, "qwen3guard")
    import_sheets(world, doc, lambda it: human[it["item_id"]])
    out = score(1)
    c = out["checks"]
    assert (c["loose_disabled"], c["strict_native_mismatch"], c["loose_unavailable"]) == (False, 0, 0)
    for view, labels in want.items():
        assert out["judges"][view]["judge_id"] == PRIMARY_ID
        for lang in [*LANGS, None]:
            block = block_of(out, view, lang)
            check_block(block, [(human[it["item_id"]], labels[it["key"]]) for it in items(doc, lang)])
            assert block["status"] == "ok"
    pooled = {view: block_of(out, view, None)["agreement"] for view in want}
    assert pooled["qwen3guard_loose"] != pooled["qwen3guard_strict"]


def test_a_record_misread_from_its_native_fields_withholds_the_loose_view(round1):
    world, doc = round1
    write_judgements(world)
    # Stored as benign while its own native fields say Unsafe: the fields are being misread.
    replace_judgement(world, "qwen3guard", items(doc, "hi")[0]["key"], label="benign", judge_label="benign",
                      native={"safety": "Unsafe", "categories": ["Violent"], "refusal": "No"})
    import_sheets(world, doc, truth_of)
    out = score(1)
    assert out["checks"]["strict_native_mismatch"] == 1
    assert out["checks"]["loose_disabled"] is True
    for lang in [*LANGS, None]:
        block = block_of(out, "qwen3guard_loose", lang)
        assert block["status"] == "native_mismatch"
        assert block["n_judged"] == 0
        assert block["agreement"] is block["kappa"] is block["meets_bar"] is None
        for view in ("qwen3guard_strict", "frontier"):
            assert block_of(out, view, lang)["status"] == "ok"
    md = (world.outputs / "metrics" / "validation" / "round1.md").read_text(encoding="utf-8")
    assert "loose view is withheld" in md


def test_missing_judgements_cover_only_the_judged_items_and_withhold_the_bar(round1):
    world, doc = round1
    hi = items(doc, "hi")
    write_judgements(world, skip={it["key"] for it in hi[:3]})
    import_sheets(world, doc, truth_of)
    out = score(1)
    labels = view_labels(world)
    for view in VIEWS:
        block = block_of(out, view, "hi")
        assert (block["n"], block["n_judged"], block["n_missing"]) == (60, 57, 3)
        assert block["status"] == "pending_judge"
        check_block(block, [(truth_of(it), labels[view][it["key"]]) for it in hi[3:]], complete=False)
        for by in ("group", "set"):
            table = out["judges"][view]["breakdown"]["hi"][by].values()
            assert (sum(b["n"] for b in table), sum(b["n_judged"] for b in table)) == (60, 57)
        pooled = block_of(out, view, None)
        assert (pooled["status"], pooled["complete"], pooled["meets_bar"]) == ("pending_judge", False, None)
        assert pooled["n"] - pooled["n_judged"] == 3
        for lang in ("zh", "es", "en"):
            other = block_of(out, view, lang)
            assert (other["status"], other["complete"]) == ("ok", True)
    assert (gate1(world)["status"], gate1(world)["meets_bar"]) == ("pending", None)


def test_a_judge_that_has_not_run_yet_is_pending_not_a_crash(round1):
    world, doc = round1
    write_judgements(world, frontier=False)
    import_sheets(world, doc, truth_of)
    out = score(1)
    assert out["judges"]["frontier"]["judge_id"] is None
    for lang in [*LANGS, None]:
        block = block_of(out, "frontier", lang)
        assert block["status"] == "pending_judge"
        assert block["n"] > 0 and block["n_judged"] == 0 and block["complete"] is False
        assert (block["agreement"], block["kappa"], block["meets_bar"], block["confusion"]) == (None,) * 4
        assert block_of(out, "qwen3guard_strict", lang)["status"] == "ok"


def test_sheets_not_imported_are_pending_labels(round1):
    world, doc = round1
    write_judgements(world)
    out = score(1)                                           # nothing imported yet
    for view in VIEWS:
        assert {block_of(out, view, lang)["status"] for lang in [*LANGS, None]} == {"pending_labels"}
    assert gate1(world)["status"] == "pending"
    import_sheets(world, doc, truth_of, sheets={"round1_hi", "round1_en"})
    out = score(1)
    for view in VIEWS:
        assert block_of(out, view, "hi")["status"] == block_of(out, view, "en")["status"] == "ok"
        for lang in ("zh", "es"):
            block = block_of(out, view, lang)
            assert (block["status"], block["n"], block["meets_bar"]) == ("pending_labels", 0, None)
        assert block_of(out, view, None)["status"] == "pending_labels"
    assert out["humans"]["zh"]["not_labelled_by_reason"] == {"not_imported": 40}
    assert out["humans"]["pooled"]["labelled"] == 100
    assert gate1(world)["status"] == "pass"


def test_items_without_a_label_are_left_out_and_counted(round1):
    world, doc = round1
    write_judgements(world)
    hi = items(doc, "hi")
    skipped, unlabelled, missing = hi[:2], hi[2:5], hi[5:6]
    gone = {it["item_id"] for it in skipped + unlabelled + missing}
    import_sheets(world, doc, lambda it: None if it["item_id"] in gone else truth_of(it),
                  notes={it["item_id"]: "skipped" for it in skipped}, drop={it["item_id"] for it in missing})
    out = score(1)
    reasons = {"skipped": 2, "unlabelled": 3, "missing": 1}
    h = out["humans"]["hi"]
    assert (h["items"], h["labelled"], h["not_labelled"], h["not_labelled_by_reason"]) == (60, 54, 6, reasons)
    assert sum(h["label_counts"].values()) == 54
    assert out["humans"]["pooled"]["labelled"] == 174
    assert out["humans"]["pooled"]["not_labelled_by_reason"] == reasons
    labels = view_labels(world)
    kept = [it for it in hi if it["item_id"] not in gone]
    for view in VIEWS:
        # Unlabelled and missing rows (not deliberate skips) keep Hindi pending: numbers, no bar check.
        block = block_of(out, view, "hi")
        assert block["status"] == "pending_labels"
        check_block(block, [(truth_of(it), labels[view][it["key"]]) for it in kept], withheld=True)
        assert block_of(out, view, "zh")["status"] == "ok"
    assert gate1(world)["status"] == "pending" and gate1(world)["not_labelled"] == 4


def test_a_judge_failure_counts_as_an_unclear_judgement(round1):
    world, doc = round1
    write_judgements(world)
    hi = items(doc, "hi")
    replace_judgement(world, "qwen3guard", hi[0]["key"], judge_label="unclear", label="unclear",
                      native={"judge_failure": "timeout"})
    replace_judgement(world, "frontier", hi[1]["key"], judge_label="unclear", label="unclear",
                      native={"judge_failure": "rate limited"})
    import_sheets(world, doc, truth_of)
    out = score(1)
    assert out["checks"]["judge_failures"] == {"qwen3guard": 1, "frontier": 1}
    assert out["checks"]["loose_disabled"] is False
    labels = view_labels(world)
    assert labels["qwen3guard_strict"][hi[0]["key"]] == labels["frontier"][hi[1]["key"]] == "unclear"
    for view in VIEWS:
        block = block_of(out, view, "hi")
        check_block(block, [(truth_of(it), labels[view][it["key"]]) for it in hi])
        assert block["status"] == "ok"


def test_judges_are_told_apart_by_id_not_by_directory(round1):
    world, doc = round1
    world.primary_flip |= {it["key"] for it in items(doc)[::4]}
    world.frontier_flip |= {it["key"] for it in items(doc)[1::3]}
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    before = score(1, write=False)
    agreement = {view: block_of(before, view, None)["agreement"] for view in VIEWS}
    assert agreement["qwen3guard_strict"] != agreement["frontier"]       # so a mix-up would show
    root = world.outputs / "judgements"
    (root / "qwen3guard").rename(root / "llm")               # misleading names on purpose
    (root / "frontier").rename(root / "q3g")
    after = score(1, write=False)["judges"]
    assert after == before["judges"]
    assert after["qwen3guard_strict"]["judge_id"] == after["qwen3guard_loose"]["judge_id"] == PRIMARY_ID
    assert after["frontier"]["judge_id"] == FRONTIER_ID


def two_qwen3guard_revisions(world, doc):
    """Round 1 judged and labelled, plus a Qwen3Guard rerun at another revision that disagrees with
    the readers on a third of the items. Returns the (human, judge) pairs of each revision."""
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    flipped = {it["key"] for it in items(doc)[::3]}
    gens = world.by_key()
    rerun = [judgement(gens[it["key"]], PRIMARY_ID,
                       other_label(truth_of(it)) if it["key"] in flipped else truth_of(it))
             .model_copy(update={"judge": OTHER_QWEN}) for it in items(doc)]
    write_records(world, rerun, "qwen3guard-rerun")
    return {PRIMARY_ID: [(truth_of(it), truth_of(it)) for it in items(doc)],
            OTHER_QWEN: [(truth_of(it), other_label(truth_of(it)) if it["key"] in flipped else truth_of(it))
                         for it in items(doc)]}


def test_two_qwen3guard_revisions_are_an_error_until_one_is_picked(round1):
    pairs = two_qwen3guard_revisions(*round1)
    with pytest.raises(ValidateError, match="several qwen3guard judges"):
        score(1)
    for qwen in (PRIMARY_ID, OTHER_QWEN):
        out = score(1, qwen3guard_judge=qwen, frontier_judge=FRONTIER_ID)
        assert out["judges"]["qwen3guard_strict"]["judge_id"] == qwen
        check_block(block_of(out, "qwen3guard_strict", None), pairs[qwen])
        check_block(block_of(out, "frontier", None), pairs[PRIMARY_ID])


def test_two_frontier_judges_are_an_error_until_one_is_picked(round1):
    world, doc = round1
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    another = "fakeprovider/another-model@2026-11-02"            # disagrees with the readers everywhere
    gens = world.by_key()
    write_records(world, [judgement(gens[it["key"]], another, other_label(truth_of(it)))
                          for it in items(doc)], "frontier-2")
    with pytest.raises(ValidateError, match="several frontier judges"):
        score(1, write=False)
    for frontier, says in ((FRONTIER_ID, truth), (another, lambda key: other_label(truth(key)))):
        out = score(1, frontier_judge=frontier, write=False)
        assert out["judges"]["frontier"]["judge_id"] == frontier
        assert out["judges"]["qwen3guard_strict"]["judge_id"] == PRIMARY_ID
        check_block(block_of(out, "frontier", None), [(truth_of(it), says(it["key"])) for it in items(doc)])


def test_picking_the_qwen3guard_revision_alone_settles_it(round1):
    pairs = two_qwen3guard_revisions(*round1)
    for qwen in (PRIMARY_ID, OTHER_QWEN):
        out = score(1, qwen3guard_judge=qwen)
        assert out["judges"]["qwen3guard_strict"]["judge_id"] == qwen
        assert out["judges"]["frontier"]["judge_id"] == FRONTIER_ID
        check_block(block_of(out, "qwen3guard_strict", None), pairs[qwen])


def test_swapping_the_judges_config_roles_keeps_qwen3guard_in_its_views(round1, monkeypatch):
    world, doc = round1
    world.primary_flip |= {it["key"] for it in items(doc)[::4]}
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    before = score(1, write=False)["judges"]
    real = config.load

    def swapped(name="project"):
        data = real(name)
        if name == "project":
            data = copy.deepcopy(data)
            j = data["judges"]
            j["safety_primary"], j["safety_second"] = j["safety_second"], j["safety_primary"]
        return data

    monkeypatch.setattr(config, "load", swapped)
    assert "Qwen3Guard" in config.get("judges.safety_second.id")
    after = score(1, write=False)["judges"]
    assert after == before
    assert after["qwen3guard_strict"]["judge_id"] == PRIMARY_ID
    assert after["frontier"]["judge_id"] == FRONTIER_ID


def test_hindi_overlap_gives_inter_annotator_agreement_and_not_ground_truth(round1_overlap):
    world, doc = round1_overlap
    write_judgements(world)
    b_items = items(doc, "hi", slot="b")
    assert len(b_items) == 20
    second = {it["item_id"]: other_label(truth_of(it)) if i % 4 == 0 else truth_of(it)
              for i, it in enumerate(b_items)}
    second[b_items[1]["item_id"]] = None                     # reader b leaves one row empty
    import_sheets(world, doc, lambda it: second[it["item_id"]], sheets={"round1_hi_b"})
    alone = score(1)                                         # reader b's labels never stand in for a's
    assert (block_of(alone, "qwen3guard_strict", "hi")["status"], alone["inter_annotator"]) == (
        "pending_labels", {})
    assert block_of(alone, "qwen3guard_strict", "hi")["n"] == alone["humans"]["hi"]["labelled"] == 0
    import_sheets(world, doc, truth_of, sheets={"round1_hi", "round1_zh", "round1_es", "round1_en"})
    out = score(1)
    first = {it["key"]: truth_of(it) for it in items(doc, "hi")}
    pairs = [(first[it["key"]], second[it["item_id"]]) for it in b_items if second[it["item_id"]]]
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    ia = out["inter_annotator"]
    assert list(ia) == ["hi"]
    assert ia["hi"]["n"] == len(pairs) == 19
    assert ia["hi"]["agreement"] == pytest.approx(percent_agreement(a, b)) and ia["hi"]["agreement"] < 1
    assert ia["hi"]["kappa"] == pytest.approx(cohens_kappa(a, b))
    assert ia["hi"]["agreement_binary"] == pytest.approx(percent_agreement(binary(a), binary(b)))
    assert ia["hi"]["kappa_binary"] == pytest.approx(cohens_kappa(binary(a), binary(b),
                                                                  labels=("harmful", "benign")))
    # Only the main reader is the ground truth: Hindi is reader a's 60 items, which the judges match.
    assert out["humans"]["hi"]["items"] == 60
    for view in ("qwen3guard_strict", "frontier"):
        assert (block_of(out, view, "hi")["n"], block_of(out, view, "hi")["agreement"]) == (60, 1.0)


def test_score_writes_round1_and_gate1_with_counts_only(round1):
    world, doc = round1
    world.primary_flip |= {it["key"] for it in items(doc)[::5]}
    world.frontier_flip |= {it["key"] for it in items(doc)[2::7]}
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    out = score(1)
    folder = world.outputs / "metrics" / "validation"
    assert sorted(p.name for p in folder.iterdir()) == ["gate1.json", "round1.json", "round1.md"]
    assert read_json(folder / "round1.json") == json.loads(json.dumps(out, default=str))
    written = "\n".join(p.read_text(encoding="utf-8") for p in folder.iterdir())
    shown = [cell.strip() for info in doc["sheets"].values()
             for row in read_csv(world.validation / "sheets" / info["file"])[2:] for cell in row[1:3]]
    assert len(shown) == 2 * 180
    assert [t for t in shown if t in written] == []
    assert [t for t in shown if t.split()[1] in written] == []      # not even their hash part

    gate, hi = gate1(world), block_of(out, "qwen3guard_strict", "hi")
    assert gate["status"] == ("pass" if hi["meets_bar"] else "fail")
    assert (gate["judge_view"], gate["judge_id"]) == ("qwen3guard_strict", PRIMARY_ID)
    fields = ("meets_bar", "n", "n_judged", "agreement", "kappa")
    assert [gate[k] for k in fields] == [hi[k] for k in fields]

    labels = view_labels(world)
    for view in VIEWS:
        for scope in ("pooled", "hi"):
            mine = items(doc, None if scope == "pooled" else scope)
            total = block_of(out, view, None if scope == "pooled" else scope)
            for by, names in (("group", {"stock", "adapter"}), ("set", set(SAFETY_SETS))):
                table = out["judges"][view]["breakdown"][scope][by]
                assert set(table) == names
                assert sum(b["n"] for b in table.values()) == total["n"] == len(mine)
                assert sum(b["n_judged"] for b in table.values()) == total["n_judged"]
                for name, b in table.items():
                    pairs = [(truth_of(it), labels[view][it["key"]]) for it in mine if it[by] == name]
                    human, judge = [h for h, _ in pairs], [j for _, j in pairs]
                    assert b["n"] == len(pairs)
                    assert b["agreement"] == pytest.approx(percent_agreement(human, judge))
                    assert b["human_harmful_rate"] == pytest.approx(human.count("harmful") / len(pairs))
                    assert b["judge_harmful_rate"] == pytest.approx(judge.count("harmful") / len(pairs))
                    assert b["gap"] == pytest.approx(b["judge_harmful_rate"] - b["human_harmful_rate"])

    everything = score()                                     # every round drawn so far
    assert (folder / "all.json").is_file() and (folder / "all.md").is_file()
    assert (everything["round"], everything["rounds"]) == (None, [1])
    assert everything["judges"] == out["judges"]


@pytest.mark.parametrize("case", ["pass", "fail", "pending"])
def test_gate1_is_the_primary_judges_bar_in_hindi(round1, case):
    world, doc = round1
    hi = items(doc, "hi")
    if case == "fail":
        world.primary_flip |= {it["key"] for it in hi[::2]}      # Qwen3Guard wrong on half of Hindi
    write_judgements(world, skip={hi[0]["key"]} if case == "pending" else frozenset())
    import_sheets(world, doc, truth_of)
    out = score(1)
    gate, block = gate1(world), block_of(out, "qwen3guard_strict", "hi")
    assert gate["status"] == case
    assert gate["meets_bar"] is {"pass": True, "fail": False, "pending": None}[case] is block["meets_bar"]
    assert (gate["judge_view"], gate["judge_id"], gate["lang"]) == ("qwen3guard_strict", PRIMARY_ID, "hi")


def test_gate1_follows_the_primary_judge_in_the_config_not_the_recorded_decision(round1, monkeypatch):
    world, doc = round1
    world.frontier_flip |= {it["key"] for it in items(doc, "hi")[::2]}
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    write_json(decision_path(), {"primary_after": "frontier",
                                 "inputs": {"round1_hi.jsonl": "0" * 64, "round1.json": "0" * 64}})
    # Recorded but not applied: the harness still uses Qwen3Guard, so Gate 1 checks Qwen3Guard.
    out = score(1)
    assert out["primary_view"] == "qwen3guard_strict"
    assert out["decision"] == {"recorded": "frontier", "config_primary": "qwen3guard", "applied": False}
    assert block_of(out, "qwen3guard_strict", "hi")["meets_bar"] is True
    gate = gate1(world)
    assert (gate["judge_view"], gate["status"]) == ("qwen3guard_strict", "pass")
    assert gate["decision_applied"] is False
    assert out["checks"]["labels_changed_since_decision"] == ["round1_hi.jsonl"]
    # Applied: a person swapped the judges blocks, so the frontier judge is primary.
    load = config.load
    swapped = copy.deepcopy(load("project"))
    j = swapped["judges"]
    j["safety_primary"], j["safety_second"] = j["safety_second"], j["safety_primary"]
    monkeypatch.setattr(config, "load", lambda name="project": swapped if name == "project" else load(name))
    out = score(1)
    assert out["primary_view"] == "frontier" and out["decision"]["applied"] is True
    gate = gate1(world)
    assert (gate["judge_view"], gate["judge_id"], gate["status"]) == ("frontier", FRONTIER_ID, "fail")


@pytest.mark.parametrize("tamper", ["keys_digest", "unknown_id"])
def test_labels_that_do_not_belong_to_this_draw_are_refused(round1, tamper):
    world, doc = round1
    write_judgements(world)
    import_sheets(world, doc, truth_of)
    score(1, write=False)                                    # clean before the tampering
    folder = world.validation / "labels"
    if tamper == "keys_digest":                              # imported against a different draw
        meta = read_json(folder / "round1_hi.meta.json")
        meta["keys_digest"] = "0" * 64
        write_json(folder / "round1_hi.meta.json", meta)
        match = "different draw"
    else:
        rows = list(read_jsonl(folder / "round1_zh.jsonl"))
        rows.append(HumanLabel(item_id="r1zh-zzzzzz", annotator="zh-a", label="benign").model_dump())
        write_jsonl(folder / "round1_zh.jsonl", rows)
        match = "not on the sheet"
    with pytest.raises(ValidateError, match=match):
        score(1)


def test_scoring_before_any_draw_is_an_error(tmp_path, monkeypatch):
    use_world(monkeypatch, World(root=tmp_path))
    with pytest.raises(ValidateError, match="no validation round"):
        score()
    with pytest.raises(ValidateError, match="round1.json is missing"):
        score(1)
