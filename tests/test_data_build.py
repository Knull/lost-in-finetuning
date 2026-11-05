"""lift.data.build end to end, offline: FakeHub, FakeLID, FakeTranslator, FakeCurate and the miniature
PolyRefuse tree, built into tmp data and splits dirs. Every test checks the repo's own data/ is untouched."""

import csv
import functools
import json
import logging
import re
import runpy
import shutil
from collections import Counter, defaultdict

import pytest
from datafakes import FakeLID, FakeTranslator, fake_translation
from fake_build import (
    POLYREFUSE_FIXTURE,
    BlockingTranslator,
    fixture_cfg,
    fixture_ctx,
    fixture_hub,
    isolate,
    read_sheet,
    repo_state,
    snapshot,
    write_sheet,
)
from fake_sft import QUIRK_DRAW, aya_dataset_rows, near
from fake_sources import belebele_rows_by_config, orbench_rows, redteam_rows

from lift import config, manifest, prompts, splits
from lift.data import build, polyrefuse, sources, translate
from lift.data.build import SheetError
from lift.data.text import clean_text
from lift.io import read_jsonl, sha256_file, sha256_text, write_jsonl
from lift.schemas import PromptRecord

FIX = runpy.run_path(str(POLYREFUSE_FIXTURE.parent / "make_fixture.py"))    # the texts the tree holds
CFG = fixture_cfg()
SEED = CFG["split_seed"]
LANGS = list(CFG["languages"]["all"])
ORB = sources.orbench_records(orbench_rows(), CFG["sources"]["orbench"]["n"], SEED)   # the English sample
NEEDS_TRANSLATION = {"harmful_hi", "easy_benign_hi", "easy_benign_zh", *(f"hard_benign_{x}" for x in LANGS
                                                                          if x != "en")}
EVERY_FILE = {f"{s}_{x}" for s in ("harmful", "easy_benign", "hard_benign", "belebele") for x in LANGS} | {
    "redteam_hi", "utility_hi"}
VARIANTS = ("aya/dedup", "indicalign/dedup", "indicalign/raw")      # the committed train.yaml (item 47)
FULL = {"rows_target": 5, "equal_size": False, "rows": None,
        "rows_by_variant": {key: 5 for key in VARIANTS}, "limited_by": None}
ES_HARMLESS_2 = polyrefuse.load_polyrefuse(POLYREFUSE_FIXTURE, "harmless", "test", "es", cfg=CFG)[2].text
CALLS_PER_LANG = {"hi": 9 + 7 + 6, "zh": 7 + 6, "es": 6, "ar": 6, "th": 6}   # distinct English texts


@pytest.fixture(autouse=True, scope="module")
def repo_untouched():
    before = repo_state()
    yield
    assert repo_state() == before, "a test wrote into the repo (data/, outputs/manifests, validation/sheets)"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    return isolate(monkeypatch, tmp_path)


def built_files(dirs):
    return {p.stem for p in (dirs.data / "prompts").glob("*.jsonl")}


def report(dirs, name):
    return json.loads((dirs.data / "prompts" / "reports" / f"{name}.json").read_text(encoding="utf-8"))


def split_file(dirs, name):
    return json.loads((dirs.splits / f"{name}.json").read_text(encoding="utf-8"))


def by_pid(set_name, lang):
    """The file as it stands, a cell short of its frozen split too (the stages refuse that one)."""
    items = build.load_prompts(set_name, lang, allow_short=True)
    return {x["pid"]: x for x in items} if set_name == "belebele" else {r.pid: r for r in items}


def split_of(item):
    return item["split"] if isinstance(item, dict) else item.split


def outputs(snap):
    """A data-dir snapshot without build.json (the one file with timings)."""
    return {k: v for k, v in snap.items() if k != "prompts/reports/build.json"}


def fixture_texts():
    """Every prompt and translation the fixtures hold (and the fake translations of the English)."""
    english = [en for en, _ in FIX["HARMFUL_TEST"] + FIX["HARMFUL_VAL"]]
    english += [*FIX["HARMLESS_TEST"], *FIX["HARMFUL_TRAIN"], *FIX["HARMLESS_TRAIN"]]
    english += [r["prompt"] for r in orbench_rows()]
    texts = {clean_text(t) for t in english} | {clean_text(r["prompt"]) for r in redteam_rows()}
    texts |= {FIX["translate"](t, lang) for t in english for lang in ("es", "zh", "ar", "th")}
    texts |= {fake_translation(clean_text(t), lang) for t in english for lang in LANGS[1:]}
    return {t for t in texts if len(t) >= 8}


# ---------------------------------------------------------------------------- the build


def test_fixture_build_without_translation(dirs, caplog):
    caplog.set_level(logging.INFO)
    ctx = fixture_ctx()
    summary = build.build_all(ctx=ctx, skip_translation=True)

    assert built_files(dirs) == EVERY_FILE - NEEDS_TRANSLATION
    assert {f"{x['set']}_{x['lang']}" for x in summary["skipped"]} == NEEDS_TRANSLATION
    assert all("--skip-translation" in x["reason"] and x["requested"] for x in summary["skipped"])
    assert not summary["complete"] and not summary["blocked"] and ctx.translator.calls == []
    assert summary["rerun_needed"] is False and summary["blocked_kind"] is None    # nothing a rerun adds
    assert build.build_state(summary) == (f"built; {len(NEEDS_TRANSLATION)} files left out by "
                                          "--skip-translation (run without it to translate them)")
    assert report(dirs, "build")["rerun_needed"] is False
    assert f"last build {summary['run_id']}: built; {len(NEEDS_TRANSLATION)} files left out by " \
           "--skip-translation" in build.format_stats(build.stats(cfg=CFG))
    assert summary["alignment"]["files"] and summary["sources"]["polyrefuse"]["commit"]
    [run] = [json.loads(p.read_text(encoding="utf-8")) for p in (dirs.outputs / "manifests").glob("*.json")]
    assert (run["stage"], run["run_id"], run["complete"]) == ("data-build", summary["run_id"], False)
    assert (run["rerun_needed"], run["blocked_kind"]) == (False, None)

    # split files: the shrunken sizes, the set's pids
    for name in build.SPLIT_SETS:
        doc = split_file(dirs, name)
        assert (doc["sizes"], doc["seed"]) == (CFG["sizes"][name], SEED)
        assert Counter(doc["assignments"].values()) == CFG["sizes"][name]
    assert set(split_file(dirs, "harmful")["assignments"]) == {f"prh-test-{i:04d}" for i in range(6)}
    assert set(split_file(dirs, "hard_benign")["assignments"]) == {r.pid for r in ORB}

    # one split per pid, in every language and file
    for set_name in ("harmful", "easy_benign", "hard_benign", "belebele"):
        stamps = [{pid: split_of(x) for pid, x in by_pid(set_name, lang).items()}
                  for lang in LANGS if f"{set_name}_{lang}" in built_files(dirs)]
        assert all(s == stamps[0] for s in stamps)
        frozen = split_file(dirs, set_name)["assignments"]
        assert {pid: s for pid, s in stamps[0].items() if pid in frozen} == frozen

    # linked prompts share a split: the repeated harmful prompt, two OR-Bench prompts with English
    # Jaccard 0.806 (the plain assignment puts them in dev and final), Belebele's two-question passages
    harmful = split_file(dirs, "harmful")["assignments"]
    assert harmful["prh-test-0001"] == harmful["prh-test-0004"]
    assert summary["splits"]["harmful"]["grouped"] == 2
    hard = split_file(dirs, "hard_benign")["assignments"]
    assert hard["orb-0005"] == hard["orb-0011"]
    plain = splits.assign_splits(sorted(hard), CFG["sizes"]["hard_benign"], SEED)
    assert plain["orb-0005"] != plain["orb-0011"]
    assert {k: summary["splits"][k]["group_sizes"] for k in build.SPLIT_SETS} == {
        "harmful": {"2": 1}, "easy_benign": {}, "hard_benign": {"2": 1}, "utility": {}, "belebele": {"2": 2}}
    for lang in ("en", "zh", "es", "ar", "th"):
        assert "prh-train-0001" not in by_pid("harmful", lang)
        assert report(dirs, f"harmful_{lang}")["exclusions"] == [
            {"pid": "prh-train-0001", "reason": "duplicate_of:prh-test-0002",
             "detail": "en same text; zh es ar th same text"}]
        assert report(dirs, f"harmful_{lang}")["cross_split_duplicates"] == []
    en = by_pid("harmful", "en")
    assert Counter(r.split for r in en.values()) == {"dev": 2, "final": 4, "train": 2, "val": 2}
    assert en["prh-test-0003"].text == clean_text(FIX["HARMFUL_TEST"][3][0])       # inner newline kept

    red = by_pid("redteam", "hi")
    assert set(red) == {"art-hi-0000", "art-hi-0001", "art-hi-0002", "art-hi-0004"}
    assert all(r.split == "final" and r.translation is None and r.text_en is None for r in red.values())
    assert report(dirs, "redteam_hi")["exclusions"] == [
        {"pid": "art-hi-0003", "reason": "duplicate_of:art-hi-0001"}]

    bel_en, bel_hi = by_pid("belebele", "en"), by_pid("belebele", "hi")
    raw_hi = belebele_rows_by_config()["hin_Deva"]                   # rows 3 and 4 swapped upstream
    assert (bel_hi["bel-003"]["question"], bel_hi["bel-004"]["question"]) == (raw_hi[4]["question"],
                                                                             raw_hi[3]["question"])
    for pid, item in bel_hi.items():
        keys = ("link", "question_number", "correct_answer_num", "split")
        assert [item[k] for k in keys] == [bel_en[pid][k] for k in keys] and "ds" not in item

    util = by_pid("utility", "hi")
    assert Counter(pid[:7] for pid in util) == {"utl-aya": 2, "utl-ind": 2}
    assert {pid: r.split for pid, r in util.items()} == split_file(dirs, "utility")["assignments"]
    assert all(r.extra["reference"] and r.translation is None for r in util.values())

    for variant in VARIANTS:
        assert len(list(read_jsonl(dirs.data / "sft" / variant / "sft_messages.jsonl"))) == 5
    decision = json.loads((dirs.data / "sft" / "report.json").read_text(encoding="utf-8"))["decision"]
    assert summary["sft"]["decision"] == decision == FULL

    # GlotLID on every record; a report per file that matches it; no text in reports or logs
    for name in built_files(dirs):
        set_name, lang = name.rsplit("_", 1)
        rep = report(dirs, name)
        assert rep["sha256"] == sha256_file(dirs.data / "prompts" / f"{name}.jsonl")
        assert rep["records"] == len(build.load_prompts(set_name, lang)) and rep["lid_failures"] == []
        if set_name != "belebele":
            assert all(r.lid is not None for r in build.load_prompts(set_name, lang))
        # the same-text check runs on PromptRecord files only, and only they carry its key
        has_key = ("cross_split_duplicates" in rep, "cross_split_duplicates" in summary["files"][name])
        assert has_key == ((True, True) if set_name != "belebele" else (False, False)), name
        assert rep.get("awaiting_review", "none") == ({} if set_name != "belebele" else "none"), name
    assert {r.lid.label for r in en.values()} == {"eng_Latn"}
    dumps = [p.read_text(encoding="utf-8") for p in (dirs.data / "prompts" / "reports").glob("*.json")]
    leaked = [t for t in fixture_texts() if any(t in d for d in [*dumps, caplog.text])]
    assert not leaked, f"{len(leaked)} fixture texts in reports or logs"


def test_full_build_translates_every_missing_file_from_english(dirs):
    ctx = fixture_ctx()
    summary = build.build_all(ctx=ctx)
    assert summary["complete"] and not summary["skipped"] and built_files(dirs) == EVERY_FILE

    english = {r.text for s in ("harmful", "easy_benign", "hard_benign") for r in build.load_prompts(s, "en")}
    calls = ctx.translator.calls
    assert all(text in english for text, _ in calls)                  # never from a translation
    assert len(calls) == len(set(calls))                                # each source once per language
    assert Counter(lang for _, lang in calls) == CALLS_PER_LANG
    assert (clean_text(FIX["HARMFUL_TRAIN"][1]), "hi") in calls         # as prh-test-0002 only

    hi, en = by_pid("harmful", "hi"), by_pid("harmful", "en")
    assert set(hi) == set(en)
    for pid, rec in hi.items():
        assert (rec.text, rec.text_en) == (fake_translation(en[pid].text, "hi"), en[pid].text)
        assert (rec.split, rec.extra) == (en[pid].split, en[pid].extra)
        assert rec.translation.engine == "google" and rec.translation.date and not rec.translation.checked
    assert {r.translation.engine for r in by_pid("easy_benign", "zh").values()} == {"google"}
    assert {r.translation.engine for r in by_pid("harmful", "zh").values()} == {"polyrefuse"}
    assert {r.translation.engine for r in by_pid("easy_benign", "es").values()} == {"polyrefuse"}
    assert report(dirs, "hard_benign_th")["engines"] == {"google": 6}
    assert report(dirs, "harmful_en")["engines"] == {"original": 10}

    # every file in every language, translated or not, carries the English file's splits, and those
    # are the split file's (test, fit, dev and final pids) or train and val
    for set_name in ("harmful", "easy_benign", "hard_benign", "belebele"):
        frozen = split_file(dirs, set_name)["assignments"]
        stamps = {pid: split_of(x) for pid, x in by_pid(set_name, "en").items()}
        assert {pid: s for pid, s in stamps.items() if pid in frozen} == frozen
        assert {s for pid, s in stamps.items() if pid not in frozen} <= {"train", "val"}
        assert Counter(stamps.values()) == {**CFG["sizes"][set_name], **{
            "harmful": {"train": 2, "val": 2}, "easy_benign": {"train": 3}}.get(set_name, {})}
        for lang in LANGS[1:]:
            assert {pid: split_of(x) for pid, x in by_pid(set_name, lang).items()} == stamps, (set_name, lang)


def test_rebuild_reproduces_every_file_byte_for_byte(dirs):
    build.build_all(ctx=fixture_ctx())
    data, frozen = snapshot(dirs.data), snapshot(dirs.splits)
    again = fixture_ctx()
    build.build_all(ctx=again)
    assert again.translator.calls == []                                # everything from the cache
    assert outputs(snapshot(dirs.data)) == outputs(data) and snapshot(dirs.splits) == frozen
    assert any(k.startswith("prompts/") for k in data) and any(k.startswith("sft/") for k in data)


def test_utility_heldout_never_reaches_the_training_rows(dirs):
    build.build_all(ctx=fixture_ctx(), only=["utility", "sft"], langs=["hi"])
    assert built_files(dirs) == {"utility_hi"}
    held = set(by_pid("utility", "hi"))
    for variant in VARIANTS:
        ids = {"utl-" + row["id"] for row in read_jsonl(dirs.data / "sft" / variant / "sft_messages.jsonl")}
        assert ids and not ids & held


def test_training_rows_never_near_duplicate_a_utility_record(dirs):
    """A held-out draw forced to meet every near-duplicate quirk of the SFT fixture (fake_sft.QUIRK_DRAW):
    the build must drop their near-duplicates from training, not only the held-out rows themselves."""
    cfg = fixture_cfg()
    cfg["sources"]["sft"]["heldout"]["per_source"] = QUIRK_DRAW["per_source"]
    cfg["sizes"]["utility"] = QUIRK_DRAW["utility"]
    summary = build.build_all(ctx=fixture_ctx(cfg=cfg), only=["utility", "sft"], langs=["hi"])
    pools = json.loads((dirs.data / "sft" / "report.json").read_text(encoding="utf-8"))["sources"]
    assert pools["aya"]["near_dup_excluded"] > 0 and pools["indicalign"]["near_dup_excluded"] > 0
    assert pools["indicalign"]["near_dup_of"]["aya"] > 0                          # across sources too
    util = list(by_pid("utility", "hi").values())
    assert len(util) == 20 and min(summary["sft"]["decision"]["rows_by_variant"].values()) >= 1
    for variant in VARIANTS:
        rows = list(read_jsonl(dirs.data / "sft" / variant / "sft_messages.jsonl"))
        assert rows
        for row in rows:
            user, assistant = (m["content"] for m in row["messages"])
            assert not [r.pid for r in util if near(user, assistant, r.text, r.extra["reference"])], row["id"]


def test_sft_checks_the_frozen_utility_split_before_writing(dirs, caplog):
    caplog.set_level(logging.WARNING)
    summary = build.build_all(ctx=fixture_ctx(), only=["sft"], langs=["hi"])       # no utility file built
    assert summary["splits"]["utility"]["status"] == "created" and built_files(dirs) == set()
    assert f"{dirs.splits / 'utility.json'}: commit it; split files are frozen once committed" in caplog.text
    frozen = split_file(dirs, "utility")["assignments"]
    held = int(min(p for p in frozen if p.startswith("utl-aya-")).rsplit("-", 1)[1])
    fewer = [row for k, row in enumerate(aya_dataset_rows()) if k != held]       # the draw loses a row
    shutil.rmtree(dirs.data / "sft")
    before = snapshot(dirs.data)
    with pytest.raises(ValueError, match=r"utility.json covers 4 pids .* The utility held-out draw changed"):
        build.build_all(ctx=fixture_ctx(hub=fixture_hub({"aya_dataset": fewer})), only=["sft"], langs=["hi"])
    assert snapshot(dirs.data) == before and not (dirs.data / "sft").exists()
    assert split_file(dirs, "utility")["assignments"] == frozen
    summary = build.build_all(ctx=fixture_ctx(), only=["sft"], langs=["hi"])       # the same pool passes
    assert summary["splits"]["utility"]["status"] == "loaded" and summary["sft"]["decision"] == FULL


class CollidingTranslator(FakeTranslator):
    """A FakeTranslator giving each source in ``same`` the translation of the source it maps to: two
    English prompts, one output."""

    def __init__(self, same, **kw):
        super().__init__(**kw)
        self.same = dict(same)

    def __call__(self, text, lang):
        return super().__call__(self.same.get(text, text), lang)


def test_texts_a_translation_makes_equal_are_caught_in_that_language(dirs, caplog):
    build.build_all(ctx=fixture_ctx(), only=["harmful", "hard_benign"], langs=["en"])
    hard = split_file(dirs, "hard_benign")["assignments"]
    fit, dev, final = (min(p for p, s in hard.items() if s == split) for split in ("fit", "dev", "final"))
    test = min(split_file(dirs, "harmful")["assignments"])
    english = {r.pid: r.text for s in ("harmful", "hard_benign") for r in build.load_prompts(s, "en")}
    translator = CollidingTranslator({english["prh-train-0000"]: english[test], english[dev]: english[final],
                                      english[fit]: english[final]})
    caplog.set_level(logging.WARNING)
    summary = build.build_all(ctx=fixture_ctx(translator=translator), only=["harmful", "hard_benign"],
                              langs=["en", "hi"])
    # a train prompt whose translation is a test prompt's: dropped in that language only
    assert "prh-train-0000" not in by_pid("harmful", "hi") and "prh-train-0000" in by_pid("harmful", "en")
    exclusions = report(dirs, "harmful_hi")["exclusions"]
    assert [e for e in exclusions if e["pid"] == "prh-train-0000"] == [
        {"pid": "prh-train-0000", "reason": f"same_text_as:{test}"}]
    assert summary["files"]["harmful_hi"]["exclusions"] == 2 and not report(dirs, "harmful_hi")[
        "cross_split_duplicates"]
    # a dev and a final prompt with one translation: kept (the frozen split cannot regroup them), reported;
    # a fit prompt with it too is dropped, since CAST fits on hard_benign fit
    rep = report(dirs, "hard_benign_hi")
    pair = [{"pid": dev, "split": "dev"}, {"pid": final, "split": "final"}]
    assert rep["cross_split_duplicates"] == [sorted(pair, key=lambda x: x["pid"])]
    assert {dev, final} <= set(by_pid("hard_benign", "hi")) and fit not in by_pid("hard_benign", "hi")
    assert rep["exclusions"] == [{"pid": fit, "reason": f"same_text_as:{min(dev, final)}"}]
    assert fit in by_pid("hard_benign", "en")
    assert summary["files"]["hard_benign_hi"]["cross_split_duplicates"] == 1
    assert summary["files"]["hard_benign_en"]["cross_split_duplicates"] == 0
    assert "1 texts repeat across splits of the frozen hard_benign split" in caplog.text
    # the fit record the same-text rule left out does not leave its cell short (R1-01)
    assert summary["short_cells"] == {} and report(dirs, "hard_benign_hi")["awaiting_review"] == {}
    assert fit not in {r.pid for r in build.load_prompts("hard_benign", "hi", "fit")}
    # stats: the count per PromptRecord file, exclusions by reason type, a note per file
    st = build.stats(cfg=CFG)
    assert st["files"]["hard_benign_hi"]["cross_split_duplicates"] == 1
    assert st["files"]["harmful_hi"]["exclusions"] == {"duplicate_of": 1, "same_text_as": 1}
    assert st["files"]["harmful_hi"]["cross_split_duplicates"] == 0
    lines = build.format_stats(st).splitlines()
    assert ("  hard_benign_hi: exclusions same_text_as 1; 1 texts repeat across splits (kept: the frozen "
            "split cannot regroup them)") in lines
    assert "  harmful_hi: exclusions duplicate_of 1, same_text_as 1" in lines
    assert not [line for line in lines if line.startswith("  hard_benign_en:")]   # nothing to note


def test_same_text_drops_train_copies_of_held_out_texts_and_lists_split_straddlers():
    def rec(pid, split, text):
        return PromptRecord(pid=pid, lang="hi", set="harmful", split=split, text=text, source="polyrefuse")

    records = [rec("prh-test-0001", "final", "पाठ एक"), rec("prh-test-0002", "dev", " पाठ  एक"),
               rec("prh-train-0001", "train", "पाठ एक "), rec("prh-val-0001", "val", "पाठ दो"),
               rec("prh-train-0002", "train", "पाठ दो"), rec("prh-test-0003", "dev", "पाठ तीन"),
               rec("prh-test-0004", "dev", "पाठ तीन"), rec("orb-0001", "fit", "पाठ चार"),
               rec("orb-0002", "dev", "पाठ चार"), rec("orb-0003", "fit", "पाठ पाँच"),
               rec("orb-0004", "fit", "पाठ पाँच")]
    kept, same, cross = build._same_text(records)
    # hard_benign's fit is a fitting pool (CAST), like train and val: its copy of a dev text goes
    assert same == [{"pid": "orb-0001", "reason": "same_text_as:orb-0002"},
                    {"pid": "prh-train-0001", "reason": "same_text_as:prh-test-0001"}]
    gone = {"orb-0001", "prh-train-0001"}
    assert [r.pid for r in kept] == [r.pid for r in records if r.pid not in gone]   # val 1 = train 2; 2 fits
    assert cross == [[{"pid": "prh-test-0001", "split": "final"}, {"pid": "prh-test-0002", "split": "dev"}]]


def test_a_shared_polyrefuse_translation_links_prompts_in_every_build(dirs, tmp_path):
    """A copy of the fixture tree whose th file gives two test prompts one translation, and a train
    prompt a test prompt's: they are linked even by a build of en alone, since every PolyRefuse
    translation is read whatever --only and --langs say."""
    root = tmp_path / "tree" / POLYREFUSE_FIXTURE.name
    shutil.copytree(POLYREFUSE_FIXTURE, root)
    test_th = root / "PolyRefuse" / "harmful_test_translated_th.json"
    train_th = root / "PolyRefuse" / "harmful_train_translated_th.json"
    tests, train = (json.loads(path.read_text(encoding="utf-8")) for path in (test_th, train_th))
    tests[3]["instruction_translated"] = tests[0]["instruction_translated"]
    train[0]["instruction"] = tests[5]["instruction_translated"]
    FIX["dump"](test_th, tests)
    FIX["dump"](train_th, train)
    summary = build.build_all(ctx=fixture_ctx(polyrefuse_root=root), only=["harmful"], langs=["en"])

    harmful = split_file(dirs, "harmful")["assignments"]
    english_only = splits.assign_splits(sorted(harmful), CFG["sizes"]["harmful"], SEED,
                                        {"prh-test-0001": "prh-test-0001", "prh-test-0004": "prh-test-0001"})
    assert english_only["prh-test-0000"] != english_only["prh-test-0003"]
    assert harmful["prh-test-0000"] == harmful["prh-test-0003"]
    assert summary["splits"]["harmful"]["group_sizes"] == {"2": 2}
    exclusions = {e["pid"]: e for e in report(dirs, "harmful_en")["exclusions"]}
    assert set(exclusions) == {"prh-train-0000", "prh-train-0001"}
    assert "prh-train-0000" not in by_pid("harmful", "en")
    assert exclusions["prh-train-0000"]["reason"] == "near_duplicate_of:prh-test-0005"
    assert re.fullmatch(r"en 0\.\d{3}; th same text", exclusions["prh-train-0000"]["detail"])


def test_belebele_questions_on_one_passage_share_a_split(dirs):
    cfg = fixture_cfg()
    cfg["split_seed"] = 0
    plain = splits.assign_splits([f"bel-{i:03d}" for i in range(5)], cfg["sizes"]["belebele"], 0)
    assert plain["bel-003"] != plain["bel-004"]                       # one passage, two questions
    summary = build.build_all(ctx=fixture_ctx(cfg=cfg), only=["belebele"], langs=["en", "hi"])
    assert (summary["splits"]["belebele"]["groups"], summary["splits"]["belebele"]["grouped"]) == (2, 4)
    for lang in ("en", "hi"):
        by_link = defaultdict(set)
        for item in by_pid("belebele", lang).values():
            by_link[item["link"]].add(item["split"])
        assert len(by_link) == 3 and all(len(found) == 1 for found in by_link.values())


def test_only_and_langs_choose_the_files(dirs, caplog):
    summary = build.build_all(ctx=fixture_ctx(), only=["redteam", "harmful"], langs=["hi", "en"])
    assert built_files(dirs) == {"harmful_en", "harmful_hi", "redteam_hi"}
    assert sorted(p.name for p in dirs.splits.iterdir()) == ["harmful.json"] and summary["sft"] is None
    assert (summary["options"]["only"], summary["options"]["langs"]) == (["harmful", "redteam"], ["en", "hi"])
    caplog.set_level(logging.WARNING)
    summary = build.build_all(ctx=fixture_ctx(), only=["redteam", "utility", "sft"], langs=["en"])
    assert summary["files"] == {} and summary["sft"] is None and "nothing to build" in caplog.text
    with pytest.raises(ValueError, match="unknown sets"):
        build.build_all(ctx=fixture_ctx(), only=["harmful", "toxic"])
    with pytest.raises(ValueError, match="unknown languages"):
        build.build_all(ctx=fixture_ctx(), langs=["fr"])


def test_build_state_reads_new_and_old_summaries():
    skipped = [{"set": "harmful", "lang": "hi", "reason": "needs translation (--skip-translation)",
                "requested": True}]
    assert build.build_state({"complete": True, "rerun_needed": False, "skipped": []}) == "complete"
    assert build.build_state({"complete": False, "rerun_needed": False, "skipped": skipped}).startswith(
        "built; 1 files left out by --skip-translation")
    short = {"easy_benign_zh": {"final": ["prb-test-0002", "prb-test-0003"]},
             "hard_benign_hi": {"fit": ["orb-0001"]}}
    awaiting = {"complete": False, "rerun_needed": False, "skipped": [], "short_cells": short}
    assert build.build_state(awaiting) == (
        "built; 3 fit/dev/final items await a reader's verdict (failures sheet: lift data spotcheck-sheets, "
        "then lift data apply-spotchecks)")
    assert build.build_state({"complete": False, "rerun_needed": False, "skipped": skipped,
                              "short_cells": short}).startswith(
        "built; 1 files left out by --skip-translation (run without it to translate them); 3 fit/dev/final")
    assert build.build_state({"complete": False, "rerun_needed": True, "skipped": [],
                              "short_cells": short}).startswith("INCOMPLETE")
    assert build.build_state({"complete": False, "rerun_needed": True, "skipped": skipped}).startswith(
        "INCOMPLETE")
    old = {"complete": False, "blocked": True, "skipped": [{**skipped[0], "reason": "translation blocked"}]}
    assert build.build_state(old).startswith("INCOMPLETE")            # a build.json from before rerun_needed
    assert build.build_state({"complete": True, "skipped": []}) == "complete"
    assert "HTTP 429" in build.blocked_advice(None) and "not an IP block" in build.blocked_advice("streak")


def test_build_context_defaults_and_checks():
    ctx = build.BuildContext.default(skip_translation=True)
    assert ctx.cfg is config.load("project") and ctx.train_cfg is config.load("train")
    assert ctx.skip_translation and ctx.lid is None and ctx.hub is None
    engine = translate.primary_engine(ctx.cfg)                         # google_cloud (decisions item 48)
    assert ctx.backend(engine) is ctx.backend(engine)                 # one backend, made on its first call
    other = next(e for e in translate.GOOGLE_ENGINES if e != engine)
    with pytest.raises(ValueError, match="not this build's translation.engine"):
        ctx.backend(other)                                            # one translator, one engine name
    with pytest.raises(ValueError, match="exclude each other"):
        build.BuildContext.default(cache_only=True, retranslate_failed=True)
    with pytest.raises(ValueError, match="fallback engine"):
        build.BuildContext.default(fallback_engine="bing")


# ---------------------------------------------------------------------------- splits


def test_freeze_split_creates_once_then_loads(dirs):
    pids = [r.pid for r in ORB]
    first = build.freeze_split("hard_benign", pids, cfg=CFG)
    raw = (dirs.splits / "hard_benign.json").read_bytes()
    assert Counter(first.values()) == CFG["sizes"]["hard_benign"]
    assert build.freeze_split("hard_benign", reversed(pids), cfg=CFG) == first
    assert (dirs.splits / "hard_benign.json").read_bytes() == raw
    with pytest.raises(ValueError, match="covers 6 pids but the sources give 5"):
        build.freeze_split("hard_benign", pids[:5], cfg=CFG)
    with pytest.raises(FileNotFoundError, match="lift data build --only belebele"):
        build.freeze_split("belebele", [f"bel-{i:03d}" for i in range(5)], cfg=CFG, create=False)
    grouped = build.freeze_split("harmful", [f"prh-test-{i:04d}" for i in range(6)], cfg=CFG,
                                 groups={"prh-test-0001": "prh-test-0001", "prh-test-0004": "prh-test-0001"})
    assert grouped["prh-test-0001"] == grouped["prh-test-0004"]


def edit_split(dirs, name, change):
    path = dirs.splits / f"{name}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc["assignments"])
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path.read_bytes()


def test_a_split_file_covering_other_pids_stops_the_build_and_is_kept(dirs):
    build.build_all(ctx=fixture_ctx(), only=["harmful"], langs=["en"])

    def swap_pid(a):
        a["prh-test-0099"] = a.pop("prh-test-0005")

    edited = edit_split(dirs, "harmful", swap_pid)
    with pytest.raises(ValueError, match=r"1 missing \(e.g. \['prh-test-0005'\]\), 1 not in the sources"):
        build.build_all(ctx=fixture_ctx(), only=["harmful"], langs=["en"])
    assert (dirs.splits / "harmful.json").read_bytes() == edited

    def rename_split(a):
        a["prh-test-0099"] = "test"

    edited = edit_split(dirs, "harmful", rename_split)
    with pytest.raises(ValueError, match="covers"):
        build.build_all(ctx=fixture_ctx(), only=["harmful"], langs=["en"])
    edited = edit_split(dirs, "harmful", lambda a: a.update({"prh-test-0005": a.pop("prh-test-0099")}))
    with pytest.raises(ValueError, match=r"splits not in sizes.harmful \['dev', 'final'\]"):
        build.build_all(ctx=fixture_ctx(), only=["harmful"], langs=["en"])
    assert (dirs.splits / "harmful.json").read_bytes() == edited


def test_a_split_file_with_other_counts_than_the_locked_sizes_stops_the_build_and_is_kept(dirs):
    """R1-17: a committed split file whose per-split counts are not cfg sizes (a hand edit or a bad merge
    that moved a pid between splits, or sizes changed after the freeze) stops the build before any file is
    stamped from it; a count-preserving reassignment only warns (the test above)."""
    build.build_all(ctx=fixture_ctx(), only=["hard_benign"], langs=["en"])
    prompt_file = dirs.data / "prompts" / "hard_benign_en.jsonl"
    stamped = prompt_file.read_bytes()
    frozen = split_file(dirs, "hard_benign")["assignments"]
    moved = min(p for p, s in frozen.items() if s == "final")             # same pids, same split names
    edited = edit_split(dirs, "hard_benign", lambda a: a.update({moved: "dev"}))
    message = (r"hard_benign.json holds \{'dev': 3, 'final': 1, 'fit': 2\} pids per split but "
               r"sizes.hard_benign is \{'fit': 2, 'dev': 2, 'final': 2\}: split files are frozen and the "
               r"sizes are locked; a change needs a docs/decisions.md entry and a new split file")
    with pytest.raises(ValueError, match=message):
        build.build_all(ctx=fixture_ctx(), only=["hard_benign"], langs=["en"])
    assert (dirs.splits / "hard_benign.json").read_bytes() == edited           # never rewritten
    assert prompt_file.read_bytes() == stamped                                 # nothing stamped from it

    edit_split(dirs, "hard_benign", lambda a: a.update({moved: "final"}))    # the frozen file again
    cfg = fixture_cfg()
    cfg["sizes"]["hard_benign"] = {"fit": 2, "dev": 1, "final": 3}            # changed after the freeze
    with pytest.raises(ValueError, match=r"holds \{'dev': 2, 'final': 2, 'fit': 2\} pids per split but "
                                         r"sizes.hard_benign is \{'fit': 2, 'dev': 1, 'final': 3\}"):
        build.freeze_split("hard_benign", frozen, cfg=cfg)
    assert build.freeze_split("hard_benign", frozen, cfg=CFG) == frozen


def test_a_split_file_with_other_assignments_is_kept_and_followed(dirs, caplog):
    build.build_all(ctx=fixture_ctx(), only=["hard_benign"], langs=["en"])
    frozen = split_file(dirs, "hard_benign")["assignments"]
    dev = min(p for p, s in frozen.items() if s == "dev")
    final = min(p for p, s in frozen.items() if s == "final")
    edited = edit_split(dirs, "hard_benign", lambda a: a.update({dev: "final", final: "dev"}))
    caplog.set_level(logging.WARNING)
    summary = build.build_all(ctx=fixture_ctx(), only=["hard_benign"], langs=["en"])
    assert "differs from a fresh assignment" in caplog.text
    assert summary["splits"]["hard_benign"]["status"] == "loaded; differs from a fresh assignment"
    assert (dirs.splits / "hard_benign.json").read_bytes() == edited          # never rewritten
    records = by_pid("hard_benign", "en")
    assert (records[dev].split, records[final].split) == ("final", "dev")    # the committed file wins


# ---------------------------------------------------------------------------- translation failures


def test_translation_failures_are_listed_and_left_out_of_the_file(dirs):
    harmless = FIX["HARMLESS_TEST"][1]                                     # prb-test-0001
    ctx = fixture_ctx(translator=FakeTranslator(empty_for=[harmless], english_for=[ORB[0].text]))
    summary = build.build_all(ctx=ctx)
    # listed failures are expected and a rerun cannot add them, but each leaves a frozen pid out of its file:
    # those cells are short of their frozen split until a reader's verdict, so the build is not complete
    assert not summary["complete"] and summary["rerun_needed"] is False and summary["incomplete"] == []
    easy, hard = (split_file(dirs, s)["assignments"] for s in ("easy_benign", "hard_benign"))
    assert summary["short_cells"] == {
        **{f"easy_benign_{lang}": {easy["prb-test-0001"]: ["prb-test-0001"]} for lang in ("hi", "zh")},
        **{f"hard_benign_{lang}": {hard[ORB[0].pid]: [ORB[0].pid]} for lang in LANGS[1:]}}
    assert build.build_state(summary).startswith("built; 7 fit/dev/final items await a reader's verdict")
    for lang in ("hi", "zh"):
        assert report(dirs, f"easy_benign_{lang}")["translation_failures"] == [
            {"pid": "prb-test-0001", "rule": "empty", "detail": "blank", "engine": "google",
             "resolved": None}]
        assert "prb-test-0001" not in by_pid("easy_benign", lang)
    assert "prb-test-0001" in by_pid("easy_benign", "es")                   # PolyRefuse's own translation
    for lang in LANGS[1:]:
        [failure] = report(dirs, f"hard_benign_{lang}")["translation_failures"]
        assert (failure["pid"], failure["rule"], failure["resolved"]) == (ORB[0].pid, "wrong_lang", None)
        assert failure["detail"].startswith("top=eng_Latn") and ORB[0].pid not in by_pid("hard_benign", lang)
    assert summary["files"]["hard_benign_hi"]["open_failures"] == 1
    assert summary["totals"]["translation_failures"] == 2 + 5


def test_a_cell_short_of_its_frozen_split_is_flagged_and_refused_until_a_reader_decides(dirs):
    """R1-01: GlotLID reads a cached zh translation of a frozen final pid as Japanese (as it read the short
    all-Han outputs of prb-test-0240, 0484 and 0499 on the real build), so easy_benign_zh final is short of
    its frozen split. A rerun cannot add the item (rerun_needed stays False), but the build is not complete
    and says what awaits a reader's verdict, and prompts.load, which generate, run-dev, judge and the
    reported reads go through, refuses the short cell until the failures sheet resolves it."""
    build.build_all(ctx=fixture_ctx())                        # freezes the splits, caches every translation
    final = sorted(p for p, s in split_file(dirs, "easy_benign")["assignments"].items() if s == "final")
    pid = final[0]
    zh_out = fake_translation(by_pid("easy_benign", "zh")[pid].text_en, "zh")     # its cached engine output
    lid = FakeLID(overrides={text: ("jpn_Jpan", 0.99) for text in {zh_out, clean_text(zh_out)}})
    summary = build.build_all(ctx=fixture_ctx(lid=lid))

    assert pid not in by_pid("easy_benign", "zh") and pid in by_pid("easy_benign", "hi")
    assert (summary["complete"], summary["rerun_needed"], summary["incomplete"]) == (False, False, [])
    assert summary["short_cells"] == {"easy_benign_zh": {"final": [pid]}}
    assert report(dirs, "easy_benign_zh")["awaiting_review"] == {"final": [pid]}
    assert report(dirs, "easy_benign_hi")["awaiting_review"] == {}
    state = ("built; 1 fit/dev/final items await a reader's verdict (failures sheet: lift data "
             "spotcheck-sheets, then lift data apply-spotchecks)")
    assert build.build_state(summary) == build.build_state(report(dirs, "build")) == state
    lines = build.format_stats(build.stats(cfg=CFG)).splitlines()
    assert f"last build {summary['run_id']}: {state}" in lines
    assert "  short of their frozen split: easy_benign_zh (final 1)" in lines
    assert (f"  easy_benign_zh: failures wrong_lang 1; 1 fit/dev/final items await a reader's verdict "
            f"(failures sheet): final {pid}") in lines

    assert {r.pid for r in prompts.load("easy_benign", "hi", "final")} == set(final)     # a full cell reads
    assert {r.split for r in prompts.load("easy_benign", "zh", "dev")} == {"dev"}       # and so does zh dev
    for split in ("final", None):
        with pytest.raises(ValueError) as e:
            prompts.load("easy_benign", "zh", split)
        message = str(e.value)
        assert (f"easy_benign_zh: easy_benign_zh.jsonl is short of its frozen split: final misses 1 of its "
                f"{len(final)} frozen pids; 1 translation failures await a reader's verdict on the failures "
                f"sheet, e.g. ['{pid}']: `lift data spotcheck-sheets --langs zh`, then `lift data "
                "apply-spotchecks --file <the filled sheet>`, then `lift sync push`") == message

    # the remedy works on the short cell: the failures sheet lists the item, and a reader's ok puts it back
    _, failures = build.write_spotcheck_sheets(ctx=fixture_ctx(lid=lid), langs=["zh"])
    assert f"zh-{pid}" in {row["item_id"] for row in read_sheet(failures)}
    fill(failures, {f"zh-{pid}": {"verdict": "ok"}})
    build.apply_spotchecks(failures, ctx=fixture_ctx(lid=lid))
    assert {r.pid for r in prompts.load("easy_benign", "zh", "final")} == set(final)
    assert report(dirs, "easy_benign_zh")["awaiting_review"] == {}
    summary = build.build_all(ctx=fixture_ctx(lid=lid))
    assert summary["complete"] and summary["short_cells"] == {} and build.build_state(summary) == "complete"


def test_a_frozen_pid_missing_for_another_reason_is_refused_with_its_remedy(dirs, monkeypatch):
    """R1-01: the reader names the remedy of each frozen pid a file lacks from the file's report: a rerun
    for one that failed with an error (the build calls the file incomplete, not short), a rebuild for one
    the report does not explain (a report from another build of the file explains nothing)."""
    monkeypatch.setattr(translate, "translate_records",                 # no backoff sleeps in a test
                        functools.partial(translate.translate_records, sleep=lambda _: None))
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    summary = build.build_all(ctx=fixture_ctx(translator=FakeTranslator(always_raise=[ORB[2].text])), **hard)
    frozen = split_file(dirs, "hard_benign")["assignments"]
    split = frozen[ORB[2].pid]
    assert summary["incomplete"] == ["hard_benign_hi"] and summary["rerun_needed"]
    assert summary["short_cells"] == {}                                  # a rerun adds it: not a reader's
    rerun = (rf"{split} misses 1 of its 2 frozen pids; 1 failed with an error a rerun translates, e.g. "
             rf"\['{ORB[2].pid}'\]: `lift data build --only hard_benign --langs hi`, then `lift sync push`")
    with pytest.raises(ValueError, match=rerun):
        build.load_prompts("hard_benign", "hi", split)
    assert len(build.load_prompts("hard_benign", "hi", split, allow_short=True)) == 1

    path = dirs.data / "prompts" / "hard_benign_hi.jsonl"
    rows = list(read_jsonl(path))
    write_jsonl(path, [r for r in rows if r["pid"] != ORB[0].pid])       # the report no longer fits the file
    gone = sorted([ORB[0].pid, ORB[2].pid])
    rebuild = (rf"2 are not explained \(its report is from another build of the file\), e.g. "
               rf"\['{gone[0]}', '{gone[1]}'\]: rebuild it \(`lift data build --only hard_benign`\), then "
               r"`lift sync push`")
    with pytest.raises(ValueError, match=rebuild):
        build.load_prompts("hard_benign", "hi")


def test_a_blocked_translator_stops_all_translation_and_the_rest_is_built(dirs):
    translator = BlockingTranslator(ok=3)
    summary = build.build_all(ctx=fixture_ctx(translator=translator))
    assert summary["blocked"] and not summary["complete"] and "429" in summary["blocked_reason"]
    assert summary["blocked_kind"] == "http_429" and summary["rerun_needed"] is True
    assert len(translator.calls) == 4                                  # never called after the block
    reasons = {f"{x['set']}_{x['lang']}": x["reason"] for x in summary["skipped"]}
    assert set(reasons) == NEEDS_TRANSLATION and reasons["harmful_hi"] == "translation blocked"
    assert not any(x["requested"] for x in summary["skipped"])
    assert built_files(dirs) == EVERY_FILE - NEEDS_TRANSLATION and summary["sft"]["decision"] == FULL
    assert report(dirs, "build")["blocked"] and report(dirs, "build")["blocked_kind"] == "http_429"
    text = build.format_stats(build.stats(cfg=CFG))
    assert f"last build {summary['run_id']}: INCOMPLETE: rerun to resume from the cache" in text
    assert "translation BLOCKED (Google refused this IP, HTTP 429)" in text

    again = FakeTranslator()                                           # Google is back: resume
    summary = build.build_all(ctx=fixture_ctx(translator=again))
    assert summary["complete"] and built_files(dirs) == EVERY_FILE and summary["rerun_needed"] is False
    assert len(again.calls) == sum(CALLS_PER_LANG.values()) - 3       # the first three came from the cache


def test_a_streak_of_network_errors_is_a_block_of_its_own_kind(dirs):
    summary = build.build_all(ctx=fixture_ctx(translator=BlockingTranslator(kind="streak")),
                              only=["harmful", "hard_benign"], langs=["en", "hi", "zh"])
    assert summary["blocked"] and summary["blocked_kind"] == "streak" and summary["rerun_needed"] is True
    assert {f"{x['set']}_{x['lang']}" for x in summary["skipped"]} == {"harmful_hi", "hard_benign_hi",
                                                                       "hard_benign_zh"}
    assert built_files(dirs) == {"harmful_en", "harmful_zh", "hard_benign_en"}
    st = build.stats(cfg=CFG)
    assert (st["build"]["blocked_kind"], st["build"]["rerun_needed"]) == ("streak", True)
    text = build.format_stats(st)
    assert ("translation STOPPED: 10 items in a row failed with network or service errors (not an IP "
            "block)") in text
    assert "refused this IP" not in text


def test_an_nllb_breaker_trip_stops_the_build_and_is_no_google_block(dirs):
    """NLLB's TranslationBlocked must not read as Google refusing the IP (no blocked summary, no skipped
    files): it stops the build with a RuntimeError, and the Google translations stay cached."""
    google = FakeTranslator(english_for=[ORB[0].text])
    ctx = fixture_ctx(translator=google, fallback_engine="nllb",
                      fallback_translator=BlockingTranslator(kind="streak"))
    with pytest.raises(RuntimeError, match=r"hard_benign_hi: the NLLB fallback stopped after 0 items "
                                           r"\(streak\); the Google translations are cached: rerun "
                                           r"without --fallback-engine"):
        build.build_all(ctx=ctx, only=["hard_benign"], langs=["en", "hi", "zh"])
    [run] = [json.loads(p.read_text(encoding="utf-8")) for p in (dirs.outputs / "manifests").glob("*.json")]
    assert run["error"] == "RuntimeError" and not (dirs.data / "prompts" / "reports" / "build.json").exists()
    assert len(google.calls) == 6 and all(lang == "hi" for _, lang in google.calls)   # zh never started
    again = FakeTranslator(english_for=[ORB[0].text])                  # without --fallback-engine: the cache
    summary = build.build_all(ctx=fixture_ctx(translator=again), only=["hard_benign"], langs=["en", "hi"])
    assert again.calls == [] and summary["files"]["hard_benign_hi"]["open_failures"] == 1


def test_fallback_engine_nllb_checks_its_packages_before_anything_is_written(dirs, monkeypatch):
    calls = []

    def missing():
        calls.append("check_nllb")
        raise ImportError("--fallback-engine nllb needs torch and transformers (fake)")

    start_run = manifest.start_run
    monkeypatch.setattr(translate, "check_nllb", missing)
    monkeypatch.setattr(manifest, "start_run",
                        lambda *a, **kw: calls.append("start_run") or start_run(*a, **kw))
    with pytest.raises(ImportError, match="needs torch and transformers"):
        build.build_all(ctx=fixture_ctx(fallback_engine="nllb"), only=["hard_benign"], langs=["en", "hi"])
    assert calls == ["check_nllb"]                                     # before the run manifest is started
    assert not dirs.data.exists() and not dirs.splits.exists() and not dirs.outputs.exists()
    # not when this build cannot make an NLLB translation, or has its own backend
    for overrides, options in (({}, {"langs": ["en"]}), ({"cache_only": True}, {}),
                               ({"skip_translation": True}, {}),
                               ({"fallback_translator": FakeTranslator()}, {}), ({}, {"only": ["harmful"]})):
        build.build_all(ctx=fixture_ctx(fallback_engine="nllb", **overrides),
                        **{"only": ["hard_benign"], "langs": ["en", "hi"], **options})
    assert calls.count("check_nllb") == 1 and calls.count("start_run") == 5


def test_cache_only_never_calls_the_engine_and_reports_the_missing_items(dirs):
    ctx = fixture_ctx(cache_only=True)
    summary = build.build_all(ctx=ctx, only=["hard_benign"], langs=["en", "hi"])
    assert ctx.translator.calls == [] and not summary["complete"]
    assert summary["incomplete"] == ["hard_benign_hi"] and not summary["blocked"]
    rep = report(dirs, "hard_benign_hi")
    assert rep["records"] == 0 and {(x["rule"], x["detail"]) for x in rep["translation_failures"]} == {
        ("error", "not cached")}


def test_retranslate_failed_asks_again_for_cached_failures_only(dirs):
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    build.build_all(ctx=fixture_ctx(translator=FakeTranslator(english_for=[ORB[0].text])), **hard)
    assert ORB[0].pid not in by_pid("hard_benign", "hi")
    fixed = FakeTranslator()
    summary = build.build_all(ctx=fixture_ctx(translator=fixed, retranslate_failed=True), **hard)
    assert fixed.calls == [(ORB[0].text, "hi")] and summary["complete"]
    assert ORB[0].pid in by_pid("hard_benign", "hi")
    assert report(dirs, "hard_benign_hi")["translation_failures"] == []


def test_nllb_fallback_for_hard_benign_is_kept_by_later_builds(dirs):
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    nllb = FakeTranslator()
    ctx = fixture_ctx(translator=FakeTranslator(english_for=[ORB[0].text]), fallback_engine="nllb",
                      fallback_translator=nllb)
    build.build_all(ctx=ctx, **hard)
    record = by_pid("hard_benign", "hi")[ORB[0].pid]
    assert record.translation.engine == "nllb-200-distilled-600M" and nllb.calls == [(ORB[0].text, "hi")]
    rep = report(dirs, "hard_benign_hi")
    assert rep["translation_failures"][0]["resolved"] == "fallback:nllb-200-distilled-600M"
    assert rep["fallback"] == {"engine": "nllb-200-distilled-600M", "resolved": [ORB[0].pid], "failures": []}
    first = (dirs.data / "prompts" / "hard_benign_hi.jsonl").read_bytes()

    later = FakeTranslator(english_for=[ORB[0].text])                 # no --fallback-engine this time
    build.build_all(ctx=fixture_ctx(translator=later), **hard)
    assert later.calls == [] and (dirs.data / "prompts" / "hard_benign_hi.jsonl").read_bytes() == first


# ---------------------------------------------------------------------------- spot checks


def spot_ctx(**overrides):
    """A build with hard_benign hi failures (ORB[0]: GlotLID misreads a good Hindi output; ORB[1]: an empty
    output) and an es PolyRefuse translation GlotLID rejects (kept in the file)."""
    lid = FakeLID(overrides={fake_translation(ORB[0].text, "hi"): ("mar_Deva", 0.9),
                             ES_HARMLESS_2: ("por_Latn", 0.8)})
    return fixture_ctx(lid=lid, translator=FakeTranslator(empty_for=[ORB[1].text]), **overrides)


@pytest.fixture
def spot(dirs):
    ctx = spot_ctx()
    build.build_all(ctx=ctx)
    return ctx


def test_spotcheck_sheets_sample_dev_and_final_machine_translations(dirs, spot):
    paths = build.write_spotcheck_sheets(ctx=spot)
    assert [p.name for p in paths] == [f"spotcheck_{lang}{kind}.csv" for lang in ("hi", "zh", "es")
                                       for kind in ("", "_failures")]
    assert all(p.parent == dirs.sheets for p in paths)
    raw = paths[0].read_bytes()
    assert raw.startswith(b"\xef\xbb\xbfitem_id,set,english,translation,verdict,corrected,note\n")
    assert b"\r" not in raw
    for lang in ("hi", "zh", "es"):
        rows = read_sheet(dirs.sheets / f"spotcheck_{lang}.csv")
        assert Counter(r["set"] for r in rows) == {"harmful": 2, "easy_benign": 2, "hard_benign": 2}
        for row in rows:
            assert row["item_id"].startswith(f"{lang}-")
            rec = by_pid(row["set"], lang)[row["item_id"].split("-", 1)[1]]
            assert rec.split in ("dev", "final") and rec.translation is not None
            assert (row["english"], row["translation"], row["verdict"], row["corrected"], row["note"]) == (
                rec.text_en, rec.text, "", "", "")
    es_engines = {r["set"]: by_pid(r["set"], "es")[r["item_id"][3:]].translation.engine
                  for r in read_sheet(dirs.sheets / "spotcheck_es.csv")}
    assert es_engines == {"harmful": "polyrefuse", "easy_benign": "polyrefuse", "hard_benign": "google"}

    failures = {r["item_id"]: r for r in read_sheet(dirs.sheets / "spotcheck_hi_failures.csv")}
    assert set(failures) == {f"hi-{ORB[0].pid}", f"hi-{ORB[1].pid}"}
    misread, empty = failures[f"hi-{ORB[0].pid}"], failures[f"hi-{ORB[1].pid}"]
    assert (misread["english"], misread["translation"]) == (ORB[0].text, fake_translation(ORB[0].text, "hi"))
    assert misread["note"] == "auto-check failed: wrong_lang top=mar_Deva 0.9"
    assert (empty["translation"], empty["note"]) == ("", "auto-check failed: empty blank")
    es_failures = {r["item_id"]: r for r in read_sheet(dirs.sheets / "spotcheck_es_failures.csv")}
    assert set(es_failures) == {f"es-{ORB[1].pid}", "es-prb-test-0002"}      # ORB[1] is empty everywhere
    kept = es_failures["es-prb-test-0002"]                                  # in the file, GlotLID rejected it
    assert kept["translation"] == ES_HARMLESS_2
    assert kept["note"] == "auto-check failed: wrong_lang top=por_Latn 0.8"
    zh_failures = read_sheet(dirs.sheets / "spotcheck_zh_failures.csv")
    assert [r["item_id"] for r in zh_failures] == [f"zh-{ORB[1].pid}"]

    before = snapshot(dirs.sheets)
    with pytest.raises(FileExistsError, match="overwrite"):
        build.write_spotcheck_sheets(ctx=spot)
    assert snapshot(dirs.sheets) == before
    build.write_spotcheck_sheets(ctx=spot, overwrite=True)
    assert snapshot(dirs.sheets) == before                              # the same sample every time

    [sample, _] = build.write_spotcheck_sheets(ctx=spot, langs=["hi"], n=100, out_dir=dirs.data / "all")
    every = read_sheet(sample)
    expected = {f"hi-{r.pid}" for s in build.TRANSLATED_SETS for r in by_pid(s, "hi").values()
                if r.split in ("dev", "final")}
    assert {r["item_id"] for r in every} == expected                   # never train, val or fit
    with pytest.raises(ValueError, match="no machine translations"):
        build.write_spotcheck_sheets(ctx=spot, langs=["en"])


def fill(path, verdicts):
    """Set verdict (and corrected, note) on the rows of a sheet whose item_id is in ``verdicts``."""
    rows = read_sheet(path)
    for row in rows:
        row.update(verdicts.get(row["item_id"], {}))
    write_sheet(path, rows)
    return rows


def test_apply_spotchecks_round_trip(dirs, spot):
    build.write_spotcheck_sheets(ctx=spot)
    frozen = snapshot(dirs.splits)
    hi_sheet = dirs.sheets / "spotcheck_hi.csv"
    first, second = read_sheet(hi_sheet)[:2]                            # two harmful items
    fill(hi_sheet, {first["item_id"]: {"verdict": "fixed", "corrected": "संशोधित नकली अनुवाद\r\nदूसरी पंक्ति",
                                       "note": "placeholder note"},
                    second["item_id"]: {"verdict": " OK "}})
    result = build.apply_spotchecks(hi_sheet, ctx=spot_ctx())
    fixed_pid, ok_pid = first["item_id"][3:], second["item_id"][3:]
    untouched = {"ok": 0, "fixed": [], "bad": [], "unchanged": 0, "unreviewed": 2}
    assert result["files"] == {"easy_benign_hi": untouched, "hard_benign_hi": untouched,
                               "harmful_hi": {"ok": 1, "fixed": [fixed_pid], "bad": [], "unchanged": 0,
                                              "unreviewed": 0}}
    assert result["rebuilt"] == ["harmful_hi"] and result["not_applied"] == [] and result["reviewed"] == 2
    assert (result["new"], result["unchanged"], result["lost"]) == (2, 0, {})
    assert result["files_rebuilt"]["harmful_hi"]["records"] == 10
    harmful = by_pid("harmful", "hi")
    fixed, ok = harmful[fixed_pid], harmful[ok_pid]
    assert fixed.text == "संशोधित नकली अनुवाद\nदूसरी पंक्ति" and fixed.text_en == first["english"]
    assert fixed.translation.checked and fixed.translation.edited
    assert fixed.translation.note == "placeholder note"
    assert ok.text == second["translation"] and ok.translation.checked and not ok.translation.edited
    assert report(dirs, "harmful_hi")["spotchecks"]["applied"] == {"ok": 1, "fixed": 1, "bad": 0}

    es_sheet = dirs.sheets / "spotcheck_es.csv"
    bad_row = next(r for r in read_sheet(es_sheet) if r["set"] == "easy_benign")
    fill(es_sheet, {bad_row["item_id"]: {"verdict": "bad"}})
    build.apply_spotchecks(es_sheet, ctx=spot_ctx())
    bad_pid = bad_row["item_id"][3:]
    assert bad_pid not in by_pid("easy_benign", "es")
    assert bad_pid not in {r.pid for r in build.load_prompts("easy_benign", "es")}   # left out on purpose
    [excluded] = [PromptRecord.model_validate(r) for r in read_jsonl(dirs.data / "prompts" / "excluded" /
                                                                       "easy_benign_es.jsonl")]
    assert excluded.pid == bad_pid and excluded.extra["excluded"] is True
    assert excluded.extra["excluded_reason"] == "spotcheck:bad" and excluded.translation.checked
    assert {"pid": bad_pid, "reason": "spotcheck:bad"} in report(dirs, "easy_benign_es")["exclusions"]

    failures_sheet = dirs.sheets / "spotcheck_hi_failures.csv"
    fill(failures_sheet, {f"hi-{ORB[0].pid}": {"verdict": "ok"},
                          f"hi-{ORB[1].pid}": {"verdict": "fixed", "corrected": "नकली सुधारा हुआ अनुरोध"}})
    build.apply_spotchecks(failures_sheet, ctx=spot_ctx())
    hard = by_pid("hard_benign", "hi")
    accepted, written = hard[ORB[0].pid], hard[ORB[1].pid]
    assert accepted.text == fake_translation(ORB[0].text, "hi") and accepted.text_en == ORB[0].text
    assert accepted.translation.checked and not accepted.translation.edited and accepted.translation.date
    assert written.text == "नकली सुधारा हुआ अनुरोध"
    assert written.translation.edited and written.translation.date is None
    rep = report(dirs, "hard_benign_hi")
    resolved = {x["pid"]: x["resolved"] for x in rep["translation_failures"]}
    assert resolved == {ORB[0].pid: "ok", ORB[1].pid: "fixed"}
    assert [x["pid"] for x in rep["lid_failures"]] == [ORB[0].pid]      # still flagged, kept by the reader

    verdicts = list(read_jsonl(dirs.data / "interim" / "translations" / "spotchecks.jsonl"))
    assert len(verdicts) == 5 and {v["sheet"] for v in verdicts} == {hi_sheet.name, es_sheet.name,
                                                                       failures_sheet.name}
    assert snapshot(dirs.splits) == frozen

    # a rebuild keeps every verdict; new sheets leave out what readers already checked
    applied = {k: v for k, v in snapshot(dirs.data / "prompts").items() if k.endswith(".jsonl")}
    build.build_all(ctx=spot_ctx())
    assert {k: v for k, v in snapshot(dirs.data / "prompts").items() if k.endswith(".jsonl")} == applied
    build.write_spotcheck_sheets(ctx=spot_ctx(), overwrite=True)
    resampled = {r["item_id"] for r in read_sheet(hi_sheet)}
    assert first["item_id"] not in resampled and second["item_id"] not in resampled
    assert read_sheet(failures_sheet) == []


def test_a_verdict_that_no_longer_fits_is_listed_not_applied(dirs, spot, caplog):
    build.write_spotcheck_sheets(ctx=spot)
    sheet = dirs.sheets / "spotcheck_hi.csv"
    row = read_sheet(sheet)[0]
    fill(sheet, {row["item_id"]: {"verdict": "ok"}})
    build.apply_spotchecks(sheet, ctx=spot_ctx())
    path = dirs.data / "interim" / "translations" / "spotchecks.jsonl"
    [verdict] = list(read_jsonl(path))
    path.write_text(json.dumps({**verdict, "seen_sha": "0" * 64}) + "\n", encoding="utf-8", newline="\n")
    caplog.set_level(logging.WARNING)
    build.build_all(ctx=spot_ctx(), only=["harmful"], langs=["en", "hi"])
    pid = row["item_id"][3:]
    assert not by_pid("harmful", "hi")[pid].translation.checked
    assert report(dirs, "harmful_hi")["spotchecks"]["ignored"] == [
        {"pid": pid, "verdict": "ok", "why": "the translation changed after review"}]
    assert "not applied" in caplog.text
    # the same row again: the reader saw the current translation, which the stored verdict did not, so it
    # is a new verdict (not "unchanged") and takes effect
    result = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert (result["new"], result["unchanged"], result["not_applied"]) == (1, 0, [])
    assert by_pid("harmful", "hi")[pid].translation.checked
    # a later cache row (as --retranslate-failed appends one) changes the translation, so the stored ok no
    # longer fits: the same sheet again is unchanged, and its verdict is listed as not applied
    english = by_pid("harmful", "hi")[pid].text_en
    translate.TranslationCache().put("google", "hi", english, fake_translation(english + " again", "hi"))
    build.build_all(ctx=spot_ctx(), only=["harmful"], langs=["en", "hi"])
    result = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert (result["new"], result["unchanged"], result["rebuilt"]) == (0, 1, [])
    assert result["not_applied"] == [{"pid": pid, "verdict": "ok", "set": "harmful", "lang": "hi",
                                      "why": "the translation changed after review"}]


def test_a_half_written_verdict_line_is_ignored_then_closed_off(dirs, caplog):
    path = dirs.data / "interim" / "translations" / "spotchecks.jsonl"
    path.parent.mkdir(parents=True)
    row = {"lang": "hi", "pid": "orb-0004", "set": "hard_benign", "verdict": "ok", "corrected": None,
           "note": None, "source_sha": "0" * 64, "seen_sha": "1" * 64, "sheet": "s.csv",
           "applied": "2026-11-03"}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "pid": "orb-0005"})[:40], encoding="utf-8",
                    newline="\n")
    caplog.set_level(logging.WARNING)
    assert list(build._read_verdicts()) == [("hi", "orb-0004")] and "spotchecks.jsonl:2" in caplog.text
    build._append_verdicts([{**row, "pid": "orb-0006"}])                # the cut line is closed off first
    assert list(build._read_verdicts()) == [("hi", "orb-0004"), ("hi", "orb-0006")]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    path.write_text(json.dumps({**row, "verdict": "maybe"}) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="spotchecks.jsonl:1: not a verdict row"):
        build._read_verdicts()


def test_apply_spotchecks_rejects_a_bad_sheet_before_writing_anything(dirs, spot):
    build.write_spotcheck_sheets(ctx=spot)
    sheet = dirs.sheets / "spotcheck_hi.csv"
    rows = read_sheet(sheet)                                     # rows 2-7: harmful x2, easy x2, hard x2
    rows[0]["verdict"] = "maybe"
    rows[1]["verdict"] = "fixed"
    rows[2].update(verdict="ok", corrected="कोई सुधार")
    rows[3].update(verdict="bad", set="harmful")
    rows[4].update(verdict="ok", translation="बदला हुआ नकली पाठ")
    rows[5].update(verdict="ok", english=rows[5]["english"] + " in an older wording")
    rows += [{**rows[5], "item_id": "hi-prh-test-0999", "set": "harmful", "verdict": "ok"},
             {**rows[5], "item_id": "not an item id", "verdict": "ok"},
             {**rows[5], "verdict": "ok"}]
    write_sheet(sheet, rows)
    before = snapshot(dirs.data)
    with pytest.raises(SheetError) as e:
        build.apply_spotchecks(sheet, ctx=spot_ctx())
    problems = e.value.problems
    expected = [(2, "verdict must be ok, fixed or bad"), (3, "fixed needs the corrected translation"),
                (4, "corrected is filled but the verdict is ok"),
                (5, "the item is in set easy_benign, not harmful"),
                (6, "the translation cell matches neither the current build nor what an earlier verdict on "
                    "the item saw: the sheet is older than the build or a spreadsheet changed the cell; "
                    "restore the cell or write fresh sheets"),
                (7, "english differs from the current build (a stale or re-saved sheet)"),
                (8, "no such item in the current harmful_hi"),
                (9, "item_id is not <lang>-<pid>"), (10, "the item is also on row 7")]
    assert len(problems) == len(expected)
    for number, what in expected:
        assert any(p.startswith(f"row {number}") and what in p for p in problems), (number, what)
    message = str(e.value)
    assert "9 problem(s), nothing applied" in message
    cells = ("कोई सुधार", "बदला हुआ नकली पाठ", rows[0]["english"], rows[0]["translation"], rows[5]["english"],
             "not an item id")
    assert not any(cell in message for cell in cells)
    assert snapshot(dirs.data) == before

    sheet.write_text("item_id,set,english\n", encoding="utf-8-sig")
    with pytest.raises(SheetError, match="header must be exactly"):
        build.apply_spotchecks(sheet, ctx=spot_ctx())


def test_apply_spotchecks_dry_run_checks_and_writes_nothing(dirs, spot):
    build.write_spotcheck_sheets(ctx=spot)
    sheet = dirs.sheets / "spotcheck_zh.csv"
    row = read_sheet(sheet)[0]
    fill(sheet, {row["item_id"]: {"verdict": "ok"}})
    before = snapshot(dirs.data)
    result = build.apply_spotchecks(sheet, ctx=spot_ctx(), dry_run=True)
    assert result["dry_run"] and result["rebuilt"] == []
    assert (result["reviewed"], result["unreviewed"]) == (1, 5)
    assert snapshot(dirs.data) == before


def test_an_item_is_on_one_sheet_only_and_the_sample_stays_blind_and_unbiased(dirs, spot):
    """es prb-test-0002 (GlotLID rejects PolyRefuse's translation) ranks third in its set's sample: with n=3
    it is drawn, so it is on the sample sheet with no note and off the failures sheet; the draw is the plain
    hash rank over every unchecked dev/final machine translation, GlotLID's verdict aside."""
    sample_path, failures_path = build.write_spotcheck_sheets(ctx=spot, langs=["es"], n=3)
    sample = {r["item_id"]: r for r in read_sheet(sample_path)}
    failures = {r["item_id"]: r for r in read_sheet(failures_path)}
    assert sample["es-prb-test-0002"]["note"] == "" and "prb-test-0002" in by_pid("easy_benign", "es")
    assert set(failures) == {f"es-{ORB[1].pid}"} and not set(sample) & set(failures)
    for set_name in build.TRANSLATED_SETS:
        tag = f"spotcheck:es:{set_name}"
        candidates = [r.pid for r in by_pid(set_name, "es").values()
                      if r.translation and not r.translation.checked and r.split in ("dev", "final")]
        drawn = sorted(candidates, key=lambda pid, tag=tag: sha256_text(f"{SEED}:{tag}:{pid}"))[:3]
        assert sorted(k[3:] for k, r in sample.items() if r["set"] == set_name) == sorted(drawn), set_name
    assert all(r["note"] == "" for r in sample.values())
    _, failures_path = build.write_spotcheck_sheets(ctx=spot, langs=["es"], n=2, overwrite=True)
    assert "es-prb-test-0002" in {r["item_id"] for r in read_sheet(failures_path)}    # not drawn: listed


def test_reapplying_a_sheet_changes_nothing_and_revised_verdicts_on_fixed_items_take_effect(dirs, spot,
                                                                                        monkeypatch):
    """After a fixed verdict the build holds the correction, but the sheet still shows the engine output: the
    row passes through the stored verdict's seen_sha. An unchanged row is not stored again and rebuilds
    nothing; a revised one (fixed -> bad, a new correction, fixed -> ok) is judged against the engine output
    and takes effect. ids[4]'s correction equals the engine output by dup_key (a doubled space), so only
    the exact match on the stored seen_sha says which of the two texts the reader saw."""
    build.write_spotcheck_sheets(ctx=spot)
    sheet = dirs.sheets / "spotcheck_hi.csv"
    rows = read_sheet(sheet)                                  # harmful x2, easy_benign x2, hard_benign x2
    ids = [r["item_id"] for r in rows]
    engine = {r["item_id"]: r["translation"] for r in rows}
    spaced = engine[ids[4]].replace(" ", "  ", 1)             # a correction equal to the output by dup_key
    fill(sheet, {ids[0]: {"verdict": "fixed", "corrected": "पहला नकली सुधार"}, ids[1]: {"verdict": "ok"},
                 ids[2]: {"verdict": "fixed", "corrected": "दूसरा नकली सुधार", "note": "a note"},
                 ids[3]: {"verdict": "bad"}, ids[4]: {"verdict": "fixed", "corrected": spaced}})
    first = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert (first["new"], first["unchanged"], first["not_applied"]) == (5, 0, [])
    stored = dirs.data / "interim" / "translations" / "spotchecks.jsonl"
    before, prompts = stored.read_bytes(), snapshot(dirs.data / "prompts")
    rebuilt = []
    real = build._build_file
    monkeypatch.setattr(build, "_build_file", lambda run, src, s, lang: rebuilt.append(f"{s}_{lang}")
                        or real(run, src, s, lang))

    monkeypatch.setattr("lift.lang.LangID", no_fasttext)          # nothing to rebuild: GlotLID never loads
    again = build.apply_spotchecks(sheet, ctx=fixture_ctx(lid=None))      # the very same sheet
    assert (again["new"], again["unchanged"], again["reviewed"], again["rebuilt"]) == (0, 5, 5, [])
    assert again["files"]["harmful_hi"] == {"ok": 0, "fixed": [], "bad": [], "unchanged": 2, "unreviewed": 0}
    assert again["not_applied"] == [] and rebuilt == []
    assert stored.read_bytes() == before and snapshot(dirs.data / "prompts") == prompts

    fill(sheet, {ids[0]: {"verdict": "bad", "corrected": ""}, ids[2]: {"corrected": "नया नकली सुधार"},
                 ids[4]: {"verdict": "ok", "corrected": ""}})
    result = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert (result["new"], result["unchanged"], result["not_applied"]) == (3, 2, [])
    assert rebuilt == result["rebuilt"] == ["harmful_hi", "easy_benign_hi", "hard_benign_hi"]
    latest = {v["pid"]: v for v in read_jsonl(stored)}
    for item_id in (ids[0], ids[2], ids[4]):                   # the engine output the reader saw
        assert latest[item_id[3:]]["seen_sha"] == sha256_text(engine[item_id])
    harmful, easy, hard = (by_pid(s, "hi") for s in ("harmful", "easy_benign", "hard_benign"))
    assert ids[0][3:] not in harmful                         # fixed -> bad: excluded, as the engine wrote it
    [excluded] = [PromptRecord.model_validate(r) for r in
                  read_jsonl(dirs.data / "prompts" / "excluded" / "harmful_hi.jsonl")]
    assert (excluded.pid, excluded.text, excluded.translation.checked) == (ids[0][3:], engine[ids[0]], True)
    revised = easy[ids[2][3:]]                                  # a new correction
    assert (revised.text, revised.translation.edited, revised.translation.note) == ("नया नकली सुधार", True,
                                                                                     "a note")
    accepted = hard[ids[4][3:]]                                 # fixed -> ok: the engine output, checked
    assert (accepted.text, accepted.translation.checked, accepted.translation.edited) == (engine[ids[4]],
                                                                                           True, False)
    assert report(dirs, "harmful_hi")["spotchecks"]["applied"] == {"ok": 1, "fixed": 0, "bad": 1}

    rows = read_sheet(sheet)
    rows[1]["translation"] = "कोई और नकली पाठ"                     # neither the build nor what a verdict saw
    write_sheet(sheet, rows)
    with pytest.raises(SheetError, match="row 3 .*matches neither the current build nor what an earlier "
                                         "verdict on the item saw.*restore the cell or write fresh sheets"):
        build.apply_spotchecks(sheet, ctx=spot_ctx())
    # ids[2]'s correction pasted into its translation cell: it matches the build, but a verdict is given on
    # the engine output (a rebuild compares an ok or bad with it), so none is taken on the correction
    before = snapshot(dirs.data)
    for verdict, corrected in (("ok", ""), ("bad", ""), ("fixed", "नया नकली सुधार")):
        rows = read_sheet(sheet)
        rows[1]["translation"] = engine[ids[1]]
        rows[2].update(translation="नया नकली सुधार", verdict=verdict, corrected=corrected)
        write_sheet(sheet, rows)
        with pytest.raises(SheetError) as e:
            build.apply_spotchecks(sheet, ctx=spot_ctx())
        assert e.value.problems == [f"row 4 ({ids[2]}): the translation cell shows the reader's correction "
                                    "the build holds, not the engine output a verdict is given on: restore "
                                    "the cell (a correction goes in corrected)"], verdict
    assert snapshot(dirs.data) == before


def test_a_fixed_verdict_whose_english_changed_is_stored_again(dirs):
    """The English changes in a way dup_key does not see (a doubled inner space): the stored fixed verdict no
    longer applies (another source_sha), so the same sheet applied again is a new verdict, not unchanged, and
    the correction takes effect again."""
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    build.build_all(ctx=fixture_ctx(), **hard)
    rec = build.load_prompts("hard_benign", "hi")[0]
    sheet = dirs.sheets / "sheet.csv"
    sheet.parent.mkdir(parents=True)
    write_sheet(sheet, [{"item_id": f"hi-{rec.pid}", "set": "hard_benign", "english": rec.text_en,
                         "translation": rec.text, "verdict": "fixed", "corrected": "नकली सुधार पाठ",
                         "note": ""}])
    assert build.apply_spotchecks(sheet, ctx=fixture_ctx())["new"] == 1
    rows = orbench_rows()
    row = rows[rec.extra["hf_row"]]
    row["prompt"] = row["prompt"].replace(" ", "  ", 1)
    hub = fixture_hub({"orbench": rows})
    build.build_all(ctx=fixture_ctx(hub=hub), **hard)
    stale = by_pid("hard_benign", "hi")[rec.pid]
    assert stale.text_en != rec.text_en and not stale.translation.checked
    assert report(dirs, "hard_benign_hi")["spotchecks"]["ignored"] == [
        {"pid": rec.pid, "verdict": "fixed", "why": "the English source changed after review"}]
    result = build.apply_spotchecks(sheet, ctx=fixture_ctx(hub=hub))
    assert (result["new"], result["unchanged"], result["not_applied"]) == (1, 0, [])
    assert by_pid("hard_benign", "hi")[rec.pid].text == "नकली सुधार पाठ"


def test_a_verdict_on_a_cached_nllb_record_goes_to_that_record_once(dirs):
    """hard_benign ORB[0]: Google gives English (wrong_lang), NLLB's cached translation is in the file. A
    verdict a reader gave on the NLLB text is applied to the NLLB record, once, and never listed as not
    applied; a correction keeps engine nllb. If the NLLB text later fails the checks, the verdict is listed
    once, with the reason."""
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    build.build_all(ctx=fixture_ctx(translator=FakeTranslator(english_for=[ORB[0].text]),
                                    fallback_engine="nllb", fallback_translator=FakeTranslator()), **hard)
    nllb_text, pid = fake_translation(ORB[0].text, "hi"), ORB[0].pid
    assert by_pid("hard_benign", "hi")[pid].translation.engine == "nllb-200-distilled-600M"
    sheet = dirs.sheets / "nllb.csv"
    sheet.parent.mkdir(parents=True)
    row = {"item_id": f"hi-{pid}", "set": "hard_benign", "english": ORB[0].text, "translation": nllb_text,
           "verdict": "ok", "corrected": "", "note": ""}
    ctx = fixture_ctx(translator=FakeTranslator(english_for=[ORB[0].text]))
    write_sheet(sheet, [row])
    result = build.apply_spotchecks(sheet, ctx=ctx)
    record = by_pid("hard_benign", "hi")[pid]
    assert result["not_applied"] == [] and record.translation.checked and record.text == nllb_text
    assert record.translation.engine == "nllb-200-distilled-600M" and not record.translation.edited
    rep = report(dirs, "hard_benign_hi")
    assert rep["spotchecks"]["applied"] == {"ok": 1, "fixed": 0, "bad": 0}
    assert rep["spotchecks"]["ignored"] == []
    assert rep["translation_failures"][0]["resolved"] == "fallback:nllb-200-distilled-600M"

    write_sheet(sheet, [{**row, "verdict": "fixed", "corrected": "नकली सुधरा एनएलएलबी पाठ"}])
    result = build.apply_spotchecks(sheet, ctx=ctx)
    record = by_pid("hard_benign", "hi")[pid]
    assert result["not_applied"] == [] and record.text == "नकली सुधरा एनएलएलबी पाठ"
    assert record.translation.engine == "nllb-200-distilled-600M" and record.translation.edited
    rep = report(dirs, "hard_benign_hi")
    assert rep["spotchecks"]["applied"] == {"ok": 0, "fixed": 1, "bad": 0}
    assert rep["spotchecks"]["ignored"] == []

    write_sheet(sheet, [row])                                 # back to ok, then GlotLID rejects the NLLB text
    build.apply_spotchecks(sheet, ctx=ctx)
    rejecting = FakeLID(overrides={nllb_text: ("mar_Deva", 0.9)})
    google = FakeTranslator(english_for=[ORB[0].text])
    build.build_all(ctx=fixture_ctx(lid=rejecting, translator=google), **hard)
    assert pid not in by_pid("hard_benign", "hi")
    assert report(dirs, "hard_benign_hi")["spotchecks"]["ignored"] == [
        {"pid": pid, "verdict": "ok",
         "why": "the NLLB translation it was given on fails the checks now (wrong_lang top=mar_Deva 0.9)"}]
    stored = list(read_jsonl(dirs.data / "interim" / "translations" / "spotchecks.jsonl"))[-1]
    build._append_verdicts([{**stored, "seen_sha": "0" * 64}])     # a verdict on neither Google's nor NLLB's
    build.build_all(ctx=fixture_ctx(translator=google), **hard)
    assert not by_pid("hard_benign", "hi")[pid].translation.checked
    assert report(dirs, "hard_benign_hi")["spotchecks"]["ignored"] == [            # listed once
        {"pid": pid, "verdict": "ok", "why": "the translation changed after review"}]


def test_a_verdict_on_an_output_both_engines_gave_resolves_the_google_failure(dirs):
    """Google and NLLB give ORB[0] the same output, which fails the checks for both: a reader's ok on it (the
    failures sheet shows Google's) was given on Google's output as well, so it resolves the Google failure
    rather than going to an NLLB record that never passes."""
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    english = {"english_for": [ORB[0].text]}
    build.build_all(ctx=fixture_ctx(translator=FakeTranslator(**english), fallback_engine="nllb",
                                    fallback_translator=FakeTranslator(**english)), **hard)
    assert [x["pid"] for x in report(dirs, "hard_benign_hi")["fallback"]["failures"]] == [ORB[0].pid]
    sheet = dirs.sheets / "both.csv"
    sheet.parent.mkdir(parents=True)
    write_sheet(sheet, [{"item_id": f"hi-{ORB[0].pid}", "set": "hard_benign", "english": ORB[0].text,
                         "translation": fake_translation(ORB[0].text, "en"), "verdict": "ok", "corrected": "",
                         "note": ""}])
    result = build.apply_spotchecks(sheet, ctx=fixture_ctx(translator=FakeTranslator(**english)))
    record = by_pid("hard_benign", "hi")[ORB[0].pid]
    assert result["not_applied"] == [] and record.translation.checked
    assert record.translation.engine == "google"
    assert [x["resolved"] for x in report(dirs, "hard_benign_hi")["translation_failures"]] == ["ok"]


def test_the_apply_rebuild_keeps_failures_the_cache_cannot_hold(dirs, monkeypatch):
    """Errors and empty outputs are never cached, so the cache-only rebuild of apply_spotchecks can only call
    them "error: not cached": the report keeps the build's rule and detail instead."""
    monkeypatch.setattr(translate, "translate_records",                 # no backoff sleeps in a test
                        functools.partial(translate.translate_records, sleep=lambda _: None))
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    ctx = fixture_ctx(translator=FakeTranslator(always_raise=[ORB[2].text], empty_for=[ORB[1].text]))
    build.build_all(ctx=ctx, **hard)
    failures = {(x["pid"], x["rule"], x["detail"])
                for x in report(dirs, "hard_benign_hi")["translation_failures"]}
    assert failures == {(ORB[1].pid, "empty", "blank"), (ORB[2].pid, "error", "ConnectionError x5")}
    rec = build.load_prompts("hard_benign", "hi", allow_short=True)[0]
    sheet = dirs.sheets / "spotcheck_hi.csv"
    sheet.parent.mkdir(parents=True)
    write_sheet(sheet, [{"item_id": f"hi-{rec.pid}", "set": "hard_benign", "english": rec.text_en,
                         "translation": rec.text, "verdict": "ok", "corrected": "", "note": ""}])
    result = build.apply_spotchecks(sheet, ctx=fixture_ctx())
    assert result["rebuilt"] == ["hard_benign_hi"]
    assert result["files_rebuilt"]["hard_benign_hi"]["errors"] == 1     # the ConnectionError, not the empty
    rows_after = report(dirs, "hard_benign_hi")["translation_failures"]
    assert {(x["pid"], x["rule"], x["detail"]) for x in rows_after} == failures
    # the cache has an output for ORB[2] now (English: wrong_lang): the rebuild reports what it found; only an
    # item it can only call "error: not cached" keeps the earlier rule
    translate.TranslationCache().put("google", "hi", ORB[2].text, fake_translation(ORB[2].text, "en"))
    write_sheet(sheet, [{"item_id": f"hi-{rec.pid}", "set": "hard_benign", "english": rec.text_en,
                         "translation": rec.text, "verdict": "bad", "corrected": "", "note": ""}])
    assert build.apply_spotchecks(sheet, ctx=fixture_ctx())["new"] == 1

    def rules():
        failed = report(dirs, "hard_benign_hi")["translation_failures"]
        return {x["pid"]: (x["rule"], x["detail"]) for x in failed}

    assert rules() == {ORB[1].pid: ("empty", "blank"), ORB[2].pid: ("wrong_lang", "top=eng_Latn 0.95")}
    build.build_all(ctx=fixture_ctx(cache_only=True), **hard)          # a cache-only build says what it saw
    assert rules() == {ORB[1].pid: ("error", "not cached"), ORB[2].pid: ("wrong_lang", "top=eng_Latn 0.95")}


def test_apply_refuses_to_run_without_the_translation_cache(dirs, spot, tmp_path):
    """A data dir copied without interim/translations: the cache-only rebuild would drop every translated
    record of the files a sheet has verdicts on, and a failure would lose the engine output its row shows,
    so nothing is applied, on a dry run too, and the cache is checked before any cell is compared with the
    build (a failure's translation cell would match nothing). Fresh failures sheets are refused as well.
    With one cache row gone, a failure's, that item alone is named."""
    build.write_spotcheck_sheets(ctx=spot)
    sheet, failures_sheet = dirs.sheets / "spotcheck_hi.csv", dirs.sheets / "spotcheck_hi_failures.csv"
    fill(sheet, {read_sheet(sheet)[0]["item_id"]: {"verdict": "ok"}})
    fill(failures_sheet, {f"hi-{ORB[0].pid}": {"verdict": "ok"}})      # wrong_lang: its output was cached
    google = dirs.data / "interim" / "translations" / "google"
    shutil.copytree(google, tmp_path / "google")
    shutil.rmtree(google)
    before = snapshot(dirs.data)
    hard = sorted(set(by_pid("hard_benign", "hi")) | {ORB[0].pid})                     # + its failure
    for path, problem in ((sheet, "harmful_hi: 10 items (e.g. prh-test-0000, prh-test-0001, prh-test-0002, "
                                  "prh-test-0003, prh-test-0004)"),
                          (failures_sheet, f"hard_benign_hi: 5 items (e.g. {', '.join(hard)})")):
        for dry_run in (True, False):
            with pytest.raises(SheetError) as e:
                build.apply_spotchecks(path, ctx=spot_ctx(), dry_run=dry_run)
            assert e.value.problems == [problem]
            message = str(e.value)
            assert "nothing applied" in message and "interim" in message and "matches neither" not in message
            assert "from the machine that built the files, or rerun `lift data build`" in message
            assert snapshot(dirs.data) == before and not (dirs.data / "interim" / "translations" /
                                                           "spotchecks.jsonl").exists()
    with pytest.raises(FileNotFoundError, match=rf"hard_benign_hi: the translation cache \(.*\) lacks the "
                                                rf"engine output of 1 translation failures the failures "
                                                rf"sheet shows \(e.g. {ORB[0].pid}\): copy"):
        build.write_spotcheck_sheets(ctx=spot, out_dir=tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()

    shutil.copytree(tmp_path / "google", google)
    cache_file = google / "hi.jsonl"
    write_jsonl(cache_file, [row for row in read_jsonl(cache_file) if row["source"] != ORB[0].text])
    with pytest.raises(SheetError) as e:
        build.apply_spotchecks(failures_sheet, ctx=spot_ctx(), dry_run=True)
    assert e.value.problems == [f"hard_benign_hi: 1 items (e.g. {ORB[0].pid})"]
    assert build.apply_spotchecks(sheet, ctx=spot_ctx(), dry_run=True)["new"] == 1     # harmful_hi: all there


def test_apply_needs_the_cache_rows_of_excluded_and_nllb_records(dirs):
    """The cache check reads the excluded file and every cached engine: without its row, a record a reader
    marked bad would come back as an open failure, and the NLLB record of a Google failure would be
    dropped."""
    hard = {"only": ["hard_benign"], "langs": ["en", "hi"]}
    google = FakeTranslator(english_for=[ORB[0].text])
    build.build_all(ctx=fixture_ctx(translator=google, fallback_engine="nllb",
                                    fallback_translator=FakeTranslator()), **hard)
    bad, other = [r for r in build.load_prompts("hard_benign", "hi") if r.pid != ORB[0].pid][:2]
    sheet = dirs.sheets / "sheet.csv"
    sheet.parent.mkdir(parents=True)

    def row(rec, verdict):
        return {"item_id": f"hi-{rec.pid}", "set": "hard_benign", "english": rec.text_en,
                "translation": rec.text, "verdict": verdict, "corrected": "", "note": ""}

    write_sheet(sheet, [row(bad, "bad")])
    build.apply_spotchecks(sheet, ctx=fixture_ctx(translator=google))
    assert {"pid": bad.pid, "reason": "spotcheck:bad"} in report(dirs, "hard_benign_hi")["exclusions"]
    cache_file = dirs.data / "interim" / "translations" / "google" / "hi.jsonl"
    saved = cache_file.read_bytes()
    write_jsonl(cache_file, [r for r in read_jsonl(cache_file) if r["source"] != bad.text_en])
    write_sheet(sheet, [row(other, "ok")])
    before = snapshot(dirs.data)
    with pytest.raises(SheetError) as e:
        build.apply_spotchecks(sheet, ctx=fixture_ctx(translator=google))
    assert e.value.problems == [f"hard_benign_hi: 1 items (e.g. {bad.pid})"]
    assert snapshot(dirs.data) == before

    cache_file.write_bytes(saved)
    shutil.rmtree(dirs.data / "interim" / "translations" / "nllb-200-distilled-600M")
    before = snapshot(dirs.data)
    with pytest.raises(SheetError) as e:
        build.apply_spotchecks(sheet, ctx=fixture_ctx(translator=google))
    assert e.value.problems == [f"hard_benign_hi: 1 items (e.g. {ORB[0].pid})"]
    assert snapshot(dirs.data) == before


def no_fasttext(cfg=None):
    raise ModuleNotFoundError("No module named 'fasttext'", name="fasttext")


def test_apply_loads_what_the_rebuild_needs_before_storing_a_verdict(dirs, spot, monkeypatch):
    """GlotLID cannot load on this machine: the apply stops before a verdict is stored, so the same sheet
    applies in full once it can (a stored verdict would count as unchanged and never be rebuilt)."""
    build.write_spotcheck_sheets(ctx=spot)
    sheet = dirs.sheets / "spotcheck_hi.csv"
    hard = [r["item_id"] for r in read_sheet(sheet) if r["set"] == "hard_benign"]
    fill(sheet, {hard[0]: {"verdict": "ok"}})
    monkeypatch.setattr("lift.lang.LangID", no_fasttext)
    before = snapshot(dirs.data)
    with pytest.raises(ImportError, match="fasttext"):
        build.apply_spotchecks(sheet, ctx=fixture_ctx(lid=None))
    assert snapshot(dirs.data) == before
    assert not (dirs.data / "interim" / "translations" / "spotchecks.jsonl").exists()
    result = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert (result["new"], result["unchanged"], result["rebuilt"]) == (1, 0, ["hard_benign_hi"])


def test_a_reader_written_record_needs_no_cache_row_and_lost_records_are_reported(dirs, spot, monkeypatch,
                                                                                  caplog):
    """ORB[1] had an empty output (never cached): a reader's correction makes its record, with no date and no
    cache row, and later sheets on its file still apply; an ok on that row is refused (there is no engine
    output to accept, and once stored it would replace the correction, which the rebuild would then drop). A
    record the rebuild loses is listed, an excluded one is not."""
    build.write_spotcheck_sheets(ctx=spot)
    failures_sheet = dirs.sheets / "spotcheck_hi_failures.csv"
    fill(failures_sheet, {f"hi-{ORB[1].pid}": {"verdict": "fixed", "corrected": "नकली सुधारा हुआ अनुरोध"}})
    build.apply_spotchecks(failures_sheet, ctx=spot_ctx())
    written = by_pid("hard_benign", "hi")[ORB[1].pid]
    assert written.translation.engine == "google" and written.translation.date is None
    fill(failures_sheet, {f"hi-{ORB[1].pid}": {"verdict": "ok", "corrected": ""}})
    before = snapshot(dirs.data)
    with pytest.raises(SheetError) as e:
        build.apply_spotchecks(failures_sheet, ctx=spot_ctx())
    assert e.value.problems == [f"row 3 (hi-{ORB[1].pid}): there is no engine output to accept (use fixed or "
                                "bad)"]
    assert snapshot(dirs.data) == before
    sheet = dirs.sheets / "spotcheck_hi.csv"
    rows = read_sheet(sheet)
    hard = [r["item_id"] for r in rows if r["set"] == "hard_benign"]
    fill(sheet, {hard[0]: {"verdict": "ok"}})
    assert build.apply_spotchecks(sheet, ctx=spot_ctx())["rebuilt"] == ["hard_benign_hi"]

    victim = hard[1][3:]
    real = build._translate

    def losing(run, records, lang, engine, **kw):
        done, failures = real(run, records, lang, engine, **kw)
        return [r for r in done if r.pid != victim], failures

    monkeypatch.setattr(build, "_translate", losing)
    harmful = [r["item_id"] for r in rows if r["set"] == "harmful"]
    fill(sheet, {hard[0]: {"verdict": "ok", "note": "seen again"}, harmful[0]: {"verdict": "bad"}})
    caplog.set_level(logging.ERROR)
    result = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert result["lost"] == {"hard_benign_hi": [victim]}
    assert harmful[0][3:] not in by_pid("harmful", "hi")      # excluded by the reader: not lost
    assert (f"hard_benign_hi: 1 records were in the file before the rebuild and are neither in it nor "
            f"excluded now: ['{victim}']; the verdicts are stored. The report (translation_failures) and the "
            "verdicts not applied say why") in caplog.text


# ---------------------------------------------------------------------------- spreadsheet exports


CORRECTIONS = {"hi": "नकली सुधार\nदूसरी पंक्ति", "es": "Corrección ficticia\nsegunda línea"}


def filled_sheet(dirs, spot, lang="hi"):
    """``lang``'s sample sheet, freshly written, as a reader fills it: an ok row whose note holds a comma and
    a semicolon, and a fixed row with a two-line correction."""
    build.write_spotcheck_sheets(ctx=spot, overwrite=True)
    sheet = dirs.sheets / f"spotcheck_{lang}.csv"
    rows = read_sheet(sheet)
    rows[0].update(verdict="ok", note="fine, thanks; checked")
    rows[1].update(verdict="fixed", corrected=CORRECTIONS[lang])
    return sheet, rows


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_apply_reads_semicolon_and_tab_separated_sheets(dirs, spot, delimiter, caplog):
    sheet, rows = filled_sheet(dirs, spot)
    write_sheet(sheet, rows, delimiter=delimiter)
    caplog.set_level(logging.INFO)
    result = build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert result["files"]["harmful_hi"] == {"ok": 1, "fixed": [rows[1]["item_id"][3:]], "bad": [],
                                             "unchanged": 0, "unreviewed": 0}
    ok = by_pid("harmful", "hi")[rows[0]["item_id"][3:]]
    assert ok.translation.checked and ok.translation.note == "fine, thanks; checked"
    assert by_pid("harmful", "hi")[rows[1]["item_id"][3:]].text == CORRECTIONS["hi"]
    named = {";": "semicolons", "\t": "tabs"}[delimiter]
    assert f"the header parses with {named}, not commas" in caplog.text


def test_the_delimiter_is_chosen_on_the_header_alone(dirs):
    """A semicolon sheet whose second row has the cell 'He said,"': read with commas, that quote opens a field
    that swallows the rest of the file, past the csv module's field limit. The header alone picks the
    delimiter, so the sheet reads in full."""
    path = dirs.sheets / "said.csv"
    path.parent.mkdir(parents=True)
    blank = dict.fromkeys(build.SHEET_COLUMNS, "")
    rows = [{**blank, "item_id": "hi-prh-test-0000", "set": "harmful", "english": 'He said,"'}]
    rows += [{**blank, "item_id": f"hi-prh-train-{i:04d}", "set": "harmful", "english": "x" * 1000}
             for i in range(150)]
    write_sheet(path, rows, delimiter=";")
    with open(path, encoding="utf-8-sig", newline="") as f, pytest.raises(csv.Error, match="field limit"):
        list(csv.reader(f))                                            # what reading it with commas does
    read, problems = build._read_sheet(path)
    assert problems == [] and len(read) == 151
    assert read[0] == (2, ["hi-prh-test-0000", "harmful", 'He said,"', "", "", "", ""])


def test_apply_refuses_other_encodings_and_says_how_to_save(dirs, spot):
    sheet, rows = filled_sheet(dirs, spot, "es")
    before = snapshot(dirs.data)
    save_as = ('CSV UTF-8 (Excel: File > Save As > "CSV UTF-8 (Comma delimited)"; Google Sheets: File > '
               "Download")
    write_sheet(sheet, rows, encoding="cp1252", errors="replace")      # Excel's "CSV (Comma delimited)"
    with pytest.raises(SheetError, match=r"spotcheck_es.csv: not UTF-8 \(byte \d+ does not decode\)") as e:
        build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert save_as in str(e.value)
    write_sheet(sheet, rows, delimiter="\t", encoding="utf-16")          # Excel's "Unicode Text"
    with pytest.raises(SheetError, match=r"not UTF-8 \(byte 0 does not decode\)"):
        build.apply_spotchecks(sheet, ctx=spot_ctx())
    sheet.write_text(",".join(build.SHEET_COLUMNS) + "\n" + "x" * 200_000 + ",harmful\n", encoding="utf-8")
    with pytest.raises(SheetError, match="the csv module cannot read it"):         # over its field limit
        build.apply_spotchecks(sheet, ctx=spot_ctx())
    sheet.write_text("item_id|set|english|translation|verdict|corrected|note\n", encoding="utf-8")
    with pytest.raises(SheetError, match="header must be exactly .* \\(commas, semicolons or tabs"):
        build.apply_spotchecks(sheet, ctx=spot_ctx())

    # Hindi in cp1252: a "?" for every letter, so the file is pure ASCII and decodes as UTF-8. One problem for
    # the sheet: the six translation cells (two reviewed, four not) and the fixed row's correction
    sheet, rows = filled_sheet(dirs, spot, "hi")
    write_sheet(sheet, rows, encoding="cp1252", errors="replace")
    assert sheet.read_bytes().isascii()
    with pytest.raises(SheetError) as e:
        build.apply_spotchecks(sheet, ctx=spot_ctx())
    assert e.value.problems == ["7 cell(s) hold '?' in place of letters (rows 2, 3, 4, 5, 6, 7): the sheet "
                                f"was saved in a legacy encoding; {build._SAVE_AS}"]

    # failures sheets whose reviewed rows show no engine output (ORB[1] is empty): the build's cell is "" as
    # well, so only the corrected text, a "?" and no letter, or an unreviewed row's translation gives it away
    def failures_sheet(lang, change):
        _, path = build.write_spotcheck_sheets(ctx=spot, langs=[lang], overwrite=True)
        rows = read_sheet(path)
        for row in rows:
            change(row, row["item_id"] == f"{lang}-{ORB[1].pid}")
        return path, rows

    for lang, change, problem in (
            ("hi", lambda row, empty: empty and row.update(verdict="fixed", corrected="नकली सुधारा हुआ अनुरोध"),
             "2 cell(s) hold '?' in place of letters (rows 2, 3)"),          # ORB[0]'s output, the correction
            ("zh", lambda row, empty: empty and row.update(verdict="fixed", corrected="假的更正文本"),
             "1 cell(s) hold '?' in place of letters (row 2)"),
            ("hi", lambda row, empty: empty and row.update(verdict="bad", note="नकली टिप्पणी"),
             "1 cell(s) hold '?' in place of letters (row 2)")):            # unreviewed ORB[0] alone
        path, rows = failures_sheet(lang, change)
        write_sheet(path, rows, encoding="cp1252", errors="replace")
        with pytest.raises(SheetError) as e:
            build.apply_spotchecks(path, ctx=spot_ctx())
        assert [p.split(":")[0] for p in e.value.problems] == [problem], lang

    # a correction keeps the Latin letters of a name or an acronym, and the sheet shows no other engine output
    # (ORB[0]'s row left off): its "??" gives it away, for a language not written in Latin letters alone
    for lang, corrected in (("hi", "AI के बारे में नकली अनुरोध"), ("zh", "关于AI的假请求")):
        def fix(row, empty, corrected=corrected):
            if empty:
                row.update(verdict="fixed", corrected=corrected)

        path, rows = failures_sheet(lang, fix)
        rows = [row for row in rows if row["item_id"] != f"{lang}-{ORB[0].pid}"]
        write_sheet(path, rows, encoding="cp1252", errors="replace")
        with pytest.raises(SheetError) as e:
            build.apply_spotchecks(path, ctx=spot_ctx())
        problem = "1 cell(s) hold '?' in place of letters (row 2)"
        assert [p.split(":")[0] for p in e.value.problems] == [problem], lang
    assert snapshot(dirs.data) == before

    # a "?" a reader typed, and a romanised correction ending in one, are no legacy save
    def typed(row, empty):
        if empty:
            row.update(verdict="fixed", corrected="kya yah nakli anurodh hai?")
        else:
            row["translation"] = "?"                                    # an unreviewed row, marked

    path, rows = failures_sheet("hi", typed)
    write_sheet(path, rows)
    assert build.apply_spotchecks(path, ctx=spot_ctx())["new"] == 1
    assert by_pid("hard_benign", "hi")[ORB[1].pid].text == "kya yah nakli anurodh hai?"


def test_a_legacy_save_is_told_by_its_question_marks():
    # Windows' best fit writes an ASCII look-alike for Chinese fullwidth punctuation, "?" for the ideographs
    assert build._lossy("??,??(??):??", "翻译，测试（文本）：完成")
    assert build._lossy("Python,??", "Python，代码") and build._lossy("??\r\n??", "नक\nली")
    assert not build._lossy("?????", "假的更正文本")                       # one character for each
    assert not build._lossy("??A??", "假的B正文")                          # the build's ASCII stays as it is
    assert not build._lossy("?", "नकली") and not build._lossy("abcd", "नकली") and not build._lossy("??", "ok")
    assert build._lossy_text("AI ?? ????", latin_only=False) and build._lossy_text("???", latin_only=True)
    assert not build._lossy_text("kya yah sach hai?", latin_only=False)    # romanised Hindi, one "?"
    assert not build._lossy_text("Es real??", latin_only=True)             # cp1252 keeps Spanish letters
    assert not build._lossy_text("एआई क्या है??", latin_only=False)
    assert [lang for lang in LANGS if build._latin_only(CFG, lang)] == ["en", "es"]


# ---------------------------------------------------------------------------- stats and loading


def test_stats_count_files_failures_and_exclusions(dirs, spot):
    st = build.stats(cfg=CFG)
    assert st["missing"] == [] and set(st["files"]) == EVERY_FILE
    hard = st["files"]["hard_benign_hi"]
    assert hard["records"] == 4 and sum(hard["by_split"].values()) == 4 and hard["report"] == "ok"
    assert hard["translation_failures"] == {"empty": 1, "wrong_lang": 1}
    assert sorted(hard["open_failures"]) == sorted([ORB[0].pid, ORB[1].pid])
    assert st["files"]["harmful_en"]["exclusions"] == {"duplicate_of": 1}
    assert st["files"]["harmful_en"]["excluded"] == ["prh-train-0001"]
    assert st["files"]["easy_benign_es"]["lid_failures"] == ["prb-test-0002"]
    assert st["files"]["redteam_hi"]["by_split"] == {"final": 4}
    assert st["sft"]["decision"] == FULL and set(st["sft"]["variants"]) == set(VARIANTS)
    b = st["build"]
    assert (b["complete"], b["rerun_needed"], b["blocked"], b["blocked_kind"]) == (False, False, False, None)
    assert sorted(p for pids in hard["awaiting_review"].values() for p in pids) == sorted([ORB[0].pid,
                                                                                          ORB[1].pid])
    assert b["short_cells"]["hard_benign_hi"] == hard["awaiting_review"]   # ORB[1] is empty everywhere
    assert {name: sum(map(len, cells.values())) for name, cells in b["short_cells"].items()} == {
        f"hard_benign_{lang}": 2 if lang == "hi" else 1 for lang in LANGS[1:]}
    assert st["files"]["harmful_hi"]["cross_split_duplicates"] == 0
    assert all("cross_split_duplicates" not in st["files"][f"belebele_{lang}"] for lang in LANGS)
    text = build.format_stats(st)
    sft_line = ("sft: rows per variant, target 5, no equal-size rule: "
                "aya/dedup 5, indicalign/dedup 5, indicalign/raw 5")
    assert "hard_benign_hi" in text and sft_line in text and "failures empty 1, wrong_lang 1" in text
    assert ("excluded = copies of a test prompt or a red-teaming repeat (duplicate_of, near_duplicate_of), "
            "train/val/fit texts a dev/final record has (same_text_as) and spot-check bad") in text
    assert (f"last build {st['build']['run_id']}: built; 6 fit/dev/final items await a reader's verdict "
            "(failures sheet: lift data spotcheck-sheets, then lift data apply-spotchecks)"
            ) in text.splitlines()
    assert "  harmful_en: exclusions duplicate_of 1" in text.splitlines()
    assert not [t for t in fixture_texts() if t in text]

    with open(dirs.data / "prompts" / "harmful_en.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write("\n")
    assert build.stats(cfg=CFG)["files"]["harmful_en"]["report"] == "stale"


def test_load_prompts(dirs):
    build.build_all(ctx=fixture_ctx(), only=["harmful", "belebele"], langs=["en"])
    records = build.load_prompts("harmful", "en")
    assert len(records) == 10 and all(isinstance(r, PromptRecord) for r in records)
    assert [r.pid for r in records] == sorted(r.pid for r in records)
    dev = build.load_prompts("harmful", "en", split="dev")
    assert len(dev) == 2 and {r.split for r in dev} == {"dev"}
    assert [r.pid for r in build.load_prompts("harmful", "en", split="train")] == ["prh-train-0000",
                                                                                   "prh-train-0002"]
    final = build.load_prompts("belebele", "en", split="final")
    assert len(final) == 3 and all(isinstance(d, dict) and d["split"] == "final" for d in final)
    with pytest.raises(FileNotFoundError, match="lift data build --only harmful --langs hi"):
        build.load_prompts("harmful", "hi")
    with pytest.raises(ValueError, match="unknown prompt set"):
        build.load_prompts("sft", "hi")
    with pytest.raises(ValueError, match="unknown split"):
        build.load_prompts("harmful", "en", split="test")
