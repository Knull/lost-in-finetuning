"""lift.data.build end to end, offline: FakeHub, FakeLID, FakeTranslator, FakeCurate and the miniature
PolyRefuse tree, built into tmp data and splits dirs. Every test checks the repo's own data/ is untouched."""

import json
import logging
import runpy
from collections import Counter

import pytest
from datafakes import FakeLID, FakeTranslator, fake_translation
from fake_build import (
    POLYREFUSE_FIXTURE,
    BlockingTranslator,
    fixture_cfg,
    fixture_ctx,
    isolate,
    read_sheet,
    repo_state,
    snapshot,
    write_sheet,
)
from fake_sources import belebele_rows_by_config, orbench_rows, redteam_rows

from lift import config
from lift.data import build, polyrefuse, sources
from lift.data.build import SheetError
from lift.data.text import clean_text
from lift.io import read_jsonl, sha256_file
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
VARIANTS = ("aya/dedup", "indicalign/dedup", "aya/raw")
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
    items = build.load_prompts(set_name, lang)
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
    assert all("--skip-translation" in x["reason"] for x in summary["skipped"])
    assert not summary["complete"] and not summary["blocked"] and ctx.translator.calls == []
    assert summary["alignment"]["files"] and summary["sources"]["polyrefuse"]["commit"]
    [run] = [json.loads(p.read_text(encoding="utf-8")) for p in (dirs.outputs / "manifests").glob("*.json")]
    assert (run["stage"], run["run_id"], run["complete"]) == ("data-build", summary["run_id"], False)

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

    # the repeated harmful prompt shares a split; the train row repeating a test prompt is out everywhere
    harmful = split_file(dirs, "harmful")["assignments"]
    assert harmful["prh-test-0001"] == harmful["prh-test-0004"]
    assert summary["splits"]["harmful"]["grouped"] == 2
    for lang in ("en", "zh", "es", "ar", "th"):
        assert "prh-train-0001" not in by_pid("harmful", lang)
        assert report(dirs, f"harmful_{lang}")["exclusions"] == [
            {"pid": "prh-train-0001", "reason": "duplicate_of:prh-test-0002"}]
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
    assert summary["sft"]["decision"] == decision == {"rows_target": 5, "rows": 5, "limited_by": None}

    # GlotLID on every record; a report per file that matches it; no text in reports or logs
    for name in built_files(dirs):
        set_name, lang = name.rsplit("_", 1)
        rep = report(dirs, name)
        assert rep["sha256"] == sha256_file(dirs.data / "prompts" / f"{name}.jsonl")
        assert rep["records"] == len(build.load_prompts(set_name, lang)) and rep["lid_failures"] == []
        if set_name != "belebele":
            assert all(r.lid is not None for r in build.load_prompts(set_name, lang))
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


def test_build_context_defaults_and_checks():
    ctx = build.BuildContext.default(skip_translation=True)
    assert ctx.cfg is config.load("project") and ctx.train_cfg is config.load("train")
    assert ctx.skip_translation and ctx.lid is None and ctx.hub is None
    assert ctx.backend("google") is ctx.backend("google")             # one backend, made on its first call
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
    assert summary["complete"]                     # listed failures are expected, not an incomplete build
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


def test_a_blocked_translator_stops_all_translation_and_the_rest_is_built(dirs):
    translator = BlockingTranslator(ok=3)
    summary = build.build_all(ctx=fixture_ctx(translator=translator))
    assert summary["blocked"] and not summary["complete"] and "429" in summary["blocked_reason"]
    assert len(translator.calls) == 4                                  # never called after the block
    reasons = {f"{x['set']}_{x['lang']}": x["reason"] for x in summary["skipped"]}
    assert set(reasons) == NEEDS_TRANSLATION and reasons["harmful_hi"] == "translation blocked"
    assert built_files(dirs) == EVERY_FILE - NEEDS_TRANSLATION and summary["sft"]["decision"]["rows"] == 5
    assert report(dirs, "build")["blocked"]

    again = FakeTranslator()                                           # Google is back: resume
    summary = build.build_all(ctx=fixture_ctx(translator=again))
    assert summary["complete"] and built_files(dirs) == EVERY_FILE
    assert len(again.calls) == sum(CALLS_PER_LANG.values()) - 3       # the first three came from the cache


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
    expected = {f"hi-{r.pid}" for s in build.TRANSLATED_SETS for r in build.load_prompts(s, "hi")
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
    assert result["files"] == {"easy_benign_hi": {"ok": 0, "fixed": [], "bad": [], "unreviewed": 2},
                               "hard_benign_hi": {"ok": 0, "fixed": [], "bad": [], "unreviewed": 2},
                               "harmful_hi": {"ok": 1, "fixed": [fixed_pid], "bad": [], "unreviewed": 0}}
    assert result["rebuilt"] == ["harmful_hi"] and result["not_applied"] == [] and result["reviewed"] == 2
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
                (6, "translation differs from the current build"),
                (8, "no such item in the current harmful_hi"),
                (9, "item_id is not <lang>-<pid>"), (10, "the item is also on row 7")]
    assert len(problems) == len(expected)
    for number, what in expected:
        assert any(p.startswith(f"row {number}") and what in p for p in problems), (number, what)
    message = str(e.value)
    assert "8 problem(s), nothing applied" in message
    cells = ("कोई सुधार", "बदला हुआ नकली पाठ", rows[0]["english"], rows[0]["translation"], "not an item id")
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
    assert st["sft"]["decision"]["rows"] == 5 and set(st["sft"]["variants"]) == set(VARIANTS)
    assert st["build"]["complete"] is True
    text = build.format_stats(st)
    assert "hard_benign_hi" in text and "N = 5 rows" in text and "failures empty 1, wrong_lang 1" in text
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
