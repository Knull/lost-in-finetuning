"""lift.data.polyrefuse on a miniature PolyRefuse tree (tests/fixtures/polyrefuse, placeholder text).
Offline: git is faked, except one test that clones a local file:// repo when git is installed."""

import copy
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
from datafakes import FakeLID

from lift import config
from lift.data import polyrefuse as pr
from lift.data.polyrefuse import AlignmentError, CommitMismatch
from lift.data.text import clean_text
from lift.schemas import TranslationInfo

POLYREFUSE_FIXTURE = Path(__file__).parent / "fixtures" / "polyrefuse" / "Multilingual-Refusal"
FIX = runpy.run_path(str(POLYREFUSE_FIXTURE.parent / "make_fixture.py"))   # the texts the tree holds
COUNTS = {"harmful_test": 6, "harmless_test": 4, "harmful_train": 3, "harmful_val": 2,
          "harmless_train": 3, "harmless_val": 2}
LANGS = ["en", "hi", "zh", "es", "ar", "th"]
PIN = config.load("project")["sources"]["polyrefuse"]["commit"]
HARMFUL_EN = [en for en, _ in FIX["HARMFUL_TEST"]]
HARMFUL_CATS = [cat for _, cat in FIX["HARMFUL_TEST"]]
VAL_EN = [en for en, _ in FIX["HARMFUL_VAL"]]
VAL_CATS = [cat for _, cat in FIX["HARMFUL_VAL"]]


def fixture_cfg(**polyrefuse):
    cfg = copy.deepcopy(config.load("project"))
    cfg["sources"]["polyrefuse"]["counts"] = dict(COUNTS)
    cfg["sources"]["polyrefuse"].update(polyrefuse)
    return cfg


CFG = fixture_cfg()


def read(root, rel):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def mutate(root, rel, change):
    """Rewrite one file of a copied tree the way upstream writes it."""
    rows = change(read(root, rel))
    with open(root / rel, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=4)


def set_field(i, value, *fields):
    def change(rows):
        for field in fields:
            rows[i][field] = value
        return rows
    return change


def row_texts(root):
    texts = set()
    for path in root.rglob("*.json"):
        try:
            rows = read(root, path.relative_to(root))
        except json.JSONDecodeError:
            continue
        texts |= {v for row in rows for k, v in row.items() if k.startswith("instruction")}
    return {t for text in texts for t in (text, clean_text(text)) if len(t) >= 5}


def assert_no_row_text(message, root=POLYREFUSE_FIXTURE):
    leaked = [t for t in row_texts(root) if t in message]
    assert not leaked, f"{len(leaked)} row texts in the message"


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "Multilingual-Refusal"
    shutil.copytree(POLYREFUSE_FIXTURE, root)
    return root


# --- the fixture ---------------------------------------------------------------------------------

def test_fixture_is_written_like_upstream():
    files = sorted(POLYREFUSE_FIXTURE.rglob("*.json"))
    assert len(files) == 38
    for path in files:
        raw = path.read_bytes()
        literal = path.parent.name == "harmful_test_back_translated"      # upstream: literal UTF-8
        text = raw.decode("utf-8")
        assert text == json.dumps(json.loads(text), indent=4, ensure_ascii=not literal), path.name
        assert b"\r" not in raw and not raw.endswith(b"\n") and raw.isascii() != literal, path.name
    prdir, splits = POLYREFUSE_FIXTURE / "PolyRefuse", POLYREFUSE_FIXTURE / "dataset" / "splits"
    assert not (prdir / "harmless_test_translated_zh.json").exists()
    assert not list(POLYREFUSE_FIXTURE.rglob("*_hi.json"))
    en_test = (prdir / "harmful_test_translated_en.json").read_bytes()
    assert en_test == (splits / "harmful_test.json").read_bytes()
    assert b"\\u200b" in (prdir / "harmful_test_translated_es.json").read_bytes()


def test_fixture_tree_is_what_its_builder_writes(tmp_path):
    def files(root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    FIX["build"](tmp_path)
    assert files(tmp_path) == files(POLYREFUSE_FIXTURE)


# --- loaders -------------------------------------------------------------------------------------

def test_english_sources_follow_the_verified_mapping():
    def english(kind, split):
        return pr.english_sources(POLYREFUSE_FIXTURE, kind, split, cfg=CFG)

    assert english("harmful", "test") == HARMFUL_EN
    assert english("harmless", "test") == FIX["HARMLESS_TEST"]        # PolyRefuse's en file, not the decoy
    assert not set(english("harmless", "test")) & set(FIX["DECOY_HARMLESS_TEST"])
    assert english("harmful", "train") == FIX["HARMFUL_TRAIN"]
    assert english("harmful", "val") == VAL_EN
    assert english("harmless", "train") == FIX["HARMLESS_TRAIN"]
    assert english("harmless", "val") == FIX["HARMLESS_VAL"]
    assert english("harmful", "train")[1] == HARMFUL_EN[2]              # the train/test leak the build drops


def test_en_test_records_read_the_english_instruction():
    recs = pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmful", "test", "en", cfg=CFG)
    assert [r.pid for r in recs] == [f"prh-test-{i:04d}" for i in range(6)]
    assert [r.text for r in recs] == HARMFUL_EN == [r.text_en for r in recs]
    assert [r.category for r in recs] == HARMFUL_CATS
    assert recs[1].text == recs[4].text and recs[1].category != recs[4].category   # upstream's duplicates
    assert "\n" in recs[3].text and "  " in recs[5].text                           # kept as written
    for i, r in enumerate(recs):
        assert (r.lang, r.set, r.split, r.source) == ("en", "harmful", "dev", "polyrefuse")
        assert r.translation is None and r.lid is None
        assert r.extra == {"src_file": "PolyRefuse/harmful_test_translated_en.json", "src_row": i}
    benign = pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmless", "test", "en", cfg=CFG)
    assert [r.pid for r in benign] == [f"prb-test-{i:04d}" for i in range(4)]
    assert [r.text for r in benign] == FIX["HARMLESS_TEST"]
    assert {(r.set, r.split, r.category) for r in benign} == {("easy_benign", "dev", None)}
    assert benign[0].extra["src_file"] == "PolyRefuse/harmless_test_translated_en.json"


def test_translated_test_records_read_instruction_translated():
    rel = "PolyRefuse/harmful_test_translated_th.json"
    raw = read(POLYREFUSE_FIXTURE, rel)
    recs = pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmful", "test", "th", cfg=CFG)
    assert [r.text for r in recs] == [row["instruction_translated"] for row in raw]
    assert [r.text_en for r in recs] == HARMFUL_EN == [row["instruction"] for row in raw]
    assert [r.category for r in recs] == HARMFUL_CATS
    for i, r in enumerate(recs):
        assert (r.pid, r.lang, r.set, r.split, r.source) == (f"prh-test-{i:04d}", "th", "harmful", "dev",
                                                              "polyrefuse")
        assert r.translation == TranslationInfo(engine="polyrefuse", source_lang="en") and r.lid is None
        assert r.extra == {"src_file": rel, "src_row": i}
    assert "\n" in recs[3].text
    benign = pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmless", "test", "ar", cfg=CFG)
    assert [r.text_en for r in benign] == FIX["HARMLESS_TEST"]
    assert [r.text for r in benign] == [FIX["translate"](en, "ar") for en in FIX["HARMLESS_TEST"]]
    assert {(r.set, r.category) for r in benign} == {("easy_benign", None)}


@pytest.mark.parametrize(("kind", "split", "english", "categories"), [
    ("harmful", "train", FIX["HARMFUL_TRAIN"], [None] * 3),
    ("harmful", "val", VAL_EN, VAL_CATS),
    ("harmless", "train", FIX["HARMLESS_TRAIN"], [None] * 3),
    ("harmless", "val", FIX["HARMLESS_VAL"], [None] * 2),
])
def test_translated_train_val_pair_with_the_dataset_splits_english(kind, split, english, categories):
    rel = f"PolyRefuse/{kind}_{split}_translated_es.json"
    recs = pr.load_polyrefuse(POLYREFUSE_FIXTURE, kind, split, "es", cfg=CFG)
    prefix = "prh" if kind == "harmful" else "prb"
    assert [r.pid for r in recs] == [f"{prefix}-{split}-{i:04d}" for i in range(len(english))]
    assert [r.text for r in recs] == [row["instruction"] for row in read(POLYREFUSE_FIXTURE, rel)]
    assert [r.text for r in recs] == [FIX["translate"](en, "es") for en in english]
    assert [r.text_en for r in recs] == english
    assert [r.category for r in recs] == categories
    set_name = "harmful" if kind == "harmful" else "easy_benign"
    assert {(r.lang, r.set, r.split) for r in recs} == {("es", set_name, split)}
    assert all(r.translation == TranslationInfo(engine="polyrefuse") for r in recs)
    assert [r.extra for r in recs] == [{"src_file": rel, "src_row": i} for i in range(len(english))]


@pytest.mark.parametrize(("kind", "split", "src"), [
    ("harmful", "train", "dataset/splits/harmful_train.json"),
    ("harmful", "val", "dataset/splits/harmful_val.json"),
    ("harmless", "train", "dataset/splits/harmless_train_200_sampled.json"),
    ("harmless", "val", "dataset/splits/harmless_val_200_sampled.json"),
])
def test_en_train_val_records_read_dataset_splits(kind, split, src):
    raw = read(POLYREFUSE_FIXTURE, src)
    recs = pr.load_polyrefuse(POLYREFUSE_FIXTURE, kind, split, "en", cfg=CFG)
    assert [r.text for r in recs] == [row["instruction"] for row in raw] == [r.text_en for r in recs]
    assert [r.category for r in recs] == [row["category"] for row in raw]
    assert all(r.translation is None and r.split == split and r.lang == "en" for r in recs)
    assert [r.extra for r in recs] == [{"src_file": src, "src_row": i} for i in range(len(raw))]


def test_pids_and_english_are_shared_across_languages():
    for kind, split in pr.BUILD_FILES:
        langs = [lang for lang in LANGS if pr.has_file(POLYREFUSE_FIXTURE, kind, split, lang, cfg=CFG)]
        assert "en" in langs and len(langs) >= 4
        loaded = {lang: pr.load_polyrefuse(POLYREFUSE_FIXTURE, kind, split, lang, cfg=CFG) for lang in langs}
        for lang, recs in loaded.items():
            assert [r.pid for r in recs] == [r.pid for r in loaded["en"]], (kind, split, lang)
            assert [r.text_en for r in recs] == [r.text for r in loaded["en"]], (kind, split, lang)
            assert {r.lang for r in recs} == {lang}


def test_clean_text_drops_zero_width_spaces_and_strips(tree):
    rel = "PolyRefuse/harmful_test_translated_es.json"
    raw = read(POLYREFUSE_FIXTURE, rel)[0]["instruction_translated"]
    assert "​" in raw
    recs = pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmful", "test", "es", cfg=CFG)
    assert recs[0].text == raw.replace("​", "")
    padded = " ​" + FIX["translate"](HARMFUL_EN[2], "es") + "\n"
    mutate(tree, rel, set_field(2, padded, "instruction_translated"))
    assert pr.load_polyrefuse(tree, "harmful", "test", "es", cfg=CFG)[2].text == padded.strip(" ​\n")


@pytest.mark.parametrize(("kind", "split", "lang"), [
    *[(kind, split, "hi") for kind in pr.KINDS for split in pr.SPLITS],
    ("harmless", "test", "zh"),
])
def test_missing_files_raise_file_not_found(kind, split, lang):
    assert not pr.has_file(POLYREFUSE_FIXTURE, kind, split, lang, cfg=CFG)
    with pytest.raises(FileNotFoundError, match="translated from the English") as exc:
        pr.load_polyrefuse(POLYREFUSE_FIXTURE, kind, split, lang, cfg=CFG)
    assert_no_row_text(str(exc.value))


@pytest.mark.parametrize("split", ["train", "val"])
def test_misaligned_zh_harmless_files_raise_alignment_error(split):
    rel = f"PolyRefuse/harmless_{split}_translated_zh.json"
    assert (POLYREFUSE_FIXTURE / rel).is_file()
    assert not pr.has_file(POLYREFUSE_FIXTURE, "harmless", split, "zh", cfg=CFG)
    with pytest.raises(AlignmentError, match="misaligned") as exc:
        pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmless", split, "zh", cfg=CFG)
    assert [(p["file"], p["check"]) for p in exc.value.problems] == [(rel, "misaligned")]
    assert_no_row_text(str(exc.value))
    zh_harmful = pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmful", split, "zh", cfg=CFG)
    assert len(zh_harmful) == COUNTS[f"harmful_{split}"]


def test_has_file():
    root = POLYREFUSE_FIXTURE
    for kind in pr.KINDS:
        for split in pr.SPLITS:
            assert pr.has_file(root, kind, split, "en", cfg=CFG)              # the English sources
            assert not pr.has_file(root, kind, split, "hi", cfg=CFG)
            assert all(pr.has_file(root, kind, split, lang, cfg=CFG) for lang in ("es", "ar", "th", "de"))
            assert pr.has_file(root, kind, split, "zh", cfg=CFG) == (kind == "harmful")
    assert pr.has_file(root, "harmless", "train", "zh", cfg=fixture_cfg(misaligned=[]))
    assert not pr.has_file(root.parent / "nowhere", "harmful", "test", "es", cfg=CFG)


def test_unknown_family_or_language_raise():
    with pytest.raises(ValueError, match="family"):
        pr.load_polyrefuse(POLYREFUSE_FIXTURE, "benign", "test", "es", cfg=CFG)
    with pytest.raises(ValueError, match="family"):
        pr.english_sources(POLYREFUSE_FIXTURE, "harmful", "dev", cfg=CFG)
    with pytest.raises(ValueError, match="unsupported language"):        # in the tree, not a project language
        pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmful", "test", "de", cfg=CFG)


def test_row_counts_come_from_the_config():
    real = config.load("project")                                          # 572, 500, 260, 39, 200, 200
    with pytest.raises(AlignmentError, match="row_count") as exc:
        pr.english_sources(POLYREFUSE_FIXTURE, "harmful", "test", cfg=real)
    assert exc.value.problems == [{"file": "PolyRefuse/harmful_test_translated_en.json", "check": "row_count",
                                   "row": None, "detail": "6 rows, expected 572"}]
    with pytest.raises(AlignmentError, match="row_count"):
        pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmful", "train", "es", cfg=real)
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(POLYREFUSE_FIXTURE, LANGS, cfg=real)
    assert len({p["file"] for p in exc.value.problems if p["check"] == "row_count"}) == 5 + 18


def test_a_test_file_whose_english_drifts_does_not_load(tree):
    rel = "PolyRefuse/harmful_test_translated_ar.json"
    mutate(tree, rel, set_field(4, "Another placeholder English prompt.", "instruction"))
    with pytest.raises(AlignmentError) as exc:
        pr.load_polyrefuse(tree, "harmful", "test", "ar", cfg=CFG)
    problems = [(p["file"], p["check"], p["row"]) for p in exc.value.problems]
    assert problems == [(rel, "english_instruction", 4)]
    assert_no_row_text(str(exc.value), tree)


# --- alignment checks ----------------------------------------------------------------------------

def test_check_alignment_passes_on_the_fixture():
    report = pr.check_alignment(POLYREFUSE_FIXTURE, LANGS, lid=FakeLID(), cfg=CFG)
    assert report["langs"] == ["zh", "es", "ar", "th"]
    expected = {f"PolyRefuse/{kind}_{split}_translated_{lang}.json" for kind, split in pr.BUILD_FILES
                for lang in ("zh", "es", "ar", "th") if (kind, lang) != ("harmless", "zh")}
    assert set(report["files"]) == expected and len(expected) == 18     # no de, harmless val, back-transl.
    missing, misaligned = report["skipped"]
    assert missing == {"file": "PolyRefuse/harmless_test_translated_zh.json", "reason": "missing"}
    assert misaligned["file"] == "PolyRefuse/harmless_train_translated_zh.json"
    assert misaligned["reason"] == "misaligned" and misaligned["correlation"] < pr.MIN_CORRELATION
    for rel, entry in report["files"].items():
        n = COUNTS[f"{entry['kind']}_{entry['split']}"]
        assert entry["rows"] == n, rel
        if entry["split"] == "test":
            continue
        assert [s["row"] for s in entry["spot"]] == list(range(n))          # min(5, n) evenly spaced rows
        assert all(s["ok"] and 0.1 <= s["ratio"] <= 4.0 for s in entry["spot"]), rel
        assert entry["lid_pass"] == n
        assert entry["correlation"] > 0.99 if n >= 3 else entry["correlation"] is None
    assert report["english"]["harmless_test"] == {"file": "PolyRefuse/harmless_test_translated_en.json",
                                                  "rows": 4}
    assert report["english"]["harmful_train"] == {"file": "dataset/splits/harmful_train.json", "rows": 3}
    assert report["harmless_test_source"] == {"file": "PolyRefuse/harmless_test_translated_en.json",
                                              "compared": ["es", "ar", "th"], "mismatched_rows": 0}
    json.dumps(report)                                                      # goes into the build report


def test_check_alignment_without_lid_or_translated_languages():
    entry = pr.check_alignment(POLYREFUSE_FIXTURE, ["es"], cfg=CFG)["files"][
        "PolyRefuse/harmful_train_translated_es.json"]
    assert entry["lid_pass"] is None and all(set(s) == {"row", "ratio"} for s in entry["spot"])
    report = pr.check_alignment(POLYREFUSE_FIXTURE, ["en", "hi"], cfg=CFG)
    assert report["files"] == {} and report["skipped"] == []
    assert report["harmless_test_source"]["compared"] == ["es", "ar", "th"]


def test_de_decoy_and_back_translations_are_never_read(tree):
    junk = [*tree.glob("PolyRefuse/*_de.json"), *tree.glob("PolyRefuse/harmful_test_back_translated/*.json"),
            tree / "dataset/splits/harmless_test_500_sampled.json"]
    assert len(junk) == 8
    for path in junk:
        path.write_text("not json", encoding="utf-8")
    pr.check_alignment(tree, LANGS, lid=FakeLID(), cfg=CFG)
    for kind in pr.KINDS:
        for split in pr.SPLITS:
            for lang in LANGS:
                if pr.has_file(tree, kind, split, lang, cfg=CFG):
                    pr.load_polyrefuse(tree, kind, split, lang, cfg=CFG)


ZH_TRAIN_1 = FIX["translate"](FIX["HARMFUL_TRAIN"][1], "zh")
AR_TRAIN_0 = FIX["translate"](FIX["HARMFUL_TRAIN"][0], "ar")         # 22 characters for 27 of English
TH_HARMLESS_2 = FIX["translate"](FIX["HARMLESS_TRAIN"][2], "th")     # 163 characters for 181
# name -> (file changed, change, (file, check, row) problems it must produce). On the changed file
# these are all the problems: the two length_ratio files keep their log-length correlation above
# MIN_CORRELATION (0.91 and 0.95) and their script (FakeLID), so the ratio rule is the one that fires.
FAILURES = {
    "test row English differs": (
        "PolyRefuse/harmful_test_translated_es.json",
        set_field(3, "Placeholder request 3 rewritten.", "instruction"),
        [("PolyRefuse/harmful_test_translated_es.json", "english_instruction", 3)]),
    "category mismatch": (
        "PolyRefuse/harmful_test_translated_ar.json",
        set_field(0, "Another category", "category"),
        [("PolyRefuse/harmful_test_translated_ar.json", "category", 0)]),
    "train file missing a row": (
        "PolyRefuse/harmful_train_translated_th.json",
        lambda rows: rows[:-1],
        [("PolyRefuse/harmful_train_translated_th.json", "row_count", None)]),
    "shuffled train file": (
        "PolyRefuse/harmless_train_translated_es.json",
        lambda rows: rows[::-1],
        [("PolyRefuse/harmless_train_translated_es.json", "length_correlation", None),
         ("PolyRefuse/harmless_train_translated_es.json", "length_ratio", 0)]),   # 208 of 27 characters
    "val category mismatch": (
        "PolyRefuse/harmful_val_translated_zh.json",
        lambda rows: [{**row, "category": cat} for row, cat in zip(rows, VAL_CATS[::-1], strict=True)],
        [("PolyRefuse/harmful_val_translated_zh.json", "category", 0),
         ("PolyRefuse/harmful_val_translated_zh.json", "category", 1)]),
    "harmless test English source": (
        "PolyRefuse/harmless_test_translated_en.json",
        set_field(1, "Placeholder question 1 rewritten.", "instruction", "instruction_translated"),
        [("PolyRefuse/harmless_test_translated_en.json", "harmless_test_source", 1),
         ("PolyRefuse/harmless_test_translated_es.json", "english_instruction", 1)]),
    "train spot row shortened below the ratio range": (            # 2 of 27 characters: 0.07 < 0.1
        "PolyRefuse/harmful_train_translated_ar.json",
        set_field(0, AR_TRAIN_0[:2], "instruction"),
        [("PolyRefuse/harmful_train_translated_ar.json", "length_ratio", 0)]),
    "train spot row lengthened above the ratio range": (           # 819 of 181 characters: 4.52 > 4.0
        "PolyRefuse/harmless_train_translated_th.json",
        set_field(2, " ".join([TH_HARMLESS_2] * 5), "instruction"),
        [("PolyRefuse/harmless_train_translated_th.json", "length_ratio", 2)]),
}
LID_FAILURE = ("PolyRefuse/harmful_train_translated_zh.json", "lid_spot", 1)


def misreading_lid():
    return FakeLID(overrides={ZH_TRAIN_1: [("jpn_Jpan", 0.8), ("cmn_Hani", 0.2)]})


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_each_alignment_failure_is_caught(tree, name):
    rel, change, expected = FAILURES[name]
    mutate(tree, rel, change)
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(tree, LANGS, lid=FakeLID(), cfg=CFG)
    found = {(p["file"], p["check"], p["row"]) for p in exc.value.problems}
    assert set(expected) <= found
    assert {p for p in found if p[0] == rel} == {p for p in expected if p[0] == rel}   # nothing else
    assert_no_row_text(str(exc.value), tree)


@pytest.mark.parametrize(("name", "ratio"), [("train spot row shortened below the ratio range", "0.07"),
                                             ("train spot row lengthened above the ratio range", "4.52")])
def test_the_length_ratio_rule_fires_on_its_own(tree, name, ratio):
    rel, change, [(_, _, row)] = FAILURES[name]
    mutate(tree, rel, change)
    kind = "harmful" if "harmful" in rel else "harmless"
    texts = [clean_text(r["instruction"]) for r in read(tree, rel)]
    english = pr.english_sources(tree, kind, "train", cfg=CFG)
    assert pr._log_length_correlation(english, texts) > pr.MIN_CORRELATION + 0.3
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(tree, LANGS, lid=FakeLID(), cfg=CFG)
    assert exc.value.problems == [{"file": rel, "check": "length_ratio", "row": row,
                                   "detail": f"ratio {ratio} outside [0.1, 4.0]"}]


def test_lid_spot_failure_is_caught():
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(POLYREFUSE_FIXTURE, LANGS, lid=misreading_lid(), cfg=CFG)
    assert [(p["file"], p["check"], p["row"]) for p in exc.value.problems] == [LID_FAILURE]
    assert exc.value.problems[0]["detail"] == "top jpn_Jpan 0.80; 2/3 spot rows pass, need 3"
    assert_no_row_text(str(exc.value))


def test_check_alignment_collects_every_problem_in_one_error(tree):
    for rel, change, _ in FAILURES.values():
        mutate(tree, rel, change)
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(tree, LANGS, lid=misreading_lid(), cfg=CFG)
    found = {(p["file"], p["check"], p["row"]) for p in exc.value.problems}
    expected = {problem for _, _, problems in FAILURES.values() for problem in problems} | {LID_FAILURE}
    assert expected <= found
    message = str(exc.value)
    assert all(f"{file}: {check}" in message for file, check, _ in expected)
    assert_no_row_text(message, tree)


def test_unreadable_files_are_collected_too(tree):
    (tree / "PolyRefuse/harmful_val_translated_es.json").write_text("[{", encoding="utf-8")
    (tree / "PolyRefuse/harmful_val_translated_th.json").write_bytes(b'[{"instruction": "\xff"}]')
    (tree / "PolyRefuse/harmless_train_translated_zh.json").unlink()      # listed as misaligned, now absent
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(tree, LANGS, lid=FakeLID(), cfg=CFG)
    assert [(p["file"], p["check"]) for p in exc.value.problems] == [
        ("PolyRefuse/harmful_val_translated_es.json", "format"),
        ("PolyRefuse/harmful_val_translated_th.json", "format")]
    with pytest.raises(AlignmentError, match="invalid UTF-8"):
        pr.load_polyrefuse(tree, "harmful", "val", "th", cfg=CFG)


def test_harmless_test_source_is_checked_for_a_zh_hi_only_build(tree):
    mutate(tree, "PolyRefuse/harmless_test_translated_en.json",
           set_field(2, "Placeholder question 2 rewritten.", "instruction", "instruction_translated"))
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(tree, ["hi", "zh"], lid=FakeLID(), cfg=CFG)
    assert [(p["file"], p["check"], p["row"], p["detail"]) for p in exc.value.problems] == [
        ("PolyRefuse/harmless_test_translated_en.json", "harmless_test_source", 2, "differs from es, ar, th")]
    with pytest.raises(AlignmentError, match="english_instruction"):
        pr.load_polyrefuse(tree, "harmless", "test", "es", cfg=CFG)


def test_the_misaligned_list_is_what_keeps_zh_harmless_train_out():
    unlisted = fixture_cfg(misaligned=[])
    assert len(pr.load_polyrefuse(POLYREFUSE_FIXTURE, "harmless", "train", "zh", cfg=unlisted)) == 3
    with pytest.raises(AlignmentError) as exc:
        pr.check_alignment(POLYREFUSE_FIXTURE, ["zh"], lid=FakeLID(), cfg=unlisted)
    found = {(p["file"], p["check"]) for p in exc.value.problems}
    assert ("PolyRefuse/harmless_train_translated_zh.json", "length_correlation") in found


def test_lid_spot_check_tolerates_one_misread_in_five(tree):
    extra = ["Train placeholder 3 about pressing flowers in a heavy book.", "Train placeholder 4 on jam.",
             "Train placeholder 5 about choosing paint colours for a small bedroom with one window."]
    mutate(tree, "dataset/splits/harmful_train.json",
           lambda rows: rows + [{"instruction": en, "category": None} for en in extra])
    for lang in ("es", "zh", "ar", "th", "de"):
        mutate(tree, f"PolyRefuse/harmful_train_translated_{lang}.json",
               lambda rows, lang=lang: rows + [{"instruction": FIX["translate"](en, lang), "category": None}
                                               for en in extra])
    cfg = fixture_cfg()
    cfg["sources"]["polyrefuse"]["counts"]["harmful_train"] = 6
    es = [FIX["translate"](en, "es") for en in FIX["HARMFUL_TRAIN"] + extra]
    rel = "PolyRefuse/harmful_train_translated_es.json"
    one = FakeLID(overrides={es[3]: ("eng_Latn", 0.9)})
    entry = pr.check_alignment(tree, ["es"], lid=one, cfg=cfg)["files"][rel]
    assert [s["row"] for s in entry["spot"]] == [0, 1, 2, 3, 5] and entry["lid_pass"] == 4
    assert entry["correlation"] > 0.99
    two = FakeLID(overrides={es[3]: ("eng_Latn", 0.9), es[5]: ("spa_Latn", 0.3)})    # 2nd: right, unsure
    with pytest.raises(AlignmentError, match="3/5 spot rows pass, need 4") as exc:
        pr.check_alignment(tree, ["es"], lid=two, cfg=cfg)
    assert [(p["check"], p["row"]) for p in exc.value.problems] == [("lid_spot", 3), ("lid_spot", 5)]


def test_spot_rows_and_correlation_helpers():
    assert pr._spot_rows(260) == [0, 64, 129, 194, 259]
    assert pr._spot_rows(39) == [0, 9, 19, 28, 38]
    assert pr._spot_rows(200) == [0, 49, 99, 149, 199]
    assert (pr._spot_rows(3), pr._spot_rows(1), pr._spot_rows(0)) == ([0, 1, 2], [0], [])
    assert pr._log_length_correlation(["ab", "abcd", "abcdefgh"], ["x", "xy", "xyzw"]) == pytest.approx(1.0)
    assert pr._log_length_correlation(["ab", "cd", "ef"], ["x", "xy", "xyz"]) is None    # one length only


# --- clone and commit ----------------------------------------------------------------------------

class FakeGit:
    """Stands in for subprocess.run: records each call; ``init`` makes .git; rev-parse prints ``head``."""

    def __init__(self, head=PIN, fail=None):
        self.head, self.fail, self.calls = head, fail, []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        step = argv[5]                                 # git -c core.autocrlf=false -C <root> <step> ...
        if step == self.fail:
            raise subprocess.CalledProcessError(128, argv, output="", stderr="fatal: placeholder failure\n")
        if step == "init":
            (Path(argv[4]) / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, stdout=self.head + "\n" if step == "rev-parse" else "",
                                           stderr="")


def test_ensure_clone_runs_the_pinned_git_steps(monkeypatch, tmp_path):
    git = FakeGit()
    monkeypatch.setattr(pr.subprocess, "run", git)
    root = tmp_path / "raw" / "Multilingual-Refusal"
    assert pr.ensure_clone(root, cfg=CFG) == root
    prefix = ["git", "-c", "core.autocrlf=false", "-C", str(root)]
    assert [argv for argv, _ in git.calls] == [
        [*prefix, "init", "-q"],
        [*prefix, "config", "core.autocrlf", "false"],
        [*prefix, "remote", "add", "origin", CFG["sources"]["polyrefuse"]["repo"]],
        [*prefix, "fetch", "-q", "--depth", "1", "origin", PIN],
        [*prefix, "checkout", "-q", "FETCH_HEAD"],
        [*prefix, "rev-parse", "--verify", "HEAD"],
    ]
    for argv, kw in git.calls:
        assert type(argv) is list and not kw.get("shell") and kw["check"] is True
        assert kw["env"]["GIT_TERMINAL_PROMPT"] == "0"
    git.calls.clear()
    assert pr.ensure_clone(root, cfg=CFG) == root                  # an existing clone is only checked
    assert [argv[5:] for argv, _ in git.calls] == [["rev-parse", "--verify", "HEAD"]]


def test_ensure_clone_never_moves_an_existing_clone(monkeypatch, tmp_path):
    root = tmp_path / "clone"
    (root / ".git").mkdir(parents=True)
    git = FakeGit(head="0" * 40)
    monkeypatch.setattr(pr.subprocess, "run", git)
    with pytest.raises(CommitMismatch, match="pins"):
        pr.ensure_clone(root, cfg=CFG)
    assert [argv[5] for argv, _ in git.calls] == ["rev-parse"]


@pytest.mark.parametrize("commit", ["TBD", None, "main", PIN[:12]])
def test_an_unpinned_commit_is_refused_before_any_git_call(monkeypatch, tmp_path, commit):
    git = FakeGit()
    monkeypatch.setattr(pr.subprocess, "run", git)
    cfg, root = fixture_cfg(commit=commit), tmp_path / "clone"
    with pytest.raises(CommitMismatch, match="full 40-character"):
        pr.ensure_clone(root, cfg=cfg)
    assert not root.exists()
    (root / ".git").mkdir(parents=True)
    with pytest.raises(CommitMismatch, match="full 40-character"):
        pr.check_commit(root, cfg=cfg)
    assert git.calls == []


def test_check_commit(monkeypatch, tmp_path):
    root = tmp_path / "clone"
    with pytest.raises(CommitMismatch, match="not a git clone"):    # never asks git about an enclosing repo
        pr.check_commit(root, cfg=CFG)
    (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(pr.subprocess, "run", FakeGit())
    assert pr.check_commit(root, cfg=CFG) == PIN == pr.head_commit(root)
    monkeypatch.setattr(pr.subprocess, "run", FakeGit(head="f" * 40))
    with pytest.raises(CommitMismatch, match="f" * 40):
        pr.check_commit(root, cfg=CFG)
    monkeypatch.setattr(pr.subprocess, "run", FakeGit(fail="rev-parse"))
    with pytest.raises(CommitMismatch, match="no checked-out commit"):
        pr.check_commit(root, cfg=CFG)


def test_a_failed_clone_step_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(pr.subprocess, "run", FakeGit(fail="fetch"))
    with pytest.raises(RuntimeError, match="git fetch failed .*placeholder failure.*delete"):
        pr.ensure_clone(tmp_path / "clone", cfg=CFG)


def test_default_root_follows_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LIFT_DATA_DIR", str(tmp_path))
    assert pr.default_root() == tmp_path.resolve() / "raw" / "Multilingual-Refusal"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_ensure_clone_with_real_git_keeps_upstream_bytes(monkeypatch, tmp_path):
    """A local file:// upstream, and a global git config with core.autocrlf=true (Windows' default):
    a plain clone gets CRLF files, ensure_clone's clone the upstream bytes."""
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("[core]\n\tautocrlf = true\n[user]\n\tname = Placeholder\n"
                         "\temail = placeholder@example.com\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    upstream = tmp_path / "upstream"
    shutil.copytree(POLYREFUSE_FIXTURE, upstream)

    def git(*args, cwd=upstream):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "placeholder")
    sha = git("rev-parse", "HEAD")
    cfg = fixture_cfg(repo=upstream.as_uri(), commit=sha)
    sample = "PolyRefuse/harmful_test_translated_en.json"
    git("clone", "-q", upstream.as_uri(), str(tmp_path / "plain"), cwd=tmp_path)
    assert b"\r\n" in (tmp_path / "plain" / sample).read_bytes()       # what -c core.autocrlf=false prevents

    root = tmp_path / "raw" / "Multilingual-Refusal"
    assert pr.ensure_clone(root, cfg=cfg) == root
    for path in upstream.rglob("*.json"):
        assert (root / path.relative_to(upstream)).read_bytes() == path.read_bytes(), path.name
    assert pr.head_commit(root) == sha == pr.check_commit(root, cfg=cfg)
    assert git("config", "core.autocrlf", cwd=root) == "false"
    assert pr.ensure_clone(root, cfg=cfg) == root
    with pytest.raises(CommitMismatch):
        pr.ensure_clone(root, cfg=fixture_cfg(repo=upstream.as_uri(), commit="0" * 40))
    pr.check_alignment(root, LANGS, lid=FakeLID(), cfg=cfg)
