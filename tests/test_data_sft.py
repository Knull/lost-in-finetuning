"""lift.data.sft: the SFT pools, the shared utility held-out set and the CuratorKIT candidates."""

import copy
import hashlib
import json
import logging
import random
import re
import string
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pytest
from datafakes import FakeLID
from fake_sft import (
    AYA_CROSS,
    AYA_DUPLICATE,
    AYA_EMPTY_TARGET,
    AYA_ENGLISH_INPUT,
    AYA_ENGLISH_TARGET,
    AYA_NEAR_DUP,
    AYA_REPETITIVE,
    AYA_SHORT,
    IND_CROSS,
    IND_HAD_CONTEXT,
    IND_NEAR_PAIR,
    IND_NO_MATCH,
    FakeCurate,
    SftHub,
    aya_dataset_rows,
    dolly_rows,
    indicalign_rows,
)

from lift import config, splits
from lift.data import sft
from lift.data.sft import SftRow
from lift.data.text import char_ngrams, dup_key, jaccard
from lift.io import read_jsonl, sha256_file


def project_cfg(per_source=2, utility=None):
    cfg = copy.deepcopy(config.load("project"))
    cfg["sources"]["sft"]["heldout"]["per_source"] = per_source
    cfg["sizes"]["utility"] = utility or {"dev": 1, "final": 3}
    return cfg


def training_cfg(rows=5):
    cfg = copy.deepcopy(config.load("train"))
    cfg["rows"] = rows
    return cfg


CFG = project_cfg()
SEED = CFG["split_seed"]
ND = CFG["sources"]["sft"]["heldout"]["near_dup"]
VARIANTS = ("aya/dedup", "indicalign/dedup", "aya/raw")


def near(a_ins, a_res, b_ins, b_res, nd=ND):
    """The near-duplicate rule written out pair by pair (the reference for the indexed join)."""
    ins = (bool(dup_key(a_ins)) and bool(dup_key(b_ins))
           and jaccard(char_ngrams(a_ins, nd["ngram"]), char_ngrams(b_ins, nd["ngram"])) >= nd["instruction"])
    long = len(dup_key(a_res)) >= nd["response_min_chars"] and len(dup_key(b_res)) >= nd["response_min_chars"]
    res = long and jaccard(char_ngrams(a_res, nd["ngram"]), char_ngrams(b_res, nd["ngram"])) >= nd["response"]
    return ins or res


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    root = (tmp_path / "data").resolve()
    monkeypatch.setenv("LIFT_DATA_DIR", str(root))
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")
    return root


@pytest.fixture
def pools():
    return sft.load_pools(cfg=CFG, hub=SftHub(), lid=FakeLID())


def build(pools, *, rows=5, curate=None):
    heldout = sft.build_utility_heldout(pools, cfg=CFG)
    train = sft.exclude_heldout(pools, heldout, cfg=CFG)
    report = sft.build_candidates(train, pool_counts={k: c for k, (_, c) in pools.items()}, cfg=CFG,
                                  train_cfg=training_cfg(rows), curate=curate or FakeCurate())
    return report, train, heldout


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- loading and cleaning


def test_aya_rows_map_fields_and_number_the_hindi_rows():
    raw = aya_dataset_rows()
    rows = sft.aya_rows(list(reversed(raw)), language="Hindi")
    assert [r.idx for r in rows] == list(range(len(raw)))
    assert [r.hf_row for r in rows] == [r["_row"] for r in raw]          # non-contiguous HF rows
    assert (rows[5].source, rows[5].instruction, rows[5].response) == (
        "aya", raw[5]["inputs"], raw[5]["targets"])
    assert rows[5].meta == {"annotation_type": raw[5]["annotation_type"]}
    assert (rows[0].row_id(), rows[17].row_id()) == ("aya-00000", "aya-00017")
    with pytest.raises(ValueError, match="not 'Hindi'"):
        sft.aya_rows(raw + [{**raw[0], "language": "English", "_row": 999}], language="Hindi")


def test_indicalign_rows_apply_the_dolly_context_rule():
    raw = indicalign_rows()
    rows, counts = sft.indicalign_rows(list(reversed(raw)), dolly_rows())
    assert counts == Counter(kept=15, had_context=3, no_match=1)
    kept = [r.idx for r in rows]
    assert kept == sorted(kept) and set(kept).isdisjoint(IND_HAD_CONTEXT + IND_NO_MATCH)
    # row 4 matches only with whitespace collapsed, row 5 only after NFC, row 8's context is blank
    assert {4, 5, 8} <= set(kept)
    assert raw[5]["eng_Latn"][0][0] not in {d["instruction"] for d in dolly_rows()}
    row = rows[kept.index(4)]
    assert (row.source, row.hf_row, row.row_id()) == ("indicalign", 4, "ind-00004")
    assert row.meta == {"doc_id": "doc-0004"}
    assert [row.instruction, row.response] == raw[4]["hin_Deva"][0]
    bad = indicalign_rows()
    bad[0]["hin_Deva"] = [["केवल एक भाग"]]
    with pytest.raises(ValueError, match="row 0"):
        sft.indicalign_rows(bad, dolly_rows())


def test_clean_pool_counts_every_drop_reason():
    extra = [SftRow("aya", 90, 900, " \u200b ", "यह उत्तर है पर प्रश्न खाली है।"),
             SftRow("aya", 91, 901, "  \u200bटमाटर कब लगाएँ?  ",
                    "\u200bबरसात से पहले टमाटर लगाना अच्छा रहता है। ")]
    rows = sft.aya_rows(aya_dataset_rows()) + extra
    pool, counts = sft.clean_pool(rows, FakeLID(), cfg=CFG)
    assert counts == Counter(kept=15, empty_instruction=1, empty_response=1, instruction_not_hi=1,
                             response_not_hi=1, repetitive_response=1)
    dropped = {r.idx for r in rows} - {r.idx for r in pool}
    assert dropped == {AYA_ENGLISH_INPUT, AYA_EMPTY_TARGET, AYA_REPETITIVE, AYA_ENGLISH_TARGET, 90}
    cleaned = pool[-1]
    assert cleaned.instruction == "टमाटर कब लगाएँ?"
    assert cleaned.response == "बरसात से पहले टमाटर लगाना अच्छा रहता है।"
    # a GlotLID misread of a Hindi answer drops the row; the repetition limit comes from cfg
    first = pool[0]
    misread = FakeLID({first.response: ("mar_Deva", 0.9)})
    assert sft.clean_pool([first], misread, cfg=CFG)[1]["response_not_hi"] == 1
    loose = copy.deepcopy(CFG)
    loose["unclear"]["max_repetition_ratio"] = 0.99
    assert sft.clean_pool(rows, FakeLID(), cfg=loose)[1]["repetitive_response"] == 0


def test_load_pools_reads_the_hub_and_counts_every_stage():
    hub = SftHub()
    pools = sft.load_pools(cfg=CFG, hub=hub, lid=FakeLID())
    assert sorted(hub.calls) == ["aya_dataset", "dolly", "indicalign"]
    (aya, aya_counts), (ind, ind_counts) = pools["aya"], pools["indicalign"]
    assert (aya_counts["loaded"], aya_counts["kept"], len(aya)) == (18, 14, 14)
    assert (ind_counts["loaded"], ind_counts["kept"], len(ind)) == (19, 14, 14)
    assert (ind_counts["had_context"], ind_counts["no_match"], ind_counts["response_not_hi"]) == (3, 1, 1)
    for counts in (aya_counts, ind_counts):
        assert counts["loaded"] == sum(v for k, v in counts.items() if k != "loaded")


# ---------------------------------------------------------------- held-out selection


def test_select_heldout_takes_unique_instructions_in_the_token_window(pools):
    pool = pools["aya"][0]
    eligible = {r.idx for r in pool} - {1, AYA_DUPLICATE, AYA_SHORT}      # repeated instruction; 3 tokens
    assert len(eligible) == 11
    picked = {r.idx for r in sft.select_heldout(pool, 10, seed=SEED, tag="t", cfg=CFG)}
    assert picked <= eligible and len(picked & {3, AYA_NEAR_DUP}) == 1   # never both near-duplicates
    with pytest.raises(ValueError, match="only 10 held-out rows"):
        sft.select_heldout(pool, 11, seed=SEED, tag="t", cfg=CFG)


def test_select_heldout_is_a_deterministic_hash_ranking(pools):
    pool = pools["aya"][0]
    picked = sft.select_heldout(pool, 3, seed=SEED, tag="utility:aya", cfg=CFG)
    assert picked == sft.select_heldout(list(reversed(pool)), 3, seed=SEED, tag="utility:aya", cfg=CFG)
    assert [r.idx for r in picked] == sorted(r.idx for r in picked)
    eligible = [r for r in pool if r.idx not in {1, AYA_DUPLICATE, AYA_SHORT}]
    ranked = sorted(eligible,
                    key=lambda r: hashlib.sha256(f"{SEED}:utility:aya:{r.row_id()}".encode()).hexdigest())
    expected = []
    for r in ranked:
        if len(expected) < 3 and not ({r.idx} | {e.idx for e in expected}) >= {3, AYA_NEAR_DUP}:
            expected.append(r)
    assert picked == sorted(expected, key=lambda r: r.idx)
    draws = {tuple(r.idx for r in sft.select_heldout(pool, 3, seed=s, tag="t", cfg=CFG)) for s in range(8)}
    assert len(draws) > 1


def test_select_heldout_token_window_matches_the_schema_gate():
    words = ["पहला प्रश्न यहाँ लिखा", "दूसरा अनुरोध कुछ अलग", "तीसरी बात बिल्कुल नई", "चौथा सवाल फिर से"]

    def row(i, total):            # instruction (4 words) + response = ``total`` whitespace tokens
        return SftRow("aya", i, i, words[i], " ".join(f"शब्द{i}x{k}" for k in range(total - 4)))

    pool = [row(0, 9), row(1, 10), row(2, 2048), row(3, 2049)]
    assert {r.idx for r in sft.select_heldout(pool, 2, seed=SEED, tag="t", cfg=CFG)} == {1, 2}
    with pytest.raises(ValueError):
        sft.select_heldout(pool, 3, seed=SEED, tag="t", cfg=CFG)


def test_schema_tokens_counts_like_curatorkit():
    tokens = pytest.importorskip("curatorkit.utils.tokens")
    for text in ["बीज बोने से पहले मिट्टी", "plain English words", "汉字和 English 混合", "ภาษาไทย ok", "",
                 "  a\tb\nc  ", "日本語のテキスト and कुछ हिंदी"]:
        assert sft.schema_tokens(text) == tokens.count_tokens_whitespace(text)


def test_select_heldout_avoids_the_other_sources_draw(pools):
    aya, ind = pools["aya"][0], pools["indicalign"][0]
    cross = next(r for r in aya if r.idx == AYA_CROSS)
    assert IND_CROSS in {r.idx for r in sft.select_heldout(ind, 11, seed=SEED, tag="t", cfg=CFG)}
    picked = sft.select_heldout(ind, 10, seed=SEED, tag="t", cfg=CFG, avoid=[cross])
    assert IND_CROSS not in {r.idx for r in picked}
    with pytest.raises(ValueError):
        sft.select_heldout(ind, 11, seed=SEED, tag="t", cfg=CFG, avoid=[cross])


def test_build_utility_heldout_records(pools):
    recs = sft.build_utility_heldout(pools, cfg=CFG)
    assert [r.pid for r in recs] == sorted(r.pid for r in recs)
    assert Counter(r.pid[:8] for r in recs) == {"utl-aya-": 2, "utl-ind-": 2}
    rows = {"utl-" + r.row_id(): r for name in ("aya", "indicalign") for r in pools[name][0]}
    for rec in recs:
        row = rows[rec.pid]
        assert (rec.lang, rec.set, rec.split, rec.text_en, rec.translation, rec.lid) == (
            "hi", "utility", "dev", None, None, None)
        assert (rec.text, rec.extra["reference"]) == (row.instruction, row.response)
        assert (rec.extra["hf_row"], rec.extra["src_idx"]) == (row.hf_row, row.idx)
        if row.source == "aya":
            assert rec.source == "aya-dataset" and set(rec.extra) == {"reference", "hf_row", "src_idx"}
        else:
            assert rec.source == "indicalign-dolly_t"
            assert (rec.extra["doc_id"], rec.extra["upstream_mt"]) == (row.meta["doc_id"], "IndicTrans2")
    for i, a in enumerate(recs):              # no near-duplicate pair, across sources too
        for b in recs[i + 1:]:
            assert not near(a.text, a.extra["reference"], b.text, b.extra["reference"])
    assert sft.build_utility_heldout(cfg=CFG, hub=SftHub(), lid=FakeLID()) == recs
    with pytest.raises(ValueError, match="sizes.utility"):
        sft.build_utility_heldout(pools, cfg=project_cfg(per_source=3))


# ---------------------------------------------------------------- near-duplicates and exclusion

ALPHA = string.ascii_lowercase + string.digits + "αβγδεζ"     # 42 distinct characters, 40 distinct 3-grams


def held(instruction, response):
    return SftRow("indicalign", 7, 7, instruction, response, {"doc_id": "doc-x"})


def cand(i, instruction, response):
    return SftRow("aya", i, i, instruction, response)


def test_near_duplicate_instruction_threshold_is_inclusive():
    a = ALPHA[:22]                                                   # 20 distinct 3-grams
    assert jaccard(char_ngrams(a), char_ngrams(a[:16])) == 0.7
    assert jaccard(char_ngrams(a), char_ngrams(a[:15])) == 0.65
    rows = [cand(0, a[:16], "पहला अलग उत्तर यहाँ लिखा गया है"), cand(1, a[:15], "दूसरा अलग उत्तर यहाँ लिखा गया है")]
    h = held(a, "कोई उत्तर जो किसी और से मेल नहीं खाता")
    assert sft.near_duplicates(rows, [h], cfg=CFG) == {"aya-00000": "utl-ind-00007"}
    loose = copy.deepcopy(CFG)
    loose["sources"]["sft"]["heldout"]["near_dup"]["instruction"] = 0.65
    assert sft.near_duplicates(rows, [h], cfg=loose) == {"aya-00000": "utl-ind-00007",
                                                         "aya-00001": "utl-ind-00007"}


def test_near_duplicate_response_threshold_and_minimum_length():
    r = ALPHA
    assert jaccard(char_ngrams(r), char_ngrams(r[:36])) == 0.85
    h = held("मूल प्रश्न का पाठ यहाँ है", r)
    rows = [cand(0, "बिल्कुल अलग पहला प्रश्न", r[:36]), cand(1, "बिल्कुल अलग दूसरा प्रश्न", r[:35])]
    assert sft.near_duplicates(rows, [h], cfg=CFG) == {"aya-00000": "utl-ind-00007"}
    # both responses need at least response_min_chars (20) characters
    assert sft.near_duplicates([cand(2, "अलग प्रश्न", r[:19])], [held("मूल प्रश्न", r[:19])], cfg=CFG) == {}
    assert sft.near_duplicates([cand(3, "अलग प्रश्न", r[:20])], [held("मूल प्रश्न", r[:20])], cfg=CFG) == {
        "aya-00003": "utl-ind-00007"}


@pytest.mark.parametrize("thresholds", [(0.7, 0.85), (0.3, 0.5), (0.95, 0.6), (1.0, 1.0)])
def test_near_duplicates_indexed_join_agrees_with_brute_force(thresholds):
    rng = random.Random(sum(thresholds))
    letters = "कखगघचछजझटठ"

    def text(lo, hi):
        return "".join(rng.choice(letters + " ") for _ in range(rng.randint(lo, hi)))

    def mutate(s):
        s = list(s)
        for _ in range(rng.randint(0, 4)):
            s[rng.randrange(len(s))] = rng.choice(letters)
        return "".join(s)

    base = [(text(4, 40), text(0, 70)) for _ in range(30)]
    held_rows = [SftRow("indicalign", i, i, ins, res, {"doc_id": f"d{i}"})
                 for i, (ins, res) in enumerate(base[:12])]
    rows = [cand(i, mutate(ins) if ins else ins, mutate(res) if res else res)
            for i, (ins, res) in enumerate(base)]
    rows += [cand(100 + i, text(0, 40), text(0, 70)) for i in range(30)]
    cfg = copy.deepcopy(CFG)
    nd = cfg["sources"]["sft"]["heldout"]["near_dup"]
    nd["instruction"], nd["response"] = thresholds
    expected = {}
    for row in rows:
        hits = [f"utl-ind-{h.idx:05d}" for h in held_rows
                if near(row.instruction, row.response, h.instruction, h.response, nd)]
        if hits:
            expected[row.row_id()] = min(hits)
    assert sft.near_duplicates(rows, held_rows, cfg=cfg) == expected
    assert expected


def test_heldout_ids_are_disjoint_from_both_training_pools(pools):
    recs = sft.build_utility_heldout(pools, cfg=CFG)
    train = sft.exclude_heldout(pools, recs, cfg=CFG)
    held_ids = {r.pid.removeprefix("utl-") for r in recs}
    for name, (rows, excluded) in train.items():
        ids = [r.row_id() for r in rows]
        assert ids == sorted(ids)
        splits.check_disjoint(held_ids, ids, f"held-out vs {name} training ids")
        for row in rows:
            assert not any(near(row.instruction, row.response, rec.text, rec.extra["reference"])
                           for rec in recs)
        removed = {r.row_id() for r in pools[name][0]} - set(ids)
        assert removed == {e["id"] for e in excluded}
        own = {e["id"] for e in excluded if e["reason"] == "heldout"}
        assert own == {i for i in held_ids if i.startswith(sft.TAGS[name] + "-")} and len(own) == 2
        assert all(e["reason"] == "heldout" or e["reason"].startswith("near_dup:utl-") for e in excluded)


def test_exclude_heldout_removes_near_duplicates_from_both_pools(pools):
    by_id = {r.row_id(): r for name in ("aya", "indicalign") for r in pools[name][0]}
    a, b = IND_NEAR_PAIR
    held_out = [sft.heldout_record(by_id[f"ind-{IND_CROSS:05d}"]), sft.heldout_record(by_id["aya-00003"]),
                sft.heldout_record(by_id[f"ind-{a:05d}"])]
    train = sft.exclude_heldout(pools, held_out, cfg=CFG)
    assert {e["id"]: e["reason"] for e in train["aya"][1]} == {
        "aya-00003": "heldout",
        f"aya-{AYA_NEAR_DUP:05d}": "near_dup:utl-aya-00003",              # same pool, by instruction
        f"aya-{AYA_CROSS:05d}": f"near_dup:utl-ind-{IND_CROSS:05d}"}      # the other source's held-out row
    assert {e["id"]: e["reason"] for e in train["indicalign"][1]} == {
        f"ind-{IND_CROSS:05d}": "heldout", f"ind-{a:05d}": "heldout",
        f"ind-{b:05d}": f"near_dup:utl-ind-{a:05d}"}                      # by response
    back = sft.exclude_heldout(pools, [sft.heldout_record(by_id[f"aya-{AYA_CROSS:05d}"])], cfg=CFG)
    assert back["indicalign"][1] == [{"id": f"ind-{IND_CROSS:05d}",
                                      "reason": f"near_dup:utl-aya-{AYA_CROSS:05d}"}]
    assert sft.exclude_heldout(pools, [by_id[f"aya-{AYA_CROSS:05d}"]], cfg=CFG) == back    # SftRows work too
    with pytest.raises(ValueError, match="not in the aya pool"):
        stray = sft.heldout_record(SftRow("aya", 999, 1, "कोई प्रश्न", "कोई उत्तर"))
        sft.exclude_heldout(pools, [stray], cfg=CFG)


# ---------------------------------------------------------------- candidates


def test_build_candidates_takes_the_target_when_both_pools_have_enough(data_dir, pools):
    curate = FakeCurate()
    report, train, _ = build(pools, curate=curate)
    assert report["decision"] == {"rows_target": 5, "rows": 5, "limited_by": None}
    assert [(p.relative_to(data_dir).as_posix(), d) for _, p, d in curate.calls] == [
        ("interim/sft/aya/dedup", True), ("interim/sft/indicalign/dedup", True),
        ("interim/sft/aya/raw", False)]
    assert [n for n, _, _ in curate.calls] == [len(train["aya"][0]), len(train["indicalign"][0]),
                                               len(train["aya"][0])]
    for key in VARIANTS:
        assert len(list(read_jsonl(data_dir / "sft" / key / "sft_messages.jsonl"))) == 5
        assert report["variants"][key]["rows"] == 5
    # the dedup run dropped aya's exact duplicate (idx 6 copies idx 1); the raw run kept it
    counts = {key: report["variants"][key]["counts"]["curated"] for key in VARIANTS}
    assert counts["aya/raw"] == counts["aya/dedup"] + 1 == len(train["aya"][0])


@pytest.mark.parametrize("small", ["aya", "indicalign"])
def test_build_candidates_equal_size_rule_follows_the_smaller_pool(data_dir, pools, caplog, small):
    heldout = sft.build_utility_heldout(pools, cfg=CFG)
    train = sft.exclude_heldout(pools, heldout, cfg=CFG)
    rows, excluded = train[small]
    distinct = list({dup_key(r.instruction): r for r in rows}.values())[:3]
    train[small] = (distinct, excluded)
    with caplog.at_level(logging.WARNING, logger="lift.data.sft"):
        report = sft.build_candidates(train, cfg=CFG, train_cfg=training_cfg(5), curate=FakeCurate())
    assert report["decision"] == {"rows_target": 5, "rows": 3, "limited_by": small}
    for key in VARIANTS:
        assert report["variants"][key]["rows"] == 3
        assert len(list(read_jsonl(data_dir / "sft" / key / "sft_messages.jsonl"))) == 3
    assert any("equal-size rule" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert report["sources"][small]["loaded"] is None          # no pool_counts given


def test_build_candidates_files_and_reports(data_dir, pools):
    report, train, heldout = build(pools)
    root = data_dir / "sft"
    for key in VARIANTS:
        name, variant = key.split("/")
        out = root / key
        assert b"\r" not in (out / "sft_messages.jsonl").read_bytes()
        rows = list(read_jsonl(out / "sft_messages.jsonl"))
        ids = [r["id"] for r in rows]
        assert ids == sorted(ids) and len(set(ids)) == 5
        by_id = {r.row_id(): r for r in train[name][0]}
        for r in rows:
            assert set(r) == {"id", "messages"}
            assert r["messages"] == [{"role": "user", "content": by_id[r["id"]].instruction},
                                     {"role": "assistant", "content": by_id[r["id"]].response}]
        # the N rows are the first N curated ids by sha256 rank, written in id order
        seen, curated = set(), []
        for i in sorted(by_id):
            text_key = (dup_key(by_id[i].instruction), dup_key(by_id[i].response))
            if not (variant == "dedup" and text_key in seen):
                seen.add(text_key)
                curated.append(i)
        assert len(curated) < len(by_id) or variant == "raw"      # each pool holds an exact duplicate
        tag = f"{SEED}:sft:{name}:{variant}"
        rank = sorted(curated, key=lambda i: hashlib.sha256(f"{tag}:{i}".encode()).hexdigest())
        assert ids == sorted(rank[:5])
        sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
        assert (sel["candidate"], sel["variant"], sel["dedup"]) == (name, variant, variant == "dedup")
        assert sel["tag"] == f"sft:{name}:{variant}"
        assert (sel["rows_target"], sel["rows"], sel["seed"], sel["ids"]) == (5, 5, SEED, ids)
        assert sel["counts"] == {"train_pool": len(train[name][0]), "curated": len(curated), "sampled": 5}
        assert sel["curatorkit"]["stage_counts"]["FakeDedup"]["input_count"] == len(train[name][0])
        assert sel["sha256"] == sha256_file(out / "sft_messages.jsonl")
        assert report["variants"][key] == {k: v for k, v in sel.items() if k != "ids"}
        for f in ("manifest.json", "rejected.jsonl", "lexsi_provenance.json", "curatorkit_summary.json"):
            assert b"\r" not in (out / "curatorkit" / f).read_bytes()      # FakeCurate wrote CRLF
        assert json.loads((out / "curatorkit" / "manifest.json").read_text(encoding="utf-8"))["placeholder"]
    assert json.loads((root / "report.json").read_text(encoding="utf-8")) == report
    held_ids = {r.pid.removeprefix("utl-") for r in heldout}
    for key in VARIANTS:
        sel = json.loads((root / key / "selection.json").read_text(encoding="utf-8"))
        assert held_ids.isdisjoint(sel["ids"])
    aya, ind = report["sources"]["aya"], report["sources"]["indicalign"]
    assert (aya["loaded"], aya["pool"], aya["held_out"]) == (18, 14, 2)
    assert aya["dropped"] == {"empty_instruction": 0, "empty_response": 1, "instruction_not_hi": 1,
                              "repetitive_response": 1, "response_not_hi": 1}
    assert (ind["loaded"], ind["pool"], ind["held_out"]) == (19, 14, 2)
    assert (ind["dropped"]["had_context"], ind["dropped"]["no_match"]) == (3, 1)
    for name, src in report["sources"].items():
        excluded = train[name][1]
        assert src["near_dup_excluded"] == sum(e["reason"].startswith("near_dup:") for e in excluded)
        assert sum(src["near_dup_of"].values()) == src["near_dup_excluded"]
        assert src["train_pool"] == len(train[name][0])
        assert src["train_pool"] == src["pool"] - src["held_out"] - src["near_dup_excluded"]


def test_build_candidates_rerun_is_byte_identical(data_dir):
    build(sft.load_pools(cfg=CFG, hub=SftHub(), lid=FakeLID()))
    first = snapshot(data_dir)
    build(sft.load_pools(cfg=CFG, hub=SftHub(), lid=FakeLID()))
    assert snapshot(data_dir) == first
    assert "sft/report.json" in first and "interim/sft/aya/raw/curatorkit/manifest.json" in first


def test_build_steps_never_log_row_text(data_dir, caplog):
    with caplog.at_level(logging.DEBUG):
        build(sft.load_pools(cfg=CFG, hub=SftHub(), lid=FakeLID()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "sft pool aya" in logged
    texts = [t for r in aya_dataset_rows() for t in (r["inputs"], r["targets"])]
    texts += [t for r in indicalign_rows() for t in r["hin_Deva"][0]]
    assert not [t for t in texts if len(t.strip()) >= 8 and t.strip() in logged]


def test_build_candidate_does_one_candidate_end_to_end(data_dir, pools):
    out = sft.build_candidate("aya", dedup=False, rows=4, cfg=CFG, train_cfg=training_cfg(),
                              curate=FakeCurate(), pools=pools)
    assert out == data_dir / "sft" / "aya" / "raw"
    ids = [r["id"] for r in read_jsonl(out / "sft_messages.jsonl")]
    assert len(ids) == 4
    held_ids = {r.pid.removeprefix("utl-") for r in sft.build_utility_heldout(pools, cfg=CFG)}
    assert held_ids.isdisjoint(ids)
    big = sft.build_candidate("indicalign", rows=1000, cfg=CFG, train_cfg=training_cfg(), curate=FakeCurate(),
                              pools=pools)
    sel = json.loads((big / "selection.json").read_text(encoding="utf-8"))
    assert sel["rows"] == sel["counts"]["curated"] < 1000 and sel["rows_target"] == 1000
    assert not (data_dir / "sft" / "report.json").exists()
    with pytest.raises(ValueError, match="unknown candidate"):
        sft.build_candidate("dolly", cfg=CFG, train_cfg=training_cfg(), curate=FakeCurate(), pools=pools)


# ---------------------------------------------------------------- the real CuratorKIT


def test_run_curatorkit_with_the_real_toolkit(data_dir, tmp_path):
    pytest.importorskip("curatorkit")
    pool = sft.clean_pool(sft.aya_rows(aya_dataset_rows()), FakeLID(), cfg=CFG)[0]       # 14 rows
    rows = [{"id": r.row_id(), "instruction": r.instruction, "output": r.response} for r in pool]
    first = rows[0]
    rows.append({"id": "aya-00900", "instruction": first["instruction"],
                 "output": first["output"][:-1] + "!"})
    joined = [char_ngrams((r["instruction"] + r["output"]).lower()) for r in (first, rows[-1])]
    assert 0.95 <= jaccard(*joined) < 1                                    # a near, not exact, duplicate
    dup, short = f"aya-{AYA_DUPLICATE:05d}", f"aya-{AYA_SHORT:05d}"
    cwd = Path.cwd()
    out_dir = data_dir / "interim" / "sft" / "probe" / "dedup"
    passed = sft.run_curatorkit(rows, out_dir, dedup=True)
    assert Path.cwd() == cwd
    ids = [r["id"] for r in passed]
    assert ids == [r["id"] for r in rows if r["id"] not in {dup, short, "aya-00900"}]
    src = {r["id"]: r for r in rows}
    for r in passed:
        assert r["messages"] == [
            {"role": "user", "content": unicodedata.normalize("NFC", src[r["id"]]["instruction"])},
            {"role": "assistant", "content": unicodedata.normalize("NFC", src[r["id"]]["output"])}]
    summary = json.loads((out_dir / "curatorkit_summary.json").read_text(encoding="utf-8"))
    assert summary["rejected"] == {short: "below_min_tokens:3"}
    assert summary["deduplicated"] == [dup, "aya-00900"]
    assert (summary["input"], summary["passed"], summary["dedup"]) == (15, 12, "minhash")
    assert summary["dataset"] == "interim/sft/probe/dedup/input.jsonl"
    assert "MessagesExporter" in summary["stage_counts"]
    for f in ("input.jsonl", "curatorkit_summary.json"):
        assert b"\r" not in (out_dir / f).read_bytes()
    assert [r["id"] for r in read_jsonl(out_dir / "input.jsonl")] == [r["id"] for r in rows]
    manifest = (out_dir / "curatorkit" / "manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest and tmp_path.as_posix() not in manifest
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:(\\\\|/(?!/))", manifest)  # a drive, not a URL
    assert json.loads(manifest)["provenance"]["inputs"][0]["ref"] == "interim/sft/probe/dedup/input.jsonl"
    raw = sft.run_curatorkit(rows, data_dir / "interim" / "sft" / "probe" / "raw", dedup=False)
    assert [r["id"] for r in raw] == [r["id"] for r in rows if r["id"] != short]
    (out_dir / "curatorkit" / "stale.jsonl").write_text("{}\n", encoding="utf-8")
    assert sft.run_curatorkit(rows, out_dir, dedup=True) == passed
    assert not (out_dir / "curatorkit" / "stale.jsonl").exists()
    with pytest.raises(ValueError, match="data dir"):
        sft.run_curatorkit(rows, tmp_path / "elsewhere", dedup=True)


def test_importing_sft_pulls_in_no_heavy_package():
    heavy = ("curatorkit", "datasets", "huggingface_hub", "fasttext", "torch", "transformers",
             "deep_translator")
    code = f"import sys, lift.data.sft; print(sorted(m for m in {heavy!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
