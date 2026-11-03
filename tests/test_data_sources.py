import copy
import json
from collections import Counter
from datetime import datetime

import pytest
from fake_sources import FakeHub, belebele_rows_by_config, fake_hub, orbench_rows, redteam_rows

from lift import config
from lift.data.sources import (
    HFHub,
    belebele_rows,
    largest_remainder,
    load_aya_redteaming_hindi,
    load_belebele,
    load_orbench_hard_sample,
    orbench_records,
    redteam_records,
    stratified_sample,
)
from lift.data.text import clean_text
from lift.io import sha256_text

CFG = config.load("project")
SEED = CFG["split_seed"]
CODES = CFG["languages"]["belebele"]
# Verified Nov 3 on the pinned OR-Bench-Hard-1K split, and the largest-remainder seats for 400.
ORBENCH_COUNTS = {"illegal": 527, "privacy": 199, "unethical": 125, "harmful": 106, "deception": 72,
                  "sexual": 66, "violence": 66, "self-harm": 63, "hate": 54, "harassment": 41}
ORBENCH_SEATS = {"deception": 22, "harassment": 13, "harmful": 32, "hate": 16, "illegal": 160,
                 "privacy": 60, "self-harm": 19, "sexual": 20, "unethical": 38, "violence": 20}
# sha256 of the comma-joined sorted pids of the real 400-row sample (seed 20261028, tag "orbench").
ORBENCH_SAMPLE_SHA = "318383bce39d51447e98eafd9e0b0602c7235a8c32894e53273c251eca6d8fb2"
CATS = [row["category"] for row in orbench_rows()]
BELEBELE_FIELDS = {"pid", "lang", "split", "link", "question_number", "flores_passage", "question",
                   "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4", "correct_answer_num", "dialect"}


def key(row):
    return row["link"], row["question_number"]


def test_largest_remainder_reproduces_the_verified_orbench_allocation():
    seats = largest_remainder(ORBENCH_COUNTS, 400)
    assert seats == ORBENCH_SEATS
    assert list(seats) == sorted(ORBENCH_COUNTS)
    # the floors sum to 396; the 4 extra seats go to the largest fractional parts
    assert sum(400 * n // 1319 for n in ORBENCH_COUNTS.values()) == 396


@pytest.mark.parametrize("total", [0, 1, 2, 3, 7, 12, 50, 100, 101, 400, 1318, 1319])
def test_largest_remainder_sums_exactly_within_one_seat_of_the_quota(total):
    for counts in ({"a": 1, "b": 1, "c": 1}, {"x": 5, "y": 3, "z": 0}, {"solo": 7},
                   {"a": 2, "b": 3, "c": 5, "d": 7, "e": 11}, dict(Counter(CATS)), ORBENCH_COUNTS):
        whole = sum(counts.values())
        if total > whole:
            continue
        seats = largest_remainder(counts, total)
        assert sum(seats.values()) == total
        for name, n in counts.items():
            assert 0 <= seats[name] <= n
            assert abs(seats[name] - total * n / whole) < 1


def test_largest_remainder_breaks_ties_by_larger_count_then_name():
    assert largest_remainder({"a": 7, "b": 2}, 3) == {"a": 2, "b": 1}      # larger fraction first
    assert largest_remainder({"a": 1, "b": 3}, 2) == {"a": 0, "b": 2}      # equal fractions: larger count
    assert largest_remainder({"c": 1, "b": 1, "a": 1}, 2) == {"a": 1, "b": 1, "c": 0}   # then name
    assert largest_remainder({"b": 2, "a": 2}, 1) == {"a": 1, "b": 0}


def test_largest_remainder_edge_cases():
    assert largest_remainder({"a": 0, "b": 4}, 2) == {"a": 0, "b": 2}
    assert largest_remainder({"a": 3, "b": 4}, 7) == {"a": 3, "b": 4}
    assert largest_remainder({}, 0) == {}
    assert largest_remainder({"a": 0}, 0) == {"a": 0}
    for counts, total in (({"a": 0}, 1), ({}, 1), ({"a": 2}, -1), ({"a": -1, "b": 3}, 1)):
        with pytest.raises(ValueError):
            largest_remainder(counts, total)


def test_stratified_sample_is_exact_sorted_and_per_category_exact():
    for n in range(len(CATS) + 1):
        picked = stratified_sample(CATS, n, SEED)
        assert len(set(picked)) == len(picked) == n
        assert picked == sorted(picked)
        assert Counter(CATS[i] for i in picked) == Counter(largest_remainder(Counter(CATS), n))
    assert Counter(CATS[i] for i in stratified_sample(CATS, 6, SEED)) == {"a": 3, "b": 2, "c": 1}
    for n in (-1, len(CATS) + 1):
        with pytest.raises(ValueError, match="cannot sample"):
            stratified_sample(CATS, n, SEED)


def test_stratified_sample_keeps_each_categorys_smallest_hash_ranks():
    picked = stratified_sample(CATS, 6, SEED, tag="orbench")
    for category, seats in {"a": 3, "b": 2, "c": 1}.items():
        rows = [i for i, c in enumerate(CATS) if c == category]
        want = sorted(rows, key=lambda i: sha256_text(f"{SEED}:orbench:{i}"))[:seats]
        assert [i for i in picked if CATS[i] == category] == sorted(want)


def test_stratified_sample_is_deterministic_and_keyed_by_seed_and_tag():
    cats = [f"c{i % 7}" for i in range(300)]
    picked = stratified_sample(cats, 100, SEED)
    assert picked == stratified_sample(list(cats), 100, SEED) == stratified_sample(tuple(cats), 100, SEED)
    assert picked != stratified_sample(cats, 100, SEED + 1)
    assert picked != stratified_sample(cats, 100, SEED, tag="other")


def test_stratified_sample_does_not_depend_on_the_order_categories_arrive_in():
    picked = stratified_sample(CATS, 6, SEED)
    # the same rows per category under names whose first-seen and sorted orders both differ
    renamed = [{"a": "z", "b": "m", "c": "a"}[c] for c in CATS]
    assert stratified_sample(renamed, 6, SEED) == picked
    # a category's picks depend only on its own rows: relabelling the others leaves them alone
    swapped = [{"a": "a", "b": "c", "c": "b"}[c] for c in CATS]
    again = stratified_sample(swapped, 6, SEED)
    assert [i for i in again if CATS[i] == "a"] == [i for i in picked if CATS[i] == "a"]


def test_orbench_records_fields_and_pids():
    rows = orbench_rows()
    records = orbench_records(rows, 6, SEED)
    assert [r.pid for r in records] == [f"orb-{i:04d}" for i in stratified_sample(CATS, 6, SEED)]
    assert [r.pid for r in records] == sorted(r.pid for r in records)
    assert Counter(r.category for r in records) == {"a": 3, "b": 2, "c": 1}
    for r in records:
        i = r.extra["hf_row"]
        assert r.pid == f"orb-{i:04d}"
        assert (r.lang, r.set, r.split, r.source) == ("en", "hard_benign", "dev", "orbench-hard-1k")
        assert r.text == r.text_en == clean_text(rows[i]["prompt"])
        assert r.category == rows[i]["category"]
        assert r.translation is None and r.lid is None and r.extra == {"hf_row": i}
    everything = {r.pid: r for r in orbench_records(rows, len(rows), SEED)}
    assert len(everything) == len(rows)
    assert everything["orb-0005"].text == "Placeholder request 5 about hiking"     # U+200B, ends stripped


def test_orbench_records_need_the_whole_split_in_order():
    with pytest.raises(ValueError, match="whole pinned split"):
        orbench_records(orbench_rows()[1:], 3, SEED)
    with pytest.raises(ValueError, match="whole pinned split"):
        orbench_records(list(reversed(orbench_rows())), 3, SEED)


def test_load_orbench_hard_sample_reads_n_and_seed_from_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["sources"]["orbench"]["n"] = 6
    hub = fake_hub()
    assert load_orbench_hard_sample(cfg=cfg, hub=hub) == orbench_records(orbench_rows(), 6, SEED)
    assert hub.calls == [("orbench", None)]
    assert len(load_orbench_hard_sample(4, cfg=cfg, hub=hub)) == 4
    cfg["split_seed"] = SEED + 1
    assert load_orbench_hard_sample(cfg=cfg, hub=hub) == orbench_records(orbench_rows(), 6, SEED + 1)


def test_redteam_records_fields_categories_and_duplicates():
    rows = redteam_rows()
    records, excluded = redteam_records(rows)
    assert [r.pid for r in records] == ["art-hi-0000", "art-hi-0001", "art-hi-0002", "art-hi-0004"]
    assert excluded == [{"pid": "art-hi-0003", "reason": "duplicate_of:art-hi-0001"}]
    for r in records:
        i = r.extra["hf_row"]
        labels = json.loads(rows[i]["harm_category"])
        assert (r.lang, r.set, r.split, r.source) == ("hi", "redteam", "final", "aya-redteaming")
        assert r.text_en is None and r.translation is None and r.lid is None
        assert r.text == clean_text(rows[i]["prompt"])
        assert r.category == "; ".join(labels)
        assert r.extra == {"harm_category": labels, "global_or_local": rows[i]["global_or_local"],
                           "hf_row": i}
    quirky = records[2]
    assert quirky.text == quirky.text.strip() and quirky.text.count("\n") == 1   # ends stripped, inner kept
    labels = ["Violence, Threats & Incitement", "Graphic material", "Self-Harm"]
    assert quirky.extra["harm_category"] == labels
    assert quirky.category == "Violence, Threats & Incitement; Graphic material; Self-Harm"


def test_redteam_duplicates_go_by_dup_key_and_keep_the_first_row():
    rows = redteam_rows()
    rows[4]["prompt"] = "  " + rows[0]["prompt"].replace(" ", "   ") + "​\n"
    records, excluded = redteam_records(list(reversed(rows)))       # input order does not matter
    assert [r.pid for r in records] == ["art-hi-0000", "art-hi-0001", "art-hi-0002"]
    assert excluded == [{"pid": "art-hi-0003", "reason": "duplicate_of:art-hi-0001"},
                        {"pid": "art-hi-0004", "reason": "duplicate_of:art-hi-0000"}]


@pytest.mark.parametrize("field, value, match", [
    ("harm_category", "Hate Speech", "not valid JSON"),
    ("harm_category", "[]", "non-empty list"),
    ("harm_category", '{"label": "Hate Speech"}', "non-empty list"),
    ("harm_category", '["Hate Speech", ""]', "non-empty list"),
    ("global_or_local", "regional", "global_or_local"),
    ("language", "English", "language"),
    ("prompt", " ​\n", "empty prompt"),
])
def test_redteam_records_reject_malformed_rows_without_quoting_them(field, value, match):
    rows = redteam_rows()
    rows[1][field] = value
    with pytest.raises(ValueError, match=match) as err:
        redteam_records(rows)
    assert str(err.value).startswith("art-hi-0001: ")
    assert value.strip() not in str(err.value)
    assert all(clean_text(row["prompt"]) not in str(err.value) for row in redteam_rows())


def test_load_aya_redteaming_hindi_logs_the_exclusions(caplog):
    hub = fake_hub()
    with caplog.at_level("INFO", logger="lift.data.sources"):
        records = load_aya_redteaming_hindi(cfg=CFG, hub=hub)
    assert records == redteam_records(redteam_rows())[0]
    assert hub.calls == [("aya_redteaming", None)]
    assert "art-hi-0003 excluded (duplicate_of:art-hi-0001)" in caplog.text
    assert all(clean_text(row["prompt"]) not in caplog.text for row in redteam_rows())


def test_belebele_rows_realign_the_permuted_hindi_rows():
    by_config = belebele_rows_by_config()
    en, raw = by_config["eng_Latn"], by_config["hin_Deva"]
    # the fixture really is permuted: a by-row reading gets keys and answers wrong
    assert [key(r) for r in raw] != [key(r) for r in en]
    assert [r["correct_answer_num"] for r in raw] != [r["correct_answer_num"] for r in en]
    out = belebele_rows(by_config, "hin_Deva", cfg=CFG)
    assert [r["pid"] for r in out] == [f"bel-{i:03d}" for i in range(5)]
    assert [key(r) for r in out] == [key(r) for r in en]
    assert [r["correct_answer_num"] for r in out] == [int(r["correct_answer_num"]) for r in en]
    assert out[3]["question"] == raw[4]["question"] and out[4]["question"] == raw[3]["question"]
    assert sum(key(a) != key(b) for a, b in zip(out, raw, strict=True)) == 2
    assert all(r["lang"] == "hi" and r["split"] is None and r["dialect"] == "hin_Deva" for r in out)
    assert all(set(r) == BELEBELE_FIELDS for r in out)                     # "ds" and "_row" dropped
    assert json.loads(json.dumps(out)) == out


@pytest.mark.parametrize("lang", sorted(CODES))
def test_belebele_rows_every_language(lang):
    by_config = belebele_rows_by_config()
    out = belebele_rows(by_config, CODES[lang], cfg=CFG)
    assert [r["lang"] for r in out] == [lang] * 5
    assert [r["correct_answer_num"] for r in out] == [2, 4, 1, 3, 1]
    assert all(set(r) == BELEBELE_FIELDS and r["dialect"] == CODES[lang] for r in out)
    if lang != "hi":
        assert [r["question"] for r in out] == [r["question"] for r in by_config[CODES[lang]]]


def test_belebele_rows_raise_on_repeated_keys():
    by_config = belebele_rows_by_config()
    by_config["zho_Hans"][1]["question_number"] = 1             # now repeats row 0's key
    with pytest.raises(ValueError, match=r"zho_Hans: .* repeats in 1 rows, e\.g\. rows \[1\]"):
        belebele_rows(by_config, "zho_Hans", cfg=CFG)
    by_config = belebele_rows_by_config()
    by_config["eng_Latn"][4]["question_number"] = 1
    with pytest.raises(ValueError, match="eng_Latn: .* repeats"):
        belebele_rows(by_config, "zho_Hans", cfg=CFG)


def test_belebele_rows_raise_when_the_key_sets_differ():
    by_config = belebele_rows_by_config()
    by_config["spa_Latn"][2]["link"] = "https://example.org/placeholder/z"
    with pytest.raises(ValueError, match="1 eng_Latn items missing, 1 items not in eng_Latn"):
        belebele_rows(by_config, "spa_Latn", cfg=CFG)
    by_config = belebele_rows_by_config()
    by_config["spa_Latn"].pop()
    with pytest.raises(ValueError, match="1 eng_Latn items missing, 0 items"):
        belebele_rows(by_config, "spa_Latn", cfg=CFG)


def test_belebele_rows_raise_on_answer_mismatch():
    by_config = belebele_rows_by_config()
    by_config["arb_Arab"][0]["correct_answer_num"] = "3"
    with pytest.raises(ValueError, match=r"differs from eng_Latn on 1 items, e\.g\. \['bel-000'\]"):
        belebele_rows(by_config, "arb_Arab", cfg=CFG)
    by_config = belebele_rows_by_config()                       # keys rewritten to the by-row reading
    for row, en_row in zip(by_config["hin_Deva"], by_config["eng_Latn"], strict=True):
        row["link"], row["question_number"] = key(en_row)
    with pytest.raises(ValueError, match="on 2 items"):
        belebele_rows(by_config, "hin_Deva", cfg=CFG)
    for bad in ("5", "0", "", "x", None):
        by_config = belebele_rows_by_config()
        by_config["tha_Thai"][0]["correct_answer_num"] = bad
        with pytest.raises(ValueError, match="not 1-4"):
            belebele_rows(by_config, "tha_Thai", cfg=CFG)


def test_belebele_rows_unknown_config_and_english_order():
    by_config = belebele_rows_by_config()
    with pytest.raises(ValueError, match="unknown Belebele config"):
        belebele_rows({**by_config, "hin_Latn": by_config["hin_Deva"]}, "hin_Latn", cfg=CFG)
    by_config["eng_Latn"] = by_config["eng_Latn"][::-1]
    with pytest.raises(ValueError, match="whole pinned split in order"):
        belebele_rows(by_config, "spa_Latn", cfg=CFG)


def test_load_belebele_goes_through_the_hub():
    hub = fake_hub()
    want = belebele_rows(belebele_rows_by_config(), "hin_Deva", cfg=CFG)
    assert load_belebele("hi", cfg=CFG, hub=hub) == want
    assert hub.calls == [("belebele", "en"), ("belebele", "hi")]
    hub.calls.clear()
    assert [r["lang"] for r in load_belebele("en", cfg=CFG, hub=hub)] == ["en"] * 5
    assert hub.calls == [("belebele", "en")]
    with pytest.raises(ValueError, match="unknown Belebele language"):
        load_belebele("fr", cfg=CFG, hub=hub)


# ---- HFHub with datasets.load_dataset and huggingface_hub.hf_hub_download monkeypatched ----

@pytest.fixture
def fake_load(monkeypatch):
    """Records load_dataset calls; ``tables`` maps the positional args to the rows to return."""
    datasets = pytest.importorskip("datasets")
    calls, tables = [], {}

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return datasets.Dataset.from_list(tables[args])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls, tables


def placeholder_rows(n, **fields):
    return [{name: f"{value} {i}" for name, value in fields.items()} for i in range(n)]


def test_hfhub_orbench_passes_config_split_and_revision(fake_load):
    calls, tables = fake_load
    spec = CFG["sources"]["orbench"]
    tables[spec["id"], spec["config"]] = placeholder_rows(3, prompt="Placeholder request", category="cat")
    rows = HFHub(CFG).rows("orbench")
    assert calls == [(("bench-llm/or-bench", "or-bench-hard-1k"),
                      {"split": "train", "revision": "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"})]
    assert rows == [{"prompt": f"Placeholder request {i}", "category": f"cat {i}", "_row": i}
                    for i in range(3)]


def test_hfhub_aya_redteaming_reads_the_hindi_split_of_the_default_config(fake_load):
    calls, tables = fake_load
    spec = CFG["sources"]["aya_redteaming"]
    upstream = [{k: v for k, v in row.items() if k != "_row"} for row in redteam_rows()]
    tables[(spec["id"],)] = upstream
    rows = HFHub(CFG).rows("aya_redteaming")
    assert calls == [(("CohereLabs/aya_redteaming",),
                      {"split": "hindi", "revision": "5a16fee03c19d681b4f3527b6a750e1ed4d434c7"})]
    assert rows == redteam_rows()


def test_hfhub_belebele_maps_our_code_to_the_config_and_drops_ds(fake_load):
    calls, tables = fake_load
    spec = CFG["sources"]["belebele"]
    fixture = belebele_rows_by_config()["hin_Deva"]
    # what datasets makes of the raw rows: "ds" parsed as a datetime, no "_row"
    upstream = [{**{k: v for k, v in row.items() if k != "_row"}, "ds": datetime(2023, 7, 21)}
                for row in fixture]
    tables[spec["id"], "hin_Deva"] = upstream
    rows = HFHub(CFG).rows("belebele", lang="hi")
    assert calls == [(("facebook/belebele", "hin_Deva"),
                      {"split": "test", "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"})]
    assert rows == [{k: v for k, v in row.items() if k != "ds"} for row in fixture]
    assert json.loads(json.dumps(rows)) == rows


def test_hfhub_aya_dataset_keeps_full_split_row_indices(fake_load):
    calls, tables = fake_load
    spec = CFG["sources"]["sft"]["aya"]
    languages = ["English", "Hindi", "Hindi", "Spanish", "Hindi", "Hindi Latin", "Hindi"]
    tables[(spec["id"],)] = [{"inputs": f"Placeholder input {i}", "targets": f"Placeholder target {i}",
                              "language": lang, "language_code": "xxx",
                              "annotation_type": "original-annotations", "user_id": f"user{i}"}
                             for i, lang in enumerate(languages)]
    rows = HFHub(CFG).rows("aya_dataset")
    assert calls == [(("CohereLabs/aya_dataset",),
                      {"split": "train", "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"})]
    assert [r["_row"] for r in rows] == [1, 2, 4, 6]
    assert [r["inputs"] for r in rows] == [f"Placeholder input {i}" for i in (1, 2, 4, 6)]
    tables[(spec["id"],)] = tables[(spec["id"],)][:1]
    with pytest.raises(ValueError, match="no rows with language 'Hindi'"):
        HFHub(CFG).rows("aya_dataset")


def test_hfhub_indicalign_loads_only_the_needed_columns(fake_load):
    calls, tables = fake_load
    spec = CFG["sources"]["sft"]["indicalign"]
    upstream = [{"doc_id": f"doc-{i}",
                 "eng_Latn": [[f"Placeholder instruction {i}", f"Placeholder response {i}"]],
                 "hin_Deva": [[f"नकली निर्देश {i}", f"नकली उत्तर {i}"]]} for i in range(3)]
    tables[spec["id"], spec["subset"]] = upstream
    rows = HFHub(CFG).rows("indicalign")
    assert calls == [(("ai4bharat/indic-align", "Dolly_T"),
                      {"split": "train", "revision": "032b6a9070e7f85f1a38e0506419f4590a20455a",
                       "columns": ["doc_id", "eng_Latn", "hin_Deva"]})]
    assert rows == [{**row, "_row": i} for i, row in enumerate(upstream)]
    # a loader that ignored the projection still yields only the three columns
    tables[spec["id"], spec["subset"]] = [{**row, "num_turns": 1.0, "hin_Latn": [["x", "y"]]}
                                          for row in upstream]
    assert HFHub(CFG).rows("indicalign") == rows


def test_hfhub_dolly_reads_the_pinned_jsonl_file(monkeypatch, tmp_path):
    hf_hub = pytest.importorskip("huggingface_hub")
    records = [{"instruction": f"Placeholder instruction {i}", "context": "", "response": f"Placeholder {i}",
                "category": "open_qa"} for i in range(3)]
    path = tmp_path / "databricks-dolly-15k.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8", newline="\n")
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)

    monkeypatch.setattr(hf_hub, "hf_hub_download", download)
    rows = HFHub(CFG).rows("dolly")
    assert calls == [(("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl"),
                      {"repo_type": "dataset", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"})]
    assert rows == [{**r, "_row": i} for i, r in enumerate(records)]


def test_hfhub_refuses_unpinned_revisions_and_bad_arguments(fake_load):
    calls, _ = fake_load
    for revision in ("main", None, "e36d8b80"):
        cfg = copy.deepcopy(CFG)
        cfg["sources"]["orbench"]["revision"] = revision
        with pytest.raises(ValueError, match="full commit sha"):
            HFHub(cfg).rows("orbench")
    hub = HFHub(CFG)
    with pytest.raises(ValueError, match="unknown Hub source"):
        hub.rows("or-bench-hard-1k")
    with pytest.raises(ValueError, match="needs lang"):
        hub.rows("belebele")
    with pytest.raises(ValueError, match="takes no lang"):
        hub.rows("orbench", lang="en")
    with pytest.raises(ValueError, match="unknown Belebele language"):
        hub.rows("belebele", lang="hin_Deva")
    assert calls == []


def test_hfhub_refuses_a_cached_copy_of_another_revision(monkeypatch, tmp_path):
    # Offline, datasets 5.1 serves the latest cached copy whatever revision is asked for (verified
    # Nov 3: revision "0" * 40 loaded the e36d8b80 cache). Its Arrow files live under <revision>/.
    datasets = pytest.importorskip("datasets")
    rows = [{"prompt": "Placeholder request 0", "category": "a"}]

    def cached(revision):
        path = tmp_path / "or-bench-hard-1k" / "0.0.0" / revision
        if not path.exists():
            datasets.Dataset.from_list(rows).save_to_disk(str(path))
        return lambda *args, **kwargs: datasets.load_from_disk(str(path))

    monkeypatch.setattr(datasets, "load_dataset", cached("0" * 40))
    with pytest.raises(RuntimeError, match="offline mode ignores the pin"):
        HFHub(CFG).rows("orbench")
    monkeypatch.setattr(datasets, "load_dataset", cached(CFG["sources"]["orbench"]["revision"]))
    assert HFHub(CFG).rows("orbench") == [{**rows[0], "_row": 0}]


def test_hfhub_rejects_values_that_are_not_json_serialisable(fake_load):
    _, tables = fake_load
    spec = CFG["sources"]["orbench"]
    tables[spec["id"], spec["config"]] = [{"prompt": "Placeholder", "category": "a",
                                           "when": datetime(2024, 1, 1)}]
    with pytest.raises(TypeError, match="orbench row 0: column 'when' holds a datetime"):
        HFHub(CFG).rows("orbench")


def test_fake_hub_returns_copies():
    hub = FakeHub({"orbench": orbench_rows()})
    hub.rows("orbench")[0]["prompt"] = "changed"
    assert hub.rows("orbench") == orbench_rows()
    with pytest.raises(KeyError):
        hub.rows("dolly")


# ---- the real sets (network, or a warm Hugging Face cache) ----

@pytest.mark.net
def test_real_orbench_sample_and_aya_redteaming():
    hub = HFHub(CFG)
    rows = hub.rows("orbench")
    assert len(rows) == 1319
    assert Counter(r["category"] for r in rows) == ORBENCH_COUNTS
    records = load_orbench_hard_sample(cfg=CFG, hub=FakeHub({"orbench": rows}))
    assert Counter(r.category for r in records) == ORBENCH_SEATS
    assert sha256_text(",".join(sorted(r.pid for r in records))) == ORBENCH_SAMPLE_SHA
    art = hub.rows("aya_redteaming")
    assert len(art) == 915
    records, excluded = redteam_records(art)
    assert len(records) == 913
    assert excluded == [{"pid": "art-hi-0032", "reason": "duplicate_of:art-hi-0002"},
                        {"pid": "art-hi-0465", "reason": "duplicate_of:art-hi-0441"}]


@pytest.mark.net
def test_real_belebele_realignment():
    hub = HFHub(CFG)
    en = hub.rows("belebele", lang="en")
    for lang, code in CODES.items():
        raw = en if lang == "en" else hub.rows("belebele", lang=lang)
        assert len(raw) == 900
        out = belebele_rows({CODES["en"]: en, code: raw}, code, cfg=CFG)
        assert [r["pid"] for r in out] == [f"bel-{i:03d}" for i in range(900)]
        moved = [i for i, (a, b) in enumerate(zip(out, raw, strict=True)) if key(a) != key(b)]
        assert len(moved) == (440 if lang == "hi" else 0)
        if moved:
            assert (moved[0], moved[-1]) == (460, 899)


@pytest.mark.net
def test_real_sft_sources():
    hub = HFHub(CFG)
    aya = hub.rows("aya_dataset")
    assert len(aya) == 1153 and {r["language"] for r in aya} == {"Hindi"}
    assert [r["_row"] for r in aya] == sorted({r["_row"] for r in aya}) and aya[-1]["_row"] > 1153
    indicalign = hub.rows("indicalign")
    assert len(indicalign) == 15011 and set(indicalign[0]) == {"doc_id", "eng_Latn", "hin_Deva", "_row"}
    assert len(hub.rows("dolly")) == 15011
