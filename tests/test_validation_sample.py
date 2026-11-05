"""Drawing the validation rounds and their keys (P04, Build 1): lift.validation.sample.

Every world is fake (tests/validation_fakes.py): neutral placeholder text, no model, no network,
all under tmp_path, with LIFT_DATA_DIR, LIFT_OUTPUT_DIR and LIFT_VALIDATION_DIR pointed there.
Round 1 needs at least 34 dev prompts per set, since one person reads hi and en (20 + 14 harmful).
"""

from __future__ import annotations

import copy
from collections import Counter
from datetime import datetime

import pytest
from validation_fakes import (
    DEV_VARIANTS,
    build_world,
    fill_sheet,
    make_generation,
    pid_of,
    prompt_text,
    read_csv,
    use_world,
    write_generations,
)

from lift import config, manifest
from lift.cli import main
from lift.io import read_jsonl, sha256_text, write_jsonl
from lift.schemas import PromptRecord, TranslationInfo
from lift.validation import sample
from lift.validation.labels import import_labels
from lift.validation.outputs import REPAIR_METHODS
from lift.validation.sample import (
    DEFAULT_SEED,
    ROUNDS,
    SHEET_COLUMNS,
    allocate,
    build_round,
    check_round,
    load_round,
    make_sheets,
)
from lift.validation.store import (
    GUIDELINES,
    ValidateError,
    keys_dir,
    labels_dir,
    round_keys_path,
    sheet_file,
    sheets_dir,
)

ROUND1_VARIANTS = ("stock", "adapter-aya-std-s1")
STD_ADAPTERS = ["adapter-aya-std-s1", "adapter-indicalign-std-s1"]        # both seed 1, standard recipe
STRESS = "adapter-aya-stress-s1"
REPAIRS = ("wiseft-l0.50-adapter-aya-std-s1", "cast-m1.0-adapter-aya-std-s1")
STRATA1 = ["stock/harmful", "adapter/harmful", "stock/hard_benign", "adapter/hard_benign",
           "stock/easy_benign", "adapter/easy_benign"]
STRATA2 = ["repaired/harmful", "adapter/harmful", "stock/harmful",
           "repaired/hard_benign", "adapter/hard_benign", "stock/hard_benign",
           "repaired/easy_benign", "adapter/easy_benign", "stock/easy_benign"]
READERS = (("hi", "en"), ("zh",), ("es",))       # one person labels both hi and en
NEW_SEED = DEFAULT_SEED + 1
MISSING = object()
_REAL_LOAD = config.load


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    # make_sheets stamps the keys with the git sha; these tests need no git subprocess.
    monkeypatch.setattr(manifest, "git_sha", lambda: "test")


def round1_world(tmp_path, monkeypatch, *, n_dev=40, dev_variants=ROUND1_VARIANTS, name="world"):
    """Dev outputs only, which is all round 1 draws from."""
    world = build_world(tmp_path / name, n_dev=n_dev, n_final=0, dev_variants=dev_variants,
                        final_variants=(), judgements=False, utility=False)
    return use_world(monkeypatch, world)


def two_round_world(tmp_path, monkeypatch):
    """Stock, the seed-1 adapter and two repairs on dev; stock, that adapter and one repair on final."""
    world = build_world(tmp_path / "world", dev_variants=(*ROUND1_VARIANTS, *REPAIRS),
                        final_variants=(*ROUND1_VARIANTS, REPAIRS[0]), judgements=False, utility=False)
    return use_world(monkeypatch, world)


def to_validation_dir(monkeypatch, path):
    """Draw into a fresh validation tree, as another machine would."""
    monkeypatch.setenv("LIFT_VALIDATION_DIR", str(path))


def items(doc, **match):
    return [it for it in doc["items"] if all(it[k] == v for k, v in match.items())]


def pids(doc, **match):
    return {it["pid"] for it in items(doc, **match)}


def drawn(doc, strata):
    """{lang: {group/set: items}} on the round's main sheets."""
    n = Counter((it["lang"], f"{it['group']}/{it['set']}") for it in items(doc, slot="a"))
    return {lang: {s: n[lang, s] for s in strata} for lang in doc["allocation"]}


def file_entry(doc, variant, set_name, lang, split="dev"):
    (entry,) = [f for f in doc["files"]
                if (f["variant"], f["set"], f["lang"], f["split"]) == (variant, set_name, lang, split)]
    return entry


def sheet_rows(sheet):
    """{item_id: row} of one sheet, in row order."""
    rows = read_csv(sheets_dir() / sheet_file(sheet))
    assert rows[1] == list(SHEET_COLUMNS)
    return {row[0]: row for row in rows[2:]}


def sheet_bytes():
    return {p.name: p.read_bytes() for p in sorted(sheets_dir().glob("*.csv"))}


def rewrite(world, variant, set_name, lang, change, split="dev"):
    """Rewrite one generations file as change(its records)."""
    recs = list(world.generations[(variant, set_name, lang, split)])
    write_generations(world, variant, set_name, lang, split, change(recs))


def rewrite_prompts(world, set_name, lang, change):
    path = world.data / "prompts" / f"{set_name}_{lang}.jsonl"
    prompts = [PromptRecord.model_validate(row) for row in read_jsonl(path)]
    write_jsonl(path, [change(p).model_dump() for p in prompts])


def stale(rec):
    """A record generated before its prompt's translation was fixed."""
    return rec.model_copy(update={"prompt_sha": sha256_text("the text before the fix")})


def mix_runs(recs):
    """A second run's record for one pid, written into the same file."""
    again = recs[3].model_copy(update={"key": sha256_text("run2:" + recs[3].key),
                                       "meta": {"dtype": "bf16", "run": 2}})
    return [*recs, again]


def patch_project(monkeypatch, dotted, value):
    """configs/project.yaml with one value changed, or removed when value is MISSING."""
    project = copy.deepcopy(_REAL_LOAD("project"))
    *parents, last = dotted.split(".")
    node = project
    for part in parents:
        node = node.setdefault(part, {})
    if value is MISSING:
        node.pop(last, None)
    else:
        node[last] = value
    monkeypatch.setattr(config, "load",
                        lambda name="project": project if name == "project" else _REAL_LOAD(name))


# --- the plan --------------------------------------------------------------------------------------


def test_the_two_rounds_add_up_to_the_configured_totals():
    assert ROUNDS[1].sizes == {"hi": 60, "zh": 40, "es": 40, "en": 40}
    assert ROUNDS[2].sizes == {"hi": 60, "zh": 20, "es": 20, "en": 20}
    per_language = config.get("validation.per_language")
    assert set(ROUNDS[1].sizes) | set(ROUNDS[2].sizes) == set(per_language)
    assert {lang: ROUNDS[1].sizes[lang] + ROUNDS[2].sizes[lang] for lang in per_language} == per_language


LOCKED_TOTALS = {"hi": 120, "zh": 60, "es": 60, "en": 60}            # README, Judging


@pytest.mark.parametrize("per_language", [{**LOCKED_TOTALS, "hi": 100}, {"hi": 120, "zh": 60, "es": 60},
                                          {**LOCKED_TOTALS, "ar": 60}, MISSING],
                         ids=["a count changed", "a language dropped", "a language added", "missing"])
def test_a_round_is_drawn_only_while_the_rounds_add_up_to_the_configured_totals(tmp_path, monkeypatch,
                                                                                 per_language):
    # R2-17: the round sizes are ROUNDS, and validation.per_language is what they add up to. An edit to one
    # of them alone (a count, or a language dropped, which the static test above cannot see) is refused
    # wherever a round is drawn or checked, naming both, before anything is read or written.
    round1_world(tmp_path, monkeypatch)
    patch_project(monkeypatch, "validation.per_language", per_language)
    for draw in (make_sheets, check_round, build_round):
        with pytest.raises(ValidateError, match=r"validation\.per_language") as refused:
            draw(1)
        assert str(LOCKED_TOTALS) in str(refused.value)                 # what the rounds add up to
        assert ("has no" if per_language is MISSING else str(per_language)) in str(refused.value)
    assert not keys_dir().exists() and not sheets_dir().exists()


def test_allocate_splits_evenly_with_the_remainder_on_harmful_then_hard_benign():
    strata1, strata2 = ROUNDS[1].strata(), ROUNDS[2].strata()
    assert [f"{g}/{s}" for g, s in strata1] == STRATA1
    assert [f"{g}/{s}" for g, s in strata2] == STRATA2
    assert list(allocate(60, strata1).values()) == [10] * 6
    assert list(allocate(40, strata1).values()) == [7, 7, 7, 7, 6, 6]
    assert list(allocate(60, strata2).values()) == [7] * 6 + [6] * 3
    assert list(allocate(20, strata2).values()) == [3, 3] + [2] * 7
    for n in range(30):
        counts = allocate(n, strata2)
        assert list(counts) == strata2 and sum(counts.values()) == n
        assert max(counts.values()) - min(counts.values()) <= 1


# --- round 1 ---------------------------------------------------------------------------------------


def test_round1_fills_every_stratum_from_stock_and_seed1_standard_adapters_on_dev(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch, dev_variants=(*DEV_VARIANTS, STRESS))
    paths = make_sheets(1)
    doc = load_round(1)

    expected = {"hi": dict.fromkeys(STRATA1, 10),
                **{lang: dict(zip(STRATA1, [7, 7, 7, 7, 6, 6], strict=True)) for lang in ("zh", "es", "en")}}
    assert doc["allocation"] == expected
    assert all(list(doc["allocation"][lang]) == STRATA1 for lang in expected)
    assert drawn(doc, STRATA1) == expected
    sizes = {"round1_hi": 60, "round1_zh": 40, "round1_es": 40, "round1_en": 40}
    assert {name: info["n"] for name, info in doc["sheets"].items()} == sizes
    assert round_keys_path(1).resolve() == (world.validation / "keys" / "round1.json").resolve()
    assert sorted(p.resolve() for p in paths) == sorted(
        (world.validation / "sheets" / sheet_file(name)).resolve() for name in sizes)
    for name, n in sizes.items():
        rows = read_csv(sheets_dir() / sheet_file(name))
        assert GUIDELINES in rows[0][0]
        assert len(rows) == n + 2          # the guidelines line, the column names, one row per item

    # Dev outputs only; stock, and the seed-1 standard-recipe adapters (not seed 2, not the stress
    # recipe, not the repairs).
    assert {it["split"] for it in doc["items"]} == {"dev"}
    assert {it["variant"] for it in items(doc, group="stock")} == {"stock"}
    assert {it["variant"] for it in items(doc, group="adapter")} <= set(STD_ADAPTERS)
    assert {(it["method"], it["seed"]) for it in items(doc, group="adapter")} == {("adapter", 1)}
    assert doc["variants"] == {"stock": ["stock"], "adapter": STD_ADAPTERS}
    assert set(doc["ignored_variants"]) == {"adapter-aya-std-s2", STRESS, *REPAIRS}
    # Each item maps to the generation it shows: key, variant, set, language, split and pid.
    by_key = world.by_key()
    for it in doc["items"]:
        gen = by_key[it["key"]]
        assert (gen.variant, gen.set, gen.lang, gen.split, gen.pid) == (
            it["variant"], it["set"], it["lang"], it["split"], it["pid"])
        assert it["sheet"] == f"round1_{it['lang']}"


def test_round1_takes_another_adapter_only_when_it_is_named(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch,
                 dev_variants=("stock", "adapter-aya-std-s1", STRESS, "adapter-aya-std-s2"))
    make_sheets(1, variants=["stock", STRESS])
    doc = load_round(1)
    assert doc["variants"] == {"stock": ["stock"], "adapter": [STRESS]}
    assert {it["variant"] for it in items(doc, group="adapter")} == {STRESS}
    assert doc["variants_named"] == sorted(["stock", STRESS])
    # Naming a variant lifts the recipe filter, not the seed: round 1 is the seed-1 adapters.
    with pytest.raises(ValidateError, match="adapter-aya-std-s2 fit no group"):
        build_round(1, variants=["stock", "adapter-aya-std-s2"])


# --- round 2 ---------------------------------------------------------------------------------------


def test_round2_draws_repairs_on_dev_and_final_and_never_repeats_a_round1_prompt(tmp_path, monkeypatch):
    two_round_world(tmp_path, monkeypatch)
    with pytest.raises(ValidateError, match="make round 1 first"):
        make_sheets(2)
    assert any("make round 1 first" in p for p in check_round(2)["problems"])
    make_sheets(1)
    make_sheets(2)
    r1, r2 = load_round(1), load_round(2)

    expected = {"hi": dict(zip(STRATA2, [7] * 6 + [6] * 3, strict=True)),
                **{lang: dict(zip(STRATA2, [3, 3] + [2] * 7, strict=True)) for lang in ("zh", "es", "en")}}
    assert r2["allocation"] == expected
    assert drawn(r2, STRATA2) == expected
    assert {name: info["n"] for name, info in r2["sheets"].items()} == {
        "round2_hi": 60, "round2_zh": 20, "round2_es": 20, "round2_en": 20}

    # Repairs from their dev and final outputs; the adapters and stock from final outputs only.
    assert r2["variants"] == {"repaired": sorted(REPAIRS), "adapter": ["adapter-aya-std-s1"],
                              "stock": ["stock"]}
    repaired = items(r2, group="repaired")
    assert {it["method"] for it in repaired} <= set(REPAIR_METHODS)
    assert {it["split"] for it in repaired} == {"dev", "final"}
    assert {(it["method"], it["split"]) for it in items(r2, group="adapter")} == {("adapter", "final")}
    assert {(it["variant"], it["split"]) for it in items(r2, group="stock")} == {("stock", "final")}
    pools = r2["pools"]["hi"]
    assert pools["repaired/harmful"] == 3 * 40        # wiseft on dev and final, cast on dev only
    assert pools["adapter/harmful"] == pools["stock/harmful"] == 40

    # No generation is labelled twice, and no reader sees a prompt twice: not in a later round, and
    # not across the two languages one person reads.
    assert not {it["key"] for it in r1["items"]} & {it["key"] for it in r2["items"]}
    for doc in (r1, r2):
        assert not pids(doc, lang="hi") & pids(doc, lang="en")
    for reader in READERS:
        seen = [it["pid"] for doc in (r1, r2) for it in doc["items"] if it["lang"] in reader]
        assert len(seen) == len(set(seen)), reader
    assert r2["excluded_seen_pids"] == {"hi": 100, "zh": 40, "es": 40, "en": 100}


# --- reproducibility and never twice ------------------------------------------------------------------


def test_the_draw_is_the_same_in_any_validation_dir_and_any_record_order(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    make_sheets(1)
    first = (load_round(1)["items"], sheet_bytes())
    assert len(first[1]) == 4

    to_validation_dir(monkeypatch, tmp_path / "again")
    make_sheets(1)
    assert (load_round(1)["items"], sheet_bytes()) == first

    # The same outputs with every generations file written in the reverse order.
    for (variant, set_name, lang, split), recs in list(world.generations.items()):
        write_generations(world, variant, set_name, lang, split, recs[::-1])
    to_validation_dir(monkeypatch, tmp_path / "reversed")
    make_sheets(1)
    assert (load_round(1)["items"], sheet_bytes()) == first


def test_a_different_seed_draws_a_different_sample(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    make_sheets(1)
    a, a_sheets = load_round(1), sheet_bytes()
    to_validation_dir(monkeypatch, tmp_path / "seed2")
    make_sheets(1, seed=NEW_SEED)
    b, b_sheets = load_round(1), sheet_bytes()
    assert (a["seed"], b["seed"]) == (DEFAULT_SEED, NEW_SEED)
    assert a["allocation"] == b["allocation"]
    assert {it["key"] for it in a["items"]} != {it["key"] for it in b["items"]}
    assert all(a_sheets[name] != b_sheets[name] for name in a_sheets)


def test_no_sheet_shows_a_prompt_twice_even_when_the_pools_run_short(tmp_path, monkeypatch):
    # 34 dev prompts per set is all round 1 needs for the hi/en reader: 20 + 14 harmful and hard benign.
    round1_world(tmp_path, monkeypatch, n_dev=34, dev_variants=("stock", *STD_ADAPTERS))
    make_sheets(1)
    doc = load_round(1)
    for name, info in doc["sheets"].items():
        on_sheet = items(doc, sheet=name)
        assert len({it["pid"] for it in on_sheet}) == len(on_sheet) == info["n"]
        assert set(sheet_rows(name)) == {it["item_id"] for it in on_sheet}
    ids = [it["item_id"] for it in doc["items"]]
    assert len(set(ids)) == len(ids)
    # Every pid has three outputs per language (stock and two adapters), yet the hi/en reader gets
    # each harmful and hard benign prompt exactly once.
    for set_name in ("harmful", "hard_benign"):
        shown = [it["pid"] for it in doc["items"] if it["lang"] in ("hi", "en") and it["set"] == set_name]
        assert sorted(shown) == [pid_of(set_name, i) for i in range(34)]


# --- what the pools hold ------------------------------------------------------------------------------


def test_empty_and_garbled_outputs_are_sampled(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    garbled = "zq" * 600
    rewrite(world, "stock", "harmful", "hi", lambda recs: [r.model_copy(update={"output": ""}) for r in recs])
    rewrite(world, "stock", "hard_benign", "hi",
            lambda recs: [r.model_copy(update={"output": garbled}) for r in recs])
    make_sheets(1)
    doc = load_round(1)
    rows = sheet_rows("round1_hi")
    empty = items(doc, lang="hi", group="stock", set="harmful")
    looping = items(doc, lang="hi", group="stock", set="hard_benign")
    assert len(empty) == len(looping) == 10
    assert all(rows[it["item_id"]][2] == "" for it in empty)
    assert all(rows[it["item_id"]][2] == garbled for it in looping)     # in full, never cut


def test_a_pid_with_only_stale_outputs_is_left_out_and_counted(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    make_sheets(1)
    pid = sorted(pids(load_round(1), lang="hi", set="harmful"))[0]
    # A native reader fixes that Hindi prompt after the runs, which leaves its outputs stale.
    fixed = PromptRecord(pid=pid, lang="hi", set="harmful", split="dev", source="fake",
                         text=prompt_text("harmful", "hi", pid, version=1))
    rewrite_prompts(world, "harmful", "hi", lambda p: fixed if p.pid == pid else p)
    old_keys = {r.key for v in ROUND1_VARIANTS for r in world.generations[(v, "harmful", "hi", "dev")]
                if r.pid == pid}
    to_validation_dir(monkeypatch, tmp_path / "after_fix")
    make_sheets(1)
    doc = load_round(1)
    assert pid not in pids(doc, lang="hi")
    for variant in ROUND1_VARIANTS:
        f = file_entry(doc, variant, "harmful", "hi")
        assert (f["current"], f["expected"], f["stale_pids"], f["missing"]) == (39, 40, 1, 0)
    assert sum(f["stale_pids"] for f in doc["files"]) == len(ROUND1_VARIANTS)

    # The runs resume and write the fixed prompt's outputs under new keys: the pid is current again,
    # and its old outputs are never drawn.
    for variant in ROUND1_VARIANTS:
        rewrite(world, variant, "harmful", "hi", lambda recs, v=variant: [*recs, make_generation(v, fixed)])
    to_validation_dir(monkeypatch, tmp_path / "resumed")
    make_sheets(1)
    doc = load_round(1)
    for variant in ROUND1_VARIANTS:
        f = file_entry(doc, variant, "harmful", "hi")
        assert (f["current"], f["stale_pids"], f["missing"]) == (40, 0, 0)
    assert not old_keys & {it["key"] for it in doc["items"]}


def test_a_file_whose_records_all_miss_the_prompt_text_stops_the_draw(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    rewrite(world, "stock", "harmful", "hi", lambda recs: [stale(r) for r in recs])
    with pytest.raises(ValidateError, match="stock/harmful_hi_dev: none of its 40 records"):
        make_sheets(1)
    assert not world.validation.exists()


def test_an_unfinished_run_stops_the_draw(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    rewrite(world, "adapter-aya-std-s1", "easy_benign", "zh", lambda recs: recs[:-1])
    with pytest.raises(ValidateError, match="adapter-aya-std-s1/easy_benign_zh_dev: 1 of 40 prompts have no"):
        make_sheets(1)
    assert not world.validation.exists()


def test_records_in_another_precision_and_repeated_records_are_ignored(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    spellings = ("fp16", "float16", "torch.float16")

    def add_fp16_run(recs):
        return recs + [r.model_copy(update={"key": sha256_text("fp16:" + r.key),
                                            "meta": {"dtype": spellings[i % 3]}})
                       for i, r in enumerate(recs)]

    rewrite(world, "stock", "harmful", "hi", add_fp16_run)
    rewrite(world, "adapter-aya-std-s1", "harmful", "hi", lambda recs: recs + recs[:5])   # resumed twice
    make_sheets(1)
    doc = load_round(1)
    f = file_entry(doc, "stock", "harmful", "hi")
    assert (f["current"], f["other_dtype"], f["missing"]) == (40, 40, 0)
    f = file_entry(doc, "adapter-aya-std-s1", "harmful", "hi")
    assert (f["current"], f["duplicates"], f["missing"]) == (40, 5, 0)
    fp16 = {r.key for r in world.generations[("stock", "harmful", "hi", "dev")] if r.meta["dtype"] != "bf16"}
    assert len(fp16) == 40 and not fp16 & {it["key"] for it in doc["items"]}
    assert len(items(doc, lang="hi", group="stock", set="harmful")) == 10
    assert {it["dtype"] for it in doc["items"]} == {"bf16"}


def test_two_current_outputs_for_one_pid_stop_the_draw(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    rewrite(world, "adapter-aya-std-s1", "hard_benign", "es", mix_runs)
    with pytest.raises(ValidateError, match="two current generations"):
        make_sheets(1)
    assert not world.validation.exists()


def test_prompts_a_reader_checked_as_translations_are_never_drawn_for_that_reader(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    make_sheets(1)
    base = load_round(1)
    # The hi/en reader checked the Hindi translations of prompts they would have got on either of
    # their sheets; the zh reader checked Chinese ones. Every other prompt is unchecked.
    hi_checked = {*sorted(pids(base, lang="hi", set="harmful"))[:3],
                  *sorted(pids(base, lang="en", set="harmful"))[:2]}
    zh_checked = set(sorted(pids(base, lang="zh", set="harmful"))[:3])
    for lang, chosen in (("hi", hi_checked), ("zh", zh_checked)):
        rewrite_prompts(world, "harmful", lang, lambda p, chosen=chosen: p.model_copy(update={
            "translation": TranslationInfo(engine="google", checked=p.pid in chosen)}))
    to_validation_dir(monkeypatch, tmp_path / "after_checks")
    make_sheets(1)
    doc = load_round(1)
    assert not hi_checked & (pids(doc, lang="hi") | pids(doc, lang="en"))
    assert not zh_checked & pids(doc, lang="zh")
    assert doc["excluded_seen_pids"] == {"hi": 5, "zh": 3, "es": 0, "en": 5}


# --- drawing again -----------------------------------------------------------------------------------


def test_rerunning_with_the_same_inputs_changes_nothing_and_only_writes_missing_sheets(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    paths = make_sheets(1)
    keys = round_keys_path(1).read_bytes()
    hi, es = (sheets_dir() / sheet_file(f"round1_{lang}") for lang in ("hi", "es"))
    first = read_csv(hi)[2][0]
    fill_sheet(hi, hi, {first: "benign"}, notes={first: "typed by the reader"})
    typed, original_es = hi.read_bytes(), es.read_bytes()
    es.unlink()
    assert make_sheets(1) == paths
    assert hi.read_bytes() == typed               # the reader's work is still there
    assert es.read_bytes() == original_es         # a missing sheet comes back as it was
    assert round_keys_path(1).read_bytes() == keys


def test_a_different_sample_needs_force_and_force_archives_the_old_draw(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    make_sheets(1)
    old_keys, old_sheets = round_keys_path(1).read_bytes(), sheet_bytes()
    with pytest.raises(ValidateError, match="different round 1 sample"):
        make_sheets(1, seed=NEW_SEED)
    assert round_keys_path(1).read_bytes() == old_keys and sheet_bytes() == old_sheets

    make_sheets(1, seed=NEW_SEED, force=True)
    assert load_round(1)["seed"] == NEW_SEED
    new_sheets = sheet_bytes()
    assert set(new_sheets) == set(old_sheets)
    assert all(new_sheets[name] != old_sheets[name] for name in old_sheets)
    (archived_keys,) = (keys_dir() / "archive").glob("round1-*.json")
    assert archived_keys.read_bytes() == old_keys
    archived = {p.name.rsplit("-", 1)[0] + ".csv": p.read_bytes()      # validation_round1_hi-{stamp}.csv
                for p in (sheets_dir() / "archive").glob("*.csv")}
    assert archived == old_sheets


def test_force_refuses_while_the_round_has_imported_labels_or_typed_entries(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    make_sheets(1)
    old_keys, old_sheets = round_keys_path(1).read_bytes(), sheet_bytes()
    zh_id = items(load_round(1), lang="zh")[0]["item_id"]
    filled = fill_sheet(sheets_dir() / sheet_file("round1_zh"), tmp_path / "zh_back.csv", {zh_id: "benign"})
    import_labels(1, "zh", filled)
    with pytest.raises(ValidateError, match="imported labels"):
        make_sheets(1, seed=NEW_SEED, force=True)
    assert round_keys_path(1).read_bytes() == old_keys and sheet_bytes() == old_sheets

    # With the labels moved away, a label typed into a sheet still with its reader blocks it too.
    for path in labels_dir().glob("round1_*"):
        path.unlink()
    hi = sheets_dir() / sheet_file("round1_hi")
    fill_sheet(hi, hi, {read_csv(hi)[2][0]: "refusal"})
    typed = sheet_bytes()
    with pytest.raises(ValidateError, match="typed in"):
        make_sheets(1, seed=NEW_SEED, force=True)
    assert round_keys_path(1).read_bytes() == old_keys and sheet_bytes() == typed
    assert not (keys_dir() / "archive").exists() and not (sheets_dir() / "archive").exists()


def test_force_refuses_over_a_typed_in_overlap_sheet_the_new_draw_drops(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    make_sheets(1, hindi_overlap=True)
    overlap = sheets_dir() / sheet_file("round1_hi_b")
    fill_sheet(overlap, overlap, {read_csv(overlap)[2][0]: "benign"})
    with pytest.raises(ValidateError, match="typed in"):
        make_sheets(1, hindi_overlap=False, force=True)


class _FrozenClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 5, 9, 0, tzinfo=tz)


def test_force_keeps_every_replaced_draw_even_two_in_one_second(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    monkeypatch.setattr(sample, "datetime", _FrozenClock)
    make_sheets(1)
    first = round_keys_path(1).read_bytes()
    make_sheets(1, seed=NEW_SEED, force=True)
    second = round_keys_path(1).read_bytes()
    make_sheets(1, seed=NEW_SEED + 1, force=True)
    archived = sorted(p.read_bytes() for p in (keys_dir() / "archive").glob("round*.json"))
    assert len(archived) == 2
    assert archived == sorted([first, second])
    assert len(list((keys_dir() / "archive").glob("draw*.json"))) == 2     # each draw's metadata too


def test_redrawing_round1_after_round2_never_gives_a_reader_a_round2_prompt(tmp_path, monkeypatch):
    two_round_world(tmp_path, monkeypatch)
    make_sheets(1)
    make_sheets(2)
    try:
        make_sheets(1, seed=NEW_SEED, force=True)
    except ValidateError:
        return          # refusing to redraw a round that a later round was drawn against is fine too
    r1, r2 = load_round(1), load_round(2)
    for reader in READERS:
        seen = [it["pid"] for doc in (r1, r2) for it in doc["items"] if it["lang"] in reader]
        assert len(seen) == len(set(seen)), reader


# --- the Hindi overlap ---------------------------------------------------------------------------------


def test_hindi_overlap_repeats_20_hindi_items_under_their_own_ids(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    paths = make_sheets(1, hindi_overlap=True)
    doc = load_round(1)
    assert doc["hindi_overlap"] == 20
    assert doc["sheets"]["round1_hi_b"] == {"file": sheet_file("round1_hi_b"), "lang": "hi", "slot": "b",
                                            "n": 20}
    assert sheets_dir() / sheet_file("round1_hi_b") in paths
    main_items = {it["key"]: it for it in items(doc, sheet="round1_hi")}
    overlap = items(doc, sheet="round1_hi_b")
    assert len(main_items) == 60 and len({it["key"] for it in overlap}) == len(overlap) == 20
    assert {it["key"] for it in overlap} <= set(main_items)
    assert not {it["item_id"] for it in overlap} & {it["item_id"] for it in main_items.values()}
    rows, overlap_rows = sheet_rows("round1_hi"), sheet_rows("round1_hi_b")
    assert set(overlap_rows) == {it["item_id"] for it in overlap}
    for it in overlap:
        twin = main_items[it["key"]]
        assert (it["variant"], it["set"], it["pid"], it["slot"]) == (
            twin["variant"], twin["set"], twin["pid"], "b")
        assert overlap_rows[it["item_id"]][1:3] == rows[twin["item_id"]][1:3]    # same prompt and response
    assert drawn(doc, STRATA1)["hi"] == dict.fromkeys(STRATA1, 10)               # the main sheet is unchanged


def test_hindi_overlap_defaults_to_the_config_flag_else_off(tmp_path, monkeypatch):
    round1_world(tmp_path, monkeypatch)
    patch_project(monkeypatch, "validation.hindi_overlap", MISSING)
    make_sheets(1)
    assert "round1_hi_b" not in load_round(1)["sheets"] and load_round(1)["hindi_overlap"] == 0

    patch_project(monkeypatch, "validation.hindi_overlap", True)
    to_validation_dir(monkeypatch, tmp_path / "flag_on")
    make_sheets(1)
    assert load_round(1)["sheets"]["round1_hi_b"]["n"] == 20

    to_validation_dir(monkeypatch, tmp_path / "argument_wins")
    make_sheets(1, hindi_overlap=False)
    assert "round1_hi_b" not in load_round(1)["sheets"]


# --- the dry run ---------------------------------------------------------------------------------------


def test_check_round_reports_what_would_stop_the_draw_and_writes_nothing(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    ok = check_round(1)
    assert ok["problems"] == []
    assert ok["variants"] == {"stock": ["stock"], "adapter": ["adapter-aya-std-s1"]}
    assert len(ok["files"]) == 4 * 3 * 2           # languages x sets x variants
    rewrite(world, "stock", "harmful", "hi", lambda recs: recs[1:])
    rewrite(world, "adapter-aya-std-s1", "easy_benign", "zh", lambda recs: [stale(r) for r in recs])
    out = check_round(1)
    assert len(out["problems"]) == 2
    assert any("stock/harmful_hi_dev" in p and "unfinished" in p for p in out["problems"])
    assert any("adapter-aya-std-s1/easy_benign_zh_dev" in p and "none of its" in p for p in out["problems"])
    assert not world.validation.exists()


def test_check_round_reports_a_round_the_outputs_cannot_fill(tmp_path, monkeypatch):
    # 33 dev prompts per set: one short of the 20 + 14 harmful prompts the hi/en reader needs.
    world = round1_world(tmp_path, monkeypatch, n_dev=33)
    out = check_round(1)
    assert len(out["problems"]) == 1 and "not enough outputs" in out["problems"][0]
    with pytest.raises(ValidateError, match="not enough outputs"):
        make_sheets(1)
    assert not world.validation.exists()


def test_prompt_check_off_counts_stale_records_as_current(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    rewrite(world, "adapter-aya-std-s1", "easy_benign", "zh", lambda recs: [stale(r) for r in recs])
    strict = check_round(1)
    f = file_entry(strict, "adapter-aya-std-s1", "easy_benign", "zh")
    assert strict["problems"] and (f["current"], f["stale_pids"], f["missing"]) == (0, 40, 0)
    loose = check_round(1, prompt_check=False)
    f = file_entry(loose, "adapter-aya-std-s1", "easy_benign", "zh")
    assert loose["problems"] == [] and (f["current"], f["stale_pids"], f["missing"]) == (40, 0, 0)
    assert not world.validation.exists()
    make_sheets(1, prompt_check=False)
    assert load_round(1)["prompt_check"] is False


def test_check_round_reports_a_file_that_mixes_two_runs(tmp_path, monkeypatch):
    world = round1_world(tmp_path, monkeypatch)
    rewrite(world, "adapter-aya-std-s1", "hard_benign", "es", mix_runs)
    out = check_round(1)
    assert any("two current generations" in p for p in out["problems"])
    assert not world.validation.exists()


def test_cli_check_writes_nothing_and_a_different_sample_without_force_fails(tmp_path, monkeypatch, capsys):
    world = round1_world(tmp_path, monkeypatch)
    assert main(["validate", "sample", "--round", "1", "--check"]) == 0
    assert "drawable" in capsys.readouterr().out
    assert not world.validation.exists()
    assert main(["validate", "sample", "--round", "1"]) == 0
    capsys.readouterr()
    keys = round_keys_path(1).read_bytes()
    assert main(["validate", "sample", "--round", "1", "--seed", str(NEW_SEED)]) == 1
    assert "--force" in capsys.readouterr().out
    assert round_keys_path(1).read_bytes() == keys
