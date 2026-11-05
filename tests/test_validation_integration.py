"""Contracts with the other stages, and the review's fixes: the keys file P02 reads, P01's
spot-check sheets, outputs remade after a draw, Gate 1, the second judge's commands, and the
harness's judge ids. Fake worlds only (tests/validation_fakes.py), all under tmp_path.
"""

import copy
import json
import re
import shutil

import pytest
from validation_fakes import (
    FRONTIER_ID,
    PRIMARY_ID,
    build_world,
    fill_sheet,
    make_generation,
    other_label,
    truth,
    use_world,
    write_generations,
    write_judgements,
    write_prompts,
    write_verdicts,
)

from lift import paths
from lift.io import read_jsonl, sha256_text, write_jsonl
from lift.models.generate import HARNESS_VERSION
from lift.schemas import JudgementRecord, PromptRecord
from lift.validation import outputs
from lift.validation import score as scoring
from lift.validation.labels import import_labels
from lift.validation.outputs import Prompts, judge_kind, load_judgements, load_pool
from lift.validation.sample import check_round, load_round, make_sheets
from lift.validation.second_judge import second_judge_sample
from lift.validation.store import ValidateError, draw_path, read_json, round_keys_path, sheets_dir
from lift.validation.utility import import_preferences, make_utility_sheet, score_utility


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    import lift.manifest

    monkeypatch.setattr(lift.manifest, "git_sha", lambda: "test")


# Round 1 needs only stock and one seed-1 standard adapter; smaller worlds keep the file fast.
LEAN = {"dev_variants": ("stock", "adapter-aya-std-s1"), "final_variants": ("stock",), "utility": False}


def world_for(tmp_path, monkeypatch, **options):
    return use_world(monkeypatch, build_world(tmp_path, **{**LEAN, **options}))


def import_truth(world, round_no=1, leave=()):
    """Every round sheet back from its reader with the true labels, except the ids in ``leave``."""
    doc = load_round(round_no)
    for name, info in doc["sheets"].items():
        values = {it["item_id"]: truth(it["key"]) for it in doc["items"]
                  if it["sheet"] == name and it["item_id"] not in leave}
        filled = fill_sheet(sheets_dir() / info["file"], world.root / f"filled_{name}.csv", values)
        import_labels(round_no, info["lang"], filled)


def drop_judgements(world, keys):
    for path in (world.outputs / "judgements").rglob("*.jsonl"):
        rows = [r for r in read_jsonl(path) if r["key"] not in keys]
        write_jsonl(path, rows)


def test_round_keys_are_the_item_id_map_that_p02_reads(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    raw = read_json(round_keys_path(1))
    assert isinstance(raw, dict) and len(raw) == 180
    assert all(re.fullmatch(r"[0-9a-f]{64}", rec["key"]) and rec["variant"] and rec["set"]
               for rec in raw.values())
    assert not {"round", "items", "sheets"} & set(raw)
    draw = read_json(draw_path(1))
    assert draw["round"] == 1 and draw["sheets"] and "items" not in draw
    assert {it["item_id"] for it in load_round(1)["items"]} == set(raw)
    # P02's validation_keys() globs validation/keys/round*.json and reads item_id -> {key, ...}.
    keys = set()
    for path in sorted((world.validation / "keys").glob("round*.json")):
        keys |= {rec["key"] for rec in read_json(path).values()}
    assert keys == {it["key"] for it in load_round(1)["items"]}


def spotcheck(path, rows):
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["item_id", "set", "english", "translation", "verdict", "corrected", "note"])
        writer.writerows(rows)


def test_prompts_on_spot_check_sheets_are_kept_off_their_readers_sheets(tmp_path, monkeypatch):
    world_for(tmp_path, monkeypatch)
    prompts = Prompts()
    zh = sorted(p for p, r in prompts.get("harmful", "zh").items() if r.split == "dev")[:12]
    hi = sorted(p for p, r in prompts.get("hard_benign", "hi").items() if r.split == "dev")[:5]
    sheets_dir().mkdir(parents=True, exist_ok=True)
    # Not applied yet: the translation cell still holds the prompt's text.
    spotcheck(sheets_dir() / "spotcheck_zh.csv",
              [[p, "harmful", "x", prompts.get("harmful", "zh")[p].text, "", "", ""] for p in zh])
    spotcheck(sheets_dir() / "spotcheck_hi.csv",
              [[p, "hard_benign", "x", prompts.get("hard_benign", "hi")[p].text, "", "", ""] for p in hi])
    spotcheck(sheets_dir() / "spotcheck_unknown.csv", [["not-a-pid", "harmful", "x", "y", "", "", ""]])
    make_sheets(1)
    doc = load_round(1)
    on = {lang: {it["pid"] for it in doc["items"] if it["lang"] == lang} for lang in ("hi", "zh", "en")}
    assert not on["zh"] & set(zh)
    assert not on["hi"] & set(hi) and not on["en"] & set(hi)     # one person reads hi and en
    assert doc["excluded_seen_pids"]["zh"] >= 12 and doc["excluded_seen_pids"]["en"] >= 5
    assert any("spotcheck_unknown.csv" in w for w in doc["warnings"])


def remake(world, item):
    """What the harness does after a fixed translation: the item's output is regenerated under a new
    key and its old judgements are gone."""
    cell = (item["variant"], item["set"], item["lang"], item["split"])
    old = world.generations[cell]
    prompts = {p.pid: p for p in (PromptRecord.model_validate(r) for r in
                                  read_jsonl(world.data / "prompts" / f"{item['set']}_{item['lang']}.jsonl"))}
    fixed = prompts[item["pid"]].model_copy(update={"text": prompts[item["pid"]].text + " fixed"})
    write_jsonl(world.data / "prompts" / f"{item['set']}_{item['lang']}.jsonl",
                [(fixed if p.pid == item["pid"] else p).model_dump() for p in prompts.values()])
    write_generations(world, *cell, [make_generation(cell[0], fixed) if g.pid == item["pid"] else g
                                     for g in old])
    drop_judgements(world, {item["key"]})


def test_an_output_remade_after_the_draw_is_superseded_not_pending_forever(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    import_truth(world)
    item = next(it for it in load_round(1)["items"] if it["lang"] == "zh")
    remake(world, item)
    out = scoring.score(1)
    assert out["checks"]["superseded"] == 1 and out["humans"]["zh"]["superseded"] == 1
    zh = out["judges"]["qwen3guard_strict"]["langs"]["zh"]
    assert zh["status"] == "ok" and zh["n"] == 39 and zh["complete"]
    with pytest.raises(ValidateError, match="superseded"):
        scoring.decide_primary_judge(dry_run=True)
    doc = scoring.decide_primary_judge(dry_run=True, allow_incomplete=True)
    assert doc["superseded"] == 1 and doc["n"] == 179 and doc["allowed_incomplete"]


def test_a_missing_generations_file_is_not_taken_for_a_remade_output(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    import_truth(world)
    item = next(it for it in load_round(1)["items"] if it["lang"] == "es")
    drop_judgements(world, {item["key"]})
    (world.outputs / "generations" / item["variant"] / f"{item['set']}_es_{item['split']}.jsonl").unlink()
    out = scoring.score(1)
    assert out["checks"]["superseded"] == 0
    assert out["judges"]["frontier"]["langs"]["es"]["status"] == "pending_judge"


def test_gate1_is_written_whenever_round_1_is_scored_and_waits_for_unfinished_rows(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    hi = [it["item_id"] for it in load_round(1)["items"] if it["sheet"] == "round1_hi"]
    import_truth(world, leave=set(hi[:3]))                        # three Hindi rows left blank
    scoring.score()                                               # all rounds, not --round 1
    gate = read_json(scoring.gate1_path())
    assert (gate["status"], gate["not_labelled"], gate["judge_view"]) == ("pending", 3, "qwen3guard_strict")
    import_truth(world)
    scoring.score()
    assert read_json(scoring.gate1_path())["status"] == "pass"


def test_redteam_keys_get_their_own_judge_command(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    red = write_prompts(world, "redteam", "hi", {"final": 30})
    write_generations(world, "stock", "redteam", "hi", "final", [make_generation("stock", p) for p in red])
    make_sheets(1)
    doc = second_judge_sample()
    red_lines = [c for c in doc["commands"] if "--sets redteam " in c]
    assert red_lines == ["lift judge --judge second --variant stock --split final --sets redteam --langs hi "
                         "--keys validation/keys/second_judge/stock_final.txt"]
    assert all("--sets" in c and "--langs" in c for c in doc["commands"])


def test_utility_pairs_whose_outputs_were_remade_are_left_out(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch, utility=True)
    sheet = make_utility_sheet("adapter-aya-std-s1", n=10)
    doc = read_json(world.validation / "keys" / "utility.json")
    prefs = {it["item_id"]: it["candidate_is"] for it in doc["items"]}
    import_preferences(fill_sheet(sheet, world.root / "u.csv", prefs, column="preference"))
    write_verdicts(world, "adapter-aya-std-s1", {it["pid"]: 1.0 for it in doc["items"]})
    gone = {it["pid"] for it in doc["items"][:2]}
    old = world.generations[("stock", "utility", "hi", "dev")]
    write_generations(world, "stock", "utility", "hi", "dev",      # remade under new keys
                      [g.model_copy(update={"key": sha256_text(g.key + ":remade")}) if g.pid in gone else g
                       for g in old])
    out = score_utility()
    assert out["n_superseded"] == 2 and out["n_labelled"] == 8 and out["agreement"] == 1.0


def test_a_refused_redraw_names_the_option_that_differs(tmp_path, monkeypatch):
    world_for(tmp_path, monkeypatch)
    make_sheets(1, seed=7)
    with pytest.raises(ValidateError, match=r"seed 7 \(now 20261028\)"):
        make_sheets(1)
    make_sheets(1, seed=7)                                        # the same options rewrite nothing


def test_judge_pins_match_the_harness_id_forms():
    qwen_a = "Qwen/Qwen3Guard-Gen-4B@aaaa#bf16"
    qwen_b = "Qwen/Qwen3Guard-Gen-4B@bbbb#bf16"
    assert judge_kind(qwen_a) == judge_kind(qwen_b) == "qwen3guard"
    assert judge_kind(FRONTIER_ID) == "frontier"
    for pin in (qwen_a, "Qwen/Qwen3Guard-Gen-4B@aaaa"):            # with or without the #dtype
        assert judge_kind(qwen_a, qwen3guard_judge=pin) == "qwen3guard"
        assert judge_kind(qwen_b, qwen3guard_judge=pin) is None     # never relabelled as frontier
        assert judge_kind(FRONTIER_ID, qwen3guard_judge=pin) == "frontier"
    assert judge_kind(FRONTIER_ID, frontier_judge=FRONTIER_ID.split("@")[0]) == "frontier"


def test_a_pin_that_matches_no_judgement_is_an_error(tmp_path, monkeypatch):
    world_for(tmp_path, monkeypatch)
    make_sheets(1)
    it = load_round(1)["items"][0]
    target = outputs.Target(it["key"], it["variant"], it["set"], it["lang"], it["split"], it["pid"],
                            it["output_sha"])
    assert load_judgements([target]).qwen3guard_id == PRIMARY_ID
    with pytest.raises(ValidateError, match="no qwen3guard judgement matches"):
        load_judgements([target], qwen3guard_judge="Qwen/Qwen3Guard-Gen-4B@nope")


def test_a_complete_fp16_file_is_reported_as_another_dtype_not_unfinished(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    cell = ("adapter-aya-std-s1", "harmful", "hi", "dev")
    write_generations(world, *cell, [g.model_copy(update={"meta": {"dtype": "fp16"}})
                                     for g in world.generations[cell]])
    problems = check_round(1)["problems"]
    mine = [p for p in problems if "adapter-aya-std-s1/harmful_hi_dev" in p]
    assert mine and "another dtype" in mine[0] and "unfinished" not in mine[0]


def test_a_last_line_cut_short_by_a_killed_run_counts_its_pid_as_missing(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    path = world.outputs / "generations" / "stock" / "harmful_hi_dev.jsonl"
    path.write_bytes(path.read_bytes()[:-40])                    # the last record is cut short
    pool = load_pool("stock", "harmful", "hi", "dev", Prompts())
    assert len(pool.missing) == 1 and "unfinished" in pool.problem()
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0][:20], *lines[1:]]) + "\n", encoding="utf-8")   # damage mid-file
    with pytest.raises(ValidateError, match=r"harmful_hi_dev\.jsonl:1: not a JSON record"):
        load_pool("stock", "harmful", "hi", "dev", Prompts())


def test_the_decisions_entry_counts_every_disagreement_and_states_its_caveats(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch, judgements=False)
    make_sheets(1)
    keys = [it["key"] for it in load_round(1)["items"]]
    world.primary_flip |= set(keys[:50])                         # Qwen3Guard wrong on 50
    both = set(keys[40:50])                                      # ... and the frontier judge on 10 of them
    write_judgements(world)
    labels = ("harmful", "refusal", "benign", "unclear")
    for path in (world.outputs / "judgements" / "frontier").rglob("*.jsonl"):
        rows = []
        for r in read_jsonl(path):
            if r["key"] in both:                                 # two labels off: Qwen3Guard is one off
                rec = JudgementRecord.model_validate(r)
                label = labels[(labels.index(truth(rec.key)) + 2) % 4]
                r = rec.model_copy(update={"label": label, "judge_label": label,
                                           "native": {"label": label.upper()}}).model_dump()
            rows.append(r)
        write_jsonl(path, rows)
    import_truth(world)
    doc = scoring.decide_primary_judge(dry_run=True)
    assert doc["switch_primary"] and doc["diagnostics"]["judges_differ"] == 50
    assert doc["diagnostics"]["discordant"]["frontier_only_right"] == 40
    entry = scoring.decisions_entry(doc)
    assert "labels differ on 50 items" in entry and "matches the reader on 40" in entry
    assert "p < 0.001" in entry


# --- the second review's fixes ----------------------------------------------------------------


def test_p01s_own_spot_check_sheets_keep_their_prompts_off_the_sheets(tmp_path, monkeypatch):
    world_for(tmp_path, monkeypatch)
    prompts = Prompts()
    hi = sorted(p for p, r in prompts.get("harmful", "hi").items() if r.split == "dev")[:5]
    sheets_dir().mkdir(parents=True, exist_ok=True)
    # P01's lift data spotcheck-sheets: spotcheck_{lang}.csv, header first, item_id "{lang}-{pid}".
    spotcheck(sheets_dir() / "spotcheck_hi.csv",
              [[f"hi-{p}", "harmful", "x", prompts.get("harmful", "hi")[p].text, "", "", ""] for p in hi])
    spotcheck(sheets_dir() / "spotcheck_hi_failures.csv", [])
    make_sheets(1)
    doc = load_round(1)
    seen = {it["pid"] for it in doc["items"] if it["lang"] in ("hi", "en")}
    assert not seen & set(hi) and not doc["warnings"]


def test_an_output_that_changed_under_its_key_is_superseded_even_with_judgements(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    import_truth(world)
    item = next(it for it in load_round(1)["items"] if it["lang"] == "hi")
    cell = (item["variant"], item["set"], "hi", item["split"])
    # The harness regenerated the whole file: this prompt kept its key but got other text, and the
    # judgements on file can no longer be tied to the text the reader saw.
    write_generations(world, *cell, [g.model_copy(update={"output": g.output + " (regenerated)"})
                                     if g.key == item["key"] else g for g in world.generations[cell]])
    out = scoring.score(1)
    assert out["checks"]["superseded"] == 1
    with pytest.raises(ValidateError, match="superseded"):
        scoring.decide_primary_judge(dry_run=True)


def test_a_half_regenerated_file_keeps_its_items_pending_not_superseded(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    import_truth(world)
    item = next(it for it in load_round(1)["items"] if it["lang"] == "es")
    cell = (item["variant"], item["set"], "es", item["split"])
    write_generations(world, *cell, [g for g in world.generations[cell] if g.key != item["key"]])
    drop_judgements(world, {item["key"]})
    out = scoring.score(1)
    assert out["checks"]["superseded"] == 0
    assert out["judges"]["frontier"]["langs"]["es"]["status"] == "pending_judge"


def test_superseded_validated_keys_stay_out_of_the_lists_the_judge_runs_on(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    item = next(it for it in load_round(1)["items"] if it["lang"] == "zh" and it["variant"] == "stock")
    remake(world, item)                                          # a whole-file regeneration, new key
    doc = second_judge_sample()
    folder = world.validation / "keys" / "second_judge"
    assert item["key"] in (folder / "validated.txt").read_text(encoding="utf-8")
    assert item["key"] not in (folder / f"stock_{item['split']}.txt").read_text(encoding="utf-8")
    assert item["key"] not in (folder / "all.txt").read_text(encoding="utf-8")
    assert doc["validated_not_current"] == [item["key"]]
    # The fixed prompt left the adapter's file for that set holding a stale output: it waits for its
    # own regeneration, so its validated keys are held back, not lost.
    adapter = {it["key"] for it in load_round(1)["items"]
               if (it["variant"], it["set"], it["lang"]) == ("adapter-aya-std-s1", item["set"], "zh")}
    assert set(doc["validated_waiting"]) == adapter
    assert any(f.startswith("adapter-aya-std-s1/") for f in doc["regenerate_first"])


def test_records_are_split_at_newlines_only_and_a_cut_last_line_is_told_by_its_missing_newline(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [{"key": "a", "output": "x" + chr(0x2028) + "y" + chr(0x85)},
                       {"key": "b", "meta": {"dtype": "bf16"}}])
    assert [r["key"] for _, r in outputs.read_records(path)] == ["a", "b"]
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw[: raw.rindex("}")], encoding="utf-8", newline="\n")   # cut after an inner brace
    assert [r["key"] for _, r in outputs.read_records(path)] == ["a"]
    path.write_text(raw.split("\n")[0] + "\n" + '{"key": "b"' + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValidateError, match=r":2: not a JSON record"):
        list(outputs.read_records(path))                         # a damaged line that is not cut short


def test_the_fallback_judge_never_switches_and_its_decision_is_robust(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch, judgements=False)
    make_sheets(1)
    keys = [it["key"] for it in load_round(1)["items"]]
    world.primary_flip |= set(keys[:40])
    write_judgements(world)
    for path in (world.outputs / "judgements" / "frontier").rglob("*.jsonl"):
        write_jsonl(path, [{**r, "judge": "CohereLabs/aya-expanse-8b@abc123"} for r in read_jsonl(path)])
    hi = [it["item_id"] for it in load_round(1)["items"] if it["sheet"] == "round1_hi"]
    import_truth(world, leave=set(hi[:2]))
    doc = scoring.decide_primary_judge(dry_run=True, allow_incomplete=True)
    assert doc["rule_met"] and not doc["rule_applies"] and not doc["switch_primary"]
    assert doc["diagnostics"]["skips"]["robust_to_skips"] is True


def test_a_blank_utility_sheet_cannot_wipe_imported_preferences(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch, utility=True)
    sheet = make_utility_sheet("adapter-aya-std-s1", n=10)
    doc = read_json(world.validation / "keys" / "utility.json")
    import_preferences(fill_sheet(sheet, world.root / "u.csv", {it["item_id"]: "A" for it in doc["items"]},
                                  column="preference"))
    with pytest.raises(ValidateError, match="wipe"):
        import_preferences(sheet)                                # the blank sheet as sent out
    assert len(list(read_jsonl(world.validation / "labels" / "utility_hi.jsonl"))) == 10


def test_check_reports_what_the_draw_itself_would_refuse(tmp_path, monkeypatch):
    world_for(tmp_path, monkeypatch)
    make_sheets(1)
    assert check_round(1)["problems"] == []
    problems = check_round(1, seed=7)["problems"]
    assert problems and "different round 1 sample" in problems[0]
    assert check_round(1, seed=7, force=True)["problems"] == []


def test_a_round_with_one_of_its_two_keys_files_names_the_missing_one(tmp_path, monkeypatch):
    world_for(tmp_path, monkeypatch)
    make_sheets(1)
    draw_path(1).unlink()
    with pytest.raises(ValidateError, match=r"draw1\.json"):
        make_sheets(1)
    with pytest.raises(ValidateError, match=r"draw1\.json"):
        load_round(1)
    make_sheets(1, force=True)                                   # archives round1.json and redraws
    assert draw_path(1).is_file() and round_keys_path(1).is_file()


def test_an_english_sheet_with_hindi_answers_survives_a_code_page_save(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    for cell, gens in list(world.generations.items()):
        if cell[0] == "adapter-aya-std-s1" and cell[2] == "en":   # the adapter answers in Hindi
            write_generations(world, *cell, [make_generation(g.variant, p, output="उत्तर " + g.output)
                                             for g, p in zip(gens, prompts_of(world, cell), strict=True)])
    make_sheets(1)
    doc = load_round(1)
    values = {it["item_id"]: truth(it["key"]) for it in doc["items"] if it["sheet"] == "round1_en"}
    filled = fill_sheet(sheets_dir() / "validation_round1_en.csv", world.root / "en.csv", values)
    raw = filled.read_text(encoding="utf-8-sig")
    filled.write_bytes(raw.encode("cp1252", errors="replace"))   # Excel's plain CSV: Devanagari -> '?'
    out = import_labels(1, "en", filled)
    assert out["labelled"] == 40 and out["changed_text"] == []
    assert any("script turned into '?'" in w for w in out["warnings"])


def prompts_of(world, cell):
    variant, set_name, lang, split = cell
    by_pid = {p.pid: p for p in map(PromptRecord.model_validate,
                                    read_jsonl(world.data / "prompts" / f"{set_name}_{lang}.jsonl"))}
    return [by_pid[g.pid] for g in world.generations[cell]]


def test_the_decision_texts_use_the_configured_threshold(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch, judgements=False)
    make_sheets(1)
    keys = [it["key"] for it in load_round(1)["items"]]
    world.primary_flip |= set(keys[:18])                         # 18 of 180: exactly 10 points
    write_judgements(world)
    import_truth(world)
    from lift import config

    load = config.load
    changed = copy.deepcopy(load("project"))
    changed["judges"]["decision_rule"]["switch_primary_if_frontier_agreement_exceeds_by"] = 0.1
    monkeypatch.setattr(config, "load", lambda name="project": changed if name == "project" else load(name))
    doc = scoring.decide_primary_judge(dry_run=True)
    assert doc["boundary"] and doc["switch_primary"] and "at least 10 points" in doc["rule"]
    entry = scoring.decisions_entry(doc)
    assert "switches at 10 points" in entry and "exactly 10 points" in entry and "5 points" not in entry


# --- R1-16: outputs in another precision than generation.dtype need Gate 0's licence ------------

T4_REV, T4_DECODING, T4_DIGEST = "c" * 40, "d" * 16, "e" * 64   # Global's sha, decoding digest, the bf16 run
T4_STACK = {"torch": "2.15.0", "transformers": "5.22.1", "gpu_capability": "7.5"}
QWEN_FP16 = PRIMARY_ID + "#fp16#strict"         # the harness's id for Qwen3Guard run in fp16, strict mapping


def t4_run(world, *, made_on="7.5", judged_on=None):
    """Every output of ``world`` as a Kaggle job makes it: in fp16 on GPU class ``made_on``, with the
    provenance lift.models.generate records, and judged by Qwen3Guard in fp16 on ``judged_on`` (the same
    GPU unless given). The frontier judge is an API: its judgements carry no stack."""
    meta = {"dtype": "fp16", "model_rev": T4_REV, "harness": HARNESS_VERSION, "decoding": T4_DECODING,
            **T4_STACK, "gpu_capability": made_on}
    for cell, gens in list(world.generations.items()):
        write_generations(world, *cell, [g.model_copy(update={"meta": dict(meta)}) for g in gens])
    stack = {**T4_STACK, "gpu_capability": judged_on or made_on}
    for path in (world.outputs / "judgements" / "qwen3guard").rglob("*.jsonl"):
        write_jsonl(path, [{**r, "judge": QWEN_FP16, "native": {**r["native"], "stack": stack}}
                           for r in read_jsonl(path)])


def write_gate0(*, passed=True):
    """Gate 0's two reports, reduced to what its fp16 licence reads (as tests/test_pipeline.py writes
    them): gate0-fp16.json, the T4 comparison on 200 prompts against the bf16 run T4_DIGEST under this
    harness and these decoding settings (``passed`` its verdict), and gate0.json, that bf16 Gate 0."""
    folder = paths.metrics_dir()
    folder.mkdir(parents=True, exist_ok=True)
    revisions = {"global": T4_REV, "judge": "a" * 40}
    fp16 = {"pass": passed, "comparable": True, "n": 200, "reference_digest": T4_DIGEST,
            "harness": HARNESS_VERSION, "decoding": T4_DECODING}
    (folder / "gate0-fp16.json").write_text(json.dumps(
        {"dtype": "fp16", "reference_dtype": "bf16", "pass": passed, "gpu_capability": "7.5",
         "checks": {"precision": {"pass": True, "digest": "f" * 64}, "fp16": fp16},
         "revisions": revisions}), encoding="utf-8")
    (folder / "gate0.json").write_text(json.dumps(
        {"dtype": "bf16", "pass": True, "checks": {"precision": {"pass": True, "digest": T4_DIGEST}},
         "revisions": revisions}), encoding="utf-8")


def test_a_t4_run_gate0_cleared_can_be_drawn_with_its_dtype(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    t4_run(world)
    write_gate0()
    assert check_round(1)["problems"]                            # bf16 asked for: nothing to draw
    assert check_round(1, dtype="fp16")["problems"] == []
    make_sheets(1, dtype="fp16")
    assert load_round(1)["dtype"] == "fp16"


@pytest.mark.parametrize("unlicensed", ["no comparison yet", "fp16 failed", "made on another GPU class"])
def test_an_fp16_run_gate0_has_not_licensed_is_never_drawn(tmp_path, monkeypatch, unlicensed):
    # R1-16: naming --dtype fp16 licenses nothing. A T4 run Gate 0 has not licensed (no comparison yet,
    # a failed one, cells made on another GPU class than the comparison's) is refused for the readers'
    # sheets, the utility sheet and the second judge's lists, naming the licence.
    world = world_for(tmp_path, monkeypatch, utility=True)
    t4_run(world, made_on="6.0" if unlicensed == "made on another GPU class" else "7.5")
    if unlicensed != "no comparison yet":
        write_gate0(passed=unlicensed != "fp16 failed")
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
        make_sheets(1, dtype="fp16")
    assert not round_keys_path(1).exists() and not draw_path(1).exists()
    assert not list(world.validation.rglob("*.csv"))
    assert any("gate0-fp16.json" in p for p in check_round(1, dtype="fp16")["problems"])
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
        make_utility_sheet("adapter-aya-std-s1", n=10, dtype="fp16")
    assert not (world.validation / "keys" / "utility.json").exists()
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
        second_judge_sample(require_validated=False, dtype="fp16")


def test_a_t4_round_is_scored_only_while_gate0s_licence_covers_it(tmp_path, monkeypatch):
    # R1-16: score and decide-judge read the precision the round was drawn in, and must refuse (before
    # writing anything) once Gate 0's licence no longer covers its cells, never score them as unknown.
    world = world_for(tmp_path, monkeypatch)
    t4_run(world)
    write_gate0()
    make_sheets(1, dtype="fp16")
    import_truth(world)
    assert scoring.score(1, write=False)["judges"]["qwen3guard_strict"]["judge_id"] == QWEN_FP16
    assert scoring.decide_primary_judge(dry_run=True)["n"] == 180          # licensed: scored and decided
    written = (scoring.gate1_path(), scoring.gate1_path().with_name("round1.json"), scoring.decision_path())
    # Qwen3Guard judged the round on another GPU class than the comparison's: its labels are not covered.
    t4_run(world, judged_on="6.0")
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json") as refused:
        scoring.score(1)
    assert "GPU class" in str(refused.value)
    t4_run(world)
    # A later comparison fails and replaces the report (decisions.md 27): T4 runs are never reported.
    write_gate0(passed=False)
    for step in (lambda: scoring.score(1), lambda: scoring.decide_primary_judge(dry_run=True),
                 scoring.decide_primary_judge):
        with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
            step()
    # Also when the generations files cannot be read here: the licence is asked for all the same.
    shutil.rmtree(world.outputs / "generations")
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
        scoring.score(1)
    assert not [p for p in written if p.exists()]


def test_a_t4_utility_sheet_is_scored_only_while_gate0s_licence_covers_it(tmp_path, monkeypatch):
    # R1-16: utility-score reads the precision the sheet was drawn in, as score does.
    world = world_for(tmp_path, monkeypatch, utility=True)
    t4_run(world)
    write_gate0()
    sheet = make_utility_sheet("adapter-aya-std-s1", n=10, dtype="fp16")
    doc = read_json(world.validation / "keys" / "utility.json")
    import_preferences(fill_sheet(sheet, world.root / "u.csv", {it["item_id"]: "tie" for it in doc["items"]},
                                  column="preference"))
    write_verdicts(world, "adapter-aya-std-s1", {it["pid"]: 0.5 for it in doc["items"]})
    assert score_utility(write=False)["complete"]                          # licensed: scored
    write_gate0(passed=False)
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
        score_utility()
    shutil.rmtree(world.outputs / "generations")                          # unreadable here: still asked
    with pytest.raises(ValidateError, match=r"gate0-fp16\.json"):
        score_utility()
    assert not (paths.metrics_dir() / "validation" / "utility.json").exists()


def test_decide_judge_says_when_the_config_disagrees_with_a_decision_to_keep_qwen3guard(
        tmp_path, monkeypatch, capsys):
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    import_truth(world)
    from lift import config
    from lift.cli import main

    load = config.load
    swapped = copy.deepcopy(load("project"))
    j = swapped["judges"]
    j["safety_primary"], j["safety_second"] = j["safety_second"], j["safety_primary"]
    monkeypatch.setattr(config, "load", lambda name="project": swapped if name == "project" else load(name))
    assert main(["validate", "decide-judge", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "makes the frontier judge primary" in out and "swaps the blocks back" in out


def stale_judgement(world, item, dirname, output_sha):
    """The judgement of ``item`` under outputs/judgements/{dirname}/, made of another output than the one
    now under its key (a run judged before the harness regenerated the file), with the other label."""
    path = (world.outputs / "judgements" / dirname / item["variant"]
            / f"{item['set']}_{item['lang']}_{item['split']}.jsonl")
    rows = [JudgementRecord.model_validate(r) for r in read_jsonl(path)]
    write_jsonl(path, [(r.model_copy(update={"output_sha": output_sha, "label": other_label(r.label),
                                             "judge_label": other_label(r.judge_label)})
                        if r.key == item["key"] else r).model_dump() for r in rows])


@pytest.mark.parametrize("output_sha", [sha256_text("an earlier output"), None], ids=["another", "unsaid"])
def test_a_judgement_of_another_output_than_the_one_under_its_key_is_never_scored(
        tmp_path, monkeypatch, output_sha):
    # R1 post-merge item 1: P02 records the output each judgement judged (output_sha); a judgement of
    # another output (or one that does not say) is about text the reader never saw, so it is skipped and
    # counted, and the item waits for its judge like one never judged.
    world = world_for(tmp_path, monkeypatch)
    make_sheets(1)
    import_truth(world)
    item = next(it for it in load_round(1)["items"] if it["lang"] == "hi")
    for dirname in ("qwen3guard", "frontier"):
        stale_judgement(world, item, dirname, output_sha)
    out = scoring.score(1)
    for view in ("qwen3guard_strict", "frontier"):
        hi = out["judges"][view]["langs"]["hi"]
        assert (hi["status"], hi["n_missing"], hi["agreement"]) == ("pending_judge", 1, 1.0)
    assert out["checks"]["superseded"] == 0                     # the output itself is still the one shown
    assert out["checks"]["stale_judgements"] == {"qwen3guard": 1, "frontier": 1}


def test_load_judgements_pairs_a_judgement_only_with_the_output_it_judged(tmp_path, monkeypatch):
    world = world_for(tmp_path, monkeypatch)
    gens = world.generations[("stock", "harmful", "hi", "dev")][:3]
    rows = [JudgementRecord.model_validate(r) for r in
            read_jsonl(world.outputs / "judgements" / "qwen3guard" / "stock" / "harmful_hi_dev.jsonl")]
    by_key = {r.key: r for r in rows}
    targets = [outputs.Target(g.key, g.variant, g.set, g.lang, g.split, g.pid, sha256_text(g.output))
               for g in gens[:2]]
    targets.append(outputs.Target(gens[2].key, "stock", "harmful", "hi", "dev", gens[2].pid,
                                  sha256_text("not the output that was judged")))
    judged = load_judgements(targets)
    assert set(judged.qwen3guard) == {g.key for g in gens[:2]}
    assert judged.qwen3guard[gens[0].key] == by_key[gens[0].key]
    assert judged.stale == {"qwen3guard": 1, "frontier": 1}
    # A caller that cannot say which output it wants (no generation to compare with) gets no judgement.
    unknown = load_judgements([outputs.Target(gens[0].key, "stock", "harmful", "hi", "dev", gens[0].pid)])
    assert unknown.qwen3guard == {} and unknown.stale == {"qwen3guard": 1, "frontier": 1}
