"""Contracts with the other stages, and the review's fixes: the keys file P02 reads, P01's
spot-check sheets, outputs remade after a draw, Gate 1, the second judge's commands, and the
harness's judge ids. Fake worlds only (tests/validation_fakes.py), all under tmp_path.
"""

import re

import pytest
from validation_fakes import (
    FRONTIER_ID,
    PRIMARY_ID,
    build_world,
    fill_sheet,
    make_generation,
    truth,
    use_world,
    write_generations,
    write_judgements,
    write_prompts,
    write_verdicts,
)

from lift.io import read_jsonl, write_jsonl
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


def world_for(tmp_path, monkeypatch, **options):
    return use_world(monkeypatch, build_world(tmp_path, **options))


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
    world = world_for(tmp_path, monkeypatch)
    sheet = make_utility_sheet("adapter-aya-std-s1", n=10)
    doc = read_json(world.validation / "keys" / "utility.json")
    prefs = {it["item_id"]: it["candidate_is"] for it in doc["items"]}
    import_preferences(fill_sheet(sheet, world.root / "u.csv", prefs, column="preference"))
    write_verdicts(world, "adapter-aya-std-s1", {it["pid"]: 1.0 for it in doc["items"]})
    gone = {it["pid"] for it in doc["items"][:2]}
    old = world.generations[("stock", "utility", "hi", "dev")]
    write_generations(world, "stock", "utility", "hi", "dev",
                      [g.model_copy(update={"key": "f" * 64}) if g.pid in gone else g for g in old])
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
    target = outputs.Target(it["key"], it["variant"], it["set"], it["lang"], it["split"], it["pid"])
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
