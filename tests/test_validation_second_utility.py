"""The second judge's key lists (P04 Build 6) and the blind Hindi utility check (Build 7).

Everything runs on the fake worlds of tests/validation_fakes.py: neutral placeholder texts, never a
real prompt or completion.
"""

from __future__ import annotations

import copy
import csv
import inspect
import json
import math
import random
import re
import shlex
from collections import defaultdict
from pathlib import Path

import pytest
from validation_fakes import (
    LANGS,
    UTILITY_ID,
    build_world,
    fill_sheet,
    judged_pair,
    make_generation,
    read_csv,
    use_world,
    write_generations,
    write_prompts,
    write_verdicts,
)

from lift import config, sync
from lift.cli import main
from lift.commands import run as run_cmd
from lift.io import read_jsonl, sha256_file, write_jsonl
from lift.judge import pipeline
from lift.judge.pipeline import JudgeOutcome
from lift.models.generate import ASIDE_DIR, generations_path
from lift.schemas import UtilityVerdict
from lift.stats.agreement import cohens_kappa, percent_agreement
from lift.validation.outputs import Prompts
from lift.validation.sample import DEFAULT_SEED, load_round, make_sheets
from lift.validation.second_judge import chosen_pids, second_judge_dir, second_judge_sample
from lift.validation.store import ValidateError
from lift.validation.utility import (
    UTILITY_COLUMNS,
    build_utility,
    import_preferences,
    load_utility_keys,
    make_utility_sheet,
    score_utility,
    utility_keys_path,
    utility_labels_path,
    utility_meta_path,
    utility_sheet_path,
)

CANDIDATE = "adapter-aya-std-s1"
OTHER_ADAPTER = "adapter-indicalign-std-s1"
REPAIRED = "wiseft-l0.50-adapter-aya-std-s1"
CAST = "cast-m1.0-adapter-aya-std-s1"
OTHER_JUDGE = "otherprovider/other-utility-model@2026-10-30"
OUTCOMES = ("win", "tie", "loss")


def patch_config(monkeypatch, changes: dict[str, object]) -> None:
    """configs/project.yaml with dotted keys replaced. config.load is cached, so wrap it."""
    real = config.load

    def load(name: str = "project") -> dict:
        data = real(name)
        if name != "project":
            return data
        data = copy.deepcopy(data)
        for dotted, value in changes.items():
            *parents, last = dotted.split(".")
            node = data
            for part in parents:
                node = node[part]
            node[last] = value
        return data

    monkeypatch.setattr(config, "load", load)


def write_csv(path: Path, rows: list[list[str]]) -> Path:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    return path


# ---------------------------------------------------------------------------------------------
# The second judge: a seeded 10% of the outputs plus every validated item (Build 6)
# ---------------------------------------------------------------------------------------------

# Round 1 needs 34 dev prompts per set (one person reads hi and en and never sees a prompt twice).
# A tenth of 35, 21 and 25 is not a whole number, so the ceiling shows.
N_DEV, N_FINAL, N_REDTEAM = 35, 21, 25
LISTS = ("random.txt", "validated.txt", "all.txt")
# Enough to draw round 1 and no more: every generations file read costs time on Windows.
ROUND_1 = {"dev": ("stock", CANDIDATE), "final": (), "extra_sets": False}
# No round drawn: one reader's languages are enough.
SMALL = {**ROUND_1, "langs": ("hi", "en")}


def second_judge_world(tmp_path, monkeypatch, *, dev=("stock", CANDIDATE, "adapter-aya-std-s2", REPAIRED),
                       final=("stock", REPAIRED), langs=LANGS, extra_sets=True):
    """Safety outputs of ``dev`` and ``final`` variants; with ``extra_sets`` also the Hindi red-teaming
    set (final split only, a safety set too) and the utility set (not one)."""
    world = use_world(monkeypatch, build_world(tmp_path, n_dev=N_DEV, n_final=N_FINAL, langs=langs,
                                               dev_variants=dev, final_variants=final, judgements=False,
                                               utility=extra_sets))
    if extra_sets:
        prompts = write_prompts(world, "redteam", "hi", {"final": N_REDTEAM})
        world.n["redteam"] = {"final": N_REDTEAM}
        for variant in final:
            write_generations(world, variant, "redteam", "hi", "final",
                              [make_generation(variant, p) for p in prompts])
    return world


def safety_files(world) -> list[tuple[str, str, str, str]]:
    return [f for f in world.generations if f[1] != "utility"]


def key_lists() -> dict[str, list[str]]:
    """Every .txt list the sample wrote, by file name: one key per line."""
    return {p.name: p.read_text(encoding="utf-8").splitlines()
            for p in sorted(second_judge_dir().glob("*.txt"))}


def sample_meta() -> dict:
    return json.loads((second_judge_dir() / "sample.json").read_text(encoding="utf-8"))


def undated(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k != "created"}


def test_second_judge_lists_every_validated_key_and_a_paired_tenth_of_each_set(tmp_path, monkeypatch):
    world = second_judge_world(tmp_path, monkeypatch)
    make_sheets(1)
    doc = second_judge_sample()
    lists, gens = key_lists(), world.by_key()

    validated = {it["key"] for it in load_round(1)["items"]}
    assert len(validated) == 180
    assert lists["validated.txt"] == sorted(validated)

    rnd = lists["random.txt"]
    assert rnd == sorted(set(rnd))
    picked: dict[tuple[str, str], dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    for key in rnd:
        g = gens[key]
        picked[(g.set, g.split)][(g.variant, g.lang)].add(g.pid)
    assert set(picked) == {(s, sp) for _, s, _, sp in safety_files(world)}     # never the utility set
    chosen = {}
    for (set_name, split), by_file in picked.items():
        # Every variant and language with outputs for this set and split, each with the same pids.
        assert set(by_file) == {(v, lang) for v, s, lang, sp in safety_files(world)
                                if (s, sp) == (set_name, split)}
        first = next(iter(by_file.values()))
        assert all(pids == first for pids in by_file.values())
        assert len(first) == math.ceil(0.10 * world.n[set_name][split])
        chosen[(set_name, split)] = first
    # Every output of a chosen pid is in, and nothing else.
    assert set(rnd) == {g.key for f in safety_files(world) for g in world.generations[f]
                        if g.pid in chosen[(f[1], f[3])]}

    union = set(rnd) | validated
    assert lists["all.txt"] == sorted(union)
    per_variant: dict[str, list[str]] = defaultdict(list)
    for key in sorted(union):
        per_variant[f"{gens[key].variant}_{gens[key].split}.txt"].append(key)
    assert {name: keys for name, keys in lists.items() if name not in LISTS} == per_variant
    assert {"adapter-aya-std-s2_dev.txt", "stock_final.txt", f"{REPAIRED}_final.txt"} <= set(per_variant)

    meta = sample_meta()
    both = set(rnd) & validated
    assert both and validated - both and set(rnd) - both          # every kind of membership occurs
    assert (meta["n_keys"], meta["n_random"], meta["n_validated"], meta["n_both"]) == (
        len(union), len(rnd), len(validated), len(both))
    assert meta["membership"] == {k: "both" if k in both else "random" if k in rnd else "validated"
                                  for k in sorted(union)}
    assert meta["pids"] == {f"{s}/{sp}": len(pids) for (s, sp), pids in chosen.items()}
    assert meta["per_variant"] == {name: len(keys) for name, keys in per_variant.items()}
    assert meta["fraction"] == 0.1 and meta["rounds"] == [1] and meta["incomplete_files"] == []
    assert len(meta["files"]) == len(safety_files(world)) and not any(f["skipped"] for f in meta["files"])
    assert undated(meta) == {k: v for k, v in doc.items() if k in meta and k != "created"}


def test_second_judge_validated_list_covers_every_round(tmp_path, monkeypatch):
    # Round 2 needs about 30 final prompts per set for the reader of hi and en.
    use_world(monkeypatch, build_world(tmp_path, n_dev=N_DEV, n_final=30, dev_variants=("stock", CANDIDATE),
                                       final_variants=("stock", CANDIDATE, REPAIRED), judgements=False,
                                       utility=False))
    make_sheets(1, hindi_overlap=True)
    make_sheets(2)
    doc = second_judge_sample()
    items = [it for r in (1, 2) for it in load_round(r)["items"]]
    assert any(it["slot"] == "b" for it in items)                  # the Hindi overlap sheet is there
    keys = {it["key"] for it in items}
    assert len(keys) == 300 and doc["rounds"] == [1, 2] and doc["n_validated"] == 300
    lists = key_lists()
    assert lists["validated.txt"] == sorted(keys)
    assert keys <= set(lists["all.txt"])


def test_second_judge_sample_is_reproducible_and_seeded(tmp_path, monkeypatch):
    second_judge_world(tmp_path, monkeypatch, **ROUND_1)
    make_sheets(1)
    second_judge_sample()
    first, meta = key_lists(), sample_meta()
    second_judge_sample()
    assert key_lists() == first
    assert undated(sample_meta()) == undated(meta)

    second_judge_sample(seed=7)
    other = key_lists()
    assert other["random.txt"] != first["random.txt"]
    assert other["validated.txt"] == first["validated.txt"]
    assert sample_meta()["seed"] == 7


def test_second_judge_skips_an_incomplete_file_and_a_later_run_only_adds_keys(tmp_path, monkeypatch):
    world = second_judge_world(tmp_path, monkeypatch, **ROUND_1)
    make_sheets(1)
    where = (CANDIDATE, "harmful", "hi", "dev")
    full = world.generations[where]
    write_generations(world, *where, full[:-5])                    # an unfinished run
    doc = second_judge_sample()
    assert doc["incomplete_files"] == [f"{CANDIDATE}/harmful_hi_dev.jsonl"]
    assert sample_meta()["incomplete_files"] == doc["incomplete_files"]
    entry = next(f for f in doc["files"] if (f["variant"], f["set"], f["lang"], f["split"]) == where)
    assert entry["skipped"] and entry["sampled"] == 0 and entry["missing"] == 5
    before = key_lists()
    assert not set(before["random.txt"]) & {g.key for g in full}
    # The validated items come from the round keys, so they are all still listed.
    assert before["validated.txt"] == sorted(it["key"] for it in load_round(1)["items"])

    write_generations(world, *where, full)                         # the run finishes
    assert second_judge_sample()["incomplete_files"] == []
    after = key_lists()
    added = set(after["random.txt"]) - set(before["random.txt"])
    assert set(before["random.txt"]) <= set(after["random.txt"])
    assert len(added) == math.ceil(0.10 * N_DEV)
    assert added <= {g.key for g in full}


def test_second_judge_takes_current_outputs_only(tmp_path, monkeypatch):
    world = second_judge_world(tmp_path, monkeypatch, **SMALL)
    second_judge_sample(require_validated=False)
    gens, first = world.by_key(), set(key_lists()["random.txt"])
    where = [(v, "harmful", "hi", "dev") for v in ("stock", CANDIDATE)]
    chosen = sorted({gens[k].pid for k in first if (gens[k].set, gens[k].split) == ("harmful", "dev")})
    pid = chosen[0]
    # A native reader fixes that prompt's Hindi translation after the runs: its outputs go stale.
    prompts = list(Prompts().get("harmful", "hi").values())
    fixed = [p.model_copy(update={"text": p.text + " fixed"}) if p.pid == pid else p for p in prompts]
    write_jsonl(world.data / "prompts" / "harmful_hi.jsonl", [p.model_dump() for p in fixed])
    stale = {g.key for f in where for g in world.generations[f] if g.pid == pid}
    kept = {g.key for f in where for g in world.generations[f] if g.pid in chosen[1:]}
    assert len(stale) == 2 and stale | kept <= first
    # The harness will not judge a file that still holds a stale output, so both files wait for a
    # regeneration: none of their keys is listed, and they are named.
    doc = second_judge_sample(require_validated=False)
    rnd = set(key_lists()["random.txt"])
    assert not rnd & (stale | kept)
    assert sorted(doc["regenerate_first"]) == sorted(f"{v}/harmful_hi_dev.jsonl" for v, *_ in where)
    # The harness regenerates each file whole: the fixed prompt under a new key, the others as before.
    fresh = set()
    for f in where:
        new = make_generation(f[0], next(p for p in fixed if p.pid == pid))
        write_generations(world, *f, [new if g.pid == pid else g for g in world.generations[f]])
        fresh.add(new.key)
    doc = second_judge_sample(require_validated=False)
    rnd = set(key_lists()["random.txt"])
    assert fresh <= rnd and not rnd & stale and kept <= rnd and doc["regenerate_first"] == []


def test_second_judge_without_round_keys_raises_unless_told(tmp_path, monkeypatch):
    second_judge_world(tmp_path, monkeypatch, **SMALL)
    with pytest.raises(ValidateError, match="round"):
        second_judge_sample()
    assert not second_judge_dir().exists()

    doc = second_judge_sample(require_validated=False)
    lists = key_lists()
    assert lists["validated.txt"] == [] and lists["random.txt"] and lists["all.txt"] == lists["random.txt"]
    assert doc["n_validated"] == 0 and doc["rounds"] == []
    assert set(sample_meta()["membership"].values()) == {"random"}


def test_second_judge_fraction_comes_from_the_config(tmp_path, monkeypatch):
    world = second_judge_world(tmp_path, monkeypatch, **{**SMALL, "extra_sets": True})
    patch_config(monkeypatch, {"second_judge_fraction": 0.25})
    doc = second_judge_sample(require_validated=False)
    assert doc["fraction"] == 0.25
    assert doc["pids"] == {f"{s}/{sp}": math.ceil(0.25 * n) for s, splits in world.n.items()
                           if s != "utility" for sp, n in splits.items()}
    assert second_judge_sample(fraction=0.5, require_validated=False)["pids"]["harmful/dev"] == 18


def test_cli_second_judge_sample(tmp_path, monkeypatch, capsys):
    second_judge_world(tmp_path, monkeypatch, **SMALL)
    assert main(["validate", "second-judge-sample"]) == 1          # no round keys yet
    assert capsys.readouterr().out.startswith("error:")
    assert main(["validate", "second-judge-sample", "--no-validated"]) == 0
    printed = capsys.readouterr().out
    names = sorted(p.stem for p in second_judge_dir().glob("*.txt") if p.name not in LISTS)
    assert names
    for name in names:          # one second-judge run per variant and split, sets and languages named
        variant, split = name.rsplit("_", 1)
        keys = f"validation/keys/second_judge/{name}.txt"
        assert re.search(rf"--variant {re.escape(variant)} --split {split} --sets [a-z_ ]+ --langs [a-z ]+ "
                         rf"--keys {re.escape(keys)}", printed)


# R2-08: decisions item 31's random part is drawn twice. P04's lists (lift.validation.second_judge, ranked
# with lift.validation.sample.DEFAULT_SEED, a share of second_judge_fraction) are the canonical path; the
# harness's `lift judge --judge second --sample F` (lift.judge.pipeline.second_judge_pids through judge_file,
# ranked with split_seed: lift judge has no --seed) must reproduce them exactly, and it is what a restart's
# refill line runs while P04 has drawn no list (lift.commands.run._judge_line). So that neither draw can move
# alone: the seeds and the shares are one, and the two samplers draw the same pids.

class FakeLID:
    def top(self, text, k=5):
        return [("hin_Deva", 0.99)][:k]


class FakeSecondJudge:
    """Stands in for the second judge: every output a refusal, written to outputs/judgements/fake/."""

    judge_id = "fake-second-judge@0001"
    short = "fake"

    def judge(self, items):
        return [JudgeOutcome("refusal", {"call": "refuse"}, "raw") for _ in items]


def refill_fraction(cell=("stock", "harmful", "hi", "dev")) -> float:
    """The --sample share of the line that refills the second judge's file of ``cell`` while P04 has drawn
    no key list for it, read from the line the code builds."""
    words = shlex.split(run_cmd._judge_line(*cell, role="second", dtype="bf16"))
    assert "--sample" in words and "--keys" not in words, words
    return float(words[words.index("--sample") + 1])


def test_p04s_default_seed_is_the_split_seed_the_harness_samples_with():
    assert DEFAULT_SEED == int(config.get("split_seed"))
    assert inspect.signature(second_judge_sample).parameters["seed"].default == DEFAULT_SEED


def test_the_refill_lines_sample_is_the_configured_second_judge_share():
    assert refill_fraction() == float(config.get("second_judge_fraction"))


def test_the_harness_sample_judges_exactly_p04s_random_part(tmp_path, monkeypatch):
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")          # judge_file's run manifests
    world = second_judge_world(tmp_path, monkeypatch, dev=("stock",), final=("stock",), langs=("hi", "en"))
    cells = sorted(cell for cell in world.generations if cell[1] != "utility")
    # Each cell's refill line, run as lift judge runs it (judge_file's sample= and its default seed) before
    # P04 has drawn any list, then P04's draw as `lift validate second-judge-sample --no-validated` makes it.
    judged = {}
    for cell in cells:
        out = pipeline.judge_file(generations_path(*cell), judge=FakeSecondJudge(), lid=FakeLID(),
                                  sample=refill_fraction(cell))
        judged[cell] = {r["pid"] for r in read_jsonl(out)}
    second_judge_sample(require_validated=False)
    listed = set(key_lists()["random.txt"])
    p04 = {cell: {g.pid for g in world.generations[cell] if g.key in listed} for cell in cells}
    assert all(p04.values()) and ("stock", "redteam", "hi", "final") in p04
    assert {c: sorted(pids) for c, pids in judged.items()} == {c: sorted(pids) for c, pids in p04.items()}
    # And the two samplers are one draw at any share and seed, set and split.
    for fraction, seed in ((0.1, DEFAULT_SEED), (0.25, DEFAULT_SEED), (0.5, 7)):
        p04_draw = chosen_pids(Prompts(), fraction, seed)
        assert len(p04_draw) == 7                         # three sets in dev and final, and redteam final
        harness = {k: pipeline.second_judge_pids(*k, fraction, seed) for k in p04_draw}
        assert harness == p04_draw, (fraction, seed)


# ---------------------------------------------------------------------------------------------
# The utility check: blind stock-vs-adapter Hindi pairs for the Hindi reader (Build 7)
# ---------------------------------------------------------------------------------------------

# The candidate's score from one verdict: order_ab shows it as A, order_ba shows it as B.
SCORE_AB = {"A": 1.0, "TIE": 0.5, "B": 0.0}
SCORE_BA = {"B": 1.0, "TIE": 0.5, "A": 0.0}


def utility_world(tmp_path, monkeypatch, *, extra: tuple[str, ...] = ()):
    """The fake Hindi utility set: 70 dev instructions answered by stock and the candidate."""
    world = use_world(monkeypatch, build_world(tmp_path, n_dev=1, n_final=1, langs=("hi",),
                                               dev_variants=("stock",), final_variants=(), judgements=False))
    prompts = [p for p in Prompts().get("utility", "hi").values() if p.split == "dev"]
    for variant in extra:
        write_generations(world, variant, "utility", "hi", "dev",
                          [make_generation(variant, p) for p in prompts])
    return world


def draw(n: int = 20, candidate: str = CANDIDATE, **options) -> list[dict]:
    make_utility_sheet(candidate, n=n, **options)
    return sorted(load_utility_keys()["items"], key=lambda it: it["item_id"])


def answer(world, prefs: dict[str, str], *, notes: dict[str, str] | None = None) -> dict:
    """The reader sends the sheet back with ``prefs`` (item_id -> preference cell); import it."""
    filled = fill_sheet(utility_sheet_path(), world.root / "filled.csv", prefs, column="preference",
                        notes=notes)
    return import_preferences(filled)


def verdict_file(world, orders: dict[str, tuple[str, str]], *, judge: str = UTILITY_ID) -> Path:
    """UtilityVerdicts with the given (order_ab, order_ba) winners; the score averages both orders. Each
    names the pair it judged as the world holds it now (validation_fakes.judged_pair)."""
    rows = [UtilityVerdict(pid=pid, variant=CANDIDATE, judge=judge, order_ab=ab, order_ba=ba,
                           score=(SCORE_AB[ab] + SCORE_BA[ba]) / 2,
                           **judged_pair(world, CANDIDATE, pid)).model_dump()
            for pid, (ab, ba) in sorted(orders.items())]
    path = world.outputs / "utility" / CANDIDATE / "utility_hi_dev.jsonl"
    write_jsonl(path, rows)
    return path


def outcome(score: float) -> str:
    return "win" if score > 0.5 else "loss" if score < 0.5 else "tie"


def test_utility_sheet_is_balanced_blind_and_reproducible(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    n = 21
    sheet = make_utility_sheet(CANDIDATE, n=n)
    assert sheet == utility_sheet_path()
    doc = load_utility_keys()
    items = doc["items"]
    assert (doc["candidate"], doc["n"], len(items), len({it["pid"] for it in items})) == (CANDIDATE, n, n, n)
    assert sum(it["candidate_is"] == "A" for it in items) == n // 2 == doc["candidate_as_a"]
    ids = [it["item_id"] for it in items]
    assert all(i.startswith("uhi-") for i in ids) and len(set(ids)) == n

    rows = read_csv(sheet)
    assert len(rows[0]) == 1 and rows[0][0].startswith("#")
    assert rows[1] == list(UTILITY_COLUMNS) == ["item_id", "prompt", "response_a", "response_b", "preference",
                                                "note"]
    gens, texts, by_id = world.by_key(), Prompts().get("utility", "hi"), {it["item_id"]: it for it in items}
    assert sorted(r[0] for r in rows[2:]) == sorted(ids)
    for item_id, prompt, a, b, preference, note in rows[2:]:
        it = by_id[item_id]
        cand, stock = gens[it["candidate_key"]], gens[it["stock_key"]]
        assert (cand.variant, cand.pid) == (CANDIDATE, it["pid"])
        assert (stock.variant, stock.pid) == ("stock", it["pid"])
        assert prompt == texts[it["pid"]].text
        shown_as_a = cand if it["candidate_is"] == "A" else stock
        shown_as_b = stock if it["candidate_is"] == "A" else cand
        assert (a, b) == (shown_as_a.output, shown_as_b.output)
        assert preference == note == ""
    # Nothing on the sheet says which model wrote which answer, and the order gives nothing away.
    cells = [c for row in rows[1:] for c in row]
    for secret in ("stock", "adapter", "aya", "utl-", *(it["candidate_key"][:12] for it in items),
                   *(it["stock_key"][:12] for it in items)):
        assert not any(secret in c for c in cells)
    sides = [by_id[r[0]]["candidate_is"] for r in rows[2:]]
    assert sides not in (sorted(sides), sorted(sides, reverse=True))
    pids = [by_id[r[0]]["pid"] for r in rows[2:]]
    assert pids != sorted(pids)

    # The same inputs give the same sample, and a re-run changes nothing.
    again, again_rows = build_utility(CANDIDATE, n=n)
    assert again["items"] == items and again_rows == rows[2:]
    sheet_bytes, keys_bytes = sheet.read_bytes(), utility_keys_path().read_bytes()
    assert make_utility_sheet(CANDIDATE, n=n) == sheet
    assert sheet.read_bytes() == sheet_bytes and utility_keys_path().read_bytes() == keys_bytes
    # ...not even to a sheet the reader is filling in.
    fill_sheet(sheet, sheet, {ids[0]: "A"}, column="preference")
    make_utility_sheet(CANDIDATE, n=n)
    assert next(r for r in read_csv(sheet)[2:] if r[0] == ids[0])[4] == "A"


def test_utility_ids_belong_to_one_candidate(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch, extra=(OTHER_ADAPTER,))
    first = {it["pid"]: it for it in draw(20)}
    filled = fill_sheet(utility_sheet_path(), tmp_path / "first.csv",
                        {it["item_id"]: "A" for it in first.values()}, column="preference")
    with pytest.raises(ValidateError, match="force"):
        make_utility_sheet(OTHER_ADAPTER, n=20)
    second = {it["pid"]: it for it in draw(20, OTHER_ADAPTER, force=True)}
    assert set(second) == set(first)                    # the same instructions for every candidate
    assert not {it["item_id"] for it in first.values()} & {it["item_id"] for it in second.values()}
    assert len(list((utility_keys_path().parent / "archive").glob("utility-*.json"))) == 1
    assert len(list((utility_sheet_path().parent / "archive").glob("validation_utility_hi-*.csv"))) == 1
    # A sheet made for one candidate can never be imported as another.
    with pytest.raises(ValidateError, match="unknown item_id|no item id of the utility sheet"):
        import_preferences(filled)
    assert not utility_labels_path().exists()
    # --force refuses while preferences are imported, or while the sheet has entries typed in.
    answer(world, {it["item_id"]: "B" for it in second.values()})
    with pytest.raises(ValidateError, match="imported"):
        make_utility_sheet(CANDIDATE, n=20, force=True)
    utility_labels_path().unlink()
    fill_sheet(utility_sheet_path(), utility_sheet_path(), {next(iter(second.values()))["item_id"]: "tie"},
               column="preference")
    with pytest.raises(ValidateError, match="move it away"):
        make_utility_sheet(CANDIDATE, n=20, force=True)
    assert load_utility_keys()["candidate"] == OTHER_ADAPTER


@pytest.mark.parametrize("variant", ["stock", REPAIRED, CAST])
def test_utility_sheet_refuses_a_candidate_that_is_not_an_adapter(tmp_path, monkeypatch, variant):
    utility_world(tmp_path, monkeypatch, extra=(REPAIRED, CAST))
    with pytest.raises(ValidateError, match="not an adapter"):
        make_utility_sheet(variant, n=10)
    assert not utility_keys_path().exists() and not utility_sheet_path().exists()


def test_utility_import_accepts_a_b_and_tie_in_any_case(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(12)
    cells = ["A", "a", " B ", "b", "tie", "TIE", "Tie", " tIe ", "A", "", "", "b"]
    prefs = {it["item_id"]: c for it, c in zip(items, cells, strict=True)}
    out = answer(world, prefs, notes={items[9]["item_id"]: "skipped"})     # items[10] is just left empty
    stored = {r["item_id"]: r["preference"] for r in read_jsonl(utility_labels_path())}
    canonical = {"a": "A", "b": "B", "tie": "tie"}
    assert stored == {i: canonical[c.strip().lower()] for i, c in prefs.items() if c.strip()}
    assert out["preference_counts"] == {"A": 3, "B": 3, "tie": 4}
    assert out["labelled"] == 10
    assert out["skipped"] == [items[9]["item_id"]] and out["unlabelled"] == [items[10]["item_id"]]


@pytest.mark.parametrize("cell", ["C", "AB", "A or B", "yes", "1", "harmful", "both", "equal"])
def test_utility_import_rejects_anything_but_a_b_or_tie(tmp_path, monkeypatch, cell):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(10)
    with pytest.raises(ValidateError, match="is not one of"):
        answer(world, {it["item_id"]: "A" for it in items} | {items[3]["item_id"]: cell})
    assert not utility_labels_path().exists()


def test_utility_import_reports_repeated_and_unknown_ids_together_and_writes_nothing(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(10)
    answer(world, {it["item_id"]: "B" for it in items})
    before = utility_labels_path().read_bytes()
    rows = read_csv(fill_sheet(utility_sheet_path(), tmp_path / "again.csv",
                               {it["item_id"]: "tie" for it in items}, column="preference"))
    rows.append(list(rows[4]))                    # a row pasted twice
    rows[5][0] = "uhi-zzzzzz"                     # an id that is on no sheet
    rows[6][4] = "maybe"
    with pytest.raises(ValidateError) as err:
        import_preferences(write_csv(tmp_path / "bad.csv", rows))
    message = str(err.value)
    assert "repeats" in message and "unknown item_id" in message and "'maybe'" in message
    assert not [cell for r in rows[2:] for cell in r[1:4] if cell in message]     # ids and rows, no text
    assert utility_labels_path().read_bytes() == before         # the earlier import is untouched


def test_utility_import_refuses_a_file_whose_responses_were_swapped(tmp_path, monkeypatch):
    utility_world(tmp_path, monkeypatch)
    items = draw(10)
    rows = read_csv(fill_sheet(utility_sheet_path(), tmp_path / "filled.csv",
                               {it["item_id"]: "A" for it in items}, column="preference"))
    swapped = [rows[0], rows[1]] + [[r[0], r[1], r[3], r[2], r[4], r[5]] for r in rows[2:]]
    with pytest.raises(ValidateError, match="differs from the sheet"):
        import_preferences(write_csv(tmp_path / "swapped.csv", swapped))
    assert not utility_labels_path().exists()
    # Moving whole columns, names included, is harmless: cells are matched by column name.
    moved = [rows[0]] + [[r[0], r[1], r[3], r[2], r[4], r[5]] for r in rows[1:]]
    assert import_preferences(write_csv(tmp_path / "moved.csv", moved))["labelled"] == 10


def test_score_utility_reads_the_preference_where_the_candidate_was_shown(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    as_a = [it for it in items if it["candidate_is"] == "A"]
    as_b = [it for it in items if it["candidate_is"] == "B"]
    assert len(as_a) == len(as_b) == 10
    # The judge: the candidate wins where it was shown as B and ties elsewhere.
    verdict_file(world, {it["pid"]: ("A", "B") for it in as_b} | {it["pid"]: ("TIE", "TIE") for it in as_a})

    # B where the candidate was B is a win (1); a tie is 0.5.
    answer(world, {it["item_id"]: "B" for it in as_b} | {it["item_id"]: "tie" for it in as_a})
    out = score_utility()
    assert out["human_counts"] == {"win": 10, "tie": 10, "loss": 0}
    assert out["human_win_rate"]["win_rate"] == pytest.approx(0.75)
    assert out["agreement"] == 1.0 and out["kappa"] == pytest.approx(1.0)
    assert out["confusion"]["matrix"] == [[10, 0, 0], [0, 10, 0], [0, 0, 0]]
    assert out["judge_minus_human"]["difference"] == pytest.approx(0.0)

    # B where the candidate was A is a loss (0).
    answer(world, {it["item_id"]: "b" for it in items})
    out = score_utility()
    assert out["human_counts"] == {"win": 10, "tie": 0, "loss": 10}
    assert out["human_win_rate"]["win_rate"] == pytest.approx(0.5)
    assert out["confusion"]["matrix"] == [[10, 0, 0], [0, 0, 0], [0, 10, 0]]


def test_score_utility_matches_lift_stats_on_a_mixed_sample(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(40)
    rng = random.Random(1103)
    kinds = [("A", "B"), ("A", "TIE"), ("TIE", "TIE"), ("B", "TIE"), ("B", "A"), ("A", "A"), ("B", "B")]
    prefs, orders, hs, js, human, judge, seen = {}, {}, [], [], [], [], []
    for it in items:
        side = it["candidate_is"]
        pref = rng.choice([side, "B" if side == "A" else "A", "tie"])
        ab, ba = rng.choice(kinds)
        prefs[it["item_id"]], orders[it["pid"]] = pref, (ab, ba)
        hs.append(0.5 if pref == "tie" else float(pref == side))
        js.append((SCORE_AB[ab] + SCORE_BA[ba]) / 2)
        human.append(outcome(hs[-1]))
        judge.append(outcome(js[-1]))
        shown = ab if side == "A" else ba       # the judge's verdict in the order the reader saw
        seen.append("tie" if shown == "TIE" else "win" if shown == side else "loss")
    answer(world, prefs)
    verdict_file(world, orders)
    out = score_utility()

    assert out["n_judged"] == 40 and out["complete"]
    assert 0 < out["agreement"] < 1
    assert out["agreement"] == pytest.approx(percent_agreement(human, judge))
    assert out["kappa"] == pytest.approx(cohens_kappa(human, judge, labels=OUTCOMES))
    assert out["agreement_shown_order"] == pytest.approx(percent_agreement(human, seen))
    decisive = [(h, j) for h, j in zip(human, judge, strict=True) if "tie" not in (h, j)]
    assert decisive and out["decisive_n"] == len(decisive)
    assert out["decisive_agreement"] == pytest.approx(percent_agreement([h for h, _ in decisive],
                                                                        [j for _, j in decisive]))
    assert out["human_counts"] == {k: human.count(k) for k in OUTCOMES}
    assert out["judge_counts"] == {k: judge.count(k) for k in OUTCOMES}
    conf = out["confusion"]
    assert (conf["labels"], conf["rows"], conf["columns"]) == (list(OUTCOMES), "reader", "judge")
    pairs = list(zip(human, judge, strict=True))
    assert conf["matrix"] == [[pairs.count((r, c)) for c in OUTCOMES] for r in OUTCOMES]
    assert out["human_win_rate"]["win_rate"] == pytest.approx(sum(hs) / 40)
    assert out["judge_win_rate"]["win_rate"] == pytest.approx(sum(js) / 40)
    diff = out["judge_minus_human"]
    assert diff["difference"] == pytest.approx((sum(js) - sum(hs)) / 40)
    assert diff["lo"] <= diff["difference"] <= diff["hi"]


def test_score_utility_position_consistency_counts_split_verdicts(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    # A judge that always prefers the answer it reads first splits every verdict (score 0.5).
    answer(world, {it["item_id"]: "A" for it in items})
    verdict_file(world, {it["pid"]: ("A", "A") for it in items})
    out = score_utility()
    assert out["judge_counts"] == {"win": 0, "tie": 20, "loss": 0}
    assert out["n_split_verdicts"] == 20 and out["position_consistency"] == 0.0
    # The reader picked A too: they agree in the order the reader saw, never on the averaged verdict.
    assert out["agreement_shown_order"] == 1.0 and out["agreement"] == 0.0
    assert out["decisive_n"] == 0

    # Consistent wins, ties and losses, and splits either way.
    kinds = [("A", "B"), ("TIE", "TIE"), ("B", "A"), ("A", "A"), ("B", "B")]
    verdict_file(world, {it["pid"]: kinds[i % 5] for i, it in enumerate(items)})
    out = score_utility()
    assert out["n_split_verdicts"] == 8 and out["position_consistency"] == pytest.approx(12 / 20)
    # A win in one order and a tie in the other is a split too: the two orders disagree.
    verdict_file(world, {it["pid"]: ("A", "TIE") if i < 5 else ("A", "B") for i, it in enumerate(items)})
    out = score_utility()
    assert out["n_split_verdicts"] == 5 and out["position_consistency"] == pytest.approx(15 / 20)


def test_score_utility_finds_verdicts_anywhere_under_outputs(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    answer(world, {it["item_id"]: "tie" for it in items})
    scores = {it["pid"]: (1.0, 0.75, 0.5, 0.25, 0.0)[i % 5] for i, it in enumerate(items)}
    path = write_verdicts(world, CANDIDATE, scores,
                          path=world.outputs / "p03" / "pairwise" / "run-7" / "verdicts.jsonl")
    # Another candidate's verdicts, by another judge, are not this candidate's: ignored.
    write_verdicts(world, OTHER_ADAPTER, {pid: 1.0 for pid in scores}, judge=OTHER_JUDGE)
    out = score_utility()
    assert out["judge_id"] == UTILITY_ID and out["n_judged"] == 20 and out["complete"]
    digests = {Path(f).resolve(): d for f, d in out["verdict_files"].items()}
    assert digests[path.resolve()] == sha256_file(path)
    assert out["judge_win_rate"]["win_rate"] == pytest.approx(sum(scores.values()) / 20)
    metrics = world.outputs / "metrics" / "validation"
    assert (metrics / "utility.json").is_file() and (metrics / "utility.md").is_file()
    # Or read from the files given.
    given = score_utility(verdict_files=[path])
    assert given["n_judged"] == 20 and list(given["verdict_files"]) == [str(path)]


def test_score_utility_never_reads_verdicts_from_what_lift_sync_never_syncs(tmp_path, monkeypatch):
    # R1-42: outputs/.sync-backup (local copies a forced pull replaced), outputs/.aside (what lift restart
    # set aside) and Gate 0's staging hold superseded or local-only copies. A UtilityVerdict names no
    # generation key, so read as current verdicts they would fill in the pairs the current file lacks.
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    answer(world, {it["item_id"]: "tie" for it in items})
    scores = {it["pid"]: 1.0 for it in items}
    current = write_verdicts(world, CANDIDATE, dict(list(scores.items())[:12]))   # 8 pairs not judged yet
    stamp = "20261105T000000Z"
    stores = (f"{ASIDE_DIR}/{stamp}/utility", f"{sync.BACKUP_DIR}/{stamp}/outputs/utility",
              "gate0/bf16/precision.new")
    aside, *_ = [write_verdicts(world, CANDIDATE, scores,
                                path=world.outputs / store / CANDIDATE / "utility_hi_dev.jsonl")
                 for store in stores]
    out = score_utility()
    assert (out["n_labelled"], out["n_judged"], out["n_missing"], out["complete"]) == (20, 12, 8, False)
    assert [Path(f).resolve() for f in out["verdict_files"]] == [current.resolve()]
    # Named on purpose (--verdicts), a set-aside copy is read like any other file.
    assert score_utility(verdict_files=[aside], write=False)["n_judged"] == 20


def test_score_utility_with_two_judges_needs_a_judge_id(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    answer(world, {it["item_id"]: "tie" for it in items})
    write_verdicts(world, CANDIDATE, {it["pid"]: 1.0 for it in items})
    write_verdicts(world, CANDIDATE, {it["pid"]: 0.0 for it in items}, judge=OTHER_JUDGE,
                   path=world.outputs / "utility_fallback" / "verdicts.jsonl")
    with pytest.raises(ValidateError, match="judge"):
        score_utility()
    out = score_utility(judge_id=UTILITY_ID)
    assert out["judge_id"] == UTILITY_ID and out["judge_win_rate"]["win_rate"] == 1.0
    out = score_utility(judge_id=OTHER_JUDGE)
    assert out["judge_id"] == OTHER_JUDGE and out["judge_win_rate"]["win_rate"] == 0.0


def test_score_utility_counts_pairs_without_a_verdict(tmp_path, monkeypatch):
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    skipped = items[0]["item_id"]
    answer(world, {it["item_id"]: "A" for it in items[1:]}, notes={skipped: "skipped"})
    out = score_utility()                                    # no verdicts yet
    assert (out["n_sheet"], out["n_labelled"], out["n_judged"], out["n_missing"]) == (20, 19, 0, 19)
    assert out["judge_id"] is None and out["complete"] is False

    write_verdicts(world, CANDIDATE, {it["pid"]: 1.0 for it in items[:13]})   # the skipped pair among them
    out = score_utility()
    assert (out["n_labelled"], out["n_judged"], out["n_missing"]) == (19, 12, 7)
    assert out["complete"] is False
    assert sum(out["human_counts"].values()) == 12 == out["human_win_rate"]["n"]
    report = world.outputs / "metrics" / "validation" / "utility.md"
    assert "incomplete" in report.read_text(encoding="utf-8")

    write_verdicts(world, CANDIDATE, {it["pid"]: 1.0 for it in items})
    out = score_utility()
    assert (out["n_judged"], out["n_missing"], out["complete"]) == (19, 0, True)


def test_score_utility_joins_a_verdict_only_on_the_answers_the_reader_saw(tmp_path, monkeypatch, capsys):
    # R1-43: a verdict is found by pid but joined only when it is about the pair the reader saw: the item's
    # two generation keys and, for an answer still in its file under that key, that answer's sha256. One
    # about other answers (judged before they were made again, or the same key with other text) or naming
    # no keys (written before lift utility recorded them) is left out and counted, never scored.
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    answer(world, {it["item_id"]: "tie" for it in items})
    path = write_verdicts(world, CANDIDATE, {it["pid"]: 1.0 for it in items})
    rows = list(read_jsonl(path))
    rows[0]["candidate_key"] = "0" * 64                      # a verdict on answers made before these
    rows[1]["stock_sha"] = "1" * 64                          # stock's answer under its key, other text
    rows[2].update(dict.fromkeys(("candidate_key", "stock_key", "candidate_sha", "stock_sha")))  # no keys
    write_jsonl(path, rows)
    assert main(["validate", "utility-score"]) == 0
    out = json.loads((world.outputs / "metrics" / "validation" / "utility.json").read_text(encoding="utf-8"))
    assert (out["n_labelled"], out["n_judged"], out["n_other_pair"], out["n_missing"]) == (20, 17, 3, 0)
    assert out["complete"] is False and out["human_win_rate"]["n"] == 17 and out["n_superseded"] == 0
    md = (world.outputs / "metrics" / "validation" / "utility.md").read_text(encoding="utf-8")
    assert "verdict is about other answers than the reader saw (other generation keys or text, or none " \
           "recorded): 3." in md
    printed = capsys.readouterr().out
    assert "incomplete: 0 without a verdict, 3 with one about other answers" in printed
    assert (f"left out 3 pairs whose verdict is about other answers than the reader saw: judge them again "
            f"(lift utility --variant {CANDIDATE} --split dev --judge-only)") in printed
    # With the answer files gone, the text cannot be checked (a missing file proves nothing): the verdict
    # under the item's keys is joined again; the other keys and the keyless verdict never are.
    for gens in sorted((world.outputs / "generations").rglob("utility_hi_*.jsonl")):
        gens.unlink()
    out = score_utility(write=False)
    assert (out["n_judged"], out["n_other_pair"], out["n_missing"]) == (18, 2, 0)


def test_score_utility_reports_the_warnings_its_import_gave(tmp_path, monkeypatch, capsys):
    # W3-07: the utility import records its warnings with the preferences (labels/utility_hi.meta.json),
    # and the score reports them (utility.json import_warnings, a line in utility.md, the command's output)
    world = utility_world(tmp_path, monkeypatch)
    items = draw(20)
    warned = answer(world, {it["item_id"]: "A" for it in items[1:]})["warnings"]     # one row left empty
    assert [w for w in warned if items[0]["item_id"] in w]
    assert json.loads(utility_meta_path().read_text(encoding="utf-8")).get("warnings") == warned
    write_verdicts(world, CANDIDATE, {it["pid"]: 1.0 for it in items})
    assert main(["validate", "utility-score"]) == 0
    printed = capsys.readouterr().out
    out = json.loads((world.outputs / "metrics" / "validation" / "utility.json").read_text(encoding="utf-8"))
    assert out.get("import_warnings") == warned
    md = (world.outputs / "metrics" / "validation" / "utility.md").read_text(encoding="utf-8")
    assert [ln for ln in md.splitlines() if ln.startswith("Imported with warnings") and warned[0] in ln]
    warning = "the sheet was imported with warnings, not corrected in this score (import_warnings): 1"
    assert f"warning: {warning}" in [line.strip() for line in printed.splitlines()]


def test_cli_utility_round_trip_prints_no_sheet_text(tmp_path, monkeypatch, capsys):
    world = utility_world(tmp_path, monkeypatch)
    assert main(["validate", "utility-sheet", "--variant", CANDIDATE]) == 0
    doc = load_utility_keys()
    assert (doc["n"], doc["candidate_as_a"]) == (60, 30)          # validation.utility_pairs_checked
    ties = {it["item_id"]: "tie" for it in doc["items"]}
    filled = fill_sheet(utility_sheet_path(), tmp_path / "filled.csv", ties, column="preference",
                        google_export=True)                          # as downloaded from Google Sheets
    assert main(["validate", "utility-import", "--file", str(filled)]) == 0
    assert len(list(read_jsonl(utility_labels_path()))) == 60
    write_verdicts(world, CANDIDATE, {it["pid"]: 0.5 for it in doc["items"]})
    assert main(["validate", "utility-score"]) == 0
    assert main(["validate", "utility-sheet", "--variant", "stock"]) == 1
    printed = capsys.readouterr().out
    texts = {g.output for g in world.by_key().values()}
    texts |= {p.text for p in Prompts().get("utility", "hi").values()}
    assert not [t for t in texts if t in printed]
