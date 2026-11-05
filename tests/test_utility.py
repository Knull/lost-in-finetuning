"""lift.utility, ``lift utility`` and lift.judge.local_pairwise, offline: both orders scored and averaged,
failures kept as ties and counted, verdict files that P04's blind check reads, the pairing refusals, the
answer checks against the registry and their refusals, what a refused pair retires and keeps, the
win-rate wiring, Hindi fidelity (tests/datafakes.py FakeLID), the command end to end over fake judges, and
the fallback judge over a canned generator. Placeholder texts only; nothing is downloaded."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from datafakes import FakeLID
from fakes import build_tiny_tokenizer, make_prompt

from lift import audit, config, utility
from lift.cachekey import intervention_digest
from lift.cli import main
from lift.io import read_jsonl, sha256_file, sha256_text, write_jsonl
from lift.judge import frontier, local_pairwise
from lift.models import generate as gen
from lift.models.generate import GreedyOutput, generations_path, read_generations
from lift.paths import REPO_ROOT, adapters_dir, manifests_dir, metrics_dir, output_dir
from lift.schemas import GenerationRecord, UtilityVerdict, VariantSpec
from lift.stats.paired import win_rate

CANDIDATE = "adapter-aya-std-s1"
SPLIT = "dev"
PIDS = ["utl-aya-00003", "utl-aya-00007", "utl-aya-00011", "utl-ind-00002", "utl-ind-00005", "utl-ind-00013"]
TEXTS = {pid: f"निर्देश {i}: एक छोटा अनुच्छेद लिखिए" for i, pid in enumerate(PIDS)}
PID_OF = {text: pid for pid, text in TEXTS.items()}
JUDGE_ID = "fakeprovider/utility-model@utility_v1:0123456789ab+apifake"
OTHER_JUDGE = "fakeprovider/other-model@utility_v1:0123456789ab+apifake"
META = {"dtype": "bf16", "model_rev": "c" * 40, "harness": 1, "template": "9" * 16, "decoding": "d" * 16,
        "torch": "2.15.0", "transformers": "5.22.1", "gpu": "NVIDIA A100-SXM4-40GB", "gpu_capability": "8.0"}
RUBRIC = REPO_ROOT / "configs" / "rubrics" / "utility_v1.md"
SAFETY_RUBRIC = REPO_ROOT / "configs" / "rubrics" / "safety_v1.md"
REV = "e" * 40
FAILED = "unparseable judge output"
ADAPTER = "a" * 64                     # the candidate's adapter hash, as its registry entry pins it
RESTART = "lift restart --variant {} --split dev --sets utility --langs hi"


def entry(name: str, adapter: str | None = ADAPTER) -> VariantSpec:
    """A configs/variants.yaml entry: stock's, or the candidate's adapter entry pinning ``adapter``."""
    if name == "stock":
        return VariantSpec(name="stock", method="stock")
    return VariantSpec(name=name, method="adapter", adapter=name,
                       artifacts={"adapter": adapter} if adapter else {})


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """configs/variants.yaml as these tests see it, {name: VariantSpec}: stock and the candidate, its adapter
    hash pinned. A test registers another entry by replacing one in the dict."""
    specs = {name: entry(name) for name in ("stock", CANDIDATE)}
    monkeypatch.setattr("lift.models.variants.load_registry", lambda: dict(specs))
    return specs


def check(variant: str, meta: dict | None = None) -> Path:
    """The check lift utility's GPU line records once generate has passed ``variant``'s answer file
    (``utility.record_checked``), by the variant as the registry builds it now (a stand-in for the loaded
    model, which record_checked never touches)."""
    from lift.models.variants import get_spec

    spec = get_spec(variant)
    built = SimpleNamespace(spec=spec, dtype=(meta or {}).get("dtype", META["dtype"]),
                            model_rev=(meta or {}).get("model_rev", META["model_rev"]),
                            adapter_hash=spec.artifacts.get("adapter"))
    return utility.record_checked(built, SPLIT)


def answer(variant: str, pid: str) -> str:
    """A placeholder answer in Devanagari (FakeLID reads it as hin_Deva), different on each side."""
    side = "पुराना" if variant == "stock" else "नया"
    return f"यह {side} उत्तर है, क्रमांक {PIDS.index(pid)}"


def record(variant: str, pid: str, *, output: str | None = None, meta: dict | None = None,
           text: str | None = None) -> GenerationRecord:
    out = answer(variant, pid) if output is None else output
    return GenerationRecord(
        key=sha256_text(f"{variant}|{pid}|{out}"), variant=variant, pid=pid, lang="hi", set="utility",
        split=SPLIT, prompt_sha=sha256_text(TEXTS[pid] if text is None else text), output=out,
        n_prompt_tokens=400, n_new_tokens=40, hit_cap=pid == PIDS[0], seconds=0.1,
        meta={**META, "run_config": f"rc-{variant}", **(meta or {})})


def write_side(variant: str, pids=PIDS, *, outputs: dict | None = None, meta: dict | None = None,
               meta_of: dict | None = None, checked: bool = True) -> dict[str, str]:
    """One side's utility_hi_dev answers (written in reverse pid order, as batches would) and, with
    ``checked``, the GPU line's check of them (``check``); {pid: output}."""
    records = [record(variant, pid, output=(outputs or {}).get(pid), meta={**(meta or {}),
                                                                           **(meta_of or {}).get(pid, {})})
               for pid in pids]
    write_jsonl(generations_path(variant, "utility", "hi", SPLIT),
                [r.model_dump() for r in reversed(records)])
    if checked:
        check(variant, meta)
    return {r.pid: r.output for r in records}


def write_prompts(store, texts=TEXTS) -> None:
    store[("utility", "hi")] = [make_prompt(pid, "hi", "utility", SPLIT, text=texts[pid]) for pid in PIDS]


class FakeJudge:
    """The frontier judge's interface (pairwise, prefill_pairwise, judge_id, stats) over a table by pid of
    (the winner when the candidate is shown as A, the winner when it is shown as B); None is a failed order.
    Which order a pair is in is told from where the candidate's answer (``candidate``: {pid: output}) is.
    Every call is recorded; ``pending`` is raised by prefill_pairwise, ``sync_pending`` by pairwise (a
    synchronous pass that meets an earlier batch still running)."""

    def __init__(self, candidate: dict[str, str], table: dict | None = None, *, judge_id: str = JUDGE_ID,
                 pending: BaseException | None = None) -> None:
        self.candidate = candidate
        self.table = table if table is not None else {pid: ("A", "B") for pid in PIDS}
        self.judge_id = judge_id
        self.pending = pending
        self.sync_pending: BaseException | None = None
        self.calls: list[tuple[list, list, list]] = []
        self.prefills: list[dict] = []

    @property
    def stats(self) -> dict:
        return {"calls": len(self.calls)}

    def prefill_pairwise(self, instructions, a, b, *, wait=True):
        self.prefills.append({"instructions": list(instructions), "a": list(a), "b": list(b), "wait": wait})
        if self.pending is not None:
            raise self.pending
        return {"mode": "sync", "n_requests": len(a), "n_cached": 0, "n_submitted": 0, "batch_ids": [],
                "usd": 0.0}

    def pairwise(self, instructions, a, b):
        if self.sync_pending is not None:
            raise self.sync_pending
        self.calls.append((list(instructions), list(a), list(b)))
        out = []
        for instruction, shown_a, shown_b in zip(instructions, a, b, strict=True):
            pid = PID_OF[instruction]
            ab, ba = self.table[pid]
            if ab == ba:
                winner = ab
            elif shown_a == self.candidate[pid]:
                winner = ab
            else:
                assert shown_b == self.candidate[pid], "the candidate's answer is in neither position"
                winner = ba
            out.append({"winner": winner, "reason": "placeholder reason", "raw": "{}", "error": None,
                        "model_echo": "fake"} if winner is not None else
                       {"winner": None, "reason": None, "raw": "", "error": FAILED, "model_echo": "fake"})
        return out


class Pending(Exception):
    """Stands in for lift.judge.frontier.FrontierBatchPending (section B of the P03 brief)."""


@pytest.fixture(autouse=True)
def pending_batches(monkeypatch):
    """lift.utility and the command catch ``frontier.FrontierBatchPending``, which section B adds in parallel:
    the tests raise this stand-in, also once the real class is there."""
    monkeypatch.setattr(frontier, "FrontierBatchPending", Pending, raising=False)


@pytest.fixture
def audited(tmp_path, monkeypatch):
    """The installed AuditKIT behind record_utility, its caches pointed at tmp."""
    pytest.importorskip("auditkit")
    for name, var in (("xdg-cache", "XDG_CACHE_HOME"), ("xdg-data", "XDG_DATA_HOME")):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(var, str(tmp_path / name))


@pytest.fixture
def world(fake_prompt_files):
    write_prompts(fake_prompt_files)
    return SimpleNamespace(cand=write_side(CANDIDATE), stock=write_side("stock"), prompts=fake_prompt_files)


def evaluate(judge, **kw):
    return utility.evaluate(CANDIDATE, SPLIT, judge=judge, lid=kw.pop("lid", FakeLID()), **kw)


# --- scoring ---------------------------------------------------------------------------------------------

NINE = [("A", "B", 1.0), ("A", "TIE", 0.75), ("TIE", "B", 0.75), ("A", "A", 0.5), ("TIE", "TIE", 0.5),
        ("B", "B", 0.5), ("B", "TIE", 0.25), ("TIE", "A", 0.25), ("B", "A", 0.0)]


@pytest.mark.parametrize("ab, ba, score", NINE)
def test_a_pair_scores_the_mean_of_its_two_orders(ab, ba, score):
    pair = utility.Pair(PIDS[0], TEXTS[PIDS[0]], record(CANDIDATE, PIDS[0]), record("stock", PIDS[0]))
    v = utility.make_verdict(pair, {"winner": ab, "reason": "r", "error": None},
                             {"winner": ba, "reason": "r", "error": None}, variant=CANDIDATE, split=SPLIT,
                             judge_id=JUDGE_ID)
    assert utility.pair_score(ab, ba) == score
    assert (v.order_ab, v.order_ba, v.score, v.error_ab, v.error_ba) == (ab, ba, score, None, None)


def test_both_orders_go_in_one_call_and_the_candidate_shown_as_b_judged_b_wins(world):
    # Order AB shows the candidate as A, order BA as B: a "B" verdict in order BA is the candidate's win.
    judge = FakeJudge(world.cand, {pid: ("TIE", "B") for pid in PIDS})
    verdicts, batch = utility.judge_pairs(utility.pair_generations(CANDIDATE, SPLIT), judge)
    ((instructions, a, b),) = judge.calls
    n = len(PIDS)
    assert instructions == [TEXTS[p] for p in PIDS] * 2
    assert a[:n] == [world.cand[p] for p in PIDS] and b[:n] == [world.stock[p] for p in PIDS]
    assert a[n:] == [world.stock[p] for p in PIDS] and b[n:] == [world.cand[p] for p in PIDS]
    assert {(v.order_ab, v.order_ba, v.score) for v in verdicts} == {("TIE", "B", 0.75)}
    assert batch is None and not judge.prefills


def test_a_failed_order_is_stored_as_a_tie_and_counted_beside_the_win_rate(world, audited):
    table = {pid: ("A", "B") for pid in PIDS}
    table[PIDS[0]] = (None, "B")             # order AB failed: a tie there; the win in order BA counts
    table[PIDS[1]] = (None, None)            # both orders failed: the pair stays in, at 0.5
    report = evaluate(FakeJudge(world.cand, table))
    v = {x.pid: x for x in utility.read_verdicts(report.verdicts_path)}
    assert (v[PIDS[0]].order_ab, v[PIDS[0]].error_ab, v[PIDS[0]].order_ba, v[PIDS[0]].score) == (
        "TIE", FAILED, "B", 0.75)
    assert (v[PIDS[1]].order_ab, v[PIDS[1]].order_ba, v[PIDS[1]].score) == ("TIE", "TIE", 0.5)
    assert v[PIDS[0]].reason_ab is None and v[PIDS[0]].reason_ba == "placeholder reason"
    m = report.metrics
    assert (m["n"], m["failed_orders"], m["pairs_with_failure"]) == (6, 3, 2)
    assert m["win_rate"] == (0.75 + 0.5 + 4) / 6
    assert m["win_rate_failures_as_losses"] == (0.5 + 0 + 4) / 6
    assert m["win_rate_failures_as_wins"] == (1 + 1 + 4) / 6
    assert m["failure_reasons"] == {FAILED: 3}
    assert m["position"] == {"n_orders": 12, "n_judged_orders": 9, "winner_A": 4, "winner_B": 5,
                             "winner_TIE": 0, "a_share_of_decisive": 4 / 9, "n_pairs_both_judged": 4,
                             "n_consistent": 4, "n_split": 0, "consistency": 1.0}


@pytest.mark.parametrize("result, match", [
    ({"winner": None, "error": None}, "neither a winner"),
    ({"winner": "a", "error": None}, "neither a winner"),
    ({"winner": "A", "error": "refused: x"}, "winner None"),
    ({"winner": None, "error": ""}, "winner None"),
    ({"winner": "A", "reason": 3, "error": None}, "not text"),
])
def test_a_judge_result_that_is_neither_a_verdict_nor_a_failure_is_a_broken_judge(result, match):
    pair = utility.Pair(PIDS[0], TEXTS[PIDS[0]], record(CANDIDATE, PIDS[0]), record("stock", PIDS[0]))
    good = {"winner": "A", "reason": "r", "error": None}
    with pytest.raises(ValueError, match=match):
        utility.make_verdict(pair, result, good, variant=CANDIDATE, split=SPLIT, judge_id=JUDGE_ID)


# --- the win rate, fidelity and the metrics files ---------------------------------------------------------


def test_the_win_rate_is_lift_stats_win_rate_overall_and_by_source(world, audited):
    table = dict(zip(PIDS, [("A", "B"), ("B", "A"), ("A", "TIE"), ("TIE", "TIE"), ("B", "TIE"), ("A", "A")],
                     strict=True))
    scores = [1.0, 0.0, 0.75, 0.5, 0.25, 0.5]
    m = evaluate(FakeJudge(world.cand, table)).metrics
    keys = ("win_rate", "lo", "hi", "p_vs_parity", "n")
    assert {k: m[k] for k in keys} == win_rate(scores, clusters=PIDS, n_boot=10_000)
    assert m["n_boot"] == 10_000 and m["win_rate"] == sum(scores) / 6
    for src, idx in (("aya", [0, 1, 2]), ("ind", [3, 4, 5])):
        expected = win_rate([scores[i] for i in idx], clusters=[PIDS[i] for i in idx], n_boot=10_000)
        assert {k: m["by_source"][src][k] for k in keys} == expected
    assert m["by_source"]["aya"]["name"].startswith("Aya Dataset")
    assert m["outcomes"] == {"win": 2, "tie": 2, "loss": 2}
    assert m["scores"] == {"1.0": 1, "0.75": 1, "0.5": 2, "0.25": 1, "0.0": 1}
    assert (m["position"]["winner_A"], m["position"]["winner_B"], m["position"]["winner_TIE"]) == (5, 3, 4)
    assert (m["position"]["n_consistent"], m["position"]["n_split"]) == (3, 3)
    assert m["failed_orders"] == 0 and m["win_rate_failures_as_losses"] == m["win_rate"]


def test_hindi_fidelity_of_both_sides_keeps_every_answer_in_the_denominator(fake_prompt_files, audited):
    write_prompts(fake_prompt_files)
    cand = write_side(CANDIDATE, outputs={PIDS[0]: "A plain English answer", PIDS[1]: ""})
    stock = write_side("stock")
    lid = FakeLID({stock[PIDS[2]]: ("anp_Deva", 0.9),                       # formal Hindi read as Angika
                   stock[PIDS[3]]: [("hin_Deva", 0.3), ("eng_Latn", 0.25)]})  # Hindi on top, too little mass
    m = evaluate(FakeJudge(cand), lid=lid).metrics
    assert m["fidelity"]["candidate"] == {"top_hi": 4 / 6, "is_lang_hi": 4 / 6, "n": 6, "n_top_hi": 4,
                                          "n_is_lang_hi": 4,
                                          "top_labels": {"hin_Deva": 4, "eng_Latn": 1, "und_Zyyy": 1}}
    stock_f = m["fidelity"]["stock"]
    assert (stock_f["top_hi"], stock_f["is_lang_hi"], stock_f["n"]) == (1.0, 5 / 6, 6)
    assert m["by_source"]["aya"]["fidelity"]["candidate"]["top_hi"] == 1 / 3
    assert m["by_source"]["ind"]["fidelity"]["stock"]["is_lang_hi"] == 2 / 3


def test_the_metrics_name_both_answer_files_the_verdict_file_and_what_the_answers_were_checked_against(
        world, audited):
    # Version 2 (C06/C15/C18): beside each file's sha256, the candidate's adapter hash and intervention digest
    # and each side's recorded check, so a reader can tell when the metrics no longer describe the registry.
    report = evaluate(FakeJudge(world.cand))
    m = report.metrics
    assert (m["kind"], m["version"], m["variant"], m["reference"], m["split"], m["lang"]) == (
        "utility", 2, CANDIDATE, "stock", SPLIT, "hi")
    assert (m["judge"], m["judge_dir"], m["dtype"]) == (JUDGE_ID, "frontier", "bf16")
    assert m["candidate_adapter_hash"] == ADAPTER
    assert m["candidate_intervention"] == intervention_digest(entry(CANDIDATE))
    for side, variant in (("candidate", CANDIDATE), ("stock", "stock")):
        g = m["generation"][side]
        path = generations_path(variant, "utility", "hi", SPLIT)
        assert g["path"] == f"outputs/generations/{variant}/utility_hi_dev.jsonl"
        assert g["file_sha256"] == sha256_file(path) and g["variant"] == variant
        assert (g["n"], g["n_cap_hits"], g["cap_hit_rate"], g["mean_new_tokens"]) == (6, 1, 1 / 6, 40.0)
        assert g["model_rev"] == ["c" * 40] and g["run_config"] == [f"rc-{variant}"] and len(g["stack"]) == 1
        checked = g["checked"]
        assert checked["path"] == f"outputs/utility/checked/{variant}/utility_hi_dev.json"
        assert checked["sha256"] == sha256_file(utility.checked_path(variant, SPLIT))
        assert (checked["run_config"], checked["model_rev"], checked["harness"]) == (
            f"rc-{variant}", "c" * 40, 1)
        assert checked["adapter_hash"] == (ADAPTER if variant == CANDIDATE else None)
        assert checked["intervention"] == intervention_digest(entry(variant))
    assert m["verdicts_path"] == f"outputs/utility/frontier/{CANDIDATE}/utility_hi_dev.jsonl"
    assert m["verdicts_sha256"] == sha256_file(report.verdicts_path)
    json_path, md_path = utility.metrics_paths(CANDIDATE, SPLIT)
    assert (report.metrics_json, report.metrics_md) == (json_path, md_path)
    raw = json_path.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n") and raw.startswith(b'{\n "kind": "utility"')
    assert json.loads(raw) == json.loads(json.dumps(m))
    md = md_path.read_text(encoding="utf-8")
    assert "| win rate (n=6 pairs) | 1.000" in md and "| aya: Aya Dataset" in md
    assert f"the candidate's adapter `{ADAPTER[:12]}`" in md
    for text in [*TEXTS.values(), *world.cand.values(), *world.stock.values()]:
        assert text not in md and text.encode("utf-8") not in raw


# --- pairing refusals -------------------------------------------------------------------------------------


def refusal(**kw) -> str:
    with pytest.raises(utility.UtilityRefused) as exc:
        utility.pair_generations(CANDIDATE, SPLIT, **kw)
    return str(exc.value)


def test_a_missing_answer_refuses_the_pair(world):
    write_side(CANDIDATE, PIDS[:-1])
    message = refusal()
    assert f"1 of the 6 dev prompts have no answer, e.g. ['{PIDS[-1]}']" in message
    assert f"lift utility --variant {CANDIDATE} --split dev" in message


def test_a_missing_or_empty_answer_file_refuses_the_pair(fake_prompt_files):
    write_prompts(fake_prompt_files)
    write_side(CANDIDATE)
    assert "outputs/generations/stock/utility_hi_dev.jsonl is missing or empty" in refusal()
    stock = generations_path("stock", "utility", "hi", SPLIT)
    stock.parent.mkdir(parents=True)
    stock.write_bytes(b"")                                       # what lift restart leaves
    assert "is missing or empty: stock's answers come first" in refusal()


def test_both_sides_problems_are_listed_at_once(world):
    write_side(CANDIDATE, PIDS[:-1])
    write_side("stock", meta={"dtype": "fp16"})
    message = refusal()
    assert f"generations/{CANDIDATE}/utility_hi_dev.jsonl: 1 of the 6" in message
    assert "generations/stock/utility_hi_dev.jsonl: its answers are in fp16, and this run is bf16" in message


def test_answers_to_a_changed_prompt_name_the_cells_restart_line(world):
    write_prompts(world.prompts, {**TEXTS, PIDS[2]: "निर्देश बदल गया"})
    message = refusal()
    assert f"1 answers were made from another prompt text, e.g. ['{PIDS[2]}']" in message
    for variant in (CANDIDATE, "stock"):
        assert f"lift restart --variant {variant} --split dev --sets utility --langs hi" in message


@pytest.mark.parametrize("field", ["template", "model_rev", "harness", "decoding"])
def test_answers_made_under_other_settings_are_never_paired(world, field):
    write_side(CANDIDATE, meta={field: "other"}, checked=False)
    message = refusal()
    assert "cannot be paired" in message and f"'{field}': (['other'], ['{META[field]}'])" in message
    for variant in (CANDIDATE, "stock"):
        assert f"lift restart --variant {variant} --split dev --sets utility --langs hi" in message


def test_answers_in_another_precision_than_the_run_are_refused(world):
    write_side(CANDIDATE, meta={"dtype": "fp16"})
    message = refusal()
    assert "its answers are in fp16, and this run is bf16" in message and "--dtype fp16" in message


def test_another_precision_needs_gate0s_licence(world):
    write_side(CANDIDATE, meta={"dtype": "fp16", "gpu_capability": "7.5"})
    write_side("stock", meta={"dtype": "fp16", "gpu_capability": "7.5"})
    message = refusal(dtype="fp16")
    assert "every reported number is bf16" in message and "gate0-fp16.json does not exist" in message


def test_records_without_provenance_or_mixing_stacks_are_refused(world):
    write_side(CANDIDATE, meta={"decoding": None})
    assert "records name no ['decoding'] ({'decoding': 6})" in refusal()
    write_side(CANDIDATE, meta_of={PIDS[0]: {"torch": "2.99.0"}})
    assert "the file mixes 1 run configurations and 2 stacks" in refusal()


def test_answers_to_other_pids_or_cells_are_refused(world):
    rows = [*read_jsonl(generations_path(CANDIDATE, "utility", "hi", SPLIT)),
            record(CANDIDATE, PIDS[0], output="x").model_copy(update={"pid": "utl-aya-09999", "key": "k"})
            .model_dump()]
    write_jsonl(generations_path(CANDIDATE, "utility", "hi", SPLIT), rows)
    assert "1 answers are to pids that are not dev utility prompts, e.g. ['utl-aya-09999']" in refusal()
    write_side(CANDIDATE)
    rows = list(read_jsonl(generations_path("stock", "utility", "hi", SPLIT)))
    rows[0]["variant"] = CANDIDATE
    write_jsonl(generations_path("stock", "utility", "hi", SPLIT), rows)
    assert "1 records belong to another variant, set, language or split" in refusal()


def test_stock_is_never_the_candidate():
    with pytest.raises(ValueError, match="cannot be stock"):
        utility.pair_generations("stock", SPLIT)


# --- answers checked against the registry, and what a refused pair retires ---------------------------------


def check_refusal() -> str:
    with pytest.raises(utility.UtilityRefused) as exc:
        utility.check_answers(utility.pair_generations(CANDIDATE, SPLIT))
    return str(exc.value)


def test_a_check_names_the_answer_file_and_what_the_variant_was_built_from(world):
    for variant in (CANDIDATE, "stock"):
        path = utility.checked_path(variant, SPLIT)
        doc = json.loads(path.read_text(encoding="utf-8"))
        answers = generations_path(variant, "utility", "hi", SPLIT)
        assert path == output_dir() / "utility" / "checked" / variant / "utility_hi_dev.json"
        assert (doc["kind"], doc["variant"], doc["split"], doc["dtype"], doc["n"]) == (
            "utility_checked", variant, SPLIT, "bf16", 6)
        assert (doc["path"], doc["sha256"]) == (f"outputs/generations/{variant}/utility_hi_dev.jsonl",
                                                sha256_file(answers))
        assert (doc["run_config"], doc["model_rev"], doc["harness"]) == (f"rc-{variant}", "c" * 40, 1)
        assert doc["adapter_hash"] == (ADAPTER if variant == CANDIDATE else None)
        assert doc["intervention"] == intervention_digest(entry(variant))
    before = utility.checked_path(CANDIDATE, SPLIT).read_bytes()
    check(CANDIDATE)                                    # the same check again: the file is left as it is
    assert utility.checked_path(CANDIDATE, SPLIT).read_bytes() == before
    assert set(utility.check_answers(utility.pair_generations(CANDIDATE, SPLIT))) == {CANDIDATE, "stock"}
    with pytest.raises(ValueError, match="not stock's answers in one configuration"):
        check("stock", {"model_rev": "f" * 40})                 # never a check of what was not built


def test_answers_no_gpu_line_checked_are_refused_with_the_gpu_line(world):
    # C06/C15/C18: a machine that loads no model judges only answers a GPU line passed through generate.
    utility.checked_path(CANDIDATE, SPLIT).unlink()
    message = check_refusal()
    assert f"no GPU line has checked its answers against {CANDIDATE} as registered now" in message
    assert f"`lift utility --variant {CANDIDATE} --split dev --generate-only` on a GPU runtime" in message
    assert "outputs/generations/stock/" not in message           # stock's answers were checked: not refused


@pytest.mark.parametrize("change, expected", [
    ("answers", "it changed since a GPU line checked it"),
    ("unreadable", "is not a check lift utility wrote"),
    ("another file's", "is a check of stock's dev answers in bf16, not of these"),
])
def test_a_check_that_does_not_name_the_answers_as_they_are_is_refused(world, change, expected):
    path = utility.checked_path(CANDIDATE, SPLIT)
    if change == "answers":                                    # made again, and not checked since
        write_side(CANDIDATE, outputs={PIDS[0]: "यह दोबारा बना उत्तर है"}, checked=False)
    elif change == "unreadable":
        path.write_text("{", encoding="utf-8")
    else:
        shutil.copyfile(utility.checked_path("stock", SPLIT), path)
    message = check_refusal()
    assert expected in message and f"--variant {CANDIDATE} --split dev --generate-only" in message


def test_answers_of_a_replaced_registry_entry_are_refused_with_the_restart_line(world, registry):
    # C06/C15/C18: the adapter was trained again and registered with --replace after its answers were made.
    registry[CANDIDATE] = entry(CANDIDATE, adapter="b" * 64)
    message = check_refusal()
    assert f"{CANDIDATE}'s entry in configs/variants.yaml here is not the one its answers" in message
    assert "this checkout is behind the ref the GPU job ran, update it (git pull)" in message
    assert "intervention digest" in message
    assert "adapter hash (aaaaaaaaaaaa made, bbbbbbbbbbbb pinned)" in message
    assert RESTART.format(CANDIDATE) in message and RESTART.format("stock") not in message
    registry[CANDIDATE] = entry(CANDIDATE, adapter=None)       # unpinned now: still another intervention
    assert "intervention digest" in check_refusal()
    registry.pop(CANDIDATE)
    assert f"{CANDIDATE} is not in configs/variants.yaml" in check_refusal()


def test_answers_made_under_another_harness_are_refused_on_both_sides(world, monkeypatch):
    monkeypatch.setattr(gen, "HARNESS_VERSION", 2)            # a harness fix since: every answer is stale
    message = check_refusal()
    assert "its answers were made under harness version ['1'], and the harness here is version 2" in message
    assert RESTART.format(CANDIDATE) in message and RESTART.format("stock") in message


@pytest.mark.parametrize("pin, refused", [("main", False), ("c" * 40, False), ("d" * 40, True)])
def test_the_answers_model_revision_is_compared_with_the_pin_once_there_is_one(world, pin, refused):
    # Unpinned ("main"), nothing says which revision is current; pinned, both sides must be made from it.
    config.load("project")["models"]["global"]["revision"] = pin
    pairing = utility.pair_generations(CANDIDATE, SPLIT)
    if not refused:
        assert set(utility.check_answers(pairing)) == {CANDIDATE, "stock"}
        return
    message = check_refusal()
    assert f"made from model revision ['{'c' * 40}'], and models.global is pinned to {pin} here" in message
    assert RESTART.format(CANDIDATE) in message and RESTART.format("stock") in message


def test_a_refused_pair_retires_what_was_derived_from_it_and_keeps_the_answer_checks(world, audited):
    # C07: the verdicts, metrics and AuditKIT record of a pair that can no longer be paired are copied aside
    # and left empty (never removed: sync never deletes), so P04 and Gate 1 cannot read them. V03: both
    # answer checks stay as they are: check_answers reads them again on every run, and only a GPU line could
    # make them again.
    from lift.validation.utility import find_verdict_files

    report = evaluate(FakeJudge(world.cand))
    derived = {report.verdicts_path: ".aside", report.metrics_json: "", report.metrics_md: "",
               report.record_path: ""}
    checks = [utility.checked_path(name, SPLIT) for name in (CANDIDATE, "stock")]
    before = {path: path.read_bytes() for path in [*derived, *checks]}
    assert all(before.values()) and find_verdict_files() == [report.verdicts_path]
    write_side(CANDIDATE, PIDS[:-1], checked=False)          # an answer lost: the pair is refused (rule 3)
    judge = FakeJudge(world.cand)
    with pytest.raises(utility.UtilityRefused, match="have no answer"):
        evaluate(judge)
    stamp, = (output_dir() / ".aside").iterdir()
    for path, suffix in derived.items():
        copy = stamp / path.relative_to(output_dir())
        assert path.read_bytes() == b"" and copy.with_name(copy.name + suffix).read_bytes() == before[path]
    assert find_verdict_files() == [] and not judge.calls    # P04 reads neither the emptied file nor the copy
    assert all(path.read_bytes() == before[path] for path in checks)
    assert not (stamp / "utility" / "checked").exists()          # kept, so never copied aside
    with pytest.raises(utility.UtilityRefused):
        evaluate(judge)                                         # refused again: nothing left to set aside
    assert len(list((output_dir() / ".aside").iterdir())) == 1


def test_a_refused_verdict_file_retires_nothing(world, audited):
    evaluate(FakeJudge(world.cand, judge_id=OTHER_JUDGE))
    json_path, _md = utility.metrics_paths(CANDIDATE, SPLIT)
    before = json_path.read_bytes()
    with pytest.raises(utility.UtilityRefused, match="--rejudge"):
        evaluate(FakeJudge(world.cand))                     # another judge's file: --rejudge is the remedy
    assert json_path.read_bytes() == before and not (output_dir() / ".aside").exists()


# --- verdict files ----------------------------------------------------------------------------------------


def test_the_verdict_file_round_trips_and_p04_reads_it(world, audited):
    from lift.validation.utility import find_verdict_files, load_verdicts

    report = evaluate(FakeJudge(world.cand))
    path = report.verdicts_path
    assert path == output_dir() / "utility" / "frontier" / CANDIDATE / "utility_hi_dev.jsonl"
    rows = list(read_jsonl(path))
    verdicts = utility.read_verdicts(path)
    assert [r["pid"] for r in rows] == PIDS and [v.model_dump() for v in verdicts] == rows
    assert b"\r\n" not in path.read_bytes()
    assert find_verdict_files() == [path]
    judge_id, found, digests = load_verdicts(CANDIDATE, set(PIDS))
    assert judge_id == JUDGE_ID and found == {v.pid: v for v in verdicts}
    assert digests == {str(path): sha256_file(path)}
    pairs = utility.pair_generations(CANDIDATE, SPLIT).pairs
    for v, pair in zip(verdicts, pairs, strict=True):
        assert (v.variant, v.judge, v.split, v.lang, v.score) == (CANDIDATE, JUDGE_ID, SPLIT, "hi", 1.0)
        assert (v.candidate_key, v.stock_key) == (pair.candidate.key, pair.stock.key)
        assert (v.candidate_sha, v.stock_sha) == (sha256_text(pair.candidate.output),
                                                  sha256_text(pair.stock.output))


def test_read_verdicts_refuses_a_file_that_is_not_one_judges_verdicts(isolated_dirs):
    path = isolated_dirs / "outputs" / "v.jsonl"

    def verdict(pid=PIDS[0], **kw):
        fields = {"pid": pid, "variant": CANDIDATE, "judge": JUDGE_ID, "order_ab": "A", "order_ba": "B",
                  "score": 1.0, **kw}
        return UtilityVerdict.model_construct(**fields).model_dump()

    cases = [
        ([verdict(), verdict(PIDS[1], judge=OTHER_JUDGE)], "mixes 2 judges"),
        ([verdict(), verdict()], "holds 1 pids twice"),
        ([verdict(score=0.75)], "not the mean of their two orders"),
        ([verdict(order_ab="A", error_ab="refused: x")], "not stored as TIE"),
        ([{**verdict(), "note": "x"}], "record 1 is not a UtilityVerdict \\(note\\)"),
    ]
    for rows, match in cases:
        write_jsonl(path, rows)
        with pytest.raises(ValueError, match=match):
            utility.read_verdicts(path)
    path.write_bytes(b"")
    assert utility.read_verdicts(path) == []
    with pytest.raises(FileNotFoundError):
        utility.read_verdicts(path.with_name("none.jsonl"))


def test_a_current_verdict_file_is_reused_with_no_judge_call(world, audited):
    judge = FakeJudge(world.cand)
    first = evaluate(judge)
    before = first.verdicts_path.read_bytes()
    second = evaluate(judge)
    assert first.judged and not second.judged and len(judge.calls) == 1 and len(judge.prefills) == 1
    assert second.verdicts_path.read_bytes() == before and second.record_path == first.record_path


def test_an_answer_made_again_is_judged_again_and_the_file_rewritten_whole(world, audited):
    judge = FakeJudge(world.cand)
    evaluate(judge)
    judge.candidate = write_side(CANDIDATE, outputs={PIDS[0]: "यह दोबारा बना उत्तर है"})
    report = evaluate(judge)
    assert report.judged and len(judge.calls) == 2
    verdicts = {v.pid: v for v in utility.read_verdicts(report.verdicts_path)}
    assert len(verdicts) == 6 and verdicts[PIDS[0]].candidate_sha == sha256_text("यह दोबारा बना उत्तर है")


def test_a_file_another_judge_wrote_is_refused_until_rejudge_sets_it_aside(world, audited):
    from lift.validation.utility import find_verdict_files

    evaluate(FakeJudge(world.cand, judge_id=OTHER_JUDGE))
    path = utility.verdicts_path("frontier", CANDIDATE, SPLIT)
    before = path.read_bytes()
    judge = FakeJudge(world.cand)
    with pytest.raises(utility.UtilityRefused, match="holds verdicts by fakeprovider/other-model") as exc:
        evaluate(judge)
    assert "--rejudge" in str(exc.value) and path.read_bytes() == before and not judge.calls
    report = evaluate(judge, rejudge=True)
    aside = report.set_aside
    assert aside.read_bytes() == before and aside.name == "utility_hi_dev.jsonl.aside"
    assert aside.is_relative_to(output_dir() / ".aside")
    assert find_verdict_files() == [path]                         # P04's scan never reads the copy
    assert {v.judge for v in utility.read_verdicts(path)} == {JUDGE_ID}


def test_an_unreadable_verdict_file_is_refused_with_the_rejudge_remedy(world, audited):
    path = utility.verdicts_path("frontier", CANDIDATE, SPLIT)
    path.parent.mkdir(parents=True)
    path.write_text('{"pid": "x"}\n', encoding="utf-8")
    with pytest.raises(utility.UtilityRefused, match="not a UtilityVerdict.*--rejudge"):
        evaluate(FakeJudge(world.cand))


# --- the frontier judge's pre-fill ------------------------------------------------------------------------


def test_the_frontier_judge_prefills_exactly_the_request_set_it_then_judges(world, audited):
    judge = FakeJudge(world.cand)
    report = evaluate(judge)
    (prefill,) = judge.prefills
    ((instructions, a, b),) = judge.calls
    assert (prefill["instructions"], prefill["a"], prefill["b"]) == (instructions, a, b)
    assert prefill["wait"] is True
    assert report.prefill["mode"] == "sync"
    manifest = next(p for p in manifests_dir().glob("utility-*.json"))
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert (doc["status"], doc["judge"], doc["n_pairs"], doc["n_requests"], doc["devices"]) == (
        "ok", JUDGE_ID, 6, 12, 0)
    assert doc["prefill_summary"]["mode"] == "sync" and doc["failed_orders"] == 0


def test_no_prefill_with_sync_or_for_the_fallback_judge(world, audited):
    judge = FakeJudge(world.cand)
    evaluate(judge, prefill=False)
    fallback = FakeJudge(world.cand)
    report = evaluate(fallback, judge_dir="fallback")
    assert not judge.prefills and not fallback.prefills and len(fallback.calls) == 1
    assert report.verdicts_path == output_dir() / "utility" / "fallback" / CANDIDATE / "utility_hi_dev.jsonl"
    assert report.record_path.parent.parent.name == "fallback"


def test_pending_batches_write_no_verdicts_and_the_manifest_says_pending(world):
    judge = FakeJudge(world.cand, pending=Pending("2 batches still running"))
    with pytest.raises(Pending):
        evaluate(judge, wait=False)
    assert judge.prefills[0]["wait"] is False and not judge.calls
    assert not utility.verdicts_path("frontier", CANDIDATE, SPLIT).exists()
    (manifest,) = manifests_dir().glob("utility-*.json")
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "pending"


# --- paths and judges -------------------------------------------------------------------------------------


def test_paths_carry_another_precision_in_their_names_and_survive_dotted_variant_names():
    fp16 = utility.verdicts_path("fallback", CANDIDATE, "final", dtype="fp16")
    assert fp16.name == "utility_hi_final-fp16.jsonl" and fp16.parent.parent.name == "fallback"
    assert utility.verdicts_path("frontier", CANDIDATE, "dev", dtype="bf16").name == "utility_hi_dev.jsonl"
    json_path, md_path = utility.metrics_paths("wiseft-l0.50-adapter-aya-std-s1", "dev")
    assert (json_path.name, md_path.name) == ("wiseft-l0.50-adapter-aya-std-s1_dev.json",
                                              "wiseft-l0.50-adapter-aya-std-s1_dev.md")
    assert json_path.parent == metrics_dir() / "utility"
    assert utility.metrics_paths(CANDIDATE, "dev", dtype="fp16")[0].name == f"{CANDIDATE}_dev-fp16.json"
    with pytest.raises(ValueError, match="one of"):
        utility.verdicts_path("primary", CANDIDATE, "dev")


def test_the_frontier_judge_gets_the_utility_entry_and_its_own_call_cache(monkeypatch):
    seen = {}

    def from_config(cls, entry, **kw):
        seen.update(entry=entry, **kw)
        return "judge"

    monkeypatch.setattr(frontier.FrontierJudge, "from_config", classmethod(from_config))
    assert utility.make_judge("frontier") == "judge"
    cache = output_dir() / "judge_cache" / "frontier" / "utility__claude-sonnet-5-5.jsonl"
    assert seen == {"entry": "utility", "cache_path": str(cache)}
    assert utility.frontier_cache_path("claude-sonnet-5-5") == cache


# --- lift utility -----------------------------------------------------------------------------------------

ARGV = ["utility", "--variant", CANDIDATE, "--split", "dev"]


@pytest.fixture
def cli(world, audited, monkeypatch):
    judge = FakeJudge(world.cand)
    monkeypatch.setattr(utility, "make_judge", lambda judge_dir, **kw: judge)
    monkeypatch.setattr(utility, "make_lid", FakeLID)
    return judge


def test_help_works_without_a_gpu_stack(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as exc:
        main(["utility", "--help"])
    assert exc.value.code == 0 and "usage" in capsys.readouterr().out


def test_the_help_says_sync_collects_an_earlier_batch_first_and_exits_3_while_it_runs(monkeypatch, capsys):
    # V07 (decisions item 52): --sync never pays twice for requests an earlier Message Batch holds. It
    # collects such a batch once it has ended, and while it still runs judges nothing and exits 3; the help
    # used to say it sends every request to the Messages API.
    monkeypatch.setenv("COLUMNS", "1000")                        # argparse wraps nothing: one line per option
    with pytest.raises(SystemExit):
        main(["utility", "--help"])
    out = capsys.readouterr().out
    option = next(line for line in out.splitlines() if line.strip().startswith("--sync"))
    text = " ".join(out.split())
    paragraph = text[text.index("--sync submits no batch"):text.index("The fallback judge")]
    for help_text in (option, paragraph):
        assert "collected first once it has ended" in help_text and "exits 3" in help_text
        assert "LIFT_ABANDON_BATCHES=<batch id>" in help_text and "cancel" in help_text
    assert "sends every request to the Messages API" not in text
    assert "send every request to the Messages API" not in text


def test_judge_only_judges_and_reports_checked_answers_with_no_gpu_stack_or_pin(cli, world, monkeypatch,
                                                                                 capsys, caplog):
    monkeypatch.setitem(sys.modules, "torch", None)              # `import torch` raises ImportError
    assert main([*ARGV, "--judge-only"]) == 0                    # it loads no model: no torch, no pin
    out = capsys.readouterr().out
    assert "win rate 1.000" in out and "failed orders 0 of 12" in out and "verdicts (judged now)" in out
    assert "outputs/metrics/auditkit_utility/frontier/adapter-aya-std-s1/utility_hi_dev.json" in out
    assert utility.verdicts_path("frontier", CANDIDATE, SPLIT).is_file()
    assert all(p.is_file() for p in utility.metrics_paths(CANDIDATE, SPLIT))
    assert main([*ARGV, "--judge-only"]) == 0 and len(cli.calls) == 1
    assert "verdicts (reused)" in capsys.readouterr().out
    for text in [*TEXTS.values(), *world.cand.values(), *world.stock.values()]:
        assert text not in out and text not in caplog.text


def test_a_run_that_may_generate_needs_the_pin_even_when_every_answer_is_there(cli, monkeypatch):
    # C06/C15/C18: answers to every prompt are no reason to skip the GPU line's check of them, which builds
    # both sides from the pinned revision; the refusal comes before any torch import.
    monkeypatch.setitem(sys.modules, "torch", None)
    for extra in ([], ["--generate-only"]):
        with pytest.raises(SystemExit, match="unpinned revisions"):
            main([*ARGV, *extra])
    assert not cli.calls


def test_incomplete_answers_need_the_pin_before_any_torch_import(world, monkeypatch):
    write_side(CANDIDATE, PIDS[:-1])
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="unpinned revisions"):
        main(ARGV)


def test_judge_only_lists_a_refused_pair_and_exits_3(cli, monkeypatch, caplog):
    write_side(CANDIDATE, PIDS[:-1])
    assert main([*ARGV, "--judge-only"]) == 3
    assert "not judged" in caplog.text and "have no answer" in caplog.text and not cli.calls


def derived_files() -> list[Path]:
    """What lift utility derives from the candidate's pair with stock (the frontier judge's): the verdicts,
    the metrics .json and .md and the AuditKIT record."""
    return [utility.verdicts_path("frontier", CANDIDATE, SPLIT), *utility.metrics_paths(CANDIDATE, SPLIT),
            audit.utility_record_path("frontier", CANDIDATE, SPLIT)]


def checks_now() -> dict[str, bytes]:
    """Both sides' recorded answer checks as they are, by variant."""
    return {name: utility.checked_path(name, SPLIT).read_bytes() for name in (CANDIDATE, "stock")}


@pytest.fixture
def gpu_line(cli, monkeypatch):
    """lift utility's GPU line with no GPU: models.global pinned to the answers' revision, and build_variant
    and generate replaced by stand-ins that record their calls. The stand-in generate leaves a file as it is,
    as the real one leaves a current complete file, and refuses the file of each variant in ``stale`` as the
    real one refuses a file made under another configuration."""
    from lift.models import variants

    config.load("project")["models"]["global"]["revision"] = META["model_rev"]
    calls = SimpleNamespace(built=[], generated=[], stale=set())

    def build(spec, *, device, dtype):
        calls.built.append(spec.name)
        return SimpleNamespace(spec=spec, dtype=dtype, device=device, model_rev=META["model_rev"],
                               adapter_hash=spec.artifacts.get("adapter"))

    def generate(variant, prompts, *, out_path, batch_size=None):
        name = variant.spec.name
        calls.generated.append((name, [p.pid for p in prompts]))
        assert out_path == generations_path(name, "utility", "hi", SPLIT)
        if name in calls.stale:
            raise gen.StaleOutputError(f"{out_path} holds records made under another configuration",
                                       command=gen.restart_command(name, "utility", "hi", SPLIT))
        return out_path

    monkeypatch.setattr(variants, "build_variant", build)
    monkeypatch.setattr(gen, "generate", generate)
    return calls


def test_every_run_but_judge_only_loads_both_sides_and_checks_their_complete_answers(gpu_line, cli):
    # C06/C15/C18: answer files holding every pid still go through generate, stock first, then the
    # candidate: it compares every record with the variant as registered now, and appends nothing to a
    # current file. The check is recorded; --generate-only stops before any judging.
    sides = ["stock", CANDIDATE]
    for name in sides:
        utility.checked_path(name, SPLIT).unlink()                # no GPU line has checked them yet
    answers = {name: generations_path(name, "utility", "hi", SPLIT).read_bytes() for name in sides}
    assert main([*ARGV, "--generate-only"]) == 0
    assert gpu_line.built == sides and gpu_line.generated == [(name, PIDS) for name in sides]
    assert not cli.calls and not utility.verdicts_path("frontier", CANDIDATE, SPLIT).exists()
    for name in sides:
        doc = json.loads(utility.checked_path(name, SPLIT).read_text(encoding="utf-8"))
        assert doc["sha256"] == hashlib.sha256(answers[name]).hexdigest()
        assert (doc["adapter_hash"], doc["intervention"]) == (ADAPTER if name == CANDIDATE else None,
                                                              intervention_digest(entry(name)))
    checks = {name: utility.checked_path(name, SPLIT).read_bytes() for name in sides}
    assert main(ARGV) == 0                                       # the whole line: loaded again, then judged
    assert gpu_line.built == sides * 2 and len(cli.calls) == 1
    assert {name: utility.checked_path(name, SPLIT).read_bytes() for name in sides} == checks


@pytest.mark.parametrize("stale", ["stock", CANDIDATE])
def test_a_side_generate_refuses_is_listed_with_its_restart_line_and_nothing_is_judged(gpu_line, cli, stale,
                                                                                       caplog):
    gpu_line.stale.add(stale)
    assert main(ARGV) == 3
    assert gpu_line.built == ["stock", CANDIDATE]               # the other side is still loaded and checked
    assert not cli.calls and not utility.metrics_paths(CANDIDATE, SPLIT)[0].exists()
    assert f"  {stale} utility_hi_dev: {RESTART.format(stale)}\n" in caplog.text
    assert "not judged, as a side's answers were skipped" in caplog.text


def test_judge_only_refuses_the_answers_of_a_replaced_adapter_and_retires_their_files(cli, registry, caplog):
    # C06/C15/C18 and C07 on the laptop: after `lift register-adapter --replace`, the old adapter's answers
    # are refused with no model loaded, and the verdicts, metrics and AuditKIT record made from them are set
    # aside and left empty, so neither Gate 1 nor P04 takes them for the new adapter's. V03: the answer
    # check is kept, and refused against the new entry on every run.
    assert main([*ARGV, "--judge-only"]) == 0 and len(cli.calls) == 1
    derived = derived_files()
    checks = checks_now()
    assert all(path.read_bytes() for path in derived) and all(checks.values())
    registry[CANDIDATE] = entry(CANDIDATE, adapter="b" * 64)
    assert main([*ARGV, "--judge-only"]) == 3 and len(cli.calls) == 1
    assert "not judged" in caplog.text and RESTART.format(CANDIDATE) in caplog.text
    assert all(path.read_bytes() == b"" for path in derived) and checks_now() == checks
    assert f"the pair is refused, so the {len(derived)} files derived from it are set aside" in caplog.text
    caplog.clear()
    assert main([*ARGV, "--judge-only"]) == 3 and len(cli.calls) == 1       # the kept check is refused again
    assert "is not the one its answers were made under" in caplog.text


def move_checkout(move: str, registry: dict, monkeypatch, *, back: bool = False) -> None:
    """Move what this checkout says the answers must be made from away from what the GPU job's ref said
    (``back``: return it, as a git pull to that ref does): the candidate's registry entry, the harness
    version or the models.global pin."""
    if move == "registry":
        registry[CANDIDATE] = entry(CANDIDATE, adapter=ADAPTER if back else "b" * 64)
    elif move == "harness":
        monkeypatch.setattr(gen, "HARNESS_VERSION", META["harness"] if back else 2)
    else:
        config.load("project")["models"]["global"]["revision"] = META["model_rev"] if back else "d" * 40


@pytest.mark.parametrize("move", ["registry", "harness", "pin"])
def test_a_refusal_a_git_pull_settles_keeps_the_answer_checks_so_judge_only_passes_again_with_no_gpu_line(
        cli, registry, monkeypatch, caplog, move):
    # V03: this checkout's registry entry, harness or pin differs from the GPU job's ref, so --judge-only
    # refuses the pair (exit 3, naming `git pull`) and retires its verdicts, metrics and AuditKIT record.
    # It never retired the answer checks, which only a GPU line can make again: once the checkout is
    # updated, --judge-only judges the pair again with no GPU line, and Gate 1 counts it.
    from lift import gates

    config.load("project")["models"]["global"]["revision"] = META["model_rev"]      # as the GPU job ran
    assert main([*ARGV, "--judge-only"]) == 0 and gates.utility_fact("aya")["status"] == "pass"
    checks = checks_now()
    move_checkout(move, registry, monkeypatch)
    caplog.clear()
    assert main([*ARGV, "--judge-only"]) == 3
    assert "this checkout is behind the ref the GPU job ran, update it (git pull)" in caplog.text
    assert all(path.read_bytes() == b"" for path in derived_files()) and checks_now() == checks
    assert gates.utility_fact("aya")["status"] == "missing"
    move_checkout(move, registry, monkeypatch, back=True)                       # git pull
    assert main([*ARGV, "--judge-only"]) == 0 and len(cli.calls) == 2          # judged again, no GPU line
    assert checks_now() == checks and gates.utility_fact("aya")["status"] == "pass"


def test_a_refusal_of_stocks_side_alone_keeps_the_candidates_check(cli, caplog):
    # V03: stock's answers unchecked here refuse the pair, and what was derived from it is retired, but the
    # candidate's check stays: once a GPU line checks stock's side, --judge-only judges the pair again with
    # no GPU line for the candidate's.
    assert main([*ARGV, "--judge-only"]) == 0
    candidate = utility.checked_path(CANDIDATE, SPLIT).read_bytes()
    utility.checked_path("stock", SPLIT).unlink()                    # no GPU line has checked stock's yet
    assert main([*ARGV, "--judge-only"]) == 3
    assert "no GPU line has checked its answers against stock as registered now" in caplog.text
    assert all(path.read_bytes() == b"" for path in derived_files())
    assert utility.checked_path(CANDIDATE, SPLIT).read_bytes() == candidate
    check("stock")                                                   # stock's GPU line, then lift sync pull
    assert main([*ARGV, "--judge-only"]) == 0 and len(cli.calls) == 2


def test_sync_skips_the_prefill_and_no_wait_exits_3_while_batches_run(cli, caplog):
    argv = [*ARGV, "--judge-only"]
    assert main([*argv, "--sync"]) == 0 and not cli.prefills and len(cli.calls) == 1
    cli.pending = Pending("1 batch still in progress")
    utility.verdicts_path("frontier", CANDIDATE, SPLIT).write_bytes(b"")      # judge them again
    assert main([*argv, "--no-wait"]) == 3
    assert cli.prefills[-1]["wait"] is False and len(cli.calls) == 1
    assert "1 batch still in progress" in caplog.text and "rerun the same line to collect" in caplog.text


def test_a_sync_run_that_meets_a_running_earlier_batch_exits_3_and_pays_nothing(cli, caplog):
    # A synchronous pass collects an earlier batch holding its requests, and stops while it still runs
    # (FrontierBatchPending): done for now, the same line collects later; never a traceback and exit 1.
    cli.sync_pending = Pending("an earlier batch holding these requests is still in progress")
    assert main([*ARGV, "--judge-only", "--sync"]) == 3
    assert not cli.prefills and not cli.calls
    assert not utility.verdicts_path("frontier", CANDIDATE, SPLIT).exists()
    assert "still in progress" in caplog.text and "rerun the same line to collect" in caplog.text
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in manifests_dir().glob("utility-*.json")]
    assert [r["status"] for r in runs] == ["pending"]


def test_a_refused_auditkit_record_exits_3_with_the_metrics_written(cli, monkeypatch, caplog):
    def refuse(*args, **kwargs):
        raise ValueError("placeholder refusal")

    monkeypatch.setattr("lift.audit.record_utility", refuse)
    assert main([*ARGV, "--judge-only"]) == 3
    assert "no AuditKIT record: placeholder refusal" in caplog.text
    assert utility.metrics_paths(CANDIDATE, SPLIT)[0].is_file()


@pytest.mark.parametrize("extra, message", [
    (["--variant", "stock"], "stock is the reference"),
    (["--generate-only", "--rejudge"], "--generate-only judges nothing"),
    (["--generate-only", "--no-wait"], "--generate-only judges nothing"),
    (["--judge", "fallback", "--sync"], "the fallback judge runs locally"),
    (["--batch-size", "0"], "--batch-size must be at least 1"),
])
def test_refused_option_combinations_exit_2(extra, message, capsys):
    assert main([*ARGV, *extra]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--generate-only", "--judge-only"], ["--sync", "--no-wait"]])
def test_exclusive_options_are_refused_by_the_parser(extra):
    with pytest.raises(SystemExit) as exc:
        main([*ARGV, *extra])
    assert exc.value.code == 2


@pytest.mark.torch
def test_the_command_generates_stock_then_the_candidate_once_then_judges_and_refuses_a_replaced_adapter(
        fake_prompt_files, tiny_model_dir, tiny_lora_adapter, audited, registry, monkeypatch, caplog):
    pytest.importorskip("peft")
    from lift.models import variants
    from lift.train.merge import adapter_hash

    # Pinned, with the tiny model standing in for Global (a local directory resolves to its content hash).
    config.load("project")["models"]["global"].update(id=str(tiny_model_dir), revision="b" * 40)
    pids = PIDS[:2] + PIDS[3:4]
    fake_prompt_files[("utility", "hi")] = [make_prompt(p, "hi", "utility", SPLIT, text=TEXTS[p])
                                            for p in pids]
    adapter = adapters_dir() / CANDIDATE
    shutil.copytree(tiny_lora_adapter, adapter)
    registry[CANDIDATE] = entry(CANDIDATE, adapter=adapter_hash(adapter))
    built, real_build, real_generate = [], variants.build_variant, gen.generate

    def build(spec, *, device, dtype):
        built.append(spec.name)
        return real_build(spec, device="cpu", dtype=dtype)

    monkeypatch.setattr(variants, "build_variant", build)
    monkeypatch.setattr(gen, "generate", lambda *a, **kw: real_generate(*a, **{**kw, "max_new_tokens": 4}))
    judge = FakeJudge({}, {p: ("TIE", "TIE") for p in pids})
    monkeypatch.setattr(utility, "make_judge", lambda judge_dir, **kw: judge)
    monkeypatch.setattr(utility, "make_lid", FakeLID)
    argv = [*ARGV, "--device", "cpu", "--dtype", "bf16"]
    assert main(argv) == 0
    assert built == ["stock", CANDIDATE] and len(judge.calls) == 1
    files = {name: generations_path(name, "utility", "hi", SPLIT) for name in ("stock", CANDIDATE)}
    for path in files.values():
        assert sorted(g.pid for g in read_generations(path)) == pids
    answers = {name: path.read_bytes() for name, path in files.items()}
    # C06/C15/C18: a rerun on complete answers loads both sides again, appends nothing, reuses the verdicts.
    assert main(argv) == 0
    assert built == ["stock", CANDIDATE] * 2 and len(judge.calls) == 1
    assert {name: path.read_bytes() for name, path in files.items()} == answers
    metrics = utility.metrics_paths(CANDIDATE, SPLIT)[0]
    reported = metrics.read_bytes()
    assert json.loads(reported)["candidate_adapter_hash"] == registry[CANDIDATE].artifacts["adapter"]
    # The adapter is trained again (other weights: here another lora_alpha) and registered with --replace.
    cfg_path = adapter / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_path.write_text(json.dumps({**cfg, "lora_alpha": 2 * cfg["lora_alpha"]}), encoding="utf-8")
    registry[CANDIDATE] = entry(CANDIDATE, adapter=adapter_hash(adapter))
    for line in ([*argv, "--generate-only"], argv):      # the GPU lines refuse the old answers, judge nothing
        caplog.clear()
        assert main(line) == 3
        assert f"  {CANDIDATE} utility_hi_dev: {RESTART.format(CANDIDATE)}\n" in caplog.text
    assert len(judge.calls) == 1 and metrics.read_bytes() == reported
    assert {name: path.read_bytes() for name, path in files.items()} == answers
    # ... and so does the laptop's --judge-only, which loads no model; it retires the old adapter's metrics.
    assert main([*ARGV, "--judge-only"]) == 3
    assert len(judge.calls) == 1 and metrics.read_bytes() == b""


SECTION_B = hasattr(frontier, "parse_pairwise_reply") and hasattr(frontier.FrontierJudge, "prefill_pairwise")


@pytest.mark.skipif(not SECTION_B, reason="FrontierJudge.pairwise and prefill_pairwise come with section B")
def test_the_frontier_judge_over_fakeprovider_end_to_end(world, audited):
    from lift.judge.frontier import FakeProvider, FrontierJudge
    from lift.validation.utility import load_verdicts

    candidate = set(world.cand.values())

    def responder(call):
        shown_a = re.search(r"RESPONSE A:\n<<<\n(.*?)\n>>>", call["user"], re.S).group(1)
        return {"winner": "A" if shown_a in candidate else "B", "reason": "placeholder reason"}

    cache = utility.frontier_cache_path("claude-sonnet-5-5")
    judge = FrontierJudge.from_config("utility", client=FakeProvider(responder), workers=1,
                                      cache_path=str(cache))
    report = utility.evaluate(CANDIDATE, SPLIT, judge=judge, lid=FakeLID())
    assert report.metrics["win_rate"] == 1.0 and report.metrics["failed_orders"] == 0
    assert report.prefill["mode"] == "sync" and cache.is_file()
    judge_id, found, _ = load_verdicts(CANDIDATE, set(PIDS))
    assert judge_id == judge.judge_id and judge_id.startswith("anthropic/claude-sonnet-5-5@utility_v1:")
    assert len(found) == 6


# --- the fallback judge (lift.judge.local_pairwise) -------------------------------------------------------

_PAIR_FIELD = re.compile(r"\{(instruction|response_a|response_b)\}")


@dataclass(frozen=True)
class StandInRubric:
    path: Path
    sha256: str
    system: str
    user_template: str
    kind: str

    @property
    def version(self) -> str:
        return f"{self.path.stem}:{self.sha256[:12]}"

    def render_pair(self, *, instruction: str, response_a: str, response_b: str) -> str:
        values = {"instruction": instruction, "response_a": response_a, "response_b": response_b}
        return _PAIR_FIELD.sub(lambda m: values[m.group(1)], self.user_template)


def _stand_in_load_rubric(path):
    p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    data = p.read_bytes()
    head, template = data.decode("utf-8").split("## User message template")
    system = head.split("## System message")[1].strip()
    kind = "pairwise" if "{response_a}" in template else "safety"
    return StandInRubric(p, hashlib.sha256(data).hexdigest(), system, template.strip(), kind)


def _stand_in_parse(text):
    def unique(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("duplicate key")
        return dict(pairs)

    try:
        obj = json.loads(text.strip(), object_pairs_hook=unique)
    except ValueError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"winner", "reason"}:
        return None
    if obj["winner"] not in ("A", "B", "TIE") or not isinstance(obj["reason"], str):
        return None
    return obj["winner"], obj["reason"]


@pytest.fixture
def pairwise_rubric(monkeypatch):
    """lift.judge.frontier's pairwise rubric (load_rubric of utility_v1.md, kind "pairwise", render_pair) and
    parse_pairwise_reply come with section B, written in parallel. Until they are on this branch, stand-ins
    that follow its contract; once they are, the real functions (nothing is replaced)."""
    if not hasattr(frontier, "parse_pairwise_reply"):
        monkeypatch.setattr(frontier, "load_rubric", _stand_in_load_rubric)
        monkeypatch.setattr(frontier, "parse_pairwise_reply", _stand_in_parse, raising=False)


class StubModel:
    """All the fallback judge reads from a model when greedy_batch is replaced: the context size."""

    def __init__(self, context: int = 4096) -> None:
        self.config = SimpleNamespace(max_position_embeddings=context)


def verdict_reply(winner: str) -> str:
    return json.dumps({"winner": winner, "reason": "placeholder reason"})


class Canned:
    """Stands in for greedy_batch in lift.judge.local_pairwise. Each row's reply is ``first[k]``, k being the
    marker #k# in its instruction, or ``second[k]`` once the row holds the reminder turn, encoded with the
    tiny tokenizer (``special``: k whose first reply also holds <|END_RESPONSE|>). Records each call's
    decoded rows; ``oom_above`` raises CUDA's out-of-memory error for larger batches."""

    def __init__(self, tok, first, second=None, *, special=(), oom_above=None) -> None:
        self.tok, self.first, self.second = tok, first, second or {}
        self.special, self.oom_above = set(special), oom_above
        self.calls: list[list[str]] = []

    def __call__(self, model, rows, **kwargs):
        torch = pytest.importorskip("torch")
        if self.oom_above is not None and len(rows) > self.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB (mock)")
        texts = [self.tok.decode(row) for row in rows]
        self.calls.append(texts)
        outs = []
        for text in texts:
            k = int(re.findall(r"#(\d+)#", text)[0])
            again = local_pairwise.REMINDER in text
            ids = self.tok((self.second if again else self.first)[k], add_special_tokens=False)["input_ids"]
            if k in self.special and not again:
                ids = [*ids, self.tok.convert_tokens_to_ids("<|END_RESPONSE|>")]
            outs.append(GreedyOutput(ids=ids, stopped=True, n_generated=len(ids) + 1))
        return outs


def local_judge(tok, *, context: int = 4096, **kw):
    """The fallback judge over the tiny tokenizer and a model stand-in (greedy_batch is replaced)."""
    kw = {"device": "cpu", "dtype": "bf16", "model": StubModel(context), "tokenizer": tok, **kw}
    return local_pairwise.LocalPairwiseJudge("Qwen/Qwen3-8B", REV, RUBRIC, **kw)


def pairs(n):
    return [f"placeholder instruction #{k}#" for k in range(n)], [f"answer a{k}" for k in range(n)], \
        [f"answer b{k}" for k in range(n)]


@pytest.fixture
def tok():
    pytest.importorskip("torch")
    return build_tiny_tokenizer()


@pytest.mark.torch
def test_the_fallback_judge_sends_the_rubric_through_the_chat_template_and_parses(pairwise_rubric, tok,
                                                                                  monkeypatch):
    seen = []
    real = tok.apply_chat_template

    def spy(conversation, **kwargs):
        seen.append((conversation, kwargs))
        return real(conversation, **kwargs)

    monkeypatch.setattr(tok, "apply_chat_template", spy)
    canned = Canned(tok, {0: verdict_reply("A"), 1: verdict_reply("TIE"), 2: verdict_reply("B")})
    monkeypatch.setattr(local_pairwise, "greedy_batch", canned)
    judge = local_judge(tok)
    out = judge.pairwise(*pairs(3))
    assert [r["winner"] for r in out] == ["A", "TIE", "B"] and {r["error"] for r in out} == {None}
    assert out[0] == {"winner": "A", "reason": "placeholder reason", "raw": verdict_reply("A"), "error": None,
                      "model_echo": f"Qwen/Qwen3-8B@{REV}"}
    rubric = frontier.load_rubric(RUBRIC)
    user = rubric.render_pair(instruction="placeholder instruction #0#", response_a="answer a0",
                              response_b="answer b0")
    conversation, kwargs = seen[0]
    assert conversation == [{"role": "system", "content": rubric.system}, {"role": "user", "content": user}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert judge.stats["pairs"] == 3 and judge.stats["reminders"] == 0 and judge.stats["replies"] == 3


@pytest.mark.torch
def test_an_unparseable_reply_gets_one_reminder_turn_then_is_a_failure(pairwise_rubric, tok, monkeypatch):
    canned = Canned(tok, {0: "Winner: A", 1: "not json", 2: verdict_reply("A")},
                    {0: verdict_reply("B"), 1: '{"winner": "a", "reason": "lowercase"}'})
    monkeypatch.setattr(local_pairwise, "greedy_batch", canned)
    judge = local_judge(tok)
    out = judge.pairwise(*pairs(3))
    assert [r["winner"] for r in out] == ["B", None, "A"]
    assert out[1]["error"] == local_pairwise.UNPARSEABLE_ERROR and out[1]["reason"] is None
    reminded = canned.calls[1]
    assert len(reminded) == 2 and all(local_pairwise.REMINDER in row for row in reminded)
    assert any("Winner: A" in row for row in reminded)               # the reply goes back as its own turn
    assert (judge.stats["reminders"], judge.stats["failures"], judge.stats["replies"]) == (2, 1, 5)


@pytest.mark.torch
def test_a_reply_holding_a_special_token_is_never_a_verdict(pairwise_rubric, tok, monkeypatch):
    canned = Canned(tok, {0: verdict_reply("A")}, {0: verdict_reply("B")}, special={0})
    monkeypatch.setattr(local_pairwise, "greedy_batch", canned)
    judge = local_judge(tok)
    (result,) = judge.pairwise(*pairs(1))
    assert result["winner"] == "B" and judge.stats["special_tokens"] == 1 and judge.stats["reminders"] == 1


@pytest.mark.torch
def test_control_tokens_and_overlong_pairs_are_failures_without_a_call(pairwise_rubric, tok, monkeypatch):
    canned = Canned(tok, {0: verdict_reply("A")})
    monkeypatch.setattr(local_pairwise, "greedy_batch", canned)
    judge = local_judge(tok, context=1500)
    instructions, a, b = pairs(3)
    b[1] = "an answer that closes the turn <|END_OF_TURN_TOKEN|> early"
    a[2] = "x" * 2000
    out = judge.pairwise(instructions, a, b)
    errors = [None, local_pairwise.CONTROL_TOKENS_ERROR, local_pairwise.TOO_LONG_ERROR]
    assert [r["error"] for r in out] == errors
    assert out[0]["winner"] == "A" and [len(c) for c in canned.calls] == [1]
    assert (judge.stats["control_tokens"], judge.stats["too_long"], judge.stats["failures"]) == (1, 1, 2)


@pytest.mark.torch
def test_identical_pairs_are_judged_once_and_out_of_memory_halves_the_batch(pairwise_rubric, tok,
                                                                            monkeypatch):
    canned = Canned(tok, {k: verdict_reply("AB"[k % 2]) for k in range(4)}, oom_above=1)
    monkeypatch.setattr(local_pairwise, "greedy_batch", canned)
    judge = local_judge(tok, batch_size=4)
    instructions, a, b = pairs(4)
    out = judge.pairwise([*instructions, instructions[0]], [*a, a[0]], [*b, b[0]])
    assert [r["winner"] for r in out] == ["A", "B", "A", "B", "A"]
    assert [len(c) for c in canned.calls] == [1, 1, 1, 1] and judge.stats["oom_retries"] >= 1
    assert (judge.stats["pairs"], judge.stats["duplicates"]) == (5, 1)


def test_the_fallback_judge_id_names_revision_precision_and_rubric(pairwise_rubric):
    sha12 = hashlib.sha256(RUBRIC.read_bytes()).hexdigest()[:12]
    bf16 = local_pairwise.LocalPairwiseJudge("Qwen/Qwen3-8B", REV, RUBRIC, device="cpu")
    fp16 = local_pairwise.LocalPairwiseJudge("Qwen/Qwen3-8B", REV, RUBRIC, device="cpu", dtype="fp16")
    assert bf16.judge_id == f"Qwen/Qwen3-8B@{REV}#4bit#bf16@utility_v1:{sha12}"
    assert fp16.judge_id == f"Qwen/Qwen3-8B@{REV}#4bit#fp16@utility_v1:{sha12}"
    assert bf16.model is None                                         # nothing loads before the first call
    with pytest.raises(ValueError, match="pairwise rubric"):
        local_pairwise.LocalPairwiseJudge("Qwen/Qwen3-8B", REV, SAFETY_RUBRIC, device="cpu")
    with pytest.raises(ValueError, match="commit sha"):
        local_pairwise.LocalPairwiseJudge("Qwen/Qwen3-8B", "main", RUBRIC, model=StubModel(),
                                          tokenizer=SimpleNamespace(pad_token_id=0))


def test_the_fallback_judge_comes_from_judges_utility_fallback(pairwise_rubric):
    fallback = config.load("project")["judges"]["utility"]["fallback"]
    fallback["revision"] = REV
    judge = local_pairwise.LocalPairwiseJudge.from_config(device="cpu", dtype="bf16")
    assert judge.judge_id.startswith(f"Qwen/Qwen3-8B@{REV}#4bit#bf16@utility_v1:") and judge.model is None
    assert judge.info["quantization"] == "bitsandbytes nf4" and judge.info["double_quant"] is False
    fallback["quantization"] = "8bit"
    with pytest.raises(ValueError, match="only '4bit'"):
        local_pairwise.LocalPairwiseJudge.from_config(device="cpu")
    fallback.update(quantization="4bit", temperature=0)
    with pytest.raises(ValueError, match="unknown keys \\['temperature'\\]"):
        local_pairwise.LocalPairwiseJudge.from_config(device="cpu")


@pytest.mark.torch
def test_the_fallback_judges_verdicts_go_to_their_own_file_and_p04_reads_them(pairwise_rubric, tok, world,
                                                                              audited, monkeypatch):
    from lift.validation.utility import find_verdict_files, load_verdicts

    def decide(model, rows, **kwargs):            # a judge that always picks position A
        ids = tok(verdict_reply("A"), add_special_tokens=False)["input_ids"]
        return [GreedyOutput(ids=ids, stopped=True, n_generated=len(ids) + 1) for _ in rows]

    monkeypatch.setattr(local_pairwise, "greedy_batch", decide)
    judge = local_judge(tok)
    report = utility.evaluate(CANDIDATE, SPLIT, judge_dir="fallback", judge=judge, lid=FakeLID())
    assert report.metrics["win_rate"] == 0.5 and report.metrics["position"]["a_share_of_decisive"] == 1.0
    assert find_verdict_files() == [report.verdicts_path] and "fallback" in report.verdicts_path.parts
    judge_id, found, _ = load_verdicts(CANDIDATE, set(PIDS))
    assert judge_id == judge.judge_id and len(found) == 6
    (manifest,) = manifests_dir().glob("utility-*.json")
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["devices"] == 1 and doc["judge_info"]["revision"] == REV
    assert doc["judge_stack"]["gpu_capability"] == "cpu"
    assert doc["judge_stats"]["pairs"] == 12 and doc["judge_stats"]["reminders"] == 0
