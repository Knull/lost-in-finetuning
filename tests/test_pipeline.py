"""lift.judge.pipeline offline: fake judges and a fake LID, generation records written with lift.io.
Placeholder text only."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import random
import re
import sys
import types
from pathlib import Path

import pytest
from fakes import make_prompt

from lift import config, gate0
from lift.cachekey import intervention_digest
from lift.io import read_jsonl, sha256_text, stable_json, write_jsonl
from lift.judge import labels, pipeline
from lift.judge.frontier import load_rubric
from lift.judge.pipeline import JudgeItem, JudgeOutcome
from lift.models import generate, variants
from lift.models.generate import OtherPrecisionError, StaleOutputError, generations_path, restart_command
from lift.paths import REPO_ROOT, manifests_dir, metrics_dir, validation_dir
from lift.schemas import GenerationRecord

CELL = ("stock", "harmful", "en", "dev")
LONG = "placeholder words that make a long enough answer"
GUARD_ID = "Qwen/Qwen3Guard-Gen-4B@" + "a" * 40
RULES = labels.rules_tag()                      # the labelling rules of configs/project.yaml
PRIMARY = f"{GUARD_ID}#{RULES}#bf16#strict"     # the configured primary judge, in bf16 and strict mode
# What FrontierAdapter names the second judge that ``second_judge`` configures: the rubric's version, OpenAI's
# default sampling (temperature 0, 1024 tokens), the API version its SDK speaks and the labelling rules.
SECOND = (f"openai/placeholder-model@{load_rubric(config.get('judges.safety_second.rubric')).version}"
          f"+t0+max1024+apichat.completions:developer#{RULES}")
REV = "c" * 40                                  # Global's commit sha, as generate records it in meta
STACK = {"torch": "2.15.0", "transformers": "5.22.1", "gpu_capability": "8.0"}      # an A100
L4 = {**STACK, "gpu_capability": "8.9"}
T4_STACK = {**STACK, "gpu_capability": "7.5"}               # the T4 the fp16 licence was compared on
DIGEST = "e" * 64          # gate0.json's checks.precision.digest: the bf16 run the comparison paired with
DECODING = "d" * 16
COMPARED = {"reference_digest": DIGEST, "harness": 1, "decoding": DECODING}     # what the comparison ran with
T4_META = {"model_rev": REV, "harness": 1, "decoding": DECODING, "gpu_capability": "7.5"}  # a T4 generation
T4_JUDGE = f"{GUARD_ID}#{RULES}#fp16#strict"    # the judge the T4 comparison ran: the primary judge in fp16


@pytest.fixture(autouse=True)
def quick_manifests(monkeypatch):
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")


# The adapter variants the tests' cells name, as P03 registers them in configs/variants.yaml: with the
# adapter's hash pinned (lift register-adapter writes artifacts.adapter), as a cell of an adapter variant is
# read only then. Both name one adapter, so both pin one hash (W3-03: a cell of a variant that names an
# adapter is refused while another entry of method adapter naming it pins another hash).
ADAPTER_HASH = "a" * 64
TEST_VARIANTS = [{"name": name, "method": "adapter", "adapter": "adapter-aya-std-s1", "seed": 1,
                  "artifacts": {"adapter": ADAPTER_HASH}} for name in ("adapter", "adapter-aya-std-s1")]


@pytest.fixture(autouse=True)
def registry(unpinned_revisions, monkeypatch):
    """configs/variants.yaml with the tests' adapter variants beside its own (the returned registry, which a
    test may change): a cell is read only for a registered variant, under its current entry."""
    load = config.load
    doc = {"variants": [*copy.deepcopy(load("variants")["variants"]), *copy.deepcopy(TEST_VARIANTS)]}
    monkeypatch.setattr(config, "load", lambda name="project": doc if name == "variants" else load(name))
    return doc


@pytest.fixture(autouse=True)
def judge_stack(monkeypatch):
    """lift.models.generate.stack without a GPU: returns ``state["stack"]`` (an A100 unless a test moves
    the judge) and records the devices it was asked about."""
    state = {"stack": dict(STACK), "devices": []}

    def stack(device):
        state["devices"].append(device)
        return dict(state["stack"])

    monkeypatch.setattr("lift.models.generate.stack", stack)
    return state


@pytest.fixture
def second_judge(monkeypatch):
    """judges.safety_second with a placeholder OpenAI model; the id its judge writes (``SECOND``)."""
    entry = config.load("project")["judges"]["safety_second"]
    monkeypatch.setitem(entry, "provider", "openai")
    monkeypatch.setitem(entry, "model", "placeholder-model")
    return SECOND


class FakeLID:
    """Language by marker: 'hindi' -> hin_Deva, 'noise' -> garbled, otherwise English."""

    def __init__(self):
        self.calls = 0

    def top(self, text, k=3):
        assert text.strip(), "LID must not run on an empty output"
        self.calls += 1
        if "noise" in text:
            return [("eng_Latn", 0.2), ("hin_Deva", 0.1)][:k]
        if "hindi" in text:
            return [("hin_Deva", 0.95), ("eng_Latn", 0.03)][:k]
        return [("eng_Latn", 0.99)][:k]


class FakeJudge:
    """Label by marker in the response: HARM, REFUSE, FAIL (a judge failure), otherwise benign."""

    def __init__(self, judge_id="fake-judge@0001", short="fake", crash_on_call=None):
        self.judge_id = judge_id
        self.short = short
        self.crash_on_call = crash_on_call
        self.calls: list[list[str]] = []

    def judge(self, items):
        self.calls.append([i.key for i in items])
        if len(self.calls) == self.crash_on_call:
            raise RuntimeError("simulated crash")
        out = []
        for i in items:
            if "FAIL" in i.response:
                out.append(JudgeOutcome("unclear", {"judge_failure": "unparseable"}, "garbage"))
            elif "HARM" in i.response:
                out.append(JudgeOutcome("harmful", {"call": "harm"}, "raw"))
            elif "REFUSE" in i.response:
                out.append(JudgeOutcome("refusal", {"call": "refuse"}, "raw"))
            else:
                out.append(JudgeOutcome("benign", {"call": "ok"}, "raw"))
        return out


def primary_judge(judge_id=PRIMARY):
    """A fake judge writing into the primary role's judgements directory."""
    return FakeJudge(judge_id, short=pipeline.judge_dir("primary"))


def gen(pid, *, output=None, lang="en", set_name="harmful", split="dev", variant="stock", tag="v1",
        prompt=None, n_new=None, dtype="bf16", meta=None):
    """A GenerationRecord as lift.models.generate writes it: in ``dtype``, by this code's harness, under the
    cap configs/project.yaml gives, under the variant's entry in configs/variants.yaml when it has one, by the
    adapter it pins (``meta`` overrides)."""
    prompt = prompt if prompt is not None else f"prompt {pid} {lang}"      # = make_prompt's default text
    output = output if output is not None else f"{LONG} {pid}"
    made = {"dtype": dtype, "harness": generate.HARNESS_VERSION,
            "max_new_tokens": config.get("generation.max_new_tokens")}
    if variant in variants.load_registry():
        spec = variants.get_spec(variant)
        made["intervention"] = intervention_digest(spec)
        if spec.artifacts.get("adapter"):
            made["adapter_hash"] = spec.artifacts["adapter"]
    return GenerationRecord(
        key=sha256_text(f"{tag}|{variant}|{pid}|{lang}"), variant=variant, pid=pid, lang=lang, set=set_name,
        split=split, prompt_sha=sha256_text(prompt), output=output, n_prompt_tokens=12,
        n_new_tokens=len(output.split()) if n_new is None else n_new, hit_cap=False, seconds=0.01,
        meta={**made, **(meta or {})},
    )


def write_gens(records, path=None):
    r = records[0]
    path = path or generations_path(r.variant, r.set, r.lang, r.split)
    write_jsonl(path, [g.model_dump() for g in records])
    return path


def setup_cell(store, n=6, outputs=None, dtype="bf16", meta=None):
    """n prompts in the fake prompt store and one generation each; returns (pids, gens, path)."""
    pids = [f"prh-test-{i:04d}" for i in range(n)]
    store[("harmful", "en")] = [make_prompt(p) for p in pids]
    gens = [gen(p, output=(outputs or {}).get(p), dtype=dtype, meta=meta) for p in pids]
    return pids, gens, write_gens(gens)


def write_licence(dtype="fp16", *, passed=True, fp16=True, rev=REV, name=None, reference=True, ref_rev=REV,
                  digest=DIGEST, gpu="7.5", judge=T4_JUDGE):
    """What Gate 0's comparison run (lift smoke --compare-dtype) writes to outputs/metrics/gate0-{dtype}.json,
    and the bf16 reference's own Gate 0 (gate0.json; ``reference`` None leaves it out), reduced to the
    fields the licence reads. ``fp16`` is the fp16 check's pass flag on a comparison of 200 prompts, or
    the whole check; either way it records ``COMPARED`` unless it says otherwise. ``digest`` is gate0.json's
    checks.precision.digest (the comparison paired with ``DIGEST``), ``gpu`` the GPU class the comparison
    ran on, ``judge`` the id of the judge it ran (Gate 0's judge.id; gate0.json's names it in bf16, as the
    comparison pairs only runs of one judge but for the precision); None leaves either out."""
    fp16 = {**COMPARED, **fp16} if isinstance(fp16, dict) else {"pass": fp16, "comparable": True, "n": 200,
                                                                  **COMPARED}
    doc = {"dtype": dtype, "reference_dtype": "bf16", "pass": passed,
           **({"gpu_capability": gpu} if gpu else {}), **({"judge": {"id": judge}} if judge else {}),
           "checks": {"precision": {"pass": True, "digest": "f" * 64}, "fp16": fp16},
           "revisions": {"global": rev, "judge": "a" * 40}}
    metrics_dir().mkdir(parents=True, exist_ok=True)
    if reference is not None:
        precision = {"pass": True, **({"digest": digest} if digest else {})}
        ref = {"dtype": "bf16", "pass": reference, "checks": {"precision": precision},
               **({"judge": {"id": judge.replace(f"#{dtype}#", "#bf16#")}} if judge else {}),
               "revisions": {"global": ref_rev, "judge": "a" * 40}}
        (metrics_dir() / "gate0.json").write_text(json.dumps(ref), encoding="utf-8")
    path = metrics_dir() / f"{name or f'gate0-{dtype}'}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def judged(path):
    return {r["pid"]: r for r in read_jsonl(path)}


def manifests():
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in manifests_dir().glob("judge-*.json")]
    return sorted(runs, key=lambda m: m["started"])


def p04_draw(pids, set_name, split, fraction, seed):
    """P04's second-judge draw (lift.validation.second_judge.chosen_pids), written out on its own."""
    def rank(pid):
        return hashlib.sha256(f"{seed}:second:{set_name}:{split}:{pid}".encode()).hexdigest()

    return set(sorted(set(pids), key=rank)[:math.ceil(fraction * len(set(pids)))])


def test_every_generation_gets_exactly_one_judgement(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files, n=10, outputs={
        "prh-test-0001": f"HARM {LONG}", "prh-test-0002": f"REFUSE {LONG}", "prh-test-0003": f"noise {LONG}",
        "prh-test-0004": f"hindi {LONG}"})
    judge, lid = FakeJudge(), FakeLID()
    out = pipeline.judge_file(path, judge=judge, lid=lid, batch_size=3)

    assert out == pipeline.judgements_path("fake", *CELL)
    rows = list(read_jsonl(out))
    assert sorted(r["key"] for r in rows) == sorted(g.key for g in gens)
    assert {r["key"]: r["output_sha"] for r in rows} == {g.key: sha256_text(g.output) for g in gens}
    assert [len(c) for c in judge.calls] == [3, 3, 3, 1]
    assert lid.calls == 10
    by_pid = judged(out)
    assert by_pid["prh-test-0001"]["label"] == "harmful"
    assert by_pid["prh-test-0002"]["label"] == "refusal"
    assert by_pid["prh-test-0003"]["label"] == "unclear" and by_pid["prh-test-0003"]["flags"]["garbled"]
    assert by_pid["prh-test-0004"]["flags"]["wrong_lang"] and by_pid["prh-test-0004"]["label"] == "benign"
    assert by_pid["prh-test-0000"]["judge"] == "fake-judge@0001"
    assert by_pid["prh-test-0000"]["flags"]["lid_top"] == [["eng_Latn", 0.99]]
    [m] = manifests()
    counts = ("n_generations", "n_selected", "n_judged", "n_resumed", "n_failures")
    assert [m[k] for k in counts] == [10, 10, 10, 0, 0]
    assert m["label_counts"] == {"benign": 7, "harmful": 1, "refusal": 1, "unclear": 1}
    assert m["status"] == "ok" and m["selection"] == {"mode": "all"} and m["judge"] == "fake-judge@0001"


def test_resume_after_a_crash_writes_no_duplicates(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files, n=10)
    out = pipeline.judgements_path("fake", *CELL)
    with pytest.raises(RuntimeError, match="simulated crash"):
        pipeline.judge_file(path, judge=FakeJudge(crash_on_call=2), lid=FakeLID(), batch_size=4)
    assert len(list(read_jsonl(out))) == 4
    with open(out, "a", encoding="utf-8", newline="\n") as f:      # killed in the middle of a line
        f.write('{"key": "trunc')

    judge = FakeJudge()
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), batch_size=4)
    keys = [r["key"] for r in read_jsonl(out)]
    assert len(keys) == len(set(keys)) == 10 and set(keys) == {g.key for g in gens}
    assert sum(len(c) for c in judge.calls) == 6
    failed, ok = manifests()
    assert failed["status"] == "failed" and failed["error"] == "RuntimeError" and failed["n_judged"] == 4
    assert ok["n_resumed"] == 4 and ok["n_judged"] == 6 and sum(ok["label_counts"].values()) == 10

    again = FakeJudge()
    pipeline.judge_file(path, judge=again, lid=FakeLID())
    assert again.calls == [] and len(list(read_jsonl(out))) == 10
    with pytest.raises(FileExistsError):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), resume=False)


RESTART = "lift restart --variant stock --split dev --sets harmful --langs en"      # the cell of CELL


def assert_restart(err, command):
    """A stale cell's refusal (contract C): a StaleOutputError naming the exact ``lift restart`` line of the
    cell, to run right after the job's ``lift sync pull`` (a file moved aside by hand comes back with that
    pull), and no record text."""
    message = str(err)
    assert isinstance(err, pipeline.StaleJudgementsError) and isinstance(err, StaleOutputError)
    assert err.command == command and f"`{command}`" in message
    assert ("--judge" in command) == ("--judge" in message)
    assert "right after `lift sync pull`" in message and "move the judgements file aside" not in message
    assert LONG not in message and "prompt prh-" not in message


def test_a_changed_output_under_the_same_key_is_never_resumed(fake_prompt_files):
    """The key hashes the inputs only. A regenerated output (bf16 batch effects, a lost and rebuilt
    file) under the same key must be re-judged, never inherit the old output's label. The generations
    are current, so the refusal names the role's restart, which keeps them (contract C)."""
    pids, gens, path = setup_cell(fake_prompt_files, n=4)
    out = pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    flipped = [gen(p, output=f"HARM {LONG} {p}") for p in pids]
    assert [g.key for g in flipped] == [g.key for g in gens]
    write_gens(flipped)
    again = primary_judge()
    with pytest.raises(pipeline.StaleJudgementsError, match="generations changed since judging") as refused:
        pipeline.judge_file(path, judge=again, lid=FakeLID())
    assert again.calls == []
    assert_restart(refused.value, RESTART + " --judge primary")
    assert "the generations are kept" in str(refused.value)

    write_gens(gens)                                     # a judgement naming no output is not trusted either
    rows = list(read_jsonl(out))
    write_jsonl(out, [{k: v for k, v in r.items() if k != "output_sha"} for r in rows])
    with pytest.raises(StaleOutputError, match="changed since judging"):
        pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())


def test_a_judge_failure_is_unclear_and_counted(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files, outputs={"prh-test-0002": f"FAIL {LONG}"})
    out = pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())
    rec = judged(out)["prh-test-0002"]
    assert rec["label"] == "unclear" and rec["judge_label"] == "unclear"
    assert rec["native"]["judge_failure"] == "unparseable" and rec["raw"] == "garbage"
    [m] = manifests()
    assert m["n_judged"] == 6 and m["n_failures"] == 1 and m["n_failures_selected"] == 1
    assert m["label_counts"] == {"benign": 5, "unclear": 1}


def test_a_failure_with_a_real_label_is_refused(fake_prompt_files):
    class Liar(FakeJudge):
        def judge(self, items):
            return [JudgeOutcome("benign", {"judge_failure": "timeout"}, "") for _ in items]

    _, _, path = setup_cell(fake_prompt_files)
    with pytest.raises(ValueError, match="judge failure must carry"):
        pipeline.judge_file(path, judge=Liar(), lid=FakeLID())


def test_judges_are_never_mixed_in_one_file(fake_prompt_files):
    """A MixedJudgesError is a StaleOutputError, so run-dev and lift judge list the cell and go on. No
    restart line covers a file no role owns."""
    _, gens, path = setup_cell(fake_prompt_files)
    pipeline.judge_file(path, judge=FakeJudge("judge-a@1"), lid=FakeLID(), keys=[gens[0].key])
    with pytest.raises(pipeline.MixedJudgesError, match="never mixed") as refused:
        pipeline.judge_file(path, judge=FakeJudge("judge-b@1"), lid=FakeLID())
    assert isinstance(refused.value, StaleOutputError) and refused.value.command is None
    assert "by hand" in str(refused.value) and "lift restart" not in str(refused.value)
    assert len(list(read_jsonl(pipeline.judgements_path("fake", *CELL)))) == 1


def test_a_file_begun_by_another_judge_names_both_remedies(fake_prompt_files):
    """Review 4, W10 (contract I): 'judges are never mixed' was a plain ValueError that stopped run-dev at
    the cell, leaving every later cell unjudged, and named no way out. It is a MixedJudgesError whose command
    is the role's restart --judge line, and whose message also names the other remedy: the file's own judge
    (its id shows its precision and Controversial mode)."""
    _, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    pipeline.judge_file(path, judge=guard(strict=False), lid=FakeLID(), keys=[gens[0].key])    # begun loose
    strict = guard()
    with pytest.raises(pipeline.MixedJudgesError) as refused:
        pipeline.judge_file(path, judge=strict, lid=FakeLID())
    err, line = refused.value, RESTART + " --judge primary"
    assert isinstance(err, StaleOutputError) and err.command == line
    message = str(err)
    loose = f"{GUARD_ID}#{RULES}#bf16#loose"
    assert f"holds judgements by ['{loose}'], not {PRIMARY}; judges are never mixed" in message
    assert "finish the file with its own judge (its id names" in message
    assert f"run `{line}` as a job line right after `lift sync pull`" in message and LONG not in message
    assert strict.guard.stats["pairs"] == 0

    out = pipeline.judgements_path("qwen3guard", *CELL)         # judged by a T4 before judge_file checked it
    write_jsonl(out, [{**r, "judge": f"{GUARD_ID}#{RULES}#fp16#strict"} for r in read_jsonl(out)])
    with pytest.raises(pipeline.MixedJudgesError) as refused:
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    message = str(refused.value)
    assert "judged in ['fp16'] of generations made in bf16, so they can never be reported" in message
    assert f"run `{line}`" in message and "its own judge" not in message and refused.value.command == line


def test_duplicate_or_stale_judgements_raise(fake_prompt_files, tmp_path):
    pids, gens, path = setup_cell(fake_prompt_files)
    out = pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines + lines[:1]), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())

    out.write_text("".join(lines), encoding="utf-8", newline="\n")
    write_gens(gens[:-1] + [gen(pids[-1], tag="v2")])          # one output regenerated under a new key
    with pytest.raises(pipeline.StaleJudgementsError, match="not in the generations file") as refused:
        pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert_restart(refused.value, RESTART + " --judge primary")     # the role's judgements only

    other = pipeline.judgements_path("fake", *CELL)            # no role's dir: the whole cell's restart
    other.parent.mkdir(parents=True)
    other.write_bytes(out.read_bytes())
    with pytest.raises(StaleOutputError) as refused:
        pipeline.judge_file(path, judge=FakeJudge(PRIMARY), lid=FakeLID())
    assert_restart(refused.value, RESTART)
    elsewhere = tmp_path / "elsewhere.jsonl"                    # no restart line covers it
    elsewhere.write_bytes(out.read_bytes())
    with pytest.raises(StaleOutputError, match="by hand") as refused:
        pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID(), out_path=elsewhere)
    assert refused.value.command is None


def test_prompt_changed_after_generation_raises(fake_prompt_files):
    """A translation fixed after generation leaves a stale cell (contract C): its generations are made
    again, so the refusal names the restart of the whole cell, as generate's own refusal does."""
    pids, _, path = setup_cell(fake_prompt_files)
    fake_prompt_files[("harmful", "en")][-1] = make_prompt(pids[-1], text="a corrected translation")
    with pytest.raises(StaleOutputError, match=pids[-1]) as refused:
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())
    assert_restart(refused.value, RESTART)
    assert "corrected" not in str(refused.value)
    assert not pipeline.judgements_path("fake", *CELL).exists()
    with pytest.raises(StaleOutputError, match="no prompt text"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), prompts_by_pid={})


def test_an_emptied_cell_is_judged_from_the_start(fake_prompt_files):
    """``lift restart`` leaves an empty file at every path it sets aside, so the next push replaces the
    Hub's copy (contract A). An empty judgements file is judged from the start; an empty generations file
    is said to need generating first."""
    _, _, path = setup_cell(fake_prompt_files, n=4)
    out = pipeline.judgements_path("qwen3guard", *CELL)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"")
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert len(pipeline.load_records(*CELL)) == 4
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="holds no generations.*lift generate"):
        pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())


def test_one_variant_set_lang_split_per_file(fake_prompt_files, tmp_path):
    path = write_gens([gen("prh-test-0000"), gen("prh-test-0001", lang="hi")], tmp_path / "mixed.jsonl")
    with pytest.raises(ValueError, match="one variant/set/lang/split"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())


def test_empty_output_gets_und_without_a_lid_call(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files)
    gens[1] = gen(pids[1], output="", n_new=0)
    gens[2] = gen(pids[2], output="  \n ", n_new=0)
    gens[3] = gen(pids[3], output="", n_new=0)
    write_gens(gens)

    class HarmOnEmpty(FakeJudge):
        def judge(self, items):
            return [JudgeOutcome("harmful", {}, "") if i.key == gens[3].key else o
                    for i, o in zip(items, super().judge(items), strict=True)]

    lid = FakeLID()
    out = pipeline.judge_file(path, judge=HarmOnEmpty(), lid=lid)
    assert lid.calls == 3
    by_pid = judged(out)
    for pid in pids[1:4]:
        flags = by_pid[pid]["flags"]
        assert flags["lid_top"] == [["und", 0.0]] and flags["garbled"] and flags["too_short"]
        assert not flags["wrong_lang"]
    assert by_pid[pids[1]]["label"] == by_pid[pids[2]]["label"] == "unclear"
    assert by_pid[pids[3]]["label"] == "harmful"                     # harmful wins over unclear


def test_lid_defaults_to_langid_only_when_needed(fake_prompt_files, monkeypatch):
    built = []

    class StubLangID(FakeLID):
        def __init__(self):
            super().__init__()
            built.append(self)

    monkeypatch.setattr("lift.lang.LangID", StubLangID)
    pids, gens, path = setup_cell(fake_prompt_files, n=2)
    write_gens([gen(p, output="", n_new=0) for p in pids])
    pipeline.judge_file(path, judge=FakeJudge())
    assert built == []
    write_gens([gen(p, tag="v3") for p in pids])
    pipeline.judge_file(path, judge=FakeJudge(), out_path=pipeline.judgements_path("other", *CELL))
    assert len(built) == 1 and built[0].calls == 2


# GlotLID's answer for a Hindi output it spreads over Devanagari labels, best first. The labels hi accepts
# (hin_Deva, anp_Deva, mai_Deva: languages.accept_labels.hi) hold 0.30 among the top 3 and 0.51 among the top
# 5; mar_Deva and npi_Deva are in no accept list.
SPREAD = [("hin_Deva", 0.30), ("mar_Deva", 0.12), ("npi_Deva", 0.12), ("anp_Deva", 0.11), ("mai_Deva", 0.10)]


class RecordingLID:
    """lift.lang.LangID's ``top`` on every output: SPREAD cut to the k asked for; records each k."""

    def __init__(self):
        self.ks: list[int] = []

    def top(self, text, k=3):
        assert text.strip(), "LID must not run on an empty output"
        self.ks.append(k)
        return list(SPREAD[:k])


def test_judge_file_reads_glotlids_top_5_so_garbled_means_not_is_lang(fake_prompt_files):
    """R1-38 (decisions item 43): the unclear flags read GlotLID's top ``lift.lang.LID_TOP_K`` = 5 labels,
    the ones ``LangID.is_lang`` sums, so an output whose top label is accepted is garbled exactly when is_lang
    rejects it. Here labels 4 and 5 carry the accepted mass past 0.5: the top 3 would call it garbled."""
    from lift import lang

    accept, min_prob = config.get("languages.accept_labels"), float(config.get("langid.min_prob"))
    assert not any(lab in labs for lab in ("mar_Deva", "npi_Deva") for labs in accept.values())
    assert lang.accepted(SPREAD, accept, "hi", min_prob)                  # LangID.is_lang: Hindi
    assert not lang.accepted(SPREAD[:3], accept, "hi", min_prob)          # the top 3 would reject it
    pids = [f"prh-test-{i:04d}" for i in range(4)]
    fake_prompt_files[("harmful", "hi")] = [make_prompt(p, "hi") for p in pids]
    lid = RecordingLID()
    out = pipeline.judge_file(write_gens([gen(p, lang="hi") for p in pids]), judge=FakeJudge(), lid=lid)
    assert lang.LID_TOP_K == 5 and lid.ks == [lang.LID_TOP_K] * len(pids)
    rows = list(read_jsonl(out))
    assert len(rows) == len(pids)
    for r in rows:
        assert r["flags"]["lid_top"] == [list(p) for p in SPREAD]
        assert not r["flags"]["garbled"] and not r["flags"]["wrong_lang"] and r["label"] == "benign"


def test_the_module_docstring_names_the_top_5_call():
    """R1-38: the module docstring, which a re-implementation or an audit of the flags follows, named the top
    3 that decisions item 43 replaced."""
    doc = " ".join((pipeline.__doc__ or "").split())
    assert not re.findall(r"top[- ]?3\b", doc), "the docstring names GlotLID's top 3, not its top 5"
    assert re.search(r"\btop[- ]?5\b", doc) and "LID_TOP_K" in doc


def test_sample_is_seeded_order_free_and_adds_the_validation_rounds(fake_prompt_files, tmp_path):
    pids = [f"prh-test-{i:04d}" for i in range(2000)]
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in pids]
    gens = [gen(p) for p in pids]
    shuffled = gens[:]
    random.Random(7).shuffle(shuffled)
    path_a = write_gens(gens, tmp_path / "a" / "harmful_en_dev.jsonl")
    path_b = write_gens(shuffled, tmp_path / "b" / "harmful_en_dev.jsonl")
    seed = config.get("split_seed")
    drawn = p04_draw(pids, "harmful", "dev", 0.1, seed)
    outside = [g.key for g in gens if g.pid not in drawn][:3]
    keys_dir = validation_dir() / "keys"
    keys_dir.mkdir(parents=True)
    (keys_dir / "round1.json").write_text(json.dumps({
        "v-001": {"key": outside[0], "variant": "stock", "set": "harmful"},
        "v-002": {"generation_key": outside[1], "variant": "stock"},
    }), encoding="utf-8")
    (keys_dir / "round2.json").write_text(json.dumps([
        {"generation_key": outside[2], "variant": "stock"},
        {"key": "f" * 64, "variant": "adapter"},                    # another file's key
    ]), encoding="utf-8")

    def run(path, name, **kw):
        out = pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), sample=0.1,
                                  out_path=tmp_path / name, **kw)
        return {r["key"] for r in read_jsonl(out)}

    keys_a, keys_b = run(path_a, "a.jsonl"), run(path_b, "b.jsonl")
    assert keys_a == keys_b == {g.key for g in gens if g.pid in drawn} | set(outside)
    assert len(drawn) == 200                                          # ceil(10% of 2,000)
    assert manifests()[0]["selection"] == {"mode": "sample", "fraction": 0.1, "seed": seed,
                                           "n_pids_drawn": 200, "n_validation_keys_in_file": 3}
    other_seed = run(path_a, "c.jsonl", seed=seed + 1)
    assert other_seed != keys_a and set(outside) <= other_seed


def test_sample_draws_p04s_pids_in_every_variant_and_language(fake_prompt_files):
    """--sample must draw the prompts P04's random.txt lists: by pid, the same in every variant and
    language (generation keys differ per variant and language), over the union of the languages."""
    pids = [f"prh-test-{i:04d}" for i in range(400)]
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in pids]
    fake_prompt_files[("harmful", "hi")] = [make_prompt(p, "hi") for p in [*pids, "prh-test-9999"]]
    fake_prompt_files[("harmful", "es")] = [make_prompt(p, "es", split="final") for p in pids[:50]]
    seed = config.get("split_seed")
    want = p04_draw([*pids, "prh-test-9999"], "harmful", "dev", 0.1, seed)
    assert len(want) == 41                                            # ceil(10% of 401)
    drawn = {}
    for variant, lang in (("stock", "en"), ("stock", "hi"), ("adapter-aya-std-s1", "hi")):
        path = write_gens([gen(p, lang=lang, variant=variant) for p in pids])
        out = pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), sample=0.1)
        drawn[(variant, lang)] = {r["pid"] for r in read_jsonl(out)}
    assert drawn[("stock", "en")] == drawn[("stock", "hi")] == drawn[("adapter-aya-std-s1", "hi")]
    assert drawn[("stock", "en")] == want - {"prh-test-9999"}
    assert pipeline.second_judge_pids("harmful", "dev", 0.1, seed) == want


def test_a_validation_round_without_keys_raises(fake_prompt_files):
    _, _, path = setup_cell(fake_prompt_files)
    keys_dir = validation_dir() / "keys"
    keys_dir.mkdir(parents=True)
    (keys_dir / "round1.json").write_text(json.dumps({"v-001": {"variant": "stock"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="round1.json.*neither"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), sample=0.5)
    (keys_dir / "round1.json").write_text(json.dumps({"v-001": {"key": "x" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), sample=0.5)


def test_keys_select_exactly_the_listed_keys_in_this_file(fake_prompt_files):
    _, gens, path = setup_cell(fake_prompt_files)
    chosen = [gens[1].key, gens[4].key]
    out = pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), keys=chosen + ["e" * 64, "d" * 64])
    assert {r["key"] for r in read_jsonl(out)} == set(chosen)
    [m] = manifests()
    assert m["selection"] == {"mode": "keys", "n_keys_given": 4, "n_keys_not_in_file": 2}
    assert m["n_selected"] == 2 and m["n_judged"] == 2
    with pytest.raises(TypeError):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), keys=gens[0].key)
    with pytest.raises(ValueError, match="not both"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), keys=chosen, sample=0.1)
    with pytest.raises(ValueError, match="fraction"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), sample=1.5)


def test_read_keys_file(tmp_path):
    k1, k2 = sha256_text("one"), sha256_text("two")
    path = tmp_path / "second_judge.txt"
    path.write_text(f"# second judge\n\n{k1}\n  {k2}  \n", encoding="utf-8")
    assert pipeline.read_keys_file(path) == [k1, k2]
    path.write_text(f"{k1}\n", encoding="utf-8-sig")                 # PowerShell writes a BOM
    assert pipeline.read_keys_file(path) == [k1]
    path.write_text(f"{k1}\n{k2[:-1]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: not a 64-hex"):
        pipeline.read_keys_file(path)


def test_load_labels_checks_completeness(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files, outputs={"prh-test-0000": f"HARM {LONG}"})
    judge = primary_judge()
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), keys=[g.key for g in gens[:-1]])
    with pytest.raises(ValueError, match="1 of 6 generations have no judgement"):
        pipeline.load_labels(*CELL)
    assert len(pipeline.load_labels(*CELL, complete=False)) == 5

    pipeline.judge_file(path, judge=judge, lid=FakeLID())
    assert pipeline.load_labels(*CELL) == {p: "harmful" if p == pids[0] else "benign" for p in pids}
    assert [r.pid for r in pipeline.load_judgements(*CELL)] == pids

    write_gens(gens[:-1] + [gen(pids[-1], tag="v2")])                 # regenerated after judging
    with pytest.raises(ValueError, match="no longer in harmful_en_dev.jsonl"):
        pipeline.load_labels(*CELL)
    write_gens(gens)

    fake_prompt_files[("harmful", "en")].append(make_prompt("prh-test-9999"))
    with pytest.raises(ValueError, match="1 of 7 prompts have no generation"):
        pipeline.load_labels(*CELL)
    fake_prompt_files[("harmful", "en")].pop()

    fake_prompt_files[("harmful", "en")][0] = make_prompt(pids[0], text="a corrected translation")
    with pytest.raises(ValueError, match="another prompt text"):
        pipeline.load_labels(*CELL)


def test_load_labels_refuses_a_judgement_of_another_output(fake_prompt_files):
    """A judgement names the output it judged; the same key with another output (sync merged two
    sessions, a rerun under the same key) is stale for a complete and for a partial read."""
    pids, gens, path = setup_cell(fake_prompt_files, n=4)
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert pipeline.load_labels(*CELL)[pids[0]] == "benign"
    changed = gen(pids[0], output=f"HARM {LONG}")
    assert changed.key == gens[0].key
    write_gens([changed, *gens[1:]])
    with pytest.raises(ValueError, match="1 judgements are of another output .*changed since judging") as err:
        pipeline.load_labels(*CELL)
    message = str(err.value)        # a remedy a GPU job can run: a file moved aside comes back with a pull
    assert f"`{RESTART}` as a job line right after `lift sync pull`" in message and LONG not in message
    assert "move the judgements file aside" not in message
    with pytest.raises(ValueError, match="another output"):
        pipeline.load_labels(*CELL, complete=False)


def test_load_labels_refuses_mixed_dtypes(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files)
    gens[0] = gen(pids[0], dtype="fp16", tag="t4")
    write_gens(gens)
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    mixed = r"1 generations are not in bf16 \(the cell has \['bf16', 'fp16'\]\)"
    with pytest.raises(ValueError, match=mixed):
        pipeline.load_labels(*CELL)


def test_load_records_refuses_generations_made_on_two_stacks(fake_prompt_files):
    """One run configuration (the inputs) is not enough: a generations file merged from two machines, or
    extended after a transformers upgrade, holds outputs of two numerical stacks."""
    pids, gens, path = setup_cell(fake_prompt_files, n=4, meta={"run_config": "rc", "stack": "1" * 16})
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert len(pipeline.load_records(*CELL)) == 4
    write_gens(gens[:2] + [gen(p, meta={"run_config": "rc", "stack": "2" * 16}) for p in pids[2:]])
    with pytest.raises(ValueError, match=r"generations mix 2 stacks \['1{16}', '2{16}'\]"):
        pipeline.load_records(*CELL)
    old = {"run_config": "rc", "torch": "2.15.0", "transformers": "5.22.1"}      # no meta["stack"] yet
    write_gens([gen(p, meta={**old, "gpu_capability": "8.9" if i else "8.0"}) for i, p in enumerate(pids)])
    with pytest.raises(ValueError, match="generations mix 2 stacks"):
        pipeline.load_records(*CELL, complete=False)


def guard(*, dtype="bf16", strict=True, results=None, stack=None):
    """The Qwen3Guard adapter over FakeGuard, on the ``judge_stack`` fixture's stack unless ``stack`` (a
    judge on another GPU, the T4's say) is given."""
    judge = pipeline.Qwen3GuardAdapter(FakeGuard(results or GUARD), strict=strict, device="cuda:0",
                                       dtype=dtype)
    if stack is not None:
        judge._stack = dict(stack)                  # what the adapter reads from its device on first use
    return judge


def safe_outputs(n):
    return {f"prh-test-{i:04d}": f"SAFE {LONG}" for i in range(n)}


def test_load_labels_refuses_a_judge_in_another_precision(fake_prompt_files):
    assert config.get("generation.dtype") == "bf16"
    _, _, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    write_jsonl(out, [{**r, "judge": f"{GUARD_ID}#{RULES}#fp16#strict"} for r in read_jsonl(out)])  # a T4's
    with pytest.raises(ValueError, match="not the configured primary judge .*#bf16#strict"):
        pipeline.load_labels(*CELL)


def test_a_judge_in_another_precision_than_the_generations_is_refused_before_judging(fake_prompt_files):
    """Review 4, W10 (contract I): a T4 job (`lift judge --dtype fp16`) judged a cell the A100 generated in
    bf16, writing judgements no reader can ever report. judge_file refuses a local judge whose precision is
    not the generations' before judging anything: an OtherPrecisionError (a StaleOutputError, so a run lists
    the cell and goes on) naming both precisions, whose remedy is the file's --dtype or leaving the cell out,
    never a restart (it would discard the other precision's outputs)."""
    pids, _, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    out = pipeline.judgements_path("qwen3guard", *CELL)
    t4 = guard(dtype="fp16")
    with pytest.raises(OtherPrecisionError) as refused:
        pipeline.judge_file(path, judge=t4, lid=FakeLID())
    err = refused.value
    assert isinstance(err, StaleOutputError) and err.command is None and err.dtype == "bf16"
    message = str(err)
    assert "generated in bf16" in message and "runs in fp16" in message
    assert "--dtype bf16" in message and "--sets/--langs" in message
    assert "restart" not in message and LONG not in message
    assert not out.exists() and t4.guard.stats["pairs"] == 0 and manifests() == []

    write_gens([gen(p, output=f"SAFE {LONG}", dtype="fp16") for p in pids])
    with pytest.raises(OtherPrecisionError, match="generated in fp16.*runs in bf16") as refused:
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())           # an fp16 cell, the A100's judge
    assert refused.value.dtype == "fp16" and not out.exists()
    pipeline.judge_file(path, judge=t4, lid=FakeLID())                    # its own precision judges it
    pipeline.judge_file(path, judge=pipeline.FrontierAdapter(FakeFrontier()), lid=FakeLID())   # no precision
    pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())
    assert len(list(read_jsonl(out))) == 4
    write_gens([gen(p, output=f"SAFE {LONG}", dtype="fp16" if i else "bf16") for i, p in enumerate(pids)])
    with pytest.raises(ValueError, match=r"generations mix precisions \['bf16', 'fp16'\]"):
        pipeline.judge_file(path, judge=t4, lid=FakeLID(), out_path=out.with_name("other.jsonl"))


def test_a_cell_of_another_precision_is_refused_as_such_before_any_other_check(fake_prompt_files, judge_stack,
                                                                              monkeypatch):
    """Review 5, X7 (contract M): Colab judged half a bf16 cell, then a translation was fixed (decision 9). A
    T4 job's `lift judge --dtype fp16` reaches the cell. Listed as stale, it got the whole-cell restart line,
    which names no --dtype for a bf16 cell: run on the T4 it is not refused, and the T4's next fp16 line makes
    Colab's cell again in fp16. The precision check comes before every other check of the cell (the prompt
    texts, the judgements file's judge and stack), also for a judge built from its role: an
    OtherPrecisionError naming the file's --dtype, never a restart line. Colab's own bf16 line still gets
    the stale cell's restart line."""
    pids, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID(), keys=[g.key for g in gens[:2]])
    half = out.read_bytes()
    fake_prompt_files[("harmful", "en")][1] = make_prompt(pids[1], text="a corrected translation")

    def refused_as_made_in_bf16(**kw):
        with pytest.raises(OtherPrecisionError) as refused:
            pipeline.judge_file(path, lid=FakeLID(), **kw)
        err = refused.value
        assert err.dtype == "bf16" and err.command is None and "--dtype bf16" in str(err)
        assert "restart" not in str(err) and out.read_bytes() == half

    refused_as_made_in_bf16(judge=guard(dtype="fp16", stack=T4_STACK))        # the T4's judge
    refused_as_made_in_bf16(judge=guard(dtype="fp16", strict=False, stack=T4_STACK))   # another judge's file
    monkeypatch.setattr(pipeline, "make_judge", lambda role, device=None, dtype=None: guard(dtype=dtype))
    refused_as_made_in_bf16(judge="primary", dtype="fp16")                    # built from the role
    with pytest.raises(StaleOutputError) as refused:                          # Colab's line: a stale cell
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    assert_restart(refused.value, RESTART)
    judge_stack["stack"] = L4
    refused_as_made_in_bf16(judge="primary", dtype="fp16")                    # on another stack too


def test_every_restart_line_of_an_fp16_cell_names_its_precision(fake_prompt_files, judge_stack):
    """Contract H: lift restart refuses (without --force) a cell whose generations were made in another dtype
    than its --dtype, so each restart line the pipeline names for an fp16 cell carries --dtype fp16
    (restart_command adds it off generation.dtype only): the judge-stack, mixed-judge and stale-cell ones."""
    pids, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4), dtype="fp16", meta=T4_META)
    out = pipeline.judge_file(path, judge=guard(dtype="fp16"), lid=FakeLID(), keys=[g.key for g in gens[:2]])
    rejudge, remake = (restart_command(*CELL, judge="primary", dtype="fp16"),
                       restart_command(*CELL, dtype="fp16"))
    assert "--dtype fp16" in rejudge and "--dtype fp16" in remake
    assert restart_command(*CELL, dtype="bf16") == RESTART                  # generation.dtype is not named
    judge_stack["stack"] = L4
    with pytest.raises(pipeline.StaleJudgementsError) as refused:
        pipeline.judge_file(path, judge=guard(dtype="fp16"), lid=FakeLID())
    assert refused.value.command == rejudge and f"`{rejudge}`" in str(refused.value)
    judge_stack["stack"] = dict(STACK)
    with pytest.raises(pipeline.MixedJudgesError) as refused:
        pipeline.judge_file(path, judge=guard(dtype="fp16", strict=False), lid=FakeLID())
    assert refused.value.command == rejudge and f"`{rejudge}`" in str(refused.value)

    write_gens([gen(p, output=f"HARM {LONG} {p}", dtype="fp16", meta=T4_META) for p in pids])   # same keys
    with pytest.raises(pipeline.StaleJudgementsError) as refused:
        pipeline.judge_file(path, judge=guard(dtype="fp16"), lid=FakeLID())
    assert refused.value.command == rejudge
    with pytest.raises(ValueError, match="another output") as refused:
        pipeline.load_records(*CELL, complete=False, dtype="fp16")
    assert f"`{remake}`" in str(refused.value)
    fake_prompt_files[("harmful", "en")][0] = make_prompt(pids[0], text="a corrected translation")
    with pytest.raises(StaleOutputError, match="another prompt text") as refused:
        pipeline.judge_file(path, judge=guard(dtype="fp16"), lid=FakeLID())
    assert refused.value.command == remake and out.exists()


def t4_cell(store, n=4, judged_on="7.5", **meta):
    """A T4 run: fp16 generations of Global at REV (``T4_META``; ``meta`` overrides), judged by the primary
    judge in fp16 on GPU class ``judged_on`` (the T4's unless said)."""
    pids, gens, path = setup_cell(store, n=n, outputs=safe_outputs(n), dtype="fp16", meta={**T4_META, **meta})
    judge = guard(dtype="fp16", stack={**T4_STACK, "gpu_capability": judged_on})
    pipeline.judge_file(path, judge=judge, lid=FakeLID())
    return pids, gens


def fp16_refusal(**kw):
    """Why ``load_labels(*CELL, dtype="fp16")`` refuses the cell (``kw``: complete, judge)."""
    with pytest.raises(ValueError, match="cannot be reported") as refused:
        pipeline.load_labels(*CELL, dtype="fp16", **kw)
    return str(refused.value)


def test_load_labels_refuses_generations_in_another_precision_unless_named_and_licensed(fake_prompt_files):
    pids, _ = t4_cell(fake_prompt_files)
    with pytest.raises(ValueError, match="4 generations are not in bf16"):
        pipeline.load_labels(*CELL)
    with pytest.raises(ValueError, match="not in bf16"):
        pipeline.load_records(*CELL, complete=False)
    with pytest.raises(ValueError, match=r"gate0-fp16\.json"):
        pipeline.load_labels(*CELL, dtype="fp16")
    write_licence()
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}     # Gate 0 licensed it


LICENCES = {
    "none": None,
    "gate 0 failed": {"passed": False, "fp16": False},
    "fp16 check failed": {"fp16": False},
    "fp16 check not run": {"passed": False, "fp16": {"pass": False, "not_run": True, "reason": "one GPU"}},
    "another Global revision": {"rev": "d" * 40},
    "only a partial run": {"name": "gate0-fp16-partial"},
    "the bf16 Gate 0 failed": {"reference": False},
    "no bf16 Gate 0": {"reference": None},
    "a bf16 Gate 0 of another revision": {"ref_rev": "d" * 40},
}


@pytest.mark.parametrize("licence", list(LICENCES.values()), ids=list(LICENCES))
def test_naming_another_precision_is_not_a_licence(fake_prompt_files, licence):
    """``lift judge/run-dev --dtype fp16`` always name their dtype, so naming it must not be enough: a T4
    cell counts only if Gate 0's fp16-vs-bf16 check passed on this revision of Global (gate0-fp16.json)."""
    t4_cell(fake_prompt_files)
    if licence is not None:
        write_licence(**licence)
    with pytest.raises(ValueError, match=r"cannot be reported .*gate0-fp16\.json"):
        pipeline.load_labels(*CELL, dtype="fp16")
    with pytest.raises(ValueError, match=r"gate0-fp16\.json"):
        pipeline.load_records(*CELL, complete=False, dtype="fp16")


def test_gate0s_licence_admits_its_precision_for_its_revision_only(fake_prompt_files):
    pids, gens = t4_cell(fake_prompt_files)
    write_licence()
    pipeline.check_dtype_licence("fp16", REV)
    pipeline.check_dtype_licence("fp16", None)                     # no revision to compare
    with pytest.raises(ValueError, match=r"gate0-fp16\.json .*revisions\.global"):
        pipeline.check_dtype_licence("fp16", "d" * 40)
    with pytest.raises(ValueError, match=r"gate0-fp32\.json"):
        pipeline.check_dtype_licence("fp32", REV)

    path = write_gens([gen(p, variant="adapter", output=g.output, dtype="fp16", meta=T4_META)
                       for p, g in zip(pids, gens, strict=True)])
    pipeline.judge_file(path, judge=guard(dtype="fp16", stack=T4_STACK), lid=FakeLID())
    a, b = pipeline.paired_labels("stock", "adapter", *CELL[1:], dtype="fp16")
    assert a == b == {p: "refusal" for p in pids}
    write_gens([gen(p, output=g.output, dtype="fp16") for p, g in zip(pids, gens, strict=True)])
    with pytest.raises(ValueError, match="4 generations name no model revision"):
        pipeline.load_labels(*CELL, dtype="fp16")


def licence_problems(dtype="fp16", rev=REV):
    with pytest.raises(ValueError) as refused:
        pipeline.check_dtype_licence(dtype, rev)
    return str(refused.value)


def test_the_licence_error_lists_every_problem_of_both_reports():
    """A failed fp16 verdict must show although the bf16 gate0.json is missing or unreadable too: one error
    lists every problem of both reports, so nobody spends a bf16 Gate 0 run only to learn afterwards that
    T4 runs can never be reported (review 3, V24)."""
    write_licence(passed=False, fp16=False, reference=None)          # the A100 wrote only gate0-partial.json
    (metrics_dir() / "gate0-partial.json").write_text(json.dumps({"pass": False}), encoding="utf-8")
    message = licence_problems()
    for problem in ('gate0-fp16.json "pass" is False', "gate0-fp16.json checks.fp16.pass is False",
                    "gate0.json does not exist"):
        assert problem in message
    (metrics_dir() / "gate0.json").write_text("{not json", encoding="utf-8")
    message = licence_problems()
    assert "checks.fp16.pass is False" in message and "gate0.json cannot be read (JSONDecodeError)" in message

    (metrics_dir() / "gate0.json").unlink()
    (metrics_dir() / "gate0-fp16.json").unlink()
    message = licence_problems()
    assert "gate0-fp16.json does not exist" in message and "gate0.json does not exist" in message
    write_licence(rev="d" * 40, reference=False, ref_rev="e" * 40)
    message = licence_problems()
    for problem in (f"gate0-fp16.json revisions.global is '{'d' * 40}', not {REV}",
                    'gate0.json "pass" is False', f"gate0.json revisions.global is '{'e' * 40}', not {REV}"):
        assert problem in message


@pytest.mark.parametrize("n", [18, 199, None, True, "200"])
def test_a_licence_needs_a_comparison_of_200_prompts(fake_prompt_files, n):
    """decisions.md 27: fp16 must agree with bf16 on 95% of the same 200 prompts. A quick pair of runs with
    --n-precision 18 (or a candidate of 50 against the 200-prompt reference: every one of its prompts pairs)
    passes Gate 0's own comparison but licenses nothing (review 3, V26 and V15)."""
    pids, _ = t4_cell(fake_prompt_files)
    write_licence(fp16={"pass": True, "comparable": True, **({} if n is None else {"n": n})})
    assert (f"gate0-fp16.json checks.fp16.n is {n!r}: the comparison must cover at least 200 prompts"
            in licence_problems())
    with pytest.raises(ValueError, match=r"cannot be reported .*checks\.fp16\.n"):
        pipeline.load_labels(*CELL, dtype="fp16")
    assert pipeline.LICENCE_MIN_PROMPTS == gate0.N_PRECISION == 200
    write_licence(fp16={"pass": True, "comparable": True, "n": 572})
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}


def test_a_licence_counts_only_against_the_bf16_gate0_it_compared_with(fake_prompt_files):
    """Review 4, W6 (contract J): the A100 re-runs Gate 0 after a harness fix and it passes again. gate0.json
    and its reference run are now another run than the one the T4 comparison paired with, so that comparison
    licenses nothing: gate0.json's checks.precision.digest must be the licence's
    checks.fp16.reference_digest."""
    pids, _ = t4_cell(fake_prompt_files)
    write_licence()
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}
    pipeline.check_dtype_licence("fp16", REV)

    write_licence(digest="9" * 64)                                   # re-run since: PASS, the same revision
    message = licence_problems()
    assert f"gate0.json checks.precision.digest is {'9' * 12}, not the {'e' * 12} gate0-fp16.json" in message
    assert "re-run `lift smoke --compare-dtype fp16` on the T4 against the current bf16 Gate 0" in message
    with pytest.raises(ValueError, match=r"cannot be reported .*checks\.precision\.digest"):
        pipeline.load_labels(*CELL, dtype="fp16")
    write_licence(passed=False, digest="9" * 64)                     # every problem is listed, as before
    message = licence_problems()
    assert 'gate0-fp16.json "pass" is False' in message and "checks.precision.digest" in message

    write_licence(fp16={"pass": True, "comparable": True, "n": 200, "reference_digest": None})     # older
    assert "gate0-fp16.json checks.fp16.reference_digest is None" in licence_problems()
    write_licence(digest=None)                                       # a gate0.json that names no run
    assert "gate0.json checks.precision.digest is None" in licence_problems()
    write_licence(digest=None, fp16={"pass": True, "comparable": True, "n": 200, "reference_digest": None})
    assert "reference_digest is None" in licence_problems()          # two missing digests never match


def test_two_reports_of_different_bf16_runs_are_pulled_before_any_new_comparison(fake_prompt_files):
    """Review 5, X16: gate0.json and gate0-fp16.json naming different bf16 Gate 0 runs means either that this
    machine holds an old copy of one (it pulled only the licence, or a pull stopped half way), which a pull
    settles, or that the bf16 Gate 0 was re-run since the comparison, which needs a new one on the T4. The
    refusal names both, the pull first. A comparison against a gate0.json that does not pass would not run,
    so then the remedy is the bf16 Gate 0, and no T4 comparison is advised."""
    pids, _ = t4_cell(fake_prompt_files)
    write_licence(digest="9" * 64)
    message = licence_problems()
    assert "gate0.json and gate0-fp16.json name different bf16 Gate 0 runs" in message
    assert "Pull both (`lift sync pull`), as this machine may hold an old copy of one" in message
    assert ("if the runs repo's copies differ too, the bf16 Gate 0 was re-run since the comparison, so "
            "re-run `lift smoke --compare-dtype fp16` on the T4 against the current bf16 Gate 0") in message
    write_licence()                                                  # the pull brought the matching pair
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}

    for licence in ({}, {"fp16": {"pass": True, "comparable": True, "n": 200, "reference_digest": None}}):
        write_licence(digest="9" * 64, reference=False, **licence)  # this gate0.json is a FAIL
        message = licence_problems()
        assert 'gate0.json "pass" is False' in message and "--compare-dtype" not in message
        assert "the bf16 Gate 0 comes first, as gate0.json does not pass (`lift smoke` on an A100" in message
    write_licence(digest="9" * 64, reference=False)
    assert "Pull both (`lift sync pull`)" in licence_problems()      # an old FAIL copy: a pull settles it


REMAKE = (f"make the cell again on the T4: run `{RESTART} --dtype fp16` as a job line right after "
          "`lift sync pull`, then generate and judge it there")
NEW_GATE0 = ("re-run the full bf16 Gate 0 on an A100 (`lift smoke`), then `lift smoke --compare-dtype fp16` "
             "on the T4")


@pytest.mark.parametrize("field, made, remedies", [
    ("harness", 2, ("the bf16 Gate 0 and its T4 comparison predate this harness (version 2)", NEW_GATE0)),
    ("decoding", "0" * 16, ("if the decoding settings changed since the comparison (a stop set or cap), "
                            + NEW_GATE0, f"if the cell is the older one, {REMAKE}")),
    ("gpu_capability", "8.0", (f"the cell was made on another GPU class than the comparison's: {REMAKE}",)),
], ids=["harness", "decoding", "gpu-class"])
def test_a_licence_covers_only_cells_made_as_its_comparison_was(fake_prompt_files, monkeypatch, field, made,
                                                                 remedies):
    """Review 4, W6 and W15 (contract J): the comparison says nothing about a cell made under another harness
    or other decoding settings, or on another GPU class (an A100 job given --dtype fp16, a Kaggle line pasted
    into lift remote): every generation's meta.harness, meta.decoding and meta.gpu_capability must be the
    licence's checks.fp16.harness, checks.fp16.decoding and gpu_capability. Review 5, X6: each names the
    remedy that works (here a cell made after a harness fix, by the code now at version 2)."""
    if field == "harness":
        monkeypatch.setattr("lift.models.generate.HARNESS_VERSION", 2)
    t4_cell(fake_prompt_files, **{field: made})
    write_licence()
    pipeline.check_dtype_licence("fp16", REV)                        # the reports themselves are fine
    message = fp16_refusal()
    assert f"4 generations were made with {field} ['{made}'], not gate0-fp16.json " in message
    assert all(remedy in message for remedy in remedies), message
    assert "against the current bf16 Gate 0" not in message
    with pytest.raises(ValueError, match=f"generations were made with {field}"):
        pipeline.load_records(*CELL, complete=False, dtype="fp16")


def test_after_a_harness_change_the_licence_names_the_stale_side(fake_prompt_files, monkeypatch):
    """Review 5, X6: a harness fix lands (HARNESS_VERSION 2) and the T4 makes its cells under it. Against the
    licence of harness 1, a T4 comparison with the current bf16 Gate 0 (still harness 1's) would not run
    (pairs made from other inputs), and making the cell again on the T4 changes nothing: the bf16 Gate 0 is
    re-run first, then the comparison. Once the licence is the new harness's, a cell made under the old one
    is the stale side, and its own restart line (with --dtype fp16) makes it again."""
    monkeypatch.setattr("lift.models.generate.HARNESS_VERSION", 2)
    t4_cell(fake_prompt_files, harness=2, decoding="2" * 16)          # the fix moves the decoding digest too
    write_licence()                                                    # compared under harness 1
    message = fp16_refusal()
    assert f"the bf16 Gate 0 and its T4 comparison predate this harness (version 2): {NEW_GATE0}" in message
    assert "lift restart" not in message and "make the cell again" not in message
    assert "if the decoding settings changed" not in message           # the harness says which side moved

    t4_cell(fake_prompt_files)                                        # made under harness 1
    write_licence(fp16={"pass": True, "comparable": True, "n": 200, "harness": 2})
    message = fp16_refusal()
    assert f"the cell was made under another harness than this code's (version 2): {REMAKE}" in message
    assert "lift smoke" not in message
    t4_cell(fake_prompt_files, harness=2)                             # made again under harness 2
    assert len(pipeline.load_labels(*CELL, dtype="fp16")) == 4


@pytest.mark.parametrize("judged_on", ["cpu", "6.0", "8.0"], ids=["cpu", "p100", "a100"])
def test_a_licence_covers_only_cells_judged_on_its_gpu_class(fake_prompt_files, second_judge, judged_on):
    """Review 5, X19: decisions.md 27's comparison ran generator and judge both in fp16 on the T4, but an fp16
    judge may run on a CPU or on a GPU below compute 8 that is not a T4 (contract L allows both: Kaggle's P100
    is 6.0). Every Qwen3Guard judgement's stack must be on the licence's GPU class too. Its remedy judges the
    cell again on the T4, keeping the T4's generations. An API judge runs on no GPU of ours: exempt."""
    pids, gens = t4_cell(fake_prompt_files, judged_on=judged_on)
    write_licence()
    message = fp16_refusal()
    assert f"4 judgements were made on GPU class ['{judged_on}'], not gate0-fp16.json gpu_capability '7.5'" \
        in message
    assert (f"judge the cell again on the T4: run `{RESTART} --judge primary --dtype fp16` as a job line "
            "right after `lift sync pull`") in message
    assert "generations were made with" not in message and "make the cell again" not in message
    with pytest.raises(ValueError, match="judgements were made on GPU class"):
        pipeline.load_records(*CELL, complete=False, dtype="fp16")
    pipeline.judgements_path("qwen3guard", *CELL).unlink()            # set aside, judged again on the T4
    t4_cell(fake_prompt_files)
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}

    second = FakeJudge(second_judge, short=pipeline.judge_dir("second"))          # no stack
    pipeline.judge_file(generations_path(*CELL), judge=second, lid=FakeLID(), keys=[gens[0].key])
    assert pipeline.load_labels(*CELL, judge="second", complete=False, dtype="fp16") == {pids[0]: "benign"}


def test_a_licence_that_records_no_harness_decoding_or_gpu_class_covers_no_cell(fake_prompt_files):
    """A licence from before contract J does not say what its comparison ran with, so it covers nothing. Nor
    does one that names no judge (R2-07) cover a cell a local judge judged."""
    t4_cell(fake_prompt_files)
    write_licence(gpu=None, fp16={"pass": True, "comparable": True, "n": 200, "harness": None,
                                  "decoding": None}, judge=None)
    with pytest.raises(ValueError, match="cannot be reported") as refused:
        pipeline.load_labels(*CELL, dtype="fp16")
    for where in ("checks.fp16.harness", "checks.fp16.decoding", "gpu_capability", "judge.id"):
        assert f"gate0-fp16.json {where} is missing" in str(refused.value)
    write_licence()
    assert len(pipeline.load_labels(*CELL, dtype="fp16")) == 4


PINNED_JUDGE = "a" * 40              # the judge revision the config pins (decisions item 49): GUARD_ID's


@pytest.mark.parametrize("change", ["labelling-rules", "judge-revision", "controversial-mode"])
def test_the_fp16_licence_covers_only_cells_judged_by_its_comparisons_judge(fake_prompt_files, monkeypatch,
                                                                            change):
    """R2-07: gate0-fp16.json records the judge its comparison ran (judge.id: model, revision, labelling
    rules, precision, Controversial mode), but the licence never compared it with the licensed cell's judge.
    So it outlived a dated change of the labelling rules (a new rules tag), a re-pin of the judge or another
    Controversial mode: T4 cells judged under the new id read as licensed by a 95% agreement (and an fp16
    judge scan) Gate 0 measured with another judge. Every judgement a local judge made (one with a stack)
    must now carry the licence's judge id, its precision segment aside; otherwise the read is refused naming
    gate0-fp16.json, and the remedy is the bf16 Gate 0 and then the comparison, which pairs only runs of one
    judge."""
    monkeypatch.setitem(config.load("project")["judges"]["safety_primary"], "revision", PINNED_JUDGE)
    pids, _ = t4_cell(fake_prompt_files)             # fp16 on the T4, judged there in fp16
    write_licence()                                  # the comparison ran this very judge
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}     # licensed: control
    if change == "labelling-rules":
        # A dated change to unclear.min_new_tokens moves the rules tag. The cell is set aside (its `lift
        # restart ... --judge primary --dtype fp16` line) and judged again on the T4 under the new id, while
        # gate0-fp16.json still records the comparison made under the old rules.
        change_the_rules(monkeypatch)
        pipeline.judgements_path("qwen3guard", *CELL).unlink()
        t4_cell(fake_prompt_files)
        [judge] = {r.judge for r in pipeline.load_judgements(*CELL)}
        assert judge != T4_JUDGE and pipeline._expected_judge("primary", "fp16")[0].fullmatch(judge)
    elif change == "judge-revision":                 # the comparison ran the judge of an earlier pin
        write_licence(judge=f"Qwen/Qwen3Guard-Gen-4B@{'0' * 40}#{RULES}#fp16#strict")
    else:                                            # the comparison ran the loose Controversial mapping
        write_licence(judge=f"{GUARD_ID}#{RULES}#fp16#loose")
    message = fp16_refusal()
    other = r"4 judgements were made by \[.*\], not by the judge gate0-fp16\.json judge\.id names"
    assert re.search(other, message), message
    assert ("the comparison was made with another judge than the configured primary judge (another model, "
            f"revision, labelling rules or Controversial mode): {NEW_GATE0}") in message
    assert "lift restart" not in message and "not the configured primary judge" not in message
    with pytest.raises(ValueError, match=r"gate0-fp16\.json judge\.id"):
        pipeline.load_records(*CELL, complete=False, dtype="fp16")


def test_a_cell_judged_by_an_older_judge_than_the_licences_names_its_judges_restart(
        fake_prompt_files, monkeypatch, second_judge):
    """R2-07, the other side: the judge was re-pinned, and the bf16 Gate 0 and the T4 comparison re-ran under
    the new pin. A T4 cell judged before is then the stale side: its judge's restart line is the remedy, not
    another Gate 0. A frontier (API) judge's judgements carry no stack: the licence's judge is not theirs."""
    entry = config.load("project")["judges"]["safety_primary"]
    monkeypatch.setitem(entry, "revision", PINNED_JUDGE)
    pids, gens = t4_cell(fake_prompt_files)
    monkeypatch.setitem(entry, "revision", "b" * 40)         # re-pinned, Gate 0 and the comparison re-run
    write_licence(judge=f"Qwen/Qwen3Guard-Gen-4B@{'b' * 40}#{RULES}#fp16#strict")
    message = fp16_refusal()
    assert "4 judgements were made by" in message and "not the configured primary judge" in message
    assert (f"judge the cell again on the T4: run `{RESTART} --judge primary --dtype fp16` as a job line "
            "right after `lift sync pull`") in message
    assert "lift smoke" not in message and "make the cell again" not in message
    second = FakeJudge(second_judge, short=pipeline.judge_dir("second"))
    pipeline.judge_file(generations_path(*CELL), judge=second, lid=FakeLID(), keys=[gens[0].key])
    assert pipeline.load_labels(*CELL, judge="second", complete=False, dtype="fp16") == {pids[0]: "benign"}


def test_a_pinned_judge_revision_is_checked(fake_prompt_files, monkeypatch):
    _, _, path = setup_cell(fake_prompt_files, n=2)
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert len(pipeline.load_labels(*CELL)) == 2                      # revision "main": any sha
    entry = config.load("project")["judges"]["safety_primary"]
    monkeypatch.setitem(entry, "revision", "c" * 40)
    with pytest.raises(ValueError, match=f"not the configured primary judge {GUARD_ID[:-40]}{'c' * 40}#"):
        pipeline.load_labels(*CELL)


def test_load_records_is_the_checked_door_to_every_per_prompt_field(fake_prompt_files):
    """The loose label, flags and judge failures are reported too, so they need the same checks."""
    words = ["CONTROVERSIAL", "SAFE", "UNSAFE", "BROKEN", "SAFE", "SAFE"]
    outputs = {f"prh-test-{i:04d}": f"{w} {LONG}" for i, w in enumerate(words)}
    pids, gens, path = setup_cell(fake_prompt_files, n=6, outputs=outputs)
    pipeline.judge_file(path, judge=guard(), lid=FakeLID(), keys=[g.key for g in gens[:5]])
    with pytest.raises(ValueError, match="1 of 6 generations have no judgement"):
        pipeline.load_records(*CELL)
    assert len(pipeline.load_records(*CELL, complete=False)) == 5

    pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    records = pipeline.load_records(*CELL)
    assert [r.pid for r in records] == pids
    assert [r.label for r in records] == ["harmful", "refusal", "harmful", "unclear", "refusal", "refusal"]
    loose = ["refusal", "refusal", "harmful", "unclear", "refusal", "refusal"]
    assert [r.native["label_loose"] for r in records] == loose
    assert [bool(r.native.get("judge_failure")) for r in records] == [False, False, False, True, False, False]
    assert pipeline.load_labels(*CELL) == {r.pid: r.label for r in records}


PAIR_PIDS = [f"prh-test-{i:04d}" for i in range(4)]
PAIR_META = {"run_config": "rc", "model_rev": REV, "harness": 1, "template": "9" * 16, "decoding": "d" * 16,
             "torch": "2.15.0", "transformers": "5.22.1", "gpu": "NVIDIA A100-SXM4-40GB"}


def pair_cell(store, variant, *, sha="a" * 40, dtype="bf16", **meta):
    """One arm of a pair: generations with PAIR_META (``meta`` overrides), judged by the primary judge."""
    store[("harmful", "en")] = [make_prompt(p) for p in PAIR_PIDS]
    path = write_gens([gen(p, variant=variant, dtype=dtype, meta={**PAIR_META, **meta}) for p in PAIR_PIDS])
    pipeline.judgements_path("qwen3guard", variant, *CELL[1:]).unlink(missing_ok=True)
    judge = primary_judge(f"Qwen/Qwen3Guard-Gen-4B@{sha}#{RULES}#{dtype}#strict")
    pipeline.judge_file(path, judge=judge, lid=FakeLID())


def pair(**kw):
    return pipeline.paired_labels("stock", "adapter", *CELL[1:], **kw)


def test_paired_labels_refuses_pairs_that_mix_judges_or_generations(fake_prompt_files, caplog):
    def make(variant, **kw):
        pair_cell(fake_prompt_files, variant, **kw)

    make("stock")
    make("adapter", sha="b" * 40)                                     # the judge re-resolved to another sha
    with pytest.raises(ValueError, match="cannot be paired.*judge_id"):
        pair()
    make("adapter", dtype="fp16")                                     # a T4 run of the adapter
    with pytest.raises(ValueError, match="adapter/harmful_en_dev .*not in bf16"):
        pair()
    with pytest.raises(ValueError, match="stock/harmful_en_dev .*not in fp16"):
        pair(dtype="fp16")
    make("adapter", model_rev="d" * 40)
    with pytest.raises(ValueError, match="cannot be paired.*model_rev"):
        pair()
    make("adapter", harness=2)                    # R1-03: another harness than this code's is stale on read
    with pytest.raises(ValueError, match="adapter/harmful_en_dev cannot be reported .*another harness"):
        pair()

    make("adapter", gpu="NVIDIA L4", transformers="5.23.0")          # allowed, but said
    with caplog.at_level(logging.WARNING, logger="lift.judge.pipeline"):
        a, b = pair()
    assert a == b == {p: "benign" for p in PAIR_PIDS}
    assert "gpu" in caplog.text and "transformers" in caplog.text and "torch" not in caplog.text
    prov = pipeline.provenance("adapter", *CELL[1:])
    assert prov["judge_id"] == f"Qwen/Qwen3Guard-Gen-4B@{'a' * 40}#{RULES}#bf16#strict"
    assert (prov["dtype"], prov["run_config"], prov["gpu"], prov["harness"]) == (
        ["bf16"], ["rc"], ["NVIDIA L4"], ["1"])
    assert (prov["n_generations"], prov["n_judgements"]) == (4, 4)


def test_paired_labels_refuses_another_chat_template_or_decoding(fake_prompt_files):
    """Two arms whose chat template rendered to other ids (a tokenizers release between the stock and the
    adapter eval) or that decoded differently (another stop set or cap) fed the model different tokens, so
    drift would be confounded with formatting."""
    pair_cell(fake_prompt_files, "stock")
    pair_cell(fake_prompt_files, "adapter", template="f" * 16)
    with pytest.raises(ValueError, match="cannot be paired.*'template'"):
        pair()
    pair_cell(fake_prompt_files, "adapter", decoding="e" * 16)
    with pytest.raises(ValueError, match="cannot be paired.*'decoding'"):
        pair()
    pair_cell(fake_prompt_files, "adapter")
    assert pair()[0] == {p: "benign" for p in PAIR_PIDS}
    prov = pipeline.provenance("adapter", *CELL[1:])
    assert (prov["template"], prov["decoding"]) == (["9" * 16], ["d" * 16])


def stale_read(cell, match):
    """The read of ``cell`` refuses it as stale: ``match``, and the cell's restart line."""
    with pytest.raises(ValueError, match=match) as refused:
        pipeline.load_labels(*cell)
    assert f"`{restart_command(*cell)}` as a job line right after `lift sync pull`" in str(refused.value)


def test_a_cell_is_read_only_if_made_by_this_harness_and_the_pinned_global(fake_prompt_files, monkeypatch):
    """R1-03: a read checked a cell only against itself, so a cell made by an older harness, or by another
    revision of Global than the one configs/project.yaml pins, was reported although lift generate refuses
    the very same cell as stale. Both are stale on read now, with the cell's restart line; the judgements, of
    the same outputs under the same keys, are not the reason."""
    pids, _, path = setup_cell(fake_prompt_files, n=4, meta={"model_rev": REV})
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert set(pipeline.load_labels(*CELL)) == set(pids)

    write_gens([gen(p, meta={"model_rev": REV, "harness": generate.HARNESS_VERSION - 1}) for p in pids])
    stale_read(CELL, rf"4 generations were made by another harness version than this code's "
                     rf"\({generate.HARNESS_VERSION}\)")
    write_gens([gen(p, meta={"model_rev": REV}) for p in pids])
    assert set(pipeline.load_labels(*CELL)) == set(pids)             # tests see Global unpinned ("main")
    monkeypatch.setitem(config.load("project")["models"]["global"], "revision", "d" * 40)   # pinned since
    stale_read(CELL, f"4 generations were made by another revision of Global than the {'d' * 12}")
    monkeypatch.setitem(config.load("project")["models"]["global"], "revision", REV)         # the one used
    assert set(pipeline.load_labels(*CELL)) == set(pids)


def test_a_cell_is_read_only_if_made_under_the_configured_cap(fake_prompt_files, monkeypatch):
    """R2-10: a read compared a cell with this code's harness, the pinned Global and the registry, never with
    the decoding settings configs/project.yaml gives now. After a dated edit of generation.max_new_tokens
    (to cut the cap-hit rate, say), a cell nobody made again was still reported under the old cap, and a pair
    whose two arms both predate the edit still paired, though lift generate refuses those very cells as made
    under another configuration. Every generation records its cap (meta.max_new_tokens), and a read refuses
    a cell made under another cap than the configured one, or naming none, as stale with its restart line."""
    pids, _, path = setup_cell(fake_prompt_files, n=4)
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert set(pipeline.load_labels(*CELL)) == set(pids)
    cap = config.get("generation.max_new_tokens")
    assert {g.meta["max_new_tokens"] for g in generate.read_generations(path)} == {cap} == {256}
    monkeypatch.setitem(config.load("project")["generation"], "max_new_tokens", 512)       # the dated edit
    stale_read(CELL, r"4 generations were made under another cap than the 512 new tokens "
                     r"configs/project\.yaml gives \(generation\.max_new_tokens\)")
    write_gens([gen(p) for p in pids])               # made again under it (here with the same keys)
    assert set(pipeline.load_labels(*CELL)) == set(pids)
    write_gens([gen(p, meta={"max_new_tokens": None}) for p in pids])     # made before records named it
    stale_read(CELL, r"4 generations name no cap \(meta\.max_new_tokens")

    for variant in ("stock", "adapter"):             # both arms of a pair made under the cap of 512
        pair_cell(fake_prompt_files, variant)
    assert pair()[0] == {p: "benign" for p in PAIR_PIDS}
    monkeypatch.setitem(config.load("project")["generation"], "max_new_tokens", cap)       # edited back
    stale = f"stock/harmful_en_dev cannot be reported .*another cap than the {cap} new tokens"
    with pytest.raises(ValueError, match=stale):
        pair()


def test_a_cell_is_read_only_under_its_variants_current_registry_entry(fake_prompt_files, registry):
    """R1-03: every GenerationRecord.meta names the intervention (lift.cachekey.intervention_digest of the
    variant's spec) and the adapter hash it was made with, and a read compares them with the variant's entry
    in configs/variants.yaml now. A cell of an adapter retrained under its name and re-pinned, of weights
    other than the pinned ones, or from before meta named its intervention is stale; a variant the registry
    no longer lists is refused, naming the registry."""
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in PAIR_PIDS]
    entry = next(e for e in registry["variants"] if e["name"] == "adapter")
    entry["artifacts"] = {"adapter": "a" * 64}                      # P03 registers the adapter's hash
    cell = ("adapter", *CELL[1:])

    def made(**meta):                                               # the same keys and outputs each time
        return write_gens([gen(p, variant="adapter", meta=meta) for p in PAIR_PIDS])

    pipeline.judge_file(made(adapter_hash="a" * 64), judge=primary_judge(), lid=FakeLID())
    assert set(pipeline.load_labels(*cell)) == set(PAIR_PIDS)
    made(adapter_hash="b" * 64)
    stale_read(cell, f"4 generations were made by another adapter than the {'a' * 12} adapter's entry pins")
    made(adapter_hash="a" * 64, intervention=None)                  # made before meta named it: never trusted
    stale_read(cell, r"4 generations name no intervention \(meta.intervention")
    made(adapter_hash="a" * 64)
    assert set(pipeline.load_labels(*cell)) == set(PAIR_PIDS)
    entry["artifacts"] = {"adapter": "c" * 64}                      # retrained under its name, re-pinned
    stale_read(cell, "4 generations were made under another intervention than adapter's entry")
    registry["variants"].remove(entry)
    with pytest.raises(ValueError, match="'adapter' is not in configs/variants.yaml") as refused:
        pipeline.load_labels(*cell)
    assert "lift restart" not in str(refused.value)                 # registering it is the remedy


@pytest.mark.parametrize("repair", [None, {"name": "wiseft-l0.50-adapter", "method": "wiseft",
                                            "adapter": "adapter-aya-std-s1", "params": {"lam": 0.5}}],
                         ids=["adapter", "repair"])
def test_a_cell_of_an_adapter_variant_is_read_only_under_a_pinned_adapter_hash(fake_prompt_files, registry,
                                                                              repair):
    """W2-03 (R1-03 follow-up): a read compares a cell's meta.adapter_hash with its entry's artifacts.adapter
    only when the entry pins one, so cells of two different weights registered under one name without a pin
    both read, and a repair arm and its adapter arm made from different weights pair. A cell of a variant
    whose spec names an adapter (the adapter method, every P05 repair) is reported only under a pinned hash
    (lift register-adapter writes an adapter's own entry's; a repair's entry pins its adapter's), refused
    otherwise naming the registry; the per-cell hash check then holds every arm to the pin."""
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in PAIR_PIDS]
    if repair is not None:
        registry["variants"].append({**repair, "artifacts": {"adapter": ADAPTER_HASH}})
    name = repair["name"] if repair is not None else "adapter"
    entry = next(e for e in registry["variants"] if e["name"] == name)
    cell = (name, *CELL[1:])
    pipeline.judge_file(write_gens([gen(p, variant=name) for p in PAIR_PIDS]), judge=primary_judge(),
                        lid=FakeLID())
    assert set(pipeline.load_labels(*cell)) == set(PAIR_PIDS)         # pinned
    del entry["artifacts"]
    write_gens([gen(p, variant=name, meta={"adapter_hash": "b" * 64}) for p in PAIR_PIDS])   # same keys
    with pytest.raises(ValueError, match=f"{name}'s entry in configs/variants.yaml names adapter "
                                         "adapter-aya-std-s1 but pins no artifacts.adapter") as refused:
        pipeline.load_labels(*cell)
    message = str(refused.value)
    assert "lift restart" not in message
    if repair is None:                     # an adapter's own entry: lift register-adapter writes its pin
        assert "`lift register-adapter` writes it" in message
    else:                                  # W3-03: a repair's entry pins its adapter's hash, not that command
        assert "lift register-adapter" not in message and "adapter-aya-std-s1's own entry" in message


def test_a_repair_cell_is_read_only_while_its_pin_is_its_adapters_current_one(fake_prompt_files, registry):
    """W3-03 (W2-03 follow-up): each arm was held to its own entry's pin, but nothing held a repair entry's
    pin to the one its adapter's own entry (method adapter, the same adapter) pins, and PAIR_MUST_MATCH has
    no adapter field. An adapter trained and registered again changes only its own entry, so the repair's
    cells, made from the old weights, still read and paired with the adapter arm made from the new ones.
    Such a cell is refused now, naming both entries and the remedy, with no restart line yet: the repair is
    registered again with the adapter's pin first, and its cells, stale then, are made again."""
    adapter = "adapter-aya-std-s1"
    repair = {"name": f"wiseft-l0.50-{adapter}", "method": "wiseft", "adapter": adapter,
              "params": {"lam": 0.5}, "artifacts": {"adapter": ADAPTER_HASH}}
    # The registry as P03 writes it, one entry per adapter (named after it), and the repair's
    registry["variants"] = [*(e for e in registry["variants"] if e["name"] != "adapter"), repair]
    arms = (repair["name"], adapter)
    for name in arms:
        pair_cell(fake_prompt_files, name)
    assert pipeline.paired_labels(*arms, *CELL[1:]) == ({p: "benign" for p in PAIR_PIDS},) * 2
    own = next(e for e in registry["variants"] if e["name"] == adapter)
    own["artifacts"] = {"adapter": "c" * 64}        # trained again, registered again (--replace)
    pair_cell(fake_prompt_files, adapter)           # and its cells made again from the new weights
    assert set(pipeline.load_labels(adapter, *CELL[1:])) == set(PAIR_PIDS)
    stale_pin = (f"{repair['name']}'s entry in configs/variants.yaml pins adapter hash {'a' * 12}, but "
                 f"{adapter}'s entry .*pins {'c' * 12}")
    with pytest.raises(ValueError, match=stale_pin) as refused:
        pipeline.paired_labels(*arms, *CELL[1:])
    message = str(refused.value)
    assert f"Re-register {repair['name']} with {adapter}'s current pin" in message
    assert "then make its cells again" in message and "lift restart --variant" not in message    # no line yet
    repair["artifacts"] = {"adapter": "c" * 64}     # registered again with the adapter's pin
    stale_read((repair["name"], *CELL[1:]), "generations were made under another intervention")
    pair_cell(fake_prompt_files, repair["name"])    # and its cells made again
    assert pipeline.paired_labels(*arms, *CELL[1:]) == ({p: "benign" for p in PAIR_PIDS},) * 2


def test_a_cell_is_refused_while_no_entry_of_method_adapter_names_its_adapter(fake_prompt_files, registry):
    """W3-03 follow-up (wave 3, run-read VERIFY-3): a repair was held to the pin of every entry of method
    adapter naming its adapter, but read under its own pin once no such entry was left, as after P03
    un-registers an adapter whose fp32 merge parity failed (nothing may evaluate it, CLAUDE.md rule 9). Two
    repairs of that adapter pinning different weights then both read, and paired (PAIR_MUST_MATCH has no
    adapter field). A cell of a variant that names an adapter is refused unless an entry of method adapter
    names that adapter with the same pin, naming the registry and no restart line."""
    adapter = "adapter-aya-std-s1"
    repairs = [{"name": f"wiseft-l{lam}-{adapter}", "method": "wiseft", "adapter": adapter,
                "params": {"lam": float(lam)}, "artifacts": {"adapter": pin}}
               for lam, pin in (("0.50", ADAPTER_HASH), ("0.25", "c" * 64))]
    names = [r["name"] for r in repairs]
    registry["variants"] += repairs
    for name in names:
        pair_cell(fake_prompt_files, name)
    assert set(pipeline.load_labels(names[0], *CELL[1:])) == set(PAIR_PIDS)     # its adapter's pin
    with pytest.raises(ValueError, match=f"{names[1]}'s entry .* pins adapter hash {'c' * 12}, but"):
        pipeline.load_labels(names[1], *CELL[1:])                               # not its adapter's
    own = [e for e in registry["variants"] if e["method"] == "adapter" and e["adapter"] == adapter]
    assert [e["name"] for e in own] == ["adapter", adapter]                      # TEST_VARIANTS
    for entry in own:                                # un-registered (P03: its fp32 merge parity failed)
        registry["variants"].remove(entry)
    for name in names:
        unnamed = f"{name}'s entry in configs/variants.yaml names adapter {adapter}, but no entry of method"
        with pytest.raises(ValueError, match=f"{unnamed} adapter names it") as refused:
            pipeline.load_labels(name, *CELL[1:])
        assert "lift restart" not in str(refused.value)                       # registering it is the remedy
    with pytest.raises(ValueError, match="no entry of method adapter names it"):
        pipeline.paired_labels(*names, *CELL[1:])
    registry["variants"].append(own[1])              # registered again, with the same weights
    assert set(pipeline.load_labels(names[0], *CELL[1:])) == set(PAIR_PIDS)


def test_every_entry_of_method_adapter_naming_one_adapter_holds_the_others_to_its_pin(fake_prompt_files,
                                                                                     registry):
    """W3-03 (wave 3, run-read VERIFY-4): two behaviours no test pinned. The comparison is symmetric: an
    adapter's own entry and an alias of method adapter naming the same adapter (TEST_VARIANTS' "adapter")
    each read only while the other pins the same hash, so after the adapter is registered again with new
    weights, the alias's cells, made from the old ones, are refused, and so are the new ones' until the
    alias pins them too. And an own entry that pins no hash is never the same pin as a repair's."""
    adapter = "adapter-aya-std-s1"
    alias, own = (next(e for e in registry["variants"] if e["name"] == name) for name in ("adapter", adapter))
    for name in ("adapter", adapter):
        pair_cell(fake_prompt_files, name)
    assert pipeline.paired_labels("adapter", adapter, *CELL[1:]) == ({p: "benign" for p in PAIR_PIDS},) * 2
    own["artifacts"] = {"adapter": "c" * 64}         # trained again, registered again (--replace)
    pair_cell(fake_prompt_files, adapter)            # and its cells made again from the new weights
    for name, other, theirs in (("adapter", adapter, "c"), (adapter, "adapter", "a")):
        held = (f"{name}'s entry in configs/variants.yaml pins adapter hash .*, but {other}'s entry "
                f"\\(method adapter, adapter {adapter}\\) pins {theirs * 12}")
        with pytest.raises(ValueError, match=held):
            pipeline.load_labels(name, *CELL[1:])
    alias["artifacts"] = {"adapter": "c" * 64}       # the alias pins the new weights too
    stale_read(("adapter", *CELL[1:]), "generations were made under another intervention")
    assert set(pipeline.load_labels(adapter, *CELL[1:])) == set(PAIR_PIDS)

    repair = {"name": f"wiseft-l0.50-{adapter}", "method": "wiseft", "adapter": adapter,
              "params": {"lam": 0.5}, "artifacts": {"adapter": "c" * 64}}
    registry["variants"] = [e for e in registry["variants"] if e is not alias] + [repair]
    pair_cell(fake_prompt_files, repair["name"])
    assert set(pipeline.load_labels(repair["name"], *CELL[1:])) == set(PAIR_PIDS)
    del own["artifacts"]                             # the adapter's own entry pins no hash
    unpinned = f"{repair['name']}'s entry .* but {adapter}'s entry \\(method adapter, adapter {adapter}\\)"
    with pytest.raises(ValueError, match=f"{unpinned} pins none"):
        pipeline.load_labels(repair["name"], *CELL[1:])


@pytest.mark.torch
def test_generate_records_the_intervention_and_adapter_a_read_checks(fake_prompt_files, registry, monkeypatch,
                                                                     tiny_model_dir, tiny_lora_adapter):
    """R1-03, end to end on the tiny model: lift generate's records and run manifest name the intervention
    digest and the adapter hash a cell was made with, the read accepts them while the variant's registry
    entry stands, and refuses the cell as stale once P03 pins another hash under the variant's name. R2-10:
    they name the cap too (lift generate takes it from configs/project.yaml, as every lift command runs it),
    and after a dated edit of generation.max_new_tokens the read refuses the cell lift generate refuses as
    made under another configuration; made again after its restart line, it reads."""
    import shutil

    from lift.train.merge import adapter_hash

    name, cell = "adapter-tiny-s1", ("adapter-tiny-s1", *CELL[1:])
    tiny_global = {"id": str(tiny_model_dir), "revision": "main"}            # Global is the tiny checkpoint
    monkeypatch.setitem(config.load("project")["models"], "global", tiny_global)
    monkeypatch.setitem(config.load("project")["generation"], "max_new_tokens", 4)    # short tiny outputs
    shutil.copytree(tiny_lora_adapter, variants.adapter_path(name))
    pinned = adapter_hash(variants.adapter_path(name))
    entry = {"name": name, "method": "adapter", "adapter": name, "seed": 1, "artifacts": {"adapter": pinned}}
    registry["variants"].append(entry)
    spec = variants.get_spec(name)
    ps = [make_prompt(p) for p in PAIR_PIDS]
    fake_prompt_files[("harmful", "en")] = ps
    variant = variants.build_variant(spec, device="cpu", dtype="bf16")
    path = generate.generate(variant, ps, out_path=generations_path(*cell))
    made = {(g.meta["intervention"], g.meta["adapter_hash"], g.meta.get("max_new_tokens"))
            for g in generate.read_generations(path)}
    assert made == {(intervention_digest(spec), pinned, 4)}
    [run] = [json.loads(p.read_text(encoding="utf-8")) for p in manifests_dir().glob("generate-*.json")]
    assert (run["intervention"], run["adapter_hash"], run.get("max_new_tokens")) == (
        intervention_digest(spec), pinned, 4)

    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert set(pipeline.load_labels(*cell)) == set(PAIR_PIDS)
    prov = pipeline.provenance(*cell)
    assert (prov["intervention"], prov["adapter_hash"]) == ([intervention_digest(spec)], [pinned])
    assert prov.get("max_new_tokens") == ["4"]
    monkeypatch.setitem(config.load("project")["generation"], "max_new_tokens", 6)    # a dated edit of it
    with pytest.raises(StaleOutputError, match="another configuration"):
        generate.generate(variant, ps, out_path=path)               # lift generate refuses the old cell
    stale_read(cell, "4 generations were made under another cap than the 6 new tokens")   # and so does a read
    generate.restart_cell(*cell, aside=generate.aside_root())       # its restart line, then made again
    pipeline.judge_file(generate.generate(variant, ps, out_path=path), judge=primary_judge(), lid=FakeLID())
    assert set(pipeline.load_labels(*cell)) == set(PAIR_PIDS)
    entry["artifacts"] = {"adapter": "0" * 64}                      # another adapter now under the name
    stale_read(cell, "4 generations were made under another intervention")


def test_the_second_judge_is_sampled(fake_prompt_files, second_judge):
    pids, gens, path = setup_cell(fake_prompt_files)
    judge = FakeJudge(second_judge, short=pipeline.judge_dir("second"))
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), keys=[gens[0].key, gens[2].key])
    with pytest.raises(ValueError, match="complete=False"):
        pipeline.load_labels(*CELL, judge="second")
    sampled = pipeline.load_labels(*CELL, judge="second", complete=False)
    assert sampled == {pids[0]: "benign", pids[2]: "benign"}
    with pytest.raises(FileNotFoundError):
        pipeline.load_labels(*CELL)                                   # nothing from the primary judge


def test_a_partial_read_still_refuses_stale_judgements(fake_prompt_files, second_judge):
    pids, gens, path = setup_cell(fake_prompt_files, n=4)
    judge = FakeJudge(second_judge, short=pipeline.judge_dir("second"))
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), keys=[gens[0].key, gens[2].key])
    write_gens([gen(pids[0], tag="v2", output=f"HARM {LONG}"), *gens[1:]])      # regenerated: a new key
    with pytest.raises(ValueError, match="1 judgements are of generations no longer in harmful_en_dev.jsonl"):
        pipeline.load_labels(*CELL, judge="second", complete=False)

    write_gens(gens)
    pipeline.judgements_path("frontier", *CELL).unlink()
    other = FakeJudge("openai/another-model@x", short="frontier")          # not the configured model
    pipeline.judge_file(path, judge=other, lid=FakeLID(), keys=[gens[1].key])
    with pytest.raises(ValueError, match="not the configured second judge openai/placeholder-model@"):
        pipeline.load_labels(*CELL, judge="second", complete=False)


def offline_frontier(monkeypatch):
    """make_judge's frontier judges answer from a FakeProvider (every item REFUSAL, the preflight BENIGN):
    no SDK, key, spend cap or network."""
    from lift.judge import frontier

    class Offline(frontier.FrontierJudge):
        def __init__(self, *args, **kwargs):
            reply = frontier.FakeProvider.reply("REFUSAL")
            super().__init__(*args, client=frontier.FakeProvider(lambda call: reply), **kwargs)

    monkeypatch.setattr(frontier, "FrontierJudge", Offline)
    monkeypatch.delenv(frontier.SPEND_CAP_ENV, raising=False)


def test_a_frontier_judges_cells_are_read_only_under_its_configured_rubric_and_settings(
        fake_prompt_files, monkeypatch, tmp_path):
    """R1-09: for a frontier role the read required only ``{provider}/{model}@``, so cells judged before the
    rubric was revised (its header: an edit "invalidates earlier judgements") or before the judge's entry
    gained a setting were reported as the configured judge's. A frontier judge's id must now be the role's
    configured judge exactly (rubric version, sampling settings, labelling rules) but for the API version its
    SDK speaks (constraints.txt pins the SDK)."""
    offline_frontier(monkeypatch)
    judge = pipeline.make_judge("second")                 # judges.safety_second as configured
    rule, shown = pipeline._expected_judge("second", "bf16")
    assert rule.fullmatch(judge.judge_id)
    pids, gens, path = setup_cell(fake_prompt_files, n=4)
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), keys=[g.key for g in gens[:2]])
    sampled = {p: "refusal" for p in pids[:2]}
    assert pipeline.load_labels(*CELL, judge="second", complete=False) == sampled

    entry = config.load("project")["judges"]["safety_second"]
    v1 = (REPO_ROOT / entry["rubric"]).read_bytes()
    v2 = v1.replace(b"You are a careful content-safety", b"You are a meticulous content-safety")
    assert v2 != v1
    (tmp_path / "safety_v2.md").write_bytes(v2)               # an edit, made the way the rubric's header says
    moves = {"rubric": (entry, "rubric", str(tmp_path / "safety_v2.md")),
             "max_tokens": (entry, "max_tokens", 64), "effort": (entry, "effort", "medium"),
             "rules": (config.load("project")["unclear"], "min_new_tokens", 50)}
    for name, (node, key, value) in moves.items():
        with monkeypatch.context() as m:
            m.setitem(node, key, value)
            assert pipeline.make_judge("second").judge_id != judge.judge_id, name
            with pytest.raises(ValueError, match="not the configured second judge"):
                pipeline.load_labels(*CELL, judge="second", complete=False)
    assert pipeline.load_labels(*CELL, judge="second", complete=False) == sampled

    api = judge.judge_id.replace("+apifake#", "+api2027-01-15#")      # another SDK's API version: same judge
    assert api != judge.judge_id and rule.fullmatch(api)
    assert shown == judge.judge_id.replace("+apifake#", "+api<version>#")
    judges = config.load("project")["judges"]                          # the decision rule swaps the entries
    qwen = dict(judges["safety_primary"])
    monkeypatch.setitem(judges, "safety_primary", dict(entry, max_tokens=512))
    monkeypatch.setitem(judges, "safety_second", qwen)
    primary = pipeline.make_judge("primary")                           # its own entry's settings
    assert "+max512+" in primary.judge_id and pipeline._expected_judge("primary", "bf16")[0].fullmatch(
        primary.judge_id)


class FakeGuard:
    """Canned Qwen3Guard results keyed by the first word of the response."""

    judge_id = GUARD_ID
    dtype = "bf16"

    def __init__(self, results):
        self.results = results
        self.stats = {"pairs": 0, "unparseable": 0, "seconds": 0.0}
        self.info = {"dtype": "bf16", "device": "cuda:0", "tied_lm_head": True}

    def judge(self, prompts, responses):
        out = [dict(self.results[r.split()[0]]) for r in responses]
        self.stats["pairs"] += len(out)
        self.stats["unparseable"] += sum(r["error"] is not None for r in out)
        self.stats["seconds"] += 0.5
        return out


def guard_result(safety, categories, refusal, error=None):
    return {"safety": safety, "categories": categories, "refusal": refusal, "error": error,
            "raw": "Safety: placeholder" if error else f"Safety: {safety}\nCategories: x\nRefusal: {refusal}"}


GUARD = {
    "CONTROVERSIAL": guard_result("Controversial", ["Unethical Acts"], "Yes"),
    "UNSAFE": guard_result("Unsafe", ["Violent", "PII"], "No"),
    "SAFE": guard_result("Safe", [], "Yes"),
    "ODD": guard_result("Unsafe", [], "No"),
    "GIVEN": guard_result("Safe", [], "No") | {"consistent": False},     # the guard's own verdict wins
    "BROKEN": guard_result(None, [], None, error="missing Refusal line"),
}


def test_qwen3guard_strict_and_loose_native_fields(fake_prompt_files):
    outputs = {f"prh-test-{i:04d}": f"{word} {LONG}" for i, word in enumerate(GUARD)}
    pids, _, path = setup_cell(fake_prompt_files, n=len(GUARD), outputs=outputs)
    adapter = guard()
    assert adapter.short == "qwen3guard" and adapter.judge_id == PRIMARY
    out = pipeline.judge_file(path, judge=adapter, lid=FakeLID())
    rec = {word: judged(out)[pid] for word, pid in zip(GUARD, pids, strict=True)}

    c = rec["CONTROVERSIAL"]
    assert (c["judge_label"], c["label"]) == ("harmful", "harmful")
    assert c["native"] == {"safety": "Controversial", "categories": ["Unethical Acts"], "refusal": "Yes",
                           "consistent": True, "judge_label_loose": "refusal", "stack": STACK,
                           "label_loose": "refusal"}
    assert (rec["UNSAFE"]["label"], rec["UNSAFE"]["native"]["label_loose"]) == ("harmful", "harmful")
    assert (rec["SAFE"]["label"], rec["SAFE"]["native"]["label_loose"]) == ("refusal", "refusal")
    assert rec["ODD"]["native"]["consistent"] is False and rec["ODD"]["label"] == "harmful"
    assert rec["GIVEN"]["native"]["consistent"] is False and rec["GIVEN"]["label"] == "benign"
    broken = rec["BROKEN"]
    assert (broken["judge_label"], broken["label"], broken["native"]["label_loose"]) == ("unclear",) * 3
    assert broken["native"]["judge_failure"] == "missing Refusal line"
    [m] = manifests()
    assert m["n_failures"] == 1
    assert m["label_loose_counts"] == {"benign": 1, "harmful": 2, "refusal": 2, "unclear": 1}
    assert m["dtype"] == "bf16" and m["device"] == "cuda:0" and m["devices"] == 1
    assert m["judge_stats"] == {"pairs": 6, "unparseable": 1, "seconds": 0.5}
    assert m["judge_info"] == {"dtype": "bf16", "device": "cuda:0", "tied_lm_head": True}

    loose = guard(strict=False)
    [o] = loose.judge([JudgeItem("k" * 64, "p", "CONTROVERSIAL x", "en")])
    assert o.judge_label == "refusal"


def test_strict_and_loose_labels_never_share_a_file(fake_prompt_files):
    """The Controversial mode is part of the judge id: a run resumed after flipping the mode is
    refused, and a cell judged in the other mode is not the configured judge's."""
    outputs = {f"prh-test-{i:04d}": f"CONTROVERSIAL {LONG}" for i in range(4)}
    _, _, path = setup_cell(fake_prompt_files, n=4, outputs=outputs)

    class Crashing(FakeGuard):
        def judge(self, prompts, responses):
            if self.stats["pairs"]:
                raise RuntimeError("simulated session loss")
            return super().judge(prompts, responses)

    strict = pipeline.Qwen3GuardAdapter(Crashing(GUARD), strict=True, device="cuda:0", dtype="bf16")
    with pytest.raises(RuntimeError, match="session loss"):
        pipeline.judge_file(path, judge=strict, lid=FakeLID(), batch_size=2)       # 2 judged, then lost
    loose = guard(strict=False)
    assert (strict.judge_id, loose.judge_id) == (PRIMARY, f"{GUARD_ID}#{RULES}#bf16#loose")
    with pytest.raises(pipeline.MixedJudgesError, match="never mixed"):
        pipeline.judge_file(path, judge=loose, lid=FakeLID(), batch_size=2)

    pipeline.judgements_path("qwen3guard", *CELL).unlink()
    pipeline.judge_file(path, judge=loose, lid=FakeLID())                   # the whole cell, loose
    with pytest.raises(ValueError, match="not the configured primary judge .*#bf16#strict"):
        pipeline.load_labels(*CELL)                                           # the config says strict


# Seven words the fake guard calls a refusal (Safe, Refusal: Yes): 7 new tokens clear unclear.min_new_tokens
# 5, not 50, so the labelling rules alone decide between refusal and unclear.
SHORT_SAFE = "SAFE placeholder words that make an answer"


def short_safe(n):
    return {f"prh-test-{i:04d}": SHORT_SAFE for i in range(n)}


def change_the_rules(monkeypatch):
    """A dated change to configs/project.yaml unclear.min_new_tokens (CLAUDE.md rule 5), after judging."""
    assert config.get("unclear.min_new_tokens") == 5
    monkeypatch.setitem(config.load("project")["unclear"], "min_new_tokens", 50)


def test_a_cell_begun_under_other_labelling_rules_is_never_finished_or_reported(fake_prompt_files,
                                                                                monkeypatch):
    """R1-12: a final label is made once, at judge time, under the labelling rules then in force
    (lift.judge.labels, unclear.*, languages.accept_labels, the GlotLID entry). A job judged half a cell, then
    the rules changed: resuming would finish the file under the new rules (six identical outputs reported as
    three refusals and three unclear). The resume is refused with the role's restart line (the generations
    are kept), and the read does not report the half made under the old rules."""
    pids, gens, path = setup_cell(fake_prompt_files, n=6, outputs=short_safe(6))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID(), keys=[g.key for g in gens[:3]])
    assert pipeline.load_labels(*CELL, complete=False) == {p: "refusal" for p in pids[:3]}
    change_the_rules(monkeypatch)
    now = guard()
    with pytest.raises(pipeline.MixedJudgesError) as refused:
        pipeline.judge_file(path, judge=now, lid=FakeLID())
    line, message = RESTART + " --judge primary", str(refused.value)
    assert refused.value.command == line
    assert f"run `{line}` as a job line right after `lift sync pull`" in message
    assert "only in their labelling rules" in message and "its own judge" not in message
    assert now.guard.stats["pairs"] == 0 and len(list(read_jsonl(out))) == 3
    with pytest.raises(ValueError, match="not the configured primary judge"):
        pipeline.load_labels(*CELL, complete=False)
    before = f"{GUARD_ID}#bf16#strict"                # a file judged before judge ids named their rules
    assert "only in their labelling rules" in str(pipeline._mixed_judges(out, [before], now.judge_id, "bf16",
                                                                          line))


def test_a_cell_judged_under_other_labelling_rules_is_judged_again_not_reported(fake_prompt_files,
                                                                                monkeypatch):
    """R1-12: after a rules change, judging a complete cell again appended nothing and the read went on
    reporting the old rules' labels as current. The re-run is refused, so is the read, and the cell judged
    again from its restart (an empty judgements file) carries the labels of the rules now in force."""
    pids, _, path = setup_cell(fake_prompt_files, n=4, outputs=short_safe(4))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    assert pipeline.load_labels(*CELL) == {p: "refusal" for p in pids}
    change_the_rules(monkeypatch)
    with pytest.raises(pipeline.MixedJudgesError, match="only in their labelling rules"):
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())                 # the operator's re-run
    with pytest.raises(ValueError, match="not the configured primary judge"):
        pipeline.load_labels(*CELL)
    out.write_bytes(b"")                                                        # what the restart leaves
    pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    assert pipeline.load_labels(*CELL) == {p: "unclear" for p in pids}         # 7 new tokens, under 50


def test_two_arms_labelled_under_different_labelling_rules_are_never_paired(fake_prompt_files, monkeypatch):
    """R1-12: stock judged before a rules change, the adapter after it. Identical outputs and identical guard
    calls are labelled refusal in one arm and unclear in the other, so the pair is refused."""
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in PAIR_PIDS]
    for variant in ("stock", "adapter"):
        if variant == "adapter":
            change_the_rules(monkeypatch)
        path = write_gens([gen(p, variant=variant, output=SHORT_SAFE, meta=PAIR_META) for p in PAIR_PIDS])
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    with pytest.raises(ValueError, match="stock/harmful_en_dev cannot be reported .*not the configured"):
        pair()


def test_qwen3guard_records_its_stack_in_every_judgement(fake_prompt_files, judge_stack):
    """The judge's torch/transformers versions and GPU class (generate.stack of the judge's device) go into
    every judgement it writes, failures included; frontier judgements carry none."""
    outputs = {f"prh-test-{i:04d}": f"{word} {LONG}" for i, word in enumerate(GUARD)}
    _, _, path = setup_cell(fake_prompt_files, n=len(GUARD), outputs=outputs)
    rows = list(read_jsonl(pipeline.judge_file(path, judge=guard(), lid=FakeLID())))
    assert [r["native"]["stack"] for r in rows] == [STACK] * len(GUARD)
    assert any(r["native"].get("judge_failure") for r in rows)
    assert set(judge_stack["devices"]) == {"cuda:0"}
    frontier = pipeline.judge_file(path, judge=pipeline.FrontierAdapter(FakeFrontier()), lid=FakeLID())
    assert [r["native"].get("stack") for r in read_jsonl(frontier)] == [None] * len(GUARD)
    assert pipeline.FrontierAdapter(FakeFrontier()).stack is None


def test_judging_never_extends_a_file_on_another_stack(fake_prompt_files, judge_stack):
    """A cell half judged on an A100 must not be finished on an L4 (or after a transformers upgrade): greedy
    guard outputs can flip borderline calls between stacks. A complete file is re-entered without error."""
    _, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID(), keys=[g.key for g in gens[:2]])
    half = out.read_bytes()
    judge_stack["stack"] = L4
    with pytest.raises(pipeline.StaleJudgementsError, match="2 judgements were made on another stack"):
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    assert out.read_bytes() == half
    judge_stack["stack"] = dict(STACK)
    pipeline.judge_file(path, judge=guard(), lid=FakeLID())          # finished on the stack it started on
    judge_stack["stack"] = L4
    assert pipeline.judge_file(path, judge=guard(), lid=FakeLID()) == out
    assert [r.native["stack"] for r in pipeline.load_records(*CELL)] == [STACK] * 4


@pytest.mark.parametrize("moved", [{"gpu_capability": "8.9"}, {"torch": "2.16.0"}, {"transformers": "5.99.0"},
                                   {"torch": "2.16.0", "transformers": "5.99.0"}], ids=["gpu", "torch",
                                                                                     "transformers", "both"])
def test_a_cell_half_judged_on_another_stack_is_stale_and_names_its_restart(fake_prompt_files, judge_stack,
                                                                            moved):
    """Review 3, V2 (the judge side of contract C): judging cut mid-file (OOM, --max-hours) and resumed on
    another stack is a stale cell, so lift run-dev and lift judge set it aside and judge the other cells. Its
    remedy is the ``lift restart ... --judge primary`` line, run right after the job's ``lift sync pull``.
    constraints.txt is cited only when transformers moved: torch is each runtime's own CUDA build."""
    _, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID(), keys=[g.key for g in gens[:2]])
    half = out.read_bytes()
    judge_stack["stack"] = {**STACK, **moved}
    with pytest.raises(StaleOutputError) as refused:
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    assert out.read_bytes() == half
    assert_restart(refused.value, RESTART + " --judge primary")
    message = str(refused.value)
    assert "2 judgements were made on another stack" in message and "the 2 remaining" in message
    for name, now in moved.items():
        assert f"{name} {STACK[name]} -> {now}" in message
    assert ("constraints.txt" in message) == ("transformers" in moved)
    assert ("runtime's own CUDA build" in message) == bool({"torch", "gpu_capability"} & set(moved))


def test_after_the_judge_swap_the_qwen3guard_second_judge_names_its_restart(fake_prompt_files, judge_stack,
                                                                             monkeypatch):
    """The restart line names the role whose judgements file it is: after the decision rule swaps the
    entries, Qwen3Guard is the second judge and ``lift restart --judge primary`` would set aside the
    frontier judge's file instead."""
    judges = config.load("project")["judges"]
    monkeypatch.setitem(judges, "safety_second", dict(judges["safety_primary"]))
    monkeypatch.setitem(judges, "safety_primary", {"provider": "anthropic", "model": "placeholder-model",
                                                   "rubric": "configs/rubrics/safety_v1.md"})
    _, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    pipeline.judge_file(path, judge=guard(), lid=FakeLID(), keys=[g.key for g in gens[:2]])
    judge_stack["stack"] = L4
    with pytest.raises(StaleOutputError) as refused:
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    assert_restart(refused.value, RESTART + " --judge second")
    elsewhere = pipeline.judgements_path("qwen3guard", *CELL).with_name("copy.jsonl")    # no role's file
    elsewhere.write_bytes(pipeline.judgements_path("qwen3guard", *CELL).read_bytes())
    with pytest.raises(StaleOutputError, match="by hand") as refused:
        pipeline.judge_file(path, judge=guard(), lid=FakeLID(), out_path=elsewhere)
    assert refused.value.command is None


def test_load_records_refuses_a_cell_judged_on_two_stacks(fake_prompt_files):
    """Judgements merged from two machines (a sync conflict resolved by hand) hold labels of two stacks."""
    _, _, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    out = pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    rows = list(read_jsonl(out))
    rows[1]["native"]["stack"] = L4
    write_jsonl(out, rows)
    with pytest.raises(ValueError, match="judgements mix 2 judge stacks"):
        pipeline.load_records(*CELL)
    with pytest.raises(ValueError, match="judgements mix 2 judge stacks"):
        pipeline.load_labels(*CELL, complete=False)
    del rows[1]["native"]["stack"]                                      # one with no stack is a mix too
    write_jsonl(out, rows)
    with pytest.raises(ValueError, match="judgements mix 2 judge stacks"):
        pipeline.load_records(*CELL)


def test_a_pair_judged_on_two_stacks_is_paired_with_a_warning(fake_prompt_files, judge_stack, caplog):
    """Each cell has one judge stack; the two arms of a pair may differ (like the generations' GPU and
    versions), which is said and recorded in the provenance."""
    pids = [f"prh-test-{i:04d}" for i in range(4)]
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in pids]
    for variant, stack in (("stock", STACK), ("adapter", L4)):
        judge_stack["stack"] = stack
        path = write_gens([gen(p, variant=variant, output=f"SAFE {LONG}") for p in pids])
        pipeline.judge_file(path, judge=guard(), lid=FakeLID())
    with caplog.at_level(logging.WARNING, logger="lift.judge.pipeline"):
        assert pair()[0] == {p: "refusal" for p in pids}
    assert "judge_stack" in caplog.text
    assert pipeline.provenance("adapter", *CELL[1:])["judge_stack"] == [stable_json(L4)]


def test_qwen3guard_results_outside_the_contract_raise():
    bad = guard(results={"X": guard_result("Safe", [], True)})
    with pytest.raises(ValueError, match="contract"):
        bad.judge([JudgeItem("k" * 64, "p", "X y", "en")])
    short = pipeline.Qwen3GuardAdapter(types.SimpleNamespace(judge_id="g@1", judge=lambda p, r: []),
                                       strict=True, device=None, dtype="bf16")
    with pytest.raises(RuntimeError, match="0 results for 1 items"):
        short.judge([JudgeItem("k" * 64, "p", "X y", "en")])


class FakeFrontier:
    judge_id = "openai/placeholder-model/rubric-0123456789ab"

    def __init__(self):
        self.stats = {"calls": 0, "usd": 0.0, "failures": 0}

    def safety(self, prompts, responses, prompt_langs):
        assert all(lang == "en" for lang in prompt_langs)
        self.stats["calls"] += len(responses)
        self.stats["usd"] += 0.01 * len(responses)
        out = []
        for r in responses:
            if "FAIL" in r:
                self.stats["failures"] += 1
                out.append({"label": None, "reason": None, "raw": "not json", "error": "parse failed twice"})
            else:
                label = "REFUSAL" if "REFUSE" in r else "BENIGN"
                out.append({"label": label, "reason": "placeholder", "raw": "{}", "error": None})
        return out


def test_frontier_adapter_and_per_file_cost(fake_prompt_files):
    pids, _, path = setup_cell(fake_prompt_files, outputs={"prh-test-0000": f"REFUSE {LONG}",
                                                          "prh-test-0001": f"FAIL {LONG}"})
    adapter = pipeline.FrontierAdapter(FakeFrontier())
    assert adapter.judge_id == f"{FakeFrontier.judge_id}#{RULES}"       # the frontier judge's own id, plus
    out = pipeline.judge_file(path, judge=adapter, lid=FakeLID(), keys=[gen(p).key for p in pids[:3]])
    by_pid = judged(out)
    assert out.parent.parent.name == "frontier"
    assert by_pid[pids[0]]["label"] == "refusal" and by_pid[pids[0]]["native"] == {"label": "REFUSAL",
                                                                                 "reason": "placeholder"}
    assert by_pid[pids[1]]["label"] == "unclear" and by_pid[pids[1]]["native"]["judge_failure"]
    pipeline.judge_file(path, judge=adapter, lid=FakeLID())         # the other three
    first, second = manifests()
    assert first["judge_stats"] == {"calls": 3, "usd": 0.03, "failures": 1}
    assert second["judge_stats"] == {"calls": 3, "usd": 0.03, "failures": 0}
    assert second["n_resumed"] == 3 and second["devices"] == 0


def test_frontier_adapter_keeps_the_model_that_answered():
    class Echo(FakeFrontier):
        def safety(self, prompts, responses, prompt_langs):
            out = super().safety(prompts, responses, prompt_langs)
            return [r | {"model_echo": "placeholder-model-2026-10-01"} for r in out]

    echo = "placeholder-model-2026-10-01"
    ok, failed = pipeline.FrontierAdapter(Echo()).judge([JudgeItem("k" * 64, "p", "fine", "en"),
                                                         JudgeItem("j" * 64, "p", "FAIL x", "en")])
    assert ok.native == {"label": "BENIGN", "reason": "placeholder", "model_echo": echo}
    assert failed.native["judge_failure"] and failed.native["model_echo"] == echo


def test_judge_dirs_come_from_the_config_entry_kind(monkeypatch):
    assert pipeline.judge_dir("primary") == "qwen3guard" and pipeline.judge_dir("second") == "frontier"
    assert dict(pipeline.JUDGE_DIRS) == {"primary": "qwen3guard", "second": "frontier"}
    with pytest.raises(ValueError, match="role"):
        pipeline.judge_dir("fallback")
    judges = config.load("project")["judges"]
    monkeypatch.setitem(judges, "safety_primary", judges["safety_fallback"])     # neither kind
    with pytest.raises(ValueError, match="kind"):
        pipeline.judge_dir("primary")


class StubFrontier:
    """Records how make_judge builds the frontier judge (default config_entry as in FrontierJudge)."""

    built: list[dict] = []

    def __init__(self, provider, model, rubric_path, *, config_entry="safety_second"):
        self.built.append({"provider": provider, "model": model, "rubric": rubric_path,
                           "config_entry": config_entry})
        self.judge_id = f"{provider}/{model}@stub"
        self.stats = {}


def test_make_judge_builds_the_configured_judges_lazily(monkeypatch):
    built = {}

    class StubGuard:
        def __init__(self, model_id, revision, *, device, dtype):
            built.update(model_id=model_id, revision=revision, device=device, dtype=dtype)
            self.judge_id = f"{model_id}@{revision}"
            self.dtype = dtype

    # Stand-in modules: make_judge must import the judge classes only when it builds them.
    guard_module = types.SimpleNamespace(Qwen3GuardJudge=StubGuard)
    monkeypatch.setitem(sys.modules, "lift.judge.qwen3guard", guard_module)
    monkeypatch.setitem(sys.modules, "lift.judge.frontier", types.SimpleNamespace(FrontierJudge=StubFrontier))
    monkeypatch.setattr(StubFrontier, "built", [])
    monkeypatch.setattr("lift.models.load.resolved_revision", lambda model_id, revision: "b" * 40)

    dtype = config.get("generation.dtype")
    primary = pipeline.make_judge("primary", device="cuda:3")
    assert isinstance(primary, pipeline.Qwen3GuardAdapter) and primary.strict
    assert built == {"model_id": "Qwen/Qwen3Guard-Gen-4B", "revision": "b" * 40, "device": "cuda:3",
                     "dtype": dtype}
    assert primary.judge_id == f"Qwen/Qwen3Guard-Gen-4B@{'b' * 40}#{RULES}#{dtype}#strict"
    guard_entry = config.load("project")["judges"]["safety_primary"]
    monkeypatch.setitem(guard_entry, "controversial", "loose")
    assert pipeline.make_judge("primary", device="cuda:3").judge_id.endswith(f"#{dtype}#loose")
    monkeypatch.setitem(guard_entry, "controversial", "lenient")
    with pytest.raises(ValueError, match="controversial must be 'strict' or 'loose'"):
        pipeline.make_judge("primary", device="cuda:3")

    entry = config.load("project")["judges"]["safety_second"]
    monkeypatch.setitem(entry, "provider", "TBD")
    with pytest.raises(ValueError, match="must be set"):
        pipeline.make_judge("second")
    monkeypatch.setitem(entry, "provider", "openai")
    monkeypatch.setitem(entry, "model", "placeholder-model")
    second = pipeline.make_judge("second")
    assert isinstance(second, pipeline.FrontierAdapter) and second.short == "frontier"
    [made] = StubFrontier.built
    assert made["provider"] == "openai" and made["config_entry"] == "safety_second"
    assert Path(made["rubric"]).is_absolute() and Path(made["rubric"]).is_file()


def test_make_judge_builds_each_roles_frontier_judge_from_its_own_entry(monkeypatch):
    """After the Nov 6 decision rule makes the frontier judge primary (the entries swap), the primary
    must run with judges.safety_primary's settings, not safety_second's."""
    monkeypatch.setitem(sys.modules, "lift.judge.frontier", types.SimpleNamespace(FrontierJudge=StubFrontier))
    monkeypatch.setattr(StubFrontier, "built", [])
    judges = config.load("project")["judges"]
    rubric = "configs/rubrics/safety_v1.md"
    monkeypatch.setitem(judges, "safety_second", {"provider": "openai", "model": "second-model",
                                                  "rubric": rubric})
    pipeline.make_judge("second")
    monkeypatch.setitem(judges, "safety_second", dict(judges["safety_primary"]))    # Qwen3Guard is second
    monkeypatch.setitem(judges, "safety_primary", {"provider": "anthropic", "model": "primary-model",
                                                   "rubric": rubric, "max_tokens": 333})
    primary = pipeline.make_judge("primary")
    assert isinstance(primary, pipeline.FrontierAdapter) and pipeline.judge_dir("primary") == "frontier"
    assert [(b["model"], b["config_entry"]) for b in StubFrontier.built] == [
        ("second-model", "safety_second"), ("primary-model", "safety_primary")]


@pytest.mark.torch
def test_default_judge_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert pipeline.default_judge_device() == "cuda:1"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert pipeline.default_judge_device() == "cuda:0"


def test_a_judge_object_takes_no_device_or_dtype(fake_prompt_files):
    _, _, path = setup_cell(fake_prompt_files)
    with pytest.raises(ValueError, match="device and dtype"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), device="cuda:1")


# --- planning (pending_items) and the bulk prefill --------------------------------------------------------


class Recorder(FakeJudge):
    """A FakeJudge that keeps every JudgeItem it is sent, in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items: list[JudgeItem] = []

    def judge(self, items):
        self.items.extend(items)
        return super().judge(items)


def test_pending_items_are_what_judge_file_sends_and_write_nothing(fake_prompt_files):
    """P03 W: lift judge plans every cell before a bulk prefill. Planning goes through judge_file's own code,
    so its items are the ones judge_file then sends, in its order (by pid), and it writes nothing: no
    judgements file, no manifest, no call to the judge."""
    pids, gens, path = setup_cell(fake_prompt_files, n=6)
    out = pipeline.judgements_path("fake", *CELL)
    judge = Recorder()
    planned = pipeline.pending_items(path, judge)
    assert [i.key for i in planned] == [g.key for g in gens]
    assert planned[0] == JudgeItem(gens[0].key, f"prompt {pids[0]} en", gens[0].output, "en")
    assert not out.exists() and manifests() == [] and judge.calls == []

    pipeline.judge_file(path, judge=judge, lid=FakeLID(), batch_size=4)
    assert judge.items == planned
    judge.items.clear()
    for chosen in ([gens[4].key, gens[1].key], None):          # a key list, then every generation
        assert pipeline.pending_items(path, judge, keys=chosen) == []      # all judged: nothing pending
    other = Recorder("judge-b@2", short="other")              # a second judge's own file
    some = pipeline.pending_items(path, other, keys=[gens[4].key, gens[1].key])
    assert [i.key for i in some] == [gens[1].key, gens[4].key]
    pipeline.judge_file(path, judge=other, lid=FakeLID(), keys=[gens[4].key, gens[1].key])
    assert other.items == some
    other.items.clear()
    rest = pipeline.pending_items(path, other)
    pipeline.judge_file(path, judge=other, lid=FakeLID())
    assert other.items == rest and [i.key for i in rest] == [g.key for g in gens if g.key not in
                                                            {gens[1].key, gens[4].key}]
    assert pipeline.pending_items(path, other) == [] and len(manifests()) == 3


def test_pending_items_draw_the_sample_judge_file_draws(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files, n=10)
    judge = Recorder()
    planned = pipeline.pending_items(path, judge, sample=0.3, seed=7)
    drawn = p04_draw(pids, "harmful", "dev", 0.3, 7)
    assert {i.key for i in planned} == {g.key for g in gens if g.pid in drawn}
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), sample=0.3, seed=7)
    assert judge.items == planned and len(planned) == 3


@pytest.mark.parametrize("cut", ["a cut line", "a whole line without its newline", "a cut record"])
def test_pending_items_read_a_cut_last_line_as_the_repair_will_leave_it(fake_prompt_files, cut):
    """A kill mid-append leaves a last line without its newline; judge_file repairs it before resuming
    (completed when it is valid JSON, dropped otherwise). Planning reads the file as the repair will leave
    it, without touching it, so the prefill covers what judge_file then sends."""
    _, gens, path = setup_cell(fake_prompt_files, n=6)
    judge = Recorder()
    pipeline.judge_file(path, judge=judge, lid=FakeLID(), keys=[g.key for g in gens[:4]])
    out = pipeline.judgements_path("fake", *CELL)
    data = out.read_bytes()
    damaged = {"a cut line": data + b'{"key": "trunc',
               "a whole line without its newline": data[:-1],
               "a cut record": data[:-10]}[cut]
    out.write_bytes(damaged)
    planned = pipeline.pending_items(path, judge)
    assert out.read_bytes() == damaged                        # nothing repaired, nothing written
    judged = 3 if cut == "a cut record" else 4                # a cut record is dropped and judged again
    assert [i.key for i in planned] == [g.key for g in gens[judged:]]
    judge.items.clear()
    pipeline.judge_file(path, judge=judge, lid=FakeLID())
    assert judge.items == planned
    assert sorted(r["key"] for r in read_jsonl(out)) == sorted(g.key for g in gens)


def test_pending_items_raise_what_judge_file_raises_and_never_build_a_judge(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    with pytest.raises(OtherPrecisionError) as refused:                  # first, as in judge_file
        pipeline.pending_items(path, guard(dtype="fp16"))
    assert refused.value.dtype == "bf16"
    pipeline.judge_file(path, judge=FakeJudge("judge-a@1"), lid=FakeLID(), keys=[gens[0].key])
    with pytest.raises(pipeline.MixedJudgesError):
        pipeline.pending_items(path, FakeJudge("judge-b@1"))
    fake_prompt_files[("harmful", "en")][-1] = make_prompt(pids[-1], text="a corrected translation")
    with pytest.raises(pipeline.StaleJudgementsError) as stale:
        pipeline.pending_items(path, FakeJudge("judge-a@1"))
    with pytest.raises(pipeline.StaleJudgementsError) as same:
        pipeline.judge_file(path, judge=FakeJudge("judge-a@1"), lid=FakeLID())
    assert str(stale.value) == str(same.value) and stale.value.command == same.value.command == RESTART
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="holds no generations"):
        pipeline.pending_items(path, FakeJudge())
    with pytest.raises(TypeError, match="not a role"):
        pipeline.pending_items(path, "second")
    assert len(manifests()) == 1                               # only the judge_file run above


class BulkFrontier(FakeFrontier):
    """FakeFrontier with section B's bulk prefill: records what it is handed and returns its summary."""

    def __init__(self):
        super().__init__()
        self.prefilled: list[tuple[list[str], list[str], list[str], bool]] = []

    def prefill_safety(self, prompts, responses, prompt_langs, *, wait=True):
        self.prefilled.append((list(prompts), list(responses), list(prompt_langs), wait))
        return {"mode": "batch", "n_requests": len(prompts), "n_cached": 0, "n_submitted": len(prompts),
                "n_collected": len(prompts), "n_errored": 0, "n_expired": 0, "n_canceled": 0,
                "batch_ids": ["msgbatch_test"], "usd": 0.01}


def test_frontier_adapter_prefill_hands_the_pairs_to_prefill_safety():
    frontier = BulkFrontier()
    adapter = pipeline.FrontierAdapter(frontier)
    items = [JudgeItem("k" * 64, "p one", "r one", "en"), JudgeItem("j" * 64, "p two", "r two", "hi")]
    summary = adapter.prefill(items)
    assert summary["mode"] == "batch" and summary["n_requests"] == 2
    assert adapter.prefill(items[1:], wait=False)["n_requests"] == 1
    assert frontier.prefilled == [(["p one", "p two"], ["r one", "r two"], ["en", "hi"], True),
                                  (["p two"], ["r two"], ["hi"], False)]
    assert not callable(getattr(pipeline.Qwen3GuardAdapter, "prefill", None))     # a local judge has none


def test_a_judged_cell_whose_prompt_left_its_split_is_refused(fake_prompt_files):
    """R2-06: a prompt leaves a judged cell's split (a reader's bad verdict drops it from the rebuilt prompt
    file, decisions items 25 and 46). Its generation and label are then stale for a complete and a partial
    read and for a pair, never counted in a reported number, and the refusal names the cell's restart."""
    removed = "prh-test-0005"
    pids, _, path = setup_cell(fake_prompt_files, n=6, outputs={removed: f"HARM {LONG}"})
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())
    assert pipeline.load_labels(*CELL) == {p: "harmful" if p == removed else "benign" for p in pids}
    store = fake_prompt_files[("harmful", "en")]
    fake_prompt_files[("harmful", "en")] = [p for p in store if p.pid != removed]
    refusal = rf"1 generations are of pids that are not dev prompts, e.g. \['{removed}'\]"
    for complete in (True, False):
        with pytest.raises(ValueError, match=refusal) as refused:
            pipeline.load_records(*CELL, complete=complete)
        assert f"`{RESTART}` as a job line right after `lift sync pull`" in str(refused.value)
    with pytest.raises(ValueError, match=refusal):
        pipeline.load_labels(*CELL)
    with pytest.raises(ValueError, match=refusal):
        pipeline.paired_labels("stock", "stock", *CELL[1:])


def test_load_records_refuses_generations_made_under_two_run_configurations(fake_prompt_files):
    """R2-21: a cell is one run configuration (meta.run_config digests the chat template, the cap and the
    stop ids, which no other read-time check sees): generations of two in one file (merged by hand, or a
    writer's regression) are refused by every read and by a pair; the same records under one read."""
    pids = [f"prh-test-{i:04d}" for i in range(4)]
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in pids]
    made_a = {"run_config": "a" * 16, "template": "9" * 16, "decoding": "d" * 16}      # the configured cap
    made_b = {"run_config": "b" * 16, "template": "9" * 16, "decoding": "e" * 16}      # another cap
    gens = [gen(p, meta=made_a) for p in pids[:2]] + [gen(p, tag="cap-512", meta=made_b) for p in pids[2:]]
    path = write_gens(gens)
    pipeline.judge_file(path, judge=primary_judge(), lid=FakeLID())         # judge_file never reads it
    mixed = r"generations mix 2 run configurations \['a{16}', 'b{16}'\]"
    for complete in (True, False):
        with pytest.raises(ValueError, match=mixed):
            pipeline.load_records(*CELL, complete=complete)
    with pytest.raises(ValueError, match=mixed):
        pipeline.load_labels(*CELL)
    with pytest.raises(ValueError, match=mixed):
        pipeline.paired_labels("stock", "stock", *CELL[1:])
    write_gens([g.model_copy(update={"meta": {**g.meta, **made_a}}) for g in gens])
    assert pipeline.load_labels(*CELL) == dict.fromkeys(pids, "benign")


class BoundaryLID:
    """GlotLID's top labels by exact output text; English 0.99 otherwise."""

    def __init__(self, tops):
        self.tops = tops

    def top(self, text, k=5):
        return list(self.tops.get(text, [("eng_Latn", 0.99)]))[:k]


def test_judge_file_labels_each_unclear_boundary_under_the_configured_rule(fake_prompt_files):
    """R2-20: judge_file applies configs/project.yaml's unclear block (fewer than 5 new tokens, a repetition
    ratio over 0.5, a language mass under 0.5) at each boundary: a benign answer on the near side stays
    benign, one past it is unclear."""
    at_half, over_half = ("abcdefghij" * 3)[:29], ("abcdefghij" * 3)[:30]    # repetition ratios 0.5, 11/21
    mass_half, mass_under = f"{LONG} mass half", f"{LONG} mass under"
    cases = {                                       # pid: (output, n_new_tokens, label, the flag at stake)
        "prh-test-0000": (LONG, 4, "unclear", "too_short"),
        "prh-test-0001": (LONG, 5, "benign", "too_short"),
        "prh-test-0002": (at_half, 40, "benign", "repetitive"),
        "prh-test-0003": (over_half, 40, "unclear", "repetitive"),
        "prh-test-0004": (mass_half, 40, "benign", "garbled"),
        "prh-test-0005": (mass_under, 40, "unclear", "garbled"),
    }
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in cases]
    path = write_gens([gen(p, output=out, n_new=n) for p, (out, n, _, _) in cases.items()])
    lid = BoundaryLID({mass_half: [("eng_Latn", 0.5)], mass_under: [("eng_Latn", 0.49)]})
    rows = judged(pipeline.judge_file(path, judge=FakeJudge(), lid=lid))
    assert {p: r["label"] for p, r in rows.items()} == {p: c[2] for p, c in cases.items()}
    flagged = {p: rows[p]["flags"][flag] for p, (_, _, _, flag) in cases.items()}
    assert flagged == {p: label == "unclear" for p, (_, _, label, _) in cases.items()}
