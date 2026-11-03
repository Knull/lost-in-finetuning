"""lift.judge.pipeline offline: fake judges and a fake LID, generation records written with lift.io.
Placeholder text only."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import sys
import types
from pathlib import Path

import pytest
from fakes import make_prompt

from lift import config
from lift.io import read_jsonl, sha256_text, write_jsonl
from lift.judge import pipeline
from lift.judge.pipeline import JudgeItem, JudgeOutcome
from lift.models.generate import generations_path
from lift.paths import manifests_dir, validation_dir
from lift.schemas import GenerationRecord

CELL = ("stock", "harmful", "en", "dev")
LONG = "placeholder words that make a long enough answer"
GUARD_ID = "Qwen/Qwen3Guard-Gen-4B@" + "a" * 40
PRIMARY = GUARD_ID + "#bf16#strict"             # the configured primary judge, in bf16 and strict mode
SECOND = "openai/placeholder-model@safety_v1:0123456789ab+t0+max1024"


@pytest.fixture(autouse=True)
def quick_manifests(monkeypatch):
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")


@pytest.fixture
def second_judge(monkeypatch):
    """judges.safety_second with a chosen model (configs/project.yaml still says TBD); its judge id."""
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
    prompt = prompt if prompt is not None else f"prompt {pid} {lang}"      # = make_prompt's default text
    output = output if output is not None else f"{LONG} {pid}"
    return GenerationRecord(
        key=sha256_text(f"{tag}|{variant}|{pid}|{lang}"), variant=variant, pid=pid, lang=lang, set=set_name,
        split=split, prompt_sha=sha256_text(prompt), output=output, n_prompt_tokens=12,
        n_new_tokens=len(output.split()) if n_new is None else n_new, hit_cap=False, seconds=0.01,
        meta={"dtype": dtype, **(meta or {})},
    )


def write_gens(records, path=None):
    r = records[0]
    path = path or generations_path(r.variant, r.set, r.lang, r.split)
    write_jsonl(path, [g.model_dump() for g in records])
    return path


def setup_cell(store, n=6, outputs=None, dtype="bf16"):
    """n prompts in the fake prompt store and one generation each; returns (pids, gens, path)."""
    pids = [f"prh-test-{i:04d}" for i in range(n)]
    store[("harmful", "en")] = [make_prompt(p) for p in pids]
    gens = [gen(p, output=(outputs or {}).get(p), dtype=dtype) for p in pids]
    return pids, gens, write_gens(gens)


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


def test_a_changed_output_under_the_same_key_is_never_resumed(fake_prompt_files):
    """The key hashes the inputs only. A regenerated output (bf16 batch effects, a lost and rebuilt
    file) under the same key must be re-judged, never inherit the old output's label."""
    pids, gens, path = setup_cell(fake_prompt_files, n=4)
    out = pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())
    flipped = [gen(p, output=f"HARM {LONG} {p}") for p in pids]
    assert [g.key for g in flipped] == [g.key for g in gens]
    write_gens(flipped)
    again = FakeJudge()
    with pytest.raises(ValueError, match="generations changed since judging; move the judgements file aside"):
        pipeline.judge_file(path, judge=again, lid=FakeLID())
    assert again.calls == []

    write_gens(gens)                                     # a judgement naming no output is not trusted either
    rows = list(read_jsonl(out))
    write_jsonl(out, [{k: v for k, v in r.items() if k != "output_sha"} for r in rows])
    with pytest.raises(ValueError, match="changed since judging"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())


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
    _, gens, path = setup_cell(fake_prompt_files)
    pipeline.judge_file(path, judge=FakeJudge("judge-a@1"), lid=FakeLID(), keys=[gens[0].key])
    with pytest.raises(ValueError, match="never mixed"):
        pipeline.judge_file(path, judge=FakeJudge("judge-b@1"), lid=FakeLID())
    assert len(list(read_jsonl(pipeline.judgements_path("fake", *CELL)))) == 1


def test_duplicate_or_stale_judgements_raise(fake_prompt_files):
    pids, gens, path = setup_cell(fake_prompt_files)
    out = pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines + lines[:1]), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())

    out.write_text("".join(lines), encoding="utf-8", newline="\n")
    write_gens(gens[:-1] + [gen(pids[-1], tag="v2")])          # one output regenerated under a new key
    with pytest.raises(ValueError, match="not in the generations file"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())


def test_prompt_changed_after_generation_raises(fake_prompt_files):
    pids, _, path = setup_cell(fake_prompt_files)
    fake_prompt_files[("harmful", "en")][-1] = make_prompt(pids[-1], text="a corrected translation")
    with pytest.raises(ValueError, match=pids[-1]):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID())
    assert not pipeline.judgements_path("fake", *CELL).exists()
    with pytest.raises(ValueError, match="no prompt text"):
        pipeline.judge_file(path, judge=FakeJudge(), lid=FakeLID(), prompts_by_pid={})


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
    with pytest.raises(ValueError, match="1 judgements are of another output .*changed since judging"):
        pipeline.load_labels(*CELL)
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


def guard(*, dtype="bf16", strict=True, results=None):
    return pipeline.Qwen3GuardAdapter(FakeGuard(results or GUARD), strict=strict, device="cuda:0",
                                      dtype=dtype)


def safe_outputs(n):
    return {f"prh-test-{i:04d}": f"SAFE {LONG}" for i in range(n)}


def test_load_labels_refuses_a_judge_in_another_precision(fake_prompt_files):
    assert config.get("generation.dtype") == "bf16"
    _, _, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4))
    pipeline.judge_file(path, judge=guard(dtype="fp16"), lid=FakeLID())     # bf16 outputs, a T4 judge
    with pytest.raises(ValueError, match="not the configured primary judge .*#bf16#strict"):
        pipeline.load_labels(*CELL)


def test_load_labels_refuses_generations_in_another_precision_unless_asked(fake_prompt_files):
    pids, _, path = setup_cell(fake_prompt_files, n=4, outputs=safe_outputs(4), dtype="fp16")   # a T4 run
    pipeline.judge_file(path, judge=guard(dtype="fp16"), lid=FakeLID())
    with pytest.raises(ValueError, match="4 generations are not in bf16"):
        pipeline.load_labels(*CELL)
    with pytest.raises(ValueError, match="not in bf16"):
        pipeline.load_records(*CELL, complete=False)
    assert pipeline.load_labels(*CELL, dtype="fp16") == {p: "refusal" for p in pids}     # Gate 0 licensed it


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


def test_paired_labels_refuses_pairs_that_mix_judges_or_generations(fake_prompt_files, caplog):
    pids = [f"prh-test-{i:04d}" for i in range(4)]
    fake_prompt_files[("harmful", "en")] = [make_prompt(p) for p in pids]
    base = {"run_config": "rc", "model_rev": "c" * 40, "harness": 1, "torch": "2.15.0",
            "transformers": "5.22.1", "gpu": "NVIDIA A100-SXM4-40GB"}

    def make(variant, *, sha="a" * 40, dtype="bf16", **meta):
        path = write_gens([gen(p, variant=variant, dtype=dtype, meta={**base, **meta}) for p in pids])
        pipeline.judgements_path("qwen3guard", variant, *CELL[1:]).unlink(missing_ok=True)
        judge = primary_judge(f"Qwen/Qwen3Guard-Gen-4B@{sha}#{dtype}#strict")
        pipeline.judge_file(path, judge=judge, lid=FakeLID())

    def pair(**kw):
        return pipeline.paired_labels("stock", "adapter", *CELL[1:], **kw)

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
    make("adapter", harness=2)
    with pytest.raises(ValueError, match="cannot be paired.*harness"):
        pair()

    make("adapter", gpu="NVIDIA L4", transformers="5.23.0")          # allowed, but said
    with caplog.at_level(logging.WARNING, logger="lift.judge.pipeline"):
        a, b = pair()
    assert a == b == {p: "benign" for p in pids}
    assert "gpu" in caplog.text and "transformers" in caplog.text and "torch" not in caplog.text
    prov = pipeline.provenance("adapter", *CELL[1:])
    assert prov["judge_id"] == f"Qwen/Qwen3Guard-Gen-4B@{'a' * 40}#bf16#strict"
    assert (prov["dtype"], prov["run_config"], prov["gpu"], prov["harness"]) == (
        ["bf16"], ["rc"], ["NVIDIA L4"], ["1"])
    assert (prov["n_generations"], prov["n_judgements"]) == (4, 4)


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
                           "consistent": True, "judge_label_loose": "refusal", "label_loose": "refusal"}
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
    assert (strict.judge_id, loose.judge_id) == (PRIMARY, GUARD_ID + "#bf16#loose")
    with pytest.raises(ValueError, match="never mixed"):
        pipeline.judge_file(path, judge=loose, lid=FakeLID(), batch_size=2)

    pipeline.judgements_path("qwen3guard", *CELL).unlink()
    pipeline.judge_file(path, judge=loose, lid=FakeLID())                   # the whole cell, loose
    with pytest.raises(ValueError, match="not the configured primary judge .*#bf16#strict"):
        pipeline.load_labels(*CELL)                                           # the config says strict


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
    assert primary.judge_id == f"Qwen/Qwen3Guard-Gen-4B@{'b' * 40}#{dtype}#strict"
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
