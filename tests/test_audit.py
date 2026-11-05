"""lift.audit.record_safety against the installed AuditKIT, offline, placeholder text only."""

from __future__ import annotations

import json

import pytest

auditkit = pytest.importorskip("auditkit")

from fakes import make_prompt  # noqa: E402

from lift import audit  # noqa: E402
from lift.io import read_jsonl, sha256_text, stable_json, write_jsonl  # noqa: E402
from lift.judge import labels, pipeline  # noqa: E402
from lift.models.generate import generations_path  # noqa: E402
from lift.paths import metrics_dir  # noqa: E402
from lift.schemas import GenerationRecord, JudgementRecord, UnclearFlags  # noqa: E402
from lift.stats.paired import rates  # noqa: E402

CELL = ("adapter", "harmful", "hi", "dev")
PIDS = [f"prh-test-{i:04d}" for i in range(1, 7)]
TEXTS = dict(zip(PIDS, ["placeholder prompt one", "placeholder prompt two", "placeholder prompt two",
                        "नमस्ते दुनिया", "placeholder prompt five", "placeholder prompt six"], strict=True))
LABELS = ["harmful", "refusal", "unclear", "unclear", "benign", "harmful"]
LOOSE = ["harmful", "refusal", "unclear", "unclear", "benign", "refusal"]   # the last one was Controversial
FAILED = [False, False, True, False, False, False]
OUTPUTS = [f"placeholder output number {i}" for i in range(6)]
WRONG_LANG = [False, False, False, False, True, False]
HIT_CAP = [True, False, False, True, False, False]
GUARD_ID = "Qwen/Qwen3Guard-Gen-4B@" + "a" * 40
RULES = labels.rules_tag()                        # the labelling rules of configs/project.yaml
META = {"run_config": "rc0", "model_rev": "c" * 40, "harness": 1, "template": "9" * 16, "decoding": "d" * 16,
        "torch": "2.15.0", "transformers": "5.22.1", "gpu": "NVIDIA A100-SXM4-40GB"}
GPU_CLASS = {"bf16": "8.0", "fp16": "7.5"}       # meta.gpu_capability: bf16 on the A100, fp16 on the T4
STACK = {"torch": "2.15.0", "transformers": "5.22.1", "gpu_capability": "8.0"}     # the judge's
DIGEST = "e" * 64                                 # the bf16 Gate 0 run the T4 comparison paired with


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    """AuditKIT's DiskCache and ExperimentDB would write here; both must stay empty."""
    dirs = {name: tmp_path / name for name in ("xdg-cache", "xdg-data")}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(dirs["xdg-cache"]))
    monkeypatch.setenv("XDG_DATA_HOME", str(dirs["xdg-data"]))
    return dirs


def write_cell(dtype: str = "bf16", judged_on: str | None = None) -> None:
    """Generations (written in reverse pid order) and primary-judge judgements of one cell, in ``dtype``,
    both on that precision's GPU class (``GPU_CLASS``) unless the judge ran on ``judged_on``."""
    variant, set_name, lang, split = CELL
    gens, judgements = [], []
    stack = {**STACK, "gpu_capability": judged_on or GPU_CLASS[dtype]}
    for i, pid in enumerate(PIDS):
        key = sha256_text(f"{variant}|{pid}")
        gens.append(GenerationRecord(
            key=key, variant=variant, pid=pid, lang=lang, set=set_name, split=split,
            prompt_sha=sha256_text(TEXTS[pid]), output=OUTPUTS[i], n_prompt_tokens=10, n_new_tokens=20,
            hit_cap=HIT_CAP[i], seconds=0.1,
            meta={"dtype": dtype, **META, "gpu_capability": GPU_CLASS[dtype]}))
        flags = UnclearFlags(too_short=False, repetitive=False, garbled=False, wrong_lang=WRONG_LANG[i],
                             lid_top=[("hin_Deva", 0.9)])
        native = {"label_loose": LOOSE[i], "stack": stack,
                  **({"judge_failure": "unparseable"} if FAILED[i] else {})}
        judgements.append(JudgementRecord(
            key=key, variant=variant, pid=pid, lang=lang, set=set_name, split=split,
            judge=f"{GUARD_ID}#{RULES}#{dtype}#strict", judge_label=LABELS[i], label=LABELS[i], flags=flags,
            native=native, output_sha=sha256_text(OUTPUTS[i])))
    write_jsonl(generations_path(*CELL), [g.model_dump() for g in reversed(gens)])
    write_jsonl(pipeline.judgements_path(pipeline.judge_dir("primary"), *CELL),
                [j.model_dump() for j in judgements])


@pytest.fixture
def cell(fake_prompt_files, xdg):
    fake_prompt_files[CELL[1:3]] = [make_prompt(p, CELL[2], text=TEXTS[p]) for p in PIDS]
    write_cell()
    return xdg


def test_headline_equals_lift_rates(cell):
    path = audit.record_safety(*CELL)
    assert path == metrics_dir() / "auditkit" / "qwen3guard" / "adapter" / "harmful_hi_dev.json"
    result = auditkit.RunResult.load(str(path))
    expected = rates(LABELS)
    for k in audit.RATE_KEYS:
        assert result.headline[k] == expected[k]
    assert result.headline["wrong_lang_rate"] == 1 / 6 and result.headline["cap_hit_rate"] == 2 / 6
    assert {name: stat.count for name, stat in result.stats.items()} == {
        name: 6 for name in (*audit.RATE_KEYS, *audit.FLAG_RATES)}
    assert [p.sample_id for p in result.predictions] == PIDS
    assert [p.correct for p in result.predictions] == [label != "harmful" for label in LABELS]
    assert result.model_spec == "lift:adapter" and result.failed_count == 0
    lift = result.metadata["lift"]
    assert (lift["n"], lift["n_harmful"], lift["n_unclear"]) == (6, 2, 2)
    assert (lift["variant"], lift["judge"], lift["judge_id"], lift["text"]) == (
        "adapter", "primary", f"{GUARD_ID}#{RULES}#bf16#strict", "redacted")
    assert result.config.extra["lift"]["labels_sha256"] == lift["labels_sha256"]


def test_lift_metadata_records_the_loose_rate_failures_and_provenance(cell):
    lift = auditkit.RunResult.load(str(audit.record_safety(*CELL))).metadata["lift"]
    assert lift["harmful_rate_loose"] == LOOSE.count("harmful") / 6 != lift["harmful_rate"]
    assert lift["n_judge_failures"] == 1
    assert lift["provenance"] == {"dtype": ["bf16"], "run_config": ["rc0"], "model_rev": ["c" * 40],
                                  "harness": ["1"], "template": ["9" * 16], "decoding": ["d" * 16],
                                  "torch": ["2.15.0"], "transformers": ["5.22.1"],
                                  "gpu": ["NVIDIA A100-SXM4-40GB"], "judge_stack": [stable_json(STACK)]}


def test_a_judge_without_a_loose_reading_records_none(cell):
    path = pipeline.judgements_path("qwen3guard", *CELL)
    rows = list(read_jsonl(path))
    write_jsonl(path, [{**r, "native": {"stack": STACK}} for r in rows])
    lift = auditkit.RunResult.load(str(audit.record_safety(*CELL))).metadata["lift"]
    assert lift["harmful_rate_loose"] is None and lift["n_judge_failures"] == 0
    write_jsonl(path, [{**r, "native": {"stack": STACK}} if i else r for i, r in enumerate(rows)])
    with pytest.raises(ValueError, match="1 of 6 judgements have a loose label"):
        audit.record_safety(*CELL)


def write_licence(n: int = 200, digest: str = DIGEST, **fields):
    """outputs/metrics/gate0-fp16.json as Gate 0's T4 comparison run writes it (its fp16 check compared
    ``n`` prompts with the bf16 run ``DIGEST``, under META's harness and decoding, on a T4), and the passing
    bf16 Gate 0 it compared against (gate0.json, whose run is ``digest``); only the fields the licence
    reads."""
    metrics_dir().mkdir(parents=True, exist_ok=True)
    reference = {"dtype": "bf16", "pass": True, "checks": {"precision": {"pass": True, "digest": digest}},
                 "revisions": {"global": "c" * 40}}
    (metrics_dir() / "gate0.json").write_text(json.dumps(reference), encoding="utf-8")
    fp16 = {"pass": True, "comparable": True, "n": n, "reference_digest": DIGEST, "harness": META["harness"],
            "decoding": META["decoding"]}
    doc = {"dtype": "fp16", "reference_dtype": "bf16", "pass": True, "gpu_capability": GPU_CLASS["fp16"],
           "checks": {"precision": {"pass": True}, "fp16": fp16}, "revisions": {"global": "c" * 40}, **fields}
    (metrics_dir() / "gate0-fp16.json").write_text(json.dumps(doc), encoding="utf-8")


def test_a_cell_in_another_precision_needs_gate0s_licence_and_gets_its_own_file(cell):
    from lift.commands import run as run_cmd

    write_cell("fp16")                                                # a T4 run, judged on the T4
    with pytest.raises(ValueError, match="6 generations are not in bf16"):
        audit.record_safety(*CELL)
    with pytest.raises(ValueError, match=r"gate0-fp16\.json"):        # naming the dtype is no licence
        audit.record_safety(*CELL, dtype="fp16")
    write_licence(**{"pass": False})                                  # Gate 0's fp16 check failed
    refused: dict[str, str] = {}
    run_cmd._record(*CELL, dtype="fp16", refused=refused)             # lift judge / run-dev --dtype fp16
    assert list(refused) == ["harmful_hi_dev"]
    assert 'gate0-fp16.json "pass" is False' in refused["harmful_hi_dev"]
    assert not (metrics_dir() / "auditkit").exists()

    write_licence()
    path = audit.record_safety(*CELL, dtype="fp16")
    assert path == metrics_dir() / "auditkit" / "qwen3guard" / "adapter" / "harmful_hi_dev-fp16.json"
    full = audit.record_safety(*CELL, dtype="fp16", text="full")
    assert full == path.with_name("harmful_hi_dev-fp16.full.json")
    assert sorted(p.name for p in path.parent.iterdir()) == ["harmful_hi_dev-fp16.full.json",
                                                             "harmful_hi_dev-fp16.json"]
    lift = auditkit.RunResult.load(str(path)).metadata["lift"]
    assert lift["provenance"]["dtype"] == ["fp16"] and lift["judge_id"] == f"{GUARD_ID}#{RULES}#fp16#strict"
    write_cell()
    assert audit.record_safety(*CELL, dtype="bf16") == path.with_name("harmful_hi_dev.json")


def test_a_licence_from_fewer_than_200_prompts_writes_no_record(cell):
    """Review 3, V26: a quick ``lift smoke --n-precision 18`` pair on both sides passes Gate 0, but
    decisions.md 27 compares 200 prompts, so no fp16 cell gets a reportable -fp16 record from it."""
    from lift.commands import run as run_cmd

    write_cell("fp16")
    write_licence(n=18)
    with pytest.raises(ValueError, match=r"checks\.fp16\.n is 18: the comparison must cover at least 200"):
        audit.record_safety(*CELL, dtype="fp16")
    refused: dict[str, str] = {}
    run_cmd._record(*CELL, dtype="fp16", refused=refused)
    assert list(refused) == ["harmful_hi_dev"] and "checks.fp16.n is 18" in refused["harmful_hi_dev"]
    assert not (metrics_dir() / "auditkit").exists()
    write_licence(n=200)
    assert audit.record_safety(*CELL, dtype="fp16").name == "harmful_hi_dev-fp16.json"


def test_no_fp16_record_after_the_bf16_gate0_is_re_run(cell):
    """Review 4, W6: a bf16 Gate 0 re-run after a harness fix passes again and replaces gate0.json and its
    reference run. The T4 comparison never paired with that run, so no -fp16 record is written from it until
    the comparison is re-run against the current bf16 Gate 0 (contract J)."""
    from lift.commands import run as run_cmd

    write_cell("fp16")
    write_licence(digest="9" * 64)
    with pytest.raises(ValueError, match=r"gate0\.json checks\.precision\.digest is 9{12}, not the e{12}"):
        audit.record_safety(*CELL, dtype="fp16")
    refused: dict[str, str] = {}
    run_cmd._record(*CELL, dtype="fp16", refused=refused)
    assert list(refused) == ["harmful_hi_dev"] and "checks.precision.digest" in refused["harmful_hi_dev"]
    assert not (metrics_dir() / "auditkit").exists()
    write_licence()
    assert audit.record_safety(*CELL, dtype="fp16").name == "harmful_hi_dev-fp16.json"


@pytest.mark.parametrize("judged_on", ["cpu", "6.0", "8.0"])
def test_no_fp16_record_from_a_judge_off_the_t4(cell, judged_on):
    """Review 5, X19: the T4's licence compared generator and judge both in fp16 on the T4. An fp16 judge on a
    CPU, on another GPU below compute 8 (Kaggle's P100) or on an A100 never had its labels compared, so its
    judgements of a T4 cell get no -fp16 record, through record_safety or a command's record step."""
    from lift.commands import run as run_cmd

    write_cell("fp16", judged_on=judged_on)
    write_licence()
    with pytest.raises(ValueError, match=rf"6 judgements were made on GPU class \['{judged_on}'\], not "
                                         r"gate0-fp16\.json gpu_capability '7\.5'"):
        audit.record_safety(*CELL, dtype="fp16")
    refused: dict[str, str] = {}
    run_cmd._record(*CELL, dtype="fp16", refused=refused)            # lift judge / run-dev --dtype fp16
    assert list(refused) == ["harmful_hi_dev"]
    assert "judge the cell again on the T4" in refused["harmful_hi_dev"]
    assert not (metrics_dir() / "auditkit").exists()
    write_cell("fp16")                                                # judged again on the T4
    assert audit.record_safety(*CELL, dtype="fp16").name == "harmful_hi_dev-fp16.json"


def test_a_retired_record_is_read_as_none_and_replaced(cell):
    """Contract H: ``lift restart`` empties a cell's AuditKIT records ("retired": the next push sends the
    empty file as an update), so a reader skips an empty record and record_safety writes a new one over it."""
    path = audit.auditkit_path("qwen3guard", *CELL)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    assert audit.load_record(path) is None
    assert audit.record_safety(*CELL) == path
    record = audit.load_record(path)
    assert record.headline["harmful_rate"] == rates(LABELS)["harmful_rate"]
    assert [p.name for p in path.parent.iterdir()] == [path.name]
    path.write_bytes(b"\n")                                            # whitespace only: retired as well
    assert audit.load_record(path) is None
    with pytest.raises(FileNotFoundError):
        audit.load_record(path.with_name("harmful_en_dev.json"))       # never made: not a retired record


def test_unknown_label_raises(cell, monkeypatch):
    real = pipeline.load_records

    def with_unknown(*args, **kwargs):
        records = real(*args, **kwargs)
        return [records[0].model_copy(update={"label": "partial"}), *records[1:]]

    monkeypatch.setattr(pipeline, "load_records", with_unknown)
    with pytest.raises(ValueError, match="partial"):
        audit.record_safety(*CELL)
    assert not (metrics_dir() / "auditkit").exists()


def test_incomplete_labels_raise(cell):
    path = pipeline.judgements_path("qwen3guard", *CELL)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="have no judgement"):
        audit.record_safety(*CELL)
    with pytest.raises(ValueError, match="complete=False"):
        audit.record_safety(*CELL, judge="second")
    with pytest.raises(ValueError, match="text"):
        audit.record_safety(*CELL, text="raw")


def test_identical_prompt_texts_stay_distinct(cell):
    result = auditkit.RunResult.load(str(audit.record_safety(*CELL, text="full")))
    two, three = (next(p for p in result.predictions if p.sample_id == pid) for pid in PIDS[1:3])
    assert two.prompt == three.prompt == TEXTS[PIDS[1]]
    assert (two.raw_output, three.raw_output) == (OUTPUTS[1], OUTPUTS[2])
    assert two.metadata["scores"][0]["metadata"]["key"] != three.metadata["scores"][0]["metadata"]["key"]
    assert [s["label"] for s in (two.metadata["scores"][0], three.metadata["scores"][0])] == LABELS[1:3]


def test_file_is_lf_utf8_and_loads_back(cell):
    path = audit.record_safety(*CELL)
    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    assert TEXTS[PIDS[3]].encode("utf-8") in raw                     # Devanagari kept, not \u-escaped
    loaded = auditkit.RunResult.load(str(path))
    assert stable_json(json.loads(raw)) == stable_json(json.loads(json.dumps(loaded.to_dict(), default=str)))
    assert [p.name for p in path.parent.iterdir()] == [path.name]    # no provenance or temp files


def test_redacted_record_holds_no_output_text(cell):
    redacted = audit.record_safety(*CELL)
    full = audit.record_safety(*CELL, text="full")
    assert full == redacted.with_name("harmful_hi_dev.full.json")
    red_text = redacted.read_text(encoding="utf-8")
    full_text = full.read_text(encoding="utf-8")
    for output in OUTPUTS:
        assert output not in red_text and output in full_text
        assert audit.redacted(output) in red_text
    red, whole = (auditkit.RunResult.load(str(p)) for p in (redacted, full))
    assert red.headline == whole.headline and red.fingerprint != whole.fingerprint


def test_auditkit_writes_nothing_outside_outputs(cell):
    audit.record_safety(*CELL)
    audit.record_safety(*CELL)                       # a second run must not be served from a cache either
    for d in cell.values():
        assert list(d.rglob("*")) == []
