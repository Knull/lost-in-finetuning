"""lift.audit.record_utility against the installed AuditKIT, offline, placeholder texts only."""

from __future__ import annotations

import json

import pytest

auditkit = pytest.importorskip("auditkit")

from fakes import make_prompt  # noqa: E402

from lift import audit, sync, utility  # noqa: E402
from lift.io import sha256_file, sha256_obj, sha256_text, stable_json, write_jsonl  # noqa: E402
from lift.models.generate import generations_path  # noqa: E402
from lift.paths import metrics_dir, output_dir  # noqa: E402
from lift.schemas import GenerationRecord  # noqa: E402
from lift.stats.paired import win_rate  # noqa: E402

CANDIDATE = "adapter-indicalign-std-s1"
SPLIT = "dev"
PIDS = ["utl-aya-00010", "utl-aya-00020", "utl-ind-00030", "utl-ind-00040", "utl-ind-00050"]
TEXTS = {pid: f"निर्देश {i}: एक कविता लिखिए" for i, pid in enumerate(PIDS)}
CAND = {pid: f"नया उत्तर संख्या {i}" for i, pid in enumerate(PIDS)}
STOCK = {pid: f"पुराना उत्तर संख्या {i}" for i, pid in enumerate(PIDS)}
JUDGE_ID = "fakeprovider/utility-model@utility_v1:0123456789ab+apifake"
FAILED = "unparseable judge output"
# (winner with the candidate shown as A, as B); None is a failed order, stored as a tie
ORDERS = dict(zip(PIDS, [("A", "B"), (None, "B"), ("B", "A"), (None, None), ("A", "TIE")], strict=True))
SCORES = [1.0, 0.75, 0.0, 0.5, 0.75]
FAILED_ORDER_RATE = (0 + 0.5 + 0 + 1 + 0) / 5
META = {"dtype": "bf16", "model_rev": "c" * 40, "harness": 1, "template": "9" * 16, "decoding": "d" * 16,
        "torch": "2.15.0", "transformers": "5.22.1", "gpu": "NVIDIA A100-SXM4-40GB", "gpu_capability": "8.0"}


def write_side(variant: str, outputs: dict[str, str]) -> None:
    rows = [GenerationRecord(key=sha256_text(f"{variant}|{pid}|{out}"), variant=variant, pid=pid, lang="hi",
                             set="utility", split=SPLIT, prompt_sha=sha256_text(TEXTS[pid]), output=out,
                             n_prompt_tokens=400, n_new_tokens=30, hit_cap=False, seconds=0.1,
                             meta={**META, "run_config": f"rc-{variant}"}).model_dump()
            for pid, out in outputs.items()]
    write_jsonl(generations_path(variant, "utility", "hi", SPLIT), reversed(rows))


def result(winner: str | None) -> dict:
    if winner is None:
        return {"winner": None, "reason": None, "raw": "", "error": FAILED}
    return {"winner": winner, "reason": "placeholder reason", "raw": "{}", "error": None}


def write_verdicts() -> list:
    pairing = utility.pair_generations(CANDIDATE, SPLIT)
    verdicts = [utility.make_verdict(p, result(ORDERS[p.pid][0]), result(ORDERS[p.pid][1]), variant=CANDIDATE,
                                     split=SPLIT, judge_id=JUDGE_ID) for p in pairing.pairs]
    utility.write_verdicts(utility.verdicts_path("frontier", CANDIDATE, SPLIT), verdicts)
    return verdicts


@pytest.fixture
def xdg(tmp_path, monkeypatch):
    """AuditKIT's DiskCache and ExperimentDB would write here; both must stay empty."""
    dirs = {name: tmp_path / name for name in ("xdg-cache", "xdg-data")}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(dirs["xdg-cache"]))
    monkeypatch.setenv("XDG_DATA_HOME", str(dirs["xdg-data"]))
    return dirs


@pytest.fixture
def cell(fake_prompt_files, xdg):
    fake_prompt_files[("utility", "hi")] = [make_prompt(p, "hi", "utility", SPLIT, text=TEXTS[p])
                                            for p in PIDS]
    write_side(CANDIDATE, CAND)
    write_side("stock", STOCK)
    return write_verdicts()


def test_the_headline_is_lift_stats_win_rate_and_the_failed_order_share(cell):
    path = audit.record_utility(CANDIDATE, SPLIT)
    assert path == metrics_dir() / "auditkit_utility" / "frontier" / CANDIDATE / "utility_hi_dev.json"
    result = auditkit.RunResult.load(str(path))
    expected = win_rate(SCORES, clusters=PIDS, n_boot=10_000)
    assert [v.score for v in cell] == SCORES
    assert result.headline == {"win_rate": expected["win_rate"], "failed_order_rate": FAILED_ORDER_RATE}
    counts = {name: stat.count for name, stat in result.stats.items()}
    assert counts == {"win_rate": 5, "failed_order_rate": 5}
    assert [p.sample_id for p in result.predictions] == PIDS
    assert [p.correct for p in result.predictions] == [True, True, False, False, True]   # the pair is won
    primary = [p.metadata["scores"][0] for p in result.predictions]
    assert [s["label"] for s in primary] == ["win", "win", "loss", "tie", "win"]
    for score, v in zip(primary, cell, strict=True):           # each pair's verdict is in the record
        assert score["metadata"] == {
            "order_ab": v.order_ab, "order_ba": v.order_ba, "candidate_sha": v.candidate_sha,
            "stock_sha": v.stock_sha, "key": v.candidate_key, "stock_key": v.stock_key,
            "failure_ab": FAILED if v.error_ab else None, "failure_ba": FAILED if v.error_ba else None}
    assert primary[1]["metadata"]["failure_ab"] == FAILED and primary[3]["metadata"]["failure_ba"] == FAILED
    stock_keys = [sha256_text(f"stock|{p}|{STOCK[p]}") for p in PIDS]
    assert [s["metadata"]["stock_key"] for s in primary] == stock_keys
    assert result.model_spec == f"lift:{CANDIDATE}" and result.failed_count == 0 and not result.errors


def test_the_lift_metadata_names_the_verdicts_keys_and_provenance(cell):
    result = auditkit.RunResult.load(str(audit.record_utility(CANDIDATE, SPLIT)))
    lift = result.metadata["lift"]
    expected = win_rate(SCORES, clusters=PIDS, n_boot=10_000)
    assert {k: lift[k] for k in expected} == expected
    assert (lift["failed_orders"], lift["pairs_with_failure"], lift["failed_order_rate"], lift["n_boot"]) == (
        3, 2, FAILED_ORDER_RATE, 10_000)
    assert (lift["variant"], lift["reference"], lift["set"], lift["lang"], lift["split"]) == (
        CANDIDATE, "stock", "utility", "hi", SPLIT)
    assert (lift["judge_dir"], lift["judge_id"], lift["dtype"], lift["text"]) == (
        "frontier", JUDGE_ID, "bf16", "redacted")
    verdicts = utility.verdicts_path("frontier", CANDIDATE, SPLIT)
    assert lift["verdicts_path"] == f"outputs/utility/frontier/{CANDIDATE}/utility_hi_dev.jsonl"
    assert lift["verdicts_sha256"] == sha256_file(verdicts)
    assert lift["scores_sha256"] == sha256_obj(dict(zip(PIDS, SCORES, strict=True)))
    assert lift["keys_sha256"] == sha256_obj({p: sha256_text(f"{CANDIDATE}|{p}|{CAND[p]}") for p in PIDS})
    assert lift["stock_keys_sha256"] == sha256_obj({p: sha256_text(f"stock|{p}|{STOCK[p]}") for p in PIDS})
    assert lift["provenance"]["candidate"]["run_config"] == [f"rc-{CANDIDATE}"]
    assert lift["provenance"]["stock"]["model_rev"] == ["c" * 40]
    extra = result.config.extra["lift"]
    assert extra["scores_sha256"] == lift["scores_sha256"] and "win_rate" not in extra


def test_the_redacted_record_holds_no_answer_and_the_full_one_only_the_candidates(cell):
    redacted = audit.record_utility(CANDIDATE, SPLIT)
    full = audit.record_utility(CANDIDATE, SPLIT, text="full")
    assert full == redacted.with_name("utility_hi_dev.full.json")
    red_text, full_text = (p.read_text(encoding="utf-8") for p in (redacted, full))
    for pid in PIDS:
        assert CAND[pid] not in red_text and audit.redacted(CAND[pid]) in red_text and CAND[pid] in full_text
        assert STOCK[pid] not in red_text and STOCK[pid] not in full_text      # named by key and sha256 only
        assert sha256_text(STOCK[pid]) in red_text
    red, whole = (auditkit.RunResult.load(str(p)) for p in (redacted, full))
    assert red.headline == whole.headline and red.fingerprint != whole.fingerprint


def test_the_file_is_lf_utf8_alone_in_its_folder_and_loads_back(cell):
    path = audit.record_utility(CANDIDATE, SPLIT)
    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    assert TEXTS[PIDS[0]].encode("utf-8") in raw                      # Devanagari kept, not \u-escaped
    loaded = auditkit.RunResult.load(str(path))
    assert stable_json(json.loads(raw)) == stable_json(json.loads(json.dumps(loaded.to_dict(), default=str)))
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_verdicts_on_answers_made_again_or_missing_write_nothing(cell):
    write_side(CANDIDATE, {**CAND, PIDS[0]: "दोबारा बना उत्तर"})       # made again after the judging
    with pytest.raises(utility.UtilityRefused, match="1 of 5 pairs have no verdict"):
        audit.record_utility(CANDIDATE, SPLIT)
    with pytest.raises(FileNotFoundError):
        audit.record_utility(CANDIDATE, SPLIT, judge_dir="fallback")
    with pytest.raises(ValueError, match="text"):
        audit.record_utility(CANDIDATE, SPLIT, text="raw")
    with pytest.raises(ValueError, match="one of"):
        audit.utility_record_path("primary", CANDIDATE, SPLIT)
    assert not (metrics_dir() / "auditkit_utility").exists()


def test_a_headline_that_differs_from_ours_is_refused(cell, monkeypatch):
    real = win_rate
    monkeypatch.setattr("lift.stats.paired.win_rate", lambda *a, **k: {**real(*a, **k), "win_rate": 0.123})
    with pytest.raises(RuntimeError, match="headline differs"):
        audit.record_utility(CANDIDATE, SPLIT)
    assert not (metrics_dir() / "auditkit_utility").exists()


def test_the_record_is_outside_the_folder_sync_pairs_with_safety_judgements(cell):
    path = audit.record_utility(CANDIDATE, SPLIT)
    name = "outputs/" + path.relative_to(output_dir()).as_posix()
    assert name == f"outputs/metrics/auditkit_utility/frontier/{CANDIDATE}/utility_hi_dev.json"
    assert sync._recorded_from(name) == ()
    assert sync._recorded_from(f"outputs/metrics/auditkit/frontier/{CANDIDATE}/utility_hi_dev.json")
    tagged = audit.utility_record_path("frontier", CANDIDATE, SPLIT, text="full", dtype="fp16")
    assert tagged.name == "utility_hi_dev-fp16.full.json"


def test_auditkit_writes_nothing_outside_outputs(cell, xdg):
    audit.record_utility(CANDIDATE, SPLIT)
    audit.record_utility(CANDIDATE, SPLIT)          # a second run must not be served from a cache either
    for d in xdg.values():
        assert list(d.rglob("*")) == []
