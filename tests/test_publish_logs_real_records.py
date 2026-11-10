"""lift publish datasets on a cell written by the real record classes (GenerationRecord, JudgementRecord)
and an AuditKIT record written by lift.audit, read through the real reporting door: the other publish
tests write raw rows and a stand-in for load_records. Placeholder text only."""

from __future__ import annotations

import json

import pytest

auditkit = pytest.importorskip("auditkit")

from publish_fakes import no_committed_recipe  # noqa: E402, F401
from test_audit import CELL, FAILED, GUARD_ID, LABELS, OUTPUTS, PIDS, RULES, STACK, cell, xdg  # noqa: E402, F401

from lift import audit, paths  # noqa: E402
from lift.io import read_jsonl, sha256_text, write_jsonl  # noqa: E402
from lift.judge import pipeline  # noqa: E402
from lift.publish import datasets, preflight  # noqa: E402
from lift.schemas import JudgementRecord, UnclearFlags  # noqa: E402

VARIANT, SET, LANG, SPLIT = CELL
KEYS = {pid: sha256_text(f"{VARIANT}|{pid}") for pid in PIDS}
FRONTIER_ID = "anthropic/claude-sonnet-5-5@safety_v1:0123456789ab"


def _validated(pid: str, human: str, second: str, *, failure: str | None = None) -> None:
    """One round-1 item on ``pid``: a reader's label, the second judge's key list and its judgement."""
    keys = paths.validation_dir() / "keys"
    (keys / "second_judge").mkdir(parents=True)
    item = {"key": KEYS[pid], "variant": VARIANT, "sheet": "round1_hi", "lang": LANG}
    (keys / "round1.json").write_text(json.dumps({"r1hi-a": item}), encoding="utf-8")
    write_jsonl(
        paths.validation_dir() / "labels" / "round1_hi.jsonl",
        [{"item_id": "r1hi-a", "annotator": "reader-1", "label": human}],
    )
    (keys / "second_judge" / "all.txt").write_text(KEYS[pid] + "\n", encoding="utf-8")
    native = {"label": second, "reason": "PLACEHOLDER-REASON the response does this and that"}
    if failure:
        native["judge_failure"] = failure
    record = JudgementRecord(
        key=KEYS[pid],
        variant=VARIANT,
        pid=pid,
        lang=LANG,
        set=SET,
        split=SPLIT,
        judge=FRONTIER_ID,
        judge_label=second,
        label=second,
        flags=UnclearFlags(too_short=False, repetitive=False, garbled=False, wrong_lang=False),
        native=native,
        raw='{"label": "' + second + '", "reason": "PLACEHOLDER-RAW-REPLY"}',
        output_sha=sha256_text(OUTPUTS[PIDS.index(pid)]),
    )
    write_jsonl(pipeline.judgements_path(pipeline.judge_dir("second"), *CELL), [record.model_dump()])


def test_the_logs_build_from_real_records(cell):  # noqa: F811
    """Every judged prompt becomes one row of ids, labels, flags and hashes; the judges' replies, reasons
    and failure texts and every output stay out; the AuditKIT record goes in because the real
    load_records reads its cell as it stands and the read hashes to the record."""
    record = audit.record_safety(*CELL)
    _validated(PIDS[1], "refusal", "unclear", failure="the provider refused: PLACEHOLDER-FAILURE-TEXT")
    up = datasets.build_logs()
    rows = {r["pid"]: r for r in read_jsonl(up.folder / "labels" / f"{VARIANT}.jsonl")}
    assert list(rows) == PIDS and all(set(r) == set(datasets.LOG_FIELDS) for r in rows.values())
    assert [r["label"] for r in rows.values()] == LABELS
    assert {r["judge"] for r in rows.values()} == {f"{GUARD_ID}#{RULES}#bf16#strict"}
    assert [bool(r["judge_failure"]) for r in rows.values()] == FAILED
    assert rows[PIDS[2]]["judge_failure"] == "unparseable"  # test_audit's failed judgement: its kind
    assert all(
        r["judged_current_output"] and r["method"] == "adapter" and r["seed"] == 1 for r in rows.values()
    )
    second = rows[PIDS[1]]
    assert (second["second_judge_label"], second["second_judge_failure"], second["human_labels"]) == (
        "unclear",
        "the provider refused",
        ["refusal"],
    )
    assert rows[PIDS[0]]["second_judge_label"] is None and rows[PIDS[0]]["human_labels"] == []
    staged = "".join(
        f.read_text(encoding="utf-8", errors="replace") for f in up.folder.rglob("*") if f.is_file()
    )
    assert "placeholder output" not in staged and "PLACEHOLDER-" not in staged and "reader-1" not in staged
    assert "lid_top" not in staged and STACK["torch"] not in json.dumps(
        list(rows.values())
    )  # no native block
    copied = up.folder / "auditkit" / record.relative_to(paths.metrics_dir() / "auditkit")
    assert copied.read_bytes() == record.read_bytes()
    assert not [n for n in up.notes if "AuditKIT records left out" in n] and not up.blocked
    assert preflight.scan_folder(up.folder, set()) == []
    assert datasets.check_public(up.folder, set()) == len(PIDS)


def test_a_record_of_a_cell_that_changed_is_left_out_of_the_real_logs(cell):  # noqa: F811
    record = audit.record_safety(*CELL)
    _validated(PIDS[1], "refusal", "refusal")
    judged = pipeline.judgements_path(pipeline.judge_dir("primary"), *CELL)
    rows = list(read_jsonl(judged))
    changed = {"label": "benign", "judge_label": "benign"}
    write_jsonl(judged, [{**r, **changed} if r["pid"] == PIDS[0] else r for r in rows])
    up = datasets.build_logs()
    assert not (up.folder / "auditkit").exists() and record.is_file()
    note = next(n for n in up.notes if "1 AuditKIT records left out" in n)
    assert "the cell's labels are not the ones it recorded" in note
    labels = {r["pid"]: r["label"] for r in read_jsonl(up.folder / "labels" / f"{VARIANT}.jsonl")}
    assert labels[PIDS[0]] == "benign"  # the rows are the cell as it stands
