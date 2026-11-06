"""lift publish datasets: refusal texts only, never a text any judge or human labelled harmful; raw
completions gated with manual approval; only our own translations in the prompt sets."""

from __future__ import annotations

import json

import pytest
from publish_fakes import write_inputs

from lift import paths
from lift.io import read_jsonl, write_jsonl
from lift.publish import PublishError, datasets

VARIANT = "adapter-aya-std-s1"
CELL = "harmful_hi_dev"
# key -> (output, primary label, primary native safety, second judge label, human label)
CASES = {
    "k1": ("I cannot help with that request.", "refusal", "Safe", None, None),
    "k2": ("HARMFUL-TEXT-K2 step one ...", "harmful", "Unsafe", None, None),
    "k3": ("Sorry, no. HARMFUL-TEXT-K3", "refusal", "Safe", "harmful", None),
    "k4": ("I won't. HARMFUL-TEXT-K4", "refusal", "Safe", None, "harmful"),
    "k5": ("Here is a benign answer.", "benign", "Safe", None, None),
    "k6": ("No. HARMFUL-TEXT-K6", "refusal", "Controversial", None, None),
}


def _judgement(key, label, safety, judge):
    return {
        "key": key,
        "variant": VARIANT,
        "pid": f"prh-test-{key}",
        "lang": "hi",
        "set": "harmful",
        "split": "dev",
        "judge": judge,
        "judge_label": label,
        "label": label,
        "flags": {"too_short": False, "repetitive": False, "garbled": False, "wrong_lang": False},
        "native": {"safety": safety},
        "raw": "",
        "output_sha": None,
    }


@pytest.fixture
def outputs():
    primary, second = datasets.role_dir("primary"), datasets.role_dir("second")
    write_jsonl(
        paths.generations_dir() / VARIANT / f"{CELL}.jsonl",
        [
            {
                "key": k,
                "variant": VARIANT,
                "pid": f"prh-test-{k}",
                "output": out,
                "hit_cap": False,
                "n_new_tokens": 9,
            }
            for k, (out, *_rest) in CASES.items()
        ],
    )
    write_jsonl(
        paths.judgements_dir() / primary / VARIANT / f"{CELL}.jsonl",
        [_judgement(k, label, safety, "q@x#bf16#strict") for k, (_, label, safety, _s, _h) in CASES.items()],
    )
    write_jsonl(
        paths.judgements_dir() / second / VARIANT / f"{CELL}.jsonl",
        [_judgement(k, s, "", "f@x") for k, (_, _, _, s, _h) in CASES.items() if s],
    )
    keys = paths.validation_dir() / "keys"
    keys.mkdir(parents=True)
    (keys / "round1.json").write_text(json.dumps({"hi-aaa": {"key": "k4", "variant": VARIANT}}))
    write_jsonl(
        paths.validation_dir() / "labels" / "r1_hi_a.jsonl",
        [{"item_id": "hi-aaa", "annotator": "reader-1", "label": "harmful"}],
    )
    audit = paths.metrics_dir() / "auditkit" / primary / VARIANT
    audit.mkdir(parents=True)
    (audit / f"{CELL}.json").write_text(
        json.dumps({"predictions": [{"actual_output": "[redacted sha256:ab]"}]})
    )
    (audit / f"{CELL}.full.json").write_text(
        json.dumps({"predictions": [{"actual_output": "HARMFUL-TEXT-K2"}]})  # preflight: allow (fake)
    )
    return primary


def _texts(folder):
    out = []
    for f in folder.rglob("*"):
        if f.is_file():
            out.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def test_logs_keep_only_clean_refusals(outputs):
    up = datasets.build_logs()
    rows = list(read_jsonl(up.folder / "labels" / f"{VARIANT}.jsonl"))
    assert len(rows) == len(CASES)  # every judged prompt stays in the record
    texts = {r["key"]: r["refusal_text"] for r in rows}
    assert texts == {"k1": CASES["k1"][0], "k2": None, "k3": None, "k4": None, "k5": None, "k6": None}
    by = {r["key"]: r for r in rows}
    assert by["k3"]["second_judge_label"] == "harmful" and by["k4"]["human_labels"] == ["harmful"]
    staged = _texts(up.folder)
    assert "HARMFUL-TEXT" not in staged and "reader-1" not in staged
    assert (up.folder / "auditkit" / outputs / VARIANT / f"{CELL}.json").is_file()
    assert not list(up.folder.rglob("*.full.json"))
    assert list(read_jsonl(up.folder / "examples" / "refusals.jsonl"))[0]["key"] == "k1"
    assert up.repo.gated is False


def test_check_public_stops_a_tampered_row(outputs):
    up = datasets.build_logs()
    path = up.folder / "labels" / f"{VARIANT}.jsonl"
    rows = list(read_jsonl(path))
    for r in rows:
        if r["key"] == "k4":
            r["refusal_text"] = "looks innocent"
    write_jsonl(path, rows)
    keys, texts = datasets.harmful_record()
    with pytest.raises(PublishError, match="labelled harmful"):
        datasets.check_public(up.folder, keys, texts)


def test_check_public_catches_a_harmful_text_under_another_key(tmp_path, outputs):
    write_jsonl(tmp_path / "x.jsonl", [{"key": "other", "label": "refusal", "refusal_text": CASES["k2"][0]}])
    keys, texts = datasets.harmful_record()
    with pytest.raises(PublishError):
        datasets.check_public(tmp_path, keys, texts)


def test_aggregates_are_copied_as_csv(outputs):
    write_inputs()
    up = datasets.build_logs()
    assert (up.folder / "aggregates" / "per_language.csv").read_text().startswith("lang,role,set,")
    assert (up.folder / "aggregates" / "selection.json").is_file()
    card = (up.folder / "README.md").read_text()
    assert "**repair**" in card and "[PENDING" not in card


def test_raw_is_gated_manual_and_keeps_everything(outputs):
    up = datasets.build_raw()
    assert up.repo.gated == "manual"
    assert (up.folder / "generations" / VARIANT / f"{CELL}.jsonl").is_file()
    assert (up.folder / "auditkit" / outputs / VARIANT / f"{CELL}.full.json").is_file()
    assert not (up.folder / "validation").exists()
    card = (up.folder / "README.md").read_text()
    assert "extra_gated_prompt" in card


def test_prompts_keep_only_our_translations():
    ours = {
        "pid": "prh-test-0001",
        "lang": "hi",
        "set": "harmful",
        "split": "dev",
        "text": "t",
        "text_en": "e",
        "source": "polyrefuse",
        "translation": {"engine": "google-cloud-v2", "checked": True, "edited": False},
    }
    theirs = {**ours, "lang": "es", "translation": {"engine": "polyrefuse"}}
    write_jsonl(paths.prompts_dir() / "harmful_hi.jsonl", [ours])
    write_jsonl(paths.prompts_dir() / "harmful_es.jsonl", [theirs])
    write_jsonl(paths.prompts_dir() / "harmful_en.jsonl", [{**ours, "lang": "en", "translation": None}])
    up = datasets.build_prompts()
    assert sorted(p.relative_to(up.folder).as_posix() for p in up.folder.rglob("*.jsonl")) == [
        "harmful/hi.jsonl"
    ]
    card = (up.folder / "README.md").read_text()
    assert "CC-BY-4.0" in card and "OR-Bench" in card and "| harmful | hi | 1 |" in card


def test_build_refuses_unknown_names():
    with pytest.raises(PublishError):
        datasets.build(["everything"])
