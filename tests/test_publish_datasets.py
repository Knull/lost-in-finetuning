"""lift publish datasets: the public logs hold ids, labels, flags and hashes and no output or judge text,
whatever the label (decisions item 62), and every staged file is checked by content against every output
on this machine; the logs refuse an incomplete record (human labels, primary and second judge, open
batches); an AuditKIT record goes in only while it is its cell's; raw completions gated with manual
approval; only our own translations in the prompt sets. Every output here is a made-up marker."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from publish_fakes import no_committed_recipe, report, write_inputs, write_json, write_report  # noqa: F401

from lift import paths
from lift.io import read_jsonl, sha256_obj, sha256_text, write_jsonl
from lift.publish import PublishError, datasets, hub, preflight

VARIANT = "adapter-aya-std-s1"
CELL = "harmful_hi_dev"
FILLER = "fake marker words for a harmful completion " * 3  # long enough for the 60-character windows
LONG_REFUSAL = "REFUSAL-TEXT-K8 I am sorry, but I am not able to help with this request, " * 2
LONG_BENIGN = "BENIGN-TEXT-K9 here is a perfectly ordinary answer about gardening in spring, " * 2
# name -> (output, primary label, primary native safety, second judge label, human label)
CASES = {
    "k1": ("REFUSAL-TEXT-K1 I cannot help with that.", "refusal", "Safe", None, None),
    "k2": ("HARMFUL-TEXT-K2 step one " + FILLER, "harmful", "Unsafe", None, None),
    "k3": ("Sorry, no. HARMFUL-TEXT-K3 " + FILLER, "refusal", "Safe", "harmful", None),
    "k4": ("I won't. HARMFUL-TEXT-K4", "refusal", "Safe", "refusal", "harmful"),
    "k5": ("BENIGN-TEXT-K5 here is an answer.", "benign", "Safe", None, None),
    "k6": ("No. HARMFUL-TEXT-K6", "refusal", "Controversial", None, None),
    # Judged as "refusal" on an older output; the output now under the key was never judged.
    "k7": ("STALE-REGENERATED-OUTPUT never judged", "refusal", "Safe", None, None),
    "k8": (LONG_REFUSAL, "refusal", "Safe", None, None),
    "k9": (LONG_BENIGN, "benign", "Safe", None, None),
}
JUDGED_TEXT = {"k7": "I cannot help with that, sorry."}
MARKERS = ("HARMFUL-TEXT", "STALE-REGENERATED", "REFUSAL-TEXT", "BENIGN-TEXT")
JUDGE_ID = "q@x#bf16#strict"


def K(name: str) -> str:
    """A 64-hex generation key, as lift.cachekey makes them."""
    return sha256_text(name)


NAME = {K(n): n for n in CASES}


def _judgement(name, label, safety, judge, *, lang="hi", set_name="harmful", output=None):
    return {
        "key": K(name),
        "variant": VARIANT,
        "pid": f"prh-test-{name}",
        "lang": lang,
        "set": set_name,
        "split": "dev",
        "judge": judge,
        "judge_label": label,
        "label": label,
        "flags": {"too_short": False, "repetitive": False, "garbled": False, "wrong_lang": False},
        "native": {"safety": safety},
        "raw": "",
        "output_sha": sha256_text(output if output is not None else JUDGED_TEXT.get(name, CASES[name][0])),
    }


def _generation(name, output, *, hit_cap=False):
    return {
        "key": K(name),
        "variant": VARIANT,
        "pid": f"prh-test-{name}",
        "output": output,
        "hit_cap": hit_cap,
        "n_new_tokens": 9,
    }


def _primary_path(cell=CELL):
    return paths.judgements_dir() / datasets.role_dir("primary") / VARIANT / f"{cell}.jsonl"


def _second_path(cell=CELL):
    return paths.judgements_dir() / datasets.role_dir("second") / VARIANT / f"{cell}.jsonl"


def _gen_path(cell=CELL):
    return paths.generations_dir() / VARIANT / f"{cell}.jsonl"


def _second_list():
    return paths.validation_dir() / "keys" / "second_judge" / "all.txt"


def _audit_path(cell=CELL):
    return paths.metrics_dir() / "auditkit" / datasets.role_dir("primary") / VARIANT / f"{cell}.json"


def _audit_record(rows):
    """A redacted AuditKIT record of the judgements ``rows``, with the hashes record_safety writes."""
    meta = {
        "text": "redacted",
        "variant": VARIANT,
        "set": "harmful",
        "lang": "hi",
        "split": "dev",
        "judge": "primary",
        "judge_id": rows[0]["judge"],
        "labels_sha256": sha256_obj({r["pid"]: r["label"] for r in rows}),
        "keys_sha256": sha256_obj({r["pid"]: r["key"] for r in rows}),
        "provenance": {"dtype": ["bf16"]},
    }
    return {"metadata": {"lift": meta}, "samples": [{"actual_output": "[redacted sha256:ab]"}]}


@pytest.fixture
def reporting_door(monkeypatch):
    """lift.judge.pipeline.load_records as the logs call it, over these tests' raw cells (which carry no
    provenance): the judgements of the cell as they stand, sorted by pid. ``calls`` records each read."""
    calls = []

    def load_records(variant, set_name, lang, split, *, judge="primary", complete=True, dtype=None):
        calls.append(
            {"cell": (variant, set_name, lang, split), "judge": judge, "complete": complete, "dtype": dtype}
        )
        path = (
            paths.judgements_dir() / datasets.role_dir(judge) / variant / f"{set_name}_{lang}_{split}.jsonl"
        )
        if not path.is_file():
            raise FileNotFoundError(f"{variant}/{set_name}_{lang}_{split}: no judgements")
        rows = sorted(read_jsonl(path), key=lambda r: r["pid"])
        return [SimpleNamespace(pid=r["pid"], label=r["label"], key=r["key"], judge=r["judge"]) for r in rows]

    monkeypatch.setattr("lift.judge.pipeline.load_records", load_records)
    return calls


@pytest.fixture
def outputs(reporting_door):
    write_jsonl(_gen_path(), [_generation(k, out) for k, (out, *_rest) in CASES.items()])
    primary = [_judgement(k, label, safety, JUDGE_ID) for k, (_, label, safety, _s, _h) in CASES.items()]
    write_jsonl(_primary_path(), primary)
    write_jsonl(
        _second_path(),
        [_judgement(k, s, "", "f@x") for k, (_, _, _, s, _h) in CASES.items() if s],
    )
    _second_list().parent.mkdir(parents=True)
    _second_list().write_text(K("k3") + "\n")
    keys = paths.validation_dir() / "keys"
    (keys / "round1.json").write_text(
        json.dumps({"r1hi-aaa": {"key": K("k4"), "variant": VARIANT, "sheet": "round1_hi", "lang": "hi"}})
    )
    write_jsonl(
        paths.validation_dir() / "labels" / "round1_hi.jsonl",
        [{"item_id": "r1hi-aaa", "annotator": "reader-1", "label": "harmful"}],
    )
    _audit_path().parent.mkdir(parents=True)
    _audit_path().write_text(json.dumps(_audit_record(primary), indent=1))
    kept = {"actual_output": "HARMFUL-TEXT-K2"}  # preflight: allow (fake)
    full = {"metadata": {"lift": {"text": "full"}}, "samples": [kept]}
    _audit_path().with_name(f"{CELL}.full.json").write_text(json.dumps(full))
    return datasets.role_dir("primary")


def _texts(folder):
    out = []
    for f in folder.rglob("*"):
        if f.is_file():
            out.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def _rows(up):
    return {NAME.get(r["key"], r["key"]): r for r in read_jsonl(up.folder / "labels" / f"{VARIANT}.jsonl")}


def _windows():
    return preflight.windows_of(datasets.output_record().texts)


def test_the_logs_hold_no_output_text_whatever_the_label(outputs):
    """Decisions item 62: ids, labels, flags and hashes only. A refusal's text and a benign answer stay
    out exactly like a harmful completion, because the label is one judge's word."""
    up = datasets.build_logs()
    by = _rows(up)
    assert len(by) == len(CASES)  # every judged prompt stays in the record
    assert all(set(r) == set(datasets.LOG_FIELDS) for r in by.values())
    assert {k: r["label"] for k, r in by.items()} == {k: v[1] for k, v in CASES.items()}
    assert by["k3"]["second_judge_label"] == "harmful" and by["k4"]["human_labels"] == ["harmful"]
    assert by["k1"]["output_sha"] == sha256_text(CASES["k1"][0]) and by["k1"]["judged_current_output"]
    staged = _texts(up.folder)
    assert not [m for m in MARKERS if m in staged] and "reader-1" not in staged
    assert (up.folder / "auditkit" / outputs / VARIANT / f"{CELL}.json").is_file()
    assert not list(up.folder.rglob("*.full.json"))
    assert not (up.folder / "examples").exists()  # the Space cuts its own examples (lift.publish.space)
    assert up.repo.gated is False and not up.blocked
    assert "1 human labels read" in up.notes[0] and "no output text and no judge text" in up.notes[0]
    card = (up.folder / "README.md").read_text()
    assert "refusal_text" not in card and "No output text and no judge text" in card


def test_a_stale_judgement_is_marked_not_current(outputs):
    """P08-R1-01: k7 was judged on an older output; the row says so and carries the sha that was judged."""
    row = _rows(datasets.build_logs())["k7"]
    assert row["judged_current_output"] is False
    assert row["output_sha"] == sha256_text(JUDGED_TEXT["k7"])


def test_a_failed_judgement_keeps_only_the_kind_of_its_failure(outputs):
    """The judges' replies, reasons and failure messages stay in the gated repo: a failure's text can
    quote a provider's refusal or the start of a reply that did not parse."""
    rows = list(read_jsonl(_primary_path()))
    for r in rows:
        if r["key"] == K("k5"):
            r["raw"] = "SECRET-RAW-REPLY Safety: Unsafe"
            r["native"] = {
                "judge_failure": "the provider refused: SECRET-FAILURE-TEXT I cannot assess this",
                "reason": "SECRET-JUDGE-REASON the response explains how to",
            }
            r["label"] = r["judge_label"] = "unclear"
    write_jsonl(_primary_path(), rows)
    _audit_path().write_text(json.dumps(_audit_record(rows)))
    second = list(read_jsonl(_second_path()))
    second[0]["native"] = {"judge_failure": "unparseable judge output 'SECRET-SECOND-REPLY'"}
    write_jsonl(_second_path(), second)
    up = datasets.build_logs()
    by = _rows(up)
    assert by["k5"]["judge_failure"] == "the provider refused" and by["k1"]["judge_failure"] is None
    assert by["k3"]["second_judge_failure"] == "unparseable judge output"
    assert "SECRET-" not in _texts(up.folder)
    assert "1 judge failures kept as their kind" in up.notes[0]


def test_check_public_refuses_a_field_the_logs_do_not_publish(outputs):
    up = datasets.build_logs()
    path = up.folder / "labels" / f"{VARIANT}.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["refusal_text"] = "looks innocent"
    write_jsonl(path, rows)
    with pytest.raises(PublishError, match=r"holds the fields \['refusal_text'\]"):
        datasets.check_public(up.folder, _windows())


@pytest.mark.parametrize("value", ["x" * 201, {"safety": "Safe"}, ["refusal", "y" * 201]])
def test_check_public_refuses_a_value_longer_than_an_id_or_a_label(outputs, value):
    up = datasets.build_logs()
    path = up.folder / "labels" / f"{VARIANT}.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["human_labels"] = value
    write_jsonl(path, rows)
    with pytest.raises(PublishError, match=r"public fields \['human_labels'\] hold more than an id"):
        datasets.check_public(up.folder, _windows())


@pytest.mark.parametrize("name", ["k2", "k3", "k8", "k9"])
def test_check_public_catches_an_output_under_any_label(tmp_path, outputs, name):
    """The guard reads content: a harmful completion (k2), a refusal a second judge called harmful (k3), a
    clean refusal (k8) and a benign answer (k9) are refused alike, in any field of any staged row."""
    staged = tmp_path / "staged"
    write_jsonl(staged / "scores" / "x.jsonl", [{"key": "other", "note": "see: " + CASES[name][0]}])
    with pytest.raises(PublishError, match="a run of a model output"):
        datasets.check_public(staged, _windows())
    write_jsonl(staged / "scores" / "x.jsonl", [{"key": "other", "note": "see: " + CASES["k1"][0]}])
    assert datasets.check_public(staged, _windows()) == 0  # under 60 characters: no window to match


def test_the_output_record_reads_every_generation_on_this_machine(outputs):
    """outputs/generations, what lift restart set aside and Gate 0's runs, whatever the label; a judged key
    without a generation is missing, a judged output no longer here is counted."""
    aside = (
        paths.output_dir() / ".aside" / "20261105T000000Z" / "generations" / VARIANT / "redteam_hi_dev.jsonl"
    )
    write_jsonl(aside, [_generation("old", "SET-ASIDE-OUTPUT " + FILLER)])
    gate0 = paths.output_dir() / "gate0" / "bf16" / "precision" / "generations" / "harmful_hi_dev.jsonl"
    write_jsonl(gate0, [_generation("g0", "GATE0-OUTPUT " + FILLER)])
    staged = paths.output_dir() / "publish" / "datasets" / "raw" / "generations" / VARIANT / "x.jsonl"
    write_jsonl(staged, [_generation("st", "STAGED-COPY " + FILLER)])
    found = datasets.output_record()
    assert {v[0].strip() for v in CASES.values()} <= found.texts
    assert ("SET-ASIDE-OUTPUT " + FILLER).strip() in found.texts and (
        "GATE0-OUTPUT " + FILLER
    ).strip() in found.texts
    assert not [t for t in found.texts if t.startswith("STAGED-COPY")]  # the publish staging is not a source
    assert found.gone == 1 and not found.missing  # k7 was judged on a text that is no longer here
    second = paths.judgements_dir() / datasets.role_dir("second") / VARIANT / "redteam_hi_dev.jsonl"
    write_jsonl(second, [_judgement("nowhere", "refusal", "", "f@x", output="an output that is not here")])
    found = datasets.output_record()
    assert found.missing == {K("nowhere")} and found.gone == 2
    assert found.missing_files == [
        f"outputs/judgements/{datasets.role_dir('second')}/{VARIANT}/redteam_hi_dev.jsonl"
    ]
    up = datasets.build_logs()
    assert any("1 judged outputs have no generation on this machine" in why for why in up.blocked)


# ------------------------------------------------------------------------------- P08-R1-02 human labels


@pytest.mark.parametrize("problem", ["no tree", "sheet not imported", "unmapped label", "round 2 scored"])
def test_human_labels_must_be_readable(outputs, problem):
    keys = paths.validation_dir() / "keys" / "round1.json"
    labels = paths.validation_dir() / "labels" / "round1_hi.jsonl"
    if problem == "no tree":
        keys.unlink()
        labels.unlink()
    elif problem == "sheet not imported":
        labels.unlink()
    elif problem == "unmapped label":
        write_jsonl(
            labels, [*read_jsonl(labels), {"item_id": "r1hi-zzz", "annotator": "r", "label": "refusal"}]
        )
    else:
        write_json(paths.metrics_dir() / "validation" / "round2.json", {"round": 2})
    with pytest.raises(PublishError, match="human labels cannot all be read"):
        datasets.build_logs()
    with pytest.raises(PublishError, match="lift sync pull"):
        datasets.human_labels()


def test_no_judgements_needs_no_labels():
    """A fresh checkout (nothing judged, nothing validated) stages an empty logs dataset."""
    up = datasets.build_logs()
    assert not list(up.folder.rglob("*.jsonl"))


# ------------------------------------------------------------------------------ P08-R1-10 completeness


def test_every_generation_needs_a_primary_judgement(outputs):
    write_jsonl(_primary_path(), [r for r in read_jsonl(_primary_path()) if r["key"] != K("k5")])
    with pytest.raises(PublishError, match="1 generations have no primary judgement"):
        datasets.build_logs()


def test_a_generations_cell_never_judged_is_refused(outputs):
    write_jsonl(_gen_path("easy_benign_hi_dev"), [_generation("k1", "fine")])
    with pytest.raises(PublishError, match="easy_benign_hi_dev: 1 generations have no primary judgement"):
        datasets.build_logs()


def test_every_judgement_needs_its_generation(outputs):
    write_jsonl(_gen_path(), [r for r in read_jsonl(_gen_path()) if r["key"] != K("k5")])
    with pytest.raises(PublishError, match="have no generation here"):
        datasets.build_logs()


def _final_cell(name, cell="harmful_hi_final"):
    """A final-split cell holding one judged refusal under key ``name``."""
    text = "I cannot help with that."
    write_jsonl(_gen_path(cell), [_generation(name, text)])
    write_jsonl(
        _primary_path(cell), [{**_judgement(name, "refusal", "Safe", "q@x", output=text), "split": "final"}]
    )


@pytest.mark.parametrize("problem", ["no list", "validated key", "listed key", "stale second judgement"])
def test_second_judge_must_cover_its_keys(outputs, problem):
    second = list(read_jsonl(_second_path()))
    if problem == "no list":
        _second_list().unlink()
    elif problem == "validated key":  # k4 is on a validation sheet, so the second judge must cover it
        write_jsonl(_second_path(), [r for r in second if r["key"] != K("k4")])
    elif problem == "listed key":  # a final-split key on P04's list, never judged by the second judge
        _final_cell("f1")
        _second_list().write_text(f"{K('k3')}\n{K('f1')}\n")
    else:
        write_jsonl(_second_path(), [{**r, "output_sha": sha256_text("older")} for r in second])
    with pytest.raises(PublishError, match="second judge"):
        datasets.build_logs()


def test_the_second_judge_covers_the_final_split_and_validated_keys_only(outputs):
    """XB-06: all.txt lists 10% of every complete cell, dev sweep cells included, but no step judges the
    dev ones; the second judge is reported on the final split (and the validated items), so only those
    are required. A listed dev key without a second judgement does not block the logs."""
    _final_cell("f1")
    _second_list().write_text(f"{K('k3')}\n{K('k1')}\n{K('f1')}\n")  # k1: dev, unjudged by the second
    second = list(read_jsonl(_second_path()))
    f1 = {**_judgement("f1", "refusal", "", "f@x", output="I cannot help with that."), "split": "final"}
    write_jsonl(_second_path("harmful_hi_final"), [f1])
    rows = _rows(datasets.build_logs())
    assert rows["k1"]["second_judge_label"] is None and rows["k3"]["second_judge_label"] == "harmful"
    # the validated key k4 (a dev key) stays required
    write_jsonl(_second_path(), [r for r in second if r["key"] != K("k4")])
    with pytest.raises(PublishError, match="1 keys the second judge must cover"):
        datasets.build_logs()


def test_an_open_batch_blocks_the_logs(outputs):
    state = paths.output_dir() / "judge_cache" / "frontier" / "anthropic__m.batches.jsonl"
    submitted = {"key": "batch:b1:submitted", "event": "submitted", "batch_id": "msgbatch_b1", "keys": []}
    write_jsonl(state, [submitted])
    with pytest.raises(PublishError, match="1 Message Batches are still open"):
        datasets.build_logs()
    write_jsonl(
        state, [submitted, {"key": "batch:b1:collected", "event": "collected", "batch_id": "msgbatch_b1"}]
    )
    datasets.build_logs()


# ------------------------------------------------------------------- P08-R1-11 aggregates and AuditKIT


@pytest.mark.parametrize("problem", ["full text policy", "kept output"])
def test_a_public_auditkit_record_must_be_redacted(outputs, problem):
    path = paths.metrics_dir() / "auditkit" / outputs / VARIANT / f"{CELL}.json"
    doc = json.loads(path.read_text())
    if problem == "full text policy":
        doc["metadata"]["lift"]["text"] = "full"
    else:
        doc["samples"].append({"actual_output": "a kept answer"})  # preflight: allow (fake)
    path.write_text(json.dumps(doc))
    with pytest.raises(PublishError, match="not a redacted AuditKIT record"):
        datasets.build_logs()


@pytest.mark.parametrize("name", ["k3", "k8", "k9"])
def test_the_aggregates_are_scanned_for_any_output_text(outputs, name):
    """A completion quoted in a table note stops the logs, a refusal or a benign answer like any other."""
    doc = report()
    doc["tables"]["per_language"]["note"] = "Example: " + CASES[name][0]
    write_report(doc)
    with pytest.raises(PublishError, match="text of a model output"):
        datasets.build_logs()


def test_an_output_that_repeats_its_prompt_does_not_stop_the_auditkit_record(outputs):
    """P08-R2-03: a redacted AuditKIT record keeps each prompt, and an output may restate its prompt. The
    echoed run is public by design; anything past it still counts."""
    prompt = (
        "PROMPT-TEXT please describe in detail how a garden is prepared for the spring season, step by step"
    )
    write_jsonl(paths.prompts_dir() / "harmful_hi.jsonl", [{"pid": "prh-test-echo", "text": prompt}])
    _add("echo", prompt, "benign")
    record = _audit_record(list(read_jsonl(_primary_path())))
    record["samples"].append({"input": prompt, "actual_output": "[redacted sha256:cd]"})
    _audit_path().write_text(json.dumps(record))
    up = datasets.build_logs()
    assert (up.folder / "auditkit" / outputs / VARIANT / f"{CELL}.json").is_file()


@pytest.mark.parametrize(
    "problem", ["ok", "label changed", "key changed", "another judge", "unreadable", "no meta"]
)
def test_an_auditkit_record_goes_in_only_while_it_is_its_cells_record(outputs, reporting_door, problem):
    """R2-11: a record outlives a change to its cell until lift restart retires it. It goes into the logs
    only while the reporting door reads the cell as it stands and the read hashes to the record's own
    labels and keys, under the record's judge; any other is left out and listed with the reason."""
    rows = list(read_jsonl(_primary_path()))
    if problem == "label changed":
        rows[0]["label"] = "unclear"
    elif problem == "key changed":
        rows[0]["key"] = K("regenerated")
        write_jsonl(_gen_path(), [*read_jsonl(_gen_path()), _generation("regenerated", "fine")])
        write_jsonl(_gen_path(), [g for g in read_jsonl(_gen_path()) if g["key"] != K("k1")])
    elif problem == "another judge":
        rows = [{**r, "judge": "another@judge"} for r in rows]
    if problem in ("label changed", "key changed", "another judge"):
        write_jsonl(_primary_path(), rows)
    elif problem == "unreadable":
        _primary_path().rename(_primary_path().with_suffix(".elsewhere"))
        _gen_path().rename(_gen_path().with_suffix(".elsewhere"))
    elif problem == "no meta":
        _audit_path().write_text(json.dumps({"metadata": {"lift": {"text": "redacted"}}, "samples": []}))
    up = datasets.build_logs()
    staged = up.folder / "auditkit" / outputs / VARIANT / f"{CELL}.json"
    if problem == "ok":
        assert staged.is_file() and not [n for n in up.notes if "AuditKIT records left out" in n]
        assert reporting_door[-1] == {
            "cell": (VARIANT, "harmful", "hi", "dev"),
            "judge": "primary",
            "complete": True,
            "dtype": "bf16",
        }
        return
    assert not staged.exists()
    note = next(n for n in up.notes if "1 AuditKIT records left out" in n)
    expected = {
        "label changed": "the cell's labels are not the ones it recorded",
        "key changed": "the cell's generation keys are not the ones it recorded",
        "another judge": "the cell's judge is another@judge",
        "unreadable": "its cell does not read as it stands (FileNotFoundError",
        "no meta": "its metadata.lift names no variant, set, lang, split, judge, judge_id",
    }[problem]
    assert expected in note and f"auditkit/{outputs}/{VARIANT}/{CELL}.json" in note


def test_the_decision_and_validation_files_go_into_the_aggregates(outputs):
    write_json(paths.metrics_dir() / "gate1.json", {"pass": False})
    write_json(paths.metrics_dir() / "selection_secondary.json", {"status": "decided"})
    write_json(paths.metrics_dir() / "judge_decision.json", {"primary_after": "qwen3guard"})
    write_json(paths.metrics_dir() / "validation" / "round1.json", {"round": 1})
    up = datasets.build_logs()
    for name in ("gate1.json", "selection_secondary.json", "judge_decision.json", "validation/round1.json"):
        assert (up.folder / "aggregates" / name).is_file(), name
    assert not (up.folder / "aggregates" / "gate2.json").exists()  # not made yet: nothing is invented


def test_aggregates_are_copied_as_csv(outputs):
    write_inputs()
    up = datasets.build_logs()
    assert (up.folder / "aggregates" / "per_language.csv").read_text().startswith("lang,n,stock,")
    assert (up.folder / "aggregates" / "selection.json").is_file()
    card = (up.folder / "README.md").read_text()
    assert "**repair**" in card and "[PENDING" not in card


def _bt_path(set_name, lang, variant=VARIANT):
    """P07's back-translation verdicts (lift.final.bt_path)."""
    return paths.output_dir() / "sensitivity" / "backtranslation" / variant / f"{set_name}_{lang}_final.jsonl"


def _bt_row(name, output, label, *, skipped=None, lang="th", set_name="easy_benign"):
    """One row as P07's lift.final._bt_row writes it."""
    return {
        "key": K(name),
        "variant": VARIANT,
        "pid": f"prh-test-{name}",
        "lang": lang,
        "set": set_name,
        "split": "final",
        "output_sha": sha256_text(output),
        "bt_sha": None if skipped else sha256_text("back-translated " + name),
        "judge": None if skipped else "q@x#bf16#strict",
        "judge_label": label,
        "judge_failure": None,
        "skipped": skipped,
    }


# --------------------------------------------------------------------- P08-R1-24 verdict reason sources


def test_verdict_reason_is_recorded_in_the_sources(outputs):
    write_inputs()
    datasets.build_logs()
    doc = json.loads((datasets.stage_root() / "logs.sources.json").read_text())
    pointers = {v["pointer"] for v in doc["values"]}
    assert {"/verdicts/0/reason", "/verdicts/0/verdict", "/verdicts/0/adapter"} <= pointers


# ----------------------------------------------------------------------------------- raw and prompts


def test_raw_is_gated_and_keeps_everything(outputs):
    up = datasets.build_raw()
    assert up.repo.gated == "auto" and not hub.scanned(up)  # decisions item 110
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


# ------------------------------------------------------- P08-R2-05 labels in one vocabulary


def _add(name, output, label, *, safety="Safe", judge_label=None, cell=CELL):
    """One more generation and its primary judgement in ``cell``."""
    gens = list(read_jsonl(_gen_path(cell))) if _gen_path(cell).is_file() else []
    write_jsonl(_gen_path(cell), [*gens, _generation(name, output)])
    judged = list(read_jsonl(_primary_path(cell))) if _primary_path(cell).is_file() else []
    rec = _judgement(name, label, safety, JUDGE_ID, output=output)
    rec["judge_label"] = label if judge_label is None else judge_label
    write_jsonl(_primary_path(cell), [*judged, rec])


def test_label_case_is_normalised(outputs):
    """ "Refusal" is written as "refusal"; a label outside the vocabulary is shown as it was written."""
    _add("cased", "I will not help with this request, sorry.", "Refusal", judge_label=" HARMFUL ")
    _add("odd", "Another answer.", "dangerous")
    rows, _ = datasets.log_rows()
    by = {r["key"]: r for r in rows}
    assert by[K("cased")]["label"] == "refusal" and by[K("cased")]["judge_label"] == "harmful"
    assert by[K("odd")]["label"] == "dangerous"


# --------------------------------------------------------------------- P08-R2-11 duplicated keys


@pytest.mark.parametrize("where", ["primary", "second", "generations"])
def test_a_duplicated_key_in_a_cell_is_refused(outputs, where):
    path = {"primary": _primary_path(), "second": _second_path(), "generations": _gen_path()}[where]
    rows = list(read_jsonl(path))
    write_jsonl(path, [*rows, {**rows[0]}])
    with pytest.raises(
        PublishError, match=r"harmful_hi_dev.jsonl: 1 keys are duplicated.*lift restart --variant"
    ):
        datasets.build_logs()


# ------------------------------------------------------------- P08-R2-15 outputs no longer on this machine


def test_the_preflight_summary_counts_outputs_no_longer_here(outputs, capsys):
    from lift import cli

    second = paths.judgements_dir() / datasets.role_dir("second") / VARIANT / "redteam_hi_dev.jsonl"
    write_jsonl(second, [{**_judgement("k2", "harmful", "", "f@x", output="an older output " + FILLER)}])
    cli.main(["publish", "preflight", "--no-history"])
    out = capsys.readouterr().out
    assert "2 judged outputs are no longer on this machine" in out  # k7's judged text, and this one


# ------------------------------------------- P08-R2-14 scores, back-translation verdicts and manifests


def test_scores_backtranslation_and_manifests_are_in_the_logs(outputs):
    score = {"key": "s1", "variant": VARIANT, "pid": "b-1", "lang": "hi", "task": "belebele", "correct": True}
    write_jsonl(paths.output_dir() / "scores" / "belebele" / VARIANT / "belebele_hi_dev.jsonl", [score])
    out = "I am not able to help."
    write_jsonl(paths.generations_dir() / VARIANT / "easy_benign_th_final.jsonl", [_generation("bt-ok", out)])
    write_jsonl(
        _primary_path("easy_benign_th_final"),
        [{**_judgement("bt-ok", "refusal", "Safe", JUDGE_ID, lang="th", output=out), "split": "final"}],
    )
    bt = {**_bt_row("bt-ok", out, "refusal"), "back_translation": "SECRET-BT-TEXT"}
    failed = {**_bt_row("bt-ok", out, None), "judge_failure": "the provider refused: SECRET-BT-FAILURE"}
    write_jsonl(_bt_path("easy_benign", "th"), [bt, failed])
    write_json(
        paths.output_dir() / "manifests" / "generate-1.json", {"run_id": "generate-1", "wall_seconds": 3}
    )
    up = datasets.build_logs()
    assert list(read_jsonl(up.folder / "scores" / "belebele" / VARIANT / "belebele_hi_dev.jsonl")) == [score]
    row, other = read_jsonl(up.folder / "backtranslation" / VARIANT / "easy_benign_th_final.jsonl")
    assert row["judge_label"] == "refusal" and "back_translation" not in row  # known fields only
    assert row["judge_failure"] is None and other["judge_failure"] == "the provider refused"  # its kind only
    assert "SECRET-BT" not in _texts(up.folder)
    assert (up.folder / "manifests" / "generate-1.json").is_file()
    card = (up.folder / "README.md").read_text()
    assert "which `lift report` wrote from these labels" not in card
    assert "scores/" in card and "backtranslation/" in card and "manifests/" in card


# ----------------------------------------------------------------- P08-R2-09 prompt-set counts sourced


def test_prompt_set_counts_are_recorded_in_the_sources():
    row = {
        "pid": "prh-test-0001",
        "lang": "hi",
        "set": "harmful",
        "split": "dev",
        "text": "t",
        "text_en": "e",
        "source": "polyrefuse",
        "translation": {"engine": "google-cloud-v2", "checked": True, "edited": False},
    }
    write_jsonl(paths.prompts_dir() / "harmful_hi.jsonl", [row, {**row, "pid": "prh-test-0002"}])
    datasets.build_prompts()
    doc = json.loads((datasets.stage_root() / "prompts.sources.json").read_text())
    values = {(v["file"], v["what"]): v["value"] for v in doc["values"]}
    assert values[("data/prompts/harmful_hi.jsonl", "prompts")] == 2
    assert values[("data/prompts/harmful_hi.jsonl", "native-checked")] == 2
    assert values[("data/prompts/harmful_hi.jsonl", "edited")] == 0


def test_the_figures_go_into_the_logs_and_its_card(outputs):
    """XB-08: P07's PNGs are copied into the logs dataset and shown in its card (relative links); one
    not drawn leaves the card pending, so --yes refuses it."""
    from publish_fakes import FIGURES

    write_inputs()
    up = datasets.build_logs()
    card = (up.folder / "README.md").read_text()
    for name in FIGURES:
        assert (up.folder / "figures" / f"{name}.png").read_bytes().startswith(b"\x89PNG")
        assert f"](figures/{name}.png)" in card
    assert "[PENDING" not in card
    (paths.output_dir() / "figures" / "heatmap_harmful.png").unlink()
    up = datasets.build_logs()
    card = (up.folder / "README.md").read_text()
    assert "[PENDING" in card and "heatmap_harmful.png" in card
    assert not (up.folder / "figures" / "heatmap_harmful.png").exists()


# ---------------------------------------------------- no absolute local path goes public (review XB2-04)


def test_absolute_local_paths_are_staged_repo_relative(outputs):
    """XB2-04: P05's freeze plan names a pending CAST artifact by its absolute path, and run manifests
    carry ``out_path``: the staged copies hold them relative to the repo (outputs/..., a GPU VM's
    .../outputs/... too), and nothing staged still names a home dir, /tmp or a drive."""
    from publish_fakes import selection

    from lift.publish import preflight

    sel = selection()
    cast = paths.output_dir() / "artifacts" / "cast" / "adapter-aya-std-s2" / "cast.safetensors"
    sel["freeze"]["variants"][1]["pending"] = f"{cast} not found: run `lift fit-cast`"
    write_inputs(selection_doc=sel)
    out = paths.output_dir() / "utility" / "j" / VARIANT / "utility_hi_final.jsonl"
    manifest = {
        "run_id": "utility-1",
        "out_path": str(out),
        "on_the_vm": "/content/lift/outputs/scores/belebele/x.jsonl",
        "config": str(paths.REPO_ROOT / "configs" / "project.yaml"),
    }
    write_json(paths.output_dir() / "manifests" / "utility-1.json", manifest)
    up = datasets.build_logs()
    staged = json.loads((up.folder / "aggregates" / "selection.json").read_text(encoding="utf-8"))
    assert staged["freeze"]["variants"][1]["pending"] == (
        "outputs/artifacts/cast/adapter-aya-std-s2/cast.safetensors not found: run `lift fit-cast`"
    )
    man = json.loads((up.folder / "manifests" / "utility-1.json").read_text(encoding="utf-8"))
    assert man["out_path"] == f"outputs/utility/j/{VARIANT}/utility_hi_final.jsonl"
    assert man["on_the_vm"] == "outputs/scores/belebele/x.jsonl"
    assert man["config"] == "configs/project.yaml"
    text = _texts(up.folder)
    assert str(paths.output_dir()) not in text and str(paths.REPO_ROOT) not in text
    assert not preflight.local_paths(text)
    assert not [f for f in preflight.scan_folder(up.folder, set()) if f.kind == preflight.LOCAL_PATH_KIND]
    note = next(n for n in up.notes if "absolute local path" in n)
    assert "aggregates/selection.json" in note and "manifests/utility-1.json" in note


@pytest.mark.parametrize(
    "where", ["/home/someone/.cache/lift/x.json", "C:\\Users\\someone\\lift\\x.json", "/tmp/elsewhere/x"]
)
def test_a_local_path_that_cannot_be_made_relative_is_withheld(outputs, where):
    write_inputs()
    write_json(paths.output_dir() / "manifests" / "generate-2.json", {"run_id": "generate-2", "cache": where})
    with pytest.raises(PublishError, match="manifests/generate-2.json.*absolute local path"):
        datasets.build_logs()
