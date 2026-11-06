"""lift publish datasets: refusal texts only, never a text any judge or human labelled harmful, never an
output its judgement did not cover; the logs refuse an incomplete record (human labels, primary and
second judge, open batches); raw completions gated with manual approval; only our own translations in
the prompt sets. Every harmful text here is a made-up marker assembled at run time."""

from __future__ import annotations

import json

import pytest
from publish_fakes import no_committed_recipe, report, write_inputs, write_json, write_report  # noqa: F401

from lift import paths
from lift.io import read_jsonl, sha256_text, write_jsonl
from lift.publish import PublishError, datasets

VARIANT = "adapter-aya-std-s1"
CELL = "harmful_hi_dev"
FILLER = "fake marker words for a harmful completion " * 3  # long enough for the 60-character windows
# name -> (output, primary label, primary native safety, second judge label, human label)
CASES = {
    "k1": ("I cannot help with that request.", "refusal", "Safe", None, None),
    "k2": ("HARMFUL-TEXT-K2 step one " + FILLER, "harmful", "Unsafe", None, None),
    "k3": ("Sorry, no. HARMFUL-TEXT-K3 " + FILLER, "refusal", "Safe", "harmful", None),
    "k4": ("I won't. HARMFUL-TEXT-K4", "refusal", "Safe", "refusal", "harmful"),
    "k5": ("Here is a benign answer.", "benign", "Safe", None, None),
    "k6": ("No. HARMFUL-TEXT-K6", "refusal", "Controversial", None, None),
    # Judged as "refusal" on an older output; the output now under the key was never judged.
    "k7": ("STALE-REGENERATED-OUTPUT never judged", "refusal", "Safe", None, None),
}
JUDGED_TEXT = {"k7": "I cannot help with that, sorry."}


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


@pytest.fixture
def outputs():
    write_jsonl(_gen_path(), [_generation(k, out) for k, (out, *_rest) in CASES.items()])
    write_jsonl(
        _primary_path(),
        [_judgement(k, label, safety, "q@x#bf16#strict") for k, (_, label, safety, _s, _h) in CASES.items()],
    )
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
    audit = paths.metrics_dir() / "auditkit" / datasets.role_dir("primary") / VARIANT
    audit.mkdir(parents=True)
    redacted = {
        "metadata": {"lift": {"text": "redacted"}},
        "samples": [{"actual_output": "[redacted sha256:ab]"}],
    }
    (audit / f"{CELL}.json").write_text(json.dumps(redacted, indent=1))
    kept = {"actual_output": "HARMFUL-TEXT-K2"}  # preflight: allow (fake)
    full = {"metadata": {"lift": {"text": "full"}}, "samples": [kept]}
    (audit / f"{CELL}.full.json").write_text(json.dumps(full))
    return datasets.role_dir("primary")


def _texts(folder):
    out = []
    for f in folder.rglob("*"):
        if f.is_file():
            out.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def _rows(up):
    return {NAME.get(r["key"], r["key"]): r for r in read_jsonl(up.folder / "labels" / f"{VARIANT}.jsonl")}


def test_logs_keep_only_clean_refusals(outputs):
    up = datasets.build_logs()
    by = _rows(up)
    assert len(by) == len(CASES)  # every judged prompt stays in the record
    assert {k: r["refusal_text"] for k, r in by.items()} == {
        k: (CASES["k1"][0] if k == "k1" else None) for k in CASES
    }
    assert by["k3"]["second_judge_label"] == "harmful" and by["k4"]["human_labels"] == ["harmful"]
    assert by["k1"]["output_sha"] == sha256_text(CASES["k1"][0]) and by["k1"]["judged_current_output"]
    staged = _texts(up.folder)
    assert "HARMFUL-TEXT" not in staged and "reader-1" not in staged and "STALE-REGENERATED" not in staged
    assert (up.folder / "auditkit" / outputs / VARIANT / f"{CELL}.json").is_file()
    assert not list(up.folder.rglob("*.full.json"))
    assert list(read_jsonl(up.folder / "examples" / "refusals.jsonl"))[0]["key"] == K("k1")
    assert up.repo.gated is False
    assert "1 human labels read" in up.notes[0]


# ---------------------------------------------------------------------------------- P08-R1-01 output_sha


def test_a_stale_judgement_never_publishes_the_current_output(outputs):
    up = datasets.build_logs()
    row = _rows(up)["k7"]
    assert row["refusal_text"] is None and row["judged_current_output"] is False
    assert "STALE-REGENERATED" not in _texts(up.folder)
    assert "stale" in up.notes[0]


def test_a_judgement_without_output_sha_publishes_no_text(outputs):
    rows = list(read_jsonl(_primary_path()))
    for r in rows:
        r["output_sha"] = None
    write_jsonl(_primary_path(), rows)
    _, counts = datasets.log_rows()
    assert counts.get("refusal_texts", 0) == 0 and counts["texts_stale"] >= 1


def test_check_public_refuses_a_text_its_judgement_did_not_cover(outputs):
    up = datasets.build_logs()
    path = up.folder / "labels" / f"{VARIANT}.jsonl"
    rows = list(read_jsonl(path))
    for r in rows:
        if r["key"] == K("k7"):  # the current output, under a judgement of another output
            r["refusal_text"] = CASES["k7"][0]
            r["output_sha"] = sha256_text(CASES["k7"][0])
    write_jsonl(path, rows)
    keys, texts = datasets.harmful_record()
    with pytest.raises(PublishError, match="not the output its primary judgement covered"):
        datasets.check_public(up.folder, keys, texts)


def test_check_public_stops_a_tampered_row(outputs):
    up = datasets.build_logs()
    path = up.folder / "labels" / f"{VARIANT}.jsonl"
    rows = list(read_jsonl(path))
    for r in rows:
        if r["key"] == K("k4"):
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
        datasets.harmful_record()


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


def test_the_aggregates_are_scanned_for_harmful_text(outputs):
    doc = report()
    doc["tables"]["per_language"]["note"] = "Example: " + CASES["k3"][0]  # a quoted completion in a note
    write_report(doc)
    with pytest.raises(PublishError, match="text of an output labelled harmful"):
        datasets.build_logs()


def test_aggregates_are_copied_as_csv(outputs):
    write_inputs()
    up = datasets.build_logs()
    assert (up.folder / "aggregates" / "per_language.csv").read_text().startswith("lang,n,stock,")
    assert (up.folder / "aggregates" / "selection.json").is_file()
    card = (up.folder / "README.md").read_text()
    assert "**repair**" in card and "[PENDING" not in card


# ------------------------------------------------------------ P08-R1-12, -19: texts in risky cells


def test_risky_refusal_texts_are_withheld(outputs):
    refusal = "I am not able to help with that."
    cases = {  # name -> (lang, set, hit_cap, output)
        "cap": ("hi", "harmful", True, refusal),
        "cap-unknown": ("hi", "harmful", None, refusal),
        "ar-harmful": ("ar", "harmful", False, refusal),
        "th-benign": ("th", "easy_benign", False, refusal),
        # a clean-looking refusal that repeats a run of a harmful output (refuse-then-comply boilerplate)
        "overlap": ("hi", "hard_benign", False, "I refuse. " + CASES["k3"][0][11:]),
    }
    for name, (lang, set_name, cap, out) in cases.items():
        cell = f"{set_name}_{lang}_dev"
        gens = list(read_jsonl(_gen_path(cell))) if _gen_path(cell).is_file() else []
        gen = _generation(name, out, hit_cap=cap)
        if cap is None:
            del gen["hit_cap"]
        write_jsonl(_gen_path(cell), [*gens, gen])
        judged = list(read_jsonl(_primary_path(cell))) if _primary_path(cell).is_file() else []
        write_jsonl(
            _primary_path(cell),
            [*judged, _judgement(name, "refusal", "Safe", "q", lang=lang, set_name=set_name, output=out)],
        )
    rows, counts = datasets.log_rows()
    texts = {r["key"]: r["refusal_text"] for r in rows}
    assert all(texts[K(n)] is None for n in cases)
    assert counts["texts_cap_hit"] == 2 and counts["texts_exploratory"] == 1
    assert counts["texts_backtranslation"] == 1 and counts["texts_overlap"] == 1  # th: no verdict
    assert counts["refusal_texts"] == 1  # k1 only


# ------------------------------------------------- P08-R1-12: ar/th texts need a clean back-translation


def _bt_path(set_name, lang, variant=VARIANT):
    """P07's back-translation verdicts (origin/p07-final-report lift.final.bt_path)."""
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


def test_backtranslation_verdicts_decide_ar_th_texts(outputs):
    refusal = "I am not able to help with that request."
    cell = "easy_benign_th_final"
    names = ["bt-ok", "bt-none", "bt-stale", "bt-harmful", "bt-failed", "bt-unclear"]
    outs = {n: f"{refusal} ({n})" for n in names}
    write_jsonl(_gen_path(cell), [_generation(n, outs[n]) for n in names])
    write_jsonl(
        _primary_path(cell),
        [
            {
                **_judgement(n, "refusal", "Safe", "q", lang="th", set_name="easy_benign", output=outs[n]),
                "split": "final",
            }
            for n in names
        ],
    )
    write_jsonl(
        _bt_path("easy_benign", "th"),
        [
            _bt_row("bt-ok", outs["bt-ok"], "refusal"),
            _bt_row("bt-stale", "an older output of bt-stale", "refusal"),
            _bt_row("bt-harmful", outs["bt-harmful"], "harmful"),
            _bt_row("bt-failed", outs["bt-failed"], None, skipped="back-translation failed"),
            _bt_row("bt-unclear", outs["bt-unclear"], "unclear"),
        ],
    )
    rows, counts = datasets.log_rows()
    by_key = {r["key"]: r["refusal_text"] for r in rows}
    texts = {n: by_key[K(n)] for n in names}
    assert texts["bt-ok"] == outs["bt-ok"]
    assert all(texts[n] is None for n in names if n != "bt-ok")
    assert counts["texts_backtranslation"] == 4  # none, stale, failed, unclear
    # a harmful back-translation verdict is a harmful label like any other judge's
    keys, _texts_ = datasets.harmful_record()
    assert K("bt-harmful") in keys and K("bt-ok") not in keys


def test_check_public_requires_the_backtranslation_verdict(tmp_path, outputs):
    out = "I am not able to help with that request."
    row = {
        "key": K("th1"),
        "label": "refusal",
        "judge_label": "refusal",
        "lang": "th",
        "set": "easy_benign",
        "output_sha": sha256_text(out),
        "refusal_text": out,
    }
    write_jsonl(tmp_path / "labels" / "x.jsonl", [row])
    judged, current = {K("th1"): sha256_text(out)}, {K("th1"): sha256_text(out)}
    with pytest.raises(PublishError, match="back-translation"):
        datasets.check_public(tmp_path, set(), set(), judged=judged, current=current, backtranslated={})
    clean = {K("th1"): [{"output_sha": sha256_text(out), "judge_label": "refusal", "skipped": None}]}
    assert (
        datasets.check_public(tmp_path, set(), set(), judged=judged, current=current, backtranslated=clean)
        == 1
    )


# --------------------------------------------------------------------- P08-R1-24 verdict reason sources


def test_verdict_reason_is_recorded_in_the_sources(outputs):
    write_inputs()
    datasets.build_logs()
    doc = json.loads((datasets.stage_root() / "logs.sources.json").read_text())
    pointers = {v["pointer"] for v in doc["values"]}
    assert {"/verdicts/0/reason", "/verdicts/0/verdict", "/verdicts/0/adapter"} <= pointers


# ----------------------------------------------------------------------------------- raw and prompts


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


# ------------------------------------------------------- P08-R2-05 labels outside the vocabulary


def _add(name, output, label, *, safety="Safe", judge_label=None, cell=CELL):
    """One more generation and its primary judgement in ``cell``."""
    gens = list(read_jsonl(_gen_path(cell))) if _gen_path(cell).is_file() else []
    write_jsonl(_gen_path(cell), [*gens, _generation(name, output)])
    judged = list(read_jsonl(_primary_path(cell))) if _primary_path(cell).is_file() else []
    rec = _judgement(name, label, safety, "q", output=output)
    rec["judge_label"] = label if judge_label is None else judge_label
    write_jsonl(_primary_path(cell), [*judged, rec])


ODD_TEXT = "ODD-LABEL-TEXT " + FILLER


@pytest.mark.parametrize("label", ["Harmful", "HARMFUL", " harmful ", "unsafe", "", None, 1])
def test_a_label_outside_the_vocabulary_counts_as_harmful(outputs, label):
    """A judgement labelled "Harmful" (or anything unknown) is harmful: its text gives windows, so a refusal
    quoting it is withheld, and the key is a harmful key."""
    _add("odd", ODD_TEXT, label)
    _add("quoting", "I refuse. " + ODD_TEXT, "refusal")
    rows, counts = datasets.log_rows()
    texts = {r["key"]: r["refusal_text"] for r in rows}
    assert texts[K("quoting")] is None and counts["texts_overlap"] == 1
    keys, harmful_texts = datasets.harmful_record()
    assert K("odd") in keys and ODD_TEXT.strip() in harmful_texts


@pytest.mark.parametrize("field", ["judge_label", "second", "human"])
def test_an_odd_label_anywhere_counts_as_harmful(outputs, field):
    _add("odd", ODD_TEXT, "refusal", judge_label="Harmful" if field == "judge_label" else None)
    if field == "second":
        rec = _judgement("odd", "Harmful", "", "f@x", output=ODD_TEXT)
        write_jsonl(_second_path(), [*read_jsonl(_second_path()), rec])
    if field == "human":
        keys = paths.validation_dir() / "keys" / "round1.json"
        doc = json.loads(keys.read_text())
        doc["r1hi-odd"] = {"key": K("odd"), "variant": VARIANT, "sheet": "round1_hi", "lang": "hi"}
        keys.write_text(json.dumps(doc))
        labels = paths.validation_dir() / "labels" / "round1_hi.jsonl"
        write_jsonl(
            labels, [*read_jsonl(labels), {"item_id": "r1hi-odd", "annotator": "r", "label": "HARMFUL"}]
        )
        rec = _judgement(
            "odd", "refusal", "", "f@x", output=ODD_TEXT
        )  # a validated key: second judge covers it
        write_jsonl(_second_path(), [*read_jsonl(_second_path()), rec])
    rows, _ = datasets.log_rows()
    assert {r["key"]: r["refusal_text"] for r in rows}[K("odd")] is None
    assert K("odd") in datasets.harmful_record()[0]


def test_label_case_is_normalised(outputs):
    """ "Refusal" is a refusal (its clean text may be public), written as "refusal"."""
    _add("cased", "I will not help with this request, sorry.", "Refusal")
    rows, _ = datasets.log_rows()
    row = {r["key"]: r for r in rows}[K("cased")]
    assert row["label"] == "refusal" and row["refusal_text"] == "I will not help with this request, sorry."


@pytest.mark.parametrize("bt", [("Harmful", None), (None, None), ("dangerous", None)])
def test_an_odd_backtranslation_verdict_counts_as_harmful(outputs, bt):
    label, skipped = bt
    out = "I am not able to help with that request."
    write_jsonl(_bt_path("easy_benign", "th"), [_bt_row("odd-bt", out, label, skipped=skipped)])
    assert K("odd-bt") in datasets.harmful_record()[0]


def test_a_skipped_backtranslation_is_not_a_harmful_label(outputs):
    out = "I am not able to help with that request."
    write_jsonl(_bt_path("easy_benign", "th"), [_bt_row("bt-skip", out, None, skipped="blank output")])
    assert K("bt-skip") not in datasets.harmful_record()[0]


@pytest.mark.parametrize("field", ["judge_label", "second_judge_label", "human_labels", "label"])
def test_check_public_reads_labels_in_any_case(tmp_path, outputs, field):
    out = "I cannot help with that request."
    row = {
        "key": K("k1"),
        "label": "refusal",
        "lang": "hi",
        "output_sha": sha256_text(out),
        "refusal_text": out,
        field: ["Harmful"] if field == "human_labels" else ("Unknown" if field == "label" else "HARMFUL"),
    }
    write_jsonl(tmp_path / "labels" / "x.jsonl", [row])
    judged = current = {K("k1"): sha256_text(out)}
    with pytest.raises(PublishError, match="labelled harmful"):
        datasets.check_public(tmp_path, set(), set(), judged=judged, current=current, backtranslated={})


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


def test_check_public_refuses_duplicated_judgements(outputs):
    up = datasets.build_logs()
    rows = list(read_jsonl(_primary_path()))
    write_jsonl(_primary_path(), [*rows, {**rows[0]}])
    keys, texts = datasets.harmful_record()
    with pytest.raises(PublishError, match="duplicated"):
        datasets.check_public(up.folder, keys, texts)


# ------------------------------------------------------------- P08-R2-15 harmful windows by output_sha


def test_the_harmful_windows_come_from_the_output_judged_harmful(outputs):
    """A harmful verdict on an older output: the text judged harmful (set aside by lift restart) gives the
    windows, not the output regenerated under the key, and one no longer here is counted."""
    judged = "OLD-HARMFUL-OUTPUT " + FILLER
    regenerated = "A NEW OUTPUT never judged harmful, " + FILLER.replace("harmful", "other")
    aside = (
        paths.output_dir() / ".aside" / "20261105T000000Z" / "generations" / VARIANT / "redteam_hi_dev.jsonl"
    )
    write_jsonl(aside, [_generation("re", judged)])
    _add("re", regenerated, "refusal", cell="redteam_hi_dev")
    second = paths.judgements_dir() / datasets.role_dir("second") / VARIANT / "redteam_hi_dev.jsonl"
    write_jsonl(second, [_judgement("re", "harmful", "", "f@x", output=judged)])
    gone = _judgement("gone", "harmful", "", "f@x", output="a harmful output deleted long ago " + FILLER)
    write_jsonl(second, [*read_jsonl(second), {**gone, "key": K("re")}])
    found = datasets.harmful_outputs()
    assert K("re") in found.keys
    assert judged.strip() in found.texts and regenerated.strip() not in found.texts
    assert found.gone == 1 and not found.missing


def test_the_preflight_summary_counts_outputs_no_longer_here(outputs, capsys):
    from lift import cli

    second = paths.judgements_dir() / datasets.role_dir("second") / VARIANT / "redteam_hi_dev.jsonl"
    write_jsonl(second, [{**_judgement("k2", "harmful", "", "f@x", output="an older output " + FILLER)}])
    cli.main(["publish", "preflight", "--no-history"])
    assert "1 outputs labelled harmful are no longer on this machine" in capsys.readouterr().out


# ------------------------------------------- P08-R2-14 scores, back-translation verdicts and manifests


def test_scores_backtranslation_and_manifests_are_in_the_logs(outputs):
    score = {"key": "s1", "variant": VARIANT, "pid": "b-1", "lang": "hi", "task": "belebele", "correct": True}
    write_jsonl(paths.output_dir() / "scores" / "belebele" / VARIANT / "belebele_hi_dev.jsonl", [score])
    bt = {**_bt_row("bt-ok", "I am not able to help.", "refusal"), "back_translation": "SECRET-BT-TEXT"}
    write_jsonl(_bt_path("easy_benign", "th"), [bt])
    write_json(
        paths.output_dir() / "manifests" / "generate-1.json", {"run_id": "generate-1", "wall_seconds": 3}
    )
    up = datasets.build_logs()
    assert list(read_jsonl(up.folder / "scores" / "belebele" / VARIANT / "belebele_hi_dev.jsonl")) == [score]
    [row] = read_jsonl(up.folder / "backtranslation" / VARIANT / "easy_benign_th_final.jsonl")
    assert row["judge_label"] == "refusal" and "back_translation" not in row  # known fields only
    assert "SECRET-BT-TEXT" not in _texts(up.folder)
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
