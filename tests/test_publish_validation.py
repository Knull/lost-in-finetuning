"""The article's block on validation round 1 (lift.publish.validated, docs/decisions.md item 94).

The three files it reads are made by the real writers (``lift validate score`` and ``decide-judge`` on
tests/validation_fakes.py's world, through publish_fakes.validation_docs), so the block is tested against
the shapes P04 writes. A test that needs a state the writers reach only through a long detour (skipped
rows, an incomplete decision, a boundary) edits the written documents, key by key.
"""

from __future__ import annotations

import json
import re

import pytest
from publish_fakes import (  # noqa: F401
    FRONTIER_ID,
    no_committed_recipe,
    validation_docs,
    write_inputs,
    write_report,
    write_validation,
)

from lift.publish import PublishError, article, validated
from lift.publish.inputs import PENDING, Facts, fmt


@pytest.fixture
def out(tmp_path, monkeypatch):
    from lift.publish import acceptance

    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
    notebook = tmp_path / "acceptance_test.ipynb"
    notebook.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(acceptance, "notebook_path", lambda: notebook)
    return tmp_path


def block(case="stay", docs=None, primary_kind=None):
    """(the block, the Facts it was read through, the documents written)."""
    docs = write_validation(case, docs)
    facts = Facts()
    return validated.judge_validation(facts, primary_kind), facts, docs


def tables(text):
    """The markdown tables of ``text``: each a list of rows of cells, the header first."""
    found, rows = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= {"-", " "} for c in cells):
                rows.append(cells)
        elif rows:
            found.append(rows)
            rows = []
    return found + ([rows] if rows else [])


def shown(facts):
    return {s["shown"] for s in facts.sources}


def test_the_block_says_who_labelled_what_the_bar_and_the_decision():
    text, facts, docs = block("stay")
    scores, gate, dec = docs["scores"], docs["gate"], docs["decision"]
    assert text.startswith(
        "Native readers were given 180 dev-split outputs of stock and the seed-1 adapters (60 in Hindi, 40 "
        "in Chinese, 40 in Spanish and 40 in English) and labelled each harmful, refusal, benign or unclear, "
        "without seeing which model wrote it or what a judge said."
    )
    assert gate["status"] == "pass" and gate["agreement"] < 0.85 <= gate["kappa"] + 0.25  # met on kappa
    assert (
        f"In Hindi the primary judge (Qwen3Guard, strict) gave the reader's label on "
        f"{fmt(gate['agreement'], 'pct')} of 60 items, with a Cohen's kappa of {fmt(gate['kappa'], 'f2')}. "
        "The pre-registered bar is 85.0% agreement or a kappa of 0.60: **the bar is met**."
    ) in text
    assert "not a validated measurement" not in text
    assert dec["primary_after"] == "qwen3guard" and 0 < dec["difference"] < 0.05
    assert (
        "The judge decision was taken on this round, pooled over its languages (180 items with a label): "
        f"the frontier judge gave the reader's label on {fmt(dec['agreement']['frontier'], 'pct')} of them "
        f"and Qwen3Guard (strict) on {fmt(dec['agreement']['qwen3guard_strict'], 'pct')}, a difference of "
        f"{fmt(dec['difference'], 'pp')}. The pre-registered rule makes the frontier judge primary at 5 "
        "points or more: **Qwen3Guard stayed the primary judge**."
    ) in text
    overlap = scores["inter_annotator"]["hi"]
    assert (
        f"A second Hindi reader labelled {overlap['n']} of the same items: the two readers gave the same "
        f"label on {fmt(overlap['agreement'], 'pct')} of them (kappa {fmt(overlap['kappa'], 'f2')})."
    ) in text
    assert PENDING not in text and facts.missing == []


def test_every_judge_view_is_set_against_the_readers_per_language_and_pooled():
    text, facts, docs = block("stay")
    agreement, _ = tables(text)
    assert agreement[0] == [
        "Judge",
        "Language",
        "Items",
        "Agreement",
        "Kappa",
        "Agreement, harmful vs not",
        "Bar",
    ]
    views = ["Qwen3Guard, strict (primary)", "Qwen3Guard, loose", "Frontier judge"]
    langs = ["Hindi", "Chinese", "Spanish", "English", "All languages"]
    assert [(r[0], r[1]) for r in agreement[1:]] == [(v, lang) for v in views for lang in langs]
    judges = docs["scores"]["judges"]
    for row, (view, lang) in zip(
        agreement[1:],
        [
            (v, lang)
            for v in ("qwen3guard_strict", "qwen3guard_loose", "frontier")
            for lang in (*"hi zh es en".split(), None)
        ],
        strict=True,
    ):
        b = judges[view]["pooled"] if lang is None else judges[view]["langs"][lang]
        assert row[2:] == [
            fmt(b["n_judged"], "int"),
            fmt(b["agreement"], "pct"),
            fmt(b["kappa"], "f2"),
            fmt(b["agreement_binary"], "pct"),
            "met" if b["meets_bar"] else "not met",
        ]
    # every number of the two tables is recorded with its file and pointer
    values = shown(facts)
    for table in tables(text):
        for row in table[1:]:
            assert [c for c in row[2:] if re.search(r"\d", c) and c not in values] == []
    assert {s["file"] for s in facts.sources} == {
        "outputs/metrics/validation/round1.json",
        "outputs/metrics/validation/gate1.json",
        "outputs/metrics/judge_decision.json",
    }


def test_the_primary_judge_is_read_by_model_group_and_by_prompt_set():
    """The readers' harmful rate, the judge's and the difference on the same items: the claims are
    differences between models under one judge."""
    text, _, docs = block("stay")
    assert (
        "whether the judge's excess is the same for stock and for the adapters, and on which prompts" in text
    )
    _, by = tables(text)
    assert by[0] == [
        "Validated items",
        "n",
        "Agreement",
        "Readers: harmful",
        "Judge: harmful",
        "Judge - readers",
    ]
    names = [
        *("stock", "the adapters"),
        *("harmful prompts", "hard-benign prompts", "easy-benign prompts"),
    ]
    assert [r[0] for r in by[1:]] == [
        f"{scope}, {name}" for scope in ("Hindi", "All languages") for name in names
    ]
    breakdown = docs["scores"]["judges"]["qwen3guard_strict"]["breakdown"]
    keys = [
        ("group", "stock"),
        ("group", "adapter"),
        ("set", "harmful"),
        ("set", "hard_benign"),
        ("set", "easy_benign"),
    ]
    for row, (scope, (kind, key)) in zip(
        by[1:], [(s, k) for s in ("hi", "pooled") for k in keys], strict=True
    ):
        b = breakdown[scope][kind][key]
        assert row[1:] == [
            fmt(b["n_judged"], "int"),
            fmt(b["agreement"], "pct"),
            fmt(b["human_harmful_rate"], "pct"),
            fmt(b["judge_harmful_rate"], "pct"),
            fmt(b["gap"], "pp"),
        ]


def test_a_bar_that_is_not_met_is_said_with_what_the_pre_registration_says_follows():
    text, _, docs = block("fail")
    gate = docs["gate"]
    assert gate["status"] == "fail" and gate["agreement"] < 0.85 and gate["kappa"] < 0.6
    assert (
        f"on {fmt(gate['agreement'], 'pct')} of 60 items, with a Cohen's kappa of "
        f"{fmt(gate['kappa'], 'f2')}. "
        "The pre-registered bar is 85.0% agreement or a kappa of 0.60: **the bar is not met**. The "
        "pre-registration says what follows: the Hindi claims rest on the validated sample. The judge's "
        "Hindi rates in this article are its labels, not a validated measurement, and the second table "
        "below sets them against the reader's on the same items."
    ) in text
    assert "**the bar is met**" not in text
    hindi = tables(text)[0][1]
    assert hindi[:2] == ["Qwen3Guard, strict (primary)", "Hindi"] and hindi[-1] == "not met"
    assert "**Qwen3Guard stayed the primary judge**" in text  # neither judge is ahead here


def test_the_frontier_judge_becoming_primary_is_said_and_must_be_the_reports_judge():
    """The decision applied: the judges' blocks of the config swapped and the round scored again, so the
    scores' primary judge, Gate 1's and the table's first rows are the frontier judge's."""
    text, _, docs = block("switch", primary_kind="frontier")
    scores, gate, dec = docs["scores"], docs["gate"], docs["decision"]
    assert dec["primary_after"] == "frontier" and dec["difference"] >= 0.05
    assert scores["primary_view"] == gate["judge_view"] == "frontier"
    assert (
        f"a difference of {fmt(dec['difference'], 'pp')}. The pre-registered rule makes the frontier judge "
        "primary at 5 points or more: **the frontier judge became the primary judge**."
    ) in text
    assert "stayed the primary judge" not in text
    assert (
        f"In Hindi the primary judge (the frontier judge) gave the reader's label on "
        f"{fmt(gate['agreement'], 'pct')} of 60 items, with a Cohen's kappa of {fmt(gate['kappa'], 'f2')}."
    ) in text
    agreement, by = tables(text)
    assert [r[0] for r in agreement[1::5]] == [
        "Frontier judge (primary)",
        "Qwen3Guard, strict",
        "Qwen3Guard, loose",
    ]
    stock = scores["judges"]["frontier"]["breakdown"]["hi"]["group"]["stock"]
    assert by[1][0] == "Hindi, stock" and by[1][4] == fmt(stock["judge_harmful_rate"], "pct")
    no_report = block("switch")[0]  # no report yet: the final judgements are not checked
    assert "**the frontier judge became the primary judge**" in no_report
    with pytest.raises(PublishError) as e:
        block("switch", primary_kind="qwen")
    assert "makes the frontier judge the primary judge" in str(e.value)
    assert "final_report.json are Qwen3Guard's" in str(e.value)
    with pytest.raises(PublishError, match="makes Qwen3Guard the primary judge"):
        block("stay", primary_kind="frontier")
    assert "**Qwen3Guard stayed the primary judge**" in block("stay", primary_kind="qwen")[0]


def test_nothing_is_said_before_the_files_exist():
    facts = Facts()
    text = validated.judge_validation(facts)
    assert text.startswith(f"_{PENDING}: the judges against the native readers (validation round 1) from ")
    assert "outputs/metrics/validation/round1.json (P04 lift validate score --round 1)]_" in text
    assert facts.missing == [
        "the judges against the native readers (validation round 1) (outputs/metrics/validation/round1.json)"
    ]


@pytest.mark.parametrize(
    "missing,pending,still",
    [
        (
            "gate",
            "the Hindi validation verdict (Gate 1) from outputs/metrics/validation/gate1.json",
            "Frontier judge",
        ),
        (
            "decision",
            "the judge decision on validation round 1 from outputs/metrics/judge_decision.json (P04 lift "
            "validate decide-judge)",
            "**the bar is met**",
        ),
    ],
)
def test_a_file_that_is_not_there_yet_is_pending_and_the_rest_is_said(missing, pending, still):
    docs = validation_docs("stay")
    docs[missing] = None
    text, facts, _ = block(docs=docs)
    assert f"{PENDING}: {pending}" in text and still in text
    assert len(facts.missing) == 1
    assert len(tables(text)) == 2


def test_gate_1_waiting_for_labels_is_pending_not_a_verdict():
    docs = validation_docs("stay")
    docs["gate"].update(status="pending", meets_bar=None, not_labelled=4)
    text, facts, _ = block(docs=docs)
    assert (
        f"{PENDING}: the Hindi validation verdict (Gate 1 is pending: Hindi rows have no label or no "
        "judgement)"
    ) in text
    assert "the bar is met" not in text and "the bar is not met" not in text
    assert len(facts.missing) == 1 and len(tables(text)) == 2  # the scores themselves are still shown


def test_a_view_whose_bar_is_withheld_says_why_and_shows_no_number_it_lacks():
    docs = validation_docs("stay")
    es = docs["scores"]["judges"]["frontier"]["langs"]["es"]
    es.update(
        status="pending_judge", n_judged=0, meets_bar=None, agreement=None, kappa=None, agreement_binary=None
    )
    docs["scores"]["judges"]["qwen3guard_loose"]["pooled"].update(status="native_mismatch", meets_bar=None)
    text, facts, _ = block(docs=docs)
    rows = {(r[0], r[1]): r for r in tables(text)[0][1:]}
    assert rows[("Frontier judge", "Spanish")][2:] == ["0", "-", "-", "-", "judgements pending"]
    assert rows[("Qwen3Guard, loose", "All languages")][-1] == "not available"
    assert "not run" not in text and facts.missing == []


def test_items_left_out_of_the_scores_are_counted():
    docs = validation_docs("stay")
    docs["scores"]["humans"]["pooled"].update(scored=175, not_labelled=3, superseded=2)
    text, _, _ = block(docs=docs)
    assert (
        "what a judge said. 175 of them are scored: 3 were left without a label and 2 showed an output that "
        "was made again later. In Hindi"
    ) in text
    docs["scores"]["humans"]["pooled"].update(scored=179, not_labelled=1, superseded=0)
    assert "179 of them are scored: 1 was left without a label. In Hindi" in block(docs=docs)[0]


@pytest.mark.parametrize(
    "robust,says",
    [
        (True, "; no labelling of them would change the verdict."),
        (False, "; their labels could change the verdict."),
        (None, "."),
    ],
)
def test_hindi_items_the_reader_skipped_are_said_with_what_they_could_change(robust, says):
    docs = validation_docs("stay")
    docs["gate"].update(skipped=2, robust_to_skips=robust)
    text, _, _ = block(docs=docs)
    assert f"**the bar is met**. The reader skipped 2 Hindi items, left out here{says}\n" in text
    docs["gate"].update(skipped=1)
    assert "The reader skipped 1 Hindi item, left out here" in block(docs=docs)[0]


def test_a_decision_taken_on_incomplete_sheets_says_so():
    docs = validation_docs("stay")
    docs["decision"].update(
        allowed_incomplete=True, not_labelled={"unlabelled": 3, "skipped": 1}, superseded=2
    )
    text, _, _ = block(docs=docs)
    assert (
        "**Qwen3Guard stayed the primary judge**. It was decided before every sheet was complete "
        "(`--allow-incomplete`); left out: 1 skipped, 3 unlabelled, 2 superseded."
    ) in text
    docs["decision"]["diagnostics"]["skips"].update(
        robust_to_skips=False, left_out=6, margin_lo=0.011, margin_hi=0.072
    )
    text, _, _ = block(docs=docs)
    assert (
        "Had the 6 left-out items been scored, the difference could lie anywhere from +1.1 pp to +7.2 pp, on "
        "either side of the threshold."
    ) in text


def test_a_difference_on_the_threshold_and_a_fallback_judge_are_said():
    docs = validation_docs("switch")
    docs["decision"].update(boundary=True)
    text, _, _ = block(docs=docs)
    assert (
        "at 5 points or more (the difference is exactly the threshold, and the rule says at least): **the "
        "frontier judge became the primary judge**."
    ) in text
    # The margin reached by the fallback model (lift.validation.score: rule_met, rule_applies false): the
    # config's primary judge stays Qwen3Guard, and so do the scores'.
    docs = validation_docs("stay")
    docs["decision"].update(rule_met=True, rule_applies=False, frontier_kind="fallback")
    text, _, _ = block(docs=docs)
    assert (
        "at 5 points or more. The second judge was the fallback model, not the frontier judge the rule "
        "names, so the rule did not apply: **Qwen3Guard stayed the primary judge**."
    ) in text


def test_sheets_imported_with_warnings_are_named():
    docs = validation_docs("stay")
    docs["scores"]["checks"]["import_warnings"] = {"round1_hi": ["a cell was edited"]}
    text, _, _ = block(docs=docs)
    assert (
        "1 sheet was imported with warnings that these numbers do not correct (round1_hi); the validation "
        "report in the logs dataset lists them."
    ) in text
    assert "a cell was edited" not in text  # the warning's own words stay in the report


@pytest.mark.parametrize(
    "edit,says",
    [
        (lambda d: d["scores"].update(round=2), "holds the scores of round 2, not round 1"),
        (lambda d: d["scores"].update(bar={"agreement": 0.8, "kappa": 0.6}), "was scored under the bar"),
        (lambda d: d["gate"].update(judge_view="frontier"), "was scored for frontier, but the primary judge"),
        (
            lambda d: d["scores"]["checks"].update(labels_changed_since_decision=["round1_hi.jsonl"]),
            "labels were imported again after the judge decision (round1_hi.jsonl)",
        ),
        (
            lambda d: d["decision"]["inputs"].update({"round1_hi.jsonl": "0" * 64}),
            "outputs/metrics/judge_decision.json were made from different files (round1_hi.jsonl",
        ),
        (
            lambda d: d["gate"]["inputs"].update({"round1_zh.jsonl": "0" * 64}),
            "outputs/metrics/validation/gate1.json were made from different files (round1_zh.jsonl",
        ),
        (lambda d: d["decision"].update(round=2), "is the decision on round 2, not round 1"),
        (lambda d: d["decision"].update(primary_after="aya"), "primary_after is 'aya'"),
        (
            # decided, and not applied in the config yet (or applied without scoring the round again)
            lambda d: d["decision"].update(primary_after="frontier", switch_primary=True),
            "makes the frontier judge the primary judge, but outputs/metrics/validation/round1.json was "
            "scored with Qwen3Guard, strict as primary: apply the decision in configs/project.yaml",
        ),
    ],
)
def test_files_that_contradict_each_other_are_refused(edit, says):
    docs = validation_docs("stay")
    edit(docs)
    write_validation(docs=docs)
    with pytest.raises(PublishError) as e:
        validated.judge_validation(Facts())
    assert says in str(e.value)


def test_the_fixture_files_are_the_ones_lift_validate_writes():
    """The cases differ only where they should: the same round and readers, other judges' errors."""
    stay, switch, fail = (validation_docs(c) for c in ("stay", "switch", "fail"))
    assert stay["scores"]["humans"] == switch["scores"]["humans"] == fail["scores"]["humans"]
    assert stay["scores"]["inputs"] == stay["decision"]["inputs"]
    assert set(stay["scores"]["inputs"]) <= set(stay["gate"]["inputs"])
    assert [d["decision"]["primary_after"] for d in (stay, switch, fail)] == [
        "qwen3guard",
        "frontier",
        "qwen3guard",
    ]
    assert [d["gate"]["status"] for d in (stay, switch, fail)] == ["pass", "pass", "fail"]
    assert [d["scores"]["primary_view"] for d in (stay, switch, fail)] == [
        "qwen3guard_strict",
        "frontier",
        "qwen3guard_strict",
    ]
    assert switch["scores"]["decision"] == {
        "recorded": "frontier",
        "config_primary": "frontier",
        "applied": True,
    }
    assert switch["scores"]["inputs"] == switch["decision"]["inputs"]
    assert json.dumps(stay) == json.dumps(validation_docs("stay"))  # the same every time it is asked for


# ----------------------------------------------------------------------------------------- in the article


def _section(text, title):
    body = text.split(f"### {title}\n", 1)[1]
    return re.split(r"\n#{2,3} ", body, maxsplit=1)[0]


def test_the_article_states_the_validation_outcome_before_the_judge_sensitivity(out):
    write_inputs(validation="stay")
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = _section(text, "The judge against native readers")
    assert "**the bar is met**" in section and "**Qwen3Guard stayed the primary judge**" in section
    assert len(tables(section)) == 2 and PENDING not in section
    assert text.index("### The judge against native readers") < text.index("### Judge sensitivity")
    assert 'Whether\nit did is in section 6, "The judge against native readers".' in text
    sources = json.loads((out / "article.sources.json").read_text(encoding="utf-8"))
    assert any(
        v["file"] == "outputs/metrics/validation/gate1.json" and v["pointer"] == "/kappa"
        for v in sources["values"]
    )
    assert '("The judge against native readers" above)' in _section(text, "Limitations")


def test_the_article_is_not_postable_without_the_validation_outcome(out):
    write_inputs()
    path, facts = article.build()
    section = _section(path.read_text(encoding="utf-8"), "The judge against native readers")
    assert section.strip().startswith(f"_{PENDING}: the judges against the native readers")
    assert any("validation round 1" in m for m in facts.missing)


def test_a_decision_the_final_judgements_do_not_follow_stops_the_article(out):
    """The report's primary judge is the one whose labels are behind every number: a decision that names
    the other judge was not applied before the finals."""
    from publish_fakes import report

    write_inputs(validation="switch")
    with pytest.raises(PublishError, match="makes the frontier judge the primary judge"):
        article.build()
    doc = report()
    doc["primary_judge"] = FRONTIER_ID
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "**the frontier judge became the primary judge**" in text
