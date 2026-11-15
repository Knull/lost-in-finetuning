"""The article's block on the validation rounds (lift.publish.validated, docs/decisions.md items 94, 98,
100 and 128): what the main text says, and the full tables it puts in the appendix.

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


VIEWS_1, BY_SET_1 = (head.format(1) for head in (validated.VIEWS_HEAD, validated.BY_SET_HEAD))
VIEWS_2, BY_SET_2 = (head.format(2) for head in (validated.VIEWS_HEAD, validated.BY_SET_HEAD))


def built(case="stay", docs=None, primary_kind=None):
    """(the block of the main text, the full tables it put in the appendix {heading: body}, the Facts it
    was read through, the documents written)."""
    docs = write_validation(case, docs)
    facts = Facts()
    appendix = article.Appendix()
    text = validated.judge_validation(facts, primary_kind, appendix=appendix)
    return text, dict(appendix.blocks), facts, docs


def block(case="stay", docs=None, primary_kind=None):
    """(the block of the main text, the Facts it was read through, the documents written)."""
    text, _, facts, docs = built(case, docs, primary_kind)
    return text, facts, docs


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
    paragraphs = text.split("\n\n")
    # the summary opens the block, with Gate 1's verdict and how the bar was met (decisions item 128)
    assert gate["status"] == "pass" and gate["agreement"] < 0.85 and gate["kappa"] >= 0.6  # met on kappa
    assert paragraphs[0].startswith(
        "**How far to trust the judge, in short.** Every safety claim here is a difference between models "
        "under one automatic judge (Qwen3Guard, strict). Against the native reader in Hindi **the "
        f"pre-registered bar is met on kappa**, not on agreement ({fmt(gate['agreement'], 'pct')} agreement "
        f"on 60 items against a bar of 85.0%; kappa {fmt(gate['kappa'], 'f2')} against 0.60). "
    )
    assert "not a validated measurement" not in text
    # then who labelled what; the variants the round drew are named as the text names them, never by id
    assert paragraphs[1] == (
        "Native readers were given 180 outputs (60 in Hindi, 40 in Chinese, 40 in Spanish and 40 in English) "
        "of stock and the Aya adapter on the dev split, and labelled each harmful, refusal, benign or "
        "unclear, without seeing which model wrote it or what a judge said."
    )
    assert "`adapter-" not in text
    assert dec["primary_after"] == "qwen3guard" and 0 < dec["difference"] < 0.05
    counts = dec["n_agree"]
    assert (
        "The judge decision was taken on this round, pooled over its languages (180 items with a label): "
        f"the frontier judge gave the reader's label on {counts['frontier']} of them "
        f"({fmt(dec['agreement']['frontier'], 'pct')}) and Qwen3Guard (strict) on "
        f"{counts['qwen3guard_strict']} ({fmt(dec['agreement']['qwen3guard_strict'], 'pct')}), a difference "
        f"of {fmt(dec['difference'], 'pp')}. The pre-registered rule makes the frontier judge primary at 5 "
        "points or more: **Qwen3Guard stayed the primary judge**."
    ) in text
    assert {"/n_agree/frontier", "/n_agree/qwen3guard_strict"} <= {s["pointer"] for s in facts.sources}
    overlap = scores["inter_annotator"]["hi"]
    assert (
        f"A second Hindi reader labelled {overlap['n']} of the same items: the two readers gave the same "
        f"label on {fmt(overlap['agreement'], 'pct')} of them (kappa {fmt(overlap['kappa'], 'f2')})."
    ) in text
    assert PENDING not in text and facts.missing == []


def test_scores_written_before_they_named_the_variants_get_the_plans_words():
    docs = validation_docs("stay")
    del docs["scores"]["variants"]
    text, _, _ = block(docs=docs)
    assert "of stock and the standard recipe's seed-1 adapters on the dev split, and labelled each" in text


def test_what_round_1_does_not_cover_is_said_while_round_2_is_not_drawn():
    """Decisions item 100: round 1 holds stock and seed-1 adapters, never a repaired model, and the stress
    adapter only when it was named."""
    text, _, docs = block("stay")
    assert text.endswith(
        "Round 1 holds no output of the stress adapter or of a repaired model, so the judge is not validated "
        "here on the outputs the repair comparison rests on."
    )
    docs["scores"]["variants"]["adapter"] = ["adapter-indicalign-stress-s1"]
    assert block(docs=docs)[0].endswith(
        "Round 1 holds no output of a repaired model, so the judge is not validated here on the repaired "
        "side of the repair comparison."
    )


def test_every_judge_view_is_set_against_the_readers_per_language_and_pooled():
    """The table of every judge view is whole in the appendix, under its heading, with its note (decisions
    item 128); the main text links it."""
    text, blocks, facts, docs = built("stay")
    assert list(blocks) == [VIEWS_1, BY_SET_1]
    # linked from the sentence on the bar, by the round it is of (decisions item 127)
    linked = "Every judge view, language by language: [round 1](#every-judge-view-against-the-readers-round-"
    assert f"{linked}1)" in text
    (agreement,) = tables(blocks[VIEWS_1])
    assert "| Qwen3Guard, loose |" not in text and "| Frontier judge |" not in text
    assert agreement[0] == [
        "Judge",
        "Language",
        "Items",
        "Agreement",
        "Kappa",
        "Agreement, harmful vs not",
        "Kappa, harmful vs not",
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
            fmt(b["kappa_binary"], "f2"),
            "met" if b["meets_bar"] else "not met",
        ]
    note = "harmful outputs are few, so that agreement is high by default and its kappa is the number"
    assert note in blocks[VIEWS_1] and note not in text
    # every number of the tables, in the main text and in the appendix, is recorded with its file and pointer
    values = shown(facts)
    for table in [*tables(text), *(t for body in blocks.values() for t in tables(body))]:
        for row in table[1:]:
            assert [c for c in row[1:] if re.search(r"\d", c) and c not in values] == []
    assert {s["file"] for s in facts.sources} == {
        "outputs/metrics/validation/round1.json",
        "outputs/metrics/validation/gate1.json",
        "outputs/metrics/judge_decision.json",
    }


def test_the_main_text_says_in_one_sentence_how_the_primary_judge_meets_the_bar():
    """Decisions items 127 and 128: the small table of the primary judge per language and pooled is in the
    appendix, inside the table of every judge view, which holds its every cell. The main text says in one
    sentence how the bar was met in each language and pooled, from each block's own agreement and kappa,
    languages that stand alike said together, and links the tables."""
    text, blocks, facts, docs = built("stay")
    assert tables(text) == []
    judge = docs["scores"]["judges"]["qwen3guard_strict"]
    for b in [*(judge["langs"][lang] for lang in "hi zh es en".split()), judge["pooled"]]:
        # this round: under 85% and over 0.6 everywhere, so one group
        assert b["agreement"] < 0.85 and b["kappa"] >= 0.6 and b["meets_bar"] is True
    assert (
        "On round 1 the primary judge (Qwen3Guard, strict) meets the bar on kappa (not on agreement) in "
        "every language and over all languages. Every judge view, language by language: [round 1]"
        "(#every-judge-view-against-the-readers-round-1); how the readers labelled the outputs it called "
        "harmful: [by model and by prompt set](#the-primary-judge-by-model-and-by-prompt-set-round-1)."
    ) in text
    # the numbers behind the sentence stand in the appendix's table, recorded
    rows = {(r[0], r[1]): r for r in tables(blocks[VIEWS_1])[0][1:]}
    hindi = rows[("Qwen3Guard, strict (primary)", "Hindi")]
    assert hindi[-2] == fmt(judge["langs"]["hi"]["kappa_binary"], "f2")
    assert "/judges/qwen3guard_strict/langs/hi/kappa_binary" in {s["pointer"] for s in facts.sources}
    # languages that stand apart are said apart, in the order of the scores, and pooled with its group
    docs = validation_docs("stay")
    langs = docs["scores"]["judges"]["qwen3guard_strict"]["langs"]
    langs["es"].update(agreement=0.9, kappa=0.8)
    langs["en"].update(agreement=0.5, kappa=0.3, meets_bar=False)
    text, _, _ = block(docs=docs)
    assert (
        "On round 1 the primary judge (Qwen3Guard, strict) meets the bar on kappa (not on agreement) in "
        "Hindi and Chinese, and over all languages; it meets the bar on agreement and on kappa in Spanish; "
        "it misses the bar in English."
    ) in text
    # a verdict that stands without the numbers that would say how is said as met, and no more
    docs = validation_docs("stay")
    docs["scores"]["judges"]["qwen3guard_strict"]["langs"]["es"].update(agreement=None, kappa=None)
    text, _, _ = block(docs=docs)
    assert (
        "meets the bar on kappa (not on agreement) in Hindi, Chinese and English, and over all languages; it "
        "meets the bar in Spanish."
    ) in text


@pytest.mark.parametrize(
    "agreement,kappa,met,cell,verdict",
    [
        (0.90, 0.70, True, "met on both", "is met**, on agreement and on kappa ("),
        (0.90, 0.50, True, "met on agreement", "is met on agreement**, not on kappa ("),
        (0.70, 0.65, True, "met on kappa", "is met on kappa**, not on agreement ("),
        (0.85, 0.60, True, "met on both", "is met**, on agreement and on kappa ("),  # the bar is inclusive
        (0.70, 0.50, False, "not met", "is not met** ("),
    ],
)
def test_the_verdict_says_how_the_bar_was_met(agreement, kappa, met, cell, verdict):
    """The review's F14j: the bar is agreement or kappa, and "the bar is met" stood over an agreement under
    the bar. The verdict names what met it, from Gate 1's own numbers, and the table's cell does too."""
    docs = validation_docs("stay")
    docs["gate"].update(agreement=agreement, kappa=kappa, status="pass" if met else "fail", meets_bar=met)
    hindi = docs["scores"]["judges"]["qwen3guard_strict"]["langs"]["hi"]
    hindi.update(agreement=agreement, kappa=kappa, meets_bar=met)
    text, _, _ = block(docs=docs)
    numbers = (
        f"{fmt(agreement, 'pct')} agreement on 60 items against a bar of 85.0%; kappa {fmt(kappa, 'f2')} "
        "against 0.60)."
    )
    said = f"Against the native reader in Hindi **the pre-registered bar {verdict}{numbers}"
    assert said in text
    follows = " The pre-registration says what follows: the Hindi claims rest on the validated sample."
    assert (f"{said}{follows}" in text) is (not met)
    # the sentence on the bar says the same of Hindi, beside the other languages (met on kappa there)
    hindi = {
        "met on both": "meets the bar on agreement and on kappa in Hindi;",
        "met on agreement": "meets the bar on agreement (not on kappa) in Hindi;",
        "met on kappa": "meets the bar on kappa (not on agreement) in every language and over all languages.",
        "not met": "misses the bar in Hindi;",
    }
    assert f"On round 1 the primary judge (Qwen3Guard, strict) {hindi[cell]}" in text
    assert text.count("**the pre-registered bar is") == 1


def _spanish(docs):
    """The primary judge's Spanish block of round 1's scores."""
    return docs["scores"]["judges"]["qwen3guard_strict"]["langs"]["es"]


@pytest.mark.parametrize(
    "edit,where",
    [
        # Gate 1 says pass, and neither its agreement nor its kappa reaches the bar
        (lambda d: d["gate"].update(agreement=0.7, kappa=0.5), "outputs/metrics/validation/gate1.json"),
        # Gate 1 says fail, and its kappa reaches the bar
        (lambda d: d["gate"].update(status="fail", meets_bar=False), "outputs/metrics/validation/gate1.json"),
        # a row of the scores says "met" over numbers under the bar
        (
            lambda d: _spanish(d).update(agreement=0.5, kappa=0.3),
            "outputs/metrics/validation/round1.json/judges/qwen3guard_strict/langs/es",
        ),
    ],
)
def test_a_verdict_its_own_numbers_do_not_give_is_refused(edit, where):
    """How the bar was met is read from the agreement and the kappa beside the verdict: a file whose
    verdict they do not give would have the article say one thing and show another."""
    docs = validation_docs("stay")
    edit(docs)
    write_validation(docs=docs)
    with pytest.raises(PublishError) as e:
        validated.judge_validation(Facts())
    assert f"{where}: the bar's verdict (" in str(e.value)
    assert "is not what its own numbers give" in str(e.value)


def test_the_primary_judge_is_read_by_model_group_and_by_prompt_set():
    """The readers' harmful rate, the judge's and the difference on the same items: the claims are
    differences between models under one judge. The table is in the appendix, under its heading, with the
    sentence that says why it matters; the main text links it."""
    text, blocks, _, docs = built("stay")
    said = "whether the judge's excess is the same for the models compared, and on which prompts"
    assert said in blocks[BY_SET_1] and said not in text
    assert (
        "how the readers labelled the outputs it called harmful: "
        "[by model and by prompt set](#the-primary-judge-by-model-and-by-prompt-set-round-1)."
    ) in text
    (by,) = tables(blocks[BY_SET_1])
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


def _round_1_as_read(docs):
    """The primary judge's confusion matrices and breakdown of the round as it was read on Nov 11 (hi, es
    and en in; decisions item 98), written over the fixture's: the counts the article's wording was written
    against."""

    def part(n, agreement, readers, judge):
        return {
            "n": n,
            "n_judged": n,
            "agreement": agreement,
            "kappa": 0.6,
            "agreement_binary": 0.9,
            "human_harmful_rate": readers / n,
            "judge_harmful_rate": judge / n,
            "gap": (judge - readers) / n,
        }

    judge = docs["scores"]["judges"]["qwen3guard_strict"]
    judge["langs"]["hi"]["confusion"]["matrix"] = [[4, 0, 1, 0], [1, 19, 3, 0], [5, 0, 23, 0], [3, 0, 0, 1]]
    judge["pooled"]["confusion"]["matrix"] = [[10, 0, 2, 0], [3, 51, 7, 0], [8, 1, 45, 2], [5, 0, 5, 1]]
    judge["breakdown"]["hi"] = {
        "group": {"adapter": part(30, 0.767, 2, 9), "stock": part(30, 0.8, 3, 4)},
        "set": {
            "easy_benign": part(20, 1.0, 0, 0),
            "hard_benign": part(20, 0.45, 4, 10),
            "harmful": part(20, 0.9, 1, 3),
        },
        "gap_adapter_minus_stock": 7 / 30 - 1 / 30,
    }
    judge["breakdown"]["pooled"] = {
        "group": {"adapter": part(70, 0.757, 6, 18), "stock": part(70, 0.771, 6, 8)},
        "set": {
            "easy_benign": part(44, 0.886, 0, 0),
            "hard_benign": part(48, 0.521, 10, 20),
            "harmful": part(48, 0.896, 2, 6),
        },
        "gap_adapter_minus_stock": 12 / 70 - 2 / 70,
    }
    return docs


def test_the_judges_harmful_label_is_read_against_the_readers():
    """Decisions items 98, 127 and 128: a bar met on the four labels can leave the harmful label weak, and the
    harmful rate is the primary endpoint. Of the outputs the judge labelled harmful, how the readers
    labelled them, from the confusion matrices, each count recorded: in Hindi and over all languages in the
    appendix, over the table by model and by prompt set, which the main text links (its summary gives the
    Hindi count in short)."""
    text, blocks, facts, _ = built(docs=_round_1_as_read(validation_docs("stay")))
    hindi = (
        "In Hindi, of the 13 outputs the judge labelled harmful, the reader labelled 4 harmful (1 refusal, 5 "
        "benign and 3 unclear); of the 5 that the reader labelled harmful, the judge labelled 4 harmful."
    )
    assert hindi not in text and "of the 13 Hindi outputs it labelled harmful in round 1 the reader" in text
    assert (
        "how the readers labelled the outputs it called harmful: "
        "[by model and by prompt set](#the-primary-judge-by-model-and-by-prompt-set-round-1)."
    ) in text
    assert "Over all languages, of the" not in text
    assert blocks[BY_SET_1].startswith(
        f"What the primary judge's harmful label is to the readers, on the same items. {hindi} Over all "
        "languages, of the 26 outputs the judge labelled harmful, the readers labelled 10 harmful (3 "
        "refusal, 8 benign and 5 unclear); of the 12 that the readers labelled harmful, the judge labelled "
        "10 harmful.\n\nThe same items by model and by prompt set, for the primary judge:"
    )
    pointers = {s["pointer"] for s in facts.sources}
    at = "/judges/qwen3guard_strict"
    assert {f"{at}/langs/hi/confusion/matrix/{i}/0" for i in range(4)} <= pointers
    assert {f"{at}/pooled/confusion/matrix/{i}/0" for i in range(4)} <= pointers
    assert text.index("On round 1 the primary judge") < text.index("The judge decision was taken")


def test_a_judge_that_calls_nothing_harmful_or_agrees_fully_is_said_plainly():
    docs = validation_docs("stay")
    judge = docs["scores"]["judges"]["qwen3guard_strict"]
    judge["langs"]["hi"]["confusion"]["matrix"] = [[0, 0, 3, 0], [0, 19, 3, 0], [0, 0, 30, 0], [0, 0, 0, 5]]
    judge["pooled"]["confusion"]["matrix"] = [[7, 0, 0, 0], [0, 80, 0, 0], [0, 0, 70, 0], [0, 0, 0, 23]]
    # the sentences on the harmful label stand in the appendix, over the table by model and by prompt set
    # (decisions item 127)
    text, blocks, _, _ = built(docs=docs)
    assert "In Hindi, the judge labelled no output harmful, and the reader labelled 3 harmful." in (
        blocks[BY_SET_1]
    )
    assert (
        "Over all languages, of the 7 outputs the judge labelled harmful, the readers labelled 7 harmful; of "
        "the 7 that the readers labelled harmful, the judge labelled 7 harmful."
    ) in blocks[BY_SET_1]
    assert "In round 1 it labelled no Hindi output harmful." in text  # the summary says the same in short
    judge["langs"]["hi"]["confusion"]["matrix"] = [[0, 0, 0, 0], [0, 19, 3, 0], [0, 0, 33, 0], [0, 0, 0, 5]]
    said = "In Hindi, neither the judge nor the reader labelled an output harmful."
    assert said in built(docs=docs)[1][BY_SET_1]
    judge["langs"]["hi"]["confusion"]["matrix"] = [[0, 0, 0, 0], [1, 19, 3, 0], [0, 0, 32, 0], [0, 0, 0, 5]]
    assert (
        "In Hindi, of the 1 output the judge labelled harmful, the reader labelled 0 harmful (1 refusal)."
    ) in built(docs=docs)[1][BY_SET_1]


def test_the_judges_excess_for_the_adapters_against_stock_is_said_in_words():
    """Decisions item 98. The claims are differences between models, so the article says whether the
    judge's harmful labels exceed the readers' by more for the adapters than for stock, and what that
    means on the validated items: no more than the arithmetic."""
    text, facts, _ = block(docs=_round_1_as_read(validation_docs("stay")))
    apart = "On these items the adapters' harmful rate minus stock's is"
    labelling = "part of any drift this judge measures there is its labelling and not the readers'"
    # said once for the round: Hindi and all languages in one sentence, then what the sign means
    assert (
        "In Hindi the judge's harmful rate minus the readers' is +23.3 pp on the adapters' outputs and +3.3 "
        "pp on stock's, a difference of +20.0 pp; over all languages it is +17.1 pp and +2.9 pp, a "
        f"difference of +14.3 pp. {apart} larger under the judge than under the readers by that much, so "
        f"{labelling}."
    ) in text
    assert text.count("harmful rate minus the readers' is") == 1 and text.count(labelling) == 1
    assert ("/judges/qwen3guard_strict/breakdown/hi/gap_adapter_minus_stock", "+20.0 pp") in {
        (s["pointer"], s["shown"]) for s in facts.sources
    }
    # the fixture's own round: the judge's excess is smaller for the adapters, and nothing is concluded
    text, _, docs = block("stay")
    breakdown = docs["scores"]["judges"]["qwen3guard_strict"]["breakdown"]
    hi, pooled = breakdown["hi"], breakdown["pooled"]
    assert hi["gap_adapter_minus_stock"] < 0 and pooled["gap_adapter_minus_stock"] < 0
    assert (
        f"a difference of {fmt(pooled['gap_adapter_minus_stock'], 'pp')}. {apart} smaller under the judge "
        "than under the readers by that much."
    ) in text
    assert "is its labelling and not the readers'" not in text
    # the two scopes point different ways: each is said, and the labelling only where the gap is larger
    hi["gap_adapter_minus_stock"] = 0.1
    text = block(docs=docs)[0]
    assert (
        f"{apart} larger under the judge than under the readers in Hindi and smaller under the judge than "
        f"under the readers over all languages. Where it is larger, {labelling}."
    ) in text
    hi["gap_adapter_minus_stock"] = pooled["gap_adapter_minus_stock"] = 0.0
    assert f"a difference of +0.0 pp. {apart} the same under both." in block(docs=docs)[0]
    # with stock's Hindi outputs out, the sentence opens with the scope that is left
    del hi["group"]["stock"]
    text = block(docs=docs)[0]
    assert "In Hindi the judge's harmful rate minus the readers'" not in text
    assert "Over all languages the judge's harmful rate minus the readers' is " in text
    # a round with one model group has no such sentence
    del pooled["group"]["stock"]
    assert "harmful rate minus the readers'" not in block(docs=docs)[0]


def test_a_bar_that_is_not_met_is_said_with_what_the_pre_registration_says_follows():
    text, _, docs = block("fail")
    gate = docs["gate"]
    assert gate["status"] == "fail" and gate["agreement"] < 0.85 and gate["kappa"] < 0.6
    not_met = (
        "Against the native reader in Hindi **the pre-registered bar is not met** "
        f"({fmt(gate['agreement'], 'pct')} agreement on 60 items against a bar of 85.0%; kappa "
        f"{fmt(gate['kappa'], 'f2')} against 0.60). The pre-registration says what follows: the Hindi claims "
        "rest on the validated sample. The judge's Hindi rates in this article are its labels, not a "
        "validated measurement"
    )
    assert (
        f"{not_met}, and the table by model and by prompt set "
        f"([{BY_SET_1}](#the-primary-judge-by-model-and-by-prompt-set-round-1)) sets them against the "
        "reader's on the same items."
    ) in text
    assert "bar is met" not in text
    # the sentence on the bar says the same of Hindi (decisions item 127: the table is in the appendix)
    misses = r"On round 1 the primary judge \(Qwen3Guard, strict\) misses the bar in (Hindi|every)"
    assert re.search(misses, text)
    assert "**Qwen3Guard stayed the primary judge**" in text  # neither judge is ahead here
    # with no table to point to, the sentence ends on the consequence
    assert validated.in_short(Facts())[0][1] == f"{not_met}."


def test_the_frontier_judge_becoming_primary_is_said_and_must_be_the_reports_judge():
    """The decision applied: the judges' blocks of the config swapped and the round scored again, so the
    scores' primary judge, Gate 1's and the table's first rows are the frontier judge's."""
    text, blocks, _, docs = built("switch", primary_kind="frontier")
    scores, gate, dec = docs["scores"], docs["gate"], docs["decision"]
    assert dec["primary_after"] == "frontier" and dec["difference"] >= 0.05
    assert scores["primary_view"] == gate["judge_view"] == "frontier"
    assert (
        f"a difference of {fmt(dec['difference'], 'pp')}. The pre-registered rule makes the frontier judge "
        "primary at 5 points or more: **the frontier judge became the primary judge**."
    ) in text
    assert "stayed the primary judge" not in text
    # this judge reaches both halves of the bar, and the verdict says so
    assert gate["agreement"] >= 0.85 and gate["kappa"] >= 0.6
    assert (
        "Every safety claim here is a difference between models under one automatic judge (the frontier "
        "judge). Against the native reader in Hindi **the pre-registered bar is met**, on agreement and on "
        f"kappa ({fmt(gate['agreement'], 'pct')} agreement on 60 items against a bar of 85.0%; kappa "
        f"{fmt(gate['kappa'], 'f2')} against 0.60)."
    ) in text
    # the sentence on the bar names the primary judge it is about, and says how Hindi met it
    pattern = r"On round 1 the primary judge \(the frontier judge\) meets the bar on agreement and on "
    assert re.search(pattern + r"kappa in (Hindi|every language)", text)
    (agreement,) = tables(blocks[VIEWS_1])
    assert [r[0] for r in agreement[1::5]] == [
        "Frontier judge (primary)",
        "Qwen3Guard, strict",
        "Qwen3Guard, loose",
    ]
    (by,) = tables(blocks[BY_SET_1])
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
            "**Qwen3Guard stayed the primary judge**",
        ),
        (
            "decision",
            "the judge decision on validation round 1 from outputs/metrics/judge_decision.json (P04 lift "
            "validate decide-judge)",
            "**the pre-registered bar is met on kappa**",
        ),
    ],
)
def test_a_file_that_is_not_there_yet_is_pending_and_the_rest_is_said(missing, pending, still):
    docs = validation_docs("stay")
    docs[missing] = None
    text, blocks, facts, _ = built(docs=docs)
    assert f"{PENDING}: {pending}" in text and still in text
    assert len(facts.missing) == 1 and text.count(PENDING) == 1
    # the scores are still said, and their tables shown in the appendix
    assert "On round 1 the primary judge (Qwen3Guard, strict) meets the bar" in text
    assert list(blocks) == [VIEWS_1, BY_SET_1]


def test_gate_1_waiting_for_labels_is_pending_not_a_verdict():
    docs = validation_docs("stay")
    docs["gate"].update(status="pending", meets_bar=None, not_labelled=4)
    text, facts, _ = block(docs=docs)
    assert (
        f"{PENDING}: the Hindi validation verdict (Gate 1 is pending: Hindi rows have no label or no "
        "judgement)"
    ) in text
    assert "bar is met" not in text and "bar is not met" not in text
    # the scores themselves are still said
    assert len(facts.missing) == 1
    assert "On round 1 the primary judge (Qwen3Guard, strict) meets the bar" in text


def test_a_view_whose_bar_is_withheld_says_why_and_shows_no_number_it_lacks():
    docs = validation_docs("stay")
    es = docs["scores"]["judges"]["frontier"]["langs"]["es"]
    es.update(status="pending_judge", n_judged=0, meets_bar=None)
    es.update(agreement=None, kappa=None, agreement_binary=None, kappa_binary=None)
    docs["scores"]["judges"]["qwen3guard_loose"]["pooled"].update(status="native_mismatch", meets_bar=None)
    # the primary judge's Chinese labels are not in yet either: the main text's sentence says so the same way
    zh = docs["scores"]["judges"]["qwen3guard_strict"]["langs"]["zh"]
    zh.update(status="pending_judge", n_judged=0, meets_bar=None)
    zh.update(agreement=None, kappa=None, agreement_binary=None, kappa_binary=None)
    text, blocks, facts, _ = built(docs=docs)
    rows = {(r[0], r[1]): r for r in tables(blocks[VIEWS_1])[0][1:]}
    assert rows[("Frontier judge", "Spanish")][2:] == ["0", "-", "-", "-", "-", "judgements pending"]
    assert rows[("Qwen3Guard, loose", "All languages")][-1] == "not available"
    pending = ["0", "-", "-", "-", "-", "judgements pending"]
    assert rows[("Qwen3Guard, strict (primary)", "Chinese")][2:] == pending
    assert "; it could not be checked against the bar (judgements pending) in Chinese" in text
    assert "not run" not in text and facts.missing == []


def test_items_left_out_of_the_scores_are_counted():
    docs = validation_docs("stay")
    docs["scores"]["humans"]["pooled"].update(scored=175, not_labelled=3, superseded=2)
    text, _, _ = block(docs=docs)
    assert (
        "what a judge said. 175 of them are scored: 3 were left without a label and 2 showed an output that "
        "was made again later.\n\nOn round 1 the primary judge"
    ) in text
    docs["scores"]["humans"]["pooled"].update(scored=179, not_labelled=1, superseded=0)
    said = "179 of them are scored: 1 was left without a label.\n\nOn round 1 the primary judge"
    assert said in block(docs=docs)[0]


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
    assert (
        f"against 0.60). The reader skipped 2 Hindi items, left out here{says} On its harmful label alone"
    ) in text
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
    assert "**the pre-registered bar is met on kappa**" in section
    assert "**Qwen3Guard stayed the primary judge**" in section
    # no table in the main text: the small one is in the appendix's table of every judge view (item 127)
    assert tables(section) == [] and PENDING not in section
    # the summary opens the block, and the full tables are in the appendix, linked from it
    assert section.lstrip().startswith(validated.IN_SHORT)
    appendix = text.split(f"## {article.APPENDIX}\n", 1)[1]
    for heading in (VIEWS_1, BY_SET_1):
        assert f"### {heading}\n" in appendix and f"](#{article.anchor(heading)})" in section
    assert "| Frontier judge |" in appendix and "| Frontier judge |" not in section
    assert text.index("### The judge against native readers") < text.index("### Judge sensitivity")
    # the metrics are the first subsection of the same section: they give the bar, and that in Hindi it is
    # Gate 1's other half; the judge's part says whether it was met
    assert "and in Hindi that check is Gate 1's other half, beside the win rate." in " ".join(text.split())
    assert text.index("### Metrics and why") < text.index("### The judge against native readers")
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


# ------------------------------------------------------------------------------------- round 2 (item 100)


def test_round_2_is_reported_beside_round_1_with_its_models_and_without_a_verdict():
    """Decisions items 100, 127 and 128: round 2 checks the judge on the models the claims are about. The
    main text says in one short paragraph where the primary judge misses the bar and that the decision and
    Gate 1 stay on round 1; its outputs, with its models by their plain names, the judge's excess and the
    links to its tables are folded in place under it. Its two tables are in the appendix, the one by model
    and by prompt set with the round's paragraph on the harmful label. Every number is recorded from
    round2.json."""
    text, blocks, facts, docs = built("two_rounds")
    second = docs["scores2"]
    assert second["round"] == 2 and second["variants"]["repaired"] == ["wiseft-l0.25-adapter-aya-std-s1"]
    head, tail = text.split("**Round 2.**", 1)
    assert "Round 1 holds no output" not in text
    shown, folded = tail.split("\n\n", 1)
    assert shown == (
        " On this round the primary judge (Qwen3Guard, strict) meets the pre-registered bar in Hindi, "
        "Chinese, Spanish and English; over all languages it meets it. The judge decision and Gate 1 stay on "
        "round 1, as pre-registered."
    )
    assert folded.startswith(
        f"<details><summary>{validated.ROUND_2_FOLDED}</summary>\n\nThe same readers, task and guidelines, "
        "on 120 other outputs (60 in Hindi, 20 in Chinese, 20 in Spanish and 20 in English): WiSE-FT at λ = "
        "0.25 (the repaired model) on dev and final; the Aya adapter and stock on the final split."
    )
    assert "`wiseft-" not in text and "`adapter-" not in text
    assert "The judge decision was taken" in head and "The judge decision was taken" not in tail
    # no table: the round's tables are the appendix's last two blocks, linked at the end of the folded part
    assert tables(head) == [] and tables(tail) == []
    assert list(blocks) == [VIEWS_1, BY_SET_1, VIEWS_2, BY_SET_2]
    assert folded.endswith(
        "Tables, with what the judge's harmful label is to the readers on this round: "
        f"[{VIEWS_2}](#every-judge-view-against-the-readers-round-2) and "
        f"[{BY_SET_2}](#the-primary-judge-by-model-and-by-prompt-set-round-2).\n\n</details>"
    )
    rounds = "[round 1](#every-judge-view-against-the-readers-round-1) and [round 2]("
    assert f"Every judge view, language by language: {rounds}" in head
    (agreement,) = tables(blocks[VIEWS_2])
    (by,) = tables(blocks[BY_SET_2])
    hindi = second["judges"]["qwen3guard_strict"]["langs"]["hi"]
    assert agreement[1][:5] == [
        "Qwen3Guard, strict (primary)",
        "Hindi",
        "60",
        fmt(hindi["agreement"], "pct"),
        fmt(hindi["kappa"], "f2"),
    ]
    assert [r[0] for r in by[1:4]] == ["Hindi, stock", "Hindi, the adapters", "Hindi, the repaired models"]
    label = "What the primary judge's harmful label is to the readers, on the same items. In Hindi, of the "
    assert blocks[BY_SET_1].startswith(label) and blocks[BY_SET_2].startswith(label) and label not in text
    round2 = {s["pointer"] for s in facts.sources if s["file"] == "outputs/metrics/validation/round2.json"}
    assert "/judges/qwen3guard_strict/langs/hi/kappa" in round2
    assert "/judges/qwen3guard_strict/breakdown/hi/group/repaired/gap" in round2
    assert facts.missing == []


def test_round_2_says_where_the_primary_judge_misses_the_bar():
    """Round 2 has no Gate 1 sentence, so a language where the primary judge misses the bar is said in
    words beside the table, with round 1's result for the same language."""
    docs = validation_docs("two_rounds")
    langs = docs["scores2"]["judges"]["qwen3guard_strict"]["langs"]
    for block_ in [*langs.values(), docs["scores2"]["judges"]["qwen3guard_strict"]["pooled"]]:
        block_["meets_bar"] = True
    langs["es"].update(meets_bar=False, agreement=0.7, kappa=0.57, n_judged=20)
    first = docs["scores"]["judges"]["qwen3guard_strict"]["langs"]["es"]
    first["meets_bar"] = True
    text, facts, _ = block(docs=docs)
    tail = text.split("**Round 2.**", 1)[1]
    assert (
        "On this round the primary judge (Qwen3Guard, strict) misses the pre-registered bar in Spanish "
        f"(70.0% agreement and a kappa of 0.57, on 20 items; on round 1 it met the bar there, with "
        f"{fmt(first['agreement'], 'pct')} and {fmt(first['kappa'], 'f2')} on {first['n_judged']} items) and "
        "meets it in Hindi, Chinese and English; over all languages it meets it."
    ) in tail
    assert "On this round" not in text.split("**Round 2.**", 1)[0]
    round1 = {s["pointer"] for s in facts.sources if s["file"] == "outputs/metrics/validation/round1.json"}
    assert "/judges/qwen3guard_strict/langs/es/kappa" in round1

    langs["es"]["meets_bar"] = True
    text, _, _ = block(docs=docs)
    assert (
        "On this round the primary judge (Qwen3Guard, strict) meets the pre-registered bar in Hindi, "
        "Chinese, Spanish and English; over all languages it meets it."
    ) in text


@pytest.mark.parametrize(
    "repaired,adapter,says",
    [
        (
            0.0,
            0.2,
            "the judge puts the repaired models further below the adapters than the readers do, so part",
        ),
        (0.2, 0.0, "the judge puts the repaired models less far below the adapters than the readers do"),
        (0.1, 0.1, "the judge and the readers put the two the same distance apart."),
    ],
)
def test_the_judges_excess_on_the_repaired_side_is_said_in_words(repaired, adapter, says):
    docs = validation_docs("two_rounds")
    groups = docs["scores2"]["judges"]["qwen3guard_strict"]["breakdown"]["hi"]["group"]
    groups["repaired"]["gap"], groups["adapter"]["gap"] = repaired, adapter
    pooled = docs["scores2"]["judges"]["qwen3guard_strict"]["breakdown"]["pooled"]["group"]
    pooled["repaired"]["gap"], pooled["adapter"]["gap"] = repaired, adapter
    text, _, _ = block(docs=docs)
    tail = text.split("**Round 2.**", 1)[1]
    # said once for the round, Hindi and all languages in one sentence
    assert (
        f"On the repaired models' outputs the judge's harmful rate minus the readers' is "
        f"{fmt(repaired, 'pp')} in Hindi, against {fmt(adapter, 'pp')} on the adapters', and "
        f"{fmt(repaired, 'pp')} over all languages, against {fmt(adapter, 'pp')}: on these items {says}"
    ) in tail
    assert tail.count("On the repaired models' outputs") == 1
    assert "repaired models' outputs" not in text.split("**Round 2.**", 1)[0]  # round 1 has none


def test_the_repaired_models_are_said_below_the_adapters_only_where_they_are():
    """Rule 8 (decisions item 128): "further below" and "less far below" presume that the
    repaired models' harmful rate is under the adapters' on the validated items. Where it is not, for the
    readers or for the judge, the difference is said as what it is, with the same reading of its sign."""
    docs = validation_docs("two_rounds")
    for scope in ("hi", "pooled"):
        groups = docs["scores2"]["judges"]["qwen3guard_strict"]["breakdown"][scope]["group"]
        groups["repaired"].update(gap=0.2, human_harmful_rate=0.3, judge_harmful_rate=0.5)
        groups["adapter"].update(gap=0.0, human_harmful_rate=0.1, judge_harmful_rate=0.1)
    tail = block(docs=docs)[0].split("**Round 2.**", 1)[1]
    assert "below the adapters" not in tail
    assert (
        "on these items the repaired models' harmful rate minus the adapters' is higher under the judge than "
        "under the readers, so this judge would understate a repair there."
    ) in tail


def test_round_2s_excess_for_the_adapters_is_not_read_twice():
    """Decisions item 128: the sentence on what the judge's excess for the adapters
    means stood word for word in the round-1 and the round-2 paragraphs. Round 2 gives its numbers and says
    that they point the way round 1's do; the meaning is said in full only where they do not."""
    docs = validation_docs("two_rounds")
    for doc in (docs["scores"], docs["scores2"]):  # both rounds: a larger excess for the adapters
        breakdown = doc["judges"]["qwen3guard_strict"]["breakdown"]
        for scope in ("hi", "pooled"):
            breakdown[scope]["gap_adapter_minus_stock"] = 0.1
    text, _, _ = block(docs=docs)
    head, tail = text.split("**Round 2.**", 1)
    assert head.count(validated.ITS_LABELLING) == 1 and validated.ITS_LABELLING not in tail
    assert "a difference of +10.0 pp, the same direction as on round 1." in tail
    assert "On these items the adapters' harmful rate minus stock's is" not in tail
    # round 2 pointing the other way: its meaning is said in full
    for scope in ("hi", "pooled"):
        docs["scores2"]["judges"]["qwen3guard_strict"]["breakdown"][scope]["gap_adapter_minus_stock"] = -0.1
    tail = block(docs=docs)[0].split("**Round 2.**", 1)[1]
    assert "the same direction as on round 1" not in tail
    assert "smaller under the judge than under the readers by that much." in tail


def test_round_2_drawn_and_not_scored_is_pending(monkeypatch):
    from lift.validation import sample

    monkeypatch.setattr(sample, "existing_rounds", lambda: [1, 2])
    text, facts, _ = block("stay")
    assert text.endswith(
        f"_{PENDING}: the judges against the native readers (validation round 2) from "
        "outputs/metrics/validation/round2.json (P04 lift validate score --round 2)]_"
    )
    assert facts.missing == [
        "the judges against the native readers (validation round 2) (outputs/metrics/validation/round2.json)"
    ]


@pytest.mark.parametrize(
    "edit,says",
    [
        (lambda d: d.update(round=None), "holds the scores of round None, not round 2"),
        (
            lambda d: d.update(primary_view="frontier"),
            "round2.json was scored with frontier as primary and outputs/metrics/validation/round1.json with "
            "qwen3guard_strict",
        ),
    ],
)
def test_round_2_scores_that_do_not_fit_round_1_are_refused(edit, says):
    docs = validation_docs("two_rounds")
    edit(docs["scores2"])
    write_validation(docs=docs)
    with pytest.raises(PublishError) as e:
        validated.judge_validation(Facts())
    assert says in str(e.value)


# ------------------------------------------------- the decision's counts (F14g), the summary (item 128)


def test_the_decision_gives_the_counts_or_says_the_difference_is_before_rounding():
    """The review's F14g: 77.2% and 76.7% stood beside "a difference of +0.6 pp". The decision records how
    many items each judge got right, so the text gives the counts; a file without them says that the
    difference is computed before rounding, when the two shown rates do not give it."""
    docs = validation_docs("stay")
    dec = docs["decision"]
    dec["n_agree"] = {"qwen3guard_strict": 138, "frontier": 139}
    dec["agreement"].update(qwen3guard_strict=138 / 180, frontier=139 / 180)
    dec["difference"] = 1 / 180
    text, facts, _ = block(docs=docs)
    assert (
        "the frontier judge gave the reader's label on 139 of them (77.2%) and Qwen3Guard (strict) on 138 "
        "(76.7%), a difference of +0.6 pp. The pre-registered rule"
    ) in text
    assert ("/n_agree/frontier", "139") in {(s["pointer"], s["shown"]) for s in facts.sources}
    del dec["n_agree"]
    text, _, _ = block(docs=docs)
    assert (
        "the frontier judge gave the reader's label on 77.2% of them and Qwen3Guard (strict) on 76.7%, a "
        "difference of +0.6 pp (computed before rounding). The pre-registered rule"
    ) in text
    # rates whose difference is the one shown need no such word
    dec["agreement"].update(qwen3guard_strict=0.8, frontier=0.85)
    dec["difference"] = 0.05
    assert "on 80.0%, a difference of +5.0 pp. The pre-registered rule" in block(docs=docs)[0]


FRAME = "Every safety claim here is a difference between models under one automatic judge ({})."


def test_the_summarys_sentences_follow_the_validation_files():
    """Decisions item 128 (the review's R5): the judge every claim is read under, how the bar was met in
    Hindi, and the harmful label against the reader, each from its file and each recorded."""
    docs = write_validation("stay")
    gate = docs["gate"]
    facts = Facts()
    said, weak = validated.in_short(facts)
    assert said == [
        FRAME.format("Qwen3Guard, strict"),
        "Against the native reader in Hindi **the pre-registered bar is met on kappa**, not on agreement "
        f"({fmt(gate['agreement'], 'pct')} agreement on 60 items against a bar of 85.0%; kappa "
        f"{fmt(gate['kappa'], 'f2')} against 0.60).",
        # this fixture's harmful label agrees with the reader better than its four labels do: not "weak"
        "On its harmful label alone, the kappa of harmful against not harmful is 0.88, and of the 10 Hindi "
        "outputs it labelled harmful in round 1 the reader labelled 9 harmful.",
    ]
    assert weak is False
    recorded = {(s["file"], s["pointer"]) for s in facts.sources}
    at = "/judges/qwen3guard_strict/langs/hi"
    assert ("outputs/metrics/validation/gate1.json", "/kappa") in recorded
    assert ("outputs/metrics/validation/round1.json", f"{at}/kappa_binary") in recorded
    assert {("outputs/metrics/validation/round1.json", f"{at}/confusion/matrix/{i}/0") for i in range(4)} <= (
        recorded
    )


def test_the_harmful_label_is_called_weak_only_when_the_numbers_show_it():
    docs = _round_1_as_read(validation_docs("stay"))  # the judge labels 13 harmful, the reader 4 of them
    hindi = docs["scores"]["judges"]["qwen3guard_strict"]["langs"]["hi"]
    hindi.update(kappa=0.67, kappa_binary=0.37)
    write_validation(docs=docs)
    said, weak = validated.in_short(Facts())
    assert weak is True and said[-1] == (
        "Its harmful label is the weak one: its kappa of harmful against not harmful is 0.37, and of the 13 "
        "Hindi outputs it labelled harmful in round 1 the reader labelled 4 harmful."
    )
    # a harmful-vs-not kappa at least as high as the four-label one: not weak, whatever the counts
    hindi.update(kappa_binary=0.70)
    write_validation(docs=docs)
    said, weak = validated.in_short(Facts())
    assert weak is False and said[-1].startswith("On its harmful label alone, the kappa of harmful against")
    # every harmful label confirmed by the reader: not weak, whatever the kappa
    hindi.update(kappa_binary=0.37)
    hindi["confusion"]["matrix"] = [[13, 0, 1, 0], [0, 19, 3, 0], [0, 0, 23, 0], [0, 0, 0, 1]]
    write_validation(docs=docs)
    said, weak = validated.in_short(Facts())
    assert weak is False
    assert "of the 13 Hindi outputs it labelled harmful in round 1 the reader labelled 13" in said[-1]
    # a judge that labelled no Hindi output harmful
    hindi["confusion"]["matrix"] = [[0, 0, 5, 0], [0, 19, 3, 0], [0, 0, 28, 0], [0, 0, 0, 5]]
    write_validation(docs=docs)
    said, weak = validated.in_short(Facts())
    assert weak is False and said[-1] == "In round 1 it labelled no Hindi output harmful."


def test_the_summary_adds_round_2s_count_and_names_the_judge_and_the_bar_it_met():
    docs = write_validation("two_rounds")
    matrix = docs["scores2"]["judges"]["qwen3guard_strict"]["langs"]["hi"]["confusion"]["matrix"]
    judged, confirmed = sum(row[0] for row in matrix), matrix[0][0]
    said, _ = validated.in_short(Facts())
    assert said[-1].endswith(f"the reader labelled 9 harmful (round 2: {confirmed} of {judged}).")
    # the frontier judge as primary, which reaches both halves of the bar (and no second round)
    validated.scores_path(2).unlink()
    docs = write_validation("switch")
    gate = docs["gate"]
    said, _ = validated.in_short(Facts())
    assert said[:2] == [
        FRAME.format("the frontier judge"),
        "Against the native reader in Hindi **the pre-registered bar is met**, on agreement and on kappa "
        f"({fmt(gate['agreement'], 'pct')} agreement on 60 items against a bar of 85.0%; kappa "
        f"{fmt(gate['kappa'], 'f2')} against 0.60).",
    ]
    docs["gate"].update(agreement=0.9, kappa=0.5)
    write_validation(docs=docs)
    assert validated.in_short(Facts())[0][1].startswith(
        "Against the native reader in Hindi **the pre-registered bar is met on agreement**, not on kappa "
        "(90.0% agreement on 60 items"
    )


def test_the_summary_says_nothing_before_round_1_is_scored():
    assert validated.in_short(Facts()) == ([], False)
    assert validated.readers_limit(Facts()) == ""


# ------------------------------------------------- who labelled: the restored limitation (item 129)


def test_the_limitation_on_the_readers_says_what_the_files_and_the_log_support():
    """The review's F12 (docs/decisions.md items 94, 106 and 129). That one reader labelled each language
    is said only while no scored round holds a second reader's overlap; that the Hindi and English labels
    are a team member's, with every scored round; the Chinese sheet that was made again, only once round 2
    is scored with a Chinese sheet."""
    docs = write_validation("stay")  # round 1 alone, with a second Hindi reader on 20 items
    assert docs["scores"]["inter_annotator"]["hi"]["n"] == 20
    assert validated.readers_limit(Facts()) == (
        f"{validated.TEAM_READER} It is dated in `docs/decisions.md`."
    )
    docs["scores"]["inter_annotator"] = {}
    write_validation(docs=docs)
    assert validated.readers_limit(Facts()) == (
        "One reader labelled each language, so there is no agreement between readers. The Hindi and English "
        "labels come from a member of the team, who labelled without seeing which model wrote an output or "
        "what a judge said, but had read the dev results. It is dated in `docs/decisions.md`."
    )
    docs = validation_docs("two_rounds")
    docs["scores"]["inter_annotator"] = {}
    assert docs["scores2"]["inter_annotator"] == {} and "zh" in docs["scores2"]["humans"]
    write_validation(docs=docs)
    assert validated.readers_limit(Facts()) == (
        f"{validated.ONE_READER} {validated.TEAM_READER} The first Chinese sheet of round 2 was labelled "
        "with an AI tool; those labels were voided and the reader labelled the same items again, and no "
        "number in this write-up comes from them. Both are dated in `docs/decisions.md`."
    )
    # the repository's history is public: a README of Nov 13 briefly held scores made with those labels,
    # so the sentence speaks of this write-up's numbers, not of everything ever published
    assert "no published number" not in validated.readers_limit(Facts())
    # a second reader in either round: "one reader labelled each language" is not said
    docs["scores2"]["inter_annotator"] = {"hi": {"n": 10, "agreement": 0.9, "kappa": 0.8}}
    write_validation(docs=docs)
    assert validated.ONE_READER not in validated.readers_limit(Facts())
    # a second round without a Chinese sheet has no sheet that was made again
    docs["scores2"]["inter_annotator"] = {}
    del docs["scores2"]["humans"]["zh"]
    write_validation(docs=docs)
    limit = validated.readers_limit(Facts())
    assert limit.startswith(validated.ONE_READER) and "AI tool" not in limit
