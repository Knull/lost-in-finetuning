"""The article, the model card and the logs card over a final_report.json that P07's own writer built
(lift.report.analysis.compute -> lift.report.tables.build), not the stand-in of publish_fakes: the three
adapters as they stand after Gates 1 and 2, the stress adapter carrying the repair comparison. A change
on either side of the contract (a table, a format, a verdict row, an "undefined" value) fails here."""

from __future__ import annotations

import json
import re

import pytest
from publish_fakes import (  # noqa: F401
    FIGURES,
    no_committed_recipe,
    selection,
    write_figures,
    write_report,
)
from test_final_report import (  # noqa: F401
    OTHER,
    STRESS,
    A,
    ThreeAdapters,
    one_candidate,
    three_adapters,
)

from lift.publish import article
from lift.publish.inputs import OPTIONAL, PENDING, TABLES, Facts, fmt
from lift.report import analysis, cost, tables

WINNER = ThreeAdapters.WIN.format(1)


@pytest.fixture
def real_report(tmp_path, monkeypatch, one_candidate):  # noqa: F811
    """final_report.json as lift report writes it for the three adapters, bound to a selection of the
    stress adapter whose recipe is frozen and tagged, with every figure drawn."""
    from lift.publish import acceptance

    source, sel = three_adapters(one_candidate)
    stats = analysis.compute(source, sel, use_file=False)
    doc = json.loads(
        json.dumps(tables._json_safe(tables.build(stats, cost.section([("stock", "stock")], docs=[]))))
    )
    assert tables.validate(doc) == []
    write_report(doc, selection(adapter=STRESS.format(1)))
    write_figures(FIGURES)
    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
    notebook = tmp_path / "acceptance_test.ipynb"
    notebook.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(acceptance, "notebook_path", lambda: notebook)
    return doc


def _section(text: str, head: str) -> str:
    start = text.index(f"### {head}")
    return text[start : text.index("\n### ", start + 1)]


def test_the_article_reads_the_report_p07_writes(real_report):
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    # Nothing the report is asked for is missing: what stays pending is for a person or a later command.
    pending = article.placeholders(text)
    assert not [p for p in pending if "final_report.json" in p or "lift report" in p], pending
    for name in set(TABLES) - set(OPTIONAL):
        assert f"/tables/{name}/" in json.dumps([s["pointer"] for s in facts.sources]), name
    checks = _section(text, "What each check said")
    stress, aya, other = (f"`{x.format(1)}`" for x in (STRESS, A, OTHER))
    assert (
        f"- {stress} (stress recipe, carries the repair comparison). The usual checks: held-out loss better"
        in checks
    )
    assert f"- {aya} (standard recipe, Gate 1's pick)." in checks
    assert f"- {other} (standard recipe, the candidate Gate 1 did not pick, an exploratory arm)." in checks
    line = {
        x: next(row for row in checks.splitlines() if row.startswith(f"- {x}")) for x in (stress, aya, other)
    }
    # The three stories of the fake study, each said from its interval and none typed:
    # each with the same difference counting only outputs that trip no unclear flag (decisions item 93):
    # none of this study's outputs trips one, so the two readings agree
    intact = "counting only outputs that trip no unclear flag)"
    assert f"Hindi harmful rate higher than stock's, over Gate 2's bar (also higher {intact}" in line[stress]
    assert f"Hindi harmful rate higher than stock's, over Gate 2's bar (also higher {intact}" in line[other]
    # aya: +20 points of hard-benign refusals on ten prompts, an interval that reaches zero: nothing is said
    assert (
        f"no clear change in the Hindi harmful rate (also no clear change {intact}; no clear change in "
        "refusals"
    ) in line[aya]
    assert all("held-out loss better than stock's" in row for row in line.values())
    assert all("Gate 1: reject on usefulness" in row for row in line.values())
    assert f"For {stress} and {other}, the held-out loss improved while the Hindi harmful rate rose" in checks
    assert f"For {aya}" not in checks and "the lower harmful rate comes with" not in checks
    words = checks.split("| Adapter (seed 1) |", 1)[0]
    assert not re.search(r"\d\.\d", words)
    assert "| Belebele-hi accuracy - stock |" in checks
    verdicts = _section(text, "Verdicts")
    assert (
        f"- {stress} (stress recipe, carries the repair comparison): **repair**; Gate 1: **reject on"
        in verdicts
    )
    assert (
        f"- {aya} (standard recipe, Gate 1's pick): **retain**; Gate 1: **reject on usefulness**." in verdicts
    )
    assert (
        f"- {other} (standard recipe, the candidate Gate 1 did not pick, an exploratory arm): **no verdict**"
        in verdicts
    )
    assert f'carried by {stress} (stress recipe): "the adapter" below' in text
    # The adapter lost to stock on the win rate, so that share is undefined and says so; the held-out
    # one is a number.
    # "Restores safety" is read against stock (decisions.md 90): P07's table arrives with its sentence.
    restored = " ".join(_section(text, "The frozen repair against stock per language").split())
    assert "Is safety restored? Descriptively (paired by prompt" in restored
    assert "On Hindi the frozen repair's harmful rate is " in restored and "| Hindi |" in restored
    assert text.index("### The frozen repair against stock") < text.index("### Each repair against WiSE-FT")
    assert f"- of the win-rate gain: {analysis.UNDEFINED_GAIN}" in text
    assert re.search(r"- of the held-out-loss gain: \d+\.\d% \[", text)
    # The judge's harmful label on benign prompts (decisions.md 95), from P07's own table of every cell:
    # stock's and the comparison adapter's (the verdict row names its seed-1 variant, as the cells do).
    cells = {(r["variant"], r["set"], r["lang"]): r for r in real_report["tables"]["cells"]["rows"]}
    rates = [
        fmt(cells[(variant, set_name, "hi")]["harmful_rate"], "pct")
        for set_name in ("hard_benign", "easy_benign")
        for variant in ("stock", STRESS.format(1))
    ]
    limits = " ".join(text.split("### Limitations\n", 1)[1].split("\n## ", 1)[0].split())
    assert f"it is on {rates[0]} of stock's answers and {rates[1]} of {stress}'s (" in limits
    assert f"), against {rates[2]} and {rates[3]} on the easy-benign ones." in limits


def test_the_model_card_and_the_logs_card_read_the_report_p07_writes(real_report, monkeypatch):
    from lift.publish import datasets, model
    from lift.schemas import VariantSpec

    spec = VariantSpec(
        name=WINNER,
        method="wiseft",
        adapter=STRESS.format(1),
        seed=1,
        params={"lambda": 0.5},
        artifacts={"adapter": "a" * 64},
    )
    card = model.model_card(Facts(), spec.name, spec)
    # The winner's adapter drifted and the repair interval lies below zero: the one safety sentence is made,
    # although the report also carries the standard adapter's "retain".
    assert "safer than the adapter" in card
    assert PENDING not in card.replace("[PENDING: the HF", "")
    # The README labels the stress recipe as a stress test wherever it is used, and Gate 1 rejected the
    # adapter on usefulness: the card says both, and does not offer the model for Hindi instruction following.
    assert "**Stress test.** The adapter under this repair was trained with the stress recipe" in card
    assert "**Usefulness: reject on usefulness.** On dev, the adapter's Hindi answers were preferred" in card
    assert "in 28.0% of the pairs; Gate 1 asks for 55.0%." in card
    use = card[card.index("## Intended use") : card.index("## Limitations")]
    assert "Hindi instruction following" not in use and "Research on safety drift" in use
    adapter_card = model.adapter_card(STRESS.format(1), spec)
    assert "as a stress test" in adapter_card and "not a recommended Hindi adapter" in adapter_card
    assert "standard recipe to be useful in Hindi" in model.adapter_card(A.format(1), spec)
    lines = datasets.verdict_lines(Facts())
    assert lines.count("\n") == 2 and "**no verdict**" in lines and "**not run**" not in lines
