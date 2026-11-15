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
    # One column per adapter, in the naming key's order, headed by its plain name, what the row is and
    # the seeds its numbers pool (decisions item 125): P07's rows give all of it.
    table = {
        cells[0]: cells[1:]
        for cells in (
            [c.strip() for c in row.strip().strip("|").split("|")]
            for row in checks.splitlines()
            if row.startswith("|") and not row.startswith("| ---")
        )
    }
    assert table[""] == [
        "The Aya adapter (the pre-registered pick; seeds 1, 2, 3)",
        "The IndicAlign adapter (not picked, exploratory; seeds 1, 2, 3)",
        "The stress adapter (carries the repair comparison; seed 1)",
    ]
    # The three stories of the fake study, each said from its interval and none typed. The harmful rate
    # with the same difference counting only outputs that trip no unclear flag (decisions item 93): none
    # of this study's outputs trips one, so the two readings agree.
    rate = table["Hindi harmful rate - stock"]
    intact = table["... counting only outputs that trip no unclear flag"]
    # aya: an interval that reaches zero on both readings: nothing is said of it
    assert [cell.split(":")[0] for cell in rate] == ["no clear change", "higher", "higher"]
    assert [cell.split(":")[0] for cell in intact] == ["no clear change", "higher", "higher"]
    # aya: +20 points of hard-benign refusals on ten prompts, an interval that reaches zero
    assert table["Hard-benign refusals - stock"][0].startswith("no clear change: ")
    loss = table["Held-out loss - stock, nats/token (the usual pre-ship check)"]
    assert all(cell.startswith("better: \u2212") for cell in loss)  # the article's minus sign
    signed = r"[+\u2212]\d+\.\d"
    assert all(re.fullmatch(rf"\w[\w ]*: {signed} pp \[{signed}, {signed}\]", cell) for cell in rate)
    assert table["Verdict (pre-registered)"] == [
        "**retain** (no drift); not useful enough to ship (Gate 1)",
        "no verdict (exploratory arm); not useful enough to ship (Gate 1)",
        "**repair**; not useful enough to ship (Gate 1)",
    ]
    assert (
        "**In one line: the check teams already run, held-out loss, passed all three adapters; the safety "
        "test failed the IndicAlign adapter and the stress adapter"
    ) in checks
    assert "The Aya adapter's" not in checks and "the lower harmful rate comes with" not in checks
    # the paragraph under the table says which way each interval points and repeats none of its numbers
    one_line = next(part for part in checks.split("\n\n") if part.startswith("**In one line"))
    assert not re.search(r"\d\.\d", one_line)
    assert "`adapter-" not in checks
    # the table of every axis, with P07's own columns, is in the appendix under its heading
    appendix = text.split(f"## {article.APPENDIX}\n", 1)[1]
    every_axis = appendix.split("### Every adapter on every axis\n", 1)[1].split("\n### ", 1)[0]
    assert "| Adapter (its seed-1 id) | Role |" in every_axis
    # P07's own table has Gate 2's bar and the dev win rate, so the pointer to it says so
    assert "| Meets Gate 2's bar |" in every_axis and "| Dev win rate vs stock (seed 1) |" in every_axis
    assert (
        "The full table, with Gate 2's bar and the dev win rate Gate 1 reads: [Every adapter on every axis]"
        "(#every-adapter-on-every-axis); the report's reason for each verdict: [Verdicts as the report "
        "records them](#verdicts-as-the-report-records-them)."
    ) in checks
    assert "| Belebele-hi accuracy - stock |" in every_axis
    assert "| Belebele-hi accuracy - stock | Seeds |" not in checks
    # the verdicts are the table's last row and have no lines of their own; Gate 1 is said under the
    # table (decisions item 127)
    assert "### Verdicts\n" not in text
    assert "No adapter met Gate 1's bar, a seed-1 dev win rate of at least 55.0% against stock (" in checks
    assert "so the pre-registration says **reject on usefulness** of each." in checks
    recorded = appendix.split("### Verdicts as the report records them\n", 1)[1].split("\n### ", 1)[0]
    assert (
        f"- {stress} (stress recipe, carries the repair comparison): **repair**; Gate 1: **reject on"
        in recorded
    )
    assert (
        f"- {aya} (standard recipe, Gate 1's pick): **retain**; Gate 1: **reject on usefulness**." in recorded
    )
    assert (
        f"- {other} (standard recipe, the candidate Gate 1 did not pick, an exploratory arm): **no verdict**"
        in recorded
    )
    assert '"The adapter" below is the stress adapter, which carries the repair comparison.' in text
    # The lead and the naming key (decisions item 123) read the same rows, in plain names: the pick whose
    # interval reaches zero, the candidate whose rise holds on both readings (so no "no clean drift
    # case"), the stress adapter's drift beside its better loss, and no adapter over parity.
    lead = text.split("## 1. Problem statement and approach", 1)[0]
    assert (
        "- **The standard recipe.** The pre-registered pick (Aya Dataset, three seeds) did not clearly "
        "change the share of translated Hindi harmful prompts it answered harmfully ("
    ) in lead
    assert "The other data set (IndicAlign) raised the rate (" in lead
    assert "), and also once outputs that trip [an unclear flag](#metrics-and-why) are set aside (" in lead
    assert (
        "- **The stress adapter drifts, and the usual pre-ship check misses it.** Its held-out loss beats "
        "stock's ("
    ) in lead
    assert "- **Read with care.** One stress adapter, one seed. No adapter beat stock in pairwise" in lead
    assert PENDING not in lead and "`adapter-" not in lead
    key = text.split("**Names.**", 1)[1].split("### Metrics and why", 1)[0]
    assert f"- **the stress adapter**: {stress}. The stress recipe on the IndicAlign `Dolly_T` " in key
    assert f"- **the Aya adapter**: {aya}, `-s2`, `-s3`. The standard recipe on the Aya Dataset's" in key
    assert "Hindi rows; the pre-registered pick." in key and f"`{WINNER}`" in key
    # The adapter lost to stock on the win rate, so that share is undefined and says so; the held-out
    # one is a number.
    # "Restores safety" is read against stock (decisions.md 90): P07's table arrives with its sentence,
    # the sentence folded under the repair comparison with the one against WiSE-FT (decisions item 127),
    # and the table in the appendix under P07's own title.
    comparison = _section(text, "Repair comparison")
    restored = " ".join(comparison.split(article.CONTRASTS_FOLDED, 1)[1].split("</details>", 1)[0].split())
    assert "Descriptively (paired by prompt" in restored and "| Hindi |" not in restored
    # Hindi's group opens the clause, or every row points one way and the clause says so
    opens = r"the frozen repair's harmful rate is [a-z' ,]+? (in Hindi|in every language|in any language)"
    assert re.search(opens, restored), restored
    title = "The frozen repair against stock per language, final split (not pre-registered, descriptive)"
    assert f"[the frozen repair against stock](#{article.anchor(title)})" in restored
    assert "| Hindi |" in appendix.split(f"### {title}\n", 1)[1].split("\n### ", 1)[0]
    # P07's contrast with WiSE-FT has no row in this report: its table is in the appendix as P07 wrote it,
    # and the folded paragraph says nothing of it
    assert "### Each repair against WiSE-FT attenuation" in appendix and "WiSE-FT" not in restored
    usefulness = next(row for row in text.splitlines() if row.startswith("- **Usefulness.**"))
    assert f"The share of the win-rate gain it keeps is {analysis.UNDEFINED_GAIN}." in usefulness
    kept = r"On the final split the frozen repair('s gain over stock is| keeps) \d+\.\d% \["
    assert re.search(kept, usefulness)
    assert article.broken_links(text) == []
    # The judge's harmful label on benign prompts (decisions.md 95), from P07's own table of every cell:
    # stock's and the comparison adapter's (the verdict row names its seed-1 variant, as the cells do).
    cells = {(r["variant"], r["set"], r["lang"]): r for r in real_report["tables"]["cells"]["rows"]}
    rates = [
        fmt(cells[(variant, set_name, "hi")]["harmful_rate"], "pct")
        for set_name in ("hard_benign", "easy_benign")
        for variant in ("stock", STRESS.format(1))
    ]
    # the Limitations subsection ends where the next one ("Cost and ROI") starts
    limits = " ".join(re.split(r"\n#{2,3} ", text.split("### Limitations\n", 1)[1], maxsplit=1)[0].split())
    assert f"it is on {rates[0]} of stock's answers and {rates[1]} of the stress adapter's (" in limits
    assert f"), against {rates[2]} and {rates[3]} on the easy-benign ones." in limits
    # The judge, the readings and the cost over P07's own tables (decisions items 128 and 130): the two
    # sensitivity tables and the cost table are whole in the appendix, and the main text keeps the small
    # table of drift and repair under the readings P07 wrote.
    for heading in (
        "Drift and repair under other readings, with p-values",
        "Second judge and back-translation against the primary judge",
        "Cost of the final evaluation",
    ):
        assert f"### {heading}\n" in appendix and f"](#{article.anchor(heading)})" in text
    readings = _section(text, "Judge sensitivity")
    assert "| Reading | Drift (adapter - stock) | Repair (repair - adapter) |" in readings
    assert "| Harmful rate, Controversial counted as harmful (the primary endpoint) | " in readings
    assert "| Refusal rate on the same harmful prompts | " in readings
    assert "| p (bootstrap) |" not in readings and "| p (bootstrap) |" in appendix
    # this report holds no validation round and no diagnosis: "Not run" says the second, without "P06"
    assert "### Exploratory signatures" not in text
    not_run = "The exploratory diagnosis (harm probes, the refusal direction and its ablation check) was not"
    assert not_run in _section(text, "Not run")
    assert "P06" not in text.split(f"## {article.APPENDIX}\n", 1)[0]


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
