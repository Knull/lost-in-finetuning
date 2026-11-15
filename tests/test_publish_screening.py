"""Checkpoint 3 of the review's fixes, as lift publish article and lift publish space write them: the cap-hit
rate the pre-registration says is reported (review F15, docs/decisions.md item 132), the extra on running
the test on another adapter and the Space called a results explorer (F4, item 133), and how far the readers
share the judge's refusal label (F3, item 134)."""

from __future__ import annotations

import json
import re

import pytest
from publish_fakes import (  # noqa: F401
    CAST_WINS,
    STRESS,
    WINNER,
    no_committed_recipe,
    report,
    selection,
    validation_docs,
    write_inputs,
    write_validation,
)
from test_publish_article import _appendix, _inputs, _limitations, _limits, _part, _publish_cfg

from lift.publish import article, settings, validated
from lift.publish.inputs import PENDING, Facts

LAMBDA = "λ"
REPORT = "outputs/metrics/final_report.json"


@pytest.fixture
def out(tmp_path, monkeypatch):
    """The article, its sources and the notebook it stamps, in a temporary folder (as the article's tests
    have them)."""
    from lift.publish import acceptance

    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
    notebook = tmp_path / "acceptance_test.ipynb"
    notebook.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(acceptance, "notebook_path", lambda: notebook)
    return tmp_path

# ------------------------------------------------------------- the cap-hit rate (decisions item 132)

CUT = "Answers stop at 256 new tokens, and an answer that reaches the cap is judged as it was cut."
CAP_LINK = "[Answers that reached the token cap, Hindi](#answers-that-reached-the-token-cap-hindi)"
NEXT = " - The harmful rate is one judge's strict reading"  # the bullet after it


def _bullet(limits: str) -> str:
    """The cap-hit bullet of a Limitations list on one line, without its dash."""
    rest = limits.split("- Answers stop at", 1)[1].split(NEXT, 1)[0]
    return f"Answers stop at{rest}"


def _caps(doc: dict, caps: dict) -> dict:
    """``doc`` with the cap-hit rate of the Hindi cells {(variant, set): rate} set (None: no rate)."""
    for row in doc["tables"]["cells"]["rows"]:
        if (row["variant"], row["set"]) in caps and row["lang"] == "hi":
            row["cap_hit_rate"] = caps[(row["variant"], row["set"])]
    return doc


def test_the_cap_hit_rate_is_reported_from_the_final_cells(out):
    """PREREGISTRATION.md: "We report the cap-hit rate". One bullet of the limitations, before the one on
    how the judge reads an output that runs to the cap: stock's and the comparison adapter's rates on the
    Hindi harmful and benign prompts, each read from P07's cells table and recorded, and whose answers were
    cut more often where every set says the same. The rates of stock, that adapter and each frozen setting
    with a cell are a small table in the appendix; the cells table itself stays unplaced."""
    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    limits = _limitations(text)
    assert f"do not adjudicate. - {CUT} Stock's answers" in limits
    assert _bullet(limits) == (
        f"{CUT} Stock's answers reached it on 40.0% of the Hindi harmful prompts and the Aya adapter's on "
        "15.0%; on the Hindi benign prompts, stock's on 70.0% (easy) and 80.0% (hard), the Aya adapter's on "
        "10.0% and 35.0%. So on each of these sets more of stock's answers than of the Aya adapter's were "
        f"judged cut short. Table: {CAP_LINK}."
    )
    recorded = {(s["file"], s["pointer"], s["shown"]) for s in facts.sources}
    for row, shown in ((0, "40.0%"), (5, "15.0%"), (3, "70.0%"), (1, "80.0%"), (4, "10.0%"), (2, "35.0%")):
        assert (REPORT, f"/tables/cells/rows/{row}/cap_hit_rate", shown) in recorded
    assert ("configs/project.yaml", "/generation/max_new_tokens", "256") in recorded
    block = _appendix(text, article.CAP_HEAD)
    rows = [line for line in block.splitlines() if line.startswith("|")]
    assert rows[0] == (
        "| Variant | Role | Harmful | Harmful, human-written (Aya Red-teaming) | Easy benign | Hard benign |"
    )
    # the frozen settings with no Hindi cell in the table have no row; a set with no cell is not run
    assert rows[2:] == [
        "| stock | stock | 40.0% | not run | 70.0% | 80.0% |",
        "| adapter-aya-std-s1 | adapter | 15.0% | not run | 10.0% | 35.0% |",
        f"| {WINNER} | frozen repair | 20.0% | not run | not run | not run |",
    ]
    assert (REPORT, "/tables/cells/rows/6/cap_hit_rate", "20.0%") in recorded
    assert "reached the cap of 256 new tokens, on the Hindi prompts of the final split (seed 1)" in block
    # the appendix's block stands before the cost table, whose note points to it
    blocks = [line[4:] for line in _part(text, article.APPENDIX).splitlines() if line.startswith("### ")]
    assert blocks.index(article.CAP_HEAD) + 1 == blocks.index("Cost of the final evaluation")
    assert "cells" in article.UNPLACED
    assert not any(s["pointer"] == "/tables/cells/note" for s in facts.sources)
    assert "`adapter-" not in limits and article.broken_links(text) == []


def test_the_cap_hit_sentence_says_only_what_the_numbers_show(out):
    """Whose answers were cut more often is said only where every pair points one way; a set without stock's
    rate is left out, and the adapter's rates only where it has every one stock has."""
    stock, aya = "stock", "adapter-aya-std-s1"
    higher = {(aya, "harmful"): 0.50, (aya, "easy_benign"): 0.75, (aya, "hard_benign"): 0.90}
    said = _bullet(_limits(_caps(report(), higher))[0])
    assert said.endswith(f"So on each of these sets more of the Aya adapter's answers than of stock's were "
                         f"judged cut short. Table: {CAP_LINK}.")
    mixed = {(aya, "harmful"): 0.50}  # above stock's on the harmful prompts, under it on the benign ones
    said = _bullet(_limits(_caps(report(), mixed))[0])
    assert "50.0%" in said and " So " not in said
    no_easy = {(aya, "easy_benign"): None}  # the adapter lacks a rate stock has: its rates are left out
    said = _bullet(_limits(_caps(report(), no_easy))[0])
    assert said == (
        f"{CUT} Stock's answers reached it on 40.0% of the Hindi harmful prompts; on the Hindi benign "
        f"prompts, stock's on 70.0% (easy) and 80.0% (hard). Table: {CAP_LINK}."
    )
    no_harmful = {(stock, "harmful"): None}  # stock's harmful cell has no rate: the benign ones alone
    said = _bullet(_limits(_caps(report(), no_harmful))[0])
    assert said.startswith(
        f"{CUT} Stock's answers reached it on 70.0% (easy) and 80.0% (hard) of the Hindi benign prompts, the "
        "Aya adapter's on 10.0% and 35.0%. So on each of these sets more of stock's answers"
    )
    harmful_only = {(stock, "easy_benign"): None, (stock, "hard_benign"): None}
    said = _bullet(_limits(_caps(report(), harmful_only))[0])
    assert "So more of stock's answers than of the Aya adapter's were judged cut short." in said
    none = {(stock, s): None for s in ("harmful", "easy_benign", "hard_benign")}
    said = _bullet(_limits(_caps(report(), none))[0])
    assert said == f"{CUT} No Hindi cell of stock has a cap-hit rate in the report."


def test_the_cap_hit_rate_waits_for_its_column(out):
    """A report written before P07's cells held the rate, and no report at all: the bullet is pending, and
    the appendix has no block for it."""
    doc = report()
    for row in doc["tables"]["cells"]["rows"]:
        del row["cap_hit_rate"]
    limits, facts = _limits(doc)
    assert f"- {PENDING}: the cap-hit rates of the final cells from {REPORT}" in limits
    assert any("cap-hit rates" in m for m in facts.missing)
    text = article.build()[0].read_text(encoding="utf-8")
    assert f"### {article.CAP_HEAD}" not in text and "how often they reached the token cap" not in text
    article.report_path().unlink()
    text = article.build()[0].read_text(encoding="utf-8")
    assert f"- {PENDING}: the cap-hit rates of the final cells from {REPORT}" in _limitations(text)


def test_the_stress_study_says_the_cap_hits_of_its_stress_adapter(out):
    """A study shaped like ours: the stress adapter carries the repair comparison, and its rates are the
    ones said beside stock's; the table holds every frozen setting that has a Hindi cell."""
    facts = _inputs()
    text = article.build(facts)[0].read_text(encoding="utf-8")
    said = _bullet(_limitations(text))
    assert said.startswith(
        f"{CUT} Stock's answers reached it on 40.0% of the Hindi harmful prompts and the stress adapter's on "
        "16.0%; on the Hindi benign prompts, stock's on 70.0% (easy) and 80.0% (hard), the stress adapter's "
        "on 9.0% and 36.0%. So on each of these sets more of stock's answers than of the stress adapter's"
    )
    rows = [line for line in _appendix(text, article.CAP_HEAD).splitlines() if line.startswith("|")][2:]
    assert [row.split(" | ")[0] for row in rows] == ["| stock", f"| {STRESS}", f"| wiseft-l0.25-{STRESS}"]
    assert rows[1].startswith(f"| {STRESS} | adapter | 16.0% |")
    assert rows[2].startswith(f"| wiseft-l0.25-{STRESS} | frozen repair | 17.0% |")
    note = _appendix(text, "Cost of the final evaluation")
    assert "So it moves with the length of the answers (see [how often they reached the token cap](" in note


# ------------------------------------------- the extra on the notebook (docs/decisions.md item 133)

NOTEBOOK = "https://github.com/o/r/blob/main/notebooks/acceptance_test.ipynb"


def _run_it(text: str) -> str:
    return " ".join(_part(text, article.RUN_IT).split())


def test_the_extra_on_the_notebook_says_what_it_runs_and_what_it_leaves_out(out, monkeypatch):
    """Review F4: between the related work and Reproduce, the notebook's link, what it does and needs, the
    frozen repair it applies by its plain name, that its verdict covers safety and over-refusal only, and
    that the frozen setting was chosen on our adapter, so for another it is a starting point, and that the
    notebook does not run the sweep and selection that chose it (no route is offered that a clone refuses).
    No run is timed here, so none is spoken of."""
    _publish_cfg(monkeypatch, github_repo="https://github.com/o/r")
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    heads = [line[3:] for line in text.splitlines() if line.startswith("## ")]
    assert heads[-5:] == ["Related work", article.RUN_IT, "Reproduce", "Links", article.APPENDIX]
    part = _run_it(text)
    assert part.startswith(
        f"[`notebooks/acceptance_test.ipynb`]({NOTEBOOK}) runs a screening version of this test on any LoRA "
        "adapter of Tiny Aya Global, with the study's own code: stock and your adapter answer the first "
        "`LIMIT` dev prompts of the harmful and hard-benign sets in Hindi and English"
    )
    assert f"and then the frozen repair (WiSE-FT at {LAMBDA} = 0.50) is applied to your adapter" in part
    assert (
        "Its verdict covers safety and over-refusal only: the full test also needs `lift utility` and `lift "
        f"belebele`. The frozen repair's {LAMBDA} = 0.50 was chosen on our adapter's dev prompts, so for "
        "yours it is a starting point, not a value tuned for it. The study chose it with its dev sweep and "
        "selection (`lift sweep`, then `lift select`), which the notebook does not run."
    ) in part
    assert "choose it on your own" not in part
    assert "The run on record" not in part and "minutes" not in part
    assert part.count("<details><summary>") == part.count("</details>") == 1
    # the same notebook in the links, and the Space's line points here
    assert f"- Notebook, to run the test on your adapter: {NOTEBOOK}\n" in _part(text, "Links")
    assert f"The test itself runs in the notebook: [{article.RUN_IT}](#run-the-test-on-your-adapter)." in text
    assert article.broken_links(text) == []


def test_the_extra_on_the_notebook_follows_the_frozen_repair(out, monkeypatch):
    """A CAST winner is fitted on the adapter first (the notebook's register_repair refuses it otherwise),
    and with nothing frozen the section names no setting."""
    _publish_cfg(monkeypatch, github_repo="https://github.com/o/r")
    write_inputs(selection_doc=selection(CAST_WINS))
    part = _run_it(article.build()[0].read_text(encoding="utf-8"))
    assert (
        "then the frozen repair (CAST at strength 1.0, fitted on your adapter first with `lift fit-cast`) is "
        "applied to your adapter"
    ) in part
    assert "The frozen repair's strength 1.0 was chosen on our adapter's dev prompts" in part
    article.report_path().unlink()
    article.selection_path().unlink()
    part = _run_it(article.build()[0].read_text(encoding="utf-8"))
    assert "and then the frozen repair is applied to your adapter" in part
    assert "The frozen repair is chosen on our adapter's dev prompts, so for yours it is a starting" in part


def test_everything_the_extra_on_the_notebook_names_exists(out, monkeypatch):
    """Each command the section names is one the CLI has, in a form that parses; the notebook's variable and
    helpers are in its cells and in lift.publish.acceptance; the prompt files are gitignored; and what it
    says the notebook does (the restart, the two ways to get the prompts, the scope of the verdict) the
    notebook says too. The section cannot drift from the code."""
    from lift import cli, paths
    from lift.publish import acceptance

    _publish_cfg(monkeypatch, github_repo="https://github.com/o/r")
    write_inputs(selection_doc=selection(CAST_WINS))  # the CAST winner adds lift fit-cast
    part = _run_it(article.build()[0].read_text(encoding="utf-8"))
    code = re.findall(r"`([^`]+)`", part)
    parser = cli.build_parser()
    has = set(next(a for a in parser._actions if getattr(a, "choices", None)).choices)
    commands = {m.group(1) for c in code for m in [re.match(r"lift ([a-z][a-z-]*)", c)] if m}
    assert commands == {"utility", "belebele", "sweep", "select", "fit-cast", "sync", "data"}
    assert commands <= has
    for argv in (["sync", "pull", "--only", "data"], ["data", "build"]):
        assert " ".join(["lift", *argv]) in " ".join(code)
        parser.parse_args(argv)
    nb = json.loads(acceptance.notebook_path().read_text(encoding="utf-8"))
    code_cells = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    markdown = " ".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    prose = " ".join(markdown.split())
    assert re.search(r"^LIMIT = \d+", code_cells, re.M) and "`LIMIT`" in part
    for helper in ("check_ready", "fetch_prompts"):
        assert f"`acceptance.{helper}`" in part and callable(getattr(acceptance, helper))
        assert f"acceptance.{helper}(" in code_cells
    assert "lift sync pull --only data" in code_cells and "data build" in acceptance.fetch_prompts.__doc__
    files = {c for c in code if re.fullmatch(r"[\w./-]+\.ipynb", c)}
    assert files == {"notebooks/acceptance_test.ipynb"} and (paths.REPO_ROOT / files.pop()).is_file()
    ignored = (paths.REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "`data/prompts/`" in part and "data/*" in ignored and "!data/prompts/" not in ignored
    assert "Restart the session" in prose and "session restart" in part
    assert "safety and over-refusal only" in prose
    assert "`lift utility`" in prose and "`lift belebele`" in prose
    assert "pip install" in code_cells and "-c constraints.txt" in code_cells


# ---------------------------------------------------------------- the Space (docs/decisions.md item 133)


def test_the_space_is_a_results_explorer_that_links_the_notebook(monkeypatch):
    """The Space loads no model and generates nothing: its card and its page say it is a results explorer,
    in their title and first line, and link the notebook as the way to run the test. The repo keeps its
    name."""
    import yaml

    from lift.publish import space

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    cfg = {**settings.load(), "github_repo": "https://github.com/o/r"}
    monkeypatch.setattr(settings, "load", lambda: cfg)
    up = space.build()
    assert up.repo.repo_id == "test-org/lift-acceptance-test"
    card = (up.folder / "README.md").read_text(encoding="utf-8")
    front, body = card.split("---\n", 2)[1:]
    meta = yaml.safe_load(front)
    assert meta["title"] == "Lost in Fine-Tuning: results explorer" and meta["sdk"] == "static"
    assert len(meta["short_description"]) <= 60 and "Results explorer" in meta["short_description"]
    lines = [line for line in body.strip().splitlines() if line]
    assert lines[0] == "# Lost in Fine-Tuning: results explorer"
    assert lines[1].startswith("A results explorer for the retain / repair / reject test")
    first = " ".join(body.strip().split("\n\n")[1].split())
    assert "It loads no model and generates nothing." in first
    assert first.endswith(
        f"To run the test on an adapter of your own, use the acceptance-test notebook: {NOTEBOOK}"
    )
    assert PENDING not in card and "{{" not in card
    page = (up.folder / "index.html").read_text(encoding="utf-8")
    assert "<title>Lost in Fine-Tuning: results explorer</title>" in page
    assert "<h1>Lost in Fine-Tuning: results explorer</h1>" in page
    assert '<p class="sub">A results explorer: it loads no model and generates nothing.' in page
    assert f'<a href="{NOTEBOOK}">{space.NOTEBOOK_LINK}</a>' in page
    # with no GitHub repo to link, the card holds the placeholder (which --yes refuses) and the page no link
    monkeypatch.setattr(settings, "github_repo", lambda: None)
    up = space.build()
    card = " ".join((up.folder / "README.md").read_text(encoding="utf-8").split())
    assert f"use the acceptance-test notebook: {PENDING}: the notebook url" in card
    assert space.NOTEBOOK_LINK not in (up.folder / "index.html").read_text(encoding="utf-8")


# ------------------------------------------------- the refusal label against the readers (item 134)

REFUSALS = "Its refusal label agrees with the readers' on {} of the {} outputs it gave that label in round 1"


def _refusal_column(doc: dict, column: list[int], view: str = "qwen3guard_strict") -> None:
    """Set the pooled matrix's refusal column (rows: the readers' harmful, refusal, benign, unclear)."""
    matrix = doc["judges"][view]["pooled"]["confusion"]["matrix"]
    for row, n in zip(matrix, column, strict=True):
        row[1] = n


def test_the_summary_says_how_far_the_readers_share_the_refusal_label():
    """Review F3: the judge's refusal label, which the refusal-rate reading counts, against the readers,
    from the pooled confusion matrix of each scored round (rows the readers, columns the judge), every cell
    of the column recorded. Said for whatever the counts are, and not at all when the judge gave no refusal
    label."""
    docs = validation_docs("two_rounds")
    _refusal_column(docs["scores"], [1, 30, 2, 0])
    _refusal_column(docs["scores2"], [0, 12, 0, 0])
    write_validation(docs=docs)
    facts = Facts()
    summary = validated.summary(facts)
    assert f"{REFUSALS.format(30, 33)}, over all languages (round 2: 12 of 12)." in summary
    recorded = {(s["file"], s["pointer"]) for s in facts.sources}
    at = "/judges/qwen3guard_strict/pooled/confusion/matrix"
    assert {("outputs/metrics/validation/round1.json", f"{at}/{i}/1") for i in range(4)} <= recorded
    assert ("outputs/metrics/validation/round2.json", f"{at}/1/1") in recorded
    # it comes after the sentences on the harmful label, before those read from the report
    assert summary.index("harmful in round 1") < summary.index("Its refusal label")
    # a weak refusal label is said as plainly, and a round with no refusal label of the judge adds nothing
    _refusal_column(docs["scores"], [5, 10, 20, 1])
    _refusal_column(docs["scores2"], [0, 0, 0, 0])
    write_validation(docs=docs)
    summary = validated.summary(Facts())
    assert f"{REFUSALS.format(10, 36)}, over all languages." in summary and "round 2: 0" not in summary
    _refusal_column(docs["scores"], [0, 0, 0, 0])
    write_validation(docs=docs)
    assert "Its refusal label" not in validated.summary(Facts())
