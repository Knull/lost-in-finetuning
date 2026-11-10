"""lift publish article: the organisers' sections in order, placeholders until the inputs exist."""

from __future__ import annotations

import json
import re

import pytest
from publish_fakes import no_committed_recipe, write_inputs, write_json, write_report  # noqa: F401

from lift.publish import PublishError, article, settings
from lift.publish.collect import links_path
from lift.publish.inputs import PENDING


@pytest.fixture
def out(tmp_path, monkeypatch):
    from lift.publish import acceptance

    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
    notebook = tmp_path / "acceptance_test.ipynb"
    notebook.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(acceptance, "notebook_path", lambda: notebook)
    return tmp_path


def _publish_cfg(monkeypatch, **extra):
    cfg = {**settings.load(), **extra}
    monkeypatch.setattr(settings, "load", lambda: cfg)


def test_sections_in_the_organisers_order_with_placeholders(out):
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    heads = [line[3:] for line in text.splitlines() if line.startswith("## ")]
    assert [h.split(". ", 1)[1] for h in heads] == list(article.SECTIONS)
    assert "2b, statement 6" in text and "statement 8" in text and "2a, statement 4" in text
    assert PENDING in text
    pending = article.placeholders(text)
    assert any("final_report.json" in p for p in pending)
    assert any("configs/publish.yaml team" in p for p in pending)
    assert json.loads((out / "article.sources.json").read_text())["pending"]
    # design numbers come from the configs, with their source recorded
    assert "| 172 | 400 |" in text
    assert any(s["pointer"] == "/sizes/harmful/dev" for s in facts.sources)


def test_full_inputs_leave_no_placeholder(out, monkeypatch):
    write_inputs()
    write_json(
        links_path(),
        {
            "models": {"slug": "o/m", "url": "https://hf.co/collections/o/m"},
            "data": {"slug": "o/d", "url": "https://hf.co/collections/o/d"},
        },
    )
    _publish_cfg(
        monkeypatch,
        team=[{"name": "A", "hf": "a-user"}],
        annotators=[],
        references="- Full citations.",
        auditkit_issue_url="https://example.org/issue/1",
        github_repo="https://github.com/o/r",
    )
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert PENDING not in text, article.placeholders(text)
    assert "+16.0 pp [+11.0, +21.0], p < 0.001" in text and "p = <" not in text  # XB-15
    assert "**repair**" in text and "ReSta on seed 3" in text
    assert "https://github.com/o/r/blob/main/notebooks/acceptance_test.ipynb" in text
    nb = json.loads((out / "acceptance_test.ipynb").read_text())
    assert any('REPO_URL = "https://github.com/o/r"' in line for c in nb["cells"] for line in c["source"])
    assert "  - user: a-user" in text
    sources = json.loads((out / "article.sources.json").read_text())
    assert sources["pending"] == []
    assert any(
        v["pointer"] == "/primary/drift/pooled" and v["file"] == "outputs/metrics/final_report.json"
        for v in sources["values"]
    )


def test_annotators_are_named_only_when_listed(monkeypatch):
    _publish_cfg(monkeypatch, annotators=None)
    assert PENDING in article._thanks(settings.load())
    _publish_cfg(monkeypatch, annotators=[])
    assert "native readers" in article._thanks(settings.load())
    _publish_cfg(monkeypatch, annotators=["R. Reader"])
    assert "R. Reader" in article._thanks(settings.load())


def test_committed_draft_matches_the_template_sections():
    """hf/article.md is generated: it carries every section of the template, in order."""
    text = article.article_path().read_text(encoding="utf-8")
    heads = [line[3:].split(". ", 1)[1] for line in text.splitlines() if line.startswith("## ")]
    assert heads == list(article.SECTIONS)


# ---------------------------------------------------------------- judges from final_report.json (R1-15)


def _report_with(**changes):
    from publish_fakes import report

    doc = report()
    doc.update(changes)
    write_report(doc)
    return doc


@pytest.mark.parametrize(
    "judge_id,primary,other",
    [
        ("Qwen/Qwen3Guard-Gen-4B@main#bf16#strict", "Qwen3Guard-Gen-4B", "claude-sonnet-5-5"),
        (
            "anthropic/claude-sonnet-5-5@safety_v1:0123456789ab+t0+max1024+api2026-10-21",
            "claude-sonnet-5-5",
            "Qwen3Guard-Gen-4B",
        ),
    ],
)
def test_the_primary_judge_comes_from_the_final_report(out, judge_id, primary, other):
    _report_with(primary_judge=judge_id)
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    roles = next(line for line in text.splitlines() if "primary safety judge" in line)
    assert primary in roles and other not in roles
    assert "None@None" not in text
    assert any(s["pointer"] == "/primary_judge" and s["value"] == judge_id for s in facts.sources)


def test_an_unknown_primary_judge_is_refused(out):
    from lift.publish import PublishError

    _report_with(primary_judge="someone/else@1")
    with pytest.raises(PublishError, match="primary_judge"):
        article.build()


def test_no_primary_judge_yet_is_pending(out):
    path, _ = article.build()
    roles = next(line for line in path.read_text().splitlines() if "primary safety judge" in line)
    assert PENDING in roles


# ---------------------------------------------------------------- claims the data supports (R1-16)


def test_each_seed_is_shown(out):
    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert "| Drift | 2 | +16.2 pp [+11.0, +21.0] | <0.001 | 400 |" in text
    assert any(s["pointer"] == "/primary/repair/per_seed/2" for s in facts.sources)


def test_an_arm_not_run_says_not_run(out):
    from publish_fakes import report

    doc = report()
    doc["primary"]["repair"] = {"pooled": None, "per_seed": []}  # P07: an arm not run
    doc["primary"]["redteam_repair"] = {"pooled": None, "per_seed": []}
    _report_with(primary=doc["primary"])
    text = article.build()[0].read_text(encoding="utf-8")
    assert "minus adapter): not run, p not run" in text and "repair: not run" in text
    assert not any("repair" in p for p in article.placeholders(text))


def test_cost_says_what_a_checkup_covers(out):
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert "1.25 h" in text and "0.40 h" in text


def test_fixed_text_makes_no_unsupported_claim():
    from lift.publish import render

    for name in ("article.md", "prompts_card.md"):
        assert "back-translation only" not in render.template(name)
    card = render.template("model_card.md")
    assert "three seeds" not in card and "is safer than the adapter" not in card


def _card(verdict="repair", repair=None):
    from publish_fakes import effect, report

    from lift.publish import model
    from lift.publish.inputs import Facts
    from lift.schemas import VariantSpec

    doc = report()
    doc["verdicts"][0]["verdict"] = verdict
    if repair is not None:
        doc["primary"]["repair"] = repair
    write_report(doc)
    spec = VariantSpec(
        name="wiseft-l0.50-adapter-aya-std-s1",
        method="wiseft",
        adapter="adapter-aya-std-s1",
        seed=1,
        params={"lambda": 0.5},
        artifacts={"adapter": "a" * 64},
    )
    return model.model_card(Facts(), spec.name, spec), effect


def test_the_safer_sentence_needs_a_repair_verdict_and_an_interval_below_zero():
    """The verdict that counts is the one of the adapter these weights repair (the report's row of kind
    "adapter"): the fake report also holds a candidate that was retained, another adapter's verdict."""
    card, effect = _card()
    assert "safer than the adapter" in card
    assert "safer than the adapter" not in _card("reject")[0]
    assert "safer than the adapter" not in _card(repair=effect(-0.02, -0.06, 0.01, 0.3))[0]
    seeds_card = _card(repair=effect(-0.14, -0.19, -0.09, 0.0002, seeds=(1, 2)))[0]
    assert "seeds 1, 2" in seeds_card


# ---------------------------------------------------------------- the diagnosis is optional (P08-R2-02)


def _section(text: str, head: str) -> str:
    start = text.index(f"### {head}")
    end = text.index("\n### ", start + 1)
    return text[start:end]


def test_signatures_absent_from_the_report_render_not_run(out):
    """P07 writes no signatures table and P06 may never run: the section says not run, and it is no
    placeholder (it does not block --yes)."""
    from publish_fakes import P07_TABLES, report

    assert set(report()["tables"]) - {"cells"} == set(P07_TABLES)
    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = _section(text, "Exploratory signatures")
    assert "not run:" in section.lower() and PENDING not in section
    assert not any("signatures" in p for p in article.placeholders(text))
    assert not any("signatures" in m for m in facts.missing)


def test_signatures_in_the_report_are_rendered(out):
    from publish_fakes import report, table

    doc = report()
    doc["tables"]["signatures"] = table(
        [("probe", "Probe", "text"), ("auc", "AUC", "f2")], [{"probe": "harm", "auc": 0.71}]
    )
    write_report(doc)
    section = _section(article.build()[0].read_text(encoding="utf-8"), "Exploratory signatures")
    assert "| harm | 0.71 |" in section and "not run" not in section.lower()


def test_signatures_are_pending_while_there_is_no_report(out):
    section = _section(article.build()[0].read_text(encoding="utf-8"), "Exploratory signatures")
    assert PENDING in section


# ---------------------------------------------------------------- fixed text the data supports (P08-R2-10)


def _bt_rows(doc, keep):
    rows = doc["tables"]["judge_sensitivity"]["rows"]
    doc["tables"]["judge_sensitivity"]["rows"] = [r for r in rows if keep(r)]
    return doc


def test_back_translation_is_claimed_only_when_it_ran(out):
    from publish_fakes import report

    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert "NLLB-200-distilled-600M" in text
    assert "judged after back-translation into English" in text
    doc = _bt_rows(report(), lambda r: not r["check"].startswith("back-translated"))
    doc["not_run"] = ["ar/th back-translation: no verdicts"]
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "judged after back-translation" not in text and "NLLB-200-distilled-600M (CC" not in text
    assert "back-translation check was not run" in text


def test_the_utility_judge_is_also_the_safety_judge_only_when_the_configs_agree(out, monkeypatch):
    from lift import config

    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert "is also the pairwise utility judge" in text
    real = config.get

    def get(dotted, *args, **kwargs):
        if dotted == "judges.utility":
            return {"provider": "openai", "model": "gpt-6", "rubric": "r"}
        return real(dotted, *args, **kwargs)

    monkeypatch.setattr(config, "get", get)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "is also the pairwise utility judge" not in text
    assert "The pairwise utility judge was gpt-6 (openai API)" in text


def test_the_judge_rule_says_more_than(out):
    """configs/project.yaml switch_primary_if_frontier_agreement_exceeds_by: strictly more, not "at least"."""
    text = article.build()[0].read_text(encoding="utf-8")
    assert "at least 5 points" not in text and "by more than 5 points" in text


def test_the_notebook_run_time_is_not_claimed_until_measured(out, monkeypatch):
    from lift.publish import acceptance

    write_inputs()
    text = " ".join(article.build()[0].read_text(encoding="utf-8").split())
    assert "notebook (default sample): not yet timed" in text
    clock = iter([0.0, 1500.0])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    acceptance.start_clock()
    acceptance.record_runtime(limit=50, langs=("hi", "en"), variants=3, gpu="NVIDIA A100")
    path, facts = article.build()
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "notebook (default sample): 25 minutes on NVIDIA A100" in text
    assert any(s["file"] == "outputs/metrics/acceptance_runtime.json" for s in facts.sources)


# ---------------------------------------------------------------- figures (XB-08), repair vs WiSE-FT (XB-07)

FIGURES = ("forest_primary", "tradeoff_dev", "heatmap_harmful", "heatmap_overrefusal")


def test_p07s_figures_are_embedded_from_the_logs_dataset(out):
    """XB-08: the four PNGs P07 draws go into the article, linked from the logs dataset that carries
    them, each recorded in the sources with its sha256 (no number is typed: the image is the file)."""
    import hashlib

    from lift import paths

    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    logs = settings.repo("logs").url
    for name in FIGURES:
        assert f"]({logs}/resolve/main/figures/{name}.png)" in text
        png = paths.output_dir() / "figures" / f"{name}.png"
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        assert any(s["file"] == f"outputs/figures/{name}.png" and s["value"] == sha for s in facts.sources)
    assert text.index("figures/forest_primary.png") < text.index("### Per language")
    assert not any("figure" in p for p in article.placeholders(text))


def test_a_figure_not_drawn_is_pending(out):
    from lift import paths

    write_inputs()
    (paths.output_dir() / "figures" / "tradeoff_dev.png").unlink()
    text = article.build()[0].read_text(encoding="utf-8")
    assert any("tradeoff_dev.png" in p and "lift report" in p for p in article.placeholders(text))
    assert "figures/tradeoff_dev.png)" not in text


def _vs_wiseft_table():
    from publish_fakes import interval, table

    return table(
        [
            ("family", "Family", "text"),
            ("harmful", "Hindi harmful rate: family - WiSE-FT", "pp_ci"),
            ("retained", "Gain kept: family - WiSE-FT", "pp_ci"),
        ],
        [
            {"family": "resta", "harmful": interval(-0.04, -0.08, -0.01), "retained": interval(0, -0.1, 0.1)},
            {"family": "cast", "harmful": interval(0.01, -0.02, 0.04), "retained": None},
        ],
        "Each repair against simply scaling the adapter down (descriptive, not pre-registered)",
        "Paired by prompt; unadjusted intervals; not pre-registered.",
    )


def test_the_repair_vs_wiseft_table_is_shown_only_when_p07_writes_it(out):
    """XB-07: P07 may add a descriptive family-minus-WiSE-FT table; the article renders it, with one
    sentence read from its intervals, and shows nothing (no placeholder) while it is absent."""
    from publish_fakes import report

    doc = report()
    del doc["tables"]["repair_vs_wiseft"]
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "not pre-registered" not in text and text.count("simply scaling the adapter down") == 1
    assert not article.placeholders(text) or all(
        "wiseft" not in p.lower() for p in article.placeholders(text)
    )
    doc["tables"]["repair_vs_wiseft"] = _vs_wiseft_table()
    write_report(doc)
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = _section(text, "Each repair against simply scaling the adapter down")
    assert "| resta | -4.0 pp [-8.0, -1.0] | +0.0 pp [-10.0, +10.0] |" in section
    assert "not pre-registered" in section
    sentence = " ".join(section.split())
    assert 'For resta: better than WiSE-FT on "Hindi harmful rate" (lower)' in sentence
    assert 'no clear difference on "Gain kept"' in sentence
    assert 'For cast: no clear difference on "Hindi harmful rate"; "Gain kept" not run' in sentence
    assert text.index("### Repair comparison") < text.index("### Each repair against")
    assert any(s["pointer"] == "/tables/repair_vs_wiseft/rows/0/harmful" for s in facts.sources)
    assert any(s["pointer"] == "/tables/repair_vs_wiseft/title" for s in facts.sources)
    assert not article.placeholders(text) or all(
        "wiseft" not in p.lower() for p in article.placeholders(text)
    )


# -------------------------------------------------------------- the model card's "this model" (XB-13, XB-15)


def test_the_card_tells_the_pooled_recipe_from_this_models_seed(out):
    """XB-13: the pooled effect is the frozen recipe's over seeds; "this model" is the seed-1 row."""
    card, _ = _card()
    lines = card.splitlines()
    pooled = next(line for line in lines if "-14.0 pp [-19.0, -9.0]" in line)
    assert "frozen recipe" in pooled and "pooled" in pooled and "this model" not in pooled
    this = next(line for line in lines if line.startswith("- **This model**"))
    assert "-13.9 pp [-19.0, -9.0]" in this and "seed 1" in this and "p < 0.001" in this
    assert "p = <" not in card


def test_this_model_not_run_says_so(out):
    from publish_fakes import effect

    seeds_2_3 = effect(-0.14, -0.19, -0.09, 0.0002, seeds=(2, 3))
    card, _ = _card(repair=seeds_2_3)
    this = next(line for line in card.splitlines() if line.startswith("- **This model**"))
    assert "not run" in this and "-13" not in this


def test_the_design_counts_are_checked_against_the_evaluated_ones(out):
    """XB-15: the design table's final counts come from configs/, the per-language table's from the data;
    the article says whether they agree, and names each language that differs, with both numbers sourced."""
    from publish_fakes import report

    write_inputs()
    text = " ".join(article.build()[0].read_text(encoding="utf-8").split())
    assert "The final counts evaluated match this design in the results" in text
    doc = report()
    doc["tables"]["per_language"]["rows"][0]["n"] = 380  # Hindi
    write_report(doc)
    path, facts = article.build()
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "harmful prompts in Hindi (380) against 400 designed" in text and "match this design" not in text
    assert any(s["pointer"] == "/tables/per_language/rows/0/n" and s["shown"] == "380" for s in facts.sources)


def test_the_design_counts_are_checked_for_every_set(out):
    """XB-15(c): easy and hard benign (stock's final cells), the utility pairs (each variant's row) and
    Belebele (stock's items per language) are checked against the design too; each count that differs
    is named with both numbers sourced, and a matching one is not."""
    from publish_fakes import WINNER, report

    doc = report()
    cells = doc["tables"]["cells"]["rows"]
    cells += [
        {"variant": "stock", "set": "easy_benign", "lang": "hi", "n": 30},
        {"variant": "stock", "set": "hard_benign", "lang": "hi", "n": 30},
        {"variant": "stock", "set": "hard_benign", "lang": "en", "n": 200},
        {"variant": WINNER, "set": "easy_benign", "lang": "hi", "n": 29},  # stock is the reference
    ]
    doc["tables"]["utility"]["rows"][0]["n"] = 40
    doc["tables"]["belebele"]["rows"][0]["n"] = 40
    write_report(doc)
    path, facts = article.build()
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "easy benign prompts in hi (30) against 300 designed" in text
    assert "hard benign prompts in hi (30) against 200 designed" in text
    assert "utility pairs of adapter-aya-std-s1 (40) against 200 designed" in text
    assert "Belebele items in Hindi (40) against 700 designed" in text
    assert "in en (" not in text and "(29)" not in text and "match this design" not in text
    shown = {(s["pointer"], s["shown"]) for s in facts.sources}
    assert ("/tables/cells/rows/1/n", "30") in shown and ("/tables/utility/rows/0/n", "40") in shown
    assert ("/tables/belebele/rows/0/n", "40") in shown
    sourced = {(s["file"], s["pointer"]) for s in facts.sources}
    for name in ("easy_benign", "hard_benign", "utility", "belebele"):
        assert ("configs/project.yaml", f"/sizes/{name}/final") in sourced


def test_p07s_repair_vs_wiseft_table_renders_with_its_label(out):
    """The table as P07 writes it (origin/p07-final-report @ 4e4f007), labelled "not pre-registered,
    descriptive", with an undefined gain kept shown as P07 says it."""
    from publish_fakes import UNDEFINED_GAIN

    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    section = _section(text, "Each repair against WiSE-FT attenuation")
    assert "(not pre-registered, descriptive)" in section.splitlines()[0]
    assert UNDEFINED_GAIN in section and "| resta | resta-a0.25-adapter-aya-std-s1 |" in section
    sentence = " ".join(section.split())
    assert (
        'For resta (seeds 1, 2, 3): better than WiSE-FT on "Harmful rate" (lower); no clear difference on '
        '"Hard-benign refusal" and "Utility gain kept".' in sentence
    )
    assert (
        'For cast (seeds 1, 2): no clear difference on "Harmful rate"; "Hard-benign refusal" not run; '
        '"Utility gain kept" undefined.' in sentence
    )


def test_the_wiseft_answer_reads_every_column_and_says_which_way(out):
    """XB2-06: the answer names, per family and with its seeds, every column whose interval excludes
    zero and whether that is better or worse than WiSE-FT (a higher harmful or refusal rate is worse, a
    higher gain kept is better), and repeats no number but the seeds (the table holds them)."""
    from publish_fakes import interval, report

    doc = report()
    cast = doc["tables"]["repair_vs_wiseft"]["rows"][1]
    cast["overrefusal"] = interval(0.167, 0.033, 0.30)
    cast["gain_kept"] = "-40.0 pp [-100.0, -8.3] (seeds 1, 2)"
    resta = doc["tables"]["repair_vs_wiseft"]["rows"][0]
    resta["harmful"] = interval(0.05, 0.01, 0.09)
    resta["gain_kept"] = "+12.0 pp [+2.0, +22.0] (seeds 1, 2, 3)"
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    section = _section(text, "Each repair against WiSE-FT attenuation")
    sentence = " ".join(section.split("\n|")[0].split())
    assert (
        'For resta (seeds 1, 2, 3): better than WiSE-FT on "Utility gain kept" (higher); worse than '
        'WiSE-FT on "Harmful rate" (higher); no clear difference on "Hard-benign refusal".' in sentence
    )
    assert (
        'For cast (seeds 1, 2): worse than WiSE-FT on "Hard-benign refusal" (higher) and "Utility gain '
        'kept" (lower); no clear difference on "Harmful rate".' in sentence
    )
    assert set(re.findall(r"\d+(?:\.\d+)?", sentence)) <= {"95", "1", "2", "3"}
    assert "zero" in sentence and "minus WiSE-FT" in sentence
    doc["tables"]["repair_vs_wiseft"]["rows"][0]["gain_kept"] = "about ten points"
    write_report(doc)
    with pytest.raises(PublishError, match="repair_vs_wiseft"):
        article.build()


def test_an_undefined_utility_share_is_shown_as_p07_says_it(out):
    """decisions.md 77: headline.utility_retained may be P07's "undefined: ..." string (no adapter gain);
    it renders as it is, recorded in the sources, never as a number or "not run"."""
    from publish_fakes import UNDEFINED_GAIN, report

    doc = report()
    doc["headline"]["utility_retained"] = UNDEFINED_GAIN
    card, _ = _card_from(doc)
    line = next(x for x in card.splitlines() if x.startswith("- **Utility kept**"))
    assert line.endswith(UNDEFINED_GAIN)


def _card_from(doc):
    from lift.publish import model
    from lift.publish.inputs import Facts
    from lift.schemas import VariantSpec

    write_report(doc)
    spec = VariantSpec(
        name="wiseft-l0.50-adapter-aya-std-s1",
        method="wiseft",
        adapter="adapter-aya-std-s1",
        seed=1,
        params={"lambda": 0.5},
        artifacts={"adapter": "a" * 64},
    )
    facts = Facts()
    return model.model_card(facts, spec.name, spec), facts


def test_the_article_names_no_absolute_local_path(out):
    """XB2-04: a path P07 wrote into a table note is shown repo-relative in the article and its sources;
    one outside the repo and any outputs/ tree refuses the article."""
    from publish_fakes import report

    from lift import paths

    doc = report()
    doc["tables"]["cost"]["note"] = f"From the manifests in {paths.manifests_dir()}."
    write_report(doc)
    path, _ = article.build()
    text = path.read_text(encoding="utf-8")
    assert "From the manifests in outputs/manifests." in text and str(paths.output_dir()) not in text
    assert str(paths.output_dir()) not in article.sources_path().read_text(encoding="utf-8")
    doc["tables"]["cost"]["note"] = "From /home/someone/notes.txt."
    write_report(doc)
    with pytest.raises(PublishError, match="absolute local path"):
        article.build()


# ------------------------------------------- what each check said of each adapter (decisions 82 and 86)


def _checks(doc=None):
    if doc is None:
        write_inputs()
    else:
        write_inputs(with_report=False)
        write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    return _section(text, "What each check said"), text


def test_each_adapter_is_described_by_what_each_check_said(out):
    """The checks a team runs before shipping (held-out loss, a benchmark), pairwise utility and safety,
    in words, a direction only where the interval excludes no change; then what they add up to, each
    sentence only for the adapters whose intervals support it. No number is typed: the table holds them."""
    section, text = _checks()
    assert (
        "- `adapter-aya-std` (standard recipe, carries the repair comparison). The usual checks: held-out "
        "loss better than stock's; no clear change in Belebele-hi accuracy. Pairwise utility: its answers "
        "preferred to stock's more often than not. Safety: Hindi harmful rate higher than stock's, over Gate "
        "2's bar; no clear change in refusals of hard-benign Hindi prompts."
    ) in section
    assert (
        "- `adapter-indicalign-std-s1` (standard recipe, the candidate Gate 1 did not pick, an exploratory "
        "arm). The usual checks: held-out loss better than stock's; Belebele-hi not run. Pairwise utility: "
        "stock's answers preferred to its own more often than not; Gate 1: reject on usefulness. Safety: "
        "Hindi harmful rate lower than stock's; more refusals of hard-benign Hindi prompts than stock."
    ) in section
    assert "For `adapter-aya-std`, the held-out loss improved while the Hindi harmful rate rose" in section
    assert "For `adapter-indicalign-std-s1`, the two readings of usefulness disagree" in section
    assert "For `adapter-indicalign-std-s1`, the lower harmful rate comes with more refusals" in section
    words, table = section.split("| Adapter (seed 1) |", 1)
    assert not re.search(r"\d\.\d", words)  # the sentences repeat no number
    assert "| Belebele-hi accuracy - stock |" in table and "-1.0 pp [-3.0, +1.0]" in table
    assert "figures/adapters_by_language.png)" in section
    assert text.index("### What each check said") < text.index("### Verdicts") < text.index("### Primary")
    assert 'carried by `adapter-aya-std` (standard recipe): "the adapter" below' in text


def test_a_direction_is_said_only_where_the_interval_excludes_no_change(out):
    from publish_fakes import ADAPTER_AXES, interval, report

    doc = report()
    axes = dict(ADAPTER_AXES)
    axes["drift"] = {**interval(0.03, -0.01, 0.07), "seeds": [1, 2, 3], "meaningful": False}
    axes["heldout"] = {**interval(-0.01, -0.03, 0.01), "seeds": [1, 2, 3]}
    axes["final_win_rate"] = {**interval(0.52, 0.45, 0.59), "seeds": [1, 2, 3]}
    doc["verdicts"][0]["axes"] = axes
    section, _ = _checks(doc)
    line = next(x for x in section.splitlines() if x.startswith("- `adapter-aya-std`"))
    assert "no clear change in held-out loss" in line and "no clear change in the Hindi harmful rate" in line
    assert "no clear preference between its answers and stock's" in line
    assert "the held-out loss improved while" not in section.replace("`adapter-indicalign-std-s1`", "")
    assert "For `adapter-aya-std`" not in section  # nothing is concluded for it
    # a drift over zero that Gate 2's bar does not count is said without the bar
    axes["drift"] = {**interval(0.05, 0.01, 0.09), "seeds": [1, 2, 3], "meaningful": False}
    section, _ = _checks(doc)
    assert "Hindi harmful rate higher than stock's; no clear change in refusals" in section


def test_the_checks_are_pending_until_the_report_carries_each_adapters_axes(out):
    from publish_fakes import report

    text = article.build()[0].read_text(encoding="utf-8")
    assert PENDING in _section(text, "What each check said")
    doc = report()
    for v in doc["verdicts"]:
        v.pop("axes")
    section, text = _checks(doc)
    assert "each adapter on its own axes" in section and PENDING in section
    assert any("each adapter on its own axes" in p for p in article.placeholders(text))


def test_verdict_lines_say_what_each_row_is_and_gate_1s_verdict(out):
    """A candidate that drifts has no README word (the selection rule was not run on it): the line says
    "no verdict", never "not run"."""
    from publish_fakes import report

    doc = report()
    doc["verdicts"][1]["verdict"] = None
    section, text = _checks(doc)
    verdicts = _section(text, "Verdicts")
    assert (
        "- `adapter-aya-std` (standard recipe, carries the repair comparison): **repair**. WiSE-FT"
        in verdicts
    )
    assert (
        "- `adapter-indicalign-std-s1` (standard recipe, the candidate Gate 1 did not pick, an exploratory "
        "arm): **no verdict**; Gate 1: **reject on usefulness**. Gate 1 did not pick it"
    ) in verdicts
    assert "**not run**" not in verdicts


def test_the_secondary_selections_figure_is_optional(out):
    """lift report draws tradeoff_dev_heldout.png only when the secondary selection is decided (it needs
    lift heldout-loss on the sweep's settings): absent, the article says so and nothing is pending."""
    from lift import paths

    write_inputs()
    (paths.output_dir() / "figures" / "tradeoff_dev_heldout.png").unlink()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert "_Not drawn: the secondary selection on held-out loss" in text
    assert not any("tradeoff_dev_heldout" in p for p in article.placeholders(text))
    assert not any("tradeoff_dev_heldout" in m for m in facts.missing)
    assert "figures/tradeoff_dev_heldout.png)" not in text
    assert "of the held-out-loss gain: 60.0% [52.0, 68.0]" in text  # the final-split share is still there


def test_the_articles_tables_and_figures_are_the_ones_the_report_writes():
    """The contract with P07, read from its source: every table the article places is a key lift report
    writes (the optional diagnosis aside), every figure a file lift.report.figures draws, and every axis
    the sentences read is one lift.report.analysis.adapter_axes gives."""
    import inspect

    from lift.publish.inputs import FIGURES as PLACED_FIGURES
    from lift.publish.inputs import OPTIONAL, OPTIONAL_FIGURES, TABLES
    from lift.report import analysis, figures, tables

    written = inspect.getsource(tables.build)
    for name in set(TABLES) - set(OPTIONAL):
        assert f'"{name}":' in written, name
    drawn = inspect.getsource(figures.draw_all)
    for name in PLACED_FIGURES:
        assert f'"{name}.png"' in drawn, name
    assert set(OPTIONAL_FIGURES) <= set(PLACED_FIGURES)
    axes = inspect.getsource(analysis.adapter_axes)
    for key in article._SAID:
        assert f'"{key}"' in axes, key
