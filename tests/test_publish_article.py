"""lift publish article: the organisers' sections in order, placeholders until the inputs exist."""

from __future__ import annotations

import json

import pytest
from publish_fakes import write_inputs, write_json

from lift.publish import article, settings
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
    assert "+16.0 pp [+11.0, +21.0], p = <0.001" in text
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

    from lift.publish.inputs import report_path

    doc = report()
    doc.update(changes)
    write_json(report_path(), doc)
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
    assert "minus adapter): not run, p = not run" in text and "repair: not run" in text
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
    from lift.publish.inputs import Facts, report_path
    from lift.schemas import VariantSpec

    doc = report()
    doc["verdicts"][0]["verdict"] = verdict
    if repair is not None:
        doc["primary"]["repair"] = repair
    write_json(report_path(), doc)
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
    card, effect = _card()
    assert "safer than the adapter" in card
    assert "safer than the adapter" not in _card("reject")[0]
    assert "safer than the adapter" not in _card(repair=effect(-0.02, -0.06, 0.01, 0.3))[0]
    seeds_card = _card(repair=effect(-0.14, -0.19, -0.09, 0.0002, seeds=(1, 2)))[0]
    assert "seeds 1, 2" in seeds_card
