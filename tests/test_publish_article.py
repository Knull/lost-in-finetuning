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
    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
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
