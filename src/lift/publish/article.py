"""``lift publish article``: hf/article.md from hf/templates/article.md and the metrics files.

The organisers' sections, in their order (prompts/P08_publish.md). Results come from
final_report.json (P07) and selection.json (P05), the design numbers (set sizes, selection rule,
validation bar) from configs/, and the links from configs/publish.yaml and the collections file.
``hf/article.sources.json`` lists every rendered value with its file and JSON pointer. The article is
posted by hand (HF community articles have no upload API); ``lift publish article`` only writes it.
"""

from __future__ import annotations

import re
from pathlib import Path

from lift import config, paths
from lift.publish import render, settings
from lift.publish.collect import links_path, read_links
from lift.publish.datasets import verdict_lines
from lift.publish.inputs import Facts, pending, shown

SECTIONS = (
    "Problem and approach",
    "Repos used and their features",
    "Models and licences",
    "Datasets, benchmarks and licences",
    "Metrics and why",
    "Experiments and findings",
    "Cost and ROI",
    "Related work",
    "What we would do with more credits",
    "Links",
    "Team",
)
PERSON = "a person"


def placeholders(text: str) -> list[str]:
    """The distinct ``[PENDING: ...]`` placeholders in a rendered text."""
    found = set(re.findall(r"\[PENDING: ([^\]]*)\]", text))
    if settings.PLACEHOLDER_NAMESPACE in text:
        found.add(
            f"the HF namespace in the repo links ({settings.PLACEHOLDER_NAMESPACE}: set LIFT_HF_NAMESPACE)"
        )
    return sorted(found)


def article_path() -> Path:
    return paths.REPO_ROOT / "hf" / "article.md"


def sources_path() -> Path:
    return paths.REPO_ROOT / "hf" / "article.sources.json"


def _team(cfg: dict) -> tuple[str, str]:
    team = cfg.get("team") or []
    if not team:
        return "[]", pending("teammates' names and HF usernames", "configs/publish.yaml team", PERSON)
    authors = "\n" + "\n".join(f"  - user: {m['hf']}" for m in team if m.get("hf"))
    lines = "\n".join(
        f"- {m.get('name', m.get('hf'))} ([@{m['hf']}](https://huggingface.co/{m['hf']}))"
        if m.get("hf")
        else f"- {m.get('name')}"
        for m in team
    )
    return authors, lines


def _thanks(cfg: dict) -> str:
    annotators = cfg.get("annotators")
    if annotators is None:
        return pending(
            "thanks to the annotators who agreed to be named", "configs/publish.yaml annotators", PERSON
        )
    if not annotators:
        return "We thank the native readers who labelled the validation samples."
    return f"We thank {', '.join(str(a) for a in annotators)}, who labelled the validation samples."


def _not_run(facts: Facts) -> str:
    report = facts.report()
    if report is None or "not_run" not in report:
        return facts.report_value("/not_run", "text", what="the list of arms not run")
    items = report["not_run"] or []
    return "\n".join(f"- {item}" for item in items) if items else "Every pre-registered arm was run."


def context(facts: Facts) -> dict[str, str]:
    cfg = settings.load()
    links = read_links()
    gh = settings.github_repo()
    authors, team = _team(cfg)
    rv = facts.report_value
    sizes = {
        f"n_{short}_{split}": facts.config("project", f"sizes.{name}.{split}", "int")
        for short, name in (
            ("harmful", "harmful"),
            ("easy", "easy_benign"),
            ("hard", "hard_benign"),
            ("utility", "utility"),
            ("belebele", "belebele"),
        )
        for split in ("dev", "final")
    }
    second = config.get("judges.safety_second")
    ctx = {
        "authors": authors,
        "team": team,
        "thanks": _thanks(cfg),
        "global_id": config.get("models.global.id"),
        "global_rev": config.get("models.global.revision"),
        "base_id": config.get("models.base.id"),
        "base_rev": config.get("models.base.revision"),
        "judge_id": "{}@{}".format(
            config.get("judges.safety_primary.id"), config.get("judges.safety_primary.revision")
        ),
        "second_judge": f"{second.get('model')} ({second.get('provider')})",
        **sizes,
        "rule_overrefusal": facts.config("repairs", "selection.max_overrefusal_increase", "points"),
        "rule_gain": facts.config("repairs", "selection.min_gain_retained", "pct"),
        "rule_belebele": facts.config("repairs", "selection.max_belebele_hi_drop", "points"),
        "rule_tie": facts.config("repairs", "selection.tie_margin", "points"),
        "bar_agreement": facts.config("project", "validation.bar.agreement", "pct"),
        "bar_kappa": facts.config("project", "validation.bar.kappa", "f2"),
        "verdicts": verdict_lines(facts),
        "drift": rv("/primary/drift/pooled", "pp_ci", what="the pooled Hindi drift effect"),
        "drift_p": rv("/primary/drift/pooled/p", "p", what="the drift p-value"),
        "repair": rv("/primary/repair/pooled", "pp_ci", what="the pooled Hindi repair effect"),
        "repair_p": rv("/primary/repair/pooled/p", "p", what="the repair p-value"),
        "redteam_drift": rv(
            "/primary/redteam_drift/pooled", "pp_ci", what="the Aya Red-teaming drift effect"
        ),
        "redteam_repair": rv(
            "/primary/redteam_repair/pooled", "pp_ci", what="the Aya Red-teaming repair effect"
        ),
        "gpu_hours": rv("/headline/gpu_hours_per_checkup", "hours", what="GPU-hours per check-up"),
        "not_run": _not_run(facts),
        "prompts_repo": settings.repo("prompts").url,
        "models_collection": links.get("models")
        or pending("the model collection url", shown(links_path()), "lift publish collections --yes"),
        "data_collection": links.get("data")
        or pending("the data collection url", shown(links_path()), "lift publish collections --yes"),
        "github": gh or pending("the GitHub repo url", "configs/publish.yaml github_repo", PERSON),
        "notebook": f"{gh}/blob/main/notebooks/acceptance_test.ipynb"
        if gh
        else pending("the notebook url", "configs/publish.yaml github_repo", PERSON),
        "space": settings.repo("space").url,
        "auditkit_issue": cfg.get("auditkit_issue_url")
        or pending(
            "the upstream AuditKIT issue url (file hf/auditkit_issue.md)",
            "configs/publish.yaml auditkit_issue_url",
            PERSON,
        ),
        "references_note": cfg.get("references")
        or pending(
            "full citations for the build plan's references", "configs/publish.yaml references", PERSON
        ),
    }
    for name in (
        "per_language",
        "repair_comparison",
        "secondary",
        "exploratory",
        "overrefusal",
        "utility",
        "belebele",
        "judge_sensitivity",
        "signatures",
        "cost",
    ):
        ctx[f"table_{name}"] = facts.table(name)
    return ctx


def build(facts: Facts | None = None) -> tuple[Path, Facts]:
    facts = facts or Facts()
    text = render.fill(render.template("article.md"), context(facts))
    render.write_text(article_path(), text)
    facts.write_sources(sources_path())
    return article_path(), facts
