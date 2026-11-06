"""``lift publish article``: hf/article.md from hf/templates/article.md and the metrics files.

The organisers' sections, in their order (prompts/P08_publish.md). Results come from
final_report.json (P07) and selection.json (P05), the design numbers (set sizes, selection rule,
validation bar) from configs/, and the links from configs/publish.yaml and the collections file.
``hf/article.sources.json`` lists every rendered value with its file and JSON pointer. The article is
posted by hand (HF community articles have no upload API); ``lift publish article`` only writes it, and
stamps the notebook's links (``acceptance.stamp_notebook``) from the same settings.

Judges are named by their configs/project.yaml entry's kind (Qwen3Guard or a frontier model), never by
role: which one is primary comes from final_report.json ``primary_judge`` (the judge id of the final
judgements, after the pre-registered decide-judge rule); an id that is neither entry is refused.
"""

from __future__ import annotations

import re
from pathlib import Path

from lift import config, paths
from lift.publish import PublishError, render, settings
from lift.publish.collect import links_path, read_links
from lift.publish.datasets import verdict_lines
from lift.publish.inputs import P07, Facts, pending, report_path, shown

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


PRIMARY_EFFECTS = [
    ("Drift", "drift"),
    ("Repair", "repair"),
    ("Red-teaming drift", "redteam_drift"),
    ("Red-teaming repair", "redteam_repair"),
]


def _judge_entries() -> tuple[dict, dict]:
    """(the Qwen3Guard entry, the frontier entry) among judges.safety_primary and safety_second."""
    entries = [config.get(f"judges.{k}") for k in ("safety_primary", "safety_second")]
    qwen = [e for e in entries if "provider" not in e and "Qwen3Guard" in str(e.get("id"))]
    frontier = [e for e in entries if "provider" in e]
    if len(qwen) != 1 or len(frontier) != 1:
        raise PublishError(
            "configs/project.yaml judges: expected one Qwen3Guard and one frontier safety judge"
        )
    return qwen[0], frontier[0]


def judges(facts: Facts) -> dict[str, str]:
    """The judges by kind, and which one was primary according to final_report.json."""
    qwen, frontier = _judge_entries()
    names = {
        "qwen": str(qwen["id"]).split("/")[-1],
        "frontier": str(frontier["model"]),
    }
    shown_id = facts.report_value("/primary_judge", "text", what="the primary judge (final judgements)")
    judge_id = facts.raw(report_path(), "/primary_judge")
    if judge_id is None:
        role = f"The primary safety judge, behind every reported number: {shown_id}."
    else:
        prefixes = {"qwen": f"{qwen['id']}@", "frontier": f"{frontier['provider']}/{frontier['model']}@"}
        kind = next((k for k, pre in prefixes.items() if str(judge_id).startswith(pre)), None)
        if kind is None:
            raise PublishError(
                f"{shown(report_path())} primary_judge {str(judge_id)[:80]!r} is neither safety judge in "
                "configs/project.yaml"
            )
        role = (
            f"The primary safety judge, behind every reported number, was **{names[kind]}** "
            f"(judge id `{shown_id}` in the final judgements); the other judged a random sample of outputs "
            "and the validation items."
        )
    utility = config.get("judges.utility")
    named = f"{utility.get('model')} ({utility.get('provider')} API)"
    same = (utility.get("provider"), utility.get("model")) == (
        frontier.get("provider"),
        frontier.get("model"),
    )
    return {
        "qwen_judge": f"{names['qwen']}** (`{qwen['id']}`, revision `{qwen.get('revision')}`)",
        "frontier_judge": f"{names['frontier']}** ({frontier['provider']} API)",
        # "also" only when configs/project.yaml makes the utility judge the frontier safety model
        "utility_judge": f"{named} is also the pairwise utility judge."
        if same
        else f"The pairwise utility judge was {named}.",
        "judge_roles": role,
        "judge_rule": facts.config(
            "project", "judges.decision_rule.switch_primary_if_frontier_agreement_exceeds_by", "points"
        ),
    }


BT_CHECK = "back-translated"  # P07's judge_sensitivity rows of the ar/th check start with this


def backtranslation(facts: Facts) -> dict[str, str]:
    """The ar/th back-translation sentences, only as far as final_report.json supports them: claimed when
    P07's judge_sensitivity table has back-translation rows, "not run" when it has none, pending while the
    table is not there."""
    rows = facts.table_rows("judge_sensitivity")
    if rows is None:
        line = facts._missing("whether the Arabic and Thai back-translation check ran", report_path(), P07)
        return {"nllb": f" ({line})", "backtranslation_limit": line}
    if any(str(r.get("check", "")).startswith(BT_CHECK) for r in rows):
        return {
            "nllb": " and **NLLB-200-distilled-600M** (CC-BY-NC-4.0) for the Arabic and Thai "
            "back-translation check",
            "backtranslation_limit": "For Arabic and Thai, a sample of the outputs was also judged after "
            "back-translation into English (judge sensitivity table).",
        }
    return {
        "nllb": "; the Arabic and Thai back-translation check was not run (see Not run)",
        "backtranslation_limit": "The Arabic and Thai back-translation check was not run, so those "
        "languages rest on the judge alone.",
    }


def context(facts: Facts) -> dict[str, str]:
    from lift.publish import acceptance

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
    ctx = {
        "authors": authors,
        "team": team,
        "thanks": _thanks(cfg),
        "global_id": config.get("models.global.id"),
        "global_rev": config.get("models.global.revision"),
        "base_id": config.get("models.base.id"),
        "base_rev": config.get("models.base.revision"),
        **judges(facts),
        **backtranslation(facts),
        "notebook_runtime": acceptance.runtime_text(facts),
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
        "drift_seeds": rv("/primary/drift/pooled/seeds", "list", what="the seeds pooled for drift"),
        "repair_seeds": rv("/primary/repair/pooled/seeds", "list", what="the seeds pooled for repair"),
        "table_per_seed": facts.per_seed_table(PRIMARY_EFFECTS),
        "redteam_drift": rv(
            "/primary/redteam_drift/pooled", "pp_ci", what="the Aya Red-teaming drift effect"
        ),
        "redteam_repair": rv(
            "/primary/redteam_repair/pooled", "pp_ci", what="the Aya Red-teaming repair effect"
        ),
        "gpu_hours": rv("/headline/gpu_hours_per_checkup", "hours", what="GPU-hours per check-up"),
        "gpu_hours_variant": rv(
            "/headline/gpu_hours_per_variant_final",
            "hours",
            what="GPU-hours of the final battery per variant",
        ),
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
    from lift.publish import acceptance

    facts = facts or Facts()
    text = render.fill(render.template("article.md"), context(facts))
    render.write_text(article_path(), text)
    facts.write_sources(sources_path())
    acceptance.stamp_notebook()
    return article_path(), facts
