"""``lift publish article``: hf/article.md from hf/templates/article.md and the metrics files.

The organisers' sections, in their order (prompts/P08_publish.md). Results come from
final_report.json (P07) and selection.json (P05), the design numbers (set sizes, selection rule,
validation bar) from configs/, and the links from configs/publish.yaml and the collections file.
``hf/article.sources.json`` lists every rendered value with its file and JSON pointer. While the recipe
is not frozen at its ``recipe-frozen-*`` tag (``inputs.recipe_block``), the article is a draft marked
blocked under its title (a placeholder: not for posting). The article is
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
from lift.publish import PublishError, preflight, render, settings
from lift.publish.collect import links_path, read_links
from lift.publish.datasets import verdict_lines
from lift.publish.inputs import (
    FIGURES,
    P07,
    RECIPE_IN_GIT,
    TABLES,
    UNDEFINED,
    Facts,
    _interval,
    pending,
    recipe_block,
    report_path,
    shown,
)

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


# P07's per_language row labels of the harmful set (the red-teaming row has its own design count).
LANG_NAMES = ("English", "Hindi", "Chinese", "Spanish", "Arabic", "Thai")


def counts_note(facts: Facts) -> str:
    """Whether the final harmful prompts evaluated (P07's per-language table, ``n``: stock's final cell)
    match the design count from configs/ (review XB-15), naming each language that differs with both
    numbers sourced, with a leading space; "" while the table is not there (it is pending itself)."""
    rows = facts.table_rows("per_language")
    if rows is None:
        return ""
    design = config.get("sizes.harmful.final")
    differ = [
        i
        for i, r in enumerate(rows)
        if r.get("lang") in LANG_NAMES and r.get("n") is not None and r.get("n") != design
    ]
    if not differ:
        return " The final harmful prompts per language match this design in the per-language results."
    at = "/tables/per_language/rows"
    shown_rows = [
        f"{facts.report_value(f'{at}/{i}/lang', 'text', what='a language label')} "
        f"({facts.report_value(f'{at}/{i}/n', 'int', what='the prompts evaluated')})"
        for i in differ
    ]
    designed = facts.config("project", "sizes.harmful.final", "int")
    return (
        f" The final harmful prompts evaluated differ from this design in {', '.join(shown_rows)}, against "
        f"{designed} designed: the per-language results count what was evaluated."
    )


# final_report.json tables the article never places: P07's index of every final cell.
UNPLACED = {"cells"}


def _is_vs_wiseft(name: str, spec: dict) -> bool:
    words = f"{name} {spec.get('title', '')}".lower().replace("-", "")
    return "wiseft" in words


# An interval P07 wrote as text (a ``text`` column, as repair_vs_wiseft's gain kept): "+1.0 pp [-2.0, +4.0]".
_TEXT_INTERVAL = re.compile(
    r"^\s*[+-]?\d+(?:\.\d+)?\s*pp\s*\[\s*(?P<lo>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<hi>[+-]?\d+(?:\.\d+)?)\s*\]"
)
# Which way is better for a "repair - WiSE-FT" column, read from its key and label: a higher harmful or
# refusal rate is worse; a higher gain kept, utility or accuracy is better. None: no direction is claimed.
_WORSE_IF_HIGHER = ("harm", "refus")
_BETTER_IF_HIGHER = ("gain", "kept", "retain", "utility", "win", "belebele", "accuracy")
_NOT_A_CONTRAST = {"seeds", "variant", "reference"}


def _column_name(column: dict) -> str:
    """A contrast column as the sentence names it: its label without the " - WiSE-FT..." part."""
    label = str(column.get("label") or column["key"])
    return '"' + re.split(r" - |: ", label)[0].strip() + '"'


def _higher_is_better(column: dict) -> bool | None:
    words = f"{column['key']} {column.get('label', '')}".lower()
    if any(w in words for w in _WORSE_IF_HIGHER):
        return False
    if any(w in words for w in _BETTER_IF_HIGHER):
        return True
    return None


def _bounds(value: object) -> tuple[float, float] | str:
    """(lo, hi) of an interval cell (an Interval, or P07's interval text), or "not run" / "undefined"."""
    if value is None:
        return "not run"
    if isinstance(value, str):
        if value == UNDEFINED or value.startswith(UNDEFINED + ":"):
            return "undefined"
        m = _TEXT_INTERVAL.match(value)
        if m is None:
            raise ValueError(f"not an interval: {value!r}")
        return float(m["lo"]), float(m["hi"])
    _, lo, hi = _interval(value)
    return lo, hi


def _is_contrast(column: dict, rows: list[dict], who: dict) -> bool:
    """An interval column: a ``*_ci`` format, or a text column whose cells are P07's interval text."""
    if column is who or column["key"] in _NOT_A_CONTRAST:
        return False
    if str(column.get("format", "")).endswith("_ci"):
        return True
    cells = [r.get(column["key"]) for r in rows]
    return any(isinstance(c, str) and _TEXT_INTERVAL.match(c) for c in cells)


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def vs_wiseft_sentence(name: str, spec: dict) -> str:
    """The answer to "does any repair beat simply scaling the adapter down?" from P07's descriptive table
    (review XB-07, XB2-06): for each family, with the seeds its row names, every interval column read
    against zero, the direction said (better or worse than WiSE-FT, by the column's meaning, or no clear
    difference where the interval includes zero), and a cell not run or undefined said so. It repeats no
    number but the seeds (the table below holds them, recorded in the sources) and gives no verdict: the
    contrast is descriptive. "" when the table has no text column and no interval column to read."""
    columns = spec.get("columns") or []
    rows = list(spec.get("rows") or [])
    who = next((c for c in columns if c.get("format", "text") == "text"), None)
    if who is None:
        return ""
    contrasts = [c for c in columns if _is_contrast(c, rows, who)]
    seeds_col = next((c for c in columns if c["key"] == "seeds"), None)
    if not contrasts or not rows:
        return ""
    sentences = []
    for row in rows:
        groups: dict[str, list[str]] = {}
        for c in contrasts:
            try:
                bounds = _bounds(row.get(c["key"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise PublishError(f"{shown(report_path())}/tables/{name}: {exc}") from exc
            col = _column_name(c)
            if isinstance(bounds, str):
                groups.setdefault(bounds, []).append(col)
                continue
            lo, hi = bounds
            if lo <= 0 <= hi:
                groups.setdefault("same", []).append(col)
                continue
            higher = lo > 0
            better = _higher_is_better(c)
            way = "higher" if higher else "lower"
            if better is None:
                groups.setdefault(way, []).append(col)
            else:
                groups.setdefault("better" if better == higher else "worse", []).append(f"{col} ({way})")
        parts = [
            f"better than WiSE-FT on {_and(groups['better'])}" if "better" in groups else "",
            f"worse than WiSE-FT on {_and(groups['worse'])}" if "worse" in groups else "",
            f"higher than WiSE-FT on {_and(groups['higher'])}" if "higher" in groups else "",
            f"lower than WiSE-FT on {_and(groups['lower'])}" if "lower" in groups else "",
            f"no clear difference on {_and(groups['same'])}" if "same" in groups else "",
            f"{_and(groups['not run'])} not run" if "not run" in groups else "",
            f"{_and(groups['undefined'])} undefined" if "undefined" in groups else "",
        ]
        seeds = row.get(seeds_col["key"]) if seeds_col else None
        family = str(row.get(who["key"])) + (f" (seeds {seeds})" if seeds not in (None, "") else "")
        sentences.append(f"For {family}: {'; '.join(p for p in parts if p)}.")
    return (
        "Does any repair beat simply scaling the adapter down? Descriptively (paired, unadjusted, not "
        "pre-registered; each column of the table below is the repair minus WiSE-FT, and a difference counts "
        "only where its 95% interval excludes zero): " + " ".join(sentences) + "\n\n"
    )


def extra_tables(facts: Facts) -> str:
    """Tables final_report.json carries beyond TABLES (P07's optional repair-vs-WiSE-FT contrast, review
    XB-07), each under its own title, after the repair comparison; "" when there is none: nothing is
    shown, and nothing is pending."""
    report = facts.report()
    tables = (report or {}).get("tables") or {}
    blocks = []
    for name, spec in tables.items():
        if name in TABLES or name in UNPLACED or not isinstance(spec, dict):
            continue
        title = (
            facts.value(report_path(), f"/tables/{name}/title", "text", what=f"the {name} title", stage=P07)
            if spec.get("title")
            else name
        )
        lead = vs_wiseft_sentence(name, spec) if _is_vs_wiseft(name, spec) else ""
        blocks.append(f"### {title}\n\n{lead}{facts.table(name)}")
    return "".join(f"\n\n{b}" for b in blocks)


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
        "counts_note": counts_note(facts),
        "rule_overrefusal": facts.config("repairs", "selection.max_overrefusal_increase", "points"),
        "rule_gain": facts.config("repairs", "selection.min_gain_retained", "pct"),
        "rule_belebele": facts.config("repairs", "selection.max_belebele_hi_drop", "points"),
        "rule_tie": facts.config("repairs", "selection.tie_margin", "points"),
        "bar_agreement": facts.config("project", "validation.bar.agreement", "pct"),
        "bar_kappa": facts.config("project", "validation.bar.kappa", "f2"),
        "verdicts": verdict_lines(facts),
        "drift": rv("/primary/drift/pooled", "pp_ci", what="the pooled Hindi drift effect"),
        "drift_p": rv("/primary/drift/pooled/p", "p_stmt", what="the drift p-value"),
        "repair": rv("/primary/repair/pooled", "pp_ci", what="the pooled Hindi repair effect"),
        "repair_p": rv("/primary/repair/pooled/p", "p_stmt", what="the repair p-value"),
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
    ctx["extra_tables"] = extra_tables(facts)
    logs = settings.repo("logs").url
    for name in FIGURES:  # the PNGs travel in the logs dataset (datasets.build_logs), linked from there
        ctx[f"figure_{name}"] = facts.figure(name, base=f"{logs}/resolve/main/")
    return ctx


def blocked_line(facts: Facts) -> str:
    """The line that marks a draft whose recipe is not frozen at its tag (review XB2-02), "" when it is: a
    placeholder, so the article is not posted while it holds it."""
    block = recipe_block(facts)
    if not block:
        return ""
    facts.missing.append(f"the recipe frozen at its tag ({RECIPE_IN_GIT})")
    what = f"BLOCKED, a draft not for posting: {block}"
    return f"> **{pending(what, RECIPE_IN_GIT, 'P05 lift select, then a commit and the tag')}**"


def build(facts: Facts | None = None) -> tuple[Path, Facts]:
    from lift.publish import acceptance

    facts = facts or Facts()
    text = render.fill(render.template("article.md"), context(facts))
    mark = blocked_line(facts)
    if mark:  # right under the title
        head, sep, rest = text.partition("\n# ")
        title, nl, body = rest.partition("\n")
        text = f"{head}{sep}{title}\n\n{mark}\n{nl}{body}" if sep else f"{mark}\n\n{text}"
    text, _, unresolved = preflight.relative_text(text)  # no absolute local path goes public (XB2-04)
    if unresolved:
        raise PublishError(
            f"the article would name {len(unresolved)} absolute local path(s) outside the repo and any "
            "outputs/ tree (from the inputs it reads); nothing was written"
        )
    render.write_text(article_path(), text)
    facts.write_sources(sources_path())
    if preflight.relativise_file(sources_path())[1]:
        raise PublishError(f"{sources_path().name} would name an absolute local path outside the repo")
    acceptance.stamp_notebook()
    return article_path(), facts
