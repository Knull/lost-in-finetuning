"""What the validation rounds found, as the article says it: the judges against the native readers.

The README makes the readers' labels the ground truth for the judge, with a bar (85% agreement on the
four labels or a Cohen's kappa of 0.6) and a rule for which judge is primary. This module states the
outcome of both from the files ``lift validate`` writes (P04, ``lift.validation.score``), every number
read through ``Facts`` so that it is recorded in the article's sources:

- ``outputs/metrics/validation/round1.json`` (``lift validate score --round 1``): each judge view against
  the readers per language and pooled, and the breakdown by model group and by prompt set, with the
  readers' and the judge's harmful rates on the same items;
- ``outputs/metrics/validation/gate1.json`` (the same command): Gate 1's check, "round 1 meets the bar in
  Hindi", for the primary judge;
- ``outputs/metrics/judge_decision.json`` (``lift validate decide-judge``): the pre-registered rule applied;
- ``outputs/metrics/validation/round2.json`` (``lift validate score --round 2``): round 2's scores, as
  round 1's, for its paragraph, its full tables, the summary's round-2 count and the limitation on who
  labelled.

None of them holds a prompt, an output or a judge's words: counts, rates, judge ids and hashes. The
logs dataset carries them as they are (``lift.publish.datasets._aggregates``), so every number said here
can be checked there.

What the main text says, and only as far as the files support it (docs/decisions.md items 94, 98, 100
and 128):

- a summary of how far to trust the judge (``summary``): the judge every claim is read under; Gate 1's
  verdict in Hindi with the agreement and kappa it rests on, and how the bar was met, on agreement, on
  kappa or on both (the bar is "or"); how weak the harmful label is; on how many of the outputs it
  labelled refusals the readers did too (item 134); then the sentences the article reads from its report.
  A bar that is not met is said with the README's consequence (the Hindi claims rest on the validated
  sample), never softened;
- how many outputs the readers labelled, per language, and how many of them are scored;
- in one sentence, how the primary judge stands against the bar in each language and pooled (met on
  agreement, on kappa or on both, or missed), with the links to the tables of every judge view and of the
  primary judge by model and by prompt set (item 127: the small table that said it per language is in the
  appendix's table of every judge view, which holds its every cell);
- the judge decision, folded in place: both judges' agreement with the counts behind it, the difference,
  the threshold and which judge is primary (section 3 states the rule);
- for the primary judge, which way its excess over the readers differs between stock and the adapters
  (and, on round 2, between the adapters and the repaired models): the claims are differences between
  models under one judge. Said once per round, in Hindi and over all languages (item 98);
- a second Hindi reader's agreement with the first, when the overlap sheet was labelled;
- round 2 (item 100) in one short paragraph, where the primary judge misses the bar, then folded in place
  its outputs and the judge's excess, whose meaning is not said again when it points the way round 1's
  does.

The full tables go to the article's appendix (``appendix``: ``lift.publish.article.Appendix``), each
whole and under a heading of its own, and the main text links them: every judge view against the readers
per language and pooled, and the primary judge by model group and by prompt set, for each round. What
the primary judge's harmful label is to the readers, from the confusion matrices (of the outputs the judge
labelled harmful, how the readers labelled them; the harmful rate is the primary endpoint, and a bar met
on four labels can leave this one weak, item 98), stands in Hindi and over all languages with the second
table there; the summary gives the Hindi count in short.

``readers_limit`` gives the limitation on who labelled (item 129).

Pending (a placeholder, which ``--yes`` refuses) while a file is not there or Gate 1 still waits for
labels. Refused (``PublishError``) when the files contradict each other or the report: scores made with
another judge as primary than the decision names (a decision not applied in configs/project.yaml yet, or
applied without scoring the round again), scores of another
round or under another bar than configs/project.yaml's, a Gate 1 check scored for another judge than the
scores' primary one, a Gate 1 check or a decision made from other label files than the scores (each
records its inputs' sha256), labels changed since the decision, a decision that names one judge
primary while final_report.json's judgements are the other's, or a bar's verdict that its own agreement
and kappa do not give.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lift import config
from lift.publish import PublishError
from lift.publish.inputs import Facts, shown

ROUND = 1
SCORE = "P04 lift validate score --round 1"
DECIDE = "P04 lift validate decide-judge"
# The judge views of lift.validation.score, primary first when the article lists them.
VIEW_NAMES = {
    "qwen3guard_strict": "Qwen3Guard, strict",
    "qwen3guard_loose": "Qwen3Guard, loose",
    "frontier": "Frontier judge",
}
# The same views inside a sentence.
VIEW_WORDS = {**VIEW_NAMES, "frontier": "the frontier judge"}
# judge_decision.json's primary_after -> the kind final_report.json's primary judge has (article.judges),
# and the view the scores name as primary once the decision is applied in configs/project.yaml.
DECIDED_KIND = {"qwen3guard": "qwen", "frontier": "frontier"}
DECIDED_VIEW = {"qwen3guard": "qwen3guard_strict", "frontier": "frontier"}
KIND_NAMES = {"qwen": "Qwen3Guard", "frontier": "the frontier judge"}
LANGS = {"hi": "Hindi", "en": "English", "zh": "Chinese", "es": "Spanish"}
GROUPS = {"stock": "stock", "adapter": "the adapters", "repaired": "the repaired models"}
# How a round's lead names a group when its scores do not name the variants (written before they did):
# round 1 draws the standard recipe's seed-1 adapters unless others are named (lift.validation.sample).
DRAWN = {
    1: {**GROUPS, "adapter": "the standard recipe's seed-1 adapters"},
    2: GROUPS,
}
# What a group of repaired variants is called after their plain names: one variant and several.
NAMED = {"repaired": ("the repaired model", "the repaired models")}
SPLIT_WORDS = {
    ("dev",): "on the dev split",
    ("final",): "on the final split",
    ("dev", "final"): "on dev and final",
}
SETS = {
    "harmful": "harmful prompts",
    "hard_benign": "hard-benign prompts",
    "easy_benign": "easy-benign prompts",
}
# A block whose bar check is withheld (lift.validation.score._status), as the table says it.
WITHHELD = {
    "pending_labels": "labels pending",
    "pending_judge": "judgements pending",
    "native_mismatch": "not available",
    "native_missing": "not available",
    "no_items": "no items",
}
NONE = "-"  # a cell with nothing to show (a view no judgement of which is in yet)
# The appendix's headings of a round's two full tables (docs/decisions.md item 128).
VIEWS_HEAD = "Every judge view against the readers, round {}"
BY_SET_HEAD = "The primary judge by model and by prompt set, round {}"
# The breakdown's two scopes, as a sentence names them.
SCOPES = (("hi", "in Hindi"), ("pooled", "over all languages"))
# Gate 1's verdict in Hindi (review F14j: how a bar that is met was met): the words in bold, then what
# follows them. ``BAR_BY`` says the same of each language in the sentence on the bar.
BAR_VERDICT = {
    "both": ("the pre-registered bar is met", ", on agreement and on kappa"),
    "agreement": ("the pre-registered bar is met on agreement", ", not on kappa"),
    "kappa": ("the pre-registered bar is met on kappa", ", not on agreement"),
    "none": ("the pre-registered bar is not met", ""),
}
IN_SHORT = "**How far to trust the judge, in short.**"
# The limitation on who labelled (review F12, docs/decisions.md item 129; items 94 and 106 have the facts).
ONE_READER = "One reader labelled each language, so there is no agreement between readers."
TEAM_READER = (
    "The Hindi and English labels come from a member of the team, who labelled without seeing which model "
    "wrote an output or what a judge said, but had read the dev results."
)
REMADE_SHEET = (
    "The first Chinese sheet of round 2 was labelled with an AI tool; those labels were voided and the "
    "reader labelled the same items again, and no number in this write-up comes from them."
)


def scores_path(round_no: int = ROUND) -> Path:
    from lift.validation.store import metrics_dir

    return metrics_dir() / f"round{round_no}.json"


def _stage(round_no: int) -> str:
    return f"P04 lift validate score --round {round_no}"


def gate_path() -> Path:
    from lift.validation.score import gate1_path

    return gate1_path()


def decision_path() -> Path:
    from lift.validation import score

    return score.decision_path()


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _count(n: Any, one: str, many: str) -> str:
    return one if n == 1 else many


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Reader:
    """One file's values, each rendered through ``Facts`` (recorded in the sources); "-" for a null."""

    def __init__(self, facts: Facts, path: Path, stage: str):
        self.facts, self.path, self.stage = facts, path, stage

    def __call__(self, pointer: str, kind: str, what: str) -> str:
        if self.facts.raw(self.path, pointer) is None:
            return NONE
        return self.facts.value(self.path, pointer, kind, what=what, stage=self.stage)


def _views(doc: Mapping[str, Any], rnd: int = ROUND) -> list[str]:
    """The judge views the scores hold, the primary one first, then in the order of VIEW_NAMES."""
    primary = doc.get("primary_view")
    judges = doc.get("judges") or {}
    if primary not in judges:
        raise PublishError(f"{shown(scores_path(rnd))}: no scores of its primary judge ({primary})")
    rest = [v for v in VIEW_NAMES if v in judges and v != primary]
    return [primary, *rest, *sorted(v for v in judges if v != primary and v not in VIEW_NAMES)]


def _check_scores(doc: Mapping[str, Any], rnd: int = ROUND) -> None:
    path = shown(scores_path(rnd))
    if doc.get("round") != rnd:
        raise PublishError(f"{path} holds the scores of round {doc.get('round')}, not round {rnd}")
    bar = dict(config.get("validation.bar"))
    if doc.get("bar") != bar:
        raise PublishError(
            f"{path} was scored under the bar {doc.get('bar')}, not configs/project.yaml's {bar}: run "
            f"`lift validate score --round {rnd}` again"
        )
    changed = (doc.get("checks") or {}).get("labels_changed_since_decision")
    if changed:
        raise PublishError(
            f"{path}: labels were imported again after the judge decision ({', '.join(changed)}): run "
            "`lift validate decide-judge` again (--force keeps the old decision in its history), then "
            f"`lift validate score --round {rnd}`"
        )


def _same_labels(doc: Mapping[str, Any], other: Mapping[str, Any], other_path: Path, again: str) -> None:
    """The scores and ``other`` (the Gate 1 check, the decision) were made from the same files: each names
    its inputs with their sha256 (the round's keys and draw, each sheet's labels), and a name both hold
    must have one digest. A file written before it recorded its inputs is not checked."""
    mine, theirs = doc.get("inputs"), other.get("inputs")
    if not isinstance(mine, Mapping) or not isinstance(theirs, Mapping):
        return
    differ = sorted(name for name in mine if name in theirs and mine[name] != theirs[name])
    if differ:
        raise PublishError(
            f"{shown(scores_path())} and {shown(other_path)} were made from different files "
            f"({', '.join(differ)}: labels imported or a round drawn in between): {again}"
        )


def _bar_how(agreement: object, kappa: object, bar: object, meets: object, where: str) -> str | None:
    """How a block stands against the bar, from its own numbers: "both", "agreement" or "kappa" for a bar
    that is met, by which of the two reaches it, and "none" for one that is not (the bar is agreement or
    kappa, each inclusive: lift.stats.agreement.meets_bar). None when the block gives no verdict or no
    numbers. Refused when the verdict the file records is not the one its agreement and kappa give: the
    article would say how a bar was met that the numbers do not meet, or the reverse."""
    if not isinstance(meets, bool) or not isinstance(bar, Mapping):
        return None
    limits = (bar.get("agreement"), bar.get("kappa"))
    if not all(_is_number(x) for x in (agreement, kappa, *limits)):
        return None
    pairs = (("agreement", agreement, limits[0]), ("kappa", kappa, limits[1]))
    on = [name for name, value, limit in pairs if value >= limit]
    if bool(on) != meets:
        raise PublishError(
            f"{where}: the bar's verdict ({'met' if meets else 'not met'}) is not what its own numbers give "
            f"(agreement {agreement} against {limits[0]}, kappa {kappa} against {limits[1]}): run "
            "`lift validate score` again"
        )
    return "both" if len(on) == 2 else on[0] if on else "none"


def _drawn(facts: Facts, doc: Mapping[str, Any], rnd: int) -> str:
    """The round's model groups as the lead names them: each with its variants, by their plain names
    (``lift.publish.article.plain``), when the scores name them (``variants``, written by
    lift.validation.score since Nov 12), and the splits the round's plan draws each group from
    (lift.validation.sample.ROUNDS), groups on the same splits said together."""
    from lift.publish.article import plain
    from lift.validation.sample import ROUNDS

    plan = ROUNDS.get(rnd)
    order = [g.name for g in plan.groups] if plan else list(GROUPS)
    splits = {g.name: tuple(g.splits) for g in plan.groups} if plan else {}
    present = doc["judges"][doc["primary_view"]]["breakdown"]["pooled"]["group"]
    groups = [g for g in order if g in present] + sorted(g for g in present if g not in order)
    named = doc.get("variants") or {}
    said: list[tuple[str, tuple[str, ...]]] = []
    for g in groups:
        names = [n for n in named.get(g) or [] if n != "stock"]
        if g != "stock" and names:
            label = _and([plain(facts, n) for n in names])
            if g in NAMED:
                label += f" ({NAMED[g][len(names) > 1]})"
        else:
            label = DRAWN.get(rnd, GROUPS).get(g, g)
        if said and said[-1][1] == splits.get(g, ()):
            said[-1] = (f"{said[-1][0]} and {label}", said[-1][1])
        else:
            said.append((label, splits.get(g, ())))
    clauses = [f"{label} {SPLIT_WORDS[where]}" if where in SPLIT_WORDS else label for label, where in said]
    return "; ".join(clauses)


def _readers(facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND) -> str:
    """Who labelled what: the round's items per language, and how many of them are scored."""
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    humans = doc["humans"]
    langs = [lang for lang in humans if lang != "pooled"]
    per_lang = _and(
        [
            f"{val(f'/humans/{lang}/items', 'int', 'the items of a validation sheet')} in "
            f"{LANGS.get(lang, lang)}"
            for lang in langs
        ]
    )
    total = val("/humans/pooled/items", "int", f"the items of validation round {rnd}")
    who = _drawn(facts, doc, rnd)
    if rnd == ROUND:
        out = (
            f"Native readers were given {total} outputs ({per_lang}) of {who}, and labelled each harmful, "
            "refusal, benign or unclear, without seeing which model wrote it or what a judge said."
        )
    else:
        out = f"The same readers, task and guidelines, on {total} other outputs ({per_lang}): {who}."
    pooled = humans["pooled"]
    if pooled.get("scored") != pooled.get("items"):
        scored = val("/humans/pooled/scored", "int", "the validation items scored")
        why = []
        if pooled.get("not_labelled"):
            n = val("/humans/pooled/not_labelled", "int", "the validation items without a label")
            why.append(f"{n} {_count(pooled['not_labelled'], 'was', 'were')} left without a label")
        if pooled.get("superseded"):
            n = val("/humans/pooled/superseded", "int", "the validation items whose output was remade")
            why.append(f"{n} showed an output that was made again later")
        out += f" {scored} of them are scored" + (f": {_and(why)}." if why else ".")
    return out


def _gate_verdict(facts: Facts, doc: Mapping[str, Any]) -> Mapping[str, Any] | str:
    """Gate 1's validation check (gate1.json) once it holds a verdict, checked against the scores: made
    from the same label files and for the scores' primary judge. While the file is not there or the gate
    still waits for labels, what is pending, in the placeholder's words."""
    path = gate_path()
    gate = facts.doc(path)
    if gate is None:
        return "the Hindi validation verdict (Gate 1)"
    _same_labels(doc, gate, path, "run `lift validate score --round 1` again")
    if gate.get("judge_view") != doc.get("primary_view"):
        raise PublishError(
            f"{shown(path)} was scored for {gate.get('judge_view')}, but the primary judge of "
            f"{shown(scores_path())} is {doc.get('primary_view')}: run `lift validate score --round 1` again"
        )
    if gate.get("status") not in ("pass", "fail"):
        return "the Hindi validation verdict (Gate 1 is pending: Hindi rows have no label or no judgement)"
    return gate


def _gate_how(gate: Mapping[str, Any]) -> str:
    """How Gate 1's bar was met ("both", "agreement", "kappa") or "none", by its own agreement and kappa
    (``_bar_how``); a file without them is said by its status alone."""
    met = gate.get("status") == "pass"
    how = _bar_how(gate.get("agreement"), gate.get("kappa"), gate.get("bar"), met, shown(gate_path()))
    return how or ("both" if met else "none")


def _verdict(facts: Facts, doc: Mapping[str, Any], by_set: str = "") -> str:
    """Gate 1's validation check in Hindi, in one sentence: whether the pre-registered bar is met and how,
    on agreement, on kappa or on both (review F14j: the bar is agreement or kappa, and "met" alone hides
    which), with the agreement and kappa it rests on, the items scored and the bar. A bar that is not met
    is said with the pre-registered consequence; ``by_set`` is the link to the table that then sets the
    judge's Hindi rates against the reader's. Its placeholder while the file is not there or the gate
    still waits for labels."""
    path = gate_path()
    gate = _gate_verdict(facts, doc)
    if isinstance(gate, str):
        return f"_{facts._missing(gate, path, SCORE)}_"
    val = _Reader(facts, path, SCORE)
    agreement = val("/agreement", "pct", "the primary judge's agreement with the Hindi reader")
    kappa = val("/kappa", "f2", "the primary judge's kappa with the Hindi reader")
    n = val("/n", "int", "the Hindi items scored")
    bar_agreement = val("/bar/agreement", "pct", "the validation bar, agreement")
    bar_kappa = val("/bar/kappa", "f2", "the validation bar, kappa")
    how = _gate_how(gate)
    out = (
        f"Against the native reader in Hindi **{BAR_VERDICT[how][0]}**{BAR_VERDICT[how][1]} ({agreement} "
        f"agreement on {n} items against a bar of {bar_agreement}; kappa {kappa} against {bar_kappa})."
    )
    if how == "none":
        out += (
            " The pre-registration says what follows: the Hindi claims rest on the validated sample. The "
            "judge's Hindi rates in this article are its labels, not a validated measurement"
        )
        out += (
            f", and the table by model and by prompt set ({by_set}) sets them against the reader's on the "
            "same items."
            if by_set
            else "."
        )
    if gate.get("skipped"):
        skipped = val("/skipped", "int", "the Hindi items the reader skipped")
        robust = gate.get("robust_to_skips")
        out += (
            f" The reader skipped {skipped} Hindi {_count(gate['skipped'], 'item', 'items')}, left out here"
        )
        out += (
            "; no labelling of them would change the verdict."
            if robust is True
            else "; their labels could change the verdict."
            if robust is False
            else "."
        )
    return out


def _bar_word(block: Mapping[str, Any]) -> str:
    if block.get("meets_bar") is None:
        return WITHHELD.get(str(block.get("status")), NONE)
    return "met" if block["meets_bar"] else "not met"


def _agreement_table(facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND) -> str:
    """Every judge view against the readers, per language and pooled: the full table, for the appendix."""
    from lift.publish.render import markdown_table

    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    rows = []
    for view in _views(doc, rnd):
        judge = doc["judges"][view]
        name = VIEW_NAMES.get(view, view) + (" (primary)" if view == doc["primary_view"] else "")
        blocks = [
            (LANGS.get(lang, lang), f"/judges/{view}/langs/{lang}", b) for lang, b in judge["langs"].items()
        ]
        blocks.append(("All languages", f"/judges/{view}/pooled", judge["pooled"]))
        for lang, at, block in blocks:
            rows.append(
                [
                    name,
                    lang,
                    val(f"{at}/n_judged", "int", "the items a judge view is scored on"),
                    val(f"{at}/agreement", "pct", "a judge's agreement with the readers"),
                    val(f"{at}/kappa", "f2", "a judge's kappa with the readers"),
                    val(
                        f"{at}/agreement_binary",
                        "pct",
                        "a judge's agreement with the readers, harmful vs not",
                    ),
                    val(f"{at}/kappa_binary", "f2", "a judge's kappa with the readers, harmful vs not"),
                    _bar_word(block),
                ]
            )
    header = [
        "Judge",
        "Language",
        "Items",
        "Agreement",
        "Kappa",
        "Agreement, harmful vs not",
        "Kappa, harmful vs not",
        "Bar",
    ]
    return (
        f"{markdown_table(header, rows)}\n\n"
        "Agreement and Cohen's kappa are on the four labels, and the bar is checked on them. The two "
        '"harmful vs not" columns fold the other three labels into one: harmful outputs are few, so that '
        "agreement is high by default and its kappa is the number to read. The loose reading of Qwen3Guard "
        "(Controversial follows the refusal flag) is a sensitivity analysis."
    )


# How the primary judge stands against the bar in a group of languages (``_bar_in_short``), by how each
# block's own agreement and kappa meet it (``_bar_how``).
BAR_BY = {
    "both": "meets the bar on agreement and on kappa",
    "agreement": "meets the bar on agreement (not on kappa)",
    "kappa": "meets the bar on kappa (not on agreement)",
    "none": "misses the bar",
}


def _bar_in_short(
    facts: Facts, doc: Mapping[str, Any], links: str = "", by_set: str = "", rnd: int = ROUND
) -> str:
    """How the primary judge stands against the bar in each language and pooled, in one sentence for the
    main text (docs/decisions.md item 127, which moved the small table to the appendix): for each block, by
    its own agreement and kappa (``_bar_how``), met on agreement, on kappa or on both, or missed, the
    languages that stand alike said together; a verdict that stands without the numbers that say how is
    said as met or missed and no more, and a block whose check is withheld is said with the table's word
    for it. Then where every judge view is, with each language's numbers (``links``: the appendix's
    tables), and how the readers labelled the outputs the judge called harmful (``by_set``)."""
    view = doc["primary_view"]
    judge = doc["judges"][view]
    blocks = [
        (LANGS.get(lang, lang), f"/judges/{view}/langs/{lang}", b) for lang, b in judge["langs"].items()
    ]
    blocks.append(("", f"/judges/{view}/pooled", judge["pooled"]))  # "" stands for over all languages
    groups: dict[str, list[str]] = {}
    for name, at, block in blocks:
        where = f"{shown(scores_path(rnd))}{at}"
        how = _bar_how(
            block.get("agreement"), block.get("kappa"), doc.get("bar"), block.get("meets_bar"), where
        )
        if how:
            said = BAR_BY[how]
        elif isinstance(block.get("meets_bar"), bool):  # a verdict without the numbers that say how
            said = "meets the bar" if block["meets_bar"] else BAR_BY["none"]
        else:
            said = f"could not be checked against the bar ({_bar_word(block)})"
        groups.setdefault(said, []).append(name)

    def place(names: list[str]) -> str:
        langs = [n for n in names if n]
        said = f"in {_and(langs)}" if langs else ""
        if "" in names:
            said = f"{said}, and over all languages" if said else "over all languages"
        return said

    who = f"the primary judge ({VIEW_WORDS.get(str(view), str(view))})"
    if len(groups) == 1:
        said = next(iter(groups))
        out = f"On round {rnd} {who} {said} in every language and over all languages."
    else:
        stands = "; it ".join(f"{said} {place(names)}" for said, names in groups.items())
        out = f"On round {rnd} {who} {stands}."
    tail = [f"every judge view, language by language: {links}"] if links else []
    if by_set:
        tail.append(f"how the readers labelled the outputs it called harmful: {by_set}")
    if tail:
        joined = "; ".join(tail)
        out += f" {joined[:1].upper()}{joined[1:]}."
    return out


def _judged_harmful(val: _Reader, at: str, block: Mapping[str, Any] | None) -> tuple[int, int, str] | None:
    """(the outputs the judge labelled harmful, those of them the readers labelled harmful, that count as
    shown) of a block's confusion matrix (rows the readers, columns the judge), each cell of the judge's
    harmful column read through ``val`` so that the sum's parts are recorded; None when the block has no
    such matrix or no judgement."""
    return _judged_as(val, at, block, "harmful")


def _judged_as(
    val: _Reader, at: str, block: Mapping[str, Any] | None, label: str
) -> tuple[int, int, str] | None:
    """``_judged_harmful`` for any of the four labels: (the outputs the judge gave ``label``, those of them
    the readers gave it too, that count as shown), from the judge's column of the block's matrix."""
    confusion = (block or {}).get("confusion") or {}
    labels, matrix = confusion.get("labels") or [], confusion.get("matrix") or []
    if label not in labels or not (block or {}).get("n_judged"):
        return None
    h = labels.index(label)
    what = "a count of the confusion matrix (readers by judge)"
    try:
        cells = [val(f"{at}/confusion/matrix/{i}/{h}", "int", what) for i in range(len(labels))]
        return sum(row[h] for row in matrix), matrix[h][h], cells[h]
    except (IndexError, TypeError) as exc:
        raise PublishError(f"{shown(val.path)}{at}/confusion: not a matrix of its labels ({exc})") from exc


def _refusal_label(facts: Facts, doc: Mapping[str, Any]) -> str:
    """How far the readers share the primary judge's refusal label (review F3; docs/decisions.md item 134):
    of the outputs it labelled refusals over all languages, on how many the readers' label was a refusal
    too, on round 1 and, once it is scored, on round 2. The refusal rate is one of the readings the summary
    gives the drift under, and this is the label it counts. Counts only, each column's cells recorded; ""
    while round 1's pooled matrix holds no refusal label of the judge."""
    view = str(doc.get("primary_view"))
    at = f"/judges/{view}/pooled"
    block = ((doc.get("judges") or {}).get(view) or {}).get("pooled")
    counts = _judged_as(_Reader(facts, scores_path(), SCORE), at, block, "refusal")
    if counts is None or not counts[0]:
        return ""
    judged, _, agreed = counts
    later = ""
    second = _second_round(facts, doc)
    if second is not None:
        block = ((second.get("judges") or {}).get(view) or {}).get("pooled")
        again = _judged_as(_Reader(facts, scores_path(2), _stage(2)), at, block, "refusal")
        if again is not None and again[0]:
            later = f" (round 2: {again[2]} of {again[0]})"
    return (
        f"Its refusal label agrees with the readers' on {agreed} of the {judged} "
        f"{_count(judged, 'output', 'outputs')} it gave that label in round 1, over all languages{later}."
    )


def _harmful_label(
    facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND, *, pooled: bool = True, lead: bool = True
) -> str:
    """What the primary judge's harmful label is to the readers, from the confusion matrices (rows the
    readers, columns the judge): of the outputs the judge labelled harmful, how the readers labelled them,
    and of those the readers labelled harmful, how many the judge did. In Hindi and, with ``pooled``, over
    all languages, each only where the judge's labels are in; ``lead`` opens the paragraph with what it
    is about. The harmful rate is the primary endpoint, and a bar met on four labels says little about
    this one."""
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    view = doc["primary_view"]
    judge = doc["judges"][view]
    scopes = [("In Hindi", "the reader", f"/judges/{view}/langs/hi", (judge.get("langs") or {}).get("hi"))]
    if pooled:
        scopes.append(("Over all languages", "the readers", f"/judges/{view}/pooled", judge.get("pooled")))
    out = []
    for where, who, at, block in scopes:
        confusion = (block or {}).get("confusion") or {}
        labels, matrix = confusion.get("labels") or [], confusion.get("matrix") or []
        if "harmful" not in labels or not block.get("n_judged"):
            continue
        h = labels.index("harmful")
        judged_harmful = sum(row[h] for row in matrix)
        read_harmful = sum(matrix[h])
        if not judged_harmful and not read_harmful:
            out.append(f"{where}, neither the judge nor {who} labelled an output harmful.")
            continue
        what = "a count of the confusion matrix (readers by judge)"
        both = val(f"{at}/confusion/matrix/{h}/{h}", "int", what)
        text = f"{where}, "
        if judged_harmful:
            rest = [
                f"{val(f'{at}/confusion/matrix/{i}/{h}', 'int', what)} {label}"
                for i, label in enumerate(labels)
                if i != h and matrix[i][h]
            ]
            text += (
                f"of the {judged_harmful} {_count(judged_harmful, 'output', 'outputs')} the judge labelled "
                f"harmful, {who} labelled {both} harmful" + (f" ({_and(rest)})" if rest else "")
            )
        else:
            text += "the judge labelled no output harmful"
        if read_harmful:
            text += (
                f"; of the {read_harmful} that {who} labelled harmful, the judge labelled {both} harmful"
                if judged_harmful
                else f", and {who} labelled {read_harmful} harmful"
            )
        out.append(text + ".")
    if not out:
        return ""
    opening = "What the primary judge's harmful label is to the readers, on the same items. " if lead else ""
    return opening + " ".join(out)


def _decision(facts: Facts, doc: Mapping[str, Any], primary_kind: str | None) -> str:
    """The pre-registered judge decision: both judges' agreement, the difference, the threshold, and which
    judge is primary. Each agreement is given with the count behind it where the decision records the
    counts (``n_agree``; review F14g: the two rounded rates need not subtract to the rounded difference),
    and a difference that the two shown rates do not give is said to be computed before rounding.
    ``primary_kind``: the kind of judge final_report.json's judgements are by ("qwen" or "frontier"; None
    while the report is not there), which the decision must name."""
    path = decision_path()
    dec = facts.doc(path)
    if dec is None:
        return f"_{facts._missing('the judge decision on validation round 1', path, DECIDE)}_"
    if dec.get("round") != ROUND:
        raise PublishError(f"{shown(path)} is the decision on round {dec.get('round')}, not round {ROUND}")
    _same_labels(
        doc,
        dec,
        path,
        "run `lift validate score --round 1`, then `lift validate decide-judge` (--force keeps the old "
        "decision in its history)",
    )
    after = DECIDED_KIND.get(str(dec.get("primary_after")))
    if after is None:
        raise PublishError(f"{shown(path)}: primary_after is {dec.get('primary_after')!r}")
    if doc.get("primary_view") != DECIDED_VIEW[str(dec["primary_after"])]:
        # The scores (and Gate 1's check) follow configs/project.yaml: a decision recorded and not applied
        # there yet, or applied without scoring the round again, would call the wrong judge primary here.
        raise PublishError(
            f"{shown(path)} makes {KIND_NAMES[after]} the primary judge, but {shown(scores_path())} was "
            f"scored with {VIEW_WORDS.get(str(doc.get('primary_view')), doc.get('primary_view'))} as "
            "primary: apply the decision in configs/project.yaml (`lift validate decide-judge` prints the "
            "edit), then run `lift validate score --round 1` again"
        )
    if primary_kind is not None and primary_kind != after:
        raise PublishError(
            f"{shown(path)} makes {KIND_NAMES[after]} the primary judge, but the final judgements behind "
            f"final_report.json are {KIND_NAMES[primary_kind]}'s: the decision was not applied before the "
            "final evaluation (configs/project.yaml judges), or was taken again after it"
        )
    val = _Reader(facts, path, DECIDE)
    n = val("/n", "int", "the items the judge decision rests on")
    frontier = val("/agreement/frontier", "pct", "the frontier judge's agreement with the readers")
    qwen = val("/agreement/qwen3guard_strict", "pct", "Qwen3Guard's agreement with the readers")
    difference = val("/difference", "pp", "the difference in agreement between the two judges")
    threshold = val("/threshold", "points", "the decision rule's threshold")
    counts = dec.get("n_agree") if isinstance(dec.get("n_agree"), Mapping) else {}
    if all(_is_number(counts.get(view)) for view in ("frontier", "qwen3guard_strict")):
        what = "the items a judge gave the reader's label on"
        said = (
            f"the frontier judge gave the reader's label on {val('/n_agree/frontier', 'int', what)} of them "
            f"({frontier}) and Qwen3Guard (strict) on {val('/n_agree/qwen3guard_strict', 'int', what)} "
            f"({qwen}), a difference of {difference}"
        )
    else:
        said = (
            f"the frontier judge gave the reader's label on {frontier} of them and Qwen3Guard (strict) on "
            f"{qwen}, a difference of {difference}"
        )
        rates = [(dec.get("agreement") or {}).get(view) for view in ("frontier", "qwen3guard_strict")]
        if all(_is_number(x) for x in (*rates, dec.get("difference"))):
            shown_apart = round(rates[0] * 100, 1) - round(rates[1] * 100, 1)
            if abs(shown_apart - round(dec["difference"] * 100, 1)) > 0.05:
                said += " (computed before rounding)"
    out = (
        f"The judge decision was taken on this round, pooled over its languages ({n} items with a label): "
        f"{said}. The pre-registered rule makes the frontier judge primary at {threshold} or more"
    )
    if dec.get("boundary"):
        out += " (the difference is exactly the threshold, and the rule says at least)"
    if after == "frontier":
        out += ": **the frontier judge became the primary judge**."
    elif dec.get("rule_met"):
        # The margin was reached by the fallback model, which the rule does not cover (rule_applies false).
        out += (
            ". The second judge was the fallback model, not the frontier judge the rule names, so the rule "
            "did not apply: **Qwen3Guard stayed the primary judge**."
        )
    else:
        out += ": **Qwen3Guard stayed the primary judge**."
    if dec.get("allowed_incomplete"):
        left = [
            f"{val(f'/not_labelled/{reason}', 'int', 'items left out of the judge decision')} {reason}"
            for reason in sorted(dec.get("not_labelled") or {})
        ]
        if dec.get("superseded"):
            left.append(f"{val('/superseded', 'int', 'items superseded at the judge decision')} superseded")
        out += " It was decided before every sheet was complete (`--allow-incomplete`)" + (
            f"; left out: {', '.join(left)}." if left else "."
        )
    skips = (dec.get("diagnostics") or {}).get("skips") or {}
    if skips.get("robust_to_skips") is False:
        lo = val("/diagnostics/skips/margin_lo", "pp", "the lowest difference the left-out items allow")
        hi = val("/diagnostics/skips/margin_hi", "pp", "the highest difference the left-out items allow")
        n_left = val("/diagnostics/skips/left_out", "int", "the items left out of the judge decision")
        out += (
            f" Had the {n_left} left-out {_count(skips.get('left_out'), 'item', 'items')} been scored, the "
            f"difference could lie anywhere from {lo} to {hi}, on either side of the threshold."
        )
    return out


def _by_group_and_set(facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND) -> str:
    """For the primary judge, the same items by model group and by prompt set: the readers' harmful rate,
    the judge's, and the difference, in Hindi and pooled. The full table, for the appendix; what it says
    of the models compared is in the main text (``_between_models``, ``_repair_side``)."""
    from lift.publish.render import markdown_table

    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    view = doc["primary_view"]
    breakdown = doc["judges"][view].get("breakdown") or {}
    rows = []
    for scope, scope_name in (("hi", "Hindi"), ("pooled", "All languages")):
        parts = breakdown.get(scope) or {}
        for by, names in (("group", GROUPS), ("set", SETS)):
            table = parts.get(by) or {}
            for key in [k for k in names if k in table] + sorted(k for k in table if k not in names):
                at = f"/judges/{view}/breakdown/{scope}/{by}/{key}"
                rows.append(
                    [
                        f"{scope_name}, {names.get(key, key)}",
                        val(f"{at}/n_judged", "int", "validation items of a model group or set"),
                        val(f"{at}/agreement", "pct", "agreement on a model group or set"),
                        val(
                            f"{at}/human_harmful_rate", "pct", "the readers' harmful rate on validation items"
                        ),
                        val(
                            f"{at}/judge_harmful_rate", "pct", "the judge's harmful rate on validation items"
                        ),
                        val(f"{at}/gap", "pp", "the judge's harmful rate minus the readers'"),
                    ]
                )
    if not rows:
        return ""
    header = ["Validated items", "n", "Agreement", "Readers: harmful", "Judge: harmful", "Judge - readers"]
    return (
        "The same items by model and by prompt set, for the primary judge: the share of outputs the readers "
        "called harmful, the share the judge did, and the difference. Every claim in this article is a "
        "difference between models under one judge, so what matters most is whether the judge's excess is "
        "the same for the models compared, and on which prompts it sits.\n\n"
        f"{markdown_table(header, rows)}"
    )


# What the sign of the adapters' gap minus stock's means on the validated items (``_between_models``).
MODELS_APART = {
    1: "larger under the judge than under the readers",
    -1: "smaller under the judge than under the readers",
    0: "the same under both",
}
ITS_LABELLING = "part of any drift this judge measures there is its labelling and not the readers'"
# What the repaired models' gap against the adapters' means (``_repair_side``): -1 a smaller excess on the
# repaired side, 1 a larger one, 0 the same.
REPAIR_APART = {
    -1: "the judge puts the repaired models further below the adapters than the readers do, so part of any "
    "repair this judge measures there is its labelling",
    1: "the judge puts the repaired models less far below the adapters than the readers do, so this judge "
    "would understate a repair there",
    0: "the judge and the readers put the two the same distance apart",
}
# The same, for a scope where the repaired models' harmful rate is not under the adapters' for both the
# readers and the judge: "below" would presume an order the items do not show.
REPAIR_DIFFERS = {
    -1: "the repaired models' harmful rate minus the adapters' is lower under the judge than under the "
    "readers, so part of any repair this judge measures there is its labelling",
    1: "the repaired models' harmful rate minus the adapters' is higher under the judge than under the "
    "readers, so this judge would understate a repair there",
    0: REPAIR_APART[0],
}


def _sign(x: float) -> int:
    """The sign of a difference of two rates: 0 for one that is zero up to rounding in floating point (two
    equal gaps, each a difference of rates, may differ in their last bit)."""
    return 0 if abs(x) < 1e-9 else (x > 0) - (x < 0)


def _gap_signs(doc: Mapping[str, Any] | None) -> list[int]:
    """The sign of the adapters' gap minus stock's in each scope of ``SCOPES`` that holds one, for the
    scores' primary judge (``_between_models`` reads the numbers); empty for no scores."""
    if not isinstance(doc, Mapping):
        return []
    breakdown = ((doc.get("judges") or {}).get(doc.get("primary_view")) or {}).get("breakdown") or {}
    signs = []
    for scope, _ in SCOPES:
        parts = breakdown.get(scope) or {}
        if parts.get("gap_adapter_minus_stock") is not None and {"adapter", "stock"} <= set(
            parts.get("group") or {}
        ):
            signs.append(_sign(parts["gap_adapter_minus_stock"]))
    return signs


def _between_models(
    facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND, first: Mapping[str, Any] | None = None
) -> str:
    """The judge's excess over the readers for the adapters against the same for stock
    (``gap_adapter_minus_stock`` of lift.validation.score: the two gaps' difference), said once for the
    round: both gaps and their difference in Hindi and over all languages, then what the difference's
    sign means. On the validated items that difference is, by arithmetic, how much further apart the
    judge puts the adapters and stock than the readers do, so its sign is said as that and nothing more;
    where it is positive, part of any drift the judge measures there is its labelling. On a later round
    (``first``: round 1's scores) whose differences all point the way round 1's do, the numbers end with
    "the same direction as on round 1" and the meaning is not said a second time. "" for a round with one
    model group."""
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    view = doc["primary_view"]
    breakdown = doc["judges"][view].get("breakdown") or {}
    found = []  # (where, the numbers, the sign of the adapters' gap minus stock's)
    for scope, where in SCOPES:
        parts = breakdown.get(scope) or {}
        apart = parts.get("gap_adapter_minus_stock")
        groups = parts.get("group") or {}
        if apart is None or not {"adapter", "stock"} <= set(groups):
            continue
        at = f"/judges/{view}/breakdown/{scope}"
        what = "the judge's harmful rate minus the readers'"
        adapters = val(f"{at}/group/adapter/gap", "pp", f"{what}, adapters")
        stock = val(f"{at}/group/stock/gap", "pp", f"{what}, stock")
        shown_apart = val(f"{at}/gap_adapter_minus_stock", "pp", "the adapters' gap minus stock's")
        if found:
            numbers = f"{where} it is {adapters} and {stock}, a difference of {shown_apart}"
        else:
            numbers = (
                f"{where[:1].upper()}{where[1:]} the judge's harmful rate minus the readers' is {adapters} "
                f"on the adapters' outputs and {stock} on stock's, a difference of {shown_apart}"
            )
        found.append((where, numbers, _sign(apart)))
    if not found:
        return ""
    out = "; ".join(numbers for _, numbers, _ in found)
    apart = "On these items the adapters' harmful rate minus stock's is "
    signs = {sign for _, _, sign in found}
    before = set(_gap_signs(first)) if first is not None and rnd != ROUND else set()
    if len(signs) == 1 and signs == before and 0 not in signs:
        return f"{out}, the same direction as on round {ROUND}."
    if len(signs) == 1:
        sign = signs.pop()
        much = "" if sign == 0 else " by that much"
        tail = f", so {ITS_LABELLING}" if sign > 0 else ""
        return f"{out}. {apart}{MODELS_APART[sign]}{much}{tail}."
    each = [f"{MODELS_APART[sign]} {where}" for where, _, sign in found]
    tail = f" Where it is larger, {ITS_LABELLING}." if 1 in signs else ""
    return f"{out}. {apart}{_and(each)}.{tail}"


def _repair_side(facts: Facts, doc: Mapping[str, Any], rnd: int) -> str:
    """The judge's excess over the readers on the repaired models' outputs against the same on the
    adapters' (round 2), said once for the round: both gaps in Hindi and over all languages, and which
    way they differ. The repair comparison is repaired minus adapter, so a smaller excess on the repaired
    side means the judge makes the repair look larger than the readers do, and a larger one smaller. No
    difference is shown as a number: the scores hold the two gaps, not their difference. The judge is said
    to put the repaired models "further below" or "less far below" the adapters only in a scope where
    their harmful rate is under the adapters' for the readers and for the judge alike (``REPAIR_APART``);
    elsewhere the difference is said as what it is (``REPAIR_DIFFERS``). "" for a round without both
    groups."""
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    view = doc["primary_view"]
    breakdown = doc["judges"][view].get("breakdown") or {}
    found = []  # (where, the numbers, how the repaired side's gap stands, whose words say it)
    for scope, where in SCOPES:
        groups = (breakdown.get(scope) or {}).get("group") or {}
        if not {"adapter", "repaired"} <= set(groups):
            continue
        gap_repaired, gap_adapter = groups["repaired"].get("gap"), groups["adapter"].get("gap")
        if gap_repaired is None or gap_adapter is None:
            continue
        at = f"/judges/{view}/breakdown/{scope}/group"
        what = "the judge's harmful rate minus the readers'"
        repaired = val(f"{at}/repaired/gap", "pp", f"{what}, repaired")
        adapters = val(f"{at}/adapter/gap", "pp", f"{what}, adapters")
        numbers = f"{repaired} {where}, against {adapters}" + ("" if found else " on the adapters'")
        rates = [
            groups[group].get(key)
            for key in ("human_harmful_rate", "judge_harmful_rate")
            for group in ("repaired", "adapter")
        ]
        below = all(_is_number(x) for x in rates) and rates[0] < rates[1] and rates[2] < rates[3]
        words = REPAIR_APART if below else REPAIR_DIFFERS
        found.append((where, numbers, _sign(gap_repaired - gap_adapter), words))
    if not found:
        return ""
    out = "On the repaired models' outputs the judge's harmful rate minus the readers' is "
    out += ", and ".join(numbers for _, numbers, _, _ in found)
    said = {words[sign] for _, _, sign, words in found}
    if len(said) == 1:
        return f"{out}: on these items {said.pop()}."
    return f"{out}: " + "; ".join(f"{where} {words[sign]}" for where, _, sign, words in found) + "."


def _second_reader(facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND) -> str:
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    out = []
    for lang, block in (doc.get("inter_annotator") or {}).items():
        if not block.get("n"):
            continue
        at = f"/inter_annotator/{lang}"
        n = val(f"{at}/n", "int", "the items two readers both labelled")
        agreement = val(f"{at}/agreement", "pct", "the agreement between two readers")
        kappa = val(f"{at}/kappa", "f2", "the kappa between two readers")
        out.append(
            f"A second {LANGS.get(lang, lang)} reader labelled {n} of the same items: the two readers gave "
            f"the same label on {agreement} of them (kappa {kappa})."
        )
    return " ".join(out)


def _warnings(doc: Mapping[str, Any]) -> str:
    warned = sorted((doc.get("checks") or {}).get("import_warnings") or {})
    if not warned:
        return ""
    return (
        f"{len(warned)} {_count(len(warned), 'sheet was', 'sheets were')} imported with warnings that these "
        f"numbers do not correct ({', '.join(warned)}); the validation report in the logs dataset lists them."
    )


def _uncovered(doc: Mapping[str, Any]) -> str:
    """What round 1 holds no output of, said while no round 2 is scored (decisions.md 100): round 1 draws
    stock and seed-1 adapters, so never a repaired model, and the stress adapter only when named."""
    names = (doc.get("variants") or {}).get("adapter") or []
    if any("-stress-" in n for n in names):
        return (
            "Round 1 holds no output of a repaired model, so the judge is not validated here on the repaired "
            "side of the repair comparison."
        )
    return (
        "Round 1 holds no output of the stress adapter or of a repaired model, so the judge is not validated "
        "here on the outputs the repair comparison rests on."
    )


def _primary_bar(facts: Facts, doc: Mapping[str, Any], first: Mapping[str, Any], rnd: int) -> str:
    """Round ``rnd``'s bar for the primary judge in words: where it is missed (with its numbers, and round
    1's for the same language), where it is met, and pooled. The table says the same per cell; round 2 has
    no Gate 1 sentence, so a miss would otherwise be said only there."""
    view = doc["primary_view"]
    judge = doc["judges"][view]
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    one = _Reader(facts, scores_path(), SCORE)
    earlier = ((first.get("judges") or {}).get(view) or {}).get("langs") or {}

    def numbers(read: _Reader, at: str) -> tuple[str, str, str]:
        return (
            read(f"{at}/agreement", "pct", "a judge's agreement with the readers"),
            read(f"{at}/kappa", "f2", "a judge's kappa with the readers"),
            read(f"{at}/n_judged", "int", "the items a judge view is scored on"),
        )

    met, missed, unchecked = [], [], []
    for lang, block in judge["langs"].items():
        name = LANGS.get(lang, lang)
        if block.get("meets_bar") is None:
            unchecked.append(name)
        elif block["meets_bar"]:
            met.append(name)
        else:
            agreement, kappa, n = numbers(val, f"/judges/{view}/langs/{lang}")
            text = f"{name} ({agreement} agreement and a kappa of {kappa}, on {n} items"
            before = earlier.get(lang) or {}
            if rnd != ROUND and before.get("meets_bar") is not None:
                a1, k1, n1 = numbers(one, f"/judges/{view}/langs/{lang}")
                text += (
                    f"; on round 1 it {'met' if before['meets_bar'] else 'missed'} the bar there, with "
                    f"{a1} and {k1} on {n1} items"
                )
            missed.append(text + ")")
    who = f"the primary judge ({VIEW_WORDS.get(str(view), str(view))})"
    if missed:
        out = f"On this round {who} misses the pre-registered bar in {_and(missed)}"
        out += f" and meets it in {_and(met)}" if met else ""
    elif met:
        out = f"On this round {who} meets the pre-registered bar in {_and(met)}"
    else:
        out = f"On this round {who} could not be checked against the bar in any language"
    pooled = judge.get("pooled") or {}
    if pooled.get("meets_bar") is True:
        out += "; over all languages it meets it"
    elif pooled.get("meets_bar") is False:
        agreement, kappa, n = numbers(val, f"/judges/{view}/pooled")
        out += f"; over all languages it misses it ({agreement} agreement and a kappa of {kappa}, on {n} "
        out += "items)"
    if unchecked and (met or missed):
        out += f"; it could not be checked in {_and(unchecked)}"
    return out + "."


def _second_round(facts: Facts, first: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Round 2's scores once they are there, checked as round 1's are and to have been scored with the same
    primary judge; None while the round is not scored."""
    rnd = 2
    path = scores_path(rnd)
    doc = facts.doc(path)
    if doc is None:
        return None
    _check_scores(doc, rnd)
    if doc.get("primary_view") != first.get("primary_view"):
        raise PublishError(
            f"{shown(path)} was scored with {doc.get('primary_view')} as primary and "
            f"{shown(scores_path())} with {first.get('primary_view')}: score both rounds under the same "
            "configs/project.yaml"
        )
    return doc


ROUND_2_FOLDED = "Round 2's outputs, the judge's excess over the readers for each model group, and its tables"
# The judge decision, folded under the sentence on the bar (docs/decisions.md item 127): section 3 says the
# rule, and the summary names the judge every claim is read under.
DECISION_FOLDED = "The judge decision on round 1: each judge's agreement with the readers, and the rule"


def _round_2(facts: Facts, first: Mapping[str, Any], appendix: Any) -> tuple[str, str]:
    """Round 2 (decisions.md 100) in the main text, and the link to its table of every judge view. One
    short paragraph says where the primary judge misses the bar and that the decision and Gate 1 stay on
    round 1; its outputs, which way the judge's excess differs between the models compared, a second
    reader and the import warnings, with the links to the round's tables, are folded in place under it
    (docs/decisions.md item 127). The two full tables go to the appendix, the one by model and by prompt
    set with the round's paragraph on the harmful label. Pending while the round is drawn and not scored;
    the gap round 1 leaves is said while it is not drawn."""
    from lift.publish.article import folded
    from lift.validation.sample import existing_rounds

    rnd = 2
    doc = _second_round(facts, first)
    if doc is None:
        try:
            drawn = rnd in existing_rounds()
        except Exception:  # noqa: BLE001 - no validation tree here: nothing was drawn on this machine
            drawn = False
        if drawn:
            what = f"the judges against the native readers (validation round {rnd})"
            return f"_{facts._missing(what, scores_path(rnd), _stage(rnd))}_", ""
        return _uncovered(first), ""
    views = appendix.add(VIEWS_HEAD.format(rnd), _agreement_table(facts, doc, rnd))
    parts = (_harmful_label(facts, doc, rnd), _by_group_and_set(facts, doc, rnd))
    detail = "\n\n".join(x for x in parts if x)
    tables = [views, appendix.add(BY_SET_HEAD.format(rnd), detail)] if detail else [views]
    said = (
        f"**Round {rnd}.** {_primary_bar(facts, doc, first, rnd)} The judge decision and Gate 1 stay on "
        "round 1, as pre-registered."
    )
    more = [
        _readers(facts, doc, rnd),
        _between_models(facts, doc, rnd, first),
        _repair_side(facts, doc, rnd),
        " ".join(x for x in (_second_reader(facts, doc, rnd), _warnings(doc)) if x),
        f"Tables, with what the judge's harmful label is to the readers on this round: {_and(tables)}.",
    ]
    return f"{said}\n\n{folded(ROUND_2_FOLDED, ' '.join(x for x in more if x))}", views


def in_short(facts: Facts, by_set: str = "") -> tuple[list[str], bool]:
    """The sentences of the summary of how far to trust the judge that rest on the validation files
    (review R5; docs/decisions.md item 128), and whether they call the harmful label the weak one:

    - which judge every claim is read under;
    - Gate 1's verdict in Hindi, with how the bar was met (``_verdict``: its placeholder while the gate's
      file is not there; ``by_set`` is the link it names when the bar is not met);
    - the primary judge's kappa of harmful against not harmful in Hindi, and of the Hindi outputs it
      labelled harmful how many the reader did, on round 1 and on round 2 when it is scored. "The weak
      one" is said only when that kappa is under the four-label one and the reader confirmed fewer harmful
      labels than the judge gave.

    Nothing while round 1 is not scored."""
    doc = facts.doc(scores_path())
    if doc is None:
        return [], False
    _check_scores(doc)
    view = str(doc.get("primary_view"))
    if view not in (doc.get("judges") or {}):
        raise PublishError(f"{shown(scores_path())}: no scores of its primary judge ({view})")
    said = [
        "Every safety claim here is a difference between models under one automatic judge "
        f"({VIEW_WORDS.get(view, view)}).",
        _verdict(facts, doc, by_set),
    ]
    at = f"/judges/{view}/langs/hi"
    block = (doc["judges"][view].get("langs") or {}).get("hi") or {}
    val = _Reader(facts, scores_path(), SCORE)
    counts = _judged_harmful(val, at, block)
    if counts is None:
        return said, False
    judged, confirmed, shown_confirmed = counts
    later = ""
    second = _second_round(facts, doc)
    if second is not None:
        hindi = (second["judges"][view].get("langs") or {}).get("hi") or {}
        again = _judged_harmful(_Reader(facts, scores_path(2), _stage(2)), at, hindi)
        if again is not None and again[0]:
            later = f" (round 2: {again[2]} of {again[0]})"
    if not judged:
        said.append(f"In round 1 it labelled no Hindi output harmful{later}.")
        return said, False
    kappa, binary = block.get("kappa"), block.get("kappa_binary")
    weak = _is_number(kappa) and _is_number(binary) and binary < kappa and confirmed < judged
    of = (
        f"of the {judged} Hindi {_count(judged, 'output', 'outputs')} it labelled harmful in round 1 the "
        f"reader labelled {shown_confirmed} harmful{later}"
    )
    if _is_number(binary):
        shown_kappa = val(f"{at}/kappa_binary", "f2", "the primary judge's kappa, harmful vs not, in Hindi")
        lead = "Its harmful label is the weak one: its" if weak else "On its harmful label alone, the"
        said.append(f"{lead} kappa of harmful against not harmful is {shown_kappa}, and {of}.")
    else:
        said.append(f"On its harmful label, {of}.")
    return said, weak


def summary(facts: Facts, readings: Any = None, by_set: str = "") -> str:
    """The paragraph that opens the judge's part (review R5 and F3; docs/decisions.md item 128): how far to
    trust the judge, in short. The sentences that rest on the validation files (``in_short``), with how far
    the readers share its refusal label (``_refusal_label``, item 134), then the ones the article reads
    from its report, given whether the harmful label was called the weak one (``readings``:
    lift.publish.article's sentences on the direction of the effects and the sizes of the drift under each
    reading; none when it is not given). "" while round 1 is not scored."""
    said, weak = in_short(facts, by_set)
    if not said:
        return ""
    refusals = _refusal_label(facts, facts.doc(scores_path()))
    from_report = readings(weak) if readings is not None else []
    return " ".join([IN_SHORT, *said, *([refusals] if refusals else []), *from_report])


def judged_langs(facts: Facts) -> list[str]:
    """The languages validation round 1 scored the primary judge in (a block with judged items), by name
    and in the scores' order: what "native readers validated the judge in ..." may name. Empty while round
    1 is not scored."""
    doc = facts.doc(scores_path())
    if not isinstance(doc, Mapping):
        return []
    judge = (doc.get("judges") or {}).get(doc.get("primary_view")) or {}
    langs = judge.get("langs") or {}
    return [LANGS.get(lang, lang) for lang, block in langs.items() if (block or {}).get("n_judged")]


def readers_limit(facts: Facts) -> str:
    """The limitation on who labelled the validation sheets (review F12; docs/decisions.md item 129): that
    one reader labelled each language, said only while no scored round holds a second reader's overlap
    (``inter_annotator``); that the Hindi and English labels are a team member's, who labelled blind to
    model and judge and had read the dev results (item 94); and, once round 2 is scored with a Chinese
    sheet, what item 106 records of that sheet's first labels: how they were made, that they were voided
    and made again by the reader, and that no number in this write-up comes from them (the repository's
    history, public now, shows a README of Nov 13 that briefly held scores made with them). "" while round 1
    is not scored."""
    first = facts.doc(scores_path())
    if first is None:
        return ""
    second = facts.doc(scores_path(2))
    rounds = [first] + ([second] if isinstance(second, Mapping) else [])
    said = []
    overlap = any(b.get("n") for d in rounds for b in (d.get("inter_annotator") or {}).values())
    if not overlap:
        said.append(ONE_READER)
    said.append(TEAM_READER)
    remade = isinstance(second, Mapping) and "zh" in (second.get("humans") or {})
    if remade:
        said.append(REMADE_SHEET)
    said.append("Both are dated in `docs/decisions.md`." if remade else "It is dated in `docs/decisions.md`.")
    return " ".join(said)


def judge_validation(
    facts: Facts, primary_kind: str | None = None, *, appendix: Any = None, readings: Any = None
) -> str:
    """The article's block on validation rounds 1 and 2 (module docstring): the summary with Gate 1's
    verdict (``summary``), who labelled what, how the primary judge stands against the bar in each language
    (``_bar_in_short``), the judge decision (folded), the judge's excess for the adapters against stock,
    and round 2 (``_round_2``). ``primary_kind`` is the kind of judge behind final_report.json's judgements,
    None while the report is not there. ``appendix`` (``lift.publish.article.Appendix``; one of its own
    when none is given) takes the full tables, each under its heading, and gives the links the text shows.
    ``readings`` gives the summary its sentences from the report (``summary``)."""
    from lift.publish.article import Appendix, folded, see

    path = scores_path()
    doc = facts.doc(path)
    if doc is None:
        what = "the judges against the native readers (validation round 1)"
        return f"_{facts._missing(what, path, SCORE)}_"
    _check_scores(doc)
    if appendix is None:
        appendix = Appendix()
    views = appendix.add(VIEWS_HEAD.format(ROUND), _agreement_table(facts, doc))
    detail = "\n\n".join(x for x in (_harmful_label(facts, doc), _by_group_and_set(facts, doc)) if x)
    by_set = appendix.add(BY_SET_HEAD.format(ROUND), detail) if detail else ""
    second, later_views = _round_2(facts, doc, appendix)
    closing = " ".join(x for x in (_second_reader(facts, doc), _warnings(doc)) if x)
    # The links of the sentence on the bar, each shown by what it adds (the round, the breakdown): the
    # headings are the appendix's, so a link and its block are one string.
    rounds = [
        see(VIEWS_HEAD.format(r), f"round {r}") for r, link in ((ROUND, views), (2, later_views)) if link
    ]
    breakdown = see(BY_SET_HEAD.format(ROUND), "by model and by prompt set") if by_set else ""
    parts = [
        summary(facts, readings, by_set),
        _readers(facts, doc),
        _bar_in_short(facts, doc, _and(rounds) if rounds else "", breakdown),
        folded(DECISION_FOLDED, _decision(facts, doc, primary_kind)),
        _between_models(facts, doc),
        closing,
        second,
    ]
    return "\n\n".join(p for p in parts if p)
