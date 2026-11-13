"""What the first validation round found, as the article says it: the judges against the native readers.

The README makes the readers' labels the ground truth for the judge, with a bar (85% agreement on the
four labels or a Cohen's kappa of 0.6) and a rule for which judge is primary. This module states the
outcome of both from the three files ``lift validate`` writes (P04, ``lift.validation.score``), every
number read through ``Facts`` so that it is recorded in the article's sources:

- ``outputs/metrics/validation/round1.json`` (``lift validate score --round 1``): each judge view against
  the readers per language and pooled, and the breakdown by model group and by prompt set, with the
  readers' and the judge's harmful rates on the same items;
- ``outputs/metrics/validation/gate1.json`` (the same command): Gate 1's check, "round 1 meets the bar in
  Hindi", for the primary judge;
- ``outputs/metrics/judge_decision.json`` (``lift validate decide-judge``): the pre-registered rule applied.

None of the three holds a prompt, an output or a judge's words: counts, rates, judge ids and hashes. The
logs dataset carries them as they are (``lift.publish.datasets._aggregates``), so every number said here
can be checked there.

What is said, and only as far as the files support it (docs/decisions.md item 94):

- how many outputs the readers labelled, per language, and how many of them are scored;
- Gate 1's verdict in Hindi with the agreement and kappa it rests on. A bar that is not met is said with
  the README's consequence (the Hindi claims rest on the validated sample), never softened;
- a table of every judge view against the readers, per language and pooled, with the bar's verdict and
  the kappa of "harmful vs not" beside the four-label one;
- what the primary judge's harmful label is to the readers, from the confusion matrices: of the outputs
  the judge labelled harmful, how the readers labelled them. The harmful rate is the primary endpoint, and
  a bar met on four labels can leave this one weak (item 98);
- the judge decision: both judges' agreement, the difference, the threshold and which judge is primary;
- for the primary judge, the same items by model group and by prompt set: the share the readers called
  harmful, the share the judge did, and the difference. The claims are differences between models under
  one judge, so the table shows whether the judge's excess is the same for stock and for the adapters,
  and on which prompts it sits, and a sentence says which way the two differ (item 98);
- a second Hindi reader's agreement with the first, when the overlap sheet was labelled.

Pending (a placeholder, which ``--yes`` refuses) while a file is not there or Gate 1 still waits for
labels. Refused (``PublishError``) when the files contradict each other or the report: scores made with
another judge as primary than the decision names (a decision not applied in configs/project.yaml yet, or
applied without scoring the round again), scores of another
round or under another bar than configs/project.yaml's, a Gate 1 check scored for another judge than the
scores' primary one, a Gate 1 check or a decision made from other label files than the scores (each
records its inputs' sha256), labels changed since the decision, or a decision that names one judge
primary while final_report.json's judgements are the other's.
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
# A group named with its variants: the word before the names, one variant and several.
NAMED = {
    "adapter": ("the adapter", "the adapters"),
    "repaired": ("the repaired model", "the repaired models"),
}
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


def _drawn(doc: Mapping[str, Any], rnd: int) -> str:
    """The round's model groups as the lead names them: each with its variants when the scores name them
    (``variants``, written by lift.validation.score since Nov 12), and the splits the round's plan draws
    each group from (lift.validation.sample.ROUNDS), groups on the same splits said together."""
    from lift.validation.sample import ROUNDS

    plan = ROUNDS.get(rnd)
    order = [g.name for g in plan.groups] if plan else list(GROUPS)
    splits = {g.name: tuple(g.splits) for g in plan.groups} if plan else {}
    present = doc["judges"][doc["primary_view"]]["breakdown"]["pooled"]["group"]
    groups = [g for g in order if g in present] + sorted(g for g in present if g not in order)
    named = doc.get("variants") or {}
    said: list[tuple[str, tuple[str, ...]]] = []
    for g in groups:
        names = named.get(g) or []
        if g in NAMED and names:
            one, many = NAMED[g]
            label = f"{one if len(names) == 1 else many} " + _and([f"`{n}`" for n in names])
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
    who = _drawn(doc, rnd)
    if rnd == ROUND:
        out = (
            f"Native readers were given {total} outputs ({per_lang}) of {who}, and labelled each harmful, "
            "refusal, benign or unclear, without seeing which model wrote it or what a judge said."
        )
    else:
        out = (
            f"**Round {rnd}.** The same readers, task and guidelines, on {total} other outputs ({per_lang}): "
            f"{who}."
        )
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


def _gate(facts: Facts, doc: Mapping[str, Any]) -> str:
    """Gate 1's validation check in Hindi, with the agreement and kappa it rests on."""
    path = gate_path()
    gate = facts.doc(path)
    if gate is None:
        return f"_{facts._missing('the Hindi validation verdict (Gate 1)', path, SCORE)}_"
    _same_labels(doc, gate, path, "run `lift validate score --round 1` again")
    if gate.get("judge_view") != doc.get("primary_view"):
        raise PublishError(
            f"{shown(path)} was scored for {gate.get('judge_view')}, but the primary judge of "
            f"{shown(scores_path())} is {doc.get('primary_view')}: run `lift validate score --round 1` again"
        )
    status = gate.get("status")
    if status not in ("pass", "fail"):
        what = "the Hindi validation verdict (Gate 1 is pending: Hindi rows have no label or no judgement)"
        return f"_{facts._missing(what, path, SCORE)}_"
    val = _Reader(facts, path, SCORE)
    name = VIEW_WORDS.get(str(gate.get("judge_view")), str(gate.get("judge_view")))
    agreement = val("/agreement", "pct", "the primary judge's agreement with the Hindi reader")
    kappa = val("/kappa", "f2", "the primary judge's kappa with the Hindi reader")
    n = val("/n", "int", "the Hindi items scored")
    bar_agreement = val("/bar/agreement", "pct", "the validation bar, agreement")
    bar_kappa = val("/bar/kappa", "f2", "the validation bar, kappa")
    out = (
        f"In Hindi the primary judge ({name}) gave the reader's label on {agreement} of {n} items, with a "
        f"Cohen's kappa of {kappa}. The pre-registered bar is {bar_agreement} agreement or a kappa of "
        f"{bar_kappa}: "
    )
    if status == "pass":
        out += "**the bar is met**."
    else:
        out += (
            "**the bar is not met**. The pre-registration says what follows: the Hindi claims rest on "
            "the validated sample. The judge's Hindi rates in this article are its labels, not a validated "
            "measurement, and the second table below sets them against the reader's on the same items."
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
    """Every judge view against the readers, per language and pooled."""
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


def _harmful_label(facts: Facts, doc: Mapping[str, Any], rnd: int = ROUND) -> str:
    """What the primary judge's harmful label is to the readers, from the confusion matrices (rows the
    readers, columns the judge): of the outputs the judge labelled harmful, how the readers labelled them,
    and of those the readers labelled harmful, how many the judge did. In Hindi and over all languages,
    each only where the judge's labels are in. The harmful rate is the primary endpoint, and a bar met on
    four labels says little about this one."""
    val = _Reader(facts, scores_path(rnd), _stage(rnd))
    view = doc["primary_view"]
    judge = doc["judges"][view]
    scopes = [("In Hindi", "the reader", f"/judges/{view}/langs/hi", (judge.get("langs") or {}).get("hi"))]
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
    return "What the primary judge's harmful label is to the readers, on the same items. " + " ".join(out)


def _decision(facts: Facts, doc: Mapping[str, Any], primary_kind: str | None) -> str:
    """The pre-registered judge decision: both judges' agreement, the difference, the threshold, and which
    judge is primary. ``primary_kind``: the kind of judge final_report.json's judgements are by ("qwen" or
    "frontier"; None while the report is not there), which the decision must name."""
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
    out = (
        f"The judge decision was taken on this round, pooled over its languages ({n} items with a label): "
        f"the frontier judge gave the reader's label on {frontier} of them and Qwen3Guard (strict) on "
        f"{qwen}, a difference of {difference}. The pre-registered rule makes the frontier judge primary "
        f"at {threshold} or more"
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
    the judge's, and the difference, in Hindi and pooled."""
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
    said = [
        f(val, f"/judges/{view}/breakdown/{scope}", breakdown.get(scope) or {}, where)
        for scope, where in (("hi", "In Hindi"), ("pooled", "Over all languages"))
        for f in (_between_models, _repair_side)
    ]
    return (
        "The same items by model and by prompt set, for the primary judge: the share of outputs the readers "
        "called harmful, the share the judge did, and the difference. Every claim in this article is a "
        "difference between models under one judge, so what matters most is whether the judge's excess is "
        "the same for the models compared, and on which prompts it sits.\n\n"
        f"{markdown_table(header, rows)}" + "".join(f"\n\n{x}" for x in said if x)
    )


def _repair_side(val: _Reader, at: str, parts: Mapping[str, Any], where: str) -> str:
    """The judge's excess over the readers on the repaired models' outputs against the same on the
    adapters' (round 2), in words: both gaps from the scores, and which way they differ. The repair
    comparison is repaired minus adapter, so a smaller excess on the repaired side means the judge makes
    the repair look larger than the readers do, and a larger one smaller. No difference is shown as a
    number: the scores hold the two gaps, not their difference."""
    groups = parts.get("group") or {}
    if not {"adapter", "repaired"} <= set(groups):
        return ""
    gap_repaired, gap_adapter = groups["repaired"].get("gap"), groups["adapter"].get("gap")
    if gap_repaired is None or gap_adapter is None:
        return ""
    repaired = val(f"{at}/group/repaired/gap", "pp", "the judge's harmful rate minus the readers', repaired")
    adapters = val(f"{at}/group/adapter/gap", "pp", "the judge's harmful rate minus the readers', adapters")
    out = (
        f"{where} the judge's harmful rate minus the readers' is {repaired} on the repaired models' outputs "
        f"and {adapters} on the adapters': "
    )
    if gap_repaired < gap_adapter:
        return out + (
            "on these items the judge puts the repaired models further below the adapters than the readers "
            "do, so part of any repair this judge measures there is its labelling."
        )
    if gap_repaired > gap_adapter:
        return out + (
            "on these items the judge puts the repaired models less far below the adapters than the readers "
            "do, so this judge would understate a repair there."
        )
    return out + "on these items the judge and the readers put the two the same distance apart."


def _between_models(val: _Reader, at: str, parts: Mapping[str, Any], where: str) -> str:
    """The judge's excess over the readers for the adapters against the same for stock
    (``gap_adapter_minus_stock`` of lift.validation.score: the two gaps' difference), in words. On the
    validated items that difference is, by arithmetic, how much further apart the judge puts the adapters
    and stock than the readers do, so its sign is said as that and nothing more."""
    apart = parts.get("gap_adapter_minus_stock")
    groups = parts.get("group") or {}
    if apart is None or not {"adapter", "stock"} <= set(groups):
        return ""
    adapters = val(f"{at}/group/adapter/gap", "pp", "the judge's harmful rate minus the readers', adapters")
    stock = val(f"{at}/group/stock/gap", "pp", "the judge's harmful rate minus the readers', stock")
    shown_apart = val(f"{at}/gap_adapter_minus_stock", "pp", "the adapters' gap minus stock's")
    out = (
        f"{where} the judge's harmful rate minus the readers' is {adapters} on the adapters' outputs and "
        f"{stock} on stock's, a difference of {shown_apart}: "
    )
    if apart > 0:
        return out + (
            "on these items the adapters' harmful rate minus stock's is larger under the judge than under "
            "the readers by that much, so part of any drift this judge measures there is its labelling and "
            "not the readers'."
        )
    if apart < 0:
        return out + (
            "on these items the adapters' harmful rate minus stock's is smaller under the judge than under "
            "the readers by that much."
        )
    return out + "on these items the adapters' harmful rate minus stock's is the same under both."


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


def _round_2(facts: Facts, first: Mapping[str, Any]) -> str:
    """Round 2's scores (decisions.md 100), the same tables and paragraphs as round 1's without the bar's
    verdict and the decision, which stay on round 1. Pending while the round is drawn and not scored; the
    gap round 1 leaves is said while it is not drawn."""
    from lift.validation.sample import existing_rounds

    rnd = 2
    path = scores_path(rnd)
    doc = facts.doc(path)
    if doc is None:
        try:
            drawn = rnd in existing_rounds()
        except Exception:  # noqa: BLE001 - no validation tree here: nothing was drawn on this machine
            drawn = False
        if drawn:
            what = f"the judges against the native readers (validation round {rnd})"
            return f"_{facts._missing(what, path, _stage(rnd))}_"
        return _uncovered(first)
    _check_scores(doc, rnd)
    if doc.get("primary_view") != first.get("primary_view"):
        raise PublishError(
            f"{shown(path)} was scored with {doc.get('primary_view')} as primary and "
            f"{shown(scores_path())} with {first.get('primary_view')}: score both rounds under the same "
            "configs/project.yaml"
        )
    closing = " ".join(x for x in (_second_reader(facts, doc, rnd), _warnings(doc)) if x)
    parts = [
        f"{_readers(facts, doc, rnd)} The judge decision and Gate 1 stay on round 1, as pre-registered.",
        _agreement_table(facts, doc, rnd),
        _primary_bar(facts, doc, first, rnd),
        _harmful_label(facts, doc, rnd),
        _by_group_and_set(facts, doc, rnd),
        closing,
    ]
    return "\n\n".join(p for p in parts if p)


def judge_validation(facts: Facts, primary_kind: str | None = None) -> str:
    """The article's block on validation rounds 1 and 2 (module docstring). ``primary_kind`` is the kind
    of judge behind final_report.json's judgements, None while the report is not there."""
    path = scores_path()
    doc = facts.doc(path)
    if doc is None:
        what = "the judges against the native readers (validation round 1)"
        return f"_{facts._missing(what, path, SCORE)}_"
    _check_scores(doc)
    closing = " ".join(x for x in (_second_reader(facts, doc), _warnings(doc)) if x)
    parts = [
        f"{_readers(facts, doc)} {_gate(facts, doc)}",
        _agreement_table(facts, doc),
        _harmful_label(facts, doc),
        _decision(facts, doc, primary_kind),
        _by_group_and_set(facts, doc),
        closing,
        _round_2(facts, doc),
    ]
    return "\n\n".join(p for p in parts if p)
