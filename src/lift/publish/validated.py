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
- a table of every judge view against the readers, per language and pooled, with the bar's verdict;
- the judge decision: both judges' agreement, the difference, the threshold and which judge is primary;
- for the primary judge, the same items by model group and by prompt set: the share the readers called
  harmful, the share the judge did, and the difference. The claims are differences between models under
  one judge, so the table shows whether the judge's excess is the same for stock and for the adapters,
  and on which prompts it sits;
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
# Round 1 draws its adapters at seed 1 (lift.validation.sample.ROUNDS): how the lead names the groups.
DRAWN = {**GROUPS, "adapter": "the seed-1 adapters"}
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


def scores_path() -> Path:
    from lift.validation.store import metrics_dir

    return metrics_dir() / f"round{ROUND}.json"


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


def _views(doc: Mapping[str, Any]) -> list[str]:
    """The judge views the scores hold, the primary one first, then in the order of VIEW_NAMES."""
    primary = doc.get("primary_view")
    judges = doc.get("judges") or {}
    if primary not in judges:
        raise PublishError(f"{shown(scores_path())}: no scores of its primary judge ({primary})")
    rest = [v for v in VIEW_NAMES if v in judges and v != primary]
    return [primary, *rest, *sorted(v for v in judges if v != primary and v not in VIEW_NAMES)]


def _check_scores(doc: Mapping[str, Any]) -> None:
    path = shown(scores_path())
    if doc.get("round") != ROUND:
        raise PublishError(f"{path} holds the scores of round {doc.get('round')}, not round {ROUND}")
    bar = dict(config.get("validation.bar"))
    if doc.get("bar") != bar:
        raise PublishError(
            f"{path} was scored under the bar {doc.get('bar')}, not configs/project.yaml's {bar}: run "
            "`lift validate score --round 1` again"
        )
    changed = (doc.get("checks") or {}).get("labels_changed_since_decision")
    if changed:
        raise PublishError(
            f"{path}: labels were imported again after the judge decision ({', '.join(changed)}): run "
            "`lift validate decide-judge` again (--force keeps the old decision in its history), then "
            "`lift validate score --round 1`"
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


def _readers(facts: Facts, doc: Mapping[str, Any]) -> str:
    """Who labelled what: the round's items per language, and how many of them are scored."""
    val = _Reader(facts, scores_path(), SCORE)
    humans = doc["humans"]
    langs = [lang for lang in humans if lang != "pooled"]
    per_lang = _and(
        [
            f"{val(f'/humans/{lang}/items', 'int', 'the items of a validation sheet')} in "
            f"{LANGS.get(lang, lang)}"
            for lang in langs
        ]
    )
    total = val("/humans/pooled/items", "int", "the items of validation round 1")
    drawn = doc["judges"][doc["primary_view"]]["breakdown"]["pooled"]["group"]
    groups = [g for g in GROUPS if g in drawn] + sorted(g for g in drawn if g not in GROUPS)
    who = _and([DRAWN.get(g, g) for g in groups])
    out = (
        f"Native readers were given {total} dev-split outputs of {who} ({per_lang}) and labelled each "
        "harmful, refusal, benign or unclear, without seeing which model wrote it or what a judge said."
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


def _agreement_table(facts: Facts, doc: Mapping[str, Any]) -> str:
    """Every judge view against the readers, per language and pooled."""
    from lift.publish.render import markdown_table

    val = _Reader(facts, scores_path(), SCORE)
    rows = []
    for view in _views(doc):
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
                    _bar_word(block),
                ]
            )
    header = ["Judge", "Language", "Items", "Agreement", "Kappa", "Agreement, harmful vs not", "Bar"]
    return (
        f"{markdown_table(header, rows)}\n\n"
        "Agreement and Cohen's kappa are on the four labels; the bar is checked on them. The loose reading "
        "of Qwen3Guard (Controversial follows the refusal flag) is a sensitivity analysis."
    )


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


def _by_group_and_set(facts: Facts, doc: Mapping[str, Any]) -> str:
    """For the primary judge, the same items by model group and by prompt set: the readers' harmful rate,
    the judge's, and the difference, in Hindi and pooled."""
    from lift.publish.render import markdown_table

    val = _Reader(facts, scores_path(), SCORE)
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
        "the same for stock and for the adapters, and on which prompts it sits.\n\n"
        f"{markdown_table(header, rows)}"
    )


def _second_reader(facts: Facts, doc: Mapping[str, Any]) -> str:
    val = _Reader(facts, scores_path(), SCORE)
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


def judge_validation(facts: Facts, primary_kind: str | None = None) -> str:
    """The article's block on validation round 1 (module docstring). ``primary_kind`` is the kind of judge
    behind final_report.json's judgements, None while the report is not there."""
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
        _decision(facts, doc, primary_kind),
        _by_group_and_set(facts, doc),
        closing,
    ]
    return "\n\n".join(p for p in parts if p)
