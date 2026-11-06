"""What P08 reads, and the ``Facts`` reader that renders every number with its source.

No number in a public artifact is typed by hand: each one is read here from a JSON file under
``outputs/metrics`` (results) or ``configs/`` (the pre-registered design: set sizes, thresholds), and
``Facts.sources`` lists every value with its file and JSON pointer, which ``lift publish`` writes
beside each rendered file (``*.sources.json``) for the claims audit (R3). An input that does not exist
yet renders as ``[PENDING: <what> from <file> (<stage>)]``; ``lift publish ... --yes`` refuses to upload
a text that still holds one.

Inputs (paths under ``outputs/metrics``; both are written by later stages, and the shapes below were
checked against their writers on origin/p05-repairs and origin/p07-final-report on Nov 6):

``selection.json`` (P05, ``lift select``: ``lift.repair.select.run``).

    {"kind": "selection", "version": 1, "adapter": "adapter-indicalign-std-s1",
     "status": "decided" | "incomplete",
     "overall": {"winner": "<variant>", "verdict": "repair" | "reject", "eligible": [...]},
     "families": {"wiseft": {"winner", "verdict", "eligible"}, "resta": {...}, "cast": {...}},
     "frozen": {"wiseft": "<seed-1 setting name>", "resta": ..., "cast": ...},
     "table": [...], "freeze": {...}, "missing": {...}, "problems": [...], ...}

    While ``status`` is "incomplete" there is no ``overall`` or ``frozen`` and ``families`` is the list of
    family names: nothing is frozen, and P08 publishes no model or parameters. ``frozen`` names each family's
    setting on the seed-1 adapter; seeds 2 and 3 are the same setting on ``adapter-...-s2`` and ``-s3``
    (P05's ``seed_name``), so every frozen variant is ``frozen_variants``. Only ``overall.winner`` is
    published as "the repaired model": the card's repair effect is the winner's.

``final_report.json`` (P07, ``lift stats`` / ``lift report``: ``lift.report.tables.build``), version 1:

    {"version": 1, "dtype": "bf16", "primary_judge": "<judge id of the final judgements>" | null,
     "selection": {"winner": "<overall winner>", "frozen": {family: setting},
                   "sha256": "<sha256 of the selection.json bytes the report was built from>", ...},
     "verdicts": [{"adapter": "<seed-1 adapter>", "recipe": "standard" | "stress",
                   "verdict": "retain" | "repair" | "reject" | null, "reason": "<one sentence>"}],
     "primary": {"drift": <Effect>, "repair": <Effect>,
                 "redteam_drift": <Effect>, "redteam_repair": <Effect>},
     "headline": {"utility_retained": <Interval>, "belebele_hi_change": <Interval>,
                  "overrefusal_hard_hi_change": <Interval>,
                  "gpu_hours_per_checkup": <number>,         # dev sweep + final battery of stock, the
                                                              # seed-1 adapter and the seed-1 winner
                  "gpu_hours_per_variant_final": <number>},   # mean final battery per variant
     "tables": {"<name>": {"title": str, "columns": [{"key": str, "label": str, "format": FORMAT}],
                           "rows": [{"<key>": value, ...}], "note": str}},
     "not_run": ["<arm that was not run>", ...]}

    Effect   = {"pooled": {"est", "lo", "hi", "p", "n", "seeds"} | null,
                "per_seed": [{"seed", "est", "lo", "hi", "p", "n"}]}
               (rates are fractions; est is the second variant minus the first: adapter - stock for drift,
               repair - adapter for repair; ``pooled`` null and ``per_seed`` empty when the arm was not run)
    Interval = {"est", "lo", "hi"}; any value may be null: "not run".

    ``primary_judge`` is the judge id written in the final judgement records (it follows the
    decide-judge rule; never assume which judge it is). TABLES names the tables P08 places: P07's
    ``lift.report.tables.TABLES`` (a missing one renders as pending), plus the OPTIONAL ones no stage is
    committed to write (``signatures``: P06's diagnosis, which may never run; when final_report.json exists
    without it, it renders as "not run", like an arm not run, and is not pending). A cell is a number, a
    string, an Interval (formats ending in ``_ci``) or null ("not run"). Every rendered value, text cells
    and table notes included, is recorded in the sources.

The report is bound to the selection (review XB-04): every read of final_report.json first checks that
its ``selection`` block names the winner and frozen map of the selection.json on this machine and the
sha256 of its bytes, and, when P05's committed ``configs/frozen_recipe.json`` exists (``{"version": 1,
"adapter", "selection_sha256", "overall_winner", "frozen"}``), that both agree with it; selection.json
is checked against that recipe too, report or not. Any disagreement (a ``lift select`` after ``lift
stats``, a stale pull, a report made before the block existed) refuses everything that reads them
(``PublishError``): the numbers would be another winner's.

Formats: pct (0.123 -> 12.3%), pp (signed points), points (0.05 -> 5 points), pct_ci, pp_ci, p, int,
f2, f3, hours, ms, text, list ([1, 2] -> "1, 2"). A null anywhere on a value's path (``pooled`` of an arm
not run) renders as "not run"; only a missing key renders as pending.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lift import config, paths
from lift.publish import PublishError

REPORT_VERSION = 1
PENDING = "[PENDING"
P05 = "P05 lift select"
P07 = "P07 lift stats / lift report"
P06 = "P06 diagnosis"
# The tables final_report.json carries, with what each one is for (and so what a pending one stands for).
TABLES = {
    "per_language": "per-language before/after table (harmful rate with the unclear bound, intervals)",
    "repair_comparison": "repair comparison table (every frozen family, final split)",
    "secondary": "secondary claims, Holm-corrected (zh+es transfer, English)",
    "exploratory": "exploratory claims, BH-FDR (ar, th, per-language effects)",
    "overrefusal": "easy and hard over-refusal per language",
    "utility": "utility win rate against stock and the share of the adapter's gain retained",
    "belebele": "Belebele accuracy per language",
    "judge_sensitivity": "judge sensitivity (strict vs loose, second judge, back-translation)",
    "cost": "cost: GPU-hours per check-up and per-token latency per repair",
    "signatures": "exploratory diagnostic signatures (P06)",
}
# Tables no stage is committed to write: absent from an existing final_report.json, they render as not run
# (the reason below), not as pending, so they never block ``--yes``. P07's TABLES has no signatures, and
# P06 (the exploratory diagnosis) may never be built; P08 invents no contract for it.
OPTIONAL = {"signatures": "the exploratory diagnosis (P06) put no signatures in final_report.json"}
NOT_RUN = "not run"


def selection_path() -> Path:
    return paths.metrics_dir() / "selection.json"


def report_path() -> Path:
    return paths.metrics_dir() / "final_report.json"


def frozen_recipe_path() -> Path:
    """P05's committed record of the frozen recipe (review XB-02), once ``lift select`` froze it."""
    return paths.CONFIG_DIR / "frozen_recipe.json"


RECIPE_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RESTATS = (
    "re-run lift stats and lift report (P07) on the current selection, or restore the selection.json "
    "they were built from (lift sync pull)"
)


def shown(path: Path) -> str:
    """A path as the public text shows it: relative to the repo or output tree, never a home dir."""
    path = Path(path)
    for root, label in ((paths.output_dir(), "outputs"), (paths.data_dir(), "data"), (paths.REPO_ROOT, "")):
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        return f"{label}/{rel}" if label else rel
    return path.name


# ---------------------------------------------------------------------------------------------- formats


def _num(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"not a finite number: {value!r}")
    return float(value)


def _interval(value: Any) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        return _num(value["est"]), _num(value["lo"]), _num(value["hi"])
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _num(value[0]), _num(value[1]), _num(value[2])
    raise ValueError(f"not an interval {{est, lo, hi}}: {value!r}")


def fmt(value: Any, kind: str) -> str:
    """One value as the public text shows it. None is "not run" (an arm never estimated)."""
    if value is None:
        return NOT_RUN
    if kind == "text":
        return str(value)
    if kind == "pct":
        return f"{_num(value) * 100:.1f}%"
    if kind == "pp":
        return f"{_num(value) * 100:+.1f} pp"
    if kind == "pct_ci":
        est, lo, hi = _interval(value)
        return f"{est * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    if kind == "pp_ci":
        est, lo, hi = _interval(value)
        return f"{est * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}]"
    if kind == "points":
        return f"{_num(value) * 100:g} points"
    if kind == "p":
        p = _num(value)
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    if kind == "int":
        n = _num(value)
        if n != int(n):
            raise ValueError(f"not an integer: {value!r}")
        return f"{int(n):,}"
    if kind == "f2":
        return f"{_num(value):.2f}"
    if kind == "f3":
        return f"{_num(value):.3f}"
    if kind == "hours":
        return f"{_num(value):.2f} h"
    if kind == "ms":
        return f"{_num(value):.1f} ms"
    if kind == "list":
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"not a list: {value!r}")
        return ", ".join(str(x) for x in value) or NOT_RUN
    raise ValueError(f"unknown format {kind!r}")


def resolve(doc: Any, pointer: str) -> Any:
    """``/a/b/0/c`` in ``doc``; KeyError when any step is missing. A null on the way (an arm not run, whose
    ``pooled`` P07 writes as null) resolves to None: everything under it is "not run"."""
    node = doc
    for part in [p for p in pointer.split("/") if p]:
        if node is None:
            return None
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(pointer)
    return node


def pending(what: str, source: str, stage: str) -> str:
    return f"{PENDING}: {what} from {source} ({stage})]"


# ---------------------------------------------------------------------------------------------- Facts


@dataclass
class Facts:
    """Reads the inputs once and records every value it renders."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    _docs: dict[Path, Any] = field(default_factory=dict)

    def doc(self, path: Path) -> Any | None:
        """A JSON input (None when absent or empty). final_report.json is checked (its version, and that
        it is bound to this selection: ``check_report``) before any value of it is used."""
        path = Path(path)
        if path not in self._docs:
            if not path.is_file() or path.stat().st_size == 0:
                self._docs[path] = None
            else:
                try:
                    self._docs[path] = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise PublishError(
                        f"{shown(path)} is not valid JSON ({exc.msg}, line {exc.lineno})"
                    ) from exc
                if path == report_path():
                    try:
                        check_report(self, self._docs[path])
                    except PublishError:
                        del self._docs[path]  # refused every time it is asked for, never half-read
                        raise
        return self._docs[path]

    def report(self) -> dict | None:
        return self.doc(report_path())

    def selection(self) -> dict | None:
        return self.doc(selection_path())

    def raw(self, path: Path, pointer: str) -> Any | None:
        """A value used for logic (which sentence to show), not printed; None when missing."""
        doc = self.doc(path)
        if doc is None:
            return None
        try:
            return resolve(doc, pointer)
        except KeyError:
            return None

    def _missing(self, what: str, path: Path, stage: str) -> str:
        self.missing.append(f"{what} ({shown(path)})")
        return pending(what, shown(path), stage)

    def value(self, path: Path, pointer: str, kind: str, *, what: str, stage: str) -> str:
        """One number (or text) from a JSON file, formatted, with its source recorded."""
        doc = self.doc(path)
        try:
            value = resolve(doc, pointer) if doc is not None else None
            if doc is None:
                raise KeyError(pointer)
        except KeyError:
            return self._missing(what, path, stage)
        try:
            out = fmt(value, kind)
        except (ValueError, KeyError, TypeError) as exc:
            raise PublishError(f"{shown(path)}{pointer}: {exc}") from exc
        self.sources.append({"shown": out, "file": shown(path), "pointer": pointer, "value": value})
        return out

    def report_value(self, pointer: str, kind: str, *, what: str) -> str:
        self.report()  # checks the version
        return self.value(report_path(), pointer, kind, what=what, stage=P07)

    def config(self, name: str, dotted: str, kind: str) -> str:
        """A pre-registered design number from configs/{name}.yaml."""
        value = config.get(dotted, name=name)
        out = fmt(value, kind)
        self.sources.append(
            {
                "shown": out,
                "file": f"configs/{name}.yaml",
                "pointer": "/" + dotted.replace(".", "/"),
                "value": value,
            }
        )
        return out

    def table(self, name: str) -> str:
        """A report table as markdown, or a pending line."""
        from lift.publish.render import markdown_table

        report = self.report()
        spec = (report or {}).get("tables", {}).get(name)
        if spec is None:
            if report is not None and name in OPTIONAL:
                self.sources.append(
                    {
                        "shown": NOT_RUN,
                        "file": shown(report_path()),
                        "pointer": f"/tables/{name}",
                        "value": None,
                    }
                )
                return f"_{NOT_RUN.capitalize()}: {OPTIONAL[name]}._"
            stage = P06 if name == "signatures" else P07
            return f"_{self._missing(TABLES.get(name, name), report_path(), stage)}_"
        try:
            columns = spec["columns"]
            rows = spec["rows"]
            cells = [[fmt(row.get(c["key"]), c.get("format", "text")) for c in columns] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise PublishError(f"{shown(report_path())}/tables/{name}: {exc}") from exc
        # Every cell, text included: P07's free text goes public too, so a number inside it is audited.
        for i, row in enumerate(rows):
            for c, shown_cell in zip(columns, cells[i], strict=True):
                self.sources.append(
                    {
                        "shown": shown_cell,
                        "file": shown(report_path()),
                        "pointer": f"/tables/{name}/rows/{i}/{c['key']}",
                        "value": row.get(c["key"]),
                    }
                )
        out = markdown_table([c["label"] for c in columns], cells)
        note = spec.get("note")
        if note:
            self.sources.append(
                {
                    "shown": str(note),
                    "file": shown(report_path()),
                    "pointer": f"/tables/{name}/note",
                    "value": note,
                }
            )
        return f"{out}\n\n{note}" if note else out

    def per_seed_table(self, effects: list[tuple[str, str]]) -> str:
        """Each seed of the primary effects (``[(label, key of primary)]``) as a markdown table: the
        estimate with its interval, McNemar's p and n, every value recorded; an arm not run says so."""
        from lift.publish.render import markdown_table

        if self.report() is None:
            return f"_{self._missing('the per-seed effects', report_path(), P07)}_"
        rows = []
        for label, key in effects:
            seeds = self.raw(report_path(), f"/primary/{key}/per_seed")
            if not seeds:
                if self.raw(report_path(), f"/primary/{key}") is None:
                    rows.append([label, self._missing(f"the per-seed {key} effect", report_path(), P07)])
                    continue
                rows.append([label, NOT_RUN, NOT_RUN, NOT_RUN, NOT_RUN])
                continue
            for i in range(len(seeds)):
                at = f"/primary/{key}/per_seed/{i}"
                rows.append(
                    [
                        label,
                        self.report_value(f"{at}/seed", "int", what=f"the seed of a {key} row"),
                        self.report_value(at, "pp_ci", what=f"a per-seed {key} effect"),
                        self.report_value(f"{at}/p", "p", what=f"a per-seed {key} p-value"),
                        self.report_value(f"{at}/n", "int", what=f"a per-seed {key} n"),
                    ]
                )
        rows = [r + [""] * (5 - len(r)) for r in rows]
        return markdown_table(["Effect", "Seed", "Estimate [95% interval]", "McNemar p", "n"], rows)

    def table_rows(self, name: str) -> list[dict] | None:
        report = self.report()
        spec = (report or {}).get("tables", {}).get(name)
        return None if spec is None else list(spec.get("rows") or [])

    def write_sources(self, path: Path) -> Path:
        """``*.sources.json``: every rendered value with its file and pointer, and what was pending."""
        from lift.io import stable_json

        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"values": self.sources, "pending": sorted(set(self.missing))}
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(json.loads(stable_json(doc)), indent=1, ensure_ascii=False) + "\n")
        return path


# ---------------------------------------------------------------------------------------------- frozen


SEEDS = (1, 2, 3)
# A repaired variant's name (P05's lift.repair.settings): {setting tag}-{adapter}, adapter ending -s<seed>.
_SETTING_NAME = re.compile(
    r"^(?P<tag>(?:wiseft-l|resta-a|castalways-m|cast-m)\d+\.\d+)-(?P<stem>adapter-.+-s)(?P<seed>\d+)$"
)


def seed_variant(name: str, seed: int) -> str:
    """The same frozen setting on another seed's adapter (P05's ``seed_name`` applied to the setting)."""
    m = _SETTING_NAME.match(name)
    if m is None:
        raise PublishError(
            f"{shown(selection_path())}: {name!r} is not a repair setting name (lift.repair.settings scheme)"
        )
    return f"{m.group('tag')}-{m.group('stem')}{seed}"


def _selection_sha() -> str | None:
    path = selection_path()
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _recipe(facts: Facts) -> dict | None:
    """configs/frozen_recipe.json checked to be in P05's shape, or None while there is none."""
    path = frozen_recipe_path()
    if not path.is_file():
        return None
    doc = facts.doc(path)
    where = f"{shown(path)} (the committed frozen recipe)"
    if not isinstance(doc, Mapping) or doc.get("version") != RECIPE_VERSION:
        raise PublishError(f"{where} is not version {RECIPE_VERSION} of P05's recipe record")
    problems = [k for k in ("adapter", "overall_winner") if not isinstance(doc.get(k), str) or not doc[k]]
    if not isinstance(doc.get("selection_sha256"), str) or not _SHA256.match(doc["selection_sha256"]):
        problems.append("selection_sha256")
    if not isinstance(doc.get("frozen"), Mapping) or not doc["frozen"]:
        problems.append("frozen")
    if problems:
        raise PublishError(f"{where} lacks or garbles {', '.join(problems)}")
    return dict(doc)


def _recipe_check(
    facts: Facts, *, sha: str | None, winner: Any, frozen: Any, adapter: Any, what: str
) -> None:
    """Refuse when ``what`` (selection.json or the report's selection block) disagrees with the
    committed frozen recipe, if there is one."""
    recipe = _recipe(facts)
    if recipe is None:
        return
    diffs = []
    if sha != recipe["selection_sha256"]:
        diffs.append(f"selection sha256 {str(sha)[:12]} vs {recipe['selection_sha256'][:12]}")
    if winner != recipe["overall_winner"]:
        diffs.append(f"winner {winner} vs {recipe['overall_winner']}")
    if not isinstance(frozen, Mapping) or dict(frozen) != dict(recipe["frozen"]):
        diffs.append("the frozen map")
    if adapter is not None and adapter != recipe["adapter"]:
        diffs.append(f"adapter {adapter} vs {recipe['adapter']}")
    if diffs:
        raise PublishError(
            f"{what} disagrees with {shown(frozen_recipe_path())}, the committed frozen recipe "
            f"({'; '.join(diffs)}): publish only the recipe that was frozen and evaluated. Restore that "
            f"selection.json (lift sync pull), or {RESTATS}"
        )


def check_report(facts: Facts, report: Any) -> None:
    """final_report.json must be P08's version and bound to the selection.json here (and to the frozen
    recipe, when committed): its ``selection`` block's winner, frozen map and sha256 of the selection.json
    bytes. PublishError otherwise; nothing that reads the report may render from another selection."""
    where = shown(report_path())
    if not isinstance(report, Mapping) or report.get("version") != REPORT_VERSION:
        version = report.get("version") if isinstance(report, Mapping) else None
        raise PublishError(
            f"{where} has version {version!r}; P08 reads version {REPORT_VERSION} (lift.publish.inputs)"
        )
    block = report.get("selection")
    winner = block.get("winner") if isinstance(block, Mapping) else None
    frozen = block.get("frozen") if isinstance(block, Mapping) else None
    sha = block.get("sha256") if isinstance(block, Mapping) else None
    if (
        not isinstance(winner, str)
        or not winner
        or not isinstance(frozen, Mapping)
        or not frozen
        or not isinstance(sha, str)
        or not _SHA256.match(sha)
    ):
        raise PublishError(
            f"{where} does not name the selection it was built from (selection.winner, selection.frozen and "
            f"selection.sha256): it predates that record or is garbled; {RESTATS}"
        )
    found = _selection_sha()
    if found is None:
        raise PublishError(
            f"{where} was built from a selection.json with sha256 {sha[:12]} (winner {winner}), but "
            f"{shown(selection_path())} is not here: lift sync pull"
        )
    sel = facts.selection()
    overall = sel.get("overall") if isinstance(sel, Mapping) else None
    now_winner = overall.get("winner") if isinstance(overall, Mapping) else None
    now_frozen = sel.get("frozen") if isinstance(sel, Mapping) else None
    status = sel.get("status") if isinstance(sel, Mapping) else None
    if found != sha or status != "decided" or now_winner != winner or now_frozen != dict(frozen):
        raise PublishError(
            f"{where} is not this selection's: it was built from selection.json sha256 {sha[:12]} (winner "
            f"{winner}), and {shown(selection_path())} now has sha256 {found[:12]} (status {status}, "
            f"winner {now_winner}{'' if now_frozen == dict(frozen) else ', another frozen map'}). Its "
            f"numbers would be published under another winner; {RESTATS}"
        )
    _recipe_check(
        facts,
        sha=sha,
        winner=winner,
        frozen=frozen,
        adapter=sel.get("adapter") if isinstance(sel, Mapping) else None,
        what=f"{where}'s selection",
    )


def selection_problem(facts: Facts) -> str | None:
    """Why selection.json freezes nothing yet (missing, or not decided), or None when it is decided."""
    sel = facts.selection()
    if sel is None:
        return f"no {shown(selection_path())} yet ({P05})"
    status = sel.get("status")
    if status != "decided":
        return (
            f"{shown(selection_path())} is {status or 'without a status'!s}, not decided: nothing is frozen "
            f"until every input of {P05} is there"
        )
    return None


def _decided(facts: Facts) -> dict | None:
    """selection.json when it is decided, checked to be in P05's shape; None otherwise."""
    if selection_problem(facts) is not None:
        return None
    sel = facts.selection()
    overall, frozen = sel.get("overall"), sel.get("frozen")
    if not isinstance(overall, Mapping) or not overall.get("winner"):
        raise PublishError(f"{shown(selection_path())} is decided but has no overall.winner (P05's shape)")
    if not isinstance(frozen, Mapping) or not frozen:
        raise PublishError(
            f"{shown(selection_path())} is decided but frozen is not P05's {{family: seed-1 setting}} mapping"
        )
    _recipe_check(
        facts,
        sha=_selection_sha(),
        winner=overall.get("winner"),
        frozen=frozen,
        adapter=sel.get("adapter"),
        what=shown(selection_path()),
    )
    return sel


def frozen_winner(facts: Facts) -> str | None:
    """The overall frozen winner (seed 1), from a decided selection.json; None until then."""
    sel = _decided(facts)
    return None if sel is None else str(sel["overall"]["winner"])


def selection_verdict(facts: Facts) -> str | None:
    """The selection's verdict ("repair", or "reject": nothing eligible, the best setting frozen anyway)."""
    sel = _decided(facts)
    return None if sel is None else sel["overall"].get("verdict")


def frozen_family_winners(facts: Facts) -> dict[str, str]:
    """{family: frozen seed-1 setting} of a decided selection.json ({} until then)."""
    sel = _decided(facts)
    return {} if sel is None else {str(k): str(v) for k, v in sel["frozen"].items()}


def frozen_variants(facts: Facts) -> list[str]:
    """Every frozen variant: each family's frozen setting on seeds 1-3 (the final eval's 9 repaired
    variants), in family then seed order."""
    out = [seed_variant(name, seed) for name in frozen_family_winners(facts).values() for seed in SEEDS]
    return list(dict.fromkeys(out))


def has_pending(text: str) -> bool:
    return PENDING in text
