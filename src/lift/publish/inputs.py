"""What P08 reads, and the ``Facts`` reader that renders every number with its source.

No number in a public artifact is typed by hand: each one is read here from a JSON file under
``outputs/metrics`` (results) or ``configs/`` (the pre-registered design: set sizes, thresholds), and
``Facts.sources`` lists every value with its file and JSON pointer, which ``lift publish`` writes
beside each rendered file (``*.sources.json``) for the claims audit (R3). An input that does not exist
yet renders as ``[PENDING: <what> from <file> (<stage>)]``; ``lift publish ... --yes`` refuses to upload
a text that still holds one.

Inputs (paths under ``outputs/metrics``; both are written by stages that come after P08's first draft,
so the shapes below are the contract they must write):

``selection.json`` (P05, ``lift select``): ``lift.selection.Selection`` as JSON plus the families.

    {"adapter": "adapter-indicalign-std-s1", "winner": "<variant>", "verdict": "repair" | "reject",
     "eligible": [...], "table": [<Selection.table rows>],
     "families": {"wiseft": {"winner": "<variant>", "verdict": ...}, "resta": {...}, "cast": {...}},
     "frozen": ["<every frozen variant name, seeds 1-3>"]}             # families and frozen optional

``final_report.json`` (P07, ``lift stats`` / ``lift report``), version 1:

    {"version": 1, "dtype": "bf16", "primary_judge": "<judge id>",
     "verdicts": [{"adapter": "<seed-1 adapter or recipe>", "recipe": "standard" | "stress",
                   "verdict": "retain" | "repair" | "reject", "reason": "<one sentence>"}],
     "primary": {"drift": <Effect>, "repair": <Effect>,
                 "redteam_drift": <Effect>, "redteam_repair": <Effect>},
     "headline": {"utility_retained": <Interval>, "belebele_hi_change": <Interval>,
                  "overrefusal_hard_hi_change": <Interval>, "gpu_hours_per_checkup": <number>},
     "tables": {"<name>": {"title": str, "columns": [{"key": str, "label": str, "format": FORMAT}],
                           "rows": [{"<key>": value, ...}], "note": str}},
     "not_run": ["<arm that was not run>", ...]}

    Effect   = {"pooled": {"est", "lo", "hi", "p"}, "per_seed": [{"seed", "est", "lo", "hi", "p", "n"}]}
               (rates are fractions; est is the first variant minus the second: adapter - stock for drift,
               repair - adapter for repair)
    Interval = {"est", "lo", "hi"}

    TABLES names the tables P08 places (a missing one renders as pending). A cell is a number, a string,
    an Interval (formats ending in ``_ci``) or null ("not run").

Formats: pct (0.123 -> 12.3%), pp (signed points), points (0.05 -> 5 points), pct_ci, pp_ci, p, int,
f2, f3, hours, ms, text.
"""

from __future__ import annotations

import json
import math
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
NOT_RUN = "not run"


def selection_path() -> Path:
    return paths.metrics_dir() / "selection.json"


def report_path() -> Path:
    return paths.metrics_dir() / "final_report.json"


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
    raise ValueError(f"unknown format {kind!r}")


def resolve(doc: Any, pointer: str) -> Any:
    """``/a/b/0/c`` in ``doc``; KeyError when any step is missing."""
    node = doc
    for part in [p for p in pointer.split("/") if p]:
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
        return self._docs[path]

    def report(self) -> dict | None:
        doc = self.doc(report_path())
        if doc is not None and doc.get("version") != REPORT_VERSION:
            raise PublishError(
                f"{shown(report_path())} has version {doc.get('version')!r}; P08 reads version "
                f"{REPORT_VERSION} (lift.publish.inputs)"
            )
        return doc

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
            stage = P06 if name == "signatures" else P07
            return f"_{self._missing(TABLES.get(name, name), report_path(), stage)}_"
        try:
            columns = spec["columns"]
            rows = spec["rows"]
            cells = [[fmt(row.get(c["key"]), c.get("format", "text")) for c in columns] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise PublishError(f"{shown(report_path())}/tables/{name}: {exc}") from exc
        for i, row in enumerate(rows):
            for c, shown_cell in zip(columns, cells[i], strict=True):
                if c.get("format", "text") != "text":
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
        return f"{out}\n\n{note}" if note else out

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


def frozen_winner(facts: Facts) -> str | None:
    """The overall frozen winner (seed 1), from selection.json."""
    sel = facts.selection()
    return None if sel is None else sel.get("winner")


def frozen_family_winners(facts: Facts) -> dict[str, str]:
    """{method: frozen variant} for every family selection.json lists (the overall winner at least)."""
    sel = facts.selection() or {}
    out = {
        fam: entry["winner"]
        for fam, entry in (sel.get("families") or {}).items()
        if isinstance(entry, Mapping) and entry.get("winner")
    }
    if sel.get("winner") and sel["winner"] not in out.values():
        out.setdefault("winner", sel["winner"])
    return out


def has_pending(text: str) -> bool:
    return PENDING in text
