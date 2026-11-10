"""``lift sweep --adapter A --breakdown``: the dev sweep's labels by flag and by CAST's gate (laptop).

Counts only, from the checked dev records ``lift select`` reads; no torch, no API. Decisions item 87.

Why it exists. A cell's rates do not say two things the Nov 10 sweep made matter:

- The labelling rules put the judge's "harmful" above every unclear flag (lift.judge.labels: harmful wins
  over everything). An output that runs to the token cap, repeats itself or is in no language, and that the
  judge calls unsafe, is counted harmful. Nothing in ``harmful_rate`` says how many harmful labels are of
  that kind. Here each final label is counted against the three mechanical flags (too_short, repetitive,
  garbled), the language flags, the token cap and the output's length, and "harmful, intact" is a harmful
  label on an output that trips none of the three flags.
- CAST's cost summary gives one firing rate over every prompt of a setting. Here the gate is counted per
  set and language, against the label the unrepaired adapter got on the same prompt, and the steered
  outputs (gate fired) are counted apart from the rest.

Each setting is also paired with the adapter by prompt (``pipeline.paired_labels``, the one door to a
paired read): the adapter's label against the setting's.

What is read: stock, the adapter, the other registered seed-1 adapters (``also``; default all of them)
and every setting of the chosen families, on the three safety sets in six languages. A cell that cannot be
read is listed with the reader's reason and left out; the report then says "incomplete" and the command
exits 3. Nothing is selected or frozen here and no rule reads the file: it is a description of the dev
records, written to outputs/metrics/sweep_breakdown.json and .md. It holds counts, variant names and judge
ids, never a prompt or an output, so it can go to the results branch with the other aggregate metrics.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import metrics_dir
from lift.repair import settings as rs
from lift.schemas import LANGS

REPORT = "sweep_breakdown"
VERSION = 1
SPLIT = "dev"
SETS = ("harmful", "easy_benign", "hard_benign")
LABELS = ("harmful", "refusal", "benign", "unclear")
LANGUAGE_FLAGS = ("wrong_lang", "code_mixed")


class Missing(ValueError):
    """A cell that cannot be read for reporting; the message is the reader's reason."""


def report_paths() -> tuple[Path, Path]:
    return metrics_dir() / f"{REPORT}.json", metrics_dir() / f"{REPORT}.md"


def unclear_flags() -> tuple[str, ...]:
    """The three mechanical flags that make an output unclear when the judge did not call it harmful."""
    from lift.report.analysis import UNCLEAR_FLAGS

    return tuple(UNCLEAR_FLAGS)


def _short(exc: Any, n: int = 300) -> str:
    text = str(exc)
    return text if len(text) <= n else text[: n - 3] + "..."


def _median(values: Sequence[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def read_cell(variant: str, set_name: str, lang: str, dtype: str) -> list[dict[str, Any]]:
    """One row per prompt of a dev cell, from its checked judgements and its generations: pid, the final
    label, the judge's own call, the flags, hit_cap, n_new_tokens and the gate's decision (None for a model
    without a gate). No text is read into the rows. Raises Missing with the reader's reason."""
    from lift.judge import pipeline
    from lift.models.generate import generations_path, read_generations
    from lift.repair.select import _stale

    path = generations_path(variant, set_name, lang, SPLIT)
    stale = _stale(variant, path)
    if stale:
        raise Missing(stale)
    try:
        records = pipeline.load_records(variant, set_name, lang, SPLIT, judge="primary", complete=True,
                                        dtype=dtype)
        gens = {g.key: g for g in read_generations(path)}
    except (OSError, ValueError) as exc:
        raise Missing(_short(exc)) from None
    rows = []
    for r in records:
        g = gens[r.key]
        gate = g.meta.get("gate_fired")
        rows.append({
            "pid": r.pid, "label": r.label, "judge_label": r.judge_label, "judge": r.judge,
            **{f: bool(getattr(r.flags, f)) for f in (*unclear_flags(), *LANGUAGE_FLAGS)},
            "hit_cap": bool(g.hit_cap), "n_new_tokens": int(g.n_new_tokens),
            "gate_fired": None if gate is None else bool(gate),
        })
    return rows


def _flagged(row: Mapping[str, Any]) -> bool:
    return any(row[f] for f in unclear_flags())


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts over ``rows``: how many trip each flag, any of the three unclear flags, the cap, and their
    median length; with a gate, how many it fired on."""
    out: dict[str, Any] = {"n": len(rows)}
    for f in (*unclear_flags(), *LANGUAGE_FLAGS):
        out[f] = sum(bool(r[f]) for r in rows)
    out["flagged"] = sum(_flagged(r) for r in rows)
    out["hit_cap"] = sum(bool(r["hit_cap"]) for r in rows)
    out["median_new_tokens"] = _median([r["n_new_tokens"] for r in rows])
    if any(r["gate_fired"] is not None for r in rows):
        out["gate_fired"] = sum(bool(r["gate_fired"]) for r in rows)
    return out


def cell_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts of one cell (module docstring). ``harmful_intact`` is the number of harmful labels on
    outputs that trip no unclear flag; ``steered`` and ``unsteered`` split a gated cell by the gate."""
    n = len(rows)
    labels = Counter(r["label"] for r in rows)
    by_label = {lab: _group([r for r in rows if r["label"] == lab]) for lab in LABELS}
    out: dict[str, Any] = {
        "n": n,
        "labels": {lab: labels.get(lab, 0) for lab in LABELS},
        "judge_labels": {lab: sum(r["judge_label"] == lab for r in rows) for lab in LABELS},
        "by_label": by_label,
        "harmful_intact": by_label["harmful"]["n"] - by_label["harmful"]["flagged"],
        "judges": sorted({str(r["judge"]) for r in rows}),
    }
    gated = [r for r in rows if r["gate_fired"] is not None]
    if gated:
        if len(gated) != n:
            raise Missing(f"{len(gated)} of {n} records carry a gate decision: one cell is gated on every "
                          "prompt or on none")
        out["gate_fired"] = sum(bool(r["gate_fired"]) for r in rows)
        for name, want in (("steered", True), ("unsteered", False)):
            part = [r for r in rows if bool(r["gate_fired"]) is want]
            out[name] = {**_group(part),
                         "labels": {lab: sum(r["label"] == lab for r in part) for lab in LABELS},
                         "harmful_intact": sum(r["label"] == "harmful" and not _flagged(r) for r in part)}
    return out


def pair_counts(adapter_rows: Sequence[Mapping[str, Any]],
                rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A setting against the adapter on the same prompts: {adapter label: {setting label: n}}, and for a
    gated setting {adapter label: {"n", "fired"}}: where the gate fired, by what the unrepaired adapter did
    with the prompt."""
    a = {r["pid"]: r for r in adapter_rows}
    s = {r["pid"]: r for r in rows}
    if set(a) != set(s):
        raise Missing("the adapter's cell and the setting's cover different prompts")
    table = {x: {y: 0 for y in LABELS} for x in LABELS}
    gate: dict[str, dict[str, int]] | None = None
    if any(r["gate_fired"] is not None for r in rows):
        gate = {x: {"n": 0, "fired": 0} for x in LABELS}
    for pid, row in s.items():
        before = a[pid]["label"]
        table[before][row["label"]] += 1
        if gate is not None:
            gate[before]["n"] += 1
            gate[before]["fired"] += int(bool(row["gate_fired"]))
    return {"n": len(s), "table": table, **({"gate": gate} if gate is not None else {})}


def other_adapters(adapter: str) -> list[str]:
    """The other registered adapters of seed 1 (their dev cells exist beside the swept adapter's)."""
    from lift.models.variants import load_registry

    return sorted(name for name, spec in load_registry().items()
                  if spec.method == "adapter" and spec.seed == 1 and name != adapter)


def collect(adapter: str, families: Sequence[str], dtype: str, *,
            also: Sequence[str] | None = None) -> dict[str, Any]:
    """Every readable cell's counts and every setting's pair with the adapter (module docstring)."""
    from lift.judge import pipeline

    settings = rs.settings(families)
    extra = list(also) if also is not None else other_adapters(adapter)
    names = [("stock", "stock"), (adapter, "adapter"), *((x, "adapter") for x in extra if x != adapter),
             *((s.name(adapter), s.method) for s in settings)]
    cells: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    missing: dict[str, str] = {}
    rows_of: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for name, method in names:
        for set_name in SETS:
            for lang in LANGS:
                where = f"{name}/{set_name}_{lang}_{SPLIT}"
                try:
                    rows = read_cell(name, set_name, lang, dtype)
                    counts = cell_counts(rows)
                except Missing as exc:
                    missing[where] = str(exc)
                    continue
                rows_of[(name, set_name, lang)] = rows
                cells.append({"variant": name, "method": method, "set": set_name, "lang": lang, **counts})
    for s in settings:
        name = s.name(adapter)
        for set_name in SETS:
            for lang in LANGS:
                ours, theirs = rows_of.get((name, set_name, lang)), rows_of.get((adapter, set_name, lang))
                if ours is None or theirs is None:
                    continue            # the cell itself is listed as missing
                where = f"{name} against {adapter} on {set_name}_{lang}_{SPLIT}"
                try:
                    pipeline.paired_labels(adapter, name, set_name, lang, SPLIT, judge="primary", dtype=dtype)
                    counts = pair_counts(theirs, ours)
                except (OSError, ValueError) as exc:
                    missing[where] = _short(exc)
                    continue
                pairs.append({"variant": name, "method": s.method, "set": set_name, "lang": lang, **counts})
    return {"variants": [{"name": n, "method": m} for n, m in names], "cells": cells, "pairs": pairs,
            "missing": missing}


def run(adapter: str, *, families: Sequence[str] = rs.FAMILIES, dtype: str | None = None,
        also: Sequence[str] | None = None) -> dict[str, Any]:
    """Collect and write outputs/metrics/sweep_breakdown.json and .md; the doc written."""
    dtype = dtype or str(config.get("generation.dtype"))
    got = collect(adapter, families, dtype, also=also)
    doc: dict[str, Any] = {
        "kind": REPORT, "version": VERSION, "adapter": adapter, "split": SPLIT, "dtype": dtype,
        "judge": "primary", "families": list(families), "unclear_flags": list(unclear_flags()),
        "decision": "docs/decisions.md item 87", "created": datetime.now(UTC).isoformat(),
        "status": "incomplete" if got["missing"] else "complete", **got,
    }
    json_path, md_path = report_paths()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((json_path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n"),
                       (md_path, render(doc))):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    return doc


# ------------------------------------------------------------------------------------------- rendering


def _pct(k: int, n: int) -> str:
    return "n/a" if not n else f"{100 * k / n:.1f}"


def _shown(name: str, adapter: str) -> str:
    """A setting by its tag (wiseft-l0.25); stock and the adapters by name."""
    suffix = f"-{adapter}"
    return name[: -len(suffix)] if name.endswith(suffix) and name != adapter else name


def _index(doc: Mapping[str, Any], key: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(c["variant"], c["set"], c["lang"]): c for c in doc[key]}


def _tokens(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _gate_signature(doc: Mapping[str, Any], name: str) -> str:
    """Where one gated setting's gate fired, as text to compare settings by: its count in every cell and
    against every adapter label."""
    fired = [(c["set"], c["lang"], c["n"], c.get("gate_fired")) for c in doc["cells"] if c["variant"] == name]
    against = [(p["set"], p["lang"], p.get("gate")) for p in doc["pairs"] if p["variant"] == name]
    return json.dumps([sorted(fired), sorted(against, key=lambda x: x[:2])], sort_keys=True)


def render(doc: Mapping[str, Any]) -> str:
    adapter = doc["adapter"]
    cells, pairs = _index(doc, "cells"), _index(doc, "pairs")
    names = [v["name"] for v in doc["variants"]]
    gated = [v["name"] for v in doc["variants"] if v["method"] == "cast"]
    # One artifact gates every strength, on the prompt before anything is steered: when the settings' gates
    # agree prompt count for prompt count, one is shown for all of them.
    one_gate = len(gated) > 1 and len({_gate_signature(doc, name) for name in gated}) == 1
    gate_note = ""
    if one_gate:
        gate_note = (f" The gate fired on the same counts at every strength ({len(gated)} settings, one "
                     "artifact): one is shown.")
        gated = gated[:1]
    steered = [v["name"] for v in doc["variants"] if v["method"] in ("cast", "cast_always")]
    settings = [v["name"] for v in doc["variants"] if v["method"] not in ("stock", "adapter")]
    flags = list(doc["unclear_flags"])
    lines = [
        f"# Dev sweep breakdown: {adapter} (seed-1 dev, {doc['dtype']})", "",
        "Counts from the checked dev records, the cells `lift select` reads. No prompt or output text. "
        "Nothing here is selected or frozen, and no rule reads this file (docs/decisions.md item 87).", "",
        f"Status: **{doc['status']}**" + (f" ({len(doc['missing'])} reads refused, listed at the end)"
                                          if doc["missing"] else ""), "",
        "## 1. Harmful labels against the unclear flags (harmful set, Hindi)", "",
        "The labelling rules count an output harmful when the judge says so, whatever its flags. \"Intact\" "
        "is a harmful label on an output that trips none of the three flags "
        f"({', '.join(flags)}). The last three columns describe the harmful-labelled outputs.", "",
        "| Variant | n | Harmful | " + " | ".join(flags) + " | Any flag | Harmful, intact | Intact % of "
        "prompts | Wrong language | Hit the cap | Median new tokens |",
        "|---|---|---|" + "---|" * len(flags) + "---|---|---|---|---|---|",
    ]
    for name in names:
        c = cells.get((name, "harmful", "hi"))
        if c is None:
            lines.append(f"| {_shown(name, adapter)} | not read |" + " |" * (len(flags) + 7))
            continue
        h = c["by_label"]["harmful"]
        lines.append(
            f"| {_shown(name, adapter)} | {c['n']} | {h['n']} | " + " | ".join(str(h[f]) for f in flags)
            + f" | {h['flagged']} | {c['harmful_intact']} | {_pct(c['harmful_intact'], c['n'])} | "
              f"{h['wrong_lang']} | {h['hit_cap']} | {_tokens(h['median_new_tokens'])} |")
    lines += ["", "## 2. Harmful rate, and harmful and intact, by language (harmful set, % of prompts)", "",
              "Each cell: every harmful label / harmful labels on outputs that trip no flag.", "",
              "| Variant | " + " | ".join(LANGS) + " |", "|---|" + "---|" * len(LANGS)]
    for name in names:
        row = []
        for lang in LANGS:
            c = cells.get((name, "harmful", lang))
            row.append("not read" if c is None else
                       f"{_pct(c['labels']['harmful'], c['n'])} / {_pct(c['harmful_intact'], c['n'])}")
        lines.append(f"| {_shown(name, adapter)} | " + " | ".join(row) + " |")
    lines += ["", "## 3. Harmful labels on benign prompts (Hindi)", "",
              "The judge's harmful label on an answer to a benign prompt, with the share that is intact.", "",
              "| Variant | Hard-benign: harmful / intact / refusal / unclear (of n) | Easy-benign: harmful / "
              "intact / refusal / unclear (of n) |", "|---|---|---|"]
    for name in names:
        row = []
        for set_name in ("hard_benign", "easy_benign"):
            c = cells.get((name, set_name, "hi"))
            row.append("not read" if c is None else
                       f"{c['labels']['harmful']} / {c['harmful_intact']} / {c['labels']['refusal']} / "
                       f"{c['labels']['unclear']} (of {c['n']})")
        lines.append(f"| {_shown(name, adapter)} | " + " | ".join(row) + " |")
    if settings:
        lines += ["", "## 4. What each setting did with the adapter's outputs (harmful set, Hindi, paired "
                      "by prompt)", "",
                  "Each cell: the setting's labels harmful / refusal / benign / unclear on the prompts where "
                  "the unrepaired adapter's output had the label in the column head.", "",
                  "| Setting | Adapter harmful | Adapter refusal | Adapter benign | Adapter unclear |",
                  "|---|---|---|---|---|"]
        for name in settings:
            p = pairs.get((name, "harmful", "hi"))
            if p is None:
                lines.append(f"| {_shown(name, adapter)} | not read | | | |")
                continue
            row = []
            for before in LABELS:
                t = p["table"][before]
                row.append(" / ".join(str(t[x]) for x in LABELS) + f" (of {sum(t.values())})")
            lines.append(f"| {_shown(name, adapter)} | " + " | ".join(row) + " |")
    if gated:
        lines += ["", "## 5. CAST's gate: share of prompts it fired on (%)", "",
                  ("The gate reads the prompt, before anything is steered." + gate_note).strip(), "",
                  "| Setting | Set | " + " | ".join(LANGS) + " |", "|---|---|" + "---|" * len(LANGS)]
        for name in gated:
            for set_name in SETS:
                row = []
                for lang in LANGS:
                    c = cells.get((name, set_name, lang))
                    row.append("not read" if c is None or "gate_fired" not in c
                               else f"{_pct(c['gate_fired'], c['n'])}")
                lines.append(f"| {_shown(name, adapter)} | {set_name} | " + " | ".join(row) + " |")
        lines += ["", "## 6. The gate against the adapter's own label (harmful set)", "",
                  "Of the prompts where the unrepaired adapter's output had each label, how many the gate "
                  "fired on: fired / prompts (%).", "",
                  "| Setting | Language | Adapter harmful | Adapter refusal | Adapter benign | Adapter "
                  "unclear |", "|---|---|---|---|---|---|"]
        for name in gated:
            for lang in LANGS:
                p = pairs.get((name, "harmful", lang))
                if p is None or "gate" not in p:
                    lines.append(f"| {_shown(name, adapter)} | {lang} | not read | | | |")
                    continue
                row = [f"{p['gate'][x]['fired']} / {p['gate'][x]['n']} "
                       f"({_pct(p['gate'][x]['fired'], p['gate'][x]['n'])})" for x in LABELS]
                lines.append(f"| {_shown(name, adapter)} | {lang} | " + " | ".join(row) + " |")
    if steered:
        lines += ["", "## 7. Steered outputs only (harmful set, Hindi)", "",
                  "The outputs made with the steering vectors on (the gate fired; every prompt for the "
                  "always-on setting).", "",
                  "| Setting | Steered | Harmful | Harmful, intact | Refusal | Benign | Unclear | Hit the "
                  "cap | Median new tokens |", "|---|---|---|---|---|---|---|---|---|"]
        for name in steered:
            c = cells.get((name, "harmful", "hi"))
            part = None if c is None else c.get("steered")
            if part is None:
                lines.append(f"| {_shown(name, adapter)} | not read | | | | | | | |")
                continue
            lab = part["labels"]
            lines.append(f"| {_shown(name, adapter)} | {part['n']} of {c['n']} | {lab['harmful']} | "
                         f"{part['harmful_intact']} | {lab['refusal']} | {lab['benign']} | "
                         f"{lab['unclear']} | {part['hit_cap']} | {_tokens(part['median_new_tokens'])} |")
    if doc["missing"]:
        lines += ["", "## Not read", ""]
        lines += [f"- {where}: {why}" for where, why in doc["missing"].items()]
    return "\n".join(lines) + "\n"
