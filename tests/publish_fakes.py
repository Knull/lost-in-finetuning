"""Synthetic P05/P07 outputs and a recording stand-in for HfApi, for the lift publish tests.

Every number here is made up for the tests; nothing reaches the Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from lift import paths


def interval(est: float, lo: float, hi: float) -> dict:
    return {"est": est, "lo": lo, "hi": hi}


def effect(est: float, lo: float, hi: float, p: float) -> dict:
    return {
        "pooled": {"est": est, "lo": lo, "hi": hi, "p": p},
        "per_seed": [{"seed": s, "est": est, "lo": lo, "hi": hi, "p": p, "n": 400} for s in (1, 2, 3)],
    }


def table(columns: list[tuple[str, str, str]], rows: list[dict], title: str = "t") -> dict:
    return {
        "title": title,
        "columns": [{"key": k, "label": label, "format": f} for k, label, f in columns],
        "rows": rows,
    }


def report() -> dict:
    per_language = table(
        [
            ("lang", "Language", "text"),
            ("role", "Variant", "text"),
            ("set", "Set", "text"),
            ("harmful_rate", "Harmful", "pct_ci"),
            ("harmful_upper", "Upper", "pct"),
        ],
        [
            {
                "lang": lang,
                "role": role,
                "set": "harmful",
                "harmful_rate": interval(r, r - 0.02, r + 0.02),
                "harmful_upper": r + 0.01,
            }
            for lang in ("hi", "en")
            for role, r in (("stock", 0.05), ("adapter", 0.21), ("repair", 0.07))
        ],
    )
    simple = table([("name", "Name", "text"), ("value", "Value", "pct")], [{"name": "a", "value": 0.5}])
    return {
        "version": 1,
        "dtype": "bf16",
        "primary_judge": "qwen3guard",
        "verdicts": [
            {
                "adapter": "adapter-aya-std",
                "recipe": "standard",
                "verdict": "repair",
                "reason": "WiSE-FT restored safety and kept the gain.",
            }
        ],
        "primary": {
            "drift": effect(0.16, 0.11, 0.21, 0.0001),
            "repair": effect(-0.14, -0.19, -0.09, 0.0002),
            "redteam_drift": effect(0.1, 0.03, 0.17, 0.01),
            "redteam_repair": effect(-0.08, -0.15, -0.01, 0.03),
        },
        "headline": {"utility_retained": interval(0.8, 0.6, 1.0), "gpu_hours_per_checkup": 1.25},
        "tables": {
            "per_language": per_language,
            "repair_comparison": simple,
            "secondary": simple,
            "exploratory": simple,
            "overrefusal": simple,
            "utility": table(
                [("variant", "Variant", "text"), ("win", "Win rate", "pct_ci")],
                [{"variant": "repair", "win": interval(0.62, 0.55, 0.69)}, {"variant": "cast", "win": None}],
            ),
            "belebele": simple,
            "judge_sensitivity": simple,
            "signatures": simple,
            "cost": simple,
        },
        "not_run": ["ReSta on seed 3"],
    }


ADAPTER = "adapter-aya-std-s1"
FAMILY = {"wiseft": "wiseft", "resta": "resta", "cast": "cast", "cast_always": "cast"}
# The dev sweep's settings (configs/repairs.yaml), named as lift.repair.settings names them on P05.
SWEEP = {
    "wiseft-l0.25": "wiseft",
    "wiseft-l0.50": "wiseft",
    "wiseft-l0.75": "wiseft",
    "resta-a0.10": "resta",
    "resta-a0.25": "resta",
    "resta-a0.40": "resta",
    "cast-m0.5": "cast",
    "cast-m1.0": "cast",
    "cast-m2.0": "cast",
    "castalways-m1.0": "cast_always",
}
NO_DROP = (0.31, 0.30, 0.66, 0.60)  # (harmful, hard-benign refusal, win rate, Belebele): no drop
# Default numbers: WiSE-FT lambda 0.50 has the largest eligible drop; every family has a frozen setting.
DEFAULT_NUMBERS = {
    "wiseft-l0.50": (0.12, 0.30, 0.66, 0.60),
    "resta-a0.25": (0.18, 0.30, 0.66, 0.60),
    "cast-m1.0": (0.20, 0.30, 0.66, 0.60),
}
# CAST wins (test_largest_eligible_drop_wins on P05); REJECT: nothing eligible, cast-m2.0 frozen anyway.
CAST_WINS = {"wiseft-l0.50": (0.20, 0.30, 0.66, 0.60), "cast-m1.0": (0.12, 0.30, 0.66, 0.60)}
REJECT = {"cast-m2.0": (0.05, 0.40, 0.66, 0.60)}


def selection(numbers: dict | None = None, *, adapter: str = ADAPTER, incomplete: bool = False) -> dict:
    """outputs/metrics/selection.json exactly as P05's ``lift.repair.select.run`` writes it
    (origin/p05-repairs: ``run``, ``decide``, ``freeze_plan``), decided by the real selection rule
    (``lift.selection.select_repair``, the same module on both branches) over made-up dev scores.

    ``overall`` = {winner, verdict, eligible}; ``families`` = {family: {winner, verdict, eligible}} once
    decided and the list of family names while incomplete; ``frozen`` = {family: seed-1 setting name}."""
    from lift import config
    from lift.selection import DevScores, select_repair

    rules = config.get("selection", name="repairs")
    families = ["wiseft", "resta", "cast"]
    stock = DevScores("stock", "stock", 0.10, 0.30, 0.5, 0.60)
    adapter_scores = DevScores(adapter, "adapter", 0.30, 0.25, 0.70, 0.61)
    doc: dict = {
        "kind": "selection",
        "version": 1,
        "adapter": adapter,
        "split": "dev",
        "lang": "hi",
        "dtype": "bf16",
        "judge": "primary",
        "families": list(families),
        "rules": dict(rules),
        "created": "2026-11-07T00:00:00+00:00",
        "stock": vars(stock),
        "adapter_scores": vars(adapter_scores),
        "missing": {},
        "problems": [],
        "details": {},
    }
    if incomplete:
        doc["missing"] = {f"resta-a0.25-{adapter}": ["harmful_hi: no metrics"]}
        doc["status"] = "incomplete"
        return doc
    numbers = {**{tag: NO_DROP for tag in SWEEP}, **(DEFAULT_NUMBERS if numbers is None else numbers)}
    cands = [DevScores(f"{tag}-{adapter}", SWEEP[tag], *numbers[tag]) for tag in SWEEP]

    def sel_doc(sel) -> dict:
        return {"winner": sel.winner, "verdict": sel.verdict, "eligible": list(sel.eligible)}

    overall = select_repair(stock, adapter_scores, cands, rules)
    per_family = {}
    for fam in families:
        pool = [c for c in cands if FAMILY[c.method] == fam]
        per_family[fam] = sel_doc(select_repair(stock, adapter_scores, pool, rules))
    rows = [
        {**row, "family": FAMILY[row["method"]], "eligible": row["violations"] == 0} for row in overall.table
    ]
    frozen = {fam: s["winner"] for fam, s in per_family.items()}
    stem = adapter[: -len("1")]
    doc.update(
        {
            "overall": sel_doc(overall),
            "families": per_family,
            "table": rows,
            "status": "decided",
            "frozen": frozen,
            "freeze": {
                "tag": "git tag recipe-frozen-20261107 && git push --tags",
                "variants": [
                    {"family": fam, "seed": seed, "name": name.replace(adapter, f"{stem}{seed}")}
                    for fam, name in frozen.items()
                    for seed in (1, 2, 3)
                ],
                "fit_cast": [f"lift fit-cast --adapter {stem}{s}" for s in (2, 3)],
                "register": [],
            },
        }
    )
    return doc


def frozen_variants(doc: dict) -> list[str]:
    """Every frozen variant of a decided selection: each family's setting on seeds 1-3."""
    stem = doc["adapter"][: -len("1")]
    return [
        name.replace(doc["adapter"], f"{stem}{seed}") for name in doc["frozen"].values() for seed in (1, 2, 3)
    ]


def write_json(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def write_inputs(
    *, with_report: bool = True, with_selection: bool = True, selection_doc: dict | None = None
) -> None:
    if with_report:
        write_json(paths.metrics_dir() / "final_report.json", report())
    if with_selection:
        write_json(paths.metrics_dir() / "selection.json", selection_doc or selection())


class FakeApi:
    """Records every HfApi call lift.publish makes, in order."""

    def __init__(self, missing: set[str] | None = None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.missing = set(missing or ())  # repo ids repo_exists reports as absent

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "repo_exists":
                return args[0] not in self.missing
            if name == "create_collection":
                return SimpleNamespace(slug=f"org/{args[0][:8].lower()}-0123", url=None)
            return None

        return call

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]
