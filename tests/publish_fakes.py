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


def selection(winner: str = "wiseft-l0.50-adapter-aya-std-s1") -> dict:
    return {
        "adapter": "adapter-aya-std-s1",
        "winner": winner,
        "verdict": "repair",
        "eligible": [winner],
        "table": [],
        "families": {"wiseft": {"winner": winner}, "cast": {"winner": "cast-m1.0-adapter-aya-std-s1"}},
    }


def write_json(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def write_inputs(*, with_report: bool = True, with_selection: bool = True) -> None:
    if with_report:
        write_json(paths.metrics_dir() / "final_report.json", report())
    if with_selection:
        write_json(paths.metrics_dir() / "selection.json", selection())


class FakeApi:
    """Records every HfApi call lift.publish makes, in order."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "create_collection":
                return SimpleNamespace(slug=f"org/{args[0][:8].lower()}-0123", url=None)
            return None

        return call

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]
