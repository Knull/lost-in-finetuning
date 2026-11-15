"""Synthetic P05/P07 outputs and a recording stand-in for HfApi, for the lift publish tests.

Every number here is made up for the tests; nothing reaches the Hub.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from lift import paths


def interval(est: float, lo: float, hi: float) -> dict:
    return {"est": est, "lo": lo, "hi": hi}


def effect(est: float, lo: float, hi: float, p: float, seeds: tuple[int, ...] = (1, 2, 3)) -> dict:
    """An Effect as P07's lift.report.tables.effect_out writes it."""
    return {
        "pooled": {"est": est, "lo": lo, "hi": hi, "p": p, "n": 400 * len(seeds), "seeds": list(seeds)},
        "per_seed": [{"seed": s, "est": est + s / 1000, "lo": lo, "hi": hi, "p": p, "n": 400} for s in seeds],
    }


def table(columns: list[tuple[str, str, str]], rows: list[dict], title: str = "t", note: str = "") -> dict:
    """A table spec as P07's ``lift.report.tables.table`` writes it (title, columns, rows, note)."""
    return {
        "title": title,
        "columns": [{"key": k, "label": label, "format": f} for k, label, f in columns],
        "rows": [dict(r) for r in rows],
        "note": note,
    }


QWEN_ID = "Qwen/Qwen3Guard-Gen-4B@main#bf16#strict"
FRONTIER_ID = "anthropic/claude-sonnet-5-5@safety_v1:0123456789ab+t0+max1024+api2026-10-21"

# The tables ``lift.report.tables.build`` always puts in final_report.json: its TABLES, the repair-vs-WiSE-FT
# contrast and, since decisions item 82, every adapter on its own axes, every adapter's drift, the
# held-out loss and the other readings (P07 also writes ``cells``, which P08 does not read). No
# ``signatures``: no stage writes it.
P07_TABLES = (
    "adapters",
    "adapter_drift",
    "per_language",
    "repair_comparison",
    "secondary",
    "exploratory",
    "overrefusal",
    "utility",
    "belebele",
    "heldout",
    "judge_sensitivity",
    "sensitivity_effects",
    "cost",
    "repair_vs_wiseft",
)
# P07's analysis.UNDEFINED_GAIN (decisions.md 77): a share computed but undefined says so, never None.
UNDEFINED_GAIN = "undefined: the adapter shows no utility gain over parity"
P07_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "zh": "Chinese",
    "es": "Spanish",
    "ar": "Arabic",
    "th": "Thai",
}
WINNER = "wiseft-l0.50-adapter-aya-std-s1"
# Per-language harmful rates (stock, adapter, frozen repair) behind the per_language table; Thai's repair
# arm was not run (None), as P07 writes an arm it could not read.
HARMFUL = {
    "Hindi": (0.05, 0.21, 0.07),
    "English": (0.02, 0.09, 0.03),
    "Chinese": (0.04, 0.12, 0.05),
    "Spanish": (0.03, 0.10, 0.04),
    "Arabic": (0.06, 0.08, 0.06),
    "Thai": (0.07, 0.09, None),
    "Hindi (Aya Red-teaming)": (0.10, 0.20, 0.12),
}
# Hard-benign refusal rates (stock, adapter, frozen repair) behind the overrefusal table.
HARD_REFUSAL = {"hi": (0.30, 0.25, 0.33), "en": (0.20, 0.18, 0.22), "zh": (0.25, 0.24, 0.27)}


def _p07_per_language() -> dict:
    rows = []
    for label, (stock, adapter, repair) in HARMFUL.items():
        rows.append(
            {
                "lang": label,
                "stock": stock,
                "stock_upper": stock + 0.01,
                "adapter": adapter,
                "adapter_upper": adapter + 0.01,
                "repair": repair,
                "repair_upper": None if repair is None else repair + 0.01,
                "n": 100 if "Red-teaming" in label else 400,
                "drift": interval(adapter - stock, adapter - stock - 0.05, adapter - stock + 0.05),
                "repair_effect": None
                if repair is None
                else interval(repair - adapter, repair - adapter - 0.05, repair - adapter + 0.05),
            }
        )
    return table(
        [
            ("lang", "Language", "text"),
            ("n", "Prompts", "int"),
            ("stock", "Stock", "pct"),
            ("stock_upper", "Stock (upper)", "pct"),
            ("adapter", "Adapter", "pct"),
            ("adapter_upper", "Adapter (upper)", "pct"),
            ("repair", "Frozen repair", "pct"),
            ("repair_upper", "Frozen repair (upper)", "pct"),
            ("drift", "Drift (adapter - stock)", "pp_ci"),
            ("repair_effect", "Repair (repair - adapter)", "pp_ci"),
        ],
        rows,
        "Harmful rate per language, final split (pooled over seeds)",
        "Harmful rate H/N; the upper bound counts unclear outputs as harmful, (H+U)/N.",
    )


def _p07_overrefusal() -> dict:
    rows = []
    for set_name, rates in (("easy", {"hi": (0.02, 0.03, 0.02)}), ("hard", HARD_REFUSAL)):
        for lang, (stock, adapter, repair) in rates.items():
            rows.append(
                {
                    "set": set_name,
                    "lang": P07_LANG_NAMES[lang],
                    "stock": stock,
                    "adapter": adapter,
                    "repair": repair,
                    "adapter_change": interval(
                        adapter - stock, adapter - stock - 0.03, adapter - stock + 0.03
                    ),
                    "repair_change": interval(repair - stock, repair - stock - 0.03, repair - stock + 0.03),
                }
            )
    return table(
        [
            ("set", "Benign set", "text"),
            ("lang", "Language", "text"),
            ("stock", "Stock", "pct"),
            ("adapter", "Adapter", "pct"),
            ("repair", "Frozen repair", "pct"),
            ("adapter_change", "Adapter - stock", "pp_ci"),
            ("repair_change", "Repair - stock", "pp_ci"),
        ],
        rows,
        "Over-refusal: refusal rate on benign prompts, final split",
        "Refusals of benign prompts (PolyRefuse harmless = easy, OR-Bench-Hard = hard).",
    )


def _claims(names: list[str], adj: str, title: str) -> dict:
    rows = [
        {
            "claim": name,
            "effect": interval(0.05, 0.01, 0.09) if i else None,
            "p": 0.01 if i else None,
            "p_adj": 0.04 if i else None,
            "n": 800 if i else None,
            "seeds": "1, 2, 3" if i else "not run",
        }
        for i, name in enumerate(names)
    ]
    return table(
        [
            ("claim", "Claim", "text"),
            ("effect", "Effect", "pp_ci"),
            ("p", "p", "p"),
            ("p_adj", adj, "p"),
            ("n", "Rows", "int"),
            ("seeds", "Seeds", "text"),
        ],
        rows,
        title,
        "Adjusted within this family.",
    )


def cell(
    variant: str,
    set_name: str,
    n: int,
    harmful: float,
    intact: float,
    lang: str = "hi",
    cap: float | None = None,
) -> dict:
    """A row of P07's table of every final cell (lift.report.tables.cells_table), the columns P08 reads.
    ``cap``: the share of answers that reached the token cap (P07 writes null when it has no generations)."""
    return {
        "variant": variant,
        "set": set_name,
        "lang": lang,
        "n": n,
        "harmful_rate": harmful,
        "harmful_intact_rate": intact,
        "cap_hit_rate": cap,
    }


# The judge's harmful label on answers to benign Hindi prompts (decisions item 95): frequent on the
# hard-benign ones, more so for the adapter, rare on the easy-benign ones. Stock's answers reach the token
# cap more often than the adapter's (decisions item 132).
BENIGN_CELLS = [
    cell("stock", "hard_benign", 200, 0.30, 0.29, cap=0.80),
    cell("adapter-aya-std-s1", "hard_benign", 200, 0.55, 0.45, cap=0.35),
    cell("stock", "easy_benign", 300, 0.02, 0.02, cap=0.70),
    cell("adapter-aya-std-s1", "easy_benign", 300, 0.03, 0.03, cap=0.10),
]


def _p07_tables() -> dict:
    utility_rows = [
        {
            "arm": arm,
            "variant": name,
            "seed": str(seed),
            "win_rate": None if win is None else interval(win, win - 0.07, win + 0.07),
            "n": None if win is None else 200,
            "retained": interval(1.0, 1.0, 1.0) if arm == "adapter" else interval(0.8, 0.6, 1.0),
        }
        for arm, name, seed, win in (
            ("adapter", "adapter-aya-std-s1", 1, 0.62),
            ("wiseft", WINNER, 1, 0.58),
            ("cast", "cast-m1.0-adapter-aya-std-s3", 3, None),
        )
    ]
    return {
        "per_language": _p07_per_language(),
        "repair_comparison": table(
            [
                ("family", "Family", "text"),
                ("variant", "Frozen setting (seed 1)", "text"),
                ("role", "Role", "text"),
                ("harmful_hi", "Harmful rate", "pct"),
                ("effect", "Repair - adapter", "pp_ci"),
                ("p", "p (bootstrap)", "p"),
                ("retained", "Utility gain kept", "pct_ci"),
                ("overrefusal_change", "Hard-benign refusal - stock", "pp_ci"),
                ("belebele_change", "Belebele-hi - stock", "pp_ci"),
            ],
            [
                {
                    "family": "wiseft",
                    "variant": WINNER,
                    "role": "winner",
                    "harmful_hi": 0.07,
                    "effect": interval(-0.14, -0.19, -0.09),
                    "p": 0.0002,
                    "retained": interval(0.8, 0.6, 1.0),
                    "overrefusal_change": interval(0.03, 0.0, 0.06),
                    "belebele_change": interval(-0.01, -0.03, 0.01),
                },
                {
                    "family": "resta",
                    "variant": "resta-a0.25-adapter-aya-std-s1",
                    "role": "family",
                    "harmful_hi": None,
                    "effect": None,
                    "p": None,
                    "retained": None,
                    "overrefusal_change": None,
                    "belebele_change": None,
                },
            ],
            "Every frozen repair family on the final split (Hindi, pooled over seeds)",
            "WiSE-FT attenuation is the baseline every repair is measured against.",
        ),
        "secondary": _claims(
            ["drift zh+es", "repair zh+es", "drift en", "repair en"],
            "p (Holm)",
            "Secondary claims (Holm-corrected)",
        ),
        "exploratory": _claims(
            [f"{k} {lang}" for lang in ("zh", "es", "ar", "th") for k in ("drift", "repair")],
            "q (BH)",
            "Exploratory claims (BH-FDR)",
        ),
        "overrefusal": _p07_overrefusal(),
        "utility": table(
            [
                ("arm", "Arm", "text"),
                ("variant", "Variant", "text"),
                ("seed", "Seed", "text"),
                ("win_rate", "Win rate vs stock", "pct_ci"),
                ("n", "Pairs", "int"),
                ("retained", "Adapter gain kept (pooled over seeds)", "pct_ci"),
            ],
            utility_rows,
            "Hindi utility against stock, final split (200 pairs, both orders)",
            "Win rate: mean pair score against stock.",
        ),
        "belebele": table(
            [
                ("lang", "Language", "text"),
                ("n", "Items", "int"),
                ("stock", "Stock", "pct"),
                ("adapter", "Adapter", "pct"),
                ("repair", "Frozen repair", "pct"),
                ("repair_change", "Repair - stock", "pp_ci"),
            ],
            [
                {
                    "lang": "Hindi",
                    "n": 700,
                    "stock": 0.5,
                    "adapter": 0.49,
                    "repair": 0.49,
                    "repair_change": interval(-0.01, -0.03, 0.01),
                }
            ],
            "Belebele accuracy per language, final split (700 items)",
            "Letter log-likelihood scoring, no chat template; pooled over seeds.",
        ),
        "judge_sensitivity": table(
            [
                ("check", "Check", "text"),
                ("lang", "Language", "text"),
                ("n", "Items", "int"),
                ("agreement", "Agreement", "pct"),
                ("kappa", "Kappa", "f2"),
                ("agreement_binary", "Binary agreement", "pct"),
            ],
            [
                {
                    "check": check,
                    "lang": lang,
                    "n": 40,
                    "agreement": 0.9,
                    "kappa": 0.8,
                    "agreement_binary": 0.95,
                }
                for check, lang in (
                    ("second judge vs primary", "Hindi"),
                    ("back-translated (NLLB) vs native", "Arabic"),
                    ("back-translated (NLLB) vs native", "Thai"),
                )
            ],
            "Judge sensitivity",
            "Four-class agreement and Cohen's kappa; binary = harmful vs not.",
        ),
        "cost": table(
            [
                ("variant", "Variant", "text"),
                ("role", "Role", "text"),
                ("gpu_hours", "GPU-hours", "hours"),
                ("ms_per_token", "Per generated token", "ms"),
                ("gate_share", "Gate share", "pct"),
            ],
            [
                {
                    "variant": "stock",
                    "role": "stock",
                    "gpu_hours": 0.4,
                    "ms_per_token": 21.5,
                    "gate_share": 0.0,
                },
                {
                    "variant": WINNER,
                    "role": "winner",
                    "gpu_hours": None,
                    "ms_per_token": None,
                    "gate_share": 0.0,
                },
            ],
            "Cost of the final evaluation",
            "GPU-hours: the variant's whole final evaluation.",
        ),
        "repair_vs_wiseft": repair_vs_wiseft(),
        **_p07_adapter_tables(),
        "cells": table(
            [
                ("variant", "Variant", "text"),
                ("set", "Set", "text"),
                ("lang", "Language", "text"),
                ("n", "n", "int"),
                ("harmful_rate", "Harmful", "pct"),
                ("harmful_intact_rate", "Harmful, no unclear flag", "pct"),
                ("cap_hit_rate", "Cap hit", "pct"),
            ],
            [
                cell("stock", "harmful", 400, 0.14, 0.14, cap=0.40),
                *BENIGN_CELLS,
                cell("adapter-aya-std-s1", "harmful", 400, 0.21, 0.18, cap=0.15),
                cell(WINNER, "harmful", 400, 0.07, 0.07, cap=0.20),
            ],
            "Every final cell (variant x set x language)",
        ),
    }


CANDIDATE = "adapter-indicalign-std-s1"
# verdicts[i].axes as P07's lift.report.analysis.adapter_axes writes them (decisions items 82 and 86). The
# adapter of the comparison: a better held-out loss, a benchmark that does not move, and a harmful rate
# over Gate 2's bar. The candidate: a lower harmful rate that comes with more refusals of benign prompts,
# and a better loss with a lost pairwise comparison; its Belebele was not run. drift_intact (decisions item
# 93): the same difference counting only outputs that trip no unclear flag, here the same way as drift.
ADAPTER_AXES = {
    "drift": {**interval(0.16, 0.11, 0.21), "p": 0.0001, "seeds": [1, 2, 3], "meaningful": True},
    "drift_intact": {**interval(0.12, 0.07, 0.17), "p": 0.0001, "seeds": [1, 2, 3]},
    "refusal_harmful": {**interval(-0.15, -0.20, -0.10), "p": 0.0001, "seeds": [1, 2, 3]},
    "overrefusal_hard_hi": {**interval(0.01, -0.02, 0.04), "p": 0.5, "seeds": [1, 2, 3], "exceeds": False},
    "usefulness": {"win_rate": 0.62, "lo": 0.55, "hi": 0.69, "bar": 0.55, "passes": True},
    "final_win_rate": {**interval(0.62, 0.55, 0.69), "n": 600, "seeds": [1, 2, 3]},
    "heldout": {**interval(-0.33, -0.36, -0.30), "n": 600, "n_clusters": 200, "seeds": [1, 2, 3]},
    "belebele_hi": {**interval(-0.01, -0.03, 0.01), "p": 0.4, "seeds": [1, 2, 3]},
}
CANDIDATE_AXES = {
    "drift": {**interval(-0.05, -0.09, -0.01), "p": 0.02, "seeds": [1, 2, 3], "meaningful": False},
    "drift_intact": {**interval(-0.04, -0.07, -0.01), "p": 0.03, "seeds": [1, 2, 3]},
    "refusal_harmful": {**interval(0.06, 0.02, 0.10), "p": 0.01, "seeds": [1, 2, 3]},
    "overrefusal_hard_hi": {**interval(0.10, 0.04, 0.16), "p": 0.002, "seeds": [1, 2, 3], "exceeds": True},
    "usefulness": {"win_rate": 0.40, "lo": 0.31, "hi": 0.49, "bar": 0.55, "passes": False},
    "final_win_rate": {**interval(0.40, 0.33, 0.47), "n": 600, "seeds": [1, 2, 3]},
    "heldout": {**interval(-0.30, -0.33, -0.27), "n": 600, "n_clusters": 200, "seeds": [1, 2, 3]},
    "belebele_hi": None,
}


def _p07_adapter_tables() -> dict:
    """P07's tables beyond its first nine, as ``lift.report.tables`` writes them since decisions item 82:
    every adapter on its own axes, every adapter's drift per language, the held-out loss and drift and
    repair under other readings."""
    return {
        "adapters": table(
            [
                ("adapter", "Adapter (seed 1)", "text"),
                ("role", "Role", "text"),
                ("verdict", "Verdict (pre-registered)", "text"),
                ("usefulness", "Usefulness (Gate 1)", "text"),
                ("drift", "Harmful rate - stock", "pp_ci"),
                ("drift_intact", "Harmful rate - stock, outputs that trip no unclear flag", "pp_ci"),
                ("overrefusal", "Hard-benign refusals - stock", "pp_ci"),
                ("final_win_rate", "Final win rate vs stock", "pct_ci"),
                ("heldout", "Held-out NLL - stock (nats/token)", "text"),
                ("belebele_hi", "Belebele-hi accuracy - stock", "pp_ci"),
                ("seeds", "Seeds", "text"),
            ],
            [
                {
                    "adapter": "adapter-aya-std-s1",
                    "role": "adapter of the repair comparison",
                    "verdict": "repair",
                    "usefulness": "meets Gate 1's bar",
                    "drift": interval(0.16, 0.11, 0.21),
                    "drift_intact": interval(0.12, 0.07, 0.17),
                    "overrefusal": interval(0.01, -0.02, 0.04),
                    "final_win_rate": interval(0.62, 0.55, 0.69),
                    "heldout": "-0.330 [-0.360, -0.300] (seeds 1, 2, 3)",
                    "belebele_hi": interval(-0.01, -0.03, 0.01),
                    "seeds": "1, 2, 3",
                },
                {
                    "adapter": CANDIDATE,
                    "role": "candidate not picked (exploratory)",
                    "verdict": "retain",
                    "usefulness": "reject on usefulness",
                    "drift": interval(-0.05, -0.09, -0.01),
                    "drift_intact": interval(-0.04, -0.07, -0.01),
                    "overrefusal": interval(0.10, 0.04, 0.16),
                    "final_win_rate": interval(0.40, 0.33, 0.47),
                    "heldout": "-0.300 [-0.330, -0.270] (seeds 1, 2, 3)",
                    "belebele_hi": None,
                    "seeds": "1, 2, 3",
                },
            ],
            "Every adapter on its own axes, against stock (Hindi, final split unless said)",
            "Each axis is against stock, paired by prompt and pooled over the seeds named.",
        ),
        "adapter_drift": table(
            [
                ("adapter", "Adapter (seed 1)", "text"),
                ("role", "Role", "text"),
                ("lang", "Language", "text"),
                ("stock", "Stock", "pct"),
                ("rate", "Adapter", "pct"),
                ("drift", "Adapter - stock", "pp_ci"),
                ("drift_intact", "Adapter - stock, outputs that trip no unclear flag", "pp_ci"),
                ("refusal_change", "Refusal rate: adapter - stock", "pp_ci"),
                ("seeds", "Seeds", "text"),
            ],
            [
                {
                    "adapter": CANDIDATE,
                    "role": "candidate not picked (exploratory)",
                    "lang": "Hindi",
                    "stock": 0.14,
                    "rate": 0.09,
                    "drift": interval(-0.05, -0.09, -0.01),
                    "drift_intact": interval(-0.04, -0.07, -0.01),
                    "refusal_change": interval(0.06, 0.02, 0.10),
                    "seeds": "1, 2, 3",
                }
            ],
            "Harmful rate against stock per language, every adapter (final split)",
            "Descriptive for the adapters outside the repair comparison: in no claim family.",
        ),
        "heldout": table(
            [
                ("variant", "Variant", "text"),
                ("mean_nll", "Mean NLL (nats/token)", "f3"),
                ("vs_stock", "NLL - stock", "text"),
                ("n", "References", "int"),
                ("gain_kept", "Adapter's held-out gain kept (pooled over the seeds named)", "text"),
            ],
            [
                {
                    "variant": "stock",
                    "mean_nll": 2.193,
                    "vs_stock": "the reference",
                    "n": 200,
                    "gain_kept": "the reference",
                },
                {
                    "variant": "adapter-aya-std-s1",
                    "mean_nll": 1.863,
                    "vs_stock": "-0.330 [-0.360, -0.300]",
                    "n": 200,
                    "gain_kept": "its own fit",
                },
                {
                    "variant": WINNER,
                    "mean_nll": 1.995,
                    "vs_stock": "-0.198 [-0.220, -0.176]",
                    "n": 200,
                    "gain_kept": "60.0% [52.0, 68.0]",
                },
            ],
            "Held-out loss on the Hindi utility references, final split (200 references)",
            "Gain kept: (stock - repair) / (stock - adapter) over the same references.",
        ),
        "sensitivity_effects": table(
            [
                ("reading", "Reading", "text"),
                ("kind", "Comparison", "text"),
                ("effect", "Effect", "pp_ci"),
                ("p", "p (bootstrap)", "p"),
                ("seeds", "Seeds", "text"),
            ],
            [
                {
                    "reading": "refusal rate on the same harmful prompts",
                    "kind": "drift",
                    "effect": interval(-0.15, -0.20, -0.10),
                    "p": 0.0001,
                    "seeds": "1, 2, 3",
                },
                {
                    "reading": "harmful rate under the second judge's labels, Hindi",
                    "kind": "repair",
                    "effect": None,
                    "p": None,
                    "seeds": None,
                },
            ],
            "Drift and repair under other readings of the labels (Hindi unless said, final split)",
            "Sensitivity analyses, not claims.",
        ),
    }


def repair_vs_wiseft() -> dict:
    """P07's ``tables.repair_vs_wiseft`` (origin/p07-final-report @ 4e4f007): each other frozen family
    minus WiSE-FT, descriptive; ``gain_kept`` is a text column (an interval as text, or undefined)."""
    return table(
        [
            ("family", "Family", "text"),
            ("variant", "Frozen setting (seed 1)", "text"),
            ("reference", "WiSE-FT frozen setting (seed 1)", "text"),
            ("harmful", "Harmful rate - WiSE-FT", "pp_ci"),
            ("overrefusal", "Hard-benign refusal - WiSE-FT", "pp_ci"),
            ("gain_kept", "Utility gain kept - WiSE-FT's", "text"),
            ("seeds", "Seeds", "text"),
        ],
        [
            {
                "family": "resta",
                "variant": "resta-a0.25-adapter-aya-std-s1",
                "reference": WINNER,
                "harmful": interval(-0.04, -0.08, -0.01),
                "overrefusal": interval(0.0, -0.1, 0.1),
                "gain_kept": "+0.0 pp [-10.0, +10.0] (seeds 1, 2, 3)",
                "seeds": "1, 2, 3",
            },
            {
                "family": "cast",
                "variant": "cast-m1.0-adapter-aya-std-s1",
                "reference": WINNER,
                "harmful": interval(0.01, -0.02, 0.04),
                "overrefusal": None,
                "gain_kept": UNDEFINED_GAIN,
                "seeds": "1, 2",
            },
        ],
        "Each repair against WiSE-FT attenuation, final split, Hindi (not pre-registered, descriptive)",
        "Not pre-registered, descriptive (docs/decisions.md 78): estimates and intervals only.",
    )


def report(selection_doc: dict | None = None) -> dict:
    """final_report.json exactly as P07's ``lift.report.tables.build`` writes it (origin/p07-final-report
    @ 1570807): its TABLES with their columns, labels, formats and notes, the per-language table wide (one
    row per language label, role columns), over-refusal in its own table, and the extra keys P08 does not
    read (``cells``, ``created``, ``cost_detail``, ``primary.*_loose``). ``selection`` carries the
    cross-branch contract (review XB-04): the overall winner, the frozen map and the sha256 of the
    selection.json bytes the report was built from (``selection_doc``, default ``selection()``, as
    ``write_json`` writes it)."""
    return {
        "version": 1,
        "dtype": "bf16",
        # the judge id of the final judgement records (Qwen3GuardJudge.judge_id, revision "main" in tests)
        "primary_judge": QWEN_ID,
        "created": "2026-11-11T00:00:00+00:00",
        "selection": binding(selection() if selection_doc is None else selection_doc),
        "verdicts": [
            {
                "adapter": ADAPTER,  # the seed-1 name, as P07 writes it (lift.report.analysis)
                "recipe": "standard",
                "kind": "adapter",
                "verdict": "repair",
                "usefulness_verdict": None,
                "reason": "WiSE-FT restored safety and kept the gain.",
                "axes": ADAPTER_AXES,
            },
            {
                "adapter": CANDIDATE,
                "recipe": "standard",
                "kind": "candidate",
                "verdict": "retain",
                "usefulness_verdict": "reject on usefulness",
                "reason": "Gate 1 did not pick it; its final drift does not meet Gate 2's bar.",
                "axes": CANDIDATE_AXES,
            },
        ],
        "primary": {
            "drift": effect(0.16, 0.11, 0.21, 0.0001),
            "repair": effect(-0.14, -0.19, -0.09, 0.0002),
            "redteam_drift": effect(0.1, 0.03, 0.17, 0.01),
            "redteam_repair": effect(-0.08, -0.15, -0.01, 0.03),
            "drift_loose": effect(0.15, 0.10, 0.20, 0.0001),
            "repair_loose": effect(-0.13, -0.18, -0.08, 0.0003),
        },
        "headline": {
            "utility_retained": interval(0.8, 0.6, 1.0),
            "heldout_retained": interval(0.6, 0.52, 0.68),
            "belebele_hi_change": interval(-0.01, -0.03, 0.01),
            "overrefusal_hard_hi_change": interval(0.02, 0.0, 0.04),
            "gpu_hours_per_checkup": 1.25,
            "gpu_hours_per_variant_final": 0.4,
        },
        "tables": _p07_tables(),
        "not_run": ["ReSta on seed 3"],
        "cost_detail": {"rows": [], "gpu_hours_per_checkup": 1.25},
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


def selection(
    numbers: dict | None = None,
    *,
    adapter: str = ADAPTER,
    incomplete: bool = False,
    adapter_belebele: float = 0.61,
    adapter_win: float = 0.70,
) -> dict:
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
    adapter_scores = DevScores(adapter, "adapter", 0.30, 0.25, adapter_win, adapter_belebele)
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
    from lift.selection import min_win_rate

    by_name = {c.name: c for c in cands}
    rows = [
        {
            **row,
            "family": FAMILY[row["method"]],
            "harmful_hi": by_name[row["name"]].harmful_hi,
            "overrefusal_hard_hi": by_name[row["name"]].overrefusal_hard_hi,
            "overrefusal_increase": by_name[row["name"]].overrefusal_hard_hi - stock.overrefusal_hard_hi,
            "min_win_rate": min_win_rate(adapter_scores.win_rate, rules),
            "belebele_hi": by_name[row["name"]].belebele_hi,
            "belebele_drop": stock.belebele_hi - by_name[row["name"]].belebele_hi,
            "eligible": row["violations"] == 0,
        }
        for row in overall.table
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


def secondary_selection(
    selection_doc: dict | None = None, kept: dict | None = None, *, status: str = "decided"
) -> dict:
    """outputs/metrics/selection_secondary.json as ``lift.repair.select.run_secondary`` writes it for
    ``selection_doc``'s dev numbers: the real rule (``select_repair(..., heldout_retained=...)``) with the
    share of the adapter's held-out-loss gain each setting keeps (``kept``: {tag: share}; 0.6 unless
    given). ``status`` "incomplete" or "undefined" gives those documents, which hold no selection."""
    from lift import config
    from lift.selection import DevScores, select_repair

    sel = selection() if selection_doc is None else selection_doc
    adapter = sel["adapter"]
    rules = config.get("selection", name="repairs")
    doc: dict = {
        "kind": "selection_secondary",
        "version": 1,
        "adapter": adapter,
        "split": "dev",
        "lang": "hi",
        "dtype": "bf16",
        "judge": "primary",
        "families": ["wiseft", "resta", "cast"],
        "rules": dict(rules),
        "decision": "docs/decisions.md item 73",
        "created": "2026-11-10T00:00:00+00:00",
        "stock": sel["stock"],
        "adapter_scores": sel["adapter_scores"],
        "missing": {},
        "problems": [],
        "heldout": {},
        "status": status,
    }
    if status == "incomplete":
        doc["missing"] = {f"cast-m0.5-{adapter}": ["heldout_loss: no complete held-out loss dev scores"]}
        return doc
    if status == "undefined":
        doc["undefined"] = f"{adapter} has no held-out-loss gain over stock to keep"
        return doc
    stock = DevScores(**sel["stock"])
    adapter_scores = DevScores(**sel["adapter_scores"])
    cands = [
        DevScores(
            r["name"], r["method"], r["harmful_hi"], r["overrefusal_hard_hi"], r["win_rate"], r["belebele_hi"]
        )
        for r in sel["table"]
    ]
    shares = {c.name: (kept or {}).get(c.name[: -len(adapter) - 1], 0.6) for c in cands}

    def sel_doc(x) -> dict:
        return {"winner": x.winner, "verdict": x.verdict, "eligible": list(x.eligible)}

    overall = select_repair(stock, adapter_scores, cands, rules, heldout_retained=shares)
    by_name = {c.name: c for c in cands}
    doc["overall"] = sel_doc(overall)
    doc["primary"] = sel_doc(select_repair(stock, adapter_scores, cands, rules))
    doc["families"] = {
        fam: sel_doc(
            select_repair(
                stock,
                adapter_scores,
                [c for c in cands if FAMILY[c.method] == fam],
                rules,
                heldout_retained=shares,
            )
        )
        for fam in doc["families"]
    }
    doc["adapter_gain"] = {"nll_minus_stock": -0.4, "lo": -0.45, "hi": -0.35}
    doc["table"] = [
        {
            **row,
            "family": FAMILY[row["method"]],
            "harmful_hi": by_name[row["name"]].harmful_hi,
            "overrefusal_hard_hi": by_name[row["name"]].overrefusal_hard_hi,
            "overrefusal_increase": by_name[row["name"]].overrefusal_hard_hi - stock.overrefusal_hard_hi,
            "heldout_nll_minus_stock": -0.4 * shares[row["name"]],
            "min_heldout_retained": rules["min_gain_retained"],
            "belebele_hi": by_name[row["name"]].belebele_hi,
            "belebele_drop": stock.belebele_hi - by_name[row["name"]].belebele_hi,
            "eligible": row["violations"] == 0,
        }
        for row in overall.table
    ]
    return doc


# --------------------------------------------------------------- a study shaped like ours (decisions 123)

STRESS = "adapter-indicalign-stress-s1"
# Dev scores (harmful, hard-benign refusal, win rate, Belebele) under which the rule freezes WiSE-FT at
# lambda 0.25 with the verdict repair and rejects the other two families: their best settings keep the
# adapter's own Belebele-hi accuracy, 5 points under stock's.
STRESS_NUMBERS = {
    "wiseft-l0.25": (0.07, 0.33, 0.48, 0.59),
    "resta-a0.10": (0.22, 0.20, 0.27, 0.55),
    "cast-m0.5": (0.20, 0.26, 0.28, 0.55),
}
# verdicts[i].axes of the three adapters, in the pattern of the study. The stress adapter: a better
# held-out loss, a lost pairwise comparison, a harmful rate far over stock's on both readings, fewer
# refusals and a lower Belebele-hi accuracy. The pick (aya): a lower harmful rate with more refusals of
# benign prompts. The candidate not picked: a rise on every label that the reading without flagged
# outputs does not show. None met Gate 1's bar.
STRESS_AXES = {
    "drift": {**interval(0.30, 0.25, 0.35), "p": 0.0001, "seeds": [1], "meaningful": True},
    "drift_intact": {**interval(0.20, 0.15, 0.25), "p": 0.0001, "seeds": [1]},
    "refusal_harmful": {**interval(-0.29, -0.34, -0.24), "p": 0.0001, "seeds": [1]},
    "overrefusal_hard_hi": {**interval(-0.22, -0.28, -0.16), "p": 0.0001, "seeds": [1], "exceeds": False},
    "usefulness": {"win_rate": 0.28, "lo": 0.20, "hi": 0.36, "bar": 0.55, "passes": False},
    "final_win_rate": {**interval(0.39, 0.33, 0.45), "n": 200, "seeds": [1]},
    "heldout": {**interval(-0.16, -0.23, -0.11), "n": 200, "n_clusters": 200, "seeds": [1]},
    "belebele_hi": {**interval(-0.04, -0.07, -0.01), "p": 0.008, "seeds": [1]},
}
PICK_AXES = {
    "drift": {**interval(-0.09, -0.12, -0.05), "p": 0.0001, "seeds": [1, 2, 3], "meaningful": False},
    "drift_intact": {**interval(-0.09, -0.13, -0.06), "p": 0.0001, "seeds": [1, 2, 3]},
    "refusal_harmful": {**interval(0.13, 0.09, 0.17), "p": 0.0001, "seeds": [1, 2, 3]},
    "overrefusal_hard_hi": {**interval(0.10, 0.05, 0.15), "p": 0.0002, "seeds": [1, 2, 3], "exceeds": True},
    "usefulness": {"win_rate": 0.42, "lo": 0.33, "hi": 0.51, "bar": 0.55, "passes": False},
    "final_win_rate": {**interval(0.47, 0.42, 0.53), "n": 600, "seeds": [1, 2, 3]},
    "heldout": {**interval(-0.31, -0.38, -0.25), "n": 600, "n_clusters": 200, "seeds": [1, 2, 3]},
    "belebele_hi": {**interval(-0.005, -0.02, 0.01), "p": 0.5, "seeds": [1, 2, 3]},
}
OTHER_AXES = {
    "drift": {**interval(0.22, 0.17, 0.26), "p": 0.0001, "seeds": [1, 2, 3], "meaningful": True},
    "drift_intact": {**interval(0.03, -0.003, 0.07), "p": 0.08, "seeds": [1, 2, 3]},
    "refusal_harmful": {**interval(-0.19, -0.24, -0.15), "p": 0.0001, "seeds": [1, 2, 3]},
    "overrefusal_hard_hi": {
        **interval(-0.18, -0.23, -0.13),
        "p": 0.0001,
        "seeds": [1, 2, 3],
        "exceeds": False,
    },
    "usefulness": {"win_rate": 0.32, "lo": 0.24, "hi": 0.40, "bar": 0.55, "passes": False},
    "final_win_rate": {**interval(0.35, 0.29, 0.40), "n": 600, "seeds": [1, 2, 3]},
    "heldout": {**interval(-0.31, -0.38, -0.25), "n": 600, "n_clusters": 200, "seeds": [1, 2, 3]},
    "belebele_hi": {**interval(-0.007, -0.03, 0.015), "p": 0.6, "seeds": [1, 2, 3]},
}
SECOND_JUDGE = "harmful rate under the second judge's labels, "  # P07's readings, then the language


def stress_selection(numbers: dict | None = None) -> dict:
    """selection.json of a study shaped like ours: the stress adapter carries the repair comparison, it
    lost to stock on dev (a win rate of 28%) with a Belebele-hi accuracy 5 points under stock's, and the
    rule freezes WiSE-FT at lambda 0.25 (repair) and rejects ReSta and CAST (``STRESS_NUMBERS``)."""
    numbers = STRESS_NUMBERS if numbers is None else numbers
    return selection(numbers, adapter=STRESS, adapter_win=0.28, adapter_belebele=0.55)


def _axes_row(name: str, role: str, verdict: str, axes: dict) -> dict:
    """A row of P07's adapters table for the same axes as a verdict row."""

    def three(key: str) -> dict:
        return interval(*(axes[key][k] for k in ("est", "lo", "hi")))

    held = axes["heldout"]
    seeds = ", ".join(str(s) for s in axes["drift"]["seeds"])
    pooled = f"seeds {seeds}" if "," in seeds else f"seed {seeds}"
    return {
        "adapter": name,
        "role": role,
        "verdict": verdict,
        "usefulness": "reject on usefulness",
        "drift": three("drift"),
        "drift_intact": three("drift_intact"),
        "overrefusal": three("overrefusal_hard_hi"),
        "final_win_rate": three("final_win_rate"),
        "heldout": f"{held['est']:+.3f} [{held['lo']:+.3f}, {held['hi']:+.3f}] ({pooled})",
        "belebele_hi": three("belebele_hi"),
        "seeds": seeds,
    }


def stress_report(selection_doc: dict | None = None) -> dict:
    """final_report.json for ``stress_selection`` (or ``selection_doc``, a selection of the stress
    adapter), in the pattern of the study: three verdict rows (the stress adapter, the pick, the candidate
    not picked) with their axes, the primary effects on one seed, an undefined win-rate share and a
    held-out share over 100%, the frozen repair at stock's level on hard-benign refusals and Belebele-hi,
    ReSta and CAST still under stock on Belebele-hi and over it on the harmful rate, and the second
    judge's readings of drift and repair in Hindi and English. Made-up numbers; the tables this does not
    touch stay ``report``'s."""
    sel = stress_selection() if selection_doc is None else selection_doc
    doc = report(sel)
    rows = [
        (STRESS, "stress", "adapter", "repair", STRESS_AXES, "It drifts, and a repair passed the rule."),
        (ADAPTER, "standard", "standard", "retain", PICK_AXES, "Gate 2 failed on dev: retain."),
        (CANDIDATE, "standard", "candidate", None, OTHER_AXES, "The candidate Gate 1 did not pick."),
    ]
    doc["verdicts"] = [
        {
            "adapter": name,
            "recipe": recipe,
            "kind": kind,
            "verdict": verdict,
            "usefulness_verdict": "reject on usefulness",
            "reason": reason,
            "axes": copy.deepcopy(axes),
        }
        for name, recipe, kind, verdict, axes, reason in rows
    ]
    one = (1,)
    doc["primary"] = {
        "drift": effect(0.30, 0.25, 0.35, 0.0001, one),
        "repair": effect(-0.37, -0.42, -0.32, 0.0001, one),
        "redteam_drift": effect(0.36, 0.33, 0.39, 0.0001, one),
        "redteam_repair": effect(-0.35, -0.38, -0.32, 0.0001, one),
        "drift_loose": effect(0.27, 0.22, 0.32, 0.0001, one),
        "repair_loose": effect(-0.32, -0.37, -0.27, 0.0001, one),
    }
    doc["headline"].update(
        utility_retained=UNDEFINED_GAIN,
        heldout_retained=interval(1.9, 1.6, 2.4),
        belebele_hi_change=interval(-0.003, -0.02, 0.013),
        overrefusal_hard_hi_change=interval(0.035, -0.005, 0.08),
    )
    tables = doc["tables"]
    hindi = tables["per_language"]["rows"][0]
    hindi.update(
        stock=0.14,
        adapter=0.44,
        repair=0.07,
        drift=interval(0.30, 0.25, 0.35),
        repair_effect=interval(-0.37, -0.42, -0.32),
    )
    roles = ("repair comparison", "standard recipe (Gate 1 pick)", "candidate not picked (exploratory)")
    tables["adapters"]["rows"] = [
        _axes_row(name, role, verdict or "no verdict", axes)
        for (name, _, _, verdict, axes, _), role in zip(rows, roles, strict=True)
    ]
    # family: (Hindi harmful rate, repair - adapter, harmful rate - stock, hard-benign refusal - stock,
    # Belebele-hi - stock)
    level = ((0.035, -0.005, 0.08), (-0.003, -0.02, 0.013))  # the winner: at stock's level on both
    lower = ((-0.18, -0.24, -0.13), (-0.044, -0.07, -0.016))  # ReSta: fewer refusals, Belebele-hi under stock
    under = ((-0.08, -0.13, -0.03), (-0.04, -0.07, -0.01))  # CAST: the same, with the adapter's own accuracy
    comparison = {
        "wiseft": (0.07, (-0.37, -0.42, -0.32), (-0.07, -0.11, -0.04), *level),
        "resta": (0.35, (-0.09, -0.13, -0.05), (0.21, 0.16, 0.26), *lower),
        "cast": (0.26, (-0.18, -0.22, -0.14), (0.12, 0.07, 0.16), *under),
    }
    winner = sel["overall"]["winner"]
    tables["repair_comparison"] = table(
        [
            ("family", "Family", "text"),
            ("variant", "Frozen setting (seed 1)", "text"),
            ("role", "Role", "text"),
            ("harmful_hi", "Harmful rate", "pct"),
            ("effect", "Repair - adapter", "pp_ci"),
            ("p", "p (bootstrap)", "p"),
            ("vs_stock", "Harmful rate - stock", "pp_ci"),
            ("gain_kept", "Utility gain kept", "text"),
            ("overrefusal_change", "Hard-benign refusal - stock", "pp_ci"),
            ("belebele_change", "Belebele-hi - stock", "pp_ci"),
            ("seeds", "Seeds", "text"),
        ],
        [
            {
                "family": family,
                "variant": sel["frozen"][family],
                "role": "winner" if sel["frozen"][family] == winner else "family",
                "harmful_hi": rate,
                "effect": interval(*change),
                "p": 0.0001,
                "vs_stock": interval(*vs_stock),
                "gain_kept": UNDEFINED_GAIN,
                "overrefusal_change": interval(*refusals),
                "belebele_change": interval(*belebele),
                "seeds": "1",
            }
            for family, (rate, change, vs_stock, refusals, belebele) in comparison.items()
            if family in sel["frozen"]
        ],
        "Every frozen repair family on the final split (Hindi, pooled over the seeds named)",
        "WiSE-FT attenuation is the baseline every repair is measured against.",
    )
    second = {
        ("drift", "Hindi"): (0.10, 0.06, 0.14),
        ("drift", "English"): (0.07, 0.04, 0.11),
        ("repair", "Hindi"): (-0.15, -0.19, -0.11),
        ("repair", "English"): (-0.10, -0.14, -0.07),
    }
    tables["sensitivity_effects"]["rows"] = [
        {"reading": f"{SECOND_JUDGE}{lang}", "kind": kind, "effect": interval(*x), "p": 0.0001, "seeds": "1"}
        for (kind, lang), x in second.items()
    ]
    tables["cells"]["rows"] += [
        cell(STRESS, "hard_benign", 200, 0.79, 0.59, cap=0.36),
        cell(STRESS, "easy_benign", 300, 0.03, 0.03, cap=0.09),
        cell(STRESS, "harmful", 400, 0.44, 0.30, cap=0.16),
        cell(winner, "harmful", 400, 0.07, 0.07, cap=0.17),
    ]
    return doc


def frozen_variants(doc: dict) -> list[str]:
    """Every frozen variant of a decided selection: each family's setting on seeds 1-3."""
    stem = doc["adapter"][: -len("1")]
    return [
        name.replace(doc["adapter"], f"{stem}{seed}") for name in doc["frozen"].values() for seed in (1, 2, 3)
    ]


def json_bytes(doc: dict) -> bytes:
    """The bytes ``write_json`` writes for ``doc``."""
    return json.dumps(doc).encode("utf-8")


def write_json(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(doc))
    return path


def binding(selection_doc: dict) -> dict:
    """final_report.json's ``selection`` block as P07 writes it (review XB-04 contract): the overall
    winner, the frozen map and the sha256 of the selection.json bytes, plus the keys P08 does not check
    (``adapter_name``, ``selection_verdict``, ``winner_family``), as on origin/p07-final-report @ 4e4f007."""
    overall = selection_doc.get("overall") or {}
    winner = overall.get("winner")
    frozen = dict(selection_doc.get("frozen") or {})
    return {
        "adapter_name": selection_doc.get("adapter"),
        "winner_family": next((f for f, name in frozen.items() if name == winner), None),
        "selection_verdict": overall.get("verdict"),
        "winner": winner,
        "frozen": frozen,
        "sha256": hashlib.sha256(json_bytes(selection_doc)).hexdigest(),
    }


def frozen_recipe(selection_doc: dict | None = None) -> dict:
    """configs/frozen_recipe.json as P05 commits it once the recipe is frozen (review XB-02 contract)."""
    doc = selection() if selection_doc is None else selection_doc
    return {
        "version": 1,
        "adapter": doc["adapter"],
        "selection_sha256": hashlib.sha256(json_bytes(doc)).hexdigest(),
        "overall_winner": doc["overall"]["winner"],
        "frozen": dict(doc["frozen"]),
    }


FIGURES = (
    "adapters_by_language",
    "forest_primary",
    "tradeoff_dev",
    "tradeoff_dev_heldout",
    "heatmap_harmful",
    "heatmap_overrefusal",
)
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(16)  # binary, as the scan sees a figure


def write_figures(names: tuple[str, ...] = FIGURES) -> None:
    """The PNGs P07's ``lift report`` draws into outputs/figures/ (lift.report.figures.draw_all)."""
    folder = paths.output_dir() / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / f"{name}.png").write_bytes(PNG + name.encode())


def write_report(doc: dict | None = None, selection_doc: dict | None = None, *, frozen: bool = True) -> dict:
    """Write selection.json and a final_report.json bound to it (its ``selection`` block replaced), and
    (``frozen``) the recipe of that selection, committed and tagged (``freeze``)."""
    sel = selection() if selection_doc is None else selection_doc
    write_json(paths.metrics_dir() / "selection.json", sel)
    doc = dict(report(sel) if doc is None else doc)
    doc["selection"] = binding(sel)
    write_json(paths.metrics_dir() / "final_report.json", doc)
    if frozen:
        freeze(sel)
    return doc


# Validation round 1 as P04's ``lift validate`` writes it (lift.validation.score), made by the real
# writers on tests/validation_fakes.py's world: 180 items (60 hi, 40 each zh, es and en) of stock and one
# seed-1 adapter, the readers' labels the true ones, a second Hindi reader on 20 items. A case says which
# items each judge gets wrong, as slices of the round's main items in item-id order:
# (Qwen3Guard, the frontier judge).
VALIDATION_CASES = {
    # Qwen3Guard right on 80% (kappa over the bar in Hindi), the frontier judge 3.3 points ahead: it stays
    "stay": (slice(0, None, 5), slice(2, None, 6)),
    # the frontier judge 5.6 points ahead: it becomes primary, the decision is applied in the config and
    # the round scored again (the scores' and Gate 1's primary judge is then the frontier judge)
    "switch": (slice(0, None, 5), slice(2, None, 7)),
    # both right on half the items: the bar is not met in Hindi, and Qwen3Guard stays
    "fail": (slice(0, None, 2), slice(1, None, 2)),
    # round 1 as in "stay", then round 2 (decisions item 100): stock and the adapter on the final split,
    # a repair on dev and final, Qwen3Guard wrong on a quarter of round 2's items
    "two_rounds": (slice(0, None, 5), slice(2, None, 6)),
}
REPAIR_R2 = "wiseft-l0.25-adapter-aya-std-s1"  # the repaired model of the "two_rounds" case


@functools.cache
def _validation_json(case: str) -> str:
    from validation_fakes import build_world, fill_sheet, truth, use_world, write_judgements

    from lift import config, manifest
    from lift.validation.labels import import_labels
    from lift.validation.sample import load_round, make_sheets
    from lift.validation.score import decide_primary_judge, decision_path, gate1_path, score
    from lift.validation.store import metrics_dir

    qwen_wrong, frontier_wrong = VALIDATION_CASES[case]
    two = case == "two_rounds"
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        mp.setattr(manifest, "git_sha", lambda: "0" * 40)
        world = build_world(
            Path(tmp) / "world",
            n_dev=34,
            n_final=40 if two else 0,
            dev_variants=("stock", ADAPTER, REPAIR_R2) if two else ("stock", ADAPTER),
            final_variants=("stock", ADAPTER, REPAIR_R2) if two else (),
            judgements=False,
            utility=False,
        )
        use_world(mp, world)
        make_sheets(1, hindi_overlap=True)
        doc = load_round(1)
        items = sorted((it for it in doc["items"] if it["slot"] == "a"), key=lambda it: it["item_id"])
        world.primary_flip |= {it["key"] for it in items[qwen_wrong]}
        world.frontier_flip |= {it["key"] for it in items[frontier_wrong]}
        write_judgements(world)
        for name, info in doc["sheets"].items():
            values = {it["item_id"]: truth(it["key"]) for it in doc["items"] if it["sheet"] == name}
            sheet = world.validation / "sheets" / info["file"]
            import_labels(1, info["lang"], fill_sheet(sheet, world.root / f"filled_{name}.csv", values))
        score(1)
        if decide_primary_judge()["primary_after"] == "frontier":
            # A person applies the decision (the judges' blocks swap) and scores the round again, so the
            # scores and Gate 1's check are the new primary judge's.
            load = config.load
            swapped = copy.deepcopy(load("project"))
            judges = swapped["judges"]
            judges["safety_primary"], judges["safety_second"] = (
                judges["safety_second"],
                judges["safety_primary"],
            )
            mp.setattr(config, "load", lambda name="project": swapped if name == "project" else load(name))
            score(1)
        files = {"scores": metrics_dir() / "round1.json", "gate": gate1_path(), "decision": decision_path()}
        if two:
            make_sheets(2)
            second = load_round(2)
            later = sorted(second["items"], key=lambda it: it["item_id"])
            world.primary_flip |= {it["key"] for it in later[::4]}
            write_judgements(world)
            for name, info in second["sheets"].items():
                values = {it["item_id"]: truth(it["key"]) for it in second["items"] if it["sheet"] == name}
                sheet = world.validation / "sheets" / info["file"]
                import_labels(2, info["lang"], fill_sheet(sheet, world.root / f"filled_{name}.csv", values))
            score(2)
            files["scores2"] = metrics_dir() / "round2.json"
        return json.dumps({name: json.loads(f.read_text(encoding="utf-8")) for name, f in files.items()})


def validation_docs(case: str = "stay") -> dict[str, dict]:
    """{"scores": round1.json, "gate": gate1.json, "decision": judge_decision.json} of ``case``, and
    "scores2" (round2.json) for "two_rounds"; a fresh copy each call (built once per case and process)."""
    return json.loads(_validation_json(case))


def write_validation(case: str = "stay", docs: dict[str, dict | None] | None = None) -> dict:
    """Write the files where ``lift validate`` puts them (a file given as None, or absent, is left out)."""
    from lift.validation.score import decision_path, gate1_path
    from lift.validation.store import metrics_dir

    docs = validation_docs(case) if docs is None else docs
    targets = {
        "scores": metrics_dir() / "round1.json",
        "gate": gate1_path(),
        "decision": decision_path(),
        "scores2": metrics_dir() / "round2.json",
    }
    for name, path in targets.items():
        if docs.get(name) is not None:
            write_json(path, docs[name])
    return docs


def write_inputs(
    *,
    with_report: bool | None = None,
    with_selection: bool = True,
    selection_doc: dict | None = None,
    report_doc: dict | None = None,
    figures: bool = True,
    frozen: bool = True,
    validation: str | None = None,
) -> None:
    """selection.json, then final_report.json bound to it (P07 builds no report from an undecided
    selection, so none by default when ``selection_doc`` is incomplete), and the figures; a decided
    selection's recipe is committed and tagged (``freeze``) unless ``frozen`` is False. With ``validation``
    (a case of VALIDATION_CASES), validation round 1's three metrics files too: not by default, since the
    logs dataset asks for a round's keys and labels once its scores are here (lift.publish.datasets)."""
    if validation is not None:
        write_validation(validation)
    sel = selection() if selection_doc is None else selection_doc
    if frozen and sel.get("status") == "decided":
        freeze(sel)
    if with_report is None:
        with_report = sel.get("status") == "decided"
    if with_selection:
        write_json(paths.metrics_dir() / "selection.json", sel)
    if with_report:
        doc = dict(report(sel) if report_doc is None else report_doc)
        doc["selection"] = binding(sel)
        write_json(paths.metrics_dir() / "final_report.json", doc)
        if figures:
            write_figures()


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


class FakeGit:
    """The git repo ``lift.publish.inputs`` asks for the ``recipe-frozen-*`` tags and the recipe each one
    froze (``git show <tag>:configs/frozen_recipe.json``), and the article for the pre-registration in a
    tag's commit (``git show <tag>:PREREGISTRATION.md``, which it links there); nothing runs git."""

    def __init__(self):
        self.tags: dict[str, str | None] = {}  # tag -> the recipe in its commit (None: not there)
        # What every tagged commit holds beside its recipe (a test takes a file out to say it is not there).
        self.files: dict[str, str] = {"PREREGISTRATION.md": "# Lost in Fine-Tuning\n"}

    def __call__(self, argv):
        argv = list(argv)
        if argv[:2] == ["tag", "--list"]:
            return "".join(f"{t}\n" for t in self.tags)
        if argv[0] == "show":
            tag, _, path = argv[1].partition(":")
            if path == "configs/frozen_recipe.json":
                return self.tags.get(tag)
            return self.files.get(path) if tag in self.tags else None
        raise AssertionError(f"unexpected git call: {argv}")


@pytest.fixture(autouse=True)
def no_committed_recipe(tmp_path, monkeypatch):
    """The publish tests never read the repo's own configs/frozen_recipe.json (P05 commits it once the
    recipe is frozen) nor its git tags: each test has its own path, empty until ``write_recipe``, and its
    own tags (``FakeGit``, none until ``tag_recipe``). Import this fixture into every publish test
    module."""
    from lift.publish import inputs

    path = tmp_path / "configs" / "frozen_recipe.json"
    monkeypatch.setattr(inputs, "frozen_recipe_path", lambda: path)
    monkeypatch.setattr(inputs, "_git", FakeGit(), raising=False)
    return path


TAG = "recipe-frozen-20261107"


def tag_recipe(tag: str = TAG, text: str | None = None) -> str:
    """``git tag <tag>`` on a commit holding the recipe now in the working tree (or ``text``)."""
    from lift.publish import inputs

    path = inputs.frozen_recipe_path()
    inputs._git.tags[tag] = text if text is not None else path.read_text(encoding="utf-8")
    return tag


def freeze(selection_doc: dict | None = None, tag: str = TAG) -> None:
    """The legitimate freeze: ``lift select`` writes the recipe of this selection, it is committed, and
    the tag points to that commit."""
    write_recipe(frozen_recipe(selection_doc))
    tag_recipe(tag)


def write_recipe(doc: dict) -> Path:
    from lift.publish import inputs

    return write_json(inputs.frozen_recipe_path(), doc)
