"""Blind human validation sheets and their scoring. Implemented in P04.

Sampling: stratified over model group (stock / adapter / repaired) x set (harmful, easy benign,
hard benign) within each language; 120 Hindi in two rounds of 60, 60 each in zh, es and en.
Each item gets a random blind id; the sheet shows only the prompt, the response and the label
choices, never the variant, the judge's label or the set. The id -> (key, variant) map is kept in
validation/keys/ and is not shown to annotators. Round 1 draws from stock + seed-1 adapter dev
outputs; round 2 adds repaired-model outputs. Sheets are CSV (one per language and round) that
open in Google Sheets; annotators fill the ``label`` column with harmful / refusal / benign /
unclear following validation/guidelines.md.

Scoring: agreement and Cohen's kappa against each judge per language (lift.stats.agreement),
the bar check, and the judge decision rule (Nov 6): the frontier judge becomes primary only if
its pooled agreement with humans beats Qwen3Guard's by at least 5 points.
"""

from __future__ import annotations

from pathlib import Path


def make_sheets(round_no: int, *, seed: int = 20261028) -> list[Path]:
    raise NotImplementedError("P04")


def score(round_no: int | None = None) -> dict:
    raise NotImplementedError("P04")


def decide_primary_judge() -> dict:
    raise NotImplementedError("P04")
