"""Judge a generations file into a judgements file. Implemented in P02.

For each GenerationRecord: GlotLID on the output -> unclear flags -> judge -> map -> final label
(lift.judge.labels) -> JudgementRecord appended to
outputs/judgements/{judge_short}/{variant}/{set}_{lang}_{split}.jsonl. Resumable by key.
A record is written for EVERY generation (every prompt stays in the denominator); a judge failure
is written as label "unclear" with ``native.judge_failure`` set, and counted in the report.
"""

from __future__ import annotations

from pathlib import Path


def judge_file(generations_path: Path, *, judge: str = "primary", resume: bool = True) -> Path:
    raise NotImplementedError("P02")


def load_labels(variant: str, set_name: str, lang: str, split: str, *,
                judge: str = "primary") -> dict[str, str]:
    """{pid: final label} for one variant/set/language/split."""
    raise NotImplementedError("P02")
