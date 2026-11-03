"""Other prompt sources. Implemented in P01. Needs network (HF Hub).

- OR-Bench-Hard-1K (bench-llm/or-bench, config or-bench-hard-1k, CC-BY-4.0): sample 400 rows,
  stratified by ``category`` with the project split seed; pid ``orb-{row index in the HF dataset}``.
  Split 100 fit / 100 dev / 200 final, then translated (lift.data.translate).
- Aya Red-teaming Hindi (CohereLabs/aya_redteaming, Hindi): all rows, set "redteam", split
  "final"; keep ``harm_category`` and ``global_or_local`` in ``extra``. Human-written, so no
  translation. Second final set for the primary endpoint only, never used for selection.
- Belebele (facebook/belebele): configs eng_Latn hin_Deva zho_Hans spa_Latn arb_Arab tha_Thai,
  900 parallel items each; pid ``bel-{i:03d}`` by row index (identical across languages);
  200 dev / 700 final.
"""

from __future__ import annotations

from lift.schemas import PromptRecord


def load_orbench_hard_sample(n: int = 400) -> list[PromptRecord]:
    raise NotImplementedError("P01")


def load_aya_redteaming_hindi() -> list[PromptRecord]:
    raise NotImplementedError("P01")


def load_belebele(lang: str) -> list[dict]:
    """Rows with pid, flores_passage, question, mc_answer1..4, correct_answer_num (1-4)."""
    raise NotImplementedError("P01")
