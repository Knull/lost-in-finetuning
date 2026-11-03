"""Build every prompt set and freeze the splits. Implemented in P01.

Order: load English sources -> load PolyRefuse translations -> translate the missing sets ->
assign splits once per set from sorted pids (lift.splits, sizes from configs/project.yaml) and
write data/splits/{set}.json -> stamp ``split`` on every record -> GlotLID-assert each prompt's
language (log and list failures; Devanagari and romanised Hindi both pass) -> write
data/prompts/{set}_{lang}.jsonl.

Files produced (one PromptRecord per line):
    harmful_{lang}.jsonl        test rows split dev/final + train/val rows (CAST fit), all 6 langs
    easy_benign_{lang}.jsonl    test rows split dev/final + train rows (CAST fit), all 6 langs
    hard_benign_{lang}.jsonl    OR-Bench sample, fit/dev/final, all 6 langs
    redteam_hi.jsonl            Aya Red-teaming Hindi, final
    utility_hi.jsonl            shared held-out utility set, dev/final
    belebele_{lang}.jsonl       stored as dicts with pid + split (not PromptRecord)

Split files are committed to git and never recomputed: a second build must reproduce them
exactly (lift.splits.save_split raises otherwise).
"""

from __future__ import annotations


def build_all(*, skip_translation: bool = False) -> None:
    raise NotImplementedError("P01")


def load_prompts(set_name: str, lang: str, split: str | None = None) -> list:
    """Read data/prompts/{set}_{lang}.jsonl, optionally filtered to one split."""
    raise NotImplementedError("P01")
