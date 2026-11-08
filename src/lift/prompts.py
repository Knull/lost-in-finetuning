"""Prompt records as the harness sees them. Every GPU stage reads prompts through ``load``.

The file format belongs to P01: ``lift.data.build.load_prompts`` reads data/prompts/{set}_{lang}.jsonl
and checks every record's split stamp against the committed data/splits/{set}.json first (ValueError
for a stamp the committed split does not give, or a missing split file), so dev/final membership never
comes from a synced file alone. A read of one split also refuses a cell short of its frozen split
(``lift.data.build.ShortCellError``, a ValueError naming the remedy: most often a translation failure
that awaits a reader's verdict on the failures sheet); every stage that makes or reports a cell reads
its split. A read of every split (``texts_by_pid``) checks the stamps only. This module checks what it
gets back, so a mislabelled or duplicated record fails here instead of producing a wrong denominator
downstream.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lift.io import sha256_text
from lift.schemas import LANGS, PromptRecord

SAFETY_SETS: tuple[str, ...] = ("harmful", "easy_benign", "hard_benign")


def load(set_name: str, lang: str, split: str | None = None, *,
         allow_short: bool = False) -> list[PromptRecord]:
    """PromptRecords of one set and language (optionally one split), sorted by pid. The split is the
    committed one (build.load_prompts refuses a file whose stamps differ from data/splits), and with
    ``split`` every frozen pid of it is there, unless the file's report lists it as left out on purpose
    (a reader's bad verdict; a fit record whose text a dev/final record has): ShortCellError otherwise.
    ``allow_short`` (passed on to build.load_prompts) returns the split as the file holds it, for a caller
    that only pools pids (lift.judge.pipeline.second_judge_pids)."""
    from lift.data import build

    rows = build.load_prompts(set_name, lang, split, allow_short=allow_short)
    records = [r if isinstance(r, PromptRecord) else PromptRecord.model_validate(r) for r in rows]
    wrong = [r.pid for r in records
             if r.set != set_name or r.lang != lang or (split is not None and r.split != split)]
    if wrong:
        raise ValueError(f"{set_name}_{lang}: {len(wrong)} records belong elsewhere, e.g. {wrong[:5]}")
    pids = [r.pid for r in records]
    if len(set(pids)) != len(pids):
        dupes = sorted({p for p in pids if pids.count(p) > 1})
        raise ValueError(f"{set_name}_{lang}: duplicate pids {dupes[:5]}")
    return sorted(records, key=lambda r: r.pid)


def texts_by_pid(set_name: str, lang: str) -> dict[str, str]:
    """{pid: text} for every split of one set and language (what the judge needs, and Gate 0's precision
    pairs). A read of every split: the stamps are checked, not the frozen sizes, so a split of the set and
    language short of its frozen split (whose own read refuses it) never stops a lookup."""
    return {r.pid: r.text for r in load(set_name, lang)}


def mixed_sample(n: int, *, split: str = "dev", sets: Sequence[str] = SAFETY_SETS,
                 langs: Sequence[str] = LANGS, seed: int = 0) -> list[PromptRecord]:
    """``n`` prompts spread as evenly as possible over the set x language cells, seeded.

    Cells are visited in a fixed order; each takes ``n // cells`` prompts and the first
    ``n % cells`` cells one more. Each cell draws with its own generator, seeded from ``seed`` and
    the cell's name, so the same call always returns the same prompts and one cell's pool never
    changes another cell's draw.
    """
    cells = [(s, lang) for s in sets for lang in langs]
    if n < 1 or not cells:
        raise ValueError("need n >= 1 and at least one set and language")
    base, extra = divmod(n, len(cells))
    out: list[PromptRecord] = []
    for i, (set_name, lang) in enumerate(cells):
        want = base + (1 if i < extra else 0)
        if want == 0:
            continue
        pool = load(set_name, lang, split)
        if want > len(pool):
            raise ValueError(f"{set_name}_{lang}_{split}: asked for {want} prompts, only {len(pool)}")
        cell_seed = int(sha256_text(f"{seed}:{set_name}:{lang}:{split}")[:16], 16)
        picks = np.random.default_rng(cell_seed).permutation(len(pool))[:want]
        out.extend(pool[int(j)] for j in sorted(picks))
    return out
