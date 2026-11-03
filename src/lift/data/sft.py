"""Adapter training candidates and the shared utility held-out set. Implemented in P01.

Two candidates, both Hindi instruction data (chosen on dev utility win rate only, never on drift):
- ``aya``: CohereLabs/aya_dataset rows with language Hindi.
- ``indicalign``: one Hindi subset of ai4bharat/indic-align. Check the licence of the subset
  first and record it in docs/decisions.md; pick a single-turn instruction subset.

Pipeline per candidate:
1. Load Hindi rows; normalise to (instruction, response) single turns; drop empty or
   non-Hindi rows (GlotLID on both sides).
2. Carve the shared utility held-out pool first: 150 rows from each source = 300, pid
   ``utl-{aya|ind}-{i:05d}``, split 100 dev / 200 final. Remove every near-duplicate of a held-out
   row from BOTH candidates' training pools (MinHash, threshold recorded), so neither adapter
   trains on anything the utility judge scores.
3. CuratorKIT no-API path on the rest: ``Curator(CuratorConfig(dataset=<local jsonl>,
   dedup="minhash", clean=True, export_formats=["sft_messages"] ...)).run()``. Group
   near-duplicates before sampling. Hygiene gates (secrets, PII) are optional; do not apply
   CuratorKIT's English toxicity filter to Hindi. Keep CuratorKIT's manifest.json,
   rejected.jsonl and lexsi_provenance.json next to the export.
4. Sample ``rows`` (3,000) with the split seed. The stress recipe also needs an UNdeduplicated
   3,000-row sample of the aya candidate (same held-out exclusion).

Outputs: data/sft/{candidate}/{dedup|raw}/sft_messages.jsonl (+ CuratorKIT files) and
data/prompts/utility_hi.jsonl. Training rows use the messages format TRL's SFTTrainer expects:
{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}.
"""

from __future__ import annotations

from pathlib import Path

from lift.schemas import PromptRecord


def build_utility_heldout() -> list[PromptRecord]:
    raise NotImplementedError("P01")


def build_candidate(name: str, *, dedup: bool = True, rows: int = 3000) -> Path:
    """Return the directory holding sft_messages.jsonl and the CuratorKIT provenance files."""
    raise NotImplementedError("P01")
