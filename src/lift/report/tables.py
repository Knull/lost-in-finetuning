"""Results tables, figures and the cost section. Implemented in P07.

Tables (markdown + CSV under outputs/metrics/):
- Per variant x language x set: harmful rate with the unclear upper bound, refusal rate,
  unclear rate, wrong-language rate, cap-hit rate, n.
- Primary endpoint: Hindi harmful rate on the final set (PolyRefuse and Aya Red-teaming),
  adapter vs stock (drift) and frozen repair vs adapter (repair), paired by prompt, each seed
  shown, McNemar per seed and a clustered bootstrap pooled over seeds.
- Secondary: pooled zh+es transfer effect and English, Holm-corrected. Exploratory: ar, th,
  per-language effects with BH-FDR.
- Utility win rate vs stock with intervals and the share of the adapter's gain retained;
  Belebele per language; easy and hard over-refusal per language.
- The verdict per adapter: retain, repair or reject.
Figures: per-language heatmaps (harmful rate, over-refusal), the safety-utility trade-off of every
dev setting, signatures. Cost: GPU-hours per check-up and per-token latency for each repair, from
the run manifests. No commercial ROI claims (Tiny Aya derivatives are non-commercial).
Arms that were not run are listed as not run, never estimated.
"""

from __future__ import annotations


def build_all() -> None:
    raise NotImplementedError("P07")
