"""PolyRefuse loaders (mainlp/Multilingual-Refusal, Apache-2.0). Implemented in P01.

Field mapping, verified Nov 3 against a clone of the repo:
- ``PolyRefuse/harmful_test_translated_{lang}.json`` (572 rows) and
  ``PolyRefuse/harmless_test_translated_{lang}.json`` (500 rows): ``instruction`` is the ENGLISH
  source, ``instruction_translated`` is the target-language text, plus ``category``. The ``en``
  harmful file has only ``instruction``. Rows are aligned by index across languages (the English
  ``instruction`` matches row for row).
- ``PolyRefuse/{harmful,harmless}_{train,val}_translated_{lang}.json`` (harmful 260/39, harmless
  200/200): ``instruction`` holds the TRANSLATED text and there is no English field. There is no
  ``en`` file; the English sources are ``dataset/splits/harmful_{train,val}.json`` and
  ``dataset/splits/harmless_{train,val}_200_sampled.json`` (check which file aligns, row for row,
  before trusting it).
- Languages present: ar de en es fr it ja ko nl pl ru th yo zh. No Hindi anywhere, and no zh
  harmless test file. Those come from the translation job (lift.data.translate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from lift.schemas import PromptRecord

Kind = Literal["harmful", "harmless"]
PSplit = Literal["test", "train", "val"]


def load_polyrefuse(root: Path, kind: Kind, split: PSplit, lang: str) -> list[PromptRecord]:
    """Records in file order. pid = ``prh-{split}-{i:04d}`` (harmful) or ``prb-...`` (harmless).

    ``set`` is "harmful" or "easy_benign"; ``split`` is a placeholder ("train" for train/val files,
    "dev" for test files) that lift.data.build overwrites from the committed split file.
    ``text_en`` is always filled (from the English source file where needed).
    """
    raise NotImplementedError("P01")


def english_sources(root: Path, kind: Kind, split: PSplit) -> list[str]:
    """English source texts, aligned by index with every translated file of the same kind/split."""
    raise NotImplementedError("P01")
