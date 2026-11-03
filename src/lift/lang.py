"""GlotLID language identification (cis-lmu/glotlid, fastText). Implemented in P01.

Used twice: asserting each prompt's language at build time, and the unclear/wrong-language flags
on every output. Gotchas:
- fastText rejects newlines: replace them with spaces before ``predict``.
- fasttext 0.9.2 breaks under numpy 2 ("Unable to avoid copy"); use fasttext>=0.9.3.
- Labels come back as ``__label__hin_Deva``; strip the prefix.
- Check that every label in configs/project.yaml ``languages.accept_labels`` exists in the
  model's label set (``model.get_labels()``) and fix the config if one does not.
"""

from __future__ import annotations


class LangID:
    def __init__(self, model_path: str | None = None) -> None:
        """Load GlotLID from ``model_path`` or download it with huggingface_hub."""
        raise NotImplementedError("P01")

    def top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k (label, probability), labels without the ``__label__`` prefix."""
        raise NotImplementedError("P01")

    def is_lang(self, text: str, lang: str) -> bool:
        """True if the top label is accepted for ``lang`` with probability >= min_prob."""
        raise NotImplementedError("P01")
