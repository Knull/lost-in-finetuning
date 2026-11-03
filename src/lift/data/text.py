"""Text helpers shared by the data loaders: one cleaning rule, one duplicate key, one similarity."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

# Zero-width spaces and stray BOMs ride along in PolyRefuse's translations (230 U+200B, mostly nl
# and ar). ZWJ and ZWNJ (U+200D, U+200C) are NOT removed: Indic scripts need them.
_INVISIBLE = dict.fromkeys(map(ord, "​﻿"))


def clean_text(text: str) -> str:
    """The text every prompt record carries: zero-width spaces and BOMs dropped, ends stripped.
    Inner newlines and spacing are kept as the source wrote them."""
    return text.translate(_INVISIBLE).strip()


def dup_key(text: str) -> str:
    """Key for exact-duplicate checks: NFKC, casefolded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", clean_text(text)).casefold().split())


def char_ngrams(text: str, n: int = 3) -> frozenset[str]:
    """Character n-grams of ``dup_key(text)``. Characters, not words, so Hindi, Chinese and Thai
    are treated alike. A text shorter than n is its own single gram."""
    s = dup_key(text)
    if len(s) <= n:
        return frozenset([s]) if s else frozenset()
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
