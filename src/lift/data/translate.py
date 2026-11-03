"""Machine translation for the prompt sets. Implemented in P01. Needs network.

What gets translated (from English sources, never from another translation):
- Hindi: PolyRefuse harmful test (572), harmful train (260), harmful val (39), harmless test (500),
  harmless train (200). Google Translate, matching how PolyRefuse was built.
- Chinese: PolyRefuse harmless test (500). Google Translate.
- OR-Bench-Hard sample (400) into hi, zh, es, ar, th. Google Translate (NLLB-200 as fallback).

Rules: every call is cached on disk keyed by (engine, target, sha256(source)) so reruns are free and
deterministic; an empty output, an output identical to the source, or an output GlotLID does not
assign to the target language raises (collect failures, report them, never silently keep English);
batch and retry with backoff (deep-translator's GoogleTranslator is rate limited).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lift.schemas import PromptRecord


def translate_texts(texts: Sequence[str], target: str, *, engine: str = "google",
                    cache_dir: Path | None = None) -> list[str]:
    raise NotImplementedError("P01")


def translate_records(records: Sequence[PromptRecord], target_lang: str, *,
                      engine: str = "google") -> list[PromptRecord]:
    """Copy each record into ``target_lang`` with ``text_en`` = source and TranslationInfo set."""
    raise NotImplementedError("P01")


def back_translate(texts: Sequence[str], source_lang: str) -> list[str]:
    """NLLB-200 back-translation to English, for the ar/th judge sensitivity sample. Raises on
    failure (never returns the input untranslated)."""
    raise NotImplementedError("P01")
