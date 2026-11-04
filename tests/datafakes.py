"""Offline stand-ins for the P01 data pipeline's external services, shared by the data tests.

Nothing here touches the network, fasttext, datasets or CuratorKIT. Not a conftest: P02 owns
tests/conftest.py, so data tests import this module directly (``from datafakes import ...``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence

from lift import config
from lift.lang import accepted

_SCRIPTS = (
    ("hin_Deva", re.compile(r"[ऀ-ॿ]")),
    ("cmn_Hani", re.compile(r"[一-鿿]")),
    ("arb_Arab", re.compile(r"[؀-ۿ]")),
    ("tha_Thai", re.compile(r"[฀-๿]")),
)
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_SPANISH = re.compile(r"[¿¡ñÑáéíóú]")


def script_label(text: str) -> tuple[str, float]:
    """Majority-script guess: Devanagari -> hin_Deva, Han -> cmn_Hani, Arabic, Thai, else Latin
    (spa_Latn if it has Spanish marks, eng_Latn otherwise). Nothing recognisable -> und_Zyyy 0.0."""
    counts = {label: len(rx.findall(text)) for label, rx in _SCRIPTS}
    counts["latin"] = len(_LATIN.findall(text))
    label, n = max(counts.items(), key=lambda kv: kv[1])
    if n == 0:
        return "und_Zyyy", 0.0
    if label == "latin":
        return ("spa_Latn" if _SPANISH.search(text) else "eng_Latn"), 0.95
    return label, 0.99


class FakeLID:
    """Duck-types lift.lang.LangID (``top`` and ``is_lang``) with a script heuristic.

    ``overrides`` maps an exact text to the (label, prob) to return, or to a list of them (best
    first), to simulate GlotLID errors.
    """

    def __init__(self, overrides: Mapping[str, tuple[str, float] | list[tuple[str, float]]] | None = None,
                 *,
                 accept_labels: Mapping[str, Sequence[str]] | None = None,
                 min_prob: float = 0.5) -> None:
        self.overrides = dict(overrides or {})
        self.accept = {k: list(v) for k, v in
                       (accept_labels or config.get("languages.accept_labels")).items()}
        self.min_prob = min_prob
        self.calls = 0

    def top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        self.calls += 1
        if text in self.overrides:
            value = self.overrides[text]
            return list(value)[:k] if isinstance(value, list) else [value]
        return [script_label(text)]

    def is_lang(self, text: str, lang: str) -> bool:
        return accepted(self.top(text, 5), self.accept, lang, self.min_prob)


# A few words in each target script, so FakeLID assigns the fake output to the target language.
_FAKE_WORDS = {
    "hi": "अनुवादित परीक्षण पाठ",
    "zh": "翻译测试文本",
    "es": "¡Traducción de prueba número!",
    "ar": "نص اختبار مترجم",
    "th": "ข้อความทดสอบที่แปลแล้ว",
    "en": "translated test text",
}


def fake_translation(text: str, lang: str) -> str:
    """Deterministic, unique per source, never equal to it, written in the target's script (the
    suffix is digits only, so it never tips the script count)."""
    digest = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % 10**8
    return f"{_FAKE_WORDS[lang]} {digest:08d}"


class FakeTranslator:
    """A translation backend ``(english_text, lang) -> text`` with injectable failures.

    - ``empty_for``: return "" for these sources
    - ``identical_for``: return the source unchanged
    - ``english_for``: return English (wrong language) for these sources
    - ``raise_times``: raise a transient error this many times per source, then succeed
    - ``always_raise``: raise every time for these sources
    ``calls`` records every (text, lang) the backend received, including the ones that raised.
    """

    def __init__(self, *, empty_for: Iterable[str] = (), identical_for: Iterable[str] = (),
                 english_for: Iterable[str] = (), raise_times: Mapping[str, int] | None = None,
                 always_raise: Iterable[str] = ()) -> None:
        self.empty_for = set(empty_for)
        self.identical_for = set(identical_for)
        self.english_for = set(english_for)
        self.raise_left = dict(raise_times or {})
        self.always_raise = set(always_raise)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, text: str, lang: str) -> str:
        self.calls.append((text, lang))
        if text in self.always_raise:
            raise ConnectionError("fake transient failure")
        if self.raise_left.get(text, 0) > 0:
            self.raise_left[text] -= 1
            raise ConnectionError("fake transient failure")
        if text in self.empty_for:
            return ""
        if text in self.identical_for:
            return text
        if text in self.english_for:
            return fake_translation(text, "en")
        return fake_translation(text, lang)
