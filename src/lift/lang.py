"""GlotLID language identification (cis-lmu/glotlid, fastText). Implemented in P01.

Used twice: asserting each prompt's language at build time, and the unclear/wrong-language flags
on every output. Gotchas, all verified on Nov 3 against the pinned revision:
- fastText reads one line at a time: newlines inside the text are replaced with spaces, and the
  text must END with a newline. The high-level ``model.predict`` appends it; calling the low-level
  ``model.f.predict`` without it gives garbage (a plain Hindi sentence came back as Angika, 0.99).
- fasttext 0.9.2 (``fasttext-wheel``, the only build that installs on Windows) breaks under numpy 2
  in ``model.predict`` ("Unable to avoid copy"). We always call ``model.f.predict(text + "\\n", k,
  0.0, "strict")``, which works on 0.9.2 and 0.9.3 alike.
- Labels come back as ``__label__hin_Deva``; the prefix is stripped. fastText smooths probabilities
  by 1e-5, so a sure prediction can read 1.00001; probabilities are clamped to [0, 1].
- Every label in configs/project.yaml ``languages.accept_labels`` must exist in the model's label
  set (``model.get_labels()``); loading raises otherwise. GlotLID has no macrolanguage labels (no
  zho_Hans or zho_Hani: Mandarin in either script is cmn_Hani), and it spreads genuine Arabic and
  Chinese across variety labels (14% of Belebele's Modern Standard Arabic passages are not
  arb_Arab). So zh and ar accept their varieties, and ``is_lang`` sums the accepted labels' mass.
- Blank text is Classical Chinese to GlotLID. fastText splits words on ASCII whitespace and NUL, so
  empty, whitespace-only (256 newlines included) or NUL-only text reaches it as the end-of-line
  token alone, and the answer is lzh_Hani 1.0, which zh accepts. Other invisible-only texts get
  confident labels too: BOM lzh_Hani 0.98, ideographic spaces lzh_Hani 0.99, ZWNJ yue_Hani 1.0,
  three NBSPs eng_Latn 0.93 (one: urd_Arab 0.99). So ``top`` returns [BLANK] =
  [("und_Zyyy", 0.0)] (not a GlotLID label) without calling the model when the text has no
  visible character (``has_visible``: whitespace, control and format characters only; U+200B and
  U+FEFF are format characters). Then ``is_lang`` is False and lift.judge.labels marks the output
  garbled.
- Unsegmented Chinese: fastText splits words on whitespace only, so a Chinese clause is one long
  word (GlotLID's 634,361-word vocabulary has 318 words with a Han character, 290 of them single
  characters), and short clauses get confident rare-script labels. 38 of the 870 PolyRefuse zh
  harmful prompts (test, train, val; prh-train-0226 repeats prh-test-0008), 10 to 43 characters,
  failed ``is_lang``: zxx_Latn 9, und_Tang 5 (up to 1.0), und_Cyrl 4, und_Egyp, und_Sgnw and
  und_Yiii 3 each, cmn_Hani under 0.5 3, and others. So ``top`` puts a space on each side of every
  CJK ideograph (``space_han``; runs of spaces predict exactly like one), whatever the length.
  ``is_lang`` failures unspaced -> spaced, measured Nov 3-4 at the pinned revision:
  - PolyRefuse zh harmful 38 -> 0 of 870 (test 24, train 11, val 3), harmless train and val
    9 -> 0 of 400. Belebele zh questions 1 -> 0 and passages 0 -> 0 of 900; answer options
    448 -> 206 of 3,600 (297 fixed; 55 of 2 to 14 characters, 23 of them two, newly fail, 36 as
    jpn_Jpan).
  - Answers with code or tables, which P02 gets on easy_benign (Alpaca; 15 of its 500 test
    prompts ask for code outright), fail unspaced, mostly as garbled: a Belebele passage then a
    10-line Python block 64 -> 0 of 300, one sentence around a 6-line block 228 -> 0 of 300;
    hand-written, 4 of 20 answers with code or formulas (87 to 199 non-space characters) and 3 of
    10 lists and tables (a 140-character table read glg_Latn 0.76) -> 0. So spacing has no length
    limit: unspaced, long prose passes (Belebele passages; 300 concatenations of two or three: 0
    fail), long answers with code do not. An answer that is mostly code stays garbled (a sentence
    around 15 lines: 92 of 300).
  - Hand-written traditional Chinese (6) and Cantonese (6) pass either way; 4 Cantonese sentences
    move from yue_Hani to cmn_Hani, both zh. No Belebele en/hi/es/ar/th passage or question and no
    PolyRefuse es/ar/th harmful test prompt has a Han character: their predictions are untouched.
- What spacing costs (Nov 4). Character salad reads cmn_Hani near 1.0 once spaced, so it is rarely
  garbled: random Han characters drawn by frequency from Belebele passages, 0 of 200 flagged at any
  length from 10 to 600, where unspaced lines flagged 97 of 200 at 10-29 characters and 199 at
  300-600. Unspaced lines missed most token-like salad (1-3-character pieces of real text) anyway:
  at most 45 of 200 (10-29 characters), 7 or fewer at 300-600; spaced, 0. Mixed text weighs each
  ideograph like a word, where an unspaced clause weighed little: a Chinese opening continued in
  English turns English (wrong_lang on a zh prompt) after 14, 15, 35 and 71 English words for
  openings of 9, 14, 38 and 77 ideographs (unspaced 5, 5, 12 and 20), and a 7-word English lead-in
  to a 17-ideograph Chinese sentence reads cmn_Hani 1.0 (unspaced eng_Latn 0.68). 8 hand-written
  English answers with Chinese words (2 to 16 ideographs) read eng_Latn either way.
- Japanese is never spaced: text with any hiragana or katakana letter is left whole (the middle
  dot U+30FB and the prolonged sound mark U+30FC do not count: Chinese writes them too). Spacing
  the kanji of PolyRefuse's 1,771 Japanese rows moved 7 off jpn_Jpan, 4 of them to cmn_Hani: a
  Japanese answer to a zh prompt would then pass as Chinese instead of wrong_lang. With the kana
  guard every row reads as before (jpn_Jpan top for 1,769, none accepted as zh); every one has a
  kana letter. None of the 6,670 Chinese texts above contains any U+3040-30FF character, so the
  guard skips none of them. Kanji-only Japanese is spaced and is a
  coin toss either way (of two hand-written kanji-only phrases, one went cmn_Hani -> jpn_Jpan,
  the other jpn_Jpan -> lzh_Hani).
- Short texts stay less reliable: 206 of Belebele's 3,600 zh answer options still fail. 83 have
  no Han character (kor_Hang 22, kin_Latn 12, slk_Latn 11, ...); 76 of the other 123 have only one
  or two ideographs, and 76 of the 123 read jpn_Jpan (zxx_Latn 20, kor_Hang 7). So a prompt that
  fails is listed, not silently dropped.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from lift.schemas import LidInfo

_PREFIX = "__label__"

# BLANK, has_visible, space_han (with _HAN and _KANA) and model_line (with one_line) decide what LangID.top
# returns, and with it the stored flags.lid_top and the garbled, wrong_lang and code_mixed flags. A change
# to any of them changes stored labels: bump lift.judge.labels.LABELS_VERSION with it.
BLANK: tuple[str, float] = ("und_Zyyy", 0.0)   # top() of text with no visible character
# The labels ``is_lang`` sums and the unclear flags read (lift.judge.labels.unclear_flags): GlotLID's top 5,
# so garbled means exactly "not is_lang" for an output whose top label is accepted (decisions item 43).
LID_TOP_K = 5

# CJK unified ideographs: Extension A, the main block, the compatibility ideographs, Extensions B-F
# and the compatibility supplement. Kana, Hangul, bopomofo and CJK punctuation are not spaced.
_HAN = re.compile("([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
                  "\U00020000-\U0002ebef\U0002f800-\U0002fa1f])")
# Hiragana and katakana letters: the text is Japanese. Not the middle dot U+30FB or the prolonged
# sound mark U+30FC, which Chinese also writes (a name joined by U+30FB read und_Tang 0.66 unspaced).
_KANA = re.compile("[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff]")


class LangIdentifier(Protocol):
    """What the pipeline needs from a language identifier; tests pass a fake with these methods.

    ``top`` of text with no visible character is [BLANK], as tests/datafakes.py FakeLID does."""

    def top(self, text: str, k: int = 3) -> list[tuple[str, float]]: ...

    def is_lang(self, text: str, lang: str) -> bool: ...


# one_line, has_visible, space_han and model_line are LangID.top's input handling, like BLANK: a change to
# any of them changes stored labels and needs a lift.judge.labels.LABELS_VERSION bump.
def one_line(text: str) -> str:
    """The text as one fastText input line: inner line breaks become spaces, plus the final newline."""
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ") + "\n"


def has_visible(text: str) -> bool:
    """True if the text has a character that is not whitespace, a control character (fastText also
    splits on NUL) or a format character (zero-width spaces, BOMs, joiners, direction marks)."""
    return any(not c.isspace() and unicodedata.category(c) not in ("Cc", "Cf") for c in text)


def space_han(text: str) -> str:
    """The text with a space on each side of every CJK ideograph, so fastText reads unsegmented
    Chinese one character per word, at any length (long answers with code fail unspaced). Text
    with any hiragana or katakana letter is Japanese and comes back unchanged, like text without
    ideographs: spacing its kanji costs jpn_Jpan. Costs and numbers: module docstring."""
    if _KANA.search(text):
        return text
    return _HAN.sub(r" \1 ", text)


def model_line(text: str) -> str:
    """Exactly what ``LangID.top`` sends to fastText: ``space_han(text)`` as one line."""
    return one_line(space_han(text))


def check_accept_labels(model_labels: Iterable[str], accept_labels: Mapping[str, Sequence[str]]) -> None:
    """Raise if an accepted label does not exist in the model (a typo would make every check fail)."""
    known = {label.removeprefix(_PREFIX) for label in model_labels}
    missing = sorted({lab for labs in accept_labels.values() for lab in labs if lab not in known})
    if missing:
        raise ValueError(f"accept_labels not in the GlotLID label set: {missing}; "
                         "fix languages.accept_labels in configs/project.yaml")


class LangID:
    def __init__(self, model_path: str | None = None, *, model: Any = None,
                 cfg: Mapping[str, Any] | None = None) -> None:
        """Load GlotLID from ``model_path``, or download the pinned revision with huggingface_hub.

        ``model`` takes an already loaded fastText model (or a test double exposing
        ``f.predict`` and ``get_labels``) instead. ``cfg`` is the project config (default: loaded).
        """
        if cfg is None:
            from lift import config

            cfg = config.load("project")
        lid_cfg = cfg["langid"]
        self.accept_labels: dict[str, list[str]] = {
            lang: list(labels) for lang, labels in cfg["languages"]["accept_labels"].items()
        }
        self.min_prob = float(lid_cfg["min_prob"])
        if model is None:
            if model_path is None:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(lid_cfg["id"], lid_cfg["file"],
                                             revision=lid_cfg.get("revision"))
            import fasttext

            model = fasttext.load_model(str(model_path))
        self._model = model
        self.labels = frozenset(label.removeprefix(_PREFIX) for label in model.get_labels())
        check_accept_labels(self.labels, self.accept_labels)

    def top(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k (label, probability), labels without the ``__label__`` prefix, best first.

        Text with no visible character gives [BLANK] without calling the model; otherwise the model
        reads ``model_line(text)`` (CJK ideographs spaced apart). See the module docstring.
        """
        if not has_visible(text):
            return [BLANK]
        preds = self._model.f.predict(model_line(text), k, 0.0, "strict")
        return [(label.removeprefix(_PREFIX), min(1.0, max(0.0, float(prob)))) for prob, label in preds]

    def is_lang(self, text: str, lang: str) -> bool:
        """True if the top label is accepted for ``lang`` and the accepted labels among the top
        ``LID_TOP_K`` (5) hold at least min_prob between them.

        Summing matters for languages GlotLID splits into varieties: genuine Modern Standard
        Arabic often comes back as arb_Arab 0.45 + ars_Arab 0.40, which is Arabic at 0.85.
        """
        return accepted(self.top(text, LID_TOP_K), self.accept_labels, lang, self.min_prob)


def accepted(preds: Sequence[tuple[str, float]], accept_labels: Mapping[str, Sequence[str]], lang: str,
             min_prob: float) -> bool:
    """The ``is_lang`` rule on a list of (label, prob) predictions, best first."""
    if lang not in accept_labels:
        raise ValueError(f"unknown language {lang!r}")
    ok = set(accept_labels[lang])
    if not preds or preds[0][0] not in ok:
        return False
    return sum(prob for label, prob in preds if label in ok) >= min_prob


def lid_info(lid: LangIdentifier, text: str) -> LidInfo:
    """The top prediction as the LidInfo stored on every PromptRecord. The probability is rounded
    to 4 places so rebuilt prompt files match byte for byte across machines."""
    preds = lid.top(text, 1)
    label, prob = preds[0] if preds else ("und", 0.0)
    return LidInfo(label=label, prob=round(float(prob), 4))
