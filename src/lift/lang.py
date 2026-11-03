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
- Short texts are less reliable (e.g. a five-character Chinese question split 0.51/0.49 between
  cmn_Hani and lzh_Hani), so a prompt that fails is listed, not silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from lift.schemas import LidInfo

_PREFIX = "__label__"


class LangIdentifier(Protocol):
    """What the pipeline needs from a language identifier; tests pass a fake with these methods."""

    def top(self, text: str, k: int = 3) -> list[tuple[str, float]]: ...

    def is_lang(self, text: str, lang: str) -> bool: ...


def one_line(text: str) -> str:
    """The text as one fastText input line: inner line breaks become spaces, plus the final newline."""
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ") + "\n"


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
        """Top-k (label, probability), labels without the ``__label__`` prefix."""
        preds = self._model.f.predict(one_line(text), k, 0.0, "strict")
        return [(label.removeprefix(_PREFIX), min(1.0, max(0.0, float(prob)))) for prob, label in preds]

    def is_lang(self, text: str, lang: str) -> bool:
        """True if the top label is accepted for ``lang`` and the accepted labels among the top 5
        hold at least min_prob between them.

        Summing matters for languages GlotLID splits into varieties: genuine Modern Standard
        Arabic often comes back as arb_Arab 0.45 + ars_Arab 0.40, which is Arabic at 0.85.
        """
        return accepted(self.top(text, 5), self.accept_labels, lang, self.min_prob)


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
