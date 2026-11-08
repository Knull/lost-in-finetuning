"""Data contracts between stages. Every JSONL file in data/ and outputs/ is one of these records.

Prompt ids (``pid``) are shared across languages, so pairing and splitting work by pid:

    prh-{split}-{i:04d}   PolyRefuse harmful, split in {test, train, val}, i = row index
    prb-{split}-{i:04d}   PolyRefuse harmless
    orb-{i:04d}           OR-Bench-Hard-1K, i = row index in the HF dataset
    art-hi-{i:04d}        Aya Red-teaming, Hindi (human-written, final set only)
    utl-{src}-{i:05d}     held-out Hindi utility instructions, src in {aya, ind}
    bel-{i:03d}           Belebele (parallel across languages)

``split`` is the role a prompt plays in this project (train / val / fit / dev / final), assigned
once per pid from data/splits/*.json, never per language.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Lang = Literal["en", "hi", "zh", "es", "ar", "th"]
PromptSet = Literal["harmful", "easy_benign", "hard_benign", "redteam", "utility"]
Split = Literal["train", "val", "fit", "dev", "final"]
Label = Literal["harmful", "refusal", "benign", "unclear"]
Method = Literal["stock", "adapter", "wiseft", "resta", "cast", "cast_always"]

LANGS: tuple[str, ...] = ("en", "hi", "zh", "es", "ar", "th")
LABELS: tuple[str, ...] = ("harmful", "refusal", "benign", "unclear")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TranslationInfo(_Record):
    engine: str                      # "google" | "nllb-200-distilled-600M" | "polyrefuse"
    source_lang: Lang = "en"
    date: str | None = None          # ISO date the translation was made
    checked: bool = False            # a native reader looked at this item
    edited: bool = False             # the reader fixed it (text differs from the engine output)
    note: str | None = None


class LidInfo(_Record):
    label: str                       # top GlotLID label, e.g. "hin_Deva"
    prob: float


class PromptRecord(_Record):
    pid: str
    lang: Lang
    set: PromptSet
    split: Split
    text: str                        # what the model sees, in ``lang``
    text_en: str | None = None       # English source when ``text`` is a translation
    source: str                      # polyrefuse | orbench-hard-1k | aya-redteaming | aya-dataset | ...
    category: str | None = None
    translation: TranslationInfo | None = None
    lid: LidInfo | None = None
    extra: dict[str, Any] = Field(default_factory=dict)   # e.g. harm_category, global_or_local


class VariantSpec(_Record):
    name: str                        # see configs/variants.yaml for the naming scheme
    method: Method
    adapter: str | None = None       # adapter name (adapter-{cand}-{recipe}-s{seed}) for non-stock
    seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)   # lambda | alpha | strength | layer ...
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> sha256 of files it loads


class GenerationRecord(_Record):
    key: str                         # lift.cachekey.generation_key(...)
    variant: str
    pid: str
    lang: Lang
    set: PromptSet
    split: Split
    prompt_sha: str                  # sha256 of the prompt text the model saw
    output: str
    n_prompt_tokens: int
    n_new_tokens: int
    hit_cap: bool                    # stopped at max_new_tokens rather than EOS
    seconds: float                   # wall time attributed to this prompt
    meta: dict[str, Any] = Field(default_factory=dict)    # e.g. {"gate_fired": true} for CAST


class UnclearFlags(_Record):
    too_short: bool
    repetitive: bool
    garbled: bool                    # top label's language mass below min_lid_prob (labels.language_mass)
    wrong_lang: bool                 # confident, but not the prompt's language (reported, not unclear)
    code_mixed: bool = False         # Hindi with a strong English share (reported separately)
    lid_top: list[tuple[str, float]] = Field(default_factory=list)


class JudgementRecord(_Record):
    key: str                         # the GenerationRecord key
    variant: str
    pid: str
    lang: Lang
    set: PromptSet
    split: Split
    judge: str                       # judge id: "Qwen/Qwen3Guard-Gen-4B@<sha>#rules-<8 hex>#bf16#strict"
    judge_label: Label               # the judge's own call, mapped to our four classes
    label: Label                     # final class after the unclear rules (lift.judge.labels)
    flags: UnclearFlags
    native: dict[str, Any] = Field(default_factory=dict)  # judge-native fields (Safety, Refusal, ...)
    raw: str = ""                    # raw judge output, for audits
    output_sha: str | None = None    # sha256 of the GenerationRecord.output judged (keys name inputs only)


class UtilityVerdict(_Record):
    pid: str
    variant: str                     # the candidate model; the reference is always stock
    judge: str
    order_ab: Literal["A", "B", "TIE"]   # candidate shown as A
    order_ba: Literal["A", "B", "TIE"]   # candidate shown as B
    score: float                     # candidate's score averaged over both orders: 1, 0.75, 0.5, 0.25, 0
    # Added in P03 (all optional, so P04's readers and older files still validate). A judge failure in one
    # order (a refusal, a reply unparseable after the reminder, an item-level API error) is recorded in
    # error_ab / error_ba and scored as a tie in that order (order_* "TIE"), so the pair stays in the
    # denominator; lift.utility reports how many orders failed beside every win rate.
    split: Split | None = None
    lang: Lang = "hi"
    candidate_key: str | None = None     # GenerationRecord.key of the candidate's answer judged
    stock_key: str | None = None         # GenerationRecord.key of stock's answer judged
    candidate_sha: str | None = None     # sha256 of the candidate's answer text
    stock_sha: str | None = None         # sha256 of stock's answer text
    reason_ab: str | None = None
    reason_ba: str | None = None
    error_ab: str | None = None
    error_ba: str | None = None


class HumanLabel(_Record):
    item_id: str                     # blind id from the validation sheet
    annotator: str
    label: Label
    note: str | None = None
