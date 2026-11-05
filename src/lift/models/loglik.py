"""Log-likelihood scoring, implemented in P03: Belebele accuracy by letter scoring, and the held-out NLL of
the utility set's reference answers. Both are forward passes only, through one batched, resumable,
cache-keyed scorer that treats every variant alike.

Belebele (decisions.md items 10 and 53). The prompt is frozen and the same for every variant, with no
chat template: lm-evaluation-harness's Belebele template (lm_eval/tasks/belebele/_default_template_yaml),
which strips the question (``{{question.strip()}}``) and inserts the passage and the four options as
published, so their few trailing spaces stay.

    P: {flores_passage}
    Q: {question}
    A: {mc_answer1}
    B: {mc_answer2}
    C: {mc_answer3}
    D: {mc_answer4}
    Answer:

- Context ids: the tokenizer's BOS exactly once (when it has one), then ``tokenizer(prompt,
  add_special_tokens=False)``. A plain ``tokenizer(prompt)`` adds a BOS itself on Cohere, and the
  template-free prompt must never carry two. A prompt holding a control-token string raises.
- Continuations " A", " B", " C", " D", tokenized with ``add_special_tokens=False``. For every item,
  tokenizing the prompt and a continuation together must give exactly the prompt's ids followed by the
  letter's (raises otherwise), so the scored tokens are the ones the text tokenizes to.
- A letter's score is the sum of its tokens' log-probabilities given the context (and its own earlier
  tokens), log-softmax in float32. When every letter is one token (Tiny Aya) that is the next-token
  log-probability after "Answer:", read from one forward pass per item that keeps only the last
  position's logits (scoring "next_token"). Otherwise (the tests' byte-level tokenizer) each context +
  letter sequence is scored (scoring "sequence"). The prediction is the argmax; a tie goes to the first
  letter and is recorded (``tie``). The gold letter is LETTERS[correct_answer_num - 1]. Accuracy per
  language counts every item of the split (200 dev / 700 final).

Held-out loss. For each utility record (``lift.prompts.load("utility", "hi", split)``), the NLL of its
reference answer (``extra["reference"]``) given its instruction, chat template and preamble on: ``ids,
n_prompt = encode_chat_turn(tokenizer, text, reference)``, and exactly ``ids[n_prompt:]`` is scored (the
answer and the tokens that close the turn), the tokens training supervises (lift.train.lora). Nothing is
truncated: a sequence longer than the model's context raises. Texts are scored as stored, not
NFC-normalised, as generation sees them.

Batching. Left padding, ``position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)`` and
``logits_to_keep`` = the last K positions, enough for every row's scored tokens, so batched scores equal
one-at-a-time scores (within 1e-4 on the tests' tiny model). Items go longest first. An out-of-memory
error halves the batch and retries it; no item is ever skipped. A plain variant is called as
``model(input_ids=, attention_mask=, position_ids=, logits_to_keep=, use_cache=False)``. A wrapper (CAST,
P05: ``LoadedVariant.base`` set) is scored through its ``score_logits``, which decides its gate per row on
that row's prompt: the Belebele context, or the chat prompt ``ids[:n_prompt]`` (what the gate reads at
generation time). Its ``batch_meta()`` row (e.g. ``gate_fired``) is recorded with each item. A wrapper
without ``score_logits`` raises NotImplementedError: a gated variant is never scored ungated.

Files (rule 4). One record per item, appended per batch, to
outputs/scores/belebele/{variant}/belebele_{lang}_{split}.jsonl or
outputs/scores/heldout_loss/{variant}/utility_hi_{split}.jsonl. The key is lift.cachekey.generation_key,
with the task configuration as ``gen_cfg`` ({"task", "version", "dtype"}, then "bos" and "letters" for
Belebele, "template" for the held-out loss) and the exact scored text as ``prompt_text`` (held-out: the
instruction, "\\n\\x00\\n", the reference). One file is one precision, one configuration and one stack,
as a generations file is (lift.models.generate): done keys are skipped. A stale file raises
StaleOutputError naming the command's ``--restart`` line:
- a record made under another configuration (``meta.run_config``);
- a pid under another key (a fixed text) or with another gold answer;
- an unfinished file begun under another stack.
That line sets the file aside under outputs/.aside/ (never synced) and leaves it EMPTY, as lift restart
does: sync never deletes, so the next push replaces the runs repo's copy. Then it scores the cell again.
A finished file is reused under any stack. A file made in another precision than the run's is not stale,
and nothing ever starts it over, --restart included, as lift restart refuses such a cell without
--force: that would discard scores the run's precision cannot make again (a T4's fp16 line would throw
away the bf16 scores Colab made, and the runs repo's copy at the next push). It raises
ScorePrecisionError (an OtherPrecisionError: no command line), whose remedies are the file's own --dtype
where that precision runs, or leaving the cell out. Score paths hold no dtype, so the precision that
scores a cell first holds its file.

Metrics are written whole and atomically (UTF-8, LF), each with a .md beside it. They cover the scores on
disk under one configuration and precision. A precision other than ``generation.dtype`` adds "-{dtype}"
to the name, as AuditKIT records do, and a run writes only its own precision's metrics, so a T4's fp16
run never replaces or retires the bf16 numbers. A score file qualifies when it holds every item of the
split as the prompt files give it now (same text, same gold answer); records of items no longer in the
split are left aside and counted (``n_outside_split``).
- outputs/metrics/belebele/{variant}_{split}.json: per language, the accuracy with n, from every
  qualifying score file of the variant and split made under the run's configuration. Files that do not
  qualify are listed under ``left_out`` with the reason.
- outputs/metrics/heldout_loss/{variant}_{split}.json: the token-weighted mean NLL (sum of NLL over sum of
  tokens), the mean of the per-row means and the perplexity, also by source (utl-aya: human-written
  references; utl-ind: IndicTrans2 machine translations). ``vs_stock`` (this variant minus stock, row
  means paired by pid, lift.stats.paired_bootstrap clustered by pid) is there when stock's scores of the
  split are complete under the same precision, model revision, template and task settings; otherwise
  ``vs_stock_note`` says why. It is computed when the variant's metrics are written: a stock cell scored
  again later needs the variant's line run again (``stock_scores_sha256`` names the file it paired with).
  Stock's scores in another precision are never paired, and the note then names the precision that
  pairs with them rather than a line that would score stock again (it would refuse stock's file).
Metrics are recomputed from the score files, one language at a time, and never retired whole for one
cell: a language whose file stops qualifying (started over, stale, made in another precision) moves to
``left_out`` with the reason, and every other language keeps its numbers. When nothing qualifies, existing
metrics files are retired (left empty, as lift restart retires AuditKIT records; ``read_metrics`` returns
None for them), so a metrics file never outlives the scores it describes.

Logs carry counts, pids and hashes, never a prompt or answer text (rule 6).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from lift import config
from lift.cachekey import generation_key
from lift.io import append_jsonl, read_jsonl, sha256_file, sha256_text
from lift.models import generate as gen
from lift.models.variants import LoadedVariant
from lift.paths import metrics_dir, output_dir, scores_dir
from lift.schemas import LANGS, PromptRecord

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
BELEBELE_FIELDS = ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4")
BELEBELE_FORMAT = ("P: {flores_passage}\nQ: {question}\nA: {mc_answer1}\nB: {mc_answer2}\nC: {mc_answer3}\n"
                   "D: {mc_answer4}\nAnswer:")
FORMAT_VERSION = 1          # the Belebele format and scoring rule above; in every Belebele key
HELDOUT_VERSION = 1         # the held-out NLL above; in every held-out key
METRICS_VERSION = 1
SEPARATOR = "\n\x00\n"      # a held-out key's scored text: instruction + SEPARATOR + reference
NEXT_TOKEN, SEQUENCE = "next_token", "sequence"
STAGES = {"belebele": "belebele", "heldout_nll": "heldout_loss"}     # task -> manifest stage, metrics dir
_SOURCE = re.compile(r"utl-(?P<source>[a-z]+)-\d+")
_CHUNK = 256                # positions per float32 log-softmax (Tiny Aya: 256 x 262,144 floats, 268 MB)
_BASE_FIELDS = ("key", "variant", "pid", "lang", "split", "task", "prompt_sha", "seconds", "meta")
_TASK_FIELDS = {
    "belebele": ("pred", "gold", "correct", "tie", "logprobs", "n_context_tokens", "n_letter_tokens"),
    "heldout_nll": ("source", "nll_sum", "n_tokens", "mean_nll", "n_prompt_tokens"),
}
# A score record's own meta fields. A wrapper's batch_meta must use other names (gate_fired, ...).
_OWN_META = frozenset({"dtype", "gpu", "device", "batch_size", "model_rev", "adapter_hash", "template",
                       "task", "task_version", "scoring", "run_config", "stack", *gen.STACK_FIELDS})
_PAIR_MUST_MATCH = ("dtype", "model_rev", "template", "task")    # held-out loss vs stock
# How a run leaves out a cell made in another precision (one Belebele line covers several cells).
_LEAVE_OUT = {"belebele": "leave it out with --langs", "heldout_nll": "leave this line out of the job"}


class ScorePrecisionError(gen.OtherPrecisionError):
    """A score file made in another precision (``dtype``: the file's) than the run's. As a generations
    file's OtherPrecisionError it is not stale and never started over, so ``command`` is None and
    --restart skips it too: that would discard scores the run's precision cannot make again (the module
    docstring). ``remedy`` is its listing line: the file's own --dtype where that precision runs, or
    leaving the cell out."""

    def __init__(self, message: str, *, dtype: str, remedy: str) -> None:
        super().__init__(message, dtype=dtype)
        self.remedy = remedy


# --- paths and command lines ----------------------------------------------------------------------------


def belebele_scores_path(variant: str, lang: str, split: str) -> Path:
    return scores_dir() / "belebele" / variant / f"belebele_{lang}_{split}.jsonl"


def heldout_scores_path(variant: str, split: str) -> Path:
    return scores_dir() / "heldout_loss" / variant / f"utility_hi_{split}.jsonl"


def metrics_path(task: str, variant: str, split: str, *, dtype: str | None = None) -> Path:
    """outputs/metrics/{belebele|heldout_loss}/{variant}_{split}.json, with "-{dtype}" for a precision other
    than generation.dtype (as AuditKIT records): a T4's fp16 run never replaces or retires the bf16
    metrics, as a run writes only its own precision's."""
    default = config.get("generation.dtype")
    suffix = "" if (dtype or default) == default else f"-{dtype}"
    return metrics_dir() / STAGES[task] / f"{variant}_{split}{suffix}.json"


def restart_line(task: str, variant: str, split: str, *, lang: str | None = None,
                 dtype: str | None = None) -> str:
    """The job line that starts one stale score cell over (``--restart``), in ``dtype`` when that is not
    generation.dtype (the line refuses another precision without it). A cell made in another precision
    has none: nothing starts it over (the module docstring)."""
    if task == "belebele":
        line = f"lift belebele --variant {variant} --split {split} --langs {lang} --restart"
    else:
        line = f"lift heldout-loss --variant {variant} --split {split} --restart"
    return line + (f" --dtype {dtype}" if dtype and dtype != config.get("generation.dtype") else "")


def _restart_for(out_path: Path, canonical: Path, line: str) -> str | None:
    """``line`` when ``out_path`` is the cell's own file (every lift command writes there), None for any
    other file (a test's): no command line covers it."""
    return line if out_path.resolve() == canonical.resolve() else None


def _start_over(restart: str | None, out_path: Path) -> str:
    if restart is None:
        return f"move {out_path} aside (no lift command line covers that file)"
    return (f"rerun with --restart (`{restart}`), once: it sets the file aside in outputs/.aside/ (never "
            "synced) and leaves it empty, so the next push replaces the runs repo's copy, then scores the "
            "cell again")


# --- Belebele: prompt, tokens, rows ---------------------------------------------------------------------


def belebele_prompt(row: Mapping[str, Any]) -> str:
    """The frozen Belebele prompt of one item (module docstring): the question stripped, as lm-eval's
    template does, every other field exactly as stored."""
    missing = [name for name in BELEBELE_FIELDS if not isinstance(row.get(name), str)]
    if missing:
        raise ValueError(f"belebele {row.get('pid')}: {missing} missing or not text")
    fields = {name: row[name] for name in BELEBELE_FIELDS}
    fields["question"] = fields["question"].strip()
    return BELEBELE_FORMAT.format(**fields)


def gold_letter(row: Mapping[str, Any]) -> str:
    """LETTERS[correct_answer_num - 1]: the stored int 1-4 (upstream's string "1"-"4" reads the same)."""
    value = row.get("correct_answer_num")
    try:
        number = int(str(value).strip())
    except ValueError:
        number = 0
    if not 1 <= number <= len(LETTERS):
        raise ValueError(f"belebele {row.get('pid')}: correct_answer_num {value!r} is not 1-4")
    return LETTERS[number - 1]


def letter_ids(tokenizer: Any) -> dict[str, list[int]]:
    """The ids of the continuations " A" .. " D" (``add_special_tokens=False``). Raises if a letter
    tokenizes to nothing or two letters to the same ids."""
    out = {x: [int(i) for i in tokenizer(" " + x, add_special_tokens=False)["input_ids"]] for x in LETTERS}
    if any(not ids for ids in out.values()) or len({tuple(ids) for ids in out.values()}) != len(LETTERS):
        raise ValueError(f"the continuations ' A' .. ' D' must tokenize to distinct, non-empty ids: {out}")
    return out


def scoring_path(letters: Mapping[str, Sequence[int]]) -> str:
    """NEXT_TOKEN when every letter is one token (one forward per item), else SEQUENCE."""
    return NEXT_TOKEN if all(len(ids) == 1 for ids in letters.values()) else SEQUENCE


def prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    """``tokenizer(prompt, add_special_tokens=False)`` of a template-free prompt. A control-token string in
    it (the BOS's included) raises: the tokenizer would read it as a real special token."""
    from lift.models.load import control_tokens

    found = sorted(t for t in control_tokens(tokenizer) if t in prompt)
    if found:
        raise ValueError(f"the prompt contains control tokens {found}; they would be read as special tokens")
    return [int(i) for i in tokenizer(prompt, add_special_tokens=False)["input_ids"]]


def _with_bos(bos: int | None, ids: list[int]) -> list[int]:
    if bos is None:
        return list(ids)
    if bos in ids:
        raise ValueError("the prompt tokenizes to a BOS token of its own; the context holds exactly one")
    return [bos, *ids]


def context_ids(tokenizer: Any, prompt: str) -> list[int]:
    """The Belebele context: the tokenizer's BOS exactly once (when it has one), then ``prompt_ids``."""
    bos = tokenizer.bos_token_id
    return _with_bos(None if bos is None else int(bos), prompt_ids(tokenizer, prompt))


def check_letter_boundary(tokenizer: Any, prompt: str, ids: Sequence[int],
                          letters: Mapping[str, Sequence[int]]) -> None:
    """Tokenizing ``prompt`` and " X" together must give ``ids`` (the prompt's, no BOS) followed by the
    letter's ids, for every letter: the scored tokens are then the ones the text tokenizes to."""
    for x in LETTERS:
        joint = [int(i) for i in tokenizer(prompt + " " + x, add_special_tokens=False)["input_ids"]]
        if joint != [*ids, *letters[x]]:
            raise ValueError(f"tokenizing the prompt and ' {x}' together does not give the prompt's ids "
                             f"followed by the letter's {list(letters[x])} (the joint tail is {joint[-4:]}): "
                             "the tokenizer merges across the boundary, so the letter cannot be scored apart")


def check_belebele_rows(rows: Sequence[Mapping[str, Any]], *, lang: str | None = None,
                        split: str | None = None) -> None:
    """Refuse rows that are not Belebele items (lift.data.build.load_prompts' dicts): none at all, a
    missing pid or language, a repeated pid, a field missing or not text, an answer outside 1-4, or
    (when given) another language or split."""
    if not rows:
        raise ValueError("no Belebele rows")
    pids = Counter(str(row.get("pid")) for row in rows)
    repeated = sorted(pid for pid, n in pids.items() if n > 1)
    if repeated:
        raise ValueError(f"belebele: repeated pids {repeated[:5]}")
    for row in rows:
        if not row.get("pid") or not isinstance(row.get("lang"), str):
            raise ValueError(f"belebele row {row.get('pid')!r} has no pid or language")
        belebele_prompt(row)
        gold_letter(row)
    wrong = [str(row["pid"]) for row in rows if (lang is not None and row["lang"] != lang)
             or (split is not None and row.get("split") != split)]
    if wrong:
        raise ValueError(f"{len(wrong)} rows are not belebele_{lang}_{split} items, e.g. {wrong[:5]}")


def belebele_config(variant: LoadedVariant) -> dict[str, Any]:
    """The Belebele task configuration, part of every key: format version, precision, BOS and letter ids."""
    bos = variant.tokenizer.bos_token_id
    return {"task": "belebele", "version": FORMAT_VERSION, "dtype": variant.dtype,
            "bos": None if bos is None else int(bos), "letters": letter_ids(variant.tokenizer)}


def belebele_run_config(variant: LoadedVariant) -> str:
    """``meta.run_config`` of the variant's Belebele files: everything in the key but the item."""
    return gen.run_config_digest(variant, belebele_config(variant))


# --- held-out loss: records ---------------------------------------------------------------------------


def source_of(pid: str) -> str:
    """The source of a utl-{source}-{index} pid: "aya" (human-written references) or "ind" (IndicTrans2
    machine translations)."""
    match = _SOURCE.fullmatch(pid)
    if match is None:
        raise ValueError(f"{pid!r} is not a utility pid (utl-{{source}}-{{index}})")
    return match["source"]


def check_references(prompts: Sequence[PromptRecord]) -> None:
    """Refuse utility records the held-out loss cannot score: none at all, another set or language than
    utility/hi, a repeated pid, a pid outside utl-{source}-{index}, or no reference answer
    (``extra["reference"]``, a non-empty text)."""
    if not prompts:
        raise ValueError("no utility records")
    wrong = [p.pid for p in prompts if p.set != "utility" or p.lang != "hi"]
    if wrong:
        raise ValueError(f"{len(wrong)} records are not utility/hi prompts, e.g. {wrong[:5]}")
    pids = Counter(p.pid for p in prompts)
    repeated = sorted(pid for pid, n in pids.items() if n > 1)
    if repeated:
        raise ValueError(f"utility: repeated pids {repeated[:5]}")
    for p in prompts:
        source_of(p.pid)
    missing = [p.pid for p in prompts
               if not isinstance(p.extra.get("reference"), str) or not p.extra["reference"].strip()]
    if missing:
        raise ValueError(f"{len(missing)} utility records have no reference answer (extra['reference']), "
                         f"e.g. {missing[:5]}; the held-out loss scores every record's reference")


def heldout_config(variant: LoadedVariant) -> dict[str, Any]:
    """The held-out task configuration, part of every key: version, precision and chat template mode."""
    return {"task": "heldout_nll", "version": HELDOUT_VERSION, "dtype": variant.dtype,
            "template": config.get("generation.template")}


def heldout_run_config(variant: LoadedVariant) -> str:
    """``meta.run_config`` of the variant's held-out files: everything in the key but the record."""
    return gen.run_config_digest(variant, heldout_config(variant))


# --- units: what a cell scores ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    """One item of a cell: its identity in the score file and the forward rows its scorer reads."""

    pid: str
    lang: str
    split: str | None
    key: str
    prompt_sha: str                       # sha256 of the exact scored text
    rows: tuple[tuple[int, ...], ...]     # each forward row's ids (Belebele "sequence": one per letter)
    prompt_len: int                       # leading tokens of every row that are the prompt a gate reads
    keep: int                             # last positions whose logits the scorer reads
    info: dict[str, Any] = field(default_factory=dict)   # the gold letter, the source
    fixed: tuple[tuple[str, Any], ...] = ()              # fields a resumed record must hold as they are

    @property
    def width(self) -> int:
        return max(len(r) for r in self.rows)

    @property
    def n_tokens(self) -> int:
        return sum(len(r) for r in self.rows)


# (fields, wrapper meta) per unit of a batch, in order.
_Scored = list[tuple[dict[str, Any], dict[str, Any]]]
_Scorer = Callable[[LoadedVariant, Sequence[_Unit]], _Scored]


def _key(variant: LoadedVariant, cfg: Mapping[str, Any], pid: str, lang: str, text: str) -> str:
    return generation_key(model_rev=variant.model_rev, adapter_hash=variant.adapter_hash, spec=variant.spec,
                          template_hash=variant.template_hash, gen_cfg=dict(cfg), pid=pid, lang=lang,
                          prompt_text=text)


def _context_limit(variant: LoadedVariant) -> int | None:
    value = getattr(getattr(variant.hf_model, "config", None), "max_position_embeddings", None)
    return int(value) if value else None


def _question_units(variant: LoadedVariant, rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any],
                    scoring: str) -> list[_Unit]:
    check_belebele_rows(rows)
    tok, letters = variant.tokenizer, cfg["letters"]
    longest = max(len(ids) for ids in letters.values())
    if scoring == NEXT_TOKEN and longest != 1:
        raise ValueError("the next-token path needs single-token letters")
    limit = _context_limit(variant)
    units, too_long = [], []
    for row in rows:
        pid, lang, text = str(row["pid"]), str(row["lang"]), belebele_prompt(row)
        try:
            ids = prompt_ids(tok, text)
            check_letter_boundary(tok, text, ids, letters)
            context = _with_bos(cfg["bos"], ids)
        except ValueError as exc:
            raise ValueError(f"belebele {pid}: {exc}") from None
        if limit and len(context) + longest > limit:
            too_long.append(pid)
        if scoring == NEXT_TOKEN:
            seqs, keep = (tuple(context),), 1
        else:
            seqs, keep = tuple(tuple(context + letters[x]) for x in LETTERS), longest + 1
        gold = gold_letter(row)
        units.append(_Unit(pid=pid, lang=lang, split=row.get("split"),
                           key=_key(variant, cfg, pid, lang, text), prompt_sha=sha256_text(text), rows=seqs,
                           prompt_len=len(context), keep=keep, info={"gold": gold}, fixed=(("gold", gold),)))
    if too_long:
        raise ValueError(f"{len(too_long)} Belebele prompts plus their letter exceed the {limit}-token "
                         f"context, e.g. {too_long[:5]}; nothing is truncated")
    return units


def _answer_units(variant: LoadedVariant, prompts: Sequence[PromptRecord],
                  cfg: Mapping[str, Any]) -> list[_Unit]:
    from lift.models.load import encode_chat_turn

    check_references(prompts)
    limit = _context_limit(variant)
    units, too_long = [], []
    for p in prompts:
        reference = p.extra["reference"]
        try:
            ids, n_prompt = encode_chat_turn(variant.tokenizer, p.text, reference)
        except ValueError as exc:
            raise ValueError(f"utility {p.pid}: {exc}") from None
        if limit and len(ids) > limit:
            too_long.append(f"{p.pid} ({len(ids)} tokens)")
        text = p.text + SEPARATOR + reference
        units.append(_Unit(pid=p.pid, lang=p.lang, split=p.split, key=_key(variant, cfg, p.pid, p.lang, text),
                           prompt_sha=sha256_text(text), rows=(tuple(ids),), prompt_len=n_prompt,
                           keep=len(ids) - n_prompt + 1, info={"source": source_of(p.pid)}))
    if too_long:
        raise ValueError(f"{len(too_long)} utility instructions with their reference exceed the "
                         f"{limit}-token context, e.g. {too_long[:5]}; nothing is truncated")
    return units


# --- the forward pass ---------------------------------------------------------------------------------


def check_scorable(variant: LoadedVariant) -> None:
    """Refuse a wrapper (``base`` set) without ``score_logits``: scoring the wrapped model directly would
    report a gated variant ungated."""
    if variant.score_logits is None and variant.base is not None:
        raise NotImplementedError(
            f"{variant.spec.name} wraps its model (LoadedVariant.base is set) but has no score_logits: "
            "log-likelihood scoring of a wrapper (CAST) needs its gate decided per row on that row's prompt, "
            "which P05's builder provides as LoadedVariant.score_logits; the wrapped model scored directly "
            "would report the variant ungated")


def _wrapper_meta(variant: LoadedVariant, n: int) -> list[dict[str, Any]]:
    """The wrapper's per-row meta of its last score_logits call ({} per row without batch_meta)."""
    if variant.batch_meta is None:
        return [{} for _ in range(n)]
    metas = variant.batch_meta()
    if not isinstance(metas, list) or len(metas) != n or not all(isinstance(m, dict) for m in metas):
        raise ValueError(f"{variant.spec.name}: batch_meta() must return one dict per row of the last "
                         f"score_logits call ({n} rows)")
    clash = sorted({k for m in metas for k in m} & _OWN_META)
    if clash:
        raise ValueError(f"{variant.spec.name}: batch_meta() sets {clash}, which a score record's meta sets "
                         "itself; give the wrapper's fields their own names (e.g. gate_fired)")
    return [dict(m) for m in metas]


def _forward(variant: LoadedVariant, seqs: Sequence[Sequence[int]], prompt_lengths: Sequence[int],
             keep: int) -> tuple[Any, list[dict[str, Any]], Any]:
    """(logits of the last ``keep`` positions [rows, keep, vocab], the wrapper's per-row meta, the padded
    input ids) of left-padded ``seqs``, with position ids counted from each row's first real token."""
    import torch

    check_scorable(variant)
    pad = variant.tokenizer.pad_token_id
    if pad is None:
        raise ValueError("the tokenizer has no pad token; left-padded batches need one")
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), int(pad), dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, width - len(s):] = torch.tensor(list(s), dtype=torch.long)
        mask[i, width - len(s):] = 1
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    input_ids, mask, position_ids = (t.to(variant.device) for t in (input_ids, mask, position_ids))
    with torch.inference_mode():
        if variant.score_logits is not None:
            logits = variant.score_logits(input_ids, mask, position_ids=position_ids,
                                          prompt_lengths=[int(n) for n in prompt_lengths],
                                          logits_to_keep=keep)
            metas = _wrapper_meta(variant, len(seqs))
        else:
            logits = variant.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                                   logits_to_keep=keep, use_cache=False).logits
            metas = [{} for _ in seqs]
    if logits.dim() != 3 or tuple(logits.shape[:2]) != (len(seqs), keep):
        raise ValueError(f"the model returned logits of shape {tuple(logits.shape)}; logits_to_keep={keep} "
                         f"over {len(seqs)} rows must give ({len(seqs)}, {keep}, vocab)")
    return logits, metas, input_ids


def _target_logprobs(logits: Any, input_ids: Any, n_targets: Sequence[int]) -> list[list[float]]:
    """Per row, the float32 log-probabilities of its last ``n_targets[i]`` tokens, each given every token
    before it; ``logits`` keeps the last positions of the left-padded ``input_ids``."""
    import torch

    keep, width = int(logits.shape[1]), int(input_ids.shape[1])
    out = []
    for i, t in enumerate(n_targets):
        if not 1 <= t < keep:
            raise ValueError(f"row {i}: {t} scored tokens need more than {t} kept positions, not {keep}")
        rows, targets = logits[i, keep - 1 - t: keep - 1], input_ids[i, width - t:]
        parts = []
        for s in range(0, t, _CHUNK):
            scores = torch.log_softmax(rows[s: s + _CHUNK].float(), dim=-1)
            parts.append(scores.gather(-1, targets[s: s + _CHUNK, None]).squeeze(-1))
        out.append(torch.cat(parts).tolist())
    return out


def _choice(unit: _Unit, logprobs: Mapping[str, float], n_letters: Mapping[str, int]) -> dict[str, Any]:
    bad = [x for x in LETTERS if not math.isfinite(logprobs[x])]
    if bad:
        raise ValueError(f"belebele {unit.pid}: non-finite log-probabilities for {bad} (a precision "
                         "overflow?); nothing is recorded for a broken forward pass")
    best = max(logprobs[x] for x in LETTERS)
    top = [x for x in LETTERS if logprobs[x] == best]
    gold = unit.info["gold"]
    return {"pred": top[0], "gold": gold, "correct": top[0] == gold, "tie": len(top) > 1,
            "logprobs": {x: float(logprobs[x]) for x in LETTERS}, "n_context_tokens": unit.prompt_len,
            "n_letter_tokens": dict(n_letters)}


def _belebele_scorer(cfg: Mapping[str, Any], scoring: str) -> _Scorer:
    """The scorer of one batch of Belebele items on the ``scoring`` path."""
    letters = cfg["letters"]
    n_letters = {x: len(letters[x]) for x in LETTERS}
    per_item = len(LETTERS)

    def next_token(variant: LoadedVariant, units: Sequence[_Unit]) -> tuple[list[dict], list[dict]]:
        import torch

        logits, metas, _ids = _forward(variant, [u.rows[0] for u in units], [u.prompt_len for u in units], 1)
        index = torch.tensor([letters[x][0] for x in LETTERS], device=logits.device)
        table = torch.log_softmax(logits[:, -1, :].float(), dim=-1)[:, index].tolist()
        return [dict(zip(LETTERS, row, strict=True)) for row in table], metas

    def sequence(variant: LoadedVariant, units: Sequence[_Unit]) -> tuple[list[dict], list[dict]]:
        n = [n_letters[x] for _u in units for x in LETTERS]
        logits, metas, input_ids = _forward(variant, [r for u in units for r in u.rows],
                                            [u.prompt_len for u in units for _x in LETTERS], max(n) + 1)
        per_row = _target_logprobs(logits, input_ids, n)
        scores, item_metas = [], []
        for i, u in enumerate(units):
            block = per_row[per_item * i: per_item * (i + 1)]
            scores.append({x: math.fsum(lps) for x, lps in zip(LETTERS, block, strict=True)})
            rows_meta = metas[per_item * i: per_item * (i + 1)]
            if any(m != rows_meta[0] for m in rows_meta):
                raise ValueError(f"belebele {u.pid}: the wrapper's batch_meta differs between the item's "
                                 "four rows, which share the prompt its gate reads")
            item_metas.append(rows_meta[0])
        return scores, item_metas

    def score(variant: LoadedVariant, units: Sequence[_Unit]) -> _Scored:
        scores, metas = (next_token if scoring == NEXT_TOKEN else sequence)(variant, units)
        return [(_choice(u, s, n_letters), m) for u, s, m in zip(units, scores, metas, strict=True)]

    return score


def _heldout_scorer(variant: LoadedVariant, units: Sequence[_Unit]) -> _Scored:
    """The scorer of one batch of utility references."""
    logits, metas, input_ids = _forward(variant, [u.rows[0] for u in units], [u.prompt_len for u in units],
                                        max(u.keep for u in units))
    per_row = _target_logprobs(logits, input_ids, [u.keep - 1 for u in units])
    out = []
    for u, lps, meta in zip(units, per_row, metas, strict=True):
        nll = -math.fsum(lps)
        if not math.isfinite(nll):
            raise ValueError(f"utility {u.pid}: a non-finite NLL (a precision overflow?); nothing is "
                             "recorded for a broken forward pass")
        out.append(({"source": u.info["source"], "nll_sum": nll, "n_tokens": len(lps),
                     "mean_nll": nll / len(lps), "n_prompt_tokens": u.prompt_len}, meta))
    return out


# --- batches ----------------------------------------------------------------------------------------------


@dataclass
class _Progress:
    size: int
    oom_retries: int = 0
    batch_sizes: set[int] = field(default_factory=set)
    n_written: int = 0
    n_tokens: int = 0
    seconds: float = 0.0


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _auto_batch_size(variant: LoadedVariant, units: Sequence[_Unit], *, cap: int = 64) -> int:
    """Items per forward pass from free memory: one layer's activations and attention scores plus the kept
    logits of every row. Deliberately rough, as generate's: an out-of-memory error still halves the batch."""
    import torch

    rows = max(len(u.rows) for u in units)
    if not str(variant.device).startswith("cuda") or not torch.cuda.is_available():
        return max(1, 8 // rows)
    cfg = variant.hf_model.config
    width, keep = max(u.width for u in units), max(u.keep for u in units)
    elem = 4 if variant.dtype == "fp32" else 2
    per_row = 8 * int(cfg.hidden_size) * width * elem                     # one layer's activations
    per_row += int(cfg.num_attention_heads) * width * width * 4           # attention scores, worst case
    per_row += keep * int(cfg.vocab_size) * elem                          # the kept logits
    free, _total = torch.cuda.mem_get_info(torch.device(variant.device))
    return int(max(1, min(cap, (0.6 * free) // (per_row * rows))))


def _score_batches(variant: LoadedVariant, units: Sequence[_Unit], score: _Scorer, progress: _Progress,
                   emit: Callable[[list[_Unit], _Scored, float], None]) -> None:
    """Score ``units`` in their order, ``progress.size`` at a time, handing each scored batch to ``emit``.
    An out-of-memory error halves the batch and retries it (never skips an item); any other error raises."""

    def run(batch: list[_Unit]) -> None:
        oom = False
        try:
            t0 = time.perf_counter()
            results = score(variant, batch)
            seconds = time.perf_counter() - t0
        except Exception as exc:  # only an out-of-memory error is retried; anything else re-raises
            if not gen.is_oom(exc) or len(batch) == 1:
                raise
            oom = True
        if oom:
            # Retry outside the except clause: its traceback would keep the failed batch's tensors alive.
            _free_memory()
            progress.oom_retries += 1
            half = len(batch) // 2
            progress.size = max(1, min(progress.size, half))
            log.warning("out of memory at batch size %d; retrying as %d + %d", len(batch), half,
                        len(batch) - half)
            run(batch[:half])
            run(batch[half:])
            return
        emit(batch, results, seconds)
        progress.batch_sizes.add(len(batch))
        progress.n_written += len(batch)
        progress.n_tokens += sum(u.n_tokens for u in batch)
        progress.seconds += seconds

    i = 0
    while i < len(units):
        batch = list(units[i: i + progress.size])
        run(batch)
        i += len(batch)


def _longest_first(units: Sequence[_Unit]) -> list[_Unit]:
    return sorted(units, key=lambda u: (-u.width, u.pid))


def _score_all(variant: LoadedVariant, units: Sequence[_Unit], score: _Scorer,
               batch_size: int | None) -> _Scored:
    """Every unit's (fields, meta), in the order of ``units``, with no file."""
    if not units:
        return []
    order = _longest_first(units)
    found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def emit(batch: list[_Unit], results: _Scored, _seconds: float) -> None:
        found.update((u.pid, r) for u, r in zip(batch, results, strict=True))

    size = batch_size or _auto_batch_size(variant, order)
    _score_batches(variant, order, score, _Progress(size=size), emit)
    return [found[u.pid] for u in units]


# --- the pure scorers -------------------------------------------------------------------------------------


def score_belebele(variant: LoadedVariant, rows: Sequence[dict], *,
                   batch_size: int | None = None) -> list[dict]:
    """Per item, in the order of ``rows``: pid, pred (the predicted letter), gold (the correct letter),
    correct, tie, logprobs {A, B, C, D}, n_context_tokens, n_letter_tokens and meta (the wrapper's
    batch_meta row; {} for a plain model). No file is read or written: ``belebele_cell`` keeps the
    resumable one."""
    check_scorable(variant)
    cfg = belebele_config(variant)
    scoring = scoring_path(cfg["letters"])
    units = _question_units(variant, rows, cfg, scoring)
    scored = _score_all(variant, units, _belebele_scorer(cfg, scoring), batch_size)
    return [{"pid": u.pid, **item, "meta": meta} for u, (item, meta) in zip(units, scored, strict=True)]


def heldout_nll(variant: LoadedVariant, prompts: Sequence[PromptRecord], *,
                batch_size: int | None = None) -> list[dict]:
    """Per utility record, in the order of ``prompts``: pid, source, nll_sum (nats), n_tokens (the answer
    tokens ``ids[n_prompt:]``), mean_nll, n_prompt_tokens and meta (as score_belebele). No file is read or
    written: ``heldout_cell`` keeps the resumable one."""
    check_scorable(variant)
    units = _answer_units(variant, prompts, heldout_config(variant))
    scored = _score_all(variant, units, _heldout_scorer, batch_size)
    return [{"pid": u.pid, **item, "meta": meta} for u, (item, meta) in zip(units, scored, strict=True)]


# --- score files ------------------------------------------------------------------------------------------


def read_scores(path: Path, task: str) -> list[dict]:
    """Every record of one score file, checked: the fields of ``task``, and one record per key and per pid
    (a repeat means a corrupt file). An empty file (what --restart leaves) holds none."""
    required = (*_BASE_FIELDS, *_TASK_FIELDS[task])
    records = list(read_jsonl(path))
    keys: set[str] = set()
    pids: set[str] = set()
    for n, r in enumerate(records, 1):
        missing = [name for name in required if name not in r]
        if missing or r.get("task") != task or not isinstance(r.get("meta"), dict):
            raise ValueError(f"{path}: record {n} is not a {task} score record (missing {missing})")
        if r["key"] in keys or r["pid"] in pids:
            raise ValueError(f"{path}: duplicate record for {r['pid']} (key {str(r['key'])[:12]})")
        keys.add(r["key"])
        pids.add(r["pid"])
    return records


def _refuse_other(existing: Sequence[Mapping[str, Any]], *, task: str, name: str, dtype: str,
                  run_config: str, out_path: Path, restart: str | None) -> None:
    """One file is one precision and one configuration: refuse to extend a file made otherwise. The
    precision comes first and names no command line, so --restart never discards another precision's
    scores, whatever else differs (the configuration always does: the task config holds the dtype)."""
    made = sorted({str(r["meta"]["dtype"]) for r in existing if r["meta"].get("dtype")} - {dtype})
    if made:
        was = "/".join(made)
        leave = _LEAVE_OUT[task]
        raise ScorePrecisionError(
            f"{out_path} holds scores made in {was}, and this run is {dtype}: one file is one precision, so "
            f"nothing is appended. Score this cell with --dtype {was} where {was} runs, or {leave}. It is "
            f"never started over, --restart or not: that would discard its {was} scores, which a run in "
            f"{dtype} cannot make again.",
            dtype=was, remedy=f"made in {was}: score it with --dtype {was} where {was} runs, or {leave}")
    other = sorted({str(r["pid"]) for r in existing
                    if r["variant"] != name or r["meta"].get("run_config") != run_config})
    if other:
        raise gen.StaleOutputError(
            f"{out_path} holds scores made under another configuration (variant, model revision, adapter, "
            f"template or scoring settings), e.g. {other[:5]}; one file is one configuration and scores are "
            f"never mixed. Run with the file's configuration, or start the cell over: "
            f"{_start_over(restart, out_path)}.", command=restart)


def _stack_moved(metas: Sequence[Mapping[str, Any]], env: Mapping[str, str]) -> str:
    """How the stack of the records (their meta) differs from ``env``, "" when every record has it."""
    current = gen.stack_digest(env)
    if all(gen.record_stack(m) == current for m in metas):
        return ""

    def was(name: str) -> list[str]:
        return sorted({str(m.get(name)) for m in metas})

    moved = [f"{name} {'/'.join(was(name))} -> {env.get(name)}" for name in gen.STACK_FIELDS
             if was(name) != [str(env.get(name))]]
    return ", ".join(moved) or f"stack {sorted({gen.record_stack(m) for m in metas})} -> {current}"


def _run_cell(variant: LoadedVariant, *, cfg: Mapping[str, Any], units: Sequence[_Unit], out_path: Path,
              restart: str | None, score: _Scorer, batch_size: int | None, manifest: bool,
              cell: Mapping[str, Any]) -> Path:
    """Score the units not yet in ``out_path`` and append them (the module docstring's file rules).
    ``cell`` (lang, split) goes into the manifest."""
    from lift.manifest import finish_run, start_run
    from lift.models.load import device_name

    task, name = cfg["task"], variant.spec.name
    run_config = gen.run_config_digest(variant, dict(cfg))
    env = gen.stack(variant.device)
    existing: list[dict] = []
    if out_path.exists():
        gen.repair_tail(out_path)
        existing = read_scores(out_path, task)
    _refuse_other(existing, task=task, name=name, dtype=variant.dtype, run_config=run_config,
                  out_path=out_path, restart=restart)
    done = {str(r["pid"]): r for r in existing}
    stale = [u.pid for u in units if u.pid in done
             and (done[u.pid]["key"] != u.key or any(done[u.pid].get(f) != v for f, v in u.fixed))]
    if stale:
        raise gen.StaleOutputError(
            f"{out_path}: {len(stale)} items were scored from other inputs (a fixed text or answer, or "
            f"another cache-key version), e.g. {stale[:5]}; scores are never mixed. Start the cell over: "
            f"{_start_over(restart, out_path)}.", command=restart)
    todo = _longest_first([u for u in units if u.pid not in done])
    moved = _stack_moved([r["meta"] for r in existing], env)
    if moved and todo:
        raise gen.StaleOutputError(
            f"{out_path} was scored under another stack ({moved}); appending {len(todo)} records under this "
            "one would mix stacks in one file. Only the stack that began it can finish it (constraints.txt "
            "pins transformers; torch is the runtime's own CUDA build). Otherwise start the cell over: "
            f"{_start_over(restart, out_path)}. Finished cells are reused under any stack.",
            command=restart, keys_changed=False)

    gpu = device_name(variant.device)
    provenance = {"model_rev": variant.model_rev, "adapter_hash": variant.adapter_hash,
                  "template": variant.template_hash[:16], **env, "stack": gen.stack_digest(env),
                  "task": gen.decoding_digest(dict(cfg)), "task_version": cfg["version"],
                  "run_config": run_config}
    if task == "belebele":
        provenance["scoring"] = scoring_path(cfg["letters"])
    run = start_run(STAGES[task], variant=name, out_path=str(out_path), **cell, dtype=variant.dtype,
                    device=variant.device, gpu=gpu, task_cfg=dict(cfg), n_items=len(units),
                    n_resumed=len(units) - len(todo), template_hash=variant.template_hash,
                    **provenance) if manifest else None
    if not todo:
        if moved:
            log.warning("%s: all %d items already scored, under another stack (%s); reused as they are",
                        out_path.name, len(units), moved)
        else:
            log.info("%s: all %d items already scored", out_path.name, len(units))
        if run is not None:
            finish_run(run, status="ok", n_written=0, devices=1)
        return out_path

    progress = _Progress(size=batch_size or _auto_batch_size(variant, todo))
    log.info("%s: scoring %d items (%d resumed), batch size %d, %s on %s", out_path.name, len(todo),
             len(units) - len(todo), progress.size, variant.dtype, gpu)

    def emit(batch: list[_Unit], results: _Scored, seconds: float) -> None:
        records = [{"key": u.key, "variant": name, "pid": u.pid, "lang": u.lang, "split": u.split,
                    "task": task, "prompt_sha": u.prompt_sha, **item,
                    "seconds": round(seconds / len(batch), 4),
                    "meta": {**more, "dtype": variant.dtype, "gpu": gpu, "device": variant.device,
                             "batch_size": len(batch), **provenance}}
                   for u, (item, more) in zip(batch, results, strict=True)]
        append_jsonl(out_path, records)
        log.info("%s: %d/%d", out_path.name, progress.n_written + len(batch), len(todo))

    def summary() -> dict[str, Any]:
        secs = progress.seconds or 1e-9
        return {"n_written": progress.n_written, "oom_retries": progress.oom_retries,
                "batch_sizes": sorted(progress.batch_sizes), "score_seconds": round(progress.seconds, 3),
                "items_per_sec": round(progress.n_written / secs, 4),
                "tokens_per_sec": round(progress.n_tokens / secs, 2), "n_tokens": progress.n_tokens,
                "devices": 1}

    try:
        _score_batches(variant, todo, score, progress, emit)
    except BaseException as exc:
        # A crashed run still records the GPU time it spent (the cost section reads manifests).
        if run is not None:
            finish_run(run, status="failed", error=type(exc).__name__, **summary())
        raise
    stats = summary()
    log.info("%s: wrote %d records, %.2f items/s, %.1f tokens/s", out_path.name, stats["n_written"],
             stats["items_per_sec"], stats["tokens_per_sec"])
    if run is not None:
        finish_run(run, status="ok", **stats)
    return out_path


def belebele_cell(variant: LoadedVariant, rows: Sequence[dict], *, lang: str, split: str,
                  out_path: Path | None = None, batch_size: int | None = None, manifest: bool = True) -> Path:
    """Score one language's Belebele split into its resumable file (default the cell's own,
    ``belebele_scores_path``) and return the path. Refusals are StaleOutputErrors (module docstring)."""
    check_scorable(variant)
    check_belebele_rows(rows, lang=lang, split=split)
    cfg = belebele_config(variant)
    scoring = scoring_path(cfg["letters"])
    canonical = belebele_scores_path(variant.spec.name, lang, split)
    out_path = Path(out_path) if out_path is not None else canonical
    line = restart_line("belebele", variant.spec.name, split, lang=lang, dtype=variant.dtype)
    return _run_cell(variant, cfg=cfg, units=_question_units(variant, rows, cfg, scoring), out_path=out_path,
                     restart=_restart_for(out_path, canonical, line), score=_belebele_scorer(cfg, scoring),
                     batch_size=batch_size, manifest=manifest, cell={"lang": lang, "split": split})


def heldout_cell(variant: LoadedVariant, prompts: Sequence[PromptRecord], *, split: str,
                 out_path: Path | None = None, batch_size: int | None = None, manifest: bool = True) -> Path:
    """Score the utility split's references into the resumable held-out file (default the cell's own,
    ``heldout_scores_path``) and return the path. Refusals are StaleOutputErrors (module docstring)."""
    check_scorable(variant)
    check_references(prompts)
    wrong = [p.pid for p in prompts if p.split != split]
    if wrong:
        raise ValueError(f"{len(wrong)} utility records are not in the {split} split, e.g. {wrong[:5]}")
    cfg = heldout_config(variant)
    canonical = heldout_scores_path(variant.spec.name, split)
    out_path = Path(out_path) if out_path is not None else canonical
    line = restart_line("heldout_nll", variant.spec.name, split, dtype=variant.dtype)
    return _run_cell(variant, cfg=cfg, units=_answer_units(variant, prompts, cfg), out_path=out_path,
                     restart=_restart_for(out_path, canonical, line), score=_heldout_scorer,
                     batch_size=batch_size, manifest=manifest, cell={"lang": "hi", "split": split})


def start_over(path: Path) -> gen.SetAside:
    """--restart for one stale score file: copy it to the same path under a new outputs/.aside/<UTC stamp>
    folder (never synced) and leave it empty. Emptied, never removed: sync never deletes, so a removed file
    would come back from the runs repo at the next pull, while an emptied one is a local edit the next push
    sends. The copy is written before the file is emptied, so a kill in between loses nothing. Only a
    refusal that names a --restart line leads here, which a file made in another precision never gets."""
    path = Path(path)
    if not path.is_file():
        return gen.SetAside(path, None, None)
    data = path.read_bytes()
    if not data:
        return gen.SetAside(path, 0, None)
    try:
        relative = path.resolve().relative_to(output_dir())
    except ValueError:
        raise ValueError(f"{path} is not under {output_dir()}; only an outputs/ file is started "
                         "over") from None
    copy = gen.aside_root() / relative
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(data)
    path.write_bytes(b"")
    return gen.SetAside(path, sum(1 for line in data.splitlines() if line.strip()), copy)


# --- metrics ----------------------------------------------------------------------------------------------


class Written(NamedTuple):
    """A metrics document and the files it went to."""

    doc: dict[str, Any]
    json_path: Path
    md_path: Path


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _shown(path: Path) -> str:
    """``path`` as the runs repo names it (outputs/...), the same on every machine."""
    try:
        return "outputs/" + path.resolve().relative_to(output_dir()).as_posix()
    except ValueError:
        return str(path)


def _write_text(path: Path, text: str) -> None:
    """UTF-8, LF and atomic (temp file + rename), like lift.io.write_jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_metrics(path: Path) -> dict[str, Any] | None:
    """A metrics document, or None for a retired one (an empty file: no score file qualified any more)."""
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else None


def _retire(path: Path) -> list[Path]:
    """Leave a metrics JSON and its .md empty (never removed: sync never deletes); the ones retired."""
    done = []
    for p in (path, path.with_suffix(".md")):
        if p.is_file() and p.stat().st_size:
            _write_text(p, "")
            done.append(p)
    return done


def _write_metrics(path: Path, doc: dict[str, Any] | None,
                   render: Callable[[Mapping[str, Any]], str]) -> Written | None:
    """Write ``doc`` and its markdown, or retire both (left empty, never removed) when it is None."""
    if doc is None:
        for p in _retire(path):
            log.info("%s: retired (left empty): no score file qualifies for it any more", p)
        return None
    md_path = path.with_suffix(".md")
    _write_text(path, json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
    _write_text(md_path, render(doc))
    return Written(doc, path, md_path)


def _readable(path: Path, task: str) -> tuple[list[dict], str | None]:
    """(records, None), or (records, why the file cannot be reported) for a missing, empty or corrupt one."""
    if not path.is_file():
        return [], "no score file"
    try:
        records = read_scores(path, task)
    except ValueError as exc:
        return [], f"unreadable ({exc})"
    return records, None if records else "an empty score file (started over)"


def _not_reportable(records: Sequence[Mapping[str, Any]], *, dtype: str,
                    run_config: str | None) -> str | None:
    configs = {str(r["meta"].get("run_config")) for r in records}
    dtypes = {str(r["meta"].get("dtype")) for r in records}
    if len(configs) != 1:
        return f"holds records of {len(configs)} configurations"
    if dtypes != {dtype}:
        return f"made in {'/'.join(sorted(dtypes))}, not {dtype}"
    if run_config is not None and configs != {run_config}:
        return "made under another configuration than this run's"
    return None


_Current = tuple[list[dict], int, str | None]


def _current(records: Sequence[Mapping[str, Any]], expected: Mapping[str, Any],
             identity: Callable[[Mapping[str, Any]], Any]) -> _Current:
    """(the records of exactly the split's items as they are now, in pid order; how many records are of
    items no longer in the split, which are left aside as lift.commands.run._complete leaves them; None),
    or ([], 0, why the file cannot be reported: an item missing, or one scored from another text).
    ``expected`` maps each pid of the split to the identity the current data gives it."""
    got = {str(r["pid"]): r for r in records}
    missing = expected.keys() - got.keys()
    if missing:
        return [], 0, f"incomplete: {len(expected) - len(missing)} of {len(expected)} items scored"
    changed = sorted(pid for pid, ident in expected.items() if identity(got[pid]) != ident)
    if changed:
        return [], 0, (f"{len(changed)} items were scored from other texts (a data fix), e.g. {changed[:3]}: "
                       "start it over with --restart")
    return [dict(got[pid]) for pid in sorted(expected)], len(got) - len(expected), None


def _gate_fired_rate(records: Sequence[Mapping[str, Any]]) -> float | None:
    """The share of items whose wrapper's gate fired (``meta.gate_fired``); None for a plain model."""
    values = [r["meta"].get("gate_fired") for r in records]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("gate_fired is recorded for some items and not for others")
    return sum(bool(v) for v in values) / len(values)


def _belebele_lang(path: Path, items: Sequence[Mapping[str, Any]], outside: int) -> dict[str, Any]:
    n = len(items)
    n_correct = sum(r["pred"] == r["gold"] for r in items)
    preds = Counter(str(r["pred"]) for r in items)
    return {"n": n, "n_correct": n_correct, "accuracy": n_correct / n,
            "n_ties": sum(bool(r["tie"]) for r in items),
            "pred_counts": {x: preds.get(x, 0) for x in LETTERS}, "gate_fired_rate": _gate_fired_rate(items),
            "n_outside_split": outside, "scores_path": _shown(path), "scores_sha256": sha256_file(path)}


def _belebele_items(records: Sequence[Mapping[str, Any]], lang: str, split: str, *, dtype: str,
                    run_config: str | None) -> _Current:
    """``_current`` for one language's Belebele score file, checked against its prompt file."""
    from lift.data import build

    why = _not_reportable(records, dtype=dtype, run_config=run_config)
    if why:
        return [], 0, why
    try:
        rows = build.load_prompts("belebele", lang, split)
    except FileNotFoundError:
        return [], 0, "no prompt file to check it against"
    expected = {str(row["pid"]): (sha256_text(belebele_prompt(row)), gold_letter(row)) for row in rows}
    return _current(records, expected, lambda r: (r["prompt_sha"], r["gold"]))


def belebele_metrics(variant: str, split: str, *, dtype: str | None = None, run_config: str | None = None,
                     langs: Sequence[str] = LANGS) -> dict[str, Any] | None:
    """The Belebele metrics of a variant and split (the module docstring), or None when no language's score
    file covers its split under ``run_config`` (the run's; None: any one configuration) in ``dtype``
    (default generation.dtype)."""
    dtype = dtype or config.get("generation.dtype")
    cells: dict[str, tuple[Path, list[dict], int]] = {}
    left_out: dict[str, str] = {}
    for lang in langs:
        path = belebele_scores_path(variant, lang, split)
        if not path.is_file():
            continue
        records, why = _readable(path, "belebele")
        items, outside = [], 0
        if why is None:
            items, outside, why = _belebele_items(records, lang, split, dtype=dtype, run_config=run_config)
        if why:
            left_out[lang] = why
            continue
        cells[lang] = (path, items, outside)
    if not cells:
        return None
    configs = {items[0]["meta"]["run_config"] for _path, items, _outside in cells.values()}
    if len(configs) > 1:
        raise ValueError(f"{variant} {split}: the Belebele score files of {sorted(cells)} were made under "
                         f"{len(configs)} configurations; pass the run's run_config to pick one")
    meta = next(iter(cells.values()))[1][0]["meta"]
    return {"kind": "belebele", "version": METRICS_VERSION, "variant": variant, "split": split,
            "format_version": meta.get("task_version"), "scoring": meta.get("scoring"), "dtype": dtype,
            "model_rev": meta.get("model_rev"), "adapter_hash": meta.get("adapter_hash"),
            "template": meta.get("template"), "run_config": meta.get("run_config"),
            "langs": {lang: _belebele_lang(*cell) for lang, cell in cells.items()},
            "left_out": left_out, "created": _now()}


def write_belebele_metrics(variant: str, split: str, *, dtype: str | None = None,
                           run_config: str | None = None) -> Written | None:
    """Write (or, with nothing qualifying, retire) outputs/metrics/belebele/{variant}_{split}[-dtype].json
    and its .md."""
    doc = belebele_metrics(variant, split, dtype=dtype, run_config=run_config)
    return _write_metrics(metrics_path("belebele", variant, split, dtype=dtype), doc, _belebele_md)


def _heldout_items(path: Path, prompts: Sequence[PromptRecord], *, dtype: str,
                   run_config: str | None) -> _Current:
    """``_current`` for a held-out score file against the split's references as they are now, in ``dtype``
    (and under ``run_config`` when given)."""
    records, why = _readable(path, "heldout_nll")
    why = why or _not_reportable(records, dtype=dtype, run_config=run_config)
    if why:
        return [], 0, why
    expected = {p.pid: sha256_text(p.text + SEPARATOR + p.extra["reference"]) for p in prompts}
    return _current(records, expected, lambda r: r["prompt_sha"])


def _nll(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Token-weighted mean NLL (sum of NLL over sum of tokens), the mean of the rows' means, perplexity."""
    tokens = sum(int(r["n_tokens"]) for r in records)
    mean = math.fsum(float(r["nll_sum"]) for r in records) / tokens
    return {"n_rows": len(records), "n_tokens": tokens, "mean_nll": mean,
            "row_mean_nll": math.fsum(float(r["mean_nll"]) for r in records) / len(records),
            "perplexity": math.exp(mean)}


def _paired(mine: Mapping[str, Mapping[str, Any]], stock: Mapping[str, Mapping[str, Any]],
            pids: Sequence[str]) -> dict[str, Any]:
    """This variant minus stock over ``pids``: the row means paired by pid (cluster bootstrap by pid), and
    the token-weighted means' difference."""
    from lift.stats import paired_bootstrap

    x = [float(mine[p]["mean_nll"]) for p in pids]
    y = [float(stock[p]["mean_nll"]) for p in pids]
    res = paired_bootstrap(x, y, clusters=list(pids))
    pooled = _nll([mine[p] for p in pids])["mean_nll"] - _nll([stock[p] for p in pids])["mean_nll"]
    return {"diff_row_mean_nll": res.diff, "lo": res.lo, "hi": res.hi, "p_value": res.p_value,
            "n": res.n_rows, "n_boot": res.n_boot, "diff_mean_nll": pooled}


def _vs_stock(variant: str, split: str, records: Sequence[Mapping[str, Any]],
              prompts: Sequence[PromptRecord], *, dtype: str) -> tuple[dict[str, Any] | None, str | None]:
    """(the paired comparison with stock, None), or (None, why there is none). The note names stock's line
    (in ``dtype``) only when that line could make the scores. Stock's file made in another precision is
    never paired and never started over from a ``dtype`` run, so then it names the precision that pairs
    with it instead: the line would only refuse the file."""
    if variant == "stock":
        return None, "this is stock"
    path = heldout_scores_path("stock", split)
    held, _why = _readable(path, "heldout_nll")
    made = sorted({str(r["meta"]["dtype"]) for r in held if r["meta"].get("dtype")} - {dtype})
    if made:
        was = "/".join(made)
        return None, (f"stock's held-out scores of {split} were made in {was}, not {dtype}: scores are never "
                      f"paired across precisions, and stock's file is never started over from a run in "
                      f"{dtype} (that would discard its {was} scores). Score this variant in {was}, where "
                      f"{was} runs, to pair it with them")
    stock, _outside, why = _heldout_items(path, prompts, dtype=dtype, run_config=None)
    if why:
        line = f"lift heldout-loss --variant stock --split {split}"
        if dtype != config.get("generation.dtype"):
            line += f" --dtype {dtype}"
        return None, (f"stock's held-out scores of {split} cannot be paired ({why}); run `{line}`, then this "
                      "variant's line again")
    ours, theirs = records[0]["meta"], stock[0]["meta"]
    differ = {k: [ours.get(k), theirs.get(k)] for k in _PAIR_MUST_MATCH if ours.get(k) != theirs.get(k)}
    if differ:
        return None, f"stock's held-out scores were made with other settings {differ}; they are never paired"
    mine, base = {str(r["pid"]): r for r in records}, {str(r["pid"]): r for r in stock}
    lengths = sorted(p for p in mine if mine[p]["n_tokens"] != base[p]["n_tokens"])
    if lengths:
        return None, f"{len(lengths)} references have other token counts in stock's file, e.g. {lengths[:3]}"
    out = _paired(mine, base, sorted(mine))
    out["by_source"] = {src: _paired(mine, base, sorted(p for p in mine if mine[p]["source"] == src))
                        for src in sorted({str(r["source"]) for r in records})}
    out["stock_scores_path"], out["stock_scores_sha256"] = _shown(path), sha256_file(path)
    return out, None


def heldout_metrics(variant: str, split: str, *, dtype: str | None = None,
                    run_config: str | None = None) -> dict[str, Any] | None:
    """The held-out loss metrics of a variant and split (the module docstring), or None when its score
    file does not cover the split's references as they are now in ``dtype`` (default generation.dtype),
    under ``run_config`` when given."""
    from lift.prompts import load

    dtype = dtype or config.get("generation.dtype")
    path = heldout_scores_path(variant, split)
    prompts = load("utility", "hi", split)
    check_references(prompts)
    records, outside, why = _heldout_items(path, prompts, dtype=dtype, run_config=run_config)
    if why:
        log.info("%s %s: no held-out loss metrics: %s", variant, split, why)
        return None
    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(str(r["source"]), []).append(r)
    vs_stock, note = _vs_stock(variant, split, records, prompts, dtype=dtype)
    meta = records[0]["meta"]
    return {"kind": "heldout_loss", "version": METRICS_VERSION, "variant": variant, "split": split,
            "lang": "hi", **_nll(records), "dtype": dtype, "model_rev": meta.get("model_rev"),
            "adapter_hash": meta.get("adapter_hash"), "template": meta.get("template"),
            "run_config": meta.get("run_config"), "heldout_version": meta.get("task_version"),
            "n_outside_split": outside, "scores_path": _shown(path), "scores_sha256": sha256_file(path),
            "by_source": {src: _nll(rows) for src, rows in sorted(by_source.items())},
            "vs_stock": vs_stock, "vs_stock_note": note, "created": _now()}


def write_heldout_metrics(variant: str, split: str, *, dtype: str | None = None,
                          run_config: str | None = None) -> Written | None:
    """Write (or, when its scores do not qualify, retire)
    outputs/metrics/heldout_loss/{variant}_{split}[-dtype].json and its .md."""
    doc = heldout_metrics(variant, split, dtype=dtype, run_config=run_config)
    return _write_metrics(metrics_path("heldout_nll", variant, split, dtype=dtype), doc, _heldout_md)


# --- markdown ---------------------------------------------------------------------------------------------


def _short(value: Any) -> str:
    return str(value)[:12] if value else "none"


def _belebele_md(doc: Mapping[str, Any]) -> str:
    lines = [f"# Belebele: {doc['variant']} ({doc['split']})", "",
             f"Format v{doc['format_version']}, {doc['scoring']} scoring, {doc['dtype']}, model "
             f"`{_short(doc['model_rev'])}`, adapter `{_short(doc['adapter_hash'])}`. Accuracy over every "
             "item of the split; a tie goes to the first letter.", "",
             "| Language | n | Correct | Accuracy | Ties | Gate fired | Scores (sha256) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lang, c in doc["langs"].items():
        gate = "-" if c["gate_fired_rate"] is None else f"{c['gate_fired_rate']:.3f}"
        lines.append(f"| {lang} | {c['n']} | {c['n_correct']} | {c['accuracy']:.4f} | {c['n_ties']} | "
                     f"{gate} | `{c['scores_sha256'][:12]}` |")
    if doc["left_out"]:
        lines += ["", "Left out:", ""] + [f"- {lang}: {why}" for lang, why in doc["left_out"].items()]
    return "\n".join(lines) + "\n"


def _nll_row(name: str, s: Mapping[str, Any]) -> str:
    return (f"| {name} | {s['n_rows']} | {s['n_tokens']} | {s['mean_nll']:.4f} | {s['row_mean_nll']:.4f} | "
            f"{s['perplexity']:.3f} |")


def _vs_row(name: str, s: Mapping[str, Any]) -> str:
    return (f"| {name} | {s['n']} | {s['diff_row_mean_nll']:+.4f} | [{s['lo']:+.4f}, {s['hi']:+.4f}] | "
            f"{s['p_value']:.4f} |")


def _heldout_md(doc: Mapping[str, Any]) -> str:
    lines = [f"# Held-out loss: {doc['variant']} (utility {doc['lang']}, {doc['split']})", "",
             f"{doc['dtype']}, model `{_short(doc['model_rev'])}`, adapter `{_short(doc['adapter_hash'])}`. "
             "NLL in nats per token of the reference answers (their closing tokens included, chat template "
             "on); perplexity is exp(mean NLL). aya: human-written references; ind: IndicTrans2 "
             "translations.", "",
             "| Rows | n | Tokens | Mean NLL | Row mean NLL | Perplexity |",
             "| --- | ---: | ---: | ---: | ---: | ---: |",
             _nll_row("all", doc), *(_nll_row(src, s) for src, s in doc["by_source"].items())]
    vs = doc["vs_stock"]
    if vs is None:
        lines += ["", f"vs stock: none ({doc['vs_stock_note']})."]
    else:
        lines += ["", "vs stock: this variant minus stock, row mean NLL, paired by pid (negative: it fits "
                      "the references better):", "",
                  "| Rows | n | Difference | 95% interval | p |", "| --- | ---: | ---: | --- | ---: |",
                  _vs_row("all", vs), *(_vs_row(src, s) for src, s in vs["by_source"].items())]
    return "\n".join(lines) + "\n"
