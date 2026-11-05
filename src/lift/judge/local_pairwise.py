"""Qwen3-8B in 4-bit as the utility judge's fallback (judges.utility.fallback). Implemented in P03.

Used only when a run asks for it (``lift utility --judge fallback``), never automatically: no fallback model
ever answers in the frontier judge's place, and its verdicts live apart (outputs/utility/fallback/...). It has
the frontier judge's interface (``pairwise(instructions, a, b)``: one result per pair, in input order,
{"winner": "A" | "B" | "TIE" | None, "reason", "raw", "error", "model_echo"}, a failure having winner None and
its cause in error; ``judge_id``; ``stats``) and its rules where they apply:

- The same rubric. configs/rubrics/utility_v1.md (judges.utility.rubric), read by
  ``lift.judge.frontier.load_rubric`` (kind "pairwise"): its system message is the system turn, and its user
  template is filled in one pass (``Rubric.render_pair``), so an answer holding "{response_b}" stays literal.
- Qwen3's own chat template with ``enable_thinking=False`` (the template writes an empty think block, so the
  reply is the verdict itself), the generation prompt added, tokenized without added special tokens. A pair
  holding one of the judge tokenizer's control-token strings ("<|im_end|>", ...) is not judged, because the
  template inserts text raw and the string would read as a turn marker; neither is a pair too long for the
  context (nothing is truncated). Both are failures, with no call.
- Greedy through ``lift.models.generate.greedy_batch``: an explicit greedy config, stop on the checkpoint's
  end-of-turn ids, the checkpoint's generation_config neutralised (its sampling defaults never leak in), at
  most MAX_NEW_TOKENS new tokens, batches sorted by length. An out-of-memory error halves the batch and
  retries it outside the except clause, so the failed batch's tensors are freed first.
- Strict parsing (``lift.judge.frontier.parse_pairwise_reply``): exactly one JSON object with the keys
  winner (A, B or TIE, exact case) and reason. Anything else, a reply holding a special token included, gets
  ONE reminder turn (the conversation, the reply, REMINDER); after it the item is a judge failure
  ("unparseable judge output"), never a guessed winner. Identical pairs in one call are judged once and
  share the result.
- Weights: bitsandbytes 4-bit NF4 without double quantization, computing in the run's precision (bf16 on an
  A100 or L4; fp16 on a T4, which has no bf16), at the revision resolved to a commit sha. The judge id names
  all of it, ``{model_id}@{sha}#4bit#{dtype}@{rubric version}``, so verdicts of two precisions or revisions
  never share a file. ``info`` (with the bitsandbytes version) and ``stack`` (torch, transformers, GPU class)
  go into the run's manifest.
- The model loads on the first call, so a run whose verdicts are current never loads it. Tests inject a
  model and tokenizer, and replace ``greedy_batch``.
- No instruction, answer or reply is ever logged: counts only.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lift import config
from lift.judge import frontier
from lift.models.generate import GreedyOutput, greedy_batch, is_oom

log = logging.getLogger(__name__)

QUANTIZATION = "4bit"                  # judges.utility.fallback.quantization: bitsandbytes NF4
MAX_NEW_TOKENS = 256                   # a verdict is one short JSON object; part of the judge's definition
REMINDER = (
    "Your reply was not in the required format. Reply again with exactly one JSON object and nothing "
    'else: {"winner": "A" | "B" | "TIE", "reason": "<at most 20 English words>"}'
)
CONTROL_TOKENS_ERROR = "control tokens in the judged text"
TOO_LONG_ERROR = "input too long for the judge's context"
UNPARSEABLE_ERROR = "unparseable judge output"
_FALLBACK_KEYS = frozenset({"id", "quantization", "revision"})
_STATS = ("pairs", "duplicates", "replies", "reminders", "failures", "control_tokens", "too_long", "cap_hits",
          "special_tokens", "oom_retries", "generated_tokens")


@dataclass(frozen=True)
class _Reply:
    text: str                  # the decoded reply, special tokens left out ("" when nothing was sent)
    special: bool              # the reply held a special token: never a valid verdict
    error: str | None = None   # why the conversation was not sent (control tokens, too long)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class LocalPairwiseJudge:
    """Qwen3-8B (4-bit) judging pairs of answers with the utility rubric (module docstring).

    ``revision`` is resolved to a commit sha; tests inject ``model`` and ``tokenizer`` together, and the given
    revision is then recorded as is. ``dtype`` is the compute precision of the 4-bit weights."""

    def __init__(self, model_id: str, revision: str, rubric_path: str | Path, *, device: str = "cuda:0",
                 dtype: str = "bf16", batch_size: int = 8, max_new_tokens: int = MAX_NEW_TOKENS,
                 model: Any = None, tokenizer: Any = None) -> None:
        from lift.models import load

        if dtype not in load.DTYPES:
            raise ValueError(f"dtype must be one of {load.DTYPES}, got {dtype!r}")
        if batch_size < 1 or max_new_tokens < 1:
            raise ValueError(f"batch_size and max_new_tokens must be >= 1, got {batch_size} and "
                             f"{max_new_tokens}")
        if (model is None) != (tokenizer is None):
            raise ValueError("inject both model and tokenizer (tests), or neither")
        rubric = frontier.load_rubric(rubric_path)
        if rubric.kind != "pairwise":
            raise ValueError(f"{rubric_path}: the utility judge needs a pairwise rubric (with "
                             f"{{instruction}}, {{response_a}} and {{response_b}}), not a {rubric.kind} one")
        rev = revision if model is not None else load.resolved_revision(model_id, revision)
        if not rev or rev == "main":
            raise ValueError(f"{model_id}: the judge's revision must be a commit sha, got {rev!r}")
        self.model_id = model_id
        self.revision = rev
        self.rubric = rubric
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.tokenizer: Any = None
        self._stack: dict[str, str] | None = None
        self._stats: dict[str, Any] = dict.fromkeys(_STATS, 0)
        self._stats["seconds"] = 0.0
        if model is not None:
            self._attach(model, tokenizer)

    @classmethod
    def from_config(cls, *, device: str = "cuda:0", dtype: str | None = None,
                    **kw: Any) -> LocalPairwiseJudge:
        """The judge judges.utility.fallback names ({id, quantization, optional revision}), with
        judges.utility's rubric; ``dtype`` defaults to ``generation.dtype``."""
        entry = config.get("judges.utility")
        fallback = entry.get("fallback")
        where = "judges.utility.fallback"
        if not isinstance(fallback, Mapping) or not fallback.get("id"):
            raise ValueError(f"{where} in configs/project.yaml must be a mapping with a model id")
        unknown = sorted(set(fallback) - _FALLBACK_KEYS)
        if unknown:
            raise ValueError(f"{where} has unknown keys {unknown}; known: {sorted(_FALLBACK_KEYS)}")
        if fallback.get("quantization") != QUANTIZATION:
            raise ValueError(f"{where}.quantization is {fallback.get('quantization')!r}; only "
                             f"{QUANTIZATION!r} (bitsandbytes NF4) is implemented")
        return cls(str(fallback["id"]), str(fallback.get("revision") or "main"), entry["rubric"],
                   device=device, dtype=dtype or str(config.get("generation.dtype")), **kw)

    @property
    def judge_id(self) -> str:
        """``{model_id}@{commit sha}#4bit#{compute dtype}@{rubric stem}:{sha256[:12]}``."""
        return f"{self.model_id}@{self.revision}#{QUANTIZATION}#{self.dtype}@{self.rubric.version}"

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    @property
    def info(self) -> dict[str, Any]:
        """What the judge is, for the run manifest."""
        from importlib import metadata

        try:
            bitsandbytes = metadata.version("bitsandbytes")
        except metadata.PackageNotFoundError:
            bitsandbytes = None
        return {"model_id": self.model_id, "revision": self.revision, "quantization": "bitsandbytes nf4",
                "double_quant": False, "dtype": self.dtype, "device": self.device,
                "batch_size": self.batch_size, "max_new_tokens": self.max_new_tokens,
                "rubric": self.rubric.version, "bitsandbytes": bitsandbytes}

    @property
    def stack(self) -> dict[str, str]:
        """``lift.models.generate.stack`` of the judge's device (torch, transformers, GPU class)."""
        if self._stack is None:
            from lift.models import generate

            self._stack = generate.stack(self.device)
        return self._stack

    # --------------------------------------------------------------------------------------- judging --

    def pairwise(self, instructions: Sequence[str], a: Sequence[str], b: Sequence[str]) -> list[dict]:
        """Per pair, in input order: {"winner": "A" | "B" | "TIE" | None, "reason", "raw", "error",
        "model_echo"}. A judge failure (control tokens, too long, a reply unparseable after the reminder) has
        winner None and its cause in ``error``; it is returned, never dropped."""
        if not len(instructions) == len(a) == len(b):
            raise ValueError(f"instructions, a and b differ in length: {len(instructions)}, {len(a)}, "
                             f"{len(b)}")
        for i, texts in enumerate(zip(instructions, a, b, strict=True)):
            if not all(isinstance(t, str) for t in texts):
                raise TypeError(f"pair {i}: the instruction and both responses must be str")
        users = [self.rubric.render_pair(instruction=i, response_a=x, response_b=y)
                 for i, x, y in zip(instructions, a, b, strict=True)]
        if not users:
            return []
        unique = list(dict.fromkeys(users))
        system = {"role": "system", "content": self.rubric.system}
        first = [[system, {"role": "user", "content": user}] for user in unique]
        results: dict[str, dict] = {}
        again: list[tuple[str, list[dict[str, str]]]] = []
        for user, turns, reply in zip(unique, first, self._complete(first), strict=True):
            parsed = self._parse(reply)
            if reply.error is not None or parsed is not None:
                results[user] = self._result(reply, parsed, reply.error)
                continue
            answer = [{"role": "assistant", "content": reply.text}] if reply.text.strip() else []
            again.append((user, [*turns, *answer, {"role": "user", "content": REMINDER}]))
        if again:
            self._stats["reminders"] += len(again)
            for (user, _), reply in zip(again, self._complete([t for _, t in again]), strict=True):
                parsed = self._parse(reply)
                error = reply.error or (UNPARSEABLE_ERROR if parsed is None else None)
                results[user] = self._result(reply, parsed, error)
        out = [dict(results[user]) for user in users]
        failures = sum(r["error"] is not None for r in out)
        self._stats["pairs"] += len(users)
        self._stats["duplicates"] += len(users) - len(unique)
        self._stats["failures"] += failures
        log.info("%s: judged %d pairs (%d distinct): %d reminder turns, %d failures", self.judge_id,
                 len(users), len(unique), len(again), failures)
        return out

    def _parse(self, reply: _Reply) -> tuple[str, str] | None:
        if reply.error is not None or reply.special:
            return None
        return frontier.parse_pairwise_reply(reply.text)

    def _result(self, reply: _Reply, parsed: tuple[str, str] | None, error: str | None) -> dict:
        winner, reason = parsed if parsed is not None and error is None else (None, None)
        return {"winner": winner, "reason": reason, "raw": reply.text, "error": error,
                "model_echo": f"{self.model_id}@{self.revision}"}

    # ----------------------------------------------------------------------------------- the model --

    def _ensure_loaded(self) -> None:
        """Load the tokenizer and the 4-bit model at the resolved revision, once."""
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        from lift.models import load

        load.check_dtype_supported(self.dtype, self.device)
        compute = load.torch_dtype(self.dtype)
        tokenizer = load.load_tokenizer(self.model_id, self.revision)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=compute)
        log.info("%s: loading the judge on %s", self.judge_id, self.device)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, dtype=compute,
                                                     quantization_config=quantization,
                                                     device_map={"": self.device})
        model.eval()
        model.requires_grad_(False)
        model.lift_checkpoint_generation_config = model.generation_config.to_dict()
        model.generation_config = load.neutral_generation_config(model.generation_config)
        self._attach(model, tokenizer)

    def _attach(self, model: Any, tokenizer: Any) -> None:
        from lift.models import load

        load.check_neutral_generation_config(model)
        context = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if not context:
            raise ValueError(f"{self.model_id}: the model config has no max_position_embeddings, so inputs "
                             "too long for the context cannot be detected")
        if tokenizer.pad_token_id is None:
            raise ValueError(f"{self.model_id}: the tokenizer has no pad token; left-padded batches need one")
        self.model, self.tokenizer = model, tokenizer
        self.context = int(context)
        self.pad_id = int(tokenizer.pad_token_id)
        self.stop_ids = load.stop_token_ids(tokenizer, model)
        self.control_tokens = tuple(sorted(load.control_tokens(tokenizer)))
        self.special_ids = frozenset(load.special_token_ids(tokenizer))

    def _encode(self, turns: Sequence[Mapping[str, str]]) -> list[int]:
        text = self.tokenizer.apply_chat_template([dict(t) for t in turns], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, not str")
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _complete(self, conversations: Sequence[Sequence[Mapping[str, str]]]) -> list[_Reply]:
        """One reply per conversation, in order. A conversation holding a control-token string, or too long
        for the context, is not sent and gets its error instead."""
        self._ensure_loaded()
        out: list[_Reply | None] = [None] * len(conversations)
        todo: list[tuple[int, list[int]]] = []
        for i, turns in enumerate(conversations):
            if any(token in turn["content"] for turn in turns for token in self.control_tokens):
                self._stats["control_tokens"] += 1
                out[i] = _Reply("", False, CONTROL_TOKENS_ERROR)
                continue
            ids = self._encode(turns)
            if len(ids) + self.max_new_tokens > self.context:
                self._stats["too_long"] += 1
                out[i] = _Reply("", False, TOO_LONG_ERROR)
                continue
            todo.append((i, ids))
        todo.sort(key=lambda row: (-len(row[1]), row[0]))
        self._decode(todo, out)
        replies = [r for r in out if r is not None]
        if len(replies) != len(out):
            missing = [i for i, r in enumerate(out) if r is None]
            raise RuntimeError(f"{len(missing)} conversations got no reply, e.g. indices {missing[:5]}")
        return replies

    def _decode(self, todo: list[tuple[int, list[int]]], out: list[_Reply | None]) -> None:
        """Decode ``todo`` (index, ids) in batches into ``out``. Halving after an out-of-memory error happens
        outside the ``except`` clause: its traceback holds the failed batch's tensors."""
        size = self.batch_size

        def run(batch: list[tuple[int, list[int]]]) -> None:
            nonlocal size
            out_of_memory = False
            try:
                t0 = time.perf_counter()
                outs = greedy_batch(self.model, [ids for _, ids in batch], pad_id=self.pad_id,
                                    stop_ids=self.stop_ids, max_new_tokens=self.max_new_tokens,
                                    device=self.device)
                self._stats["seconds"] += time.perf_counter() - t0
            except Exception as exc:  # only an out-of-memory error is retried; anything else re-raises
                if not is_oom(exc) or len(batch) == 1:
                    raise
                out_of_memory = True
            if out_of_memory:
                _free_memory()
                self._stats["oom_retries"] += 1
                half = len(batch) // 2
                size = max(1, min(size, half))
                log.warning("utility judge out of memory at batch size %d; retrying as %d + %d", len(batch),
                            half, len(batch) - half)
                run(batch[:half])
                run(batch[half:])
                return
            for (i, _), reply in zip(batch, outs, strict=True):
                out[i] = self._reply(reply)

        start = 0
        while start < len(todo):
            batch = todo[start: start + size]
            run(batch)
            start += len(batch)

    def _reply(self, output: GreedyOutput) -> _Reply:
        self._stats["replies"] += 1
        self._stats["generated_tokens"] += output.n_generated
        self._stats["cap_hits"] += int(not output.stopped)
        special = any(t in self.special_ids for t in output.ids)
        self._stats["special_tokens"] += int(special)
        return _Reply(self.tokenizer.decode(output.ids, skip_special_tokens=True), special)
