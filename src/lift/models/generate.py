"""The frozen generation harness. Implemented in P02; frozen once the stock dev eval starts.

After the freeze it only takes bug fixes, and each fix means re-running anything it touched (bump
``HARNESS_VERSION``: it is part of every cache key).

Behaviour:
- Greedy, ``max_new_tokens`` 256, default chat template with the preamble, left padding. The model's
  own generation_config.json (sampling defaults, repetition penalty, ...) is replaced by an explicit
  greedy config, so nothing leaks in from the checkpoint.
- Stops on the model's end-of-turn tokens (``lift.models.load.stop_token_ids``). ``hit_cap`` means no
  end-of-turn token within ``max_new_tokens``. ``n_new_tokens`` counts the content tokens of the
  output (special tokens and the stop token excluded), the count the "fewer than 5 new tokens"
  unclear rule reads; ``meta.n_generated`` counts every generated token for throughput.
- A prompt that would not fit in the context with ``max_new_tokens`` more tokens raises; nothing is
  ever truncated.
- Batches sorted by prompt length (longest first); batch size chosen from free memory unless given.
  An out-of-memory error halves the batch and retries it (never skips a prompt).
- One GenerationRecord per prompt, appended to
  outputs/generations/{variant}/{set}_{lang}_{split}.jsonl as each batch finishes. On restart, keys
  already present are skipped (resumable across sessions); a record for the same pid under a
  different key (stale translation, other dtype, other template) raises instead of mixing.
- ``meta`` carries dtype, GPU name, device, batch size and, for CAST wrappers, ``gate_fired``.
- Every run writes a manifest (lift.manifest) with prompts/sec and tokens/sec.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lift import config
from lift.cachekey import generation_key
from lift.io import append_jsonl, read_jsonl, sha256_obj, sha256_text
from lift.models.variants import LoadedVariant
from lift.paths import generations_dir
from lift.schemas import GenerationRecord, PromptRecord

log = logging.getLogger(__name__)

HARNESS_VERSION = 1


class StaleOutputError(RuntimeError):
    """A generations file holds records whose keys no longer match the current inputs."""


def generations_path(variant_name: str, set_name: str, lang: str, split: str) -> Path:
    return generations_dir() / variant_name / f"{set_name}_{lang}_{split}.jsonl"


def gen_config(variant: LoadedVariant, *, max_new_tokens: int | None = None) -> dict[str, Any]:
    """Everything about decoding that can change an output. It goes into every cache key, so bf16
    and fp16 outputs (or a different stop set or cap) never share a key."""
    from lift.models.load import stop_token_ids

    gen = config.get("generation")
    if gen["do_sample"] is not False:
        raise ValueError("generation.do_sample must be false: decoding is greedy (locked Oct 28)")
    return {
        "harness": HARNESS_VERSION,
        "do_sample": False,
        "max_new_tokens": int(max_new_tokens or gen["max_new_tokens"]),
        "dtype": variant.dtype,
        "template": gen["template"],
        "stop_ids": stop_token_ids(variant.tokenizer, variant.hf_model),
    }


def stack(device: str) -> dict[str, str]:
    """The software and GPU class a generation ran on. Not in the cache key (a library upgrade must
    not silently re-key finished cells), but in ``run_config``: one file never mixes stacks."""
    from importlib import metadata

    out = {name: metadata.version(name) for name in ("torch", "transformers")}
    out["gpu_capability"] = "cpu"
    if str(device).startswith("cuda"):
        import torch

        major, minor = torch.cuda.get_device_capability(torch.device(device))
        out["gpu_capability"] = f"{major}.{minor}"
    return out


def run_config_digest(variant: LoadedVariant, gen_cfg: dict[str, Any], env: dict[str, str]) -> str:
    """Everything in the cache key except the prompt, plus the stack, so a generations file can prove
    it holds one configuration, also for prompts outside the current run (a --limit run in another
    dtype, or a resume after the runtime upgraded transformers or moved to another GPU class)."""
    from lift.cachekey import intervention_digest

    return sha256_obj({"model_rev": variant.model_rev, "adapter_hash": variant.adapter_hash,
                       "intervention": intervention_digest(variant.spec),
                       "template_hash": variant.template_hash, "gen_cfg": gen_cfg, "stack": env})[:16]


def record_key(variant: LoadedVariant, prompt: PromptRecord, gen_cfg: dict[str, Any]) -> str:
    return generation_key(
        model_rev=variant.model_rev, adapter_hash=variant.adapter_hash, spec=variant.spec,
        template_hash=variant.template_hash, gen_cfg=gen_cfg, pid=prompt.pid, lang=prompt.lang,
        prompt_text=prompt.text,
    )


def prompt_id(set_name: str, lang: str, pid: str) -> tuple[str, str, str]:
    """A prompt's identity in a generations file. pids are shared across languages by design."""
    return (set_name, lang, pid)


def read_generations(path: Path) -> list[GenerationRecord]:
    """All records of one generations file. Raises on a duplicate key or prompt (a corrupt file)."""
    records = [GenerationRecord.model_validate(row) for row in read_jsonl(path)]
    seen_keys: set[str] = set()
    seen_prompts: set[tuple[str, str, str]] = set()
    for r in records:
        ident = prompt_id(r.set, r.lang, r.pid)
        if r.key in seen_keys or ident in seen_prompts:
            raise ValueError(f"{path}: duplicate record for {r.set}/{r.lang}/{r.pid} (key {r.key[:12]})")
        seen_keys.add(r.key)
        seen_prompts.add(ident)
    return records


def repair_tail(path: Path) -> bool:
    """Make a file killed mid-append readable again: a last line without its newline is completed
    if it is valid JSON, dropped otherwise. Returns True if the file changed.

    Only the partial tail is touched (an append, or an in-place truncate), so a kill during the
    repair itself can never lose a complete record."""
    import json

    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail.decode("utf-8"))
        complete = True
    except (UnicodeDecodeError, json.JSONDecodeError):
        complete = False
    with open(path, "r+b") as f:
        if complete:
            f.seek(0, 2)
            f.write(b"\n")
        else:
            f.truncate(cut)
    log.warning("%s: repaired a truncated last line (%d bytes)", path, len(tail))
    return True


@dataclass(frozen=True)
class GreedyOutput:
    ids: list[int]          # content ids, without the stop token
    stopped: bool           # an end-of-turn token ended it (False: hit the cap)
    n_generated: int        # every generated token, including the stop token


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def greedy_config(*, max_new_tokens: int, stop_ids: Sequence[int], pad_id: int) -> Any:
    """An explicit greedy GenerationConfig.

    Values are spelled out as their neutral settings where they have one: transformers fills every
    None from the model's own generation_config, which ``check_neutral_generation_config`` has
    already checked sets nothing but special-token ids. So a checkpoint's repetition penalty or
    sampling defaults can never reach greedy decoding.
    """
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=max_new_tokens,
        min_length=0, eos_token_id=list(stop_ids), pad_token_id=pad_id,
        temperature=1.0, top_k=50, top_p=1.0, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0,
        repetition_penalty=1.0, encoder_repetition_penalty=1.0, length_penalty=1.0,
        no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0, renormalize_logits=False,
        remove_invalid_values=False, use_cache=True, output_scores=False, return_dict_in_generate=False,
    )


def greedy_batch(model: Any, rows: Sequence[Sequence[int]], *, pad_id: int, stop_ids: Sequence[int],
                 max_new_tokens: int, device: str, gen_config_obj: Any | None = None) -> list[GreedyOutput]:
    """Greedy-decode a batch of token id lists with left padding and an attention mask."""
    import torch

    if not rows:
        return []
    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, width - len(r):] = torch.tensor(list(r), dtype=torch.long)
        mask[i, width - len(r):] = 1
    cfg = gen_config_obj or greedy_config(max_new_tokens=max_new_tokens, stop_ids=stop_ids, pad_id=pad_id)
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                             generation_config=cfg)
    sequences = out if isinstance(out, torch.Tensor) else out.sequences
    stop = set(stop_ids)
    results = []
    for new in sequences[:, width:].tolist():
        at = next((j for j, t in enumerate(new) if t in stop), None)
        if at is None:
            if len(new) != max_new_tokens:
                raise RuntimeError(f"generation ended after {len(new)} tokens with no stop token "
                                   f"and below the cap of {max_new_tokens}")
            results.append(GreedyOutput(ids=new, stopped=False, n_generated=len(new)))
        else:
            results.append(GreedyOutput(ids=new[:at], stopped=True, n_generated=at + 1))
    return results


def auto_batch_size(variant: LoadedVariant, max_len: int, *, cap: int = 64) -> int:
    """A batch size the KV cache of ``max_len``-token sequences should fit in, from free memory.
    Deliberately rough: an OOM still halves the batch."""
    import torch

    if not str(variant.device).startswith("cuda") or not torch.cuda.is_available():
        return 8
    cfg = variant.hf_model.config
    layers = int(cfg.num_hidden_layers)
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    elem = 4 if variant.dtype == "fp32" else 2
    per_seq = 2 * layers * kv_heads * head_dim * elem * max_len          # K and V
    per_seq += 2 * int(cfg.vocab_size) * 4                                # last-step logits, fp32
    per_seq += 8 * int(cfg.hidden_size) * max_len * elem                  # prefill activations
    free, _total = torch.cuda.mem_get_info(torch.device(variant.device))
    return int(max(1, min(cap, (0.6 * free) // per_seq)))


def _device_name(device: str) -> str:
    from lift.models.load import device_name

    return device_name(device)


@dataclass
class _Row:
    prompt: PromptRecord
    key: str
    ids: list[int]


def generate(variant: LoadedVariant, prompts: Sequence[PromptRecord], *, out_path: Path,
             batch_size: int | None = None, resume: bool = True, max_new_tokens: int | None = None,
             manifest: bool = True) -> Path:
    """Generate one record per prompt into ``out_path`` (see the module docstring)."""
    from lift.manifest import finish_run, start_run
    from lift.models.load import check_neutral_generation_config, encode_chat

    out_path = Path(out_path)
    check_neutral_generation_config(variant.hf_model)
    gen_cfg = gen_config(variant, max_new_tokens=max_new_tokens)
    cap = gen_cfg["max_new_tokens"]
    stop_ids = gen_cfg["stop_ids"]
    tok = variant.tokenizer
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise ValueError("the tokenizer has no pad token; left-padded batches need one")
    idents = [prompt_id(p.set, p.lang, p.pid) for p in prompts]
    if len(set(idents)) != len(idents):
        raise ValueError("duplicate prompts (same set, language and pid) in the prompt list")

    rows = [_Row(p, record_key(variant, p, gen_cfg), encode_chat(tok, p.text)) for p in prompts]
    context = getattr(variant.hf_model.config, "max_position_embeddings", None)
    if context:
        too_long = [r.prompt.pid for r in rows if len(r.ids) + cap > context]
        if too_long:
            raise ValueError(f"{len(too_long)} prompts plus {cap} new tokens exceed the {context}-token "
                             f"context, e.g. {too_long[:5]}; nothing is truncated")

    env = stack(variant.device)
    run_config = run_config_digest(variant, gen_cfg, env)
    done: dict[tuple[str, str, str], str] = {}
    if out_path.exists():
        if not resume:
            raise FileExistsError(f"{out_path} exists; pass resume=True or move it aside")
        repair_tail(out_path)
        existing = read_generations(out_path)
        other = sorted({r.pid for r in existing if r.meta.get("run_config") != run_config})
        if any(r.variant != variant.spec.name for r in existing) or other:
            raise StaleOutputError(
                f"{out_path} holds records made under another configuration (variant, dtype, template, "
                f"model, adapter, harness version, torch/transformers version or GPU class), e.g. "
                f"{other[:5]}. Move the file aside; one file is one configuration, outputs are never mixed.")
        done = {prompt_id(r.set, r.lang, r.pid): r.key for r in existing}

    def ident(row: _Row) -> tuple[str, str, str]:
        return prompt_id(row.prompt.set, row.prompt.lang, row.prompt.pid)

    stale = [row.prompt.pid for row in rows if ident(row) in done and done[ident(row)] != row.key]
    if stale:
        raise StaleOutputError(
            f"{out_path}: {len(stale)} records were made with other inputs (prompt text, dtype, template, "
            f"model or harness version), e.g. {stale[:5]}. Move the file aside; outputs are never mixed.")
    todo = sorted((row for row in rows if ident(row) not in done),
                  key=lambda row: (-len(row.ids), row.prompt.lang, row.prompt.pid))

    gpu = _device_name(variant.device)
    provenance = {"model_rev": variant.model_rev, "harness": HARNESS_VERSION,
                  "template": variant.template_hash[:16], **env}
    run = start_run("generate", variant=variant.spec.name, out_path=str(out_path), dtype=variant.dtype,
                    device=variant.device, gpu=gpu, gen_cfg=gen_cfg, n_prompts=len(rows),
                    n_resumed=len(rows) - len(todo), run_config=run_config, adapter_hash=variant.adapter_hash,
                    template_hash=variant.template_hash, **provenance) if manifest else None
    if not todo:
        log.info("%s: all %d prompts already generated", out_path.name, len(rows))
        if run is not None:
            finish_run(run, status="ok", n_written=0)
        return out_path

    size = batch_size or auto_batch_size(variant, max(len(r.ids) for r in todo) + cap)
    log.info("%s: generating %d prompts (%d resumed), batch size %d, %s on %s",
             out_path.name, len(todo), len(rows) - len(todo), size, variant.dtype, gpu)
    stats = {"n_written": 0, "n_generated_tokens": 0, "cap_hits": 0, "oom_retries": 0, "seconds": 0.0,
             "batch_sizes": set()}
    gcfg = greedy_config(max_new_tokens=cap, stop_ids=stop_ids, pad_id=pad_id)

    def run_batch(batch: list[_Row]) -> None:
        nonlocal size
        oom = False
        try:
            t0 = time.perf_counter()
            outs = greedy_batch(variant.model, [r.ids for r in batch], pad_id=pad_id, stop_ids=stop_ids,
                                max_new_tokens=cap, device=variant.device, gen_config_obj=gcfg)
            seconds = time.perf_counter() - t0
        except Exception as exc:  # only an out-of-memory error is retried; anything else re-raises
            if not is_oom(exc) or len(batch) == 1:
                raise
            oom = True
        if oom:
            # Retry outside the except clause: its traceback would keep the failed batch's tensors
            # (KV cache, logits) alive, and the halves could run out of memory again.
            _free_memory()
            stats["oom_retries"] += 1
            half = len(batch) // 2
            size = max(1, min(size, half))
            log.warning("out of memory at batch size %d; retrying as %d + %d", len(batch), half,
                        len(batch) - half)
            run_batch(batch[:half])
            run_batch(batch[half:])
            return
        extra = variant.batch_meta() if variant.batch_meta is not None else [{}] * len(batch)
        records = []
        for r, o, more in zip(batch, outs, extra, strict=True):
            meta = {**more, **provenance, "run_config": run_config}
            rec = _record(variant, r, o, seconds / len(batch), len(batch), gpu, tok, meta)
            records.append(rec.model_dump())
            stats["n_generated_tokens"] += o.n_generated
            stats["cap_hits"] += int(not o.stopped)
        append_jsonl(out_path, records)
        stats["n_written"] += len(records)
        stats["seconds"] += seconds
        stats["batch_sizes"].add(len(batch))

    def summary() -> dict[str, Any]:
        secs = stats["seconds"] or 1e-9
        return {
            "n_written": stats["n_written"], "cap_hits": stats["cap_hits"],
            "oom_retries": stats["oom_retries"],
            "batch_sizes": sorted(stats["batch_sizes"]), "generate_seconds": round(stats["seconds"], 3),
            "prompts_per_sec": round(stats["n_written"] / secs, 4),
            "tokens_per_sec": round(stats["n_generated_tokens"] / secs, 2),
            "n_generated_tokens": stats["n_generated_tokens"], "devices": 1,
        }

    i = 0
    try:
        while i < len(todo):
            batch = todo[i: i + size]
            run_batch(batch)
            i += len(batch)
            log.info("%s: %d/%d", out_path.name, stats["n_written"], len(todo))
    except BaseException as exc:
        # A crashed run still records the GPU time it spent (the cost section reads manifests).
        if run is not None:
            finish_run(run, status="failed", error=type(exc).__name__, **summary())
        raise
    done_stats = summary()
    log.info("%s: wrote %d records, %.2f prompts/s, %.1f tokens/s, %d hit the cap", out_path.name,
             done_stats["n_written"], done_stats["prompts_per_sec"], done_stats["tokens_per_sec"],
             done_stats["cap_hits"])
    if run is not None:
        finish_run(run, status="ok", **done_stats)
    return out_path


def _record(variant: LoadedVariant, row: _Row, out: GreedyOutput, seconds: float, batch_size: int,
            gpu: str, tok: Any, more: dict) -> GenerationRecord:
    from lift.models.load import special_token_ids

    special = special_token_ids(tok)
    return GenerationRecord(
        key=row.key, variant=variant.spec.name, pid=row.prompt.pid, lang=row.prompt.lang,
        set=row.prompt.set, split=row.prompt.split, prompt_sha=sha256_text(row.prompt.text),
        output=tok.decode(out.ids, skip_special_tokens=True),
        n_prompt_tokens=len(row.ids), n_new_tokens=sum(t not in special for t in out.ids),
        hit_cap=not out.stopped, seconds=round(seconds, 4),
        meta={"dtype": variant.dtype, "gpu": gpu, "device": variant.device, "batch_size": batch_size,
              "n_generated": out.n_generated, **more},
    )


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def greedy_token_ids(model: Any, tokenizer: Any, texts: Sequence[str], *, device: str,
                     max_new_tokens: int | None = None, batch_size: int = 4) -> list[GreedyOutput]:
    """Greedy outputs for raw user texts, through the same template, stop set and decoder as
    ``generate`` but without records (merge parity checks use it)."""
    from lift.models.load import check_neutral_generation_config, encode_chat, stop_token_ids

    check_neutral_generation_config(model)
    cap = int(max_new_tokens or config.get("generation.max_new_tokens"))
    stop_ids = stop_token_ids(tokenizer, model)
    ids = [encode_chat(tokenizer, t) for t in texts]
    out: list[GreedyOutput] = []
    for i in range(0, len(ids), batch_size):
        out.extend(greedy_batch(model, ids[i: i + batch_size], pad_id=tokenizer.pad_token_id,
                                stop_ids=stop_ids, max_new_tokens=cap, device=device))
    return out
