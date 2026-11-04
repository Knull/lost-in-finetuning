"""Loading Tiny Aya and formatting prompts. Implemented in P02.

Tiny Aya facts that matter here (CohereLabs/tiny-aya-global, gated, CC-BY-NC-4.0):
- cohere2 / Cohere2ForCausalLM, 36 layers at ``model.model.layers``, d_model 2048, 262k vocab,
  tied embeddings, parallel attention+MLP blocks, 3:1 sliding-window:global attention.
- Weights are bf16 (~6.7 GB). Reported runs use bf16 on a Colab A100/L4. A T4 has no bf16, so it
  loads fp16, which may overflow on a bf16-trained model: Gate 0 compares T4 fp16 with bf16 on 200
  prompts (finite activations, same judge labels) before any T4 number is reported.
- The chat template always injects a default system preamble (~366 tokens, "contextual safety
  mode"). We keep it on for training and evaluation alike. Never strip it except in the explicit
  preamble on/off diagnosis.
- tiny-aya-base has no chat template; it is only ever loaded as the ReSta reference.
- Always resolve ``revision`` to a commit sha and record it; "main" is not a revision.

Every model loaded here gets a neutral ``generation_config`` (only its end-of-turn, pad and BOS ids
are kept). transformers fills any generation setting a caller leaves unset from the checkpoint's
generation_config.json, so sampling defaults or a repetition penalty shipped with a checkpoint
would otherwise leak into "greedy" decoding.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lift.io import sha256_file, sha256_obj

DTYPES = ("fp32", "fp16", "bf16")
# Tokens that end a Cohere chat turn. Any of them present in the vocabulary is a stop token, on top
# of the checkpoint's eos ids.
END_OF_TURN_TOKENS = ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>")
PROBE_TEXT = "Probe: what is 2 + 2? / नमस्ते / 你好"
_SHA = re.compile(r"[0-9a-f]{40}")


def torch_dtype(name: str) -> Any:
    import torch

    table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"dtype must be one of {DTYPES}, got {name!r}")
    return table[name]


def device_name(device: str) -> str:
    """The GPU's name for records and manifests ("cpu" off GPU)."""
    if not str(device).startswith("cuda"):
        return "cpu"
    import torch

    return torch.cuda.get_device_name(torch.device(device))


def check_dtype_supported(dtype: str, device: str) -> None:
    """Refuse bf16 on a GPU that only emulates it (a T4): every reported number is native bf16."""
    torch_dtype(dtype)
    if dtype != "bf16" or not str(device).startswith("cuda"):
        return
    import torch

    major, _minor = torch.cuda.get_device_capability(torch.device(device))
    if major < 8:
        raise ValueError(f"{device_name(device)} has no native bf16 (compute capability {major}.x); "
                         "use fp16 there, which counts only if Gate 0's fp16-vs-bf16 check passes")


def _local_revision(path: Path) -> str:
    """A content hash standing in for the commit sha of a model saved to a local directory."""
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    digest = sha256_obj([[p.relative_to(path).as_posix(), sha256_file(p)] for p in files])
    return f"local-{digest[:40]}"


def resolved_revision(model_id: str, revision: str) -> str:
    """Commit sha for ``revision`` (huggingface_hub.model_info(...).sha).

    A full 40-character sha is returned as is; a local directory gets ``local-<content hash>``.
    """
    if Path(model_id).is_dir():
        return _local_revision(Path(model_id))
    if revision and _SHA.fullmatch(revision):
        return revision
    from huggingface_hub import HfApi

    sha = HfApi().model_info(model_id, revision=revision or "main").sha
    if not sha or not _SHA.fullmatch(sha):
        raise ValueError(f"could not resolve {model_id}@{revision} to a commit sha (got {sha!r})")
    return sha


def _hub_revision(model_id: str, revision: str) -> str | None:
    return None if Path(model_id).is_dir() else revision


def load_tokenizer(model_id: str, revision: str) -> Any:
    """Tokenizer with ``padding_side="left"`` for batched generation."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=_hub_revision(model_id, revision))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        raise ValueError(f"{model_id} has no pad token; batched generation needs one")
    return tok


def neutral_generation_config(original: Any) -> Any:
    """A GenerationConfig that keeps only the special-token ids of ``original``."""
    from transformers import GenerationConfig

    eos = getattr(original, "eos_token_id", None)
    return GenerationConfig(
        bos_token_id=getattr(original, "bos_token_id", None),
        eos_token_id=eos,
        pad_token_id=getattr(original, "pad_token_id", None),
    )


_NEUTRAL_KEYS = frozenset({"bos_token_id", "eos_token_id", "pad_token_id", "transformers_version",
                           "_from_model_config"})


def check_neutral_generation_config(model: Any) -> None:
    """Raise if ``model.generation_config`` sets anything beyond its special-token ids. Models must
    come through ``load_model``; a wrapper without a generation_config (CAST) is checked by its
    builder instead."""
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    extra = sorted(k for k in cfg.to_diff_dict() if k not in _NEUTRAL_KEYS)
    if extra:
        raise ValueError(f"the model's generation_config sets {extra}; load models with "
                         "lift.models.load.load_model, which neutralises it")


def load_model(model_id: str, revision: str, *, dtype: str = "bf16", device: str = "cuda:0") -> Any:
    """The model in ``dtype`` on ``device``, in eval mode, with a neutral generation config.

    ``revision`` should already be resolved (``resolved_revision``) so the weights loaded are the
    ones the cache key records.
    """
    from transformers import AutoModelForCausalLM

    check_dtype_supported(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=_hub_revision(model_id, revision), dtype=torch_dtype(dtype),
        device_map={"": device},
    )
    model.eval()
    model.requires_grad_(False)
    model.lift_checkpoint_generation_config = model.generation_config.to_dict()
    model.generation_config = neutral_generation_config(model.generation_config)
    return model


def chat_text(tokenizer: Any, user_text: str) -> str:
    """The exact string the model sees: chat template, default preamble, generation prompt."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                         add_generation_prompt=True)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, not str")
    return text


def control_tokens(tokenizer: Any) -> set[str]:
    """The text of every special token. The chat template inserts user text raw, so one of these
    inside a prompt would be tokenized as a real turn marker and reshape the conversation."""
    names = set(getattr(tokenizer, "all_special_tokens", None) or [])
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    names.update(t.content for t in decoder.values() if getattr(t, "special", False))
    return {n for n in names if n}


def encode_chat(tokenizer: Any, user_text: str) -> list[int]:
    """Token ids of ``chat_text``. The template already writes the BOS token, so the tokenizer must
    not add another one (a doubled BOS shifts every position the model was trained on). A prompt
    containing a control-token string raises instead of silently injecting a turn."""
    found = sorted(t for t in control_tokens(tokenizer) if t in user_text)
    if found:
        raise ValueError(f"the prompt text contains control tokens {found}; fix the prompt, the chat "
                         "template would treat them as turn markers")
    ids = list(tokenizer(chat_text(tokenizer, user_text), add_special_tokens=False)["input_ids"])
    bos = tokenizer.bos_token_id
    if bos is not None and ids[:2] == [bos, bos]:
        raise ValueError("the chat text starts with a doubled BOS token")
    return ids


def encode_chat_turn(tokenizer: Any, user_text: str, assistant_text: str) -> tuple[list[int], int]:
    """Token ids of one user turn followed by the assistant's answer, as the chat template renders the
    pair, and how many of them are the prompt.

    ``ids[:n_prompt]`` is exactly ``encode_chat(tokenizer, user_text)``, what the model sees at generation
    time (BOS, the default preamble, the user turn, the generation prompt). ``ids[n_prompt:]`` is the
    answer as the template writes it, with the tokens that close the turn (``<|END_RESPONSE|>
    <|END_OF_TURN_TOKEN|>`` for Tiny Aya), the tokens the model emits before it stops. Training
    supervises exactly ``ids[n_prompt:]`` (lift.train.lora) and the held-out loss scores exactly them
    (lift.models.loglik), so the two can never disagree on what "assistant tokens" are.

    The ids are the template's own tokenization of the whole conversation, so the answer is tokenized in
    context. Raises if either text holds a control-token string, if the template does not render the
    answer right after the generation prompt, if the conversation's ids do not start with exactly the
    prompt's ids (a merge across the boundary would make the supervised tokens differ from what the
    model sees at generation time), or if the answer renders to no tokens."""
    found = sorted(t for t in control_tokens(tokenizer) if t in assistant_text)
    if found:
        raise ValueError(f"the answer text contains control tokens {found}; the chat template would treat "
                         "them as turn markers")
    prompt_ids = encode_chat(tokenizer, user_text)
    prompt_text = chat_text(tokenizer, user_text)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}],
        tokenize=False, add_generation_prompt=False)
    if not isinstance(full, str):
        raise TypeError(f"apply_chat_template(tokenize=False) returned {type(full).__name__}, not str")
    if not full.startswith(prompt_text):
        raise ValueError("the chat template does not render the assistant's answer right after the "
                         "generation prompt, so the answer tokens cannot be told from the prompt's")
    full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("the conversation's token ids do not start with the prompt's: the tokenizer merges "
                         "across the end of the generation prompt, so the supervised tokens would not follow "
                         "what the model sees at generation time")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("the assistant's answer renders to no tokens")
    return full_ids, len(prompt_ids)


def template_hash(tokenizer: Any) -> str:
    """sha256 over the chat template source and the rendering of a fixed probe message, so a
    changed preamble or template changes every cache key."""
    return sha256_obj({
        "chat_template": tokenizer.chat_template,
        "probe_text": chat_text(tokenizer, PROBE_TEXT),
        "probe_ids": encode_chat(tokenizer, PROBE_TEXT),
        "special": [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token],
    })


def special_token_ids(tokenizer: Any) -> set[int]:
    """Every id ``decode(skip_special_tokens=True)`` drops: the named special tokens plus every added
    token flagged special (the Cohere turn and response markers are added tokens)."""
    ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    ids.update(int(i) for i, t in decoder.items() if getattr(t, "special", False))
    return ids


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(value)]


def stop_token_ids(tokenizer: Any, model: Any = None) -> list[int]:
    """Every id that ends a turn: the checkpoint's eos ids, the tokenizer's eos, and the Cohere
    end-of-turn tokens present in the vocabulary. Sorted, so it can sit in a cache key."""
    ids = set(_as_list(getattr(getattr(model, "generation_config", None), "eos_token_id", None)))
    ids.update(_as_list(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    ids.update(vocab[t] for t in END_OF_TURN_TOKENS if t in vocab)
    if not ids:
        raise ValueError("no end-of-turn token found; generation would only ever stop at the cap")
    return sorted(ids)


def template_stats(tokenizer: Any) -> dict[str, Any]:
    """What Gate 0 reports about the template: token counts before and after the user's text, the
    system/preamble block's length, and the BOS check. Counts only, no template text."""
    sentinel = "LIFTSENTINEL"
    text = chat_text(tokenizer, sentinel)
    if text.count(sentinel) != 1:
        raise ValueError("the chat template renders the user's text other than exactly once")
    prefix, suffix = text.split(sentinel)
    count = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    ids = encode_chat(tokenizer, sentinel)
    bos = tokenizer.bos_token_id
    stats: dict[str, Any] = {
        "n_tokens_with_empty_user": len(encode_chat(tokenizer, "")),
        "prefix_tokens": count(prefix),
        "suffix_tokens": count(suffix),
        "bos_count": ids.count(bos) if bos is not None else 0,
        "starts_with_bos": bos is not None and ids[:1] == [bos],
        "generation_prompt": suffix.strip()[-60:],
    }
    if "<|SYSTEM_TOKEN|>" in prefix:
        system = prefix.split("<|SYSTEM_TOKEN|>", 1)[1].split("<|END_OF_TURN_TOKEN|>", 1)[0]
        stats["system_block_tokens"] = count(system)
    return stats
