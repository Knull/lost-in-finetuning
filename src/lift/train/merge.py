"""Merge a LoRA adapter into Global and check parity. Implemented in P02 (the harness needs it).

Parity check (Gate 0 and every new adapter): greedy outputs of merged vs unmerged on 20 dev
prompts must match token for token in fp32; in fp16 report the share that match and the first
divergence position.

The adapter hash goes into every cache key for variants built on it. It covers the sha256 of
adapter_model.safetensors and the adapter_config.json fields that change what the weights do
(rank, alpha, target modules, ...): the same weights with another lora_alpha are another model.
Paths and version strings in adapter_config.json are left out, so a copy on another machine
hashes the same.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lift import config
from lift.io import sha256_file, sha256_obj

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
BEHAVIOUR_KEYS = ("peft_type", "r", "lora_alpha", "target_modules", "use_rslora", "use_dora",
                  "fan_in_fan_out", "bias", "lora_bias", "modules_to_save", "layers_to_transform",
                  "layers_pattern", "rank_pattern", "alpha_pattern", "exclude_modules")


def adapter_hash(adapter_dir: Path) -> str:
    path = Path(adapter_dir) / ADAPTER_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (expected a PEFT adapter saved as safetensors)")
    cfg_path = Path(adapter_dir) / ADAPTER_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} not found")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    behaviour = {k: cfg.get(k) for k in BEHAVIOUR_KEYS}
    if isinstance(behaviour["target_modules"], list):
        behaviour["target_modules"] = sorted(behaviour["target_modules"])
    return sha256_obj({"weights_sha256": sha256_file(path), "config": behaviour})


def _base(base_id: str | None, base_revision: str | None) -> tuple[str, str]:
    from lift.models.load import resolved_revision

    base_id = base_id or config.get("models.global.id")
    return base_id, resolved_revision(base_id, base_revision or config.get("models.global.revision"))


def check_adapter_base(adapter_dir: Path, base_id: str) -> None:
    """Raise if the adapter says it was trained on a different hub model than ``base_id``. A local
    path in either place (a test, or a snapshot directory) is not comparable and passes."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8"))
    trained_on = cfg.get("base_model_name_or_path")
    if not trained_on or Path(trained_on).exists() or Path(base_id).exists():
        return
    if trained_on != base_id:
        raise ValueError(f"{adapter_dir} was trained on {trained_on}, not {base_id}")


TEMPLATE_PROBES = ("Probe: what is 2 + 2?", "नमस्ते, आप कैसे हैं?", "你好", "")


def check_adapter_template(adapter_dir: Path, tokenizer: Any) -> None:
    """Raise if the tokenizer saved with the adapter (TRL saves one) renders chat text differently
    from Global's: evaluating with a template the adapter was not trained with would confound drift
    with formatting. Renderings and token ids are compared, not template sources, so Global's named
    templates (saved as chat_template.jinja plus additional_chat_templates/) compare correctly."""
    from lift.models.load import chat_text, encode_chat

    directory = Path(adapter_dir)
    if not any((directory / f).is_file() for f in ("tokenizer_config.json", "tokenizer.json",
                                                    "chat_template.jinja")):
        return
    from transformers import AutoTokenizer

    try:
        saved = AutoTokenizer.from_pretrained(str(directory))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{directory} has tokenizer files that cannot be loaded to compare its chat "
                         f"template with Global's ({type(exc).__name__})") from exc
    if saved.chat_template is None:
        return
    for probe in TEMPLATE_PROBES:
        if chat_text(saved, probe) != chat_text(tokenizer, probe) or \
                encode_chat(saved, probe) != encode_chat(tokenizer, probe):
            raise ValueError(f"{directory} has a chat template that renders differently from Global's")


def merged_model(adapter_dir: Path, *, device: str = "cuda:0", dtype: str = "bf16",
                 base_id: str | None = None, base_revision: str | None = None) -> Any:
    """Global with the LoRA adapter merged in (PEFT ``merge_and_unload``), in ``dtype`` on ``device``."""
    from peft import PeftModel

    from lift.models.load import load_model

    base_id, rev = _base(base_id, base_revision)
    check_adapter_base(adapter_dir, base_id)
    base = load_model(base_id, rev, dtype=dtype, device=device)
    merged = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).merge_and_unload()
    merged.eval()
    return merged


def parity_check(adapter_dir: Path, n_prompts: int = 20, *, prompts: Sequence[str] | None = None,
                 device: str = "cuda:0", dtype: str = "fp32", base_id: str | None = None,
                 base_revision: str | None = None, max_new_tokens: int | None = None,
                 batch_size: int = 4, seed: int = 0) -> dict:
    """Greedy outputs of the unmerged (PEFT) and merged adapter on the same prompts, token by token.

    ``prompts`` defaults to ``n_prompts`` dev prompts mixed across sets and languages. Both passes
    use the harness's template, stop set and decoder (``greedy_token_ids``). Returns counts, the
    share that match, and each prompt's first divergence position (None when identical).
    """
    from peft import PeftModel

    from lift.models.generate import greedy_token_ids
    from lift.models.load import load_model, load_tokenizer

    if prompts is None:
        from lift.prompts import mixed_sample

        prompts = [r.text for r in mixed_sample(n_prompts, split="dev", seed=seed)]
    base_id, rev = _base(base_id, base_revision)
    check_adapter_base(adapter_dir, base_id)
    tok = load_tokenizer(base_id, rev)
    base = load_model(base_id, rev, dtype=dtype, device=device)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    peft_model.eval()
    kw = {"device": device, "max_new_tokens": max_new_tokens, "batch_size": batch_size}
    unmerged = greedy_token_ids(peft_model, tok, prompts, **kw)
    merged = greedy_token_ids(peft_model.merge_and_unload(), tok, prompts, **kw)
    divergence = [_first_divergence(a.ids, b.ids) for a, b in zip(unmerged, merged, strict=True)]
    n_match = sum(d is None for d in divergence)
    return {
        "adapter_hash": adapter_hash(adapter_dir), "dtype": dtype, "n": len(prompts), "n_match": n_match,
        "match_share": n_match / len(prompts) if prompts else 0.0, "all_match": n_match == len(prompts),
        "first_divergence": divergence,
    }


def _first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
