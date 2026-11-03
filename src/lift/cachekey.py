"""Cache keys for generations.

A key covers everything that can change an output: model revision, adapter hash, the full
intervention config (method, lambda, alpha, strength, threshold, layer, artifact hashes), the chat
template, the generation config, and the prompt (id, language and the exact text). The prompt text
hash is there so a fixed translation can never reuse a stale output. A key is never shared across
interventions because the intervention digest is part of it.
"""

from __future__ import annotations

from typing import Any

from lift.io import sha256_obj, sha256_text
from lift.schemas import VariantSpec

KEY_VERSION = 1


def intervention_config(spec: VariantSpec) -> dict[str, Any]:
    """The parts of a variant that change the model's behaviour. The variant name is left out."""
    return {
        "method": spec.method,
        "adapter": spec.adapter,
        "seed": spec.seed,
        "params": spec.params,
        "artifacts": spec.artifacts,
    }


def intervention_digest(spec: VariantSpec) -> str:
    return sha256_obj(intervention_config(spec))


def generation_key(
    *,
    model_rev: str,
    adapter_hash: str | None,
    spec: VariantSpec,
    template_hash: str,
    gen_cfg: dict[str, Any],
    pid: str,
    lang: str,
    prompt_text: str,
) -> str:
    if not model_rev or model_rev == "main":
        raise ValueError("model_rev must be a resolved commit sha, not a branch name")
    if spec.method != "stock" and not adapter_hash:
        raise ValueError(f"variant {spec.name} needs an adapter hash")
    payload = {
        "v": KEY_VERSION,
        "model_rev": model_rev,
        "adapter_hash": adapter_hash,
        "intervention": intervention_digest(spec),
        "template_hash": template_hash,
        "gen_cfg": gen_cfg,
        "pid": pid,
        "lang": lang,
        "prompt_sha": sha256_text(prompt_text),
    }
    return sha256_obj(payload)
