"""CAST: conditional activation steering with one global threshold. Implemented in P05.

SafeTune 0.1.12 (all badged Faithful):
- ``fit_cast_condition(model, harmful_prompts, benign_prompts, tokenizer, candidate_layers=None,
  pool="last", chat_template=None, device=...)`` grid-searches (layer, threshold, comparator) by
  held-out F1 and returns a CASTCondition. ``chat_template=None`` means True (fit on the same
  chat-formatted prompts the model sees at inference). Do not set ``legacy_cast_gate``.
- ``extract_caa_vectors(model, tokenizer, positive_prompts, negative_prompts, config)`` gives the
  per-layer steering vectors (difference of means).
- ``CASTModel(model, steering_vectors, condition=cond, alpha=strength)``; ``cast.generate(...)``
  gates each prompt of a batch on its own (per_prompt_gate defaults to True). ``with cast:``
  installs the vectors unconditionally: that is the gate-always-on setting.

Decisions (docs/decisions.md): the condition and the vectors are fitted per seed on the adapter
being repaired, from train/fit data only (PolyRefuse harmful train vs harmless train + the 100
OR-Bench fit prompts, all 6 languages pooled). Strengths {0.5, 1.0, 2.0} plus always-on at 1.0.

How it is used here (decisions.md item 66):
- Condition: ``fit_cast_condition`` with its default candidate layers (n/4 to 3n/4), ``pool`` from
  configs/repairs.yaml (``last``), its default threshold grid and held-out half, chat template on.
- Steering vectors: ``extract_caa_vectors(model, tok, harmful, benign, CAAConfig(target_layers=...))``,
  so each vector is mean(harmful) - mean(benign) at the last prompt token (CAA's paper-faithful
  pooling, not normalised), the direction that adds refusal. The layers are SafeTune CASTTrainer's own
  default band, layers 14-18 of a 32-layer model scaled to the model's depth (``behavior_layers``:
  16-20 of Tiny Aya's 36), stored in configs/repairs.yaml ``cast.behavior_band_of_32``.
- Artifact: one ``cast.safetensors`` per adapter in outputs/artifacts/cast/{adapter}/ holding the
  condition vector and the steering vectors (float32), with the layer, threshold (exact repr),
  comparator, pool, held-out F1 and the provenance (adapter hash, model revision, dtype, template, fit
  prompt digests) in its metadata. Safetensors, not torch.save: no pickle, and the same tensors and
  metadata always give the same bytes, so the file's sha256 is the artifact hash that goes into the
  VariantSpec (``artifacts.cast``) and so into every cache key. ``fit.json`` beside it is the report
  (counts, F1, per-language firing rates, SafeTune's runtime config): not hashed.
- The gate. ``GatedCAST`` is SafeTune's CASTModel with one change to the prefill gate pass: it runs
  with position ids counted from each row's first real token (as generation and lift.models.loglik
  count them; the fit read each prompt alone, unpadded, which is the same) and ``logits_to_keep=1``
  (the pass reads only the condition layer's hidden states; full logits for a padded batch cost
  gigabytes on Tiny Aya's 262k vocabulary). It records each row's decision, similarity and the pass's
  time (``gate_fired``, ``gate_sim``, ``gate_seconds`` per row) for GenerationRecord.meta: the gate's
  prefill is measured apart from the rest of generation.
- Log-likelihood scoring (Belebele) decides the gate per row on that row's prompt (the Belebele
  context), exactly as at generation time, then scores the row with the vectors on the rows whose gate
  fired (``score_logits``, which lift.models.loglik calls).
- Always on (``cast_always``): the same artifact under ``with cast:``, every row steered.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lift.paths import artifacts_dir

ARTIFACT = "cast.safetensors"
REPORT = "fit.json"
FORMAT = "lift-cast-v1"
REF_DEPTH = 32                # SafeTune's steering layer defaults are set for 32-layer models


def artifact_dir(adapter: str) -> Path:
    return artifacts_dir() / "cast" / adapter


def artifact_path(adapter: str) -> Path:
    return artifact_dir(adapter) / ARTIFACT


def behavior_layers(n_layers: int, band: Sequence[int] | None = None) -> list[int]:
    """SafeTune CASTTrainer's default steering band (``runner.steer._activation._default_layers``): the
    32-layer band ``band`` (configs/repairs.yaml cast.behavior_band_of_32, [14, 18]) at the same relative
    depth, both end points scaled and rounded."""
    from lift import config

    lo32, hi32 = band if band is not None else config.get("cast.behavior_band_of_32", name="repairs")
    n = int(n_layers)
    if n < 1:
        raise ValueError("the model has no decoder layers")
    lo, hi = (min(n - 1, round(i * n / REF_DEPTH)) for i in (lo32, hi32))
    return list(range(lo, hi + 1))


def heldout_indices(n: int, val_frac: float = 0.5) -> list[int]:
    """The held-out indices ``fit_cast_condition`` scores its F1 on, for one class of ``n`` prompts: its
    deterministic interleaved split (SafeTune 0.1.12, ``fit_cast_condition._split``), reproduced so the
    firing-rate diagnostic reads exactly the half the threshold was chosen on (the condition vector was
    fitted on the other half)."""
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_idx = list(range(0, n, max(1, n // max(1, n_val))))[:n_val]
    return val_idx or list(range(n))


def _num_layers(model: Any) -> int:
    from safetune._refusal_helpers import _get_decoder_layers

    return len(_get_decoder_layers(model))


def _check_runtime() -> dict[str, Any]:
    """SafeTune's runtime config, refusing the legacy gate (raw prompts, first-prompt-decides)."""
    import dataclasses

    from safetune.config import get_config

    cfg = get_config()
    if getattr(cfg, "legacy_cast_gate", False):
        raise ValueError("safetune.configure(legacy_cast_gate=True) is set: CAST must fit and gate on "
                         "chat-formatted prompts, each prompt on its own")
    out = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(vars(cfg))
    return json.loads(json.dumps(out, default=str))


def fit(model: Any, tokenizer: Any, harmful: list[str], benign: list[str], *, out_dir: Path,
        meta: Mapping[str, Any] | None = None, device: str = "cpu") -> Path:
    """Fit the condition and the CAA vectors; save them; return the artifact path.

    ``meta`` (strings, numbers, lists) is the provenance written into the artifact's metadata: what the
    model and the fit prompts were. Nothing time-dependent goes there, so the same fit gives the same
    bytes."""
    from safetune.interventions.steer.caa import CAAConfig, extract_caa_vectors
    from safetune.interventions.steer.cast import fit_cast_condition

    from lift import config

    if not harmful or not benign:
        raise ValueError("CAST needs harmful and benign fit prompts")
    _check_runtime()
    pool = config.get("cast.fit.pool", name="repairs")
    layers = behavior_layers(_num_layers(model))
    condition = fit_cast_condition(model, list(harmful), list(benign), tokenizer, pool=pool,
                                   device=str(device))
    vectors = extract_caa_vectors(model, tokenizer, list(harmful), list(benign),
                                  CAAConfig(target_layers=layers))
    if sorted(int(k) for k in vectors) != layers:
        raise RuntimeError(f"extract_caa_vectors returned layers {sorted(vectors)}, not {layers}")
    info = {
        "format": FORMAT, "condition_layer": int(condition.condition_layer),
        "threshold": float(condition.threshold), "comparator": str(condition.comparator),
        "pool": str(condition.pool), "f1_heldout": float(condition.f1), "behavior_layers": layers,
        "n_harmful": len(harmful), "n_benign": len(benign), **dict(meta or {}),
    }
    return save(Path(out_dir) / ARTIFACT, condition.condition_vector, vectors, info)


def save(path: Path, condition_vector: Any, vectors: Mapping[int, Any], info: Mapping[str, Any]) -> Path:
    """Write the artifact (float32 tensors, ``info`` as JSON in the metadata) atomically."""
    import os
    import tempfile

    from safetensors.torch import save_file

    tensors = {"condition_vector": condition_vector.detach().float().cpu().contiguous()}
    for layer, vec in sorted(vectors.items()):
        tensors[f"steering.{int(layer)}"] = vec.detach().float().cpu().contiguous()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        save_file(tensors, tmp, metadata={"lift": json.dumps(dict(info), sort_keys=True)})
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def artifact_hash(path: Path) -> str:
    from lift.io import sha256_file

    return sha256_file(Path(path))


def load(path: Path) -> tuple[Any, dict[int, Any], dict[str, Any]]:
    """(CASTCondition, {layer: steering vector}, info) of an artifact."""
    from safetensors import safe_open
    from safetune.interventions.steer.cast import CASTCondition

    with safe_open(str(path), framework="pt") as f:
        info = json.loads((f.metadata() or {})["lift"])
        if info.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} artifact")
        cv = f.get_tensor("condition_vector")
        vectors = {int(k.split(".", 1)[1]): f.get_tensor(k) for k in f.keys() if k.startswith("steering.")}
    if sorted(vectors) != list(info["behavior_layers"]):
        raise ValueError(f"{path}: steering layers {sorted(vectors)} differ from its metadata")
    cond = CASTCondition(condition_vector=cv, condition_layer=int(info["condition_layer"]),
                         threshold=float(info["threshold"]), comparator=info["comparator"],
                         f1=float(info["f1_heldout"]), pool=info["pool"])
    return cond, vectors, info


def _sync(device: Any) -> None:
    import torch

    if getattr(device, "type", str(device)).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _left_pad(rows: Sequence[Sequence[int]], pad_id: int, device: Any) -> tuple[Any, Any]:
    import torch

    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r):] = torch.tensor(list(r), dtype=torch.long)
        mask[i, width - len(r):] = 1
    return ids.to(device), mask.to(device)


def _gated_class() -> type:
    """GatedCAST, built on first use so this module imports without SafeTune and torch."""
    global _GATED
    if _GATED is not None:
        return _GATED
    from safetune.interventions.steer.cast import CASTModel

    class GatedCAST(CASTModel):
        """SafeTune's CASTModel, its gate pass run with real position ids and the last position's logits
        only, recording each row's decision for ``batch_meta`` (module docstring). Generation is
        CASTModel.generate itself."""

        def __init__(self, model: Any, vectors: Mapping[int, Any], condition: Any, *, alpha: float,
                     pad_id: int) -> None:
            super().__init__(model, dict(vectors), condition=condition, alpha=alpha, per_prompt_gate=True)
            device = next(model.parameters()).device
            # Moved once (CASTModel's hook moves a CPU vector at every layer of every decoding step).
            self.steering_vectors = {k: v.float().to(device) for k, v in self.steering_vectors.items()}
            self.pad_id = int(pad_id)
            self.last_meta: list[dict[str, Any]] = []

        def _gate_fires(self, input_ids: Any, **kwargs: Any) -> Any:
            mask = kwargs.get("attention_mask")
            if mask is not None and "position_ids" not in kwargs:
                kwargs["position_ids"] = (mask.long().cumsum(-1) - 1).clamp(min=0)
            kwargs.setdefault("logits_to_keep", 1)
            kwargs.setdefault("use_cache", False)
            _sync(input_ids.device)
            t0 = time.perf_counter()
            fires, sims = super()._gate_fires(input_ids, **kwargs)
            _sync(input_ids.device)
            seconds = time.perf_counter() - t0
            if isinstance(fires, bool):          # never with per_prompt_gate=True; kept explicit
                fires, sims = [fires], [sims]
            n = len(fires)
            self.last_meta = [{"gate_fired": bool(f), "gate_sim": float(s), "gate_seconds": seconds / n}
                              for f, s in zip(fires, sims, strict=True)]
            return list(fires), list(sims)

        def batch_meta(self) -> list[dict[str, Any]]:
            return [dict(m) for m in self.last_meta]

        def score_logits(self, input_ids: Any, attention_mask: Any, *, position_ids: Any,
                         prompt_lengths: Sequence[int], logits_to_keep: int) -> Any:
            """Logits of left-padded rows, each gated on its own prompt (its first ``prompt_lengths[i]``
            real tokens), as at generation time. Rows sharing a prompt share one gate decision."""
            width = int(input_ids.shape[1])
            prompts = []
            for i, n in enumerate(prompt_lengths):
                start = width - int(attention_mask[i].sum())
                prompts.append(tuple(input_ids[i, start:start + int(n)].tolist()))
            unique = list(dict.fromkeys(prompts))
            ids, mask = _left_pad(unique, self.pad_id, input_ids.device)
            fires, _sims = self._gate_fires(ids, attention_mask=mask)
            per_prompt = dict(zip(unique, self.last_meta, strict=True))
            share = sum(m["gate_seconds"] for m in self.last_meta) / len(prompts)
            self.last_meta = [{**per_prompt[p], "gate_seconds": share} for p in prompts]
            row_fires = [m["gate_fired"] for m in self.last_meta]
            out = self._steered(row_fires, self.model, input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, logits_to_keep=logits_to_keep, use_cache=False)
            return out.logits

    _GATED = GatedCAST
    return GatedCAST


_GATED: type | None = None


class AlwaysOnCAST:
    """The gate always on: CASTModel's vectors installed for every call (``with cast:``), the plain-steering
    ablation. The same ``generate`` / ``score_logits`` / ``batch_meta`` interface as the gated wrapper."""

    def __init__(self, cast: Any) -> None:
        self.cast = cast
        self.model = cast.model
        self._n = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        ids = kwargs.get("input_ids", args[0] if args else None)
        self._n = int(ids.shape[0]) if ids is not None else 0
        with self.cast:
            return self.model.generate(*args, **kwargs)

    def score_logits(self, input_ids: Any, attention_mask: Any, *, position_ids: Any,
                     prompt_lengths: Sequence[int], logits_to_keep: int) -> Any:
        self._n = int(input_ids.shape[0])
        with self.cast:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                              logits_to_keep=logits_to_keep, use_cache=False).logits

    def batch_meta(self) -> list[dict[str, Any]]:
        return [{"gate_fired": True, "gate_always_on": True} for _ in range(self._n)]


def make(model: Any, artifact_path: Path, strength: float, *, gate: bool = True, pad_id: int = 0) -> Any:
    """A CASTModel (gate on) or an always-on wrapper with the same ``generate`` interface."""
    condition, vectors, _info = load(artifact_path)
    _check_runtime()
    strength = float(strength)
    if strength <= 0:
        raise ValueError(f"CAST strength must be positive, got {strength}")
    cast = _gated_class()(model, vectors, condition, alpha=strength, pad_id=pad_id)
    return cast if gate else AlwaysOnCAST(cast)


def firing(cast: Any, tokenizer: Any, prompts: Sequence[str], *,
           batch_size: int = 16) -> list[dict[str, Any]]:
    """The gate's decision on each prompt (chat-formatted as generation formats it), through the wrapper's
    own gate pass: [{"gate_fired", "gate_sim", "gate_seconds"}]."""
    from lift.models.load import encode_chat

    device = next(cast.model.parameters()).device
    out: list[dict[str, Any]] = []
    for i in range(0, len(prompts), batch_size):
        rows = [encode_chat(tokenizer, p) for p in prompts[i:i + batch_size]]
        ids, mask = _left_pad(rows, tokenizer.pad_token_id, device)
        cast._gate_fires(ids, attention_mask=mask)
        out.extend(cast.batch_meta())
    return out
