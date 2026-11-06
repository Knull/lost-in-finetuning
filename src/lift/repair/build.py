"""The repair builders of lift.models.variants' dispatch table (P05): wiseft, resta, cast, cast_always.

Registered on import of lift.models.variants (its last lines import this module), so ``build_variant``
is still the one path every repaired variant is built through (CLAUDE.md rule 2).

Every builder starts like the adapter's: ``checked_adapter`` (the hash the registry pins, the recorded
merge parity), the template check, then a FRESH merge of the adapter on the run's device and dtype: the
weight-space repairs mutate in place, so no setting ever starts from a model another one touched. The
reference models of the weight-space repairs (stock Global, and Base for ReSta) are loaded on the CPU in
the run's dtype and freed once applied: on one T4 (16 GB) only the repaired model sits on the GPU.

- wiseft: params {"lambda"}; ``lift.repair.wiseft.make(model, stock_cpu, lambda)``.
- resta: params {"alpha", "dare", "base", "base_revision"}; Base loaded at exactly the revision the entry
  names (a local directory's content hash in tests), ``lift.repair.resta.make``.
- cast / cast_always: params {"strength", ...}, artifacts {"adapter", "cast"}; the artifact must hash to
  the pinned value and have been fitted on this adapter hash and Global revision. A wrapper variant:
  ``LoadedVariant.model`` is the wrapper (``generate``), ``base`` the merged model, ``batch_meta`` the
  gate's per-row record and ``score_logits`` the gated scorer lift.models.loglik calls.
"""

from __future__ import annotations

import gc
from typing import Any

from lift.models.variants import BuildContext, LoadedVariant, builder, checked_adapter
from lift.schemas import VariantSpec


def _param(spec: VariantSpec, key: str) -> float:
    if key not in spec.params:
        raise ValueError(f"{spec.name}: a {spec.method} variant needs params.{key}")
    return float(spec.params[key])


def _merged(spec: VariantSpec, ctx: BuildContext) -> tuple[Any, Any, str]:
    """(fresh merged adapter on ctx.device, tokenizer, adapter hash)."""
    from lift.models.load import load_tokenizer
    from lift.train.merge import check_adapter_template, merged_model

    directory, digest = checked_adapter(spec)
    tok = load_tokenizer(ctx.model_id, ctx.model_rev)
    check_adapter_template(directory, tok)
    model = merged_model(directory, device=ctx.device, dtype=ctx.dtype, base_id=ctx.model_id,
                         base_revision=ctx.model_rev)
    return model, tok, digest


def _loaded(spec: VariantSpec, ctx: BuildContext, model: Any, tok: Any, digest: str,
            **extra: Any) -> LoadedVariant:
    from lift.models.load import template_hash

    return LoadedVariant(model=model, tokenizer=tok, spec=spec, model_rev=ctx.model_rev, adapter_hash=digest,
                         template_hash=template_hash(tok), dtype=ctx.dtype, device=ctx.device, **extra)


def _free(*objs: Any) -> None:
    del objs
    gc.collect()


@builder("wiseft")
def _build_wiseft(spec: VariantSpec, ctx: BuildContext) -> LoadedVariant:
    from lift.models.load import load_model
    from lift.repair import wiseft

    lam = _param(spec, "lambda")
    wiseft.alpha_for(lam)                                  # range check before any load
    model, tok, digest = _merged(spec, ctx)
    stock_cpu = load_model(ctx.model_id, ctx.model_rev, dtype=ctx.dtype, device="cpu")
    wiseft.make(model, stock_cpu, lam)
    del stock_cpu
    _free()
    return _loaded(spec, ctx, model, tok, digest)


@builder("resta")
def _build_resta(spec: VariantSpec, ctx: BuildContext) -> LoadedVariant:
    from lift.models.load import load_model, resolved_revision
    from lift.repair import resta

    alpha = _param(spec, "alpha")
    if spec.params.get("dare") is not False:
        raise ValueError(f"{spec.name}: params.dare must be false (DARE is never enabled)")
    base_id, base_rev = spec.params.get("base"), spec.params.get("base_revision")
    if not base_id or not base_rev:
        raise ValueError(f"{spec.name}: a resta variant names its Base model (params.base, "
                         "params.base_revision)")
    resolved = resolved_revision(base_id, base_rev)
    if resolved != base_rev:
        raise ValueError(f"{spec.name}: Base {base_id} resolves to {resolved}, not the {base_rev} its entry "
                         "names (a local Base directory changed?)")
    model, tok, digest = _merged(spec, ctx)
    base_cpu = load_model(base_id, base_rev, dtype=ctx.dtype, device="cpu")
    stock_cpu = load_model(ctx.model_id, ctx.model_rev, dtype=ctx.dtype, device="cpu")
    resta.make(model, base_cpu, stock_cpu, alpha)
    del base_cpu, stock_cpu
    _free()
    return _loaded(spec, ctx, model, tok, digest)


def _build_cast(spec: VariantSpec, ctx: BuildContext, *, gate: bool) -> LoadedVariant:
    from lift.io import sha256_file
    from lift.repair import cast

    strength = _param(spec, "strength")
    pinned = spec.artifacts.get("cast")
    if not pinned:
        raise ValueError(f"{spec.name}: a {spec.method} variant pins its CAST artifact (artifacts.cast)")
    path = cast.artifact_path(spec.adapter or "")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `lift fit-cast --adapter {spec.adapter}`, or `lift "
                                "sync "
                                "pull` to fetch it")
    if sha256_file(path) != pinned:
        raise ValueError(f"{spec.name}: {path} does not match the artifact hash pinned in "
                         "configs/variants.yaml "
                         "(fitted again since registration?)")
    model, tok, digest = _merged(spec, ctx)
    _cond, _vecs, info = cast.load(path)
    wrong = {k: (info.get(k), v) for k, v in (("adapter_hash", digest), ("model_rev", ctx.model_rev))
             if info.get(k) != v}
    if wrong:
        raise ValueError(f"{spec.name}: {path} was fitted on another model ({wrong}, artifact vs now)")
    held = {"layer": info["condition_layer"], "threshold": info["threshold"],
            "comparator": info["comparator"]}
    for key in held:
        if key in spec.params and spec.params[key] != held[key]:
            raise ValueError(f"{spec.name}: params.{key} {spec.params[key]!r} differs from the artifact's "
                             f"{held[key]!r}")
    wrapper = cast.make(model, path, strength, gate=gate, pad_id=tok.pad_token_id)
    return _loaded(spec, ctx, wrapper, tok, digest, base=model, batch_meta=wrapper.batch_meta,
                   score_logits=wrapper.score_logits)


@builder("cast")
def _build_cast_gated(spec: VariantSpec, ctx: BuildContext) -> LoadedVariant:
    return _build_cast(spec, ctx, gate=True)


@builder("cast_always")
def _build_cast_always(spec: VariantSpec, ctx: BuildContext) -> LoadedVariant:
    return _build_cast(spec, ctx, gate=False)
