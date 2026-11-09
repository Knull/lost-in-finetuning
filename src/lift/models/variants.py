"""Build any variant from its VariantSpec at load time. P02 does stock and adapter; P05 adds the
repairs. Every variant goes through this one function so every number comes from one code path.

    stock        Global
    adapter      Global + LoRA merged (lift.train.merge)
    wiseft       lift.repair.wiseft.make(adapter_model, stock_model, lam)
    resta        lift.repair.resta.make(adapter_model, base_cpu, stock_cpu, alpha)   (dare=False)
    cast         lift.repair.cast.make(adapter_model, artifacts, strength, gate=True)
    cast_always  same with the gate always on (plain steering)

Weight-space repairs mutate in place: always start from freshly loaded adapter weights, never from
a model a previous repair touched. Keep Base and Global copies on CPU and stream.

P05 adds a repair by writing a builder ``(spec, ctx) -> LoadedVariant`` and registering it in
``BUILDERS`` under its method (``@builder("wiseft")``); stock and adapter code stays untouched. The repair
builders live in lift.repair.build, imported at the end of this module so the table is always complete.

configs/variants.yaml holds one entry per variant: the VariantSpec fields (name, method, adapter,
seed, params, artifacts) plus the documentation-only keys in ``REGISTRY_ONLY_KEYS``. An adapter's
files live in outputs/adapters/{adapter}/ (synced through the runs repo); when the entry pins
``artifacts.adapter``, the value is ``lift.train.merge.adapter_hash(dir)`` (weights plus the
behaviour fields of adapter_config.json, not the bare file sha256) and the loaded adapter must match.
A cell of a variant that names an adapter (the adapter method, every repair) is reported only when its
entry pins that hash, an entry of method adapter names that adapter, and every such entry pins the same
hash (lift register-adapter writes the adapter's own entry, and a repair's entry copies its pin;
lift.judge.pipeline refuses the cell otherwise).

A wrapper variant (CAST) sets ``LoadedVariant.base`` to the HF model it wraps; the harness reads
config, generation_config and the stop set from ``hf_model`` and calls ``model.generate``. That must pass
the ``logits_processor`` it is handed on to the model's own generate, the check that decoding stays finite
(lift.models.generate._NonFiniteWatch): a wrapper that never calls it is refused, as is one that returns
another number of sequences than the batch has rows. It may decode the batch in pieces (CAST's per-prompt
gate), passing the processor on to every piece, the pieces holding each row once: pieces whose batch sizes
do not add up to the batch's rows are refused. The check then cannot tell the rows apart, so a NaN or +inf
score in any piece marks every row, and generate refuses that batch whole.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import adapters_dir
from lift.schemas import VariantSpec

REGISTRY_ONLY_KEYS = frozenset({"location", "note"})


@dataclass
class LoadedVariant:
    """What the generator needs: ``model`` (or a CAST wrapper with ``.generate``), ``tokenizer``,
    and the hashes that go into the cache key."""

    model: Any
    tokenizer: Any
    spec: VariantSpec
    model_rev: str
    adapter_hash: str | None
    template_hash: str
    dtype: str
    device: str
    # CAST wrappers: per-row meta (e.g. {"gate_fired": True}) for the last generate call.
    batch_meta: Callable[[], list[dict]] | None = None
    # The HF model under a wrapper (CAST). config, generation_config and the stop set are read from it,
    # so a wrapper that only exposes .generate stops on exactly the tokens stock and adapter stop on.
    base: Any = None
    # Log-likelihood scoring of a wrapper variant (lift.models.loglik: Belebele, held-out loss). A CAST
    # builder (P05) sets it to ``fn(input_ids, attention_mask, *, position_ids, prompt_lengths,
    # logits_to_keep) -> logits``, the gate decided per row on that row's prompt (its first
    # ``prompt_lengths[i]`` real tokens) exactly as at generation time, with ``batch_meta()`` giving each
    # row's decision afterwards. None for a plain model, which loglik calls directly; loglik refuses a
    # wrapper (``base`` set) without it, so a gated variant is never scored ungated.
    score_logits: Callable[..., Any] | None = None
    # What the builder loaded, by the key spec.artifacts names it under: the sha256 of each file (the
    # adapter's hash, a CAST artifact's) and, for ReSta, the Base revision. build_variant refuses a variant
    # whose declared ``spec.artifacts`` this does not match (R1-05).
    loaded: dict[str, str] = field(default_factory=dict)

    @property
    def hf_model(self) -> Any:
        return self.base if self.base is not None else self.model


@dataclass(frozen=True)
class BuildContext:
    model_id: str         # Global's id (or a local directory in tests)
    model_rev: str        # resolved commit sha
    device: str
    dtype: str


Builder = Callable[[VariantSpec, BuildContext], LoadedVariant]
BUILDERS: dict[str, Builder] = {}


def builder(method: str) -> Callable[[Builder], Builder]:
    def register(fn: Builder) -> Builder:
        if method in BUILDERS:
            raise ValueError(f"a builder for {method!r} is already registered")
        BUILDERS[method] = fn
        return fn
    return register


def adapter_path(adapter: str) -> Path:
    return adapters_dir() / adapter


def build_variant(spec: VariantSpec, *, device: str = "cuda:0", dtype: str = "bf16",
                  model_id: str | None = None, revision: str | None = None) -> LoadedVariant:
    """Load ``spec`` on ``device`` in ``dtype``. ``model_id``/``revision`` default to Global from
    configs/project.yaml (tests pass a local tiny model)."""
    from lift.models.load import resolved_revision

    make = BUILDERS.get(spec.method)
    if make is None:
        raise NotImplementedError(f"no builder for method {spec.method!r} yet (repairs arrive in P05)")
    model_id = model_id or config.get("models.global.id")
    rev = resolved_revision(model_id, revision or config.get("models.global.revision"))
    variant = make(spec, BuildContext(model_id=model_id, model_rev=rev, device=device, dtype=dtype))
    check_loaded(spec, variant.loaded)
    return variant


def check_loaded(spec: VariantSpec, loaded: dict[str, str]) -> None:
    """Refuse a variant whose builder loaded other files than its registry entry declares (R1-05): each
    ``spec.artifacts`` hash must be the one the builder loaded under that key. A declared file the builder
    never loaded is refused too, as a typo in the entry would otherwise key cells on a hash nothing
    checked."""
    wrong = {key: (want, loaded.get(key)) for key, want in spec.artifacts.items() if loaded.get(key) != want}
    if wrong:
        shown = "; ".join(f"artifacts.{k}: declared {str(w)[:12]}, loaded "
                          f"{str(g)[:12] if g else 'nothing under that key'}"
                          for k, (w, g) in sorted(wrong.items()))
        raise ValueError(f"{spec.name}: the builder loaded other files than configs/variants.yaml declares "
                         f"({shown}). Register the variant again from the files it should load, then make "
                         "its cells again.")


@builder("stock")
def _build_stock(spec: VariantSpec, ctx: BuildContext) -> LoadedVariant:
    from lift.models.load import load_model, load_tokenizer, template_hash

    if spec.adapter or spec.params:
        raise ValueError(f"{spec.name}: a stock variant takes no adapter or params")
    tok = load_tokenizer(ctx.model_id, ctx.model_rev)
    model = load_model(ctx.model_id, ctx.model_rev, dtype=ctx.dtype, device=ctx.device)
    return LoadedVariant(model=model, tokenizer=tok, spec=spec, model_rev=ctx.model_rev, adapter_hash=None,
                         template_hash=template_hash(tok), dtype=ctx.dtype, device=ctx.device)


@builder("adapter")
def _build_adapter(spec: VariantSpec, ctx: BuildContext) -> LoadedVariant:
    from lift.models.load import load_tokenizer, template_hash
    from lift.train.merge import check_adapter_template, merged_model

    directory, digest = checked_adapter(spec)
    tok = load_tokenizer(ctx.model_id, ctx.model_rev)
    check_adapter_template(directory, tok)
    model = merged_model(directory, device=ctx.device, dtype=ctx.dtype, base_id=ctx.model_id,
                         base_revision=ctx.model_rev)
    return LoadedVariant(model=model, tokenizer=tok, spec=spec, model_rev=ctx.model_rev, adapter_hash=digest,
                         template_hash=template_hash(tok), dtype=ctx.dtype, device=ctx.device,
                         loaded={"adapter": digest})


def checked_adapter(spec: VariantSpec) -> tuple[Path, str]:
    """The adapter's directory and hash, after checking the hash pinned in the registry (if any) and
    the merge parity lift train recorded. An adapter whose registration.json (synced with it) records a
    failed fp32 merge parity for this very hash must not be evaluated, whatever a registry holds
    (CLAUDE.md rule 9; lift.train.lora.recorded_parity_failure): an fp16 parity on a T4 may have
    registered it before the fp32 one failed, and this is the one path every adapter variant is built
    through. Without a registration.json (the tests' adapters) only the pinned hash is checked. Repair
    builders (P05) start from this too."""
    from lift.train.lora import REGISTRATION, recorded_parity_failure
    from lift.train.merge import adapter_hash

    if not spec.adapter:
        raise ValueError(f"{spec.name}: method {spec.method} needs an adapter name")
    directory = adapter_path(spec.adapter)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} not found; run `lift sync pull` to fetch adapters")
    digest = adapter_hash(directory)
    pinned = spec.artifacts.get("adapter")
    if pinned and pinned != digest:
        raise ValueError(f"{spec.name}: {directory} does not match the adapter hash pinned in "
                         "configs/variants.yaml; the files changed after registration")
    failure = recorded_parity_failure(directory, digest)
    if failure:
        raise ValueError(f"{spec.name}: adapter {spec.adapter} (hash {digest[:12]}) failed fp32 merge "
                         "parity: it must not be evaluated (CLAUDE.md rule 9). "
                         f"{directory / REGISTRATION} records {failure}. Find the cause before anything "
                         f"evaluates it; `lift register-adapter {spec.adapter}` on the laptop (after "
                         "`lift sync pull`) removes its configs/variants.yaml entry")
    return directory, digest


def load_registry() -> dict[str, VariantSpec]:
    """configs/variants.yaml -> {name: VariantSpec}."""
    doc = config.load("variants")
    out: dict[str, VariantSpec] = {}
    for entry in doc.get("variants") or []:
        fields = {k: v for k, v in dict(entry).items() if k not in REGISTRY_ONLY_KEYS}
        spec = VariantSpec.model_validate(fields)
        if spec.name in out:
            raise ValueError(f"configs/variants.yaml lists {spec.name!r} twice")
        out[spec.name] = spec
    return out


def get_spec(name: str) -> VariantSpec:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"variant {name!r} is not in configs/variants.yaml (known: {sorted(registry)})")
    return registry[name]


# The repair builders (wiseft, resta, cast, cast_always) register themselves in BUILDERS (P05).
from lift.repair import build as _repair_builders  # noqa: E402, F401
