"""Build any variant from its VariantSpec at load time. P02 does stock and adapter; P04 adds the
repairs. Every variant goes through this one function so every number comes from one code path.

    stock        Global
    adapter      Global + LoRA merged (lift.train.merge)
    wiseft       lift.repair.wiseft.make(adapter_model, stock_model, lam)
    resta        lift.repair.resta.make(adapter_model, base_cpu, stock_cpu, alpha)   (dare=False)
    cast         lift.repair.cast.make(adapter_model, artifacts, strength, gate=True)
    cast_always  same with the gate always on (plain steering)

Weight-space repairs mutate in place: always start from freshly loaded adapter weights, never from
a model a previous repair touched. Keep Base and Global copies on CPU and stream.
"""

from __future__ import annotations

from typing import Any

from lift.schemas import VariantSpec


class LoadedVariant:
    """What the generator needs: ``model`` (or a CAST wrapper with ``.generate``), ``tokenizer``,
    and the hashes that go into the cache key."""

    model: Any
    tokenizer: Any
    spec: VariantSpec
    model_rev: str
    adapter_hash: str | None
    template_hash: str


def build_variant(spec: VariantSpec, *, device: str = "cuda:0", dtype: str = "fp16") -> LoadedVariant:
    raise NotImplementedError("P02 (stock, adapter), P04 (repairs)")


def load_registry() -> dict[str, VariantSpec]:
    """configs/variants.yaml -> {name: VariantSpec}."""
    raise NotImplementedError("P02")
