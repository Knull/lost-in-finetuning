"""``lift publish model``: the repaired model, the repair parameters and the gated adapters.

- Repaired model (public): the frozen winner on seed 1 (selection.json). WiSE-FT and ReSta are saved
  as merged weights; CAST as the adapter-merged model it wraps, plus its artifacts and
  ``cast_generate.py``, a small wrapper around ``lift.repair.cast.make``. The weights come from
  ``lift.models.variants.build_variant`` (the one code path), materialised by ``--build`` on a machine
  with the GPU stack and room for the model (the Colab VM: WiSE-FT and ReSta hold two or three copies).
- Repair parameters (public): every frozen setting's VariantSpec (lambda, alpha or strength, artifact
  hashes) and the CAST artifacts from outputs/artifacts/cast/{adapter}/.
- Adapters (gated, automatic approval): every registered adapter in outputs/adapters/, its weights
  and adapter_config.json only (registration.json stays private: it names the runs repo).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lift import config, paths
from lift.publish import PublishError, render, settings
from lift.publish.hub import Upload
from lift.publish.inputs import (
    P05,
    Facts,
    frozen_family_winners,
    frozen_winner,
    pending,
    selection_path,
    shown,
)

CAST_METHODS = ("cast", "cast_always")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
ARTIFACT_SUFFIXES = {".pt", ".json", ".safetensors", ".yaml", ".npy", ".npz", ".txt"}
METHOD_TEXT = {
    "wiseft": "WiSE-FT attenuation: Global + lambda x (adapter - Global), merged into the weights",
    "resta": "ReSta-A: adapter + alpha x (Global - Base), merged into the weights (DARE off)",
    "cast": "CAST: the adapter-merged model plus a refusal steering vector, added only when a harm "
    "condition clears one global threshold",
    "cast_always": "plain activation steering: the adapter-merged model with the refusal vector always on",
    "adapter": "the unrepaired adapter, merged",
}


def stage_root() -> Path:
    return paths.output_dir() / "publish"


def _spec(name: str) -> Any | None:
    from lift.models.variants import get_spec

    try:
        return get_spec(name)
    except KeyError:
        return None


def frozen_names(facts: Facts) -> set[str]:
    sel = facts.selection() or {}
    return {*frozen_family_winners(facts).values(), *(sel.get("frozen") or [])}


def params_doc(spec: Any) -> dict[str, Any]:
    rules = config.load("repairs").get("selection", {})
    return {
        "variant": spec.name,
        "method": spec.method,
        "adapter": spec.adapter,
        "seed": spec.seed,
        "params": dict(spec.params),
        "artifacts": dict(spec.artifacts),
        "base_model": config.get("models.global"),
        "reference_model": config.get("models.base"),
        "selection_rule": rules,
        "frozen_by": shown(selection_path()),
    }


def cast_artifacts(adapter: str) -> Path:
    return paths.artifacts_dir() / "cast" / adapter


def _copy_artifacts(src: Path, dest: Path) -> list[str]:
    out = []
    for f in sorted(src.rglob("*")) if src.is_dir() else []:
        if f.is_file() and f.suffix in ARTIFACT_SUFFIXES:
            target = dest / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
            out.append(target.relative_to(dest).as_posix())
    return out


def weights_present(folder: Path) -> bool:
    return any(folder.glob("*.safetensors")) and (folder / "config.json").is_file()


def materialise(
    spec: Any,
    folder: Path,
    *,
    device: str = "cuda:0",
    dtype: str | None = None,
    build: Callable[..., Any] | None = None,
) -> Path:
    """Build the variant through build_variant and save the HF model (for CAST: the model it wraps)."""
    if build is None:
        from lift.models.variants import build_variant as build
    dtype = dtype or config.get("generation.dtype")
    lv = build(spec, device=device, dtype=dtype)
    model = lv.base if spec.method in CAST_METHODS else lv.model
    if model is None:
        raise PublishError(f"{spec.name}: the CAST builder left LoadedVariant.base empty; nothing to save")
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder, safe_serialization=True)
    lv.tokenizer.save_pretrained(folder)
    render.write_text(
        folder / "lift_variant.json",
        json.dumps(
            {
                "spec": spec.model_dump(),
                "model_rev": lv.model_rev,
                "adapter_hash": lv.adapter_hash,
                "template_hash": lv.template_hash,
                "dtype": lv.dtype,
            },
            indent=1,
        ),
    )
    return folder


# ------------------------------------------------------------------------------------------- cards


def _links() -> dict[str, str]:
    return {
        "logs_repo": settings.repo("logs").url,
        "raw_repo": settings.repo("raw").url,
        "params_repo": settings.repo("repair_params").url,
        "github": settings.github_repo()
        or pending("the GitHub repo url", "configs/publish.yaml github_repo", "a person"),
    }


def model_card(facts: Facts, variant: str | None, spec: Any | None) -> str:
    from lift.publish.datasets import verdict_lines

    method = spec.method if spec else None
    stage_name = variant or pending("the frozen winner", shown(selection_path()), P05)
    params = (
        (", ".join(f"{k} = {v}" for k, v in spec.params.items()) or "none")
        if spec
        else pending("the frozen parameters", "configs/variants.yaml", "P05 freeze")
    )
    cast_note = ""
    if method in CAST_METHODS:
        cast_note = render.template("cast_usage.md")
    adapter_repo = (
        settings.repo("adapter", adapter=spec.adapter).url
        if spec and spec.adapter
        else pending("the adapter repo", "configs/variants.yaml", "P05 freeze")
    )
    context = {
        "variant": stage_name,
        "method": METHOD_TEXT.get(method, pending("the repair method", shown(selection_path()), P05)),
        "params": params,
        "adapter": f"`{spec.adapter}`" if spec and spec.adapter else adapter_repo,
        "adapter_repo": adapter_repo,
        "base_model": config.get("models.global.id"),
        "base_revision": config.get("models.global.revision"),
        "verdicts": verdict_lines(facts),
        "drift": facts.report_value("/primary/drift/pooled", "pp_ci", what="the pooled Hindi drift effect"),
        "drift_p": facts.report_value("/primary/drift/pooled/p", "p", what="the drift p-value"),
        "repair": facts.report_value(
            "/primary/repair/pooled", "pp_ci", what="the pooled Hindi repair effect"
        ),
        "repair_p": facts.report_value("/primary/repair/pooled/p", "p", what="the repair p-value"),
        "utility_retained": facts.report_value(
            "/headline/utility_retained", "pct_ci", what="the share of the adapter's utility gain retained"
        ),
        "per_language": facts.table("per_language"),
        "overrefusal": facts.table("overrefusal"),
        "belebele": facts.table("belebele"),
        "cast_usage": cast_note,
        "max_new_tokens": facts.config("project", "generation.max_new_tokens", "int"),
        **_links(),
    }
    meta = {
        "license": "cc-by-nc-4.0",
        "base_model": config.get("models.global.id"),
        "language": list(config.get("languages.all")),
        "library_name": "transformers",
        "pipeline_tag": "text-generation",
        "tags": [
            "safety",
            "multilingual",
            "hindi",
            "tiny-aya",
            "model-repair",
            "lexsi-hackathon",
            *([method] if method else []),
        ],
    }
    return render.frontmatter(meta) + "\n" + render.fill(render.template("model_card.md"), context)


def adapter_card(name: str, spec: Any) -> str:
    meta = {
        "license": "cc-by-nc-4.0",
        "base_model": config.get("models.global.id"),
        "library_name": "peft",
        "language": ["hi"],
        "tags": ["lora", "hindi", "tiny-aya", "safety-drift", "research-only"],
        "extra_gated_heading": "Unrepaired research adapter: it drifted on safety",
        "extra_gated_prompt": render.template("gate_adapter.txt").strip(),
        "extra_gated_fields": {"Intended use": "text", "I will not deploy this adapter": "checkbox"},
    }
    body = render.fill(
        render.template("adapter_card.md"),
        {
            "name": name,
            "seed": str(spec.seed),
            "base_model": config.get("models.global.id"),
            "base_revision": config.get("models.global.revision"),
            "adapter_hash": spec.artifacts.get("adapter", "?"),
            "model_repo_note": "The frozen repair of the seed-1 adapter is published separately; see the "
            "model collection.",
            **_links(),
        },
    )
    return render.frontmatter(meta) + "\n" + body


# ------------------------------------------------------------------------------------------- builders


def build_model(
    facts: Facts, variant: str | None = None, *, build_weights: bool = False, device: str = "cuda:0"
) -> Upload:
    winner = frozen_winner(facts)
    variant = variant or winner
    frozen = frozen_names(facts)
    if variant and facts.selection() is not None and variant not in frozen:
        raise PublishError(
            f"{variant} is not a frozen variant in {shown(selection_path())} "
            f"(frozen: {', '.join(sorted(frozen)) or 'none'})"
        )
    spec = _spec(variant) if variant else None
    repo = settings.repo("model", variant=variant or "pending-winner")
    folder = stage_root() / "model" / (variant or "pending-winner")
    folder.mkdir(parents=True, exist_ok=True)
    expected: list[str] = []
    notes: list[str] = []
    if variant is None or facts.selection() is None:
        expected.append(f"the frozen winner ({shown(selection_path())}, {P05})")
    if variant is not None and spec is None:
        expected.append(f"the configs/variants.yaml entry of {variant} (P05 freeze)")
    elif spec is not None:
        if spec.seed not in (None, 1):
            notes.append(f"{variant} is seed {spec.seed}; the model card describes the seed-1 winner")
        render.write_text(folder / "repair" / "params.json", json.dumps(params_doc(spec), indent=1))
        if spec.method in CAST_METHODS:
            copied = _copy_artifacts(cast_artifacts(spec.adapter), folder / "repair" / "cast")
            if not copied:
                expected.append(f"CAST artifacts in {shown(cast_artifacts(spec.adapter))} (lift fit-cast)")
            shutil.copyfile(render.templates_dir() / "cast_generate.py", folder / "cast_generate.py")
        if build_weights:
            materialise(spec, folder, device=device)
        if not weights_present(folder):
            expected.append("model weights (lift publish model --build, on the GPU VM)")
    render.write_text(folder / "README.md", model_card(facts, variant, spec))
    return Upload(
        repo=repo,
        folder=folder,
        title=f"repaired model ({variant or 'winner pending'})",
        expected=expected,
        notes=notes,
    )


def build_params(facts: Facts) -> Upload:
    repo = settings.repo("repair_params")
    folder = stage_root() / "repair_params"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    rows, expected = [], []
    names = sorted(frozen_names(facts))
    for name in names:
        spec = _spec(name)
        if spec is None:
            expected.append(f"the configs/variants.yaml entry of {name}")
            continue
        render.write_text(folder / "params" / f"{name}.json", json.dumps(params_doc(spec), indent=1))
        if spec.method in CAST_METHODS:
            copied = _copy_artifacts(cast_artifacts(spec.adapter), folder / "cast" / spec.adapter)
            if not copied:
                expected.append(f"CAST artifacts for {spec.adapter} (lift fit-cast)")
        rows.append(
            [
                f"`{name}`",
                spec.method,
                f"`{spec.adapter}`",
                str(spec.seed),
                ", ".join(f"{k} = {v}" for k, v in spec.params.items()),
            ]
        )
    if not names:
        expected.append(f"the frozen settings ({shown(selection_path())}, {P05})")
    table = (
        render.markdown_table(["variant", "method", "adapter", "seed", "parameters"], rows)
        if rows
        else pending("the frozen settings", shown(selection_path()), P05)
    )
    meta = {
        "license": "cc-by-nc-4.0",
        "base_model": config.get("models.global.id"),
        "tags": ["safety", "model-repair", "activation-steering", "tiny-aya"],
    }
    body = render.fill(render.template("params_card.md"), {"table": table, **_links()})
    render.write_text(folder / "README.md", render.frontmatter(meta) + "\n" + body)
    return Upload(repo=repo, folder=folder, title="repair parameters", expected=expected)


def registered_adapters() -> dict[str, Any]:
    from lift.models.variants import load_registry

    return {name: spec for name, spec in load_registry().items() if spec.method == "adapter"}


def build_adapters(facts: Facts) -> list[Upload]:
    out = []
    for name, spec in sorted(registered_adapters().items()):
        src = paths.adapters_dir() / name
        if not src.is_dir():
            continue
        folder = stage_root() / "adapters" / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        expected = []
        for f in ADAPTER_FILES:
            if (src / f).is_file():
                shutil.copyfile(src / f, folder / f)
            else:
                expected.append(f"{shown(src / f)}")
        render.write_text(folder / "README.md", adapter_card(name, spec))
        out.append(
            Upload(
                repo=settings.repo("adapter", adapter=name),
                folder=folder,
                title=f"adapter {name} (gated, automatic approval)",
                expected=expected,
            )
        )
    return out


def build_all(
    facts: Facts | None = None,
    variant: str | None = None,
    *,
    build_weights: bool = False,
    device: str = "cuda:0",
    adapters: bool = True,
) -> list[Upload]:
    facts = facts or Facts()
    uploads = [build_model(facts, variant, build_weights=build_weights, device=device), build_params(facts)]
    if adapters:
        uploads += build_adapters(facts)
    return uploads
