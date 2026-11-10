"""``lift publish model``: the repaired model, the repair parameters and the gated adapters.

- Repaired model: the overall frozen winner on seed 1 (selection.json ``overall.winner``), and only it:
  the card prints the winner's repair effect. WiSE-FT and ReSta are saved as merged weights; CAST as the
  adapter-merged model it wraps, plus its artifacts and ``cast_generate.py``, a small wrapper around
  ``lift.repair.cast.make``. The weights come from ``lift.models.variants.build_variant`` (the one code
  path), materialised by ``--build`` on a machine with the GPU stack and room for the model (the Colab
  VM: WiSE-FT and ReSta hold two or three copies). ``--build`` clears the staging folder first; without
  it, the weights already staged are uploaded only while their ``lift_variant.json`` names the variant's
  registry entry as it is now (spec, adapter hash, Global revision), so ``--build`` once, check the dry
  run, then ``--yes`` uploads the same weights without building them again.
  Access (docs/decisions.md item 84, pending team confirmation): always gated with automatic approval,
  like the adapters (item 14), whatever the method or verdict. Its weights and the public parameters
  give back the gated adapter: WiSE-FT as Global + (weights - Global) / lambda, ReSta as
  weights - alpha x (Global - Base), and a CAST model's weights are the adapter, merged (the repair is
  only in the wrapper). A public ``repos.model`` in configs/publish.yaml is refused. The card and the
  gate text give this model's own reasons (``gate_reasons``).
- Repair parameters (public): every frozen setting's VariantSpec on seeds 1-3 (lambda, alpha or
  strength, artifact hashes) and the CAST artifacts from outputs/artifacts/cast/{adapter}/. They stay
  public because no repaired weights are: on their own they rebuild nothing.
- A CAST artifact is staged only when its sha256 is the one its registry entry pins (``artifacts.cast``,
  which ``cast_generate.py`` checks again); a stale or missing one is listed as expected.
- Every value the model card renders is recorded in outputs/publish/model.sources.json.
- Adapters (gated, automatic approval): every registered adapter (configs/variants.yaml), from
  outputs/adapters/, its weights and adapter_config.json only (registration.json stays private: it names
  the runs repo). One not pulled to this machine is listed as missing, which blocks ``--yes``.
- Every staged file names no absolute local path: each is made repo-relative (a Hugging Face cache
  snapshot, its repo id) or the build refused (``lift.publish.preflight.relativise_folder``, XB2-04).
- The repaired model and the parameters name the frozen recipe, so both are blocked (``--yes`` refuses
  them; the dry run still stages and prints them) until configs/frozen_recipe.json is the one the newest
  ``recipe-frozen-*`` tag froze (``lift.publish.inputs.recipe_block``, review XB2-02).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lift import config, paths
from lift.io import sha256_file
from lift.publish import PublishError, preflight, render, settings
from lift.publish.hub import Upload
from lift.publish.inputs import (
    NOT_RUN,
    P05,
    Facts,
    frozen_variants,
    frozen_winner,
    pending,
    recipe_block,
    report_path,
    selection_path,
    selection_problem,
    selection_verdict,
    shown,
)

CAST_METHODS = ("cast", "cast_always")
CAST_ARTIFACT = "cast.safetensors"  # lift.repair.cast.ARTIFACT, the file cast_generate.py loads
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
    return set(frozen_variants(facts))


# Why each method's weights are gated (docs/decisions.md item 84): with the public parameters, they give
# back the gated adapter (item 14).
METHOD_GATE = {
    "wiseft": "it is a WiSE-FT repair, Global + lambda x (adapter - Global), and lambda is public, so "
    "anyone could rebuild the gated adapter from these weights as Global + (weights - Global) / lambda",
    "resta": "it is a ReSta repair, adapter + alpha x (Global - Base), and alpha, Global and Base are "
    "public, so anyone could rebuild the gated adapter from these weights as weights - alpha x "
    "(Global - Base)",
    "cast": "it is a CAST repair: these weights are the unrepaired adapter, merged, and the repair is "
    "applied only by the wrapper (cast_generate.py); without it the model behaves as the adapter, which "
    "may answer harmful requests that Tiny Aya Global refuses",
    "adapter": "these weights are the unrepaired adapter, merged",
}
METHOD_GATE["cast_always"] = METHOD_GATE["cast"]


def gate_reasons(facts: Facts, spec: Any | None) -> list[str]:
    """Why the repaired model's repo is gated (docs/decisions.md item 84, pending team confirmation).
    Never empty: every repaired model is gated, and these are this model's reasons, for the card and the
    gate text. An unknown method or verdict says so."""
    method = spec.method if spec is not None else None
    reasons = [METHOD_GATE.get(method, "its repair method is not known yet")]
    verdicts = [
        str(v.get("verdict"))
        for v in (facts.raw(report_path(), "/verdicts") or [])
        if isinstance(v, dict) and v.get("verdict")
    ]
    chosen = selection_verdict(facts)
    if chosen == "reject" or "reject" in verdicts:
        reasons.append(
            'the verdict is "reject": no setting met every selection constraint, so this is the best setting '
            "frozen anyway, not a repair"
        )
    elif chosen != "repair":
        reasons.append("the selection verdict is not known yet")
    return reasons


def model_repo(facts: Facts, variant: str | None, spec: Any | None) -> settings.RepoSpec:
    """The repaired model's repo (configs/publish.yaml ``repos.model``), which must be gated whatever the
    method or verdict (docs/decisions.md item 84): a public one is refused."""
    repo = settings.repo("model", variant=variant or "pending-winner")
    if not repo.gated:
        raise PublishError(
            "configs/publish.yaml repos.model must be gated (gated-auto): with the public repair parameters, "
            "any repaired model's weights give back the gated adapter (docs/decisions.md item 84, pending "
            "team confirmation; changing it needs a new dated entry there)"
        )
    return repo


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


def cast_artifact_problem(spec: Any) -> str | None:
    """Why ``spec``'s CAST artifact cannot be staged (None when it can): ``cast.safetensors`` must have the
    sha256 the registry entry pins (``artifacts.cast``), or a stale artifact would go up beside a
    params.json that pins another one (and cast_generate.py would refuse it)."""
    path = cast_artifacts(spec.adapter) / CAST_ARTIFACT
    pinned = (spec.artifacts or {}).get("cast")
    if not pinned:
        return f"the sha256 of {spec.name}'s {CAST_ARTIFACT} (configs/variants.yaml artifacts.cast)"
    if not path.is_file():
        return f"{shown(path)} with sha256 {pinned[:12]} (lift fit-cast, then lift sync pull)"
    found = sha256_file(path)
    if found != pinned:
        return (
            f"{shown(path)} with sha256 {pinned[:12]}, as configs/variants.yaml pins for {spec.name} (found "
            f"{found[:12]}: a stale artifact, refitted or not pulled; lift sync pull)"
        )
    return None


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


VARIANT_FILE = "lift_variant.json"
NOT_WEIGHTS = {"README.md", "cast_generate.py", VARIANT_FILE}  # staged by build_model, not by save_pretrained


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def weights_mismatch(folder: Path, spec: Any) -> str | None:
    """Why the weights staged in ``folder`` are not this registry entry's (None when they are): the spec,
    the adapter hash and the Global revision must match what ``materialise`` recorded."""
    if not weights_present(folder):
        return "no weights staged"
    path = folder / VARIANT_FILE
    if not path.is_file():
        return f"no {VARIANT_FILE} records what the staged weights were built from"
    try:
        made = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"{VARIANT_FILE} is not valid JSON"
    if made.get("spec") != _jsonable(spec.model_dump()):
        return "they were built from another registry entry of this variant (re-frozen or re-registered)"
    if spec.artifacts.get("adapter") and made.get("adapter_hash") != spec.artifacts["adapter"]:
        return "they were built from another adapter (hash differs from configs/variants.yaml)"
    rev = config.get("models.global.revision")
    if made.get("model_rev") and rev and made["model_rev"] != rev:
        return f"they were built on Global {str(made['model_rev'])[:12]}, not the pinned {str(rev)[:12]}"
    return None


def _clear(folder: Path, *, keep: set[str]) -> None:
    """Empty the staging folder except the files named in ``keep`` (paths relative to it)."""
    root = stage_root().resolve()
    if root not in folder.resolve().parents:
        raise PublishError(f"refusing to clear {folder}: not under {root}")
    for f in sorted(folder.rglob("*"), reverse=True):
        rel = f.relative_to(folder).as_posix()
        if f.is_file() and rel not in keep:
            f.unlink()
        elif f.is_dir() and not any(f.iterdir()):
            f.rmdir()


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
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    render.write_text(
        folder / VARIANT_FILE,
        json.dumps(
            {
                "spec": _jsonable(spec.model_dump()),
                "model_rev": lv.model_rev,
                "adapter_hash": lv.adapter_hash,
                "template_hash": lv.template_hash,
                "dtype": lv.dtype,
                # what save_pretrained wrote (top level): kept across a later run without --build
                "files": [f for f in files if "/" not in f and f not in NOT_WEIGHTS],
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


def model_card(facts: Facts, variant: str | None, spec: Any | None, gated: list[str] | None = None) -> str:
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
        "drift_p": facts.report_value("/primary/drift/pooled/p", "p_stmt", what="the drift p-value"),
        "repair": facts.report_value(
            "/primary/repair/pooled", "pp_ci", what="the pooled Hindi repair effect"
        ),
        "repair_p": facts.report_value("/primary/repair/pooled/p", "p_stmt", what="the repair p-value"),
        **this_model(facts),
        "drift_seeds": facts.report_value(
            "/primary/drift/pooled/seeds", "list", what="the seeds pooled for drift"
        ),
        "repair_seeds": facts.report_value(
            "/primary/repair/pooled/seeds", "list", what="the seeds pooled for repair"
        ),
        "per_seed": facts.per_seed_table([("Drift", "drift"), ("Repair", "repair")]),
        "safety_claim": safety_claim(facts),
        "utility_retained": facts.report_value(
            "/headline/utility_retained", "pct_ci", what="the share of the adapter's utility gain retained"
        ),
        "per_language": facts.table("per_language"),
        "overrefusal": facts.table("overrefusal"),
        "belebele": facts.table("belebele"),
        "cast_usage": cast_note,
        "access_note": _access_note(gated or []),
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
    if gated:
        meta.update(
            {
                "extra_gated_heading": "Research model: gated (see the model card)",
                "extra_gated_prompt": render.fill(
                    render.template("gate_model.txt"), {"reasons": "; and ".join(gated)}
                ).strip(),
                "extra_gated_fields": {"Intended use": "text", "I will not deploy this model": "checkbox"},
            }
        )
    return render.frontmatter(meta) + "\n" + render.fill(render.template("model_card.md"), context)


def this_model(facts: Facts) -> dict[str, str]:
    """The repair effect of these weights: the seed-1 row of P07's per-seed repair effect (review XB-13),
    "not run" when seed 1 was not, pending while there is no report."""
    seeds = facts.raw(report_path(), "/primary/repair/per_seed")
    if seeds is None:
        line = facts.report_value("/primary/repair/per_seed", "pp_ci", what="the seed-1 repair effect")
        return {"repair_this": line, "repair_this_p": ""}
    i = next((i for i, r in enumerate(seeds) if isinstance(r, dict) and r.get("seed") == 1), None)
    if i is None:
        return {"repair_this": NOT_RUN, "repair_this_p": f"p {NOT_RUN}"}
    at = f"/primary/repair/per_seed/{i}"
    return {
        "repair_this": facts.report_value(at, "pp_ci", what="the seed-1 repair effect"),
        "repair_this_p": facts.report_value(f"{at}/p", "p_stmt", what="the seed-1 repair p-value"),
    }


def safety_claim(facts: Facts) -> str:
    """The one sentence about safety the card may make: "safer than the adapter on our benchmark" only
    when every final verdict is repair and the pooled Hindi repair interval lies wholly below zero."""
    verdicts = [
        v.get("verdict") for v in (facts.raw(report_path(), "/verdicts") or []) if isinstance(v, dict)
    ]
    hi = facts.raw(report_path(), "/primary/repair/pooled/hi")
    supported = (
        bool(verdicts)
        and all(v == "repair" for v in verdicts)
        and isinstance(hi, (int, float))
        and not isinstance(hi, bool)
        and hi < 0
    )
    if supported:
        return (
            "On our Hindi benchmark the frozen recipe is safer than the adapter (its pooled repair interval "
            "above lies below zero), which does not make this model safe in general."
        )
    return "This card makes no safety claim beyond the verdict and the intervals above."


def _access_note(gated: list[str]) -> str:
    if not gated:
        return ""
    why = "; and ".join(gated)
    return (
        "**This repo is gated (automatic approval), like the unrepaired adapters, because " + why + ".** "
        "It is shared so the evaluation can be reproduced (docs/decisions.md item 84).\n"
    )


def adapter_card(name: str, spec: Any) -> str:
    meta = {
        "license": "cc-by-nc-4.0",
        "base_model": config.get("models.global.id"),
        "library_name": "peft",
        "language": ["hi"],
        "tags": ["lora", "hindi", "tiny-aya", "safety-research", "research-only"],
        "extra_gated_heading": "Unrepaired research adapter: fine-tuning may affect safety",
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
    """The overall winner's repo. Another variant is refused: the card's repair effect is the winner's."""
    block = recipe_block(facts)
    winner = frozen_winner(facts)
    if variant is not None and variant != winner:
        raise PublishError(
            f"only the overall winner of {shown(selection_path())} is published as the repaired model "
            f"(overall.winner: {winner or 'none yet'}; {variant} is not it): its card shows the winner's "
            "repair effect. Every frozen setting's parameters go to the repair-parameters repo."
        )
    variant = winner
    spec = _spec(variant) if variant else None
    gated = gate_reasons(facts, spec)
    repo = model_repo(facts, variant, spec)
    folder = stage_root() / "model" / (variant or "pending-winner")
    folder.mkdir(parents=True, exist_ok=True)
    expected: list[str] = []
    notes: list[str] = []
    problem = selection_problem(facts)
    if problem:
        expected.append(f"the frozen winner: {problem}")
    if variant is not None and spec is None:
        expected.append(f"the configs/variants.yaml entry of {variant} (P05 freeze)")
    # Only the weights materialised for this very registry entry survive; everything else is staged anew.
    keep: set[str] = set()
    if spec is not None and not build_weights and weights_mismatch(folder, spec) is None:
        made = json.loads((folder / VARIANT_FILE).read_text(encoding="utf-8"))
        keep = {*made.get("files", []), VARIANT_FILE}
    _clear(folder, keep=keep)
    if spec is not None:
        if spec.seed not in (None, 1):
            raise PublishError(f"{variant} is seed {spec.seed}; the repaired model is the seed-1 winner")
        render.write_text(folder / "repair" / "params.json", json.dumps(params_doc(spec), indent=1))
        if spec.method in CAST_METHODS:
            problem = cast_artifact_problem(spec)
            if problem:
                expected.append(f"the CAST artifact: {problem}")
            else:
                _copy_artifacts(cast_artifacts(spec.adapter), folder / "repair" / "cast")
            shutil.copyfile(render.templates_dir() / "cast_generate.py", folder / "cast_generate.py")
        if build_weights:
            materialise(spec, folder, device=device)
        why = weights_mismatch(folder, spec)
        if why:
            expected.append(f"model weights ({why}: lift publish model --build, on the GPU VM)")
    notes.append("gated (automatic approval), docs/decisions.md item 84: " + "; ".join(gated))
    render.write_text(folder / "README.md", model_card(facts, variant, spec, gated))
    facts.write_sources(stage_root() / "model.sources.json")
    notes += preflight.relativised_note(preflight.relativise_folder(folder))
    return Upload(
        repo=repo,
        folder=folder,
        title=f"repaired model ({variant or 'winner pending'})",
        expected=expected,
        notes=notes,
        blocked=[block] if block else [],
    )


def build_params(facts: Facts) -> Upload:
    block = recipe_block(facts)
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
            problem = cast_artifact_problem(spec)
            if problem:
                expected.append(f"the CAST artifact of {spec.adapter}: {problem}")
            else:
                _copy_artifacts(cast_artifacts(spec.adapter), folder / "cast" / spec.adapter)
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
        expected.append(f"the frozen settings: {selection_problem(facts) or 'none listed'}")
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
    notes = preflight.relativised_note(preflight.relativise_folder(folder))
    blocked = [block] if block else []
    return Upload(
        repo=repo, folder=folder, title="repair parameters", expected=expected, blocked=blocked, notes=notes
    )


def registered_adapters() -> dict[str, Any]:
    from lift.models.variants import load_registry

    return {name: spec for name, spec in load_registry().items() if spec.method == "adapter"}


def build_adapters(facts: Facts) -> list[Upload]:
    """One gated upload per registered adapter (decisions item 14: every artifact on HF). An adapter not
    on this machine is still planned, its files listed as missing, so the dry run shows it and ``--yes``
    refuses until it is pulled (review XB-05)."""
    out = []
    for name, spec in sorted(registered_adapters().items()):
        src = paths.adapters_dir() / name
        folder = stage_root() / "adapters" / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        expected = []
        for f in ADAPTER_FILES:
            if (src / f).is_file():
                shutil.copyfile(src / f, folder / f)
            else:
                expected.append(f"{shown(src / f)} (not on this machine: lift sync pull)")
        render.write_text(folder / "README.md", adapter_card(name, spec))
        out.append(
            Upload(
                repo=settings.repo("adapter", adapter=name),
                folder=folder,
                title=f"adapter {name} (gated, automatic approval)",
                expected=expected,
                notes=preflight.relativised_note(preflight.relativise_folder(folder)),
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
