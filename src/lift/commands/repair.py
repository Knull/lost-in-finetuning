"""``lift fit-cast|register-repairs|sweep|select``: the repairs' dev sweep and the selection rule (P05).

The rules are in lift.repair (cast, settings, sweep, select). Torch is imported inside the GPU handlers,
after the checks that need none, so ``lift --help``, ``lift register-repairs``, ``lift select`` and
``lift sweep --judge-only`` work on a laptop without the GPU stack.

Order (prompts/P05_repairs.md):
  1. lift fit-cast --adapter adapter-X-s1                       GPU job line, after lift sync pull
  2. lift register-repairs --adapter adapter-X-s1               laptop, after lift sync pull; commit
                                                                configs/variants.yaml and push the ref
  3. lift sweep --adapter adapter-X-s1                          GPU job line (registers the same entries
                                                                in the VM's checkout too)
  4. lift sweep --adapter adapter-X-s1 --judge-only             laptop, after lift sync pull
  5. lift select --adapter adapter-X-s1                         laptop
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from lift.schemas import LANGS

log = logging.getLogger("lift.repair")

FIT_HELP = """\
Fit CAST's condition (layer, threshold, comparator by held-out F1) and its steering vectors on one
adapter's activations, from train/fit prompts only (configs/repairs.yaml cast.fit: PolyRefuse harmful
train against harmless train plus the OR-Bench fit prompts, all six languages pooled). Writes
outputs/artifacts/cast/{adapter}/cast.safetensors (the artifact whose sha256 every CAST entry pins) and
fit.json (counts, held-out F1, SafeTune's runtime config, and the gate's firing rate per language and set
on the fit set's held-out half: it should fire on harmful prompts and not on benign ones). An artifact
fitted on the same adapter, model and prompts is kept (exit 0); another is refused unless --refit, which
makes every CAST entry registered on the old one stale (lift register-repairs --replace).
"""

SWEEP_HELP = """\
The seed-1 dev sweep (lift.repair.sweep): for each repair setting (3 WiSE-FT, 3 ReSta, 3 CAST strengths
and CAST with the gate always on), lift run-dev (dev safety, 6 languages), lift utility --generate-only
(100 Hindi pairs against stock) and lift belebele --langs hi, in this process. Resumes after a crash
(outputs/metrics/sweep/{adapter}.json); each setting gets a run manifest and a cost summary (GPU seconds
per generated token; CAST's gate prefill apart). Exit 0 done, 3 some stage left cells listed with their
remedies, 1 some stage failed. --judge-only (laptop) runs lift utility --judge-only for every setting.

A session that drops: rerun the same line (stages already done are skipped). To split it over sessions,
run one family per job: --repairs wiseft, then resta, then cast.
"""


def _families(value: str | None) -> list[str]:
    from lift.repair.settings import FAMILIES

    names = [x for x in (value or ",".join(FAMILIES)).split(",") if x]
    bad = sorted(set(names) - set(FAMILIES))
    if bad:
        raise SystemExit(f"unknown repair families {bad}; choose from {list(FAMILIES)}")
    return list(dict.fromkeys(names))


def _adapter(name: str) -> Any:
    from lift.repair.settings import adapter_spec

    try:
        return adapter_spec(name)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc.args[0] if exc.args else exc)) from None


# --- lift fit-cast --------------------------------------------------------------------------------------


def fit_prompts() -> dict[str, list[Any]]:
    """{"harmful": [...], "benign": [...]} PromptRecords from configs/repairs.yaml cast.fit, language by
    language in its order (benign: each benign set in order within a language). Train/fit splits only."""
    from lift import config
    from lift.prompts import load

    if config.get("cast.language_aware", name="repairs") is not False:
        raise SystemExit("configs/repairs.yaml cast.language_aware must be false: language-aware CAST is cut "
                         "(docs/decisions.md item 2)")
    fit = config.get("cast.fit", name="repairs")
    sources = {"harmful": [fit["harmful"]], "benign": list(fit["benign"])}
    out: dict[str, list[Any]] = {"harmful": [], "benign": []}
    for kind, sets in sources.items():
        for lang in fit["langs"]:
            for s in sets:
                if s["split"] not in ("train", "fit"):
                    raise SystemExit(f"cast.fit uses {s['set']} {s['split']}: CAST fits on train/fit prompts "
                                     "only")
                try:
                    out[kind].extend(load(s["set"], lang, s["split"]))
                except FileNotFoundError as exc:
                    raise SystemExit(f"{exc} (a GPU job gets the built prompts with `lift sync "
                                     "pull`)") from None
    for kind, recs in out.items():
        if not recs:
            raise SystemExit(f"no {kind} fit prompts")
    return out


def fit_digest(prompts: dict[str, list[Any]]) -> str:
    from lift.io import sha256_obj, sha256_text

    return sha256_obj({k: [[r.set, r.lang, r.pid, sha256_text(r.text)] for r in v]
                       for k, v in prompts.items()})


def firing_report(cast_model: Any, tokenizer: Any, prompts: dict[str, list[Any]]) -> dict[str, Any]:
    """The gate's firing rate per language (and per set for the benign side) on the held-out half of each
    class, as fit_cast_condition split it."""
    from lift.repair import cast

    out: dict[str, Any] = {}
    for kind, recs in prompts.items():
        held = [recs[i] for i in cast.heldout_indices(len(recs))]
        metas = cast.firing(cast_model, tokenizer, [r.text for r in held])
        for r, m in zip(held, metas, strict=True):
            cell = out.setdefault(kind, {}).setdefault(r.lang, {}).setdefault(r.set, {"n": 0, "fired": 0})
            cell["n"] += 1
            cell["fired"] += int(m["gate_fired"])
    for kinds in out.values():
        for sets in kinds.values():
            for cell in sets.values():
                cell["rate"] = cell["fired"] / cell["n"]
    return out


def _fit_cast(args: argparse.Namespace) -> int:
    from lift.commands import run

    run._logging()
    run.require_pinned("models.global")
    spec = _adapter(args.adapter)
    dtype = run._dtype(args.dtype)
    run._refuse_other_dtype(dtype, args.device)
    prompts = fit_prompts()
    digest = fit_digest(prompts)
    from lift.repair import cast
    from lift.repair.settings import read_cast_info

    path = cast.artifact_path(args.adapter)
    if path.is_file() and not args.refit:
        from lift import config
        from lift.models.load import resolved_revision

        info = read_cast_info(path)
        model_rev = resolved_revision(config.get("models.global.id"), config.get("models.global.revision"))
        if (info.get("adapter_hash"), info.get("fit_digest"), info.get("dtype"), info.get("model_rev")) == \
                (spec.artifacts["adapter"], digest, dtype, model_rev):
            print(f"{path}: already fitted on {args.adapter} ({spec.artifacts['adapter'][:12]}) with these "
                  "prompts "
                  f"in {dtype}; kept (sha256 {cast.artifact_hash(path)[:12]})")
            return 0
        print(f"{path} exists and was fitted on another adapter, Global revision, prompt set or precision; "
              "--refit replaces it "
              "(every CAST entry registered on it then goes stale: lift register-repairs --replace)",
              file=sys.stderr)
        return 2
    from lift.manifest import finish_run, start_run
    from lift.models.load import template_hash
    from lift.models.variants import build_variant

    variant = build_variant(spec, device=args.device, dtype=dtype)
    import safetune

    from lift import config

    meta = {"adapter": args.adapter, "adapter_hash": variant.adapter_hash, "model_rev": variant.model_rev,
            "dtype": dtype, "template": template_hash(variant.tokenizer), "fit_digest": digest,
            "safetune": safetune.__version__,
            "fit_sets": json_safe(config.get("cast.fit", name="repairs"))}
    manifest = start_run("fit_cast", adapter=args.adapter, dtype=dtype, device=args.device, devices=1,
                         n_harmful=len(prompts["harmful"]), n_benign=len(prompts["benign"]))
    try:
        out = cast.fit(variant.model, variant.tokenizer, [r.text for r in prompts["harmful"]],
                       [r.text for r in prompts["benign"]], out_dir=path.parent, meta=meta,
                       device=args.device)
        _cond, _vecs, info = cast.load(out)
        gated = cast.make(variant.model, out, 1.0, gate=True, pad_id=variant.tokenizer.pad_token_id)
        firing = firing_report(gated, variant.tokenizer, prompts)
    except BaseException as exc:
        finish_run(manifest, status="failed", error=type(exc).__name__)
        raise
    report = {"artifact": cast.ARTIFACT, "sha256": cast.artifact_hash(out), **info,
              "counts": _counts(prompts), "firing_heldout": firing,
              "safetune_config": cast._check_runtime()}
    from lift.repair.sweep import write_json

    write_json(path.parent / cast.REPORT, report)
    from lift.train.lora import _shown

    finish_run(manifest, status="ok", artifact=_shown(out), sha256=report["sha256"])
    _print_fit(args.adapter, report)
    return 0


def json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _counts(prompts: dict[str, list[Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for kind, recs in prompts.items():
        for r in recs:
            key = f"{r.set}_{r.lang}_{r.split}"
            out.setdefault(kind, {})[key] = out.setdefault(kind, {}).get(key, 0) + 1
    return out


def _print_fit(adapter: str, report: dict[str, Any]) -> None:
    print(f"CAST fitted on {adapter}: layer {report['condition_layer']}, threshold "
          f"{report['threshold']:.4f}, "
          f"comparator {report['comparator']}, held-out F1 {report['f1_heldout']:.3f}; steering layers "
          f"{report['behavior_layers']}; sha256 {report['sha256'][:12]}")
    print("gate firing rate on the held-out half (harmful should be high, benign low):")
    for lang in LANGS:
        cells = []
        for kind in ("harmful", "benign"):
            for set_name, cell in sorted(report["firing_heldout"].get(kind, {}).get(lang, {}).items()):
                cells.append(f"{set_name} {100 * cell['rate']:.1f}% of {cell['n']}")
        if cells:
            print(f"  {lang}: " + ", ".join(cells))


# --- lift register-repairs ------------------------------------------------------------------------------


def _chosen(args: argparse.Namespace) -> list[Any]:
    from lift.repair import settings as rs

    chosen = rs.settings(_families(args.repairs))
    if getattr(args, "settings", None):
        tags = [t for t in args.settings.split(",") if t]
        by_tag = {s.tag: s for s in rs.settings()}
        bad = sorted(set(tags) - set(by_tag))
        if bad:
            raise SystemExit(f"unknown settings {bad}; choose from {sorted(by_tag)}")
        chosen = [by_tag[t] for t in tags]
    return chosen


def _register(args: argparse.Namespace) -> int:
    from lift.repair import settings as rs
    from lift.train.lora import RegistryConflict

    _adapter(args.adapter)
    try:
        specs = rs.specs_for(args.adapter, _chosen(args))
        outcome = rs.register(specs, replace=args.replace)
    except (FileNotFoundError, ValueError, RegistryConflict) as exc:
        print(f"lift register-repairs: {exc}", file=sys.stderr)
        return 2
    for name, what in outcome.items():
        print(f"{name}: {what}")
    if any(v != "unchanged" for v in outcome.values()):
        print("commit configs/variants.yaml and push it to the ref the GPU jobs run")
    return 0


# --- lift sweep -----------------------------------------------------------------------------------------


def _sweep(args: argparse.Namespace) -> int:
    from lift.commands import run
    from lift.repair import sweep

    run._logging()
    families = _families(args.repairs)
    _adapter(args.adapter)
    if args.judge_only:
        return sweep.judge_only(args.adapter, families=families, no_wait=args.no_wait, sync=args.sync)
    if args.no_wait or args.sync:
        print("lift sweep: --no-wait and --sync apply to --judge-only", file=sys.stderr)
        return 2
    keys = ["models.global"] + (["models.base"] if "resta" in families else [])
    run.require_pinned(*keys)
    from lift.train.lora import RegistryConflict

    try:
        return sweep.gpu(args.adapter, families=families, device=args.device, judge_device=args.judge_device,
                         dtype=args.dtype, batch_size=args.batch_size, recheck=args.recheck)
    except (FileNotFoundError, ValueError, RegistryConflict) as exc:
        print(f"lift sweep: {exc}", file=sys.stderr)
        return 2


# --- lift select ----------------------------------------------------------------------------------------


def _select(args: argparse.Namespace) -> int:
    from lift.repair import select

    _adapter(args.adapter)
    try:
        doc = select.run(args.adapter, families=_families(args.repairs), dtype=args.dtype,
                         refreeze=args.refreeze)
    except select.RecipeFrozen as exc:
        print(f"lift select: {exc}", file=sys.stderr)
        return 2
    json_path, md_path = select.report_paths()
    print_selection(doc)
    print(f"wrote {json_path} and {md_path.name}")
    if doc.get("recipe"):
        print(f"wrote {select.RECIPE}")
    return 0 if doc["status"] == "decided" else 3


def print_selection(doc: dict[str, Any]) -> None:
    from lift.repair import select

    print(select.render(doc), end="")
    if doc["status"] != "decided":
        return
    plan = doc["freeze"]
    recipe = doc.get("recipe")
    print(f"\nFreeze (Gate 3): commit {select.RECIPE} and tag it, after committing outputs/metrics/"
          "selection.md's numbers to the record:")
    if recipe:
        print(f"  # {select.RECIPE}: overall winner {recipe['overall_winner']}; frozen "
              f"{', '.join(f'{k} {v}' for k, v in sorted(recipe['frozen'].items()))}; selection.json sha256 "
              f"{recipe['selection_sha256']}")
    print(f"  {plan['tag']}")
    print("\nconfigs/variants.yaml entries of the frozen variants (seeds 1-3; seed 1's are registered "
          "already):")
    for item in plan["variants"]:
        if "entry" in item:
            print(select.yaml_entry(item["entry"]))
        else:
            print(f"  # {item['name']}: not yet ({item['pending']})")
    if plan["fit_cast"]:
        print("\nCAST for seeds 2-3: fit on train data only (GPU job lines), then register at the frozen "
              "strength:")
        for line in plan["fit_cast"]:
            print(f"  {line}")
    print("\nThen on the laptop, after lift sync pull (writes the entries above; commit "
          "configs/variants.yaml):")
    for line in plan["register"]:
        print(f"  {line}")


# --- registration ---------------------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("fit-cast", help="fit CAST's gate and steering vectors on an adapter (train/fit "
                                        "prompts; "
                                        "GPU)", description=FIT_HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="a registered adapter, e.g. adapter-aya-std-s1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.add_argument("--refit", action="store_true", help="replace an artifact fitted on something else")
    p.set_defaults(func=_fit_cast)

    p = sub.add_parser("register-repairs", help="write the repair settings' configs/variants.yaml entries "
                                                "(laptop, after lift sync pull)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--repairs", default=None, help="families, comma-separated (default wiseft,resta,cast)")
    p.add_argument("--settings", default=None,
                   help="only these settings, by tag (e.g. wiseft-l0.50,cast-m1.0): the frozen ones on seeds "
                        "2-3")
    p.add_argument("--replace", action="store_true",
                   help="replace an entry registered differently (a CAST artifact fitted again): its outputs "
                        "go stale")
    p.set_defaults(func=_register)

    p = sub.add_parser("sweep", help="the seed-1 dev sweep of the repair settings (GPU; --judge-only on the "
                                     "laptop)", description=SWEEP_HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="the seed-1 adapter, e.g. adapter-aya-std-s1")
    p.add_argument("--repairs", default=None, help="families, comma-separated (default wiseft,resta,cast)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--judge-device", default="auto", help="auto = cuda:1 if two GPUs, else cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default=None, help="default: generation.dtype")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--recheck", action="store_true", help="run every stage again (each resumes by cache key)")
    p.add_argument("--judge-only", action="store_true", help="laptop: judge every setting's utility pairs")
    p.add_argument("--no-wait", action="store_true", help="with --judge-only: submit the batches and exit 3")
    p.add_argument("--sync", action="store_true", help="with --judge-only: no Message Batches")
    p.set_defaults(func=_sweep)

    p = sub.add_parser("select", help="the selection rule on the seed-1 dev sweep, overall and per family "
                                      "(laptop)")
    p.add_argument("--adapter", required=True, help="the seed-1 adapter the sweep repaired")
    p.add_argument("--repairs", default=None, help="families, comma-separated (default wiseft,resta,cast)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.add_argument("--refreeze", action="store_true",
                   help="replace a recipe already frozen in configs/frozen_recipe.json (needs a dated "
                        "docs/decisions.md entry; refused while a recipe-frozen-* tag exists, which must be "
                        "deleted or replaced by hand first). The replaced selection.json is kept as "
                        "selection.<sha12>.json")
    p.set_defaults(func=_select)
