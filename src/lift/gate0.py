"""Gate 0: the pipeline check before any reported run (README, Gates). GPU only.

Checks (each PASS or FAIL; numbers go to outputs/metrics/gate0.json and gate0.md):

  preamble    the chat template renders the default preamble block once, with one BOS; the tokenizer
              transformers builds matches the checkpoint's own tokenizer.json; every dev and final
              prompt fits the context with 256 new tokens; the preamble's token length is reported.
  precision   200 dev prompts mixed across languages and sets, generated and judged in the run dtype
              (bf16 on an A100). A teacher-forced pass over prompt + output checks that every hidden
              state and logit is finite and records the max |activation| per layer.
  fp16        only with --compare-dtype fp16, on a T4: the same 200 prompts in fp16 against the stored
              bf16 run (pulled from the runs repo): greedy output agreement and judge-label agreement.
              This decides whether T4 runs can ever be reported.
  throughput  batch sizes 8, 16 and 32 at real prompt lengths: prompts/sec, tokens/sec, peak memory.
  dry_run     20 prompts x 6 languages x 3 sets through generate and judge: exactly one generation
              and one judgement per prompt, and no judge failure.
  parity      a random LoRA on Global: merged and unmerged greedy outputs match token for token in
              fp32 on 20 dev prompts (the README's merged-adapter parity, before any real adapter exists).

Decision rule (README, Gate 0): any failure means fix first. If the fp16 check fails, T4 runs are
never reported and everything reported runs in bf16 on an A100/L4.

Pass thresholds that the README leaves open are pinned in docs/decisions.md (Nov 3, P02):
the fp16 run passes if every activation is finite and its final labels agree with bf16 on at least
95% of the 200 prompts; exact token agreement is reported, not required (fp16 and bf16 round
differently, so long greedy outputs are expected to diverge somewhere).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import ensure, metrics_dir, output_dir
from lift.prompts import SAFETY_SETS
from lift.schemas import LANGS, PromptRecord

log = logging.getLogger(__name__)

SEED = 20261104
N_PRECISION = 200
N_DRY = 20
N_PARITY = 20
N_THROUGHPUT = 64
BATCH_SIZES = (8, 16, 32)
FP16_MAX = 65504.0
FP16_MIN_LABEL_AGREEMENT = 0.95
CHECKS = ("preamble", "precision", "fp16", "throughput", "dry_run", "parity")
DECISION = {
    "pass": "Gate 0 passes: the pipeline may produce reported numbers.",
    "fail": "Fix first (README, Gate 0). Nothing is reported until every check passes.",
    "fp16_fail": "fp16 failed: T4 runs are never reported; everything reported runs in bf16 on an A100/L4.",
    "incomplete": "Incomplete: {} did not run. Gate 0 passes only when every required check has run.",
}


def gate0_dir(dtype: str) -> Path:
    return output_dir() / "gate0" / dtype


# --- preamble -------------------------------------------------------------------------------------


def check_preamble(variant: Any, prompts: Sequence[PromptRecord], *,
                   reference_tokenizer: Any = None) -> dict[str, Any]:
    """Template, BOS, reference-tokenizer and context checks on the variant's tokenizer.

    The reference tokenizer is compared on every prompt (tokenizing is cheap), per language: a
    tokenizer class that differs only on Devanagari, Han, Arabic or Thai text must not pass on
    English agreement alone."""
    from lift.models.load import chat_text, encode_chat, template_stats

    tok = variant.tokenizer
    stats = template_stats(tok)
    a, b = chat_text(tok, "first probe"), chat_text(tok, "second probe, longer")
    same_prefix = a.split("first probe")[0] == b.split("second probe, longer")[0]
    context = getattr(variant.hf_model.config, "max_position_embeddings", None)
    cap = int(config.get("generation.max_new_tokens"))
    ids = {(p.set, p.lang, p.pid): encode_chat(tok, p.text) for p in prompts}
    lengths = {f"{s}/{lang}/{pid}": len(v) for (s, lang, pid), v in ids.items()}
    longest = max(lengths.values()) if lengths else 0
    too_long = sorted(k for k, n in lengths.items() if context and n + cap > context)
    out: dict[str, Any] = {
        **stats, "context": context, "n_prompts_checked": len(lengths), "longest_prompt_tokens": longest,
        "n_too_long": len(too_long), "too_long_examples": too_long[:5],
        "preamble_identical_across_prompts": same_prefix,
    }
    agreement_ok = True
    if reference_tokenizer is not None:
        per_lang: dict[str, list[bool]] = {}
        for p in prompts:
            ref = list(reference_tokenizer.encode(chat_text(tok, p.text), add_special_tokens=False).ids)
            per_lang.setdefault(p.lang, []).append(ref == ids[(p.set, p.lang, p.pid)])
        for probe in ("first probe", "second probe, longer", "2026-11-03: 1,234.5"):
            ref = list(reference_tokenizer.encode(chat_text(tok, probe), add_special_tokens=False).ids)
            per_lang.setdefault("probes", []).append(ref == encode_chat(tok, probe))
        out["reference_tokenizer_agreement"] = {lang: sum(v) / len(v) for lang, v in sorted(per_lang.items())}
        agreement_ok = all(share == 1.0 for share in out["reference_tokenizer_agreement"].values())
    has_system = (stats.get("system_block_tokens") or 0) > 0
    out["pass"] = bool(stats["bos_count"] == 1 and stats["starts_with_bos"] and has_system and same_prefix
                       and not too_long and agreement_ok)
    return out


def reference_tokenizer(model_id: str, revision: str) -> Any:
    """The checkpoint's own tokenizer.json, loaded with the tokenizers library (no transformers code),
    to catch a transformers tokenizer class that silently tokenizes differently."""
    from tokenizers import Tokenizer

    local = Path(model_id) / "tokenizer.json"
    if local.is_file():
        return Tokenizer.from_file(str(local))
    from huggingface_hub import hf_hub_download

    return Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json", revision=revision))


# --- precision --------------------------------------------------------------------------------------


def activation_scan(model: Any, tokenizer: Any, pairs: Sequence[tuple[str, str]], *,
                    device: str) -> dict[str, Any]:
    """Teacher-forced forward passes (batch 1, no padding) over chat(prompt) + output in the model's
    dtype: non-finite counts and the max |value| of every decoder layer's output (forward hooks on
    model.model.layers; output_hidden_states would give the final layer after the norm) and of the
    logits."""
    import torch

    from lift.models.load import encode_chat

    layers = model.model.layers
    max_abs = [0.0] * len(layers)
    nonfinite = [0] * len(layers)
    logits_nonfinite = 0
    logits_max = 0.0

    def watch(i: int) -> Any:
        def hook(_module: Any, _args: Any, output: Any) -> None:
            h = output[0] if isinstance(output, tuple) else output
            finite = torch.isfinite(h)
            nonfinite[i] += int((~finite).sum())
            if finite.any():
                max_abs[i] = max(max_abs[i], float(h[finite].abs().max()))
        return hook

    handles = [layer.register_forward_hook(watch(i)) for i, layer in enumerate(layers)]
    try:
        for prompt, output in pairs:
            out_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
            ids = encode_chat(tokenizer, prompt) + list(out_ids)
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits
            finite = torch.isfinite(logits)
            logits_nonfinite += int((~finite).sum())
            if finite.any():
                logits_max = max(logits_max, float(logits[finite].abs().max()))
            del logits
    finally:
        for handle in handles:
            handle.remove()
    return {
        "n_sequences": len(pairs), "hidden_nonfinite": sum(nonfinite), "logits_nonfinite": logits_nonfinite,
        "max_abs_per_layer": [round(v, 3) for v in max_abs], "logits_max_abs": round(logits_max, 3),
        "fp16_overflow_layers": [i for i, v in enumerate(max_abs) if v > FP16_MAX],
        "all_finite": sum(nonfinite) == 0 and logits_nonfinite == 0,
    }


def _by_cell(prompts: Sequence[PromptRecord]) -> dict[tuple[str, str], list[PromptRecord]]:
    cells: dict[tuple[str, str], list[PromptRecord]] = {}
    for p in prompts:
        cells.setdefault((p.set, p.lang), []).append(p)
    return cells


def generate_and_judge(variant: Any, judge: Any, prompts: Sequence[PromptRecord], root: Path, *,
                       lid: Any = None, batch_size: int | None = None,
                       max_new_tokens: int | None = None) -> dict[str, Any]:
    """Generate and judge ``prompts`` under ``root`` (one file per set and language), then count:
    generations and judgements per prompt, judge failures, labels, cap hits."""
    from lift.judge.pipeline import judge_file
    from lift.models.generate import generate, read_generations
    from lift.schemas import JudgementRecord

    counts: Counter = Counter()
    labels: Counter = Counter()
    problems: list[str] = []
    for (set_name, lang), cell in sorted(_by_cell(prompts).items()):
        gen_path = root / "generations" / f"{set_name}_{lang}_dev.jsonl"
        jud_path = root / "judgements" / f"{set_name}_{lang}_dev.jsonl"
        generate(variant, cell, out_path=gen_path, batch_size=batch_size, max_new_tokens=max_new_tokens)
        texts = {p.pid: p.text for p in cell}
        judge_file(gen_path, judge=judge, lid=lid, out_path=jud_path, prompts_by_pid=texts)
        gens = read_generations(gen_path)
        juds = [JudgementRecord.model_validate(r) for r in _read(jud_path)]
        gen_keys = Counter(g.key for g in gens)
        jud_keys = Counter(j.key for j in juds)
        if sorted(g.pid for g in gens) != sorted(texts):
            problems.append(f"{set_name}_{lang}: generations do not cover the prompts one to one")
        if jud_keys != gen_keys:
            problems.append(f"{set_name}_{lang}: judgements do not match generations one to one")
        counts["prompts"] += len(cell)
        counts["generations"] += len(gens)
        counts["judgements"] += len(juds)
        counts["cap_hits"] += sum(g.hit_cap for g in gens)
        counts["judge_failures"] += sum("judge_failure" in j.native for j in juds)
        counts["wrong_lang"] += sum(j.flags.wrong_lang for j in juds)
        labels.update(f"{set_name}:{j.label}" for j in juds)
    return {"counts": dict(counts), "labels": dict(sorted(labels.items())), "problems": problems}


def _read(path: Path) -> list[dict]:
    from lift.io import read_jsonl

    return list(read_jsonl(path)) if path.exists() else []


def precision_pairs(root: Path) -> list[tuple[str, str]]:
    """(prompt text, output) for every generation under ``root``, for the activation scan."""
    from lift.models.generate import read_generations
    from lift.prompts import texts_by_pid

    pairs = []
    for path in sorted((root / "generations").glob("*.jsonl")):
        set_name, lang, _split = path.stem.rsplit("_", 2)
        texts = texts_by_pid(set_name, lang)
        pairs.extend((texts[g.pid], g.output) for g in read_generations(path))
    return pairs


PROVENANCE_FIELDS = ("model_rev", "harness", "template")


def compare_runs(reference_root: Path, candidate_root: Path, tokenizer: Any) -> dict[str, Any]:
    """Pair two runs of the same prompts by (set, lang, pid): exact output agreement, first token
    divergence, final-label agreement and kappa, harmful rate in each.

    The pairs are only valid if both runs saw the same prompt text, model revision, chat template,
    harness and judge revision (only the dtype may differ), so every mismatch is counted and listed
    in ``invalid``: a stale reference run (a fixed translation, a harness fix) must be re-run first."""
    from lift.models.generate import read_generations
    from lift.schemas import JudgementRecord
    from lift.stats import cohens_kappa, mcnemar_exact

    def load(root: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], set[str]]:
        out, judges = {}, set()
        for gpath in sorted((root / "generations").glob("*.jsonl")):
            juds = [JudgementRecord.model_validate(j) for j in _read(root / "judgements" / gpath.name)]
            labels = {j.key: j.label for j in juds}
            judges.update(j.judge.split("#", 1)[0] for j in juds)
            for g in read_generations(gpath):
                out[(g.set, g.lang, g.pid)] = {
                    "output": g.output, "label": labels.get(g.key, "missing"), "prompt_sha": g.prompt_sha,
                    **{f: g.meta.get(f) for f in PROVENANCE_FIELDS}}
        return out, judges

    (ref, ref_judges), (cand, cand_judges) = load(reference_root), load(candidate_root)
    shared = sorted(set(ref) & set(cand))
    if not shared:
        return {"n": 0, "n_reference": len(ref), "n_candidate": len(cand), "pass": False,
                "error": f"no shared prompts; is the reference run under {reference_root}?"}
    invalid: dict[str, int] = {}
    for k in shared:
        for field in ("prompt_sha", *PROVENANCE_FIELDS):
            a, b = ref[k][field], cand[k][field]
            if a is None or b is None or a != b:
                invalid[field] = invalid.get(field, 0) + 1
    if ref_judges != cand_judges:
        invalid["judge_revision"] = len(shared)
    same_text = [ref[k]["output"] == cand[k]["output"] for k in shared]
    divergence = []
    for k in shared:
        a = tokenizer(ref[k]["output"], add_special_tokens=False)["input_ids"]
        b = tokenizer(cand[k]["output"], add_special_tokens=False)["input_ids"]
        divergence.append(next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                               None if len(a) == len(b) else min(len(a), len(b))))
    ref_labels = [ref[k]["label"] for k in shared]
    cand_labels = [cand[k]["label"] for k in shared]
    missing = sum(x == "missing" for x in ref_labels + cand_labels)
    agree = sum(x == y for x, y in zip(ref_labels, cand_labels, strict=True)) / len(shared)
    kappa = cohens_kappa(ref_labels, cand_labels) if not missing else None
    mc = mcnemar_exact([x == "harmful" for x in cand_labels], [x == "harmful" for x in ref_labels])
    diverged = [d for d in divergence if d is not None]
    return {
        "n": len(shared), "n_reference": len(ref), "n_candidate": len(cand), "missing_labels": missing,
        "invalid": invalid,
        "exact_output_agreement": sum(same_text) / len(shared),
        "median_first_divergence_token": sorted(diverged)[len(diverged) // 2] if diverged else None,
        "label_agreement": agree, "kappa": kappa,
        "harmful_rate_reference": sum(x == "harmful" for x in ref_labels) / len(shared),
        "harmful_rate_candidate": sum(x == "harmful" for x in cand_labels) / len(shared),
        "harmful_mcnemar_p": mc.p_value,
    }


# --- throughput -------------------------------------------------------------------------------------


def throughput(variant: Any, prompts: Sequence[PromptRecord], batch_sizes: Sequence[int] = BATCH_SIZES, *,
               max_new_tokens: int | None = None) -> dict[str, Any]:
    """Generate the same real-length prompts at each batch size into a scratch dir (deleted after).

    Also reports, for information, the share of prompts whose greedy output is identical at every
    batch size: on a GPU, padding and batch composition can flip near-tied tokens in bf16.
    """
    from lift.models.generate import generate, read_generations

    rows = []
    outputs: dict[tuple[str, str, str], set[str]] = {}
    cuda = str(variant.device).startswith("cuda")
    if cuda:
        import torch
    for bs in batch_sizes:
        tmp = Path(tempfile.mkdtemp(prefix=f"lift-gate0-bs{bs}-"))
        try:
            if cuda:
                torch.cuda.synchronize(variant.device)
                torch.cuda.reset_peak_memory_stats(variant.device)
            t0 = time.perf_counter()
            path = generate(variant, prompts, out_path=tmp / "g.jsonl", batch_size=bs, resume=False,
                            manifest=False, max_new_tokens=max_new_tokens)
            if cuda:
                torch.cuda.synchronize(variant.device)
            seconds = time.perf_counter() - t0
            recs = read_generations(path)
            for r in recs:
                outputs.setdefault((r.set, r.lang, r.pid), set()).add(r.output)
            tokens = sum(int(r.meta.get("n_generated", 0)) for r in recs)
            rows.append({
                "batch_size": bs, "n": len(recs), "seconds": round(seconds, 2),
                "prompts_per_sec": round(len(recs) / seconds, 3),
                "tokens_per_sec": round(tokens / seconds, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated(variant.device) / 1e9, 2) if cuda else None,
                "effective_batch_sizes": sorted({int(r.meta["batch_size"]) for r in recs}),
                "cap_hit_rate": sum(r.hit_cap for r in recs) / len(recs),
            })
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    best = max(rows, key=lambda r: r["prompts_per_sec"]) if rows else None
    same = sum(len(v) == 1 for v in outputs.values())
    return {"rows": rows, "best_batch_size": best["batch_size"] if best else None, "pass": bool(rows),
            "batch_invariant_share": same / len(outputs) if outputs else None}


# --- parity -----------------------------------------------------------------------------------------


def random_lora(model_id: str, revision: str, out_dir: Path, *, device: str, seed: int = SEED,
                scale: float = 0.01) -> Path:
    """Save a LoRA adapter with small random weights (r 16, alpha 32, the training targets) on
    Global, so merge parity can be checked before any real adapter exists."""
    import torch
    from peft import LoraConfig, get_peft_model

    from lift.models.load import load_model

    base = load_model(model_id, revision, dtype="fp32", device=device)
    lora = config.get("standard.lora", name="train")
    torch.manual_seed(seed)
    peft_model = get_peft_model(base, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=0.0, target_modules=list(lora["target_modules"]),
        init_lora_weights=False, task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.mul_(scale)
    peft_model.save_pretrained(out_dir)
    del peft_model, base
    return out_dir


def check_parity(model_id: str, revision: str, prompts: Sequence[PromptRecord], *, device: str,
                 max_new_tokens: int | None = None) -> dict[str, Any]:
    from lift.train.merge import parity_check

    tmp = Path(tempfile.mkdtemp(prefix="lift-gate0-lora-"))
    try:
        adapter = random_lora(model_id, revision, tmp, device=device)
        _free()
        out = parity_check(adapter, prompts=[p.text for p in prompts], device=device, dtype="fp32",
                           base_id=model_id, base_revision=revision, max_new_tokens=max_new_tokens)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        _free()
    out["pass"] = bool(out["all_match"])
    return out


def _free() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- the whole gate ---------------------------------------------------------------------------------


def all_prompts(sets: Sequence[str] = SAFETY_SETS, langs: Sequence[str] = LANGS) -> list[PromptRecord]:
    """Every dev and final prompt of the safety sets, plus Hindi red-teaming and utility if built."""
    from lift.prompts import load

    out = []
    for set_name in sets:
        for lang in langs:
            out.extend(p for p in load(set_name, lang) if p.split in ("dev", "final"))
    for set_name in ("redteam", "utility"):
        try:
            out.extend(p for p in load(set_name, "hi") if p.split in ("dev", "final"))
        except FileNotFoundError:
            log.info("no %s_hi prompts yet; the context check skips them", set_name)
    return out


JUDGE_GB = 9.0               # Qwen3Guard-Gen-4B in bf16 or fp16, as in lift.commands.run
JUDGE_HEADROOM_GB = 4.0      # its KV cache and the activation scan's logits


def required_checks(compare_dtype: str | None) -> tuple[str, ...]:
    """The checks a run must complete before its verdict counts. A comparison run (a T4) only has to
    settle the fp16 question; the pipeline itself was checked by the reference run."""
    return ("precision", "fp16") if compare_dtype else tuple(c for c in CHECKS if c != "fp16")


def report_name(result: dict[str, Any]) -> str:
    """gate0 or gate0-fp16, with "-partial" when a required check did not run, so an incomplete run
    never replaces the full record."""
    name = f"gate0-{result['dtype']}" if result.get("reference_dtype") else "gate0"
    return name + ("-partial" if result.get("incomplete") else "")


def run_gate0(*, device: str = "cuda:0", judge_device: str | None = None, compare_dtype: str | None = None,
              skip: Sequence[str] = (), n_precision: int = N_PRECISION, n_dry: int = N_DRY,
              batch_sizes: Sequence[int] = BATCH_SIZES, n_throughput: int = N_THROUGHPUT,
              model_id: str | None = None, revision: str | None = None, judge: Any = None, lid: Any = None,
              max_new_tokens: int | None = None, resolve_pins: bool = True) -> dict[str, Any]:
    """Run every check not in ``skip`` and return the full result (``write_report`` saves it).

    A check that raises is recorded as a failure with its error and the gate goes on, so the report
    is always written. ``model_id``/``revision``, ``judge``, ``lid`` and ``max_new_tokens`` are for
    tests (a tiny local model, fakes, short outputs); the real gate decodes the locked 256 new tokens.
    ``resolve_pins`` also resolves Base's and the judge's revisions for the pin list.
    """
    from lift.judge.pipeline import make_judge
    from lift.models.load import device_name, resolved_revision
    from lift.models.variants import build_variant
    from lift.prompts import mixed_sample
    from lift.schemas import VariantSpec

    unknown = set(skip) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown checks {sorted(unknown)}; choose from {CHECKS}")
    base_dtype = config.get("generation.dtype")
    dtype = compare_dtype or base_dtype
    if compare_dtype and compare_dtype == base_dtype:
        raise ValueError(f"--compare-dtype must differ from the configured dtype ({base_dtype})")
    capability = _capability(device)
    if compare_dtype == "fp16" and capability and capability[0] >= 8:
        raise ValueError(f"{device_name(device)} runs bf16 natively; the fp16 check licenses T4 runs, so run "
                         "it on the GPU it licenses (Kaggle T4 x2)")
    to_run = [c for c in CHECKS if c not in skip and (c != "fp16" or compare_dtype)]
    model_id = model_id or config.get("models.global.id")
    rev = resolved_revision(model_id, revision or config.get("models.global.revision"))
    result: dict[str, Any] = {
        "dtype": dtype, "reference_dtype": base_dtype if compare_dtype else None, "device": device,
        "gpu": device_name(device), "gpu_capability": ".".join(map(str, capability)) if capability else "cpu",
        "model_id": model_id, "revisions": {"global": rev}, "required": list(required_checks(compare_dtype)),
        "skipped": sorted(skip), "checks": {},
    }
    for role, key in (("base", "models.base"), ("judge", "judges.safety_primary")) if resolve_pins else ():
        try:
            ident, pinned = config.get(f"{key}.id"), config.get(f"{key}.revision")
            result["revisions"][role] = resolved_revision(ident, pinned)
        except Exception as exc:  # recorded; a metadata lookup never blocks the gate
            result["revisions"][role] = f"unresolved: {type(exc).__name__}"
    checks = result["checks"]

    def attempt(name: str, fn: Any) -> None:
        t0 = time.perf_counter()
        try:
            checks[name] = fn()
        except Exception as exc:  # the gate goes on and the report says what broke
            log.exception("Gate 0 check %s raised", name)
            checks[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            _free()
        checks[name]["seconds"] = round(time.perf_counter() - t0, 1)
        log.info("%s: %s", name, "PASS" if checks[name].get("pass") else "FAIL")

    # Gate 0 tests the code as it is now: this dtype's earlier outputs of a check that runs now are
    # discarded rather than resumed (a fixed parser must re-judge). The reference dtype's run is kept.
    stale = {"precision": "precision", "fp16": "precision", "dry_run": "dry_run"}
    for check in to_run:
        if check in stale and (gate0_dir(dtype) / stale[check]).exists():
            log.info("removing the previous Gate 0 outputs in %s", gate0_dir(dtype) / stale[check])
            shutil.rmtree(gate0_dir(dtype) / stale[check], ignore_errors=True)

    try:
        stock = build_variant(VariantSpec(name="stock", method="stock"), device=device, dtype=dtype,
                              model_id=model_id, revision=rev)
    except Exception as exc:
        checks["load"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        return decide(result)

    if "preamble" in to_run:
        attempt("preamble", lambda: check_preamble(stock, all_prompts(),
                                                   reference_tokenizer=reference_tokenizer(model_id, rev)))
    if "throughput" in to_run:          # before the judge loads, so peak memory is the generator's own
        sample = mixed_sample(n_throughput, split="dev", seed=SEED + 1)
        attempt("throughput", lambda: throughput(stock, sample, batch_sizes, max_new_tokens=max_new_tokens))

    needs_judge = {"precision", "fp16", "dry_run"} & set(to_run)
    if needs_judge and judge is None:
        problem = _judge_memory_problem(device, judge_device)
        if problem:
            for check in sorted(needs_judge):
                checks[check] = {"pass": False, "error": problem}
            needs_judge = set()
        else:
            try:
                judge = make_judge("primary", device=judge_device, dtype=dtype)
            except Exception as exc:
                for check in sorted(needs_judge):
                    checks[check] = {"pass": False, "error": f"judge: {type(exc).__name__}: {exc}"[:500]}
                needs_judge = set()
    if judge is not None:
        # tied_lm_head False would mean the judge is not numerically the model Qwen evaluated.
        result["judge"] = {"id": judge.judge_id, "info": getattr(judge, "info", None)}

    if {"precision", "fp16"} & needs_judge:
        root = gate0_dir(dtype) / "precision"

        def precision() -> dict[str, Any]:
            run = generate_and_judge(stock, judge, mixed_sample(n_precision, split="dev", seed=SEED), root,
                                     lid=lid, max_new_tokens=max_new_tokens)
            scan = activation_scan(stock.model, stock.tokenizer, precision_pairs(root), device=device)
            return {**scan, **run, "pass": scan["all_finite"] and not run["problems"]}

        attempt("precision", precision)
        if "fp16" in needs_judge:

            def fp16() -> dict[str, Any]:
                cmp = compare_runs(gate0_dir(base_dtype) / "precision", root, stock.tokenizer)
                cmp["min_label_agreement"] = FP16_MIN_LABEL_AGREEMENT
                cmp["pass"] = bool(cmp.get("n") == n_precision and checks["precision"].get("all_finite")
                                   and not cmp.get("missing_labels") and not cmp.get("invalid")
                                   and cmp.get("label_agreement", 0.0) >= FP16_MIN_LABEL_AGREEMENT)
                return cmp

            attempt("fp16", fp16)

    if "dry_run" in needs_judge:

        def dry_run() -> dict[str, Any]:
            sample = mixed_sample(n_dry * len(SAFETY_SETS) * len(LANGS), split="dev", seed=SEED + 2)
            run = generate_and_judge(stock, judge, sample, gate0_dir(dtype) / "dry_run", lid=lid,
                                     max_new_tokens=max_new_tokens)
            c = run["counts"]
            one_each = c["prompts"] == c["generations"] == c["judgements"] == len(sample)
            run["pass"] = bool(not run["problems"] and one_each and c["judge_failures"] == 0)
            return run

        attempt("dry_run", dry_run)

    if "parity" in to_run:
        del stock, judge
        _free()
        sample = mixed_sample(N_PARITY, split="dev", seed=SEED + 3)
        attempt("parity", lambda: check_parity(model_id, rev, sample, device=device,
                                               max_new_tokens=max_new_tokens))
    return decide(result)


def _capability(device: str) -> tuple[int, int] | None:
    if not str(device).startswith("cuda"):
        return None
    import torch

    return tuple(torch.cuda.get_device_capability(torch.device(device)))


def _judge_memory_problem(device: str, judge_device: str | None) -> str | None:
    """Why the judge cannot load next to the generator, or None: two GPUs, or one large enough."""
    if not str(device).startswith("cuda") or (judge_device and judge_device != device):
        return None
    import torch

    free = torch.cuda.mem_get_info(torch.device(device))[0] / 1e9
    if free >= JUDGE_GB + JUDGE_HEADROOM_GB:
        return None
    return (f"only {free:.1f} GB free on {device} after loading the generator; the judge needs about "
            f"{JUDGE_GB + JUDGE_HEADROOM_GB:.0f} GB beside it. Run Gate 0 on two GPUs (Kaggle T4 x2, "
            "--judge-device cuda:1) or on a larger GPU")


def decide(result: dict[str, Any]) -> dict[str, Any]:
    """Overall verdict with the README's decision rule. A required check that did not run makes the
    run incomplete, which never passes."""
    checks = result["checks"]
    failed = sorted(name for name, c in checks.items() if not c.get("pass"))
    missing = [c for c in result.get("required", ()) if c not in checks]
    result["failed"] = failed
    result["incomplete"] = missing
    result["pass"] = not failed and not missing and bool(checks)
    if "fp16" in failed:
        others = set(failed) - {"fp16"}
        result["decision"] = DECISION["fp16_fail"] + (" " + DECISION["fail"] if others else "")
    elif failed:
        result["decision"] = DECISION["fail"]
    elif missing:
        result["decision"] = DECISION["incomplete"].format(", ".join(missing))
    else:
        result["decision"] = DECISION["pass"]
    return result


def write_report(result: dict[str, Any], *, name: str = "gate0") -> tuple[Path, Path]:
    """outputs/metrics/{name}.json and .md. Counts and rates only: no prompt or output text."""
    out = ensure(metrics_dir())
    json_path, md_path = out / f"{name}.json", out / f"{name}.md"
    text = json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n"
    json_path.write_text(text, encoding="utf-8", newline="\n")
    verdict = "PASS" if result["pass"] else "FAIL"
    lines = [f"# Gate 0 ({result['dtype']}, {result['gpu']})", "", f"**{verdict}**: {result['decision']}", "",
             "| Check | Result | Key numbers |", "| --- | --- | --- |"]
    for check, c in result["checks"].items():
        lines.append(f"| {check} | {'PASS' if c.get('pass') else 'FAIL'} | {_summary(check, c)} |")
    lines += ["", "Revisions (pin these in configs/project.yaml with a docs/decisions.md entry):", ""]
    lines += [f"- {k}: `{v}`" for k, v in result["revisions"].items()]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return json_path, md_path


def _summary(check: str, c: dict[str, Any]) -> str:
    if check == "preamble":
        return (f"system block {c.get('system_block_tokens')} tokens, prefix {c.get('prefix_tokens')}, "
                f"BOS {c.get('bos_count')}, longest prompt {c.get('longest_prompt_tokens')} "
                f"of {c.get('context')}, {c.get('n_too_long')} too long, "
                f"reference tokenizer {c.get('reference_tokenizer_agreement')}")
    if check == "precision":
        return (f"non-finite hidden {c.get('hidden_nonfinite')}, logits {c.get('logits_nonfinite')}, "
                f"max |h| {max(c.get('max_abs_per_layer') or [0])}, "
                f"fp16-overflow layers {c.get('fp16_overflow_layers')}")
    if check == "fp16":
        return (f"n {c.get('n')}, label agreement {c.get('label_agreement')}, kappa {c.get('kappa')}, "
                f"exact output {c.get('exact_output_agreement')}")
    if check == "throughput":
        return "; ".join(f"bs {r['batch_size']}: {r['prompts_per_sec']} p/s, {r['tokens_per_sec']} tok/s, "
                         f"{r['peak_gb']} GB" for r in c.get("rows", []))
    if check == "dry_run":
        k = c.get("counts", {})
        return (f"{k.get('generations')} generations, {k.get('judgements')} judgements, "
                f"{k.get('judge_failures')} judge failures, {k.get('cap_hits')} cap hits")
    if check == "parity":
        return f"{c.get('n_match')}/{c.get('n')} identical in {c.get('dtype')}"
    return ""


def pin_hint(result: dict[str, Any]) -> str:
    """The config lines to pin the resolved revisions (a person makes the change, with a dated entry)."""
    revs = result["revisions"]
    return "\n".join([
        "configs/project.yaml:",
        f"  models.global.revision: {revs.get('global')}",
        f"  models.base.revision: {revs.get('base')}",
        f"  judges.safety_primary.revision: {revs.get('judge')}",
    ])

