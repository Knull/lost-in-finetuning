"""Gate 0: the pipeline check before any reported run (README, Gates). GPU only.

Checks (each PASS, FAIL or NOT RUN; numbers go to outputs/metrics/gate0.json and gate0.md):

  preamble    the chat template renders the default preamble block once (one system block, holding the
              "contextual safety mode" preamble), with one BOS, and ends with the generation prompt; the
              tokenizer transformers builds matches the checkpoint's own tokenizer.json; every dev and
              final prompt fits the context with 256 new tokens; the preamble's token length is reported.
  precision   200 dev prompts mixed across languages and sets, generated and judged in the run dtype
              (bf16 on an A100). A teacher-forced pass over prompt + output checks that every hidden
              state and logit is finite and records the max |activation| per layer. Its digest
              (``precision_digest``) names the outputs and labels the verdict is about. The reference
              run makes them in gate0/<dtype>/precision.new, which replaces gate0/<dtype>/precision
              only with a full report (gate0.json, PASS or FAIL), so a later --skip, reduced or stopped
              run never replaces the outputs gate0.json judged (``_settle``).
  fp16        only with --compare-dtype fp16, on a T4: the same 200 prompts in fp16 against the stored
              bf16 run (pulled from the runs repo): greedy output agreement and judge-label agreement.
              The fp16 run is the generator's and the judge's: its precision check also scans a local
              judge (Qwen3Guard, teacher-forced over its own input and raw output for each pair), and
              fails on any judge failure, as the dry run does.
              This decides whether T4 runs can ever be reported, so it counts only against a bf16 Gate 0
              that passed (gate0.json) for the same revision, on all of at least 200 prompts made with
              the same inputs and decoding settings, and only with the very outputs that Gate 0 judged
              (its checks.precision.digest). It records that digest and the pairs' harness and decoding,
              which the licence holds every fp16 cell to (check_dtype_licence).
  throughput  batch sizes 8, 16 and 32 at real prompt lengths: prompts/sec, tokens/sec, peak memory.
  dry_run     20 prompts x 6 languages x 3 sets through generate and judge: exactly one generation
              and one judgement per prompt, and no judge failure.
  parity      a random LoRA on Global: merged and unmerged greedy outputs match token for token in
              fp32 on 20 dev prompts (the README's merged-adapter parity, before any real adapter exists).

Decision rule (README, Gate 0): any failure means fix first. If the fp16 check fails, T4 runs are
never reported and everything reported runs in bf16 on an A100/L4. The fp16 check fails only on a
real result: a non-finite fp16 run, or a valid comparison under the agreement bar.

Verdicts. A check that raises is a FAIL with its error, and so is a model whose revision cannot be
resolved or that cannot load (the "load" check), so a report is always written. A check that cannot
run for want of a resource or an input (one GPU with no room for the judge; an fp16 run with no valid
comparison) is NOT RUN: it decides nothing and leaves the run incomplete, never failed. So is a
precision or dry-run check of the reference gate that found nothing wrong on fewer prompts than
decisions.md 27 pins (--n-precision under 200, --n-dry under 20): a quick check is no verdict.
Any failure is a full verdict, gate0.json (gate0-fp16.json for the comparison run), so a failed
re-run replaces an earlier PASS; a failure on a reduced sample is one too. A run that is only
incomplete (--skip, NOT RUN) is written as gate0-partial.json (gate0-fp16-partial.json) and never
replaces the full record. A refused configuration writes nothing: a size under 1 (--n-dry 0 would
make the dry run raise, a FAIL; --skip leaves a check out), --compare-dtype fp16 on a GPU with native
bf16, and the reference gate (no --compare-dtype) on one without (a T4), whose failed load would
replace the A100's gate0.json.

Pass thresholds that the README leaves open are pinned in docs/decisions.md (Nov 3, P02):
the fp16 run passes if every activation of the generator and of the judge is finite and its final
labels agree with bf16 on at least 95% of the 200 prompts; exact token agreement is reported, not
required (fp16 and bf16 round differently, so long greedy outputs are expected to diverge somewhere).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lift import config
from lift.models.load import DTYPES
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
FLAGS = ("too_short", "repetitive", "garbled", "wrong_lang", "code_mixed")      # of UnclearFlags
PIN_KEYS = {"global": "models.global", "base": "models.base", "judge": "judges.safety_primary"}
DECISION = {
    "pass": "Gate 0 passes: the pipeline may produce reported numbers.",
    "fail": "Fix first (README, Gate 0). Nothing is reported until every check passes.",
    "fp16_fail": "fp16 failed: T4 runs are never reported; everything reported runs in bf16 on an A100/L4.",
    "incomplete": "Incomplete: {} did not run. Gate 0 passes only when every required check has run.",
    "not_comparable": ("Not comparable: this {dtype} run could not be paired with a passing {reference} "
                       "Gate 0 of this revision ({reason}). Nothing is decided about T4 runs: pull the "
                       "{reference} Gate 0 (lift sync pull), or re-run the {reference} Gate 0 on this code "
                       "until gate0.json passes, then run this check again with the same --n-precision, "
                       "at least {n} (the default)."),
    "stub": ("Checks reached the {} stubs: run GPU jobs on a ref that has both P01 and P02 (main after both "
             "merge, or an integration branch with p01-data merged into p02-harness)."),
}
_DTYPE_SEGMENT = re.compile(rf"#(?:{'|'.join(DTYPES)})(?=#|$)")
_STUB = re.compile(r"NotImplementedError: (P\d\d)\b")


def gate0_dir(dtype: str) -> Path:
    return output_dir() / "gate0" / dtype


def not_run(reason: str) -> dict[str, Any]:
    """The record of a check that could not run (no room for the judge, no valid fp16 comparison).
    It decides nothing: ``decide`` counts it as missing, never as failed."""
    return {"pass": False, "not_run": True, "reason": reason}


def _error(exc: BaseException, prefix: str = "") -> dict[str, Any]:
    return {"pass": False, "error": f"{prefix}{type(exc).__name__}: {exc}"[:500]}


def check_status(check: dict[str, Any]) -> str:
    """PASS, FAIL or NOT RUN, as the report prints a check."""
    return "NOT RUN" if check.get("not_run") else "PASS" if check.get("pass") else "FAIL"


def verdict(result: dict[str, Any]) -> str:
    """PASS, FAIL, or INCOMPLETE when nothing failed but a required check did not run."""
    return "PASS" if result["pass"] else "FAIL" if result.get("failed") else "INCOMPLETE"


# --- preamble -------------------------------------------------------------------------------------


def check_preamble(variant: Any, prompts: Sequence[PromptRecord], *,
                   reference_tokenizer: Any = None) -> dict[str, Any]:
    """Template, BOS, reference-tokenizer and context checks on the variant's tokenizer.

    The template must render exactly one system block, the first holding the default preamble (its
    ``SAFETY_PREAMBLE_MARKER``), and end with the generation prompt (``template_stats``). The reference
    tokenizer is compared on every prompt (tokenizing is cheap), per language: a tokenizer class that
    differs only on Devanagari, Han, Arabic or Thai text must not pass on English agreement alone."""
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
    preamble_once = stats["system_blocks"] == 1 and stats["safety_preamble"]
    out["pass"] = bool(stats["bos_count"] == 1 and stats["starts_with_bos"] and has_system and same_prefix
                       and not too_long and agreement_ok and preamble_once
                       and stats["ends_with_generation_prompt"])
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
    from lift.models.load import encode_chat

    rows = [encode_chat(tokenizer, prompt) + list(tokenizer(output, add_special_tokens=False)["input_ids"])
            for prompt, output in pairs]
    return _scan_rows(model, rows, device=device)


def judge_activation_scan(judge: Any, root: Path) -> dict[str, Any] | None:
    """``activation_scan`` of the run's judge when it is a local Qwen3Guard (``judge.guard``), else None
    (an API judge or a test fake has no model here): one teacher-forced pass of the guard over each pair it
    judged under ``root``, its own input (the moderation prompt it rendered for the native prompt and the
    output, tokenized as it tokenizes it) followed by its raw output. A pair the guard never ran (a
    control-token string in it, an input too long for it) is skipped and counted in ``n_skipped``."""
    guard = getattr(judge, "guard", None)
    if getattr(guard, "model", None) is None:
        return None
    from lift.models.generate import read_generations
    from lift.prompts import texts_by_pid
    from lift.schemas import JudgementRecord

    tok, rows, skipped = guard.tokenizer, [], 0
    for gpath in sorted((root / "generations").glob("*.jsonl")):
        set_name, lang, _split = gpath.stem.rsplit("_", 2)
        texts = texts_by_pid(set_name, lang)
        juds = [JudgementRecord.model_validate(j) for j in _read(root / "judgements" / gpath.name)]
        raw = {j.key: j.raw for j in juds}
        for g in read_generations(gpath):
            if g.key not in raw:            # not judged: generate_and_judge lists it as a problem
                continue
            prompt = texts[g.pid]
            if guard._has_control_tokens(prompt) or guard._has_control_tokens(g.output):
                skipped += 1                # refused before the guard's model ran
                continue
            ids = list(tok(guard._render(prompt, g.output), add_special_tokens=False)["input_ids"])
            if len(ids) + guard.max_new_tokens > guard.context:
                skipped += 1                # too long for it: not run either
                continue
            rows.append(ids + list(tok(raw[g.key], add_special_tokens=False)["input_ids"]))
    return {**_scan_rows(guard.model, rows, device=guard.device), "n_skipped": skipped}


def _scan_rows(model: Any, rows: Sequence[Sequence[int]], *, device: str) -> dict[str, Any]:
    """``activation_scan``'s teacher-forced passes and counts, over token id rows."""
    import torch

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
        for ids in rows:
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([list(ids)], device=device), use_cache=False).logits
            finite = torch.isfinite(logits)
            logits_nonfinite += int((~finite).sum())
            if finite.any():
                logits_max = max(logits_max, float(logits[finite].abs().max()))
            del logits
    finally:
        for handle in handles:
            handle.remove()
    return {
        "n_sequences": len(rows), "hidden_nonfinite": sum(nonfinite), "logits_nonfinite": logits_nonfinite,
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
    generations and judgements per prompt, judge failures, cap hits, final labels per set and
    language ("{set}:{lang}:{label}"), and per language the unclear labels and each flag of
    ``FLAGS`` (``by_lang``), so a language the LID misreads (clean Hindi called garbled, hence
    unclear) shows. Counts only."""
    from lift.judge.pipeline import judge_file
    from lift.models.generate import generate, read_generations
    from lift.schemas import JudgementRecord

    counts: Counter = Counter()
    labels: Counter = Counter()
    by_lang: dict[str, Counter] = {}
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
        labels.update(f"{set_name}:{lang}:{j.label}" for j in juds)
        tally = by_lang.setdefault(lang, Counter())
        tally["n"] += len(juds)
        tally["unclear"] += sum(j.label == "unclear" for j in juds)
        for flag in FLAGS:
            tally[flag] += sum(getattr(j.flags, flag) for j in juds)
    return {"counts": dict(counts), "labels": dict(sorted(labels.items())),
            "by_lang": {lang: {k: by_lang[lang][k] for k in ("n", "unclear", *FLAGS)}
                        for lang in sorted(by_lang, key=LANGS.index)},
            "problems": problems}


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


def precision_digest(root: Path) -> str:
    """Which run a precision verdict is about: sha256 of the sorted (set, lang, pid, key, sha256(output),
    label) of every generation under ``root``, with its judgement's label ("missing" without one).
    gate0.json records it (checks.precision.digest) and a comparison recomputes it over the reference
    outputs on disk, so outputs other than the ones gate0.json passed (a copy not pushed or pulled with
    it) never pass for that run."""
    from lift.io import sha256_obj, sha256_text
    from lift.models.generate import read_generations
    from lift.schemas import JudgementRecord

    rows = []
    for gpath in sorted((root / "generations").glob("*.jsonl")):
        juds = [JudgementRecord.model_validate(j) for j in _read(root / "judgements" / gpath.name)]
        labels = {j.key: j.label for j in juds}
        rows.extend((g.set, g.lang, g.pid, g.key, sha256_text(g.output), labels.get(g.key, "missing"))
                    for g in read_generations(gpath))
    return sha256_obj(sorted(rows))


PROVENANCE_FIELDS = ("model_rev", "harness", "template", "decoding")


def _judge_identity(judge_id: str) -> str:
    """A judge id without its precision segment ("...@sha#bf16#strict" -> "...@sha#strict")."""
    return _DTYPE_SEGMENT.sub("", judge_id)


def compare_runs(reference_root: Path, candidate_root: Path, tokenizer: Any) -> dict[str, Any]:
    """Pair two runs of the same prompts by (set, lang, pid): exact output agreement, first token
    divergence, final-label agreement and kappa, harmful rate in each.

    The pairs are only valid if both runs saw the same prompt text, model revision, chat template,
    harness, decoding settings (``meta.decoding``: cap, stop tokens) and judge (model, revision and
    Controversial mode: only the precision may differ), so every mismatch is counted and listed in
    ``invalid``: a stale reference run (a fixed translation, a harness fix, another cap) must be re-run
    first. ``harness`` and ``decoding`` are the pairs' one setting (None when they hold several): the
    fp16 licence covers only cells made with them."""
    from lift.models.generate import read_generations
    from lift.schemas import JudgementRecord
    from lift.stats import cohens_kappa, mcnemar_exact

    def load(root: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], set[str]]:
        out, judges = {}, set()
        for gpath in sorted((root / "generations").glob("*.jsonl")):
            juds = [JudgementRecord.model_validate(j) for j in _read(root / "judgements" / gpath.name)]
            labels = {j.key: j.label for j in juds}
            judges.update(_judge_identity(j.judge) for j in juds)
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
        invalid["judge"] = len(shared)
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
    pairs = [run[k] for run in (ref, cand) for k in shared]
    return {
        "n": len(shared), "n_reference": len(ref), "n_candidate": len(cand), "missing_labels": missing,
        "invalid": invalid, **{field: _one_setting(pairs, field) for field in ("harness", "decoding")},
        "exact_output_agreement": sum(same_text) / len(shared),
        "median_first_divergence_token": sorted(diverged)[len(diverged) // 2] if diverged else None,
        "label_agreement": agree, "kappa": kappa,
        "harmful_rate_reference": sum(x == "harmful" for x in ref_labels) / len(shared),
        "harmful_rate_candidate": sum(x == "harmful" for x in cand_labels) / len(shared),
        "harmful_mcnemar_p": mc.p_value,
    }


def _one_setting(rows: Sequence[dict[str, Any]], field: str) -> Any:
    """The one value of ``field`` in ``rows``, or None when they hold several."""
    values = {row[field] for row in rows}
    return values.pop() if len(values) == 1 else None


def fp16_verdict(cmp: dict[str, Any], precision: dict[str, Any], n_precision: int, *,
                 reference: str, reference_problem: str | None) -> dict[str, Any]:
    """The fp16 check from ``compare_runs`` and the finished fp16 run's precision check (decisions.md,
    entry 27). It fails only on a real result: a non-finite fp16 run (the generator's scan, or the local
    judge's: the precision check's ``all_finite``), or a valid comparison under
    ``FP16_MIN_LABEL_AGREEMENT``. Without a valid comparison (no reference run, other prompts, fewer
    than ``N_PRECISION``, missing labels, pairs made from other inputs, or a reference whose own Gate 0
    did not pass for this revision on these very outputs: ``reference_problem``, see
    ``reference_gate_problem``) it is not run: nothing about fp16 is known, and the remedy is the
    reference run, never giving up T4 runs."""
    out = {k: v for k, v in cmp.items() if k not in ("pass", "error")}
    out["min_label_agreement"] = FP16_MIN_LABEL_AGREEMENT
    if precision.get("all_finite") is False:
        scans = {"generator": precision, "judge": precision.get("judge_scan") or {}}
        parts = [part for part, s in scans.items() if s.get("hidden_nonfinite") or s.get("logits_nonfinite")]
        where = f" ({' and '.join(parts)})" if parts else ""
        return {**out, "pass": False, "failure": f"non-finite values in the fp16 run{where}"}
    why = _not_comparable(cmp, n_precision, reference, reference_problem)
    if why:
        return {**out, "comparable": False, **not_run(why)}
    ok = cmp["label_agreement"] >= FP16_MIN_LABEL_AGREEMENT
    out.update({"comparable": True, "pass": ok})
    if not ok:
        out["failure"] = f"label agreement {cmp['label_agreement']:.3f} is under {FP16_MIN_LABEL_AGREEMENT}"
    return out


def _not_comparable(cmp: dict[str, Any], n_precision: int, reference: str,
                    reference_problem: str | None) -> str | None:
    """Why ``compare_runs`` made no valid comparison, or None. A valid one pairs all ``n_precision``
    prompts, at least ``N_PRECISION``, with a reference run whose own Gate 0 passed for this revision
    on those outputs: what check_dtype_licence asks of the licence, so a comparison it would refuse
    never reads PASS."""
    reasons = [reference_problem] if reference_problem else []
    if n_precision < N_PRECISION:
        reasons.append(f"{n_precision} prompts, under the {N_PRECISION} decisions.md 27 requires")
    if not cmp.get("n"):
        reasons.append(cmp.get("error") or f"no prompt shared with the {reference} run")
    elif cmp["n"] != n_precision:
        reasons.append(f"{cmp['n']} of {n_precision} prompts shared with the {reference} run, which has "
                       f"{cmp.get('n_reference')}")
    if cmp.get("missing_labels"):
        reasons.append(f"{cmp['missing_labels']} labels missing")
    if cmp.get("invalid"):
        reasons.append(f"pairs made from other inputs {cmp['invalid']}")
    return "; ".join(reasons) or None


def reference_gate_problem(revision: str, reference: str, *, digest: str) -> str | None:
    """Why the reference run's own Gate 0 verdict (outputs/metrics/gate0.json) cannot back a comparison
    with the reference outputs on disk, whose ``precision_digest`` is ``digest``, or None: it must be
    there, pass, be for ``revision``, and have judged those very outputs (its checks.precision.digest).
    A comparison against a failed, missing or stale reference proves nothing, nor does one against other
    outputs than those (a later --skip, reduced or stopped run never replaces them: ``_settle``), and
    check_dtype_licence would refuse its licence."""
    path = metrics_dir() / "gate0.json"
    name = f"the {reference} Gate 0 ({path.name})"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return f"{name} is missing"
    except (OSError, ValueError) as exc:          # unreadable, or not JSON
        return f"{name} cannot be read ({type(exc).__name__})"
    if _field(doc, "pass") is not True:
        return f"{name} did not pass"
    got = _field(doc, "revisions", "global")
    if got != revision:
        return f"{name} is for Global revision {got}, not {revision}"
    passed = _field(doc, "checks", "precision", "digest")
    if passed is None:
        return f"{name} records no checks.precision.digest (an older Gate 0): what it judged is unknown"
    if passed != digest:
        return (f"the {reference} precision outputs here are not the ones {path.name} passed: the two were "
                "not pushed or pulled together")
    return None


def _field(node: Any, *keys: str) -> Any:
    """``node[k1][k2]...``, or None where a level is missing or not a mapping."""
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node


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
    settle the fp16 question; the pipeline itself was checked by the reference run, whose gate0.json
    must have passed on the outputs compared with (``reference_gate_problem``)."""
    return ("precision", "fp16") if compare_dtype else tuple(c for c in CHECKS if c != "fp16")


def report_name(result: dict[str, Any]) -> str:
    """gate0, or gate0-{dtype} for a comparison run (gate0-fp16). A failure (the model did not load,
    or a required check failed) is a full verdict even when other checks did not run, so a failed
    re-run replaces an earlier PASS. A run that is only incomplete (--skip removed a required check,
    or one did not run) gets "-partial", so it never replaces the full record."""
    name = f"gate0-{result['dtype']}" if result.get("reference_dtype") else "gate0"
    failed = set(result.get("failed", ())) & {"load", *result.get("required", ())}
    return name + ("-partial" if result.get("incomplete") and not failed else "")


def pin_revisions() -> dict[str, str]:
    """Base's and the primary judge's commit shas for the pin list; one that cannot be resolved is
    "unresolved: <error>" (a metadata lookup never blocks the gate). A primary judge with no revision
    (the frontier judge, once P04's decision swaps the judges) has nothing to pin and is left out."""
    from lift.models.load import resolved_revision

    roles = ("base", "judge") if "revision" in (_local_primary_judge() or {}) else ("base",)
    out = {}
    for role in roles:
        try:
            out[role] = resolved_revision(config.get(f"{PIN_KEYS[role]}.id"),
                                          config.get(f"{PIN_KEYS[role]}.revision"))
        except Exception as exc:  # recorded; a metadata lookup never blocks the gate
            out[role] = f"unresolved: {type(exc).__name__}"
    return out


def _local_primary_judge() -> dict[str, Any] | None:
    """The judges.safety_primary entry when that judge runs here (Qwen3Guard), else None: the frontier
    judge, once P04's decision makes it primary, needs no GPU memory and has no revision."""
    from lift.judge.pipeline import judge_dir

    try:
        return config.get(PIN_KEYS["judge"]) if judge_dir("primary") == "qwen3guard" else None
    except (KeyError, ValueError):      # a broken entry: make_judge says what is wrong
        return None


def run_gate0(*, device: str = "cuda:0", judge_device: str | None = None, compare_dtype: str | None = None,
              skip: Sequence[str] = (), n_precision: int = N_PRECISION, n_dry: int = N_DRY,
              batch_sizes: Sequence[int] = BATCH_SIZES, n_throughput: int = N_THROUGHPUT,
              model_id: str | None = None, revision: str | None = None, judge: Any = None, lid: Any = None,
              max_new_tokens: int | None = None, resolve_pins: bool = True) -> dict[str, Any]:
    """Run every check not in ``skip`` and return the full result (``write_report`` saves it).

    A refused configuration raises ValueError before any check, so nothing is written: an unknown
    check, a size under 1 (``n_precision``, ``n_dry`` or a batch size: its check would raise, a FAIL
    replacing gate0.json, and with it the fp16 licence), --compare-dtype fp16 on a GPU with native bf16,
    or the reference gate (no ``compare_dtype``) on one without (a T4), whose failed load would
    otherwise be a full FAIL replacing the A100's gate0.json. An unusable ``device`` raises from torch.
    Otherwise there is always a result to write: a check that raises is recorded as a failure with
    its error and the gate goes on; a revision that cannot be resolved or a model that cannot load
    is the failed "load" check; every check draws its own prompts, so missing prompt files (or a P01
    stub) fail the checks that need them. The reference gate's precision and dry run on fewer prompts
    than decisions.md 27 pins (``n_precision`` under ``N_PRECISION``, ``n_dry`` under ``N_DRY``) are
    NOT RUN unless they failed (``_reduced``), and its precision outputs replace the stored ones only
    when the result is a full report (``_settle``).
    ``model_id``/``revision``, ``judge``, ``lid`` and ``max_new_tokens`` are for tests (a tiny local
    model, fakes, short outputs); the real gate decodes the locked 256 new tokens. ``resolve_pins``
    also resolves Base's and the primary judge's revisions for the pin list (``pin_revisions``).
    """
    from lift.models.load import device_name, resolved_revision
    from lift.models.variants import build_variant
    from lift.prompts import mixed_sample
    from lift.schemas import VariantSpec

    unknown = set(skip) - set(CHECKS)
    if unknown:
        raise ValueError(f"unknown checks {sorted(unknown)}; choose from {CHECKS}")
    sizes = {"--n-precision": [n_precision], "--n-dry": [n_dry], "--batch-sizes": list(batch_sizes)}
    small = [f"{flag} {' '.join(map(str, v)) or '(none)'}" for flag, v in sizes.items()
             if not v or min(v) < 1]
    if small:
        raise ValueError(f"{', '.join(small)}: every size must be at least 1 (leave a check out with --skip)")
    base_dtype = config.get("generation.dtype")
    dtype = compare_dtype or base_dtype
    if compare_dtype and compare_dtype == base_dtype:
        raise ValueError(f"--compare-dtype must differ from the configured dtype ({base_dtype})")
    capability = _capability(device)
    if compare_dtype == "fp16" and capability and capability[0] >= 8:
        raise ValueError(f"{device_name(device)} runs bf16 natively; the fp16 check licenses T4 runs, so run "
                         "it on the GPU it licenses (Kaggle T4 x2)")
    if not compare_dtype and capability and capability[0] < 8:
        major, minor = capability
        raise ValueError(f"{device_name(device)} has no native bf16 (compute capability {major}.{minor}): "
                         f"the reference Gate 0 runs in {base_dtype} on an A100/L4; on a T4, run lift smoke "
                         f"--compare-dtype fp16, which checks fp16 against the stored {base_dtype} run")
    to_run = [c for c in CHECKS if c not in skip and (c != "fp16" or compare_dtype)]
    model_id = model_id or config.get("models.global.id")
    result: dict[str, Any] = {
        "dtype": dtype, "reference_dtype": base_dtype if compare_dtype else None, "device": device,
        "gpu": device_name(device), "gpu_capability": ".".join(map(str, capability)) if capability else "cpu",
        "model_id": model_id, "revisions": {}, "required": list(required_checks(compare_dtype)),
        "skipped": sorted(skip), "checks": {},
    }
    checks, revisions = result["checks"], result["revisions"]
    try:
        rev = resolved_revision(model_id, revision or config.get("models.global.revision"))
        revisions["global"] = rev
        if resolve_pins:
            revisions.update(pin_revisions())
        stock = build_variant(VariantSpec(name="stock", method="stock"), device=device, dtype=dtype,
                              model_id=model_id, revision=rev)
    except Exception as exc:  # no check can run; the verdict says why
        log.exception("Gate 0 could not load %s", model_id)
        revisions.setdefault("global", f"unresolved: {type(exc).__name__}")
        checks["load"] = _error(exc)
        return decide(result)

    def attempt(name: str, fn: Any) -> None:
        t0 = time.perf_counter()
        try:
            checks[name] = fn()
        except Exception as exc:  # the gate goes on and the report says what broke
            log.exception("Gate 0 check %s raised", name)
            checks[name] = _error(exc)
            _free()
        checks[name]["seconds"] = round(time.perf_counter() - t0, 1)
        log.info("%s: %s", name, check_status(checks[name]))

    if "preamble" in to_run:
        attempt("preamble", lambda: check_preamble(stock, all_prompts(),
                                                   reference_tokenizer=reference_tokenizer(model_id, rev)))
    if "throughput" in to_run:          # before the judge loads, so peak memory is the generator's own
        attempt("throughput", lambda: throughput(
            stock, mixed_sample(n_throughput, split="dev", seed=SEED + 1), batch_sizes,
            max_new_tokens=max_new_tokens))
        _free()                         # its batches' cached blocks are room for the judge, not use

    needs_judge = {"precision", "fp16", "dry_run"} & set(to_run)
    if needs_judge and judge is None:
        judge, refusal = _primary_judge(device, judge_device, dtype)
        if refusal is not None:
            checks.update({check: dict(refusal) for check in CHECKS if check in needs_judge})
            needs_judge = set()
    if judge is not None:
        # tied_lm_head False would mean the judge is not numerically the model Qwen evaluated.
        result["judge"] = {"id": judge.judge_id, "info": getattr(judge, "info", None)}

    staging = None if compare_dtype else _fresh(gate0_dir(dtype) / "precision.new")    # see _settle
    if {"precision", "fp16"} & needs_judge:
        root = staging or gate0_dir(dtype) / "precision"

        def precision() -> dict[str, Any]:
            _fresh(root)
            sample = mixed_sample(n_precision, split="dev", seed=SEED)
            run = generate_and_judge(stock, judge, sample, root, lid=lid, max_new_tokens=max_new_tokens)
            scan = activation_scan(stock.model, stock.tokenizer, precision_pairs(root), device=device)
            out = {**scan, **run, "digest": precision_digest(root),
                   "pass": scan["all_finite"] and not run["problems"]}
            if not compare_dtype:
                return _reduced(out, len(sample), N_PRECISION)
            # The compared run is the generator's and the judge's (decisions.md 27): a local judge is scanned
            # too, all_finite (what fp16_verdict reads) needs both scans, and, as in the dry run, any judge
            # failure fails the check.
            judge_scan = judge_activation_scan(judge, root)
            if judge_scan is not None:
                out.update(judge_scan=judge_scan, all_finite=scan["all_finite"] and judge_scan["all_finite"])
            failures = run["counts"].get("judge_failures", 0)
            out["pass"] = out["all_finite"] and not run["problems"] and not failures
            if failures:
                out["failure"] = (f"{failures} judge failures (the compared run, like the dry run, "
                                  "allows none)")
            return out

        def fp16() -> dict[str, Any]:
            if "all_finite" not in checks["precision"]:      # it raised: there is no fp16 run to compare
                return {"comparable": False, **not_run("the precision run did not finish")}
            reference_root = gate0_dir(base_dtype) / "precision"
            cmp = compare_runs(reference_root, root, stock.tokenizer)
            digest = precision_digest(reference_root)
            problem = reference_gate_problem(rev, base_dtype, digest=digest)
            out = fp16_verdict(cmp, checks["precision"], n_precision, reference=base_dtype,
                               reference_problem=problem)
            return {**out, "reference_digest": digest}

        attempt("precision", precision)
        if "fp16" in needs_judge:
            attempt("fp16", fp16)

    if "dry_run" in needs_judge:

        def dry_run() -> dict[str, Any]:
            out = _fresh(gate0_dir(dtype) / "dry_run")
            sample = mixed_sample(n_dry * len(SAFETY_SETS) * len(LANGS), split="dev", seed=SEED + 2)
            run = generate_and_judge(stock, judge, sample, out, lid=lid, max_new_tokens=max_new_tokens)
            c = run["counts"]
            one_each = c["prompts"] == c["generations"] == c["judgements"] == len(sample)
            run["pass"] = bool(not run["problems"] and one_each and c["judge_failures"] == 0)
            return run if compare_dtype else _reduced(run, len(sample), N_DRY * len(SAFETY_SETS) * len(LANGS))

        attempt("dry_run", dry_run)

    if "parity" in to_run:
        del stock, judge
        _free()
        attempt("parity", lambda: check_parity(
            model_id, rev, mixed_sample(N_PARITY, split="dev", seed=SEED + 3), device=device,
            max_new_tokens=max_new_tokens))
    decide(result)
    if staging is not None:
        _settle(staging, replace=report_name(result) == "gate0")
    return result


def _reduced(check: dict[str, Any], n: int, pinned: int) -> dict[str, Any]:
    """A reference check that found nothing wrong on ``n`` prompts, fewer than the ``pinned`` that
    decisions.md 27 requires, is NOT RUN with its numbers kept: a quick --n-precision or --n-dry run is
    incomplete (gate0-partial.json), so it never replaces gate0.json, a FAIL included. A failure stays
    one: it is real on any sample."""
    if n >= pinned or not check.get("pass"):
        return check
    return {**check, **not_run(f"reduced sample: {n} of the {pinned} decisions.md 27 requires")}


def _fresh(path: Path) -> Path:
    """``path``, emptied. A check that runs starts with it: Gate 0 tests the code as it is now, so this
    dtype's earlier outputs are discarded rather than resumed (a fixed parser must re-judge). The
    reference run (another dtype), the outputs of a check that does not run and the precision outputs
    gate0.json passed (a reference run empties its staging dir instead: ``_settle``) are kept."""
    if path.exists():
        log.info("removing the previous Gate 0 outputs in %s", path)
        shutil.rmtree(path, ignore_errors=True)
    return path


def _settle(staging: Path, *, replace: bool) -> None:
    """A reference run's precision outputs, made in ``staging`` (gate0/<dtype>/precision.new), replace
    gate0/<dtype>/precision only when its report is a full verdict (gate0.json, PASS or FAIL: ``replace``),
    which records their digest; otherwise they are deleted. So a --skip, reduced or stopped run never
    replaces the outputs gate0.json passed, which the fp16 comparison pairs with (``reference_gate_problem``),
    and the job's push has no others to send in their place. A stopped run leaves ``staging`` behind,
    never synced nor read: the next reference run empties it before its own precision check."""
    if replace and staging.exists():
        staging.rename(_fresh(staging.with_name("precision")))
    elif staging.exists():
        log.info("discarding %s: the report is not a full verdict, so the outputs gate0.json passed stay",
                 staging)
        shutil.rmtree(staging, ignore_errors=True)


def _primary_judge(device: str, judge_device: str | None,
                   dtype: str) -> tuple[Any, dict[str, Any] | None]:
    """The primary judge, or None and the record of each check that needs it: not run when one GPU has
    no room for a local judge beside the generator (a resource refusal decides nothing), failed when
    the judge does not load."""
    from lift.judge.pipeline import make_judge

    try:
        problem = _judge_memory_problem(device, judge_device) if _local_primary_judge() is not None else None
        if problem:
            return None, not_run(problem)
        return make_judge("primary", device=judge_device, dtype=dtype), None
    except Exception as exc:  # each judged check fails with it
        log.exception("Gate 0 could not load the primary judge")
        return None, _error(exc, "judge: ")


def _capability(device: str) -> tuple[int, int] | None:
    if not str(device).startswith("cuda"):
        return None
    import torch

    return tuple(torch.cuda.get_device_capability(torch.device(device)))


def _judge_memory_problem(device: str, judge_device: str | None) -> str | None:
    """Why the judge cannot load next to the generator, or None: two GPUs, or one large enough.

    It reads the driver's free memory (torch.cuda.mem_get_info), which counts the blocks PyTorch's
    caching allocator keeps for freed tensors as used. So run_gate0 calls ``_free`` (gc.collect and
    torch.cuda.empty_cache) right after the throughput check, before this check: the blocks its batches
    left would be reused by the judge's own allocations, and are room, not use."""
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
    """Overall verdict with the README's decision rule. A failed check fails the gate. A required check
    that did not run (--skip) and a check recorded as not run (``not_run``: no room for the judge, no
    valid fp16 comparison) decide nothing: the run is incomplete, which never passes and is never a
    failure. "fp16 failed" needs a real fp16 result (the check's ``failure``)."""
    checks = result["checks"]
    unrun = [name for name, c in checks.items() if c.get("not_run")]
    failed = sorted(name for name, c in checks.items() if not c.get("pass") and not c.get("not_run"))
    wanted = dict.fromkeys([*result.get("required", ()), *unrun])
    incomplete = [c for c in wanted if c not in checks or c in unrun]
    result.update(failed=failed, not_run=unrun, incomplete=incomplete)
    result["pass"] = not failed and not incomplete and bool(checks)
    result["decision"] = _decision(result)
    return result


def _decision(result: dict[str, Any]) -> str:
    checks, failed, missing = result["checks"], result["failed"], result["incomplete"]
    fp16, precision = checks.get("fp16", {}), checks.get("precision", {})
    if "fp16" in failed and fp16.get("failure"):
        others = set(failed) - {"fp16"}
        if precision.get("all_finite") is False and not precision.get("problems"):
            others.discard("precision")         # the same non-finite fp16 run, not a second problem
        text = DECISION["fp16_fail"] + (" " + DECISION["fail"] if others else "")
    elif failed:
        text = DECISION["fail"]
    elif missing:
        text = " ".join([DECISION["incomplete"].format(", ".join(missing)), *_not_run_notes(result)])
    else:
        text = DECISION["pass"]
    stubs = sorted({m.group(1) for c in checks.values() if (m := _STUB.search(str(c.get("error", ""))))})
    return text + (" " + DECISION["stub"].format(", ".join(stubs)) if stubs else "")


def _not_run_notes(result: dict[str, Any]) -> list[str]:
    """Why the incomplete checks did not run, each reason once; a missing fp16 comparison says how
    to get one."""
    notes = []
    for name in result["incomplete"]:
        c = result["checks"].get(name, {})
        if name == "fp16" and c.get("comparable") is False:
            notes.append(DECISION["not_comparable"].format(
                dtype=result.get("dtype"), reference=result.get("reference_dtype"), reason=c["reason"],
                n=N_PRECISION))
        elif c.get("reason"):
            notes.append(f"Not run: {c['reason']}.")
    return list(dict.fromkeys(notes))


def write_report(result: dict[str, Any], *, name: str = "gate0") -> tuple[Path, Path]:
    """outputs/metrics/{name}.json and .md. Counts, rates, and each check's error or reason for not
    running: no prompt or output text."""
    out = ensure(metrics_dir())
    json_path, md_path = out / f"{name}.json", out / f"{name}.md"
    text = json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n"
    json_path.write_text(text, encoding="utf-8", newline="\n")
    lines = [f"# Gate 0 ({result['dtype']}, {result['gpu']})", "",
             f"**{verdict(result)}**: {result['decision']}", "",
             "| Check | Result | Key numbers |", "| --- | --- | --- |"]
    for check, c in result["checks"].items():
        lines.append(f"| {check} | {check_status(c)} | {_cell(_summary(check, c))} |")
    pins, unresolved = _pins(result)
    lines += ["", "Revisions to pin in configs/project.yaml (with a dated docs/decisions.md entry):", ""]
    lines += [f"- `{pin}`" for pin in pins] + [f"- not resolved: {key}" for key in unresolved]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return json_path, md_path


def _cell(text: str) -> str:
    """One markdown table cell: one line, pipes escaped."""
    return " ".join(text.split()).replace("|", "\\|")


def _summary(check: str, c: dict[str, Any]) -> str:
    """A check's key numbers; a check not run gives its reason, then the numbers of a reduced run."""
    if c.get("not_run"):
        ran = _summary(check, {k: v for k, v in c.items() if k != "not_run"}) if "counts" in c else ""
        return f"{c.get('reason')}; {ran}" if ran else str(c.get("reason"))
    if c.get("error"):
        return str(c["error"])[:200]
    if check == "preamble":
        return (f"system blocks {c.get('system_blocks')}, the first {c.get('system_block_tokens')} tokens "
                f"(safety preamble {c.get('safety_preamble')}), generation prompt at the end "
                f"{c.get('ends_with_generation_prompt')}, prefix {c.get('prefix_tokens')}, "
                f"BOS {c.get('bos_count')}, longest prompt {c.get('longest_prompt_tokens')} "
                f"of {c.get('context')}, {c.get('n_too_long')} too long, "
                f"reference tokenizer {c.get('reference_tokenizer_agreement')}")
    if check == "precision":
        js, failures = c.get("judge_scan"), (c.get("counts") or {}).get("judge_failures")
        judge = ""
        if js:
            judge = (f"; judge non-finite hidden {js.get('hidden_nonfinite')}, logits "
                     f"{js.get('logits_nonfinite')}, fp16-overflow layers {js.get('fp16_overflow_layers')}")
        judge += f"; {failures} judge failures" if failures else ""
        return (f"non-finite hidden {c.get('hidden_nonfinite')}, logits {c.get('logits_nonfinite')}, "
                f"max |h| {max(c.get('max_abs_per_layer') or [0])}, "
                f"fp16-overflow layers {c.get('fp16_overflow_layers')}{judge}{_lang_shares(c)}")
    if check == "fp16":
        numbers = (f"n {c.get('n')}, label agreement {c.get('label_agreement')}, kappa {c.get('kappa')}, "
                   f"exact output {c.get('exact_output_agreement')}")
        return f"{c['failure']}; {numbers}" if c.get("failure") else numbers
    if check == "throughput":
        return "; ".join(f"bs {r['batch_size']}: {r['prompts_per_sec']} p/s, {r['tokens_per_sec']} tok/s, "
                         f"{r['peak_gb']} GB" for r in c.get("rows", []))
    if check == "dry_run":
        k = c.get("counts", {})
        return (f"{k.get('generations')} generations, {k.get('judgements')} judgements, "
                f"{k.get('judge_failures')} judge failures, {k.get('cap_hits')} cap hits{_lang_shares(c)}")
    if check == "parity":
        return f"{c.get('n_match')}/{c.get('n')} identical in {c.get('dtype')}"
    return ""


def _lang_shares(c: dict[str, Any]) -> str:
    """The report's "; garbled/unclear by language: en 0.00/0.01, hi 0.12/0.15, ...": shares of each
    language's judgements, so a language the LID misreads stands out."""
    shares = [f"{lang} {v['garbled'] / v['n']:.2f}/{v['unclear'] / v['n']:.2f}"
              for lang, v in (c.get("by_lang") or {}).items() if v.get("n")]
    return "; garbled/unclear by language: " + ", ".join(shares) if shares else ""


def _pins(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    """("models.global.revision: <sha>", ...) for the resolved revisions, and the config keys of the
    unresolved ones with their error: those are named, never offered as a value to pin."""
    pins, unresolved = [], []
    for role, rev in result["revisions"].items():
        key = PIN_KEYS.get(role, role)
        if str(rev).startswith("unresolved"):
            unresolved.append(f"{key} ({str(rev).removeprefix('unresolved: ')})")
        else:
            pins.append(f"{key}.revision: {rev}")
    return pins, unresolved


def pin_hint(result: dict[str, Any]) -> str:
    """The config lines to pin the resolved revisions (a person makes the change, with a dated entry).
    A revision that could not be resolved is named on its own line, never printed as a value to pin,
    and a primary judge with nothing to pin (the frontier judge) is not listed."""
    pins, unresolved = _pins(result)
    lines = ["configs/project.yaml:", *(f"  {pin}" for pin in pins)]
    if unresolved:
        lines.append("not resolved (resolve by hand, or re-run Gate 0): " + ", ".join(unresolved))
    return "\n".join(lines)
