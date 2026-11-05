"""``lift belebele`` and ``lift heldout-loss``: log-likelihood scoring on a GPU runtime (P03). The rules
(format, tokens, keys, files, metrics) are lift.models.loglik's.

Both commands follow ``lift generate`` (lift.commands.run). They refuse unpinned revisions before
importing torch, and a precision other than ``generation.dtype`` on a GPU with native bf16. They read and
check every prompt file before loading the model, so a missing or malformed file costs no GPU time. Then
they score each cell into its resumable file and write the metrics.

A cell the run refuses is logged and skipped, and the other cells go on. That is a cell made under another
configuration or from other texts, begun under another stack, or made in another precision. At the end
each such cell is listed with its remedy, and the command exits 3, which the job runners treat as "done,
cells listed". ``--restart`` starts such a cell over instead: its file goes to outputs/.aside/ (never
synced) and an empty one takes its place, so the next push replaces the runs repo's copy. The metrics
derived from it are rewritten (or retired) before it is scored again. A current cell is left as it is,
so a job line that keeps --restart costs nothing when it runs again.

Exit codes: 0 done; 3 done, but cells were skipped; 2 a variant these commands cannot score (a wrapper
without score_logits, P05); 1 an error.
"""

from __future__ import annotations

import argparse
import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lift.commands import run
from lift.schemas import LANGS

log = logging.getLogger("lift.loglik")

# Each kind of skipped cell, and the head of its end-of-run listing (as lift.commands.run.SKIPPED).
SKIPPED = {
    "stale": ("were skipped as stale (see the errors above). To start one over, run its line below once: "
              "it sets the cell's score file aside in outputs/.aside/ (never synced) and leaves it empty, so "
              "the next lift sync push replaces the runs repo's copy, then scores the cell again"),
    "precision": ("were made in another precision and left as they are (see the errors above): score each "
                  "with its own --dtype where that precision runs, leave it out, or discard its scores with "
                  "the --restart line below"),
}
_SKIPPED_AS = {"stale": "stale", "precision": "made in another precision"}

BELEBELE_HELP = """\
Belebele accuracy by letter log-likelihood, in the frozen format of lift.models.loglik (P/Q/A-D/Answer:,
no chat template, the same for every variant). Each --langs cell of the split is scored into
outputs/scores/belebele/{variant}/belebele_{lang}_{split}.jsonl (resumable, cache-keyed). Then
outputs/metrics/belebele/{variant}_{split}.json and .md hold the accuracy of every complete language under
this configuration. Exits 3 when a cell was skipped (listed with its remedy: --restart starts it over).

example job (lift remote --job, or the Kaggle runner's JOB):
  lift sync pull
  lift belebele --variant stock --split dev --langs hi,en
"""

HELDOUT_HELP = """\
The held-out loss: the mean per-token NLL of the Hindi utility set's reference answers given their
instructions, chat template and preamble on, over exactly the tokens training supervises (the answer and
the tokens that close the turn). Nothing is truncated. Scores go to
outputs/scores/heldout_loss/{variant}/utility_hi_{split}.jsonl (resumable, cache-keyed). Metrics go to
outputs/metrics/heldout_loss/{variant}_{split}.json and .md: pooled, by source (aya, ind) and, when
stock's scores of the split exist, against stock, paired by pid. Run stock first. --split defaults to dev:
the final split is only for frozen variants (CLAUDE.md rule 1).
"""


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _spec(name: str) -> Any:
    from lift.models.variants import get_spec

    try:
        return get_spec(name)
    except KeyError as exc:
        raise SystemExit(exc.args[0]) from None


def _belebele_rows(lang: str, split: str) -> list[dict]:
    """One language's Belebele items of the split, checked before any model is loaded."""
    from lift.data import build
    from lift.models.loglik import check_belebele_rows

    try:
        rows = build.load_prompts("belebele", lang, split)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc} (a GPU job gets the built prompts with `lift sync pull`)") from None
    try:
        check_belebele_rows(rows, lang=lang, split=split)
    except ValueError as exc:
        raise SystemExit(f"belebele_{lang}: {exc}") from None
    return rows


def _utility_prompts(split: str) -> list:
    """The utility records of the split with their reference answers, checked before any model is loaded."""
    from lift.models.loglik import check_references
    from lift.prompts import load

    try:
        prompts = load("utility", "hi", split)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc} (a GPU job gets the built prompts with `lift sync pull`)") from None
    try:
        check_references(prompts)
    except ValueError as exc:
        raise SystemExit(f"utility_hi {split}: {exc}") from None
    return prompts


def _load_variant(args: argparse.Namespace, spec: Any, dtype: str) -> Any:
    """The variant, or None (logged) when these commands cannot score it (exit 2)."""
    from lift.models.loglik import check_scorable
    from lift.models.variants import build_variant

    variant = build_variant(spec, device=args.device, dtype=dtype)
    try:
        check_scorable(variant)
    except NotImplementedError as exc:
        log.error("%s", exc)
        return None
    return variant


def _skip(skipped: dict[str, dict[str, str]], cell: str, exc: Exception) -> None:
    """Log a refused cell and keep its remedy under its kind (``SKIPPED``)."""
    from lift.models.loglik import ScorePrecisionError

    kind = "precision" if isinstance(exc, ScorePrecisionError) else "stale"
    log.error("%s: %s, skipped: %s", cell, _SKIPPED_AS[kind], exc)
    command = getattr(exc, "command", None)
    if command is None:
        remedy = "no lift command line covers it (see its error above)"
    elif isinstance(exc, ScorePrecisionError):
        made = exc.dtype
        remedy = (f"made in {made}: score it with --dtype {made} where {made} runs, leave it out, or discard "
                  f"its {made} scores and score it again: {command}")
    else:
        remedy = command
    skipped.setdefault(kind, {})[cell] = remedy


def _cell(score: Callable[[], Any], *, cell: str, path: Path, restart: bool, refresh: Callable[[], Any],
          retire: Callable[[str], Any], skipped: dict[str, dict[str, str]]) -> None:
    """Score one cell. A refused cell is skipped (its remedy kept in ``skipped``) or, with ``restart``, set
    aside and scored again. Before that, the metrics derived from the file are refreshed: this run's
    (``refresh``), and, when the file was made in another precision, that precision's are retired
    (``retire(dtype)``), as they described the scores set aside."""
    from lift.models.generate import StaleOutputError
    from lift.models.loglik import ScorePrecisionError, start_over

    try:
        score()
        return
    except StaleOutputError as exc:
        if not restart:
            _skip(skipped, cell, exc)
            return
        log.warning("%s: refused, so --restart starts it over: %s", cell, exc)
        discarded = exc.dtype if isinstance(exc, ScorePrecisionError) else None
    done = start_over(path)
    if done.copy is not None:
        log.info("%s: %d record(s) set aside in %s; the file is left empty, and the next lift sync push "
                 "replaces the runs repo's copy", path, done.records, done.copy)
    if discarded:
        retire(discarded)
    refresh()
    score()


def _retire(task: str, variant: str, split: str, dtype: str) -> None:
    from lift.models.loglik import retire_metrics

    retire_metrics(task, variant, split, dtype=dtype)


def _exit(skipped: dict[str, dict[str, str]]) -> int:
    """3 when a cell was skipped (each kind listed with its cells' remedies), else 0."""
    for kind, head in SKIPPED.items():
        cells = skipped.get(kind)
        if cells:
            log.error("%d cells %s:\n%s", len(cells), head,
                      "".join(f"  {cell}: {remedy}\n" for cell, remedy in cells.items()))
    return 3 if any(skipped.values()) else 0


# --- lift belebele --------------------------------------------------------------------------------------


def _belebele(args: argparse.Namespace) -> int:
    run._logging()
    run.require_pinned("models.global")
    langs = run._names(args.langs, LANGS, ())
    dtype = run._dtype(args.dtype)
    run._refuse_other_dtype(dtype, args.device)
    spec = _spec(args.variant)
    rows = {lang: _belebele_rows(lang, args.split) for lang in langs}
    from lift.models import loglik

    variant = _load_variant(args, spec, dtype)
    if variant is None:
        return 2
    refresh = functools.partial(loglik.write_belebele_metrics, spec.name, args.split, dtype=dtype,
                                run_config=loglik.belebele_run_config(variant))
    retire = functools.partial(_retire, "belebele", spec.name, args.split)
    skipped: dict[str, dict[str, str]] = {}
    for lang in langs:
        score = functools.partial(loglik.belebele_cell, variant, rows[lang], lang=lang, split=args.split,
                                  batch_size=args.batch_size)
        _cell(score, cell=f"belebele_{lang}_{args.split}",
              path=loglik.belebele_scores_path(spec.name, lang, args.split), restart=args.restart,
              refresh=refresh, retire=retire, skipped=skipped)
    _report_belebele(refresh(), spec.name, args.split)
    return _exit(skipped)


def _report_belebele(written: Any, variant: str, split: str) -> None:
    """Accuracies with n, and the paths: numbers only."""
    if written is None:
        print(f"belebele {variant} {split}: no score file is complete under this configuration, so no "
              "metrics")
        return
    doc = written.doc
    langs = ", ".join(f"{lang} {c['accuracy']:.4f} ({c['n_correct']}/{c['n']})"
                      for lang, c in doc["langs"].items())
    print(f"belebele {variant} {split} ({doc['dtype']}, {doc['scoring']} scoring): {langs}")
    for lang, why in doc["left_out"].items():
        print(f"  left out: {lang}: {why}")
    print(f"wrote {written.json_path} and {written.md_path.name}")


# --- lift heldout-loss ----------------------------------------------------------------------------------


def _heldout(args: argparse.Namespace) -> int:
    run._logging()
    run.require_pinned("models.global")
    dtype = run._dtype(args.dtype)
    run._refuse_other_dtype(dtype, args.device)
    spec = _spec(args.variant)
    prompts = _utility_prompts(args.split)
    from lift.models import loglik

    variant = _load_variant(args, spec, dtype)
    if variant is None:
        return 2
    refresh = functools.partial(loglik.write_heldout_metrics, spec.name, args.split, dtype=dtype,
                                run_config=loglik.heldout_run_config(variant))
    score = functools.partial(loglik.heldout_cell, variant, prompts, split=args.split,
                              batch_size=args.batch_size)
    skipped: dict[str, dict[str, str]] = {}
    _cell(score, cell=f"utility_hi_{args.split}", path=loglik.heldout_scores_path(spec.name, args.split),
          restart=args.restart, refresh=refresh,
          retire=functools.partial(_retire, "heldout_nll", spec.name, args.split), skipped=skipped)
    _report_heldout(refresh(), spec.name, args.split)
    return _exit(skipped)


def _report_heldout(written: Any, variant: str, split: str) -> None:
    """The pooled and by-source NLL, the comparison with stock, and the paths: numbers only."""
    if written is None:
        print(f"held-out loss {variant} {split}: the score file is not complete under this configuration, "
              "so no metrics")
        return
    doc = written.doc
    sources = ", ".join(f"{src} {s['mean_nll']:.4f} ({s['n_rows']})" for src, s in doc["by_source"].items())
    print(f"held-out loss {variant} {split} ({doc['dtype']}): mean NLL {doc['mean_nll']:.4f} nats/token "
          f"(perplexity {doc['perplexity']:.3f}) over {doc['n_tokens']} tokens of {doc['n_rows']} "
          f"references; {sources}")
    vs = doc["vs_stock"]
    if vs is None:
        print(f"  vs stock: none ({doc['vs_stock_note']})")
    else:
        print(f"  vs stock: row mean NLL {vs['diff_row_mean_nll']:+.4f} [{vs['lo']:+.4f}, {vs['hi']:+.4f}], "
              f"p {vs['p_value']:.4f}, n {vs['n']}")
    print(f"wrote {written.json_path} and {written.md_path.name}")


# --- registration ---------------------------------------------------------------------------------------


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None,
                   help="default: generation.dtype")
    p.add_argument("--batch-size", type=_positive, default=None,
                   help="items per forward pass (default: chosen from free memory)")
    p.add_argument("--restart", action="store_true",
                   help="start over a cell this run refuses (stale, begun under another stack, or made in "
                        "another precision): its file goes to outputs/.aside/ and an empty one takes its "
                        "place; a current cell is left as it is")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("belebele", help="Belebele accuracy by letter log-likelihood (frozen format, no "
                                        "chat template; GPU)",
                       description=BELEBELE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, help="a name from configs/variants.yaml")
    p.add_argument("--split", required=True, choices=["dev", "final"])
    p.add_argument("--langs", nargs="+", required=True, help=f"hi,en or hi en; any of: {' '.join(LANGS)}")
    _common(p)
    p.set_defaults(func=_belebele)

    p = sub.add_parser("heldout-loss", help="mean per-token NLL of the Hindi utility references, chat "
                                            "template on (GPU)",
                       description=HELDOUT_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, help="a name from configs/variants.yaml")
    p.add_argument("--split", default="dev", choices=["dev", "final"], help="default: dev")
    _common(p)
    p.set_defaults(func=_heldout)
