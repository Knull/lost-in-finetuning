"""``lift publish datasets``: the public logs dataset, the gated raw completions and the translated
prompt sets, staged under ``outputs/publish/datasets/``.

Logs (public). One row per primary-judge judgement: variant, method, seed, pid, language, set, split,
the final label, the judge's own label, the unclear flags, the cap hit, the second judge's label and any
human labels where they exist, the sha256 of the output judged, and ``refusal_text``. Plus the aggregate
tables (final_report.json as JSON and one CSV per table), selection.json, the redacted AuditKIT records
(never ``*.full.json``, and each checked to be redacted) and a small sample of refusals for the Space.

A text is public only when all of these hold (otherwise it is withheld and counted by reason):

- the primary label is ``refusal``, and no judge (label, judge label, or a Qwen3Guard native Safety of
  Unsafe or Controversial, in any judgements directory, current or not) and no human ever labelled that
  generation harmful;
- the judgement names the output now under its key (``output_sha``): a regenerated output was never
  judged (``texts_stale``);
- the generation stopped at EOS (``hit_cap`` false; a capped text may refuse, then comply);
- its language is not exploratory for the harmful and red-team sets (the judge's precision there has no
  human check), and not a back-translation language at all (below);
- it holds no 60-character window of an output labelled harmful (``lift.publish.preflight``).

Nothing is staged unless the inputs are complete (``PublishError`` names the remedy): the human labels
are readable (every round that was drawn or scored has its keys and every sheet imported, every label
maps to a key), every generation of a safety cell has a primary judgement and every judgement its
generation, the second judge covers its key list (validation/keys/second_judge/all.txt plus every
validated key) on the current outputs, and no Message Batch is still open. ``check_public`` then re-reads
every staged row (harmful anywhere, or not the output its judgement covered, stops the publish) and the
whole staged folder is scanned like a public upload (``preflight.scan_folder``).

Back-translation (P08-R1-12): P07 judges NLLB back-translations of ar and th outputs, and where those
verdicts live is not agreed yet. Verdicts stored under outputs/judgements/{dir}/{variant}/ with the
generation key are already read as harmful labels; until the contract is fixed, every ar and th text is
withheld (``texts_backtranslation``).

Raw completions (gated, manual approval). Every generations, judgements and utility-verdict file, and
the full AuditKIT records, as they are in ``outputs/``. Human labels (with annotator ids) stay private.

Translated prompt sets (public). Every prompt record we translated (TranslationInfo.engine one of
``lift.data.translate.ENGINE_NAMES``), with its English source and the native spot-check fields
(checked, edited, note). PolyRefuse's own translations and the English sources are not republished.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from lift import config, paths
from lift.io import read_jsonl, sha256_text, write_jsonl
from lift.publish import PublishError, preflight, render, settings
from lift.publish.hub import Upload
from lift.publish.inputs import Facts, report_path, selection_path, shown

HARMFUL_NATIVE = {"unsafe", "controversial"}
SAFETY_SETS = ("harmful", "easy_benign", "hard_benign", "redteam")  # judged cells (validation.second_judge)
EXPLORATORY_WITHHELD_SETS = ("harmful", "redteam")
WITHHELD = (
    "texts_not_refusal",
    "texts_stale",
    "texts_cap_hit",
    "texts_exploratory",
    "texts_backtranslation",
    "texts_overlap",
)
PULL = "run `lift sync pull` first (outputs/ and validation/ come only from the private runs repo)"
EXAMPLES_PER_CELL = 20
LICENCES = {  # README, Licences; docs/decisions.md items 14, 45
    "harmful": ("PolyRefuse (mainlp/Multilingual-Refusal)", "Apache-2.0"),
    "easy_benign": ("PolyRefuse harmless prompts, from Stanford Alpaca", "CC-BY-NC-4.0"),
    "hard_benign": ("OR-Bench-Hard-1K (bench-llm/or-bench)", "CC-BY-4.0"),
}


def stage_root() -> Path:
    return paths.output_dir() / "publish" / "datasets"


def _fresh(path: Path) -> Path:
    """An empty staging folder (only ever under outputs/publish)."""
    root = (paths.output_dir() / "publish").resolve()
    if root not in path.resolve().parents:
        raise PublishError(f"refusing to clear {path}: not under {root}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


# ------------------------------------------------------------------------------------- reading outputs


def _cells(root: Path) -> Iterator[tuple[str, str, Path]]:
    """(variant, cell stem, path) for every JSONL under root/{variant}/."""
    if not root.is_dir():
        return
    for vdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(vdir.glob("*.jsonl")):
            yield vdir.name, f.stem, f


def judge_dirs() -> list[str]:
    root = paths.judgements_dir()
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def judgement_files() -> list[Path]:
    """Every judgements file on this machine, of every judge."""
    return [f for jd in judge_dirs() for _, _, f in _cells(paths.judgements_dir() / jd)]


def role_dir(role: str) -> str:
    from lift.judge.pipeline import judge_dir

    return judge_dir(role)


def _is_harmful(rec: dict) -> bool:
    native = rec.get("native") or {}
    return (
        "harmful" in (rec.get("label"), rec.get("judge_label"))
        or str(native.get("safety") or "").strip().lower() in HARMFUL_NATIVE
    )


def _round_no(name: str, pattern: str) -> int | None:
    m = re.fullmatch(pattern, name)
    return int(m.group(1)) if m else None


def expected_rounds(*, judged: bool) -> set[int]:
    """The validation rounds whose labels must be readable: round 1 once anything is judged (Gate 1 needs
    it), and every round with keys, a draw or a score here (validation/keys, outputs/metrics/validation)."""
    from lift.validation.store import keys_dir
    from lift.validation.store import metrics_dir as validation_metrics

    rounds = {1} if judged else set()
    for folder, pattern in (
        (keys_dir(), r"(?:round|draw)(\d+)\.json"),
        (validation_metrics(), r"round(\d+)\.json"),
    ):
        for f in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            n = _round_no(f.name, pattern)
            if n is not None:
                rounds.add(n)
    if (validation_metrics() / "gate1.json").is_file():
        rounds.add(1)
    return rounds


def round_items(*, judged: bool | None = None) -> tuple[dict[str, str], list[str]]:
    """({item_id: generation key} of every validation round, problems). A problem is a round that was
    drawn or scored whose keys file, or one of whose sheets' imported labels, is not here."""
    from lift.validation.store import keys_dir, labels_dir

    judged = bool(judgement_files()) if judged is None else judged
    items: dict[str, str] = {}
    problems: list[str] = []
    for n in sorted(expected_rounds(judged=judged)):
        path = keys_dir() / f"round{n}.json"
        if not path.is_file():
            problems.append(f"{shown(path)} is missing")
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            problems.append(f"{shown(path)} is not valid JSON")
            continue
        if isinstance(doc, dict):
            entries = list(doc.items())
        elif isinstance(doc, list):
            entries = [(r.get("item_id") if isinstance(r, dict) else None, r) for r in doc]
        else:
            problems.append(f"{shown(path)} is neither a mapping nor a list of items")
            continue
        sheets: set[str] = set()
        for item_id, rec in entries:
            key = (rec.get("key") or rec.get("generation_key")) if isinstance(rec, dict) else None
            if not item_id or not isinstance(key, str):
                problems.append(f"{shown(path)}: an item without an item_id or a generation key")
                continue
            items[str(item_id)] = key
            sheets.add(str(rec.get("sheet") or f"round{n}_{rec.get('lang')}"))
        for sheet in sorted(sheets):
            if not (labels_dir() / f"{sheet}.jsonl").is_file():
                problems.append(
                    f"sheet {sheet} has no imported labels ({shown(labels_dir() / f'{sheet}.jsonl')}): "
                    f"import it (lift validate import --round {n} ...)"
                )
    return items, problems


def human_labels() -> dict[str, list[str]]:
    """{generation key: [human labels]} from validation/labels/round*.jsonl and the round keys. Refuses
    (PublishError) when they cannot all be read: an empty result would make the "no human labelled it
    harmful" check pass everything."""
    from lift.validation.store import labels_dir

    items, problems = round_items()
    out: dict[str, list[str]] = defaultdict(list)
    unmapped = 0
    for f in sorted(labels_dir().glob("round*.jsonl")) if labels_dir().is_dir() else []:
        for row in read_jsonl(f):
            key = items.get(str(row.get("item_id")))
            if key is None:
                unmapped += 1
            elif row.get("label"):
                out[key].append(str(row["label"]))
    if unmapped:
        problems.append(f"{unmapped} human labels name an item_id that no validation/keys/round*.json maps")
    if problems:
        raise PublishError(
            "the human labels cannot all be read, so a text a reader labelled harmful could go public: "
            + "; ".join(problems)
            + f". {PULL}"
        )
    return dict(out)


def harmful_outputs(humans: dict[str, list[str]] | None = None) -> tuple[set[str], set[str], set[str]]:
    """(every generation key any judge, in any judgements directory, or any human labelled harmful; the
    output texts of those generations now here; the keys among them with no generation here)."""
    humans = human_labels() if humans is None else humans
    keys = {k for k, labels in humans.items() if "harmful" in labels}
    for f in judgement_files():
        keys.update(r["key"] for r in read_jsonl(f) if _is_harmful(r))
    texts, found = set(), set()
    for _, _, f in _cells(paths.generations_dir()):
        for r in read_jsonl(f):
            if r.get("key") in keys:
                found.add(r["key"])
                if (r.get("output") or "").strip():
                    texts.add(r["output"].strip())
    return keys, texts, keys - found


def harmful_record(humans: dict[str, list[str]] | None = None) -> tuple[set[str], set[str]]:
    """Every generation key any judge or human labelled harmful, and those generations' output texts."""
    keys, texts, _ = harmful_outputs(humans)
    return keys, texts


def open_batches() -> list[str]:
    """Message Batches the frontier judge's state files list as submitted and neither collected nor
    abandoned (lift.judge.frontier): their verdicts, a "harmful" among them, are not in yet."""
    root = paths.output_dir() / "judge_cache" / "frontier"
    out: list[str] = []
    for f in sorted(root.glob("*.batches.jsonl")) if root.is_dir() else []:
        submitted, closed = set(), set()
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                if row.get("event") == "submitted":
                    submitted.add(str(row.get("batch_id")))
                elif row.get("event") in ("collected", "abandoned"):
                    closed.add(str(row.get("batch_id")))
        out += sorted(submitted - closed)
    return out


def second_judge_keys(round_keys: set[str]) -> tuple[set[str] | None, Path]:
    """The keys the second judge must cover: validation/keys/second_judge/all.txt (P04) plus every
    validated key; None when the list is not here."""
    from lift.judge.pipeline import read_keys_file
    from lift.validation.second_judge import second_judge_dir

    path = second_judge_dir() / "all.txt"
    if not path.is_file():
        return None, path
    try:
        listed = set(read_keys_file(path))
    except ValueError as exc:
        raise PublishError(f"{shown(path)}: {exc}") from None
    return listed | round_keys, path


# ------------------------------------------------------------------------------------------- logs


def _flags(rec: dict) -> dict[str, bool]:
    flags = rec.get("flags") or {}
    return {k: bool(flags.get(k)) for k in ("too_short", "repetitive", "garbled", "wrong_lang", "code_mixed")}


def _current_shas() -> dict[str, str]:
    """{generation key: sha256 of the output now under it} over every generations file here."""
    return {
        r["key"]: sha256_text(r.get("output") or "")
        for _, _, f in _cells(paths.generations_dir())
        for r in read_jsonl(f)
        if "key" in r
    }


def _withheld_reason(rec: dict, gen: dict, *, harmful_keys: set[str], windows: set[str]) -> str | None:
    """Why this judgement's output may not be public (a ``texts_*`` count), None when it may."""
    out = gen.get("output") or ""
    if rec.get("label") != "refusal" or rec["key"] in harmful_keys:
        return "texts_not_refusal"
    if not rec.get("output_sha") or rec["output_sha"] != sha256_text(out):
        return "texts_stale"  # judged on another output than the one now under the key (or unknown)
    if gen.get("hit_cap") is not False:
        return "texts_cap_hit"  # stopped at the cap (or not recorded): may refuse, then comply
    if rec.get("lang") in config.get("languages.exploratory") and rec.get("set") in EXPLORATORY_WITHHELD_SETS:
        return "texts_exploratory"
    if rec.get("lang") in (config.get("back_translation.langs") or []):
        # TODO(P08-R1-12): read P07's back-translation verdicts once their path and key are agreed, and
        # publish a text only when none of them is harmful. Until then no ar/th text is public.
        return "texts_backtranslation"
    if preflight.has_window(out, windows):
        return "texts_overlap"  # shares a 60-character run with an output labelled harmful
    return None


def log_rows(*, humans: dict[str, list[str]] | None = None) -> tuple[list[dict], dict[str, int]]:
    """The public rows (with refusal texts only where allowed) and counts of what was withheld. Refuses
    (PublishError) unless the record is complete: the module docstring lists the checks."""
    from lift.validation.outputs import variant_info
    from lift.validation.second_judge import parse_generations_name

    humans = human_labels() if humans is None else humans
    harmful_keys, harmful_texts = harmful_record(humans)
    windows = preflight.windows_of(harmful_texts)
    primary, second = role_dir("primary"), role_dir("second")
    problems: list[str] = []
    current = _current_shas()
    second_current: dict[str, str] = {}  # labels of second judgements of the output now under the key
    for _, _, f in _cells(paths.judgements_dir() / second):
        for r in read_jsonl(f):
            if r.get("output_sha") and r["output_sha"] == current.get(r["key"]):
                second_current[r["key"]] = r["label"]
    primary_cells = {(v, stem): f for v, stem, f in _cells(paths.judgements_dir() / primary)}
    for variant, stem, gen_path in _cells(paths.generations_dir()):
        parsed = parse_generations_name(stem)
        if parsed is None or parsed[0] not in SAFETY_SETS:
            continue
        if (variant, stem) not in primary_cells and variant_info(variant) is None:
            continue  # not a study variant, and never judged: nothing of it is published
        judged = (
            {r["key"] for r in read_jsonl(primary_cells[(variant, stem)])}
            if (variant, stem) in primary_cells
            else set()
        )
        unjudged = sum(1 for r in read_jsonl(gen_path) if r.get("key") not in judged)
        if unjudged:
            problems.append(
                f"{variant}/{stem}: {unjudged} generations have no primary judgement (every prompt stays in "
                "the denominator: judge the cell, or set it aside with its lift restart line)"
            )
    rows: list[dict] = []
    counts: Counter = Counter()
    no_generation = 0
    for (variant, stem), f in primary_cells.items():
        gen_path = paths.generations_dir() / variant / f"{stem}.jsonl"
        gens = {r["key"]: r for r in read_jsonl(gen_path)} if gen_path.is_file() else {}
        info = variant_info(variant)
        for rec in read_jsonl(f):
            gen = gens.get(rec["key"])
            if gen is None:
                no_generation += 1
                gen = {}
            text = None
            if (gen.get("output") or "").strip():
                reason = _withheld_reason(rec, gen, harmful_keys=harmful_keys, windows=windows)
                if reason is None:
                    text = gen["output"]
                    counts["refusal_texts"] += 1
                else:
                    counts["texts_withheld"] += 1
                    counts[reason] += 1
            rows.append(
                {
                    "key": rec["key"],
                    "variant": variant,
                    "method": info.method if info else None,
                    "seed": info.seed if info else None,
                    "pid": rec["pid"],
                    "lang": rec["lang"],
                    "set": rec["set"],
                    "split": rec["split"],
                    "judge": rec.get("judge"),
                    "label": rec["label"],
                    "judge_label": rec.get("judge_label"),
                    **_flags(rec),
                    "hit_cap": gen.get("hit_cap"),
                    "n_new_tokens": gen.get("n_new_tokens"),
                    "output_sha": rec.get("output_sha"),
                    "judged_current_output": bool(gen) and rec.get("output_sha") == current.get(rec["key"]),
                    "second_judge_label": second_current.get(rec["key"]),
                    "human_labels": humans.get(rec["key"], []),
                    "refusal_text": text,
                }
            )
            counts["rows"] += 1
    if no_generation:
        problems.append(
            f"{no_generation} primary judgements have no generation here (outputs/generations): {PULL}, or "
            "make the cell again with its lift restart line"
        )
    if rows:
        round_keys = set(round_items()[0].values())
        required, list_path = second_judge_keys(round_keys)
        if required is None:
            problems.append(
                f"the second judge's key list {shown(list_path)} is not here (lift validate "
                f"second-judge-sample, or {PULL})"
            )
        else:
            lacking = sorted(k for k in required if k in current and k not in second_current)
            if lacking:
                problems.append(
                    f"{len(lacking)} keys the second judge must cover have no judgement of their current "
                    f"output (e.g. {', '.join(k[:12] for k in lacking[:3])}): run the lift judge --judge "
                    "second lines of lift validate second-judge-sample"
                )
        pending = open_batches()
        if pending:
            problems.append(
                f"{len(pending)} Message Batches are still open ({', '.join(pending[:3])}): rerun their lift "
                "judge line to collect them (LIFT_ABANDON_BATCHES for one a person deleted)"
            )
    if problems:
        raise PublishError("the logs are not complete, nothing was staged:\n  " + "\n  ".join(problems))
    return rows, dict(counts)


def check_public(
    folder: Path,
    harmful_keys: set[str],
    harmful_texts: set[str],
    *,
    judged: dict[str, str | None] | None = None,
    current: dict[str, str] | None = None,
) -> int:
    """Re-read every staged JSONL row: no text may come from (or equal) a generation labelled harmful by
    a judge or a human, only refusals carry text, and a text must be the output now under its key and the
    output its primary judgement covered (``judged``: {key: output_sha}; ``current``: {key: sha256 of the
    output}; both read from outputs/ when None). Returns the number of texts checked."""
    if judged is None:
        judged = {
            r["key"]: r.get("output_sha")
            for _, _, f in _cells(paths.judgements_dir() / role_dir("primary"))
            for r in read_jsonl(f)
        }
    current = _current_shas() if current is None else current
    checked = 0
    for f in sorted(folder.rglob("*.jsonl")):
        for i, row in enumerate(read_jsonl(f), 1):
            text = row.get("refusal_text") or row.get("text_shown")
            if not text:
                continue
            checked += 1
            key = row.get("key")
            bad = (
                key in harmful_keys
                or text.strip() in harmful_texts
                or row.get("label") != "refusal"
                or "harmful" in (row.get("human_labels") or [])
                or row.get("second_judge_label") == "harmful"
                or row.get("judge_label") == "harmful"
            )
            if bad:
                raise PublishError(
                    f"{shown(f)} line {i}: a public text belongs to a generation labelled harmful "
                    f"(key {str(key)[:12]}): nothing was published"
                )
            sha = sha256_text(text)
            if not (row.get("output_sha") == sha and judged.get(key) == sha and current.get(key) == sha):
                raise PublishError(
                    f"{shown(f)} line {i}: a public text is not the output its primary judgement covered "
                    f"(key {str(key)[:12]}): nothing was published"
                )
    return checked


def _write_csv(path: Path, columns: list[dict], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([c["key"] for c in columns])
        for row in rows:
            w.writerow(
                [
                    json.dumps(v) if isinstance(v, (dict, list)) else ("" if v is None else v)
                    for v in (row.get(c["key"]) for c in columns)
                ]
            )


def _aggregates(folder: Path, facts: Facts) -> list[str]:
    copied = []
    report = facts.report()
    if report is not None:
        render.write_text(
            folder / "aggregates" / "final_report.json", json.dumps(report, indent=1, ensure_ascii=False)
        )
        for name, spec in (report.get("tables") or {}).items():
            _write_csv(folder / "aggregates" / f"{name}.csv", spec["columns"], spec["rows"])
        copied.append("final_report.json")
    if facts.selection() is not None:
        shutil.copyfile(selection_path(), folder / "aggregates" / "selection.json")
        copied.append("selection.json")
    audit_root = paths.metrics_dir() / "auditkit"
    for f in sorted(audit_root.rglob("*.json")) if audit_root.is_dir() else []:
        if f.name.endswith(".full.json") or f.stat().st_size == 0:
            continue
        try:
            problem = preflight.auditkit_problem(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            problem = "it is not valid JSON"
        if problem:
            raise PublishError(
                f"{shown(f)} would go public but is not a redacted AuditKIT record ({problem}): write it "
                "again with lift.audit's default text='redacted' (a full record is *.full.json, gated)"
            )
        dest = folder / "auditkit" / f.relative_to(audit_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        copied.append(f"auditkit/{f.relative_to(audit_root).as_posix()}")
    return copied


def build_logs(facts: Facts | None = None) -> Upload:
    facts = facts or Facts()
    repo = settings.repo("logs")
    folder = _fresh(stage_root() / "logs")
    humans = human_labels()
    keys, texts = harmful_record(humans)
    rows, counts = log_rows(humans=humans)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    for variant, vrows in by_variant.items():
        write_jsonl(folder / "labels" / f"{variant}.jsonl", vrows)
    examples: dict[tuple, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["variant"], r["lang"], r["set"], r["pid"])):
        cell = (r["variant"], r["lang"], r["set"])
        if r["refusal_text"] and len(examples[cell]) < EXAMPLES_PER_CELL:
            examples[cell].append(
                {
                    k: r[k]
                    for k in (
                        "key",
                        "variant",
                        "method",
                        "pid",
                        "lang",
                        "set",
                        "split",
                        "label",
                        "judge_label",
                        "second_judge_label",
                        "human_labels",
                        "output_sha",
                        "refusal_text",
                    )
                }
            )
    if examples:
        write_jsonl(
            folder / "examples" / "refusals.jsonl", [e for cell in sorted(examples) for e in examples[cell]]
        )
    copied = _aggregates(folder, facts)
    checked = check_public(folder, keys, texts)
    card = logs_card(
        facts, counts=counts, variants=sorted(by_variant), copied=copied, raw=settings.repo("raw")
    )
    render.write_text(folder / "README.md", card)
    facts.write_sources(stage_root() / "logs.sources.json")
    found = preflight.scan_folder(folder, preflight.windows_of(texts))
    if found:
        raise PublishError(
            "the staged logs dataset failed the public-text scan (matched text is never shown), nothing was "
            "published:\n  " + "\n  ".join(str(x) for x in found[:50])
        )
    reasons = ", ".join(
        f"{counts[k]} {k.removeprefix('texts_').replace('_', ' ')}" for k in sorted(counts) if k in WITHHELD
    )
    notes = [
        f"{counts.get('rows', 0)} labelled rows, {checked} refusal texts checked against "
        f"{len(keys)} generations labelled harmful by a judge or a human "
        f"({sum(len(v) for v in humans.values())} human labels read); "
        f"{counts.get('texts_withheld', 0)} texts withheld" + (f" ({reasons})" if reasons else "")
    ]
    if not rows:
        notes.append("no judgements in outputs/judgements yet: only the card and aggregates are staged")
    return Upload(repo=repo, folder=folder, title="logs dataset", notes=notes)


def logs_card(
    facts: Facts, *, counts: dict[str, int], variants: list[str], copied: list[str], raw: settings.RepoSpec
) -> str:
    meta = {
        "license": "cc-by-nc-4.0",
        "language": list(config.get("languages.all")),
        "pretty_name": "Lost in Fine-Tuning: evaluation logs",
        "tags": ["safety", "multilingual", "evaluation", "tiny-aya", "lexsi-hackathon"],
        "configs": [
            {"config_name": "labels", "data_files": "labels/*.jsonl", "default": True},
            {"config_name": "refusal_examples", "data_files": "examples/refusals.jsonl"},
        ],
    }
    body = render.fill(
        render.template("logs_card.md"),
        {
            "raw_repo": raw.url,
            "n_rows": f"{counts.get('rows', 0):,}",
            "n_texts": f"{counts.get('refusal_texts', 0):,}",
            "variants": ", ".join(f"`{v}`" for v in variants) or "none yet",
            "aggregates": "\n".join(f"- `{c}`" for c in copied) or f"- {facts_pending(facts)}",
            "verdicts": verdict_lines(facts),
        },
    )
    return render.frontmatter(meta) + "\n" + body


def facts_pending(facts: Facts) -> str:
    from lift.publish.inputs import P07, pending

    return pending("the aggregate tables", shown(report_path()), P07)


def verdict_lines(facts: Facts) -> str:
    """One line per adapter verdict from final_report.json, or a pending line."""
    from lift.publish.inputs import P07

    verdicts = facts.raw(report_path(), "/verdicts")
    if verdicts is None:
        return facts.report_value("/verdicts", "text", what="the verdict per adapter")
    if not verdicts:
        return "_No verdict recorded._"
    lines = []
    for i, v in enumerate(verdicts):
        verdict = facts.value(report_path(), f"/verdicts/{i}/verdict", "text", what="verdict", stage=P07)
        adapter = facts.value(report_path(), f"/verdicts/{i}/adapter", "text", what="adapter", stage=P07)
        # Free text from P07 goes public too, so it is recorded in the sources like every number.
        reason = (
            facts.value(report_path(), f"/verdicts/{i}/reason", "text", what="verdict reason", stage=P07)
            if v.get("reason")
            else ""
        )
        recipe = (
            facts.value(report_path(), f"/verdicts/{i}/recipe", "text", what="recipe", stage=P07)
            if v.get("recipe")
            else "standard"
        )
        lines.append(f"- `{adapter}` ({recipe} recipe): **{verdict}**. {reason}".rstrip())
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- raw

RAW_TREES = ("generations", "judgements", "utility")


def build_raw(facts: Facts | None = None) -> Upload:
    repo = settings.repo("raw")
    folder = _fresh(stage_root() / "raw")
    n = 0
    for tree in RAW_TREES:
        src = paths.output_dir() / tree
        for f in sorted(src.rglob("*.jsonl")) if src.is_dir() else []:
            dest = folder / tree / f.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            n += 1
    audit_root = paths.metrics_dir() / "auditkit"
    for f in sorted(audit_root.rglob("*.full.json")) if audit_root.is_dir() else []:
        if f.stat().st_size:
            dest = folder / "auditkit" / f.relative_to(audit_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            n += 1
    meta = {
        "license": "cc-by-nc-4.0",
        "language": list(config.get("languages.all")),
        "pretty_name": "Lost in Fine-Tuning: raw completions (gated)",
        "tags": ["safety", "red-teaming", "harmful-content", "tiny-aya"],
        "extra_gated_heading": "Access to raw model completions, including harmful ones",
        "extra_gated_prompt": render.template("gate_raw.txt").strip(),
        "extra_gated_fields": {
            "Name": "text",
            "Affiliation": "text",
            "Intended use": "text",
            "I will not redistribute harmful completions": "checkbox",
        },
    }
    body = render.fill(
        render.template("raw_card.md"), {"logs_repo": settings.repo("logs").url, "n_files": str(n)}
    )
    render.write_text(folder / "README.md", render.frontmatter(meta) + "\n" + body)
    return Upload(
        repo=repo,
        folder=folder,
        title="raw completions (gated, manual approval)",
        notes=[f"{n} files from outputs/{{{','.join(RAW_TREES)}}} and the full AuditKIT records"],
    )


# ------------------------------------------------------------------------------------------- prompts


def _ours() -> set[str]:
    from lift.data.translate import ENGINE_NAMES

    return set(ENGINE_NAMES.values())


def build_prompts(facts: Facts | None = None) -> Upload:
    repo = settings.repo("prompts")
    folder = _fresh(stage_root() / "prompts")
    ours = _ours()
    table = []
    src = paths.prompts_dir()
    for f in sorted(src.glob("*.jsonl")) if src.is_dir() else []:
        rows = [r for r in read_jsonl(f) if (r.get("translation") or {}).get("engine") in ours]
        if not rows:
            continue
        set_name, lang = rows[0]["set"], rows[0]["lang"]
        keep = ("pid", "lang", "set", "split", "text", "text_en", "source", "category", "translation")
        write_jsonl(folder / set_name / f"{lang}.jsonl", [{k: r.get(k) for k in keep} for r in rows])
        tr = [r["translation"] for r in rows]
        table.append(
            [
                set_name,
                lang,
                f"{len(rows):,}",
                ", ".join(sorted({t["engine"] for t in tr})),
                f"{sum(bool(t.get('checked')) for t in tr):,}",
                f"{sum(bool(t.get('edited')) for t in tr):,}",
                LICENCES.get(set_name, ("?", "see the source"))[1],
            ]
        )
    sets = sorted({row[0] for row in table})
    meta = {
        "license": "other",
        "license_name": "mixed-see-card",
        "language": sorted({row[1] for row in table}) or list(config.get("languages.all")),
        "pretty_name": "Lost in Fine-Tuning: translated safety prompt sets",
        "tags": ["safety", "multilingual", "over-refusal", "machine-translated"],
        "configs": [{"config_name": s, "data_files": f"{s}/*.jsonl"} for s in sets],
    }
    body = render.fill(
        render.template("prompts_card.md"),
        {
            "files": render.markdown_table(
                ["set", "lang", "prompts", "engine", "native-checked", "edited", "licence"], table
            )
            if table
            else "_No translated prompt files in data/prompts yet._",
            "licences": "\n".join(f"- `{s}`: {LICENCES[s][0]}, {LICENCES[s][1]}." for s in LICENCES),
        },
    )
    render.write_text(folder / "README.md", render.frontmatter(meta) + "\n" + body)
    return Upload(
        repo=repo,
        folder=folder,
        title="translated prompt sets",
        notes=[f"{len(table)} files"] if table else ["no translated prompt files yet (lift data build)"],
    )


BUILDERS = {"logs": build_logs, "raw": build_raw, "prompts": build_prompts}


def build(only: list[str] | None = None) -> list[Upload]:
    facts = Facts()
    out = []
    for name in only or list(BUILDERS):
        if name not in BUILDERS:
            raise PublishError(f"unknown dataset {name!r} (one of {', '.join(BUILDERS)})")
        out.append(BUILDERS[name](facts))
    return out
