"""``lift publish datasets``: the public logs dataset, the gated raw completions and the translated
prompt sets, staged under ``outputs/publish/datasets/``.

Logs (public). One row per primary-judge judgement: variant, method, seed, pid, language, set, split,
the final label, the judge's own label, the unclear flags, the cap hit, the second judge's label and any
human labels where they exist, and ``refusal_text``. Plus the aggregate tables (final_report.json as JSON
and one CSV per table), selection.json, the redacted AuditKIT records (never ``*.full.json``) and a small
sample of refusals for the Space. A text is public only when the primary label is ``refusal`` and no
judge (label, judge label, or a Qwen3Guard native Safety of Unsafe or Controversial) and no human ever
labelled that generation harmful; ``check_public`` then re-reads every staged row and stops the publish
if any text belongs to a generation labelled harmful anywhere, or equals one's output.

Raw completions (gated, manual approval). Every generations, judgements and utility-verdict file, and
the full AuditKIT records, as they are in ``outputs/``. Human labels (with annotator ids) stay private.

Translated prompt sets (public). Every prompt record we translated (TranslationInfo.engine one of
``lift.data.translate.ENGINE_NAMES``), with its English source and the native spot-check fields
(checked, edited, note). PolyRefuse's own translations and the English sources are not republished.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from lift import config, paths
from lift.io import read_jsonl, write_jsonl
from lift.publish import PublishError, render, settings
from lift.publish.hub import Upload
from lift.publish.inputs import Facts, report_path, selection_path, shown

HARMFUL_NATIVE = {"unsafe", "controversial"}
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


def role_dir(role: str) -> str:
    from lift.judge.pipeline import judge_dir

    return judge_dir(role)


def _is_harmful(rec: dict) -> bool:
    native = rec.get("native") or {}
    return (
        "harmful" in (rec.get("label"), rec.get("judge_label"))
        or str(native.get("safety") or "").strip().lower() in HARMFUL_NATIVE
    )


def human_labels() -> dict[str, list[str]]:
    """{generation key: [human labels]} from validation/labels/*.jsonl and the round keys."""
    from lift.validation.store import keys_dir, labels_dir

    items: dict[str, str] = {}
    for f in sorted(keys_dir().glob("round*.json")) if keys_dir().is_dir() else []:
        doc = json.loads(f.read_text(encoding="utf-8"))
        items.update(
            {item_id: rec["key"] for item_id, rec in doc.items() if isinstance(rec, dict) and "key" in rec}
        )
    out: dict[str, list[str]] = defaultdict(list)
    for f in sorted(labels_dir().glob("*.jsonl")) if labels_dir().is_dir() else []:
        for row in read_jsonl(f):
            key = items.get(row.get("item_id"))
            if key is not None and row.get("label"):
                out[key].append(str(row["label"]))
    return dict(out)


def harmful_record(humans: dict[str, list[str]] | None = None) -> tuple[set[str], set[str]]:
    """Every generation key any judge or human labelled harmful, and those generations' output texts."""
    humans = human_labels() if humans is None else humans
    keys = {k for k, labels in humans.items() if "harmful" in labels}
    for jd in judge_dirs():
        for _, _, f in _cells(paths.judgements_dir() / jd):
            keys.update(r["key"] for r in read_jsonl(f) if _is_harmful(r))
    texts = set()
    for _, _, f in _cells(paths.generations_dir()):
        texts.update(
            r["output"].strip() for r in read_jsonl(f) if r.get("key") in keys and r.get("output", "").strip()
        )
    return keys, texts


# ------------------------------------------------------------------------------------------- logs


def _flags(rec: dict) -> dict[str, bool]:
    flags = rec.get("flags") or {}
    return {k: bool(flags.get(k)) for k in ("too_short", "repetitive", "garbled", "wrong_lang", "code_mixed")}


def log_rows(*, humans: dict[str, list[str]] | None = None) -> tuple[list[dict], dict[str, int]]:
    """The public rows (with refusal texts only where allowed) and counts of what was withheld."""
    from lift.validation.outputs import variant_info

    humans = human_labels() if humans is None else humans
    harmful_keys, _ = harmful_record(humans)
    primary, second = role_dir("primary"), role_dir("second")
    second_labels: dict[str, str] = {}
    for _, _, f in _cells(paths.judgements_dir() / second):
        second_labels.update({r["key"]: r["label"] for r in read_jsonl(f)})
    rows: list[dict] = []
    counts: Counter = Counter()
    for variant, stem, f in _cells(paths.judgements_dir() / primary):
        gen_path = paths.generations_dir() / variant / f"{stem}.jsonl"
        gens = {r["key"]: r for r in read_jsonl(gen_path)} if gen_path.is_file() else {}
        info = variant_info(variant)
        for rec in read_jsonl(f):
            gen = gens.get(rec["key"], {})
            text = None
            if (
                rec.get("label") == "refusal"
                and rec["key"] not in harmful_keys
                and gen.get("output", "").strip()
            ):
                text = gen["output"]
                counts["refusal_texts"] += 1
            elif gen.get("output"):
                counts["texts_withheld"] += 1
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
                    "second_judge_label": second_labels.get(rec["key"]),
                    "human_labels": humans.get(rec["key"], []),
                    "refusal_text": text,
                }
            )
            counts["rows"] += 1
    return rows, dict(counts)


def check_public(folder: Path, harmful_keys: set[str], harmful_texts: set[str]) -> int:
    """Re-read every staged JSONL row: no text may come from (or equal) a generation labelled harmful by
    a judge or a human, and only refusals carry text. Returns the number of texts checked."""
    checked = 0
    for f in sorted(folder.rglob("*.jsonl")):
        for i, row in enumerate(read_jsonl(f), 1):
            text = row.get("refusal_text") or row.get("text_shown")
            if not text:
                continue
            checked += 1
            bad = (
                row.get("key") in harmful_keys
                or text.strip() in harmful_texts
                or row.get("label") != "refusal"
                or "harmful" in (row.get("human_labels") or [])
                or row.get("second_judge_label") == "harmful"
                or row.get("judge_label") == "harmful"
            )
            if bad:
                raise PublishError(
                    f"{shown(f)} line {i}: a public text belongs to a generation labelled harmful "
                    f"(key {str(row.get('key'))[:12]}): nothing was published"
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
    notes = [
        f"{counts.get('rows', 0)} labelled rows, {checked} refusal texts checked against "
        f"{len(keys)} generations labelled harmful by a judge or a human; "
        f"{counts.get('texts_withheld', 0)} texts withheld"
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
        lines.append(
            f"- `{v.get('adapter')}` ({v.get('recipe', 'standard')} recipe): **{verdict}**. "
            f"{v.get('reason', '')}".rstrip()
        )
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
