"""``lift publish datasets``: the public logs dataset, the gated raw completions and the translated
prompt sets, staged under ``outputs/publish/datasets/``.

Logs (public; decisions items 14 and 62). No output text and no judge text. One row per primary-judge
judgement (``LOG_FIELDS``): variant, method, seed, pid, language, set, split, the final label, the
judge's own label, the unclear flags, the cap hit, the second judge's label and any human labels where
they exist, and the sha256 of the output judged. A failed judgement keeps only the kind of its failure
(``lift.validation.score.failure_kind``: the text before its first ":" or quote), never the failure's
text, which can quote a provider's refusal or the start of a reply. ``JudgementRecord.raw`` and
``native.reason`` (the judges' replies and reasons) are never read into a row. Plus the aggregate tables
(final_report.json as JSON and one CSV per table), selection.json and the secondary selection, the gate
and validation reports, and the redacted AuditKIT records (never ``*.full.json``, each checked to be
redacted, and each only while it is still the record of its cell: ``audit_record_problem``).

No output is public because of its label: for about 90% of outputs the label is Qwen3Guard's alone, and
the judge bar tolerates 15% disagreement with readers. So the guard reads content. ``check_public``
re-reads every staged row (only ``LOG_FIELDS``, only values as short as an id or a label), and the whole
staged folder is scanned like a public upload (``preflight.scan_folder``) against the windows of every
model output on this machine, whatever its label (``output_record``: outputs/generations, what
``lift restart`` set aside, Gate 0's runs), after every absolute local path in it is made repo-relative
or the build refused (``preflight.relativise_folder``, XB2-04).

Nothing is staged unless the inputs are complete (``PublishError`` names the remedy): the human labels
are readable (every round that was drawn or scored has its keys and every sheet imported, every label
maps to a key), every generation of a safety cell has a primary judgement and every judgement its
generation, the second judge covers its key list (the final-split keys of
validation/keys/second_judge/all.txt plus every validated key) on the current outputs, and no Message
Batch is still open.

Labels are shown in one vocabulary (``schemas.Label``: harmful, refusal, benign, unclear), case and
surrounding space ignored (P08-R2-05); any other label is shown as it was written. A cell that holds one
key twice (generations, primary or second judgements) is refused with its ``lift restart`` line: which
row counts would be a guess (P08-R2-11).

The logs also carry P07's figures (``figures/``, review XB-08), which the article and the logs card
show. They also hold the records behind the other numbers (P08-R2-14), none with a text: the per-item
Belebele and held-out-loss scores (``scores/``), P07's back-translation verdicts (``backtranslation/``,
known fields only, a failure as its kind) and the run manifests behind the cost numbers
(``manifests/``). The utility verdicts quote both answers, so they are in the gated raw completions.

Raw completions (gated, manual approval). Every generations, judgements and utility-verdict file, and
the full AuditKIT records, as they are in ``outputs/``: every judged output, refusals and benign answers
included, and the judges' raw replies, reasons and failure messages. Human labels (with annotator ids)
stay private.

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
from typing import NamedTuple

from lift import config, paths
from lift.io import read_jsonl, sha256_file, sha256_text, write_jsonl
from lift.publish import PublishError, preflight, render, settings
from lift.publish.hub import Upload
from lift.publish.inputs import FIGURES, Facts, recipe_block, report_path, selection_path, shown

SAFETY_SETS = ("harmful", "easy_benign", "hard_benign", "redteam")  # judged cells (validation.second_judge)
# What a public row may hold (check_public refuses any other field): ids, labels, flags and hashes.
LOG_FIELDS = (
    "key",
    "variant",
    "method",
    "seed",
    "pid",
    "lang",
    "set",
    "split",
    "judge",
    "label",
    "judge_label",
    "judge_failure",
    "too_short",
    "repetitive",
    "garbled",
    "wrong_lang",
    "code_mixed",
    "hit_cap",
    "n_new_tokens",
    "output_sha",
    "judged_current_output",
    "second_judge_label",
    "second_judge_failure",
    "human_labels",
)
MAX_VALUE_CHARS = 200  # a key is 64 characters and a judge id about 80: no public value is a sentence
PULL = "run `lift sync pull` first (outputs/ and validation/ come only from the private runs repo)"
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


LABELS = ("harmful", "refusal", "benign", "unclear")  # lift.schemas.Label


def shown_label(value: object) -> object:
    """A label as the public rows show it: one of LABELS in lower case, anything else as it was."""
    if isinstance(value, str) and value.strip().casefold() in LABELS:
        return value.strip().casefold()
    return value


def failure_of(text: object) -> str | None:
    """The kind of a recorded judge failure, None when there is none: ``failure_kind`` of its text (what
    comes before its first ":" or quote), never the text itself, which can quote a provider's refusal or
    the start of a reply the judge could not parse (decisions item 62)."""
    from lift.validation.score import failure_kind

    if not text:
        return None
    return failure_kind(text) if isinstance(text, str) else type(text).__name__


def _restart_line(path: Path, role: str | None) -> str:
    """The ``lift restart`` line that starts the cell of a generations or judgements file over."""
    from lift.models.generate import restart_command
    from lift.validation.second_judge import parse_generations_name

    parsed = parse_generations_name(path.stem)
    if parsed is None:
        return f"set {shown(path)} aside with its lift restart line"
    dtype = None
    gen = paths.generations_dir() / path.parent.name / path.name
    for r in read_jsonl(gen) if gen.is_file() else []:
        dtype = (r.get("meta") or {}).get("dtype")
        break
    return restart_command(path.parent.name, *parsed, judge=role, dtype=dtype)


def keyed(path: Path, *, role: str | None = None) -> dict[str, dict]:
    """{key: row} of one generations (``role`` None) or judgements file. Refuses (PublishError) a file
    that holds a key twice, naming its lift restart line: which row counts would be a guess."""
    out: dict[str, dict] = {}
    dup: list[str] = []
    for r in read_jsonl(path):
        key = r.get("key")
        if key in out:
            dup.append(str(key))
        out[key] = r
    if dup:
        raise PublishError(
            f"{shown(path)}: {len(dup)} keys are duplicated (e.g. {dup[0][:12]}), so which row counts is not "
            f"known and nothing was staged: start the cell over: {_restart_line(path, role)}"
        )
    return out


def primary_judgements() -> dict[str, dict]:
    """{key: the primary judgement} over every primary judgements file: the one map both ``log_rows`` and
    ``check_public`` read. Refuses a key judged twice, in one file or in two."""
    out: dict[str, dict] = {}
    for _, _, f in _cells(paths.judgements_dir() / role_dir("primary")):
        for key, r in keyed(f, role="primary").items():
            if key in out:
                raise PublishError(
                    f"{shown(f)}: key {str(key)[:12]} has a primary judgement in another file too: start the "
                    f"cell over: {_restart_line(f, 'primary')}"
                )
            out[key] = r
    return out


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
    (PublishError) when they cannot all be read: the public record would silently lack them."""
    from lift.validation.store import labels_dir

    items, problems = round_items()
    out: dict[str, list[str]] = defaultdict(list)
    unmapped = 0
    for f in sorted(labels_dir().glob("round*.jsonl")) if labels_dir().is_dir() else []:
        for row in read_jsonl(f):
            key = items.get(str(row.get("item_id")))
            if key is None:
                unmapped += 1
            elif row.get("label") is not None and str(row["label"]).strip():  # blank: not labelled yet
                out[key].append(shown_label(row["label"]))
    if unmapped:
        problems.append(f"{unmapped} human labels name an item_id that no validation/keys/round*.json maps")
    if problems:
        raise PublishError(
            "the human labels cannot all be read, so the logs would leave some out: "
            + "; ".join(problems)
            + f". {PULL}"
        )
    return dict(out)


def backtranslation_dir() -> Path:
    """P07's back-translation verdicts (``lift.final.bt_path``)."""
    return paths.output_dir() / "sensitivity" / "backtranslation"


class Outputs(NamedTuple):
    texts: set[str]  # every model output text on this machine, whatever its label
    missing: set[str]  # judged keys with no generation here at all
    missing_files: list[str]  # the judgement files that name them
    gone: int  # judged outputs (by output_sha) whose text is no longer on this machine


def _generation_files() -> Iterator[Path]:
    """Every generations file on this machine: each JSONL under a ``generations`` folder of outputs/
    (outputs/generations, what lift restart set aside in outputs/.aside, Gate 0's runs in outputs/gate0),
    outside the publish staging."""
    root = paths.output_dir()
    if not root.is_dir():
        return
    staging = root / "publish"
    for folder in sorted(p for p in root.rglob("generations") if p.is_dir() and staging not in p.parents):
        yield from sorted(folder.rglob("*.jsonl"))


def output_record() -> Outputs:
    """Every model output text on this machine, whatever its label: the guard of the public files reads
    content, never labels (decisions item 62). Also the judged keys with no generation here (their text
    cannot be checked against: ``lift sync pull``) and how many judged outputs, named by the
    ``output_sha`` of a judgement or a back-translation verdict, are no longer here (regenerated and not
    set aside)."""
    texts: set[str] = set()
    keys: set[object] = set()
    shas: set[str] = set()
    for f in _generation_files():
        for r in read_jsonl(f):
            out = r.get("output")
            if not isinstance(out, str):
                continue
            keys.add(r.get("key"))
            shas.add(sha256_text(out))
            if out.strip():
                texts.add(out.strip())
    judged: set[str] = set()
    missing: set[str] = set()
    files: list[str] = []
    for f in [*judgement_files(), *(f for _, _, f in _cells(backtranslation_dir()))]:
        lacking = False
        for r in read_jsonl(f):
            if r.get("key") not in keys:
                missing.add(str(r.get("key")))
                lacking = True
            if r.get("output_sha"):
                judged.add(r["output_sha"])
        if lacking:
            files.append(shown(f))
    return Outputs(texts, missing, files, len(judged - shas))


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


def generation_splits() -> dict[str, str]:
    """{generation key: split} from the generations cells' names ({set}_{lang}_{split}.jsonl)."""
    from lift.validation.second_judge import parse_generations_name

    out: dict[str, str] = {}
    for _, stem, f in _cells(paths.generations_dir()):
        parsed = parse_generations_name(stem)
        if parsed is not None:
            out.update(dict.fromkeys((k for k in keyed(f) if k is not None), parsed[2]))
    return out


def second_judge_keys(
    round_keys: set[str], splits: dict[str, str] | None = None
) -> tuple[set[str] | None, Path]:
    """The keys the second judge must cover: the final-split keys of validation/keys/second_judge/all.txt
    (P04) plus every validated key; None when the list is not here. all.txt samples every complete cell,
    the dev sweep's too, but the second judge is reported on the final split only and no step judges the
    dev keys (review XB-06). A listed key whose split is unknown (no generation here) stays required."""
    from lift.judge.pipeline import read_keys_file
    from lift.validation.second_judge import second_judge_dir

    path = second_judge_dir() / "all.txt"
    if not path.is_file():
        return None, path
    try:
        listed = set(read_keys_file(path))
    except ValueError as exc:
        raise PublishError(f"{shown(path)}: {exc}") from None
    splits = generation_splits() if splits is None else splits
    return {k for k in listed if splits.get(k, "final") == "final"} | round_keys, path


# ------------------------------------------------------------------------------------------- logs


def _flags(rec: dict) -> dict[str, bool]:
    flags = rec.get("flags") or {}
    return {k: bool(flags.get(k)) for k in ("too_short", "repetitive", "garbled", "wrong_lang", "code_mixed")}


def _current_shas() -> dict[str, str]:
    """{generation key: sha256 of the output now under it} over every generations file here (a file that
    holds a key twice is refused: ``keyed``)."""
    return {
        key: sha256_text(r.get("output") or "")
        for _, _, f in _cells(paths.generations_dir())
        for key, r in keyed(f).items()
        if key is not None
    }


def log_rows(*, humans: dict[str, list[str]] | None = None) -> tuple[list[dict], dict[str, int]]:
    """The public rows (``LOG_FIELDS``: no output text, no judge text) and their count. Refuses
    (PublishError) unless the record is complete: the module docstring lists the checks."""
    from lift.validation.outputs import variant_info
    from lift.validation.second_judge import parse_generations_name

    humans = human_labels() if humans is None else humans
    primary, second = role_dir("primary"), role_dir("second")
    problems: list[str] = []
    current = _current_shas()
    second_current: dict[str, dict] = {}  # second judgements of the output now under the key
    for _, _, f in _cells(paths.judgements_dir() / second):
        for key, r in keyed(f, role="second").items():
            if r.get("output_sha") and r["output_sha"] == current.get(key):
                second_current[key] = r
    primary_all = primary_judgements()  # refuses a key judged twice
    primary_cells = {(v, stem): f for v, stem, f in _cells(paths.judgements_dir() / primary)}
    for variant, stem, gen_path in _cells(paths.generations_dir()):
        parsed = parse_generations_name(stem)
        if parsed is None or parsed[0] not in SAFETY_SETS:
            continue
        if (variant, stem) not in primary_cells and variant_info(variant) is None:
            continue  # not a study variant, and never judged: nothing of it is published
        judged = set(keyed(primary_cells[(variant, stem)])) if (variant, stem) in primary_cells else set()
        unjudged = sum(1 for key in keyed(gen_path) if key not in judged)
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
        gens = keyed(gen_path) if gen_path.is_file() else {}
        info = variant_info(variant)
        for key in keyed(f, role="primary"):
            rec = primary_all[key]
            gen = gens.get(rec["key"])
            if gen is None:
                no_generation += 1
                gen = {}
            other = second_current.get(rec["key"])
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
                    "label": shown_label(rec.get("label")),
                    "judge_label": shown_label(rec.get("judge_label")),
                    "judge_failure": failure_of((rec.get("native") or {}).get("judge_failure")),
                    **_flags(rec),
                    "hit_cap": gen.get("hit_cap"),
                    "n_new_tokens": gen.get("n_new_tokens"),
                    "output_sha": rec.get("output_sha"),
                    "judged_current_output": bool(gen) and rec.get("output_sha") == current.get(rec["key"]),
                    "second_judge_label": shown_label(other.get("label")) if other else None,
                    "second_judge_failure": (
                        failure_of((other.get("native") or {}).get("judge_failure")) if other else None
                    ),
                    "human_labels": humans.get(rec["key"], []),
                }
            )
            counts["rows"] += 1
            counts["judge_failures"] += bool(rows[-1]["judge_failure"])
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


def _too_long(value: object) -> bool:
    if isinstance(value, str):
        return len(value) > MAX_VALUE_CHARS
    if isinstance(value, (list, tuple)):
        return any(_too_long(v) for v in value)
    return isinstance(value, dict)  # no public field is a mapping (a judge's native block never goes in)


def check_public(folder: Path, windows: set[str]) -> int:
    """Re-read every staged row of the labels tables (``labels/*.jsonl``): it may hold only the fields of
    ``LOG_FIELDS``, no value longer than a key, an id or a label, and no window of a model output
    (``windows``: ``preflight.windows_of`` every output here, whatever its label). Any other staged JSONL
    row is checked for output windows only (``preflight.scan_folder`` then scans every file again).
    Refuses (PublishError) on the first row that fails; returns the number of label rows checked."""
    checked = 0
    for f in sorted(folder.rglob("*.jsonl")):
        labels = "labels" in f.relative_to(folder).parts[:-1]
        for i, row in enumerate(read_jsonl(f), 1):
            if labels:
                checked += 1
                extra = sorted(set(row) - set(LOG_FIELDS))
                if extra:
                    raise PublishError(
                        f"{shown(f)} line {i}: a public row holds the fields {extra}, which the logs do not "
                        "publish (no output text, no judge text): nothing was published"
                    )
                long = sorted(k for k, v in row.items() if _too_long(v))
                if long:
                    raise PublishError(
                        f"{shown(f)} line {i}: the public fields {long} hold more than an id or a label "
                        f"(over {MAX_VALUE_CHARS} characters, or a mapping): nothing was published"
                    )
            if any(preflight.has_window(v, windows) for v in preflight._strings(row)):
                raise PublishError(
                    f"{shown(f)} line {i}: a public row holds a run of a model output "
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


AUDIT_NEEDS = ("variant", "set", "lang", "split", "judge", "judge_id", "labels_sha256", "keys_sha256")
# Decision and validation files that go in as they are: none holds an output or a judge's words (the
# validation reports and the judge decision count failures by kind: lift.validation.score).
DECISION_FILES = ("selection_secondary.json", "gate1.json", "gate2.json", "judge_decision.json")


def audit_record_problem(doc: object) -> str | None:
    """Why a redacted AuditKIT record of a safety cell is left out of the logs (None when it goes in).

    A record outlives a change that makes its cell unreportable (a fixed prompt text, other labelling
    rules, a re-registered variant, a re-pinned Global, a harness bump) until that cell's ``lift restart``
    retires it, so a record on disk proves nothing (R2-11). It goes in only while the reporting door
    accepts its cell as it stands (``lift.judge.pipeline.load_records``, ``complete=True``, with the judge
    role and the precision its ``metadata.lift`` names), and only if that read is the record's: its labels
    and keys hash to the record's ``labels_sha256`` and ``keys_sha256`` (as ``lift.audit.record_safety``
    computes them) and its judge id is the record's ``judge_id``."""
    from lift.io import sha256_obj
    from lift.judge import pipeline

    meta = doc.get("metadata") if isinstance(doc, dict) else None
    meta = meta.get("lift") if isinstance(meta, dict) else None
    if not isinstance(meta, dict):
        return "it has no metadata.lift"
    lacking = [k for k in AUDIT_NEEDS if not isinstance(meta.get(k), str) or not meta[k]]
    if lacking:
        return f"its metadata.lift names no {', '.join(lacking)}"
    dtypes = (meta.get("provenance") or {}).get("dtype") or []
    dtypes = [dtypes] if isinstance(dtypes, str) else list(dtypes)
    if len(dtypes) > 1:
        return f"its cell was generated in more than one precision ({', '.join(map(str, dtypes))})"
    try:
        records = pipeline.load_records(
            meta["variant"],
            meta["set"],
            meta["lang"],
            meta["split"],
            judge=meta["judge"],
            complete=True,
            dtype=str(dtypes[0]) if dtypes else None,
        )
    except Exception as exc:  # any refusal of the reporting door: the record is not this cell's any more
        why = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
        return f"its cell does not read as it stands ({type(exc).__name__}: {why})"
    if not records:
        return "its cell holds no judgements"
    if sha256_obj({r.pid: r.label for r in records}) != meta["labels_sha256"]:
        return "the cell's labels are not the ones it recorded"
    if sha256_obj({r.pid: r.key for r in records}) != meta["keys_sha256"]:
        return "the cell's generation keys are not the ones it recorded"
    if records[0].judge != meta["judge_id"]:
        return f"the cell's judge is {records[0].judge}, not the {meta['judge_id']} it recorded"
    return None


def _aggregates(folder: Path, facts: Facts) -> tuple[list[str], list[str]]:
    """Copy the aggregate tables, the decision files and the redacted AuditKIT records into the staged
    logs. Returns (what was copied, the AuditKIT records left out, each with its reason)."""
    from lift.validation.store import metrics_dir as validation_metrics

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
    for name in DECISION_FILES:
        src = paths.metrics_dir() / name
        if src.is_file() and src.stat().st_size:
            (folder / "aggregates").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, folder / "aggregates" / name)
            copied.append(name)
    root = validation_metrics()
    for f in sorted(root.glob("*.json")) if root.is_dir() else []:
        if f.stat().st_size:
            dest = folder / "aggregates" / "validation" / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
            copied.append(f"validation/{f.name}")
    dropped: list[str] = []
    audit_root = paths.metrics_dir() / "auditkit"
    for f in sorted(audit_root.rglob("*.json")) if audit_root.is_dir() else []:
        if f.name.endswith(".full.json") or f.stat().st_size == 0:
            continue
        rel = f.relative_to(audit_root).as_posix()
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            problem = preflight.auditkit_problem(doc)
        except json.JSONDecodeError:
            doc, problem = None, "it is not valid JSON"
        if problem:
            raise PublishError(
                f"{shown(f)} would go public but is not a redacted AuditKIT record ({problem}): write it "
                "again with lift.audit's default text='redacted' (a full record is *.full.json, gated)"
            )
        stale = audit_record_problem(doc)
        if stale:
            dropped.append(f"auditkit/{rel}: {stale}")
            continue
        dest = folder / "auditkit" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        copied.append(f"auditkit/{rel}")
    return copied, dropped


BT_FIELDS = (  # P07's lift.final._bt_row, without any text (judge_failure goes in as its kind)
    "key",
    "variant",
    "pid",
    "lang",
    "set",
    "split",
    "output_sha",
    "bt_sha",
    "judge",
    "judge_label",
    "judge_failure",
    "skipped",
)


def _records(folder: Path) -> dict[str, int]:
    """Copy the text-free records behind the other numbers into the logs: per-item scores (scores/),
    back-translation verdicts (backtranslation/, BT_FIELDS only, a judge failure as its kind) and run
    manifests (manifests/). Returns the number of files of each."""
    counts = {"scores": 0, "backtranslation": 0, "manifests": 0}
    src = paths.scores_dir()
    for f in sorted(src.rglob("*.jsonl")) if src.is_dir() else []:
        dest = folder / "scores" / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        counts["scores"] += 1
    for variant, _, f in _cells(backtranslation_dir()):
        rows = [
            {**{k: r.get(k) for k in BT_FIELDS}, "judge_failure": failure_of(r.get("judge_failure"))}
            for r in read_jsonl(f)
        ]
        write_jsonl(folder / "backtranslation" / variant / f.name, rows)
        counts["backtranslation"] += 1
    src = paths.manifests_dir()
    for f in sorted(src.glob("*.json")) if src.is_dir() else []:
        dest = folder / "manifests" / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        counts["manifests"] += 1
    return counts


def _figures(folder: Path) -> list[str]:
    """Copy P07's figures (outputs/figures/, ``inputs.FIGURES``) into the logs' figures/: the article and
    the logs card link them there (review XB-08). Returns the names copied."""
    copied = []
    for name in FIGURES:
        src = paths.figures_dir() / f"{name}.png"
        if src.is_file() and src.stat().st_size:
            dest = folder / "figures" / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            copied.append(name)
    return copied


def build_logs(facts: Facts | None = None) -> Upload:
    facts = facts or Facts()
    # selection.json and the report name the repaired model: blocked until the recipe is the tag's (XB2-02)
    block = recipe_block(facts) if facts.selection() is not None or facts.report() is not None else None
    repo = settings.repo("logs")
    folder = _fresh(stage_root() / "logs")
    humans = human_labels()
    found = output_record()
    rows, counts = log_rows(humans=humans)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)
    for variant, vrows in by_variant.items():
        write_jsonl(folder / "labels" / f"{variant}.jsonl", vrows)
    copied, dropped = _aggregates(folder, facts)
    records = _records(folder)
    _figures(folder)
    card = logs_card(
        facts,
        counts=counts,
        variants=sorted(by_variant),
        copied=copied,
        raw=settings.repo("raw"),
        records=records,
    )
    render.write_text(folder / "README.md", card)
    facts.write_sources(stage_root() / "logs.sources.json")
    relativised = preflight.relativise_folder(folder)  # no absolute local path goes public (XB2-04)
    # The content guard (decisions item 62): every output here, whatever its label. A run an output
    # copied from a prompt is public anyway (the redacted AuditKIT records keep each prompt): P08-R2-03.
    windows = preflight.windows_of(found.texts)
    windows -= preflight.windows_in_texts(preflight.prompt_texts(), windows)
    checked = check_public(folder, windows)
    scan = preflight.scan_folder(folder, windows)
    if scan:
        raise PublishError(
            "the staged logs dataset failed the public-text scan (matched text is never shown), nothing was "
            "published:\n  " + "\n  ".join(str(x) for x in scan[:50])
        )
    notes = [
        f"{counts.get('rows', 0)} labelled rows ({checked} checked: ids, labels, flags and hashes only), "
        f"{sum(len(v) for v in humans.values())} human labels read, {counts.get('judge_failures', 0)} judge "
        f"failures kept as their kind; no output text and no judge text; the staged files were scanned "
        f"against {len(found.texts)} model outputs on this machine ({len(windows)} windows), whatever their "
        "label"
    ]
    if dropped:
        notes.append(
            f"{len(dropped)} AuditKIT records left out (not the record of their cell as it stands): "
            + "; ".join(dropped[:5])
            + (f"; and {len(dropped) - 5} more" if len(dropped) > 5 else "")
        )
    if found.gone:
        notes.append(
            f"{found.gone} judged outputs are no longer on this machine (regenerated): their text cannot be "
            "checked against"
        )
    if not rows:
        notes.append("no judgements in outputs/judgements yet: only the card and aggregates are staged")
    notes += preflight.relativised_note(relativised)
    blocked = [block] if block else []
    if rows and found.missing:
        blocked.append(
            f"{len(found.missing)} judged outputs have no generation on this machine (e.g. in "
            f"{', '.join(found.missing_files[:3])}), so their text cannot be checked against: {PULL}"
        )
    return Upload(repo=repo, folder=folder, title="logs dataset", notes=notes, blocked=blocked)


RECORD_LINES = {
    "scores": "`scores/`: the per-item Belebele and held-out-loss scores ({n} files)",
    "backtranslation": "`backtranslation/`: the verdicts on the back-translated ar and th outputs "
    "({n} files; the output's sha256 and the label, no text)",
    "manifests": "`manifests/`: one manifest per run (code version, GPU, wall time) behind the cost "
    "numbers ({n} files)",
}


def logs_card(
    facts: Facts,
    *,
    counts: dict[str, int],
    variants: list[str],
    copied: list[str],
    raw: settings.RepoSpec,
    records: dict[str, int] | None = None,
) -> str:
    meta = {
        "license": "cc-by-nc-4.0",
        "language": list(config.get("languages.all")),
        "pretty_name": "Lost in Fine-Tuning: evaluation logs",
        "tags": ["safety", "multilingual", "evaluation", "tiny-aya", "lexsi-hackathon"],
        "configs": [{"config_name": "labels", "data_files": "labels/*.jsonl", "default": True}],
    }
    body = render.fill(
        render.template("logs_card.md"),
        {
            "raw_repo": raw.url,
            "n_rows": f"{counts.get('rows', 0):,}",
            "variants": ", ".join(f"`{v}`" for v in variants) or "none yet",
            "aggregates": "\n".join(f"- `{c}`" for c in copied) or f"- {facts_pending(facts)}",
            "verdicts": verdict_lines(facts),
            "records": "\n".join(
                f"- {line.format(n=(records or {}).get(name, 0))}" for name, line in RECORD_LINES.items()
            ),
            # relative links: build_logs copies the PNGs into figures/ (a missing one is pending)
            "figures": "\n\n".join(facts.figure(name) for name in FIGURES),
        },
    )
    return render.frontmatter(meta) + "\n" + body


def facts_pending(facts: Facts) -> str:
    from lift.publish.inputs import P07, pending

    return pending("the aggregate tables", shown(report_path()), P07)


# What a verdict row is, by P07's ``kind`` (lift.report.analysis: decisions items 80-81).
VERDICT_ROLES = {
    "adapter": "carries the repair comparison",
    "standard": "Gate 1's pick",
    "candidate": "the candidate Gate 1 did not pick, an exploratory arm",
}
NO_VERDICT = "no verdict"  # a candidate that drifts: the selection rule was not run on it


def verdict_lines(facts: Facts) -> str:
    """One line per adapter verdict from final_report.json, or a pending line: the README's word, what the
    row is (``VERDICT_ROLES``), Gate 1's usefulness verdict when the report gives one (decisions item 82)
    and P07's reason. A candidate without a word says so (``NO_VERDICT``): it is not an arm "not run"."""
    from lift.publish.inputs import P07

    verdicts = facts.raw(report_path(), "/verdicts")
    if verdicts is None:
        return facts.report_value("/verdicts", "text", what="the verdict per adapter")
    if not verdicts:
        return "_No verdict recorded._"
    lines = []
    for i, v in enumerate(verdicts):

        def said(key: str, what: str, at: int = i) -> str:
            return facts.value(report_path(), f"/verdicts/{at}/{key}", "text", what=what, stage=P07)

        verdict = NO_VERDICT if v.get("verdict") is None and v.get("kind") == "candidate" else None
        verdict = verdict or said("verdict", "verdict")
        adapter = said("adapter", "adapter")
        # Free text from P07 goes public too, so it is recorded in the sources like every number.
        reason = said("reason", "verdict reason") if v.get("reason") else ""
        recipe = said("recipe", "recipe") if v.get("recipe") else "standard"
        role = VERDICT_ROLES.get(v.get("kind"))
        useful = (
            f"; Gate 1: **{said('usefulness_verdict', 'Gate 1 verdict')}**"
            if v.get("usefulness_verdict")
            else ""
        )
        who = f"{recipe} recipe" + (f", {role}" if role else "")
        lines.append(f"- `{adapter}` ({who}): **{verdict}**{useful}. {reason}".rstrip())
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
    sources = Facts()  # the card's counts, for the claims audit (P08-R2-09)
    src = paths.prompts_dir()
    for f in sorted(src.glob("*.jsonl")) if src.is_dir() else []:
        rows = [r for r in read_jsonl(f) if (r.get("translation") or {}).get("engine") in ours]
        if not rows:
            continue
        set_name, lang = rows[0]["set"], rows[0]["lang"]
        keep = ("pid", "lang", "set", "split", "text", "text_en", "source", "category", "translation")
        write_jsonl(folder / set_name / f"{lang}.jsonl", [{k: r.get(k) for k in keep} for r in rows])
        tr = [r["translation"] for r in rows]
        for what, value in (
            ("prompts", len(rows)),
            ("native-checked", sum(bool(t.get("checked")) for t in tr)),
            ("edited", sum(bool(t.get("edited")) for t in tr)),
        ):
            sources.sources.append(
                {
                    "shown": f"{value:,}",
                    "file": shown(f),
                    "what": what,
                    "rows": f"translation.engine one of {sorted(ours)}",
                    "value": value,
                    "file_sha256": sha256_file(f),
                }
            )
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
    sources.write_sources(stage_root() / "prompts.sources.json")
    notes = [f"{len(table)} files"] if table else ["no translated prompt files yet (lift data build)"]
    notes += preflight.relativised_note(preflight.relativise_folder(folder))
    return Upload(repo=repo, folder=folder, title="translated prompt sets", notes=notes)


BUILDERS = {"logs": build_logs, "raw": build_raw, "prompts": build_prompts}


def build(only: list[str] | None = None) -> list[Upload]:
    facts = Facts()
    out = []
    for name in only or list(BUILDERS):
        if name not in BUILDERS:
            raise PublishError(f"unknown dataset {name!r} (one of {', '.join(BUILDERS)})")
        out.append(BUILDERS[name](facts))
    return out
