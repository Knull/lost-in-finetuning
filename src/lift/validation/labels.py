"""Import a filled validation sheet as HumanLabel records. Implemented in P04.

``lift validate import --round N --lang L --file filled.csv`` reads the CSV a reader sends back
(downloaded from Google Sheets, or saved from Excel) and writes validation/labels/{sheet}.jsonl.

- Only the four classes count: harmful, refusal, benign, unclear (case and surrounding spaces are
  ignored).
- An empty label is left out of scoring and counted. With "skip" in the note it is a deliberate
  skip (validation/guidelines.md); without one it is unlabelled, probably an unfinished row. A row
  the reader deleted is missing. decide-judge refuses unlabelled and missing rows unless told
  otherwise.
- Bad labels, repeated ids, ids that are not on a round N sheet for language L, and ids from two
  different sheets are errors. They are all reported together, and nothing is written until the
  file is clean.
- The prompt and response cells are compared with what the sheet showed (ignoring whitespace).
  A few differences are warnings (an edited cell); many mean the file is another version of the
  sheet or its id column was sorted on its own, and the import stops. A cell whose script came
  back as "?" (Excel's plain CSV) is a warning of its own. A label column that comes back in
  alphabetical order, either way round, while the ids are still in the order the sheet was made
  was sorted on its own: its labels no longer sit next to their items, which the text check
  cannot see, so the import stops too. With the ids in item_id order instead (either way round:
  the sheet sorted by item_id, then the label column alone) it is a warning. A whole sheet sorted
  by label, rows kept together, leaves the ids in neither order and is fine.
- Which sheet a file belongs to (the main one or the Hindi overlap) follows from its ids.
- Re-importing a sheet replaces its labels. The import records which draw of the round it
  belongs to, so a later re-draw cannot be scored against old labels.

Messages carry row numbers, ids and counts, never sheet text.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from lift.io import sha256_file, sha256_text, write_jsonl
from lift.schemas import LABELS, HumanLabel
from lift.validation.outputs import read_records
from lift.validation.sample import load_round, sample_digest
from lift.validation.store import (
    ValidateError,
    check_writable,
    labels_dir,
    normalise,
    read_json,
    read_sheet,
    write_json,
)

MAX_LISTED = 40
MAX_CHANGED = (3, 0.10)        # more changed rows than max(3, 10%) stops an import
_TOKEN = re.compile(r"[A-Za-z0-9_.-]{1,24}")


def labels_path(sheet: str) -> Path:
    return labels_dir() / f"{sheet}.jsonl"


def labels_meta_path(sheet: str) -> Path:
    return labels_dir() / f"{sheet}.meta.json"


def read_labels(sheet: str) -> dict[str, HumanLabel] | None:
    """{item_id: HumanLabel} for an imported sheet, None if it has not been imported."""
    path = labels_path(sheet)
    if not path.is_file():
        return None
    return {rec.item_id: rec for rec in (HumanLabel.model_validate(r) for _, r in read_records(path))}


def read_labels_meta(sheet: str) -> dict | None:
    path = labels_meta_path(sheet)
    return read_json(path) if path.is_file() else None


def listing(problems: list[str]) -> str:
    shown = problems[:MAX_LISTED]
    more = len(problems) - len(shown)
    return "\n  ".join(shown) + (f"\n  ... and {more} more" if more > 0 else "")


def short(value: str) -> str:
    """A cell as it may appear in a message: quoted only if it looks like an id or a label, since a
    reader can paste a prompt or response into the wrong cell."""
    value = value.strip()
    if _TOKEN.fullmatch(value):
        return repr(value)
    return f"a {len(value)}-character value"


def _sorted_on_its_own(values: list[str]) -> bool:
    return len(values) >= 20 and len(set(values)) >= 3 and values in (sorted(values),
                                                                     sorted(values, reverse=True))


def _in_order_made(ids: list[str], items: dict[str, dict], sheet: str) -> bool:
    """Whether a file's ``ids`` (in row order) keep the order of ``sheet`` as made: ``items`` holds each
    sheet's ids in its row order, as its keys file does (lift.validation.sample.load_round)."""
    made = {i: n for n, i in enumerate(i for i, it in items.items() if it["sheet"] == sheet)}
    rows = [made[i] for i in ids]
    return rows == sorted(rows)


def _in_id_order(ids: list[str]) -> bool:
    """Whether a file's ``ids`` (in row order) are in item_id order, either way round: the sheet was sorted
    by its id column, rows kept together (a sheet is made in a seeded random order, never this one)."""
    return ids in (sorted(ids), sorted(ids, reverse=True))


def parse_filled(file: Path, items: dict[str, dict], *, value_column: str,
                 allowed: dict[str, str], what: str) -> dict:
    """Shared by the safety and utility imports. ``items`` maps the item ids this file may contain
    to their keys entries (with a "shown" map of column -> sha256 of the normalised text), each
    sheet's ids in the order its rows were made; ``allowed`` maps a normalised cell value to the
    stored value. A cell whose shown text had non-ASCII characters (the entry's "shown_non_ascii")
    and that comes back as ASCII with "?" in it lost its script to a code-page save (Excel's plain
    CSV writes Devanagari and most Han as "?", which is still valid UTF-8): a warning, not an
    edited cell.

    Returns {"sheet", "values": {item_id: (value, note)}, "skipped", "unlabelled", "missing",
    "changed", "warnings", "rows", "items_on_sheet"}; raises ValidateError listing every problem.
    """
    filled = read_sheet(file, required=("item_id", value_column))
    errors: list[str] = []
    lost: list[str] = []
    seen: dict[str, int] = {}
    values: dict[str, tuple[str, str | None]] = {}
    in_order: list[str] = []
    skipped: list[str] = []
    unlabelled: list[str] = []
    changed: list[str] = []
    sheets_in_file: set[str] = set()
    for rowno, row in filled.rows:
        item_id = row.get("item_id", "").strip()
        raw = row.get(value_column, "")
        note = row.get("note", "").strip() or None
        if not item_id:
            errors.append(f"row {rowno}: no item_id")
            continue
        if item_id in seen:
            errors.append(f"row {rowno}: item_id {short(item_id)} repeats row {seen[item_id]}")
            continue
        seen[item_id] = rowno
        item = items.get(item_id)
        if item is None:
            errors.append(f"row {rowno}: unknown item_id {short(item_id)} (not on {what})")
            continue
        sheets_in_file.add(item["sheet"])
        value = raw.strip().lower()
        if value and value not in allowed:
            errors.append(f"row {rowno}: {value_column} {short(raw)} is not one of "
                          f"{', '.join(dict.fromkeys(allowed.values()))}")
            continue
        if filled.utf8:
            non_ascii = item.get("shown_non_ascii", {})
            for column, digest in item.get("shown", {}).items():
                cell = row.get(column)
                if cell is None or sha256_text(normalise(cell)) == digest:
                    continue
                if non_ascii.get(column) and cell.isascii() and "?" in cell:
                    lost.append(item_id)
                    continue
                changed.append(item_id)
                break
        if value:
            values[item_id] = (allowed[value], note)
            in_order.append(allowed[value])
        elif note and "skip" in note.lower():
            skipped.append(item_id)
        else:
            unlabelled.append(item_id)
    unknown = sum("unknown item_id" in e or "no item_id" in e for e in errors)
    if filled.rows and unknown > len(filled.rows) / 2:
        raise ValidateError(f"{file}: {unknown} of {len(filled.rows)} rows have no item id of {what}, so "
                            "the file was not read as that sheet: was it opened by double-click in Excel "
                            "or saved with another delimiter? Download it from Google Sheets as CSV, or "
                            "save it from Excel as 'CSV UTF-8'")
    if len(sheets_in_file) > 1:
        errors.append(f"the file mixes ids from {len(sheets_in_file)} sheets "
                      f"({', '.join(sorted(sheets_in_file))}); import each sheet's file on its own")
    limit = max(MAX_CHANGED[0], int(MAX_CHANGED[1] * len(filled.rows)))
    if len(changed) > limit:
        errors.append(f"{len(changed)} rows show a prompt or response that differs from the sheet as "
                      "made: this looks like another version of the sheet, or its id column was "
                      "sorted on its own")
    if errors:
        raise ValidateError(f"{file}: {len(errors)} problem(s), nothing imported:\n  {listing(errors)}")
    if not sheets_in_file:
        raise ValidateError(f"{file}: no item id of {what} in the file; is it the right sheet?")
    sheet = next(iter(sheets_in_file))
    values_sorted = _sorted_on_its_own(in_order)
    if values_sorted and _in_order_made(list(seen), items, sheet):
        raise ValidateError(f"{file}: the {value_column} column is in alphabetical order while the item ids "
                            "are still in the order the sheet was made, so it was sorted on its own and its "
                            "values no longer sit next to their items. Nothing imported: undo the sort (or "
                            "fill in a fresh copy of the sheet) and import the file again")
    on_sheet = sorted(i for i, it in items.items() if it["sheet"] == sheet)
    missing = [i for i in on_sheet if i not in seen]
    warnings = []
    if not filled.utf8 or lost:
        how = "the file is not UTF-8" if not filled.utf8 else (
            f"{len(set(lost))} row(s) came back with their script turned into '?'")
        warnings.append(f"{how} (saved by Excel as plain CSV?): ids and values were read, but those "
                        "prompt and response cells could not be checked. Save as 'CSV UTF-8' or "
                        "download from Google Sheets next time")
    if filled.delimiter != ",":
        warnings.append(f"the file separates cells with {filled.delimiter!r}; read it that way")
    if changed:
        warnings.append(f"{len(changed)} row(s) show a prompt or response that differs from the sheet "
                        f"as made (an edited cell?): {', '.join(sorted(changed)[:MAX_LISTED])}")
    if values_sorted and _in_id_order(list(seen)):
        warnings.append(f"the {value_column} column is in alphabetical order and the item ids in item_id "
                        f"order: if the sheet was sorted by item_id and then the {value_column} column on "
                        "its own, the values no longer sit next to their items. Check before scoring")
    if unlabelled:
        warnings.append(f"{len(unlabelled)} row(s) have no {value_column} and no 'skipped' note "
                        f"(unfinished?): {', '.join(sorted(unlabelled)[:MAX_LISTED])}")
    if missing:
        warnings.append(f"{len(missing)} item(s) of the sheet are not in the file: "
                        f"{', '.join(missing[:MAX_LISTED])}")
    if not values:
        warnings.append("the file has no filled-in values")
    return {"sheet": sheet, "values": values, "skipped": sorted(skipped), "unlabelled": sorted(unlabelled),
            "missing": missing, "changed": sorted(changed), "warnings": warnings,
            "rows": len(filled.rows), "items_on_sheet": len(on_sheet)}


def import_labels(round_no: int, lang: str, file: Path, *, annotator: str | None = None) -> dict:
    """Validate a filled round ``round_no`` sheet for ``lang`` and write its HumanLabel records."""
    doc = load_round(round_no)
    items = {}
    for it in doc["items"]:
        if it["lang"] == lang:
            items[it["item_id"]] = {**it, "shown": {"prompt": it["shown_prompt_sha"],
                                                    "response": it["shown_response_sha"]}}
    if not items:
        raise ValidateError(f"round {round_no} has no {lang} sheet")
    parsed = parse_filled(Path(file), items, value_column="label",
                          allowed={c: c for c in LABELS}, what=f"a round {round_no} {lang} sheet")
    sheet = parsed["sheet"]
    slot = doc["sheets"][sheet]["slot"]
    annotator = annotator or f"{lang}-{slot}"
    records = [HumanLabel(item_id=i, annotator=annotator, label=v, note=note)
               for i, (v, note) in sorted(parsed["values"].items())]
    previous = read_labels(sheet)
    if previous and not records:
        raise ValidateError(f"{file} has no labels, but {labels_path(sheet).name} holds {len(previous)}: "
                            "importing it would wipe them (the blank sheet as sent out?)")
    check_writable([labels_path(sheet), labels_meta_path(sheet)])
    write_jsonl(labels_path(sheet), [r.model_dump() for r in records])
    counts = Counter(r.label for r in records)
    summary = {
        "sheet": sheet, "round": round_no, "lang": lang, "slot": slot, "annotator": annotator,
        "keys_digest": sample_digest(doc),
        "file": str(file), "file_sha256": sha256_file(file),
        "imported": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": parsed["rows"], "items_on_sheet": parsed["items_on_sheet"],
        "labelled": len(records), "skipped": parsed["skipped"], "unlabelled": parsed["unlabelled"],
        "missing": parsed["missing"], "changed_text": parsed["changed"],
        "label_counts": {c: counts.get(c, 0) for c in LABELS},
        "replaced": None if previous is None else {
            "labels": len(previous),
            "changed": sum(1 for r in records if r.item_id in previous
                           and previous[r.item_id].label != r.label),
            "dropped": sum(1 for i in previous if i not in {r.item_id for r in records}),
        },
    }
    write_json(labels_meta_path(sheet), summary)
    return {**summary, "labels_path": str(labels_path(sheet)), "warnings": parsed["warnings"]}
