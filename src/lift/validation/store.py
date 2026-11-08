"""Where the validation files live and how they are read and written. Implemented in P04.

Everything under ``validation/`` except guidelines.md is private: gitignored, and carried between
machines only by ``lift sync`` (the private runs repo). Sheets hold full prompts and responses,
because annotators need them, so nothing in lift.validation prints their text; messages carry ids,
row numbers and counts.

    validation/sheets/validation_round{N}_{lang}.csv    the blind sheet one reader fills in
    validation/sheets/validation_round{N}_hi_b.csv      the optional Hindi overlap, second reader
    validation/keys/round{N}.json                       item_id -> generation key, variant, set, ...
    validation/keys/draw{N}.json                        the draw: seed, plan, pools, sheets
    validation/labels/round{N}_{lang}.jsonl             HumanLabel records from the filled sheet
    validation/labels/round{N}_{lang}.meta.json         what the import saw (skipped, missing, ...)
    validation/sheets/validation_utility_hi.csv, keys/utility.json, labels/utility_hi.jsonl
    validation/keys/second_judge/                       keys the second judge must cover
    validation/keys/archive/, sheets/archive/           what --force replaced

``LIFT_VALIDATION_DIR`` moves the whole tree (``lift.paths.validation_dir``): the tests use it, and so
do remote runtimes (the Kaggle runner puts it under /kaggle/temp), where ``lift sync`` carries
sheets/, keys/ and labels/ from wherever it points. P01's spot-check sheets live in the same sheets/.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lift import paths

GUIDELINES = "validation/guidelines.md"

# A cell starting with one of these is read as a formula by Google Sheets or Excel on import.
_FORMULA_START = ("=", "+", "-", "@", "\t")
_BOM = chr(0xFEFF)


class ValidateError(RuntimeError):
    """A problem the person running the command has to fix. The CLI prints it without a traceback."""


def validation_dir() -> Path:
    return paths.validation_dir()


def sheets_dir() -> Path:
    return validation_dir() / "sheets"


def keys_dir() -> Path:
    return validation_dir() / "keys"


def labels_dir() -> Path:
    return validation_dir() / "labels"


def metrics_dir() -> Path:
    return paths.metrics_dir() / "validation"


def round_keys_path(round_no: int) -> Path:
    """The keys of a round as the spec has them: {item_id: {key, variant, set, ...}}. P02's
    ``validation_keys()`` reads every validation/keys/round*.json this way."""
    return keys_dir() / f"round{round_no}.json"


def draw_path(round_no: int) -> Path:
    """Everything else about a round's draw (plan, pools, sheets, seed). Not round*.json, so that
    readers of the keys never see it."""
    return keys_dir() / f"draw{round_no}.json"


def sheet_file(sheet: str) -> str:
    """File name of a sheet as sent to its reader: distinct from P01's spot-check sheets."""
    return f"validation_{sheet}.csv"


def sheet_cell(text: str) -> str:
    """Text as it goes into a sheet cell: LF line ends, and a leading space before anything a
    spreadsheet would run as a formula (a response starting with "- " is common)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " " + text if text.startswith(_FORMULA_START) else text


def normalise(text: str) -> str:
    """Whitespace-insensitive form used to check that a filled sheet still shows the same text."""
    return " ".join(text.replace(_BOM, "").split())


def _locked(path: Path) -> ValidateError:
    return ValidateError(f"cannot write {path}: it is open in another program (Excel?). Close it and "
                         "run the command again")


def check_writable(targets: Iterable[Path]) -> None:
    """Fail before writing anything if a target is locked (Windows locks files open in Excel)."""
    for path in targets:
        if path.is_file():
            try:
                with open(path, "a"):
                    pass
            except PermissionError:
                raise _locked(path) from None


def _replace(tmp: str, path: Path) -> None:
    try:
        os.replace(tmp, path)
    except PermissionError:
        Path(tmp).unlink(missing_ok=True)
        raise _locked(path) from None


def write_sheet(path: Path, header_line: str, columns: Sequence[str],
                rows: Iterable[Sequence[str]]) -> Path:
    """Write a sheet atomically: a one-cell comment row, the column names, then the rows.

    UTF-8 with a BOM so Excel shows Hindi and Chinese correctly (Google Sheets ignores it), and LF
    line ends like every other file in the project.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([header_line])
            writer.writerow(list(columns))
            for row in rows:
                writer.writerow([sheet_cell(str(cell)) for cell in row])
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _replace(tmp, path)
    return path


@dataclass
class FilledSheet:
    rows: list[tuple[int, dict[str, str]]]   # (spreadsheet row number, {column: cell})
    utf8: bool                               # False: cells were decoded with replacement characters
    delimiter: str


def _decode(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    if raw[:4] == b"PK\x03\x04":
        raise ValidateError(f"{path} is a spreadsheet workbook, not a CSV file: in Google Sheets use "
                            "File > Download > Comma-separated values; in Excel save as 'CSV UTF-8'")
    try:
        return raw.decode("utf-8-sig"), True
    except UnicodeDecodeError:
        # Excel's plain "CSV" is saved in the Windows code page. Ids and labels are ASCII and survive;
        # the prompt and response cells do not, so the caller skips the text check.
        return raw.decode("utf-8", errors="replace").lstrip(_BOM), False


def _header_index(rows: list[list[str]], required: Sequence[str]) -> int | None:
    for i, row in enumerate(rows):
        cells = [c.replace(_BOM, "").strip().lower() for c in row]
        if not any(cells) or cells[0].startswith("#"):
            continue
        return i if all(c in cells for c in required) else None
    return None


def read_sheet(path: Path, required: Sequence[str]) -> FilledSheet:
    """Read a filled sheet.

    Skips the comment rows before the header (a Google Sheets download pads them with empty
    cells), matches columns by name (case and spaces ignored, extra columns allowed), drops rows
    that are entirely empty, and accepts the semicolon or tab delimiters some Excel locales save.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidateError(f"{path}: no such file")
    text, utf8 = _decode(path)
    try:
        for delimiter in (",", ";", chr(9)):           # comma, semicolon, tab
            rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
            header_at = _header_index(rows, required)
            if header_at is not None:
                break
        else:
            raise ValidateError(f"{path}: no header row with the columns {', '.join(required)}")
    except csv.Error as e:
        raise ValidateError(f"{path}: not a readable CSV file ({e})") from None
    columns = [c.replace(_BOM, "").strip().lower() for c in rows[header_at]]
    named = [c for c in columns if c]
    if len(set(named)) != len(named):
        raise ValidateError(f"{path}: row {header_at + 1} repeats a column name")
    out = []
    for i in range(header_at + 1, len(rows)):
        row = rows[i]
        if not any(c.strip() for c in row):
            continue
        out.append((i + 1, {col: (row[j] if j < len(row) else "")
                            for j, col in enumerate(columns) if col}))
    return FilledSheet(rows=out, utf8=utf8, delimiter=delimiter)


def write_json(path: Path, doc: Any) -> Path:
    """Atomic, LF-only, readable JSON (keys files, sidecars, metrics)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _replace(tmp, path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except PermissionError:
        raise _locked(path) from None
    return path
