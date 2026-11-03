"""lift validate import: a filled round 1 sheet becomes HumanLabel records (P04, Build 3).

The filled files are made from the fake world's sheets the way readers send them back: a Google
Sheets download, Excel files with other delimiters or code pages, and the usual slips (a typo in a
label, a repeated or foreign id, an edited cell, a column sorted on its own).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import pytest
from validation_fakes import (
    SAFETY_SETS,
    World,
    build_world,
    fill_sheet,
    make_generation,
    other_label,
    output_text,
    pid_of,
    prompt_text,
    read_csv,
    truth,
    use_world,
    write_generations,
)

from lift.cli import main
from lift.io import read_jsonl, write_jsonl
from lift.schemas import LABELS, HumanLabel, PromptRecord
from lift.validation.labels import import_labels
from lift.validation.sample import SHEET_COLUMNS, load_round, make_sheets, sample_digest
from lift.validation.store import ValidateError

VARIANTS = ("stock", "adapter-aya-std-s1")   # round 1: stock and the seed-1 standard adapter
N_DEV = 34                                   # dev prompts per set: enough for hi + en, one reader
ID, PROMPT, RESPONSE, LABEL, NOTE = (SHEET_COLUMNS.index(c)
                                     for c in ("item_id", "prompt", "response", "label", "note"))
# Neutral words ("question", "answer") in the script each language's real sheets are written in.
NATIVE = {"hi": ("प्रश्न", "उत्तर"), "es": ("pregunta número", "respuesta está")}


def round1_world(root: Path, langs=("hi", "zh", "es", "en")) -> World:
    return build_world(root, n_dev=N_DEV, n_final=0, langs=langs, dev_variants=VARIANTS,
                       final_variants=(), judgements=False, utility=False)


def write_native(world: World, lang: str, question: str, answer: str) -> None:
    """Prompts and outputs for ``lang`` that start with a word in its own script."""
    for set_name in SAFETY_SETS:
        prompts = [PromptRecord(pid=pid, lang=lang, set=set_name, split="dev", source="fake",
                                text=f"{question} {prompt_text(set_name, lang, pid)}")
                   for pid in (pid_of(set_name, i) for i in range(N_DEV))]
        write_jsonl(world.data / "prompts" / f"{set_name}_{lang}.jsonl", [p.model_dump() for p in prompts])
        for variant in VARIANTS:
            write_generations(world, variant, set_name, lang, "dev", [
                make_generation(variant, p, output=f"{answer} {output_text(variant, lang, p.pid)}")
                for p in prompts])


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    """Round 1 drawn once, with the Hindi overlap sheet; every test works on its own copy."""
    root = tmp_path_factory.mktemp("round1")
    with pytest.MonkeyPatch.context() as mp:
        use_world(mp, round1_world(root))
        make_sheets(1, hindi_overlap=True)
    return root


@pytest.fixture(scope="module")
def native_template(tmp_path_factory) -> Path:
    """Round 1 whose Hindi and Spanish cells hold Devanagari and accented letters, as real ones do."""
    root = tmp_path_factory.mktemp("native")
    with pytest.MonkeyPatch.context() as mp:
        world = use_world(mp, round1_world(root, langs=("zh", "en")))
        for lang, (question, answer) in NATIVE.items():
            write_native(world, lang, question, answer)
        make_sheets(1, hindi_overlap=False)
    return root


def copy_world(template: Path, tmp_path: Path, monkeypatch) -> World:
    """A private copy of the template's validation/ tree. An import reads only the round's keys
    and the filled file, so data/ and outputs/ point at empty folders of this test."""
    root = tmp_path / "world"
    shutil.copytree(template / "validation", root / "validation")
    return use_world(monkeypatch, World(root=root))


@pytest.fixture
def world(template, tmp_path, monkeypatch) -> World:
    return copy_world(template, tmp_path, monkeypatch)


@pytest.fixture
def native_world(native_template, tmp_path, monkeypatch) -> World:
    return copy_world(native_template, tmp_path, monkeypatch)


def sheet_path(world: World, sheet: str) -> Path:
    return world.validation / "sheets" / f"validation_{sheet}.csv"


def filled(world: World, sheet: str = "round1_zh") -> list[list[str]]:
    """The sheet's rows (comment, header, items) with every label filled in correctly.
    rows[i] is row i + 1 of the spreadsheet, so the first item is row 3."""
    rows = read_csv(sheet_path(world, sheet))
    key = {it["item_id"]: it["key"] for it in load_round(1)["items"]}
    for row in rows[2:]:
        row[LABEL] = truth(key[row[ID]])
    return rows


def expected(rows: list[list[str]]) -> dict[str, str]:
    return {row[ID]: row[LABEL].strip().lower() for row in rows[2:] if row[LABEL].strip()}


def write(path: Path, rows, *, delimiter=",", encoding="utf-8-sig", errors="strict",
          newline="\n") -> Path:
    with open(path, "w", encoding=encoding, errors=errors, newline="") as f:
        csv.writer(f, delimiter=delimiter, lineterminator=newline).writerows(rows)
    return path


def excel_plain_csv(path: Path, rows) -> Path:
    """Excel's "CSV (Comma delimited)" on a Windows PC in the usual Western code page: cp1252,
    CRLF, no BOM, and "?" for every character cp1252 lacks."""
    return write(path, rows, encoding="cp1252", errors="replace", newline="\r\n")


def imported(world: World, sheet: str = "round1_zh") -> dict[str, HumanLabel]:
    path = world.validation / "labels" / f"{sheet}.jsonl"
    return {r.item_id: r for r in (HumanLabel.model_validate(x) for x in read_jsonl(path))}


def labels_of(world: World, sheet: str = "round1_zh") -> dict[str, str]:
    return {i: r.label for i, r in imported(world, sheet).items()}


def meta_of(world: World, sheet: str = "round1_zh") -> dict:
    return json.loads((world.validation / "labels" / f"{sheet}.meta.json").read_text(encoding="utf-8"))


def label_files(world: World) -> list[str]:
    folder = world.validation / "labels"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def problems(err: pytest.ExceptionInfo, path: Path) -> str:
    """The error text without the file's path (a temporary path can hold any number)."""
    return str(err.value).replace(str(path), "<file>")


def names_row(message: str, n: int) -> bool:
    return re.search(rf"\brow {n}\b", message) is not None


def sheet_texts(world: World) -> set[str]:
    """Every prompt and response cell of the round's sheets."""
    return {row[col] for path in (world.validation / "sheets").glob("*.csv")
            for row in read_csv(path)[2:] for col in (PROMPT, RESPONSE)}


def quotes_sheet_text(message: str, texts) -> bool:
    """Whether ``message`` holds one of ``texts``, or the hash tag that makes each fake text unique."""
    return any(t in message for t in texts if t) or any(
        token in message for t in texts for token in t.split() if len(token) >= 10)


# A correct file -------------------------------------------------------------------------------


def test_a_filled_sheet_becomes_human_labels_and_a_meta_file(world, tmp_path):
    rows = filled(world)
    want = expected(rows)
    typed = dict(want)
    first, second, third, noted = (row[ID] for row in rows[2:6])
    typed[first] = f" {want[first].title()} "               # case and surrounding spaces do not count
    typed[second] = want[second].upper()
    typed[third] = f"\t{want[third]}\n"
    path = fill_sheet(sheet_path(world, "round1_zh"), tmp_path / "filled.csv", typed,
                      notes={noted: "the reply is in another language"})
    out = import_labels(1, "zh", path)

    labels = imported(world)
    assert {i: r.label for i, r in labels.items()} == want
    assert {r.annotator for r in labels.values()} == {"zh-a"}
    assert labels[noted].note == "the reply is in another language"
    assert out["warnings"] == []
    meta = meta_of(world)
    assert meta["skipped"] == meta["unlabelled"] == meta["missing"] == []
    assert meta["label_counts"] == {c: list(want.values()).count(c) for c in LABELS}
    assert sample_digest(load_round(1)) in meta.values()     # the draw these labels belong to


def test_the_cli_imports_a_file_and_prints_only_ids_and_counts(world, tmp_path, capsys):
    texts = sheet_texts(world)
    rows = filled(world)
    rows[2][LABEL], rows[2][NOTE] = "", "skipped"
    good = write(tmp_path / "good.csv", rows)
    assert main(["validate", "import", "--round", "1", "--lang", "zh", "--file", str(good)]) == 0
    assert labels_of(world) == expected(rows)
    printed = capsys.readouterr().out
    rows[3][LABEL] = "harmfull"
    bad = write(tmp_path / "bad.csv", rows)
    assert main(["validate", "import", "--round", "1", "--lang", "zh", "--file", str(bad)]) == 1
    error = capsys.readouterr().out
    assert names_row(error.replace(str(bad), "<file>"), 4)
    assert not quotes_sheet_text(printed + error, texts)


# Errors: reported together, nothing written ----------------------------------------------------


@pytest.mark.parametrize("bad", ["harmfull", "safe", "refusal / benign", "1"])
def test_a_label_outside_the_four_classes_names_its_row_and_writes_nothing(world, tmp_path, bad):
    rows = filled(world)
    rows[9][LABEL] = bad
    path = write(tmp_path / "filled.csv", rows)
    with pytest.raises(ValidateError) as err:
        import_labels(1, "zh", path)
    assert names_row(problems(err, path), 10)
    assert label_files(world) == []


def test_a_failed_import_leaves_the_earlier_labels_untouched(world, tmp_path):
    rows = filled(world)
    import_labels(1, "zh", write(tmp_path / "first.csv", rows))
    before = {p.name: p.read_bytes() for p in (world.validation / "labels").iterdir()}
    rows[3][LABEL] = other_label(rows[3][LABEL])          # a change the reader wants...
    rows[9][LABEL] = "harmfull"                           # ...and a typo, which stops the whole file
    with pytest.raises(ValidateError):
        import_labels(1, "zh", write(tmp_path / "second.csv", rows))
    assert {p.name: p.read_bytes() for p in (world.validation / "labels").iterdir()} == before


def test_rows_are_numbered_as_the_spreadsheet_shows_them(world, tmp_path):
    rows = filled(world)
    for row in rows[2:10]:
        row[RESPONSE] = row[RESPONSE].replace(" ", "\n\n")   # multi-line cells, same text
    rows[11][LABEL] = "harmfull"                             # row 12 of the sheet, line 28 of the file
    path = write(tmp_path / "filled.csv", rows)
    with pytest.raises(ValidateError) as err:
        import_labels(1, "zh", path)
    assert names_row(problems(err, path), 12)


def test_a_repeated_id_is_an_error(world, tmp_path):
    rows = filled(world)
    again = list(rows[6])
    again[LABEL] = other_label(again[LABEL])     # the reader "corrects" row 7 at the bottom
    rows.append(again)                           # row 43
    path = write(tmp_path / "filled.csv", rows)
    with pytest.raises(ValidateError) as err:
        import_labels(1, "zh", path)
    msg = problems(err, path)
    assert names_row(msg, 43) and names_row(msg, 7)
    assert label_files(world) == []


@pytest.mark.parametrize(("lang", "source"), [
    ("zh", None),               # an id on no sheet at all
    ("zh", "round1_hi"),        # another language's sheet of the same round
    ("zh", "round1_hi_b"),      # the Hindi overlap sheet
    ("hi", "round1_en"),        # the other language the Hindi reader labels
])
def test_an_id_that_is_not_on_this_languages_sheet_is_an_error(world, tmp_path, lang, source):
    rows = filled(world, f"round1_{lang}")
    foreign = read_csv(sheet_path(world, source))[2][ID] if source else f"r1{lang}-000000"
    assert foreign not in {row[ID] for row in rows[2:]}
    rows[5][ID] = foreign
    path = write(tmp_path / "filled.csv", rows)
    with pytest.raises(ValidateError) as err:
        import_labels(1, lang, path)
    msg = problems(err, path)
    assert names_row(msg, 6) and foreign in msg
    assert label_files(world) == []


def test_a_sheet_imported_under_another_language_is_refused(world, tmp_path):
    with pytest.raises(ValidateError):
        import_labels(1, "zh", write(tmp_path / "es.csv", filled(world, "round1_es")))
    assert label_files(world) == []


@pytest.mark.parametrize(("sheet", "stray"), [("round1_hi", "round1_hi_b"), ("round1_hi_b", "round1_hi")])
def test_a_file_mixing_the_main_and_overlap_sheets_is_an_error(world, tmp_path, sheet, stray):
    rows = filled(world, sheet)
    rows.append(filled(world, stray)[2])         # a correct row, but of the other Hindi sheet
    with pytest.raises(ValidateError):
        import_labels(1, "hi", write(tmp_path / "filled.csv", rows))
    assert label_files(world) == []


def test_every_problem_of_a_file_is_reported_in_one_error(world, tmp_path):
    rows = filled(world)
    rows[2][LABEL] = "harmfull"                               # row 3: not a label
    rows[4][ID] = filled(world, "round1_hi")[2][ID]           # row 5: a Hindi id
    rows[8][ID] = "r1zh-000000"                               # row 9: an id on no sheet
    again = list(rows[6])
    again[LABEL] = other_label(again[LABEL])
    rows.append(again)                                        # row 43: repeats row 7
    path = write(tmp_path / "filled.csv", rows)
    with pytest.raises(ValidateError) as err:
        import_labels(1, "zh", path)
    msg = problems(err, path)
    for row in (3, 5, 9, 43):
        assert names_row(msg, row), row
    assert label_files(world) == []


# Empty labels ------------------------------------------------------------------------------------


def test_empty_labels_are_left_out_and_counted(world, tmp_path):
    rows = filled(world)
    skipped = sorted([rows[2][ID], rows[3][ID]])
    unlabelled = sorted([rows[4][ID], rows[5][ID]])
    missing = rows[6][ID]
    rows[2][LABEL], rows[2][NOTE] = "", "skipped"
    rows[3][LABEL], rows[3][NOTE] = "", "Skipped, too upsetting"
    rows[4][LABEL] = ""                                                   # not done yet
    rows[5][LABEL], rows[5][NOTE] = "  ", "unsure between refusal and benign"
    del rows[6]                                                           # deleted by the reader
    import_labels(1, "zh", write(tmp_path / "filled.csv", rows))
    labels = labels_of(world)
    assert len(labels) == 35 and labels == expected(rows)
    assert not set(labels) & {*skipped, *unlabelled, missing}
    meta = meta_of(world)
    assert sorted(meta["skipped"]) == skipped
    assert sorted(meta["unlabelled"]) == unlabelled
    assert list(meta["missing"]) == [missing]


# Reading the files readers send back -----------------------------------------------------------


def test_a_google_sheets_download_imports_cleanly(world, tmp_path):
    rows = filled(world)
    src = sheet_path(world, "round1_zh")
    dst = fill_sheet(src, tmp_path / "download.csv", expected(rows), google_export=True)
    raw = dst.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and raw.split(b"\r\n")[0].endswith(b",,,,")
    out = import_labels(1, "zh", dst)
    assert labels_of(world) == expected(rows)
    assert out["warnings"] == []


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_an_excel_file_with_another_delimiter_imports(world, tmp_path, delimiter):
    rows = filled(world)
    import_labels(1, "zh", write(tmp_path / "filled.csv", rows, delimiter=delimiter, newline="\r\n"))
    assert labels_of(world) == expected(rows)
    assert meta_of(world)["changed_text"] == []      # the cells were split where the sheet had them


def test_a_file_that_is_not_utf8_imports_its_labels_with_a_warning(native_world, tmp_path):
    rows = filled(native_world, "round1_es")
    path = excel_plain_csv(tmp_path / "es.csv", rows)
    with pytest.raises(UnicodeDecodeError):
        path.read_bytes().decode("utf-8")
    out = import_labels(1, "es", path)
    assert labels_of(native_world, "round1_es") == expected(rows)
    assert any("UTF-8" in w for w in out["warnings"])


def test_a_hindi_sheet_saved_by_excel_as_plain_csv_is_handled(native_world, tmp_path):
    rows = filled(native_world, "round1_hi")
    path = excel_plain_csv(tmp_path / "hi.csv", rows)
    assert path.read_bytes().isascii()                  # every Devanagari letter became "?"
    try:
        out = import_labels(1, "hi", path)
    except ValidateError as e:
        # Refusing would be defensible, but then the message must point at the encoding rather
        # than at "another version of the sheet".
        assert "UTF-8" in str(e)
    else:
        assert labels_of(native_world, "round1_hi") == expected(rows)
        assert out["warnings"]


@pytest.mark.parametrize("name", ["filled.xlsx", "filled.csv"])
def test_a_workbook_is_refused_with_a_hint_to_save_as_csv(world, tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"PK\x03\x04" + bytes(60))          # how every .xlsx (a zip file) starts
    with pytest.raises(ValidateError) as err:
        import_labels(1, "zh", path)
    assert "CSV" in problems(err, path)
    assert label_files(world) == []


# The prompt and response cells -------------------------------------------------------------------


def test_a_few_edited_cells_are_a_warning(world, tmp_path):
    rows = filled(world)
    rows[3][PROMPT] += " (typo fixed)"
    rows[10][RESPONSE] = "something else"
    out = import_labels(1, "zh", write(tmp_path / "filled.csv", rows))
    assert labels_of(world) == expected(rows)                 # their labels still count
    assert any(rows[3][ID] in w and rows[10][ID] in w for w in out["warnings"])
    assert sorted(meta_of(world)["changed_text"]) == sorted([rows[3][ID], rows[10][ID]])


@pytest.mark.parametrize(("sheet", "edited", "stops"), [
    ("round1_hi_b", 3, False), ("round1_hi_b", 4, True),     # 20 rows: the floor of 3
    ("round1_zh", 4, False), ("round1_zh", 5, True),         # 40 rows: 10% is 4
    ("round1_hi", 6, False), ("round1_hi", 7, True),         # 60 rows: 10% is 6
])
def test_more_edited_rows_than_max_3_or_10_percent_stop_the_import(world, tmp_path, sheet, edited, stops):
    rows = filled(world, sheet)
    for row in rows[2:2 + edited]:
        row[RESPONSE] += " edited"
    path = write(tmp_path / "filled.csv", rows)
    lang = sheet.split("_")[1]
    if stops:
        with pytest.raises(ValidateError):
            import_labels(1, lang, path)
        assert label_files(world) == []
    else:
        out = import_labels(1, lang, path)
        assert labels_of(world, sheet) == expected(rows)
        assert out["warnings"]


def test_whitespace_differences_are_not_edits(world, tmp_path):
    rows = filled(world)
    for row in rows[2:]:
        row[PROMPT] = "  " + row[PROMPT].replace(" ", "\n  ") + " "
        row[RESPONSE] = row[RESPONSE].replace(" ", " \t ") + "\r\n"
    out = import_labels(1, "zh", write(tmp_path / "filled.csv", rows))
    assert out["warnings"] == []
    assert labels_of(world) == expected(rows)


def test_an_id_column_sorted_on_its_own_stops_the_import(world, tmp_path):
    rows = filled(world)
    for row, item_id in zip(rows[2:], sorted(row[ID] for row in rows[2:]), strict=True):
        row[ID] = item_id
    with pytest.raises(ValidateError):
        import_labels(1, "zh", write(tmp_path / "filled.csv", rows))
    assert label_files(world) == []


def test_a_label_column_sorted_on_its_own_is_a_warning(world, tmp_path):
    rows = filled(world)
    labels = [row[LABEL] for row in rows[2:]]
    assert len(set(labels)) >= 3 and labels != sorted(labels)
    assert import_labels(1, "zh", write(tmp_path / "as_made.csv", rows))["warnings"] == []
    for row, label in zip(rows[2:], sorted(labels), strict=True):
        row[LABEL] = label
    out = import_labels(1, "zh", write(tmp_path / "sorted.csv", rows))
    assert out["warnings"]
    assert labels_of(world) == expected(rows)                 # a warning, not an error


# Re-imports and the overlap sheet --------------------------------------------------------------


def test_reimporting_a_sheet_replaces_its_labels(world, tmp_path):
    rows = filled(world)
    import_labels(1, "zh", write(tmp_path / "first.csv", rows))
    for row in rows[3:6]:
        row[LABEL] = other_label(row[LABEL])        # the reader changed their mind
    rows[6][LABEL] = ""                             # and emptied a row
    import_labels(1, "zh", write(tmp_path / "second.csv", rows))
    assert labels_of(world) == expected(rows)


def test_the_overlap_sheet_is_imported_for_the_second_hindi_reader(world, tmp_path):
    main_rows, overlap_rows = filled(world, "round1_hi"), filled(world, "round1_hi_b")
    import_labels(1, "hi", write(tmp_path / "hi_b.csv", overlap_rows))
    import_labels(1, "hi", write(tmp_path / "hi.csv", main_rows))
    overlap, main_sheet = imported(world, "round1_hi_b"), imported(world, "round1_hi")
    assert {i: r.label for i, r in overlap.items()} == expected(overlap_rows)
    assert {i: r.label for i, r in main_sheet.items()} == expected(main_rows)
    assert {r.annotator for r in overlap.values()} == {"hi-b"}
    assert {r.annotator for r in main_sheet.values()} == {"hi-a"}
    import_labels(1, "hi", write(tmp_path / "again.csv", overlap_rows), annotator="hi-c")
    assert {r.annotator for r in imported(world, "round1_hi_b").values()} == {"hi-c"}


# Nothing prints sheet text -----------------------------------------------------------------------


def test_import_messages_never_quote_a_prompt_or_response(world, tmp_path):
    texts = sheet_texts(world)
    messages = []

    def refused(rows, lang="zh"):
        path = write(tmp_path / f"refused{len(messages)}.csv", rows)
        with pytest.raises(ValidateError) as err:
            import_labels(1, lang, path)
        messages.append(str(err.value))

    rows = filled(world)                              # every kind of row problem at once
    rows[2][LABEL] = "harmfull"
    rows[4][ID] = "r1zh-000000"
    rows.append(list(rows[6]))
    refused(rows)
    rows = filled(world)                              # the id column sorted on its own
    for row, item_id in zip(rows[2:], sorted(row[ID] for row in rows[2:]), strict=True):
        row[ID] = item_id
    refused(rows)
    rows = filled(world, "round1_hi")                 # the main and overlap sheets in one file
    rows.append(filled(world, "round1_hi_b")[2])
    refused(rows, "hi")
    rows = filled(world)                              # imported, with warnings about edits and gaps
    rows[3][PROMPT] += " edited"
    rows[4][LABEL] = ""
    del rows[5]
    warnings = import_labels(1, "zh", write(tmp_path / "warned.csv", rows))["warnings"]
    assert warnings
    for msg in [*messages, *warnings]:
        assert not quotes_sheet_text(msg, texts), msg


@pytest.mark.parametrize("column", ["label", "item_id"])
def test_a_prompt_or_response_pasted_into_the_wrong_cell_is_not_quoted(world, tmp_path, column):
    rows = filled(world)
    col = SHEET_COLUMNS.index(column)
    pasted = [rows[4][RESPONSE], rows[6][PROMPT]]
    rows[4][col], rows[6][col] = pasted
    path = write(tmp_path / "filled.csv", rows)
    with pytest.raises(ValidateError) as err:
        import_labels(1, "zh", path)
    msg = problems(err, path)
    assert names_row(msg, 5) and names_row(msg, 7)          # still reported, by row number
    assert not quotes_sheet_text(msg, pasted)
