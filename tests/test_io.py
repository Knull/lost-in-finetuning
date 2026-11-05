"""lift.io's JSONL helpers, in particular the resume its docstring documents: append_jsonl plus
existing_keys after a job was killed in the middle of an append. Placeholder rows only."""

from lift.io import append_jsonl, existing_keys, read_jsonl, stable_json

ROWS = [{"key": k, "pid": f"prh-test-000{i}", "output": f"placeholder output {k} " * 8}
        for i, k in enumerate("abcd")]


def resume(path, rows):
    """The resume lift.io documents: skip the keys existing_keys finds, append the rest."""
    done = existing_keys(path)
    return append_jsonl(path, [r for r in rows if r["key"] not in done])


def killed_append(path, row, cut):
    """What an append_jsonl killed while writing ``row`` leaves behind: the first ``cut`` bytes of its
    line, without the newline."""
    line = (stable_json(row) + "\n").encode("utf-8")
    assert 0 < cut < len(line)
    with open(path, "ab") as f:
        f.write(line[:cut])


# R1-26: append_jsonl wrote the first resumed record onto the end of the cut line, so that record was
# lost and read_jsonl refused the file ("line 3: bad JSON line").
def test_a_resume_after_an_append_killed_mid_line_keeps_every_record(tmp_path):
    path = tmp_path / "scores.jsonl"
    append_jsonl(path, ROWS[:2])
    killed_append(path, ROWS[2], len(stable_json(ROWS[2])) // 2)
    assert existing_keys(path) == {"a", "b"}                       # the cut line is ignored, not fatal

    assert resume(path, ROWS) == 2                                  # c and d are written again
    assert existing_keys(path) == {"a", "b", "c", "d"}, "record c went onto the end of the cut line"
    assert list(read_jsonl(path)) == ROWS                           # one readable record per row


# R1-26: a cut just before the newline leaves a whole record, which existing_keys counts as written; the
# next record went onto the end of its line, and both were lost.
def test_a_resume_after_an_append_killed_before_its_newline_keeps_every_record(tmp_path):
    path = tmp_path / "scores.jsonl"
    append_jsonl(path, ROWS[:2])
    killed_append(path, ROWS[2], len(stable_json(ROWS[2])))         # all of c but its newline
    assert existing_keys(path) == {"a", "b", "c"}                  # c parses, so it counts as written

    assert resume(path, ROWS) == 1                                  # d only
    assert existing_keys(path) == {"a", "b", "c", "d"}, "record d went onto the end of c's line"
    assert list(read_jsonl(path)) == ROWS


# R1-26: existing_keys read the file as text, so a cut inside a multi-byte character (two cut points
# in three inside Devanagari) raised UnicodeDecodeError instead of being ignored.
def test_a_line_cut_inside_a_multibyte_character_is_ignored_then_dropped(tmp_path):
    rows = [{"key": k, "output": "नमस्ते दुनिया " * 10} for k in "ab"]
    path = tmp_path / "scores_hi.jsonl"
    append_jsonl(path, rows[:1])
    line = stable_json(rows[1]).encode("utf-8")
    killed_append(path, rows[1], line.index("न".encode()) + 1)    # one byte into a 3-byte character
    assert existing_keys(path) == {"a"}

    assert resume(path, rows) == 1
    assert list(read_jsonl(path)) == rows


def test_a_cut_line_longer_than_one_scan_block_or_alone_in_its_file_is_dropped(tmp_path):
    # The tail is scanned backwards in 64 KiB blocks; this cut line spans several.
    rows = [{"key": "a", "output": "placeholder"}, {"key": "b", "output": "placeholder " * 20_000}]
    path = tmp_path / "long.jsonl"
    append_jsonl(path, rows[:1])
    killed_append(path, rows[1], 150_000)
    assert resume(path, rows) == 1
    assert list(read_jsonl(path)) == rows
    # A file whose only line was cut, mid-line or just before its newline.
    for name, cut in (("mid.jsonl", 10), ("whole.jsonl", len(stable_json(ROWS[0])))):
        path = tmp_path / name
        killed_append(path, ROWS[0], cut)
        resume(path, ROWS[:2])
        assert list(read_jsonl(path)) == ROWS[:2]


def test_a_file_that_ends_in_a_newline_is_only_appended_to(tmp_path):
    path = tmp_path / "scores.jsonl"
    assert append_jsonl(path, ROWS[:2]) == 2
    before = path.read_bytes()
    assert append_jsonl(path, []) == 0 and path.read_bytes() == before
    assert append_jsonl(path, ROWS[2:]) == 2
    data = path.read_bytes()
    assert data.startswith(before) and data.endswith(b"\n") and b"\r" not in data
    assert list(read_jsonl(path)) == ROWS


# R1-26: a line that parsed to something other than an object (a list, a string, a number, null)
# made existing_keys raise TypeError.
def test_existing_keys_skips_lines_that_are_not_records(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(b'{"key": "a"}\n[1, 2]\n"text"\n42\nnull\n{"other": "b"}\n\n{"key": "c"}\n')
    assert existing_keys(path) == {"a", "c"}
    assert existing_keys(path, field="other") == {"b"}
    assert existing_keys(tmp_path / "missing.jsonl") == set()


# R1-26 follow-up: a record whose field held a list or an object, which no set can hold, made existing_keys
# raise TypeError (unhashable type) instead of passing the line over like the other non-records.
def test_existing_keys_skips_records_whose_field_is_a_list_or_an_object(tmp_path):
    path = tmp_path / "unhashable.jsonl"
    path.write_bytes(b'{"key": "a"}\n{"key": ["x"]}\n{"key": {"k": "y"}}\n{"key": []}\n{"key": "b"}\n')
    assert existing_keys(path) == {"a", "b"}
