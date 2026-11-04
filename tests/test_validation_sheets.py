"""The blind validation sheets (P04, Build 2): one CSV per language and round, nothing in them that
tells a reader the model, the set or a judge's call, the keys kept apart, and cells a spreadsheet
will not run as formulas. Every text is a neutral placeholder from validation_fakes."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import mean

import pytest
from validation_fakes import (
    DEV_VARIANTS,
    FINAL_VARIANTS,
    FRONTIER_ID,
    LANGS,
    PRIMARY_ID,
    SAFETY_SETS,
    UTILITY_ID,
    build_world,
    fill_sheet,
    make_generation,
    output_text,
    prompt_text,
    read_csv,
    read_round,
    use_world,
    write_generations,
)

from lift import paths
from lift.cli import main
from lift.io import read_jsonl, sha256_text, write_jsonl
from lift.schemas import LABELS, PromptRecord
from lift.validation import sample, store, utility

BOM = b"\xef\xbb\xbf"
COLUMNS = ["item_id", "prompt", "response", "label", "note"]
UTILITY_COLUMNS = ["item_id", "prompt", "response_a", "response_b", "preference", "note"]
SIZES = {1: {"hi": 60, "zh": 40, "es": 40, "en": 40}, 2: {"hi": 60, "zh": 20, "es": 20, "en": 20}}
ROUND1_VARIANTS = ("stock", "adapter-aya-std-s1")
CANDIDATE = "adapter-aya-std-s1"
FORMULA_STARTS = ("=", "+", "-", "@")
# What a reader must never learn from a sheet. The comment row may name the four labels (it tells
# the reader what to type), so it is checked against everything except them.
VARIANT_WORDS = (*DEV_VARIANTS, *FINAL_VARIANTS, "stock", "adapter", "wiseft", "resta", "cast", "repaired")
SET_WORDS = ("easy_benign", "hard_benign", "easy benign", "hard benign")
JUDGE_WORDS = (PRIMARY_ID, FRONTIER_ID, "qwen", "guard", "frontier", "fakeprovider", "controversial",
               "unsafe")


def no_git(mp: pytest.MonkeyPatch) -> None:
    # make_sheets records the git sha by running git twice; nothing on a sheet depends on it.
    mp.setattr("lift.manifest.git_sha", lambda: "test")


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    no_git(monkeypatch)


def small_world(root: Path):
    """Enough for round 1 only: stock and one seed-1 adapter on 40 dev prompts per set."""
    return build_world(root, n_dev=40, n_final=0, dev_variants=ROUND1_VARIANTS, final_variants=(),
                       judgements=False, utility=False)


@pytest.fixture(scope="module")
def _drawn(tmp_path_factory):
    world = build_world(tmp_path_factory.mktemp("drawn"), utility=False)
    with pytest.MonkeyPatch.context() as mp:
        no_git(mp)
        use_world(mp, world)
        sample.make_sheets(1, hindi_overlap=True)
        sample.make_sheets(2)
    return world


@pytest.fixture
def drawn(_drawn, monkeypatch):
    """The default world (every variant, both judges) with round 1 (and its Hindi overlap sheet)
    and round 2 drawn. Built once per module; the tests that use it only read it."""
    return use_world(monkeypatch, _drawn)


def body(path: Path) -> list[list[str]]:
    """The data rows of a sheet: everything after the comment row and the column names."""
    return read_csv(path)[2:]


def sheet_paths(world) -> list[Path]:
    return sorted((world.validation / "sheets").glob("*.csv"))


def prompt_texts(world) -> dict[tuple[str, str, str], str]:
    return {(r["set"], r["lang"], r["pid"]): r["text"]
            for path in (world.data / "prompts").glob("*.jsonl") for r in read_jsonl(path)}


def shown(world, round_no: int, file: str) -> list[tuple[list[str], dict, object, str]]:
    """(row, keys entry, generation, prompt text) for every data row of a sheet, via the keys."""
    doc = read_round(world.validation, round_no)
    items = {it["item_id"]: it for it in doc["items"]}
    gens, texts = world.by_key(), prompt_texts(world)
    out = []
    for row in body(world.validation / "sheets" / file):
        gen = gens[items[row[0]]["key"]]
        out.append((row, items[row[0]], gen, texts[(gen.set, gen.lang, gen.pid)]))
    return out


def rewrite_hindi(world, prompt_of, output_of) -> None:
    """New texts for every Hindi prompt and round 1 output (the generations stay current)."""
    for set_name in SAFETY_SETS:
        path = world.data / "prompts" / f"{set_name}_hi.jsonl"
        prompts = [PromptRecord.model_validate(r) for r in read_jsonl(path)]
        prompts = [p.model_copy(update={"text": prompt_of(p)}) for p in prompts]
        write_jsonl(path, [p.model_dump() for p in prompts])
        dev = [p for p in prompts if p.split == "dev"]
        for variant in ROUND1_VARIANTS:
            gens = [make_generation(variant, p, output=output_of(i, p)) for i, p in enumerate(dev)]
            write_generations(world, variant, set_name, "hi", "dev", gens)


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings_in(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from strings_in(v)


def leaks(cells, words) -> list[tuple[str, str]]:
    return [(w, c) for c in cells for w in words if w.lower() in c.lower()]


def test_one_sheet_per_language_and_round_with_the_header_and_columns(drawn):
    sheets = drawn.validation / "sheets"
    assert {p.name for p in sheets.iterdir()} == {
        f"validation_round{r}_{lang}.csv" for r in SIZES for lang in LANGS} | {"validation_round1_hi_b.csv"}
    for round_no, sizes in SIZES.items():
        for lang, n in sizes.items():
            path = sheets / f"validation_round{round_no}_{lang}.csv"
            raw = path.read_bytes()
            assert raw.startswith(BOM) and b"\r" not in raw
            comment, columns, *rows = read_csv(path)
            assert len(comment) == 1 and comment[0].startswith("#")
            assert "validation/guidelines.md" in comment[0]
            assert columns == COLUMNS
            assert len(rows) == n
            assert all(len(row) == 5 and row[3] == "" and row[4] == "" for row in rows)


def test_cells_show_the_full_prompt_and_output_and_the_keys_say_which_generation(drawn):
    gens = drawn.by_key()
    assert not list((drawn.validation / "sheets").glob("*.json"))
    for round_no in SIZES:
        doc = read_round(drawn.validation, round_no)
        items = {it["item_id"]: it for it in doc["items"]}
        assert len(items) == len(doc["items"])
        on_sheets = []
        for info in doc["sheets"].values():
            rows = shown(drawn, round_no, info["file"])
            on_sheets += [row[0] for row, *_ in rows]
            for row, it, gen, text in rows:
                assert (it["key"], it["variant"], it["set"], it["lang"], it["pid"], it["split"]) == (
                    gen.key, gen.variant, gen.set, gen.lang, gen.pid, gen.split)
                assert it["group"] == {"stock": "stock", "adapter": "adapter"}.get(
                    gen.variant.split("-")[0], "repaired")
                assert row[1] == text == prompt_text(gen.set, gen.lang, gen.pid)
                assert row[2] == gen.output == output_text(gen.variant, gen.lang, gen.pid)
                assert it["shown_prompt_sha"] == sha256_text(store.normalise(row[1]))
                assert it["shown_response_sha"] == sha256_text(store.normalise(row[2]))
        assert sorted(on_sheets) == sorted(items)
    raw = "".join(p.read_text(encoding="utf-8-sig") for p in sheet_paths(drawn))
    assert not [k for k in gens if k[:16] in raw]          # the keys stay in validation/keys
    assert not [pid for (_, _, pid) in prompt_texts(drawn) if pid in raw]   # a pid names its set


def test_no_cell_leaks_the_variant_set_or_judge_label(drawn):
    judged = set()
    # Every file comes from the same fake judges; one variant's files hold every value they use
    # (checked below), and reading all 216 is slow where a virus scanner opens each new file.
    for path in (drawn.outputs / "judgements").glob("*/stock/*.jsonl"):
        for row in read_jsonl(path):
            judged |= {row["judge"], row["judge_label"], row["label"], *strings_in(row["native"])}
    assert {PRIMARY_ID, FRONTIER_ID, *LABELS, "HARMFUL", "Controversial", "Unsafe"} <= judged
    names = {*VARIANT_WORDS, *SAFETY_SETS, *SET_WORDS, *JUDGE_WORDS, *LABELS}
    # Only strings that cannot occur in the placeholder texts make the check meaningful.
    texts = [t.lower() for t in prompt_texts(drawn).values()] + [
        g.output.lower() for g in drawn.by_key().values()]
    words = {w for w in names | judged if w and not any(w.lower() in t for t in texts)}
    assert names <= words
    found = []
    for path in sheet_paths(drawn):
        comment, columns, *rows = read_csv(path)
        found += leaks([path.name, *comment, *columns], (*VARIANT_WORDS, *SET_WORDS, *JUDGE_WORDS))
        found += leaks([cell for row in rows for cell in row], words)
    assert not found


def test_item_ids_are_blind_ids_of_one_shape(drawn):
    seen = Counter()
    for path in sheet_paths(drawn):
        m = re.fullmatch(r"validation_round(\d)_([a-z]{2})(_b)?\.csv", path.name)
        prefix = f"r{m[1]}{m[2]}" + ("b" if m[3] else "")
        for row in body(path):
            assert re.fullmatch(rf"{prefix}-[0-9a-f]{{6}}", row[0]), row[0]
            seen[row[0]] += 1
    assert seen and max(seen.values()) == 1


def stratum_stats(strata: list[tuple[bool, str]]) -> tuple[float, float, float]:
    """For (is stock, set) per row: how much earlier stock rows and harmful rows sit than the
    others (difference of mean positions over the sheet length), and same-stratum neighbours
    relative to their expected number in a random order."""
    n = len(strata)

    def bias(flags: list[bool]) -> float:
        yes = [i for i, f in enumerate(flags) if f]
        no = [i for i, f in enumerate(flags) if not f]
        return (mean(yes) - mean(no)) / n

    expected = sum(c * (c - 1) for c in Counter(strata).values()) / n
    neighbours = sum(a == b for a, b in pairwise(strata))
    return bias([s[0] for s in strata]), bias([s[1] == "harmful" for s in strata]), neighbours / expected


def test_row_order_and_ids_do_not_follow_the_stratum(tmp_path, monkeypatch):
    world = use_world(monkeypatch, small_world(tmp_path / "world"))
    gens = {g.output: g for g in world.by_key().values()}
    stats: dict[str, list] = {"rows": [], "sorted ids": []}
    samples = set()
    seeds = range(1, 5)
    for seed in seeds:
        monkeypatch.setenv("LIFT_VALIDATION_DIR", str(tmp_path / f"validation{seed}"))
        for path in sample.make_sheets(1, seed=seed):
            rows = body(path)
            samples.add((path.name, frozenset(r[2] for r in rows)))
            for order, ranked in (("rows", rows), ("sorted ids", sorted(rows))):
                stats[order].append(stratum_stats([(gens[r[2]].variant == "stock", gens[r[2]].set)
                                                   for r in ranked]))
    assert len(samples) == len(seeds) * 4          # every seed draws a different sample
    # Ordered by stratum, a sheet has a bias of 0.5 and six times the random number of same-stratum
    # neighbours; alternating strata have none.
    for order, values in stats.items():
        group_bias, set_bias, neighbours = zip(*values, strict=True)
        assert max(map(abs, group_bias + set_bias)) < 0.4, order
        assert abs(mean(group_bias)) < 0.1 and abs(mean(set_bias)) < 0.1, order
        assert 0.6 < mean(neighbours) < 1.6, order


def test_hindi_overlap_sheet_repeats_items_under_its_own_ids(drawn):
    sheets = drawn.validation / "sheets"
    main_sheet, overlap = read_csv(sheets / "validation_round1_hi.csv"), read_csv(
        sheets / "validation_round1_hi_b.csv")
    assert overlap[0] == main_sheet[0] and overlap[1] == COLUMNS
    rows, main_rows = overlap[2:], main_sheet[2:]
    assert len(rows) == sample.HINDI_OVERLAP == 20
    assert all(re.fullmatch(r"r1hib-[0-9a-f]{6}", r[0]) and r[3:] == ["", ""] for r in rows)
    assert not {r[0] for r in rows} & {r[0] for r in main_rows}
    assert {(r[1], r[2]) for r in rows} <= {(r[1], r[2]) for r in main_rows}
    items = read_round(drawn.validation, 1)["items"]
    keys = {s: {it["key"] for it in items if it["sheet"] == s} for s in ("round1_hi", "round1_hi_b")}
    assert len(keys["round1_hi_b"]) == 20 and keys["round1_hi_b"] <= keys["round1_hi"]


def long_text(kind: str, tag: str) -> str:
    return "\n".join(f'{kind} {tag} line {i}: with a comma, a "quote", नमस्ते and 你好' for i in range(80))


def test_sheets_show_long_multiline_text_in_full_and_keep_empty_outputs(tmp_path, monkeypatch):
    world = use_world(monkeypatch, small_world(tmp_path))
    rewrite_hindi(world, lambda p: long_text("question", sha256_text(p.pid)[:8]),
                  lambda i, p: "" if i % 4 == 0 else long_text("answer", sha256_text(f"{i}{p.pid}")[:8]))
    sample.make_sheets(1)
    rows = shown(world, 1, "validation_round1_hi.csv")
    assert len(rows) == 60
    for row, _, gen, text in rows:
        assert row[1] == text and row[2] == gen.output
    assert min(len(row[1]) for row, *_ in rows) > 3000 and max(len(row[2]) for row, *_ in rows) > 3000
    assert any(row[2] == "" for row, *_ in rows)          # an empty output is still an item


def test_formula_starts_and_carriage_returns_are_made_safe(tmp_path, monkeypatch):
    world = use_world(monkeypatch, small_world(tmp_path))
    rewrite_hindi(world, lambda p: p.text,
                  lambda i, p: f"{FORMULA_STARTS[i % 4]}{sha256_text(p.pid)[:8]} one\r\ntwo\rthree")
    sample.make_sheets(1)
    raw = (world.validation / "sheets" / "validation_round1_hi.csv").read_bytes()
    assert raw.startswith(BOM) and b"\r" not in raw and raw.endswith(b"\n")
    rows = shown(world, 1, "validation_round1_hi.csv")
    for row, it, gen, text in rows:
        assert row[2] == " " + gen.output.replace("\r\n", "\n").replace("\r", "\n")
        assert row[1] == text
        assert it["shown_response_sha"] == sha256_text(store.normalise(gen.output))
    assert {gen.output[0] for _, _, gen, _ in rows} == set(FORMULA_STARTS)
    for row, _, gen, _ in shown(world, 1, "validation_round1_zh.csv"):
        assert row[2] == gen.output                         # ordinary text is left alone


def test_write_sheet_defuses_formulas_and_writes_bom_and_lf(tmp_path):
    cells = ["=SUM(A1:A3)", "+44 20", "-5", "@here", "plain", "a=b", " =spaced", "a\r\nb", "c\rd",
             'say "hi", twice', "नमस्ते", "你好", ""]
    safe = [" =SUM(A1:A3)", " +44 20", " -5", " @here", "plain", "a=b", " =spaced", "a\nb", "c\nd",
            'say "hi", twice', "नमस्ते", "你好", ""]
    assert [store.sheet_cell(c) for c in cells] == safe
    path = store.write_sheet(tmp_path / "s.csv", "# see validation/guidelines.md", ["item_id", "text"],
                             [[f"id{i}", c] for i, c in enumerate(cells)])
    raw = path.read_bytes()
    assert raw.startswith(BOM) and not raw[3:].startswith(BOM) and b"\r" not in raw
    rows = read_csv(path)
    assert rows[:2] == [["# see validation/guidelines.md"], ["item_id", "text"]]
    assert rows[2:] == [[f"id{i}", c] for i, c in enumerate(safe)]
    # no temporary file left behind (lift-tree is the isolated data/output tree from conftest.py)
    assert [p.name for p in tmp_path.iterdir() if p.name != "lift-tree"] == ["s.csv"]


@pytest.mark.parametrize("n", [None, 20])
def test_utility_sheet_is_blind_and_balanced(tmp_path, monkeypatch, n):
    world = use_world(monkeypatch, build_world(tmp_path, n_dev=1, n_final=0, langs=("hi",),
                                               dev_variants=("stock",), final_variants=(),
                                               judgements=False))
    path = utility.make_utility_sheet(CANDIDATE, n=n)
    n = n or 60                                     # validation.utility_pairs_checked
    assert path == store.sheets_dir() / "validation_utility_hi.csv"
    comment, columns, *rows = read_csv(path)
    assert len(comment) == 1 and comment[0].startswith("#")
    assert columns == UTILITY_COLUMNS and len(rows) == n
    pids = {r["text"]: r["pid"] for r in read_jsonl(world.data / "prompts" / "utility_hi.jsonl")}
    stock = {g.pid: g.output for g in world.generations[("stock", "utility", "hi", "dev")]}
    cand = {g.pid: g.output for g in world.generations[(CANDIDATE, "utility", "hi", "dev")]}
    sides = []
    for item_id, prompt, a, b, preference, note in rows:
        pid = pids[prompt]
        assert re.fullmatch(r"uhi-[0-9a-f]{6}", item_id)
        assert sorted([a, b]) == sorted([stock[pid], cand[pid]]) and a != b
        assert preference == note == ""
        sides.append("A" if a == cand[pid] else "B")
    assert sides.count("A") == sides.count("B") == n // 2
    assert len({r[0] for r in rows}) == n
    if n == 60:
        # Seeded order: the candidate's side runs in neither blocks nor a strict alternation.
        assert n // 5 <= sum(x != y for x, y in pairwise(sides)) <= 4 * n // 5
        assert abs(mean(i for i, s in enumerate(sides) if s == "A")
                   - mean(i for i, s in enumerate(sides) if s == "B")) / n < 0.35
    keys = store.read_json(store.keys_dir() / "utility.json")
    side_of = {it["item_id"]: it["candidate_is"] for it in keys["items"]}
    assert [side_of[r[0]] for r in rows] == sides and keys["candidate_as_a"] == n // 2
    raw = path.read_text(encoding="utf-8-sig")
    assert not leaks([cell for row in read_csv(path) for cell in row],
                     ("stock", "adapter", CANDIDATE, UTILITY_ID, "fakeprovider"))
    assert not [k for k in world.by_key() if k[:16] in raw]
    assert not [pid for pid in pids.values() if pid in raw]


def test_cli_sample_writes_the_sheets_and_prints_no_sheet_text(tmp_path, monkeypatch, capsys):
    world = use_world(monkeypatch, small_world(tmp_path))
    assert main(["validate", "sample", "--round", "1"]) == 0
    out = capsys.readouterr()
    printed = out.out + out.err
    for lang in LANGS:
        path = world.validation / "sheets" / f"validation_round1_{lang}.csv"
        assert path.name in printed
        for row in body(path):
            assert row[1] not in printed and row[2] not in printed
    assert (world.validation / "keys" / "round1.json").is_file()
    assert not [k for k in world.by_key() if k[:16] in printed]


def test_redrawing_never_overwrites_a_sheet_being_filled(tmp_path, monkeypatch):
    world = use_world(monkeypatch, small_world(tmp_path))
    made = {p.name: p for p in sample.make_sheets(1)}
    hi, zh = made["validation_round1_hi.csv"], made["validation_round1_zh.csv"]
    keys = world.validation / "keys" / "round1.json"
    keys_before, zh_before = keys.read_bytes(), zh.read_bytes()
    fill_sheet(hi, hi, {body(hi)[0][0]: "benign"})
    hi_filled = hi.read_bytes()
    zh.unlink()
    sample.make_sheets(1)                         # same draw: only the missing sheet is written
    assert hi.read_bytes() == hi_filled and zh.read_bytes() == zh_before and keys.read_bytes() == keys_before
    with pytest.raises(store.ValidateError):
        sample.make_sheets(1, seed=7)             # a different draw needs --force
    with pytest.raises(store.ValidateError):
        sample.make_sheets(1, seed=7, force=True)  # and never replaces a sheet with entries
    assert hi.read_bytes() == hi_filled and keys.read_bytes() == keys_before
    assert not (world.validation / "sheets" / "archive").exists()


def test_sheets_keys_and_labels_default_to_gitignored_folders(monkeypatch):
    monkeypatch.delenv("LIFT_VALIDATION_DIR", raising=False)
    root = paths.REPO_ROOT / "validation"
    assert (store.sheets_dir(), store.keys_dir(), store.labels_dir()) == (
        root / "sheets", root / "keys", root / "labels")
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    private = ["validation/sheets/validation_round1_hi.csv", "validation/sheets/validation_utility_hi.csv",
               "validation/keys/round1.json", "validation/labels/round1_hi.jsonl"]
    out = subprocess.run([git, "check-ignore", *private, "validation/guidelines.md"],
                         cwd=paths.REPO_ROOT, capture_output=True, text=True)
    if out.returncode == 128:
        pytest.skip("not a git checkout")
    assert out.stdout.split() == private          # and guidelines.md, which readers get, is not
