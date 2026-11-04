"""Grouped splits: lift.splits.assign_splits with groups, and how lift.data.build links prompts, groups
the linked test prompts and excludes the train/val prompts linked to one (in-memory records, no files).
"""

from fractions import Fraction

import pytest

from lift import config
from lift.data import build
from lift.schemas import PromptRecord
from lift.splits import assign_splits

SEED = config.get("split_seed")
SIZES = config.get("sizes.harmful")
PIDS = [f"prh-test-{i:04d}" for i in range(572)]
# PolyRefuse's harmful test repeats 10 English prompts (JailbreakBench rows reappearing later).
PAIRS = [(4, 174), (14, 163), (24, 243), (39, 402), (51, 193), (53, 257), (54, 187), (60, 145),
         (61, 470), (63, 491)]
GROUPS = {f"prh-test-{b:04d}": f"prh-test-{a:04d}" for a, b in PAIRS}
NEAR = build._split_near(config.load("project"))


def test_singleton_groups_reproduce_the_plain_assignment():
    plain = assign_splits(PIDS, SIZES, SEED)
    assert assign_splits(PIDS, SIZES, SEED, groups=None) == plain
    assert assign_splits(PIDS, SIZES, SEED, groups={}) == plain      # a set with no duplicates


def test_groups_never_straddle_and_sizes_stay_exact():
    plain = assign_splits(PIDS, SIZES, SEED)
    assert any(plain[f"prh-test-{a:04d}"] != plain[f"prh-test-{b:04d}"] for a, b in PAIRS)
    grouped = assign_splits(PIDS, SIZES, SEED, groups=GROUPS)
    for a, b in PAIRS:
        assert grouped[f"prh-test-{a:04d}"] == grouped[f"prh-test-{b:04d}"]
    assert sum(v == "dev" for v in grouped.values()) == SIZES["dev"]
    assert sum(v == "final" for v in grouped.values()) == SIZES["final"]
    assert grouped == assign_splits(list(reversed(PIDS)), SIZES, SEED, groups=GROUPS)


def test_groups_that_cannot_fill_a_split_exactly_raise():
    with pytest.raises(ValueError, match="exactly"):
        assign_splits(["a", "b", "c"], {"dev": 1, "final": 2}, seed=1,
                      groups={"a": "g", "b": "g", "c": "g"})


# ---------------------------------------------------------------- links, groups and exclusions


def block(i: int, n: int = 22) -> str:
    """``n`` distinct Han characters, none shared with another block: n - 2 distinct 3-grams, so a
    prefix of m characters has Jaccard (m - 2) / (n - 2) with the whole (16 of 22: 0.7 exactly)."""
    return "".join(chr(0x4E00 + 64 * i + k) for k in range(n))


def rec(pid: str, english: str, text: str | None = None, lang: str = "en") -> PromptRecord:
    """A PolyRefuse-like record: split "dev" (the placeholder) for test pids, else train or val."""
    split = "dev" if "-test-" in pid else pid.split("-")[1]
    return PromptRecord(pid=pid, lang=lang, set="harmful" if pid.startswith("prh") else "easy_benign",
                        split=split, text=english if text is None else text, text_en=english,
                        source="polyrefuse")


def test_near_config_is_the_heldout_instruction_rule():
    nd = config.load("project")["sources"]["sft"]["heldout"]["near_dup"]
    assert NEAR == (nd["ngram"], Fraction(str(nd["instruction"]))) == (3, Fraction(7, 10))


def test_links_take_equal_and_near_duplicate_english_at_the_threshold():
    records = [rec("prh-test-0000", "Placeholder request about kites"),
               rec("prh-test-0001", "  placeholder REQUEST\nabout kites "),      # equal by dup_key
               rec("prh-test-0002", block(1)), rec("prh-test-0003", block(1)[:16]),   # Jaccard 0.7
               rec("prh-test-0004", block(2)), rec("prh-test-0005", block(2)[:15])]   # Jaccard 0.65
    links = build._links(records, {}, NEAR)
    assert links == {("prh-test-0000", "prh-test-0001"): ("en",), ("prh-test-0002", "prh-test-0003"): ("en",)}
    by_pid = {r.pid: r for r in records}
    assert build._evidence(by_pid["prh-test-0001"], by_pid["prh-test-0000"], ("en",), NEAR) == "en same text"
    assert build._evidence(by_pid["prh-test-0003"], by_pid["prh-test-0002"], ("en",), NEAR) == "en 0.700"
    assert build._evidence(by_pid["prh-test-0005"], by_pid["prh-test-0004"], (), NEAR) == "en 0.650"
    looser = build._Near(3, Fraction(13, 20))
    assert ("prh-test-0004", "prh-test-0005") in build._links(records, {}, looser)


def test_links_take_texts_a_translation_makes_equal():
    pids = ("prh-test-0000", "prh-test-0001", "prh-train-0000")
    records = [rec(pid, block(i)) for i, pid in enumerate(pids)]
    zh_texts = ("占位翻译甲", " 占位翻译甲 ", "占位翻译乙")              # equal by dup_key: the first two
    zh = [rec(pid, block(i), zh_texts[i], "zh") for i, pid in enumerate(pids)]
    th = [rec(pid, block(i), "ข้อความตัวอย่าง", "th") for i, pid in enumerate(pids) if i]
    links = build._links(records, {"zh": zh, "th": th}, NEAR)
    assert links == {(pids[0], pids[1]): ("zh",), (pids[1], pids[2]): ("th",)}
    evidence = build._evidence(records[2], records[1], links[(pids[1], pids[2])], NEAR)
    assert evidence == "en 0.000; th same text"


def test_groups_are_the_connected_components_among_the_split_pids():
    links = {("prh-test-0003", "prh-test-0007"): ("en",), ("prh-test-0007", "prh-test-0009"): ("th",),
             ("prh-test-0002", "prh-test-0009"): ("zh",),                   # joins 3-7-9 through 9
             ("prh-test-0005", "prh-test-0006"): ("en", "zh"),
             ("prh-test-0001", "prh-train-0004"): ("en",), ("prh-test-0008", "prh-train-0004"): ("en",),
             ("prh-test-0000", "prb-test-0000"): ("en",)}                   # another set's split file
    pids = [f"prh-test-{i:04d}" for i in range(10)]
    chain = ("prh-test-0002", "prh-test-0003", "prh-test-0007", "prh-test-0009")
    groups = build._groups(pids, links)
    assert groups == {**dict.fromkeys(chain, "prh-test-0002"),
                      "prh-test-0005": "prh-test-0005", "prh-test-0006": "prh-test-0005"}
    assert build._groups(pids, list(reversed(list(links)))) == groups        # the order of the links
    assert build._groups(pids, {}) == {}
    grouped = assign_splits(pids, {"dev": 4, "final": 6}, SEED, groups)
    assert len({grouped[p] for p in chain}) == 1 and grouped["prh-test-0005"] == grouped["prh-test-0006"]


def test_exclusions_take_only_train_and_val_prompts_linked_directly_to_a_test_prompt():
    a, b, c = block(0), block(1), block(2)
    english = [rec("prh-test-0000", a), rec("prh-test-0001", b), rec("prh-test-0002", c),
               rec("prh-test-0003", c[:17]),                       # 0.75 to test 2: the two are linked
               rec("prb-test-0000", block(3)),
               rec("prh-train-0000", f" {a}"),                     # test 0 again
               rec("prh-train-0001", b[:16]),                      # 0.7 to test 1
               rec("prh-train-0002", b[:15]),                      # 0.65 to test 1, 0.93 to train 1 only
               rec("prh-train-0003", c[:16]),                      # 0.7 to test 2, 0.93 to test 3
               rec("prh-train-0004", c[:17]),                      # test 3 again, 0.75 to test 2
               rec("prh-val-0000", block(4)),                      # its th text is test 1's
               rec("prb-train-0000", block(5))]                    # its zh text is prh-test-0000's
    same_th, same_zh = "ข้อความเดียวกัน", "同一段占位文字"
    th = [rec("prh-test-0001", b, same_th, "th"), rec("prh-val-0000", block(4), same_th, "th")]
    zh = [rec("prh-test-0000", a, same_zh, "zh"), rec("prb-train-0000", block(5), same_zh, "zh")]
    links = build._links(english, {"zh": zh, "th": th}, NEAR)
    assert ("prh-train-0001", "prh-train-0002") in links and ("prh-test-0002", "prh-test-0003") in links
    excluded = {e["pid"]: (e["reason"], e["detail"]) for e in build._test_links(english, links, NEAR)}
    assert excluded == {
        "prh-train-0000": ("duplicate_of:prh-test-0000", "en same text"),
        "prh-train-0001": ("near_duplicate_of:prh-test-0001", "en 0.700"),
        "prh-train-0003": ("near_duplicate_of:prh-test-0002", "en 0.700; also prh-test-0003"),
        # equal English wins over a smaller near pid
        "prh-train-0004": ("duplicate_of:prh-test-0003", "en same text; also prh-test-0002"),
        "prh-val-0000": ("near_duplicate_of:prh-test-0001", "en 0.000; th same text"),
        "prb-train-0000": ("near_duplicate_of:prh-test-0000", "en 0.000; zh same text"),     # across kinds
    }
    assert "prh-train-0002" not in excluded               # linked to a train prompt only: it stays
    groups = build._groups([r.pid for r in english if r.pid.startswith("prh-test-")], links)
    assert groups == {"prh-test-0002": "prh-test-0002", "prh-test-0003": "prh-test-0002"}


def test_belebele_questions_on_one_passage_are_one_group():
    items = [{"pid": f"bel-{i:03d}", "link": f"https://example.org/passage/{i // 2}"} for i in range(5)]
    items.append({"pid": "bel-005", "link": "https://example.org/passage/0"})
    pids, groups = build._split_pids(build._Sources(belebele_en=items), "belebele")
    assert pids == [f"bel-{i:03d}" for i in range(6)]
    assert groups == {"bel-000": "bel-000", "bel-001": "bel-000", "bel-005": "bel-000",   # bel-004: alone
                      "bel-002": "bel-002", "bel-003": "bel-002"}
