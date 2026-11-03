import pytest

from lift import config
from lift.splits import assign_splits

SEED = config.get("split_seed")
SIZES = config.get("sizes.harmful")
PIDS = [f"prh-test-{i:04d}" for i in range(572)]
# PolyRefuse's harmful test repeats 10 English prompts (JailbreakBench rows reappearing later).
PAIRS = [(4, 174), (14, 163), (24, 243), (39, 402), (51, 193), (53, 257), (54, 187), (60, 145),
         (61, 470), (63, 491)]
GROUPS = {f"prh-test-{b:04d}": f"prh-test-{a:04d}" for a, b in PAIRS}


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
