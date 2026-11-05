import pytest

from lift import config
from lift.selection import DevScores, select_repair

RULES = config.get("selection", name="repairs")
STOCK = DevScores("stock", "stock", harmful_hi=0.10, overrefusal_hard_hi=0.30, win_rate=0.5,
                  belebele_hi=0.60)
ADAPTER = DevScores("adapter-aya-std-s1", "adapter", harmful_hi=0.30, overrefusal_hard_hi=0.25,
                    win_rate=0.70, belebele_hi=0.61)


def cand(name, method, harmful, overref=0.30, win=0.66, bele=0.60):
    return DevScores(name, method, harmful, overref, win, bele)


def test_largest_eligible_drop_wins():
    sel = select_repair(STOCK, ADAPTER, [
        cand("wiseft-l0.50", "wiseft", 0.20),
        cand("cast-m1.0", "cast", 0.12),
    ], RULES)
    assert sel.winner == "cast-m1.0" and sel.verdict == "repair"


def test_constraints_exclude_settings():
    sel = select_repair(STOCK, ADAPTER, [
        cand("cast-m2.0", "cast", 0.05, overref=0.40),    # over-refuses: +10 points vs stock
        cand("resta-a0.40", "resta", 0.08, win=0.55),     # keeps only a quarter of the gain
        cand("wiseft-l0.25", "wiseft", 0.09, bele=0.55),  # Belebele-hi down 5 points
        cand("wiseft-l0.50", "wiseft", 0.18),
    ], RULES)
    assert sel.eligible == ["wiseft-l0.50"]
    assert sel.winner == "wiseft-l0.50"


def test_half_the_gain_boundary():
    # adapter wins 70% -> repaired model must still win at least 60%
    assert select_repair(STOCK, ADAPTER, [cand("a", "resta", 0.1, win=0.60)], RULES).verdict == "repair"
    assert select_repair(STOCK, ADAPTER, [cand("a", "resta", 0.1, win=0.59)], RULES).verdict == "reject"


def test_ties_go_to_the_cheaper_repair():
    # R1-39: the cheaper setting has the lower win rate, so cost must decide before the win rate.
    sel = select_repair(STOCK, ADAPTER, [
        cand("cast-m1.0", "cast", 0.120, win=0.68),
        cand("wiseft-l0.50", "wiseft", 0.135, win=0.62),   # 1.5 points less drop: tied within 2 points
    ], RULES)
    assert sel.winner == "wiseft-l0.50"


def test_equal_cost_ties_go_to_the_higher_win_rate():
    # R1-39: the winner has the higher win rate, the smaller drop and the later name.
    sel = select_repair(STOCK, ADAPTER, [
        cand("wiseft-l0.25", "wiseft", 0.120, win=0.62),
        cand("wiseft-l0.50", "wiseft", 0.135, win=0.68),   # tied within 2 points, same cost
    ], RULES)
    assert sel.eligible == ["wiseft-l0.25", "wiseft-l0.50"]
    assert sel.winner == "wiseft-l0.50"


def test_no_eligible_setting_freezes_best_and_rejects():
    # R1-39: with nothing eligible, decision 7 freezes the fewest violations, then the largest drop; the
    # 2-point tie margin does not apply.
    sel = select_repair(STOCK, ADAPTER, [
        cand("resta-a0.10", "resta", 0.02, overref=0.40, bele=0.55),  # largest drop, but two violations
        cand("cast-m2.0", "cast", 0.05, overref=0.40),                # over-refuses: one violation
        cand("wiseft-l0.50", "wiseft", 0.06, bele=0.55),              # Belebele -5: one; cheaper, 1 point
                                                                      # less drop (within the tie margin)
        cand("wiseft-l0.75", "wiseft", 0.31),     # makes things worse: violates drop_ok only
    ], RULES)
    assert sel.verdict == "reject"
    assert sel.winner == "cast-m2.0"
    assert sel.eligible == []


def test_rejects_non_repair_methods():
    with pytest.raises(ValueError):
        select_repair(STOCK, ADAPTER, [cand("adapter", "adapter", 0.3)], RULES)
