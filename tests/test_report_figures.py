"""The figures of the final report (lift.report.figures): the names they give the variants, the ink of the
heatmaps' numbers and the rows of the forest plot. The statistics are synthetic, in the shapes ``lift
stats`` and ``lift select`` write; nothing is read from outputs/. No torch."""

from __future__ import annotations

import hashlib
import re

import pytest

from lift.report import figures
from lift.schemas import LANGS

pytest.importorskip("matplotlib")  # the report extra: without it lift report draws nothing

LAMBDA, ALPHA = chr(0x3BB), chr(0x3B1)  # the Greek letters, spelled here apart from the module under test
STRESS = "adapter-indicalign-stress-s1"
AYA = [f"adapter-aya-std-s{s}" for s in (1, 2, 3)]
IND = [f"adapter-indicalign-std-s{s}" for s in (1, 2, 3)]
WISEFT, RESTA, CAST = (f"{tag}-{STRESS}" for tag in ("wiseft-l0.25", "resta-a0.10", "cast-m0.5"))
NO_ROLES = {"adapter": {}, "winner": {}, "families": {}, "standard": {}, "candidates": {}}
# Each variant's Hindi harmful rate; another language's is a share of it (SHARE).
HINDI = {"stock": 0.10, AYA[0]: 0.04, AYA[1]: 0.05, AYA[2]: 0.06, **dict.fromkeys(IND, 0.30)}
HINDI |= {STRESS: 0.50, WISEFT: 0.06, RESTA: 0.35, CAST: 0.25}
SHARE = {"hi": 1.0, "en": 0.5, "zh": 0.7, "es": 0.6, "ar": 0.7, "th": 0.6}
PLAIN_ROWS = [
    "stock",
    "Aya adapter (standard)",
    "IndicAlign adapter (standard)",
    "stress adapter",
    f"WiSE-FT, {LAMBDA} = 0.25",
    f"ReSta, {ALPHA} = 0.10",
    "CAST, strength 0.5",
]


def roles() -> dict:
    """final_stats.json's roles as the study stands: the stress adapter carries the repair comparison,
    Gate 1 picked aya and did not pick indicalign. The seeds are strings, as JSON has them."""
    return {
        "adapter": {"1": STRESS},
        "winner": {"1": WISEFT},
        "families": {"wiseft": {"1": WISEFT}, "resta": {"1": RESTA}, "cast": {"1": CAST}},
        "standard": {str(s): name for s, name in enumerate(AYA, 1)},
        "candidates": {IND[0]: {str(s): name for s, name in enumerate(IND, 1)}},
    }


def cells(hindi: dict[str, float]) -> list[dict]:
    """final_stats.json's cells of the harmful set: one per variant and language, each of 400 prompts."""
    return [
        {
            "variant": variant,
            "set": "harmful",
            "lang": lang,
            "status": "ok",
            "n": 400,
            "harmful_rate": rate * SHARE[lang],
        }
        for variant, rate in hindi.items()
        for lang in LANGS
    ]


def contrast(a, b) -> float:
    """WCAG 2's contrast ratio of two sRGB colours, computed here apart from the module under test."""
    from matplotlib.colors import to_rgb

    def luminance(colour) -> float:
        r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    light, dark = sorted((luminance(a), luminance(b)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


@pytest.fixture
def drawn(monkeypatch):
    """Figures stay open after they are saved, so a test can read what was drawn: ``drawn.gcf()``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    yield plt
    monkeypatch.undo()
    plt.close("all")


def heatmap(plt, tmp_path, roles_, hindi=None):
    """Draw the harmful-rate heatmap: (its axes, {(column, row): the text written in that cell})."""
    stats = {"roles": roles_, "cells": cells(HINDI if hindi is None else hindi)}
    figures.heatmap(stats, "harmful", "harmful_rate", "a title", tmp_path / "heatmap.png")
    ax = plt.gcf().axes[0]
    return ax, {(round(t.get_position()[0]), round(t.get_position()[1])): t for t in ax.texts}


def tick_names(axis) -> list[str]:
    return [t.get_text() for t in axis.get_ticklabels()]


# --------------------------------------------------------------------------------------------- names


def test_a_variant_is_named_as_the_write_up_names_it_from_its_id():
    assert figures.plain_name("stock") == "stock"
    assert [figures.plain_name(v) for v in AYA] == ["Aya adapter (standard)"] * 3  # the seed is no part
    assert figures.plain_name(IND[0]) == "IndicAlign adapter (standard)"
    assert figures.plain_name(STRESS) == "stress adapter"
    # The stress recipe trains indicalign (configs/train.yaml): a stress adapter of another candidate
    # is named with it, so no two adapters share a name.
    assert figures.plain_name("adapter-aya-stress-s1") == "Aya stress adapter"
    assert figures.plain_name(WISEFT) == f"WiSE-FT, {LAMBDA} = 0.25"
    assert figures.plain_name(RESTA) == f"ReSta, {ALPHA} = 0.10"  # the digits the id has
    assert figures.plain_name(CAST) == "CAST, strength 0.5"
    assert figures.plain_name(f"castalways-m1.0-{STRESS}") == "CAST always on, strength 1.0"
    # A candidate the table does not list keeps its key; an id outside the naming scheme stays as it is.
    assert figures.plain_name("adapter-newset-std-s1") == "newset adapter (standard)"
    for unknown in ("lora-v2", "adapter-aya-fast-s1", "adapter-aya-std", "prune-p0.5-adapter-aya-std-s1"):
        assert figures.plain_name(unknown) == unknown


def test_the_figures_and_the_write_up_name_a_variant_with_the_same_words():
    """The write-up's naming key (lift.publish.article.plain, decisions item 123) and the figures name a
    variant by two rules, one for each form: "the Aya adapter" in a sentence, "Aya adapter (standard)" on a
    row. Both must say the same candidate, method, Greek letter and digits."""
    from lift.publish import article

    assert figures.CANDIDATE_NAMES == article.CANDIDATE_NAMES
    assert (figures.LAMBDA, figures.ALPHA) == (article.LAMBDA, article.ALPHA) == (LAMBDA, ALPHA)
    for variant in (AYA[0], IND[1], STRESS, "adapter-aya-stress-s1", WISEFT, RESTA, CAST):
        drawn, said = figures.plain_name(variant), article.plain(None, variant)
        words = set(re.findall(r"[^\s,()=]+", drawn)) - {"standard"}  # the recipe: the key says it apart
        assert words <= set(re.findall(r"[^\s,()=]+", said)), (variant, drawn, said)


def test_a_setting_on_a_trade_off_point_has_its_familys_name_and_its_strength():
    assert figures.short_name(WISEFT) == "WiSE-FT 0.25"
    assert figures.short_name(RESTA) == "ReSta 0.10"
    assert figures.short_name(CAST) == "CAST 0.5"
    assert figures.short_name(f"castalways-m1.0-{STRESS}") == "CAST always on 1.0"
    assert figures.short_name("prune-p0.5-adapter-aya-std-s1") == "prune-p0.5"  # its tag, as before
    assert figures.short_name("stock") == "stock"


# ------------------------------------------------------------------------------------------ heatmaps


def test_the_heatmap_names_its_rows_plainly_with_the_repaired_adapter_above_its_repairs(drawn, tmp_path):
    ax, texts = heatmap(drawn, tmp_path, roles())
    assert tick_names(ax.yaxis) == PLAIN_ROWS
    assert tick_names(ax.xaxis) == ["Hindi", "English", "Chinese", "Spanish", "Arabic", "Thai"]
    assert ax.get_xlabel() == "The last 3 rows are the stress adapter with each frozen repair applied."
    # Each row holds its own variants: Hindi first, and the standard adapter pooled over its three seeds.
    hindi = [texts[(0, row)].get_text() for row in range(len(PLAIN_ROWS))]
    assert hindi == ["10.0", "5.0", "30.0", "50.0", "6.0", "35.0", "25.0"]
    stress = [texts[(col, 3)].get_text() for col in range(6)]
    assert stress == ["50.0", "25.0", "35.0", "30.0", "35.0", "30.0"]


def test_the_heatmap_of_a_standard_adapter_that_carries_the_repairs_itself(drawn, tmp_path):
    """The pre-registered case: the standard recipe drifts, so its three seeds are repaired. No other
    adapter's row comes between stock and it."""
    repaired = {
        family: {s: f"{tag}-{name}" for s, name in enumerate(AYA, 1)}
        for family, tag in (("wiseft", "wiseft-l0.50"), ("resta", "resta-a0.25"), ("cast", "cast-m1.0"))
    }
    standard = {**NO_ROLES, "adapter": dict(enumerate(AYA, 1)), "families": repaired}  # seeds as ints
    hindi = {"stock": 0.1, **dict.fromkeys(AYA, 0.4)}
    hindi |= {name: 0.2 for by_seed in repaired.values() for name in by_seed.values()}
    ax, texts = heatmap(drawn, tmp_path, standard, hindi)
    assert tick_names(ax.yaxis) == [
        "stock",
        "Aya adapter (standard)",
        f"WiSE-FT, {LAMBDA} = 0.50",
        f"ReSta, {ALPHA} = 0.25",
        "CAST, strength 1.0",
    ]
    note = "The last 3 rows are the Aya adapter (standard) with each frozen repair applied."
    assert ax.get_xlabel() == note
    assert [texts[(0, row)].get_text() for row in range(5)] == ["10.0", "40.0", "20.0", "20.0", "20.0"]


def test_the_heatmaps_numbers_are_white_on_dark_cells_and_near_black_on_light_ones(drawn, tmp_path):
    """Review F19: the darkest cell's number was black on near-black red."""
    ax, texts = heatmap(drawn, tmp_path, roles())
    assert texts[(0, 3)].get_text() == "50.0" and texts[(0, 3)].get_color() == figures.WHITE  # the darkest
    assert texts[(1, 1)].get_text() == "2.5" and texts[(1, 1)].get_color() == figures.INK  # the lightest
    image = ax.images[0]
    for text in texts.values():
        cell = image.cmap(image.norm(float(text.get_text())))
        assert contrast(text.get_color(), cell) > 3, text.get_text()


def test_the_ink_reads_on_every_step_of_the_ramp():
    from matplotlib import colormaps

    ramp = [colormaps["Reds"](i / 255) for i in range(256)]
    inks = [figures._ink_on(colour) for colour in ramp]
    assert inks[0] == figures.INK and inks[-1] == figures.WHITE
    assert sum(a != b for a, b in zip(inks, inks[1:], strict=False)) == 1  # one change of ink, down the ramp
    assert min(contrast(ink, colour) for ink, colour in zip(inks, ramp, strict=True)) > 3
    # Black, as every number was written before, does not read on the darkest step; white does.
    assert contrast("black", ramp[-1]) < 1.6 < 4.5 < contrast(inks[-1], ramp[-1])


def test_a_cell_that_was_not_run_says_so_in_near_black(drawn, tmp_path):
    hindi = {v: rate for v, rate in HINDI.items() if v != AYA[2]}  # one seed of the standard adapter
    ax, texts = heatmap(drawn, tmp_path, roles(), hindi)
    assert tick_names(ax.yaxis) == PLAIN_ROWS
    for col in range(6):  # a row is drawn only when every seed was run
        assert texts[(col, 1)].get_text() == "not run" and texts[(col, 1)].get_color() == figures.INK
    assert texts[(0, 0)].get_text() == "10.0"


def test_a_row_the_naming_rule_does_not_know_keeps_its_id(drawn, tmp_path):
    # Ids outside the naming scheme: the row keeps its first id.
    other = {**NO_ROLES, "adapter": {"1": "lora-x-s1", "2": "lora-x-s2"}}
    ax, texts = heatmap(drawn, tmp_path, other, {"stock": 0.1, "lora-x-s1": 0.2, "lora-x-s2": 0.4})
    assert tick_names(ax.yaxis) == ["stock", "lora-x-s1"] and texts[(0, 1)].get_text() == "30.0"
    assert ax.get_xlabel() == ""
    # Two rows the rule would give one name: each keeps its id, so no two rows read alike.
    twice = {**NO_ROLES, "adapter": {"1": AYA[0]}, "standard": {"1": AYA[1]}}
    ax, _texts = heatmap(drawn, tmp_path, twice)
    assert tick_names(ax.yaxis) == ["stock", AYA[1], AYA[0]]
    # No selection yet: the adapter's row is there, empty, under its role.
    ax, texts = heatmap(drawn, tmp_path, dict(NO_ROLES))
    assert tick_names(ax.yaxis) == ["stock", "adapter"]
    assert {texts[(col, 1)].get_text() for col in range(6)} == {"not run"}


def test_the_line_under_a_heatmap_says_only_what_the_ids_say():
    """Which rows are repairs, and of which adapter, is read from the repairs' own ids."""
    one = [("wiseft", [WISEFT])]
    note = figures._repairs_note(roles(), one, "stress adapter")
    assert note == "The last row is the stress adapter with the frozen repair applied."
    # A standard adapter's three seeds, each with its repair.
    standard = {**NO_ROLES, "adapter": {s: name for s, name in enumerate(AYA, 1)}}
    repaired = [("wiseft", [f"wiseft-l0.50-{name}" for name in AYA]), ("cast", [f"cast-m1.0-{AYA[0]}"])]
    note = figures._repairs_note(standard, repaired, "Aya adapter (standard)")
    assert note == "The last 2 rows are the Aya adapter (standard) with each frozen repair applied."
    # An adapter named by its id takes no article.
    fast = "adapter-aya-fast-s1"
    odd = {**NO_ROLES, "adapter": {"1": fast}}
    note = figures._repairs_note(odd, [("wiseft", [f"wiseft-l0.50-{fast}"])], fast)
    assert note == "The last row is adapter-aya-fast-s1 with the frozen repair applied."
    # Nothing is said when there is no repair, when a repair's id names another adapter or a seed the
    # adapter's row does not hold, or when it is no repair setting's id.
    assert figures._repairs_note(roles(), [], "stress adapter") == ""
    assert figures._repairs_note(roles(), [("wiseft", [f"wiseft-l0.25-{AYA[0]}"])], "stress adapter") == ""
    two_seeds = [("wiseft", [WISEFT, WISEFT.replace("-s1", "-s2")])]
    assert figures._repairs_note(roles(), two_seeds, "stress adapter") == ""
    assert figures._repairs_note(roles(), [("wiseft", ["some-repair"])], "stress adapter") == ""
    assert figures._repairs_note(dict(NO_ROLES), one, "adapter") == ""


# -------------------------------------------------------------------------------------------- forest


def effect(seeds: dict[int, tuple[float, float, float]], pooled: tuple[float, float, float] | None) -> dict:
    """An effect of final_stats.json's primary block: {seed: (est, lo, hi)} and the pooled (est, lo, hi)."""
    keys = ("est", "lo", "hi")
    return {
        "per_seed": [{"seed": s, **dict(zip(keys, v, strict=True))} for s, v in seeds.items()],
        "pooled": None if pooled is None else {**dict(zip(keys, pooled, strict=True)), "seeds": list(seeds)},
    }


def forest_rows(plt) -> tuple[list[str], list[float]]:
    """(the rows' names from the top down, the estimate drawn on each) of the forest plot just drawn."""
    ax = plt.gcf().axes[0]
    by_height = {t.get_position()[1]: t.get_text() for t in ax.yaxis.get_ticklabels()}
    line = ax.containers[0][0]
    estimates = dict(zip(line.get_ydata(), line.get_xdata(), strict=True))
    heights = sorted(by_height, reverse=True)
    return [by_height[y] for y in heights], [float(estimates[y]) for y in heights]


def test_the_forest_plot_has_one_row_per_effect_when_one_seed_was_run(drawn, tmp_path):
    """Review F19: with one seed the pooled estimate is that seed's own, and was drawn twice."""
    drift, repair = (0.315, 0.2625, 0.37), (-0.3925, -0.445, -0.3425)
    stats = {"primary": {"drift": effect({1: drift}, drift), "repair": effect({1: repair}, repair)}}
    assert figures.forest(stats, tmp_path / "forest.png") == tmp_path / "forest.png"
    names, estimates = forest_rows(drawn)
    assert names == ["drift seed 1", "repair seed 1"]
    assert estimates == pytest.approx([31.5, -39.25])
    assert drawn.gcf().get_size_inches()[1] == pytest.approx(0.45 * 2 + 1.5)


def test_the_forest_plot_keeps_each_seed_and_the_pooled_row_when_more_seeds_were_run(drawn, tmp_path):
    seeds = {1: (0.30, 0.20, 0.40), 2: (0.32, 0.22, 0.42), 3: (0.28, 0.18, 0.38)}
    stats = {
        "primary": {
            "drift": effect(seeds, (0.30, 0.25, 0.35)),
            # one seed of three run: its row alone, since the pooled estimate is that seed's
            "repair": effect({2: (-0.25, -0.35, -0.15)}, (-0.25, -0.35, -0.15)),
        }
    }
    figures.forest(stats, tmp_path / "forest.png")
    names, estimates = forest_rows(drawn)
    assert names == ["drift seed 1", "drift seed 2", "drift seed 3", "drift pooled", "repair seed 2"]
    assert estimates == pytest.approx([30.0, 32.0, 28.0, 30.0, -25.0])


def test_a_pooled_row_that_says_something_else_than_its_only_seed_is_kept(drawn, tmp_path):
    """The pooled row is left out only when it repeats the seed's estimate and interval."""
    one = {1: (0.30, 0.20, 0.40)}
    stats = {"primary": {"drift": effect(one, (0.30, 0.21, 0.40)), "repair": effect({}, None)}}
    figures.forest(stats, tmp_path / "forest.png")
    assert forest_rows(drawn)[0] == ["drift seed 1", "drift pooled"]  # and no row for the arm not run


def test_no_forest_plot_is_drawn_from_nothing(tmp_path):
    stats = {"primary": {"drift": effect({}, None), "repair": effect({}, None)}}
    assert figures.forest(stats, tmp_path / "none.png") is None and not (tmp_path / "none.png").exists()


# ------------------------------------------------------------------------- adapters, trade-off legends


def drift_rows(adapters: dict[tuple[str, str], float]) -> list[dict]:
    """final_stats.json's adapter_drift rows on the harmful set: {(kind, seed-1 id): the Hindi rate}."""
    return [
        {
            "set": "harmful",
            "kind": kind,
            "adapter": name,
            "lang": lang,
            "rate": rate * SHARE[lang],
            "stock": 0.1,
        }
        for (kind, name), rate in adapters.items()
        for lang in LANGS
    ]


def test_the_adapters_figure_names_each_adapter_plainly_with_its_role(drawn, tmp_path):
    adapters = {("adapter", STRESS): 0.5, ("standard", AYA[0]): 0.05, ("candidate", IND[0]): 0.3}
    figures.adapters_by_language({"adapter_drift": drift_rows(adapters)}, tmp_path / "adapters.png")
    assert [t.get_text() for t in drawn.gcf().axes[0].get_legend().get_texts()] == [
        "stock",
        "Aya adapter (standard): the pre-registered pick",
        "IndicAlign adapter (standard): the data set not picked (exploratory)",
        "stress adapter: carries the repair comparison",
    ]
    # An id outside the naming scheme is written as it is.
    other = {"adapter_drift": drift_rows({("adapter", "lora-x"): 0.5})}
    figures.adapters_by_language(other, tmp_path / "other.png")
    legend = drawn.gcf().axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["stock", "lora-x: carries the repair comparison"]


def dev_table() -> list[dict]:
    """A selection table (selection.json, selection_secondary.json) of five dev settings: one per method,
    and one whose id is no repair setting's."""
    settings = [
        (WISEFT, "wiseft", 0.06, 0.50, 1.8),
        (RESTA, "resta", 0.35, 0.27, 0.4),
        (CAST, "cast", 0.31, 0.28, 1.0),
        (f"castalways-m1.0-{STRESS}", "cast", 0.02, 0.04, -13.0),
        (f"prune-p0.5-{STRESS}", "prune", 0.20, 0.40, 0.7),
    ]
    return [
        {
            "name": name,
            "family": family,
            "harmful_hi": harmful,
            "win_rate": win,
            "min_win_rate": 0.26,
            "overrefusal_increase": 0.01 * i,
            "belebele_drop": 0.02 * i,
            "heldout_retained": kept,
            "min_heldout_retained": 0.5,
            "eligible": i == 0,
        }
        for i, (name, family, harmful, win, kept) in enumerate(settings)
    ]


def dev_selection() -> dict:
    return {
        "status": "decided",
        "table": dev_table(),
        "stock": {"harmful_hi": 0.14},
        "adapter_scores": {"harmful_hi": 0.49},
        "rules": {"max_overrefusal_increase": 0.05, "max_belebele_hi_drop": 0.03},
    }


def test_the_trade_off_figures_name_the_settings_and_the_families_as_the_write_up_does(drawn, tmp_path):
    """Review F19: tags such as castalways-m1.0 were never explained."""
    families = [f"WiSE-FT ({LAMBDA})", f"ReSta ({ALPHA})", "CAST (strength)", "prune"]  # unknown: as it is
    points = {"WiSE-FT 0.25", "ReSta 0.10", "CAST 0.5", "CAST always on 1.0", "prune-p0.5"}
    figures.tradeoff(dev_selection(), tmp_path / "tradeoff.png")
    fig = drawn.gcf()
    assert [t.get_text() for t in fig.legends[0].get_texts()] == families
    for ax in fig.axes[:2]:
        assert points <= {t.get_text() for t in ax.texts}
    figures.tradeoff_heldout(dev_selection(), tmp_path / "heldout.png")
    ax = drawn.gcf().axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == families
    # The setting far under stock's loss is at the edge, with its share after its name.
    names = {t.get_text() for t in ax.texts}
    assert points - {"CAST always on 1.0"} <= names and "CAST always on 1.0 (-1300%)" in names


# --------------------------------------------------------------------------------------- determinism


def test_a_figure_drawn_twice_is_the_same_file(tmp_path):
    """The write-up records each figure by the sha256 of its PNG (hf/article.sources.json), and
    scripts/verify_numbers.py compares it: drawing again from the same numbers must give the same bytes."""
    drift, repair = (0.315, 0.2625, 0.37), (-0.3925, -0.445, -0.3425)
    adapters = {("adapter", STRESS): 0.5, ("standard", AYA[0]): 0.05, ("candidate", IND[0]): 0.3}
    stats = {
        "roles": roles(),
        "cells": cells(HINDI),
        "primary": {"drift": effect({1: drift}, drift), "repair": effect({1: repair}, repair)},
        "adapter_drift": drift_rows(adapters),
    }
    draws = {
        "heatmap": lambda path: figures.heatmap(stats, "harmful", "harmful_rate", "a title", path),
        "adapters": lambda path: figures.adapters_by_language(stats, path),
        "tradeoff": lambda path: figures.tradeoff(dev_selection(), path),
        "heldout": lambda path: figures.tradeoff_heldout(dev_selection(), path),
        "forest": lambda path: figures.forest(stats, path),
    }
    for name, draw in draws.items():
        first, second = tmp_path / f"{name}_1.png", tmp_path / f"{name}_2.png"
        assert draw(first) == first and draw(second) == second
        assert first.stat().st_size > 5_000
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (first, second)}
        assert len(digests) == 1, name
