"""lift.publish.inputs and render: formats, placeholders, provenance, templates without typed numbers."""

from __future__ import annotations

import json
import re

import pytest
from publish_fakes import no_committed_recipe, write_inputs, write_json, write_report  # noqa: F401

from lift import paths
from lift.publish import PublishError, render
from lift.publish.inputs import NOT_RUN, PENDING, Facts, fmt, report_path


@pytest.mark.parametrize(
    "value,kind,shown",
    [
        (0.1234, "pct", "12.3%"),
        (0.041, "pp", "+4.1 pp"),
        (-0.14, "pp", "-14.0 pp"),
        ({"est": 0.2, "lo": 0.1, "hi": 0.3}, "pct_ci", "20.0% [10.0, 30.0]"),
        ([-0.1, -0.15, -0.05], "pp_ci", "-10.0 pp [-15.0, -5.0]"),
        (0.0004, "p", "<0.001"),
        (0.0312, "p", "0.031"),
        (1234, "int", "1,234"),
        (0.05, "points", "5 points"),
        (1.5, "hours", "1.50 h"),
        (None, "pct", NOT_RUN),
        ("WiSE-FT", "text", "WiSE-FT"),
        # decisions.md 77: a value computed but undefined is P07's text, shown as it is in any format
        (
            "undefined: the adapter shows no utility gain over parity",
            "pct_ci",
            "undefined: the adapter shows no utility gain over parity",
        ),
        ("undefined", "f2", "undefined"),
        ("undefined", "p_stmt", "undefined"),
    ],
)
def test_formats(value, kind, shown):
    assert fmt(value, kind) == shown


@pytest.mark.parametrize(
    "value,kind",
    [
        ("x", "pct"),
        ("not undefined", "pct_ci"),
        (float("nan"), "pct"),
        (True, "pct"),
        (1.5, "int"),
        ({"est": 1}, "pct_ci"),
        (1, "nope"),
    ],
)
def test_bad_values_are_refused(value, kind):
    with pytest.raises((ValueError, KeyError)):
        fmt(value, kind)


def test_missing_report_renders_pending_and_records_it():
    facts = Facts()
    out = facts.report_value("/primary/drift/pooled", "pp_ci", what="the drift")
    assert out.startswith(PENDING) and "outputs/metrics/final_report.json" in out and "P07" in out
    assert facts.missing and not facts.sources
    assert facts.table("per_language").startswith("_" + PENDING)


def test_values_and_tables_record_their_source(tmp_path):
    write_inputs()
    facts = Facts()
    assert facts.report_value("/primary/drift/pooled", "pp_ci", what="d") == "+16.0 pp [+11.0, +21.0]"
    table = facts.table("utility")
    assert "| Variant | Seed | Win rate vs stock |" in table and "62.0% [55.0, 69.0]" in table
    assert "not run" in table
    pointers = {s["pointer"] for s in facts.sources}
    assert "/primary/drift/pooled" in pointers and "/tables/utility/rows/0/win_rate" in pointers
    assert {s["file"] for s in facts.sources} == {"outputs/metrics/final_report.json"}
    out = facts.write_sources(tmp_path / "x.sources.json")
    assert json.loads(out.read_text())["pending"] == []


def test_config_values_are_recorded():
    facts = Facts()
    assert facts.config("project", "sizes.harmful.dev", "int") == "172"
    assert facts.sources[-1]["file"] == "configs/project.yaml"


def test_report_version_is_checked():
    write_json(report_path(), {"version": 99})
    with pytest.raises(PublishError, match="version"):
        Facts().report()


def test_bad_table_cell_names_the_table():
    doc = {
        "version": 1,
        "tables": {
            "cost": {"columns": [{"key": "x", "label": "X", "format": "pct"}], "rows": [{"x": "lots"}]}
        },
    }
    write_report(doc)  # bound to selection.json (XB-04)
    with pytest.raises(PublishError, match="tables/cost"):
        Facts().table("cost")


def test_fill_refuses_unknown_slots():
    assert render.fill("a {{ x }} b", {"x": "1"}) == "a 1 b"
    with pytest.raises(PublishError, match="y"):
        render.fill("{{ x }} {{ y }}", {"x": "1"})


def test_templates_type_no_result_numbers():
    """Rule: no number is typed by hand. Templates may name years, versions and the 95% interval
    convention, but no rate, difference or decimal."""
    bad = re.compile(r"\d+\.\d+|\d+\s*(?:%|pp\b|points\b)")
    for path in sorted(render.templates_dir().glob("*.md")) + sorted(render.templates_dir().glob("*.txt")):
        text = path.read_text(encoding="utf-8").replace("95% interval", "")
        text = re.sub(r"[A-Za-z]-\d+\.\d+", "", text)  # licence names: CC-BY-4.0, Apache-2.0
        assert not bad.findall(text), f"{path.name}: {bad.findall(text)}"


def test_shown_paths_never_leak_home_dirs():
    from lift.publish.inputs import shown

    assert shown(paths.metrics_dir() / "x.json") == "outputs/metrics/x.json"
    assert shown(paths.REPO_ROOT / "configs" / "a.yaml") == "configs/a.yaml"


# ------------------------------------------------------------------ selection.json as P05 writes it (R1-06)


def test_selection_is_read_in_p05s_shape():
    from publish_fakes import selection

    from lift.publish.inputs import frozen_family_winners, frozen_variants, frozen_winner, selection_verdict

    write_inputs()
    facts = Facts()
    assert frozen_winner(facts) == "wiseft-l0.50-adapter-aya-std-s1"
    assert selection_verdict(facts) == "repair"
    assert frozen_family_winners(facts) == selection()["frozen"]
    names = frozen_variants(facts)
    assert len(names) == 9 and "wiseft" not in names and "cast" not in names
    assert "resta-a0.25-adapter-aya-std-s3" in names and "cast-m1.0-adapter-aya-std-s2" in names


def test_a_stress_adapters_frozen_settings_are_on_its_one_seed():
    """The stress recipe trains seed 1 only (docs/decisions.md 82), so its frozen settings exist on seed 1
    alone, as the final eval ran them: the parameters repo waits for no seed-2 or seed-3 entry."""
    from publish_fakes import selection

    from lift.publish.inputs import frozen_variants

    doc = selection()
    doc["frozen"] = {
        f: n.replace("adapter-aya-std-s1", "adapter-indicalign-stress-s1") for f, n in doc["frozen"].items()
    }
    doc["overall"]["winner"] = doc["overall"]["winner"].replace(
        "adapter-aya-std-s1", "adapter-indicalign-stress-s1"
    )
    write_inputs(selection_doc=doc)
    assert frozen_variants(Facts()) == list(doc["frozen"].values())


def test_incomplete_selection_freezes_nothing():
    from publish_fakes import selection

    from lift.publish.inputs import frozen_family_winners, frozen_variants, frozen_winner, selection_problem

    write_inputs(selection_doc=selection(incomplete=True))
    facts = Facts()
    assert (
        frozen_winner(facts) is None and frozen_family_winners(facts) == {} and frozen_variants(facts) == []
    )
    assert "incomplete" in selection_problem(facts)


def test_selection_in_another_shape_is_refused():
    from lift.publish.inputs import frozen_winner, selection_path

    write_json(selection_path(), {"status": "decided", "winner": "x", "frozen": ["x"]})
    with pytest.raises(PublishError, match="overall"):
        frozen_winner(Facts())


# ------------------------------------------------------------------ free text reaches the sources (R1-24)


def test_table_text_cells_and_notes_are_recorded():
    doc = {
        "version": 1,
        "tables": {
            "cost": {
                "columns": [{"key": "x", "label": "X", "format": "text"}],
                "rows": [{"x": "a note with 12.5 inside"}],
                "note": "Pooled over 3 seeds.",
            }
        },
    }
    write_report(doc)  # bound to selection.json (XB-04)
    facts = Facts()
    facts.table("cost")
    pointers = {s["pointer"]: s["shown"] for s in facts.sources}
    assert pointers["/tables/cost/rows/0/x"] == "a note with 12.5 inside"
    assert pointers["/tables/cost/note"] == "Pooled over 3 seeds."


# -------------------------------------------------------------- the report is bound to the selection (XB-04)


def _bound():
    from publish_fakes import report, selection

    write_inputs()
    return report(), selection()


def test_a_report_bound_to_the_current_selection_is_read():
    from lift.publish.inputs import frozen_winner

    _bound()
    facts = Facts()
    assert facts.report()["selection"]["winner"] == frozen_winner(facts)


def test_a_selection_changed_after_lift_stats_is_refused():
    """XB-04: selection.json's winner changed after lift stats; the report's numbers are another
    winner's, so nothing that reads them may render."""
    from publish_fakes import CAST_WINS, selection

    from lift.publish.inputs import selection_path

    _bound()
    write_json(selection_path(), selection(CAST_WINS))
    with pytest.raises(PublishError, match="lift stats"):
        Facts().report()
    with pytest.raises(PublishError, match="lift stats"):
        Facts().raw(report_path(), "/verdicts")  # every reader of the report, not only report()
    with pytest.raises(PublishError, match="lift stats"):
        Facts().report_value("/primary/repair/pooled", "pp_ci", what="r")


@pytest.mark.parametrize(
    "change",
    ["no block", "no sha", "bad sha", "winner", "frozen", "no selection.json", "undecided"],
)
def test_a_report_that_disagrees_with_the_selection_is_refused(change):
    from publish_fakes import selection

    from lift.publish.inputs import selection_path

    _bound()
    doc = json.loads(report_path().read_text())
    block = doc["selection"]
    if change == "no block":
        doc["selection"] = {"adapter": "adapter-aya-std-s1", "winner_family": "wiseft"}  # P07 before XB-04
    elif change == "no sha":
        del block["sha256"]
    elif change == "bad sha":
        block["sha256"] = "0" * 64
    elif change == "winner":
        block["winner"] = "cast-m1.0-adapter-aya-std-s1"
    elif change == "frozen":
        block["frozen"] = {**block["frozen"], "cast": "cast-m2.0-adapter-aya-std-s1"}
    elif change == "no selection.json":
        selection_path().unlink()
    else:
        write_json(selection_path(), selection(incomplete=True))
    write_json(report_path(), doc)
    with pytest.raises(PublishError, match="selection"):
        Facts().report()


def test_the_committed_frozen_recipe_must_agree():
    """configs/frozen_recipe.json (P05, XB-02) pins the selection: a selection.json or a report that
    disagrees with it is refused, with or without a report (the recipe as its tag froze it: XB2-02)."""
    from publish_fakes import CAST_WINS, frozen_recipe, selection, tag_recipe, write_recipe

    def freeze(doc):  # committed and tagged, so only the agreement is under test here
        write_recipe(doc)
        tag_recipe()

    from lift.publish.inputs import frozen_variants, frozen_winner, selection_path

    _bound()
    freeze(frozen_recipe())
    assert Facts().report() and frozen_winner(Facts()) == selection()["overall"]["winner"]
    for change in ("selection_sha256", "overall_winner", "frozen", "adapter", "version"):
        doc = frozen_recipe()
        doc[change] = {"frozen": {"wiseft": "x"}, "version": 2}.get(change, "0" * 64)
        freeze(doc)
        with pytest.raises(PublishError, match="frozen_recipe.json"):
            Facts().report()
    # without any report: the model and parameters read selection.json alone, and it must match too
    freeze(frozen_recipe(selection(CAST_WINS)))
    report_path().unlink()
    with pytest.raises(PublishError, match="frozen_recipe.json"):
        frozen_winner(Facts())
    with pytest.raises(PublishError, match="frozen_recipe.json"):
        frozen_variants(Facts())
    write_json(selection_path(), selection(CAST_WINS))
    assert frozen_winner(Facts()) == "cast-m1.0-adapter-aya-std-s1"
    freeze({"version": 1})
    with pytest.raises(PublishError, match="frozen_recipe.json"):
        frozen_winner(Facts())
