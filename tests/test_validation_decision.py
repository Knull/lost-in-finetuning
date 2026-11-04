"""lift validate decide-judge: the Nov 6 judge decision rule (P04 spec, Build 5).

lift.validation.score.decision is the rule itself; decide_primary_judge applies it to round 1,
pooled, writes outputs/metrics/judge_decision.json and never edits configs/project.yaml or
docs/decisions.md. The test the spec asks for is the boundary: exactly 5 points, just under, just
over.

Round 1 has 180 items (60 hi, 40 each zh, es and en), labelled here with the true label of each
fake output. In the boundary cases Qwen3Guard gets 36 items wrong (144 right) and the frontier judge
27, 28 or 26 (153, 152 or 154 right). The frontier judge's whole edge is on Hindi items and its own
errors are on Spanish ones, so only the pooled count sits on the line. Worlds are fake
(tests/validation_fakes.py): neutral placeholder text, no model, no network, all under tmp_path.

Round 1 is drawn and its true labels imported once per module, and each set of judgements is
written once; every test hard-links these files into its own tree. With on-access virus scanning
(Windows) the first open of a new file can cost 10-20 ms and a decision opens about 70 files, while
a hard link to a file opened before is fast.
"""

import copy
import csv
import dataclasses
import itertools
import json
import os
import re
import shutil
import textwrap
from datetime import date
from fractions import Fraction
from pathlib import Path

import pytest
import yaml
from scipy.stats import binomtest
from validation_fakes import (
    FRONTIER_ID,
    LANGS,
    PRIMARY_ID,
    SAFETY_SETS,
    World,
    build_world,
    fill_sheet,
    judgement,
    make_generation,
    other_label,
    prompt_text,
    read_csv,
    truth,
    use_world,
    write_generations,
)

from lift import config, paths
from lift.cli import main
from lift.io import read_jsonl, write_jsonl
from lift.schemas import LABELS, PromptRecord
from lift.validation import score as scoring
from lift.validation.labels import import_labels
from lift.validation.sample import load_round, make_sheets, sample_digest
from lift.validation.store import ValidateError, read_json, sheets_dir

THRESHOLD = 0.05            # configs/project.yaml judges.decision_rule
N = 180                     # round 1: 60 hi + 40 zh + 40 es + 40 en
RULE_KEY = "judges.decision_rule.switch_primary_if_frontier_agreement_exceeds_by"
LOCKED = (paths.CONFIG_DIR / "project.yaml", paths.REPO_ROOT / "docs" / "decisions.md")
_SHARED: dict = {}          # set by the drawn fixture: files made once and hard-linked into tests


# Helpers ------------------------------------------------------------------------------------------


def warm(root: Path) -> None:
    """Read every file once, so that later opens (also through hard links) are fast."""
    for p in root.rglob("*"):
        if p.is_file():
            p.read_bytes()


def _link(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:                  # a file system without hard links
        shutil.copy2(src, dst)


def link_tree(src: Path, dst: Path) -> None:
    """Replace ``dst`` with hard links to the files of ``src``. lift.validation replaces files
    atomically (temp file and rename), so a test never changes the shared copy through a link."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, copy_function=_link)


def made_once(name: tuple, make) -> Path:
    """The directory ``make(path)`` writes, made and read once per module."""
    made = _SHARED["made"]
    if name not in made:
        path = _SHARED["dir"] / f"made{len(made)}"
        make(path)
        warm(path)
        made[name] = path
    return made[name]


def round_items() -> list[dict]:
    return load_round(1)["items"]


def wrong_keys(*, both: int = 10, qwen_only: int = 26, frontier_only: int = 17) -> tuple[set, set]:
    """Keys of the round 1 items each judge gets wrong: ``both`` zh items for both judges,
    ``qwen_only`` hi items for Qwen3Guard alone and ``frontier_only`` es (then en) items for the
    frontier judge alone. The defaults put the frontier judge exactly 5 points ahead."""
    keys = {lang: [it["key"] for it in round_items() if it["lang"] == lang and it["slot"] == "a"]
            for lang in LANGS}
    shared = keys["zh"][:both]
    return (set(shared + keys["hi"][:qwen_only]),
            set(shared + (keys["es"] + keys["en"])[:frontier_only]))


def judge(world: World, dirname: str, judge_id: str, wrong: set) -> None:
    """outputs/judgements/{dirname}/: ``judge_id`` judges every output, right except on ``wrong``
    (records from validation_fakes.judgement, as write_judgements makes them)."""

    def make(path: Path) -> None:
        for (variant, set_name, lang, split), recs in world.generations.items():
            rows = [judgement(g, judge_id, other_label(truth(g.key)) if g.key in wrong else truth(g.key))
                    for g in recs]
            write_jsonl(path / variant / f"{set_name}_{lang}_{split}.jsonl", [r.model_dump() for r in rows])

    link_tree(made_once((judge_id, frozenset(wrong)), make), world.outputs / "judgements" / dirname)


def judge_round(world: World, qwen_wrong: set, frontier_wrong: set) -> None:
    """Both judges judge every output, right except on their wrong keys."""
    judge(world, "qwen3guard", PRIMARY_ID, qwen_wrong)
    judge(world, "frontier", FRONTIER_ID, frontier_wrong)


def drop_rows(path: Path, item_ids) -> None:
    """The reader deleted these rows before sending the sheet back."""
    rows = read_csv(path)
    kept = rows[:2] + [r for r in rows[2:] if r[0] not in item_ids]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(kept)


def fill_and_import(world: World, *, values=None, notes=None, drop=(), langs=LANGS, slot="a") -> None:
    """Fill the round 1 sheets with the true labels (``values`` overrides cells) and import them.
    The filled files go next to the world's tree, never into it (fill_sheet writes in place)."""
    doc = load_round(1)
    for name, info in doc["sheets"].items():
        if info["lang"] not in langs or info["slot"] != slot:
            continue
        cells = {it["item_id"]: truth(it["key"]) for it in doc["items"] if it["sheet"] == name}
        filled = world.root.parent / "filled" / f"{name}.csv"
        filled.parent.mkdir(exist_ok=True)
        fill_sheet(sheets_dir() / info["file"], filled, {**cells, **(values or {})}, notes=notes)
        if drop:
            drop_rows(filled, drop)
        import_labels(1, info["lang"], filled)


def import_round(world: World, **options) -> None:
    """fill_and_import; without options, links in the true labels the drawn fixture imported (on a
    tree with no labels yet: the link replaces validation/labels)."""
    if options or sample_digest(load_round(1)) != _SHARED["digest"]:
        fill_and_import(world, **options)
    else:
        link_tree(_SHARED["labels"], world.validation / "labels")


def setup_round(world: World, *, frontier_only: int = 17) -> tuple[set, set]:
    """Round 1 judged and fully labelled: Qwen3Guard 144 of 180 right, the frontier judge
    170 - frontier_only (153 by default: exactly 5 points ahead)."""
    qwen_wrong, frontier_wrong = wrong_keys(frontier_only=frontier_only)
    judge_round(world, qwen_wrong, frontier_wrong)
    import_round(world)
    return qwen_wrong, frontier_wrong


def judged_as(key: str, qwen_wrong: set, frontier_wrong: set) -> tuple[str, str, str]:
    """(reader, Qwen3Guard, frontier) labels of an item; the reader gives the true label."""
    t = truth(key)
    return t, other_label(t) if key in qwen_wrong else t, other_label(t) if key in frontier_wrong else t


def drop_judgement(world: World, judge_dir: str, item: dict) -> None:
    """The judge has not run on this item yet."""
    path = (world.outputs / "judgements" / judge_dir / item["variant"]
            / f"{item['set']}_{item['lang']}_{item['split']}.jsonl")
    write_jsonl(path, [r for r in read_jsonl(path) if r["key"] != item["key"]])


def decision_file(world: World) -> Path:
    return world.outputs / "metrics" / "judge_decision.json"


def snapshot(root: Path) -> dict:
    """Every path under ``root`` with its size and modification time."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*")}


def locked_files() -> dict:
    return {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in LOCKED}


def on_day(monkeypatch, day: date) -> None:
    """lift.validation.score sees ``day`` as today."""

    class Today(date):
        @classmethod
        def today(cls):
            return cls(day.year, day.month, day.day)

    monkeypatch.setattr(scoring, "date", Today)


def patch_config(monkeypatch, changes: dict) -> None:
    """configs/project.yaml with ``changes`` (dotted key -> value), for one test."""
    project = copy.deepcopy(config.load("project"))
    for dotted, value in changes.items():
        *parents, last = dotted.split(".")
        node = project
        for part in parents:
            node = node[part]
        node[last] = copy.deepcopy(value)
    real = config.load
    monkeypatch.setattr(config, "load", lambda name="project": project if name == "project" else real(name))


def yaml_blocks(text: str) -> tuple[dict, dict]:
    """The "Now:" and "After:" YAML blocks of config_change's text."""
    now, after = text.split("\nNow:\n", 1)[1].split("\nAfter:\n", 1)
    return yaml.safe_load(textwrap.dedent(now)), yaml.safe_load(textwrap.dedent(after))


def next_entry_number() -> int:
    """One more than the highest numbered entry in the real docs/decisions.md."""
    text = (paths.REPO_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    return max(int(n) for n in re.findall(r"^(\d+)\.", text, flags=re.M)) + 1


# Fixtures -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drawn(tmp_path_factory) -> World:
    """Round 1 drawn once from stock and seed-1 adapter dev outputs, with no judgements or labels
    in it; the true labels are imported once and kept aside for import_round."""
    base = tmp_path_factory.mktemp("decision")
    world = build_world(base / "drawn", n_dev=40, n_final=0, dev_variants=("stock", "adapter-aya-std-s1"),
                        final_variants=(), judgements=False, utility=False)
    _SHARED.clear()
    _SHARED.update(dir=base / "shared", made={}, labels=base / "labels")
    with pytest.MonkeyPatch.context() as mp:
        use_world(mp, world)
        mp.setattr("lift.manifest.git_sha", lambda: "test")
        make_sheets(1)
        fill_and_import(world)
        _SHARED["digest"] = sample_digest(load_round(1))
    shutil.move(world.validation / "labels", _SHARED["labels"])
    warm(base)
    return world


@pytest.fixture
def world(drawn, tmp_path, monkeypatch) -> World:
    """This test's own tree: hard links to the drawn round (prompts, outputs, keys, sheets)."""
    root = tmp_path / "world"
    link_tree(drawn.root, root)
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")
    return use_world(monkeypatch, dataclasses.replace(drawn, root=root, generations=dict(drawn.generations),
                                                      primary_flip=set(), frontier_flip=set()))


# The rule ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "qwen", "frontier", "switch", "boundary"), [
    (180, 144, 153, True, True),        # exactly 5 points
    (180, 144, 152, False, False),      # just under
    (180, 144, 154, True, False),       # just over
    (100, 80, 85, True, True),
    (100, 80, 84, False, False),
    (100, 80, 86, True, False),
])
def test_decision_switches_at_5_points_or_more(n, qwen, frontier, switch, boundary):
    out = scoring.decision(n, qwen, frontier, THRESHOLD)
    assert out["switch_primary"] is switch
    assert out["boundary"] is boundary
    assert out["n"] == n
    assert out["n_agree"] == {"qwen3guard_strict": qwen, "frontier": frontier}
    assert out["agreement"] == {"qwen3guard_strict": pytest.approx(qwen / n),
                                "frontier": pytest.approx(frontier / n)}
    assert out["difference"] == pytest.approx((frontier - qwen) / n)
    assert out["threshold"] == THRESHOLD


def test_exactly_5_points_is_not_lost_to_float_rounding():
    # Subtracting the two agreement rates in floats lands just under 0.05 in both cases.
    assert 153 / 180 - 144 / 180 < THRESHOLD
    assert 85 / 100 - 80 / 100 < THRESHOLD
    for n, qwen, frontier in ((180, 144, 153), (100, 80, 85)):
        out = scoring.decision(n, qwen, frontier, THRESHOLD)
        assert out["difference_exact"] == "1/20"
        assert out["difference"] == 0.05
        assert out["switch_primary"] is True and out["boundary"] is True


@pytest.mark.parametrize(("qwen", "frontier"), [(153, 144), (144, 144), (180, 0), (144, 152)])
def test_a_frontier_judge_that_is_not_5_points_better_never_switches(qwen, frontier):
    out = scoring.decision(N, qwen, frontier, THRESHOLD)
    assert out["switch_primary"] is False
    assert out["boundary"] is False        # 5 points worse is not the boundary either


@pytest.mark.parametrize(("n", "qwen", "frontier"), [(0, 0, 0), (-4, 0, 0), (10, 11, 5), (10, 5, 11),
                                                     (10, -1, 5)])
def test_decision_rejects_impossible_counts(n, qwen, frontier):
    with pytest.raises(ValueError):
        scoring.decision(n, qwen, frontier, THRESHOLD)


# Round 1 through the files ------------------------------------------------------------------------


@pytest.mark.parametrize(("frontier_only", "agree_frontier", "switch", "boundary"), [
    (17, 153, True, True),        # 85.0% vs 80.0%: exactly 5 points
    (18, 152, False, False),      # just under
    (16, 154, True, False),       # just over
], ids=["exactly-5-points", "just-under", "just-over"])
def test_decide_primary_judge_on_round_1(world, frontier_only, agree_frontier, switch, boundary):
    setup_round(world, frontier_only=frontier_only)
    doc = scoring.decide_primary_judge()
    assert doc["n"] == N
    assert doc["n_agree"] == {"qwen3guard_strict": 144, "frontier": agree_frontier}
    assert doc["agreement"]["qwen3guard_strict"] == pytest.approx(0.8)
    assert doc["agreement"]["frontier"] == pytest.approx(agree_frontier / N)
    assert doc["threshold"] == THRESHOLD
    assert doc["boundary"] is boundary
    assert doc["rule_met"] is switch
    assert doc["rule_applies"] is True
    assert doc["switch_primary"] is switch
    assert doc["primary_after"] == ("frontier" if switch else "qwen3guard")
    assert (doc["qwen3guard_judge"], doc["frontier_judge"]) == (PRIMARY_ID, FRONTIER_ID)
    assert doc["allowed_incomplete"] is False
    # Pooled over the four languages: Hindi alone is 34 vs 60 of 60.
    per_lang = doc["per_lang"]
    assert sum(per_lang[lang]["frontier"]["n_agree"] for lang in LANGS) == agree_frontier
    assert sum(per_lang[lang]["qwen3guard_strict"]["n_agree"] for lang in LANGS) == 144
    assert (per_lang["hi"]["qwen3guard_strict"]["n_agree"], per_lang["hi"]["frontier"]["n_agree"]) == (34, 60)
    path = decision_file(world)
    assert path == scoring.decision_path()
    assert read_json(path) == json.loads(json.dumps(doc, default=str))
    # Recorded, not applied: a person edits the config, so the primary judge has not changed yet.
    assert scoring.recorded_decision() == ("frontier" if switch else "qwen3guard")
    assert doc["config_primary"] == "qwen3guard"
    assert scoring.primary_view() == "qwen3guard_strict"


def test_refuses_before_round_1_is_drawn(tmp_path, monkeypatch):
    use_world(monkeypatch, World(root=tmp_path))
    with pytest.raises(ValidateError):
        scoring.decide_primary_judge()


def test_refuses_until_every_round_1_sheet_is_imported(world):
    judge_round(world, *wrong_keys())
    import_round(world, langs=("hi", "zh", "es"))
    for allow in (False, True):          # allow_incomplete is about rows, not whole sheets
        with pytest.raises(ValidateError, match="round1_en"):
            scoring.decide_primary_judge(allow_incomplete=allow)
    assert not decision_file(world).exists()


@pytest.mark.parametrize("gap", ["unlabelled", "missing"])
def test_refuses_rows_without_a_label_unless_allow_incomplete(world, gap):
    judge_round(world, *wrong_keys())
    item = next(it for it in round_items() if it["lang"] == "zh")
    if gap == "unlabelled":              # label left empty, no "skipped" note
        import_round(world, values={item["item_id"]: ""})
    else:                                # the row was deleted from the sheet
        import_round(world, drop={item["item_id"]})
    with pytest.raises(ValidateError, match=gap):
        scoring.decide_primary_judge()
    assert not decision_file(world).exists()
    doc = scoring.decide_primary_judge(allow_incomplete=True)
    assert doc["n"] == N - 1
    assert doc["allowed_incomplete"] is True
    assert doc["not_labelled"] == {gap: 1}
    assert doc["diagnostics"]["skips"]["not_labelled"] == 1
    recorded = read_json(decision_file(world))
    assert recorded["allowed_incomplete"] is True and recorded["not_labelled"] == {gap: 1}


@pytest.mark.parametrize("missing", [
    "qwen3guard",                                       # on the first item
    "frontier",
    pytest.param("frontier_late"),        # on the last item
    pytest.param("frontier_not_run"),     # on every item
])
def test_refuses_when_a_judge_misses_a_labelled_item(world, missing):
    qwen_wrong, frontier_wrong = wrong_keys()
    if missing == "frontier_not_run":
        judge(world, "qwen3guard", PRIMARY_ID, qwen_wrong)
    else:
        judge_round(world, qwen_wrong, frontier_wrong)
        drop_judgement(world, missing.split("_")[0], round_items()[-1 if missing == "frontier_late" else 0])
    import_round(world)
    for allow in (False, True):          # allow_incomplete is about labels, not judgements
        with pytest.raises(ValidateError, match="judge"):
            scoring.decide_primary_judge(allow_incomplete=allow)
    assert not decision_file(world).exists()


def test_an_item_remade_after_the_draw_is_counted_not_dropped(world):
    judge_round(world, *wrong_keys())
    import_round(world)
    item = [it for it in round_items() if (it["set"], it["lang"]) == ("harmful", "hi")][-1]
    # Its prompt's translation was fixed and the output remade under a new key, so the frontier
    # judge, which had not judged the old output yet, never will.
    path = world.data / "prompts" / "harmful_hi.jsonl"
    fixed = [p.model_copy(update={"text": prompt_text("harmful", "hi", p.pid, version=1)})
             if p.pid == item["pid"] else p for p in map(PromptRecord.model_validate, read_jsonl(path))]
    write_jsonl(path, [p.model_dump() for p in fixed])
    cell = (item["variant"], "harmful", "hi", "dev")
    remade = make_generation(item["variant"], next(p for p in fixed if p.pid == item["pid"]))
    write_generations(world, *cell, [*world.generations[cell], remade])
    drop_judgement(world, "frontier", item)
    with pytest.raises(ValidateError, match="superseded"):
        scoring.decide_primary_judge()
    doc = scoring.decide_primary_judge(allow_incomplete=True)
    assert (doc["n"], doc["superseded"], doc["allowed_incomplete"]) == (N - 1, 1, True)
    skips = doc["diagnostics"]["skips"]
    assert (skips["superseded"], skips["not_labelled"], skips["left_out"]) == (1, 0, 1)   # in the bounds


def test_refuses_a_round_with_no_labels(world):
    judge_round(world, *wrong_keys())
    every = [it["item_id"] for it in round_items()]
    import_round(world, values=dict.fromkeys(every, ""), notes=dict.fromkeys(every, "skipped"))
    with pytest.raises(ValidateError):
        scoring.decide_primary_judge()
    assert not decision_file(world).exists()


def test_deliberate_skips_are_allowed_and_bound_the_margin(world):
    qwen_wrong, frontier_wrong = wrong_keys()
    judge_round(world, qwen_wrong, frontier_wrong)
    items = round_items()
    # Two Hindi items only Qwen3Guard gets wrong (the judges disagree), and one both get right.
    skipped = [it for it in items if it["key"] in qwen_wrong - frontier_wrong][:2]
    skipped.append(next(it for it in items if it["key"] not in qwen_wrong | frontier_wrong))
    ids = [it["item_id"] for it in skipped]
    import_round(world, values=dict.fromkeys(ids, ""), notes=dict.fromkeys(ids, "skipped"))
    doc = scoring.decide_primary_judge()
    assert doc["n"] == N - 3
    assert doc["n_agree"] == {"qwen3guard_strict": 143, "frontier": 150}
    assert doc["switch_primary"] is False          # 7 of 177 is under 5 points
    assert doc["not_labelled"] == {"skipped": 3}
    assert doc["allowed_incomplete"] is False
    s = doc["diagnostics"]["skips"]
    assert (s["not_labelled"], s["judges_disagree_or_missing"]) == (3, 2)
    # Every label the reader could have given the three skipped items lies within the bounds, and
    # both bounds are reached. With the true labels the margin is exactly 5 points, so it could flip.
    judges = [judged_as(it["key"], qwen_wrong, frontier_wrong)[1:] for it in skipped]
    margins = [Fraction(doc["diagnostics"]["margin_items"]
                        + sum((h == f) - (h == q) for h, (q, f) in zip(combo, judges, strict=True)), N)
               for combo in itertools.product(LABELS, repeat=len(skipped))]
    assert s["margin_lo"] == pytest.approx(float(min(margins)))
    assert s["margin_hi"] == pytest.approx(float(max(margins)))
    assert s["margin_lo"] <= 9 / N <= s["margin_hi"]
    # Robust only if every way of labelling them gives the decision taken.
    assert s["robust_to_skips"] is all((m >= Fraction(1, 20)) == doc["switch_primary"] for m in margins)
    assert s["robust_to_skips"] is False


def test_skips_that_would_dilute_the_margin_below_5_points_are_not_robust(world):
    qwen_wrong, frontier_wrong = wrong_keys(frontier_only=18)        # 152 vs 144: 8 items ahead
    judge_round(world, qwen_wrong, frontier_wrong)
    agreed = [it for it in round_items() if it["key"] not in qwen_wrong | frontier_wrong][:20]
    ids = [it["item_id"] for it in agreed]
    import_round(world, values=dict.fromkeys(ids, ""), notes=dict.fromkeys(ids, "skipped"))
    doc = scoring.decide_primary_judge()
    assert (doc["n"], doc["boundary"], doc["switch_primary"]) == (N - 20, True, True)   # 8 of 160
    # Both judges are right on the skipped items, so however they are labelled the margin is 8 of
    # 180, under 5 points: the bounds agree with each other but not with the decision taken.
    s = doc["diagnostics"]["skips"]
    assert (s["not_labelled"], s["judges_disagree_or_missing"]) == (20, 0)
    assert s["margin_lo"] == s["margin_hi"] == pytest.approx(8 / N)
    assert s["robust_to_skips"] is False


def test_a_skipped_item_needs_no_judgement(world):
    qwen_wrong, frontier_wrong = wrong_keys()
    judge_round(world, qwen_wrong, frontier_wrong)
    item = next(it for it in round_items() if it["key"] not in qwen_wrong | frontier_wrong)
    for judge_dir in ("qwen3guard", "frontier"):     # neither judge has run on it
        drop_judgement(world, judge_dir, item)
    import_round(world, values={item["item_id"]: ""}, notes={item["item_id"]: "skipped"})
    doc = scoring.decide_primary_judge()
    assert doc["n"] == N - 1
    assert doc["n_agree"] == {"qwen3guard_strict": 143, "frontier": 152}
    assert doc["switch_primary"] is True           # 9 of 179
    s = doc["diagnostics"]["skips"]
    assert (s["not_labelled"], s["judges_disagree_or_missing"]) == (1, 1)
    assert (s["margin_lo"], s["margin_hi"]) == (pytest.approx(8 / N), pytest.approx(10 / N))
    assert s["robust_to_skips"] is False


def test_diagnostics_explain_the_margin(world):
    qwen_wrong, frontier_wrong = setup_round(world)
    doc = scoring.decide_primary_judge()
    d = doc["diagnostics"]
    rows = [(it, *judged_as(it["key"], qwen_wrong, frontier_wrong)) for it in round_items()]
    gain = [(f == h) - (q == h) for _, h, q, f in rows]       # +1: only the frontier judge is right
    assert d["margin_items"] == sum(gain) == 9

    by_class = d["margin_by_human_class"]
    assert set(by_class) == set(LABELS)
    for c in LABELS:
        mine = sum(g for (_, h, _, _), g in zip(rows, gain, strict=True) if h == c)
        assert by_class[c] == pytest.approx(mine / N)
    assert sum(by_class.values()) == pytest.approx(doc["difference"])

    clear = [g for (_, h, _, _), g in zip(rows, gain, strict=True) if h != "unclear"]
    assert d["n_without_unclear"] == len(clear)
    assert d["margin_without_unclear"] == pytest.approx(sum(clear) / len(clear))
    harm = [((f == "harmful") == (h == "harmful")) - ((q == "harmful") == (h == "harmful"))
            for _, h, q, f in rows]
    assert d["margin_binary"] == pytest.approx(sum(harm) / N)
    sizes = {s: config.get(f"sizes.{s}.dev") for s in SAFETY_SETS}
    by_set = {s: [g for (it, *_), g in zip(rows, gain, strict=True) if it["set"] == s] for s in SAFETY_SETS}
    weighted = sum(sizes[s] * sum(v) / len(v) for s, v in by_set.items()) / sum(sizes.values())
    assert d["margin_weighted_by_dev_set_size"] == pytest.approx(weighted)

    # The 26 Hindi items only Qwen3Guard gets wrong and the 17 Spanish ones only the frontier does.
    assert d["discordant"]["frontier_only_right"] == 26
    assert d["discordant"]["qwen3guard_only_right"] == 17
    assert d["discordant"]["mcnemar_p"] == pytest.approx(binomtest(26, 43, 0.5).pvalue)
    assert d["judges_differ"] == sum(q != f for _, _, q, f in rows) == 43   # both-wrong items agree
    boot = d["bootstrap"]
    assert boot["difference"] == pytest.approx(doc["difference"])
    assert boot["lo"] <= boot["difference"] <= boot["hi"]
    assert boot["clusters"] == len({it["pid"] for it in round_items()})    # one cluster per prompt
    assert d["skips"] == {"left_out": 0, "not_labelled": 0, "superseded": 0,
                          "judges_disagree_or_missing": 0, "margin_lo": 0.05, "margin_hi": 0.05,
                          "robust_to_skips": True}


# The decision file --------------------------------------------------------------------------------


def test_dry_run_writes_nothing(world):
    setup_round(world)
    before = snapshot(world.root)
    doc = scoring.decide_primary_judge(dry_run=True)
    assert doc["switch_primary"] is True
    assert snapshot(world.root) == before
    assert not decision_file(world).exists()
    # Nor over a recorded decision that the current inputs would change.
    scoring.decide_primary_judge()
    judge_round(world, *wrong_keys(frontier_only=18))
    before = snapshot(world.root)
    doc = scoring.decide_primary_judge(dry_run=True)
    assert doc["switch_primary"] is False
    assert snapshot(world.root) == before


def test_a_second_run_with_the_same_inputs_keeps_the_file(world, monkeypatch):
    setup_round(world)
    on_day(monkeypatch, date(2026, 11, 6))
    scoring.decide_primary_judge()
    first = decision_file(world).read_bytes()
    on_day(monkeypatch, date(2026, 11, 8))
    again = scoring.decide_primary_judge()
    assert again["decided"] == "2026-11-06"
    assert again["history"] == []
    assert decision_file(world).read_bytes() == first
    scoring.decide_primary_judge(force=True)               # nothing to replace: no history entry
    assert decision_file(world).read_bytes() == first


def test_a_different_decision_needs_force_and_keeps_the_old_one(world, monkeypatch):
    setup_round(world)
    on_day(monkeypatch, date(2026, 11, 6))
    scoring.decide_primary_judge()
    path = decision_file(world)
    first = read_json(path)
    judge_round(world, *wrong_keys(frontier_only=18))      # the frontier judge now one item worse
    on_day(monkeypatch, date(2026, 11, 8))
    with pytest.raises(ValidateError, match="different decision"):
        scoring.decide_primary_judge()
    assert read_json(path) == first
    doc = scoring.decide_primary_judge(force=True)
    recorded = read_json(path)
    assert (doc["switch_primary"], recorded["switch_primary"], recorded["primary_after"]) == (
        False, False, "qwen3guard")
    assert recorded["decided"] == "2026-11-08"
    assert recorded["history"] == [{k: v for k, v in first.items() if k != "history"}]
    assert scoring.decide_primary_judge()["history"] == recorded["history"]    # same inputs again
    # A third decision keeps both earlier ones, oldest first.
    judge_round(world, *wrong_keys(frontier_only=16))
    scoring.decide_primary_judge(force=True)
    recorded = read_json(path)
    assert recorded["n_agree"]["frontier"] == 154
    assert [h["n_agree"]["frontier"] for h in recorded["history"]] == [153, 152]


# Which judge the rule is about, and where its threshold comes from ------------------------------


@pytest.mark.parametrize("named", ["fallback_judge", "config_names_it"])
def test_the_fallback_judge_never_becomes_primary(world, monkeypatch, named):
    qwen_wrong, frontier_wrong = wrong_keys(frontier_only=0)      # 170 vs 144: 14.4 points ahead
    if named == "fallback_judge":
        second = f"{config.get('judges.safety_fallback.id')}@0123abcd"
        judge(world, "qwen3guard", PRIMARY_ID, qwen_wrong)
        judge(world, "fallback", second, frontier_wrong)
    else:
        patch_config(monkeypatch, {"judges.safety_fallback.id": FRONTIER_ID.split("@", 1)[0]})
        second = FRONTIER_ID
        judge_round(world, qwen_wrong, frontier_wrong)
    import_round(world)
    doc = scoring.decide_primary_judge()
    assert doc["frontier_judge"] == second
    assert doc["n_agree"] == {"qwen3guard_strict": 144, "frontier": 170}
    assert doc["rule_met"] is True
    assert doc["rule_applies"] is False
    assert doc["switch_primary"] is False
    assert doc["primary_after"] == "qwen3guard"
    assert read_json(decision_file(world))["switch_primary"] is False
    assert scoring.recorded_decision() == "qwen3guard"


def test_the_decision_stands_after_the_judge_blocks_are_swapped(world, monkeypatch, capsys):
    setup_round(world)
    first = scoring.decide_primary_judge()
    assert first["switch_primary"] is True
    # A person applies the change: the frontier judge is now primary, and re-running finds the same
    # two judges, keeps the record and proposes nothing more.
    primary, second = config.get("judges.safety_primary"), config.get("judges.safety_second")
    patch_config(monkeypatch, {"judges.safety_primary": second, "judges.safety_second": primary})
    assert scoring.primary_view() == "frontier"
    again = scoring.decide_primary_judge()
    assert (again["qwen3guard_judge"], again["frontier_judge"]) == (PRIMARY_ID, FRONTIER_ID)
    assert (again["n_agree"], again["switch_primary"], again["history"]) == (first["n_agree"], True, [])
    assert again["config_primary"] == "frontier"
    capsys.readouterr()
    assert main(["validate", "decide-judge"]) == 0
    out = capsys.readouterr().out
    assert "judges.safety_primary" not in out and "## " not in out


def test_several_frontier_judges_need_one_named(world):
    other = "otherprovider/other-model@2026-11-02"
    judge_round(world, *wrong_keys())
    judge(world, "frontier-other", other, set())          # right on every item
    import_round(world)
    with pytest.raises(ValidateError, match="frontier"):
        scoring.decide_primary_judge()
    doc = scoring.decide_primary_judge(frontier_judge=FRONTIER_ID)
    assert (doc["frontier_judge"], doc["n_agree"]["frontier"]) == (FRONTIER_ID, 153)
    for named in (other, other.split("@", 1)[0]):          # id@revision, or the id alone
        pick = scoring.decide_primary_judge(frontier_judge=named, dry_run=True)
        assert (pick["frontier_judge"], pick["n_agree"]["frontier"]) == (other, N)


@pytest.mark.parametrize(("frontier_only", "switch", "boundary"), [
    (8, True, True),          # 162 vs 144: exactly 10 points (0.0999... in floats)
    (17, False, False),       # 5 points: enough for the locked rule, not for this one
])
def test_the_threshold_comes_from_judges_decision_rule(world, monkeypatch, frontier_only, switch, boundary):
    patch_config(monkeypatch, {RULE_KEY: 0.1})
    setup_round(world, frontier_only=frontier_only)
    doc = scoring.decide_primary_judge()
    assert doc["threshold"] == 0.1
    assert (doc["switch_primary"], doc["boundary"]) == (switch, boundary)


def test_only_the_main_readers_labels_decide(world):
    make_sheets(1, hindi_overlap=True, force=True)       # adds the second Hindi reader's sheet
    assert any(it["slot"] == "b" for it in round_items())
    setup_round(world)
    doc = scoring.decide_primary_judge()                  # the overlap sheet is not needed
    assert (doc["n"], doc["switch_primary"]) == (N, True)
    # A second reader who disagrees with the first on every item changes nothing.
    overlap = {it["item_id"]: other_label(truth(it["key"])) for it in round_items() if it["slot"] == "b"}
    import_round(world, values=overlap, langs=("hi",), slot="b")
    again = scoring.decide_primary_judge()
    assert (again["n"], again["n_agree"], again["switch_primary"]) == (N, doc["n_agree"], True)


# What a person applies by hand ------------------------------------------------------------------


def test_config_change_swaps_the_primary_and_second_blocks(world):
    setup_round(world)
    doc = scoring.decide_primary_judge()
    text = scoring.config_change(doc)
    now, after = yaml_blocks(text)
    primary, second = config.get("judges.safety_primary"), config.get("judges.safety_second")
    assert now == {"judges": {"safety_primary": primary, "safety_second": second}}
    assert after == {"judges": {"safety_primary": second, "safety_second": primary}}
    assert FRONTIER_ID in text


@pytest.mark.parametrize(("frontier_only", "frontier_pct", "exact"),
                         [(17, "85.0%", True), (16, "85.6%", False)], ids=["exactly-5-points", "just-over"])
def test_decisions_entry(world, monkeypatch, frontier_only, frontier_pct, exact):
    setup_round(world, frontier_only=frontier_only)
    doc = scoring.decide_primary_judge()
    text = scoring.decisions_entry(doc, today=date(2026, 11, 6))
    assert re.match(r"## [A-Z][a-z]{2} \d{1,2}, \d{4}: ", text)
    assert text.startswith("## Nov 6, 2026: ")
    numbered = [line for line in text.splitlines() if re.match(r"\d+\. ", line)]
    assert len(numbered) == 1
    assert numbered[0].startswith(f"{next_entry_number()}. **")
    assert frontier_pct in text and "80.0%" in text
    assert FRONTIER_ID in text and PRIMARY_ID in text
    assert ("exactly 5 points" in text) is exact
    on_day(monkeypatch, date(2026, 11, 7))
    assert scoring.decisions_entry(doc).startswith("## Nov 7, 2026: ")


@pytest.mark.parametrize(("frontier_only", "switch"), [(17, True), (18, False)],
                         ids=["frontier-wins", "qwen3guard-stays"])
def test_the_cli_prints_the_changes_and_edits_neither_file(world, capsys, monkeypatch, frontier_only, switch):
    before = locked_files()
    setup_round(world, frontier_only=frontier_only)
    on_day(monkeypatch, date(2026, 11, 6))      # the entry's date, in the CLI and below
    assert main(["validate", "decide-judge"]) == 0
    out = capsys.readouterr().out
    doc = read_json(decision_file(world))
    assert doc["switch_primary"] is switch
    change, entry = scoring.config_change(doc), scoring.decisions_entry(doc)
    assert (change in out) is switch
    assert (entry in out) is switch
    assert locked_files() == before       # configs/project.yaml and docs/decisions.md untouched
