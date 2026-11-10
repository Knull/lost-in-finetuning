"""P07's lift final without a model: the final-split guard (a frozen variant and a recipe-frozen-* tag, in
lift final and in every command that makes final outputs), the 13-variant order, the stages, resume and
GPU-hours manifest, the laptop half's lines and the ar/th back-translation sample on synthetic records.
No torch."""

from __future__ import annotations

import base64
import json

import pytest
from fakes import make_prompt

from lift import cli, config, final
from lift.io import sha256_text, write_jsonl
from lift.judge.pipeline import JudgeOutcome
from lift.models.generate import generations_path
from lift.paths import manifests_dir, metrics_dir, validation_dir
from lift.schemas import LANGS, GenerationRecord, JudgementRecord, UnclearFlags

A = "adapter-aya-std-s{}"
WINNER = "wiseft-l0.50-adapter-aya-std-s{}"
RESTA = "resta-a0.25-adapter-aya-std-s{}"
CAST = "cast-m1.0-adapter-aya-std-s{}"
OTHER = "adapter-indicalign-std-s{}"  # the data candidate these selections did not pick
TAG = ["recipe-frozen-20261107"]
SELECTION = {
    "status": "decided",
    "adapter": A.format(1),
    "overall": {"winner": RESTA.format(1), "verdict": "repair"},
    "frozen": {"wiseft": WINNER.format(1), "resta": RESTA.format(1), "cast": CAST.format(1)},
}
# configs/frozen_recipe.json as P05's lift select writes it and the tag freezes it (its sha is checked
# against selection.json's bytes by verified_recipe; the tests that pass it in skip that read).
RECIPE = {
    "version": 1,
    "adapter": A.format(1),
    "selection_sha256": "0" * 64,
    "overall_winner": RESTA.format(1),
    "frozen": dict(SELECTION["frozen"]),
}


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """Stock, three adapter seeds, every frozen setting on seeds 1 and 2 (seed 3's CAST is not registered
    yet), and an unfrozen WiSE-FT setting."""
    entries = [{"name": "stock", "method": "stock"}]
    entries += [
        {"name": A.format(s), "method": "adapter", "adapter": A.format(s), "seed": s} for s in (1, 2, 3)
    ]
    for pattern, method, params in (
        (WINNER, "wiseft", {"lambda": 0.5}),
        (RESTA, "resta", {"alpha": 0.25}),
        (CAST, "cast", {"strength": 1.0}),
    ):
        for s in (1, 2, 3):
            if pattern == CAST and s == 3:
                continue
            entries.append(
                {
                    "name": pattern.format(s),
                    "method": method,
                    "adapter": A.format(s),
                    "seed": s,
                    "params": params,
                }
            )
    entries.append(
        {
            "name": "wiseft-l0.25-adapter-aya-std-s1",
            "method": "wiseft",
            "adapter": A.format(1),
            "seed": 1,
            "params": {"lambda": 0.25},
        }
    )
    doc = {"variants": entries}
    real = config.load
    monkeypatch.setattr(config, "load", lambda name="project": doc if name == "variants" else real(name))
    return doc


def write_selection(doc=SELECTION):
    metrics_dir().mkdir(parents=True, exist_ok=True)
    (metrics_dir() / "selection.json").write_text(json.dumps(doc), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ guard


@pytest.mark.parametrize("name", ["stock", A.format(2), WINNER.format(3), CAST.format(2), RESTA.format(1)])
def test_the_guard_opens_the_final_split_to_frozen_variants_with_the_tag(name):
    assert final.check_final(name, selection=SELECTION, tags=TAG, recipe=RECIPE) == TAG[0]


def test_the_guard_refuses_without_the_tag():
    with pytest.raises(final.Refused, match="no recipe-frozen-\\* git tag"):
        final.check_final("stock", selection=SELECTION, tags=[])
    with pytest.raises(final.Refused, match="git tag"):
        final.check_final(WINNER.format(1), selection=SELECTION, tags=[])


@pytest.mark.parametrize(
    ("name", "selection", "why"),
    [
        ("wiseft-l0.25-adapter-aya-std-s1", SELECTION, "not a frozen variant"),  # swept, not frozen
        (WINNER.format(1), None, "selection.json is missing"),
        (WINNER.format(1), {**SELECTION, "status": "incomplete"}, "'incomplete'"),
        ("adapter-unknown-std-s1", SELECTION, "not registered"),
    ],
)
def test_the_guard_refuses_an_unfrozen_variant(name, selection, why):
    with pytest.raises(final.Refused, match=why):
        final.check_final(name, selection=selection, tags=TAG, use_file=False, recipe=RECIPE)


def test_the_frozen_list_in_p08s_list_form_is_read_too():
    listed = {**SELECTION, "frozen": [WINNER.format(1), CAST.format(1)]}
    assert final.frozen_variants(listed) == [WINNER.format(s) for s in (1, 2, 3)] + [
        CAST.format(s) for s in (1, 2, 3)
    ]


def test_tags_come_from_git_then_the_remote_with_the_token_as_a_header(monkeypatch):
    calls = []

    def git(argv, env=None):
        calls.append((list(argv), env))
        if argv[:1] == ["tag"]:
            return ""
        if argv[:2] == ["remote", "get-url"]:
            return "https://github.com/owner/repo.git\n"
        return "abc\trefs/tags/recipe-frozen-20261107\nabd\trefs/tags/recipe-frozen-20261107^{}\n"

    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("GIT_TRACE", "1")
    assert final.frozen_tags(git) == ["recipe-frozen-20261107"]
    argv, env = calls[-1]
    basic = base64.b64encode(b"x-access-token:secret-token").decode("ascii")
    assert "ls-remote" in argv
    # The token is never on a command line (visible in ps), not even base64-encoded in a -c header: it
    # reaches git through GIT_CONFIG_* variables of that one process.
    for args, _env in calls:
        assert not any("secret-token" in a or basic in a or "extraheader" in a for a in args)
    count = int(env["GIT_CONFIG_COUNT"])
    pairs = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(count)}
    assert pairs["http.https://github.com/.extraheader"] == f"AUTHORIZATION: basic {basic}"
    assert "GIT_TRACE" not in env and env["GIT_TERMINAL_PROMPT"] == "0"
    # Config the caller already passes through GIT_CONFIG_* is kept, the header appended after it.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.askpass")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "")
    assert final.frozen_tags(git) == ["recipe-frozen-20261107"]
    env = calls[-1][1]
    assert env["GIT_CONFIG_COUNT"] == "2" and env["GIT_CONFIG_KEY_0"] == "core.askpass"
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    for name in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
        monkeypatch.delenv(name)
    monkeypatch.delenv("GITHUB_TOKEN")
    assert final.frozen_tags(lambda argv, env=None: "" if argv[0] == "tag" else "x") == []
    assert final.frozen_tags(lambda argv, env=None: "recipe-frozen-a\nnot-a-tag\n") == ["recipe-frozen-a"]


@pytest.mark.parametrize(
    "argv",
    [
        ["final", "--variant", "stock"],
        ["generate", "--variant", "stock", "--split", "final"],
        ["utility", "--variant", A.format(1), "--split", "final", "--generate-only"],
        ["belebele", "--variant", "stock", "--split", "final", "--langs", "hi"],
        ["heldout-loss", "--variant", "stock", "--split", "final"],
    ],
)
def test_every_command_that_makes_final_outputs_refuses_without_the_tag(
    argv, monkeypatch, capsys, fake_prompt_files
):
    fake_prompt_files[("utility", "hi")] = [make_prompt("utl-aya-00001", "hi", "utility", "final")]
    monkeypatch.setattr(final, "frozen_tags", lambda git=None: [])
    monkeypatch.setitem(config.load("project")["models"]["global"], "revision", "c" * 40)
    write_selection()
    if argv[0] == "final":
        assert cli.main(argv) == 2
        assert "git tag" in capsys.readouterr().err
    else:
        with pytest.raises(SystemExit, match="closed to"):
            cli.main(argv)


def test_lift_final_refuses_an_unfrozen_variant_even_with_the_tag(monkeypatch, capsys):
    monkeypatch.setattr(final, "frozen_tags", lambda git=None: TAG)
    monkeypatch.setattr(final, "verified_recipe", lambda tag, **kw: RECIPE)
    write_selection()
    assert cli.main(["final", "--variant", "wiseft-l0.25-adapter-aya-std-s1"]) == 2
    assert "not a frozen variant" in capsys.readouterr().err
    assert not manifests_dir().exists() or not list(manifests_dir().glob("final_variant-*.json"))


# XB-01: lift select named an overall winner (wiseft-l0.25) its family did not freeze (wiseft-l0.50).
UNFROZEN_WINNER = {
    **SELECTION,
    "overall": {"winner": "wiseft-l0.25-adapter-aya-std-s1", "verdict": "repair"},
}


def test_a_selection_whose_overall_winner_is_not_frozen_is_refused():
    with pytest.raises(final.Refused, match="not one of the frozen settings.*lift select"):
        final.core_plan(UNFROZEN_WINNER)
    for name in ("stock", WINNER.format(1)):
        with pytest.raises(final.Refused, match="not one of the frozen settings"):
            final.check_final(name, selection=UNFROZEN_WINNER, tags=TAG, recipe=RECIPE)
    with pytest.raises(final.Refused, match="not one of the frozen settings"):
        final.run_all(final.Options(), selection=UNFROZEN_WINNER, tags=TAG, recipe=RECIPE, devices=1)


class FakeGit:
    """git show HEAD:/<tag>: of configs/frozen_recipe.json; a shallow clone has no tag until it fetches it."""

    def __init__(self, head, tagged, *, shallow=False):
        self.head, self.tagged, self.shallow, self.calls = head, tagged, shallow, []

    def __call__(self, argv, env=None):
        self.calls.append((list(argv), env))
        if argv[:1] == ["show"]:
            rev = argv[1].split(":", 1)[0]
            if rev == "HEAD":
                return self.head
            return None if self.shallow else self.tagged
        if argv[:2] == ["remote", "get-url"]:
            return "https://github.com/owner/repo.git\n"
        if argv[:1] == ["fetch"]:
            self.shallow = False
            return ""
        return None


@pytest.fixture
def frozen_recipe():
    """selection.json on disk and configs/frozen_recipe.json naming its sha, written as P05's lift select
    writes it (select.recipe_text) at lift.paths.frozen_recipe_path(), which conftest moves to tmp."""
    from lift import paths
    from lift.repair.select import recipe_text

    write_selection()
    digest = final.sha256_file(final.selection_path())
    doc = {**RECIPE, "selection_sha256": digest}
    path = paths.frozen_recipe_path()
    assert final.recipe_file() == path and "lift-tree" in str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = recipe_text(doc)
    path.write_text(text, encoding="utf-8")
    return doc, text, path


def test_the_guard_binds_the_final_split_to_the_committed_frozen_recipe(frozen_recipe, monkeypatch):
    doc, text, path = frozen_recipe
    assert final.verified_recipe(TAG[0], git=FakeGit(text, text)) == doc
    # Through check_final: the tag's recipe opens the split to its frozen settings.
    monkeypatch.setattr(final, "_git", FakeGit(text, text))
    assert final.check_final(WINNER.format(2), tags=TAG) == TAG[0]
    # The recipe was edited after the tag (committed, or not even committed).
    other = text.replace(WINNER.format(1), "wiseft-l0.25-adapter-aya-std-s1")
    with pytest.raises(final.Refused, match="changed since the tag"):
        final.verified_recipe(TAG[0], git=FakeGit(other, text))
    with pytest.raises(final.Refused, match="differs from HEAD"):
        final.verified_recipe(TAG[0], git=FakeGit(text.replace("1", "2", 1), text.replace("1", "2", 1)))
    with pytest.raises(final.Refused, match="not committed"):
        final.verified_recipe(TAG[0], git=FakeGit(None, text))
    with pytest.raises(final.Refused, match="cannot be read at the tag"):
        final.verified_recipe(TAG[0], git=FakeGit(text, None))
    path.unlink()
    with pytest.raises(final.Refused, match="frozen_recipe.json does not exist"):
        final.verified_recipe(TAG[0], git=FakeGit(text, text))


def test_the_reader_accepts_exactly_what_p05s_writer_writes():
    """select.selection_text / recipe_doc / write_recipe (origin/p05-repairs) against verified_recipe."""
    from lift.repair import select

    doc = {**SELECTION, "kind": "selection", "version": 1}
    text = select.selection_text(doc)
    path = final.selection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    recipe = select.recipe_doc(doc, text)
    written = select.write_recipe(recipe).read_text(encoding="utf-8")
    assert final.verified_recipe(TAG[0], git=FakeGit(written, written)) == recipe
    assert final.check_final(WINNER.format(3), selection=doc, tags=TAG, recipe=recipe) == TAG[0]


def test_lift_select_run_again_after_the_tag_is_refused_at_final_time(frozen_recipe, monkeypatch):
    """XB-02: a later lift select rewrites selection.json (another frozen set): its sha no longer matches the
    recipe the tag froze, so the guard refuses every variant, the new set's above all."""
    doc, text, _path = frozen_recipe
    moved = {**SELECTION, "frozen": {**SELECTION["frozen"], "wiseft": "wiseft-l0.25-adapter-aya-std-s1"}}
    write_selection(moved)
    with pytest.raises(final.Refused, match="selection.json is not the one the recipe froze"):
        final.verified_recipe(TAG[0], git=FakeGit(text, text))
    monkeypatch.setattr(final, "_git", FakeGit(text, text))
    for name in ("stock", "wiseft-l0.25-adapter-aya-std-s1"):
        with pytest.raises(final.Refused, match="not the one the recipe froze"):
            final.check_final(name, tags=TAG)
    # Even with the recipe read, a variant outside its frozen map is not frozen, whatever selection.json says.
    with pytest.raises(final.Refused, match="not a frozen variant"):
        final.check_final("wiseft-l0.25-adapter-aya-std-s1", selection=moved, tags=TAG, recipe=RECIPE)
    with pytest.raises(final.Refused, match="differs from the frozen recipe"):
        final.check_final("stock", selection=moved, tags=TAG, recipe=RECIPE)
    rc, summary = final.run_all(final.Options(), selection=moved, tags=TAG, devices=1, stage_fn=lambda *a: 0)
    assert summary["results"] == {} and all("recipe froze" in why for why in summary["not_run"].values())


def test_a_shallow_clone_fetches_the_tag_with_the_token_kept_out_of_argv(frozen_recipe, monkeypatch):
    doc, text, _path = frozen_recipe
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    git = FakeGit(text, text, shallow=True)
    assert final.verified_recipe(TAG[0], git=git) == doc
    fetch = [(argv, env) for argv, env in git.calls if argv[:1] == ["fetch"]]
    assert len(fetch) == 1 and f"refs/tags/{TAG[0]}:refs/tags/{TAG[0]}" in fetch[0][0][-1]
    basic = base64.b64encode(b"x-access-token:secret-token").decode("ascii")
    for argv, _env in git.calls:
        assert not any("secret-token" in a or basic in a for a in argv)
    env = fetch[0][1]
    assert env[f"GIT_CONFIG_VALUE_{int(env['GIT_CONFIG_COUNT']) - 1}"] == f"AUTHORIZATION: basic {basic}"
    monkeypatch.delenv("GITHUB_TOKEN")
    with pytest.raises(final.Refused, match="cannot be read at the tag"):
        final.verified_recipe(TAG[0], git=FakeGit(text, text, shallow=True))


def test_every_final_line_refuses_without_the_frozen_recipe(monkeypatch):
    monkeypatch.setattr(final, "frozen_tags", lambda git=None: TAG)
    write_selection()
    with pytest.raises(SystemExit, match="frozen_recipe.json does not exist"):
        final.guard_split("stock", "final")


def test_the_dev_split_is_not_guarded():
    final.guard_split("wiseft-l0.25-adapter-aya-std-s1", "dev")


# ------------------------------------------------------------------------------------------------- plan


def test_the_core_plan_completes_the_primary_endpoint_first():
    plan = final.core_plan(SELECTION)
    assert [p.name for p in plan] == [
        "stock",
        *(A.format(s) for s in (1, 2, 3)),
        *(RESTA.format(s) for s in (1, 2, 3)),  # the overall winner
        *(WINNER.format(s) for s in (1, 2, 3)),  # then cost order
        *(CAST.format(s) for s in (1, 2, 3)),
        *(OTHER.format(s) for s in (1, 2, 3)),  # the candidate not picked: exploratory, last
    ]
    assert len(plan) == 16 and [p.role for p in plan[4:7]] == ["winner"] * 3
    assert [p.role for p in plan[13:]] == ["candidate"] * 3 and {p.family for p in plan[13:]} == {None}
    with pytest.raises(final.Refused):
        final.core_plan({"status": "incomplete"})


def test_a_stress_adapter_has_the_one_seed_its_recipe_trains():
    """README, Adapter: "A stress recipe (..., seed 1)"; configs/train.yaml stress.seeds is [1]. The plan
    holds the stress adapter and each repair frozen on it once, never seeds 2-3 as arms "not run"; the
    standard adapters keep their three seeds."""
    from lift import gates

    assert gates.recipe_seeds("adapter-indicalign-stress-s1") == (1,)
    assert gates.recipe_seeds(A.format(2)) == (1, 2, 3)
    with pytest.raises(ValueError, match="not an adapter name"):
        gates.recipe_seeds("stock")
    assert final.seeds_of("stock") == (1, 2, 3) and final.seeds_of("adapter-x-stress-s1") == (1,)
    stress = "adapter-indicalign-stress-s1"
    sel = {
        "status": "decided",
        "adapter": stress,
        "standard_adapter": A.format(1),
        "overall": {"winner": f"cast-m1.0-{stress}", "verdict": "repair"},
        "frozen": {
            "wiseft": f"wiseft-l0.25-{stress}",
            "resta": f"resta-a0.10-{stress}",
            "cast": f"cast-m1.0-{stress}",
        },
    }
    plan = final.core_plan(sel)
    assert [(p.name, p.role) for p in plan] == [
        ("stock", "stock"),
        (stress, "adapter"),
        (f"cast-m1.0-{stress}", "winner"),
        *((A.format(s), "standard") for s in (1, 2, 3)),
        (f"wiseft-l0.25-{stress}", "family"),
        (f"resta-a0.10-{stress}", "family"),
        *((OTHER.format(s), "candidate") for s in (1, 2, 3)),
    ]
    assert final.frozen_variants(sel) == [
        f"wiseft-l0.25-{stress}",
        f"resta-a0.10-{stress}",
        f"cast-m1.0-{stress}",
    ]


def test_the_candidate_that_was_not_picked_joins_the_plan(monkeypatch):
    """docs/decisions.md 81: Gate 1 picks a candidate by dev win rate only, so the one that drifts can be
    the one left out. Every data candidate's standard adapter the plan does not already hold is evaluated
    on the final split as an exploratory arm."""
    assert final.other_candidates(SELECTION) == [OTHER.format(1)]
    stress = {**SELECTION, "adapter": "adapter-indicalign-stress-s1"}
    # Before the Gate 1 pick is recorded, both standard adapters are candidates.
    assert final.other_candidates(stress) == [A.format(1), OTHER.format(1)]
    (metrics_dir() / "gate2.json").parent.mkdir(parents=True, exist_ok=True)
    (metrics_dir() / "gate2.json").write_text(json.dumps({"candidate": "aya"}), encoding="utf-8")
    assert final.standard_adapter(stress) == A.format(1)
    assert final.other_candidates(stress) == [OTHER.format(1)]
    assert final.other_candidates({**stress, "standard_adapter": OTHER.format(1)}) == [A.format(1)]
    monkeypatch.setattr("lift.gates.candidates", lambda: ["aya"])
    assert final.other_candidates(SELECTION) == []


def test_run_variant_runs_each_stage_resumes_and_records_gpu_hours():
    calls = []

    def stage_fn(stage, name, opts):
        calls.append((stage, name))
        return 3 if stage == "belebele" else 0

    opts = final.Options()
    results = final.run_variant(A.format(1), opts, stage_fn=stage_fn, devices=2, tag=TAG[0])
    assert results == {"safety": 0, "utility": 0, "belebele": 3, "heldout": 0, "backtranslation": 0}
    assert final.exit_code({"x": results}) == 3
    calls.clear()
    assert final.run_variant(A.format(1), opts, stage_fn=lambda s, n, o: calls.append(s) or 0, devices=2) == {
        "safety": 0,
        "utility": 0,
        "belebele": 0,
        "heldout": 0,
        "backtranslation": 0,
    }
    assert calls == ["belebele"]  # only the stage that did not finish runs again
    docs = [json.loads(p.read_text()) for p in manifests_dir().glob("final_variant-*.json")]
    assert len(docs) == 2 and all(d["devices"] == 2 and d["gpu_hours"] >= 0 for d in docs)
    assert {d["frozen_tag"] for d in docs} == {TAG[0], None}
    # Stock is the utility reference: no utility stage of its own.
    assert "utility" not in final.run_variant("stock", opts, stage_fn=lambda s, n, o: 0, devices=1)


def test_stage_lines_are_the_harness_commands_on_the_final_split():
    opts = final.Options(dtype="fp16", batch_size=8)
    assert final.stage_argv("utility", "x", opts) == [
        "utility",
        "--variant=x",
        "--split=final",
        "--generate-only",
        "--device=cuda:0",
        "--dtype=fp16",
        "--batch-size=8",
    ]
    assert final.stage_argv("belebele", "x", opts)[3] == f"--langs={','.join(LANGS)}"
    assert final.stage_argv("heldout", "x", opts) == [
        "heldout-loss",
        "--variant=x",
        "--split=final",
        "--device=cuda:0",
        "--dtype=fp16",
        "--batch-size=8",
    ]


def test_run_all_goes_in_order_and_lists_what_it_could_not_run():
    seen = []
    rc, summary = final.run_all(
        final.Options(),
        selection=SELECTION,
        tags=TAG,
        recipe=RECIPE,
        devices=1,
        stage_fn=lambda s, n, o: seen.append(n) or 0,
    )
    assert rc == 3
    # not registered yet: the third CAST seed, and the candidate that was not picked
    assert list(summary["not_run"]) == [CAST.format(3), *(OTHER.format(s) for s in (1, 2, 3))]
    assert list(dict.fromkeys(seen))[:7] == [
        "stock",
        *(A.format(s) for s in (1, 2, 3)),
        *(RESTA.format(s) for s in (1, 2, 3)),
    ]
    rc, summary = final.run_all(
        final.Options(), selection=SELECTION, tags=[], devices=1, stage_fn=lambda s, n, o: 0
    )
    assert summary["results"] == {} and len(summary["not_run"]) == 16


def made(name, *, but=()):
    """Empty generations files for every final safety cell of ``name`` except ``but``."""
    from lift.models.generate import generations_path

    for set_name, lang in final.safety_cells():
        if (set_name, lang) not in but:
            path = generations_path(name, set_name, lang, "final")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")


def test_the_laptop_half_runs_the_second_judge_sample_and_utility(monkeypatch):
    calls = []
    made("stock")
    made(A.format(1))
    assert final.judge_only(["stock", A.format(1)], call=lambda argv: calls.append(list(argv)) or 0) == 0
    assert calls[0] == [
        "judge",
        "--variant=stock",
        "--split=final",
        "--judge=second",
        "--sample=0.10",
        "--sets",
        "harmful",
        "easy_benign",
        "hard_benign",
        "--langs",
        *LANGS,
    ]
    assert calls[1][-4:] == ["--sets", "redteam", "--langs", "hi"]
    assert ["utility", f"--variant={A.format(1)}", "--split=final", "--judge-only"] in calls
    assert not any(c[0] == "utility" and "--variant=stock" in c for c in calls)
    listed = validation_dir() / "keys" / "second_judge" / f"{A.format(1)}_final.txt"
    listed.parent.mkdir(parents=True)
    listed.write_text("k\n", encoding="utf-8")
    assert final.second_judge_argv(A.format(1)) == [
        ["judge", f"--variant={A.format(1)}", "--split=final", "--judge=second", f"--keys={listed}"]
    ]


def test_the_second_judge_lines_leave_out_a_cell_with_no_generations_and_say_so(capsys):
    """R1: `lift judge` over named cells stops on a missing generations file, and a final cell short of its
    frozen split (easy_benign zh until its reader's verdict) never gets one. The sample lines name only
    the cells that exist, the others are listed, and the laptop half exits 3 instead of 0."""
    name = A.format(1)
    made(name, but={("easy_benign", "zh"), ("redteam", "hi")})
    lines, absent = final.second_judge_plan(name)
    assert absent == ["easy_benign_zh_final", "redteam_hi_final"]
    tails = [line[line.index("--sets") :] for line in lines]
    assert tails == [
        ["--sets", "harmful", "hard_benign", "--langs", *LANGS],
        ["--sets", "easy_benign", "--langs", *(x for x in LANGS if x != "zh")],
    ]
    assert all(
        line[:5] == ["judge", f"--variant={name}", "--split=final", "--judge=second", "--sample=0.10"]
        for line in lines
    )
    calls = []
    assert final.judge_only([name], call=lambda argv: calls.append(list(argv)) or 0) == 3
    assert [c for c in calls if c[0] == "judge"] == lines
    out = capsys.readouterr().out
    assert f"{name}: no generations file for easy_benign_zh_final, redteam_hi_final" in out
    assert f"`lift final --variant {name}` (GPU)" in out and "reader's verdict" in out
    # Nothing made at all: no judge line, only the utility one, and every cell is listed.
    lines, absent = final.second_judge_plan("stock")
    assert lines == [] and len(absent) == len(final.safety_cells())


# ------------------------------------------------------------------------------------- back-translation


class FakeJudge:
    judge_id = "fake-judge"

    def __init__(self):
        self.items = []

    def judge(self, items):
        self.items += list(items)
        return [JudgeOutcome(judge_label="refusal", native={}, raw="") for _ in items]


def test_backtranslation_judges_the_same_sampled_pids_and_resumes(fake_prompt_files, monkeypatch):
    pids = [f"prh-test-{i:04d}" for i in range(10)]
    for set_name in final.SAFETY:
        for lang in LANGS:
            fake_prompt_files[(set_name, lang)] = [
                make_prompt(p, lang, set_name, "final", text=f"text {p}").model_copy(
                    update={"text_en": f"en {p}"}
                )
                for p in pids
            ]
    records = {}
    for set_name in final.SAFETY:
        for lang in ("ar", "th"):
            gens = [
                GenerationRecord(
                    key=sha256_text(f"{set_name}{lang}{p}"),
                    variant="stock",
                    pid=p,
                    lang=lang,
                    set=set_name,
                    split="final",
                    prompt_sha="x",
                    output="" if i == 0 else f"out {p}",
                    n_prompt_tokens=1,
                    n_new_tokens=1,
                    hit_cap=False,
                    seconds=0.1,
                )
                for i, p in enumerate(pids)
            ]
            write_jsonl(generations_path("stock", set_name, lang, "final"), [g.model_dump() for g in gens])
            flags = UnclearFlags(too_short=False, repetitive=False, garbled=False, wrong_lang=False)
            records[(set_name, lang)] = [
                JudgementRecord(
                    key=g.key,
                    variant="stock",
                    pid=g.pid,
                    lang=lang,
                    set=set_name,
                    split="final",
                    judge="j",
                    judge_label="refusal",
                    label="refusal",
                    flags=flags,
                )
                for g in gens
            ]

    def load_records(variant, set_name, lang, split, dtype=None):
        if variant != "stock":
            raise ValueError(f"{variant}: no judgements")
        return records[(set_name, lang)]

    monkeypatch.setattr("lift.judge.pipeline.load_records", load_records)
    judge = FakeJudge()
    translated = []

    def translator(texts, lang):
        translated.extend(texts)
        return [f"english {t}" for t in texts]

    assert final.backtranslation("stock", final.Options(), judge=judge, translator=translator) == 0
    sample = final.bt_pids("harmful")
    assert len(sample) == 2  # 20% of 10 pids, the same in each language
    rows = final.read_bt("stock", "harmful", "ar")
    assert {r["pid"] for r in rows} == sample
    blank = [r for r in rows if r["skipped"]]
    assert all(r["skipped"] == "blank output" for r in blank)  # counted, never dropped
    assert all(r["judge_label"] == "refusal" and r["judge"] == "fake-judge" for r in rows if not r["skipped"])
    assert all(item.prompt.startswith("en ") and item.prompt_lang == "en" for item in judge.items)
    n = len(judge.items)
    assert final.backtranslation("stock", final.Options(), judge=judge, translator=translator) == 0
    assert len(judge.items) == n  # resumed: nothing judged twice
    assert final.backtranslation(A.format(1), final.Options(), judge=judge, translator=translator) == 3
