"""P07's lift final without a model: the final-split guard (a frozen variant and a recipe-frozen-* tag, in
lift final and in every command that makes final outputs), the 13-variant order, the stages, resume and
GPU-hours manifest, the laptop half's lines and the ar/th back-translation sample on synthetic records.
No torch."""

from __future__ import annotations

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
TAG = ["recipe-frozen-20261107"]
SELECTION = {
    "status": "decided",
    "adapter": A.format(1),
    "overall": {"winner": RESTA.format(1), "verdict": "repair"},
    "frozen": {"wiseft": WINNER.format(1), "resta": RESTA.format(1), "cast": CAST.format(1)},
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
    assert final.check_final(name, selection=SELECTION, tags=TAG) == TAG[0]


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
        final.check_final(name, selection=selection, tags=TAG, use_file=False)


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
    assert "ls-remote" in argv and "secret-token" not in " ".join(argv)  # base64 in a header only
    assert any(a.startswith("http.https://github.com/.extraheader=AUTHORIZATION: basic ") for a in argv)
    assert "GIT_TRACE" not in env
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
    write_selection()
    assert cli.main(["final", "--variant", "wiseft-l0.25-adapter-aya-std-s1"]) == 2
    assert "not a frozen variant" in capsys.readouterr().err
    assert not manifests_dir().exists() or not list(manifests_dir().glob("final_variant-*.json"))


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
    ]
    assert len(plan) == 13 and [p.role for p in plan[4:7]] == ["winner"] * 3
    with pytest.raises(final.Refused):
        final.core_plan({"status": "incomplete"})


def test_run_variant_runs_each_stage_resumes_and_records_gpu_hours():
    calls = []

    def stage_fn(stage, name, opts):
        calls.append((stage, name))
        return 3 if stage == "belebele" else 0

    opts = final.Options()
    results = final.run_variant(A.format(1), opts, stage_fn=stage_fn, devices=2, tag=TAG[0])
    assert results == {"safety": 0, "utility": 0, "belebele": 3, "backtranslation": 0}
    assert final.exit_code({"x": results}) == 3
    calls.clear()
    assert final.run_variant(A.format(1), opts, stage_fn=lambda s, n, o: calls.append(s) or 0, devices=2) == {
        "safety": 0,
        "utility": 0,
        "belebele": 0,
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


def test_run_all_goes_in_order_and_lists_what_it_could_not_run():
    seen = []
    rc, summary = final.run_all(
        final.Options(),
        selection=SELECTION,
        tags=TAG,
        devices=1,
        stage_fn=lambda s, n, o: seen.append(n) or 0,
    )
    assert rc == 3
    assert list(summary["not_run"]) == [CAST.format(3)]  # not registered yet
    assert list(dict.fromkeys(seen))[:7] == [
        "stock",
        *(A.format(s) for s in (1, 2, 3)),
        *(RESTA.format(s) for s in (1, 2, 3)),
    ]
    rc, summary = final.run_all(
        final.Options(), selection=SELECTION, tags=[], devices=1, stage_fn=lambda s, n, o: 0
    )
    assert summary["results"] == {} and len(summary["not_run"]) == 13


def test_the_laptop_half_runs_the_second_judge_sample_and_utility(monkeypatch):
    calls = []
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
