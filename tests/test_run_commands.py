import json
import logging
import re
import sys

import pytest

from lift.cli import build_parser, main
from lift.commands import run as run_cmd
from lift.io import sha256_text, write_jsonl
from lift.paths import generations_dir, judgements_dir, metrics_dir, output_dir

GUARD_SHA = "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
RECORD = b'{"metadata": {"lift": {"refusal_rate": 1.0}}}\n'      # an AuditKIT record's placeholder


def write_generation(set_name, lang, split, *, variant="stock", dtype="bf16"):
    """One generations file holding one record (pid "{set[:3]}-0001", prompt text "t", no real output)
    made in ``dtype``; returns the record's key."""
    from lift.models.generate import generations_path

    key = sha256_text(f"{variant}:{set_name}:{lang}:{split}")
    write_jsonl(generations_path(variant, set_name, lang, split), [{
        "key": key, "variant": variant, "pid": f"{set_name[:3]}-0001", "lang": lang, "set": set_name,
        "split": split, "prompt_sha": sha256_text("t"), "output": "o", "n_prompt_tokens": 1,
        "n_new_tokens": 1, "hit_cap": False, "seconds": 0.1, "meta": {"dtype": dtype}}])
    return key


def write_judgements(judge_dir, set_name, lang, split, *, variant="stock", n=2):
    """A judgements file of ``n`` placeholder records (only lift restart reads it here)."""
    path = judgements_dir() / judge_dir / variant / f"{set_name}_{lang}_{split}.jsonl"
    write_jsonl(path, [{"key": sha256_text(f"{judge_dir}:{variant}:{set_name}:{lang}:{split}:{i}"),
                        "pid": f"{set_name[:3]}-{i:04d}", "label": "refusal"} for i in range(n)])
    return path


def write_record(judge_dir, name, *, variant="stock"):
    """An AuditKIT record file named ``name`` (outputs/metrics/auditkit/{judge_dir}/{variant}/{name})."""
    path = metrics_dir() / "auditkit" / judge_dir / variant / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(RECORD)
    return path


def pin(judge=False):
    """Pin models.global (and, with ``judge``, the primary judge) in this test's configs/project.yaml."""
    from lift import config

    project = config.load("project")
    project["models"]["global"]["revision"] = "b" * 40
    if judge:
        project["judges"]["safety_primary"]["revision"] = GUARD_SHA


class StubGuard:
    """Stands in for lift.judge.qwen3guard.Qwen3GuardJudge (its result contract) with placeholder
    verdicts. ``die_on_call`` raises like a CUDA out-of-memory error in the middle of a file."""

    model_id = "Qwen/Qwen3Guard-Gen-4B"
    revision = GUARD_SHA
    judge_id = f"{model_id}@{revision}"

    def __init__(self, die_on_call=None):
        self.die_on_call = die_on_call
        self.calls = 0

    def judge(self, prompts, responses):
        self.calls += 1
        if self.calls == self.die_on_call:
            raise RuntimeError("judge died (simulated out of memory)")
        return [{"safety": "Safe", "categories": [], "refusal": "Yes", "consistent": True,
                 "raw": "Safety: Safe\nCategories: None\nRefusal: Yes", "error": None} for _ in prompts]


def guard(die_on_call=None, dtype="bf16"):
    """The real Qwen3Guard adapter (it writes its stack into every judgement) over a StubGuard."""
    from lift.judge.pipeline import Qwen3GuardAdapter

    return Qwen3GuardAdapter(StubGuard(die_on_call), strict=True, device="cpu", dtype=dtype)


class FakeLID:
    def top(self, text, k=3):
        return [("eng_Latn", 0.95)]


def judged(variant, set_name, lang, split="dev"):
    """How many judgements the primary judge's file of a cell holds (0 when there is none)."""
    from lift.judge.pipeline import judgements_path, load_judgements

    if not judgements_path("qwen3guard", variant, set_name, lang, split).exists():
        return 0
    return len(load_judgements(variant, set_name, lang, split))


def judge_stacks(variant, set_name, lang, split="dev"):
    from lift.judge.pipeline import load_judgements

    records = load_judgements(variant, set_name, lang, split)
    return {json.dumps(r.native.get("stack"), sort_keys=True) for r in records}


def moved_stack(monkeypatch, **changes):
    """The next session's runtime (generator and judge alike): a newer transformers or torch."""
    from lift.models import generate as gen

    real = gen.stack
    monkeypatch.setattr(gen, "stack", lambda device: {**real(device), **changes})


def offline_runs_repo(monkeypatch):
    """lift sync against tests/test_sync.py's FakeHub: the Hub offline, a dummy token, a test runs repo."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hf_constants, "HF_DEBUG", False)
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_not_a_real_token")
    monkeypatch.setenv("LIFT_RUNS_REPO", "tester/lift-runs")


def short_runs(monkeypatch, *, chunk=2, new_tokens=6):
    """Generation capped at a few tokens, and judging in small chunks so a kill can land mid-file."""
    from lift.judge import pipeline
    from lift.models import generate as gen

    real_generate, real_judge_file = gen.generate, pipeline.judge_file
    monkeypatch.setattr(gen, "generate",
                        lambda *a, **kw: real_generate(*a, **{**kw, "max_new_tokens": new_tokens}))
    monkeypatch.setattr(pipeline, "judge_file",
                        lambda *a, **kw: real_judge_file(*a, **{**kw, "batch_size": chunk}))


@pytest.mark.parametrize("command", ["smoke", "generate", "judge", "run-dev", "sync", "remote", "restart"])
def test_help_works_without_a_gpu_stack(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_names_accept_commas_and_spaces_and_reject_unknowns():
    assert run_cmd._names(["hi,en", "zh"], ["en", "hi", "zh"], ["en"]) == ["hi", "en", "zh"]
    assert run_cmd._names(None, ["en", "hi"], ["en", "hi"]) == ["en", "hi"]
    assert run_cmd._names(["hi,"], ["en", "hi"], ["en"]) == ["hi"]
    with pytest.raises(SystemExit):
        run_cmd._names(["xx"], ["en"], ["en"])
    for nothing in ([""], [","], ["", ","]):          # R2-15: a flag given that names nothing ran no cell
        with pytest.raises(SystemExit, match="names nothing"):
            run_cmd._names(nothing, ["en", "hi"], ["en", "hi"])


LISTS = {"generate": ["generate", "--variant", "stock", "--split", "dev", "--device", "cpu"],
         "judge": ["judge", "--variant", "stock", "--split", "dev", "--device", "cpu"],
         "run-dev": ["run-dev", "--variant", "stock", "--device", "cpu", "--judge-device", "cpu"],
         "restart": ["restart", "--variant", "stock", "--split", "dev"]}


@pytest.mark.parametrize("command", list(LISTS))
def test_a_repeated_sets_or_langs_flag_adds_its_values(command):
    # R2-15: --sets and --langs kept only the last of repeated flags (argparse's store action), so
    # `--sets harmful --sets easy_benign` ran easy_benign alone and exited 0, listing nothing as left out
    args = build_parser().parse_args([*LISTS[command], "--sets", "harmful", "--sets", "easy_benign",
                                      "--langs", "hi", "en", "--langs", "zh"])
    assert args.sets == ["harmful", "easy_benign"] and args.langs == ["hi", "en", "zh"]


@pytest.mark.parametrize("command", list(LISTS))
@pytest.mark.parametrize("names", [["--sets", "", "--langs", "hi"], ["--sets", "harmful", "--langs", ""],
                                   ["--sets", ",", "--langs", "hi"]],
                         ids=["sets-empty", "langs-empty", "comma"])
def test_a_sets_or_langs_value_that_names_nothing_is_refused(command, names, monkeypatch):
    # R2-15: `--sets ''` (or `--sets ,`) resolved to no cell, so the line ran nothing and exited 0 ("done")
    pin(judge=True)
    write_generation("harmful", "hi", "dev")                 # what lift judge checks before its judge loads
    for target in ("lift.models.variants.build_variant", "lift.judge.pipeline.make_judge",
                   "lift.models.generate.restart_cell"):
        monkeypatch.setattr(target, lambda *a, **kw: pytest.fail("the line went on"))
    with pytest.raises(SystemExit, match="names nothing"):
        main([*LISTS[command], *names])
    assert not (output_dir() / ".aside").exists()


@pytest.mark.parametrize("keys", ["", "  "])
def test_an_empty_keys_value_is_refused_at_parse(keys, monkeypatch, capsys):
    # R2-15: `lift judge --judge second --keys ''` read the empty path as no key list, so the second
    # (frontier) judge judged every generation of the cells instead of P04's sample, and the line exited 0
    write_generation("harmful", "hi", "dev")
    monkeypatch.setattr("lift.judge.pipeline.make_judge", lambda *a, **kw: pytest.fail("a judge loaded"))
    with pytest.raises(SystemExit) as exc:
        main(["judge", "--variant", "stock", "--split", "dev", "--judge", "second", "--sets", "harmful",
              "--langs", "hi", "--keys", keys])
    assert exc.value.code == 2
    assert "argument --keys: needs the path of P04's key list" in capsys.readouterr().err


def test_primary_judge_refuses_subsets():
    parser = build_parser()
    args = parser.parse_args(["judge", "--variant", "stock", "--split", "dev", "--sample", "0.1"])
    with pytest.raises(SystemExit, match="second judge"):
        args.func(args)
    args = parser.parse_args(["judge", "--variant", "stock", "--split", "dev", "--judge", "second",
                              "--sample", "0.1", "--keys", "k.txt"])
    with pytest.raises(SystemExit, match="not both"):
        args.func(args)


@pytest.mark.parametrize("command", ["generate", "run-dev"])
@pytest.mark.parametrize("flag, value", [("--batch-size", "-1"), ("--batch-size", "0"), ("--limit", "-3"),
                                         ("--limit", "0")])
def test_a_batch_size_or_limit_below_one_is_refused_before_a_model_loads(command, flag, value, monkeypatch,
                                                                         capsys):
    # R2-14: both flags took any int. A negative --batch-size made generate() decode empty batches for ever
    # (the job held the GPU until its deadline), --limit 0 meant no limit and --limit -N cut each cell's last
    # N prompts, all with exit 0. A value below 1 is refused at parse now (exit 2), before a model loads.
    pin(judge=True)
    monkeypatch.setattr("lift.models.variants.build_variant", lambda *a, **kw: pytest.fail("a model loaded"))
    argv = {"generate": ["generate", "--variant", "stock", "--split", "dev"],
            "run-dev": ["run-dev", "--variant", "stock"]}[command]
    with pytest.raises(SystemExit) as exc:
        main([*argv, "--device", "cpu", flag, value])
    assert exc.value.code == 2
    assert f"argument {flag}: must be at least 1, got {value}" in capsys.readouterr().err
    assert getattr(build_parser().parse_args([*argv, flag, "1"]), flag[2:].replace("-", "_")) == 1


def test_study_commands_refuse_unpinned_revisions():
    # configs/project.yaml still says revision: main until Gate 0's shas are pinned.
    parser = build_parser()
    for argv in (["generate", "--variant", "stock", "--split", "dev"], ["run-dev", "--variant", "stock"]):
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit, match="unpinned revisions"):
            args.func(args)


def test_the_pin_refusal_says_to_commit_and_push_the_pin():
    # The GPU job runs a ref (lift remote --ref, Kaggle BRANCH), not the laptop's working tree.
    with pytest.raises(SystemExit) as exc:
        run_cmd.require_pinned("models.global")
    message = str(exc.value)
    assert "decisions.md" in message and "commit" in message and "push" in message and "--ref" in message


@pytest.mark.parametrize("argv", [
    ["generate", "--variant", "stock", "--split", "dev"],
    ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "hi"],
    ["run-dev", "--variant", "stock"],
])
def test_the_pin_refusal_comes_before_torch(argv, monkeypatch):
    # A laptop install without the GPU stack (.[dev]) gets the pin message, not ModuleNotFoundError.
    write_generation("harmful", "hi", "dev")              # whatever lift judge checks before the pin
    monkeypatch.setitem(sys.modules, "torch", None)       # `import torch` raises ImportError
    with pytest.raises(SystemExit, match="unpinned revisions"):
        main(argv)


def test_require_pinned_accepts_a_full_sha(monkeypatch):
    from lift import config

    real = config.get
    monkeypatch.setattr(config, "get", lambda key, *a, **k: "a" * 40 if key.endswith(".revision")
                        else real(key, *a, **k))
    run_cmd.require_pinned("models.global", "judges.safety_primary")


def test_run_dev_pins_the_primary_judge_only_when_it_is_local(monkeypatch):
    # P04's decision rule can make the frontier judge primary: the safety_primary and safety_second
    # blocks are swapped as P04 prints them, so the primary block has a provider and model, no revision.
    from lift import config
    from lift.judge.pipeline import judge_dir

    project = config.load("project")
    project["models"]["global"]["revision"] = "b" * 40
    built = []
    monkeypatch.setattr("lift.models.variants.build_variant", lambda *a, **k: built.append(a) or object())
    monkeypatch.setattr("lift.models.variants.get_spec", lambda name: name)
    monkeypatch.setattr(run_cmd, "run_cells", lambda *a, **k: [])
    argv = ["run-dev", "--variant", "stock", "--device", "cpu", "--judge-device", "cpu"]
    with pytest.raises(SystemExit, match=r"judges\.safety_primary\.revision"):     # Qwen3Guard: pinned
        main(argv)
    judges = project["judges"]
    qwen = dict(judges["safety_primary"], revision="c" * 40)
    judges["safety_primary"] = dict(judges["safety_second"], provider="openai", model="placeholder-model")
    judges["safety_second"] = qwen
    assert "revision" not in judges["safety_primary"] and judge_dir("primary") == "frontier"
    assert main(argv) == 0 and built


def test_judge_keys_listed_in_no_generations_file_fail_loudly(monkeypatch):
    key = write_generation("harmful", "hi", "dev")
    keys = generations_dir().parent / "keys.txt"
    keys.write_text(key + "\n" + "b" * 64 + "\n", encoding="utf-8")
    monkeypatch.setattr("lift.judge.pipeline.judge_dir", lambda role: "frontier")
    args = build_parser().parse_args(["judge", "--variant", "stock", "--split", "dev", "--judge", "second",
                                      "--sets", "harmful", "--langs", "hi", "--keys", str(keys)])
    with pytest.raises(SystemExit, match="1 listed keys are in none"):
        args.func(args)


@pytest.mark.parametrize("explicit_sets", [False, True])
def test_judge_with_keys_covers_every_file_holding_one(monkeypatch, caplog, explicit_sets):
    # P04's key list for a variant and split holds red-team keys too. Without --sets, every file holding a
    # listed key is judged (the default safety sets would silently skip the red-team keys). With an
    # explicit --sets, as in P04's one line per group over one list, the keys outside are logged as
    # counts and left to their own line, and the command still succeeds.
    from lift.judge.pipeline import JudgeOutcome

    sets = ("harmful", "easy_benign", "hard_benign", "redteam")
    keys = {s: write_generation(s, "hi", "final") for s in sets}
    listed = generations_dir().parent / "stock_final.txt"
    listed.write_text(keys["harmful"] + "\n" + keys["redteam"] + "\n", encoding="utf-8")
    judged = []

    class Judge:
        judge_id = "openai/placeholder-model@x"
        short = "frontier"

        def judge(self, items):
            judged.extend(i.key for i in items)
            return [JudgeOutcome("benign", {"label": "BENIGN"}, "") for _ in items]

    class LID:
        def top(self, text, k=3):
            return [("hin_Deva", 0.99)]

    monkeypatch.setattr("lift.judge.pipeline.judge_dir", lambda role: "frontier")
    monkeypatch.setattr("lift.judge.pipeline.make_judge", lambda *a, **k: Judge())
    monkeypatch.setattr("lift.prompts.texts_by_pid", lambda s, lang: {f"{s[:3]}-0001": "t"})
    monkeypatch.setattr("lift.lang.LangID", LID)
    argv = ["judge", "--variant", "stock", "--split", "final", "--judge", "second", "--langs", "hi",
            "--keys", str(listed)]
    rc = main(argv + ["--sets", "harmful"] * explicit_sets)
    assert rc == 0
    if explicit_sets:
        assert judged == [keys["harmful"]]
        assert "1 listed keys" in caplog.text and "'redteam_hi': 1" in caplog.text
        assert keys["redteam"][:12] not in caplog.text    # counts by set and language only
    else:
        assert judged == [keys["harmful"], keys["redteam"]]
        assert "outside --sets/--langs" not in caplog.text


def test_a_refused_auditkit_record_is_reported_not_fatal(monkeypatch, caplog):
    # Contract G: "done, but some cells are listed" is exit 3, never 1 (an error, which ends a GPU job).
    def refuse(*a, **kw):
        raise ValueError("3 generations are not in bf16")

    monkeypatch.setattr("lift.audit.record_safety", refuse)
    monkeypatch.setattr(run_cmd, "_complete", lambda *a: True)
    refused: dict[str, str] = {}
    run_cmd._record("stock", "harmful", "hi", "dev", dtype="bf16", refused=refused)
    assert list(refused) == ["harmful_hi_dev"] and "not in bf16" in caplog.text
    caplog.clear()
    assert run_cmd._refused_exit(refused) == 3
    assert "harmful_hi_dev: 3 generations are not in bf16" in caplog.text     # listed with its remedy
    assert run_cmd._refused_exit({}) == 0 and run_cmd._refused_exit({}, {}) == 0
    assert run_cmd._refused_exit({}, {"stale": {"harmful_hi_dev": "lift restart ..."}}) == 3
    assert run_cmd._refused_exit({}, {"precision": {"harmful_hi_dev": "made in fp16"}}) == 3


# --- lift restart ---------------------------------------------------------------------------------------


RESTART = ["restart", "--variant", "stock", "--split", "dev"]


def restart_tree():
    """The cell harmful_hi_dev (generations, and judgements by both judges) and the files lift restart
    must leave alone: another language, set, split and variant. Returns {path: bytes} of every file."""
    cells = (("stock", "harmful", "hi", "dev"), ("stock", "harmful", "en", "dev"),
             ("stock", "easy_benign", "hi", "dev"), ("stock", "harmful", "hi", "final"),
             ("other", "harmful", "hi", "dev"))
    for variant, set_name, lang, split in cells:
        write_generation(set_name, lang, split, variant=variant)
        for judge in ("qwen3guard", "frontier"):
            write_judgements(judge, set_name, lang, split, variant=variant)
    return {p: p.read_bytes() for p in output_dir().rglob("*.jsonl")}


@pytest.mark.parametrize("judge", [None, "primary", "second"])
def test_restart_sets_exactly_the_cells_files_aside_and_leaves_them_empty(judge, monkeypatch, caplog):
    from lift.judge.pipeline import judge_dir
    from lift.models.generate import generations_path

    monkeypatch.setitem(sys.modules, "torch", None)          # a laptop without the GPU stack runs it too
    caplog.set_level(logging.INFO, logger="lift.run")
    before = restart_tree()
    name = "harmful_hi_dev.jsonl"
    if judge is None:                                        # the cell: generations and every judgements file
        targets = [generations_path("stock", "harmful", "hi", "dev"),
                   *(judgements_dir() / j / "stock" / name for j in ("frontier", "qwen3guard"))]
    else:
        targets = [judgements_dir() / judge_dir(judge) / "stock" / name]
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi", *(["--judge", judge] if judge else [])]) == 0
    stamp, = (output_dir() / ".aside").iterdir()
    assert sorted(p for p in stamp.rglob("*") if p.is_file()) == sorted(
        stamp / t.relative_to(output_dir()) for t in targets)
    for path, data in before.items():
        if path in targets:                                  # emptied, never removed: sync never deletes
            assert path.read_bytes() == b"" and (stamp / path.relative_to(output_dir())).read_bytes() == data
        else:
            assert path.read_bytes() == data
    assert caplog.text.count("set aside in") == len(targets)       # one line per file, with its count
    assert "har-0001" not in caplog.text and "har-0000" not in caplog.text


def test_restart_says_the_next_push_may_report_an_emptied_conflict_instead(caplog):
    # W2-04 (R1-08): the next lift sync push replaces the runs repo's copies only when the Hub's copy has not
    # changed since this machine's last sync; otherwise the push reports the emptied file as a conflict (a
    # merge would fill it again), which lift sync push --force settles. The help and the log said it always
    # replaced them.
    caplog.set_level(logging.INFO, logger="lift.run")
    restart_tree()
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi"]) == 0
    [done] = [r.getMessage() for r in caplog.records if "the next lift sync push" in r.getMessage()]
    for text in (done, " ".join(run_cmd.RESTART_HELP.split())):
        assert "changed since this machine's last sync" in text and "lift sync push --force" in text, text
        assert "emptied-file conflict" in text, text


def test_skipped_judges_cells_name_the_restart_line_when_only_the_labelling_rules_differ():
    # W2-04 (R1-12 follow-up): a judge id names its labelling rules too, and judgements made under other rules
    # can never be finished with the file's judge, as no current judge applies them: only the role's lift
    # restart --judge line remedies such a cell
    head = " ".join(run_cmd.SKIPPED["judges"].split())
    assert "labelling rules" in head and "--judge" in head and "the file's judge" in head
    assert "its id names its precision and Controversial mode" not in head


def test_restart_skips_files_that_are_not_here_and_a_second_run_finds_them_empty(caplog):
    from lift.models.generate import generations_path

    caplog.set_level(logging.INFO, logger="lift.run")
    frontier = write_judgements("frontier", "redteam", "hi", "dev")        # no generations file
    assert main([*RESTART, "--sets", "redteam", "--langs", "hi"]) == 0
    missing = generations_path("stock", "redteam", "hi", "dev")
    assert frontier.read_bytes() == b"" and not missing.exists() and f"{missing}: not here" in caplog.text
    qwen = judgements_dir() / "qwen3guard" / "stock" / "redteam_hi_dev.jsonl"
    assert main([*RESTART, "--sets", "redteam", "--langs", "hi", "--judge", "primary"]) == 0
    assert not qwen.exists() and f"{qwen}: not here" in caplog.text
    caplog.clear()
    assert main([*RESTART, "--sets", "redteam", "--langs", "hi"]) == 0
    assert f"{frontier}: already empty" in caplog.text
    assert len(list((output_dir() / ".aside").iterdir())) == 1             # nothing new set aside


def test_judge_refuses_a_restarted_cell_before_loading_the_judge(monkeypatch):
    # lift restart leaves the cell's generations file empty: lift judge needs lift generate first, and says
    # so before it loads a judge.
    from lift.judge import pipeline

    write_generation("harmful", "hi", "dev")
    write_generation("harmful", "en", "dev")
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi"]) == 0
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: pytest.fail("the judge was loaded"))
    argv = ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "hi",
            "--judge", "second"]
    with pytest.raises(SystemExit, match="1 generations files are missing or empty.*harmful_hi_dev"):
        main(argv)


def test_restart_refuses_an_unknown_variant_and_needs_sets_and_langs():
    with pytest.raises(SystemExit, match="not in configs/variants.yaml"):
        main(["restart", "--variant", "stokc", "--split", "dev", "--sets", "harmful", "--langs", "hi"])
    with pytest.raises(SystemExit) as exc:
        main([*RESTART, "--sets", "harmful"])
    assert exc.value.code == 2
    assert not (output_dir() / ".aside").exists()


CELL_RECORDS = ("harmful_hi_dev.json", "harmful_hi_dev-fp16.json", "harmful_hi_dev.full.json",
                "harmful_hi_dev-fp16.full.json", "harmful_hi_dev-bf16.json")


@pytest.mark.parametrize("judge", [None, "primary", "second"])
def test_restart_retires_the_cells_auditkit_records(judge, caplog):
    # W11/W19: a restarted cell's records would outlive its outputs on the runs repo (the push that sends
    # the emptied outputs sees them unchanged), describing outputs that were set aside, and a record of the
    # other precision is never rewritten. lift restart copies and empties them: an empty record is retired.
    from lift.judge.pipeline import judge_dir

    caplog.set_level(logging.INFO, logger="lift.run")
    write_generation("harmful", "hi", "dev")
    cell = {d: [write_record(d, name) for name in CELL_RECORDS] for d in ("qwen3guard", "frontier")}
    others = [write_record(d, name) for d in ("qwen3guard", "frontier")
              for name in ("harmful_hi_final.json", "harmful_en_dev.json", "easy_benign_hi_dev-fp16.json",
                           "harmful_hi_dev-draft.json")]
    others.append(write_record("qwen3guard", "harmful_hi_dev.json", variant="other"))
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi", *(["--judge", judge] if judge else [])]) == 0
    stamp, = (output_dir() / ".aside").iterdir()
    retired = ("qwen3guard", "frontier") if judge is None else (judge_dir(judge),)
    for judge_name, paths in cell.items():
        for path in paths:
            copy = stamp / path.relative_to(output_dir())
            if judge_name in retired:                        # emptied, never removed: the next push sends it
                assert path.read_bytes() == b"" and copy.read_bytes() == RECORD
            else:
                assert path.read_bytes() == RECORD and not copy.exists()
    assert all(p.read_bytes() == RECORD for p in others)
    assert caplog.text.count("AuditKIT record retired") == len(CELL_RECORDS) * len(retired)
    caplog.clear()
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi", *(["--judge", judge] if judge else [])]) == 0
    assert "AuditKIT record retired" not in caplog.text                    # already retired: nothing new
    assert len(list((output_dir() / ".aside").iterdir())) == 1


def test_restart_refuses_a_cell_made_in_another_precision_unless_forced(caplog):
    # W3: a restart would discard a cell of another precision (the T4 backup's fp16 line over Colab's bf16
    # cell), and the next push would replace the runs repo's copy. Refused, with nothing moved for any listed
    # cell, unless --force.
    caplog.set_level(logging.INFO, logger="lift.run")
    write_generation("harmful", "hi", "dev", dtype="bf16")
    write_generation("easy_benign", "hi", "dev", dtype="fp16")
    write_judgements("qwen3guard", "harmful", "hi", "dev")
    write_record("qwen3guard", "harmful_hi_dev.json")
    before = {p: p.read_bytes() for p in output_dir().rglob("*") if p.is_file()}
    both = [*RESTART, "--sets", "harmful", "easy_benign", "--langs", "hi"]
    for argv in ([*both, "--dtype", "fp16"], both, [*both, "--dtype", "fp16", "--judge", "primary"]):
        caplog.clear()
        assert main(argv) == 1
        assert {p: p.read_bytes() for p in output_dir().rglob("*") if p.is_file()} == before
        assert not (output_dir() / ".aside").exists()
        made, run_in = ("bf16", "fp16") if "fp16" in argv else ("fp16", "bf16")
        assert f"made in {made}" in caplog.text and f"not {run_in}" in caplog.text
        assert "would discard" in caplog.text and "--force" in caplog.text
    assert main([*RESTART, "--sets", "easy_benign", "--langs", "hi", "--dtype", "fp16"]) == 0
    caplog.clear()
    assert main([*both, "--dtype", "fp16", "--force"]) == 0
    assert generations_dir().joinpath("stock", "harmful_hi_dev.jsonl").read_bytes() == b""
    assert "--force" in caplog.text and "bf16" in caplog.text           # the discarded precision is named


def repo_layout(monkeypatch, isolated_dirs):
    """A GPU runtime's layout: validation/ (here the test's own) at the root of the repo, so job lines name
    P04's key lists relative to it."""
    monkeypatch.setattr("lift.paths.REPO_ROOT", isolated_dirs.resolve())


def p04_list(*pids, variant="stock", split="dev", cell=("harmful", "en")):
    """P04's second-judge key list of ``variant``'s ``split`` (decisions.md 31), drawn now: the keys the
    cell's generations file holds for ``pids``."""
    from lift.models.generate import generations_path, read_generations
    from lift.paths import validation_dir

    path = validation_dir() / "keys" / "second_judge" / f"{variant}_{split}.txt"
    held = {g.pid: g.key for g in read_generations(generations_path(variant, *cell, split))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(held[pid] + "\n" for pid in pids), encoding="utf-8")
    return path


@pytest.mark.parametrize("swapped", [False, True])
def test_a_whole_cell_restart_names_the_line_that_refills_the_second_judges_file(swapped, monkeypatch,
                                                                                caplog, isolated_dirs):
    # W9: a whole-cell restart also sets aside the second judge's judgements of the cell, and the line that
    # failed (run-dev, or generate and judge) never judges with the second judge. The log names the line
    # that refills it: P04's key list (decisions.md 31), the run's precision when it is not the default.
    # Swapped: P04's decision rule made the frontier judge primary, so the Qwen3Guard file is the second's.
    # X9: a --keys line reads P04's list, which P04 draws again when the cell made again has new keys.
    from lift import config
    from lift.judge.pipeline import judge_dir

    if swapped:
        judges = config.load("project")["judges"]
        judges["safety_primary"], judges["safety_second"] = judges["safety_second"], judges["safety_primary"]
    caplog.set_level(logging.INFO, logger="lift.run")
    write_generation("harmful", "hi", "dev", dtype="fp16")
    for judge in ("qwen3guard", "frontier"):
        write_judgements(judge, "harmful", "hi", "dev")
    second = judgements_dir() / judge_dir("second") / "stock" / "harmful_hi_dev.jsonl"
    repo_layout(monkeypatch, isolated_dirs)
    p04_list("har-0001", cell=("harmful", "hi"))
    refill = ("lift judge --variant stock --split dev --sets harmful --langs hi --judge second "
              "--keys validation/keys/second_judge/stock_dev.txt --dtype fp16")
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi", "--dtype", "fp16"]) == 0
    assert f"  {second}: {refill}\n" in caplog.text
    assert "--judge primary" not in caplog.text and "lift run-dev" in caplog.text
    assert ("only after P04 draws the key list again (lift validate second-judge-sample, then lift sync "
            "push)") in caplog.text
    for role in ("primary", "second"):                     # one role's restart: the failed line refills it
        for judge in ("qwen3guard", "frontier"):
            write_judgements(judge, "harmful", "hi", "dev")
        caplog.clear()
        assert main([*RESTART, "--sets", "harmful", "--langs", "hi", "--dtype", "fp16", "--judge", role]) == 0
        assert caplog.text.count("set aside in") == 1 and "lift judge --variant" not in caplog.text


def test_the_refill_line_quotes_a_key_list_outside_the_repo():
    # LIFT_VALIDATION_DIR moved validation/ (here a tmp dir): the line names the list's own path, quoted.
    import shlex

    from lift.paths import validation_dir

    listed = validation_dir() / "keys" / "second_judge" / "stock_dev.txt"
    listed.parent.mkdir(parents=True)
    listed.write_text("", encoding="utf-8")
    line = run_cmd._judge_line("stock", "harmful", "hi", "dev", role="second", dtype="bf16")
    words = shlex.split(line)
    assert words[-2:] == ["--keys", str(listed)]
    assert "--dtype" not in words                          # generation.dtype is the default
    assert "--keys" not in run_cmd._judge_line("stock", "harmful", "hi", "dev", role="primary", dtype="bf16")


def test_the_second_judges_refill_line_samples_while_p04_has_drawn_no_key_list(caplog):
    # X9: before P04 exists the second judge runs with --sample; a refill line naming a list that does not
    # exist would end in a FileNotFoundError. It names decisions.md 31's --sample 0.10 instead.
    from lift.judge.pipeline import judge_dir

    caplog.set_level(logging.INFO, logger="lift.run")
    line = run_cmd._judge_line("stock", "harmful", "hi", "dev", role="second", dtype="fp16")
    assert line == ("lift judge --variant stock --split dev --sets harmful --langs hi --judge second "
                    "--sample 0.10 --dtype fp16")
    write_generation("harmful", "hi", "dev")
    for judge in ("qwen3guard", "frontier"):
        write_judgements(judge, "harmful", "hi", "dev")
    second = judgements_dir() / judge_dir("second") / "stock" / "harmful_hi_dev.jsonl"
    assert main([*RESTART, "--sets", "harmful", "--langs", "hi"]) == 0
    assert f"  {second}: lift judge --variant stock --split dev --sets harmful --langs hi --judge second " \
           "--sample 0.10\n" in caplog.text
    assert "--keys" not in caplog.text and "P04 draws" not in caplog.text


def test_a_restart_retires_the_runs_repo_copies_of_a_cell_taken_over_in_another_precision(tmp_path,
                                                                                            monkeypatch):
    # W11: the runs repo holds the T4's licensed fp16 cell and its -fp16 record; Colab takes the cell over in
    # bf16. The bf16 run refuses it as another precision, so the operator restarts it with --force; the next
    # push replaces the cell's outputs AND its record with empty files (the record is retired), so no fp16
    # record outlives the outputs it describes.
    offline_runs_repo(monkeypatch)
    from test_sync import GEN, JUD, FakeHub, gen_row, jud_row, lines, machine, record_path

    from lift import sync
    from lift.io import sha256_obj

    rec = record_path(dtype="fp16")
    gens = [gen_row(f"{i:064x}", f"pid-{i}", dtype="fp16") for i in range(2)]
    judged = [jud_row(g["key"], g["pid"], "refusal", output_sha=sha256_text(g["output"])) for g in gens]
    record = {"metadata": {"lift": {"keys_sha256": sha256_obj({g["pid"]: g["key"] for g in gens}),
                                    "labels_sha256": sha256_obj({g["pid"]: "refusal" for g in gens})}}}
    hub = FakeHub({GEN: lines(*gens), JUD: lines(*judged), rec: (json.dumps(record) + "\n").encode()})
    machine(tmp_path / "colab", monkeypatch)
    assert sync.pull(api=hub).ok
    argv = ["restart", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en"]
    assert main(argv) == 1                                  # bf16 by default: it would discard fp16 outputs
    assert main([*argv, "--force"]) == 0
    report = sync.push(api=hub)
    assert report.ok
    assert hub.files[GEN] == hub.files[JUD] == hub.files[rec] == b""


@pytest.mark.torch
@pytest.mark.parametrize("push_first", [True, False])
def test_restart_after_the_pull_lets_the_next_session_generate_the_cell_again(tiny_model_dir, tmp_path,
                                                                              monkeypatch, push_first):
    # A cell half generated on one stack is refused on another. Moving the file aside by hand is undone
    # by the job's own `lift sync pull` (sync never deletes). The refusal's lift restart line, right
    # after the pull, empties it: the next push replaces the runs repo's copy, and the cell is generated
    # again (in the same job, or in the next session after a push).
    offline_runs_repo(monkeypatch)
    from fakes import make_prompt
    from test_sync import GEN, FakeHub, machine

    from lift import sync
    from lift.models import generate as gen
    from lift.models.variants import build_variant, get_spec

    hub = FakeHub()
    variant = build_variant(get_spec("stock"), device="cpu", dtype="bf16", model_id=str(tiny_model_dir))
    prompts = [make_prompt(f"prh-test-{i:04d}", "en", text=f"placeholder {i}") for i in range(4)]

    def cell():
        return gen.generations_path("stock", "harmful", "en", "dev")

    machine(tmp_path / "s1", monkeypatch)                  # session 1 ends with the cell half done
    gen.generate(variant, prompts[:2], out_path=cell(), max_new_tokens=6)
    assert sync.push(api=hub).ok
    moved_stack(monkeypatch, torch="2.99.0+cu130")         # session 2: the runtime's torch moved
    machine(tmp_path / "s2", monkeypatch)
    assert sync.pull(api=hub).ok                           # the job's first line
    with pytest.raises(gen.StaleOutputError) as refused:
        gen.generate(variant, prompts, out_path=cell(), max_new_tokens=6)
    line = "lift restart --variant stock --split dev --sets harmful --langs en"
    assert f"run `{line}` as a job line right after `lift sync pull`" in str(refused.value)
    assert main(line.split()[1:]) == 0
    assert cell().read_bytes() == b""
    if push_first:                                         # e.g. the session ends before the cell is redone
        report = sync.push(api=hub)
        assert report.ok and report.files[GEN] == sync.UPDATE and hub.files[GEN] == b""
        machine(tmp_path / "s3", monkeypatch)
        assert sync.pull(api=hub).ok
    gen.generate(variant, prompts, out_path=cell(), max_new_tokens=6)        # no StaleOutputError
    report = sync.push(api=hub)
    assert report.ok and report.files[GEN] == sync.UPDATE
    assert not [p for p in hub.files if "/.aside/" in p]   # what was set aside stays on its machine
    machine(tmp_path / "laptop", monkeypatch)
    assert sync.pull(api=hub).ok
    got = gen.read_generations(cell())
    assert len(got) == 4 and len({r.meta["stack"] for r in got}) == 1


# --- stale cells: logged, listed with their restart line, the others go on -----------------------------


@pytest.mark.torch
def test_generate_skips_a_stale_cell_and_exits_3(fake_prompt_files, tiny_model_dir, monkeypatch, caplog):
    # Contract G: exit 3 is "done, but some cells were skipped", so a GPU job goes on with its next lines
    # (exit 1, an error, ends it: one stale cell would cost every later line of the session).
    pytest.importorskip("torch")
    from fakes import make_prompt

    from lift.models import generate as gen
    from lift.models import variants

    pin()
    for lang in ("en", "hi"):
        fake_prompt_files[("harmful", lang)] = [make_prompt(f"har-{i}", lang, "harmful", "dev", text=f"q {i}")
                                                for i in range(3)]
    real_build = variants.build_variant
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: real_build(
        spec, device="cpu", dtype=dtype, model_id=str(tiny_model_dir)))
    short_runs(monkeypatch)
    part = gen.generations_path("stock", "harmful", "en", "dev")
    variant = variants.build_variant(variants.get_spec("stock"), device="cpu", dtype="bf16")
    gen.generate(variant, fake_prompt_files[("harmful", "en")][:2], out_path=part)     # a session cut short
    before = part.read_bytes()
    moved_stack(monkeypatch, torch="2.99.0+cu130")
    argv = ["generate", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "hi",
            "--device", "cpu", "--dtype", "bf16"]
    assert main(argv) == 3
    assert part.read_bytes() == before
    assert len(gen.read_generations(gen.generations_path("stock", "harmful", "hi", "dev"))) == 3
    line = "lift restart --variant stock --split dev --sets harmful --langs en"
    assert f"harmful_en_dev: {line}\n" in caplog.text


SHORT_PID = "prb-test-0003"         # the frozen final pid frozen_cell leaves out of easy_benign zh


def frozen_cell(tmp_path, monkeypatch):
    """data/splits/easy_benign.json (in tmp) freezing prb-test-0000..0003 as final, and easy_benign prompt
    files the real lift.data.build.load_prompts reads: en holds every frozen pid, zh all but SHORT_PID, which
    zh's report lists as an open wrong_lang translation failure (it awaits a reader's verdict, as
    prb-test-0240 does on the real data). Placeholder texts."""
    from fakes import make_prompt

    from lift import splits
    from lift.io import sha256_file
    from lift.paths import prompts_dir

    pids = [f"prb-test-{i:04d}" for i in range(4)]
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")
    splits.save_split("easy_benign", dict.fromkeys(pids, "final"), seed=0, sizes={"final": 4})
    for lang in ("en", "zh"):
        path = prompts_dir() / f"easy_benign_{lang}.jsonl"
        write_jsonl(path, [make_prompt(p, lang, "easy_benign", "final", text=f"q {p}").model_dump(mode="json")
                           for p in pids if lang == "en" or p != SHORT_PID])
        failures = [{"pid": SHORT_PID, "rule": "wrong_lang", "detail": "top=jpn_Jpan 0.99",
                     "engine": "google", "resolved": None}] if lang == "zh" else []
        report = {"set": "easy_benign", "lang": lang, "file": path.name, "sha256": sha256_file(path),
                  "translation_failures": failures, "exclusions": []}
        reports = prompts_dir() / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / f"easy_benign_{lang}.json").write_text(json.dumps(report), encoding="utf-8")


def short_remedy():
    """frozen_cell's zh final remedy, as lift.data.build.load_prompts names it."""
    from lift.data import build

    return (f"1 translation failures await a reader's verdict on the failures sheet, e.g. ['{SHORT_PID}']: "
            + build.reader_remedy(["zh"]))


def test_a_short_cell_is_listed_with_its_remedy_whatever_precision_its_file_was_made_in():
    # W2-01: a short cell's remedy is its prompt file's, on the laptop; it discards nothing, so it is listed
    # as short with that remedy even when its generations file was made in another precision than the run's
    from lift.data.build import ShortCellError

    write_generation("easy_benign", "zh", "final", dtype="fp16")
    exc = ShortCellError("easy_benign_zh: short", set_name="easy_benign", lang="zh", missing={"final": ["p"]},
                         remedy="the remedy")
    skipped: dict[str, dict[str, str]] = {}
    run_cmd._skip(skipped, "stock", "easy_benign", "zh", "final", exc, judged=None, dtype="bf16")
    assert skipped == {"short": {"easy_benign_zh_final": "the remedy"}}
    assert run_cmd._listing(skipped) == [
        f"1 cells {run_cmd.SKIPPED['short']}:\n  easy_benign_zh_final: the remedy\n"]
    head = " ".join(run_cmd.SKIPPED["short"].split())
    assert "reader's verdict" in head and "nothing was generated" in head


@pytest.mark.torch
def test_generate_skips_a_short_cell_lists_its_remedy_and_exits_3(tiny_model_dir, tmp_path, monkeypatch,
                                                                  caplog):
    # W2-01: a cell short of its frozen split (easy_benign zh final on the real data, until the zh reader's
    # verdicts are applied) stopped `lift generate` with a traceback, so no later cell of the line ran. It is
    # skipped and listed with its remedy now, and the line exits 3, as for a stale cell.
    pytest.importorskip("torch")
    from lift.models import generate as gen
    from lift.models import variants

    pin()
    frozen_cell(tmp_path, monkeypatch)
    real_build = variants.build_variant
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: real_build(
        spec, device="cpu", dtype=dtype, model_id=str(tiny_model_dir)))
    short_runs(monkeypatch)
    argv = ["generate", "--variant", "stock", "--split", "final", "--sets", "easy_benign", "--langs", "zh",
            "en", "--device", "cpu", "--dtype", "bf16"]
    assert main(argv) == 3
    assert not gen.generations_path("stock", "easy_benign", "zh", "final").exists()
    assert len(gen.read_generations(gen.generations_path("stock", "easy_benign", "en", "final"))) == 4
    assert listing(caplog, "are short of their frozen split") == (
        f"1 cells {run_cmd.SKIPPED['short']}:\n  easy_benign_zh_final: {short_remedy()}\n")
    assert "easy_benign_zh_final: short of its frozen split, skipped" in caplog.text


def frozen_generations(lang, pids, *, variant="stock"):
    """easy_benign ``lang`` final generations of ``pids``, from frozen_cell's prompt texts, as lift generate
    made them before a read of the split refused a cell short of its frozen split (R1-01). Placeholder
    outputs."""
    from lift.models.generate import generations_path

    write_jsonl(generations_path(variant, "easy_benign", lang, "final"), [{
        "key": sha256_text(f"{variant}:{lang}:{pid}"), "variant": variant, "pid": pid, "lang": lang,
        "set": "easy_benign", "split": "final", "prompt_sha": sha256_text(f"q {pid}"), "output": "o",
        "n_prompt_tokens": 1, "n_new_tokens": 1, "hit_cap": False, "seconds": 0.1, "meta": {"dtype": "bf16"}}
        for pid in pids])


def test_judge_refuses_a_short_cells_record_with_its_remedy_and_exits_3(tmp_path, monkeypatch, caplog):
    # W3-02: lift judge reads a cell's prompt texts by pid, so it judges a short cell's generations made
    # before R1-01 (easy_benign zh final on the real data). Its AuditKIT record then loaded the split outside
    # the record's refusal handling: ShortCellError was a traceback after the judgements were written, and no
    # later cell of the line ran. The record is refused with the error's remedy now (and the lines that then
    # make it), the other cells are judged and recorded, and the line exits 3.
    from lift.judge import pipeline
    from lift.judge.pipeline import JudgeOutcome

    class Judge:                                    # an API primary judge: no pin, no GPU, no precision
        judge_id = "openai/placeholder-model@x"
        short = "frontier"

        def judge(self, items):
            return [JudgeOutcome("benign", {"label": "BENIGN"}, "") for _ in items]

    recorded = []

    def record_safety(variant, set_name, lang, split, **kw):
        recorded.append(lang)
        return "a record"

    frozen_cell(tmp_path, monkeypatch)
    pids = [f"prb-test-{i:04d}" for i in range(4)]
    for lang in ("zh", "en"):
        frozen_generations(lang, [p for p in pids if lang == "en" or p != SHORT_PID])
    monkeypatch.setattr(pipeline, "judge_dir", lambda role: "frontier")
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: Judge())
    monkeypatch.setattr("lift.lang.LangID", FakeLID)
    monkeypatch.setattr("lift.audit.record_safety", record_safety)
    argv = ["judge", "--variant", "stock", "--split", "final", "--sets", "easy_benign", "--langs", "zh", "en"]
    assert main(argv) == 3
    judged_n = [len(pipeline.load_judgements("stock", "easy_benign", lang, "final")) for lang in ("zh", "en")]
    assert judged_n == [3, 4] and recorded == ["en"]
    generate = "lift generate --variant stock --split final --sets easy_benign --langs zh"
    assert (f"  easy_benign_zh_final: short of its frozen split: {short_remedy()}; then, after a later job's "
            f"`lift sync pull`, `{generate}` makes the prompts the completed file adds, and the line that "
            "failed judges them and writes the record\n") in caplog.text
    assert "stock easy_benign_zh_final: no AuditKIT record: easy_benign_zh: " in caplog.text


def test_the_module_docstring_lists_every_kind_of_skipped_cell_with_its_remedy():
    # W3-02: the module docstring still listed three kinds after W2-01 and W2-02 added short and nonfinite,
    # and its judges remedy left out the cell only the role's restart line remedies
    head = " ".join((run_cmd.__doc__ or "").split()).split(" Exit codes:")[0]
    for name in ("MixedJudgesError", "ShortCellError", "NonFiniteOutputError"):
        assert name in head, f"the module docstring does not name {name}"
    for kind in run_cmd.SKIPPED:                    # each under its name, as the end-of-run listing heads it
        assert re.search(rf"\. {kind} \(", head), f"the module docstring does not list the {kind} kind"
    assert "labelling rules" in head and "reader's verdict on the failures sheet" in head
    assert "no AuditKIT record" in head
    assert "made in bf16 on an A100 or L4" in head and "fault to report (CLAUDE.md rule 9)" in head


FAULT = "a fault in {}, no fp16 overflow: report it (CLAUDE.md rule 9), never tune around it"


def overflow_on(monkeypatch, call):
    """greedy_batch marking every row of its ``call``-th batch non-finite (an fp16 overflow, as its check
    marks it); the other batches decode as they do."""
    from dataclasses import replace

    from lift.models import generate as gen

    real, calls = gen.greedy_batch, []

    def greedy_batch(model, rows, **kw):
        calls.append(len(rows))
        outs = real(model, rows, **kw)
        return [replace(o, nonfinite=True) for o in outs] if len(calls) == call else outs

    monkeypatch.setattr(gen, "greedy_batch", greedy_batch)


def test_a_non_finite_cell_is_listed_apart_with_its_remedy():
    # W2-02: an fp16 cell that went non-finite is made in bf16 on an A100 or L4: its restart line first when
    # its file holds fp16 records (with the refill lines of the judgements it sets aside, made in bf16), else
    # nothing to set aside; in bf16 or fp32 it is a fault to report, no line
    from lift.models.generate import NonFiniteOutputError

    write_generation("harmful", "en", "dev", dtype="fp16")                 # an earlier batch's records
    write_judgements("frontier", "harmful", "en", "dev")                    # the second judge's sample
    line = "lift restart --variant stock --split dev --sets harmful --langs en --dtype fp16"
    errors = {"held": ("en", NonFiniteOutputError(f"... `{line}` ...", pids=["har-0002"], dtype="fp16",
                                                  command=line)),
              "fresh": ("hi", NonFiniteOutputError("...", pids=["har-0001"], dtype="fp16")),
              "bf16": ("zh", NonFiniteOutputError("...", pids=["har-0001"], dtype="bf16"))}
    remedies = {}
    for name, (lang, exc) in errors.items():
        skipped: dict[str, dict[str, str]] = {}
        run_cmd._skip(skipped, "stock", "harmful", lang, "dev", exc, judged=None, dtype=exc.dtype)
        [(kind, cells)] = skipped.items()
        assert (kind, list(cells)) == ("nonfinite", [f"harmful_{lang}_dev"]), name
        remedies[name] = cells[f"harmful_{lang}_dev"]
    cell = "--variant stock --split dev --sets harmful --langs en"
    assert remedies["held"] == (f"{line}\n    and once it is generated again: lift judge {cell} --judge "
                                "second --sample 0.10")
    assert remedies["fresh"] == ("nothing to set aside: run the line that failed in bf16 on an A100 or L4 "
                                 "(without --dtype)")
    assert remedies["bf16"] == FAULT.format("bf16")
    head = " ".join(run_cmd.SKIPPED["nonfinite"].split())
    assert "nothing of that batch was written" in head and "made in bf16" in head and "rule 9" in head


@pytest.mark.torch
def test_generate_skips_a_non_finite_cell_lists_it_and_exits_3(fake_prompt_files, tiny_model_dir, monkeypatch,
                                                               caplog):
    # W2-02: a batch that went non-finite stopped `lift generate` (and the job: its exit was 1), so no later
    # cell ran. NonFiniteOutputError is a StaleOutputError now: the cell is listed apart, the others go on,
    # and the line exits 3.
    pytest.importorskip("torch")
    from fakes import make_prompt

    from lift.models import generate as gen
    from lift.models import variants

    pin()
    for lang in ("en", "hi"):
        fake_prompt_files[("harmful", lang)] = [make_prompt(f"har-{i}", lang, "harmful", "dev", text=f"q {i}")
                                                for i in range(3)]
    real_build = variants.build_variant
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: real_build(
        spec, device="cpu", dtype=dtype, model_id=str(tiny_model_dir)))
    short_runs(monkeypatch)
    overflow_on(monkeypatch, 1)                                         # harmful_en's one batch
    argv = ["generate", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "hi",
            "--device", "cpu", "--dtype", "bf16"]
    assert main(argv) == 3
    assert not gen.generations_path("stock", "harmful", "en", "dev").exists()
    assert len(gen.read_generations(gen.generations_path("stock", "harmful", "hi", "dev"))) == 3
    assert listing(caplog, "went non-finite") == (
        f"1 cells {run_cmd.SKIPPED['nonfinite']}:\n  harmful_en_dev: {FAULT.format('bf16')}\n")
    assert "harmful_en_dev: decoding went non-finite, skipped" in caplog.text


@pytest.mark.parametrize("role", ["primary", "second"])
def test_judge_skips_a_stale_cell_and_exits_3(role, monkeypatch, caplog):
    # judge_file's refusal (a StaleOutputError) names the cell's restart line; lift judge lists it and
    # judges the other cells.
    from lift.judge import pipeline
    from lift.models.generate import generations_path

    pin(judge=True)
    for lang in ("en", "hi"):
        write_generation("harmful", lang, "dev")
    stale_path = generations_path("stock", "harmful", "en", "dev")
    line = f"lift restart --variant stock --split dev --sets harmful --langs en --judge {role}"
    judged_paths, recorded = [], []

    def judge_file(path, **kw):
        if path == stale_path:
            raise pipeline.StaleJudgementsError(f"{path}: 1 judgements made on another stack", command=line)
        judged_paths.append(path)
        return path

    monkeypatch.setattr(pipeline, "judge_file", judge_file)
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: object())
    monkeypatch.setattr(run_cmd, "_record", lambda variant, set_name, lang, *a, **kw: recorded.append(lang))
    argv = ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "hi",
            "--judge", role, "--device", "cpu"]
    assert main(argv) == 3
    assert judged_paths == [generations_path("stock", "harmful", "hi", "dev")]
    assert recorded == (["hi"] if role == "primary" else [])
    assert f"harmful_en_dev: {line}\n" in caplog.text


def listing(caplog, heading):
    """The one end-of-run listing whose message starts with "N cells " + ``heading`` ("" if none)."""
    found = [r.getMessage() for r in caplog.records if re.match(rf"\d+ cells {heading}", r.getMessage())]
    assert len(found) <= 1
    return found[0] if found else ""


def test_judge_lists_stale_other_precision_and_other_judge_cells_apart(monkeypatch, caplog):
    # Contract I: three kinds of skipped cell, three remedies. A stale one is started over with its restart
    # line; one made in another precision is never started over (that would discard it): its --dtype, or
    # leave it out; one holding another judge's judgements: the file's judge, or its --judge restart line.
    from lift.judge import pipeline
    from lift.models.generate import OtherPrecisionError

    pin(judge=True)
    for lang in ("en", "hi", "zh", "es"):
        write_generation("harmful", lang, "dev")
    restart = "lift restart --variant stock --split dev --sets harmful --langs {} --judge primary"
    errors = {
        "en": pipeline.StaleJudgementsError("judged on another stack", command=restart.format("en")),
        "hi": OtherPrecisionError("generated in fp16, and the judge runs in bf16", dtype="fp16"),
        "zh": pipeline.MixedJudgesError("another judge's judgements", command=restart.format("zh")),
    }
    judged_langs = []

    def judge_file(path, **kw):
        lang = path.stem.split("_")[1]
        if lang in errors:
            raise errors[lang]
        judged_langs.append(lang)
        return path

    monkeypatch.setattr(pipeline, "judge_file", judge_file)
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: object())
    monkeypatch.setattr(run_cmd, "_record", lambda *a, **kw: None)
    argv = ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "hi", "zh",
            "es", "--device", "cpu"]
    assert main(argv) == 3
    assert judged_langs == ["es"]
    stale = listing(caplog, "were skipped as stale")
    precision = listing(caplog, "were made in another precision")
    judges = listing(caplog, "hold another judge's judgements")
    assert f"  harmful_en_dev: {restart.format('en')}\n" in stale
    assert ("harmful_hi_dev: made in fp16: run it with --dtype fp16 where fp16 runs, or leave it out with "
            "--sets/--langs") in precision
    assert "lift restart" not in precision and "would discard" in precision
    assert f"  harmful_zh_dev: {restart.format('zh')}" in judges and "the file's judge" in judges
    for text, own in ((stale, "en"), (precision, "hi"), (judges, "zh")):
        assert [c for c in ("en", "hi", "zh", "es") if f"harmful_{c}_dev" in text] == [own]


@pytest.mark.parametrize("role", ["primary", "second"])
def test_a_cell_made_in_another_precision_is_listed_as_such_whatever_refusal_came_first(role, monkeypatch,
                                                                                       caplog):
    # X7 / contract M: a T4 judge line (--dtype fp16) over Colab's bf16 cells. Whatever judge_file refused
    # first (a stale cell, another judge's judgements, a file no line covers), each is listed as made in bf16
    # with its --dtype remedy, never with a restart line: the T4 restarts in bf16 by default, so that line
    # would discard Colab's outputs. The frontier second judge has no precision for judge_file to check.
    # A cell made in the run's own precision keeps its kind and its line.
    from lift.judge import pipeline

    pin(judge=True)
    for lang in ("en", "hi", "zh"):
        write_generation("harmful", lang, "dev", dtype="bf16")
    write_generation("harmful", "es", "dev", dtype="fp16")
    line = "lift restart --variant stock --split dev --sets harmful --langs {}"
    lines = {"en": line.format("en"), "hi": line.format("hi") + f" --judge {role}",
             "es": line.format("es") + f" --judge {role} --dtype fp16"}
    errors = {
        "en": pipeline.StaleJudgementsError(f"prompt changed; run `{lines['en']}`", command=lines["en"]),
        "hi": pipeline.MixedJudgesError(f"another judge; run `{lines['hi']}`", command=lines["hi"]),
        "zh": pipeline.StaleJudgementsError("no restart line covers that file"),
        "es": pipeline.StaleJudgementsError(f"another stack; run `{lines['es']}`", command=lines["es"]),
    }

    def judge_file(path, **kw):
        raise errors[path.stem.split("_")[1]]

    monkeypatch.setattr(pipeline, "judge_file", judge_file)
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: object())
    monkeypatch.setattr(run_cmd, "_record", lambda *a, **kw: None)
    argv = ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "hi", "zh",
            "es", "--judge", role, "--device", "cpu", "--dtype", "fp16"]
    assert main(argv) == 3
    precision = listing(caplog, "were made in another precision")
    for lang in ("en", "hi", "zh"):
        assert (f"  harmful_{lang}_dev: made in bf16: run it with --dtype bf16 where bf16 runs, or leave it "
                "out with --sets/--langs\n") in precision
        assert f"harmful_{lang}_dev: made in another precision, skipped" in caplog.text
    assert lines["en"] not in caplog.text and lines["hi"] not in caplog.text      # not even in its error
    assert listing(caplog, "hold another judge's judgements") == ""
    assert listing(caplog, "were skipped as stale") == f"1 cells {run_cmd.SKIPPED['stale']}:\n" \
                                                       f"  harmful_es_dev: {lines['es']}\n"


def test_judge_keys_refuse_a_restarted_cell_before_counting_its_keys_as_missing(monkeypatch):
    # X8: lift restart left harmful_en_dev empty, and the second judge's line runs right after it, as its
    # listing said. The listed keys of that cell are in no file, which says nothing about the key list: the
    # cell needs generating. It is refused with lift generate's message before the unknown-key check, also
    # with --allow-missing (which would judge nothing and exit 0), with or without P04's --sets/--langs; so is
    # a missing file an explicit --sets and --langs name. Nothing is loaded. When every listed key is found,
    # an empty cell (say a restarted utility cell, which the second judge never reads) stops nothing.
    from lift.judge import pipeline

    keys = [write_generation("harmful", lang, "dev") for lang in ("en", "hi")]
    listed = generations_dir().parent / "stock_dev.txt"
    listed.write_text("".join(k + "\n" for k in keys), encoding="utf-8")
    assert main([*RESTART, "--sets", "harmful", "--langs", "en"]) == 0
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: pytest.fail("the judge was loaded"))
    second = ["judge", "--variant", "stock", "--split", "dev", "--judge", "second", "--keys", str(listed)]
    refused = r"^1 generations files are missing or empty, e\.g\. \[[^]]*harmful_{}_dev\.jsonl"
    for extra in ([], ["--allow-missing"], ["--sets", "harmful", "--langs", "en"],
                  ["--sets", "harmful", "--langs", "en", "--allow-missing"]):
        with pytest.raises(SystemExit) as exc:
            main([*second, *extra])
        message = str(exc.value.code)
        assert re.search(refused.format("en"), message)
        assert "run lift generate first" in message and "listed keys" not in message
    with pytest.raises(SystemExit, match=refused.format("zh")):
        main([*second, "--sets", "harmful", "--langs", "hi", "zh"])
    # An empty cell no listed key can be in (every key is found) is no reason to stop the line.
    write_generation("utility", "en", "dev")
    assert main([*RESTART, "--sets", "utility", "--langs", "en"]) == 0
    assert run_cmd._key_cells("stock", "dev", keys[1:], sets=None, langs=None, allow_missing=False) == [
        ("harmful", "hi")]


def test_a_judge_lines_whole_cell_restart_names_the_line_that_generates_the_cell_again(monkeypatch, caplog):
    # X8: a whole-cell restart empties the generations too, and lift judge judges only what is there: its
    # listing names the lift generate line (in the run's precision) to run before the line that failed, then
    # the other judge's refill line (lift run-dev and lift generate make the cell themselves).
    from lift.judge import pipeline

    pin(judge=True)
    write_generation("harmful", "en", "dev", dtype="fp16")
    for judge in ("qwen3guard", "frontier"):
        write_judgements(judge, "harmful", "en", "dev")
    restart = "lift restart --variant stock --split dev --sets harmful --langs en --dtype fp16"
    error = pipeline.StaleJudgementsError("1 generations were made from another prompt text", command=restart)

    def judge_file(path, **kw):
        raise error

    monkeypatch.setattr(pipeline, "judge_file", judge_file)
    monkeypatch.setattr(pipeline, "make_judge", lambda *a, **kw: object())
    argv = ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "--device",
            "cpu", "--dtype", "fp16"]
    assert main(argv) == 3
    stale = listing(caplog, "were skipped as stale")
    cell = "--variant stock --split dev --sets harmful --langs en"
    assert (f"  harmful_en_dev: {restart}\n"
            f"    first generate it again: lift generate {cell} --dtype fp16\n"
            f"    and once it is generated again: lift judge {cell} --judge second --sample 0.10 "
            "--dtype fp16\n") in stale
    assert "followed by the lines under it where they say" in stale
    skipped: dict[str, dict[str, str]] = {}                 # lift run-dev or lift generate: no generate line
    run_cmd._skip(skipped, "stock", "harmful", "en", "dev", error, judged="primary", dtype="fp16")
    assert "lift generate" not in skipped["stale"]["harmful_en_dev"]


@pytest.mark.parametrize("argv", [
    ["generate", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "hi"],
    ["run-dev", "--variant", "stock", "--sets", "harmful", "--langs", "hi", "--judge-device", "cuda:0"],
    ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "hi"],
])
@pytest.mark.parametrize("capability, dtype, refused", [
    ("8.0", "fp16", True), ("9.0", "fp16", True),           # an A100 or H100: bf16 is native there
    ("8.0", "bf16", False),                                 # generation.dtype is never refused
    ("7.5", "fp16", False),                                 # the T4 the fp16 licence is for
])
def test_another_precision_than_the_default_is_refused_on_a_gpu_with_native_bf16(argv, capability, dtype,
                                                                                 refused, monkeypatch):
    # W15 / contract L: an fp16 line reused on an A100 (say the Kaggle quota ran out) would make fp16 cells
    # the licence never compared, read for reporting under the T4's comparison. Refused before a model loads.
    from lift.judge import pipeline
    from lift.models import generate as gen
    from lift.models import variants

    class Loaded(Exception):
        pass

    def loading(*a, **kw):
        raise Loaded

    pin(judge=True)
    write_generation("harmful", "hi", "dev")
    monkeypatch.setattr(gen, "stack", lambda device: {"torch": "2.9.0", "transformers": "5.22.1",
                                                      "gpu_capability": capability})
    monkeypatch.setattr(variants, "build_variant", loading)
    monkeypatch.setattr(pipeline, "make_judge", loading)
    full = [*argv, "--device", "cuda:0", "--dtype", dtype]
    if not refused:
        with pytest.raises(Loaded):
            main(full)
        return
    with pytest.raises(SystemExit) as exc:
        main(full)
    message = str(exc.value.code)
    assert f"--dtype {dtype}" in message and capability in message and "T4" in message


@pytest.mark.parametrize("argv", [
    ["generate", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "hi"],
    ["run-dev", "--variant", "stock", "--sets", "harmful", "--langs", "hi", "--judge-device", "cpu"],
    ["judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "hi"],
])
def test_a_cpu_device_is_never_refused_another_precision(argv, monkeypatch):
    # Contract L's exemption: tests (and a laptop) run in fp32 on CPU.
    from lift.judge import pipeline
    from lift.models import variants

    class Loaded(Exception):
        pass

    def loading(*a, **kw):
        raise Loaded

    pin(judge=True)
    write_generation("harmful", "hi", "dev")
    monkeypatch.setattr(variants, "build_variant", loading)
    monkeypatch.setattr(pipeline, "make_judge", loading)
    dtype = "fp16" if argv[0] == "run-dev" else "fp32"        # run-dev offers bf16 and fp16 only
    with pytest.raises(Loaded):
        main([*argv, "--device", "cpu", "--dtype", dtype])


@pytest.mark.torch
def test_run_dev_skips_a_stale_cell_exits_3_and_its_restart_line_lets_it_finish(fake_prompt_files,
                                                                                 tiny_model_dir, monkeypatch,
                                                                                 caplog):
    # R2 on the judge side, through lift run-dev with the real Qwen3Guard adapter. Session 1's judge dies in
    # the middle of easy_benign's judgements (chunks of 2: calls 1-2 judge harmful, call 3 the first two of
    # easy_benign). Session 2 resolved a newer transformers: easy_benign cannot be finished on it, so it is
    # skipped and listed with its restart line, hard_benign is still done, and run-dev exits 3.
    pytest.importorskip("torch")
    from fakes import make_prompt

    from lift.judge import pipeline
    from lift.models import variants

    pin(judge=True)
    sets = ("harmful", "easy_benign", "hard_benign")
    for s in sets:
        fake_prompt_files[(s, "en")] = [make_prompt(f"{s[:3]}-{i}", "en", s, "dev", text=f"question {i}")
                                        for i in range(3)]
    real_build = variants.build_variant
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: real_build(
        spec, device="cpu", dtype=dtype, model_id=str(tiny_model_dir)))
    short_runs(monkeypatch)
    judges = iter([guard(die_on_call=4), guard(), guard()])
    monkeypatch.setattr(pipeline, "make_judge", lambda role, **kw: next(judges))
    monkeypatch.setattr("lift.lang.LangID", FakeLID)
    recorded = []
    monkeypatch.setattr("lift.audit.record_safety", lambda *a, **kw: recorded.append(a[1]) or "record.json")
    argv = ["run-dev", "--variant", "stock", "--langs", "en", "--sets", *sets, "--device", "cpu",
            "--judge-device", "cpu", "--dtype", "bf16"]

    with pytest.raises(RuntimeError, match="judge died"):
        main(argv)
    assert [judged("stock", s, "en") for s in sets] == [3, 2, 0]
    moved_stack(monkeypatch, transformers="5.99.0")
    caplog.clear()
    assert main(argv) == 3
    assert [judged("stock", s, "en") for s in sets] == [3, 2, 3]
    line = "lift restart --variant stock --split dev --sets easy_benign --langs en --judge primary"
    assert f"easy_benign_en_dev: {line}\n" in caplog.text
    assert "question" not in caplog.text                  # cell names and counts, never text

    assert main(line.split()[1:]) == 0                    # the job line right after the pull, then run-dev
    assert main(argv) == 0
    assert [judged("stock", s, "en") for s in sets] == [3, 3, 3]
    assert len(judge_stacks("stock", "easy_benign", "en")) == 1
    assert recorded.count("easy_benign") == 1             # its AuditKIT record waits for the whole cell


def tiny_run_dev(fake_prompt_files, tiny_model_dir, monkeypatch, sets, *, n=3):
    """run-dev on the tiny model (CPU), short outputs, the real Qwen3Guard adapter in the run's precision
    over a StubGuard, a fake LID; the en prompts of ``sets`` are "question {i}" placeholders. The tiny model
    stands in for Global at the revision ``pin`` pins, which its generations record as a real run's do (a
    read compares them with the pin)."""
    from dataclasses import replace

    from fakes import make_prompt

    from lift.judge import pipeline
    from lift.models import variants

    pin(judge=True)
    for s in sets:
        fake_prompt_files[(s, "en")] = [make_prompt(f"{s[:3]}-{i}", "en", s, "dev", text=f"question {i}")
                                        for i in range(n)]
    real_build = variants.build_variant
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: replace(real_build(
        spec, device="cpu", dtype=dtype, model_id=str(tiny_model_dir)), model_rev="b" * 40))
    short_runs(monkeypatch)
    monkeypatch.setattr(pipeline, "make_judge", lambda role, **kw: guard(dtype=kw.get("dtype") or "bf16"))
    monkeypatch.setattr("lift.lang.LangID", FakeLID)
    return ["run-dev", "--variant", "stock", "--langs", "en", "--device", "cpu", "--judge-device", "cpu"]


class Frontier:
    """Stands in for the frontier second judge (an API judge: no stack, no precision), under the id the
    pipeline gives the configured one: its rubric's version, OpenAI's default sampling, a stand-in API
    version and the tag of the labelling rules (``second_judged`` reads its judgements back)."""

    short = "frontier"

    @property
    def judge_id(self):
        from lift import config
        from lift.judge.frontier import load_rubric
        from lift.judge.labels import rules_tag

        rubric = load_rubric(config.get("judges.safety_second.rubric")).version
        return f"openai/placeholder-model@{rubric}+t0+max1024+apitest#{rules_tag()}"

    def judge(self, items):
        from lift.judge.pipeline import JudgeOutcome

        return [JudgeOutcome("refusal", {"label": "REFUSAL", "reason": "placeholder"}, "") for _ in items]


def with_second_judge(monkeypatch):
    """The frontier second judge (``Frontier``) configured beside the primary, the real Qwen3Guard adapter in
    the precision it is built with."""
    from lift import config
    from lift.judge import pipeline

    config.load("project")["judges"]["safety_second"].update(provider="openai", model="placeholder-model")
    monkeypatch.setattr(pipeline, "make_judge", lambda role, **kw: (
        guard(dtype=kw.get("dtype") or "bf16") if role == "primary" else Frontier()))


def second_judged():
    """How many judgements the second judge's file of stock harmful_en_dev holds."""
    from lift.judge import pipeline

    return len(pipeline.load_records("stock", "harmful", "en", "dev", judge="second", complete=False))


@pytest.mark.torch
def test_an_fp16_run_leaves_a_bf16_cell_alone_and_a_restart_needs_force(fake_prompt_files, tiny_model_dir,
                                                                         monkeypatch, caplog):
    # W3: Colab finished harmful_en in bf16; the Kaggle backup then runs the variant in fp16 (every T4 line
    # takes --dtype fp16). The bf16 cell is listed as made in another precision, with its two remedies and
    # never a restart line (following one would replace the runs repo's bf16 cell with fp16 outputs), the
    # other cells are made in fp16, and run-dev exits 3. A restart in fp16 is refused unless --force.
    pytest.importorskip("torch")
    from lift.models.generate import generations_path, read_generations

    argv = tiny_run_dev(fake_prompt_files, tiny_model_dir, monkeypatch, ("harmful", "easy_benign"))
    monkeypatch.setattr("lift.audit.record_safety", lambda *a, **kw: "record.json")
    assert main([*argv, "--sets", "harmful", "--dtype", "bf16"]) == 0          # Colab
    cell = generations_path("stock", "harmful", "en", "dev")
    bf16 = cell.read_bytes()
    caplog.clear()
    assert main([*argv, "--sets", "harmful", "easy_benign", "--dtype", "fp16"]) == 3     # the T4 backup
    assert cell.read_bytes() == bf16
    assert "lift restart" not in caplog.text
    precision = listing(caplog, "were made in another precision")
    assert "harmful_en_dev: made in bf16: run it with --dtype bf16 where bf16 runs, or leave" in precision
    made = read_generations(generations_path("stock", "easy_benign", "en", "dev"))
    assert {g.meta["dtype"] for g in made} == {"fp16"} and judged("stock", "easy_benign", "en") == 3

    restart = ["restart", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en",
               "--dtype", "fp16"]
    assert main(restart) == 1 and cell.read_bytes() == bf16
    assert not (output_dir() / ".aside").exists()
    assert main([*restart, "--force"]) == 0 and cell.read_bytes() == b""


@pytest.mark.torch
def test_a_restarted_cells_second_judge_sample_is_refilled_by_the_line_both_logs_name(fake_prompt_files,
                                                                                     tiny_model_dir,
                                                                                     monkeypatch, caplog):
    # W9, with decision 9's case: a translation is fixed after the second judge read the cell. run-dev lists
    # the whole-cell restart line AND the line that refills the second judge's sample (the restart sets that
    # file aside too, and run-dev never judges with the second judge); the restart's log names it again. The
    # documented job (restart, run-dev) plus that line leaves no empty second-judge sample behind.
    pytest.importorskip("torch")
    import shlex

    from fakes import make_prompt

    argv = tiny_run_dev(fake_prompt_files, tiny_model_dir, monkeypatch, ("harmful",), n=4)
    with_second_judge(monkeypatch)
    monkeypatch.setattr("lift.audit.record_safety", lambda *a, **kw: "record.json")

    assert main([*argv, "--sets", "harmful"]) == 0
    keys = p04_list("har-0", "har-1")                       # P04's list (decisions.md 31)
    assert main(["judge", "--variant", "stock", "--split", "dev", "--judge", "second", "--keys",
                 str(keys)]) == 0
    assert second_judged() == 2

    fake_prompt_files[("harmful", "en")][1] = make_prompt("har-1", "en", "harmful", "dev", text="question 1b")
    caplog.clear()
    assert main([*argv, "--sets", "harmful"]) == 3
    stale = listing(caplog, "were skipped as stale")
    restart = "lift restart --variant stock --split dev --sets harmful --langs en"
    refill = ["lift", "judge", "--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en",
              "--judge", "second", "--keys", str(keys)]
    found = re.search(r"harmful_en_dev: (lift restart [^\n]+)\n\s+.*?(lift judge [^\n]+)", stale)
    assert found and found.group(1) == restart and shlex.split(found.group(2)) == refill

    caplog.clear()
    assert main(restart.split()[1:]) == 0                   # the job: restart, the failed line, the refill
    logged = re.search(r"frontier\S*harmful_en_dev\.jsonl: (lift judge [^\n]+)", caplog.text)
    assert logged and shlex.split(logged.group(1)) == refill
    assert main([*argv, "--sets", "harmful"]) == 0
    p04_list("har-0", "har-1")                              # P04's list, drawn again for the new keys
    assert main(refill[1:]) == 0
    assert second_judged() == 2 and judged("stock", "harmful", "en") == 4


@pytest.mark.torch
def test_a_refill_line_waits_for_p04_to_draw_the_key_list_again_when_the_keys_changed(fake_prompt_files,
                                                                                     tiny_model_dir,
                                                                                     monkeypatch, caplog):
    # X9: P04's list holds har-1, the prompt fixed after the second judge read the cell (every validated item
    # is in it). The cell made again has a new key for it, so the refill line in run-dev's listing, and the
    # restart's log, say to run it only after P04 draws the list again (run before, it would exit 1 and end
    # a Colab job). Followed literally, the second judge's sample is back.
    pytest.importorskip("torch")
    import shlex

    from fakes import make_prompt

    argv = tiny_run_dev(fake_prompt_files, tiny_model_dir, monkeypatch, ("harmful",), n=4)
    with_second_judge(monkeypatch)
    monkeypatch.setattr("lift.audit.record_safety", lambda *a, **kw: "record.json")
    assert main([*argv, "--sets", "harmful"]) == 0
    second = ["judge", "--variant", "stock", "--split", "dev", "--judge", "second", "--keys"]
    assert main([*second, str(p04_list("har-1", "har-3"))]) == 0

    fake_prompt_files[("harmful", "en")][1] = make_prompt("har-1", "en", "harmful", "dev", text="question 1b")
    caplog.clear()
    assert main([*argv, "--sets", "harmful"]) == 3
    found = re.search(r"harmful_en_dev: (lift restart [^\n]+)\n    and once it is generated again, after P04 "
                      r"draws the key list again \(lift validate second-judge-sample, then lift sync push\): "
                      r"(lift judge [^\n]+)\n", listing(caplog, "were skipped as stale"))
    assert found
    restart, refill = found.group(1), shlex.split(found.group(2))
    caplog.clear()
    assert main(restart.split()[1:]) == 0
    assert "run a --keys line only after P04 draws the key list again" in caplog.text
    assert main([*argv, "--sets", "harmful"]) == 0
    with pytest.raises(SystemExit, match="1 listed keys are in none"):        # P04's list, not drawn again
        main(refill[1:])
    p04_list("har-1", "har-3")                              # P04 draws it again: har-1's new key
    assert main(refill[1:]) == 0
    assert second_judged() == 2


def test_a_refill_line_needs_no_new_key_list_when_the_cell_keeps_its_keys(monkeypatch, isolated_dirs):
    # X9: a cell begun under another stack is made again under the same keys (the key leaves the stack out),
    # so P04's list still holds and the refill line is plain; any other cause may change them.
    from lift.models.generate import StaleOutputError

    write_generation("harmful", "en", "dev")
    write_judgements("frontier", "harmful", "en", "dev")
    p04_list("har-0001")
    repo_layout(monkeypatch, isolated_dirs)
    restart = "lift restart --variant stock --split dev --sets harmful --langs en"
    refill = ("lift judge --variant stock --split dev --sets harmful --langs en --judge second "
              "--keys validation/keys/second_judge/stock_dev.txt")
    redraw = ", after P04 draws the key list again (lift validate second-judge-sample, then lift sync push)"
    for keys_changed, note in ((False, ""), (True, redraw)):
        skipped: dict[str, dict[str, str]] = {}
        exc = StaleOutputError("x", command=restart, keys_changed=keys_changed)
        run_cmd._skip(skipped, "stock", "harmful", "en", "dev", exc, judged="primary", dtype="bf16")
        remedy = f"{restart}\n    and once it is generated again{note}: {refill}"
        assert skipped == {"stale": {"harmful_en_dev": remedy}}


@pytest.mark.torch
def test_the_second_judges_own_stale_line_followed_literally_judges_its_sample_again(fake_prompt_files,
                                                                                     tiny_model_dir,
                                                                                     monkeypatch, caplog):
    # X8: after a translation fix (decision 9) the canonical second-judge line (decisions.md 31) finds the
    # cell stale. Its listing names the whole-cell restart and, under it, the lift generate line a judge line
    # needs (the restart empties the generations too), when to run the line that failed (P04's list may hold
    # a changed key), and the primary judge's refill. Run too early, the judge line is refused with lift
    # generate's message, also with --allow-missing, never as a stale key list. Followed literally, every
    # judgement of the cell is back.
    pytest.importorskip("torch")
    from fakes import make_prompt

    argv = tiny_run_dev(fake_prompt_files, tiny_model_dir, monkeypatch, ("harmful",), n=4)
    with_second_judge(monkeypatch)
    monkeypatch.setattr("lift.audit.record_safety", lambda *a, **kw: "record.json")
    assert main([*argv, "--sets", "harmful"]) == 0
    second = ["judge", "--variant", "stock", "--split", "dev", "--judge", "second", "--keys",
              str(p04_list("har-0", "har-3"))]
    assert main(second) == 0 and second_judged() == 2

    fake_prompt_files[("harmful", "en")][1] = make_prompt("har-1", "en", "harmful", "dev", text="question 1b")
    caplog.clear()
    assert main(second) == 3
    found = re.search(r"harmful_en_dev: (lift restart [^\n]+)\n"
                      r"    first generate it again: (lift generate [^\n]+)\n"
                      r"    then the line that failed, after P04 draws the key list again \(lift validate "
                      r"second-judge-sample, then lift sync push\)\n"
                      r"    and once it is generated again: (lift judge [^\n]+)\n",
                      listing(caplog, "were skipped as stale"))
    assert found
    restart, generate, refill = found.groups()
    assert generate == "lift generate --variant stock --split dev --sets harmful --langs en"
    assert refill == "lift judge --variant stock --split dev --sets harmful --langs en --judge primary"

    assert main(restart.split()[1:]) == 0                   # the job: right after lift sync pull
    for extra in ([], ["--allow-missing"]):                 # too early: the cell is empty
        with pytest.raises(SystemExit, match="run lift generate first"):
            main([*second, *extra])
    assert main(generate.split()[1:]) == 0
    p04_list("har-0", "har-3")                              # P04 draws the list again
    assert main(second) == 0
    assert main(refill.split()[1:]) == 0
    assert second_judged() == 2 and judged("stock", "harmful", "en") == 4


@pytest.mark.torch
def test_a_t4_line_lists_a_bf16_cell_with_a_fixed_prompt_as_made_in_bf16(fake_prompt_files, tiny_model_dir,
                                                                          monkeypatch, caplog):
    # X7: Colab made harmful_en in bf16, then a translation was fixed (decision 9). On the T4, kaggle/README's
    # separate-steps job (lift generate, then lift judge, both --dtype fp16) and the frontier second judge's
    # line all list the cell as made in bf16, never with the restart line: the T4 restarts in bf16 by default,
    # so following it would discard Colab's cell and let the T4's next fp16 line make it in fp16.
    pytest.importorskip("torch")
    from fakes import make_prompt

    from lift.models.generate import generations_path

    argv = tiny_run_dev(fake_prompt_files, tiny_model_dir, monkeypatch, ("harmful",), n=3)
    with_second_judge(monkeypatch)
    monkeypatch.setattr("lift.audit.record_safety", lambda *a, **kw: "record.json")
    assert main([*argv, "--sets", "harmful", "--dtype", "bf16"]) == 0               # Colab
    cell = generations_path("stock", "harmful", "en", "dev")
    bf16 = cell.read_bytes()
    listed = p04_list("har-0", "har-1")
    fake_prompt_files[("harmful", "en")][1] = make_prompt("har-1", "en", "harmful", "dev", text="question 1b")
    t4 = ["--variant", "stock", "--split", "dev", "--sets", "harmful", "--langs", "en", "--device", "cpu",
          "--dtype", "fp16"]
    second = ["--judge", "second", "--keys", str(listed)]
    for line in (["generate", *t4], ["judge", *t4], ["judge", *t4, *second]):
        caplog.clear()
        assert main(line) == 3
        assert ("  harmful_en_dev: made in bf16: run it with --dtype bf16 where bf16 runs, or leave it out "
                "with --sets/--langs\n") in listing(caplog, "were made in another precision")
        assert listing(caplog, "were skipped as stale") == "" and "lift restart" not in caplog.text
    assert cell.read_bytes() == bf16


# --- lift smoke -------------------------------------------------------------------------------------------


@pytest.mark.torch
def test_smoke_prints_why_a_check_failed(monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    from lift import gate0

    failure = "label agreement 0.912 is under 0.95"
    checks = {"precision": {"pass": True}, "fp16": {"pass": False, "comparable": True, "failure": failure}}
    result = gate0.decide({"dtype": "fp16", "reference_dtype": "bf16", "device": "cuda:0", "gpu": "Tesla T4",
                           "gpu_capability": "7.5", "model_id": "x", "revisions": {"global": "a" * 40},
                           "required": list(gate0.required_checks("fp16")), "skipped": [], "checks": checks})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate0, "run_gate0", lambda **kw: result)
    assert main(["smoke", "--compare-dtype", "fp16", "--judge-device", "cuda:1"]) == 1
    assert f"FAIL  fp16  ({failure})" in capsys.readouterr().out


@pytest.mark.torch
def test_smoke_writes_no_report_for_a_refused_configuration(monkeypatch, capsys):
    # e.g. a bf16 Gate 0 on a GPU without native bf16, or --compare-dtype fp16 on an A100
    torch = pytest.importorskip("torch")
    from lift import gate0
    from lift.paths import metrics_dir

    def refuse(**kw):
        raise ValueError("compute capability 7.5 has no native bf16: run --compare-dtype fp16 here")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gate0, "run_gate0", refuse)
    assert main(["smoke", "--judge-device", "cuda:1"]) == 2
    assert "no native bf16" in capsys.readouterr().err
    assert not metrics_dir().exists() or not list(metrics_dir().glob("gate0*"))


@pytest.mark.torch
class TestRunCells:
    """run-dev's orchestration with the tiny model, a fake judge and a fake LID."""

    @pytest.fixture
    def setup(self, fake_prompt_files, tiny_model_dir):
        pytest.importorskip("torch")
        from fakes import make_prompt

        from lift.models.variants import build_variant
        from lift.schemas import VariantSpec

        for set_name in ("harmful", "easy_benign"):
            for lang in ("en", "hi"):
                fake_prompt_files[(set_name, lang)] = [
                    make_prompt(f"{set_name[:3]}-{i}", lang, set_name, "dev", text=f"question {i} {lang}")
                    for i in range(3)]
        def holder():
            return {"variant": build_variant(VariantSpec(name="stock", method="stock"), device="cpu",
                                             dtype="fp32", model_id=str(tiny_model_dir))}

        cells = [(s, lang) for s in ("harmful", "easy_benign") for lang in ("en", "hi")]
        return holder, cells

    class Judge:
        judge_id = "fake/judge@test"
        short = "qwen3guard"

        def __init__(self, explode=False, crash_on_call=None):
            self.explode = explode
            self.crash_on_call = crash_on_call
            self.calls = 0

        def judge(self, items):
            from lift.judge.pipeline import JudgeOutcome

            self.calls += 1
            if self.explode or self.calls == self.crash_on_call:
                raise RuntimeError("judge crashed")
            return [JudgeOutcome("benign", {}, "Safety: Safe") for _ in items]

    class LID:
        def top(self, text, k=3):
            return [("eng_Latn", 0.95)]

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_every_cell_is_generated_and_judged(self, setup, mode):
        from lift.judge.pipeline import load_judgements

        make_holder, cells = setup
        recorded = []
        paths = run_cmd.run_cells(make_holder(), lambda: self.Judge(), cells, concurrent=mode == "concurrent",
                                  interleave=mode == "interleave", lid=self.LID(), batch_size=2,
                                  record=lambda s, lang: recorded.append((s, lang)))
        assert len(paths) == len(cells) and sorted(recorded) == sorted(cells)
        for set_name, lang in cells:
            assert len(load_judgements("stock", set_name, lang, "dev")) == 3

    def test_the_one_gpu_path_frees_the_generator_before_the_judge_loads(self, setup):
        import gc
        import weakref

        make_holder, cells = setup
        holder = make_holder()
        model = weakref.ref(holder["variant"].model)
        alive_when_judge_loads = []

        def make_judge():
            gc.collect()
            alive_when_judge_loads.append(model() is not None)
            return self.Judge()

        run_cmd.run_cells(holder, make_judge, cells, concurrent=False, interleave=False, lid=self.LID())
        assert alive_when_judge_loads == [False] and holder == {}

    def test_finished_cells_are_judged_after_the_stack_moved(self, setup, monkeypatch):
        # Session 1 (one GPU) generates every cell, then its judge dies on the second file. Session 2
        # resolved a newer transformers: the finished files are reused as they are, the rest is judged.
        from lift.judge.pipeline import judgements_path, load_judgements
        from lift.models import generate as gen_mod

        make_holder, cells = setup
        real_generate, real_stack = gen_mod.generate, gen_mod.stack
        monkeypatch.setattr(gen_mod, "generate", lambda *a, **kw: real_generate(*a, **kw, max_new_tokens=8))

        def session(judge):
            run_cmd.run_cells(make_holder(), lambda: judge, cells, concurrent=False, interleave=False,
                              lid=self.LID())

        with pytest.raises(RuntimeError, match="judge crashed"):
            session(self.Judge(crash_on_call=2))
        judged = [judgements_path("qwen3guard", "stock", *c, "dev").exists() for c in cells]
        assert judged == [True, False, False, False]
        files = {c: gen_mod.generations_path("stock", *c, "dev").read_bytes() for c in cells}
        monkeypatch.setattr(gen_mod, "stack", lambda device: {**real_stack(device), "transformers": "5.99.0"})
        session(self.Judge())
        assert {c: gen_mod.generations_path("stock", *c, "dev").read_bytes() for c in cells} == files
        assert all(len(load_judgements("stock", *c, "dev")) == 3 for c in cells)

    def session(self, mode, make_holder, judge, cells, **kw):
        return run_cmd.run_cells(make_holder(), lambda: judge, cells, concurrent=mode == "concurrent",
                                 interleave=mode == "interleave", lid=self.LID(), **kw)

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_a_cell_judged_halfway_on_another_stack_is_listed_and_the_others_are_judged(self, setup,
                                                                                       monkeypatch, mode):
        # The verifier's R2 scenario with the real Qwen3Guard adapter. Session 1's judge dies in the middle
        # of harmful_hi's judgements (chunks of 2: calls 1-2 judge harmful_en, call 3 the first two of
        # harmful_hi). Session 2 resolved a newer transformers: harmful_hi cannot be finished on it, so it
        # is skipped with its restart line, and every other cell is judged. The judge runs in the cells'
        # precision (fp32 here): a judge in another one is refused before it judges anything.
        make_holder, cells = setup
        short_runs(monkeypatch)
        with pytest.raises(RuntimeError, match="judge died"):
            self.session(mode, make_holder, guard(die_on_call=4, dtype="fp32"), cells)
        assert [judged("stock", *c) for c in cells[:2]] == [3, 2]
        moved_stack(monkeypatch, transformers="5.99.0")
        skipped: dict[str, dict[str, str]] = {}
        self.session(mode, make_holder, guard(dtype="fp32"), cells, skipped=skipped)
        line = "lift restart --variant stock --split dev --sets harmful --langs hi --judge primary"
        assert skipped == {"stale": {"harmful_hi_dev": line + " --dtype fp32"}}        # the cell's precision
        assert [judged("stock", *c) for c in cells] == [3, 2, 3, 3]

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_a_cell_generated_halfway_on_another_stack_is_listed_and_the_others_go_on(self, setup,
                                                                                     monkeypatch, mode):
        from lift.models import generate as gen_mod
        from lift.prompts import load

        make_holder, cells = setup
        short_runs(monkeypatch)
        part = gen_mod.generations_path("stock", "harmful", "hi", "dev")
        cut_short = load("harmful", "hi", "dev")[:2]
        gen_mod.generate(make_holder()["variant"], cut_short, out_path=part)
        before = part.read_bytes()
        moved_stack(monkeypatch, torch="2.99.0+cu130")
        skipped: dict[str, dict[str, str]] = {}
        self.session(mode, make_holder, self.Judge(), cells, skipped=skipped)
        line = "lift restart --variant stock --split dev --sets harmful --langs hi --dtype fp32"   # the run's
        assert skipped == {"stale": {"harmful_hi_dev": line}}
        assert part.read_bytes() == before
        assert [judged("stock", *c) for c in cells] == [3, 0, 3, 3]

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_a_cell_made_in_another_precision_is_listed_apart_with_its_dtype(self, setup, monkeypatch, mode):
        # W3: the cell is never offered a restart (that would discard the other precision's outputs).
        import dataclasses

        from lift.models import generate as gen_mod
        from lift.prompts import load

        make_holder, cells = setup
        short_runs(monkeypatch)
        part = gen_mod.generations_path("stock", "harmful", "hi", "dev")
        other = dataclasses.replace(make_holder()["variant"], dtype="fp16")
        gen_mod.generate(other, load("harmful", "hi", "dev"), out_path=part)
        before = part.read_bytes()
        skipped: dict[str, dict[str, str]] = {}
        self.session(mode, make_holder, self.Judge(), cells, skipped=skipped)
        assert skipped == {"precision": {"harmful_hi_dev": (
            "made in fp16: run it with --dtype fp16 where fp16 runs, or leave it out with --sets/--langs")}}
        assert part.read_bytes() == before
        assert [judged("stock", *c) for c in cells] == [3, 0, 3, 3]

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_a_short_cell_is_listed_with_its_remedy_and_the_others_go_on(self, tiny_model_dir, tmp_path,
                                                                         monkeypatch, mode):
        # W2-01: run-dev met a cell short of its frozen split with a traceback, so no later cell ran
        from lift.models.variants import build_variant
        from lift.schemas import VariantSpec

        frozen_cell(tmp_path, monkeypatch)
        short_runs(monkeypatch)
        holder = {"variant": build_variant(VariantSpec(name="stock", method="stock"), device="cpu",
                                           dtype="fp32", model_id=str(tiny_model_dir))}
        skipped: dict[str, dict[str, str]] = {}
        run_cmd.run_cells(holder, lambda: self.Judge(), [("easy_benign", "zh"), ("easy_benign", "en")],
                          split="final", concurrent=mode == "concurrent", interleave=mode == "interleave",
                          lid=self.LID(), skipped=skipped)
        assert skipped == {"short": {"easy_benign_zh_final": short_remedy()}}
        assert [judged("stock", "easy_benign", lang, "final") for lang in ("zh", "en")] == [0, 4]

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_a_cell_that_goes_non_finite_is_listed_and_the_others_go_on(self, setup, monkeypatch, mode):
        # W2-02: a non-finite batch stopped run-dev (and the job), so no later cell ran
        make_holder, cells = setup
        short_runs(monkeypatch)
        overflow_on(monkeypatch, 2)                                      # harmful_hi's one batch
        skipped: dict[str, dict[str, str]] = {}
        self.session(mode, make_holder, self.Judge(), cells, skipped=skipped)
        assert skipped == {"nonfinite": {"harmful_hi_dev": FAULT.format("fp32")}}
        assert [judged("stock", *c) for c in cells] == [3, 0, 3, 3]

    @pytest.mark.parametrize("mode", ["concurrent", "interleave", "sequential"])
    def test_without_a_collector_the_other_cells_go_on_and_the_stale_ones_raise_at_the_end(self, setup,
                                                                                          monkeypatch, mode):
        from lift.models import generate as gen_mod
        from lift.prompts import load

        make_holder, cells = setup
        short_runs(monkeypatch)
        part = gen_mod.generations_path("stock", "harmful", "hi", "dev")
        gen_mod.generate(make_holder()["variant"], load("harmful", "hi", "dev")[:2], out_path=part)
        moved_stack(monkeypatch, torch="2.99.0+cu130")
        with pytest.raises(gen_mod.StaleOutputError) as exc:
            self.session(mode, make_holder, self.Judge(), cells)
        line = "lift restart --variant stock --split dev --sets harmful --langs hi --dtype fp32"
        assert "1 cells were skipped as stale" in str(exc.value)
        assert f"harmful_hi_dev: {line}" in str(exc.value)
        assert [judged("stock", *c) for c in cells] == [3, 0, 3, 3]

    def test_a_judge_crash_in_the_thread_surfaces(self, setup):
        make_holder, cells = setup
        with pytest.raises(RuntimeError, match="judge crashed"):
            run_cmd.run_cells(make_holder(), lambda: self.Judge(explode=True), cells, concurrent=True,
                              lid=self.LID())

    def test_both_failures_surface_when_generation_also_fails(self, setup, monkeypatch):
        from lift.models import generate as gen_mod

        make_holder, cells = setup
        calls = {"n": 0}
        real = gen_mod.generate

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                import time

                time.sleep(0.5)                  # let the judge thread fail on cell 1 first
                raise ValueError("generation died")
            return real(*a, **kw)

        monkeypatch.setattr(gen_mod, "generate", flaky)
        with pytest.raises(ValueError, match="generation died") as exc:
            run_cmd.run_cells(make_holder(), lambda: self.Judge(explode=True), cells, concurrent=True,
                              lid=self.LID())
        assert isinstance(exc.value.__cause__, RuntimeError)
