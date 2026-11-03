import pytest

from lift.cli import build_parser, main
from lift.commands import run as run_cmd


@pytest.mark.parametrize("command", ["smoke", "generate", "judge", "run-dev", "sync", "remote"])
def test_help_works_without_a_gpu_stack(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_names_accept_commas_and_spaces_and_reject_unknowns():
    assert run_cmd._names(["hi,en", "zh"], ["en", "hi", "zh"], ["en"]) == ["hi", "en", "zh"]
    assert run_cmd._names(None, ["en", "hi"], ["en", "hi"]) == ["en", "hi"]
    with pytest.raises(SystemExit):
        run_cmd._names(["xx"], ["en"], ["en"])


def test_primary_judge_refuses_subsets():
    parser = build_parser()
    args = parser.parse_args(["judge", "--variant", "stock", "--split", "dev", "--sample", "0.1"])
    with pytest.raises(SystemExit, match="second judge"):
        args.func(args)
    args = parser.parse_args(["judge", "--variant", "stock", "--split", "dev", "--judge", "second",
                              "--sample", "0.1", "--keys", "k.txt"])
    with pytest.raises(SystemExit, match="not both"):
        args.func(args)


def test_study_commands_refuse_unpinned_revisions():
    # configs/project.yaml still says revision: main until Gate 0's shas are pinned.
    parser = build_parser()
    for argv in (["generate", "--variant", "stock", "--split", "dev"], ["run-dev", "--variant", "stock"]):
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit, match="unpinned revisions"):
            args.func(args)


def test_require_pinned_accepts_a_full_sha(monkeypatch):
    from lift import config

    real = config.get
    monkeypatch.setattr(config, "get", lambda key, *a, **k: "a" * 40 if key.endswith(".revision")
                        else real(key, *a, **k))
    run_cmd.require_pinned("models.global", "judges.safety_primary")


def test_judge_keys_listed_in_no_generations_file_fail_loudly(monkeypatch):
    from lift.io import write_jsonl
    from lift.paths import generations_dir

    path = generations_dir() / "stock" / "harmful_hi_dev.jsonl"
    rec = {"key": "a" * 64, "variant": "stock", "pid": "p1", "lang": "hi", "set": "harmful", "split": "dev",
           "prompt_sha": "s", "output": "o", "n_prompt_tokens": 1, "n_new_tokens": 1, "hit_cap": False,
           "seconds": 0.1, "meta": {}}
    write_jsonl(path, [rec])
    keys = generations_dir().parent / "keys.txt"
    keys.write_text("a" * 64 + "\n" + "b" * 64 + "\n", encoding="utf-8")
    monkeypatch.setattr("lift.judge.pipeline.judge_dir", lambda role: "frontier")
    args = build_parser().parse_args(["judge", "--variant", "stock", "--split", "dev", "--judge", "second",
                                      "--sets", "harmful", "--langs", "hi", "--keys", str(keys)])
    with pytest.raises(SystemExit, match="1 listed keys are in none"):
        args.func(args)


def test_a_refused_auditkit_record_is_reported_not_fatal(monkeypatch, caplog):
    def refuse(*a, **kw):
        raise ValueError("3 generations are not in bf16")

    monkeypatch.setattr("lift.audit.record_safety", refuse)
    monkeypatch.setattr(run_cmd, "_complete", lambda *a: True)
    refused: list[str] = []
    run_cmd._record("stock", "harmful", "hi", "dev", dtype="bf16", refused=refused)
    assert refused == ["harmful_hi_dev"] and "not in bf16" in caplog.text
    assert run_cmd._refused_exit(refused) == 1 and run_cmd._refused_exit([]) == 0


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

        def __init__(self, explode=False):
            self.explode = explode

        def judge(self, items):
            from lift.judge.pipeline import JudgeOutcome

            if self.explode:
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
