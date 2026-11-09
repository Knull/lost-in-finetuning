import pytest

from lift import config
from lift.cli import build_parser, main

REAL_LOAD = config.load          # at import, before conftest's autouse fixture serves its views


def test_configs_load_and_respect_the_dev_budget():
    for name in ("project", "train", "repairs", "variants"):
        assert isinstance(config.load(name), dict)
    budget = config.get("dev_budget_max_settings", 4, name="repairs")
    assert len(config.get("wiseft.lambdas", name="repairs")) <= budget
    assert len(config.get("resta.alphas", name="repairs")) <= budget
    assert len(config.get("cast.strengths", name="repairs")) + 1 <= budget   # + always-on
    assert config.get("resta.dare", name="repairs") is False
    assert config.get("seeds") == [1, 2, 3]


def test_the_committed_registry_parses(monkeypatch):
    """conftest serves configs/variants.yaml with its stock entries only; the file itself, with every
    adapter and repair a run registered, must still load: one entry per name, each a valid spec."""
    from lift.models import variants

    monkeypatch.setattr(config, "load", REAL_LOAD)
    registry = variants.load_registry()
    assert "stock" in registry
    for name, spec in registry.items():
        assert spec.name == name
        if spec.method == "adapter":
            assert spec.adapter == name and len(spec.artifacts["adapter"]) == 64


def test_build_variant_refuses_a_builder_that_loaded_other_files_than_declared():
    """R1-05: every hash a registry entry declares (``spec.artifacts``) must be the one its builder loaded
    under that key; a declared file the builder never loaded is refused too. What a builder loads but the
    entry does not declare (an unpinned adapter in a test) is left to the reporting door."""
    from lift.models import variants
    from lift.schemas import VariantSpec

    spec = VariantSpec(name="cast-m1.0-adapter-aya-std-s1", method="cast", adapter="adapter-aya-std-s1",
                       seed=1, params={"strength": 1.0}, artifacts={"adapter": "a" * 64, "cast": "c" * 64})
    variants.check_loaded(spec, {"adapter": "a" * 64, "cast": "c" * 64, "base": "ignored"})
    with pytest.raises(ValueError, match=r"artifacts\.cast: declared cccccccccccc, loaded dddddddddddd"):
        variants.check_loaded(spec, {"adapter": "a" * 64, "cast": "d" * 64})
    with pytest.raises(ValueError, match=r"artifacts\.cast: declared cccccccccccc, loaded nothing under"):
        variants.check_loaded(spec, {"adapter": "a" * 64})
    unpinned = VariantSpec(name="adapter-aya-std-s1", method="adapter", adapter="adapter-aya-std-s1", seed=1)
    variants.check_loaded(unpinned, {"adapter": "a" * 64})


def test_generate_and_run_dev_need_base_pinned_for_a_resta_variant(monkeypatch):
    """R1-04: a ReSta variant's lines need models.base pinned, as Global is, before torch loads."""
    from lift.cli import main as lift_main

    name = "resta-a0.25-adapter-aya-std-s1"
    doc = {"variants": [{"name": "stock", "method": "stock"},
                        {"name": name, "method": "resta", "adapter": "adapter-aya-std-s1", "seed": 1,
                         "params": {"alpha": 0.25, "dare": False, "base": "CohereLabs/tiny-aya-base",
                                    "base_revision": "b" * 40}, "artifacts": {"adapter": "a" * 64}}]}
    served = config.load
    monkeypatch.setattr(config, "load", lambda name_="project": doc if name_ == "variants" else served(name_))
    config.load("project")["models"]["global"]["revision"] = "a" * 40
    config.load("project")["judges"]["safety_primary"]["revision"] = "c" * 40
    for argv in (["generate", "--variant", name, "--split", "dev"], ["run-dev", "--variant", name]):
        with pytest.raises(SystemExit, match=r"models\.base\.revision = 'main'"):
            lift_main(argv)
    from lift.commands import run

    assert run._base_pin(name) == ["models.base"]
    assert run._base_pin("stock") == [] and run._base_pin("no-such-variant") == []   # refused later, by name


def test_locked_generation_settings():
    gen = config.get("generation")
    assert gen["do_sample"] is False and gen["max_new_tokens"] == 256
    assert config.get("languages.target") == "hi"
    assert set(config.get("languages.all")) == {"en", "hi", "zh", "es", "ar", "th"}


def test_cli_lists_commands_and_info_runs(capsys):
    parser = build_parser()
    assert "info" in parser.format_help()
    assert main(["info"]) == 0
    assert "config_digest" in capsys.readouterr().out
