from lift import config
from lift.cli import build_parser, main


def test_configs_load_and_respect_the_dev_budget():
    for name in ("project", "train", "repairs", "variants"):
        assert isinstance(config.load(name), dict)
    budget = config.get("dev_budget_max_settings", 4, name="repairs")
    assert len(config.get("wiseft.lambdas", name="repairs")) <= budget
    assert len(config.get("resta.alphas", name="repairs")) <= budget
    assert len(config.get("cast.strengths", name="repairs")) + 1 <= budget   # + always-on
    assert config.get("resta.dare", name="repairs") is False
    assert config.get("seeds") == [1, 2, 3]


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
