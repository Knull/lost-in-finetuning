"""P05's commands without a model: lift select on synthetic metrics (the same answers as
tests/test_selection.py), the sweep's stages, resume and exit codes, and register-repairs. No torch."""

from __future__ import annotations

import json
import shutil

import pytest

from lift import cli, config
from lift.paths import manifests_dir
from lift.repair import select, sweep
from lift.repair import settings as rs
from lift.selection import DevScores, select_repair
from lift.train import lora

ADAPTER = "adapter-aya-std-s1"
DIGEST = "a" * 64
RULES = config.get("selection", name="repairs")
STOCK = DevScores("stock", "stock", harmful_hi=0.10, overrefusal_hard_hi=0.30, win_rate=0.5, belebele_hi=0.60)
ADAPTER_SCORES = DevScores(ADAPTER, "adapter", harmful_hi=0.30, overrefusal_hard_hi=0.25, win_rate=0.70,
                           belebele_hi=0.61)


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """configs/variants.yaml served with seed 1's adapter registered; writes go to a tmp copy."""
    doc = {"variants": [{"name": "stock", "method": "stock"},
                        {"name": ADAPTER, "method": "adapter", "adapter": ADAPTER, "seed": 1,
                         "artifacts": {"adapter": DIGEST}}]}
    real = config.load
    monkeypatch.setattr(config, "load", lambda name="project": doc if name == "variants" else real(name))
    path = tmp_path / "variants.yaml"
    shutil.copy(config.CONFIG_DIR / "variants.yaml", path)
    monkeypatch.setattr(lora, "default_registry", lambda: path)
    return path


def cand(name, method, harmful, overref=0.30, win=0.66, bele=0.60):
    return DevScores(name, method, harmful, overref, win, bele)


def fake_reader(numbers):
    """A read_scores stand-in: stock and the adapter as in test_selection, settings from ``numbers``
    ({tag: (harmful, overref, win, bele)}); a setting not listed is missing."""

    def read(name, method, dtype):
        if name == "stock":
            return STOCK, [], {}
        if name == ADAPTER:
            return ADAPTER_SCORES, [], {}
        tag = name[: -len(ADAPTER) - 1]
        if tag not in numbers:
            return None, [f"{name}: no metrics"], {}
        return cand(name, method, *numbers[tag]), [], {}

    return read


DEFAULT = (0.31, 0.30, 0.66, 0.60)       # a setting that does not lower the harmful rate


def case(**overrides):
    numbers = {s.tag: DEFAULT for s in rs.settings()}
    numbers.update(overrides)
    return numbers


CASES = {
    # test_largest_eligible_drop_wins: cast's larger drop wins overall
    "largest": case(**{"wiseft-l0.50": (0.20, 0.30, 0.66, 0.60), "cast-m1.0": (0.12, 0.30, 0.66, 0.60)}),
    # test_constraints_exclude_settings
    "constraints": case(**{"cast-m2.0": (0.05, 0.40, 0.66, 0.60), "resta-a0.40": (0.08, 0.30, 0.55, 0.60),
                           "wiseft-l0.25": (0.09, 0.30, 0.66, 0.55),
                           "wiseft-l0.50": (0.18, 0.30, 0.66, 0.60)}),
    # test_ties_go_to_the_cheaper_repair
    "ties": case(**{"cast-m1.0": (0.120, 0.30, 0.66, 0.60), "wiseft-l0.50": (0.135, 0.30, 0.66, 0.60)}),
    # test_no_eligible_setting_freezes_best_and_rejects
    "reject": case(**{"cast-m2.0": (0.05, 0.40, 0.66, 0.60)}),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_select_on_synthetic_metrics_matches_the_rule(name, registry):
    numbers = CASES[name]
    doc = select.run(ADAPTER, reader=fake_reader(numbers), date="20261107")
    cands = [cand(s.name(ADAPTER), s.method, *numbers[s.tag]) for s in rs.settings()]
    direct = select_repair(STOCK, ADAPTER_SCORES, cands, RULES)
    assert doc["status"] == "decided"
    assert doc["overall"] == {"winner": direct.winner, "verdict": direct.verdict, "eligible": direct.eligible}
    for fam in rs.FAMILIES:
        pool = [c for c in cands if rs.FAMILY[c.method] == fam]
        assert doc["families"][fam]["winner"] == select_repair(STOCK, ADAPTER_SCORES, pool, RULES).winner
    assert len(doc["table"]) == 10 and {r["family"] for r in doc["table"]} == set(rs.FAMILIES)
    assert doc["frozen"] == {f: doc["families"][f]["winner"] for f in rs.FAMILIES}
    plan = doc["freeze"]
    assert plan["tag"] == "git tag recipe-frozen-20261107 && git push --tags"
    assert len(plan["variants"]) == 9                      # 3 families x 3 seeds
    assert plan["fit_cast"] == [f"lift fit-cast --adapter adapter-aya-std-s{s}" for s in (2, 3)]
    on_disk = json.loads(select.report_paths()[0].read_text(encoding="utf-8"))
    assert on_disk["overall"]["winner"] == direct.winner
    assert direct.winner in select.report_paths()[1].read_text(encoding="utf-8")


def test_the_selection_cases_decide_as_test_selection_says(registry):
    def winner(name):
        return select.run(ADAPTER, reader=fake_reader(CASES[name]))["overall"]["winner"]

    assert winner("largest") == f"cast-m1.0-{ADAPTER}"
    sel = select.run(ADAPTER, reader=fake_reader(CASES["constraints"]))
    assert sel["overall"]["eligible"] == [f"wiseft-l0.50-{ADAPTER}"]
    assert winner("ties") == f"wiseft-l0.50-{ADAPTER}"
    rej = select.run(ADAPTER, reader=fake_reader(CASES["reject"]))
    assert rej["overall"]["verdict"] == "reject" and rej["overall"]["eligible"] == []


def test_seed_1_entries_are_written_and_later_seeds_wait(registry, monkeypatch):
    doc = select.run(ADAPTER, reader=fake_reader(CASES["ties"]), families=["wiseft"])
    items = doc["freeze"]["variants"]
    assert [i["name"] for i in items] == [f"wiseft-l0.50-adapter-aya-std-s{s}" for s in (1, 2, 3)]
    assert items[0]["entry"]["params"] == {"lambda": 0.5}
    assert items[0]["entry"]["artifacts"] == {"adapter": DIGEST}
    assert "pending" in items[1] and "pending" in items[2]
    assert doc["freeze"]["fit_cast"] == []
    assert doc["freeze"]["register"][0] == ("lift register-repairs --adapter adapter-aya-std-s2 "
                                            "--settings wiseft-l0.50")


def test_select_refuses_to_decide_with_a_missing_setting(registry, monkeypatch, capsys):
    numbers = case()
    del numbers["resta-a0.25"]
    monkeypatch.setattr(select, "read_scores", fake_reader(numbers))
    assert cli.main(["select", "--adapter", ADAPTER]) == 3
    doc = json.loads(select.report_paths()[0].read_text(encoding="utf-8"))
    assert doc["status"] == "incomplete" and list(doc["missing"]) == [f"resta-a0.25-{ADAPTER}"]
    assert "overall" not in doc
    numbers["resta-a0.25"] = DEFAULT
    assert cli.main(["select", "--adapter", ADAPTER]) == 0
    out = capsys.readouterr().out
    assert "git tag recipe-frozen-" in out and "lift fit-cast --adapter adapter-aya-std-s2" in out


# --- the sweep ------------------------------------------------------------------------------------------


def test_sweep_runs_every_stage_resumes_and_registers(registry):
    calls = []

    def call(argv):
        calls.append(list(argv))
        return 3 if (argv[0] == "belebele" and "resta-a0.10" in argv[1]) else 0

    rc = sweep.gpu(ADAPTER, families=["wiseft", "resta"], device="cuda:0", judge_device="cuda:0", dtype=None,
                   batch_size=None, call=call)
    assert rc == 3
    assert [c[0] for c in calls[:3]] == ["run-dev", "utility", "belebele"]
    assert calls[0][1] == f"--variant=wiseft-l0.25-{ADAPTER}" and "--generate-only" in calls[1]
    assert "--langs=hi" in calls[2] and "--split=dev" in calls[2]
    assert len(calls) == 6 * 3
    # The entries were registered (resta pins Base by its revision: the test config's is "main", so the
    # base stand-in is never reached: specs_for read the pinned sha below).
    text = registry.read_text(encoding="utf-8")
    assert f"wiseft-l0.75-{ADAPTER}" in text and f"resta-a0.40-{ADAPTER}" in text
    state = sweep.read_state(ADAPTER)
    assert state["settings"][f"resta-a0.10-{ADAPTER}"]["stages"]["belebele"]["rc"] == 3
    # Resume: only the stage that did not finish runs again.
    calls.clear()
    assert sweep.gpu(ADAPTER, families=["wiseft", "resta"], device="cuda:0", judge_device="cuda:0",
                     dtype=None, batch_size=None, call=lambda argv: calls.append(list(argv)) or 0) == 0
    assert calls == [["belebele", f"--variant=resta-a0.10-{ADAPTER}", "--split=dev", "--langs=hi",
                      "--device=cuda:0"]]
    manifests = list(manifests_dir().glob("sweep_setting-*.json"))
    assert len(manifests) == 12
    calls.clear()
    assert sweep.gpu(ADAPTER, families=["wiseft"], device="cuda:0", judge_device="cuda:0", dtype=None,
                     batch_size=None, recheck=True, call=lambda argv: calls.append(list(argv)) or 1) == 1
    assert len(calls) == 9


@pytest.fixture(autouse=True)
def pinned_base(monkeypatch):
    view = config.load("project")
    monkeypatch.setitem(view["models"]["base"], "revision", "d" * 40)


def test_sweep_judge_only_runs_utility_for_every_setting(registry):
    calls = []
    rc = sweep.judge_only(ADAPTER, families=["cast"], no_wait=True,
                          call=lambda argv: calls.append(list(argv)) or 3)
    assert rc == 3 and len(calls) == 4
    assert calls[-1] == ["utility", f"--variant=castalways-m1.0-{ADAPTER}", "--split=dev", "--judge-only",
                         "--no-wait"]


def test_register_repairs_command(registry, capsys):
    assert cli.main(["register-repairs", "--adapter", ADAPTER, "--repairs", "wiseft,resta"]) == 0
    assert "commit configs/variants.yaml" in capsys.readouterr().out
    assert cli.main(["register-repairs", "--adapter", ADAPTER, "--settings", "wiseft-l0.50"]) == 0
    assert "unchanged" in capsys.readouterr().out
    # CAST needs its artifact first.
    assert cli.main(["register-repairs", "--adapter", ADAPTER, "--repairs", "cast"]) == 2
    assert "lift fit-cast" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="register-adapter"):
        cli.main(["register-repairs", "--adapter", "adapter-aya-std-s9"])


def test_cost_summary_reads_the_gate_apart(registry):
    from lift.io import write_jsonl
    from lift.models.generate import generations_path

    rows = []
    for i in range(4):
        rows.append({"key": f"k{i}", "variant": "v", "pid": f"p{i}", "lang": "hi", "set": "harmful",
                     "split": "dev", "prompt_sha": "s", "output": "", "n_prompt_tokens": 3, "n_new_tokens": 5,
                     "hit_cap": False, "seconds": 1.0,
                     "meta": {"dtype": "bf16", "n_generated": 10, "gate_seconds": 0.25, "gate_fired": i < 1}})
    path = generations_path("v", "harmful", "hi", "dev")
    path.parent.mkdir(parents=True)
    write_jsonl(path, rows)
    gen = sweep.cost_summary("v", "bf16")["generation"]
    assert gen["n"] == 4 and gen["seconds_per_generated_token"] == pytest.approx(0.1)
    assert gen["gate_share"] == pytest.approx(0.25) and gen["gate_fire_rate"] == pytest.approx(0.25)
    assert sweep.cost_summary("v", "fp16")["generation"]["n"] == 0


def test_sweep_refuses_cast_without_its_artifact(registry, capsys):
    view = config.load("project")
    view["models"]["global"]["revision"] = "e" * 40
    assert cli.main(["sweep", "--adapter", ADAPTER, "--repairs", "cast"]) == 2
    assert "lift fit-cast" in capsys.readouterr().err


def test_read_scores_names_every_missing_input_with_its_remedy(registry):
    scores, missing, _ = select.read_scores(f"wiseft-l0.50-{ADAPTER}", "wiseft", "bf16")
    assert scores is None and len(missing) == 1 and "register-repairs" in missing[0]   # not registered
    doc = config.load("variants")
    doc["variants"].append(rs.entry(rs.specs_for(ADAPTER, [rs.Setting("wiseft", 0.5)])[0]))
    scores, missing, _ = select.read_scores(f"wiseft-l0.50-{ADAPTER}", "wiseft", "bf16")
    assert scores is None and [m.split(":")[0] for m in missing] == ["harmful_hi", "overrefusal_hard_hi",
                                                                      "win_rate", "belebele_hi"]
    assert "lift run-dev" in missing[0] and "--judge-only" in missing[2] and "lift belebele" in missing[3]
    scores, missing, details = select.read_scores(ADAPTER, "adapter", "bf16")
    assert [m.split(":")[0] for m in missing] == ["harmful_hi", "overrefusal_hard_hi", "win_rate"]
    assert "note" in details["belebele_hi"]           # the adapter's Belebele is not required


def test_select_refuses_safety_and_belebele_cells_of_a_replaced_cast_entry(registry, monkeypatch):
    """After lift fit-cast --refit and register-repairs --replace, cells made under the old entry are
    stale: lift select names each with its restart line instead of selecting from them."""
    from lift.cachekey import intervention_digest
    from lift.io import write_jsonl
    from lift.judge import pipeline
    from lift.models import loglik
    from lift.models.generate import generations_path

    params = {"layer": 2, "threshold": 0.1, "comparator": "smaller", "pool": "last", "behavior_layers": [2]}
    old, new = (rs.spec_for(rs.Setting("cast", 1.0), ADAPTER, cast=(params, c * 64)) for c in "12")
    name = new.name
    config.load("variants")["variants"].append(rs.entry(new))          # the refit artifact, registered
    # The other inputs read fine (each has its own tests): only the cells' entry is in question here.
    monkeypatch.setattr(pipeline, "paired_labels", lambda *a, **k: ({"p0": "refusal"}, {"p0": "refusal"}))
    monkeypatch.setattr(loglik, "belebele_metrics", lambda *a, **k: {
        "langs": {"hi": {"accuracy": 0.6, "n": 1}}, "adapter_hash": DIGEST, "model_rev": "r" * 40})
    monkeypatch.setattr(select, "_win_rate", lambda *a, **k: (0.66, {}))

    def cells(spec, intervention=True):
        meta = {"dtype": "bf16", **({"intervention": intervention_digest(spec)} if intervention else {})}
        for set_name in ("harmful", "hard_benign"):
            path = generations_path(name, set_name, "hi", "dev")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(path, [{"key": "k0", "variant": name, "pid": "p0", "lang": "hi", "set": set_name,
                                "split": "dev", "prompt_sha": "s", "output": "", "n_prompt_tokens": 1,
                                "n_new_tokens": 1, "hit_cap": False, "seconds": 0.1, "meta": meta}])
        path = loglik.belebele_scores_path(name, "hi", "dev", dtype="bf16")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, [{"key": "k0", "pid": "p0", "task": "belebele", "meta": meta}])

    cells(old)
    scores, missing, _ = select.read_scores(name, "cast", "bf16")
    assert scores is None and [m.split(":")[0] for m in missing] == ["harmful_hi", "overrefusal_hard_hi",
                                                                      "belebele_hi"]
    assert f"lift restart --variant {name} --split dev --sets harmful --langs hi" in missing[0]
    assert "--sets hard_benign" in missing[1] and "another registry entry" in missing[0]
    assert f"lift belebele --variant {name} --split dev --langs hi --restart" in missing[2]
    cells(new, intervention=False)          # a repair's records must name the entry they were made under
    assert select.read_scores(name, "cast", "bf16")[0] is None
    cells(new)
    scores, missing, _ = select.read_scores(name, "cast", "bf16")
    assert missing == [] and scores is not None and scores.win_rate == 0.66
    # Stock's and the adapter's records from before records named their entry are read as they are.
    stock_cell = generations_path("stock", "harmful", "hi", "dev")
    stock_cell.parent.mkdir(parents=True)
    write_jsonl(stock_cell, [{"meta": {"dtype": "bf16"}}])
    assert select._stale("stock", stock_cell) is None
    write_jsonl(stock_cell, [{"meta": {"dtype": "bf16", "intervention": "0" * 64}}])
    assert "another registry entry" in select._stale("stock", stock_cell)
