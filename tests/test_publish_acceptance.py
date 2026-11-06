"""lift.publish.acceptance (the notebook's helpers): the screening verdict, registering the frozen repair
on a new adapter, and fetching an adapter, all offline."""

from __future__ import annotations

import shutil

import pytest
import yaml
from publish_fakes import no_committed_recipe  # noqa: F401

from lift import paths
from lift.models import variants
from lift.publish import PublishError, acceptance
from lift.schemas import VariantSpec

N = 40


def labels_for(rates):
    """A fake label reader: variant -> harmful share on the harmful set; hard benign is all benign."""

    def read(variant, set_name, lang):
        if set_name == "hard_benign":
            return {f"p{i}": "benign" for i in range(N)}
        k = int(rates[variant] * N)
        return {f"p{i}": "harmful" if i < k else "refusal" for i in range(N)}

    return read


@pytest.mark.parametrize(
    "rates,repaired,verdict",
    [
        ({"stock": 0.05, "a": 0.05}, None, "retain"),
        ({"stock": 0.0, "a": 0.5}, None, "repair needed"),
        ({"stock": 0.0, "a": 0.5, "r": 0.05}, "r", "repair (safety and over-refusal only)"),
        ({"stock": 0.0, "a": 0.5, "r": 0.6}, "r", "reject"),
    ],
)
def test_screening_verdicts(rates, repaired, verdict):
    s = acceptance.summarise("stock", "a", repaired, langs=("hi",), labels=labels_for(rates))
    assert s["verdict"] == verdict
    assert "drift_hi" in s["tests"]
    md = acceptance.markdown(s)
    assert "Screening verdict" in md and "| variant | set | lang |" in md


@pytest.fixture
def dev_prompts(fake_prompt_files, monkeypatch):
    """Dev prompts for every set and language the notebook reads, LangID installed, Qwen3Guard primary."""
    from fakes import make_prompt

    for set_name in acceptance.SETS:
        for lang in ("hi", "en"):
            fake_prompt_files[(set_name, lang)] = [make_prompt(f"p{i}", lang, set_name) for i in range(3)]
    monkeypatch.setattr(acceptance, "_installed", lambda module: True)
    return fake_prompt_files


def test_run_check_calls_run_dev(dev_prompts):
    seen = []
    codes = acceptance.run_check(
        ["stock", "a"], langs=("hi",), limit=7, main=lambda argv: seen.append(argv) or 0
    )
    assert codes == {"stock": 0, "a": 0}
    assert seen[0] == [
        "run-dev",
        "--variant",
        "stock",
        "--langs",
        "hi",
        "--sets",
        "harmful",
        "hard_benign",
        "--limit",
        "7",
    ]


@pytest.fixture
def registry_copy(tmp_path, monkeypatch):
    path = tmp_path / "variants.yaml"
    shutil.copyfile(paths.CONFIG_DIR / "variants.yaml", path)
    target = VariantSpec(
        name="adapter-notebook-std-s1",
        method="adapter",
        adapter="adapter-notebook-std-s1",
        seed=1,
        artifacts={"adapter": "c" * 64},
    )
    monkeypatch.setattr(variants, "get_spec", lambda name: target)
    return path


def test_register_repair_applies_the_frozen_parameters(registry_copy):
    frozen = {
        "variant": "wiseft-l0.50-adapter-aya-std-s1",
        "adapter": "adapter-aya-std-s1",
        "method": "wiseft",
        "params": {"lambda": 0.5},
        "artifacts": {"adapter": "a" * 64},
    }
    name = acceptance.register_repair("adapter-notebook-std-s1", frozen, registry=registry_copy)
    assert name == "wiseft-l0.50-adapter-notebook-std-s1"
    entries = {e["name"]: e for e in yaml.safe_load(registry_copy.read_text())["variants"]}
    assert entries[name]["params"] == {"lambda": 0.5} and entries[name]["artifacts"] == {"adapter": "c" * 64}
    assert acceptance.register_repair("adapter-notebook-std-s1", frozen, registry=registry_copy) == name


def test_cast_needs_its_own_fit(registry_copy):
    with pytest.raises(PublishError, match="fit-cast"):
        acceptance.register_repair("adapter-notebook-std-s1", {"method": "cast"}, registry=registry_copy)


def test_fetch_adapter_downloads_and_registers(tmp_path, monkeypatch):
    path = tmp_path / "variants.yaml"
    shutil.copyfile(paths.CONFIG_DIR / "variants.yaml", path)
    calls = []

    def download(*, repo_id, revision, local_dir, allow_patterns):
        calls.append((repo_id, allow_patterns))
        target = paths.adapters_dir() / "adapter-notebook-std-s1"
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text('{"r": 16, "base_model_name_or_path": "x"}')
        (target / "adapter_model.safetensors").write_bytes(b"\0" * 16)

    name = acceptance.fetch_adapter("someone/lora", download=download, registry=path)
    assert name == "adapter-notebook-std-s1" and calls[0][0] == "someone/lora"
    entries = {e["name"]: e for e in yaml.safe_load(path.read_text())["variants"]}
    assert entries[name]["method"] == "adapter" and len(entries[name]["artifacts"]["adapter"]) == 64


def test_a_prompt_some_variant_lacks_is_never_dropped_silently():
    """R1-22 (rule 3): the summary compares only prompts every variant answered; a prompt one variant
    lacks stops it, or is counted and shown when the caller allows it."""
    full = labels_for({"stock": 0.0, "a": 0.5})

    def read(variant, set_name, lang):
        out = full(variant, set_name, lang)
        if variant == "a" and set_name == "harmful":
            out.pop("p0")
            out.pop("p1")
        return out

    with pytest.raises(PublishError, match="2 prompts"):
        acceptance.summarise("stock", "a", langs=("hi",), labels=read)
    s = acceptance.summarise("stock", "a", langs=("hi",), labels=read, allow_dropped=True)
    assert s["dropped"] == {"harmful_hi": {"a": 2}}
    assert "2 prompts" in acceptance.markdown(s)


# ---------------------------------------------------------------- a fresh clone (P08-R2-06)


def test_a_fresh_clone_stops_before_any_gpu_work_and_says_how_to_get_the_prompts(fake_prompt_files):
    seen = []
    with pytest.raises(PublishError) as err:
        acceptance.run_check(["stock"], langs=("hi",), main=lambda argv: seen.append(argv) or 0)
    assert not seen  # nothing was generated
    msg = str(err.value)
    assert "harmful_hi" in msg and "hard_benign_hi" in msg
    assert "lift sync pull" in msg and "HF token" in msg and "lift data build" in msg


def test_a_prompt_file_without_dev_records_is_not_ready(fake_prompt_files, monkeypatch):
    from fakes import make_prompt

    monkeypatch.setattr(acceptance, "_installed", lambda module: True)
    for set_name in acceptance.SETS:
        fake_prompt_files[(set_name, "hi")] = [make_prompt("p0", "hi", set_name, split="final")]
    with pytest.raises(PublishError, match="no dev prompts"):
        acceptance.check_ready(("hi",))


def test_check_ready_names_the_missing_extra(dev_prompts, monkeypatch):
    monkeypatch.setattr(acceptance, "_installed", lambda module: module != "fasttext")
    with pytest.raises(PublishError, match=r"\.\[gpu,data,eval\]"):
        acceptance.check_ready(("hi", "en"))


def test_a_frontier_primary_judge_needs_its_key_and_a_spend_cap(dev_prompts, monkeypatch):
    from lift import config

    real = config.get

    def get(dotted, *args, **kwargs):
        if dotted == "judges.safety_primary":
            return {"provider": "anthropic", "model": "claude-sonnet-5-5", "rubric": "r"}
        return real(dotted, *args, **kwargs)

    monkeypatch.setattr(config, "get", get)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIFT_JUDGE_MAX_USD", raising=False)
    with pytest.raises(PublishError) as err:
        acceptance.check_ready(("hi", "en"))
    assert "ANTHROPIC_API_KEY" in str(err.value) and "LIFT_JUDGE_MAX_USD" in str(err.value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "5")
    assert acceptance.check_ready(("hi", "en")) == {"prompts": 12}


def test_check_ready_counts_the_dev_prompts(dev_prompts):
    assert acceptance.check_ready(("hi", "en")) == {"prompts": 12}
    assert acceptance.check_ready(("hi", "en"), limit=2) == {"prompts": 8}


# ---------------------------------------------------------------- the run time (P08-R1-26)


def test_the_run_time_is_recorded_from_a_measured_run(monkeypatch):
    import json

    clock = iter([1000.0, 1000.0 + 12 * 60 + 5])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    acceptance.start_clock()
    doc = acceptance.record_runtime(limit=50, langs=("hi", "en"), variants=3, gpu="NVIDIA A100-SXM4-40GB")
    assert doc["minutes"] == 13 and doc["seconds"] == pytest.approx(725.0)
    on_disk = json.loads(acceptance.runtime_path().read_text())
    assert on_disk == doc and on_disk["gpu"] == "NVIDIA A100-SXM4-40GB"


def test_record_runtime_needs_a_started_clock(monkeypatch):
    monkeypatch.setattr(acceptance, "_STARTED", None)
    with pytest.raises(PublishError, match="start_clock"):
        acceptance.record_runtime(limit=50, langs=("hi",), variants=3, gpu="x")


def test_the_notebook_says_not_yet_timed_until_a_run_is_measured(tmp_path, monkeypatch):
    from lift.publish.inputs import Facts

    nb = tmp_path / "nb.ipynb"
    nb.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    acceptance.stamp_notebook(nb)
    assert "not yet timed" in _runtime_line(nb)
    assert acceptance.runtime_text(Facts()) == "not yet timed"
    clock = iter([0.0, 600.0])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    acceptance.start_clock()
    acceptance.record_runtime(limit=50, langs=("hi", "en"), variants=3, gpu="NVIDIA A100")
    acceptance.stamp_notebook(nb)
    line = _runtime_line(nb)
    assert "10 minutes" in line and "NVIDIA A100" in line and "not yet timed" not in line
    facts = Facts()
    assert "10 minutes" in acceptance.runtime_text(facts)
    assert any(s["pointer"] == "/minutes" for s in facts.sources)


def _runtime_line(nb):
    import json

    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    lines = [line for c in cells if c["cell_type"] == "markdown" for line in c["source"]]
    return next(line for line in lines if line.startswith(acceptance.RUNTIME_MARK))
