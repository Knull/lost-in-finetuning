"""lift.publish.acceptance (the notebook's helpers): the screening verdict, registering the frozen repair
on a new adapter, and fetching an adapter, all offline."""

from __future__ import annotations

import shutil

import pytest
import yaml

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
        ({"stock": 0.0, "a": 0.5, "r": 0.05}, "r", "repair"),
        ({"stock": 0.0, "a": 0.5, "r": 0.6}, "r", "reject"),
    ],
)
def test_screening_verdicts(rates, repaired, verdict):
    s = acceptance.summarise("stock", "a", repaired, langs=("hi",), labels=labels_for(rates))
    assert s["verdict"] == verdict
    assert "drift_hi" in s["tests"]
    md = acceptance.markdown(s)
    assert "Screening verdict" in md and "| variant | set | lang |" in md


def test_run_check_calls_run_dev():
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
