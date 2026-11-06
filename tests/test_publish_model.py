"""lift publish model: the frozen winner's card and parameters, materialised weights through
build_variant, CAST artifacts and wrapper, and the gated adapters (weights only)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from publish_fakes import write_inputs

from lift import paths
from lift.models import variants
from lift.publish import PublishError, model
from lift.publish.inputs import Facts
from lift.schemas import VariantSpec

HASH = "a" * 64
SPECS = {
    "stock": VariantSpec(name="stock", method="stock"),
    "adapter-aya-std-s1": VariantSpec(
        name="adapter-aya-std-s1",
        method="adapter",
        adapter="adapter-aya-std-s1",
        seed=1,
        artifacts={"adapter": HASH},
    ),
    "wiseft-l0.50-adapter-aya-std-s1": VariantSpec(
        name="wiseft-l0.50-adapter-aya-std-s1",
        method="wiseft",
        adapter="adapter-aya-std-s1",
        seed=1,
        params={"lambda": 0.5},
        artifacts={"adapter": HASH},
    ),
    "cast-m1.0-adapter-aya-std-s1": VariantSpec(
        name="cast-m1.0-adapter-aya-std-s1",
        method="cast",
        adapter="adapter-aya-std-s1",
        seed=1,
        params={"strength": 1.0},
        artifacts={"adapter": HASH, "cast": "b" * 64},
    ),
}


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(variants, "load_registry", lambda: dict(SPECS))
    monkeypatch.setattr(
        variants,
        "get_spec",
        lambda name: SPECS[name] if name in SPECS else (_ for _ in ()).throw(KeyError(name)),
    )
    adapter = paths.adapters_dir() / "adapter-aya-std-s1"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "registration.json").write_text(json.dumps({"pushes": [{"repo": "secret-runs-repo"}]}))
    cast = paths.artifacts_dir() / "cast" / "adapter-aya-std-s1"
    cast.mkdir(parents=True)
    (cast / "condition.pt").write_bytes(b"cond")
    (cast / "config.json").write_text("{}")


def test_without_selection_everything_is_pending():
    ups = model.build_all()
    assert ups[0].expected and "selection.json" in ups[0].expected[0]
    assert "[PENDING" in (ups[0].folder / "README.md").read_text()
    assert ups[1].expected


def test_winner_card_params_and_adapters(registry):
    write_inputs()
    ups = model.build_all()
    m, params, *adapters = ups
    assert m.repo.repo_id.endswith("/lift-wiseft-l0.50-adapter-aya-std-s1") and m.repo.gated is False
    doc = json.loads((m.folder / "repair" / "params.json").read_text())
    assert doc["params"] == {"lambda": 0.5} and doc["selection_rule"]["tie_margin"] == 0.02
    assert m.expected == ["model weights (lift publish model --build, on the GPU VM)"]
    card = (m.folder / "README.md").read_text()
    assert "license: cc-by-nc-4.0" in card and "acceptable-use policy" in card
    assert "+16.0 pp [+11.0, +21.0]" in card and "[PENDING" not in card
    # repair parameters: both frozen families, CAST artifacts copied
    assert (params.folder / "params" / "cast-m1.0-adapter-aya-std-s1.json").is_file()
    assert (params.folder / "cast" / "adapter-aya-std-s1" / "condition.pt").is_file()
    # adapters: gated (automatic approval), weights only, terms in the card
    (a,) = adapters
    assert a.repo.gated == "auto"
    assert sorted(p.name for p in a.folder.iterdir()) == [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    assert "extra_gated_prompt" in (a.folder / "README.md").read_text()


def test_variant_must_be_frozen(registry):
    write_inputs()
    with pytest.raises(PublishError, match="not a frozen variant"):
        model.build_model(Facts(), "adapter-aya-std-s1")


class _Saved:
    def __init__(self, label):
        self.label = label

    def save_pretrained(self, folder, **kwargs):
        (folder / "config.json").write_text(json.dumps({"saved": self.label}))
        (folder / "model.safetensors").write_bytes(b"w")


def _fake_build(spec, *, device, dtype):
    return SimpleNamespace(
        model=_Saved("model"),
        base=_Saved("base"),
        tokenizer=SimpleNamespace(
            save_pretrained=lambda folder: (folder / "tokenizer.json").write_text("{}")
        ),
        model_rev="rev",
        adapter_hash=HASH,
        template_hash="t",
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "name,saved", [("wiseft-l0.50-adapter-aya-std-s1", "model"), ("cast-m1.0-adapter-aya-std-s1", "base")]
)
def test_materialise_saves_what_build_variant_built(registry, tmp_path, name, saved):
    folder = model.materialise(SPECS[name], tmp_path / "m", device="cpu", build=_fake_build)
    assert json.loads((folder / "config.json").read_text()) == {"saved": saved}
    assert model.weights_present(folder) and (folder / "tokenizer.json").is_file()
    assert json.loads((folder / "lift_variant.json").read_text())["spec"]["name"] == name


def test_cast_model_carries_artifacts_and_wrapper(registry, monkeypatch):
    write_inputs()
    up = model.build_model(Facts(), "cast-m1.0-adapter-aya-std-s1")
    assert (up.folder / "cast_generate.py").is_file()
    assert (up.folder / "repair" / "cast" / "condition.pt").is_file()
    assert "Using the CAST repair" in (up.folder / "README.md").read_text()
