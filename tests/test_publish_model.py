"""lift publish model: the frozen winner's card and parameters, materialised weights through
build_variant, CAST artifacts and wrapper, and the gated adapters (weights only)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from publish_fakes import CAST_WINS, REJECT, FakeApi, frozen_variants, selection, write_inputs

from lift import paths
from lift.models import variants
from lift.publish import PublishError, model
from lift.publish.inputs import Facts
from lift.schemas import VariantSpec

HASH = "a" * 64
PARAMS = {"wiseft": "lambda", "resta": "alpha", "cast": "strength", "castalways": "strength"}


def _spec(name: str) -> VariantSpec:
    prefix = name.split("-")[0]
    method = "cast_always" if prefix == "castalways" else prefix
    key = PARAMS[prefix]
    value = float(name.split("-")[1][1:])
    adapter = name.split("-", 2)[2]
    arts = {"adapter": HASH, **({"cast": "b" * 64} if method.startswith("cast") else {})}
    return VariantSpec(
        name=name, method=method, adapter=adapter, seed=int(adapter[-1]), params={key: value}, artifacts=arts
    )


SPECS = {
    "stock": VariantSpec(name="stock", method="stock"),
    "adapter-aya-std-s1": VariantSpec(
        name="adapter-aya-std-s1",
        method="adapter",
        adapter="adapter-aya-std-s1",
        seed=1,
        artifacts={"adapter": HASH},
    ),
}
for _doc in (selection(), selection(CAST_WINS), selection(REJECT)):
    SPECS.update({n: _spec(n) for n in frozen_variants(_doc)})


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
    for seed in (1, 2, 3):  # CAST is fitted per seed on the adapter being repaired
        cast = paths.artifacts_dir() / "cast" / f"adapter-aya-std-s{seed}"
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
    assert m.expected == ["model weights (no weights staged: lift publish model --build, on the GPU VM)"]
    card = (m.folder / "README.md").read_text()
    assert "license: cc-by-nc-4.0" in card and "acceptable-use policy" in card
    assert "+16.0 pp [+11.0, +21.0]" in card and "[PENDING" not in card
    # repair parameters: every family's frozen setting on seeds 1-3 (R1-06), CAST artifacts copied
    assert params.expected == []
    assert sorted(p.stem for p in (params.folder / "params").iterdir()) == sorted(
        frozen_variants(selection())
    )
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
    with pytest.raises(PublishError, match="overall winner"):
        model.build_model(Facts(), "adapter-aya-std-s1")


@pytest.mark.parametrize("name", ["cast-m1.0-adapter-aya-std-s1", "wiseft-l0.50-adapter-aya-std-s2"])
def test_only_the_overall_winner_is_the_repaired_model(registry, name):
    """R1-05: the card prints the winner's repair effect, so no other frozen variant goes up under it."""
    write_inputs()
    with pytest.raises(PublishError, match="overall winner"):
        model.build_model(Facts(), name)


def test_incomplete_selection_publishes_no_model(registry):
    write_inputs(selection_doc=selection(incomplete=True))
    up = model.build_model(Facts())
    assert any("incomplete" in e for e in up.expected)
    assert model.build_params(Facts()).expected


@pytest.mark.parametrize("numbers,why", [(CAST_WINS, "CAST"), (REJECT, "reject")])
def test_cast_or_reject_winner_model_is_gated(registry, numbers, why):
    """R1-04, decisions.md 74: a CAST winner's repo is the adapter-merged model (the unrepaired adapter
    without the wrapper), and a reject verdict's best-anyway setting is not a repair: both gated."""
    doc = selection(numbers)
    write_inputs(selection_doc=doc)
    up = model.build_model(Facts())
    assert up.repo.repo_id.endswith("/lift-" + doc["overall"]["winner"])
    assert up.repo.gated == "auto"
    card = (up.folder / "README.md").read_text()
    assert "extra_gated_prompt" in card and why in card
    assert "gated-auto" not in card


def test_repair_winner_model_is_public(registry):
    write_inputs()
    assert model.build_model(Facts()).repo.gated is False


def test_collections_list_the_winner_repo_whatever_its_access(registry, monkeypatch):
    from lift.publish import collect

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs(selection_doc=selection(CAST_WINS))
    plan = collect.plan()
    assert plan["models"][0].repo_id == "test-org/lift-cast-m1.0-adapter-aya-std-s1"
    assert plan["models"][0].gated == "auto"


def test_collections_check_every_repo_exists_before_creating_anything(registry, monkeypatch):
    """R1-27: a repo not uploaded yet would leave a half-built collection."""
    from lift.publish import collect

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs()
    plan = collect.plan()
    api = FakeApi(missing={"test-org/lift-raw-completions"})
    with pytest.raises(PublishError, match="lift-raw-completions"):
        collect.execute(plan, api_factory=lambda: api, log=lambda _: None)
    assert "create_collection" not in api.names() and "add_collection_item" not in api.names()


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
        model_rev="main",  # the revision tests see pinned (conftest unpinned_revisions)
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
    write_inputs(selection_doc=selection(CAST_WINS))
    up = model.build_model(Facts(), "cast-m1.0-adapter-aya-std-s1")
    assert (up.folder / "cast_generate.py").is_file()
    assert (up.folder / "repair" / "cast" / "condition.pt").is_file()
    assert "Using the CAST repair" in (up.folder / "README.md").read_text()


# ------------------------------------------------------------------ stale weights (R1-08, R1-25)


def test_built_weights_upload_without_building_again(registry, monkeypatch):
    """R1-25: --build once (dry run), then --yes without --build uploads the same weights."""
    write_inputs()
    name = "wiseft-l0.50-adapter-aya-std-s1"
    folder = model.stage_root() / "model" / name
    model.materialise(SPECS[name], folder, device="cpu", build=_fake_build)
    up = model.build_model(Facts())
    assert up.expected == [] and (up.folder / "model.safetensors").is_file()


def test_weights_of_another_spec_are_not_uploaded(registry, monkeypatch):
    """R1-08: weights materialised for an earlier spec (re-frozen params, another adapter) never go up
    beside a newer card."""
    write_inputs()
    name = "wiseft-l0.50-adapter-aya-std-s1"
    folder = model.stage_root() / "model" / name
    old = SPECS[name].model_copy(update={"params": {"lambda": 0.75}})
    model.materialise(old, folder, device="cpu", build=_fake_build)
    up = model.build_model(Facts())
    assert any("model weights" in e for e in up.expected)
    assert not (up.folder / "model.safetensors").exists()


def test_weights_of_another_adapter_hash_are_not_uploaded(registry):
    write_inputs()
    name = "wiseft-l0.50-adapter-aya-std-s1"
    folder = model.stage_root() / "model" / name

    def other_hash(spec, *, device, dtype):
        lv = _fake_build(spec, device=device, dtype=dtype)
        lv.adapter_hash = "c" * 64
        return lv

    model.materialise(SPECS[name], folder, device="cpu", build=other_hash)
    assert any("model weights" in e for e in model.build_model(Facts()).expected)


def test_build_clears_the_staging_folder(registry, monkeypatch):
    write_inputs()
    name = "wiseft-l0.50-adapter-aya-std-s1"
    folder = model.stage_root() / "model" / name
    folder.mkdir(parents=True)
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"old shard")
    (folder / "repair").mkdir()
    (folder / "repair" / "left-over.pt").write_bytes(b"x")
    orig = model.materialise
    monkeypatch.setattr(
        model, "materialise", lambda spec, f, **k: orig(spec, f, device="cpu", build=_fake_build)
    )
    up = model.build_model(Facts(), build_weights=True, device="cpu")
    files = sorted(p.relative_to(up.folder).as_posix() for p in up.files())
    assert "model-00002-of-00002.safetensors" not in files and "repair/left-over.pt" not in files
    assert up.expected == [] and "model.safetensors" in files


def test_adapter_gate_does_not_assert_drift(registry):
    """R1-18: Gate 2 may find no drift, so the gate heading is hedged."""
    (a,) = model.build_adapters(Facts())
    card = (a.folder / "README.md").read_text()
    assert "it drifted" not in card and "may affect safety" in card and "safety-drift" not in card
