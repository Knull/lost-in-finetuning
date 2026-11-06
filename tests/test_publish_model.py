"""lift publish model: the frozen winner's card and parameters, materialised weights through
build_variant, CAST artifacts and wrapper, and the gated adapters (weights only)."""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

import pytest
import yaml
from publish_fakes import (  # noqa: F401
    CAST_WINS,
    REJECT,
    FakeApi,
    frozen_variants,
    no_committed_recipe,
    selection,
    write_inputs,
)

from lift import paths
from lift.models import variants
from lift.publish import PublishError, model, settings
from lift.publish.inputs import Facts
from lift.schemas import VariantSpec

HASH = "a" * 64
PARAMS = {"wiseft": "lambda", "resta": "alpha", "cast": "strength", "castalways": "strength"}
# ReSta wins: its dev drop is the largest eligible one.
RESTA_WINS = {"wiseft-l0.50": (0.20, 0.30, 0.66, 0.60), "resta-a0.25": (0.12, 0.30, 0.66, 0.60)}


def _cast_bytes(adapter: str) -> bytes:
    return f"cast artifact of {adapter}".encode()


def _cast_sha(adapter: str) -> str:
    return hashlib.sha256(_cast_bytes(adapter)).hexdigest()


def _spec(name: str) -> VariantSpec:
    prefix = name.split("-")[0]
    method = "cast_always" if prefix == "castalways" else prefix
    key = PARAMS[prefix]
    value = float(name.split("-")[1][1:])
    adapter = name.split("-", 2)[2]
    arts = {"adapter": HASH, **({"cast": _cast_sha(adapter)} if method.startswith("cast") else {})}
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
for _doc in (selection(), selection(CAST_WINS), selection(REJECT), selection(RESTA_WINS)):
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
        (cast / "cast.safetensors").write_bytes(_cast_bytes(f"adapter-aya-std-s{seed}"))
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
    assert m.repo.repo_id.endswith("/lift-wiseft-l0.50-adapter-aya-std-s1") and m.repo.gated == "auto"
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
    assert (params.folder / "cast" / "adapter-aya-std-s1" / "cast.safetensors").is_file()
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


def _meta(card: str) -> dict:
    return yaml.safe_load(card.split("---\n")[1])


@pytest.mark.parametrize(
    "numbers,why",
    [
        (None, "lambda"),  # WiSE-FT: adapter = Global + (these - Global) / lambda
        (RESTA_WINS, "alpha"),  # ReSta: adapter = these - alpha x (Global - Base)
        (CAST_WINS, "wrapper"),  # CAST: these weights are the adapter, merged
        (REJECT, "reject"),
    ],
)
def test_every_repaired_model_is_gated(registry, numbers, why):
    """R1-04, R2-07, decisions.md 74: with its public parameters, any repaired model's weights give back
    the gated adapter (WiSE-FT and ReSta by arithmetic, CAST directly), so every one is gated like it."""
    doc = selection(numbers)
    write_inputs(selection_doc=doc)
    up = model.build_model(Facts())
    assert up.repo.repo_id.endswith("/lift-" + doc["overall"]["winner"])
    assert up.repo.gated == "auto"
    card = (up.folder / "README.md").read_text()
    assert "extra_gated_prompt" in card and why in card
    assert "gated-auto" not in card
    assert any("item 74" in n for n in up.notes)


def test_a_public_model_repo_is_refused(registry, monkeypatch):
    """R2-07: configs/publish.yaml cannot make the repaired model public while item 74 stands."""
    cfg = settings.load()
    cfg = {**cfg, "repos": {**cfg["repos"], "model": {**cfg["repos"]["model"], "access": "public"}}}
    monkeypatch.setattr(settings, "load", lambda: cfg)
    write_inputs()
    with pytest.raises(PublishError, match="item 74"):
        model.build_model(Facts())


@pytest.mark.parametrize(
    "numbers,says,never",
    [
        (None, "lambda", "wrapper"),
        (RESTA_WINS, "alpha", "wrapper"),
        (CAST_WINS, "wrapper", "lambda"),
        (REJECT, "reject", "lambda"),
    ],
)
def test_gate_prompt_gives_this_models_reasons(registry, numbers, says, never):
    """R2-10: the gate text is filled from gate_reasons: no wrapper sentence on a WiSE-FT or ReSta model."""
    write_inputs(selection_doc=selection(numbers))
    up = model.build_model(Facts())
    prompt = _meta((up.folder / "README.md").read_text())["extra_gated_prompt"]
    assert says in prompt and never not in prompt
    assert "Do not deploy it" in prompt and "{{" not in prompt


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
    assert (up.folder / "repair" / "cast" / "cast.safetensors").is_file()
    assert "Using the CAST repair" in (up.folder / "README.md").read_text()
    assert not any("CAST artifact" in e for e in up.expected)


@pytest.mark.parametrize("stale", ["other bytes", None])
def test_cast_artifact_must_match_the_registry_sha(registry, stale):
    """R2-08: a stale (refitted, or not pulled) or missing cast.safetensors is never staged beside a
    params.json that pins another sha; both repos list it as expected instead."""
    write_inputs(selection_doc=selection(CAST_WINS))
    art = model.cast_artifacts("adapter-aya-std-s1") / "cast.safetensors"
    if stale is None:
        art.unlink()
    else:
        art.write_bytes(stale.encode())
    up = model.build_model(Facts())
    assert any("cast.safetensors" in e and "sha256" in e for e in up.expected)
    assert not (up.folder / "repair" / "cast").exists()
    params = model.build_params(Facts())
    assert any("cast.safetensors" in e and "sha256" in e for e in params.expected)
    assert not (params.folder / "cast" / "adapter-aya-std-s1").exists()
    assert (params.folder / "cast" / "adapter-aya-std-s2" / "cast.safetensors").is_file()


def test_model_card_values_are_recorded(registry):
    """R2-09: every value the model card renders goes to outputs/publish/model.sources.json."""
    write_inputs()
    model.build_model(Facts())
    doc = json.loads((model.stage_root() / "model.sources.json").read_text())
    pointers = {v["pointer"] for v in doc["values"]}
    assert {"/primary/drift/pooled", "/primary/repair/pooled", "/headline/utility_retained"} <= pointers
    assert any(p.startswith("/tables/per_language/rows/") for p in pointers)
    assert any(p.startswith("/tables/belebele/rows/") for p in pointers)
    assert "/generation/max_new_tokens" in pointers
    assert doc["pending"] == []


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


def test_a_winner_changed_after_lift_stats_publishes_nothing(registry):
    """XB-04: selection.json's winner changed after lift stats. The card would print WiSE-FT's repair
    effect under CAST's name, so lift publish model refuses (exit 1) and stages no card."""
    from lift import cli
    from lift.publish.inputs import selection_path

    write_inputs()
    selection_path().write_text(json.dumps(selection(CAST_WINS)), encoding="utf-8")
    with pytest.raises(PublishError, match="lift stats"):
        model.build_all()
    assert cli.main(["publish", "model"]) == 1
    assert not (model.stage_root() / "model" / "cast-m1.0-adapter-aya-std-s1" / "README.md").exists()


def test_an_adapter_not_pulled_is_listed_as_missing_and_blocks_yes(registry, monkeypatch):
    """XB-05: every registered adapter goes to HF (decisions item 14). One not pulled locally is in the
    plan with its files missing, so the dry run shows it and --yes refuses; the collections list it."""
    from lift.publish import collect, hub

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    absent = VariantSpec(
        name="adapter-aya-std-s2", method="adapter", adapter="adapter-aya-std-s2", seed=2, artifacts={}
    )
    monkeypatch.setattr(variants, "load_registry", lambda: {**SPECS, absent.name: absent})
    write_inputs()
    ups = model.build_adapters(Facts())
    by = {u.repo.repo_id.rsplit("/", 1)[1]: u for u in ups}
    assert set(by) == {"lift-adapter-aya-std-s1", "lift-adapter-aya-std-s2"}
    s2 = by["lift-adapter-aya-std-s2"]
    assert s2.expected and all("lift sync pull" in e for e in s2.expected)
    assert {e.split()[0].rsplit("/", 1)[1] for e in s2.expected} == set(model.ADAPTER_FILES)
    assert any("missing until built" in line for line in s2.describe())
    with pytest.raises(PublishError, match="adapter adapter-aya-std-s2.*not built yet"):
        hub.check_ready([s2], windows=set())
    repos = [r.repo_id for r in collect.plan()["models"]]
    assert "test-org/lift-adapter-aya-std-s2" in repos and "test-org/lift-adapter-aya-std-s1" in repos


# ------------------------------------------------------- the recipe the newest tag froze (review XB2-02)


@pytest.fixture
def article_out(tmp_path, monkeypatch):
    """hf/article.md, its sources and the notebook written under tmp_path, never into the repo."""
    from lift.publish import acceptance, article

    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
    notebook = tmp_path / "acceptance_test.ipynb"
    notebook.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(acceptance, "notebook_path", lambda: notebook)
    return tmp_path


def _reselect_after_the_tag(numbers):
    """XB2-02's probe: after the final, ``lift select --refreeze`` wrote another winner to selection.json
    and configs/frozen_recipe.json (and ``lift stats``/``report`` followed it); nothing was tagged."""
    from publish_fakes import frozen_recipe, write_recipe, write_report

    doc = selection(numbers)
    write_report(selection_doc=doc, frozen=False)
    write_recipe(frozen_recipe(doc))
    return doc


def test_a_reselection_after_the_tag_publishes_nothing(registry, monkeypatch, capsys, article_out):
    """XB2-02: the tag still names the old winner; a recipe that differs from the tag's refuses the
    model, the parameters, the logs, the collections and the article, dry runs included."""
    from lift import cli
    from lift.publish import article, collect, datasets

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs()
    assert not any(u.blocked for u in model.build_all())
    _reselect_after_the_tag(RESTA_WINS)
    for build in (model.build_all, collect.plan, article.build, lambda: datasets.build_logs(Facts())):
        with pytest.raises(PublishError, match="recipe-frozen-20261107"):
            build()
    for argv in (["publish", "model"], ["publish", "collections"], ["publish", "article"]):
        assert cli.main(argv) == 1
    assert "not the recipe the tag recipe-frozen-20261107 froze" in capsys.readouterr().err
    assert not (model.stage_root() / "model" / "resta-a0.25-adapter-aya-std-s1" / "README.md").exists()


def test_the_newest_tag_decides_and_must_hold_the_recipe(registry):
    from publish_fakes import TAG, frozen_recipe, tag_recipe, write_report

    from lift.publish.inputs import frozen_winner, recipe_block

    write_inputs()
    tag_recipe("recipe-frozen-20261106", text='{"version": 1}')  # an older freeze: not the newest
    assert recipe_block(Facts()) is None and frozen_winner(Facts()) == selection()["overall"]["winner"]
    # line endings do not count (a Windows checkout)
    tag_recipe(TAG, text=json.dumps(frozen_recipe()).replace("\n", "\r\n"))
    assert recipe_block(Facts()) is None
    # a later freeze with its own dated tag (and a decisions entry) is the one published
    doc = selection(RESTA_WINS)
    write_report(selection_doc=doc, frozen=False)
    from publish_fakes import write_recipe

    write_recipe(frozen_recipe(doc))
    tag_recipe("recipe-frozen-20261108")
    assert frozen_winner(Facts()) == "resta-a0.25-adapter-aya-std-s1"
    # a tag on a commit without the recipe
    tag_recipe("recipe-frozen-20261109", text=None)
    from lift.publish import inputs

    inputs._git.tags["recipe-frozen-20261109"] = None
    with pytest.raises(PublishError, match="cannot be read at recipe-frozen-20261109"):
        model.build_all()


@pytest.mark.parametrize("state", ["no recipe", "no tag"])
def test_without_a_tagged_recipe_the_plans_are_printed_blocked_and_yes_refuses(
    registry, monkeypatch, capsys, tmp_path, article_out, state
):
    """XB2-02: before the freeze is committed and tagged, the dry runs still stage and print their plan,
    marked BLOCKED; --yes refuses every plan that names the repaired model (with --allow-pending too),
    the collections are not created, the article is a draft marked blocked (exit 1) and the notebook's
    WINNER is not stamped. The adapters name no winner and are not blocked."""
    from publish_fakes import frozen_recipe, tag_recipe, write_recipe

    from lift import cli
    from lift.publish import acceptance, article, collect, datasets, hub

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs(frozen=False)
    if state == "no tag":
        write_recipe(frozen_recipe())
    ups = model.build_all()
    by = {u.title.split(" (")[0]: u for u in ups}
    why = "no configs/frozen_recipe.json" if state == "no recipe" else "no recipe-frozen-\\* tag"
    for title in ("repaired model", "repair parameters"):
        assert by[title].blocked and re.search(why, by[title].blocked[0])
        assert any(line.startswith("  BLOCKED: ") for line in by[title].describe())
        with pytest.raises(PublishError, match="blocked"):
            hub.check_ready([by[title]], allow_pending=True, windows=set())
    assert not by["adapter adapter-aya-std-s1"].blocked
    assert datasets.build_logs(Facts()).blocked
    with pytest.raises(PublishError, match="blocked"):
        collect.execute(collect.plan(), api_factory=FakeApi, log=lambda _: None)
    assert cli.main(["publish", "model"]) == 0  # the dry run prints the plan
    assert cli.main(["publish", "collections"]) == 0
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "lift-wiseft-l0.50-adapter-aya-std-s1" in out
    assert cli.main(["publish", "model", "--yes", "--allow-pending"]) == 1
    assert cli.main(["publish", "article"]) == 1
    assert "BLOCKED" in capsys.readouterr().err
    text = article.article_path().read_text(encoding="utf-8")
    title = next(i for i, line in enumerate(text.splitlines()) if line.startswith("# "))
    assert "BLOCKED" in text.splitlines()[title + 2]
    assert any("BLOCKED" in p for p in article.placeholders(text))
    nb = tmp_path / "nb.ipynb"
    nb.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    assert "WINNER" not in acceptance.stamp_notebook(nb)
    # the freeze is committed and tagged: nothing is blocked any more
    if state == "no recipe":
        write_recipe(frozen_recipe())
    tag_recipe()
    assert not any(u.blocked for u in model.build_all())
    assert "WINNER" in acceptance.stamp_notebook(nb)
    assert cli.main(["publish", "article"]) == 0
    assert "BLOCKED" not in article.article_path().read_text(encoding="utf-8")


def test_staged_model_files_name_no_absolute_local_path(registry):
    """XB2-04: an adapter_config.json that names the base model by its Hugging Face cache snapshot goes up
    with the repo id; one under a home dir that is no such snapshot is withheld (the build refuses)."""
    write_inputs()
    config = paths.adapters_dir() / "adapter-aya-std-s1" / "adapter_config.json"
    snapshot = "/root/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/" + "ab12" * 10
    config.write_text(json.dumps({"base_model_name_or_path": snapshot, "r": 16}))
    (a,) = model.build_adapters(Facts())
    staged = json.loads((a.folder / "adapter_config.json").read_text())
    assert staged == {"base_model_name_or_path": "CohereLabs/tiny-aya-global", "r": 16}
    assert any("adapter_config.json" in n for n in a.notes)
    config.write_text(json.dumps({"base_model_name_or_path": "/home/someone/models/aya"}))
    with pytest.raises(PublishError, match="adapter_config.json holds 1 absolute local path"):
        model.build_adapters(Facts())
