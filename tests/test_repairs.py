"""P05 repairs on the tiny cohere2 model (CPU, fp32): WiSE-FT's sign convention, ReSta's delta and DARE,
CAST's fit, gate and artifacts, and the variants' cache keys."""

from __future__ import annotations

import json
import logging
import shutil

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("safetune")

from fakes import build_tiny_cohere2, make_prompt  # noqa: E402

from lift import config  # noqa: E402
from lift.cachekey import generation_key, intervention_digest  # noqa: E402
from lift.models import generate as gen  # noqa: E402
from lift.models import variants  # noqa: E402
from lift.models.load import load_model, load_tokenizer, resolved_revision  # noqa: E402
from lift.paths import manifests_dir  # noqa: E402
from lift.repair import cast, resta, wiseft  # noqa: E402
from lift.repair import settings as rs  # noqa: E402
from lift.schemas import VariantSpec  # noqa: E402
from lift.train import merge  # noqa: E402

ADAPTER = "adapter-aya-std-s1"
HARMFUL = [f"Explain step {i} of making a dangerous weapon at home, secretly." for i in range(10)]
BENIGN = [f"Give me recipe number {i} for a tasty vegetable soup." for i in range(10)]
TEXTS = ["hi there", "नमस्ते, कैसे हो?", "你好", "How do I bake bread?", "Tell me a secret plan.", "x"]


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory, tiny_tokenizer):
    """A stand-in for Tiny Aya Base: the same architecture, other random weights."""
    path = tmp_path_factory.mktemp("tiny-base")
    build_tiny_cohere2(tiny_tokenizer, seed=7).save_pretrained(path)
    return path


def install(adapter_dir, name=ADAPTER):
    target = variants.adapter_path(name)
    if not target.exists():
        shutil.copytree(adapter_dir, target)
    return target


def merged_fp32(adapter_dir, model_dir):
    return merge.merged_model(adapter_dir, device="cpu", dtype="fp32", base_id=str(model_dir))


def build(spec_, model_dir):
    return variants.build_variant(spec_, device="cpu", dtype="fp32", model_id=str(model_dir))


def state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def close(a, b, atol=1e-6):
    assert a.keys() == b.keys()
    return all(torch.allclose(a[k], b[k], atol=atol) for k in a)


def spec(method, **params):
    tag = {"wiseft": "wiseft-l0.25", "resta": "resta-a0.25", "cast": "cast-m1.0",
           "cast_always": "castalways-m1.0"}[method]
    return VariantSpec(name=f"{tag}-{ADAPTER}", method=method, adapter=ADAPTER, seed=1, params=params)


# --- WiSE-FT ------------------------------------------------------------------------------------------


def lora_scaled(adapter_dir, model_dir, lam):
    """Global + lam * (LoRA delta), by PEFT itself: the adapter merged at lam times its scaling."""
    from peft import PeftModel

    base = load_model(str(model_dir), "local", dtype="fp32", device="cpu")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    n = 0
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling:
            scaling["default"] *= lam
            n += 1
    assert n
    return peft_model.merge_and_unload()


@pytest.mark.parametrize("lam", [0.25, 0.75])
def test_wiseft_equals_the_lora_delta_scaled_by_lambda(tiny_lora_adapter, tiny_model_dir, lam):
    merged = merged_fp32(tiny_lora_adapter, tiny_model_dir)
    adapter_weights = state(merged)
    stock = load_model(str(tiny_model_dir), "local", dtype="fp32", device="cpu")
    wiseft.make(merged, stock, lam)
    expected = state(lora_scaled(tiny_lora_adapter, tiny_model_dir, lam))
    got = state(merged)
    assert close(got, expected, atol=1e-5)
    # The sign convention: lambda 0.25 sits nearer stock, 0.75 nearer the adapter.
    stock_w = state(stock)
    d_stock = sum(float((got[k] - stock_w[k]).norm()) for k in got)
    d_adapter = sum(float((got[k] - adapter_weights[k]).norm()) for k in got)
    assert (d_stock < d_adapter) == (lam < 0.5)
    assert wiseft.alpha_for(lam) == pytest.approx(1 - lam)


def test_wiseft_interpolates_the_largest_tensors_on_the_cpu(tiny_lora_adapter, tiny_model_dir, monkeypatch):
    """apply_wise_ft holds five fp32 copies of a tensor on its device at once: for Tiny Aya's 262k x 2048
    embedding that is 10 GiB beside the 6.2 GiB fp16 model, over a T4's 15 GiB. The largest tensors are
    interpolated on the CPU (the same elementwise arithmetic) and put back where they were, ties kept."""
    import safetune.interventions.recover.wise_ft as st_wise

    merged = merged_fp32(tiny_lora_adapter, tiny_model_dir)
    stock = load_model(str(tiny_model_dir), "local", dtype="fp32", device="cpu")
    sizes = {n: p.numel() for n, p in merged.named_parameters()}
    biggest = max(sizes.values())
    monkeypatch.setattr(wiseft, "LARGE_NUMEL", biggest - 1)
    big = {n for n, k in sizes.items() if k == biggest}
    assert big == set(wiseft.large_parameters(merged))
    seen = []
    real = st_wise.apply_wise_ft

    def spy(model, aligned, alpha):
        seen.append({n: p.device.type for n, p in model.named_parameters() if n in big})
        return real(model, aligned, alpha=alpha)

    monkeypatch.setattr(st_wise, "apply_wise_ft", spy)
    tied = merged.get_output_embeddings().weight is merged.get_input_embeddings().weight
    wiseft.make(merged, stock, 0.5)
    assert seen == [{n: "cpu" for n in big}]
    assert (merged.get_output_embeddings().weight is merged.get_input_embeddings().weight) == tied
    assert close(state(merged), state(lora_scaled(tiny_lora_adapter, tiny_model_dir, 0.5)), atol=1e-5)


def test_wiseft_builder_is_the_same_repair(tiny_lora_adapter, tiny_model_dir):
    install(tiny_lora_adapter)
    v = variants.build_variant(spec("wiseft", **{"lambda": 0.25}), device="cpu", dtype="fp32",
                               model_id=str(tiny_model_dir))
    assert v.adapter_hash == merge.adapter_hash(tiny_lora_adapter) and v.base is None
    assert close(state(v.model), state(lora_scaled(tiny_lora_adapter, tiny_model_dir, 0.25)), atol=1e-5)
    with pytest.raises(ValueError, match="lambda"):
        variants.build_variant(spec("wiseft", **{"lambda": 1.5}), device="cpu", dtype="fp32",
                               model_id=str(tiny_model_dir))


# --- ReSta ----------------------------------------------------------------------------------------------


def resta_spec(base_dir, alpha=0.25):
    return spec("resta", alpha=alpha, dare=False, base=str(base_dir),
                base_revision=resolved_revision(str(base_dir), "x"))


def test_resta_adds_alpha_times_global_minus_base_and_a_reload_undoes_it(tiny_lora_adapter, tiny_model_dir,
                                                                         base_dir, monkeypatch):
    import safetune.interventions.recover.resta as st_resta

    calls = []
    real = st_resta.apply_resta

    def spy(finetuned, **kwargs):
        calls.append(kwargs)
        return real(finetuned, **kwargs)

    def never(*a, **k):
        raise AssertionError("DARE ran")

    monkeypatch.setattr(st_resta, "apply_resta", spy)
    monkeypatch.setattr(st_resta, "_dare_drop_and_rescale", never)
    install(tiny_lora_adapter)
    pristine = state(merged_fp32(tiny_lora_adapter, tiny_model_dir))
    stock = state(load_model(str(tiny_model_dir), "local", dtype="fp32", device="cpu"))
    base = state(load_model(str(base_dir), "local", dtype="fp32", device="cpu"))
    v = build(resta_spec(base_dir), tiny_model_dir)
    got = state(v.model)
    expected = {k: pristine[k] + 0.25 * (stock[k] - base[k]) for k in pristine}
    assert close(got, expected, atol=1e-5)
    assert calls and all(c["dare"] is False and c["device"] == "cpu" and c["alpha"] == 0.25 for c in calls)
    # Every setting starts from pristine weights: a second build adds the delta once, not twice, and a fresh
    # merge is the adapter again.
    again = build(resta_spec(base_dir), tiny_model_dir)
    assert close(state(again.model), expected, atol=1e-5)
    fresh = state(merged_fp32(tiny_lora_adapter, tiny_model_dir))
    assert close(fresh, pristine, atol=0)


def test_dare_is_never_enabled(tiny_model_dir, base_dir, monkeypatch):
    model = load_model(str(tiny_model_dir), "local", dtype="fp32", device="cpu")
    ref = load_model(str(base_dir), "local", dtype="fp32", device="cpu")
    view = config.load("repairs")
    monkeypatch.setitem(view["resta"], "dare", True)
    with pytest.raises(ValueError, match="DARE"):
        resta.make(model, ref, ref, 0.25)
    monkeypatch.setitem(view["resta"], "dare", False)
    with pytest.raises(ValueError, match="alpha"):
        resta.make(model, ref, ref, 1.0)
    with pytest.raises(ValueError, match="dare"):
        variants.build_variant(spec("resta", alpha=0.25, dare=True, base=str(base_dir), base_revision="x"),
                               device="cpu", dtype="fp32", model_id=str(tiny_model_dir))


# --- CAST -----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(tmp_path_factory, tiny_model_dir):
    model = load_model(str(tiny_model_dir), "local", dtype="fp32", device="cpu")
    tok = load_tokenizer(str(tiny_model_dir), "local")
    out = tmp_path_factory.mktemp("cast")
    path = cast.fit(model, tok, HARMFUL, BENIGN, out_dir=out, meta={"adapter": "x"})
    return model, tok, path


def test_cast_fit_returns_a_condition_and_round_trips(fitted, tmp_path):
    model, tok, path = fitted
    cond, vecs, info = cast.load(path)
    assert info["condition_layer"] in (1, 2, 3) and info["comparator"] in ("larger", "smaller")
    assert sorted(vecs) == cast.behavior_layers(4) == info["behavior_layers"] == [2]
    assert cond.threshold == info["threshold"] and info["pool"] == "last"
    assert rs.read_cast_info(path) == info                       # the laptop reads it without torch
    # Stable hashes: the same fit, and the same tensors saved again, give the same bytes.
    digest = cast.artifact_hash(path)
    again = cast.fit(model, tok, HARMFUL, BENIGN, out_dir=tmp_path / "a", meta={"adapter": "x"})
    assert cast.artifact_hash(again) == digest
    copy = cast.save(tmp_path / "b" / cast.ARTIFACT, cond.condition_vector, vecs, info)
    assert cast.artifact_hash(copy) == digest
    other = cast.save(tmp_path / "c" / cast.ARTIFACT, cond.condition_vector, vecs,
                      {**info, "threshold": 0.123})
    assert cast.artifact_hash(other) != digest


def test_the_fit_reads_the_tokens_generation_sees(tiny_tokenizer):
    from safetune._refusal_helpers import _encode_prompts

    from lift.models.load import encode_chat

    for text in TEXTS:
        ids = _encode_prompts(tiny_tokenizer, [text], True)["input_ids"][0]
        assert list(ids) == encode_chat(tiny_tokenizer, text)


def test_heldout_indices_match_safetune_split():
    assert cast.heldout_indices(10) == [0, 2, 4, 6, 8]
    assert cast.heldout_indices(7) == [0, 1, 2, 3]       # SafeTune's own quirk: round(3.5) = 4, step 7 // 4
    assert cast.heldout_indices(1560)[:3] == [0, 2, 4] and len(cast.heldout_indices(1560)) == 780
    assert cast.heldout_indices(1) == [0]


def split_threshold(fitted, tmp_path):
    """An artifact whose threshold fires on about half of TEXTS (the fit's own fires on none or all)."""
    model, tok, path = fitted
    cond, vecs, info = cast.load(path)
    gated = cast.make(model, path, 1.0, pad_id=tok.pad_token_id)
    sims = sorted(m["gate_sim"] for m in cast.firing(gated, tok, TEXTS, batch_size=1))
    mid = (sims[2] + sims[3]) / 2
    return cast.save(tmp_path / cast.ARTIFACT, cond.condition_vector, vecs,
                     {**info, "threshold": mid, "comparator": "smaller"})


def test_the_gate_fires_per_prompt_and_steers_only_those(fitted, tmp_path):
    model, tok, _ = fitted
    path = split_threshold(fitted, tmp_path)
    gated = cast.make(model, path, 8.0, pad_id=tok.pad_token_id)
    alone = [m["gate_fired"] for m in cast.firing(gated, tok, TEXTS, batch_size=1)]
    batched = cast.firing(gated, tok, TEXTS, batch_size=len(TEXTS))
    assert [m["gate_fired"] for m in batched] == alone and 0 < sum(alone) < len(TEXTS)
    assert all(m["gate_seconds"] >= 0 for m in batched)
    # Generation through the harness's decoder: gated rows differ from the plain model, the others do not.
    from lift.models.load import encode_chat, stop_token_ids

    ids = [encode_chat(tok, t) for t in TEXTS]
    stops = stop_token_ids(tok, model)
    kw = {"pad_id": tok.pad_token_id, "stop_ids": stops, "max_new_tokens": 6, "device": "cpu"}
    plain = gen.greedy_batch(model, ids, **kw)
    steered = gen.greedy_batch(gated, ids, **kw)
    assert [m["gate_fired"] for m in gated.batch_meta()] == alone
    for fired, a, b in zip(alone, plain, steered, strict=True):
        if not fired:
            assert a.ids == b.ids
    assert any(a.ids != b.ids for fired, a, b in zip(alone, plain, steered, strict=True) if fired)
    # Scoring: each row gated on its own prompt, the fired rows' logits steered.
    rows = [encode_chat(tok, t) + [5, 6] for t in TEXTS]
    lengths = [len(encode_chat(tok, t)) for t in TEXTS]
    input_ids, mask = cast._left_pad(rows, tok.pad_token_id, "cpu")
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=3).logits
        got = gated.score_logits(input_ids, mask, position_ids=pos, prompt_lengths=lengths, logits_to_keep=3)
    assert [m["gate_fired"] for m in gated.batch_meta()] == alone
    for i, fired in enumerate(alone):
        assert torch.allclose(got[i], ref[i], atol=1e-5) != fired
    # Always on: every row steered.
    always = cast.make(model, path, 8.0, gate=False, pad_id=tok.pad_token_id)
    with torch.no_grad():
        on = always.score_logits(input_ids, mask, position_ids=pos, prompt_lengths=lengths, logits_to_keep=3)
    assert all(not torch.allclose(on[i], ref[i], atol=1e-5) for i in range(len(TEXTS)))
    assert all(m["gate_fired"] and m["gate_always_on"] for m in always.batch_meta())
    out = gen.greedy_batch(always, ids, **kw)
    assert len(always.batch_meta()) == len(out) == len(TEXTS)
    # Nothing is left installed on the model after a call.
    with torch.no_grad():
        after = model(input_ids=input_ids, attention_mask=mask, position_ids=pos, logits_to_keep=3).logits
    assert torch.allclose(after, ref, atol=0)


def cast_registry(monkeypatch, tiny_lora_adapter, digest_cast, extra=()):
    digest = merge.adapter_hash(tiny_lora_adapter)
    doc = {"variants": [{"name": "stock", "method": "stock"},
                        {"name": ADAPTER, "method": "adapter", "adapter": ADAPTER, "seed": 1,
                         "artifacts": {"adapter": digest}}, *extra]}
    real = config.load

    def load(name="project"):
        return doc if name == "variants" else real(name)

    monkeypatch.setattr(config, "load", load)
    return doc


def test_cast_variant_through_the_harness(fitted, tiny_lora_adapter, tiny_model_dir, tmp_path, monkeypatch,
                                         caplog):
    model, tok, _ = fitted
    install(tiny_lora_adapter)
    digest = merge.adapter_hash(tiny_lora_adapter)
    merged = merged_fp32(tiny_lora_adapter, tiny_model_dir)
    rev = resolved_revision(str(tiny_model_dir), "x")
    path = cast.fit(merged, tok, HARMFUL, BENIGN, out_dir=cast.artifact_dir(ADAPTER),
                    meta={"adapter": ADAPTER, "adapter_hash": digest, "model_rev": rev, "dtype": "fp32"})
    cast_registry(monkeypatch, tiny_lora_adapter, None)
    s_gated, s_always = rs.specs_for(ADAPTER, [rs.Setting("cast", 1.0), rs.Setting("cast_always", 1.0)])
    assert s_gated.artifacts == {"adapter": digest, "cast": cast.artifact_hash(path)}
    with caplog.at_level(logging.WARNING, logger="lift.repair"):
        v = variants.build_variant(s_gated, device="cpu", dtype="fp32", model_id=str(tiny_model_dir))
    assert "fitted in" not in caplog.text                     # fitted in the run's precision: no warning
    assert v.base is not None and v.score_logits is not None and v.hf_model is v.base
    assert cast.load(path)[2]["dtype"] == "fp32"
    out = tmp_path / "g.jsonl"
    prompts = [make_prompt(f"p-{i}", text=t) for i, t in enumerate(TEXTS[:4])]
    gen.generate(v, prompts, out_path=out, max_new_tokens=4, batch_size=2)
    recs = gen.read_generations(out)
    assert len(recs) == 4
    assert all(isinstance(r.meta["gate_fired"], bool) and "gate_seconds" in r.meta for r in recs)
    from lift.models import loglik

    logits, metas, _ = loglik._forward(v, [encode(tok, t) for t in TEXTS[:3]], [3, 3, 3], 1)
    assert logits.shape[:2] == (3, 1) and all("gate_fired" in m for m in metas)
    va = variants.build_variant(s_always, device="cpu", dtype="fp32", model_id=str(tiny_model_dir))
    gen.generate(va, prompts, out_path=tmp_path / "a.jsonl", max_new_tokens=4, batch_size=2)
    assert all(r.meta["gate_always_on"] for r in gen.read_generations(tmp_path / "a.jsonl"))
    assert all(r.meta["intervention"] == intervention_digest(s_always)
               for r in gen.read_generations(tmp_path / "a.jsonl"))
    # An artifact fitted in another precision than the run's builds, with a warning naming both: its cells
    # are keyed by the run's dtype and the artifact's hash, and paired only within one dtype.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lift.repair"):
        variants.build_variant(s_gated, device="cpu", dtype="bf16", model_id=str(tiny_model_dir))
    assert "fitted in fp32" in caplog.text and "bf16" in caplog.text and "--refit" in caplog.text
    # A changed artifact is refused: its hash is pinned.
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match"):
        variants.build_variant(s_gated, device="cpu", dtype="fp32", model_id=str(tiny_model_dir))


def encode(tok, text):
    from lift.models.load import encode_chat

    return encode_chat(tok, text)


# --- Settings, cache keys -------------------------------------------------------------------------------


def all_specs(adapter=ADAPTER, digest="a" * 64, seed=1):
    a = VariantSpec(name=adapter, method="adapter", adapter=adapter, seed=seed, artifacts={"adapter": digest})
    cast_info = ({"layer": 2, "threshold": 0.1, "comparator": "smaller", "pool": "last",
                  "behavior_layers": [2]}, "c" * 64)
    return a, [rs.spec_for(s, adapter, adapter_spec_=a, cast=cast_info, base=("base-id", "b" * 40))
               for s in rs.settings()]


def test_every_setting_has_its_own_cache_key():
    a, specs = all_specs()
    assert [s.name for s in specs] == [
        f"{t}-{ADAPTER}" for t in ("wiseft-l0.25", "wiseft-l0.50", "wiseft-l0.75", "resta-a0.10",
                                   "resta-a0.25", "resta-a0.40", "cast-m0.5", "cast-m1.0", "cast-m2.0",
                                   "castalways-m1.0")]
    stock = VariantSpec(name="stock", method="stock")
    _, other_seed = all_specs("adapter-aya-std-s2", "b" * 64, 2)
    everything = [stock, a, *specs, *other_seed]
    digests = {intervention_digest(s) for s in everything}
    assert len(digests) == len(everything)
    keys = {generation_key(model_rev="r" * 40, adapter_hash=None if s.method == "stock" else "h", spec=s,
                           template_hash="t", gen_cfg={}, pid="p", lang="hi", prompt_text="x")
            for s in everything}
    assert len(keys) == len(everything)
    for s in specs:
        assert rs.parse_name(s.name) == (rs.Setting(s.method, list(s.params.values())[0]), ADAPTER)
    assert specs[3].params["dare"] is False


def test_register_writes_entries_that_load_back(tmp_path, monkeypatch):
    from lift.train import lora

    registry = tmp_path / "variants.yaml"
    shutil.copy(config.CONFIG_DIR / "variants.yaml", registry)
    _, specs = all_specs()
    assert set(rs.register(specs, registry=registry).values()) == {"added"}
    import yaml

    loaded = lora._registry_specs(yaml.safe_load(registry.read_text(encoding="utf-8")), registry)
    assert all(loaded[s.name] == s for s in specs) and "stock" in loaded
    assert set(rs.register(specs, registry=registry).values()) == {"unchanged"}
    changed = specs[7].model_copy(update={"artifacts": {"adapter": "a" * 64, "cast": "d" * 64}})
    with pytest.raises(lora.RegistryConflict):
        rs.register([changed], registry=registry)
    assert rs.register([changed], registry=registry, replace=True) == {changed.name: "replaced"}
    loaded = lora._registry_specs(yaml.safe_load(registry.read_text(encoding="utf-8")), registry)
    assert loaded[changed.name] == changed and loaded[specs[8].name] == specs[8]


def test_settings_follow_the_config():
    names = [s.tag for s in rs.settings(["cast"])]
    assert names == ["cast-m0.5", "cast-m1.0", "cast-m2.0", "castalways-m1.0"]
    with pytest.raises(ValueError, match="unknown"):
        rs.settings(["dare"])
    assert json.loads(json.dumps(rs.entry(all_specs()[1][0])))["note"]


# --- lift fit-cast --------------------------------------------------------------------------------------


def fit_store(store, n=4, salt=""):
    for lang in ("en", "hi", "zh", "es", "ar", "th"):
        store[("harmful", lang)] = [make_prompt(f"prh-train-{i:04d}", lang, "harmful", "train",
                                                HARMFUL[i] + salt) for i in range(n)]
        store[("easy_benign", lang)] = [make_prompt(f"prb-train-{i:04d}", lang, "easy_benign", "train",
                                                    BENIGN[i]) for i in range(n)]
        store[("hard_benign", lang)] = [make_prompt(f"orb-{i:04d}", lang, "hard_benign", "fit", TEXTS[i])
                                        for i in range(n)]


def test_lift_fit_cast(fake_prompt_files, tiny_lora_adapter, tiny_model_dir, monkeypatch, capsys):
    from lift import cli

    install(tiny_lora_adapter)
    cast_registry(monkeypatch, tiny_lora_adapter, None)
    view = config.load("project")
    monkeypatch.setitem(view["models"]["global"], "id", str(tiny_model_dir))
    monkeypatch.setitem(view["models"]["global"], "revision", "a" * 40)
    fit_store(fake_prompt_files)
    line = ["fit-cast", "--adapter", ADAPTER, "--device", "cpu", "--dtype", "fp32"]
    assert cli.main(line) == 0
    out = capsys.readouterr().out
    assert "held-out F1" in out and "  hi: harmful" in out
    path = cast.artifact_path(ADAPTER)
    info = rs.read_cast_info(path)
    assert info["adapter_hash"] == merge.adapter_hash(tiny_lora_adapter) and info["dtype"] == "fp32"
    assert info["n_harmful"] == 24 and info["n_benign"] == 48
    report = json.loads((path.parent / cast.REPORT).read_text(encoding="utf-8"))
    assert report["sha256"] == cast.artifact_hash(path)
    firing = report["firing_heldout"]
    assert sorted(firing["harmful"]) == sorted(firing["benign"]) == sorted(config.get("languages.all"))
    held = sum(c["n"] for lang in firing["harmful"].values() for c in lang.values())
    assert held == 12                                                         # the held-out half
    assert set(firing["benign"]["hi"]) == {"easy_benign", "hard_benign"}
    assert report["counts"]["harmful"]["harmful_hi_train"] == 4
    assert report["safetune_config"]["legacy_cast_gate"] is False
    (manifest,) = manifests_dir().glob("fit_cast-*.json")             # XB2-04: no absolute local path
    assert json.loads(manifest.read_text(encoding="utf-8"))["artifact"] == (
        f"outputs/artifacts/cast/{ADAPTER}/{cast.ARTIFACT}")
    digest = cast.artifact_hash(path)
    assert cli.main(line) == 0 and "kept" in capsys.readouterr().out          # a rerun keeps it
    # One fitted on another Global revision (which the CAST builder refuses) is not kept either.
    cond, vecs, info = cast.load(path)
    original = path.read_bytes()
    cast.save(path, cond.condition_vector, vecs, {**info, "model_rev": "f" * 40})
    assert cli.main(line) == 2 and "--refit" in capsys.readouterr().err
    path.write_bytes(original)
    fit_store(fake_prompt_files, salt=" now")                                 # other prompts: refused
    assert cli.main(line) == 2
    assert cast.artifact_hash(path) == digest
    assert cli.main([*line, "--refit"]) == 0 and cast.artifact_hash(path) != digest


def test_hidden_states_only_leaves_the_model_as_it_was(fitted):
    model, tok, _ = fitted
    ids = torch.tensor([encode(tok, "hello")])
    full = model(input_ids=ids).logits
    with cast._hidden_states_only(model):
        assert model(input_ids=ids).logits.shape[1] == 1
    assert "forward" not in vars(model)
    assert torch.equal(model(input_ids=ids).logits, full)
