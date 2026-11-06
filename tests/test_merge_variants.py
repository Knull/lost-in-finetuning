import json
import shutil

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fakes import TINY_CHAT_TEMPLATE, build_tiny_tokenizer, make_prompt  # noqa: E402

from lift import config  # noqa: E402
from lift.io import sha256_file  # noqa: E402
from lift.models import generate as gen  # noqa: E402
from lift.models import variants  # noqa: E402
from lift.schemas import VariantSpec  # noqa: E402
from lift.train import merge  # noqa: E402

TEXTS = ["hi", "नमस्ते", "你好，世界", "A longer English prompt about the weather."]


def install_adapter(adapter_dir, name="adapter-aya-std-s1"):
    """Copy the tiny adapter where the harness looks for adapters (outputs/adapters/{name})."""
    target = variants.adapter_path(name)
    shutil.copytree(adapter_dir, target)
    return target


def adapter_spec(name="adapter-aya-std-s1", **artifacts):
    return VariantSpec(name=name, method="adapter", adapter=name, seed=1, artifacts=artifacts)


def test_adapter_hash_covers_weights_and_behaviour_config(tiny_lora_adapter, tmp_path):
    h = merge.adapter_hash(tiny_lora_adapter)
    assert h == merge.adapter_hash(tiny_lora_adapter)
    copy = tmp_path / "copy"
    shutil.copytree(tiny_lora_adapter, copy)
    cfg = json.loads((copy / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = "/some/other/machine/path"          # not behaviour: same hash
    cfg["target_modules"] = list(reversed(cfg["target_modules"]))
    (copy / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert merge.adapter_hash(copy) == h
    cfg["lora_alpha"] = cfg["lora_alpha"] * 2                          # behaviour: new hash
    (copy / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert merge.adapter_hash(copy) != h
    weights = copy / "adapter_model.safetensors"
    assert sha256_file(weights) == sha256_file(tiny_lora_adapter / "adapter_model.safetensors")
    with pytest.raises(FileNotFoundError):
        merge.adapter_hash(tmp_path / "empty")


def test_merged_adapter_matches_unmerged(tiny_lora_adapter, tiny_model_dir):
    out = merge.parity_check(tiny_lora_adapter, prompts=TEXTS, device="cpu", dtype="fp32",
                             base_id=str(tiny_model_dir), max_new_tokens=10, batch_size=2)
    assert out["all_match"] and out["n"] == len(TEXTS) and out["n_match"] == len(TEXTS)
    assert out["first_divergence"] == [None] * len(TEXTS)
    assert out["adapter_hash"] == merge.adapter_hash(tiny_lora_adapter)


def test_the_tiny_adapter_changes_outputs(tiny_lora_adapter, tiny_model_dir):
    from lift.models.load import load_model, load_tokenizer

    tok = load_tokenizer(str(tiny_model_dir), "local")
    base = load_model(str(tiny_model_dir), "local", dtype="fp32", device="cpu")
    merged = merge.merged_model(tiny_lora_adapter, device="cpu", dtype="fp32", base_id=str(tiny_model_dir))
    a = gen.greedy_token_ids(base, tok, TEXTS, device="cpu", max_new_tokens=10)
    b = gen.greedy_token_ids(merged, tok, TEXTS, device="cpu", max_new_tokens=10)
    assert [x.ids for x in a] != [x.ids for x in b]


def test_build_adapter_variant_records_the_hash(tiny_lora_adapter, tiny_model_dir, tmp_path):
    install_adapter(tiny_lora_adapter)
    v = variants.build_variant(adapter_spec(), device="cpu", dtype="fp32", model_id=str(tiny_model_dir))
    assert v.adapter_hash == merge.adapter_hash(tiny_lora_adapter)
    assert v.dtype == "fp32" and v.model_rev.startswith("local-")
    path = tmp_path / "g.jsonl"
    gen.generate(v, [make_prompt("p-1", text="hi")], out_path=path, max_new_tokens=4)
    stock = variants.build_variant(VariantSpec(name="stock", method="stock"), device="cpu", dtype="fp32",
                                   model_id=str(tiny_model_dir))
    cfg = gen.gen_config(stock, max_new_tokens=4)
    assert gen.read_generations(path)[0].key != gen.record_key(stock, make_prompt("p-1", text="hi"), cfg)


def test_pinned_hash_and_template_are_enforced(tiny_lora_adapter, tiny_model_dir):
    target = install_adapter(tiny_lora_adapter)
    with pytest.raises(ValueError, match="pinned"):
        variants.build_variant(adapter_spec(adapter="0" * 64), device="cpu", dtype="fp32",
                               model_id=str(tiny_model_dir))
    # The trainer saved a tokenizer whose template differs from Global's: refused.
    other = build_tiny_tokenizer(TINY_CHAT_TEMPLATE.replace("contextual safety mode", "another mode"))
    other.save_pretrained(target)
    with pytest.raises(ValueError, match="renders differently"):
        variants.build_variant(adapter_spec(), device="cpu", dtype="fp32", model_id=str(tiny_model_dir))
    # Global's own tokenizer saved beside the adapter: accepted.
    build_tiny_tokenizer().save_pretrained(target)
    assert variants.build_variant(adapter_spec(), device="cpu", dtype="fp32",
                                  model_id=str(tiny_model_dir)).adapter_hash


def test_an_adapter_whose_fp32_merge_parity_failed_is_never_built(tiny_lora_adapter, tiny_model_dir):
    # CLAUDE.md rule 9 at the one code path: the adapter's registration.json (lift train's record, synced with
    # it) records a failed fp32 merge parity for this very hash, so it is refused whatever the registry pins.
    target = install_adapter(tiny_lora_adapter)
    digest = merge.adapter_hash(target)
    record = target / "registration.json"

    def write(parity, adapter_hash=digest):
        record.write_text(json.dumps({"name": "adapter-aya-std-s1", "adapter_hash": adapter_hash,
                                      "parity": parity}), encoding="utf-8")

    failed = {"fp32": {"n": 20, "n_match": 19, "all_match": False},
              "fp16": {"n": 20, "n_match": 20, "all_match": True}}         # the fp32 result decides
    write(failed)
    refused = r"failed fp32 merge parity: it must not be evaluated \(CLAUDE.md rule 9\)"
    for spec in (adapter_spec(), adapter_spec(adapter=digest)):
        with pytest.raises(ValueError, match=refused):
            variants.build_variant(spec, device="cpu", dtype="fp32", model_id=str(tiny_model_dir))
    write(failed, adapter_hash="0" * 64)              # written for another adapter (set aside): not this one
    assert variants.checked_adapter(adapter_spec()) == (target, digest)
    for parity in ({"fp32": {"n": 20, "n_match": 20, "all_match": True}},
                   {"fp16": {"n": 20, "n_match": 3, "all_match": False}},     # a T4's: recorded, not required
                   {}):                                                       # not checked: not a failure
        write(parity)
        assert variants.checked_adapter(adapter_spec()) == (target, digest)
    record.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read"):
        variants.checked_adapter(adapter_spec())
    record.unlink()                                   # no registration.json (the tests' adapter): as before
    assert variants.build_variant(adapter_spec(), device="cpu", dtype="fp32",
                                  model_id=str(tiny_model_dir)).adapter_hash == digest


def test_named_templates_saved_with_the_adapter_pass(tmp_path):
    """Global ships named templates (default plus tool_use, as Cohere's chat models do); TRL saves the
    same tokenizer beside the adapter as chat_template.jinja plus additional_chat_templates/."""
    named = {"default": TINY_CHAT_TEMPLATE, "tool_use": "{{ bos_token }}tools{{ messages[0]['content'] }}"}
    global_tok = build_tiny_tokenizer(named)
    assert isinstance(global_tok.chat_template, dict)
    global_tok.save_pretrained(tmp_path)
    merge.check_adapter_template(tmp_path, global_tok)                   # the same tokenizer: no error
    merge.check_adapter_template(tmp_path, build_tiny_tokenizer())       # same default rendering: no error


def test_missing_adapter_and_unknown_method():
    with pytest.raises(FileNotFoundError, match="sync pull"):
        variants.build_variant(adapter_spec(), device="cpu", dtype="fp32", model_id="unused",
                               revision="a" * 40)
    # Every method has a builder since P05; a repair on a missing adapter fails like the adapter does.
    assert set(variants.BUILDERS) == {"stock", "adapter", "wiseft", "resta", "cast", "cast_always"}
    with pytest.raises(FileNotFoundError, match="sync pull"):
        variants.build_variant(VariantSpec(name="w", method="wiseft", adapter="x", params={"lambda": 0.5}),
                               device="cpu", model_id="unused", revision="a" * 40)


def test_adapter_trained_on_another_hub_model_is_refused(tiny_lora_adapter, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(tiny_lora_adapter, bad)
    cfg = json.loads((bad / "adapter_config.json").read_text(encoding="utf-8"))
    cfg["base_model_name_or_path"] = "CohereLabs/tiny-aya-base"
    (bad / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="trained on"):
        merge.check_adapter_base(bad, "CohereLabs/tiny-aya-global")
    merge.check_adapter_base(bad, "CohereLabs/tiny-aya-base")


def test_registry_loads_and_rejects_duplicates(monkeypatch):
    reg = variants.load_registry()
    assert reg["stock"].method == "stock"
    assert variants.get_spec("stock") == reg["stock"]
    with pytest.raises(KeyError, match="known"):
        variants.get_spec("nope")
    doc = {"variants": [{"name": "stock", "method": "stock"},
                        {"name": "adapter-aya-std-s1", "method": "adapter", "adapter": "adapter-aya-std-s1",
                         "seed": 1, "artifacts": {"adapter": "a" * 64}, "location": "outputs/adapters/x",
                         "note": "registered by P03"}]}
    monkeypatch.setattr(config, "load", lambda name="project": doc if name == "variants" else {})
    assert variants.load_registry()["adapter-aya-std-s1"].artifacts == {"adapter": "a" * 64}
    doc["variants"].append({"name": "stock", "method": "stock"})
    with pytest.raises(ValueError, match="twice"):
        variants.load_registry()
    doc["variants"] = [{"name": "x", "method": "stock", "path": "typo"}]
    with pytest.raises(ValueError):
        variants.load_registry()


def test_builder_registry_is_open_for_repairs():
    assert {"stock", "adapter"} <= set(variants.BUILDERS)
    with pytest.raises(ValueError, match="already"):
        variants.builder("stock")(lambda spec, ctx: None)
