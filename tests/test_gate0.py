"""Gate 0 end to end on CPU: the tiny cohere2 model as Global, a fake judge and a fake LID."""

import json

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fakes import make_prompt  # noqa: E402

from lift import gate0  # noqa: E402
from lift.prompts import SAFETY_SETS  # noqa: E402
from lift.schemas import LANGS  # noqa: E402

TEXTS = {"en": "How do I bake bread at home?", "hi": "मैं घर पर रोटी कैसे बनाऊं?", "zh": "我怎样在家烤面包？",
         "es": "¿Cómo hago pan en casa?", "ar": "كيف أخبز الخبز في المنزل؟", "th": "ฉันจะอบขนมปังที่บ้านได้อย่างไร"}


class FakeLID:
    LABELS = {"en": "eng_Latn", "hi": "hin_Deva", "zh": "cmn_Hani", "es": "spa_Latn", "ar": "arb_Arab",
              "th": "tha_Thai"}

    def top(self, text, k=3):
        return [("eng_Latn", 0.9)]


class FakeJudge:
    """Labels by output length parity; the label is irrelevant, the plumbing is what is tested."""

    judge_id = "fake/judge@test"
    short = "qwen3guard"

    def __init__(self, fail_every=0):
        self.calls = 0
        self.fail_every = fail_every

    def judge(self, items):
        from lift.judge.pipeline import JudgeOutcome

        out = []
        for item in items:
            self.calls += 1
            if self.fail_every and self.calls % self.fail_every == 0:
                out.append(JudgeOutcome("unclear", {"judge_failure": "unparseable"}, "x"))
            else:
                label = "refusal" if len(item.response) % 2 else "benign"
                out.append(JudgeOutcome(label, {}, "Safety: Safe"))
        return out


@pytest.fixture
def prompt_store(fake_prompt_files):
    for set_name in SAFETY_SETS:
        for lang in LANGS:
            fake_prompt_files[(set_name, lang)] = [
                make_prompt(f"{set_name[:3]}-{i:04d}", lang, set_name, "dev" if i < 4 else "final",
                            text=f"{TEXTS[lang]} #{i}") for i in range(6)]
    return fake_prompt_files


@pytest.fixture
def stock(tiny_model_dir):
    from lift.models.variants import build_variant
    from lift.schemas import VariantSpec

    return build_variant(VariantSpec(name="stock", method="stock"), device="cpu", dtype="fp32",
                         model_id=str(tiny_model_dir))


def test_preamble_check_passes_on_the_tiny_template(stock, prompt_store, tiny_model_dir):
    ref = gate0.reference_tokenizer(str(tiny_model_dir), "local")
    out = gate0.check_preamble(stock, gate0.all_prompts(), reference_tokenizer=ref)
    assert out["pass"], out
    assert out["reference_tokenizer_agreement"] == {lang: 1.0 for lang in [*LANGS, "probes"]}
    assert out["bos_count"] == 1
    assert out["n_prompts_checked"] == len(SAFETY_SETS) * len(LANGS) * 6 and out["n_too_long"] == 0


class _HindiBlindReference:
    """A reference tokenizer that disagrees on Devanagari only: English agreement alone must not pass."""

    def __init__(self, real):
        self.real = real

    def encode(self, text, add_special_tokens=False):
        enc = self.real.encode(text, add_special_tokens=add_special_tokens)
        if any("\u0900" <= ch <= "\u097f" for ch in text):
            from types import SimpleNamespace

            return SimpleNamespace(ids=list(enc.ids) + [0])
        return enc


def test_reference_tokenizer_is_checked_in_every_language(stock, prompt_store, tiny_model_dir):
    ref = _HindiBlindReference(gate0.reference_tokenizer(str(tiny_model_dir), "local"))
    out = gate0.check_preamble(stock, gate0.all_prompts(), reference_tokenizer=ref)
    assert not out["pass"]
    agreement = out["reference_tokenizer_agreement"]
    assert agreement["hi"] == 0.0 and agreement["en"] == 1.0


def test_preamble_check_fails_on_a_prompt_that_cannot_fit(stock, prompt_store):
    long = make_prompt("har-9999", "en", "harmful", "final", text="x" * 900)
    out = gate0.check_preamble(stock, gate0.all_prompts() + [long])
    assert not out["pass"] and out["n_too_long"] == 1 and out["too_long_examples"] == ["harmful/en/har-9999"]


def test_activation_scan_covers_every_layer(stock):
    out = gate0.activation_scan(stock.model, stock.tokenizer, [("hello", "world"), (TEXTS["hi"], "ठीक")],
                                device="cpu")
    assert out["all_finite"] and out["n_sequences"] == 2
    assert len(out["max_abs_per_layer"]) == stock.model.config.num_hidden_layers
    assert all(v > 0 for v in out["max_abs_per_layer"]) and out["fp16_overflow_layers"] == []


def test_activation_scan_catches_non_finite_values(stock):
    layer = stock.model.model.layers[1]
    handle = layer.register_forward_hook(lambda m, a, o: (o[0] * float("inf"),) + tuple(o[1:])
                                         if isinstance(o, tuple) else o * float("inf"))
    try:
        out = gate0.activation_scan(stock.model, stock.tokenizer, [("hello", "world")], device="cpu")
    finally:
        handle.remove()
    assert not out["all_finite"] and out["hidden_nonfinite"] > 0


def test_generate_and_judge_counts_one_record_each(stock, prompt_store, tmp_path):
    from lift.prompts import mixed_sample

    sample = mixed_sample(len(SAFETY_SETS) * len(LANGS), split="dev", seed=1)
    run = gate0.generate_and_judge(stock, FakeJudge(), sample, tmp_path / "dry", lid=FakeLID(),
                                   max_new_tokens=6)
    c = run["counts"]
    assert c["prompts"] == c["generations"] == c["judgements"] == len(sample) and not run["problems"]
    assert c["judge_failures"] == 0
    again = gate0.generate_and_judge(stock, FakeJudge(fail_every=2), sample, tmp_path / "dry2", lid=FakeLID(),
                                     max_new_tokens=6)
    assert again["counts"]["judge_failures"] > 0


def test_compare_runs_pairs_by_prompt(stock, prompt_store, tmp_path):
    from lift.prompts import mixed_sample

    sample = mixed_sample(len(SAFETY_SETS) * len(LANGS), split="dev", seed=1)
    gate0.generate_and_judge(stock, FakeJudge(), sample, tmp_path / "a", lid=FakeLID(), max_new_tokens=6)
    gate0.generate_and_judge(stock, FakeJudge(), sample, tmp_path / "b", lid=FakeLID(), max_new_tokens=6)
    same = gate0.compare_runs(tmp_path / "a", tmp_path / "b", stock.tokenizer)
    assert same["n"] == len(sample) and same["invalid"] == {}
    assert same["exact_output_agreement"] == 1.0 and same["label_agreement"] == 1.0
    empty = gate0.compare_runs(tmp_path / "missing", tmp_path / "b", stock.tokenizer)
    assert empty["n"] == 0 and not empty["pass"]


def test_compare_runs_refuses_pairs_with_other_inputs(stock, prompt_store, tmp_path):
    """A fixed translation (same pid, new text) or a harness change in the reference run makes the
    pairs invalid: they are counted, never scored as fp16-vs-bf16 agreement."""
    from lift.prompts import mixed_sample

    sample = mixed_sample(len(SAFETY_SETS) * len(LANGS), split="dev", seed=1)
    gate0.generate_and_judge(stock, FakeJudge(), sample, tmp_path / "a", lid=FakeLID(), max_new_tokens=6)
    first = sample[0]
    fixed = [p if p is not first else make_prompt(p.pid, p.lang, p.set, p.split, text=p.text + " (fixed)")
             for p in sample]
    prompt_store[(first.set, first.lang)] = [p if p.pid != first.pid else fixed[0]
                                             for p in prompt_store[(first.set, first.lang)]]
    gate0.generate_and_judge(stock, FakeJudge(), fixed, tmp_path / "b", lid=FakeLID(), max_new_tokens=6)
    cmp = gate0.compare_runs(tmp_path / "a", tmp_path / "b", stock.tokenizer)
    assert cmp["invalid"] == {"prompt_sha": 1}


def test_throughput_rows(stock, prompt_store):
    from lift.prompts import mixed_sample

    out = gate0.throughput(stock, mixed_sample(4, split="dev", seed=2), batch_sizes=(1, 2), max_new_tokens=6)
    assert [r["batch_size"] for r in out["rows"]] == [1, 2] and out["pass"]
    assert all(r["n"] == 4 and r["tokens_per_sec"] > 0 and r["peak_gb"] is None for r in out["rows"])
    assert out["batch_invariant_share"] == 1.0                 # fp32 on CPU: exactly batch-invariant


def test_decide_applies_the_readme_rule():
    ok = gate0.decide({"checks": {"preamble": {"pass": True}, "dry_run": {"pass": True}}})
    assert ok["pass"] and ok["decision"] == gate0.DECISION["pass"]
    partial = gate0.decide({"required": ["preamble", "dry_run", "parity"],
                            "checks": {"preamble": {"pass": True}, "dry_run": {"pass": True}}})
    assert not partial["pass"] and partial["incomplete"] == ["parity"] and "Incomplete" in partial["decision"]
    assert gate0.report_name({"dtype": "bf16", "reference_dtype": None, "incomplete": ["parity"]}) == \
        "gate0-partial"
    assert gate0.report_name({"dtype": "fp16", "reference_dtype": "bf16", "incomplete": []}) == "gate0-fp16"
    fp16 = gate0.decide({"checks": {"precision": {"pass": True}, "fp16": {"pass": False}}})
    assert not fp16["pass"] and fp16["decision"] == gate0.DECISION["fp16_fail"]
    both = gate0.decide({"checks": {"dry_run": {"pass": False}, "fp16": {"pass": False}}})
    assert gate0.DECISION["fail"] in both["decision"] and both["failed"] == ["dry_run", "fp16"]
    assert not gate0.decide({"checks": {}})["pass"]


def test_run_gate0_end_to_end_on_cpu(stock, prompt_store, tiny_model_dir, isolated_dirs):
    result = gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", judge=FakeJudge(),
                             lid=FakeLID(), n_precision=18, n_dry=1, batch_sizes=(2,), n_throughput=4,
                             max_new_tokens=6, resolve_pins=False)
    assert set(result["checks"]) == {"preamble", "precision", "throughput", "dry_run", "parity"}
    assert result["pass"], {k: v.get("pass") for k, v in result["checks"].items()}
    assert result["checks"]["parity"]["all_match"]
    json_path, md_path = gate0.write_report(result)
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["pass"] and doc["revisions"]["global"].startswith("local-")
    md = md_path.read_text(encoding="utf-8")
    assert "PASS" in md and TEXTS["en"] not in md and b"\r" not in md_path.read_bytes()
    assert result["dtype"] == "bf16"                  # the configured dtype, here on CPU
    assert (isolated_dirs / "outputs" / "gate0" / "bf16" / "precision" / "generations").is_dir()
    assert result["incomplete"] == [] and gate0.report_name(result) == "gate0"


def test_skipped_checks_make_the_gate_incomplete_and_keep_other_outputs(prompt_store, tiny_model_dir,
                                                                         isolated_dirs):
    kept = isolated_dirs / "outputs" / "gate0" / "bf16" / "precision" / "keep.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("x", encoding="utf-8")
    result = gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", judge=FakeJudge(),
                             lid=FakeLID(), skip=["precision", "throughput", "dry_run", "parity"],
                             max_new_tokens=6, resolve_pins=False)
    assert not result["pass"] and result["incomplete"] == ["precision", "throughput", "dry_run", "parity"]
    assert gate0.report_name(result) == "gate0-partial"
    assert kept.exists()                             # --skip precision never deletes the precision run


def test_a_check_that_raises_is_recorded_and_the_gate_goes_on(prompt_store, tiny_model_dir, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("tokenizer.json download failed")

    monkeypatch.setattr(gate0, "reference_tokenizer", broken)
    result = gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", judge=FakeJudge(),
                             lid=FakeLID(), skip=["precision", "dry_run", "parity"], batch_sizes=(2,),
                             n_throughput=4, max_new_tokens=6, resolve_pins=False)
    assert result["checks"]["preamble"]["error"].startswith("RuntimeError: tokenizer.json download failed")
    assert result["checks"]["throughput"]["pass"] and not result["pass"]


def test_fp16_compare_reads_the_reference_run(stock, prompt_store, tiny_model_dir, monkeypatch):
    """The comparison run looks for the reference dtype's precision run and pairs the same prompts."""
    from lift import config

    real_get = config.get
    monkeypatch.setattr(config, "get", lambda key, *a, **k: "fp32" if key == "generation.dtype"
                        else real_get(key, *a, **k))
    kw = dict(device="cpu", model_id=str(tiny_model_dir), revision="local", lid=FakeLID(), n_precision=18,
              skip=["preamble", "throughput", "dry_run", "parity"], resolve_pins=False, max_new_tokens=6)
    reference = gate0.run_gate0(judge=FakeJudge(), **kw)
    assert reference["checks"]["precision"]["pass"]
    # The candidate run: same prompts in another dtype (bf16 on CPU here), compared against fp32.
    cand = gate0.run_gate0(judge=FakeJudge(), compare_dtype="bf16", **kw)
    fp = cand["checks"]["fp16"]
    assert fp["n"] == 18 and fp["n_reference"] == 18 and 0.0 <= fp["label_agreement"] <= 1.0
    assert fp["invalid"] == {}                        # same prompts, model, template, harness and judge
    assert cand["reference_dtype"] == "fp32" and cand["dtype"] == "bf16"
    assert cand["required"] == ["precision", "fp16"] and gate0.report_name(cand) == "gate0-bf16"
