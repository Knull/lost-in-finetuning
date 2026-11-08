"""Gate 0 end to end on CPU: the tiny cohere2 model as Global, a fake judge and a fake LID."""

import json

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fakes import TINY_CHAT_TEMPLATE, TINY_PREAMBLE, build_tiny_tokenizer, make_prompt  # noqa: E402

from lift import gate0  # noqa: E402
from lift.judge.labels import rules_tag  # noqa: E402
from lift.prompts import SAFETY_SETS  # noqa: E402
from lift.schemas import LANGS  # noqa: E402

TEXTS = {"en": "How do I bake bread at home?", "hi": "मैं घर पर रोटी कैसे बनाऊं?", "zh": "我怎样在家烤面包？",
         "es": "¿Cómo hago pan en casa?", "ar": "كيف أخبز الخبز في المنزل؟", "th": "ฉันจะอบขนมปังที่บ้านได้อย่างไร"}


class FakeLID:
    LABELS = {"en": "eng_Latn", "hi": "hin_Deva", "zh": "cmn_Hani", "es": "spa_Latn", "ar": "arb_Arab",
              "th": "tha_Thai"}

    def top(self, text, k=3):
        return [("eng_Latn", 0.9)]


class GarbledLID:
    """Every output reads as garbled: the top label's language holds under unclear.min_lid_prob.

    After P01's item 43 the mass is summed over the accept_labels list holding the top label, and
    anp_Deva is in hi's list, so the old [anp_Deva 0.4, hin_Deva 0.3] summed to 0.7 and was not
    garbled. hin_Deva 0.3 with English second keeps the Hindi mass at 0.3."""

    def top(self, text, k=3):
        return [("hin_Deva", 0.3), ("eng_Latn", 0.2)][:k]


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


class HarmfulJudge(FakeJudge):
    """The same judge id, but every output is harmful: no label agrees with FakeJudge's. ``fail_every``
    makes every n-th item a judge failure instead, as in FakeJudge."""

    def judge(self, items):
        from lift.judge.pipeline import JudgeOutcome

        out = []
        for _ in items:
            self.calls += 1
            failed = self.fail_every and self.calls % self.fail_every == 0
            out.append(JudgeOutcome("unclear", {"judge_failure": "unparseable"}, "x") if failed
                       else JudgeOutcome("harmful", {}, "Safety: Unsafe"))
        return out


@pytest.fixture
def fp32_configured(monkeypatch):
    """generation.dtype fp32, so a CPU run can be the reference and bf16 the compared dtype (the
    analogue of bf16 on Colab and fp16 on a T4)."""
    from lift import config

    real_get = config.get
    monkeypatch.setattr(config, "get", lambda key, *a, **k: "fp32" if key == "generation.dtype"
                        else real_get(key, *a, **k))


@pytest.fixture
def reference_passed(tiny_model_dir):
    """vouch(): the reference Gate 0's own PASS (outputs/metrics/gate0.json, as `lift sync pull` brings it
    from the A100) for the tiny model's revision and the fp32 precision outputs now on disk (their digest).
    A comparison counts only against it (contracts E and J), so call it once the reference run is there."""
    from lift.models.load import resolved_revision
    from lift.paths import ensure, metrics_dir

    def vouch():
        path = ensure(metrics_dir()) / "gate0.json"
        rev = resolved_revision(str(tiny_model_dir), "local")
        digest = gate0.precision_digest(gate0.gate0_dir("fp32") / "precision")
        doc = {"pass": True, "revisions": {"global": rev}, "checks": {"precision": {"digest": digest}}}
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    return vouch


@pytest.fixture
def cpu_sized(monkeypatch):
    """decisions.md 27's samples (precision and the licence on 200 prompts, the dry run on 20 per set and
    language) lowered to the CPU tests' 18 prompts and 1 per set and language."""
    monkeypatch.setattr(gate0, "N_PRECISION", 18)
    monkeypatch.setattr(gate0, "N_DRY", 1)


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


# R1-23: the tiny template, and broken copies a changed Tiny Aya revision could ship. Each broken one keeps
# everything Gate 0 checked before (one BOS, a non-empty first system block, the same prefix, every prompt
# fitting, the reference tokenizer agreeing), so only the new rules can catch it.
PREAMBLE_SOURCE = TINY_PREAMBLE.replace("\n", "{{ '\\n' }}")          # how TINY_CHAT_TEMPLATE writes it
GENERATION_PROMPT = "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
TEMPLATES = {
    "unchanged": TINY_CHAT_TEMPLATE,
    "preamble-twice": TINY_CHAT_TEMPLATE.replace(          # a second system turn before the user's
        "{% for message in loop_messages %}",
        f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{PREAMBLE_SOURCE}<|END_OF_TURN_TOKEN|>"
        "{% for message in loop_messages %}", 1),
    "no-preamble": TINY_CHAT_TEMPLATE.replace(PREAMBLE_SOURCE, "Hi.", 1),       # the system turn kept
    "no-safety-mode": TINY_CHAT_TEMPLATE.replace("contextual safety mode", "unrestricted mode", 1),
    "no-generation-prompt": TINY_CHAT_TEMPLATE.replace(
        "{% if add_generation_prompt %}" + GENERATION_PROMPT + "{% endif %}", "", 1),
}


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_the_preamble_check_needs_one_default_preamble_and_the_generation_prompt(prompt_store, tmp_path,
                                                                                  name):
    """R1-23: decisions.md 27 and this module's docstring: the template renders the default preamble block
    once (the contextual-safety-mode one, CLAUDE.md) and ends in the generation prompt. The check only asked
    for a non-empty first system block, so a doubled preamble, a gutted or replaced one, or a template that
    ignores add_generation_prompt passed. Run as run_gate0 runs it: the tokenizer saved and reloaded, its
    reference read from the same tokenizer.json."""
    from types import SimpleNamespace

    from lift.models import load

    template = TEMPLATES[name]
    assert name == "unchanged" or template != TINY_CHAT_TEMPLATE, "the mutation applied"
    build_tiny_tokenizer(template).save_pretrained(tmp_path)
    tok = load.load_tokenizer(str(tmp_path), "local")
    variant = SimpleNamespace(tokenizer=tok,
                              hf_model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)))
    out = gate0.check_preamble(variant, gate0.all_prompts(),
                               reference_tokenizer=gate0.reference_tokenizer(str(tmp_path), "local"))
    text = load.chat_text(tok, "probe")
    shape = {"unchanged": (1, True, True), "preamble-twice": (2, True, True), "no-preamble": (1, False, True),
             "no-safety-mode": (1, False, True), "no-generation-prompt": (1, True, False)}[name]
    first_block = text.split("<|SYSTEM_TOKEN|>", 1)[1].split("<|END_OF_TURN_TOKEN|>", 1)[0]
    assert (text.count("<|SYSTEM_TOKEN|>"), "contextual safety mode" in first_block,
            text.endswith(GENERATION_PROMPT)) == shape                  # broken the way its name says
    assert out["bos_count"] == 1 and out["starts_with_bos"] and out["system_block_tokens"] > 0
    assert out["preamble_identical_across_prompts"] and out["n_too_long"] == 0
    assert set(out["reference_tokenizer_agreement"].values()) == {1.0}  # and in no other way

    numbers = {k: v for k, v in out.items() if k != "too_long_examples"}
    assert out["pass"] is (name == "unchanged"), (name, numbers)
    new = (out.get("system_blocks"), out.get("safety_preamble"), out.get("ends_with_generation_prompt"))
    assert new == shape, numbers
    summary = gate0._summary("preamble", out)
    assert f"system blocks {shape[0]}" in summary and f"safety preamble {shape[1]}" in summary
    assert f"generation prompt at the end {shape[2]}" in summary


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
    low = {"pass": False, "failure": "label agreement 0.900 is under 0.95"}
    fp16 = gate0.decide({"checks": {"precision": {"pass": True}, "fp16": low}})
    assert not fp16["pass"] and fp16["decision"] == gate0.DECISION["fp16_fail"]
    both = gate0.decide({"checks": {"dry_run": {"pass": False}, "fp16": low}})
    assert both["decision"] == gate0.DECISION["fp16_fail"] + " " + gate0.DECISION["fail"]
    assert both["failed"] == ["dry_run", "fp16"]
    # an fp16 check that raised is a bug to fix, not the verdict that rules T4 runs out
    raised = gate0.decide({"checks": {"precision": {"pass": True},
                                      "fp16": {"pass": False, "error": "KeyError: x"}}})
    assert raised["decision"] == gate0.DECISION["fail"]
    assert not gate0.decide({"checks": {}})["pass"]
    # a failed required check is a full verdict, even when another required check did not run
    failed = gate0.decide({"dtype": "bf16", "reference_dtype": None, "required": ["preamble", "parity"],
                           "checks": {"preamble": {"pass": False}}})
    assert failed["incomplete"] == ["parity"] and gate0.report_name(failed) == "gate0"
    assert gate0.verdict(failed) == "FAIL" and gate0.verdict(partial) == "INCOMPLETE"


def test_fp16_fails_only_on_a_real_result():
    valid = {"n": 200, "n_reference": 200, "n_candidate": 200, "missing_labels": 0, "invalid": {},
             "label_agreement": 0.96}
    finite = {"pass": True, "all_finite": True, "problems": []}
    kw = dict(reference="bf16", reference_problem=None)
    ok = gate0.fp16_verdict(valid, finite, 200, **kw)
    assert ok["pass"] and ok["comparable"] and "failure" not in ok
    low = gate0.fp16_verdict({**valid, "label_agreement": 0.9}, finite, 200, **kw)
    assert not low["pass"] and low["comparable"] and low["failure"] == "label agreement 0.900 is under 0.95"
    # a non-finite fp16 run fails whatever the comparison, and is one problem, not two
    overflow = {"pass": False, "all_finite": False, "problems": []}
    nothing = {"n": 0, "pass": False, "error": "no shared prompts"}
    inf = gate0.fp16_verdict(nothing, overflow, 200, reference="bf16", reference_problem="no reference")
    assert not inf["pass"] and not inf.get("not_run") and "non-finite" in inf["failure"]
    run = gate0.decide({"required": ["precision", "fp16"], "checks": {"precision": overflow, "fp16": inf}})
    assert run["decision"] == gate0.DECISION["fp16_fail"] and run["failed"] == ["fp16", "precision"]
    for cmp, why in [({**valid, "n": 199, "n_reference": 199}, "199 of 200"),
                     ({**valid, "missing_labels": 2}, "2 labels missing"),
                     ({**valid, "invalid": {"harness": 200}}, "{'harness': 200}")]:
        out = gate0.fp16_verdict(cmp, finite, 200, **kw)
        assert out["not_run"] and not out["pass"] and out["comparable"] is False and why in out["reason"]


def test_a_licence_needs_the_full_sample_and_a_passing_reference():
    """decisions.md 27: labels agree on the same 200 prompts, against a bf16 Gate 0 that passed. A
    quick --n-precision 50 whose 50 prompts all pair, or a reference whose own Gate 0 failed, is no
    valid comparison: NOT RUN, whatever the agreement (V15, V6)."""
    finite = {"pass": True, "all_finite": True, "problems": []}
    fifty = {"n": 50, "n_reference": 200, "n_candidate": 50, "missing_labels": 0, "invalid": {},
             "label_agreement": 1.0}
    small = gate0.fp16_verdict(fifty, finite, 50, reference="bf16", reference_problem=None)
    assert small["not_run"] and small["comparable"] is False and not small["pass"]
    assert small["reason"] == "50 prompts, under the 200 decisions.md 27 requires"
    valid = {**fifty, "n": 200, "n_candidate": 200}
    failed = "the bf16 Gate 0 (gate0.json) did not pass"
    out = gate0.fp16_verdict(valid, finite, 200, reference="bf16", reference_problem=failed)
    assert out["not_run"] and out["comparable"] is False and out["reason"] == failed


def test_reference_gate_problem_names_what_is_wrong():
    """The bf16 Gate 0's own verdict, as check_dtype_licence reads it: present, readable, "pass" true,
    for this revision of Global, and about the precision outputs on disk (its digest, contract J)."""
    from lift.paths import ensure, metrics_dir

    rev, digest = "a" * 40, "f" * 64
    path = ensure(metrics_dir()) / "gate0.json"
    missing = gate0.reference_gate_problem(rev, "bf16", digest=digest)
    assert missing == "the bf16 Gate 0 (gate0.json) is missing"
    ok = {"pass": True, "revisions": {"global": rev}, "checks": {"precision": {"digest": digest}}}
    for doc, why in [(ok, None),
                     ({**ok, "pass": False}, "did not pass"),
                     ({**ok, "pass": "true"}, "did not pass"),
                     ([True], "did not pass"),
                     ({**ok, "revisions": {"global": "e" * 40}}, f"revision {'e' * 40}, not {rev}"),
                     ({**ok, "revisions": "x"}, f"revision None, not {rev}"),
                     ({**ok, "checks": {"precision": {"pass": True, "digest": "0" * 64}}},
                      "the bf16 precision outputs here are not the ones gate0.json passed"),
                     ({**ok, "checks": {"precision": {"pass": True}}}, "records no checks.precision.digest"),
                     ({**ok, "checks": ["precision"]}, "records no checks.precision.digest")]:
        path.write_text(json.dumps(doc), encoding="utf-8")
        got = gate0.reference_gate_problem(rev, "bf16", digest=digest)
        assert (got is None) if why is None else (why in got), (doc, got)
    path.write_text('{"pass": tr', encoding="utf-8")                  # a truncated download
    assert gate0.reference_gate_problem(rev, "bf16", digest=digest) == \
        "the bf16 Gate 0 (gate0.json) cannot be read (JSONDecodeError)"


def test_run_gate0_end_to_end_on_cpu(stock, prompt_store, tiny_model_dir, isolated_dirs, cpu_sized,
                                     monkeypatch):
    # R2-24: every sample a check draws is dev (CLAUDE.md rule 1; prompt_store holds final prompts too): the
    # splits of what each mixed_sample call returned, of what the parity check decoded (it keeps no records)
    # and of every generation the precision and dry-run checks wrote.
    from lift import prompts
    from lift.io import read_jsonl

    drawn, decoded = [], []
    real_sample, real_parity = prompts.mixed_sample, gate0.check_parity

    def mixed_sample(*args, **kwargs):
        out = real_sample(*args, **kwargs)
        drawn.append({p.split for p in out})
        return out

    def check_parity(model_id, revision, sample, **kwargs):
        decoded.extend(p.split for p in sample)
        return real_parity(model_id, revision, sample, **kwargs)

    monkeypatch.setattr(prompts, "mixed_sample", mixed_sample)
    monkeypatch.setattr(gate0, "check_parity", check_parity)
    result = gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", judge=FakeJudge(),
                             lid=FakeLID(), n_precision=18, n_dry=1, batch_sizes=(2,), n_throughput=4,
                             max_new_tokens=6, resolve_pins=False)
    assert drawn == [{"dev"}] * 4                     # throughput, precision, dry run, parity
    assert decoded == ["dev"] * gate0.N_PARITY
    for check in ("precision", "dry_run"):
        files = sorted((gate0.gate0_dir("bf16") / check / "generations").glob("*.jsonl"))
        assert [r["split"] for path in files for r in read_jsonl(path)] == ["dev"] * 18, check
    assert set(result["checks"]) == {"preamble", "precision", "throughput", "dry_run", "parity"}
    assert result["pass"], {k: v.get("pass") for k, v in result["checks"].items()}
    assert result["checks"]["parity"]["all_match"]
    digest = gate0.precision_digest(gate0.gate0_dir("bf16") / "precision")
    assert result["checks"]["precision"]["digest"] == digest
    json_path, md_path = gate0.write_report(result)
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["pass"] and doc["revisions"]["global"].startswith("local-")
    md = md_path.read_text(encoding="utf-8")
    assert "PASS" in md and TEXTS["en"] not in md and b"\r" not in md_path.read_bytes()
    assert result["dtype"] == "bf16"                  # the configured dtype, here on CPU
    assert (isolated_dirs / "outputs" / "gate0" / "bf16" / "precision" / "generations").is_dir()
    assert not (isolated_dirs / "outputs" / "gate0" / "bf16" / "precision.new").exists()    # moved into place
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


def test_fp16_compare_reads_the_reference_run(stock, prompt_store, tiny_model_dir, fp32_configured,
                                              cpu_sized):
    """The comparison run looks for the reference dtype's precision run and pairs the same prompts."""
    kw = dict(device="cpu", model_id=str(tiny_model_dir), revision="local", lid=FakeLID(), n_precision=18,
              skip=["preamble", "throughput", "dry_run", "parity"], resolve_pins=False, max_new_tokens=6)
    reference, name, _ = _gate(tiny_model_dir, judge=FakeJudge())   # a full report: its outputs stay
    assert name == "gate0" and reference["checks"]["precision"]["pass"]
    # The candidate run: same prompts in another dtype (bf16 on CPU here), compared against fp32.
    cand = gate0.run_gate0(judge=FakeJudge(), compare_dtype="bf16", **kw)
    fp = cand["checks"]["fp16"]
    assert fp["n"] == 18 and fp["n_reference"] == 18 and 0.0 <= fp["label_agreement"] <= 1.0
    assert fp["invalid"] == {} and fp["comparable"]   # same prompts, model, template, harness and judge
    assert fp["reference_digest"] == reference["checks"]["precision"]["digest"]
    assert cand["reference_dtype"] == "fp32" and cand["dtype"] == "bf16"
    assert cand["required"] == ["precision", "fp16"] and gate0.report_name(cand) == "gate0-bf16"


# --- every Gate 0 run writes a verdict (R8, R1) ---------------------------------------------------------

GATE = dict(device="cpu", revision="local", lid=FakeLID(), n_precision=18, n_dry=1, batch_sizes=(2,),
            n_throughput=4, max_new_tokens=6, resolve_pins=False)


@pytest.mark.parametrize("skip", [[], ["throughput"]], ids=["all", "skip-throughput"])
@pytest.mark.parametrize("error, message", [(FileNotFoundError, "no prompts for harmful_hi"),
                                            (NotImplementedError, "P01")], ids=["missing-file", "p01-stub"])
def test_prompts_that_cannot_load_still_give_a_written_verdict(tiny_model_dir, monkeypatch, error, message,
                                                               skip):
    """A prompt file missing on the VM, or a ref whose P01 loader is still a stub: every check that
    needs prompts (the throughput and parity samples included) fails with the error, and the
    verdict is a full report, so the operator sees what broke."""

    def load_prompts(*a, **k):
        raise error(message)

    monkeypatch.setattr("lift.data.build.load_prompts", load_prompts)
    result = gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), skip=skip, **GATE)
    ran = [c for c in gate0.CHECKS if c != "fp16" and c not in skip]
    assert result["failed"] == sorted(ran) and not result["pass"]
    assert all(result["checks"][c]["error"] == f"{error.__name__}: {message}" for c in ran)
    if error is NotImplementedError:
        assert "P01 and P02" in result["decision"]           # the ref lacks P01: say which ref to run
    name = gate0.report_name(result)
    assert name == "gate0"                                    # a failed required check is a full verdict
    json_path, md_path = gate0.write_report(result, name=name)
    assert json.loads(json_path.read_text(encoding="utf-8"))["pass"] is False
    assert "| throughput |" in md_path.read_text(encoding="utf-8") or "throughput" in skip


def test_an_unresolvable_revision_is_recorded_as_the_load_check(tiny_model_dir, monkeypatch):
    def offline(model_id, revision):
        raise OSError("the Hub cannot be reached")

    monkeypatch.setattr("lift.models.load.resolved_revision", offline)
    result = gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), **GATE)
    assert set(result["checks"]) == {"load"} and result["failed"] == ["load"]
    assert result["checks"]["load"]["error"] == "OSError: the Hub cannot be reached"
    assert result["revisions"] == {"global": "unresolved: OSError"}
    assert gate0.report_name(result) == "gate0" and not result["pass"]
    hint = gate0.pin_hint(result)
    assert "models.global.revision" not in hint and "models.global" in hint   # named, never a value to pin


def test_a_rerun_that_cannot_load_replaces_the_previous_verdict(prompt_store, tiny_model_dir, monkeypatch):
    """Gate 0 passed, a harness fix follows, and the re-run cannot load the model: the FAIL must
    replace gate0.json, the record every message points at, not land beside it."""
    from lift.paths import metrics_dir

    previous = metrics_dir() / "gate0.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({"pass": True}), encoding="utf-8")
    kept = gate0.gate0_dir("bf16") / "precision" / "keep.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("x", encoding="utf-8")

    def out_of_memory(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("lift.models.variants.build_variant", out_of_memory)
    result = gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), **GATE)
    assert set(result["checks"]) == {"load"} and result["failed"] == ["load"]
    name = gate0.report_name(result)
    assert name == "gate0"
    json_path, md_path = gate0.write_report(result, name=name)
    assert json_path == previous and json.loads(previous.read_text(encoding="utf-8"))["pass"] is False
    assert "| load | FAIL | RuntimeError" in md_path.read_text(encoding="utf-8")
    assert kept.exists()                              # nothing ran, so no earlier output is discarded


# --- the fp16 verdict: only a real comparison decides (R10, R13) --------------------------------------

CANDIDATE = dict(device="cpu", revision="local", lid=FakeLID(), n_precision=18, max_new_tokens=6,
                 skip=["preamble", "throughput", "dry_run", "parity"], resolve_pins=False)


def _reference_without_one_cell():
    from lift.paths import output_dir

    files = sorted((output_dir() / "gate0" / "fp32" / "precision" / "generations").glob("*.jsonl"))
    files[0].unlink()


def _reference_from_another_harness():
    from lift.io import read_jsonl, write_jsonl
    from lift.paths import output_dir

    for path in (output_dir() / "gate0" / "fp32" / "precision" / "generations").glob("*.jsonl"):
        rows = list(read_jsonl(path))
        for row in rows:
            row["meta"]["harness"] = "an older harness"
        write_jsonl(path, rows)


@pytest.mark.parametrize("damage, why", [(None, "no shared prompts"),
                                         (_reference_without_one_cell, "17 of 18"),
                                         (_reference_from_another_harness, "harness")],
                         ids=["no-reference", "fewer-prompts", "stale-reference"])
def test_fp16_without_a_valid_comparison_decides_nothing(prompt_store, tiny_model_dir, fp32_configured,
                                                         reference_passed, cpu_sized, damage, why):
    """No reference run (its push failed), one with other prompts, or one made by an older harness:
    nothing was compared, so the T4 question stays open. The run is incomplete, written beside the
    full record (-partial), and the decision says to pull or re-run the reference Gate 0."""
    from lift.paths import metrics_dir

    if damage is not None:
        reference, name, _ = _gate(tiny_model_dir, judge=FakeJudge())
        assert name == "gate0" and reference["checks"]["precision"]["pass"]
        damage()
    reference_passed()                                # gate0.json vouches for what is there, damage and all
    licence = metrics_dir() / "gate0-bf16.json"
    licence.parent.mkdir(parents=True, exist_ok=True)
    licence.write_text(json.dumps({"pass": True, "checks": {"fp16": {"pass": True}}}), encoding="utf-8")
    cand = gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), compare_dtype="bf16", **CANDIDATE)
    fp = cand["checks"]["fp16"]
    assert cand["checks"]["precision"]["pass"]
    assert fp["not_run"] and fp["comparable"] is False and not fp["pass"] and why in fp["reason"]
    assert cand["failed"] == [] and cand["incomplete"] == ["fp16"] and not cand["pass"]
    assert gate0.DECISION["fp16_fail"] not in cand["decision"]
    assert "lift sync pull" in cand["decision"] and "re-run the fp32 Gate 0" in cand["decision"]
    name = gate0.report_name(cand)
    assert name == "gate0-bf16-partial"
    gate0.write_report(cand, name=name)
    assert json.loads(licence.read_text(encoding="utf-8"))["pass"] is True     # the licence record stands


def test_a_valid_comparison_under_95_percent_is_the_fp16_verdict(prompt_store, tiny_model_dir,
                                                                 fp32_configured, cpu_sized):
    reference, name, _ = _gate(tiny_model_dir, judge=FakeJudge())
    assert name == "gate0" and reference["checks"]["precision"]["pass"]
    cand = gate0.run_gate0(model_id=str(tiny_model_dir), judge=HarmfulJudge(), compare_dtype="bf16",
                           **CANDIDATE)
    fp = cand["checks"]["fp16"]
    assert fp["comparable"] and fp["n"] == 18 and fp["label_agreement"] == 0.0 and not fp.get("not_run")
    assert cand["failed"] == ["fp16"] and cand["incomplete"] == []
    assert cand["decision"] == gate0.DECISION["fp16_fail"]
    assert gate0.report_name(cand) == "gate0-bf16"                   # a real verdict replaces the record


@pytest.mark.parametrize("compare_dtype", [None, "bf16"], ids=["reference-run", "comparison-run"])
def test_one_gpu_without_room_for_the_judge_decides_nothing(prompt_store, tiny_model_dir, fp32_configured,
                                                            monkeypatch, compare_dtype):
    """One 16 GB T4: the judge cannot load beside the generator, so precision, fp16 and the dry run
    never run. That is a resource refusal, not a verdict: never 'fp16 failed', never the full record."""
    from lift.paths import metrics_dir

    problem = ("only 8.5 GB free on cuda:0 after loading the generator; the judge needs about 13 GB beside "
               "it. Run Gate 0 on two GPUs (Kaggle T4 x2, --judge-device cuda:1) or on a larger GPU")
    monkeypatch.setattr(gate0, "_judge_memory_problem", lambda device, judge_device: problem)
    full = metrics_dir() / ("gate0-bf16.json" if compare_dtype else "gate0.json")
    full.parent.mkdir(parents=True)
    full.write_text(json.dumps({"pass": True}), encoding="utf-8")
    kept = gate0.gate0_dir(compare_dtype or "fp32") / "precision" / "keep.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("x", encoding="utf-8")
    result = gate0.run_gate0(device="cpu", judge_device="cpu", compare_dtype=compare_dtype,
                             model_id=str(tiny_model_dir), revision="local", lid=FakeLID(), n_precision=18,
                             n_dry=1, skip=["preamble", "throughput", "parity"], max_new_tokens=6,
                             resolve_pins=False)
    refused = ["precision", "fp16", "dry_run"] if compare_dtype else ["precision", "dry_run"]
    assert all(result["checks"][c]["not_run"] and result["checks"][c]["reason"] == problem for c in refused)
    assert result["failed"] == [] and set(refused) <= set(result["incomplete"]) and not result["pass"]
    assert gate0.DECISION["fp16_fail"] not in result["decision"] and problem in result["decision"]
    name = gate0.report_name(result)
    assert name == ("gate0-bf16-partial" if compare_dtype else "gate0-partial")
    gate0.write_report(result, name=name)
    assert json.loads(full.read_text(encoding="utf-8"))["pass"] is True
    assert kept.exists()                              # a check that did not run discards nothing


class OneGPU:
    """torch.cuda's calls on one card (an L4 by default), with memory answered as the driver and PyTorch's
    caching allocator answer it: mem_get_info counts the blocks the allocator keeps for freed tensors as
    used, until empty_cache hands them back. Each decoded batch leaves CACHED_GB there: an ESTIMATE of what
    throughput's batch-32 run leaves with Tiny Aya (about 120 MB per sequence), not a measurement."""

    GB = 1e9
    USED_GB = 6.7 + 0.5           # Global's bf16 weights and the CUDA context
    CACHED_GB = 3.84

    def __init__(self, total_gb: float = 23.65):         # an L4: torch reports 22.03 GiB
        self.total, self.cached, self.peak_cached = total_gb * self.GB, 0.0, 0.0

    def decode(self) -> None:
        self.cached = max(self.cached, self.CACHED_GB * self.GB)
        self.peak_cached = max(self.peak_cached, self.cached)

    def free(self) -> int:
        return int(self.total - self.USED_GB * self.GB - self.cached)

    def install(self, monkeypatch, name: str = "NVIDIA L4", capability: tuple = (8, 9)) -> "OneGPU":
        calls = {"is_available": lambda: True, "device_count": lambda: 1,
                 "get_device_capability": lambda device=None: capability,
                 "get_device_name": lambda device=None: name, "synchronize": lambda device=None: None,
                 "reset_peak_memory_stats": lambda device=None: None,
                 "max_memory_allocated": lambda device=None: int(self.USED_GB * self.GB + self.peak_cached),
                 "mem_get_info": lambda device=None: (self.free(), int(self.total)),
                 "empty_cache": lambda: setattr(self, "cached", 0.0)}
        for attr, value in calls.items():
            monkeypatch.setattr(torch.cuda, attr, value)
        return self


def test_one_gpu_measures_the_judge_room_with_the_throughput_cache_released(
        prompt_store, stock, tiny_model_dir, fp32_configured, cpu_sized, monkeypatch):
    """R1-47: `lift remote --gpu L4 --job "lift sync pull; lift smoke"`: one GPU, so the judge loads on
    cuda:0 beside the generator, where it fits (16.45 GB free, 13 GB needed). The throughput check runs
    first, and the blocks its batches leave in PyTorch's cache read as used to mem_get_info (12.6 GB), so
    precision and dry_run were NOT RUN and the reference Gate 0 on an L4 INCOMPLETE. The judge's own
    allocations would reuse those blocks, so the cache is released before the judge's room is measured.
    Global is the tiny model: lift sees it on cuda:0 while it computes on the CPU."""
    import dataclasses

    from lift.models import generate

    card = OneGPU().install(monkeypatch)
    assert gate0._judge_memory_problem("cuda:0", "cuda:0") is None           # nothing cached: the judge fits
    real_batch, real_scan = generate.greedy_batch, gate0.activation_scan

    def greedy_batch(model, rows, **kw):
        out = real_batch(model, rows, **{**kw, "device": "cpu"})
        card.decode()
        return out

    monkeypatch.setattr("lift.models.variants.build_variant",
                        lambda spec, *, device, dtype, **kw: dataclasses.replace(stock, device=device))
    monkeypatch.setattr(generate, "greedy_batch", greedy_batch)
    monkeypatch.setattr(gate0, "activation_scan",
                        lambda model, tok, pairs, *, device: real_scan(model, tok, pairs, device="cpu"))
    monkeypatch.setattr("lift.judge.pipeline.make_judge", lambda role, **kw: FakeJudge())
    result = gate0.run_gate0(model_id=str(tiny_model_dir), judge_device="cuda:0", skip=["preamble", "parity"],
                             **{**GATE, "device": "cuda:0"})
    checks = result["checks"]
    assert result["gpu"] == "NVIDIA L4" and checks["throughput"]["pass"], checks
    assert card.peak_cached == OneGPU.CACHED_GB * OneGPU.GB             # its batches left their blocks cached
    for name in ("precision", "dry_run"):
        c = checks[name]
        assert not c.get("not_run"), f"{name} is NOT RUN on one L4 the judge fits on: {c.get('reason')}"
        assert c["pass"], (name, c.get("error"), c.get("problems"))
    assert result["judge"]["id"] == FakeJudge.judge_id and result["failed"] == []
    OneGPU(16.0).install(monkeypatch)                                    # a 16 GB card is still refused
    assert gate0._judge_memory_problem("cuda:0", "cuda:0").startswith("only 8.8 GB free on cuda:0")


# --- the reference Gate 0 never runs on a T4 (V1, V8, V9; contract F) -------------------------------------


@pytest.fixture
def t4(monkeypatch):
    """torch.cuda reporting one Tesla T4 (compute capability 7.5: no native bf16), on a CPU build."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Tesla T4")


def test_a_plain_smoke_on_a_t4_is_refused_and_writes_nothing(prompt_store, tiny_model_dir, t4, monkeypatch):
    """`lift sync pull; lift smoke` on a T4 without --compare-dtype fp16. The bf16 reference Gate 0
    cannot run there, and its failed load would be a full verdict replacing the A100's gate0.json PASS
    (and with it the fp16 licence) in the runs repo. Like --compare-dtype fp16 on an A100, it is a
    refused configuration: ValueError before any check, exit 2, no report."""
    from lift.cli import main
    from lift.paths import ensure, metrics_dir, output_dir

    rev = "a" * 40
    for name, doc in {"gate0.json": {"pass": True, "revisions": {"global": rev}},
                      "gate0-fp16.json": {"pass": True, "checks": {"fp16": {"pass": True, "n": 200}},
                                          "revisions": {"global": rev}}}.items():
        (ensure(metrics_dir()) / name).write_text(json.dumps(doc), encoding="utf-8")
    pulled = {p.name: p.read_bytes() for p in metrics_dir().iterdir()}
    with pytest.raises(ValueError, match=r"^Tesla T4 has no native bf16 .*lift smoke --compare-dtype fp16"):
        gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), **{**GATE, "device": "cuda:0"})
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")
    assert main(["smoke"]) == 2                               # the job line: refused, like a usage error
    assert {p.name: p.read_bytes() for p in metrics_dir().iterdir()} == pulled
    assert not (output_dir() / "gate0").exists()


# --- a size under 1 is a refused configuration (review 5: X17) -------------------------------------------


@pytest.fixture
def a100(monkeypatch):
    """torch.cuda reporting one A100 (compute capability 8.0: native bf16), on a CPU build."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA A100-SXM4-40GB")


def _passing_record():
    from lift.paths import ensure, metrics_dir

    record = ensure(metrics_dir()) / "gate0.json"
    record.write_text(json.dumps({"pass": True, "revisions": {"global": "a" * 40}}), encoding="utf-8")
    return record, record.read_bytes()


@pytest.mark.parametrize("size", [{"n_dry": 0}, {"n_precision": 0}, {"batch_sizes": ()},
                                  {"batch_sizes": (8, 0)}],
                         ids=["n-dry-0", "n-precision-0", "no-batch-sizes", "batch-size-0"])
def test_a_size_under_one_is_refused_and_writes_nothing(prompt_store, tiny_model_dir, size):
    """X17: `lift smoke --n-dry 0` (meant as 'no dry run'; --skip dry_run is the way) or --n-precision 0 made
    its check raise: a FAIL, so a full verdict that replaced the passing gate0.json, and with it the T4
    licence. Like an unknown --skip, a size under 1 is a refused configuration: ValueError before any check,
    nothing written."""
    from lift.paths import output_dir

    record, pulled = _passing_record()
    with pytest.raises(ValueError, match=r"must be at least 1 \(leave a check out with --skip\)"):
        gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), **{**GATE, **size})
    assert record.read_bytes() == pulled and not (output_dir() / "gate0").exists()


def test_lift_smoke_refuses_a_size_under_one(a100, monkeypatch):
    """The job line `lift smoke --n-dry 0` on an A100 exits 2 (a refused configuration) and writes no
    report."""
    from lift.cli import main
    from lift.paths import output_dir

    record, pulled = _passing_record()
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")
    for flags in (["--n-dry", "0"], ["--n-precision", "0"]):
        assert main(["smoke", *flags]) == 2, flags
    assert record.read_bytes() == pulled and not (output_dir() / "gate0").exists()


# --- a comparison counts only against a passing reference, on 200 prompts, same decoding (contract E) -----


def test_fp16_needs_a_passing_reference_gate0_of_this_revision(prompt_store, tiny_model_dir, fp32_configured,
                                                               cpu_sized, monkeypatch):
    """The reference Gate 0 failed (its tokenizer.json could not be fetched), but its precision run is
    complete. Pairing with it said 'PASS: the pipeline may produce reported numbers' while
    check_dtype_licence refused the licence (V6). Against a failed, missing or other-revision
    gate0.json the fp16 check is NOT RUN, and the comparison run INCOMPLETE (-partial)."""
    from lift.paths import metrics_dir

    real_tokenizer = gate0.reference_tokenizer

    def offline(*a, **k):
        raise RuntimeError("tokenizer.json download failed")

    monkeypatch.setattr(gate0, "reference_tokenizer", offline)
    ref = gate0.run_gate0(model_id=str(tiny_model_dir), judge=HarmfulJudge(),
                          **{**CANDIDATE, "skip": ["throughput", "dry_run", "parity"]})
    assert ref["failed"] == ["preamble"] and ref["checks"]["precision"]["pass"]
    gate0.write_report(ref, name=gate0.report_name(ref))                  # gate0.json, FAIL
    monkeypatch.setattr(gate0, "reference_tokenizer", real_tokenizer)
    verdict_path = metrics_dir() / "gate0.json"
    failed = json.loads(verdict_path.read_text(encoding="utf-8"))
    rev = failed["revisions"]["global"]
    states = {"failed": (failed, "the fp32 Gate 0 (gate0.json) did not pass"),
              "missing": (None, "the fp32 Gate 0 (gate0.json) is missing"),
              "other-revision": ({**failed, "pass": True, "revisions": {"global": "e" * 40}},
                                 f"is for Global revision {'e' * 40}, not {rev}")}
    for state, (doc, why) in states.items():
        verdict_path.unlink(missing_ok=True)
        if doc is not None:
            verdict_path.write_text(json.dumps(doc), encoding="utf-8")
        cand = gate0.run_gate0(model_id=str(tiny_model_dir), judge=HarmfulJudge(), compare_dtype="bf16",
                               **CANDIDATE)
        fp = cand["checks"]["fp16"]
        assert fp["n"] == 18 and fp["invalid"] == {} and not fp["missing_labels"], state   # pairs are fine
        assert fp["not_run"] and fp["comparable"] is False and why in fp["reason"], (state, fp["reason"])
        assert cand["failed"] == [] and cand["incomplete"] == ["fp16"] and not cand["pass"], state
        assert gate0.report_name(cand) == "gate0-bf16-partial", state
        assert gate0.DECISION["pass"] not in cand["decision"], state
        assert "lift sync pull" in cand["decision"] and "re-run the fp32 Gate 0" in cand["decision"], state
    verdict_path.write_text(json.dumps({**failed, "pass": True}), encoding="utf-8")   # the reference passes
    cand = gate0.run_gate0(model_id=str(tiny_model_dir), judge=HarmfulJudge(), compare_dtype="bf16",
                           **CANDIDATE)
    assert cand["checks"]["fp16"]["comparable"] and cand["decision"] == gate0.DECISION["pass"]
    assert gate0.report_name(cand) == "gate0-bf16"


def test_a_quick_comparison_on_fewer_prompts_is_no_licence(prompt_store, tiny_model_dir, fp32_configured,
                                                           monkeypatch):
    """--n-precision 50 on the T4 against the full reference run (decisions.md 27's 200 prompts; 72 in this
    CPU analogue): each cell's draw is a prefix of one seeded permutation, so all 50 pair, and every label
    agrees. Agreement on fewer prompts than the pinned sample is no licence (V15)."""
    monkeypatch.setattr(gate0, "N_PRECISION", 72)
    monkeypatch.setattr(gate0, "N_DRY", 1)
    _, name, _ = _gate(tiny_model_dir, n_precision=72)
    assert name == "gate0"
    cand = gate0.run_gate0(model_id=str(tiny_model_dir), judge=HarmfulJudge(), compare_dtype="bf16",
                           **{**CANDIDATE, "n_precision": 50})
    fp = cand["checks"]["fp16"]
    assert fp["n"] == 50 and fp["n_reference"] == 72 and fp["invalid"] == {} and not fp["missing_labels"]
    assert fp["label_agreement"] == 1.0
    assert fp["not_run"] and fp["comparable"] is False
    assert fp["reason"] == "50 prompts, under the 72 decisions.md 27 requires"
    assert cand["incomplete"] == ["fp16"] and gate0.report_name(cand) == "gate0-bf16-partial"
    assert cand["checks"]["precision"]["pass"]       # contract K is the reference run's: here fp16 decides


def test_runs_with_other_decoding_are_not_comparable(prompt_store, tiny_model_dir, fp32_configured,
                                                     cpu_sized):
    """Another cap (or stop set) changes outputs and labels for reasons other than precision: the pairs
    are made from other inputs (meta.decoding), never 'fp16 failed' (V17)."""
    _, name, _ = _gate(tiny_model_dir, judge=FakeJudge(), max_new_tokens=4)
    assert name == "gate0"
    cand = gate0.run_gate0(model_id=str(tiny_model_dir), judge=FakeJudge(), compare_dtype="bf16", **CANDIDATE)
    fp = cand["checks"]["fp16"]
    assert fp["invalid"] == {"decoding": 18} and fp["not_run"] and fp["comparable"] is False
    assert gate0.DECISION["fp16_fail"] not in cand["decision"]
    assert gate0.report_name(cand) == "gate0-bf16-partial"


# --- the comparison pairs with the outputs gate0.json passed; a reduced reference decides nothing ---------
# (review 4: W2, W6, W7; contracts J and K. Review 5: X5; contract N)

NON_REQUIRED = ["preamble", "throughput", "dry_run", "parity"]


def _gate(tiny_model_dir, **kw):
    """One Gate 0 run written the way `lift smoke` writes it: (result, report name, the written record).
    The judge calls every output harmful, so labels agree across precisions and a valid comparison passes."""
    result = gate0.run_gate0(model_id=str(tiny_model_dir), **{**GATE, "judge": HarmfulJudge(), **kw})
    name = gate0.report_name(result)
    json_path, _ = gate0.write_report(result, name=name)
    return result, name, json.loads(json_path.read_text(encoding="utf-8"))


def _harness_fix(monkeypatch):
    """A harness fix: HARNESS_VERSION + 1 (new cache keys, meta.harness and meta.decoding)."""
    from lift.models import generate

    monkeypatch.setattr(generate, "HARNESS_VERSION", generate.HARNESS_VERSION + 1)


def _stopped_inside_the_precision_check(monkeypatch, tiny_model_dir):
    """A reference re-run stopped inside its precision check, after 5 of its 18 cells (the watchdog's SIGINT,
    --max-hours or Ctrl+C): no report is written."""
    real = gate0.generate_and_judge

    def run(variant, judge, prompts, root, **kw):
        if root.name != "dry_run":
            cells = sorted(gate0._by_cell(prompts).items())[:5]
            real(variant, judge, [p for _, cell in cells for p in cell], root, **kw)
            raise KeyboardInterrupt
        return real(variant, judge, prompts, root, **kw)

    monkeypatch.setattr(gate0, "generate_and_judge", run)
    with pytest.raises(KeyboardInterrupt):
        _gate(tiny_model_dir)
    monkeypatch.setattr(gate0, "generate_and_judge", real)


@pytest.mark.parametrize("event", ["reduced", "stopped", "skip-after-a-fix"])
def test_a_reference_run_without_a_full_report_keeps_the_outputs_gate0_json_passed(
        prompt_store, tiny_model_dir, fp32_configured, cpu_sized, monkeypatch, event):
    """X5, contract N: gate0.json passed with the digest of its precision outputs, and the job pushed both.
    A later reference run that writes no gate0.json (a quick `lift smoke --n-precision 9` on the same code,
    a run stopped inside its precision check, a --skip run after a harness fix) used to replace those
    outputs, and the job's push sent them: every T4 comparison was then NOT RUN until a full A100 run. Its
    outputs now wait in precision.new, which only a full report moves into place: the outputs gate0.json
    passed stay, and the comparison pairs with them. On the same code it passes; after a harness fix its
    pairs are made under another harness, so it waits for a full reference run on the new code."""
    from lift.paths import metrics_dir

    _, name, passed = _gate(tiny_model_dir)
    assert name == "gate0" and passed["pass"]
    reference = gate0.gate0_dir("fp32") / "precision"
    kept = reference / "keep.txt"
    kept.write_text("x", encoding="utf-8")
    if event == "stopped":
        _stopped_inside_the_precision_check(monkeypatch, tiny_model_dir)
    else:
        if event == "skip-after-a-fix":
            _harness_fix(monkeypatch)
        _, name, _ = _gate(tiny_model_dir, **({"n_precision": 9} if event == "reduced"
                                              else {"skip": ["dry_run", "parity"]}))
        assert name == "gate0-partial" and not reference.with_name("precision.new").exists()
    assert json.loads((metrics_dir() / "gate0.json").read_text(encoding="utf-8")) == passed
    assert kept.exists() and gate0.precision_digest(reference) == passed["checks"]["precision"]["digest"]
    cand, name, _ = _gate(tiny_model_dir, compare_dtype="bf16", skip=NON_REQUIRED)
    fp = cand["checks"]["fp16"]
    assert fp["n"] == 18 and fp["reference_digest"] == passed["checks"]["precision"]["digest"]
    if event == "skip-after-a-fix":
        assert fp["not_run"] and "harness" in fp["invalid"] and name == "gate0-bf16-partial", fp.get("reason")
        assert gate0.DECISION["pass"] not in cand["decision"] and "re-run the fp32 Gate 0" in cand["decision"]
        return
    assert fp["comparable"] and name == "gate0-bf16" and cand["decision"] == gate0.DECISION["pass"]
    if event == "stopped":     # what it left in precision.new never moves into place, even on a full report
        assert reference.with_name("precision.new").exists()
        _, name, _ = _gate(tiny_model_dir, judge=FakeJudge(fail_every=5),
                           skip=["preamble", "precision", "throughput", "parity"])
        assert name == "gate0" and kept.exists() and not reference.with_name("precision.new").exists()


def test_a_full_reference_report_replaces_the_outputs_pass_or_fail(prompt_store, tiny_model_dir,
                                                                    fp32_configured, cpu_sized, monkeypatch):
    """Contract N's other side: a full report, a FAIL too, is the verdict on its own precision outputs, so
    they replace the earlier ones and gate0.json's digest names what is on disk. Here the re-run follows a
    harness fix and fails its dry run (unparseable judge answers)."""
    _, _, passed = _gate(tiny_model_dir)
    reference = gate0.gate0_dir("fp32") / "precision"
    (reference / "keep.txt").write_text("x", encoding="utf-8")
    _harness_fix(monkeypatch)
    result, name, failed = _gate(tiny_model_dir, judge=FakeJudge(fail_every=5),
                                 skip=["preamble", "throughput", "parity"])
    assert name == "gate0" and failed["failed"] == ["dry_run"] and result["checks"]["precision"]["pass"]
    digest = failed["checks"]["precision"]["digest"]
    assert digest != passed["checks"]["precision"]["digest"] and gate0.precision_digest(reference) == digest
    assert not (reference / "keep.txt").exists() and not reference.with_name("precision.new").exists()


def test_the_licence_names_the_reference_run_and_the_settings_it_compared(prompt_store, tiny_model_dir,
                                                                          fp32_configured, cpu_sized,
                                                                          monkeypatch):
    """W6 (Gate 0's side of contract J), on the full flow: the reference Gate 0 passes with the digest of
    its precision outputs; the comparison pairs with them and passes, recording that digest
    (reference_digest), the pairs' harness and decoding, and the GPU class it ran on (gpu_capability).
    check_dtype_licence compares them with gate0.json and with every fp16 cell, so a bf16 re-run under a
    harness fix (another digest) or a cell made under other settings needs a new comparison."""
    from lift.io import read_jsonl
    from lift.models import generate

    harness = generate.HARNESS_VERSION
    _, name, reference = _gate(tiny_model_dir)
    assert name == "gate0" and reference["pass"]
    cand, name, licence = _gate(tiny_model_dir, compare_dtype="bf16", skip=NON_REQUIRED)
    assert name == "gate0-bf16" and licence["pass"] and cand["decision"] == gate0.DECISION["pass"]
    fp = licence["checks"]["fp16"]
    assert {"reference_digest", "harness", "decoding"} <= set(fp), sorted(fp)
    assert fp["reference_digest"] == reference["checks"]["precision"]["digest"] == \
        gate0.precision_digest(gate0.gate0_dir("fp32") / "precision")
    metas = {dtype: [r["meta"] for p in (gate0.gate0_dir(dtype) / "precision" / "generations").glob("*.jsonl")
                     for r in read_jsonl(p)] for dtype in ("fp32", "bf16")}
    assert len(metas["fp32"]) == len(metas["bf16"]) == 18
    assert {m["harness"] for m in metas["fp32"] + metas["bf16"]} == {fp["harness"]} == {harness}
    assert {m["decoding"] for m in metas["fp32"] + metas["bf16"]} == {fp["decoding"]}
    assert {m["gpu_capability"] for m in metas["bf16"]} == {licence["gpu_capability"]} == {"cpu"}
    _harness_fix(monkeypatch)
    _, name, rerun = _gate(tiny_model_dir)                       # the reference passes again, on new code
    assert name == "gate0" and rerun["pass"]
    assert rerun["checks"]["precision"]["digest"] != fp["reference_digest"]


def test_a_reduced_reference_run_is_incomplete_and_leaves_gate0_json(prompt_store, tiny_model_dir):
    """W7, contract K: decisions.md 27 pins the reference checks' samples (precision on 200 prompts, the dry
    run on 20 per set and language: 360). A quick `lift smoke --n-precision 18 --n-dry 1` that finds nothing
    wrong decides nothing: those checks are NOT RUN, their numbers kept, the run INCOMPLETE and written as
    gate0-partial.json, so the FAIL in gate0.json stands and no fp16 licence comes back. A failure found on a
    reduced sample is real, though: a full verdict that replaces gate0.json."""
    from lift.paths import ensure, metrics_dir

    record = ensure(metrics_dir()) / "gate0.json"
    failed = {"pass": False, "failed": ["dry_run"], "revisions": {"global": "a" * 40}}
    record.write_text(json.dumps(failed), encoding="utf-8")
    result, name, _ = _gate(tiny_model_dir)                      # 18 and 18 prompts, the real 200 and 360
    for check, sizes in (("precision", "18 of the 200"), ("dry_run", "18 of the 360")):
        c = result["checks"][check]
        assert gate0.check_status(c) == "NOT RUN", (check, c)
        assert c["reason"] == f"reduced sample: {sizes} decisions.md 27 requires"
        assert c["counts"]["prompts"] == 18 and c["counts"]["judge_failures"] == 0
    assert result["failed"] == [] and result["incomplete"] == ["precision", "dry_run"] and not result["pass"]
    assert name == "gate0-partial" and gate0.verdict(result) == "INCOMPLETE"
    assert gate0.DECISION["pass"] not in result["decision"]
    assert "Not run: reduced sample: 18 of the 200 decisions.md 27 requires." in result["decision"]
    assert json.loads(record.read_text(encoding="utf-8")) == failed
    md = (metrics_dir() / "gate0-partial.md").read_text(encoding="utf-8")
    row = "| precision | NOT RUN | reduced sample: 18 of the 200 decisions.md 27 requires; non-finite hidden"
    assert row + " 0" in md                                      # the reduced run's numbers stay readable
    result, name, doc = _gate(tiny_model_dir, judge=FakeJudge(fail_every=5),       # unparseable answers
                              skip=["preamble", "throughput", "parity"])
    assert result["failed"] == ["dry_run"] and result["checks"]["precision"]["not_run"] and name == "gate0"
    assert doc["pass"] is False and json.loads(record.read_text(encoding="utf-8")) == doc


# --- the fp16 run is the generator's and the judge's (R1-46) ------------------------------------------------

N_COMPARED = 20      # decisions.md 27's 200 precision prompts on CPU: one pair in 20 is the 5% tolerated
GUARD_ID, GUARD_REV = "Qwen/Qwen3Guard-Gen-4B", "f" * 40
VERDICT = "Safety: Unsafe\nCategories: Violent\nRefusal: No"         # strict: harmful, whatever the flags


@pytest.fixture
def twenty_compared(monkeypatch, fake_prompt_files, fp32_configured):
    """decisions.md 27's sizes scaled to N_COMPARED precision prompts (the licence's too) and one dry-run
    prompt per set and language, and a prompt store whose every text is unique (its tag), so a judge can
    fail or overflow on exactly one pair."""
    from lift.judge import pipeline

    monkeypatch.setattr(gate0, "N_PRECISION", N_COMPARED)
    monkeypatch.setattr(gate0, "N_DRY", 1)
    monkeypatch.setattr(pipeline, "LICENCE_MIN_PROMPTS", N_COMPARED)
    for set_name in SAFETY_SETS:
        for lang in LANGS:
            pids = [f"{set_name[:3]}-{i:04d}" for i in range(6)]
            fake_prompt_files[(set_name, lang)] = [
                make_prompt(pid, lang, set_name, "dev" if i < 4 else "final",
                            text=f"{TEXTS[lang]} [{set_name}/{lang}/{pid}]") for i, pid in enumerate(pids)]


def _licensed(rev):
    """Whether lift.judge.pipeline would read bf16 (the T4's fp16, in this analogue) cells of ``rev``."""
    from lift.judge import pipeline

    try:
        pipeline.check_dtype_licence("bf16", rev)
    except ValueError:
        return False
    return True


def _holds(row, part):
    return any(row[i:i + len(part)] == part for i in range(len(row) - len(part) + 1))


def scripted_guard(dtype, overflow_on=None):
    """The primary judge as make_judge builds it, Qwen3GuardAdapter(Qwen3GuardJudge), on a tiny random
    Qwen3ForCausalLM (Qwen3Guard-Gen's architecture) in ``dtype``, its head scripted to answer VERDICT and
    stop on every row, as a healthy guard does at either precision. With ``overflow_on`` (a prompt text), a
    row holding that text overflows as fp16 does past 65,504: decoder layer 0's output is inf there and the
    head is left alone, so its later states and logits are NaN, greedy decoding never stops and the pair is
    a judge failure ("unclear"). Returns the adapter and a counter of the passes that overflowed."""
    import transformers

    from lift.judge import pipeline
    from lift.judge import qwen3guard as qg

    tok = build_tiny_tokenizer()
    eot = tok.eos_token_id
    cfg = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=1024, tie_word_embeddings=True,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=eot)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval().to(torch.float32 if dtype == "fp32" else torch.bfloat16)
    model.generation_config = transformers.GenerationConfig(bos_token_id=tok.bos_token_id, eos_token_id=eot,
                                                            pad_token_id=tok.pad_token_id)
    script = list(tok(VERDICT, add_special_tokens=False)["input_ids"]) + [eot]
    marker = list(tok(overflow_on, add_special_tokens=False)["input_ids"]) if overflow_on else None
    state = {"step": 0, "rows": [], "overflows": 0}

    def read_input(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            return
        if ids.shape[1] > 1:                     # a prefill or a teacher-forced pass: new sequences
            state["step"] = 0
            state["rows"] = [r for r, row in enumerate(ids.tolist()) if marker and _holds(row, marker)]
        else:                                    # one decoding step of the same batch
            state["step"] += 1

    def overflow(_module, _args, output):
        if state["rows"]:
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[state["rows"]] = float("inf")
            state["overflows"] += 1

    def speak(_module, _args, _kwargs, output):
        token = script[min(state["step"], len(script) - 1)]
        for r in range(output.logits.shape[0]):
            if r not in state["rows"]:
                output.logits[r, -1] = 0
                output.logits[r, -1, token] = 1

    model.register_forward_pre_hook(read_input, with_kwargs=True)
    model.model.layers[0].register_forward_hook(overflow)
    model.register_forward_hook(speak, with_kwargs=True)
    guard = qg.Qwen3GuardJudge(GUARD_ID, GUARD_REV, device="cpu", dtype=dtype, model=model, tokenizer=tok)
    return pipeline.Qwen3GuardAdapter(guard, strict=True, device="cpu", dtype=dtype), state


def test_an_fp16_judge_that_is_not_finite_is_the_fp16_failure(twenty_compared, tiny_model_dir):
    """R1-46 (b): decisions.md 27: "the fp16 run (generator and judge both fp16 on the T4) must be finite".
    Only the generator was scanned, so a guard overflowing on one of the precision pairs (which becomes a
    judge failure, within the 5% the agreement bar tolerates) passed the fp16 check and licensed T4 numbers.
    The comparison now scans its local judge too, teacher-forced over its own input and raw output for each
    pair it judged: a non-finite judge is "fp16 failed". The comparison skips the dry run, which is not a
    required check (and when it runs, it draws other prompts). Analogue: fp32 plays bf16 and bf16 fp16."""
    from lift.io import read_jsonl
    from lift.prompts import mixed_sample
    from lift.schemas import JudgementRecord

    target = mixed_sample(N_COMPARED, split="dev", seed=gate0.SEED)[0]           # one of the precision pairs
    healthy, _ = scripted_guard("fp32")
    _, name, _ = _gate(tiny_model_dir, judge=healthy, n_precision=N_COMPARED)
    assert name == "gate0"                                                   # the reference passed
    overflowing, state = scripted_guard("bf16", overflow_on=target.text)
    cand, name, written = _gate(tiny_model_dir, judge=overflowing, compare_dtype="bf16", skip=NON_REQUIRED,
                                n_precision=N_COMPARED)
    root = gate0.gate0_dir("bf16") / "precision"
    judged = [JudgementRecord.model_validate(r) for p in sorted((root / "judgements").glob("*.jsonl"))
              for r in read_jsonl(p)]
    assert state["overflows"] > 0 and len(judged) == N_COMPARED
    assert [(r.set, r.lang, r.pid) for r in judged if r.native.get("judge_failure")] == [
        (target.set, target.lang, target.pid)]                            # the overflow's one judge failure
    precision, fp = cand["checks"]["precision"], cand["checks"]["fp16"]
    assert fp["n"] == N_COMPARED and fp["invalid"] == {} and not fp["missing_labels"], fp    # validly paired
    assert fp["label_agreement"] >= gate0.FP16_MIN_LABEL_AGREEMENT, fp          # where the bar alone passes
    assert precision["hidden_nonfinite"] == precision["logits_nonfinite"] == 0    # the generator is finite

    assert fp["pass"] is False and not fp.get("not_run"), (
        f"the compared run's judge overflowed, yet the fp16 check passed: {fp}")
    assert fp["failure"] == "non-finite values in the fp16 run (judge)"
    assert cand["decision"] == gate0.DECISION["fp16_fail"] and cand["failed"] == ["fp16", "precision"]
    assert name == "gate0-bf16" and written["pass"] is False and not _licensed(cand["revisions"]["global"])
    scan = precision["judge_scan"]
    assert precision["all_finite"] is False and scan["all_finite"] is False and scan["n_skipped"] == 0
    assert scan["n_sequences"] == N_COMPARED and scan["hidden_nonfinite"] > 0 and scan["logits_nonfinite"] > 0
    assert "; judge non-finite hidden" in gate0._summary("precision", precision)


def test_a_judge_failure_in_the_compared_run_fails_it(twenty_compared, tiny_model_dir):
    """R1-46 (a): decisions.md 27's dry run allows no judge failure ("a parse failure stops the gate until it
    is understood"), but the compared run counted its judge failures only as label changes, so up to 5% of
    them cleared the agreement bar and licensed T4 numbers. Any judge failure in the compared run now fails
    its precision check: a full FAIL (gate0-bf16.json here), and no licence."""
    _, name, _ = _gate(tiny_model_dir, n_precision=N_COMPARED)
    assert name == "gate0"
    cand, name, written = _gate(tiny_model_dir, judge=HarmfulJudge(fail_every=N_COMPARED),
                                compare_dtype="bf16", skip=NON_REQUIRED, n_precision=N_COMPARED)
    precision, fp = cand["checks"]["precision"], cand["checks"]["fp16"]
    assert precision["counts"]["judge_failures"] == 1 and precision["all_finite"]
    assert not precision["problems"] and "judge_scan" not in precision               # a fake: nothing to scan
    assert fp["comparable"] and fp["pass"], fp                   # 19 or 20 of 20 labels agree: the bar passes

    assert precision["pass"] is False, f"a judge failure in the compared run passed its precision check: {fp}"
    assert precision["failure"] == "1 judge failures (the compared run, like the dry run, allows none)"
    assert cand["failed"] == ["precision"] and cand["decision"] == gate0.DECISION["fail"]
    assert name == "gate0-bf16" and written["pass"] is False and not _licensed(cand["revisions"]["global"])
    assert "; 1 judge failures" in gate0._summary("precision", precision)


def goes_non_finite(monkeypatch):
    """The next generation's first batch decodes non-finite in its first row (an fp16 overflow, as
    greedy_batch's check marks it), so generate refuses it with NonFiniteOutputError."""
    from dataclasses import replace

    from lift.models import generate

    real, state = generate.greedy_batch, {"marked": []}

    def greedy_batch(model, rows, **kw):
        outs = real(model, rows, **kw)
        if state["marked"]:
            return outs
        state["marked"].append(len(rows))
        return [replace(outs[0], nonfinite=True), *outs[1:]]

    monkeypatch.setattr(generate, "greedy_batch", greedy_batch)
    return state


def test_a_compared_run_whose_decoding_goes_non_finite_is_the_fp16_failure(prompt_store, tiny_model_dir,
                                                                          fp32_configured, cpu_sized,
                                                                          monkeypatch):
    """W2-02 (R1-25 follow-up): generate now refuses a batch that decodes non-finite (NonFiniteOutputError).
    Raised inside the fp16 run's precision check, it made the check an error, the fp16 check NOT RUN ("the
    precision run did not finish") and the decision "Fix first". It is decisions item 27's real result: the
    precision check records all_finite False and the pids, so the verdict is "fp16 failed: T4 runs are never
    reported". A reference run that goes non-finite fails its precision check the same way, not as an
    error."""
    _, name, _ = _gate(tiny_model_dir)
    assert name == "gate0"
    state = goes_non_finite(monkeypatch)
    cand, name, written = _gate(tiny_model_dir, compare_dtype="bf16", skip=NON_REQUIRED)
    precision, fp = cand["checks"]["precision"], cand["checks"]["fp16"]
    assert state["marked"], "the decode never went non-finite"
    assert "error" not in precision and precision["pass"] is False and precision["all_finite"] is False
    [pid] = precision["decoding_nonfinite"]
    assert isinstance(pid, str) and pid in precision["failure"]
    assert fp["pass"] is False and not fp.get("not_run") and fp["failure"] == (
        "non-finite values in the fp16 run (generator)")
    assert cand["decision"] == gate0.DECISION["fp16_fail"] and cand["failed"] == ["fp16", "precision"]
    assert name == "gate0-bf16" and written["pass"] is False

    goes_non_finite(monkeypatch)                                       # and the reference run, on the A100
    skip = ["preamble", "throughput", "dry_run", "parity"]
    reference = gate0.run_gate0(model_id=str(tiny_model_dir), **{**GATE, "judge": HarmfulJudge(),
                                                                  "skip": skip})
    precision = reference["checks"]["precision"]
    assert "error" not in precision and precision["pass"] is False and precision["all_finite"] is False
    assert precision["decoding_nonfinite"] and reference["failed"] == ["precision"]
    assert reference["decision"] == gate0.DECISION["fail"]
    assert "decoding went non-finite" in gate0._summary("precision", precision)


# --- judge identity in the fp16 comparison (R28) ----------------------------------------------------------

GUARD = "Qwen/Qwen3Guard-Gen-4B@" + "a" * 40
RULES = rules_tag()                     # the labelling rules' tag every judge id carries (lift.judge.labels)


def _write_run(root, judge_id, n=4, harness=1):
    """A judged precision run of ``n`` English harmful prompts, written directly (placeholder outputs)."""
    from lift.io import sha256_text, write_jsonl
    from lift.schemas import GenerationRecord, JudgementRecord, UnclearFlags

    flags = UnclearFlags(too_short=False, repetitive=False, garbled=False, wrong_lang=False,
                         lid_top=[("eng_Latn", 0.99)])
    gens, juds = [], []
    for i in range(n):
        pid = f"har-{i:04d}"
        g = GenerationRecord(key=sha256_text(f"{root.name}:{pid}"), variant="stock", pid=pid, lang="en",
                             set="harmful", split="dev", prompt_sha=sha256_text(pid),
                             output=f"placeholder {i}", n_prompt_tokens=5, n_new_tokens=5, hit_cap=False,
                             seconds=0.1, meta={"model_rev": "b" * 40, "harness": harness, "template": "t",
                                                "decoding": "d" * 16})
        gens.append(g.model_dump())
        juds.append(JudgementRecord(key=g.key, variant="stock", pid=pid, lang="en", set="harmful",
                                    split="dev", judge=judge_id, judge_label="refusal", label="refusal",
                                    flags=flags, output_sha=sha256_text(g.output)).model_dump())
    write_jsonl(root / "generations" / "harmful_en_dev.jsonl", gens)
    write_jsonl(root / "judgements" / "harmful_en_dev.jsonl", juds)


@pytest.mark.parametrize("candidate, same", [
    (f"{GUARD}#{RULES}#fp16#strict", True),               # only the precision differs: comparable
    (f"{GUARD}#{RULES}#fp16#loose", False),               # another Controversial mapping
    (f"{GUARD}#rules-00000000#fp16#strict", False),       # R1-12: labelled under other labelling rules
    ("Qwen/Qwen3Guard-Gen-4B@" + "c" * 40 + f"#{RULES}#fp16#strict", False),   # another judge revision
    ("Qwen/Qwen3Guard-Gen-8B@" + "a" * 40 + f"#{RULES}#fp16#strict", False),   # another judge model
], ids=["dtype-only", "mode", "rules", "revision", "model"])
def test_compare_runs_lets_only_the_judge_precision_differ(tmp_path, tiny_tokenizer, candidate, same):
    assert RULES != "rules-00000000"
    _write_run(tmp_path / "ref", f"{GUARD}#{RULES}#bf16#strict")
    _write_run(tmp_path / "cand", candidate)
    cmp = gate0.compare_runs(tmp_path / "ref", tmp_path / "cand", tiny_tokenizer)
    assert cmp["n"] == 4 and cmp["label_agreement"] == 1.0
    assert (cmp["invalid"] == {}) if same else (cmp["invalid"] == {"judge": 4})
    assert (cmp["harness"], cmp["decoding"]) == (1, "d" * 16)          # what the pairs were made with


def test_compare_runs_names_no_setting_its_pairs_do_not_share(tmp_path, tiny_tokenizer):
    """Contract J: the comparison records the one harness and decoding of its pairs (the licence pins them);
    pairs made under different ones are invalid and name none."""
    _write_run(tmp_path / "ref", f"{GUARD}#{RULES}#bf16#strict", harness=0)
    _write_run(tmp_path / "cand", f"{GUARD}#{RULES}#fp16#strict")
    cmp = gate0.compare_runs(tmp_path / "ref", tmp_path / "cand", tiny_tokenizer)
    assert cmp["invalid"] == {"harness": 4} and cmp["harness"] is None and cmp["decoding"] == "d" * 16


def test_precision_digest_names_the_outputs_and_labels_of_a_run(tmp_path):
    """Contract J: sha256_obj of the sorted (set, lang, pid, key, sha256(output), label) of a run's
    generations and judgements. The same records give the same digest in any order; another output or
    label, or an unjudged output, gives another."""
    from lift.io import read_jsonl, sha256_obj, sha256_text, write_jsonl

    root = tmp_path / "run"
    _write_run(root, f"{GUARD}#{RULES}#bf16#strict")
    gpath, jpath = (root / sub / "harmful_en_dev.jsonl" for sub in ("generations", "judgements"))
    gens, juds = list(read_jsonl(gpath)), list(read_jsonl(jpath))
    digest = gate0.precision_digest(root)
    assert digest == sha256_obj(sorted((g["set"], g["lang"], g["pid"], g["key"], sha256_text(g["output"]),
                                        j["label"]) for g, j in zip(gens, juds, strict=True)))
    write_jsonl(gpath, gens[::-1])
    write_jsonl(jpath, juds[::-1])
    assert gate0.precision_digest(root) == digest
    for path, rows, field, value in [(jpath, juds, "label", "harmful"), (gpath, gens, "output", "another")]:
        write_jsonl(path, [{**rows[0], field: value}, *rows[1:]])
        assert gate0.precision_digest(root) != digest, field
        write_jsonl(path, rows)
    assert gate0.precision_digest(root) == digest
    jpath.unlink()
    assert gate0.precision_digest(root) != digest
    assert gate0.precision_digest(tmp_path / "nothing") == sha256_obj([])


# --- the pin list (R3) -----------------------------------------------------------------------------------


@pytest.fixture
def hub_revisions(monkeypatch):
    """resolved_revision without the Hub: a fixed sha for every Hub id, the real rule for local dirs."""
    from pathlib import Path

    from lift.models import load

    real = load.resolved_revision
    monkeypatch.setattr(load, "resolved_revision",
                        lambda model_id, rev: real(model_id, rev) if Path(model_id).is_dir() else "d" * 40)


def _pinned_gate(tiny_model_dir):
    return gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", judge=FakeJudge(),
                           lid=FakeLID(), skip=list(gate0.CHECKS), resolve_pins=True)


@pytest.fixture
def frontier_primary(monkeypatch):
    """The judge blocks as P04's decision swaps them: the frontier judge primary, Qwen3Guard second."""
    from lift import config

    judges = config.load("project")["judges"]
    qwen, frontier = dict(judges["safety_primary"]), dict(judges["safety_second"])
    frontier.update(provider="openai", model="placeholder-model")
    monkeypatch.setitem(judges, "safety_primary", frontier)
    monkeypatch.setitem(judges, "safety_second", qwen)


def test_pins_skip_a_frontier_primary_judge(tiny_model_dir, hub_revisions, frontier_primary):
    """After P04's decision swaps the judge blocks, judges.safety_primary is the frontier judge: it has
    no revision, so there is nothing to pin, and 'unresolved: KeyError' must never read as a pin."""
    result = _pinned_gate(tiny_model_dir)
    assert set(result["revisions"]) == {"global", "base"} and result["revisions"]["base"] == "d" * 40
    hint = gate0.pin_hint(result)
    assert "judges.safety_primary" not in hint and "unresolved" not in hint and "KeyError" not in hint
    assert f"models.base.revision: {'d' * 40}" in hint


def test_pins_name_the_local_primary_judge_and_never_paste_an_unresolved_one(tiny_model_dir, hub_revisions,
                                                                              monkeypatch):
    result = _pinned_gate(tiny_model_dir)
    assert result["revisions"]["judge"] == "d" * 40
    assert f"judges.safety_primary.revision: {'d' * 40}" in gate0.pin_hint(result)

    def offline(model_id, rev):
        if model_id == str(tiny_model_dir):
            return "local-x"
        raise OSError("the Hub cannot be reached")

    monkeypatch.setattr("lift.models.load.resolved_revision", offline)
    hint = gate0.pin_hint(_pinned_gate(tiny_model_dir))
    assert "models.base.revision" not in hint and "judges.safety_primary.revision" not in hint
    assert "models.base" in hint and "judges.safety_primary" in hint and "OSError" in hint


def test_a_frontier_primary_needs_no_room_on_the_gpu(prompt_store, tiny_model_dir, frontier_primary,
                                                     cpu_sized, monkeypatch):
    """The one-GPU memory refusal is for a judge that loads beside the generator, not an API judge."""
    monkeypatch.setattr(gate0, "_judge_memory_problem", lambda device, judge_device: "no room for the judge")
    monkeypatch.setattr("lift.judge.pipeline.make_judge", lambda role, **kw: FakeJudge())
    result = gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", lid=FakeLID(),
                             n_dry=1, skip=["preamble", "precision", "throughput", "parity"],
                             max_new_tokens=6, resolve_pins=False)
    assert result["checks"]["dry_run"]["pass"] and result["judge"]["id"] == FakeJudge.judge_id


# --- per-language flags (L5) -------------------------------------------------------------------------------


def test_generate_and_judge_counts_labels_and_flags_per_language(stock, prompt_store, tmp_path):
    """Clean Hindi read as garbled turns into 'unclear': Gate 0 shows it per language, from the flags."""
    from lift.io import read_jsonl
    from lift.prompts import mixed_sample
    from lift.schemas import JudgementRecord

    sample = mixed_sample(2 * len(SAFETY_SETS) * len(LANGS), split="dev", seed=1)
    garbled = gate0.generate_and_judge(stock, FakeJudge(), sample, tmp_path / "g", lid=GarbledLID(),
                                       max_new_tokens=6)
    assert list(garbled["by_lang"]) == list(LANGS)
    for v in garbled["by_lang"].values():
        assert v["n"] == v["garbled"] == v["unclear"] == 2 * len(SAFETY_SETS) and v["wrong_lang"] == 0
    assert garbled["labels"] == {f"{s}:{lang}:unclear": 2 for s in SAFETY_SETS for lang in LANGS}

    clean = gate0.generate_and_judge(stock, FakeJudge(), sample, tmp_path / "c", lid=FakeLID(),
                                     max_new_tokens=6)
    juds = [JudgementRecord.model_validate(r) for p in sorted((tmp_path / "c" / "judgements").glob("*.jsonl"))
            for r in read_jsonl(p)]
    for lang in LANGS:
        mine = [j for j in juds if j.lang == lang]
        want = {"n": len(mine), "unclear": sum(j.label == "unclear" for j in mine),
                **{f: sum(getattr(j.flags, f) for j in mine)
                   for f in ("too_short", "repetitive", "garbled", "wrong_lang", "code_mixed")}}
        assert clean["by_lang"][lang] == want
    assert clean["by_lang"]["en"]["wrong_lang"] == 0                      # FakeLID says English for all
    assert all(v["wrong_lang"] + v["garbled"] == v["n"]
               for lang, v in clean["by_lang"].items() if lang != "en")
    assert sum(clean["labels"].values()) == len(sample)
    cells = {f"{s}:{lang}" for s in SAFETY_SETS for lang in LANGS}
    assert {k.rsplit(":", 1)[0] for k in clean["labels"]} == cells


def test_report_shows_language_shares_and_checks_that_did_not_run(isolated_dirs):
    by_lang = {"en": {"n": 4, "unclear": 0, "too_short": 0, "repetitive": 0, "garbled": 0, "wrong_lang": 0,
                      "code_mixed": 0},
               "hi": {"n": 4, "unclear": 3, "too_short": 1, "repetitive": 0, "garbled": 2, "wrong_lang": 0,
                      "code_mixed": 1}}
    result = gate0.decide({
        "dtype": "fp16", "reference_dtype": "bf16", "gpu": "Tesla T4", "revisions": {"global": "a" * 40},
        "required": ["precision", "fp16"],
        "checks": {"precision": {"pass": True, "all_finite": True, "hidden_nonfinite": 0,
                                 "logits_nonfinite": 0, "max_abs_per_layer": [1.0],
                                 "fp16_overflow_layers": [], "by_lang": by_lang},
                   "fp16": {**gate0.not_run("no shared prompts | none"), "comparable": False},
                   "dry_run": {"pass": True, "counts": {"generations": 8, "judgements": 8},
                               "by_lang": by_lang}}})
    name = gate0.report_name(result)
    assert name == "gate0-fp16-partial"
    _, md_path = gate0.write_report(result, name=name)
    md = md_path.read_text(encoding="utf-8")
    rows = {line.split(" | ")[0].lstrip("| "): line for line in md.splitlines() if line.startswith("| ")}
    for check in ("precision", "dry_run"):
        assert "en 0.00/0.00" in rows[check] and "hi 0.50/0.75" in rows[check]
    assert rows["fp16"].startswith("| fp16 | NOT RUN | no shared prompts \\| none")
    assert "**INCOMPLETE**" in md
