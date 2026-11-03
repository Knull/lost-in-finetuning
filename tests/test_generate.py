import dataclasses
import json

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fakes import make_prompt  # noqa: E402

from lift.io import sha256_text  # noqa: E402
from lift.models import generate as gen  # noqa: E402
from lift.models.load import encode_chat, stop_token_ids  # noqa: E402
from lift.models.variants import build_variant  # noqa: E402
from lift.schemas import VariantSpec  # noqa: E402

CAP = 12
TEXTS = ["hi", "नमस्ते, आप कैसे हैं?", "你好，世界", "Explain photosynthesis in one long sentence, please.",
         "¿Dónde está la biblioteca?", "ok"]


@pytest.fixture(scope="module")
def stock(tiny_model_dir):
    return build_variant(VariantSpec(name="stock", method="stock"), device="cpu", dtype="fp32",
                         model_id=str(tiny_model_dir))


def prompts(texts=TEXTS, lang="en"):
    return [make_prompt(f"p-{i:03d}", lang, text=t) for i, t in enumerate(texts)]


def run(variant, path, ps=None, **kw):
    kw.setdefault("max_new_tokens", CAP)
    gen.generate(variant, prompts() if ps is None else ps, out_path=path, **kw)
    return {r.pid: r for r in gen.read_generations(path)}


def manual_greedy(model, ids, stop, cap):
    """Plain argmax decoding with no cache and no logits processors: the reference for "greedy"."""
    seq, out = list(ids), []
    for _ in range(cap):
        with torch.no_grad():
            nxt = int(model(torch.tensor([seq])).logits[0, -1].argmax())
        if nxt in stop:
            return out, True
        out.append(nxt)
        seq.append(nxt)
    return out, False


def test_batched_equals_unbatched(stock, tmp_path):
    one = run(stock, tmp_path / "a.jsonl", batch_size=1)
    four = run(stock, tmp_path / "b.jsonl", batch_size=4)
    assert one.keys() == four.keys() and len(one) == len(TEXTS)
    # Guard the test itself: rows must differ, or a broken mask or padding would pass unnoticed.
    assert len({r.output for r in one.values()}) == len(TEXTS)
    for pid in one:
        assert one[pid].output == four[pid].output
        assert one[pid].n_new_tokens == four[pid].n_new_tokens
        assert one[pid].key == four[pid].key


def test_harness_is_pure_greedy_despite_the_checkpoint_sampling_defaults(stock):
    # tiny_model_dir ships do_sample, temperature, top_k/top_p, repetition_penalty and
    # no_repeat_ngram_size in its generation_config.json; none of them may reach decoding.
    stop = set(stop_token_ids(stock.tokenizer, stock.model))
    outs = gen.greedy_token_ids(stock.model, stock.tokenizer, TEXTS, device="cpu", max_new_tokens=CAP,
                                batch_size=3)
    for text, out in zip(TEXTS, outs, strict=True):
        ref, stopped = manual_greedy(stock.model, encode_chat(stock.tokenizer, text), stop, CAP)
        assert out.ids == ref and out.stopped == stopped


def test_unneutralised_model_is_refused(tiny_model_dir, stock, tmp_path):
    from transformers import AutoModelForCausalLM

    raw = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    with pytest.raises(ValueError, match="neutralises"):
        gen.generate(dataclasses.replace(stock, model=raw), prompts(), out_path=tmp_path / "x.jsonl",
                     max_new_tokens=CAP)


def test_records_carry_hashes_counts_and_meta(stock, tmp_path, isolated_dirs):
    recs = run(stock, tmp_path / "g.jsonl", batch_size=2)
    p = prompts()[1]
    r = recs[p.pid]
    assert r.prompt_sha == sha256_text(p.text)
    assert r.n_prompt_tokens == len(encode_chat(stock.tokenizer, p.text))
    assert (r.variant, r.lang, r.set, r.split) == ("stock", "en", "harmful", "dev")
    assert r.meta["dtype"] == "fp32" and r.meta["gpu"] == "cpu" and r.meta["batch_size"] in (1, 2)
    assert all(not x.hit_cap or x.meta["n_generated"] == CAP for x in recs.values())
    assert all(x.meta["n_generated"] <= CAP for x in recs.values())
    provenance = ("model_rev", "harness", "template", "torch", "transformers", "gpu_capability", "run_config")
    for field in provenance:
        assert r.meta[field]
    assert r.meta["model_rev"] == stock.model_rev and r.meta["harness"] == gen.HARNESS_VERSION
    manifests = sorted((isolated_dirs / "outputs" / "manifests").glob("generate-*.json"))
    assert len(manifests) == 1
    doc = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert doc["n_written"] == len(TEXTS) and doc["prompts_per_sec"] > 0 and doc["tokens_per_sec"] > 0
    assert doc["gen_cfg"]["dtype"] == "fp32" and doc["gen_cfg"]["max_new_tokens"] == CAP


def test_hit_cap_and_stop(stock):
    ids = encode_chat(stock.tokenizer, "hi")
    pad = stock.tokenizer.pad_token_id
    capped = gen.greedy_batch(stock.model, [ids], pad_id=pad, stop_ids=[-1], max_new_tokens=5, device="cpu")
    assert not capped[0].stopped and len(capped[0].ids) == 5 and capped[0].n_generated == 5
    first = capped[0].ids[0]
    # Make the first token the model emits a stop token: zero content, stopped, not capped.
    stopped = gen.greedy_batch(stock.model, [ids], pad_id=pad, stop_ids=[first], max_new_tokens=5,
                               device="cpu")
    assert stopped[0].stopped and stopped[0].ids == [] and stopped[0].n_generated == 1


def test_n_new_tokens_excludes_special_tokens(stock):
    tok = stock.tokenizer
    special = tok.convert_tokens_to_ids("<|START_RESPONSE|>")
    a, b = tok("ab", add_special_tokens=False)["input_ids"]
    row = gen._Row(make_prompt("p"), "k" * 64, [1, 2, 3])
    rec = gen._record(stock, row, gen.GreedyOutput(ids=[special, a, b], stopped=True, n_generated=4), 0.5, 1,
                      "cpu", tok, {})
    assert rec.n_new_tokens == 2 and rec.output == "ab" and not rec.hit_cap


def test_resume_after_a_crash_writes_no_duplicates(stock, tmp_path, monkeypatch):
    clean = run(stock, tmp_path / "clean.jsonl", batch_size=2)
    path = tmp_path / "crash.jsonl"
    real = gen.greedy_batch
    calls = {"n": 0}

    def dies_on_second_batch(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("session killed")
        return real(*a, **kw)

    monkeypatch.setattr(gen, "greedy_batch", dies_on_second_batch)
    with pytest.raises(RuntimeError, match="killed"):
        run(stock, path, batch_size=2)
    monkeypatch.setattr(gen, "greedy_batch", real)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    with open(path, "ab") as f:                       # and the kill landed mid-line
        f.write(b'{"key": "abc", "outp')
    resumed = run(stock, path, batch_size=2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(TEXTS) == len(resumed)
    assert {r.key for r in resumed.values()} == {r.key for r in clean.values()}
    assert all(resumed[p].output == clean[p].output for p in clean)


def test_resume_skips_done_prompts_and_appends_new_ones(stock, tmp_path):
    path = tmp_path / "g.jsonl"
    run(stock, path, ps=prompts()[:3])
    run(stock, path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(TEXTS)


def test_stale_records_are_never_mixed(stock, tmp_path):
    path = tmp_path / "g.jsonl"
    run(stock, path)
    fixed = prompts()
    fixed[0] = make_prompt(fixed[0].pid, text=fixed[0].text + " (fixed translation)")
    with pytest.raises(gen.StaleOutputError, match="p-000"):
        run(stock, path, ps=fixed)
    with pytest.raises(gen.StaleOutputError):
        run(dataclasses.replace(stock, dtype="fp16"), path)        # another precision, other keys
    with pytest.raises(FileExistsError):
        run(stock, path, resume=False)


def test_a_file_never_mixes_configurations_even_outside_the_current_prompts(stock, tmp_path):
    # A --limit run in one dtype, then a run on other prompts in another dtype, into the same file.
    path = tmp_path / "g.jsonl"
    run(stock, path, ps=prompts()[:2])
    with pytest.raises(gen.StaleOutputError, match="another configuration"):
        run(dataclasses.replace(stock, dtype="fp16"), path, ps=prompts()[2:])
    assert all(r.meta["run_config"] == gen.read_generations(path)[0].meta["run_config"]
               for r in gen.read_generations(path))


def test_a_resume_under_another_stack_is_refused(stock, tmp_path, monkeypatch):
    # The runtime upgraded transformers (or moved to another GPU class) between two sessions.
    path = tmp_path / "g.jsonl"
    run(stock, path, ps=prompts()[:2])
    real = gen.stack
    monkeypatch.setattr(gen, "stack", lambda device: {**real(device), "transformers": "5.99.0"})
    with pytest.raises(gen.StaleOutputError, match="transformers"):
        run(stock, path, ps=prompts()[2:])


class _GenerateOnly:
    """A CAST-like wrapper: only .generate, no config or generation_config of its own."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def generate(self, **kw):
        self.calls += 1
        return self.inner.generate(**kw)


def test_a_wrapper_with_only_generate_uses_its_base_models_config_and_stop_set(stock, tmp_path):
    wrapper = _GenerateOnly(stock.model)
    wrapped = dataclasses.replace(stock, model=wrapper, base=stock.model,
                                  batch_meta=lambda: [{"gate_fired": True}] * 2)
    stops = gen.gen_config(stock, max_new_tokens=CAP)["stop_ids"]
    assert gen.gen_config(wrapped, max_new_tokens=CAP)["stop_ids"] == stops
    got = run(wrapped, tmp_path / "w.jsonl", ps=prompts()[:2], batch_size=2)
    plain = run(stock, tmp_path / "s.jsonl", ps=prompts()[:2], batch_size=2)
    assert wrapper.calls == 1 and all(r.meta["gate_fired"] for r in got.values())
    assert {p: r.output for p, r in got.items()} == {p: r.output for p, r in plain.items()}


def test_oom_retry_happens_outside_the_except_clause(stock, tmp_path, monkeypatch):
    import sys

    real = gen.greedy_batch
    seen = []

    def oom_then_record(model, rows, **kw):
        seen.append((len(rows), sys.exc_info()[0]))
        if len(rows) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (mock)")
        return real(model, rows, **kw)

    monkeypatch.setattr(gen, "greedy_batch", oom_then_record)
    run(stock, tmp_path / "g.jsonl", ps=prompts()[:2], batch_size=2)
    assert [s for s, _ in seen] == [2, 1, 1]
    assert all(exc is None for _, exc in seen)           # no live exception while retrying


def test_keys_differ_across_variants_prompts_and_dtypes(stock):
    cfg = gen.gen_config(stock, max_new_tokens=CAP)
    p = prompts()[0]
    base = gen.record_key(stock, p, cfg)
    adapter = dataclasses.replace(stock, spec=VariantSpec(name="adapter-x", method="adapter", adapter="x",
                                                          seed=1), adapter_hash="h" * 64)
    assert gen.record_key(adapter, p, cfg) != base
    assert gen.record_key(stock, make_prompt(p.pid, text=p.text + "!"), cfg) != base
    assert gen.record_key(stock, make_prompt(p.pid, lang="hi", text=p.text), cfg) != base
    fp16 = dataclasses.replace(stock, dtype="fp16")
    assert gen.gen_config(fp16, max_new_tokens=CAP)["dtype"] == "fp16"
    assert gen.record_key(fp16, p, gen.gen_config(fp16, max_new_tokens=CAP)) != base
    renamed = dataclasses.replace(stock, spec=VariantSpec(name="stock-again", method="stock"))
    assert gen.record_key(renamed, p, cfg) == base                 # the name is not behaviour


def test_out_of_memory_halves_the_batch_and_never_skips(stock, tmp_path, monkeypatch):
    clean = run(stock, tmp_path / "clean.jsonl", batch_size=4)
    real = gen.greedy_batch
    sizes = []

    def oom_above_one(model, rows, **kw):
        sizes.append(len(rows))
        if len(rows) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB (mock)")
        return real(model, rows, **kw)

    monkeypatch.setattr(gen, "greedy_batch", oom_above_one)
    got = run(stock, tmp_path / "oom.jsonl", batch_size=4)
    assert sizes[:3] == [4, 2, 1] and set(sizes[3:]) <= {1, 2}
    assert got.keys() == clean.keys()
    assert all(got[p].output == clean[p].output and got[p].meta["batch_size"] == 1 for p in got)

    def always_oom(model, rows, **kw):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (mock)")

    monkeypatch.setattr(gen, "greedy_batch", always_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run(stock, tmp_path / "dead.jsonl", batch_size=2)


def test_too_long_prompt_raises_instead_of_truncating(stock, tmp_path):
    long = make_prompt("p-long", text="x" * 700)
    with pytest.raises(ValueError, match="p-long"):
        gen.generate(stock, [long], out_path=tmp_path / "g.jsonl")             # 256 new tokens: too long
    assert not (tmp_path / "g.jsonl").exists()                                # nothing was written
    gen.generate(stock, [long], out_path=tmp_path / "g.jsonl", max_new_tokens=8)
    assert len(gen.read_generations(tmp_path / "g.jsonl")) == 1


def test_duplicate_prompts_are_refused_but_a_pid_may_repeat_across_languages(stock, tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        gen.generate(stock, [make_prompt("a"), make_prompt("a")], out_path=tmp_path / "g.jsonl")
    both = [make_prompt("a", "en", text="hello"), make_prompt("a", "hi", text="नमस्ते")]
    got = gen.read_generations(gen.generate(stock, both, out_path=tmp_path / "mixed.jsonl", max_new_tokens=4))
    assert sorted((r.lang, r.pid) for r in got) == [("en", "a"), ("hi", "a")]
    gen.generate(stock, both, out_path=tmp_path / "mixed.jsonl", max_new_tokens=4)     # resumes: nothing new
    assert len(gen.read_generations(tmp_path / "mixed.jsonl")) == 2


def test_a_crashed_run_still_writes_its_manifest(stock, tmp_path, isolated_dirs, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("kernel died")

    monkeypatch.setattr(gen, "greedy_batch", boom)
    with pytest.raises(RuntimeError):
        run(stock, tmp_path / "g.jsonl")
    docs = [json.loads(p.read_text(encoding="utf-8"))
            for p in (isolated_dirs / "outputs" / "manifests").glob("generate-*.json")]
    assert [d["status"] for d in docs] == ["failed"] and docs[0]["error"] == "RuntimeError"


def test_repair_tail_and_read_generations(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert gen.repair_tail(path) and path.read_bytes() == b'{"a": 1}\n{"a": 2}\n'
    path.write_bytes(b'{"a": 1}\n{"a": ')
    assert gen.repair_tail(path) and path.read_bytes() == b'{"a": 1}\n'
    assert not gen.repair_tail(path)
    rec = {"key": "k", "variant": "v", "pid": "p", "lang": "en", "set": "harmful", "split": "dev",
           "prompt_sha": "s", "output": "o", "n_prompt_tokens": 1, "n_new_tokens": 1, "hit_cap": False,
           "seconds": 0.1}
    path.write_text(json.dumps(rec) + "\n" + json.dumps(dict(rec, key="k2")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        gen.read_generations(path)
