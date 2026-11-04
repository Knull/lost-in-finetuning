import dataclasses
import json
import logging
import re

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
    provenance = ("model_rev", "harness", "template", "torch", "transformers", "gpu_capability", "stack",
                  "decoding", "run_config")
    for field in provenance:
        assert r.meta[field]
    assert r.meta["model_rev"] == stock.model_rev and r.meta["harness"] == gen.HARNESS_VERSION
    manifests = sorted((isolated_dirs / "outputs" / "manifests").glob("generate-*.json"))
    assert len(manifests) == 1
    doc = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert doc["n_written"] == len(TEXTS) and doc["prompts_per_sec"] > 0 and doc["tokens_per_sec"] > 0
    assert doc["gen_cfg"]["dtype"] == "fp32" and doc["gen_cfg"]["max_new_tokens"] == CAP
    assert {f: doc[f] for f in provenance} == {f: r.meta[f] for f in provenance}     # the run's own fields


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
    # A --limit run in one dtype, then a run on other prompts in another dtype (or another configuration),
    # into the same file.
    path = tmp_path / "g.jsonl"
    run(stock, path, ps=prompts()[:2])
    with pytest.raises(gen.OtherPrecisionError, match="made in fp32"):
        run(dataclasses.replace(stock, dtype="fp16"), path, ps=prompts()[2:])
    with pytest.raises(gen.StaleOutputError, match="another configuration"):
        run(stock, path, ps=prompts()[2:], max_new_tokens=CAP - 1)
    assert all(r.meta["run_config"] == gen.read_generations(path)[0].meta["run_config"]
               for r in gen.read_generations(path))


@pytest.mark.parametrize("finished", [True, False])
def test_a_cell_made_in_another_precision_is_never_started_over(stock, finished):
    # W3: a cell made on the A100 is met by the T4 backup's run in fp16 (here the fixture's fp32 file and an
    # fp16 run). It is not stale: a restart would discard the other precision's outputs, and the next push
    # would replace the runs repo's copy. So the refusal names both precisions and the two remedies (the
    # file's --dtype, or leaving the cell out), and carries no restart line.
    path = cell_path()
    run(stock, path, ps=prompts() if finished else prompts()[:2])
    before = path.read_bytes()
    with pytest.raises(gen.OtherPrecisionError) as exc:
        run(dataclasses.replace(stock, dtype="fp16"), path)
    message = str(exc.value)
    assert isinstance(exc.value, gen.StaleOutputError)          # commands skip it like a stale cell
    assert exc.value.command is None and exc.value.dtype == "fp32"
    assert "made in fp32" in message and "this run is fp16" in message
    assert "--dtype fp32" in message and "--sets/--langs" in message
    assert "lift restart" not in message
    assert path.read_bytes() == before


def moved_stack(monkeypatch, **changes):
    """The next session's runtime: a newer transformers or torch, or another GPU class."""
    real = gen.stack
    monkeypatch.setattr(gen, "stack", lambda device: {**real(device), **changes})


# The cell every prompts() record belongs to, and the job line that starts it over. The stock fixture runs
# in fp32, not generation.dtype (bf16), so its line names its precision: lift restart refuses a cell made
# in another precision than its --dtype.
RESTART = "lift restart --variant stock --split dev --sets harmful --langs en --dtype fp32"


def cell_path():
    return gen.generations_path("stock", "harmful", "en", "dev")


@pytest.mark.parametrize("change", [{"transformers": "5.99.0"}, {"torch": "2.99.0+cu130"},
                                    {"gpu_capability": "8.9"}, {"transformers": "5.99.0", "torch": "2.99.0"}])
def test_appending_under_another_stack_is_refused(stock, monkeypatch, change):
    # A job line can follow the refusal: moving the file aside by hand is undone by the job's own
    # `lift sync pull`. constraints.txt pins transformers only; torch is each runtime's own build.
    path = cell_path()
    run(stock, path, ps=prompts()[:2])
    before = path.read_bytes()
    moved_stack(monkeypatch, **change)
    with pytest.raises(gen.StaleOutputError) as exc:
        run(stock, path)
    message = str(exc.value)
    for field, value in change.items():
        assert f"{field} " in message and f"-> {value}" in message           # names what moved
    assert f"run `{RESTART}` as a job line right after `lift sync pull`, followed by the line that failed" \
        in message
    assert ("constraints.txt" in message) == ("transformers" in change)
    runtime = bool({"torch", "gpu_capability"} & set(change))
    assert ("torch is the runtime's own CUDA build" in message) == runtime
    assert not re.search(r"\bmove\b", message.replace(str(path), ""))      # by hand, the pull would undo it
    assert exc.value.command == RESTART and path.read_bytes() == before


def test_changed_inputs_name_the_restart_line(stock, tmp_path):
    path = cell_path()
    run(stock, path)
    fixed = prompts()
    fixed[0] = make_prompt(fixed[0].pid, text=fixed[0].text + " (fixed translation)")
    with pytest.raises(gen.StaleOutputError, match="a fixed prompt text") as exc:
        run(stock, path, ps=fixed)
    assert f"run `{RESTART}` as a job line right after `lift sync pull`" in str(exc.value)
    assert exc.value.command == RESTART
    with pytest.raises(gen.StaleOutputError, match="another configuration") as exc:
        run(stock, path, max_new_tokens=CAP - 1)               # other decoding settings: other inputs
    assert f"run `{RESTART}` as a job line right after `lift sync pull`" in str(exc.value)
    assert exc.value.command == RESTART
    other = tmp_path / "elsewhere.jsonl"                    # not a cell's file: no lift restart line for it
    run(stock, other)
    with pytest.raises(gen.StaleOutputError, match=f"move {re.escape(str(other))} aside") as exc:
        run(stock, other, ps=fixed)
    assert "`lift restart" not in str(exc.value) and exc.value.command is None


def test_a_refusal_says_whether_the_cell_made_again_keeps_its_keys(stock, monkeypatch):
    # X9: a key list drawn from the cell (P04's second-judge list) still holds after a restart only when the
    # cell is made again under the same keys: only the stack moved, which the key leaves out. A fixed prompt
    # text or another configuration makes new keys, so the list is drawn again before a --keys line.
    path = cell_path()
    run(stock, path, ps=prompts()[:2])
    fixed = prompts()
    fixed[0] = make_prompt(fixed[0].pid, text=fixed[0].text + " (fixed translation)")
    with pytest.raises(gen.StaleOutputError, match="a fixed prompt text") as exc:
        run(stock, path, ps=fixed)
    assert exc.value.keys_changed is True
    with pytest.raises(gen.StaleOutputError, match="another configuration") as exc:
        run(stock, path, max_new_tokens=CAP - 1)
    assert exc.value.keys_changed is True
    moved_stack(monkeypatch, torch="2.99.0+cu130")
    with pytest.raises(gen.StaleOutputError, match="another stack") as exc:
        run(stock, path)
    assert exc.value.keys_changed is False and exc.value.command == RESTART
    assert gen.StaleOutputError("a cause no one marked").keys_changed is True     # the list is drawn again


def test_restart_command_names_the_cell_the_judge_and_a_precision_other_than_the_default():
    assert gen.restart_command("stock", "easy_benign", "hi", "final") == (
        "lift restart --variant stock --split final --sets easy_benign --langs hi")
    assert gen.restart_command("stock", "harmful", "hi", "dev", judge="second").endswith(" --judge second")
    plain = gen.restart_command("stock", "harmful", "hi", "dev")
    assert gen.restart_command("stock", "harmful", "hi", "dev", dtype="bf16") == plain    # generation.dtype
    assert gen.restart_command("stock", "harmful", "hi", "dev", dtype="fp16") == plain + " --dtype fp16"
    assert gen.restart_command("stock", "harmful", "hi", "dev", judge="primary", dtype="fp16") == (
        plain + " --judge primary --dtype fp16")


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_the_restart_line_of_a_stale_cell_names_the_runs_precision(stock, monkeypatch, dtype):
    # A T4's fp16 cell cut short, met on another stack: its line carries --dtype fp16, so the restart it
    # names is not refused as another precision. A bf16 cell (generation.dtype) gets the plain line.
    variant = dataclasses.replace(stock, dtype=dtype)
    run(variant, cell_path(), ps=prompts()[:2])
    moved_stack(monkeypatch, torch="2.99.0+cu130")
    with pytest.raises(gen.StaleOutputError) as exc:
        run(variant, cell_path())
    line = "lift restart --variant stock --split dev --sets harmful --langs en"
    assert exc.value.command == line + (" --dtype fp16" if dtype == "fp16" else "")
    assert f"run `{exc.value.command}` as a job line" in str(exc.value)


def test_a_finished_file_is_reused_under_another_stack(stock, tmp_path, monkeypatch, caplog):
    # A session died while judging; the next one resolved a newer transformers. Nothing would be
    # appended, so the finished file is returned as it is (the judge goes on) and the move is logged.
    path = tmp_path / "g.jsonl"
    run(stock, path)
    before = path.read_bytes()
    moved_stack(monkeypatch, transformers="5.99.0")
    with caplog.at_level(logging.INFO, logger=gen.__name__):
        assert gen.generate(stock, prompts(), out_path=path, max_new_tokens=CAP) == path
    assert path.read_bytes() == before
    assert "another stack" in caplog.text and "-> 5.99.0" in caplog.text


def test_run_config_is_the_inputs_and_the_stack_and_decoding_have_their_own_digests(stock, tmp_path,
                                                                                   monkeypatch):
    from lift.io import sha256_obj

    first = run(stock, tmp_path / "a.jsonl", ps=prompts()[:1])["p-000"].meta
    moved_stack(monkeypatch, gpu_capability="8.9")
    moved = run(stock, tmp_path / "b.jsonl", ps=prompts()[:1])["p-000"].meta
    fp16 = run(dataclasses.replace(stock, dtype="fp16"), tmp_path / "c.jsonl", ps=prompts()[:1])["p-000"].meta
    assert first["run_config"] == moved["run_config"] != fp16["run_config"]
    assert first["stack"] != moved["stack"] == fp16["stack"]
    for meta in (first, moved):
        fields = {f: meta[f] for f in ("torch", "transformers", "gpu_capability")}       # exactly these
        assert meta["stack"] == sha256_obj(fields)[:16]
    cfg = gen.gen_config(stock, max_new_tokens=CAP)
    decoding = sha256_obj({k: v for k, v in cfg.items() if k != "dtype"})[:16]      # every setting but dtype
    assert first["decoding"] == fp16["decoding"] == decoding


def test_changed_inputs_are_refused_under_any_stack(stock, tmp_path, monkeypatch):
    path = tmp_path / "g.jsonl"
    run(stock, path)
    moved_stack(monkeypatch, transformers="5.99.0")
    with pytest.raises(gen.OtherPrecisionError, match="made in fp32"):
        run(dataclasses.replace(stock, dtype="fp16"), path)
    with pytest.raises(gen.StaleOutputError, match="another configuration"):
        run(stock, path, ps=prompts()[:2], max_new_tokens=CAP - 1)


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
