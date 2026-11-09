"""lift.models.loglik, ``lift belebele`` and ``lift heldout-loss`` (P03).

Covered: the frozen Belebele format and its letter scoring (BOS once, the letter boundary, the next-token
and sequence paths, a forced answer through a stub model), the held-out NLL over exactly the tokens
training supervises, batched against one-at-a-time scoring, the resumable cache-keyed score files (resume,
a crash, out of memory, stale cells and the --restart remedy, a stack refusal; each precision's own files,
named -{dtype} outside generation.dtype, and the guard for a file holding another precision's scores,
which is never started over), wrappers (score_logits and its gate), the metrics files (per language and per
precision; each precision's vs_stock pairs with stock's file of that precision), and both commands end to
end on the tiny model.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import sys
from types import SimpleNamespace

import pytest

from lift import splits
from lift.cli import main
from lift.io import read_jsonl, sha256_file, write_jsonl
from lift.models import loglik
from lift.models.generate import StaleOutputError
from lift.paths import manifests_dir, metrics_dir, output_dir, prompts_dir, scores_dir
from lift.schemas import PromptRecord, VariantSpec

WORDS = ("river", "hill", "garden", "market", "school", "bridge")


def bel_row(i: int, *, lang: str = "en", split: str = "dev", answer: int = 1,
            passage: str | None = None) -> dict:
    """One Belebele item as lift.data.build.load_prompts returns it; passages differ in length by item."""
    words = " ".join(WORDS[: 1 + i % len(WORDS)])
    return {"pid": f"bel-{i:03d}", "lang": lang, "split": split, "link": f"https://example.org/p/{i // 2}",
            "question_number": 1 + i % 2, "flores_passage": passage or f"Passage {i} is about the {words}.",
            "question": f"What is passage {i} about?", "mc_answer1": f"The {WORDS[i % len(WORDS)]}",
            "mc_answer2": "A bridge", "mc_answer3": "Nothing at all", "mc_answer4": f"Item {i}",
            "correct_answer_num": answer, "dialect": "eng_Latn"}


def bel_rows(n: int = 5, **kw) -> list[dict]:
    return [bel_row(i, answer=1 + i % 4, **kw) for i in range(n)]


@pytest.fixture(autouse=True)
def own_splits(tmp_path, monkeypatch):
    """The committed split files these tests' prompt files are stamped from (R1's split-stamp check reads
    lift.splits.SPLITS_DIR): a folder of the test's own, which write_belebele fills."""
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path / "splits")


def write_belebele(lang: str, rows: list[dict]) -> None:
    """A real data/prompts/belebele_{lang}.jsonl in the test's LIFT_DATA_DIR (fake_prompt_files holds
    PromptRecords only), and its pids in the test's committed Belebele split, each with the split its row
    carries (a pid keeps one split in every language, as lift data build freezes it)."""
    write_jsonl(prompts_dir() / f"belebele_{lang}.jsonl", rows)
    try:
        frozen = splits.load_split("belebele")
    except FileNotFoundError:
        frozen = {}
    frozen.update({r["pid"]: r["split"] for r in rows})
    sizes = {s: sum(v == s for v in frozen.values()) for s in set(frozen.values())}
    splits.save_split("belebele", frozen, seed=0, sizes=sizes, overwrite=True)


def utility(i: int, source: str = "aya", *, split: str = "dev", reference: str | None = None) -> PromptRecord:
    """A utility record with its reference answer, as lift.data.sft builds it."""
    answer = reference if reference is not None else f"उत्तर {i}: " + "फूल " * (1 + i % 4)
    return PromptRecord(pid=f"utl-{source}-{i:05d}", lang="hi", set="utility", split=split,
                        text=f"प्रश्न {i}: बगीचे के बारे में बताइए।",
                        source="aya-dataset" if source == "aya" else "indicalign-dolly_t",
                        extra={"reference": answer})


def plain(model, tok, *, name: str = "stock", dtype: str = "fp32"):
    """A LoadedVariant around ``model`` (a stub or a tiny model) and ``tok``, without a registry entry."""
    from lift.models.load import template_hash
    from lift.models.variants import LoadedVariant

    stock = name == "stock"
    spec = VariantSpec(name=name, method="stock") if stock else VariantSpec(name=name, method="adapter",
                                                                            adapter=name)
    return LoadedVariant(model=model, tokenizer=tok, spec=spec, model_rev="local-" + "1" * 40,
                         adapter_hash=None if stock else "a" * 64, template_hash=template_hash(tok),
                         dtype=dtype, device="cpu")


def letter_tokenizer():
    """The tests' byte-level tokenizer (tests/fakes.py) plus four merges that make " A" .. " D" one token
    each, as Tiny Aya's vocabulary does: Belebele's next-token path."""
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from fakes import SPECIAL_TOKENS, TINY_CHAT_TEMPLATE
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[ch] = len(vocab)
    merges = [("Ġ", x) for x in loglik.LETTERS]           # "Ġ" is the byte-level space
    for a, b in merges:
        vocab[a + b] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<UNK>", fuse_unk=False))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL_TOKENS])
    backend.post_processor = processors.TemplateProcessing(
        single="<BOS_TOKEN> $A:0", pair="<BOS_TOKEN> $A:0 <BOS_TOKEN> $B:1",
        special_tokens=[("<BOS_TOKEN>", vocab["<BOS_TOKEN>"])])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<BOS_TOKEN>", eos_token="<|END_OF_TURN_TOKEN|>",
        unk_token="<UNK>", pad_token="<PAD>", padding_side="left",
        model_input_names=["input_ids", "attention_mask"], model_max_length=1024,
        chat_template=TINY_CHAT_TEMPLATE, clean_up_tokenization_spaces=False)


class Wrapped:
    """A tokenizer whose attributes are another's unless overridden (e.g. ``bos_token_id=None``)."""

    def __init__(self, tok, **overrides):
        self._tok = tok
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, text, **kw):
        return self._tok(text, **kw)


class Merging(Wrapped):
    """Tokenizes ': C' at the end of a text as one token, as a BPE merge across the boundary would."""

    def __call__(self, text, **kw):
        out = self._tok(text, **kw)
        if text.endswith(": C"):
            return {"input_ids": [*out["input_ids"][:-3], 999]}
        return out


class Favours:
    """A stand-in model whose logits favour one token at every position (a forced answer). It checks what
    it is given: left padding, position ids counted from each row's first real token, no cache."""

    def __init__(self, vocab: int, token: int, *, oom_above: int | None = None):
        self.config = SimpleNamespace(max_position_embeddings=1024, vocab_size=vocab)
        self.token, self.oom_above = token, oom_above
        self.calls: list[tuple[int, int]] = []

    def __call__(self, *, input_ids, attention_mask, position_ids, logits_to_keep, use_cache):
        import torch

        assert use_cache is False
        assert torch.equal(position_ids, (attention_mask.cumsum(-1) - 1).clamp(min=0))
        assert bool((attention_mask[:, -1] == 1).all())         # left padding: every row ends on a real token
        if self.oom_above is not None and input_ids.shape[0] > self.oom_above:
            raise RuntimeError("CUDA out of memory (simulated)")
        self.calls.append((int(input_ids.shape[0]), int(logits_to_keep)))
        logits = torch.zeros(input_ids.shape[0], logits_to_keep, self.config.vocab_size)
        logits[..., self.token] = 5.0
        return SimpleNamespace(logits=logits)


class Gate:
    """A stand-in CAST wrapper: score_logits runs the wrapped model unsteered and records each call, and
    the gate "fires" on prompts of an even number of tokens, decided per row on that row's prompt."""

    def __init__(self, model):
        self.model = model
        self.calls: list[dict] = []
        self.last: list[dict] = []

    def score_logits(self, input_ids, attention_mask, *, position_ids, prompt_lengths, logits_to_keep):
        self.calls.append({"rows": int(input_ids.shape[0]), "prompt_lengths": list(prompt_lengths),
                           "keep": logits_to_keep})
        self.last = [{"gate_fired": n % 2 == 0} for n in prompt_lengths]
        return self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                          logits_to_keep=logits_to_keep, use_cache=False).logits

    def batch_meta(self):
        return list(self.last)


@pytest.fixture(scope="module")
def stock(tiny_model_dir):
    """Stock on the tiny model in fp32: " A" is two byte tokens there, so Belebele takes the sequence path."""
    pytest.importorskip("torch")
    from lift.models.variants import build_variant

    return build_variant(VariantSpec(name="stock", method="stock"), device="cpu", dtype="fp32",
                         model_id=str(tiny_model_dir))


@pytest.fixture(scope="module")
def letters_variant():
    """A tiny model whose tokenizer makes each letter one token: the next-token path."""
    pytest.importorskip("torch")
    from fakes import build_tiny_cohere2

    tok = letter_tokenizer()
    return plain(build_tiny_cohere2(tok), tok)


def pin():
    from lift import config

    config.load("project")["models"]["global"]["revision"] = "b" * 40


def tiny_build(monkeypatch, tiny_model_dir):
    """Every command builds its variant on the tiny model, on the CPU."""
    from lift.models import variants

    real = variants.build_variant
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: real(
        spec, device="cpu", dtype=dtype, model_id=str(tiny_model_dir)))


def manifests(stage: str) -> list[dict]:
    paths = sorted(manifests_dir().glob(f"{stage}-*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


# --- the frozen format and the letters ------------------------------------------------------------------


def test_the_belebele_format_is_lm_evals_with_only_the_question_stripped():
    # lm-evaluation-harness's _default_template_yaml: "Q: {{question.strip()}}", options as published.
    row = {"pid": "bel-001", "flores_passage": "p ", "question": " q ", "mc_answer1": "a ", "mc_answer2": "b",
           "mc_answer3": "c", "mc_answer4": "d"}
    assert loglik.belebele_prompt(row) == "P: p \nQ: q\nA: a \nB: b\nC: c\nD: d\nAnswer:"
    assert loglik.FORMAT_VERSION == 1
    with pytest.raises(ValueError, match="mc_answer2"):
        loglik.belebele_prompt({**row, "mc_answer2": None})


@pytest.mark.parametrize(("value", "letter"), [(1, "A"), (2, "B"), (3, "C"), (4, "D"), ("3", "C")])
def test_the_gold_letter_is_the_answer_number_mapped_to_a_letter(value, letter):
    assert loglik.gold_letter({"pid": "bel-000", "correct_answer_num": value}) == letter


@pytest.mark.parametrize("value", [0, 5, None, "x", 2.5])
def test_any_other_answer_number_is_refused(value):
    with pytest.raises(ValueError, match="bel-000: correct_answer_num"):
        loglik.gold_letter({"pid": "bel-000", "correct_answer_num": value})


def test_rows_that_are_not_one_cells_items_are_refused():
    rows = bel_rows(3)
    loglik.check_belebele_rows(rows, lang="en", split="dev")
    with pytest.raises(ValueError, match="no Belebele rows"):
        loglik.check_belebele_rows([])
    with pytest.raises(ValueError, match="repeated pids"):
        loglik.check_belebele_rows([*rows, rows[0]])
    with pytest.raises(ValueError, match="not belebele_hi_dev items"):
        loglik.check_belebele_rows(rows, lang="hi", split="dev")
    with pytest.raises(ValueError, match="not belebele_en_final items"):
        loglik.check_belebele_rows(rows, lang="en", split="final")
    with pytest.raises(ValueError, match="correct_answer_num"):
        loglik.check_belebele_rows([{**rows[0], "correct_answer_num": 7}])


@pytest.mark.torch
def test_the_context_holds_the_bos_exactly_once(tiny_tokenizer):
    prompt = loglik.belebele_prompt(bel_row(3))
    ids = loglik.context_ids(tiny_tokenizer, prompt)
    bos = tiny_tokenizer.bos_token_id
    assert ids == [bos, *tiny_tokenizer(prompt, add_special_tokens=False)["input_ids"]]
    assert ids.count(bos) == 1
    assert tiny_tokenizer(prompt)["input_ids"][0] == bos     # a plain call adds one itself: never used
    assert loglik.context_ids(Wrapped(tiny_tokenizer, bos_token_id=None), prompt) == ids[1:]
    with pytest.raises(ValueError, match="control tokens"):
        loglik.context_ids(tiny_tokenizer, prompt + "<BOS_TOKEN>")


@pytest.mark.torch
def test_a_tokenizer_that_merges_across_the_letter_boundary_is_refused(tiny_tokenizer):
    prompt = loglik.belebele_prompt(bel_row(0))
    ids = loglik.prompt_ids(tiny_tokenizer, prompt)
    loglik.check_letter_boundary(tiny_tokenizer, prompt, ids, loglik.letter_ids(tiny_tokenizer))
    merging = Merging(tiny_tokenizer)
    with pytest.raises(ValueError, match="' C' together"):
        loglik.check_letter_boundary(merging, prompt, ids, loglik.letter_ids(merging))
    with pytest.raises(ValueError, match="belebele bel-000"):       # before any forward pass
        loglik.score_belebele(plain(None, merging), [bel_row(0)])


# --- letter scoring ---------------------------------------------------------------------------------------


@pytest.mark.torch
def test_a_forced_answer_wins_with_single_token_letters():
    tok = letter_tokenizer()
    letters = loglik.letter_ids(tok)
    assert letters == {x: [tok.convert_tokens_to_ids("Ġ" + x)] for x in loglik.LETTERS}
    assert loglik.scoring_path(letters) == loglik.NEXT_TOKEN
    model = Favours(len(tok), letters["C"][0])
    rows = bel_rows(6)
    items = loglik.score_belebele(plain(model, tok), rows, batch_size=4)
    assert [i["pid"] for i in items] == [r["pid"] for r in rows]                 # the rows' order
    assert {i["pred"] for i in items} == {"C"} and not any(i["tie"] for i in items)
    assert [i["gold"] for i in items] == [loglik.gold_letter(r) for r in rows]
    assert [i["correct"] for i in items] == [i["gold"] == "C" for i in items]
    assert all(i["n_letter_tokens"] == dict.fromkeys(loglik.LETTERS, 1) and i["meta"] == {} for i in items)
    assert model.calls == [(4, 1), (2, 1)]            # one row per item, only the last position's logits


@pytest.mark.torch
def test_a_forced_answer_wins_with_multi_token_letters(tiny_tokenizer):
    letters = loglik.letter_ids(tiny_tokenizer)
    assert all(len(ids) == 2 for ids in letters.values())                        # byte-level: "Ġ", "C"
    assert loglik.scoring_path(letters) == loglik.SEQUENCE
    model = Favours(len(tiny_tokenizer), letters["C"][-1])
    items = loglik.score_belebele(plain(model, tiny_tokenizer), bel_rows(3), batch_size=2)
    assert {i["pred"] for i in items} == {"C"} and not any(i["tie"] for i in items)
    assert all(i["n_letter_tokens"] == dict.fromkeys(loglik.LETTERS, 2) for i in items)
    assert model.calls == [(8, 3), (4, 3)]            # four rows per item; each letter's two positions + 1


@pytest.mark.torch
def test_a_tie_goes_to_the_first_letter_and_is_recorded(tiny_tokenizer):
    model = Favours(len(tiny_tokenizer), 0)          # the pad token: every letter scores the same
    item = loglik.score_belebele(plain(model, tiny_tokenizer), [bel_row(1)])[0]
    assert item["pred"] == "A" and item["tie"]


@pytest.mark.torch
@pytest.mark.parametrize("path", [loglik.SEQUENCE, loglik.NEXT_TOKEN])
def test_batched_scores_equal_one_at_a_time(path, stock, letters_variant):
    variant = stock if path == loglik.SEQUENCE else letters_variant
    assert loglik.scoring_path(loglik.letter_ids(variant.tokenizer)) == path
    rows = bel_rows(6)
    one = loglik.score_belebele(variant, rows, batch_size=1)
    many = loglik.score_belebele(variant, rows, batch_size=4)
    assert len({i["n_context_tokens"] for i in one}) > 1                         # padding is exercised
    assert len({json.dumps(i["logprobs"], sort_keys=True) for i in one}) == len(rows)   # rows really differ
    for a, b in zip(one, many, strict=True):
        assert a["pid"] == b["pid"] and a["pred"] == b["pred"]
        assert all(abs(a["logprobs"][x] - b["logprobs"][x]) < 1e-4 for x in loglik.LETTERS)


@pytest.mark.torch
def test_the_next_token_and_sequence_paths_agree_on_single_token_letters(letters_variant):
    cfg = loglik.belebele_config(letters_variant)
    rows = bel_rows(4)
    got = {}
    for path in (loglik.NEXT_TOKEN, loglik.SEQUENCE):
        units = loglik._question_units(letters_variant, rows, cfg, path)
        got[path] = loglik._score_all(letters_variant, units, loglik._belebele_scorer(cfg, path), 2)
    for (a, _), (b, _) in zip(got[loglik.NEXT_TOKEN], got[loglik.SEQUENCE], strict=True):
        assert a["pred"] == b["pred"]
        assert all(abs(a["logprobs"][x] - b["logprobs"][x]) < 1e-5 for x in loglik.LETTERS)


# --- held-out NLL -----------------------------------------------------------------------------------------


@pytest.mark.torch
def test_heldout_nll_is_the_cross_entropy_of_exactly_the_answer_tokens(stock):
    import torch

    from lift.models.load import encode_chat_turn

    prompts = [utility(1), utility(2, "ind")]
    items = loglik.heldout_nll(stock, prompts, batch_size=2)
    for p, item in zip(prompts, items, strict=True):
        ids, n_prompt = encode_chat_turn(stock.tokenizer, p.text, p.extra["reference"])
        with torch.no_grad():
            logits = stock.model(torch.tensor([ids])).logits[0]
        ce = torch.nn.functional.cross_entropy(logits[n_prompt - 1:-1].float(), torch.tensor(ids[n_prompt:]),
                                               reduction="sum").item()
        assert (item["pid"], item["source"]) == (p.pid, p.pid.split("-")[1])
        assert item["n_tokens"] == len(ids) - n_prompt and item["n_prompt_tokens"] == n_prompt
        assert abs(item["nll_sum"] - ce) < 1e-4
        assert item["mean_nll"] == pytest.approx(item["nll_sum"] / item["n_tokens"])


@pytest.mark.torch
def test_heldout_batched_equals_one_at_a_time(stock):
    prompts = [utility(i, "aya" if i % 2 else "ind") for i in range(5)]
    one = loglik.heldout_nll(stock, prompts, batch_size=1)
    many = loglik.heldout_nll(stock, prompts, batch_size=5)
    assert len({i["n_tokens"] for i in one}) > 1                                 # padding is exercised
    for a, b in zip(one, many, strict=True):
        assert a["n_tokens"] == b["n_tokens"] and abs(a["nll_sum"] - b["nll_sum"]) < 1e-4


@pytest.mark.torch
def test_nothing_is_truncated(stock):
    with pytest.raises(ValueError, match="nothing is truncated"):          # far past the 1,024 positions
        loglik.heldout_nll(stock, [utility(1, reference="फूल " * 400)])
    with pytest.raises(ValueError, match="nothing is truncated"):
        loglik.score_belebele(stock, [bel_row(0, passage="word " * 300)])


def test_a_record_without_a_reference_is_refused():
    no_reference = PromptRecord(pid="utl-aya-00003", lang="hi", set="utility", split="dev", text="t",
                                source="aya-dataset")
    with pytest.raises(ValueError, match=r"utl-aya-00003.*held-out loss scores every record's reference"):
        loglik.check_references([utility(1), no_reference])
    with pytest.raises(ValueError, match="no reference answer"):
        loglik.check_references([utility(2, reference="  ")])
    with pytest.raises(ValueError, match="not a utility pid"):
        loglik.check_references([utility(1).model_copy(update={"pid": "x-1"})])
    with pytest.raises(ValueError, match="not utility/hi"):
        loglik.check_references([utility(1).model_copy(update={"set": "harmful"})])


# --- wrappers ---------------------------------------------------------------------------------------------


@pytest.mark.torch
def test_a_wrapper_without_score_logits_is_refused(stock):
    wrapper = dataclasses.replace(stock, model=object(), base=stock.model)
    for call in (lambda: loglik.score_belebele(wrapper, bel_rows(1)),
                 lambda: loglik.heldout_nll(wrapper, [utility(1)]),
                 lambda: loglik.belebele_cell(wrapper, bel_rows(1), lang="en", split="dev")):
        with pytest.raises(NotImplementedError, match="P05"):
            call()
    assert not scores_dir().exists()                                     # no score file, in any precision


@pytest.mark.torch
def test_score_logits_is_used_when_set_and_its_gate_is_recorded(stock):
    gate = Gate(stock.model)
    wrapper = dataclasses.replace(stock, model=object(), base=stock.model, score_logits=gate.score_logits,
                                  batch_meta=gate.batch_meta)
    rows = bel_rows(4)
    items = loglik.score_belebele(wrapper, rows, batch_size=2)
    reference = loglik.score_belebele(stock, rows, batch_size=2)
    assert gate.calls and all(c["keep"] == 3 and c["rows"] == 8 for c in gate.calls)     # the sequence path
    # The gate reads each row's prompt: the item's context, the same for its four rows.
    assert sorted(n for c in gate.calls for n in c["prompt_lengths"]) == sorted(
        i["n_context_tokens"] for i in reference for _ in loglik.LETTERS)
    for a, b in zip(items, reference, strict=True):
        assert a["logprobs"] == pytest.approx(b["logprobs"], abs=1e-6) and b["meta"] == {}
        assert a["meta"] == {"gate_fired": a["n_context_tokens"] % 2 == 0}

    gate.calls.clear()                                     # held-out: the gate reads the chat prompt
    prompts = [utility(1), utility(2, "ind")]
    nll = loglik.heldout_nll(wrapper, prompts, batch_size=2)
    assert len(gate.calls) == 1
    assert sorted(gate.calls[0]["prompt_lengths"]) == sorted(i["n_prompt_tokens"] for i in nll)
    assert [i["meta"] for i in nll] == [{"gate_fired": i["n_prompt_tokens"] % 2 == 0} for i in nll]

    write_belebele("en", rows)                             # the file records it, the metrics report the share
    path = loglik.belebele_cell(wrapper, rows, lang="en", split="dev")
    recorded = {r["pid"]: r["meta"]["gate_fired"] for r in read_jsonl(path)}
    assert recorded == {i["pid"]: i["meta"]["gate_fired"] for i in items}
    doc = loglik.belebele_metrics("stock", "dev", dtype="fp32")
    assert doc["langs"]["en"]["gate_fired_rate"] == pytest.approx(sum(recorded.values()) / len(recorded))


@pytest.mark.torch
def test_a_wrapper_meta_that_reuses_a_record_field_is_refused(stock):
    gate = Gate(stock.model)
    wrapper = dataclasses.replace(stock, model=object(), base=stock.model, score_logits=gate.score_logits,
                                  batch_meta=lambda: [{"dtype": "bf16"} for _ in gate.last])
    with pytest.raises(ValueError, match=r"batch_meta\(\) sets \['dtype'\]"):
        loglik.score_belebele(wrapper, bel_rows(1))


# --- score files ------------------------------------------------------------------------------------------


@pytest.mark.torch
def test_a_cell_resumes_and_skips_done_keys(stock):
    from lift.cachekey import generation_key, intervention_digest

    rows = bel_rows(5)
    path = loglik.belebele_cell(stock, rows[:3], lang="en", split="dev", batch_size=2)
    assert path == loglik.belebele_scores_path("stock", "en", "dev", dtype="fp32")     # the variant's dtype
    first = path.read_bytes()
    assert len(first.splitlines()) == 3
    loglik.belebele_cell(stock, rows, lang="en", split="dev", batch_size=2)
    after = path.read_bytes()
    assert after.startswith(first) and len(after.splitlines()) == 5               # appended, never rewritten
    loglik.belebele_cell(stock, rows, lang="en", split="dev")
    assert path.read_bytes() == after                                              # nothing left to do
    records = {r["pid"]: r for r in read_jsonl(path)}
    cfg = {"task": "belebele", "version": 1, "dtype": "fp32", "bos": stock.tokenizer.bos_token_id,
           "letters": loglik.letter_ids(stock.tokenizer)}
    for row in rows:                                         # the key: generation_key over the task config
        assert records[row["pid"]]["key"] == generation_key(
            model_rev=stock.model_rev, adapter_hash=None, spec=stock.spec, template_hash=stock.template_hash,
            gen_cfg=cfg, pid=row["pid"], lang="en", prompt_text=loglik.belebele_prompt(row))
    meta = records["bel-000"]["meta"]
    assert (meta["dtype"], meta["scoring"], meta["task_version"]) == ("fp32", "sequence", 1)
    assert meta["run_config"] == loglik.belebele_run_config(stock) and meta["model_rev"] == stock.model_rev
    assert meta["intervention"] == intervention_digest(stock.spec)         # lift select checks the entry
    assert all(meta[k] for k in ("stack", "template", "task", "torch", "transformers", "gpu_capability"))
    docs = manifests("belebele")
    assert sorted((d["n_resumed"], d["n_written"]) for d in docs) == [(0, 3), (3, 2), (5, 0)]
    assert all(d["status"] == "ok" and d["devices"] == 1 and d["lang"] == "en" for d in docs)


@pytest.mark.torch
def test_a_crash_mid_cell_resumes_without_duplicates(stock, monkeypatch):
    rows = bel_rows(4)
    real, calls = loglik._forward, {"n": 0}

    def dies_on_the_second_batch(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("session killed")
        return real(*a, **kw)

    monkeypatch.setattr(loglik, "_forward", dies_on_the_second_batch)
    with pytest.raises(RuntimeError, match="session killed"):
        loglik.belebele_cell(stock, rows, lang="en", split="dev", batch_size=2)
    path = loglik.belebele_scores_path("stock", "en", "dev", dtype="fp32")
    assert len(path.read_bytes().splitlines()) == 2
    with path.open("ab") as f:                                       # and a kill mid-line on top of that
        f.write(b'{"key": "trunc')
    assert manifests("belebele")[-1]["status"] == "failed"
    monkeypatch.setattr(loglik, "_forward", real)
    loglik.belebele_cell(stock, rows, lang="en", split="dev", batch_size=2)
    assert sorted(r["pid"] for r in read_jsonl(path)) == [r["pid"] for r in rows]


@pytest.mark.torch
def test_an_out_of_memory_error_halves_the_batch_and_skips_nothing(tiny_tokenizer):
    model = Favours(len(tiny_tokenizer), 0, oom_above=4)            # more than one item of four rows: OOM
    path = loglik.belebele_cell(plain(model, tiny_tokenizer), bel_rows(5), lang="en", split="dev",
                                batch_size=4)
    assert len(path.read_bytes().splitlines()) == 5 and {rows for rows, _keep in model.calls} == {4}
    assert manifests("belebele")[-1]["oom_retries"] >= 1


@pytest.mark.torch
def test_a_changed_text_or_answer_is_stale_and_its_restart_line_starts_the_cell_over(stock):
    rows = bel_rows(3)
    path = loglik.belebele_cell(stock, rows, lang="en", split="dev")
    before = path.read_bytes()
    fixed = [rows[0], {**rows[1], "flores_passage": rows[1]["flores_passage"] + " (fixed)"}, rows[2]]
    line = "lift belebele --variant stock --split dev --langs en --restart --dtype fp32"
    with pytest.raises(StaleOutputError, match="bel-001") as exc:
        loglik.belebele_cell(stock, fixed, lang="en", split="dev")
    assert exc.value.command == line and line in str(exc.value) and exc.value.keys_changed
    answer = {**rows[2], "correct_answer_num": 1 + rows[2]["correct_answer_num"] % 4}
    with pytest.raises(StaleOutputError, match="bel-002"):           # the same text, another gold answer
        loglik.belebele_cell(stock, [*rows[:2], answer], lang="en", split="dev")
    assert path.read_bytes() == before                               # nothing was appended
    done = loglik.start_over(path)
    assert path.read_bytes() == b"" and done.records == 3 and done.copy.read_bytes() == before
    assert done.copy.is_relative_to(output_dir() / ".aside")
    loglik.belebele_cell(stock, fixed, lang="en", split="dev")
    assert len(path.read_bytes().splitlines()) == 3


def test_a_score_file_in_another_precision_than_generation_dtype_is_named_for_it():
    """V05: as metrics_path, a precision other than generation.dtype (bf16) adds "-{dtype}" to a score
    file's name, so a T4's fp16 cell and Colab's bf16 cell of one variant are two files."""
    bf16 = scores_dir() / "belebele" / "stock" / "belebele_hi_dev.jsonl"
    assert loglik.belebele_scores_path("stock", "hi", "dev") == bf16
    assert loglik.belebele_scores_path("stock", "hi", "dev", dtype="bf16") == bf16
    assert loglik.belebele_scores_path("stock", "hi", "dev", dtype="fp16") == bf16.with_name(
        "belebele_hi_dev-fp16.jsonl")
    held = scores_dir() / "heldout_loss" / "adapter-x" / "utility_hi_final.jsonl"
    assert loglik.heldout_scores_path("adapter-x", "final") == held
    assert loglik.heldout_scores_path("adapter-x", "final", dtype="bf16") == held
    assert loglik.heldout_scores_path("adapter-x", "final", dtype="fp32") == held.with_name(
        "utility_hi_final-fp32.jsonl")
    assert loglik.metrics_path("belebele", "stock", "dev", dtype="fp16").name == "stock_dev-fp16.json"
    assert loglik.metrics_path("heldout_nll", "stock", "dev", dtype="bf16").name == "stock_dev.json"


@pytest.mark.torch
def test_each_precision_scores_a_cell_into_its_own_file(stock):
    """V05: the bf16 run (generation.dtype) of a cell an fp32 run scored first (a T4's fp16 line, on the
    CPU) scores into its own file: neither is refused, appended to or replaced, and each resumes its own."""
    rows, prompts = bel_rows(2), [utility(1), utility(2, "ind")]
    bf16 = dataclasses.replace(stock, dtype="bf16")
    cells = {"belebele": (lambda v: loglik.belebele_cell(v, rows, lang="en", split="dev"),
                          lambda dtype: loglik.belebele_scores_path("stock", "en", "dev", dtype=dtype)),
             "heldout": (lambda v: loglik.heldout_cell(v, prompts, split="dev"),
                         lambda dtype: loglik.heldout_scores_path("stock", "dev", dtype=dtype))}
    for score, path in cells.values():
        first = score(stock)
        assert first == path("fp32") and first.name.endswith("_dev-fp32.jsonl")
        before = first.read_bytes()
        second = score(bf16)
        assert second == path("bf16") == path(None) and second.name.endswith("_dev.jsonl")
        assert first.read_bytes() == before                                  # the fp32 file as it was
        after = second.read_bytes()
        for made, dtype in ((first, "fp32"), (second, "bf16")):
            records = list(read_jsonl(made))
            assert len(records) == 2 and {r["meta"]["dtype"] for r in records} == {dtype}
        assert (score(stock), score(bf16)) == (first, second)                # each resumes its own: done
        assert (first.read_bytes(), second.read_bytes()) == (before, after)


@pytest.mark.torch
def test_a_file_holding_another_precisions_scores_is_refused_and_never_started_over(stock):
    """The guard (C08, kept by V05). lift writes each precision's scores to its own file, so a file at the
    bf16 path holding fp32 scores was put there by hand. One file is one precision: nothing is appended. It
    is not stale and names no command line, as a generations file's OtherPrecisionError, so --restart skips
    it: starting it over would discard scores bf16 cannot make again. The remedy says where they belong."""
    from lift.models.generate import OtherPrecisionError

    rows, prompts = bel_rows(2), [utility(1)]
    bf16 = dataclasses.replace(stock, dtype="bf16")
    placed = []
    for made in (loglik.belebele_cell(stock, rows, lang="en", split="dev"),
                 loglik.heldout_cell(stock, prompts, split="dev")):
        placed.append(made.with_name(made.name.replace("-fp32", "")))          # at bf16's path, by hand
        shutil.copyfile(made, placed[-1])
    before = [p.read_bytes() for p in placed]
    with pytest.raises(loglik.ScorePrecisionError) as exc:
        loglik.belebele_cell(bf16, rows, lang="en", split="dev")
    assert isinstance(exc.value, OtherPrecisionError) and exc.value.dtype == "fp32"
    assert exc.value.command is None and "lift belebele" not in str(exc.value)
    assert exc.value.remedy == ("holds fp32 scores, which belong in outputs/scores/belebele/stock/"
                                "belebele_en_dev-fp32.jsonl: move them there by hand, leaving this cell's "
                                "file empty (sync never deletes), or leave it out with --langs")
    with pytest.raises(loglik.ScorePrecisionError) as exc:
        loglik.heldout_cell(bf16, prompts, split="dev")
    assert exc.value.command is None and "lift heldout-loss" not in str(exc.value)
    assert exc.value.remedy == ("holds fp32 scores, which belong in outputs/scores/heldout_loss/stock/"
                                "utility_hi_dev-fp32.jsonl: move them there by hand, leaving this cell's "
                                "file empty (sync never deletes), or leave this line out of the job")
    assert [p.read_bytes() for p in placed] == before                        # nothing appended
    fp32 = loglik.belebele_scores_path("stock", "en", "dev", dtype="fp32")
    kept = fp32.read_bytes()
    with pytest.raises(StaleOutputError, match="another configuration") as exc:
        loglik.belebele_cell(dataclasses.replace(stock, template_hash="f" * 64), rows, lang="en", split="dev")
    assert not isinstance(exc.value, loglik.ScorePrecisionError)
    assert fp32.read_bytes() == kept


@pytest.mark.torch
def test_a_moved_stack_refuses_to_append_but_reuses_a_finished_file(stock, monkeypatch):
    from lift.models import generate as gen

    rows = bel_rows(3)
    path = loglik.belebele_cell(stock, rows[:2], lang="en", split="dev")
    before = path.read_bytes()
    real = gen.stack
    monkeypatch.setattr(gen, "stack", lambda device: {**real(device), "transformers": "9.9.9"})
    with pytest.raises(StaleOutputError, match="another stack") as exc:
        loglik.belebele_cell(stock, rows, lang="en", split="dev")
    assert exc.value.keys_changed is False and exc.value.command.endswith("--restart --dtype fp32")
    assert loglik.belebele_cell(stock, rows[:2], lang="en", split="dev") == path        # finished: reused
    assert path.read_bytes() == before


# --- metrics ----------------------------------------------------------------------------------------------


@pytest.mark.torch
def test_belebele_metrics_cover_every_complete_language_and_list_the_rest(stock):
    for lang in ("en", "hi"):
        write_belebele(lang, [*bel_rows(4, lang=lang), bel_row(9, lang=lang, split="final")])
    path = loglik.belebele_cell(stock, bel_rows(4, lang="en"), lang="en", split="dev")
    loglik.belebele_cell(stock, bel_rows(4, lang="hi")[:2], lang="hi", split="dev")    # hi: incomplete
    written = loglik.write_belebele_metrics("stock", "dev", dtype="fp32",
                                            run_config=loglik.belebele_run_config(stock))
    assert written.json_path == metrics_dir() / "belebele" / "stock_dev-fp32.json"   # not generation.dtype
    doc = json.loads(written.json_path.read_text(encoding="utf-8"))
    assert doc == json.loads(json.dumps(written.doc))
    head = ("kind", "version", "variant", "split", "format_version", "dtype")
    assert tuple(doc[k] for k in head) == ("belebele", 1, "stock", "dev", 1, "fp32")
    assert doc["model_rev"] == stock.model_rev and doc["adapter_hash"] is None
    assert doc["scoring"] == "sequence"
    records = list(read_jsonl(path))
    en = doc["langs"]["en"]
    assert en["n"] == 4 and en["n_correct"] == sum(r["pred"] == r["gold"] for r in records)
    assert en["accuracy"] == en["n_correct"] / 4 and en["gate_fired_rate"] is None
    assert sum(en["pred_counts"].values()) == 4 and en["n_ties"] == 0
    assert en["scores_path"] == "outputs/scores/belebele/stock/belebele_en_dev-fp32.jsonl"
    assert en["scores_sha256"] == sha256_file(path) and doc["created"]
    assert list(doc["langs"]) == ["en"] and doc["left_out"] == {"hi": "incomplete: 2 of 4 items scored"}
    md = written.md_path.read_text(encoding="utf-8")
    assert f"| en | 4 | {en['n_correct']} |" in md and "- hi: incomplete" in md
    assert en["n_outside_split"] == 0
    # An item that left the split (an exclusion) is left aside and counted; the other three are the split.
    # Its report lists the exclusion, as lift data apply-spotchecks records a reader's bad verdict: since R1
    # a prompt file short of its frozen split is refused unless its report explains every missing pid.
    write_belebele("en", bel_rows(4, lang="en")[1:])
    shown = prompts_dir() / "belebele_en.jsonl"
    report = {"set": "belebele", "lang": "en", "file": shown.name, "sha256": sha256_file(shown),
              "translation_failures": [], "exclusions": [{"pid": "bel-000", "reason": "spotcheck:bad"}]}
    (prompts_dir() / "reports").mkdir(parents=True, exist_ok=True)
    (prompts_dir() / "reports" / "belebele_en.json").write_text(json.dumps(report), encoding="utf-8")
    en = loglik.belebele_metrics("stock", "dev", dtype="fp32")["langs"]["en"]
    assert en["n"] == 3 and en["n_outside_split"] == 1
    assert en["n_correct"] == sum(r["pred"] == r["gold"] for r in records if r["pid"] != "bel-000")
    # Under another configuration nothing qualifies: the files are retired (left empty), never removed.
    assert loglik.write_belebele_metrics("stock", "dev", dtype="fp32", run_config="0" * 16) is None
    assert written.json_path.read_bytes() == b"" and written.md_path.read_bytes() == b""
    assert loglik.read_metrics(written.json_path) is None


@pytest.mark.torch
def test_heldout_metrics_pool_by_source_and_pair_with_stock(stock, fake_prompt_files):
    from fakes import build_tiny_cohere2

    from lift.stats import paired_bootstrap

    prompts = [utility(i, "aya" if i < 3 else "ind") for i in range(5)]
    fake_prompt_files[("utility", "hi")] = [*prompts, utility(9, split="final")]
    adapter = dataclasses.replace(stock, model=build_tiny_cohere2(stock.tokenizer, seed=7),
                                  adapter_hash="a" * 64,
                                  spec=VariantSpec(name="adapter-x", method="adapter", adapter="adapter-x"))
    loglik.heldout_cell(adapter, prompts, split="dev")
    doc = loglik.write_heldout_metrics("adapter-x", "dev", dtype="fp32").doc
    # Stock's line in this run's precision: without --dtype it would score stock in generation.dtype.
    assert doc["vs_stock"] is None
    assert "run `lift heldout-loss --variant stock --split dev --dtype fp32`, then" in doc["vs_stock_note"]
    loglik.heldout_cell(stock, prompts, split="dev")
    written = loglik.write_heldout_metrics("adapter-x", "dev", dtype="fp32")
    doc = json.loads(written.json_path.read_text(encoding="utf-8"))
    assert written.json_path == metrics_dir() / "heldout_loss" / "adapter-x_dev-fp32.json"
    mine = {r["pid"]: r for r in read_jsonl(loglik.heldout_scores_path("adapter-x", "dev", dtype="fp32"))}
    base = {r["pid"]: r for r in read_jsonl(loglik.heldout_scores_path("stock", "dev", dtype="fp32"))}
    assert (doc["kind"], doc["version"], doc["lang"], doc["n_rows"]) == ("heldout_loss", 1, "hi", 5)
    assert doc["n_outside_split"] == 0 and doc["split"] == "dev"
    assert doc["n_tokens"] == sum(r["n_tokens"] for r in mine.values())
    assert doc["mean_nll"] == pytest.approx(sum(r["nll_sum"] for r in mine.values()) / doc["n_tokens"])
    assert doc["row_mean_nll"] == pytest.approx(sum(r["mean_nll"] for r in mine.values()) / 5)
    assert doc["perplexity"] == pytest.approx(math.exp(doc["mean_nll"]))
    assert doc["adapter_hash"] == "a" * 64 and doc["scores_sha256"] == sha256_file(
        loglik.heldout_scores_path("adapter-x", "dev", dtype="fp32"))
    assert {s: v["n_rows"] for s, v in doc["by_source"].items()} == {"aya": 3, "ind": 2}
    pids = sorted(mine)
    res = paired_bootstrap([mine[p]["mean_nll"] for p in pids], [base[p]["mean_nll"] for p in pids],
                           clusters=pids)
    vs = doc["vs_stock"]
    assert (vs["diff_row_mean_nll"], vs["lo"], vs["hi"], vs["p_value"], vs["n"]) == (
        res.diff, res.lo, res.hi, res.p_value, 5)
    assert vs["diff_row_mean_nll"] != 0                                   # guard: the two models differ
    assert set(vs["by_source"]) == {"aya", "ind"} and vs["by_source"]["ind"]["n"] == 2
    assert vs["stock_scores_path"] == "outputs/scores/heldout_loss/stock/utility_hi_dev-fp32.jsonl"
    assert vs["stock_scores_sha256"] == sha256_file(loglik.heldout_scores_path("stock", "dev", dtype="fp32"))
    assert "vs stock: this variant minus stock" in written.md_path.read_text(encoding="utf-8")
    stock_doc = loglik.write_heldout_metrics("stock", "dev", dtype="fp32").doc
    assert stock_doc["vs_stock"] is None and stock_doc["vs_stock_note"] == "this is stock"


@pytest.mark.torch
def test_the_secondary_selection_reads_the_heldout_gain_from_the_score_files(stock, fake_prompt_files,
                                                                             monkeypatch):
    """Decisions item 73: lift select's secondary rule reads each variant's pooled NLL difference from stock
    (heldout_metrics' vs_stock), and only while the scores are the registered adapter's and pair with
    stock's; anything else is missing, with the line that makes it."""
    from fakes import build_tiny_cohere2

    from lift import config
    from lift.repair import select

    prompts = [utility(i, "aya" if i < 3 else "ind") for i in range(5)]
    fake_prompt_files[("utility", "hi")] = prompts
    registry = {"variants": [{"name": "stock", "method": "stock"},
                             {"name": "adapter-x", "method": "adapter", "adapter": "adapter-x", "seed": 1,
                              "artifacts": {"adapter": "a" * 64}}]}
    real = config.load
    monkeypatch.setattr(config, "load", lambda name="project": registry if name == "variants" else real(name))
    with pytest.raises(select.Missing, match="run `lift heldout-loss --variant adapter-x --dtype fp32`"):
        select._heldout("adapter-x", "fp32")
    spec = VariantSpec(name="adapter-x", method="adapter", adapter="adapter-x", seed=1,
                       artifacts={"adapter": "a" * 64})
    adapter = dataclasses.replace(stock, model=build_tiny_cohere2(stock.tokenizer, seed=7),
                                  adapter_hash="a" * 64, spec=spec)
    loglik.heldout_cell(adapter, prompts, split="dev")
    with pytest.raises(select.Missing, match="cannot be paired with stock's"):
        select._heldout("adapter-x", "fp32")
    loglik.heldout_cell(stock, prompts, split="dev")
    value, details = select._heldout("adapter-x", "fp32")
    doc = loglik.heldout_metrics("adapter-x", "dev", dtype="fp32")
    stock_doc = loglik.heldout_metrics("stock", "dev", dtype="fp32")
    assert value == doc["vs_stock"]["diff_mean_nll"] == pytest.approx(doc["mean_nll"] - stock_doc["mean_nll"])
    assert value != 0 and details["n_rows"] == 5 and details["mean_nll"] == doc["mean_nll"]
    assert (details["lo"], details["hi"]) == (doc["vs_stock"]["lo"], doc["vs_stock"]["hi"])
    # Scores made under another registry entry (an adapter trained again) are not this entry's.
    registry["variants"][1]["artifacts"]["adapter"] = "b" * 64
    with pytest.raises(select.Missing, match="made under another registry entry.*--restart --dtype fp32"):
        select._heldout("adapter-x", "fp32")
    with pytest.raises(select.Missing, match="does not list adapter-y"):
        select._heldout("adapter-y", "fp32")


@pytest.mark.torch
def test_each_precision_pairs_with_stocks_scores_of_its_own_precision(stock, fake_prompt_files):
    """V05: an adapter's held-out loss pairs with stock's file of its own precision, and neither precision's
    files or metrics change when the other's cells are scored. bf16 stands for Colab's runs and fp32 for a
    T4's fp16 ones; bf16's stock is another tiny model, so that the two stock files really differ."""
    from fakes import build_tiny_cohere2

    from lift.stats import paired_bootstrap

    prompts = [utility(i, "aya" if i % 2 else "ind") for i in range(4)]
    fake_prompt_files[("utility", "hi")] = prompts
    adapter = dataclasses.replace(stock, model=build_tiny_cohere2(stock.tokenizer, seed=7),
                                  adapter_hash="a" * 64,
                                  spec=VariantSpec(name="adapter-x", method="adapter", adapter="adapter-x"))
    loglik.heldout_cell(dataclasses.replace(stock, model=build_tiny_cohere2(stock.tokenizer, seed=11),
                                            dtype="bf16"), prompts, split="dev")
    loglik.heldout_cell(dataclasses.replace(adapter, dtype="bf16"), prompts, split="dev")
    bf16 = loglik.write_heldout_metrics("adapter-x", "dev")
    assert bf16.json_path.name == "adapter-x_dev.json" and bf16.doc["dtype"] == "bf16"
    kept = [bf16.json_path, bf16.md_path, loglik.heldout_scores_path("stock", "dev"),
            loglik.heldout_scores_path("adapter-x", "dev")]
    kept = {p: p.read_bytes() for p in kept}
    loglik.heldout_cell(adapter, prompts, split="dev")                   # fp32, and stock has no fp32 file
    doc = loglik.write_heldout_metrics("adapter-x", "dev", dtype="fp32").doc
    assert doc["vs_stock"] is None                                        # never paired with bf16's stock
    assert doc["vs_stock_note"] == ("stock's held-out scores of dev cannot be paired (no score file); run "
                                    "`lift heldout-loss --variant stock --split dev --dtype fp32`, then this "
                                    "variant's line again")
    loglik.heldout_cell(stock, prompts, split="dev")                     # that line: stock's own fp32 file
    fp32 = loglik.write_heldout_metrics("adapter-x", "dev", dtype="fp32")
    assert fp32.json_path.name == "adapter-x_dev-fp32.json"
    assert {p: p.read_bytes() for p in kept} == kept                      # bf16's files never touched
    for written, dtype in ((bf16, "bf16"), (fp32, "fp32")):
        vs = written.doc["vs_stock"]
        mine, base = (loglik.heldout_scores_path(name, "dev", dtype=dtype) for name in ("adapter-x", "stock"))
        assert vs["stock_scores_path"] == "outputs/scores/heldout_loss/stock/" + base.name
        assert vs["stock_scores_sha256"] == sha256_file(base)
        x, y = ({r["pid"]: r["mean_nll"] for r in read_jsonl(p)} for p in (mine, base))
        pids = sorted(x)
        res = paired_bootstrap([x[p] for p in pids], [y[p] for p in pids], clusters=pids)
        assert (vs["diff_row_mean_nll"], vs["lo"], vs["hi"], vs["n"]) == (res.diff, res.lo, res.hi, 4)
    assert bf16.doc["vs_stock"]["diff_row_mean_nll"] != fp32.doc["vs_stock"]["diff_row_mean_nll"]
    again = loglik.heldout_metrics("adapter-x", "dev")                    # bf16's numbers, recomputed
    assert {k: v for k, v in again.items() if k != "created"} == {
        k: v for k, v in bf16.doc.items() if k != "created"}


@pytest.mark.torch
def test_vs_stock_never_pairs_a_file_holding_another_precisions_scores(stock, fake_prompt_files):
    """The guard in vs_stock (C08): stock's fp32 file holding bf16 scores (put there by hand) is never
    paired, and the note says where they belong rather than only naming stock's fp32 line, which would
    refuse the file."""
    prompts = [utility(i, "aya" if i % 2 else "ind") for i in range(3)]
    fake_prompt_files[("utility", "hi")] = prompts
    made = loglik.heldout_cell(dataclasses.replace(stock, dtype="bf16"), prompts, split="dev")
    shutil.copyfile(made, loglik.heldout_scores_path("stock", "dev", dtype="fp32"))      # by hand
    adapter = dataclasses.replace(stock, adapter_hash="a" * 64,
                                  spec=VariantSpec(name="adapter-x", method="adapter", adapter="adapter-x"))
    loglik.heldout_cell(adapter, prompts, split="dev")                                    # fp32: a T4's fp16
    doc = loglik.heldout_metrics("adapter-x", "dev", dtype="fp32")
    assert doc["n_rows"] == 3 and doc["vs_stock"] is None
    note = doc["vs_stock_note"]
    assert note.startswith("stock's fp32 held-out file of dev holds scores made in bf16, which belong in "
                           "outputs/scores/heldout_loss/stock/utility_hi_dev.jsonl: ")
    assert note.endswith("then run `lift heldout-loss --variant stock --split dev --dtype fp32` and this "
                         "variant's line again")
    assert "Move them there by hand, leaving that file empty (sync never deletes)" in note
    assert "--restart" not in note


# --- the commands -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["belebele", "heldout-loss"])
def test_help_works_without_a_gpu_stack(command, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)                     # `import torch` raises ImportError
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0 and "usage" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["belebele", "--variant", "stock", "--split", "dev", "--langs", "hi"],
                                  ["heldout-loss", "--variant", "stock"]])
def test_the_pin_refusal_comes_before_torch(argv, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="unpinned revisions"):
        main(argv)


def test_a_missing_prompt_file_stops_before_any_model_is_loaded(monkeypatch):
    from lift.models import variants

    pin()
    monkeypatch.setattr(variants, "build_variant", lambda *a, **k: pytest.fail("no model may load"))
    with pytest.raises(SystemExit, match="lift sync pull"):
        main(["belebele", "--variant", "stock", "--split", "dev", "--langs", "hi", "--device", "cpu"])


@pytest.mark.torch
def test_lift_belebele_scores_each_language_and_writes_the_metrics(tiny_model_dir, monkeypatch, capsys):
    from fake_sources import belebele_rows_by_config

    from lift.data.sources import belebele_rows

    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    by_config = belebele_rows_by_config()
    for lang, code in (("en", "eng_Latn"), ("hi", "hin_Deva")):
        rows = belebele_rows(by_config, code)
        write_belebele(lang, [{**r, "split": "dev" if i < 3 else "final"} for i, r in enumerate(rows)])
    argv = ["belebele", "--variant", "stock", "--split", "dev", "--langs", "hi,en", "--device", "cpu"]
    assert main(argv) == 0
    doc = loglik.read_metrics(metrics_dir() / "belebele" / "stock_dev.json")
    assert list(doc["langs"]) == ["en", "hi"] and doc["dtype"] == "bf16" and not doc["left_out"]
    assert all(c["n"] == 3 for c in doc["langs"].values())
    out = capsys.readouterr().out
    assert "belebele stock dev (bf16, sequence scoring): en " in out and "stock_dev.json" in out
    assert "gardening" not in out and "बागवानी" not in out                    # numbers only (rule 6)
    assert main(argv) == 0                                                  # all scored: nothing to do
    assert sorted(d["n_written"] for d in manifests("belebele")) == [0, 0, 3, 3]


@pytest.mark.torch
def test_lift_belebele_skips_a_stale_cell_and_restart_starts_it_over(tiny_model_dir, monkeypatch, caplog):
    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    for lang in ("en", "hi"):
        write_belebele(lang, bel_rows(3, lang=lang))
    argv = ["belebele", "--variant", "stock", "--split", "dev", "--langs", "en", "hi", "--device", "cpu"]
    assert main(argv) == 0
    en, hi = (loglik.belebele_scores_path("stock", lang, "dev") for lang in ("en", "hi"))
    en_before, hi_before = en.read_bytes(), hi.read_bytes()
    rows = bel_rows(3, lang="hi")
    rows[0] = {**rows[0], "question": rows[0]["question"] + " (fixed)"}
    write_belebele("hi", rows)
    assert main(argv) == 3                                                  # en done, hi listed
    line = "lift belebele --variant stock --split dev --langs hi --restart"
    assert f"belebele_hi_dev: {line}\n" in caplog.text
    assert hi.read_bytes() == hi_before
    doc = loglik.read_metrics(metrics_dir() / "belebele" / "stock_dev.json")
    assert list(doc["langs"]) == ["en"] and "other texts" in doc["left_out"]["hi"]
    assert main([*argv, "--restart"]) == 0
    aside = list((output_dir() / ".aside").rglob("belebele_hi_dev.jsonl"))
    assert len(aside) == 1 and aside[0].read_bytes() == hi_before
    assert len(hi.read_bytes().splitlines()) == 3 and hi.read_bytes() != hi_before
    assert en.read_bytes() == en_before                                     # a current cell is left as it is
    doc = loglik.read_metrics(metrics_dir() / "belebele" / "stock_dev.json")
    assert list(doc["langs"]) == ["en", "hi"] and not doc["left_out"]


@pytest.mark.torch
def test_lift_belebele_keeps_each_precisions_cells_apart(tiny_model_dir, monkeypatch):
    """V05: Colab's bf16 line over the cells a T4's fp16 line scored first (fp32 here, on the CPU), and the
    other way round, scores its own files: nothing is refused or set aside, --restart included, and neither
    precision's files or metrics change when the other runs."""
    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    for lang in ("en", "hi"):
        write_belebele(lang, bel_rows(2, lang=lang))
    argv = ["belebele", "--variant", "stock", "--split", "dev", "--langs", "hi", "en", "--device", "cpu"]
    files = {dtype: [loglik.belebele_scores_path("stock", lang, "dev", dtype=dtype) for lang in ("en", "hi")]
             for dtype in ("bf16", "fp32")}
    metrics = {dtype: loglik.metrics_path("belebele", "stock", "dev", dtype=dtype) for dtype in files}
    assert main([*argv, "--dtype", "fp32"]) == 0                          # the T4's line first
    for first, flag in (("fp32", []), ("bf16", ["--dtype", "fp32"])):
        kept = {p: p.read_bytes() for p in (*files[first], metrics[first], metrics[first].with_suffix(".md"))}
        for restart in ([], ["--restart"]):
            assert main([*argv, *flag, *restart]) == 0                    # the other precision's line
            assert {p: p.read_bytes() for p in kept} == kept
    assert not (output_dir() / ".aside").exists()
    for dtype, paths in files.items():
        assert all({r["meta"]["dtype"] for r in read_jsonl(p)} == {dtype} for p in paths)
        doc = loglik.read_metrics(metrics[dtype])
        assert doc["dtype"] == dtype and list(doc["langs"]) == ["en", "hi"] and not doc["left_out"]
        assert [c["scores_path"] for c in doc["langs"].values()] == [
            "outputs/scores/belebele/stock/" + p.name for p in paths]


GUARD_LISTED = {
    "belebele": ("belebele_hi_dev: holds fp32 scores, which belong in outputs/scores/belebele/stock/"
                 "belebele_hi_dev-fp32.jsonl: move them there by hand, leaving this cell's file empty (sync "
                 "never deletes), or leave it out with --langs\n"),
    "heldout_nll": ("utility_hi_dev: holds fp32 scores, which belong in outputs/scores/heldout_loss/stock/"
                    "utility_hi_dev-fp32.jsonl: move them there by hand, leaving this cell's file empty "
                    "(sync never deletes), or leave this line out of the job\n"),
}


@pytest.mark.torch
@pytest.mark.parametrize("task", ["belebele", "heldout_nll"])
def test_lift_never_starts_over_a_file_holding_another_precisions_scores(task, tiny_model_dir, monkeypatch,
                                                                          caplog, request):
    """The guard (C08, kept by V05), end to end: a score file at the bf16 path holding fp32 scores (put
    there by hand) is skipped and listed with where its scores belong, --restart included, never with a
    --restart line; nothing is set aside, and the fp32 file and metrics are left as they are."""
    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    stock_line = ["--variant", "stock", "--device", "cpu"]
    if task == "belebele":
        write_belebele("hi", bel_rows(2, lang="hi"))
        line = ["belebele", *stock_line, "--split", "dev", "--langs", "hi"]
        made, placed = (loglik.belebele_scores_path("stock", "hi", "dev", dtype=d) for d in ("fp32", "bf16"))
    else:                                   # fake_prompt_files replaces the reader the Belebele line needs
        store = request.getfixturevalue("fake_prompt_files")
        store[("utility", "hi")] = [utility(i, "aya" if i % 2 else "ind") for i in range(2)]
        line = ["heldout-loss", *stock_line]
        made, placed = (loglik.heldout_scores_path("stock", "dev", dtype=d) for d in ("fp32", "bf16"))
    assert main([*line, "--dtype", "fp32"]) == 0
    shutil.copyfile(made, placed)                                         # by hand, at bf16's path
    fp32 = loglik.metrics_path(task, "stock", "dev", dtype="fp32")
    kept = {p: p.read_bytes() for p in (made, placed, fp32, fp32.with_suffix(".md"))}
    for restart in ([], ["--restart"]):
        caplog.clear()
        assert main([*line, *restart]) == 3
        assert GUARD_LISTED[task] in caplog.text
        assert " --restart\n" not in caplog.text and "set aside in" not in caplog.text
        assert {p: p.read_bytes() for p in kept} == kept
    assert not (output_dir() / ".aside").exists()
    assert not loglik.metrics_path(task, "stock", "dev").exists()         # nothing qualifies in bf16


@pytest.mark.torch
def test_belebele_metrics_are_per_language_and_per_precision(tiny_model_dir, monkeypatch):
    """C09: a refused or started-over cell never empties another language's numbers, and a run never
    changes another precision's: each precision has its own score files (V05), so the bf16 line scores
    beside fp32's, and an fp32 restart sets aside only fp32's file. The run's own metrics are recomputed
    from the files that still qualify, the cell listed under left_out meanwhile."""
    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    for lang in ("en", "hi"):
        write_belebele(lang, bel_rows(3, lang=lang))
    argv = ["belebele", "--variant", "stock", "--split", "dev", "--device", "cpu"]
    assert main([*argv, "--langs", "hi", "en", "--dtype", "fp32"]) == 0
    fp32 = metrics_dir() / "belebele" / "stock_dev-fp32.json"
    before = fp32.read_bytes()
    doc = loglik.read_metrics(fp32)
    assert list(doc["langs"]) == ["en", "hi"] and not doc["left_out"]
    assert main([*argv, "--langs", "hi", "en", "--restart"]) == 0    # bf16: its own files, beside fp32's
    assert fp32.read_bytes() == before and not (output_dir() / ".aside").exists()
    bf16 = metrics_dir() / "belebele" / "stock_dev.json"
    assert list(loglik.read_metrics(bf16)["langs"]) == ["en", "hi"]
    bf16_kept = {p: p.read_bytes() for p in (bf16, bf16.with_suffix(".md"), *(
        loglik.belebele_scores_path("stock", lang, "dev") for lang in ("en", "hi")))}
    # In the run's own precision a restart moves only that language to left_out while it is scored again.
    rows = bel_rows(3, lang="en")
    rows[0] = {**rows[0], "question": rows[0]["question"] + " (fixed)"}
    write_belebele("en", rows)
    seen, real = [], loglik.belebele_cell

    def watched(variant, cell_rows, *, lang, **kw):
        if lang == "en" and not loglik.belebele_scores_path("stock", "en", "dev", dtype="fp32").read_bytes():
            seen.append(loglik.read_metrics(fp32))                          # emptied, before it is scored
        return real(variant, cell_rows, lang=lang, **kw)

    monkeypatch.setattr(loglik, "belebele_cell", watched)
    assert main([*argv, "--langs", "en", "--dtype", "fp32", "--restart"]) == 0
    assert len(seen) == 1 and list(seen[0]["langs"]) == ["hi"]
    assert seen[0]["langs"]["hi"] == doc["langs"]["hi"]
    assert seen[0]["left_out"] == {"en": "an empty score file (started over)"}
    final = loglik.read_metrics(fp32)
    assert list(final["langs"]) == ["en", "hi"] and not final["left_out"]
    assert final["langs"]["hi"] == doc["langs"]["hi"] and final["langs"]["en"]["n"] == 3
    # bf16's en file is stale too, but only a bf16 line starts it over: its files and numbers stay.
    assert {p: p.read_bytes() for p in bf16_kept} == bf16_kept
    assert [p.name for p in (output_dir() / ".aside").rglob("*.jsonl")] == ["belebele_en_dev-fp32.jsonl"]


@pytest.mark.torch
def test_lift_heldout_loss_for_stock_then_an_adapter_pairs_them(tiny_model_dir, tiny_lora_adapter,
                                                                fake_prompt_files, monkeypatch, capsys):
    pytest.importorskip("peft")
    from lift.models import variants
    from lift.train.merge import adapter_hash

    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    name = "adapter-aya-std-s1"
    shutil.copytree(tiny_lora_adapter, variants.adapter_path(name))
    real_spec = variants.get_spec
    monkeypatch.setattr(variants, "get_spec", lambda n: VariantSpec(name=name, method="adapter", adapter=name,
                                                                    seed=1) if n == name else real_spec(n))
    fake_prompt_files[("utility", "hi")] = [*(utility(i, "aya" if i % 2 else "ind") for i in range(4)),
                                            utility(7, split="final")]
    assert main(["heldout-loss", "--variant", "stock", "--device", "cpu"]) == 0
    assert main(["heldout-loss", "--variant", name, "--device", "cpu"]) == 0
    doc = loglik.read_metrics(metrics_dir() / "heldout_loss" / f"{name}_dev.json")
    assert doc["n_rows"] == 4 and doc["split"] == "dev" and doc["vs_stock"]["n"] == 4
    assert doc["adapter_hash"] == adapter_hash(variants.adapter_path(name))
    stock_doc = loglik.read_metrics(metrics_dir() / "heldout_loss" / "stock_dev.json")
    assert stock_doc["vs_stock"] is None and stock_doc["adapter_hash"] is None
    out = capsys.readouterr().out
    assert "vs stock: row mean NLL" in out and "फूल" not in out and "बगीचे" not in out


@pytest.mark.torch
def test_lift_heldout_loss_pairs_each_precision_with_its_own_stock(tiny_model_dir, tiny_lora_adapter,
                                                                    fake_prompt_files, monkeypatch):
    """V05 for the held-out loss, end to end: a T4's fp16 lines (fp32 here) after Colab's bf16 ones make
    stock's and the adapter's own files of that precision, --restart included, and the adapter pairs with
    stock's scores of its own precision. Stock's bf16 file, which every bf16 adapter pairs with, and the
    bf16 metrics are never touched."""
    pytest.importorskip("peft")
    from lift.models import variants

    pin()
    tiny_build(monkeypatch, tiny_model_dir)
    name = "adapter-aya-std-s1"
    shutil.copytree(tiny_lora_adapter, variants.adapter_path(name))
    real_spec = variants.get_spec
    monkeypatch.setattr(variants, "get_spec", lambda n: VariantSpec(name=name, method="adapter", adapter=name,
                                                                    seed=1) if n == name else real_spec(n))
    fake_prompt_files[("utility", "hi")] = [utility(i, "aya" if i % 2 else "ind") for i in range(4)]
    for variant in ("stock", name):
        assert main(["heldout-loss", "--variant", variant, "--device", "cpu"]) == 0
    bf16 = [loglik.metrics_path("heldout_nll", v, "dev") for v in ("stock", name)]
    bf16 += [m.with_suffix(".md") for m in bf16]
    bf16 += [loglik.heldout_scores_path(v, "dev") for v in ("stock", name)]
    kept = {p: p.read_bytes() for p in bf16}
    for variant in ("stock", name):
        assert main(["heldout-loss", "--variant", variant, "--device", "cpu", "--dtype", "fp32",
                     "--restart"]) == 0
    assert {p: p.read_bytes() for p in kept} == kept and not (output_dir() / ".aside").exists()
    for dtype in ("bf16", "fp32"):
        doc = loglik.read_metrics(loglik.metrics_path("heldout_nll", name, "dev", dtype=dtype))
        stock_file = loglik.heldout_scores_path("stock", "dev", dtype=dtype)
        assert doc["dtype"] == dtype and doc["n_rows"] == 4 and doc["vs_stock"]["n"] == 4
        assert doc["vs_stock"]["stock_scores_path"] == "outputs/scores/heldout_loss/stock/" + stock_file.name
        assert doc["vs_stock"]["stock_scores_sha256"] == sha256_file(stock_file)
        assert {r["meta"]["dtype"] for r in read_jsonl(stock_file)} == {dtype}


@pytest.mark.torch
def test_a_wrapper_without_score_logits_exits_2(stock, fake_prompt_files, monkeypatch):
    from lift.models import variants

    pin()
    monkeypatch.setattr(variants, "build_variant", lambda spec, *, device, dtype: dataclasses.replace(
        stock, model=object(), base=stock.model))
    fake_prompt_files[("utility", "hi")] = [utility(1)]
    assert main(["heldout-loss", "--variant", "stock", "--device", "cpu"]) == 2
    assert not scores_dir().exists()                                     # no score file, in any precision
