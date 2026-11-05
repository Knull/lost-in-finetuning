"""lift.train.lora, ``lift train`` and ``lift register-adapter`` (P03), on the tiny cohere2 model, offline,
on CPU.

The runs repo is never reached: lift.sync.push and lift.sync.preflight are replaced by a recorder
(``hub``). The default registry is a small synthetic file in tmp in the layout of configs/variants.yaml
(``registry``, ``REGISTRY_TEXT``): no test reads or writes the real one, so the entries committed there
(the pilot's adapter, P05's repairs) change no outcome. Merge parity decodes 8 new tokens instead of 256
(``short_parity``), which changes nothing it compares.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("datasets")

import yaml  # noqa: E402
from fakes import make_prompt  # noqa: E402

from lift import config, gate0, sync  # noqa: E402
from lift.cli import main  # noqa: E402
from lift.io import sha256_file, sha256_text, write_jsonl  # noqa: E402
from lift.models import variants  # noqa: E402
from lift.models.load import encode_chat  # noqa: E402
from lift.paths import adapters_dir, manifests_dir, output_dir, sft_dir  # noqa: E402
from lift.prompts import SAFETY_SETS  # noqa: E402
from lift.schemas import LANGS  # noqa: E402
from lift.train import lora, merge  # noqa: E402

NAME = "adapter-aya-std-s1"
PARITY = ["hi", "नमस्ते, आप कैसे हैं?"]
COMMIT = "b" * 40
SHA = "c" * 40
# The user turn alone (byte-level tokens) fills max_seq_len 1024 after the tiny preamble.
LONG = {"id": "aya-09999", "messages": [{"role": "user", "content": "x " * 600},
                                        {"role": "assistant", "content": "ok"}]}
# The tests' registry: configs/variants.yaml's layout (comments, a blank line, a two-space list), stock only.
REGISTRY_TEXT = ("# A test registry in the layout of configs/variants.yaml; its comments must survive.\n"
                 "# Names: stock, adapter-{candidate}-{std|stress}-s{seed}\n"
                 "\n"
                 "variants:\n"
                 "  - name: stock\n"
                 "    method: stock\n")


@pytest.fixture(autouse=True)
def registry(tmp_path, monkeypatch):
    """``REGISTRY_TEXT`` in tmp as the default registry: no test reads or writes configs/variants.yaml,
    whose entries (the pilot's adapter once registered, P05's repairs) must change no test's outcome."""
    path = tmp_path / "configs" / "variants.yaml"
    path.parent.mkdir()
    path.write_text(REGISTRY_TEXT, encoding="utf-8", newline="\n")
    monkeypatch.setattr(lora, "default_registry", lambda: path)
    return path


class Hub:
    """Stands in for the runs repo: records every push and preflight. ``fail`` holds exceptions the
    next pushes raise, in order; ``refuse`` is what preflight raises."""

    def __init__(self):
        self.pushes: list[list[str]] = []
        self.preflights = 0
        self.fail: list[BaseException] = []
        self.refuse: BaseException | None = None

    def push(self, subdirs=(), *, force=False, api=None):
        self.pushes.append(list(subdirs))
        if self.fail:
            raise self.fail.pop(0)
        return sync.SyncReport("push", "tester/lift-runs", "a" * 40, list(subdirs),
                               files={f"{subdirs[0]}/adapter_model.safetensors": "new"}, commits=[COMMIT])

    def preflight(self, *, write=True, api=None):
        self.preflights += 1
        if self.refuse is not None:
            raise self.refuse


@pytest.fixture(autouse=True)
def hub(monkeypatch):
    recorder = Hub()
    monkeypatch.setattr(sync, "push", recorder.push)
    monkeypatch.setattr(sync, "preflight", recorder.preflight)
    return recorder


@pytest.fixture(autouse=True)
def short_parity(monkeypatch):
    real = merge.parity_check
    monkeypatch.setattr(merge, "parity_check", lambda *a, **kw: real(*a, **{**kw, "max_new_tokens": 8}))


@pytest.fixture
def utility(fake_prompt_files):
    """Held-out utility prompts (lift.prompts.load("utility", "hi"): dev and final)."""
    fake_prompt_files[("utility", "hi")] = [
        make_prompt("utl-aya-00001", lang="hi", set_name="utility", split="dev", text="चाय कैसे बनती है?"),
        make_prompt("utl-ind-00002", lang="hi", set_name="utility", split="final",
                    text="café ka matlab kya hai?"),
    ]
    return fake_prompt_files


def rows(n: int, start: int = 0) -> list[dict]:
    return [{"id": f"aya-{i:05d}", "messages": [
        {"role": "user", "content": f"Question {i}: how is tea made?"},
        {"role": "assistant", "content": f"Answer {i}: boil water, add the leaves, then milk."}]}
        for i in range(start, start + n)]


def write_sft(train_rows: list[dict], key: str = "aya/dedup", *, planned: int | None = None,
              sha: str | None = None):
    """The training file and the data/sft/report.json entry that plans it (like lift data build)."""
    path = sft_dir() / key / "sft_messages.jsonl"
    write_jsonl(path, train_rows)
    report = {"decision": {"rows_by_variant": {key: len(train_rows) if planned is None else planned}},
              "variants": {key: {"sha256": sha or sha256_file(path)}}}
    (sft_dir() / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return path


def train(tiny_model_dir, candidate="aya", recipe="standard", seed=1, **options):
    kw = {"model_id": str(tiny_model_dir), "device": "cpu", "dtype": "fp32", "max_steps": 2,
          "parity_prompts": PARITY, **options}
    return lora.train_lora(candidate, recipe, seed, **kw)


def plan(tiny_model_dir, candidate="aya", recipe="standard", seed=1, **options):
    kw = {"model_id": str(tiny_model_dir), "device": "cpu", "dtype": "fp32", **options}
    return lora.plan_training(candidate, recipe, seed, **kw)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifests(**match):
    """The train manifests written so far whose fields equal ``match``."""
    if not manifests_dir().is_dir():
        return []
    found = [read(p) for p in sorted(manifests_dir().glob("train-*.json"))]
    return [m for m in found if all(m.get(k) == v for k, v in match.items())]


# --- recipe and request ------------------------------------------------------------------------------


def test_the_stress_recipe_inherits_the_standard_one():
    train_cfg = config.load("train")
    std, stress = lora.recipe_config("standard"), lora.recipe_config("stress")
    assert "inherit" not in stress and stress["lora"] == std["lora"] == train_cfg["standard"]["lora"]
    for key in ("lr", "epochs", "dedup", "candidate", "seeds"):
        assert stress[key] == train_cfg["stress"][key]
    assert stress["max_seq_len"] == std["max_seq_len"] and stress["precision"] == std["precision"]
    stress["lora"]["r"] = -1                               # a copy: the cached config is untouched
    assert config.load("train")["standard"]["lora"]["r"] == train_cfg["standard"]["lora"]["r"] != -1
    assert lora.adapter_name("aya", "standard", 1) == NAME
    assert lora.adapter_name("indicalign", "stress", 1) == "adapter-indicalign-stress-s1"


def test_requests_the_recipe_does_not_allow_are_refused(tiny_model_dir):
    with pytest.raises(lora.TrainRefused, match="trains only 'indicalign'"):
        plan(tiny_model_dir, "aya", "stress", 1)                  # decisions.md item 47
    with pytest.raises(lora.TrainRefused, match=r"seeds \[1\]"):
        plan(tiny_model_dir, "indicalign", "stress", 2)
    with pytest.raises(lora.TrainRefused, match="unknown candidate"):
        plan(tiny_model_dir, "dolly", "standard", 1)
    with pytest.raises(lora.TrainRefused, match="unknown recipe"):
        plan(tiny_model_dir, "aya", "huge", 1)
    with pytest.raises(lora.TrainRefused, match="CUDA_VISIBLE_DEVICES"):
        plan(tiny_model_dir, device="cuda:1")
    with pytest.raises(lora.TrainRefused, match="parity-dtype"):
        lora.check_parity_dtype("bf16", "cpu")
    stress = plan(tiny_model_dir, "indicalign", "stress", 1)
    assert stress.name == "adapter-indicalign-stress-s1"
    assert lora.sft_key("indicalign", stress.cfg) == "indicalign/raw"


def test_the_per_device_batch_must_divide_the_effective_batch(tiny_model_dir):
    default = plan(tiny_model_dir)
    assert (default.per_device_batch, default.grad_accum) == (1, 16)         # configs/train.yaml
    for k, accum in ((2, 8), (4, 4), (16, 1)):
        p = plan(tiny_model_dir, per_device_batch=k)
        assert (p.per_device_batch, p.grad_accum, p.effective_batch) == (k, accum, 16)
    for k in (0, 3, 5, 32):
        with pytest.raises(lora.TrainRefused, match="does not divide the effective batch 16"):
            plan(tiny_model_dir, per_device_batch=k)


def test_each_gpu_class_trains_in_its_one_precision(monkeypatch):
    def supported(dtype, device):           # lift.models.load.check_dtype_supported on the GPU simulated
        if dtype == "bf16" and lora._native_bf16(device) is None:
            raise ValueError("Tesla T4 has no native bf16 (compute capability 7.x)")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("lift.models.load.check_dtype_supported", supported)
    monkeypatch.setattr(lora, "_native_bf16", lambda device: "8.0")            # an A100
    lora._check_precision("bf16", "cuda:0", "bf16")
    for dtype in ("fp16", "fp32"):
        with pytest.raises(lora.TrainRefused, match="runs bf16 natively"):
            lora._check_precision(dtype, "cuda:0", "bf16")
    with pytest.raises(lora.TrainRefused, match="for a T4"):
        lora.check_parity_dtype("fp16", "cuda:0")
    lora.check_parity_dtype("fp32", "cuda:0")
    monkeypatch.setattr(lora, "_native_bf16", lambda device: None)             # a T4
    lora._check_precision("fp16", "cuda:0", "bf16")
    lora.check_parity_dtype("fp16", "cuda:0")
    with pytest.raises(lora.TrainRefused, match="no native bf16.*--dtype fp16"):
        lora._check_precision("bf16", "cuda:0", "bf16")
    with pytest.raises(lora.TrainRefused, match="Pass --dtype fp16"):
        lora._check_precision("fp32", "cuda:0", "bf16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(lora.TrainRefused, match="no CUDA GPU here"):
        lora._check_precision("bf16", "cuda:0", "bf16")
    lora._check_precision("fp32", "cpu", "bf16")                               # the tests' CPU


def test_seeds_set_the_data_seed_far_apart(tiny_model_dir):
    assert [plan(tiny_model_dir, seed=s).data_seed for s in (1, 2, 3)] == [1000, 2000, 3000]


def test_an_unpinned_global_is_refused_without_a_model_id():
    # The tests see every revision as "main" (conftest unpinned_revisions), as an unpinned project would.
    with pytest.raises(SystemExit, match="unpinned revisions"):
        lora.plan_training("aya", "standard", 1, device="cpu", dtype="fp32")


# --- data --------------------------------------------------------------------------------------------


def test_stale_or_malformed_training_files_are_refused():
    cfg = lora.recipe_config("standard")
    with pytest.raises(FileNotFoundError, match="lift data build --only sft"):
        lora.load_training_data("aya", cfg)
    write_sft(rows(4), planned=5)
    with pytest.raises(lora.TrainRefused, match="holds 4 rows, but report.json plans 5"):
        lora.load_training_data("aya", cfg)
    write_sft(rows(4), sha="0" * 64)
    with pytest.raises(lora.TrainRefused, match="stale copy"):
        lora.load_training_data("aya", cfg)
    with pytest.raises(lora.TrainRefused, match="plans no indicalign/raw"):
        lora.load_training_data("indicalign", lora.recipe_config("stress"))
    three = rows(2)
    three[1]["messages"].append({"role": "user", "content": "and more?"})
    write_sft(three)
    with pytest.raises(lora.TrainRefused, match="row aya-00001: messages must be exactly one user turn"):
        lora.load_training_data("aya", cfg)
    empty = rows(2)
    empty[0]["messages"][1]["content"] = "  "
    write_sft(empty)
    with pytest.raises(lora.TrainRefused, match="row aya-00000: a turn is empty"):
        lora.load_training_data("aya", cfg)
    twice = rows(2)
    twice[1]["id"] = twice[0]["id"]
    write_sft(twice)
    with pytest.raises(lora.TrainRefused, match="more than once"):
        lora.load_training_data("aya", cfg)
    path = write_sft(rows(3))
    data = lora.load_training_data("aya", cfg)
    assert (data.key, data.sha256, len(data.rows)) == ("aya/dedup", sha256_file(path), 3)
    assert data.repo_path == "data/sft/aya/dedup/sft_messages.jsonl"


def test_a_training_row_equal_to_a_held_out_prompt_is_refused(utility):
    assert lora.check_heldout(rows(3)) == 2
    stale = rows(3)
    stale[1]["messages"][0]["content"] = "  चाय कैसे बनती है?\n"                 # a dev prompt, stripped
    stale[2]["messages"][0]["content"] = "café ka matlab kya hai?"         # a final one, decomposed
    with pytest.raises(lora.TrainRefused, match=r"2 training rows .*\['aya-00001', 'aya-00002'\].*item 41"):
        lora.check_heldout(stale)


# --- examples ----------------------------------------------------------------------------------------


def test_only_the_answer_and_its_closing_tokens_are_supervised(tiny_tokenizer):
    row = rows(1)[0]
    user, answer = (m["content"] for m in row["messages"])
    example = lora.build_example(tiny_tokenizer, row, 1024)
    n = example["n_prompt"]
    assert example["input_ids"][:n] == encode_chat(tiny_tokenizer, user)    # BOS, preamble, user, gen. prompt
    prompt = tiny_tokenizer.decode(example["input_ids"][:n])
    assert "contextual safety mode" in prompt and user in prompt
    assert set(example["labels"][:n]) == {-100} and example["labels"][n:] == example["input_ids"][n:]
    assert tiny_tokenizer.decode(example["input_ids"][n:]) == answer + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    assert example["n_supervised"] == example["n_answer"] == len(example["input_ids"]) - n
    assert not example["truncated"] and example["id"] == row["id"]


def test_a_long_row_is_cut_from_the_start(tiny_tokenizer):
    row = rows(1)[0]
    full = lora.build_example(tiny_tokenizer, row, 4096)
    n = full["n_prompt"]
    cut = lora.build_example(tiny_tokenizer, row, n + 3)
    assert cut["truncated"] and cut["input_ids"] == full["input_ids"][:n + 3] and cut["n_supervised"] == 3
    assert cut["n_answer"] == full["n_answer"] and cut["labels"][n:] == full["input_ids"][n:n + 3]
    examples, unfit = lora.build_examples(tiny_tokenizer, [row, rows(1, start=1)[0]], n)
    assert [e["n_supervised"] for e in examples] == [0, 0]
    assert unfit[0] == {"id": row["id"], "prompt_tokens": n, "limit": n}
    stats = lora.data_stats([full, cut])
    assert stats["rows"] == 2 and stats["truncated_rows"] == 1 and stats["truncated_ids"] == [row["id"]]
    assert stats["supervised_tokens"] == full["n_supervised"] + 3 and stats["prompt_tokens"]["min"] == n


def test_a_row_left_without_an_answer_token_stops_the_run_before_the_model_loads(tiny_model_dir, utility,
                                                                                 monkeypatch):
    write_sft(rows(3) + [LONG])
    monkeypatch.setattr("lift.models.load.load_model", lambda *a, **k: pytest.fail("the model was loaded"))
    with pytest.raises(lora.TrainRefused) as refused:
        train(tiny_model_dir)
    message = str(refused.value)
    assert "aya-09999" in message and "limit 1024" in message and "prompt tokens" in message
    assert "max_seq_len" in message and "--drop-unfit" in message and "decisions.md" in message
    assert not variants.adapter_path(NAME).exists() and not manifests()


def test_drop_unfit_drops_exactly_those_rows_and_records_them(tiny_model_dir, utility):
    write_sft(rows(3) + [LONG])
    path = train(tiny_model_dir, drop_unfit=True, max_steps=1, parity=False)
    data = read(path / "lift_train.json")["data"]
    assert data["dropped_unfit"] == {"count": 1, "ids": ["aya-09999"]}
    assert (data["rows_in_file"], data["rows_trained"]) == (4, 3)
    [run] = manifests(name=NAME)
    assert run["dropped_unfit"] == {"count": 1, "ids": ["aya-09999"]} and run["n_rows_trained"] == 3
    reg = read(path / "registration.json")                   # parity=False: nothing to register on
    assert reg["entry"] is None and reg["registered"] is None and reg["parity"] == {}


# --- collator, heartbeat, data order -----------------------------------------------------------------


def test_the_collator_refuses_a_batch_without_supervised_tokens():
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    counter = lora.SupervisedTokenCounter(DataCollatorForLanguageModeling(pad_token_id=0))
    with pytest.raises(ValueError, match="a batch with zero supervised tokens"):
        counter([{"input_ids": [5, 6, 7], "labels": [-100, -100, -100]}])
    with pytest.raises(ValueError, match="a batch with zero supervised tokens"):   # position 0 is no target
        counter([{"input_ids": [5, 6, 7], "labels": [5, -100, -100]}])
    batch = counter([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                     {"input_ids": [5, 6], "labels": [-100, 6]}])
    assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 6, 0]]                  # right padding
    assert batch["labels"].tolist() == [[-100, 6, 7], [-100, 6, -100]]


def test_the_counter_books_micro_batches_to_their_optimizer_step():
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    counter = lora.SupervisedTokenCounter(DataCollatorForLanguageModeling(pad_token_id=0))
    for n in (2, 1, 3):                     # collated ahead of use, as accelerate's lookahead does
        counter([{"input_ids": [5] + [6] * n, "labels": [-100] + [6] * n}] * (4 - n))
    counter.consumed()
    assert counter.in_step == 1
    counter.step_end()                      # step 1: two micro-batches
    counter.step_end()                      # step 2: one
    assert counter.per_batch == [4, 3, 3] and counter.per_step == [7, 3] and counter.in_step == 0
    assert counter.rows_per_batch == [2, 3, 1] and counter.rows_per_step == [5, 1]
    with pytest.raises(RuntimeError, match="never built"):
        counter.consumed()


def test_the_heartbeat_logs_at_least_once_a_minute(caplog):
    now = [0.0]
    beat = lora.Heartbeat(NAME, every=60, clock=lambda: now[0])
    with caplog.at_level(logging.INFO, logger="lift.train"):
        beat.start(10)
        assert beat.beat(1, loss=2.0, lr=2e-4, epoch=0.1, force=True)
        now[0] = 30.0
        assert not beat.beat(2, loss=1.9)
        now[0] = 91.0
        assert beat.beat(2, micro=3)                     # mid-step, 61 s after the last line
        now[0] = 100.0
        assert not beat.beat(3, loss=1.8)
        assert beat.beat(10, loss=1.0, force=True)
    lines = [r.getMessage() for r in caplog.records if r.name == "lift.train"]
    assert len(lines) == 4 and beat.lines == 3
    assert lines[1].startswith(f"{NAME}: step 1/10, epoch 0.10, loss 2.0000, lr 0.0002, 0m00s elapsed")
    assert "step 2/10 under way (3 micro-batches done)" in lines[2] and "1m31s elapsed" in lines[2]
    assert "about 13m39s left" in lines[2] and "left" not in lines[3]


def test_seeds_change_the_data_order(tiny_model_dir, tiny_tokenizer, tmp_path):
    examples = [lora.build_example(tiny_tokenizer, row, 1024) for row in rows(12)]
    index = {tuple(e["input_ids"]): e["id"] for e in examples}

    def orders(seed: int, epochs: int = 1) -> list[list[str]]:
        p = plan(tiny_model_dir, seed=seed)
        model, _info = lora.lora_model(p)
        trainer, _counter = lora.make_trainer(model, tiny_tokenizer, examples, p,
                                              output_dir=tmp_path / f"s{seed}")
        loader = trainer.get_train_dataloader()
        return [[index[tuple(batch["input_ids"][0].tolist())] for batch in loader] for _ in range(epochs)]

    one, two, again = orders(1, epochs=2), orders(2), orders(1)
    assert sorted(one[0]) == sorted(index.values()) and sorted(one[1]) == sorted(index.values())
    assert one[0] != two[0] and one[0] == again[0]
    assert one[1] != two[0]         # data_seed = 1000 x seed: seed 1's 2nd epoch is not seed 2's 1st


@pytest.mark.parametrize("dtype, base", [("fp32", torch.float32), ("bf16", torch.bfloat16),
                                         ("fp16", torch.float16)])
def test_lora_wraps_every_projection_of_every_layer_in_fp32(tiny_model_dir, dtype, base):
    # On a bf16 (A100) or fp16 (T4) base the LoRA weights must still be fp32: the frozen base's own dtype
    # is checked too, so no case can fall back to an fp32 base, where fp32 LoRA weights prove nothing.
    p = plan(tiny_model_dir, dtype=dtype)
    model, info = lora.lora_model(p)
    assert info["layers"] == 4 and info["wrapped_modules"] == 4 * 7
    assert info["trainable_dtypes"] == ["float32"] and 0 < info["trainable_params"] < info["total_params"]
    assert {q.dtype for q in model.parameters() if not q.requires_grad} == {base}
    assert model.get_base_model().config.use_cache is False
    if dtype == "fp32":                         # the target_modules refusal does not depend on the dtype
        doubled = {**p.cfg["lora"], "target_modules": ["q_proj", "q_proj"]}
        twice = dataclasses.replace(p, cfg={**p.cfg, "lora": doubled})
        with pytest.raises(RuntimeError, match="exactly once"):
            lora.lora_model(twice)


def test_lora_weights_left_in_the_base_precision_stop_the_run(tiny_model_dir, monkeypatch):
    import peft

    real = peft.get_peft_model
    monkeypatch.setattr(peft, "get_peft_model",          # PEFT without its fp32 cast (TRL's ZeRO-3 path)
                        lambda model, cfg, **kw: real(model, cfg, **{**kw, "autocast_adapter_dtype": False}))
    with pytest.raises(RuntimeError, match=r"LoRA weights are \['bfloat16'\], not float32, on a bf16 base"):
        lora.lora_model(plan(tiny_model_dir, dtype="bf16"))


# --- one device, the recipe's batch ------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true:UserWarning")
def test_two_visible_gpus_train_on_one_at_the_recipe_batch(tiny_model_dir, tiny_tokenizer, tmp_path,
                                                           monkeypatch):
    # A plain process that sees two GPUs (a Kaggle T4 x2): TrainingArguments counts both (n_gpu 2), and the
    # Trainer would wrap the model in nn.DataParallel and take 2 x 16 = 32 rows an optimizer step. Faked
    # without a GPU: torch.cuda.device_count() says 2 while a cuda:0 plan's config and Trainer are built;
    # torch.cuda.is_available() stays False, so the model (loaded by a CPU plan) and every batch stay on the
    # CPU, where nn.DataParallel would pass the whole batch through.
    from transformers.training_args import ParallelMode

    cpu = plan(tiny_model_dir)
    cuda = dataclasses.replace(cpu, device="cuda:0")
    real_model, real_trainer, built = lora.lora_model, lora.make_trainer, {}

    def kept(*a, **kw):
        built["trainer"], counter = real_trainer(*a, **kw)
        return built["trainer"], counter

    monkeypatch.setattr(lora, "lora_model", lambda p: real_model(cpu))
    monkeypatch.setattr(lora, "make_trainer", kept)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    for stat in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, stat, lambda *a, **k: 0)
    examples = [lora.build_example(tiny_tokenizer, row, 1024) for row in rows(20)]
    fit = lora._fit(cuda, tiny_tokenizer, examples, tmp_path, {})
    trainer = built["trainer"]
    assert trainer.args.n_gpu == 1 and trainer.args.parallel_mode == ParallelMode.NOT_PARALLEL
    assert trainer._train_batch_size == 1 and trainer.get_total_train_batch_size(trainer.args) == 16
    assert not isinstance(trainer.model_wrapped, torch.nn.DataParallel)
    assert fit["batch"] == {"per_device": 1, "grad_accum": 16, "effective": 16, "n_gpu": 1, "visible_gpus": 2,
                            "processes": 1, "micro_batch": 1, "per_step": 16}
    # 2 epochs of 20 rows, 16 a step: each epoch a full step and one of the 4 rows left (n_gpu 2: one of 20)
    assert fit["global_step"] == 4 and fit["rows_per_step"] == [16, 4, 16, 4]
    assert len(fit["supervised_tokens_per_batch"]) == 40


def test_a_trainer_off_the_recipe_batch_is_refused_before_its_first_step(tiny_model_dir, tiny_tokenizer,
                                                                        tmp_path, monkeypatch):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    p = plan(tiny_model_dir)
    real_config = lora.sft_config

    def two_gpus(plan, output_dir):        # what a launcher or another transformers could leave: n_gpu 2
        args = real_config(plan, output_dir)
        args._n_gpu = 2
        return args

    monkeypatch.setattr(lora, "sft_config", two_gpus)
    model, _info = lora.lora_model(p)
    examples = [lora.build_example(tiny_tokenizer, row, 1024) for row in rows(4)]
    with pytest.raises(lora.TrainRefused, match=r"take 32 rows per optimizer step \(2 per micro-batch, 1 "
                                                r"process\(es\), n_gpu 2\), not the recipe's 16 .*torchrun.*"
                                                r"CUDA_VISIBLE_DEVICES=0"):
        lora.make_trainer(model, tiny_tokenizer, examples, p, output_dir=tmp_path / "refused")
    # Past that check, the collator still stops the first micro-batch of more rows than the recipe's.
    monkeypatch.setattr(lora, "check_batch", lambda plan, batch: None)
    trainer, counter = lora.make_trainer(model, tiny_tokenizer, examples, p, output_dir=tmp_path / "t")
    with pytest.raises(RuntimeError, match="put 2 rows in one micro-batch, but the recipe takes 1"):
        trainer.train()
    assert trainer.state.global_step == 0 and counter.per_batch == []
    counter = lora.SupervisedTokenCounter(DataCollatorForLanguageModeling(pad_token_id=0), max_rows=2)
    assert counter([{"input_ids": [5, 6], "labels": [-100, 6]}] * 2)["input_ids"].shape[0] == 2


# --- a whole run -------------------------------------------------------------------------------------


def test_a_five_step_run_trains_pushes_checks_parity_registers_and_resumes(
        tiny_model_dir, tiny_tokenizer, utility, hub, registry, monkeypatch):
    write_sft(rows(4))
    path = train(tiny_model_dir, max_steps=5)
    assert path == variants.adapter_path(NAME)
    files = {p.name for p in path.iterdir()}
    assert {"adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "lift_train.json",
            "trainer_log.json", "registration.json"} <= files and "README.md" not in files
    assert not (adapters_dir() / ".cache").exists() and not list(output_dir().glob("**/*.tmp"))
    digest = merge.adapter_hash(path)

    trained = read(path / "lift_train.json")
    assert trained["data"]["sha256"] == sha256_file(sft_dir() / "aya" / "dedup" / "sft_messages.jsonl")
    assert (trained["data"]["rows_trained"], trained["data_seed"], trained["dtype"]) == (4, 1000, "fp32")
    assert trained["lora"]["wrapped_modules"] == 28 and trained["train"]["global_step"] == 5
    # The plan's batch beside the Trainer's (on the CPU n_gpu is 0), and the rows each step took: one
    # epoch of the 4 rows a step (max_steps 5 runs 5 epochs).
    assert trained["batch"] == {"per_device": 1, "grad_accum": 16, "effective": 16, "n_gpu": 0,
                                "visible_gpus": torch.cuda.device_count(), "processes": 1, "micro_batch": 1,
                                "per_step": 16}
    assert trained["rows_per_step"] == [4] * 5
    log = read(path / "trainer_log.json")
    assert log["training_args"]["loss_type"] == "nll" and log["training_args"]["data_seed"] == 1000

    [run] = manifests(name=NAME)
    assert run["status"] == "ok" and run["devices"] == 1 and run["adapter_hash"] == digest
    assert run["batch"] == trained["batch"] and run["rows_per_step"] == [4] * 5
    assert [e["step"] for e in run["loss_curve"]] == [1, 2, 3, 4, 5]
    assert all("loss" in e for e in run["loss_curve"])
    # 4 rows, 1 per micro-batch: each step (one epoch with max_steps 5) sums every row's supervised tokens
    per_row = [lora.build_example(tiny_tokenizer, row, 1024)["n_supervised"] for row in rows(4)]
    assert run["supervised_tokens_per_step"] == [sum(per_row)] * 5
    assert sorted(run["supervised_tokens_per_batch"]) == sorted(per_row * 5)
    assert run["parity"]["dtype"] == "fp32" and run["parity"]["all_match"] and run["parity"]["n"] == 2
    assert hub.preflights == 1 and hub.pushes == [[f"outputs/adapters/{NAME}"]] * 2

    reg = read(path / "registration.json")
    assert reg["adapter_hash"] == digest and reg["registered"]["outcome"] == "added"
    assert reg["entry"]["location"] == f"runs repo ($LIFT_RUNS_REPO): outputs/adapters/{NAME}@{COMMIT}"
    assert "fp32 2/2 prompts token-identical" in reg["entry"]["note"]
    text = registry.read_text(encoding="utf-8")
    assert text.startswith(REGISTRY_TEXT)                                               # comments kept
    entry = next(e for e in yaml.safe_load(text)["variants"] if e["name"] == NAME)
    assert entry == reg["entry"] and entry["artifacts"] == {"adapter": digest} and entry["seed"] == 1
    spec = lora._registry_specs(yaml.safe_load(text), registry)[NAME]
    built = variants.build_variant(spec, device="cpu", dtype="fp32", model_id=str(tiny_model_dir))
    assert built.adapter_hash == digest                                     # the pinned hash loads

    # The same line again: no training, no second parity, the same entry, one more push.
    before = {p.name: sha256_file(p) for p in path.iterdir() if p.name != "registration.json"}
    monkeypatch.setattr(lora, "_fit", lambda *a, **k: pytest.fail("trained again"))
    monkeypatch.setattr(merge, "parity_check", lambda *a, **k: pytest.fail("parity checked again"))
    assert train(tiny_model_dir, max_steps=5) == path
    assert {p.name: sha256_file(p) for p in path.iterdir() if p.name != "registration.json"} == before
    [resumed] = manifests(name=NAME, resumed=True)
    assert resumed["status"] == "ok" and resumed["devices"] == 0 and resumed["parity"] is None
    reg = read(path / "registration.json")
    assert reg["registered"]["outcome"] == "unchanged" and len(reg["pushes"]) == 2
    assert registry.read_text(encoding="utf-8") == text and len(hub.pushes) == 4
    summary = "\n".join(lora.summary(path))
    assert digest in summary and "fp32 2/2" in summary and f"lift register-adapter {NAME}" in summary


def test_retrain_sets_the_old_adapter_aside_and_replaces_its_entry(tiny_model_dir, utility, registry):
    write_sft(rows(4))
    old = merge.adapter_hash(train(tiny_model_dir))
    path = train(tiny_model_dir, max_steps=3, retrain=True)
    new = merge.adapter_hash(path)
    assert new != old
    [aside] = list((output_dir() / ".aside" / "adapters").iterdir())
    assert aside.name.startswith(f"{NAME}-") and merge.adapter_hash(aside) == old
    assert read(path / "lift_train.json")["retrained_from"]["adapter_hash"] == old
    assert read(path / "registration.json")["registered"]["outcome"] == "replaced"
    listed = yaml.safe_load(registry.read_text(encoding="utf-8"))["variants"]
    assert [e["artifacts"]["adapter"] for e in listed if e["name"] == NAME] == [new]


def test_a_failed_fp32_parity_is_never_registered(tiny_model_dir, utility, registry, monkeypatch):
    write_sft(rows(4))
    real = merge.parity_check

    def diverging(*a, **kw):
        out = real(*a, **kw)
        return {**out, "n_match": out["n"] - 1, "all_match": False,
                "first_divergence": [3] + [None] * (out["n"] - 1)}

    monkeypatch.setattr(merge, "parity_check", diverging)
    before = registry.read_text(encoding="utf-8")
    with pytest.raises(lora.ParityFailed, match="NOT registered.*rule 9"):
        train(tiny_model_dir)
    path = variants.adapter_path(NAME)
    reg = read(path / "registration.json")
    assert reg["entry"] is None and reg["registered"] is None and reg["parity"]["fp32"]["all_match"] is False
    assert registry.read_text(encoding="utf-8") == before
    [run] = manifests(name=NAME)
    assert run["status"] == "parity_failed" and run["adapter_hash"] == merge.adapter_hash(path)
    # The rerun reuses the recorded parity and refuses again; so does the laptop.
    monkeypatch.setattr(merge, "parity_check", lambda *a, **k: pytest.fail("parity checked again"))
    with pytest.raises(lora.ParityFailed):
        train(tiny_model_dir)
    with pytest.raises(ValueError, match="FAILED"):
        lora.register_pulled(NAME)


def test_an_fp16_parity_registers_with_a_warning_whatever_its_share(tiny_model_dir, utility, registry,
                                                                    monkeypatch, caplog):
    # The T4 exception as the code applies it: the fp16 share is recorded, not required.
    write_sft(rows(4))

    def fp16_parity(adapter_dir, *, prompts, dtype, **kw):
        assert dtype == "fp16"
        return {"adapter_hash": merge.adapter_hash(adapter_dir), "dtype": dtype, "n": len(prompts),
                "n_match": 0, "match_share": 0.0, "all_match": False, "first_divergence": [1] * len(prompts)}

    monkeypatch.setattr(merge, "parity_check", fp16_parity)
    with caplog.at_level(logging.WARNING, logger="lift.train"):
        path = train(tiny_model_dir, parity_dtype="fp16")
    reg = read(path / "registration.json")
    assert set(reg["parity"]) == {"fp16"} and reg["parity"]["fp16"]["n_match"] == 0
    assert reg["registered"]["outcome"] == "added"
    assert "merge parity fp16 0/2 prompts token-identical (T4; fp32 not checked)" in reg["entry"]["note"]
    assert any("registered on an fp16 parity" in r.getMessage() for r in caplog.records)
    assert lora.register_pulled(NAME) == "unchanged"                     # the laptop applies the same rule


def test_the_t4_parity_exception_is_stated_as_the_code_applies_it(capsys):
    # lift.train.lora's docstring and lift train --help state the exception exactly: fp16 parity only on a
    # GPU without native bf16, its share recorded, the adapter registered with a warning whatever it is.
    with pytest.raises(SystemExit):
        main(["train", "--help"])
    for text in (" ".join(capsys.readouterr().out.split()), " ".join(lora.__doc__.split())):
        assert "The one exception is a GPU without native bf16 (a T4" in text
        assert "--parity-dtype fp16" in text and "records the share of prompts that match" in text
        assert "registers the adapter with a warning, whatever that share" in text


@pytest.fixture
def safety_prompts(fake_prompt_files):
    """Six dev prompts in every safety set and language: the pool Gate 0's parity sample is drawn from."""
    for set_name in SAFETY_SETS:
        for lang in LANGS:
            fake_prompt_files[(set_name, lang)] = [
                make_prompt(f"{set_name[:3]}-{i:04d}", lang, set_name, "dev", text=f"{set_name} {lang} {i}")
                for i in range(6)]
    return fake_prompt_files


def test_the_adapter_parity_runs_on_gate0s_parity_prompts(tiny_model_dir, safety_prompts, utility,
                                                          monkeypatch):
    # The prompts Gate 0's own random-LoRA parity used, as run_gate0 hands them to check_parity.
    from lift.prompts import mixed_sample

    handed = []

    def check_parity(model_id, revision, prompts, *, device, max_new_tokens=None):
        handed.append(list(prompts))
        return {"pass": True, "all_match": True}

    monkeypatch.setattr(gate0, "check_parity", check_parity)
    gate0.run_gate0(device="cpu", model_id=str(tiny_model_dir), revision="local", resolve_pins=False,
                    skip=["preamble", "precision", "throughput", "dry_run"])
    [drawn] = handed
    texts, info = lora._parity_texts(None)
    assert len(drawn) == gate0.N_PARITY == 20 and texts == [p.text for p in drawn]
    assert info["n"] == 20 and info["pids"] == [f"{p.set}_{p.lang}:{p.pid}" for p in drawn]
    assert info["source"] == ("Gate 0's merge-parity sample: lift.prompts.mixed_sample(20, split='dev', "
                              f"seed={gate0.SEED + 3})")
    assert texts != [p.text for p in mixed_sample(20, split="dev", seed=0)]    # parity_check's own default
    assert lora._parity_texts(PARITY) == (PARITY, {"source": "given", "n": 2,
                                                   "sha256": [sha256_text(t) for t in PARITY]})

    # A run without parity_prompts checks the adapter on those very prompts and records which they were.
    write_sft(rows(4))
    checked = []

    def parity_check(adapter_dir, *, prompts, dtype, **kw):
        checked.append(list(prompts))
        return {"adapter_hash": merge.adapter_hash(adapter_dir), "dtype": dtype, "n": len(prompts),
                "n_match": len(prompts), "match_share": 1.0, "all_match": True,
                "first_divergence": [None] * len(prompts)}

    monkeypatch.setattr(merge, "parity_check", parity_check)
    path = train(tiny_model_dir, parity_prompts=None, max_steps=1)
    assert checked == [texts]
    recorded = read(path / "registration.json")["parity"]["fp32"]["prompts"]
    assert recorded == {"source": info["source"], "n": 20, "pids": info["pids"]}


def test_a_failed_push_never_fails_training(tiny_model_dir, utility, hub):
    write_sft(rows(4))
    hub.fail = [sync.SyncError("the Hub answered HTTP 500; rerun"), OSError("connection reset")]
    path = train(tiny_model_dir)
    reg = read(path / "registration.json")
    first = reg["pushes"][0]
    assert (first["ok"], first["error"], first["permanent"]) == (False, "SyncError", False)
    assert reg["entry"]["location"] == f"runs repo ($LIFT_RUNS_REPO): outputs/adapters/{NAME}"   # no commit
    assert reg["registered"]["outcome"] == "added"
    [run] = manifests(name=NAME)
    assert run["status"] == "ok" and [p["error"] for p in run["push"]] == ["SyncError", "OSError"]


def test_a_push_that_cannot_work_is_refused_before_training(tiny_model_dir, utility, hub, monkeypatch):
    write_sft(rows(4))

    class Loaded(Exception):
        pass

    def loaded(*a, **k):
        raise Loaded

    monkeypatch.setattr("lift.models.load.load_tokenizer", loaded)
    hub.refuse = sync.SyncError("LIFT_RUNS_REPO is not set: export LIFT_RUNS_REPO=...", permanent=True)
    with pytest.raises(lora.TrainRefused, match="LIFT_RUNS_REPO is not set.*--no-push"):
        train(tiny_model_dir)
    hub.refuse = sync.SyncError("could not reach the Hub (ConnectError); check the connection and rerun")
    with pytest.raises(Loaded):                          # a transient preflight error lets the run go on
        train(tiny_model_dir)
    hub.refuse = sync.SyncError("never asked", permanent=True)
    with pytest.raises(Loaded):                          # --no-push: no preflight
        train(tiny_model_dir, push=False)
    assert hub.preflights == 2 and not hub.pushes


def test_seeds_of_a_recipe_train_on_one_gpu_class_and_precision(tiny_model_dir, utility):
    write_sft(rows(4))
    other = variants.adapter_path("adapter-aya-std-s2")
    other.mkdir(parents=True)
    (other / "lift_train.json").write_text(json.dumps({"dtype": "bf16", "gpu_capability": "8.0",
                                                       "gpu": "NVIDIA A100-SXM4-40GB"}), encoding="utf-8")
    with pytest.raises(lora.TrainRefused, match="adapter-aya-std-s2 was trained in bf16.*one GPU class"):
        train(tiny_model_dir)


def test_an_adapter_elsewhere_is_neither_pushed_nor_registered(tiny_model_dir, utility, hub, registry,
                                                                tmp_path):
    elsewhere = tmp_path / "elsewhere"
    with pytest.raises(lora.TrainRefused, match="push=False"):
        train(tiny_model_dir, out_dir=elsewhere)
    write_sft(rows(4))
    before = registry.read_text(encoding="utf-8")
    assert train(tiny_model_dir, out_dir=elsewhere, push=False, max_steps=1) == elsewhere
    assert (elsewhere / "adapter_model.safetensors").is_file() and not variants.adapter_path(NAME).exists()
    assert read(elsewhere / "registration.json")["registered"] is None
    assert registry.read_text(encoding="utf-8") == before and not hub.pushes


def test_a_folder_without_weights_is_refused(tiny_model_dir, utility):
    variants.adapter_path(NAME).mkdir(parents=True)
    with pytest.raises(lora.TrainRefused, match="--retrain"):
        train(tiny_model_dir)


# --- the registry ------------------------------------------------------------------------------------


def test_the_tests_registry_holds_stock_alone_whatever_the_real_one_holds(registry):
    # A copy of configs/variants.yaml would carry its entries in: once the pilot's adapter is registered
    # there (another hash than any test's), register_adapter would refuse every test's entry for it.
    text = registry.read_text(encoding="utf-8")
    assert text == REGISTRY_TEXT and list(lora._registry_specs(yaml.safe_load(text), registry)) == ["stock"]
    assert lora.default_registry() == registry != config.CONFIG_DIR / "variants.yaml"


def test_register_adapter_edits_the_text_once_and_refuses_another_hash(registry, monkeypatch):
    digest, other = "a" * 64, "b" * 64
    original = registry.read_text(encoding="utf-8")
    kw = {"candidate": "aya", "recipe": "standard", "seed": 1, "note": "aya, standard recipe",
          "location": f"runs repo ($LIFT_RUNS_REPO): outputs/adapters/{NAME}"}
    assert lora.register_adapter(NAME, digest, **kw) == "added"
    added = registry.read_text(encoding="utf-8")
    assert added.startswith(original) and added.endswith("\n")             # comments and stock kept verbatim
    assert lora.register_adapter(NAME, digest, **{**kw, "note": "another note"}) == "unchanged"
    assert registry.read_text(encoding="utf-8") == added
    with pytest.raises(lora.RegistryConflict, match="artifacts.*--replace"):
        lora.register_adapter(NAME, other, **kw)
    assert registry.read_text(encoding="utf-8") == added
    lora.register_adapter("adapter-indicalign-std-s1", digest, candidate="indicalign", recipe="standard",
                          seed=1, location="x", note="y")
    assert lora.register_adapter(NAME, other, replace=True, **kw) == "replaced"
    specs = lora._registry_specs(yaml.safe_load(registry.read_text(encoding="utf-8")), registry)
    assert list(specs) == ["stock", NAME, "adapter-indicalign-std-s1"]          # replaced in place
    assert specs[NAME].artifacts == {"adapter": other}
    assert specs["adapter-indicalign-std-s1"].artifacts == {"adapter": digest}
    real = config.load

    def served(name="project"):
        return yaml.safe_load(registry.read_text(encoding="utf-8")) if name == "variants" else real(name)

    monkeypatch.setattr(config, "load", served)
    assert variants.get_spec(NAME) == specs[NAME]                 # load_registry reads the text as written
    with pytest.raises(ValueError, match="not the adapter name"):
        lora.register_adapter("adapter-aya-std-s2", digest, **kw)
    with pytest.raises(ValueError, match="64 hex"):
        lora.register_adapter(NAME, "abc", **kw)


def test_register_adapter_refuses_a_layout_it_cannot_edit(registry):
    registry.write_text("variants: [{name: stock, method: stock}]\n", encoding="utf-8")
    kw = {"candidate": "aya", "recipe": "standard", "seed": 1, "location": "x", "note": "y"}
    with pytest.raises(ValueError, match="add the entry by hand"):
        lora.register_adapter(NAME, "a" * 64, **kw)
    assert registry.read_text(encoding="utf-8") == "variants: [{name: stock, method: stock}]\n"


def write_registration(directory, **changes):
    digest = merge.adapter_hash(directory)
    entry = lora.adapter_entry(NAME, digest, seed=1, note="aya, standard recipe, seed 1",
                               location=f"runs repo ($LIFT_RUNS_REPO): outputs/adapters/{NAME}@{COMMIT}")
    doc = {"name": NAME, "adapter_hash": digest, "candidate": "aya", "recipe": "standard", "seed": 1,
           "parity": {"fp32": {"n": 20, "n_match": 20, "all_match": True}}, "entry": entry, **changes}
    (directory / "registration.json").write_text(json.dumps(doc), encoding="utf-8")
    return entry


def test_register_adapter_on_the_laptop_checks_the_pulled_files(tiny_lora_adapter, registry, capsys):
    directory = variants.adapter_path(NAME)
    assert main(["register-adapter", NAME]) == 1 and "lift sync pull" in capsys.readouterr().err
    shutil.copytree(tiny_lora_adapter, directory)
    assert main(["register-adapter", NAME]) == 1 and "registration.json" in capsys.readouterr().err
    entry = write_registration(directory)
    assert main(["register-adapter", NAME]) == 0
    out = capsys.readouterr().out
    assert f"{NAME}: added" in out and "git add configs/variants.yaml" in out
    assert next(e for e in yaml.safe_load(registry.read_text(encoding="utf-8"))["variants"]
                if e["name"] == NAME) == entry
    assert main(["register-adapter", NAME]) == 0 and "unchanged" in capsys.readouterr().out
    write_registration(directory, adapter_hash="c" * 64)
    assert main(["register-adapter", NAME]) == 1
    assert "not the files the job checked" in capsys.readouterr().err
    write_registration(directory, parity={"fp32": {"n": 20, "n_match": 19, "all_match": False}})
    assert main(["register-adapter", NAME]) == 1 and "FAILED" in capsys.readouterr().err
    write_registration(directory, parity={"fp16": {"n": 20, "n_match": 17, "all_match": False}})
    assert main(["register-adapter", NAME]) == 0                             # a T4's fp16 parity registers


# --- the command -------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["train", "register-adapter"])
def test_help_works_without_a_gpu_stack(command, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0 and "usage" in capsys.readouterr().out


def test_lift_train_exits_2_on_a_refused_request_and_stops_on_unpinned_revisions(capsys):
    assert main(["train", "--candidate", "aya", "--recipe", "stress", "--seed", "1", "--device", "cpu"]) == 2
    assert "refused" in capsys.readouterr().err
    assert main(["train", "--candidate", "aya", "--recipe", "standard", "--seed", "1", "--device", "cpu",
                 "--per-device-batch", "3"]) == 2
    with pytest.raises(SystemExit, match="unpinned revisions"):
        main(["train", "--candidate", "aya", "--recipe", "standard", "--seed", "1", "--device", "cpu"])


# --- lift train --check ------------------------------------------------------------------------------


def snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its sha256: what a step that writes nothing leaves as it was."""
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_check_builds_every_example_as_training_does_and_writes_nothing(tiny_model_dir, tiny_tokenizer,
                                                                        utility, hub, registry,
                                                                        isolated_dirs, monkeypatch):
    write_sft(rows(3) + [LONG])
    before = snapshot(isolated_dirs), registry.read_text(encoding="utf-8")
    with monkeypatch.context() as m:
        m.setattr("lift.models.load.load_model", lambda *a, **k: pytest.fail("a model was loaded"))
        m.setattr(lora, "register_adapter", lambda *a, **k: pytest.fail("the registry was written"))
        m.setattr(torch.cuda, "is_available", lambda: False)          # no GPU here, and none is asked for
        report = lora.check_training("aya", "standard", 1, model_id=str(tiny_model_dir))     # device cuda:0
        dropped = lora.check_training("aya", "standard", 1, model_id=str(tiny_model_dir), drop_unfit=True)
        with pytest.raises(lora.TrainRefused, match="no CUDA GPU here"):     # the same request, training
            lora.train_lora("aya", "standard", 1, model_id=str(tiny_model_dir), push=False)
    assert (snapshot(isolated_dirs), registry.read_text(encoding="utf-8")) == before
    assert not hub.pushes and hub.preflights == 0 and not manifests()

    examples = [lora.build_example(tiny_tokenizer, row, 1024) for row in rows(3) + [LONG]]
    assert report["counts"] == lora.data_stats(examples)
    assert (report["counts"]["rows"], report["counts"]["truncated_rows"]) == (4, 1)
    assert report["unfit"] == [{"id": "aya-09999", "prompt_tokens": examples[3]["n_prompt"], "limit": 1024}]
    assert report["stops"] and report["kept"] == lora.data_stats(examples[:3])
    assert report["data"]["sha256"] == sha256_file(sft_dir() / "aya" / "dedup" / "sft_messages.jsonl")
    assert report["data"]["heldout_prompts_checked"] == 2 and report["model"]["revision"].startswith("local-")
    out = "\n".join(lora.check_lines(report))
    assert "rows: 4; truncated at 1024 tokens: 1" in out
    assert f"    aya-09999, {examples[3]['n_prompt']}, 1024" in out
    assert "lift train stops on these rows" in out and "--drop-unfit" in out
    assert f"supervised tokens: {report['counts']['supervised_tokens']} (per row: min 0," in out
    assert "Question 0" not in out and "x x x" not in out                 # ids and counts, never text

    # --drop-unfit: what the check says training takes is what training records it took.
    assert not dropped["stops"] and dropped["kept"] == report["kept"]
    assert "lift train --drop-unfit drops exactly these and trains on 3 rows" in "\n".join(
        lora.check_lines(dropped))
    path = train(tiny_model_dir, drop_unfit=True, max_steps=1, parity=False)
    assert read(path / "lift_train.json")["counts"] == report["kept"]


def test_lift_train_check_reads_the_pinned_tokenizer_and_exits_0(tiny_tokenizer, utility, hub, monkeypatch,
                                                                 capsys):
    write_sft(rows(3) + [LONG])
    line = ["train", "--candidate", "aya", "--recipe", "standard", "--seed", "1", "--check"]
    loaded = []

    def load_tokenizer(model_id, revision):
        loaded.append((model_id, revision))
        return tiny_tokenizer

    monkeypatch.setattr("lift.models.load.load_tokenizer", load_tokenizer)
    monkeypatch.setattr("lift.models.load.load_model", lambda *a, **k: pytest.fail("a model was loaded"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="unpinned revisions"):    # the tests see models.global at "main"
        main(line)
    assert not loaded
    config.load("project")["models"]["global"]["revision"] = SHA   # pinned, as configs/project.yaml is now
    assert main(line) == 0
    out = capsys.readouterr().out
    assert loaded == [(config.get("models.global.id"), SHA)]
    assert f"tokenizer: {config.get('models.global.id')}@{SHA[:12]}; max_seq_len 1024" in out
    assert "aya-09999, " in out and "lift train stops on these rows" in out
    assert main([*line, "--drop-unfit"]) == 0
    assert "lift train --drop-unfit drops exactly these and trains on 3 rows" in capsys.readouterr().out
    assert main([*line, "--per-device-batch", "3"]) == 2
    assert "lift train --check: refused" in capsys.readouterr().err
    assert not hub.pushes and hub.preflights == 0 and not manifests()
    assert main(line[:-1]) == 2 and "no CUDA GPU here" in capsys.readouterr().err   # training: refused
