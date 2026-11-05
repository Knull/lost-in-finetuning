"""Hindi LoRA SFT on Tiny Aya Global: ``lift train`` and ``lift register-adapter``. Implemented in P03.

``train_lora(candidate, recipe, seed)`` trains ``adapter-{candidate}-{std|stress}-s{seed}`` into
outputs/adapters/{name}/, pushes it, checks merge parity and registers it. The rules, and why:

Recipe and batch
- configs/train.yaml, with ``stress.inherit: standard`` resolved by a shallow merge: stress overrides
  lr, epochs and dedup and inherits the rest (the lora block whole). The stress recipe trains only
  ``stress.candidate`` (indicalign, decisions.md item 47), and a seed must be one of the recipe's seeds.
- The effective batch is the recipe's per_device_batch x grad_accum (16). ``--per-device-batch K`` must
  divide it and sets grad_accum to the quotient (A100: 4 x 4; T4: 1 x 16), so every GPU takes the same
  optimizer steps. max_seq_len is never below 512: the ~366-token preamble counts against it.
- One process trains on one device: cuda:0, the first visible GPU, or the CPU (the tests). Run as a
  plain process with several GPUs visible (a Kaggle T4 x2), the Trainer would count them all (n_gpu 2),
  wrap the model in nn.DataParallel and put per_device_batch x n_gpu rows in every micro-batch: 32 rows
  an optimizer step instead of 16, half the steps. ``sft_config`` therefore holds the Trainer to one GPU
  (n_gpu 1, what the Trainer itself does for a model-parallel model) before the Trainer is built, which
  takes its DataLoader batch from it; the other GPUs stay idle. The built Trainer's rows per optimizer
  step (its DataLoader batch x accumulation x processes) must then equal the effective batch, or the run
  is refused before its first step (a torchrun or accelerate launch, say), and the collator stops any
  micro-batch with more rows than per_device_batch. lift_train.json and the manifest record n_gpu, the
  visible GPUs, the Trainer's batch and the rows each optimizer step took.

Data
- data/sft/{candidate}/{dedup|raw}/sft_messages.jsonl (recipe dedup true: dedup), found through
  data/sft/report.json, never by listing folders: a folder left by an earlier rule can sit beside the
  planned ones. Its row count must equal report.json's ``decision.rows_by_variant`` entry and its sha256
  the one the report recorded, or the file is stale and refused. Every row is exactly one user turn then
  one assistant turn, both non-empty strings, under a unique id. No user turn may equal (NFC, stripped) a
  held-out utility prompt of any split (decisions.md item 41). The build already sampled and capped the
  rows: all of them train, and the seed only orders them.

Chat template on, loss on the answer only
- Each row is ``lift.models.load.encode_chat_turn``: the prompt ids are exactly what generation feeds the
  model (BOS, the default preamble, the user turn, the generation prompt) and get label -100; the answer
  and the tokens that close the turn are supervised, so the model learns to stop where the harness
  stops, and the held-out loss (lift.models.loglik) scores exactly these tokens. A row longer than
  max_seq_len is cut keeping its start, and counted as truncated.
- Every row is tokenized before the model loads. A row whose prompt alone reaches max_seq_len (a long
  IndicAlign user turn after the preamble) has no answer token left to learn from: the run stops and
  lists such rows (id, prompt tokens, limit). The remedies, each with a dated docs/decisions.md entry:
  raise max_seq_len, or drop the rows with ``--drop-unfit``, which drops exactly those rows and records
  their ids (never the default).
- ``check_training`` (``lift train --check``) shows this before a GPU run: it builds every example the
  way training does (``prepare``, one code path) with Global's tokenizer the only thing loaded (no model,
  no GPU, no push, no registry), reports the counts and the rows left without a supervised token, and
  writes nothing.
- TRL gets the rows pre-tokenized (input_ids, labels) with ``skip_prepare_dataset``: its own preparation
  would cut them again and silently drop rows left without supervised tokens. ``loss_type="nll"``: TRL's
  default chunked loss patches the model's forward. The collator is TRL's, wrapped so every micro-batch's
  supervised tokens (labels[:, 1:] != -100, the positions the shifted causal-LM loss scores) are counted,
  per micro-batch and per optimizer step, and a micro-batch with none stops the run.

Seeds and precision
- ``transformers.set_seed(seed)`` runs before the LoRA weights are made; the Trainer seeds dropout with
  ``seed``; accelerate orders epoch e of the data with ``data_seed + e``. data_seed is 1000 x seed, so
  the seeds never share an epoch's order (data_seed = seed would give seed 1's second epoch the order of
  seed 2's first).
- bf16 weights and bf16 autocast on a GPU with native bf16 (A100, L4), where configs/train.yaml's
  precision is the only one allowed, as for generation (decisions.md 27); fp16 weights and fp16
  autocast on a T4, which has no bf16; fp32 on CPU (the tests). PEFT keeps the LoRA weights in fp32 (its
  autocast_adapter_dtype), and the run stops before training if they are not: fp16 autocast cannot
  unscale fp16 gradients, and Adam's updates lose precision in bf16.
- All seeds of a recipe train on one GPU class and precision (the P03 plan): a seed is refused when
  another seed's adapter here was trained on another (its lift_train.json says so).

Files
- Nothing synced is written while training: the Trainer's output_dir and the staging folder are a
  temporary folder outside outputs/, and nothing is checkpointed. The adapter is saved there (PEFT: the
  LoRA weights as safetensors and adapter_config.json; PEFT's boilerplate README is not kept, P08 writes
  the model cards) with Global's tokenizer, trainer_log.json (the Trainer's log history, its arguments)
  and lift_train.json (recipe, data file and hash, counts, LoRA, precision, GPU, versions). It is then
  copied under outputs/adapters/.cache/ (never synced, on the same disk) and renamed to
  outputs/adapters/{name}/, which therefore holds every file or does not exist.
- The adapter hash is lift.train.merge.adapter_hash (the weights plus the behaviour fields of the
  config), the value every cache key built on the adapter carries.

After training
- Push: the adapter folder goes to the runs repo at once (lift.sync.push), so a later crash loses
  nothing. LIFT_RUNS_REPO and the token are checked (lift.sync.preflight) before any GPU time. A failed
  push is logged and recorded and never fails the run: the runners' own pushes send the folder again
  (sync compares content).
- Parity: the training model is freed, then lift.train.merge.parity_check compares the merged and the
  unmerged adapter's greedy outputs, on the base it was trained on, on 20 dev prompts drawn by Gate 0's
  parity rule (``gate0_parity_sample``: ``mixed_sample(gate0.N_PARITY, split="dev",
  seed=gate0.SEED + 3)``, mixed over sets and languages) from the dev prompt files as they are now.
  They are the prompts Gate 0's random-LoRA parity ran only while those dev pools are unchanged since
  Gate 0, which records no pids: a spot-check exclusion or a fixed translation changes the draw.
  registration.json records the rule and the pids drawn.
  In fp32 they must match token for token, or the adapter is not registered and the run fails: it is
  saved and pushed, but nothing may evaluate it until the cause is understood (CLAUDE.md rule 9).
  lift.models.variants.checked_adapter, the one path every adapter variant is built through, refuses an
  adapter whose registration.json records a failed fp32 parity for its hash, whatever a registry holds.
- The one exception is a GPU without native bf16 (a T4, where the fp32 base, about 13.4 GB, may not
  fit): there ``--parity-dtype fp16`` checks parity in fp16, records the share of prompts that match
  (registration.json, and the registry note "fp16 n/20 prompts token-identical (T4; fp32 not
  checked)") and registers the adapter with a warning, whatever that share: fp16 rounds the merged
  weights differently, so an exact match is not expected. fp16 parity is refused on a GPU with native
  bf16 (an A100, an L4) and allowed on the CPU (the tests). An fp32 parity recorded later for the same
  adapter hash (a resume where the fp32 base fits) decides instead, and ``lift register-adapter``
  applies the same rule on the laptop. When that fp32 parity fails, the adapter is un-registered:
  registration.json drops its entry and its registered block (kept as ``unregistered``, with the reason
  and the time), the run removes the entry from the registry it writes when that entry pins this hash
  (``unregister_adapter``) and fails (ParityFailed), and on the laptop ``lift register-adapter NAME``
  removes the committed entry the same way ("removed"; commit configs/variants.yaml).
- Registration: an entry appended to configs/variants.yaml as text, so its comments stay: name, method
  adapter, the adapter, the seed, artifacts.adapter = the hash, location (the runs-repo path and push
  commit, never the repo id: LIFT_RUNS_REPO is a secret) and a note (candidate, recipe, date, GPU,
  precision, parity). The same entry again is a no-op; another one under the name is refused unless
  replacing (``--retrain``, ``lift register-adapter --replace``), since every output made from the old
  entry has other cache keys and goes stale. outputs/adapters/{name}/registration.json keeps the hash,
  the parity results, the pushes made before it and the entry, and is pushed too. configs/ travels by
  git, not by lift sync, so a GPU runtime's registration dies with the session: the laptop runs ``lift
  sync pull``, then ``lift register-adapter NAME`` (``register_pulled``: it recomputes the hash from the
  pulled files, checks it against registration.json and that parity passed, writes the same entry, or
  removes the entry when the fp32 parity failed), and commits configs/variants.yaml.
- Resume: an adapter folder that holds the weights is never trained again. The run does what is
  missing (the push; parity, unless registration.json holds one for this hash in that precision; the
  registration) and exits 0. ``--retrain`` first sets the folder aside under
  outputs/.aside/adapters/{name}-{utc stamp} (never synced), then trains anew.
- One manifest per run (stage "train"): what was trained, on what, the batch (the plan's and the
  Trainer's: n_gpu, visible GPUs, rows per micro-batch and per optimizer step) and the rows each step
  took, the supervised tokens per micro-batch and per optimizer step, the loss curve, the training time,
  peak GPU memory, the hash, parity, pushes and registration; status "failed" when the run stops on an
  error.

Logs carry counts, ids, hashes and paths, never row or output text. While training, a heartbeat line
(step, loss, elapsed) comes at least once a minute: the Colab and Kaggle runners kill a line silent for
45 minutes, and a T4 optimizer step can take minutes.

On Kaggle/Colab, ``pip uninstall -y torchao`` before importing SafeTune or TRL if imports fail.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lift import config
from lift.io import read_jsonl, sha256_file, sha256_text
from lift.models.load import DTYPES
from lift.paths import CONFIG_DIR, REPO_ROOT, ensure, output_dir, sft_dir

log = logging.getLogger("lift.train")

RECIPES = ("standard", "stress")
RECIPE_TAGS = {"standard": "std", "stress": "stress"}      # adapter names say std (configs/variants.yaml)
PARITY_DTYPES = ("fp32", "fp16")
MIN_SEQ_LEN = 512
DATA_SEED_FACTOR = 1000
GATE0_PARITY_SEED_OFFSET = 3      # lift.gate0.run_gate0 draws its parity prompts with seed SEED + 3
HEARTBEAT_SECONDS = 60.0
IGNORE_INDEX = -100
LIFT_TRAIN = "lift_train.json"
TRAINER_LOG = "trainer_log.json"
REGISTRATION = "registration.json"
RECORD_VERSION = 1
_RECIPE_KEYS = ("lora", "lr", "epochs", "max_seq_len", "per_device_batch", "grad_accum", "warmup_ratio",
                "scheduler", "gradient_checkpointing", "precision", "loss", "dedup", "seeds")
_LORA_KEYS = ("r", "alpha", "dropout", "target_modules")
_CURVE_KEYS = ("step", "epoch", "loss", "grad_norm", "learning_rate", "entropy", "mean_token_accuracy",
               "num_tokens")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# A registry list item that starts with its name: "  - name: adapter-aya-std-s1".
_ITEM = re.compile(r"^(?P<indent>[ ]*)-[ ]+name:[ ]*(?P<name>\S+?)[ ]*$")


class TrainRefused(ValueError):
    """A request the recipe, the data or the hardware rules refuse, before any GPU time is spent (a
    candidate or seed the recipe does not allow, a batch that does not divide the effective batch, a
    precision the GPU must not use, a stale or malformed training file, rows the template leaves nothing
    to learn from, a push that cannot work), or, once the model is loaded, before the first training
    step (a Trainer that would not take the recipe's batch). The message names the remedy; ``lift
    train`` exits 2."""


class ParityFailed(RuntimeError):
    """The merged adapter's greedy outputs differ from the unmerged adapter's in fp32. The adapter is
    saved and pushed and registration.json records the parity, but it is not registered (a registration
    an earlier fp16 parity made is withdrawn, and the registry this run writes loses an entry pinning
    this hash) and must not be evaluated until the cause is understood (CLAUDE.md rule 9: build_variant
    refuses it). ``lift train`` exits 1."""


class RegistryConflict(ValueError):
    """The registry already holds another entry under the adapter's name. Replacing it changes the cache
    key of every output made from the old entry, so it happens only when asked for (``replace``)."""


# --- recipe -------------------------------------------------------------------------------------------


def recipe_config(recipe: str) -> dict[str, Any]:
    """configs/train.yaml's block for ``recipe`` with ``inherit`` resolved (a shallow merge: the child's
    keys win, a nested block such as ``lora`` is inherited whole), checked."""
    if recipe not in RECIPES:
        raise TrainRefused(f"unknown recipe {recipe!r}; choose from {list(RECIPES)}")
    cfg = copy.deepcopy(_resolved(config.load("train"), recipe, ()))    # never the cached config's own dicts
    where = f"configs/train.yaml's {recipe} recipe"
    missing = [k for k in _RECIPE_KEYS if k not in cfg]
    lora = cfg.get("lora") if isinstance(cfg.get("lora"), dict) else {}
    missing += [f"lora.{k}" for k in _LORA_KEYS if k not in lora]
    if missing:
        raise TrainRefused(f"{where} lacks {missing}")
    if cfg["loss"] != "assistant_only":
        raise TrainRefused(f"{where} asks for loss {cfg['loss']!r}; lift.train.lora implements "
                           "assistant_only (the answer and its closing tokens)")
    if int(cfg["max_seq_len"]) < MIN_SEQ_LEN:
        raise TrainRefused(f"{where} sets max_seq_len {cfg['max_seq_len']}, under {MIN_SEQ_LEN}: the "
                           "~366-token preamble would leave too little room (CLAUDE.md)")
    if not 0 <= float(cfg["warmup_ratio"]) < 1:
        raise TrainRefused(f"{where} sets warmup_ratio {cfg['warmup_ratio']}; a ratio is in [0, 1)")
    if cfg["precision"] not in DTYPES:
        raise TrainRefused(f"{where} sets precision {cfg['precision']!r}; choose from {list(DTYPES)}")
    return cfg


def _resolved(doc: Mapping[str, Any], name: str, seen: tuple[str, ...]) -> dict[str, Any]:
    if name in seen:
        raise TrainRefused(f"configs/train.yaml: the recipes inherit in a cycle "
                           f"({' -> '.join((*seen, name))})")
    block = doc.get(name)
    if not isinstance(block, dict):
        raise TrainRefused(f"configs/train.yaml has no {name!r} recipe block")
    block = dict(block)
    parent = block.pop("inherit", None)
    return block if parent is None else {**_resolved(doc, str(parent), (*seen, name)), **block}


def adapter_name(candidate: str, recipe: str, seed: int) -> str:
    """``adapter-{candidate}-{std|stress}-s{seed}`` (the naming scheme in configs/variants.yaml)."""
    if recipe not in RECIPE_TAGS:
        raise TrainRefused(f"unknown recipe {recipe!r}; choose from {list(RECIPES)}")
    return f"adapter-{candidate}-{RECIPE_TAGS[recipe]}-s{int(seed)}"


@dataclass(frozen=True)
class Plan:
    """Everything a run decides before it loads anything."""

    candidate: str
    recipe: str
    seed: int
    cfg: dict[str, Any]                 # the recipe, inherit resolved
    model_id: str
    model_rev: str
    device: str
    dtype: str
    per_device_batch: int
    grad_accum: int
    max_steps: int | None = None

    @property
    def name(self) -> str:
        return adapter_name(self.candidate, self.recipe, self.seed)

    @property
    def data_seed(self) -> int:
        return DATA_SEED_FACTOR * self.seed

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum


def plan_training(candidate: str, recipe: str, seed: int, *, model_id: str | None = None,
                  revision: str | None = None, device: str = "cuda:0", dtype: str | None = None,
                  per_device_batch: int | None = None, max_steps: int | None = None,
                  hardware: bool = True) -> Plan:
    """Check a request against the recipe and the hardware rules; nothing is loaded. ``model_id`` and
    ``revision`` default to models.global, whose revision must then be pinned; a caller passing
    ``model_id`` (the tests: a local tiny model) vouches for it. ``hardware=False`` leaves out the
    checks that ask the GPU (that it exists, and the precision it may train in), for ``check_training``,
    which runs where there is none."""
    cfg = recipe_config(recipe)
    candidates = list(config.load("train").get("candidates") or [])
    if candidate not in candidates:
        raise TrainRefused(f"unknown candidate {candidate!r}; configs/train.yaml trains {candidates}")
    only = cfg.get("candidate")
    if only is not None and candidate != only:
        raise TrainRefused(f"the {recipe} recipe trains only {only!r} (configs/train.yaml "
                           f"{recipe}.candidate, decisions.md item 47), not {candidate!r}")
    seeds = [int(s) for s in cfg["seeds"]]
    if int(seed) not in seeds:
        raise TrainRefused(f"seed {seed} is not one of the {recipe} recipe's seeds {seeds} "
                           "(configs/train.yaml)")
    if max_steps is not None and int(max_steps) < 1:
        raise TrainRefused(f"max_steps must be at least 1, got {max_steps}")
    per_device, accum = _batch(cfg, per_device_batch)
    device = _device(device)
    dtype = dtype or str(cfg["precision"])
    if dtype not in DTYPES:
        raise TrainRefused(f"--dtype must be one of {list(DTYPES)}, got {dtype!r}")
    if model_id is None:
        from lift.commands.run import require_pinned

        require_pinned("models.global")
        model_id = str(config.get("models.global.id"))
    from lift.models.load import resolved_revision

    rev = resolved_revision(model_id, revision or config.get("models.global.revision"))
    if hardware:
        _check_precision(dtype, device, str(cfg["precision"]))
    return Plan(candidate=candidate, recipe=recipe, seed=int(seed), cfg=cfg, model_id=model_id,
                model_rev=rev, device=device, dtype=dtype, per_device_batch=per_device, grad_accum=accum,
                max_steps=None if max_steps is None else int(max_steps))


def _batch(cfg: Mapping[str, Any], per_device_batch: int | None) -> tuple[int, int]:
    """(per-device batch, gradient accumulation) keeping the recipe's effective batch."""
    effective = int(cfg["per_device_batch"]) * int(cfg["grad_accum"])
    k = int(cfg["per_device_batch"]) if per_device_batch is None else int(per_device_batch)
    if k < 1 or effective % k:
        fits = [d for d in range(1, effective + 1) if effective % d == 0]
        raise TrainRefused(f"--per-device-batch {k} does not divide the effective batch {effective} "
                           f"(configs/train.yaml per_device_batch x grad_accum); choose from {fits}")
    return k, effective // k


def _device(device: str) -> str:
    name = str(device).strip()
    name = "cuda:0" if name == "cuda" else name
    if name not in ("cpu", "cuda:0"):
        raise TrainRefused(f"lift train runs on one device, cuda:0 (or cpu, for the tests), not {device!r}: "
                           "it holds the Trainer to the first visible GPU, however many are visible. To "
                           "train on another GPU, make it the first visible one (CUDA_VISIBLE_DEVICES)")
    return name


def _native_bf16(device: str) -> str | None:
    from lift.commands.run import _native_bf16 as native

    return native(device)


def _check_precision(dtype: str, device: str, precision: str) -> None:
    """Refuse a GPU that is missing; on a GPU that runs bf16 natively, any precision but the recipe's
    (decisions.md 27); on one that does not (a T4), any but fp16 (bf16 is only emulated there, and fp32
    weights do not fit beside training). A CPU (the tests) is never refused."""
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise TrainRefused(f"no CUDA GPU here for --device {device}: lift train runs on a GPU runtime "
                           "(lift remote, or the Kaggle runner); --device cpu is for the tests' tiny model")
    from lift.models.load import check_dtype_supported

    try:
        check_dtype_supported(dtype, device)
    except ValueError as exc:
        raise TrainRefused(f"{exc} (lift train --dtype fp16 on this GPU)") from exc
    capability = _native_bf16(device)
    if capability and dtype != precision:
        raise TrainRefused(f"--dtype {dtype} on {device} (compute capability {capability}) is refused: this "
                           f"GPU runs {precision} natively, the training precision in configs/train.yaml. "
                           "Another precision exists only for a T4 (decisions.md 27): train there, or here "
                           "without --dtype")
    if not capability and dtype != "fp16":
        raise TrainRefused(f"--dtype {dtype} on {device}, a GPU without native bf16 (a T4), is refused: it "
                           "trains in fp16 (fp16 weights and autocast, the LoRA weights in fp32). Pass "
                           "--dtype fp16")


def check_parity_dtype(parity_dtype: str, device: str) -> None:
    """fp32 parity everywhere; fp16 only on a GPU without native bf16 (a T4, where the fp32 base may not
    fit beside the adapter) or the CPU."""
    if parity_dtype not in PARITY_DTYPES:
        raise TrainRefused(f"--parity-dtype must be one of {list(PARITY_DTYPES)}, got {parity_dtype!r}")
    if parity_dtype == "fp16" and str(device).startswith("cuda"):
        capability = _native_bf16(device)
        if capability:
            raise TrainRefused(f"--parity-dtype fp16 is for a T4, where the fp32 base (about 13.4 GB) may "
                               f"not fit: {device} (compute capability {capability}) runs the fp32 parity, "
                               "which must match token for token. Leave --parity-dtype out")


# --- data ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingData:
    """One SFT variant's rows, checked against data/sft/report.json."""

    key: str                         # "{candidate}/{dedup|raw}", report.json's name for the file
    path: Path
    sha256: str
    rows: list[dict[str, Any]]

    @property
    def repo_path(self) -> str:
        return f"data/sft/{self.key}/sft_messages.jsonl"


def sft_key(candidate: str, cfg: Mapping[str, Any]) -> str:
    return f"{candidate}/{'dedup' if cfg['dedup'] else 'raw'}"


def load_training_data(candidate: str, cfg: Mapping[str, Any]) -> TrainingData:
    """The recipe's training file for ``candidate``, refused unless it is the one report.json planned
    (its row count and sha256) and every row is one non-empty user turn then one assistant turn."""
    key = sft_key(candidate, cfg)
    report_path = sft_dir() / "report.json"
    rebuild = "rebuild the SFT files with `lift data build --only sft`"
    if not report_path.is_file():
        raise FileNotFoundError(f"{report_path} not found: the training files are built by `lift data build "
                                "--only sft` and reach a GPU job through `lift sync pull`")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    planned = (report.get("decision") or {}).get("rows_by_variant") or {}
    if key not in planned:
        raise TrainRefused(f"{report_path} plans no {key} training file (it plans {sorted(planned)}): "
                           f"{rebuild}")
    recorded = ((report.get("variants") or {}).get(key) or {}).get("sha256")
    if not recorded:
        raise TrainRefused(f"{report_path} records no sha256 for {key}: {rebuild}")
    path = sft_dir() / key / "sft_messages.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `lift sync pull` (or `lift data build --only sft`)")
    digest = sha256_file(path)
    if digest != recorded:
        raise TrainRefused(f"{path} (sha256 {digest[:12]}) is not the file {report_path.name} recorded "
                           f"(sha256 {recorded[:12]}): a stale copy. Run `lift sync pull`, or {rebuild}")
    rows = list(read_jsonl(path))
    ids = [row_turns(row, f"{path.name} line {i}")[0] for i, row in enumerate(rows, 1)]
    twice = sorted(i for i, n in Counter(ids).items() if n > 1)
    if twice:
        raise TrainRefused(f"{path} holds {len(twice)} ids more than once, e.g. {twice[:5]}: {rebuild}")
    if len(rows) != int(planned[key]):
        raise TrainRefused(f"{path} holds {len(rows)} rows, but {report_path.name} plans {planned[key]} for "
                           f"{key}: a stale file. Run `lift sync pull`, or {rebuild}")
    return TrainingData(key=key, path=path, sha256=digest, rows=rows)


def row_turns(row: Any, where: str = "a row") -> tuple[str, str, str]:
    """(id, user text, assistant text) of one training row. Raises naming the row unless it is exactly
    one non-empty user turn then one non-empty assistant turn under a string id."""
    row_id = row.get("id") if isinstance(row, dict) else None
    if not isinstance(row_id, str) or not row_id:
        raise TrainRefused(f"{where}: a training row needs a string id")
    messages = row.get("messages")
    roles = ([m.get("role") if isinstance(m, dict) else None for m in messages]
             if isinstance(messages, list) else None)
    if roles != ["user", "assistant"]:
        raise TrainRefused(f"row {row_id}: messages must be exactly one user turn then one assistant turn, "
                           f"got roles {roles}")
    texts = [m.get("content") for m in messages]
    if not all(isinstance(t, str) and t.strip() for t in texts):
        raise TrainRefused(f"row {row_id}: a turn is empty or not a string")
    return row_id, texts[0], texts[1]


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def check_heldout(rows: Sequence[Mapping[str, Any]]) -> int:
    """Refuse rows whose user turn equals (NFC, stripped) a held-out utility prompt of any split: a stale
    training file must never train on held-out rows (decisions.md item 41). Returns how many held-out
    prompts were checked."""
    from lift.prompts import load

    held = {_norm(r.text) for r in load("utility", "hi")}
    hits = sorted(row_id for row_id, user, _answer in (row_turns(r) for r in rows) if _norm(user) in held)
    if hits:
        raise TrainRefused(f"{len(hits)} training rows have a user turn equal to a held-out utility prompt, "
                           f"e.g. {hits[:5]}: the training file is stale (decisions.md item 41). Rebuild it "
                           "with `lift data build --only sft`")
    return len(held)


# --- examples -----------------------------------------------------------------------------------------


def build_example(tokenizer: Any, row: Mapping[str, Any], max_len: int) -> dict[str, Any]:
    """One training example: the chat-templated conversation (``encode_chat_turn``) cut to ``max_len``
    tokens from the start, with label -100 on the prompt (BOS, preamble, user turn, generation prompt)
    and the answer's own ids, closing tokens included, on the rest. ``n_prompt`` and ``n_answer`` count
    the uncut parts; ``n_supervised`` the labels kept."""
    from lift.models.load import encode_chat_turn

    row_id, user, answer = row_turns(row)
    try:
        ids, n_prompt = encode_chat_turn(tokenizer, user, answer)
    except ValueError as exc:
        raise TrainRefused(f"row {row_id}: {exc}") from exc
    labels = [IGNORE_INDEX] * n_prompt + ids[n_prompt:]
    kept = labels[:max_len]
    return {"id": row_id, "input_ids": ids[:max_len], "labels": kept, "n_prompt": n_prompt,
            "n_answer": len(ids) - n_prompt, "n_supervised": sum(x != IGNORE_INDEX for x in kept),
            "truncated": len(ids) > max_len}


def build_examples(tokenizer: Any, rows: Sequence[Mapping[str, Any]],
                   max_len: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(every row's example, the rows left without a supervised token: id, prompt tokens, limit)."""
    examples = [build_example(tokenizer, row, max_len) for row in rows]
    unfit = [{"id": e["id"], "prompt_tokens": e["n_prompt"], "limit": max_len}
             for e in examples if not e["n_supervised"]]
    return examples, unfit


def unfit_message(unfit: Sequence[Mapping[str, Any]], max_len: int) -> str:
    listed = ", ".join(f"{u['id']} ({u['prompt_tokens']} prompt tokens)" for u in unfit)
    return (f"{len(unfit)} training rows leave no answer token within max_seq_len {max_len}, the chat "
            f"template's prompt (BOS, the ~366-token preamble, the user turn) fills it first: {listed}; "
            f"limit {max_len}. Each would train on nothing. Remedies, each with a dated docs/decisions.md "
            "entry: raise max_seq_len in configs/train.yaml, or drop exactly these rows by rerunning with "
            "--drop-unfit (their ids and count are recorded in the manifest and lift_train.json)")


def _spread(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": round(sum(values) / len(values), 2), "max": max(values)}


def data_stats(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Token counts of the examples that train (no text)."""
    return {
        "rows": len(examples),
        "truncated_rows": sum(bool(e["truncated"]) for e in examples),
        "truncated_ids": [e["id"] for e in examples if e["truncated"]],
        "prompt_tokens": _spread([e["n_prompt"] for e in examples]),
        "answer_tokens": _spread([e["n_answer"] for e in examples]),
        "sequence_tokens": _spread([len(e["input_ids"]) for e in examples]),
        "supervised_tokens": sum(e["n_supervised"] for e in examples),
        "supervised_per_row": _spread([e["n_supervised"] for e in examples]),
    }


@dataclass(frozen=True)
class Prepared:
    """A run's rows as training takes them, before any model loads (``prepare``)."""

    data: TrainingData
    n_heldout: int                      # held-out utility prompts the user turns were checked against
    tokenizer: Any
    max_len: int
    examples: list[dict[str, Any]]      # every row's example, in the file's order
    unfit: list[dict[str, Any]]         # rows left without a supervised token: id, prompt tokens, limit


def prepare(plan: Plan) -> Prepared:
    """The examples of a run, built the one way ``train_lora`` and ``check_training`` share: the training
    file (checked against report.json), the held-out guard, Global's tokenizer at the plan's revision,
    then every row through ``build_example``. Loads no model and writes nothing."""
    from lift.models.load import load_tokenizer

    data = load_training_data(plan.candidate, plan.cfg)
    n_heldout = check_heldout(data.rows)
    tokenizer = load_tokenizer(plan.model_id, plan.model_rev)
    max_len = int(plan.cfg["max_seq_len"])
    examples, unfit = build_examples(tokenizer, data.rows, max_len)
    return Prepared(data=data, n_heldout=n_heldout, tokenizer=tokenizer, max_len=max_len, examples=examples,
                    unfit=unfit)


def trained_examples(prepared: Prepared, *, drop_unfit: bool) -> list[dict[str, Any]]:
    """The examples that train: all of them, unless some rows are left without a supervised token, which
    stops the run (``unfit_message``) or, with ``drop_unfit``, are dropped, exactly those."""
    if prepared.unfit and not drop_unfit:
        raise TrainRefused(unfit_message(prepared.unfit, prepared.max_len))
    return [e for e in prepared.examples if e["n_supervised"]]


# --- training -----------------------------------------------------------------------------------------


class SupervisedTokenCounter:
    """Wraps TRL's collator. Each micro-batch's supervised tokens are ``labels[:, 1:] != -100``, the
    positions the shifted causal-LM loss scores; a micro-batch with none raises (the preamble crowding
    out the target). Its rows are counted too, and with ``max_rows`` (the plan's per-device batch) a
    micro-batch with more raises before the step uses it: the DataLoader's own batch, whatever the
    Trainer's arguments claim (nn.DataParallel over two GPUs collates two rows for one). The DataLoader
    collates one micro-batch ahead of the one the Trainer uses (accelerate looks ahead for the end), so
    counts wait in collation order and are booked when the Trainer reports a micro-batch used
    (``consumed``, ``step_end``): exact per micro-batch and per optimizer step. A batch collated ahead
    that training never reached (max_steps) stays unbooked."""

    def __init__(self, collate: Callable[[list[dict[str, Any]]], dict[str, Any]], *,
                 max_rows: int | None = None) -> None:
        self._collate = collate
        self.max_rows = max_rows
        self._pending: deque[tuple[int, int]] = deque()       # (supervised tokens, rows) per micro-batch
        self._step: list[tuple[int, int]] = []
        self.per_batch: list[int] = []
        self.per_step: list[int] = []
        self.rows_per_batch: list[int] = []
        self.rows_per_step: list[int] = []

    @property
    def in_step(self) -> int:
        return len(self._step)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.max_rows is not None and len(examples) > self.max_rows:
            raise RuntimeError(f"the Trainer's DataLoader put {len(examples)} rows in one micro-batch, but "
                               f"the recipe takes {self.max_rows} (several GPUs used through "
                               "nn.DataParallel?): stopped before any step trained on it. lift train runs on "
                               "one GPU: make one visible (CUDA_VISIBLE_DEVICES=0) and look at sft_config")
        batch = self._collate(examples)
        n = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
        if n == 0:
            raise ValueError(f"a batch with zero supervised tokens ({len(examples)} rows): the prompt fills "
                             "every position the loss scores, so the step would learn nothing "
                             "(build_examples refuses such rows before training; this batch got past it)")
        self._pending.append((n, len(examples)))
        return batch

    def consumed(self) -> None:
        if not self._pending:
            raise RuntimeError("the Trainer used a micro-batch the collator never built: the "
                               "supervised-token accounting is out of step with the training loop")
        n, rows = self._pending.popleft()
        self.per_batch.append(n)
        self.rows_per_batch.append(rows)
        self._step.append((n, rows))

    def step_end(self) -> None:
        self.consumed()
        self.per_step.append(sum(n for n, _rows in self._step))
        self.rows_per_step.append(sum(rows for _n, rows in self._step))
        self._step = []


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Heartbeat:
    """Progress lines for the job logs (counts only): the first and last optimizer steps, and one at
    least every ``every`` seconds in between, mid-step too (a T4 step accumulates 16 micro-batches)."""

    def __init__(self, name: str, *, every: float = HEARTBEAT_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.name, self.every, self._clock = name, every, clock
        self._t0: float | None = None
        self._last: float | None = None
        self.max_steps = 0
        self.lines = 0

    def start(self, max_steps: int) -> None:
        self._t0 = self._last = self._clock()
        self.max_steps = int(max_steps)
        log.info("%s: training for %d optimizer steps", self.name, self.max_steps)

    def due(self) -> bool:
        return self._last is None or self._clock() - self._last >= self.every

    def beat(self, step: int, *, loss: float | None = None, lr: float | None = None,
             epoch: float | None = None, micro: int | None = None, force: bool = False) -> bool:
        """Log one line if ``force`` or one is due. ``micro``: micro-batches done in a step under way."""
        if not (force or self.due()):
            return False
        now = self._clock()
        elapsed = now - (self._t0 if self._t0 is not None else now)
        under_way = f" under way ({micro} micro-batches done)" if micro else ""
        parts = [f"step {step}/{self.max_steps}{under_way}"]
        if epoch is not None:
            parts.append(f"epoch {epoch:.2f}")
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.3g}")
        parts.append(f"{_duration(elapsed)} elapsed")
        done = step - (1 if micro else 0)
        if 0 < done < self.max_steps:
            parts.append(f"about {_duration(elapsed / done * (self.max_steps - done))} left")
        log.info("%s: %s", self.name, ", ".join(parts))
        self._last = now
        self.lines += 1
        return True


def _callback(counter: SupervisedTokenCounter, heartbeat: Heartbeat) -> Any:
    """The TrainerCallback that books the counter and beats the heartbeat (transformers imported here)."""
    from transformers import TrainerCallback

    class LiftCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            heartbeat.start(state.max_steps)

        def on_substep_end(self, args, state, control, **kwargs):
            counter.consumed()
            heartbeat.beat(state.global_step + 1, micro=counter.in_step)

        def on_step_end(self, args, state, control, **kwargs):
            counter.step_end()

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs:
                heartbeat.beat(state.global_step, loss=logs.get("loss"), lr=logs.get("learning_rate"),
                               epoch=state.epoch, force=state.global_step in (1, state.max_steps))

    return LiftCallback()


def sft_config(plan: Plan, output_dir: Path) -> Any:
    """The SFTConfig of a run (trl imported here), held to one device. ``max_steps`` is set only when the
    caller asked.

    Outside a distributed launch, TrainingArguments counts every visible GPU (n_gpu =
    torch.cuda.device_count(), set by its device setup, which runs when the config is built); with n_gpu
    over 1 the Trainer wraps the model in nn.DataParallel and makes each micro-batch per_device_batch x
    n_gpu rows (32 rows an optimizer step on a Kaggle T4 x2, not 16). lift loads the model on cuda:0 and
    trains it there alone, so n_gpu is set to 1 here, the override the Trainer itself applies to a
    model-parallel model. It has to happen before the Trainer is built: Trainer.__init__ takes the batch
    its DataLoader uses from this config (``make_trainer`` then checks it)."""
    from trl import SFTConfig

    cfg = plan.cfg
    checkpointing = bool(cfg["gradient_checkpointing"])
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir), num_train_epochs=float(cfg["epochs"]), learning_rate=float(cfg["lr"]),
        lr_scheduler_type=str(cfg["scheduler"]),
        warmup_steps=float(cfg["warmup_ratio"]),            # a float in [0, 1) is a ratio of the steps
        per_device_train_batch_size=plan.per_device_batch, gradient_accumulation_steps=plan.grad_accum,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if checkpointing else None,
        bf16=plan.dtype == "bf16", fp16=plan.dtype == "fp16", use_cpu=plan.device == "cpu",
        logging_steps=1, save_strategy="no", report_to="none", disable_tqdm=True, dataloader_num_workers=0,
        max_length=None, packing=False, dataset_kwargs={"skip_prepare_dataset": True}, loss_type="nll",
        optim="adamw_torch", seed=plan.seed, data_seed=plan.data_seed,
    )
    if plan.max_steps is not None:
        kwargs["max_steps"] = plan.max_steps
    args = SFTConfig(**kwargs)
    if args.n_gpu > 1:          # the device setup has run (SFTConfig.__post_init__), so this stays
        args._n_gpu = 1
    return args


def lora_model(plan: Plan) -> tuple[Any, dict[str, Any]]:
    """Global loaded through lift.models.load (neutral generation config, frozen, eval) with trainable
    LoRA on every target projection of every layer, initialised after ``set_seed(seed)``. The LoRA
    weights must be fp32 on any base (PEFT's autocast_adapter_dtype casts them up from a bf16 or fp16
    base), or this raises before training: fp16 autocast cannot unscale fp16 gradients, and Adam's
    updates lose precision in bf16."""
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import set_seed

    from lift.models.load import load_model

    lora = plan.cfg["lora"]
    targets = [str(t) for t in lora["target_modules"]]
    base = load_model(plan.model_id, plan.model_rev, dtype=plan.dtype, device=plan.device)
    base.config.use_cache = False
    set_seed(plan.seed)
    model = get_peft_model(base, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                                            lora_dropout=float(lora["dropout"]), target_modules=targets,
                                            bias="none", task_type="CAUSAL_LM"))
    layers = int(base.config.num_hidden_layers)
    wrapped = sum(isinstance(m, LoraLayer) for m in model.modules())
    if wrapped != layers * len(targets):
        raise RuntimeError(f"LoRA wrapped {wrapped} modules, not {layers} layers x {len(targets)} targets: "
                           "configs/train.yaml's target_modules must name each projection of every decoder "
                           "layer exactly once (cohere2: q_proj k_proj v_proj o_proj gate_proj up_proj "
                           "down_proj)")
    trainable, total = model.get_nb_trainable_parameters()
    dtypes = sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters() if p.requires_grad})
    if dtypes != ["float32"]:
        raise RuntimeError(f"the trainable LoRA weights are {dtypes}, not float32, on a {plan.dtype} base: "
                           "PEFT must keep them in fp32 (get_peft_model's autocast_adapter_dtype), since "
                           "fp16 autocast cannot unscale fp16 gradients and Adam's updates lose precision "
                           "in bf16")
    info = {"r": lora["r"], "alpha": lora["alpha"], "dropout": float(lora["dropout"]),
            "target_modules": targets, "bias": "none", "layers": layers, "wrapped_modules": wrapped,
            "trainable_params": int(trainable), "total_params": int(total), "trainable_dtypes": dtypes}
    return model, info


def make_trainer(model: Any, tokenizer: Any, examples: Sequence[Mapping[str, Any]], plan: Plan, *,
                 output_dir: Path, heartbeat: Heartbeat | None = None) -> tuple[Any, SupervisedTokenCounter]:
    """The SFTTrainer over the pre-tokenized examples (input_ids and labels only), with the counting
    collator and the progress callback; the Trainer's own per-step printing gives way to the heartbeat.
    Refused (TrainRefused) unless the built Trainer takes the recipe's batch (``check_batch``)."""
    from datasets import Dataset
    from transformers import PrinterCallback
    from trl import SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    counter = SupervisedTokenCounter(DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id),
                                     max_rows=plan.per_device_batch)
    dataset = Dataset.from_list([{"input_ids": list(e["input_ids"]), "labels": list(e["labels"])}
                                 for e in examples])
    trainer = SFTTrainer(model=model, args=sft_config(plan, output_dir), train_dataset=dataset,
                         processing_class=tokenizer, data_collator=counter,
                         callbacks=[_callback(counter, heartbeat or Heartbeat(plan.name))])
    trainer.remove_callback(PrinterCallback)
    check_batch(plan, trainer_batch(trainer))
    return trainer, counter


def trainer_batch(trainer: Any) -> dict[str, int]:
    """The batch a built Trainer takes: n_gpu (after ``sft_config``'s pin), its processes, the rows per
    micro-batch its training DataLoader is built with, and the rows per optimizer step (that x gradient
    accumulation x data-parallel processes: Trainer.get_total_train_batch_size); and the GPUs this
    process sees (torch.cuda.device_count(): on a Kaggle T4 x2 the second stays idle)."""
    import torch

    args = trainer.args
    return {"n_gpu": int(args.n_gpu), "visible_gpus": int(torch.cuda.device_count()),
            "processes": int(args.world_size), "micro_batch": int(trainer._train_batch_size),
            "per_step": int(trainer.get_total_train_batch_size(args))}


def check_batch(plan: Plan, batch: Mapping[str, int]) -> None:
    """Refuse a Trainer that would not take the recipe's batch: one process on at most one GPU, the
    plan's per-device batch per micro-batch, its effective batch per optimizer step."""
    if (batch["n_gpu"] <= 1 and batch["processes"] == 1 and batch["micro_batch"] == plan.per_device_batch
            and batch["per_step"] == plan.effective_batch):
        return
    took = (f"{batch['per_step']} rows per optimizer step ({batch['micro_batch']} per micro-batch, "
            f"{batch['processes']} process(es), n_gpu {batch['n_gpu']})")
    raise TrainRefused(f"the Trainer would take {took}, not the recipe's {plan.effective_batch} "
                       f"({plan.per_device_batch} x {plan.grad_accum} accumulation): lift train is one plain "
                       "process on one GPU. Run it as `lift train`, not under torchrun or accelerate launch, "
                       "with one GPU visible if need be (CUDA_VISIBLE_DEVICES=0)")


def _fit(plan: Plan, tokenizer: Any, examples: Sequence[Mapping[str, Any]], scratch: Path,
         progress: dict[str, Any]) -> dict[str, Any]:
    """Train and save the adapter into ``scratch/adapter``; returns plain data only, so the model and
    the trainer die with this frame. While training, ``progress`` holds the counter and the trainer (its
    log history), for the manifest of a run that stops midway; the trainer leaves it once training ends.
    ``batch`` is the plan's batch beside the one the built Trainer takes (``trainer_batch``), and
    ``rows_per_step`` the rows each optimizer step took, as the collator counted them."""
    import torch

    model, lora_info = lora_model(plan)
    trainer, counter = make_trainer(model, tokenizer, examples, plan, output_dir=scratch / "trainer")
    batch = {"per_device": plan.per_device_batch, "grad_accum": plan.grad_accum,
             "effective": plan.effective_batch, **trainer_batch(trainer)}
    progress.update(counter=counter, trainer=trainer, batch=batch)
    log.info("%s: %d rows per optimizer step (%d per micro-batch x %d accumulation) on %s, n_gpu %d, "
             "%d GPU(s) visible", plan.name, batch["per_step"], batch["micro_batch"], plan.grad_accum,
             plan.device, batch["n_gpu"], batch["visible_gpus"])
    cuda = plan.device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(plan.device)
    started = time.monotonic()
    output = trainer.train()
    seconds = round(time.monotonic() - started, 3)
    peak = ({"allocated": int(torch.cuda.max_memory_allocated(plan.device)),
             "reserved": int(torch.cuda.max_memory_reserved(plan.device))} if cuda else None)
    staging = scratch / "adapter"
    model.save_pretrained(str(staging))                 # PEFT: the adapter only, as safetensors
    (staging / "README.md").unlink(missing_ok=True)     # PEFT's boilerplate model card
    tokenizer.save_pretrained(str(staging))
    history = [dict(entry) for entry in trainer.state.log_history]
    curve = _curve(history)
    train_output = {"global_step": int(output.global_step), "training_loss": float(output.training_loss),
                    "metrics": dict(output.metrics)}
    result = {
        "lora": lora_info, "staging": staging, "train_seconds": seconds, "peak_memory_bytes": peak,
        "global_step": int(trainer.state.global_step), "max_steps": int(trainer.state.max_steps),
        "metrics": dict(output.metrics), "loss_curve": curve, "batch": batch,
        "rows_per_step": list(counter.rows_per_step),
        "supervised_tokens_per_batch": list(counter.per_batch),
        "supervised_tokens_per_step": list(counter.per_step),
        "trainer_log": {"log_history": history, "train_output": train_output,
                        "max_steps": int(trainer.state.max_steps),
                        "num_train_epochs": int(trainer.state.num_train_epochs),
                        "training_args": trainer.args.to_dict()},
    }
    log.info("%s: trained %d optimizer steps in %s (final loss %s)", plan.name, result["global_step"],
             _duration(seconds), f"{curve[-1]['loss']:.4f}" if curve else "none logged")
    progress.pop("trainer")
    trainer.accelerator.free_memory()
    return result


def _curve(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: entry[k] for k in _CURVE_KEYS if k in entry} for entry in history if "loss" in entry]


def _release() -> None:
    """Collect the freed model's cycles and hand its CUDA blocks back (before parity loads a base)."""
    from lift.commands.run import _release_memory

    _release_memory()


# --- files --------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, obj: Any) -> None:
    """UTF-8, LF, written to a temp file beside it (``*.tmp``, never synced) and renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return doc


def _publish(staging: Path, final: Path) -> None:
    """Copy the staged adapter under ``final``'s parent's .cache/ (never synced, same disk as ``final``)
    and rename it to ``final``: the folder appears whole or not at all."""
    if final.exists():
        raise FileExistsError(f"{final} appeared while this run trained (another process?); the new adapter "
                              "was not moved there")
    hidden = ensure(final.parent / ".cache")
    target = Path(tempfile.mkdtemp(prefix=f"{final.name}-", dir=hidden))
    try:
        shutil.copytree(staging, target, dirs_exist_ok=True)
        os.replace(target, final)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    try:
        hidden.rmdir()                                  # only when nothing else is staged there
    except OSError:
        pass


def _set_aside(directory: Path, name: str) -> dict[str, Any] | None:
    """--retrain: move an existing adapter folder under outputs/.aside/adapters/{name}-{utc stamp}."""
    from lift.models.generate import ASIDE_DIR
    from lift.train.merge import adapter_hash

    if not directory.exists():
        log.info("--retrain: %s does not exist; nothing to set aside", directory)
        return None
    try:
        old = adapter_hash(directory)
    except FileNotFoundError:
        old = None
    base = ensure(output_dir() / ASIDE_DIR / "adapters") / f"{name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    target, n = base, 0
    while target.exists():
        n += 1
        target = base.with_name(f"{base.name}-{n}")
    shutil.move(str(directory), str(target))
    log.warning("--retrain: %s set aside in %s (never synced). The runs repo keeps the old files until "
                "this run's push replaces them, and every output made from the old adapter (hash %s) goes "
                "stale once the new one is registered: their cache keys pin the old hash", directory, target,
                old[:12] if old else "none: no weights")
    return {"adapter_hash": old, "aside": _shown(target)}


def _shown(path: Path) -> str:
    """A path for records: relative to outputs/ or the repo when under them, POSIX."""
    for root, prefix in ((output_dir(), "outputs"), (REPO_ROOT, "")):
        try:
            rel = Path(path).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        return f"{prefix}/{rel}" if prefix else rel
    return Path(path).as_posix()


# --- push, parity, registration -----------------------------------------------------------------------


def _preflight() -> None:
    """lift.sync.preflight(write=True) before any GPU time: a problem no rerun fixes (no LIFT_RUNS_REPO,
    no token, a read token) refuses the run; a transient one is logged and the push is tried anyway."""
    from lift import sync

    try:
        sync.preflight(write=True)
    except sync.SyncError as exc:
        if exc.permanent:
            raise TrainRefused(f"the push after training would fail: {exc}. Fix it, or pass --no-push (the "
                               "runner's own pushes still send the adapter)") from exc
        log.warning("lift sync preflight: %s; training goes on, and the push after it is tried anyway", exc)


def _push(name: str) -> dict[str, Any]:
    """Push outputs/adapters/{name} now. Never raises: an error or a conflict is logged and recorded,
    and the runner's next lift sync push sends the folder again (sync compares content)."""
    from lift import sync

    when = _now()
    try:
        report = sync.push([f"outputs/adapters/{name}"])
    except Exception as exc:                # a push never fails training (Ctrl+C still stops the run)
        permanent = bool(getattr(exc, "permanent", False))
        log.error("%s: pushing outputs/adapters/%s failed (%s%s); the run goes on, and the runner's next "
                  "lift sync push sends it again: %s", name, name, type(exc).__name__,
                  ", permanent" if permanent else "", exc)
        return {"when": when, "ok": False, "error": type(exc).__name__, "permanent": permanent}
    for line in report.summary():
        (log.info if report.ok else log.warning)("%s", line)
    if not report.ok:
        log.warning("%s: the push left %d file(s) in conflict with the runs repo's copies; settle them (lift "
                    "sync) before relying on the Hub's adapter", name, len(report.conflicts))
    commit = (report.commits[-1] if report.commits else report.revision) if report.ok else None
    return {"when": when, "ok": bool(report.ok), "revision": report.revision, "commits": list(report.commits),
            "commit": commit, "files": dict(sorted(Counter(report.files.values()).items())),
            "conflicts": list(report.conflicts)}


def gate0_parity_seed() -> int:
    """The seed of Gate 0's merge-parity rule: lift.gate0.run_gate0 draws its sample with SEED + 3."""
    from lift import gate0

    return gate0.SEED + GATE0_PARITY_SEED_OFFSET


def gate0_parity_sample() -> list[Any]:
    """The merge-parity prompts drawn by Gate 0's parity rule (``mixed_sample(N_PARITY, split="dev",
    seed=SEED + 3)``, as lift.gate0.run_gate0 draws them for its random-LoRA parity: 20 dev prompts mixed
    over the safety sets and languages) from the dev prompt files as they are now. They are the prompts
    Gate 0's parity ran only while those 18 dev pools are unchanged since Gate 0, which records no pids:
    a spot-check exclusion changes its cell's pick, and a fixed translation changes the text.
    registration.json lists the pids drawn (``_parity_texts``)."""
    from lift import gate0
    from lift.prompts import mixed_sample

    return mixed_sample(gate0.N_PARITY, split="dev", seed=gate0_parity_seed())


def _parity_texts(given: Sequence[str] | None) -> tuple[list[str], dict[str, Any]]:
    """The parity prompts (``gate0_parity_sample``, unless given) and how to find them again (the rule
    and the pids drawn, or the given texts' hashes; never the texts)."""
    if given is not None:
        texts = [str(t) for t in given]
        if not texts:
            raise TrainRefused("parity_prompts is empty")
        return texts, {"source": "given", "n": len(texts), "sha256": [sha256_text(t) for t in texts]}
    from lift import gate0

    records = gate0_parity_sample()
    source = ("drawn by Gate 0's merge-parity rule over the current dev prompts: "
              f"lift.prompts.mixed_sample({gate0.N_PARITY}, split='dev', seed={gate0_parity_seed()})")
    return [r.text for r in records], {"source": source, "n": len(records),
                                       "pids": [f"{r.set}_{r.lang}:{r.pid}" for r in records]}


def parity_verdict(parity: Mapping[str, Mapping[str, Any]]) -> tuple[bool, str]:
    """(may be registered, a one-line summary) from registration.json's parity results by precision:
    fp32 must match token for token; fp16 alone (a T4) registers with a warning; none does not."""
    fp32, fp16 = parity.get("fp32"), parity.get("fp16")
    if fp32 is not None:
        line = f"fp32 {fp32['n_match']}/{fp32['n']} prompts token-identical"
        return bool(fp32["all_match"]), line if fp32["all_match"] else f"FAILED: {line}"
    if fp16 is not None:
        return True, f"fp16 {fp16['n_match']}/{fp16['n']} prompts token-identical (T4; fp32 not checked)"
    return False, "not checked"


def recorded_parity_failure(directory: Path, digest: str) -> str | None:
    """The failed fp32 merge parity the adapter folder's registration.json records for adapter hash
    ``digest`` (parity_verdict's summary, "FAILED: fp32 n/20 prompts token-identical"), or None: no
    registration.json (an adapter lift train did not make, such as the tests' tiny one), one written for
    another hash (a set-aside or replaced adapter), or a parity that did not fail in fp32 (it passed, a
    T4's fp16 alone, or none was checked). lift.models.variants.checked_adapter refuses an adapter with
    such a record (CLAUDE.md rule 9). A registration.json that cannot be read raises ValueError: whether
    it records a failure cannot be told."""
    path = Path(directory) / REGISTRATION
    if not path.is_file():
        return None
    try:
        reg = _read_json(path)
        parity = reg.get("parity") or {}
        if reg.get("adapter_hash") != digest or parity.get("fp32") is None:
            return None
        ok, summary = parity_verdict(parity)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{path} cannot be read ({type(exc).__name__}: {exc}), so whether it records a "
                         "failed merge parity for this adapter cannot be told: pull it again (`lift sync "
                         "pull`) or look at it") from exc
    return None if ok else summary


def _parity(plan: Plan, final: Path, digest: str, prompts: Sequence[str],
            prompt_info: Mapping[str, Any] | None, parity_dtype: str) -> dict[str, Any]:
    from lift.models.load import device_name
    from lift.train.merge import parity_check

    _release()
    log.info("%s: merge parity in %s on %d prompts", plan.name, parity_dtype, len(prompts))
    result = parity_check(final, prompts=list(prompts), device=plan.device, dtype=parity_dtype,
                          base_id=plan.model_id, base_revision=plan.model_rev)
    if result["adapter_hash"] != digest:
        raise RuntimeError(f"parity_check hashed {final} as {result['adapter_hash'][:12]}, not "
                           f"{digest[:12]}: the folder changed while parity ran")
    result.update(checked=_now(), device=plan.device, gpu=device_name(plan.device),
                  prompts=dict(prompt_info or {}))
    log.info("%s: merge parity %s: %d/%d prompts token-identical", plan.name, parity_dtype,
             result["n_match"], result["n"])
    return result


def _location(name: str, pushes: Sequence[Mapping[str, Any]]) -> str:
    """Where the adapter lives in the runs repo, with the commit of the latest push that left it there.
    The repo id itself is a secret (LIFT_RUNS_REPO), so the entry names the variable."""
    where = f"runs repo ($LIFT_RUNS_REPO): outputs/adapters/{name}"
    commit = next((p.get("commit") for p in reversed(pushes) if p.get("ok") and p.get("commit")), None)
    return f"{where}@{commit}" if commit else where


def _note(trained: Mapping[str, Any], parity: str) -> str:
    return (f"{trained['candidate']}, {trained['recipe']} recipe, seed {trained['seed']}: trained "
            f"{str(trained.get('created', ''))[:10]} on {trained.get('gpu')} in {trained.get('dtype')}, "
            f"{trained['data']['rows_trained']} rows of {trained['data']['key']}; merge parity {parity}")


def adapter_entry(name: str, digest: str, *, seed: int, location: str, note: str) -> dict[str, Any]:
    """The configs/variants.yaml entry of a trained adapter (location and note document it; the rest is
    the VariantSpec every cache key built on it digests)."""
    return {"name": name, "method": "adapter", "adapter": name, "seed": int(seed),
            "artifacts": {"adapter": digest}, "location": location, "note": note}


def _spec(entry: Mapping[str, Any]) -> Any:
    from lift.models.variants import REGISTRY_ONLY_KEYS
    from lift.schemas import VariantSpec

    return VariantSpec.model_validate({k: v for k, v in dict(entry).items() if k not in REGISTRY_ONLY_KEYS})


def _registration_doc(final: Path, digest: str, plan: Plan) -> dict[str, Any]:
    """registration.json for this hash: the one in ``final`` when it was written for it, else a new one."""
    path = final / REGISTRATION
    if path.is_file():
        doc = _read_json(path)
        if doc.get("adapter_hash") == digest:
            doc.setdefault("parity", {})
            doc.setdefault("pushes", [])
            return doc
        log.warning("%s was written for another adapter hash (%s, a set-aside or replaced adapter): a new "
                    "one is started for %s", path, str(doc.get("adapter_hash"))[:12], digest[:12])
    return {"version": RECORD_VERSION, "name": plan.name, "adapter_hash": digest, "candidate": plan.candidate,
            "recipe": plan.recipe, "seed": plan.seed, "parity": {}, "entry": None, "registered": None,
            "unregistered": None, "registry_error": None, "pushes": [], "created": _now()}


@dataclass
class _Post:
    """What the steps after training did (registration.json as written, and this run's own records)."""

    status: str                                  # ok | parity_failed | registry_conflict
    registration: dict[str, Any]
    parity: dict[str, Any] | None = None         # the parity this run checked
    pushes: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None


def _post_steps(plan: Plan, final: Path, digest: str, trained: Mapping[str, Any], *, push: bool,
                parity: bool, prompts: Sequence[str], prompt_info: Mapping[str, Any] | None,
                parity_dtype: str, registry: Path | None, replace: bool, registrable: bool) -> _Post:
    """Push, parity (unless registration.json holds it for this hash and precision), registration,
    registration.json, and the push that sends it (recorded in the manifest only: a record of it inside
    the file it sends would change the file again)."""
    reg = _registration_doc(final, digest, plan)
    post = _Post(status="ok", registration=reg)
    if push:
        post.pushes.append(_push(plan.name))
        reg["pushes"].append(post.pushes[-1])
    if parity and parity_dtype not in reg["parity"]:
        post.parity = _parity(plan, final, digest, prompts, prompt_info, parity_dtype)
        reg["parity"][parity_dtype] = post.parity
    elif parity:
        log.info("%s: parity in %s is already recorded for this adapter hash (registration.json); not run "
                 "again", plan.name, parity_dtype)
    ok, summary = parity_verdict(reg["parity"])
    if not ok and reg["parity"].get("fp32") is not None:
        post.status = "parity_failed"
        withdrawn = _unregister_failed(plan.name, digest, reg, registry=registry, registrable=registrable)
        post.message = _parity_failed(plan.name, final, reg["parity"]["fp32"], withdrawn)
        log.error("%s", post.message)
    elif not ok:
        log.warning("%s: not registered: no merge parity was checked (parity=False). Run the same lift train "
                    "line again, which resumes, to check parity and register", plan.name)
    elif not registrable:
        log.warning("%s: not registered: %s is not outputs/adapters/%s, where build_variant loads adapters "
                    "from", plan.name, final, plan.name)
    else:
        if "fp32" not in reg["parity"]:
            log.warning("%s: registered on an fp16 parity (%s): fp16 rounds the merged weights differently, "
                        "so the match is recorded, not required. Check it in fp32 where the memory allows",
                        plan.name, summary)
        entry = adapter_entry(plan.name, digest, seed=plan.seed, location=_location(plan.name, reg["pushes"]),
                              note=_note(trained, summary))
        reg["entry"] = entry
        path = Path(registry) if registry is not None else default_registry()
        try:
            outcome = register_adapter(plan.name, digest, candidate=plan.candidate, recipe=plan.recipe,
                                       seed=plan.seed, location=entry["location"], note=entry["note"],
                                       registry=path, replace=replace)
        except RegistryConflict as exc:
            post.status, post.message = "registry_conflict", str(exc)
            reg["registered"], reg["registry_error"] = None, str(exc)
            log.error("%s", exc)
        else:
            reg["registered"] = {"registry": _shown(path), "outcome": outcome, "when": _now()}
            reg["registry_error"] = None
    reg["updated"] = _now()
    _write_json(final / REGISTRATION, reg)
    if push:
        post.pushes.append(_push(plan.name))
    return post


def _unregister_failed(name: str, digest: str, reg: dict[str, Any], *, registry: Path | None,
                       registrable: bool) -> str | None:
    """A failed fp32 parity un-registers the adapter (CLAUDE.md rule 9). registration.json loses its entry
    and its registered block, which an earlier fp16 parity (a T4) may have written; that block is kept
    as ``unregistered`` (what it was, the reason, the time). The registry this run writes loses the
    adapter's entry when that entry pins this hash (``unregister_adapter``): the entry the laptop
    committed on the fp16 registration, at the ref the job runs, or one this machine wrote. Returns the
    sentences ParityFailed adds about it, or None when no registration is known (a first parity that
    failed)."""
    old = reg.get("registered")
    reg["entry"], reg["registered"], reg["registry_error"] = None, None, None
    if old:
        reg["unregistered"] = {"was": old, "reason": "fp32 merge parity failed", "when": _now()}
    kept = "registration.json keeps the fp16 parity's registration as unregistered" if old else None
    if not registrable:                 # an adapter elsewhere: the registry describes outputs/adapters/{name}
        return f"An fp16 parity had registered it: {kept}" if kept else None
    path = Path(registry) if registry is not None else default_registry()
    laptop = (f"On the laptop, run `lift sync pull`, then `lift register-adapter {name}`, which removes an "
              "entry configs/variants.yaml holds for this adapter hash, and commit configs/variants.yaml "
              "before any job runs")
    try:
        removed = unregister_adapter(name, digest, registry=path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        reg["registry_error"] = str(exc)
        return f"Checking {_shown(path)} here for its entry failed: {exc}. {laptop}"
    if not (removed or kept):
        return None
    what = (f"This run removed its entry (adapter hash {digest[:12]}) from {_shown(path)}" if removed
            else f"{_shown(path)} here holds no entry for it")
    return f"{what}{f', and {kept}' if kept else ''}. {laptop}"


def _parity_failed(name: str, final: Path, result: Mapping[str, Any], withdrawn: str | None = None) -> str:
    """ParityFailed's message: what diverged, where the adapter is, and (``withdrawn``) what became of a
    registration an earlier fp16 parity made."""
    firsts = [d for d in result.get("first_divergence") or [] if d is not None]
    where = f" (first divergence at token {min(firsts)})" if firsts else ""
    text = (f"merge parity failed for {name}: in fp32 the merged adapter's greedy outputs differ from the "
            f"unmerged adapter's on {result['n'] - result['n_match']} of {result['n']} prompts{where}. The "
            f"adapter is saved in {final} and pushed, and registration.json records the parity, but it is "
            "NOT registered and must not be evaluated until the cause is understood (CLAUDE.md rule 9)")
    return f"{text}. {withdrawn}" if withdrawn else text


def _laptop_lines(name: str) -> list[str]:
    return ["configs/ travels by git, not by lift sync: a GPU runtime loses this registration with the "
            f"session. On the laptop run `lift sync pull`, then `lift register-adapter {name}`, then commit "
            "configs/variants.yaml and push it to the ref the GPU jobs run (lift remote --ref, Kaggle "
            "BRANCH)"]


# --- the registry -------------------------------------------------------------------------------------


def default_registry() -> Path:
    """configs/variants.yaml, the registry build_variant reads."""
    return CONFIG_DIR / "variants.yaml"


def register_adapter(name: str, digest: str, *, candidate: str, recipe: str, seed: int, location: str,
                     note: str, registry: Path | None = None, replace: bool = False) -> str:
    """Add the adapter's entry to the registry (configs/variants.yaml unless ``registry``) as text, so
    the file's comments stay: "added", "unchanged" (the same entry is there: location and note aside,
    the fields every cache key digests are equal) or "replaced" (another entry under the name, only with
    ``replace``: each output made from it goes stale). The edited text must parse to exactly the old
    entries plus this one, as lift.models.variants.load_registry reads them, or nothing is written."""
    if name != adapter_name(candidate, recipe, seed):
        raise ValueError(f"{name!r} is not the adapter name of {candidate}, {recipe}, seed {seed} "
                         f"({adapter_name(candidate, recipe, seed)!r})")
    if not _HEX64.fullmatch(str(digest)):
        raise ValueError(f"{digest!r} is not an adapter hash (64 hex: lift.train.merge.adapter_hash)")
    path = Path(registry) if registry is not None else default_registry()
    entry = adapter_entry(name, digest, seed=seed, location=location, note=note)
    text = path.read_text(encoding="utf-8")
    before = _registry_specs(yaml.safe_load(text), path)
    spec, old = _spec(entry), before.get(name)
    if old == spec:
        log.info("%s: %s already registers it with adapter hash %s", name, _shown(path), digest[:12])
        return "unchanged"
    if old is not None and not replace:
        raise RegistryConflict(f"{_shown(path)} already registers {name} differently "
                               f"({_differs(old, spec)}). Replacing it changes the cache key of every output "
                               "made from the registered entry, so those cells go stale (lift restart): "
                               "replace it only when this adapter is meant to replace that one (lift "
                               "register-adapter --replace, or lift train --retrain)")
    new_text = _replaced_entry(text, entry, path) if old is not None else _appended_entry(text, entry)
    try:
        after = _registry_specs(yaml.safe_load(new_text), path)
    except (yaml.YAMLError, ValueError):
        after = None
    others = {k: v for k, v in before.items() if k != name}
    if after is None or after.get(name) != spec or {k: v for k, v in after.items() if k != name} != others:
        raise ValueError(f"editing {path} as text would not leave exactly its entries plus {name} (its "
                         f"layout is not the one this writer expects); add the entry by hand: "
                         f"{json.dumps(entry)}")
    _write_text(path, new_text)
    _check_loaded(path, name, spec)
    outcome = "replaced" if old is not None else "added"
    log.info("%s: %s in %s (adapter hash %s)", name, outcome, _shown(path), digest[:12])
    return outcome


def unregister_adapter(name: str, digest: str, *, registry: Path | None = None) -> bool:
    """Remove ``name``'s entry from the registry (configs/variants.yaml unless ``registry``) when it pins
    adapter hash ``digest``: that adapter failed its fp32 merge parity after an fp16 parity registered it,
    and must not be evaluated (CLAUDE.md rule 9). True when the entry was removed; False when the registry
    holds no entry under ``name`` pinning ``digest`` (an entry pinning another hash is another adapter's,
    and stays). A text edit, so the file's comments stay: the edited text must parse to exactly the
    other entries and the rest of the document, or nothing is written and ValueError names the entry to
    delete by hand."""
    path = Path(registry) if registry is not None else default_registry()
    text = path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    old = _registry_specs(doc, path).get(name)
    if old is None or old.artifacts.get("adapter") != digest:
        return False
    expected = {**doc, "variants": [e for e in doc.get("variants") or [] if e.get("name") != name]}
    try:
        new_text = _removed_entry(text, name)
        after = yaml.safe_load(new_text)
    except (yaml.YAMLError, ValueError):
        after = None
    if not isinstance(after, dict) or {**after, "variants": after.get("variants") or []} != expected:
        raise ValueError(f"{_shown(path)} still registers {name} (adapter hash {digest[:12]}), and editing "
                         "it as text would not leave exactly its other entries (its layout is not the one "
                         "this writer expects): delete that entry by hand and commit configs/variants.yaml")
    _write_text(path, new_text)
    _check_loaded(path, name, None)
    log.warning("%s: removed from %s (adapter hash %s): its fp32 merge parity failed, and nothing may "
                "evaluate it (CLAUDE.md rule 9)", name, _shown(path), digest[:12])
    return True


def _differs(old: Any, new: Any) -> str:
    a, b = old.model_dump(), new.model_dump()
    return "; ".join(f"{k} {a[k]!r} there, {b[k]!r} here" for k in a if a[k] != b[k])


def _registry_specs(doc: Any, path: Path) -> dict[str, Any]:
    """{name: VariantSpec} of a registry document, read as lift.models.variants.load_registry reads
    configs/variants.yaml (documentation-only keys dropped, duplicates refused), for any path."""
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must hold a mapping with a 'variants:' list")
    out: dict[str, Any] = {}
    for entry in doc.get("variants") or []:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: a variants entry is not a mapping")
        spec = _spec(entry)
        if spec.name in out:
            raise ValueError(f"{path} lists {spec.name!r} twice")
        out[spec.name] = spec
    return out


def _entry_lines(entry: Mapping[str, Any], indent: str) -> list[str]:
    """The entry as YAML block lines, strings quoted as JSON (valid YAML double-quoted scalars)."""
    inner = indent + "  "
    return [f"{indent}- name: {entry['name']}", f"{inner}method: {entry['method']}",
            f"{inner}adapter: {entry['adapter']}", f"{inner}seed: {int(entry['seed'])}",
            f"{inner}artifacts: {{adapter: {json.dumps(entry['artifacts']['adapter'])}}}",
            f"{inner}location: {json.dumps(entry['location'], ensure_ascii=False)}",
            f"{inner}note: {json.dumps(entry['note'], ensure_ascii=False)}"]


def _appended_entry(text: str, entry: Mapping[str, Any]) -> str:
    items = [m for line in text.splitlines() if (m := _ITEM.match(line))]
    indent = items[0].group("indent") if items else "  "
    body = text if not text or text.endswith("\n") else text + "\n"
    return body + "\n".join(_entry_lines(entry, indent)) + "\n"


def _entry_span(lines: Sequence[str], name: str) -> tuple[int, int] | None:
    """(first line, last line + 1) of ``name``'s entry in a registry's lines: its one '- name:' line,
    then the lines more indented than its '- ', up to a blank line. None unless exactly one '- name:'
    line names it."""
    starts = [i for i, line in enumerate(lines)
              if (m := _ITEM.match(line)) and m.group("name").strip("\"'") == name]
    if len(starts) != 1:
        return None
    start = starts[0]
    indent = len(_ITEM.match(lines[start]).group("indent"))
    end = start + 1
    while end < len(lines) and lines[end].strip() and _indent(lines[end]) > indent:
        end += 1
    return start, end


def _replaced_entry(text: str, entry: Mapping[str, Any], path: Path) -> str:
    lines = text.splitlines()
    span = _entry_span(lines, entry["name"])
    if span is None:
        raise ValueError(f"{path}: cannot find {entry['name']}'s entry as one '- name: {entry['name']}' "
                         f"line; replace it by hand: {json.dumps(dict(entry))}")
    start, end = span
    indent = _ITEM.match(lines[start]).group("indent")
    return "\n".join(lines[:start] + _entry_lines(entry, indent) + lines[end:]) + "\n"


def _removed_entry(text: str, name: str) -> str:
    """``text`` without ``name``'s entry (``_entry_span``), and without the blank line before it when a
    blank line or the end of the file follows it, so no gap is left doubled."""
    lines = text.splitlines()
    span = _entry_span(lines, name)
    if span is None:
        raise ValueError(f"cannot find {name}'s entry as one '- name: {name}' line")
    start, end = span
    if start and not lines[start - 1].strip() and (end == len(lines) or not lines[end].strip()):
        start -= 1
    return "\n".join(lines[:start] + lines[end:]) + "\n"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _check_loaded(path: Path, name: str, spec: Any) -> None:
    """When ``path`` is the registry lift.models.variants reads, clear the config cache and check that
    load_registry() parses it and returns ``spec`` under ``name`` (None: no entry, after a removal): this
    process reads the new registry from now."""
    from lift.models import variants

    if path.resolve() != (config.CONFIG_DIR / "variants.yaml").resolve():
        return
    clear = getattr(config.load, "cache_clear", None)
    if clear is not None:
        clear()
    if variants.load_registry().get(name) != spec:
        raise RuntimeError(f"{path} was written, but lift.models.variants.load_registry() does not read "
                           f"{name} as written; look at the file")


def register_pulled(name: str, *, registry: Path | None = None, replace: bool = False) -> str:
    """The laptop's registration (``lift register-adapter``), after ``lift sync pull``: recompute the
    hash of outputs/adapters/{name}/, check it against registration.json and that parity passed, and
    write the entry registration.json holds (the one the training job wrote): register_adapter's
    "added", "unchanged" or "replaced".

    When the fp32 parity failed, the adapter must not be evaluated (CLAUDE.md rule 9): an entry the
    registry holds for this hash (one an earlier fp16 parity registered, committed) is removed the way
    ``unregister_adapter`` removes it, and "removed" returned, for the caller to have that commit made;
    otherwise this raises, as for any parity that did not pass."""
    from lift.models.variants import adapter_path
    from lift.train.merge import ADAPTER_WEIGHTS, adapter_hash

    directory = adapter_path(name)
    rerun = "rerun its lift train line (it resumes without training), then lift sync pull"
    if not (directory / ADAPTER_WEIGHTS).is_file():
        raise FileNotFoundError(f"{directory} holds no {ADAPTER_WEIGHTS}: run `lift sync pull` first (the "
                                "training job pushes the adapter there)")
    if not (directory / REGISTRATION).is_file():
        raise FileNotFoundError(f"{directory} holds no {REGISTRATION}: the training job has not checked "
                                f"parity yet. {rerun.capitalize()}")
    reg = _read_json(directory / REGISTRATION)
    digest = adapter_hash(directory)
    if reg.get("adapter_hash") != digest:
        raise ValueError(f"{directory}'s files hash to {digest[:12]}, but {REGISTRATION} was written for "
                         f"{str(reg.get('adapter_hash'))[:12]}: these are not the files the job checked (a "
                         "partial pull, or a retrain still under way). Run lift sync pull again, or rerun "
                         "the lift train line")
    ok, summary = parity_verdict(reg.get("parity") or {})
    if not ok:
        if (reg.get("parity") or {}).get("fp32") is not None:          # failed in fp32
            try:
                if unregister_adapter(name, digest, registry=registry):
                    return "removed"
            except ValueError as exc:
                raise ValueError(f"{name}: merge parity {summary}, so it must not be evaluated (CLAUDE.md "
                                 f"rule 9); {exc}") from exc
        raise ValueError(f"{name}: merge parity {summary}; it is not registered (CLAUDE.md rule 9)")
    entry = reg.get("entry")
    if not isinstance(entry, dict):
        raise ValueError(f"{directory / REGISTRATION} holds no entry (see the training job's log): {rerun}")
    if entry.get("name") != name or (entry.get("artifacts") or {}).get("adapter") != digest:
        raise ValueError(f"{directory / REGISTRATION}'s entry is not {name} with adapter hash {digest[:12]}")
    return register_adapter(name, digest, candidate=str(reg["candidate"]), recipe=str(reg["recipe"]),
                            seed=int(entry["seed"]), location=str(entry["location"]),
                            note=str(entry["note"]), registry=registry, replace=replace)


# --- the run ------------------------------------------------------------------------------------------


def _check_same_hardware(plan: Plan) -> None:
    """Refuse a seed whose GPU class or precision differs from another seed's of the recipe here."""
    from lift.models.generate import stack
    from lift.models.load import device_name
    from lift.models.variants import adapter_path

    mine = {"dtype": plan.dtype, "gpu_capability": stack(plan.device)["gpu_capability"]}
    for other in sorted({int(s) for s in plan.cfg["seeds"]} - {plan.seed}):
        name = adapter_name(plan.candidate, plan.recipe, other)
        path = adapter_path(name) / LIFT_TRAIN
        if not path.is_file():
            continue
        theirs = _read_json(path)
        was = {"dtype": theirs.get("dtype"), "gpu_capability": theirs.get("gpu_capability")}
        if was != mine:
            raise TrainRefused(f"{name} was trained in {was['dtype']} on a GPU of compute capability "
                               f"{was['gpu_capability']} ({theirs.get('gpu')}); this run would train in "
                               f"{mine['dtype']} on {mine['gpu_capability']} ({device_name(plan.device)}). "
                               "All seeds of a recipe train on one GPU class and precision: run this seed "
                               f"where {name} ran")


def _train_record(plan: Plan, data: TrainingData, n_heldout: int, stats: Mapping[str, Any],
                  unfit: Sequence[Mapping[str, Any]], fit: Mapping[str, Any],
                  replaced: Mapping[str, Any] | None, gpu: str) -> dict[str, Any]:
    """lift_train.json: what was trained, from what, how, on what."""
    from lift import manifest
    from lift.models.generate import stack

    env = stack(plan.device)
    metrics = fit["metrics"]
    return {
        "version": RECORD_VERSION, "name": plan.name, "candidate": plan.candidate, "recipe": plan.recipe,
        "recipe_tag": RECIPE_TAGS[plan.recipe], "seed": plan.seed, "data_seed": plan.data_seed,
        "recipe_config": plan.cfg,
        "data": {"path": data.repo_path, "key": data.key, "sha256": data.sha256,
                 "rows_in_file": len(data.rows), "rows_trained": stats["rows"],
                 "dropped_unfit": {"count": len(unfit), "ids": [u["id"] for u in unfit]},
                 "heldout_prompts_checked": n_heldout},
        "counts": dict(stats), "lora": fit["lora"], "batch": dict(fit["batch"]),
        "dtype": plan.dtype, "autocast": plan.dtype if plan.dtype in ("bf16", "fp16") else None,
        "model": {"id": plan.model_id, "revision": plan.model_rev}, "device": plan.device, "gpu": gpu,
        "gpu_capability": env["gpu_capability"], "stack": env, "versions": manifest.versions(),
        "git": manifest.git_sha(), "max_steps": plan.max_steps,
        "train": {"global_step": fit["global_step"], "max_steps": fit["max_steps"],
                  "train_runtime": metrics.get("train_runtime"), "train_seconds": fit["train_seconds"],
                  "train_loss": metrics.get("train_loss"), "peak_memory_bytes": fit["peak_memory_bytes"]},
        "rows_per_step": fit["rows_per_step"],
        "supervised_tokens_per_step": fit["supervised_tokens_per_step"],
        "retrained_from": dict(replaced) if replaced else None, "created": _now(),
    }


def train_lora(candidate: str, recipe: str, seed: int, *, out_dir: Path | None = None,
               model_id: str | None = None, revision: str | None = None, device: str = "cuda:0",
               dtype: str | None = None, per_device_batch: int | None = None, max_steps: int | None = None,
               push: bool = True, parity: bool = True, parity_prompts: Sequence[str] | None = None,
               parity_dtype: str = "fp32", registry: Path | None = None, retrain: bool = False,
               drop_unfit: bool = False) -> Path:
    """Return the adapter directory: outputs/adapters/adapter-{candidate}-{std|stress}-s{seed}/, trained
    now or by an earlier run (then only the steps still missing run: module docstring).

    ``out_dir`` puts the adapter elsewhere: such an adapter is never pushed (``push`` must be False) or
    registered, since the push and the registry describe outputs/adapters/{name}. ``parity_prompts``
    default to the 20 dev prompts drawn by Gate 0's parity rule from the dev prompts as they are now
    (``gate0_parity_sample``). ``registry`` (tests: a file in tmp) defaults to configs/variants.yaml.
    Raises TrainRefused (a refused request, before any GPU time, or a Trainer off the recipe's batch,
    before its first step), ParityFailed (fp32 merge parity failed: saved and pushed, not registered,
    and an fp16 parity's registration withdrawn) or RegistryConflict (another entry under the name)."""
    from lift.models.variants import adapter_path
    from lift.train.merge import ADAPTER_WEIGHTS

    plan = plan_training(candidate, recipe, seed, model_id=model_id, revision=revision, device=device,
                         dtype=dtype, per_device_batch=per_device_batch, max_steps=max_steps)
    check_parity_dtype(parity_dtype, plan.device)
    canonical = adapter_path(plan.name)
    final = Path(out_dir) if out_dir is not None else canonical
    registrable = final.resolve() == canonical.resolve()
    if push and not registrable:
        raise TrainRefused(f"{final} is not outputs/adapters/{plan.name}, the folder the push sends and the "
                           "registry points to: leave out_dir out, or pass push=False")
    if push:
        _preflight()
    if (final / ADAPTER_WEIGHTS).is_file() and not retrain:
        return _resume(plan, final, push=push, parity=parity, parity_prompts=parity_prompts,
                       parity_dtype=parity_dtype, registry=registry, registrable=registrable)
    if final.exists() and not retrain:
        raise TrainRefused(f"{final} exists but holds no {ADAPTER_WEIGHTS}: not an adapter lift train "
                           "finished (it publishes the whole folder at once). Look at it, then rerun with "
                           "--retrain, which sets it aside")
    _check_same_hardware(plan)
    prepared = prepare(plan)
    data, n_heldout, unfit, tokenizer = prepared.data, prepared.n_heldout, prepared.unfit, prepared.tokenizer
    examples = trained_examples(prepared, drop_unfit=drop_unfit)
    if unfit:
        log.warning("%s: --drop-unfit drops %d rows the template leaves nothing to learn from: %s",
                    plan.name, len(unfit), [u["id"] for u in unfit])
    prompts, prompt_info = _parity_texts(parity_prompts) if parity else ([], None)
    stats = data_stats(examples)
    log.info("%s: %d rows of %s (sha256 %s), %d truncated at %d tokens, %d supervised tokens; %d held-out "
             "prompts checked", plan.name, stats["rows"], data.repo_path, data.sha256[:12],
             stats["truncated_rows"], prepared.max_len, stats["supervised_tokens"], n_heldout)
    from lift.models.load import device_name

    replaced = _set_aside(final, plan.name) if retrain else None
    gpu = device_name(plan.device)
    from lift.manifest import finish_run, start_run

    run = start_run("train", name=plan.name, candidate=plan.candidate, recipe=plan.recipe, seed=plan.seed,
                    data_seed=plan.data_seed, data_path=data.repo_path, data_sha256=data.sha256,
                    n_rows=len(data.rows), n_rows_trained=stats["rows"],
                    dropped_unfit={"count": len(unfit), "ids": [u["id"] for u in unfit]},
                    recipe_config=plan.cfg, lora_config=plan.cfg["lora"], device=plan.device, gpu=gpu,
                    dtype=plan.dtype, per_device_batch=plan.per_device_batch, grad_accum=plan.grad_accum,
                    max_steps=plan.max_steps, model_id=plan.model_id, model_rev=plan.model_rev,
                    retrained_from=replaced, devices=1)
    progress: dict[str, Any] = {}
    fit: dict[str, Any] | None = None
    try:
        scratch = tempfile.TemporaryDirectory(prefix=f"lift-train-{plan.name}-", ignore_cleanup_errors=True)
        with scratch as tmp:
            fit = _fit(plan, tokenizer, examples, Path(tmp), progress)
            _release()
            trained = _train_record(plan, data, n_heldout, stats, unfit, fit, replaced, gpu)
            _write_json(fit["staging"] / TRAINER_LOG, fit["trainer_log"])
            _write_json(fit["staging"] / LIFT_TRAIN, trained)
            _publish(fit["staging"], final)
        from lift.train.merge import adapter_hash

        digest = adapter_hash(final)
        log.info("%s: saved in %s, adapter hash %s", plan.name, final, digest)
        post = _post_steps(plan, final, digest, trained, push=push, parity=parity, prompts=prompts,
                           prompt_info=prompt_info, parity_dtype=parity_dtype, registry=registry,
                           replace=bool(replaced), registrable=registrable)
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__, **_so_far(fit, progress))
        raise
    finish_run(run, status=post.status, batch=fit["batch"], rows_per_step=fit["rows_per_step"],
               supervised_tokens_per_batch=fit["supervised_tokens_per_batch"],
               supervised_tokens_per_step=fit["supervised_tokens_per_step"], loss_curve=fit["loss_curve"],
               train_runtime=fit["metrics"].get("train_runtime"), train_seconds=fit["train_seconds"],
               peak_memory_bytes=fit["peak_memory_bytes"], lora=fit["lora"], data_stats=stats,
               adapter_hash=digest, parity=post.parity, push=post.pushes,
               registration=_registration_summary(post.registration))
    return _conclude(plan, final, post)


def _resume(plan: Plan, final: Path, *, push: bool, parity: bool, parity_prompts: Sequence[str] | None,
            parity_dtype: str, registry: Path | None, registrable: bool) -> Path:
    """An adapter folder that holds the weights: no training, only the steps still missing."""
    from lift.manifest import finish_run, start_run
    from lift.models.load import device_name
    from lift.train.merge import adapter_hash

    record = final / LIFT_TRAIN
    if not record.is_file():
        raise FileNotFoundError(f"{final} holds the adapter weights but no {LIFT_TRAIN}: an incomplete copy "
                                "(a partial lift sync pull?). Pull again, or rerun with --retrain")
    trained = _read_json(record)
    mine = {"candidate": plan.candidate, "recipe": plan.recipe, "seed": plan.seed}
    theirs = {k: trained.get(k) for k in mine}
    if theirs != mine:
        raise TrainRefused(f"{final} was trained as {theirs}, not {mine}: look at it, then rerun with "
                           "--retrain")
    digest = adapter_hash(final)
    reg = _registration_doc(final, digest, plan)
    needs_parity = parity and parity_dtype not in reg["parity"]
    prompts, prompt_info = _parity_texts(parity_prompts) if needs_parity else ([], None)
    log.info("%s is already trained (%s, adapter hash %s): not training again; running the steps still "
             "missing", plan.name, final, digest[:12])
    run = start_run("train", name=plan.name, candidate=plan.candidate, recipe=plan.recipe, seed=plan.seed,
                    resumed=True, adapter_hash=digest, device=plan.device, gpu=device_name(plan.device),
                    parity_dtype=parity_dtype, devices=1 if needs_parity else 0)
    try:
        post = _post_steps(plan, final, digest, trained, push=push, parity=parity, prompts=prompts,
                           prompt_info=prompt_info, parity_dtype=parity_dtype, registry=registry,
                           replace=bool(trained.get("retrained_from")), registrable=registrable)
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__)
        raise
    finish_run(run, status=post.status, adapter_hash=digest, parity=post.parity, push=post.pushes,
               registration=_registration_summary(post.registration))
    return _conclude(plan, final, post)


def _so_far(fit: Mapping[str, Any] | None, progress: dict[str, Any]) -> dict[str, Any]:
    """The counts a run that stopped had made: the finished training's, or the training under way's."""
    keys = ("batch", "rows_per_step", "supervised_tokens_per_batch", "supervised_tokens_per_step",
            "loss_curve", "train_seconds")
    if fit is not None:
        return {k: fit[k] for k in keys}
    counter, trainer = progress.get("counter"), progress.pop("trainer", None)
    return {"batch": progress.get("batch"), "rows_per_step": list(counter.rows_per_step) if counter else None,
            "supervised_tokens_per_batch": list(counter.per_batch) if counter else None,
            "supervised_tokens_per_step": list(counter.per_step) if counter else None,
            "loss_curve": _curve(trainer.state.log_history) if trainer is not None else None}


def _registration_summary(reg: Mapping[str, Any]) -> dict[str, Any]:
    return {"entry": reg.get("entry"), "registered": reg.get("registered"),
            "unregistered": reg.get("unregistered"), "error": reg.get("registry_error")}


def _conclude(plan: Plan, final: Path, post: _Post) -> Path:
    if post.status == "parity_failed":
        raise ParityFailed(post.message)
    if post.status == "registry_conflict":
        raise RegistryConflict(post.message)
    if post.registration.get("registered"):
        for line in _laptop_lines(plan.name):
            log.warning("%s", line)
    return final


def summary(directory: Path) -> list[str]:
    """What lift train prints at the end (hashes, counts and paths): the adapter, parity, registration.
    "registered" only while registration.json holds both the entry and where it was written; a
    registration a failed fp32 parity withdrew is named as such."""
    reg = _read_json(Path(directory) / REGISTRATION)
    _ok, parity = parity_verdict(reg.get("parity") or {})
    lines = [f"{reg['name']}: {directory}", f"  adapter hash {reg['adapter_hash']}",
             f"  merge parity: {parity}"]
    registered = reg.get("registered")
    if registered and reg.get("entry"):
        lines.append(f"  registered in {registered['registry']} ({registered['outcome']})")
        lines += [f"  {line}" for line in _laptop_lines(str(reg["name"]))]
    else:
        lines.append("  not registered" + (f": {reg['registry_error']}" if reg.get("registry_error") else ""))
        withdrawn = reg.get("unregistered")
        if withdrawn:
            was = withdrawn.get("was") or {}
            lines.append(f"  unregistered {withdrawn.get('when')}: {withdrawn.get('reason')} (an fp16 parity "
                         f"had registered it in {was.get('registry')}, {was.get('outcome')})")
    return lines


# --- lift train --check -------------------------------------------------------------------------------


def check_training(candidate: str, recipe: str, seed: int, *, model_id: str | None = None,
                   revision: str | None = None, device: str = "cuda:0", dtype: str | None = None,
                   per_device_batch: int | None = None, drop_unfit: bool = False) -> dict[str, Any]:
    """``lift train --check``: what the same request would train on, built exactly as ``train_lora``
    builds it (``prepare``), so a person can decide on ``--drop-unfit`` before a GPU run.

    Only Global's tokenizer is loaded, at the revision training uses (models.global, which must be
    pinned, unless ``model_id`` is given): no model, no GPU or precision check (``plan_training``'s
    ``hardware=False``), no push or preflight, no registry, no adapter folder, no manifest; nothing is
    written. The request and the training file are checked as for training, so what training refuses
    there raises TrainRefused here too; rows left without a supervised token are reported, not refused.
    Returns counts, ids and hashes, never text: the counts of every row of the file (``data_stats``),
    those rows (id, prompt tokens, limit), whether lift train would stop on them (it does without
    ``drop_unfit``), and the counts of the rows that train once they are dropped."""
    plan = plan_training(candidate, recipe, seed, model_id=model_id, revision=revision, device=device,
                         dtype=dtype, per_device_batch=per_device_batch, hardware=False)
    prepared = prepare(plan)
    kept = trained_examples(prepared, drop_unfit=True)
    return {
        "name": plan.name, "candidate": plan.candidate, "recipe": plan.recipe, "seed": plan.seed,
        "model": {"id": plan.model_id, "revision": plan.model_rev}, "max_seq_len": prepared.max_len,
        "data": {"path": prepared.data.repo_path, "key": prepared.data.key, "sha256": prepared.data.sha256,
                 "rows_in_file": len(prepared.data.rows), "heldout_prompts_checked": prepared.n_heldout},
        "counts": data_stats(prepared.examples), "unfit": [dict(u) for u in prepared.unfit],
        "drop_unfit": bool(drop_unfit), "stops": bool(prepared.unfit) and not drop_unfit,
        "kept": data_stats(kept),
    }


def _spread_text(spread: Mapping[str, Any]) -> str:
    return f"min {spread['min']}, mean {spread['mean']}, max {spread['max']}"


def check_lines(report: Mapping[str, Any]) -> list[str]:
    """What ``lift train --check`` prints (counts, ids, hashes and paths, never text)."""
    data, counts, model = report["data"], report["counts"], report["model"]
    limit, unfit, kept = report["max_seq_len"], report["unfit"], report["kept"]
    lines = [f"{report['name']}: lift train --check (the tokenizer the only thing loaded; nothing written)",
             f"  data: {data['path']} ({data['key']}, sha256 {data['sha256'][:12]}), {data['rows_in_file']} "
             f"rows; none equals one of the {data['heldout_prompts_checked']} held-out utility prompts",
             f"  tokenizer: {model['id']}@{model['revision'][:12]}; max_seq_len {limit}",
             f"  rows: {counts['rows']}; truncated at {limit} tokens: {counts['truncated_rows']}",
             f"  prompt tokens: {_spread_text(counts['prompt_tokens'])}",
             f"  answer tokens: {_spread_text(counts['answer_tokens'])}",
             f"  supervised tokens: {counts['supervised_tokens']} (per row: "
             f"{_spread_text(counts['supervised_per_row'])})"]
    if not unfit:
        lines.append(f"  rows with zero supervised tokens: none; lift train trains on all {kept['rows']} "
                     "rows")
        return lines
    lines.append(f"  rows with zero supervised tokens: {len(unfit)} (id, prompt tokens, limit)")
    lines += [f"    {u['id']}, {u['prompt_tokens']}, {u['limit']}" for u in unfit]
    then = f"{kept['rows']} rows, {kept['supervised_tokens']} supervised tokens"
    if report["stops"]:
        lines.append("  lift train stops on these rows. Remedies, each with a dated docs/decisions.md entry: "
                     "raise max_seq_len in configs/train.yaml, or rerun with --drop-unfit, which drops "
                     f"exactly these and trains on {then}")
    else:
        lines.append(f"  lift train --drop-unfit drops exactly these and trains on {then}")
    return lines
