# Toolkit notes (verified against source on Nov 3, 2026)

These notes were read from the code, not the docs. Pin these versions: the tools change weekly.

| Tool | Version | Commit read |
| --- | --- | --- |
| SafeTune | 0.1.12 (PyPI), Python >= 3.12, transformers >=5.15,<6, trl <1.15 | 3901a9f (Nov 3) |
| CuratorKIT | 0.1.22 (PyPI), core deps pydantic/typer/tqdm/pyyaml | dd9e835 |
| AuditKIT | 1.1.3 (PyPI), zero required deps | 71316da |

## SafeTune

All of these are badged Faithful in `docs/reference/feature-map.md`.

### `apply_wise_ft(model, aligned, alpha=0.5, param_filter=None)`

Lives in `safetune.interventions.recover.wise_ft`.

- Computes `W = alpha * W_aligned + (1 - alpha) * W_model` for every parameter present in both.
  It mutates `model` and returns it.
- alpha is the weight on `aligned`. Our lambda (0 = stock, 1 = adapter) maps to
  `alpha = 1 - lambda`.

### `apply_resta(finetuned, *, base, aligned, alpha=1.0, param_filter=None, dare=False, ...)`

Lives in `safetune.interventions.recover.resta`.

- Computes `finetuned + alpha * (aligned - base)` one tensor at a time in fp32. It mutates
  `finetuned`.
- `device="cpu"` keeps the extra memory off the GPU when base and aligned are on CPU.
- DARE: `dare` defaults to False in 0.1.12, with drop rate 0.3 when on. Pass `dare=False`
  explicitly anyway, because ReStaTrainer and older releases defaulted to DARE.
- Tied embeddings are handled (the last name wins).
- SafeTune's Tiny Aya notes: alpha around 0.25; alpha 1 breaks the model.

### `fit_cast_condition(model, harmful_prompts, benign_prompts, tokenizer, candidate_layers=None, *, pool="last", threshold_range=(-1, 1), threshold_step=0.01, val_frac=0.5, device="cpu", chat_template=None)`

Lives in `safetune.interventions.steer.cast`.

- The condition vector is the normalised difference of means between harmful and benign pooled
  hidden states.
- It grid-searches layer x threshold x comparator, picking by held-out F1 with a deterministic
  interleaved split.
- Default candidate layers are n/4 to 3n/4.
- `chat_template=None` means True: prompts are formatted as user turns, matching what
  `CASTModel.generate` sees.
- Returns a `CASTCondition` (condition_vector, condition_layer, threshold, comparator, f1, pool).

### `CASTModel(model, steering_vectors, condition=cond, alpha=1.0, per_prompt_gate=None)`

- `.generate(...)` runs a condition-only prefill pass per prompt. If the gate fires it installs
  the CAA vectors and re-runs; otherwise the input passes through unchanged.
- `per_prompt_gate=None` means True, so each prompt in a batch is gated on its own.
- `with cast: model.generate(...)` installs the vectors unconditionally. That is our
  gate-always-on setting.

### `extract_caa_vectors(model, tokenizer, positive_prompts, negative_prompts, config=None)`

Lives in `safetune.interventions.steer.caa`. Returns `{layer: vector}`.

### `extract_refusal_direction(model, tokenizer, harmful, harmless, config=None)`

Lives in `safetune.interventions.steer.refusal_direction`.

- Returns `(direction, layer_idx, all_layer_directions)`, taken as the difference of means at the
  last token.
- Its layer-selection sweep scores refusal with an English substring list (`_is_refusal`). That
  is wrong for Hindi, Chinese, Thai and the rest, so always pass
  `RefusalDirectionConfig(pick_layer=L)`, choosing L on stock (start at 24).
- `orthogonalize_weights(...)` and `restore_weights(model, snapshots)` do the weight-space
  ablation. `RefusalDirectionModel` does the hook version.

### Runtime config and gotchas

- Runtime config is `safetune.configure(...)`. Leave every `legacy_*` flag False.
- Default `dtype="auto"` means bf16 where supported, else fp16 on CUDA. Set dtypes explicitly on
  a T4.
- Decoder layers are found at `model.model.layers` (Llama/Qwen/Gemma/Cohere layout).
- Don't use `ReStaTrainer` or `RefusalDirectionTrainer`: no audit badge.
- On Colab/Kaggle, `pip uninstall -y torchao` if imports fail.

## CuratorKIT

No-API path:

```python
from curatorkit import Curator, CuratorConfig
result = Curator(CuratorConfig(
    dataset="data/interim/sft/aya_hi.jsonl",   # local path string, HF name, or {"name", "split", "subset"}
    dedup="minhash", clean=True,
    export_formats=["sft_messages"],
    output_dir="data/sft/aya/dedup",
)).run()
result.print_summary()
```

- **Outputs.** Every run writes `manifest.json`, `rejected.jsonl`, `dataset_card.md`,
  `lexsi_provenance.json` and `checksums.txt`. The output folder loads as an HF dataset with
  config `sft_messages`.
- **Install extras.** HF Hub sources need the `hf` or `connectors` extra.
- **Hygiene.** The hygiene extra (detoxify, presidio, spacy) is heavy and English-centric; don't
  apply the toxicity gate to Hindi.
- **Input format.** A local JSONL row in alpaca shape (`instruction`, `output`) or messages shape
  is auto-detected; check the detection in the run summary. Split options are under
  `output_split`.

## AuditKIT

Corrected Nov 3 (P02) after reading 1.1.3's source: the `evaluate(model=..., prompts=...,
metrics=...)` form written here before raises `TypeError` (there is no `prompts=` or `metrics=`).

- **Call.** `auditkit.evaluate(dataset, model, scorers=None, *, config=None, tags=None, **opts)`
  goes Dataset -> Adapter -> Model -> Metrics -> RunResult.
  - `dataset` is a `list[Sample]` (`Sample(input, id=..., metadata=..., actual_output=...)`);
    a list of strings or dicts raises.
  - `model` is a `Model`, a `list[str] -> list[str]` callable, or `PrecomputedModel` (reads each
    sample's `actual_output`, no model call). `lift.audit` uses a PrecomputedModel subclass with a
    stable name, so the run's fingerprint and `model_spec` hold no memory address.
- **Custom metrics** subclass `auditkit.metric.Metric` (not exported at the top level), must set
  `direction` at class level, and score one sample at a time; a stat's mean is the headline.
  - A metric that raises on a sample, or whose `required_fields` are missing, silently drops that
    sample from the denominator. `lift.audit` asserts `failed_count == 0`, every `stat.count == N`,
    and the headline equal to `lift.stats.rates()`.
- **Side effects.** Every run reads and writes an on-disk cache of the full RunResult (prompts and
  outputs) under `~/.cache/auditkit/runs`, keyed by a fingerprint that misses data held outside the
  samples: a second variant can be served the first one's numbers. `lift.audit` patches the cache out
  and never passes `experiment_name` (which writes another copy under `~/.local/share`).
- **Saving.** `RunResult.save()` writes CRLF and `\u` escapes on Windows plus a timestamped
  `lexsi_provenance.json`; `lift.audit` writes `to_dict()` itself (UTF-8, LF). Outputs are stored
  redacted (`[redacted sha256:...]`) so the record can be published.
- **Comparisons.** `WinRate`, `EloScore` and `PreferenceAccuracy` are reference- or overlap-based,
  not judge-based; P03's utility record needs a custom metric over our verdict scores.
- **Bug: don't use `paired_bootstrap`** (`src/auditkit/_bootstrap.py` at 71316da, unchanged in
  1.1.3). It resamples the uncentred paired differences: p is 1.0 when every difference is equal
  and about 0.5 when they vary but are all positive, and it returns no intervals.
  `RunComparison.significance()` and the other `significance()` methods use it too.
  - File an issue upstream with a minimal repro; it's on the submission checklist.
  - `lift.stats.paired_bootstrap` is the replacement, with a regression test for this case.

## Harness dependencies (verified Nov 3 for P02)

- **transformers 5.22.1.** `generate()` fills every generation setting a caller leaves `None` from
  the checkpoint's generation_config.json, and logits processors such as `repetition_penalty` act
  in greedy mode too. `lift.models.load` therefore replaces the model's generation_config with a
  neutral one, and the harness passes an explicit greedy config. A prompt longer than the context
  only warns. `apply_chat_template(tokenize=True)` returns a BatchEncoding, and `tokenizer(text)`
  adds a second BOS to chat text (always tokenize it with `add_special_tokens=False`).
- **Tiny Aya.** Gated with automatic approval: the HF account behind `HF_TOKEN` must accept the
  terms of both tiny-aya-global and tiny-aya-base. `<|START_RESPONSE|>` is part of the generation
  prompt; a system message never replaces the default preamble, it is appended as a "Developer
  Preamble". generation_config.json is byte-identical to Base's.
- **Qwen3Guard-Gen-4B** (main = 6ec42827, Apache-2.0). Pass exactly `[user, assistant]` to
  `apply_chat_template(..., tokenize=False)`: the template writes the moderation instruction and its
  own assistant prefix. Its tokenizer pads right by default (set left). The output is three lines
  (`Safety:`, `Categories:`, `Refusal:`); the card's regex parser and the official eval's substring
  check both default silently on malformed output, so `lift.judge.qwen3guard` parses strictly.
- **huggingface_hub 2.4.** `snapshot_download`/`hf_hub_download` return stale local files with only
  a warning when the Hub is unreachable, `DatasetInfo.private` can be `None`, and `upload_folder`
  mutates the caller's `ignore_patterns` list. With hf_xet, `upload_folder` also re-sends unchanged
  files and splits large uploads into several commits of which only the first carries
  `parent_commit`, so a concurrent push can be silently reverted. `lift.sync` therefore writes with
  `create_commit` (only the files it planned, every part guarded by `parent_commit`), verifies every
  download's hash, and refuses a repo unless `private is True`.
- **Colab CLI 0.7.5.** See colab/README.md. `colab run --env` writes values into local history and
  `colab.log`, so secrets go through `colab upload` of a temp file only; `colab exec` exits 0 even
  when the code fails; `--timeout` above 4294967 overflows on Windows. The default auth is oauth2
  (the CLI's own README says adc); `colab stop` exits 0 for a session name it does not know; an
  unknown `--gpu` silently becomes A100.
- **OpenAI SDK 3.43 / Anthropic SDK 2.4.** Anthropic's API version 2026-10-21 (the SDK default)
  rejects `temperature`, `top_p` and `top_k`; its 529 `OverloadedError` is not an
  `InternalServerError` subclass; both SDKs log full request bodies at DEBUG (the judge pins their
  loggers to WARNING); OpenAI's Responses API stores responses unless `store=False`.
