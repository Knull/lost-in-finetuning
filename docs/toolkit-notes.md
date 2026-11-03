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

- **Flow.** `auditkit.evaluate(model=..., prompts=..., metrics=...)` goes
  Dataset -> Adapter -> Model -> Metrics -> RunResult. Any `list[str] -> list[str]` callable is
  a model.
- **Comparisons.** Pairwise and preference metrics include win rate and Elo. `compare_models()`
  also exists.
- **Bug: don't use `paired_bootstrap`** (`src/auditkit/_bootstrap.py` at 71316da). It resamples
  the uncentred paired differences, so its p-value is about 0.5 when every difference is
  positive, and it returns no intervals.
  - File an issue upstream with a minimal repro; it's on the submission checklist.
  - `lift.stats.paired_bootstrap` is the replacement, with a regression test for this case.
