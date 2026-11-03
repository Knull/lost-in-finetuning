# P02: Model harness, judges, sync and Gate 0

Branch `p02-harness` · ultracode · Opus 5.5 · starts now · runs alongside P01

## Goal

Turn a variant plus a prompt set into judged, resumable, cache-keyed records on a Kaggle T4 x2,
and run the Gate 0 checks. After this merges and R1's fixes land, the generation harness is
frozen: from then on it only takes bug fixes, and each fix means re-running whatever it touched.

## Read first

- CLAUDE.md, README.md, docs/toolkit-notes.md, docs/decisions.md
- src/lift/schemas.py, cachekey.py, io.py, manifest.py, judge/labels.py, stats/
- the stubs in src/lift/models/, judge/, train/merge.py, audit.py and sync.py: their docstrings
  are the spec

## Build

1. **Test fixtures first, in `tests/conftest.py`.**
   - `tiny_cohere2`: a randomly initialised `Cohere2ForCausalLM` from a small `Cohere2Config`
     (2 to 4 layers, hidden size 64, small vocab, tied embeddings, a sliding-window layer
     pattern).
   - A tiny tokenizer built offline in code with `tokenizers`. Its chat template mimics Tiny
     Aya's structure: special tokens, a default system preamble block, a user turn, then the
     chatbot turn and generation prompt.
   - `tiny_lora_adapter`: PEFT LoRA on the tiny model, saved to tmp.
   - Mark these tests `@pytest.mark.torch`, and skip cleanly when torch or transformers are
     missing.
2. **`models/load.py`**, as specified.
   - Resolve revisions to commit shas.
   - Support dtype fp16, fp32 and bf16.
   - Use left padding.
   - `chat_text` must equal `apply_chat_template([{"role": "user", "content": text}],
     tokenize=False, add_generation_prompt=True)` and keep the default preamble.
3. **`train/merge.py`**: `merged_model` (PEFT `merge_and_unload`), `adapter_hash` and
   `parity_check`.
4. **`models/variants.py`**: the stock and adapter variants, plus loading the registry from
   configs/variants.yaml. Leave a dispatch table by method so P05 can add the repairs without
   touching stock or adapter code.
5. **`models/generate.py`**, as specified.
   - Greedy decoding. Override any sampling defaults from the model's `generation_config.json`
     explicitly.
   - Stop on the model's end-of-turn tokens.
   - Batch by sorted length. On OOM, halve the batch and retry. Resume by key.
   - Record per-prompt seconds and `hit_cap`, and write a manifest with prompts/sec and
     tokens/sec.
   - If a prompt plus 256 new tokens exceeds the context, raise. Never truncate silently.
6. **`judge/qwen3guard.py`.**
   - Load on `cuda:1` and batch the calls.
   - Use exactly the response-moderation prompt format from the model card.
   - Parse strictly, with tests on canned outputs. A malformed output is an explicit failure,
     never a default label.
7. **`judge/frontier.py`, safety only.**
   - One small provider interface covering OpenAI and Anthropic. Take the provider and model
     from configs/project.yaml.
   - Use the rubric from configs/rubrics/safety_v1.md verbatim, at temperature 0.
   - Cache calls on disk; parse JSON strictly; retry with backoff.
   - Enforce the spend cap from `LIFT_JUDGE_MAX_USD`.
   - Ship a fake client for tests.
8. **`judge/pipeline.py`**, as specified.
   - GlotLID comes through `lift.lang.LangID`. P01 implements it: code against the interface
     and use a fake in tests.
   - Unclear flags and final labels come from `lift.judge.labels`.
   - `--judge second --sample 0.10` takes a seeded random 10% of generations, plus every item in
     the validation sheets once they exist.
9. **`audit.py`, `record_safety`.** Write an AuditKIT RunResult per variant, set, language and
   split from our labels. Check the installed auditkit API first; docs/toolkit-notes.md has the
   shape.
10. **`sync.py` and `lift sync push|pull`.**
    - Covers `data/prompts`, `data/sft`, `data/interim/translations`, `outputs/`, and
      `validation/{sheets,keys,labels}`.
    - The target is the private HF dataset repo in `LIFT_RUNS_REPO`. Refuse to run if the repo
      is public.
11. **`lift/commands/run.py`.**
    - `lift smoke`, Gate 0, GPU only. It writes `outputs/metrics/gate0.json` plus a markdown
      summary and prints PASS/FAIL per check with the README's decision rule. It runs:
      - (a) fp16 vs fp32 on 200 dev prompts, mixed across languages and sets: any non-finite
        hidden state or logit in fp16, the max absolute activation per layer, greedy output
        agreement, and agreement of the primary judge's labels between the two runs;
      - (b) a throughput benchmark at real prompt lengths, batch sizes 8, 16 and 32: prompts/sec,
        tokens/sec and peak memory;
      - (c) an end-to-end dry run of 20 prompts x 6 languages x 3 sets through generate and
        judge;
      - (d) the preamble's token length.
    - `lift generate --variant NAME --sets ... --langs ... --split dev|final [--limit N] [--device cuda:0]`
    - `lift judge --variant NAME --sets ... --langs ... --split ... [--judge primary|second] [--sample F] [--keys FILE] [--device cuda:1]`.
      `--keys` judges exactly the generation keys listed in a file. P04 uses it so the second
      judge covers every validated item.
    - `lift run-dev --variant NAME`: generate, then judge, every dev safety set in all 6
      languages, with the generator on cuda:0 and the judge on cuda:1 working at the same time,
      so a T4 x2 session uses both GPUs.
12. **Kaggle.** Check `kaggle/bootstrap.sh` as far as you can offline, and add
    `kaggle/runner.ipynb` implementing the two cells in kaggle/README.md.

## Tests (offline, CPU, fp32, on `tiny_cohere2`)

- Batched greedy output equals unbatched output (left padding and attention masks correct).
- After a simulated crash, resuming writes no duplicates and the same keys.
- Keys differ across variants and across prompt texts.
- `hit_cap` is set when the cap is hit.
- The OOM path retries with smaller batches (mocked).
- A merged adapter matches the unmerged one on the tiny adapter.
- Judge pipeline end to end with a fake judge and fake LID: every generation gets exactly one
  judgement.
- The Qwen3Guard parser, including malformed outputs.
- The frontier JSON parser, cache and spend cap.
- Sync refuses a public repo (mocked hub).

## Done when

- `pytest` and `ruff check src tests` are clean.
- `lift smoke --help`, `lift generate --help`, `lift judge --help` and `lift sync --help` work.
- A PR from `p02-harness` to `main` is open.

## Hand back

End your final message with:
1. The Gate 0 Kaggle job and the stock dev-eval job (T4 x2, Internet on) as runner-cell command
   lists. Data must already be built by P01 and synced.
2. How to read gate0.json, and what each failure means (fall back to fp32, or move generation to
   an L4).
3. Anything in the spec that turned out wrong.

## Don't

- Strip or change the preamble, sample, or truncate prompts.
- Print raw generations in logs or test output. Log counts and keys.
- Load Tiny Aya in a test.
