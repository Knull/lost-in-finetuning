# R1: Adversarial review of the data pipeline and harness

Branch `r1-review` · ultracode · Opus 5.5 · after P01 and P02 are merged, before the stock dev
eval. The harness freezes after this review.

## Goal

Find the bugs that would silently corrupt results, fix them, and stop when two review rounds in
a row find nothing new.

## How to review

- Reviewers work independently. Each one reads the code and the spec (README, CLAUDE.md,
  docs/decisions.md, the module docstrings), not the PR descriptions, so the review isn't
  anchored on what the authors believed.
- Every finding needs a failing test that reproduces it before the fix lands.
- A finding that can't be shown with a test goes in the report as "suspected" with the reason.

## Post-merge items to close first

P01, P02 and P04 were merged on Nov 4. The merge itself already:
- fixed two stale tests (P02's fake GlotLID predates P01's Hindi label folding; P04's file check
  predates P02's isolated test tree);
- switched the judge pipeline to GlotLID's top 5 (P01 decisions item 43).

Still open, each with a failing test first:
1. P04's judgement readers must skip, or refuse, a judgement whose output hash (P02's field) no
   longer matches the generation under the same key.
2. P01's `lift data spotcheck-sheets` and `apply-spotchecks` must write and read under
   `paths.validation_dir()` (`LIFT_VALIDATION_DIR`), as P04's sheets do, so remote runtimes work.
3. `tests/test_kaggle.py::test_setup_cell_clones_with_the_token_only_in_a_header` fails when the
   parent environment already has `GIT_CONFIG_*` variables (some sandboxes set them). Isolate the
   test's environment; check whether the setup cell should also strip inherited `GIT_CONFIG_*`.

## Check at least these

- **Data.**
  - PolyRefuse field mapping: `instruction` vs `instruction_translated`, and the train/val files
    whose `instruction` is already translated.
  - Row alignment, and the pid scheme.
  - Splits identical across languages and exactly the locked sizes.
  - No overlap between the utility held-out set and either SFT pool, near-duplicates included.
  - Translations made from English only, and translation failures raised rather than kept.
- **Prompting.**
  - The chat template is applied once: no doubled BOS or system block.
  - The preamble is present and `add_generation_prompt` is set.
  - Left padding and attention masks are correct.
  - An over-long prompt raises rather than being truncated.
- **Generation.**
  - Greedy really is greedy: no sampling defaults leaking in from `generation_config.json`.
  - `max_new_tokens` is 256.
  - EOS handling covers every end-of-turn token Tiny Aya uses.
  - `hit_cap` is right.
  - Resume doesn't duplicate rows.
  - Outputs don't depend on batch order or batch size, within fp tolerance.
- **Cache keys.**
  - Every behaviour-changing input is included.
  - Revisions are resolved to commit shas.
  - The adapter hash is of the right file.
  - The template hash changes if the preamble changes.
  - The dtype is in `gen_cfg`, so bf16 and fp16 outputs can never share a key. Each record's
    `meta` carries the dtype and the GPU name.
- **Runners.** `colab/job.py` always pushes outputs, even after a failure. Secrets never reach
  the repo, a log or a notebook output. A dropped session resumes cleanly.
- **Judges.**
  - The Qwen3Guard prompt format matches the model card exactly, and the parser is strict.
  - The strict and loose mappings both work.
  - Every generation is judged exactly once, and judge failures are counted.
  - The frontier judge uses the frozen rubric verbatim.
- **Labels and stats.**
  - The unclear rules follow docs/decisions.md item 4.
  - Rates keep every prompt in the denominator.
  - The McNemar direction is right (adapter minus stock).
  - The bootstrap clusters by pid.
- **Leakage.**
  - Nothing under `data/` (except splits) or `outputs/` is tracked by git.
  - No harmful completion is printed in logs, tests or notebooks.
  - The sync target is private.

## Done when

- `pytest` and `ruff` are clean.
- The PR has a findings table (severity, file, failing test, fix) and a list of the things you
  checked that were fine.

## Hand back

- Whether the harness is safe to freeze.
- Anything that should block the stock dev eval.
- The exact commit to treat as the frozen harness (`git tag harness-frozen-YYYYMMDD`).
