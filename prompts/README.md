# Implementation prompts and schedule

Each file here is one Claude Code task. The prompt files hold the full spec and CLAUDE.md holds
the project context, so what you type stays short:

```
ultracode: carry out prompts/P01_data.md
```

## How to run them

- **Model: Opus 5.5 for every prompt.** Fable 5.1 carries extra safeguards for LLM R&D, and this
  project is exactly that: fine-tuning, refusal-direction ablation, and pipelines that generate
  and score harmful completions. It's more likely to stall or refuse partway through a workflow.
- **Run Claude Code locally, in a clone of this repo.** That machine can reach huggingface.co,
  Kaggle and Google Translate. Claude Code on the web can write and test the code too, but its
  sandbox blocks huggingface.co unless you allow the domain in claude.ai settings.
- **ultracode** turns a prompt into a background workflow: many agents, then adversarial review.
  - It only works when you type the keyword in an interactive session (not with `claude -p`).
  - It uses xhigh effort and a lot of tokens, so use it for the big passes marked below.
  - Watch progress with `/workflows`. A workflow can't ask you questions mid-run, so the prompts
    contain every decision it needs.
- **Normal prompts:** run them at high effort. Type `carry out prompts/P04_validation.md` without
  the keyword.
- **One branch per prompt, one working copy per parallel session:**

  ```bash
  git clone https://github.com/Knull/lost-in-finetuning && cd lost-in-finetuning
  git worktree add ../lift-p01 -b p01-data
  git worktree add ../lift-p02 -b p02-harness
  # terminal 1: cd ../lift-p01 && claude     terminal 2: cd ../lift-p02 && claude
  ```

  Each session ends by opening a PR. Merge in prompt order, and rebase the second PR if it
  conflicts (they touch different modules, so conflicts should be small).
- **Fixing Kaggle failures.** A Kaggle run will surface bugs the offline tests can't. Paste the
  traceback into a normal session on the same branch: "fix this, add a test that reproduces it".

## Run order

| # | File | Mode | Starts after | Runs alongside |
| --- | --- | --- | --- | --- |
| P01 | P01_data.md: prompt sets, translations, splits, SFT candidates | ultracode | now | P02 |
| P02 | P02_harness.md: loading, generation, judges, sync, Gate 0 | ultracode | now | P01 |
| R1 | R1_review.md: adversarial review of data and harness; the harness freezes after it | ultracode | P01 + P02 merged | |
| P03 | P03_train.md: LoRA training, utility, Belebele, Gates 1-2 | ultracode | R1 | P04 |
| P04 | P04_validation.md: blind sheets, agreement, judge decision | normal | P02 | P03 |
| P05 | P05_repairs.md: WiSE-FT, ReSta, CAST, dev sweep, selection | ultracode | P03 | P06 |
| P06 | P06_diagnosis.md: probes, refusal direction, ablation | normal | P02 | P05 |
| R2 | R2_prefreeze.md: review repairs, selection and final-split guard | ultracode | P05 | |
| P07 | P07_final_report.md: final eval, stats, tables, figures, cost | ultracode | R2 | |
| P08 | P08_publish.md: HF model, datasets, Space, article | normal | P07 | |
| R3 | R3_claims.md: every public number traced to a file | ultracode | P08 | |

## Schedule (solo, from Nov 3)

The original gates moved because nothing existed on Nov 3 (docs/decisions.md). GPU hours are
estimates until Gate 0 measures throughput.

| Day | Claude Code | Kaggle (GPU unless noted) | You |
| --- | --- | --- | --- |
| Tue Nov 3 | P01 and P02 in parallel, merge | data build (CPU session, Internet on; translation takes about an hour) once P01 merges | Kaggle secrets, HF licences, private runs repo, judge API credits; ask the readers for Nov 5 and Nov 8 |
| Wed Nov 4 | R1, then P03 and P04 | Gate 0 (`lift smoke`, ~1 h), stock dev eval (~0.5 h), then the 3 pilot fine-tunes overnight (~4.5 h) | send translation spot-check sheets (hi, zh, es); pick the frontier judge model |
| Thu Nov 5 | P05 and P06 | pilot dev evals (~2 h); Gates 1 and 2; launch seeds 2-3 of the chosen recipe (~3 h, unattended) | validation round 1 sheets out (60 hi, 40 each zh/es/en); apply spot-check fixes |
| Fri Nov 6 | fixes from Kaggle | fit CAST on seed 1; dev sweep of 10 settings (~4 h); diagnosis on stock + seed 1 (~1 h) | import round 1 labels; `lift validate decide-judge` |
| Sat Nov 7 | R2, then P07 | finish the sweep if needed; `lift select`; tag `recipe-frozen-*` (Gate 3); start final evals | approve the selection output before tagging |
| Sun Nov 8 to Tue Nov 10 | fixes; P08 drafts with placeholders | final evals of 13 variants (~13-20 h, primary endpoint first); second-judge sample (API) | validation round 2 (60 hi, 20 each zh/es/en) |
| Wed Nov 11 | `lift stats`, `lift report` | stragglers only | Gate 4: results freeze |
| Thu Nov 12 | P08, then R3 | | HF uploads (`--yes`), Space check, article read-through |
| Fri Nov 13 | | | submit |
| Sat Nov 14 | | | buffer; hard stop 23:59 UK |

**Compute.** The plan above needs roughly 40 T4-hours. One Kaggle account gives 30 a week, so
use a second account or HF credits on an L4 if a week runs short. Final evals go in priority
order: stock, the adapter seeds, and the frozen winner come first.

**If behind,** cut in this order (from the plan): signatures on seeds 2-3, then ar and th, then
the Aya Red-teaming check, then Belebele beyond hi and en. Never cut: the 3 seeds, the
attenuation baseline, the native validation samples, or the dev/final separation.

## Things only a person can do

- Accept the Tiny Aya licences on HF; create the private runs repo; add the Kaggle secrets.
- Buy API credits for the frontier judge and set a spend cap (`LIFT_JUDGE_MAX_USD`).
- Native labelling:
  - Hindi: 120 items in two rounds, 50 translation spot checks per set, 60 utility pairs.
  - Chinese and Spanish: 60 items each plus spot checks.
  - English: 60 items.
- Start Kaggle runs and approve gate decisions. The rules are fixed in the README; you check
  they were applied.
- Tag the frozen recipe, run the HF uploads, submit.
