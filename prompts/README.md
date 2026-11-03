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
- **Run Claude Code in the desktop app, on a local folder.** The laptop can reach
  huggingface.co, Colab, Kaggle and Google Translate. Once P02 has wired up the Colab CLI, a
  session can launch GPU jobs itself with `lift remote --gpu A100 --job "..."`. Claude Code on
  the web can write and test code too, but its sandbox blocks huggingface.co unless you allow
  the domain in claude.ai settings.
- **ultracode** turns a prompt into a background workflow: many agents, then adversarial review.
  - It only works when you type the keyword in an interactive session (not with `claude -p`).
  - It uses xhigh effort and a lot of tokens, so use it for the big passes marked below.
  - Watch progress with `/workflows`. A workflow can't ask you questions mid-run, so the prompts
    contain every decision it needs.
- **Normal prompts:** run them at high effort. Type `carry out prompts/P04_validation.md` without
  the keyword.
- **One branch per prompt, one folder per parallel session.** In PowerShell:

  ```powershell
  Set-Location C:\dev\lost-in-finetuning
  git worktree add ..\lift-p01 -b p01-data
  git worktree add ..\lift-p02 -b p02-harness
  git worktree add ..\lift-p04 -b p04-validation
  ```

  In the desktop app, start one Code session per folder (`C:\dev\lift-p01` and so on) and paste
  that prompt's line. Each session ends by opening a PR. Merge in prompt order, and rebase a
  later PR if it conflicts (they touch different modules, so conflicts should be small).
- **Fixing GPU failures.** A GPU run will surface bugs the offline tests can't. Paste the
  traceback into a normal session on the same branch: "fix this, add a test that reproduces it".

## Run order

| # | File | Mode | Starts after | Runs alongside |
| --- | --- | --- | --- | --- |
| P01 | P01_data.md: prompt sets, translations, splits, SFT candidates | ultracode | now | P02 |
| P02 | P02_harness.md: loading, generation, judges, sync, Colab runner, Gate 0 | ultracode | now | P01 |
| R1 | R1_review.md: adversarial review of data and harness; the harness freezes after it | ultracode | P01 + P02 merged | |
| P03 | P03_train.md: LoRA training, utility, Belebele, Gates 1-2 | ultracode | R1 | P06 |
| P04 | P04_validation.md: blind sheets, agreement, judge decision (codes against the schemas, so it can start now) | normal | now | P01, P02 |
| P05 | P05_repairs.md: WiSE-FT, ReSta, CAST, dev sweep, selection | ultracode | P03 | P06 |
| P06 | P06_diagnosis.md: probes, refusal direction, ablation | normal | P02 | P03, P05 |
| R2 | R2_prefreeze.md: review repairs, selection and final-split guard | ultracode | P05 | |
| P07 | P07_final_report.md: final eval, stats, tables, figures, cost | ultracode | R2 | |
| P08 | P08_publish.md: HF artifacts, article, notebook, Space | normal | P07 | |
| R3 | R3_claims.md: every public number traced to a file | ultracode | P08 | |

## Schedule (solo, from Nov 3)

The official deadline is Nov 14 23:59 AoE (Nov 15 11:59 UK). We still aim for Nov 13, with Nov 14
23:59 UK as the hard stop. GPU times assume a Colab A100 at roughly 5x a T4. Gate 0 measures the
real figure.

| Day | Claude Code | GPU (Colab A100 unless noted) | You |
| --- | --- | --- | --- |
| Tue Nov 3 | P01, P02 and P04 in parallel, merge; R1 tonight if there's time | data build on the laptop (no GPU; translation takes about an hour) once P01 merges | redeem the HF credits; HF licences; private runs repo; secrets; judge API credits |
| Wed Nov 4 | R1 if not done, then P03 and P06 | Gate 0 (~15 min), stock dev eval (~10 min), the 3 pilot fine-tunes (~1 to 1.5 h total) | Hindi spot checks; send the zh and es spot-check sheets; pick the frontier judge model |
| Thu Nov 5 | P05 and P06 | pilot dev evals (~30 min); Gates 1 and 2; seeds 2-3 of the chosen recipe (~45 min) | validation round 1: you do 60 hi and 40 en; send 40 zh and 40 es; apply spot-check fixes |
| Fri Nov 6 | fixes from GPU runs | fit CAST on seed 1; dev sweep of 10 settings (~1 h); diagnosis on stock + seed 1 (~15 min) | import round 1 labels; `lift validate decide-judge` |
| Sat Nov 7 | R2, then P07 | `lift select`; tag `recipe-frozen-*` (Gate 3); start final evals | approve the selection output before tagging |
| Sun Nov 8 to Tue Nov 10 | fixes; P08 drafts with placeholders | final evals of 13 variants (~3 to 4 h, primary endpoint first); second-judge sample (API) | validation round 2 (60 hi, 20 each zh/es/en); utility blind check (60 Hindi pairs) |
| Wed Nov 11 | `lift stats`, `lift report` | stragglers only | Gate 4: results freeze |
| Thu Nov 12 | P08, then R3 | | HF uploads (`--yes`), make the repo public, article read-through |
| Fri Nov 13 | | | submit through the organisers' form |
| Sat Nov 14 | | | buffer; hard stop 23:59 UK |

**Compute.** The project needs roughly 40 T4-hours, which is about 8 to 10 A100-hours, or 45 to
75 Colab compute units out of the 200 a month. Kaggle's free T4s are overflow. A T4 number can
be reported only if Gate 0's fp16-vs-bf16 check passes, and never paired against a bf16 one.
Final evals go in priority order: stock, the adapter seeds and the frozen winner come first.

**If behind,** cut in this order: the Space, then signatures on seeds 2-3, then ar and th, then
the Aya Red-teaming check, then Belebele beyond hi and en. Never cut: the 3 seeds, the
attenuation baseline, the native validation samples, or the dev/final separation.

## Things only a person can do

- **Redeem the hackathon HF credits now.** They go to the first 800 people.
  - Your HF account needs the email you registered with, and a payment method on file.
  - You can link more than one email to one HF account.
- **Set up accounts and secrets:**
  - accept the Tiny Aya licences on HF;
  - create the private runs repo;
  - add the secrets for Colab and Kaggle;
  - log the Colab CLI in with the Google AI Pro account.
- **Buy API credits for the frontier judge** and set a spend cap (`LIFT_JUDGE_MAX_USD`).
- **Native labelling:**
  - Hindi (you): 120 items in two rounds, 50 translation spot checks per translated set, and 60
    utility pairs.
  - Chinese and Spanish (one reader each): 60 items each, plus the spot checks.
  - English (you): 60 items.
- **Approve gate decisions.** The rules are fixed in the README; you check they were applied.
- **Finish up:** tag the frozen recipe, run the HF uploads, make the repo public, and submit
  through the form the organisers email a day or two before the deadline.
