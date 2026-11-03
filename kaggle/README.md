# Running on Kaggle (backup GPU)

The primary GPU is a Colab A100 in bf16 (colab/README.md). Kaggle's free T4 x2 sessions are
overflow. A T4 has no bf16, so it runs fp16: its numbers count only if Gate 0's fp16-vs-bf16 check
passes, and they are never paired with bf16 numbers. Claude Code prints the exact `lift ...`
lines; a person pastes them into the runner notebook, `kaggle/runner.ipynb`, and starts it.

## One-time setup

1. **Accept the model licences.** On Hugging Face, open CohereLabs/tiny-aya-global and
   CohereLabs/tiny-aya-base and accept the terms. Do the same for CohereLabs/aya-expanse-8b, but
   only if the fallback judge is ever needed.
2. **Create the runs repo.** Make a private HF dataset repo, e.g. `<hf-user>/lift-runs`. It moves
   outputs between sessions and holds the raw generations, so it must stay private.
3. **Import the notebook.** In Kaggle, create a notebook, then File > Import Notebook and upload
   `kaggle/runner.ipynb` from the repo. Keep the notebook private.
4. **Add Kaggle secrets** (Add-ons > Secrets) and attach them to the runner notebook:
   - `HF_TOKEN`: a write token, so it can push to the runs repo.
   - `GITHUB_TOKEN`: a fine-grained token with read-only Contents access on Knull/lost-in-finetuning.
   - `LIFT_RUNS_REPO`: e.g. `<hf-user>/lift-runs`.
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: for the frontier judge.
   - `LIFT_JUDGE_MAX_USD`: the frontier judge's spend cap.

   The first three are required; the others are needed only by frontier-judge jobs.
5. **Notebook settings.** Set Accelerator to GPU T4 x2 and turn Internet on. A CPU-only job, such
   as the data build, runs the same way with no accelerator.

## The runner notebook

A markdown cell on top repeats the usage and the T4 rule. Then come two code cells.

**Cell 1, setup, once per session.** Set `BRANCH` (default `main`) at its top, then run it.
- It reads the attached secrets into the `SECRETS` dict and skips the unattached or empty optional
  ones silently. It never prints a value. If `HF_TOKEN`, `GITHUB_TOKEN` or `LIFT_RUNS_REPO` is
  missing, it stops and names the secret.
- The secrets never go into `os.environ`. Every child process gets `os.environ` without them and
  without Kaggle's client tokens (`KAGGLE_USER_SECRETS_TOKEN`, `KAGGLE_IAP_TOKEN`), with which any
  process could fetch every attached secret. So git, `bootstrap.sh` and the third-party build code
  that pip and uv run never see a secret; cell 2 gives the secrets to the `lift` commands only.
- It points `LIFT_DATA_DIR`, `LIFT_OUTPUT_DIR` and `LIFT_VALIDATION_DIR` at
  `/kaggle/temp/lift/{data,outputs,validation}`, outside `/kaggle/working` (see below).
- It shallow-clones the repo into `/kaggle/temp/lift/repo`. The token goes only into an HTTP
  header, set through `GIT_CONFIG_*` variables of that one git process, with
  `GIT_TERMINAL_PROMPT=0`. It never goes into the URL, the command line or `.git/config`. If the
  clone fails, the error shows git's message with the token redacted. Rerunning the cell deletes
  and re-clones only the clone.
- It runs `kaggle/bootstrap.sh`:
  - It first unsets the secrets and Kaggle's client tokens, in case a caller other than cell 1
    passes them.
  - If the image's Python is 3.12 or newer (SafeTune needs 3.12), it installs `.[gpu,data,eval]`
    into the image with the image's pip.
  - Otherwise, uv builds a Python 3.12 venv in `/kaggle/temp/lift/venv`. A uv venv has no pip, so
    every install and uninstall there goes through uv. `UV_TORCH_BACKEND=auto` makes uv pick the
    torch build that matches the NVIDIA driver.
  - It removes torchao, which breaks SafeTune and TRL imports on some images, and checks that it
    is gone.
  - On a GPU session, it stops if torch cannot use the GPU.
  - It refuses to continue if the data, output or validation dir lies inside `/kaggle/working`.
  - It writes `LIFT_PY=<interpreter>` to `/kaggle/working/lift_env`, then runs `lift info`.
    Check `data_dir`, `output_dir` and `gpus` in its output.

**Cell 2, the job.** Paste the lines into `JOB` and run the cell:

```python
JOB = """
lift sync pull
lift info
# paste the stage's commands here, one per line (on a T4, generate/judge/run-dev take --dtype fp16)
"""
```

- Each line runs in order. A line starting with `lift` runs as `$LIFT_PY -m lift ...`, with the
  interpreter from `/kaggle/working/lift_env` and the secrets in its environment. Any other line
  runs as is, without them (for example `nvidia-smi`), and lines starting with `#` are skipped.
- Lines run in `/kaggle/temp/lift`, which holds `data/`, `outputs/` and `validation/` just as the
  repo root does on the laptop. A relative path such as `validation/keys/second_judge.txt` means
  the same file.
- The first failing line stops the job. A push is the exception: see below.
- After every successful line except `lift sync ...` lines, the runner runs `lift sync push` as a
  checkpoint. It pushes once more in `finally`, even after a failure or an interrupt, so the job
  never needs its own push line.
- `lift sync push` exits 0 when everything is sent, 1 when not everything was sent, and 3 when
  some files conflicted: their local copies go under `conflicts/<stamp>/` in the runs repo, so
  nothing is lost. Every push, a `lift sync push` line in `JOB` included, is retried when it
  exits 1, with 3 attempts over about 3 minutes. An exit 3 is reported and never retried.
- A push never stops the job: a failed one is reported, and the next push sends whatever is still
  missing.
- At the end the cell raises if any line or push did not succeed, an exit 3 included, so a
  committed version shows as failed.

For anything over 20 minutes, use **Save Version > Save & Run All (Commit)**. It runs unattended
for up to 12 hours, and the session is billed for that whole time. When the 12-hour limit stops the
session, `finally` may not get to run, so count only on the pushes after completed lines. Split a
long stage into several lines.

## Jobs on a T4 x2

- **fp16 everywhere.** Every `lift generate`, `lift judge` and `lift run-dev` line takes
  `--dtype fp16`. The loader refuses bf16 on a GPU without native bf16, so a missing flag fails
  loudly. The dtype is part of the cache key, so fp16 outputs never mix with bf16 ones.
- **Two GPUs.** A T4 x2 is two separate 16 GB devices. The generator runs on `cuda:0` and the judge
  on `cuda:1`: `--judge-device auto` (run-dev and smoke) and `--device auto` (judge) pick `cuda:1`
  when there are two GPUs. In `lift run-dev`, both work at the same time.
- **Gate 0 first.** Gate 0's fp16 check compares fp16 outputs and primary-judge labels with the
  bf16 Gate 0 run of the same 200 prompts. The bf16 run must already be in the runs repo, so run
  it on Colab first. It writes `outputs/metrics/gate0-fp16.json` and `.md`:

  ```
  lift sync pull
  lift smoke --compare-dtype fp16
  ```

  If it fails, T4 runs are never reported, and everything runs in bf16 on Colab.
- **Then, for example, the stock dev eval:**

  ```
  lift sync pull
  lift run-dev --variant stock --dtype fp16
  ```

  or as separate steps:

  ```
  lift sync pull
  lift generate --variant stock --split dev --dtype fp16 --device cuda:0
  lift judge --variant stock --split dev --dtype fp16 --device cuda:1
  ```

## Where files live

Every notebook version saves `/kaggle/working` and the notebook's printed output, and a version can
be made public by accident. So raw generations, judge logs, validation sheets, adapters and the
clone stay out of `/kaggle/working`:

| Path | What |
| --- | --- |
| `/kaggle/temp/lift/repo` | the clone |
| `/kaggle/temp/lift/venv` | the Python 3.12 venv, made only when the image's Python is older |
| `/kaggle/temp/lift/data`, `outputs`, `validation` | `LIFT_DATA_DIR`, `LIFT_OUTPUT_DIR`, `LIFT_VALIDATION_DIR` |
| `/kaggle/working/lift_env` | one line, `LIFT_PY=<interpreter>`: the only file in `/kaggle/working` |

**Unverified.** Kaggle's starter notebook describes `/kaggle/temp` as not saved outside the
session, but we have not checked that on a committed version. On the first committed run, check
that the version's Output tab holds only `lift_env`. If `/kaggle/temp` is missing or too small,
change `ROOT` at the top of cell 1. The bootstrap's guard still refuses any dir inside
`/kaggle/working`. Model weights go to the Hugging Face cache (`~/.cache/huggingface` unless the
image sets `HF_HOME`), also outside `/kaggle/working`. This too is unverified on Kaggle.

`bootstrap.sh` reads `LIFT_VENV_DIR` and `LIFT_ENV_FILE`. Other notebook runtimes, such as Colab's
notebook fallback, can use the same cells. Set those two variables, then change `ROOT`, the
secrets getter, and the `lift_env` path in cell 2.

## Habits

- **Pull first.** Every job starts with `lift sync pull`, so it resumes from the runs repo instead
  of redoing work, and the runner pushes after every line. A push never puts a stale copy over a
  newer one. A file changed on both sides is either merged by key (generation, judgement and
  judge-cache JSONL) or reported as a conflict (lift.sync has the rules). On a conflict,
  `lift sync push` exits 3 and uploads the local copy under `conflicts/<stamp>/` in the runs repo:
  the job goes on, and the cell raises at the end. `lift sync pull` exits 3 and keeps the local
  copy; like any failing line, that stops the job before more GPU time is spent. Either way nothing
  is lost, and the conflict can be settled from any machine. If a job dies, rerun it: generation
  and judging resume by cache key.
- **Two GPUs.** Running the generator and the judge in the same session halves the cost per
  variant.
- **Quota.** Check the remaining GPU quota (30 h a week per account) before a long job.
  Rough costs:
  - final eval of one variant: about 1 to 1.5 h
  - dev sweep setting: about 0.4 h
  - fine-tune: about 1 to 1.5 h
- **Measure first.** Gate 0's throughput benchmark replaces those estimates on Nov 4.
- **Don't print raw outputs.** The `lift` commands log counts and keys, never texts. Keep it that
  way: printed output is saved with every notebook version.

## Troubleshooting

- **`attach these Kaggle secrets ...`.** Attach the named secrets (Add-ons > Secrets) and rerun
  cell 1.
- **`git clone of 'main' failed ...`.** Check that `BRANCH` exists, and that `GITHUB_TOKEN` can
  still read the repo. Fine-grained tokens expire.
- **`torch cannot use the GPU`.** Most likely, the venv's torch build is newer than the driver.
  `nvidia-smi` shows the CUDA version the driver supports. Run
  `import os; os.environ["UV_TORCH_BACKEND"] = "cu126"` (a version at or below the driver's) in a
  new cell, delete `/kaggle/temp/lift/venv`, and rerun cell 1. If there is no venv (the image's
  Python is 3.12 or newer), the image's torch is broken or pip replaced it: report it.
- **`refusing: ... inside /kaggle/working`.** Cell 1's dirs were edited. Point them back under
  `/kaggle/temp`.
- **`run the setup cell (cell 1) first`.** The kernel restarted (or `%reset` ran), so `SECRETS` or
  the environment is gone. Rerun cell 1.
- **`... exited 1 on all 3 attempts`.** A push kept failing (the Hub or the network). Outputs since
  the last good push exist only in this session: before it ends, run cell 2 again with `JOB` set to
  `lift sync push`.
- **`... exited 3 (conflicts ...)`.** Files changed both here and on the Hub since the last sync.
  Nothing is lost (see Habits); `lift sync status` on any machine shows the conflicts.
