# Running on Kaggle (backup GPU)

The primary GPU is a Colab A100 in bf16 (colab/README.md). Kaggle's free T4 x2 sessions are
overflow. A T4 has no bf16, so it runs fp16: its numbers count only while Gate 0's fp16-vs-bf16
check and the bf16 Gate 0 it compared against both pass for the same Global revision, and they are
never paired with bf16 numbers. Claude Code prints the exact `lift ...` lines; a person pastes them
into the runner notebook, `kaggle/runner.ipynb`, and starts it.

## One-time setup

1. **Accept the model licences.** On Hugging Face, open CohereLabs/tiny-aya-global and
   CohereLabs/tiny-aya-base and accept the terms. Do the same for CohereLabs/aya-expanse-8b, but
   only if the fallback judge is ever needed.
2. **Create the runs repo.** Make a private HF dataset repo, e.g. `<hf-user>/lift-runs`. It moves
   outputs between sessions and holds the raw generations, so it must stay private.
3. **Import the notebook.** In Kaggle, create a notebook, then File > Import Notebook and upload
   `kaggle/runner.ipynb` from the repo. Keep the notebook private.
4. **Add Kaggle secrets** (Add-ons > Secrets) and attach them to the runner notebook:
   - `HF_TOKEN`: a write token, so it can push to the runs repo. A read token is enough for the
     gated models and for a pull, but cell 2's check refuses it before any GPU work.
   - `GITHUB_TOKEN`: a fine-grained token with read-only Contents access on Knull/lost-in-finetuning.
   - `LIFT_RUNS_REPO`: e.g. `<hf-user>/lift-runs`.
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: for the frontier judge.
   - `LIFT_JUDGE_MAX_USD`: the frontier judge's spend cap.

   The first three are required; the others are needed only by frontier-judge jobs.
5. **Notebook settings.** Set Accelerator to GPU T4 x2 and turn Internet on. A CPU-only job, such
   as the data build, runs the same way with no accelerator.

## The runner notebook

A markdown cell on top repeats the usage and the T4 rule. Then come two code cells.

**Cell 1, setup, once per session.** Set `BRANCH` at its top, then run it.
- `BRANCH` must be a ref that has both P01 and P02: `main` (the default) once both are merged,
  and until then an integration branch with `p01-data` merged into `p02-harness`, pushed to
  GitHub. `p02-harness` alone has P01's prompt loader and `LangID` as stubs, so Gate 0 and every
  eval fail on it. A ref from before P02 stops in this cell: the bootstrap finds no
  `constraints.txt`.
- The clone is taken when the cell runs. After a commit to `BRANCH`, such as the pin step after
  Gate 0 (below), rerun cell 1 before the next job.
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
  - Every install uses `-c constraints.txt` (the repo root): the exact versions the harness was
    tested with, torch excepted, which has to match the driver.
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

- Before the first line, it runs `lift sync check --write`: the runs repo must exist and be
  private, and `HF_TOKEN` must be able to write to it. So a read token or a public repo fails the
  job before any GPU work, instead of failing every push hours later.
- Each line runs in order. A line starting with `lift` runs as `$LIFT_PY -m lift ...`, with the
  interpreter from `/kaggle/working/lift_env` and the secrets in its environment. Any other line
  runs as is, without them (for example `nvidia-smi`), and lines starting with `#` are skipped.
- Lines run in `/kaggle/temp/lift`, which holds `data/`, `outputs/` and `validation/` just as the
  repo root does on the laptop. A relative path such as
  `validation/keys/second_judge/stock_dev.txt` means the same file.
- The first failing line stops the job, with three exceptions. One is a push: see below. Another is
  a `lift generate`, `lift judge` or `lift run-dev` line that exits 3: it is done, but skipped some
  cells (stale, made in another precision, or judged by another judge) or had their AuditKIT record
  refused, and it lists each with its remedy (Troubleshooting). Its checkpoint push and the next
  lines still run, and the cell raises at the end. The third is a `lift sync` line that exits 3
  (below): the next lines still run too. Any other line's exit 3 stops the job.
- After every line that succeeds (or exits 3, as above), `lift sync ...` lines excepted, the runner
  runs `lift sync push` as a checkpoint. It pushes once more in `finally`, even after a failure or
  an interrupt, so the job never needs its own push line.
- `lift sync` exits 0 when done, 1 on an error a retry may fix (not everything was transferred),
  2 on a problem no retry fixes, and 3 when it is done but some files conflicted (a push puts their
  local copies under `conflicts/<stamp>/` in the runs repo and a pull keeps them, so nothing is
  lost) or some judgements files or AuditKIT records do not pair with their cell's generations (the
  summary names the `lift restart` line that sets them aside).
- A `lift sync` command (the check, a pull or a push, a line in `JOB` included) that exits 1 is
  retried: 3 attempts over about 3 minutes. An exit 3 is reported and never retried, and after a
  `lift sync` line in `JOB` that exits 3, a pull's included, the next lines run. A committed
  version starts on an empty disk, so its pull meets every file in the runs repo that does not
  pair, and the `lift restart` line the pull's summary names, put right after `lift sync pull`
  (Habits), then runs.
- An exit 2 is a configuration problem: `HF_TOKEN` cannot read or write the runs repo, the repo
  is missing or public, a secret is missing, or `BRANCH` has no `lift sync`. A `lift sync` line in
  `JOB` also exits 2 for its own bad arguments (a typo, an inline comment, an `--only` path outside
  the synced trees, such as `output/metrics` for `outputs/metrics`); one inside them that matches
  nothing, such as `outputs/generation`, moves nothing and exits 0, so check its summary's counts.
  An exit 2 is never retried, and no further line runs. After the check's or a plain
  `lift sync push`'s (the runner's own, or such a line in `JOB`), no push runs either: the final
  push is that very command, and would fail the same way. After any other `lift sync` line's in
  `JOB` (a pull, `status`, a `check`, a push with more words), the final push still runs: an exit 2
  returns at once, so trying costs seconds.
- Apart from an exit 2, a push never stops the job: a failed one is reported, and the next push
  sends whatever is still missing. A pull that fails (exit 1 on every attempt, or stopped), like any
  failing line, ends the job; its exit 3 does not (above).
- At the end the cell raises if any line or push did not succeed, an exit 3 included, so a
  committed version shows as failed.

**The watchdog.** As on Colab (`colab/job.py`), every command runs in its own process group, with
its output copied to the cell as it comes. The limits are set at the top of cell 2:
- A line that prints nothing for 45 minutes (`MAX_SILENT_MINUTES`) is hung: it is stopped and
  fails, and the final push still runs. Raise the limit for a step that is silent longer by design.
- A `lift sync` command that runs longer than 30 minutes (`PUSH_LIMIT_MINUTES`) is stopped. A push
  is then tried again, as above, up to about 1.5 hours in all; a stopped pull, check or status is
  not, and ends the job like a failing line (the final push still runs). huggingface_hub sets no
  network timeout, so one stalled read would otherwise hold the session, and the week's GPU quota,
  until Kaggle's 12-hour limit.
- The final push gets one try after a stall when no line but `lift sync` has run in this kernel and
  `JOB` holds no `lift sync push` line (a committed run whose check or first pull stalled, say):
  this kernel made nothing to push. Running cell 2 again does not reset that, so after an earlier
  job in the same kernel a stalled final push is tried again as usual. Outputs a kernel restart left
  behind are still in `/kaggle/temp/lift`: to push them, run cell 2 with `JOB` set to
  `lift sync push`. Its final push is tried again after a stall, even when the check stalled first,
  and its own line runs once the check has passed.
- Kaggle saves the printed output with every version, so the cell prints `***` in place of any
  attached secret (8 characters or more) that a command prints.
- Stopping a command sends it SIGINT, so lift can close its files, and SIGKILL to its whole process
  group 30 seconds later (`KILL_GRACE_SECONDS`). Interrupting the cell stops the running command
  the same way, and the final push still runs.

Its own process group takes a command out of the kernel's, which a forced kernel restart or
shutdown kills. So that a running line never goes on as an orphan, holding the GPUs and writing to
`outputs/` while the next job starts, every command starts through a small shim that asks Linux to
SIGKILL it when the kernel dies (`PR_SET_PDEATHSIG`), then runs it unchanged: the same arguments,
environment, exit code and signal handling. **Interrupt cell 2 before restarting the kernel** all
the same: the interrupt stops the line with SIGINT and runs the final push, while a restart kills
the line outright and pushes nothing. Outputs since the last push then stay in `/kaggle/temp/lift`
until the next job's pushes, or until cell 2 runs with `JOB` set to `lift sync push` (after cell 1).

For anything over 20 minutes, use **Save Version > Save & Run All (Commit)**. It runs unattended
for up to 12 hours, and the session is billed for that whole time. When the 12-hour limit stops the
session, `finally` may not get to run, so count only on the pushes after completed lines. Split a
long stage into several lines.

## Jobs on a T4 x2

- **fp16 everywhere.** Every `lift generate`, `lift judge` and `lift run-dev` line takes
  `--dtype fp16`, and so does a `lift restart` line, as lift prints it here. The loader refuses
  bf16 on a GPU without native bf16, so a missing flag fails loudly. The dtype is part of the cache
  key, so fp16 outputs never mix with bf16 ones. These lines are for the T4 alone: on a GPU with
  native bf16 (compute capability 8 or more, such as Colab's A100 or L4), generate, judge and
  run-dev refuse any `--dtype` but bf16, since fp16 runs exist only for the T4's licence.
- **Leave Colab's bf16 cells out.** Output paths hold no dtype, so an fp16 run over a variant
  Colab started meets its bf16 cells, which are the reported ones. Give the line `--sets` and
  `--langs` that leave them out, in several lines if need be: the variant's files under
  `outputs/generations/` in the runs repo show which cells exist. A bf16 cell left in is refused
  as made in another precision, listed, and the line exits 3. Never restart such a cell from here:
  a restart would discard its bf16 outputs, so `lift restart --dtype fp16` refuses it, with nothing
  moved, unless `--force` is given, which is never right here.
- **Two GPUs.** A T4 x2 is two separate 16 GB devices. The generator runs on `cuda:0` and the judge
  on `cuda:1`: `--judge-device auto` (run-dev and smoke) and `--device auto` (judge) pick `cuda:1`
  when there are two GPUs. In `lift run-dev`, both work at the same time.

The first jobs go in this order:

1. **The bf16 Gate 0, on Colab** (colab/README.md). It prints the revision shas to pin, and its
   outputs, which the fp16 check compares against, go to the runs repo.
2. **The pin step**, after Gate 0 and before any `generate`, `judge` or `run-dev` line. A person
   copies the shas Gate 0 printed into `configs/project.yaml` with a dated `docs/decisions.md`
   entry, commits, and pushes to `BRANCH`. Those commands refuse a revision that is not a sha, and
   a session reads the `configs/project.yaml` of its own clone, so rerun cell 1 after the push.
3. **The fp16 Gate 0, here.** It compares fp16 outputs and primary-judge labels with the bf16 run
   on the same 200 prompts:

   ```
   lift sync pull
   lift smoke --compare-dtype fp16
   ```

   On a T4, a plain `lift smoke` (the bf16 Gate 0) is refused before any check, with exit 2 and
   nothing written: it runs on Colab. The fp16 run's record goes to `outputs/metrics/` (`.json`
   and `.md`) and to the runs repo:
   - `gate0-fp16.json`, PASS: T4 fp16 numbers may be reported for the Global revision it names,
     but only while the bf16 `gate0.json` it compared against (Colab, all checks run) passes for
     that revision too and is still that run, with at least 200 prompts compared, and only for
     cells made under the comparison's harness, decoding settings and GPU class (a T4, for the
     generator and the judge). A bf16 re-run therefore needs a new comparison, and a harness change
     needs the full bf16 Gate 0 run again on Colab first, then a new comparison. Without all of
     this, lift refuses to load fp16 labels, so `lift run-dev --dtype fp16` writes no AuditKIT
     record and exits 3.
   - `gate0-fp16.json`, FAIL with the decision "fp16 failed": a valid comparison missed the 95%
     rule, or the fp16 run was not finite. T4 runs are never reported, and everything runs in
     bf16 on Colab.
   - `gate0-fp16.json`, FAIL on another check (the model did not load, say): fix that and run the
     job again.
   - `gate0-fp16-partial.json`: no valid comparison was made, so nothing is decided. The bf16
     reference is not in the runs repo, has no passing `gate0.json` for this Global revision, or
     does not match (another `--n-precision`, fewer than 200 prompts, an older harness, missing
     labels), or one GPU had no room for the judge. Make sure the bf16 Gate 0's outputs and its
     passing `gate0.json` are in the runs repo (re-run its push, or Gate 0 itself, on Colab), then
     run this job again on a T4 x2: it pulls them first.
4. **Then, for example, the stock dev eval:**

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

   If Colab has made some of stock's cells already, add `--sets` and `--langs` that leave them
   out (above).

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
  `lift sync push` exits 3 and uploads the local copy under `conflicts/<stamp>/` in the runs repo,
  and `lift sync pull` exits 3 and keeps the local copy. Either way the job goes on, the cell raises
  at the end, nothing is lost, and the conflict can be settled from any machine. A pull also exits
  3 when judgements files or AuditKIT records do not pair with their cell's generations, and a
  committed version starts on an empty disk, so its pull meets every such file in the runs repo.
  The pull's summary names the `lift restart ... --judge ROLE` line that sets them aside: put it in
  `JOB` right after `lift sync pull`, as in the next habit. If a job dies, rerun it: generation and
  judging resume by cache key.
- **Finish a cell in one session when you can.** The venv installs torch fresh in every session, so
  a torch release between two sessions changes the stack. A finished cell is reused as it is, but
  `lift generate`, `lift judge` and `lift run-dev` refuse to append to a half-done cell under
  another stack (`StaleOutputError`), because one file never mixes stacks. They go on with the
  other cells, then exit 3 and print a `lift restart` line for each refused cell. Put that line
  in `JOB` right after `lift sync pull` and run the job once:

  ```
  lift sync pull
  lift restart --variant stock --split dev --sets harmful --langs hi --dtype fp16
  lift run-dev --variant stock --dtype fp16
  ```

  `lift restart` moves the cell's generations, judgements and AuditKIT records (with `--judge`,
  only that judge's judgements and records) under `outputs/.aside/`, which is never synced, and
  leaves empty files in their place; an empty record is retired, and readers skip it. The
  checkpoint push after it then replaces the runs repo's copies, and the job regenerates that cell
  and reuses the others. A restart without `--judge` also empties the other judges' judgements,
  which `lift run-dev` does not refill: its log names the line that does, such as
  `lift judge ... --judge second` with P04's `--keys` list (`--sample 0.10` when P04 has drawn
  none), to add to `JOB` after the line that regenerates the cell. When the cell's prompt text
  changed, P04 draws the key list again first (`lift validate second-judge-sample`, then
  `lift sync push`): the changed prompt has a new key, and the old list would exit 1. Take the restart line out of `JOB` afterwards: every run of it starts the
  cell over. Moving a file aside by hand does not work, because the job's own `lift sync pull`
  brings it back.
- **`lift restart` keeps another precision's cells.** `--dtype` names the precision the cell is
  made again in (bf16 when it is left out, so the T4's line carries `--dtype fp16`). A cell whose
  generations were made in another precision is refused, since a restart would discard them: the
  line exits 1 with nothing moved, and the job stops there. `--force` sets them aside all the same:
  never use it here on a bf16 cell (Jobs on a T4 x2).
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
- **`git clone of 'main' failed ...`.** Check that `BRANCH` exists on GitHub (push it first), and
  that `GITHUB_TOKEN` can still read the repo. Fine-grained tokens expire.
- **`no constraints.txt in ...`.** `BRANCH` predates P02. Set it to a ref with both P01 and P02
  and rerun cell 1.
- **`torch cannot use the GPU`.** Most likely, the venv's torch build is newer than the driver.
  `nvidia-smi` shows the CUDA version the driver supports. Run
  `import os; os.environ["UV_TORCH_BACKEND"] = "cu126"` (a version at or below the driver's) in a
  new cell, delete `/kaggle/temp/lift/venv`, and rerun cell 1. If there is no venv (the image's
  Python is 3.12 or newer), the image's torch is broken or pip replaced it: report it.
- **`refusing: ... inside /kaggle/working`.** Cell 1's dirs were edited. Point them back under
  `/kaggle/temp`.
- **`run the setup cell (cell 1) first`.** The kernel restarted (or `%reset` ran), so `SECRETS` or
  the environment is gone. Rerun cell 1. Next time, interrupt cell 2 before restarting the kernel
  (The watchdog, above).
- **`... exited 2: a configuration problem ...`.** `lift sync`'s own message, just above, says
  which. HTTP 401 or 403, or a read token: attach a write token as `HF_TOKEN`. HTTP 404 or "not
  private": fix `LIFT_RUNS_REPO`, or make the repo private. `HF_DEBUG` set in the kernel: unset it.
  `invalid choice: 'sync'` (or `'check'`): `BRANCH` predates P02. Then rerun cell 1, which reads
  the secrets and clones `BRANCH` again. A `lift sync` line in `JOB` with bad arguments (a typo,
  an inline comment, an `--only` path outside the synced trees, such as `output/metrics`) exits 2
  too: fix the line. An `--only` path inside them that matches nothing (`outputs/generation`) is no
  error: the line moves nothing and exits 0, so check the counts in its summary. After the check's
  or a plain `lift sync push`'s exit 2 no further push ran, so the outputs since the last push are
  only in `/kaggle/temp/lift`: once it is fixed, and before the session ends, run cell 2 with `JOB`
  set to `lift sync push`.
- **A line exits 3 and lists skipped cells.** `lift generate`, `lift judge` and `lift run-dev`
  finish the other cells, then list each skipped cell with its remedy. The job went on with the
  next lines.
  - A stale cell, started under another stack (torch, transformers or the GPU class): put its
    `lift restart` line in `JOB` right after `lift sync pull`, run the job once, then take it out
    (Habits).
  - A cell made in another precision (bf16, by Colab): leave it out with `--sets` and `--langs`,
    and never restart it from here (Jobs on a T4 x2).
  - A cell judged by another judge: run the line with the judge the message names (its id shows
    the dtype and mode), or start that judge's judgements over with the `lift restart ... --judge`
    line it prints, as for a stale cell.
  - A refused AuditKIT record: its error, above the list, says why; on a T4 it is most often the
    fp16 licence, which does not cover the cell (step 3 of the first jobs above). The judgements
    are written, so once that is fixed, the next run of the line writes the record.
- **`NotImplementedError: P01` in a line.** `BRANCH` lacks P01 (`p02-harness` alone). Use a ref
  with both P01 and P02 and rerun cell 1.
- **`unpinned revisions in configs/project.yaml ...`.** The pin step has not reached this
  session's clone. Push the pin commit to `BRANCH`, rerun cell 1, then run the job again.
- **`... failed on all 3 attempts`.** A `lift sync` command kept failing or kept stalling (the Hub
  or the network). Outputs since the last good push exist only in this session: before it ends,
  run cell 2 again with `JOB` set to `lift sync push`.
- **`... was stopped and is not tried again`.** A `lift sync` command stalled past
  `PUSH_LIMIT_MINUTES` (the Hub or the network). A stopped pull, check or status ends the job: run
  it again later. A stopped final push with "no line but lift sync has run in this kernel" had
  nothing this kernel made to send, but outputs an earlier kernel left (before a kernel restart) are
  not pushed either: once the Hub answers, run cell 2 with `JOB` set to `lift sync push`.
- **`... was stopped: no output for ...`.** The line printed nothing for `MAX_SILENT_MINUTES` and
  was most likely hung (a stalled download, a deadlock between the generator and the judge). Run
  the job again: generation and judging resume by cache key. If the step is silent that long by
  design, raise `MAX_SILENT_MINUTES` at the top of cell 2.
- **`... exited 3 (conflicts or unpaired files ...)`.** Files changed both here and on the Hub since
  the last sync, or judgements or AuditKIT records no longer pair with their cell's generations (a
  committed version starts on an empty disk, so its pull meets every such file in the runs repo,
  even one of another variant). The job went on with the next lines, and the cell raised at the end.
  Nothing is lost (see Habits); `lift sync status` on any machine shows both. For unpaired files,
  put the `lift restart ... --judge ROLE` line the summary names in `JOB` right after
  `lift sync pull`, then the line that judges the cell again, run the job once, and take the restart
  line out (Habits).
