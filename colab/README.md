# Running on Colab (primary GPU)

Reported runs use a Colab A100 in bf16. The Google AI Pro plan gives 200 Colab compute units a
month. A plain A100 is 40 GB; turning on High-RAM gives the 80 GB card (`--high-mem` asks the CLI
for that shape; not yet checked on a real VM).

Rates from Colab's GPU price list, April 2024. Check the Resources panel after connecting,
since rates change:

| GPU | Units per hour | Hours from 200 units |
| --- | --- | --- |
| T4 | 1.19 | ~168 |
| L4 | 1.71 | ~117 |
| A100 40 GB | 5.40 | ~37 |
| A100 80 GB | 7.52 | ~26 |

For batched generation of a 3B model, an A100 should be roughly 5 times as fast as a T4. That
puts the project's ~40 T4-hours at about 8 to 10 A100-hours, or 45 to 75 units. Gate 0 measures
the real speed. Idle runtimes still burn units, so a VM is stopped as soon as its job ends.

Jobs are launched from the laptop with `lift remote`, which drives Google's Colab CLI
(`google-colab-cli` 0.7.5, command `colab`). The notebook path at the end is the fallback.

## One-time setup

1. **Accept the model licences** on Hugging Face: CohereLabs/tiny-aya-global and
   CohereLabs/tiny-aya-base.
2. **Create the private runs repo**: `hf repos create <hf-user>/lift-runs --type dataset --private`.
   It holds raw generations, so it must stay private.
3. **Make a GitHub token**: fine-grained, read-only Contents access on Knull/lost-in-finetuning.
4. **Install the CLI**, pinned: `uv tool install google-colab-cli==0.7.5`. It lands in
   `%USERPROFILE%\.local\bin\colab.exe`, which is not on PATH; `lift remote` finds it there.
5. **Log in once**, in an interactive PowerShell window. It prints a URL; sign in with the Google
   AI Pro account and paste the code back. This writes `~\.config\colab-cli\token.json`.

   ```powershell
   & "$env:USERPROFILE\.local\bin\colab.exe" sessions
   ```

6. **Set the secrets** in the PowerShell session that runs `lift remote` (never in the repo), each
   with a hidden prompt, never typed or pasted on a command line or into a chat:

   ```powershell
   $env:GITHUB_TOKEN = [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'GITHUB_TOKEN')).Password
   $env:LIFT_RUNS_REPO = "<hf-user>/lift-runs"
   # A write token, or `hf auth login` once, with a WRITE token:
   $env:HF_TOKEN = [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'HF_TOKEN')).Password
   # Frontier-judge jobs only: the key (or ANTHROPIC_API_KEY) and the spend cap.
   $env:OPENAI_API_KEY = [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'OPENAI_API_KEY')).Password
   $env:LIFT_JUDGE_MAX_USD = "<cap>"
   ```

   For Claude Code sessions, whose tool calls cannot prompt, set them once at user scope instead, in
   your own PowerShell window, then restart the Claude desktop app (the judge key and the cap only
   for frontier-judge jobs):

   ```powershell
   [Environment]::SetEnvironmentVariable('GITHUB_TOKEN', [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'GITHUB_TOKEN')).Password, 'User')
   [Environment]::SetEnvironmentVariable('LIFT_RUNS_REPO', '<hf-user>/lift-runs', 'User')
   [Environment]::SetEnvironmentVariable('HF_TOKEN', [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'HF_TOKEN')).Password, 'User')
   [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', [Net.NetworkCredential]::new('', (Read-Host -AsSecureString 'OPENAI_API_KEY')).Password, 'User')
   [Environment]::SetEnvironmentVariable('LIFT_JUDGE_MAX_USD', '<cap>', 'User')
   ```

   This form keeps each value in plain text in the registry (`HKCU\Environment`) until it is removed.
   When you no longer need one, remove it in your own PowerShell window with
   `[Environment]::SetEnvironmentVariable('NAME', $null, 'User')`.

   `hf auth login` must store a WRITE token for the runs repo too, not the read token that
   downloads the models: the job's pushes use it, and so does the push retry of
   `lift remote watch`.

7. **Check the runs repo**: `lift sync check --write` says whether it exists, is private and takes
   this token's writes. `lift remote` runs the same check before it creates any VM.

## Running a job

From the repo root, with the venv active (`.venv\Scripts\Activate.ps1`). The VM runs `--ref`, which
must hold both P01 and P02 (see "Code version" below). The first jobs go in this order:

1. **Gate 0.**

   ```powershell
   lift remote --gpu A100 --job "lift sync pull; lift smoke"
   ```

2. **Pin the revisions**, before any generate, judge or run-dev job. Those commands refuse a model
   or judge revision that is not a commit sha, and on the VM the refusal comes only after its setup
   (about 10 A100-minutes). Gate 0 ends by printing the three shas to pin; they are also under
   `revisions` in `outputs/metrics/gate0.json`, which `lift sync pull` brings to the laptop.
   - Copy them into `configs/project.yaml`: `models.global.revision`, `models.base.revision` and
     `judges.safety_primary.revision`.
   - Add a dated entry with the reason to `docs/decisions.md`.
   - Commit, and push to the ref the GPU job runs (`--ref`; Kaggle's `BRANCH`). The VM reads the
     ref's copy of `configs/project.yaml`, not your working tree's.

   `lift remote` (and `--dry-run`) warns when a job line would refuse a revision in your local copy,
   checked as that line's command checks it: `models.global` for generate and run-dev, and the entry
   of the judge a judge or run-dev line runs, when that judge is Qwen3Guard (a frontier judge has no
   revision). It cannot see the ref's copy, so pushing the pin is up to you.

3. **The stock dev eval**, then the later stages:

   ```powershell
   lift remote --gpu A100 --job "lift sync pull; lift run-dev --variant stock"
   ```

**Gate 0's files**, in `outputs/metrics/` (pushed with the job's outputs). Read the one for your
run, never an older one:

| File | Written by | What it is |
| --- | --- | --- |
| `gate0.json` and `gate0.md` | `lift smoke` (bf16, on the A100; a T4, which has no bf16, is refused with exit 2 and no report) | The verdict, PASS or FAIL. A failed re-run replaces an earlier PASS: a model that does not load, or any failed required check, is a full FAIL. A run on a reduced sample (`--n-precision` under 200 or `--n-dry` under 20) whose checks pass is INCOMPLETE and never replaces this file. |
| `gate0-fp16.json` and `.md` | `lift smoke --compare-dtype fp16` (on a T4) | The fp16-vs-bf16 verdict. A T4 number counts only while this says PASS, with at least 200 prompts compared, and the bf16 `gate0.json` it compared against says PASS too and is still that run (its precision digest), both for the same Global revision that made the cell, and only for cells made under the comparison's harness, decoding settings and GPU class (the generator's and the judge's), and judged by the judge it ran (the same model, revision, labelling rules and Controversial mode). So a bf16 re-run needs a new comparison, and a harness change or a change of the judge (a re-pin, new labelling rules, another mode) needs the full bf16 Gate 0 run again first, then a new comparison. `lift.judge.pipeline` refuses an fp16 cell otherwise, so pull both files before reading T4 labels. |
| `gate0-partial.json`, `gate0-fp16-partial.json` | either, when incomplete | `--skip` removed a required check, or a required check did not run (`"not_run": true`: no room for the judge on one GPU, or nothing valid to compare the fp16 run with: no passing bf16 `gate0.json` for the same Global revision, or a different number of prompts, or fewer than 200). It never passes and never replaces the full file. |

What it does:

1. Checks the options (`--gpu` must be A100, L4 or T4: the CLI turns an unknown value into an
   A100), the login and the secrets, before anything is called. Then it checks the runs repo from
   the laptop, as `lift sync check --write` does: missing, not private or not writable with this
   token refuses (exit 2), and a transient Hub error exits 1, both before any VM exists.
2. Notes the VMs that `colab sessions` lists without a record (`[?]` lines), so that a new one can
   be told apart later. Then `colab new` creates the VM under a fresh name, such as
   `lift-20261104-091500-ab12`. Its endpoint goes straight into
   `outputs/manifests/remote-<name>.json`.
3. Uploads the secrets as a file (see below), then starts `colab exec -f colab/job.py` in the
   background. On the VM, job.py fetches the repo at `--ref` and installs `.[gpu,data,eval]` with
   the ref's `constraints.txt` (`pip install -c constraints.txt -e ...`). It checks that the ref
   has `lift sync` (`lift sync --help`): a ref without `constraints.txt` or `lift sync` is a setup
   failure (exit 3), before any line. Then it runs the lines in order, stopping at the first
   failure. A generate, judge or run-dev line that exits 3 is not such a failure: see "Skipped
   cells" below.
4. Every `--poll` seconds (default 45), downloads `/content/lift_job.log` to
   `outputs/logs/remote-<name>.log` and prints the new lines.
5. The job pushes outputs with `lift sync push` after every line (except sync lines and the last),
   and once more at the end, also after a failure, once a line other than a `lift sync` line ran
   (nothing else leaves new files on the VM). A trailing `lift sync push` line is dropped as
   redundant; one mid-job runs as a push and never stops the job. `lift sync`'s exit code decides
   the rest:
   - **1** (an error): tried again, 4 tries in about 10 minutes. A `lift sync pull` line too.
   - **3** (done, with conflicts or unpaired files): conflicting copies went under `conflicts/` in
     the runs repo, so nothing is lost, and judgements or records that do not pair with their cell's
     generations are named in the summary with the `lift restart` line that sets them aside.
     Reported, not retried.
   - **2** (a problem no retry can fix: a token that cannot write, the runs repo missing or public,
     `LIFT_RUNS_REPO` unset, `HF_DEBUG` set): never retried. The job ends there; the final push is
     still tried once.

   The job's last log line is `@@LIFT_DONE {json}`, which carries its status. If the log cannot be
   written (a full disk), job.py prints that line instead and `lift remote` finds it in
   `remote-<name>.exec.log`.
6. If the job's final push still failed, `lift remote` tries once more itself, unless that push
   exited 2. It uploads the secrets again and runs job.py in push mode, which runs one
   `lift sync push` and writes its own last line. It does the same for a job that ended with no
   last line after running lines, but never on a VM the Colab CLI has lost (reclaimed, or its
   record gone), which no upload or exec reaches. A first Ctrl+C during the retry only says that it
   goes on; the second stops the VM at once. If the outputs still could not be pushed, it prints a
   loud warning that the outputs since the last successful push are lost.
7. Stops the VM, then checks with `colab sessions` that it is gone. This also happens on Ctrl+C,
   on any error, and after `--max-hours` (default 12).

Exit codes (also in `lift remote --help`):

| Code | Meaning |
| --- | --- |
| 0 | The job succeeded, its outputs are pushed, and the VM is released. A push that hit conflicts counts as pushed; `lift remote` says so. |
| 1 | A job line failed, or a generate, judge or run-dev line exited 3 (done, but some cells were skipped: the job ran its other lines, and its log lists each cell with its remedy); or the job's status is unknown; or the runs repo check met a transient error (nothing was created: try again). |
| 2 | `lift remote` refused to start: bad options, no login, a missing secret, a runs repo that is missing, not private or not writable with this token, or a job that another `lift remote` on this machine follows already (`watch` names its pid). Nothing was created or changed. |
| 3 | The job could not be set up: `colab new` or the secrets upload failed, or job.py's spec, secrets, python, fetch, pip or sync-check step did; or a `lift sync` call on the VM exited 2 before anything was lost. |
| 4 | The VM could not be confirmed released and may still be billing. This overrides every other code. `lift remote stop` returns 0 or 4. |
| 5 | The job ended, but its outputs could not be pushed: its final push failed, and so did the retry from the laptop (none is tried for a push that exited 2, or on a VM the Colab CLI has lost). |
| 6 | `watch` kept the VM running, and billing, on purpose: it holds outputs that no push sent. A push retry is due that its shell cannot run (secrets missing, the runs repo check failed, or the retry's push exited 2: the banner says what to set, then rerun `lift remote watch <name>`), or `--no-stop` kept it. `lift remote stop <name>` releases it, and the outputs with it. |
| 130 | Interrupted (Ctrl+C). |

Afterwards, `lift sync pull` brings the results to the laptop.

**Ctrl+C, the cap and hung commands:**

- **Ctrl+C** asks the job to stop. `lift remote` uploads `/content/lift_job.stop`; within seconds
  job.py stops the running line (never a push), skips the rest and runs its final push. A stop
  request also cuts short job.py's wait before another try of a checkpoint push or a pull, and
  ends those tries. `lift remote` waits up to 15 minutes for the job's last line, and up to 55
  while the job's log shows a push still running (job.py kills a stalled push only after 30
  minutes, then tries it again). Then it stops the VM. A second Ctrl+C stops the VM at once, and
  the outputs since the last successful push are then lost. If the stop request cannot be uploaded
  (the VM's proxy failing), it is tried again 15 and 30 s later, then every 2 minutes while
  `lift remote` waits, and the waits count from when it went up: a failed upload never stops the VM
  sooner. When nothing ran yet (a Ctrl+C before the job started, or before its log appeared),
  `lift remote` says that nothing was lost. While `lift remote` already waits for the job's final
  push (at the cap, or after an error) or runs its push retry, a first Ctrl+C only says that the
  wait goes on; the second stops the VM at once.
- **`--max-hours`.** The job stops itself the same way 30 minutes before the cap (a quarter of the
  cap when that is shorter), so its final push fits. At the cap, `lift remote` reads the log once
  more, then asks the job to stop and waits as for Ctrl+C, also when that request cannot be
  uploaded: the job has usually stopped itself by then and is pushing. A first Ctrl+C during that
  wait only says that it goes on; the second stops the VM at once.
- **Hung commands.** job.py kills a line (or a setup step such as `pip install`) that prints nothing
  for `--max-silent-minutes` (default 45): it fails, and after a line the final push still runs. A
  `lift sync pull` killed this way is not tried again: its line fails like any other, so a stalled
  Hub costs one silence limit of idle GPU, not four (a pull that exits 1 is still tried again). It
  kills a `lift sync push` that runs longer than 30 minutes (huggingface_hub sets no network
  timeout) and tries it again. Its heartbeat, every 5 minutes, says what runs and how long the
  running command has been silent.

**Skipped cells (exit 3).** `lift generate`, `lift judge` and `lift run-dev` skip a cell they must
not add to, go on with the other cells, and exit 3 at the end, listing each skipped cell with its
remedy (and each cell whose AuditKIT record was refused). job.py reports a line that exits 3, the
next lines run (they do not depend on those cells), and the job fails at the end: `lift remote`
exits 1. Any other failing line still ends the job. A cell is skipped when:

- **It is stale**: its file was made under another library stack, or from inputs that have changed
  since. The remedy is its restart line, such as
  `lift restart --variant stock --split dev --sets harmful --langs hi` (with `--judge second`, or
  `--judge primary`, when only that judge's file is stale). Rerun the job with each printed restart
  line right after `lift sync pull` (it moves the copies the pull brought), then the line that
  exited 3. For a `lift judge` line, the listing also names the `lift generate` line that must come
  first, and for a `--keys` line it says when P04's key list must be drawn again first (a changed
  prompt changes its key):

  ```powershell
  lift remote --gpu A100 --job "lift sync pull; lift restart --variant stock --split dev --sets harmful --langs hi; lift run-dev --variant stock"
  ```

- **It was made in another precision**: its generations were made in another dtype than this run's
  (fp16 on a T4, say). Run with `--dtype <that dtype>` on the kind of GPU that made them, or leave
  the cell out with `--sets`/`--langs`. This is never a restart, which would discard them.
- **Its judge differs**: its judgements file was made by another judge (another judge id, which
  names the labelling rules and, for Qwen3Guard, the dtype and mode:
  `{model}@{sha}#rules-<tag>#{dtype}#{mode}`; a frontier judge's id ends in `#rules-<tag>`). Run with
  that judge, or the printed `lift restart ... --judge ROLE` line. A file that differs only in its
  rules tag (the labelling rules changed since it was judged) can only be judged again: its message
  names only the restart line, and running with the old judge never finishes it.
- **It is short of its frozen split** (`lift generate`, `lift run-dev`): its prompt file lacks some
  of the split's frozen pids, most often translation failures that await a reader's verdict, and
  nothing is generated for it. Its remedy comes first, on the laptop, as its listing says: most
  often the language's failures sheet filled in, `lift data apply-spotchecks`, then
  `lift sync push`. The line exits 3 and the job goes on; a later job's `lift sync pull` brings the
  completed prompt file, and the same line makes the cell.
- **Its decoding went non-finite** (`lift generate`, `lift run-dev`): a NaN or +inf score, as an
  fp16 overflow gives, so nothing of that batch was written. A T4's fp16 cell is made in bf16 here
  instead: its `lift restart ... --dtype fp16` line first, right after `lift sync pull`, when its
  file holds fp16 records, then the line that failed without `--dtype`. In bf16 it is a fault to
  report (CLAUDE.md rule 9), never one to tune around.

`lift restart` moves the cell's files to `outputs/.aside/<UTC stamp>/` on the VM, which is never
synced, and leaves each one empty; it empties the cell's AuditKIT records too (an empty record is a
retired one). The checkpoint push after it replaces the runs repo's copies with the empty files (if
another machine pushed only files the restart emptied since this machine's last sync, the push keeps
the runs repo's copies and reports a conflict instead, which `lift sync push --force` settles by
completing the restart; if it pushed any other file of the cell, such as more generations after a
`--judge` restart, the summary names its own remedy: for those, `lift sync pull`, then
`lift sync push`), and
the line after it refills them. With `--judge ROLE` it does this only for that judge's file and
records. Its log names the `lift judge ... --judge ROLE` line (with P04's `--keys` for the second
judge) that refills another judge's file it set aside: `lift run-dev` refills only the primary's. It
refuses (exit 1, nothing moved) a cell whose generations were made in another precision than its
`--dtype` (default: the config's `generation.dtype`, bf16), since a restart would discard them;
`--force` restarts such a cell anyway.

**fp16 runs.** `lift generate`, `lift judge` and `lift run-dev` refuse a `--dtype` other than the
config's on a GPU with native bf16 (Colab's A100 or L4): fp16 runs exist only for the T4 licence
(`gate0-fp16.json`, in the table above).

Notes:

- **Code version.** The VM runs `--ref` (a branch, tag or full commit sha; default `main`) from
  GitHub, not your working tree, so push first. GPU jobs need a ref with both P01 and P02: `main`
  once both have merged, or until then an integration branch, `p01-data` merged into
  `p02-harness` and pushed under its own name. `main` before P02 has no `lift sync` and no
  `constraints.txt`, which job.py refuses as a setup failure (exit 3) before any line runs. So is
  a branch pushed without `constraints.txt` (`git commit -a` leaves a new file out): commit it.
  `p02-harness` alone passes that check but has P01's prompt loader and language ID as stubs, so
  Gate 0 and every eval fail at their first prompt. Kaggle's `BRANCH` and the notebook fallback's
  `REF` follow the same rule.
- **Job lines** are separated by `;` or newlines. Inside the double-quoted `--job`, quote
  arguments with single quotes. Start every job with `lift sync pull`, since the VM starts empty.
- **Dry run.** `--dry-run` prints the plan, with secrets masked, and calls nothing.
- **From Claude Code.** Start `lift remote` with `run_in_background`, because foreground commands
  stop at 10 minutes, and check its exit code (table above): a 4 means a VM may still be billing,
  and a 6 means `watch` kept one running on purpose, billing, with outputs on it that no push sent.
  A session never asks for or accepts a secret in the chat. Each tool call runs in a new shell, so
  it loads each name that `lift remote` lists as missing (an exit 2, or the exit-6 banner) from the
  user environment (step 6) in the same command line:
  `$env:HF_TOKEN = [Environment]::GetEnvironmentVariable('HF_TOKEN', 'User'); lift remote watch <name>`.
  If `lift remote` still names one, it is not set there: the session stops and asks the user to set
  it in their own terminal. After a 6, `lift remote stop <name>` instead gives the outputs up. Keep
  the laptop awake for the whole run.

## What happens to secrets

- **What is sent.** `HF_TOKEN`, `GITHUB_TOKEN`, `LIFT_RUNS_REPO`, plus `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY` and `LIFT_JUDGE_MAX_USD` when they are set.
- **How.** They are written to a temp file and sent with `colab upload` to
  `/content/.lift_secrets.json`. The local file is deleted when the upload ends, and job.py deletes
  the remote copy before doing anything else.
- **Never `--env` or code.** The CLI writes `--env` values and exec'd code in plain text into
  `~\.config\colab-cli\history\<session>.jsonl` and `colab.log`. So `--env` carries only the job
  spec (repo, ref, lines, name, the job's deadline and its silence limit).
- **On the VM.** The GitHub token reaches git only as an http header passed with `git -c`, never in
  a URL. The secrets go into lift's environment, not git's or pip's. Every secret value is masked as
  `***` in the job log, and `lift remote` masks again before printing.
- **colab.log.** It holds each VM's short-lived proxy token. Don't share it.

## Watching, stopping, recovering

- **Files.** `outputs/manifests/remote-<name>.json` (endpoint, status, timings),
  `outputs/logs/remote-<name>.log` (the job log) and `remote-<name>.exec.log` (the heartbeats and
  the CLI's own messages).
- **After a laptop sleep or crash.** First check whether the original `lift remote` is still
  running, in its window: after a sleep it carries on by itself. One `lift remote` process at a
  time follows a job (it holds `outputs/manifests/remote-<name>.lock`, which the OS lets go of when
  that process ends, a crash included), so `watch` is refused with exit 2 and that process's pid
  while another one follows. Otherwise `lift remote watch <name>` prints the log again and stops the
  VM when the job ends, after one push retry if the job's final push failed (not with exit 2). That
  retry runs with watch's own shell: it needs a WRITE token for the runs repo in `HF_TOKEN` (or from
  `hf auth login`, logged in with a WRITE token, not the read token that downloads the models) and
  `LIFT_RUNS_REPO`, which a new PowerShell window lacks unless step 6 set them at user scope;
  `GITHUB_TOKEN` is not needed. Before the retry, watch checks the runs repo as a launch does
  (`lift sync check --write`). When the secrets are missing (watch says so when it starts), that
  check fails, or the retry's push exits 2, watch keeps the VM, with the outputs on it, and prints a
  banner that names what to set (exit 6): then run watch again, or `lift remote stop <name>` to
  give the outputs up. A VM the Colab CLI has lost (reclaimed, or its record gone) gets no retry,
  since nothing reaches it: watch stops it, by its endpoint if need be, and outputs that a failed
  final push left on it are reported lost (exit 5). `--no-stop` leaves the VM running: a failed
  final push then exits 6, not 5, since the outputs are still on the VM, and
  `lift remote watch <name>` without `--no-stop` retries the push, then stops it. If the manifest
  says the VM is already released, watch only prints the log as last downloaded and returns the
  recorded code; it never replaces a recorded job status with a weaker one, and it reads the record
  again before a push retry and before its last write, so what another process recorded meanwhile
  (a release, the job's status, a retry that pushed) is kept. An
  unexpected error in watch is handled as in a launch: the job is asked to stop and push, and the
  VM is still stopped (not with `--no-stop`). `lift remote stop <name>` stops the VM straight away,
  by its recorded endpoint even if the CLI has lost its own record; once it confirms the release,
  the recorded code becomes the run's own. A recorded 4 or 6 is answered, and outputs that watch
  left on the VM after a failed final push go with it: the record says 5, and stop prints the
  warning that they are lost.
- **A VM without a recorded endpoint.** This happens with no manifest (the job was launched from
  another worktree) or when `colab new` failed or timed out. A `[?]` line in `colab sessions` may
  then be this job's VM, so `lift remote` never reports it released. It prints the alarm with each
  `[?]` endpoint instead (after a failed `colab new`, only the new ones, also kept in the manifest
  as `orphan_endpoints`). Check it, then run `lift remote stop <name> --endpoint <endpoint>`.
- **Checking by hand.** Run `& "$env:USERPROFILE\.local\bin\colab.exe" sessions`. A `[?]` line is a
  VM the CLI has no record of, and it is still billed. The last resort is the browser:
  colab.research.google.com > Runtime > Manage sessions > Terminate.
- **Don't trust these.** `colab status` is local bookkeeping only. `colab stop` exits 0 even for a
  name it doesn't know. `lift remote` prints a loud warning whenever it cannot confirm the VM is
  gone.
- **Only lift remote stops the VM.** If the laptop sleeps, crashes or closes, nothing stops it after
  the job ends. Run `watch` or `stop` as soon as you can.

## Windows caveats

- **Support.** The CLI officially supports Linux and macOS only. On Windows, `new`, `upload`,
  `exec`, `download`, `stop` and `sessions` were checked against fake servers, not against real
  Colab. Don't run `colab update --install`, which refuses on Windows. Upgrade on purpose with
  `uv tool install google-colab-cli==<version> --force`, and retest.
- **Encoding.** Without `PYTHONUTF8=1`, the CLI reads and prints in cp1252 and crashes on Hindi.
  `lift remote` sets it for every call, along with stdin from DEVNULL and a timeout. By hand, run
  `$env:PYTHONUTF8 = "1"` first.
- **Use PowerShell, not Git Bash.** Git Bash rewrites `/content/...` arguments into Windows paths.
- **Timeout cap.** `colab exec --timeout` above 4294967 overflows on Windows; `lift remote` caps it.
- **WSL alternative.** WSL Ubuntu is a supported OS. Install the CLI there
  (`uv tool install google-colab-cli==0.7.5`), log in there, then run the repo's `lift remote`
  from a Linux venv.

## Before the first long job: check that sessions stay alive

Multi-hour CLI sessions are unverified. The CLI has no keepalive; its README says only that the
backend keeps a session alive while the kernel is busy. Test this once on a cheap CPU runtime
(no `--gpu`) before relying on multi-hour A100 jobs. Keep the window open and the laptop awake
for the 3 hours:

```powershell
$colab = "$env:USERPROFILE\.local\bin\colab.exe"; $env:PYTHONUTF8 = "1"
& $colab new -s liveness-1
"import time`nfor i in range(36):`n    print('beat', i, flush=True)`n    time.sleep(300)" | & $colab exec -s liveness-1 --timeout 14400
& $colab stop -s liveness-1; & $colab sessions
```

If the beats stop early, or `colab sessions` drops the VM, split long work into shorter jobs. Each
job resumes by cache key.

## Notebook fallback

If the CLI won't cooperate, use a normal Colab notebook:

1. Set Runtime > Change runtime type to A100 GPU (High-RAM for the 80 GB card).
2. Add the secrets under the key icon (Colab secrets) and give the notebook access:
   `HF_TOKEN`, `GITHUB_TOKEN`, `LIFT_RUNS_REPO`, and the judge key and `LIFT_JUDGE_MAX_USD` if
   needed.
3. Run the two cells below. Colab Pro has no background execution, so keep the tab open. When the
   job ends, use Runtime > Disconnect and delete runtime.

The first cell installs the repo. `REF` must hold both P01 and P02 (see "Code version" above). The
secrets stay in `ENV`, which only the lift commands get.

```python
import base64, os, subprocess, sys
from google.colab import userdata

assert sys.version_info >= (3, 12), sys.version  # SafeTune needs 3.12
os.environ["GIT_TERMINAL_PROMPT"] = "0"
ENV = dict(os.environ, MPLBACKEND="Agg", HF_HUB_DISABLE_PROGRESS_BARS="1")
for name in ["HF_TOKEN", "GITHUB_TOKEN", "LIFT_RUNS_REPO", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
             "LIFT_JUDGE_MAX_USD"]:
    try:
        ENV[name] = userdata.get(name)
    except Exception:  # optional secrets may be absent
        pass
REF, REPO = "main", "/content/lift"  # a ref with P01 and P02
basic = base64.b64encode(f"x-access-token:{ENV['GITHUB_TOKEN']}".encode()).decode()

def run(*argv):
    if subprocess.run(list(argv)).returncode != 0:
        raise RuntimeError(f"{argv[0]} failed")  # never print argv: the fetch carries the token

run("git", "init", "-q", REPO)
run("git", "-C", REPO, "remote", "add", "origin", "https://github.com/Knull/lost-in-finetuning.git")
run("git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
    "-C", REPO, "fetch", "-q", "--depth", "1", "origin", REF)
run("git", "-C", REPO, "checkout", "-q", "--detach", "FETCH_HEAD")
run(sys.executable, "-m", "pip", "install", "-q", "-c", f"{REPO}/constraints.txt",
    "-e", f"{REPO}[gpu,data,eval]")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
```

The second cell runs the job with job.py's rules. Command output goes to `/content/lift_job.log`,
never into the notebook (a saved notebook can be shared by accident). It pushes after every line
except sync lines and the last, so a closed tab or a reclaimed runtime loses one line's work at
most. A pull or push that exits 1 is tried again, and a failed push is reported, not fatal: a
`lift sync push` line never stops the job. A sync call that exits 2 (a problem no retry can fix) is
never tried again and ends the job. A generate, judge or run-dev line that exits 3 (done, but some
cells were skipped, each listed in the log with its remedy) is reported, and the next lines run. The
final push runs even after a failure, once a line other than a `lift sync` line ran, and the cell
raises at the end if a line or the final push failed, or a line exited 3. Don't add your own
trailing `lift sync push`.

```python
import shlex, subprocess, sys, time

JOB = """
lift sync pull
lift smoke
"""
LOG = "/content/lift_job.log"
DELAYS = (60, 180, 360)  # as colab/job.py: 4 tries in about 10 minutes for a sync call that exits 1
PARTIAL = ("generate", "judge", "run-dev")  # exit 3: done, but some cells were skipped (see the log)

def lift(*args):
    with open(LOG, "a") as log:
        return subprocess.run([sys.executable, "-m", "lift", *args], cwd=REPO, env=ENV,
                              stdout=log, stderr=subprocess.STDOUT).returncode

def sync(*args):
    # lift sync: 0 done; 1 an error, tried again; 2 a problem no retry can fix; 3 done, with conflicts
    # (a push keeps this runtime's copies under conflicts/ in the runs repo)
    for delay in (*DELAYS, None):
        rc = lift("sync", *args)
        if rc != 1 or delay is None:
            return rc
        time.sleep(delay)

lines = [x.strip() for x in JOB.splitlines() if x.strip() and not x.strip().startswith("#")]
failed, new_outputs = [], False
try:
    for i, line in enumerate(lines):
        words = shlex.split(line)
        assert words[0] == "lift", line
        print(time.strftime("%H:%M:%S"), line, flush=True)
        new_outputs = new_outputs or words[1] != "sync"
        rc = sync(*words[2:]) if words[1:3] in (["sync", "pull"], ["sync", "push"]) else lift(*words[1:])
        print("  rc", rc, flush=True)
        if words[1] == "sync" and rc == 2:
            failed.append(f"{line} (rc 2: a problem no retry can fix, see the log)")
            break
        if words[1:3] == ["sync", "push"] or (words[1] == "sync" and rc == 3):
            continue  # a push line never stops the job: the next push sends what it missed
        if rc == 3 and words[1] in PARTIAL:  # the next lines do not depend on the skipped cells
            failed.append(f"{line} (rc 3: some cells were skipped, the log lists each with its remedy)")
        elif rc != 0:
            failed.append(f"{line} (rc {rc})")
            break
        if i < len(lines) - 1 and words[1] != "sync":
            rc = sync("push")
            print("  checkpoint push rc", rc, flush=True)
            if rc == 2:
                failed.append("a checkpoint push (rc 2: a problem no retry can fix, see the log)")
                break
finally:
    rc = sync("push") if new_outputs else 0
    if rc not in (0, 3):
        failed.append(f"the final push (rc {rc}): the outputs since the last push are only on this runtime")
    print("log:", LOG, flush=True)
if failed:
    raise RuntimeError("the job failed: " + "; ".join(failed))
```
