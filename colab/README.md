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

6. **Set the secrets** in the PowerShell session that runs `lift remote` (never in the repo):

   ```powershell
   $env:GITHUB_TOKEN = "<the read-only token>"
   $env:LIFT_RUNS_REPO = "<hf-user>/lift-runs"
   $env:HF_TOKEN = "<a write token>"          # or run `hf auth login` once
   $env:OPENAI_API_KEY = "<key>"              # or ANTHROPIC_API_KEY: frontier-judge jobs only
   $env:LIFT_JUDGE_MAX_USD = "<cap>"          # spend cap for frontier-judge jobs
   ```

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
| `gate0.json` and `gate0.md` | `lift smoke` (bf16, on the A100; a T4, which has no bf16, is refused with exit 2 and no report) | The verdict, PASS or FAIL. A failed re-run replaces an earlier PASS: a model that does not load, or any failed required check, is a full FAIL. |
| `gate0-fp16.json` and `.md` | `lift smoke --compare-dtype fp16` (on a T4) | The fp16-vs-bf16 verdict. A T4 number counts only while this says PASS, with at least 200 prompts compared, and the bf16 `gate0.json` it compared against says PASS too, both for the same Global revision that made the cell. `lift.judge.pipeline` refuses an fp16 cell otherwise, so pull both files before reading T4 labels. |
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
   failure.
4. Every `--poll` seconds (default 45), downloads `/content/lift_job.log` to
   `outputs/logs/remote-<name>.log` and prints the new lines.
5. The job pushes outputs with `lift sync push` after every line (except sync lines and the last),
   and once more at the end, also after a failure, once a line other than a `lift sync` line ran
   (nothing else leaves new files on the VM). A trailing `lift sync push` line is dropped as
   redundant; one mid-job runs as a push and never stops the job. `lift sync`'s exit code decides
   the rest:
   - **1** (an error): tried again, 4 tries in about 10 minutes. A `lift sync pull` line too.
   - **3** (done, with conflicts): the VM's copies went under `conflicts/` in the runs repo, so
     nothing is lost. Reported, not retried.
   - **2** (a problem no retry can fix: a token that cannot write, the runs repo missing or public,
     `LIFT_RUNS_REPO` unset, `HF_DEBUG` set): never retried. The job ends there; the final push is
     still tried once.

   The job's last log line is `@@LIFT_DONE {json}`, which carries its status. If the log cannot be
   written (a full disk), job.py prints that line instead and `lift remote` finds it in
   `remote-<name>.exec.log`.
6. If the job's final push still failed, `lift remote` tries once more itself, unless that push
   exited 2. It uploads the secrets again and runs job.py in push mode, which runs one
   `lift sync push` and writes its own last line. It does the same for a job that ended with no
   last line after running lines. If the outputs still could not be pushed, it prints a loud
   warning that the outputs since the last successful push are lost.
7. Stops the VM, then checks with `colab sessions` that it is gone. This also happens on Ctrl+C,
   on any error, and after `--max-hours` (default 12).

Exit codes (also in `lift remote --help`):

| Code | Meaning |
| --- | --- |
| 0 | The job succeeded, its outputs are pushed, and the VM is released. A push that hit conflicts counts as pushed; `lift remote` says so. |
| 1 | A job line failed, or the job's status is unknown; or the runs repo check met a transient error (nothing was created: try again). |
| 2 | `lift remote` refused to start: bad options, no login, a missing secret, or a runs repo that is missing, not private or not writable with this token. Nothing was created. |
| 3 | The job could not be set up: `colab new` or the secrets upload failed, or job.py's spec, secrets, python, fetch, pip or sync-check step did; or a `lift sync` call on the VM exited 2 before anything was lost. |
| 4 | The VM could not be confirmed released and may still be billing. This overrides every other code. `lift remote stop` returns 0 or 4. |
| 5 | The job ended, but its outputs could not be pushed: its final push failed, and so did the retry from the laptop (none is tried for a push that exited 2). |
| 130 | Interrupted (Ctrl+C). |

Afterwards, `lift sync pull` brings the results to the laptop.

**Ctrl+C, the cap and hung commands:**

- **Ctrl+C** asks the job to stop. `lift remote` uploads `/content/lift_job.stop`; within seconds
  job.py stops the running line (never a push), skips the rest and runs its final push. A stop
  request also cuts short job.py's wait before another try of a checkpoint push or a pull, and
  ends those tries. `lift remote` waits up to 15 minutes for the job's last line, and up to 55
  while the job's log shows a push still running (job.py kills a stalled push only after 30
  minutes, then tries it again). Then it stops the VM. A second Ctrl+C stops the VM at once, and
  the outputs since the last successful push are then lost.
- **`--max-hours`.** The job stops itself the same way 30 minutes before the cap (a quarter of the
  cap when that is shorter), so its final push fits. At the cap, `lift remote` reads the log once
  more, then asks the job to stop and waits as for Ctrl+C.
- **Hung commands.** job.py kills a line (or a setup step such as `pip install`) that prints nothing
  for `--max-silent-minutes` (default 45): it fails, and after a line the final push still runs. A
  `lift sync pull` killed this way is not tried again: its line fails like any other, so a stalled
  Hub costs one silence limit of idle GPU, not four (a pull that exits 1 is still tried again). It
  kills a `lift sync push` that runs longer than 30 minutes (huggingface_hub sets no network
  timeout) and tries it again. Its heartbeat, every 5 minutes, says what runs and how long the
  running command has been silent.

**A stale cell.** `lift generate`, `lift judge` and `lift run-dev` refuse to add to a cell whose file
was made under another library stack, or from inputs that have changed since. The refusal names that
cell's restart line, such as `lift restart --variant stock --split dev --sets harmful --langs hi`
(with `--judge second`, or `--judge primary`, when only that judge's file is stale). The other cells
go on, and the line exits 1 at the end, listing every stale cell. Rerun the job with each printed
restart line right after `lift sync pull` (it moves the copies the pull brought), then the line that
failed:

```powershell
lift remote --gpu A100 --job "lift sync pull; lift restart --variant stock --split dev --sets harmful --langs hi; lift run-dev --variant stock"
```

`lift restart` moves the cell's files to `outputs/.aside/<UTC stamp>/` on the VM, which is never
synced, and leaves each one empty. The checkpoint push after it replaces the runs repo's copies with
the empty files, and the line after it refills them.

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
  stop at 10 minutes, and check its exit code (table above): a 4 means a VM may still be billing.
  Keep the laptop awake for the whole run.

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
- **After a laptop sleep or crash.** `lift remote watch <name>` prints the log again and stops the VM
  when the job ends, after one push retry if the job's final push failed (not with exit 2).
  `--no-stop` leaves the VM running: a failed final push then exits 1, not 5, since the outputs
  are still on the VM, and `lift remote watch <name>` without `--no-stop` retries the push, then
  stops it. If the manifest says the VM is already released, watch only prints the log as last
  downloaded and returns the recorded code; it never replaces a recorded job status with a weaker
  one. An unexpected error in watch is handled as in a launch: the job is asked to stop and push,
  and the VM is still stopped (not with `--no-stop`). `lift remote stop <name>` stops the VM
  straight away, by its recorded endpoint even if the CLI has lost its own record; once it confirms
  the release, the recorded code becomes the run's own. A recorded 4 is answered, and outputs that
  `--no-stop` left on the VM after a failed final push go with it: the record says 5, and stop
  prints the warning that they are lost.
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
never tried again and ends the job. The final push runs even after a failure, once a line other
than a `lift sync` line ran, and the cell raises at the end if a line or the final push failed.
Don't add your own trailing `lift sync push`.

```python
import shlex, subprocess, sys, time

JOB = """
lift sync pull
lift smoke
"""
LOG = "/content/lift_job.log"
DELAYS = (60, 180, 360)  # as colab/job.py: 4 tries in about 10 minutes for a sync call that exits 1

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
        if rc != 0:
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
