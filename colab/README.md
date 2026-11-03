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

## Running a job

From the repo root, with the venv active (`.venv\Scripts\Activate.ps1`):

```powershell
lift remote --gpu A100 --job "lift sync pull; lift smoke"                     # Gate 0
lift remote --gpu A100 --job "lift sync pull; lift run-dev --variant stock"   # stock dev eval
```

What it does:

1. Checks the options (`--gpu` must be A100, L4 or T4: the CLI turns an unknown value into an
   A100), the login and the secrets, before anything is called.
2. Notes the VMs that `colab sessions` lists without a record (`[?]` lines), so that a new one can
   be told apart later. Then `colab new` creates the VM under a fresh name, such as
   `lift-20261104-091500-ab12`. Its endpoint goes straight into
   `outputs/manifests/remote-<name>.json`.
3. Uploads the secrets as a file (see below), then starts `colab exec -f colab/job.py` in the
   background. On the VM, job.py fetches the repo at `--ref`, installs `.[gpu,data,eval]`, and
   runs the lines in order, stopping at the first failure.
4. Every `--poll` seconds (default 45), downloads `/content/lift_job.log` to
   `outputs/logs/remote-<name>.log` and prints the new lines.
5. The job pushes outputs with `lift sync push` after every line (except sync lines and the last),
   and once more at the end, also after a failure. A trailing `lift sync push` line is dropped as
   redundant. A push that exits 1 (an error) is tried again, 4 tries in about 10 minutes. A push
   that exits 3 hit conflicts: the VM's copies went under `conflicts/` in the runs repo, so nothing
   is lost, and it is reported, not retried. The job's last log line is `@@LIFT_DONE {json}`, which
   carries its status.
6. If the job's final push still failed, `lift remote` tries once more itself. It uploads the
   secrets again and runs job.py in push mode, which runs one `lift sync push` and writes its own
   last line. If that fails too, it prints a loud warning that the outputs since the last
   successful push are lost.
7. Stops the VM, then checks with `colab sessions` that it is gone. This also happens on Ctrl+C,
   on any error, and after `--max-hours` (default 12).

Exit codes (also in `lift remote --help`):

| Code | Meaning |
| --- | --- |
| 0 | The job succeeded, its outputs are pushed, and the VM is released. A push that hit conflicts counts as pushed; `lift remote` says so. |
| 1 | A job line failed, or the job's status is unknown. |
| 2 | `lift remote` refused to start (bad options, no login, a missing secret). Nothing was created. |
| 3 | The job could not be set up: `colab new` or the secrets upload failed, or job.py's spec, secrets, python, fetch or pip step did. |
| 4 | The VM could not be confirmed released and may still be billing. This overrides every other code. `lift remote stop` returns 0 or 4. |
| 5 | The job ended, but its outputs could not be pushed, even by the retry from the laptop. |
| 130 | Interrupted (Ctrl+C). |

Afterwards, `lift sync pull` brings the results to the laptop.

**Ctrl+C, the cap and hung commands:**

- **Ctrl+C** asks the job to stop. `lift remote` uploads `/content/lift_job.stop`; within seconds
  job.py stops the running line (never a push), skips the rest and runs its final push. `lift
  remote` waits up to 15 minutes for the job's last line, then stops the VM. A second Ctrl+C stops
  the VM at once, and the outputs since the last successful push are then lost.
- **`--max-hours`.** The job stops itself the same way 30 minutes before the cap (a quarter of the
  cap when that is shorter), so its final push fits. At the cap, `lift remote` reads the log once
  more, then asks the job to stop and waits up to 15 minutes, as for Ctrl+C.
- **Hung commands.** job.py kills a line (or a setup step such as `pip install`) that prints nothing
  for `--max-silent-minutes` (default 45): it fails, and after a line the final push still runs. It
  kills a `lift sync push` that runs longer
  than 30 minutes (huggingface_hub sets no network timeout) and tries it again. Its heartbeat, every
  5 minutes, says how long the running command has been silent.

Notes:

- **Code version.** The VM runs `--ref` (a branch, tag or full commit sha; default `main`) from
  GitHub, not your working tree, so push first. Before P02 merges, use `--ref p02-harness`.
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
  when the job ends (`--no-stop` leaves it running), after one push retry if the job's final push
  failed. If the manifest says the VM is already released, it only prints the log as last
  downloaded; it never replaces a recorded job status with a weaker one. `lift remote stop <name>`
  stops the VM straight away, by its recorded endpoint even if the CLI has lost its own record.
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

The first cell installs the repo. The secrets stay in `ENV`, which only the lift commands get.

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
REF, REPO = "main", "/content/lift"
basic = base64.b64encode(f"x-access-token:{ENV['GITHUB_TOKEN']}".encode()).decode()

def run(*argv):
    if subprocess.run(list(argv)).returncode != 0:
        raise RuntimeError(f"{argv[0]} failed")  # never print argv: the fetch carries the token

run("git", "init", "-q", REPO)
run("git", "-C", REPO, "remote", "add", "origin", "https://github.com/Knull/lost-in-finetuning.git")
run("git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}",
    "-C", REPO, "fetch", "-q", "--depth", "1", "origin", REF)
run("git", "-C", REPO, "checkout", "-q", "--detach", "FETCH_HEAD")
run(sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[gpu,data,eval]")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
```

The second cell runs the job with job.py's loop. Command output goes to `/content/lift_job.log`,
never into the notebook (a saved notebook can be shared by accident). It pushes after every line
except sync lines and the last, so a closed tab or a reclaimed runtime loses one line's work at
most. A failed checkpoint push is reported, not fatal. A push that exits 1 is tried again, the final
push runs even after a failure, and the cell raises at the end if a line or the final push failed.
Don't add your own trailing `lift sync push`.

```python
import shlex, subprocess, sys, time

JOB = """
lift sync pull
lift smoke
"""
LOG = "/content/lift_job.log"
PUSH_DELAYS = (60, 180, 360)  # as colab/job.py: 4 tries in about 10 minutes for a push that exits 1

def lift(*args):
    with open(LOG, "a") as log:
        return subprocess.run([sys.executable, "-m", "lift", *args], cwd=REPO, env=ENV,
                              stdout=log, stderr=subprocess.STDOUT).returncode

def push(what):
    for delay in (*PUSH_DELAYS, None):
        rc = lift("sync", "push")  # 3: done, the conflicting copies kept under conflicts/ in the runs repo
        print(f"  {what} push rc {rc}", flush=True)
        if rc != 1 or delay is None:
            return rc
        time.sleep(delay)

lines = [x.strip() for x in JOB.splitlines() if x.strip() and not x.strip().startswith("#")]
failed = []
try:
    for i, line in enumerate(lines):
        words = shlex.split(line)
        assert words[0] == "lift", line
        print(time.strftime("%H:%M:%S"), line, flush=True)
        rc = lift(*words[1:])
        print("  rc", rc, flush=True)
        if rc != 0:
            failed.append(f"{line} (rc {rc})")
            break
        if i < len(lines) - 1 and words[1] != "sync" and push("checkpoint") not in (0, 3):
            print("  checkpoint push failed; continuing, the final push sends everything", flush=True)
finally:
    rc = push("final")
    if rc not in (0, 3):
        failed.append(f"the final push (rc {rc}): the outputs since the last push are only on this runtime")
    print("log:", LOG, flush=True)
if failed:
    raise RuntimeError("the job failed: " + "; ".join(failed))
```
