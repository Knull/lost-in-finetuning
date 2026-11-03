# Running on Kaggle (backup GPU)

The primary GPU is a Colab A100 in bf16 (colab/README.md). Kaggle's free T4 x2 sessions are
overflow. A T4 runs fp16, so its numbers count only if Gate 0's fp16-vs-bf16 check passes, and
they are never paired with bf16 numbers. Claude Code prints the exact `lift ...` commands; a
person pastes them into the runner notebook and starts it.

## One-time setup

1. **Accept the model licences.** On Hugging Face, open CohereLabs/tiny-aya-global and
   CohereLabs/tiny-aya-base and accept the terms. Do the same for CohereLabs/aya-expanse-8b, but
   only if the fallback judge is ever needed.
2. **Create the runs repo.** Make a private HF dataset repo, e.g. `<hf-user>/lift-runs`. It moves
   outputs between sessions and holds the raw generations, so it must stay private.
3. **Add Kaggle secrets** (Add-ons > Secrets), and attach them to the runner notebook:
   - `HF_TOKEN`: a write token, so it can push to the runs repo.
   - `GITHUB_TOKEN`: a fine-grained token with read-only Contents access on Knull/lost-in-finetuning.
   - `LIFT_RUNS_REPO`: e.g. `<hf-user>/lift-runs`.
   - `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: for the frontier judge.
4. **Notebook settings.** Set Accelerator to GPU T4 x2 and turn Internet on.

## The runner notebook

The first cell is the same for every job:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for name in ["HF_TOKEN", "GITHUB_TOKEN", "LIFT_RUNS_REPO", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
    try:
        os.environ[name] = secrets.get_secret(name)
    except Exception:
        pass
os.environ["LIFT_DATA_DIR"] = "/kaggle/working/data"
os.environ["LIFT_OUTPUT_DIR"] = "/kaggle/working/outputs"

BRANCH = "main"
url = f"https://x-access-token:{os.environ['GITHUB_TOKEN']}@github.com/Knull/lost-in-finetuning"
subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, "/kaggle/working/lift"], check=True)
subprocess.run(["bash", "/kaggle/working/lift/kaggle/bootstrap.sh"], check=True)
```

The second cell holds the job: the commands Claude Code printed, run with the interpreter the
bootstrap chose.

```python
import subprocess, shlex
LIFT_PY = open("/kaggle/working/lift_env").read().split("=", 1)[1].strip()
JOB = """
lift sync pull
lift info
# paste the stage's commands here, one per line
lift sync push
"""
for line in JOB.strip().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    cmd = shlex.split(line)
    if cmd[0] == "lift":
        cmd = [LIFT_PY, "-m", "lift", *cmd[1:]]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd="/kaggle/working/lift")
```

For anything over 20 minutes, use **Save Version > Save & Run All (Commit)**. It runs unattended
for up to 12 hours, and the session is billed for that whole time.

## Habits

- **Pull and push.** Every job starts with `lift sync pull` and ends with `lift sync push`. If a
  job dies, rerun it: generation and judging resume by cache key.
- **Two GPUs.** The generator runs on `cuda:0` and the judge on `cuda:1`. Running both in the
  same session halves the cost per variant.
- **Quota.** Check the remaining GPU quota (30 h a week per account) before a long job.
  Rough costs:
  - final eval of one variant: about 1 to 1.5 h
  - dev sweep setting: about 0.4 h
  - fine-tune: about 1 to 1.5 h
- **Measure first.** Gate 0's throughput benchmark replaces those estimates on Nov 4.
- **Don't print raw outputs.** Keep raw harmful completions out of the notebook's printed
  output. Saved notebook versions can be made public by accident.
