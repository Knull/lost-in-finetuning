#!/usr/bin/env bash
# Install the repo on a Kaggle notebook. Cell 1 of kaggle/runner.ipynb runs it right after the clone:
#   bash /kaggle/temp/lift/repo/kaggle/bootstrap.sh
# Writes LIFT_PY, the interpreter every later cell must use, to $LIFT_ENV_FILE
# (default /kaggle/working/lift_env). It needs no secret and never traces commands (no set -x).
set -euo pipefail
# pip and uv run third-party build code, so the runner's secrets and Kaggle's client tokens (with which
# any process can fetch every attached secret) are dropped before anything runs. Cell 1 passes none;
# another caller might.
unset HF_TOKEN GITHUB_TOKEN LIFT_RUNS_REPO OPENAI_API_KEY ANTHROPIC_API_KEY LIFT_JUDGE_MAX_USD \
  KAGGLE_USER_SECRETS_TOKEN KAGGLE_IAP_TOKEN

REPO=$(cd "$(dirname "$0")/.." && pwd)
# Kaggle saves /kaggle/working with every notebook version, so the venv lives under /kaggle/temp.
VENV=${LIFT_VENV_DIR:-/kaggle/temp/lift/venv}
ENV_FILE=${LIFT_ENV_FILE:-/kaggle/working/lift_env}
EXTRAS="gpu,data,eval"
# Every GPU runtime installs with the repo's constraints.txt: the exact versions the harness was tested
# with (torch excepted: it has to match the driver). A ref without it predates P02 and cannot run a job.
CONSTRAINTS="${REPO}/constraints.txt"
if [ ! -f "$CONSTRAINTS" ]; then
  echo "no constraints.txt in ${REPO}: BRANCH must be a ref with both P01 and P02 (kaggle/README.md)" >&2
  exit 1
fi

if python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "$(python -V) is new enough for SafeTune; installing into the image"
  LIFT_PY=$(command -v python)
  "$LIFT_PY" -m pip install -q -c "$CONSTRAINTS" -e "${REPO}[${EXTRAS}]"
  uninstall() { "$LIFT_PY" -m pip uninstall -y -q "$@"; }
else
  # SafeTune needs Python 3.12: build a 3.12 venv with uv (torch is reinstalled there). A uv venv
  # has no pip, so every install and uninstall in it goes through uv.
  echo "$(python -V) is too old for SafeTune; using a Python 3.12 venv in ${VENV}"
  python -m pip install -q uv
  [ -x "${VENV}/bin/python" ] || python -m uv venv -q --python 3.12 "${VENV}"
  LIFT_PY=${VENV}/bin/python
  # auto picks the torch build that matches the NVIDIA driver (CPU wheels when there is no GPU).
  UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-auto} \
    python -m uv pip install -q --python "$LIFT_PY" -c "$CONSTRAINTS" -e "${REPO}[${EXTRAS}]"
  uninstall() { python -m uv pip uninstall -q --python "$LIFT_PY" "$@"; }
fi

# torchao breaks SafeTune/TRL imports on some images (SafeTune's own Colab note).
has_torchao() { "$LIFT_PY" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("torchao") is None)'; }
if has_torchao; then
  uninstall torchao
  if has_torchao; then echo "could not remove torchao from ${LIFT_PY}" >&2; exit 1; fi
fi

# On a GPU session, a torch that cannot see the GPU would only fail once a job starts: stop here.
if nvidia-smi -L > /dev/null 2>&1; then
  "$LIFT_PY" - <<'PY'
import sys

import torch

n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}): {n} usable GPU(s)")
if n == 0:
    sys.exit("torch cannot use the GPU (a CUDA build newer than the driver?); see kaggle/README.md")
PY
fi

# Kaggle saves /kaggle/working with every notebook version, so no data, output or validation file
# (raw generations, judge logs, human labels) may live there.
"$LIFT_PY" - <<'PY'
import sys
from pathlib import Path

from lift import paths

working = Path("/kaggle/working").resolve()
inside = [str(p) for p in (paths.data_dir(), paths.output_dir(), paths.validation_dir())
          if p == working or working in p.parents]
if inside:
    sys.exit(f"refusing: {inside} would be saved with every notebook version (inside {working}); "
             "point LIFT_DATA_DIR, LIFT_OUTPUT_DIR and LIFT_VALIDATION_DIR at /kaggle/temp")
PY

echo "LIFT_PY=${LIFT_PY}" > "${ENV_FILE}"
"$LIFT_PY" -m lift info
