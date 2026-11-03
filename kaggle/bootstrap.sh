#!/usr/bin/env bash
# Install the repo on a Kaggle GPU notebook. Run after the clone cell in kaggle/README.md:
#   bash /kaggle/working/lift/kaggle/bootstrap.sh
# Writes /kaggle/working/lift_env with LIFT_PY, the interpreter every later cell must use.
# Untested until the first Kaggle run (P02 verifies and fixes it).
set -euo pipefail

REPO=${LIFT_REPO_DIR:-/kaggle/working/lift}
EXTRAS="gpu,data,eval,report"

if python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "Python $(python -V) is new enough; installing into the image"
  pip install -q -e "${REPO}[${EXTRAS}]"
  LIFT_PY=$(command -v python)
else
  # SafeTune needs Python 3.12. Build a separate 3.12 env (torch is reinstalled there).
  echo "Python $(python -V) is too old for SafeTune; creating a 3.12 venv with uv"
  pip install -q uv
  uv venv /kaggle/working/venv --python 3.12
  VIRTUAL_ENV=/kaggle/working/venv uv pip install -q -e "${REPO}[${EXTRAS}]"
  LIFT_PY=/kaggle/working/venv/bin/python
fi

# torchao on some images breaks SafeTune/TRL imports (SafeTune's own Colab note).
"${LIFT_PY}" -m pip uninstall -y -q torchao 2>/dev/null || true

echo "LIFT_PY=${LIFT_PY}" > /kaggle/working/lift_env
"${LIFT_PY}" -m lift info
