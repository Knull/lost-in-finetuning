"""Where things live. Data, outputs and validation files can be moved with env vars (the Kaggle
runner puts them under /kaggle/temp/lift/, outside the /kaggle/working that notebook versions save).

The frozen split id lists always live in the repo (``data/splits``) so they are versioned in git.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def _env_dir(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


def data_dir() -> Path:
    return _env_dir("LIFT_DATA_DIR", REPO_ROOT / "data")


def output_dir() -> Path:
    return _env_dir("LIFT_OUTPUT_DIR", REPO_ROOT / "outputs")


def raw_dir() -> Path:
    """Downloaded sources (PolyRefuse clone, HF snapshots)."""
    return data_dir() / "raw"


def interim_dir() -> Path:
    """Translation caches, CuratorKIT working folders."""
    return data_dir() / "interim"


def prompts_dir() -> Path:
    """Built prompt sets: ``{set}_{lang}.jsonl`` of PromptRecord."""
    return data_dir() / "prompts"


def sft_dir() -> Path:
    """CuratorKIT exports of the adapter training candidates."""
    return data_dir() / "sft"


def generations_dir() -> Path:
    return output_dir() / "generations"


def judgements_dir() -> Path:
    return output_dir() / "judgements"


def adapters_dir() -> Path:
    return output_dir() / "adapters"


def artifacts_dir() -> Path:
    """Repair artifacts (CAST vectors and conditions, direction tensors)."""
    return output_dir() / "artifacts"


def metrics_dir() -> Path:
    return output_dir() / "metrics"


def figures_dir() -> Path:
    return output_dir() / "figures"


def manifests_dir() -> Path:
    """One JSON per run: git sha, config digest, versions, GPU, wall time."""
    return output_dir() / "manifests"


def validation_dir() -> Path:
    """Human validation files (sheets, keys, labels). Private: gitignored, moved only by lift sync.
    ``LIFT_VALIDATION_DIR`` moves the tree, as in lift.validation (P04)."""
    return _env_dir("LIFT_VALIDATION_DIR", REPO_ROOT / "validation")


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
