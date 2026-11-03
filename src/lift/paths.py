"""Where things live. Data and outputs can be moved with env vars (Kaggle: /kaggle/working/...).

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


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
