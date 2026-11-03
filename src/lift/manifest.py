"""Run manifests: one JSON per run with what produced it and what it cost.

The cost section of the article (GPU-hours per check-up, per-token latency) is computed from these,
so every GPU stage must call ``start_run`` / ``finish_run``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import REPO_ROOT, ensure, manifests_dir

_TRACKED = ("torch", "transformers", "peft", "trl", "safetune", "curatorkit", "auditkit",
            "accelerate", "datasets", "fasttext")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions() -> dict[str, str]:
    found = {}
    for name in _TRACKED:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def gpu_info() -> list[str]:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return []


def start_run(stage: str, **fields: Any) -> dict[str, Any]:
    return {
        "run_id": f"{stage}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
        "stage": stage,
        "started": datetime.now(UTC).isoformat(),
        "_t0": time.monotonic(),
        "git": git_sha(),
        "config_digest": config.digest(),
        "python": platform.python_version(),
        "versions": versions(),
        "gpus": gpu_info(),
        **fields,
    }


def finish_run(run: dict[str, Any], **fields: Any) -> Path:
    run = dict(run)
    t0 = run.pop("_t0")
    run["finished"] = datetime.now(UTC).isoformat()
    run["wall_seconds"] = round(time.monotonic() - t0, 3)
    # Kaggle bills session hours whether one or both T4s are busy; the report derives device-hours
    # from wall_seconds and the "devices" field the stage records.
    run.update(fields)
    path = ensure(manifests_dir()) / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8", newline="\n")
    return path
