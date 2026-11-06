"""``lift publish space``: stage hf/space/ (Gradio, no model loading) with the logs dataset it reads.

The Space reads only the public aggregate logs dataset; ``space_config.json`` names it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from lift import paths
from lift.publish import render, settings
from lift.publish.hub import Upload

FILES = ("app.py", "README.md", "requirements.txt")


def source_dir() -> Path:
    return paths.REPO_ROOT / "hf" / "space"


def build() -> Upload:
    folder = paths.output_dir() / "publish" / "space"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(source_dir() / name, folder / name)
    logs = settings.repo("logs")
    render.write_text(folder / "space_config.json", json.dumps({"logs_dataset": logs.repo_id}, indent=1))
    return Upload(
        repo=settings.repo("space"),
        folder=folder,
        title="Space (Gradio, no model loading)",
        notes=[f"reads only {logs.url}"],
    )
