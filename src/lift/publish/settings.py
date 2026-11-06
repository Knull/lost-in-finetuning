"""configs/publish.yaml: the HF namespace, repo ids and access of every published artifact."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from lift import config, paths
from lift.publish import PublishError

PLACEHOLDER_NAMESPACE = "YOUR-HF-ORG"
ACCESS = {"public": (False, False), "gated-auto": (False, "auto"), "gated-manual": (False, "manual")}
_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


@dataclass(frozen=True)
class RepoSpec:
    kind: str  # model | repair_params | adapter | logs | raw | prompts | space
    repo_id: str
    repo_type: str  # model | dataset | space
    access: str  # public | gated-auto | gated-manual
    sdk: str | None = None

    @property
    def gated(self) -> str | bool:
        return ACCESS[self.access][1]

    @property
    def url(self) -> str:
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[self.repo_type]
        return f"https://huggingface.co/{prefix}{self.repo_id}"


def load() -> dict[str, Any]:
    return config.load("publish")


def namespace() -> str:
    """The team's HF namespace, or PLACEHOLDER_NAMESPACE while nobody has set it."""
    cfg = load()
    value = os.environ.get(cfg.get("namespace_env") or "LIFT_HF_NAMESPACE") or cfg.get("namespace")
    if not value:
        return PLACEHOLDER_NAMESPACE
    if not _NAMESPACE.match(str(value)):
        raise PublishError(
            f"the HF namespace in {cfg.get('namespace_env')} or configs/publish.yaml is not a "
            "valid HF user or organisation name"
        )
    return str(value)


def namespace_is_set() -> bool:
    return namespace() != PLACEHOLDER_NAMESPACE


def repo(kind: str, **fields: str) -> RepoSpec:
    entry = load()["repos"][kind]
    if entry["access"] not in ACCESS:
        raise PublishError(f"configs/publish.yaml repos.{kind}.access must be one of {sorted(ACCESS)}")
    name = entry["name"].format(**fields)
    return RepoSpec(
        kind=kind,
        repo_id=f"{namespace()}/{name}",
        repo_type=entry["type"],
        access=entry["access"],
        sdk=entry.get("sdk"),
    )


def github_repo() -> str | None:
    """The public GitHub URL: configs/publish.yaml, else the origin remote with any credentials stripped."""
    configured = load().get("github_repo")
    if configured:
        return str(configured)
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    url = re.sub(r"//[^/@]*@", "//", url)  # never carry a token embedded in the remote
    return url.removesuffix(".git") or None
