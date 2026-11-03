"""``lift info``: paths, config digest, git state and installed versions."""

from __future__ import annotations

import argparse
import json


def _info(args: argparse.Namespace) -> int:
    from lift import config, paths
    from lift.manifest import git_sha, gpu_info, versions

    print(json.dumps({
        "repo": str(paths.REPO_ROOT),
        "data_dir": str(paths.data_dir()),
        "output_dir": str(paths.output_dir()),
        "git": git_sha(),
        "config_digest": config.digest(),
        "global_model": config.get("models.global"),
        "versions": versions(),
        "gpus": gpu_info(),
    }, indent=1))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("info", help="print paths, config digest, git state, versions")
    p.set_defaults(func=_info)
