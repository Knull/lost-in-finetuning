"""``lift publish collections``: the two HF collections the article links to.

"models" lists the repaired model, the repair parameters, the gated adapters and the Space; "data"
lists the logs dataset, the gated raw completions and the translated prompt sets. With ``--yes`` every
listed repo is first checked to exist on the Hub (nothing is created while one is missing, so no
collection is left half-built), then the collections are created (or reused: ``exists_ok``) under the
team namespace and every item is added, and their urls are written to outputs/publish/collections.json,
which the article reads for its links.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lift import paths
from lift.publish import PublishError, hub, render, settings
from lift.publish.inputs import Facts, frozen_winner


def links_path() -> Path:
    return paths.output_dir() / "publish" / "collections.json"


def read_links() -> dict[str, str]:
    path = links_path()
    if not path.is_file():
        return {}
    return {k: v["url"] for k, v in json.loads(path.read_text(encoding="utf-8")).items() if "url" in v}


def plan(facts: Facts | None = None) -> dict[str, list[settings.RepoSpec]]:
    from lift.publish.model import _spec, model_repo, registered_adapters

    facts = facts or Facts()
    winner = frozen_winner(facts)
    models = [model_repo(facts, winner, _spec(winner) if winner else None), settings.repo("repair_params")]
    models += [
        settings.repo("adapter", adapter=name)
        for name in sorted(registered_adapters())
        if (paths.adapters_dir() / name).is_dir()
    ]
    models.append(settings.repo("space"))
    data = [settings.repo("logs"), settings.repo("raw"), settings.repo("prompts")]
    return {"models": models, "data": data}


def describe(p: dict[str, list[settings.RepoSpec]]) -> list[str]:
    cfg = settings.load()["collections"]
    lines = []
    for key, items in p.items():
        lines.append(f"collection {cfg[key]['title']!r} under {settings.namespace()}:")
        lines += [f"  {r.repo_type:8s} {r.repo_id}" for r in items]
    return lines


def execute(
    p: dict[str, list[settings.RepoSpec]],
    *,
    api_factory: Callable[[], Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, str]]:
    problems = [
        r.repo_id
        for items in p.values()
        for r in items
        if r.repo_id.startswith(settings.PLACEHOLDER_NAMESPACE + "/") or "pending-winner" in r.repo_id
    ]
    if problems:
        raise PublishError(
            "not creating collections: no namespace or no frozen winner yet for " + ", ".join(problems)
        )
    api = (api_factory or hub.default_api)()
    absent = [
        f"{r.repo_type} {r.repo_id}"
        for items in p.values()
        for r in items
        if not api.repo_exists(r.repo_id, repo_type=r.repo_type)
    ]
    if absent:
        raise PublishError(
            "not creating collections: these repos are not on the Hub yet (upload them first): "
            + ", ".join(absent)
        )
    cfg = settings.load()["collections"]
    out: dict[str, dict[str, str]] = {}
    for key, items in p.items():
        col = api.create_collection(
            cfg[key]["title"],
            namespace=settings.namespace(),
            description=cfg[key]["description"],
            exists_ok=True,
        )
        for r in items:
            api.add_collection_item(col.slug, r.repo_id, r.repo_type, exists_ok=True)
        out[key] = {
            "slug": col.slug,
            "url": getattr(col, "url", None) or f"https://huggingface.co/collections/{col.slug}",
        }
        log(f"collection {key}: {out[key]['url']} ({len(items)} items)")
    render.write_text(links_path(), json.dumps(out, indent=1))
    return out
