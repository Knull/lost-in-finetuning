"""``lift publish collections``: the two HF collections the article links to.

"models" lists the repaired model, the repair parameters, the gated adapters and the Space; "data"
lists the logs dataset, the gated raw completions, the translated prompt sets and the training sets.
With ``--yes`` every listed repo is first checked to exist on the Hub (nothing is created while one is
missing, so no collection is left half-built), then the collections are created (or reused:
``exists_ok``) under the team namespace and every item is added. Each collection's slug, url and items
(the repo ids it was given, every one checked to be on the Hub) are written to
outputs/publish/collections.json: the article reads the urls for its links (``read_links``), and calls
the training sets published only once they are among the data collection's items (``read_items``).
Both name the repaired model, so ``--yes`` refuses while the recipe is not frozen at its
``recipe-frozen-*`` tag (``lift.publish.inputs.recipe_block``, review XB2-02).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lift import paths
from lift.publish import PublishError, hub, render, settings
from lift.publish.inputs import Facts, frozen_winner, recipe_block


def links_path() -> Path:
    return paths.output_dir() / "publish" / "collections.json"


def _recorded() -> dict[str, Any]:
    path = links_path()
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in doc.items() if isinstance(v, dict)} if isinstance(doc, dict) else {}


def read_links() -> dict[str, str]:
    """{collection: its url}, as ``execute`` recorded them in outputs/publish/collections.json."""
    return {k: v["url"] for k, v in _recorded().items() if "url" in v}


def read_items() -> dict[str, list[str]]:
    """{collection: the repo ids ``execute`` added to it}, from outputs/publish/collections.json. A file
    written before the items were recorded gives none, so nothing reads as published on its account."""
    return {
        k: [i for i in v["items"] if isinstance(i, str)]
        for k, v in _recorded().items()
        if isinstance(v.get("items"), list)
    }


def plan(facts: Facts | None = None) -> dict[str, list[settings.RepoSpec]]:
    from lift.publish.model import _spec, model_repo, registered_adapters

    facts = facts or Facts()
    winner = frozen_winner(facts)
    models = [model_repo(facts, winner, _spec(winner) if winner else None), settings.repo("repair_params")]
    # every registered adapter, pulled here or not (review XB-05): execute checks each repo is on the Hub
    models += [settings.repo("adapter", adapter=name) for name in sorted(registered_adapters())]
    models.append(settings.repo("space"))
    data = [settings.repo("logs"), settings.repo("raw"), settings.repo("prompts"), settings.repo("sft")]
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
) -> dict[str, dict[str, Any]]:
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
    block = recipe_block(Facts())
    if block:
        raise PublishError(f"not creating collections: blocked: {block}")
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
    out: dict[str, dict[str, Any]] = {}
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
            "items": [r.repo_id for r in items],
        }
        log(f"collection {key}: {out[key]['url']} ({len(items)} items)")
    render.write_text(links_path(), json.dumps(out, indent=1))
    return out
