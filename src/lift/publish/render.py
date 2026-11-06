"""Fill a template's ``{{ name }}`` slots and write markdown tables.

The templates live in ``hf/templates/``. A slot whose name the context lacks is an error (so a typo
never ships an empty sentence); ``{{ name }}`` is the only syntax, and values are inserted as they are.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from lift import paths
from lift.publish import PublishError

SLOT = re.compile(r"\{\{\s*([A-Za-z0-9_.:-]+)\s*\}\}")


def templates_dir() -> Path:
    return paths.REPO_ROOT / "hf" / "templates"


def template(name: str) -> str:
    return (templates_dir() / name).read_text(encoding="utf-8")


def slots(text: str) -> list[str]:
    return SLOT.findall(text)


def fill(text: str, context: Mapping[str, str]) -> str:
    unknown = sorted({s for s in slots(text) if s not in context})
    if unknown:
        raise PublishError(f"template slots with no value: {', '.join(unknown)}")
    return SLOT.sub(lambda m: str(context[m.group(1)]), text)


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def frontmatter(meta: Mapping[str, object]) -> str:
    """YAML front matter for an HF card (yaml.safe_dump, keys in the given order)."""
    import yaml

    return "---\n" + yaml.safe_dump(dict(meta), sort_keys=False, allow_unicode=True, width=100) + "---\n"


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    return path
