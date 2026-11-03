"""The data build's offline world: the shrunken config every data test builds to (contract section 8),
a Hub serving every fake source, a BuildContext wired to the fakes, and isolated data, splits,
outputs and sheets dirs. All text is fake and benign. Import directly (``from fake_build import
...``), like datafakes: P02 owns tests/conftest.py.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datafakes import FakeLID, FakeTranslator
from fake_sft import FakeCurate, SftHub
from fake_sources import FakeHub, fake_hub

from lift import config, manifest, splits
from lift.data import build
from lift.data.translate import TranslationBlocked
from lift.paths import REPO_ROOT

POLYREFUSE_FIXTURE = Path(__file__).parent / "fixtures" / "polyrefuse" / "Multilingual-Refusal"
COUNTS = {"harmful_test": 6, "harmless_test": 4, "harmful_train": 3, "harmful_val": 2,
          "harmless_train": 3, "harmless_val": 2}


def fixture_cfg() -> dict[str, Any]:
    """The project config with every size shrunk to the fixtures (contract section 8)."""
    cfg = copy.deepcopy(config.load("project"))
    cfg["sources"]["polyrefuse"]["counts"] = dict(COUNTS)
    cfg["sizes"]["harmful"] = {"dev": 2, "final": 4}
    cfg["sizes"]["easy_benign"] = {"dev": 2, "final": 2}
    cfg["sources"]["orbench"]["n"] = 6
    cfg["sizes"]["hard_benign"] = {"fit": 2, "dev": 2, "final": 2}
    cfg["sizes"]["belebele"] = {"dev": 2, "final": 3}
    cfg["sources"]["sft"]["heldout"]["per_source"] = 2
    cfg["sizes"]["utility"] = {"dev": 1, "final": 3}
    cfg["validation"]["translation_spot_check"] = 2
    return cfg


def fixture_train_cfg(rows: int = 5) -> dict[str, Any]:
    cfg = copy.deepcopy(config.load("train"))
    cfg["rows"] = rows
    return cfg


def fixture_hub() -> FakeHub:
    """orbench, aya_redteaming and belebele (every language) plus the three SFT sources."""
    return fake_hub(extra=SftHub().sources)


def fixture_ctx(**overrides: Any) -> build.BuildContext:
    """A BuildContext on the fakes: FakeLID, the fixture hub, a FakeTranslator, FakeCurate and the
    miniature PolyRefuse tree (not a git clone, so no commit check)."""
    fields: dict[str, Any] = {"cfg": fixture_cfg(), "train_cfg": fixture_train_cfg(), "lid": FakeLID(),
                              "hub": fixture_hub(), "translator": FakeTranslator(), "curate": FakeCurate(),
                              "polyrefuse_root": POLYREFUSE_FIXTURE}
    fields.update(overrides)
    return build.BuildContext(**fields)


def isolate(monkeypatch: Any, tmp_path: Path) -> SimpleNamespace:
    """Point the data dir, outputs dir (run manifests), split files and spot-check sheets at tmp. The
    manifests' git lookup is stubbed (two git calls per build add up across the suite)."""
    root = Path(tmp_path).resolve()
    dirs = SimpleNamespace(data=root / "data", splits=root / "splits", outputs=root / "outputs",
                           sheets=root / "sheets")
    monkeypatch.setenv("LIFT_DATA_DIR", str(dirs.data))
    monkeypatch.setenv("LIFT_OUTPUT_DIR", str(dirs.outputs))
    monkeypatch.setattr(splits, "SPLITS_DIR", dirs.splits)
    monkeypatch.setattr(build, "SHEETS_DIR", dirs.sheets)
    monkeypatch.setattr(manifest, "git_sha", lambda: "0" * 40)
    return dirs


def repo_state() -> dict[str, tuple[int, int]]:
    """(size, mtime) of every file a build could touch in the real repo: data/ (the PolyRefuse clone's
    .git aside), outputs/manifests and validation/sheets. Equal before and after = untouched."""
    out = {}
    for top in (REPO_ROOT / "data", REPO_ROOT / "outputs" / "manifests", REPO_ROOT / "validation" / "sheets"):
        if not top.exists():
            continue
        for path in top.rglob("*"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if path.is_file() and "/Multilingual-Refusal/.git/" not in rel:
                stat = path.stat()
                out[rel] = (stat.st_size, stat.st_mtime_ns)
    return out


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, by posix path relative to it."""
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def read_sheet(path: Path) -> list[dict[str, str]]:
    """A spot-check sheet as a reader's spreadsheet sees it: one dict per row."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    """Save a filled sheet the way a spreadsheet exports it (UTF-8 with BOM, CRLF)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(build.SHEET_COLUMNS), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


class BlockingTranslator(FakeTranslator):
    """A FakeTranslator that answers the first ``ok`` calls, then raises TranslationBlocked on every call,
    as Google does once it answers HTTP 429."""

    def __init__(self, ok: int = 0, **kw: Any) -> None:
        super().__init__(**kw)
        self.ok = ok

    def __call__(self, text: str, lang: str) -> str:
        if len(self.calls) >= self.ok:
            self.calls.append((text, lang))
            raise TranslationBlocked("fake HTTP 429")
        return super().__call__(text, lang)
