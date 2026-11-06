"""Figures for the final report (``lift report``), in outputs/figures/ (PNG). matplotlib is imported here
(the ``report`` extra); without it no figure is drawn and the command says so.

- heatmap_harmful.png and heatmap_overrefusal.png: harmful rate (harmful set) and hard-benign refusal rate
  per language (columns) for stock, the adapter and every frozen family (rows, pooled over seeds). A cell
  that was not run is blank and labelled "not run".
- tradeoff_dev.png: the safety-utility trade-off of every seed-1 dev setting from selection.json's table:
  harmful rate against win rate, with the win-rate floor and the adapter's harmful rate drawn in, and
  over-refusal increase against Belebele-hi drop with their limits; ineligible settings are hollow.
- forest_primary.png: the primary endpoint (drift, repair) per seed and pooled, with 95% intervals.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lift import config
from lift.paths import figures_dir, metrics_dir
from lift.schemas import LANGS

log = logging.getLogger("lift.report")


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rate(
    cells: Sequence[Mapping[str, Any]], names: Sequence[str], set_name: str, lang: str, key: str
) -> float | None:
    from lift.report.tables import _rate as rate

    return rate(cells, names, set_name, lang, key)


def heatmap(stats: Mapping[str, Any], set_name: str, key: str, title: str, path: Path) -> Path:
    import numpy as np

    plt = _plt()
    roles = stats["roles"]
    rows = [("stock", ["stock"]), ("adapter", list(roles["adapter"].values()))]
    rows += [(fam, list(by_seed.values())) for fam, by_seed in roles["families"].items()]
    data = np.full((len(rows), len(LANGS)), np.nan)
    for i, (_label, names) in enumerate(rows):
        for j, lang in enumerate(LANGS):
            v = _rate(stats["cells"], names, set_name, lang, key)
            if v is not None:
                data[i, j] = 100 * v
    fig, ax = plt.subplots(figsize=(1.2 * len(LANGS) + 2, 0.6 * len(rows) + 1.5))
    im = ax.imshow(
        np.ma.masked_invalid(data),
        cmap="Reds",
        vmin=0,
        vmax=max(1.0, float(np.nanmax(data)) if np.isfinite(data).any() else 1.0),
    )
    for i in range(len(rows)):
        for j in range(len(LANGS)):
            ax.text(
                j,
                i,
                "not run" if np.isnan(data[i, j]) else f"{data[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.set_xticks(range(len(LANGS)), LANGS)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def tradeoff(selection: Mapping[str, Any] | None, path: Path) -> Path | None:
    if not selection or selection.get("status") != "decided" or not selection.get("table"):
        return None
    plt = _plt()
    rules = selection.get("rules") or config.get("selection", name="repairs")
    rows = selection["table"]
    adapter = selection.get("adapter_scores") or {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    colours = {"wiseft": "tab:blue", "resta": "tab:orange", "cast": "tab:green"}
    for r in rows:
        c = colours.get(r.get("family"), "tab:gray")
        face = c if r.get("eligible") else "none"
        a1.scatter(100 * r["win_rate"], 100 * r["harmful_hi"], edgecolors=c, facecolors=face, s=50)
        a1.annotate(r["name"].split("-adapter-")[0], (100 * r["win_rate"], 100 * r["harmful_hi"]), fontsize=7)
        a2.scatter(
            100 * r["overrefusal_increase"], 100 * r["belebele_drop"], edgecolors=c, facecolors=face, s=50
        )
        a2.annotate(
            r["name"].split("-adapter-")[0],
            (100 * r["overrefusal_increase"], 100 * r["belebele_drop"]),
            fontsize=7,
        )
    if rows:
        a1.axvline(
            100 * rows[0]["min_win_rate"], ls="--", c="k", lw=1, label="win-rate floor (half the gain)"
        )
    if adapter.get("harmful_hi") is not None:
        a1.axhline(100 * adapter["harmful_hi"], ls=":", c="r", lw=1, label="adapter (no drop)")
    a1.set_xlabel("dev win rate vs stock (%)")
    a1.set_ylabel("Hindi dev harmful rate (%)")
    a1.legend(fontsize=8)
    a2.axvline(100 * rules["max_overrefusal_increase"], ls="--", c="k", lw=1, label="over-refusal limit")
    a2.axhline(100 * rules["max_belebele_hi_drop"], ls="--", c="gray", lw=1, label="Belebele-hi limit")
    a2.set_xlabel("hard-benign refusal - stock (points)")
    a2.set_ylabel("Belebele-hi drop vs stock (points)")
    a2.legend(fontsize=8)
    fig.suptitle("Seed-1 dev settings and the selection constraints (hollow = not eligible)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def forest(stats: Mapping[str, Any], path: Path) -> Path | None:
    plt = _plt()
    labels, est, lo, hi = [], [], [], []
    for kind in ("drift", "repair"):
        e = stats["primary"][kind]
        for r in e.get("per_seed", []):
            labels.append(f"{kind} seed {r['seed']}")
            est.append(100 * r["est"])
            lo.append(100 * r["lo"])
            hi.append(100 * r["hi"])
        if e.get("pooled"):
            p = e["pooled"]
            labels.append(f"{kind} pooled")
            est.append(100 * p["est"])
            lo.append(100 * p["lo"])
            hi.append(100 * p["hi"])
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.5))
    ys = list(range(len(labels)))[::-1]
    ax.errorbar(
        est,
        ys,
        xerr=[[e - a for e, a in zip(est, lo, strict=True)], [b - e for e, b in zip(est, hi, strict=True)]],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, c="k", lw=1)
    ax.set_yticks(ys, labels)
    ax.set_xlabel("difference in Hindi harmful rate (points), final split")
    ax.set_title("Primary endpoint: drift (adapter - stock) and repair (repair - adapter)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def draw_all(stats: Mapping[str, Any]) -> list[Path]:
    try:
        _plt()
    except ImportError:
        log.warning("matplotlib is not installed (the report extra): no figures drawn")
        return []
    folder = figures_dir()
    folder.mkdir(parents=True, exist_ok=True)
    out = [
        heatmap(
            stats, "harmful", "harmful_rate", "Harmful rate, final split (%)", folder / "heatmap_harmful.png"
        ),
        heatmap(
            stats,
            "hard_benign",
            "refusal_rate",
            "Hard-benign refusal rate, final split (%)",
            folder / "heatmap_overrefusal.png",
        ),
    ]
    sel_path = metrics_dir() / "selection.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.is_file() else None
    for p in (tradeoff(selection, folder / "tradeoff_dev.png"), forest(stats, folder / "forest_primary.png")):
        if p is not None:
            out.append(p)
    return out
