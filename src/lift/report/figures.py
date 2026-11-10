"""Figures for the final report (``lift report``), in outputs/figures/ (PNG). matplotlib is imported here
(the ``report`` extra); without it no figure is drawn and the command says so.

- heatmap_harmful.png and heatmap_overrefusal.png: harmful rate (harmful set) and hard-benign refusal rate
  per language (columns) for stock, the adapter, the other adapters described (the Gate 1 pick and the
  candidate not picked, docs/decisions.md 81) and every frozen family (rows, pooled over seeds). A cell
  that was not run is blank and labelled "not run".
- adapters_by_language.png: the harmful rate of stock and of every adapter in each language, final split
  (grouped columns; an adapter's level is drawn only when every seed was run).
- tradeoff_dev.png: the safety-utility trade-off of every seed-1 dev setting from selection.json's table:
  harmful rate against win rate, with the win-rate floor and the adapter's harmful rate drawn in, and
  over-refusal increase against Belebele-hi drop with their limits; ineligible settings are hollow.
- tradeoff_dev_heldout.png: the same settings from selection_secondary.json (docs/decisions.md 73): harmful
  rate against the share of the adapter's held-out-loss gain each keeps, with the half-gain line, stock's
  and the adapter's harmful rates; settings the secondary rule leaves out are hollow.
- forest_primary.png: the primary endpoint (drift, repair) per seed and pooled, with 95% intervals.

Colours: three categorical hues (blue, orange, aqua) that stay apart under the common colour-vision
deficiencies, always in this order and tied to what they stand for (a repair family, an adapter's role),
never to a rank; stock, the reference, is a neutral grey. Every figure has its numbers in a table of the
report, so no reading depends on colour alone.
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

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
REFERENCE, INK, MUTED, GRID, SURFACE = "#8a8983", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FAMILY_COLOURS = {"wiseft": BLUE, "resta": ORANGE, "cast": AQUA}
KIND_COLOURS = {"standard": BLUE, "candidate": ORANGE, "adapter": AQUA}
KIND_LABELS = {
    "standard": "standard recipe (Gate 1 pick)",
    "candidate": "candidate not picked (exploratory)",
    "adapter": "adapter of the repair comparison",
}
LANG_LABELS = {"en": "English", "hi": "Hindi", "zh": "Chinese", "es": "Spanish", "ar": "Arabic", "th": "Thai"}


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
    if roles.get("standard"):
        rows.append(("Gate 1 pick", list(roles["standard"].values())))
    rows += [
        (f"not picked: {name}", list(by_seed.values()))
        for name, by_seed in (roles.get("candidates") or {}).items()
    ]
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


def _quiet(ax: Any) -> None:
    """Recessive axes: no top or right spine, a hairline grid behind the marks, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def adapters_by_language(stats: Mapping[str, Any], path: Path) -> Path | None:
    """Grouped columns: the harmful rate of stock and of each described adapter in every language, the
    target language first. An adapter's level is drawn only when every seed of its recipe was run."""
    rows = stats.get("adapter_drift") or []
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    stock: dict[str, Any] = {}
    for r in rows:
        if r["set"] != "harmful":
            continue
        groups.setdefault((r["kind"], r["adapter"]), {})[r["lang"]] = r["rate"]
        stock[r["lang"]] = r["stock"]
    groups = {k: v for k, v in groups.items() if any(x is not None for x in v.values())}
    if not groups or all(v is None for v in stock.values()):
        return None
    plt = _plt()
    target = str(config.get("languages.target"))
    langs = [target] + [lang for lang in LANGS if lang != target]
    order = [k for kind in ("standard", "candidate", "adapter") for k in groups if k[0] == kind]
    series = [("stock", REFERENCE, stock)]
    series += [
        (f"{name}: {KIND_LABELS[kind]}", KIND_COLOURS[kind], groups[(kind, name)]) for kind, name in order
    ]
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (label, colour, values) in enumerate(series):
        shown = [(j, values[lang]) for j, lang in enumerate(langs) if values.get(lang) is not None]
        xs = [j + (i - (len(series) - 1) / 2) * width for j, _ in shown]
        ys = [100 * v for _, v in shown]
        # A surface-coloured edge is the gap between neighbouring columns, not a border.
        ax.bar(xs, ys, width=width, color=colour, edgecolor=SURFACE, linewidth=2, label=label)
        if shown and shown[0][0] == 0:  # the value is written on the target language's columns only
            ax.annotate(
                f"{ys[0]:.0f}",
                (xs[0], ys[0]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
    _quiet(ax)
    ax.grid(False, axis="x")
    ax.set_xticks(range(len(langs)), [LANG_LABELS[lang] for lang in langs])
    ax.set_ylabel("outputs judged harmful (%)", color=MUTED, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Harmful rate on harmful prompts by language, final split", color=INK, fontsize=11, loc="left"
    )
    # Below the plot, so it never sits on a column.
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def tradeoff_heldout(secondary: Mapping[str, Any] | None, path: Path) -> Path | None:
    """The secondary selection's view (docs/decisions.md 73): each dev setting's Hindi harmful rate against
    the share of the adapter's held-out-loss gain it keeps."""
    if not secondary or secondary.get("status") != "decided" or not secondary.get("table"):
        return None
    plt = _plt()
    rows = secondary["table"]
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    seen = set()
    for r in rows:
        c = FAMILY_COLOURS.get(r.get("family"), REFERENCE)
        x, y = 100 * r["heldout_retained"], 100 * r["harmful_hi"]
        label = r.get("family") if r.get("family") not in seen else None
        seen.add(r.get("family"))
        ax.scatter(
            x,
            y,
            s=60,
            edgecolors=c,
            facecolors=c if r.get("eligible") else "none",
            linewidths=2,
            label=label,
            zorder=3,
        )
        ax.annotate(
            r["name"].split("-adapter-")[0],
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    ax.axvline(100 * rows[0]["min_heldout_retained"], color=MUTED, linewidth=1)
    ax.annotate(
        "half of the gain",
        (100 * rows[0]["min_heldout_retained"], 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    for key, text in (("adapter_scores", "the adapter"), ("stock", "stock")):
        level = (secondary.get(key) or {}).get("harmful_hi")
        if level is not None:
            ax.axhline(100 * level, color=MUTED, linewidth=1)
            ax.annotate(
                text,
                (1, 100 * level),
                xycoords=("axes fraction", "data"),
                xytext=(-4, 3),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=MUTED,
            )
    _quiet(ax)
    ax.set_xlabel("share of the adapter's held-out-loss gain kept (%)", color=MUTED, fontsize=9)
    ax.set_ylabel("Hindi dev harmful rate (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "Seed-1 dev settings: safety against what the adapter learned\n(hollow = left out by the "
        "secondary rule)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, title="repair family", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
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
    colours = FAMILY_COLOURS
    for r in rows:
        c = colours.get(r.get("family"), REFERENCE)
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
        a1.axvline(100 * rows[0]["min_win_rate"], ls="--", c="k", lw=1, label="win-rate floor")
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

    def read(name: str) -> Any:
        path = metrics_dir() / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    for p in (
        adapters_by_language(stats, folder / "adapters_by_language.png"),
        tradeoff(read("selection.json"), folder / "tradeoff_dev.png"),
        tradeoff_heldout(read("selection_secondary.json"), folder / "tradeoff_dev_heldout.png"),
        forest(stats, folder / "forest_primary.png"),
    ):
        if p is not None:
            out.append(p)
    return out
