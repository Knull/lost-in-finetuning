"""Figures for the final report (``lift report``), in outputs/figures/ (PNG). matplotlib is imported here
(the ``report`` extra); without it no figure is drawn and the command says so.

- heatmap_harmful.png and heatmap_overrefusal.png: harmful rate (harmful set) and hard-benign refusal rate
  per language (columns) for stock, the adapter, the other adapters described (the Gate 1 pick and the
  candidate not picked, docs/decisions.md 81) and every frozen family (rows, pooled over seeds). A cell
  that was not run is blank and labelled "not run".
- adapters_by_language.png: the harmful rate of stock and of every adapter in each language, final split
  (grouped columns; an adapter's level is drawn only when every seed was run).
- tradeoff_dev.png: the safety-utility trade-off of every seed-1 dev setting from selection.json's table:
  harmful rate against win rate, with the win-rate floor, parity, and stock's and the adapter's harmful
  rates drawn in, and over-refusal increase against Belebele-hi drop with their limits; the region each
  panel's conditions allow is tinted, and a setting that met every condition is filled.
- tradeoff_dev_heldout.png: the same settings from selection_secondary.json (docs/decisions.md 73): harmful
  rate against each setting's held-out-loss gain over stock as a share of the full adapter's (over 100%
  when it fits the references better than the adapter, negative when worse than stock), with stock's
  loss, the half-gain line and the full adapter marked, stock's and the adapter's harmful rates, and a
  setting far below stock's loss drawn at the left edge with its share written out; a setting that met
  every condition of the secondary rule is filled.

Names are placed so they sit on no other name or point (``_label_points``); settings on one spot share a
name.
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


# Where a point's name may sit, tried in this order: offset in points, horizontal and vertical alignment.
_LABEL_SPOTS = (
    (6, 5, "left", "bottom"),
    (6, -5, "left", "top"),
    (-6, 5, "right", "bottom"),
    (-6, -5, "right", "top"),
    (0, 8, "center", "bottom"),
    (0, -8, "center", "top"),
)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """The area two boxes (x0, y0, x1, y1) share."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _label_points(ax: Any, points: Sequence[tuple[float, float, str]], *, fontsize: float = 7) -> None:
    """Write each point's name beside it without sitting on another name or another point. Points drawn
    on one spot are named once, their names joined. For each name the six spots around its point are
    tried in order and the first whose box meets no placed name, no other point and stays inside the axes
    is taken; with none free, the spot that overlaps least. Boxes are in display units, with the text's
    width estimated from its length (short tags in one font size: close enough to keep names apart).
    Call it once the limits and the layout are final."""
    merged: dict[tuple[float, float], list[str]] = {}
    for x, y, name in points:
        merged.setdefault((round(x, 1), round(y, 1)), []).append(name)
    scale = ax.figure.dpi / 72  # points to display units
    frame = ax.bbox.extents
    centres = {key: ax.transData.transform(key) for key in merged}
    radius = 5 * scale
    marks = {key: (c[0] - radius, c[1] - radius, c[0] + radius, c[1] + radius) for key, c in centres.items()}
    placed: list[tuple[float, float, float, float]] = []
    for key, names in merged.items():
        text = ", ".join(names)
        width, height = 0.58 * fontsize * len(text) * scale, 1.3 * fontsize * scale
        cx, cy = centres[key]
        best = None
        for dx, dy, ha, va in _LABEL_SPOTS:
            x0 = cx + dx * scale - (width if ha == "right" else width / 2 if ha == "center" else 0)
            y0 = cy + dy * scale - (height if va == "top" else 0)
            box = (x0, y0, x0 + width, y0 + height)
            outside = (
                max(0.0, frame[0] - box[0])
                + max(0.0, box[2] - frame[2])
                + max(0.0, frame[1] - box[1])
                + max(0.0, box[3] - frame[3])
            )
            cost = (
                sum(_overlap(box, other) for other in placed)
                + sum(_overlap(box, mark) for k, mark in marks.items() if k != key)
                + outside * height
            )
            if best is None or cost < best[0]:
                best = (cost, dx, dy, ha, va, box)
            if cost == 0:
                break
        _cost, dx, dy, ha, va, box = best
        placed.append(box)
        ax.annotate(
            text,
            key,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=MUTED,
        )


def _reference_line(ax: Any, at: float, text: str, *, vertical: bool) -> None:
    """A dashed reference line, quieter than the data, named outside the plot area: above it for a
    vertical line, to its right for a horizontal one."""
    style = {"color": MUTED, "linewidth": 0.8, "linestyle": (0, (4, 3)), "zorder": 1}
    if vertical:
        ax.axvline(at, **style)
        ax.annotate(
            text,
            (at, 1),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
    else:
        ax.axhline(at, **style)
        ax.annotate(
            text,
            (1, at),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=MUTED,
        )


def _allowed(ax: Any, *, x: tuple[float | None, float | None], y: tuple[float | None, float | None]) -> None:
    """Tint the rectangle a panel's conditions allow; None is the axis's own end. The limits are kept."""
    from matplotlib.patches import Rectangle

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0, x1 = (xlim[0] if x[0] is None else x[0]), (xlim[1] if x[1] is None else x[1])
    y0, y1 = (ylim[0] if y[0] is None else y[0]), (ylim[1] if y[1] is None else y[1])
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=GRID, edgecolor="none", alpha=0.55, zorder=0)
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _family_legend(fig_or_ax: Any, families: Sequence[str], **where: Any) -> None:
    """One filled marker per repair family: the fill of a point says whether it met the rule, so the legend
    shows the hue only."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=FAMILY_COLOURS.get(f, REFERENCE),
            markeredgecolor=FAMILY_COLOURS.get(f, REFERENCE),
            label=f,
        )
        for f in families
    ]
    fig_or_ax.legend(
        handles=handles, fontsize=8, frameon=False, title="repair family", title_fontsize=8, **where
    )


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


OFF_SCALE = -25.0  # % of the adapter's gain: a setting further below stock's loss is drawn at the edge


def tradeoff_heldout(secondary: Mapping[str, Any] | None, path: Path) -> Path | None:
    """The secondary selection's view (docs/decisions.md 73): each dev setting's Hindi harmful rate against
    the share of the adapter's held-out-loss gain it keeps. The share is not confined to 0-100%: a setting
    that fits the held-out references better than the full adapter is over 100%, and one that fits them
    worse than stock is negative. The axis runs from OFF_SCALE to past the largest share; a setting further
    left is drawn at the edge as a left-pointing marker with its share written out, so one broken setting
    does not squeeze the rest into a corner (docs/decisions.md 92)."""
    if not secondary or secondary.get("status") != "decided" or not secondary.get("table"):
        return None
    plt = _plt()
    rows = secondary["table"]
    floor = 100 * rows[0]["min_heldout_retained"]
    shares = [100 * r["heldout_retained"] for r in rows]
    right = max([*shares, 100.0, floor]) + 18
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    named: list[tuple[float, float, str]] = []
    families: list[str] = []
    for r, share in zip(rows, shares, strict=True):
        family = str(r.get("family"))
        c = FAMILY_COLOURS.get(family, REFERENCE)
        if family not in families:
            families.append(family)
        off = share < OFF_SCALE
        x, y = (OFF_SCALE if off else share), 100 * r["harmful_hi"]
        ax.scatter(
            x,
            y,
            s=60,
            marker="<" if off else "o",
            edgecolors=c,
            facecolors=c if r.get("eligible") else "none",
            linewidths=2,
            zorder=3,
        )
        tag = r["name"].split("-adapter-")[0]
        named.append((x, y, f"{tag} ({share:.0f}%)" if off else tag))
    for at, text in ((0.0, "stock's loss"), (floor, "half of the gain"), (100.0, "the full adapter")):
        _reference_line(ax, at, text, vertical=True)
    for key, text in (("adapter_scores", "the adapter"), ("stock", "stock")):
        level = (secondary.get(key) or {}).get("harmful_hi")
        if level is not None:
            _reference_line(ax, 100 * level, text, vertical=False)
    _quiet(ax)
    ax.set_xlim(OFF_SCALE - 12, right)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        "held-out-loss gain over stock, as a share of the full adapter's (%)", color=MUTED, fontsize=9
    )
    ax.set_ylabel("Hindi dev harmful rate (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "Seed-1 dev settings: safety against what the adapter learned\n(filled = meets every condition of "
        "the secondary rule; a left-pointing marker is off the scale)",
        color=INK,
        fontsize=10,
        loc="left",
        pad=20,
    )
    # Below the plot, so it never sits on a point.
    _family_legend(ax, families, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3)
    fig.tight_layout()
    _label_points(ax, named)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def tradeoff(selection: Mapping[str, Any] | None, path: Path) -> Path | None:
    """The primary selection's view: every seed-1 dev setting against the rule's four conditions
    (selection.json). Left: the Hindi harmful rate against the win rate, with the win-rate floor, parity,
    and stock's and the adapter's harmful rates. Right: the two side conditions, the rise in hard-benign
    refusals and the Belebele-hi drop, with their limits. A filled point met every condition."""
    if not selection or selection.get("status") != "decided" or not selection.get("table"):
        return None
    plt = _plt()
    rules = selection.get("rules") or config.get("selection", name="repairs")
    rows = selection["table"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.6), facecolor=SURFACE)
    named1: list[tuple[float, float, str]] = []
    named2: list[tuple[float, float, str]] = []
    families: list[str] = []
    for r in rows:
        family = str(r.get("family"))
        c = FAMILY_COLOURS.get(family, REFERENCE)
        if family not in families:
            families.append(family)
        look = {"s": 60, "edgecolors": c, "facecolors": c if r.get("eligible") else "none", "linewidths": 2}
        tag = r["name"].split("-adapter-")[0]
        p1 = (100 * r["win_rate"], 100 * r["harmful_hi"])
        p2 = (100 * r["overrefusal_increase"], 100 * r["belebele_drop"])
        a1.scatter(*p1, zorder=3, **look)
        a2.scatter(*p2, zorder=3, **look)
        named1.append((*p1, tag))
        named2.append((*p2, tag))
    _reference_line(a1, 100 * rows[0]["min_win_rate"], "the floor", vertical=True)
    _reference_line(a1, 50.0, "parity with stock", vertical=True)
    for key, text in (("adapter_scores", "the adapter"), ("stock", "stock")):
        level = (selection.get(key) or {}).get("harmful_hi")
        if level is not None:
            _reference_line(a1, 100 * level, text, vertical=False)
    _reference_line(a2, 100 * rules["max_overrefusal_increase"], "over-refusal limit", vertical=True)
    _reference_line(a2, 100 * rules["max_belebele_hi_drop"], "Belebele limit", vertical=False)
    for ax, title, xlabel, ylabel in (
        (
            a1,
            "Safety against answer quality (tinted: over the floor, under the adapter)",
            "dev win rate against stock (%)",
            "Hindi dev harmful rate (%)",
        ),
        (
            a2,
            "The two side conditions (tinted: inside both limits)",
            "hard-benign refusals, setting minus stock (points)",
            "Belebele-hi accuracy lost against stock (points)",
        ),
    ):
        ax.set_facecolor(SURFACE)
        _quiet(ax)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=20)
    a1.set_xlim(left=0, right=max(60.0, a1.get_xlim()[1]))
    a1.set_ylim(bottom=0)
    # The region each panel's conditions allow, as a quiet tint behind the marks.
    level = (selection.get("adapter_scores") or {}).get("harmful_hi")
    if level is not None:
        _allowed(a1, x=(100 * rows[0]["min_win_rate"], None), y=(None, 100 * level))
    _allowed(
        a2, x=(None, 100 * rules["max_overrefusal_increase"]), y=(None, 100 * rules["max_belebele_hi_drop"])
    )
    fig.suptitle(
        "Seed-1 dev settings against the selection rule (filled = meets every condition)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    _family_legend(fig, families, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.07, 1, 0.99), w_pad=6)
    _label_points(a1, named1)
    _label_points(a2, named2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
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
