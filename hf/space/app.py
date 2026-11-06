"""Gradio Space: the verdicts, per-language heatmaps, utility retained and redacted examples of the
retain / repair / reject test, read only from the public aggregate logs dataset (no model loading).

Data: ``aggregates/final_report.json`` and ``examples/refusals.jsonl`` of the logs dataset named in
``space_config.json`` (``lift publish space`` writes it), or a local folder with the same layout in
``LIFT_SPACE_DATA`` (for testing). Every number shown comes from final_report.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPORT = "aggregates/final_report.json"
EXAMPLES = "examples/refusals.jsonl"
MAX_WORDS = 10


def _fetch(name: str) -> Path | None:
    local = os.environ.get("LIFT_SPACE_DATA")
    if local:
        path = Path(local) / name
        return path if path.is_file() else None
    cfg = HERE / "space_config.json"
    if not cfg.is_file():
        return None
    repo = json.loads(cfg.read_text(encoding="utf-8")).get("logs_dataset")
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename=name, repo_type="dataset"))
    except Exception:  # a missing file or no network: the Space says the data is not there yet
        return None


def load_data() -> dict[str, Any]:
    report_path, examples_path = _fetch(REPORT), _fetch(EXAMPLES)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else None
    examples = []
    if examples_path:
        examples = [
            json.loads(line)
            for line in examples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {"report": report, "examples": examples}


def redact(text: str, label: str | None) -> str:
    """Refusals are shown as they are; anything else is cut to MAX_WORDS words and marked."""
    if label == "refusal":
        return text
    words = str(text).split()
    return " ".join(words[:MAX_WORDS]) + " [redacted]"


def _est(value: Any) -> float | None:
    if isinstance(value, dict):
        return value.get("est")
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value if isinstance(value, (int, float)) else None


def verdict_markdown(report: dict | None) -> str:
    if not report:
        return "The final results are not published yet."
    lines = ["## Verdicts", ""]
    for v in report.get("verdicts") or []:
        lines.append(
            f"- **{v.get('adapter')}** ({v.get('recipe', 'standard')} recipe): "
            f"**{v.get('verdict')}**. {v.get('reason', '')}"
        )
    if report.get("not_run"):
        lines += ["", "Not run: " + "; ".join(report["not_run"])]
    return "\n".join(lines)


def heatmap_matrix(report: dict | None, metric: str = "harmful_rate", set_name: str = "harmful"):
    """(row labels, column labels, matrix) of ``metric`` per variant role and language."""
    rows = (((report or {}).get("tables") or {}).get("per_language") or {}).get("rows") or []
    rows = [r for r in rows if r.get("set", set_name) == set_name and metric in r]
    order = ["hi", "en", "zh", "es", "ar", "th"]
    langs = sorted({r["lang"] for r in rows}, key=lambda x: (order.index(x) if x in order else len(order), x))
    names = list(dict.fromkeys(str(r.get("role") or r.get("variant")) for r in rows))
    matrix = [[None] * len(langs) for _ in names]
    for r in rows:
        matrix[names.index(str(r.get("role") or r.get("variant")))][langs.index(r["lang"])] = _est(r[metric])
    return names, langs, matrix


def heatmap_figure(report: dict | None, metric: str = "harmful_rate", set_name: str = "harmful"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, langs, matrix = heatmap_matrix(report, metric, set_name)
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * max(1, len(langs)), 1 + 0.6 * max(1, len(names))))
    if not names:
        ax.text(0.5, 0.5, "not published yet", ha="center", va="center")
        ax.axis("off")
        return fig
    values = [[float("nan") if v is None else v * 100 for v in row] for row in matrix]
    im = ax.imshow(values, cmap="Reds", vmin=0, aspect="auto")
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(names)), names)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, "-" if v != v else f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"{metric} (%)")
    fig.tight_layout()
    return fig


def table_rows(report: dict | None, name: str) -> tuple[list[str], list[list[str]]]:
    spec = (((report or {}).get("tables") or {}).get(name)) or {}
    cols = spec.get("columns") or []
    out = []
    for r in spec.get("rows") or []:
        cells = []
        for c in cols:
            v = r.get(c["key"])
            est = _est(v)
            if v is None:
                cells.append("not run")
            elif isinstance(v, dict) and {"lo", "hi"} <= v.keys():
                cells.append(f"{est * 100:.1f} [{v['lo'] * 100:.1f}, {v['hi'] * 100:.1f}]")
            else:
                cells.append(str(v))
        out.append(cells)
    return [c["label"] for c in cols], out


def example_rows(
    examples: list[dict], lang: str | None = None, variant: str | None = None
) -> list[list[str]]:
    rows = []
    for e in examples:
        if (lang and e.get("lang") != lang) or (variant and e.get("variant") != variant):
            continue
        rows.append(
            [
                e.get("variant", ""),
                e.get("lang", ""),
                e.get("set", ""),
                e.get("pid", ""),
                redact(e.get("refusal_text") or "", e.get("label")),
            ]
        )
    return rows


def build_ui(data: dict[str, Any] | None = None):
    import gradio as gr

    data = data or load_data()
    report, examples = data["report"], data["examples"]
    with gr.Blocks(title="Lost in Fine-Tuning") as demo:
        gr.Markdown("# Lost in Fine-Tuning\nRetain, repair or reject a Hindi adapter of Tiny Aya Global.")
        gr.Markdown(verdict_markdown(report))
        with gr.Tab("Harmful rate"):
            gr.Plot(heatmap_figure(report, "harmful_rate", "harmful"))
        with gr.Tab("Over-refusal"):
            gr.Plot(heatmap_figure(report, "refusal_rate", "hard_benign"))
        with gr.Tab("Utility retained"):
            head, rows = table_rows(report, "utility")
            gr.Dataframe(value=rows, headers=head or None)
        with gr.Tab("Example refusals"):
            langs = sorted({e.get("lang") for e in examples if e.get("lang")})
            pick = gr.Dropdown(choices=["all", *langs], value="all", label="language")
            table = gr.Dataframe(
                value=example_rows(examples), headers=["variant", "lang", "set", "pid", "text"], wrap=True
            )
            pick.change(lambda x: example_rows(examples, None if x == "all" else x), pick, table)
    return demo


if __name__ == "__main__":
    build_ui().launch()
