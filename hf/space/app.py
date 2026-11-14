"""The Space: the verdicts, per-language heatmaps, utility retained and a few cut examples of the
retain / repair / reject test, read only from the public aggregate logs dataset (no model loading).

HF hosts a Gradio Space on free CPU only for a PRO account, while a static one is free (402 for the
organisation, then for a user, Nov 14; docs/decisions.md 113). So ``render_html`` draws the same views as
one static page, which ``lift publish space`` builds as index.html; ``build_ui`` stays as a local Gradio
preview (``python hf/space/app.py``).

Data: ``aggregates/final_report.json`` of the logs dataset named in ``space_config.json`` (``lift publish
space`` writes it), and ``examples.json`` beside this file: a few outputs, each cut when the Space was built
to at most MAX_WORDS words and MAX_CHARS characters plus ``[redacted]``, whatever its label. ``redact``
cuts each one again before it is shown, so no longer text can appear here. A local folder with the same
layout in ``LIFT_SPACE_DATA`` replaces both (for testing). Every number shown comes from
final_report.json, formatted by each column's ``format`` exactly as the article does (``fmt`` is a copy of
the project's formatter in src/lift/publish/inputs.py: the Space imports nothing from the project).
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPORT = "aggregates/final_report.json"
EXAMPLES = "examples.json"
MAX_WORDS = 10
MAX_CHARS = 58
MARK = "[redacted]"
NOT_RUN = "not run"
# Scripts written without spaces between words: two characters count as one word there.
_UNSPACED = re.compile(
    "[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002fa1f]"
)


# A copy of the project's fmt (src/lift/publish/inputs.py); tests/test_publish_space_notebook.py checks it.
def _num(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"not a finite number: {value!r}")
    return float(value)


def _interval(value: Any) -> tuple[float, float, float]:
    if isinstance(value, dict):
        return _num(value["est"]), _num(value["lo"]), _num(value["hi"])
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return _num(value[0]), _num(value[1]), _num(value[2])
    raise ValueError(f"not an interval {{est, lo, hi}}: {value!r}")


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return NOT_RUN
    if isinstance(value, str) and (value == "undefined" or value.startswith("undefined:")):
        return value  # computed but undefined, as the report says it (never a number or "not run")
    if kind == "text":
        return str(value)
    if kind == "pct":
        return f"{_num(value) * 100:.1f}%"
    if kind == "pp":
        return f"{_num(value) * 100:+.1f} pp"
    if kind == "pct_ci":
        est, lo, hi = _interval(value)
        return f"{est * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    if kind == "pp_ci":
        est, lo, hi = _interval(value)
        return f"{est * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}]"
    if kind == "points":
        return f"{_num(value) * 100:g} points"
    if kind == "p":
        p = _num(value)
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    if kind == "int":
        n = _num(value)
        if n != int(n):
            raise ValueError(f"not an integer: {value!r}")
        return f"{int(n):,}"
    if kind == "f2":
        return f"{_num(value):.2f}"
    if kind == "f3":
        return f"{_num(value):.3f}"
    if kind == "hours":
        return f"{_num(value):.2f} h"
    if kind == "ms":
        return f"{_num(value):.1f} ms"
    if kind == "list":
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"not a list: {value!r}")
        return ", ".join(str(x) for x in value) or NOT_RUN
    raise ValueError(f"unknown format {kind!r}")


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
    report_path = _fetch(REPORT)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path else None
    local = os.environ.get("LIFT_SPACE_DATA")
    examples_path = (Path(local) if local else HERE) / EXAMPLES
    examples = json.loads(examples_path.read_text(encoding="utf-8")) if examples_path.is_file() else []
    return {"report": report, "examples": examples if isinstance(examples, list) else []}


# A copy of the project's cut (src/lift/publish/space.py); tests/test_publish_space_notebook.py checks it.
def redact(text: Any) -> str:
    """The start of ``text``, then ``[redacted]``, whatever its label: at most MAX_WORDS words (two
    characters count as a word in a script written without spaces) and at most MAX_CHARS characters, ending
    on a whole word unless the first word alone is longer."""
    kept: list[str] = []
    budget, room = MAX_WORDS, MAX_CHARS
    for word in str(text).removesuffix(MARK).split():
        if budget <= 0:
            break
        cost = max(1, -(-len(_UNSPACED.findall(word)) // 2))
        if cost > budget:
            word, cost = word[: 2 * budget], budget
        if len(word) > room:  # no room left for the whole word: only a first word is cut inside
            if not kept:
                kept.append(word[:room])
            break
        kept.append(word)
        budget -= cost
        room -= len(word) + 1
    return f"{' '.join(kept)} {MARK}".strip()


def verdict_markdown(report: dict | None) -> str:
    if not report:
        return "The final results are not published yet."
    lines = ["## Verdicts", ""]
    for v in report.get("verdicts") or []:
        useful = f"; Gate 1: **{v['usefulness_verdict']}**" if v.get("usefulness_verdict") else ""
        lines.append(
            f"- **{v.get('adapter')}** ({v.get('recipe', 'standard')} recipe): "
            f"**{v.get('verdict') or 'no verdict'}**{useful}. {v.get('reason', '')}"
        )
    if report.get("not_run"):
        lines += ["", "Not run: " + "; ".join(report["not_run"])]
    return "\n".join(lines)


# The heatmaps, read from the tables P07 writes (its report/tables.py): ``per_language`` has one row
# per language label with a column per role; ``overrefusal`` has one row per benign set and language.
ROLES = ("stock", "adapter", "repair")
HEATMAPS = {
    "harmful": ("per_language", None, "harmful rate (%)"),
    "overrefusal": ("overrefusal", ("set", "hard"), "hard-benign refusal rate (%)"),
}


def heatmap_matrix(report: dict | None, kind: str = "harmful"):
    """(role labels, language labels, matrix[role][language]) of P07's table for ``kind``; a value not run
    stays None. Empty lists when the table is not there yet."""
    name, keep, _ = HEATMAPS[kind]
    spec = (((report or {}).get("tables") or {}).get(name)) or {}
    labels = {c.get("key"): c.get("label") for c in spec.get("columns") or []}
    rows = [r for r in spec.get("rows") or [] if keep is None or r.get(keep[0]) == keep[1]]
    roles = [k for k in ROLES if k in labels]
    if not rows or not roles:
        return [], [], []
    langs = [str(r.get("lang")) for r in rows]
    matrix = [[r.get(role) for r in rows] for role in roles]
    return [str(labels[k]) for k in roles], langs, matrix


def heatmap_figure(report: dict | None, kind: str = "harmful"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, langs, matrix = heatmap_matrix(report, kind)
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * max(1, len(langs)), 1 + 0.6 * max(1, len(names))))
    if not names:
        ax.text(0.5, 0.5, "not published yet", ha="center", va="center")
        ax.axis("off")
        return fig
    values = [[float("nan") if v is None else float(v) * 100 for v in row] for row in matrix]
    im = ax.imshow(values, cmap="Reds", vmin=0, aspect="auto")
    ax.set_xticks(range(len(langs)), langs, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, "-" if v != v else f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=HEATMAPS[kind][2])
    fig.tight_layout()
    return fig


def table_rows(report: dict | None, name: str) -> tuple[list[str], list[list[str]]]:
    spec = (((report or {}).get("tables") or {}).get(name)) or {}
    cols = spec.get("columns") or []
    out = [[fmt(r.get(c["key"]), c.get("format", "text")) for c in cols] for r in spec.get("rows") or []]
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
                e.get("label") or "not judged",
                redact(e.get("text") or ""),
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
            gr.Plot(heatmap_figure(report, "harmful"))
        with gr.Tab("Over-refusal"):
            gr.Plot(heatmap_figure(report, "overrefusal"))
        with gr.Tab("Utility retained"):
            head, rows = table_rows(report, "utility")
            gr.Dataframe(value=rows, headers=head or None)
        with gr.Tab("Examples (cut)"):
            gr.Markdown(
                f"The first words of a few outputs, each cut to at most {MAX_WORDS} words and marked "
                f"`{MARK}`, whatever its label. The full outputs are in the gated dataset only."
            )
            langs = sorted({e.get("lang") for e in examples if e.get("lang")})
            pick = gr.Dropdown(choices=["all", *langs], value="all", label="language")
            table = gr.Dataframe(
                value=example_rows(examples),
                headers=["variant", "lang", "set", "pid", "label", "text"],
                wrap=True,
            )
            pick.change(lambda x: example_rows(examples, None if x == "all" else x), pick, table)
    return demo


# ---------------------------------------------------------------- the static page (decisions item 113)

# One hue, light to dark, for the heatmap cells (the value is printed in each cell as well).
_RAMP = ("#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15")
_CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --line: #d1d9e0; --card: #f6f8fa; --link: #0b57d0; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --line: #3d444d; --card: #151b23; --link: #8ab4f8; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1060px; margin: 0 auto; padding: 28px 16px 56px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 19px; margin: 36px 0 8px; }
p { margin: 8px 0; }
a { color: var(--link); }
.sub, .note { color: var(--muted); }
.note { font-size: 13px; }
nav { margin: 12px 0 4px; display: flex; flex-wrap: wrap; gap: 6px 18px; }
ul.verdicts { padding-left: 20px; }
ul.verdicts li { margin: 8px 0; }
.wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; margin: 8px 0; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
th { background: var(--card); font-weight: 600; white-space: nowrap; }
td.num { text-align: right; white-space: nowrap; }
td.heat { text-align: right; white-space: nowrap; color: #1f2328; }
td.heat.dark { color: #ffffff; }
td.na { text-align: right; color: var(--muted); }
label { font-weight: 600; }
select { font: inherit; padding: 3px 8px; margin-left: 6px; }
"""
_FILTER = (
    'var s=document.getElementById("lang");s.onchange=function(){'
    'var r=document.getElementById("examples").tBodies[0].rows,i;'
    'for(i=0;i<r.length;i++)r[i].hidden=s.value!==""&&r[i].dataset.lang!==s.value}'
)


# The page's own code is written compactly (no spaces around punctuation), unlike the formatted HTML, CSS
# and JS a model writes when asked for a web page: the public-text scan reads every 60-character run of the
# page, and a formatted head or CSS reset is the same run as such an output's (the first upload, Nov 14).
_TIGHT = re.compile(r"\s*([{};:,>])\s*")


def _tight(code: str) -> str:
    return _TIGHT.sub(r"\1", " ".join(code.split()))


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def verdict_html(report: dict | None) -> str:
    """verdict_markdown's content as HTML."""
    if not report:
        return "<p>The final results are not published yet.</p>"
    items = []
    for v in report.get("verdicts") or []:
        gate = v.get("usefulness_verdict")
        useful = f"; Gate 1: <strong>{_e(gate)}</strong>" if gate else ""
        items.append(
            f"\n<li><strong>{_e(v.get('adapter'))}</strong> ({_e(v.get('recipe', 'standard'))} recipe): "
            f"<strong>{_e(v.get('verdict') or 'no verdict')}</strong>{useful}. {_e(v.get('reason', ''))}</li>"
        )
    out = f'<ul class="verdicts">{"".join(items)}</ul>' if items else "<p>No verdicts in the report.</p>"
    if report.get("not_run"):
        out += f'<p class="note">Not run: {_e("; ".join(report["not_run"]))}</p>'
    return out


def heatmap_html(report: dict | None, kind: str = "harmful") -> str:
    """heatmap_matrix as a table: one row per role, one column per language, each cell its rate (fmt) on
    a one-hue ramp scaled to the table's largest value; a cell not run says so."""
    names, langs, matrix = heatmap_matrix(report, kind)
    if not names:
        return "<p>Not published yet.</p>"
    values = [v for row in matrix for v in row if isinstance(v, (int, float)) and not isinstance(v, bool)]
    top = max(values, default=0) or 1
    head = "".join(f"<th>{_e(lang)}</th>" for lang in langs)
    rows = []
    for name, row in zip(names, matrix, strict=True):
        cells = []
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                cells.append(f'<td class="na">{_e(fmt(None, "pct"))}</td>')
                continue
            step = min(len(_RAMP) - 1, max(0, round(float(v) / top * (len(_RAMP) - 1))))
            dark = " dark" if step >= len(_RAMP) - 2 else ""
            cells.append(f'<td class="heat{dark}" style="background:{_RAMP[step]}">{_e(fmt(v, "pct"))}</td>')
        rows.append(f"\n<tr><th>{_e(name)}</th>{''.join(cells)}</tr>")
    return (
        f'<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f'<p class="note">{_e(HEATMAPS[kind][2])}, by model and language; darker is higher.</p>'
    )


def table_html(report: dict | None, name: str) -> str:
    head, rows = table_rows(report, name)
    if not head:
        return "<p>Not published yet.</p>"
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join("\n<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def examples_html(examples: list[dict]) -> str:
    """example_rows as a table (each text cut again by redact), with a language filter."""
    rows = example_rows(examples)
    if not rows:
        return "<p>No examples yet.</p>"
    langs = sorted({r[1] for r in rows if r[1]})
    options = "".join(f'<option value="{_e(lang)}">{_e(lang)}</option>' for lang in langs)
    head = "".join(f"<th>{h}</th>" for h in ("variant", "lang", "set", "pid", "label", "text"))
    body = "".join(
        f'\n<tr data-lang="{_e(r[1])}">' + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return (
        f'<p><label for=lang>Language</label> <select id=lang><option value="">all</option>{options}'
        "</select></p>"
        f'<div class="wrap"><table id=examples><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
        f"</div><script>{_FILTER}</script>"
    )


def render_html(data: dict[str, Any] | None = None, links: dict[str, str] | None = None) -> str:
    """The whole Space as one static HTML page: no script or style from elsewhere, every number from
    final_report.json through fmt, every example cut by redact."""
    data = data or load_data()
    report, examples = data.get("report"), data.get("examples") or []
    nav = "".join(f'<a href="{_e(url)}">{_e(label)}</a>' for label, url in (links or {}).items() if url)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Lost in Fine-Tuning</title>\n<style>{_tight(_CSS)}</style>\n</head><body><main>\n"
        "<h1>Lost in Fine-Tuning</h1>"
        '<p class="sub">Retain, repair or reject a Hindi LoRA adapter of Tiny Aya Global: does it drift on '
        "safety, in Hindi and in other languages, and which cheap repair restores safety without losing the "
        "adapter's gain?</p>"
        f"<nav>{nav}</nav>\n"
        f"<h2>Verdicts</h2>\n{verdict_html(report)}\n"
        f"<h2>Harmful rate</h2>\n{heatmap_html(report, 'harmful')}\n"
        f"<h2>Over-refusal</h2>\n{heatmap_html(report, 'overrefusal')}\n"
        f"<h2>Utility retained</h2>\n{table_html(report, 'utility')}\n"
        "<h2>Examples (cut)</h2>\n"
        f'<p class="note">The first words of a few outputs, each cut to at most {MAX_WORDS} words and marked '
        f"{_e(MARK)}, whatever its label. The full outputs are in the gated dataset only.</p>"
        f"{examples_html(examples)}\n"
        '<p class="note">A static page built from the public aggregate logs dataset '
        "(aggregates/final_report.json). It loads no model and generates nothing.</p>"
        "\n</main></body></html>\n"
    )


if __name__ == "__main__":
    build_ui().launch()
