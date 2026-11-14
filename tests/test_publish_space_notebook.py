"""The Space's data functions (no gradio, no network), its staging, and the notebook's structure: valid
JSON, every code cell parses, and every lift helper it calls exists."""

from __future__ import annotations

import ast
import importlib.util
import json
import re

import pytest
from publish_fakes import (  # noqa: F401
    HARD_REFUSAL,
    HARMFUL,
    P07_LANG_NAMES,
    no_committed_recipe,
    report,
    write_json,
)

from lift import paths
from lift.publish import acceptance, space

NOTEBOOK = paths.REPO_ROOT / "notebooks" / "acceptance_test.ipynb"


@pytest.fixture
def app(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("space_app", paths.REPO_ROOT / "hf" / "space" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = tmp_path / "logs"
    write_json(data / "aggregates" / "final_report.json", report())
    example = {"variant": "stock", "lang": "hi", "set": "harmful", "pid": "p", "label": "refusal"}
    write_json(
        data / "examples.json",
        [
            {**example, "text": "I can't help with that. [redacted]"},
            # a row that was never cut (a hand-edited file): the app cuts it again before showing it
            {**example, "pid": "q", "label": None, "text": " ".join(f"w{i}" for i in range(30))},
        ],
    )
    monkeypatch.setenv("LIFT_SPACE_DATA", str(data))
    return mod


def test_space_reads_the_aggregate_dataset(app):
    data = app.load_data()
    assert "**repair**" in app.verdict_markdown(data["report"])
    head, rows = app.table_rows(data["report"], "utility")
    assert head[3] == "Win rate vs stock" and rows[2][3] == "not run"
    first, second = app.example_rows(data["examples"], "hi")
    assert first == ["stock", "hi", "harmful", "p", "refusal", "I can't help with that. [redacted]"]
    assert second[-2:] == ["not judged", "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 [redacted]"]
    assert app.example_rows(data["examples"], "zh") == []
    assert app.verdict_markdown(None).startswith("The final results are not published")


def test_space_harmful_heatmap_reads_p07s_per_language_table(app):
    """P08-R2-01: P07's per_language table is wide (one row per language label, a column per role)."""
    names, langs, matrix = app.heatmap_matrix(app.load_data()["report"], "harmful")
    assert names == ["Stock", "Adapter", "Frozen repair"]
    assert langs == list(HARMFUL)
    for j, lang in enumerate(langs):
        assert [matrix[i][j] for i in range(3)] == list(HARMFUL[lang])
    assert matrix[2][langs.index("Thai")] is None  # an arm not run stays empty, never 0


def test_space_overrefusal_heatmap_reads_p07s_overrefusal_table(app):
    names, langs, matrix = app.heatmap_matrix(app.load_data()["report"], "overrefusal")
    assert names == ["Stock", "Adapter", "Frozen repair"]
    assert langs == [P07_LANG_NAMES[x] for x in HARD_REFUSAL]  # the hard set only
    for j, code in enumerate(HARD_REFUSAL):
        assert [matrix[i][j] for i in range(3)] == list(HARD_REFUSAL[code])


def test_space_heatmaps_are_empty_without_a_report(app):
    assert app.heatmap_matrix(None, "harmful") == ([], [], [])
    assert app.heatmap_matrix({"tables": {}}, "overrefusal") == ([], [], [])


CUT_SAMPLES = [
    "I can't help with that.",
    " ".join(f"w{i}" for i in range(30)),
    "这是一个很长的中文回答，它会持续很多很多个字，远远超过十个词的长度，所以应该被截断。",
    "นี่คือคำตอบภาษาไทยที่ยาวมากและควรถูกตัดให้สั้นลงเพราะไม่มีช่องว่างระหว่างคำ",
    "मैं इस अनुरोध में आपकी सहायता नहीं कर सकता क्योंकि यह हानिकारक हो सकता है और नीति के विरुद्ध है।",
    "Supercalifragilisticexpialidocious-" * 5,
    "mixed 中文和 English words 在一起 for a while longer than ten words in total",
    "",
    "w0 w1 w2 [redacted]",
    # six words filling 59 characters, then more: cut at 59, the cut and the space before its mark were the
    # output's first window, and the Space upload failed the scan (Nov 14)
    " ".join(["abcdefghi"] * 6) + " next words follow here",
]


@pytest.mark.parametrize("text", CUT_SAMPLES)
def test_space_cuts_every_example_whatever_its_label(app, text):
    """Decisions item 62: at most about ten words, then [redacted], refusals included. A script written
    without spaces counts two characters as a word, and no cut is as long as the scan's window, so a cut
    never holds a run the public-text scan looks for. The Space's copy agrees with the project's."""
    from lift.publish import preflight

    cut = space.cut(text)
    assert app.redact(text) == cut and space.cut(cut) == cut  # the copy agrees, and cutting twice is stable
    assert cut.endswith(space.MARK)
    body = cut.removesuffix(space.MARK).rstrip()
    assert text.startswith(body) and len(body) <= space.MAX_CHARS < preflight.WINDOW
    assert len(body.split()) <= space.MAX_WORDS
    assert len(space._UNSPACED.findall(body)) <= 2 * space.MAX_WORDS
    assert not preflight.has_window(cut, preflight.windows_of([text]))
    assert (app.MAX_WORDS, app.MAX_CHARS, app.MARK) == (space.MAX_WORDS, space.MAX_CHARS, space.MARK)


def test_the_space_examples_are_cut_when_it_is_built(monkeypatch):
    """A few outputs per study variant, language and set of the final split, each cut whatever its label;
    the staged Space holds no longer run of any output and passes the public-text scan."""
    from lift.io import sha256_text, write_jsonl
    from lift.publish import datasets, preflight

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    variant = "adapter-aya-std-s1"
    outs = {
        "p1": (
            "HARMFUL-SPACE-TEXT step one of a made-up completion " + "and then more of it " * 6,
            "harmful",
        ),
        "p2": ("REFUSAL-SPACE-TEXT I am sorry, but I am not able to help with this request " * 2, "refusal"),
        "p3": ("BENIGN-SPACE-TEXT a third answer that is not among the examples " * 2, "benign"),
    }

    def cell(set_name, lang, split, rows, judged=True):
        name = f"{set_name}_{lang}_{split}"
        gens = [
            {
                "key": sha256_text(f"{name}|{pid}"),
                "variant": variant,
                "pid": pid,
                "output": out,
                "hit_cap": False,
            }
            for pid, (out, _) in rows.items()
        ]
        write_jsonl(paths.generations_dir() / variant / f"{name}.jsonl", gens)
        if judged:
            write_jsonl(
                paths.judgements_dir() / datasets.role_dir("primary") / variant / f"{name}.jsonl",
                [
                    {
                        "key": g["key"],
                        "pid": g["pid"],
                        "label": rows[g["pid"]][1],
                        "output_sha": sha256_text(g["output"]),
                    }
                    for g in gens
                ],
            )

    cell("harmful", "hi", "final", outs)
    cell(
        "harmful",
        "hi",
        "dev",
        {"d1": ("DEV-SPACE-TEXT a dev output that is never an example " * 2, "harmful")},
    )
    cell(
        "easy_benign",
        "zh",
        "final",
        {"z1": ("这是一个很长的中文回答，它会持续很多很多个字，远远超过十个词。" * 3, "benign")},
        judged=False,
    )
    cell(
        "utility", "hi", "final", {"u1": ("UTILITY-SPACE-TEXT an answer to a utility prompt " * 3, "benign")}
    )
    up = space.build()
    rows = json.loads((up.folder / "examples.json").read_text(encoding="utf-8"))
    assert [(r["set"], r["lang"], r["pid"], r["label"]) for r in rows] == [
        ("easy_benign", "zh", "z1", None),  # not judged yet: no label is made up
        ("harmful", "hi", "p1", "harmful"),
        ("harmful", "hi", "p2", "refusal"),
    ]
    assert all(r["variant"] == variant and r["text"] == space.cut(r["text"]) for r in rows)
    assert rows[1]["text"] == "HARMFUL-SPACE-TEXT step one of a made-up completion and [redacted]"
    assert rows[2]["text"].endswith("[redacted]") and len(rows[2]["text"].split()) <= space.MAX_WORDS + 1
    windows = preflight.windows_of(datasets.output_record().texts)
    assert windows and preflight.scan_folder(up.folder, windows) == []
    staged = "".join(f.read_text(encoding="utf-8") for f in up.folder.rglob("*") if f.is_file())
    assert (
        "DEV-SPACE-TEXT" not in staged and "UTILITY-SPACE-TEXT" not in staged and "BENIGN-SPACE" not in staged
    )
    assert any("3 examples" in n and "whatever its label" in n for n in up.notes)
    # a longer text in the staged Space fails the scan like any public upload
    (up.folder / "examples.json").write_text(json.dumps([{"text": outs["p2"][0]}]), encoding="utf-8")
    assert [f.kind for f in preflight.scan_folder(up.folder, windows)] == [preflight.OUTPUT_TEXT]


def test_space_heatmap_figure_plots_the_table_values(app):
    pytest.importorskip("matplotlib")
    import math

    fig = app.heatmap_figure(app.load_data()["report"], "harmful")
    ax = fig.axes[0]
    image = ax.images[0].get_array()
    langs = list(HARMFUL)
    for j, lang in enumerate(langs):
        for i, value in enumerate(HARMFUL[lang]):
            got = image[i, j]
            if value is None:
                assert image.mask[i, j] or math.isnan(image.data[i, j])
            else:
                assert float(got) == pytest.approx(value * 100)
    assert [t.get_text() for t in ax.get_xticklabels()] == langs
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Stock", "Adapter", "Frozen repair"]
    cells = {t.get_text() for t in ax.texts}
    assert "21.0" in cells and "-" in cells  # Hindi adapter; Thai repair not run
    empty = app.heatmap_figure(None, "harmful")
    assert not empty.axes[0].images and empty.axes[0].texts[0].get_text() == "not published yet"


def test_space_staging_is_a_static_page_under_the_team(monkeypatch):
    """HF hosts a Gradio Space on free CPU only for PRO, a static one for everyone (decisions item 113): the
    staged Space is index.html, its card and the examples, under the team's namespace."""
    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    up = space.build()
    assert sorted(p.name for p in up.folder.iterdir()) == ["README.md", "examples.json", "index.html"]
    assert up.repo.repo_id == "test-org/lift-acceptance-test" and up.repo.sdk == "static"
    assert "sdk: static" in (up.folder / "README.md").read_text() and up.repo.repo_type == "space"
    page = (up.folder / "index.html").read_text()
    assert (
        page.startswith("<!doctype html>")
        and "https://huggingface.co/datasets/test-org/lift-eval-logs" in page
    )
    assert "The final results are not published yet." in page  # no report made here yet
    assert json.loads((up.folder / "examples.json").read_text()) == []  # nothing generated here yet


def test_the_static_page_draws_the_report_and_cuts_every_example(app):
    """The page shows what the Gradio preview shows, every number through fmt and every text through
    redact (a row never cut is cut here), escaped, with nothing loaded from elsewhere."""
    data = app.load_data()
    data["examples"].append(
        {
            "variant": "stock",
            "lang": "zh",
            "set": "harmful",
            "pid": "x",
            "label": "harmful",
            "text": "<script>alert(1)</script> and the rest",
        }
    )
    page = app.render_html(data, {"Aggregate logs dataset": "https://example.org/logs"})
    assert "<strong>repair</strong>" in page and 'href="https://example.org/logs"' in page
    head, rows = app.table_rows(data["report"], "utility")
    assert all(f"<th>{h}</th>" in page for h in head) and all(f"<td>{c}</td>" in page for c in rows[0])
    names, langs, matrix = app.heatmap_matrix(data["report"], "harmful")
    assert f"<th>{names[0]}</th>" in page and app.fmt(matrix[0][0], "pct") in page
    assert "w9 [redacted]" in page and "w10" not in page  # the never-cut row, cut by redact
    assert "<script>alert" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert 'data-lang="zh"' in page and '<option value="hi">hi</option>' in page
    assert "src=" not in page and "<link" not in page and "@import" not in page  # nothing from elsewhere
    empty = app.render_html({"report": None, "examples": []})
    assert "The final results are not published yet." in empty and "No examples yet." in empty


def test_space_loads_no_model():
    source = (paths.REPO_ROOT / "hf" / "space" / "app.py").read_text()
    for banned in ("transformers", "torch", "from_pretrained", "lift."):
        assert banned not in source


def test_notebook_structure():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert len(md) >= len(code) / 2  # short markdown cells explain each step
    imported, called = set(), set()
    for cell in code:
        py = "\n".join(line for line in cell.splitlines() if not line.lstrip().startswith(("!", "%")))
        tree = ast.parse(py)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
        called.update(re.findall(r"acceptance\.(\w+)", cell))
        assert all(not o.get("outputs") for o in nb["cells"] if o["cell_type"] == "code")
    assert imported <= {"lift", "huggingface_hub", "IPython"}
    assert called and all(hasattr(acceptance, name) for name in called), called - set(dir(acceptance))


# ---------------------------------------------------------------- Space: pins and formats (R1-21)


def test_space_formats_cells_like_the_article(app):
    from lift.publish.inputs import fmt

    data = app.load_data()
    head, rows = app.table_rows(data["report"], "utility")
    assert rows[0][3] == "62.0% [55.0, 69.0]"
    _, rows = app.table_rows(data["report"], "belebele")
    assert rows[0][1:3] == ["700", "50.0%"]  # P07 writes the Items column before the levels
    for value, kind in [
        (0.123, "pct"),
        (-0.04, "pp"),
        ([0.2, 0.1, 0.3], "pct_ci"),
        (0.0004, "p"),
        (1234, "int"),
        (1.5, "hours"),
        (None, "pct"),
        ("x", "text"),
        (0.05, "points"),
        ("undefined: the adapter shows no utility gain over parity", "pct_ci"),
        ("undefined", "f2"),
    ]:
        assert app.fmt(value, kind) == fmt(value, kind)


# A web page as a model writes one when asked (formatted head, CSS reset, a filter script): the Space's own
# code must share no 60-character run with it, or the public-text scan refuses the page (Nov 14).
MODEL_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Results</title>
    <style>
        :root { --bg: #ffffff; --fg: #1f2328; }
        * { box-sizing: border-box; }
        body { margin: 0; background: var(--bg); color: var(--fg);
               font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
        main { max-width: 1060px; margin: 0 auto; padding: 28px 16px 56px; }
        table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
        th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left;
                 vertical-align: top; }
        .wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
        @media (prefers-color-scheme: dark) { :root { --bg: #0d1117; --fg: #e6edf3; } }
    </style>
</head>
<body>
    <label for="lang">Language</label><select id="lang"><option value="">all</option></select>
    <script>
        document.getElementById("lang").addEventListener("change", function (e) {
            var v = e.target.value;
            document.querySelectorAll("#examples tbody tr").forEach(function (tr) {
                tr.hidden = v !== "" && tr.getAttribute("data-lang") !== v;
            });
        });
    </script>
</body>
</html>
"""


def test_the_static_page_shares_no_run_with_a_page_a_model_writes(app):
    from lift.publish import preflight

    # every alignment of the scan's 30-character stride over the model's page
    windows = preflight.windows_of([("x" * k) + MODEL_PAGE for k in range(30)])
    page = app.render_html(app.load_data(), {"Aggregate logs dataset": "https://example.org/logs"})
    assert not preflight.has_window(page, windows)


def test_space_requirements_are_pinned():
    """The local Gradio preview's dependencies stay pinned; the Space itself is static (item 113)."""
    lines = [x for x in (paths.REPO_ROOT / "hf" / "space" / "requirements.txt").read_text().splitlines() if x]
    assert lines and all("==" in x for x in lines)
    readme = (paths.REPO_ROOT / "hf" / "space" / "README.md").read_text()
    assert "sdk: static" in readme and "sdk_version" not in readme


# ---------------------------------------------------------------- CAST wrapper (R1-09)


def test_cast_generate_loads_the_artifact_file_as_the_evaluation(tmp_path, monkeypatch):
    import hashlib
    import sys
    import types

    repo = tmp_path / "repo"
    (repo / "repair" / "cast").mkdir(parents=True)
    artifact = repo / "repair" / "cast" / "cast.safetensors"
    artifact.write_bytes(b"cast artifact")
    params = {
        "method": "cast",
        "params": {"strength": 1.0},
        "artifacts": {"cast": hashlib.sha256(b"cast artifact").hexdigest()},
    }
    (repo / "repair" / "params.json").write_text(json.dumps(params))
    tok = types.SimpleNamespace(pad_token_id=7)
    model = types.SimpleNamespace(to=lambda device: model)
    fake_tf = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda d: tok),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda d, **k: model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    spec = importlib.util.spec_from_file_location(
        "cast_generate", paths.REPO_ROOT / "hf/templates/cast_generate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}

    def make(m, path, strength, *, gate=True, pad_id=0):
        seen.update(path=path, strength=strength, gate=gate, pad_id=pad_id)
        return "wrapped"

    assert mod.load(repo, device="cpu", make=make) == ("wrapped", tok)
    assert seen == {"path": artifact, "strength": 1.0, "gate": True, "pad_id": 7}
    artifact.write_bytes(b"another artifact")
    with pytest.raises(ValueError, match="sha256"):
        mod.load(repo, device="cpu", make=make)


# ---------------------------------------------------------------- notebook: the repo url (R1-26)


def test_notebook_links_are_stamped_from_publish_settings(tmp_path, monkeypatch):
    from lift.publish import settings

    nb = tmp_path / "nb.ipynb"
    nb.write_text(NOTEBOOK.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = {**settings.load(), "github_repo": "https://github.com/o/r"}
    monkeypatch.setattr(settings, "load", lambda: cfg)
    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    acceptance.stamp_notebook(nb)
    code = "".join(
        "".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"
    )
    assert 'REPO_URL = "https://github.com/o/r"' in code
    assert 'PARAMS_REPO = "test-org/lift-repair-params"' in code
    assert "Knull" not in code


# ---------------------------------------------------------------- notebook: a fresh clone (P08-R2-06, R1-26)


def test_notebook_installs_the_extras_and_checks_before_any_gpu_work():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    install = next(c for c in code if "pip install" in c)
    assert '".[gpu,data,eval]"' in install  # LangID (fasttext) is in the data extra
    first = {
        name: next(i for i, c in enumerate(code) if f"acceptance.{name}(" in c)
        for name in ("check_ready", "run_check")
    }
    assert first["check_ready"] < first["run_check"]
    timing = [
        i for i, c in enumerate(code) if "acceptance.start_clock(" in c or "acceptance.record_runtime(" in c
    ]
    assert len(timing) == 2 and timing[0] < first["run_check"] and timing[1] == len(code) - 1


def test_notebook_claims_no_run_time_it_has_not_measured():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert not re.search(r"\d+\s*(?:-\s*)?minutes?", text.replace(acceptance.RUNTIME_MARK, "")), (
        "an unmeasured duration"
    )
    assert f"{acceptance.RUNTIME_MARK} not yet timed" in text
