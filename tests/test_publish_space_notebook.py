"""The Space's data functions (no gradio, no network), its staging, and the notebook's structure: valid
JSON, every code cell parses, and every lift helper it calls exists."""

from __future__ import annotations

import ast
import importlib.util
import json
import re

import pytest
from publish_fakes import report, write_json

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
    (data / "examples").mkdir()
    (data / "examples" / "refusals.jsonl").write_text(
        json.dumps(
            {
                "variant": "stock",
                "lang": "hi",
                "set": "harmful",
                "pid": "p",
                "label": "refusal",
                "refusal_text": "I can't help with that.",
            }
        )
        + "\n"
    )
    monkeypatch.setenv("LIFT_SPACE_DATA", str(data))
    return mod


def test_space_reads_the_aggregate_dataset(app):
    data = app.load_data()
    assert "**repair**" in app.verdict_markdown(data["report"])
    names, langs, matrix = app.heatmap_matrix(data["report"])
    assert names == ["stock", "adapter", "repair"] and langs == ["hi", "en"]
    assert matrix[1][0] == pytest.approx(0.21)
    head, rows = app.table_rows(data["report"], "utility")
    assert head == ["Variant", "Win rate"] and rows[1][1] == "not run"
    assert app.example_rows(data["examples"], "hi")[0][-1] == "I can't help with that."
    assert app.verdict_markdown(None).startswith("The final results are not published")


def test_space_redacts_anything_but_refusals(app):
    text = " ".join(f"w{i}" for i in range(30))
    assert app.redact(text, "refusal") == text
    cut = app.redact(text, "harmful")
    assert cut.endswith("[redacted]") and len(cut.split()) == app.MAX_WORDS + 1


def test_space_heatmap_figure(app):
    pytest.importorskip("matplotlib")
    fig = app.heatmap_figure(app.load_data()["report"])
    assert fig.axes


def test_space_staging_names_the_logs_dataset(monkeypatch):
    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    up = space.build()
    assert json.loads((up.folder / "space_config.json").read_text()) == {
        "logs_dataset": "test-org/lift-eval-logs"
    }
    assert "sdk: gradio" in (up.folder / "README.md").read_text() and up.repo.repo_type == "space"


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
