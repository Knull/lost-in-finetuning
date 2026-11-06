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
    head, rows = app.table_rows(data["report"], "utility")
    assert head[3] == "Win rate vs stock" and rows[2][3] == "not run"
    assert app.example_rows(data["examples"], "hi")[0][-1] == "I can't help with that."
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


def test_space_redacts_anything_but_refusals(app):
    text = " ".join(f"w{i}" for i in range(30))
    assert app.redact(text, "refusal") == text
    cut = app.redact(text, "harmful")
    assert cut.endswith("[redacted]") and len(cut.split()) == app.MAX_WORDS + 1


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


# ---------------------------------------------------------------- Space: pins and formats (R1-21)


def test_space_formats_cells_like_the_article(app):
    from lift.publish.inputs import fmt

    data = app.load_data()
    head, rows = app.table_rows(data["report"], "utility")
    assert rows[0][3] == "62.0% [55.0, 69.0]"
    _, rows = app.table_rows(data["report"], "belebele")
    assert rows[0][1] == "50.0%"
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
    ]:
        assert app.fmt(value, kind) == fmt(value, kind)


def test_space_requirements_are_pinned():
    lines = [x for x in (paths.REPO_ROOT / "hf" / "space" / "requirements.txt").read_text().splitlines() if x]
    assert lines and all("==" in x for x in lines)
    readme = (paths.REPO_ROOT / "hf" / "space" / "README.md").read_text()
    gradio = next(x for x in lines if x.startswith("gradio=="))
    assert f"sdk_version: {gradio.split('==')[1]}" in readme


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
