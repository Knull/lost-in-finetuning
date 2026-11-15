"""scripts/verify_numbers.py, the check a reader runs from a fresh clone: every value of
hf/article.sources.json against the public logs dataset, and the figures against figures/.

Offline: a fake fetcher stands in for the dataset, and the script is loaded by its path (it is not part of
the package and imports nothing of it). Every number here is made up for the tests. The last three tests
read the committed hf/article.sources.json: the files it names, and the design values it takes from
configs/, which the script only counts.
"""

from __future__ import annotations

import email.message
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import urllib.error
from collections import Counter
from pathlib import Path

import pytest
from publish_fakes import no_committed_recipe, write_inputs, write_json  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_numbers.py"
SOURCES = ROOT / "hf" / "article.sources.json"  # the committed list, written by lift publish article


def _load():
    spec = importlib.util.spec_from_file_location("verify_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()

REPORT_FILE = "outputs/metrics/final_report.json"  # in the dataset: aggregates/final_report.json
REPORT = {
    "headline": {"rate": 0.125, "n": 400, "ci": {"est": 0.1, "lo": 0.05, "hi": 0.2}},
    "primary": {"drift": {"pooled": None}},
    "tables": {"cells": {"rows": [{"label": "stock", "judge_failures": None}, {"label": "adapter"}]}},
}
NO_FIGURES = "0 figures match the copies in figures/; 0 differ"
NO_CONFIGS = "0 design values are in configs/*.yaml of this repository, not in the logs: not checked"


def value(file: str, pointer: str, recorded: object) -> dict:
    """One entry of a sources file, as ``Facts`` records it."""
    return {"shown": str(recorded), "file": file, "pointer": pointer, "value": recorded}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Logs:
    """A stand-in for the public dataset, {path: document}, that records every fetch."""

    def __init__(self, files: dict[str, object]) -> None:
        self.files = {path: json.dumps(doc).encode() for path, doc in files.items()}
        self.fetched: list[str] = []

    def __call__(self, path: str) -> bytes | None:
        self.fetched.append(path)
        return self.files.get(path)


def no_fetch(path: str) -> bytes | None:
    raise AssertionError(f"{path} was fetched")


def run(tmp_path: Path, capsys, values: list[dict], fetch) -> tuple[int, list[str], str]:
    """The script on a sources file that holds ``values``: (exit code, the lines of stdout, stderr)."""
    sources = tmp_path / "hf" / "article.sources.json"
    sources.parent.mkdir(exist_ok=True)
    sources.write_text(json.dumps({"values": values, "pending": []}), encoding="utf-8", newline="\n")
    code = script.main(["--sources", str(sources), "--figures", str(tmp_path / "figures")], fetch=fetch)
    captured = capsys.readouterr()
    return code, captured.out.splitlines(), captured.err


def test_values_that_match_the_public_logs(tmp_path, capsys):
    logs = Logs(
        {
            "aggregates/final_report.json": REPORT,
            "aggregates/validation/round1.json": {"judges": {"primary": {"kappa": 0.67}}},
            "auditkit/qwen3guard/stock/harmful_hi_final.json": {"metrics": {"harmful_rate": 0.25}},
            "manifests/final-20261112T010203Z-abc123.json": {"gpu_hours": 0.5},
        }
    )
    audit_record = "outputs/metrics/auditkit/qwen3guard/stock/harmful_hi_final.json"
    values = [
        value(REPORT_FILE, "/headline/rate", 0.125),
        value(REPORT_FILE, "/headline/n", 400.0),  # an int there and a float here: the same value
        value(REPORT_FILE, "/headline/ci", {"hi": 0.2, "lo": 0.05, "est": 0.1}),  # a whole interval
        value(REPORT_FILE, "/primary/drift/pooled/est", None),  # under a null: an arm that was not run
        value(REPORT_FILE, "/tables/cells/rows/0/judge_failures", None),  # a null that is there
        value(REPORT_FILE, "/tables/cells/rows/1/label", "adapter"),
        value("outputs/metrics/validation/round1.json", "/judges/primary/kappa", 0.67),
        value(audit_record, "/metrics/harmful_rate", 0.25),
        value("outputs/manifests/final-20261112T010203Z-abc123.json", "/gpu_hours", 0.5),
    ]
    code, out, err = run(tmp_path, capsys, values, logs)
    assert (code, err) == (0, "")
    assert out == ["9 values match the public logs; 0 differ; 0 not in the logs", NO_FIGURES, NO_CONFIGS]
    assert sorted(logs.fetched) == sorted(logs.files)  # each public file once, whatever it holds


def test_values_are_compared_parsed_never_by_the_text_the_article_shows(tmp_path, capsys):
    """The article prints a negative number with a real minus sign (U+2212): the check reads the recorded
    value, never the shown text, so the sign changes nothing, and a shown text that agrees with the
    article does not save a value that differs."""
    doc = {"primary": {"repair": {"pooled": {"est": -0.392, "lo": -0.445, "hi": -0.342}}}}
    logs = Logs({"aggregates/final_report.json": doc})
    pooled = {"est": -0.392, "lo": -0.445, "hi": -0.342}
    shown = "\u221239.2 pp [\u221244.5, \u221234.2]"
    values = [{"shown": shown, "file": REPORT_FILE, "pointer": "/primary/repair/pooled", "value": pooled}]
    code, out, _ = run(tmp_path, capsys, values, logs)
    assert code == 0 and out[0] == "1 values match the public logs; 0 differ; 0 not in the logs"
    values[0]["value"] = {**pooled, "est": -0.391}
    code, out, _ = run(tmp_path, capsys, values, logs)
    assert code == 1 and out[0].startswith("0 values match the public logs; 1 differ")


def test_a_value_that_differs_is_named_and_the_exit_code_is_1(tmp_path, capsys):
    logs = Logs({"aggregates/final_report.json": REPORT})
    values = [
        value(REPORT_FILE, "/headline/rate", 0.125),
        value(REPORT_FILE, "/headline/n", 401),
        value(REPORT_FILE, "/headline/ci", {"est": 0.1, "lo": 0.05, "hi": 0.25}),
        value(REPORT_FILE, "/tables/cells/rows/0/label", "st\u00f6ck"),
        value(REPORT_FILE, "/tables/cells/rows/1/label", "x" * 300),
        value(REPORT_FILE, "/tables/cells/rows/0/judge_failures", 0),  # null there: not the same as 0
    ]
    code, out, err = run(tmp_path, capsys, values, logs)
    assert (code, err) == (1, "")
    assert out[:3] == ["1 values match the public logs; 5 differ; 0 not in the logs", NO_FIGURES, NO_CONFIGS]
    assert out[3:] == [
        f"differs: {REPORT_FILE} /headline/n: recorded 401, public 400",
        f'differs: {REPORT_FILE} /headline/ci: recorded {{"est": 0.1, "hi": 0.25, "lo": 0.05}}, '
        'public {"est": 0.1, "hi": 0.2, "lo": 0.05}',
        # text outside ASCII is escaped, so that any console can print the line
        f'differs: {REPORT_FILE} /tables/cells/rows/0/label: recorded "st\\u00f6ck", public "stock"',
        # a long value is cut to about 100 characters
        f'differs: {REPORT_FILE} /tables/cells/rows/1/label: recorded "{"x" * 99}..., public "adapter"',
        f"differs: {REPORT_FILE} /tables/cells/rows/0/judge_failures: recorded 0, public null",
    ]


def test_a_value_the_logs_do_not_hold_is_counted_and_listed(tmp_path, capsys):
    logs = Logs({"aggregates/final_report.json": REPORT})
    values = [
        value(REPORT_FILE, "/headline/rate", 0.125),
        value(REPORT_FILE, "/tables/signatures", None),  # the table of an analysis that was not run
        value(REPORT_FILE, "/tables/cells/rows/2/label", "repair"),  # there is no third row
        value(REPORT_FILE, "/tables/\u03bb", 0.25),
        value("outputs/metrics/sweep_breakdown.json", "/counts/harmful", 84),  # a file the dataset lacks
        value("outputs/metrics/sweep_breakdown.json", "/counts/refusal", 76),
        value("data/sft/report.json", "/rows", 677),  # a tree the dataset keeps no copy of
    ]
    code, out, err = run(tmp_path, capsys, values, logs)
    assert (code, err) == (0, "")  # nothing differs: the values are listed, and the check passes
    assert out[:3] == ["1 values match the public logs; 0 differ; 6 not in the logs", NO_FIGURES, NO_CONFIGS]
    no_pointer = "(aggregates/final_report.json has no such pointer)"
    no_file = "(the logs dataset has no aggregates/sweep_breakdown.json)"
    assert out[3:] == [
        f"not in the logs: {REPORT_FILE} /tables/signatures: recorded null {no_pointer}",
        f'not in the logs: {REPORT_FILE} /tables/cells/rows/2/label: recorded "repair" {no_pointer}',
        f"not in the logs: {REPORT_FILE} /tables/\\u03bb: recorded 0.25 {no_pointer}",
        f"not in the logs: outputs/metrics/sweep_breakdown.json /counts/harmful: recorded 84 {no_file}",
        f"not in the logs: outputs/metrics/sweep_breakdown.json /counts/refusal: recorded 76 {no_file}",
        "not in the logs: data/sft/report.json /rows: recorded 677 (the logs dataset keeps no copy of this "
        "file)",
    ]
    # the missing file is asked for once, and a file of a tree the dataset does not keep never
    assert logs.fetched == ["aggregates/final_report.json", "aggregates/sweep_breakdown.json"]


def test_design_values_from_configs_are_counted_and_never_fetched(tmp_path, capsys):
    values = [
        value("configs/project.yaml", "/sizes/harmful/final", 400),
        value("configs/repairs.yaml", "/selection/tie_margin", 0.02),
    ]
    code, out, err = run(tmp_path, capsys, values, no_fetch)
    assert (code, err) == (0, "")
    assert out == [
        "0 values match the public logs; 0 differ; 0 not in the logs",
        NO_FIGURES,
        "2 design values are in configs/*.yaml of this repository, not in the logs: not checked",
    ]


def test_figures_are_checked_against_the_copies_beside_the_readme(tmp_path, capsys):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "heatmap_harmful.png").write_bytes(b"\x89PNG as recorded")
    (figures / "forest_primary.png").write_bytes(b"\x89PNG drawn again")
    heatmap = value("outputs/figures/heatmap_harmful.png", "", sha256(b"\x89PNG as recorded"))
    not_drawn = value("outputs/figures/tradeoff_dev_heldout.png", "", None)  # recorded as not drawn: no copy
    code, out, err = run(tmp_path, capsys, [heatmap, not_drawn], no_fetch)  # never the dataset's copy
    assert (code, err) == (0, "")
    assert out == [
        "0 values match the public logs; 0 differ; 0 not in the logs",
        "2 figures match the copies in figures/; 0 differ",
        NO_CONFIGS,
    ]

    recorded, found = sha256(b"\x89PNG the first drawing"), sha256(b"\x89PNG drawn again")
    redrawn = value("outputs/figures/forest_primary.png", "", recorded)
    gone = value("outputs/figures/tradeoff_dev.png", "", recorded)
    code, out, err = run(tmp_path, capsys, [heatmap, redrawn, gone], no_fetch)
    assert (code, err) == (1, "")  # a figure that differs fails the check, like a number
    assert out[:3] == [
        "0 values match the public logs; 0 differ; 0 not in the logs",
        "1 figures match the copies in figures/; 2 differ",
        NO_CONFIGS,
    ]
    assert out[3:] == [
        f'differs: outputs/figures/forest_primary.png: recorded "{recorded}", figures/forest_primary.png '
        f'has "{found}"',
        f'differs: outputs/figures/tradeoff_dev.png: recorded "{recorded}", figures/tradeoff_dev.png is not '
        "there",
    ]


def test_the_script_runs_on_the_standard_library_alone(tmp_path):
    """As a reader runs it, in an interpreter that sees no installed package (-I -S). A list of design
    values and figures needs no fetch, so this stays offline."""
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "heatmap_harmful.png").write_bytes(b"\x89PNG as recorded")
    values = [
        value("configs/project.yaml", "/sizes/harmful/final", 400),
        value("outputs/figures/heatmap_harmful.png", "", sha256(b"\x89PNG as recorded")),
    ]
    sources = tmp_path / "article.sources.json"
    sources.write_text(json.dumps({"values": values, "pending": []}), encoding="utf-8", newline="\n")
    argv = [sys.executable, "-I", "-S", str(SCRIPT), "--sources", str(sources), "--figures", str(figures)]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    assert (done.returncode, done.stderr) == (0, "")
    assert done.stdout.splitlines() == [
        "0 values match the public logs; 0 differ; 0 not in the logs",
        "1 figures match the copies in figures/; 0 differ",
        "1 design values are in configs/*.yaml of this repository, not in the logs: not checked",
    ]
    (figures / "heatmap_harmful.png").write_bytes(b"\x89PNG drawn again")
    done = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 1 and done.stdout.splitlines()[3].startswith("differs: outputs/figures/heatmap")


def test_a_fetch_error_exits_2_and_names_the_url(tmp_path, capsys):
    url = "https://huggingface.co/datasets/org/logs/resolve/main/"

    def down(path: str) -> bytes | None:
        raise script.FetchError(f"{url}{path}: HTTP 503 (Service Unavailable)")

    values = [value(REPORT_FILE, "/headline/rate", 0.125)]
    code, out, err = run(tmp_path, capsys, values, down)
    assert (code, out) == (2, [])  # no summary: nothing was compared
    assert err == f"verify_numbers: {url}aggregates/final_report.json: HTTP 503 (Service Unavailable)\n"

    # A public file that is not JSON (an error page, say) is a failure too, never a value "not in the logs".
    code, out, err = run(tmp_path, capsys, values, lambda path: b"<html>Sign in</html>")
    not_json = "verify_numbers: aggregates/final_report.json of the logs dataset is not valid JSON ("
    assert (code, out) == (2, []) and err.startswith(not_json)


@pytest.mark.parametrize(
    "text,said",
    [
        (None, "cannot be read (No such file or directory)"),
        ("lift publish article", "is not valid JSON"),
        ('{"pending": []}', 'has no "values" list'),
        ('{"values": [{"shown": "1", "file": "outputs/metrics/x.json", "value": 1}]}', "values[0] does not"),
    ],
)
def test_a_list_that_cannot_be_read_exits_2(tmp_path, capsys, text, said):
    sources = tmp_path / "article.sources.json"
    if text is not None:
        sources.write_text(text, encoding="utf-8", newline="\n")
    code = script.main(["--sources", str(sources)], fetch=no_fetch)
    captured = capsys.readouterr()
    assert (code, captured.out) == (2, "")
    assert captured.err.startswith("verify_numbers: ") and said in captured.err


def test_an_unknown_flag_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as stop:
        script.main(["--token", "none-is-needed"], fetch=no_fetch)
    assert stop.value.code == 2
    assert "unrecognized arguments: --token" in capsys.readouterr().err


@pytest.mark.parametrize("timeout", ["-1", "0", "inf", "nan", "soon"])
def test_a_timeout_that_is_no_number_of_seconds_is_a_usage_error(capsys, timeout):
    """urlopen raises ValueError or OverflowError for such a timeout: uncaught, that is a traceback and
    exit code 1, which the script keeps for a value that differs. The flag refuses it first."""
    with pytest.raises(SystemExit) as stop:
        script.main([f"--timeout={timeout}"], fetch=no_fetch)
    assert stop.value.code == 2
    assert f"argument --timeout: '{timeout}' is not a number of seconds above 0" in capsys.readouterr().err


def test_a_timeout_in_seconds_reaches_the_fetcher(tmp_path, capsys, monkeypatch):
    asked = []
    monkeypatch.setattr(script, "http_fetcher", lambda repo, rev, timeout: asked.append(timeout) or no_fetch)
    sources = tmp_path / "article.sources.json"
    sources.write_text(json.dumps({"values": []}), encoding="utf-8", newline="\n")
    assert script.main(["--sources", str(sources), "--timeout", "2.5"]) == 0
    assert asked == [2.5]


def test_the_default_dataset_is_the_public_logs_repo_of_the_publish_config():
    """The script reads no config (standard library only), so it names the dataset itself: the one
    ``lift publish datasets`` uploads the logs to, which anyone can read without a token."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "publish.yaml").read_text(encoding="utf-8"))
    logs = cfg["repos"]["logs"]
    assert script.LOGS_REPO == f"{logs.get('namespace') or cfg['namespace']}/{logs['name']}"
    assert logs["access"] == "public"


def test_the_https_fetcher_tells_a_missing_file_from_a_failure(monkeypatch):
    """Offline: urlopen is replaced. The Hub answers 404 with the error code EntryNotFound when the dataset
    and the revision exist and the file does not, and with RevisionNotFound for an unknown revision: only
    the first is a file the logs do not hold. Anything else stops the check."""
    asked: list[tuple[str, float, str | None]] = []

    def http_error(url: str, status: int, code: str | None, message: str) -> urllib.error.HTTPError:
        headers = email.message.Message()
        if code:
            headers["X-Error-Code"] = code
        headers["X-Error-Message"] = message
        return urllib.error.HTTPError(url, status, "Not Found", headers, None)

    def urlopen(request, timeout):
        url = request.full_url
        asked.append((url, timeout, request.get_header("User-agent")))
        if "/no-such-dataset/" in url:
            raise http_error(url, 401, None, "Invalid username or password.")
        if "/resolve/typo/" in url:
            raise http_error(url, 404, "RevisionNotFound", "Invalid rev id: typo")
        if url.endswith("/aggregates/final_report.json"):
            return io.BytesIO(b'{"version": 1}')
        if url.endswith("/aggregates/sweep_breakdown.json"):
            raise http_error(url, 404, "EntryNotFound", "Entry not found")
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(script.urllib.request, "urlopen", urlopen)
    base = "https://huggingface.co/datasets/org/logs/resolve"
    fetch = script.http_fetcher("org/logs", "main", 5.0)
    assert fetch("aggregates/final_report.json") == b'{"version": 1}'
    assert asked == [(f"{base}/main/aggregates/final_report.json", 5.0, "verify_numbers.py")]
    assert fetch("aggregates/sweep_breakdown.json") is None
    assert script.http_fetcher("org/logs", "refs/pr/1", 5.0)("aggregates/final_report.json")
    assert asked[-1][0] == f"{base}/refs%2Fpr%2F1/aggregates/final_report.json"  # a revision is one segment
    with pytest.raises(script.FetchError) as failure:
        script.http_fetcher("org/logs", "typo", 5.0)("aggregates/final_report.json")
    assert str(failure.value) == f"{base}/typo/aggregates/final_report.json: HTTP 404 (Invalid rev id: typo)"
    unknown = r"HTTP 401 \(Invalid username or password\.\): the Hub answers 401 for a dataset that does not"
    with pytest.raises(script.FetchError, match=unknown):
        script.http_fetcher("org/no-such-dataset", "main", 5.0)("aggregates/final_report.json")
    with pytest.raises(script.FetchError) as failure:
        fetch("manifests/run.json")
    assert str(failure.value).startswith(f"{base}/main/manifests/run.json: ")
    assert "no route to host" in str(failure.value)


def test_equality_is_that_of_the_parsed_json():
    same = script.same
    assert same(1, 1.0) and same({"a": [1, {"b": 2.0}], "c": None}, {"c": None, "a": [1.0, {"b": 2}]})
    assert same("a", "a") and same(None, None) and same(True, True) and same(float("nan"), float("nan"))
    assert not same(True, 1) and not same(0, False)  # Python's own == would call these equal
    assert not same(0.1, 0.1000001) and not same("1", 1) and not same(None, 0) and not same([], {})
    assert not same([1, 2], [1, 2, 3]) and not same([1, 2], [2, 1]) and not same({"a": 1}, {"a": 2})
    assert not same({"a": 1}, {"a": 1, "b": 2}) and not same({"a": 1}, {"b": 1})


def test_pointers_resolve_as_in_lift_publish_inputs():
    """The script has its own copy of the resolver (it imports nothing of the package): the two agree."""
    from lift.publish.inputs import resolve

    doc = {"a": {"b": [1, {"c": None}], "n": None}, "0": "zero", "rows": []}

    def outcome(function, pointer: str) -> tuple[str, object]:
        try:
            return "found", function(doc, pointer)
        except KeyError:
            return "missing", None

    for pointer in (
        "",
        "/",
        "/a",
        "/a/b/0",
        "/a/b/1",
        "/a/b/1/c",
        "/a/b/1/c/under/a/null",
        "/a/n/x",
        "/a/b/2",
        "/a/b/-1",
        "/a/b/x",
        "/a/missing",
        "/0",
        "/rows/0",
        "a/b/0",
        "/a//b/0/",
    ):
        assert outcome(script.resolve, pointer) == outcome(resolve, pointer), pointer
    assert outcome(script.resolve, "/a/b/1/c/under/a/null") == ("found", None)
    assert outcome(script.resolve, "/a/b/2") == ("missing", None)
    assert outcome(script.resolve, "") == ("found", doc)


def test_every_value_of_the_committed_sources_falls_in_a_class_the_script_checks(capsys):
    """No value of hf/article.sources.json is skipped: its file is a design config, a figure, or a file the
    logs dataset keeps a copy of. A value read from any other file fails here. A reader cannot check it, so
    either its file goes public and the script learns where, or the README says what is not checked."""
    values = script.read_sources(SOURCES)
    assert values, "hf/article.sources.json records no value"
    classes = Counter(script.source_class(entry["file"]) for entry in values)
    unknown = sorted({entry["file"] for entry in values if script.source_class(entry["file"]) is None})
    assert unknown == [], f"scripts/verify_numbers.py has no check for the values read from {unknown}"

    # With its defaults (the committed list and figures/, found from the script's own place) and a dataset
    # that holds nothing, every value is counted once, and the figures beside the README are the ones the
    # list recorded: a figure that differed would fail the check for every reader.
    code = script.main([], fetch=lambda path: None)
    out = capsys.readouterr().out.splitlines()
    assert [line for line in out if line.startswith("differs: ")] == []
    assert out[:3] == [
        f"0 values match the public logs; 0 differ; {classes['logs']} not in the logs",
        f"{classes['figure']} figures match the copies in figures/; 0 differ",
        f"{classes['config']} design values are in configs/*.yaml of this repository, not in the logs: not "
        "checked",
    ]
    assert len(out) == 3 + classes["logs"] and code == 0
    assert sum(classes.values()) == len(values)


def test_the_logs_builder_stages_every_cited_file_where_the_script_looks(tmp_path):
    """The script's map of outputs/ onto the logs dataset is the builder's own (lift.publish.datasets): given
    each file the committed write-up cites, ``lift publish datasets`` stages it at the path the script asks
    the dataset for. A cited file the builder does not publish fails here: no reader could check its values.
    (An AuditKIT record is staged only with its judged cell, which tests/test_publish_datasets.py covers.)"""
    from lift import paths
    from lift.publish import datasets
    from lift.publish.inputs import Facts

    files = {entry["file"] for entry in script.read_sources(SOURCES)}
    cited = sorted(
        f for f in files if script.source_class(f) == "logs" and not f.startswith("outputs/metrics/auditkit/")
    )
    assert "outputs/metrics/final_report.json" in cited
    write_inputs(figures=False)  # a selection.json and the final_report.json bound to it, frozen and tagged
    for file in cited:
        path = paths.output_dir() / file[len("outputs/") :]
        if not path.exists():
            write_json(path, {"made": "up"})
    staged = tmp_path / "staged"
    datasets._aggregates(staged, Facts())  # the aggregate, decision and validation files
    datasets._records(staged)  # the run manifests
    unpublished = [file for file in cited if not (staged / script.public_path(file)).is_file()]
    assert unpublished == [], f"lift publish datasets stages no copy of {unpublished} where the script looks"


def test_the_design_values_the_script_only_counts_are_those_of_the_committed_configs():
    """The script has no YAML parser, so it counts the values read from configs/*.yaml and checks none.
    This test does: each one the committed list records is the value at its pointer in the config."""
    import yaml

    docs: dict[str, object] = {}
    for entry in script.read_sources(SOURCES):
        file, pointer, recorded = entry["file"], entry["pointer"], entry["value"]
        if script.source_class(file) != "config":
            continue
        if file not in docs:
            docs[file] = yaml.safe_load((ROOT / file).read_text(encoding="utf-8"))
        # through JSON, as the list itself was written (a date becomes its text)
        found = json.loads(json.dumps(script.resolve(docs[file], pointer), default=str))
        assert script.same(recorded, found), f"{file} {pointer}: recorded {recorded!r}, found {found!r}"
