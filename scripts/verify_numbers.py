"""Check the numbers of the write-up against the public evaluation logs.

    python scripts/verify_numbers.py

Standard library only: nothing to install, no token, no GPU.

``lift publish article`` writes the README and, in the same run, hf/article.sources.json: every value the
text shows, with the file and the JSON pointer it was read from. This script takes that list and checks
each value against a copy anyone can read:

- A value read from outputs/ (the metrics files, a run manifest) is compared with the same file in the
  public logs dataset (``LOGS_REPO`` below, on the Hugging Face Hub), at the same pointer, for exact
  equality once both are parsed (an int and a float of the same value are equal). The dataset keeps what
  it publishes of outputs/metrics/ under aggregates/, the redacted AuditKIT records of
  outputs/metrics/auditkit/ under auditkit/ and the run manifests of outputs/manifests/ under manifests/
  (src/lift/publish/datasets.py). A value whose file the dataset does not keep, or whose pointer its copy
  does not have (the table of an analysis that was not run), is counted "not in the logs" and listed.
- A figure is recorded by the sha256 of its PNG, which is compared with the copy in figures/ that the
  README shows.
- A design number from configs/*.yaml (a set size, a threshold) lives in this repository, not in the logs.
  These are counted and not checked here (tests/test_verify_numbers.py compares them with the configs).

What it cannot check: that the README prints these values and no others (the list is trusted to be the
README's: one command writes both), and that the aggregates follow from the per-output records behind
them (``lift stats`` and ``lift report`` compute them from the labels, scores and verdicts; nothing is
recomputed here).

Exit code: 0 when no value differs, 1 when one does (a figure included), 2 when the list cannot be read or
a public file cannot be fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for the annotations only, which are not evaluated: the script also runs on Python 3.9
    from collections.abc import Callable, Sequence

    # A fetcher: the path of a file in the dataset -> its bytes, or None when the dataset has no such file.
    Fetch = Callable[[str], bytes | None]

ROOT = Path(__file__).resolve().parents[1]  # the repository: this file is scripts/verify_numbers.py
LOGS_REPO = "racetracklexsi/lift-eval-logs"
# Where the logs dataset keeps each tree of outputs/ (lift.publish.datasets): the first prefix that fits.
PUBLIC_TREES = (
    ("outputs/metrics/auditkit/", "auditkit/"),
    ("outputs/metrics/", "aggregates/"),
    ("outputs/manifests/", "manifests/"),
)
CUT = 100  # a value is shown cut to about this many characters
ABSENT = object()  # stands for a file the dataset does not hold


class SourcesError(Exception):
    """The list of values cannot be read."""


class FetchError(Exception):
    """A public file could not be fetched or parsed. The message names it and says what failed."""


def public_path(file: str) -> str | None:
    """Where the logs dataset keeps a file of outputs/, or None for a file it keeps no copy of."""
    for local, public in PUBLIC_TREES:
        if file.startswith(local):
            return public + file[len(local) :]
    return None


def source_class(file: str) -> str | None:
    """How a recorded value is checked, by its file: "config" (a design number in configs/*.yaml: counted),
    "figure" (a PNG, by its sha256), "logs" (a file the logs dataset keeps a copy of), or None: a file this
    script has no check for, whose values are listed as not in the logs."""
    if file.startswith("configs/") and file.endswith(".yaml"):
        return "config"
    if file.startswith("outputs/figures/") and file.endswith(".png"):
        return "figure"
    return "logs" if public_path(file) is not None else None


def resolve(doc: Any, pointer: str) -> Any:
    """``/a/b/0/c`` in ``doc``, by the rules of lift.publish.inputs.resolve: KeyError when a step is
    missing, and None for anything under a null (an arm that was not run)."""
    node = doc
    for part in (p for p in pointer.split("/") if p):
        if node is None:
            return None
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(pointer)
    return node


def same(a: Any, b: Any) -> bool:
    """Deep equality of two parsed JSON values. An int and a float of the same value are equal; true and
    false equal only themselves (Python alone would call true equal to 1)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(same, a, b))
    return a == b or (a != a and b != b)  # NaN, which json.loads reads, equals NaN here


def brief(value: Any) -> str:
    """A value as one line of JSON, cut to about CUT characters."""
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= CUT else text[:CUT] + "..."


def read_sources(path: Path) -> list[dict[str, Any]]:
    """The values of a sources file, ``{"values": [{"shown", "file", "pointer", "value"}, ...]}``, as
    ``Facts.write_sources`` (src/lift/publish/inputs.py) writes it."""
    try:
        doc = json.loads(path.read_bytes())
    except OSError as exc:
        raise SourcesError(
            f"{path} cannot be read ({exc.strerror or exc}): run this from a clone of the repository, or "
            "name the list with --sources"
        ) from exc
    except ValueError as exc:
        raise SourcesError(f"{path} is not valid JSON ({exc})") from exc
    values = doc.get("values") if isinstance(doc, dict) else None
    if not isinstance(values, list):
        raise SourcesError(f'{path} has no "values" list: it is not a sources file of lift publish article')
    for i, entry in enumerate(values):
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("file"), str)
            and isinstance(entry.get("pointer"), str)
            and "value" in entry
        ):
            raise SourcesError(f'{path}: values[{i}] does not have a "file", a "pointer" and a "value"')
    return values


def http_fetcher(repo: str, revision: str, timeout: float) -> Fetch:
    """A fetcher of the dataset's files over HTTPS, with no token: the file's bytes, None when the dataset
    has no such file, FetchError for anything else (no network, an unknown dataset or revision)."""
    base = f"https://huggingface.co/datasets/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/"

    def fetch(path: str) -> bytes | None:
        url = base + urllib.parse.quote(path)
        request = urllib.request.Request(url, headers={"User-Agent": "verify_numbers.py"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # The Hub says "EntryNotFound" when the dataset and the revision exist and the file does not.
            if exc.code == 404 and exc.headers.get("X-Error-Code") == "EntryNotFound":
                return None
            said = exc.headers.get("X-Error-Message") or exc.reason
            hint = ": the Hub answers 401 for a dataset that does not exist or is not public"
            raise FetchError(f"{url}: HTTP {exc.code} ({said})" + (hint if exc.code == 401 else "")) from exc
        except (OSError, http.client.HTTPException) as exc:  # no network, a timeout, a certificate
            raise FetchError(f"{url}: {exc}") from exc

    return fetch


def verify(values: list[dict[str, Any]], fetch: Fetch, figures: Path) -> tuple[Counter[str], list[str]]:
    """Check every recorded value. Returns how many had each outcome ("match", "differ" and "absent" for
    the logs, "figure" and "figure differs", "config") and one line per value that differs or is absent,
    the differing ones first."""

    fetched: dict[str, Any] = {}

    def public(path: str) -> Any:
        """A file of the dataset, parsed, or ABSENT. Each one is fetched once and kept in memory."""
        if path not in fetched:
            data = fetch(path)
            try:
                fetched[path] = ABSENT if data is None else json.loads(data)
            except ValueError as exc:
                raise FetchError(f"{path} of the logs dataset is not valid JSON ({exc})") from exc
        return fetched[path]

    counts: Counter[str] = Counter()
    differing: list[str] = []
    absent: list[str] = []
    for entry in values:
        file, pointer, recorded = entry["file"], entry["pointer"], entry["value"]
        kind, path = source_class(file), public_path(file)
        if kind == "config":
            outcome, note = "config", ""
        elif kind == "figure":
            copy = figures / file.rsplit("/", 1)[-1]
            found = hashlib.sha256(copy.read_bytes()).hexdigest() if copy.is_file() else None
            outcome = "figure" if same(recorded, found) else "figure differs"
            note = f"{figures.name}/{copy.name} " + (f"has {brief(found)}" if found else "is not there")
        elif path is None:
            outcome, note = "absent", "the logs dataset keeps no copy of this file"
        elif public(path) is ABSENT:
            outcome, note = "absent", f"the logs dataset has no {path}"
        else:
            try:
                found = resolve(public(path), pointer)
            except KeyError:
                outcome, note = "absent", f"{path} has no such pointer"
            else:
                outcome, note = ("match" if same(recorded, found) else "differ"), f"public {brief(found)}"
        counts[outcome] += 1
        said = f"{file} {pointer}".strip() + f": recorded {brief(recorded)}"
        if outcome in ("differ", "figure differs"):
            differing.append(f"differs: {said}, {note}")
        elif outcome == "absent":
            absent.append(f"not in the logs: {said} ({note})")
    return counts, differing + absent


def seconds(text: str) -> float:
    """A --timeout value: a finite number of seconds above 0. argparse reports anything else as a usage
    error (exit 2), so a bad value never looks like a value that differs (exit 1)."""
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not 0 < value < float("inf"):  # false for NaN too
        raise argparse.ArgumentTypeError(f"{text!r} is not a number of seconds above 0")
    return value


def say(line: str, *, error: bool = False) -> None:
    """Print one line, with anything outside ASCII escaped so that every console can show it."""
    print(line.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr if error else sys.stdout)


def main(argv: Sequence[str] | None = None, fetch: Fetch | None = None) -> int:
    """Run the check and print its summary. ``fetch`` replaces the HTTPS fetcher (the tests pass a fake)."""
    parser = argparse.ArgumentParser(
        description="Check the numbers of the write-up against the public evaluation logs.",
        epilog="Exit code: 0 when no value differs, 1 when one does, 2 when the check could not be made.",
    )
    add = parser.add_argument
    add("--sources", type=Path, default=ROOT / "hf" / "article.sources.json", metavar="PATH",
        help="the list of values to check (default: hf/article.sources.json of this repository)")
    add("--figures", type=Path, default=ROOT / "figures", metavar="DIR",
        help="the figures the README shows (default: figures/ of this repository)")
    add("--repo", default=LOGS_REPO, metavar="ID", help=f"the public logs dataset (default: {LOGS_REPO})")
    add("--revision", default="main", metavar="REV", help="its branch, tag or commit (default: main)")
    add("--timeout", type=seconds, default=30.0, metavar="SECONDS",
        help="how long to wait for each file (default: 30)")
    args = parser.parse_args(argv)
    try:
        values = read_sources(args.sources)
        fetch = fetch or http_fetcher(args.repo, args.revision, args.timeout)
        counts, lines = verify(values, fetch, args.figures)
    except (SourcesError, FetchError) as exc:
        say(f"verify_numbers: {exc}", error=True)
        return 2
    c = counts
    say(f"{c['match']} values match the public logs; {c['differ']} differ; {c['absent']} not in the logs")
    say(f"{c['figure']} figures match the copies in {args.figures.name}/; {c['figure differs']} differ")
    say(f"{c['config']} design values are in configs/*.yaml of this repository, not in the logs: not checked")
    for line in lines:
        say(line)
    return 1 if c["differ"] or c["figure differs"] else 0


if __name__ == "__main__":
    sys.exit(main())
