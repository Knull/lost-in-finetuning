"""``lift publish preflight``: may this git repo be made public? Also the text scan every public upload
goes through (``scan_folder``, called by ``lift.publish.hub.check_ready`` and the logs dataset build).

It fails if any tracked file, or any file in the git history (every blob reachable from any ref,
since making a repo public publishes its history too), holds:

- a secret or token (HF, GitHub, Anthropic, OpenAI-style, Google API and OAuth, AWS, Kaggle, a private
  key);
- a raw completion: a JSON line shaped like a GenerationRecord (key, variant, output) or an AuditKIT
  record that keeps outputs, a JudgementRecord line that keeps the judge's raw output, or a
  UtilityVerdict line; or any 60-character window (taken every 30 characters, so any run of 90 characters
  or more is caught) of any model output in this machine's outputs/, whatever its label (decisions
  item 62: for about 90% of outputs the label is Qwen3Guard's alone, so the guard reads content, never
  labels). A window of fewer than MIN_DISTINCT distinct characters is no one's text (a rule of dashes, a
  repeated syllable; a tokenizer's vocabulary holds such runs) and is left out.
  Both sides are normalised the same way (``_norm``: NFKC, format characters such as zero-width spaces,
  joiners and soft hyphens removed, case folded, whitespace collapsed), so an editor's Unicode form, a
  change of case or an inserted zero-width character does not hide a copy. JSON, JSONL, notebook and CSV
  files are also scanned as their decoded string values, and any text with HTML entities also unescaped,
  so an escaped text is found too. A window that is also a run of a prompt text (``data/prompts``, any
  language, the English source too) is dropped: prompts are public by design, and an output that
  restates its prompt must not block the prompt set. Only the echoed run goes: a window that reaches
  into the completion's own text stays;
- a notebook committed with outputs or execution counts (a Colab run saved with its results);
- a path that must never be committed: outputs/, data/ outside data/splits/, validation sheets, labels
  and keys, model weights, .env, *.token, kaggle.json.

Every text blob is scanned, however large (a big one in chunks, the blobs streamed one at a time from
``git cat-file``); only binary blobs are path-checked.
The text check needs this machine's outputs: ``load_windows`` refuses when there are no generations
here, or a judgement names a key with no generation here (``lift sync pull`` first), so the check is
never a silent no-op. GitHub publishes every branch, tag and pull-request ref of the remote, not only
the local ones: the CLI fetches them all first (``fetch_public_refs``), and the steps list what a person
must still read by hand (PR and issue text, Actions logs).

A finding names the file, line and kind, never the matched text. A line holding ``preflight: allow``
is skipped (for a documented fake in a test). When it passes it prints the steps to make the repo public.

Absolute local paths (review XB2-04). A staged public file must name no home dir (``/home/<user>``,
``/Users/<user>``, ``/root``), temp dir (``/tmp``, ``/var/folders``...), GPU VM dir (``/content``,
``/kaggle``) or drive letter (``C:``): the public-text scan (``scan_folder``) finds them, a tokenizer's
vocabulary aside (``VOCAB_FILES``). The staging rewrites them first (``relativise_folder``): a path under
this machine's output, validation or data dir or the repo becomes repo-relative (``outputs/...``,
``configs/...``), one under another machine's ``outputs/`` tree (a GPU VM's) is cut to it, a Hugging Face
cache snapshot becomes its repo id; any other is withheld, and the upload refused (``PublishError``).
The tracked files are not checked for them (docs name /tmp and drive letters by design).
"""

from __future__ import annotations

import csv
import functools
import html
import io
import json
import re
import subprocess
import sys
import threading
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from lift import paths
from lift.publish import PublishError

ALLOW_MARK = "preflight: allow"
SECRETS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"),
    "Anthropic API key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    "Google OAuth token": re.compile(r"\bya29\.[0-9A-Za-z_-]{20,}"),
    "Google OAuth client secret": re.compile(r"\bGOCSPX-[0-9A-Za-z_-]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Kaggle API key": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "Kaggle API key (KAGGLE_KEY)": re.compile(r"\bKAGGLE_KEY\b[\"']?\s*[:=]\s*[\"']?[0-9a-f]{32}\b"),
}
FORBIDDEN_PATHS = (
    "outputs/*",
    "validation/sheets/*",
    "validation/labels/*",
    "validation/keys/*",
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.gguf",
    ".env",
    "*/.env",
    "*.token",
    "kaggle.json",
    "*/kaggle.json",
)
WINDOW = 60
STRIDE = 30  # windows start every STRIDE characters, so any run of WINDOW + STRIDE characters holds one
CHUNK_CHARS = 8 << 20  # a longer text is scanned a chunk at a time (windows_in), never skipped
MIN_DISTINCT = 8  # a window of fewer distinct characters (a rule of dashes, "ha ha ha") is no one's text
OUTPUT_TEXT = "text of a model output"
NOTEBOOK_OUTPUTS = "notebook with outputs or execution counts"
JSON_SUFFIXES = (".json", ".jsonl", ".ipynb")


LOCAL_PATH_KIND = "absolute local path (a home dir, /tmp, a VM dir or a drive)"
_LOCAL_PATH = re.compile(
    r"(?<![\w.~$%/\\:-])"
    r"(?:/(?:home|Users|root)/"
    r"|/(?:tmp|var/tmp|var/folders|private/var|private/tmp|content|kaggle)(?=[/\\\s\"'`]|$)"
    r"|[A-Za-z]:[\\/])"
    r"[^\s\"'`<>|]*"
)
_TRAILING = ".,;:)]}"
# A tokenizer's vocabulary may hold tokens such as "C:\\" or "/home/": not paths, and never rewritten.
VOCAB_FILES = frozenset(
    {"tokenizer.json", "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model", "spiece.model"}
)
# A tokenizer's own files at the top level of a model or adapter upload: upstream's public text (the
# vocabulary, and the chat template with Tiny Aya's default preamble). They are scanned for secrets, raw
# records and local paths like any file, but not for model-output windows: an output that restates the
# preamble would block the model upload, and nothing the pipeline writes goes into these files.
TOKENIZER_FILES = VOCAB_FILES | {
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "special_tokens_map.json",
    "added_tokens.json",
}
_HF_SNAPSHOT = re.compile(r"models--(?P<org>[^/\\]+)--(?P<name>[^/\\]+)[/\\]snapshots[/\\][0-9a-f]+")


def local_paths(text: str) -> list[str]:
    """The absolute local paths in ``text`` (module docstring), trailing punctuation left out."""
    return [m.group(0).rstrip(_TRAILING) for m in _LOCAL_PATH.finditer(text)]


def _roots() -> list[tuple[str, str]]:
    """(a root as text, what it becomes), longest first: this machine's trees and the repo."""
    forms = set()
    for root, label in (
        (paths.output_dir(), "outputs"),
        (paths.validation_dir(), "validation"),
        (paths.data_dir(), "data"),
        (paths.REPO_ROOT, ""),
    ):
        for form in {str(root), root.as_posix(), str(root.resolve()), root.resolve().as_posix()}:
            form = form.rstrip("/\\")
            if len(form) > 3:  # never "/" or "C:"
                forms.add((form, label))
    return sorted(forms, key=lambda f: -len(f[0]))


def relative_text(text: str) -> tuple[str, int, list[str]]:
    """``text`` with its absolute local paths made repo-relative (module docstring): (the text, how many
    were rewritten, the ones that cannot be)."""
    n = 0
    for form, label in _roots():
        if form not in text:
            continue
        pattern = re.compile(re.escape(form) + r"(?:(?P<sep>[\\/])|(?=[\s\"'`<>|),;:\]}]|$))")

        def root(m: re.Match[str], label: str = label) -> str:
            nonlocal n
            n += 1
            if m.group("sep"):
                return f"{label}/" if label else ""
            return label or "."

        text = pattern.sub(root, text)
    unresolved: list[str] = []

    def other(m: re.Match[str]) -> str:
        nonlocal n
        token = m.group(0)
        core = token.rstrip(_TRAILING)
        tail = token[len(core) :]
        snap = _HF_SNAPSHOT.search(core)
        if snap:
            n += 1
            return f"{snap['org']}/{snap['name']}{tail}"
        parts = [p for p in re.split(r"[\\/]+", core) if p]
        if "outputs" in parts[1:]:
            n += 1
            i = len(parts) - 1 - parts[::-1].index("outputs")
            return "/".join(parts[i:]) + tail
        unresolved.append(core)
        return token

    return _LOCAL_PATH.sub(other, text), n, unresolved


def _relative_json(value: object, count: list[int], unresolved: list[str]) -> object:
    if isinstance(value, str):
        text, n, bad = relative_text(value)
        count[0] += n
        unresolved += bad
        return text
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[_relative_json(k, count, unresolved)] = _relative_json(v, count, unresolved)
        return out
    if isinstance(value, list):
        return [_relative_json(v, count, unresolved) for v in value]
    return value


def relativise_file(path: Path) -> tuple[int, list[str]]:
    """Rewrite one text file's absolute local paths in place (JSON and JSONL by their string values, so
    an escaped path is read as it is): (how many were rewritten, the ones that cannot be). A binary file,
    a tokenizer's vocabulary or a file with none is left as it is."""
    path = Path(path)
    if path.name in VOCAB_FILES:
        return 0, []
    raw = path.read_bytes()
    if _binary(raw[:8192]):
        return 0, []
    text = raw.decode("utf-8", "replace")
    if not _LOCAL_PATH.search(text) and not any(form in text for form, _ in _roots()):
        return 0, []
    count, unresolved = [0], []
    new = None
    try:
        if path.suffix in (".json", ".ipynb"):
            doc = _relative_json(json.loads(text), count, unresolved)
            new = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
        elif path.suffix == ".jsonl":
            lines = []
            for line in text.splitlines():
                if line.strip():
                    row = _relative_json(json.loads(line), count, unresolved)
                    lines.append(json.dumps(row, ensure_ascii=False))
            new = "".join(f"{line}\n" for line in lines)
    except json.JSONDecodeError:
        count, unresolved, new = [0], [], None
    if new is None:
        new, count[0], unresolved = relative_text(text)
    if count[0]:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
    return count[0], unresolved


def relativise_folder(folder: Path) -> dict[str, int]:
    """Make every absolute local path in a staged upload repo-relative (module docstring): {file: paths
    rewritten}. PublishError, naming each file, when one cannot be (withheld: nothing is published)."""
    done: dict[str, int] = {}
    bad: dict[str, int] = {}
    for f in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        n, unresolved = relativise_file(f)
        rel = f.relative_to(folder).as_posix()
        if n:
            done[rel] = n
        if unresolved:
            bad[rel] = len(unresolved)
    if bad:
        raise PublishError(
            "withheld, nothing was published: "
            + "; ".join(f"{rel} holds {k} absolute local path(s)" for rel, k in bad.items())
            + " outside the repo and any outputs/ tree (a home dir, /tmp, a VM dir or a drive): have the "
            "stage that writes it record the path relative to the repo"
        )
    return done


def relativised_note(done: dict[str, int]) -> list[str]:
    """The plan's note on what ``relativise_folder`` rewrote ([] when nothing)."""
    if not done:
        return []
    return [
        f"{sum(done.values())} absolute local path(s) staged repo-relative in {', '.join(sorted(done))}"
        + (
            " (selection.json's bytes then differ from the sha256 the recipe pins)"
            if any(f.endswith("selection.json") for f in done)
            else ""
        )
    ]


@dataclass(frozen=True)
class Finding:
    where: str  # "tree", "history <short sha>" or "staged"
    path: str
    line: int
    kind: str
    detail: str = ""  # where in a JSON file a model-output window is (field path and id), never the text

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.kind}: {loc} ({self.where})" + (f", in {self.detail}" if self.detail else "")


# Zero-width characters that stand for a space (they may replace one, or sit inside a word).
ZERO_WIDTH_SPACES = "\u200b\u2060\ufeff\u180e"
_ZW_SPACE = re.compile(f"[{ZERO_WIDTH_SPACES}]")
MODES = ("space", "drop")  # a zero-width space read as a space, or dropped like every other format char


@functools.cache
def _format_chars() -> re.Pattern[str]:
    """Every character of Unicode category Cf (format: zero-width joiners, soft hyphen, bidi marks...)."""
    cf = [chr(c) for c in range(sys.maxunicode + 1) if unicodedata.category(chr(c)) == "Cf"]
    return re.compile("[" + "".join(re.escape(c) for c in cf) + "]")


def _prenorm(text: str, mode: str = "space") -> str:
    """``text`` with its format characters dropped (zero-width spaces made spaces in mode "space"), NFKC
    normalised and case folded; whitespace still as it was (``_norm`` collapses it)."""
    if mode == "space":
        text = _ZW_SPACE.sub(" ", text)
    text = _format_chars().sub("", text)
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())


def _norm(text: str, mode: str = "space") -> str:
    """The form both sides of the window scan are compared in: ``_prenorm``, whitespace collapsed."""
    return " ".join(_prenorm(text, mode).split())


def _modes(text: str) -> tuple[str, ...]:
    """A text holding a zero-width space is read both ways: as a space (one put in place of a space) and
    dropped (one put inside a word)."""
    return MODES if _ZW_SPACE.search(text) else MODES[:1]


def windows_of(texts: Iterable[str]) -> set[str]:
    """The WINDOW-character windows of each text (normalised, ``_norm``) starting every STRIDE characters,
    plus its last window: a copy of any WINDOW + STRIDE characters of a text holds one of them whole. A
    window of fewer than MIN_DISTINCT distinct characters is left out (the module docstring says why)."""
    out: set[str] = set()
    for t in texts:
        for mode in _modes(t):
            n = _norm(t, mode)
            if len(n) < WINDOW:
                continue
            out.update(n[s : s + WINDOW] for s in range(0, len(n) - WINDOW + 1, STRIDE))
            out.add(n[-WINDOW:])
    return {w for w in out if len(set(w)) >= MIN_DISTINCT}


def has_window(text: str, windows: set[str] | frozenset) -> bool:
    """Whether ``text`` (normalised, ``_norm``) holds one of ``windows`` anywhere."""
    return any(_in_flat(_norm(text, mode), windows) for mode in _modes(text))


def windows_in_texts(texts: Iterable[str], windows: set[str]) -> set[str]:
    """The windows that occur anywhere in ``texts`` (normalised like the scan)."""
    hit: set[str] = set()
    if not windows:
        return hit
    for t in texts:
        for mode in _modes(t):
            flat = _norm(t, mode)
            hit.update(w for s in range(0, len(flat) - WINDOW + 1) if (w := flat[s : s + WINDOW]) in windows)
    return hit


def prompt_texts() -> Iterator[str]:
    """Every prompt text on this machine (``data/prompts/*.jsonl``: the text and its English source), and
    each with its HTML entities unescaped where that changes it: the scan also reads a file that way
    (``_bodies``), so a prompt holding an entity (``&quot;``, or ``&not`` inside a word) that an output
    echoes decoded is still the prompt, not the output's own text."""
    from lift.io import read_jsonl

    src = paths.prompts_dir()
    for f in sorted(src.glob("*.jsonl")) if src.is_dir() else []:
        for row in read_jsonl(f):
            for field in ("text", "text_en"):
                text = row.get(field)
                if isinstance(text, str):
                    for form in (text, html.unescape(text)) if "&" in text else (text,):
                        yield form
                        yield f'"{form}"'  # quoted, as an output that quotes its prompt holds it


def _in_flat(flat: str, windows: set[str] | frozenset) -> bool:
    return bool(windows) and any(flat[s : s + WINDOW] in windows for s in range(0, len(flat) - WINDOW + 1))


def output_windows() -> set[str]:
    """The windows of every model output in this machine's outputs/, whatever its label (may be empty:
    the publish paths use ``load_windows``, which refuses an empty set)."""
    from lift.publish.datasets import output_record

    return windows_of(output_record().texts)


def load_windows() -> tuple[set[str], int, int]:
    """(windows, number of outputs they come from, number of judged outputs whose text is no longer on
    this machine), or PublishError when this machine cannot say what the outputs are: no generations
    here, judgements of keys whose generations are not here, or no window at all. A check against an
    empty set would pass anything."""
    from lift.publish.datasets import output_record

    pull = "run `lift sync pull` first (outputs/ and validation/ come only from the private runs repo)"
    found = output_record()
    if not found.texts:
        raise PublishError(
            f"no model outputs found on this machine (no generations in {paths.generations_dir().name}/), "
            f"so the text check would be empty: {pull}"
        )
    if found.missing:
        raise PublishError(
            f"{len(found.missing)} judged outputs have no generation on this machine (e.g. in "
            f"{', '.join(found.missing_files[:3])}), so their text cannot be checked: {pull}, or start the "
            "cell over with its lift restart line"
        )
    windows = windows_of(found.texts)
    # P08-R2-03: a run the output copied from a prompt is public anyway (the prompt sets); a window that
    # reaches into the completion's own text is never a run of a prompt, so it stays.
    windows -= windows_in_texts(prompt_texts(), windows)
    if not windows:
        raise PublishError(
            f"no model outputs found on this machine ({len(found.texts)} outputs, but none of {WINDOW} "
            f"characters or more beyond what it repeats of a prompt), so the text check would be empty: "
            f"{pull}"
        )
    return windows, len(found.texts), found.gone


def forbidden_path(path: str) -> bool:
    if path.startswith("data/") and not path.startswith("data/splits/"):
        return True
    return any(fnmatch(path, pat) for pat in FORBIDDEN_PATHS)


# An AuditKIT record line that keeps an output (a redacted record stores "[redacted sha256:...]").
_KEPT_OUTPUT = re.compile(r'"actual_output"\s*:\s*"(?!\[redacted)[^"]')


_RECORD_FIELDS = ('"output"', '"raw"', '"order_ab"')


def _raw_record(line: str) -> bool:
    """A line that is (or keeps) a raw completion: an AuditKIT record keeping an output, a
    GenerationRecord with its output, a JudgementRecord with the judge's raw output (a frontier judge's
    reasoning can quote the completion), or a UtilityVerdict (its reasons quote both answers)."""
    s = line.strip()
    if _KEPT_OUTPUT.search(s):
        return True
    if not (s.startswith("{") and any(f in s for f in _RECORD_FIELDS)):
        return False
    try:
        row = json.loads(s)
    except json.JSONDecodeError:
        return False
    if not isinstance(row, dict):
        return False
    if {"key", "variant", "output"} <= row.keys() and bool(str(row.get("output") or "").strip()):
        return True
    if {"key", "judge", "label"} <= row.keys() and bool(str(row.get("raw") or "").strip()):
        return True
    return {"pid", "variant", "judge", "order_ab", "order_ba"} <= row.keys()


def allowed_lines(text: str) -> set[str]:
    """The lines a file marks as documented fakes, without the mark, so the same line in an older
    commit (written before it was marked) is allowed too."""
    out = set()
    for line in text.splitlines():
        if ALLOW_MARK in line:
            out.add(line.split("#")[0].strip() if "#" in line else line.replace(ALLOW_MARK, "").strip())
    return out


def _strings(value: object, *, keys: bool = False) -> Iterator[str]:
    """Every string value in a decoded JSON document (its keys too with ``keys``)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if keys and isinstance(k, str):
                yield k
            yield from _strings(v, keys=keys)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v, keys=keys)


def decoded_values(text: str, path: str) -> list[str] | None:
    """A JSON, JSONL or notebook file's strings, each value and each key on its own (a CSV file's cells; a
    JSONL line that does not parse, as it is), for the model-output scan: a window is looked for inside
    one value, never across the file's syntax or two values. None for other files, or a file that does
    not parse. Reading the raw text of a JSON file instead made a quoted prompt match: an output that
    quotes its prompt ("..." in its own text) holds a window of the quote and the prompt, and the raw
    file holds the same around the prompt's value (decisions item 109)."""
    try:
        if path.endswith((".json", ".ipynb")):
            return list(_strings(json.loads(text), keys=True))
        if path.endswith(".jsonl"):
            out = []
            for line in text.splitlines():
                if line.strip():
                    try:
                        out.extend(_strings(json.loads(line), keys=True))
                    except json.JSONDecodeError:
                        out.append(line)
            return out
        if path.endswith(".csv"):
            return [cell for row in csv.reader(io.StringIO(text)) for cell in row]
    except (json.JSONDecodeError, csv.Error, RecursionError):
        return None
    return None


def decoded_text(text: str, path: str) -> str | None:
    """A JSON, JSONL or notebook file's string values (a CSV file's cells), one per line, so that an
    escaped text (\\n, \\uXXXX, doubled quotes) is scanned as it reads. None for other files, or a file
    that does not parse."""
    try:
        if path.endswith((".json", ".ipynb")):
            return "\n".join(_strings(json.loads(text)))
        if path.endswith(".jsonl"):
            out = []
            for line in text.splitlines():
                if line.strip():
                    try:
                        out.extend(_strings(json.loads(line)))
                    except json.JSONDecodeError:
                        out.append(line)
            return "\n".join(out)
        if path.endswith(".csv"):
            return "\n".join(cell for row in csv.reader(io.StringIO(text)) for cell in row)
    except (json.JSONDecodeError, csv.Error, RecursionError):
        return None
    return None


SAFE_CUT_SEARCH = 4096  # how far past a chunk's end to look for a character a cut may go before


def _safe_cut(ch: str) -> bool:
    """Whether a chunk may end right before ``ch``: the normal form of the text on each side then does
    not depend on the other (no combining mark, no Hangul jamo, no entity character)."""
    if ch.isspace():
        return True
    if unicodedata.combining(ch) or ch in "&#;" or "\u1100" <= ch <= "\u11ff":
        return False
    return unicodedata.category(ch)[0] in "LNPS" and not (
        "\ua960" <= ch <= "\ua97f" or "\ud7b0" <= ch <= "\ud7ff"
    )


def _chunks(text: str) -> Iterator[str]:
    """``text`` in pieces of about CHUNK_CHARS, each cut right before a character ``_safe_cut`` allows
    (within SAFE_CUT_SEARCH characters; past that the cut is forced)."""
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        limit = min(len(text), end + SAFE_CUT_SEARCH)
        while end < limit and not _safe_cut(text[end]):
            end += 1
        yield text[start:end]
        start = end


def windows_in(text: str, windows: set[str] | frozenset) -> bool:
    """``has_window`` for a text of any size: one longer than CHUNK_CHARS is normalised and scanned a
    chunk at a time (cut where normalising each side alone gives the same text), each chunk led by the
    last WINDOW - 1 normalised characters before it, so a window across a cut is still found (the result
    equals ``has_window`` on the whole text)."""
    if len(text) <= CHUNK_CHARS:
        return has_window(text, windows)
    if not windows:
        return False
    return any(_windows_in_chunks(text, windows, mode) for mode in _modes(text))


def _windows_in_chunks(text: str, windows: set[str] | frozenset, mode: str) -> bool:
    carry, space_before = "", False
    for chunk in _chunks(text):
        pre = _prenorm(chunk, mode)
        if not pre:
            continue  # only format characters: nothing between the neighbours
        flat = " ".join(pre.split())
        if not flat:
            space_before = True
            continue
        sep = " " if carry and (space_before or pre[0].isspace()) else ""
        joined = carry + sep + flat
        if _in_flat(joined, windows):
            return True
        carry, space_before = joined[-(WINDOW - 1) :], pre[-1].isspace()
    return False


def notebook_has_outputs(text: str) -> bool:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return False
    cells = doc.get("cells") if isinstance(doc, dict) else None
    return any(
        isinstance(c, dict) and (c.get("outputs") or c.get("execution_count") is not None)
        for c in cells or []
    )


def scan_text(
    text: str, path: str, where: str, windows: set[str], allowed: set[str] | frozenset = frozenset()
) -> list[Finding]:
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line or line.strip() in allowed:
            continue
        for kind, pat in SECRETS.items():
            if pat.search(line):
                found.append(Finding(where, path, i, kind))
        if _raw_record(line):
            found.append(Finding(where, path, i, "raw completion record"))
    if path.endswith(".ipynb") and notebook_has_outputs(text):
        found.append(Finding(where, path, 0, NOTEBOOK_OUTPUTS))
    if windows:
        values = decoded_values(text, path)
        # an ASCII string shorter than a window cannot hold one (_norm never lengthens ASCII): ids, labels
        long = (v for v in values or () if len(v) >= WINDOW or not v.isascii())
        bodies = _bodies(text, None) if values is None else (b for v in long for b in _bodies(v, None))
        for body in bodies:
            if windows_in(body, windows):
                line = _first_line(text, windows) if values is None else 0
                found.append(Finding(where, path, line, OUTPUT_TEXT, _json_where(text, path, windows)))
                break
    return found


def _first_line(text: str, windows: set[str] | frozenset) -> int:
    """For a text file (not JSON), the first line where a model-output window starts, the window running
    onto the next line at most: a line number, never the text. 0 when only a longer stretch holds one."""

    def holds(s: str) -> bool:
        return any(windows_in(b, windows) for b in _bodies(s, None))

    lines = text.splitlines()
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if holds(line) or (nxt and holds(f"{line}\n{nxt}") and not holds(nxt)):
            return i + 1
    return 0


def _json_where(text: str, path: str, windows: set[str] | frozenset) -> str:
    """For a JSON file, the string values that hold a model-output window, as field paths with the id of
    the object holding each (an AuditKIT sample's pid): names and ids only, never the text. "" for another
    file, or when no single value holds one (a window across two values)."""
    if not path.endswith(".json"):
        return ""
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return ""
    hits: list[str] = []

    def walk(value: object, at: str, owner: object) -> None:
        if isinstance(value, str):
            if any(windows_in(b, windows) for b in _bodies(value, None)):
                oid = owner.get("id") if isinstance(owner, dict) else None
                hits.append(at + (f" (id {oid})" if isinstance(oid, str) and len(oid) <= 80 else ""))
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, str) and any(windows_in(b, windows) for b in _bodies(k, None)):
                    hits.append(f"a key of {at or 'the top level'}")
                walk(v, f"{at}/{k}" if at else str(k), value)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{at}/{i}", owner)

    walk(doc, "", None)
    if not hits:
        return ""
    more = len(hits) - 3
    return ", ".join(hits[:3]) + (f" and {more} more" if more > 0 else "")


def _bodies(text: str, decoded: str | None) -> Iterator[str]:
    """The text as it is, its decoded string values, and each of those with HTML entities unescaped."""
    for body in (text, decoded):
        if body is None:
            continue
        yield body
        if "&" in body:
            unescaped = html.unescape(body)
            if unescaped != body:
                yield unescaped


def auditkit_problem(doc: object) -> str | None:
    """Why a public AuditKIT record is not redacted (None when it is): ``metadata.lift.text`` must be
    "redacted" and every ``actual_output`` a ``[redacted sha256:...]`` stand-in."""
    meta = doc.get("metadata") if isinstance(doc, dict) else None
    lift_meta = meta.get("lift") if isinstance(meta, dict) else None
    if not isinstance(lift_meta, dict) or lift_meta.get("text") != "redacted":
        return 'its metadata.lift.text is not "redacted"'

    def outputs(value: object) -> Iterator[object]:
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "actual_output":
                    yield v
                else:
                    yield from outputs(v)
        elif isinstance(value, list):
            for v in value:
                yield from outputs(v)

    kept = sum(1 for v in outputs(doc) if v is not None and not str(v).startswith("[redacted"))
    return f"{kept} actual_output values are not redacted" if kept else None


def _binary(head: bytes) -> bool:
    return b"\0" in head[:8192]


def scan_folder(folder: Path, windows: set[str], *, where: str = "staged") -> list[Finding]:
    """Scan every text file of a staged upload as the preflight scans a tracked file (secrets, raw
    completion records, notebooks with outputs, model-output windows) and for absolute local paths
    (``LOCAL_PATH_KIND``, a tokenizer's vocabulary aside), and every AuditKIT record (a .json under an
    ``auditkit`` folder) for redaction. Binary files (weights) are skipped, and a tokenizer's own files
    at the top level (``TOKENIZER_FILES``) are not read for model-output windows."""
    found: list[Finding] = []
    for f in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        rel = f.relative_to(folder).as_posix()
        with open(f, "rb") as fh:
            if _binary(fh.read(8192)):
                continue
        text = f.read_bytes().decode("utf-8", "replace")
        upstream = "/" not in rel and f.name in TOKENIZER_FILES
        found += scan_text(text, rel, where, set() if upstream else windows)
        if f.name not in VOCAB_FILES:
            found += [
                Finding(where, rel, i, LOCAL_PATH_KIND)
                for i, line in enumerate(text.splitlines(), 1)
                if ALLOW_MARK not in line and _LOCAL_PATH.search(line)
            ]
        if "auditkit" in rel.split("/")[:-1] and rel.endswith(".json"):
            try:
                problem = auditkit_problem(json.loads(text))
            except json.JSONDecodeError:
                problem = "it is not valid JSON"
            if problem:
                found.append(Finding(where, rel, 0, f"AuditKIT record that is not redacted ({problem})"))
    return found


def _git(*args: str, cwd: Path, input: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, input=input, capture_output=True, check=True).stdout


def tracked_files(repo: Path) -> list[str]:
    return [p for p in _git("ls-files", "-z", cwd=repo).decode("utf-8").split("\0") if p]


def _history_blobs(repo: Path) -> dict[str, str]:
    """{blob sha: one path it had} for every blob reachable from any ref."""
    out: dict[str, str] = {}
    for line in _git("rev-list", "--all", "--objects", cwd=repo).decode("utf-8", "replace").splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in out:
            out[sha] = path
    return out


def _cat_blobs(repo: Path, shas: Iterable[str]) -> Iterator[tuple[str, str, bytes | None]]:
    """(sha, kind, body for a blob) for each object, streamed one at a time from ``git cat-file --batch``
    (the history is never held in one buffer: P08-R2-16). The shas are written from a thread, so neither
    pipe can fill up while the other waits."""
    shas = list(shas)
    if not shas:
        return
    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    assert proc.stdin is not None and proc.stdout is not None

    def feed() -> None:
        try:
            for sha in shas:
                proc.stdin.write(sha.encode() + b"\n")
            proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass  # the reader stopped early (or git died): the read side reports it

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        for _ in shas:
            header = proc.stdout.readline()
            if not header:
                raise PublishError("git cat-file stopped before every blob of the history was read")
            sha, kind, *rest = header.decode().split()
            if kind == "missing":
                continue
            size = int(rest[0])
            body = proc.stdout.read(size)
            proc.stdout.read(1)  # the newline after each object
            if len(body) != size:
                raise PublishError("git cat-file stopped in the middle of a blob of the history")
            yield sha, kind, (body if kind == "blob" else None)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()
        writer.join(timeout=5)


def _decode(body: bytes) -> str | None:
    """A blob's text, or None for a binary blob (only path-checked). Size never skips a text blob."""
    if _binary(body):
        return None
    return body.decode("utf-8", "replace")


FETCH_REFSPECS = ("+refs/heads/*:refs/remotes/{remote}/*", "+refs/pull/*/head:refs/remotes/pr/*")


def fetch_public_refs(repo: Path | None = None, remote: str = "origin") -> bool:
    """Fetch every branch, tag and pull-request head of ``remote`` (what GitHub publishes with the repo,
    deleted branches' PR refs included), so the history scan sees them. False when the repo has no such
    remote; PublishError when the fetch fails (git's own message is not shown: it can name a URL with
    credentials)."""
    repo = Path(repo or paths.REPO_ROOT)
    if remote not in _git("remote", cwd=repo).decode().split():
        return False
    specs = [spec.format(remote=remote) for spec in FETCH_REFSPECS]
    try:
        _git("fetch", "--quiet", "--tags", "--no-write-fetch-head", remote, *specs, cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise PublishError(
            f"could not fetch every branch, tag and pull-request ref from {remote} (git exit "
            f"{exc.returncode}); they go public with the repo, so they must be scanned: check the network "
            "and credentials and run again (--no-fetch scans local refs only: not enough to go public)"
        ) from None
    return True


def run(repo: Path | None = None, *, history: bool = True, windows: set[str] | None = None) -> list[Finding]:
    """Every finding in the tracked files (and the history of every ref). ``windows`` None loads them with
    ``load_windows``, which refuses a machine that does not hold the outputs to check against."""
    repo = Path(repo or paths.REPO_ROOT)
    windows = load_windows()[0] if windows is None else windows
    found: list[Finding] = []
    allowed: set[str] = set()
    for path in tracked_files(repo):
        if forbidden_path(path):
            found.append(Finding("tree", path, 0, "file that must never be committed"))
        f = repo / path
        if f.is_file():
            text = _decode(f.read_bytes())
            if text is not None:
                found += scan_text(text, path, "tree", windows)
                allowed |= allowed_lines(text)
    if history:
        seen = {(x.path, x.kind) for x in found}
        blobs = _history_blobs(repo)
        for sha, kind, body in _cat_blobs(repo, blobs):
            if kind != "blob" or body is None:
                continue
            path = blobs[sha]
            if forbidden_path(path) and (path, "file that must never be committed") not in seen:
                found.append(Finding(f"history {sha[:10]}", path, 0, "file that must never be committed"))
            text = _decode(body)
            if text is None:
                continue
            for x in scan_text(text, blobs[sha], f"history {sha[:10]}", windows, allowed):
                if (x.path, x.kind) not in seen:
                    found.append(x)
    return found


PUBLIC_STEPS = """\
Steps to make the GitHub repo public (a person runs them):
  1. lift sync pull, then lift publish preflight   (it must pass on main, history included: it fetches every
     branch, tag and pull-request ref of origin first, and checks against every model output on this machine)
  2. Check that the runs repo is still private: lift sync check (it refuses a runs repo that is not private)
  3. Check the frozen recipe tag is pushed: git tag --list "recipe-frozen-*"; git push --tags
  4. Read by hand what GitHub also publishes and this check cannot see: every PR and issue title, body and
     comment (gh pr list --state all; gh issue list --state all) and the Actions logs (gh run list). Delete
     any that quotes a completion (gh run delete for a run log). A finding in a refs/pull/* ref stays
     public after its branch is deleted: only GitHub Support can purge it, or publish a fresh repo.
  5. gh repo edit <owner>/<repo> --visibility public --accept-visibility-change-consequences
  6. Open the repo logged out and check the README, LICENSE and notebooks/acceptance_test.ipynb render.
"""
