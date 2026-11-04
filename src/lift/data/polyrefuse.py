"""PolyRefuse loaders and alignment checks (mainlp/Multilingual-Refusal, Apache-2.0). Implemented in P01.

The clone lives at ``data/raw/Multilingual-Refusal`` at the commit pinned in configs/project.yaml
(``sources.polyrefuse.commit``), checked on every build. Field mapping, verified Nov 3 at 985169d:

- Every file is one JSON array written with ``json.dump(rows, f, indent=4)``: ``\\u`` escapes, LF, no
  trailing newline. Windows' system git config sets core.autocrlf=true, so every git call here
  passes ``-c core.autocrlf=false`` and the files stay byte-identical to upstream.
- Test files ``PolyRefuse/{harmful,harmless}_test_translated_{xx}.json`` (572 / 500 rows):
  ``instruction`` is the ENGLISH source, ``instruction_translated`` the target text, ``category`` a
  string (harmful, identical in every language) or null (harmless). The English sources are the
  ``instruction`` of ``PolyRefuse/harmful_test_translated_en.json`` (keys instruction and category
  only; byte-identical to dataset/splits/harmful_test.json) and of
  ``PolyRefuse/harmless_test_translated_en.json`` (instruction_translated == instruction). NOT
  dataset/splits/harmless_test_500_sampled.json: a decoy holding other prompts.
- Train/val files ``PolyRefuse/{kind}_{split}_translated_{xx}.json`` (harmful 260 / 39, harmless
  200 / 200): ``instruction`` is the TRANSLATED text, ``category`` null (harmful val copies the
  English category). No English field and no en file: the English sources are
  dataset/splits/harmful_{train,val}.json and dataset/splits/harmless_{train,val}_200_sampled.json,
  aligned by row index (log-length correlation 0.75-0.97 in every language).
- Languages: ar de es fr it ja ko nl pl ru th yo zh. No Hindi anywhere and no zh harmless test. zh
  harmless train and val are another, unstored 200-row draw that lines up with no English file
  (correlation ~0.0): they are listed in ``sources.polyrefuse.misaligned`` and never loaded. Hindi,
  the zh harmless test and zh harmless train are translated from the English sources
  (lift.data.translate). ``PolyRefuse/harmful_test_back_translated/`` (literal UTF-8) is ignored.
- Kept as upstream wrote them (lift.data.build handles them): the harmful test repeats 10 English
  prompts under different categories, (4, 174) (14, 163) (24, 243) (39, 402) (51, 193) (53, 257)
  (54, 187) (60, 145) (61, 470) (63, 491); harmful_train[226] is harmful_test[8]; inner newlines
  and double spaces. ``clean_text`` drops the translations' zero-width spaces (230 U+200B).

Records: pid ``prh-{split}-{i:04d}`` (harmful) or ``prb-{split}-{i:04d}`` (harmless), i = the row
index, the same in every language. Row counts and the misaligned list come from the config.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from lift import config
from lift.data.text import clean_text
from lift.lang import LangIdentifier, lid_info
from lift.paths import raw_dir
from lift.schemas import LANGS, PromptRecord, TranslationInfo

log = logging.getLogger(__name__)

Kind = Literal["harmful", "harmless"]
PSplit = Literal["test", "train", "val"]

REPO_DIRNAME = "Multilingual-Refusal"
KINDS: tuple[str, ...] = ("harmful", "harmless")
SPLITS: tuple[str, ...] = ("test", "train", "val")
# The (kind, split) families the build reads. Harmless val is not used.
BUILD_FILES: tuple[tuple[str, str], ...] = (("harmful", "test"), ("harmful", "train"), ("harmful", "val"),
                                            ("harmless", "test"), ("harmless", "train"))
SET_NAMES = {"harmful": "harmful", "harmless": "easy_benign"}
PID_PREFIX = {"harmful": "prh", "harmless": "prb"}
_TRAIN_VAL_ENGLISH = {
    ("harmful", "train"): "dataset/splits/harmful_train.json",
    ("harmful", "val"): "dataset/splits/harmful_val.json",
    ("harmless", "train"): "dataset/splits/harmless_train_200_sampled.json",
    ("harmless", "val"): "dataset/splits/harmless_val_200_sampled.json",
}

# Train/val alignment: aligned files correlate 0.75-0.97, the misaligned zh files ~0.0.
MIN_CORRELATION = 0.5
RATIO_RANGE = (0.1, 4.0)
SPOT_ROWS = 5

_GIT = ("git", "-c", "core.autocrlf=false")
_SHA = re.compile(r"[0-9a-f]{40}")


class AlignmentError(ValueError):
    """PolyRefuse rows that do not line up with their English source, or a file that is not what was
    verified. ``problems`` holds {"file", "check", "row", "detail"} dicts, never row text."""

    def __init__(self, message: str, problems: Sequence[Mapping[str, Any]] = ()) -> None:
        super().__init__(message)
        self.problems: list[dict[str, Any]] = [dict(p) for p in problems]


class CommitMismatch(RuntimeError):
    """The clone is not at the pinned commit, or no commit is pinned."""


def default_root() -> Path:
    return raw_dir() / REPO_DIRNAME


# --- clone -------------------------------------------------------------------------------------

def ensure_clone(root: Path | None = None, *, cfg: Mapping[str, Any] | None = None) -> Path:
    """The clone at the pinned commit, made first if ``root`` is missing (or an empty folder).

    A new clone is ``git init`` + ``fetch --depth 1 origin <sha>`` + ``checkout FETCH_HEAD``, with
    core.autocrlf=false on every call and in the clone's own config. An existing clone is only
    checked, never moved: CommitMismatch if it is at another commit. A failed clone leaves a partial
    folder; delete it and rerun.
    """
    cfg = _cfg(cfg)
    pin = _pinned(cfg)
    root = Path(root) if root is not None else default_root()
    if not root.exists() or (root.is_dir() and not any(root.iterdir())):
        repo = cfg["sources"]["polyrefuse"]["repo"]
        log.info("cloning %s at %s into %s", repo, pin[:12], root)
        root.mkdir(parents=True, exist_ok=True)
        steps = (("init", "-q"), ("config", "core.autocrlf", "false"), ("remote", "add", "origin", repo),
                 ("fetch", "-q", "--depth", "1", "origin", pin), ("checkout", "-q", "FETCH_HEAD"))
        for step in steps:
            try:
                _git(root, *step)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"git {step[0]} failed in {root} ({_stderr(e)}); "
                                   f"delete {root} and rerun") from e
    check_commit(root, cfg=cfg)
    return root


def head_commit(root: Path) -> str:
    """The commit checked out in ``root``. CommitMismatch if ``root`` is not a git clone with one."""
    root = Path(root)
    if not (root / ".git").exists():     # else git would answer for an enclosing repo
        raise CommitMismatch(f"{root} is not a git clone (no .git); delete it and rerun")
    try:
        return _git(root, "rev-parse", "--verify", "HEAD").strip()
    except subprocess.CalledProcessError as e:
        raise CommitMismatch(f"{root} has no checked-out commit ({_stderr(e)}); delete it and rerun") from e


def check_commit(root: Path, *, cfg: Mapping[str, Any] | None = None) -> str:
    """The clone's HEAD sha. CommitMismatch unless it is ``sources.polyrefuse.commit``."""
    pin = _pinned(_cfg(cfg))
    head = head_commit(root)
    if head != pin:
        raise CommitMismatch(f"{root} is at commit {head}, but sources.polyrefuse.commit pins {pin}; "
                             "delete the clone and rerun (a new pin needs a docs/decisions.md entry)")
    return head


def _git(root: Path, *args: str) -> str:
    """git in ``root`` with core.autocrlf=false: a list argv, no shell, never a credential prompt."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        done = subprocess.run([*_GIT, "-C", str(root), *args], check=True, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", env=env)
    except FileNotFoundError as e:
        raise RuntimeError("git is not installed or not on PATH") from e
    return done.stdout


def _stderr(e: subprocess.CalledProcessError) -> str:
    lines = (e.stderr or "").strip().splitlines()
    return f"exit {e.returncode}: {lines[-1][:200]}" if lines else f"exit {e.returncode}"


def _pinned(cfg: Mapping[str, Any]) -> str:
    pin = cfg["sources"]["polyrefuse"].get("commit")
    if not isinstance(pin, str) or not _SHA.fullmatch(pin):
        raise CommitMismatch(f"sources.polyrefuse.commit must be a full 40-character commit sha, got {pin!r}")
    return pin


# --- loaders -----------------------------------------------------------------------------------

def has_file(root: Path, kind: Kind, split: PSplit, lang: str, *,
             cfg: Mapping[str, Any] | None = None) -> bool:
    """True if PolyRefuse provides (kind, split, lang): the file exists and is not listed as
    misaligned. For en, the English source exists. False for hi, the zh harmless test and zh
    harmless train/val: those are translated from English."""
    _check_family(kind, split)
    cfg = _cfg(cfg)
    if lang != "en" and _misaligned(cfg, kind, split, lang):
        return False
    return (Path(root) / _rel(cfg, kind, split, lang)).is_file()


def english_sources(root: Path, kind: Kind, split: PSplit, *,
                    cfg: Mapping[str, Any] | None = None) -> list[str]:
    """English source texts (clean_text applied), aligned by index with every translated file of the
    same kind and split. AlignmentError if the row count is not ``sources.polyrefuse.counts``."""
    _check_family(kind, split)
    return _english(Path(root), kind, split, _cfg(cfg))[0]


def load_polyrefuse(root: Path, kind: Kind, split: PSplit, lang: str, *,
                    cfg: Mapping[str, Any] | None = None) -> list[PromptRecord]:
    """Records in file order.

    pid as in the module docstring; ``set`` "harmful" or "easy_benign"; ``split`` "dev" for test
    rows (a placeholder lift.data.build overwrites from the split file), "train"/"val" otherwise;
    ``text`` = clean_text(target); ``text_en`` = the English source (always, also for en);
    ``category`` from the row; ``translation`` = TranslationInfo(engine="polyrefuse") except for en;
    ``extra`` = {"src_file": path relative to the clone, "src_row": i}. en reads the English source
    for every split. FileNotFoundError when PolyRefuse has no file (hi; the zh harmless test);
    AlignmentError for a misaligned file (zh harmless train/val), a wrong row count, or a test file
    whose English column differs from the English source.
    """
    _check_family(kind, split)
    if lang not in LANGS:
        raise ValueError(f"unsupported language {lang!r}; expected one of {LANGS}")
    cfg = _cfg(cfg)
    root = Path(root)
    rel = _rel(cfg, kind, split, lang)
    if lang != "en":
        if _misaligned(cfg, kind, split, lang):
            raise AlignmentError(f"{rel} does not line up with its English source (listed in "
                                 f"sources.polyrefuse.misaligned): translate {kind} {split} into {lang} "
                                 "from English instead",
                                 [_problem(rel, "misaligned", None, "sources.polyrefuse.misaligned")])
        if not (root / rel).is_file():
            raise FileNotFoundError(f"PolyRefuse has no {kind} {split} file for {lang} ({rel}); "
                                    "it is translated from the English source")
    english, en_categories = _english(root, kind, split, cfg)
    if lang == "en":
        texts, categories, translation = english, en_categories, None
    else:
        texts, categories = _translated(root, rel, kind, split, english, cfg)
        translation = TranslationInfo(engine="polyrefuse", source_lang="en")
    prefix, set_name = PID_PREFIX[kind], SET_NAMES[kind]
    rec_split = "dev" if split == "test" else split
    log.debug("loaded %d rows from %s", len(texts), rel)
    return [PromptRecord(pid=f"{prefix}-{split}-{i:04d}", lang=lang, set=set_name, split=rec_split,
                         text=text, text_en=english[i], source="polyrefuse", category=categories[i],
                         translation=translation, extra={"src_file": rel, "src_row": i})
            for i, text in enumerate(texts)]


def _translated(root: Path, rel: str, kind: str, split: str, english: Sequence[str],
                cfg: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    """Target texts and categories of one translated file, paired by index with ``english``."""
    rows = _read(root, rel)
    problems: list[dict[str, Any]] = []
    texts: list[str] = []
    if _count_ok(rows, rel, _expected(cfg, kind, split), problems):
        texts = _column(rows, _target_field(split), rel, problems)
        if split == "test":
            own = _column(rows, "instruction", rel, problems)
            problems += _english_mismatches(rel, own, english)
    if problems:
        raise AlignmentError(_describe(problems), problems)
    return texts, [row.get("category") for row in rows]


def _english(root: Path, kind: str, split: str, cfg: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    """English texts and categories of one family."""
    rel = _rel(cfg, kind, split, "en")
    if not (root / rel).is_file():
        raise FileNotFoundError(f"PolyRefuse English source {rel} not found under {root}")
    rows = _read(root, rel)
    problems: list[dict[str, Any]] = []
    texts: list[str] = []
    if _count_ok(rows, rel, _expected(cfg, kind, split), problems):
        texts = _column(rows, "instruction", rel, problems)
    if problems:
        raise AlignmentError(_describe(problems), problems)
    return texts, [row.get("category") for row in rows]


# --- alignment checks --------------------------------------------------------------------------

def check_alignment(root: Path, langs: Iterable[str], *, lid: LangIdentifier | None = None,
                    cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check every PolyRefuse file the build loads for ``langs`` against its English source.

    Families: harmful test/train/val and harmless test/train; languages: ``langs`` minus en and hi,
    skipping missing and misaligned files (recorded under "skipped"). Checks:
    - every English source and translated file has the row count in ``sources.polyrefuse.counts``;
    - every row's ``category`` equals the English file's;
    - test files: the English ``instruction`` equals the English source, row for row;
    - train/val files: Pearson correlation of log character lengths with the English rows >= 0.5
      (files with >= 3 rows), and a spot check of min(5, n) evenly spaced rows: every char-length
      ratio within [0.1, 4.0] and, with ``lid``, at least ceil(0.8 * spot rows) pass
      ``lid.is_lang`` (one short-string misread is normal: real zh harmful val row 0 read und_Nshu
      before lift.lang spaced Chinese ideographs, and cmn_Hani 1.0 since);
    - the harmless test English source (what zh and hi are translated from) equals the English
      column of every harmless test file of ``langs`` and of the configured languages, row for row.
    Collects every problem and raises ONE AlignmentError listing them (file, check, row; no text).
    Returns the report: English sources, files checked (rows, correlation, spot rows), skipped files.
    """
    cfg = _cfg(cfg)
    root = Path(root)
    langs = [lang for lang in dict.fromkeys(langs) if lang not in ("en", "hi")]
    problems: list[dict[str, Any]] = []
    report: dict[str, Any] = {"langs": langs, "english": {}, "files": {}, "skipped": [],
                              "harmless_test_source": None}
    english: dict[tuple[str, str], tuple[list[str], list[Any]] | None] = {}
    for kind, split in BUILD_FILES:
        rel = _rel(cfg, kind, split, "en")
        found: tuple[list[str], list[Any]] | None = None
        try:
            found = _english(root, kind, split, cfg)
        except AlignmentError as e:
            problems += e.problems
        except FileNotFoundError:
            problems.append(_problem(rel, "missing", None, "English source not found"))
        else:
            report["english"][f"{kind}_{split}"] = {"file": rel, "rows": len(found[0])}
        english[(kind, split)] = found

    for kind, split in BUILD_FILES:
        for lang in langs:
            rel = _rel(cfg, kind, split, lang)
            if _misaligned(cfg, kind, split, lang):
                report["skipped"].append({"file": rel, "reason": "misaligned", "correlation":
                                          _info_correlation(root, rel, split, english[(kind, split)])})
            elif not (root / rel).is_file():
                report["skipped"].append({"file": rel, "reason": "missing"})
            else:
                report["files"][rel] = _check_file(root, rel, kind, split, lang, english[(kind, split)],
                                                   lid, cfg, problems)

    report["harmless_test_source"] = _check_harmless_source(root, langs, english[("harmless", "test")],
                                                            cfg, problems)
    if problems:
        raise AlignmentError(_describe(problems), problems)
    log.info("PolyRefuse alignment ok: %d files checked, %d skipped", len(report["files"]),
             len(report["skipped"]))
    return report


def _check_file(root: Path, rel: str, kind: str, split: str, lang: str,
                english: tuple[list[str], list[Any]] | None, lid: LangIdentifier | None,
                cfg: Mapping[str, Any], problems: list[dict[str, Any]]) -> dict[str, Any]:
    entry: dict[str, Any] = {"kind": kind, "split": split, "lang": lang, "rows": None}
    try:
        rows = _read(root, rel)
    except AlignmentError as e:
        problems += e.problems
        return entry
    entry["rows"] = len(rows)
    if not _count_ok(rows, rel, _expected(cfg, kind, split), problems) or english is None:
        return entry
    en_texts, en_categories = english
    texts = _column(rows, _target_field(split), rel, problems)
    problems += [_problem(rel, "category", i, "differs from the English file")
                 for i, row in enumerate(rows) if row.get("category") != en_categories[i]]
    if split == "test":
        own = _column(rows, "instruction", rel, problems)
        problems += _english_mismatches(rel, own, en_texts)
    elif all(texts):
        entry.update(_length_and_lid_checks(rel, lang, texts, en_texts, lid, problems))
    return entry


def _length_and_lid_checks(rel: str, lang: str, texts: Sequence[str], english: Sequence[str],
                           lid: LangIdentifier | None, problems: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"correlation": None, "spot": [], "lid_pass": None}
    if len(texts) >= 3:
        r = _log_length_correlation(english, texts)
        out["correlation"] = None if r is None else round(r, 4)
        if r is None or r < MIN_CORRELATION:
            shown = "undefined" if r is None else f"{r:.3f}"
            problems.append(_problem(rel, "length_correlation", None, f"r = {shown} < {MIN_CORRELATION}"))
    lo, hi = RATIO_RANGE
    passed = 0
    for i in _spot_rows(len(texts)):
        ratio = len(texts[i]) / len(english[i])
        spot: dict[str, Any] = {"row": i, "ratio": round(ratio, 4)}
        if not lo <= ratio <= hi:
            problems.append(_problem(rel, "length_ratio", i, f"ratio {ratio:.2f} outside [{lo}, {hi}]"))
        if lid is not None:
            info = lid_info(lid, texts[i])
            spot.update(label=info.label, prob=info.prob, ok=bool(lid.is_lang(texts[i], lang)))
            passed += spot["ok"]
        out["spot"].append(spot)
    if lid is not None:
        k = len(out["spot"])
        need = -(-4 * k // 5)                   # ceil(0.8 * k), in integers
        out["lid_pass"] = passed
        if passed < need:
            tally = f"{passed}/{k} spot rows pass, need {need}"
            problems += [_problem(rel, "lid_spot", s["row"], f"top {s['label']} {s['prob']:.2f}; {tally}")
                         for s in out["spot"] if not s["ok"]]
    return out


def _check_harmless_source(root: Path, langs: Sequence[str], source: tuple[list[str], list[Any]] | None,
                           cfg: Mapping[str, Any], problems: list[dict[str, Any]]) -> dict[str, Any]:
    """The harmless test English source against every harmless test file of ``langs`` and of the
    configured languages (whatever ``langs`` is, so a hi-only build still checks it)."""
    rel = _rel(cfg, "harmless", "test", "en")
    candidates = dict.fromkeys([*langs, *cfg["languages"]["all"]])
    compared = [lang for lang in candidates
                if lang not in ("en", "hi") and has_file(root, "harmless", "test", lang, cfg=cfg)]
    out: dict[str, Any] = {"file": rel, "compared": compared, "mismatched_rows": None}
    if source is None:                          # the source's own problems are already listed
        return out
    if not compared:
        problems.append(_problem(rel, "harmless_test_source", None, "no harmless test file to compare with"))
        return out
    differs: dict[int, list[str]] = {}
    for lang in compared:
        other = _rel(cfg, "harmless", "test", lang)
        try:
            column = _column(_read(root, other), "instruction", other, [])
        except AlignmentError:
            problems.append(_problem(rel, "harmless_test_source", None, f"{lang} file unreadable"))
            continue
        if len(column) != len(source[0]):
            problems.append(_problem(rel, "harmless_test_source", None,
                                     f"{lang} has {len(column)} rows, the source {len(source[0])}"))
            continue
        for i, (a, b) in enumerate(zip(source[0], column, strict=True)):
            if a != b:
                differs.setdefault(i, []).append(lang)
    problems += [_problem(rel, "harmless_test_source", i, "differs from " + ", ".join(found))
                 for i, found in sorted(differs.items())]
    out["mismatched_rows"] = len(differs)
    return out


def _info_correlation(root: Path, rel: str, split: str,
                      english: tuple[list[str], list[Any]] | None) -> float | None:
    """Log-length correlation of a skipped (misaligned) file, for the report only."""
    if not (root / rel).is_file():
        return None
    try:
        texts = _column(_read(root, rel), _target_field(split), rel, [])
    except AlignmentError:
        return None
    if english is None or len(texts) != len(english[0]) or len(texts) < 3 or not all(texts):
        return None
    r = _log_length_correlation(english[0], texts)
    return None if r is None else round(r, 4)


def _log_length_correlation(english: Sequence[str], texts: Sequence[str]) -> float | None:
    """Pearson correlation of log character lengths; None when either side has a single length."""
    x = [math.log(len(t)) for t in english]
    y = [math.log(len(t)) for t in texts]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def _spot_rows(n: int, k: int = SPOT_ROWS) -> list[int]:
    """min(k, n) evenly spaced row indices, the first and last row included."""
    k = min(k, n)
    if k < 2:
        return list(range(k))
    return [j * (n - 1) // (k - 1) for j in range(k)]


# --- files -------------------------------------------------------------------------------------

def _cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return cfg if cfg is not None else config.load("project")


def _check_family(kind: str, split: str) -> None:
    if kind not in KINDS or split not in SPLITS:
        raise ValueError(f"unknown PolyRefuse family {kind!r}/{split!r}; kinds {KINDS}, splits {SPLITS}")


def _rel(cfg: Mapping[str, Any], kind: str, split: str, lang: str) -> str:
    """A family's file for ``lang``, relative to the clone (forward slashes); en is the English source."""
    if lang == "en" and split != "test":
        return _TRAIN_VAL_ENGLISH[(kind, split)]
    return f"{cfg['sources']['polyrefuse']['dir']}/{kind}_{split}_translated_{lang}.json"


def _target_field(split: str) -> str:
    return "instruction_translated" if split == "test" else "instruction"


def _expected(cfg: Mapping[str, Any], kind: str, split: str) -> int:
    return int(cfg["sources"]["polyrefuse"]["counts"][f"{kind}_{split}"])


def _misaligned(cfg: Mapping[str, Any], kind: str, split: str, lang: str) -> bool:
    return f"{kind}_{split}_{lang}" in (cfg["sources"]["polyrefuse"].get("misaligned") or ())


def _read(root: Path, rel: str) -> list[dict[str, Any]]:
    """One file: a JSON array of objects (AlignmentError otherwise)."""
    try:
        rows = json.loads((root / rel).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        found = [_problem(rel, "format", None, f"invalid JSON at line {e.lineno} column {e.colno}")]
        raise AlignmentError(_describe(found), found) from e
    except UnicodeDecodeError as e:
        found = [_problem(rel, "format", None, f"invalid UTF-8 at byte {e.start}")]
        raise AlignmentError(_describe(found), found) from e
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        found = [_problem(rel, "format", None, "not a JSON array of objects")]
        raise AlignmentError(_describe(found), found)
    return rows


def _column(rows: Sequence[Mapping[str, Any]], field: str, rel: str,
            problems: list[dict[str, Any]]) -> list[str]:
    """clean_text of ``field`` in every row; a missing, non-string or empty value is a problem."""
    out = []
    for i, row in enumerate(rows):
        value = row.get(field)
        text = clean_text(value) if isinstance(value, str) else ""
        if not text:
            problems.append(_problem(rel, "row_format", i, f"{field} missing or empty"))
        out.append(text)
    return out


def _count_ok(rows: Sequence[Any], rel: str, expected: int, problems: list[dict[str, Any]]) -> bool:
    if len(rows) == expected:
        return True
    problems.append(_problem(rel, "row_count", None, f"{len(rows)} rows, expected {expected}"))
    return False


def _english_mismatches(rel: str, own: Sequence[str], english: Sequence[str]) -> list[dict[str, Any]]:
    return [_problem(rel, "english_instruction", i, "differs from the English source")
            for i, (a, b) in enumerate(zip(own, english, strict=True)) if a and a != b]


def _problem(file: str, check: str, row: int | None = None, detail: str = "") -> dict[str, Any]:
    return {"file": file, "check": check, "row": row, "detail": detail}


def _describe(problems: Sequence[Mapping[str, Any]], limit: int = 30) -> str:
    """One line per (file, check) with its rows and first detail. Never row text."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for p in problems:
        groups.setdefault((p["file"], p["check"]), []).append(p)
    lines = []
    for (file, check), group in groups.items():
        line = f"{file}: {check}"
        rows = [p["row"] for p in group if p["row"] is not None]
        if rows:
            shown = ", ".join(map(str, rows[:10])) + (", ..." if len(rows) > 10 else "")
            line += f" at {len(rows)} row(s) [{shown}]"
        if group[0]["detail"]:
            line += f" ({group[0]['detail']})"
        lines.append(line)
    if len(lines) > limit:
        lines[limit:] = [f"... and {len(lines) - limit} more (see .problems)"]
    return f"PolyRefuse alignment: {len(problems)} problem(s)\n  " + "\n  ".join(lines)
