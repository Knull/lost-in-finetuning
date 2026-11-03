"""Prompt sources on the Hugging Face Hub, and the Hub reader they all go through. Implemented in P01.

``HFHub`` (datasets and huggingface_hub, imported on use; needs the network or a warm cache) reads
every source at the commit pinned in configs/project.yaml ``sources``. Offline, datasets 5.1 serves
the latest cached copy whatever revision is asked for, so HFHub also checks the revision directory
the Arrow files came from. Everything after the load is pure and takes a ``Hub``, so tests pass a
fake. Facts verified Nov 3 at the pinned revisions:

- OR-Bench-Hard-1K (bench-llm/or-bench, config or-bench-hard-1k, split train, CC-BY-4.0): 1,319
  rows of ``prompt`` and ``category`` (10 lowercase values, illegal 527 down to harassment 41), no
  duplicates, a stable row order, prompts ASCII on one line. The sample of 400 is stratified by
  category with largest-remainder seats (deception 22, harassment 13, harmful 32, hate 16, illegal
  160, privacy 60, self-harm 19, sexual 20, unethical 38, violence 20), each filled with the
  category's rows of smallest sha256 rank under the project split seed. No numpy RNG, so the sample
  is the same on every numpy and OS. pid ``orb-{row index in the pinned split}``. Split 100 fit /
  100 dev / 200 final, then translated (lift.data.translate).
- Aya Red-teaming (CohereLabs/aya_redteaming, Apache-2.0): the only config is "default" and Hindi
  is a SPLIT, "hindi" (``load_dataset(id, "hindi")`` raises). 915 rows. ``harm_category`` is a
  JSON-encoded list of 1-6 labels: parsed, and kept in ``extra`` with ``global_or_local`` ("global"
  or "local"). 913 unique prompts: row 32 repeats row 2's prompt (other labels) and row 465 is a
  copy of row 441, so 32 and 465 are excluded. Rows 184 and 454 end in a newline and rows 262 and
  338 hold two U+200B each (after a space); clean_text drops both. Human-written Hindi: no
  translation, no ``text_en``. Set "redteam", split "final": a second final set for the primary
  endpoint only, never used for selection.
- Belebele (facebook/belebele, CC-BY-SA-4.0): configs eng_Latn hin_Deva zho_Hans spa_Latn arb_Arab
  tha_Thai, split test, 900 items each. ``correct_answer_num`` is a STRING "1"-"4" (made an int);
  datasets parses ``ds`` as a datetime (dropped). hin_Deva is NOT row-parallel: its rows 460-899
  are a permutation of eng_Latn's. ``(link, question_number)`` is unique in every config, so each
  language is put in eng_Latn's row order by that key and ``bel-{i:03d}`` is the eng_Latn row
  index; the correct answers then match English on 900/900 in all six. 200 dev / 700 final. The
  texts are kept exactly as published (no clean_text), so letter scoring sees the standard items.
- The SFT sources (lift.data.sft) come through the same Hub: "aya_dataset" (CohereLabs/aya_dataset
  train, filtered to the 1,153 Hindi rows after loading the whole split, so ``_row`` is the
  full-split index), "indicalign" (ai4bharat/indic-align Dolly_T, 15,011 rows; only doc_id,
  eng_Latn and hin_Deva are loaded) and "dolly" (databricks-dolly-15k.jsonl, 15,011 records).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from lift import config
from lift.data.text import clean_text, dup_key
from lift.io import read_jsonl, sha256_text
from lift.schemas import PromptRecord

log = logging.getLogger(__name__)

# Hub source -> its entry under cfg["sources"].
SOURCES: dict[str, tuple[str, ...]] = {
    "orbench": ("orbench",),
    "aya_redteaming": ("aya_redteaming",),
    "belebele": ("belebele",),
    "aya_dataset": ("sft", "aya"),
    "indicalign": ("sft", "indicalign"),
    "dolly": ("sft", "dolly"),
}
_SHA = re.compile(r"[0-9a-f]{40}")


class Hub(Protocol):
    """Where the Hub sources come from: HFHub reads the pinned revisions, tests pass a fake.

    Sources: "orbench", "aya_redteaming", "belebele" (``lang`` = our code, "en" included),
    "aya_dataset" (only the rows whose ``language`` is cfg sources.sft.aya.language), "indicalign"
    (only doc_id, eng_Latn and the cfg column) and "dolly" (databricks-dolly-15k records). Every row
    is a plain dict of JSON-serialisable values plus "_row": its index in the pinned split (or
    file) before any filtering.
    """

    def rows(self, source: str, *, lang: str | None = None) -> list[dict]: ...


class HFHub:
    """The real Hub. Every load passes the commit sha pinned in cfg ``sources``; a branch name, a
    missing pin, or a cached copy of another revision (offline mode) raises."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = cfg if cfg is not None else config.load("project")

    def rows(self, source: str, *, lang: str | None = None) -> list[dict]:
        if source not in SOURCES:
            raise ValueError(f"unknown Hub source {source!r}; expected one of {sorted(SOURCES)}")
        if (source == "belebele") != (lang is not None):
            raise ValueError(f"source {source!r} {'needs' if lang is None else 'takes no'} lang")
        spec: Mapping[str, Any] = self.cfg["sources"]
        for part in SOURCES[source]:
            spec = spec[part]
        start = time.perf_counter()
        if source == "dolly":
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(spec["id"], spec["file"], repo_type="dataset",
                                   revision=_pinned(source, spec))
            rows = _plain_rows(source, list(read_jsonl(path)))
        elif source == "aya_dataset":
            ds = self._load(source, spec, spec.get("config"))
            index = [i for i, value in enumerate(ds["language"]) if value == spec["language"]]
            if not index:
                raise ValueError(f"{source}: no rows with language {spec['language']!r}")
            rows = _plain_rows(source, ds.select(index).to_list(), index)
        elif source == "indicalign":                  # 28 language columns upstream; read three
            columns = ["doc_id", "eng_Latn", spec["column"]]
            ds = self._load(source, spec, spec["subset"], columns=columns)
            rows = _plain_rows(source, ds.select_columns(columns).to_list())
        elif source == "belebele":
            codes = self.cfg["languages"]["belebele"]
            if lang not in codes:
                raise ValueError(f"unknown Belebele language {lang!r}; expected one of {sorted(codes)}")
            ds = self._load(source, spec, codes[lang])
            if "ds" in ds.column_names:          # parsed as a datetime: not JSON-serialisable
                ds = ds.remove_columns(["ds"])
            rows = _plain_rows(source, ds.to_list())
        else:                                    # orbench (config or-bench-hard-1k), aya_redteaming
            rows = _plain_rows(source, self._load(source, spec, spec.get("config")).to_list())
        log.info("hub %s%s: %d rows in %.1f s", source, f" {lang}" if lang else "", len(rows),
                 time.perf_counter() - start)
        return rows

    @staticmethod
    def _load(source: str, spec: Mapping[str, Any], name: str | None = None, **kwargs: Any) -> Any:
        from datasets import load_dataset

        revision = _pinned(source, spec)
        args = (spec["id"],) if name is None else (spec["id"], name)
        ds = load_dataset(*args, split=spec["split"], revision=revision, **kwargs)
        # Offline (HF_HUB_OFFLINE / HF_DATASETS_OFFLINE), datasets serves the latest cached copy
        # whatever revision was asked for. Its Arrow files sit in a directory named after the
        # revision they came from, so a different sha there means the pin was not honoured.
        found = {part for f in ds.cache_files for part in Path(f["filename"]).parts if _SHA.fullmatch(part)}
        if found - {revision}:
            raise RuntimeError(f"sources {source}: datasets served cached revision {sorted(found)}, not the "
                               f"pinned {revision} (offline mode ignores the pin)")
        return ds


def _pinned(source: str, spec: Mapping[str, Any]) -> str:
    revision = str(spec.get("revision") or "")
    if not _SHA.fullmatch(revision):
        raise ValueError(f"sources {source}: revision must be a full commit sha, got {revision!r}")
    return revision


def _is_plain(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float):
        return True
    if isinstance(value, list):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    return False


def _plain_rows(source: str, rows: Sequence[Mapping[str, Any]],
                index: Sequence[int] | None = None) -> list[dict]:
    """Rows as plain dicts with "_row" (``index[j]``, default j). Raises TypeError on a value that is
    not JSON-serialisable."""
    out = []
    for j, row in enumerate(rows):
        i = j if index is None else int(index[j])
        for key, value in row.items():
            if not _is_plain(value):
                raise TypeError(f"{source} row {i}: column {key!r} holds a {type(value).__name__}, "
                                "which is not JSON-serialisable")
        out.append({**row, "_row": i})
    return out


def largest_remainder(counts: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` seats in proportion to ``counts`` (Hamilton's method, exact fractions).

    Every name gets the floor of its quota ``total * n / sum(counts)``; the seats left over go to
    the largest fractional parts, ties broken by the larger count, then by name. The seats sum to
    ``total`` exactly, and no name gets more than its count while ``total <= sum(counts)``.
    Keys come back sorted.
    """
    counts = {name: int(n) for name, n in counts.items()}
    if total < 0 or any(n < 0 for n in counts.values()):
        raise ValueError("counts and total must be non-negative")
    whole = sum(counts.values())
    if not whole:
        if total:
            raise ValueError(f"cannot split {total} seats over zero counts")
        return dict.fromkeys(sorted(counts), 0)
    quota = {name: Fraction(total * n, whole) for name, n in counts.items()}
    seats = {name: q.numerator // q.denominator for name, q in quota.items()}
    left = total - sum(seats.values())
    for name in sorted(counts, key=lambda c: (seats[c] - quota[c], -counts[c], c))[:left]:
        seats[name] += 1
    return {name: seats[name] for name in sorted(counts)}


def stratified_sample(categories: Sequence[str], n: int, seed: int, tag: str = "orbench") -> list[int]:
    """Indices of ``n`` rows stratified by category, sorted.

    Each category gets its largest_remainder share of ``n`` and keeps that many of its row indices
    with the smallest ``sha256_text(f"{seed}:{tag}:{i}")``. A row's rank depends only on its own
    index, so the sample does not depend on numpy or on the order the categories arrive in.
    """
    if not 0 <= n <= len(categories):
        raise ValueError(f"cannot sample {n} of {len(categories)} rows")
    members: dict[str, list[int]] = {}
    for i, category in enumerate(categories):
        members.setdefault(category, []).append(i)
    seats = largest_remainder({c: len(idx) for c, idx in members.items()}, n)

    def rank(i: int) -> tuple[str, int]:
        return sha256_text(f"{seed}:{tag}:{i}"), i

    chosen: list[int] = []
    for category, idx in members.items():
        chosen += sorted(idx, key=rank)[: seats[category]]
    return sorted(chosen)


def orbench_records(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[PromptRecord]:
    """The stratified sample of ``n`` as English hard_benign records, sorted by pid. ``split`` is
    a placeholder ("dev") the build stamps from data/splits/hard_benign.json.

    ``rows`` must be the whole pinned split in order (``_row`` == position): the sample ranks rows
    by position and the pid carries ``_row``, so both are the HF row index.
    """
    for j, row in enumerate(rows):
        if row["_row"] != j:
            raise ValueError(f"orbench rows must be the whole pinned split in order: row {j} has "
                             f"_row {row['_row']}")
    records = []
    for i in stratified_sample([row["category"] for row in rows], n, seed, tag="orbench"):
        pid = f"orb-{i:04d}"
        text = clean_text(rows[i]["prompt"])
        if not text:
            raise ValueError(f"{pid}: empty prompt")
        records.append(PromptRecord(pid=pid, lang="en", set="hard_benign", split="dev", text=text,
                                    text_en=text, source="orbench-hard-1k", category=rows[i]["category"],
                                    extra={"hf_row": i}))
    return sorted(records, key=lambda r: r.pid)


def load_orbench_hard_sample(n: int | None = None, *, cfg: Mapping[str, Any] | None = None,
                             hub: Hub | None = None) -> list[PromptRecord]:
    """OR-Bench-Hard-1K sample of ``n`` (default cfg sources.orbench.n), seeded with cfg split_seed."""
    cfg = cfg if cfg is not None else config.load("project")
    hub = hub if hub is not None else HFHub(cfg)
    n = int(cfg["sources"]["orbench"]["n"]) if n is None else n
    rows = hub.rows("orbench")
    records = orbench_records(rows, n, int(cfg["split_seed"]))
    log.info("OR-Bench-Hard-1K: %d of %d rows, by category %s", len(records), len(rows),
             dict(sorted(Counter(r.category for r in records).items())))
    return records


def _harm_categories(pid: str, value: Any) -> list[str]:
    """``harm_category`` is a JSON-encoded list of labels (an already decoded list is accepted)."""
    try:
        labels = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        raise ValueError(f"{pid}: harm_category is not valid JSON") from None
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
        raise ValueError(f"{pid}: harm_category is not a non-empty list of labels")
    return labels


def redteam_records(rows: Sequence[Mapping[str, Any]]) -> tuple[list[PromptRecord], list[dict]]:
    """Aya Red-teaming Hindi rows as "redteam" records (split "final"), sorted by pid, and the
    exclusions: a row whose prompt has the dup_key of an earlier row's is dropped and listed as
    ``{"pid", "reason": "duplicate_of:<first pid>"}``."""
    records: list[PromptRecord] = []
    excluded: list[dict] = []
    first: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: r["_row"]):
        i = int(row["_row"])
        pid = f"art-hi-{i:04d}"
        if row.get("language", "Hindi") != "Hindi":
            raise ValueError(f"{pid}: language is not 'Hindi'")
        if row["global_or_local"] not in ("global", "local"):
            raise ValueError(f"{pid}: global_or_local is not 'global' or 'local'")
        labels = _harm_categories(pid, row["harm_category"])
        text = clean_text(row["prompt"])
        if not text:
            raise ValueError(f"{pid}: empty prompt")
        key = dup_key(text)
        if key in first:
            excluded.append({"pid": pid, "reason": f"duplicate_of:{first[key]}"})
            continue
        first[key] = pid
        records.append(PromptRecord(pid=pid, lang="hi", set="redteam", split="final", text=text,
                                    text_en=None, source="aya-redteaming", category="; ".join(labels),
                                    translation=None,
                                    extra={"harm_category": labels, "global_or_local": row["global_or_local"],
                                           "hf_row": i}))
    return sorted(records, key=lambda r: r.pid), excluded


def load_aya_redteaming_hindi(*, cfg: Mapping[str, Any] | None = None,
                              hub: Hub | None = None) -> list[PromptRecord]:
    """The Hindi red-teaming records; duplicates are logged and left out (see redteam_records)."""
    cfg = cfg if cfg is not None else config.load("project")
    hub = hub if hub is not None else HFHub(cfg)
    rows = hub.rows("aya_redteaming")
    records, excluded = redteam_records(rows)
    for ex in excluded:
        log.info("aya_redteaming: %s excluded (%s)", ex["pid"], ex["reason"])
    log.info("Aya Red-teaming Hindi: %d rows, %d records, %d excluded", len(rows), len(records),
             len(excluded))
    return records


_BELEBELE_FIELDS = ("link", "flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3",
                    "mc_answer4", "dialect")


def _belebele_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["link"]), int(row["question_number"])


def _belebele_positions(rows: Sequence[Mapping[str, Any]], code: str) -> dict[tuple[str, int], int]:
    positions: dict[tuple[str, int], int] = {}
    repeats = []
    for j, row in enumerate(rows):
        if positions.setdefault(_belebele_key(row), j) != j:
            repeats.append(j)
    if repeats:
        raise ValueError(f"belebele {code}: (link, question_number) repeats in {len(repeats)} rows, "
                         f"e.g. rows {repeats[:5]}")
    return positions


def _belebele_answer(row: Mapping[str, Any], code: str, j: int) -> int:
    """``correct_answer_num`` is the string "1"-"4" upstream."""
    try:
        answer = int(str(row["correct_answer_num"]).strip())
    except ValueError:
        answer = 0
    if not 1 <= answer <= 4:
        raise ValueError(f"belebele {code} row {j}: correct_answer_num is not 1-4")
    return answer


def belebele_rows(rows_by_config: Mapping[str, Sequence[Mapping[str, Any]]], lang_code: str,
                  en_code: str = "eng_Latn", *, cfg: Mapping[str, Any] | None = None) -> list[dict]:
    """The items of config ``lang_code`` in ``en_code``'s row order, matched on (link,
    question_number): item ``bel-{i:03d}`` is English row i in every language.

    Each dict has pid, lang (our code, from cfg languages.belebele), split None, link,
    question_number, flores_passage, question, mc_answer1..4, correct_answer_num (int 1-4) and
    dialect; "ds" and "_row" are dropped. Raises ValueError if a key repeats, the key sets differ, or
    a correct answer differs from English's after alignment.
    """
    cfg = cfg if cfg is not None else config.load("project")
    ours = {code: lang for lang, code in cfg["languages"]["belebele"].items()}
    if lang_code not in ours:
        raise ValueError(f"unknown Belebele config {lang_code!r}; expected one of {sorted(ours)}")
    en_rows, rows = rows_by_config[en_code], rows_by_config[lang_code]
    en_pos = _belebele_positions(en_rows, en_code)
    pos = _belebele_positions(rows, lang_code)
    if en_pos.keys() != pos.keys():
        raise ValueError(f"belebele {lang_code}: {len(en_pos.keys() - pos.keys())} {en_code} items "
                         f"missing, {len(pos.keys() - en_pos.keys())} items not in {en_code}")
    out, wrong, moved = [], [], 0
    for i, en_row in enumerate(en_rows):
        if en_row.get("_row", i) != i:
            raise ValueError(f"belebele {en_code} rows must be the whole pinned split in order: row {i} "
                             f"has _row {en_row['_row']}")
        j = pos[_belebele_key(en_row)]
        row = rows[j]
        moved += j != i
        answer = _belebele_answer(row, lang_code, j)
        if answer != _belebele_answer(en_row, en_code, i):
            wrong.append(f"bel-{i:03d}")
        item = {"pid": f"bel-{i:03d}", "lang": ours[lang_code], "split": None,
                "question_number": int(row["question_number"]), "correct_answer_num": answer}
        item.update((field, row[field]) for field in _BELEBELE_FIELDS)
        out.append(item)
    if wrong:
        raise ValueError(f"belebele {lang_code}: correct_answer_num differs from {en_code} on "
                         f"{len(wrong)} items, e.g. {wrong[:5]}")
    log.info("belebele %s: %d items, %d rows realigned to %s order", lang_code, len(out), moved, en_code)
    return out


def load_belebele(lang: str, *, cfg: Mapping[str, Any] | None = None, hub: Hub | None = None) -> list[dict]:
    """The Belebele items in ``lang`` (our code), in eng_Latn order (see belebele_rows)."""
    cfg = cfg if cfg is not None else config.load("project")
    codes = cfg["languages"]["belebele"]
    if lang not in codes:
        raise ValueError(f"unknown Belebele language {lang!r}; expected one of {sorted(codes)}")
    hub = hub if hub is not None else HFHub(cfg)
    en_rows = hub.rows("belebele", lang="en")
    rows = en_rows if lang == "en" else hub.rows("belebele", lang=lang)
    return belebele_rows({codes["en"]: en_rows, codes[lang]: rows}, codes[lang], codes["en"], cfg=cfg)
