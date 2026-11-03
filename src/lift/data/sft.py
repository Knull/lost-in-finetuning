"""Adapter training candidates and the shared utility held-out set. Implemented in P01.

Two Hindi instruction candidates, chosen on dev utility win rate only, never on drift. Facts
verified on Nov 3 at the revisions pinned in configs/project.yaml (``sources.sft``):
- ``aya``: CohereLabs/aya_dataset rows with language "Hindi" (language_code "hin"): 1,153, all in
  train (none in test), single-turn ``inputs`` -> ``targets``. About 120 have English inputs with
  Hindi targets (translation tasks), which the instruction LID check drops; 164 repeat an earlier
  input.
- ``indicalign``: ai4bharat/indic-align ``Dolly_T`` (databricks-dolly-15k, written by people and
  translated with IndicTrans2), 15,011 single-turn rows; the Hindi pair is ``hin_Deva[0]`` (never
  ``hin_Latn``: some of its turns lack a part, row 0 is swapped). IndicAlign DROPPED Dolly's
  context passage from every closed_qa, information_extraction and summarization row, which then
  asks about a text the model never sees. A row is kept only if its English instruction (NFC,
  whitespace collapsed; 19 rows match only after that) equals a databricks-dolly-15k instruction
  and no record with that instruction has a context: 10,526 kept, 4,482 had a context, 3 match none.

Pipeline (build.py drives it; ``build_candidate`` does one candidate end to end), with the Nov 3
counts as aya / indicalign:
1. ``load_pools``: ``clean_pool`` applies clean_text to both fields and drops, counting the reason,
   empty fields, an instruction or response GlotLID does not read as Hindi, and a response whose
   character 10-gram repetition exceeds unclear.max_repetition_ratio (the judge's "repetitive"
   rule): 1,153 -> 1,000 / 10,526 -> 10,322.
2. ``build_utility_heldout``: ``heldout.per_source`` (150) rows per source, ranked by
   sha256(seed, tag, row id) among rows whose instruction occurs once in the pool and that
   CuratorKIT's schema gate keeps (10-2,048 tokens), with no two near-duplicates among the 300 (the
   indicalign draw also avoids the aya draw). pid ``utl-{aya|ind}-{idx:05d}``; the build splits
   them 100 dev / 200 final.
3. ``exclude_heldout``: removes the held-out rows, and every row that near-duplicates a held-out row
   of EITHER source, from BOTH training pools: 1,000 -> 790 / 10,322 -> 10,144. Near-duplicate
   (``heldout.near_dup``): exact character 3-gram Jaccard (the quantity MinHash estimates) of the
   instructions >= 0.7, or of the responses >= 0.85 when both normalised responses have >= 20
   characters; empty texts never match.
4. ``build_candidates``: CuratorKIT's no-API path on each training pool (``run_curatorkit``): schema
   gate, exact then MinHash dedup (0.85 on character 3-grams; the first copy in row-id order stays;
   the ``raw`` stress pool skips dedup), cleaning with strip_html and collapse_whitespace off:
   790 -> 679 (raw 788) / 10,144 -> 9,958. Then the pre-registered equal-size rule: N =
   min(train.yaml rows, aya dedup, indicalign dedup) = 679, limited by aya and far below 3,000;
   both candidates AND the aya raw sample take N rows by sha256 rank. The build logs a warning and
   report.json records the decision.

Files under the data dir (training rows use the messages format TRL's SFTTrainer reads):
    interim/sft/{cand}/{variant}/input.jsonl             CuratorKIT input {id, instruction, output}
    interim/sft/{cand}/{variant}/curatorkit/             CuratorKIT's own output folder (CRLF on Windows)
    interim/sft/{cand}/{variant}/curatorkit_summary.json stage counts, rejected and deduplicated ids
    sft/{cand}/{variant}/sft_messages.jsonl              the N rows {"id", "messages"}, by id (LF)
    sft/{cand}/{variant}/selection.json                  ids, counts, seed, sha256 of the export
    sft/{cand}/{variant}/curatorkit/                     LF copies of CuratorKIT's manifest.json,
                                                         rejected.jsonl, lexsi_provenance.json + summary
    sft/report.json                                      every stage count and the N decision
variant is ``dedup`` (both candidates) or ``raw`` (aya, stress recipe). The utility records go to
prompts/utility_hi.jsonl through build.py. CuratorKIT's own files carry run timestamps; everything
this module writes is byte-identical on a rebuild.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from lift import config
from lift.data.text import char_ngrams, clean_text, dup_key
from lift.io import sha256_file, sha256_text, write_jsonl
from lift.judge.labels import repetition_ratio
from lift.paths import data_dir, interim_dir, sft_dir
from lift.schemas import PromptRecord

log = logging.getLogger(__name__)

TAGS = {"aya": "aya", "indicalign": "ind"}
SOURCE_NAMES = {"aya": "aya-dataset", "indicalign": "indicalign-dolly_t"}
LANG = "hi"
DROP_REASONS = ("empty_instruction", "empty_response", "instruction_not_hi", "response_not_hi",
                "repetitive_response")
# CuratorKIT's schema gate (its defaults, passed explicitly so the held-out filter matches it).
SCHEMA_MIN_TOKENS = 10
SCHEMA_MAX_TOKENS = 2048
# Off: strip_html deletes anything between "<" and ">" (maths, comparisons), collapse_whitespace
# flattens code indentation. NFC, encoding fixes, control-character removal and strip stay on.
CLEAN_TRANSFORMS = {"strip_html": False, "collapse_whitespace": False}
CURATORKIT_FILES = ("manifest.json", "rejected.jsonl", "lexsi_provenance.json")


@dataclass(frozen=True)
class SftRow:
    """One single-turn Hindi (instruction, response) pair from a candidate source."""

    source: str                  # "aya" | "indicalign"
    idx: int                     # position among the source's Hindi rows, HF order (Dolly_T: the HF row)
    hf_row: int                  # row index in the pinned HF split
    instruction: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict, hash=False)   # annotation_type | doc_id

    def row_id(self) -> str:
        """``aya-00042``; the held-out pid is ``utl-`` + this."""
        return f"{TAGS[self.source]}-{self.idx:05d}"


class Curate(Protocol):
    """What ``build_candidates`` calls: ``run_curatorkit``, or a fake in tests."""

    def __call__(self, rows: list[dict], out_dir: Path, *, dedup: bool) -> list[dict]: ...


# ---------------------------------------------------------------- loading


def aya_rows(rows: Iterable[Mapping[str, Any]], *, language: str | None = None) -> list[SftRow]:
    """Hub "aya_dataset" rows (the Hindi rows, each with its ``_row``) -> SftRow in HF order.
    ``idx`` counts Hindi rows only. With ``language``, a row in another language raises."""
    rows = sorted(rows, key=lambda r: int(r["_row"]))
    if language is not None:
        other = [int(r["_row"]) for r in rows if r.get("language") != language]
        if other:
            raise ValueError(f"{len(other)} aya_dataset rows are not {language!r}, e.g. rows {other[:5]}")
    return [SftRow("aya", k, int(r["_row"]), r["inputs"] or "", r["targets"] or "",
                   {"annotation_type": r.get("annotation_type")}) for k, r in enumerate(rows)]


def _dolly_key(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def indicalign_rows(rows: Iterable[Mapping[str, Any]],
                    dolly_rows: Iterable[Mapping[str, Any]]) -> tuple[list[SftRow], Counter[str]]:
    """Hub "indicalign" rows (Dolly_T: doc_id, eng_Latn, hin_Deva, _row) -> SftRow, applying the
    Dolly context rule. Counts: kept, had_context (some matching Dolly record has a context),
    no_match (the English instruction matches no Dolly record)."""
    has_context: dict[str, bool] = {}
    for d in dolly_rows:
        key = _dolly_key(d["instruction"])
        has_context[key] = has_context.get(key, False) or bool((d.get("context") or "").strip())
    out: list[SftRow] = []
    counts: Counter[str] = Counter(dict.fromkeys(("kept", "had_context", "no_match"), 0))
    for r in sorted(rows, key=lambda r: int(r["_row"])):
        key = _dolly_key(r["eng_Latn"][0][0])
        if key not in has_context:
            counts["no_match"] += 1
            continue
        if has_context[key]:
            counts["had_context"] += 1
            continue
        turn = r["hin_Deva"][0]
        if len(turn) != 2:
            raise ValueError(f"Dolly_T row {r['_row']}: hin_Deva turn has {len(turn)} parts, expected 2")
        out.append(SftRow("indicalign", int(r["_row"]), int(r["_row"]), turn[0] or "", turn[1] or "",
                          {"doc_id": r["doc_id"]}))
        counts["kept"] += 1
    return out, counts


def clean_pool(rows: Iterable[SftRow], lid: Any, *,
               cfg: Mapping[str, Any] | None = None) -> tuple[list[SftRow], Counter[str]]:
    """clean_text both fields, then drop a row for the first reason in DROP_REASONS that applies.
    ``lid`` is any lift.lang.LangIdentifier. Counts every reason (zeros included) and "kept"."""
    cfg = cfg if cfg is not None else config.load("project")
    max_rep = float(cfg["unclear"]["max_repetition_ratio"])
    out: list[SftRow] = []
    counts: Counter[str] = Counter(dict.fromkeys((*DROP_REASONS, "kept"), 0))
    for row in rows:
        row = replace(row, instruction=clean_text(row.instruction), response=clean_text(row.response))
        if not row.instruction:
            reason = "empty_instruction"
        elif not row.response:
            reason = "empty_response"
        elif not lid.is_lang(row.instruction, LANG):
            reason = "instruction_not_hi"
        elif not lid.is_lang(row.response, LANG):
            reason = "response_not_hi"
        elif repetition_ratio(row.response) > max_rep:
            reason = "repetitive_response"
        else:
            reason = "kept"
            out.append(row)
        counts[reason] += 1
    return out, counts


def load_pools(*, cfg: Mapping[str, Any] | None = None, hub: Any = None,
               lid: Any = None) -> dict[str, tuple[list[SftRow], Counter[str]]]:
    """The two cleaned pools with their counts: loaded, every drop reason (indicalign also
    had_context and no_match), kept. hub None -> sources.HFHub(cfg); lid None -> LangID(cfg=cfg)."""
    cfg = cfg if cfg is not None else config.load("project")
    if hub is None:
        from lift.data.sources import HFHub

        hub = HFHub(cfg)
    if lid is None:
        from lift.lang import LangID

        lid = LangID(cfg=cfg)
    aya = aya_rows(hub.rows("aya_dataset"), language=cfg["sources"]["sft"]["aya"]["language"])
    ind_rows = hub.rows("indicalign")
    ind, ctx = indicalign_rows(ind_rows, hub.rows("dolly"))
    pools: dict[str, tuple[list[SftRow], Counter[str]]] = {}
    for name, rows, before in (("aya", aya, Counter()),
                               ("indicalign", ind, Counter(had_context=ctx["had_context"],
                                                           no_match=ctx["no_match"]))):
        pool, counts = clean_pool(rows, lid, cfg=cfg)
        loaded = len(rows) + sum(before.values())
        pools[name] = (pool, Counter({"loaded": loaded, **before, **counts}))
        dropped = {k: v for k, v in pools[name][1].items() if k not in ("loaded", "kept") and v}
        log.info("sft pool %s: loaded %d, kept %d, dropped %s", name, loaded, len(pool), dropped)
    return pools


def _rows(pool: Any) -> list[SftRow]:
    """A pool as load_pools / exclude_heldout return it, (rows, ...), or a bare row list."""
    return list(pool[0]) if isinstance(pool, tuple) else list(pool)


# ---------------------------------------------------------------- near-duplicates


class _NearDup(NamedTuple):
    ngram: int
    instruction: Fraction        # thresholds as exact fractions (0.7 -> 7/10)
    response: Fraction
    min_chars: int


class _Grams(NamedTuple):
    ins: frozenset[str]
    res: frozenset[str]          # empty when the response is too short to compare


def _near_cfg(cfg: Mapping[str, Any] | None) -> _NearDup:
    cfg = cfg if cfg is not None else config.load("project")
    nd = cfg["sources"]["sft"]["heldout"]["near_dup"]
    thresholds = Fraction(str(nd["instruction"])), Fraction(str(nd["response"]))
    if not all(0 < t <= 1 for t in thresholds):
        raise ValueError(f"near_dup thresholds must lie in (0, 1], got {nd}")
    return _NearDup(int(nd["ngram"]), *thresholds, int(nd["response_min_chars"]))


def _grams(instruction: str, response: str, nd: _NearDup) -> _Grams:
    long_enough = len(dup_key(response)) >= nd.min_chars
    return _Grams(char_ngrams(instruction, nd.ngram),
                  char_ngrams(response, nd.ngram) if long_enough else frozenset())


def _at_least(a: frozenset[str], b: frozenset[str], t: Fraction) -> bool:
    """Exact Jaccard(a, b) >= t, in integers; empty sets never match. |A & B| <= min(|A|, |B|), so
    the sizes alone rule out most pairs before an intersection is built."""
    if not a or not b:
        return False
    la, lb = len(a), len(b)
    if min(la, lb) * t.denominator < t.numerator * max(la, lb):
        return False
    inter = len(a & b)
    return inter * t.denominator >= t.numerator * (la + lb - inter)


def _is_near(a: _Grams, b: _Grams, nd: _NearDup) -> bool:
    """The near-duplicate rule on one pair (the reference ``_join`` agrees with)."""
    return _at_least(a.ins, b.ins, nd.instruction) or _at_least(a.res, b.res, nd.response)


def _join(left: Sequence[frozenset[str]], right: Sequence[frozenset[str]],
          t: Fraction) -> list[tuple[int, int]]:
    """Every (i, j) with Jaccard(left[i], right[j]) >= t; empty sets never match.

    Exact, via prefix filtering: rank grams by how many sets hold them (rarest first, then by the
    gram). Two sets that similar share at least ceil(t * |S|) grams, so the first gram they share
    lies in each one's first |S| - ceil(t * |S|) + 1 grams. An inverted index from the right sets'
    prefix grams proposes the candidates; ``_at_least`` confirms them.
    """
    df = Counter(chain.from_iterable((*left, *right)))
    rank = {g: k for k, g in enumerate(sorted(df, key=lambda g: (df[g], g)))}

    def prefix(s: frozenset[str]) -> list[str]:
        return sorted(s, key=rank.__getitem__)[: len(s) - math.ceil(t * len(s)) + 1]

    index: dict[str, list[int]] = defaultdict(list)
    for j, s in enumerate(right):
        if s:
            for g in prefix(s):
                index[g].append(j)
    pairs: list[tuple[int, int]] = []
    for i, s in enumerate(left):
        if s:
            cands = {j for g in prefix(s) for j in index.get(g, ())}
            pairs.extend((i, j) for j in sorted(cands) if _at_least(s, right[j], t))
    return pairs


def _heldout_items(heldout: Iterable[PromptRecord | SftRow]) -> list[tuple[str, str, str]]:
    """(pid, instruction, response) for utility PromptRecords or held-out SftRows."""
    items = []
    for h in heldout:
        if isinstance(h, SftRow):
            items.append(("utl-" + h.row_id(), h.instruction, h.response))
        else:
            items.append((h.pid, h.text, h.extra["reference"]))
    return items


def near_duplicates(rows: Sequence[SftRow], heldout: Iterable[PromptRecord | SftRow], *,
                    cfg: Mapping[str, Any] | None = None) -> dict[str, str]:
    """row_id -> the (smallest) held-out pid it near-duplicates. ``heldout`` takes the utility
    PromptRecords (text = instruction, extra.reference = response) or SftRows. A held-out row
    passed in ``rows`` matches itself: exclude_heldout removes those first."""
    nd = _near_cfg(cfg)
    items = _heldout_items(heldout)
    held = [_grams(ins, res, nd) for _, ins, res in items]
    mine = [_grams(r.instruction, r.response, nd) for r in rows]
    hits: dict[int, set[int]] = defaultdict(set)
    for i, j in _join([g.ins for g in mine], [g.ins for g in held], nd.instruction):
        hits[i].add(j)
    for i, j in _join([g.res for g in mine], [g.res for g in held], nd.response):
        hits[i].add(j)
    return {rows[i].row_id(): min(items[j][0] for j in js) for i, js in sorted(hits.items())}


# ---------------------------------------------------------------- held-out set

# CuratorKIT 0.1.22's whitespace token count (curatorkit.utils.tokens.count_tokens_whitespace):
# words, plus one token per character of a script written without spaces. Copied so the held-out
# filter needs no curatorkit import; a test checks it against the original.
_UNSPACED = re.compile("[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf"
                       "\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0003134f]")


def schema_tokens(text: str) -> int:
    n = len(_UNSPACED.findall(text))
    return n + len(_UNSPACED.sub(" ", text).split()) if n else len(text.split())


def _schema_ok(row: SftRow) -> bool:
    """Would CuratorKIT's schema gate keep this row (null bytes, 10-2,048 tokens)?"""
    text = f"{row.instruction} {row.response}".strip()
    return "\x00" not in text and SCHEMA_MIN_TOKENS <= schema_tokens(text) <= SCHEMA_MAX_TOKENS


def select_heldout(pool: Sequence[SftRow], n: int, *, seed: int, tag: str,
                   cfg: Mapping[str, Any] | None = None, avoid: Sequence[SftRow] = ()) -> list[SftRow]:
    """``n`` held-out rows, in idx order. Candidates: rows whose instruction (dup_key) occurs once in
    the pool and that CuratorKIT's schema gate keeps. Ranked by sha256(f"{seed}:{tag}:{row_id}") and
    taken greedily, skipping a near-duplicate of a row already taken or of one in ``avoid`` (the
    other source's held-out rows). ValueError if fewer than ``n`` qualify."""
    nd = _near_cfg(cfg)
    seen = Counter(dup_key(r.instruction) for r in pool)
    cands = [r for r in pool if seen[dup_key(r.instruction)] == 1 and _schema_ok(r)]
    cands.sort(key=lambda r: sha256_text(f"{seed}:{tag}:{r.row_id()}"))
    taken: list[SftRow] = []
    grams = [_grams(r.instruction, r.response, nd) for r in avoid]
    for row in cands:
        if len(taken) == n:
            break
        g = _grams(row.instruction, row.response, nd)
        if any(_is_near(g, h, nd) for h in grams):
            continue
        taken.append(row)
        grams.append(g)
    if len(taken) < n:
        raise ValueError(f"{tag}: only {len(taken)} held-out rows qualify, need {n} "
                         f"(pool {len(pool)}, unique and in the token window {len(cands)})")
    return sorted(taken, key=lambda r: r.idx)


def heldout_record(row: SftRow) -> PromptRecord:
    """A held-out row as a utility PromptRecord (split "dev" is a placeholder the build stamps)."""
    extra: dict[str, Any] = {"reference": clean_text(row.response), "hf_row": row.hf_row, "src_idx": row.idx}
    if row.source == "indicalign":
        extra |= {"doc_id": row.meta["doc_id"], "upstream_mt": "IndicTrans2"}
    return PromptRecord(pid="utl-" + row.row_id(), lang=LANG, set="utility", split="dev",
                        text=clean_text(row.instruction), text_en=None, source=SOURCE_NAMES[row.source],
                        translation=None, extra=extra)


def build_utility_heldout(pools: Mapping[str, Any] | None = None, *, cfg: Mapping[str, Any] | None = None,
                          hub: Any = None, lid: Any = None) -> list[PromptRecord]:
    """The shared utility set: heldout.per_source rows from each pool (tag f"utility:{name}"), as
    PromptRecords sorted by pid. Aya is drawn first; the indicalign draw skips near-duplicates of
    it. pools None -> load_pools(cfg=cfg, hub=hub, lid=lid)."""
    cfg = cfg if cfg is not None else config.load("project")
    per = int(cfg["sources"]["sft"]["heldout"]["per_source"])
    if per * len(TAGS) != sum(cfg["sizes"]["utility"].values()):
        raise ValueError(f"sizes.utility sums to {sum(cfg['sizes']['utility'].values())}, but "
                         f"heldout.per_source x {len(TAGS)} sources = {per * len(TAGS)}")
    if pools is None:
        pools = load_pools(cfg=cfg, hub=hub, lid=lid)
    taken: list[SftRow] = []
    for name in TAGS:
        taken += select_heldout(_rows(pools[name]), per, seed=int(cfg["split_seed"]),
                                tag=f"utility:{name}", cfg=cfg, avoid=taken)
    return sorted((heldout_record(r) for r in taken), key=lambda rec: rec.pid)


def exclude_heldout(pools: Mapping[str, Any], heldout: Iterable[PromptRecord | SftRow], *,
                    cfg: Mapping[str, Any] | None = None) -> dict[str, tuple[list[SftRow], list[dict]]]:
    """Per pool: (training rows in idx order, exclusions [{"id", "reason"}] sorted by id). Removes
    the pool's own held-out rows ("heldout") and every near-duplicate of any held-out row, either
    source ("near_dup:<pid>"). A held-out pid missing from its own pool raises ValueError."""
    heldout = list(heldout)
    held_pids = {pid for pid, _, _ in _heldout_items(heldout)}
    out: dict[str, tuple[list[SftRow], list[dict]]] = {}
    for name, pool in pools.items():
        rows = _rows(pool)
        own = {pid for pid in held_pids if pid.startswith(f"utl-{TAGS[name]}-")}
        missing = sorted(own - {"utl-" + r.row_id() for r in rows})
        if missing:
            raise ValueError(f"{len(missing)} held-out pids are not in the {name} pool, e.g. {missing[:5]}")
        rest = [r for r in rows if "utl-" + r.row_id() not in own]
        excluded = [{"id": r.row_id(), "reason": "heldout"} for r in rows if "utl-" + r.row_id() in own]
        near = near_duplicates(rest, heldout, cfg=cfg)
        excluded += [{"id": rid, "reason": f"near_dup:{pid}"} for rid, pid in near.items()]
        train = sorted((r for r in rest if r.row_id() not in near), key=lambda r: r.idx)
        out[name] = (train, sorted(excluded, key=lambda e: e["id"]))
        log.info("sft %s: %d held out, %d near-duplicates excluded, %d left to train on",
                 name, len(own), len(near), len(train))
    return out


# ---------------------------------------------------------------- CuratorKIT and the candidates


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def run_curatorkit(rows: list[dict], out_dir: Path, *, dedup: bool) -> list[dict]:
    """Run CuratorKIT's no-API path on ``rows`` ({"id", "instruction", "output"}, in idx order) and
    return the rows it passed as {"id", "messages"}, in input order.

    Writes <out_dir>/input.jsonl, CuratorKIT's folder <out_dir>/curatorkit (removed first: a reused
    folder keeps stale exports) and <out_dir>/curatorkit_summary.json (counts and ids, no text).
    CuratorKIT copies the dataset string into its manifest, provenance and rejected.jsonl, so it
    runs from data_dir() on relative paths, and out_dir must lie inside data_dir(). Its own files
    are CRLF on Windows; the export rows carry no id, so ids come from ``result.passed``.
    """
    from curatorkit import Curator, CuratorConfig
    from curatorkit import __version__ as ck_version
    from curatorkit.exporters.messages import build_messages

    base = data_dir()
    out_dir = Path(out_dir).resolve()
    try:
        rel = out_dir.relative_to(base)
    except ValueError:
        raise ValueError(f"CuratorKIT runs inside the data dir; {out_dir} is not under {base}") from None
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in the CuratorKIT input")
    write_jsonl(out_dir / "input.jsonl", ({"id": i, "instruction": r["instruction"], "output": r["output"]}
                                          for i, r in zip(ids, rows, strict=True)))
    ck_dir = out_dir / "curatorkit"
    if ck_dir.exists():
        shutil.rmtree(ck_dir)
    dataset = (rel / "input.jsonl").as_posix()
    ck_cfg = CuratorConfig(dataset=dataset, dedup="minhash" if dedup else "none", clean=True,
                           clean_transforms=dict(CLEAN_TRANSFORMS), export_formats=["messages"],
                           output_dir=(rel / "curatorkit").as_posix(), min_tokens=SCHEMA_MIN_TOKENS,
                           max_tokens=SCHEMA_MAX_TOKENS)
    with contextlib.chdir(base):
        result = Curator(ck_cfg).run()
    # An unknown export format only warns and exports nothing (the "sft_messages" trap).
    if "MessagesExporter" not in result.stage_counts:
        raise RuntimeError("CuratorKIT exported nothing: MessagesExporter did not run")
    with open(ck_dir / "sft_messages.jsonl", encoding="utf-8") as f:
        exported = sum(1 for line in f if line.strip())
    if exported != len(result.passed):
        raise RuntimeError(f"CuratorKIT exported {exported} rows but passed {len(result.passed)}")
    out = [{"id": s.id, "messages": build_messages(s)} for s in result.passed]
    passed = [r["id"] for r in out]
    order = {i: k for k, i in enumerate(ids)}
    if any(i not in order for i in passed) or [order[i] for i in passed] != sorted(order[i] for i in passed):
        raise RuntimeError("CuratorKIT changed the row ids or their order")
    rejected = {s.id: s.rejection_reason for s in result.rejected}
    kept = set(passed)
    manifest = json.loads((ck_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "curatorkit": ck_version, "dataset": dataset, "dedup": ck_cfg.dedup,
        "clean_transforms": dict(CLEAN_TRANSFORMS), "input": len(ids), "passed": len(out),
        "stage_counts": manifest["stage_counts"], "rejected_breakdown": manifest["rejected_breakdown"],
        "dedup_stats": manifest["dedup_stats"], "minhash_threshold": manifest["minhash_threshold"],
        "rejected": dict(sorted(rejected.items())),
        "deduplicated": [i for i in ids if i not in kept and i not in rejected],
    }
    _write_json(out_dir / "curatorkit_summary.json", summary)
    log.info("CuratorKIT (%s) on %s: %d in, %d passed, %d rejected, %d deduplicated", ck_cfg.dedup,
             dataset, len(ids), len(out), len(rejected), len(summary["deduplicated"]))
    return out


def _variants(train_cfg: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(candidate, variant): every candidate deduplicated, plus the stress candidate raw."""
    return [(name, "dedup") for name in train_cfg["candidates"]] + [(train_cfg["stress"]["candidate"], "raw")]


def _curate(name: str, variant: str, rows: Sequence[SftRow],
            curate: Curate) -> tuple[list[dict], dict | None]:
    """Run ``curate`` on one training pool in an emptied work dir (nothing stale can pass for this
    run's provenance); check the rows it returns; read its summary if it wrote one."""
    work = interim_dir() / "sft" / name / variant
    if work.exists():
        shutil.rmtree(work)
    inputs = [{"id": r.row_id(), "instruction": r.instruction, "output": r.response}
              for r in sorted(rows, key=lambda r: r.idx)]
    out = curate(inputs, work, dedup=variant == "dedup")
    known = {r["id"] for r in inputs}
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids) or not set(ids) <= known:
        raise RuntimeError(f"curate returned unknown or repeated ids for {name}/{variant}")
    for r in out:
        roles = [m["role"] for m in r["messages"]]
        if roles != ["user", "assistant"] or not all(str(m["content"]).strip() for m in r["messages"]):
            raise RuntimeError(f"{name}/{variant} row {r['id']}: not one non-empty user/assistant exchange")
    summary_path = work / "curatorkit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return out, summary


def _write_variant(name: str, variant: str, curated: Sequence[dict], n: int, *, rows_target: int,
                   seed: int, train_pool: int, summary: dict | None) -> dict:
    """Sample ``n`` curated rows by sha256 rank, write the export, the CuratorKIT copies and
    selection.json into sft_dir()/name/variant (rebuilt from scratch). Returns selection.json."""
    tag = f"sft:{name}:{variant}"
    ranked = sorted(curated, key=lambda r: sha256_text(f"{seed}:{tag}:{r['id']}"))
    chosen = sorted(ranked[:n], key=lambda r: r["id"])
    out = sft_dir() / name / variant
    if out.exists():
        shutil.rmtree(out)
    write_jsonl(out / "sft_messages.jsonl", ({"id": r["id"], "messages": r["messages"]} for r in chosen))
    work = interim_dir() / "sft" / name / variant
    for src in [work / "curatorkit" / f for f in CURATORKIT_FILES] + [work / "curatorkit_summary.json"]:
        if src.exists():      # LF copies, so the folder hashes the same on every OS
            dst = out / "curatorkit" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    ck = None if summary is None else {
        k: summary.get(k) for k in ("stage_counts", "rejected_breakdown", "dedup_stats", "minhash_threshold")}
    selection = {
        "candidate": name, "variant": variant, "dedup": variant == "dedup", "rows_target": rows_target,
        "rows": n, "seed": seed, "tag": tag, "ids": [r["id"] for r in chosen],
        "counts": {"train_pool": train_pool, "curated": len(curated), "sampled": n},
        "curatorkit": ck,
        "sha256": sha256_file(out / "sft_messages.jsonl"),
    }
    _write_json(out / "selection.json", selection)
    log.info("sft %s/%s: train pool %d, curated %d, sampled %d", name, variant, train_pool, len(curated), n)
    return selection


def _source_report(pool: Sequence[SftRow], exclusions: Sequence[Mapping[str, str]],
                   counts: Mapping[str, int] | None) -> dict:
    by_tag = {tag: name for name, tag in TAGS.items()}
    near = Counter(by_tag[e["reason"].split(":", 1)[1].split("-")[1]]
                   for e in exclusions if e["reason"].startswith("near_dup:"))
    counts = dict(counts or {})
    return {
        "loaded": counts.pop("loaded", None), "pool": counts.pop("kept", None),
        "dropped": dict(sorted(counts.items())),
        "held_out": sum(e["reason"] == "heldout" for e in exclusions),
        "near_dup_excluded": sum(near.values()),
        "near_dup_of": {name: near[name] for name in TAGS},
        "train_pool": len(pool),
    }


def build_candidates(train_pools: Mapping[str, Any], *,
                     pool_counts: Mapping[str, Mapping[str, int]] | None = None,
                     cfg: Mapping[str, Any] | None = None, train_cfg: Mapping[str, Any] | None = None,
                     curate: Curate | None = None) -> dict:
    """Curate and sample every variant (aya/dedup, indicalign/dedup, aya/raw) under the equal-size
    rule, write sft_dir()/report.json and return it.

    ``train_pools`` is exclude_heldout's result; ``pool_counts`` (load_pools' counters, by source)
    fills the loaded and dropped counts of the report. curate None -> run_curatorkit. The report:
    {"decision": {"rows_target", "rows": N, "limited_by": candidate | None}, "seed",
     "sources": {name: {loaded, dropped, pool, held_out, near_dup_excluded, near_dup_of, train_pool}},
     "variants": {"aya/dedup": selection.json without ids, ...}}.
    """
    cfg = cfg if cfg is not None else config.load("project")
    train_cfg = train_cfg if train_cfg is not None else config.load("train")
    curate = curate if curate is not None else run_curatorkit
    target, seed = int(train_cfg["rows"]), int(cfg["split_seed"])
    plan = _variants(train_cfg)
    curated = {(name, variant): _curate(name, variant, _rows(train_pools[name]), curate)
               for name, variant in plan}
    sizes = {name: len(curated[(name, "dedup")][0]) for name in train_cfg["candidates"]}
    n = min(target, *sizes.values())
    limited_by = None if n == target else min(sizes, key=lambda k: sizes[k])
    if n < 1:
        raise ValueError(f"no SFT rows left to train on: curated sizes {sizes}")
    if n < target:
        log.warning("SFT candidates use %d rows, below the target %d (limited by %s, curated sizes %s): "
                    "both candidates and the %s raw sample take %d (equal-size rule)",
                    n, target, limited_by, sizes, train_cfg["stress"]["candidate"], n)
    for (name, variant), (rows, _) in curated.items():
        if len(rows) < n:
            raise ValueError(f"{name}/{variant} has {len(rows)} curated rows, fewer than N = {n}")
    variants = {}
    for name, variant in plan:
        rows, summary = curated[(name, variant)]
        selection = _write_variant(name, variant, rows, n, rows_target=target, seed=seed,
                                   train_pool=len(_rows(train_pools[name])), summary=summary)
        variants[f"{name}/{variant}"] = {k: v for k, v in selection.items() if k != "ids"}
    report = {
        "decision": {"rows_target": target, "rows": n, "limited_by": limited_by},
        "seed": seed,
        "sources": {name: _source_report(_rows(pool), pool[1] if isinstance(pool, tuple) else [],
                                         (pool_counts or {}).get(name))
                    for name, pool in train_pools.items()},
        "variants": variants,
    }
    _write_json(sft_dir() / "report.json", report)
    return report


def build_candidate(name: str, *, dedup: bool = True, rows: int = 3000, cfg: Mapping[str, Any] | None = None,
                    train_cfg: Mapping[str, Any] | None = None, curate: Curate | None = None,
                    pools: Mapping[str, Any] | None = None) -> Path:
    """One candidate end to end (held-out set and exclusion included) with min(rows, curated) rows,
    no equal-size rule and no report.json; build.py uses build_candidates. pools None ->
    load_pools(cfg=cfg). Returns the directory holding sft_messages.jsonl and the CuratorKIT copies."""
    cfg = cfg if cfg is not None else config.load("project")
    train_cfg = train_cfg if train_cfg is not None else config.load("train")
    if name not in train_cfg["candidates"]:
        raise ValueError(f"unknown candidate {name!r}; train.yaml lists {train_cfg['candidates']}")
    pools = pools if pools is not None else load_pools(cfg=cfg)
    train = exclude_heldout(pools, build_utility_heldout(pools, cfg=cfg), cfg=cfg)
    variant = "dedup" if dedup else "raw"
    pool = train[name][0]
    curated, summary = _curate(name, variant, pool, curate if curate is not None else run_curatorkit)
    _write_variant(name, variant, curated, min(rows, len(curated)), rows_target=rows,
                   seed=int(cfg["split_seed"]), train_pool=len(pool), summary=summary)
    return sft_dir() / name / variant
