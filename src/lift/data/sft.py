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
  and no record with that instruction has a context (10,526 rows; 4,482 had a context, 3 match
  none), and its Hindi pair is in its English pair's order. hin_Deva[0] is answer first in 5 rows,
  635, 2496, 5630, 7510 and 13137 (checked Nov 5, all 15,011 rows being 1 turn of 2 parts): their
  "?" endings mirror the English pair's and, in all but 13137, so do their lengths (both ratios
  over 3, in opposite directions). ``_hindi_reversed`` flags exactly these; 7510 and 13137 had a
  context, the other 3 are dropped as hindi_reversed: 10,523 kept.

Pipeline (build.py drives it; ``build_candidate`` does one candidate end to end), with the counts
of the pinned sources under the Nov 5 rules as aya / indicalign (sft/report.json holds each
build's counts; the Nov 4 build, before two R1 fixes took 3 more rows out of each pool, counted
10,526 / 10,322 / 10,144 / 9,958 for indicalign and 790 / 679 for aya):
1. ``load_pools``: ``clean_pool`` applies clean_text to both fields and drops, counting the reason,
   empty fields, an instruction or response GlotLID does not read as Hindi, and a response whose
   character 10-gram repetition exceeds unclear.max_repetition_ratio (the judge's "repetitive"
   rule): 1,153 -> 1,000 / 10,523 -> 10,319.
2. ``build_utility_heldout``: ``heldout.per_source`` (150) rows per source, ranked by
   sha256(seed, tag, row id) among rows whose instruction occurs once in the pool and that
   CuratorKIT's schema gate keeps (10-2,048 tokens), with no two near-duplicates among the 300 (the
   indicalign draw also avoids the aya draw). pid ``utl-{aya|ind}-{idx:05d}``; the build splits
   them 100 dev / 200 final into data/splits/utility.json. The draw is a function of the pools, so
   a change to the pools' revisions or cleaning (unclear.max_repetition_ratio,
   languages.accept_labels.hi, langid) or to ``heldout`` draws other rows. Once the split file
   exists, every build of utility or sft (build.py freezes or checks it first) and
   ``build_candidate`` (``check_frozen_heldout``) stop with ValueError before writing a training
   row unless the draw gives exactly its pids.
3. ``exclude_heldout``: removes the held-out rows, and every row that near-duplicates a held-out row
   of EITHER source, from BOTH training pools: 1,000 -> 787 / 10,319 -> 10,141. Near-duplicate
   (``heldout.near_dup``): exact character 3-gram Jaccard (the quantity MinHash estimates) of the
   instructions >= 0.7, or of the responses >= 0.85, at any length (decisions item 24; a text
   under 3 characters is its own single gram); empty texts never match. Until Nov 5 answers under
   20 characters were skipped, a minimum item 24 never had: without it 3 more aya rows match (two
   copy the 11- and 13-character answers of final utility items), so 63 rows go, not 60.
4. ``build_candidates``: CuratorKIT's no-API path on each training pool (``run_curatorkit``): schema
   gate, exact then MinHash dedup (0.85 on character 3-grams; the first copy in row-id order stays;
   the ``raw`` stress pool skips dedup), cleaning with strip_html and collapse_whitespace off:
   787 -> 677 (raw 785) / 10,141 -> 9,955 (raw 9,982). Then the row counts: since Nov 4
   (decisions item 47, ``equal_size: false``) each variant takes min(train.yaml rows, its own
   curated rows) by sha256 rank, so aya trains on 677 and indicalign on 3,000, and the stress
   recipe's raw sample is indicalign's. The Nov 3 rule (``equal_size: true``: every variant takes
   min(rows, aya dedup, indicalign dedup) = 677) stays available. report.json records the decision.

Files under the data dir (training rows use the messages format TRL's SFTTrainer reads):
    interim/sft/{cand}/{variant}/input.jsonl             CuratorKIT input {id, instruction, output}
    interim/sft/{cand}/{variant}/curatorkit/             CuratorKIT's own output folder (CRLF on Windows)
    interim/sft/{cand}/{variant}/curatorkit_summary.json stage counts, rejected and deduplicated ids
    sft/{cand}/{variant}/sft_messages.jsonl              the N rows {"id", "messages"}, by id (LF)
    sft/{cand}/{variant}/selection.json                  ids, counts, seed, sha256 of the export, rules
    sft/{cand}/{variant}/curatorkit/                     LF copies of CuratorKIT's manifest.json,
                                                         rejected.jsonl, lexsi_provenance.json + summary
    sft/report.json                                      every stage count, the N decision, the rules
variant is ``dedup`` (both candidates, the standard recipe: train.yaml ``standard.dedup: true``)
or ``raw`` (the stress candidate, ``stress.candidate``: ``stress.dedup: false``). The plan is
fixed, so ``build_candidates`` refuses any other dedup setting (ValueError naming the keys), and a
train.yaml without ``equal_size`` (KeyError), before anything is written. It leaves sft/ holding
exactly the variants report.json lists: after its checks pass it removes every sft/{cand}/{variant} and
interim/sft/{cand}/{variant} folder the plan does not make (such as the Nov 3 aya/raw once item 47
moved the raw sample to indicalign), and a {cand} folder left empty. lift sync deletes nothing,
so a removed variant must also be deleted from the runs repo by hand. The utility records go to
prompts/utility_hi.jsonl through build.py. CuratorKIT's own files carry run timestamps (manifest.json:
run_timestamp, wall_clock_seconds and its provenance's created_at; lexsi_provenance.json:
created_at; rejected.jsonl: each row's provenance_chain timestamps), so interim/sft/ and the
sft/{cand}/{variant}/curatorkit/ copies of those three files change on every run; everything else
this module writes is byte-identical on a rebuild (checked Nov 4 on the pinned data).

The rules a build applied travel with the exports: report.json and every selection.json hold its
``rules_stamp`` ("rules": RULES_VERSION, the project and train settings that shape the rows, the
sha256 of data/splits/utility.json). ``export_problems`` lists every reason the exports on disk must
not be trained on: a report.json that is missing, has no stamp (the Nov 4 build, RULES_VERSION 1,
whose aya/dedup holds two copies of final utility answers) or another stamp than the current code
and configs give; a variant folder it does not list (the Nov 3 aya/raw); an export that does not
match its selection.json. A trainer (P03's) must call it before it reads an export and refuse while
it is not empty; ``lift data stats`` prints each problem as stale. Each names its remedy, most often
``lift data build --only sft`` then ``lift sync push``.
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

from lift import config, splits
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
# The rules this code applies to decide which rows an export holds (``rules_stamp``): bump it with any
# change here that changes them. 1: the Nov 4 build (the held-out near-duplicate rule skipped answers
# under 20 characters; Dolly_T rows whose Hindi pair is answer first were kept). 2: R1-20 (answers of
# any length are compared) and R1-21 (those Dolly_T rows are dropped).
RULES_VERSION = 2
_REBUILD = "rebuild with `lift data build --only sft`, then `lift sync push`"


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


def _hindi_reversed(eng: Sequence[str | None], hin: Sequence[str | None]) -> bool:
    """Is a Dolly_T row's Hindi pair in the reverse order of its English (instruction, response)
    pair? Either sign is enough: the "?" endings are mirrored (the English instruction ends in "?"
    and its response does not, while Hindi part 0 does not and part 1 does), or log(len0 / len1) of
    the English pair and of the Hindi pair both exceed log 3, with opposite signs. Texts are
    compared after clean_text; a pair without exactly 2 parts is never called reversed."""
    if len(eng) != 2 or len(hin) != 2:
        return False
    e0, e1, h0, h1 = (clean_text(t or "") for t in (*eng, *hin))
    if e0.endswith("?") and not e1.endswith("?") and not h0.endswith("?") and h1.endswith("?"):
        return True
    if not (e0 and e1 and h0 and h1):
        return False
    en, hi = math.log(len(e0) / len(e1)), math.log(len(h0) / len(h1))
    return min(abs(en), abs(hi)) > math.log(3) and (en > 0) != (hi > 0)


def indicalign_rows(rows: Iterable[Mapping[str, Any]],
                    dolly_rows: Iterable[Mapping[str, Any]]) -> tuple[list[SftRow], Counter[str]]:
    """Hub "indicalign" rows (Dolly_T: doc_id, eng_Latn, hin_Deva, _row) -> SftRow, applying the
    Dolly context rule. Counts: kept, had_context (some matching Dolly record has a context),
    no_match (the English instruction matches no Dolly record), hindi_reversed (hin_Deva[0] is in
    the reverse order of eng_Latn[0], ``_hindi_reversed``: dropped rather than swapped, as both of
    its signs are heuristics)."""
    has_context: dict[str, bool] = {}
    for d in dolly_rows:
        key = _dolly_key(d["instruction"])
        has_context[key] = has_context.get(key, False) or bool((d.get("context") or "").strip())
    out: list[SftRow] = []
    counts: Counter[str] = Counter(dict.fromkeys(("kept", "had_context", "no_match", "hindi_reversed"), 0))
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
        if _hindi_reversed(r["eng_Latn"][0], turn):
            counts["hindi_reversed"] += 1
            continue
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
    had_context, no_match and hindi_reversed), kept. hub None -> sources.HFHub(cfg); lid None ->
    LangID(cfg=cfg)."""
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
                                                           no_match=ctx["no_match"],
                                                           hindi_reversed=ctx["hindi_reversed"]))):
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


class _Grams(NamedTuple):
    ins: frozenset[str]
    res: frozenset[str]          # empty only for an empty response, which never matches


def _near_cfg(cfg: Mapping[str, Any] | None) -> _NearDup:
    cfg = cfg if cfg is not None else config.load("project")
    nd = cfg["sources"]["sft"]["heldout"]["near_dup"]
    thresholds = Fraction(str(nd["instruction"])), Fraction(str(nd["response"]))
    if not all(0 < t <= 1 for t in thresholds):
        raise ValueError(f"near_dup thresholds must lie in (0, 1], got {nd}")
    return _NearDup(int(nd["ngram"]), *thresholds)


def _grams(instruction: str, response: str, nd: _NearDup) -> _Grams:
    return _Grams(char_ngrams(instruction, nd.ngram), char_ngrams(response, nd.ngram))


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
    """The near-duplicate rule on one pair (the reference ``jaccard_pairs`` agrees with)."""
    return _at_least(a.ins, b.ins, nd.instruction) or _at_least(a.res, b.res, nd.response)


def jaccard_pairs(left: Sequence[frozenset[str]], right: Sequence[frozenset[str]],
                  t: Fraction) -> list[tuple[int, int]]:
    """Every (i, j) with Jaccard(left[i], right[j]) >= t; empty sets never match. Passing one list
    twice gives a self-join: (i, i) and both orders of every pair (lift.data.build links prompts so).

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
    for i, j in jaccard_pairs([g.ins for g in mine], [g.ins for g in held], nd.instruction):
        hits[i].add(j)
    for i, j in jaccard_pairs([g.res for g in mine], [g.res for g in held], nd.response):
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
    """(candidate, variant): every candidate deduplicated (the standard recipe), plus the stress
    candidate raw (the stress recipe). The plan is fixed, as README pre-registers the recipes, so
    train.yaml must say so: ValueError naming the keys unless standard.dedup is true and stress.dedup
    false (a missing stress.dedup would inherit true), raised before anything is written."""
    std, stress = train_cfg["standard"], train_cfg["stress"]
    wrong = [f"{key} is {'missing' if value is None else json.dumps(value)}, not {json.dumps(want)}"
             for key, value, want in (("standard.dedup", std.get("dedup"), True),
                                      ("stress.dedup", stress.get("dedup"), False)) if value is not want]
    if wrong:
        raise ValueError(f"configs/train.yaml: {'; '.join(wrong)}. lift.data.sft deduplicates every "
                         "candidate (the standard recipe) and gives the stress candidate its raw rows (the "
                         "stress recipe trains on undeduplicated rows, README): another plan needs a code "
                         "change and a docs/decisions.md entry; nothing was written")
    return [(name, "dedup") for name in train_cfg["candidates"]] + [(stress["candidate"], "raw")]


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


def _remove_other_variants(plan: Iterable[tuple[str, str]]) -> None:
    """Remove every {cand}/{variant} folder of sft_dir() and of interim/sft that ``plan`` does not
    make (a variant train.yaml no longer asks for), then each {cand} folder left empty."""
    keep, exports = set(plan), sft_dir()
    for root in (exports, interim_dir() / "sft"):
        if not root.is_dir():
            continue
        for cand in sorted(p for p in root.iterdir() if p.is_dir()):
            for folder in sorted(p for p in cand.iterdir() if p.is_dir()):
                if (cand.name, folder.name) in keep:
                    continue
                shutil.rmtree(folder)
                if root == exports:
                    log.warning("sft %s/%s: removed, the current plan does not make it; lift sync deletes "
                                "nothing, so delete data/sft/%s/%s from the runs repo by hand",
                                cand.name, folder.name, cand.name, folder.name)
            if not any(cand.iterdir()):
                cand.rmdir()


def _write_variant(name: str, variant: str, curated: Sequence[dict], n: int, *, rows_target: int,
                   seed: int, train_pool: int, summary: dict | None, rules: Mapping[str, Any]) -> dict:
    """Sample ``n`` curated rows by sha256 rank, write the export, the CuratorKIT copies and
    selection.json (with ``rules``, the build's ``rules_stamp``) into sft_dir()/name/variant (rebuilt
    from scratch). Returns selection.json."""
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
        "rules": dict(rules),
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
    """Curate and sample every variant (each candidate deduplicated, plus the stress candidate raw),
    write sft_dir()/report.json and return it. Once every variant is curated and its count checked
    (a refused build changes nothing under sft_dir()), it removes every variant folder of
    sft_dir() and interim/sft the plan does not make, so sft_dir() holds exactly the report's variants.

    The plan (``_variants``) is fixed: train.yaml standard.dedup must be true and stress.dedup false
    (ValueError otherwise). Row counts (train.yaml ``equal_size``, a required key: a train config
    without it is refused with KeyError, never given the rule item 47 replaced): with ``equal_size``
    true, every variant takes N = min(rows, each candidate's deduplicated count), the Nov 3 rule. With
    ``equal_size`` false (the committed value since Nov 4, decisions item 47), each variant takes
    min(rows, its own curated count). Both checks run before anything is written.

    ``train_pools`` is exclude_heldout's result; ``pool_counts`` (load_pools' counters, by source)
    fills the loaded and dropped counts of the report. curate None -> run_curatorkit. The report:
    {"decision": {"rows_target", "equal_size", "rows": N | None (None unless equal_size),
                  "rows_by_variant": {"aya/dedup": n, ...},
                  "limited_by": candidate | None (equal_size) or [variants under the target] | None},
     "rules": ``rules_stamp`` (also in every selection.json; ``export_problems`` compares it),
     "seed",
     "sources": {name: {loaded, dropped, pool, held_out, near_dup_excluded, near_dup_of, train_pool}},
     "variants": {"aya/dedup": selection.json without ids and rules, ...}}.
    """
    cfg = cfg if cfg is not None else config.load("project")
    train_cfg = train_cfg if train_cfg is not None else config.load("train")
    curate = curate if curate is not None else run_curatorkit
    target, seed = int(train_cfg["rows"]), int(cfg["split_seed"])
    equal = bool(train_cfg["equal_size"])           # required: no fallback to the replaced Nov 3 rule
    plan = _variants(train_cfg)
    stamp = rules_stamp(cfg=cfg, train_cfg=train_cfg)
    curated = {(name, variant): _curate(name, variant, _rows(train_pools[name]), curate)
               for name, variant in plan}
    if equal:
        sizes = {name: len(curated[(name, "dedup")][0]) for name in train_cfg["candidates"]}
        n = min(target, *sizes.values())
        limited_by: Any = None if n == target else min(sizes, key=lambda k: sizes[k])
        if n < 1:
            raise ValueError(f"no SFT rows left to train on: curated sizes {sizes}")
        if n < target:
            log.warning("SFT candidates use %d rows, below the target %d (limited by %s, curated sizes %s): "
                        "both candidates and the %s raw sample take %d (equal-size rule)",
                        n, target, limited_by, sizes, train_cfg["stress"]["candidate"], n)
        for (name, variant), (rows, _) in curated.items():
            if len(rows) < n:
                raise ValueError(f"{name}/{variant} has {len(rows)} curated rows, fewer than N = {n}")
        counts = {key: n for key in plan}
    else:
        n = None
        counts = {key: min(target, len(curated[key][0])) for key in plan}
        empty = [f"{name}/{variant}" for (name, variant), c in counts.items() if c < 1]
        if empty:
            raise ValueError(f"no SFT rows left to train on for {', '.join(empty)}")
        short = [f"{name}/{variant}" for (name, variant), c in counts.items() if c < target]
        limited_by = short or None
        if short:
            log.warning("SFT variants below the target %d rows (no equal-size rule): %s", target,
                        ", ".join(f"{name}/{variant} {c}" for (name, variant), c in counts.items()))
    _remove_other_variants(plan)
    variants = {}
    for name, variant in plan:
        rows, summary = curated[(name, variant)]
        selection = _write_variant(name, variant, rows, counts[(name, variant)], rows_target=target,
                                   seed=seed, train_pool=len(_rows(train_pools[name])), summary=summary,
                                   rules=stamp)
        variants[f"{name}/{variant}"] = {k: v for k, v in selection.items() if k not in ("ids", "rules")}
    report = {
        "decision": {"rows_target": target, "equal_size": equal, "rows": n,
                     "rows_by_variant": {f"{name}/{variant}": c for (name, variant), c in counts.items()},
                     "limited_by": limited_by},
        "rules": stamp,
        "seed": seed,
        "sources": {name: _source_report(_rows(pool), pool[1] if isinstance(pool, tuple) else [],
                                         (pool_counts or {}).get(name))
                    for name, pool in train_pools.items()},
        "variants": variants,
    }
    _write_json(sft_dir() / "report.json", report)
    return report


def check_frozen_heldout(heldout: Iterable[PromptRecord | SftRow]) -> None:
    """If data/splits/utility.json exists (lift.splits.SPLITS_DIR), it must hold exactly the pids of
    ``heldout``. The draw is a function of the pools, so another pool revision, cleaning rule
    (unclear.max_repetition_ratio, languages.accept_labels.hi, langid.min_prob or revision) or
    ``heldout`` setting draws other rows, and training could then take frozen dev/final rows.
    ValueError naming the mismatch (counts and pids); nothing to check while no split file exists."""
    path = splits.SPLITS_DIR / "utility.json"
    if not path.exists():
        return
    frozen = set(splits.load_split("utility"))
    drawn = {pid for pid, _, _ in _heldout_items(heldout)}
    if frozen != drawn:
        lost, new = sorted(frozen - drawn), sorted(drawn - frozen)
        raise ValueError(f"{path} holds {len(frozen)} utility pids but the held-out draw gives {len(drawn)}: "
                         f"{len(lost)} frozen pids not drawn (e.g. {lost[:5]}), {len(new)} drawn pids not "
                         f"frozen (e.g. {new[:5]}). The SFT pools, their cleaning or sources.sft.heldout "
                         "changed since the split was frozen; nothing was written")


def build_candidate(name: str, *, dedup: bool = True, rows: int = 3000, cfg: Mapping[str, Any] | None = None,
                    train_cfg: Mapping[str, Any] | None = None, curate: Curate | None = None,
                    pools: Mapping[str, Any] | None = None) -> Path:
    """One candidate end to end (held-out set and exclusion included) with min(rows, curated) rows,
    no equal-size rule and no report.json (so ``export_problems`` lists its folder: it is no training
    export); build.py uses build_candidates. pools None -> load_pools(cfg=cfg). A frozen utility split
    must hold exactly the held-out pids (``check_frozen_heldout``, before anything is written). Returns
    the directory holding sft_messages.jsonl and the CuratorKIT copies."""
    cfg = cfg if cfg is not None else config.load("project")
    train_cfg = train_cfg if train_cfg is not None else config.load("train")
    if name not in train_cfg["candidates"]:
        raise ValueError(f"unknown candidate {name!r}; train.yaml lists {train_cfg['candidates']}")
    pools = pools if pools is not None else load_pools(cfg=cfg)
    heldout = build_utility_heldout(pools, cfg=cfg)
    check_frozen_heldout(heldout)
    train = exclude_heldout(pools, heldout, cfg=cfg)
    variant = "dedup" if dedup else "raw"
    pool = train[name][0]
    curated, summary = _curate(name, variant, pool, curate if curate is not None else run_curatorkit)
    _write_variant(name, variant, curated, min(rows, len(curated)), rows_target=rows,
                   seed=int(cfg["split_seed"]), train_pool=len(pool), summary=summary,
                   rules=rules_stamp(cfg=cfg, train_cfg=train_cfg))
    return sft_dir() / name / variant


# ---------------------------------------------------------------- the rules stamp and the trainer's check


def rules_stamp(*, cfg: Mapping[str, Any] | None = None,
                train_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The rules a build of the exports applies besides the pinned data: RULES_VERSION (this code's),
    the settings that shape the rows (configs/project.yaml: sources.sft, its pins and heldout;
    unclear.max_repetition_ratio; languages.accept_labels.hi; langid; split_seed, which ranks the
    held-out draw and the samples; configs/train.yaml: rows, equal_size, candidates and
    stress.candidate) and the sha256 of the frozen data/splits/utility.json (lift.splits.SPLITS_DIR;
    None while it does not exist). JSON values, so a stamp read back from a file compares equal.
    build_candidates writes it into sft/report.json and every selection.json ("rules");
    ``export_problems`` compares them with the stamp the current code and configs give. KeyError for
    a config that lacks one of these settings."""
    cfg = cfg if cfg is not None else config.load("project")
    train_cfg = train_cfg if train_cfg is not None else config.load("train")
    utility = splits.SPLITS_DIR / "utility.json"
    stamp = {
        "version": RULES_VERSION,
        "sources_sft": cfg["sources"]["sft"],
        "max_repetition_ratio": cfg["unclear"]["max_repetition_ratio"],
        "accept_labels_hi": cfg["languages"]["accept_labels"][LANG],
        "langid": cfg["langid"],
        "split_seed": int(cfg["split_seed"]),
        "train": {"rows": int(train_cfg["rows"]), "equal_size": bool(train_cfg["equal_size"]),
                  "candidates": list(train_cfg["candidates"]),
                  "stress_candidate": train_cfg["stress"]["candidate"]},
        "utility_split_sha256": sha256_file(utility) if utility.exists() else None,
    }
    return json.loads(json.dumps(stamp))


def export_problems(*, cfg: Mapping[str, Any] | None = None,
                    train_cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Every reason the exports under sft_dir() must not be trained on, each with its remedy; empty
    when they may be. A trainer calls it before it reads an export and refuses unless it is empty. No
    torch and no network: it reads the configs, data/splits/utility.json and the files under sft_dir().
    The reasons:
    - sft/report.json is missing or cannot be read (then nothing else is checked);
    - it carries no rules stamp (a build from before the stamp, such as the Nov 4 one, which applied
      RULES_VERSION 1), or another stamp than ``rules_stamp`` gives now (the settings that differ are
      named), or the configs give none (a missing key; a dedup setting the plan refuses);
    - it does not list a variant train.yaml's plan makes;
    - a {cand}/{variant} folder on disk that it does not list (a variant the plan no longer makes, such
      as the Nov 3 aya/raw: lift sync deletes nothing, so a pull brings it back until it is deleted from
      the runs repo by hand);
    - a variant it lists whose sft_messages.jsonl or selection.json is missing, whose selection.json is
      not the one it lists (another sha256 or stamp: a build cut short, a folder from another build), or
      whose export does not match its selection.json's sha256 (edited, or synced in part)."""
    cfg = cfg if cfg is not None else config.load("project")
    train_cfg = train_cfg if train_cfg is not None else config.load("train")
    root = sft_dir()
    path = root / "report.json"
    if not path.exists():
        return [f"data/sft/report.json is missing: {_REBUILD}"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        listed = dict(report["variants"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"data/sft/report.json cannot be read ({type(e).__name__}): {_REBUILD}"]
    problems: list[str] = []
    stamp = report.get("rules")
    try:
        now, plan = rules_stamp(cfg=cfg, train_cfg=train_cfg), _variants(train_cfg)
    except (KeyError, ValueError) as e:
        now, plan = None, []
        problems.append(f"the configs give no rules stamp or plan to check the exports against "
                        f"({type(e).__name__}: {e})")
    if stamp is None:
        problems.append("data/sft/report.json carries no rules stamp: it was built before lift.data.sft "
                        "recorded the rules a build applies (the Nov 4 build applied RULES_VERSION 1: "
                        "answers under 20 characters were never compared with the held-out set, and reversed "
                        f"Dolly_T pairs were kept), so its exports may hold rows the current rules drop: "
                        f"{_REBUILD}")
    elif now is not None and stamp != now:
        problems.append("data/sft/report.json was built under other rules than the current code and configs "
                        f"give ({_stamp_diff(stamp, now)}): {_REBUILD}")
    problems += [f"data/sft/report.json does not list {name}/{variant}, which train.yaml's plan makes: "
                 f"{_REBUILD}" for name, variant in plan if f"{name}/{variant}" not in listed]
    for key in sorted(f"{c.name}/{v.name}" for c in root.iterdir() if c.is_dir()
                      for v in c.iterdir() if v.is_dir()):
        if key not in listed:
            problems.append(f"data/sft/{key} is on disk, but data/sft/report.json does not list it (a "
                            "variant the plan no longer makes, such as the Nov 3 aya/raw, or a folder from "
                            f"another build): delete data/sft/{key} from the runs repo by hand (lift sync "
                            "deletes nothing), and here (a rebuild removes it too)")
    for key, entry in sorted(listed.items()):
        problems += _variant_problems(key, entry, stamp)
    return problems


def _variant_problems(key: str, entry: Any, stamp: Any) -> list[str]:
    """What keeps the variant ``key`` that report.json lists (as ``entry``, under the report's rules
    ``stamp``) from being trained on: a missing file, a selection.json that is not the listed one, an
    export whose sha256 is not its selection.json's."""
    folder = sft_dir() / key
    export, sel_path = folder / "sft_messages.jsonl", folder / "selection.json"
    missing = [p.name for p in (export, sel_path) if not p.is_file()]
    if missing:
        return [f"data/sft/{key}: data/sft/report.json lists it, but its {' and '.join(missing)} is missing "
                f"(synced in part, or a build cut short): `lift sync pull` again, or {_REBUILD}"]
    try:
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
        recorded = selection["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"data/sft/{key}/selection.json cannot be read ({type(e).__name__}): {_REBUILD}"]
    out = []
    listed_sha = entry.get("sha256") if isinstance(entry, dict) else None
    if recorded != listed_sha or selection.get("rules") != stamp:
        out.append(f"data/sft/{key}/selection.json is not the one data/sft/report.json lists (another "
                   f"sha256 or rules stamp: a build cut short, or a folder from another build): {_REBUILD}")
    actual = sha256_file(export)
    if actual != recorded:
        out.append(f"data/sft/{key}/sft_messages.jsonl does not match its selection.json (sha256 "
                   f"{actual[:12]}, recorded {str(recorded)[:12]}): edited by hand, or synced in part: "
                   f"{_REBUILD}")
    return out


def _stamp_diff(old: Any, new: Mapping[str, Any], n: int = 6) -> str:
    """The settings two rules stamps disagree on, "key old, now new" with dotted keys (at most ``n``)."""
    a, b = _flat(old if isinstance(old, dict) else {}), _flat(new)
    keys = sorted(k for k in a.keys() | b.keys() if k not in a or k not in b or a[k] != b[k])

    def shown(d: Mapping[str, Any], k: str) -> str:
        if k not in d:
            return "missing"
        text = json.dumps(d[k], ensure_ascii=False)
        return text if len(text) <= 40 else text[:37] + "..."

    if not keys:
        return "they differ"
    parts = [f"{k} {shown(a, k)}, now {shown(b, k)}" for k in keys[:n]]
    return "; ".join(parts) + (f"; and {len(keys) - n} more" if len(keys) > n else "")


def _flat(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """A nested dict as {dotted key: leaf value}."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out
