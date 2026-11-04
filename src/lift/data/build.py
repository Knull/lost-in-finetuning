"""Build every prompt set, freeze the splits and make the SFT candidates (``lift data build``).

Inputs, all pinned in configs/project.yaml: the PolyRefuse clone (raw/Multilingual-Refusal, made
or checked by ``polyrefuse.ensure_clone``, then ``polyrefuse.check_alignment`` on every build that
reads it), the Hub sources through one ``Hub`` (OR-Bench-Hard-1K, Aya Red-teaming, Belebele, the SFT
sources), Google Translate through ``translate.translate_records`` (one rate-limited backend for the
whole build; every output cached), GlotLID and CuratorKIT. ``BuildContext`` holds them; tests pass
fakes.

Where each file comes from (``build_all``; sets in SETS order, languages in config order):
    harmful      en: PolyRefuse's English sources (test, train, val); zh es ar th: PolyRefuse's own
                 translations; hi: translated from the English records.
    easy_benign  test + train (harmless val is unused): en: English sources; es ar th: PolyRefuse;
                 zh and hi: translated (no zh harmless test file; the zh train file is misaligned).
    hard_benign  en: the OR-Bench sample; hi zh es ar th: translated from it.
    redteam      hi only: Aya Red-teaming Hindi, split "final".
    utility      hi only: the shared held-out set (lift.data.sft).
    belebele     every language, in eng_Latn order (dicts, not PromptRecords).
    sft          the CuratorKIT candidates (lift.data.sft), excluding the same held-out set.
A PolyRefuse family comes from PolyRefuse iff ``polyrefuse.has_file``; otherwise en reads the
English source and every other language is translated from the English records, never from another
translation. redteam, utility and sft are Hindi: a build whose --langs leave out hi skips them.
Whatever --only and --langs say, a build of harmful or easy_benign reads the English of all five
PolyRefuse families and every translation of languages.all that has_file lists (both kinds), so the
links below, and the splits and exclusions they make, never depend on the build's options.

Links (``_links``; configs/project.yaml split_near_dup, the held-out instruction rule): two prompts
are linked if their English (text_en; utility: its Hindi text) is equal by dup_key or has a character
3-gram Jaccard >= 0.7 (exact fractions, one self-join on an inverted index), or if their PolyRefuse
translations into one language are equal by dup_key. On the pinned data the 1,571 PolyRefuse prompts
(both kinds, test train val; harmless val is unused) give 43 linked pairs, all harmful and all by
their English: 11 equal (the 10 test repeats and prh-train-0226) and 32 near (0.707-0.946); 21 join
two test prompts, 18 a train/val and a test prompt, 4 two train/val prompts. 14 are also equal in a
translation (th 12, zh 11, es 10, ar 7) and none in a translation alone, so that rule is a guard
today. No harmful-harmless link.

Per (set, lang), in order: the base records (English; Hindi for redteam and utility) -> pid
exclusions -> PolyRefuse load or translation -> readers' spot-check verdicts -> split stamp ->
same-text check -> GlotLID -> the file, its excluded file and its report.
- Exclusions, in every language and before anything is translated (``_test_links``): a train/val
  prompt of either kind with a direct link to a test prompt of either kind is out, as
  ``duplicate_of:<pid>`` naming the smallest test pid with the same English, else
  ``near_duplicate_of:<pid>`` naming the smallest linked test pid, with a ``detail`` giving the
  evidence for that pair and the other linked test pids ("en 0.946; th same text", "en 0.819; also
  prh-test-0251"). One linked only to train/val prompts stays. On the pinned data 17, all harmful:
  14 train, 3 val; prh-train-0226 is the one duplicate_of (prh-test-0008); prh-train-0009 and
  prh-train-0252 are also prh-test-0050 and prh-test-0079 word for word in th. Aya Red-teaming's
  repeats (sources.redteam_records) are ``duplicate_of`` too.
- Same text after the split stamp (``_same_text``, every PromptRecord file): a train/val record whose
  text (dup_key) a dev/final record of the file also has is left out of that file alone, listed as
  ``same_text_as:<the smallest such pid>``. One text in two split-file splits (fit, dev, final)
  cannot be regrouped once the split is frozen: listed as ``cross_split_duplicates`` (pids and splits)
  in the report and counted in build.json's files, with a WARNING. The links make this a no-op in
  PolyRefuse's own languages; it guards hi, our zh harmless translations and hard_benign's.
- Translation failures (translate.py's rules) are listed in the report and the pid is left out of
  that file, never replaced by English. TranslationBlocked (HTTP 429, or ten items in a row failing
  every retry) stops all translation for the rest of the run: every file that still needs one is
  skipped (never a partial file), everything else is built, and the summary says blocked. So does
  --skip-translation. A rerun resumes from the cache. Items that failed with a transport error, or
  with --cache-only were not cached, leave their file incomplete (summary "incomplete").
- hard_benign items Google failed take an NLLB translation from the cache when one passes the same
  checks; ``fallback_engine="nllb"`` (--fallback-engine nllb, needs torch and transformers) makes
  the missing ones. Opt-in, so the files stay a function of the cache whatever the flags.
- GlotLID: every record gets ``lid``; one its language check rejects stays in the file and is
  listed (lid_failures). Belebele: the question is checked, no field added.

Splits (``freeze_split``): data/splits/{harmful,easy_benign,hard_benign,utility,belebele}.json, over
the test pids of harmful and easy_benign and every pid of the others. A committed file is loaded and
must cover exactly the set's pids and only the configured split names (ValueError otherwise: the
sources changed); it is never rewritten, and a fresh assignment that differs only logs a warning. A
missing file is assigned once from the sorted pids, cfg sizes and split_seed and saved, with a
WARNING to commit it. Groups share a split (assign_splits ``groups``): the connected components of
the links among a set's split pids (a train prompt never joins two test prompts; hard_benign and
utility get the English rule too), and Belebele's questions on one passage (one ``link``). On the
pinned data: harmful 21 groups of two (the 10 repeats and 11 near pairs at 0.715-0.890; prh-test-0000
and prh-test-0142 are also one zh text); belebele 412 passages with two questions (76 have one); no
group in easy_benign, hard_benign (closest pair 0.584) or utility (0.692). Train and val pids keep
"train" and "val"; redteam is "final". save_split stamps the creation date into a new file. A build
of utility or sft freezes (or checks) the utility split first and every training pool must be
disjoint from its pids, so a held-out draw that changed (lift.data.sft) stops it with ValueError
before a training row is written.

Spot checks: ``write_spotcheck_sheets`` writes, per language (default hi zh es), n unchecked
dev/final machine translations per set by sha256 rank (ours and PolyRefuse's) and a failures sheet
with every open translation failure plus every machine translation GlotLID rejected. Readers fill
verdict ok / fixed (with the corrected text) / bad. ``apply_spotchecks`` checks a returned sheet
against the current build, raising one SheetError with every problem before anything is written,
then appends the verdicts to interim/translations/spotchecks.jsonl and rebuilds those files from the
cache (no network, splits untouched). Every build applies the latest verdict per (lang, pid): fixed
replaces the text (checked; edited when it differs) if the English is unchanged; ok marks it checked
and bad moves it to prompts/excluded/ if the reader saw the current text; a verdict on a failed item
builds the record from the English one (ok accepts the engine's output). A verdict that no longer
fits is logged and listed, never applied.

Files under the data dir (lift.paths). A rebuild reproduces every file byte for byte except build.json
(run id, timings) and CuratorKIT's run records (lift.data.sft): interim/sft/ and the
sft/{cand}/{variant}/curatorkit/ copies of manifest.json and lexsi_provenance.json carry
run_timestamp, wall_clock_seconds and created_at.
    prompts/{set}_{lang}.jsonl           PromptRecord lines (belebele: dicts), sorted by pid
    prompts/excluded/{set}_{lang}.jsonl  records readers marked bad (extra.excluded, excluded_reason)
    prompts/reports/{set}_{lang}.json    counts by split and engine; lid_failures, translation
                                         failures (and how each was resolved), exclusions (reason;
                                         detail for links), spot-check counts, the file's sha256;
                                         PromptRecord files also list cross_split_duplicates
                                         (belebele reports have no such key); no text
    prompts/reports/build.json           the last build: options, sources, splits (status, counts,
                                         groups and their sizes), files (PromptRecord files also
                                         count cross_split_duplicates), skipped, the SFT row
                                         decision, PolyRefuse alignment, timings
    interim/translations/                the translation cache and spotchecks.jsonl
    sft/...                              the candidates and sft/report.json (lift.data.sft)
Logs, reports and exception messages carry counts, pids, labels and hashes, never prompt text.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, NamedTuple

from lift import config, manifest, splits
from lift.data import polyrefuse, sft, sources, translate
from lift.data.sources import Hub
from lift.data.text import char_ngrams, clean_text, dup_key
from lift.data.translate import (
    Backend,
    TranslationBlocked,
    TranslationCache,
    TranslationError,
    TranslationFailure,
)
from lift.io import append_jsonl, read_jsonl, sha256_file, sha256_text, write_jsonl
from lift.lang import LangIdentifier, lid_info
from lift.paths import REPO_ROOT, interim_dir, prompts_dir, sft_dir
from lift.schemas import PromptRecord, TranslationInfo

log = logging.getLogger(__name__)

SETS: tuple[str, ...] = ("harmful", "easy_benign", "hard_benign", "redteam", "utility", "belebele", "sft")
SPLIT_SETS: tuple[str, ...] = ("harmful", "easy_benign", "hard_benign", "utility", "belebele")
TRANSLATED_SETS: tuple[str, ...] = ("harmful", "easy_benign", "hard_benign")   # English in, all languages out
HINDI_SETS: tuple[str, ...] = ("redteam", "utility", "sft")
SHEET_COLUMNS: tuple[str, ...] = ("item_id", "set", "english", "translation", "verdict", "corrected", "note")
VERDICTS: tuple[str, ...] = ("ok", "fixed", "bad")
SHEETS_DIR = REPO_ROOT / "validation" / "sheets"          # gitignored

_KIND = {set_name: kind for kind, set_name in polyrefuse.SET_NAMES.items()}   # easy_benign -> harmless
_SPLIT_NAMES = ("train", "val", "fit", "dev", "final")
_ITEM_ID = re.compile(r"[a-z]{2}-[a-z]{3}-[a-z0-9-]{1,24}")
_VERDICT_FIELDS = ("lang", "pid", "set", "verdict", "corrected", "note", "source_sha", "seen_sha")


class SheetError(ValueError):
    """A returned spot-check sheet that cannot be applied. ``problems`` lists every problem by sheet
    row and item id, never cell text."""

    def __init__(self, message: str, problems: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.problems = list(problems)


class _OnFirstCall:
    """A backend made on its first call, so a build that translates nothing never loads the engine."""

    def __init__(self, make: Callable[[], Backend]) -> None:
        self._make = make
        self._backend: Backend | None = None

    def __call__(self, text: str, lang: str) -> str | None:
        if self._backend is None:
            self._backend = self._make()
        return self._backend(text, lang)


@dataclass
class BuildContext:
    """What a build reads and how it translates. A service left None is made on first use and kept,
    so one build loads GlotLID once and spaces every Google request through one backend."""

    cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    lid: LangIdentifier | None = None                  # None -> lift.lang.LangID(cfg=cfg)
    hub: Hub | None = None                             # None -> sources.HFHub(cfg)
    translator: Backend | None = None                  # None -> translate.google_backend(cfg)
    curate: Callable[..., list[dict]] | None = None    # None -> sft.run_curatorkit (CuratorKIT)
    polyrefuse_root: Path | None = None                # None -> polyrefuse.ensure_clone(cfg=cfg)
    skip_translation: bool = False
    cache_only: bool = False
    retranslate_failed: bool = False
    fallback_engine: str | None = None                 # "nllb": make NLLB translations for hard_benign
    fallback_translator: Backend | None = None         # None -> translate.nllb_backend(cfg=cfg)

    def __post_init__(self) -> None:
        if self.cache_only and self.retranslate_failed:
            raise ValueError("cache_only and retranslate_failed exclude each other")
        if self.fallback_engine not in (None, "nllb"):
            raise ValueError(f"unknown fallback engine {self.fallback_engine!r}; the only one is 'nllb'")

    @classmethod
    def default(cls, **overrides: Any) -> BuildContext:
        """The project and train configs plus ``overrides``; every service made on first use."""
        fields: dict[str, Any] = {"cfg": config.load("project"), "train_cfg": config.load("train")}
        fields.update(overrides)
        return cls(**fields)

    def load_lid(self) -> LangIdentifier:
        if self.lid is None:
            from lift.lang import LangID

            self.lid = LangID(cfg=self.cfg)
        return self.lid

    def load_hub(self) -> Hub:
        if self.hub is None:
            self.hub = sources.HFHub(self.cfg)
        return self.hub

    def load_root(self) -> Path:
        if self.polyrefuse_root is None:
            self.polyrefuse_root = polyrefuse.ensure_clone(cfg=self.cfg)
        return Path(self.polyrefuse_root)

    def backend(self, engine: str) -> Backend:
        """The one backend per engine for this context ("google" or "nllb"), made on its first call."""
        cfg = self.cfg
        if engine == "google":
            if self.translator is None:
                self.translator = _OnFirstCall(lambda: translate.google_backend(cfg))
            return self.translator
        if engine == "nllb":
            if self.fallback_translator is None:
                self.fallback_translator = _OnFirstCall(lambda: translate.nllb_backend(cfg=cfg))
            return self.fallback_translator
        raise ValueError(f"unknown engine {engine!r}")


# ------------------------------------------------------------------------------------ the build


class _Run:
    """One pass over some files: its options, the readers' verdicts, and what the summary reports."""

    def __init__(self, ctx: BuildContext, *, skip_translation: bool, cache_only: bool,
                 retranslate_failed: bool, fallback_engine: str | None, create_splits: bool = True) -> None:
        if cache_only and retranslate_failed:
            raise ValueError("cache_only and retranslate_failed exclude each other")
        self.ctx, self.cfg = ctx, ctx.cfg
        self.skip_translation, self.cache_only = skip_translation, cache_only
        self.retranslate_failed, self.fallback_engine = retranslate_failed, fallback_engine
        self.create_splits = create_splits
        self.verdicts = _read_verdicts()
        self.blocked: str | None = None
        self.skipped: list[dict[str, str]] = []
        self.files: dict[str, dict[str, Any]] = {}
        self.splits: dict[str, dict[str, Any]] = {}
        self.timings: dict[str, float] = {}

    def skip(self, set_name: str, lang: str, reason: str) -> None:
        log.warning("%s_%s not written: %s", set_name, lang, reason)
        self.skipped.append({"set": set_name, "lang": lang, "reason": reason})


@dataclass
class _Sources:
    """What every language of a set is built from, and the frozen splits."""

    base: dict[str, list[PromptRecord]] = field(default_factory=dict)       # set -> records, exclusions out
    families: dict[str, dict[tuple[str, str], list[PromptRecord]]] = field(default_factory=dict)
    exclusions: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    links: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)   # (pid, pid) -> how (_links)
    splits: dict[str, dict[str, str]] = field(default_factory=dict)
    root: Path | None = None
    alignment: dict[str, Any] | None = None
    belebele_en: list[dict] = field(default_factory=list)
    pools: dict[str, Any] | None = None
    train_pools: dict[str, Any] | None = None


@dataclass
class _File:
    """One (set, lang) file while it is assembled."""

    set_name: str
    lang: str
    records: list[PromptRecord] = field(default_factory=list)
    excluded: list[PromptRecord] = field(default_factory=list)
    exclusions: list[dict[str, str]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    fallback: dict[str, Any] | None = None
    applied: Counter[str] = field(default_factory=Counter)
    ignored: list[dict[str, str]] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)        # pids whose verdict was looked at


def build_all(*, only: Iterable[str] | None = None, langs: Iterable[str] | None = None,
              skip_translation: bool = False, ctx: BuildContext | None = None) -> dict[str, Any]:
    """Build ``only`` these sets (default SETS) in ``langs`` (default languages.all), freeze their
    splits, write prompts/reports/build.json and return that summary.

    ``summary["complete"]`` is False when translation was blocked (``blocked``), a file was skipped
    for translation (``skipped``) or a file misses items that only a rerun can translate
    (``incomplete``). Listed failures (wrong language, empty, identical) do not make it incomplete.
    """
    ctx = ctx if ctx is not None else BuildContext.default()
    cfg = ctx.cfg
    sets = _check_sets(only)
    langs = _check_langs(langs, cfg)
    pairs = [(s, lang) for s in sets if s != "sft" for lang in _set_langs(s, cfg) if lang in langs]
    want_sft = "sft" in sets and cfg["languages"]["target"] in langs
    run = _Run(ctx, skip_translation=skip_translation or ctx.skip_translation, cache_only=ctx.cache_only,
               retranslate_failed=ctx.retranslate_failed, fallback_engine=ctx.fallback_engine)
    options = {"only": sets, "langs": langs, "skip_translation": run.skip_translation,
               "cache_only": run.cache_only, "retranslate_failed": run.retranslate_failed,
               "fallback_engine": run.fallback_engine}
    record = manifest.start_run("data-build", **options)
    started = time.perf_counter()
    try:
        needed = {s for s, _ in pairs} | ({"sft"} if want_sft else set())
        if not needed:
            log.warning("nothing to build: sets %s in languages %s", sets, langs)
        src = _timed(run, "sources", lambda: _prepare(run, needed))
        for set_name, lang in pairs:
            _timed(run, f"{set_name}_{lang}", lambda s=set_name, lang=lang: _build_file(run, src, s, lang))
        sft_summary = _timed(run, "sft", lambda: _build_sft(run, src)) if want_sft else None
    except BaseException as e:
        manifest.finish_run(record, error=type(e).__name__)
        raise
    run.timings["total"] = round(time.perf_counter() - started, 3)
    summary = _summary(run, src, options, sft_summary, record["run_id"])
    _write_json(prompts_dir() / "reports" / "build.json", summary)
    manifest.finish_run(record, complete=summary["complete"], blocked=summary["blocked"],
                        files=len(run.files), skipped=len(run.skipped))
    log.info("data build %s: %d files, %d records, %d skipped%s", "complete" if summary["complete"] else
             "INCOMPLETE", summary["totals"]["files"], summary["totals"]["records"], len(run.skipped),
             "; translation BLOCKED: rerun later to resume from the cache" if run.blocked else "")
    return summary


def _timed(run: _Run, name: str, step: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    out = step()
    run.timings[name] = round(time.perf_counter() - start, 3)
    return out


def _check_sets(only: Iterable[str] | None) -> list[str]:
    if only is None:
        return list(SETS)
    chosen = list(dict.fromkeys(only))
    unknown = [s for s in chosen if s not in SETS]
    if unknown:
        raise ValueError(f"unknown sets {unknown}; choose from {list(SETS)}")
    return [s for s in SETS if s in chosen]


def _check_langs(langs: Iterable[str] | None, cfg: Mapping[str, Any]) -> list[str]:
    known = list(cfg["languages"]["all"])
    if langs is None:
        return known
    chosen = set(langs)
    unknown = sorted(chosen - set(known))
    if unknown:
        raise ValueError(f"unknown languages {unknown}; choose from {known}")
    return [lang for lang in known if lang in chosen]


def _set_langs(set_name: str, cfg: Mapping[str, Any]) -> list[str]:
    return [cfg["languages"]["target"]] if set_name in HINDI_SETS else list(cfg["languages"]["all"])


def _base_lang(set_name: str, cfg: Mapping[str, Any]) -> str:
    return "en" if set_name in TRANSLATED_SETS else cfg["languages"]["target"]


def _pid(rec: PromptRecord) -> str:
    return rec.pid


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


# ---------------------------------------------------------------------------- sources and splits


class _Near(NamedTuple):
    """split_near_dup: the character n-gram size and the Jaccard threshold (exact: 0.7 -> 7/10)."""

    ngram: int
    jaccard: Fraction


def _split_near(cfg: Mapping[str, Any]) -> _Near:
    nd = cfg["split_near_dup"]
    near = _Near(int(nd["ngram"]), Fraction(str(nd["jaccard"])))
    if near.ngram < 1 or not 0 < near.jaccard <= 1:
        raise ValueError(f"split_near_dup needs ngram >= 1 and jaccard in (0, 1], got {dict(nd)}")
    return near


def _prepare(run: _Run, sets: set[str]) -> _Sources:
    """Load what the sets are built from (exclusions applied), link their prompts, then freeze their
    splits. "sft" freezes (or checks) the utility split as well, so a held-out draw that no longer
    matches a frozen utility split stops the build before any training row is written."""
    ctx, cfg = run.ctx, run.cfg
    near = _split_near(cfg)
    src = _Sources()
    if sets & {"harmful", "easy_benign"}:
        _prepare_polyrefuse(run, src, near)
    if "hard_benign" in sets:
        src.base["hard_benign"] = sources.load_orbench_hard_sample(cfg=cfg, hub=ctx.load_hub())
    if "redteam" in sets:
        records, excluded = sources.redteam_records(ctx.load_hub().rows("aya_redteaming"))
        src.base["redteam"], src.exclusions["redteam"] = records, excluded
        log.info("redteam: %d records, %d excluded %s", len(records), len(excluded),
                 [e["pid"] for e in excluded])
    if sets & {"utility", "sft"}:
        src.pools = sft.load_pools(cfg=cfg, hub=ctx.load_hub(), lid=ctx.load_lid())
        src.base["utility"] = sft.build_utility_heldout(src.pools, cfg=cfg)
        if "sft" in sets:
            src.train_pools = sft.exclude_heldout(src.pools, src.base["utility"], cfg=cfg)
    if "belebele" in sets:
        src.belebele_en = sources.load_belebele("en", cfg=cfg, hub=ctx.load_hub())
    for name in ("hard_benign", "utility"):
        if name in src.base:
            links = _links(src.base[name], {}, near)
            src.links.update(links)
            log.info("%s: %d linked pairs of prompts", name, len(links))
    for name in SPLIT_SETS:
        if name in sets or (name == "utility" and "sft" in sets):
            pids, groups = _split_pids(src, name)
            mapping, status = _freeze(name, pids, cfg=cfg, groups=groups, create=run.create_splits)
            src.splits[name] = mapping
            sizes = Counter(Counter(groups.values()).values())
            run.splits[name] = {"status": status, "n": len(mapping), "by_split": _count(mapping.values()),
                                "grouped": len(groups), "groups": sum(sizes.values()),
                                "group_sizes": {str(n): sizes[n] for n in sorted(sizes)}}
    return src


def _prepare_polyrefuse(run: _Run, src: _Sources, near: _Near) -> None:
    """The PolyRefuse English records per family, their links, and the train/val prompts linked to a
    test prompt excluded. Every translation polyrefuse.has_file lists for languages.all is loaded,
    whatever --only and --langs say, so the links, and with them the splits, never depend on the
    build's options."""
    cfg = run.cfg
    root = src.root = run.ctx.load_root()
    langs = list(cfg["languages"]["all"])
    src.alignment = polyrefuse.check_alignment(root, langs, lid=run.ctx.load_lid(), cfg=cfg)
    english = {(kind, split): polyrefuse.load_polyrefuse(root, kind, split, "en", cfg=cfg)
               for kind, split in polyrefuse.BUILD_FILES}
    translated = {lang: [rec for kind, split in polyrefuse.BUILD_FILES
                         if polyrefuse.has_file(root, kind, split, lang, cfg=cfg)
                         for rec in polyrefuse.load_polyrefuse(root, kind, split, lang, cfg=cfg)]
                  for lang in langs if lang != "en"}
    records = [rec for recs in english.values() for rec in recs]
    links = _links(records, translated, near)
    src.links.update(links)
    excluded = {e["pid"]: e for e in _test_links(records, links, near)}
    by = Counter(how for hows in links.values() for how in hows)
    log.info("PolyRefuse: %d linked pairs of prompts (%s), %d of them between test prompts; %d train/val "
             "prompts linked to a test prompt", len(links), ", ".join(f"{k} {v}" for k, v in by.items()),
             sum(_is_test(a) and _is_test(b) for a, b in links), len(excluded))
    for kind, set_name in polyrefuse.SET_NAMES.items():
        families = {(k, split): [r for r in recs if r.pid not in excluded]
                    for (k, split), recs in english.items() if k == kind}
        src.families[set_name] = families
        src.base[set_name] = sorted((r for recs in families.values() for r in recs), key=_pid)
        prefix = f"{polyrefuse.PID_PREFIX[kind]}-"
        src.exclusions[set_name] = [e for pid, e in sorted(excluded.items()) if pid.startswith(prefix)]
        for e in src.exclusions[set_name]:
            log.info("%s: %s excluded in every language: %s (%s)", set_name, e["pid"], e["reason"],
                     e["detail"])


def _english(rec: PromptRecord) -> str:
    """The text links compare: the English source (text_en), else the text itself (utility: Hindi)."""
    return rec.text_en if rec.text_en is not None else rec.text


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _is_test(pid: str) -> bool:
    """A PolyRefuse test pid (prh-test-*, prb-test-*)."""
    return any(pid.startswith(f"{prefix}-test-") for prefix in polyrefuse.PID_PREFIX.values())


def _links(records: Sequence[PromptRecord], translated: Mapping[str, Sequence[PromptRecord]],
           near: _Near) -> dict[tuple[str, str], tuple[str, ...]]:
    """Every linked pair of prompts (smaller pid first) -> what links it: "en" if their English
    (``_english``) is equal by dup_key or has a character n-gram Jaccard >= near.jaccard, then each
    language of ``translated`` (lang -> records, same pids) in which their texts are equal by dup_key.
    One self-join on an inverted n-gram index (sft.jaccard_pairs): 1,571 prompts take well under a
    second."""
    found: dict[tuple[str, str], list[str]] = defaultdict(list)
    pids = [rec.pid for rec in records]
    grams = [char_ngrams(_english(rec), near.ngram) for rec in records]
    for i, j in sft.jaccard_pairs(grams, grams, near.jaccard):          # equal dup_keys have Jaccard 1
        if i < j:
            found[_pair(pids[i], pids[j])].append("en")
    for lang, recs in translated.items():
        same: dict[str, set[str]] = defaultdict(set)
        for rec in recs:
            if key := dup_key(rec.text):
                same[key].add(rec.pid)
        for group in same.values():
            for a, b in combinations(sorted(group), 2):
                found[(a, b)].append(lang)
    return {pair: tuple(found[pair]) for pair in sorted(found)}


def _evidence(a: PromptRecord, b: PromptRecord, how: Iterable[str], near: _Near) -> str:
    """Why two prompts are linked, for a report: "en same text" or their English character n-gram
    Jaccard (rounded down to 3 places, so a pair under the threshold never shows as reaching it), then
    the languages whose texts are equal, e.g. "en 0.947; th same text"."""
    ga, gb = char_ngrams(_english(a), near.ngram), char_ngrams(_english(b), near.ngram)
    if dup_key(_english(a)) == dup_key(_english(b)):
        en = "en same text"
    else:
        score = Fraction(len(ga & gb), len(ga | gb)) if ga | gb else Fraction(0)
        en = f"en {math.floor(score * 1000) / 1000:.3f}"
    same = [lang for lang in how if lang != "en"]
    return "; ".join([en, *([" ".join(same) + " same text"] if same else [])])


def _test_links(records: Sequence[PromptRecord], links: Mapping[tuple[str, str], Sequence[str]],
                near: _Near) -> list[dict[str, str]]:
    """The exclusion {"pid", "reason", "detail"} of every train/val prompt with a direct link to a test
    prompt of either kind, by pid: ``duplicate_of:<pid>`` naming the smallest test pid with the same
    English dup_key (prh-train-0226 is prh-test-0008), else ``near_duplicate_of:<pid>`` naming the
    smallest linked test pid. ``detail`` is the evidence for that pair (``_evidence``) and the other
    linked test pids ("also ..."). A train/val prompt linked only to train/val prompts stays."""
    by_pid = {rec.pid: rec for rec in records}
    linked: dict[str, set[str]] = defaultdict(set)
    for a, b in links:
        if _is_test(a) != _is_test(b):
            test, other = (a, b) if _is_test(a) else (b, a)
            linked[other].add(test)
    out = []
    for pid, tests in sorted(linked.items()):
        key = dup_key(_english(by_pid[pid]))
        same = sorted(t for t in tests if dup_key(_english(by_pid[t])) == key)
        first = (same or sorted(tests))[0]
        detail = _evidence(by_pid[pid], by_pid[first], links[_pair(pid, first)], near)
        if len(tests) > 1:
            detail += "; also " + " ".join(sorted(tests - {first}))
        out.append({"pid": pid, "reason": f"{'duplicate_of' if same else 'near_duplicate_of'}:{first}",
                    "detail": detail})
    return out


def _split_pids(src: _Sources, name: str) -> tuple[list[str], dict[str, str]]:
    """The pids a split file covers, and the groups that must share a split (``_groups``): the
    connected components of src.links among them; Belebele's questions on one passage (one link)."""
    if name == "belebele":
        pids = [d["pid"] for d in src.belebele_en]
        first: dict[str, str] = {}
        links = [(first.setdefault(d["link"], d["pid"]), d["pid"]) for d in src.belebele_en]
        return pids, _groups(pids, links)
    pids = [r.pid for r in src.base[name] if _in_scope(name, r.pid)]
    return pids, _groups(pids, src.links)


def _in_scope(set_name: str, pid: str) -> bool:
    """Does the set's split file cover this pid? PolyRefuse sets: test rows only."""
    if set_name in _KIND:
        return pid.startswith(f"{polyrefuse.PID_PREFIX[_KIND[set_name]]}-test-")
    return set_name in SPLIT_SETS


def _groups(pids: Iterable[str], links: Iterable[tuple[str, str]]) -> dict[str, str]:
    """pid -> the smallest pid of its group: the connected components of ``links`` among ``pids``.
    A link with an end outside ``pids`` is ignored, so a train prompt never joins two test prompts.
    Pids in no link are left out (assign_splits gives each its own slot)."""
    inside = set(pids)
    parent: dict[str, str] = {}

    def root(pid: str) -> str:
        while parent[pid] != pid:
            pid = parent[pid]
        return pid

    for a, b in links:
        if a == b or a not in inside or b not in inside:
            continue
        ra, rb = root(parent.setdefault(a, a)), root(parent.setdefault(b, b))
        parent[max(ra, rb)] = min(ra, rb)              # the root stays the component's smallest pid
    return {pid: root(pid) for pid in sorted(parent)}


def _shown(path: Path) -> str:
    """A path for a message: relative to the repo (data/splits/harmful.json) when inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def freeze_split(name: str, pids: Iterable[str], *, cfg: Mapping[str, Any],
                 groups: Mapping[str, str] | None = None, create: bool = True) -> dict[str, str]:
    """The frozen split of one set: pid -> split name.

    If data/splits/{name}.json exists it is the truth: it must cover exactly ``pids`` and use only the
    split names of cfg sizes (ValueError naming counts and example pids otherwise: the sources
    changed). It is never rewritten; a fresh assignment that differs only logs a warning, and so do
    groups the file splits. If it is missing (and ``create``): assign_splits(sorted pids, cfg sizes,
    cfg split_seed, groups), saved with save_split, and a WARNING says to commit it. ``create=False``
    raises FileNotFoundError instead.
    """
    return _freeze(name, pids, cfg=cfg, groups=groups, create=create)[0]


def _freeze(name: str, pids: Iterable[str], *, cfg: Mapping[str, Any], groups: Mapping[str, str] | None,
            create: bool) -> tuple[dict[str, str], str]:
    pids = sorted(pids)
    if len(set(pids)) != len(pids):
        raise ValueError(f"split {name}: duplicate pids")
    sizes, seed = dict(cfg["sizes"][name]), int(cfg["split_seed"])
    path = splits.SPLITS_DIR / f"{name}.json"
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"{path} does not exist: run `lift data build --only {name}` first")
        mapping = splits.assign_splits(pids, sizes, seed, groups)
        splits.save_split(name, mapping, seed=seed, sizes=sizes)
        log.warning("split %s: created %s: commit it; split files are frozen once committed (%s)", name,
                    _shown(path), ", ".join(f"{k} {v}" for k, v in _count(mapping.values()).items()))
        return mapping, "created"
    mapping = splits.load_split(name)
    missing, extra = sorted(set(pids) - mapping.keys()), sorted(mapping.keys() - set(pids))
    if missing or extra:
        why = ("The utility held-out draw changed (the SFT pools, their cleaning or sources.sft.heldout), "
               "so nothing is trained on" if name == "utility" else "The sources changed")
        raise ValueError(f"{path} covers {len(mapping)} pids but the sources give {len(pids)}: "
                         f"{len(missing)} missing (e.g. {missing[:5]}), {len(extra)} not in the sources "
                         f"(e.g. {extra[:5]}). {why}; split files are frozen and never rewritten (a new "
                         "split needs a docs/decisions.md entry)")
    unknown = sorted(pid for pid, split in mapping.items() if split not in sizes)
    if unknown:
        raise ValueError(f"{path} assigns {len(unknown)} pids to splits not in sizes.{name} {list(sizes)}, "
                         f"e.g. {unknown[:5]}")
    status = "loaded"
    try:
        fresh: dict[str, str] | None = splits.assign_splits(pids, sizes, seed, groups)
    except ValueError:
        fresh = None
    if fresh != mapping:
        differ = len(pids) if fresh is None else sum(fresh[p] != mapping[p] for p in pids)
        log.warning("split %s: %s differs from a fresh assignment on %d pids (file %s, sizes.%s %s); "
                    "keeping the committed file", name, path.name, differ, _count(mapping.values()), name,
                    sizes)
        status = "loaded; differs from a fresh assignment"
    members: dict[str, set[str]] = {}
    for pid, key in (groups or {}).items():
        members.setdefault(key, set()).add(mapping[pid])
    straddle = sorted(key for key, found in members.items() if len(found) > 1)
    if straddle:
        log.warning("split %s: %d groups of linked prompts straddle splits in %s, e.g. %s", name,
                    len(straddle), path.name, straddle[:5])
    return mapping, status


# ---------------------------------------------------------------------------- one file


def _build_file(run: _Run, src: _Sources, set_name: str, lang: str) -> None:
    if set_name == "belebele":
        _build_belebele(run, src, lang)
    else:
        _build_prompts(run, src, set_name, lang)


def _to_translate(run: _Run, src: _Sources, set_name: str, lang: str) -> list[PromptRecord]:
    """The English records this (set, lang) is translated from; empty if it needs no translation."""
    if lang == "en" or set_name not in TRANSLATED_SETS:
        return []
    if set_name == "hard_benign":
        return list(src.base[set_name])
    return [rec for (kind, split), recs in src.families[set_name].items()
            if not polyrefuse.has_file(src.root, kind, split, lang, cfg=run.cfg) for rec in recs]


def _from_files(run: _Run, src: _Sources, set_name: str, lang: str) -> list[PromptRecord]:
    """The records this (set, lang) takes as they are: the base records in their own language, or
    PolyRefuse's translations (exclusions applied)."""
    if lang == _base_lang(set_name, run.cfg):
        return list(src.base[set_name])
    if set_name not in _KIND:
        return []
    excluded = {e["pid"] for e in src.exclusions[set_name]}
    out: list[PromptRecord] = []
    for kind, split in src.families[set_name]:
        if polyrefuse.has_file(src.root, kind, split, lang, cfg=run.cfg):
            out += [r for r in polyrefuse.load_polyrefuse(src.root, kind, split, lang, cfg=run.cfg)
                    if r.pid not in excluded]
    return out


def _build_prompts(run: _Run, src: _Sources, set_name: str, lang: str) -> None:
    english = _to_translate(run, src, set_name, lang)
    if english and (run.skip_translation or run.blocked):
        run.skip(set_name, lang, "translation blocked earlier in this run" if run.blocked
                 else "needs translation (--skip-translation)")
        return
    f = _File(set_name, lang, exclusions=list(src.exclusions.get(set_name, ())))
    built = _from_files(run, src, set_name, lang)
    failures: list[TranslationFailure] = []
    by_pid = {rec.pid: rec for rec in english}
    try:
        if english:
            done, failures = _translate(run, english, lang, "google", cache_only=run.cache_only,
                                        retranslate=run.retranslate_failed)
            built += done
        f.records = _apply_verdicts(run, f, built)
        still_open = _resolve_failures(run, f, failures, by_pid)
        _fallback(run, f, still_open, by_pid)
    except TranslationBlocked as e:
        run.blocked = str(e)
        log.error("%s_%s: translation blocked after %d items (%s). No more translation in this run: every "
                  "file that still needs one is skipped; rerun later to resume from the cache",
                  set_name, lang, e.done, e)
        run.skip(set_name, lang, "translation blocked")
        return
    for (v_lang, pid), v in run.verdicts.items():
        if v_lang == lang and v.get("set") == set_name and pid not in f.touched:
            _ignore(f, pid, v, "the item is not in this build")
    _write_prompts(run, src, f)


def _translate(run: _Run, records: Sequence[PromptRecord], lang: str, engine: str, *, cache_only: bool,
               retranslate: bool) -> tuple[list[PromptRecord], list[TranslationFailure]]:
    """translate_records through the context's one backend: (successes, failures)."""
    try:
        done = translate.translate_records(records, lang, engine=engine, lid=run.ctx.load_lid(),
                                           backend=run.ctx.backend(engine), cfg=run.cfg,
                                           cache_only=cache_only, retranslate_failed=retranslate)
        return done, []
    except TranslationError as e:
        return list(e.records), list(e.failures)


def _verdict(run: _Run, f: _File, pid: str) -> dict[str, Any] | None:
    v = run.verdicts.get((f.lang, pid))
    if v is not None:
        f.touched.add(pid)
    return v


def _stale(v: Mapping[str, Any], english: str | None, current: str) -> str | None:
    """Why a verdict no longer applies (None if it does): every verdict needs the English the reader
    saw; ok and bad also the translation."""
    if english is None or v["source_sha"] != sha256_text(english):
        return "the English source changed after review"
    if v["verdict"] != "fixed" and v["seen_sha"] != sha256_text(current):
        return "the translation changed after review"
    return None


def _ignore(f: _File, pid: str, v: Mapping[str, Any], why: str) -> None:
    log.warning("%s_%s: the %s verdict on %s is not applied: %s", f.set_name, f.lang, v["verdict"], pid, why)
    f.ignored.append({"pid": pid, "verdict": v["verdict"], "why": why})


def _exclude(f: _File, rec: PromptRecord) -> None:
    f.excluded.append(rec.model_copy(update={"extra": {**rec.extra, "excluded": True,
                                                       "excluded_reason": "spotcheck:bad"}}))
    f.exclusions.append({"pid": rec.pid, "reason": "spotcheck:bad"})


def _apply_verdicts(run: _Run, f: _File, records: Iterable[PromptRecord]) -> list[PromptRecord]:
    """The readers' latest verdicts on built records: fixed and ok stay (checked), bad moves to
    f.excluded. A verdict that no longer fits is listed and the record kept as built."""
    out = []
    for rec in records:
        v = _verdict(run, f, rec.pid)
        if v is None:
            out.append(rec)
            continue
        why = "not a machine translation" if rec.translation is None else _stale(v, rec.text_en, rec.text)
        if why or rec.translation is None:
            _ignore(f, rec.pid, v, why or "")
            out.append(rec)
            continue
        verdict = v["verdict"]
        info = rec.translation.model_copy(update={"checked": True, "note": v.get("note")})
        if verdict == "fixed":
            text = clean_text(v["corrected"])
            rec = rec.model_copy(update={"text": text,
                                         "translation": info.model_copy(update={"edited": text != rec.text})})
        else:
            rec = rec.model_copy(update={"translation": info})
        f.applied[verdict] += 1
        if verdict == "bad":
            _exclude(f, rec)
        else:
            out.append(rec)
    return out


def _from_english(en: PromptRecord, lang: str, text: str, info: TranslationInfo) -> PromptRecord:
    """The record translate_records makes from ``en``, with ``text`` and ``info``."""
    fields = en.model_dump()
    fields.update(lang=lang, text=text, text_en=en.text, lid=None, translation=info,
                  extra=copy.deepcopy(en.extra))
    return PromptRecord.model_validate(fields)


def _resolve_failures(run: _Run, f: _File, failures: Sequence[TranslationFailure],
                      english: Mapping[str, PromptRecord]) -> list[TranslationFailure]:
    """List every failure; build the record of each one a reader resolved (fixed: the reader's text;
    ok: the engine's output; bad: to f.excluded). Returns the failures still open."""
    still_open = []
    cache: TranslationCache | None = None
    for fail in sorted(failures, key=lambda x: x.pid or ""):
        row: dict[str, Any] = {"pid": fail.pid, "rule": fail.rule, "detail": fail.detail,
                               "engine": fail.engine, "resolved": None}
        f.failures.append(row)
        v = _verdict(run, f, fail.pid)
        if v is None:
            still_open.append(fail)
            continue
        en = english[fail.pid]
        output = clean_text(fail.output) if fail.output else ""
        why = _stale(v, en.text, output)
        if why is None and v["verdict"] == "ok" and not output:
            why = "no engine output to accept"
        if why:
            _ignore(f, fail.pid, v, why)
            still_open.append(fail)
            continue
        cache = cache if cache is not None else TranslationCache()
        hit = cache.get(fail.engine, f.lang, en.text)
        text = clean_text(v["corrected"]) if v["verdict"] == "fixed" else output
        info = TranslationInfo(engine=fail.engine, source_lang="en", date=hit["date"] if hit else None,
                               checked=True, edited=text != output, note=v.get("note"))
        rec = _from_english(en, f.lang, text, info)
        row["resolved"] = v["verdict"]
        f.applied[v["verdict"]] += 1
        if v["verdict"] == "bad":
            _exclude(f, rec)
        else:
            f.records.append(rec)
    return still_open


def _fallback(run: _Run, f: _File, still_open: Sequence[TranslationFailure],
              english: Mapping[str, PromptRecord]) -> None:
    """NLLB for the hard_benign items Google failed: a cached NLLB translation that passes the checks
    is always used; with fallback_engine "nllb" the missing ones are made (and cached)."""
    if f.set_name != "hard_benign" or not still_open:
        return
    name = translate.ENGINE_NAMES["nllb"]
    make = run.fallback_engine == "nllb" and not run.cache_only
    todo = [english[x.pid] for x in still_open]
    if not make:
        cache = TranslationCache()
        todo = [rec for rec in todo if cache.get(name, f.lang, rec.text) is not None]
        if not todo:
            return
    done, failed = _translate(run, todo, f.lang, "nllb", cache_only=not make, retranslate=False)
    resolved = {rec.pid for rec in done}
    for row in f.failures:
        if row["pid"] in resolved and row["resolved"] is None:
            row["resolved"] = f"fallback:{name}"
    f.fallback = {"engine": name, "resolved": sorted(resolved),
                  "failures": [{"pid": x.pid, "rule": x.rule, "detail": x.detail}
                               for x in sorted(failed, key=lambda x: x.pid or "")]}
    log.info("%s_%s: NLLB fallback resolved %d of %d failed items", f.set_name, f.lang, len(resolved),
             len(todo))
    f.records += _apply_verdicts(run, f, done)


def _stamp(set_name: str, rec: PromptRecord, mapping: Mapping[str, str] | None) -> PromptRecord:
    if mapping is not None and _in_scope(set_name, rec.pid):
        if rec.pid not in mapping:
            raise ValueError(f"{set_name}: {rec.pid} has no split in data/splits/{set_name}.json")
        return rec.model_copy(update={"split": mapping[rec.pid]})
    return rec


def _same_text(records: Sequence[PromptRecord]) -> tuple[list[PromptRecord], list[dict[str, str]],
                                                         list[list[dict[str, str]]]]:
    """Texts one file holds more than once (dup_key), after the split stamp. A train/val record whose
    text a dev/final record also has is dropped (``same_text_as:<the smallest such pid>``). Records of
    one text in different split-file splits (fit, dev, final) cannot be regrouped once the split is
    frozen: they come back as cross-split groups [{"pid", "split"}, ...]. Returns (the records kept,
    the exclusions, the cross-split groups)."""
    by_text: dict[str, list[PromptRecord]] = defaultdict(list)
    for rec in records:
        by_text[dup_key(rec.text)].append(rec)
    dropped: dict[str, str] = {}
    cross: list[list[dict[str, str]]] = []
    for recs in by_text.values():
        if len(recs) < 2:
            continue
        held = sorted(r.pid for r in recs if r.split in ("dev", "final"))
        if held:
            dropped.update((r.pid, held[0]) for r in recs if r.split in ("train", "val"))
        framed = sorted((r for r in recs if r.split not in ("train", "val")), key=_pid)
        if len({r.split for r in framed}) > 1:
            cross.append([{"pid": r.pid, "split": r.split} for r in framed])
    same = [{"pid": pid, "reason": f"same_text_as:{first}"} for pid, first in sorted(dropped.items())]
    return [r for r in records if r.pid not in dropped], same, sorted(cross, key=lambda g: g[0]["pid"])


def _write_prompts(run: _Run, src: _Sources, f: _File) -> None:
    """Stamp splits, drop the train/val records whose text a dev/final record shares and list the
    texts that straddle splits (``_same_text``: it matters for our own translations, hi, zh harmless
    and hard_benign; PolyRefuse's languages are linked before), run GlotLID, write the file, its
    excluded file and its report."""
    name = f"{f.set_name}_{f.lang}"
    mapping = src.splits.get(f.set_name)
    kept, same, cross = _same_text([_stamp(f.set_name, rec, mapping) for rec in sorted(f.records, key=_pid)])
    if same:
        log.info("%s: dropped %d train/val records whose text a dev/final record has: %s", name, len(same),
                 ", ".join(f"{e['pid']} ({e['reason']})" for e in same))
    if cross:
        log.warning("%s: %d texts repeat across splits of the frozen %s split, which cannot regroup them: %s",
                    name, len(cross), f.set_name,
                    "; ".join(" ".join(f"{x['pid']} {x['split']}" for x in group) for group in cross))
    lid = run.ctx.load_lid()
    records, lid_failures = [], []
    for rec in kept:
        info = lid_info(lid, rec.text)
        if not lid.is_lang(rec.text, f.lang):
            lid_failures.append({"pid": rec.pid, "label": info.label, "prob": info.prob})
        records.append(rec.model_copy(update={"lid": info}))
    excluded = [_stamp(f.set_name, rec, mapping) for rec in sorted(f.excluded, key=_pid)]
    pids = [rec.pid for rec in records + excluded]
    if len(set(pids)) != len(pids):
        repeated = sorted(p for p, n in Counter(pids).items() if n > 1)
        raise ValueError(f"{f.set_name}_{f.lang}: repeated pids {repeated[:5]}")
    report = {
        "set": f.set_name, "lang": f.lang, "records": len(records),
        "by_split": _count(rec.split for rec in records),
        "engines": _count(rec.translation.engine if rec.translation else "original" for rec in records),
        "lid_failures": lid_failures,
        "translation_failures": sorted(f.failures, key=lambda x: (x["pid"], x["engine"])),
        "exclusions": sorted([*f.exclusions, *same], key=lambda x: x["pid"]),
        "cross_split_duplicates": cross,
        "spotchecks": {"applied": {v: f.applied[v] for v in VERDICTS},
                       "checked": sum(bool(r.translation and r.translation.checked) for r in records),
                       "edited": sum(bool(r.translation and r.translation.edited) for r in records),
                       "ignored": sorted(f.ignored, key=lambda x: x["pid"])},
    }
    if f.fallback is not None:
        report["fallback"] = f.fallback
    _write_file(run, f.set_name, f.lang, [rec.model_dump(mode="json") for rec in records],
                [rec.model_dump(mode="json") for rec in excluded], report)
    run.files[name]["cross_split_duplicates"] = len(cross)


def _build_belebele(run: _Run, src: _Sources, lang: str) -> None:
    items = (src.belebele_en if lang == "en"
             else sources.load_belebele(lang, cfg=run.cfg, hub=run.ctx.load_hub()))
    mapping = src.splits["belebele"]
    lid = run.ctx.load_lid()
    rows, lid_failures = [], []
    for item in sorted(items, key=lambda d: d["pid"]):
        if item["pid"] not in mapping:
            raise ValueError(f"belebele: {item['pid']} has no split in data/splits/belebele.json")
        row = {**item, "split": mapping[item["pid"]]}
        if not lid.is_lang(row["question"], lang):
            info = lid_info(lid, row["question"])
            lid_failures.append({"pid": row["pid"], "label": info.label, "prob": info.prob})
        rows.append(row)
    report = {"set": "belebele", "lang": lang, "records": len(rows),
              "by_split": _count(r["split"] for r in rows),
              "engines": {}, "lid_failures": lid_failures, "translation_failures": [], "exclusions": [],
              "spotchecks": {"applied": dict.fromkeys(VERDICTS, 0), "checked": 0, "edited": 0, "ignored": []}}
    _write_file(run, "belebele", lang, rows, [], report)


def _write_file(run: _Run, set_name: str, lang: str, rows: list[dict], excluded: list[dict],
                report: dict[str, Any]) -> None:
    """The prompt file, then its excluded file (removed when empty), then its report (with the
    file's sha256, so a report left over from another build shows as stale)."""
    name = f"{set_name}_{lang}"
    path = prompts_dir() / f"{name}.jsonl"
    write_jsonl(path, rows)
    excluded_path = prompts_dir() / "excluded" / path.name
    if excluded:
        write_jsonl(excluded_path, excluded)
    else:
        excluded_path.unlink(missing_ok=True)
    report = {**report, "file": path.name, "sha256": sha256_file(path)}
    _write_json(_report_path(set_name, lang), report)
    still_open = [x for x in report["translation_failures"] if x["resolved"] is None]
    run.files[name] = {"records": len(rows), "by_split": report["by_split"],
                       "lid_failures": len(report["lid_failures"]),
                       "translation_failures": len(report["translation_failures"]),
                       "open_failures": len(still_open),
                       "errors": sum(x["rule"] == "error" for x in still_open),
                       "exclusions": len(report["exclusions"])}
    log.info("%s: %d records %s, %d lid failures, %d translation failures (%d open), %d excluded", name,
             len(rows), report["by_split"], len(report["lid_failures"]), len(report["translation_failures"]),
             len(still_open), len(report["exclusions"]))


def _build_sft(run: _Run, src: _Sources) -> dict[str, Any]:
    """The CuratorKIT candidates from the training pools the held-out set was removed from. _prepare
    froze or checked the utility split; every training pool must be disjoint from its pids before
    anything is written."""
    if src.pools is None or src.train_pools is None:
        raise RuntimeError("the SFT pools were not prepared")
    held = set(src.splits["utility"])
    for name, (rows, _) in src.train_pools.items():
        splits.check_disjoint(held, ("utl-" + r.row_id() for r in rows),
                              what=f"the frozen utility split and the {name} training pool")
    report = sft.build_candidates(src.train_pools, pool_counts={k: c for k, (_, c) in src.pools.items()},
                                  cfg=run.cfg, train_cfg=run.ctx.train_cfg, curate=run.ctx.curate)
    return {"decision": report["decision"],
            "variants": {k: v["rows"] for k, v in sorted(report["variants"].items())},
            "report": "sft/report.json"}


def _summary(run: _Run, src: _Sources, options: Mapping[str, Any], sft_summary: dict[str, Any] | None,
             run_id: str) -> dict[str, Any]:
    files = dict(sorted(run.files.items()))
    incomplete = [name for name, entry in files.items() if entry["errors"]]
    totals = {key: sum(entry[key] for entry in files.values())
              for key in ("records", "lid_failures", "translation_failures", "open_failures", "exclusions")}
    return {
        "run_id": run_id, "options": dict(options),
        "complete": run.blocked is None and not run.skipped and not incomplete,
        "blocked": run.blocked is not None, "blocked_reason": run.blocked,
        "skipped": run.skipped, "incomplete": incomplete,
        "files": files, "totals": {"files": len(files), **totals},
        "splits": run.splits, "sft": sft_summary, "alignment": src.alignment,
        "sources": _pins(run.cfg), "timings": run.timings,
    }


def _pins(cfg: Mapping[str, Any]) -> dict[str, Any]:
    s = cfg["sources"]
    pins: dict[str, Any] = {
        "polyrefuse": {"repo": s["polyrefuse"]["repo"], "commit": s["polyrefuse"]["commit"]},
        "langid": {"id": cfg["langid"]["id"], "revision": cfg["langid"].get("revision")}}
    for name, spec in (("orbench", s["orbench"]), ("aya_redteaming", s["aya_redteaming"]),
                       ("belebele", s["belebele"]), ("aya_dataset", s["sft"]["aya"]),
                       ("indicalign", s["sft"]["indicalign"]), ("dolly", s["sft"]["dolly"])):
        pins[name] = {"id": spec["id"], "revision": spec.get("revision")}
    return pins


# ---------------------------------------------------------------------------- files


def _report_path(set_name: str, lang: str) -> Path:
    return prompts_dir() / "reports" / f"{set_name}_{lang}.json"


def _read_report(set_name: str, lang: str) -> dict[str, Any] | None:
    path = _report_path(set_name, lang)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, obj: Any) -> None:
    """Atomically (temp file + rename): indent 1, sorted keys, unicode kept, LF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_prompts(set_name: str, lang: str, split: str | None = None) -> list:
    """Read data/prompts/{set}_{lang}.jsonl (PromptRecords; belebele: dicts), optionally one split.
    Excluded records (prompts/excluded/) are never returned."""
    if set_name not in SETS or set_name == "sft":
        raise ValueError(f"unknown prompt set {set_name!r}; choose from {[s for s in SETS if s != 'sft']}")
    if split is not None and split not in _SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; choose from {list(_SPLIT_NAMES)}")
    path = prompts_dir() / f"{set_name}_{lang}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: build it with `lift data build --only {set_name} "
                                f"--langs {lang}`")
    rows = list(read_jsonl(path))
    items: list = rows if set_name == "belebele" else [PromptRecord.model_validate(row) for row in rows]
    if split is not None:
        items = [x for x in items if (x["split"] if set_name == "belebele" else x.split) == split]
    return items


# ---------------------------------------------------------------------------- spot checks


def _verdicts_path() -> Path:
    return interim_dir() / "translations" / "spotchecks.jsonl"


def _read_verdicts() -> dict[tuple[str, str], dict[str, Any]]:
    """The latest verdict row per (lang, pid). A line a killed run left half written is skipped with a
    warning (apply that sheet again); a row that parses but is not a verdict raises."""
    path = _verdicts_path()
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.warning("%s:%d: skipping a half-written line", path.name, lineno)
            continue
        missing = [k for k in _VERDICT_FIELDS if not isinstance(row, dict) or k not in row]
        if missing or row["verdict"] not in VERDICTS:
            raise ValueError(f"{path}:{lineno}: not a verdict row (missing {missing} or unknown verdict)")
        out[(row["lang"], row["pid"])] = row
    return out


def _append_verdicts(rows: Sequence[dict[str, Any]]) -> None:
    path = _verdicts_path()
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with open(path, "a", encoding="utf-8", newline="\n") as f:       # close off a half-written line
            f.write("\n")
    append_jsonl(path, rows)


def write_spotcheck_sheets(*, langs: Iterable[str] | None = None, n: int | None = None,
                           overwrite: bool = False, ctx: BuildContext | None = None,
                           out_dir: Path | None = None) -> list[Path]:
    """Write the native readers' sheets into ``out_dir`` (default SHEETS_DIR, validation/sheets).

    Per language (default languages.primary + transfer_validated, minus en): spotcheck_{lang}.csv
    with, for each translated set, the ``n`` (default validation.translation_spot_check) unchecked
    dev/final records that are machine translations (ours and PolyRefuse's) with the smallest
    sha256(f"{seed}:spotcheck:{lang}:{set}:{pid}"); and spotcheck_{lang}_failures.csv with every
    open translation failure (translation = the engine's output from the cache, or empty) and every
    unchecked machine translation GlotLID rejected, the note prefilled with the auto-check result.
    Columns SHEET_COLUMNS; item_id "{lang}-{pid}". Refuses to overwrite any sheet unless
    ``overwrite`` (FileExistsError, before writing anything). Returns the paths written.
    """
    cfg = ctx.cfg if ctx is not None else config.load("project")
    lang_cfg = cfg["languages"]
    if langs is None:
        langs = [lang for lang in dict.fromkeys([*lang_cfg["primary"], *lang_cfg["transfer_validated"]])
                 if lang != "en"]
    langs = list(dict.fromkeys(langs))
    wrong = [lang for lang in langs if lang == "en" or lang not in lang_cfg["all"]]
    if wrong:
        raise ValueError(f"no machine translations to check in {wrong}")
    n = int(cfg["validation"]["translation_spot_check"]) if n is None else int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    out_dir = Path(out_dir) if out_dir is not None else SHEETS_DIR
    seed = int(cfg["split_seed"])
    sheets: dict[Path, list[list[str]]] = {}
    for lang in langs:
        sample: list[list[str]] = []
        failed: list[list[str]] = []
        for set_name in TRANSLATED_SETS:
            records = load_prompts(set_name, lang)
            tag = f"spotcheck:{lang}:{set_name}"
            candidates = [r for r in records if r.translation is not None and not r.translation.checked
                          and r.split in ("dev", "final")]
            ranked = sorted(candidates, key=lambda r, tag=tag: sha256_text(f"{seed}:{tag}:{r.pid}"))
            sample += [[f"{lang}-{r.pid}", set_name, r.text_en or "", r.text, "", "", ""]
                       for r in sorted(ranked[:n], key=_pid)]
            failed += _failure_rows(set_name, lang, records)
        sheets[out_dir / f"spotcheck_{lang}.csv"] = sample
        sheets[out_dir / f"spotcheck_{lang}_failures.csv"] = failed
        log.info("spot-check sheets %s: %d sampled items, %d failures", lang, len(sample), len(failed))
    existing = [p.name for p in sheets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"spot-check sheets already exist in {out_dir}: {existing}; readers may be "
                              "filling them. Pass overwrite (--overwrite) to replace them")
    for path, rows in sheets.items():
        _write_sheet(path, rows)
    return list(sheets)


def _failure_rows(set_name: str, lang: str, records: Sequence[PromptRecord]) -> list[list[str]]:
    """The failures sheet rows of one (set, lang): open translation failures (absent from the file)
    and unchecked machine translations GlotLID rejected (in the file)."""
    report = _read_report(set_name, lang)
    if report is None:
        raise FileNotFoundError(f"{_report_path(set_name, lang)} does not exist: run `lift data build`")
    by_pid = {r.pid: r for r in records}
    rows = []
    still_open = [x for x in report["translation_failures"]
                  if x["resolved"] is None and x["pid"] not in by_pid]
    if still_open:
        english = {r.pid: r.text for r in load_prompts(set_name, "en")}
        cache = TranslationCache()
        for x in still_open:
            hit = cache.get(x["engine"], lang, english[x["pid"]])
            shown = clean_text(hit["output"]) if hit else ""
            note = f"auto-check failed: {x['rule']} {x['detail']}".strip()
            rows.append([f"{lang}-{x['pid']}", set_name, english[x["pid"]], shown, "", "", note])
    for x in report["lid_failures"]:
        rec = by_pid.get(x["pid"])
        if rec is not None and rec.translation is not None and not rec.translation.checked:
            note = f"auto-check failed: wrong_lang top={x['label']} {x['prob']}"
            rows.append([f"{lang}-{rec.pid}", set_name, rec.text_en or "", rec.text, "", "", note])
    return sorted(rows, key=lambda row: row[0])


def _write_sheet(path: Path, rows: Sequence[Sequence[str]]) -> None:
    """UTF-8 with a BOM (spreadsheets then read Devanagari and Han right), LF, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(SHEET_COLUMNS)
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_sheet(path: Path) -> tuple[list[tuple[int, list[str]]], list[str]]:
    """(spreadsheet row number, the 7 cells) of every non-blank row, and the row problems. SheetError
    if the header is not exactly SHEET_COLUMNS (trailing empty cells aside)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        table = list(csv.reader(f))

    def trim(cells: list[str]) -> list[str]:
        while cells and not cells[-1].strip():
            cells = cells[:-1]
        return cells

    header = [cell.strip() for cell in trim(table[0])] if table else []
    if tuple(header) != SHEET_COLUMNS:
        raise SheetError(f"{path.name}: the header must be exactly {','.join(SHEET_COLUMNS)}")
    rows, problems = [], []
    for number, cells in enumerate(table[1:], start=2):
        cells = trim(cells)
        if not cells:
            continue
        if len(cells) > len(SHEET_COLUMNS):
            problems.append(f"row {number}: {len(cells)} cells, expected {len(SHEET_COLUMNS)}")
            continue
        rows.append((number, cells + [""] * (len(SHEET_COLUMNS) - len(cells))))
    return rows, problems


def _cell_text(cell: str) -> str:
    """A reader's text: spreadsheet line breaks (CRLF, CR) as LF, then clean_text."""
    return clean_text(cell.replace("\r\n", "\n").replace("\r", "\n"))


class _Items:
    """The current build's spot-checkable items per (set, lang): pid -> (English, translation as
    built, is a machine translation). Records in the prompt file, then the excluded file, then open
    translation failures (English from the en file, translation = the cached engine output)."""

    def __init__(self) -> None:
        self._files: dict[tuple[str, str], dict[str, tuple[str, str, bool]]] = {}
        self._cache: TranslationCache | None = None

    def find(self, set_name: str, lang: str, pid: str) -> tuple[str, str, bool] | None:
        if (set_name, lang) not in self._files:
            self._files[(set_name, lang)] = self._load(set_name, lang)
        return self._files[(set_name, lang)].get(pid)

    def _load(self, set_name: str, lang: str) -> dict[str, tuple[str, str, bool]]:
        items: dict[str, tuple[str, str, bool]] = {}
        name = f"{set_name}_{lang}.jsonl"
        for path in (prompts_dir() / name, prompts_dir() / "excluded" / name):
            if path.exists():
                for row in read_jsonl(path):
                    rec = PromptRecord.model_validate(row)
                    items.setdefault(rec.pid, (rec.text_en or "", rec.text, rec.translation is not None))
        report = _read_report(set_name, lang)
        failed = [x for x in (report or {}).get("translation_failures", []) if x["pid"] not in items]
        if failed:
            english = {rec.pid: rec.text for rec in load_prompts(set_name, "en")}
            self._cache = self._cache if self._cache is not None else TranslationCache()
            for x in failed:
                hit = self._cache.get(x["engine"], lang, english[x["pid"]])
                items[x["pid"]] = (english[x["pid"]], clean_text(hit["output"]) if hit else "", True)
        return items


def apply_spotchecks(path: str | Path, *, ctx: BuildContext | None = None,
                     dry_run: bool = False) -> dict[str, Any]:
    """Apply a returned spot-check sheet.

    Every row is checked against the current build first, and every problem is raised together as one
    SheetError before anything is written: item ids that do not parse or name no item of that set and
    language (prompt file, excluded file or translation failure), a set the item is not in, a
    verdict other than ok / fixed / bad (empty = not reviewed: counted, ignored), fixed without a
    corrected text or corrected filled for another verdict, ok on an item with no engine output,
    and english or translation cells that no longer match the build (a stale or re-saved sheet;
    compared by dup_key). Then the verdicts are appended to interim/translations/spotchecks.jsonl
    (source_sha and seen_sha are the sha256 of the item's English and of its translation as built)
    and the files they touch are rebuilt from the cache, splits untouched. ``dry_run`` only checks.
    Returns per file: ok (count), fixed and bad (pids), unreviewed (count).
    """
    ctx = ctx if ctx is not None else BuildContext.default()
    cfg = ctx.cfg
    path = Path(path)
    rows, problems = _read_sheet(path)
    langs = [lang for lang in cfg["languages"]["all"] if lang != "en"]
    items = _Items()
    entries: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any]] = {}
    seen: dict[str, int] = {}
    today = date.today().isoformat()
    for number, cells in rows:
        item_id, set_name, english, shown, verdict, corrected, note = (cell.strip() for cell in cells)
        valid = bool(_ITEM_ID.fullmatch(item_id))
        where = f"row {number}" + (f" ({item_id})" if valid else "")
        lang, _, pid = item_id.partition("-")
        if not valid or lang not in langs:
            problems.append(f"{where}: item_id is not <lang>-<pid> with a language in {langs}")
            continue
        if item_id in seen:
            problems.append(f"{where}: the item is also on row {seen[item_id]}")
            continue
        seen[item_id] = number
        if set_name not in TRANSLATED_SETS:
            problems.append(f"{where}: set must be one of {', '.join(TRANSLATED_SETS)}")
            continue
        found = items.find(set_name, lang, pid)
        if found is None:
            other = [s for s in TRANSLATED_SETS if s != set_name and items.find(s, lang, pid) is not None]
            problems.append(f"{where}: " + (f"the item is in set {other[0]}, not {set_name}" if other
                                            else f"no such item in the current {set_name}_{lang} build"))
            continue
        tally = files.setdefault(f"{set_name}_{lang}", {"ok": 0, "fixed": [], "bad": [], "unreviewed": 0})
        verdict = verdict.lower()
        if not verdict:
            tally["unreviewed"] += 1
            continue
        if verdict not in VERDICTS:
            problems.append(f"{where}: verdict must be ok, fixed or bad (empty: not reviewed)")
            continue
        english_text, current, machine = found
        corrected, note = _cell_text(corrected), _cell_text(note)
        stale = "differs from the current build (a stale or re-saved sheet)"
        found_problems = [
            (not machine, "not a machine translation"),
            (verdict == "fixed" and not corrected, "fixed needs the corrected translation"),
            (verdict != "fixed" and bool(corrected), f"corrected is filled but the verdict is {verdict}"),
            (verdict == "ok" and not current, "there is no engine output to accept (use fixed or bad)"),
            (dup_key(english) != dup_key(english_text), f"english {stale}"),
            (dup_key(shown) != dup_key(current), f"translation {stale}"),
        ]
        problems += [f"{where}: {what}" for bad, what in found_problems if bad]
        entries.append({"lang": lang, "pid": pid, "set": set_name, "verdict": verdict,
                        "corrected": corrected or None, "note": note or None,
                        "source_sha": sha256_text(english_text), "seen_sha": sha256_text(current),
                        "sheet": path.name, "applied": today})
        if verdict == "ok":
            tally["ok"] += 1
        else:
            tally[verdict].append(pid)
    if problems:
        listed = problems[:50] + ([f"... and {len(problems) - 50} more"] if len(problems) > 50 else [])
        raise SheetError(f"{path.name}: {len(problems)} problem(s), nothing applied:\n  "
                         + "\n  ".join(listed), problems)
    summary: dict[str, Any] = {"sheet": path.name, "rows": len(rows), "reviewed": len(entries),
                               "unreviewed": sum(t["unreviewed"] for t in files.values()), "dry_run": dry_run,
                               "files": dict(sorted(files.items())), "rebuilt": [], "not_applied": []}
    if dry_run or not entries:
        return summary
    _append_verdicts(entries)
    order = {s: i for i, s in enumerate(SETS)}
    pairs = sorted({(e["set"], e["lang"]) for e in entries},
                   key=lambda p: (order[p[0]], cfg["languages"]["all"].index(p[1])))
    run = _Run(ctx, skip_translation=False, cache_only=True, retranslate_failed=False, fallback_engine=None,
               create_splits=False)
    src = _prepare(run, {s for s, _ in pairs})
    for set_name, lang in pairs:
        _build_file(run, src, set_name, lang)
    ours = {(e["lang"], e["pid"]) for e in entries}
    for set_name, lang in pairs:
        report = _read_report(set_name, lang) or {}
        ignored = report.get("spotchecks", {}).get("ignored", [])
        summary["not_applied"] += [{**x, "set": set_name, "lang": lang} for x in ignored
                                   if (lang, x["pid"]) in ours]
    summary["rebuilt"] = [f"{s}_{lang}" for s, lang in pairs]
    summary["files_rebuilt"] = {f"{s}_{lang}": run.files.get(f"{s}_{lang}") for s, lang in pairs}
    log.info("applied %d verdicts from %s to %s (%d unreviewed rows)", len(entries), path.name,
             ", ".join(summary["rebuilt"]), summary["unreviewed"])
    return summary


# ---------------------------------------------------------------------------- stats


def stats(*, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Counts from the prompt files (records per set, language and split) and their reports (GlotLID
    rejections, translation failures by rule, exclusions by reason, spot-check counts), the SFT row
    decision (sft/report.json) and the last build's state. Counts and pids only."""
    cfg = cfg if cfg is not None else config.load("project")
    files: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for set_name in SETS:
        if set_name == "sft":
            continue
        for lang in _set_langs(set_name, cfg):
            name = f"{set_name}_{lang}"
            path = prompts_dir() / f"{name}.jsonl"
            if not path.exists():
                missing.append(name)
                continue
            rows = list(read_jsonl(path))
            entry: dict[str, Any] = {"records": len(rows), "by_split": _count(row["split"] for row in rows)}
            report = _read_report(set_name, lang)
            if report is None:
                entry["report"] = "missing"
            else:
                failures = report["translation_failures"]
                entry.update({
                    "report": "ok" if report.get("sha256") == sha256_file(path) else "stale",
                    "lid_failures": [x["pid"] for x in report["lid_failures"]],
                    "translation_failures": _count(x["rule"] for x in failures),
                    "open_failures": [x["pid"] for x in failures if x["resolved"] is None],
                    "exclusions": _count(x["reason"].split(":", 1)[0] for x in report["exclusions"]),
                    "excluded": [x["pid"] for x in report["exclusions"]],
                    "checked": report["spotchecks"]["checked"], "edited": report["spotchecks"]["edited"],
                })
            files[name] = entry
    out: dict[str, Any] = {"files": files, "missing": missing, "sft": None, "build": None}
    sft_report = sft_dir() / "report.json"
    if sft_report.exists():
        report = json.loads(sft_report.read_text(encoding="utf-8"))
        out["sft"] = {"decision": report["decision"],
                      "variants": {k: {"rows": v["rows"], "curated": v["counts"]["curated"],
                                       "train_pool": v["counts"]["train_pool"]}
                                   for k, v in sorted(report["variants"].items())}}
    build_json = prompts_dir() / "reports" / "build.json"
    if build_json.exists():
        last = json.loads(build_json.read_text(encoding="utf-8"))
        out["build"] = {k: last.get(k) for k in ("run_id", "complete", "blocked", "skipped", "incomplete")}
    return out


def format_stats(st: Mapping[str, Any]) -> str:
    """``stats()`` as a table plus notes, for the terminal."""
    splits_seen = [s for s in _SPLIT_NAMES if any(s in e["by_split"] for e in st["files"].values())]
    head = ["file", "records", *splits_seen, "lid", "failed", "open", "excluded", "checked", "edited"]
    table = [head]
    for name, e in st["files"].items():
        failed = sum(e.get("translation_failures", {}).values())
        counts = (len(e.get("lid_failures", [])), failed, len(e.get("open_failures", [])),
                  len(e.get("excluded", [])), e.get("checked", 0), e.get("edited", 0))
        table.append([name, str(e["records"]), *(str(e["by_split"].get(s, "-")) for s in splits_seen),
                      *(str(x) for x in counts)])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    lines = ["  ".join(cell.ljust(w) if i == 0 else cell.rjust(w)
                       for i, (cell, w) in enumerate(zip(row, widths, strict=True))) for row in table]
    lines.append("lid = GlotLID rejections (kept, listed); failed = translation failures (open = still "
                 "absent from the file); excluded = duplicates and spot-check bad")
    for name, e in st["files"].items():
        notes = []
        if e.get("translation_failures"):
            notes.append("failures " + ", ".join(f"{k} {v}" for k, v in e["translation_failures"].items()))
        if e.get("exclusions"):
            notes.append("exclusions " + ", ".join(f"{k} {v}" for k, v in e["exclusions"].items()))
        if e.get("report") != "ok":
            notes.append(f"report {e.get('report')}")
        if notes:
            lines.append(f"  {name}: " + "; ".join(notes))
    if st["missing"]:
        lines.append("missing files: " + ", ".join(st["missing"]))
    if st.get("sft"):
        d = st["sft"]["decision"]
        lines.append(f"sft: N = {d['rows']} rows per candidate (target {d['rows_target']}"
                     + (f", limited by {d['limited_by']})" if d.get("limited_by") else ")"))
        for k, v in st["sft"]["variants"].items():
            lines.append(f"  {k}: {v['rows']} of {v['curated']} curated (training pool {v['train_pool']})")
    else:
        lines.append("sft: not built")
    if st.get("build"):
        b = st["build"]
        state = "complete" if b["complete"] else "INCOMPLETE"
        state += " (translation blocked)" if b["blocked"] else ""
        lines.append(f"last build {b['run_id']}: {state}")
        if b.get("skipped"):
            lines.append("  skipped: " + ", ".join(f"{x['set']}_{x['lang']} ({x['reason']})"
                                                   for x in b["skipped"]))
        if b.get("incomplete"):
            lines.append("  missing items a rerun can translate: " + ", ".join(b["incomplete"]))
    return "\n".join(lines)
