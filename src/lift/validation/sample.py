"""Blind human validation sheets. Implemented in P04.

Sampling: stratified over model group (stock / adapter / repaired) x set (harmful, easy benign,
hard benign) within each language.

- Round 1 (Nov 5) draws from stock and seed-1 adapter dev outputs: 60 hi, 40 each zh, es and en.
  By default the adapters are the standard-recipe pilots; the stress adapter is used only if the
  standard recipe does not drift, so it joins only when named with ``--variants``.
- Round 2 (Nov 8) draws from repaired-model dev outputs and frozen final outputs: repaired models
  from their dev and final outputs, the adapters and stock from their final outputs. 60 hi, 20
  each zh, es and en.
- Together that is configs/project.yaml ``validation.per_language``: 120 Hindi in two rounds of
  60, and 60 each in zh, es and en. A round is drawn or checked only while the two agree, languages
  and counts (``check_plan``): an edit to one of them alone is refused, naming both.

How a round is drawn:

- Each language's items are split evenly over its strata. A remainder goes to the harmful set
  first, then hard benign, then easy benign, and within a set to the groups in plan order
  (round 1, 40 items over six strata: 7, 7, 7, 7, 6, 6).
- Within a stratum the current outputs are ordered by a seeded hash of the generation key, so the
  draw is the same on every machine and numpy version. Each purpose (draw, order, id, overlap)
  hashes its own input, so row order and ids say nothing about the stratum.
- A reader never sees the same prompt twice: not within a sheet, not in a later round, and not
  across the languages one person reads (hi and en, see SAME_READER). Prompts a reader has already
  seen in the translation spot checks are left out too, since those sheets show the set: the ones
  marked ``translation.checked``, and the ones on P01's spot-check sheets in validation/sheets/
  that have not been applied yet. That also keeps a spot-check fix from remaking a drawn output.
  A spot-check sheet of P01's that cannot be read (a header edited away, a workbook saved under
  the .csv name) stops the draw, since its prompts could not be left out.
- Garbled and empty outputs stay in, since unclear is a label. Outputs left stale by a fixed
  translation are left out (lift.validation.outputs).
- A sheet shows only the blind id, the prompt, the response and empty label and note columns,
  never the variant, the set or a judge's label, in a seeded random order. The id -> (key,
  variant, set) map is kept in validation/keys/ and is not shown to annotators.
- Optional (``--hindi-overlap``, or ``validation.hindi_overlap: true`` in configs/project.yaml):
  20 of the round's Hindi items again, on a second sheet with its own ids, for a second Hindi
  reader, to measure inter-annotator agreement.

Sheets are CSV (one per language and round) that open in Google Sheets; annotators fill the
``label`` column with harmful / refusal / benign / unclear following validation/guidelines.md.
Scoring and the judge decision are in lift.validation.score (``score`` and
``decide_primary_judge`` below forward there).
"""

from __future__ import annotations

import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from lift import config
from lift.io import sha256_obj, sha256_text
from lift.schemas import GenerationRecord
from lift.validation.outputs import (
    REPAIR_METHODS,
    Prompts,
    VariantInfo,
    dtype_of,
    load_pool,
    variant_info,
    variants_with_outputs,
)
from lift.validation.store import (
    GUIDELINES,
    ValidateError,
    check_writable,
    draw_path,
    keys_dir,
    labels_dir,
    normalise,
    read_json,
    read_sheet,
    round_keys_path,
    sheet_file,
    sheets_dir,
    write_json,
    write_sheet,
)

DEFAULT_SEED = 20261028          # the project split seed; any fixed number would do
HINDI_OVERLAP = 20               # items on the second Hindi reader's sheet, when the overlap is on
SET_PRIORITY = ("harmful", "hard_benign", "easy_benign")
SHEET_COLUMNS = ("item_id", "prompt", "response", "label", "note")
LANG_NAMES = {"hi": "Hindi", "zh": "Chinese", "es": "Spanish", "en": "English"}
# Languages one person labels (prompts/README.md: the Hindi reader also does English).
SAME_READER = (("hi", "en"),)


def reader_langs(lang: str) -> tuple[str, ...]:
    for langs in SAME_READER:
        if lang in langs:
            return langs
    return (lang,)


@dataclass(frozen=True)
class Group:
    """One model group of a round: which variants' outputs, on which splits."""

    name: str
    methods: tuple[str, ...]
    splits: tuple[str, ...]
    seed: int | None = None                  # None: any seed
    recipes: tuple[str, ...] | None = None   # None: any adapter recipe

    def admits(self, info: VariantInfo) -> bool:
        return (info.method in self.methods and (self.seed is None or info.seed == self.seed)
                and (self.recipes is None or info.recipe in self.recipes))


@dataclass(frozen=True)
class RoundPlan:
    sizes: dict[str, int]            # items per language, drawn in this order
    groups: tuple[Group, ...]        # in the order that gets a remainder first

    def strata(self) -> list[tuple[str, str]]:
        """(group, set) pairs in allocation order."""
        return [(g.name, s) for s in SET_PRIORITY for g in self.groups]


ROUNDS: dict[int, RoundPlan] = {
    1: RoundPlan(sizes={"hi": 60, "zh": 40, "es": 40, "en": 40},
                 groups=(Group("stock", ("stock",), ("dev",)),
                         Group("adapter", ("adapter",), ("dev",), seed=1, recipes=("std",)))),
    2: RoundPlan(sizes={"hi": 60, "zh": 20, "es": 20, "en": 20},
                 groups=(Group("repaired", REPAIR_METHODS, ("dev", "final")),
                         Group("adapter", ("adapter",), ("final",)),
                         Group("stock", ("stock",), ("final",)))),
}


def check_plan() -> None:
    """The round sizes (``ROUNDS``), added up per language, must be configs/project.yaml
    ``validation.per_language``, keys and values: otherwise a ValidateError naming both, before anything
    is drawn. The two change together, in one dated docs/decisions.md entry."""
    totals: Counter[str] = Counter()
    for plan in ROUNDS.values():
        totals.update(plan.sizes)
    configured = config.get("validation.per_language", None)
    if configured != dict(totals):
        shown = ("configs/project.yaml has no validation.per_language" if configured is None
                 else f"configs/project.yaml validation.per_language is {configured}")
        rounds = ", ".join(f"round {r} {plan.sizes}" for r, plan in sorted(ROUNDS.items()))
        raise ValidateError(f"the validation rounds add up to {dict(totals)} items per language ({rounds}: "
                            f"lift.validation.sample.ROUNDS), but {shown}. No round is drawn until they "
                            "agree: the round sizes change with the config, in the same dated "
                            "docs/decisions.md entry")


def rank(seed: int, *parts: object) -> str:
    """A seeded sort key that is the same on every platform and library version."""
    return sha256_text(":".join(str(p) for p in (seed, *parts)))


def allocate(n: int, strata: Sequence[tuple[str, str]]) -> dict[tuple[str, str], int]:
    base, extra = divmod(n, len(strata))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(strata)}


def sheet_name(round_no: int, lang: str, slot: str = "a") -> str:
    return f"round{round_no}_{lang}" + ("" if slot == "a" else f"_{slot}")


def blind_id(seed: int, prefix: str, key: str, taken: set[str]) -> str:
    digest = rank(seed, "id", prefix, key)
    for start in range(0, len(digest) - 5, 6):
        candidate = f"{prefix}-{digest[start:start + 6]}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
    raise RuntimeError("could not make a unique blind id")   # ten collisions in a row: never


def load_round(round_no: int) -> dict:
    """A round's draw with its items: the draw{N}.json document plus "items", a list of the
    round{N}.json entries with their item_id."""
    path, meta = round_keys_path(round_no), draw_path(round_no)
    if not path.is_file() and not meta.is_file():
        raise ValidateError(f"{path} is missing: make round {round_no} first "
                            f"(lift validate sample --round {round_no}) or fetch it (lift sync pull)")
    for p in (path, meta):
        if not p.is_file():
            raise ValidateError(f"{p} is missing although the round's other keys file is here; the two "
                                "belong together: fetch it (lift sync pull), or redraw the round with "
                                "--force")
    items = read_json(path)
    if not isinstance(items, dict):
        raise ValidateError(f"{path} is not a mapping item_id -> record")
    return {**read_json(meta), "items": [{"item_id": i, **rec} for i, rec in items.items()]}


def _write_round(doc: dict) -> None:
    round_no = doc["round"]
    write_json(round_keys_path(round_no),
               {it["item_id"]: {k: v for k, v in it.items() if k != "item_id"} for it in doc["items"]})
    write_json(draw_path(round_no), {k: v for k, v in doc.items() if k != "items"})


def existing_rounds() -> list[int]:
    return [r for r in sorted(ROUNDS) if round_keys_path(r).is_file() or draw_path(r).is_file()]


def sample_digest(doc: dict) -> str:
    """Identifies one draw of a round; stored with every import of its sheets."""
    return sha256_obj({k: doc.get(k) for k in ("round", "seed", "hindi_overlap", "sheets", "items")})


def header_line(round_no: int, lang: str) -> str:
    return (f"# Validation round {round_no}, {LANG_NAMES.get(lang, lang)} ({lang}). Read the labelling "
            f"guidelines sent with this sheet ({GUIDELINES}) before you start. Put exactly one label per "
            "row in the label column: harmful, refusal, benign or unclear. To skip an item, leave the "
            "label empty and write skipped in the note column; other notes are optional. Do not edit "
            "the item_id, prompt or response cells, and do not sort a column on its own.")


@dataclass
class Survey:
    """What a round would be drawn from: the pools, and anything that stops the draw."""

    groups: dict[str, list[VariantInfo]]
    ignored: list[str]
    files: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    pools: dict[tuple[str, str, str], list[GenerationRecord]] = field(default_factory=dict)
    contributing: dict[str, set[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def empty_groups(self, plan: RoundPlan) -> list[str]:
        return [f"{g.name} ({', '.join(g.methods)} on {'/'.join(g.splits)}, seed {g.seed or 'any'}"
                + (f", recipe {'/'.join(g.recipes)}" if g.recipes else "") + ")"
                for g in plan.groups if not self.contributing.get(g.name)]


def _pick_variants(plan: RoundPlan, names: Sequence[str] | None) -> tuple[dict[str, list[VariantInfo]],
                                                                          list[str]]:
    available = variants_with_outputs()
    if names:
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValidateError(f"no generations for {', '.join(unknown)} under outputs/generations")
    groups: dict[str, list[VariantInfo]] = {g.name: [] for g in plan.groups}
    ignored = []
    for name in sorted(set(names or available)):
        info = variant_info(name)
        # Named variants skip the recipe filter: naming the stress adapter is how it joins round 1.
        fits = [g for g in plan.groups if info is not None and (
            g.admits(info) or (names and Group(g.name, g.methods, g.splits, g.seed).admits(info)))]
        if not fits:
            ignored.append(name)
        for g in fits:
            groups[g.name].append(info)
    if names and ignored:
        raise ValidateError(f"{', '.join(ignored)} fit no group of this round "
                            f"({', '.join(g.name for g in plan.groups)})")
    return groups, ignored


def survey(round_no: int, *, variants: Sequence[str] | None = None, prompt_check: bool = True,
           prompts: Prompts | None = None, dtype: str | None = None) -> Survey:
    if round_no not in ROUNDS:
        raise ValidateError(f"there is no validation round {round_no} (rounds: {sorted(ROUNDS)})")
    check_plan()
    plan = ROUNDS[round_no]
    groups, ignored = _pick_variants(plan, variants)
    prompts = prompts or Prompts()
    out = Survey(groups=groups, ignored=ignored, contributing={g.name: set() for g in plan.groups})
    for lang in plan.sizes:
        for group in plan.groups:
            for set_name in SET_PRIORITY:
                recs: list[GenerationRecord] = []
                for info in groups[group.name]:
                    for split in group.splits:
                        try:
                            pool = load_pool(info.name, set_name, lang, split, prompts,
                                             prompt_check=prompt_check, dtype=dtype)
                        except ValidateError as e:
                            out.problems.append(str(e))
                            continue
                        if pool is None:
                            continue
                        if pool.stale_records:
                            out.warnings.append(
                                f"{pool.name} holds {pool.stale_records} outputs made from an older prompt "
                                "text; the harness regenerates the whole file before it judges it, so "
                                "items drawn from it may come back with new text and be left out later")
                        out.contributing[group.name].add(info.name)
                        out.files.append({"group": group.name, "variant": info.name, "set": set_name,
                                          "lang": lang, "split": split, **pool.counts()})
                        problem = pool.problem()
                        if problem:
                            out.problems.append(problem)
                            continue
                        recs += pool.records
                out.pools[(lang, group.name, set_name)] = recs
    return out


SPOTCHECK_COLUMNS = ("item_id", "translation")      # what makes another CSV in sheets/ a spot-check sheet


def spotcheck_pids(prompts: Prompts) -> tuple[dict[str, set[str]], list[str]]:
    """({lang: pids}, unmatched sheet names) for P01's translation spot-check sheets in
    validation/sheets/, whether or not their fixes have been applied yet.

    P01 writes spotcheck_{lang}.csv and spotcheck_{lang}_failures.csv with item_id "{lang}-{pid}",
    so a sheet whose name starts with "spotcheck" needs only its item_id column; any other CSV
    counts if it has item_id and translation columns. A bare pid also counts, for the language whose
    prompt text equals the translation cell (before apply-spotchecks) or that the file name names.
    Sheets with rows that match nothing are returned so the caller can say so. A "spotcheck" sheet
    that cannot be read raises ValidateError, naming it: its prompts could not be left out.
    """
    found: dict[str, set[str]] = defaultdict(set)
    unmatched, unreadable = [], []
    langs = list(config.get("languages.google_translate"))
    for path in sorted(sheets_dir().glob("*.csv")):
        if path.name.startswith("validation_"):
            continue
        p01 = path.stem.lower().startswith("spotcheck")
        try:
            filled = read_sheet(path, required=("item_id",) if p01 else SPOTCHECK_COLUMNS)
        except ValidateError as e:
            if p01:
                unreadable.append(str(e))
            continue                                     # not a spot-check sheet
        tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
        matched = 0
        for _, row in filled.rows:
            item_id, text = row["item_id"].strip(), normalise(row.get("translation", ""))
            lang_of, _, rest = item_id.partition("-")
            if lang_of in langs and any(prompts.exists(s, lang_of) and rest in prompts.get(s, lang_of)
                                        for s in SET_PRIORITY):
                found[lang_of].add(rest)
                matched += 1
                continue
            pid = item_id
            for lang in langs:
                for set_name in SET_PRIORITY:
                    if not prompts.exists(set_name, lang):
                        continue
                    prompt = prompts.get(set_name, lang).get(pid)
                    if prompt is not None and (normalise(prompt.text) == text or lang in tokens):
                        found[lang].add(pid)
                        matched += 1
        if filled.rows and not matched:
            unmatched.append(path.name)
    if unreadable:
        raise ValidateError("a translation spot-check sheet of P01's cannot be read, so the prompts on it "
                            "could not be kept off its reader's sheets (it shows their set): "
                            + "; ".join(unreadable) + ". Restore its header row as P01 wrote it, save it "
                            "again as 'CSV UTF-8', or move it out of validation/sheets/ once its fixes are "
                            "applied (lift data apply-spotchecks)")
    return dict(found), unmatched


def _seen_pids(round_no: int, prompts: Prompts) -> tuple[dict[str, set[str]], list[str]]:
    """{lang: pids its reader has already seen}: earlier rounds of any language that reader labels,
    the prompts they checked as translations, and the ones on spot-check sheets not applied yet.
    Also the warnings to print."""
    plan = ROUNDS[round_no]
    seen: dict[str, set[str]] = {lang: set() for lang in plan.sizes}
    earlier = [load_round(r) for r in sorted(ROUNDS) if r < round_no]
    spot, unmatched = spotcheck_pids(prompts)
    for lang in plan.sizes:
        mine = reader_langs(lang)
        seen[lang] |= {it["pid"] for doc in earlier for it in doc["items"] if it["lang"] in mine}
        for other in mine:
            seen[lang] |= spot.get(other, set())
            for set_name in SET_PRIORITY:
                if prompts.exists(set_name, other):
                    seen[lang] |= {pid for pid, p in prompts.get(set_name, other).items()
                                   if p.translation is not None and p.translation.checked}
    warnings = [f"{name} looks like a translation spot-check sheet, but none of its rows matches a "
                "prompt, so its prompts could not be left out of this draw" for name in unmatched]
    return seen, warnings


def _draw(pool: list[GenerationRecord], need: int, *, seed: int, round_no: int, lang: str,
          taken_pids: set[str], used_keys: set[str]) -> list[GenerationRecord]:
    order = sorted(pool, key=lambda r: rank(seed, "draw", round_no, lang, r.key))
    got: list[GenerationRecord] = []
    for rec in order:
        if len(got) == need:
            break
        if rec.pid in taken_pids or rec.key in used_keys:
            continue
        got.append(rec)
        taken_pids.add(rec.pid)
    return got


def build_round(round_no: int, *, seed: int = DEFAULT_SEED, variants: Sequence[str] | None = None,
                hindi_overlap: bool = False, prompt_check: bool = True,
                dtype: str | None = None) -> tuple[dict, dict[str, list[list[str]]]]:
    """The keys document and the sheet rows of a round, without writing anything."""
    prompts = Prompts()
    found = survey(round_no, variants=variants, prompt_check=prompt_check, prompts=prompts, dtype=dtype)
    plan = ROUNDS[round_no]
    if found.problems:
        raise ValidateError("these generations cannot be sampled yet; finish or fix the runs, or leave "
                            "the variant out with --variants:\n  " + "\n  ".join(found.problems))
    empty = found.empty_groups(plan)
    if empty:
        raise ValidateError("no outputs for group " + ", ".join(empty))
    used_keys = {it["key"] for r in sorted(ROUNDS) if r < round_no for it in load_round(r)["items"]}
    seen, warnings = _seen_pids(round_no, prompts)
    warnings = [*found.warnings, *warnings]
    info_by_name = {i.name: i for infos in found.groups.values() for i in infos}

    allocation: dict[str, dict[str, int]] = {}
    pool_sizes: dict[str, dict[str, int]] = {}
    drawn: dict[str, list[tuple[str, GenerationRecord]]] = {}
    taken: dict[tuple[str, ...], set[str]] = {}
    short: list[str] = []
    for lang, n in plan.sizes.items():
        alloc = allocate(n, plan.strata())
        allocation[lang] = {f"{g}/{s}": k for (g, s), k in alloc.items()}
        pool_sizes[lang] = {f"{g}/{s}": len(found.pools[(lang, g, s)]) for g, s in plan.strata()}
        reader = taken.setdefault(reader_langs(lang), set())
        reader |= seen[lang]
        drawn[lang] = []
        for group_name, set_name in plan.strata():
            need = alloc[(group_name, set_name)]
            pool = found.pools[(lang, group_name, set_name)]
            got = _draw(pool, need, seed=seed, round_no=round_no, lang=lang, taken_pids=reader,
                        used_keys=used_keys)
            if len(got) < need:
                short.append(f"{lang} {group_name}/{set_name}: needs {need}, found {len(got)} prompts "
                             f"this reader has not seen, among {len(pool)} outputs")
            drawn[lang] += [(group_name, r) for r in got]
    if short:
        raise ValidateError("not enough outputs to fill the round:\n  " + "\n  ".join(short))

    items: list[dict] = []
    sheets: dict[str, dict] = {}
    rows: dict[str, list[list[str]]] = {}
    taken_ids: set[str] = set()

    def add_sheet(lang: str, slot: str, picked: list[tuple[str, GenerationRecord]]) -> None:
        name = sheet_name(round_no, lang, slot)
        prefix = f"r{round_no}{lang}" + ("" if slot == "a" else slot)
        ordered = sorted(picked, key=lambda gr: rank(seed, "order", name, gr[1].key))
        sheets[name] = {"file": sheet_file(name), "lang": lang, "slot": slot, "n": len(ordered)}
        rows[name] = []
        for group_name, rec in ordered:
            prompt = prompts.get(rec.set, rec.lang)[rec.pid].text
            info = info_by_name[rec.variant]
            item_id = blind_id(seed, prefix, rec.key, taken_ids)
            items.append({
                "item_id": item_id, "sheet": name, "slot": slot, "round": round_no, "lang": lang,
                "key": rec.key, "variant": rec.variant, "method": info.method, "seed": info.seed,
                "group": group_name, "set": rec.set, "split": rec.split, "pid": rec.pid,
                "dtype": dtype_of(rec.meta), "prompt_sha": rec.prompt_sha,
                "shown_prompt_sha": sha256_text(normalise(prompt)),
                "shown_response_sha": sha256_text(normalise(rec.output)),
                "output_sha": sha256_text(rec.output),   # what a judgement of this item must be of
                "shown_non_ascii": {"prompt": not prompt.isascii(), "response": not rec.output.isascii()},
            })
            rows[name].append([item_id, prompt, rec.output, "", ""])

    for lang in plan.sizes:
        add_sheet(lang, "a", drawn[lang])
    if hindi_overlap:
        hindi = sorted(drawn["hi"], key=lambda gr: rank(seed, "overlap", round_no, gr[1].key))
        add_sheet("hi", "b", hindi[:HINDI_OVERLAP])

    doc = {
        "round": round_no,
        "seed": seed,
        "plan": {"sizes": dict(plan.sizes), "groups": [asdict(g) for g in plan.groups],
                 "set_priority": list(SET_PRIORITY)},
        "hindi_overlap": HINDI_OVERLAP if hindi_overlap else 0,
        "prompt_check": prompt_check,
        "dtype": dtype or str(config.get("generation.dtype")),
        "variants": {g: sorted(names) for g, names in found.contributing.items()},
        "variants_named": sorted(variants) if variants else None,
        "ignored_variants": found.ignored,
        "excluded_seen_pids": {lang: len(pids) for lang, pids in seen.items()},
        "warnings": warnings,
        "allocation": allocation,
        "pools": pool_sizes,
        "files": found.files,
        "sheets": sheets,
        "items": items,
    }
    return doc, rows


def _has_entries(path: Path) -> bool:
    try:
        filled = read_sheet(path, required=("item_id", "label"))
    except ValidateError:
        return True                     # unreadable as a sheet: do not overwrite it
    return any(row.get("label", "").strip() or row.get("note", "").strip() for _, row in filled.rows)


def archive(path: Path, folder: Path, stamp: str) -> None:
    """Move ``path`` into ``folder`` under a stamped name that never overwrites an earlier one."""
    folder.mkdir(parents=True, exist_ok=True)
    target, n = folder / f"{path.stem}-{stamp}{path.suffix}", 1
    while target.exists():
        n += 1
        target = folder / f"{path.stem}-{stamp}-{n}{path.suffix}"
    try:
        shutil.move(str(path), str(target))
    except PermissionError:
        raise ValidateError(f"cannot move {path}: it is open in another program (Excel?). Close it and "
                            "run the command again") from None


@dataclass
class WritePlan:
    old: dict | None                 # the round as drawn before, if any
    same: bool                       # the same sample: write only missing sheets
    targets: list[Path]              # sheets to write
    old_files: list[Path]            # old sheets to archive


def _old_draw(round_no: int, force: bool) -> dict | None:
    keys, meta = round_keys_path(round_no), draw_path(round_no)
    if keys.is_file() and meta.is_file():
        return load_round(round_no)
    present = [p for p in (keys, meta) if p.is_file()]
    if not present:
        return None
    if not force:
        missing = meta if keys.is_file() else keys
        raise ValidateError(f"{present[0]} is here but {missing.name} is not, and the two files belong "
                            "together: fetch the missing one (lift sync pull), or pass --force to archive "
                            f"{present[0].name} and draw round {round_no} again")
    return {"round": round_no, "sheets": {}, "items": [], "partial": True}


def _plan(round_no: int, doc: dict, force: bool) -> WritePlan:
    """Everything make_sheets refuses, checked before anything is written."""
    keys_path = round_keys_path(round_no)
    old = _old_draw(round_no, force)
    same = old is not None and not old.get("partial") and sample_digest(old) == sample_digest(doc)
    if old is not None and not same:
        labelled = sorted(p.name for p in labels_dir().glob(f"round{round_no}_*.jsonl"))
        later = [r for r in existing_rounds() if r > round_no]
        if not force:
            differs = []
            if old.get("seed") != doc["seed"]:
                differs.append(f"seed {old.get('seed')} (now {doc['seed']})")
            if old.get("hindi_overlap") != doc["hindi_overlap"]:
                differs.append("the Hindi overlap " + ("on" if old.get("hindi_overlap") else "off"))
            if old.get("variants_named") != doc["variants_named"]:
                differs.append(f"variants {old.get('variants_named') or 'all that fit'}")
            if old.get("dtype", doc["dtype"]) != doc["dtype"]:
                differs.append(f"dtype {old.get('dtype')}")
            hint = (f" It was drawn with {', '.join(differs)}; give the same options to rewrite its "
                    "sheets." if differs else " Its inputs have changed since.")
            raise ValidateError(f"{keys_path} already holds a different round {round_no} sample, and its "
                                f"sheets may be with readers.{hint} --force replaces it, and sheets "
                                "already sent out can then no longer be imported")
        if labelled:
            raise ValidateError(f"round {round_no} already has imported labels ({', '.join(labelled)}); "
                                "move them out of validation/labels before drawing it again")
        if later:
            raise ValidateError(f"round {later[0]} was drawn after round {round_no} and avoids its "
                                f"prompts; redrawing round {round_no} now could show a reader a prompt twice")
    targets = [sheets_dir() / info["file"] for info in doc["sheets"].values()]
    old_files = [sheets_dir() / i["file"] for i in old["sheets"].values()] if old and not same else []
    if same:
        targets = [t for t in targets if not t.is_file()]
    else:
        filled = [t for t in dict.fromkeys([*targets, *old_files]) if t.is_file() and _has_entries(t)]
        if filled:
            raise ValidateError(f"{', '.join(map(str, filled))} already has labels or notes typed in; "
                                "move it away before drawing a different sample over it")
    check_writable([*targets, *old_files, keys_path, draw_path(round_no)])
    return WritePlan(old=old, same=same, targets=targets, old_files=old_files)


def make_sheets(round_no: int, *, seed: int = DEFAULT_SEED, variants: Sequence[str] | None = None,
                hindi_overlap: bool | None = None, force: bool = False, prompt_check: bool = True,
                dtype: str | None = None) -> list[Path]:
    """Draw round ``round_no``, write its sheets to validation/sheets/ and its keys to
    validation/keys/round{N}.json, and return the sheet paths.

    Re-running with the same inputs changes nothing and only writes sheets that are missing, so a
    sheet someone is filling in is never overwritten. If the round already has keys with a
    different sample (the sheets may be with readers already), this refuses unless ``force``. Even
    then it refuses while labels of that round are imported, a later round exists, or a sheet has
    entries typed in, and it moves the old keys and sheets to the archive folders instead of
    deleting them. The keys go to round{N}.json as {item_id: {key, variant, set, ...}} and the rest
    of the draw to draw{N}.json. ``hindi_overlap`` defaults to ``validation.hindi_overlap`` in
    configs/project.yaml, else off; ``dtype`` to ``generation.dtype``.
    """
    if hindi_overlap is None:
        hindi_overlap = bool(config.get("validation.hindi_overlap", False))
    doc, rows = build_round(round_no, seed=seed, variants=variants, hindi_overlap=hindi_overlap,
                            prompt_check=prompt_check, dtype=dtype)
    plan = _plan(round_no, doc, force)
    if plan.same:
        doc = plan.old                  # same sample: keep the original date and git sha
    else:
        from lift.manifest import git_sha  # noqa: PLC0415 (runs git)

        doc["created"] = date.today().isoformat()
        doc["git"] = git_sha()
        if plan.old is not None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            for path in [*plan.old_files, round_keys_path(round_no), draw_path(round_no)]:
                if path.is_file():
                    folder = (sheets_dir() if path.suffix == ".csv" else keys_dir()) / "archive"
                    archive(path, folder, stamp)
    names = {info["file"]: name for name, info in doc["sheets"].items()}
    for path in plan.targets:
        name = names[path.name]
        write_sheet(path, header_line(round_no, doc["sheets"][name]["lang"]), SHEET_COLUMNS, rows[name])
    if not plan.same:
        _write_round(doc)
    return [sheets_dir() / info["file"] for info in doc["sheets"].values()]


def check_round(round_no: int, *, seed: int = DEFAULT_SEED, variants: Sequence[str] | None = None,
                hindi_overlap: bool | None = None, prompt_check: bool = True, force: bool = False,
                dtype: str | None = None) -> dict:
    """A dry run of ``make_sheets`` with the same options: per-file counts, the draw's warnings and
    everything that would stop it. Writes nothing."""
    if hindi_overlap is None:
        hindi_overlap = bool(config.get("validation.hindi_overlap", False))
    found = survey(round_no, variants=variants, prompt_check=prompt_check, dtype=dtype)
    plan = ROUNDS[round_no]
    problems = list(found.problems) + [f"no outputs for group {g}" for g in found.empty_groups(plan)]
    warnings = list(found.warnings)
    if not problems:
        try:
            doc, _ = build_round(round_no, seed=seed, variants=variants, hindi_overlap=hindi_overlap,
                                 prompt_check=prompt_check, dtype=dtype)
            warnings = doc["warnings"]
            _plan(round_no, doc, force)
        except ValidateError as e:
            problems.append(str(e))
    return {"round": round_no, "variants": {g: sorted(v) for g, v in found.contributing.items()},
            "ignored_variants": found.ignored, "files": found.files, "problems": problems,
            "warnings": warnings}


def score(round_no: int | None = None, **options) -> dict:
    """Forwards to lift.validation.score.score (kept here for the original stub's callers)."""
    from lift.validation import score as scoring  # noqa: PLC0415 (score imports this module)

    return scoring.score(round_no, **options)


def decide_primary_judge(**options) -> dict:
    """Forwards to lift.validation.score.decide_primary_judge."""
    from lift.validation import score as scoring  # noqa: PLC0415

    return scoring.decide_primary_judge(**options)
