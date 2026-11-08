"""A fake world for the validation tests: prompt sets, generations, judgements and utility
verdicts written with the real schemas into a temporary directory. No model, no network.

Texts are neutral placeholders, so no test holds a real harmful prompt or completion, and they
never contain a variant name, a set name or a judge label (the leak tests rely on that).

The world commits its own split files (``World.splits_dir``, data/splits in the repo), which
``use_world`` makes lift.splits.SPLITS_DIR: the prompt files are read through
lift.data.build.load_prompts, which refuses a split stamp the committed split does not give.

The judge ids are the ones the configured judges write (``qwen_judge_id``, ``frontier_judge_id``),
since the validation reads only those (lift.validation.outputs.judge_rules).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from lift import config
from lift import splits as frozen_splits
from lift.data.build import SPLIT_SETS
from lift.io import sha256_obj, sha256_text, write_jsonl
from lift.judge import frontier
from lift.judge.labels import rules_tag
from lift.schemas import (
    LABELS,
    GenerationRecord,
    JudgementRecord,
    PromptRecord,
    UnclearFlags,
    UtilityVerdict,
)

QWEN_REV = "0123456789abcdef0123456789abcdef01234567"   # any sha: tests see the configured one as "main"
FRONTIER_API = "2026-11-01"                             # any API version fits the configured rule


def _judges_entry(kind: str) -> tuple[str, dict]:
    """(name, block) of the judges block of configs/project.yaml that makes the ``kind`` judge."""
    for name in ("safety_primary", "safety_second"):
        block = config.get(f"judges.{name}")
        if ("provider" in block) == (kind == "frontier"):
            return name, block
    raise LookupError(f"no judges block makes a {kind} judge")


def qwen_judge_id(*, dtype: str = "bf16", mode: str | None = None, revision: str = QWEN_REV,
                  rules: str | None = None) -> str:
    """A Qwen3Guard judge id as the harness writes it, {model}@{revision}#{rules tag}#{dtype}#{mode}; by
    default one the configured rule accepts (the configured model and mode, the labelling rules now)."""
    _, entry = _judges_entry("qwen3guard")
    return f"{entry['id']}@{revision}#{rules or rules_tag()}#{dtype}#{mode or entry['controversial']}"


def frontier_judge_id(*, api: str = FRONTIER_API, rubric: str | None = None, rules: str | None = None) -> str:
    """A frontier judge id as the harness writes it, {FrontierJudge.judge_id}#{rules tag}; by default the
    configured judge's (its provider, model, rubric version and settings tag, which
    lift.judge.pipeline._expected_judge builds its rule from), spoken to over API version ``api``."""
    name, entry = _judges_entry("frontier")
    s = frontier._settings(entry["provider"], name, {})
    tag = frontier._settings_tag(s["temperature"], s["effort"], s["thinking"], s["max_tokens"])
    version = rubric or frontier.load_rubric(entry["rubric"]).version
    return f"{entry['provider']}/{entry['model']}@{version}+{tag}+api{api}#{rules or rules_tag()}"


PRIMARY_ID = qwen_judge_id()
FRONTIER_ID = frontier_judge_id()
UTILITY_ID = "fakeprovider/utility-model@2026-11-01"
LANGS = ("hi", "zh", "es", "en")
SAFETY_SETS = ("harmful", "easy_benign", "hard_benign")
PID_PREFIX = {"harmful": "prh-test-", "easy_benign": "prb-test-", "hard_benign": "orb-",
              "redteam": "art-hi-", "utility": "utl-aya-"}
LID = {"hi": "hin_Deva", "zh": "cmn_Hani", "es": "spa_Latn", "en": "eng_Latn"}

# Round 1 needs stock and seed-1 adapters on dev; round 2 needs repaired dev and final outputs.
DEV_VARIANTS = ("stock", "adapter-aya-std-s1", "adapter-indicalign-std-s1", "adapter-aya-std-s2",
                "wiseft-l0.50-adapter-aya-std-s1", "cast-m1.0-adapter-aya-std-s1")
FINAL_VARIANTS = ("stock", "adapter-aya-std-s1", "wiseft-l0.50-adapter-aya-std-s1")


def pid_of(set_name: str, i: int) -> str:
    width = 5 if set_name == "utility" else 4
    return f"{PID_PREFIX[set_name]}{i:0{width}d}"


def prompt_text(set_name: str, lang: str, pid: str, version: int = 0) -> str:
    # The set is not spelled out: only a hash of it, so a sheet cell never names the set.
    tag = sha256_text(f"{set_name}:{lang}:{pid}")[:10]
    return f"question {tag} v{version}"


def output_text(variant: str, lang: str, pid: str) -> str:
    return f"answer {sha256_text(f'{variant}:{lang}:{pid}')[:12]}"


def truth(key: str) -> str:
    """The 'real' label of a generation: what a perfect reader would say."""
    return LABELS[int(key[:8], 16) % 4]


def qwen_native(label: str, key: str) -> tuple[dict, UnclearFlags]:
    """Qwen3Guard native fields and flags whose strict final label is ``label``."""
    flags = UnclearFlags(too_short=False, repetitive=False, garbled=False, wrong_lang=False,
                         lid_top=[("hin_Deva", 0.99)])
    if label == "harmful":
        # Every second harmful call is Controversial + refusal: harmful when strict, refusal when loose.
        if int(key[8:10], 16) % 2:
            return {"safety": "Controversial", "categories": ["None"], "refusal": "Yes"}, flags
        return {"safety": "Unsafe", "categories": ["Violent"], "refusal": "No"}, flags
    if label == "refusal":
        return {"safety": "Safe", "categories": ["None"], "refusal": "Yes"}, flags
    if label == "benign":
        return {"safety": "Safe", "categories": ["None"], "refusal": "No"}, flags
    short = flags.model_copy(update={"too_short": True})
    return {"safety": "Safe", "categories": ["None"], "refusal": "No"}, short


@dataclass
class World:
    root: Path
    n: dict[str, dict[str, int]] = field(default_factory=dict)          # set -> split -> count
    generations: dict[tuple[str, str, str, str], list[GenerationRecord]] = field(default_factory=dict)
    primary_flip: set[str] = field(default_factory=set)      # keys where Qwen3Guard disagrees with truth
    frontier_flip: set[str] = field(default_factory=set)     # keys where the frontier judge disagrees

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def validation(self) -> Path:
        return self.root / "validation"

    @property
    def splits_dir(self) -> Path:
        """The world's committed split files: lift.splits.SPLITS_DIR while it is in use."""
        return self.data / "splits"

    def by_key(self) -> dict[str, GenerationRecord]:
        return {g.key: g for recs in self.generations.values() for g in recs}


def freeze_split(world: World, set_name: str, assignments: dict[str, str]) -> Path:
    """Commit ``assignments`` (pid -> split) to the world's split file of ``set_name``, in the layout of
    data/splits/{set}.json, with the pids already there (a pid has one split in every language)."""
    path = world.splits_dir / f"{set_name}.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))["assignments"] if path.is_file() else {}
    clash = sorted(pid for pid, split in assignments.items() if frozen.get(pid, split) != split)
    if clash:
        raise ValueError(f"{set_name}: {clash[:5]} are frozen in another split already")
    frozen.update(assignments)
    doc = {"name": set_name, "seed": 0, "sizes": dict(sorted(Counter(frozen.values()).items())),
           "created": "2026-11-04", "n": len(frozen), "assignments": dict(sorted(frozen.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def write_prompts(world: World, set_name: str, lang: str, splits: dict[str, int],
                  version: int = 0) -> list[PromptRecord]:
    """The prompt file of ``set_name`` in ``lang``, pids in order over ``splits`` (split -> count), and
    those splits frozen in the world's split file when the set has one (redteam is final, with none)."""
    records, i = [], 0
    for split, count in splits.items():
        for _ in range(count):
            pid = pid_of(set_name, i)
            records.append(PromptRecord(pid=pid, lang=lang, set=set_name, split=split,
                                        text=prompt_text(set_name, lang, pid, version),
                                        source="fake"))
            i += 1
    write_jsonl(world.data / "prompts" / f"{set_name}_{lang}.jsonl", [r.model_dump() for r in records])
    if set_name in SPLIT_SETS:
        freeze_split(world, set_name, {r.pid: r.split for r in records})
    return records


def make_generation(variant: str, prompt: PromptRecord, *, output: str | None = None) -> GenerationRecord:
    key = sha256_obj({"variant": variant, "pid": prompt.pid, "lang": prompt.lang, "text": prompt.text})
    if output is None:
        output = output_text(variant, prompt.lang, prompt.pid)
    return GenerationRecord(key=key, variant=variant, pid=prompt.pid, lang=prompt.lang, set=prompt.set,
                            split=prompt.split, prompt_sha=sha256_text(prompt.text), output=output,
                            n_prompt_tokens=400, n_new_tokens=40, hit_cap=False, seconds=0.1,
                            meta={"dtype": "bf16"})


def write_generations(world: World, variant: str, set_name: str, lang: str, split: str,
                      records: list[GenerationRecord]) -> Path:
    path = world.outputs / "generations" / variant / f"{set_name}_{lang}_{split}.jsonl"
    write_jsonl(path, [r.model_dump() for r in records])
    world.generations[(variant, set_name, lang, split)] = records
    return path


def judgement(gen: GenerationRecord, judge: str, label: str) -> JudgementRecord:
    """``judge``'s judgement of ``gen`` with final label ``label``: Qwen3Guard's native fields for a
    Qwen3Guard id (any revision, precision or mode), a frontier judge's otherwise."""
    qwen = "Qwen3Guard" in judge
    native, flags = qwen_native(label, gen.key)
    if not qwen:
        native = {"label": label.upper(), "reason": "fake"}
    return JudgementRecord(key=gen.key, variant=gen.variant, pid=gen.pid, lang=gen.lang, set=gen.set,
                           split=gen.split, judge=judge,
                           judge_label=label if label != "unclear" or not qwen else "benign",
                           label=label, flags=flags, native=native, output_sha=sha256_text(gen.output))


def other_label(label: str) -> str:
    return LABELS[(LABELS.index(label) + 1) % 4]


def write_judgements(world: World, *, frontier: bool = True, skip: set[str] = frozenset()) -> None:
    """Qwen3Guard (dir qwen3guard) and frontier (dir frontier) judgements of every generation,
    except the keys in ``skip``. Judges agree with ``truth`` except on their flip sets."""
    for (variant, set_name, lang, split), recs in world.generations.items():
        if set_name == "utility":
            continue
        for judge, dirname, flips in ((PRIMARY_ID, "qwen3guard", world.primary_flip),
                                      (FRONTIER_ID, "frontier", world.frontier_flip)):
            if judge == FRONTIER_ID and not frontier:
                continue
            rows = []
            for g in recs:
                if g.key in skip:
                    continue
                label = truth(g.key)
                rows.append(judgement(g, judge, other_label(label) if g.key in flips else label))
            path = world.outputs / "judgements" / dirname / variant / f"{set_name}_{lang}_{split}.jsonl"
            write_jsonl(path, [r.model_dump() for r in rows])


def build_world(root: Path, *, n_dev: int = 40, n_final: int = 40, langs=LANGS,
                dev_variants=DEV_VARIANTS, final_variants=FINAL_VARIANTS, judgements: bool = True,
                utility: bool = True) -> World:
    world = World(root=root)
    for set_name in SAFETY_SETS:
        splits = {"dev": n_dev, "final": n_final}
        world.n[set_name] = splits
        for lang in langs:
            prompts = write_prompts(world, set_name, lang, splits)
            for split, variants in (("dev", dev_variants), ("final", final_variants)):
                mine = [p for p in prompts if p.split == split]
                for variant in variants:
                    write_generations(world, variant, set_name, lang, split,
                                      [make_generation(variant, p) for p in mine])
    if utility:
        prompts = write_prompts(world, "utility", "hi", {"dev": 70, "final": 10})
        world.n["utility"] = {"dev": 70, "final": 10}
        dev = [p for p in prompts if p.split == "dev"]
        for variant in ("stock", "adapter-aya-std-s1"):
            write_generations(world, variant, "utility", "hi", "dev",
                              [make_generation(variant, p) for p in dev])
    if judgements:
        write_judgements(world)
    return world


def use_world(monkeypatch, world: World) -> World:
    monkeypatch.setenv("LIFT_DATA_DIR", str(world.data))
    monkeypatch.setenv("LIFT_OUTPUT_DIR", str(world.outputs))
    monkeypatch.setenv("LIFT_VALIDATION_DIR", str(world.validation))
    monkeypatch.setattr(frozen_splits, "SPLITS_DIR", world.splits_dir)
    return world


def write_verdicts(world: World, candidate: str, scores: dict[str, float], *,
                   judge: str = UTILITY_ID, path: Path | None = None) -> Path:
    """UtilityVerdict records for ``candidate`` (pid -> score); order fields chosen to match."""
    def orders(score: float) -> tuple[str, str]:
        return {1.0: ("A", "B"), 0.75: ("A", "TIE"), 0.5: ("TIE", "TIE"), 0.25: ("B", "TIE"),
                0.0: ("B", "A")}[score]

    rows = [UtilityVerdict(pid=pid, variant=candidate, judge=judge, order_ab=orders(s)[0],
                           order_ba=orders(s)[1], score=s).model_dump()
            for pid, s in sorted(scores.items())]
    path = path or world.outputs / "utility" / candidate / "utility_hi_dev.jsonl"
    write_jsonl(path, rows)
    return path


def read_round(validation: Path, round_no: int) -> dict:
    """A round's draw with its items, from validation/keys/draw{N}.json and round{N}.json (the
    item_id -> record map), as lift.validation.sample.load_round returns it."""
    items = json.loads((validation / "keys" / f"round{round_no}.json").read_text(encoding="utf-8"))
    draw = json.loads((validation / "keys" / f"draw{round_no}.json").read_text(encoding="utf-8"))
    return {**draw, "items": [{"item_id": i, **rec} for i, rec in items.items()]}


def read_csv(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def fill_sheet(src: Path, dst: Path, values: dict[str, str], *, column: str = "label",
               google_export: bool = False, notes: dict[str, str] | None = None) -> Path:
    """What a reader sends back: ``values`` (item_id -> cell) in ``column``.

    ``google_export`` mimics a Google Sheets download: no BOM, CRLF line ends, and the one-cell
    comment row padded with empty cells to the full width.
    """
    rows = read_csv(src)
    header = rows[1]
    col = header.index(column)
    note_col = header.index("note")
    out = [rows[0], header]
    for row in rows[2:]:
        row = list(row)
        if row[0] in values:
            row[col] = values[row[0]]
        if notes and row[0] in notes:
            row[note_col] = notes[row[0]]
        out.append(row)
    if google_export:
        out[0] = out[0] + [""] * (len(header) - 1)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            csv.writer(f, lineterminator="\r\n").writerows(out)
    else:
        with open(dst, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(out)
    return dst
