"""The generation keys the second (frontier) judge must cover. Implemented in P04.

README, Judging: the frontier judge runs "on the human-validated samples and a random 10% of
outputs". This is the one sampler for that 10%: run the second judge with ``--keys`` on these
lists rather than with a ``--sample`` of its own, and P07 reads random.txt for "the second judge vs
the primary on its sample".

- random: for each (set, split) of the safety sets, ceil(fraction x number of pids) prompt ids
  picked by a seeded hash of the pid, and every current output for those pids, in every language
  and variant whose generations file is complete. Incomplete files are listed and skipped, so a
  later run only adds keys. Folders under outputs/generations that are not study variants (the
  names in configs/variants.yaml) are listed and skipped.
- validated: every generation key on any validation sheet (all rounds, the overlap included).

Picking prompt ids from data/prompts, not keys from the files, keeps the random sample paired
across variants and languages (the second judge's view of drift and repair can be compared prompt
by prompt) and independent of how far a run has got.

Writes validation/keys/second_judge/: random.txt, validated.txt, all.txt (one key per line,
sorted), {variant}_{split}.txt (both lists for one variant and split, for one
``lift judge --judge second --variant V --split S --keys FILE`` call each) and sample.json
(counts, and which list each key is on).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from pathlib import Path

from lift import config, paths
from lift.schemas import LANGS
from lift.validation.outputs import Prompts, included, load_pool, variant_info, variants_with_outputs
from lift.validation.sample import DEFAULT_SEED, existing_rounds, load_round, rank
from lift.validation.store import ValidateError, check_writable, keys_dir, write_json, write_text

SECOND_JUDGE_SETS = ("harmful", "easy_benign", "hard_benign", "redteam")
SPLITS = ("train", "val", "fit", "dev", "final")


def second_judge_dir() -> Path:
    return keys_dir() / "second_judge"


def parse_generations_name(stem: str) -> tuple[str, str, str] | None:
    """``hard_benign_hi_dev`` -> ("hard_benign", "hi", "dev"); None for anything else."""
    parts = stem.rsplit("_", 2)
    if len(parts) != 3 or parts[1] not in LANGS or parts[2] not in SPLITS:
        return None
    return parts[0], parts[1], parts[2]


def chosen_pids(prompts: Prompts, fraction: float, seed: int) -> dict[tuple[str, str], set[str]]:
    """{(set, split): pids}: ceil(fraction x n) of each split's pids, the same in every language."""
    population: dict[tuple[str, str], set[str]] = defaultdict(set)
    for set_name in SECOND_JUDGE_SETS:
        for lang in config.get("languages.all"):
            if not prompts.exists(set_name, lang):
                continue
            for pid, p in prompts.get(set_name, lang).items():
                if included(p):
                    population[(set_name, p.split)].add(pid)
    out = {}
    for (set_name, split), pids in sorted(population.items()):
        k = math.ceil(fraction * len(pids))
        out[(set_name, split)] = set(sorted(pids, key=lambda p: rank(seed, "second", set_name, split, p))[:k])
    return out


def second_judge_sample(*, fraction: float | None = None, seed: int = DEFAULT_SEED,
                        require_validated: bool = True, prompt_check: bool = True) -> dict:
    fraction = float(config.get("second_judge_fraction")) if fraction is None else float(fraction)
    if not 0 < fraction <= 1:
        raise ValidateError("the second-judge fraction must be in (0, 1]")
    rounds = existing_rounds()
    if not rounds and require_validated:
        raise ValidateError("no validation round keys here (validation/keys/round*.json), so the list "
                            "would miss every validated item: lift sync pull first, or pass "
                            "--no-validated to write the random sample alone")
    prompts = Prompts()
    chosen = chosen_pids(prompts, fraction, seed)
    random_keys: dict[str, tuple[str, str, str, str]] = {}      # key -> (variant, split, set, lang)
    files, incomplete, ignored = [], [], []
    for variant in variants_with_outputs():
        if variant_info(variant) is None:       # not a study variant (a smoke test, say)
            ignored.append(variant)
            continue
        for path in sorted((paths.generations_dir() / variant).glob("*.jsonl")):
            parsed = parse_generations_name(path.stem)
            if parsed is None or parsed[0] not in SECOND_JUDGE_SETS:
                continue
            set_name, lang, split = parsed
            pool = load_pool(variant, set_name, lang, split, prompts, prompt_check=prompt_check)
            if pool is None:
                continue
            entry = {"variant": variant, "set": set_name, "lang": lang, "split": split, **pool.counts()}
            if pool.problem():
                incomplete.append(f"{variant}/{path.name}")
                files.append({**entry, "sampled": 0, "skipped": True})
                continue
            keys = [r.key for r in pool.records if r.pid in chosen.get((set_name, split), set())]
            random_keys.update({k: (variant, split, set_name, lang) for k in keys})
            files.append({**entry, "sampled": len(keys), "skipped": False})
    validated = {it["key"]: (it["variant"], it["split"], it["set"], it["lang"])
                 for r in rounds for it in load_round(r)["items"]}

    out_dir = second_judge_dir()
    per_variant: dict[tuple[str, str], set[str]] = defaultdict(set)
    cells: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for key, (variant, split, set_name, lang) in [*random_keys.items(), *validated.items()]:
        per_variant[(variant, split)].add(key)
        cells[(variant, split)][set_name].add(lang)
    outputs = {"random.txt": sorted(random_keys), "validated.txt": sorted(validated),
               "all.txt": sorted(set(random_keys) | set(validated))}
    outputs.update({f"{v}_{s}.txt": sorted(keys) for (v, s), keys in sorted(per_variant.items())})
    check_writable([out_dir / name for name in outputs] + [out_dir / "sample.json"])
    if out_dir.is_dir():
        for old in out_dir.glob("*.txt"):
            old.unlink()
    for name, keys in outputs.items():
        write_text(out_dir / name, "".join(k + "\n" for k in keys))
    doc = {
        "fraction": fraction,
        "seed": seed,
        "rounds": rounds,
        "prompt_check": prompt_check,
        "n_keys": len(outputs["all.txt"]),
        "n_random": len(random_keys),
        "n_validated": len(validated),
        "n_both": len(set(random_keys) & set(validated)),
        "pids": {f"{s}/{sp}": len(v) for (s, sp), v in sorted(chosen.items())},
        "incomplete_files": incomplete,
        "ignored_variants": ignored,
        "files": files,
        "per_variant": {f"{v}_{s}.txt": len(k) for (v, s), k in sorted(per_variant.items())},
        "membership": {k: ("both" if k in random_keys and k in validated
                           else "random" if k in random_keys else "validated")
                       for k in outputs["all.txt"]},
        "created": date.today().isoformat(),
    }
    write_json(out_dir / "sample.json", doc)
    return {**doc, "dir": str(out_dir), "commands": judge_commands(cells)}


def judge_commands(cells: dict[tuple[str, str], dict[str, set[str]]]) -> list[str]:
    """One ``lift judge`` line per variant, split and group of sets with the same languages, with
    --sets and --langs spelled out (the red-team set is Hindi and final only, so it gets its own)."""
    lines = []
    for (variant, split), by_set in sorted(cells.items()):
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for set_name, langs in by_set.items():
            groups[tuple(sorted(langs))].append(set_name)
        for langs, sets in sorted(groups.items()):
            lines.append(f"lift judge --judge second --variant {variant} --split {split} "
                         f"--sets {' '.join(sorted(sets))} --langs {' '.join(langs)} "
                         f"--keys validation/keys/second_judge/{variant}_{split}.txt")
    return lines
