"""``lift select``: the selection rule on the seed-1 dev sweep, overall and within each repair family (P05).

Runs on the laptop after the sweep's outputs are pulled and its utility pairs judged; no torch. For stock,
the adapter and every setting of the chosen families (lift.repair.settings) it builds a DevScores from:

- harmful_hi: the Hindi dev harmful rate H/N under the primary judge, read with
  ``pipeline.paired_labels("stock", variant, "harmful", "hi", "dev")``, the only door to a paired
  comparison (it refuses incomplete cells, other judges, precisions, revisions, templates, decoding), so
  every number shares stock's configuration. Every prompt counts (rule 3).
- overrefusal_hard_hi: the refusal rate on hard_benign hi dev, read the same way.
- win_rate: the dev utility win rate against stock from lift utility's metrics, counted only while they
  describe the answers, verdicts and adapter as they now stand (Gate 1's own checks,
  ``lift.gates._utility_values`` and ``_current``). Stock is 0.5 by definition.
- belebele_hi: Belebele-hi dev accuracy, recomputed from the score files (``loglik.belebele_metrics``),
  every item of the split; stock's and each setting's must share the model revision. The adapter's own
  Belebele is not used by the rule and is reported when present.

Every safety cell and Belebele score file read must also have been made under the variant's registry
entry as it is now (``_stale``: each record's ``meta.intervention``), so cells of a replaced entry (a CAST
artifact fitted again) are never read as the new entry's; the utility metrics are checked against it by
Gate 1's checks above.

Nothing is selected while an input is missing or stale: the report says "incomplete" with each reason and
its remedy (rule 9: a subset would be a selection the pre-registration did not make). Otherwise
``lift.selection.select_repair`` decides the overall winner over all settings, and the same rule within
each other family (wiseft, resta, cast with cast_always) gives that family's frozen setting; the overall
winner is its own family's frozen setting (decisions item 71: the tie window moves with the pool, so the
rule over one family alone can pick a sibling). 3 families x 3 seeds = the 9 repaired variants of the
final eval, the overall winner always among them. outputs/metrics/selection.json and .md hold the full
table (every setting's numbers, checks and violations) and the drop's paired interval against the
adapter.

A decided selection also writes the frozen recipe, configs/frozen_recipe.json (``write_recipe``): the
adapter, the overall winner, the frozen map {family: setting} and the sha256 of the selection.json bytes
just written, which Gate 3 commits and tags and the final guard compares with selection.json. A recipe is
frozen once: while the file exists, a run that would write another one or another selection.json (other
numbers, or an undecided run) is refused before anything is written (``RecipeFrozen``) unless
``refreeze``; the same inputs again keep both files as they are. "The same" is the decision's content:
the "created" time and the freeze plan (which seeds are registered yet, today's date in the tag line) are
left out of the comparison. ``refreeze`` itself is refused while a local ``recipe-frozen-*`` tag exists
(``frozen_tags``): after Gate 3's tag, re-freezing needs a dated docs/decisions.md entry and the tag
deleted or replaced by hand. A selection.json that is replaced while a recipe or a tag stands is first
copied to outputs/metrics/selection.<sha12>.json, so the pinned bytes are never lost. An undecided run never
writes the recipe.

selection.json names paths relative to the repo (outputs/..., configs/...), never the machine's absolute
paths (``_unrooted``): it is published with the logs.

The command then prints the freeze steps: the commit-and-tag command, the configs/variants.yaml entries
of the frozen settings on seeds 1-3 (seed 2 and 3 adapters must be registered; CAST needs its artifact,
fitted first), and the fit-cast lines of seeds 2-3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lift import config, paths
from lift.paths import metrics_dir
from lift.repair import settings as rs
from lift.selection import DevScores, Selection, select_repair

REPORT = "selection"
CELLS = {"harmful_hi": ("harmful", "hi", "dev"), "overrefusal_hard_hi": ("hard_benign", "hi", "dev")}


Read = tuple[DevScores | None, list[str], dict[str, Any]]


RECIPE = "configs/frozen_recipe.json"
TAG_PATTERN = "recipe-frozen-*"


class RecipeFrozen(RuntimeError):
    """A recipe is frozen already and this run would replace it or the selection.json it pins."""


class Missing(ValueError):
    """An input of one variant's DevScores cannot be read; the message says why and what remakes it."""


@dataclass
class Collected:
    stock: DevScores | None
    adapter: DevScores | None
    candidates: list[DevScores]
    missing: dict[str, list[str]] = field(default_factory=dict)
    details: dict[str, dict[str, Any]] = field(default_factory=dict)


def report_paths() -> tuple[Path, Path]:
    return metrics_dir() / f"{REPORT}.json", metrics_dir() / f"{REPORT}.md"


def _dtype_flag(dtype: str) -> str:
    return "" if dtype == config.get("generation.dtype") else f" --dtype {dtype}"


def _stale(variant: str, path: Path) -> str | None:
    """Why the records at ``path`` were not made under ``variant``'s registry entry as it is now, or None.

    Each generation and score record names the entry it was made under (``meta.intervention``, lift.cachekey's
    intervention digest: method, adapter, seed, params and artifacts, so a refit CAST artifact's hash too).
    The paired_labels and Belebele checks compare a cell with itself, its prompts and stock, not with the
    registry: after ``lift fit-cast --refit`` and ``lift register-repairs --replace`` the old artifact's
    cells would still read as complete. A repair's records must name their entry; stock's and the
    adapter's, made before records named one, are checked only where they do."""
    from lift.cachekey import intervention_digest
    from lift.io import read_jsonl
    from lift.models.variants import get_spec

    if not path.is_file():
        return None
    try:
        spec = get_spec(variant)
    except KeyError:
        return None
    now = intervention_digest(spec)
    made = {(r.get("meta") or {}).get("intervention") for r in read_jsonl(path)}
    if None in made and spec.method in rs.METHODS:
        return (f"{path.name} has records that do not name the registry entry they were made under, so "
                "nothing shows they are this entry's")
    other = sorted(str(x) for x in made - {None, now})
    if other:
        return (f"{path.name} was made under another registry entry of {variant} (intervention "
                f"{', '.join(x[:12] for x in other)}; the entry now is {now[:12]}: a CAST artifact fitted "
                "again and registered with --replace?)")
    return None


def _rate(variant: str, which: str, dtype: str) -> tuple[float, dict[str, Any]]:
    from lift.judge import pipeline
    from lift.models.generate import generations_path, restart_command
    from lift.stats import rates

    set_name, lang, split = CELLS[which]
    stale = _stale(variant, generations_path(variant, set_name, lang, split))
    if stale:
        raise Missing(f"{set_name}_{lang}_{split}: {stale}; start the cell over: run "
                      f"`{restart_command(variant, set_name, lang, split, dtype=dtype)}` as a job line right "
                      f"after `lift sync pull`, then `lift run-dev --variant {variant}{_dtype_flag(dtype)}` "
                      "(or lift sweep), then lift sync pull")
    try:
        _stock, labels = pipeline.paired_labels("stock", variant, set_name, lang, split, judge="primary",
                                                dtype=dtype)
    except (OSError, ValueError) as exc:
        raise Missing(f"{set_name}_{lang}_{split} cannot be read against stock ({_short(exc)}); run `lift "
                      "run-dev "
                      f"--variant {variant}{_dtype_flag(dtype)}` (or lift sweep) on a GPU, then lift sync "
                      "pull") from None
    r = rates(list(labels.values()))
    value = r["harmful_rate"] if which == "harmful_hi" else r["refusal_rate"]
    return float(value), {"n": r["n"], **{k: r[k] for k in r if k.startswith("n_")},
                          "harmful_upper": r["harmful_upper"]}


def _win_rate(variant: str, dtype: str) -> tuple[float, dict[str, Any]]:
    from lift import gates

    path = gates.utility_metrics_path(variant, "dev", dtype=dtype)
    rerun = f"`lift utility --variant {variant} --split dev{_dtype_flag(dtype)} --judge-only` (laptop)"
    if not path.is_file():
        raise Missing(f"no dev utility metrics ({path.name}): answers from `lift sweep`, then {rerun}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        values = gates._utility_values(doc, variant, dtype)
        gates._current(doc, variant, dtype)
    except (OSError, ValueError) as exc:
        raise Missing(f"utility metrics {path.name} are not usable ({_short(exc)}); run {rerun} "
                      "again") from None
    return float(values["win_rate"]), {k: values.get(k) for k in ("n", "lo", "hi", "failed_orders", "judge")}


def _belebele(variant: str, dtype: str, *, pinned: str | None) -> tuple[float, dict[str, Any]]:
    from lift.models import loglik

    rerun = f"`lift belebele --variant {variant} --split dev --langs hi{_dtype_flag(dtype)}` (GPU)"
    stale = _stale(variant, loglik.belebele_scores_path(variant, "hi", "dev", dtype=dtype))
    if stale:
        raise Missing(f"Belebele-hi dev: {stale}; start it over: run `lift belebele --variant {variant} "
                      f"--split dev --langs hi --restart{_dtype_flag(dtype)}` (GPU), then lift sync pull")
    try:
        doc = loglik.belebele_metrics(variant, "dev", dtype=dtype, langs=("hi",))
    except (OSError, ValueError) as exc:
        raise Missing(f"Belebele-hi dev scores cannot be read ({_short(exc)}); run {rerun}") from None
    if doc is None or "hi" not in doc["langs"]:
        why = (doc or {}).get("left_out", {}).get("hi", "no score file")
        raise Missing(f"no complete Belebele-hi dev scores ({_short(why)}); run {rerun}")
    if pinned and doc.get("adapter_hash") != pinned:
        raise Missing(f"Belebele-hi dev scores were made from adapter {str(doc.get('adapter_hash'))[:12]}, "
                      "not the "
                      f"pinned {pinned[:12]}; run {rerun} --restart")
    hi = doc["langs"]["hi"]
    return float(hi["accuracy"]), {"n": hi.get("n"), "model_rev": doc.get("model_rev")}


def read_scores(name: str, method: str, dtype: str) -> Read:
    """(DevScores or None, what is missing, details) of one variant on seed-1 dev in ``dtype``."""
    from lift.models.variants import get_spec

    missing: list[str] = []
    details: dict[str, Any] = {}
    values: dict[str, float] = {}
    pinned = None
    if name != "stock":
        try:
            pinned = get_spec(name).artifacts.get("adapter")
        except KeyError:
            return None, [f"configs/variants.yaml does not list {name}: run `lift register-repairs` (laptop) "
                          "and "
                          "commit configs/variants.yaml"], details
    readers: dict[str, Callable[[], tuple[float, dict[str, Any]]]] = {
        "harmful_hi": lambda: _rate(name, "harmful_hi", dtype),
        "overrefusal_hard_hi": lambda: _rate(name, "overrefusal_hard_hi", dtype),
        "win_rate": (lambda: (0.5, {"note": "stock is the reference"})) if name == "stock"
        else (lambda: _win_rate(name, dtype)),
        "belebele_hi": lambda: _belebele(name, dtype, pinned=pinned),
    }
    for key, read in readers.items():
        try:
            values[key], details[key] = read()
        except Missing as exc:
            if method == "adapter" and key == "belebele_hi":    # not used by the rule: reported if present
                values[key], details[key] = math.nan, {"note": str(exc)}
                continue
            missing.append(f"{key}: {exc}")
    if missing:
        return None, missing, details
    return DevScores(name=name, method=method, **values), [], details


def collect(adapter: str, families: Sequence[str], dtype: str, *,
            reader: Callable[[str, str, str], Read] | None = None) -> Collected:
    read = reader or read_scores
    out = Collected(stock=None, adapter=None, candidates=[])
    names = [("stock", "stock"), (adapter, "adapter")]
    names += [(s.name(adapter), s.method) for s in rs.settings(families)]
    for name, method in names:
        scores, missing, details = read(name, method, dtype)
        out.details[name] = details
        if missing:
            out.missing[name] = missing
            continue
        if method == "stock":
            out.stock = scores
        elif method == "adapter":
            out.adapter = scores
        else:
            out.candidates.append(scores)
    return out


def _check_revisions(c: Collected) -> list[str]:
    """Belebele-hi of stock and of every setting must come from the same Global revision."""
    revs = {name: d.get("belebele_hi", {}).get("model_rev") for name, d in c.details.items()}
    stock = revs.get("stock")
    return [f"{name}: Belebele-hi made on Global {str(rev)[:12]}, stock's on {str(stock)[:12]}"
            for name, rev in revs.items() if rev and stock and rev != stock]


def _drop_interval(adapter: str, setting: str, dtype: str) -> dict[str, Any] | None:
    """The paired drop (adapter minus setting harmful) on Hindi dev, by prompt: bootstrap and McNemar."""
    from lift.judge import pipeline
    from lift.stats import mcnemar_exact, paired_bootstrap

    try:
        a, s = pipeline.paired_labels(adapter, setting, "harmful", "hi", "dev", judge="primary", dtype=dtype)
    except (OSError, ValueError):
        return None
    pids = sorted(a)
    x = [a[p] == "harmful" for p in pids]
    y = [s[p] == "harmful" for p in pids]
    boot = paired_bootstrap(x, y, clusters=pids)
    mc = mcnemar_exact(x, y)
    return {"diff": boot.diff, "lo": boot.lo, "hi": boot.hi, "mcnemar_p": mc.p_value, "n": len(pids)}


def _selection_doc(sel: Selection) -> dict[str, Any]:
    return {"winner": sel.winner, "verdict": sel.verdict, "eligible": list(sel.eligible)}


def decide(c: Collected, rules: Mapping[str, Any], families: Sequence[str]) -> dict[str, Any]:
    """The overall selection and each family's, over complete inputs."""
    assert c.stock is not None and c.adapter is not None
    overall = select_repair(c.stock, c.adapter, c.candidates, rules)
    winner_family = rs.FAMILY[next(x.method for x in c.candidates if x.name == overall.winner)]
    per_family: dict[str, Any] = {}
    for fam in families:
        pool = [x for x in c.candidates if rs.FAMILY[x.method] == fam]
        if not pool:
            continue
        own = select_repair(c.stock, c.adapter, pool, rules)
        per_family[fam] = _selection_doc(own)
        if fam == winner_family and own.winner != overall.winner:
            # Decisions item 71: the overall winner is frozen as its family's setting, so it is the one
            # evaluated on all three seeds; the family's own pick is kept for the record.
            per_family[fam].update(winner=overall.winner, verdict=overall.verdict,
                                   family_rule_winner=own.winner)
    min_win = 0.5 + rules["min_gain_retained"] * (c.adapter.win_rate - 0.5)
    rows = []
    for row in overall.table:
        cand = next(x for x in c.candidates if x.name == row["name"])
        rows.append({**row, "family": rs.FAMILY[row["method"]], "harmful_hi": cand.harmful_hi,
                     "overrefusal_hard_hi": cand.overrefusal_hard_hi,
                     "overrefusal_increase": cand.overrefusal_hard_hi - c.stock.overrefusal_hard_hi,
                     "min_win_rate": min_win, "belebele_hi": cand.belebele_hi,
                     "belebele_drop": c.stock.belebele_hi - cand.belebele_hi,
                     "eligible": row["violations"] == 0})
    return {"overall": _selection_doc(overall), "families": per_family, "table": rows}


def _today() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def seed_name(adapter: str, seed: int) -> str:
    m = re.fullmatch(r"(?P<stem>adapter-.+-s)(?P<seed>\d+)", adapter)
    if m is None:
        raise ValueError(f"{adapter!r} does not end in -s<seed>")
    return f"{m.group('stem')}{seed}"


def freeze_plan(adapter: str, frozen: Mapping[str, str], *, date: str | None = None) -> dict[str, Any]:
    """The freeze steps for the frozen settings ({family: setting name on ``adapter``}): the tag command,
    for each seed 1-3 each frozen variant's name and its registry entry where it can be written now (the
    seed's adapter registered; CAST: its artifact fitted), else the lines that make it, and the fit-cast
    lines of seeds 2-3."""
    plan: dict[str, Any] = {"tag": f'git add {RECIPE} && git commit -m "Freeze the recipe" && '
                                   f"git tag recipe-frozen-{date or _today()} && git push && git push --tags",
                            "variants": [], "fit_cast": [], "register": []}
    tags = []
    needs_cast = False
    for family, name in frozen.items():
        setting, _ = rs.parse_name(name)
        tags.append(setting.tag)
        needs_cast |= setting.family == "cast"
        for seed in (1, 2, 3):
            seed_adapter = seed_name(adapter, seed)
            item: dict[str, Any] = {"family": family, "seed": seed, "name": setting.name(seed_adapter)}
            try:
                spec = rs.spec_for(setting, seed_adapter)
                item["entry"] = rs.entry(spec)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                item["pending"] = _short(exc)
            plan["variants"].append(item)
    for seed in (2, 3):
        seed_adapter = seed_name(adapter, seed)
        if needs_cast:
            plan["fit_cast"].append(f"lift fit-cast --adapter {seed_adapter}")
        plan["register"].append(f"lift register-repairs --adapter {seed_adapter} --settings {','.join(tags)}")
    return plan


def _short(exc: Any, n: int = 300) -> str:
    text = str(exc)
    return text if len(text) <= n else text[: n - 3] + "..."


def run(adapter: str, *, families: Sequence[str] = rs.FAMILIES, dtype: str | None = None,
        reader: Callable[..., Any] | None = None, date: str | None = None,
        refreeze: bool = False) -> dict[str, Any]:
    """Collect, decide (when complete) and write outputs/metrics/selection.json and .md, and when decided
    the frozen recipe (module docstring). Raises RecipeFrozen, writing nothing, when a recipe is frozen
    and this run would change it or its selection.json, unless ``refreeze``; ``refreeze`` is refused while a
    recipe-frozen-* tag exists (or git cannot list the tags). The returned doc is the one written, plus
    "recipe" (the recipe written, when decided)."""
    dtype = dtype or config.get("generation.dtype")
    rules = config.get("selection", name="repairs")
    c = collect(adapter, families, dtype, reader=reader)
    problems = _check_revisions(c) if reader is None else []
    doc: dict[str, Any] = {
        "kind": "selection", "version": 1, "adapter": adapter, "split": "dev", "lang": "hi", "dtype": dtype,
        "judge": "primary", "families": list(families), "rules": dict(rules),
        "created": datetime.now(UTC).isoformat(), "stock": asdict(c.stock) if c.stock else None,
        "adapter_scores": _nan_safe(asdict(c.adapter)) if c.adapter else None,
        "missing": c.missing, "problems": problems, "details": _nan_safe(c.details),
    }
    tags = frozen_tags()
    if refreeze and (tags is None or tags):
        raise RecipeFrozen(_tagged_message(tags))
    if c.missing or problems or c.stock is None or c.adapter is None:
        doc["status"] = "incomplete"
    else:
        doc.update(decide(c, rules, families))
        doc["status"] = "decided"
        if reader is None:
            for row in doc["table"]:
                row["drop_paired"] = _drop_interval(adapter, row["name"], dtype)
        frozen = {fam: sel["winner"] for fam, sel in doc["families"].items()}
        doc["frozen"] = frozen
        doc["freeze"] = freeze_plan(adapter, frozen, date=date)
    doc = _unrooted(doc)
    path = paths.frozen_recipe_path()
    held = path.read_text(encoding="utf-8") if path.is_file() else None
    text = selection_text(doc)
    if held is not None or tags:
        text = _unchanged(doc) or text
    recipe = recipe_doc(doc, text) if doc["status"] == "decided" else None
    if held is not None and not refreeze and (recipe is None or recipe_text(recipe) != held):
        raise RecipeFrozen(_frozen_message(path, held, recipe, tags))
    if held is not None or tags is None or tags:
        _keep_copy(text)
    write(doc, json_text=text)
    if recipe is not None and recipe_text(recipe) != held:
        write_recipe(recipe)
    return {**doc, "recipe": recipe}


def selection_text(doc: Mapping[str, Any]) -> str:
    """outputs/metrics/selection.json's text: the bytes the recipe's sha256 pins."""
    return json.dumps(_nan_safe(dict(doc)), indent=1, ensure_ascii=False) + "\n"


# Left out when two selections are compared: not the decision, only when it was made and what can be
# registered yet (the freeze plan's entries, its tag line's date).
NOT_DECISION = ("created", "freeze")


def _unchanged(doc: Mapping[str, Any]) -> str | None:
    """selection.json's text on disk when it says what ``doc`` says (``NOT_DECISION`` apart), else None:
    the same inputs again keep the bytes a frozen recipe pins."""
    path = report_paths()[0]
    try:
        text = path.read_text(encoding="utf-8")
        old = json.loads(text)
    except (OSError, ValueError):
        return None
    new = json.loads(selection_text(doc))
    if not isinstance(old, dict):
        return None
    for key in NOT_DECISION:
        old.pop(key, None)
        new.pop(key, None)
    return text if old == new else None


def _keep_copy(text: str) -> Path | None:
    """Copy selection.json to selection.<sha12>.json beside it before ``text`` replaces it (a recipe or a
    tag may pin those bytes); None when nothing would change."""
    path = report_paths()[0]
    try:
        old = path.read_bytes()
    except OSError:
        return None
    if old == text.encode("utf-8"):
        return None
    copy = path.with_name(f"{path.stem}.{hashlib.sha256(old).hexdigest()[:12]}{path.suffix}")
    if not copy.is_file():
        copy.write_bytes(old)
    return copy


def frozen_tags(cwd: Path | None = None) -> list[str] | None:
    """The local ``recipe-frozen-*`` tags (``git tag -l``), sorted; None when git cannot answer there."""
    try:
        out = subprocess.run(["git", "tag", "-l", TAG_PATTERN], cwd=cwd or paths.REPO_ROOT,
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(out.stdout.split())


def _tagged_message(tags: Sequence[str] | None) -> str:
    if tags is None:
        where = "git could not list this checkout's tags, so a recipe-frozen-* tag cannot be ruled out"
    else:
        where = f"the recipe is tagged ({', '.join(tags)})"
    return (f"--refreeze refused: {where}. Nothing was written. Re-freezing after the tag needs a dated "
            "docs/decisions.md entry saying why, and the tag deleted or replaced by hand (git tag -d "
            f"<tag>, and git push --delete origin <tag> if it was pushed); then lift select --refreeze. "
            "The same inputs again need no --refreeze")


def _unrooted(obj: Any) -> Any:
    """``obj`` with every absolute path under the repo, outputs/ or data/ written relative to the repo."""
    if isinstance(obj, str):
        return _relative(obj)
    if isinstance(obj, dict):
        return {k: _unrooted(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unrooted(v) for v in obj]
    return obj


def _relative(text: str) -> str:
    from lift import gates

    text = gates._scrub(text)                     # the (movable) outputs and data roots
    root = paths.REPO_ROOT
    for form in (str(root).replace("\\", "\\\\"), str(root), root.as_posix()):
        for sep in ("\\\\", "\\", "/"):
            text = text.replace(form + sep, "")
    return text


def recipe_doc(doc: Mapping[str, Any], selection_json: str) -> dict[str, Any]:
    """The frozen recipe of a decided selection whose selection.json text is ``selection_json``."""
    return {"version": 1, "adapter": doc["adapter"],
            "selection_sha256": hashlib.sha256(selection_json.encode("utf-8")).hexdigest(),
            "overall_winner": doc["overall"]["winner"], "frozen": dict(doc["frozen"])}


def recipe_text(recipe: Mapping[str, Any]) -> str:
    return json.dumps(dict(recipe), indent=2, sort_keys=True) + "\n"


def write_recipe(recipe: Mapping[str, Any]) -> Path:
    path = paths.frozen_recipe_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(recipe_text(recipe))
    os.replace(tmp, path)
    return path


def _frozen_message(path: Path, held: str, recipe: Mapping[str, Any] | None,
                    tags: Sequence[str] | None = ()) -> str:
    try:
        old = json.loads(held)
        what = (f"overall winner {old.get('overall_winner')} on {old.get('adapter')}, selection.json sha256 "
                f"{str(old.get('selection_sha256'))[:12]}")
    except ValueError:
        old, what = {}, "unreadable"
    if recipe is None:
        change = "this run is undecided and would overwrite the selection.json it pins"
    elif recipe["frozen"] != old.get("frozen") or recipe["overall_winner"] != old.get("overall_winner"):
        change = (f"this run decides another one (overall winner {recipe['overall_winner']}, frozen "
                  f"{', '.join(recipe['frozen'].values())})")
    else:
        change = "this run would write another selection.json (other numbers, same frozen settings)"
    if tags is None or tags:
        how = ("Re-freezing after the tag (" + (", ".join(tags) if tags else "git could not list the tags")
               + ") needs a dated docs/decisions.md entry saying why and the tag deleted or replaced by hand "
               "(git tag -d <tag>), then lift select --refreeze")
    else:
        how = (f"Re-freezing needs deleting {RECIPE} and a dated docs/decisions.md entry saying why, then "
               "lift select again (or --refreeze)")
    return (f"a recipe is already frozen in {_relative(str(path))} ({what}); {change}. Nothing was written. "
            f"{how}")


def _nan_safe(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    return obj


def _pct(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.1f}"


def render(doc: Mapping[str, Any]) -> str:
    lines = [f"# Repair selection: {doc['adapter']} (seed-1 dev, Hindi, {doc['dtype']})", "",
             f"Status: **{doc['status']}**", ""]
    if doc["status"] != "decided":
        lines += ["Nothing is selected while an input is missing:", ""]
        for name, why in {**doc["missing"], **{"problems": doc["problems"]}}.items():
            for w in why:
                lines.append(f"- {name}: {w}")
        return "\n".join(lines) + "\n"
    s, a = doc["stock"], doc["adapter_scores"]
    lines += [f"Stock: harmful {_pct(s['harmful_hi'])}%, hard-benign refusal "
              f"{_pct(s['overrefusal_hard_hi'])}%, "
              f"Belebele-hi {_pct(s['belebele_hi'])}%. Adapter: harmful {_pct(a['harmful_hi'])}%, win rate "
              f"{_pct(a['win_rate'])}%.", "",
              f"Overall winner: **{doc['overall']['winner']}** (verdict {doc['overall']['verdict']}).", ""]
    for fam, sel in doc["families"].items():
        note = (f"; the overall winner, frozen in place of {sel['family_rule_winner']} by the family's rule "
                "alone (decisions item 71)") if sel.get("family_rule_winner") else ""
        lines.append(f"- {fam}: {sel['winner']} (verdict {sel['verdict']}{note})")
    lines += ["", "| setting | harmful hi % | drop pts | drop 95% CI | over-refusal +pts | win % (min) | "
              "Belebele-hi % (drop pts) | violations | eligible |", "|---|---|---|---|---|---|---|---|---|"]
    for r in doc["table"]:
        ci = r.get("drop_paired")
        ci_text = f"[{_pct(ci['lo'])}, {_pct(ci['hi'])}]" if ci else "n/a"
        lines.append(f"| {r['name']} | {_pct(r['harmful_hi'])} | {_pct(r['drop'])} | {ci_text} | "
                     f"{_pct(r['overrefusal_increase'])} | {_pct(r['win_rate'])} ({_pct(r['min_win_rate'])}) "
                     "| "
                     f"{_pct(r['belebele_hi'])} ({_pct(r['belebele_drop'])}) | {r['violations']} | "
                     f"{'yes' if r['eligible'] else 'no'} |")
    return "\n".join(lines) + "\n"


def write(doc: Mapping[str, Any], *, json_text: str | None = None) -> tuple[Path, Path]:
    json_path, md_path = report_paths()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((json_path, json_text if json_text is not None else selection_text(doc)),
                       (md_path, render(doc))):
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    return json_path, md_path


def yaml_entry(entry: Mapping[str, Any]) -> str:
    return "\n".join(rs.entry_lines(entry, "  "))
