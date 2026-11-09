"""``lift final``: the final evaluation of the frozen variants (P07). The CLI is lift.commands.final.

The final split is touched only here (CLAUDE.md rule 1), so every final line passes ``check_final`` first:

- a ``recipe-frozen-*`` git tag exists (Gate 3, README): listed by ``git tag`` in this checkout, or, on a GPU
  VM whose shallow fetch of the job's ref carries no tags, by ``git ls-remote --tags origin`` with the
  GITHUB_TOKEN the job hands to lift lines (sent as an http header through that git process's GIT_CONFIG_*
  variables, as kaggle/runner.ipynb fetches: never in a URL, on a command line or in a log);
- the tag froze a concrete recipe (``verified_recipe``, docs/decisions.md 79): configs/frozen_recipe.json,
  written by P05's ``lift select`` and committed before the tag, exists, is the same at HEAD (and in the
  working tree) as in the commit the newest tag points to (``git show <tag>:configs/frozen_recipe.json``; a
  shallow VM clone fetches that one tag first, the token again only in GIT_CONFIG_* variables), and its
  ``selection_sha256`` is the sha256 of outputs/metrics/selection.json's bytes now. So a ``lift select`` run
  after the tag (another frozen set, or the same set from other dev records) is refused at final time;
- the selection is coherent (XB-01): its ``overall.winner`` is one of its frozen settings, and its frozen
  map and winner are the recipe's;
- the variant is frozen: stock, a registered adapter (method "adapter": an adapter seed), or a repaired
  variant the recipe's ``frozen`` map names (one setting per family on the seed-1 adapter, applied to seeds
  1-3: ``frozen_variants``) and configs/variants.yaml registers. A selection that is not "decided" freezes
  nothing.

The same guard runs in every command that makes final outputs (``lift generate``, ``lift utility`` but
--judge-only, ``lift belebele``, ``lift heldout-loss`` with ``--split final``), so no side line can reach the
final split either (docs/decisions.md 74).

GPU half (``lift final --variant V``). Four stages, each through the harness (rule 2), in this process:

    safety           harmful, easy and hard benign in all six languages and Aya Red-teaming Hindi (final):
                     generated and judged by the primary judge (``lift.commands.run.run_cells``, run-dev's
                     path with split "final": with two GPUs the judge runs on its own one beside the
                     generator), each cell's AuditKIT record written
    utility          lift utility --variant V --split final --generate-only      200 Hindi pairs vs stock
                     (not for stock, the reference: any candidate's line answers for it)
    belebele         lift belebele --variant V --split final --langs en hi zh es ar th      700 x 6 items
    backtranslation  ar and th: a seeded 20% of the final pids of each safety set (back_translation in
                     configs/project.yaml; the same pids in every variant), the outputs translated into
                     English by NLLB-200 (on the judge's device) and judged again by the primary judge
                     against the prompt's English source; outputs/sensitivity/backtranslation/{variant}/
                     {set}_{lang}_final.jsonl (``lift stats`` compares them with the native judgements)

Every stage resumes by cache key, as its command does. A stage that ends 0 is recorded in
outputs/metrics/final/state/{variant}.json with the variant's intervention digest and the dtype, so a rerun
goes straight to the first stage not done. A stage that exits 3 or fails is not recorded as done; the run
goes on with the next stage and exits 3 or 1 at the end. Each run writes a manifest (stage "final_variant")
with every stage's wall seconds, the devices used and ``gpu_hours`` (wall time x devices), which the cost
section reads.

Laptop half (``lift final --variant V --judge-only``): the second judge's 10% sample, over P04's key list
of the variant's final split when it exists (validation/keys/second_judge/{variant}_final.txt), else
``--sample 0.10`` (decisions.md 31), through ``lift judge --judge second`` (Message Batches: a batch still
running exits 3, the same line collects it), and the utility pairs through ``lift utility --judge-only``.

``lift final --all`` runs the 13 core variants (``core_plan``) in an order that completes the primary
endpoint first: stock, the three adapter seeds, the overall winner on all three seeds, then the other two
families (cost order). When the stress adapter carries the repair comparison (Gate 2 failed), the standard
adapter's three seeds follow the winner, for the standard recipe's "retain with its interval" (16 variants).
A variant that is not registered yet is listed as not run, with its remedy.
"""

from __future__ import annotations

import base64
import gc
import json
import logging
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lift import config
from lift.io import sha256_file
from lift.paths import REPO_ROOT, metrics_dir, output_dir
from lift.schemas import LANGS

log = logging.getLogger("lift.final")

SPLIT = "final"
TAG_GLOB = "recipe-frozen-*"
RECIPE = "configs/frozen_recipe.json"  # P05's lift select writes it; committed before the tag
RECIPE_VERSION = 1
SAFETY = ("harmful", "easy_benign", "hard_benign")
REDTEAM = ("redteam", "hi")
STAGES = ("safety", "utility", "belebele", "backtranslation")
SEEDS = (1, 2, 3)
BT_SEED = 70  # the back-translation sample's draw (pipeline._rank), decisions.md 74


class Refused(ValueError):
    """The final split is not open to this variant (or not at all); the message says why and what to do."""


# ------------------------------------------------------------------------------------------------ guard


def _git(argv: Sequence[str], *, env: Mapping[str, str] | None = None) -> str | None:
    """stdout of ``git <argv>`` in the repo, or None when git fails (never raises, never logs argv)."""
    try:
        out = subprocess.run(
            ["git", *argv],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _tag_names(text: str | None) -> list[str]:
    names = set()
    for line in (text or "").splitlines():
        ref = line.split()[-1] if line.split() else ""
        ref = ref.removeprefix("refs/tags/").removesuffix("^{}")
        if re.fullmatch(r"recipe-frozen-[A-Za-z0-9._-]+", ref):
            names.add(ref)
    return sorted(names)


def frozen_tags(git: Callable[..., str | None] = _git) -> list[str]:
    """The ``recipe-frozen-*`` tags this checkout knows, or failing that the remote's (module docstring)."""
    local = _tag_names(git(["tag", "--list", TAG_GLOB]))
    if local:
        return local
    env = _remote_env(git)
    if env is None:
        return []
    return _tag_names(git(["ls-remote", "--tags", "origin", f"refs/tags/{TAG_GLOB}"], env=env))


def _remote_env(git: Callable[..., str | None]) -> dict[str, str] | None:
    """The environment of one git process that talks to origin with the job's GITHUB_TOKEN (module
    docstring), or None when there is no token or origin is not a github.com https remote."""
    token = os.environ.get("GITHUB_TOKEN")
    url = (git(["remote", "get-url", "origin"]) or "").strip()
    if not token or not url.startswith("https://github.com/"):
        return None
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_TRACE") and k != "GIT_CURL_VERBOSE"}
    env["GIT_TERMINAL_PROMPT"] = "0"
    # The header goes in through GIT_CONFIG_* variables of this one git process, never argv (``ps`` shows
    # every user a process's command line), appended after any config the caller already passes this way.
    try:
        n = int(env.get("GIT_CONFIG_COUNT") or 0)
    except ValueError:
        n = 0
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    env[f"GIT_CONFIG_KEY_{n}"] = "http.https://github.com/.extraheader"
    env[f"GIT_CONFIG_VALUE_{n}"] = f"AUTHORIZATION: basic {basic}"
    return env


def recipe_file() -> Path:
    """configs/frozen_recipe.json in this checkout (``lift.paths.frozen_recipe_path``, P05's writer's)."""
    from lift import paths

    return paths.frozen_recipe_path()


def _text(raw: str | bytes | None) -> str | None:
    if raw is None:
        return None
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return text.replace("\r\n", "\n")


def _tagged_recipe(tag: str, git: Callable[..., str | None]) -> str | None:
    """configs/frozen_recipe.json in the commit ``tag`` points to; a shallow clone (a GPU job's fetch of
    its ref carries no tags) fetches that one tag first, with the token kept out of argv."""
    text = git(["show", f"{tag}:{RECIPE}"])
    if text is None:
        env = _remote_env(git)
        if env is not None:
            ref = f"refs/tags/{tag}"
            git(["fetch", "--quiet", "--depth=1", "--no-tags", "origin", f"+{ref}:{ref}"], env=env)
            text = git(["show", f"{tag}:{RECIPE}"])
    return _text(text)


def verified_recipe(
    tag: str, *, git: Callable[..., str | None] | None = None, selection_file: Path | None = None
) -> dict[str, Any]:
    """configs/frozen_recipe.json as the tag froze it, or ``Refused`` naming what does not hold (module
    docstring): {"version": 1, "adapter", "selection_sha256", "overall_winner",
    "frozen": {family: setting}}."""
    git = git or _git
    path = recipe_file()
    remedy = (
        "the recipe is frozen by `lift select` (which writes it), a commit of it, then the tag it prints "
        "(git tag recipe-frozen-YYYYMMDD && git push --tags)"
    )
    if not path.is_file():
        raise Refused(f"{RECIPE} does not exist: {remedy}")
    work = _text(path.read_bytes())
    head = _text(git(["show", f"HEAD:{RECIPE}"]))
    problems = []
    if head is None:
        problems.append(f"{RECIPE} is not committed at HEAD: {remedy}")
    elif head != work:
        problems.append(
            f"{RECIPE} differs from HEAD's (an edit not committed): the frozen recipe is never edited"
        )
    tagged = _tagged_recipe(tag, git)
    if tagged is None:
        problems.append(
            f"{RECIPE} cannot be read at the tag {tag}: the tag must point to the commit that holds it "
            "(on a GPU job, GITHUB_TOKEN lets the shallow clone fetch the tag)"
        )
    elif head is not None and tagged != head:
        problems.append(
            f"{RECIPE} changed since the tag {tag}: the recipe the tag froze is the one evaluated; a new "
            "freeze needs a new dated tag, made only with a decisions.md entry (Gate 3 is passed once)"
        )
    if problems:
        raise Refused("; ".join(problems))
    try:
        recipe = json.loads(work or "")
    except ValueError as exc:
        raise Refused(f"{RECIPE} is not JSON ({exc})") from None
    if (
        not isinstance(recipe, dict)
        or recipe.get("version") != RECIPE_VERSION
        or not isinstance(recipe.get("frozen"), Mapping)
        or not recipe.get("selection_sha256")
    ):
        raise Refused(f"{RECIPE} is not version {RECIPE_VERSION} with frozen and selection_sha256")
    sel = selection_file or selection_path()
    digest = sha256_file(sel) if sel.is_file() else None
    if digest != recipe["selection_sha256"]:
        why = "is missing" if digest is None else f"has sha256 {digest[:12]}"
        raise Refused(
            f"outputs/metrics/selection.json is not the one the recipe froze (it {why}; the recipe names "
            f"{str(recipe['selection_sha256'])[:12]}): `lift select` ran after the tag, or the pull brought "
            "another. Restore the frozen selection.json (lift sync pull of the frozen run); selecting again "
            "after the freeze is not allowed (rule 1)"
        )
    problem = _winner_problem(recipe.get("overall_winner"), recipe["frozen"])
    if problem:
        raise Refused(f"{RECIPE}: {problem}")
    return recipe


def selection_path() -> Path:
    return metrics_dir() / "selection.json"


def read_selection(path: Path | None = None) -> dict[str, Any] | None:
    path = path or selection_path()
    if not path.is_file() or not path.stat().st_size:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def seed_name(adapter: str, seed: int) -> str:
    from lift.repair.select import seed_name as name

    return name(adapter, seed)


def frozen_settings(selection: Mapping[str, Any] | None) -> dict[str, str]:
    """{family: frozen seed-1 setting} of a decided selection ({} otherwise). P05 writes "frozen" as that
    mapping; a plain list of names (P08's reader's form) is read too, each under its family."""
    from lift.repair import settings as rs

    if not selection or selection.get("status", "decided") != "decided":
        return {}
    frozen = selection.get("frozen") or {}
    if isinstance(frozen, Mapping):
        return {str(k): str(v) for k, v in frozen.items()}
    out: dict[str, str] = {}
    for name in frozen:
        setting, _adapter = rs.parse_name(str(name))
        out.setdefault(setting.family, str(name))
    return out


def frozen_variants(selection: Mapping[str, Any] | None) -> list[str]:
    """Every frozen repaired variant: each family's frozen setting on seeds 1-3 of its adapter."""
    from lift.repair import settings as rs

    out = []
    for name in frozen_settings(selection).values():
        setting, adapter = rs.parse_name(name)
        out += [setting.name(seed_name(adapter, seed)) for seed in SEEDS]
    return list(dict.fromkeys(out))


def overall_winner(selection: Mapping[str, Any] | None) -> str | None:
    if not selection:
        return None
    winner = (selection.get("overall") or {}).get("winner") or selection.get("winner")
    return str(winner) if winner else None


def _winner_problem(winner: Any, frozen: Mapping[str, Any]) -> str | None:
    """XB-01: the overall winner must be the setting its family froze, or the primary endpoint's repair arm
    is a variant nothing evaluates (and P08 would publish it)."""
    if not winner:
        return "it names no overall winner"
    if str(winner) in {str(v) for v in frozen.values()}:
        return None
    names = ", ".join(map(str, frozen.values()))
    return (
        f"the overall winner {winner} is not one of the frozen settings ({names})"
        ": rerun `lift select` on the fixed p05-repairs (it freezes the overall winner as its family's "
        "setting), commit configs/frozen_recipe.json and tag again"
    )


def _selection_problems(selection: Mapping[str, Any] | None, recipe: Mapping[str, Any] | None) -> list[str]:
    """What keeps a decided selection from being the frozen recipe's: an overall winner its family did not
    freeze (XB-01), or another frozen map or winner than the recipe's (decisions.md 79)."""
    settings = frozen_settings(selection)
    if not settings:
        return []
    problems = []
    problem = _winner_problem(overall_winner(selection), settings)
    if problem:
        problems.append(f"outputs/metrics/selection.json: {problem}")
    if recipe is not None and (
        settings != {str(k): str(v) for k, v in recipe["frozen"].items()}
        or overall_winner(selection) != recipe.get("overall_winner")
    ):
        problems.append(
            f"selection.json's frozen map or winner differs from the frozen recipe ({RECIPE}): only "
            "the recipe the tag froze is evaluated"
        )
    return problems


def reported_recipe(
    selection: Mapping[str, Any] | None,
    repaired: Sequence[str] = (),
    *,
    tags: Sequence[str] | None = None,
    git: Callable[..., str | None] | None = None,
) -> dict[str, Any] | None:
    """The check ``lift stats`` and ``lift report`` make before they read or show final results of repaired
    variants (XB2-02; decisions.md 79: only the recipe the tag froze is evaluated, and so reported): the
    recipe as the newest tag froze it (``verified_recipe``: working tree, HEAD and tag agree, and its
    selection_sha256 is outputs/metrics/selection.json's), ``selection`` its frozen map and winner, and every
    repaired variant reported (``repaired``; default every frozen variant of ``selection``) one the recipe
    froze. Returns that recipe, or None when no repaired variant is reported (stock and the adapter seeds
    need no recipe here; lift final checked them). ``Refused`` names what does not hold and the remedy."""
    names = list(dict.fromkeys([*repaired, *frozen_variants(selection)]))
    if not names:
        return None
    tags = list(frozen_tags() if tags is None else tags)
    if not tags:
        raise Refused(
            f"no {TAG_GLOB} git tag exists, so no final result of a repaired variant is reported: the "
            "recipe is frozen (Gate 3) by `lift select`, a commit of configs/frozen_recipe.json and the tag "
            "it prints (a clone without tags: git fetch --tags)"
        )
    recipe = verified_recipe(tags[-1], git=git)
    problems = _selection_problems(selection, recipe)
    if not frozen_settings(selection):
        problems.append(
            "outputs/metrics/selection.json is missing or not decided, so it is not the selection the recipe "
            "froze: lift sync pull of the frozen run"
        )
    frozen = set(frozen_variants({"frozen": dict(recipe["frozen"])}))
    outside = [n for n in names if n not in frozen]
    if outside:
        problems.append(
            f"{', '.join(outside)} {'is' if len(outside) == 1 else 'are'} not frozen by the recipe the tag "
            f"{tags[-1]} froze: only its variants are reported"
        )
    if problems:
        raise Refused("; ".join(problems))
    return recipe


def check_final(
    variant: str,
    *,
    selection: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    use_file: bool = True,
    recipe: Mapping[str, Any] | None = None,
) -> str:
    """Refuse (``Refused``) unless the final split is open to ``variant`` (module docstring); returns the
    tag that opened it. ``recipe``: the frozen recipe already verified (``verified_recipe``); None reads and
    verifies it against the newest tag."""
    from lift.models.variants import get_spec

    tags = list(frozen_tags() if tags is None else tags)
    if selection is None and use_file:
        selection = read_selection()
    problems = []
    if not tags:
        problems.append(
            f"no {TAG_GLOB} git tag exists: the recipe is frozen (Gate 3) by `lift select` and the "
            "tag it prints (git tag recipe-frozen-YYYYMMDD && git push --tags), and only then is "
            "the final split opened"
        )
    elif recipe is None:
        try:
            recipe = verified_recipe(tags[-1])
        except Refused as exc:
            problems.append(str(exc))
    problems += _selection_problems(selection, recipe)
    try:
        spec = get_spec(variant)
    except KeyError:
        spec = None
        problems.append(f"{variant} is not registered in configs/variants.yaml")
    if spec is not None and spec.method not in ("stock", "adapter"):
        frozen = frozen_variants(selection)
        if recipe is not None:
            mine = set(frozen_variants({"frozen": dict(recipe["frozen"])}))
            frozen = [v for v in frozen if v in mine]
        if variant not in frozen:
            if not selection:
                why = "outputs/metrics/selection.json is missing (lift select, after lift sync pull)"
            elif selection.get("status", "decided") != "decided":
                why = f"outputs/metrics/selection.json is {selection.get('status')!r}, so nothing is frozen"
            else:
                why = f"the frozen recipe froze {frozen or 'nothing'}"
            problems.append(
                f"{variant} is not a frozen variant: {why}. Only stock, the adapter seeds and the "
                "frozen settings on seeds 1-3 are evaluated on the final split (rule 1)"
            )
    if problems:
        raise Refused(f"the final split is closed to {variant}: " + "; ".join(problems))
    return tags[-1]


def guard_split(variant: str, split: str) -> None:
    """The guard of a command that makes outputs (generate, utility, belebele, heldout-loss): on the final
    split, ``check_final`` or SystemExit with its refusal; other splits pass."""
    if split != SPLIT:
        return
    try:
        check_final(variant)
    except Refused as exc:
        raise SystemExit(f"{exc}") from None


# ------------------------------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class Planned:
    name: str
    role: str  # stock | adapter | winner | standard | family
    family: str | None
    seed: int | None


def standard_adapter(selection: Mapping[str, Any] | None) -> str | None:
    """The seed-1 standard adapter whose recipe gets "retain with its interval" when the stress adapter
    carries the repair comparison (README Gate 2; docs/decisions.md 80); None when the selection's adapter
    is a standard one. Read from selection.json's ``standard_adapter`` when it names one, else from Gate 2's
    report (outputs/metrics/gate2.json, ``candidate``: the Gate 1 pick, ``lift gate 2 --candidate NAME``);
    None when neither says (the verdict is then listed as not run with that remedy)."""
    adapter = str((selection or {}).get("adapter") or "")
    if "-stress-" not in adapter:
        return None
    named = (selection or {}).get("standard_adapter")
    if named:
        return str(named)
    path = metrics_dir() / "gate2.json"
    try:
        candidate = json.loads(path.read_text(encoding="utf-8")).get("candidate") if path.is_file() else None
    except (OSError, ValueError):
        candidate = None
    if not candidate:
        return None
    from lift.gates import adapter_name

    return adapter_name(str(candidate), "standard", 1)


def core_plan(selection: Mapping[str, Any]) -> list[Planned]:
    """The 13 core variants in priority order (module docstring). Needs a decided selection."""
    from lift.repair import settings as rs

    frozen = frozen_settings(selection)
    if not frozen:
        raise Refused(
            "outputs/metrics/selection.json is missing or not decided: lift final --all runs the "
            "frozen variants it names (lift select)"
        )
    adapter = str(selection["adapter"])
    winner = overall_winner(selection)
    problem = _winner_problem(winner, frozen)
    if problem:
        raise Refused(f"outputs/metrics/selection.json: {problem}")
    win_setting, _ = rs.parse_name(str(winner))
    plan = [Planned("stock", "stock", None, None)]
    plan += [Planned(seed_name(adapter, s), "adapter", None, s) for s in SEEDS]
    order = [win_setting.family] + [f for f in rs.FAMILIES if f != win_setting.family and f in frozen]
    standard = standard_adapter(selection)
    for family in order:
        name = winner if family == win_setting.family else frozen[family]
        setting, base = rs.parse_name(str(name))
        role = "winner" if family == win_setting.family else "family"
        plan += [Planned(setting.name(seed_name(base, s)), role, family, s) for s in SEEDS]
        if role == "winner" and standard:
            plan += [Planned(seed_name(standard, s), "standard", None, s) for s in SEEDS]
    return plan


# ------------------------------------------------------------------------------------------------ state


def state_path(variant: str) -> Path:
    return metrics_dir() / "final" / "state" / f"{variant}.json"


def read_state(variant: str) -> dict[str, Any]:
    path = state_path(variant)
    if not path.is_file() or not path.stat().st_size:
        return {"variant": variant, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(variant: str, state: Mapping[str, Any]) -> None:
    from lift.repair.sweep import write_json

    write_json(state_path(variant), dict(state))


def _done(state: Mapping[str, Any], stage: str, digest: str, dtype: str) -> bool:
    got = state.get("stages", {}).get(stage) or {}
    return got.get("rc") == 0 and got.get("digest") == digest and got.get("dtype") == dtype


@dataclass
class Options:
    device: str = "cuda:0"
    judge_device: str = "auto"
    dtype: str | None = None
    batch_size: int | None = None
    recheck: bool = False
    stages: Sequence[str] = STAGES
    extra: dict[str, Any] = field(default_factory=dict)


def stage_argv(stage: str, name: str, opts: Options) -> list[str]:
    """The lift command line of a stage that is a lift command (utility, belebele)."""
    tail = ([f"--dtype={opts.dtype}"] if opts.dtype else []) + (
        [f"--batch-size={opts.batch_size}"] if opts.batch_size else []
    )
    if stage == "utility":
        return [
            "utility",
            f"--variant={name}",
            f"--split={SPLIT}",
            "--generate-only",
            f"--device={opts.device}",
            *tail,
        ]
    if stage == "belebele":
        return [
            "belebele",
            f"--variant={name}",
            f"--split={SPLIT}",
            f"--langs={','.join(LANGS)}",
            f"--device={opts.device}",
            *tail,
        ]
    raise ValueError(f"{stage!r} is not a lift command stage")


def applies(stage: str, name: str) -> bool:
    return not (stage == "utility" and name == "stock")


def run_stage(stage: str, name: str, opts: Options) -> int:
    """One stage of one variant (module docstring): its exit code (0, 3, or 1 for a failure, logged)."""
    from lift.repair.sweep import _call

    if stage in ("utility", "belebele"):
        return _call(stage_argv(stage, name, opts))
    try:
        if stage == "safety":
            return safety(name, opts)
        if stage == "backtranslation":
            return backtranslation(name, opts)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        log.error("%s %s stopped: %s", name, stage, exc.code)
        return 1
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("%s %s failed", name, stage)
        return 1
    finally:
        _release()
    raise ValueError(f"unknown stage {stage!r}")


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _devices(opts: Options) -> int:
    judge = opts.judge_device
    if judge == "auto":
        from lift.commands.run import _judge_device

        try:
            judge = _judge_device("auto")
        except ImportError:
            judge = opts.device
    return len({opts.device, judge})


def run_variant(
    name: str,
    opts: Options,
    *,
    stage_fn: Callable[[str, str, Options], int] = run_stage,
    devices: int | None = None,
    tag: str | None = None,
) -> dict[str, int]:
    """Every stage of one variant not done yet (module docstring). {stage: exit code}. The caller checks
    the guard first (``check_final``)."""
    from lift.cachekey import intervention_digest
    from lift.manifest import finish_run, start_run
    from lift.models.variants import get_spec

    dtype = opts.dtype or str(config.get("generation.dtype"))
    digest = intervention_digest(get_spec(name))
    state = read_state(name)
    n_devices = devices if devices is not None else _devices(opts)
    run = start_run(
        "final_variant",
        variant=name,
        split=SPLIT,
        dtype=dtype,
        device=opts.device,
        judge_device=opts.judge_device,
        devices=n_devices,
        frozen_tag=tag,
    )
    results: dict[str, int] = {}
    seconds: dict[str, float] = {}
    try:
        for stage in opts.stages:
            if not applies(stage, name):
                continue
            if not opts.recheck and _done(state, stage, digest, dtype):
                log.info("%s %s: done in an earlier run", name, stage)
                results[stage] = 0
                continue
            log.info("%s %s: starting", name, stage)
            t0 = datetime.now(UTC)
            rc = stage_fn(stage, name, opts)
            seconds[stage] = round((datetime.now(UTC) - t0).total_seconds(), 3)
            results[stage] = rc
            state.setdefault("stages", {})[stage] = {
                "rc": rc,
                "digest": digest,
                "dtype": dtype,
                "seconds": seconds[stage],
                "finished": datetime.now(UTC).isoformat(),
            }
            write_state(name, state)
    except BaseException as exc:
        finish_run(run, status="failed", error=type(exc).__name__, stages=results, stage_seconds=seconds)
        raise
    wall = sum(seconds.values())
    finish_run(
        run,
        status="ok" if all(v == 0 for v in results.values()) else "partial",
        stages=results,
        stage_seconds=seconds,
        gpu_hours=round(wall * n_devices / 3600, 6),
    )
    return results


def exit_code(results: Mapping[str, Mapping[str, int]]) -> int:
    codes = {rc for r in results.values() for rc in r.values()}
    if codes - {0, 3}:
        return 1
    return 3 if 3 in codes else 0


def run_all(
    opts: Options,
    *,
    selection: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    stage_fn: Callable[[str, str, Options], int] = run_stage,
    devices: int | None = None,
    only: Sequence[str] | None = None,
    recipe: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """``lift final --all``: every core variant in order (``core_plan``); one that the guard refuses (not
    registered yet, say) is listed as not run with the refusal, and the rest go on. The frozen recipe is
    verified once (``recipe``: already verified)."""
    selection = selection if selection is not None else read_selection()
    plan = core_plan(selection or {})
    tags = list(frozen_tags() if tags is None else tags)
    recipe_refused = None
    if tags and recipe is None:
        try:
            recipe = verified_recipe(tags[-1])
        except Refused as exc:
            recipe_refused = str(exc)
    results: dict[str, dict[str, int]] = {}
    not_run: dict[str, str] = {}
    for item in plan:
        if only and item.name not in only:
            continue
        if recipe_refused:
            not_run[item.name] = f"the final split is closed to {item.name}: {recipe_refused}"
            log.error("%s: not run: %s", item.name, recipe_refused)
            continue
        try:
            tag = check_final(item.name, selection=selection, tags=tags, recipe=recipe)
        except Refused as exc:
            not_run[item.name] = str(exc)
            log.error("%s: not run: %s", item.name, exc)
            continue
        results[item.name] = run_variant(item.name, opts, stage_fn=stage_fn, devices=devices, tag=tag)
    rc = exit_code(results)
    if not_run:
        rc = max(rc, 3) if rc != 1 else 1
    return rc, {"results": results, "not_run": not_run, "plan": [item.__dict__ for item in plan]}


# --------------------------------------------------------------------------------------------- safety


def safety_cells() -> list[tuple[str, str]]:
    return [(s, lang) for s in SAFETY for lang in LANGS] + [REDTEAM]


def safety(name: str, opts: Options) -> int:
    """The final safety cells of one variant: lift run-dev's path (``run_cells``) on the final split, with
    the AuditKIT record of each judged cell. Exit 0 or 3 (cells listed with their remedies)."""
    from lift.commands import run
    from lift.judge.pipeline import judge_dir, make_judge
    from lift.models.variants import build_variant, get_spec

    local = judge_dir("primary") == "qwen3guard"
    judge_pin = [run._judge_entry("primary")] if local else []
    run.require_pinned("models.global", *run._base_pin(name), *judge_pin)
    dtype = run._dtype(opts.dtype)
    gen_device, judge_device = opts.device, run._judge_device(opts.judge_device)
    run._refuse_other_dtype(dtype, gen_device, judge_device if local else None)
    spec = get_spec(name)
    holder = {"variant": build_variant(spec, device=gen_device, dtype=dtype)}
    concurrent = gen_device != judge_device
    interleave = True
    if not concurrent and run._on_cuda(gen_device):
        interleave = run._free_gb(gen_device) >= run.JUDGE_GB + run.JUDGE_KV_HEADROOM_GB
    refused: dict[str, str] = {}
    skipped: dict[str, dict[str, str]] = {}

    def record(set_name: str, lang: str) -> None:
        run._record(name, set_name, lang, SPLIT, dtype=dtype, refused=refused)

    run.run_cells(
        holder,
        lambda: make_judge("primary", device=judge_device, dtype=dtype),
        safety_cells(),
        split=SPLIT,
        concurrent=concurrent,
        interleave=interleave,
        batch_size=opts.batch_size,
        record=record,
        skipped=skipped,
    )
    return run._refused_exit(refused, skipped)


# ------------------------------------------------------------------------------------ back-translation


def bt_path(variant: str, set_name: str, lang: str) -> Path:
    return output_dir() / "sensitivity" / "backtranslation" / variant / f"{set_name}_{lang}_{SPLIT}.jsonl"


def bt_pids(set_name: str) -> set[str]:
    """The back-translation sample of one set: a seeded fraction of its final pids (configs/project.yaml
    back_translation.fraction), the same in every variant and language."""
    from lift.judge.pipeline import second_judge_pids

    return second_judge_pids(set_name, SPLIT, float(config.get("back_translation.fraction")), BT_SEED)


def read_bt(variant: str, set_name: str, lang: str) -> list[dict[str, Any]]:
    from lift.io import read_jsonl

    path = bt_path(variant, set_name, lang)
    return list(read_jsonl(path)) if path.is_file() else []


def backtranslation(
    name: str,
    opts: Options,
    *,
    judge: Any = None,
    translator: Callable[[list[str], str], list[str]] | None = None,
) -> int:
    """The ar/th back-translation sample of one variant (module docstring). Needs the safety cells judged;
    resumes by generation key and output. A blank output or a failed translation is recorded with its
    reason (counted, never dropped). Exit 0, or 3 when a cell was not there yet."""
    from lift.data import translate
    from lift.io import append_jsonl, sha256_text
    from lift.judge.pipeline import JudgeItem, load_records
    from lift.models.generate import generations_path, read_generations
    from lift.prompts import load

    langs = [str(x) for x in config.get("back_translation.langs")]
    dtype = opts.dtype or str(config.get("generation.dtype"))
    missing = []
    for set_name in SAFETY:
        wanted = bt_pids(set_name)
        for lang in langs:
            try:
                native = {r.key: r for r in load_records(name, set_name, lang, SPLIT, dtype=dtype)}
            except (OSError, ValueError) as exc:
                missing.append(f"{set_name}_{lang}_{SPLIT}: {exc}")
                continue
            gens = [
                g
                for g in read_generations(generations_path(name, set_name, lang, SPLIT))
                if g.pid in wanted and g.key in native
            ]
            done = {(r["key"], r["output_sha"]) for r in read_bt(name, set_name, lang)}
            todo = [g for g in gens if (g.key, sha256_text(g.output)) not in done]
            if not todo:
                continue
            english = {p.pid: p.text_en or p.text for p in load(set_name, lang, SPLIT)}
            rows: list[dict[str, Any]] = []
            texts = [g for g in todo if g.output.strip()]
            for g in todo:
                if not g.output.strip():
                    rows.append(_bt_row(name, g, None, None, "blank output"))
            translated: list[str | None] = []
            if texts:
                translator = translator or _gpu_translator(opts)
                try:
                    translated = list(
                        translate.back_translate([g.output for g in texts], lang, translator=translator)
                    )
                except translate.BackTranslationError as exc:
                    translated = list(exc.outputs)
                    log.warning(
                        "%s %s_%s: %d back-translations failed", name, set_name, lang, len(exc.failures)
                    )
            judge = judge or _primary_judge(opts, dtype)
            items = [
                JudgeItem(key=g.key, prompt=english[g.pid], response=t, prompt_lang="en")
                for g, t in zip(texts, translated, strict=True)
                if t
            ]
            outcomes = iter(judge.judge(items)) if items else iter(())
            for g, t in zip(texts, translated, strict=True):
                if not t:
                    rows.append(_bt_row(name, g, None, None, "back-translation failed"))
                    continue
                o = next(outcomes)
                rows.append(_bt_row(name, g, sha256_text(t), o, None, judge_id=judge.judge_id))
            append_jsonl(bt_path(name, set_name, lang), rows)
            log.info("%s %s_%s: %d back-translated outputs judged", name, set_name, lang, len(rows))
    for why in missing:
        log.error("back-translation sample: %s not readable yet (the safety stage first)", why)
    return 3 if missing else 0


def _bt_row(
    variant: str, g: Any, bt_sha: str | None, outcome: Any, reason: str | None, *, judge_id: str | None = None
) -> dict[str, Any]:
    from lift.io import sha256_text

    return {
        "key": g.key,
        "variant": variant,
        "pid": g.pid,
        "lang": g.lang,
        "set": g.set,
        "split": g.split,
        "output_sha": sha256_text(g.output),
        "bt_sha": bt_sha,
        "judge": judge_id,
        "judge_label": outcome.judge_label if outcome is not None else None,
        "judge_failure": (outcome.native or {}).get("judge_failure") if outcome is not None else None,
        "skipped": reason,
    }


def _gpu_translator(opts: Options) -> Callable[[list[str], str], list[str]]:
    from lift.commands.run import _judge_device
    from lift.data.translate import _Nllb

    cfg = config.load("project")
    codes = cfg["languages"]["belebele"]
    model = _Nllb(cfg, device=_judge_device(opts.judge_device), batch_size=32)
    return lambda texts, lang: model(texts, codes[lang], codes["en"], factor=1.5)


def _primary_judge(opts: Options, dtype: str) -> Any:
    from lift.commands.run import _judge_device
    from lift.judge.pipeline import judge_dir, make_judge

    local = judge_dir("primary") == "qwen3guard"
    return make_judge(
        "primary", device=_judge_device(opts.judge_device) if local else None, dtype=dtype if local else None
    )


# -------------------------------------------------------------------------------------------- laptop


def second_judge_plan(name: str) -> tuple[list[list[str]], list[str]]:
    """(the second judge's lines of one variant's final split, the final cells they leave out).

    With P04's key list of the variant the line is ``--keys`` on it, which reads only the files holding
    its keys. Without one the lines are ``--sample``, and ``lift judge`` over named cells stops on the
    first generations file that is missing (R1). A final cell can lack its file for good: one short of its
    frozen split is skipped by ``lift final`` until a reader's verdict (easy_benign zh on the real data).
    So the lines name only the cells whose generations file exists, grouped by the languages a set has,
    and the others are returned as "{set}_{lang}_final" to be listed, never judged around silently."""
    from lift.models.generate import generations_path
    from lift.paths import validation_dir

    listed = validation_dir() / "keys" / "second_judge" / f"{name}_{SPLIT}.txt"
    if listed.is_file():
        return [["judge", f"--variant={name}", f"--split={SPLIT}", "--judge=second", f"--keys={listed}"]], []
    fraction = f"{float(config.get('second_judge_fraction')):.2f}"
    present: dict[str, list[str]] = {}
    absent: list[str] = []
    for set_name, lang in safety_cells():
        if generations_path(name, set_name, lang, SPLIT).is_file():
            present.setdefault(set_name, []).append(lang)
        else:
            absent.append(f"{set_name}_{lang}_{SPLIT}")
    groups: dict[tuple[str, ...], list[str]] = {}
    for set_name, langs in present.items():
        groups.setdefault(tuple(langs), []).append(set_name)
    lines = [
        [
            "judge",
            f"--variant={name}",
            f"--split={SPLIT}",
            "--judge=second",
            f"--sample={fraction}",
            "--sets",
            *sets,
            "--langs",
            *langs,
        ]
        for langs, sets in groups.items()
    ]
    return lines, absent


def second_judge_argv(name: str) -> list[list[str]]:
    """The second judge's lines of one variant's final split (``second_judge_plan``)."""
    return second_judge_plan(name)[0]


def judge_only(
    names: Sequence[str],
    *,
    no_wait: bool = False,
    sync: bool = False,
    call: Callable[[Sequence[str]], int] | None = None,
) -> int:
    """The laptop half for ``names``: the second judge's sample and the utility pairs (module docstring)."""
    from lift.repair.sweep import _call

    call = call or _call
    codes: dict[str, int] = {}
    left_out: dict[str, list[str]] = {}
    for name in names:
        lines, absent = second_judge_plan(name)
        if absent:
            left_out[name] = absent
        if name != "stock":
            lines.append(
                [
                    "utility",
                    f"--variant={name}",
                    f"--split={SPLIT}",
                    "--judge-only",
                    *(["--no-wait"] if no_wait else []),
                    *(["--sync"] if sync else []),
                ]
            )
        for argv in lines:
            codes[" ".join(argv)] = call(argv)
    for line, rc in codes.items():
        print(f"lift {line}: exit {rc}")
    for name, cells in left_out.items():
        print(
            f"{name}: no generations file for {', '.join(cells)}, so the second judge's sample leaves "
            f"{'it' if len(cells) == 1 else 'them'} out. `lift final --variant {name}` (GPU) makes a cell "
            "or lists why it is skipped (a cell short of its frozen split waits for a reader's verdict); "
            "then this line again"
        )
    values = set(codes.values())
    if values - {0, 3}:
        return 1
    return 3 if 3 in values or left_out else 0
