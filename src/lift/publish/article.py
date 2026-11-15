"""``lift publish article``: hf/article.md from hf/templates/article.md and the metrics files.

The organisers' nine sections, numbered and in their order (``SECTIONS``: the kickoff slides'
"Submission artifacts" list, docs/decisions.md item 119), then ``EXTRAS``, unnumbered: related work, how
to reproduce the numbers, and the links. Results come from
final_report.json (P07) and selection.json (P05), what validation round 1 found from the three files
``lift validate`` writes (P04; ``lift.publish.validated``), the design numbers (set sizes, selection rule,
validation bar) from configs/, the rows each adapter was trained on from the ``lift train`` manifests,
and the links from configs/publish.yaml and the collections file. Sections 5 and 6 list the repos of the
two collections as ``lift publish collections`` plans them, each with the access configs/publish.yaml
gives it and the reason for it.
``hf/article.sources.json`` lists every rendered value with its file and JSON pointer. While the recipe
is not frozen at its ``recipe-frozen-*`` tag (``inputs.recipe_block``), the article is a draft marked
blocked under its title (a placeholder: not for posting). The article is the repository's README.md
(docs/decisions.md item 96); ``lift publish article`` writes both files, and
stamps the notebook's links (``acceptance.stamp_notebook``) from the same settings.

Judges are named by their configs/project.yaml entry's kind (Qwen3Guard or a frontier model), never by
role: which one is primary comes from final_report.json ``primary_judge`` (the judge id of the final
judgements, after the pre-registered decide-judge rule); an id that is neither entry is refused.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping
from pathlib import Path

from lift import config, paths
from lift.publish import PublishError, preflight, render, settings, validated
from lift.publish.collect import links_path, read_links
from lift.publish.datasets import VERDICT_ROLES, verdict_lines
from lift.publish.inputs import (
    FIGURES,
    P05,
    P07,
    RECIPE_IN_GIT,
    TABLES,
    UNDEFINED,
    Facts,
    _decided,
    _interval,
    frozen_winner,
    pending,
    recipe_block,
    report_path,
    selection_path,
    shown,
)

# The organisers' nine sections, in their order: the "Submission artifacts" list of the kickoff slides
# (https://hackathon.lexsi.ai/kickoff-slides/, docs/decisions.md item 119). Each heading keeps the slide's
# words, and the template numbers them 1 to 9. "Metrics and why" and "Cost and ROI" are the first and the
# last subsection of the seventh.
SECTIONS = (
    "Problem statement and approach",
    "Hackathon repos used and their features",
    "Models used and their licences",
    "Datasets and benchmarks used and their licences",
    "Model artifacts, as an HF collection",
    "Data artifacts, as an HF collection",
    "Experiments run, findings and what else was tested",
    "With more credits",
    "Teammates",
)
# What follows the nine, unnumbered and in this order.
EXTRAS = ("Related work", "Reproduce", "Links")
MODEL_ARTIFACTS, DATA_ARTIFACTS, EXPERIMENTS = SECTIONS[4], SECTIONS[5], SECTIONS[6]
PERSON = "a person"


def section_ref(name: str) -> str:
    """ "section 7" for one of the nine numbered sections, by its heading: a reference in the text follows
    the list when a section moves."""
    return f"section {SECTIONS.index(name) + 1}"


def placeholders(text: str) -> list[str]:
    """The distinct ``[PENDING: ...]`` placeholders in a rendered text."""
    found = set(re.findall(r"\[PENDING: ([^\]]*)\]", text))
    if settings.PLACEHOLDER_NAMESPACE in text:
        found.add(
            f"the HF namespace in the repo links ({settings.PLACEHOLDER_NAMESPACE}: set LIFT_HF_NAMESPACE)"
        )
    return sorted(found)


def article_path() -> Path:
    return paths.REPO_ROOT / "hf" / "article.md"


def sources_path() -> Path:
    return paths.REPO_ROOT / "hf" / "article.sources.json"


def readme_path() -> Path:
    """The repository's README, which is the write-up the judges read (docs/decisions.md item 96: HF
    articles need a PRO account; item 119: the organisers confirmed that a markdown README in the GitHub
    repo counts as the article). ``lift publish article`` writes the article's text there too
    (``readme_text``), and the figures it shows beside it (``readme_figures``)."""
    return paths.REPO_ROOT / "README.md"


# A figure's link in the article: the PNG in the logs dataset ({logs}/resolve/main/figures/{name}.png).
FIGURE_LINK = re.compile(r"\]\(https?://[^)\s]*?/resolve/main/(figures/[^)\s/]+\.png)\)")


def seeds_label(seeds: object) -> str:
    """ "seed 1" for one seed, "seeds 1, 2, 3" for more (as lift.report.tables writes them)."""
    text = ", ".join(str(s) for s in seeds) if isinstance(seeds, (list, tuple)) else str(seeds)
    return f"seeds {text}" if "," in text else f"seed {text}"


def _seeds_word(text: str) -> str:
    """ "seed 1" or "seeds 1, 2" before the seeds a value pools; a pending line as it is."""
    return text if text.startswith(("[", "_")) else seeds_label(text)


def readme_text(text: str) -> str:
    """The article as the README: the same text without the YAML front matter an HF article opens with
    (GitHub would render it as a table above the title), and with each figure linked from figures/ beside
    the README (``readme_figures``) rather than from the logs dataset, which goes up only once every
    label is final (hf/README.md)."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            text = text[end + 5 :].lstrip("\n")
    return FIGURE_LINK.sub(r"](\1)", text)


def readme_figures() -> list[Path]:
    """Copy each figure P07 drew (outputs/figures/) into figures/ beside the README, which links them
    there: charts of aggregates, no output text. A figure that is not drawn has no link, and a copy an
    earlier run left of it is removed. The copies written."""
    folder = readme_path().parent / "figures"
    written = []
    for name in FIGURES:
        src, dst = paths.figures_dir() / f"{name}.png", folder / f"{name}.png"
        if src.is_file() and src.stat().st_size > 0:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            written.append(dst)
        elif dst.exists():
            dst.unlink()
    return written


def _team(cfg: dict) -> tuple[str, str]:
    team = cfg.get("team") or []
    if not team:
        return "[]", pending("teammates' names and HF usernames", "configs/publish.yaml team", PERSON)
    authors = "\n" + "\n".join(f"  - user: {m['hf']}" for m in team if m.get("hf"))
    lines = "\n".join(
        f"- {m.get('name', m.get('hf'))} ([@{m['hf']}](https://huggingface.co/{m['hf']}))"
        if m.get("hf")
        else f"- {m.get('name')}"
        for m in team
    )
    return authors, lines


def _thanks(cfg: dict) -> str:
    annotators = cfg.get("annotators")
    if annotators is None:
        return pending(
            "thanks to the annotators who agreed to be named", "configs/publish.yaml annotators", PERSON
        )
    if not annotators:  # [] thanks nobody: no line at all
        return ""
    return f"We thank {', '.join(str(a) for a in annotators)}, who labelled the validation samples."


def _not_run(facts: Facts) -> str:
    report = facts.report()
    if report is None or "not_run" not in report:
        return facts.report_value("/not_run", "text", what="the list of arms not run")
    items = report["not_run"] or []
    return "\n".join(f"- {item}" for item in items) if items else "Every pre-registered arm was run."


# ------------------------------------------------- sections 5 and 6: what is published, where and why

# A repo's access as the two sections say it. The text says every gated repo opens when its terms are
# accepted, so a repo that waits for a person's approval is refused (the organisers may count one as
# missing: docs/decisions.md item 110).
ACCESS_WORDS = {"public": "Public", "gated-auto": "Gated, automatic approval"}
# The access each line's reason is written for: another one in configs/publish.yaml needs another sentence.
ACCESS_OF = {
    "model": "gated-auto",
    "adapter": "gated-auto",
    "raw": "gated-auto",
    "repair_params": "public",
    "space": "public",
    "logs": "public",
    "prompts": "public",
    "sft": "public",
}
# How the repaired model's repo holds each method's weights (lift.publish.model).
MODEL_FORMS = {
    "wiseft": "saved as merged weights",
    "resta": "saved as merged weights",
    "cast": "saved as the adapter-merged model it wraps, with the steering artifacts and the wrapper that "
    "applies them",
}
MODEL_FORMS["castalways"] = MODEL_FORMS["cast"]
# An adapter's training data, by the candidate in its name (configs/train.yaml ``candidates``).
CANDIDATE_WORDS = {"aya": "the Aya Dataset's Hindi rows", "indicalign": "the IndicAlign `Dolly_T` sample"}
_ADAPTER_NAME = re.compile(r"adapter-(?P<candidate>.+)-(?:std|stress)-s(?P<seed>\d+)")
PARAMS_WHAT = (
    "Every frozen setting's parameters: the WiSE-FT lambda, the ReSta alpha, and the CAST vectors, condition "
    "and threshold"
)
PARAMS_WHY = "every repaired model's weights are gated, so on their own the parameters rebuild nothing"
# (title, what the repo holds, why it has its access) of each dataset the data collection can list, in the
# order section 6 lists them. The training sets are listed only once the collection holds them
# (docs/decisions.md item 118).
DATA_LINES = {
    "logs": (
        "Evaluation logs",
        "One row per judged output (its prompt id, labels, flags and the sha256 of the output), the "
        "aggregate tables and run manifests the numbers here are read from, the redacted AuditKIT records, "
        "the per-item scores and the figures",
        "it holds no output text and no judge text",
    ),
    "prompts": (
        "Translated prompt sets",
        "The translations we made, each with its English source and its native spot-check record: "
        "PolyRefuse's harmful and harmless prompts in Hindi, its harmless prompts in Chinese, and the "
        "OR-Bench-Hard sample in five languages",
        "prompts only, no model output",
    ),
    "sft": (
        "Training sets",
        "The Hindi training sets the adapters were trained on, each with CuratorKIT's manifest, rejected "
        "rows and provenance",
        "training rows of public datasets, no model output",
    ),
    "raw": (
        "Raw completions",
        "Every generation, judgement and utility verdict in full",
        "it is the only repo that holds the outputs in full, harmful ones included, so it sits behind a "
        "login and terms that keep it to safety research",
    ),
}


def _access(r: settings.RepoSpec) -> str:
    """A repo's access in words; refused when configs/publish.yaml gives it another access than the one
    the article's reason is written for."""
    if r.access != ACCESS_OF.get(r.kind) or r.access not in ACCESS_WORDS:
        raise PublishError(
            f"configs/publish.yaml repos.{r.kind}.access is {r.access}, and the article gives its reason for "
            f"{ACCESS_OF.get(r.kind, 'no repo of that kind')}: change what lift.publish.article says of it, "
            "with a dated docs/decisions.md entry"
        )
    return ACCESS_WORDS[r.access]


def _link(r: settings.RepoSpec) -> str:
    return f"[`{r.repo_id}`]({r.url})"


def _adapter_words(name: str, spec: object) -> str:
    """An unrepaired adapter as section 5 says it: its training data and recipe from its name
    (adapter-{candidate}-{std|stress}-s{seed}) and its seed from the registry."""
    from lift.publish.model import recipe_of

    m = _ADAPTER_NAME.fullmatch(name)
    recipe = recipe_of(name)
    if m is None or recipe is None:
        return f"The LoRA adapter `{name}`"
    seed = getattr(spec, "seed", None)
    data = CANDIDATE_WORDS.get(m["candidate"], f"the `{m['candidate']}` rows")
    return (
        f"A Hindi LoRA adapter trained on {data} with the {recipe} recipe, "
        f"seed {m['seed'] if seed is None else seed}"
    )


def model_artifacts(facts: Facts) -> str:
    """Section 5's list: one line per repo of the model collection, in the order ``lift publish
    collections`` lists them (``collect.plan``: the repaired model, the repair parameters, every registered
    adapter, the Space). Each line has the repo's link, what it is, its access as configs/publish.yaml sets
    it (``_access``) and the reason for that access. The repaired model's reason is the one its card and its
    access form give (``model.gate_reasons``); it is pending while the selection is not decided. Then where
    to read which adapters drifted."""
    from types import SimpleNamespace

    from lift.publish import model
    from lift.publish.collect import plan

    winner = frozen_winner(facts)
    adapters = {
        settings.repo("adapter", adapter=name).repo_id: (name, spec)
        for name, spec in model.registered_adapters().items()
    }
    lines = []
    for r in plan(facts)["models"]:
        access = _access(r)
        if r.kind == "model" and winner is None:
            what = facts._missing("the repaired model (the frozen winner)", selection_path(), P05)
            lines.append(f"- **Repaired model**: {what}")
        elif r.kind == "model":
            family = winner.split("-", 1)[0]
            # The registry entry names the method; before the winner is registered, its name does.
            spec = model._spec(winner) or SimpleNamespace(
                method="cast_always" if family == "castalways" else family
            )
            form = f", {MODEL_FORMS[family]}" if family in MODEL_FORMS else ""
            why = "; and ".join(model.gate_reasons(facts, spec))
            lines.append(
                f"- **Repaired model**: {_link(r)}. The frozen repair `{winner}` ({_method(winner)}){form}. "
                f"{access}, because {why}."
            )
        elif r.kind == "repair_params":
            lines.append(f"- **Repair parameters**: {_link(r)}. {PARAMS_WHAT}. {access}: {PARAMS_WHY}.")
        elif r.kind == "adapter" and r.repo_id in adapters:
            lines.append(
                f"- **Unrepaired adapter**: {_link(r)}. {_adapter_words(*adapters[r.repo_id])}. {access}: a "
                "research artifact whose safety may have drifted."
            )
        elif r.kind == "space":
            # What the page holds of the outputs is said in full: lift.publish.space cuts a few outputs of
            # every cell, whatever their label, so harmful-labelled ones are among them.
            lines.append(
                f"- **Space**: {_link(r)}. A static page of the results: the verdicts, the heatmaps, the "
                "utility table and a few example outputs for each variant, language and safety prompt set of "
                "the final split, each cut to its first words and marked `[redacted]`, whatever its label. "
                f"{access}: it loads no model and generates nothing, and it holds no more of any output than "
                "that cut."
            )
        else:
            raise PublishError(f"the model collection lists {r.repo_id}, which the article has no line for")
    if adapters:
        lines.append(f"\nThe verdicts in {section_ref(EXPERIMENTS)} say which adapters drifted.")
    return "\n".join(lines)


def _training_sets_repo(facts: Facts) -> settings.RepoSpec | None:
    """The training-sets dataset when the data collection lists it, else None: the article names it only
    once it is published (docs/decisions.md item 118)."""
    from lift.publish.collect import plan

    return next((r for r in plan(facts)["data"] if r.kind == "sft"), None)


def data_artifacts(facts: Facts) -> str:
    """Section 6's list: one line per dataset of the data collection (``collect.plan``), in ``DATA_LINES``'
    order, then the repair parameters, which sit in the model collection and are data a reader needs. Each
    line has the repo's link, what it holds, its access as configs/publish.yaml sets it (``_access``) and
    the reason for that access. A dataset the collection lists and ``DATA_LINES`` does not is refused."""
    from lift.publish.collect import plan

    repos = {r.kind: r for r in plan(facts)["data"]}
    unknown = sorted(r.repo_id for kind, r in repos.items() if kind not in DATA_LINES)
    if unknown:
        raise PublishError(
            f"the data collection lists {', '.join(unknown)}, which the article has no line for"
        )
    lines = []
    for kind, (title, what, why) in DATA_LINES.items():
        if kind in repos:
            lines.append(f"- **{title}**: {_link(repos[kind])}. {what}. {_access(repos[kind])}: {why}.")
    params = settings.repo("repair_params")
    lines.append(
        f"- **Repair parameters**: {_link(params)}. {PARAMS_WHAT}. They are in the model collection "
        f"({section_ref(MODEL_ARTIFACTS)}) and listed here as well, because they are data a reader needs to "
        f"apply a repair. {_access(params)}: {PARAMS_WHY}."
    )
    return "\n".join(lines)


def tool_chain(facts: Facts) -> dict[str, str]:
    """Section 2's opening sentence, the three tools as one chain, and the words that say where
    CuratorKIT's files are published. The chain is said either way and names only what is public. While
    the data collection does not list the training sets (``_training_sets_repo``; docs/decisions.md item
    118), it points at the adapters and the repaired model and at AuditKIT's records, and the CuratorKIT
    line says only that its files are kept. Once the collection lists them, the chain's first link is
    that dataset and the line says where the files are."""
    models, data = section_ref(MODEL_ARTIFACTS), section_ref(DATA_ARTIFACTS)
    repo = _training_sets_repo(facts)
    if repo is None:
        return {
            "tool_chain": "The three tools run as one chain: CuratorKIT builds the Hindi training sets, the "
            "LoRA adapters are trained on them, SafeTune repairs an adapter, and AuditKIT records every "
            f"reported rate. The adapters and the repaired model are in {models}, and AuditKIT's redacted "
            f"records are in the logs dataset ({data}).\n\n",
            "curatorkit_files": "",
        }
    return {
        "tool_chain": "The three tools run as one chain, and each link of it is published: CuratorKIT's "
        f"training sets ({_link(repo)}), the LoRA adapters trained on them and SafeTune's repair "
        f"({models}), and AuditKIT's records of every reported rate (the logs dataset, {data}).\n\n",
        "curatorkit_files": f" and published with the training sets ({_link(repo)})",
    }


# ------------------------------------------------------------------- section 4: the training data

TRAIN = "lift train"
# A training set's source and licence, by the candidate its folder under data/sft/ names; the sentence
# under the table (hf/templates/article.md) says why IndicAlign's rows carry Dolly's licence.
TRAINING_SOURCES = {
    "aya": "Aya Dataset, Hindi rows (Apache-2.0)",
    "indicalign": "IndicAlign `Dolly_T`, Hindi (CC-BY-SA-3.0 through Dolly)",
}
TRAINING_VARIANTS = {"dedup": "deduplicated", "raw": "not deduplicated"}
_SFT_PATH = re.compile(r"data/sft/(?P<candidate>[^/]+)/(?P<variant>[^/]+)/[^/]+")


def training_runs() -> dict[str, tuple[Path, dict] | None]:
    """{registered adapter: (its train manifest, the manifest's document)}, None for an adapter with no
    training run on record. An adapter's run is the one under its name whose adapter hash the registry
    pins (``lift.publish.datasets.train_runs``: outputs/manifests/train-*.json; ``datasets.runs_of``: a
    run that a retrain set aside records another hash, and one that never hashed an adapter is no run of
    a registered adapter)."""
    from lift.publish import datasets, model

    runs = datasets.train_runs()
    out: dict[str, tuple[Path, dict] | None] = {}
    for name, spec in sorted(model.registered_adapters().items()):
        own = datasets.runs_of(name, spec, runs)
        out[name] = own[-1] if own else None
    return out


def _count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def training_data(facts: Facts) -> str:
    """Section 4's training data: one row per training set (the file its ``lift train`` runs read: each
    manifest's ``data_path``), with its source and licence, the rows in the file, the rows trained when
    they are fewer, and the recipe and seeds trained on it. Every count, recipe and seed is read from the
    train manifest of a registered adapter's run (``training_runs``; the logs dataset carries the
    manifests under manifests/) and recorded in the sources. The runs of one set must have read the same
    file (``data_sha256``) and agree on its counts, and each run's rows trained must be its rows less the
    ones it dropped as unfit (``lift.train.lora``), or the table is refused. Pending while no adapter is
    registered, and for an adapter whose run is not on record."""
    runs = training_runs()
    manifests = paths.manifests_dir()
    if not runs:
        return f"_{facts._missing('the training sets of the registered adapters', manifests, TRAIN)}_"
    by_set: dict[str, list[tuple[str, Path, dict]]] = {}
    for name, run in runs.items():
        if run is not None:
            by_set.setdefault(str(run[1].get("data_path")), []).append((name, *run))

    def val(f: Path, pointer: str, kind: str, what: str) -> str:
        return facts.value(f, pointer, kind, what=what, stage=TRAIN)

    rows, fewer = [], False
    for data_path, members in sorted(by_set.items()):
        # Two builds of one path are two training sets, whether or not their counts are equal.
        hashes = {name: str(doc.get("data_sha256")) for name, _, doc in members}
        if len(set(hashes.values())) != 1:
            read = ", ".join(f"{name}: sha256 {hashes[name][:12]} ({shown(f)})" for name, f, _ in members)
            raise PublishError(
                f"the training runs of {data_path} read different files under that path, and the article "
                f"gives one row per training set: {read}"
            )
        counts = {(doc.get("n_rows"), doc.get("n_rows_trained")) for _, _, doc in members}
        said = ", ".join(
            f"{name}: {doc.get('n_rows')} rows, {doc.get('n_rows_trained')} trained, "
            f"{(doc.get('dropped_unfit') or {}).get('count')} dropped ({shown(f)})"
            for name, f, doc in members
        )
        if len(counts) != 1:
            raise PublishError(
                f"the training runs of {data_path} disagree on its row counts, and the article gives one "
                f"count per training set: {said}"
            )
        ((n, trained),) = counts
        dropped = {(doc.get("dropped_unfit") or {}).get("count") for _, _, doc in members}
        if not _count(n) or not _count(trained) or trained > n or dropped != {n - trained}:
            raise PublishError(
                f"the row counts of the training runs of {data_path} do not add up (rows trained must be "
                f"the rows less the ones dropped as unfit): {said}"
            )
        # Every run's own counts are recorded, so each seed's manifest is checked, and shown once.
        in_file, used = [], []
        by_recipe: dict[str, list[str]] = {}
        for name, f, _ in sorted(members, key=lambda x: (str(x[2].get("recipe")), x[2].get("seed") or 0)):
            in_file.append(val(f, "/n_rows", "int", f"the rows of the file {name} trained on"))
            used.append(val(f, "/n_rows_trained", "int", f"the rows {name} was trained on"))
            recipe = val(f, "/recipe", "text", f"the recipe of {name}")
            by_recipe.setdefault(recipe, []).append(val(f, "/seed", "int", f"the seed of {name}"))
        at = _SFT_PATH.fullmatch(data_path)
        source = TRAINING_SOURCES.get(at["candidate"], f"`{at['candidate']}`") if at else ""
        if at and at["variant"] in TRAINING_VARIANTS:
            source += f", {TRAINING_VARIANTS[at['variant']]}"
        fewer = fewer or trained < n
        rows.append(
            [
                f"`{at['candidate']}/{at['variant']}`" if at else f"`{data_path}`",
                source,
                in_file[0] if trained == n else f"{in_file[0]} ({used[0]} trained)",
                "; ".join(f"{recipe} recipe, {seeds_label(seeds)}" for recipe, seeds in by_recipe.items()),
            ]
        )
    blocks = []
    if rows:
        header = ["Training set", "Source and licence", "Rows", "Adapters trained on it"]
        note = (
            "The row counts are the ones each training run recorded in its manifest "
            "(`manifests/train-*.json` in the logs dataset)."
        )
        if fewer:
            note += (
                " Where fewer rows were trained than the file holds, the others were dropped before "
                "training: their prompt alone, with the chat template's preamble, reaches the training "
                "length and leaves no answer token to learn from."
            )
        blocks += [render.markdown_table(header, rows), note]
    absent = [name for name, run in runs.items() if run is None]
    if absent:
        runs_word = "run" if len(absent) == 1 else "runs"
        what = f"the training {runs_word} of {_and([f'`{name}`' for name in absent])}"
        blocks.append(f"_{facts._missing(what, manifests, TRAIN)}_")
    return "\n\n".join(blocks)


def python_requirement() -> str:
    """ "Python 3.12 or later", from pyproject.toml's ``requires-python`` (">=3.12"): the template types
    no version. Another form of requirement is shown as it is written."""
    import tomllib

    path = paths.REPO_ROOT / "pyproject.toml"
    try:
        spec = str(tomllib.loads(path.read_text(encoding="utf-8"))["project"]["requires-python"]).strip()
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise PublishError(f"pyproject.toml gives no requires-python for the article ({exc})") from exc
    m = re.fullmatch(r">=\s*(\d+(?:\.\d+)*)", spec)
    return f"Python {m.group(1)} or later" if m else f"Python {spec}"


# ---------------------------------------------------------------- what each check said of each adapter

PARITY = 0.5  # a pairwise win rate at which neither side is preferred


def _direction(axis: object, reference: float = 0.0) -> str | None:
    """Where an axis {est, lo, hi} of the report stands against ``reference``: "higher" or "lower" when its
    95% interval excludes it, "same" when it does not; None when the axis was not run."""
    if axis is None:
        return None
    try:
        lo, hi = float(axis["lo"]), float(axis["hi"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise PublishError(f"{shown(report_path())}/verdicts: an axis without an interval: {exc}") from exc
    return "higher" if lo > reference else "lower" if hi < reference else "same"


# (axis key, the sentence for each direction). Lower held-out loss is a better fit of the references.
_SAID = {
    "heldout": {
        "lower": "held-out loss better than stock's",
        "higher": "held-out loss worse than stock's",
        "same": "no clear change in held-out loss",
        None: "held-out loss not run",
    },
    "belebele_hi": {
        "higher": "Belebele-hi accuracy higher than stock's",
        "lower": "Belebele-hi accuracy lower than stock's",
        "same": "no clear change in Belebele-hi accuracy",
        None: "Belebele-hi not run",
    },
    "final_win_rate": {
        "higher": "its answers preferred to stock's more often than not",
        "lower": "stock's answers preferred to its own more often than not",
        "same": "no clear preference between its answers and stock's",
        None: "the final win rate not run",
    },
    "drift": {
        "higher": "Hindi harmful rate higher than stock's",
        "lower": "Hindi harmful rate lower than stock's",
        "same": "no clear change in the Hindi harmful rate",
        None: "the Hindi harmful rate not run",
    },
    "overrefusal_hard_hi": {
        "higher": "more refusals of hard-benign Hindi prompts than stock",
        "lower": "fewer refusals of hard-benign Hindi prompts than stock",
        "same": "no clear change in refusals of hard-benign Hindi prompts",
        None: "hard-benign refusals not run",
    },
}


# The Hindi harmful rate against stock with flagged outputs out of the harmful count (decisions.md 87, 93).
INTACT_WORDS = {"higher": "higher", "lower": "lower", "same": "no clear change"}
INTACT_READING = "counting only outputs that trip no unclear flag"
# A rise on every label that the intact reading does not show: how the summary says the second reading.
INTACT_AGAINST_A_RISE = {"same": "but not clearly once", "lower": "and fell once"}


def adapter_checks(facts: Facts) -> str:
    """Each adapter on the checks a team runs before shipping (held-out loss, a benchmark), on pairwise
    utility and on safety, in words, from the report's verdict rows (``axes``: P07's
    lift.report.analysis.adapter_axes; decisions items 82 and 86). A direction is said only where the
    axis's 95% interval excludes no change (parity for the win rate); it repeats no number (the adapters
    table below holds them, recorded in the sources). Then what the checks add up to, each sentence only
    for the adapters whose intervals support it: a better loss with a higher harmful rate, a better loss
    with a lost pairwise comparison, a lower harmful rate with more refusals of benign prompts.

    Beside the harmful rate, the same difference counting a harmful label only on an output that trips no
    unclear flag (``drift_intact``; decisions.md 87 and 93), whichever way it points. Where the rate rose
    on every label and that reading does not show it, the adapter is left out of "the safety test said no"
    and gets a sentence that says what the labels leave open."""
    verdicts = facts.raw(report_path(), "/verdicts")
    if verdicts is None:
        return facts.report_value("/verdicts", "text", what="each adapter's checks")
    rows = [v for v in verdicts if isinstance(v, dict) and isinstance(v.get("axes"), dict)]
    if not rows:
        return f"_{facts._missing('each adapter on its own axes', report_path(), P07)}_"
    lines = []
    where: dict[str, dict[str, str | None]] = {}
    for v in rows:
        axes = v["axes"]
        name = f"`{v.get('adapter')}`"
        d = where[name] = {
            key: _direction(axes.get(key), PARITY if key == "final_win_rate" else 0.0) for key in _SAID
        }
        role = VERDICT_ROLES.get(v.get("kind"))
        who = f"{v.get('recipe') or 'standard'} recipe" + (f", {role}" if role else "")
        drift = _SAID["drift"][d["drift"]]
        if d["drift"] == "higher" and (axes.get("drift") or {}).get("meaningful"):
            drift += ", over Gate 2's bar"
        # The same difference with outputs that trip an unclear flag left out of the harmful count
        # (docs/decisions.md 87 and 93): said beside the pre-registered reading, whichever way it points.
        intact = _direction(axes.get("drift_intact"), 0.0)
        d["drift_intact"] = intact
        if intact is not None and d["drift"] is not None:
            also = "also " if intact == d["drift"] else ""
            drift += f" ({also}{INTACT_WORDS[intact]} {INTACT_READING})"
        useful = f"; Gate 1: {v['usefulness_verdict']}" if v.get("usefulness_verdict") else ""
        lines.append(
            f"- {name} ({who}). The usual checks: {_SAID['heldout'][d['heldout']]}; "
            f"{_SAID['belebele_hi'][d['belebele_hi']]}. Pairwise utility: "
            f"{_SAID['final_win_rate'][d['final_win_rate']]}{useful}. Safety: {drift}; "
            f"{_SAID['overrefusal_hard_hi'][d['overrefusal_hard_hi']]}."
        )

    def those(**want: str) -> list[str]:
        return [name for name, d in where.items() if all(d[k] == v for k, v in want.items())]

    summary = []
    rose = those(heldout="lower", drift="higher")
    # A rise that does not hold once flagged outputs are left out is said apart, with what it leaves open.
    shipped = [name for name in rose if where[name]["drift_intact"] not in INTACT_AGAINST_A_RISE]
    if shipped:
        summary.append(
            f"For {_and(shipped)}, the held-out loss improved while the Hindi harmful rate rose: the check "
            "a team runs before shipping said yes, and the safety test said no."
        )
    for reading, words in INTACT_AGAINST_A_RISE.items():
        unsettled = [name for name in rose if where[name]["drift_intact"] == reading]
        if unsettled:
            summary.append(
                f"For {_and(unsettled)}, the held-out loss improved and the Hindi harmful rate rose on the "
                f"pre-registered reading, {words} outputs that trip an unclear flag are left out: these "
                "labels do not settle how much of that rise is harmful content and how much is degenerate "
                "text the judge called unsafe."
            )
    disagree = those(heldout="lower", final_win_rate="lower")
    if disagree:
        summary.append(
            f"For {_and(disagree)}, the two readings of usefulness disagree: a better fit of the held-out "
            "references, and answers the pairwise judge preferred less often than stock's."
        )
    refusing = those(drift="lower", overrefusal_hard_hi="higher")
    if refusing:
        summary.append(
            f"For {_and(refusing)}, the lower harmful rate comes with more refusals of benign prompts: "
            "part of that safety is refusing more."
        )
    return "\n".join(lines) + ("\n\n" + " ".join(summary) if summary else "")


def comparison_adapter(facts: Facts) -> str:
    """The adapter that carries the repair comparison, as the report names it: the first verdict row (the
    selection's adapter), with its recipe; pending while the report is not there."""
    verdicts = facts.raw(report_path(), "/verdicts")
    if not verdicts:
        return facts.report_value("/verdicts/0/adapter", "text", what="the adapter of the repair comparison")
    name = facts.value(report_path(), "/verdicts/0/adapter", "text", what="adapter", stage=P07)
    recipe = (
        facts.value(report_path(), "/verdicts/0/recipe", "text", what="recipe", stage=P07)
        if verdicts[0].get("recipe")
        else "standard"
    )
    return f"`{name}` ({recipe} recipe)"


FAMILY_NAMES = {"wiseft": "WiSE-FT", "resta": "ReSta", "cast": "CAST"}
# The rule's conditions as selection.json's rows flag them, in the order the article names them.
CONDITIONS = (
    ("belebele_ok", "Belebele-hi"),
    ("overrefusal_ok", "hard-benign over-refusal"),
    ("gain_ok", "the gain to keep"),
    ("drop_ok", "a lower harmful rate than the adapter's"),
)


def secondary_path() -> Path:
    return paths.metrics_dir() / "selection_secondary.json"


def _plural(n: int, noun: str = "setting") -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def dev_selection(facts: Facts) -> str:
    """What the selection rule decided on seed-1 dev, in words, from selection.json: how many settings met
    every condition, the pick with its Hindi harmful rate against the adapter's and stock's, which condition
    the others failed, and each other family's frozen setting with its verdict. Then what the same rule
    decides with the gain read on held-out loss (selection_secondary.json, docs/decisions.md item 73), and
    the share of the adapter's held-out-loss gain the frozen winner keeps on dev: the primary rule cannot
    see that share when the adapter has no win-rate gain over stock to keep. Every number is read through
    ``facts`` (recorded in the sources); nothing is said that the two files do not hold. Pending while
    selection.json is not decided; the secondary part says so, without a placeholder, while its file is
    not decided (it freezes nothing and blocks nothing)."""
    sel = _decided(facts)
    if sel is None:
        return facts.value(
            selection_path(), "/overall/winner", "text", what="the dev selection (lift select)", stage=P05
        )
    path = selection_path()
    rows = list(sel.get("table") or [])
    index = {r["name"]: i for i, r in enumerate(rows)}
    winner = str(sel["overall"]["winner"])
    if winner not in index:
        raise PublishError(f"{shown(path)}: the overall winner {winner} has no row in its table")

    def val(pointer: str, kind: str, what: str) -> str:
        return facts.value(path, pointer, kind, what=what, stage=P05)

    n_ok = sum(bool(r.get("eligible")) for r in rows)
    verdict = val("/overall/verdict", "text", "the selection's verdict")
    harmful = val(f"/table/{index[winner]}/harmful_hi", "pct", "the winner's dev harmful rate")
    adapter = val("/adapter_scores/harmful_hi", "pct", "the adapter's dev harmful rate")
    stock = val("/stock/harmful_hi", "pct", "stock's dev harmful rate")
    if n_ok:
        lead = (
            f"On seed-1 dev, {n_ok} of the {_plural(len(rows))} met every condition of the rule. Its pick is "
            f"`{winner}` (verdict **{verdict}**)"
        )
    else:
        lead = (
            f"On seed-1 dev, none of the {_plural(len(rows))} met every condition of the rule, so it froze "
            f"the setting with the fewest violations and then the largest drop: `{winner}` (verdict "
            f"**{verdict}**)"
        )
    parts = [
        f"{lead}, with a Hindi harmful rate of {harmful} against the adapter's {adapter} and stock's {stock}."
    ]
    failed = [
        f"{label} ({_plural(n)})"
        for key, label in CONDITIONS
        if (n := sum(r.get(key) is False for r in rows))
    ]
    if failed:
        parts.append(
            f"Conditions not met, with the number of settings that missed each: {'; '.join(failed)}."
        )
    own_win = (sel.get("adapter_scores") or {}).get("win_rate")
    scaled = [r for r in rows if r.get("method") == "wiseft"]
    if (
        isinstance(own_win, (int, float))
        and own_win <= 0.5
        and scaled
        and all(r.get("gain_ok") is True for r in scaled)
    ):
        # Said only when selection.json shows it (docs/decisions.md 72-73, 89): an adapter at or under
        # parity has no win-rate gain to keep, the floor sits under its own win rate, and every setting
        # that scales it toward stock meets the floor.
        win = val("/adapter_scores/win_rate", "pct", "the adapter's dev win rate against stock")
        floor = val(f"/table/{index[scaled[0]['name']]}/min_win_rate", "pct", "the rule's win-rate floor")
        parts.append(
            f"The adapter's own answers were preferred to stock's in {win} of the pairs, so it has no "
            f"win-rate gain to keep: the floor is {floor}, and every WiSE-FT setting ({len(scaled)}) met it. "
            "On this adapter that condition does not bind along the path toward stock; the reading on "
            "held-out loss below is the one that can."
        )
    gated = [r for r in rows if r.get("method") == "cast"]
    own = (sel.get("adapter_scores") or {}).get("belebele_hi")
    if (
        gated
        and isinstance(own, (int, float))
        and all(
            r.get("belebele_ok") is False
            and isinstance(r.get("belebele_hi"), (int, float))
            and abs(r["belebele_hi"] - own) < 1e-12
            for r in gated
        )
    ):
        # Said only when selection.json shows it: every conditional CAST setting has the unrepaired adapter's
        # own Belebele-hi accuracy, and that accuracy is outside the rule's margin (docs/decisions.md 88).
        accuracy = val("/adapter_scores/belebele_hi", "pct", "the adapter's own dev Belebele-hi accuracy")
        parts.append(
            f"Every conditional CAST setting ({len(gated)}) has the unrepaired adapter's own Belebele-hi "
            f"accuracy, {accuracy}, which is outside the rule's margin: on that condition CAST was decided "
            "by the adapter it repairs, not by what it does to harmful prompts."
        )
    families = sel.get("families") if isinstance(sel.get("families"), Mapping) else {}
    others = []
    for family, name in sel["frozen"].items():
        if str(name) == winner:
            continue
        word = (families.get(family) or {}).get("verdict")
        others.append(
            f"`{name}` for {FAMILY_NAMES.get(str(family), str(family))}"
            + (f" (verdict **{word}**)" if word else "")
        )
    if others:
        parts.append(f"The other families' frozen settings: {_and(others)}.")
    parts.append(_secondary_selection(facts, winner, str(sel.get("adapter"))))
    return " ".join(x for x in parts if x)


def _secondary_selection(facts: Facts, winner: str, adapter: str) -> str:
    """The secondary selection's part of ``dev_selection``: what the rule decides with the gain read on
    held-out loss, and the share of that gain the frozen winner keeps on dev. A file made for another
    adapter, or from other dev numbers than selection.json's (its own ``primary`` pick is not the frozen
    winner), is refused: the two files would contradict each other in one paragraph."""
    path = secondary_path()
    doc = facts.doc(path)
    lead = (
        "With the gain read on held-out loss instead of the win rate (the secondary selection, which freezes "
        "nothing)"
    )
    if doc is None:
        return f"{lead}: not decided, the dev held-out scores of the settings are not there."
    status = doc.get("status")
    if status == "undefined":
        return f"{lead}: undefined, the adapter fits the held-out references no better than stock."
    if status != "decided":
        return f"{lead}: not decided, the dev held-out scores of some settings are not there."
    again = (
        f"run `lift select --adapter {adapter}` again (it rewrites the file and leaves the recipe as it is)"
    )
    if doc.get("adapter") != adapter:
        raise PublishError(
            f"{shown(path)} is the secondary selection of {doc.get('adapter')}, not {adapter}: {again}"
        )
    if (doc.get("primary") or {}).get("winner") != winner:
        raise PublishError(
            f"{shown(path)} was made from other dev numbers than {shown(selection_path())}: on its numbers "
            f"the primary rule picks {(doc.get('primary') or {}).get('winner')}, not the frozen {winner}; "
            f"{again}"
        )

    def val(pointer: str, kind: str, what: str) -> str:
        return facts.value(path, pointer, kind, what=what, stage=P05)

    rows = list(doc.get("table") or [])
    index = {r["name"]: i for i, r in enumerate(rows)}
    pick = str((doc.get("overall") or {}).get("winner"))
    if pick not in index or winner not in index:
        raise PublishError(
            f"{shown(path)}: its table has no row for {pick} or for the frozen winner {winner}"
        )
    word = val("/overall/verdict", "text", "the secondary selection's verdict")
    kept = val(
        f"/table/{index[winner]}/heldout_retained", "pct", "the held-out gain the frozen winner keeps on dev"
    )
    need = val("/rules/min_gain_retained", "pct", "the share of the gain the rule asks for")
    n_ok = sum(bool(r.get("eligible")) for r in rows)
    same = "the same setting" if pick == winner else f"`{pick}`"
    if n_ok:
        out = (
            f"{lead}, {n_ok} of the {_plural(len(rows))} met every condition and the same rule picks {same} "
            f"(verdict **{word}**)."
        )
    else:
        out = (
            f"{lead}, none of the {_plural(len(rows))} met every condition: the fewest violations and then "
            f"the largest drop give {same} (verdict **{word}**)."
        )
    share = rows[index[winner]].get("heldout_retained")
    if isinstance(share, (int, float)) and share > 1:
        # Over 100%: scaled down, the adapter fits the held-out references better than at full strength.
        out += (
            f" On dev the frozen winner's held-out loss is lower than the full adapter's: its gain over "
            f"stock is {kept} of the adapter's own, where the rule asks for {need}."
        )
    else:
        out += (
            f" The frozen winner keeps {kept} of the adapter's held-out-loss gain on dev; the rule asks for "
            f"{need}."
        )
    if pick != winner:
        other = val(
            f"/table/{index[pick]}/heldout_retained",
            "pct",
            "the held-out gain the secondary pick keeps on dev",
        )
        out += f" `{pick}` keeps {other}."
    return out


PRIMARY_EFFECTS = [
    ("Drift", "drift"),
    ("Repair", "repair"),
    ("Red-teaming drift", "redteam_drift"),
    ("Red-teaming repair", "redteam_repair"),
]


def _judge_entries() -> tuple[dict, dict]:
    """(the Qwen3Guard entry, the frontier entry) among judges.safety_primary and safety_second."""
    entries = [config.get(f"judges.{k}") for k in ("safety_primary", "safety_second")]
    qwen = [e for e in entries if "provider" not in e and "Qwen3Guard" in str(e.get("id"))]
    frontier = [e for e in entries if "provider" in e]
    if len(qwen) != 1 or len(frontier) != 1:
        raise PublishError(
            "configs/project.yaml judges: expected one Qwen3Guard and one frontier safety judge"
        )
    return qwen[0], frontier[0]


def primary_kind(facts: Facts) -> str | None:
    """Which safety judge final_report.json's judgements are by: "qwen" or "frontier" (the kinds of
    ``_judge_entries``), None while the report or its ``primary_judge`` is not there. An id that is neither
    judge of configs/project.yaml is refused."""
    judge_id = facts.raw(report_path(), "/primary_judge")
    if judge_id is None:
        return None
    qwen, frontier = _judge_entries()
    prefixes = {"qwen": f"{qwen['id']}@", "frontier": f"{frontier['provider']}/{frontier['model']}@"}
    kind = next((k for k, pre in prefixes.items() if str(judge_id).startswith(pre)), None)
    if kind is None:
        raise PublishError(
            f"{shown(report_path())} primary_judge {str(judge_id)[:80]!r} is neither safety judge in "
            "configs/project.yaml"
        )
    return kind


def judges(facts: Facts) -> dict[str, str]:
    """The judges by kind, and which one was primary according to final_report.json."""
    qwen, frontier = _judge_entries()
    names = {
        "qwen": str(qwen["id"]).split("/")[-1],
        "frontier": str(frontier["model"]),
    }
    shown_id = facts.report_value("/primary_judge", "text", what="the primary judge (final judgements)")
    kind = primary_kind(facts)
    if kind is None:
        role = f"The primary safety judge, behind every reported number: {shown_id}."
    else:
        role = (
            f"The primary safety judge, behind every reported number, was **{names[kind]}** "
            f"(judge id `{shown_id}` in the final judgements); the other judged a random sample of outputs "
            "and the validation items."
        )
    utility = config.get("judges.utility")
    named = f"{utility.get('model')} ({utility.get('provider')} API)"
    same = (utility.get("provider"), utility.get("model")) == (
        frontier.get("provider"),
        frontier.get("model"),
    )
    return {
        "qwen_judge": f"{names['qwen']}** (`{qwen['id']}`, revision `{qwen.get('revision')}`)",
        "frontier_judge": f"{names['frontier']}** ({frontier['provider']} API)",
        # "also" only when configs/project.yaml makes the utility judge the frontier safety model
        "utility_judge": f"{named} is also the pairwise utility judge."
        if same
        else f"The pairwise utility judge was {named}.",
        "judge_roles": role,
        "judge_rule": facts.config(
            "project", "judges.decision_rule.switch_primary_if_frontier_agreement_exceeds_by", "points"
        ),
    }


BT_CHECK = "back-translated"  # P07's judge_sensitivity rows of the ar/th check start with this


def backtranslation(facts: Facts) -> dict[str, str]:
    """The ar/th back-translation sentences, only as far as final_report.json supports them: claimed when
    P07's judge_sensitivity table has back-translation rows, "not run" when it has none, pending while the
    table is not there."""
    rows = facts.table_rows("judge_sensitivity")
    if rows is None:
        line = facts._missing("whether the Arabic and Thai back-translation check ran", report_path(), P07)
        return {"nllb": f" ({line})", "backtranslation_limit": line}
    if any(str(r.get("check", "")).startswith(BT_CHECK) for r in rows):
        return {
            "nllb": " and **NLLB-200-distilled-600M** (CC-BY-NC-4.0) for the Arabic and Thai "
            "back-translation check",
            "backtranslation_limit": "For Arabic and Thai, a sample of the outputs was also judged after "
            "back-translation into English (judge sensitivity table).",
        }
    return {
        "nllb": "; the Arabic and Thai back-translation check was not run (see Not run)",
        "backtranslation_limit": "The Arabic and Thai back-translation check was not run, so those "
        "languages rest on the judge alone.",
    }


# P07's per_language row labels of the harmful set (the red-teaming row has its own design count).
LANG_NAMES = ("English", "Hindi", "Chinese", "Spanish", "Arabic", "Thai")


# The benign sets P07's cells table names, as the note says them (stock's final cells are the reference).
BENIGN_SETS = {"easy_benign": "easy benign", "hard_benign": "hard benign"}


def counts_note(facts: Facts) -> str:
    """Whether the final counts evaluated match the design counts from configs/ (review XB-15): the
    harmful prompts per language (P07's per-language table, ``n``: stock's final cell), the easy and hard
    benign prompts per language (stock's rows of the cells table), the utility pairs (each variant's row
    of the utility table) and the Belebele items per language (the belebele table, stock's). Each count
    that differs is named, with both numbers sourced; a count not there (an arm not run, a table not
    written) is not checked. With a leading space; "" while no table is there (each is pending itself)."""
    checks = []  # (pointer of the count, what it is, how its row is named, configs size)
    rows = facts.table_rows("per_language")
    for i, r in enumerate(rows or []):
        if r.get("lang") in LANG_NAMES:
            checks.append((f"/tables/per_language/rows/{i}", "harmful prompts in", "lang", "harmful"))
    for i, r in enumerate(facts.table_rows("cells") or []):
        if r.get("variant") == "stock" and r.get("set") in BENIGN_SETS:
            what = f"{BENIGN_SETS[r['set']]} prompts in"
            checks.append((f"/tables/cells/rows/{i}", what, "lang", r["set"]))
    for i, _ in enumerate(facts.table_rows("utility") or []):
        checks.append((f"/tables/utility/rows/{i}", "utility pairs of", "variant", "utility"))
    for i, _ in enumerate(facts.table_rows("belebele") or []):
        checks.append((f"/tables/belebele/rows/{i}", "Belebele items in", "lang", "belebele"))
    if not checks:
        return ""
    differ = []
    for at, what, label, size in checks:
        n = facts.raw(report_path(), f"{at}/n")
        if n is None or n == config.get(f"sizes.{size}.final"):
            continue
        named = facts.report_value(f"{at}/{label}", "text", what="the row a count belongs to")
        count = facts.report_value(f"{at}/n", "int", what="a count evaluated")
        designed = facts.config("project", f"sizes.{size}.final", "int")
        differ.append(f"{what} {named} ({count}) against {designed} designed")
    if not differ:
        return (
            " The final counts evaluated match this design in the results tables (harmful, easy and hard "
            "benign prompts per language, utility pairs, Belebele items)."
        )
    return (
        f" The final counts evaluated differ from this design: {'; '.join(differ)}. The results count what "
        "was evaluated."
    )


# final_report.json tables the article never places: P07's index of every final cell.
UNPLACED = {"cells"}


def judge_on_benign(facts: Facts) -> str:
    """What the judge's harmful label does on benign prompts (docs/decisions.md items 95 and 98), read
    from P07's table of every final cell, which the article does not place: on the hard-benign Hindi
    prompts, the share of stock's answers and of the comparison adapter's (seed 1) labelled harmful, with
    the share on outputs that trip no unclear flag, against the same shares on the easy-benign ones. Their
    source wrote those prompts as benign requests, which says nothing of the answers (the readers of
    validation round 1 called a fifth of the validated ones harmful), so the sentence says what the label
    is about there and points at the readers' labels of such answers. Only what the table holds is said:
    an adapter or a set without its cell is left out, and with no hard-benign Hindi cell of stock the
    sentence says the cells were not run. Pending while the report or its cells table is not there.
    """
    rows = facts.table_rows("cells")
    if rows is None:
        what = "the judge's harmful label on answers to benign prompts"
        return facts.report_value("/tables/cells/rows", "text", what=what)
    index = {(r.get("variant"), r.get("set"), r.get("lang")): i for i, r in enumerate(rows)}
    adapter = facts.raw(report_path(), "/verdicts/0/adapter")

    def rate(variant: object, set_name: str, key: str = "harmful_rate") -> str | None:
        i = index.get((variant, set_name, "hi"))
        if i is None or rows[i].get(key) is None:
            return None
        what = f"the judge's harmful label on {BENIGN_SETS[set_name]} Hindi prompts"
        return facts.report_value(f"/tables/cells/rows/{i}/{key}", "pct", what=what)

    def pair(set_name: str, key: str = "harmful_rate") -> tuple[str | None, str | None]:
        """(stock's rate, the adapter's), each None when its cell or the rate is not there."""
        return rate("stock", set_name, key), rate(adapter, set_name, key)

    stock_hard, adapter_hard = pair("hard_benign")
    if stock_hard is None:
        return (
            "The hard-benign Hindi cells of the final split were not run, so the judge's harmful label on "
            "answers to benign prompts is not reported."
        )
    with_adapter = adapter_hard is not None

    def said(values: tuple[str | None, str | None]) -> str | None:
        """The pair as the sentence lists it (stock's alone without the adapter's cell); None when a
        value the sentence would need is missing."""
        stock, other = values
        if stock is None or (with_adapter and other is None):
            return None
        return f"{stock} and {other}" if with_adapter else stock

    out = (
        "The judge's harmful label also falls on answers to benign prompts: on the hard-benign Hindi "
        f"prompts of the final split it is on {stock_hard} of stock's answers"
    ) + (f" and {adapter_hard} of `{adapter}`'s" if with_adapter else "")
    intact = said(pair("hard_benign", "harmful_intact_rate"))
    if intact:
        out += f" ({intact} counting only outputs that trip no unclear flag)"
    easy = said(pair("easy_benign"))
    if easy:
        out += f", against {easy} on the easy-benign ones"
    return (
        f"{out}. Their source (OR-Bench-Hard) wrote those prompts as benign requests that look harmful, so "
        "there the label is about the answer and not the request. How often readers called such answers "
        'harmful is in the table by prompt set under "The judge against native readers".'
    )


def _is_vs_wiseft(name: str, spec: dict) -> bool:
    words = f"{name} {spec.get('title', '')}".lower().replace("-", "")
    return "wiseft" in words


# An interval P07 wrote as text (a ``text`` column, as repair_vs_wiseft's gain kept): "+1.0 pp [-2.0, +4.0]".
_TEXT_INTERVAL = re.compile(
    r"^\s*[+-]?\d+(?:\.\d+)?\s*pp\s*\[\s*(?P<lo>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<hi>[+-]?\d+(?:\.\d+)?)\s*\]"
)
# Which way is better for a "repair - WiSE-FT" column, read from its key and label: a higher harmful or
# refusal rate is worse; a higher gain kept, utility or accuracy is better. None: no direction is claimed.
_WORSE_IF_HIGHER = ("harm", "refus")
_BETTER_IF_HIGHER = ("gain", "kept", "retain", "utility", "win", "belebele", "accuracy")
_NOT_A_CONTRAST = {"seeds", "variant", "reference"}


def _column_name(column: dict) -> str:
    """A contrast column as the sentence names it: its label without the " - WiSE-FT..." part."""
    label = str(column.get("label") or column["key"])
    return '"' + re.split(r" - |: ", label)[0].strip() + '"'


def _higher_is_better(column: dict) -> bool | None:
    words = f"{column['key']} {column.get('label', '')}".lower()
    if any(w in words for w in _WORSE_IF_HIGHER):
        return False
    if any(w in words for w in _BETTER_IF_HIGHER):
        return True
    return None


def _bounds(value: object) -> tuple[float, float] | str:
    """(lo, hi) of an interval cell (an Interval, or P07's interval text), or "not run" / "undefined"."""
    if value is None:
        return "not run"
    if isinstance(value, str):
        if value == UNDEFINED or value.startswith(UNDEFINED + ":"):
            return "undefined"
        m = _TEXT_INTERVAL.match(value)
        if m is None:
            raise ValueError(f"not an interval: {value!r}")
        return float(m["lo"]), float(m["hi"])
    _, lo, hi = _interval(value)
    return lo, hi


def _is_contrast(column: dict, rows: list[dict], who: dict) -> bool:
    """An interval column: a ``*_ci`` format, or a text column whose cells are P07's interval text."""
    if column is who or column["key"] in _NOT_A_CONTRAST:
        return False
    if str(column.get("format", "")).endswith("_ci"):
        return True
    cells = [r.get(column["key"]) for r in rows]
    return any(isinstance(c, str) and _TEXT_INTERVAL.match(c) for c in cells)


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def vs_wiseft_sentence(name: str, spec: dict) -> str:
    """The answer to "does any repair beat simply scaling the adapter down?" from P07's descriptive table
    (review XB-07, XB2-06): for each family, with the seeds its row names, every interval column read
    against zero, the direction said (better or worse than WiSE-FT, by the column's meaning, or no clear
    difference where the interval includes zero), and a cell not run or undefined said so. It repeats no
    number but the seeds (the table below holds them, recorded in the sources) and gives no verdict: the
    contrast is descriptive. "" when the table has no text column and no interval column to read."""
    columns = spec.get("columns") or []
    rows = list(spec.get("rows") or [])
    who = next((c for c in columns if c.get("format", "text") == "text"), None)
    if who is None:
        return ""
    contrasts = [c for c in columns if _is_contrast(c, rows, who)]
    seeds_col = next((c for c in columns if c["key"] == "seeds"), None)
    if not contrasts or not rows:
        return ""
    sentences = []
    for row in rows:
        groups: dict[str, list[str]] = {}
        for c in contrasts:
            try:
                bounds = _bounds(row.get(c["key"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise PublishError(f"{shown(report_path())}/tables/{name}: {exc}") from exc
            col = _column_name(c)
            if isinstance(bounds, str):
                groups.setdefault(bounds, []).append(col)
                continue
            lo, hi = bounds
            if lo <= 0 <= hi:
                groups.setdefault("same", []).append(col)
                continue
            higher = lo > 0
            better = _higher_is_better(c)
            way = "higher" if higher else "lower"
            if better is None:
                groups.setdefault(way, []).append(col)
            else:
                groups.setdefault("better" if better == higher else "worse", []).append(f"{col} ({way})")
        parts = [
            f"better than WiSE-FT on {_and(groups['better'])}" if "better" in groups else "",
            f"worse than WiSE-FT on {_and(groups['worse'])}" if "worse" in groups else "",
            f"higher than WiSE-FT on {_and(groups['higher'])}" if "higher" in groups else "",
            f"lower than WiSE-FT on {_and(groups['lower'])}" if "lower" in groups else "",
            f"no clear difference on {_and(groups['same'])}" if "same" in groups else "",
            f"{_and(groups['not run'])} not run" if "not run" in groups else "",
            f"{_and(groups['undefined'])} undefined" if "undefined" in groups else "",
        ]
        seeds = row.get(seeds_col["key"]) if seeds_col else None
        family = str(row.get(who["key"])) + (f" ({seeds_label(seeds)})" if seeds not in (None, "") else "")
        sentences.append(f"{'for' if not sentences else 'For'} {family}: {'; '.join(p for p in parts if p)}.")
    return (
        "Does any repair beat simply scaling the adapter down? Descriptively (paired, unadjusted, not "
        "pre-registered; each column of the table below is the repair minus WiSE-FT, and a difference counts "
        "only where its 95% interval excludes zero): " + " ".join(sentences) + "\n\n"
    )


def _is_vs_stock(name: str, spec: dict) -> bool:
    return name == "repair_vs_stock"


def vs_stock_sentence(name: str, spec: dict) -> str:
    """The answer to "is safety restored?" from P07's descriptive table of the frozen repair against stock
    (docs/decisions.md 90): the Hindi harmful rate first, then the other languages, each read from its
    interval against zero (lower than stock's, higher, so part of the drift remains, or no clear
    difference), and a row not run said so. It repeats no number (the table below holds them, recorded in
    the sources) and gives no verdict. "" when the table has no row or no such column."""
    rows = list(spec.get("rows") or [])
    column = next((c for c in spec.get("columns") or [] if c.get("key") == "change"), None)
    if not rows or column is None:
        return ""
    groups: dict[str, list[str]] = {}
    hindi = None
    for row in rows:
        try:
            bounds = _bounds(row.get("change"))
        except (KeyError, TypeError, ValueError) as exc:
            raise PublishError(f"{shown(report_path())}/tables/{name}: {exc}") from exc
        if isinstance(bounds, str):
            way = bounds
        else:
            lo, hi = bounds
            way = "lower" if hi < 0 else "higher" if lo > 0 else "same"
        label = str(row.get("lang"))
        if label == "Hindi":
            hindi = way
        else:
            groups.setdefault(way, []).append(label)
    said = {
        "lower": "lower than stock's",
        "higher": "higher than stock's, so part of the drift remains",
        "same": "not clearly different from stock's",
        "not run": "not run",
        "undefined": "undefined",
    }
    parts = []
    if hindi is not None:
        parts.append(f"on Hindi the frozen repair's harmful rate is {said[hindi]}.")
    others = [
        f"lower than stock's on {_and(groups['lower'])}" if "lower" in groups else "",
        f"higher on {_and(groups['higher'])}" if "higher" in groups else "",
        f"not clearly different on {_and(groups['same'])}" if "same" in groups else "",
        f"not run on {_and(groups['not run'])}" if "not run" in groups else "",
    ]
    others = [x for x in others if x]
    if others:
        parts.append(f"{'On' if parts else 'on'} the other rows it is {'; '.join(others)}.")
    return (
        "Is safety restored? Descriptively (paired by prompt, unadjusted, not pre-registered; a difference "
        "counts only where its 95% interval excludes zero): " + " ".join(parts) + "\n\n"
    )


def extra_tables(facts: Facts) -> str:
    """Tables final_report.json carries beyond TABLES (P07's optional repair-vs-WiSE-FT contrast, review
    XB-07), each under its own title, after the repair comparison; "" when there is none: nothing is
    shown, and nothing is pending."""
    report = facts.report()
    tables = (report or {}).get("tables") or {}
    blocks = []
    for name, spec in tables.items():
        if name in TABLES or name in UNPLACED or not isinstance(spec, dict):
            continue
        title = (
            facts.value(report_path(), f"/tables/{name}/title", "text", what=f"the {name} title", stage=P07)
            if spec.get("title")
            else name
        )
        lead = (
            vs_wiseft_sentence(name, spec)
            if _is_vs_wiseft(name, spec)
            else vs_stock_sentence(name, spec)
            if _is_vs_stock(name, spec)
            else ""
        )
        blocks.append(f"### {title}\n\n{lead}{facts.table(name)}")
    return "".join(f"\n\n{b}" for b in blocks)


def context(facts: Facts) -> dict[str, str]:
    from lift.publish import acceptance

    cfg = settings.load()
    links = read_links()
    gh = settings.github_repo()
    authors, team = _team(cfg)
    thanks = _thanks(cfg)
    rv = facts.report_value
    sizes = {
        f"n_{short}_{split}": facts.config("project", f"sizes.{name}.{split}", "int")
        for short, name in (
            ("harmful", "harmful"),
            ("easy", "easy_benign"),
            ("hard", "hard_benign"),
            ("utility", "utility"),
            ("belebele", "belebele"),
        )
        for split in ("dev", "final")
    }
    ctx = {
        "authors": authors,
        "team": team,
        # under the team list, with its own blank line: with nobody thanked the section ends on the list
        "thanks": f"\n\n{thanks}" if thanks else "",
        **tool_chain(facts),
        "training_data": training_data(facts),
        "model_artifacts": model_artifacts(facts),
        "data_artifacts": data_artifacts(facts),
        "models_section": section_ref(MODEL_ARTIFACTS),
        "data_section": section_ref(DATA_ARTIFACTS),
        "experiments_section": section_ref(EXPERIMENTS),
        "logs_repo": settings.repo("logs").url,
        "python_requirement": python_requirement(),
        "global_id": config.get("models.global.id"),
        "global_rev": config.get("models.global.revision"),
        "base_id": config.get("models.base.id"),
        "base_rev": config.get("models.base.revision"),
        **judges(facts),
        **backtranslation(facts),
        "notebook_runtime": acceptance.runtime_text(facts),
        **sizes,
        "counts_note": counts_note(facts),
        "rule_overrefusal": facts.config("repairs", "selection.max_overrefusal_increase", "points"),
        "rule_gain": facts.config("repairs", "selection.min_gain_retained", "pct"),
        "rule_belebele": facts.config("repairs", "selection.max_belebele_hi_drop", "points"),
        "rule_tie": facts.config("repairs", "selection.tie_margin", "points"),
        "bar_agreement": facts.config("project", "validation.bar.agreement", "pct"),
        "bar_kappa": facts.config("project", "validation.bar.kappa", "f2"),
        "verdicts": verdict_lines(facts),
        "adapter_checks": adapter_checks(facts),
        "comparison_adapter": comparison_adapter(facts),
        "dev_selection": dev_selection(facts),
        "utility_retained": rv(
            "/headline/utility_retained",
            "pct_ci",
            what="the share of the adapter's win-rate gain the frozen repair keeps",
        ),
        "heldout_retained": rv(
            "/headline/heldout_retained",
            "pct_ci",
            what="the share of the adapter's held-out-loss gain the frozen repair keeps",
        ),
        "drift": rv("/primary/drift/pooled", "pp_ci", what="the pooled Hindi drift effect"),
        "drift_p": rv("/primary/drift/pooled/p", "p_stmt", what="the drift p-value"),
        "repair": rv("/primary/repair/pooled", "pp_ci", what="the pooled Hindi repair effect"),
        "repair_p": rv("/primary/repair/pooled/p", "p_stmt", what="the repair p-value"),
        "drift_seeds": _seeds_word(
            rv("/primary/drift/pooled/seeds", "list", what="the seeds pooled for drift")
        ),
        "repair_seeds": _seeds_word(
            rv("/primary/repair/pooled/seeds", "list", what="the seeds pooled for repair")
        ),
        "table_per_seed": facts.per_seed_table(PRIMARY_EFFECTS),
        "redteam_drift": rv(
            "/primary/redteam_drift/pooled", "pp_ci", what="the Aya Red-teaming drift effect"
        ),
        "redteam_repair": rv(
            "/primary/redteam_repair/pooled", "pp_ci", what="the Aya Red-teaming repair effect"
        ),
        "gpu_hours": rv("/headline/gpu_hours_per_checkup", "hours", what="GPU-hours per check-up"),
        "gpu_hours_variant": rv(
            "/headline/gpu_hours_per_variant_final",
            "hours",
            what="GPU-hours of the final battery per variant",
        ),
        "not_run": _not_run(facts),
        "prompts_repo": settings.repo("prompts").url,
        "models_collection": links.get("models")
        or pending("the model collection url", shown(links_path()), "lift publish collections --yes"),
        "data_collection": links.get("data")
        or pending("the data collection url", shown(links_path()), "lift publish collections --yes"),
        "github": gh or pending("the GitHub repo url", "configs/publish.yaml github_repo", PERSON),
        "notebook": f"{gh}/blob/main/notebooks/acceptance_test.ipynb"
        if gh
        else pending("the notebook url", "configs/publish.yaml github_repo", PERSON),
        "space": settings.repo("space").url,
        # Optional: while nobody has filed it, the article points at the draft in the repo.
        "safetune_issue": cfg.get("safetune_issue_url")
        or "our report for upstream is `hf/safetune_issue.md` in the repository",
        "references_note": cfg.get("references")
        or pending(
            "full citations for the build plan's references", "configs/publish.yaml references", PERSON
        ),
    }
    for name in (
        "adapters",
        "adapter_drift",
        "per_language",
        "repair_comparison",
        "secondary",
        "exploratory",
        "overrefusal",
        "utility",
        "belebele",
        "heldout",
        "judge_sensitivity",
        "sensitivity_effects",
        "signatures",
        "cost",
    ):
        ctx[f"table_{name}"] = facts.table(name)
    ctx["extra_tables"] = extra_tables(facts)
    ctx["judge_validation"] = validated.judge_validation(facts, primary_kind(facts))
    ctx["judge_on_benign"] = judge_on_benign(facts)
    logs = settings.repo("logs").url
    for name in FIGURES:  # the PNGs travel in the logs dataset (datasets.build_logs), linked from there
        ctx[f"figure_{name}"] = facts.figure(name, base=f"{logs}/resolve/main/")
    ctx["in_short"] = in_short(facts, ctx)
    return ctx


METHOD_WORDS = {
    "wiseft": "WiSE-FT, the adapter scaled back toward stock",
    "resta": "ReSta-A, adding back the safety difference between Global and Base",
    "cast": "CAST, conditional activation steering",
    "castalways": "activation steering with the gate always on",
}


def _method(name: str) -> str:
    family = name.split("-", 1)[0]
    return METHOD_WORDS.get(family, family)


def in_short(facts: Facts, ctx: Mapping[str, str]) -> str:
    """The lead a reader sees first: the drift, the frozen repair and its verdict, the other families'
    frozen settings, each from the values the sections below say in full (the same slots, so the same
    sources). Pending while the selection is not decided."""
    sel = _decided(facts)
    if sel is None:
        return pending("the summary (the frozen repair)", shown(selection_path()), P05)
    winner = str(sel["overall"]["winner"])
    verdict = sel["overall"].get("verdict") or "not decided"
    lines = [
        f"- **Drift.** {ctx['comparison_adapter']} changed the Hindi harmful rate by {ctx['drift']} against "
        f"stock on the final split ({ctx['drift_p']}).",
        f"- **Repair.** The frozen repair, `{winner}` ({_method(winner)}), changed it by {ctx['repair']} "
        f"against the adapter ({ctx['repair_p']}): verdict **{verdict}**.",
    ]
    others = [
        f"`{fam.get('winner')}` (**{fam.get('verdict')}**)"
        for _family, fam in sorted((sel.get("families") or {}).items())
        if isinstance(fam, Mapping) and fam.get("winner") and fam.get("winner") != winner
    ]
    if others:
        lines.append(f"- **The other repairs.** Their frozen settings: {', '.join(others)}.")
    lines.append(
        f"- Each adapter's verdict, its usefulness and every interval are in {section_ref(EXPERIMENTS)}; "
        "every number's source is in `hf/article.sources.json`."
    )
    return "**In short.**\n\n" + "\n".join(lines)


def blocked_line(facts: Facts) -> str:
    """The line that marks a draft whose recipe is not frozen at its tag (review XB2-02), "" when it is: a
    placeholder, so the article is not posted while it holds it."""
    block = recipe_block(facts)
    if not block:
        return ""
    facts.missing.append(f"the recipe frozen at its tag ({RECIPE_IN_GIT})")
    what = f"BLOCKED, a draft not for posting: {block}"
    return f"> **{pending(what, RECIPE_IN_GIT, 'P05 lift select, then a commit and the tag')}**"


def build(facts: Facts | None = None, *, readme: bool = False) -> tuple[Path, Facts]:
    """Write the article, its sources and the notebook's stamp; with ``readme`` (the command), the same
    text as the repository's README.md too, with its figures beside it."""
    from lift.publish import acceptance

    facts = facts or Facts()
    text = render.fill(render.template("article.md"), context(facts)).rstrip() + "\n"
    mark = blocked_line(facts)
    if mark:  # right under the title
        head, sep, rest = text.partition("\n# ")
        title, nl, body = rest.partition("\n")
        text = f"{head}{sep}{title}\n\n{mark}\n{nl}{body}" if sep else f"{mark}\n\n{text}"
    text, _, unresolved = preflight.relative_text(text)  # no absolute local path goes public (XB2-04)
    if unresolved:
        raise PublishError(
            f"the article would name {len(unresolved)} absolute local path(s) outside the repo and any "
            "outputs/ tree (from the inputs it reads); nothing was written"
        )
    render.write_text(article_path(), text)
    if readme:
        render.write_text(readme_path(), readme_text(text))
        readme_figures()
    facts.write_sources(sources_path())
    if preflight.relativise_file(sources_path())[1]:
        raise PublishError(f"{sources_path().name} would name an absolute local path outside the repo")
    acceptance.stamp_notebook()
    return article_path(), facts
