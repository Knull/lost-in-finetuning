"""lift publish article: the organisers' nine sections in order and the extras after them, placeholders
until the inputs exist."""

from __future__ import annotations

import json
import re

import pytest
from publish_fakes import (  # noqa: F401
    ADAPTER,
    CAST_WINS,
    WINNER,
    no_committed_recipe,
    secondary_selection,
    selection,
    write_inputs,
    write_json,
    write_report,
)

from lift.publish import PublishError, article, settings
from lift.publish.collect import links_path
from lift.publish.inputs import PENDING, Facts


@pytest.fixture
def out(tmp_path, monkeypatch):
    from lift.publish import acceptance

    monkeypatch.setattr(article, "article_path", lambda: tmp_path / "article.md")
    monkeypatch.setattr(article, "sources_path", lambda: tmp_path / "article.sources.json")
    notebook = tmp_path / "acceptance_test.ipynb"
    notebook.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(acceptance, "notebook_path", lambda: notebook)
    return tmp_path


def _publish_cfg(monkeypatch, **extra):
    cfg = {**settings.load(), **extra}
    monkeypatch.setattr(settings, "load", lambda: cfg)


HASH = "a" * 64
# The adapters of a study like ours, with the training set each read: (its folder under data/sft/, the
# rows in the file, the rows trained). Made-up counts; the IndicAlign runs dropped rows as unfit.
TRAINED = {
    "adapter-aya-std-s1": ("aya/dedup", 40, 40),
    "adapter-aya-std-s2": ("aya/dedup", 40, 40),
    "adapter-indicalign-std-s1": ("indicalign/dedup", 1200, 1199),
    "adapter-indicalign-stress-s1": ("indicalign/raw", 1200, 1197),
}


def train_manifest(name: str, key: str, rows: int, trained: int, **changes) -> dict:
    """A ``lift train`` manifest (lift.train.lora): the fields the article and ``datasets.train_runs``
    read."""
    doc = {
        "run_id": f"train-{name}",
        "stage": "train",
        "name": name,
        "candidate": name.split("-")[1],
        "recipe": "stress" if "-stress-" in name else "standard",
        "seed": int(name[-1]),
        "data_path": f"data/sft/{key}/sft_messages.jsonl",
        "data_sha256": "b" * 64,
        "n_rows": rows,
        "n_rows_trained": trained,
        "dropped_unfit": {"count": rows - trained, "ids": []},
        "adapter_hash": HASH,
        "status": "ok",
    }
    return {**doc, **changes}


def manifest_path(name: str, run: str = "20261108T000000Z"):
    from lift import paths

    return paths.manifests_dir() / f"train-{run}-{name}.json"


def register_adapters(monkeypatch, trained=None, *, manifests: bool = True) -> dict:
    """Register ``trained``'s adapters (default TRAINED) in the registry the publishing code reads, each
    with the hash its training run recorded, and write each one's train manifest."""
    from lift.models import variants
    from lift.schemas import VariantSpec

    trained = TRAINED if trained is None else trained
    specs = {"stock": VariantSpec(name="stock", method="stock")}
    for name in trained:
        specs[name] = VariantSpec(
            name=name, method="adapter", adapter=name, seed=int(name[-1]), artifacts={"adapter": HASH}
        )

    def get_spec(name):
        if name not in specs:
            raise KeyError(name)
        return specs[name]

    monkeypatch.setattr(variants, "load_registry", lambda: dict(specs))
    monkeypatch.setattr(variants, "get_spec", get_spec)
    if manifests:
        for name, (key, rows, used) in trained.items():
            write_json(manifest_path(name), train_manifest(name, key, rows, used))
    return specs


def _part(text: str, heading: str) -> str:
    """The body of a level-two section, by its heading as written."""
    return text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


# ---------------------------------------------------------------- the nine sections and the extras (B5)

# The "Submission artifacts" list of the organisers' kickoff slides, word for word (docs/decisions.md item
# 119), and the words of each line that its heading in the article must keep.
SLIDE = (
    ("Problem statement and approach, in brief", "problem statement and approach"),
    ("Hackathon repos used: which ones, for what, and which features", "hackathon repos used features"),
    ("Models used and their licenses", "models used licences"),
    ("Datasets and benchmarks used and their licenses", "datasets benchmarks used licences"),
    ("Model artifacts (weights, adapters, …) public, as an HF collection", "model artifacts hf collection"),
    (
        "Data artifacts (datasets, logs, benchmarks, …) public, as an HF collection",
        "data artifacts hf collection",
    ),
    ("Experiments run, findings and what else was tested", "experiments run findings what else was tested"),
    ("With more credits: your approach, before and after results", "with more credits"),
    ("Teammates tagged", "teammates"),
)


def _heads(text: str) -> list[str]:
    """The level-two headings of an article, as written."""
    return [line[3:] for line in text.splitlines() if line.startswith("## ")]


def _numbered() -> list[str]:
    """The headings the nine sections and the extras must give, in order."""
    return [f"{i}. {name}" for i, name in enumerate(article.SECTIONS, 1)] + list(article.EXTRAS)


def test_the_numbered_sections_are_the_organisers_nine():
    """B5: the article's numbered sections are the nine of the kickoff slides, one heading per line of
    the slide and in its order, each keeping the words of its line; everything else comes after them,
    unnumbered."""
    assert len(article.SECTIONS) == len(SLIDE) == 9
    for heading, (line, words) in zip(article.SECTIONS, SLIDE, strict=True):
        assert set(words.split()) <= set(re.findall(r"[a-z]+", heading.lower())), (heading, line)
    assert article.EXTRAS == ("Related work", "Reproduce", "Links")
    assert not set(article.EXTRAS) & set(article.SECTIONS)
    assert article.section_ref(article.EXPERIMENTS) == "section 7"
    assert article.section_ref(article.MODEL_ARTIFACTS) == "section 5"
    assert article.section_ref(article.DATA_ARTIFACTS) == "section 6"


def test_sections_in_the_organisers_order_with_placeholders(out):
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert _heads(text) == _numbered()
    assert "2b, statement 6" in text and "statement 8" in text and "2a, statement 4" in text
    assert PENDING in text
    pending = article.placeholders(text)
    assert any("final_report.json" in p for p in pending)
    assert any("configs/publish.yaml team" in p for p in pending)
    assert json.loads((out / "article.sources.json").read_text())["pending"]
    # design numbers come from the configs, with their source recorded
    assert "| 172 | 400 |" in text
    assert any(s["pointer"] == "/sizes/harmful/dev" for s in facts.sources)


def test_metrics_open_the_experiments_section_and_cost_closes_it(out):
    """B5: "Metrics and why" and "Cost and ROI" are no sections of the organisers' list. They are the
    first and the last subsection of section 7, each with everything it held as a section."""
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    body = text.split("## 7. Experiments run, findings and what else was tested\n", 1)[1]
    body = body.split("\n## ", 1)[0]
    subs = [line[4:] for line in body.splitlines() if line.startswith("### ")]
    assert subs[0] == "Metrics and why" and subs[-1] == "Cost and ROI" and subs[-2] == "Limitations"
    assert subs.index("What each check said") == 1
    metrics = body.split("### Metrics and why\n", 1)[1].split("\n### ", 1)[0]
    for said in ("**Harmful rate**", "**Held-out loss**", "The selection rule, run on seed-1 dev only"):
        assert said in metrics
    assert 'Whether\nit did is under "The judge against native readers", below.' in metrics
    cost = body.split("### Cost and ROI\n", 1)[1]
    assert "| Variant | Role | GPU-hours |" in cost and "the repairs here need no gradient step" in cost
    assert "1.25 h of GPU time" in cost and "we\nmake no commercial claims." in cost
    # every reference to a section names it by the number it now has
    assert "section 6," not in metrics and re.findall(r"section \d+", text.split("## 1. ", 1)[0]) == [
        "section 7"
    ]


def test_full_inputs_leave_no_placeholder(out, monkeypatch):
    write_inputs(validation="stay")
    register_adapters(monkeypatch)
    write_json(
        links_path(),
        {
            "models": {"slug": "o/m", "url": "https://hf.co/collections/o/m"},
            "data": {"slug": "o/d", "url": "https://hf.co/collections/o/d"},
        },
    )
    _publish_cfg(
        monkeypatch,
        team=[{"name": "A", "hf": "a-user"}],
        annotators=[],
        references="- Full citations.",
        github_repo="https://github.com/o/r",
    )
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert PENDING not in text, article.placeholders(text)
    assert _heads(text) == _numbered()
    assert "+16.0 pp [+11.0, +21.0], p < 0.001" in text and "p = <" not in text  # XB-15
    assert "**repair**" in text and "ReSta on seed 3" in text
    assert "https://github.com/o/r/blob/main/notebooks/acceptance_test.ipynb" in text
    nb = json.loads((out / "acceptance_test.ipynb").read_text())
    assert any('REPO_URL = "https://github.com/o/r"' in line for c in nb["cells"] for line in c["source"])
    assert "  - user: a-user" in text
    # the lead: drift, the frozen repair and its verdict, the other families, before section 1
    lead = text.split("## 1. Problem statement and approach", 1)[0]
    assert "**In short.**" in lead and "+16.0 pp [+11.0, +21.0]" in lead and "verdict **repair**" in lead
    assert (
        "**The other repairs.** Their frozen settings: `cast-m1.0-adapter-aya-std-s1` (**repair**), "
        "`resta-a0.25-adapter-aya-std-s1` (**repair**)."
    ) in lead
    assert "every interval are in section 7; every number's source is in" in lead
    # annotators: [] thanks nobody, so the teammates section is its list and nothing else
    assert "We thank" not in text
    assert _part(text, "9. Teammates").strip() == "- A ([@a-user](https://huggingface.co/a-user))"
    # the article ends on its links, the last of the extras
    assert text.endswith(f"- Space: {settings.repo('space').url}\n")
    assert "Later releases centre them" in text
    sources = json.loads((out / "article.sources.json").read_text())
    assert sources["pending"] == []
    assert any(
        v["pointer"] == "/primary/drift/pooled" and v["file"] == "outputs/metrics/final_report.json"
        for v in sources["values"]
    )


def test_annotators_are_named_only_when_listed(monkeypatch):
    _publish_cfg(monkeypatch, annotators=None)
    assert PENDING in article._thanks(settings.load())
    _publish_cfg(monkeypatch, annotators=[])
    assert article._thanks(settings.load()) == ""  # [] thanks nobody: no line at all
    _publish_cfg(monkeypatch, annotators=["R. Reader"])
    assert "R. Reader" in article._thanks(settings.load())


def test_the_thanks_stay_under_the_team_list(out, monkeypatch):
    """The teammates are section 9 and the extras follow it, so the thanks line sits under the team list,
    in the same section, and no longer ends the article."""
    _publish_cfg(monkeypatch, team=[{"name": "A", "hf": "a-user"}], annotators=["R. Reader"])
    text = article.build()[0].read_text(encoding="utf-8")
    assert _part(text, "9. Teammates").strip() == (
        "- A ([@a-user](https://huggingface.co/a-user))\n\n"
        "We thank R. Reader, who labelled the validation samples."
    )


def test_committed_draft_matches_the_template_sections():
    """hf/article.md is generated: it carries every section of the template, in order."""
    text = article.article_path().read_text(encoding="utf-8")
    assert _heads(text) == _numbered()


def test_the_committed_readme_is_the_committed_article():
    """Decisions item 96: README.md is the write-up, the same text as hf/article.md without its HF front
    matter, both written by ``lift publish article``. One regenerated without the other fails here."""
    from lift.paths import REPO_ROOT

    assert (REPO_ROOT / "README.md").read_text(encoding="utf-8") == article.readme_text(
        article.article_path().read_text(encoding="utf-8")
    )
    assert "PREREGISTRATION.md" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (REPO_ROOT / "PREREGISTRATION.md").is_file()


def test_the_committed_write_up_tags_the_team_of_the_publish_config():
    """Decisions item 122: the team is read from configs/publish.yaml, so a member added there is tagged
    only once the write-up is generated again. The committed README and article tag every member the
    committed config names, each with the HF account it gives, and nobody else."""
    import yaml

    from lift.paths import CONFIG_DIR, REPO_ROOT

    team = yaml.safe_load((CONFIG_DIR / "publish.yaml").read_text(encoding="utf-8")).get("team") or []
    assert team, "configs/publish.yaml names no team"
    listed = _part((REPO_ROOT / "README.md").read_text(encoding="utf-8"), "9. Teammates")
    front = article.article_path().read_text(encoding="utf-8").split("\n---\n", 1)[0]
    assert all(m.get("hf") for m in team), "the organisers ask for every teammate to be tagged"
    tagged = [f"- {m['name']} ([@{m['hf']}](https://huggingface.co/{m['hf']}))" for m in team]
    assert [line for line in listed.splitlines() if line.startswith("- ")] == tagged
    assert [line for line in front.splitlines() if line.startswith("  - user: ")] == [
        f"  - user: {m['hf']}" for m in team
    ]


def test_the_readme_is_written_only_when_asked(out):
    path, _ = article.build()
    assert not article.readme_path().exists()  # tests' README lives under tmp_path (conftest)
    path, _ = article.build(readme=True)
    text = path.read_text(encoding="utf-8")
    readme = article.readme_path().read_text(encoding="utf-8")
    assert text.startswith("---\n") and readme == article.readme_text(text)
    assert readme.startswith("# Lost in Fine-Tuning") and "thumbnail:" not in readme  # GitHub shows no table
    assert article.readme_path().parent != article.paths.REPO_ROOT


def test_the_readme_shows_the_figures_from_copies_beside_it(out):
    """Nov 13: the article links each figure in the logs dataset, which goes up only once every label is
    final, so on GitHub the README showed broken images. The README links copies kept beside it in the
    repository's figures/ folder instead; hf/article.md keeps the dataset's links."""
    from lift import paths

    write_inputs()
    path, _ = article.build(readme=True)
    text = path.read_text(encoding="utf-8")
    readme = article.readme_path().read_text(encoding="utf-8")
    folder = article.readme_path().parent / "figures"
    assert "/resolve/main/figures/adapters_by_language.png)" in text
    assert "](figures/adapters_by_language.png)" in readme and "/resolve/main/figures/" not in readme
    for name in article.FIGURES:
        drawn = paths.figures_dir() / f"{name}.png"
        assert (folder / f"{name}.png").read_bytes() == drawn.read_bytes()
    # A figure no longer drawn loses its copy, and the README its link.
    (paths.figures_dir() / "tradeoff_dev_heldout.png").unlink()
    article.build(readme=True)
    assert not (folder / "tradeoff_dev_heldout.png").exists()
    assert "figures/tradeoff_dev_heldout.png" not in article.readme_path().read_text(encoding="utf-8")


# ------------------------------------------- sections 5 and 6: what is published, where and why (B5, B6)

AUTOMATIC = (
    "A gated repo here is always gated with automatic approval: log in to Hugging Face, accept the terms on "
    "the repo's page, and access is immediate, with no review by a person."
)
DRIFTED = "a research artifact whose safety may have drifted."


def _lines(part: str) -> list[str]:
    return [line for line in part.splitlines() if line.startswith("- ")]


def test_section_5_lists_every_repo_of_the_model_collection_with_its_access_and_why(out, monkeypatch):
    """B5 and B6: the model collection's link, then one line per repo that `lift publish collections`
    puts in it, from the plan and configs/publish.yaml: what it is, its access, and the reason for it.
    Every gate is said to be automatic and immediate, as Tiny Aya Global's own is."""
    from lift.publish import collect

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs()
    register_adapters(monkeypatch)
    write_json(links_path(), {"models": {"slug": "o/m", "url": "https://hf.co/collections/o/m"}})
    text = article.build()[0].read_text(encoding="utf-8")
    part = _part(text, "5. Model artifacts, as an HF collection")
    assert part.lstrip().startswith(
        "The model artifacts are in one Hugging Face collection: https://hf.co/collections/o/m\n"
    )
    flat = " ".join(part.split())
    assert AUTOMATIC in flat and "is gated the same way by Cohere Labs" in flat
    assert "No repo is private, and none is approved by hand." in flat
    lines = _lines(part)
    plan = collect.plan()["models"]
    assert len(lines) == len(plan) == 7  # the repaired model, the parameters, four adapters, the Space
    for line, repo in zip(lines, plan, strict=True):
        assert f"[`{repo.repo_id}`]({repo.url})" in line  # the collection's own order, each with its link
        assert ("Public: " in line) == (repo.access == "public")
        assert ("Gated, automatic approval" in line) == (repo.access == "gated-auto")
    assert lines[0] == (
        f"- **Repaired model**: [`test-org/lift-{WINNER}`](https://huggingface.co/test-org/lift-{WINNER}). "
        f"The frozen repair `{WINNER}` (WiSE-FT, the adapter scaled back toward stock), saved as merged "
        "weights. Gated, automatic approval, because it is a WiSE-FT repair, Global + lambda x (adapter - "
        "Global), and lambda is public, so anyone could rebuild the gated adapter from these weights as "
        "Global + (weights - Global) / lambda."
    )
    assert lines[1].startswith("- **Repair parameters**: [`test-org/lift-repair-params`](")
    assert lines[1].endswith(
        "Public: every repaired model's weights are gated, so on their own the parameters rebuild nothing."
    )
    assert lines[2] == (
        "- **Unrepaired adapter**: [`test-org/lift-adapter-aya-std-s1`]"
        "(https://huggingface.co/test-org/lift-adapter-aya-std-s1). A Hindi LoRA adapter trained on the Aya "
        f"Dataset's Hindi rows with the standard recipe, seed 1. Gated, automatic approval: {DRIFTED}"
    )
    assert "the Aya Dataset's Hindi rows with the standard recipe, seed 2." in lines[3]
    assert "trained on the IndicAlign `Dolly_T` sample with the standard recipe, seed 1." in lines[4]
    assert "trained on the IndicAlign `Dolly_T` sample with the stress recipe, seed 1." in lines[5]
    assert all(line.endswith(DRIFTED) for line in lines[2:6])
    assert lines[6].startswith("- **Space**: [`test-org/lift-acceptance-test`](https://huggingface.co/spaces/")
    # the Space is public and shows the start of some outputs, harmful-labelled ones included: said in full
    assert lines[6].endswith(
        "A static page of the results: the verdicts, the heatmaps, the utility table and a few example "
        "outputs for each variant, language and safety prompt set of the final split, each cut to its first "
        "words and marked `[redacted]`, whatever its label. Public: it loads no model and generates "
        "nothing, and it holds no more of any output than that cut."
    )
    assert part.rstrip().endswith("The verdicts in section 7 say which adapters drifted.")


def test_the_repaired_models_line_gives_its_methods_own_reason(out, monkeypatch):
    """The reason is the one the model's card and access form give (model.gate_reasons): a CAST winner's
    weights are the unrepaired adapter, and a rejected selection says so."""
    from publish_fakes import REJECT

    write_inputs(selection_doc=selection(CAST_WINS))
    line = _lines(article.model_artifacts(Facts()))[0]
    assert (
        "The frozen repair `cast-m1.0-adapter-aya-std-s1` (CAST, conditional activation steering), saved as "
        "the adapter-merged model it wraps, with the steering artifacts and the wrapper that applies them."
    ) in line
    assert "because it is a CAST repair: these weights are the unrepaired adapter, merged" in line
    write_inputs(selection_doc=selection(REJECT))
    line = _lines(article.model_artifacts(Facts()))[0]
    assert '; and the verdict is "reject": no setting met every selection constraint' in line


def test_the_repaired_model_is_pending_until_the_selection_is_decided(out):
    facts = Facts()
    lines = _lines(article.model_artifacts(facts))
    assert lines[0].startswith(
        f"- **Repaired model**: {PENDING}: the repaired model (the frozen winner) from "
    )
    assert "pending-winner" not in "\n".join(lines) and any("the repaired model" in m for m in facts.missing)
    assert [line.split("**")[1] for line in lines] == ["Repaired model", "Repair parameters", "Space"]
    assert "say which adapters drifted" not in article.model_artifacts(Facts())  # no adapter is listed


@pytest.mark.parametrize(
    ("kind", "access", "build"),
    [
        ("adapter", "gated-manual", "model_artifacts"),
        ("model", "gated-manual", "model_artifacts"),
        ("repair_params", "gated-auto", "model_artifacts"),
        ("raw", "gated-manual", "data_artifacts"),
        ("raw", "public", "data_artifacts"),
        ("logs", "gated-auto", "data_artifacts"),
    ],
)
def test_an_access_the_article_gives_no_reason_for_is_refused(out, monkeypatch, kind, access, build):
    """B6: the two sections say that every gate opens when its terms are accepted, and give each repo's
    reason for the access configs/publish.yaml sets. Another access, a gate a person approves above all,
    is refused: the text would say what the repo is not."""
    import copy

    write_inputs()
    register_adapters(monkeypatch)
    cfg = copy.deepcopy(settings.load())
    cfg["repos"][kind]["access"] = access
    monkeypatch.setattr(settings, "load", lambda: cfg)
    with pytest.raises(PublishError, match=rf"repos\.{kind}\.access is {access}"):
        getattr(article, build)(Facts())
    with pytest.raises(PublishError, match=rf"repos\.{kind}\.access is {access}"):
        article.build()


def test_section_6_lists_the_data_collection_then_the_parameters(out, monkeypatch):
    """B5 and B6: the data collection's link, one line per dataset the collection holds, and the repair
    parameters, which sit in the model collection. The raw completions are the only gated one, and the
    section says the gate is automatic and that every number can be checked from public files."""
    from lift.publish import collect

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs()
    write_json(links_path(), {"data": {"slug": "o/d", "url": "https://hf.co/collections/o/d"}})
    text = article.build()[0].read_text(encoding="utf-8")
    part = _part(text, "6. Data artifacts, as an HF collection")
    assert part.lstrip().startswith(
        "The data artifacts are in one Hugging Face collection: https://hf.co/collections/o/d\n"
    )
    lines = _lines(part)
    assert [line.split("**")[1] for line in lines] == [
        "Evaluation logs",
        "Translated prompt sets",
        "Raw completions",
        "Repair parameters",
    ]
    listed = {r.kind: r for r in collect.plan()["data"]}
    assert set(listed) == {"logs", "prompts", "raw"}  # the training sets are not published (item 118)
    for line, kind in zip(lines, ("logs", "prompts", "raw"), strict=False):
        assert f"[`{listed[kind].repo_id}`]({listed[kind].url})" in line
    assert lines[0].endswith("Public: it holds no output text and no judge text.")
    assert "the aggregate tables and run manifests the numbers here are read from" in lines[0]
    assert lines[1].endswith("Public: prompts only, no model output.")
    assert lines[2] == (
        "- **Raw completions**: [`test-org/lift-raw-completions`]"
        "(https://huggingface.co/datasets/test-org/lift-raw-completions). Every generation, judgement and "
        "utility verdict in full. Gated, automatic approval: it is the only repo that holds the outputs in "
        "full, harmful ones included, so it sits behind a login and terms that keep it to safety research."
    )
    assert "[`test-org/lift-repair-params`](https://huggingface.co/test-org/lift-repair-params)" in lines[3]
    assert "They are in the model collection (section 5) and listed here as well" in lines[3]
    flat = " ".join(part.split())
    # B6: this section too says the gate is automatic and instant, as Tiny Aya Global's own is
    assert (
        "The gate on the raw completions is automatic, like the gates of section 5 and like Tiny Aya "
        "Global's own: log in to Hugging Face, accept the terms on the dataset's page, and access is "
        "immediate, with no review by a person. Nothing in either collection is private or approved by hand."
    ) in flat
    assert (
        "Every number in this write-up can be checked from public files alone. `hf/article.sources.json` "
        "names the file and the JSON pointer each value was read from."
    ) in flat
    # the script compares the result values; the design numbers of configs/ it only counts (Reproduce)
    assert "`python scripts/verify_numbers.py` compares every result value with its public copy" in flat
    assert "every recorded value" not in text
    # nothing here, nor in section 2, names the training sets as published
    assert "lift-hindi-sft" not in text and "Training sets" not in part
    section_2 = _part(text, "2. Hackathon repos used and their features")
    assert "with its manifest, rejected rows and provenance kept.\n" in section_2
    assert "published" not in section_2


def test_section_2_opens_with_the_three_tools_as_one_chain(out, monkeypatch):
    """The review's F8: no sentence showed the three Lexsi tools as one connected system. Section 2 opens
    with one. While the training sets are not published (decisions item 118) it claims nothing of them: it
    points at the adapters and the repaired model, and at AuditKIT's records in the logs dataset."""
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    section_2 = _part(text, "2. Hackathon repos used and their features")
    chain, first_tool = section_2.strip().split("\n\n", 1)
    assert " ".join(chain.split()) == (
        "The three tools run as one chain: CuratorKIT builds the Hindi training sets, the LoRA adapters are "
        "trained on them, SafeTune repairs an adapter, and AuditKIT records every reported rate. The "
        "adapters and the repaired model are in section 5, and AuditKIT's redacted records are in the logs "
        "dataset (section 6)."
    )
    assert first_tool.startswith("- **CuratorKIT** built the Hindi training sets")
    assert "lift-hindi-sft" not in section_2 and "published" not in section_2
    assert article.tool_chain(Facts())["curatorkit_files"] == ""


def test_the_training_sets_are_named_only_once_the_data_collection_lists_them(out, monkeypatch):
    """Decisions item 118: the training sets with CuratorKIT's files are staged and not uploaded, so the
    article does not name them. Once `lift publish collections` lists the dataset, section 6 has its line
    and section 2 says where CuratorKIT's files are, with the three tools as one chain of public links."""
    from lift.publish import collect

    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    write_inputs()
    held = article.tool_chain(Facts())
    assert held["curatorkit_files"] == "" and "lift-hindi-sft" not in held["tool_chain"]
    real = collect.plan

    def plan(facts=None):
        p = real(facts)
        return {**p, "data": [*p["data"], settings.repo("sft")]}

    monkeypatch.setattr(collect, "plan", plan)
    text = article.build()[0].read_text(encoding="utf-8")
    link = "[`test-org/lift-hindi-sft`](https://huggingface.co/datasets/test-org/lift-hindi-sft)"
    lines = _lines(_part(text, "6. Data artifacts, as an HF collection"))
    assert [line.split("**")[1] for line in lines] == [
        "Evaluation logs",
        "Translated prompt sets",
        "Training sets",
        "Raw completions",
        "Repair parameters",
    ]
    assert lines[2] == (
        f"- **Training sets**: {link}. The Hindi training sets the adapters were trained on, each with "
        "CuratorKIT's manifest, rejected rows and provenance. Public: training rows of public datasets, no "
        "model output."
    )
    section_2 = _part(text, "2. Hackathon repos used and their features")
    assert section_2.lstrip().startswith(
        "The three tools run as one chain, and each link of it is published: CuratorKIT's training sets "
        f"({link}), the LoRA adapters trained on them and SafeTune's repair (section 5), and AuditKIT's "
        "records of every reported rate (the logs dataset, section 6).\n\n- **CuratorKIT** built"
    )
    assert f"provenance kept and published with the training sets ({link}).\n" in section_2


def test_a_dataset_the_article_has_no_line_for_is_refused(out, monkeypatch):
    from lift.publish import collect

    write_inputs()
    real = collect.plan
    extra = settings.RepoSpec(kind="other", repo_id="o/other", repo_type="dataset", access="public")
    monkeypatch.setattr(collect, "plan", lambda facts=None: {**real(facts), "data": [extra]})
    with pytest.raises(PublishError, match="the data collection lists o/other"):
        article.data_artifacts(Facts())
    monkeypatch.setattr(collect, "plan", lambda facts=None: {"models": [extra], "data": []})
    with pytest.raises(PublishError, match="repos.other.access is public"):
        article.model_artifacts(Facts())


# ------------------------------------------- sections 3 and 4: the licences and the training data (F7)


def test_every_model_and_tool_has_its_licence(out):
    """F7: section 3 gives a licence for every model and tool, the reference model, the judges and the
    language tools included; an API judge is used under its provider's terms."""
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    part = " ".join(_part(text, "3. Models used and their licences").split())
    cohere = "CC-BY-NC-4.0 plus Cohere Labs' acceptable-use policy"
    assert f"the model we adapt and repair. {cohere}." in part
    assert f"only the ReSta reference. The same licence as Global: {cohere}." in part
    assert "a safety classifier (Apache-2.0), run on the native text." in part
    assert (
        "(`configs/rubrics/safety_v1.md`), used through the provider's API under its terms of service."
    ) in part
    assert "- **GlotLID** (Apache-2.0 plus notices, per its card) for language identification and" in part
    assert "**NLLB-200-distilled-600M** (CC-BY-NC-4.0)" in part


TRAINING_HEAD = "| Training set | Source and licence | Rows | Adapters trained on it |"
DOLLY = "IndicAlign `Dolly_T`, Hindi (CC-BY-SA-3.0 through Dolly)"


def test_the_training_data_is_listed_with_its_licences_and_the_rows_each_run_recorded(out, monkeypatch):
    """F7: section 4 lists the training data. Each set's rows, and the rows trained where a run dropped
    some, are read from the train manifests of the registered adapters, every value recorded with its
    manifest and pointer; the utility row names the licences of its two sources."""
    write_inputs()
    register_adapters(monkeypatch)
    path, facts = article.build()
    part = _part(path.read_text(encoding="utf-8"), "4. Datasets and benchmarks used and their licences")
    assert (
        "| Utility | held-out Hindi instructions from the two training sources: Aya Dataset (Apache-2.0) and "
        "IndicAlign `Dolly_T` (CC-BY-SA-3.0 through Dolly, see the training data) |"
    ) in part
    table = part[part.index(TRAINING_HEAD) :].split("\n\n", 1)[0].splitlines()
    assert table[2:] == [
        "| `aya/dedup` | Aya Dataset, Hindi rows (Apache-2.0), deduplicated | 40 | standard recipe, "
        "seeds 1, 2 |",
        f"| `indicalign/dedup` | {DOLLY}, deduplicated | 1,200 (1,199 trained) | standard recipe, seed 1 |",
        f"| `indicalign/raw` | {DOLLY}, not deduplicated | 1,200 (1,197 trained) | stress recipe, seed 1 |",
    ]
    flat = " ".join(part.split())
    assert (
        "The row counts are the ones each training run recorded in its manifest (`manifests/train-*.json` "
        "in the logs dataset). Where fewer rows were trained than the file holds, the others were dropped "
        "before training: their prompt alone, with the chat template's preamble, reaches the training "
        "length and leaves no answer token to learn from."
    ) in flat
    assert (
        "IndicAlign's card gives CC-BY-4.0, but its `Dolly_T` rows are a translation of "
        "databricks-dolly-15k, which is CC-BY-SA-3.0, so the share-alike terms apply to them."
    ) in flat
    assert "These rows are in no evaluation split" in flat
    # every run's own counts, recipe and seed are recorded: a reader can check each seed's manifest
    recorded = {(s["file"], s["pointer"]): s["value"] for s in facts.sources}
    for name, (_, rows, used) in TRAINED.items():
        file = f"outputs/manifests/{manifest_path(name).name}"
        assert recorded[(file, "/n_rows")] == rows and recorded[(file, "/n_rows_trained")] == used
        assert recorded[(file, "/seed")] == int(name[-1])
        assert recorded[(file, "/recipe")] == ("stress" if "-stress-" in name else "standard")
    assert PENDING not in part


def test_a_training_set_trained_whole_says_nothing_of_dropped_rows(out, monkeypatch):
    register_adapters(monkeypatch, {"adapter-aya-std-s1": ("aya/dedup", 40, 40)})
    said = article.training_data(Facts())
    assert (
        "| `aya/dedup` | Aya Dataset, Hindi rows (Apache-2.0), deduplicated | 40 | standard recipe, seed 1 |"
    ) in said
    assert "trained)" not in said and "dropped" not in said
    assert said.endswith("(`manifests/train-*.json` in the logs dataset).")


def test_the_run_that_counts_is_the_one_that_made_the_registered_weights(out, monkeypatch):
    """A retrain leaves the earlier run's manifest on record under the same name, with another adapter
    hash: the registry pins the weights, and the article reads the run that made them."""
    name = "adapter-aya-std-s1"
    register_adapters(monkeypatch, {name: ("aya/dedup", 40, 40)})
    stale = train_manifest(name, "aya/dedup", 55, 55, adapter_hash="c" * 64)
    write_json(manifest_path(name, "20261109T000000Z"), stale)
    facts = Facts()
    assert "| 40 | standard recipe, seed 1 |" in article.training_data(facts)
    assert {s["file"] for s in facts.sources} == {f"outputs/manifests/{manifest_path(name).name}"}
    assert article.training_runs()[name][0] == manifest_path(name)


def test_a_run_that_never_hashed_its_adapter_is_no_run_of_a_registered_adapter(out, monkeypatch):
    """Where the registry pins no weights for an adapter, a run under its name counts only if it recorded
    an adapter hash: one that stopped before that trained nothing that is registered, and its counts are
    not printed."""
    from lift.models import variants
    from lift.schemas import VariantSpec

    name = "adapter-aya-std-s1"
    spec = VariantSpec(name=name, method="adapter", adapter=name, seed=1, artifacts={})
    monkeypatch.setattr(variants, "load_registry", lambda: {name: spec})
    monkeypatch.setattr(variants, "get_spec", lambda n: {name: spec}[n])
    crashed = {k: v for k, v in train_manifest(name, "aya/dedup", 55, 55).items() if k != "adapter_hash"}
    write_json(manifest_path(name), {**crashed, "status": "error"})
    assert article.training_runs() == {name: None}
    assert article.training_data(Facts()) == (
        f"_{PENDING}: the training run of `{name}` from outputs/manifests (lift train)]_"
    )
    write_json(manifest_path(name, "20261109T000000Z"), train_manifest(name, "aya/dedup", 40, 40))
    assert article.training_runs()[name][0] == manifest_path(name, "20261109T000000Z")
    assert "| 40 | standard recipe, seed 1 |" in article.training_data(Facts())


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"n_rows": 41, "n_rows_trained": 41}, "disagree on its row counts"),
        ({"n_rows_trained": 39}, "disagree on its row counts"),
        # the same counts from another build of the path: two training sets, not one
        ({"data_sha256": "d" * 64}, "read different files under that path"),
    ],
)
def test_runs_that_disagree_on_a_training_sets_rows_are_refused(out, monkeypatch, change, says):
    """One count is given per training set, so the runs trained on it must have read the same file and
    record the same counts."""
    register_adapters(monkeypatch)
    name = "adapter-aya-std-s2"
    doc = {**train_manifest(name, "aya/dedup", 40, 40), **change}
    doc["dropped_unfit"] = {"count": doc["n_rows"] - doc["n_rows_trained"], "ids": []}
    write_json(manifest_path(name), doc)
    with pytest.raises(PublishError, match=says) as e:
        article.training_data(Facts())
    assert "data/sft/aya/dedup/sft_messages.jsonl" in str(e.value) and name in str(e.value)


@pytest.mark.parametrize(
    "change",
    [
        {"dropped_unfit": {"count": 0, "ids": []}},  # rows went missing that the run did not drop
        {"dropped_unfit": None},
        {"n_rows_trained": 1201},
        {"n_rows": "1200"},
        {"n_rows_trained": True},
    ],
)
def test_row_counts_that_do_not_add_up_are_refused(out, monkeypatch, change):
    """The rows trained are the file's rows less the ones the run dropped as unfit, as its manifest
    records them (lift.train.lora); anything else is refused, not printed."""
    name = "adapter-indicalign-std-s1"
    register_adapters(monkeypatch, {name: TRAINED[name]})
    write_json(manifest_path(name), {**train_manifest(name, *TRAINED[name]), **change})
    with pytest.raises(PublishError, match="do not add up"):
        article.training_data(Facts())
    with pytest.raises(PublishError, match="do not add up"):
        article.build()


def test_the_training_data_is_pending_without_the_adapters_or_their_runs(out, monkeypatch):
    """No adapter registered, or an adapter whose training run is not on this machine: the table says so
    with a placeholder, which blocks a posting, and never guesses a count."""
    facts = Facts()
    said = article.training_data(facts)
    assert said == (
        f"_{PENDING}: the training sets of the registered adapters from outputs/manifests (lift train)]_"
    )
    assert facts.missing == ["the training sets of the registered adapters (outputs/manifests)"]
    # adapters registered, none of their runs here: no table at all, one placeholder that names them
    register_adapters(monkeypatch, {"adapter-aya-std-s1": TRAINED["adapter-aya-std-s1"]}, manifests=False)
    assert article.training_data(Facts()) == (
        f"_{PENDING}: the training run of `adapter-aya-std-s1` from outputs/manifests (lift train)]_"
    )
    register_adapters(monkeypatch)
    manifest_path("adapter-indicalign-stress-s1").unlink()
    manifest_path("adapter-aya-std-s2").unlink()
    facts = Facts()
    said = article.training_data(facts)
    assert "| `aya/dedup` |" in said and "standard recipe, seed 1 |" in said
    assert "`indicalign/raw`" not in said
    assert said.endswith(
        f"_{PENDING}: the training runs of `adapter-aya-std-s2` and `adapter-indicalign-stress-s1` from "
        "outputs/manifests (lift train)]_"
    )
    assert len(facts.missing) == 1


# ------------------------------------------------------------- the extras: how to reproduce (B4)


def _reproduce(text: str) -> str:
    return _part(text, "Reproduce")


def test_the_reproduce_section_names_the_number_check_the_tests_and_the_stages(out):
    """B4: how to check every number, how to run the offline tests and how to run the study again, between
    the related work and the links."""
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert _heads(text)[-3:] == ["Related work", "Reproduce", "Links"]
    part = _reproduce(text)
    flat = " ".join(part.split())
    assert "```\npython scripts/verify_numbers.py\n```" in part
    assert f"from the logs dataset ({settings.repo('logs').url})" in flat
    assert "needs no GPU, no token and nothing installed beyond Python" in flat
    assert "Three things it does not check:" in flat and "which it does not recompute" in flat
    assert '```\npip install -e ".[sync,dev]"\npytest\n```' in part
    assert f"need {article.python_requirement()}:" in flat and "or later" in article.python_requirement()
    # what is known of the suite, and no more: the failures on Windows are named, nothing is promised
    assert (
        "On Windows a few tests are known to fail for reasons of portability (path separators, line "
        "endings, process ids); none of them touches a reported number."
    ) in flat
    assert "known to pass" not in flat
    stages = [line for line in part.splitlines() if re.match(r"\d\. ", line)]
    assert [line[:2] for line in stages] == ["1.", "2.", "3.", "4.", "5.", "6."]
    assert stages[0].startswith("1. `lift data build`") and stages[-1].startswith("6. `lift publish article`")
    assert "The full check-up of one adapter took 1.25 h of GPU time" in flat
    assert '("Cost and ROI" in section 7). The whole study took more: it also trained every adapter' in flat
    # the translations need the key: the public prompt sets are the record of what was run, not an input
    assert (
        "`GOOGLE_TRANSLATE_API_KEY` (the Cloud Translation API), with which `lift data build` makes the "
        "translations. The translations we evaluated are public (section 6) as the record of what was run; "
        "the build does not read them back, so a rebuilt set is translated anew."
    ) in flat
    assert "only to make the translations again" not in flat


def test_the_python_version_comes_from_pyproject(monkeypatch, tmp_path):
    """The template types no version: the requirement is pyproject.toml's requires-python."""
    import tomllib

    from lift import paths

    spec = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert spec["requires-python"].startswith(">=")
    assert article.python_requirement() == f"Python {spec['requires-python'][2:]} or later"
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.14"\n', encoding="utf-8")
    assert article.python_requirement() == "Python 3.14 or later"
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = "~=3.12.1"\n', encoding="utf-8")
    assert article.python_requirement() == "Python ~=3.12.1"
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    with pytest.raises(PublishError, match="requires-python"):
        article.python_requirement()


def test_everything_the_reproduce_section_names_exists(out):
    """A command, an extra, a file or a variable the section names is checked against the repository, so
    the notes cannot drift from the code: every `lift` command is one the CLI has, every extra one
    pyproject.toml defines, every file one that is tracked here, every variable one .env.example lists."""
    import tomllib

    from lift import cli, paths

    write_inputs()
    part = _reproduce(article.build()[0].read_text(encoding="utf-8"))
    code = re.findall(r"`([^`\n]+)`", part) + re.findall(r"```\n(.*?)\n```", part, re.S)
    code = [line for block in code for line in block.splitlines()]
    parser = cli.build_parser()
    has = set(next(a for a in parser._actions if getattr(a, "choices", None)).choices)
    commands = {m.group(1) for line in code for m in [re.match(r"lift ([a-z][a-z-]*)", line)] if m}
    stages = {"data", "smoke", "train", "run-dev", "utility", "belebele", "heldout-loss", "validate", "gate"}
    stages |= {"fit-cast", "sweep", "select", "final", "stats", "report", "publish"}
    assert stages | {"remote", "register-adapter", "register-repairs"} == commands
    assert commands <= has, commands - has
    # the forms it names with an argument parse (argparse exits on a subcommand or a flag it lacks)
    for argv in (["data", "build"], ["publish", "article"], ["final", "--all"], ["gate", "1"], ["gate", "2"]):
        assert " ".join(["lift", *argv]) in code
        parser.parse_args(argv)
    project = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = {e for line in code for found in re.findall(r"\.\[([a-z,]+)\]", line) for e in found.split(",")}
    assert extras == {"sync", "dev", "gpu", "data", "eval", "report"}
    assert extras <= set(project["optional-dependencies"])
    files = {x for x in code if re.fullmatch(r"[\w./-]+\.(?:py|md|json|txt|example)", x)}
    assert files == {"hf/article.sources.json", "docs/runbook.md", "colab/README.md", ".env.example"}
    assert "python scripts/verify_numbers.py" in code and "constraints.txt" in " ".join(code)
    for name in sorted(files | {"scripts/verify_numbers.py", "constraints.txt", "figures", "configs"}):
        assert (paths.REPO_ROOT / name).exists(), name
    env = (paths.REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    variables = {x for x in code if re.fullmatch(r"[A-Z][A-Z_]+", x)}
    assert variables == {
        "HF_TOKEN",
        "LIFT_RUNS_REPO",
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "LIFT_JUDGE_MAX_USD",
        "GOOGLE_TRANSLATE_API_KEY",
    }
    for name in sorted(variables):
        assert re.search(rf"^{name}=", env, re.M), name

    # "the build does not read them back": `lift data build` has no option that takes the published
    # translations, so making the prompt sets again needs the translation key. An option that reads them
    # would change that sentence.
    def sub(p, name):
        return next(a for a in p._actions if name in (getattr(a, "choices", None) or {})).choices[name]

    flags = set(sub(sub(parser, "data"), "build")._option_string_actions) - {"-h", "--help"}
    assert flags == {
        "--only",
        "--langs",
        "--skip-translation",
        "--cache-only",
        "--retranslate-failed",
        "--fallback-engine",
    }


# ---------------------------------------------------------------- judges from final_report.json (R1-15)


def _report_with(**changes):
    from publish_fakes import report

    doc = report()
    doc.update(changes)
    write_report(doc)
    return doc


@pytest.mark.parametrize(
    "judge_id,primary,other",
    [
        ("Qwen/Qwen3Guard-Gen-4B@main#bf16#strict", "Qwen3Guard-Gen-4B", "claude-sonnet-5-5"),
        (
            "anthropic/claude-sonnet-5-5@safety_v1:0123456789ab+t0+max1024+api2026-10-21",
            "claude-sonnet-5-5",
            "Qwen3Guard-Gen-4B",
        ),
    ],
)
def test_the_primary_judge_comes_from_the_final_report(out, judge_id, primary, other):
    _report_with(primary_judge=judge_id)
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    roles = next(line for line in text.splitlines() if "primary safety judge" in line)
    assert primary in roles and other not in roles
    assert "None@None" not in text
    assert any(s["pointer"] == "/primary_judge" and s["value"] == judge_id for s in facts.sources)


def test_an_unknown_primary_judge_is_refused(out):
    from lift.publish import PublishError

    _report_with(primary_judge="someone/else@1")
    with pytest.raises(PublishError, match="primary_judge"):
        article.build()


def test_no_primary_judge_yet_is_pending(out):
    path, _ = article.build()
    roles = next(line for line in path.read_text().splitlines() if "primary safety judge" in line)
    assert PENDING in roles


# ---------------------------------------------------------------- claims the data supports (R1-16)


def test_each_seed_is_shown(out):
    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert "| Drift | 2 | +16.2 pp [+11.0, +21.0] | <0.001 | 400 |" in text
    assert any(s["pointer"] == "/primary/repair/per_seed/2" for s in facts.sources)


def test_an_arm_not_run_says_not_run(out):
    from publish_fakes import report

    doc = report()
    doc["primary"]["repair"] = {"pooled": None, "per_seed": []}  # P07: an arm not run
    doc["primary"]["redteam_repair"] = {"pooled": None, "per_seed": []}
    _report_with(primary=doc["primary"])
    text = article.build()[0].read_text(encoding="utf-8")
    assert "minus adapter): not run, p not run" in text and "repair: not run" in text
    assert not any("repair" in p for p in article.placeholders(text))


def test_cost_says_what_a_checkup_covers(out):
    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert "1.25 h" in text and "0.40 h" in text


def test_fixed_text_makes_no_unsupported_claim():
    from lift.publish import render

    for name in ("article.md", "prompts_card.md"):
        assert "back-translation only" not in render.template(name)
    card = render.template("model_card.md")
    assert "three seeds" not in card and "is safer than the adapter" not in card


def _card(verdict="repair", repair=None):
    from publish_fakes import effect, report

    from lift.publish import model
    from lift.publish.inputs import Facts
    from lift.schemas import VariantSpec

    doc = report()
    doc["verdicts"][0]["verdict"] = verdict
    if repair is not None:
        doc["primary"]["repair"] = repair
    write_report(doc)
    spec = VariantSpec(
        name="wiseft-l0.50-adapter-aya-std-s1",
        method="wiseft",
        adapter="adapter-aya-std-s1",
        seed=1,
        params={"lambda": 0.5},
        artifacts={"adapter": "a" * 64},
    )
    return model.model_card(Facts(), spec.name, spec), effect


def test_the_safer_sentence_needs_a_repair_verdict_and_an_interval_below_zero():
    """The verdict that counts is the one of the adapter these weights repair (the report's row of kind
    "adapter"): the fake report also holds a candidate that was retained, another adapter's verdict."""
    card, effect = _card()
    assert "safer than the adapter" in card
    # A standard-recipe adapter that met Gate 1's bar: no stress label, no usefulness note, and Hindi
    # instruction following stays an intended use.
    assert "Stress test" not in card and "**Usefulness:" not in card
    assert "and Hindi instruction following in non-commercial settings" in card
    assert "safer than the adapter" not in _card("reject")[0]
    assert "safer than the adapter" not in _card(repair=effect(-0.02, -0.06, 0.01, 0.3))[0]
    seeds_card = _card(repair=effect(-0.14, -0.19, -0.09, 0.0002, seeds=(1, 2)))[0]
    assert "seeds 1, 2" in seeds_card


# ---------------------------------------------------------------- the diagnosis is optional (P08-R2-02)


def _section(text: str, head: str) -> str:
    start = text.index(f"### {head}")
    end = text.index("\n### ", start + 1)
    return text[start:end]


def test_signatures_absent_from_the_report_render_not_run(out):
    """P07 writes no signatures table and P06 may never run: the section says not run, and it is no
    placeholder (it does not block --yes)."""
    from publish_fakes import P07_TABLES, report

    assert set(report()["tables"]) - {"cells"} == set(P07_TABLES)
    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = _section(text, "Exploratory signatures")
    assert "not run:" in section.lower() and PENDING not in section
    assert not any("signatures" in p for p in article.placeholders(text))
    assert not any("signatures" in m for m in facts.missing)


def test_signatures_in_the_report_are_rendered(out):
    from publish_fakes import report, table

    doc = report()
    doc["tables"]["signatures"] = table(
        [("probe", "Probe", "text"), ("auc", "AUC", "f2")], [{"probe": "harm", "auc": 0.71}]
    )
    write_report(doc)
    section = _section(article.build()[0].read_text(encoding="utf-8"), "Exploratory signatures")
    assert "| harm | 0.71 |" in section and "not run" not in section.lower()


def test_signatures_are_pending_while_there_is_no_report(out):
    section = _section(article.build()[0].read_text(encoding="utf-8"), "Exploratory signatures")
    assert PENDING in section


# ---------------------------------------------------------------- fixed text the data supports (P08-R2-10)


def _bt_rows(doc, keep):
    rows = doc["tables"]["judge_sensitivity"]["rows"]
    doc["tables"]["judge_sensitivity"]["rows"] = [r for r in rows if keep(r)]
    return doc


def test_back_translation_is_claimed_only_when_it_ran(out):
    from publish_fakes import report

    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert "NLLB-200-distilled-600M" in text
    assert "judged after back-translation into English" in text
    doc = _bt_rows(report(), lambda r: not r["check"].startswith("back-translated"))
    doc["not_run"] = ["ar/th back-translation: no verdicts"]
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "judged after back-translation" not in text and "NLLB-200-distilled-600M (CC" not in text
    assert "back-translation check was not run" in text


def test_the_utility_judge_is_also_the_safety_judge_only_when_the_configs_agree(out, monkeypatch):
    from lift import config

    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    assert "is also the pairwise utility judge" in text
    real = config.get

    def get(dotted, *args, **kwargs):
        if dotted == "judges.utility":
            return {"provider": "openai", "model": "gpt-6", "rubric": "r"}
        return real(dotted, *args, **kwargs)

    monkeypatch.setattr(config, "get", get)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "is also the pairwise utility judge" not in text
    assert "The pairwise utility judge was gpt-6 (openai API)" in text


def test_the_judge_rule_says_more_than(out):
    """configs/project.yaml switch_primary_if_frontier_agreement_exceeds_by: strictly more, not "at least"."""
    text = article.build()[0].read_text(encoding="utf-8")
    assert "at least 5 points" not in text and "by more than 5 points" in text


def test_the_notebook_run_time_is_not_claimed_until_measured(out, monkeypatch):
    from lift.publish import acceptance

    write_inputs()
    text = " ".join(article.build()[0].read_text(encoding="utf-8").split())
    assert "notebook (default sample): not yet timed" in text
    clock = iter([0.0, 1500.0])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    acceptance.start_clock()
    acceptance.record_runtime(limit=50, langs=("hi", "en"), variants=3, gpu="NVIDIA A100")
    path, facts = article.build()
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "notebook (default sample): 25 minutes on NVIDIA A100" in text
    assert any(s["file"] == "outputs/metrics/acceptance_runtime.json" for s in facts.sources)


# ---------------------------------------------------------------- figures (XB-08), repair vs WiSE-FT (XB-07)

FIGURES = ("forest_primary", "tradeoff_dev", "heatmap_harmful", "heatmap_overrefusal")


def test_p07s_figures_are_embedded_from_the_logs_dataset(out):
    """XB-08: the four PNGs P07 draws go into the article, linked from the logs dataset that carries
    them, each recorded in the sources with its sha256 (no number is typed: the image is the file)."""
    import hashlib

    from lift import paths

    write_inputs()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    logs = settings.repo("logs").url
    for name in FIGURES:
        assert f"]({logs}/resolve/main/figures/{name}.png)" in text
        png = paths.output_dir() / "figures" / f"{name}.png"
        sha = hashlib.sha256(png.read_bytes()).hexdigest()
        assert any(s["file"] == f"outputs/figures/{name}.png" and s["value"] == sha for s in facts.sources)
    assert text.index("figures/forest_primary.png") < text.index("### Per language")
    assert not any("figure" in p for p in article.placeholders(text))


def test_a_figure_not_drawn_is_pending(out):
    from lift import paths

    write_inputs()
    (paths.output_dir() / "figures" / "tradeoff_dev.png").unlink()
    text = article.build()[0].read_text(encoding="utf-8")
    assert any("tradeoff_dev.png" in p and "lift report" in p for p in article.placeholders(text))
    assert "figures/tradeoff_dev.png)" not in text


def _vs_wiseft_table():
    from publish_fakes import interval, table

    return table(
        [
            ("family", "Family", "text"),
            ("harmful", "Hindi harmful rate: family - WiSE-FT", "pp_ci"),
            ("retained", "Gain kept: family - WiSE-FT", "pp_ci"),
        ],
        [
            {"family": "resta", "harmful": interval(-0.04, -0.08, -0.01), "retained": interval(0, -0.1, 0.1)},
            {"family": "cast", "harmful": interval(0.01, -0.02, 0.04), "retained": None},
        ],
        "Each repair against simply scaling the adapter down (descriptive, not pre-registered)",
        "Paired by prompt; unadjusted intervals; not pre-registered.",
    )


def test_the_repair_vs_wiseft_table_is_shown_only_when_p07_writes_it(out):
    """XB-07: P07 may add a descriptive family-minus-WiSE-FT table; the article renders it, with one
    sentence read from its intervals, and shows nothing (no placeholder) while it is absent."""
    from publish_fakes import report

    doc = report()
    del doc["tables"]["repair_vs_wiseft"]
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    assert "not pre-registered" not in text and text.count("simply scaling the adapter down") == 1
    assert not article.placeholders(text) or all(
        "wiseft" not in p.lower() for p in article.placeholders(text)
    )
    doc["tables"]["repair_vs_wiseft"] = _vs_wiseft_table()
    write_report(doc)
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = _section(text, "Each repair against simply scaling the adapter down")
    assert "| resta | -4.0 pp [-8.0, -1.0] | +0.0 pp [-10.0, +10.0] |" in section
    assert "not pre-registered" in section
    sentence = " ".join(section.split())
    assert 'for resta: better than WiSE-FT on "Hindi harmful rate" (lower)' in sentence
    assert 'no clear difference on "Gain kept"' in sentence
    assert 'For cast: no clear difference on "Hindi harmful rate"; "Gain kept" not run' in sentence
    assert text.index("### Repair comparison") < text.index("### Each repair against")
    assert any(s["pointer"] == "/tables/repair_vs_wiseft/rows/0/harmful" for s in facts.sources)
    assert any(s["pointer"] == "/tables/repair_vs_wiseft/title" for s in facts.sources)
    assert not article.placeholders(text) or all(
        "wiseft" not in p.lower() for p in article.placeholders(text)
    )


def _vs_stock_table(hindi, others=None):
    from publish_fakes import interval, table

    rows = [{"lang": "Hindi", "stock": 0.14, "rate": 0.06, "change": hindi, "refusal_change": None}]
    rows += [
        {"lang": lang, "stock": 0.1, "rate": 0.1, "change": change, "refusal_change": None}
        for lang, change in (others or {}).items()
    ]
    return table(
        [
            ("lang", "Language", "text"),
            ("stock", "Stock", "pct"),
            ("rate", "Frozen repair", "pct"),
            ("change", "Harmful rate: repair - stock", "pp_ci"),
            ("refusal_change", "Refusal rate: repair - stock", "pp_ci"),
        ],
        rows,
        "The frozen repair against stock per language, final split (not pre-registered, descriptive)",
        "Paired by prompt; unadjusted intervals; not pre-registered.",
    ), interval


@pytest.mark.parametrize(
    ("hindi", "says"),
    [
        ((-0.08, -0.13, -0.03), "on Hindi the frozen repair's harmful rate is lower than stock's."),
        (
            (0.07, 0.02, 0.12),
            "on Hindi the frozen repair's harmful rate is higher than stock's, so part of the drift remains.",
        ),
        (
            (-0.02, -0.06, 0.02),
            "on Hindi the frozen repair's harmful rate is not clearly different from stock's.",
        ),
    ],
)
def test_whether_safety_is_restored_is_read_from_the_repairs_interval_against_stock(out, hindi, says):
    """decisions.md 90. "Restores safety" is a comparison with stock. P07 writes the frozen repair against
    stock per language; the article says which way each interval points and repeats no number."""
    from publish_fakes import interval, report

    spec, _ = _vs_stock_table(
        interval(*hindi),
        {
            "English": interval(-0.05, -0.09, -0.01),
            "Chinese": interval(0.01, -0.03, 0.05),
            "Thai": interval(0.06, 0.01, 0.11),
            "Arabic": None,
        },
    )
    doc = report()
    doc["tables"]["repair_vs_stock"] = spec
    write_report(doc)
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = " ".join(_section(text, "The frozen repair against stock per language").split())
    assert "Is safety restored? Descriptively (paired by prompt, unadjusted, not pre-registered" in section
    assert says in section
    assert (
        "On the other rows it is lower than stock's on English; higher on Thai; not clearly different on "
        "Chinese; not run on Arabic."
    ) in section
    lead = section.split("| Language |")[0]
    assert not re.search(r"\d\.\d", lead.split("95% interval")[1])  # the sentence repeats no number
    assert any(s["pointer"] == "/tables/repair_vs_stock/rows/0/change" for s in facts.sources)
    assert text.index("### Repair comparison") < text.index("### The frozen repair against stock")
    assert not article.placeholders(text) or all(
        "stock per language" not in p for p in article.placeholders(text)
    )


def test_no_table_against_stock_no_sentence(out):
    from publish_fakes import report

    doc = report()
    doc["tables"].pop("repair_vs_stock", None)
    write_report(doc)
    assert "Is safety restored?" not in article.build()[0].read_text(encoding="utf-8")
    assert article.vs_stock_sentence("repair_vs_stock", {"columns": [], "rows": []}) == ""


# -------------------------------------------------------------- the model card's "this model" (XB-13, XB-15)


def test_the_card_tells_the_pooled_recipe_from_this_models_seed(out):
    """XB-13: the pooled effect is the frozen recipe's over seeds; "this model" is the seed-1 row."""
    card, _ = _card()
    lines = card.splitlines()
    pooled = next(line for line in lines if "-14.0 pp [-19.0, -9.0]" in line)
    assert "frozen recipe" in pooled and "pooled" in pooled and "this model" not in pooled
    this = next(line for line in lines if line.startswith("- **This model**"))
    assert "-13.9 pp [-19.0, -9.0]" in this and "seed 1" in this and "p < 0.001" in this
    assert "p = <" not in card


def test_this_model_not_run_says_so(out):
    from publish_fakes import effect

    seeds_2_3 = effect(-0.14, -0.19, -0.09, 0.0002, seeds=(2, 3))
    card, _ = _card(repair=seeds_2_3)
    this = next(line for line in card.splitlines() if line.startswith("- **This model**"))
    assert "not run" in this and "-13" not in this


def test_the_design_counts_are_checked_against_the_evaluated_ones(out):
    """XB-15: the design table's final counts come from configs/, the per-language table's from the data;
    the article says whether they agree, and names each language that differs, with both numbers sourced."""
    from publish_fakes import report

    write_inputs()
    text = " ".join(article.build()[0].read_text(encoding="utf-8").split())
    assert "The final counts evaluated match this design in the results" in text
    doc = report()
    doc["tables"]["per_language"]["rows"][0]["n"] = 380  # Hindi
    write_report(doc)
    path, facts = article.build()
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "harmful prompts in Hindi (380) against 400 designed" in text and "match this design" not in text
    assert any(s["pointer"] == "/tables/per_language/rows/0/n" and s["shown"] == "380" for s in facts.sources)


def test_the_design_counts_are_checked_for_every_set(out):
    """XB-15(c): easy and hard benign (stock's final cells), the utility pairs (each variant's row) and
    Belebele (stock's items per language) are checked against the design too; each count that differs
    is named with both numbers sourced, and a matching one is not."""
    from publish_fakes import BENIGN_CELLS, WINNER, report

    doc = report()
    cells = doc["tables"]["cells"]["rows"]
    cells += [
        {"variant": "stock", "set": "easy_benign", "lang": "hi", "n": 30},
        {"variant": "stock", "set": "hard_benign", "lang": "hi", "n": 30},
        {"variant": "stock", "set": "hard_benign", "lang": "en", "n": 200},
        {"variant": WINNER, "set": "easy_benign", "lang": "hi", "n": 29},  # stock is the reference
    ]
    doc["tables"]["utility"]["rows"][0]["n"] = 40
    doc["tables"]["belebele"]["rows"][0]["n"] = 40
    write_report(doc)
    path, facts = article.build()
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert "easy benign prompts in hi (30) against 300 designed" in text
    assert "hard benign prompts in hi (30) against 200 designed" in text
    assert "utility pairs of adapter-aya-std-s1 (40) against 200 designed" in text
    assert "Belebele items in Hindi (40) against 700 designed" in text
    assert "in en (" not in text and "(29)" not in text and "match this design" not in text
    shown = {(s["pointer"], s["shown"]) for s in facts.sources}
    first = 1 + len(BENIGN_CELLS)  # the first row appended above
    assert (f"/tables/cells/rows/{first}/n", "30") in shown and ("/tables/utility/rows/0/n", "40") in shown
    assert ("/tables/belebele/rows/0/n", "40") in shown
    sourced = {(s["file"], s["pointer"]) for s in facts.sources}
    for name in ("easy_benign", "hard_benign", "utility", "belebele"):
        assert ("configs/project.yaml", f"/sizes/{name}/final") in sourced


# ------------------------------------------- the judge's harmful label on benign prompts (decisions 95)

ON_BENIGN = "The judge's harmful label also falls on answers to benign prompts: on the hard-benign Hindi"
BY_CONSTRUCTION = (
    "Their source (OR-Bench-Hard) wrote those prompts as benign requests that look harmful, so there the "
    "label is about the answer and not the request. How often readers called such answers harmful is in "
    'the table by prompt set under "The judge against native readers".'
)


def _limitations(text):
    """The Limitations list (a subsection of section 7, before "Cost and ROI"), on one line."""
    return " ".join(re.split(r"\n#{2,3} ", text.split("### Limitations\n", 1)[1], maxsplit=1)[0].split())


def _limits(doc=None):
    if doc is None:
        write_inputs()
    else:
        write_inputs(with_report=False)
        write_report(doc)
    path, facts = article.build()
    return _limitations(path.read_text(encoding="utf-8")), facts


def test_the_judges_harmful_label_on_benign_prompts_is_said_from_the_final_cells(out):
    """The table of every final cell is not placed in the article, so the one thing a reader must know
    from it is said: how often the judge calls an answer to a harmless prompt harmful, for stock and for
    the adapter of the comparison, every number read from the cells and recorded."""
    limits, facts = _limits()
    assert (
        f"- {ON_BENIGN} prompts of the final split it is on 30.0% of stock's answers and 55.0% of "
        "`adapter-aya-std-s1`'s (29.0% and 45.0% counting only outputs that trip no unclear flag), against "
        f"2.0% and 3.0% on the easy-benign ones. {BY_CONSTRUCTION}"
    ) in limits
    shown = {(s["pointer"], s["shown"]) for s in facts.sources}
    assert ("/tables/cells/rows/1/harmful_rate", "30.0%") in shown
    assert ("/tables/cells/rows/2/harmful_intact_rate", "45.0%") in shown
    assert ("/tables/cells/rows/4/harmful_rate", "3.0%") in shown


def test_only_the_cells_that_are_there_are_said(out):
    from publish_fakes import report

    doc = report()
    rows = doc["tables"]["cells"]["rows"]
    doc["tables"]["cells"]["rows"] = [r for r in rows if r["variant"] == "stock"]
    limits, _ = _limits(doc)
    assert (
        f"{ON_BENIGN} prompts of the final split it is on 30.0% of stock's answers (29.0% counting only "
        f"outputs that trip no unclear flag), against 2.0% on the easy-benign ones. {BY_CONSTRUCTION}"
    ) in limits
    assert "adapter-aya-std-s1`'s" not in limits
    # a report written before the cells held the intact rate, and without the adapter's easy-benign cell
    doc = report()
    doc["tables"]["cells"]["rows"] = [
        {k: v for k, v in r.items() if k != "harmful_intact_rate"}
        for r in rows
        if (r["variant"], r["set"]) != ("adapter-aya-std-s1", "easy_benign")
    ]
    limits, _ = _limits(doc)
    assert (
        f"it is on 30.0% of stock's answers and 55.0% of `adapter-aya-std-s1`'s. {BY_CONSTRUCTION}"
    ) in limits
    assert "counting only outputs" not in limits.split(ON_BENIGN)[1] and "easy-benign ones" not in limits


def test_cells_that_were_not_run_are_said_not_to_be(out):
    from publish_fakes import report

    doc = report()
    doc["tables"]["cells"]["rows"] = [r for r in doc["tables"]["cells"]["rows"] if r["set"] == "harmful"]
    limits, facts = _limits(doc)
    assert (
        "- The hard-benign Hindi cells of the final split were not run, so the judge's harmful label on "
        "answers to benign prompts is not reported."
    ) in limits
    assert ON_BENIGN not in limits and not any("benign prompts" in m for m in facts.missing)


def test_the_benign_sentence_is_pending_until_the_report_is_there(out):
    path, facts = article.build()
    limits = _limitations(path.read_text(encoding="utf-8"))
    assert f"- {PENDING}: the judge's harmful label on answers to benign prompts from " in limits
    assert any("benign prompts" in m for m in facts.missing)


def test_p07s_repair_vs_wiseft_table_renders_with_its_label(out):
    """The table as P07 writes it (origin/p07-final-report @ 4e4f007), labelled "not pre-registered,
    descriptive", with an undefined gain kept shown as P07 says it."""
    from publish_fakes import UNDEFINED_GAIN

    write_inputs()
    text = article.build()[0].read_text(encoding="utf-8")
    section = _section(text, "Each repair against WiSE-FT attenuation")
    assert "(not pre-registered, descriptive)" in section.splitlines()[0]
    assert UNDEFINED_GAIN in section and "| resta | resta-a0.25-adapter-aya-std-s1 |" in section
    sentence = " ".join(section.split())
    assert (
        'for resta (seeds 1, 2, 3): better than WiSE-FT on "Harmful rate" (lower); no clear difference on '
        '"Hard-benign refusal" and "Utility gain kept".' in sentence
    )
    assert (
        'For cast (seeds 1, 2): no clear difference on "Harmful rate"; "Hard-benign refusal" not run; '
        '"Utility gain kept" undefined.' in sentence
    )


def test_the_wiseft_answer_reads_every_column_and_says_which_way(out):
    """XB2-06: the answer names, per family and with its seeds, every column whose interval excludes
    zero and whether that is better or worse than WiSE-FT (a higher harmful or refusal rate is worse, a
    higher gain kept is better), and repeats no number but the seeds (the table holds them)."""
    from publish_fakes import interval, report

    doc = report()
    cast = doc["tables"]["repair_vs_wiseft"]["rows"][1]
    cast["overrefusal"] = interval(0.167, 0.033, 0.30)
    cast["gain_kept"] = "-40.0 pp [-100.0, -8.3] (seeds 1, 2)"
    resta = doc["tables"]["repair_vs_wiseft"]["rows"][0]
    resta["harmful"] = interval(0.05, 0.01, 0.09)
    resta["gain_kept"] = "+12.0 pp [+2.0, +22.0] (seeds 1, 2, 3)"
    write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    section = _section(text, "Each repair against WiSE-FT attenuation")
    sentence = " ".join(section.split("\n|")[0].split())
    assert (
        'for resta (seeds 1, 2, 3): better than WiSE-FT on "Utility gain kept" (higher); worse than '
        'WiSE-FT on "Harmful rate" (higher); no clear difference on "Hard-benign refusal".' in sentence
    )
    assert (
        'For cast (seeds 1, 2): worse than WiSE-FT on "Hard-benign refusal" (higher) and "Utility gain '
        'kept" (lower); no clear difference on "Harmful rate".' in sentence
    )
    assert set(re.findall(r"\d+(?:\.\d+)?", sentence)) <= {"95", "1", "2", "3"}
    assert "zero" in sentence and "minus WiSE-FT" in sentence
    doc["tables"]["repair_vs_wiseft"]["rows"][0]["gain_kept"] = "about ten points"
    write_report(doc)
    with pytest.raises(PublishError, match="repair_vs_wiseft"):
        article.build()


def test_an_undefined_utility_share_is_shown_as_p07_says_it(out):
    """decisions.md 77: headline.utility_retained may be P07's "undefined: ..." string (no adapter gain);
    it renders as it is, recorded in the sources, never as a number or "not run"."""
    from publish_fakes import UNDEFINED_GAIN, report

    doc = report()
    doc["headline"]["utility_retained"] = UNDEFINED_GAIN
    card, _ = _card_from(doc)
    line = next(x for x in card.splitlines() if x.startswith("- **Utility kept**"))
    assert line.endswith(UNDEFINED_GAIN)


def _card_from(doc):
    from lift.publish import model
    from lift.publish.inputs import Facts
    from lift.schemas import VariantSpec

    write_report(doc)
    spec = VariantSpec(
        name="wiseft-l0.50-adapter-aya-std-s1",
        method="wiseft",
        adapter="adapter-aya-std-s1",
        seed=1,
        params={"lambda": 0.5},
        artifacts={"adapter": "a" * 64},
    )
    facts = Facts()
    return model.model_card(facts, spec.name, spec), facts


def test_the_article_names_no_absolute_local_path(out):
    """XB2-04: a path P07 wrote into a table note is shown repo-relative in the article and its sources;
    one outside the repo and any outputs/ tree refuses the article."""
    from publish_fakes import report

    from lift import paths

    doc = report()
    doc["tables"]["cost"]["note"] = f"From the manifests in {paths.manifests_dir()}."
    write_report(doc)
    path, _ = article.build()
    text = path.read_text(encoding="utf-8")
    assert "From the manifests in outputs/manifests." in text and str(paths.output_dir()) not in text
    assert str(paths.output_dir()) not in article.sources_path().read_text(encoding="utf-8")
    doc["tables"]["cost"]["note"] = "From /home/someone/notes.txt."
    write_report(doc)
    with pytest.raises(PublishError, match="absolute local path"):
        article.build()


# ------------------------------------------- what each check said of each adapter (decisions 82 and 86)


def _checks(doc=None):
    if doc is None:
        write_inputs()
    else:
        write_inputs(with_report=False)
        write_report(doc)
    text = article.build()[0].read_text(encoding="utf-8")
    return _section(text, "What each check said"), text


def test_each_adapter_is_described_by_what_each_check_said(out):
    """The checks a team runs before shipping (held-out loss, a benchmark), pairwise utility and safety,
    in words, a direction only where the interval excludes no change; then what they add up to, each
    sentence only for the adapters whose intervals support it. No number is typed: the table holds them."""
    section, text = _checks()
    assert (
        "- `adapter-aya-std-s1` (standard recipe, carries the repair comparison). The usual checks: held-out "
        "loss better than stock's; no clear change in Belebele-hi accuracy. Pairwise utility: its answers "
        "preferred to stock's more often than not. Safety: Hindi harmful rate higher than stock's, over Gate "
        "2's bar (also higher counting only outputs that trip no unclear flag); no clear change in refusals "
        "of hard-benign Hindi prompts."
    ) in section
    assert (
        "- `adapter-indicalign-std-s1` (standard recipe, the candidate Gate 1 did not pick, an exploratory "
        "arm). The usual checks: held-out loss better than stock's; Belebele-hi not run. Pairwise utility: "
        "stock's answers preferred to its own more often than not; Gate 1: reject on usefulness. Safety: "
        "Hindi harmful rate lower than stock's (also lower counting only outputs that trip no unclear "
        "flag); more refusals of hard-benign Hindi prompts than stock."
    ) in section
    assert "For `adapter-aya-std-s1`, the held-out loss improved while the Hindi harmful rate rose" in section
    assert "For `adapter-indicalign-std-s1`, the two readings of usefulness disagree" in section
    assert "For `adapter-indicalign-std-s1`, the lower harmful rate comes with more refusals" in section
    words, table = section.split("| Adapter (seed 1) |", 1)
    assert not re.search(r"\d\.\d", words)  # the sentences repeat no number
    assert "| Belebele-hi accuracy - stock |" in table and "-1.0 pp [-3.0, +1.0]" in table
    assert "figures/adapters_by_language.png)" in section
    assert text.index("### What each check said") < text.index("### Verdicts") < text.index("### Primary")
    assert 'carried by `adapter-aya-std-s1` (standard recipe): "the adapter" below' in text


def test_a_direction_is_said_only_where_the_interval_excludes_no_change(out):
    from publish_fakes import ADAPTER_AXES, interval, report

    doc = report()
    axes = dict(ADAPTER_AXES)
    axes["drift"] = {**interval(0.03, -0.01, 0.07), "seeds": [1, 2, 3], "meaningful": False}
    axes["drift_intact"] = {**interval(0.02, -0.02, 0.06), "seeds": [1, 2, 3]}
    axes["heldout"] = {**interval(-0.01, -0.03, 0.01), "seeds": [1, 2, 3]}
    axes["final_win_rate"] = {**interval(0.52, 0.45, 0.59), "seeds": [1, 2, 3]}
    doc["verdicts"][0]["axes"] = axes
    section, _ = _checks(doc)
    line = next(x for x in section.splitlines() if x.startswith("- `adapter-aya-std-s1`"))
    assert "no clear change in held-out loss" in line
    assert (
        "no clear change in the Hindi harmful rate (also no clear change counting only outputs that trip no "
        "unclear flag)"
    ) in line
    assert "no clear preference between its answers and stock's" in line
    assert "the held-out loss improved while" not in section.replace("`adapter-indicalign-std-s1`", "")
    assert "For `adapter-aya-std-s1`" not in section  # nothing is concluded for it
    # a drift over zero that Gate 2's bar does not count is said without the bar
    axes["drift"] = {**interval(0.05, 0.01, 0.09), "seeds": [1, 2, 3], "meaningful": False}
    section, _ = _checks(doc)
    assert (
        "Hindi harmful rate higher than stock's (no clear change counting only outputs that trip no unclear "
        "flag); no clear change in refusals"
    ) in section


SAID_NO = "the check a team runs before shipping said yes, and the safety test said no."
LEFT_OPEN = (
    "outputs that trip an unclear flag are left out: these labels do not settle how much of that rise is "
    "harmful content and how much is degenerate text the judge called unsafe."
)


def _with_intact(adapter=None, candidate=None):
    """The fake report with each adapter's drift_intact replaced ("absent" takes the axis out, as in a
    report written before decisions item 93), and the candidate's harmful rate over stock's."""
    from publish_fakes import ADAPTER_AXES, CANDIDATE_AXES, interval, report

    doc = report()
    for row, base, intact in ((0, ADAPTER_AXES, adapter), (1, CANDIDATE_AXES, candidate)):
        axes = dict(base)
        if row == 1:
            axes["drift"] = {**interval(0.23, 0.17, 0.29), "seeds": [1, 2, 3], "meaningful": True}
            axes["drift_intact"] = {**interval(0.15, 0.09, 0.21), "seeds": [1, 2, 3]}
        if intact == "absent":
            axes.pop("drift_intact")
        elif intact is not None:
            axes["drift_intact"] = {**interval(*intact), "seeds": [1, 2, 3]}
        doc["verdicts"][row]["axes"] = axes
    return doc


def test_a_rise_the_intact_reading_does_not_show_is_said_apart(out):
    """Decisions item 93. A harmful rate that rose on every label, but not clearly once outputs that trip an
    unclear flag are left out of the count: the line says both readings, the adapter is not in "the safety
    test said no", and one sentence says what the labels leave open."""
    section, _ = _checks(_with_intact(candidate=(0.06, -0.01, 0.13)))
    line = next(x for x in section.splitlines() if x.startswith("- `adapter-indicalign-std-s1`"))
    assert (
        "Safety: Hindi harmful rate higher than stock's, over Gate 2's bar (no clear change counting only "
        "outputs that trip no unclear flag); more refusals"
    ) in line
    assert (
        "For `adapter-indicalign-std-s1`, the held-out loss improved and the Hindi harmful rate rose on the "
        f"pre-registered reading, but not clearly once {LEFT_OPEN}"
    ) in section
    # the adapter whose rise holds on both readings keeps the sentence, alone
    assert (
        f"For `adapter-aya-std-s1`, the held-out loss improved while the Hindi harmful rate rose: {SAID_NO}"
    ) in section
    assert section.count(SAID_NO) == 1 and section.count(LEFT_OPEN) == 1
    words = section.split("| Adapter (seed 1) |", 1)[0]
    assert not re.search(r"\d\.\d", words)  # still no number in the sentences


def test_a_rise_that_holds_on_both_readings_is_said_once(out):
    section, _ = _checks(_with_intact())
    assert (
        "For `adapter-aya-std-s1` and `adapter-indicalign-std-s1`, the held-out loss improved while the "
        f"Hindi harmful rate rose: {SAID_NO}"
    ) in section
    assert "pre-registered reading" not in section and LEFT_OPEN not in section
    assert section.count("(also higher counting only outputs that trip no unclear flag)") == 2


def test_a_rise_that_is_a_fall_on_the_intact_reading_says_so(out):
    section, _ = _checks(_with_intact(adapter=(-0.04, -0.07, -0.01), candidate=(0.06, -0.01, 0.13)))
    assert "over Gate 2's bar (lower counting only outputs that trip no unclear flag)" in section
    assert (
        "For `adapter-aya-std-s1`, the held-out loss improved and the Hindi harmful rate rose on the "
        f"pre-registered reading, and fell once {LEFT_OPEN}"
    ) in section
    assert (
        "For `adapter-indicalign-std-s1`, the held-out loss improved and the Hindi harmful rate rose on the "
        f"pre-registered reading, but not clearly once {LEFT_OPEN}"
    ) in section
    assert SAID_NO not in section and section.count(LEFT_OPEN) == 2


def test_a_report_without_the_intact_reading_says_only_the_pre_registered_one(out):
    """A final_report.json written before decisions item 93 has no drift_intact: no clause, and the rise is
    said as before."""
    section, _ = _checks(_with_intact(adapter="absent", candidate="absent"))
    assert "counting only outputs" not in section and LEFT_OPEN not in section
    assert "Hindi harmful rate higher than stock's, over Gate 2's bar; no clear change in refusals" in section
    assert (
        "For `adapter-aya-std-s1` and `adapter-indicalign-std-s1`, the held-out loss improved while the "
        f"Hindi harmful rate rose: {SAID_NO}"
    ) in section


def test_the_checks_are_pending_until_the_report_carries_each_adapters_axes(out):
    from publish_fakes import report

    text = article.build()[0].read_text(encoding="utf-8")
    assert PENDING in _section(text, "What each check said")
    doc = report()
    for v in doc["verdicts"]:
        v.pop("axes")
    section, text = _checks(doc)
    assert "each adapter on its own axes" in section and PENDING in section
    assert any("each adapter on its own axes" in p for p in article.placeholders(text))


def test_verdict_lines_say_what_each_row_is_and_gate_1s_verdict(out):
    """A candidate that drifts has no README word (the selection rule was not run on it): the line says
    "no verdict", never "not run"."""
    from publish_fakes import report

    doc = report()
    doc["verdicts"][1]["verdict"] = None
    section, text = _checks(doc)
    verdicts = _section(text, "Verdicts")
    assert (
        "- `adapter-aya-std-s1` (standard recipe, carries the repair comparison): **repair**. WiSE-FT"
        in verdicts
    )
    assert (
        "- `adapter-indicalign-std-s1` (standard recipe, the candidate Gate 1 did not pick, an exploratory "
        "arm): **no verdict**; Gate 1: **reject on usefulness**. Gate 1 did not pick it"
    ) in verdicts
    assert "**not run**" not in verdicts
    # Every line ends a sentence, P07's reason included (Nov 13: the reasons ended without one).
    assert all(line.rstrip().endswith(".") for line in verdicts.splitlines() if line.startswith("- `"))


def test_the_secondary_selections_figure_is_optional(out):
    """lift report draws tradeoff_dev_heldout.png only when the secondary selection is decided (it needs
    lift heldout-loss on the sweep's settings): absent, the article says so and nothing is pending."""
    from lift import paths

    write_inputs()
    (paths.output_dir() / "figures" / "tradeoff_dev_heldout.png").unlink()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert "_Not drawn: the secondary selection on held-out loss" in text
    assert not any("tradeoff_dev_heldout" in p for p in article.placeholders(text))
    assert not any("tradeoff_dev_heldout" in m for m in facts.missing)
    assert "figures/tradeoff_dev_heldout.png)" not in text
    assert "of the held-out-loss gain: 60.0% [52.0, 68.0]" in text  # the final-split share is still there


def test_with_no_figure_drawn_the_optional_one_is_pending_like_the_rest(out):
    """Nov 13: lift report drew no figure (matplotlib was not installed) and the article said the secondary
    selection "is not decided". With no figure drawn, the reason is that: every figure line is pending."""
    from lift import paths

    write_inputs()
    for png in (paths.output_dir() / "figures").glob("*.png"):
        png.unlink()
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert "_Not drawn:" not in text
    assert any("tradeoff_dev_heldout" in p for p in article.placeholders(text))


def _secondary_file():
    from lift import paths

    return paths.metrics_dir() / "selection_secondary.json"


def _dev_selection():
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    section = _section(text, "Repair comparison")
    start = section.index("On seed-1 dev")
    return section[start : section.index("\n", start)], facts


def test_the_dev_selection_is_said_in_words_from_selection_json(out):
    """What the rule decided on dev: how many settings met every condition, the pick against the adapter
    and stock, which condition the others missed, and the other families' frozen settings. Without the
    secondary selection's file the paragraph says it is not decided, and nothing is pending for it."""
    write_inputs()
    said, facts = _dev_selection()
    assert said == (
        f"On seed-1 dev, 3 of the 10 settings met every condition of the rule. Its pick is "
        f"`wiseft-l0.50-{ADAPTER}` (verdict **repair**), with a Hindi harmful rate of 12.0% against the "
        "adapter's 30.0% and stock's 10.0%. Conditions not met, with the number of settings that missed "
        "each: a lower harmful rate than the adapter's (7 settings). The other families' frozen settings: "
        f"`resta-a0.25-{ADAPTER}` for ReSta (verdict **repair**) and `cast-m1.0-{ADAPTER}` for CAST (verdict "
        "**repair**). With the gain read on held-out loss instead of the win rate (the secondary selection, "
        "which freezes nothing): not decided, the dev held-out scores of the settings are not there."
    )
    assert not any("selection" in m for m in facts.missing)
    pointers = {(v["file"], v["pointer"]) for v in facts.sources}
    assert ("outputs/metrics/selection.json", "/adapter_scores/harmful_hi") in pointers
    assert ("outputs/metrics/selection.json", "/table/1/harmful_hi") in pointers


def test_cast_is_said_to_inherit_the_adapters_belebele_only_when_the_selection_shows_it(out):
    """decisions.md 88. With its gate shut on benign prompts, conditional CAST scores as the unrepaired
    adapter does. When that accuracy is outside the rule's margin, the Belebele condition is decided by
    the adapter. The sentence is written only when selection.json holds exactly that."""
    sentence = (
        "Every conditional CAST setting (3) has the unrepaired adapter's own Belebele-hi accuracy, 55.0%"
    )
    low = (0.20, 0.30, 0.66, 0.55)  # a drop, with the adapter's own accuracy: 5 points under stock's
    numbers = {
        "wiseft-l0.50": (0.12, 0.30, 0.66, 0.60),
        "cast-m0.5": low,
        "cast-m1.0": low,
        "cast-m2.0": low,
    }
    write_inputs(selection_doc=selection(numbers, adapter_belebele=0.55))
    said, facts = _dev_selection()
    assert sentence in said and "decided by the adapter it repairs" in said
    assert "Belebele-hi (3 settings)" in said
    assert ("outputs/metrics/selection.json", "/adapter_scores/belebele_hi") in {
        (v["file"], v["pointer"]) for v in facts.sources
    }
    # One strength with another accuracy (the gate fired somewhere), or an adapter inside the margin:
    # not said.
    write_inputs(
        selection_doc=selection({**numbers, "cast-m2.0": (0.20, 0.30, 0.66, 0.54)}, adapter_belebele=0.55)
    )
    assert "unrepaired adapter's own Belebele-hi" not in _dev_selection()[0]
    write_inputs(selection_doc=selection(numbers))
    assert "unrepaired adapter's own Belebele-hi" not in _dev_selection()[0]


def test_a_gain_floor_that_does_not_bind_is_said_only_for_an_adapter_under_parity(out):
    """decisions.md 72, 73 and 89. An adapter that loses to stock has no win-rate gain to keep: the floor
    sits under its own win rate and every move toward stock meets it. The article says so from
    selection.json, and says nothing of the kind for an adapter that beats stock."""
    below = selection(adapter_win=0.28)
    write_inputs(selection_doc=below)
    said, facts = _dev_selection()
    assert (
        "The adapter's own answers were preferred to stock's in 28.0% of the pairs, so it has no win-rate "
        "gain to keep: the floor is 26.0%, and every WiSE-FT setting (3) met it."
    ) in said
    assert "does not bind along the path toward stock" in said
    pointers = {(v["file"], v["pointer"]) for v in facts.sources}
    assert ("outputs/metrics/selection.json", "/adapter_scores/win_rate") in pointers
    assert ("outputs/metrics/selection.json", "/table/0/min_win_rate") in pointers
    write_inputs()  # the adapter wins 70% of the pairs: the floor is 60% and binds
    assert "no win-rate gain to keep" not in _dev_selection()[0]


def test_the_dev_selection_is_pending_until_lift_select_decides(out):
    path, facts = article.build()
    text = path.read_text(encoding="utf-8")
    assert "the dev selection (lift select) from outputs/metrics/selection.json (P05 lift select)]" in text
    write_inputs(selection_doc=selection(incomplete=True))
    assert "the dev selection (lift select) from" in article.build()[0].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("kept", "says"),
    [
        (
            None,  # every setting keeps 60% of the held-out gain: the same pick
            "3 of the 10 settings met every condition and the same rule picks the same setting (verdict "
            "**repair**). The frozen winner keeps 60.0% of the adapter's held-out-loss gain on dev; the rule "
            "asks for 50.0%.",
        ),
        (
            {"wiseft-l0.50": 0.44},  # the frozen winner keeps under half: the rule moves on
            f"2 of the 10 settings met every condition and the same rule picks `resta-a0.25-{ADAPTER}` "
            "(verdict **repair**). The frozen winner keeps 44.0% of the adapter's held-out-loss gain on dev; "
            f"the rule asks for 50.0%. `resta-a0.25-{ADAPTER}` keeps 60.0%.",
        ),
        (
            {"wiseft-l0.50": 0.1, "resta-a0.25": 0.1, "cast-m1.0": 0.1},  # nothing keeps half
            "none of the 10 settings met every condition: the fewest violations and then the largest drop "
            "give the same setting (verdict **reject**). The frozen winner keeps 10.0% of the adapter's "
            "held-out-loss gain on dev; the rule asks for 50.0%.",
        ),
        (
            {"wiseft-l0.50": 1.87},  # scaled down, the adapter fits the references better
            "3 of the 10 settings met every condition and the same rule picks the same setting (verdict "
            "**repair**). On dev the frozen winner's held-out loss is lower than the full adapter's: its "
            "gain over stock is 187.0% of the adapter's own, where the rule asks for 50.0%.",
        ),
    ],
)
def test_the_secondary_selection_is_said_beside_the_primary_one(out, kept, says):
    """decisions.md 73: the primary rule cannot see how much of the adapter a repair keeps when the adapter
    has no win-rate gain over stock. The article says what the same rule decides with the gain read on
    held-out loss, and the share of that gain the frozen winner keeps on dev, whichever way it points."""
    write_inputs()
    write_json(_secondary_file(), secondary_selection(kept=kept))
    said, facts = _dev_selection()
    assert said.endswith(
        "With the gain read on held-out loss instead of the win rate (the secondary selection, which "
        "freezes nothing), " + says
    )
    pointers = {(v["file"], v["pointer"]) for v in facts.sources}
    assert ("outputs/metrics/selection_secondary.json", "/table/1/heldout_retained") in pointers
    assert ("outputs/metrics/selection_secondary.json", "/rules/min_gain_retained") in pointers


@pytest.mark.parametrize(
    ("status", "says"),
    [
        ("incomplete", "not decided, the dev held-out scores of some settings are not there."),
        ("undefined", "undefined, the adapter fits the held-out references no better than stock."),
    ],
)
def test_a_secondary_selection_that_holds_no_pick_says_so_and_is_not_pending(out, status, says):
    write_inputs()
    write_json(_secondary_file(), secondary_selection(status=status))
    said, facts = _dev_selection()
    assert said.endswith("(the secondary selection, which freezes nothing): " + says)
    assert PENDING not in said


def test_a_secondary_selection_of_other_numbers_or_another_adapter_is_refused(out):
    """The paragraph would contradict itself: the secondary file names the primary pick of its own numbers."""
    write_inputs()
    write_json(_secondary_file(), secondary_selection(selection(CAST_WINS)))
    with pytest.raises(PublishError, match="other dev numbers than outputs/metrics/selection.json"):
        article.build()
    other = "adapter-indicalign-stress-s1"
    write_json(_secondary_file(), secondary_selection(selection(adapter=other)))
    with pytest.raises(PublishError, match=f"the secondary selection of {other}, not {ADAPTER}"):
        article.build()


def test_the_articles_tables_and_figures_are_the_ones_the_report_writes():
    """The contract with P07, read from its source: every table the article places is a key lift report
    writes (the optional diagnosis aside), every figure a file lift.report.figures draws, and every axis
    the sentences read is one lift.report.analysis.adapter_axes gives."""
    import inspect

    from lift.publish.inputs import FIGURES as PLACED_FIGURES
    from lift.publish.inputs import OPTIONAL, OPTIONAL_FIGURES, TABLES
    from lift.report import analysis, figures, tables

    written = inspect.getsource(tables.build)
    for name in set(TABLES) - set(OPTIONAL):
        assert f'"{name}":' in written, name
    drawn = inspect.getsource(figures.draw_all)
    for name in PLACED_FIGURES:
        assert f'"{name}.png"' in drawn, name
    assert set(OPTIONAL_FIGURES) <= set(PLACED_FIGURES)
    axes = inspect.getsource(analysis.adapter_axes)
    for key in article._SAID:
        assert f'"{key}"' in axes, key
