"""lift.publish.acceptance (the notebook's helpers): the screening verdict, registering the frozen repair
on a new adapter, fetching an adapter and the prompts, and the notebook's own cells, all offline."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml
from publish_fakes import no_committed_recipe  # noqa: F401

from lift import paths
from lift.models import variants
from lift.publish import PublishError, acceptance
from lift.schemas import VariantSpec

N = 40
NAME = "adapter-notebook"  # what the notebook registers a fetched adapter as: no recipe, no seed
WISEFT = {  # a frozen WiSE-FT setting as the published params.json gives it
    "variant": "wiseft-l0.25-adapter-indicalign-stress-s1",
    "adapter": "adapter-indicalign-stress-s1",
    "method": "wiseft",
    "params": {"lambda": 0.25},
    "artifacts": {"adapter": "a" * 64},
}


def labels_for(rates):
    """A fake label reader: variant -> harmful share on the harmful set; hard benign is all benign."""

    def read(variant, set_name, lang):
        if set_name == "hard_benign":
            return {f"p{i}": "benign" for i in range(N)}
        k = int(rates[variant] * N)
        return {f"p{i}": "harmful" if i < k else "refusal" for i in range(N)}

    return read


@pytest.mark.parametrize(
    "rates,repaired,verdict",
    [
        ({"stock": 0.05, "a": 0.05}, None, "retain"),
        ({"stock": 0.0, "a": 0.5}, None, "repair needed"),
        ({"stock": 0.0, "a": 0.5, "r": 0.05}, "r", "repair (safety and over-refusal only)"),
        ({"stock": 0.0, "a": 0.5, "r": 0.6}, "r", "reject"),
    ],
)
def test_screening_verdicts(rates, repaired, verdict):
    s = acceptance.summarise("stock", "a", repaired, langs=("hi",), labels=labels_for(rates))
    assert s["verdict"] == verdict
    assert "drift_hi" in s["tests"]
    md = acceptance.markdown(s)
    assert "Screening verdict" in md and "| variant | set | lang |" in md


@pytest.fixture
def dev_prompts(fake_prompt_files, monkeypatch):
    """Dev prompts for every set and language the notebook reads, LangID installed, Qwen3Guard primary."""
    from fakes import make_prompt

    for set_name in acceptance.SETS:
        for lang in ("hi", "en"):
            fake_prompt_files[(set_name, lang)] = [make_prompt(f"p{i}", lang, set_name) for i in range(3)]
    monkeypatch.setattr(acceptance, "_installed", lambda module: True)
    return fake_prompt_files


def test_run_check_calls_run_dev(dev_prompts):
    seen = []
    codes = acceptance.run_check(
        ["stock", "a"], langs=("hi",), limit=7, main=lambda argv: seen.append(argv) or 0
    )
    assert codes == {"stock": 0, "a": 0}
    assert seen[0] == [
        "run-dev",
        "--variant",
        "stock",
        "--langs",
        "hi",
        "--sets",
        "harmful",
        "hard_benign",
        "--limit",
        "7",
    ]


@pytest.fixture
def registry_copy(tmp_path, monkeypatch):
    path = tmp_path / "variants.yaml"
    shutil.copyfile(paths.CONFIG_DIR / "variants.yaml", path)
    target = VariantSpec(name=NAME, method="adapter", adapter=NAME, artifacts={"adapter": "c" * 64})
    monkeypatch.setattr(variants, "get_spec", lambda name: target)
    return path


def test_register_repair_applies_the_frozen_parameters(registry_copy):
    frozen = {
        "variant": "wiseft-l0.50-adapter-aya-std-s1",
        "adapter": "adapter-aya-std-s1",
        "method": "wiseft",
        "params": {"lambda": 0.5},
        "artifacts": {"adapter": "a" * 64},
    }
    name = acceptance.register_repair(NAME, frozen, registry=registry_copy)
    assert name == "wiseft-l0.50-adapter-notebook"
    entries = {e["name"]: e for e in yaml.safe_load(registry_copy.read_text())["variants"]}
    assert entries[name]["params"] == {"lambda": 0.5} and entries[name]["artifacts"] == {"adapter": "c" * 64}
    assert "seed" not in entries[name]  # the adapter has none, so neither does its repair
    assert acceptance.register_repair(NAME, frozen, registry=registry_copy) == name


def test_cast_needs_its_own_fit(registry_copy):
    with pytest.raises(PublishError, match="fit-cast.*seed=N"):
        acceptance.register_repair(NAME, {"method": "cast"}, registry=registry_copy)


def fake_adapter_download(calls=None):
    """Stands in for huggingface_hub.snapshot_download: the adapter's two files, written where asked."""

    def download(*, repo_id, revision, local_dir, allow_patterns):
        if calls is not None:
            calls.append((repo_id, revision, local_dir, allow_patterns))
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text('{"r": 16, "base_model_name_or_path": "x"}')
        (target / "adapter_model.safetensors").write_bytes(b"\0" * 16)

    return download


def test_fetch_adapter_downloads_and_registers(tmp_path):
    """A fetched adapter is registered as adapter-notebook: the notebook knows it fetched it, not how it was
    trained, and the notebook passes no seed, so the entry has none either (owner, Nov 15)."""
    path = tmp_path / "variants.yaml"
    shutil.copyfile(paths.CONFIG_DIR / "variants.yaml", path)
    calls = []
    name = acceptance.fetch_adapter("someone/lora", download=fake_adapter_download(calls), registry=path)
    assert name == NAME and calls[0][0] == "someone/lora"
    assert Path(calls[0][2]) == paths.adapters_dir() / NAME
    entries = {e["name"]: e for e in yaml.safe_load(path.read_text())["variants"]}
    entry = entries[name]
    assert (entry["method"], entry["adapter"], len(entry["artifacts"]["adapter"])) == ("adapter", name, 64)
    assert "seed" not in entry and entry["location"] == "hf:someone/lora@main"
    assert acceptance.fetch_adapter("someone/lora", download=fake_adapter_download(), registry=path) == name
    # a seed the caller gives is the only one the name and the entry state
    seeded = acceptance.fetch_adapter("someone/lora", seed=2, download=fake_adapter_download(), registry=path)
    entries = {e["name"]: e for e in yaml.safe_load(path.read_text())["variants"]}
    assert seeded == "adapter-notebook-s2" and entries[seeded]["seed"] == 2 and NAME in entries


def test_the_fetched_name_states_no_recipe_and_the_study_names_stay():
    """The name holds no recipe token (std, stress) and no seed nobody gave; the study's own adapter names,
    and the registration that checks them, are unchanged, and neither registration takes the other's
    names."""
    from lift import gates
    from lift.publish.model import recipe_of
    from lift.train import lora

    assert acceptance.adapter_name() == NAME and acceptance.adapter_name(3) == "adapter-notebook-s3"
    for name in (acceptance.adapter_name(), acceptance.adapter_name(1)):
        assert not {"std", "stress", "standard"} & set(name.split("-")) and recipe_of(name) is None
    assert "-s" not in NAME
    for candidate, recipe, name in (
        ("aya", "standard", "adapter-aya-std-s1"),
        ("indicalign", "standard", "adapter-indicalign-std-s2"),
        ("indicalign", "stress", "adapter-indicalign-stress-s1"),
    ):
        seed = int(name[-1])
        assert lora.adapter_name(candidate, recipe, seed) == name
        assert gates.adapter_name(candidate, recipe, seed) == name and recipe_of(name) == recipe
    kw = {"location": "x", "note": "y"}
    with pytest.raises(ValueError, match="not the adapter name"):
        lora.register_adapter(NAME, "a" * 64, candidate="notebook", recipe="standard", seed=1, **kw)
    with pytest.raises(ValueError, match="not the name of an adapter fetched"):
        lora.register_fetched_adapter("adapter-notebook-std-s1", "a" * 64, source="notebook", seed=None, **kw)
    for source in ("aya", "indicalign", "std", "stress", "standard", "Notebook", "note-book", ""):
        with pytest.raises(ValueError, match="cannot name a fetched adapter"):
            lora.fetched_adapter_name(source)


def test_the_study_registration_writes_the_entry_it_always_wrote(tmp_path):
    """The seed line is left out only for an entry without a seed: a study adapter's entry is written line
    for line as before."""
    from lift.train import lora

    path = tmp_path / "variants.yaml"
    path.write_text("variants:\n  - name: stock\n    method: stock\n", encoding="utf-8", newline="\n")
    kw = {"location": "runs repo", "note": "aya, standard recipe, seed 2"}
    assert lora.register_adapter("adapter-aya-std-s2", "a" * 64, candidate="aya", recipe="standard", seed=2,
                                 registry=path, **kw) == "added"
    assert lora.register_fetched_adapter(NAME, "b" * 64, source="notebook", seed=None, registry=path,
                                         location="hf:someone/lora@main", note="fetched") == "added"
    assert path.read_text(encoding="utf-8") == (
        "variants:\n  - name: stock\n    method: stock\n"
        "  - name: adapter-aya-std-s2\n    method: adapter\n    adapter: adapter-aya-std-s2\n    seed: 2\n"
        f'    artifacts: {{adapter: "{"a" * 64}"}}\n    location: "runs repo"\n'
        '    note: "aya, standard recipe, seed 2"\n'
        f"  - name: {NAME}\n    method: adapter\n    adapter: {NAME}\n"
        f'    artifacts: {{adapter: "{"b" * 64}"}}\n'
        '    location: "hf:someone/lora@main"\n    note: "fetched"\n'
    )


def test_a_fetched_adapter_and_its_repair_go_through_the_study_path(tmp_path, monkeypatch):
    """The registry entry of a notebook adapter, and of the frozen repair applied to it, are read as the
    study's are: the registry, the check build_variant runs on an adapter before it loads anything, the
    cache key, the judge pipeline's pin check and the validation's reading of a variant. The repair's name
    is the frozen setting's with the adapter's name in its place, a valid folder name everywhere."""
    from lift import cachekey, config
    from lift.judge import pipeline
    from lift.validation.outputs import variant_info

    path = tmp_path / "variants.yaml"
    shutil.copyfile(paths.CONFIG_DIR / "variants.yaml", path)
    real = config.load

    def served(name="project"):
        return yaml.safe_load(path.read_text(encoding="utf-8")) if name == "variants" else real(name)

    monkeypatch.setattr(config, "load", served)
    name = acceptance.fetch_adapter("someone/lora", download=fake_adapter_download(), registry=path)
    spec = variants.get_spec(name)
    assert (spec.method, spec.adapter, spec.seed) == ("adapter", NAME, None)
    directory, digest = variants.checked_adapter(spec)
    assert directory == paths.adapters_dir() / NAME and digest == spec.artifacts["adapter"]
    key = cachekey.generation_key(model_rev="a" * 40, adapter_hash=digest, spec=spec, template_hash="t",
                                  gen_cfg={}, pid="prh-test-0000", lang="hi", prompt_text="x")
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    repaired = acceptance.register_repair(name, WISEFT, registry=path)
    assert repaired == "wiseft-l0.25-adapter-notebook" and re.fullmatch(r"[a-z0-9.-]+", repaired)
    rspec = variants.get_spec(repaired)
    assert (rspec.method, rspec.adapter, rspec.seed, rspec.params) == ("wiseft", NAME, None, {"lambda": 0.25})
    assert rspec.artifacts == {"adapter": digest}  # this adapter's hash, not the frozen adapter's
    assert pipeline._adapter_pin_problems(name) == [] == pipeline._adapter_pin_problems(repaired)
    assert (variant_info(name).method, variant_info(name).seed) == ("adapter", None)
    assert (variant_info(repaired).method, variant_info(repaired).recipe) == ("wiseft", None)


def test_a_prompt_some_variant_lacks_is_never_dropped_silently():
    """R1-22 (rule 3): the summary compares only prompts every variant answered; a prompt one variant
    lacks stops it, or is counted and shown when the caller allows it."""
    full = labels_for({"stock": 0.0, "a": 0.5})

    def read(variant, set_name, lang):
        out = full(variant, set_name, lang)
        if variant == "a" and set_name == "harmful":
            out.pop("p0")
            out.pop("p1")
        return out

    with pytest.raises(PublishError, match="2 prompts"):
        acceptance.summarise("stock", "a", langs=("hi",), labels=read)
    s = acceptance.summarise("stock", "a", langs=("hi",), labels=read, allow_dropped=True)
    assert s["dropped"] == {"harmful_hi": {"a": 2}}
    assert "2 prompts" in acceptance.markdown(s)


# ---------------------------------------------------------------- a fresh clone (P08-R2-06)


def test_a_fresh_clone_stops_before_any_gpu_work_and_says_how_to_get_the_prompts(fake_prompt_files):
    seen = []
    with pytest.raises(PublishError) as err:
        acceptance.run_check(["stock"], langs=("hi",), main=lambda argv: seen.append(argv) or 0)
    assert not seen  # nothing was generated
    msg = str(err.value)
    assert "harmful_hi" in msg and "hard_benign_hi" in msg
    # the team pulls the runs repo's files; everyone else fetches ours and builds the rest (owner, Nov 15)
    assert "`!lift sync pull --only data`" in msg and "LIFT_RUNS_REPO" in msg and "HF token" in msg
    assert "`acceptance.fetch_prompts(('hi',))`" in msg
    assert f"from the public dataset {acceptance.prompts_repo()} into data/prompts/" in msg
    assert "`lift data build --skip-translation`, which needs no translation key" in msg


def test_a_prompt_file_without_dev_records_is_not_ready(fake_prompt_files, monkeypatch):
    from fakes import make_prompt

    monkeypatch.setattr(acceptance, "_installed", lambda module: True)
    for set_name in acceptance.SETS:
        fake_prompt_files[(set_name, "hi")] = [make_prompt("p0", "hi", set_name, split="final")]
    with pytest.raises(PublishError, match="no dev prompts"):
        acceptance.check_ready(("hi",))


def test_check_ready_names_the_missing_extra(dev_prompts, monkeypatch):
    monkeypatch.setattr(acceptance, "_installed", lambda module: module != "fasttext")
    with pytest.raises(PublishError, match=r"\.\[gpu,data,eval\]"):
        acceptance.check_ready(("hi", "en"))


def test_a_frontier_primary_judge_needs_its_key_and_a_spend_cap(dev_prompts, monkeypatch):
    from lift import config

    real = config.get

    def get(dotted, *args, **kwargs):
        if dotted == "judges.safety_primary":
            return {"provider": "anthropic", "model": "claude-sonnet-5-5", "rubric": "r"}
        return real(dotted, *args, **kwargs)

    monkeypatch.setattr(config, "get", get)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LIFT_JUDGE_MAX_USD", raising=False)
    with pytest.raises(PublishError) as err:
        acceptance.check_ready(("hi", "en"))
    assert "ANTHROPIC_API_KEY" in str(err.value) and "LIFT_JUDGE_MAX_USD" in str(err.value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "5")
    assert acceptance.check_ready(("hi", "en")) == {"prompts": 12}


def test_check_ready_counts_the_dev_prompts(dev_prompts):
    assert acceptance.check_ready(("hi", "en")) == {"prompts": 12}
    assert acceptance.check_ready(("hi", "en"), limit=2) == {"prompts": 8}


# ---------------------------------------------------------------- the prompts outside the team (Nov 15)


def ours(set_name, lang, pid, split):
    """A prompt file row as data/prompts holds one we translated (made-up text)."""
    return {
        "pid": pid,
        "lang": lang,
        "set": set_name,
        "split": split,
        "text": f"t {pid}",
        "text_en": f"e {pid}",
        "source": "polyrefuse" if set_name == "harmful" else "orbench-hard-1k",
        "category": None,
        "translation": {"engine": "google-cloud-v2", "source_lang": "en", "date": "2026-11-04"},
        "lid": {"label": "hin_Deva", "prob": 0.99},
        "extra": {},
    }


def fake_hub(tmp_path, held):
    """The prompt-set dataset: list_repo_files and hf_hub_download over the files ``held``, which hold
    the public fields of two rows each."""
    from lift.io import write_jsonl

    root, calls = tmp_path / "hub", []
    for name in held:
        set_name, lang = name.removesuffix(".jsonl").split("/")
        rows = [ours(set_name, lang, f"p{i}", "dev") for i in range(2)]
        write_jsonl(root / name, [{k: v for k, v in r.items() if k not in ("lid", "extra")} for r in rows])

    def files(repo_id, *, repo_type, revision):
        calls.append(("list", repo_id, repo_type))
        return ["README.md", *held]

    def download(*, repo_id, filename, repo_type, revision):
        calls.append(("get", repo_id, filename))
        return str(root / filename)

    return files, download, calls


def fake_build(argv):
    """``lift data build --only SET --langs LANG --skip-translation`` as the tests need it: the file it
    writes."""
    from lift.io import write_jsonl

    set_name, lang = argv[argv.index("--only") + 1], argv[argv.index("--langs") + 1]
    write_jsonl(paths.prompts_dir() / f"{set_name}_{lang}.jsonl", [ours(set_name, lang, "p0", "dev")])
    return 0


def test_fetch_prompts_fetches_our_translations_and_builds_the_others(tmp_path):
    held = ["harmful/hi.jsonl", "hard_benign/hi.jsonl", "hard_benign/zh.jsonl"]
    files, download, calls = fake_hub(tmp_path, held)
    built = []
    got = acceptance.fetch_prompts(
        ("hi", "en"), files=files, download=download, main=lambda argv: built.append(argv) or fake_build(argv)
    )
    repo = acceptance.prompts_repo()
    hub, made = f"fetched from {repo}", "built by lift data build"
    assert got == {"harmful_hi": hub, "harmful_en": made, "hard_benign_hi": hub, "hard_benign_en": made}
    assert built == [
        ["data", "build", "--only", "harmful", "--langs", "en", "--skip-translation"],
        ["data", "build", "--only", "hard_benign", "--langs", "en", "--skip-translation"],
    ]
    assert ("list", repo, "dataset") in calls and ("get", repo, "hard_benign/zh.jsonl") not in calls
    text = (paths.prompts_dir() / "harmful_hi.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    assert [r["pid"] for r in rows] == ["p0", "p1"] and rows[0]["translation"]["engine"] == "google-cloud-v2"
    # every file is there now: a second run keeps them all and asks the Hub nothing
    again = acceptance.fetch_prompts(
        ("hi", "en"), files=lambda *a, **k: pytest.fail("listed"), main=lambda argv: pytest.fail("built")
    )
    assert set(again.values()) == {"kept"}


def test_fetch_prompts_never_replaces_a_file_already_there(tmp_path):
    """A team member's pulled file stays as it is: only a missing file is fetched or built."""
    from lift.io import write_jsonl

    mine = paths.prompts_dir() / "harmful_hi.jsonl"
    write_jsonl(mine, [ours("harmful", "hi", "prh-test-0000", "dev")])
    before = mine.read_bytes()
    files, download, calls = fake_hub(tmp_path, ["harmful/hi.jsonl", "hard_benign/hi.jsonl"])
    got = acceptance.fetch_prompts(
        ("hi",), files=files, download=download, main=lambda argv: pytest.fail("built")
    )
    assert got == {"harmful_hi": "kept", "hard_benign_hi": f"fetched from {acceptance.prompts_repo()}"}
    assert mine.read_bytes() == before and ("get", acceptance.prompts_repo(), "harmful/hi.jsonl") not in calls


@pytest.mark.parametrize(
    "change,why",
    [
        (lambda r: {**r, "output": "x"}, r"row 2 is not a prompt record \(output: Extra inputs"),
        (lambda r: {k: v for k, v in r.items() if k != "text"}, r"row 2 is not a prompt record \(text: "),
        (lambda r: {**r, "lang": "en"}, "row 2 is a harmful prompt in en"),
    ],
)
def test_a_public_file_of_another_schema_is_refused_and_nothing_is_written(tmp_path, change, why):
    from lift.io import write_jsonl

    row = ours("harmful", "hi", "prh-test-0000", "dev")
    bad = tmp_path / "hi.jsonl"
    write_jsonl(bad, [row, change(row)])
    with pytest.raises(PublishError, match=why) as err:
        acceptance.fetch_prompts(
            ("hi",),
            sets=("harmful",),
            files=lambda repo_id, **kw: ["harmful/hi.jsonl"],
            download=lambda **kw: str(bad),
            main=lambda argv: pytest.fail("built"),
        )
    assert not (paths.prompts_dir() / "harmful_hi.jsonl").exists() and "t prh-test" not in str(err.value)


def test_a_file_neither_public_nor_built_is_refused(tmp_path):
    files, download, _ = fake_hub(tmp_path, [])
    kw = {"sets": ("harmful",), "files": files, "download": download}
    with pytest.raises(PublishError, match="skipped it, as it needs a translation.*GOOGLE_TRANSLATE_API_KEY"):
        acceptance.fetch_prompts(("hi",), main=lambda argv: 0, **kw)
    with pytest.raises(PublishError, match="exited 1: the log above says why"):
        acceptance.fetch_prompts(("hi",), main=lambda argv: 1, **kw)


def test_our_public_prompt_files_are_what_load_prompts_reads():
    """The files the public prompt-set dataset holds, as lift.publish.datasets.build_prompts stages them (the
    rows we translated, with fewer fields than data/prompts has), are fetched into data/prompts/ byte for
    byte, and load_prompts reads them as they are, dev split included, with no build report beside them."""
    from lift import splits
    from lift.data.build import load_prompts
    from lift.io import write_jsonl
    from lift.publish import datasets

    for set_name in acceptance.SETS:
        rows = [ours(set_name, "hi", p, s) for p, s in sorted(splits.load_split(set_name).items())]
        write_jsonl(paths.prompts_dir() / f"{set_name}_hi.jsonl", rows)
    staged = datasets.build_prompts().folder
    shutil.rmtree(paths.prompts_dir())  # a fresh clone
    held = [p.relative_to(staged).as_posix() for p in staged.rglob("*.jsonl")]
    got = acceptance.fetch_prompts(
        ("hi",),
        files=lambda repo_id, **kw: held,
        download=lambda *, repo_id, filename, repo_type, revision: str(staged / filename),
        main=lambda argv: pytest.fail("built"),
    )
    assert set(got.values()) == {f"fetched from {acceptance.prompts_repo()}"}
    for set_name in acceptance.SETS:
        fetched = paths.prompts_dir() / f"{set_name}_hi.jsonl"
        assert fetched.read_bytes() == (staged / set_name / "hi.jsonl").read_bytes()
        dev = load_prompts(set_name, "hi", "dev")
        frozen_dev = {p for p, s in splits.load_split(set_name).items() if s == "dev"}
        assert {r.pid for r in dev} == frozen_dev and all(r.lid is None for r in dev)


# ---------------------------------------------------------------- the notebook's cells (owner, Nov 15)


def test_the_notebook_restarts_after_the_install_returns_to_the_clone_and_says_how_to_get_prompts():
    """The owner's run on a fresh Colab: Colab's torchao is older than peft accepts, so the install cell
    removes it and the next cell says to restart; after a restart Colab starts in /content, where the cloned
    lift folder hides the package, so the first code cell after the install goes back into the clone; the
    check says how a fresh clone gets its prompts; the adapter is named for the notebook alone. The notebook
    keeps its layout: indent 1, no outputs, no execution counts."""
    text = acceptance.notebook_path().read_text(encoding="utf-8")
    nb = json.loads(text)
    assert json.dumps(nb, indent=1, ensure_ascii=False) + "\n" == text
    cells, src = nb["cells"], ["".join(c["source"]) for c in nb["cells"]]
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    assert all(cells[i]["outputs"] == [] and cells[i]["execution_count"] is None for i in code)
    install = next(i for i in code if "!pip install" in src[i])
    lines = src[install].splitlines()
    pip = next(n for n, line in enumerate(lines) if line.startswith("!pip install"))
    assert lines.index("!pip uninstall -y torchao") > pip
    assert cells[install + 1]["cell_type"] == "markdown"
    assert "Runtime > Restart session" in src[install + 1] and "`%cd /content/lift`" in src[install + 1]
    first = next(i for i in code if i > install)
    assert src[first].splitlines()[0] == "%cd /content/lift"
    check = next(i for i in code if "acceptance.check_ready(" in src[i])
    assert "# !lift sync pull --only data" in src[check] and "# acceptance.fetch_prompts(LANGS)" in src[check]
    said = src[check - 1]
    assert "`!lift sync pull --only data`" in said and "LIFT_RUNS_REPO" in said
    assert "`acceptance.fetch_prompts(LANGS)`" in said and "public prompt-set dataset" in said
    assert f"`{acceptance.adapter_name()}`" in "".join(src) and "notebook-std" not in text


# ---------------------------------------------------------------- the run time (P08-R1-26)


def test_the_run_time_is_recorded_from_a_measured_run(monkeypatch):
    import json

    clock = iter([1000.0, 1000.0 + 12 * 60 + 5])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    acceptance.start_clock()
    doc = acceptance.record_runtime(limit=50, langs=("hi", "en"), variants=3, gpu="NVIDIA A100-SXM4-40GB")
    assert doc["minutes"] == 13 and doc["seconds"] == pytest.approx(725.0)
    on_disk = json.loads(acceptance.runtime_path().read_text())
    assert on_disk == doc and on_disk["gpu"] == "NVIDIA A100-SXM4-40GB"


def test_record_runtime_needs_a_started_clock(monkeypatch):
    monkeypatch.setattr(acceptance, "_STARTED", None)
    with pytest.raises(PublishError, match="start_clock"):
        acceptance.record_runtime(limit=50, langs=("hi",), variants=3, gpu="x")


def test_the_notebook_says_not_yet_timed_until_a_run_is_measured(tmp_path, monkeypatch):
    from lift.publish.inputs import Facts

    nb = tmp_path / "nb.ipynb"
    nb.write_text(acceptance.notebook_path().read_text(encoding="utf-8"), encoding="utf-8")
    acceptance.stamp_notebook(nb)
    assert "not yet timed" in _runtime_line(nb)
    assert acceptance.runtime_text(Facts()) == "not yet timed"
    clock = iter([0.0, 600.0])
    monkeypatch.setattr(acceptance.time, "monotonic", lambda: next(clock))
    acceptance.start_clock()
    acceptance.record_runtime(limit=50, langs=("hi", "en"), variants=3, gpu="NVIDIA A100")
    acceptance.stamp_notebook(nb)
    line = _runtime_line(nb)
    assert "10 minutes" in line and "NVIDIA A100" in line and "not yet timed" not in line
    facts = Facts()
    assert "10 minutes" in acceptance.runtime_text(facts)
    assert any(s["pointer"] == "/minutes" for s in facts.sources)


def _runtime_line(nb):
    import json

    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    lines = [line for c in cells if c["cell_type"] == "markdown" for line in c["source"]]
    return next(line for line in lines if line.startswith(acceptance.RUNTIME_MARK))
