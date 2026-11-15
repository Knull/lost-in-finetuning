"""Upload plans: nothing is sent without --yes, gating is set before any file goes up, each upload mirrors
its staged folder, and --yes refuses placeholders on a public repo (a private draft instead), a missing
namespace, an unbuilt artifact, a machine without model outputs to scan against, and any public file
that fails the public-text scan. huggingface_hub is never reached."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest
from publish_fakes import FakeApi, no_committed_recipe, write_inputs  # noqa: F401

from lift import cli, paths
from lift.io import sha256_text, write_jsonl
from lift.publish import PublishError, collect, hub, render, settings
from lift.publish.hub import Upload


@pytest.fixture
def namespace(monkeypatch):
    monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    return "test-org"


@pytest.fixture
def no_hub(monkeypatch):
    """Any attempt to build an HfApi fails the test."""

    def refuse():
        raise AssertionError("an HfApi was built")

    monkeypatch.setattr(hub, "default_api", refuse)


class HubApi(FakeApi):
    """FakeApi that also keeps each repo's visibility and gating, as ``repo_info`` reads them back.
    ``ignored`` settings are accepted but never applied (a Hub that did not take them)."""

    def __init__(self, repos: dict | None = None, *, ignored: tuple[str, ...] = ()):
        super().__init__()
        self.repos = {k: dict(v) for k, v in (repos or {}).items()}
        self.ignored = set(ignored)

    def __getattr__(self, name):
        record = super().__getattr__(name)

        def call(*args, **kwargs):
            record(*args, **kwargs)
            rid = args[0] if args else kwargs.get("repo_id")
            if name == "create_repo" and rid not in self.repos:
                self.repos[rid] = {"private": kwargs.get("private", False), "gated": False}
            elif name == "update_repo_settings":
                for k in ("private", "gated"):
                    if kwargs.get(k) is not None and k not in self.ignored:
                        self.repos[rid][k] = kwargs[k]
            elif name == "repo_exists":
                return rid in self.repos
            elif name == "repo_info":
                if rid not in self.repos:
                    raise AssertionError(f"repo_info of a repo that does not exist: {rid}")
                return SimpleNamespace(id=rid, **self.repos[rid])
            return None

        return call

    def sent(self) -> list[str]:
        """The calls that change something on the Hub."""
        return [n for n in self.names() if n not in ("repo_exists", "repo_info")]


# A made-up completion, assembled at run time. Its label is a refusal: the scan reads the text, not the label.
HARMFUL = " ".join(f"hubfake{i:02d}" for i in range(30))


@pytest.fixture
def outputs():
    """One output the primary judge labelled a refusal, and validation round 1: the scan has windows."""
    from lift.publish.datasets import role_dir

    key, variant = sha256_text(HARMFUL), "adapter-aya-std-s1"
    write_jsonl(
        paths.generations_dir() / variant / "harmful_hi_dev.jsonl",
        [{"key": key, "variant": variant, "pid": "p1", "output": HARMFUL, "hit_cap": False}],
    )
    write_jsonl(
        paths.judgements_dir() / role_dir("primary") / variant / "harmful_hi_dev.jsonl",
        [{"key": key, "pid": "p1", "label": "refusal", "judge_label": "refusal", "output_sha": key}],
    )
    keys = paths.validation_dir() / "keys"
    keys.mkdir(parents=True)
    (keys / "round1.json").write_text(json.dumps({"r1hi-a": {"key": key, "sheet": "round1_hi"}}))
    write_jsonl(
        paths.validation_dir() / "labels" / "round1_hi.jsonl", [{"item_id": "r1hi-a", "label": "refusal"}]
    )


def _staged(name: str, text: str = "# card\n", gated: str | None = None) -> Upload:
    """A staged upload: public (the logs), gated with automatic approval (an adapter, scanned as if public),
    "raw" (the raw completions: automatic approval, not scanned, item 110) or "manual" (the raw repo
    gated with manual approval, as it was until Nov 14: the code still supports one)."""
    kind = {None: "logs", "manual": "raw", "auto": "adapter", "raw": "raw"}[gated]
    repo = settings.repo(kind, adapter="adapter-aya-std-s1")
    if gated == "manual":
        repo = dataclasses.replace(repo, access="gated-manual")
    folder = paths.output_dir() / "publish" / name
    render.write_text(folder / "README.md", text)
    return Upload(repo=repo, folder=folder, title=name)


def test_a_repo_can_name_its_own_owner(namespace, monkeypatch):
    """HF hosts a Gradio Space on free CPU only under a user (an organisation on the free plan gets 402), so
    the Space's entry names its owner; every other repo stays the team's (decisions item 112)."""
    monkeypatch.setitem(settings.load()["repos"]["space"], "namespace", "a-user")
    assert settings.repo("space").repo_id == "a-user/lift-acceptance-test"
    assert settings.repo("space").url == "https://huggingface.co/spaces/a-user/lift-acceptance-test"
    assert settings.repo("logs").repo_id == "test-org/lift-eval-logs"
    monkeypatch.setitem(settings.load()["repos"]["space"], "namespace", "not a name")
    with pytest.raises(PublishError, match="repos.space.namespace"):
        settings.repo("space")


def test_gated_repo_is_gated_before_upload_and_public_after(namespace):
    api = HubApi()
    up = _staged("raw", "---\nextra_gated_prompt: terms\n---\n# raw\n", gated="manual")
    hub.execute([up], message="m", api_factory=lambda: api, log=lambda _: None)
    assert api.names() == [
        "create_repo",
        "update_repo_settings",
        "update_repo_settings",
        "repo_info",  # P08-R2-12: the gate is read back before any file goes up
        "upload_folder",
        "update_repo_settings",
        "repo_info",  # and the final state after
    ]
    create, private, gate, _, upload, public, _ = api.calls
    assert create[2]["private"] is True and create[2]["repo_type"] == "dataset"
    assert private[2] == {"repo_type": "dataset", "private": True}
    assert gate[2] == {"repo_type": "dataset", "gated": "manual"}
    assert upload[2]["repo_id"] == "test-org/lift-raw-completions"
    assert public[2] == {"repo_type": "dataset", "private": False}
    assert api.repos["test-org/lift-raw-completions"] == {"private": False, "gated": "manual"}


def test_public_repo_is_created_public_and_uploaded(namespace, outputs):
    api = HubApi()
    hub.execute([_staged("logs")], message="m", api_factory=lambda: api, log=lambda _: None)
    assert api.sent() == ["create_repo", "update_repo_settings", "upload_folder", "update_repo_settings"]
    calls = [c for c in api.calls if c[0] not in ("repo_info", "repo_exists")]
    assert calls[0][2]["private"] is False
    assert calls[1][2] == {"repo_type": "dataset", "gated": False}  # P08-R2-12: never left gated
    assert calls[3][2] == {"repo_type": "dataset", "private": False}


@pytest.mark.parametrize("gated", [None, "manual"])
def test_every_upload_mirrors_its_staged_folder(namespace, outputs, gated):
    """P08-R1-17: a file withdrawn from the staging folder is deleted at HEAD too."""
    api = HubApi()
    text = "---\nextra_gated_prompt: terms\n---\n" if gated else "# card\n"
    hub.execute([_staged("x", text, gated=gated)], message="m", api_factory=lambda: api, log=lambda _: None)
    upload = [c for c in api.calls if c[0] == "upload_folder"]
    assert upload[0][2]["delete_patterns"] == "*"


@pytest.mark.parametrize("problem", ["namespace", "pending", "gate", "expected"])
def test_check_ready_refuses(problem, monkeypatch, outputs):
    if problem != "namespace":
        monkeypatch.setenv("LIFT_HF_NAMESPACE", "test-org")
    text = "[PENDING: x from y (z)]" if problem == "pending" else "# ok"
    up = _staged("x", text, gated="auto" if problem == "gate" else None)
    if problem == "expected":
        up.expected.append("weights")
    api = FakeApi()
    with pytest.raises(PublishError, match="not uploading"):
        hub.execute([up], message="m", api_factory=lambda: api)
    assert api.calls == []


def test_allow_pending_uploads_a_private_draft(namespace, outputs):
    """P08-R1-20: placeholders never go public, not even with --allow-pending."""
    api = HubApi()
    up = _staged("x", "[PENDING: x from y (z)]")
    hub.execute([up], message="m", api_factory=lambda: api, allow_pending=True, log=lambda _: None)
    assert api.sent() == ["create_repo", "update_repo_settings", "update_repo_settings", "upload_folder"]
    calls = [c for c in api.calls if c[0] not in ("repo_info", "repo_exists")]
    assert calls[0][2]["private"] is True
    assert calls[1][2] == {"repo_type": "dataset", "private": True}  # an existing private draft too
    assert all(c[2].get("private") is not False for c in api.calls)
    assert api.repos["test-org/lift-eval-logs"]["private"] is True


# --------------------------------------------------------------- P08-R1-11 the public-text scan


@pytest.mark.parametrize("gated", [None, "auto"])
def test_a_public_upload_quoting_an_output_is_refused(namespace, outputs, gated):
    """Whatever the output's label (here a refusal): decisions item 62."""
    text = "---\nextra_gated_prompt: terms\n---\n" if gated else ""
    up = _staged("model", text + "An example answer: " + HARMFUL[20:150] + "\n", gated=gated)
    api = FakeApi()
    with pytest.raises(PublishError, match="text of a model output"):
        hub.execute([up], message="m", api_factory=lambda: api)
    assert api.calls == []


def test_a_dry_run_runs_the_scan_too(namespace, outputs):
    """The content guard covers the dry run: it reports the file (never the text) and says it failed."""
    up = _staged("model", "An example answer: " + HARMFUL[20:150] + "\n")
    lines: list[str] = []
    assert hub.report([up], yes=False, log=lines.append) is False
    failed = [x for x in lines if x.startswith("FAILED the public-text scan")]
    assert len(failed) == 1 and "README.md" in failed[0] and "hubfake" not in "\n".join(lines)
    clean = _staged("clean", "# a card with no output in it\n")
    assert hub.report([clean], yes=False, log=lines.append) is True


def test_a_dry_run_without_outputs_says_the_scan_did_not_run(namespace):
    up = _staged("model", "# card\n")
    lines: list[str] = []
    assert hub.report([up], yes=False, log=lines.append) is True
    assert any("the public-text scan did not run" in x and "lift sync pull" in x for x in lines)


def test_an_output_text_in_any_staged_file_is_refused(namespace, outputs):
    up = _staged("logs")
    write_jsonl(up.folder / "aggregates" / "examples.jsonl", [{"note": HARMFUL.replace(" ", "\n")}])
    with pytest.raises(PublishError, match="examples.jsonl"):
        hub.execute([up], message="m", api_factory=FakeApi)


def test_an_unredacted_auditkit_record_is_refused(namespace, outputs):
    up = _staged("logs")
    record = {"metadata": {"lift": {"text": "full"}}, "samples": []}
    render.write_text(up.folder / "auditkit" / "qwen3guard" / "v" / "harmful_hi_dev.json", json.dumps(record))
    with pytest.raises(PublishError, match="AuditKIT record that is not redacted"):
        hub.execute([up], message="m", api_factory=FakeApi)


@pytest.mark.parametrize("gated", ["raw", "manual"])
def test_the_raw_completions_are_not_scanned(namespace, outputs, gated):
    """The raw completions hold the outputs by design: gated with automatic approval since Nov 14
    (decisions item 110), with manual approval before. Either way they are not scanned."""
    api = HubApi()
    up = _staged("raw", f"---\nextra_gated_prompt: terms\n---\n{HARMFUL}\n", gated=gated)
    hub.execute([up], message="m", api_factory=lambda: api, log=lambda _: None)
    assert "upload_folder" in api.names()
    assert api.repos["test-org/lift-raw-completions"]["gated"] == ("auto" if gated == "raw" else "manual")


def test_only_the_outputs_repo_may_hold_output_text_behind_an_automatic_gate(namespace, outputs):
    """An adapter repo gated with automatic approval is scanned as if public, and the raw repo made
    public is scanned too: only a gated outputs repo holds output text (decisions item 110)."""
    adapter = _staged("a", f"---\nextra_gated_prompt: terms\n---\n{HARMFUL}\n", gated="auto")
    with pytest.raises(PublishError, match="text of a model output"):
        hub.execute([adapter], message="m", api_factory=FakeApi, log=lambda _: None)
    public_raw = dataclasses.replace(_staged("r", f"# raw\n{HARMFUL}\n", gated="raw").repo, access="public")
    assert hub.scanned(Upload(repo=public_raw, folder=adapter.folder, title="r"))


def test_nothing_public_goes_up_from_a_machine_without_outputs(namespace):
    api = FakeApi()
    with pytest.raises(PublishError, match="public-text scan cannot run.*lift sync pull"):
        hub.execute([_staged("logs")], message="m", api_factory=lambda: api)
    assert api.calls == []


def test_namespace_must_look_like_an_hf_name(monkeypatch):
    monkeypatch.setenv("LIFT_HF_NAMESPACE", "not a name/with slash")
    with pytest.raises(PublishError):
        settings.namespace()


def test_cli_dry_runs_never_build_an_api(no_hub, capsys):
    for argv in (
        ["publish", "datasets"],
        ["publish", "model"],
        ["publish", "space"],
        ["publish", "collections"],
    ):
        assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "YOUR-HF-ORG/lift-eval-logs" in out
    assert "super_squash_history" in out  # how to retract a file HF history keeps


def test_cli_yes_refuses_without_namespace(no_hub, capsys):
    assert cli.main(["publish", "datasets", "--only", "prompts", "--yes"]) == 1
    assert "no HF namespace" in capsys.readouterr().err


def test_cli_yes_uploads_with_a_fake_api(namespace, monkeypatch, outputs):
    api = HubApi()
    monkeypatch.setattr(hub, "default_api", lambda: api)
    write_jsonl(
        paths.prompts_dir() / "harmful_hi.jsonl",
        [
            {
                "pid": "prh-test-0001",
                "lang": "hi",
                "set": "harmful",
                "split": "dev",
                "text": "t",
                "text_en": "e",
                "source": "polyrefuse",
                "translation": {"engine": "google-cloud-v2"},
            }
        ],
    )
    assert cli.main(["publish", "datasets", "--only", "prompts", "--yes"]) == 0
    upload = [c for c in api.calls if c[0] == "upload_folder"]
    assert len(upload) == 1 and upload[0][2]["repo_id"] == "test-org/lift-multilingual-safety-prompts"


def test_collections(namespace, monkeypatch):
    """The data collection lists the training sets too (published with the rows configs/publish.yaml
    withholds, the owner's decision of Nov 15). collections.json keeps each collection's slug and url and
    records the repo ids it was given, every one checked to be on the Hub first: the article names the
    training sets only once they are among the data collection's items."""
    write_inputs()
    plan = collect.plan()
    assert [r.repo_id for r in plan["data"]] == [
        "test-org/lift-eval-logs",
        "test-org/lift-raw-completions",
        "test-org/lift-multilingual-safety-prompts",
        "test-org/lift-hindi-sft",
    ]
    assert plan["models"][0].repo_id == "test-org/lift-wiseft-l0.50-adapter-aya-std-s1"
    api = FakeApi()
    out = collect.execute(plan, api_factory=lambda: api, log=lambda _: None)
    assert set(out) == {"models", "data"} and collect.read_links()["data"].startswith("https://")
    items = [c for c in api.calls if c[0] == "add_collection_item"]
    assert len(items) == len(plan["models"]) + len(plan["data"])
    assert all(c[2] == {"exists_ok": True} for c in items)
    names = api.names()
    checked = [c[1][0] for c in api.calls if c[0] == "repo_exists"]
    assert checked == [r.repo_id for r in plan["models"] + plan["data"]]  # every one, before any collection
    assert names.index("create_collection") > max(i for i, n in enumerate(names) if n == "repo_exists")
    recorded = json.loads(collect.links_path().read_text(encoding="utf-8"))
    for key in ("models", "data"):
        assert set(recorded[key]) == {"slug", "url", "items"}
        assert recorded[key]["items"] == [r.repo_id for r in plan[key]]
    assert collect.read_items() == {key: [r.repo_id for r in plan[key]] for key in ("models", "data")}


def test_a_collections_file_without_items_still_gives_its_links(namespace):
    """collections.json as it was written before the items were recorded (Nov 15): its links still read,
    and it records no item, so nothing reads as published on its account."""
    render.write_text(
        collect.links_path(),
        json.dumps({"data": {"slug": "o/d", "url": "https://hf.co/collections/o/d"}}, indent=1),
    )
    assert collect.read_links() == {"data": "https://hf.co/collections/o/d"}
    assert collect.read_items() == {}
    collect.links_path().unlink()
    assert collect.read_links() == {} and collect.read_items() == {}


def test_collections_refuse_while_the_training_sets_are_not_on_the_hub(namespace):
    """The data collection lists the training sets, so creating the collections waits for their upload,
    and nothing is created or recorded meanwhile."""
    write_inputs()
    api = FakeApi(missing={"test-org/lift-hindi-sft"})
    with pytest.raises(PublishError, match="not on the Hub yet.*dataset test-org/lift-hindi-sft"):
        collect.execute(collect.plan(), api_factory=lambda: api, log=lambda _: None)
    assert "create_collection" not in api.names() and not collect.links_path().exists()


def test_collections_refuse_without_a_winner(namespace):
    with pytest.raises(PublishError, match="frozen winner"):
        collect.execute(collect.plan(), api_factory=FakeApi)


def test_github_url_never_carries_credentials(monkeypatch):
    import subprocess

    class Done:
        stdout = "https://user:tok@github.com/o/r.git\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    assert settings.github_repo() == "https://github.com/o/r"


# ---------------------------------------------------------- P08-R2-12 the Hub state is read back


LOGS = "test-org/lift-eval-logs"


def test_a_draft_never_takes_over_a_live_public_repo(namespace, outputs):
    """--allow-pending over a repo that is already public would make it private and delete its files."""
    api = HubApi({LOGS: {"private": False, "gated": False}})
    up = _staged("x", "[PENDING: x from y (z)]")
    with pytest.raises(PublishError, match="already public"):
        hub.execute([up], message="m", api_factory=lambda: api, allow_pending=True, log=lambda _: None)
    assert api.sent() == []
    assert api.repos[LOGS] == {"private": False, "gated": False}


def test_a_draft_checks_every_repo_before_sending_anything(namespace, outputs):
    api = HubApi({LOGS: {"private": False, "gated": False}})
    first = _staged("raw", "---\nextra_gated_prompt: terms\n---\n# raw\n", gated="manual")
    draft = _staged("x", "[PENDING: x from y (z)]")
    with pytest.raises(PublishError, match="already public"):
        hub.execute([first, draft], message="m", api_factory=lambda: api, allow_pending=True)
    assert api.sent() == []


def test_a_public_repo_gated_earlier_is_ungated(namespace, outputs):
    api = HubApi({LOGS: {"private": False, "gated": "auto"}})
    hub.execute([_staged("logs")], message="m", api_factory=lambda: api, log=lambda _: None)
    assert api.repos[LOGS] == {"private": False, "gated": False}
    sent = api.sent()
    assert sent.index("update_repo_settings") < sent.index("upload_folder")


@pytest.mark.parametrize("ignored", ["gated", "private"])
def test_a_gate_the_hub_did_not_take_is_never_made_public(namespace, ignored):
    """The gate and the privacy are read back before any file goes up: a gated repo whose gate did not
    take (or that stayed public) gets no file and is never made public."""
    api = HubApi({"test-org/lift-raw-completions": {"private": False, "gated": False}}, ignored=(ignored,))
    up = _staged("raw", "---\nextra_gated_prompt: terms\n---\n# raw\n", gated="manual")
    with pytest.raises(PublishError, match="lift-raw-completions"):
        hub.execute([up], message="m", api_factory=lambda: api, log=lambda _: None)
    assert "upload_folder" not in api.names()
    assert all(c[2].get("private") is not False for c in api.calls if c[0] == "update_repo_settings")


def test_the_final_state_is_verified(namespace, outputs):
    """A repo that should be public but stayed private (the Hub ignored the setting) is reported."""
    api = HubApi({LOGS: {"private": True, "gated": False}}, ignored=("private",))
    with pytest.raises(PublishError, match="still private"):
        hub.execute([_staged("logs")], message="m", api_factory=lambda: api, log=lambda _: None)


def test_a_prompt_echoed_by_a_harmful_output_can_go_public(namespace):
    """P08-R2-03: a harmful output that restates its prompt does not block the public prompt set."""
    from lift.publish import datasets

    prompt = " ".join(f"echoedprompt{i:02d}" for i in range(9))
    output = "Sure. " + prompt + ": " + HARMFUL
    key, variant = sha256_text(output), "adapter-aya-std-s1"
    write_jsonl(
        paths.generations_dir() / variant / "harmful_hi_dev.jsonl",
        [{"key": key, "variant": variant, "pid": "p1", "output": output, "hit_cap": False}],
    )
    write_jsonl(
        paths.judgements_dir() / datasets.role_dir("primary") / variant / "harmful_hi_dev.jsonl",
        [{"key": key, "pid": "p1", "label": "refusal", "judge_label": "refusal", "output_sha": key}],
    )
    keys = paths.validation_dir() / "keys"
    keys.mkdir(parents=True)
    (keys / "round1.json").write_text(json.dumps({"r1hi-a": {"key": key, "sheet": "round1_hi"}}))
    write_jsonl(
        paths.validation_dir() / "labels" / "round1_hi.jsonl", [{"item_id": "r1hi-a", "label": "refusal"}]
    )
    record = {
        "pid": "p1",
        "lang": "hi",
        "set": "harmful",
        "split": "dev",
        "text": prompt,
        "text_en": "e",
        "source": "polyrefuse",
        "translation": {"engine": "google-cloud-v2"},
    }
    write_jsonl(paths.prompts_dir() / "harmful_hi.jsonl", [record])
    [up] = datasets.build(["prompts"])
    api = HubApi()
    hub.execute([up], message="m", api_factory=lambda: api, log=lambda _: None)
    assert "upload_folder" in api.names()
