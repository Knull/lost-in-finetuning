"""Upload plans: nothing is sent without --yes, gating is set before any file goes up, and --yes refuses
placeholders, a missing namespace or an unbuilt artifact. huggingface_hub is never reached."""

from __future__ import annotations

import pytest
from publish_fakes import FakeApi, write_inputs

from lift import cli, paths
from lift.io import write_jsonl
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


def _staged(name: str, text: str = "# card\n", gated: str | None = None) -> Upload:
    kind = {None: "logs", "manual": "raw", "auto": "adapter"}[gated]
    repo = settings.repo(kind, adapter="adapter-aya-std-s1")
    folder = paths.output_dir() / "publish" / name
    render.write_text(folder / "README.md", text)
    return Upload(repo=repo, folder=folder, title=name)


def test_gated_repo_is_gated_before_upload_and_public_after(namespace):
    api = FakeApi()
    up = _staged("raw", "---\nextra_gated_prompt: terms\n---\n# raw\n", gated="manual")
    hub.execute([up], message="m", api_factory=lambda: api, log=lambda _: None)
    assert api.names() == [
        "create_repo",
        "update_repo_settings",
        "update_repo_settings",
        "upload_folder",
        "update_repo_settings",
    ]
    create, private, gate, upload, public = api.calls
    assert create[2]["private"] is True and create[2]["repo_type"] == "dataset"
    assert private[2] == {"repo_type": "dataset", "private": True}
    assert gate[2] == {"repo_type": "dataset", "gated": "manual"}
    assert upload[2]["repo_id"] == "test-org/lift-raw-completions"
    assert public[2] == {"repo_type": "dataset", "private": False}


def test_public_repo_is_created_public_and_uploaded(namespace):
    api = FakeApi()
    hub.execute([_staged("logs")], message="m", api_factory=lambda: api, log=lambda _: None)
    assert api.names() == ["create_repo", "upload_folder"]
    assert api.calls[0][2]["private"] is False


@pytest.mark.parametrize("problem", ["namespace", "pending", "gate", "expected"])
def test_check_ready_refuses(problem, monkeypatch):
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
    if problem == "pending":
        hub.execute([up], message="m", api_factory=lambda: api, allow_pending=True, log=lambda _: None)
        assert "upload_folder" in api.names()


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


def test_cli_yes_refuses_without_namespace(no_hub, capsys):
    assert cli.main(["publish", "datasets", "--only", "prompts", "--yes"]) == 1
    assert "no HF namespace" in capsys.readouterr().err


def test_cli_yes_uploads_with_a_fake_api(namespace, monkeypatch):
    api = FakeApi()
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
    write_inputs()
    plan = collect.plan()
    assert [r.repo_id for r in plan["data"]] == [
        "test-org/lift-eval-logs",
        "test-org/lift-raw-completions",
        "test-org/lift-multilingual-safety-prompts",
    ]
    assert plan["models"][0].repo_id == "test-org/lift-wiseft-l0.50-adapter-aya-std-s1"
    api = FakeApi()
    out = collect.execute(plan, api_factory=lambda: api, log=lambda _: None)
    assert set(out) == {"models", "data"} and collect.read_links()["data"].startswith("https://")
    items = [c for c in api.calls if c[0] == "add_collection_item"]
    assert len(items) == len(plan["models"]) + len(plan["data"])
    assert all(c[2] == {"exists_ok": True} for c in items)


def test_collections_refuse_without_a_winner(namespace):
    with pytest.raises(PublishError, match="frozen winner"):
        collect.execute(collect.plan(), api_factory=FakeApi)


def test_github_url_never_carries_credentials(monkeypatch):
    import subprocess

    class Done:
        stdout = "https://user:tok@github.com/o/r.git\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    assert settings.github_repo() == "https://github.com/o/r"
