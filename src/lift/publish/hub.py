"""Upload plans: what each ``lift publish`` command would send, and the one place that sends it.

Every command builds a list of ``Upload`` (one staged folder per repo), prints it, and calls
``execute`` only with ``--yes``. ``execute`` refuses while the namespace is the placeholder, while a
staged text file still holds a ``[PENDING`` placeholder (unless ``allow_pending``), or while a gated
repo's card lacks its terms. For a gated repo the order is: create it private, set the gating, upload,
and only then make it public, so its files are never public without the gate. The token comes from
huggingface_hub's own lookup (HF_TOKEN or the saved login) and is never printed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lift.publish import PublishError
from lift.publish.inputs import PENDING, shown
from lift.publish.settings import PLACEHOLDER_NAMESPACE, RepoSpec

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".txt", ".py", ".yaml", ".yml"}
GATE_FIELDS = ("extra_gated_prompt",)


@dataclass
class Upload:
    repo: RepoSpec
    folder: Path
    title: str
    notes: list[str] = field(default_factory=list)
    # Files that the plan expects but that are made by a later step (weights not yet materialised).
    expected: list[str] = field(default_factory=list)

    def files(self) -> list[Path]:
        if not self.folder.is_dir():
            return []
        return sorted(p for p in self.folder.rglob("*") if p.is_file())

    def describe(self) -> list[str]:
        files = self.files()
        size = sum(p.stat().st_size for p in files)
        gate = {False: "public", "auto": "GATED (automatic approval)", "manual": "GATED (manual approval)"}
        lines = [
            f"{self.title}: {self.repo.url}  [{self.repo.repo_type}, {gate[self.repo.gated]}]",
            f"  from {shown(self.folder)}: {len(files)} files, {size / 1e6:.1f} MB",
        ]
        for p in files[:12]:
            lines.append(f"    {p.relative_to(self.folder).as_posix()}")
        if len(files) > 12:
            lines.append(f"    ... and {len(files) - 12} more")
        lines += [f"  missing until built: {name}" for name in self.expected]
        lines += [f"  note: {n}" for n in self.notes]
        return lines


def pending_files(uploads: Sequence[Upload]) -> list[str]:
    out = []
    for u in uploads:
        for p in u.files():
            if p.suffix in TEXT_SUFFIXES and PENDING in p.read_text(encoding="utf-8", errors="replace"):
                out.append(shown(p))
    return out


def check_ready(uploads: Sequence[Upload], *, allow_pending: bool = False) -> None:
    problems = []
    for u in uploads:
        if u.repo.repo_id.startswith(PLACEHOLDER_NAMESPACE + "/"):
            problems.append(
                f"{u.title}: no HF namespace (set LIFT_HF_NAMESPACE or configs/publish.yaml namespace)"
            )
        if u.expected:
            problems.append(f"{u.title}: not built yet: {', '.join(u.expected)}")
        if not u.files():
            problems.append(f"{u.title}: nothing staged in {shown(u.folder)}")
        if u.repo.gated:
            card = u.folder / "README.md"
            text = card.read_text(encoding="utf-8") if card.is_file() else ""
            if not all(f in text for f in GATE_FIELDS):
                problems.append(f"{u.title}: a gated repo needs a README.md with {', '.join(GATE_FIELDS)}")
    if not allow_pending:
        problems += [f"{p}: still holds a [PENDING ...] placeholder" for p in pending_files(uploads)]
    if problems:
        raise PublishError("not uploading:\n  " + "\n  ".join(problems))


def default_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def execute(
    uploads: Sequence[Upload],
    *,
    message: str,
    allow_pending: bool = False,
    api_factory: Callable[[], Any] | None = None,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Upload every plan (checked first, all of them, before anything is sent). Returns the repo urls."""
    check_ready(uploads, allow_pending=allow_pending)
    api = (api_factory or default_api)()
    done = []
    for u in uploads:
        r = u.repo
        kwargs: dict[str, Any] = {"repo_type": r.repo_type, "exist_ok": True, "private": bool(r.gated)}
        if r.repo_type == "space":
            kwargs["space_sdk"] = r.sdk or "gradio"
        api.create_repo(r.repo_id, **kwargs)
        if r.gated:
            # Gate first, while the repo is private: its files are never public ungated.
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, private=True)
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, gated=r.gated)
        api.upload_folder(
            repo_id=r.repo_id, repo_type=r.repo_type, folder_path=str(u.folder), commit_message=message
        )
        if r.gated:
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, private=False)
        log(f"uploaded {u.title}: {r.url}")
        done.append(r.url)
    return done


def report(uploads: Sequence[Upload], *, yes: bool, log: Callable[[str], None] = print) -> None:
    for u in uploads:
        for line in u.describe():
            log(line)
    pend = pending_files(uploads)
    if pend:
        log(f"{len(pend)} staged file(s) still hold [PENDING ...] placeholders (inputs not made yet):")
        for p in pend:
            log(f"  {p}")
    if not yes:
        log("dry run: nothing was uploaded. Add --yes to upload.")
