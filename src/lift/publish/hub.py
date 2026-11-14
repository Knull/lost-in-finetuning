"""Upload plans: what each ``lift publish`` command would send, and the one place that sends it.

Every command builds a list of ``Upload`` (one staged folder per repo), prints it, and calls
``execute`` only with ``--yes``. ``execute`` refuses (``check_ready``, every plan checked before anything
is sent) while the namespace is the placeholder, while something is not built, while the plan is blocked
(it names the repaired model or its numbers and the recipe is not frozen at a ``recipe-frozen-*`` tag:
``lift.publish.inputs.recipe_block``; ``allow_pending`` does not lift it), while a gated repo's card
lacks its terms, while a staged text still holds a ``[PENDING`` placeholder (unless ``allow_pending``),
and while any file of an upload other than a gated outputs repo (``OUTPUT_KINDS``) fails the public-text scan
(``lift.publish.preflight.scan_folder``: secrets, raw completion records, notebooks with outputs, any run
of any model output on this machine, whatever its label, unredacted AuditKIT records). That scan needs
this machine's outputs: without them (``lift sync pull``) nothing public is uploaded.

For a gated repo the order is: create it private, set the gating, read the gating and privacy back
from the Hub, upload, and only then make it public, so its files are never public without the gate. A
public repo is sent ``gated=False`` (one gated earlier is ungated). With ``allow_pending``, an upload that
still holds a placeholder goes up as a PRIVATE draft and stays private; a draft never replaces a repo
that is already public (that would make the live repo private and delete its files), and every draft is
checked before anything is sent. After each upload the final state is read back again
(``repo_info``): what the Hub reports, not what was asked, decides. Each upload mirrors its staged folder
(``delete_patterns="*"``: a file no longer staged is deleted at HEAD). The token comes from
huggingface_hub's own lookup (HF_TOKEN or the saved login) and is never printed.

HF keeps every earlier commit of a repo, so a file once uploaded stays readable in its history even after
a later upload deletes it: publish the logs only after every label is final, and see ``RETRACTION``.
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
RETRACTION = (
    "HF keeps every past commit: to retract a file already uploaded (say a label a reader later corrects), "
    "upload again, then squash the history with HfApi().super_squash_history(repo_id, repo_type=...) or "
    "delete and recreate the repo"
)


@dataclass
class Upload:
    repo: RepoSpec
    folder: Path
    title: str
    notes: list[str] = field(default_factory=list)
    # Files that the plan expects but that are made by a later step (weights not yet materialised).
    expected: list[str] = field(default_factory=list)
    # Why it may not be uploaded at all yet (the recipe is not frozen at its tag); --yes refuses it.
    blocked: list[str] = field(default_factory=list)

    def files(self) -> list[Path]:
        if not self.folder.is_dir():
            return []
        return sorted(p for p in self.folder.rglob("*") if p.is_file())

    def describe(self) -> list[str]:
        files = self.files()
        size = sum(p.stat().st_size for p in files)
        gate = {False: "public", "auto": "GATED (automatic approval)", "manual": "GATED (manual approval)"}
        lines = [
            f"{self.title}: {self.repo.url}  [{self.repo.repo_type}, {gate[self.repo.gated]}]"
            + ("  BLOCKED" if self.blocked else ""),
            f"  from {shown(self.folder)}: {len(files)} files, {size / 1e6:.1f} MB",
        ]
        for p in files[:12]:
            lines.append(f"    {p.relative_to(self.folder).as_posix()}")
        if len(files) > 12:
            lines.append(f"    ... and {len(files) - 12} more")
        lines += [f"  BLOCKED: {why}" for why in self.blocked]
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


OUTPUT_KINDS = frozenset({"raw"})  # the repos that hold model outputs by design


def scanned(u: Upload) -> bool:
    """Whether an upload goes through the public-text scan: everything but a gated repo that holds the
    outputs by design (``OUTPUT_KINDS``: the raw completions, harmful ones included; automatic approval
    since Nov 14, docs/decisions.md 110) and any repo gated with manual approval. Every other repo with
    automatic approval is scanned as if public, and an outputs repo made public is scanned, so it fails."""
    if u.repo.gated == "manual":
        return False
    return not (u.repo.kind in OUTPUT_KINDS and u.repo.gated)


def check_ready(
    uploads: Sequence[Upload], *, allow_pending: bool = False, windows: set[str] | None = None
) -> None:
    """Refuse (PublishError) unless every plan may be uploaded. ``windows`` None loads the windows of
    every model output on this machine (``preflight.load_windows``) when a plan needs the public-text
    scan."""

    problems = []
    for u in uploads:
        if u.repo.repo_id.startswith(PLACEHOLDER_NAMESPACE + "/"):
            problems.append(
                f"{u.title}: no HF namespace (set LIFT_HF_NAMESPACE or configs/publish.yaml namespace)"
            )
        if u.blocked:
            problems.append(f"{u.title}: blocked: {'; '.join(u.blocked)}")
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
    found, why = scan_public(uploads, windows)
    if why:
        problems.append(f"the public-text scan cannot run: {why}")
    problems += found
    if problems:
        raise PublishError("not uploading:\n  " + "\n  ".join(problems))


def scan_public(uploads: Sequence[Upload], windows: set[str] | None = None) -> tuple[list[str], str | None]:
    """(what the public-text scan finds in every upload that goes through it, why it could not run).
    ``windows`` None loads the windows of every model output on this machine (``preflight.load_windows``),
    which refuses a machine that does not hold them: the second value then says why, and nothing was
    scanned."""
    from lift.publish import preflight

    public = [u for u in uploads if scanned(u) and u.files()]
    if not public:
        return [], None
    if windows is None:
        try:
            windows = preflight.load_windows()[0]
        except PublishError as exc:
            return [], str(exc)
    return [f"{u.title}: {x}" for u in public for x in preflight.scan_folder(u.folder, windows)], None


def default_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def _state(api: Any, r: RepoSpec) -> tuple[object, object]:
    """(private, gated) as the Hub reports them (``repo_info``; gated None read as False)."""
    info = api.repo_info(r.repo_id, repo_type=r.repo_type)
    missing = object()
    private, gated = getattr(info, "private", missing), getattr(info, "gated", missing)
    if private is missing or gated is missing:
        raise PublishError(
            f"{r.url}: the Hub did not report its visibility and gating; nothing more was sent"
        )
    return private, (gated or False)


def _expect(api: Any, r: RepoSpec, *, gated: object, private: bool | None, when: str) -> None:
    """Refuse (PublishError) unless the Hub reports ``gated`` (and ``private`` unless None)."""
    is_private, is_gated = _state(api, r)
    problems = []
    if is_gated != gated:
        problems.append(f"gated is {is_gated!r}, not {gated!r}")
    if private is not None and is_private is not private:
        problems.append("it is still private" if is_private else "it is public")
    if problems:
        raise PublishError(f"{r.url} {when}: {'; '.join(problems)} (read back from the Hub); stopped there")


def execute(
    uploads: Sequence[Upload],
    *,
    message: str,
    allow_pending: bool = False,
    api_factory: Callable[[], Any] | None = None,
    log: Callable[[str], None] = print,
    windows: set[str] | None = None,
) -> list[str]:
    """Upload every plan (checked first, all of them, before anything is sent). Returns the repo urls."""
    check_ready(uploads, allow_pending=allow_pending, windows=windows)
    drafts = {id(u) for u in uploads if pending_files([u])}  # only reachable with allow_pending
    api = (api_factory or default_api)()
    live = []
    for u in uploads:
        r = u.repo
        if id(u) in drafts and api.repo_exists(r.repo_id, repo_type=r.repo_type):
            if _state(api, r)[0] is not True:
                live.append(r.url)
    if live:
        raise PublishError(
            "not uploading: a draft (it still holds placeholders) would make a repo that is already public "
            "private and delete its files: " + ", ".join(live) + ". Fill the placeholders and upload without "
            "--allow-pending"
        )
    done = []
    for u in uploads:
        r = u.repo
        draft = id(u) in drafts
        private = bool(r.gated) or draft
        kwargs: dict[str, Any] = {"repo_type": r.repo_type, "exist_ok": True, "private": private}
        if r.repo_type == "space":
            kwargs["space_sdk"] = r.sdk or "gradio"
        api.create_repo(r.repo_id, **kwargs)
        if private:
            # Private first (the repo may already exist, public): a gated repo's files are never public
            # ungated, and a draft's placeholders never public at all.
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, private=True)
        if r.gated:
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, gated=r.gated)
        elif r.repo_type != "space":  # a Space has no access gate
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, gated=False)
        _expect(api, r, gated=r.gated or False, private=True if private else None, when="before the upload")
        api.upload_folder(
            repo_id=r.repo_id,
            repo_type=r.repo_type,
            folder_path=str(u.folder),
            commit_message=message,
            delete_patterns="*",  # mirror the staged folder: withdrawn or renamed files leave HEAD
        )
        if draft:
            _expect(api, r, gated=r.gated or False, private=True, when="after the upload")
            log(f"uploaded {u.title} as a PRIVATE draft (it still holds placeholders): {r.url}")
        else:
            api.update_repo_settings(r.repo_id, repo_type=r.repo_type, private=False)
            try:
                _expect(api, r, gated=r.gated or False, private=False, when="after the upload")
            except PublishError:
                if r.gated:  # public without its gate: make it private again before saying so
                    api.update_repo_settings(r.repo_id, repo_type=r.repo_type, private=True)
                raise
            log(f"uploaded {u.title}: {r.url}")
        done.append(r.url)
    return done


def report(
    uploads: Sequence[Upload],
    *,
    yes: bool,
    log: Callable[[str], None] = print,
    windows: set[str] | None = None,
) -> bool:
    """Print the plan. A dry run also runs the public-text scan (the content guard of decisions item 62
    covers the dry run too) and returns False when a file that would be public fails it; a machine without
    the outputs to check against gets a note instead (``--yes`` refuses there)."""
    for u in uploads:
        for line in u.describe():
            log(line)
    pend = pending_files(uploads)
    if pend:
        log(f"{len(pend)} staged file(s) still hold [PENDING ...] placeholders (inputs not made yet):")
        for p in pend:
            log(f"  {p}")
        log("  (--yes refuses them; --yes --allow-pending uploads those repos as PRIVATE drafts)")
    log(f"note: {RETRACTION}.")
    if yes:
        return True
    found, why = scan_public(uploads, windows)
    if why:
        log(f"note: the public-text scan did not run, and --yes refuses until it can: {why}")
    for x in found:
        log(f"FAILED the public-text scan (matched text is never shown): {x}")
    log("dry run: nothing was uploaded. Add --yes to upload.")
    return not found
