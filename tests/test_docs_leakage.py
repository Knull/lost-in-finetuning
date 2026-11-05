"""What the publication plan lets out (CLAUDE.md rule 6, docs/decisions.md item 14), read from the two
documents P08 builds from: nothing publishes yet. The checks read tracked files only and print no
prompt or output text."""

from __future__ import annotations

import re

from lift.paths import REPO_ROOT

HF_README = REPO_ROOT / "hf" / "README.md"
P08 = REPO_ROOT / "prompts" / "P08_publish.md"

# A phrase that puts per-output model text in an artifact: "refusal texts", "output text", "completions",
# "raw ...", "model outputs", "responses". Labels, keys, hashes and aggregates (tables, rates) do not match.
OUTPUT_TEXT = re.compile(r"\b(?:refusal|output|response|completion|answer|generation|model)s?\s+texts?\b"
                         r"|\bcompletions?\b|\braw\b|\b(?:model|judged)\s+outputs?\b|\bresponses\b", re.I)
NEGATED = re.compile(r"\bno\s+(?:other\s+)?$", re.I)  # "no output text", "no other output text"
ANY_LABEL = re.compile(r"\bwhatever (?:its|their) label\b", re.I)


def flat(text: str) -> str:
    return " ".join(text.split())


def promised_output_text(contents: str) -> list[str]:
    """The phrases of ``contents`` that put per-output text in the artifact (a negated one says the
    opposite)."""
    return [m.group(0) for m in OUTPUT_TEXT.finditer(contents) if not NEGATED.search(contents[:m.start()])]


def artifact_rows() -> dict[str, tuple[str, str]]:
    """hf/README.md's artifact table: {artifact: (access, contents)}."""
    rows = {}
    for line in HF_README.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 3 and cells[0] != "Artifact" and set(cells[0]) - set("- "):
            rows[cells[0]] = (cells[1], cells[2])
    return rows


def p08_items(number: int) -> list[str]:
    """The bullet items of step ``number`` of P08's Build list, each on one line."""
    build = P08.read_text(encoding="utf-8").split("## Build", 1)[1]
    start = re.search(rf"^{number}\. ", build, re.M)
    assert start, f"P08 has no step {number}"
    end = re.search(rf"^{number + 1}\. ", build, re.M)
    step = build[start.end():end.start() if end else None]
    return [flat(item) for item in re.split(r"\n\s*- ", step)[1:]]


def public_contents(item: str) -> str | None:
    """What a "(public)" item says the artifact holds: its first sentence after "(public)". None for an
    item that is not public (the sentences after the first are instructions, not contents)."""
    m = re.match(r"[^(]*\(public[^)]*\):?\s*(.*?)(?:\.\s|\.$|$)", item)
    return m.group(1) if m else None


# R1-14: the public logs dataset listed "refusal texts". A refusal label is Qwen3Guard's alone for about
# 90% of outputs, so a harmful completion it called a refusal would have been published. No public
# artifact holds output text; every judged output goes to the gated, manual-approval repo.
def test_hf_readme_public_artifacts_hold_no_output_text():
    rows = artifact_rows()
    assert {"Logs dataset", "Raw completions", "Space"} <= set(rows), "hf/README.md's table was not parsed"
    leaks = {name: found for name, (access, contents) in rows.items()
             if "public" in access and "gated" not in access
             for found in [promised_output_text(contents)] if found}
    assert leaks == {}, f"hf/README.md puts per-output text in a public artifact: {leaks}"
    logs = rows["Logs dataset"][1]
    assert "no output text" in logs.lower() and "output hashes" in logs
    access, contents = rows["Raw completions"]
    assert "gated" in access and "manual approval" in access
    assert "every judged output" in contents.lower() and "refusals" in contents
    space = rows["Space"][1]  # examples only as a cut of about 10 words, whatever their label
    assert "about 10 words" in space and "[redacted]" in space and ANY_LABEL.search(space)


def test_p08_publishes_no_output_text_and_guards_public_files_by_content():
    items = p08_items(2)  # `lift publish datasets`
    public = {item.split(" (", 1)[0]: contents for item in items
              for contents in [public_contents(item)] if contents is not None}
    assert any("logs dataset" in name.lower() for name in public), "P08's logs dataset item was not parsed"
    leaks = {name: found for name, contents in public.items() for found in [promised_output_text(contents)]
             if found}
    assert leaks == {}, f"P08 tells `lift publish datasets` to put per-output text in public: {leaks}"
    logs = next(contents for name, contents in public.items() if "logs dataset" in name.lower())
    assert "no output text" in logs.lower() and "output hashes" in logs
    gated = [item for item in items if "(gated, manual approval)" in item]
    assert len(gated) == 1 and "every judged output" in gated[0].lower() and "refusals" in gated[0]
    # The guard reads the content, never a label (most labels are one judge's call).
    assert not any("harmful label" in item for item in items)
    guards = [item for item in items if "lift publish preflight" in item and ANY_LABEL.search(item)]
    assert len(guards) == 1 and "output text" in guards[0] and "substring" in guards[0]


def test_p08_space_shows_output_text_only_cut_to_about_10_words():
    step = " ".join(p08_items(6))  # the Space
    assert "refusals only" not in step  # a refusal is shown no fuller than any other output
    assert "about 10 words" in step and "[redacted]" in step and ANY_LABEL.search(step)
