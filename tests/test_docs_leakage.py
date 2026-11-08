"""What the publication plan lets out (CLAUDE.md rule 6, docs/decisions.md item 14), read from the two
documents P08 builds from: nothing publishes yet. The checks read tracked files only and print no
prompt or output text."""

from __future__ import annotations

import re

from lift.paths import REPO_ROOT

HF_README = REPO_ROOT / "hf" / "README.md"
P08 = REPO_ROOT / "prompts" / "P08_publish.md"

# A phrase that puts per-output model text in an artifact: "refusal texts", "output text", "completions",
# "raw ...", "model outputs", "responses". Labels, keys, hashes and aggregates (tables, rates) do not match,
# nor does a record field (JudgementRecord.raw): a name, which JUDGE_TEXT checks.
OUTPUT_TEXT = re.compile(r"\b(?:refusal|output|response|completion|answer|generation|model)s?\s+texts?\b"
                         r"|\bcompletions?\b|(?<!\.)\braw\b|\b(?:model|judged)\s+outputs?\b"
                         r"|\bresponses\b", re.I)
NEGATED = re.compile(r"\bno\s+(?:other\s+)?$", re.I)  # "no output text", "no other output text"
ANY_LABEL = re.compile(r"\bwhatever (?:its|their) label\b", re.I)
# The judges' own text, judge logs that CLAUDE.md rule 6 gates like the outputs: a judgement's raw reply
# (Qwen3Guard's, or a frontier judge's JSON), a frontier judge's reason, and a judge failure's text, which
# quotes the judge (a frontier refusal: up to 200 characters of the provider's text; a Qwen3Guard parse
# error: up to 40 of the guard's reply).
JUDGE_TEXT = ("JudgementRecord.raw", "native.reason", "native.judge_failure")
KEPT_OUT = re.compile(r"\bstays?\s+out\b", re.I)
# Of a judge failure, only that the judge failed and the failure's kind go in.
FAILURE_KIND = re.compile(r"\bonly that (?:its|the) judge failed\b[^.]*\bkind\b", re.I)


def flat(text: str) -> str:
    return " ".join(text.split())


def promised_output_text(contents: str) -> list[str]:
    """The phrases of ``contents`` that put per-output text in the artifact (a negated one says the
    opposite)."""
    return [m.group(0) for m in OUTPUT_TEXT.finditer(contents) if not NEGATED.search(contents[:m.start()])]


def keeps_judge_text_out(contents: str) -> bool:
    """Whether one sentence of ``contents`` names every JUDGE_TEXT field and keeps them out."""
    return any(all(name in sentence for name in JUDGE_TEXT) and KEPT_OUT.search(sentence)
               for sentence in re.split(r"(?<=\.)\s+", flat(contents)))


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
    # R1-14 follow-up: nor the judges' text, which the judgements files hold beside each label, a judge
    # failure's included (W3-08c): a failed judgement keeps only that its judge failed, and how.
    assert keeps_judge_text_out(logs), "hf/README.md's logs dataset does not keep JUDGE_TEXT out"
    assert FAILURE_KIND.search(flat(logs)), "hf/README.md's logs dataset does not keep a failure's kind alone"
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
    logs_item = next(item for item in items if "logs dataset" in item.split(" (", 1)[0].lower())
    assert keeps_judge_text_out(logs_item), "P08's logs dataset does not keep JUDGE_TEXT out"
    assert FAILURE_KIND.search(logs_item), "P08's logs dataset does not keep a failure's kind alone"
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


# R2-11: a record outlives a change that makes its cell unreportable (a fixed prompt text, other labelling
# rules, a re-registered variant, a re-pinned Global, a harness bump) until that cell's `lift restart`
# retires it, and lift.audit.load_record serves any record that is not retired. So the logs dataset takes
# a record only while the reporting door accepts its cell as it stands, and only the record of that read.
def test_p08_publishes_an_auditkit_record_only_while_its_cell_reads_as_it_stands():
    import inspect

    from lift import audit
    from lift.judge import pipeline

    logs_item = next(item for item in p08_items(2) if "logs dataset" in item.split(" (", 1)[0].lower())
    rule = [s for s in re.split(r"(?<=\.)\s+", logs_item) if "AuditKIT record" in s and "load_records" in s]
    assert rule, "P08's logs dataset publishes every AuditKIT record it finds, those of stale cells included"
    (sentence,) = rule
    assert "`lift.judge.pipeline.load_records`" in sentence and "as it stands" in sentence, sentence
    assert re.search(r"\bonly\b", sentence) and re.search(r"\bdropped\b", sentence), sentence
    # The read must give the record's own hashes, the ones record_safety writes into metadata.lift.
    source = inspect.getsource(audit.record_safety)
    for field in ("labels_sha256", "keys_sha256", "judge_id"):
        assert f"`{field}`" in sentence and f'"{field}"' in source, field
    assert {"judge", "complete", "dtype"} <= set(inspect.signature(pipeline.load_records).parameters)
