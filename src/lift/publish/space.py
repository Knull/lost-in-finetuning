"""``lift publish space``: stage hf/space/ (Gradio, no model loading) with the logs dataset it reads and a
few cut examples.

The Space reads only the public aggregate logs dataset (``space_config.json`` names it), plus
``examples.json``: a few outputs per variant, language and set of the final split, each cut when the
Space is built to at most MAX_WORDS words and under WINDOW characters and marked ``[redacted]``, whatever
its label, refusals included (decisions item 62: no output is shown because of its label). No other
output text goes in. The staged folder passes the public-text scan like every public upload: a cut is
shorter than the scan's window, so it holds no run the scan looks for, and anything longer fails it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from lift import paths
from lift.io import read_jsonl, sha256_text
from lift.publish import preflight, render, settings
from lift.publish.hub import Upload

FILES = ("app.py", "README.md", "requirements.txt")
EXAMPLES = "examples.json"
EXAMPLES_PER_CELL = 2
SPLIT = "final"
MAX_WORDS = 10
# A cut and the space before its mark stay under the scan's window: a 59-character cut ending on a word,
# then that space, was the output's first window (the word break after it) and failed the scan (Nov 14).
MAX_CHARS = preflight.WINDOW - 2
MARK = "[redacted]"
# Scripts written without spaces between words (Thai, Lao, Burmese, Khmer, kana, Han): two characters
# count as one word there, or "ten words" of Chinese would be a whole answer.
_UNSPACED = re.compile("[฀-໿က-႟ក-៿぀-ヿ㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f]")


def source_dir() -> Path:
    return paths.REPO_ROOT / "hf" / "space"


def cut(text: object) -> str:
    """The start of ``text``, then ``[redacted]``: at most MAX_WORDS words (two characters count as a word
    in a script written without spaces) and at most MAX_CHARS characters, ending on a whole word unless the
    first word alone is longer. ``hf/space/app.py`` holds a copy (the Space imports nothing from the
    project); tests/test_publish_space_notebook.py checks they agree."""
    kept: list[str] = []
    budget, room = MAX_WORDS, MAX_CHARS
    for word in str(text).removesuffix(MARK).split():
        if budget <= 0:
            break
        cost = max(1, -(-len(_UNSPACED.findall(word)) // 2))
        if cost > budget:
            word, cost = word[: 2 * budget], budget
        if len(word) > room:  # no room left for the whole word: only a first word is cut inside
            if not kept:
                kept.append(word[:room])
            break
        kept.append(word)
        budget -= cost
        room -= len(word) + 1
    return f"{' '.join(kept)} {MARK}".strip()


def examples() -> list[dict]:
    """A few cut outputs per study variant, language and safety set of the final split: the first
    EXAMPLES_PER_CELL pids of each cell, with the primary judge's final label when it covers the output
    now under the key. Each ``text`` is ``cut``, whatever the label."""
    from lift.publish.datasets import SAFETY_SETS, _cells, primary_judgements, shown_label
    from lift.validation.outputs import variant_info
    from lift.validation.second_judge import parse_generations_name

    judged = primary_judgements()
    out: list[dict] = []
    for variant, stem, f in _cells(paths.generations_dir()):
        parsed = parse_generations_name(stem)
        info = variant_info(variant)
        if parsed is None or info is None or parsed[0] not in SAFETY_SETS or parsed[2] != SPLIT:
            continue
        rows = sorted(
            (r for r in read_jsonl(f) if isinstance(r.get("output"), str) and r["output"].strip()),
            key=lambda r: str(r.get("pid")),
        )
        # the first rows whose cut holds no window of their own output, as the scan reads it (a text whose
        # characters grow when normalised, an ellipsis say, can still reach one): the scan would refuse it
        rows = [
            r for r in rows if not preflight.has_window(cut(r["output"]), preflight.windows_of([r["output"]]))
        ]
        for r in rows[:EXAMPLES_PER_CELL]:
            rec = judged.get(r.get("key")) or {}
            current = rec.get("output_sha") == sha256_text(r["output"])
            out.append(
                {
                    "variant": variant,
                    "method": info.method,
                    "set": parsed[0],
                    "lang": parsed[1],
                    "pid": r.get("pid"),
                    "label": shown_label(rec.get("label")) if current else None,
                    "text": cut(r["output"]),
                }
            )
    return out


def build() -> Upload:
    folder = paths.output_dir() / "publish" / "space"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(source_dir() / name, folder / name)
    logs = settings.repo("logs")
    render.write_text(folder / "space_config.json", json.dumps({"logs_dataset": logs.repo_id}, indent=1))
    rows = examples()
    render.write_text(folder / EXAMPLES, json.dumps(rows, indent=1, ensure_ascii=False))
    return Upload(
        repo=settings.repo("space"),
        folder=folder,
        title="Space (Gradio, no model loading)",
        notes=[
            f"reads only {logs.url}",
            f"{len(rows)} examples of the {SPLIT} split, each cut to at most {MAX_WORDS} words and "
            f"{MAX_CHARS} characters plus {MARK}, whatever its label",
            *preflight.relativised_note(preflight.relativise_folder(folder)),
        ],
    )
