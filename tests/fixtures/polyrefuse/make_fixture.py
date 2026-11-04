"""Writes the miniature PolyRefuse tree next to this file (run it as a script from the repo root).

Placeholder text only, laid out like the pinned upstream commit (see lift.data.polyrefuse) and written
the same way: ``json.dump(rows, f, indent=4)`` (ASCII ``\\u`` escapes), LF, no trailing newline.
Rows per file, matching the tests' shrunken ``sources.polyrefuse.counts``: harmful test 6, harmless
test 4, harmful train 3, harmful val 2, harmless train 3, harmless val 2.

Quirks, as upstream: harmful test rows 1 and 4 repeat one English prompt under different categories;
harmful train row 1 is harmful test row 2's English; harmful test row 3 and harmless test row 2 have
an inner newline, harmful test row 5 a double space; es harmful test row 0 carries a U+200B; no zh
harmless test; zh harmless train/val are another draw (lengths uncorrelated with the English); de is
present (the project ignores it); dataset/splits/harmless_test_500_sampled.json is a decoy with other
prompts; PolyRefuse/harmful_test_back_translated/ exists (literal UTF-8, ignored). Translations are
a fixed multiple of their English line lengths, so the log-length check passes on these files and
fails on a shuffled copy. A test checks that the committed tree is exactly what this script writes;
delete the tree before rerunning if a file is dropped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSLATED = ("es", "zh", "ar", "th", "de")

# (English, category). Categories mimic upstream's three source blocks (JBB, HarmBench, StrongREJECT).
HARMFUL_TEST = [
    ("Placeholder request 0 about garden tools.", "Garden care"),
    ("Placeholder request 1 about baking bread at home: list the steps, the oven settings and how long "
     "the dough should rest.", "Home baking"),
    ("Placeholder request 2 about painting a wooden fence.", "outdoor_hobbies"),
    ("Placeholder request 3 about birdwatching near a lake.\nAnswer in two short paragraphs and mention "
     "binoculars, field guides, notebooks and the best seasons for spotting migrating birds.",
     "outdoor_hobbies"),
    ("Placeholder request 1 about baking bread at home: list the steps, the oven settings and how long "
     "the dough should rest.", "Cooking and baking"),
    ("Placeholder request 5  on tea.", "Hobbies and crafts"),
]
HARMLESS_TEST = [
    "Placeholder question 0: name three colours.",
    "Placeholder question 1: write a short poem about autumn leaves falling in a quiet park.",
    "Placeholder question 2:\nsummarise a recipe for vegetable soup in a few lines, mentioning carrots, "
    "onions, celery, potatoes and fresh herbs, and suggest a bread to serve with it.",
    "Placeholder question 3 on maps.",
]
DECOY_HARMLESS_TEST = [
    "Decoy placeholder question 0 that the test files never use.",
    "Decoy placeholder question 1 about folding paper boats.",
    "Decoy placeholder question 2 about tidying a desk drawer.",
    "Decoy placeholder question 3 about naming a goldfish.",
]
HARMFUL_TRAIN = [
    "Train placeholder 0: kites.",
    HARMFUL_TEST[2][0],
    "Train placeholder 2 about planning a community garden: describe the layout, which vegetables to "
    "plant in spring, how to share the watering rota and how to keep records of the harvest.",
]
HARMFUL_VAL = [
    ("Validation placeholder 0 about sorting buttons.", "sorting_hobbies"),
    ("Validation placeholder 1 about organising a library shelf by colour, size and author, with a note "
     "on keeping a small catalogue up to date.", "home_organisation"),
]
HARMLESS_TRAIN = [
    "Benign train placeholder 0.",
    "Benign train placeholder 1: describe how a bicycle gear works in simple words for a child.",
    "Benign train placeholder 2: write a friendly note to a neighbour about a shared garden, mention the "
    "tomatoes, the new bench, the watering rota and an invitation to a weekend picnic.",
]
HARMLESS_VAL = [
    "Benign val placeholder 0 about clouds.",
    "Benign val placeholder 1: explain why the sky looks blue at noon and orange at sunset, in three "
    "sentences.",
]
# zh harmless train/val: another draw, unrelated to the English rows (short where they are long).
ZH_OTHER_TRAIN = ["另一批样本甲关于天气和季节的较长问题", "另批乙", "另一批样本丙问答"]
ZH_OTHER_VAL = ["另一批验证样本甲关于星星和月亮", "乙"]

# Placeholder words in each target script (tests' FakeLID reads the script) and a length ratio near
# the real files' medians.
_HEAD = {"es": "Petición ficticia", "zh": "占位请求", "ar": "طلب نموذجي", "th": "คำขอตัวอย่าง",
         "de": "Platzhalteranfrage"}
_FILL = {"es": " sobre jardinería pequeña, canciones y señales", "zh": "关于园艺和烹饪的示例文本",
         "ar": " عن البستنة والطبخ والقراءة", "th": "เกี่ยวกับการทำสวนและการทำอาหาร",
         "de": " über Gartenarbeit und Kochen"}
_RATIO = {"es": 1.15, "zh": 0.3, "ar": 0.8, "th": 0.9, "de": 1.25}


def translate(text: str, lang: str) -> str:
    """A placeholder 'translation': per line, the target script at a fixed multiple of the line's
    length, tagged with digits from the line's hash (same English, same translation)."""
    return "\n".join(_line(line, lang) for line in text.split("\n"))


def _line(line: str, lang: str) -> str:
    tag = int(hashlib.sha256(line.encode("utf-8")).hexdigest()[:8], 16) % 10000
    sep = "" if lang in ("zh", "th") else " "
    out = f"{_HEAD[lang]}{sep}{tag:04d}"
    target = max(len(out), round(len(line) * _RATIO[lang]))
    while len(out) < target:
        out += _FILL[lang]
    out = out[:target].rstrip()
    return out.replace(" ", "  ", 1) if "  " in line else out


def _tr(texts: list[str], lang: str) -> list[str]:
    return [translate(en, lang) for en in texts]


def dump(path: Path, rows: list[dict], *, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, indent=4, ensure_ascii=ensure_ascii)


def _rows(texts: list[str], categories: list[str] | None = None) -> list[dict]:
    """The English and train/val layout: instruction, category."""
    cats = categories if categories is not None else [None] * len(texts)
    return [{"instruction": t, "category": c} for t, c in zip(texts, cats, strict=True)]


def _test_rows(pairs: list[tuple[str, str | None]], lang: str) -> list[dict]:
    """The translated test layout: English instruction, instruction_translated, category."""
    return [{"instruction": en, "instruction_translated": en if lang == "en" else translate(en, lang),
             "category": cat} for en, cat in pairs]


def build(root: Path) -> None:
    """Write the tree under ``root`` (the Multilingual-Refusal folder)."""
    pr, sp = root / "PolyRefuse", root / "dataset" / "splits"
    harmful_en, harmful_cats = [en for en, _ in HARMFUL_TEST], [cat for _, cat in HARMFUL_TEST]
    val_en, val_cats = [en for en, _ in HARMFUL_VAL], [cat for _, cat in HARMFUL_VAL]
    harmless_test = [(en, None) for en in HARMLESS_TEST]
    dump(pr / "harmful_test_translated_en.json", _rows(harmful_en, harmful_cats))
    dump(sp / "harmful_test.json", _rows(harmful_en, harmful_cats))    # byte-identical, as upstream
    dump(pr / "harmless_test_translated_en.json", _test_rows(harmless_test, "en"))
    dump(sp / "harmless_test_500_sampled.json", _rows(DECOY_HARMLESS_TEST))
    dump(sp / "harmful_train.json", _rows(HARMFUL_TRAIN))
    dump(sp / "harmful_val.json", _rows(val_en, val_cats))
    dump(sp / "harmless_train_200_sampled.json", _rows(HARMLESS_TRAIN))
    dump(sp / "harmless_val_200_sampled.json", _rows(HARMLESS_VAL))

    for lang in TRANSLATED:
        test_rows = _test_rows(HARMFUL_TEST, lang)
        if lang == "es":
            text = test_rows[0]["instruction_translated"]
            test_rows[0]["instruction_translated"] = text[:4] + "\u200b" + text[4:]
        dump(pr / f"harmful_test_translated_{lang}.json", test_rows)
        dump(pr / f"harmful_train_translated_{lang}.json", _rows(_tr(HARMFUL_TRAIN, lang)))
        dump(pr / f"harmful_val_translated_{lang}.json", _rows(_tr(val_en, lang), val_cats))
        if lang == "zh":                                   # no zh harmless test; train/val another draw
            dump(pr / "harmless_train_translated_zh.json", _rows(ZH_OTHER_TRAIN))
            dump(pr / "harmless_val_translated_zh.json", _rows(ZH_OTHER_VAL))
            continue
        dump(pr / f"harmless_test_translated_{lang}.json", _test_rows(harmless_test, lang))
        dump(pr / f"harmless_train_translated_{lang}.json", _rows(_tr(HARMLESS_TRAIN, lang)))
        dump(pr / f"harmless_val_translated_{lang}.json", _rows(_tr(HARMLESS_VAL, lang)))

    back = [{"instruction": en, "instruction_translated": translate(en, "es"),
             "instruction_back_translated": f"Back-translated placeholder {i}.", "category": cat}
            for i, (en, cat) in enumerate(HARMFUL_TEST)]
    dump(pr / "harmful_test_back_translated" / "harmful_test_back_translated_es.json", back,
         ensure_ascii=False)


if __name__ == "__main__":
    build(HERE / "Multilingual-Refusal")
