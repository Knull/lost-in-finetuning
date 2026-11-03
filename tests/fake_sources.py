"""Offline stand-ins for the Hub sources of lift.data.sources: rows shaped like the pinned splits.

All text is fake and benign. Import directly (``from fake_sources import ...``), like datafakes.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence

from lift import config

# OR-Bench-Hard-1K stand-in: 12 rows, categories a x6, b x4, c x2 (n=6 seats: a 3, b 2, c 1).
_ORBENCH_CATEGORIES = ("a", "b", "a", "c", "a", "b", "a", "b", "a", "c", "a", "b")
_TOPICS = ("gardening", "baking", "cycling", "painting", "knitting", "hiking")


def orbench_rows() -> list[dict]:
    """Hub rows {"prompt", "category", "_row"}. Row 5 has a stray U+200B and spaces at the ends."""
    rows = [{"prompt": f"Placeholder request {i} about {_TOPICS[i % len(_TOPICS)]}", "category": c, "_row": i}
            for i, c in enumerate(_ORBENCH_CATEGORIES)]
    rows[5]["prompt"] = f"  {rows[5]['prompt']}​ "
    return rows


def _labels(*labels: str) -> str:
    return json.dumps(list(labels))


def redteam_rows() -> list[dict]:
    """5 rows shaped like the aya_redteaming "hindi" split. ``harm_category`` is a JSON string, as
    upstream (one label contains a comma). Row 2 has whitespace at the ends and an inner newline;
    row 3 repeats row 1's prompt with other labels (upstream rows 2 and 32 do the same)."""
    prompt = "बागवानी के बारे में नकली अनुरोध {}"          # "fake request {} about gardening"
    rows = [
        {"prompt": prompt.format(0), "language": "Hindi", "harm_category": _labels("Profanity"),
         "global_or_local": "global", "explanation": None,
         "literal_translation": "Placeholder request 0 about gardening", "semantic_translation": None},
        {"prompt": prompt.format(1), "language": "Hindi",
         "harm_category": _labels("Hate Speech", "Bullying & Harassment"), "global_or_local": "local",
         "explanation": "Placeholder explanation 1",
         "literal_translation": "Placeholder request 1 about gardening",
         "semantic_translation": "Placeholder semantic translation 1"},
        {"prompt": f"  {prompt.format(2)}\nदूसरी पंक्ति \n", "language": "Hindi",
         "harm_category": _labels("Violence, Threats & Incitement", "Graphic material", "Self-Harm"),
         "global_or_local": "global", "explanation": None,
         "literal_translation": "Placeholder request 2 about gardening\nsecond line",
         "semantic_translation": None},
        {"prompt": prompt.format(1), "language": "Hindi",
         "harm_category": _labels("Discrimination & Injustice"), "global_or_local": "global",
         "explanation": None, "literal_translation": "Placeholder request 1 about gardening, again",
         "semantic_translation": None},
        {"prompt": prompt.format(4), "language": "Hindi",
         "harm_category": _labels("Harms of Representation Allocation and Quality of Service"),
         "global_or_local": "local", "explanation": "Placeholder explanation 4",
         "literal_translation": "Placeholder request 4 about gardening", "semantic_translation": None},
    ]
    return [{**row, "_row": i} for i, row in enumerate(rows)]


# Belebele stand-in: 5 items, (link, question_number) sorted like eng_Latn, two links with two
# questions. Items 3 and 4 share a link and have different answers, so a by-row reading of
# hin_Deva (rows 3 and 4 swapped) gets both answers wrong.
_BELEBELE_KEYS = (("https://example.org/placeholder/a", 1), ("https://example.org/placeholder/a", 2),
                  ("https://example.org/placeholder/b", 1), ("https://example.org/placeholder/c", 1),
                  ("https://example.org/placeholder/c", 2))
_BELEBELE_ANSWERS = ("2", "4", "1", "3", "1")
# (passage, question, answer) templates in each config's script, so FakeLID reads the right language.
_BELEBELE_TEXT = {
    "eng_Latn": ("Placeholder passage {k} about gardening.", "Which plant is in passage {k}?",
                 "Answer {a} for item {k}"),
    "hin_Deva": ("बागवानी के बारे में नकली अनुच्छेद {k}।", "अनुच्छेद {k} में कौन सा पौधा है?",
                 "प्रश्न {k} का उत्तर {a}"),
    "zho_Hans": ("关于园艺的占位段落{k}。", "第{k}段中是哪种植物？", "第{k}题的答案{a}"),
    "spa_Latn": ("Pasaje de relleno {k} sobre jardinería.", "¿Qué planta aparece en el pasaje {k}?",
                 "Respuesta {a} del ítem {k}"),
    "arb_Arab": ("فقرة نموذجية {k} عن البستنة.", "ما النبات في الفقرة {k}؟", "الإجابة {a} للسؤال {k}"),
    "tha_Thai": ("ย่อหน้าตัวอย่าง {k} เกี่ยวกับการทำสวน", "พืชชนิดใดอยู่ในย่อหน้า {k}",
                 "คำตอบ {a} ของข้อ {k}"),
}
_BELEBELE_DS = {"eng_Latn": "2023-05-03", "hin_Deva": "2023-07-21", "zho_Hans": "2023-07-21",
                "spa_Latn": "2023-06-01", "arb_Arab": "2023-06-13", "tha_Thai": "2023-06-01"}


def belebele_rows_by_config() -> dict[str, list[dict]]:
    """{config: Hub rows} for the six configs, 5 items each. hin_Deva rows 3 and 4 are swapped
    relative to eng_Latn (upstream permutes its rows 460-899). ``correct_answer_num`` is a string and
    every row has the raw "ds" date string, as upstream."""
    out = {}
    for code, (passage, question, answer) in _BELEBELE_TEXT.items():
        items = []
        for k, ((link, number), correct) in enumerate(zip(_BELEBELE_KEYS, _BELEBELE_ANSWERS, strict=True)):
            row = {"link": link, "question_number": number, "flores_passage": passage.format(k=k),
                   "question": question.format(k=k), "correct_answer_num": correct, "dialect": code,
                   "ds": _BELEBELE_DS[code]}
            row.update((f"mc_answer{a}", answer.format(a=a, k=k)) for a in range(1, 5))
            items.append(row)
        if code == "hin_Deva":
            items[3], items[4] = items[4], items[3]
        out[code] = [{**row, "_row": j} for j, row in enumerate(items)]
    return out


class FakeHub:
    """A Hub (lift.data.sources.Hub) serving fixed rows. ``tables`` maps a source name, or
    (source, lang) for "belebele", to its rows; every call returns a deep copy. ``calls`` records
    (source, lang)."""

    def __init__(self, tables: Mapping[str | tuple[str, str], Sequence[Mapping]]) -> None:
        self.tables = dict(tables)
        self.calls: list[tuple[str, str | None]] = []

    def rows(self, source: str, *, lang: str | None = None) -> list[dict]:
        self.calls.append((source, lang))
        key = source if lang is None else (source, lang)
        if key not in self.tables:
            raise KeyError(f"FakeHub has no rows for {key!r}")
        return [copy.deepcopy(dict(row)) for row in self.tables[key]]


def fake_hub(extra: Mapping[str | tuple[str, str], Sequence[Mapping]] | None = None) -> FakeHub:
    """A FakeHub with orbench, aya_redteaming and belebele (every language, by our code), plus
    ``extra`` tables (e.g. the SFT sources "aya_dataset", "indicalign", "dolly")."""
    by_config = belebele_rows_by_config()
    tables: dict[str | tuple[str, str], Sequence[Mapping]] = {
        "orbench": orbench_rows(), "aya_redteaming": redteam_rows()}
    for lang, code in config.load("project")["languages"]["belebele"].items():
        tables["belebele", lang] = by_config[code]
    tables.update(extra or {})
    return FakeHub(tables)
