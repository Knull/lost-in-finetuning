"""Offline stand-ins for the SFT sources (lift.data.sft), shaped like the hub rows of the real sets.

Every text is invented, benign and short: Hindi questions about tea, kites and tomato plants, and
English "Placeholder request N about TOPIC" instructions. Built-in quirks, by design:
- aya_dataset_rows(): 18 Hindi rows with non-contiguous ``_row``. idx 2 has an English input, 4 an
  empty target (only U+200B and spaces), 6 repeats idx 1 exactly, 8 near-duplicates idx 3's
  instruction, 9 has a degenerate repetitive target, 11 an English target, 13 is under CuratorKIT's
  10-token minimum, and 17's instruction near-duplicates IndicAlign row 12's.
- indicalign_rows() / dolly_rows(): 19 Dolly_T rows. Rows 2, 3 and 7 had a Dolly context (row 3's
  instruction has two Dolly records, one with a context), row 6 matches no Dolly record; row 4
  matches only after whitespace is collapsed, row 5 only after NFC, row 8's context is blank. Row 9
  has an English response, 14's response near-duplicates 13's, 15 repeats 11's Hindi exactly.
With sources.sft.heldout.per_source = 2 both pools keep at least 8 deduplicated training rows, but
that draw (the project seed) hits none of the near-duplicate quirks. With per_source = 10
(``QUIRK_DRAW``) the draw is forced to hit them all: aya has 11 eligible rows and may take only one
of the near pair 3/8, so it takes the other ten (3 by rank, and 17); indicalign has 12 and may take
neither ind 12 (a near-duplicate of aya 17) nor both of 13/14, so it takes the other ten (13 by
rank). The exclusion then removes aya 8, ind 14 and, across sources, ind 12, leaving aya 1, 6, 13
and ind 11, 15 to train on.
Not a conftest (P02 owns tests/conftest.py): import with ``from fake_sft import ...``.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lift import config
from lift.data.text import char_ngrams, dup_key, jaccard

# sources.sft.heldout.per_source and sizes.utility for the draw that hits every near-duplicate quirk.
QUIRK_DRAW: dict[str, Any] = {"per_source": 10, "utility": {"dev": 5, "final": 15}}


def near(a_ins: str, a_res: str, b_ins: str, b_res: str, nd: Mapping[str, Any] | None = None) -> bool:
    """The held-out near-duplicate rule written out pair by pair (the reference for sft's indexed
    join). ``nd``: sources.sft.heldout.near_dup, the project config's by default."""
    nd = nd if nd is not None else config.load("project")["sources"]["sft"]["heldout"]["near_dup"]
    ins = (bool(dup_key(a_ins)) and bool(dup_key(b_ins))
           and jaccard(char_ngrams(a_ins, nd["ngram"]), char_ngrams(b_ins, nd["ngram"])) >= nd["instruction"])
    long = len(dup_key(a_res)) >= nd["response_min_chars"] and len(dup_key(b_res)) >= nd["response_min_chars"]
    res = long and jaccard(char_ngrams(a_res, nd["ngram"]), char_ngrams(b_res, nd["ngram"])) >= nd["response"]
    return ins or res

# (instruction, response) pairs, Hindi, invented.
_AYA = [
    ("टमाटर के पौधों को गर्मियों में कितनी बार पानी देना चाहिए?",
     "गर्मियों में टमाटर के पौधों को रोज़ सुबह हल्का पानी दें और ध्यान रखें कि मिट्टी पूरी तरह सूखने न पाए।"),
    ("घर पर अदरक वाली चाय बनाने का आसान तरीका बताइए।",
     "एक कप पानी उबालें, उसमें कुटी हुई अदरक और चायपत्ती डालें, फिर दूध और चीनी मिलाकर दो मिनट तक पकाएँ।"),
    ("Translate into Hindi: a placeholder sentence about gardening tools.",
     "बागवानी के औज़ारों के बारे में एक नमूना वाक्य।"),
    ("बरसात के मौसम में छाता साथ रखना क्यों अच्छा रहता है?",
     "बरसात में कभी भी अचानक बारिश हो सकती है, इसलिए छाता रखने से कपड़े और किताबें भीगने से बच जाती हैं।"),
    ("रंगोली बनाने के लिए किन रंगों का उपयोग अच्छा रहता है?", "\u200b  "),
    ("बच्चों के लिए कोई छोटी और मज़ेदार पहेली सुनाइए।",
     "ऐसी कौन सी चीज़ है जिसके दाँत तो होते हैं पर वह कभी काटती नहीं? इसका उत्तर है कंघी।"),
    None,   # idx 6: exact copy of idx 1
    ("रोज़ सुबह टहलने के तीन फ़ायदे लिखिए।",
     "सुबह टहलने से शरीर में ताज़गी आती है, रात को नींद बेहतर होती है और पूरे दिन मन प्रसन्न रहता है।"),
    ("बरसात के मौसम में छाता साथ रखना क्यों अच्छा रहता है, बताइए?",
     "छाता हल्का और मोड़ने वाला चुनें ताकि वह बस्ते में आसानी से आ जाए और भीड़ में किसी को चुभे नहीं।"),
    ("एक शब्द को बार-बार लिखकर दिखाइए।", " ".join(["नमस्ते"] * 40)),
    ("पुस्तकालय से किताब उधार लेने की प्रक्रिया समझाइए।",
     "अपना सदस्यता कार्ड दिखाइए, किताब चुनिए और काउंटर पर उसका विवरण दर्ज करवाकर लौटाने की तारीख़ नोट कर लीजिए।"),
    ("मौसम का हाल बताने वाले ऐप के लिए एक नाम सुझाइए।",
     "You could call it Weather Buddy, a short and friendly placeholder name for the app."),
    ("साइकिल की चेन में समय-समय पर तेल डालना क्यों ज़रूरी है?",
     "तेल डालने से चेन पर जंग नहीं लगती, पैडल चलाना आसान होता है और पुर्ज़े लंबे समय तक चलते हैं।"),
    ("नमस्ते कहिए।", "नमस्ते!"),
    ("किसी मित्र को जन्मदिन की शुभकामना का छोटा संदेश लिखिए।",
     "जन्मदिन की ढेर सारी बधाई! तुम्हारा नया साल हँसी, अच्छी सेहत और नई सफलताओं से भरा रहे।"),
    ("आसमान नीला क्यों दिखाई देता है, सरल शब्दों में बताइए।",
     "सूरज की रोशनी हवा के छोटे कणों से टकराकर बिखरती है और नीला रंग सबसे ज़्यादा बिखरता है, इसलिए आसमान नीला दिखता है।"),
    ("गणित में सम और विषम संख्याओं का अंतर क्या है?",
     "जो संख्या दो से पूरी तरह बँट जाए वह सम कहलाती है, जैसे आठ, और जो न बँटे वह विषम है, जैसे सात।"),
    ("तुलसी के पौधे की देखभाल घर की बालकनी में कैसे करें?",
     "तुलसी को रोज़ थोड़ी धूप और नियमित पानी चाहिए, और गमले में पानी निकलने का छेद होना ज़रूरी है।"),
]
_AYA_ROWS = [3, 7, 12, 20, 21, 35, 36, 50, 64, 65, 80, 81, 99, 120, 121, 140, 141, 160]

AYA_ENGLISH_INPUT, AYA_EMPTY_TARGET, AYA_DUPLICATE, AYA_NEAR_DUP = 2, 4, 6, 8   # idx in the fixture
AYA_REPETITIVE, AYA_ENGLISH_TARGET, AYA_SHORT, AYA_CROSS = 9, 11, 13, 17

# Dolly_T: (English instruction as IndicAlign stores it, Hindi instruction, Hindi response).
_IND = [
    ("Placeholder request 0 about gardening", "बीज बोने से पहले मिट्टी को कैसे तैयार किया जाता है?",
     "मिट्टी को खोदकर ढीला करें, उसमें से कंकड़ निकालें, थोड़ी खाद मिलाएँ और बुआई से एक दिन पहले हल्की सिंचाई करें।"),
    ("Placeholder request 1 about cooking", "दाल को जल्दी पकाने के लिए क्या किया जा सकता है?",
     "दाल को पकाने से पहले आधे घंटे भिगो दें और प्रेशर कुकर का उपयोग करें, इससे समय और ईंधन दोनों बचते हैं।"),
    ("Placeholder request 2 about weather", "दिए गए अनुच्छेद के अनुसार बादल कैसे बनते हैं?",
     "अनुच्छेद के अनुसार जलवाष्प ठंडी होकर छोटी बूँदों में बदलती है और बादल बनते हैं।"),
    ("Placeholder request 3 about travel", "दिए गए विवरण से रेलगाड़ियों के नाम निकालिए।",
     "विवरण में तीन रेलगाड़ियों के नाम दिए गए हैं जिन्हें सूची में लिखा जा सकता है।"),
    ("Placeholder request 4 about music", "हारमोनियम सीखना शुरू करने वाले के लिए कोई सुझाव दीजिए।",
     "पहले सरगम के सातों स्वर धीरे-धीरे बजाना सीखें, रोज़ थोड़ा अभ्यास करें और किसी अनुभवी शिक्षक से सलाह लेते रहें।"),
    (unicodedata.normalize("NFD", "Placeholder request 5 about café"),
     "किसी छोटे कैफ़े का नाम सुझाइए जो याद रखने में आसान हो।",
     "आप अपने कैफ़े का नाम चाय चौपाल रख सकते हैं, यह छोटा है, सुनने में अपना लगता है और आसानी से याद रहता है।"),
    ("Placeholder request 6 about painting", "जलरंग से चित्र बनाने के लिए कौन सा काग़ज़ अच्छा है?",
     "जलरंग के लिए मोटा और खुरदुरा काग़ज़ अच्छा रहता है क्योंकि वह पानी सोख लेता है।"),
    ("Placeholder request 7 about summaries", "इस लेख का सारांश तीन वाक्यों में लिखिए।",
     "लेख में शहर के पुराने बाज़ार, उसकी गलियों और वहाँ मिलने वाली मिठाइयों का वर्णन है।"),
    ("Placeholder request 8 about cricket", "क्रिकेट में एक ओवर में कितनी गेंदें फेंकी जाती हैं?",
     "क्रिकेट के एक ओवर में गेंदबाज़ छह वैध गेंदें फेंकता है, वाइड और नो बॉल इनमें नहीं गिनी जातीं।"),
    ("Placeholder request 9 about books", "बच्चों के लिए कहानी की किताब चुनते समय किन बातों का ध्यान रखें?",
     "Choose books with large pictures, short sentences and a story that matches the child's age."),
    ("Placeholder request 10 about the moon", "चाँद हर रात अलग आकार का क्यों दिखाई देता है?",
     "चाँद सूरज की रोशनी को परावर्तित करता है और पृथ्वी के चारों ओर घूमते हुए "
     "उसका प्रकाशित भाग हमें अलग-अलग मात्रा में दिखता है।"),
    ("Placeholder request 11 about water", "घर में पानी बचाने के कुछ आसान उपाय बताइए।",
     "नल को ब्रश करते समय बंद रखें, टपकते नलों की तुरंत मरम्मत करवाएँ और पौधों को सुबह या शाम पानी दें।"),
    ("Placeholder request 12 about tulsi", "तुलसी के पौधे की देखभाल घर की बालकनी में कैसे करें, बताइए?",
     "तुलसी का गमला ऐसी जगह रखें जहाँ सुबह की धूप आती हो और सर्दियों में उसे पाले से बचाएँ।"),
    ("Placeholder request 13 about skin care", "सर्दियों में त्वचा को रूखा होने से कैसे बचाएँ?",
     "सर्दियों में नहाने के बाद नारियल का तेल लगाएँ, खूब पानी पिएँ और बहुत गरम पानी से नहाने से बचें।"),
    ("Placeholder request 14 about cold days", "ठंड के दिनों में हाथ-पैर फटने से रोकने का उपाय क्या है?",
     "सर्दियों में नहाने के बाद नारियल का तेल लगाएँ, खूब पानी पिएँ और बहुत गरम पानी से नहाने से हमेशा बचें।"),
    ("Placeholder request 15 about saving water", "घर में पानी बचाने के कुछ आसान उपाय बताइए।",
     "नल को ब्रश करते समय बंद रखें, टपकते नलों की तुरंत मरम्मत करवाएँ और पौधों को सुबह या शाम पानी दें।"),
    ("Placeholder request 16 about kitchens", "रसोई की सफ़ाई रोज़ करना क्यों आवश्यक है?",
     "रोज़ सफ़ाई करने से रसोई में कीड़े-मकोड़े नहीं आते, खाना सुरक्षित रहता है और काम करने में भी मन लगता है।"),
    ("Placeholder request 17 about exercise", "दफ़्तर में लंबे समय तक बैठने वालों के लिए कोई हल्का व्यायाम बताइए।",
     "हर घंटे उठकर दो मिनट चलें, कंधों को धीरे-धीरे घुमाएँ और गर्दन को दाएँ-बाएँ झुकाकर खिंचाव दें।"),
    ("Placeholder request 18 about stars", "रात में तारे टिमटिमाते हुए क्यों दिखते हैं?",
     "तारों की रोशनी पृथ्वी के वायुमंडल की अलग-अलग परतों से गुज़रते हुए मुड़ती है, इसलिए वे टिमटिमाते हुए दिखाई देते हैं।"),
]
IND_HAD_CONTEXT, IND_NO_MATCH = (2, 3, 7), (6,)
IND_ENGLISH_RESPONSE, IND_CROSS, IND_NEAR_PAIR, IND_DUPLICATE = 9, 12, (13, 14), (11, 15)


def aya_dataset_rows() -> list[dict]:
    """Hub "aya_dataset" rows: the Hindi rows only, each with its HF ``_row``."""
    rows = []
    for k, (pair, hf_row) in enumerate(zip(_AYA, _AYA_ROWS, strict=True)):
        inputs, targets = pair if pair is not None else _AYA[1]
        rows.append({"inputs": inputs, "targets": targets, "language": "Hindi", "language_code": "hin",
                     "annotation_type": "re-annotations" if k % 3 == 0 else "original-annotations",
                     "user_id": f"user{k % 4}", "_row": hf_row})
    return rows


def indicalign_rows() -> list[dict]:
    """Hub "indicalign" rows (Dolly_T with only doc_id, eng_Latn, hin_Deva, plus ``_row``)."""
    return [{"doc_id": f"doc-{i:04d}", "eng_Latn": [[en, f"Placeholder answer {i}."]],
             "hin_Deva": [[hi, ans]], "_row": i} for i, (en, hi, ans) in enumerate(_IND)]


def dolly_rows() -> list[dict]:
    """Hub "dolly" rows (databricks-dolly-15k records): three of the instructions above had a
    context, one (request 6) has no record, one record matches nothing at all."""
    ctx = {2: "Placeholder passage about clouds.", 7: "Placeholder passage to summarise."}
    records = []
    for i, (en, _, _) in enumerate(_IND):
        if i in IND_NO_MATCH:
            continue
        instruction = unicodedata.normalize("NFC", en)
        if i == 4:
            instruction = "Placeholder  request 4 about\tmusic "
        records.append({"instruction": instruction, "context": ctx.get(i, "   " if i == 8 else ""),
                        "response": f"Placeholder answer {i}.", "category": "open_qa"})
        if i == 3:
            records.append({"instruction": en, "context": "Placeholder passage about trains.",
                            "response": "Placeholder answer.", "category": "information_extraction"})
    records.append({"instruction": "Placeholder request 99 about chess", "context": "",
                    "response": "Placeholder answer 99.", "category": "open_qa"})
    return [{**r, "_row": i} for i, r in enumerate(records)]


class SftHub:
    """A Hub (lift.data.sources) serving the three SFT sources from the fixtures above, or from
    ``sources`` ({source: rows}) when given. ``calls`` lists the sources requested."""

    def __init__(self, sources: dict[str, list[dict]] | None = None) -> None:
        self.sources = sources or {"aya_dataset": aya_dataset_rows(), "indicalign": indicalign_rows(),
                                   "dolly": dolly_rows()}
        self.calls: list[str] = []

    def rows(self, source: str, *, lang: str | None = None) -> list[dict]:
        self.calls.append(source)
        return [dict(r) for r in self.sources[source]]


class FakeCurate:
    """Stands in for lift.data.sft.run_curatorkit: with ``dedup`` drops exact duplicates (dup_key of
    instruction and output), keeps order, writes placeholder CuratorKIT files into
    out_dir/curatorkit (manifest.json with CRLF, as CuratorKIT writes on Windows) and
    out_dir/curatorkit_summary.json. ``calls`` records (rows, out_dir, dedup)."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, Path, bool]] = []

    def __call__(self, rows: list[dict], out_dir: Path, dedup: bool) -> list[dict]:
        out_dir = Path(out_dir)
        self.calls.append((len(rows), out_dir, dedup))
        seen: set[tuple[str, str]] = set()
        kept = []
        for r in rows:
            key = (dup_key(r["instruction"]), dup_key(r["output"]))
            if dedup and key in seen:
                continue
            seen.add(key)
            kept.append(r)
        ck = out_dir / "curatorkit"
        ck.mkdir(parents=True, exist_ok=True)
        manifest = {"placeholder": True, "input": len(rows), "passed": len(kept)}
        (ck / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                          newline="\r\n")
        (ck / "rejected.jsonl").write_text("", encoding="utf-8", newline="\n")
        (ck / "lexsi_provenance.json").write_text(json.dumps({"placeholder": True}) + "\n",
                                                  encoding="utf-8", newline="\n")
        summary = {"stage_counts": {"FakeDedup": {"input_count": len(rows), "output_count": len(kept),
                                                  "rejected_count": 0}},
                   "rejected_breakdown": {},
                   "dedup_stats": {"exact_duplicates_removed": len(rows) - len(kept)},
                   "minhash_threshold": None}
        (out_dir / "curatorkit_summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n",
                                                         encoding="utf-8", newline="\n")
        return [{"id": r["id"], "messages": [{"role": "user", "content": r["instruction"]},
                                             {"role": "assistant", "content": r["output"]}]} for r in kept]
