from types import SimpleNamespace

import pytest

from lift import config
from lift.lang import LangID, check_accept_labels, lid_info, one_line

CFG = config.load("project")
GLOTLID_LABELS = sorted({lab for labs in CFG["languages"]["accept_labels"].values() for lab in labs}
                        | {"mar_Deva", "fas_Arab", "jpn_Jpan"})


class FakeFastText:
    """Mimics the fastText model surface LangID uses: ``get_labels`` and ``f.predict``."""

    def __init__(self, preds, labels=GLOTLID_LABELS):
        self.preds = preds                      # list of (prob, "__label__x")
        self.labels = [f"__label__{x}" for x in labels]
        self.seen = []
        self.f = SimpleNamespace(predict=self._predict)

    def get_labels(self):
        return list(self.labels)

    def _predict(self, text, k, threshold, on_unicode_error):
        self.seen.append(text)
        return self.preds[:k]


def lid(preds, **kw):
    return LangID(model=FakeFastText(preds, **kw), cfg=CFG)


def test_top_strips_prefix_clamps_and_respects_k():
    m = lid([(1.00001, "__label__hin_Deva"), (0.2, "__label__mar_Deva"), (0.1, "__label__eng_Latn")])
    assert m.top("x", k=2) == [("hin_Deva", 1.0), ("mar_Deva", 0.2)]


def test_text_reaches_fasttext_as_one_line_ending_in_newline():
    fake = FakeFastText([(0.9, "__label__eng_Latn")])
    m = LangID(model=fake, cfg=CFG)
    m.top("first line\nsecond line\r\nthird\rfourth")
    (sent,) = fake.seen
    assert sent.endswith("\n") and sent.count("\n") == 1 and "\r" not in sent
    assert sent == "first line second line third fourth\n"
    assert one_line("") == "\n"


def test_is_lang_uses_accept_labels_and_min_prob():
    assert lid([(0.97, "__label__hin_Deva")]).is_lang("x", "hi")
    assert lid([(0.7, "__label__hin_Latn")]).is_lang("x", "hi")          # romanised Hindi is Hindi
    assert not lid([(0.49, "__label__hin_Deva")]).is_lang("x", "hi")     # below min_prob
    assert not lid([(0.99, "__label__mar_Deva")]).is_lang("x", "hi")     # another Devanagari language
    assert lid([(0.51, "__label__cmn_Hani")]).is_lang("x", "zh")
    assert not lid([(0.6, "__label__jpn_Jpan")]).is_lang("x", "zh")
    assert not lid([]).is_lang("x", "en")
    with pytest.raises(ValueError):
        lid([(0.9, "__label__eng_Latn")]).is_lang("x", "fr")


def test_is_lang_sums_the_varieties_of_one_language():
    # GlotLID often splits Modern Standard Arabic between arb_Arab and a dialect label.
    split = [(0.45, "__label__arb_Arab"), (0.40, "__label__ars_Arab"), (0.1, "__label__fas_Arab")]
    assert lid(split).is_lang("x", "ar")
    assert lid([(0.30, "__label__cmn_Hani"), (0.25, "__label__lzh_Hani")]).is_lang("x", "zh")
    assert not lid([(0.45, "__label__arb_Arab"), (0.40, "__label__fas_Arab")]).is_lang("x", "ar")
    # the top label itself must belong to the language
    assert not lid([(0.45, "__label__fas_Arab"), (0.40, "__label__arb_Arab"),
                    (0.15, "__label__ars_Arab")]).is_lang("x", "ar")


def test_missing_accept_labels_raise_at_load():
    with pytest.raises(ValueError, match="cmn_Hani"):
        lid([(0.9, "__label__eng_Latn")], labels=[x for x in GLOTLID_LABELS if x != "cmn_Hani"])
    with pytest.raises(ValueError, match="zho_Hans"):
        check_accept_labels(["__label__cmn_Hani"], {"zh": ["cmn_Hani", "zho_Hans"]})
    check_accept_labels([f"__label__{x}" for x in GLOTLID_LABELS], CFG["languages"]["accept_labels"])


def test_lid_info_rounds_and_handles_empty_predictions():
    info = lid_info(lid([(0.123456789, "__label__spa_Latn")]), "x")
    assert (info.label, info.prob) == ("spa_Latn", 0.1235)
    assert lid_info(lid([]), "x").label == "und"


@pytest.mark.net
def test_real_glotlid_labels_and_predictions():
    m = LangID(cfg=CFG)                                   # downloads the pinned revision (~1.6 GB)
    check_accept_labels(m.labels, CFG["languages"]["accept_labels"])
    assert m.is_lang("यह एक परीक्षण वाक्य है।", "hi")
    assert m.is_lang("这是一个测试句子。", "zh")
    assert m.is_lang("Esta es una frase de prueba.", "es")
    assert m.is_lang("هذه جملة اختبار.", "ar")
    assert m.is_lang("นี่คือประโยคทดสอบ", "th")
    assert m.is_lang("How do I kill a Python process?", "en")
