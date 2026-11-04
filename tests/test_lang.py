import re
import unicodedata
from types import SimpleNamespace

import pytest
from datafakes import FakeLID

from lift import config
from lift.judge.labels import unclear_flags
from lift.lang import (
    BLANK,
    LangID,
    check_accept_labels,
    has_visible,
    lid_info,
    model_line,
    one_line,
    space_han,
)

CFG = config.load("project")
ACCEPT = CFG["languages"]["accept_labels"]
GLOTLID_LABELS = sorted({lab for labs in ACCEPT.values() for lab in labs}
                        | {"mar_Deva", "fas_Arab", "jpn_Jpan", "und_Tang"})


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


class GlotLIDTraps(FakeFastText):
    """Also mimics two verified GlotLID failures. A line of only whitespace, control and format
    characters reads lzh_Hani 1.0 (exactly so for whitespace and NUL, which leave fastText the
    end-of-line token alone; BOMs, ideographic spaces and ZWNJ get Chinese labels at 0.98-1.0).
    A run of two or more unspaced Han characters reads und_Tang 1.0."""

    def _predict(self, text, k, threshold, on_unicode_error):
        self.seen.append(text)
        if all(unicodedata.category(c) in ("Cc", "Cf", "Zs", "Zl", "Zp") for c in text):
            return [(1.00001, "__label__lzh_Hani"), (1e-05, "__label__yue_Hani")][:k]
        if re.search(r"[\u4e00-\u9fff]{2}", text):
            return [(1.00001, "__label__und_Tang"), (1e-05, "__label__cmn_Hani")][:k]
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


BLANKS = ["", " ", "\n" * 256, "\r\n\t\v\f", "\x00", "\u200b\ufeff", "\u00a0\u3000\u2028",
          "\u200c\u200d\u200e\u2060\x1f"]


@pytest.mark.parametrize("text", BLANKS)
def test_blank_text_is_no_language_and_garbled(text):
    fake = GlotLIDTraps([(0.99, "__label__eng_Latn")])
    m = LangID(model=fake, cfg=CFG)
    assert fake.f.predict(one_line(text), 3, 0.0, "strict")[0][1] == "__label__lzh_Hani"   # the trap
    fake.seen.clear()
    assert BLANK == ("und_Zyyy", 0.0) and BLANK[0] not in GLOTLID_LABELS
    assert m.top(text, 5) == [BLANK] == FakeLID().top(text, 5)        # P02's fake agrees
    assert fake.seen == []                                              # the model is never asked
    assert not any(m.is_lang(text, lang) for lang in ACCEPT)
    assert lid_info(m, text).label == "und_Zyyy"
    f = unclear_flags(text=text, n_new_tokens=256, lid_top=m.top(text, 5), prompt_lang="zh",
                      accept_labels=ACCEPT)
    assert f.garbled and not f.wrong_lang


def test_one_visible_character_reaches_the_model():
    fake = GlotLIDTraps([(0.9, "__label__hin_Deva")])
    m = LangID(model=fake, cfg=CFG)
    for text in ("\u200b.\ufeff", "  x  \n", "\u200dक"):
        assert has_visible(text)
        assert m.top(text) == [("hin_Deva", 0.9)]
    assert len(fake.seen) == 3
    assert not has_visible("".join(BLANKS))


def test_han_is_spaced_before_fasttext_sees_it():
    fake = FakeFastText([(0.99, "__label__cmn_Hani")])
    m = LangID(model=fake, cfg=CFG)
    m.top("我不能帮你。\n好的")
    m.top("用Python编程")
    m.top("\u3400\uf900\U00020000\U0002f800")    # Extension A, compatibility, Extension B, supplement
    assert [line.split() for line in fake.seen] == [
        ["我", "不", "能", "帮", "你", "。", "好", "的"],
        ["用", "Python", "编", "程"],
        ["\u3400", "\uf900", "\U00020000", "\U0002f800"],
    ]
    assert all(line.endswith("\n") and line.count("\n") == 1 for line in fake.seen)
    assert fake.seen[0] == model_line("我不能帮你。\n好的")


def test_long_text_is_spaced_too():
    # No length limit: unspaced, long Chinese answers with code read as garbled (module docstring).
    def ideograph(c):
        return unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH")

    fake = FakeFastText([(0.99, "__label__cmn_Hani")])
    text = ("下面是一个计算平均值的函数：\n\n"
            "```python\ndef average(xs):\n    return sum(xs) / len(xs)\n```\n\n"
            + "这个函数先求和，再除以元素个数。" * 30)
    assert len(text) > 500
    LangID(model=fake, cfg=CFG).top(text, 5)
    (line,) = fake.seen
    han_words = [w for w in line.split() if any(map(ideograph, w))]
    assert len(han_words) == sum(map(ideograph, text)) > 400
    assert all(len(w) == 1 for w in han_words)
    assert "average(xs):" in line.split()


def test_han_spacing_fixes_unsegmented_short_chinese():
    m = LangID(model=GlotLIDTraps([(0.99, "__label__cmn_Hani")]), cfg=CFG)
    assert m.top("怎样种西红柿", 2) == [("cmn_Hani", 0.99)]
    assert m.is_lang("怎样种西红柿", "zh")
    # what the unspaced line would have given
    assert m._model.f.predict(one_line("怎样种西红柿"), 1, 0.0, "strict")[0][1] == "__label__und_Tang"


def test_japanese_and_text_without_ideographs_are_not_spaced():
    for text in ("東京は日本の首都です。", "コンピューター", "안녕하세요", "Hola, ¿qué tal?", "नमस्ते",
                 "，。！", "ㄅㄆㄇ"):
        assert space_han(text) == text
    assert space_han("日本国憲法").split() == list("日本国憲法")   # kanji only: nothing says Japanese
    # the middle dot and the prolonged sound mark are not kana letters; Chinese writes them too
    assert space_han("约翰・列侬").split() == ["约", "翰", "・", "列", "侬"]
    assert space_han("咖啡ー").split() == ["咖", "啡", "ー"]
    assert space_han("ラーメン店") == "ラーメン店"
    fake = FakeFastText([(0.99, "__label__jpn_Jpan")])
    LangID(model=fake, cfg=CFG).top("申し訳ありませんが、お答えできません。")
    assert fake.seen == ["申し訳ありませんが、お答えできません。\n"]


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
    # blank text is no language (GlotLID itself says lzh_Hani 1.0 for it)
    for text in ("", "\n" * 3, " \u200b\ufeff "):
        assert m.top(text, 5) == [BLANK] and not m.is_lang(text, "zh")
    # short unsegmented Chinese: und_Tang 0.98 and und_Grek 1.0 unspaced (verified Nov 3)
    assert m.is_lang("怎样种西红柿", "zh") and m.is_lang("数学作业", "zh")
    # a Chinese answer with code: eng_Latn 0.10 unspaced, so garbled (verified Nov 4)
    code_answer = ("下面是一个计算列表平均值的 Python 函数：\n\n"
                   "```python\ndef average(numbers):\n    if not numbers:\n        return 0\n"
                   "    return sum(numbers) / len(numbers)\n\nprint(average([1, 2, 3, 4]))\n```\n\n"
                   "这个函数首先检查列表是否为空，避免除以零的错误，"
                   "然后用 sum() 求和，再除以元素个数 len()。运行后会输出 2.5。")
    assert m.is_lang(code_answer, "zh")
    # English with a few Chinese words stays English
    assert m.is_lang("The Chinese word for 'thank you' is 谢谢 (xièxie), and the word for 'hello' is "
                     "你好 (nǐ hǎo). Both are used every day.", "en")
    # Japanese (it has kana) is not spaced and stays Japanese
    assert m.top("東京は日本の首都です。", 1)[0][0] == "jpn_Jpan"
    assert not m.is_lang("申し訳ありませんが、その質問にはお答えできません。", "zh")
