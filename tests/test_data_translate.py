"""lift.data.translate, offline: FakeTranslator/FakeLID backends, a fake deep_translator module and
fake torch/transformers modules. Nothing here calls Google; the only network test is the real NLLB
one (marker ``net``). deep-translator's and requests' real exception classes are used where they are
installed (no network: only their classes)."""

import copy
import datetime as dt
import functools
import importlib.abc
import logging
import subprocess
import sys
import traceback
import types

import pytest
from datafakes import FakeLID, FakeTranslator, fake_translation

import lift.lang
from lift import config
from lift.data import translate as tr
from lift.data.translate import (
    ENGINE_NAMES,
    BackTranslationError,
    TranslationBlocked,
    TranslationCache,
    TranslationError,
    TranslationFailure,
    back_translate,
    check_nllb,
    google_backend,
    is_transient,
    nllb_backend,
    translate_records,
    translate_texts,
    validate,
)
from lift.io import append_jsonl, sha256_obj, sha256_text
from lift.paths import output_dir
from lift.schemas import LidInfo, PromptRecord, TranslationInfo

CFG = copy.deepcopy(config.load("project"))
SRC = [f"Placeholder request {i} about gardening, part {chr(97 + i)}" for i in range(16)]
TOO_LONG = "Placeholder text about gardening. " * 150          # 5,100 characters
NLLB_REPO = CFG["back_translation"]["engine"]


class Sleeps(list):
    """A fake time.sleep that records the delays."""

    def __call__(self, seconds):
        self.append(seconds)


class Scripted:
    """A backend with fixed answers per source: a string, None, or ``RAISE``; else fake_translation."""

    RAISE = object()

    def __init__(self, table):
        self.table = dict(table)
        self.calls = []

    def __call__(self, text, lang):
        self.calls.append(text)
        value = self.table.get(text, fake_translation)
        if value is Scripted.RAISE:
            raise ConnectionError("fake transient failure")
        return value(text, lang) if callable(value) else value


class Failing:
    """A backend that raises ``make(text)`` for the source ``fail`` (its first ``times`` calls, or
    every call) and answers fake_translation otherwise. ``raised`` keeps each error it raised."""

    def __init__(self, make, fail, times=None):
        self.make, self.fail, self.left = make, fail, times
        self.calls, self.raised = [], []

    def __call__(self, text, lang):
        self.calls.append(text)
        if text == self.fail and (self.left is None or self.left > 0):
            if self.left is not None:
                self.left -= 1
            self.raised.append(self.make(text))
            raise self.raised[-1]
        return fake_translation(text, lang)


def run(texts, lang, root, backend, **kw):
    kw.setdefault("lid", FakeLID())
    kw.setdefault("sleep", Sleeps())
    kw.setdefault("cfg", CFG)
    return translate_texts(texts, lang, cache_dir=root, backend=backend, **kw)


def english_records(n=4):
    return [PromptRecord(pid=f"orb-{i:04d}", lang="en", set="hard_benign", split="dev", text=SRC[i],
                         text_en=SRC[i], source="orbench-hard-1k", category="privacy",
                         extra={"hf_row": 100 + i}) for i in range(n)]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Default paths (the cache's interim/translations, back_translate's outputs/backtranslations)
    point into a tmp dir, never the repo."""
    path = tmp_path / "data"
    monkeypatch.setenv("LIFT_DATA_DIR", str(path))
    monkeypatch.setenv("LIFT_OUTPUT_DIR", str(tmp_path / "outputs"))
    return path


@pytest.fixture
def root(tmp_path):
    return tmp_path / "cache"


# --- the cache ----------------------------------------------------------------------------------

def test_cache_key_file_layout_and_rows(root, data_dir):
    cache = TranslationCache(root)
    row = cache.put("google", "hi", SRC[0], "पहला अनुवाद")
    assert row["key"] == sha256_obj(["google", "hi", sha256_text(SRC[0])])
    assert row["key"] == cache.key("google", "hi", SRC[0])
    assert set(row) == {"key", "engine", "lang", "source_sha", "source", "output", "date"}
    assert (row["engine"], row["lang"], row["source"]) == ("google", "hi", SRC[0])
    assert row["source_sha"] == sha256_text(SRC[0])
    dt.date.fromisoformat(row["date"])
    path = root / "google" / "hi.jsonl"
    assert path.is_file() and list(root.rglob("*.jsonl")) == [path]

    cache.put("google", "hi", SRC[0], "दूसरा अनुवाद")                 # a later row for the same key wins
    fresh = TranslationCache(root)
    assert fresh.get("google", "hi", SRC[0])["output"] == "दूसरा अनुवाद"
    assert fresh.get("google", "zh", SRC[0]) is None                  # the language is part of the key
    assert fresh.get(ENGINE_NAMES["nllb"], "hi", SRC[0]) is None      # and so is the engine
    assert fresh.get("google", "hi", SRC[1]) is None
    data = path.read_bytes()
    assert data.count(b"\n") == 2 and data.endswith(b"\n") and b"\r" not in data

    assert TranslationCache().root == data_dir / "interim" / "translations"
    with pytest.raises(ValueError):
        cache.put("google", "hi", SRC[1], " \n")                      # blank outputs are never cached
    with pytest.raises(ValueError):
        cache.path("..", "hi")
    with pytest.raises(ValueError):
        cache.path("google", "ar/en")


def test_truncated_last_line_is_ignored_and_the_item_requested_again(root):
    run(SRC[:3], "hi", root, FakeTranslator())
    path = root / "google" / "hi.jsonl"
    lines = path.read_bytes().split(b"\n")
    path.write_bytes(lines[0] + b"\n" + lines[1] + b"\n" + lines[2][:40])   # killed mid-row

    again = FakeTranslator()
    assert run(SRC[:4], "hi", root, again) == [fake_translation(s, "hi") for s in SRC[:4]]
    assert again.calls == [(SRC[2], "hi"), (SRC[3], "hi")]
    # the cut line was ended once, so the new rows sit on lines of their own: no blank lines, every
    # row reads back, and nothing is requested twice
    data = path.read_bytes()
    assert data.count(b"\n") == 5 and b"\n\n" not in data and data.endswith(b"\n")
    fresh = TranslationCache(root)
    assert all(fresh.get("google", "hi", s)["output"] == fake_translation(s, "hi") for s in SRC[:4])
    third = FakeTranslator()
    run(SRC[:4], "hi", root, third)
    assert third.calls == []


def test_a_malformed_complete_row_raises_with_its_line(root):
    TranslationCache(root).put("google", "es", SRC[0], "¡Hola jardín!")
    path = root / "google" / "es.jsonl"
    append_jsonl(path, [{"key": "0" * 64, "engine": "google", "lang": "es", "source_sha": "x",
                         "source": SRC[1], "output": "¡Otro!", "date": "2026-11-03"}])
    with pytest.raises(ValueError, match=r"es\.jsonl:2: key does not match"):
        TranslationCache(root).get("google", "es", SRC[0])


def test_a_second_call_is_served_from_the_cache(root):
    first = FakeTranslator()
    out = run(SRC[:5], "zh", root, first)
    assert len(first.calls) == 5 and all("翻译" in o for o in out)
    second = FakeTranslator()
    assert run(SRC[:5], "zh", root, second) == out
    assert second.calls == []


def test_a_repeated_source_is_requested_once(root):
    fake = FakeTranslator(empty_for=[SRC[0]])
    with pytest.raises(TranslationError) as ei:
        run([SRC[0], SRC[1], SRC[0], SRC[1]], "ar", root, fake)
    assert fake.calls == [(SRC[0], "ar"), (SRC[1], "ar")]
    assert [(f.index, f.rule) for f in ei.value.failures] == [(0, "empty"), (2, "empty")]
    assert ei.value.outputs[1] == ei.value.outputs[3] == fake_translation(SRC[1], "ar")


# --- the failure rules --------------------------------------------------------------------------

def test_validate_order_and_details():
    lid = FakeLID({"मराठी मजकूर": ("mar_Deva", 0.62)})
    assert validate("Hello there", None, "hi", lid) == ("empty", "returned None")
    assert validate("Hello there", " \n ", "hi", lid) == ("empty", "blank")
    assert validate("Hello there", "​", "hi", lid) == ("empty", "blank")
    assert validate("Hello  there", "HELLO there\n", "hi", lid) == ("identical", "")
    assert lid.calls == 0                                       # GlotLID only runs past those two
    assert validate("Hello there", "translated test text", "hi", lid) == ("wrong_lang", "top=eng_Latn 0.95")
    assert validate("Hello there", "मराठी मजकूर", "hi", lid) == ("wrong_lang", "top=mar_Deva 0.62")
    assert validate("Hello there", "नमस्ते वहाँ", "hi", lid) is None


def test_every_failure_rule_is_collected_into_one_error(root):
    srcs = SRC[:10]
    srcs[6] = TOO_LONG
    backend = Scripted({srcs[1]: "", srcs[2]: None, srcs[3]: srcs[3],
                        srcs[4]: "  " + srcs[4].upper().replace(" ", "  ") + "\n",
                        srcs[5]: fake_translation(srcs[5], "en"), srcs[7]: Scripted.RAISE, srcs[9]: " \n\t"})
    pids = [f"prh-test-{i:04d}" for i in range(10)]
    sleeps = Sleeps()
    with pytest.raises(TranslationError) as ei:
        run(srcs, "hi", root, backend, pids=pids, sleep=sleeps)
    e = ei.value
    got = [(f.index, f.pid, f.rule, f.detail) for f in e.failures]
    assert got == [
        (1, "prh-test-0001", "empty", "blank"),
        (2, "prh-test-0002", "empty", "returned None"),
        (3, "prh-test-0003", "identical", ""),
        (4, "prh-test-0004", "identical", ""),
        (5, "prh-test-0005", "wrong_lang", "top=eng_Latn 0.95"),
        (6, "prh-test-0006", "too_long", f"{len(TOO_LONG)} chars"),
        (7, "prh-test-0007", "error", "ConnectionError x5"),
        (9, "prh-test-0009", "empty", "blank"),
    ]
    assert all(f.lang == "hi" and f.engine == "google" for f in e.failures)
    assert [f.source_sha for f in e.failures] == [sha256_text(srcs[f.index]) for f in e.failures]
    # every item was tried (the ones after the failures too); the too-long one was never sent
    assert TOO_LONG not in backend.calls and srcs[8] in backend.calls
    assert backend.calls.count(srcs[7]) == 5 and len(sleeps) == 4
    # outputs: None for each failure, never an English fallback
    assert [i for i, o in enumerate(e.outputs) if o is not None] == [0, 8]
    assert e.outputs[0] == fake_translation(srcs[0], "hi")
    assert all(o != s for o, s in zip(e.outputs, srcs, strict=True))
    # the engine output is kept on the failure for the readers, but out of its repr
    by_index = {f.index: f for f in e.failures}
    assert by_index[3].output == srcs[3] and by_index[5].output == fake_translation(srcs[5], "en")
    assert by_index[2].output is None and by_index[7].output is None
    assert "Placeholder" not in repr(by_index[3]) and "output" not in repr(by_index[3])
    assert str(e) == "8 of 10 translations into hi (google) failed: empty 3, error 1, identical 2, " \
                     "too_long 1, wrong_lang 1"
    # what was cached: the non-blank outputs (the failing ones too), nothing else
    cache = TranslationCache(root)
    assert [i for i, s in enumerate(srcs) if cache.get("google", "hi", s)] == [0, 3, 4, 5, 8]


def test_retries_back_off_exponentially(root):
    fake = FakeTranslator(raise_times={SRC[0]: 3})
    sleeps = Sleeps()
    assert run([SRC[0]], "es", root, fake, sleep=sleeps, base_delay=5.0) == [fake_translation(SRC[0], "es")]
    assert len(fake.calls) == 4
    assert len(sleeps) == 3 and all(d <= s <= d * 1.1 for s, d in zip(sleeps, [5, 10, 20], strict=True))

    fake = FakeTranslator(always_raise=[SRC[1]])
    sleeps = Sleeps()
    with pytest.raises(TranslationError) as ei:
        run([SRC[1]], "es", root, fake, sleep=sleeps, base_delay=2.0, max_retries=4)
    assert len(fake.calls) == 5
    assert len(sleeps) == 4 and all(d <= s <= d * 1.1 for s, d in zip(sleeps, [2, 4, 8, 16], strict=True))
    assert ei.value.failures[0].detail == "ConnectionError x5"

    fake = FakeTranslator(always_raise=[SRC[2]])
    sleeps = Sleeps()
    with pytest.raises(TranslationError):
        run([SRC[2]], "es", root, fake, sleep=sleeps, max_retries=0)
    assert len(fake.calls) == 1 and sleeps == []


def test_ten_items_in_a_row_failing_every_retry_trip_the_circuit_breaker(root, caplog):
    fake = FakeTranslator(always_raise=SRC)
    caplog.set_level(logging.ERROR, logger="lift")
    with pytest.raises(TranslationBlocked) as ei:
        run(SRC[:12], "th", root, fake, max_retries=1)
    assert ei.value.done == 10 and ei.value.kind == "streak"
    assert len(fake.calls) == 20                                  # 10 items x 2 attempts, then stop
    assert ei.value.__context__ is None and "Placeholder" not in str(ei.value)
    # it says what happened: a streak of transient errors, not Google refusing the IP
    message = str(ei.value)
    assert message.startswith("10 items in a row failed every retry with transient (network or service) "
                              "errors (ConnectionError x2): an outage, not an IP block. Stopped after "
                              "item 9 of 12")
    assert "429" not in message and "rerun" in message
    assert caplog.messages[-1] == ("google -> th item 9: 10 items in a row failed every retry with "
                                   "transient errors; stopping")

    # one success in between resets the count: 9 + 9 errors end in a TranslationError
    texts = [f"Placeholder note {i} about gardening" for i in range(19)]
    fake = FakeTranslator(always_raise=texts[:9] + texts[10:])
    with pytest.raises(TranslationError) as ei:
        run(texts, "th", root, fake, max_retries=0)
    assert [f.index for f in ei.value.failures] == [i for i in range(19) if i != 9]

    # cache hits neither reset nor count: 5 errors, a cached item, 5 errors trip it
    TranslationCache(root).put("google", "th", texts[9], fake_translation(texts[9], "th"))
    fake = FakeTranslator(always_raise=texts)
    with pytest.raises(TranslationBlocked) as ei:
        run(texts[4:15], "th", root, fake, max_retries=0)
    assert ei.value.done == 11 and len(fake.calls) == 10


# --- which errors are retried -------------------------------------------------------------------

def _requests_error(name):
    def make(text):
        return getattr(pytest.importorskip("requests.exceptions"), name)("fake failure")
    return make


TEXT = object()                     # stands for the item's text in a deep-translator error's arguments


def _deep_translator_error(name, *args):
    def make(text):
        cls = getattr(pytest.importorskip("deep_translator.exceptions"), name)
        return cls(*[text if a is TEXT else a for a in args])
    return make


class BadArgumentOSError(OSError, ValueError):
    """Like requests' MissingSchema or InvalidURL: an OSError that is a ValueError, i.e. a bug."""


TRANSIENT = {
    "ConnectionError": lambda text: ConnectionError("fake: connection refused"),
    "TimeoutError": lambda text: TimeoutError("fake: timed out"),
    "OSError": lambda text: OSError(104, "fake: connection reset by peer"),
    "requests ConnectTimeout": _requests_error("ConnectTimeout"),
    "requests ReadTimeout": _requests_error("ReadTimeout"),
    "requests ConnectionError": _requests_error("ConnectionError"),
    "requests SSLError": _requests_error("SSLError"),
    "requests ChunkedEncodingError": _requests_error("ChunkedEncodingError"),
    "deep-translator RequestError": _deep_translator_error("RequestError"),
    "deep-translator ServerException": _deep_translator_error("ServerException", 503),
    "deep-translator TranslationNotFound": _deep_translator_error("TranslationNotFound", TEXT),
}
NOT_TRANSIENT = {
    "ImportError": lambda text: ImportError("fake: cannot import name 'GoogleTranslator'"),
    "ModuleNotFoundError": lambda text: ModuleNotFoundError("fake: No module named 'torch'", name="torch"),
    "ValueError": lambda text: ValueError("fake: no code for 'fr'"),
    "TypeError": lambda text: TypeError("fake: unexpected keyword argument"),
    "KeyError": lambda text: KeyError("fake"),
    "AttributeError": lambda text: AttributeError("fake: no attribute 'translate'"),
    "RuntimeError": lambda text: RuntimeError("fake: out of memory"),
    "OSError that is a ValueError": lambda text: BadArgumentOSError("fake: bad argument"),
    "requests MissingSchema": _requests_error("MissingSchema"),
    "requests InvalidURL": _requests_error("InvalidURL"),
    "requests StreamConsumedError": _requests_error("StreamConsumedError"),
}


@pytest.mark.parametrize("name", list(TRANSIENT))
def test_transient_errors_are_retried(root, caplog, name):
    make = TRANSIENT[name]
    assert is_transient(make(SRC[1]))                             # (skips if the package is missing)
    caplog.set_level(logging.DEBUG, logger="lift")
    backend, sleeps = Failing(make, SRC[1], times=2), Sleeps()
    assert run(SRC[:3], "hi", root, backend, sleep=sleeps) == [fake_translation(s, "hi") for s in SRC[:3]]
    assert backend.calls == [SRC[0], SRC[1], SRC[1], SRC[1], SRC[2]] and len(sleeps) == 2

    # every retry fails: an "error" failure naming the type, and the items after it are still tried
    backend, sleeps = Failing(make, SRC[4]), Sleeps()
    with pytest.raises(TranslationError) as ei:
        run(SRC[3:6], "hi", root, backend, sleep=sleeps)
    kind = type(backend.raised[0]).__name__
    assert [(f.index, f.rule, f.detail) for f in ei.value.failures] == [(1, "error", f"{kind} x5")]
    assert backend.calls == [SRC[3]] + [SRC[4]] * 5 + [SRC[5]] and len(sleeps) == 4
    assert not [m for m in caplog.messages if "Placeholder" in m or "gardening" in m]


@pytest.mark.parametrize("name", list(NOT_TRANSIENT))
def test_errors_that_are_not_transient_propagate_at_once(root, caplog, name):
    make = NOT_TRANSIENT[name]
    assert not is_transient(make(SRC[1]))
    caplog.set_level(logging.DEBUG, logger="lift")
    backend, sleeps = Failing(make, SRC[1]), Sleeps()
    with pytest.raises(Exception) as ei:
        run(SRC[:4], "hi", root, backend, sleep=sleeps)
    assert ei.value is backend.raised[0]                         # its own error, type and traceback
    assert sleeps == [] and backend.calls == [SRC[0], SRC[1]]   # no retry, nothing after it
    cache = TranslationCache(root)
    assert [bool(cache.get("google", "hi", s)) for s in SRC[:3]] == [True, False, False]
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [f"google -> hi item 1: {type(ei.value).__name__} is not a transient error; stopping "
                      "without a retry"]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "retry" in r.getMessage()]


@pytest.mark.parametrize(("name", "args"), [("NotValidLength", (TEXT, 0, 5000)),
                                            ("NotValidPayload", (TEXT,))])
def test_deep_translator_errors_holding_the_text_are_raised_without_it(root, name, args):
    make = _deep_translator_error(name, *args)
    assert SRC[1] in str(make(SRC[1]))                           # deep-translator's own message
    backend, sleeps = Failing(make, SRC[1]), Sleeps()
    with pytest.raises(RuntimeError) as ei:
        run(SRC[:3], "hi", root, backend, sleep=sleeps)
    e = ei.value
    assert type(e) is RuntimeError and str(e) == (f"google -> hi item 1: deep-translator raised {name}, "
                                                  "not a transient error (its message is withheld: it holds "
                                                  "the text)")
    assert e.__cause__ is None and e.__context__ is None
    printed = "".join(traceback.format_exception(e))
    assert "Placeholder" not in printed and "in __call__" in printed   # the backend's frame is kept
    assert sleeps == [] and backend.calls == [SRC[0], SRC[1]]


def test_is_transient_never_imports_deep_translator(monkeypatch):
    for name in ("deep_translator", "deep_translator.exceptions"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert not is_transient(RuntimeError("fake")) and is_transient(TimeoutError("fake"))
    assert "deep_translator" not in sys.modules and "deep_translator.exceptions" not in sys.modules


def test_translation_blocked_kinds():
    assert TranslationBlocked("fake").kind == "http_429"          # what a backend raises on HTTP 429
    blocked = TranslationBlocked("fake", done=3, kind="streak")
    assert (blocked.kind, blocked.done) == ("streak", 3)
    with pytest.raises(ValueError):
        TranslationBlocked("fake", kind="dns")


# --- cache_only and retranslate_failed ----------------------------------------------------------

def test_cache_only_never_calls_the_backend(root, monkeypatch):
    run(SRC[:2], "ar", root, FakeTranslator())

    def forbidden(text, lang):
        raise AssertionError("backend called")

    with pytest.raises(TranslationError) as ei:
        run(SRC, "ar", root, forbidden, cache_only=True)          # 14 uncached in a row: no breaker
    assert {(f.rule, f.detail) for f in ei.value.failures} == {("error", "not cached")}
    assert [f.index for f in ei.value.failures] == list(range(2, 16))
    assert ei.value.outputs[:2] == [fake_translation(s, "ar") for s in SRC[:2]]

    def no_google(*a, **kw):
        raise AssertionError("google_backend created")

    monkeypatch.setattr(tr, "google_backend", no_google)
    assert run(SRC[:2], "ar", root, None, cache_only=True) == ei.value.outputs[:2]


def test_retranslate_failed_requests_only_cached_failures(root):
    with pytest.raises(TranslationError):
        run(SRC[:3], "th", root, FakeTranslator(english_for=[SRC[1]]))
    plain = FakeTranslator()
    with pytest.raises(TranslationError) as ei:                   # without the flag the failure stays
        run(SRC[:3], "th", root, plain)
    assert plain.calls == [] and [(f.index, f.rule) for f in ei.value.failures] == [(1, "wrong_lang")]

    again = FakeTranslator()
    out = run(SRC[:3], "th", root, again, retranslate_failed=True)
    assert again.calls == [(SRC[1], "th")]
    assert out == [fake_translation(s, "th") for s in SRC[:3]]
    path = root / "google" / "th.jsonl"
    assert path.read_bytes().count(b"\n") == 4                    # appended, never rewritten
    assert TranslationCache(root).get("google", "th", SRC[1])["output"] == fake_translation(SRC[1], "th")


def test_argument_checks(root):
    fake = FakeTranslator()
    for target in ("en", "fr"):
        with pytest.raises(ValueError):
            run(SRC[:1], target, root, fake)
    with pytest.raises(ValueError):
        run(SRC[:1], "hi", root, fake, engine="deepl")
    with pytest.raises(ValueError):
        run(SRC[:2], "hi", root, fake, pids=["prh-test-0000"])
    with pytest.raises(ValueError):
        run(SRC[:1], "hi", root, fake, cache_only=True, retranslate_failed=True)
    with pytest.raises(ValueError):
        run([SRC[0], "  "], "hi", root, fake)
    assert fake.calls == []


# --- translate_records --------------------------------------------------------------------------

def test_translate_records_copies_fields_and_dates_from_the_cache(root):
    recs = english_records(4)
    recs[1] = recs[1].model_copy(update={"lid": LidInfo(label="eng_Latn", prob=0.99)})
    append_jsonl(root / "google" / "es.jsonl", [{
        "key": TranslationCache.key("google", "es", SRC[0]), "engine": "google", "lang": "es",
        "source_sha": sha256_text(SRC[0]), "source": SRC[0], "output": " ¡Jardinería de prueba!​\n",
        "date": "2026-01-02"}])
    fake = FakeTranslator(english_for=[SRC[2]])
    with pytest.raises(TranslationError) as ei:
        translate_records(recs, "es", cache_dir=root, lid=FakeLID(), backend=fake, cfg=CFG, sleep=Sleeps())
    e = ei.value
    assert [(f.pid, f.rule) for f in e.failures] == [("orb-0002", "wrong_lang")]
    assert [r.pid for r in e.records] == ["orb-0000", "orb-0001", "orb-0003"]
    assert (SRC[0], "es") not in fake.calls                       # served from the cache

    r0, r1, _ = e.records
    assert r0.text == "¡Jardinería de prueba!"                    # clean_text on the output
    assert r0.translation == TranslationInfo(engine="google", source_lang="en", date="2026-01-02")
    today = TranslationCache(root).get("google", "es", SRC[1])["date"]
    assert r1.translation.date == today and r1.lid is None
    for rec, src in zip(e.records, [recs[0], recs[1], recs[3]], strict=True):
        assert rec.lang == "es" and rec.text_en == src.text
        assert (rec.pid, rec.set, rec.split, rec.source, rec.category) == \
               (src.pid, src.set, src.split, src.source, src.category)
        assert rec.extra == src.extra and rec.extra is not src.extra

    assert translate_records(recs[:2], "es", cache_dir=root, lid=FakeLID(), backend=fake, cfg=CFG) == \
        e.records[:2]                                             # deterministic rerun from the cache


def test_translate_records_refuses_anything_but_english_sources(root):
    fake = FakeTranslator()
    recs = english_records(2)
    hindi = recs[1].model_copy(update={"lang": "hi"})
    translated = recs[1].model_copy(update={"translation": TranslationInfo(engine="google")})
    for bad in (hindi, translated):
        with pytest.raises(ValueError, match="orb-0001"):
            translate_records([recs[0], bad], "zh", cache_dir=root, lid=FakeLID(), backend=fake, cfg=CFG)
    with pytest.raises(ValueError, match="duplicate"):
        translate_records([recs[0], recs[0]], "zh", cache_dir=root, lid=FakeLID(), backend=fake, cfg=CFG)
    with pytest.raises(TypeError):
        translate_records(recs, "zh", cache_dir=root, lid=FakeLID(), backend=fake, cfg=CFG, pids=["a", "b"])
    assert fake.calls == []


def test_the_nllb_engine_has_its_own_name_and_cache_file(root):
    out = translate_records(english_records(2), "hi", engine="nllb", cache_dir=root, lid=FakeLID(),
                            backend=FakeTranslator(), cfg=CFG)
    assert {r.translation.engine for r in out} == {"nllb-200-distilled-600M"}
    assert (root / "nllb-200-distilled-600M" / "hi.jsonl").is_file() and not (root / "google").exists()
    backend = nllb_backend(cfg=CFG)                                # nothing loads until the first call
    with pytest.raises(ValueError):
        backend(SRC[0], "fr")


# --- the NLLB engine's dependencies, with fake torch and transformers ---------------------------

@pytest.fixture
def fake_nllb(monkeypatch):
    """Fake torch and transformers in sys.modules. from_pretrained records the load and raises
    ``state.load_error``, an OSError as transformers raises for a model it cannot find or fetch."""
    state = types.SimpleNamespace(loads=[], load_error=OSError("fake: not in the local cache"))
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    transformers = types.ModuleType("transformers")

    class Auto:
        @classmethod
        def from_pretrained(cls, repo, **kw):
            state.loads.append((cls.__name__, repo, kw.get("revision")))
            raise state.load_error

    transformers.AutoTokenizer = type("AutoTokenizer", (Auto,), {})
    transformers.AutoModelForSeq2SeqLM = type("AutoModelForSeq2SeqLM", (Auto,), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("DISABLE_SAFETENSORS_CONVERSION", "1")      # _load's setdefault stays in the test
    return state


def test_check_nllb_passes_when_torch_and_transformers_import(fake_nllb):
    assert check_nllb() is None and fake_nllb.loads == []          # nothing is loaded or downloaded


@pytest.mark.parametrize("missing", ["torch", "transformers"])
def test_check_nllb_names_the_missing_package(fake_nllb, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)               # its import raises ModuleNotFoundError
    with pytest.raises(ImportError, match="^--fallback-engine nllb needs torch and transformers") as ei:
        check_nllb()
    assert ei.value.name == missing and f"{missing} does not import (ModuleNotFoundError)" in str(ei.value)
    assert 'pip install -e ".[gpu]"' in str(ei.value) and isinstance(ei.value.__cause__, ModuleNotFoundError)


def test_check_nllb_turns_a_broken_torch_into_import_error(fake_nllb, monkeypatch):
    class BrokenTorch(importlib.abc.MetaPathFinder):              # a torch DLL that fails to load
        def find_spec(self, name, path=None, target=None):
            if name == "torch":
                raise OSError("[WinError 126] fake: a DLL failed to load")
            return None

    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(sys, "meta_path", [BrokenTorch(), *sys.meta_path])
    with pytest.raises(ImportError, match=r"torch does not import \(OSError\)") as ei:
        check_nllb()
    assert not is_transient(ei.value)                             # never retried as a network error


def test_a_missing_torch_stops_the_nllb_engine_at_once(root, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    sleeps = Sleeps()
    with pytest.raises(ImportError, match="^--fallback-engine nllb needs torch and transformers"):
        run(SRC[:3], "hi", root, nllb_backend(cfg=CFG), engine="nllb", sleep=sleeps)
    assert sleeps == [] and not root.exists()


def test_a_model_that_does_not_load_stops_the_nllb_engine_at_once(root, fake_nllb):
    sleeps = Sleeps()
    with pytest.raises(RuntimeError, match=f"^cannot load {NLLB_REPO} at revision ") as ei:
        run(SRC[:3], "hi", root, nllb_backend(cfg=CFG), engine="nllb", sleep=sleeps)
    assert ei.value.__cause__ is fake_nllb.load_error               # transformers' error, kept as the cause
    assert fake_nllb.loads == [("AutoTokenizer", NLLB_REPO, CFG["back_translation"]["revision"])]
    assert sleeps == [] and not root.exists()


def test_a_missing_deep_translator_stops_at_once(root, monkeypatch):
    for name in ("deep_translator", "deep_translator.google", "deep_translator.exceptions"):
        monkeypatch.setitem(sys.modules, name, None)
    made = []

    def on_first_call(text, lang):                               # build.py makes the backend in the call
        made.append(text)
        return google_backend(CFG)(text, lang)

    sleeps = Sleeps()
    with pytest.raises(ModuleNotFoundError, match="deep_translator"):
        run(SRC[:3], "hi", root, on_first_call, sleep=sleeps)
    assert made == [SRC[0]] and sleeps == []


def test_glotlid_is_loaded_once_and_only_when_needed(root, monkeypatch):
    made = []

    class CountingLID(FakeLID):
        def __init__(self, *, cfg=None):
            made.append(cfg)
            super().__init__()

    monkeypatch.setattr(lift.lang, "LangID", CountingLID)
    with pytest.raises(TranslationError):
        run(SRC[:3], "hi", root, FakeTranslator(empty_for=SRC[:3]), lid=None)
    assert made == []                                             # nothing reached the language check
    run(SRC[3:6], "hi", root, FakeTranslator(), lid=None)
    assert made == [CFG]


# --- google_backend with a fake deep_translator -------------------------------------------------

class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_dt(monkeypatch):
    """A fake deep_translator package in sys.modules: GoogleTranslator, and the exceptions module
    with 1.11.4's classes the code looks up by name. ``raise_for`` maps a text to the errors its
    next calls raise; a text in ``block`` raises TooManyRequests."""
    requests = pytest.importorskip("requests")
    try:
        import deep_translator.google as real_google
    except ImportError:
        real_google = None
    state = types.SimpleNamespace(made=[], calls=[], block=set(), raise_for={}, clock=FakeClock(),
                                  real_get=requests.get, real_google=real_google, requests=requests,
                                  real_google_requests=getattr(real_google, "requests", None))

    class TooManyRequests(Exception):
        pass

    class RequestError(Exception):
        pass

    class ServerException(Exception):
        def __init__(self, status_code, *args):
            super().__init__("API server error", *args)

    class BaseError(Exception):                     # as in 1.11.4: str(e) starts with the input text
        def __init__(self, val, message):
            self.val, self.message = val, message
            super().__init__()

        def __str__(self):
            return f"{self.val} --> {self.message}"

    class TranslationNotFound(BaseError):
        def __init__(self, val, message="No translation was found using the current translator."):
            super().__init__(val, message)

    class NotValidLength(BaseError):
        def __init__(self, val, min_chars, max_chars):
            super().__init__(val, f"Text length need to be between {min_chars} and {max_chars} characters")

    class GoogleTranslator:
        def __init__(self, source="auto", target="en", proxies=None, **kwargs):
            self.source, self.target = source, target
            state.made.append(self)

        def translate(self, text):
            state.calls.append((state.clock.now, self.target, text))
            if text in state.block:
                raise TooManyRequests()
            if state.raise_for.get(text):
                raise state.raise_for[text].pop(0)
            return fake_translation(text, {"zh-CN": "zh"}.get(self.target, self.target))

    pkg = types.ModuleType("deep_translator")
    pkg.__path__ = []
    google = types.ModuleType("deep_translator.google")
    google.GoogleTranslator, google.requests = GoogleTranslator, requests
    exceptions = types.ModuleType("deep_translator.exceptions")
    for cls in (TooManyRequests, RequestError, ServerException, BaseError, TranslationNotFound,
                NotValidLength):
        setattr(exceptions, cls.__name__, cls)
    pkg.google, pkg.exceptions = google, exceptions
    state.exceptions = exceptions
    for name, module in (("deep_translator", pkg), ("deep_translator.google", google),
                         ("deep_translator.exceptions", exceptions)):
        monkeypatch.setitem(sys.modules, name, module)
    state.google = google
    return state


def test_google_backend_patches_the_timeout_into_deep_translator_only(fake_dt):
    google_backend(CFG, sleep=fake_dt.clock.sleep, clock=fake_dt.clock)
    get = fake_dt.google.requests.get
    assert isinstance(get, functools.partial) and get.func is fake_dt.real_get
    assert get.keywords == {"timeout": (10, 30)} and get.args == ()
    assert fake_dt.requests.get is fake_dt.real_get                # the global requests.get is untouched
    if fake_dt.real_google is not None:                            # and so is the real deep_translator
        assert fake_dt.real_google.requests is fake_dt.real_google_requests
    google_backend(CFG, timeout=(3, 7), sleep=fake_dt.clock.sleep, clock=fake_dt.clock)
    assert fake_dt.google.requests.get.keywords == {"timeout": (3, 7)}


def test_google_backend_maps_codes_and_reuses_one_translator_per_language(fake_dt):
    backend = google_backend(CFG, jitter=0.0, sleep=fake_dt.clock.sleep, clock=fake_dt.clock)
    assert backend(SRC[0], "zh") == fake_translation(SRC[0], "zh")
    backend(SRC[1], "zh")
    backend(SRC[2], "hi")
    assert [(t.source, t.target) for t in fake_dt.made] == [("en", "zh-CN"), ("en", "hi")]
    with pytest.raises(ValueError):
        backend(SRC[3], "fr")
    assert len(fake_dt.made) == 2 and len(fake_dt.calls) == 3


def test_google_backend_spaces_request_starts(fake_dt):
    clock = fake_dt.clock
    backend = google_backend(CFG, min_interval=1.0, jitter=0.0, sleep=clock.sleep, clock=clock)
    for s in SRC[:3]:
        backend(s, "es")
    assert [t for t, _, _ in fake_dt.calls] == [0.0, 1.0, 2.0] and clock.sleeps == [1.0, 1.0]
    clock.now += 5.0                                               # time already passed: no wait
    backend(SRC[3], "es")
    assert clock.sleeps == [1.0, 1.0]

    fake_dt.calls.clear()
    backend = google_backend(CFG, min_interval=1.0, jitter=0.5, sleep=clock.sleep, clock=clock)
    for s in SRC[:6]:
        backend(s, "es")
    starts = [t for t, _, _ in fake_dt.calls]
    assert all(1.0 <= b - a <= 1.5 for a, b in zip(starts, starts[1:], strict=False))


def test_google_backend_turns_429_into_translation_blocked_without_retrying(fake_dt, root):
    clock = fake_dt.clock
    backend = google_backend(CFG, sleep=clock.sleep, clock=clock)
    fake_dt.block = {SRC[2]}
    retry_sleeps = Sleeps()
    with pytest.raises(TranslationBlocked) as ei:
        run(SRC[:5], "hi", root, backend, sleep=retry_sleeps)
    assert ei.value.done == 2 and ei.value.kind == "http_429" and "429" in str(ei.value)
    assert [text for _, _, text in fake_dt.calls] == SRC[:3]      # called once, nothing after it
    assert retry_sleeps == []
    cache = TranslationCache(root)
    assert [bool(cache.get("google", "hi", s)) for s in SRC[:3]] == [True, True, False]

    with pytest.raises(TranslationBlocked) as ei:
        backend(SRC[2], "hi")
    assert ei.value.kind == "http_429"


def test_google_backend_errors_a_retry_can_cure_are_retried(fake_dt, root, caplog):
    exc = fake_dt.exceptions
    backend = google_backend(CFG, jitter=0.0, sleep=fake_dt.clock.sleep, clock=fake_dt.clock)
    fake_dt.raise_for = {SRC[1]: [exc.RequestError(), exc.TranslationNotFound(SRC[1]),
                                  exc.ServerException(503)]}
    sleeps = Sleeps()
    caplog.set_level(logging.DEBUG, logger="lift")
    assert run(SRC[:3], "hi", root, backend, sleep=sleeps) == [fake_translation(s, "hi") for s in SRC[:3]]
    assert [text for _, _, text in fake_dt.calls] == [SRC[0]] + [SRC[1]] * 4 + [SRC[2]] and len(sleeps) == 3
    retries = [m for m in caplog.messages if "retry" in m]
    assert [m.split(": ")[1].split(";")[0] for m in retries] == ["RequestError", "TranslationNotFound",
                                                                  "ServerException"]

    # one that holds the text and no retry cures: raised at once, without its message
    fake_dt.raise_for = {SRC[4]: [exc.NotValidLength(SRC[4], 0, 5000)]}
    with pytest.raises(RuntimeError, match="deep-translator raised NotValidLength, not a transient") as ei:
        run(SRC[3:6], "hi", root, backend, sleep=sleeps)
    assert "Placeholder" not in str(ei.value) and len(sleeps) == 3
    assert [text for _, _, text in fake_dt.calls][-2:] == [SRC[3], SRC[4]]
    assert not [m for m in caplog.messages if "Placeholder" in m]


# --- logging hygiene ----------------------------------------------------------------------------

def test_no_log_line_or_message_carries_text(root, caplog):
    class LeakyError(ConnectionError):                             # transient, so retried; its str(e)
        def __str__(self):                                         # starts with the input text, like
            return f"{self.args[0]} --> No translation was found"  # deep-translator's

    def leaky(text, lang):
        if text == SRC[3]:
            raise LeakyError(text)
        return fake.__call__(text, lang)

    fake = FakeTranslator(empty_for=[SRC[0]], identical_for=[SRC[1]], english_for=[SRC[2]],
                          raise_times={SRC[4]: 2})
    caplog.set_level(logging.DEBUG, logger="lift")
    with pytest.raises(TranslationError) as ei:
        run(SRC[:8], "hi", root, leaky, pids=[f"orb-{i:04d}" for i in range(8)], max_retries=2,
            progress_every=2)
    path = root / "google" / "hi.jsonl"
    path.write_bytes(path.read_bytes() + b'{"key": "trunc')       # exercise the cache-load logs too
    with pytest.raises(TranslationError):
        run(SRC[:8], "hi", root, fake, cache_only=True)
    with pytest.raises(BackTranslationError) as eb:
        back_translate([fake_translation(SRC[5], "ar")], "ar", cfg=CFG, cache_dir=root,
                       translator=lambda texts, lang: list(texts))

    messages = [r.getMessage() for r in caplog.records]
    assert any("retry" in m for m in messages) and any("failed" in m for m in messages)
    assert any("orb-0003" in m for m in messages)                 # ids are fine
    assert all(r.exc_info is None for r in caplog.records)
    texts = SRC[:8] + [fake_translation(s, lang) for s in SRC[:8] for lang in ("hi", "en", "ar")]
    blob = "\n".join(messages + [str(ei.value), repr(ei.value.failures), str(eb.value),
                                 repr(eb.value.failures)])
    assert "gardening" not in blob and "Placeholder" not in blob
    assert not [t for t in texts if t in blob]


# --- back-translation ---------------------------------------------------------------------------

def test_back_translate_caches_under_outputs_by_default(tmp_path, data_dir):
    """Its inputs are generated outputs (raw completions): outputs/, never the data tree."""
    src = [fake_translation(SRC[0], "ar")]
    answer = [f"English for {sha256_text(src[0])[:6]}"]
    assert back_translate(src, "ar", translator=lambda texts, lang: list(answer), cfg=CFG) == answer
    outputs = tmp_path.resolve() / "outputs"
    assert output_dir() == outputs                                # LIFT_OUTPUT_DIR (the autouse fixture)
    path = outputs / "backtranslations" / "nllb-200-distilled-600M" / "ar-en.jsonl"
    assert path.is_file() and [p.resolve() for p in tmp_path.rglob("*.jsonl")] == [path]
    assert not data_dir.exists()

    def forbidden(texts, lang):
        raise AssertionError("translator called")

    assert back_translate(src, "ar", translator=forbidden, cfg=CFG) == answer     # served from there
    assert back_translate(src, "ar", translator=lambda texts, lang: ["Another"], cfg=CFG,
                          cache_dir=tmp_path / "elsewhere") == ["Another"]           # cache_dir overrides
    assert (tmp_path / "elsewhere" / "nllb-200-distilled-600M" / "ar-en.jsonl").is_file()


def test_back_translate_raises_on_empty_or_identical_and_caches(root):
    src = [fake_translation(s, "ar") for s in SRC[:3]]
    answers = {src[0]: "An English rendering", src[1]: "", src[2]: src[2]}
    calls = []

    def translator(texts, lang):
        calls.append((list(texts), lang))
        return [answers.get(t, f"English for {sha256_text(t)[:6]}") for t in texts]

    with pytest.raises(BackTranslationError) as ei:
        back_translate([src[0], src[1], src[2], src[0]], "ar", translator=translator, cfg=CFG,
                       cache_dir=root)
    e = ei.value
    assert calls == [([src[0], src[1], src[2]], "ar")]            # distinct texts, one call
    assert [(f.index, f.rule) for f in e.failures] == [(1, "empty"), (2, "identical")]
    assert {(f.lang, f.engine) for f in e.failures} == {("ar-en", "nllb-200-distilled-600M")}
    assert e.outputs == ["An English rendering", None, None, "An English rendering"]
    assert "1 empty" in str(e) and "2 identical" in str(e)
    path = root / "nllb-200-distilled-600M" / "ar-en.jsonl"
    assert path.is_file() and path.read_bytes().count(b"\n") == 2  # src[0] and src[2]; blank not cached

    calls.clear()
    answers[src[1]] = "Another English rendering"
    with pytest.raises(BackTranslationError) as ei:
        back_translate(src, "ar", translator=translator, cfg=CFG, cache_dir=root)
    assert calls == [([src[1]], "ar")]                            # only the uncached one
    assert [(f.index, f.rule) for f in ei.value.failures] == [(2, "identical")]

    calls.clear()
    out = back_translate(src[:2], "ar", translator=translator, cfg=CFG, cache_dir=root)
    assert out == ["An English rendering", "Another English rendering"] and calls == []

    th = [fake_translation(SRC[0], "th")]
    assert back_translate(th, "th", translator=translator, cfg=CFG, cache_dir=root) == \
        [f"English for {sha256_text(th[0])[:6]}"]
    assert (root / "nllb-200-distilled-600M" / "th-en.jsonl").is_file()
    with pytest.raises(ValueError):
        back_translate(th, "th", translator=lambda texts, lang: [], cfg=CFG, cache_dir=root / "x")
    for lang in ("en", "fr"):
        with pytest.raises(ValueError):
            back_translate(th, lang, translator=translator, cfg=CFG, cache_dir=root)


# --- imports ------------------------------------------------------------------------------------

def test_importing_the_module_stays_light():
    code = ("import sys, lift.data.translate; heavy = {'deep_translator', 'requests', 'transformers', "
            "'torch', 'fasttext', 'datasets', 'huggingface_hub'} & set(sys.modules); print(sorted(heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_failure_model_is_frozen_and_strict():
    f = TranslationFailure(index=0, lang="hi", engine="google", source_sha="0" * 64, rule="empty")
    with pytest.raises(ValueError):
        TranslationFailure(index=0, lang="hi", engine="google", source_sha="0", rule="vanished")
    with pytest.raises(ValueError):
        TranslationFailure(index=0, lang="hi", engine="google", source_sha="0", rule="empty", text="x")
    with pytest.raises(ValueError):
        f.index = 1


# --- the real NLLB (downloads 2.5 GB) -----------------------------------------------------------

@pytest.mark.net
def test_real_nllb_back_and_forward(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    out = back_translate(["هذه جملة اختبار عن البستنة."], "ar", cfg=CFG, cache_dir=tmp_path)
    assert out[0].isascii() and "garden" in out[0].lower()
    hindi = translate_texts(["This is a short test sentence about gardening."], "hi", engine="nllb",
                            cache_dir=tmp_path, lid=FakeLID(), cfg=CFG)
    assert FakeLID().is_lang(hindi[0], "hi")
