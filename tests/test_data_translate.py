"""lift.data.translate, offline: FakeTranslator/FakeLID backends and a fake deep_translator module.
Nothing here calls Google; the only network test is the real NLLB one (marker ``net``)."""

import copy
import datetime as dt
import functools
import logging
import subprocess
import sys
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
    google_backend,
    nllb_backend,
    translate_records,
    translate_texts,
    validate,
)
from lift.io import append_jsonl, sha256_obj, sha256_text
from lift.schemas import LidInfo, PromptRecord, TranslationInfo

CFG = copy.deepcopy(config.load("project"))
SRC = [f"Placeholder request {i} about gardening, part {chr(97 + i)}" for i in range(16)]
TOO_LONG = "Placeholder text about gardening. " * 150          # 5,100 characters


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
    """Default paths (the cache's interim/translations) point into a tmp dir, never the repo."""
    path = tmp_path / "data"
    monkeypatch.setenv("LIFT_DATA_DIR", str(path))
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


def test_ten_items_in_a_row_failing_every_retry_trip_the_circuit_breaker(root):
    fake = FakeTranslator(always_raise=SRC)
    with pytest.raises(TranslationBlocked) as ei:
        run(SRC[:12], "th", root, fake, max_retries=1)
    assert ei.value.done == 10
    assert len(fake.calls) == 20                                  # 10 items x 2 attempts, then stop
    assert ei.value.__context__ is None and "Placeholder" not in str(ei.value)

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
    """A fake deep_translator package in sys.modules: GoogleTranslator, exceptions.TooManyRequests."""
    requests = pytest.importorskip("requests")
    try:
        import deep_translator.google as real_google
    except ImportError:
        real_google = None
    state = types.SimpleNamespace(made=[], calls=[], block=set(), clock=FakeClock(),
                                  real_get=requests.get, real_google=real_google, requests=requests,
                                  real_google_requests=getattr(real_google, "requests", None))

    class TooManyRequests(Exception):
        pass

    class GoogleTranslator:
        def __init__(self, source="auto", target="en", proxies=None, **kwargs):
            self.source, self.target = source, target
            state.made.append(self)

        def translate(self, text):
            state.calls.append((state.clock.now, self.target, text))
            if text in state.block:
                raise TooManyRequests()
            return fake_translation(text, {"zh-CN": "zh"}.get(self.target, self.target))

    pkg = types.ModuleType("deep_translator")
    pkg.__path__ = []
    google = types.ModuleType("deep_translator.google")
    google.GoogleTranslator, google.requests = GoogleTranslator, requests
    exceptions = types.ModuleType("deep_translator.exceptions")
    exceptions.TooManyRequests = TooManyRequests
    pkg.google, pkg.exceptions = google, exceptions
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
    assert ei.value.done == 2 and "429" in str(ei.value)
    assert [text for _, _, text in fake_dt.calls] == SRC[:3]      # called once, nothing after it
    assert retry_sleeps == []
    cache = TranslationCache(root)
    assert [bool(cache.get("google", "hi", s)) for s in SRC[:3]] == [True, True, False]

    with pytest.raises(TranslationBlocked):
        backend(SRC[2], "hi")


# --- logging hygiene ----------------------------------------------------------------------------

def test_no_log_line_or_message_carries_text(root, caplog):
    class LeakyError(Exception):                                   # deep-translator's str(e) starts
        def __str__(self):                                         # with the input text
            return f"{self.args[0]} --> No translation was found"

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
