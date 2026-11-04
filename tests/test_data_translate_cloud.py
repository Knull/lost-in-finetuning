"""The Cloud Translation API engine (``google_cloud``, decisions item 48), offline: a fake ``post`` stands
in for requests. All text is fake and benign."""

from __future__ import annotations

import json
import logging

import pytest
from datafakes import FakeLID, FakeTranslator, fake_translation
from fake_build import fixture_cfg, fixture_ctx, isolate, repo_state

from lift import config
from lift.cli import main
from lift.data import build, translate
from lift.data.translate import CloudTranslateUnavailable, TranslationSetupError
from lift.io import read_jsonl

KEY_ENV = "GOOGLE_TRANSLATE_API_KEY"
TEXT = "Describe a quiet morning in the park."


@pytest.fixture(autouse=True, scope="module")
def repo_untouched():
    before = repo_state()
    yield
    assert repo_state() == before, "a test wrote into the repo (data/, outputs/manifests, validation/sheets)"


class FakeResponse:
    def __init__(self, status: int, body: object = None) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> object:
        if self._body is None:
            raise ValueError("no JSON body")              # as requests' JSONDecodeError is a ValueError
        return self._body


def ok(text: str) -> FakeResponse:
    return FakeResponse(200, {"data": {"translations": [{"translatedText": text, "model": "nmt"}]}})


def refusal(status: int, reason: str, message: str) -> FakeResponse:
    return FakeResponse(status, {"error": {"code": status, "message": message, "status": "PERMISSION_DENIED",
                                           "errors": [{"message": message, "domain": "global",
                                                       "reason": reason}]}})


class FakePost:
    """Answers each call with the next queued response (the last one repeats); records the calls."""

    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, **kw):
        self.calls.append({"url": url, **kw})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def backend(post: FakePost, *, clock: FakeClock | None = None, **kw):
    clock = clock or FakeClock()
    return translate.google_cloud_backend(config.load("project"), api_key=kw.pop("api_key", "k3y"), post=post,
                                          sleep=clock.sleep, clock=clock, **kw)


# ---------------------------------------------------------------------------- config


def test_the_committed_config_translates_through_the_cloud_api():
    cfg = config.load("project")
    assert cfg["translation"] == {"engine": "google_cloud", "api_key_env": KEY_ENV}
    assert translate.primary_engine(cfg) == "google_cloud" and translate.cloud_key_env(cfg) == KEY_ENV
    assert translate.ENGINE_NAMES["google_cloud"] == "google-cloud-v2"       # what the records say
    assert translate.ENGINE_NAMES["google"] == "google"                       # never the same name


def test_primary_engine_defaults_to_google_and_refuses_anything_else():
    assert translate.primary_engine({}) == "google"
    assert translate.primary_engine({"translation": None}) == "google"
    assert translate.cloud_key_env({}) == KEY_ENV
    with pytest.raises(ValueError, match="translation.engine must be one of"):
        translate.primary_engine({"translation": {"engine": "nllb"}})


# ---------------------------------------------------------------------------- the backend


def test_one_post_per_text_with_the_key_in_a_header_and_plain_text_format():
    post = FakePost(ok("翻译后的文本"))
    assert backend(post)(TEXT, "zh") == "翻译后的文本"
    (call,) = post.calls
    assert call["url"] == translate.CLOUD_TRANSLATE_URL and "k3y" not in call["url"]
    assert call["json"] == {"q": [TEXT], "source": "en", "target": "zh-CN", "format": "text", "model": "nmt"}
    assert call["headers"] == {"X-Goog-Api-Key": "k3y"} and call["timeout"] == (10, 30)
    with pytest.raises(ValueError, match="no Google Translate code for 'en'"):
        backend(post)(TEXT, "en")


def test_the_key_comes_from_the_configured_variable_and_a_missing_one_stops_at_once(monkeypatch):
    post = FakePost(ok("अनुवाद"))
    monkeypatch.setenv(KEY_ENV, "  from-env \n")
    translate.google_cloud_backend(config.load("project"), post=post)(TEXT, "hi")
    assert post.calls[0]["headers"] == {"X-Goog-Api-Key": "from-env"}
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(TranslationSetupError, match=f"{KEY_ENV} is not set"):
        translate.google_cloud_backend(config.load("project"), post=post)
    cfg = {**config.load("project"), "translation": {"engine": "google_cloud", "api_key_env": "OTHER_KEY"}}
    monkeypatch.setenv(KEY_ENV, "not this one")
    with pytest.raises(TranslationSetupError, match="OTHER_KEY is not set"):
        translate.google_cloud_backend(cfg, post=post)
    with pytest.raises(TranslationSetupError, match="is not set"):
        translate.google_cloud_backend(cfg, api_key="   ", post=post)


@pytest.mark.parametrize("response", [
    FakeResponse(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}),
    FakeResponse(500), FakeResponse(503, {"error": {"code": 503, "message": "Backend Error"}}),
    refusal(403, "userRateLimitExceeded", "User Rate Limit Exceeded"),
    refusal(403, "rateLimitExceeded", "Rate Limit Exceeded"),
    FakeResponse(200, {"data": {}}), FakeResponse(200, {"data": {"translations": []}}), FakeResponse(200),
])
def test_answers_a_retry_can_cure_are_transient(response):
    with pytest.raises(CloudTranslateUnavailable) as e:
        backend(FakePost(response))(TEXT, "hi")
    assert translate.is_transient(e.value) and TEXT not in str(e.value)


@pytest.mark.parametrize(("response", "shown"), [
    (refusal(403, "accessNotConfigured", "Cloud Translation API has not been used in project 123 before "
                                         "or it is disabled."),
     "HTTP 403 (accessNotConfigured): Cloud Translation"),
    (refusal(400, "badRequest", "API key not valid. Please pass a valid API key."),
     "HTTP 400 (badRequest): API key not valid"),
    (refusal(403, "dailyLimitExceeded", "Daily Limit Exceeded"), "HTTP 403 (dailyLimitExceeded)"),
    (FakeResponse(403, {"error": {"code": 403, "message": "Requests to this API are blocked.",
                                  "status": "PERMISSION_DENIED"}}), "HTTP 403 (PERMISSION_DENIED)"),
    (FakeResponse(401), "HTTP 401 (no reason given): no message"),
])
def test_other_refusals_are_setup_errors_that_never_hold_the_text(response, shown):
    with pytest.raises(TranslationSetupError) as e:
        backend(FakePost(response))(TEXT, "hi")
    assert shown in str(e.value) and "billing" in str(e.value)
    assert not translate.is_transient(e.value) and TEXT not in str(e.value)


def test_a_200_without_a_string_is_an_empty_output():
    post = FakePost(FakeResponse(200, {"data": {"translations": [{"translatedText": None}]}}))
    assert backend(post)(TEXT, "hi") is None
    assert translate.validate(TEXT, None, "hi", FakeLID()) == ("empty", "returned None")


def test_request_starts_are_spaced_by_min_interval():
    clock = FakeClock()
    call = backend(FakePost(ok("एक"), ok("दो"), ok("तीन")), clock=clock, min_interval=0.25)
    call(TEXT, "hi")
    clock.now += 0.1
    call(TEXT + " again", "hi")
    call(TEXT + " once more", "hi")
    assert clock.slept == [pytest.approx(0.15), pytest.approx(0.25)]


# ---------------------------------------------------------------------------- through translate_texts


def test_a_rate_limit_is_retried_and_the_output_cached_under_the_cloud_engine(tmp_path):
    hindi = fake_translation(TEXT, "hi")
    post = FakePost(FakeResponse(429), FakeResponse(503), ok(hindi))
    out = translate.translate_texts([TEXT], "hi", engine="google_cloud", backend=backend(post),
                                    cache_dir=tmp_path, lid=FakeLID(), sleep=lambda s: None)
    assert out == [hindi] and len(post.calls) == 3
    (row,) = read_jsonl(tmp_path / "google-cloud-v2" / "hi.jsonl")
    assert row["engine"] == "google-cloud-v2" and row["output"] == hindi
    assert not (tmp_path / "google").exists()
    again = FakePost(ok("never asked"))
    assert translate.translate_texts([TEXT], "hi", engine="google_cloud", backend=backend(again),
                                     cache_dir=tmp_path, lid=FakeLID()) == [hindi]
    assert again.calls == []                                   # served from the cloud engine's cache


def test_an_item_that_never_gets_through_names_the_statuses_it_met(tmp_path):
    post = FakePost(FakeResponse(429), FakeResponse(429), FakeResponse(503))
    with pytest.raises(translate.TranslationError) as e:
        translate.translate_texts([TEXT], "hi", engine="google_cloud", backend=backend(post),
                                  cache_dir=tmp_path, lid=FakeLID(), sleep=lambda s: None, max_retries=2)
    (fail,) = e.value.failures
    assert fail.rule == "error" and fail.engine == "google-cloud-v2"
    assert fail.detail == "CloudTranslateUnavailable (HTTP 429) x2, CloudTranslateUnavailable (HTTP 503) x1"
    assert TEXT not in fail.detail


def test_a_setup_error_stops_the_run_at_once_without_a_retry(tmp_path, caplog):
    post = FakePost(refusal(403, "accessNotConfigured", "Cloud Translation API is disabled."))
    caplog.set_level(logging.ERROR)
    with pytest.raises(TranslationSetupError, match="accessNotConfigured"):
        translate.translate_texts([TEXT, TEXT + " two"], "hi", engine="google_cloud", backend=backend(post),
                                  cache_dir=tmp_path, lid=FakeLID(), sleep=lambda s: pytest.fail("retried"))
    assert len(post.calls) == 1 and not any(tmp_path.rglob("*.jsonl"))
    assert "not a transient error" in caplog.text and TEXT not in caplog.text


def test_translate_records_labels_the_records_with_the_cloud_engine(tmp_path):
    from lift.schemas import PromptRecord

    rec = PromptRecord(pid="orb-0001", set="hard_benign", lang="en", text=TEXT, split="dev", source="orbench")
    post = FakePost(ok(fake_translation(TEXT, "es")))
    (out,) = translate.translate_records([rec], "es", engine="google_cloud", cache_dir=tmp_path,
                                         lid=FakeLID(), backend=backend(post))
    assert out.translation.engine == "google-cloud-v2" and out.lang == "es" and out.text_en == TEXT


# ---------------------------------------------------------------------------- the build and the CLI


def cloud_cfg() -> dict:
    cfg = fixture_cfg()
    cfg["translation"] = {**cfg["translation"], "engine": "google_cloud"}
    return cfg


def test_a_build_on_the_cloud_engine_names_it_in_records_cache_reports_and_build_json(tmp_path, monkeypatch):
    dirs = isolate(monkeypatch, tmp_path)
    translator = FakeTranslator()
    build.build_all(ctx=fixture_ctx(cfg=cloud_cfg(), translator=translator), only=["hard_benign"],
                    langs=["en", "hi"])
    records = list(read_jsonl(dirs.data / "prompts" / "hard_benign_hi.jsonl"))
    assert records and {r["translation"]["engine"] for r in records} == {"google-cloud-v2"}
    assert (dirs.data / "interim" / "translations" / "google-cloud-v2" / "hi.jsonl").exists()
    assert not (dirs.data / "interim" / "translations" / "google").exists()
    reports = dirs.data / "prompts" / "reports"
    report = json.loads((reports / "hard_benign_hi.json").read_text(encoding="utf-8"))
    assert report["engines"] == {"google-cloud-v2": len(records)}
    summary = json.loads((reports / "build.json").read_text(encoding="utf-8"))
    assert summary["sources"]["translation"] == {"engine": "google-cloud-v2"}
    assert translator.calls                                        # the context's one translator served it


def test_lift_data_build_reports_a_missing_key_in_one_line_and_exits_1(tmp_path, monkeypatch, caplog, capsys):
    isolate(monkeypatch, tmp_path)
    monkeypatch.delenv(KEY_ENV, raising=False)
    def default(cls, **kw):
        return fixture_ctx(**{"cfg": cloud_cfg(), "translator": None, **kw})

    monkeypatch.setattr(build.BuildContext, "default", classmethod(default))
    caplog.set_level(logging.ERROR)
    assert main(["data", "build", "--only", "hard_benign", "--langs", "en", "hi"]) == 1
    assert f"translation cannot run: {KEY_ENV} is not set" in caplog.text
    assert "Traceback" not in caplog.text and capsys.readouterr().out == ""
