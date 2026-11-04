"""The frontier safety judge: rubric, strict parser, retries, preflight, cache, spend cap, providers.

Offline only. The judge logic runs on FakeProvider; the OpenAI and Anthropic providers run on real SDK
clients built with dummy keys whose create() is monkeypatched. No test reads a real key, and the base
URLs point at a closed local port in case anything ever tried to send a request.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import logging
import math
import subprocess
import sys
import threading

import pytest
import yaml

from lift import config
from lift.judge import frontier
from lift.judge.frontier import (
    LABELS,
    REMINDER,
    SDK_LOGGERS,
    AnthropicProvider,
    FakeProvider,
    FrontierConfigError,
    FrontierJudge,
    OpenAIProvider,
    ProviderError,
    ProviderResult,
    SpendCapExceeded,
    load_rubric,
    parse_safety_reply,
)
from lift.paths import REPO_ROOT, output_dir

RUBRIC = "configs/rubrics/safety_v1.md"
HAIKU = "claude-haiku-5-5"
PRICE = {"input": 1.0, "output": 1.0}
NO_SAMPLING = {"temperature", "top_p", "top_k", "fallbacks", "extra_body", "betas"}
# Anthropic's answer once usage reaches a spend limit set in the Console: HTTP 400 invalid_request_error
# (platform.claude.com/docs/en/api/rate-limits, "Setting your own spend limit"); and once the
# organisation's credits are exhausted (docs/en/about-claude/api-credits-for-subscribers; no status given).
SPEND_LIMIT = ("You have reached your specified API usage limits. You will regain access on 2026-12-01 at "
               "00:00 UTC.")
WORKSPACE_LIMIT = SPEND_LIMIT.replace("specified API", "specified workspace API")
CREDIT_LOW = ("Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to "
              "upgrade or purchase credits.")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No key, no spend cap, no reachable endpoint, no real sleeping. Returns the recorded sleeps."""
    for var in ("LIFT_JUDGE_MAX_USD", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "OPENAI_LOG", "ANTHROPIC_LOG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    sleeps: list[float] = []
    monkeypatch.setattr(frontier.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture
def judge_config(monkeypatch):
    """Replace judges.safety_second in the loaded project config by the committed one plus settings."""
    base = copy.deepcopy(config.load("project"))

    def set_entry(**settings):
        cfg = copy.deepcopy(base)
        cfg["judges"]["safety_second"].update(settings)
        monkeypatch.setattr(config, "load", lambda name="project": cfg)

    return set_entry


def make_judge(script=(), *, provider="anthropic", model=HAIKU, **kw):
    fake = script if isinstance(script, FakeProvider) else FakeProvider(script)
    kw.setdefault("workers", 1)
    if provider == "openai":
        kw.setdefault("price_usd_per_mtok", PRICE)
    return FrontierJudge(provider, model, RUBRIC, client=fake, **kw), fake


def items(n, lang="hi"):
    return [f"prompt-{i:02d}" for i in range(n)], [f"response-{i:02d}" for i in range(n)], [lang] * n


def item_calls(fake):
    return [c for c in fake.calls if not c["preflight"]]


def raiser(exc):
    def create(**kwargs):
        raise exc

    return create


# ------------------------------------------------------------------------------------- rubric --
def test_rubric_is_used_verbatim():
    text = (REPO_ROOT / RUBRIC).read_text(encoding="utf-8")
    system = text.split("## System message", 1)[1].split("## User message template", 1)[0].strip()
    template = text.split("## User message template", 1)[1].strip()
    rubric = load_rubric(RUBRIC)
    assert (rubric.system, rubric.user_template) == (system, template)
    assert rubric.system.startswith("You are a careful content-safety annotator")
    assert rubric.system.endswith('"reason": "<at most 20 English words, no harmful details>"}')
    assert rubric.version == f"safety_v1:{hashlib.sha256((REPO_ROOT / RUBRIC).read_bytes()).hexdigest()[:12]}"

    judge, fake = make_judge([FakeProvider.reply("REFUSAL")])
    judge.safety(["p"], ["r"], ["hi"])
    assert {c["system"] for c in fake.calls} == {system}
    expected = template.replace("{prompt_lang}", "Hindi").replace("{prompt}", "p").replace("{response}", "r")
    assert item_calls(fake)[0]["user"] == expected


def test_substitution_is_a_single_pass():
    rubric = load_rubric(RUBRIC)
    tricky = r"see {prompt} and {response} and {prompt_lang} \1 \g<0>"
    out = rubric.render(prompt="PROMPT-TEXT", response=tricky, prompt_lang="Thai")
    assert f"<<<\n{tricky}\n>>>" in out
    assert out.count("PROMPT-TEXT") == 1 and "PROMPT (Thai):" in out
    out = rubric.render(prompt="{response}", response="ANSWER-TEXT", prompt_lang="English")
    assert "<<<\n{response}\n>>>" in out and out.count("ANSWER-TEXT") == 1


@pytest.mark.parametrize("mutate", [
    lambda t: t.replace("## System message", "## System"),
    lambda t: t + "\n## User message template\n{prompt} {response} {prompt_lang}\n",
    lambda t: t.replace("{response}", "{answer}"),
    lambda t: t.replace("\n", "\r\n"),
])
def test_malformed_rubric_fails_loudly(tmp_path, mutate):
    path = tmp_path / "rubric.md"
    path.write_bytes(mutate((REPO_ROOT / RUBRIC).read_text(encoding="utf-8")).encode("utf-8"))
    with pytest.raises(ValueError):
        load_rubric(path)


def test_inputs_are_checked_before_any_call():
    judge, fake = make_judge()
    with pytest.raises(ValueError, match="unknown prompt language"):
        judge.safety(["p"], ["r"], ["Hindi"])
    with pytest.raises(ValueError, match="differ in length"):
        judge.safety(["p", "q"], ["r"], ["hi"])
    with pytest.raises(TypeError):
        judge.safety(["p"], [None], ["hi"])
    assert judge.safety([], [], []) == []
    assert fake.calls == []


# ------------------------------------------------------------------------------------- parser --
GOOD = '{"label": "HARMFUL", "reason": "gives working steps"}'


@pytest.mark.parametrize(("text", "expected"), [
    (GOOD, ("HARMFUL", "gives working steps")),
    ("  \n" + GOOD + "\n ", ("HARMFUL", "gives working steps")),
    ('{"reason": "", "label": "UNCLEAR"}', ("UNCLEAR", "")),
    ('{"label":"BENIGN","reason":"उत्तर ठीक है"}', ("BENIGN", "उत्तर ठीक है")),
])
def test_parser_accepts_exactly_one_object(text, expected):
    assert parse_safety_reply(text) == expected


@pytest.mark.parametrize("text", [
    "```json\n" + GOOD + "\n```",                                   # code fence
    '{"label": "HARMFUL", "reason": "x", "confidence": 0.9}',        # extra key
    '{"label": "harmful", "reason": "x"}',                           # lowercase label
    '{"label": "HARMFUL"}',                                          # missing reason
    '{"label": "HARMFUL", "reason": null}',                          # reason not a string
    GOOD + GOOD,                                                     # two objects
    GOOD + "\n" + GOOD,
    "Here is my answer: " + GOOD,                                    # prose before
    GOOD + " I hope this helps.",                                    # prose after
    '{"label": "PARTIAL", "reason": "x"}',                           # unknown label
    '{"label": "HARMFUL", "label": "BENIGN", "reason": "x"}',        # duplicate key
    "[" + GOOD + "]",
    '"HARMFUL"',
    "HARMFUL",
    "",
])
def test_parser_rejects_everything_else(text):
    assert parse_safety_reply(text) is None


# ------------------------------------------------------------------------ reminder and refusal --
def test_parse_failure_gets_one_reminder_then_succeeds():
    reply = FakeProvider.reply("HARMFUL", "gives steps")
    judge, fake = make_judge(["Sure! The label is HARMFUL.", reply])
    assert judge.safety(["p"], ["r"], ["en"]) == [
        {"label": "HARMFUL", "reason": "gives steps", "raw": reply, "error": None, "model_echo": HAIKU}]
    first, second = item_calls(fake)
    assert second["user"] == REMINDER
    assert second["history"] == [{"role": "user", "content": first["user"]},
                                 {"role": "assistant", "content": "Sure! The label is HARMFUL."}]
    assert (judge.stats["reminders"], judge.stats["failures"]) == (1, 0)


def test_second_parse_failure_is_a_judge_failure_not_a_guess():
    fenced = "```json\n" + FakeProvider.reply("BENIGN") + "\n```"
    judge, fake = make_judge(["HARMFUL", fenced])
    assert judge.safety(["p"], ["r"], ["en"]) == [{"label": None, "reason": None, "raw": fenced,
                                                   "error": "unparseable judge output", "model_echo": HAIKU}]
    assert len(item_calls(fake)) == 2
    assert judge.stats["failures"] == 1


def test_an_empty_reply_is_not_replayed_as_an_assistant_turn():
    judge, fake = make_judge(["  ", FakeProvider.reply("REFUSAL")])
    assert judge.safety(["p"], ["r"], ["en"])[0]["label"] == "REFUSAL"
    assert [t["role"] for t in item_calls(fake)[1]["history"]] == ["user"]


def test_a_provider_refusal_is_a_failure_with_its_reason():
    judge, fake = make_judge([FakeProvider.refusal("bio: policy"), "not json", FakeProvider.refusal(None)])
    first, second = judge.safety(["p", "q"], ["r", "s"], ["en", "zh"])
    assert first == {"label": None, "reason": None, "raw": "", "error": "refused: bio: policy",
                     "model_echo": HAIKU}                     # a refusal still names the model that refused
    assert second["error"] == "refused: no reason given"     # the reminder turn was refused
    assert len(item_calls(fake)) == 3                        # no reminder after a refusal
    assert (judge.stats["refusals"], judge.stats["failures"]) == (2, 2)


# -------------------------------------------------------------------------- retries and errors --
def test_retryable_errors_back_off_then_succeed(offline, monkeypatch):
    monkeypatch.setattr(frontier.random, "random", lambda: 1.0)
    script = [FakeProvider.error(429), FakeProvider.error(None), FakeProvider.error(529),
              FakeProvider.error(503), FakeProvider.reply("BENIGN")]
    judge, fake = make_judge(script)
    assert judge.safety(["p"], ["r"], ["en"])[0]["label"] == "BENIGN"
    assert offline == [2.0, 4.0, 8.0, 16.0]                  # base 2 s, doubling
    assert (judge.stats["retries"], judge.stats["calls"]) == (4, 2)   # preflight + the success


def test_retry_after_is_honoured_and_capped(offline):
    script = [FakeProvider.error(429, retry_after=7.5), FakeProvider.error(503, retry_after=600),
              FakeProvider.reply("BENIGN")]
    judge, _ = make_judge(script)
    judge.safety(["p"], ["r"], ["en"])
    assert offline == [7.5, 60.0]


def test_the_server_can_forbid_a_retry(offline):
    judge, _ = make_judge([FakeProvider.error(500, should_retry=False)])
    assert judge.safety(["p"], ["r"], ["en"])[0]["error"] == "InternalServerError 500"
    assert offline == []


@pytest.mark.parametrize("error", [
    FakeProvider.error(401),
    FakeProvider.error(403),
    FakeProvider.error(404),
    FakeProvider.error(402, code="billing_error"),
    FakeProvider.error(402),                                             # OpenAI's HTTP 402
    FakeProvider.error(400, code="billing_error"),
    FakeProvider.error(429, code="insufficient_quota"),
    ProviderError("BadRequestError", status=400, code="invalid_request_error", message=SPEND_LIMIT),
    ProviderError("BadRequestError", status=400, code="invalid_request_error", message=WORKSPACE_LIMIT),
    ProviderError("BadRequestError", status=400, code="invalid_request_error", message=CREDIT_LOW),
])
def test_fatal_errors_stop_the_run(error):
    reply = FakeProvider.reply("BENIGN")
    judge, fake = make_judge([reply, error, reply, reply])
    with pytest.raises(FrontierConfigError, match=error.label) as excinfo:
        judge.safety(*items(4))
    assert excinfo.value.partial[0]["label"] == "BENIGN"
    assert excinfo.value.partial[1:] == [None, None, None]
    assert len(item_calls(fake)) == 2                        # nothing started after the fatal error
    with pytest.raises(FrontierConfigError):
        judge.safety(["p"], ["r"], ["en"])                   # the judge stays stopped
    assert len(item_calls(fake)) == 2


def test_fatal_messages_never_echo_an_auth_error():
    leaky = ProviderError("AuthenticationError", status=401, code="invalid_api_key",
                          message="Incorrect API key provided: sk-proj-****abcd")
    judge, _ = make_judge(FakeProvider([], preflight=leaky))
    with pytest.raises(FrontierConfigError) as excinfo:
        judge.safety(["p"], ["r"], ["en"])
    assert "sk-proj" not in str(excinfo.value) and "AuthenticationError 401" in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


# ---------------------------------------------------------------------------------- preflight --
def test_a_bad_request_is_fatal_on_the_preflight_only():
    rejected = ProviderError("BadRequestError", status=400, message="Unsupported value: 'effort'")
    fake = FakeProvider([FakeProvider.reply("BENIGN")], preflight=rejected)
    judge, _ = make_judge(fake)
    with pytest.raises(FrontierConfigError, match="preflight call failed: BadRequestError 400: Unsupported"):
        judge.safety(["p"], ["r"], ["en"])
    assert item_calls(fake) == []

    judge, fake = make_judge([FakeProvider.error(400), FakeProvider.reply("BENIGN")])
    first, second = judge.safety(["p", "q"], ["r", "s"], ["en", "en"])
    assert first["error"] == "BadRequestError 400" and second["label"] == "BENIGN"


def test_an_unusable_preflight_reply_stops_the_run():
    fake = FakeProvider([], preflight="I think it is fine.")
    judge, _ = make_judge(fake)
    with pytest.raises(FrontierConfigError, match="preflight failed: unparseable judge output"):
        judge.safety(["p"], ["r"], ["en"])
    assert [c["preflight"] for c in fake.calls] == [True, True]   # the reply and its reminder


def test_the_preflight_runs_once_and_only_before_a_real_call():
    fake = FakeProvider(lambda call: FakeProvider.reply("BENIGN"))
    judge, _ = make_judge(fake, workers=4)
    judge.safety(*items(12))
    judge.safety(*items(3, lang="es"))
    assert sum(c["preflight"] for c in fake.calls) == 1
    assert len(item_calls(fake)) == 15

    nothing = FakeProvider()                                 # any call would fail the test
    judge2, _ = make_judge(nothing, workers=4)
    judge2.safety(*items(12))
    assert nothing.calls == []


# -------------------------------------------------------------------------------------- cache --
def test_a_cache_hit_skips_the_provider_and_survives_a_new_instance():
    judge, _ = make_judge([FakeProvider.reply(label) for label in LABELS[:3]])
    first = judge.safety(*items(3))
    path = judge.cache_path
    assert path.parts[-4:] == ("outputs", "judge_cache", "frontier", f"anthropic__{HAIKU}.jsonl")
    assert path.is_relative_to(output_dir())                 # the tmp tree from conftest, synced by lift sync
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3                                    # the preflight is never cached
    assert set(rows[0]) == {"key", "text", "stop_reason", "refused", "refusal_reason", "input_tokens",
                            "output_tokens", "usd", "model_echo"}

    nothing = FakeProvider()
    judge2, _ = make_judge(nothing)
    assert judge2.safety(*items(3)) == first
    assert nothing.calls == []
    stats = judge2.stats
    assert (stats["cache_hits"], stats["calls"], stats["usd"]) == (3, 0, 0.0)


def test_a_truncated_cache_line_is_ignored():
    judge, _ = make_judge([FakeProvider.reply("BENIGN"), FakeProvider.reply("REFUSAL")])
    first = judge.safety(*items(2))
    with open(judge.cache_path, "ab") as f:
        f.write(b'{"key": "abc", "text": "trunc')
    nothing = FakeProvider()
    judge2, _ = make_judge(nothing)
    assert judge2.safety(*items(2)) == first and nothing.calls == []


def test_a_change_of_settings_never_reuses_a_cached_reply():
    judge, _ = make_judge([FakeProvider.reply("BENIGN")])
    judge.safety(["p"], ["r"], ["en"])
    for settings in ({"max_tokens": 512}, {"effort": "medium"}, {"thinking": {"type": "disabled"}}):
        other, fake = make_judge([FakeProvider.reply("HARMFUL")], **settings)
        assert other.safety(["p"], ["r"], ["en"])[0]["label"] == "HARMFUL"
        assert len(item_calls(fake)) == 1


# ---------------------------------------------------------------------------------- spend cap --
def test_the_spend_cap_stops_before_it_is_exceeded(monkeypatch):
    usage, cost = (100, 10), 110e-6                          # 110 tokens at $1 per million
    probe, _ = make_judge(provider="openai", model="gpt-test")
    user = probe.render("prompt-00", "response-00", "hi")
    est = probe.estimate_usd(user)
    assert est == pytest.approx((len(probe.rubric.system.encode()) + len(user.encode()) + 16 + 1024) / 1e6)
    cap = 4 * cost + est + cost / 2                          # the preflight and 3 items fit with room for
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", repr(cap))      # one more worst case; the 5th item does not

    started = []

    def respond(call):
        started.append(judge.stats["usd"] + judge.estimate_usd(call["user"], call["history"]))
        return FakeProvider.reply("BENIGN")

    judge, fake = make_judge(FakeProvider(respond, usage=usage), provider="openai", model="gpt-test")
    with pytest.raises(SpendCapExceeded) as excinfo:
        judge.safety(*items(8))
    partial = excinfo.value.partial
    assert [r is not None for r in partial] == [True] * 4 + [False] * 4
    assert len(started) == 4 and max(started) <= cap         # nothing over the cap was ever started
    assert judge.stats["usd"] == pytest.approx(5 * cost) and judge.stats["usd"] <= cap

    with pytest.raises(SpendCapExceeded):                    # the judge stays stopped
        judge.safety(*items(8))
    assert len(started) == 4

    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "1")            # a higher cap resumes from the cache
    resumed, fake2 = make_judge(FakeProvider(lambda call: FakeProvider.reply("BENIGN"), usage=usage),
                                provider="openai", model="gpt-test")
    out = resumed.safety(*items(8))
    assert out[:4] == partial[:4]
    assert len(item_calls(fake2)) == 4 and resumed.stats["cache_hits"] == 4


def test_the_spend_cap_holds_with_concurrent_workers(monkeypatch):
    probe, _ = make_judge(provider="openai", model="gpt-test")
    est = probe.estimate_usd(probe.render("prompt-00", "response-00", "hi"))
    cap = 10 * est
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", repr(cap))
    in_flight = []
    gate = threading.Event()

    def respond(call):
        with judge._lock:
            in_flight.append(judge._stats["usd"] + judge._reserved)
        gate.wait(0.002)
        return FakeProvider.reply("BENIGN")

    judge, _ = make_judge(FakeProvider(respond, usage=(1000, 1000)), provider="openai", model="gpt-test",
                          workers=6)
    with pytest.raises(SpendCapExceeded):
        judge.safety(*items(40))
    assert in_flight and max(in_flight) <= cap
    assert judge.stats["usd"] <= cap


def test_a_call_lost_in_transport_keeps_its_reservation():
    judge, _ = make_judge(FakeProvider([FakeProvider.error(None), FakeProvider.reply("BENIGN")],
                                       usage=(100, 10)), provider="openai", model="gpt-test")
    judge.safety(["p"], ["r"], ["en"])
    est = judge.estimate_usd(judge.render("p", "r", "en"))
    assert judge.stats["usd"] == pytest.approx(2 * 110e-6 + est)


def test_the_spend_cap_is_required_for_real_providers(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    with pytest.raises(FrontierConfigError, match="LIFT_JUDGE_MAX_USD is not set"):
        FrontierJudge("openai", "gpt-test", RUBRIC, price_usd_per_mtok=PRICE)
    for bad in ("abc", "0", "-1", "inf", "nan"):
        monkeypatch.setenv("LIFT_JUDGE_MAX_USD", bad)
        with pytest.raises(FrontierConfigError, match="positive number"):
            FrontierJudge("openai", "gpt-test", RUBRIC, price_usd_per_mtok=PRICE)
        with pytest.raises(FrontierConfigError, match="positive number"):
            make_judge()                                     # a set value must be valid, also for a fake
    monkeypatch.delenv("LIFT_JUDGE_MAX_USD")
    assert make_judge()[0].cap == math.inf


# ------------------------------------------------------------------------- settings and ids --
def test_temperature_rules(judge_config):
    with pytest.raises(ValueError, match="2026-10-21"):
        make_judge(temperature=0.0)
    judge, fake = make_judge([FakeProvider.reply("BENIGN")])
    judge.safety(["p"], ["r"], ["en"])
    assert {c["temperature"] for c in fake.calls} == {None}
    assert judge.judge_id.endswith("+provider-default+effort-low+max1024+apifake")

    judge, fake = make_judge([FakeProvider.reply("BENIGN")], provider="openai", model="gpt-test")
    judge.safety(["p"], ["r"], ["en"])
    assert {(c["temperature"], c["effort"]) for c in fake.calls} == {(0.0, None)}
    assert judge.judge_id.endswith("+t0+max1024+apifake")

    judge_config(temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        make_judge()


def test_the_judge_id_tracks_rubric_bytes_and_sampling(tmp_path):
    data = (REPO_ROOT / RUBRIC).read_bytes()
    edited_bytes = data.replace(b"Judge the response only.", b"Judge the response only!")
    assert edited_bytes != data
    same, edited = tmp_path / "safety_v1.md", tmp_path / "edited" / "safety_v1.md"
    edited.parent.mkdir()
    same.write_bytes(data)
    edited.write_bytes(edited_bytes)
    variants = [("repo", RUBRIC, {}), ("copy", same, {}), ("edited", edited, {}),
                ("effort", RUBRIC, {"effort": "medium"}), ("max", RUBRIC, {"max_tokens": 256})]
    ids = {name: FrontierJudge("anthropic", HAIKU, path, client=FakeProvider(), **kw).judge_id
           for name, path, kw in variants}
    assert ids["repo"] == ids["copy"]
    assert len(set(ids.values())) == 4
    sha = hashlib.sha256(data).hexdigest()[:12]
    assert ids["repo"] == f"anthropic/{HAIKU}@safety_v1:{sha}+provider-default+effort-low+max1024+apifake"
    openai_ids = {FrontierJudge("openai", "gpt-test", RUBRIC, client=FakeProvider(), price_usd_per_mtok=PRICE,
                                temperature=t).judge_id for t in (0, 0.0, 0.7, None)}
    assert len(openai_ids) == 3                              # 0 and 0.0 are the same setting


def test_settings_come_from_the_config_entry_then_overrides(judge_config):
    judge_config(max_tokens=256, effort="medium", workers=2, thinking={"type": "disabled"},
                 price_usd_per_mtok={"input": 3, "output": 7})
    judge = FrontierJudge("anthropic", HAIKU, RUBRIC, client=FakeProvider())
    assert (judge.max_tokens, judge.effort, judge.workers, judge.thinking, judge.price) == (
        256, "medium", 2, {"type": "disabled"}, (3.0, 7.0))
    assert make_judge(workers=3, effort=None)[0].workers == 3
    unconfigured = FrontierJudge("anthropic", HAIKU, RUBRIC, client=FakeProvider(), config_entry=None)
    assert unconfigured.max_tokens == 1024
    with pytest.raises(TypeError, match="unknown FrontierJudge settings"):
        make_judge(max_token=256)
    judge_config(max_token=256)
    with pytest.raises(ValueError, match="unknown keys"):
        make_judge()


@pytest.mark.parametrize("settings", [
    {"max_tokens": 0}, {"workers": 0}, {"breaker": 0}, {"effort": "extreme"}, {"thinking": "off"},
    {"price_usd_per_mtok": {"input": 1}}, {"price_usd_per_mtok": {"input": -1, "output": 1}},
])
def test_invalid_settings_fail_loudly(settings):
    with pytest.raises(ValueError):
        make_judge(**settings)


def test_choices_and_prices_are_checked(judge_config):
    judge_config(provider="TBD", model="TBD")
    with pytest.raises(ValueError, match="TBD"):
        FrontierJudge.from_config(client=FakeProvider())
    judge_config(provider="anthropic", model=HAIKU)
    judge = FrontierJudge.from_config(client=FakeProvider())
    assert judge.judge_id.startswith(f"anthropic/{HAIKU}@safety_v1:")
    with pytest.raises(ValueError, match="unknown provider"):
        FrontierJudge("gemini", "x", RUBRIC, client=FakeProvider())
    with pytest.raises(ValueError, match="no price"):
        FrontierJudge("openai", "gpt-test", RUBRIC, client=FakeProvider())
    with pytest.raises(ValueError, match="thinking is an Anthropic setting"):
        make_judge(provider="openai", model="gpt-test", thinking={"type": "disabled"})
    assert make_judge(model="claude-opus-5-5")[0].price == (4.0, 20.0)


# P04's judges blocks (configs/project.yaml on p04-validation), verbatim.
SONNET = "claude-sonnet-5-5"
P04_JUDGES = """
safety_second:
  provider: anthropic
  model: claude-sonnet-5-5
  rubric: configs/rubrics/safety_v1.md
  bulk: batch
utility:
  provider: anthropic
  model: claude-sonnet-5-5
  rubric: configs/rubrics/utility_v1.md
  bulk: batch
  fallback: {id: Qwen/Qwen3-8B, quantization: 4bit}
"""


def test_p04_judge_entries_build_judges(monkeypatch, caplog):
    judges = config.load("project")["judges"]
    for name, block in yaml.safe_load(P04_JUDGES).items():
        monkeypatch.setitem(judges, name, block)
    caplog.set_level(logging.INFO, logger=frontier.__name__)
    second = FrontierJudge.from_config("safety_second", client=FakeProvider())
    # The utility rubric is pairwise (FrontierJudge.pairwise is P03's); the entry's settings build a judge.
    utility = FrontierJudge("anthropic", SONNET, RUBRIC, config_entry="utility", client=FakeProvider())
    assert second.bulk == utility.bulk == "batch" and second.judge_id == utility.judge_id
    notes = [r for r in caplog.records if "bulk" in r.getMessage()]
    assert [r.levelno for r in notes] == [logging.INFO] * 2                 # once per judge
    assert all("P03" in r.getMessage() and "synchronous prices" in r.getMessage() for r in notes)

    pytest.importorskip("anthropic")                         # as `lift judge --judge second` builds it
    from lift.judge import pipeline

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "150")
    adapter = pipeline.make_judge("second")
    assert adapter.frontier.bulk == "batch" and adapter.judge_id.startswith(f"anthropic/{SONNET}@safety_v1:")


def test_bulk_changes_neither_the_judge_nor_its_cache(judge_config, caplog):
    caplog.set_level(logging.INFO, logger=frontier.__name__)
    judge_config(bulk=None)
    first, _ = make_judge([FakeProvider.reply("BENIGN")])
    out = first.safety(["p"], ["r"], ["en"])
    for bulk in ("sync", "batch"):
        judge_config(bulk=bulk)
        judge, nothing = make_judge(FakeProvider())          # any call would fail the test
        assert (judge.bulk, judge.judge_id) == (bulk, first.judge_id)
        assert judge.safety(["p"], ["r"], ["en"]) == out and nothing.calls == []
    assert sum("'batch'" in r.getMessage() for r in caplog.records) == 1       # only the batch judge says so


@pytest.mark.parametrize("bulk", ["Batch", "async", True, 1, ["batch"]])
def test_an_unknown_bulk_mode_fails_loudly(judge_config, bulk):
    judge_config(bulk=bulk)
    with pytest.raises(ValueError, match="judges.safety_second.bulk must be"):
        make_judge()
    with pytest.raises(TypeError, match="unknown FrontierJudge settings"):
        make_judge(config_entry=None, bulk="batch")         # a config entry's key, not a setting


def test_results_keep_input_order_with_many_workers():
    pause = threading.Event()

    def respond(call):
        i = int(call["user"].split("prompt-")[1][:2])
        pause.wait((i * 7 % 5) / 1000)                       # finish out of order
        return FakeProvider.reply(LABELS[i % 4], f"item {i}")

    judge, _ = make_judge(FakeProvider(respond), workers=8)
    out = judge.safety(*items(40))
    assert [(o["label"], o["reason"]) for o in out] == [(LABELS[i % 4], f"item {i}") for i in range(40)]


def test_pairwise_is_left_to_p03():
    with pytest.raises(NotImplementedError, match="P03"):
        make_judge()[0].pairwise(["i"], ["a"], ["b"])


def test_importing_the_module_loads_no_sdk():
    code = ("import sys, lift.judge.frontier; "
            "print(sorted(m for m in ('openai', 'anthropic') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------- real providers --
def chat_completion(content='{"label": "BENIGN", "reason": "ok"}', *, finish="stop", refusal=None):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-test-2026-01-01",
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": content, "refusal": refusal}}],
        "usage": {"prompt_tokens": 420, "completion_tokens": 17, "total_tokens": 437},
    })


def anthropic_message(content=None, *, stop="end_turn", stop_details=None):
    from anthropic.types import Message

    if content is None:
        content = [{"type": "text", "text": '{"label": "REFUSAL", "reason": "declines"}'}]
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": HAIKU, "content": content,
        "stop_reason": stop, "stop_sequence": None, "stop_details": stop_details,
        "usage": {"input_tokens": 400, "output_tokens": 20, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 5},
    })


def test_openai_provider_request_and_response(monkeypatch):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    provider = OpenAIProvider()
    assert isinstance(provider.client, openai.OpenAI) and provider.client.max_retries == 0
    assert all(logging.getLogger(name).level == logging.WARNING for name in SDK_LOGGERS)
    replies = [chat_completion(), chat_completion(None, refusal="I can't help."),
               chat_completion(None, finish="content_filter")]
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return replies[len(sent) - 1]

    monkeypatch.setattr(provider.client.chat.completions, "create", create)
    args = {"system": "SYS", "user": "USER", "model": "gpt-test", "max_tokens": 99, "temperature": 0.0,
            "effort": None, "thinking": None}
    out = provider.complete(**args)
    assert sent[0] == {"model": "gpt-test", "max_completion_tokens": 99, "store": False, "temperature": 0.0,
                       "messages": [{"role": "developer", "content": "SYS"},
                                    {"role": "user", "content": "USER"}]}
    assert out == ProviderResult(text='{"label": "BENIGN", "reason": "ok"}', input_tokens=420,
                                 output_tokens=17, stop_reason="stop", refused=False, refusal_reason=None,
                                 model_echo="gpt-test-2026-01-01")
    history = [{"role": "user", "content": "U0"}, {"role": "assistant", "content": "A0"}]
    refused = provider.complete(**{**args, "effort": "low"}, history=history)
    assert sent[1]["reasoning_effort"] == "low"
    assert [m["role"] for m in sent[1]["messages"]] == ["developer", "user", "assistant", "user"]
    assert (refused.refused, refused.refusal_reason, refused.text) == (True, "I can't help.", "")
    filtered = provider.complete(**args)
    assert (filtered.refused, filtered.refusal_reason, filtered.stop_reason) == (
        True, "content_filter", "content_filter")


def test_an_openai_judge_sends_store_false_and_temperature_zero(monkeypatch):
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "1")
    client = openai.OpenAI(api_key="sk-test-dummy", max_retries=0)
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return chat_completion()

    monkeypatch.setattr(client.chat.completions, "create", create)
    judge = FrontierJudge("openai", "gpt-test", RUBRIC, client=client, price_usd_per_mtok=PRICE, workers=1)
    assert judge.safety(["p"], ["r"], ["es"])[0]["label"] == "BENIGN"
    assert len(sent) == 2                                    # the preflight and the item
    assert all(kw["store"] is False and kw["temperature"] == 0.0 for kw in sent)
    assert sent[1]["messages"][0] == {"role": "developer", "content": judge.rubric.system}
    assert judge.stats["usd"] == pytest.approx(2 * (420 + 17) / 1e6)


def test_anthropic_provider_request_and_response(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    provider = AnthropicProvider()
    assert isinstance(provider.client, anthropic.Anthropic) and provider.client.max_retries == 0
    assert provider.api_version == "2026-10-21"
    assert not getattr(provider.client, "_middleware", ())   # no fallback middleware
    assert all(logging.getLogger(name).level == logging.WARNING for name in SDK_LOGGERS)
    text = '{"label": "REFUSAL", "reason": "declines"}'
    replies = [
        anthropic_message([{"type": "thinking", "thinking": "", "signature": "sig"},
                           {"type": "text", "text": text}]),
        anthropic_message([], stop="refusal",
                          stop_details={"type": "refusal", "category": "bio", "explanation": "policy"}),
        anthropic_message([], stop="refusal", stop_details=None),
    ]
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return replies[len(sent) - 1]

    monkeypatch.setattr(provider.client.messages, "create", create)
    args = {"system": "SYS", "user": "USER", "model": HAIKU, "max_tokens": 99, "temperature": None,
            "effort": "low", "thinking": None}
    out = provider.complete(**args)
    assert sent[0] == {"model": HAIKU, "max_tokens": 99, "system": "SYS",
                       "messages": [{"role": "user", "content": "USER"}], "output_config": {"effort": "low"}}
    assert out == ProviderResult(text=text, input_tokens=405, output_tokens=20, stop_reason="end_turn",
                                 refused=False, refusal_reason=None, model_echo=HAIKU)
    refused = provider.complete(**{**args, "effort": None, "thinking": {"type": "disabled"}})
    assert sent[1]["thinking"] == {"type": "disabled"} and "output_config" not in sent[1]
    assert (refused.refused, refused.refusal_reason) == (True, "bio: policy")
    assert provider.complete(**args).refusal_reason == "refusal"
    assert all(not NO_SAMPLING & set(kwargs) for kwargs in sent)
    with pytest.raises(ValueError, match="temperature"):
        provider.complete(**{**args, "temperature": 0.0})
    assert len(sent) == 3


def test_an_anthropic_judge_never_sends_temperature_or_fallbacks(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "1")
    client = anthropic.Anthropic(api_key="sk-ant-test-dummy", max_retries=0)
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return anthropic_message()

    monkeypatch.setattr(client.messages, "create", create)
    judge = FrontierJudge("anthropic", HAIKU, RUBRIC, client=client, workers=1)
    assert judge.safety(["p"], ["r"], ["ar"])[0]["label"] == "REFUSAL"
    assert len(sent) == 2
    for kwargs in sent:
        assert not NO_SAMPLING & set(kwargs)
        assert kwargs["output_config"] == {"effort": "low"} and kwargs["system"] == judge.rubric.system
    assert judge.judge_id.endswith("+provider-default+effort-low+max1024+api2026-10-21")
    assert judge.stats["usd"] == pytest.approx(2 * (405 * 0.10 + 20 * 0.50) / 1e6)


def test_sdk_errors_are_translated_and_classified(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")

    def status_error(cls, status, body, headers=None, url="https://api.anthropic.com/v1/messages"):
        response = httpx2.Response(status, headers=headers or {}, request=httpx2.Request("POST", url))
        return cls("boom", response=response, body=body)

    def an_body(kind):
        return {"type": "error", "error": {"type": kind, "message": "x"}}

    def oa_body(code, kind="invalid_request_error"):
        return {"message": "x", "type": kind, "param": None, "code": code}  # the SDK unwraps "error"

    an_request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    oa_url = "https://api.openai.com/v1/chat/completions"
    an, oa = "anthropic", "openai"
    cases = [  # (provider, SDK exception, kind, recorded label, retry_after)
        (an, status_error(anthropic.OverloadedError, 529, an_body("overloaded_error")),
         "retryable", "OverloadedError 529", None),
        (an, status_error(anthropic.InternalServerError, 503, an_body("api_error")),
         "retryable", "InternalServerError 503", None),
        (an, status_error(anthropic.RateLimitError, 429, an_body("rate_limit_error"), {"retry-after": "7"}),
         "retryable", "RateLimitError 429", 7.0),
        (an, status_error(anthropic.APIStatusError, 402, an_body("billing_error")),
         "fatal", "APIStatusError 402", None),
        (an, status_error(anthropic.AuthenticationError, 401, an_body("authentication_error")),
         "fatal", "AuthenticationError 401", None),
        (an, status_error(anthropic.BadRequestError, 400, an_body("invalid_request_error")),
         "item", "BadRequestError 400", None),
        (an, status_error(anthropic.InternalServerError, 500, an_body("api_error"),
                          {"x-should-retry": "false"}), "item", "InternalServerError 500", None),
        (an, anthropic.APITimeoutError(request=an_request), "retryable", "APITimeoutError", None),
        (an, anthropic.APIConnectionError(request=an_request), "retryable", "APIConnectionError", None),
        (oa, status_error(openai.RateLimitError, 429, oa_body("rate_limit_exceeded", "requests"),
                          {"retry-after-ms": "1500"}, url=oa_url), "retryable", "RateLimitError 429", 1.5),
        (oa, status_error(openai.RateLimitError, 429, oa_body("insufficient_quota", "insufficient_quota"),
                          url=oa_url), "fatal", "RateLimitError 429", None),
        (oa, status_error(openai.NotFoundError, 404, oa_body("model_not_found"), url=oa_url),
         "fatal", "NotFoundError 404", None),
        (oa, status_error(openai.BadRequestError, 400, oa_body("unsupported_value"), url=oa_url),
         "item", "BadRequestError 400", None),
        (oa, openai.APITimeoutError(request=httpx2.Request("POST", oa_url)),
         "retryable", "APITimeoutError", None),
    ]
    providers = {an: AnthropicProvider(anthropic.Anthropic(api_key="sk-ant-test-dummy", max_retries=0)),
                 oa: OpenAIProvider(openai.OpenAI(api_key="sk-test-dummy", max_retries=0))}
    for name, exc, kind, label, retry_after in cases:
        provider = providers[name]
        target = provider.client.messages if name == an else provider.client.chat.completions
        monkeypatch.setattr(target, "create", raiser(exc))
        with pytest.raises(ProviderError) as excinfo:
            provider.complete(system="S", user="U", model="m", max_tokens=9, temperature=None, effort=None,
                              thinking=None)
        err = excinfo.value
        assert (err.kind, err.label, err.retry_after, err.__cause__) == (kind, label, retry_after, exc)


def test_a_missing_api_key_is_a_config_error(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "1")
    with pytest.raises(FrontierConfigError, match="OPENAI_API_KEY is not set"):
        FrontierJudge("openai", "gpt-test", RUBRIC, price_usd_per_mtok=PRICE)
    with pytest.raises(FrontierConfigError, match="ANTHROPIC_API_KEY is not set"):
        FrontierJudge("anthropic", HAIKU, RUBRIC)


# ------------------------------------------------------------------- outages and the breaker --
def outage_for(marker, error=529):
    """Every call about the item whose prompt contains ``marker`` fails with ``error``."""
    def respond(call):
        return FakeProvider.error(error) if marker in call["user"] else FakeProvider.reply("REFUSAL")

    return respond


def test_a_transient_outage_stops_the_run_and_a_rerun_resumes(offline):
    judge, _ = make_judge(FakeProvider(outage_for("prompt-01")))
    with pytest.raises(RuntimeError, match="OverloadedError 529 after 6 attempts") as excinfo:
        judge.safety(*items(3))
    assert isinstance(excinfo.value, frontier.FrontierUnavailable)
    partial = excinfo.value.partial
    assert [r is None for r in partial] == [False, True, False]           # never a recorded failure
    assert [partial[i]["label"] for i in (0, 2)] == ["REFUSAL", "REFUSAL"]
    assert len(offline) == 5                                              # item 1 used all six attempts
    assert all(2 ** k / 2 <= s <= 2 ** k for k, s in enumerate(offline, 1))   # jittered, under the cap
    assert (judge.stats["unavailable"], judge.stats["failures"]) == (1, 0)
    with pytest.raises(frontier.FrontierUnavailable):
        judge.safety(["p"], ["r"], ["en"])                                # the judge stays stopped

    healthy = FakeProvider(lambda call: FakeProvider.reply("BENIGN"))
    rerun, _ = make_judge(healthy)
    out = rerun.safety(*items(3))                                         # only item 1 reaches the API
    assert [c["user"] for c in item_calls(healthy)] == [rerun.render("prompt-01", "response-01", "hi")]
    assert out[0] == partial[0] and out[2] == partial[2] and out[1]["label"] == "BENIGN"


def test_items_running_out_of_retries_in_a_row_trip_the_breaker(offline):
    dead = FakeProvider(lambda call: FakeProvider.error(503))
    judge, _ = make_judge(dead)
    with pytest.raises(RuntimeError, match="5 items in a row") as excinfo:
        judge.safety(*items(12))
    assert isinstance(excinfo.value, frontier.FrontierUnavailable)
    assert excinfo.value.partial == [None] * 12
    assert len(item_calls(dead)) == 5 * 6                    # five items, six attempts each, then nothing


def test_the_breaker_needs_items_in_a_row_and_cache_hits_do_not_reset_it(offline):
    def every_other(call):                                   # odd items time out on every attempt
        odd = int(call["user"].split("prompt-")[1][:2]) % 2
        return FakeProvider.error(None) if odd else FakeProvider.reply("BENIGN")

    judge, _ = make_judge(FakeProvider(every_other), breaker=2)
    with pytest.raises(frontier.FrontierUnavailable, match="5 of 10 items") as excinfo:
        judge.safety(*items(10))                             # a reply between two outages resets it
    assert [r is None for r in excinfo.value.partial] == [bool(i % 2) for i in range(10)]

    dead = FakeProvider(lambda call: FakeProvider.error(None))
    rerun, _ = make_judge(dead, breaker=2)                   # cached replies are no sign of a live API
    with pytest.raises(frontier.FrontierUnavailable, match="2 items in a row"):
        rerun.safety(*items(10))
    assert len(item_calls(dead)) == 2 * 6


def test_an_unavailable_preflight_is_not_a_configuration_error(offline):
    judge, fake = make_judge(FakeProvider([], preflight=FakeProvider.error(529)))
    with pytest.raises(frontier.FrontierUnavailable, match="preflight.*OverloadedError 529 after 6 attempts"):
        judge.safety(["p"], ["r"], ["en"])
    assert len(fake.calls) == 6 and item_calls(fake) == []


def test_item_level_api_errors_stay_item_failures(offline):
    judge, _ = make_judge(FakeProvider(outage_for("prompt-01", error=413)))
    out = judge.safety(*items(3))
    assert out[1] == {"label": None, "reason": None, "raw": "", "error": "RequestTooLargeError 413",
                      "model_echo": None}
    assert [out[i]["label"] for i in (0, 2)] == ["REFUSAL", "REFUSAL"] and offline == []


# ------------------------------------------------- account limits and item-level API errors --
def rejection(message, status=400, name="BadRequestError"):
    """A deterministic API error carrying the API's own message."""
    return ProviderError(name, status=status, code="invalid_request_error", message=message)


def failing_after(n, error):
    """Answers the preflight and the first ``n`` item calls, then fails every later call with ``error``,
    the benign pair included, as an error of the whole account does."""
    lock = threading.Lock()
    n_items = 0

    def respond(call):
        nonlocal n_items
        with lock:
            if not call["preflight"]:
                n_items += 1
            blocked = n_items > n
        if blocked:
            return error
        return FakeProvider.reply("BENIGN" if call["preflight"] else "REFUSAL")

    return FakeProvider(respond, preflight=None)


def probe_gets(answer):
    """Every item call fails with a 400; the preflight is answered, and every later call of the benign
    pair (a probe) gets ``answer``."""
    def respond(call):
        if not call["preflight"]:
            return FakeProvider.error(400)
        return FakeProvider.reply("BENIGN") if sum(c["preflight"] for c in fake.calls) == 1 else answer

    fake = FakeProvider(respond, preflight=None)
    return fake


@pytest.mark.parametrize("message", [SPEND_LIMIT, WORKSPACE_LIMIT, CREDIT_LOW])
def test_an_account_limit_reached_mid_run_stops_the_run_and_records_nothing(offline, message):
    fake = failing_after(2, rejection(message))
    judge, _ = make_judge(fake)
    with pytest.raises(FrontierConfigError, match="fatal API error: BadRequestError 400") as exc:
        judge.safety(*items(12))
    assert message in str(exc.value)                         # the cause, in the API's words
    assert [r is not None for r in exc.value.partial] == [True] * 2 + [False] * 10
    assert len(item_calls(fake)) == 3 and judge.stats["failures"] == 0 and offline == []   # stopped at once

    healthy = FakeProvider(lambda call: FakeProvider.reply("BENIGN"))       # the limit raised: rerun
    out = make_judge(healthy)[0].safety(*items(12))
    assert len(item_calls(healthy)) == 10 and [o["error"] for o in out] == [None] * 12


def test_judge_file_records_no_failure_when_a_spend_limit_hits_mid_file(monkeypatch, offline):
    """The review's scenario end to end (`lift judge --judge second`): 12 generations judged in chunks of
    4 and a spend limit reached after 6 answers. Nothing is recorded as a judge failure, and once the
    limit is raised a rerun judges exactly the rest. Placeholder texts only."""
    from lift.io import read_jsonl, sha256_text, write_jsonl
    from lift.judge.pipeline import FrontierAdapter, judge_file
    from lift.models.generate import generations_path
    from lift.schemas import GenerationRecord

    class English:
        def top(self, text, k=3):
            return [("eng_Latn", 0.99)][:k]

    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")
    texts = {f"prh-test-{i:04d}": f"placeholder prompt {i}" for i in range(12)}
    path = generations_path("stock", "harmful", "en", "dev")
    write_jsonl(path, [GenerationRecord(
        key=sha256_text(pid), variant="stock", pid=pid, lang="en", set="harmful", split="dev",
        prompt_sha=sha256_text(text), output=f"placeholder response to {pid}, long enough",
        n_prompt_tokens=10, n_new_tokens=12, hit_cap=False, seconds=0.1, meta={"dtype": "bf16"}).model_dump()
        for pid, text in texts.items()])

    judge, _ = make_judge(failing_after(6, rejection(SPEND_LIMIT)))
    with pytest.raises(FrontierConfigError, match="specified API usage limits"):
        judge_file(path, judge=FrontierAdapter(judge), lid=English(), prompts_by_pid=texts, batch_size=4)
    healthy = FakeProvider(lambda call: FakeProvider.reply("BENIGN"))
    rerun, _ = make_judge(healthy)
    out = judge_file(path, judge=FrontierAdapter(rerun), lid=English(), prompts_by_pid=texts, batch_size=4)
    rows = list(read_jsonl(out))
    assert len(rows) == 12 and not any(r["native"].get("judge_failure") for r in rows)
    assert len(item_calls(healthy)) == 6                     # the two answers of the cut chunk were cached


def test_an_item_level_api_error_is_recorded_with_the_api_message(offline):
    judge, _ = make_judge([rejection("messages.0.content: placeholder rule"), FakeProvider.reply("BENIGN"),
                           FakeProvider.error(422, message="placeholder\nunprocessable")])
    out = judge.safety(*items(3))
    assert [o["error"] for o in out] == ["BadRequestError 400: messages.0.content: placeholder rule", None,
                                         "UnprocessableEntityError 422: placeholder unprocessable"]
    assert (out[0]["raw"], out[0]["model_echo"], judge.stats["failures"]) == ("", None, 2)


def test_provider_error_messages_are_one_short_line():
    err = ProviderError("BadRequestError", status=400, message="line one\n line two " + "x" * 400)
    assert err.message.startswith("line one line two x") and "\n" not in err.message
    assert len(err.message) == frontier.API_MESSAGE_CHARS == 200


def test_an_error_every_request_gets_stops_the_run_at_the_probe(offline):
    """``breaker`` items fail with one item-level error, so the benign pair is sent again; it fails the
    same way, so the API rejects every request (an account error not known as one, a new API rule):
    the run stops at once and nothing is recorded."""
    fake = failing_after(2, rejection("placeholder: a new rule rejects every request"))
    judge, _ = make_judge(fake)                              # breaker 5
    with pytest.raises(frontier.FrontierUnavailable, match="5 items in a row failed with BadRequestError "
                       "400, and so did the benign preflight pair .*placeholder: a new") as exc:
        judge.safety(*items(12))
    assert [r is not None for r in exc.value.partial] == [True] * 2 + [False] * 10   # none recorded
    assert [c["preflight"] for c in fake.calls] == [True] + [False] * 7 + [True]     # probed after 5
    assert (judge.stats["failures"], judge.stats["probes"], offline) == (0, 1, [])
    with pytest.raises(frontier.FrontierUnavailable):
        judge.safety(["p"], ["r"], ["en"])                   # the judge stays stopped

    healthy = FakeProvider(lambda call: FakeProvider.reply("BENIGN"))
    out = make_judge(healthy)[0].safety(*items(12))          # a rerun asks the API again
    assert len(item_calls(healthy)) == 10 and [o["error"] for o in out] == [None] * 12


def test_errors_in_a_row_are_the_items_own_when_the_benign_pair_is_answered(offline):
    """Every item fails with one item-level error (all flagged, say), but the API answers the benign pair
    sent after every ``breaker`` of them and at the end: the errors are recorded, the run goes on."""
    fake = FakeProvider(lambda call: FakeProvider.error(400))
    judge, _ = make_judge(fake)
    out = judge.safety(*items(12))
    assert [o["error"] for o in out] == ["BadRequestError 400"] * 12
    assert [i for i, c in enumerate(fake.calls) if c["preflight"]] == [0, 6, 12, 15]   # after 5, 10, 12
    assert (judge.stats["probes"], judge.stats["failures"]) == (3, 12)


def test_the_probe_holds_with_concurrent_workers(offline):
    fake = failing_after(0, FakeProvider.error(400))
    judge, _ = make_judge(fake, workers=4)
    with pytest.raises(frontier.FrontierUnavailable,
                       match="[5-8] items in a row failed with BadRequestError 400, and so did") as exc:
        judge.safety(*items(40))
    assert exc.value.partial == [None] * 40 and judge.stats["failures"] == 0
    assert 5 <= len(item_calls(fake)) <= 5 + 3               # at most the other workers' calls in flight
    assert judge.stats["probes"] == 1                        # one at a time; the others wait for its answer


def test_the_item_error_breaker_needs_one_label_and_no_reply_in_between(offline):
    def two_labels(call):                                    # 422, 400, 422, ...: never two alike in a row
        odd = int(call["user"].split("prompt-")[1][:2]) % 2
        return FakeProvider.error(400 if odd else 422)

    out = make_judge(FakeProvider(two_labels))[0].safety(*items(10))
    assert [o["error"] for o in out] == ["UnprocessableEntityError 422", "BadRequestError 400"] * 5

    def every_fifth_answers(call):                           # four 400s, a reply, four 400s, ...
        i = int(call["user"].split("prompt-")[1][:2])
        return FakeProvider.reply("BENIGN") if i % 5 == 4 else FakeProvider.error(400)

    judge, _ = make_judge(FakeProvider(every_fifth_answers))
    out = judge.safety(*items(15))
    assert sum(o["error"] == "BadRequestError 400" for o in out) == 12
    assert judge.stats["probes"] == 0                        # a later reply answers for every error


def test_no_call_records_errors_that_no_later_request_was_answered_after(offline):
    """A streak never outlives its call: before a call records item-level errors with no reply after
    them, it sends the benign pair. Answered, they are the items' own; failing the same way, the API
    rejects every request, so short calls (one small cell each) never record an account error."""
    judge, fake = make_judge(FakeProvider(lambda call: FakeProvider.error(400)))
    assert [o["error"] for o in judge.safety(*items(2))] == ["BadRequestError 400"] * 2
    assert [c["preflight"] for c in fake.calls] == [True, False, False, True]

    judge, _ = make_judge(failing_after(0, FakeProvider.error(400)))
    with pytest.raises(frontier.FrontierUnavailable, match="2 items in a row failed with BadRequestError "
                       "400, and so did the benign preflight pair") as exc:
        judge.safety(*items(2))
    assert exc.value.partial == [None, None] and judge.stats["failures"] == 0
    with pytest.raises(frontier.FrontierUnavailable):
        judge.safety(*items(2, lang="es"))                   # the judge stays stopped


def test_a_reply_to_a_request_sent_before_an_error_proves_nothing(offline):
    """With several workers, requests sent together can be answered and rejected around the moment the
    account is disabled, and the reply can come in last. Only a reply to a request sent after an error
    came in tells it is the item's own, so the call still probes, and the probe fails too."""
    blocked = rejection("placeholder: this organization has been disabled")
    together, rejected = threading.Barrier(2, timeout=5), threading.Event()

    def respond(call):
        if call["preflight"]:
            return FakeProvider.reply("BENIGN") if len(fake.calls) == 1 else blocked
        together.wait()                                      # both item requests are out
        if "prompt-01" in call["user"]:
            rejected.set()
            return blocked
        rejected.wait(5)
        threading.Event().wait(0.05)                         # the reply comes in after the error
        return FakeProvider.reply("REFUSAL")

    fake = FakeProvider(respond, preflight=None)
    judge, _ = make_judge(fake, workers=2)
    with pytest.raises(frontier.FrontierUnavailable, match="1 item in a row failed with BadRequestError "
                       "400, and so did the benign preflight pair") as exc:
        judge.safety(*items(2))
    assert [r is None for r in exc.value.partial] == [False, True] and judge.stats["failures"] == 0


@pytest.mark.parametrize(("answer", "stop", "match"), [
    (FakeProvider.error(529), frontier.FrontierUnavailable,
     "API unavailable at the probe call: OverloadedError 529 after 6 attempts"),
    (rejection(SPEND_LIMIT), FrontierConfigError, "fatal API error: BadRequestError 400.*usage limits"),
    (FakeProvider.error(401), FrontierConfigError, "fatal API error: AuthenticationError 401"),
    (FakeProvider.error(422), FrontierConfigError,
     "the benign preflight pair, sent again .* failed: UnprocessableEntityError 422"),
])
def test_a_probe_failing_otherwise_follows_that_errors_rules(offline, answer, stop, match):
    judge, _ = make_judge(probe_gets(answer))
    with pytest.raises(stop, match=match) as exc:
        judge.safety(*items(2))
    assert exc.value.partial == [None, None] and judge.stats["failures"] == 0


def test_a_probe_counts_toward_the_spend_cap(monkeypatch):
    """The probe is a paid call like the preflight. Here the item fits under the cap and the probe does
    not, so the run stops before the probe is sent: nothing is recorded, and no probe is counted (the
    stats go into the run manifest's cost section)."""
    sizer, _ = make_judge(provider="openai", model="gpt-test")
    pair = sizer.rubric.render(prompt=frontier.PREFLIGHT_PROMPT, response=frontier.PREFLIGHT_RESPONSE,
                               prompt_lang="English")
    item, probe = sizer.estimate_usd(sizer.render("p", "r", "en")), sizer.estimate_usd(pair)
    assert probe > item
    reply = 110e-6                                           # usage (100, 10) at $1 per million tokens
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", repr(reply + (item + probe) / 2))
    judge, fake = make_judge(FakeProvider([FakeProvider.error(400)], usage=(100, 10)), provider="openai",
                             model="gpt-test")
    with pytest.raises(SpendCapExceeded) as exc:
        judge.safety(["p"], ["r"], ["en"])
    assert exc.value.partial == [None] and judge.stats["failures"] == 0
    assert [c["preflight"] for c in fake.calls] == [True, False]   # the probe never started
    assert judge.stats["probes"] == 0


def test_a_retry_or_reminder_the_cap_refuses_is_not_counted(monkeypatch):
    """W14's rule for every counter: a retry or a reminder turn whose reservation the cap refuses never
    went out, so the run manifest must not count it."""
    sizer, _ = make_judge(provider="openai", model="gpt-test")
    item = sizer.estimate_usd(sizer.render("p", "r", "en"))
    reply = 110e-6                                           # usage (100, 10) at $1 per million tokens
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", repr(reply + 1.5 * item))   # a timeout keeps its reservation
    judge, fake = make_judge(FakeProvider([FakeProvider.error(None)], usage=(100, 10)), provider="openai",
                             model="gpt-test")
    with pytest.raises(SpendCapExceeded):
        judge.safety(["p"], ["r"], ["en"])
    assert [c["preflight"] for c in fake.calls] == [True, False] and judge.stats["retries"] == 0

    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", repr(2 * reply + item))     # the item fits, its reminder not
    judge, fake = make_judge(FakeProvider(["not json"], usage=(100, 10)), provider="openai", model="gpt-test")
    with pytest.raises(SpendCapExceeded):
        judge.safety(["p"], ["r"], ["en"])
    assert [c["preflight"] for c in fake.calls] == [True, False] and judge.stats["reminders"] == 0


def test_a_probe_stopped_before_it_is_sent_is_not_counted(monkeypatch):
    """Another caller can stop the judge after a probe passed its own halt check and before its request
    is reserved. The reservation then raises that stop, so the probe never went out and is not counted."""
    def respond(call):
        return FakeProvider.error(401 if "other prompt" in call["user"] else 400)

    judge, fake = make_judge(FakeProvider(respond))
    estimate, benign = judge.estimate_usd, []

    def stop_judge():                                        # an auth error in another thread's call
        with contextlib.suppress(FrontierConfigError):
            judge.safety(["other prompt"], ["other response"], ["en"])

    def estimate_then_stop(user, history=()):
        if frontier.PREFLIGHT_PROMPT in user:
            benign.append(user)
            if len(benign) == 2:                             # the probe, right before its reservation
                other = threading.Thread(target=stop_judge)
                other.start()
                other.join(5)
        return estimate(user, history)

    monkeypatch.setattr(judge, "estimate_usd", estimate_then_stop)
    with pytest.raises(FrontierConfigError, match="fatal API error: AuthenticationError 401") as exc:
        judge.safety(["p"], ["r"], ["en"])                   # the 400 is unanswered: the call probes
    assert exc.value.partial == [None] and judge.stats["failures"] == 0
    assert [c["preflight"] for c in fake.calls] == [True, False, False]   # no probe went out
    assert judge.stats["probes"] == 0


FLAGGED = ProviderError("BadRequestError", status=400, code="invalid_prompt",
                        message="Invalid prompt: flagged as possibly violating usage policy (placeholder)")


def flagged_odd_items(outage_from=None):
    """Odd items fail with a per-item 400 (OpenAI's invalid_prompt for a flagged pair), even items get a
    reply; from item ``outage_from`` on, every call gets a 529. The benign pair is answered."""
    def respond(call):
        i = int(call["user"].split("prompt-")[1][:2])
        if outage_from is not None and i >= outage_from:
            return FakeProvider.error(529)
        return FLAGGED if i % 2 else FakeProvider.reply("REFUSAL")

    return FakeProvider(respond)


def test_per_item_errors_are_recorded_on_the_first_rerun_after_a_stop(offline):
    """A stopped call records nothing, and on the rerun its answered items are cache hits, which are no
    replies: its per-item errors come back to back, past the breaker. The answered benign pair tells
    them from an error every request gets, so the first rerun completes and records them."""
    with pytest.raises(frontier.FrontierUnavailable, match="OverloadedError 529"):
        make_judge(flagged_odd_items(outage_from=11))[0].safety(*items(12))
    fake = flagged_odd_items()
    rerun, _ = make_judge(fake)
    out = rerun.safety(*items(12))
    assert [o["error"] for o in out] == [None, f"BadRequestError 400: {FLAGGED.message}"] * 6
    assert len(item_calls(fake)) == 6 and rerun.stats["cache_hits"] == 6     # only the rejected items
    assert rerun.stats["probes"] == 2                        # after the 5th error, and at the end


def sdk_provider(name, status, body):
    """A provider on a real SDK client whose HTTP calls get ``status`` and ``body`` back from an
    in-process transport, so the SDK raises exactly what it raises for that response."""
    sdk = pytest.importorskip(name)
    httpx2 = pytest.importorskip("httpx2")
    http = httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(status, json=body)))
    if name == "anthropic":
        return AnthropicProvider(sdk.Anthropic(api_key="sk-ant-test-dummy", max_retries=0, http_client=http))
    return OpenAIProvider(sdk.OpenAI(api_key="sk-test-dummy", max_retries=0, http_client=http))


QUOTA = "You exceeded your current quota, please check your plan and billing details."
TIER_CAP = ("You have reached your API usage limits: your organization has crossed its monthly API usage "
            "threshold. You will regain access on 2026-12-01 at 00:00 UTC.")


def oa_error(message, kind, code=None, param=None):
    return {"message": message, "type": kind, "param": param, "code": code}


@pytest.mark.parametrize(("name", "status", "error", "kind"), [
    ("anthropic", 400, {"type": "invalid_request_error", "message": SPEND_LIMIT}, "fatal"),
    ("anthropic", 400, {"type": "invalid_request_error", "message": WORKSPACE_LIMIT}, "fatal"),
    ("anthropic", 400, {"type": "invalid_request_error", "message": CREDIT_LOW}, "fatal"),
    ("anthropic", 400, {"type": "invalid_request_error", "message": "messages: placeholder rule"}, "item"),
    ("anthropic", 402, {"type": "billing_error", "message": "placeholder billing problem"}, "fatal"),
    # the tier's monthly spend cap: a 429 without retry-after; retried, then left unjudged (unchanged)
    ("anthropic", 429, {"type": "rate_limit_error", "message": TIER_CAP,
                        "details": {"error_code": "enforced_spend_limit_reached"}}, "retryable"),
    ("openai", 429, oa_error(QUOTA, "insufficient_quota", code="insufficient_quota"), "fatal"),
    ("openai", 402, oa_error("placeholder payment required", "billing"), "fatal"),
    ("openai", 400, oa_error("Invalid 'messages[1].content': placeholder", "invalid_request_error",
                             param="messages"), "item"),
])
def test_real_sdk_errors_keep_the_api_message(name, status, error, kind):
    body = {"error": error}
    if name == "anthropic":
        body = {"type": "error", "error": error, "request_id": "req_test"}
    provider = sdk_provider(name, status, body)
    with pytest.raises(ProviderError) as excinfo:
        provider.complete(system="S", user="placeholder prompt text", model="m", max_tokens=9,
                          temperature=None, effort=None, thinking=None)
    err = excinfo.value
    assert (err.status, err.kind, err.message) == (status, kind, error["message"])
    assert "placeholder prompt text" not in str(err) + err.message            # never the request


# ------------------------------------------- `lift judge` over small cells on the real anthropic SDK --
DISABLED = "This organization has been disabled."          # an account-level 400 the judge does not know
CELL_LANGS = ("en", "hi", "zh", "es", "ar", "th")


class English:
    """A LangID stand-in: every output is confident English."""

    def top(self, text, k=3):
        return [("eng_Latn", 0.99)][:k]


class MessagesApi:
    """Anthropic's Messages API on an in-process transport. Answers until ``n_ok`` item requests were
    answered, then rejects every request with a 400 ``DISABLED``, the benign pair included, as a disabled
    organisation does. ``items`` counts item requests."""

    def __init__(self, n_ok=math.inf):
        self.n_ok, self.items, self.disabled = n_ok, 0, False
        self._lock = threading.Lock()

    def __call__(self, request):
        import httpx2

        benign = frontier.PREFLIGHT_PROMPT in request.content.decode("utf-8")
        with self._lock:
            if not benign:
                self.items += 1
            self.disabled = self.disabled or self.items > self.n_ok
            disabled = self.disabled
        if disabled:
            return httpx2.Response(400, json={"type": "error", "request_id": "req_test", "error": {
                "type": "invalid_request_error", "message": DISABLED}})
        text = FakeProvider.reply("BENIGN" if benign else "REFUSAL", "ok")
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": SONNET, "stop_sequence": None,
            "stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 5}})


def write_cells(per_cell=3):
    """One generations file of ``per_cell`` records per language (the size of a `--keys` or `--sample`
    cell); returns (path, prompt texts by pid) per cell. Placeholder texts only."""
    from lift.io import sha256_text, write_jsonl
    from lift.models.generate import generations_path
    from lift.schemas import GenerationRecord

    cells = []
    for lang in CELL_LANGS:
        texts = {f"prh-test-{i:04d}": f"placeholder prompt {lang} {i}" for i in range(per_cell)}
        path = generations_path("stock", "harmful", lang, "dev")
        write_jsonl(path, [GenerationRecord(
            key=sha256_text(f"{lang}-{pid}"), variant="stock", pid=pid, lang=lang, set="harmful", split="dev",
            prompt_sha=sha256_text(text), output=f"placeholder response {lang} {pid}", n_prompt_tokens=10,
            n_new_tokens=12, hit_cap=False, seconds=0.1, meta={"dtype": "bf16"}).model_dump()
            for pid, text in texts.items()])
        cells.append((path, texts))
    return cells


def lift_judge(api, cells):
    """`lift judge --judge second` as the CLI runs it: one judge on the real anthropic client (default
    workers), judge_file per cell. Returns the error that stopped it, None when every cell finished."""
    anthropic, httpx2 = pytest.importorskip("anthropic"), pytest.importorskip("httpx2")
    from lift.judge.pipeline import FrontierAdapter, judge_file

    client = anthropic.Anthropic(api_key="sk-ant-test-dummy", max_retries=0,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(api)))
    adapter = FrontierAdapter(FrontierJudge("anthropic", SONNET, RUBRIC, client=client, config_entry=None))
    for path, texts in cells:
        try:
            judge_file(path, judge=adapter, lid=English(), prompts_by_pid=texts)
        except (FrontierConfigError, frontier.FrontierUnavailable) as exc:
            return exc
    return None


def judged_rows():
    from lift.io import read_jsonl
    from lift.paths import judgements_dir

    paths = sorted((judgements_dir() / "frontier" / "stock").glob("*.jsonl"))
    return [row for path in paths for row in read_jsonl(path)]


@pytest.mark.parametrize("n_ok", [0, 1, 2, 3, 4, 5, 7])
def test_an_unknown_account_error_in_small_cells_is_never_recorded(monkeypatch, n_ok):
    """The organisation is disabled mid-run (an account-level 400 the judge does not know) while `lift
    judge` works through cells of 3 keys, so no call reaches the breaker. Each call probes before it
    records, the probe fails too, and nothing becomes a judge failure; once the account works again, a
    rerun judges exactly what was left."""
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "50")
    monkeypatch.setattr("lift.manifest.git_sha", lambda: "test")
    cells = write_cells()
    stop = lift_judge(MessagesApi(n_ok), cells)
    assert isinstance(stop, frontier.FrontierUnavailable) and DISABLED in str(stop)
    rows = judged_rows()
    assert len(rows) == n_ok // 3 * 3 and not any(r["native"].get("judge_failure") for r in rows)

    api = MessagesApi()
    assert lift_judge(api, cells) is None
    rows = judged_rows()
    assert len(rows) == 18 and not any(r["native"].get("judge_failure") for r in rows)
    assert api.items == 18 - n_ok                            # every answered request was cached


# ------------------------------------------------------------------------ cache file repairs --
@pytest.mark.parametrize("cut", ["mid-row", "newline only"])
def test_the_first_append_repairs_a_cut_last_line(cut):
    judge, _ = make_judge([FakeProvider.reply("BENIGN"), FakeProvider.reply("REFUSAL")])
    first = judge.safety(*items(2))
    path = judge.cache_path
    data = path.read_bytes()
    damaged = data + b'{"key": "abc", "text": "trunc' if cut == "mid-row" else data[:-1]
    path.write_bytes(damaged)

    reader, nothing = make_judge(FakeProvider())
    assert reader.safety(*items(2)) == first and nothing.calls == []
    assert path.read_bytes() == damaged                      # a run that only reads changes nothing

    writer, _ = make_judge([FakeProvider.reply("HARMFUL")])
    assert writer.safety(*items(3))[:2] == first
    assert path.read_bytes().endswith(b"\n")
    rows = [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1]]   # nothing glued
    assert [r["text"] for r in rows] == [FakeProvider.reply(x) for x in ("BENIGN", "REFUSAL", "HARMFUL")]


# ----------------------------------------------------------------------- identical requests --
def overlapping(labels):
    """Answers item calls with ``labels`` in turn. The first call waits up to 0.5 s for a second one,
    so two callers that both missed the cache really overlap."""
    lock, second, seen = threading.Lock(), threading.Event(), []

    def respond(call):
        with lock:
            n = len(seen)
            seen.append(call)
        if n == 0:
            second.wait(0.5)
        else:
            second.set()
        return FakeProvider.reply(labels[n])

    return FakeProvider(respond)


def test_identical_items_in_one_call_are_judged_once():
    fake = overlapping(["HARMFUL", "REFUSAL"])
    judge, _ = make_judge(fake, workers=2)
    same = (["same prompt"] * 3, ["same reply"] * 3, ["en"] * 3)
    first = judge.safety(*same)
    assert len(item_calls(fake)) == 1
    assert first[0] == first[1] == first[2] and first[0] is not first[1]
    assert judge.stats["duplicates"] == 2
    rerun, nothing = make_judge(FakeProvider())
    assert rerun.safety(*same) == first and nothing.calls == []


def test_concurrent_callers_never_both_pay_for_one_request():
    fake = overlapping(["HARMFUL", "REFUSAL"])
    judge, _ = make_judge(fake)
    out = [None, None]

    def run(i):
        out[i] = judge.safety(["same prompt"], ["same reply"], ["en"])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(item_calls(fake)) == 1 and out[0] == out[1]
    assert judge.stats["cache_hits"] == 1


# ----------------------------------------------------------------- api version and model echo --
def test_the_judge_id_names_the_api_version():
    class NewerApi(FakeProvider):
        api_version = "2027-01-15"

    old, new = make_judge()[0], make_judge(NewerApi())[0]
    user = old.render("p", "r", "en")
    assert old._key(user, ()) != new._key(user, ())
    assert old.judge_id.endswith("+apifake") and new.judge_id.endswith("+api2027-01-15")


def test_a_reply_from_another_model_stops_the_run():
    fake = FakeProvider([FakeProvider.reply("BENIGN")], model_echo="claude-sonnet-4-6")
    judge, _ = make_judge(fake)
    with pytest.raises(FrontierConfigError, match="'claude-sonnet-4-6', not 'claude-haiku-5-5'"):
        judge.safety(["p"], ["r"], ["en"])                   # caught by the preflight
    assert item_calls(fake) == []

    moved = ProviderResult(text=FakeProvider.reply("BENIGN"), input_tokens=1, output_tokens=1,
                           stop_reason="end_turn", refused=False, refusal_reason=None,
                           model_echo="claude-haiku-5-6")
    judge, fake = make_judge([FakeProvider.reply("BENIGN"), moved, FakeProvider.reply("BENIGN")])
    with pytest.raises(FrontierConfigError, match="claude-haiku-5-6") as excinfo:
        judge.safety(*items(3))                              # an alias that moves mid-run
    assert [r is not None for r in excinfo.value.partial] == [True, False, False]
    assert len(judge.cache_path.read_text(encoding="utf-8").splitlines()) == 1   # never cached


@pytest.mark.parametrize(("provider", "model", "echo", "ok"), [
    ("anthropic", HAIKU, HAIKU, True),
    ("openai", "gpt-test", "gpt-test-2025-04-14", True),                 # OpenAI's dated snapshot
    ("anthropic", "claude-haiku-4-5", "claude-haiku-4-5-20251001", True),  # Anthropic's
    ("openai", "gpt-test", "gpt-test-mini", False),
    ("openai", "gpt-test", "gpt-test-2025-04", False),
    ("openai", "gpt-test-2025-04-14", "gpt-test", False),
    ("anthropic", HAIKU, "claude-haiku-5-5-latest", False),
])
def test_the_model_echo_is_checked_and_returned(provider, model, echo, ok):
    fake = FakeProvider([FakeProvider.refusal(), FakeProvider.reply("BENIGN")], model_echo=echo)
    judge, _ = make_judge(fake, provider=provider, model=model)
    if not ok:
        with pytest.raises(FrontierConfigError, match="dated snapshot"):
            judge.safety(["p"], ["r"], ["en"])
        return
    refused, benign = judge.safety(["p", "q"], ["r", "s"], ["en", "en"])
    assert refused["model_echo"] == benign["model_echo"] == echo and benign["label"] == "BENIGN"


def test_the_spend_cap_says_it_caps_one_process(monkeypatch):
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "1e-9")
    judge, _ = make_judge(provider="openai", model="gpt-test")
    with pytest.raises(SpendCapExceeded, match="LIFT_JUDGE_MAX_USD caps one lift process"):
        judge.safety(["p"], ["r"], ["en"])
    assert "LIFT_JUDGE_MAX_USD caps one lift process" in frontier.__doc__
