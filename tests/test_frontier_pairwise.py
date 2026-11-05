"""The frontier judge's rubric kinds and its pairwise (utility) judging: the rubric's placeholders decide
its kind, pairwise() runs the same machinery as safety() with the winner schema, and the safety judge's
id and cache keys are exactly what they were before pairwise judging existed.

Offline only, on FakeProvider; placeholder texts only.
"""

from __future__ import annotations

import json
import logging
import threading

import pytest

from lift.io import sha256_file
from lift.judge import frontier
from lift.judge.frontier import (
    PAIRWISE_REMINDER,
    PREFLIGHT_PROMPT,
    PREFLIGHT_RESPONSE,
    PREFLIGHT_WORSE,
    REMINDER,
    FakeProvider,
    FrontierConfigError,
    FrontierJudge,
    ProviderError,
    load_rubric,
    parse_pairwise_reply,
)
from lift.paths import REPO_ROOT

SAFETY = "configs/rubrics/safety_v1.md"
UTILITY = "configs/rubrics/utility_v1.md"
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


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


def make_pairwise(script=(), **kw):
    fake = script if isinstance(script, FakeProvider) else FakeProvider(script)
    kw.setdefault("workers", 1)
    return FrontierJudge("anthropic", HAIKU, UTILITY, client=fake, config_entry=None, **kw), fake


def make_safety(script=(), **kw):
    fake = script if isinstance(script, FakeProvider) else FakeProvider(script)
    kw.setdefault("workers", 1)
    return FrontierJudge("anthropic", HAIKU, SAFETY, client=fake, config_entry=None, **kw), fake


def winner(w, reason="test reason"):
    return json.dumps({"winner": w, "reason": reason})


def pairs(n):
    return ([f"instruction-{i:02d}" for i in range(n)], [f"answer-a-{i:02d}" for i in range(n)],
            [f"answer-b-{i:02d}" for i in range(n)])


def item_calls(fake):
    return [c for c in fake.calls if not c["preflight"]]


# ---------------------------------------------------------------------------------- rubric kinds --
def test_the_safety_rubric_is_unchanged():
    """Its sections, sha, judge id and cache keys are byte for byte what they were before rubric kinds
    existed (the keys were computed by the code before this change): every cached safety reply stays
    valid and judge_file keeps resuming its files."""
    rubric = load_rubric(SAFETY)
    text = (REPO_ROOT / SAFETY).read_text(encoding="utf-8")
    system, template = text.split("## System message", 1)[1].split("## User message template", 1)
    assert rubric.kind == "safety"
    assert (rubric.system, rubric.user_template) == (system.strip(), template.strip())
    assert rubric.sha256 == sha256_file(REPO_ROOT / SAFETY)
    judge, _ = make_safety()
    assert judge.judge_id == (f"anthropic/{HAIKU}@safety_v1:529fccb10401+provider-default+effort-low+max1024"
                              "+apifake")
    user = judge.render("placeholder prompt", "placeholder response", "hi")
    assert judge._key(user, ()) == "a526420676e6a9c01bdef02423ab6b54789321d9bc1e170b504cf9c43443b2a1"
    history = ({"role": "user", "content": user}, {"role": "assistant", "content": "not json"})
    assert judge._key(REMINDER, history) == "abf1bb991f27e991b213c70d686120d19b2313f290c4c06311895a1ecaa41b5d"
    assert judge._key(judge._benign_user(), ()) == (
        "4a0b357582bbabd2ab03bd05678f3973de10839234661df27f722428acdc2ed3")


def test_the_utility_rubric_loads_as_pairwise():
    rubric = load_rubric(UTILITY)
    text = (REPO_ROOT / UTILITY).read_text(encoding="utf-8")
    assert rubric.kind == "pairwise"
    assert rubric.version == "utility_v1:821bed4ebfe8" and rubric.sha256 == sha256_file(REPO_ROOT / UTILITY)
    assert rubric.user_template == text.split("## User message template", 1)[1].strip()
    assert rubric.system.endswith('{"winner": "A" | "B" | "TIE", "reason": "<at most 20 English words>"}')
    out = rubric.render_pair(instruction="INSTR", response_a="ANSWER-A", response_b="ANSWER-B")
    assert out == ("INSTRUCTION:\n<<<\nINSTR\n>>>\n\nRESPONSE A:\n<<<\nANSWER-A\n>>>\n\n"
                   "RESPONSE B:\n<<<\nANSWER-B\n>>>")


@pytest.mark.parametrize("template", [
    "{instruction} {response_a} {response}",                  # a mix of both kinds
    "{instruction} {response_a}",                             # a field missing
    "{instruction} {response_a} {response_b} {prompt}",       # one too many
    "{prompt} {response} {prompt_lang} {response_a}",
])
def test_a_template_of_neither_kind_fails_loudly(tmp_path, template):
    path = tmp_path / "rubric.md"
    path.write_text(f"# x\n\n## System message\n\nJudge.\n\n## User message template\n\n{template}\n",
                    encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="a safety rubric.*a pairwise rubric"):
        load_rubric(path)


def test_render_pair_is_a_single_pass():
    rubric = load_rubric(UTILITY)
    tricky = r"see {instruction} and {response_b} and {response_a} \1 \g<0>"
    out = rubric.render_pair(instruction="QUESTION-TEXT", response_a=tricky, response_b="{response_a}")
    assert f"RESPONSE A:\n<<<\n{tricky}\n>>>" in out and "RESPONSE B:\n<<<\n{response_a}\n>>>" in out
    assert out.count("QUESTION-TEXT") == 1


def test_each_rubric_is_filled_with_its_own_fields_only():
    with pytest.raises(ValueError, match="pairwise rubric.*render_pair"):
        load_rubric(UTILITY).render(prompt="p", response="r", prompt_lang="Hindi")
    with pytest.raises(ValueError, match="safety rubric.*render"):
        load_rubric(SAFETY).render_pair(instruction="i", response_a="a", response_b="b")


# ---------------------------------------------------------------------------------------- parser --
GOOD = '{"winner": "B", "reason": "B follows the format"}'


@pytest.mark.parametrize(("text", "expected"), [
    (GOOD, ("B", "B follows the format")),
    ("  \n" + GOOD + "\n ", ("B", "B follows the format")),
    ('{"reason": "", "winner": "TIE"}', ("TIE", "")),
    ('{"winner":"A","reason":"उत्तर A सही है"}', ("A", "उत्तर A सही है")),
])
def test_the_pairwise_parser_accepts_exactly_one_object(text, expected):
    assert parse_pairwise_reply(text) == expected


@pytest.mark.parametrize("text", [
    "```json\n" + GOOD + "\n```",                                    # code fence
    '{"winner": "A", "reason": "x", "confidence": 0.9}',             # extra key
    '{"winner": "a", "reason": "x"}',                                # lowercase winner
    '{"winner": "tie", "reason": "x"}',
    '{"winner": "Tie", "reason": "x"}',
    '{"winner": "A"}',                                               # missing reason
    '{"winner": "A", "reason": null}',                               # reason not a string
    '{"winner": "C", "reason": "x"}',                                # unknown winner
    '{"winner": ["A"], "reason": "x"}',
    '{"winner": "A", "winner": "B", "reason": "x"}',                 # duplicate key
    '{"label": "BENIGN", "reason": "x"}',                            # the safety schema
    GOOD + GOOD,                                                     # two objects
    "Here is my verdict: " + GOOD,                                   # prose before
    GOOD + " I hope this helps.",                                    # prose after
    "[" + GOOD + "]",
    '"A"',
    "A",
    "",
])
def test_the_pairwise_parser_rejects_everything_else(text):
    assert parse_pairwise_reply(text) is None


def test_the_pairwise_reminder_mirrors_the_safety_one():
    lead = REMINDER.split("{", 1)[0]
    assert PAIRWISE_REMINDER.startswith(lead)
    schema = '{"winner": "A" | "B" | "TIE", "reason": "<at most 20 English words>"}'
    assert PAIRWISE_REMINDER == lead + schema
    assert load_rubric(UTILITY).system.endswith(schema)       # the schema the rubric itself asks for


# --------------------------------------------------------------------------------- pairwise judging --
def test_pairwise_returns_one_verdict_per_pair_in_input_order():
    judge, fake = make_pairwise([winner("A", "a is right"), winner("TIE", "both fine"), winner("B")])
    out = judge.pairwise(*pairs(3))
    assert out == [
        {"winner": "A", "reason": "a is right", "raw": winner("A", "a is right"), "error": None,
         "model_echo": HAIKU},
        {"winner": "TIE", "reason": "both fine", "raw": winner("TIE", "both fine"), "error": None,
         "model_echo": HAIKU},
        {"winner": "B", "reason": "test reason", "raw": winner("B"), "error": None, "model_echo": HAIKU}]
    first = item_calls(fake)[0]
    assert first["system"] == load_rubric(UTILITY).system and first["history"] == []
    assert first["user"] == judge.render_pair("instruction-00", "answer-a-00", "answer-b-00")
    assert judge.judge_id == (f"anthropic/{HAIKU}@utility_v1:821bed4ebfe8+provider-default+effort-low+max1024"
                              "+apifake")


def test_the_pairwise_preflight_sends_its_own_benign_pair():
    """The question, its right answer as A and a non-answer as B; FakeProvider's default reply fits the
    winner schema, so a pairwise judge needs no preflight script."""
    judge, fake = make_pairwise([winner("A")])
    judge.pairwise(*pairs(1))
    [preflight] = [c for c in fake.calls if c["preflight"]]
    assert preflight["user"] == judge.render_pair(PREFLIGHT_PROMPT, PREFLIGHT_RESPONSE, PREFLIGHT_WORSE)
    assert judge._key(preflight["user"], ()) not in {judge._key(c["user"], ()) for c in item_calls(fake)}


def test_a_preflight_that_does_not_pick_a_only_warns(caplog):
    caplog.set_level(logging.WARNING, logger=frontier.__name__)
    judge, _ = make_pairwise(FakeProvider([winner("A")], preflight=winner("B")))
    assert judge.pairwise(*pairs(1))[0]["winner"] == "A"
    assert any("benign pair got winner B, not A" in r.getMessage() for r in caplog.records)


def test_a_safety_shaped_preflight_reply_stops_a_pairwise_judge():
    fake = FakeProvider([], preflight=FakeProvider.reply("BENIGN"))
    judge, _ = make_pairwise(fake)
    with pytest.raises(FrontierConfigError, match="preflight failed: unparseable judge output"):
        judge.pairwise(*pairs(1))
    assert [c["preflight"] for c in fake.calls] == [True, True]          # the reply and its reminder
    assert fake.calls[1]["user"] == PAIRWISE_REMINDER


def test_a_parse_failure_gets_the_pairwise_reminder_then_succeeds():
    reply = winner("A", "a is right")
    judge, fake = make_pairwise(["Response A is better.", reply])
    assert judge.pairwise(*pairs(1))[0] == {"winner": "A", "reason": "a is right", "raw": reply,
                                            "error": None, "model_echo": HAIKU}
    first, second = item_calls(fake)
    assert second["user"] == PAIRWISE_REMINDER
    assert second["history"] == [{"role": "user", "content": first["user"]},
                                 {"role": "assistant", "content": "Response A is better."}]
    assert (judge.stats["reminders"], judge.stats["failures"]) == (1, 0)


def test_a_second_parse_failure_is_a_judge_failure_not_a_guess():
    lowercase = '{"winner": "a", "reason": "x"}'
    judge, _ = make_pairwise(["A", lowercase])
    assert judge.pairwise(*pairs(1)) == [{"winner": None, "reason": None, "raw": lowercase,
                                          "error": "unparseable judge output", "model_echo": HAIKU}]
    assert judge.stats["failures"] == 1


def test_a_refusal_is_a_failure_with_its_reason():
    judge, fake = make_pairwise([FakeProvider.refusal("general_harms: policy"), winner("B")])
    refused, judged = judge.pairwise(*pairs(2))
    assert refused == {"winner": None, "reason": None, "raw": "", "error": "refused: general_harms: policy",
                       "model_echo": HAIKU}
    assert judged["winner"] == "B"
    assert len(item_calls(fake)) == 2                                   # no reminder after a refusal
    assert (judge.stats["refusals"], judge.stats["failures"]) == (1, 1)


def test_an_item_level_api_error_is_recorded_after_the_benign_pair_is_answered(offline):
    """No reply came in after the errors, so the call sends the pairwise benign pair again (a probe)
    before it records them."""
    rejected = ProviderError("BadRequestError", status=400, message="placeholder rule")
    judge, fake = make_pairwise([rejected, rejected])
    out = judge.pairwise(*pairs(2))
    assert out == [{"winner": None, "reason": None, "raw": "", "model_echo": None,
                    "error": "BadRequestError 400: placeholder rule"}] * 2
    assert offline == [] and judge.stats["failures"] == 2
    probes = [c for c in fake.calls if c["preflight"]][1:]
    assert len(probes) == judge.stats["probes"] == 1
    assert probes[0]["user"] == judge.render_pair(PREFLIGHT_PROMPT, PREFLIGHT_RESPONSE, PREFLIGHT_WORSE)


def test_identical_pairs_are_judged_once_and_the_two_orders_are_two_requests():
    judge, fake = make_pairwise([winner("A"), winner("B")])
    out = judge.pairwise(["same instruction"] * 3, ["one", "one", "two"], ["two", "two", "one"])
    assert [o["winner"] for o in out] == ["A", "A", "B"]
    assert len(item_calls(fake)) == 2 and judge.stats["duplicates"] == 1
    assert out[0] == out[1] and out[0] is not out[1]


def test_a_cached_verdict_is_reused_by_a_new_judge():
    judge, _ = make_pairwise([winner("A"), FakeProvider.refusal(), winner("TIE")])
    first = judge.pairwise(*pairs(3))
    nothing = FakeProvider()                                            # any call would fail the test
    again, _ = make_pairwise(nothing)
    assert again.pairwise(*pairs(3)) == first and nothing.calls == []
    assert again.stats["cache_hits"] == 3


def test_pairwise_keeps_input_order_with_many_workers():
    pause = threading.Event()

    def respond(call):
        i = int(call["user"].split("instruction-")[1][:2])
        pause.wait((i * 7 % 5) / 1000)                                  # finish out of order
        return winner(("A", "B", "TIE")[i % 3], f"pair {i}")

    judge, _ = make_pairwise(FakeProvider(respond), workers=8)
    out = judge.pairwise(*pairs(30))
    expected = [(("A", "B", "TIE")[i % 3], f"pair {i}") for i in range(30)]
    assert [(o["winner"], o["reason"]) for o in out] == expected


def test_pairwise_checks_its_inputs_before_any_call():
    judge, fake = make_pairwise()
    with pytest.raises(ValueError, match="differ in length"):
        judge.pairwise(["i", "j"], ["a"], ["b"])
    with pytest.raises(TypeError):
        judge.pairwise(["i"], [None], ["b"])
    assert judge.pairwise([], [], []) == [] and fake.calls == []


# ---------------------------------------------------------------------------------- kind refusals --
def test_a_judge_serves_its_rubrics_kind_only():
    safety, safety_fake = make_safety()
    pairwise, pairwise_fake = make_pairwise()
    for call in (lambda: safety.pairwise(["i"], ["a"], ["b"]),
                 lambda: safety.prefill_pairwise(["i"], ["a"], ["b"]),
                 lambda: safety.render_pair("i", "a", "b")):
        with pytest.raises(ValueError, match="needs a pairwise rubric, but safety_v1.md is a safety rubric"):
            call()
    for call in (lambda: pairwise.safety(["p"], ["r"], ["hi"]),
                 lambda: pairwise.prefill_safety(["p"], ["r"], ["hi"]),
                 lambda: pairwise.render("p", "r", "hi")):
        with pytest.raises(ValueError, match="needs a safety rubric, but utility_v1.md is a pairwise rubric"):
            call()
    assert safety_fake.calls == [] and pairwise_fake.calls == []


def test_the_utility_entry_builds_a_pairwise_judge():
    """judges.utility in the committed config (anthropic, claude-sonnet-5-5, utility_v1.md, bulk batch)."""
    judge = FrontierJudge.from_config("utility", client=FakeProvider([winner("B")]))
    assert (judge.rubric.kind, judge.model, judge.bulk) == ("pairwise", SONNET, "batch")
    assert judge.judge_id == (f"anthropic/{SONNET}@utility_v1:821bed4ebfe8+provider-default+effort-low"
                              "+max1024+apifake")
    assert judge.pairwise(*pairs(1))[0]["winner"] == "B"


def test_pairwise_and_safety_keys_never_collide_in_one_cache_file():
    """Both judges of one provider and model share the default cache file; the rubric sha in every key
    keeps their rows apart."""
    safety, _ = make_safety([FakeProvider.reply("BENIGN")])
    pairwise, _ = make_pairwise([winner("A")])
    assert safety.cache_path == pairwise.cache_path
    safety.safety(["instruction-00"], ["answer-a-00"], ["en"])
    pairwise.pairwise(*pairs(1))
    rows = [json.loads(line) for line in safety.cache_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({r["key"] for r in rows}) == 2
