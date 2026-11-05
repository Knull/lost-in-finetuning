"""The frontier judge's bulk pre-fill through the Message Batches API: prefill_pairwise / prefill_safety
put a whole run's first turns into the call cache, so the synchronous pass that follows makes no call for
them and returns what a synchronous run returns. Covers the capability gate, the batch state file and
resume, what a 404 for an earlier batch means and LIFT_ABANDON_BATCHES, the spend reservation, costing
from usage, failures, and the real anthropic SDK path.

Offline only: FakeBatchProvider, or a real anthropic client on an in-process transport. Placeholder
texts only.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from lift import config
from lift.io import append_jsonl
from lift.judge import frontier
from lift.judge.frontier import (
    PAIRWISE_REMINDER,
    SYSTEM_CACHE_CONTROL,
    FakeBatchProvider,
    FakeProvider,
    FrontierBatchPending,
    FrontierConfigError,
    FrontierJudge,
    FrontierUnavailable,
    ProviderError,
    ProviderResult,
    SpendCapExceeded,
)

SAFETY = "configs/rubrics/safety_v1.md"
UTILITY = "configs/rubrics/utility_v1.md"
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
SPEND_LIMIT = ("You have reached your specified API usage limits. You will regain access on 2026-12-01 at "
               "00:00 UTC.")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No key, no spend cap, no reachable endpoint, no real sleeping. Returns the recorded sleeps."""
    for var in ("LIFT_JUDGE_MAX_USD", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "OPENAI_LOG", "ANTHROPIC_LOG", "LIFT_ABANDON_BATCHES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    sleeps: list[float] = []
    monkeypatch.setattr(frontier.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture
def utility_entry(monkeypatch):
    """Replace judges.utility in the loaded project config by the committed one plus settings."""
    real = config.load
    base = copy.deepcopy(real("project"))

    def set_entry(**settings):
        cfg = copy.deepcopy(base)
        cfg["judges"]["utility"].update(settings)
        monkeypatch.setattr(config, "load", lambda name="project": cfg if name == "project" else real(name))

    return set_entry


def winner(w, reason="test reason"):
    return json.dumps({"winner": w, "reason": reason})


def pairs(n, start=0):
    idx = range(start, start + n)
    return ([f"instruction-{i:02d}" for i in idx], [f"answer-a-{i:02d}" for i in idx],
            [f"answer-b-{i:02d}" for i in idx])


def index(call):
    """The pair a call is about (a reminder turn: the pair of its first turn)."""
    text = call["user"] if "instruction-" in call["user"] else call["history"][0]["content"]
    return int(text.split("instruction-")[1][:2])


def verdicts(call):
    i = index(call)
    return winner(("A", "B", "TIE")[i % 3], f"pair {i}")


def batch_judge(fake, *, rubric=UTILITY, entry="utility", model=HAIKU, **kw):
    """A judge on the committed config entry (bulk: batch) that batches from two uncached requests."""
    kw.setdefault("workers", 1)
    kw.setdefault("batch_min_items", 2)
    return FrontierJudge("anthropic", model, rubric, client=fake, config_entry=entry, **kw)


def sync_reference(script, *, rubric=UTILITY):
    """The same judge without batches, on its own cache file: what a synchronous run returns."""
    return FrontierJudge("anthropic", HAIKU, rubric, client=FakeProvider(script), config_entry=None,
                         workers=1, cache_path="judge_cache/frontier/sync-reference.jsonl")


def item_calls(fake):
    return [c for c in fake.calls if not c["preflight"]]


def preflight_usd(**kw):
    """What a batch judge's synchronous preflight costs (FakeProvider's default reply, bytes / 4 tokens)."""
    reference = batch_judge(FakeProvider(), cache_path="judge_cache/frontier/preflight-reference.jsonl", **kw)
    reference._ensure_preflight()
    return reference.stats["usd"]


def state_rows(judge):
    return [json.loads(line) for line in judge.batch_state_path.read_text(encoding="utf-8").splitlines()]


def cache_rows(judge):
    return [json.loads(line) for line in judge.cache_path.read_text(encoding="utf-8").splitlines()]


def open_batch(n=3, *, polls=5):
    """A batch of the first ``n`` pairs that an earlier process submitted and left running (a pre-fill with
    wait=False): its provider and id. It reads in_progress for ``polls`` looks, the submitter's included."""
    fake = FakeBatchProvider(verdicts, polls=polls)
    with pytest.raises(FrontierBatchPending):
        batch_judge(fake).prefill_pairwise(*pairs(n), wait=False)
    [batch_id] = fake.batches
    return fake, batch_id


def days_ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")


def recorded_batch(judge, batch_id, created, n=3):
    """Record in ``judge``'s state file a batch of the first ``n`` pairs that an earlier process of this judge
    submitted at ``created`` (None: no time recorded). A fresh provider's API does not know it (a 404)."""
    keys = [judge._key(judge.render_pair(*p), ()) for p in zip(*pairs(n), strict=True)]
    row = {"key": f"batch:{batch_id}:submitted", "event": "submitted", "batch_id": batch_id,
           "judge": judge.judge_id, "api_version": judge.api_version, "keys": keys, "n": n}
    append_jsonl(judge.batch_state_path, [row if created is None else {**row, "created": created}])


def status_looks(fake):
    """The batch ids ``fake.batch_status`` is asked about from now on, a 404 included."""
    looked: list[str] = []
    status = fake.batch_status

    def spy(batch_id):
        looked.append(batch_id)
        return status(batch_id)

    fake.batch_status = spy
    return looked


# --------------------------------------------------------------------------------- the main path --
def test_prefill_fills_the_cache_and_the_sync_pass_makes_no_call(offline):
    fake = FakeBatchProvider(verdicts, polls=1)
    judge = batch_judge(fake)
    summary = judge.prefill_pairwise(*pairs(5))
    [batch_id] = fake.batches
    assert summary == {"mode": "batch", "reason": None, "n_requests": 5, "n_cached": 0, "n_submitted": 5,
                       "n_collected": 5, "n_errored": 0, "n_expired": 0, "n_canceled": 0,
                       "batch_ids": [batch_id], "usd": round(judge.stats["usd"], 6)}
    assert [c["preflight"] for c in fake.calls] == [True]               # the synchronous preflight only
    assert offline == [30.0]                                            # one poll interval
    users = [judge.render_pair(*p) for p in zip(*pairs(5), strict=True)]
    requests = fake.batches[batch_id]
    assert [r["custom_id"] for r in requests] == [judge._key(u, ()) for u in users]
    assert all(r["user"] == u and r["system"] == judge.rubric.system and r["history"] == ()
               and (r["model"], r["max_tokens"], r["effort"], r["thinking"], r["temperature"]) ==
               (HAIKU, 1024, "low", None, None) for r, u in zip(requests, users, strict=True))

    out = judge.pairwise(*pairs(5))
    assert len(fake.calls) == 1 and judge.stats["cache_hits"] == 5
    assert out == sync_reference(verdicts).pairwise(*pairs(5))          # identical judgements
    rows = cache_rows(judge)
    assert len(rows) == 5 and {(r["via"], r["batch_id"]) for r in rows} == {("batch", batch_id)}


def test_prefill_safety_then_safety_matches_the_sync_path():
    prompts, responses = [f"prompt-{i:02d}" for i in range(4)], [f"response-{i:02d}" for i in range(4)]
    langs = ["hi", "en", "zh", "es"]

    def labels(call):
        return FakeProvider.reply(("REFUSAL", "BENIGN")[int(call["user"].split("prompt-")[1][:2]) % 2])

    fake = FakeBatchProvider(labels)
    judge = batch_judge(fake, rubric=SAFETY, entry="safety_second")
    assert judge.prefill_safety(prompts, responses, langs)["n_collected"] == 4
    out = judge.safety(prompts, responses, langs)
    assert item_calls(fake) == []
    assert out == sync_reference(labels, rubric=SAFETY).safety(prompts, responses, langs)


def test_only_distinct_uncached_first_turns_are_submitted():
    seeded = FrontierJudge("anthropic", HAIKU, UTILITY, client=FakeProvider(verdicts), config_entry=None,
                           workers=1)
    seeded.pairwise(*pairs(2))                                         # pairs 0 and 1 are cached
    fake = FakeBatchProvider(verdicts)
    judge = batch_judge(fake)
    instructions, a, b = pairs(6)
    summary = judge.prefill_pairwise(instructions + instructions[3:4], a + a[3:4], b + b[3:4])
    assert (summary["n_requests"], summary["n_cached"], summary["n_submitted"]) == (6, 2, 4)
    [requests] = fake.batches.values()
    uncached = [judge._key(judge.render_pair(*p), ()) for p in zip(*pairs(4, start=2), strict=True)]
    assert [r["custom_id"] for r in requests] == uncached


def test_an_unparseable_first_turn_gets_its_reminder_synchronously():
    def first_unparseable(call):
        if call.get("batch") and index(call) == 1:
            return "Response A is better."
        return verdicts(call)

    fake = FakeBatchProvider(first_unparseable)
    judge = batch_judge(fake)
    judge.prefill_pairwise(*pairs(3))
    out = judge.pairwise(*pairs(3))
    [reminder] = item_calls(fake)
    assert reminder["user"] == PAIRWISE_REMINDER
    assert reminder["history"][1] == {"role": "assistant", "content": "Response A is better."}
    assert out[1] == {"winner": "B", "reason": "pair 1", "raw": winner("B", "pair 1"), "error": None,
                      "model_echo": HAIKU}
    assert judge.stats["reminders"] == 1


# ------------------------------------------------------------------------------------- no-op gates --
@pytest.mark.parametrize("case", ["bulk unset", "bulk sync", "no batch API", "below batch_min_items",
                                  "all cached"])
def test_prefill_does_nothing_when_batches_do_not_apply(case, utility_entry):
    fake = FakeBatchProvider(verdicts)
    kw = {}
    if case == "bulk unset":
        kw["entry"] = None
    elif case == "bulk sync":
        utility_entry(bulk="sync")
    elif case == "no batch API":
        fake = FakeProvider(verdicts)
    elif case == "below batch_min_items":
        kw["batch_min_items"] = 20
    else:
        sync_reference(verdicts).pairwise(*pairs(5))
        kw["cache_path"] = "judge_cache/frontier/sync-reference.jsonl"
    judge = batch_judge(fake, **kw)
    summary = judge.prefill_pairwise(*pairs(5))
    reasons = {"bulk unset": "the bulk mode is not set", "bulk sync": "the bulk mode is sync",
               "no batch API": "FakeProvider has no Message Batches API",
               "below batch_min_items": "5 uncached requests, fewer than batch_min_items (20)",
               "all cached": "every request is cached"}
    assert (summary["mode"], summary["reason"], summary["n_submitted"]) == ("sync", reasons[case], 0)
    assert fake.calls == [] and not getattr(fake, "batches", {})        # not even the preflight
    assert not judge.batch_state_path.exists()


def test_a_plain_fake_provider_has_no_batch_api():
    assert not frontier._batch_capable(FakeProvider())
    assert frontier._batch_capable(FakeBatchProvider())


def test_the_note_on_bulk_mode_names_a_provider_without_a_batch_api(caplog):
    caplog.set_level(logging.INFO, logger=frontier.__name__)
    batch_judge(FakeProvider())
    batch_judge(FakeBatchProvider())
    notes = [r.getMessage() for r in caplog.records if "bulk is 'batch'" in r.getMessage()]
    assert len(notes) == 2 and "FakeProvider has no Message Batches API" in notes[0]
    assert "no Message Batches API" not in notes[1]


@pytest.mark.parametrize("settings", [{"batch_min_items": 0}, {"batch_min_items": 2.5}, {"batch_poll_s": 0},
                                      {"batch_poll_s": "30"}, {"batch_wait_s": -1}, {"batch_wait_s": True}])
def test_invalid_batch_settings_fail_loudly(settings):
    with pytest.raises(ValueError, match=next(iter(settings))):
        batch_judge(FakeBatchProvider(), **settings)


def test_batch_settings_come_from_the_config_entry(utility_entry):
    defaults = FrontierJudge.from_config("utility", client=FakeBatchProvider())
    assert (defaults.batch_min_items, defaults.batch_poll_s, defaults.batch_wait_s) == (20, 30.0, 86_400.0)
    utility_entry(batch_min_items=50, batch_poll_s=5, batch_wait_s=600)
    judge = FrontierJudge.from_config("utility", client=FakeBatchProvider())
    assert (judge.batch_min_items, judge.batch_poll_s, judge.batch_wait_s) == (50, 5.0, 600.0)
    with pytest.raises(TypeError, match="unknown FrontierJudge settings"):
        batch_judge(FakeBatchProvider(), bulk="batch")                 # bulk stays a config entry's key


# ---------------------------------------------------------------- results left to the sync pass --
def test_errored_expired_and_canceled_requests_are_left_for_the_sync_pass():
    def batch_trouble(call):                     # errored as AnthropicProvider reports a batch item's error
        if call.get("batch") and index(call) == 1:
            return ProviderError("BatchItemError", status=400, code="invalid_request_error",
                                 message="placeholder: rejected in the batch")
        return verdicts(call)

    fake = FakeBatchProvider(batch_trouble, expire=lambda call: index(call) == 2,
                             cancel=lambda call: index(call) == 3)
    judge = batch_judge(fake)
    summary = judge.prefill_pairwise(*pairs(5))
    counts = [summary[f"n_{name}"] for name in ("collected", "errored", "expired", "canceled")]
    assert counts == [2, 1, 1, 1]
    assert len(cache_rows(judge)) == 2                                  # nothing recorded for the others
    out = judge.pairwise(*pairs(5))
    assert sorted(index(c) for c in item_calls(fake)) == [1, 2, 3]      # sent again, synchronously
    assert out == sync_reference(verdicts).pairwise(*pairs(5)) and judge.stats["failures"] == 0
    [collected] = [r for r in state_rows(judge) if r["event"] == "collected"]
    assert collected["errors"] == {"invalid_request_error": 1}
    assert (collected["n_errored"], collected["n_expired"], collected["n_canceled"]) == (1, 1, 1)


@pytest.mark.parametrize(("fail", "stop", "match"), [
    (ProviderError("BatchItemError", status=401, code="authentication_error", message="placeholder"),
     FrontierConfigError, "configuration or account error"),
    (ProviderError("BatchItemError", status=400, code="invalid_request_error", message=SPEND_LIMIT),
     FrontierConfigError, "configuration or account error.*usage limits"),
    (ProviderError("BatchItemError", status=529, code="overloaded_error", message="placeholder"),
     FrontierUnavailable, "says nothing about the items"),
    (ProviderError("BatchItemError", status=400, code="invalid_request_error", message="placeholder rule"),
     FrontierUnavailable, r"errors by type \{'invalid_request_error': 5\}"),
    ("expired", FrontierUnavailable, "answered none of its 5 requests \\(5 expired"),
])
def test_a_batch_in_which_nothing_succeeded_stops_the_run(fail, stop, match):
    if fail == "expired":
        fake = FakeBatchProvider(verdicts, expire=lambda call: True)
    else:
        fake = FakeBatchProvider(verdicts, fail_all=fail)
    judge = batch_judge(fake)
    with pytest.raises(stop, match=match):
        judge.prefill_pairwise(*pairs(5))
    with pytest.raises(stop):
        judge.pairwise(*pairs(5))                                       # the judge stays stopped
    assert item_calls(fake) == [] and fake.batch_calls == []            # nothing sent one by one
    [submitted, collected] = state_rows(judge)
    assert collected["event"] == "collected" and collected["n_succeeded"] == 0

    healthy = FakeBatchProvider(verdicts)                               # fixed: a rerun submits them again
    assert batch_judge(healthy).prefill_pairwise(*pairs(5))["n_collected"] == 5


def test_a_reply_from_another_model_is_never_cached():
    def moved(call):
        return ProviderResult(text=winner("A"), input_tokens=1, output_tokens=1, stop_reason="end_turn",
                              refused=False, refusal_reason=None, model_echo="claude-haiku-5-6")

    judge = batch_judge(FakeBatchProvider(moved, preflight=winner("A")))
    with pytest.raises(FrontierConfigError, match="'claude-haiku-5-6', not 'claude-haiku-5-5'"):
        judge.prefill_pairwise(*pairs(3))
    assert not judge.cache_path.exists() and judge.stats["calls"] == 2   # the preflight, and the reply paid


# ----------------------------------------------------------------------------------- resume rules --
def test_a_rerun_after_a_crash_collects_the_batch_instead_of_submitting_again():
    fake = FakeBatchProvider(verdicts, polls=1)
    judge = batch_judge(fake)

    def killed(batch_id):
        raise KeyboardInterrupt

    fake.batch_status = killed                                          # Ctrl+C (or a kill) while polling
    with pytest.raises(KeyboardInterrupt):
        judge.prefill_pairwise(*pairs(5))
    del fake.batch_status
    [submitted] = state_rows(judge)
    [batch_id] = fake.batches
    assert (submitted["event"], submitted["batch_id"], submitted["n"]) == ("submitted", batch_id, 5)

    rerun = batch_judge(fake)                                           # a new process
    summary = rerun.prefill_pairwise(*pairs(5))
    assert (summary["n_submitted"], summary["n_collected"], summary["batch_ids"]) == (0, 5, [batch_id])
    assert len(fake.batches) == 1
    assert [c["preflight"] for c in fake.calls] == [True, True]          # the rerun's own, before it looks
    collected = state_rows(rerun)[-1]
    assert collected["event"] == "collected" and collected["n_cached"] == 5 and collected["usd"] > 0
    # the batch was reserved by the crashed process; the rerun pays its preflight and counts the batch
    assert rerun.stats["usd"] == pytest.approx(preflight_usd() + collected["usd"], abs=1e-6)
    assert rerun.pairwise(*pairs(5)) == sync_reference(verdicts).pairwise(*pairs(5))
    assert item_calls(fake) == []


def test_wait_false_raises_pending_then_a_second_prefill_collects(offline):
    fake = FakeBatchProvider(verdicts, polls=1)
    judge = batch_judge(fake)
    with pytest.raises(FrontierBatchPending, match="rerun the same command") as excinfo:
        judge.prefill_pairwise(*pairs(5), wait=False)
    [batch_id] = fake.batches
    pending = excinfo.value
    assert (pending.batch_ids, pending.n_pending) == ([batch_id], 5)
    assert [pending.summary[k] for k in ("mode", "n_submitted", "n_collected")] == ["batch", 5, 0]
    assert offline == [] and judge._reserved > 0                         # nothing waited; still reserved

    summary = judge.prefill_pairwise(*pairs(5), wait=False)              # the rerun finds it ended
    assert (summary["n_submitted"], summary["n_collected"]) == (0, 5) and len(fake.batches) == 1
    assert judge._reserved == pytest.approx(0, abs=1e-12)                # every reservation settled


def test_polling_logs_a_heartbeat_per_look(offline, caplog):
    caplog.set_level(logging.INFO, logger=frontier.__name__)
    fake = FakeBatchProvider(verdicts, polls=3)
    batch_judge(fake, batch_poll_s=7).prefill_pairwise(*pairs(3))
    [batch_id] = fake.batches
    beats = [r for r in caplog.records if batch_id in r.getMessage() and "processing" in r.getMessage()]
    assert len(beats) == 3 and offline == [7.0] * 3 and fake.polled[batch_id] == 4


def test_the_wait_gives_up_after_batch_wait_s(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(frontier.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(frontier.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    fake = FakeBatchProvider(verdicts, polls=100)
    judge = batch_judge(fake, batch_poll_s=30, batch_wait_s=90)
    with pytest.raises(FrontierBatchPending):
        judge.prefill_pairwise(*pairs(3))
    [batch_id] = fake.batches
    assert fake.polled[batch_id] == 4 and clock == [90.0]               # looks at 0, 30, 60 and 90 s


def test_a_404_for_a_young_batch_stops_the_run_and_records_nothing():
    first, lost = open_batch()
    other = FakeBatchProvider(verdicts)                                 # another workspace's key: a 404
    judge = batch_judge(other)
    [submitted] = state_rows(judge)
    before = judge.batch_state_path.read_bytes()
    with pytest.raises(FrontierConfigError) as excinfo:
        judge.prefill_pairwise(*pairs(3))
    message = str(excinfo.value)
    for part in (f"batch {lost}", f"submitted on {submitted['created']}", "NotFoundError 404",
                 "another workspace or organisation", "ANTHROPIC_BASE_URL",
                 "rerun with the key that submitted it", f"LIFT_ABANDON_BATCHES={lost}"):
        assert part in message
    assert judge.batch_state_path.read_bytes() == before                # no row, so none syncs anywhere
    assert other.batches == {} and [c["preflight"] for c in other.calls] == [True]   # the key itself works
    with pytest.raises(FrontierConfigError):
        judge.pairwise(*pairs(3))                                       # the judge stays stopped
    assert item_calls(other) == []
    assert batch_judge(first).prefill_pairwise(*pairs(3))["n_collected"] == 3   # the submitting key does


@pytest.mark.parametrize("created", [days_ago(28), None, "not a time"],
                         ids=["28 days", "no time", "unreadable"])
def test_a_404_stops_the_run_unless_the_batch_is_past_the_results_window(created):
    judge = batch_judge(FakeBatchProvider(verdicts))
    recorded_batch(judge, "msgbatch_young", created)
    before = judge.batch_state_path.read_bytes()
    with pytest.raises(FrontierConfigError, match="batch msgbatch_young, recorded in .* unknown to the API"):
        judge.prefill_pairwise(*pairs(3))
    assert judge.batch_state_path.read_bytes() == before


def test_a_404_for_a_batch_past_the_results_window_abandons_it_and_submits_again(caplog):
    caplog.set_level(logging.WARNING, logger=frontier.__name__)
    judge = batch_judge(FakeBatchProvider(verdicts))
    recorded_batch(judge, "msgbatch_old", days_ago(30))                 # its results are no longer kept
    summary = judge.prefill_pairwise(*pairs(3))
    assert (summary["n_submitted"], summary["n_collected"]) == (3, 3)
    assert "msgbatch_old" not in summary["batch_ids"]
    rows = state_rows(judge)
    assert [r["event"] for r in rows] == ["submitted", "abandoned", "submitted", "collected"]
    assert rows[1]["batch_id"] == "msgbatch_old" and rows[1]["reason"].startswith("NotFoundError 404")
    assert any("batch msgbatch_old" in r.getMessage() and "abandoned" in r.getMessage()
               for r in caplog.records)


def test_with_every_call_a_404_the_preflight_stops_the_run_before_any_batch_is_looked_up():
    _, lost = open_batch()
    not_found = FakeProvider.error(404, message="placeholder: no such path")
    gateway = FakeBatchProvider(verdicts, preflight=not_found)          # ANTHROPIC_BASE_URL elsewhere
    looked = status_looks(gateway)
    judge = batch_judge(gateway)
    before = judge.batch_state_path.read_bytes()
    with pytest.raises(FrontierConfigError, match="preflight call failed: NotFoundError 404"):
        judge.prefill_pairwise(*pairs(3))
    assert looked == [] and gateway.batches == {} and judge.batch_state_path.read_bytes() == before


def test_lift_abandon_batches_gives_up_a_batch_by_hand_before_the_prefill(monkeypatch, caplog):
    _, lost = open_batch()
    monkeypatch.setenv("LIFT_ABANDON_BATCHES", f" {lost}, msgbatch_never_recorded ")
    caplog.set_level(logging.INFO, logger=frontier.__name__)
    other = FakeBatchProvider(verdicts)                                 # the batch was deleted: a 404 for it
    looked = status_looks(other)
    judge = batch_judge(other)
    summary = judge.prefill_pairwise(*pairs(3))
    assert (summary["n_submitted"], summary["n_collected"]) == (3, 3) and lost not in looked
    rows = state_rows(judge)
    assert [r["event"] for r in rows] == ["submitted", "abandoned", "submitted", "collected"]
    assert (rows[1]["batch_id"], rows[1]["reason"]) == (lost, "abandoned by hand")
    assert any("msgbatch_never_recorded" in r.getMessage() and "nothing to abandon" in r.getMessage()
               for r in caplog.records)
    batch_judge(FakeBatchProvider(verdicts)).pairwise(*pairs(4))       # still set: nothing more to abandon
    assert [r["event"] for r in state_rows(judge)] == ["submitted", "abandoned", "submitted", "collected"]


def test_a_batch_recorded_by_another_judge_is_never_collected():
    fake = FakeBatchProvider(verdicts)
    judge = batch_judge(fake)
    keys = [judge._key(judge.render_pair(*p), ()) for p in zip(*pairs(3), strict=True)]
    append_jsonl(judge.batch_state_path, [{
        "key": "batch:msgbatch_other:submitted", "event": "submitted", "batch_id": "msgbatch_other",
        "judge": "anthropic/another-judge", "api_version": "fake", "keys": keys, "n": 3, "created": "x"}])
    with pytest.raises(FrontierConfigError, match="submitted by 'anthropic/another-judge'"):
        judge.prefill_pairwise(*pairs(3))
    assert fake.batches == {} and fake.calls == []


# --------------------------------------------------------------------------------- the create call --
@pytest.mark.parametrize("error", [FakeProvider.error(None), FakeProvider.error(503),
                                   FakeProvider.error(529)])
def test_a_create_that_may_have_gone_through_is_never_sent_again(error, caplog):
    fake = FakeBatchProvider(verdicts, submit_errors=[error])
    judge = batch_judge(fake)
    with pytest.raises(FrontierUnavailable, match="may have created the batch of 3 requests anyway"):
        judge.prefill_pairwise(*pairs(3))
    [unknown] = state_rows(judge)
    assert (unknown["event"], unknown["n"]) == ("create_unknown", 3)
    assert unknown["key"].startswith("batch-create:")
    estimate = sum(judge._batch_estimate(judge.render_pair(*p)) for p in zip(*pairs(3), strict=True))
    assert judge.stats["usd"] >= estimate and judge._reserved == pytest.approx(0, abs=1e-12)   # kept as spent
    with pytest.raises(FrontierUnavailable):
        judge.prefill_pairwise(*pairs(3))                               # never sent again by this judge
    assert fake.batches == {}

    caplog.set_level(logging.WARNING, logger=frontier.__name__)
    rerun = batch_judge(fake)                                           # the person reran it
    assert rerun.prefill_pairwise(*pairs(3))["n_collected"] == 3
    assert any("a batch create of 3 of these requests failed" in r.getMessage() for r in caplog.records)


def test_a_create_the_api_refused_releases_its_reservation():
    refused = FakeProvider.error(400, message="placeholder: too big")
    fake = FakeBatchProvider(verdicts, submit_errors=[refused])
    judge = batch_judge(fake)
    with pytest.raises(FrontierConfigError, match="refused a batch of 3 requests.*placeholder: too big"):
        judge.prefill_pairwise(*pairs(3))
    assert not judge.batch_state_path.exists() and judge._reserved == pytest.approx(0, abs=1e-12)
    assert judge.stats["usd"] == pytest.approx(preflight_usd())      # the preflight only: nothing billed


def test_a_rate_limited_create_is_retried(offline, monkeypatch):
    monkeypatch.setattr(frontier.random, "random", lambda: 1.0)
    fake = FakeBatchProvider(verdicts, submit_errors=[FakeProvider.error(429), None])
    summary = batch_judge(fake).prefill_pairwise(*pairs(3))
    assert summary["n_collected"] == 3 and len(fake.batches) == 1 and offline == [2.0]


def test_a_transient_error_while_polling_is_retried(offline, monkeypatch):
    monkeypatch.setattr(frontier.random, "random", lambda: 1.0)
    fake = FakeBatchProvider(verdicts, status_errors=[FakeProvider.error(529)],
                             results_errors=[FakeProvider.error(None)])
    summary = batch_judge(fake).prefill_pairwise(*pairs(3))
    assert summary["n_collected"] == 3 and offline == [2.0, 2.0]        # one backoff each, no poll interval


def test_a_fatal_error_while_polling_stops_the_run():
    fake = FakeBatchProvider(verdicts, status_errors=[FakeProvider.error(401)])
    judge = batch_judge(fake)
    with pytest.raises(FrontierConfigError, match="the status of batch .* failed: AuthenticationError 401"):
        judge.prefill_pairwise(*pairs(3))
    assert [r["event"] for r in state_rows(judge)] == ["submitted"]     # a rerun collects it


# ----------------------------------------------------------------------------- spend and costing --
def test_the_batch_is_reserved_at_batch_prices_before_it_is_submitted():
    fake = FakeBatchProvider(verdicts)
    judge = batch_judge(fake)
    reserved_at_create = []
    submit = fake.submit_batch

    def spy(requests):
        reserved_at_create.append(judge._reserved)
        return submit(requests)

    fake.submit_batch = spy
    judge.prefill_pairwise(*pairs(5))
    p_in, p_out = judge.price
    system = len(judge.rubric.system.encode("utf-8"))
    users = [judge.render_pair(*p) for p in zip(*pairs(5), strict=True)]
    worst = sum(0.5 * ((2 * system + len(u.encode("utf-8")) + 16) * p_in + 1024 * p_out) / 1e6 for u in users)
    assert reserved_at_create == [pytest.approx(worst)]
    assert judge._reserved == pytest.approx(0, abs=1e-12)
    batch_usd = sum(r["usd"] for r in cache_rows(judge))
    assert judge.stats["usd"] == pytest.approx(preflight_usd() + batch_usd)   # actual usage, not the reserve


def test_the_spend_cap_stops_a_batch_before_it_is_submitted(monkeypatch):
    sizer = batch_judge(FakeBatchProvider())
    benign = sizer.estimate_usd(sizer._benign_user())
    worst = sum(sizer._batch_estimate(sizer.render_pair(*p)) for p in zip(*pairs(5), strict=True))
    assert worst > benign
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", repr((benign + worst) / 2))
    fake = FakeBatchProvider(verdicts, usage=(1, 1))
    judge = batch_judge(fake)
    with pytest.raises(SpendCapExceeded):
        judge.prefill_pairwise(*pairs(5))
    assert fake.batches == {} and [c["preflight"] for c in fake.calls] == [True]   # the preflight fit
    with pytest.raises(SpendCapExceeded):
        judge.pairwise(*pairs(5))                                       # the judge stays stopped


def test_a_batch_result_is_costed_from_its_usage_with_cache_tokens():
    def cached_usage(call):
        if not call.get("batch"):
            return winner("A")
        return ProviderResult(text=winner("A"), input_tokens=1000, output_tokens=50, stop_reason="end_turn",
                              refused=False, refusal_reason=None, model_echo="",
                              cache_creation_input_tokens=500, cache_creation_5m_input_tokens=200,
                              cache_creation_1h_input_tokens=300, cache_read_input_tokens=100)

    judge = batch_judge(FakeBatchProvider(cached_usage), model=SONNET)  # $2 / $10 per MTok
    summary = judge.prefill_pairwise(*pairs(2))
    # half of (400 uncached + 1.25 x 200 + 2 x 300 + 0.1 x 100) x $2 + 50 x $10, per million tokens
    each = 0.5 * ((400 + 1.25 * 200 + 2 * 300 + 0.1 * 100) * 2 + 50 * 10) / 1e6
    rows = cache_rows(judge)
    assert [r["usd"] for r in rows] == [pytest.approx(each)] * 2
    fields = ("cache_creation_input_tokens", "cache_creation_5m_input_tokens",
              "cache_creation_1h_input_tokens", "cache_read_input_tokens", "input_tokens")
    assert {tuple(r[name] for name in fields) for r in rows} == {(500, 200, 300, 100, 1000)}
    stats = judge.stats
    counted = [stats[k] for k in ("cache_creation_input_tokens", "cache_read_input_tokens", "batch_results")]
    assert counted == [1000, 200, 2]
    assert stats["usd"] == pytest.approx(preflight_usd(model=SONNET) + 2 * each)
    assert summary["usd"] == pytest.approx(stats["usd"], abs=1e-6)
    [collected] = [r for r in state_rows(judge) if r["event"] == "collected"]
    assert collected["usd"] == pytest.approx(2 * each) and collected["cache_read_input_tokens"] == 200


@pytest.mark.parametrize(("model", "price", "result", "expected"), [
    # no TTL split: every write at the 1-hour rate (2x)
    (SONNET, None, dict(input_tokens=1000, cache_creation_input_tokens=500, cache_read_input_tokens=100),
     0.5 * ((400 + 2 * 500 + 0.1 * 100) * 2 + 50 * 10) / 1e6),
    ("claude-opus-5-5", None, dict(input_tokens=1000, cache_read_input_tokens=100),      # reads at 0.05x
     0.5 * ((900 + 0.05 * 100) * 4 + 50 * 20) / 1e6),
    ("claude-fable-5-1", None, dict(input_tokens=1000, cache_read_input_tokens=100),     # reads at 0.025x
     0.5 * ((900 + 0.025 * 100) * 10 + 50 * 50) / 1e6),
    ("claude-mythos-5-1", {"input": 10, "output": 50}, dict(input_tokens=1000, cache_read_input_tokens=100),
     0.5 * ((900 + 0.025 * 100) * 10 + 50 * 50) / 1e6),
    (HAIKU, None, dict(input_tokens=1000), 0.5 * (1000 * 0.10 + 50 * 0.50) / 1e6),     # no cache at all
    (HAIKU, None, dict(input_tokens=None), 0.123),                                     # no usage: estimate
])
def test_batch_costs(model, price, result, expected):
    judge = FrontierJudge("anthropic", model, UTILITY, client=FakeProvider(), config_entry=None,
                          **({"price_usd_per_mtok": price} if price else {}))
    reply = ProviderResult(text="", output_tokens=50, stop_reason="end_turn", refused=False,
                           refusal_reason=None, model_echo=model, **result)
    assert judge._batch_cost(reply, 0.123) == pytest.approx(expected)


# ---------------------------------------------------------------------------------- state file --
def test_every_state_row_has_a_unique_string_key_and_lift_sync_keeps_them_all(monkeypatch):
    from lift.sync import merge_jsonl_by_key, merge_kind

    with pytest.raises(FrontierUnavailable):                            # create_unknown
        batch_judge(FakeBatchProvider(verdicts, submit_errors=[FakeProvider.error(None)])).prefill_pairwise(
            *pairs(3))
    _, given_up = open_batch(polls=1)                                   # submitted
    monkeypatch.setenv("LIFT_ABANDON_BATCHES", given_up)
    judge = batch_judge(FakeBatchProvider(verdicts))                    # abandoned, submitted, collected
    judge.prefill_pairwise(*pairs(3))
    rows = state_rows(judge)
    events = ["create_unknown", "submitted", "abandoned", "submitted", "collected"]
    assert [r["event"] for r in rows] == events
    assert all(isinstance(r["key"], str) for r in rows) and len({r["key"] for r in rows}) == len(rows)
    assert all(r["judge"] == judge.judge_id and r["api_version"] == "fake" for r in rows)
    relative = judge.batch_state_path.relative_to(frontier.output_dir().parent).as_posix()
    assert relative.endswith(".batches.jsonl") and merge_kind(relative) == "judge_cache"
    merged = merge_jsonl_by_key(judge.batch_state_path, judge.batch_state_path, kind="judge_cache")
    assert merged.dropped == 0 and merged.rows == rows


def test_the_state_file_sits_beside_its_cache():
    cache = "judge_cache/frontier/utility__claude-haiku-5-5.jsonl"
    judge = batch_judge(FakeBatchProvider(), cache_path=cache)
    assert judge.batch_state_path == judge.cache_path.with_name("utility__claude-haiku-5-5.batches.jsonl")


def test_a_cut_state_line_is_repaired_before_the_next_append():
    fake, _ = open_batch(polls=1)
    judge = batch_judge(fake)                                           # a rerun collects the batch
    path = judge.batch_state_path
    path.write_bytes(path.read_bytes() + b'{"key": "batch:cut')         # a kill mid-append
    judge.prefill_pairwise(*pairs(3))
    assert [r["event"] for r in state_rows(judge)] == ["submitted", "collected"]


# --------------------------------------------------------------------------- the real anthropic SDK --
BATCH_ID = "msgbatch_01TestPlaceholder"
RESULTS_URL = f"https://api.anthropic.com/v1/messages/batches/{BATCH_ID}/results"   # absolute, as the API's


def message(text, usage):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": SONNET, "stop_sequence": None,
            "stop_reason": "end_turn", "content": [{"type": "text", "text": text}], "usage": usage}


class BatchesApi:
    """Anthropic's Messages and Message Batches APIs on an in-process transport. A message is answered
    with winner A; a batch ends on its second look, and its results are, per request in create order:
    succeeded (with prompt-cache usage), errored (invalid_request_error), expired, and a result type this
    SDK does not know. ``cut`` cuts the first results download short."""

    def __init__(self, cut=False):
        self.cut, self.looks, self.downloads = cut, 0, 0
        self.created: list[dict] = []
        self.seen: list[tuple[str, str, dict]] = []

    def batch(self, status):
        ended = status == "ended"
        n = sum(len(body["requests"]) for body in self.created)
        return {"id": BATCH_ID, "type": "message_batch", "processing_status": status,
                "request_counts": {"processing": 0 if ended else n, "succeeded": 1 if ended else 0,
                                   "errored": 1 if ended else 0, "expired": 1 if ended else 0, "canceled": 0},
                "created_at": "2026-11-04T10:00:00Z", "expires_at": "2026-11-05T10:00:00Z",
                "ended_at": "2026-11-04T10:30:00Z" if ended else None,
                "results_url": RESULTS_URL if ended else None}

    def results(self):
        ids = [r["custom_id"] for r in self.created[0]["requests"]]
        usage = {"input_tokens": 300, "output_tokens": 20, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "service_tier": "batch",
                 "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}
        lines = [
            {"custom_id": ids[3], "result": {"type": "future_kind"}},
            {"custom_id": ids[1], "result": {"type": "errored", "error": {
                "type": "error", "request_id": "req_test",
                "error": {"type": "invalid_request_error", "message": "placeholder: rejected"}}}},
            {"custom_id": ids[0], "result": {"type": "succeeded",
                                             "message": message(winner("B", "batch"), usage)}},
            {"custom_id": ids[2], "result": {"type": "expired"}},
        ]
        return b"".join(json.dumps(line).encode("utf-8") + b"\n" for line in lines)

    def __call__(self, request):
        import httpx2

        path = request.url.path
        self.seen.append((request.method, path, dict(request.headers)))
        if request.method == "POST" and path == "/v1/messages":
            usage = {"input_tokens": 10, "output_tokens": 5}
            return httpx2.Response(200, json=message(winner("A", "sync"), usage))
        if request.method == "POST" and path == "/v1/messages/batches":
            self.created.append(json.loads(request.content))
            return httpx2.Response(200, json=self.batch("in_progress"))
        if request.method == "GET" and path == f"/v1/messages/batches/{BATCH_ID}":
            self.looks += 1
            return httpx2.Response(200, json=self.batch("ended" if self.looks > 1 else "in_progress"))
        if request.method == "GET" and path == f"/v1/messages/batches/{BATCH_ID}/results":
            self.downloads += 1
            body = self.results()
            return httpx2.Response(200, content=body[:-30] if self.cut and self.downloads == 1 else body)
        return httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error",
                                                                     "message": "placeholder: no such path"}})


def sdk_judge(api):
    anthropic, httpx2 = pytest.importorskip("anthropic"), pytest.importorskip("httpx2")
    client = anthropic.Anthropic(api_key="sk-ant-test-dummy", max_retries=0,
                                 http_client=httpx2.Client(transport=httpx2.MockTransport(api)))
    return FrontierJudge("anthropic", SONNET, UTILITY, client=client, config_entry="utility", workers=1,
                         batch_min_items=2)


@pytest.mark.parametrize("cut", [False, True])
def test_the_real_sdk_path_submits_polls_and_collects(monkeypatch, offline, cut):
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "5")
    monkeypatch.setattr(frontier.random, "random", lambda: 1.0)
    api = BatchesApi(cut=cut)
    judge = sdk_judge(api)
    assert judge.api_version == "2026-10-21"
    summary = judge.prefill_pairwise(*pairs(4))
    assert [summary[f"n_{k}"] for k in ("submitted", "collected", "errored", "expired")] == [4, 1, 1, 1]
    assert summary["batch_ids"] == [BATCH_ID] and api.downloads == (2 if cut else 1)
    assert offline == ([30.0, 2.0] if cut else [30.0])                  # a cut download is fetched again

    [body] = api.created
    users = [judge.render_pair(*p) for p in zip(*pairs(4), strict=True)]
    assert [r["custom_id"] for r in body["requests"]] == [judge._key(u, ()) for u in users]
    assert body["requests"][0]["params"] == {
        "model": SONNET, "max_tokens": 1024, "output_config": {"effort": "low"},
        "system": [{"type": "text", "text": judge.rubric.system, "cache_control": SYSTEM_CACHE_CONTROL}],
        "messages": [{"role": "user", "content": users[0]}]}
    creates = [h for method, path, h in api.seen if (method, path) == ("POST", "/v1/messages/batches")]
    assert len(creates) == 1 and creates[0].get("x-stainless-retry-count", "0") == "0"
    assert creates[0]["anthropic-version"] == "2026-10-21"

    [row] = cache_rows(judge)
    assert (row["key"], row["via"], row["batch_id"], row["text"]) == (
        body["requests"][0]["custom_id"], "batch", BATCH_ID, winner("B", "batch"))
    assert row["usd"] == pytest.approx(0.5 * (300 * 2 + 20 * 10) / 1e6)  # batch prices, no cache tokens
    assert (row["cache_creation_input_tokens"], row["cache_read_input_tokens"]) == (0, 0)
    submitted, collected = state_rows(judge)
    assert (submitted["api_version"], submitted["batch_id"]) == ("2026-10-21", BATCH_ID)
    assert collected["errors"] == {"invalid_request_error": 1} and collected["n_other"] == 1

    before = sum(path == "/v1/messages" for _, path, _ in api.seen)
    out = judge.pairwise(*pairs(4))
    assert sum(path == "/v1/messages" for _, path, _ in api.seen) - before == 3   # the three left over
    assert [o["winner"] for o in out] == ["B", "A", "A", "A"] and judge.stats["failures"] == 0


def test_the_anthropic_provider_has_the_batch_api():
    anthropic = pytest.importorskip("anthropic")
    provider = frontier.AnthropicProvider(anthropic.Anthropic(api_key="sk-ant-test-dummy", max_retries=2))
    assert frontier._batch_capable(provider)
    with pytest.raises(ValueError, match="temperature"):                # refused before anything is sent
        provider.submit_batch([{"custom_id": "k", "system": "S", "user": "U", "model": SONNET,
                                "max_tokens": 9, "temperature": 0.0, "effort": None, "thinking": None}])
    openai = pytest.importorskip("openai")
    assert not frontier._batch_capable(frontier.OpenAIProvider(openai.OpenAI(api_key="sk-test-dummy")))
