"""lift.proclock and the frontier judge's lock on its call cache (decisions item 101): one lift process at a
time judges with one cache, a second is refused and told who holds it, and the OS lets go with the holder."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager

import pytest

from lift import proclock
from lift.judge.frontier import FakeProvider, FrontierBusy, FrontierJudge

RUBRIC = "configs/rubrics/safety_v1.md"
HAIKU = "claude-haiku-5-5"

HOLD = textwrap.dedent("""
    import sys
    from lift import proclock
    proclock.acquire(sys.argv[1])
    print("held", flush=True)
    sys.stdin.read()
""")

TRY = textwrap.dedent("""
    import sys
    from lift import proclock
    try:
        proclock.acquire(sys.argv[1])
    except proclock.Busy as busy:
        print("busy", busy.holder)
    else:
        print("free")
""")

JUDGE = textwrap.dedent("""
    import sys
    from lift.judge.frontier import FrontierBusy, FrontierJudge

    class Quiet:
        api_version = "test"

        def complete(self, request):
            raise AssertionError("no call is made")

    try:
        FrontierJudge("anthropic", sys.argv[1], sys.argv[2], client=Quiet(), config_entry=None)
    except FrontierBusy as exc:
        print("busy", exc)
    else:
        print("free")
""")


class Quiet:
    """A provider that is not FakeProvider, as a real one is to the lock; it is never called here."""

    api_version = "test"

    def complete(self, request):
        raise AssertionError("no call is made")


@pytest.fixture(autouse=True)
def spend_cap(monkeypatch):
    monkeypatch.setenv("LIFT_JUDGE_MAX_USD", "1")


def run(code: str, *args: str) -> str:
    out = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True,
                         timeout=120, check=True)
    return out.stdout.strip()


@contextmanager
def held_elsewhere(path):
    """Another process holding the lock at ``path`` until the block ends."""
    proc = subprocess.Popen([sys.executable, "-c", HOLD, str(path)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        yield proc
    finally:
        proc.stdin.close()
        proc.wait(timeout=60)
        proc.stdout.close()


def test_a_process_holds_its_lock_again_and_lets_go_at_the_last_release(tmp_path):
    path = tmp_path / "cache.jsonl.lock"
    proclock.acquire(path)
    proclock.acquire(path)
    assert proclock.held(path)
    assert run(TRY, path).startswith("busy")
    proclock.release(path)
    assert proclock.held(path) and run(TRY, path).startswith("busy")
    proclock.release(path)
    assert not proclock.held(path)
    assert run(TRY, path) == "free"
    proclock.release(path)                       # not held: nothing happens
    assert not proclock.held(path)


def test_a_second_process_is_refused_and_told_who_holds_it(tmp_path):
    path = tmp_path / "sub" / "cache.jsonl.lock"
    with held_elsewhere(path) as proc:
        with pytest.raises(proclock.Busy) as caught:
            proclock.acquire(path)
        assert f"pid {proc.pid}" in caught.value.holder and str(path) in str(caught.value)
        assert not proclock.held(path)
    proclock.acquire(path)                       # the holder ended: the OS let go of its lock
    try:
        assert proclock.held(path)
    finally:
        proclock.release(path)


def test_a_killed_holder_leaves_no_lock_behind(tmp_path):
    path = tmp_path / "cache.jsonl.lock"
    proc = subprocess.Popen([sys.executable, "-c", HOLD, str(path)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        assert run(TRY, path).startswith("busy")
        proc.kill()
        proc.wait(timeout=60)
    finally:
        proc.stdin.close()
        proc.stdout.close()
    assert run(TRY, path) == "free"


def test_a_judge_on_a_real_provider_holds_its_cache_until_it_is_gone(tmp_path):
    judge = FrontierJudge("anthropic", HAIKU, RUBRIC, client=Quiet(), config_entry=None)
    lock = judge.lock_path
    assert lock == judge.cache_path.with_name(judge.cache_path.name + ".lock") and proclock.held(lock)
    busy = run(JUDGE, HAIKU, RUBRIC)                     # a second process, the same cache (LIFT_OUTPUT_DIR)
    assert busy.startswith("busy") and f"pid {os.getpid()}" in busy
    assert str(judge.cache_path) in busy and "Nothing was read or written" in busy
    again = FrontierJudge("anthropic", HAIKU, RUBRIC, client=Quiet(), config_entry=None)
    assert again.lock_path == lock                       # this process again: lift final --judge-only
    del judge
    gc.collect()
    assert proclock.held(lock)                           # `again` still holds it
    del again
    gc.collect()
    assert not proclock.held(lock)
    assert run(JUDGE, HAIKU, RUBRIC) == "free"


def test_a_second_process_judging_with_the_cache_stops_before_it_reads_or_writes(tmp_path):
    with held_elsewhere(_lock_of_default_cache()) as proc:
        with pytest.raises(FrontierBusy, match=f"pid {proc.pid}"):
            FrontierJudge("anthropic", HAIKU, RUBRIC, client=Quiet(), config_entry=None)


def _lock_of_default_cache():
    from lift.judge.frontier import _cache_path

    cache = _cache_path("anthropic", HAIKU, None)
    return cache.with_name(cache.name + ".lock")


def test_a_judge_on_a_fake_provider_takes_no_lock(tmp_path):
    judge = FrontierJudge("anthropic", HAIKU, RUBRIC, client=FakeProvider(), config_entry=None)
    assert judge.lock_path is None
    assert not judge.cache_path.with_name(judge.cache_path.name + ".lock").exists()
