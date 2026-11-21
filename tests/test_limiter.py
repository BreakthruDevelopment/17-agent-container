"""The model-call limiter: a `ModelProvider` decorator, so the contract holds and the loop is
unchanged."""

from __future__ import annotations

import asyncio
import gc
import itertools
import threading
import time
from collections.abc import Sequence

import pytest

from agent_container.core import (
    AgentLoop,
    Budget,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    RateLimitedProvider,
    ReplayProvider,
)
from agent_container.testing import ModelProviderContract, OverlapProbe
from tests.support import ProbedReplay, scripted


class TestRateLimitedProviderContract(ModelProviderContract):
    def make_provider(
        self, replies: Sequence[ModelResponse], probe: OverlapProbe | None = None
    ) -> ModelProvider:
        # Three calls must be able to overlap, so the limit has to admit three.
        return RateLimitedProvider(ProbedReplay(replies, probe), max_concurrent=3)


class Gauge(ModelProvider):
    """Holds each call open for a moment and notes how many overlapped."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.seen: list[ModelRequest] = []

    async def converse(self, request: ModelRequest) -> ModelResponse:
        self.seen.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ModelResponse.of("ok")


@pytest.mark.parametrize("limit", [1, 3])
async def test_never_more_than_the_limit_in_flight(limit: int) -> None:
    inner = Gauge()
    limited = RateLimitedProvider(inner, max_concurrent=limit)
    req = ModelRequest("s", ())
    await asyncio.gather(*(limited.converse(req) for _ in range(10)))
    assert inner.peak == limit
    assert limited.peak_in_flight == limit
    assert limited.in_flight == 0


async def test_without_the_limiter_the_same_fan_out_overlaps() -> None:
    """The control: the limit above is the limiter's doing, not the test's."""
    inner = Gauge()
    req = ModelRequest("s", ())
    await asyncio.gather(*(inner.converse(req) for _ in range(10)))
    assert inner.peak == 10


async def test_calls_are_spaced_by_the_minimum_interval() -> None:
    now = [0.0]
    starts: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        now[0] += seconds

    class Stamp(ModelProvider):
        async def converse(self, request: ModelRequest) -> ModelResponse:
            starts.append(now[0])
            return ModelResponse.of("ok")

    limited = RateLimitedProvider(
        Stamp(), max_concurrent=5, min_interval_seconds=2.0, clock=lambda: now[0], sleep=fake_sleep
    )
    await asyncio.gather(*(limited.converse(ModelRequest("s", ())) for _ in range(4)))
    assert sorted(starts) == [0.0, 2.0, 4.0, 6.0]


async def test_a_failing_call_frees_its_slot() -> None:
    class Boom(ModelProvider):
        async def converse(self, request: ModelRequest) -> ModelResponse:
            raise RuntimeError("boom")

    limited = RateLimitedProvider(Boom(), max_concurrent=1)
    for _ in range(3):  # a leaked slot would hang the second call
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(limited.converse(ModelRequest("s", ())), 1)
    assert limited.in_flight == 0


async def test_the_request_and_reply_pass_through_untouched() -> None:
    inner = ReplayProvider([ModelResponse.of("hello", stop_reason="a_future_value")])
    req = ModelRequest("s", ())
    got = await RateLimitedProvider(inner).converse(req)
    assert inner.requests == [req]
    assert (got.text, got.stop_reason) == ("hello", "a_future_value")


async def test_the_loop_runs_through_it_unchanged() -> None:
    inner = scripted(ModelResponse.of("done"))
    agent = AgentLoop(RateLimitedProvider(inner), system_prompt="p")
    result = await agent.run("go", Budget())
    assert result.final_text == "done"


def test_bad_settings_are_refused() -> None:
    with pytest.raises(ValueError):
        RateLimitedProvider(ReplayProvider([]), max_concurrent=0)
    with pytest.raises(ValueError):
        RateLimitedProvider(ReplayProvider([]), min_interval_seconds=-1.0)


# -- review findings on PR #10 ------------------------------------------------------------------


class ThreadedProvider(ModelProvider):
    """Like `BedrockProvider`: the call is a worker thread that cancelling the await cannot stop."""

    def __init__(self, fail: bool = False) -> None:
        self.release = threading.Event()
        self.started = threading.Semaphore(0)
        self._lock = threading.Lock()
        self._fail = fail
        self.running = 0
        self.peak_running = 0
        self.begun = 0

    def _work(self) -> ModelResponse:
        with self._lock:
            self.running += 1
            self.begun += 1
            self.peak_running = max(self.peak_running, self.running)
        self.started.release()
        try:
            self.release.wait(5)
            if self._fail:
                raise RuntimeError("the call failed after its caller gave up")
            return ModelResponse.of("ok")
        finally:
            with self._lock:
                self.running -= 1

    async def converse(self, request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self._work)


async def test_the_permit_is_held_until_the_real_call_finishes_not_until_the_caller_gives_up() -> (
    None
):
    inner = ThreadedProvider()
    limited = RateLimitedProvider(inner, max_concurrent=1)
    req = ModelRequest("s", ())

    # The first caller times out, as the loop's wall clock would make it.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(limited.converse(req), 0.05)
    assert inner.running == 1  # the thread, and so the HTTP request, is still going
    assert limited.in_flight == 1  # and the limiter says so

    second = asyncio.ensure_future(limited.converse(req))
    await asyncio.sleep(0.1)
    assert inner.begun == 1, "a second real call was admitted while the first was still running"
    assert not second.done()

    inner.release.set()
    await asyncio.wait_for(second, 5)
    assert inner.peak_running == 1
    assert limited.peak_in_flight == 1
    assert limited.in_flight == 0


async def test_an_abandoned_calls_exception_is_not_left_unretrieved() -> None:
    inner = ThreadedProvider(fail=True)
    limited = RateLimitedProvider(inner, max_concurrent=1)
    reported: list[dict[str, object]] = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: reported.append(ctx))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(limited.converse(ModelRequest("s", ())), 0.05)
    inner.release.set()
    for _ in range(100):  # let the abandoned call finish and its callbacks run
        if limited.in_flight == 0:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    gc.collect()
    assert limited.in_flight == 0
    assert reported == []


async def test_a_caller_cancelled_while_waiting_does_not_leak_a_permit() -> None:
    inner = ThreadedProvider()
    limited = RateLimitedProvider(inner, max_concurrent=1)
    req = ModelRequest("s", ())
    first = asyncio.ensure_future(limited.converse(req))
    await asyncio.sleep(0.05)
    waiting = asyncio.ensure_future(limited.converse(req))
    await asyncio.sleep(0.05)
    waiting.cancel()
    inner.release.set()
    await first
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await asyncio.wait_for(limited.converse(req), 5)  # would hang on a leaked permit
    assert limited.in_flight == 0


async def test_spacing_holds_when_the_event_loop_stalls_past_several_reservations() -> None:
    """Reserved timestamps are not enough: after a stall every overdue sleep ends together.

    The stall is injected, not hoped for: a coroutine blocks the loop for 80 ms while the second
    and third callers are waiting out 30 ms and 60 ms. Reserving future start times made both fire
    in the same iteration, about 10 microseconds apart.
    """
    starts: list[float] = []

    class Stamp(ModelProvider):
        async def converse(self, request: ModelRequest) -> ModelResponse:
            starts.append(time.monotonic())
            return ModelResponse.of("ok")

    limited = RateLimitedProvider(Stamp(), max_concurrent=5, min_interval_seconds=0.03)

    async def stall() -> None:
        await asyncio.sleep(0.005)
        time.sleep(0.08)  # noqa: ASYNC251 - blocking the loop is the point

    req = ModelRequest("s", ())
    await asyncio.gather(*(limited.converse(req) for _ in range(3)), stall())
    gaps = [b - a for a, b in itertools.pairwise(sorted(starts))]
    assert len(gaps) == 2
    assert all(gap >= 0.025 for gap in gaps), gaps  # 30 ms, less clock slack
