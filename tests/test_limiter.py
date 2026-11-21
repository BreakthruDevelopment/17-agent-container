"""The model-call limiter: a `ModelProvider` decorator, so the contract holds and the loop is
unchanged."""

from __future__ import annotations

import asyncio
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
from agent_container.testing import ModelProviderContract
from tests.support import scripted


class TestRateLimitedProviderContract(ModelProviderContract):
    def make_provider(self, replies: Sequence[ModelResponse]) -> ModelProvider:
        return RateLimitedProvider(ReplayProvider(replies), max_concurrent=2)


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
