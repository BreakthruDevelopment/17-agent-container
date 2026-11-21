"""`RateLimitedProvider`: the in-process model-call limiter, as a `ModelProvider` decorator.

The spec's throttling decision: one limiter per run, so a parallel fan-out (app 20) cannot trip
Bedrock's rate limits. It wraps any provider and changes nothing about what is sent or returned,
which is why the loop needs no change and `ModelProviderContract` holds for it.

Share one instance between every agent in a run (a coordinator and its subagents) and they share one
allowance. An instance is bound to the event loop it first waits on, so make one per run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from agent_container.core.messages import ModelRequest, ModelResponse
from agent_container.core.provider import ModelProvider


class RateLimitedProvider(ModelProvider):
    """At most `max_concurrent` calls in flight, and calls started at least `min_interval` apart."""

    def __init__(
        self,
        inner: ModelProvider,
        *,
        max_concurrent: int = 4,
        min_interval_seconds: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        self._inner = inner
        self._slots = asyncio.Semaphore(max_concurrent)
        self._interval = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._next_start = 0.0
        self.in_flight = 0
        self.peak_in_flight = 0

    async def converse(self, request: ModelRequest) -> ModelResponse:
        async with self._slots:
            if self._interval > 0:
                # No await between reading and writing `_next_start`, so concurrent callers each
                # claim a distinct start time. A caller cancelled while waiting forfeits its slot,
                # which only spaces the others a little further apart.
                now = self._clock()
                start = max(now, self._next_start)
                self._next_start = start + self._interval
                if start > now:
                    await self._sleep(start - now)
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            try:
                return await self._inner.converse(request)
            finally:
                self.in_flight -= 1


__all__ = ["RateLimitedProvider"]
