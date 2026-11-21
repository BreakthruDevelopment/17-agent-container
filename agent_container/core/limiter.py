"""`RateLimitedProvider`: the in-process model-call limiter, as a `ModelProvider` decorator.

The spec's throttling decision: one limiter per run, so a parallel fan-out (app 20) cannot trip
Bedrock's rate limits. It wraps any provider and changes nothing about what is sent or returned,
which is why the loop needs no change and `ModelProviderContract` holds for it.

Share one instance between every agent in a run (a coordinator and its subagents) and they share one
allowance. An instance is bound to the event loop it first waits on, so make one per run.

Two properties are deliberate, and both come from a cancelled caller not being a finished call:

* **The permit is held until the underlying call has finished**, not until the caller's await
  unwinds. The loop's wall clock cancels `converse`, but cancelling an
  `await asyncio.to_thread(...)` leaves the worker thread (the HTTP request) running. Releasing
  on cancellation would admit a second real call while the first is still on the wire. So the
  inner call runs as its own task, the caller awaits it through `asyncio.shield`, and the permit
  is released when that task ends, inside the task. A timed-out caller still returns promptly;
  the abandoned call finishes in the background. A done-callback retrieves its exception, so it
  is never reported as unretrieved.
* **Spacing is measured from the previous call's actual start.** Admission is serialised, and after
  every wait the interval is re-checked against the real start time of the last call. Reserving
  future timestamps is not enough: if the event loop stalls past several reservations, every overdue
  sleep ends together and the calls would start back to back.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable

from agent_container.core.messages import ModelRequest, ModelResponse
from agent_container.core.provider import ModelProvider


class RateLimitedProvider(ModelProvider):
    """At most `max_concurrent` calls in flight, and calls started at least `min_interval` apart.

    `in_flight` and `peak_in_flight` count calls that are really running, including ones whose
    caller has been cancelled.
    """

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
        self._admit = asyncio.Lock()
        self._interval = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_start = -math.inf
        self.in_flight = 0
        self.peak_in_flight = 0

    async def converse(self, request: ModelRequest) -> ModelResponse:
        await self._slots.acquire()
        task: asyncio.Task[ModelResponse] | None = None
        try:
            async with self._admit:
                while (wait := self._last_start + self._interval - self._clock()) > 0:
                    await self._sleep(wait)
                started = asyncio.Event()
                self.in_flight += 1
                self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
                task = asyncio.ensure_future(self._run(request, started))
                task.add_done_callback(self._retrieve)  # from here the permit is the task's
                # Wait for the call to really begin so the next admission measures from it.
                await started.wait()
        except BaseException:
            if task is None:
                self._slots.release()
            raise
        return await asyncio.shield(task)

    async def _run(self, request: ModelRequest, started: asyncio.Event) -> ModelResponse:
        # Accounting and release happen here, synchronously with the call ending, not in a
        # callback: a callback runs a loop iteration later, so a caller could return (and read
        # `in_flight`, or call again) before the permit was back.
        self._last_start = self._clock()
        started.set()
        try:
            return await self._inner.converse(request)
        finally:
            self.in_flight -= 1
            self._slots.release()

    @staticmethod
    def _retrieve(task: asyncio.Task[ModelResponse]) -> None:
        if not task.cancelled():
            task.exception()  # a caller that gave up is not left to report it as unretrieved


__all__ = ["RateLimitedProvider"]
