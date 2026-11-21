"""The event bus (Observer).

Publishes the four events the public interface names: `ModelCalled`, `ToolCalled`,
`ToolFailed`, `BudgetExceeded`. Those are `LoopEventKind` members; the loop already
emits them as frozen `LoopEvent` values, and this bus fans each one out to subscribers.

Subscribers never change the loop's behavior. Events are frozen, return values are
ignored, a subscriber that raises is logged and dropped, and one that stalls is bounded
by the loop's observer timeout (the bus is the loop's observer). Fan-out is concurrent,
so one slow subscriber does not delay the others' notifications.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from agent_container.core.loop import LoopEvent, LoopEventKind, LoopObserver

_log = logging.getLogger("agent_container.events")

# The four events named in the public interface, in the order a typical turn emits them.
# `BudgetExceeded` the event kind lives here; `BudgetExceeded` the budget report is a
# different type in `budget.py`, so this module does not alias the fourth name.
ModelCalled = LoopEventKind.MODEL_CALLED
ToolCalled = LoopEventKind.TOOL_CALLED
ToolFailed = LoopEventKind.TOOL_FAILED

EVENTS: tuple[LoopEventKind, ...] = (
    LoopEventKind.MODEL_CALLED,
    LoopEventKind.TOOL_CALLED,
    LoopEventKind.TOOL_FAILED,
    LoopEventKind.BUDGET_EXCEEDED,
)


class EventBus:
    """Fan-out for loop events. One bus per run or per app, never a singleton.

    The bus *is* a `LoopObserver`: pass it to `AgentLoop(observers=[bus])` (or to
    `AgentBuilder`) and subscribe everything else to the bus. It cannot subscribe to
    itself, so using the bus as its own subscriber cannot recurse without bound.
    """

    def __init__(self) -> None:
        self._subscribers: list[LoopObserver] = []

    @property
    def subscribers(self) -> Sequence[LoopObserver]:
        return tuple(self._subscribers)

    def subscribe(self, subscriber: LoopObserver) -> None:
        if subscriber is self:
            raise ValueError("an EventBus cannot subscribe to itself")
        if any(existing is subscriber for existing in self._subscribers):
            return
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: LoopObserver) -> None:
        self._subscribers = [s for s in self._subscribers if s is not subscriber]

    async def on_event(self, event: LoopEvent) -> None:
        """Notify every subscriber. Isolated: a failure here never propagates."""
        subscribers = tuple(self._subscribers)
        if not subscribers:
            return
        results = await asyncio.gather(
            *(self._notify(subscriber, event) for subscriber in subscribers),
            return_exceptions=True,
        )
        for subscriber, result in zip(subscribers, results, strict=True):
            if isinstance(result, BaseException):
                # A real cancellation of this gather (the loop's observer timeout)
                # raises from `gather` itself and never reaches here. A subscriber
                # that raises CancelledError is a bug in the subscriber, not a
                # signal to cancel the run.
                _log.warning(
                    "subscriber %s failed on %s: %s",
                    type(subscriber).__name__,
                    event.kind,
                    type(result).__name__,
                )

    async def _notify(self, subscriber: LoopObserver, event: LoopEvent) -> None:
        await subscriber.on_event(event)


__all__ = [
    "EVENTS",
    "EventBus",
    "ModelCalled",
    "ToolCalled",
    "ToolFailed",
]
