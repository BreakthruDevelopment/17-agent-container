"""Tool decorators: `Retrying`, `Cached`, `Redacting`, `Metered`. Each wraps a `Tool` and is one.

They stack in any order, forward name, description, effect and input model to what they wrap, and
return it from `wrapped`, so `walk_stack` and the gate check see straight through them.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace

from pydantic import BaseModel

from agent_container.core.tool import (
    AnyTool,
    Effect,
    Identity,
    StagedChange,
    Tool,
    ToolContext,
    ToolResult,
)


class Decorator[A: BaseModel](Tool[A]):
    """Base for a decorator: forwards everything and delegates `run`. Override `run`."""

    def __init__(self, inner: Tool[A]) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def effect(self) -> Effect:
        return self._inner.effect

    @property
    def input_model(self) -> type[A]:
        return self._inner.input_model

    @property
    def wrapped(self) -> AnyTool | None:
        return self._inner

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        return await self._inner.run(args, ctx)


Sleep = Callable[[float], Awaitable[None]]


class Retrying[A: BaseModel](Decorator[A]):
    """Run the tool again when it answers with a retryable error, up to `max_attempts` in all.

    Only a result the tool itself marked `is_retryable` is retried; a raised exception is not (the
    loop turns it into an internal, non-retryable result) and neither is cancellation. The delay
    doubles from `base_delay` up to `max_delay`; `sleep` is injectable so tests do not wait.

    Retrying repeats the tool, so put it beneath a `PermissionGate` for a write: the gate then
    stages once and the retries happen when a person approves, not before.
    """

    def __init__(
        self,
        inner: Tool[A],
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        super().__init__(inner)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay < 0 or max_delay < 0:
            raise ValueError("delays cannot be negative")
        self.max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        attempt = 1
        while True:
            result = await self._inner.run(args, ctx)
            if not (result.is_error and result.is_retryable) or attempt >= self.max_attempts:
                return result
            await self._sleep(min(self._max_delay, self._base_delay * 2 ** (attempt - 1)))
            attempt += 1


def identity_key(identity: Identity) -> tuple[str, tuple[str, ...]]:
    """What distinguishes one caller from another: who they are and what they may do."""
    return identity.subject, tuple(sorted(identity.permissions))


class Cached[A: BaseModel](Decorator[A]):
    """Remember successful results, per caller.

    The key is the acting identity from `ctx` (subject *and* permissions), the tool's name and the
    arguments. Without the identity, `Cached(PermissionGate(write))` -- valid by construction --
    would serve one caller's gate decision to another. Two callers who differ only in permissions
    do not share an entry.

    Not cached: errors (a denial must be re-decided, a transient failure retried) and any result
    carrying a staged change (a `StagedChange` is single-use, so a second caller must get its
    own). Entries are not coalesced, so two simultaneous identical calls both run. Nothing is
    stored until a call finishes, so a cancelled call leaves no entry. `max_entries` bounds
    memory (least recently used goes first); `ttl` seconds bounds age, off by default.
    """

    def __init__(
        self,
        inner: Tool[A],
        *,
        max_entries: int = 256,
        ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(inner)
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        self._max_entries = max_entries
        self._ttl = ttl
        self._clock = clock
        self._entries: OrderedDict[tuple[object, ...], tuple[float, ToolResult]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        key = (*identity_key(ctx.identity), self.name, args.model_dump_json())
        hit = self._entries.get(key)
        if hit is not None:
            stored_at, result = hit
            if self._ttl is None or self._clock() - stored_at < self._ttl:
                self._entries.move_to_end(key)
                return result
            del self._entries[key]
        result = await self._inner.run(args, ctx)
        if not result.is_error and result.staged is None:
            self._entries[key] = (self._clock(), result)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return result


class _RedactedStaged(StagedChange):
    """A staged change whose description a person reads with the secrets removed."""

    def __init__(self, change: StagedChange, redact: Callable[[str], str]) -> None:
        self._change = change
        self._redact = redact

    @property
    def description(self) -> str:
        return self._redact(self._change.description)

    async def apply(self) -> None:
        await self._change.apply()


class Redacting[A: BaseModel](Decorator[A]):
    """Replace anything matching a pattern in the result, and in a staged change's description.

    One pass per pattern, in order. Put a length cap (`TurnOutputCap`) *outside* this, not inside:
    cutting a secret in half before redaction leaves a fragment no pattern matches.
    """

    def __init__(
        self,
        inner: Tool[A],
        patterns: Iterable[str | re.Pattern[str]],
        *,
        replacement: str = "[redacted]",
    ) -> None:
        super().__init__(inner)
        self._patterns = tuple(re.compile(p) for p in patterns)
        self._replacement = replacement

    def redact(self, text: str) -> str:
        for pattern in self._patterns:
            text = pattern.sub(self._replacement, text)
        return text

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        result = await self._inner.run(args, ctx)
        staged = result.staged
        return replace(
            result,
            content=self.redact(result.content),
            staged=None if staged is None else _RedactedStaged(staged, self.redact),
        )


@dataclass(slots=True)
class ToolMetrics:
    """Counts for one tool name. Shared by every `Metered` given the same `Metrics`."""

    calls: int = 0
    errors: int = 0
    seconds: float = 0.0


class Metrics:
    """Where `Metered` records, by tool name. One per run or per app, as the caller chooses."""

    def __init__(self) -> None:
        self._by_tool: dict[str, ToolMetrics] = {}

    def for_tool(self, name: str) -> ToolMetrics:
        return self._by_tool.setdefault(name, ToolMetrics())

    def snapshot(self) -> dict[str, ToolMetrics]:
        return {k: replace(v) for k, v in self._by_tool.items()}


class Metered[A: BaseModel](Decorator[A]):
    """Count calls, errors and time. An exception or a cancellation counts as an error call: the
    record is made in `finally`, so a call that never returned is still a call."""

    def __init__(
        self,
        inner: Tool[A],
        metrics: Metrics,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(inner)
        self._metrics = metrics
        self._clock = clock

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        started = self._clock()
        failed = True
        try:
            result = await self._inner.run(args, ctx)
            failed = result.is_error
            return result
        finally:
            entry = self._metrics.for_tool(self.name)
            entry.calls += 1
            entry.errors += int(failed)
            entry.seconds += max(0.0, self._clock() - started)


__all__ = [
    "Cached",
    "Decorator",
    "Metered",
    "Metrics",
    "Redacting",
    "Retrying",
    "ToolMetrics",
    "identity_key",
]
