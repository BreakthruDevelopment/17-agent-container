"""Tool decorators: `Retrying`, `Cached`, `Redacting`, `Metered`. Each wraps a `Tool` and is one.

They stack in any order, forward name, description, effect and input model to what they wrap, and
return it from `wrapped`, so `walk_stack` and the gate check see straight through them.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable, Iterable
from dataclasses import dataclass, replace

from pydantic import BaseModel

from agent_container.core.errors import StagedChangeError
from agent_container.core.tool import (
    AnyTool,
    Effect,
    Identity,
    StagedChange,
    Tool,
    ToolContext,
    ToolResult,
    has_write_gate,
    walk_stack,
    writes,
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

# A member of a stack that cuts tool output sets this to True (`Intercepted` reports it for its
# chain). Read by attribute, as the gate marker is, so this module need not import the cap.
CAPS_OUTPUT_MARKER = "caps_output"

# The longest staged-change description a person is shown. Applied last, after redaction.
MAX_DESCRIPTION_CHARS = 1000


def stack_caps_output(tool: AnyTool) -> bool:
    """Whether anything in this stack (the tool included) cuts the output it returns."""
    return any(getattr(member, CAPS_OUTPUT_MARKER, False) is True for member in walk_stack(tool))


def limit_description(text: str) -> str:
    return text if len(text) <= MAX_DESCRIPTION_CHARS else text[:MAX_DESCRIPTION_CHARS] + "..."


class Retrying[A: BaseModel](Decorator[A]):
    """Run the tool again when it answers with a retryable error, up to `max_attempts` in all.

    Only a result the tool itself marked `is_retryable` is retried; a raised exception is not (the
    loop turns it into an internal, non-retryable result) and neither is cancellation. The delay
    doubles from `base_delay` up to `max_delay`; `sleep` is injectable so tests do not wait.

    Placement: a `PermissionGate` stages a write with an ordinary `ok` result, which is never
    retried. So `Retrying(PermissionGate(write))` retries nothing and the write runs once, when
    approved; `PermissionGate(Retrying(write))` retries the write itself when it is applied. No
    write happens before approval in either order -- the order decides only whether the applied
    call is retried.
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
    """What distinguishes one caller from another: who they are and what they may do.

    Only for a plain `Identity`. A subclass may carry fields (a tenant, say) this does not read,
    and keying on a subset would let two callers share an entry, so it raises rather than guess.
    `Cached(key_for=...)` is how an application says what makes its own identity type distinct.
    """
    if type(identity) is not Identity:
        raise TypeError(
            f"cannot key a cache on {type(identity).__name__}: it may carry fields the default "
            "key does not read; pass key_for= to Cached"
        )
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

    A cache must not sit over a write that has no gate beneath it. It would remember the applied
    success and answer every later approval of the same call from memory, so a person approves a
    transfer, `apply()` reports success, and nothing ran. That covers
    `PermissionGate(Cached(write))` and a bare `Cached(write)` alike, and construction refuses
    it. `Cached(PermissionGate(write))` is fine: the gate is beneath,
    and a staged result is never stored.

    A hit returns without running anything beneath, so a `TurnOutputCap` beneath a `Cached` would
    never see the delivered result and the cap would be bypassed. Construction refuses that;
    put the cap outside the cache, where it charges every result, hits included.

    The default key accepts a plain `Identity` only; for a subclass pass `key_for`, which must
    return something hashable that includes every field distinguishing one caller from another.
    """

    def __init__(
        self,
        inner: Tool[A],
        *,
        max_entries: int = 256,
        ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        key_for: Callable[[Identity], Hashable] = identity_key,
    ) -> None:
        super().__init__(inner)
        if writes(inner) and not has_write_gate(inner):
            raise ValueError(
                "a Cached over a write with no permission gate beneath it would answer a "
                "repeated, approved write from memory; use Cached(PermissionGate(write))"
            )
        if stack_caps_output(inner):
            raise ValueError(
                "a TurnOutputCap beneath a Cached is bypassed by cache hits; put it outside"
            )
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive")
        self._max_entries = max_entries
        self._ttl = ttl
        self._clock = clock
        self._key_for = key_for
        self._entries: OrderedDict[tuple[object, ...], tuple[float, ToolResult]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        key = (self._key_for(ctx.identity), self.name, args.model_dump_json())
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
    """A staged change a person reads, and is told about, with the secrets removed.

    Three paths carry text from the change to a person, and all three go through the redactor:
    the description (redacted whole, *then* shortened, so a secret straddling the display limit
    cannot survive as a fragment), and the message of any failure when applying.

    The two description paths are rendered once, at staging; only the failure message is redacted
    when it is raised. See `__init__`.
    """

    def __init__(self, change: StagedChange, redact: Callable[[str], str]) -> None:
        self._change = change
        # Kept for `apply()`. A failure message is produced when the change is applied, not when
        # it is staged, so it is redacted then, with the patterns as they stand.
        self._redact = redact
        # Rendered once, now, for the reason `ToolCallChange.__init__` renders its own text once:
        # `description` is the record of what the approver was shown, and a record that moves is
        # worse than none, because it reads as corroboration. Freezing the text beneath this
        # wrapper was not enough, because both steps this wrapper adds were themselves live:
        # redaction read `Redacting._replacement` on every read, so replacing it after staging
        # moved the record (measured: `{"amount":[redacted]}` became `{"amount":[CHANGED]}`), and
        # the shortening read the module-level `MAX_DESCRIPTION_CHARS` the same way. Neither is
        # reachable from the applied call -- a tool or an interceptor beneath the gate sees only
        # `ToolContext`, which carries no decorator configuration -- so this is the letter of
        # `StagedChange.description`'s contract rather than a live defect. Frozen anyway: the
        # contract is that the string does not move, and "nothing can reach it today" is not a
        # property three dependent apps should have to re-derive.
        #
        # Rendering here, rather than on first read, moves any failure in the change beneath
        # (a description that raises) into `Redacting.run`, where the loop turns it into an
        # `internal` result and nothing is staged. That mirrors what freezing `ToolCallChange`
        # did at the gate, and is tested in the same shape.
        full = getattr(change, "full_description", None)
        self._full_description = redact(full if isinstance(full, str) else change.description)
        self._description = limit_description(self._full_description)

    @property
    def full_description(self) -> str:
        """Redacted, not shortened. An outer wrapper redacts this again before shortening.

        Fixed when the change was staged; see `__init__`.
        """
        return self._full_description

    @property
    def description(self) -> str:
        return self._description

    async def apply(self) -> None:
        failure: str | None = None
        try:
            await self._change.apply()
        except StagedChangeError as exc:
            failure = str(exc)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        if failure is not None:
            # Raised outside the `except`, so the raw exception is not kept as `__context__`.
            raise StagedChangeError(self._redact(failure))


class Redacting[A: BaseModel](Decorator[A]):
    """Replace anything matching a pattern in the result, and in a staged change's description.

    One pass per pattern, in order. A length cap (`TurnOutputCap`) must sit *outside* this:
    cutting a secret in half before redaction leaves a fragment no pattern matches. Construction
    refuses a cap anywhere beneath, through any decorators between.
    """

    def __init__(
        self,
        inner: Tool[A],
        patterns: Iterable[str | re.Pattern[str]],
        *,
        replacement: str = "[redacted]",
    ) -> None:
        super().__init__(inner)
        if stack_caps_output(inner):
            raise ValueError(
                "a TurnOutputCap beneath Redacting can cut a secret before it is redacted; "
                "put the cap outside"
            )
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
    "CAPS_OUTPUT_MARKER",
    "Cached",
    "Decorator",
    "Metered",
    "Metrics",
    "Redacting",
    "Retrying",
    "ToolMetrics",
    "identity_key",
]
