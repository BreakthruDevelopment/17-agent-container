"""The agent loop (Template Method).

`AgentLoop.run` is the fixed skeleton: call the model, branch on the stop reason, run the tools,
append the results, check the budget. A subclass changes behavior by overriding `before_model`,
`after_tool` and `should_stop`, and nowhere else; overriding the skeleton raises `TypeError` when
the subclass is defined.

The loop ends on the stop reason the model reports, never on what its text says. Each recognized
stop reason has its own branch and its own named ending, and an unrecognized one raises
`UnrecognizedStopReason`.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from abc import ABCMeta
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, assert_never, final

from agent_container.core.budget import Budget, BudgetExceeded
from agent_container.core.errors import InputTooLarge, MalformedModelResponse
from agent_container.core.fence import check_nonce, fence_tool_output, system_notice
from agent_container.core.messages import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from agent_container.core.provider import ModelProvider
from agent_container.core.tool import (
    TOOL_NAME_PATTERN,
    AnyTool,
    ErrorCategory,
    Identity,
    StagedChange,
    ToolContext,
    ToolResult,
    has_write_gate,
    writes,
)

MAX_USER_MESSAGE_CHARS = 8_000

# How long one observer may take per event. Telemetry is the expected observer and a stalled
# exporter must not hold a run open, so this is a ceiling on reporting, not on work: exceeding it
# is logged and the run carries on, exactly as an observer that raises already did.
OBSERVER_TIMEOUT_SECONDS = 5.0

# A tool name is an identifier. The same shape the provider contract enforces, applied at the
# boundary so a model-chosen name never reaches a fenced result, a header or a log line.
_TOOL_NAME = TOOL_NAME_PATTERN  # one definition, in core.tool


def _safe_name(name: str) -> str:
    """A model-controlled name, made safe to display."""
    return name if _TOOL_NAME.match(name) else f"rejected-{len(name)}-chars"


_log = logging.getLogger("agent_container.loop")

# Everything in the skeleton. `__init_subclass__` refuses a subclass that defines any of these.
# Anything that fences tool output, sanitises a model-chosen name, records a call or publishes an
# event belongs here: a subclass that could replace one of those could unfence the model's input.
# `_record` was added on 2026-11-20 when block-building moved out of `_run_tool` into it -- the
# extraction silently took guarded behaviour out of the guard, which is the hazard to watch for
# whenever this loop is refactored.
_SKELETON = ("run", "_call_model", "_run_tool", "_record", "_finish", "_publish")


class EndReason(StrEnum):
    """Why the loop ended. Each model stop reason that ends a run has its own name."""

    END_TURN = "end_turn"
    TRUNCATED = "truncated"  # stop reason max_tokens: the reply was cut off
    STOP_SEQUENCE = "stop_sequence"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    CONTENT_FILTERED = "content_filtered"
    BUDGET_EXCEEDED = "budget_exceeded"
    STOPPED_BY_HOOK = "stopped_by_hook"  # a subclass's `should_stop` said so


_END_DETAIL = {
    EndReason.END_TURN: "the model finished its turn",
    EndReason.TRUNCATED: "the reply was truncated: the model hit its max_tokens limit",
    EndReason.STOP_SEQUENCE: "the model emitted a stop sequence",
    EndReason.GUARDRAIL_INTERVENED: "a guardrail intervened and blocked the reply",
    EndReason.CONTENT_FILTERED: "the content filter blocked the reply",
}


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    turn: int
    tool: str
    tool_use_id: str
    is_error: bool
    error_category: ErrorCategory | None


@dataclass(frozen=True, slots=True)
class RunState:
    """A read-only snapshot handed to the hooks. Changing it changes nothing in the run."""

    run_id: str
    turn: int
    messages: tuple[Message, ...]
    input_tokens: int
    output_tokens: int
    tool_calls: tuple[ToolCallRecord, ...]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RunResult:
    """What a run reports. A run that stops early still returns one; only misuse and contract
    violations raise."""

    run_id: str
    final_text: str
    stop_reason: str | None  # the model's last stop reason, as reported
    end_reason: EndReason
    end_detail: str  # the stated reason, in words
    turns_used: int
    input_tokens: int
    output_tokens: int
    tool_calls: tuple[ToolCallRecord, ...]
    staged_changes: tuple[StagedChange, ...]
    budget_exceeded: BudgetExceeded | None
    messages: tuple[Message, ...]


class LoopEventKind(StrEnum):
    MODEL_CALLED = "model_called"
    TOOL_CALLED = "tool_called"
    TOOL_FAILED = "tool_failed"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass(frozen=True, slots=True)
class LoopEvent:
    """A fact about the run, published to observers. Frozen: read-only for them."""

    kind: LoopEventKind
    run_id: str
    turn: int
    tool: str | None = None
    detail: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LoopObserver(Protocol):
    """Something that watches a run. The `EventBus` (17.5) is one.

    The loop awaits `on_event` and ignores what it does: an exception is logged and dropped, and
    nothing it returns is read. An observer cannot change the loop's behavior.
    """

    async def on_event(self, event: LoopEvent) -> None: ...


class _Run:
    """Mutable bookkeeping for one run, private to the skeleton."""

    def __init__(self, run_id: str, budget: Budget, nonce: str, system: str) -> None:
        self.run_id = run_id
        self.budget = budget
        self.nonce = nonce
        self.system = system
        self.messages: list[Message] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls: list[ToolCallRecord] = []
        self.staged: list[StagedChange] = []
        self.last: ModelResponse | None = None

    @property
    def turn(self) -> int:
        return self.budget.turns_used

    def state(self) -> RunState:
        return RunState(
            run_id=self.run_id,
            turn=self.turn,
            messages=tuple(self.messages),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            tool_calls=tuple(self.tool_calls),
            elapsed_seconds=self.budget.elapsed,
        )


# The answer to a tool call the run had no time for, whether it was cut off mid-flight or never
# dispatched. One definition so both paths report the same thing to the model.
_CUT_OFF = ToolResult.error(
    ErrorCategory.TRANSIENT,
    "the tool was cut off: the run's time budget ran out",
    retryable=False,
)


class _LoopMeta(ABCMeta):
    """Refuse a subclass that replaces any part of the skeleton.

    This is a metaclass rather than `__init_subclass__` because that hook can be switched off:
    a base ordered before `AgentLoop` that defines its own `__init_subclass__` and forgets
    `super()` means the hook never runs, so the guard cannot police itself from inside it. A
    metaclass runs on every class creation regardless. It extends `ABCMeta`, not `type`, so a
    subclass may still be an ABC without a metaclass conflict.

    The comparison is on what the MRO *resolves*, not on what the class *declares*: a mixin
    ordered first supplies the method, which is how cross-cutting behaviour is normally written
    and needs no intent to circumvent. Measured before this guard existed: such a subclass was
    accepted and the model received unfenced tool output.
    """

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any
    ) -> _LoopMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        # Identity, with the name match only as the bootstrap fallback while `AgentLoop` is
        # still being created and the module global does not exist yet. Matching on the name
        # alone would let a consumer's own class called `AgentLoop` stand in as the base.
        anchor = globals().get("AgentLoop")
        base = next(
            (
                b
                for b in cls.__mro__[1:]
                if b is anchor or (anchor is None and b.__name__ == "AgentLoop")
            ),
            None,
        )
        if base is None:  # AgentLoop itself
            return cls
        replaced = [
            member
            for member in _SKELETON
            if getattr(cls, member, None) is not getattr(base, member, None)
        ]
        # The guard is part of the skeleton too, by whichever route it arrives.
        # An AgentLoop-derived class that defines `__init_subclass__` is refused: it would
        # silence `AgentLoop`'s own hook for every descendant. Restricted to `_LoopMeta`
        # instances because ordinary bases legitimately define one -- `typing.Generic` does, and
        # so does any cooperative mixin that consumes class keyword arguments. Refusing those
        # was a false positive, and a guard that wrongly refuses costs apps 18-20 as much as one
        # that wrongly allows. The metaclass runs whatever the MRO does, so nothing here is
        # load-bearing for the guard itself. Identity comparison does not work -- the implicit
        # classmethod binds per class -- so this asks which classes actually define one.
        defines_hook = [
            ancestor.__name__
            for ancestor in cls.__mro__[: cls.__mro__.index(base)]
            if "__init_subclass__" in vars(ancestor) and isinstance(ancestor, _LoopMeta)
        ]
        if defines_hook:
            replaced.append(f"__init_subclass__ (on {', '.join(defines_hook)})")
        if replaced:
            raise TypeError(
                f"{name} overrides the loop skeleton ({', '.join(replaced)}); "
                "override before_model, after_tool or should_stop instead"
            )
        return cls


class AgentLoop(metaclass=_LoopMeta):
    """Runs a model and a set of tools until the model's stop reason, or the budget, says stop."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Consume PEP 487 class keyword arguments. The guard itself lives in `_LoopMeta`.

        They are forwarded, not swallowed: a cooperative mixin that consumes `tag` gets it from
        either side of `AgentLoop` in the bases, and a name nothing consumes still reaches
        `object.__init_subclass__` and is rejected -- so `tagg="x"` is a `TypeError` rather than
        a silent no-op. `AgentLoop` itself defines no class keyword arguments. Defining this
        hook on a subclass is refused by the metaclass, so it cannot silence the guard.
        """
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        provider: ModelProvider,
        tools: Sequence[AnyTool] = (),
        *,
        system_prompt: str,
        identity: Identity | None = None,
        observers: Sequence[LoopObserver] = (),
        max_reply_tokens: int | None = None,
        fence_nonce: str | None = None,
    ) -> None:
        by_name: dict[str, AnyTool] = {}
        for tool in tools:
            if tool.name in by_name:
                raise ValueError(f"two tools are named {tool.name!r}")
            # `AGENTS.md`: an `effect: write` tool without a `PermissionGate` anywhere in its stack
            # is a construction error. Enforced here, the one place every tool must pass through,
            # rather than at dispatch -- by then the tool is already in the model's tool list, and
            # a caller that built the loop has been told nothing is wrong.
            if writes(tool) and not has_write_gate(tool):
                raise ValueError(
                    f"tool {tool.name!r} writes but has no permission gate in its stack; "
                    "wrap it in a gate (17.4's PermissionGate) before giving it to a loop"
                )
            by_name[tool.name] = tool
        self._provider = provider
        self._tools = by_name
        self._specs = tuple(ToolSpec(t.name, t.description, t.input_schema()) for t in tools)
        self._system_prompt = system_prompt
        self._identity = identity or Identity.anonymous()
        self._observers = tuple(observers)
        self._max_reply_tokens = max_reply_tokens
        if fence_nonce is not None:
            check_nonce(fence_nonce)  # fail at construction, not on the first tool result
        self._fence_nonce = fence_nonce

    # -- hooks: the only things a subclass may override ------------------------------------------

    async def before_model(self, state: RunState, request: ModelRequest) -> ModelRequest:
        """Last chance to adjust a request, for example to set `tool_choice` or trim history."""
        return request

    async def after_tool(
        self, state: RunState, call: ToolUseBlock, result: ToolResult
    ) -> ToolResult:
        """Adjust a tool's result before it is fenced and returned to the model."""
        return result

    async def should_stop(self, state: RunState) -> str | None:
        """Return a reason to end the run before the next model call, or `None` to carry on.

        Called only when the model has asked for tools and they have run. It can end a run earlier;
        it cannot keep one going past a stop reason that ended it, and it should read `state`, not
        text.
        """
        return None

    # -- the skeleton ------------------------------------------------------------------------------

    @final
    async def run(self, user_message: str, budget: Budget) -> RunResult:
        """Run to completion. `budget` is this run's own; it must not have been used before."""
        if not user_message.strip():
            raise InputTooLarge("the user message is empty")
        if len(user_message) > MAX_USER_MESSAGE_CHARS:
            raise InputTooLarge(f"the user message is over {MAX_USER_MESSAGE_CHARS} characters")
        budget.start()
        nonce = self._fence_nonce or secrets.token_hex(8)
        run = _Run(
            uuid.uuid4().hex,
            budget,
            nonce,
            f"{self._system_prompt}\n\n{system_notice(nonce)}",
        )
        run.messages.append(Message.user_text(user_message))

        while True:
            exceeded = budget.check()
            if exceeded is not None:
                return await self._finish(
                    run, EndReason.BUDGET_EXCEEDED, exceeded.message, exceeded
                )

            request = ModelRequest(
                system=run.system,
                messages=tuple(run.messages),
                tools=self._specs,
                max_tokens=self._max_reply_tokens,
            )
            request = await self.before_model(run.state(), request)

            response = await self._call_model(run, request)
            if response is None:
                exceeded = BudgetExceeded("wall_clock", budget.elapsed, budget.max_seconds)
                return await self._finish(
                    run, EndReason.BUDGET_EXCEEDED, exceeded.message, exceeded
                )
            run.messages.append(Message("assistant", response.content))

            # Branch on the stop reason. Every member is named; an unrecognized value raised above.
            reason = StopReason.parse(response.stop_reason)
            match reason:
                case StopReason.TOOL_USE:
                    if not response.tool_uses:
                        raise MalformedModelResponse(
                            "stop reason tool_use but no tool call in the reply"
                        )
                    blocks: list[ToolResultBlock] = []
                    for call in response.tool_uses:
                        # Check the WALL CLOCK between dispatches -- only that, not the whole
                        # budget. `asyncio.timeout(0)` fires at the next yield, so a tool that
                        # completes without awaiting anything runs after the clock has expired,
                        # and one reply can carry many calls. Turns and tokens are deliberately
                        # not checked here: the model already produced this reply within budget,
                        # so its tool calls belong to a turn that was allowed, and stopping them
                        # mid-batch would make `max_turns=1` mean "no tools at all".
                        # Found by Codex on 2026-11-20.
                        if run.budget.remaining_seconds <= 0.0:
                            # Stop dispatching, but answer every call that is left. An
                            # unanswered ToolUseBlock makes the transcript unsendable, and 17.3
                            # owns making it resumable without being allowed to edit `core/`.
                            for skipped in response.tool_uses[len(blocks) :]:
                                # Through `after_tool` like any other result: its contract is
                                # "adjust a tool's result before it is fenced and returned to the
                                # model", and a cut-off result is fenced and returned. A subclass
                                # redacting or annotating results must not have this one escape.
                                cut = await self.after_tool(run.state(), skipped, _CUT_OFF)
                                blocks.append(await self._record(run, skipped, cut))
                            exceeded = BudgetExceeded(
                                "wall_clock", run.budget.elapsed, run.budget.max_seconds
                            )
                            run.messages.append(Message("user", tuple(blocks)))
                            return await self._finish(
                                run, EndReason.BUDGET_EXCEEDED, exceeded.message, exceeded
                            )
                        blocks.append(await self._run_tool(run, call))
                    run.messages.append(Message("user", tuple(blocks)))
                    stop = await self.should_stop(run.state())
                    if stop is not None:
                        return await self._finish(run, EndReason.STOPPED_BY_HOOK, stop)
                case StopReason.END_TURN:
                    return await self._finish(run, EndReason.END_TURN)
                case StopReason.MAX_TOKENS:
                    return await self._finish(run, EndReason.TRUNCATED)
                case StopReason.STOP_SEQUENCE:
                    return await self._finish(run, EndReason.STOP_SEQUENCE)
                case StopReason.GUARDRAIL_INTERVENED:
                    return await self._finish(run, EndReason.GUARDRAIL_INTERVENED)
                case StopReason.CONTENT_FILTERED:
                    return await self._finish(run, EndReason.CONTENT_FILTERED)
                case _:
                    assert_never(reason)

    async def _call_model(self, run: _Run, request: ModelRequest) -> ModelResponse | None:
        """One model call, bounded by the wall-clock budget. `None`: the clock ran out."""
        try:
            async with asyncio.timeout(run.budget.remaining_seconds) as scope:
                response = await self._provider.converse(request)
        except TimeoutError:
            if scope.expired():
                return None
            raise
        run.budget.record_turn(response.usage.total)
        run.input_tokens += response.usage.input_tokens
        run.output_tokens += response.usage.output_tokens
        run.last = response
        await self._publish(
            run,
            LoopEventKind.MODEL_CALLED,
            detail=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

    async def _run_tool(self, run: _Run, call: ToolUseBlock) -> ToolResultBlock:
        """Run one requested tool and return its fenced result. Unregistered tools never run."""
        tool = self._tools.get(call.name) if _TOOL_NAME.match(call.name) else None
        if tool is None:
            # Never echo the model's own string back: `call.name` reaches the fenced body, and a
            # name carrying a re-forming nonce could close the fence early. A tool name is an
            # identifier, so anything outside the allowlist is simply unavailable.
            result = ToolResult.error(
                ErrorCategory.PERMISSION, "that tool is not available to this agent"
            )
        else:
            ctx = ToolContext(run.run_id, run.turn, call.id, self._identity)
            try:
                async with asyncio.timeout(run.budget.remaining_seconds) as scope:
                    result = await tool.invoke(call.input, ctx)
            except Exception as exc:
                if isinstance(exc, TimeoutError) and scope.expired():
                    result = _CUT_OFF
                else:
                    # Log the type and the tool, not the message: it may carry user data.
                    _log.error("tool %s raised %s", _safe_name(call.name), type(exc).__name__)
                    result = ToolResult.error(
                        ErrorCategory.INTERNAL,
                        "the tool failed with an internal error and cannot be retried",
                    )
        result = await self.after_tool(run.state(), call, result)

        # Every sink takes the sanitised name, not just the fence and the log line. The event bus
        # is where 17.5's JSON-lines logger sits, and RunResult is what a caller reports on, so a
        # model-chosen name must not arrive raw at either.
        return await self._record(run, call, result)

    async def _record(self, run: _Run, call: ToolUseBlock, result: ToolResult) -> ToolResultBlock:
        """Turn a result into the block that answers `call`, recording and publishing it.

        Every `ToolUseBlock` must be answered by a `ToolResultBlock` carrying the same id
        (`messages.py`), or the transcript cannot be sent again -- which is what app 19's session
        resumption does. So a call the loop declines to dispatch is answered here too, not left
        dangling.
        """
        safe = _safe_name(call.name)
        run.tool_calls.append(
            ToolCallRecord(run.turn, safe, call.id, result.is_error, result.error_category)
        )
        if result.staged is not None:
            run.staged.append(result.staged)
        await self._publish(run, LoopEventKind.TOOL_CALLED, tool=safe)
        if result.is_error:
            category = result.error_category.value if result.error_category else ""
            await self._publish(run, LoopEventKind.TOOL_FAILED, tool=safe, detail=category)
        return ToolResultBlock(
            tool_use_id=call.id,
            content=fence_tool_output(
                # Sanitised, not raw: the header reads as library-written, so a model-chosen name
                # carrying instructions is worse there than in the body.
                tool=safe,
                tool_use_id=call.id,
                content=result.content,
                nonce=run.nonce,
                category=result.error_category,
                retryable=result.is_retryable,
            ),
            is_error=result.is_error,
        )

    async def _finish(
        self,
        run: _Run,
        end_reason: EndReason,
        detail: str | None = None,
        exceeded: BudgetExceeded | None = None,
    ) -> RunResult:
        if exceeded is not None:
            await self._publish(run, LoopEventKind.BUDGET_EXCEEDED, detail=exceeded.message)
        return RunResult(
            run_id=run.run_id,
            final_text=run.last.text if run.last else "",
            stop_reason=run.last.stop_reason if run.last else None,
            end_reason=end_reason,
            end_detail=detail or _END_DETAIL[end_reason],
            turns_used=run.budget.turns_used,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            tool_calls=tuple(run.tool_calls),
            staged_changes=tuple(run.staged),
            budget_exceeded=exceeded,
            messages=tuple(run.messages),
        )

    async def _publish(
        self,
        run: _Run,
        kind: LoopEventKind,
        *,
        tool: str | None = None,
        detail: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        event = LoopEvent(kind, run.run_id, run.turn, tool, detail, input_tokens, output_tokens)
        for observer in self._observers:
            try:
                # Bounded, and not by the run budget. An observer must not be able to hang a run:
                # the budget's own timeouts wrap the model call and each tool, and publishing
                # happens *outside* them, so an unbounded `on_event` would outlive the budget it
                # is reporting on -- including the final BUDGET_EXCEEDED event, and the publish
                # that follows an `end_turn` response, which would strand a finished answer.
                async with asyncio.timeout(OBSERVER_TIMEOUT_SECONDS):
                    await observer.on_event(event)
            except TimeoutError:
                _log.warning(
                    "observer %s timed out after %ss on %s",
                    type(observer).__name__,
                    OBSERVER_TIMEOUT_SECONDS,
                    kind,
                )
            except Exception as exc:
                _log.warning(
                    "observer %s failed on %s: %s",
                    type(observer).__name__,
                    kind,
                    type(exc).__name__,
                )


__all__ = [
    "MAX_USER_MESSAGE_CHARS",
    "AgentLoop",
    "EndReason",
    "LoopEvent",
    "LoopEventKind",
    "LoopObserver",
    "RunResult",
    "RunState",
    "ToolCallRecord",
]
