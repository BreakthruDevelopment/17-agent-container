"""Test doubles shared across the suite."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_container.core import (
    AnyTool,
    Effect,
    Identity,
    Intercepted,
    Interceptor,
    InterceptorChain,
    ModelRequest,
    ModelResponse,
    PermissionGate,
    Policy,
    ReplayProvider,
    StagedChange,
    TextBlock,
    Tool,
    ToolContext,
    ToolResult,
    ToolUseBlock,
    Usage,
)
from agent_container.core.decorators import MAX_DESCRIPTION_CHARS
from agent_container.testing import OverlapProbe


class ProbedReplay(ReplayProvider):
    """A `ReplayProvider` that honours an `OverlapProbe`, for the provider contract."""

    def __init__(self, replies: Sequence[ModelResponse], probe: OverlapProbe | None = None) -> None:
        super().__init__(replies)
        self._probe = probe

    async def converse(self, request: ModelRequest) -> ModelResponse:
        if self._probe is not None:
            await self._probe.wait_async()
        return await super().converse(request)


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool[EchoArgs]):
    """A read tool that returns what it is given, or whatever `reply` says."""

    name = "echo"
    description = "Echo the text back."
    effect = Effect.READ
    input_model = EchoArgs

    def __init__(self, reply: Callable[[EchoArgs], ToolResult] | None = None) -> None:
        self.reply = reply
        self.calls: list[EchoArgs] = []
        self.contexts: list[ToolContext] = []

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        self.contexts.append(ctx)
        return self.reply(args) if self.reply else ToolResult.ok(f"echo: {args.text}")


class DeleteTool(Tool[EchoArgs]):
    """A write tool, for the effect tests."""

    name = "delete"
    description = "Delete something."
    effect = Effect.WRITE
    input_model = EchoArgs

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("deleted")


class PassThrough[A: BaseModel](Tool[A]):
    """The smallest decorator: wraps a tool and forwards everything (17.4's have this shape)."""

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


class FakeGate[A: BaseModel](PassThrough[A]):
    """A stand-in for 17.4's `PermissionGate`: it only advertises the marker.

    17.2 owns the construction-time check, not the gate, so the tests need something that
    satisfies the contract without 17.4 existing. It stages nothing and approves nothing -- when
    `PermissionGate` lands, these tests should keep passing against the real one.
    """

    gates_writes = True


class LyingDecorator[A: BaseModel](PassThrough[A]):
    """Reports `READ` over whatever it wraps: the bypass the gate check must not allow."""

    @property
    def effect(self) -> Effect:
        return Effect.READ


class WriteClaimingDecorator[A: BaseModel](PassThrough[A]):
    """Reports `WRITE` over a read-only leaf -- a wrapper that writes something of its own.

    This is the stack that makes `writes()` need `any(...)`: the leaf is honest and read-only, so
    `innermost(...).effect` would call the whole stack a read and let it past the gate check.
    """

    @property
    def effect(self) -> Effect:
        return Effect.WRITE


class TruthyGate[A: BaseModel](PassThrough[A]):
    """Advertises the marker with a truthy non-`True` value: must not pass for a gate."""

    gates_writes = "yes"  # not `True`: wrong on purpose


class StringEffectTool(Tool[EchoArgs]):
    """A write tool whose `effect` is the plain string, not the enum member.

    `Effect` is a `StrEnum`, so this compares equal to `Effect.WRITE` but is not identical to it.
    A gate check written with `is` would call this a read tool and let it through ungated.
    """

    name = "delete"
    description = "Delete something."
    effect = "write"  # type: ignore[assignment]  # the whole point of the test
    input_model = EchoArgs

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("deleted")


class SelfApprovingWriteTool(Tool[EchoArgs]):
    """A write leaf that claims to be its own gate -- it wraps nothing, so it cannot be one."""

    name = "delete"
    description = "Delete something."
    effect = Effect.WRITE
    input_model = EchoArgs
    gates_writes = True

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("deleted")


def tool_call(text: str = "hi", *, id: str = "call-1", name: str = "echo") -> ToolUseBlock:
    return ToolUseBlock(id, name, {"text": text})


def use_tool(*calls: ToolUseBlock, text: str = "", usage: Usage | None = None) -> ModelResponse:
    return ModelResponse.of(text, stop_reason="tool_use", tool_uses=calls, usage=usage)


def scripted(*replies: ModelResponse) -> ReplayProvider:
    return ReplayProvider(list(replies))


# -- a stand-in for the boto3 bedrock-runtime client ---------------------------------------------


def to_wire(reply: ModelResponse) -> dict[str, Any]:
    """A `ModelResponse` as the dict Bedrock's Converse API returns."""
    content: list[dict[str, Any]] = []
    for block in reply.content:
        if isinstance(block, TextBlock):
            content.append({"text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append(
                {"toolUse": {"toolUseId": block.id, "name": block.name, "input": dict(block.input)}}
            )
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": reply.stop_reason,
        "usage": {
            "inputTokens": reply.usage.input_tokens,
            "outputTokens": reply.usage.output_tokens,
            "totalTokens": reply.usage.total,  # required by the schema
        },
        "metrics": {"latencyMs": 1},  # required by the schema
    }


def check_transcript(messages: Sequence[Mapping[str, Any]]) -> None:
    """Rules a resumed transcript must satisfy; raises `ValueError` on a break.

    ASSUMPTIONS, not the API contract. Neither rule is in botocore's Converse schema (see
    `tests/test_converse_schema.py` for what is): both are runtime rules that Bedrock applies per
    model family. They are encoded here because the Anthropic models this track targets enforce
    them, and a transcript that breaks them is what app 19's resumption must not send.

    * roles alternate, starting with the user;
    * every `toolUse` is answered by a `toolResult` with the same id in the very next message, and
      a `toolResult` answers a call that was actually made.
    """
    pending: set[str] = set()
    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(f"message {i} is {message['role']}, expected {expected}")
        results = {b["toolResult"]["toolUseId"] for b in message["content"] if "toolResult" in b}
        if results != pending:
            raise ValueError(f"message {i} answers {sorted(results)}, expected {sorted(pending)}")
        pending = {b["toolUse"]["toolUseId"] for b in message["content"] if "toolUse" in b}
    if pending and messages and messages[-1]["role"] == "assistant":
        raise ValueError(f"tool calls {sorted(pending)} were never answered")


class StubConverseClient:
    """Answers `converse` with the given replies in order and records every request.

    No network, no boto3. With `strict=True` it also refuses a transcript Bedrock would refuse.
    """

    def __init__(
        self,
        replies: Sequence[ModelResponse] = (),
        *,
        strict: bool = False,
        probe: OverlapProbe | None = None,
    ) -> None:
        self._probe = probe
        self._wire = [to_wire(r) for r in replies]
        self._lock = threading.Lock()  # `converse` runs on worker threads
        self._strict = strict
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> Mapping[str, Any]:
        if self._probe is not None:
            self._probe.wait_blocking()  # on the provider's worker thread
        if self._strict:
            check_transcript(kwargs["messages"])
        with self._lock:
            self.requests.append(copy.deepcopy(kwargs))
            if not self._wire:
                raise AssertionError("the stub client ran out of replies")
            return self._wire.pop(0)


class ClampArgs(BaseModel):
    amount: int = 5000


class ClampingPay(Tool[ClampArgs]):
    """A write tool that records what it was actually asked to pay."""

    name = "pay"
    description = "Pay a supplier."
    effect = Effect.WRITE
    input_model = ClampArgs

    def __init__(self, description: str | None = None) -> None:
        self.paid: list[int] = []
        # The same record at model granularity: the idiom `EchoTool.calls` uses. A probe reporting
        # this instead of the scalar defeated an earlier, one-sided version of
        # `StagedChangeContract`, so a case using it is checked in.
        self.received: list[ClampArgs] = []
        if description is not None:
            self.description = description

    async def run(self, args: ClampArgs, ctx: ToolContext) -> ToolResult:
        self.paid.append(args.amount)
        self.received.append(args)
        return ToolResult.ok(f"paid {args.amount}")


class ClampTo(Interceptor):
    """A `before` hook that rewrites an argument in place, as a policy hook legitimately may.

    Positioned *beneath* a gate it holds the tool's own authority, so this is not a privilege
    escalation -- which is exactly why the gate cannot rely on the arguments staying put.
    """

    def __init__(self, amount: int = 100) -> None:
        self.amount = amount

    async def before(self, tool: str, args: BaseModel, ctx: ToolContext) -> ToolResult | None:
        if isinstance(args, ClampArgs):
            args.amount = self.amount
        return None


def staged_pay_behind_a_clamp(
    *, amount: int = 5000, clamp_to: int = 100, long_description: bool = False
) -> tuple[StagedChange, ClampingPay]:
    """Stage one `pay` call through a gate with a clamping interceptor beneath it.

    Returns the staged change and the leaf tool, so a caller can see both the record and what ran.
    `long_description` pushes `full_description` past `MAX_DESCRIPTION_CHARS` so that
    `limit_description` truncates the arguments off the end -- the shape that defeated an earlier
    version of `StagedChangeContract`.
    """
    pay = ClampingPay(
        "Pay a supplier. " + "x" * MAX_DESCRIPTION_CHARS if long_description else None
    )
    gate = PermissionGate(
        Intercepted(pay, InterceptorChain([ClampTo(clamp_to)])),
        Policy.STAGED_WRITE,
        requires="pay.write",
    )
    ctx = ToolContext("run-1", 1, "call-1", identity=Identity("u1", frozenset({"pay.write"})))
    result = asyncio.run(gate.run(ClampArgs(amount=amount), ctx))
    change = result.staged
    assert change is not None, "the gate did not stage the write"
    return change, pay
