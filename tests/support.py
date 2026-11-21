"""Test doubles shared across the suite."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from agent_container.core import (
    AnyTool,
    Effect,
    ModelResponse,
    ReplayProvider,
    Tool,
    ToolContext,
    ToolResult,
    ToolUseBlock,
    Usage,
)


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
