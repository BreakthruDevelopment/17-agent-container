"""The shared contract suites: what every `ModelProvider` and every `Tool` (decorators included)
must do.

Substitutability is proven, not asserted: a provider swap or a decorator stack is safe because each
implementation passes the same tests. 17.3 runs `BedrockProvider` (against a stubbed client) through
`ModelProviderContract`; 17.4 runs `Retrying`, `Cached`, `Redacting`, `Metered` and `PermissionGate`
through `ToolContract`.

The suites are plain classes whose tests are synchronous, so they need no pytest plugin. They do not
import pytest. Checks raise `AssertionError` explicitly so they still fail under `python -O`.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, JsonValue

from agent_container.core.errors import UnrecognizedStopReason
from agent_container.core.messages import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolUseBlock,
    Usage,
)
from agent_container.core.provider import ModelProvider
from agent_container.core.tool import (
    AnyTool,
    Effect,
    ErrorCategory,
    ToolContext,
    ToolResult,
    innermost,
    walk_stack,
)

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _request(text: str = "hello") -> ModelRequest:
    return ModelRequest(system="You are a test.", messages=(Message.user_text(text),))


class ModelProviderContract(ABC):
    """Subclass, implement `make_provider`, and every test below runs against your provider."""

    @abstractmethod
    def make_provider(self, replies: Sequence[ModelResponse]) -> ModelProvider:
        """A provider that answers successive calls with exactly these replies, in order."""

    def test_converse_is_async(self) -> None:
        provider = self.make_provider([ModelResponse.of("hi")])
        _require(inspect.iscoroutinefunction(provider.converse), "converse must be `async def`")

    def test_returns_text_and_stop_reason(self) -> None:
        provider = self.make_provider([ModelResponse.of("hello there", stop_reason="end_turn")])
        response = asyncio.run(provider.converse(_request()))
        _require(isinstance(response, ModelResponse), "converse must return a ModelResponse")
        _require(response.text == "hello there", f"text was {response.text!r}")
        _require(response.stop_reason == "end_turn", f"stop reason was {response.stop_reason!r}")

    def test_returns_tool_calls_intact(self) -> None:
        call = ToolUseBlock("call-1", "lookup", {"q": "x", "n": 2})
        provider = self.make_provider([ModelResponse.of(stop_reason="tool_use", tool_uses=[call])])
        response = asyncio.run(provider.converse(_request()))
        _require(response.stop_reason == "tool_use", "tool_use stop reason lost")
        _require(response.tool_uses == (call,), f"tool call changed: {response.tool_uses!r}")

    def test_reports_usage(self) -> None:
        provider = self.make_provider([ModelResponse.of("x", usage=Usage(12, 7))])
        response = asyncio.run(provider.converse(_request()))
        _require(response.usage == Usage(12, 7), f"usage was {response.usage!r}")

    def test_passes_every_known_stop_reason_through_unchanged(self) -> None:
        for reason in StopReason:
            provider = self.make_provider([ModelResponse.of("x", stop_reason=reason.value)])
            response = asyncio.run(provider.converse(_request()))
            _require(
                response.stop_reason == reason.value,
                f"{reason.value!r} became {response.stop_reason!r}",
            )

    def test_never_maps_an_unknown_stop_reason_onto_a_known_one(self) -> None:
        provider = self.make_provider([ModelResponse.of("x", stop_reason="not_a_real_reason")])
        try:
            response = asyncio.run(provider.converse(_request()))
        except UnrecognizedStopReason:
            return  # refusing is fine; the loop would have raised too
        _require(
            response.stop_reason == "not_a_real_reason",
            f"an unknown stop reason was turned into {response.stop_reason!r}",
        )

    def test_does_not_modify_the_request(self) -> None:
        provider = self.make_provider([ModelResponse.of("x")])
        request = _request()
        before = copy.deepcopy(request)
        asyncio.run(provider.converse(request))
        _require(request == before, "converse changed the request it was given")

    def test_concurrent_calls_each_get_a_reply(self) -> None:
        replies = [ModelResponse.of(f"reply {i}") for i in range(3)]
        provider = self.make_provider(replies)

        async def fan_out() -> list[ModelResponse]:
            return list(await asyncio.gather(*(provider.converse(_request()) for _ in replies)))

        got = asyncio.run(fan_out())
        _require(
            sorted(r.text for r in got) == sorted(r.text for r in replies),
            "concurrent calls did not each receive one reply",
        )


class ToolContract(ABC):
    """Subclass and supply a tool plus one valid and one invalid input. Use it for leaf tools and
    for every decorator stack: the same tests hold for both, which is what makes a decorator
    substitutable."""

    @abstractmethod
    def make_tool(self) -> AnyTool:
        """A fresh tool (or decorator stack) to test."""

    @abstractmethod
    def valid_input(self) -> Mapping[str, JsonValue]:
        """Arguments that pass the tool's input model."""

    @abstractmethod
    def invalid_input(self) -> Mapping[str, JsonValue]:
        """Arguments that fail the tool's input model."""

    def _ctx(self) -> ToolContext:
        return ToolContext(run_id="run-1", turn=1, tool_use_id="call-1")

    def test_has_a_usable_identity(self) -> None:
        tool = self.make_tool()
        _require(bool(_TOOL_NAME.match(tool.name)), f"bad tool name {tool.name!r}")
        _require(bool(tool.description.strip()), "a tool needs a description")
        _require(isinstance(tool.effect, Effect), "effect must be an Effect, read or write")
        _require(issubclass(tool.input_model, BaseModel), "input_model must be a pydantic model")
        _require(
            tool.input_schema().get("type") == "object", "the input schema must describe an object"
        )

    def test_run_is_async(self) -> None:
        _require(inspect.iscoroutinefunction(self.make_tool().invoke), "invoke must be `async def`")

    def test_stack_can_be_walked_to_the_innermost_tool(self) -> None:
        tool = self.make_tool()
        stack = list(walk_stack(tool))
        _require(stack[0] is tool, "the walk must start at the tool itself")
        _require(innermost(tool) is stack[-1], "innermost must be the last tool in the walk")
        _require(stack[-1].wrapped is None, "the innermost tool must not wrap anything")

    def test_decorators_inherit_the_innermost_effect_and_identity(self) -> None:
        tool = self.make_tool()
        leaf = innermost(tool)
        for layer in walk_stack(tool):
            _require(layer.effect == leaf.effect, f"{type(layer).__name__} changed the effect")
            _require(layer.name == leaf.name, f"{type(layer).__name__} changed the name")
            _require(
                layer.input_model is leaf.input_model,
                f"{type(layer).__name__} changed the input model",
            )

    def test_valid_input_returns_a_result(self) -> None:
        result = asyncio.run(self.make_tool().invoke(self.valid_input(), self._ctx()))
        _require(isinstance(result, ToolResult), "invoke must return a ToolResult")

    def test_invalid_input_is_a_validation_result_not_an_exception(self) -> None:
        result = asyncio.run(self.make_tool().invoke(self.invalid_input(), self._ctx()))
        _require(result.is_error, "invalid input must produce an error result")
        _require(
            result.error_category is ErrorCategory.VALIDATION,
            f"category was {result.error_category!r}",
        )

    def test_a_second_call_works(self) -> None:
        tool = self.make_tool()
        asyncio.run(tool.invoke(self.valid_input(), self._ctx()))
        again = asyncio.run(tool.invoke(self.valid_input(), self._ctx()))
        _require(isinstance(again, ToolResult), "a tool must be callable more than once")
