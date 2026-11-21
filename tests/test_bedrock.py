"""`BedrockProvider` against a stubbed client. Nothing here touches AWS."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import JsonValue

from agent_container.core import (
    BedrockProvider,
    MalformedModelResponse,
    Message,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ToolChoice,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from agent_container.core import bedrock as bedrock_module
from agent_container.core.messages import TextBlock
from agent_container.testing import ModelProviderContract, OverlapProbe
from tests.support import StubConverseClient, to_wire

# The autouse fixture replaces `make_client` per test; this is the real one, for the config tests.
REAL_MAKE_CLIENT = bedrock_module.make_client
MODEL = "anthropic.test-model-v1:0"


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that reaches for the real boto3 client fails instead of reaching AWS."""

    def refuse(region_name: str | None = None) -> None:
        raise AssertionError("a test tried to build a real Bedrock client")

    monkeypatch.setattr(bedrock_module, "make_client", refuse)


class TestBedrockProviderContract(ModelProviderContract):
    def make_provider(
        self, replies: Sequence[ModelResponse], probe: OverlapProbe | None = None
    ) -> ModelProvider:
        return BedrockProvider(MODEL, client=StubConverseClient(replies, probe=probe))


def request(**overrides: Any) -> ModelRequest:
    fields: dict[str, Any] = {"system": "sys", "messages": (Message.user_text("hi"),)}
    fields.update(overrides)
    return ModelRequest(**fields)


async def call(req: ModelRequest, reply: ModelResponse | None = None) -> dict[str, Any]:
    client = StubConverseClient([reply or ModelResponse.of("ok")])
    await BedrockProvider(MODEL, client=client).converse(req)
    return client.requests[0]


async def test_the_request_is_translated_to_converse_shape() -> None:
    schema: dict[str, JsonValue] = {"type": "object", "properties": {"q": {"type": "string"}}}
    wire = await call(
        request(
            messages=(
                Message.user_text("find it"),
                Message(
                    "assistant", (TextBlock("looking"), ToolUseBlock("t1", "lookup", {"q": "x"}))
                ),
                Message("user", (ToolResultBlock("t1", "found", False),)),
            ),
            tools=(ToolSpec("lookup", "Look it up.", schema),),
            max_tokens=256,
            tool_choice=ToolChoice("tool", "lookup"),
        )
    )
    assert wire["modelId"] == MODEL
    assert wire["system"] == [{"text": "sys"}]
    assert wire["inferenceConfig"] == {"maxTokens": 256}
    assert wire["messages"][1]["content"][1] == {
        "toolUse": {"toolUseId": "t1", "name": "lookup", "input": {"q": "x"}}
    }
    assert wire["messages"][2]["content"] == [
        {"toolResult": {"toolUseId": "t1", "content": [{"text": "found"}], "status": "success"}}
    ]
    assert wire["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "lookup",
                    "description": "Look it up.",
                    "inputSchema": {"json": schema},
                }
            }
        ],
        "toolChoice": {"tool": {"name": "lookup"}},
    }


async def test_an_error_result_is_marked_as_an_error() -> None:
    wire = await call(request(messages=(Message("user", (ToolResultBlock("t1", "no", True),)),)))
    assert wire["messages"][0]["content"][0]["toolResult"]["status"] == "error"


@pytest.mark.parametrize(
    ("choice", "expected"),
    [(ToolChoice(), {"auto": {}}), (ToolChoice("any"), {"any": {}})],
)
async def test_tool_choice_modes(choice: ToolChoice, expected: dict[str, Any]) -> None:
    wire = await call(request(tools=(ToolSpec("t", "d", {"type": "object"}),), tool_choice=choice))
    assert wire["toolConfig"]["toolChoice"] == expected


async def test_optional_parts_are_left_out_when_unset() -> None:
    wire = await call(request(system=" "))
    assert set(wire) == {"modelId", "messages"}


async def test_blank_text_blocks_are_dropped_and_an_empty_message_is_refused() -> None:
    wire = await call(request(messages=(Message("user", (TextBlock(" "), TextBlock("real"))),)))
    assert wire["messages"][0]["content"] == [{"text": "real"}]
    with pytest.raises(ValueError, match="no content"):
        await call(request(messages=(Message("user", (TextBlock(""),)),)))


async def test_the_wire_request_shares_nothing_with_the_callers_objects() -> None:
    nested: dict[str, JsonValue] = {"a": [1, {"b": 2}]}
    sent: dict[str, Any] = {}

    class Spy:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            sent.update(kwargs)
            return to_wire(ModelResponse.of("ok"))

    req = request(
        messages=(Message("assistant", (ToolUseBlock("t", "n", nested),)), Message.user_text("x"))
    )
    await BedrockProvider(MODEL, client=Spy()).converse(req)
    sent["messages"][0]["content"][0]["toolUse"]["input"]["a"][1]["b"] = 99
    assert nested == {"a": [1, {"b": 2}]}


async def test_the_response_is_translated_back_with_usage() -> None:
    reply = ModelResponse.of(
        "thinking",
        stop_reason="tool_use",
        tool_uses=[ToolUseBlock("t9", "lookup", {"q": {"deep": [1, 2]}})],
        usage=Usage(31, 5),
    )
    got = await BedrockProvider(MODEL, client=StubConverseClient([reply])).converse(request())
    assert got == reply


@pytest.mark.parametrize(
    "stop_reason",
    [
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
        "malformed_model_output",
        "malformed_tool_use",
        "model_context_window_exceeded",
        "a_value_from_the_future",
    ],
)
async def test_the_stop_reason_is_returned_exactly_as_bedrock_said_it(stop_reason: str) -> None:
    client = StubConverseClient([ModelResponse.of("x", stop_reason=stop_reason)])
    got = await BedrockProvider(MODEL, client=client).converse(request())
    assert got.stop_reason == stop_reason


async def test_content_blocks_with_no_matching_type_are_ignored() -> None:
    class Odd:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "output": {
                    "message": {"content": [{"reasoningContent": {"x": 1}}, {"text": "answer"}]}
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 2},
            }

    got = await BedrockProvider(MODEL, client=Odd()).converse(request())
    assert got.text == "answer"


@pytest.mark.parametrize(
    "wire",
    [
        {"output": {"message": {"content": []}}},
        {"stopReason": 7},
        {"stopReason": "end_turn", "usage": {"inputTokens": -1}},
        {"stopReason": "end_turn", "usage": {"inputTokens": True}},
        {
            "stopReason": "tool_use",
            "output": {
                "message": {"content": [{"toolUse": {"toolUseId": "a", "name": "n", "input": "s"}}]}
            },
        },
    ],
)
async def test_a_malformed_response_raises(wire: dict[str, Any]) -> None:
    class Bad:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            return wire

    with pytest.raises(MalformedModelResponse):
        await BedrockProvider(MODEL, client=Bad()).converse(request())


async def test_no_client_is_built_until_the_first_call() -> None:
    BedrockProvider(MODEL)  # the autouse fixture would fail the test if this built one


def test_a_model_id_is_required() -> None:
    with pytest.raises(ValueError):
        BedrockProvider(" ")


async def test_a_client_error_propagates_untouched() -> None:
    class Throttled(Exception):
        pass

    class Failing:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise Throttled

    with pytest.raises(Throttled):
        await BedrockProvider(MODEL, client=Failing()).converse(request())


async def test_the_lazy_client_is_built_on_a_worker_thread_not_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import boto3` and the credential lookup take hundreds of ms; the loop must not pay it.

    Asserted by thread identity, which is deterministic. A bound on loop stall time would also
    count time the process spends descheduled, so it is deliberately not used.
    """
    built_on: list[int] = []

    def slow_make_client(region_name: str | None = None) -> StubConverseClient:
        built_on.append(threading.get_ident())
        time.sleep(0.3)
        return StubConverseClient([ModelResponse.of("ok")])

    monkeypatch.setattr(bedrock_module, "make_client", slow_make_client)
    await BedrockProvider(MODEL).converse(request())
    assert built_on
    assert threading.get_ident() not in built_on


# -- the real client's configuration (built offline: no request is made) -----------------------


@pytest.fixture
def fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("AWS_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("AWS_RETRY_MODE", raising=False)


def test_the_real_client_makes_exactly_one_attempt(
    monkeypatch: pytest.MonkeyPatch, fake_credentials: None
) -> None:
    """A botocore retry runs inside the worker thread, past the limiter, so there must be none.

    Asserts the *resolved* configuration: botocore rewrites what it is given, and `max_attempts`
    means retries after the first request, so `max_attempts=1` resolves to two attempts.
    """
    # Even an environment that asks for more must not win over the explicit setting.
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "5")
    client: Any = REAL_MAKE_CLIENT("us-east-1")
    retries = client.meta.config.retries
    assert retries["total_max_attempts"] == 1
    assert "max_attempts" not in retries


def test_the_trap_this_guards_against_is_real(fake_credentials: None) -> None:
    """Control: `max_attempts=1`, as first written, resolves to two attempts."""
    import boto3
    from botocore.config import Config

    client: Any = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=Config(retries={"max_attempts": 1, "mode": "standard"}),
    )
    assert client.meta.config.retries["total_max_attempts"] == 2


def test_the_real_client_keeps_its_timeouts(fake_credentials: None) -> None:
    client: Any = REAL_MAKE_CLIENT("us-east-1")
    assert client.meta.config.connect_timeout == bedrock_module.DEFAULT_CONNECT_TIMEOUT_SECONDS
    assert client.meta.config.read_timeout == bedrock_module.DEFAULT_READ_TIMEOUT_SECONDS
