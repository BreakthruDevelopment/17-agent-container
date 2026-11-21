"""Checks against botocore's own copy of the Converse API model, which ships offline.

This is the oracle for what the API *says*. It settles three things as fact: the set of stop
reasons, the shape of what `BedrockProvider` sends, and the shape of what the test stub returns.
It does not settle runtime rules such as role alternation or blank-text rejection: those are not in
the schema, they depend on the model family, and `tests.support.check_transcript` labels them as the
assumptions they are.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.model import OperationModel, StructureShape
from botocore.session import get_session
from botocore.validate import ParamValidator

from agent_container.core import Message, ModelRequest, ModelResponse, StopReason, ToolSpec
from agent_container.core.bedrock import build_request
from agent_container.core.messages import (
    TextBlock,
    ToolChoice,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from tests.support import to_wire


@pytest.fixture(scope="module")
def converse() -> OperationModel:
    return get_session().get_service_model("bedrock-runtime").operation_model("Converse")


def errors(shape: Any, params: dict[str, Any]) -> str:
    report = ParamValidator().validate(params, shape)
    return report.generate_report() if report.has_errors() else ""


def test_stop_reason_is_exactly_the_set_the_api_defines(converse: OperationModel) -> None:
    output = converse.output_shape
    assert isinstance(output, StructureShape)
    assert set(output.members["stopReason"].enum) == {r.value for r in StopReason}


def test_a_full_request_is_valid_against_the_converse_input_shape(converse: OperationModel) -> None:
    request = ModelRequest(
        system="sys",
        messages=(
            Message.user_text("find it"),
            Message(
                "assistant",
                (
                    TextBlock("looking"),
                    ToolUseBlock("t1", "lookup", {"q": "x", "n": [1, {"a": 2}]}),
                ),
            ),
            Message(
                "user", (ToolResultBlock("t1", "found", False), ToolResultBlock("t2", "no", True))
            ),
        ),
        tools=(ToolSpec("lookup", "Look it up.", {"type": "object", "properties": {}}),),
        max_tokens=256,
        tool_choice=ToolChoice("tool", "lookup"),
    )
    assert errors(converse.input_shape, build_request("anthropic.model-v1:0", request)) == ""


@pytest.mark.parametrize("choice", [ToolChoice(), ToolChoice("any")])
def test_every_tool_choice_is_valid(converse: OperationModel, choice: ToolChoice) -> None:
    request = ModelRequest(
        "s",
        (Message.user_text("x"),),
        tools=(ToolSpec("t", "d", {"type": "object"}),),
        tool_choice=choice,
    )
    assert errors(converse.input_shape, build_request("m", request)) == ""


def test_the_validator_does_reject_a_bad_request(converse: OperationModel) -> None:
    """The control: the checks above bite."""
    wire = build_request("m", ModelRequest("s", (Message.user_text("x"),)))
    wire["messages"][0]["role"] = 7
    assert errors(converse.input_shape, wire) != ""


@pytest.mark.parametrize("reason", [r.value for r in StopReason])
def test_the_stub_returns_what_the_api_returns(converse: OperationModel, reason: str) -> None:
    reply = ModelResponse.of(
        "x", stop_reason=reason, tool_uses=[ToolUseBlock("a", "n", {"k": 1})], usage=Usage(3, 4)
    )
    assert errors(converse.output_shape, to_wire(reply)) == ""
