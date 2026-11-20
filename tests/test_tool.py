from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_container.core import (
    Effect,
    ErrorCategory,
    Tool,
    ToolContext,
    ToolResult,
    innermost,
    walk_stack,
)
from tests.support import DeleteTool, EchoArgs, EchoTool, PassThrough

CTX = ToolContext(run_id="r", turn=1, tool_use_id="c")


def test_a_tool_must_declare_its_effect() -> None:
    class NoEffect(Tool[EchoArgs]):
        name = "x"
        description = "x"
        input_model = EchoArgs

        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            return ToolResult.ok("")

    with pytest.raises(TypeError, match="effect"):
        NoEffect()  # type: ignore[abstract]  # the point of the test


def test_effect_is_read_or_write() -> None:
    assert {e.value for e in Effect} == {"read", "write"}


def test_the_innermost_tool_is_reachable_through_a_stack() -> None:
    leaf = DeleteTool()
    stack = PassThrough(PassThrough(PassThrough(leaf)))
    assert [type(t).__name__ for t in walk_stack(stack)] == [
        "PassThrough",
        "PassThrough",
        "PassThrough",
        "DeleteTool",
    ]
    assert innermost(stack) is leaf
    assert innermost(stack).effect is Effect.WRITE
    assert innermost(leaf) is leaf


def test_a_stack_that_wraps_itself_is_detected() -> None:
    a = PassThrough(EchoTool())
    a._inner = a  # build the cycle on purpose
    with pytest.raises(ValueError, match="wraps itself"):
        innermost(a)


def test_the_input_schema_is_json_schema() -> None:
    schema = EchoTool().input_schema()
    assert schema["type"] == "object"
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "text" in properties


async def test_invalid_input_names_the_field_and_never_echoes_the_value() -> None:
    marker = "hunter2-do-not-echo"
    result = await EchoTool().invoke({"text": {"nested": marker}}, CTX)
    assert result.is_error
    assert result.error_category is ErrorCategory.VALIDATION
    assert "text" in result.content
    assert marker not in result.content


async def test_a_decorator_validates_with_the_wrapped_tools_model() -> None:
    inner = EchoTool()
    result = await PassThrough(inner).invoke({"text": "ok"}, CTX)
    assert not result.is_error
    assert [c.text for c in inner.calls] == ["ok"]


def test_input_model_is_a_pydantic_model() -> None:
    assert issubclass(EchoTool().input_model, BaseModel)


# -- ToolResult ----------------------------------------------------------------------------------


def test_ok_result_carries_no_error_fields() -> None:
    r = ToolResult.ok("fine")
    assert (r.is_error, r.error_category, r.is_retryable) == (False, None, False)


@pytest.mark.parametrize(
    ("category", "retryable"),
    [
        (ErrorCategory.TRANSIENT, True),
        (ErrorCategory.VALIDATION, False),
        (ErrorCategory.PERMISSION, False),
        (ErrorCategory.BUSINESS, False),
    ],
)
def test_error_results_default_retryability_by_category(
    category: ErrorCategory, retryable: bool
) -> None:
    r = ToolResult.error(category, "nope")
    assert r.is_error
    assert r.error_category is category
    assert r.is_retryable is retryable


def test_retryability_can_be_overridden() -> None:
    assert ToolResult.error(ErrorCategory.VALIDATION, "x", retryable=True).is_retryable
    assert not ToolResult.error(ErrorCategory.TRANSIENT, "x", retryable=False).is_retryable


def test_the_four_categories_are_exactly_the_specs() -> None:
    assert {c.value for c in ErrorCategory} == {"transient", "validation", "permission", "business"}


def test_inconsistent_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="error_category"):
        ToolResult("x", is_error=True)
    with pytest.raises(ValueError, match="only an error"):
        ToolResult("x", is_retryable=True)
    with pytest.raises(ValueError, match="only an error"):
        ToolResult("x", error_category=ErrorCategory.BUSINESS)
