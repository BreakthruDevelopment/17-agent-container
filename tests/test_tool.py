from __future__ import annotations

from dataclasses import replace
from typing import cast

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


def _wire(value: str) -> ErrorCategory:
    """A plain `str` where the annotation says `ErrorCategory`, which is the whole point of #11.

    The cast is the test's subject, not a convenience: it is exactly what an untyped caller --
    JSON, a consumer app's helper, a `# type: ignore` -- hands this library at runtime.
    """
    return cast(ErrorCategory, value)


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


def test_the_categories_are_exactly_the_specs() -> None:
    """The spec named four; `internal` was added on 2026-11-20 (review of PR #9).

    An unhandled exception in a tool is not `business` ("a rule said no") -- app 18 would explain a
    policy that does not exist. Adding the member now is free; adding it after 17.3 and 17.4 ship
    `match` statements over this enum is a break across four repos.
    """
    assert {c.value for c in ErrorCategory} == {
        "transient",
        "validation",
        "permission",
        "business",
        "internal",
    }


def test_inconsistent_results_are_rejected() -> None:
    with pytest.raises(ValueError, match="error_category"):
        ToolResult("x", is_error=True)
    with pytest.raises(ValueError, match="only an error"):
        ToolResult("x", is_retryable=True)
    with pytest.raises(ValueError, match="only an error"):
        ToolResult("x", error_category=ErrorCategory.BUSINESS)


# -- the wire value is coerced to a member (#11) --------------------------------------------------
#
# `ErrorCategory` is a `StrEnum`, so `"transient" == ErrorCategory.TRANSIENT` while
# `"transient" is ErrorCategory.TRANSIENT` is False. The annotations say `ErrorCategory`, but
# nothing enforces that at runtime, and a caller holding the wire value -- a tool reading a
# category out of JSON, a consumer app's own helper -- used to get a result whose
# `error_category` stayed a plain `str`. That made the retry flag wrong here and every
# downstream `is` and `match` wrong in the apps that import this library. The category is
# normalised at construction instead, so no site has to remember.


def test_the_wire_value_still_defaults_to_retryable() -> None:
    """`is_retryable` was False for `"transient"`: the old default compared with `is`.

    This is the defect #11 was filed for. It fails closed -- a transient error is reported
    instead of retried -- but app 20's throttling eval depends on the retry actually happening.
    """
    r = ToolResult.error(_wire("transient"), "throttled, try again")
    assert r.is_retryable is True
    assert r.error_category is ErrorCategory.TRANSIENT


def test_the_wire_value_is_a_member_however_the_result_was_built() -> None:
    """Not just through `error()`: direct construction and `replace` go through the same gate.

    `Cached` and `Redacting` rebuild results with `dataclasses.replace`, so a result that
    entered a decorator stack carrying a plain string used to leave it carrying one.
    """
    direct = ToolResult("nope", is_error=True, error_category=_wire("validation"))
    assert direct.error_category is ErrorCategory.VALIDATION

    copied = replace(direct, content="still nope")
    assert copied.error_category is ErrorCategory.VALIDATION


def test_an_unknown_category_is_a_value_error_not_a_default() -> None:
    """The library's standing rule, the same one `Policy(policy)` and `StopReason` follow.

    A raise inside a tool is turned by the loop into a non-retryable `internal` result with a
    fixed message, so this cannot reach the model or a client as text.
    """
    with pytest.raises(ValueError):
        ToolResult.error(_wire("nonsense"), "x")
    with pytest.raises(ValueError):
        ToolResult("x", is_error=True, error_category=_wire("nonsense"))


def test_a_category_that_is_not_even_a_string_is_refused() -> None:
    """Nothing honest to coerce, so it is refused rather than stored for a later reader.

    The message names the field and not the value: this one is reachable from a tool.
    """
    with pytest.raises(ValueError, match="error_category"):
        ToolResult("x", is_error=True, error_category=cast(ErrorCategory, 3))


def test_coercion_does_not_loosen_the_consistency_rules() -> None:
    """A wire value must not become a way past `__post_init__`'s other checks."""
    with pytest.raises(ValueError, match="only an error"):
        ToolResult("x", error_category=_wire("business"))
