from __future__ import annotations

import pytest

from agent_container.core import Budget, BudgetAlreadyStarted, ErrorCategory, fence_tool_output
from agent_container.core.fence import MAX_TOOL_OUTPUT_CHARS


def test_a_fresh_budget_allows_a_call() -> None:
    assert Budget().check() is None


def test_each_limit_is_reached_not_passed() -> None:
    now = [0.0]
    b = Budget(max_turns=2, max_tokens=50, max_seconds=10, clock=lambda: now[0])
    b.start()
    b.record_turn(20)
    assert b.check() is None
    now[0] = 10.0
    exceeded = b.check()
    assert exceeded is not None
    assert exceeded.limit == "wall_clock"
    assert "10 of 10 seconds" in exceeded.message


def test_a_budget_cannot_be_started_twice() -> None:
    b = Budget()
    b.start()
    with pytest.raises(BudgetAlreadyStarted):
        b.start()


def test_negative_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        Budget(max_tokens=-1)


def test_fence_marks_errors_and_retryability() -> None:
    text = fence_tool_output(
        tool="t",
        tool_use_id="i",
        content="boom",
        nonce="n1",
        category=ErrorCategory.TRANSIENT,
        retryable=True,
    )
    assert 'status="error:transient" retryable="true"' in text
    assert text.startswith("<tool_result tool=")
    assert text.endswith("</tool_result fence=n1>")


def test_fence_removes_the_nonce_and_neutralises_header_quotes() -> None:
    text = fence_tool_output(tool='a"b', tool_use_id="i\nd", content="has n1 inside", nonce="n1")
    assert "has  inside" in text
    assert 'tool="a\'b"' in text
    assert 'id="i d"' in text


def test_fence_cuts_long_output() -> None:
    text = fence_tool_output(
        tool="t", tool_use_id="i", content="x" * (MAX_TOOL_OUTPUT_CHARS + 1), nonce="n"
    )
    assert "truncated by the library" in text
