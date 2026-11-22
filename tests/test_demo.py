"""Demo agent: recorded three-turn run, Retrying around a transient lookup failure."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_container.core import (
    EndReason,
    ErrorCategory,
    RateLimitedProvider,
    ReplayProvider,
    ToolContext,
)
from agent_container.core.bedrock import BedrockProvider
from agent_container.demo import (
    DEFAULT_MESSAGE,
    DemoOptions,
    LookupArgs,
    LookupTool,
    execute_demo,
    load_demo_replies,
    lookup_stack,
    make_provider,
)


async def test_lookup_retries_a_transient_error_then_reports_it() -> None:
    stacked = lookup_stack()
    lookup = stacked.wrapped
    assert isinstance(lookup, LookupTool)
    ctx = ToolContext("r", 1, "c")
    args = LookupArgs(query="status")
    first = await stacked.run(args, ctx)
    assert first.is_error
    assert first.error_category is ErrorCategory.TRANSIENT
    assert first.is_retryable
    assert lookup.attempts == LookupTool._FAILURES
    second = await stacked.run(args, ctx)
    assert not second.is_error
    assert "status: ok" in second.content
    assert lookup.attempts == LookupTool._FAILURES + 1


async def test_the_committed_fixture_is_consumed_exactly() -> None:
    replies = load_demo_replies()
    assert len(replies) == 3
    assert replies[0].stop_reason == "tool_use"
    assert replies[-1].stop_reason == "end_turn"
    provider = ReplayProvider(replies)
    result, reasons = await execute_demo(DemoOptions(), provider=provider, log_stream=None)
    assert provider.remaining == 0
    assert result.end_reason is EndReason.END_TURN
    assert result.stop_reason == "end_turn"
    assert result.turns_used == 3
    assert result.final_text == "Status is ok."
    assert reasons == ("tool_use", "tool_use", "end_turn")
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool == "lookup"
    assert result.tool_calls[0].is_error
    assert result.tool_calls[0].error_category is ErrorCategory.TRANSIENT
    assert not result.tool_calls[1].is_error


async def test_replay_provider_is_the_default() -> None:
    provider = make_provider(live=False, model="ignored", region="ignored")
    assert isinstance(provider, ReplayProvider)
    assert provider.remaining == 3


def test_live_provider_is_bedrock_behind_the_limiter() -> None:
    provider = make_provider(live=True, model="test.model", region="us-east-2")
    assert isinstance(provider, RateLimitedProvider)
    inner = provider._inner
    assert isinstance(inner, BedrockProvider)


async def test_default_message_runs_to_end_turn() -> None:
    assert 0 < len(DEFAULT_MESSAGE) <= 8000
    result, _ = await execute_demo(DemoOptions(), log_stream=None)
    assert result.end_reason is EndReason.END_TURN


def test_lookup_args_reject_an_empty_query() -> None:
    with pytest.raises(ValidationError):
        LookupArgs(query="")
