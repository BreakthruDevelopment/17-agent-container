"""Demo agent: recorded three-turn run, Retrying around a transient lookup failure."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

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
    LOOKUP_FAILURES,
    WRITE_PERMISSION,
    DemoOptions,
    LookupArgs,
    LookupTool,
    RecordStatusTool,
    demo_manifest_text,
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
    assert lookup.attempts == LOOKUP_FAILURES
    second = await stacked.run(args, ctx)
    assert not second.is_error
    assert "status: ok" in second.content
    assert lookup.attempts == LOOKUP_FAILURES + 1


async def test_the_committed_fixture_is_consumed_exactly() -> None:
    replies = load_demo_replies()
    assert len(replies) == 4
    assert replies[0].stop_reason == "tool_use"
    assert replies[-1].stop_reason == "end_turn"
    provider = ReplayProvider(replies)
    outcome = await execute_demo(DemoOptions(), provider=provider, log_stream=None)
    result, reasons = outcome.result, outcome.stop_reasons
    assert provider.remaining == 0
    assert result.end_reason is EndReason.END_TURN
    assert result.stop_reason == "end_turn"
    assert result.turns_used == 4
    assert result.final_text == "Status is ok; recording it is staged for approval."
    assert reasons == ("tool_use", "tool_use", "tool_use", "end_turn")
    assert len(result.tool_calls) == 3
    assert result.tool_calls[2].tool == "record_status"
    assert not result.tool_calls[2].is_error
    assert len(result.staged_changes) == 1
    assert len(outcome.approved) == 1
    assert outcome.succeeded
    assert result.tool_calls[0].tool == "lookup"
    assert result.tool_calls[0].is_error
    assert result.tool_calls[0].error_category is ErrorCategory.TRANSIENT
    assert not result.tool_calls[1].is_error


async def test_replay_provider_is_the_default() -> None:
    provider = make_provider(live=False, model="ignored", region="ignored")
    assert isinstance(provider, ReplayProvider)
    assert provider.remaining == 4


def test_live_provider_is_bedrock_behind_the_limiter() -> None:
    provider = make_provider(live=True, model="test.model", region="us-east-2")
    assert isinstance(provider, RateLimitedProvider)
    inner = provider._inner
    assert isinstance(inner, BedrockProvider)


async def test_default_message_runs_to_end_turn() -> None:
    assert 0 < len(DEFAULT_MESSAGE) <= 8000
    outcome = await execute_demo(DemoOptions(), log_stream=None)
    assert outcome.result.end_reason is EndReason.END_TURN


def test_lookup_args_reject_an_empty_query() -> None:
    with pytest.raises(ValidationError):
        LookupArgs(query="")


async def test_the_write_runs_only_after_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[RecordStatusTool] = []
    original = RecordStatusTool.__init__

    def spy(self: RecordStatusTool) -> None:
        original(self)
        ran.append(self)

    monkeypatch.setattr(RecordStatusTool, "__init__", spy)
    provider = ReplayProvider(load_demo_replies())
    outcome = await execute_demo(DemoOptions(), provider=provider, log_stream=None)
    (tool,) = ran
    assert tool.recorded == ["ok"]  # applied once, by the approval step, not by the loop
    assert len(outcome.approved) == 1


async def test_without_the_permission_the_gate_refuses_and_nothing_is_staged() -> None:
    provider = ReplayProvider(load_demo_replies())
    outcome = await execute_demo(DemoOptions(), provider=provider, log_stream=None, permissions=[])
    refused = outcome.result.tool_calls[2]
    assert refused.tool == "record_status"
    assert refused.is_error
    assert refused.error_category is ErrorCategory.PERMISSION
    assert outcome.result.staged_changes == ()
    assert outcome.approved == ()


def test_the_manifest_grants_the_write_permission() -> None:
    assert WRITE_PERMISSION in demo_manifest_text()


async def test_a_failed_run_stages_the_write_but_never_applies_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that ended on a limit is not an approval.

    The apply loop used to run unconditionally, before anything looked at `end_reason`. With a
    truncated run the demo printed `approved: record_status: ... {"status":"ok"}` and then
    `demo failed: the run ended with end_reason=truncated`, exit 1 -- having already written.
    """
    ran: list[RecordStatusTool] = []
    original = RecordStatusTool.__init__

    def spy(self: RecordStatusTool) -> None:
        original(self)
        ran.append(self)

    monkeypatch.setattr(RecordStatusTool, "__init__", spy)
    replies = load_demo_replies()
    truncated = [*replies[:-1], replace(replies[-1], stop_reason="max_tokens")]
    outcome = await execute_demo(DemoOptions(), provider=ReplayProvider(truncated), log_stream=None)

    assert not outcome.succeeded
    assert outcome.result.end_reason is EndReason.TRUNCATED
    # The gate still staged it -- the write was authorised, it simply must not be applied.
    assert len(outcome.result.staged_changes) == 1
    assert outcome.approved == ()
    (tool,) = ran
    assert tool.recorded == [], "a failed run applied its staged write"


def test_the_apply_gate_and_succeeded_share_one_predicate() -> None:
    """They were separate once, and the apply ran unconditionally. Drift reintroduces the bug.

    Parsed, not grepped. The first version of this test searched
    `source.split("async def execute_demo")[1]` for the helper's name -- which was everything to
    end of file, so it was not scoped to the function at all, and a comment line reading
    `# gate: _run_succeeded(result)` satisfied it with the guard reverted. It detected nothing
    the behavioural test above does not.
    """
    tree = ast.parse((Path(__file__).resolve().parents[1] / "agent_container/demo.py").read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_demo"
    )
    called = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_run_succeeded" in called, "the apply gate no longer calls the shared predicate"
    for node in ast.walk(function):
        if isinstance(node, ast.Compare):
            rendered = [ast.unparse(node.left), *(ast.unparse(c) for c in node.comparators)]
            assert not any("end_reason" in part for part in rendered), (
                f"execute_demo compares end_reason directly ({' '.join(rendered)}) "
                "instead of using _run_succeeded"
            )
