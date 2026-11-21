"""`Budget.child()`: subagents draw on one parent allowance."""

from __future__ import annotations

import asyncio

import pytest

from agent_container.core import AgentLoop, Budget, BudgetAlreadyStarted, EndReason, ModelResponse
from tests.support import scripted, tool_call, use_tool


def test_a_child_spends_from_the_parent_and_keeps_its_own_count() -> None:
    parent = Budget(max_turns=10, max_tokens=1000)
    a, b = parent.child(), parent.child()
    a.record_turn(100)
    b.record_turn(50)
    assert (a.turns_used, a.tokens_used) == (1, 100)  # a sibling's spend is not a's
    assert (b.turns_used, b.tokens_used) == (1, 50)
    assert (parent.turns_used, parent.tokens_used) == (2, 150)


def test_the_run_stops_at_the_parents_limit_not_the_childs() -> None:
    parent = Budget(max_turns=3, max_tokens=10_000)
    a, b = parent.child(max_turns=10), parent.child(max_turns=10)
    a.record_turn(1)
    a.record_turn(1)
    assert a.check() is None and b.check() is None
    b.record_turn(1)
    for child in (a, b):
        exceeded = child.check()
        assert exceeded is not None and exceeded.limit == "turns"
        assert (exceeded.used, exceeded.maximum) == (3, 3)  # the parent's numbers


def test_a_childs_own_tighter_limit_applies_without_touching_the_parent() -> None:
    parent = Budget(max_turns=10)
    a = parent.child(max_turns=1)
    a.record_turn(1)
    assert a.check() is not None
    assert parent.check() is None and parent.child().check() is None


def test_a_sibling_cannot_starve_another_childs_own_allowance() -> None:
    parent = Budget(max_turns=10, max_tokens=1_000_000)
    greedy, other = parent.child(max_turns=3), parent.child(max_turns=3)
    for _ in range(3):
        greedy.record_turn(1)
    assert greedy.check() is not None
    assert other.check() is None  # its own count is untouched


def test_wall_clock_is_shared_and_the_parent_starts_with_the_first_child() -> None:
    now = [0.0]
    parent = Budget(max_seconds=10, clock=lambda: now[0])
    a = parent.child()
    assert not parent.started
    now[0] = 4.0
    a.start()
    started: bool = parent.started
    assert started and parent.elapsed == 0.0
    now[0] = 9.0
    b = parent.child()
    assert b.remaining_seconds == 5.0  # the parent's clock (started at 4), not a fresh 10
    b.start()
    now[0] = 14.0
    exceeded = b.check()
    assert exceeded is not None and exceeded.limit == "wall_clock"
    # reported honestly: elapsed against the ceiling that actually bound it
    assert exceeded.used >= exceeded.maximum


def test_a_child_is_single_use_and_a_started_parent_is_fine() -> None:
    parent = Budget()
    parent.start()
    child = parent.child()
    child.start()
    with pytest.raises(BudgetAlreadyStarted):
        child.start()
    parent.child().start()  # a sibling starts independently


def test_grandchildren_charge_every_ancestor() -> None:
    root = Budget(max_turns=2)
    leaf = root.child().child()
    leaf.record_turn(5)
    leaf.record_turn(5)
    assert root.turns_used == 2 and leaf.check() is not None


async def _drain(parent: Budget, leaves: list[Budget]) -> int:
    async def agent(child: Budget) -> None:
        while child.check() is None:
            await asyncio.sleep(0)  # the "model call"
            child.record_turn(1)

    await asyncio.gather(*(agent(leaf) for leaf in leaves))
    return parent.turns_used


async def test_overshoot_is_bounded_by_the_concurrent_leaves_in_any_shape() -> None:
    """The bound is leaves - 1, at any depth -- not the number of direct children."""
    limit = 5

    def flat(n: int) -> tuple[Budget, list[Budget]]:
        p = Budget(max_turns=limit, max_tokens=10**9)
        return p, [p.child() for _ in range(n)]

    def tree(n: int) -> tuple[Budget, list[Budget]]:
        p = Budget(max_turns=limit, max_tokens=10**9)
        mid = p.child()
        return p, [mid.child() for _ in range(n)]

    for build, n in ((flat, 4), (tree, 4), (tree, 8)):
        parent, leaves = build(n)
        used = await _drain(parent, leaves)
        assert limit <= used <= limit + n - 1, (build.__name__, n, used)
    # the case that made "one per child" false: 8 grandchildren under one child
    parent, leaves = tree(8)
    assert await _drain(parent, leaves) == limit + 8 - 1
    # and the limit holds exactly when the leaves run one at a time
    parent, leaves = tree(8)
    for leaf in leaves:
        await _drain(parent, [leaf])
    assert parent.turns_used == limit


async def test_a_loop_runs_under_a_child_and_reports_the_parents_limit() -> None:
    parent = Budget(max_turns=1)
    provider = scripted(
        use_tool(tool_call()),
        ModelResponse.of("never", stop_reason="end_turn"),
    )
    from tests.support import EchoTool

    result = await AgentLoop(provider, [EchoTool()], system_prompt="s").run("go", parent.child())
    assert result.end_reason is EndReason.BUDGET_EXCEEDED
    assert result.budget_exceeded is not None and result.budget_exceeded.maximum == 1
    assert parent.turns_used == 1
