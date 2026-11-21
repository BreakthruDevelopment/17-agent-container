"""The run budget. An ordinary object, created once per run and passed in. Not a singleton."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agent_container.core.errors import BudgetAlreadyStarted

Limit = Literal["turns", "tokens", "wall_clock"]


@dataclass(frozen=True, slots=True)
class BudgetExceeded:
    """A limit was reached. A value the loop reports in its result, not an exception."""

    limit: Limit
    used: float
    maximum: float

    @property
    def message(self) -> str:
        unit = {"turns": "turns", "tokens": "tokens", "wall_clock": "seconds"}[self.limit]
        return f"{self.limit} budget exceeded: {self.used:g} of {self.maximum:g} {unit} used"


class Budget:
    """Maximum turns, tokens and wall-clock seconds for one run.

    Tokens are input plus output summed over every model call. Input tokens recur each turn because
    the whole conversation is resent, which is what the model bills, so the sum tracks cost and not
    just length. Each limit is "no more once reached": a call is not started when a limit has been
    hit.

    Create one per run and hand it to `AgentLoop.run`. Passing a used budget to a second run raises
    `BudgetAlreadyStarted` instead of quietly sharing one allowance between runs. `clock` is
    injectable so wall-clock behaviour is testable without sleeping.

    `child()` makes a budget that draws on this one, for a subagent: see there.
    """

    def __init__(
        self,
        *,
        max_turns: int = 10,
        max_tokens: int = 100_000,
        max_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        _parent: Budget | None = None,
    ) -> None:
        if max_turns < 0 or max_tokens < 0 or max_seconds < 0:
            raise ValueError("budget limits cannot be negative")
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._own_max_seconds = max_seconds
        self._clock = clock
        self._parent = _parent
        self._started_at: float | None = None
        self.turns_used = 0
        self.tokens_used = 0

    @property
    def parent(self) -> Budget | None:
        return self._parent

    @property
    def max_seconds(self) -> float:
        """The wall-clock ceiling for this budget, counted from its own start.

        For a child it is the lesser of its own limit and what its parent has left, so the loop's
        "elapsed of max" report stays true when the parent's clock is what ran out.
        """
        if self._parent is None:
            return self._own_max_seconds
        return min(self._own_max_seconds, self.elapsed + self._parent.remaining_seconds)

    def child(
        self,
        *,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        max_seconds: float | None = None,
    ) -> Budget:
        """A budget for one subagent that draws on this one.

        The child keeps its own counters and limits, so what a sibling spends never counts against
        it, and every turn and token it records is also recorded here, so the run as a whole stops
        at this budget's limits. Unset limits mean "no tighter than the parent". Give each
        subagent its own child; a child is single-use like any budget.

        Starting a child starts this budget's wall clock too, if nothing has: the allowance then
        runs from the first subagent. Read `check()` on the child, which also reports an exhausted
        ancestor, and the loop already does.

        Limit, stated exactly: a limit is checked before a model call and spent after it. Children
        that run in parallel can each pass the check with one turn left, so the parent may end
        past its limit by at most one in-flight call per child. Nothing here reserves in advance.
        """
        return Budget(
            max_turns=self.max_turns if max_turns is None else max_turns,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            max_seconds=self._own_max_seconds if max_seconds is None else max_seconds,
            clock=self._clock,
            _parent=self,
        )

    @property
    def started(self) -> bool:
        return self._started_at is not None

    def start(self) -> None:
        """Start the wall clock. Called by the loop once, at the beginning of a run."""
        if self._started_at is not None:
            raise BudgetAlreadyStarted("this budget has already been used; create one per run")
        self._started_at = self._clock()
        if self._parent is not None and not self._parent.started:
            self._parent.start()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started_at is None else self._clock() - self._started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    def record_turn(self, tokens: int) -> None:
        self.turns_used += 1
        self.tokens_used += tokens
        if self._parent is not None:
            self._parent.record_turn(tokens)

    def check(self) -> BudgetExceeded | None:
        """The first limit that has been reached, or `None` if another model call is allowed."""
        if self.turns_used >= self.max_turns:
            return BudgetExceeded("turns", self.turns_used, self.max_turns)
        if self.tokens_used >= self.max_tokens:
            return BudgetExceeded("tokens", self.tokens_used, self.max_tokens)
        if self.elapsed >= self.max_seconds:
            return BudgetExceeded("wall_clock", self.elapsed, self.max_seconds)
        return None if self._parent is None else self._parent.check()
