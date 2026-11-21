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
    """

    def __init__(
        self,
        *,
        max_turns: int = 10,
        max_tokens: int = 100_000,
        max_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_turns < 0 or max_tokens < 0 or max_seconds < 0:
            raise ValueError("budget limits cannot be negative")
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self._clock = clock
        self._started_at: float | None = None
        self.turns_used = 0
        self.tokens_used = 0

    @property
    def started(self) -> bool:
        return self._started_at is not None

    def start(self) -> None:
        """Start the wall clock. Called by the loop once, at the beginning of a run."""
        if self._started_at is not None:
            raise BudgetAlreadyStarted("this budget has already been used; create one per run")
        self._started_at = self._clock()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._started_at is None else self._clock() - self._started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    def record_turn(self, tokens: int) -> None:
        self.turns_used += 1
        self.tokens_used += tokens

    def check(self) -> BudgetExceeded | None:
        """The first limit that has been reached, or `None` if another model call is allowed."""
        if self.turns_used >= self.max_turns:
            return BudgetExceeded("turns", self.turns_used, self.max_turns)
        if self.tokens_used >= self.max_tokens:
            return BudgetExceeded("tokens", self.tokens_used, self.max_tokens)
        if self.elapsed >= self.max_seconds:
            return BudgetExceeded("wall_clock", self.elapsed, self.max_seconds)
        return None
