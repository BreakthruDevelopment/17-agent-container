"""The interceptor chain (Chain of Responsibility) and the per-turn output cap.

Interceptors run around each tool call. `before` runs in order and may answer the call itself by
returning a `ToolResult`: the chain stops there and the tool never runs. That is what lets a policy
violation be blocked in code instead of asked of the model in the prompt. `after` runs in reverse
order and may replace the result.

Exactly which hooks run, because it is easy to assume otherwise:

- `before` of each interceptor in order, until one returns a result.
- The tool runs only if every `before` returned `None`.
- `after` runs, in reverse, for each interceptor whose `before` returned `None`. The interceptor
  that short-circuited does not get `after` for its own answer, and the ones after it never ran.
- An exception from any hook or from the tool propagates and no further hook runs.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel

from agent_container.core.decorators import Decorator
from agent_container.core.fence import MAX_TOOL_OUTPUT_CHARS
from agent_container.core.tool import Tool, ToolContext, ToolResult


class Interceptor:
    """Override either hook; the defaults do nothing."""

    async def before(self, tool: str, args: BaseModel, ctx: ToolContext) -> ToolResult | None:
        """Return a result to answer the call without running the tool, or `None` to go on."""
        return None

    async def after(
        self, tool: str, args: BaseModel, ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        """Return the result to pass outward: `result` unchanged, or a replacement."""
        return result


class InterceptorChain:
    def __init__(self, interceptors: Sequence[Interceptor]) -> None:
        self._interceptors = tuple(interceptors)

    async def run(
        self,
        tool: str,
        args: BaseModel,
        ctx: ToolContext,
        call: Callable[[], Awaitable[ToolResult]],
    ) -> ToolResult:
        entered: list[Interceptor] = []
        result: ToolResult | None = None
        for interceptor in self._interceptors:
            result = await interceptor.before(tool, args, ctx)
            if result is not None:
                break
            entered.append(interceptor)
        if result is None:
            result = await call()
        for interceptor in reversed(entered):
            result = await interceptor.after(tool, args, ctx, result)
        return result


class Intercepted[A: BaseModel](Decorator[A]):
    """Put a tool behind an `InterceptorChain`. Share one chain between tools to share its state."""

    def __init__(self, inner: Tool[A], chain: InterceptorChain) -> None:
        super().__init__(inner)
        self._chain = chain

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        return await self._chain.run(self.name, args, ctx, lambda: self._inner.run(args, ctx))


TURN_CAP_NOTICE = "\n[output truncated by the library: this turn's tool output limit was reached]"
TURN_CAP_OMITTED = "[output omitted by the library: this turn's tool output limit was reached]"
DEFAULT_MAX_TURN_CHARS = 2 * MAX_TOOL_OUTPUT_CHARS


class TurnOutputCap(Interceptor):
    """Bound the total tool output a single model turn can inject.

    `MAX_TOOL_OUTPUT_CHARS` bounds one result, so a turn calling n tools could still put n times
    that in front of the model. Share one `TurnOutputCap` across every tool's chain: results are
    charged to the (run, turn) in `ctx`, the one that crosses the limit is cut and says so, and
    later ones in the same turn are replaced by a fixed notice.

    The bound is `max_chars` of tool content per turn plus one fixed notice per result that was
    cut or omitted; the notices are library text, and the per-call fence cap still applies on
    top. Charging is synchronous (no `await` between reading and updating the total), so
    concurrent calls cannot both be admitted on the same remaining room. Turns of a run are
    tracked in an LRU of `max_runs` runs; a run evicted by more than that many concurrent runs
    starts its turn's count again.
    """

    def __init__(self, max_chars: int = DEFAULT_MAX_TURN_CHARS, *, max_runs: int = 4096) -> None:
        if max_chars < 1 or max_runs < 1:
            raise ValueError("max_chars and max_runs must be at least 1")
        self.max_chars = max_chars
        self._max_runs = max_runs
        self._used: OrderedDict[str, tuple[int, int]] = OrderedDict()  # run_id -> (turn, chars)

    def used(self, run_id: str, turn: int) -> int:
        entry = self._used.get(run_id)
        return entry[1] if entry is not None and entry[0] == turn else 0

    async def after(
        self, tool: str, args: BaseModel, ctx: ToolContext, result: ToolResult
    ) -> ToolResult:
        used = self.used(ctx.run_id, ctx.turn)
        room = self.max_chars - used
        content = result.content
        if len(content) <= room:
            charged = len(content)
        elif room > 0:
            content = content[:room] + TURN_CAP_NOTICE
            charged = room
        else:
            content = TURN_CAP_OMITTED
            charged = 0
        self._used[ctx.run_id] = (ctx.turn, used + charged)
        self._used.move_to_end(ctx.run_id)
        while len(self._used) > self._max_runs:
            self._used.popitem(last=False)
        if content == result.content:
            return result
        return ToolResult(
            content,
            is_error=result.is_error,
            error_category=result.error_category,
            is_retryable=result.is_retryable,
            staged=result.staged,
        )
