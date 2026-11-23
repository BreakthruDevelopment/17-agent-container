"""Decorators, the gate, staged changes, the interceptor chain and the per-turn cap (17.4)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, JsonValue
from pydantic_core import PydanticSerializationError

from agent_container.core import (
    AgentLoop,
    AnyTool,
    Budget,
    Cached,
    Effect,
    ErrorCategory,
    Identity,
    Intercepted,
    Interceptor,
    InterceptorChain,
    Metered,
    Metrics,
    PermissionGate,
    Policy,
    Redacting,
    Retrying,
    StagedChange,
    StagedChangeError,
    Tool,
    ToolContext,
    ToolResult,
    TurnOutputCap,
    has_write_gate,
)
from agent_container.core.decorators import (
    MAX_DESCRIPTION_CHARS,
    _RedactedStaged,
    limit_description,
)
from agent_container.core.interceptors import TURN_CAP_NOTICE, TURN_CAP_OMITTED
from agent_container.testing import ToolContract
from tests.support import (
    DeleteTool,
    EchoArgs,
    EchoTool,
    scripted,
    staged_pay_behind_a_clamp,
    tool_call,
    use_tool,
)

from agent_container.core import ModelResponse  # isort: skip


def ctx(
    subject: str = "alice",
    permissions: frozenset[str] = frozenset(),
    *,
    run_id: str = "r",
    turn: int = 1,
) -> ToolContext:
    return ToolContext(run_id, turn, "c", Identity(subject, permissions))


class CountingWrite(DeleteTool):
    """A write tool that records each real write."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        self.writes.append(args.text)
        return ToolResult.ok(f"wrote {args.text}")


# -- Cached: the point of the slice --------------------------------------------------------------


async def test_two_callers_with_different_permissions_do_not_share_a_cached_result() -> None:
    """`Cached(PermissionGate(write))` is valid. The gate decides per caller, so the cache must
    too: a permitted caller's staged-or-allowed answer must never reach one who lacks permission."""
    tool = EchoTool()
    gated_read = Cached(PermissionGate(tool, Policy.READ, requires="read:echo"))
    args = EchoArgs(text="secret")

    allowed = await gated_read.run(args, ctx("alice", frozenset({"read:echo"})))
    assert not allowed.is_error and allowed.content == "echo: secret"
    # Same subject, same arguments, no permission: must be denied, not served alice's result.
    denied = await gated_read.run(args, ctx("alice", frozenset()))
    assert denied.is_error and denied.error_category is ErrorCategory.PERMISSION
    # Different subject, same permission: a separate entry, so the tool really runs again.
    await gated_read.run(args, ctx("bob", frozenset({"read:echo"})))
    assert len(tool.calls) == 2
    assert len(gated_read) == 2


async def test_the_same_caller_is_served_from_the_cache() -> None:
    tool = EchoTool()
    cached = Cached(tool)
    c = ctx("alice", frozenset({"a", "b"}))
    first = await cached.run(EchoArgs(text="x"), c)
    again = await cached.run(EchoArgs(text="x"), ctx("alice", frozenset({"b", "a"})))
    assert again is first
    assert len(tool.calls) == 1


async def test_a_cached_denial_is_not_remembered() -> None:
    gated = Cached(PermissionGate(EchoTool(), Policy.READ, requires="p"))
    a = EchoArgs(text="x")
    assert (await gated.run(a, ctx("alice"))).is_error
    ok = await gated.run(a, ctx("alice", frozenset({"p"})))
    assert not ok.is_error


async def test_cached_does_not_store_errors_or_staged_changes() -> None:
    calls = 0

    def reply(args: EchoArgs) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult.error(ErrorCategory.TRANSIENT, "down")

    failing = Cached(EchoTool(reply))
    await failing.run(EchoArgs(text="x"), ctx())
    await failing.run(EchoArgs(text="x"), ctx())
    assert calls == 2

    write = CountingWrite()
    staged = Cached(PermissionGate(write))
    one = await staged.run(EchoArgs(text="x"), ctx())
    two = await staged.run(EchoArgs(text="x"), ctx())
    assert one.staged is not None and two.staged is not None
    assert one.staged is not two.staged  # a staged change is single-use, never shared
    assert len(staged) == 0


async def test_cached_honours_ttl_and_the_size_bound() -> None:
    now = [0.0]
    tool = EchoTool()
    cached = Cached(tool, ttl=10, clock=lambda: now[0])
    await cached.run(EchoArgs(text="x"), ctx())
    now[0] = 9.0
    await cached.run(EchoArgs(text="x"), ctx())
    assert len(tool.calls) == 1
    now[0] = 10.0
    await cached.run(EchoArgs(text="x"), ctx())
    assert len(tool.calls) == 2

    small = Cached(EchoTool(), max_entries=2)
    for i in range(5):
        await small.run(EchoArgs(text=str(i)), ctx())
    assert len(small) == 2


async def test_a_cancelled_call_leaves_nothing_in_the_cache() -> None:
    started = asyncio.Event()

    class Hangs(EchoTool):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            started.set()
            await asyncio.sleep(3600)
            return ToolResult.ok("never")

    cached = Cached(Hangs())
    task = asyncio.create_task(cached.run(EchoArgs(text="x"), ctx()))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(cached) == 0


# -- PermissionGate and staged changes -----------------------------------------------------------


def test_a_gate_is_found_anywhere_in_the_stack() -> None:
    # Built step by step: mypy infers a nested generic call badly when the outer expected type is
    # the `AnyTool` protocol, so each layer is bound to a name first.
    write = CountingWrite()
    gate = PermissionGate(write)
    cached = Cached(gate)
    assert has_write_gate(cached)
    metered = Metered(cached, Metrics())
    assert has_write_gate(metered)
    ungated = Metered(Metered(write, Metrics()), Metrics())
    assert not has_write_gate(ungated)


def test_the_loop_accepts_a_gated_write_under_cached_and_rejects_an_ungated_one() -> None:
    def build(tool: AnyTool) -> AgentLoop:
        return AgentLoop(scripted(), [tool], system_prompt="s")

    gated = Cached(PermissionGate(CountingWrite()))
    build(gated)
    bare = Metered(Metered(CountingWrite(), Metrics()), Metrics())
    with pytest.raises(ValueError, match="no permission gate"):
        build(bare)


async def test_a_staged_write_does_nothing_until_applied() -> None:
    write = CountingWrite()
    result = await PermissionGate(write).run(EchoArgs(text="x"), ctx())
    assert write.writes == []
    assert not result.is_error and result.staged is not None
    assert "x" not in result.content  # what the model reads is a fixed string
    assert "x" in result.staged.description and "delete" in result.staged.description
    await result.staged.apply()
    assert write.writes == ["x"]


async def test_a_staged_change_applies_once_even_when_it_failed() -> None:
    class Flaky(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            self.writes.append(args.text)
            return ToolResult.error(ErrorCategory.TRANSIENT, "down")

    write = Flaky()
    staged = (await PermissionGate(write).run(EchoArgs(text="x"), ctx())).staged
    assert staged is not None
    with pytest.raises(StagedChangeError, match="failed when applied"):
        await staged.apply()
    with pytest.raises(StagedChangeError, match="already"):
        await staged.apply()
    assert write.writes == ["x"]


async def test_a_cancelled_apply_cannot_be_repeated() -> None:
    started = asyncio.Event()

    class Hangs(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            self.writes.append(args.text)
            started.set()
            await asyncio.sleep(3600)
            return ToolResult.ok("never")

    write = Hangs()
    staged = (await PermissionGate(write).run(EchoArgs(text="x"), ctx())).staged
    assert staged is not None
    task = asyncio.create_task(staged.apply())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # bounded, so a regression fails here instead of hanging on a second, hung write
    with pytest.raises(StagedChangeError, match="already"):
        await asyncio.wait_for(staged.apply(), timeout=5)
    assert write.writes == ["x"]


async def test_concurrent_applies_run_the_write_once() -> None:
    class Yields(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            await asyncio.sleep(0)  # lets the second apply start while the first is in flight
            return await super().run(args, ctx)

    write = Yields()
    staged = (await PermissionGate(write).run(EchoArgs(text="x"), ctx())).staged
    assert staged is not None
    outcomes = await asyncio.gather(staged.apply(), staged.apply(), return_exceptions=True)
    assert sorted(type(o).__name__ for o in outcomes) == ["NoneType", "StagedChangeError"]
    assert write.writes == ["x"]


async def test_deny_and_missing_permission_run_nothing() -> None:
    write = CountingWrite()
    for gate in (
        PermissionGate(write, Policy.DENY),
        PermissionGate(write, "staged_write", requires="write:x"),
    ):
        result = await gate.run(EchoArgs(text="x"), ctx(permissions=frozenset({"other"})))
        assert result.is_error and result.error_category is ErrorCategory.PERMISSION
        assert not result.is_retryable and result.staged is None
    assert write.writes == []


def test_policy_must_match_the_tool_and_be_known() -> None:
    with pytest.raises(ValueError, match="writes"):
        PermissionGate(CountingWrite(), Policy.READ)
    with pytest.raises(ValueError, match="reads"):
        PermissionGate(EchoTool(), Policy.STAGED_WRITE)
    with pytest.raises(ValueError):
        PermissionGate(CountingWrite(), "allow")
    # a plain string compares equal to the enum: `==`, never `is`
    assert PermissionGate(CountingWrite(), "staged_write").policy == Policy.STAGED_WRITE
    assert PermissionGate(CountingWrite(), "deny").policy == Policy.DENY


async def test_a_gated_write_stages_through_the_loop_and_is_reported() -> None:
    write = CountingWrite()
    provider = scripted(
        use_tool(tool_call("x", name="delete")),
        ModelResponse.of("done", stop_reason="end_turn"),
    )
    gated = Cached(PermissionGate(write))
    agent = AgentLoop(provider, [gated], system_prompt="s")
    result = await agent.run("go", Budget())
    assert write.writes == []
    assert len(result.staged_changes) == 1
    await result.staged_changes[0].apply()
    assert write.writes == ["x"]


# -- Retrying, Redacting, Metered ----------------------------------------------------------------


async def test_retrying_retries_only_what_the_tool_called_retryable() -> None:
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    def flaky(n: int) -> Callable[[EchoArgs], ToolResult]:
        left = [n]

        def reply(args: EchoArgs) -> ToolResult:
            if left[0] > 0:
                left[0] -= 1
                return ToolResult.error(ErrorCategory.TRANSIENT, "down")
            return ToolResult.ok("up")

        return reply

    tool = EchoTool(flaky(2))
    result = await Retrying(tool, max_attempts=3, base_delay=1.0, sleep=sleep).run(
        EchoArgs(text="x"), ctx()
    )
    assert result.content == "up" and len(tool.calls) == 3
    assert delays == [1.0, 2.0]

    gave_up = EchoTool(flaky(10))
    result = await Retrying(gave_up, max_attempts=2, sleep=sleep).run(EchoArgs(text="x"), ctx())
    assert result.is_error and len(gave_up.calls) == 2

    business = EchoTool(lambda a: ToolResult.error(ErrorCategory.BUSINESS, "no"))
    await Retrying(business, max_attempts=5, sleep=sleep).run(EchoArgs(text="x"), ctx())
    assert len(business.calls) == 1


async def test_retrying_does_not_swallow_cancellation() -> None:
    started = asyncio.Event()

    async def sleep(seconds: float) -> None:
        started.set()
        await asyncio.sleep(3600)

    tool = EchoTool(lambda a: ToolResult.error(ErrorCategory.TRANSIENT, "down"))
    task = asyncio.create_task(Retrying(tool, sleep=sleep).run(EchoArgs(text="x"), ctx()))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(tool.calls) == 1


async def test_redacting_cleans_the_result_and_a_staged_description() -> None:
    leak = EchoTool(lambda a: ToolResult.error(ErrorCategory.BUSINESS, "key sk-12345 rejected"))
    result = await Redacting(leak, [r"sk-\d+"]).run(EchoArgs(text="x"), ctx())
    assert result.content == "key [redacted] rejected" and result.is_error
    assert result.error_category is ErrorCategory.BUSINESS

    write = CountingWrite()
    staged = (
        await Redacting(PermissionGate(write), [r"sk-\d+"]).run(EchoArgs(text="sk-999"), ctx())
    ).staged
    assert staged is not None
    assert "sk-999" not in staged.description and "[redacted]" in staged.description
    await staged.apply()
    assert write.writes == ["sk-999"]  # redaction is for what is shown, not what is done


async def test_metered_counts_calls_errors_and_exceptions() -> None:
    metrics = Metrics()
    now = [0.0]

    def clock() -> float:
        now[0] += 2.0
        return now[0]

    ok = Metered(EchoTool(), metrics, clock=clock)
    await ok.run(EchoArgs(text="x"), ctx())
    bad = Metered(EchoTool(lambda a: ToolResult.error(ErrorCategory.INTERNAL, "x")), metrics)
    await bad.run(EchoArgs(text="x"), ctx())

    def boom(a: EchoArgs) -> ToolResult:
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        await Metered(EchoTool(boom), metrics).run(EchoArgs(text="x"), ctx())
    seen = metrics.snapshot()["echo"]
    assert (seen.calls, seen.errors) == (3, 2)
    assert seen.seconds >= 2.0


# -- the interceptor chain -----------------------------------------------------------------------


class Trace(Interceptor):
    def __init__(self, name: str, log: list[str], *, block: bool = False) -> None:
        self.name, self.log, self.block = name, log, block

    async def before(self, tool: str, args: object, ctx: ToolContext) -> ToolResult | None:
        self.log.append(f"before:{self.name}")
        return ToolResult.error(ErrorCategory.BUSINESS, "blocked") if self.block else None

    async def after(
        self,
        tool: str,
        args: object,
        ctx: ToolContext,
        result: ToolResult,
    ) -> ToolResult:
        self.log.append(f"after:{self.name}")
        return result


async def test_interceptors_run_in_order_and_unwind_in_reverse() -> None:
    log: list[str] = []

    def reply(args: EchoArgs) -> ToolResult:
        log.append("tool")
        return ToolResult.ok("x")

    tool = EchoTool(reply)
    chain = InterceptorChain([Trace("a", log), Trace("b", log)])
    await Intercepted(tool, chain).run(EchoArgs(text="x"), ctx())
    assert log == ["before:a", "before:b", "tool", "after:b", "after:a"]


async def test_an_interceptor_can_short_circuit_and_the_tool_never_runs() -> None:
    log: list[str] = []
    tool = EchoTool()
    chain = InterceptorChain([Trace("a", log), Trace("b", log, block=True), Trace("c", log)])
    result = await Intercepted(tool, chain).run(EchoArgs(text="x"), ctx())
    assert result.is_error and result.content == "blocked"
    assert tool.calls == []
    # `c` never ran; `b` answered and gets no `after` for its own answer; `a` unwinds.
    assert log == ["before:a", "before:b", "after:a"]


async def test_interceptors_can_block_a_policy_violation_in_code() -> None:
    class NoSecrets(Interceptor):
        async def before(self, tool: str, args: object, ctx: ToolContext) -> ToolResult | None:
            if "secret" in str(args):
                return ToolResult.error(ErrorCategory.PERMISSION, "policy: no secrets")
            return None

    tool = EchoTool()
    guarded = Intercepted(tool, InterceptorChain([NoSecrets()]))
    assert (await guarded.run(EchoArgs(text="my secret"), ctx())).is_error
    assert tool.calls == []


# -- the per-turn output cap ---------------------------------------------------------------------


def big_tool(size: int) -> EchoTool:
    return EchoTool(lambda a: ToolResult.ok("x" * size))


async def test_a_turn_cannot_inject_more_than_the_cap_across_tools() -> None:
    cap = TurnOutputCap(100)
    chain = InterceptorChain([cap])
    tools = [Intercepted(big_tool(60), chain) for _ in range(4)]
    outputs = [(await t.run(EchoArgs(text="x"), ctx(turn=1))).content for t in tools]
    assert outputs[0] == "x" * 60
    assert outputs[1] == "x" * 40 + TURN_CAP_NOTICE
    assert outputs[2] == outputs[3] == TURN_CAP_OMITTED
    content = sum(len(o.removesuffix(TURN_CAP_NOTICE)) for o in outputs[:2])
    assert content == 100


async def test_the_cap_resets_each_turn_and_is_per_run() -> None:
    cap = TurnOutputCap(50)
    tool = Intercepted(big_tool(50), InterceptorChain([cap]))
    for turn in (1, 2):
        assert (await tool.run(EchoArgs(text="x"), ctx(turn=turn))).content == "x" * 50
    # another run, same turn number, has its own allowance
    assert (await tool.run(EchoArgs(text="x"), ctx(run_id="other", turn=2))).content == "x" * 50
    assert (await tool.run(EchoArgs(text="x"), ctx(turn=2))).content == TURN_CAP_OMITTED


async def test_the_cap_keeps_the_error_status_and_the_staged_change() -> None:
    cap = TurnOutputCap(5)
    err = Intercepted(
        EchoTool(lambda a: ToolResult.error(ErrorCategory.BUSINESS, "y" * 50)),
        InterceptorChain([cap]),
    )
    result = await err.run(EchoArgs(text="x"), ctx())
    assert result.is_error and result.error_category is ErrorCategory.BUSINESS
    assert result.content.startswith("yyyyy")


async def test_the_run_total_is_bounded_end_to_end_through_the_loop() -> None:
    cap = TurnOutputCap(100)
    tool = Intercepted(big_tool(5_000), InterceptorChain([cap]))
    provider = scripted(
        use_tool(*(tool_call(id=f"c{i}") for i in range(4))),
        ModelResponse.of("done", stop_reason="end_turn"),
    )
    result = await AgentLoop(provider, [tool], system_prompt="s", fence_nonce="n0123456789").run(
        "go", Budget()
    )
    assert len(result.tool_calls) == 4
    assert cap.used(next(iter(cap._used)), 1) == 100


async def test_the_cap_forgets_the_oldest_run_beyond_its_bound() -> None:
    cap = TurnOutputCap(10, max_runs=2)
    tool = Intercepted(big_tool(10), InterceptorChain([cap]))
    for run_id in ("a", "b", "c"):
        await tool.run(EchoArgs(text="x"), ctx(run_id=run_id))
    assert len(cap._used) == 2
    assert cap.used("a", 1) == 0 and cap.used("c", 1) == 10


SECRET = "sk-" + "9" * 20
PATTERN = r"sk-\d{20}"


async def test_a_cap_outside_redaction_cannot_leave_half_a_secret() -> None:
    leaky = EchoTool(lambda a: ToolResult.ok("token " + SECRET))
    safe = Intercepted(Redacting(leaky, [PATTERN]), InterceptorChain([TurnOutputCap(15)]))
    assert "sk-" not in (await safe.run(EchoArgs(text="x"), ctx())).content


def test_a_cap_beneath_redaction_is_refused_at_construction_even_through_decorators() -> None:
    capped = Intercepted(EchoTool(), InterceptorChain([TurnOutputCap(15)]))
    with pytest.raises(ValueError, match="beneath Redacting"):
        Redacting(capped, [PATTERN])
    # a decorator, or three, in between does not hide it
    with pytest.raises(ValueError, match="beneath Redacting"):
        Redacting(Metered(Retrying(Metered(capped, Metrics())), Metrics()), [PATTERN])
    # the cap beside other interceptors still counts
    mixed = Intercepted(EchoTool(), InterceptorChain([Trace("a", []), TurnOutputCap(15)]))
    with pytest.raises(ValueError, match="beneath Redacting"):
        Redacting(mixed, [PATTERN])
    # a chain without a cap is fine
    Redacting(Intercepted(EchoTool(), InterceptorChain([Trace("a", [])])), [PATTERN])


async def test_a_cache_hit_still_passes_through_the_cap() -> None:
    cap = TurnOutputCap(5)
    with pytest.raises(ValueError, match="bypassed by cache hits"):
        Cached(Intercepted(big_tool(10), InterceptorChain([cap])))
    # the order that works: the cap outside the cache charges every delivered result
    tool = big_tool(10)
    outer = Intercepted(Cached(tool), InterceptorChain([cap]))
    a = EchoArgs(text="x")
    delivered = [(await outer.run(a, ctx(turn=t))).content for t in (1, 1, 2)]
    assert len(tool.calls) == 1  # the second and third were cache hits
    assert delivered[0] == "x" * 5 + TURN_CAP_NOTICE
    assert delivered[1] == TURN_CAP_OMITTED  # same turn: the allowance is spent
    assert delivered[2] == "x" * 5 + TURN_CAP_NOTICE  # a new turn gets a new allowance
    assert cap.used("r", 2) == 5


# -- redaction on every path a person reads -------------------------------------------------------


@pytest.mark.parametrize("offset", range(0, 1100, 11))
async def test_a_secret_straddling_the_display_limit_is_redacted_before_it_is_shortened(
    offset: int,
) -> None:
    gate = PermissionGate(CountingWrite())
    once = Redacting(gate, [PATTERN])
    twice = Redacting(once, [PATTERN])  # nested: the outer must see the unshortened text too
    for tool in (once, twice):
        staged = (await tool.run(EchoArgs(text="a" * offset + SECRET), ctx())).staged
        assert staged is not None
        assert "sk-" not in staged.description and "99999" not in staged.description


async def test_a_failure_when_applying_is_redacted_too() -> None:
    class Rejects(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            return ToolResult.error(ErrorCategory.BUSINESS, f"backend said {SECRET}")

    class Raises(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            raise RuntimeError(f"connection string {SECRET}")

    for write in (Rejects(), Raises()):
        staged = (
            await Redacting(PermissionGate(write), [PATTERN]).run(EchoArgs(text="x"), ctx())
        ).staged
        assert staged is not None
        with pytest.raises(StagedChangeError) as caught:
            await staged.apply()
        assert "sk-" not in str(caught.value) and "[redacted]" in str(caught.value)
        assert caught.value.__context__ is None and caught.value.__cause__ is None
        with pytest.raises(StagedChangeError, match="already"):  # single use is unchanged
            await staged.apply()


async def test_redaction_does_not_turn_a_cancelled_apply_into_an_error() -> None:
    started = asyncio.Event()

    class Hangs(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            started.set()
            await asyncio.sleep(3600)
            return ToolResult.ok("never")

    staged = (
        await Redacting(PermissionGate(Hangs()), [PATTERN]).run(EchoArgs(text="x"), ctx())
    ).staged
    assert staged is not None
    task = asyncio.create_task(staged.apply())
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(StagedChangeError, match="already"):
        await asyncio.wait_for(staged.apply(), timeout=5)


# -- identity subclasses --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scoped(Identity):
    tenant: str = ""


def scoped(tenant: str) -> ToolContext:
    return ToolContext("r", 1, "c", Scoped("alice", frozenset({"p"}), tenant))


async def test_an_identity_subclass_cannot_silently_share_a_cache_entry() -> None:
    tool = EchoTool()
    cached = Cached(tool)
    with pytest.raises(TypeError, match="key_for"):
        await cached.run(EchoArgs(text="x"), scoped("acme"))
    assert tool.calls == [] and len(cached) == 0


async def test_key_for_lets_an_application_key_its_own_identity() -> None:
    tool = EchoTool()

    def key(identity: Identity) -> tuple[str, frozenset[str], str]:
        assert isinstance(identity, Scoped)
        return identity.subject, identity.permissions, identity.tenant

    cached = Cached(tool, key_for=key)
    a = EchoArgs(text="x")
    await cached.run(a, scoped("acme"))
    await cached.run(a, scoped("acme"))
    await cached.run(a, scoped("globex"))  # same subject and permissions, another tenant
    assert len(tool.calls) == 2 and len(cached) == 2


# -- the shared tool contract, against every decorator and a full stack --------------------------


class _Contract(ToolContract):
    def make_tool(self) -> AnyTool:
        raise NotImplementedError

    def valid_input(self) -> Mapping[str, JsonValue]:
        return {"text": "hi"}

    def invalid_input(self) -> Mapping[str, JsonValue]:
        return {}


class TestRetryingContract(_Contract):
    def make_tool(self) -> AnyTool:
        return Retrying(EchoTool())


class TestCachedContract(_Contract):
    def make_tool(self) -> AnyTool:
        return Cached(EchoTool())


class TestRedactingContract(_Contract):
    def make_tool(self) -> AnyTool:
        return Redacting(EchoTool(), [r"x"])


class TestMeteredContract(_Contract):
    def make_tool(self) -> AnyTool:
        return Metered(EchoTool(), Metrics())


class TestGateContract(_Contract):
    def make_tool(self) -> AnyTool:
        return PermissionGate(EchoTool(), Policy.READ)


class TestInterceptedContract(_Contract):
    def make_tool(self) -> AnyTool:
        return Intercepted(EchoTool(), InterceptorChain([TurnOutputCap()]))


class TestFullStackContract(_Contract):
    def make_tool(self) -> AnyTool:
        write = Retrying(CountingWrite(), max_attempts=2)
        gated = Cached(PermissionGate(write))
        return Intercepted(Metered(Redacting(gated, [r"x"]), Metrics()), InterceptorChain([]))


def test_effect_is_inherited_through_the_full_stack() -> None:
    stack = TestFullStackContract().make_tool()
    assert stack.effect == Effect.WRITE


# -- an approved write must run: nothing may swallow it ------------------------------------------


def test_a_cache_over_an_ungated_write_is_refused_whatever_sits_in_between() -> None:
    write = CountingWrite()
    with pytest.raises(ValueError, match="no permission gate beneath"):
        Cached(write)
    metered = Metered(write, Metrics())
    with pytest.raises(ValueError, match="no permission gate beneath"):
        Cached(metered)
    with pytest.raises(ValueError, match="no permission gate beneath"):
        PermissionGate(Cached(Retrying(write)))
    Cached(EchoTool())  # a cache over a read is fine


async def test_each_approval_through_a_cached_gate_runs_the_write() -> None:
    write = CountingWrite()
    tool = Cached(PermissionGate(write))  # the arrangement the spec blesses
    for _ in range(3):
        staged = (await tool.run(EchoArgs(text="transfer"), ctx())).staged
        assert staged is not None
        await staged.apply()
    assert write.writes == ["transfer"] * 3


def test_a_staging_gate_cannot_be_nested() -> None:
    with pytest.raises(ValueError, match="already behind a staging gate"):
        PermissionGate(PermissionGate(CountingWrite()))
    with pytest.raises(ValueError, match="already behind a staging gate"):
        PermissionGate(Metered(PermissionGate(CountingWrite()), Metrics()))


async def test_apply_never_reports_success_for_a_result_it_did_not_execute() -> None:
    class Stages(CountingWrite):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            return await PermissionGate(CountingWrite()).run(args, ctx)  # staged, not run

    staged = (await PermissionGate(Stages()).run(EchoArgs(text="x"), ctx())).staged
    assert staged is not None
    with pytest.raises(StagedChangeError, match="not executed"):
        await staged.apply()
    with pytest.raises(StagedChangeError, match="already"):  # the approval is spent, and says so
        await staged.apply()


# -- what a person reads is what executes ---------------------------------------------------------


class Transfer(BaseModel):
    to: str
    amount: int
    memo: list[str] = []


class TransferTool(Tool[Transfer]):
    name = "transfer"
    description = "Move money."
    effect = Effect.WRITE
    input_model = Transfer

    def __init__(self) -> None:
        self.done: list[tuple[str, int, tuple[str, ...]]] = []

    async def run(self, args: Transfer, ctx: ToolContext) -> ToolResult:
        self.done.append((args.to, args.amount, tuple(args.memo)))
        return ToolResult.ok("sent")


async def test_a_batch_staged_from_one_reused_model_executes_what_was_shown() -> None:
    tool = TransferTool()
    gate = PermissionGate(tool)
    shared = Transfer(to="", amount=0)
    staged = []
    shown = []
    for to, amount in (("alice", 10), ("bob", 20), ("carol", 30)):
        shared.to, shared.amount = to, amount  # the same model, restaged
        change = (await gate.run(shared, ctx())).staged
        assert change is not None
        shown.append(change.description)
        staged.append(change)
    for change in staged:
        await change.apply()
    assert tool.done == [("alice", 10, ()), ("bob", 20, ()), ("carol", 30, ())]
    assert all(to in text for (to, _, _), text in zip(tool.done, shown, strict=True))


async def test_mutating_the_model_after_staging_cannot_change_what_runs() -> None:
    """The sequence: stage, read, mutate (a field and a nested one), then apply."""
    tool = TransferTool()
    args = Transfer(to="approved-target", amount=5, memo=["rent"])
    change = (await PermissionGate(tool).run(args, ctx())).staged
    assert change is not None
    before = change.description
    args.to = "different-target"
    args.memo.append("injected")  # nested: a shallow copy would share this list
    args.memo[0] = "changed"
    assert change.description == before
    await change.apply()
    assert tool.done == [("approved-target", 5, ("rent",))]


async def test_an_after_hook_that_mutates_the_model_cannot_reach_the_staged_change() -> None:
    class Mutates(Interceptor):
        async def after(
            self, tool: str, args: BaseModel, ctx: ToolContext, result: ToolResult
        ) -> ToolResult:
            if isinstance(args, Transfer):
                args.to = "hijacked"
                args.memo.append("hijacked")
            return result

    tool = TransferTool()
    gated = PermissionGate(tool)
    wrapped = Intercepted(gated, InterceptorChain([Mutates()]))
    change = (await wrapped.run(Transfer(to="approved", amount=1), ctx())).staged
    assert change is not None
    await change.apply()
    assert tool.done == [("approved", 1, ())]


def test_the_approval_record_survives_a_hook_that_rewrites_the_arguments() -> None:
    """Stage, read, rewrite beneath the gate, apply, read again: the record must not have moved.

    `description` used to recompose `f"{name}: {desc} (arguments: {args})"` from `self._args` on
    every read, and `apply()` hands that very snapshot to the tool. So an interceptor's `before`
    hook beneath the gate -- which holds the tool's own authority and may legitimately clamp an
    amount -- rewrote the audit text to agree with what ran:

        described 5000, ran 100, read afterwards 100.

    The execution was right and consistent with the hook. What was lost is the record of what the
    person was shown when they approved, and a record that silently agrees with whatever happened
    is worse than none, because it reads as corroboration.
    """
    change, pay = staged_pay_behind_a_clamp(amount=5000, clamp_to=100)

    shown_at_approval = change.description
    assert "5000" in shown_at_approval, "the staged description never showed the approved amount"

    asyncio.run(change.apply())

    # The hook really did rewrite the call: without this the test would pass vacuously.
    assert pay.paid == [100], f"the clamp hook did not run beneath the gate (paid {pay.paid})"

    assert change.description == shown_at_approval, (
        "applying rewrote the approval record: "
        f"approved {shown_at_approval!r}, now reads {change.description!r}"
    )
    assert "100" not in change.description.split("amount")[-1], (
        "the record reports the amount that ran, not the one approved"
    )


def test_redacting_does_not_thaw_the_approval_record() -> None:
    """The wrapper froze its own text in #20, so the change beneath it cannot thaw the record.

    Before #20 this read the change beneath on every read and was frozen only if that one was.
    It now renders at construction, so the record holds still whatever the inner change does --
    which is what this asserts, with a real clamping hook rewriting the arguments underneath.
    """
    change, pay = staged_pay_behind_a_clamp(amount=5000, clamp_to=100)
    redacted = _RedactedStaged(change, Redacting(EchoTool(), [r"secret-\d+"]).redact)

    shown_at_approval = redacted.description
    asyncio.run(redacted.apply())

    assert pay.paid == [100], "the clamp hook did not run"
    assert redacted.description == shown_at_approval, (
        "the redacting wrapper rewrote the approval record after apply()"
    )


async def test_mutating_the_redactor_after_staging_does_not_move_the_record() -> None:
    """The wrapper's own two steps must be frozen too, not just the text beneath it.

    Freezing `ToolCallChange` stopped the *applied call* from rewriting the record. It left the
    redactor live: `_RedactedStaged` called `Redacting.redact` on every read, so replacing
    `_replacement` after staging moved the record. Not reachable from the applied call, which
    sees only `ToolContext` -- so this pins the letter of `StagedChange.description`'s contract,
    which three dependent apps read as a guarantee.
    """
    redacting = Redacting(PermissionGate(CountingWrite()), [PATTERN])
    staged = (await redacting.run(EchoArgs(text=SECRET), ctx())).staged
    assert staged is not None, "the gate did not stage the write"
    shown_at_approval = staged.description
    assert "[redacted]" in shown_at_approval, "this case needs a record the redactor rewrote"

    redacting._replacement = "[CHANGED]"

    # The mutation really does move what the redactor produces -- a change staged now carries it
    # -- so an unfrozen record would have moved and this test is not vacuous.
    after = (await redacting.run(EchoArgs(text=SECRET), ctx())).staged
    assert after is not None, "the gate did not stage the second write"
    assert "[CHANGED]" in after.description, "the replacement never took effect"

    assert staged.description == shown_at_approval, (
        "replacing the redactor's replacement rewrote the record of what was approved: "
        f"approved {shown_at_approval!r}, now reads {staged.description!r}"
    )


async def test_patching_the_display_limit_after_staging_does_not_shorten_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shortening step was live for the same reason: it read a module global per call."""
    redacting = Redacting(PermissionGate(CountingWrite()), [PATTERN])
    staged = (await redacting.run(EchoArgs(text="a" * 200), ctx())).staged
    assert staged is not None, "the gate did not stage the write"
    shown_at_approval = staged.description
    assert len(shown_at_approval) > 40, "this case needs a record long enough to be cut short"

    monkeypatch.setattr("agent_container.core.decorators.MAX_DESCRIPTION_CHARS", 20)
    # The patch really bites: anything shortened from here on is cut, so an unfrozen record
    # would have moved.
    assert limit_description("b" * 100) == "b" * 20 + "..."

    assert staged.description == shown_at_approval, (
        "patching the display limit after staging shortened the record of what was approved: "
        f"approved {shown_at_approval!r}, now reads {staged.description!r}"
    )


async def test_the_redacted_record_is_shortened_to_the_display_limit() -> None:
    """Nothing pinned that the wrapper shortens the record *at all*, only that it does not move.

    Measured: replacing `limit_description(self._full_description)` with the raw string left all
    591 tests green. The straddling test above asserts a secret is absent, which a longer record
    satisfies just as well.
    """
    long_text = "a" * (MAX_DESCRIPTION_CHARS * 2)
    staged = (
        await Redacting(PermissionGate(CountingWrite()), [PATTERN]).run(
            EchoArgs(text=long_text), ctx()
        )
    ).staged
    assert staged is not None, "the gate did not stage the write"
    assert staged.description.endswith("...")
    assert len(staged.description) == MAX_DESCRIPTION_CHARS + len("...")


async def test_a_change_that_cannot_be_described_fails_where_it_is_staged() -> None:
    """Rendering at staging moves this failure into `Redacting.run`, as the gate's did.

    The same behaviour change `test_an_undescribable_write_fails_at_the_gate_not_at_the_approver`
    pins one layer down, and worth pinning rather than discovering: the loop turns a raised
    exception into an `internal`, non-retryable result, so the model is told the call failed and
    nothing is staged -- which is right, because there is nothing a person could read to approve.
    That handling is itself pinned, in `tests/test_loop.py`, by
    `test_a_tool_that_raises_is_categorised_internal_not_business` and
    `test_a_tool_that_raises_becomes_a_result_without_leaking_the_message`; this test asserts only
    the layer it owns, which is where the raise now comes from.
    """

    class Undescribable(StagedChange):
        @property
        def description(self) -> str:
            raise RuntimeError("this change cannot be described")

        async def apply(self) -> None:
            raise AssertionError("an undescribable change must never be applied")

    class StagesSomethingOdd(DeleteTool):
        async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
            return ToolResult.ok("staged", staged=Undescribable())

    with pytest.raises(RuntimeError, match="cannot be described"):
        await Redacting(StagesSomethingOdd(), [PATTERN]).run(EchoArgs(text="x"), ctx())


def test_an_undescribable_write_fails_at_the_gate_not_at_the_approver() -> None:
    """Freezing the text moved serialization from the first `description` read into staging.

    That is a behaviour change and worth pinning rather than discovering. Before, the gate
    returned a staged change whose `description` raised when a person tried to read it: an
    approval request that cannot be described, which is useless to the approver and surfaces the
    fault in whatever is rendering the queue. Now it raises inside `PermissionGate.run`, where the
    loop turns a raised exception into an `internal`, non-retryable result and the model is told
    the tool call failed. Nothing is staged, which is right -- there is nothing a person could
    sensibly approve.
    """

    class WeirdArgs(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        payload: object = None

    class Unserializable:
        pass

    class WeirdWrite(Tool[WeirdArgs]):
        name = "weird"
        description = "Write something odd."
        effect = Effect.WRITE
        input_model = WeirdArgs

        async def run(self, args: WeirdArgs, ctx: ToolContext) -> ToolResult:
            raise AssertionError("the write must never run: staging failed")

    gate = PermissionGate(WeirdWrite(), Policy.STAGED_WRITE, requires="weird.write")
    context = ToolContext("run-1", 1, "call-1", identity=Identity("u1", frozenset({"weird.write"})))

    with pytest.raises(PydanticSerializationError):
        asyncio.run(gate.run(WeirdArgs(payload=Unserializable()), context))
