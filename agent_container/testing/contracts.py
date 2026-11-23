"""The shared contract suites: what every `ModelProvider` and every `Tool` (decorators included)
must do.

Substitutability is proven, not asserted: a provider swap or a decorator stack is safe because each
implementation passes the same tests. 17.3 runs `BedrockProvider` (against a stubbed client) through
`ModelProviderContract`; 17.4 runs `Retrying`, `Cached`, `Redacting`, `Metered` and `PermissionGate`
through `ToolContract`.

The suites are plain classes whose tests are synchronous, so they need no pytest plugin. They do not
import pytest. Checks raise `AssertionError` explicitly so they still fail under `python -O`.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue

from agent_container.core.errors import UnrecognizedStopReason
from agent_container.core.messages import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolUseBlock,
    Usage,
)
from agent_container.core.provider import ModelProvider
from agent_container.core.tool import (
    TOOL_NAME_PATTERN,
    AnyTool,
    Effect,
    ErrorCategory,
    StagedChange,
    ToolContext,
    ToolResult,
    innermost,
    walk_stack,
)

_TOOL_NAME = TOOL_NAME_PATTERN  # one definition, in core.tool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _request(text: str = "hello") -> ModelRequest:
    return ModelRequest(system="You are a test.", messages=(Message.user_text(text),))


class OverlapProbe:
    """Proves calls overlap. Each call waits at a barrier until `parties` calls have all arrived.

    A provider that serialises its calls can never get a second call to the barrier while the first
    is waiting, so it times out and the test fails; "every call got a reply" cannot tell the two
    apart. The barrier is deterministic: no sleeps, no timing windows.
    """

    TIMEOUT_SECONDS = 5.0

    def __init__(self, parties: int) -> None:
        self.parties = parties
        self.arrived = 0
        self._lock = threading.Lock()
        self._thread_barrier = threading.Barrier(parties, timeout=self.TIMEOUT_SECONDS)
        self._async_barrier = asyncio.Barrier(parties)

    def _arrive(self) -> None:
        with self._lock:
            self.arrived += 1

    async def wait_async(self) -> None:
        """For a provider whose call is a coroutine: await this inside the call."""
        self._arrive()
        try:
            async with asyncio.timeout(self.TIMEOUT_SECONDS):
                await self._async_barrier.wait()
        except TimeoutError:
            raise AssertionError("calls did not overlap: a call waited alone") from None

    def wait_blocking(self) -> None:
        """For a provider whose call runs on a worker thread: call this inside that thread."""
        self._arrive()
        try:
            self._thread_barrier.wait()
        except threading.BrokenBarrierError:
            raise AssertionError("calls did not overlap: a call waited alone") from None


class ModelProviderContract(ABC):
    """Subclass, implement `make_provider`, and every test below runs against your provider."""

    @abstractmethod
    def make_provider(
        self, replies: Sequence[ModelResponse], probe: OverlapProbe | None = None
    ) -> ModelProvider:
        """A provider that answers successive calls with exactly these replies, in order.

        When `probe` is given, the provider must call `probe.wait_async()` (or, if the call runs on
        a worker thread, `probe.wait_blocking()` from that thread) once per call, at the point where
        the underlying model call is in flight. A decorator passes the probe to what it wraps and
        must not hold anything that would stop three calls being in flight at once.
        """

    def test_converse_is_async(self) -> None:
        provider = self.make_provider([ModelResponse.of("hi")])
        _require(inspect.iscoroutinefunction(provider.converse), "converse must be `async def`")

    def test_returns_text_and_stop_reason(self) -> None:
        provider = self.make_provider([ModelResponse.of("hello there", stop_reason="end_turn")])
        response = asyncio.run(provider.converse(_request()))
        _require(isinstance(response, ModelResponse), "converse must return a ModelResponse")
        _require(response.text == "hello there", f"text was {response.text!r}")
        _require(response.stop_reason == "end_turn", f"stop reason was {response.stop_reason!r}")

    def test_returns_tool_calls_intact(self) -> None:
        call = ToolUseBlock("call-1", "lookup", {"q": "x", "n": 2})
        provider = self.make_provider([ModelResponse.of(stop_reason="tool_use", tool_uses=[call])])
        response = asyncio.run(provider.converse(_request()))
        _require(response.stop_reason == "tool_use", "tool_use stop reason lost")
        _require(response.tool_uses == (call,), f"tool call changed: {response.tool_uses!r}")

    def test_reports_usage(self) -> None:
        provider = self.make_provider([ModelResponse.of("x", usage=Usage(12, 7))])
        response = asyncio.run(provider.converse(_request()))
        _require(response.usage == Usage(12, 7), f"usage was {response.usage!r}")

    def test_passes_every_known_stop_reason_through_unchanged(self) -> None:
        for reason in StopReason:
            provider = self.make_provider([ModelResponse.of("x", stop_reason=reason.value)])
            response = asyncio.run(provider.converse(_request()))
            _require(
                response.stop_reason == reason.value,
                f"{reason.value!r} became {response.stop_reason!r}",
            )

    def test_never_maps_an_unknown_stop_reason_onto_a_known_one(self) -> None:
        provider = self.make_provider([ModelResponse.of("x", stop_reason="not_a_real_reason")])
        try:
            response = asyncio.run(provider.converse(_request()))
        except UnrecognizedStopReason:
            return  # refusing is fine; the loop would have raised too
        _require(
            response.stop_reason == "not_a_real_reason",
            f"an unknown stop reason was turned into {response.stop_reason!r}",
        )

    def test_does_not_modify_the_request(self) -> None:
        provider = self.make_provider([ModelResponse.of("x")])
        request = _request()
        before = copy.deepcopy(request)
        asyncio.run(provider.converse(request))
        _require(request == before, "converse changed the request it was given")

    def test_concurrent_calls_each_get_a_reply(self) -> None:
        replies = [ModelResponse.of(f"reply {i}") for i in range(3)]
        provider = self.make_provider(replies)

        async def fan_out() -> list[ModelResponse]:
            return list(await asyncio.gather(*(provider.converse(_request()) for _ in replies)))

        got = asyncio.run(fan_out())
        _require(
            sorted(r.text for r in got) == sorted(r.text for r in replies),
            "concurrent calls did not each receive one reply",
        )

    def test_concurrent_calls_are_in_flight_together(self) -> None:
        """Answering everyone is not concurrency: a provider that queues calls does that too."""
        replies = [ModelResponse.of(f"reply {i}") for i in range(3)]
        probe = OverlapProbe(len(replies))
        provider = self.make_provider(replies, probe)

        async def fan_out() -> list[ModelResponse]:
            return list(await asyncio.gather(*(provider.converse(_request()) for _ in replies)))

        got = asyncio.run(fan_out())
        _require(len(got) == len(replies), "a concurrent call got no reply")
        _require(
            probe.arrived == len(replies),
            f"{probe.arrived} of {len(replies)} calls reached the model call; the probe was "
            "not honoured by `make_provider`",
        )


class ToolContract(ABC):
    """Subclass and supply a tool plus one valid and one invalid input. Use it for leaf tools and
    for every decorator stack: the same tests hold for both, which is what makes a decorator
    substitutable."""

    @abstractmethod
    def make_tool(self) -> AnyTool:
        """A fresh tool (or decorator stack) to test."""

    @abstractmethod
    def valid_input(self) -> Mapping[str, JsonValue]:
        """Arguments that pass the tool's input model."""

    @abstractmethod
    def invalid_input(self) -> Mapping[str, JsonValue]:
        """Arguments that fail the tool's input model."""

    def _ctx(self) -> ToolContext:
        return ToolContext(run_id="run-1", turn=1, tool_use_id="call-1")

    def test_has_a_usable_identity(self) -> None:
        tool = self.make_tool()
        _require(bool(_TOOL_NAME.match(tool.name)), f"bad tool name {tool.name!r}")
        _require(bool(tool.description.strip()), "a tool needs a description")
        _require(isinstance(tool.effect, Effect), "effect must be an Effect, read or write")
        _require(issubclass(tool.input_model, BaseModel), "input_model must be a pydantic model")
        _require(
            tool.input_schema().get("type") == "object", "the input schema must describe an object"
        )

    def test_run_is_async(self) -> None:
        _require(inspect.iscoroutinefunction(self.make_tool().invoke), "invoke must be `async def`")

    def test_stack_can_be_walked_to_the_innermost_tool(self) -> None:
        tool = self.make_tool()
        stack = list(walk_stack(tool))
        _require(stack[0] is tool, "the walk must start at the tool itself")
        _require(innermost(tool) is stack[-1], "innermost must be the last tool in the walk")
        _require(stack[-1].wrapped is None, "the innermost tool must not wrap anything")

    def test_decorators_inherit_the_innermost_effect_and_identity(self) -> None:
        tool = self.make_tool()
        leaf = innermost(tool)
        for layer in walk_stack(tool):
            _require(layer.effect == leaf.effect, f"{type(layer).__name__} changed the effect")
            _require(layer.name == leaf.name, f"{type(layer).__name__} changed the name")
            _require(
                layer.input_model is leaf.input_model,
                f"{type(layer).__name__} changed the input model",
            )

    def test_valid_input_returns_a_result(self) -> None:
        result = asyncio.run(self.make_tool().invoke(self.valid_input(), self._ctx()))
        _require(isinstance(result, ToolResult), "invoke must return a ToolResult")

    def test_invalid_input_is_a_validation_result_not_an_exception(self) -> None:
        result = asyncio.run(self.make_tool().invoke(self.invalid_input(), self._ctx()))
        _require(result.is_error, "invalid input must produce an error result")
        _require(
            result.error_category is ErrorCategory.VALIDATION,
            f"category was {result.error_category!r}",
        )

    def test_a_second_call_works(self) -> None:
        tool = self.make_tool()
        asyncio.run(tool.invoke(self.valid_input(), self._ctx()))
        again = asyncio.run(tool.invoke(self.valid_input(), self._ctx()))
        _require(isinstance(again, ToolResult), "a tool must be callable more than once")


@dataclass(frozen=True)
class StagedCase:
    """One staged change to put through `StagedChangeContract`, with what the suite needs to
    check that the case really exercises a freeze.

    `approved_value` and `rewritten_value` are not taken on trust: the suite asserts each appears
    where it must and is absent where it must not, so a case that does not genuinely rewrite the
    described argument fails rather than passing quietly.
    """

    change: StagedChange
    """Freshly staged, not yet applied."""

    applied: Callable[[], Sequence[object]]
    """What the tool beneath actually received, in call order. Empty before `apply()`."""

    approved_value: str
    """The argument as the staged record reports it -- must appear in `description`."""

    rewritten_value: str
    """The argument as the applied call receives it -- must not appear in `description`."""

    recomposed: Callable[[], str]
    """What the record *would* read if it were rendered now, from the arguments as they stand.

    Called twice: before `apply()`, where the suite requires it to **equal** the record, and
    after, where it must **differ**. Without the difference the four substring relations above can
    all hold while the rewritten argument is one the record never displays, in which case a live
    description would not have moved either and the freeze test passes on a change that is not
    frozen. Without the equality the difference can come from anywhere -- a timestamp, a read
    counter, a step this renderer does not share with the record -- which is how five
    provably-unfrozen witnesses passed. It must render the record, not merely resemble it: same
    pipeline, reading real state rather than literals."""


class StagedChangeContract(ABC):
    """What every `StagedChange` must do: hold still.

    `description` is the record of what a person was shown when they approved. Applying the change
    must not alter it, whatever the applied call does -- an interceptor beneath a gate may clamp or
    normalise an argument, and the execution then rightly follows the hook, but the record of the
    approval must still say what the approver read. An audit trail that agrees with whatever
    happened is worse than none: it reads as corroboration.

    Subclass and supply a `StagedCase`. Every part of it is checked, because two earlier versions
    of this suite could be passed by something that was not frozen at all:

    - The first asked the subclass to *declare* `rewrites_arguments() -> True`. An unverifiable
      promise: a change that recomposed its description live, with a **no-op** `apply()`, passed
      all three tests green.
    - The second checked a probe, but one-sidedly -- it asserted the applied value was *absent*
      from the record and never that the approved value was *present*. Two cases defeated it with
      a live, unfrozen description: a probe reporting the arguments model (`'amount=5000'`) rather
      than the scalar, so the substring never matched; and a tool description long enough that
      `limit_description` truncated the arguments off the end, so the check was trivially true.

    So both values are pinned from both sides: a case must have a value the record reports and the
    applied call did not receive, *and* a value the applied call received and the record does not
    report. Those four relations are still only facts about two strings, and a third version of
    this suite overstated them -- nothing in them ties the two values to the **same** argument, so
    a case that rewrote a field the record never displays satisfied all four while a provably live
    description went green. `recomposed` closes that: the suite renders the record from the
    arguments as they stand after `apply()` and requires it to differ from the staged record, so
    the rewrite must be one a live description would have shown.

    A fourth version checked that difference and nothing else, which let the difference arise for
    a reason unrelated to the rewrite. Five of six provably-unfrozen witnesses passed it: a
    `recomposed` carrying a timestamp or a read counter differed on every call; one differing by a
    pipeline step the record has and it does not (which is not fabrication -- it can be a real,
    byte-for-byte render of real state, just not of the record under test); one rendering the same
    argument a different way, so it differed *because of* the rewrite and looked like the check
    firing correctly. So the suite now **calibrates the two renderers before applying**: rendered
    at stage time, `recomposed()` must equal the record. A renderer that agrees with the record
    before `apply()` and differs after can only differ because of what `apply()` did.

    What remains on trust: `applied` and `recomposed` are the subclass's own observations of its
    own implementation. A subclass that fabricates either -- a probe returning a value no call
    produced, a renderer returning a string it never renders -- can still pass, though the
    calibration narrows that too: a fabricated renderer must now match the record exactly at stage
    time. The suite checks that `applied` is consistent with the record, and that `recomposed`
    agrees with it before `apply()` and disagrees after; it cannot check that either is honest.
    Write them as thin views over real state, not as literals.
    """

    @abstractmethod
    def make_change(self) -> StagedCase:
        """A fresh `StagedCase`. Called more than once; return a new change each time."""

    def test_the_description_says_something(self) -> None:
        case = self.make_change()
        _require(bool(case.change.description.strip()), "a staged change needs a description")

    def test_reading_the_description_twice_gives_the_same_answer(self) -> None:
        change = self.make_change().change
        first = change.description
        second = change.description
        _require(first == second, f"description moved between reads: {first!r} then {second!r}")

    def test_applying_does_not_rewrite_the_approval_record(self) -> None:
        case = self.make_change()
        _require(not case.applied(), "the probe reports a call before apply() was called")
        shown_at_approval = case.change.description

        # The record must report the approved value. Without this the rest can be satisfied by a
        # description that never mentioned the argument at all -- a truncated one, say.
        _require(
            case.approved_value in shown_at_approval,
            f"the staged description {shown_at_approval!r} does not report the approved value "
            f"{case.approved_value!r}, so this case cannot show whether the record is frozen",
        )
        _require(
            case.rewritten_value not in shown_at_approval,
            f"the staged description already reports {case.rewritten_value!r}, the value the "
            "applied call receives, so a live description would read the same as a frozen one",
        )

        # Calibration, before anything is applied: the case's own renderer must agree with the
        # record it is meant to re-render. Without this the difference asserted after `apply()`
        # can come from anywhere -- a timestamp, a read counter, a pipeline step the renderer
        # does not share with the record -- and five provably-unfrozen witnesses passed on
        # exactly that. Agreement now, disagreement later, is what ties the difference to the
        # rewrite.
        _require(
            case.recomposed() == shown_at_approval,
            "this case's `recomposed` does not render the record: before apply() it reads "
            f"{case.recomposed()!r} while the record reads {shown_at_approval!r}. A renderer "
            "that disagrees with the record already cannot show that applying moved it.",
        )

        # A failing apply is fine here: the record must survive either way.
        with contextlib.suppress(Exception):
            asyncio.run(case.change.apply())

        received = list(case.applied())
        _require(bool(received), "apply() never reached the tool, so this test proves nothing")
        applied_repr = str(received[-1])
        _require(
            case.rewritten_value in applied_repr,
            f"the applied call received {applied_repr!r}, not the rewritten value "
            f"{case.rewritten_value!r} this case promised",
        )
        _require(
            case.approved_value not in applied_repr,
            f"the applied call received {applied_repr!r}, which still carries the approved value "
            f"{case.approved_value!r}: nothing rewrote the argument, so this test proves nothing",
        )

        # The rewrite must be one the record would have shown. Otherwise a live description would
        # read the same as a frozen one and the assertion below proves nothing.
        would_read_now = case.recomposed()
        _require(
            would_read_now != shown_at_approval,
            f"rendering the record from the arguments as they now stand gives {would_read_now!r}, "
            f"the same text as the staged record: the applied call rewrote something the record "
            "never displays, so this test cannot tell a frozen description from a live one",
        )

        _require(
            case.change.description == shown_at_approval,
            "applying the change rewrote the record of what was approved: "
            f"approved {shown_at_approval!r}, now reads {case.change.description!r}",
        )
