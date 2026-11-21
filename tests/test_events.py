"""Event bus (Observer): the four named events, and subscribers that cannot change a run."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent_container.core import (
    EVENTS,
    AgentLoop,
    AnyTool,
    Budget,
    EndReason,
    ErrorCategory,
    EventBus,
    LoopEvent,
    LoopEventKind,
    ModelCalled,
    ModelProvider,
    ModelResponse,
    ToolCalled,
    ToolFailed,
    ToolResult,
    Usage,
)
from tests.support import EchoTool, scripted, tool_call, use_tool

NONCE = "testnonce0123456"


class Recorder:
    def __init__(self, *, explode: bool = False) -> None:
        self.events: list[LoopEvent] = []
        self.explode = explode

    async def on_event(self, event: LoopEvent) -> None:
        self.events.append(event)
        if self.explode:
            raise RuntimeError("subscriber bug")


def _loop(provider: ModelProvider, *tools: AnyTool, bus: EventBus) -> AgentLoop:
    return AgentLoop(
        provider,
        list(tools),
        system_prompt="You are a test agent.",
        observers=[bus],
        fence_nonce=NONCE,
    )


def test_the_public_interface_names_four_events() -> None:
    assert EVENTS == (
        LoopEventKind.MODEL_CALLED,
        LoopEventKind.TOOL_CALLED,
        LoopEventKind.TOOL_FAILED,
        LoopEventKind.BUDGET_EXCEEDED,
    )
    assert ModelCalled is LoopEventKind.MODEL_CALLED
    assert ToolCalled is LoopEventKind.TOOL_CALLED
    assert ToolFailed is LoopEventKind.TOOL_FAILED


async def test_the_bus_publishes_the_four_events() -> None:
    bus = EventBus()
    rec = Recorder()
    bus.subscribe(rec)
    bad = EchoTool(lambda _: ToolResult.error(ErrorCategory.BUSINESS, "no"))
    provider = scripted(use_tool(tool_call(), usage=Usage(5, 5)), ModelResponse.of("x"))
    await _loop(provider, bad, bus=bus).run("go", Budget(max_turns=1))
    kinds = [e.kind for e in rec.events]
    assert kinds == [
        ModelCalled,
        ToolCalled,
        ToolFailed,
        LoopEventKind.BUDGET_EXCEEDED,
    ]
    assert rec.events[0].output_tokens == 5
    assert rec.events[1].tool == "echo"


async def test_a_failing_subscriber_cannot_change_the_run() -> None:
    bus = EventBus()
    bus.subscribe(Recorder(explode=True))
    with_bug = scripted(use_tool(tool_call()), ModelResponse.of("same"))
    clean = scripted(use_tool(tool_call()), ModelResponse.of("same"))
    a = await _loop(with_bug, EchoTool(), bus=bus).run("go", Budget())
    b = await AgentLoop(
        clean,
        [EchoTool()],
        system_prompt="You are a test agent.",
        fence_nonce=NONCE,
    ).run("go", Budget())
    assert (a.end_reason, a.final_text, a.turns_used) == (b.end_reason, b.final_text, b.turns_used)
    assert a.end_reason is EndReason.END_TURN


async def test_one_failing_subscriber_does_not_silence_the_others() -> None:
    bus = EventBus()
    silent = Recorder(explode=True)
    other = Recorder()
    bus.subscribe(silent)
    bus.subscribe(other)
    await _loop(scripted(ModelResponse.of("x")), bus=bus).run("go", Budget())
    assert [e.kind for e in other.events] == [ModelCalled]


async def test_an_event_cannot_be_modified_by_a_subscriber() -> None:
    bus = EventBus()
    rec = Recorder()
    bus.subscribe(rec)
    await _loop(scripted(ModelResponse.of("x")), bus=bus).run("go", Budget())
    with pytest.raises(FrozenInstanceError):
        rec.events[0].kind = ToolFailed  # type: ignore[misc]


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    rec = Recorder()
    bus.subscribe(rec)
    bus.unsubscribe(rec)
    await bus.on_event(
        LoopEvent(LoopEventKind.MODEL_CALLED, run_id="r", turn=1),
    )
    assert rec.events == []


def test_subscribing_twice_delivers_once() -> None:
    bus = EventBus()
    rec = Recorder()
    bus.subscribe(rec)
    bus.subscribe(rec)
    assert list(bus.subscribers) == [rec]


def test_a_bus_cannot_subscribe_to_itself() -> None:
    bus = EventBus()
    with pytest.raises(ValueError, match="itself"):
        bus.subscribe(bus)


async def test_a_subscriber_return_value_is_ignored() -> None:
    """`on_event` is typed to return None; EventBus never reads a result even if one slips out."""

    class ReturnsNone:
        async def on_event(self, event: LoopEvent) -> None:
            return None

    bus = EventBus()
    bus.subscribe(ReturnsNone())
    result = await _loop(scripted(ModelResponse.of("ok")), bus=bus).run("go", Budget())
    assert result.final_text == "ok"
    assert result.end_reason is EndReason.END_TURN
