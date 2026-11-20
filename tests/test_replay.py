from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_container.core import (
    ModelRequest,
    ModelResponse,
    ReplayExhausted,
    ReplayFormatError,
    ReplayProvider,
    ToolUseBlock,
    Usage,
    dump_replies,
    parse_replies,
)
from agent_container.core.messages import Message

REQUEST = ModelRequest(system="s", messages=(Message.user_text("hi"),))


async def test_replies_come_back_in_order_and_requests_are_recorded() -> None:
    provider = ReplayProvider([ModelResponse.of("one"), ModelResponse.of("two")])
    assert (await provider.converse(REQUEST)).text == "one"
    assert (await provider.converse(REQUEST)).text == "two"
    assert provider.requests == [REQUEST, REQUEST]
    assert provider.remaining == 0


async def test_running_out_raises() -> None:
    provider = ReplayProvider([ModelResponse.of("only")])
    await provider.converse(REQUEST)
    with pytest.raises(ReplayExhausted):
        await provider.converse(REQUEST)


def test_a_recording_round_trips() -> None:
    replies = [
        ModelResponse.of(
            "looking",
            stop_reason="tool_use",
            tool_uses=[ToolUseBlock("t1", "lookup", {"q": "x", "n": [1, 2]})],
            usage=Usage(120, 18),
        ),
        ModelResponse.of("done", usage=Usage(150, 4)),
    ]
    data = json.loads(json.dumps(dump_replies(replies)))
    assert parse_replies(data) == replies


def test_the_scaffolding_fixture_still_loads() -> None:
    path = Path(__file__).parent.parent / "agent_container" / "fixtures" / "demo.json"
    provider = ReplayProvider.from_file(path)
    assert provider.remaining == 2


def test_an_unrecognized_stop_reason_is_carried_through_for_the_loop_to_reject() -> None:
    (reply,) = parse_replies({"replies": [{"stop_reason": "banana"}]})
    assert reply.stop_reason == "banana"


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"replies": [{}]},
        {"replies": [{"stop_reason": "end_turn", "surprise": 1}]},
        {"replies": [{"stop_reason": "end_turn", "usage": {"input_tokens": -1}}]},
        {"replies": [{"stop_reason": "tool_use", "tool_uses": [{"id": "x"}]}]},
    ],
)
def test_malformed_recordings_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ReplayFormatError):
        parse_replies(bad)


def test_files_that_are_not_recordings_are_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{nope")
    with pytest.raises(ReplayFormatError):
        ReplayProvider.from_file(broken)
    listing = tmp_path / "list.json"
    listing.write_text("[]")
    with pytest.raises(ReplayFormatError):
        ReplayProvider.from_file(listing)
