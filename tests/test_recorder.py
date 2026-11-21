"""The reply recorder, round-tripped through `ReplayProvider`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agent_container.core import (
    AgentLoop,
    BedrockProvider,
    Budget,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    RecordingProvider,
    ReplayProvider,
    RunResult,
    ToolUseBlock,
    Usage,
)
from agent_container.testing import ModelProviderContract, OverlapProbe
from tests.support import EchoTool, ProbedReplay, StubConverseClient, tool_call

NONCE = "testnonce0123456"


def agent(provider: ModelProvider) -> AgentLoop:
    return AgentLoop(provider, [EchoTool()], system_prompt="p", fence_nonce=NONCE)


def comparable(r: RunResult) -> dict[str, object]:
    """Everything about a run except its random id."""
    return {f: getattr(r, f) for f in r.__dataclass_fields__ if f != "run_id"}


def stored(recorder: RecordingProvider) -> list[dict[str, Any]]:
    data: Any = recorder.to_dict()
    replies: list[dict[str, Any]] = data["replies"]
    return replies


class TestRecordingProviderContract(ModelProviderContract):
    def make_provider(
        self, replies: Sequence[ModelResponse], probe: OverlapProbe | None = None
    ) -> ModelProvider:
        return RecordingProvider(ProbedReplay(replies, probe))


SCRIPT = [
    ModelResponse.of(
        "Checking.",
        stop_reason="tool_use",
        tool_uses=[tool_call("ping", id="a"), ToolUseBlock("b", "echo", {"text": "pong"})],
        usage=Usage(100, 20),
    ),
    ModelResponse.of("All done.", usage=Usage(150, 8)),
]


async def test_a_recorded_run_replays_to_the_same_result(tmp_path: Path) -> None:
    # Record: a "live" run through the Bedrock provider (stubbed client) with the recorder on top.
    recorder = RecordingProvider(BedrockProvider("m", client=StubConverseClient(SCRIPT)))
    live = await agent(recorder).run("go", Budget())
    fixture = tmp_path / "run.json"
    recorder.save(fixture)

    # Replay: the file alone, no client.
    replay = ReplayProvider.from_file(fixture)
    replayed = await agent(replay).run("go", Budget())

    assert replay.remaining == 0
    assert comparable(replayed) == comparable(live)
    assert replayed.final_text == "All done."
    assert replayed.turns_used == 2
    assert (replayed.input_tokens, replayed.output_tokens) == (250, 28)


async def test_the_fixture_is_the_replay_format(tmp_path: Path) -> None:
    recorder = RecordingProvider(ReplayProvider(SCRIPT))
    await agent(recorder).run("go", Budget())
    data = recorder.to_dict()
    assert json.loads(json.dumps(data)) == data
    first = stored(recorder)[0]
    assert first["stop_reason"] == "tool_use"
    assert first["usage"] == {"input_tokens": 100, "output_tokens": 20}


async def test_an_unrecognized_stop_reason_is_recorded_as_said(tmp_path: Path) -> None:
    recorder = RecordingProvider(ReplayProvider([ModelResponse.of("x", stop_reason="new_one")]))
    await recorder.converse(ModelRequest("s", ()))
    assert stored(recorder)[0]["stop_reason"] == "new_one"


async def test_redaction_applies_to_the_stored_copy_only() -> None:
    recorder = RecordingProvider(
        ReplayProvider(
            [
                ModelResponse.of(
                    "mail ann@example.com",
                    stop_reason="tool_use",
                    tool_uses=[ToolUseBlock("a", "echo", {"text": "ann@example.com", "n": 1})],
                )
            ]
        ),
        redact=lambda s: s.replace("ann@example.com", "[email]"),
    )
    got = await recorder.converse(ModelRequest("s", ()))
    assert "ann@example.com" in got.text  # the live run is not altered
    (reply,) = stored(recorder)
    assert reply["text"] == "mail [email]"
    assert reply["tool_uses"][0]["input"] == {"text": "[email]", "n": 1}


def test_saving_an_empty_recording_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RecordingProvider(ReplayProvider([])).save(tmp_path / "x.json")
