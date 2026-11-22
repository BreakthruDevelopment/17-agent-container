"""The `agent-container demo` CLI: replay by default, Bedrock only with `--live`."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, cast

import pytest

from agent_container import __version__, cli
from agent_container import demo as demo_module
from agent_container.core import bedrock as bedrock_module
from agent_container.core.builder import AgentBuilder
from agent_container.core.loop import MAX_USER_MESSAGE_CHARS
from agent_container.core.replay import parse_replies
from agent_container.demo import (
    DEFAULT_REGION,
    demo_recording,
    load_demo_replies,
    validate_event_stream,
)
from tests.support import StubConverseClient


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLI test that reaches for the real boto3 client fails instead of reaching AWS."""

    def refuse(region_name: str | None = None) -> None:
        raise AssertionError("a test tried to build a real Bedrock client")

    monkeypatch.setattr(bedrock_module, "make_client", refuse)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AGENT_CONTAINER_MODEL", raising=False)


def _stub_live(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[StubConverseClient, dict[str, Any]]:
    client = StubConverseClient(load_demo_replies())
    seen: dict[str, Any] = {}

    def fake_make_client(region_name: str | None = None) -> StubConverseClient:
        seen["region_name"] = region_name
        return client

    monkeypatch.setattr(bedrock_module, "make_client", fake_make_client)
    return client, seen


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_demo_runs_on_replay(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo"]) == 0
    err = capsys.readouterr().err
    assert "stop_reason=tool_use" in err
    assert "stop_reason=end_turn" in err
    assert "demo ok (replay, no network)" in err
    assert "turns_used=4" in err


def test_replay_does_not_build_a_bedrock_client() -> None:
    # The autouse fixture would fail this test if demo constructed a real client.
    assert cli.main(["demo"]) == 0


def test_live_uses_bedrock_with_a_stubbed_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client, seen = _stub_live(monkeypatch)
    assert cli.main(["demo", "--live"]) == 0
    captured = capsys.readouterr()
    assert "demo ok (live)" in captured.err
    assert "stop_reason=end_turn" in captured.err
    assert client.requests
    assert client.requests[0]["modelId"] == "global.anthropic.claude-sonnet-4-6"
    assert seen["region_name"] == DEFAULT_REGION


def test_live_region_and_model_are_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client, seen = _stub_live(monkeypatch)
    assert cli.main(["demo", "--live", "--model", "test.model", "--region", "eu-west-1"]) == 0
    assert "demo ok (live)" in capsys.readouterr().err
    assert client.requests[0]["modelId"] == "test.model"
    assert seen["region_name"] == "eu-west-1"


def test_live_without_a_client_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--live"]) == 1
    captured = capsys.readouterr()
    assert "demo ok" not in captured.out
    assert "demo ok" not in captured.err
    assert "demo failed: AssertionError" in captured.err


def test_a_too_long_message_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--message", "x" * (MAX_USER_MESSAGE_CHARS + 1)]) == 2
    captured = capsys.readouterr()
    assert "invalid demo arguments" in captured.err
    assert "x" * 20 not in captured.err


def test_a_blank_message_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--message", "   "]) == 2
    assert "invalid demo arguments" in capsys.readouterr().err


def _stdout_rows(out: str) -> list[dict[str, Any]]:
    """Every line of the data stream must parse: no filtering, so an interleaved line fails."""
    lines = out.splitlines()
    assert lines
    return [json.loads(line) for line in lines]


def test_json_lines_go_to_stdout_and_prose_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo"]) == 0
    captured = capsys.readouterr()
    rows = _stdout_rows(captured.out)
    assert {row["outcome"] for row in rows} >= {"model_called", "tool_called", "tool_failed"}
    assert "record_status" in {row["tool"] for row in rows}
    for row in rows:
        assert "run_id" in row
        assert "query" not in row
        assert DEFAULT_REGION not in json.dumps(row)
    assert captured.err.startswith("turn ")
    assert "demo ok (replay, no network)" in captured.err
    for line in captured.err.splitlines():
        assert not line.startswith("{")


def test_the_gate_stages_the_write_and_it_is_approved(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo"]) == 0
    err = capsys.readouterr().err
    assert "staged_changes=1" in err
    assert "approved: record_status:" in err
    assert '"status":"ok"' in err


def test_a_misbehaving_observer_cannot_corrupt_stdout() -> None:
    """Real streams, real process: the bus's warning reaches stderr, stdout stays all JSON."""
    code = (
        "import sys\n"
        "from agent_container import cli, demo\n"
        "async def boom(self, event):\n"
        "    raise RuntimeError('observer bug')\n"
        "demo._StopReasons.on_event = boom\n"
        "sys.exit(cli.main(['demo']))\n"
    )
    done = subprocess.run(  # noqa: S603 - fixed argv, no untrusted input
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert len(_stdout_rows(done.stdout)) >= 8
    assert "observer _StopReasons failed" in done.stderr


def test_a_run_that_does_not_end_on_end_turn_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = demo_recording()
    replies = cast(list[dict[str, Any]], recording["replies"])
    replies[-1]["stop_reason"] = "max_tokens"
    monkeypatch.setattr(demo_module, "load_demo_replies", lambda: parse_replies(recording))
    assert cli.main(["demo"]) == 1
    err = capsys.readouterr().err
    assert "end_reason=truncated" in err
    assert "demo ok" not in err
    assert "demo failed" in err


def test_the_smoke_check_fails_when_no_events_are_logged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo"]) == 0
    validate_event_stream(capsys.readouterr().out)  # the real stream passes
    monkeypatch.setattr(AgentBuilder, "with_json_log", lambda self, stream=None: self)
    assert cli.main(["demo"]) == 0  # the CLI cannot tell; the smoke check must
    with pytest.raises(ValueError, match="empty"):
        validate_event_stream(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "empty"),
        ("not json\n", "not JSON"),
        ('{"a":1}\n', "not an event"),
        (
            '{"run_id":"r","turn":1,"tool":null,"outcome":"model_called","tokens":1,"duration":0}\n',
            "tool_called",
        ),
    ],
)
def test_validate_event_stream_rejects(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_event_stream(text)


def test_help_lists_live(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["demo", "--help"])
    assert caught.value.code == 0
    assert "--live" in capsys.readouterr().out
