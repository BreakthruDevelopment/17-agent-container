"""The `agent-container demo` CLI: replay by default, Bedrock only with `--live`."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_container import __version__, cli
from agent_container.core import bedrock as bedrock_module
from agent_container.core.loop import MAX_USER_MESSAGE_CHARS
from agent_container.demo import DEFAULT_REGION, load_demo_replies
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
    out = capsys.readouterr().out
    assert "stop_reason=tool_use" in out
    assert "stop_reason=end_turn" in out
    assert "demo ok (replay, no network)" in out
    assert "turns_used=3" in out


def test_replay_does_not_build_a_bedrock_client() -> None:
    # The autouse fixture would fail this test if demo constructed a real client.
    assert cli.main(["demo"]) == 0


def test_live_uses_bedrock_with_a_stubbed_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client, seen = _stub_live(monkeypatch)
    assert cli.main(["demo", "--live"]) == 0
    captured = capsys.readouterr()
    assert "demo ok (live)" in captured.out
    assert "stop_reason=end_turn" in captured.out
    assert client.requests
    assert client.requests[0]["modelId"] == "global.anthropic.claude-sonnet-4-6"
    assert seen["region_name"] == DEFAULT_REGION


def test_live_region_and_model_are_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client, seen = _stub_live(monkeypatch)
    assert cli.main(["demo", "--live", "--model", "test.model", "--region", "eu-west-1"]) == 0
    assert "demo ok (live)" in capsys.readouterr().out
    assert client.requests[0]["modelId"] == "test.model"
    assert seen["region_name"] == "eu-west-1"


def test_live_without_a_client_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--live"]) == 1
    captured = capsys.readouterr()
    assert "demo ok" not in captured.out
    assert "demo failed: AssertionError" in captured.err


def test_a_too_long_message_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--message", "x" * (MAX_USER_MESSAGE_CHARS + 1)]) == 2
    captured = capsys.readouterr()
    assert "invalid demo arguments" in captured.err
    assert "x" * 20 not in captured.err


def test_a_blank_message_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--message", "   "]) == 2
    assert "invalid demo arguments" in capsys.readouterr().err


def test_json_lines_go_to_stderr_not_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("turn ")
    assert "demo ok" in captured.out
    rows = [json.loads(line) for line in captured.err.splitlines() if line.startswith("{")]
    assert rows
    assert {row["outcome"] for row in rows} >= {"model_called", "tool_called", "tool_failed"}
    for row in rows:
        assert "run_id" in row
        assert "query" not in row
        assert DEFAULT_REGION not in json.dumps(row)


def test_help_lists_live(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["demo", "--help"])
    assert caught.value.code == 0
    assert "--live" in capsys.readouterr().out
