import pytest

from agent_container import __version__, cli


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_replies_load_and_end_on_end_turn() -> None:
    replies = cli.load_replies()
    assert replies[-1]["stop_reason"] == "end_turn"


def test_demo_runs_on_replay(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "stop_reason=tool_use" in out
    assert "demo ok" in out


def test_live_is_refused_without_network(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["demo", "--live"]) == 2
    assert "not available yet" in capsys.readouterr().err
