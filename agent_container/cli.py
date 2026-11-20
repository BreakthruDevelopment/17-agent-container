"""Command-line entry point. `demo` replays recorded replies; nothing here touches the network."""

import argparse
import json
import sys
from importlib import resources
from typing import Any

_KNOWN_STOP_REASONS = frozenset(
    {
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "guardrail_intervened",
        "content_filtered",
    }
)


def load_replies() -> list[dict[str, str]]:
    """Load the recorded replies and check each stop reason is one we handle."""
    raw: Any = json.loads(
        resources.files("agent_container").joinpath("fixtures/demo.json").read_text("utf-8")
    )
    replies: list[dict[str, str]] = raw["replies"]
    for reply in replies:
        if reply["stop_reason"] not in _KNOWN_STOP_REASONS:
            # An unrecognized stop reason raises rather than falling through (spec design rule).
            raise ValueError(f"unrecognized stop reason: {reply['stop_reason']!r}")
    return replies


def demo() -> int:
    """Print the recorded transcript, ending on the stop reason and not on the text."""
    for turn, reply in enumerate(load_replies(), start=1):
        print(f"turn {turn}: stop_reason={reply['stop_reason']} text={reply['text']!r}")
    print("demo ok (replay, no network)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-container")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="run the demo on recorded replies")
    demo_parser.add_argument("--live", action="store_true", help="use Bedrock (not built yet)")
    args = parser.parse_args(argv)
    if args.command == "demo":
        if args.live:
            print(
                "--live needs the Bedrock provider (slice 17.3); not available yet", file=sys.stderr
            )
            return 2
        return demo()
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
