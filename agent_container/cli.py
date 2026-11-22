"""Command-line entry point. `demo` replays recorded replies; `--live` switches to Bedrock."""

from __future__ import annotations

import argparse
import asyncio
import sys

from pydantic import ValidationError

from agent_container.core.errors import AgentContainerError
from agent_container.demo import (
    DEFAULT_MESSAGE,
    DemoOptions,
    execute_demo,
    print_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-container")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="run the demo on recorded replies")
    demo_parser.add_argument(
        "--live",
        action="store_true",
        help="use Bedrock instead of recorded replies (needs credentials; costs money)",
    )
    demo_parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="user message (max 8000 characters)",
    )
    demo_parser.add_argument(
        "--model",
        default=None,
        help="Bedrock model / inference profile id (default: the demo manifest)",
    )
    demo_parser.add_argument(
        "--region",
        default=None,
        help="AWS region for --live (default: us-east-2)",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> DemoOptions:
    return DemoOptions.model_validate(
        {
            "live": bool(args.live),
            "message": args.message,
            "model": args.model,
            "region": args.region,
        }
    )


def demo(options: DemoOptions) -> int:
    """Run the demo and print a summary. 0 on success; 1 on a run failure; 2 on bad input."""
    try:
        result, reasons = asyncio.run(execute_demo(options, log_stream=sys.stderr))
    except AgentContainerError as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"demo failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print_run(result, live=options.live, stop_reasons=reasons)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "demo":
        return 2  # pragma: no cover
    try:
        options = _options_from_args(args)
    except ValidationError:
        print("invalid demo arguments", file=sys.stderr)
        return 2
    return demo(options)


if __name__ == "__main__":
    sys.exit(main())
