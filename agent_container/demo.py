"""The command-line demo: a canned lookup agent on recorded replies, or Bedrock with `--live`.

The library ships no business tools; `LookupTool` exists only so this demo has something to
call. Replay is the default so CI and a laptop demo need no network.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from importlib import resources
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_container.core.bedrock import BedrockProvider
from agent_container.core.builder import AgentBuilder, ManifestError
from agent_container.core.decorators import Retrying
from agent_container.core.errors import ReplayFormatError
from agent_container.core.limiter import RateLimitedProvider
from agent_container.core.loop import (
    MAX_USER_MESSAGE_CHARS,
    LoopEvent,
    LoopEventKind,
    RunResult,
)
from agent_container.core.messages import ModelResponse
from agent_container.core.provider import ModelProvider
from agent_container.core.replay import ReplayProvider, parse_replies
from agent_container.core.tool import Effect, ErrorCategory, Tool, ToolContext, ToolResult

DEFAULT_MESSAGE = "What is the demo status?"
DEFAULT_REGION = "us-east-2"
MODEL_ENV = "AGENT_CONTAINER_MODEL"


def _package_text(name: str) -> str:
    return resources.files("agent_container").joinpath("fixtures", name).read_text(encoding="utf-8")


def demo_manifest_text() -> str:
    return _package_text("demo.yaml")


def demo_recording() -> dict[str, object]:
    raw: object = json.loads(_package_text("demo.json"))
    if not isinstance(raw, dict):
        raise ReplayFormatError("demo fixture is not a replay recording")
    return raw


def load_demo_replies() -> list[ModelResponse]:
    return parse_replies(demo_recording())


class DemoOptions(BaseModel):
    """CLI arguments for `demo`, validated at the boundary."""

    model_config = ConfigDict(extra="forbid")

    live: bool = False
    message: str = Field(default=DEFAULT_MESSAGE, min_length=1, max_length=MAX_USER_MESSAGE_CHARS)
    model: str | None = None
    region: str | None = None

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("model", "region")
    @classmethod
    def optional_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


class LookupArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class LookupTool(Tool[LookupArgs]):
    """A canned lookup. The first three calls fail transiently so Retrying has work to do."""

    name = "lookup"
    description = "Look up a canned demo fact. Use query 'status' for the demo."
    effect = Effect.READ
    input_model = LookupArgs

    # Fail this many times, then succeed. Matches Retrying's default max_attempts: the first
    # model-visible call is retried then reported as a transient error; the second succeeds.
    _FAILURES = 3

    def __init__(self) -> None:
        self.attempts = 0

    async def run(self, args: LookupArgs, ctx: ToolContext) -> ToolResult:
        self.attempts += 1
        if self.attempts <= self._FAILURES:
            return ToolResult.error(
                ErrorCategory.TRANSIENT,
                "lookup is temporarily unavailable",
            )
        return ToolResult.ok(f"status: ok (query={args.query})")


class _StopReasons:
    """Collect each model stop reason so the CLI can print it. Does not change the run."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    async def on_event(self, event: LoopEvent) -> None:
        if event.kind is LoopEventKind.MODEL_CALLED and event.detail:
            self.reasons.append(event.detail)


def resolve_region(options: DemoOptions) -> str:
    return (options.region or os.environ.get("AWS_REGION") or DEFAULT_REGION).strip()


def resolve_model(options: DemoOptions, manifest_model: str | None) -> str:
    chosen = options.model or os.environ.get(MODEL_ENV) or manifest_model
    if chosen is None or not chosen.strip():
        raise ManifestError("a model id is required")
    return chosen


def make_provider(*, live: bool, model: str, region: str) -> ModelProvider:
    """Replay by default; Bedrock behind the limiter only when `live` is true.

    The inference profile id is `model`, passed in from configuration (manifest, --model, or
    env). Nothing here hard-codes it.
    """
    if live:
        return RateLimitedProvider(BedrockProvider(model, region_name=region))
    return ReplayProvider(load_demo_replies())


def lookup_stack() -> Retrying[LookupArgs]:
    """The demo tool: Retrying around lookup, delay zero so CI does not wait."""
    return Retrying(
        LookupTool(),
        max_attempts=LookupTool._FAILURES,
        base_delay=0.0,
        max_delay=0.0,
    )


async def execute_demo(
    options: DemoOptions,
    *,
    provider: ModelProvider | None = None,
    log_stream: TextIO | None = None,
) -> tuple[RunResult, tuple[str, ...]]:
    """Build the demo agent and run it. `provider` is for tests; the CLI leaves it unset.

    `log_stream` is the JSON-lines sink. The CLI passes `sys.stderr` at call time so a test
    harness that replaces stderr still captures the lines. `None` skips the log.
    """
    builder = AgentBuilder.from_yaml(demo_manifest_text())
    if options.model:
        builder.with_model(options.model)
    elif (from_env := os.environ.get(MODEL_ENV)) is not None and from_env.strip():
        builder.with_model(from_env)
    model_id = resolve_model(options, builder.model)
    region = resolve_region(options)
    if provider is None:
        provider = make_provider(live=options.live, model=model_id, region=region)
    reasons = _StopReasons()
    if log_stream is not None:
        builder.with_json_log(log_stream)
    loop = builder.with_provider(provider).with_tool(lookup_stack()).with_observer(reasons).build()
    result = await loop.run(options.message, builder.budget())
    return result, tuple(reasons.reasons)


def print_run(
    result: RunResult,
    *,
    live: bool,
    stop_reasons: Sequence[str],
    stream: TextIO | None = None,
) -> None:
    """Human summary on stdout. CI greps `stop_reason=end_turn` and `demo ok` here."""
    out = sys.stdout if stream is None else stream
    reasons = list(stop_reasons) or ([result.stop_reason] if result.stop_reason else [])
    for turn, reason in enumerate(reasons, start=1):
        print(f"turn {turn}: stop_reason={reason}", file=out)
    print(f"end_reason={result.end_reason}", file=out)
    print(f"turns_used={result.turns_used}", file=out)
    print(f"tokens_in={result.input_tokens} tokens_out={result.output_tokens}", file=out)
    print(f"tool_calls={len(result.tool_calls)}", file=out)
    if result.staged_changes:
        print(f"staged_changes={len(result.staged_changes)}", file=out)
    if live:
        print("demo ok (live)", file=out)
    else:
        print("demo ok (replay, no network)", file=out)


__all__ = [
    "DEFAULT_MESSAGE",
    "DEFAULT_REGION",
    "MODEL_ENV",
    "DemoOptions",
    "LookupArgs",
    "LookupTool",
    "demo_manifest_text",
    "demo_recording",
    "execute_demo",
    "load_demo_replies",
    "lookup_stack",
    "make_provider",
    "print_run",
    "resolve_model",
    "resolve_region",
]
