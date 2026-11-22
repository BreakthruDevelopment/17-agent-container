"""The command-line demo: a canned lookup agent on recorded replies, or Bedrock with `--live`.

The library ships no business tools; `LookupTool` exists only so this demo has something to
call. Replay is the default so CI and a laptop demo need no network.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_container.core.bedrock import BedrockProvider
from agent_container.core.builder import AgentBuilder, ManifestError
from agent_container.core.decorators import Retrying
from agent_container.core.errors import ReplayFormatError
from agent_container.core.gate import PermissionGate
from agent_container.core.limiter import RateLimitedProvider
from agent_container.core.loop import (
    MAX_USER_MESSAGE_CHARS,
    EndReason,
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
WRITE_PERMISSION = "status.write"  # also declared under `permissions:` in fixtures/demo.yaml

# `lookup` fails this many times, then succeeds. Equal to Retrying's default max_attempts: the
# first model-visible call is retried, then reported as a transient error; the second succeeds.
LOOKUP_FAILURES = 3


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

    def __init__(self) -> None:
        self.attempts = 0

    async def run(self, args: LookupArgs, ctx: ToolContext) -> ToolResult:
        self.attempts += 1
        if self.attempts <= LOOKUP_FAILURES:
            return ToolResult.error(
                ErrorCategory.TRANSIENT,
                "lookup is temporarily unavailable",
            )
        return ToolResult.ok(f"status: ok (query={args.query})")


class RecordStatusArgs(BaseModel):
    status: str = Field(min_length=1, max_length=100)


class RecordStatusTool(Tool[RecordStatusArgs]):
    """A demo write. It only runs when a person approves the staged change."""

    name = "record_status"
    description = "Record the demo status. A write: it is staged for approval, not run at once."
    effect = Effect.WRITE
    input_model = RecordStatusArgs

    def __init__(self) -> None:
        self.recorded: list[str] = []

    async def run(self, args: RecordStatusArgs, ctx: ToolContext) -> ToolResult:
        self.recorded.append(args.status)
        return ToolResult.ok(f"recorded status: {args.status}")


def _run_succeeded(result: RunResult) -> bool:
    """Whether the run finished on its own rather than on a limit or a fault.

    One predicate, shared by `DemoOutcome.succeeded` and by the gate that decides whether to
    apply staged writes. They were separate and the apply ran unconditionally, so a truncated
    run printed `approved: record_status: ...` and then `demo failed`, having already written.
    If the two drift apart that defect comes straight back.

    `==`, not `is`: `EndReason` is a `StrEnum`, and an equal plain string is a different object.
    This field is set by the loop and never read off the wire, so `is` happened to work here --
    but `17-agent-container#11` is the same comparison in `ToolResult.failure`, where the value
    DOES come from the wire and `is` silently returns the wrong answer.
    """
    return result.end_reason == EndReason.END_TURN


@dataclass(frozen=True, slots=True)
class DemoOutcome:
    """What a demo run produced: the loop's result, each stop reason, and the approved writes."""

    result: RunResult
    stop_reasons: tuple[str, ...]
    approved: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        """The run ended because the model finished, not because a limit or a fault stopped it."""
        return _run_succeeded(self.result)


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
        max_attempts=LOOKUP_FAILURES,
        base_delay=0.0,
        max_delay=0.0,
    )


def status_stack() -> PermissionGate[RecordStatusArgs]:
    """The demo write behind its gate: staged for approval, for a caller holding the permission."""
    return PermissionGate(RecordStatusTool(), requires=WRITE_PERMISSION)


async def execute_demo(
    options: DemoOptions,
    *,
    provider: ModelProvider | None = None,
    log_stream: TextIO | None = None,
    permissions: Sequence[str] | None = None,
) -> DemoOutcome:
    """Build the demo agent, run it, then approve what it staged. `provider` is for tests.

    `log_stream` is the JSON-lines sink: the CLI passes `sys.stdout` at call time (not as a
    default argument) so a harness that replaces stdout still captures the lines. `None` skips
    the log. `permissions`, when given, replaces the manifest's, so a test can run the same
    agent as a caller without `status.write` and watch the gate refuse.

    The approval is automatic here and says so: in a real deployment a person reads each
    staged change's description and decides. Nothing staged runs until this point.
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
    if permissions is not None:
        builder.with_permissions(permissions)
    builder.with_provider(provider).with_tool(lookup_stack()).with_tool(status_stack())
    loop = builder.with_observer(reasons).build()
    result = await loop.run(options.message, builder.budget())
    approved: list[str] = []
    # Apply staged writes ONLY when the run actually succeeded. A staged change exists so that a
    # person decides; a run that ended on a limit or a fault is not an approval.
    if _run_succeeded(result):
        for change in result.staged_changes:
            await change.apply()
            approved.append(change.description)
    return DemoOutcome(result, tuple(reasons.reasons), tuple(approved))


def print_run(outcome: DemoOutcome, *, live: bool, stream: TextIO | None = None) -> None:
    """The human summary, on stderr. Stdout is reserved for the JSON-lines event stream.

    `demo ok` is printed only when the run succeeded; the CLI's exit code says the same.

    The model's final answer is part of the summary: without it `demo --live --message ...`
    reports metrics and stop reasons and never shows what the agent actually said.
    """
    out = sys.stderr if stream is None else stream
    result = outcome.result
    reasons = list(outcome.stop_reasons) or ([result.stop_reason] if result.stop_reason else [])
    for turn, reason in enumerate(reasons, start=1):
        print(f"turn {turn}: stop_reason={reason}", file=out)
    if result.final_text.strip():
        # The model's own words, indented under a header. Indented because every other line of
        # this summary is a fixed prefix at column 0 that CI greps for (`^demo ok`,
        # `^staged_changes=1`): model text printed flush left could forge one and make a failed
        # run look green. Tool results are data, and so is the final answer.
        print("answer:", file=out)
        for line in result.final_text.splitlines():
            print(f"  {line}", file=out)
    print(f"end_reason={result.end_reason}", file=out)
    print(f"turns_used={result.turns_used}", file=out)
    print(f"tokens_in={result.input_tokens} tokens_out={result.output_tokens}", file=out)
    print(f"tool_calls={len(result.tool_calls)}", file=out)
    if result.staged_changes:
        print(f"staged_changes={len(result.staged_changes)}", file=out)
    for description in outcome.approved:
        print(f"approved: {description}", file=out)
    if not outcome.succeeded:
        print(f"demo failed: the run ended with end_reason={result.end_reason}", file=out)
    elif live:
        print("demo ok (live)", file=out)
    else:
        print("demo ok (replay, no network)", file=out)


EVENT_FIELDS = frozenset({"run_id", "turn", "tool", "outcome", "tokens", "duration"})
REQUIRED_OUTCOMES = frozenset({"model_called", "tool_called"})


def validate_event_stream(text: str) -> None:
    """Raise ValueError unless `text` is a well-formed JSON-lines event stream.

    Every non-blank line must be a JSON object with exactly the six event fields, and the stream
    must show the loop working (a model call and a tool call). An empty stream fails: a dead
    event bus is exactly what the CI smoke step must catch.
    """
    outcomes: set[str] = set()
    count = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row: object = json.loads(line)
        except ValueError:
            raise ValueError(f"line {number} of the event stream is not JSON") from None
        if not isinstance(row, dict) or set(row) != EVENT_FIELDS:
            raise ValueError(f"line {number} of the event stream is not an event object")
        outcomes.add(str(row["outcome"]))
        count += 1
    if count == 0:
        raise ValueError("the event stream is empty: no events were logged")
    if missing := REQUIRED_OUTCOMES - outcomes:
        raise ValueError(f"the event stream never shows {sorted(missing)}")


__all__ = [
    "DEFAULT_MESSAGE",
    "DEFAULT_REGION",
    "LOOKUP_FAILURES",
    "MODEL_ENV",
    "WRITE_PERMISSION",
    "DemoOptions",
    "DemoOutcome",
    "LookupArgs",
    "LookupTool",
    "RecordStatusArgs",
    "RecordStatusTool",
    "demo_manifest_text",
    "demo_recording",
    "execute_demo",
    "load_demo_replies",
    "lookup_stack",
    "make_provider",
    "print_run",
    "resolve_model",
    "resolve_region",
    "status_stack",
    "validate_event_stream",
]
