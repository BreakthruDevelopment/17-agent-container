"""The conversation types shared by the loop and every provider.

They follow the shape of the Bedrock Converse API (roles, content blocks, a stop reason, token
usage) without depending on it, so `BedrockProvider` (17.3) translates at its edge and nothing else
knows boto3 exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import JsonValue

from agent_container.core.errors import UnrecognizedStopReason

JsonObject = Mapping[str, JsonValue]

Role = Literal["user", "assistant"]


class StopReason(StrEnum):
    """Every Bedrock Converse `stopReason` the loop has a decision for.

    The first six are the ones the spec lists; the last three are the newer Converse values
    (`malformed_model_output`, `malformed_tool_use`, `model_context_window_exceeded`), added in 17.3
    with a branch each in the loop. The list is closed on purpose: a value that is not here must be
    decided on, in this enum and in the loop, never defaulted, so `parse` raises on it.
    """

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    CONTENT_FILTERED = "content_filtered"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    MALFORMED_TOOL_USE = "malformed_tool_use"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"

    @classmethod
    def parse(cls, value: str) -> StopReason:
        """Return the stop reason for a wire value, or raise `UnrecognizedStopReason`."""
        try:
            return cls(value)
        except ValueError:
            raise UnrecognizedStopReason(value) from None


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """The model asking for a tool. `id` pairs it with the `ToolResultBlock` that answers it."""

    id: str
    name: str
    input: JsonObject


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """A tool's answer going back to the model. `content` is already fenced by the loop."""

    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: tuple[ContentBlock, ...]

    @classmethod
    def user_text(cls, text: str) -> Message:
        return cls("user", (TextBlock(text),))


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the model is told about a tool: name, description and a JSON Schema."""

    name: str
    description: str
    input_schema: JsonObject


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """Constrains tool selection (exam 2.3). `None` on a request means the model decides (auto)."""

    mode: Literal["auto", "any", "tool"] = "auto"
    name: str | None = None

    def __post_init__(self) -> None:
        if (self.mode == "tool") != (self.name is not None):
            raise ValueError("a tool choice names a tool if and only if its mode is 'tool'")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    max_tokens: int | None = None
    tool_choice: ToolChoice | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One model reply.

    `stop_reason` is the raw wire string on purpose. A provider passes it through untouched and the
    loop is the single place that decides what each value means, so a provider cannot quietly map an
    unknown value onto a known one. Use `StopReason.parse` to interpret it.
    """

    content: tuple[ContentBlock, ...]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))

    @classmethod
    def of(
        cls,
        text: str = "",
        *,
        stop_reason: str = "end_turn",
        tool_uses: Sequence[ToolUseBlock] = (),
        usage: Usage | None = None,
    ) -> ModelResponse:
        blocks: list[ContentBlock] = [TextBlock(text)] if text else []
        blocks.extend(tool_uses)
        return cls(tuple(blocks), stop_reason, usage or Usage())
