"""`ReplayProvider`: returns recorded replies in order. What CI, the tests and the default demo run
on.

The recorded format is owned here, so the recorder (17.3) and the replayer cannot drift:

    {"replies": [{"stop_reason": "tool_use", "text": "Looking that up.",
                  "tool_uses": [{"id": "t1", "name": "lookup", "input": {"q": "x"}}],
                  "usage": {"input_tokens": 120, "output_tokens": 18}}, ...]}

Only `stop_reason` is required, so the hand-written stub fixture from the scaffolding still loads.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from agent_container.core.errors import ReplayExhausted, ReplayFormatError
from agent_container.core.messages import (
    ModelRequest,
    ModelResponse,
    ToolUseBlock,
    Usage,
)
from agent_container.core.provider import ModelProvider


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ToolUse(_Strict):
    id: str
    name: str
    input: dict[str, JsonValue] = Field(default_factory=dict)


class _Usage(_Strict):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class _Reply(_Strict):
    stop_reason: str
    text: str = ""
    tool_uses: list[_ToolUse] = Field(default_factory=list)
    usage: _Usage = Field(default_factory=_Usage)


class _Recording(BaseModel):
    # `description` and similar notes are allowed in a fixture; only `replies` is read.
    model_config = ConfigDict(extra="ignore", frozen=True)
    replies: list[_Reply]


def parse_replies(data: Mapping[str, object]) -> list[ModelResponse]:
    """Turn a decoded recording into replies, or raise `ReplayFormatError`.

    An unrecognized stop reason is *not* rejected here: it is carried through so the loop, which
    owns the decision, is the one that raises.
    """
    try:
        recording = _Recording.model_validate(data)
    except ValidationError as exc:
        raise ReplayFormatError(f"not a replay recording: {exc.error_count()} problem(s)") from exc
    return [
        ModelResponse.of(
            r.text,
            stop_reason=r.stop_reason,
            tool_uses=[ToolUseBlock(u.id, u.name, u.input) for u in r.tool_uses],
            usage=Usage(r.usage.input_tokens, r.usage.output_tokens),
        )
        for r in recording.replies
    ]


def dump_replies(replies: Sequence[ModelResponse]) -> dict[str, JsonValue]:
    """The inverse of `parse_replies`, for the recorder."""
    out: list[JsonValue] = []
    for r in replies:
        out.append(
            {
                "stop_reason": r.stop_reason,
                "text": r.text,
                "tool_uses": [
                    {"id": u.id, "name": u.name, "input": dict(u.input)} for u in r.tool_uses
                ],
                "usage": {
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                },
            }
        )
    return {"replies": out}


class ReplayProvider(ModelProvider):
    """Hands back the given replies in order, one per call, and records the requests it received.

    Running out of replies raises `ReplayExhausted`: a loop that makes more calls than were recorded
    has diverged from the recording, and padding it with a default would hide that.
    """

    def __init__(self, replies: Sequence[ModelResponse]) -> None:
        self._replies = list(replies)
        self._next = 0
        self.requests: list[ModelRequest] = []

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ReplayProvider:
        return cls(parse_replies(data))

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayProvider:
        try:
            data = json.loads(Path(path).read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise ReplayFormatError(f"{path} is not valid JSON") from exc
        if not isinstance(data, dict):
            raise ReplayFormatError(f"{path} is not a replay recording")
        return cls.from_dict(data)

    @property
    def remaining(self) -> int:
        return len(self._replies) - self._next

    async def converse(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(
            0
        )  # a real provider yields; so does this, so concurrency bugs show here too
        if self._next >= len(self._replies):
            raise ReplayExhausted(
                f"the recording has {len(self._replies)} replies; call {self._next + 1} has none"
            )
        reply = self._replies[self._next]
        self._next += 1
        self.requests.append(request)
        return reply
