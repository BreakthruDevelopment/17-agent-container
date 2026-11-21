"""`RecordingProvider`: wraps a provider and writes what it said as a replay fixture.

The file format is `replay.py`'s (`dump_replies` / `parse_replies`), so a recording is consumed by
`ReplayProvider.from_file` with nothing in between and the two cannot drift.

Record one run at a time. Replies are stored in the order they *complete*, and `ReplayProvider`
hands them back in that order, so a recording of concurrent calls would replay in an arbitrary
order. Content is normalised to the replay shape: the text blocks of a reply are joined and its tool
calls follow.

Fixtures are committed, so `redact` is applied to every string in a reply (its text and the string
values inside tool inputs) before it is stored. The spec routes this through 17.4's `Redacting`;
until that lands, pass a function. A fixture still needs a person's review before commit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import JsonValue

from agent_container.core.messages import (
    JsonObject,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolUseBlock,
)
from agent_container.core.provider import ModelProvider
from agent_container.core.replay import dump_replies


def _redact_json(value: JsonValue, redact: Callable[[str], str]) -> JsonValue:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_json(v, redact) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_json(v, redact) for v in value]
    return value


class RecordingProvider(ModelProvider):
    """Passes every call through and keeps the reply."""

    def __init__(self, inner: ModelProvider, *, redact: Callable[[str], str] | None = None) -> None:
        self._inner = inner
        self._redact = redact
        self.replies: list[ModelResponse] = []

    async def converse(self, request: ModelRequest) -> ModelResponse:
        response = await self._inner.converse(request)
        self.replies.append(self._scrub(response))
        return response  # the caller gets the real reply; only the stored copy is redacted

    def _scrub(self, response: ModelResponse) -> ModelResponse:
        redact = self._redact
        if redact is None:
            return response
        blocks = tuple(
            TextBlock(redact(b.text))
            if isinstance(b, TextBlock)
            else ToolUseBlock(b.id, b.name, _redact_object(b.input, redact))
            if isinstance(b, ToolUseBlock)
            else b
            for b in response.content
        )
        return ModelResponse(blocks, response.stop_reason, response.usage)

    def to_dict(self) -> dict[str, JsonValue]:
        return dump_replies(self.replies)

    def save(self, path: str | Path) -> None:
        """Write the recording as JSON. Raises if there is nothing to write."""
        if not self.replies:
            raise ValueError("nothing was recorded")
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", "utf-8")


def _redact_object(value: JsonObject, redact: Callable[[str], str]) -> dict[str, JsonValue]:
    return {k: _redact_json(v, redact) for k, v in value.items()}


__all__ = ["RecordingProvider"]
