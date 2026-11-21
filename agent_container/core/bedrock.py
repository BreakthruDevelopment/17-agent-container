"""`BedrockProvider`: the Bedrock Converse API behind the `ModelProvider` contract.

This module is the only place that knows the Converse wire format and the only one that touches
boto3. Everything the loop sees is `ModelRequest` / `ModelResponse`.

boto3 clients are synchronous, so each call runs on a worker thread (`asyncio.to_thread`). That
keeps `converse` awaitable concurrently, which app 20's fan-out needs. A cancelled await (the
loop's wall clock) cannot stop a thread already inside the HTTP call, so the default client is
built with explicit connect and read timeouts and no botocore retries: a stuck call ends on
its own, and throttling is the limiter's job (`RateLimitedProvider`), not a hidden retry loop.

Credentials come from boto3's default chain. Nothing here reads or stores a key.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import JsonValue

from agent_container.core.errors import MalformedModelResponse
from agent_container.core.messages import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolChoice,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from agent_container.core.provider import ModelProvider

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_READ_TIMEOUT_SECONDS = 60


class ConverseClient(Protocol):
    """The one method of a boto3 `bedrock-runtime` client this module uses. Tests stub it."""

    def converse(self, **kwargs: Any) -> Mapping[str, Any]: ...


def make_client(region_name: str | None = None) -> ConverseClient:
    """A `bedrock-runtime` client on the default credential chain, with bounded timeouts."""
    import boto3
    from botocore.config import Config

    config = Config(
        connect_timeout=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout=DEFAULT_READ_TIMEOUT_SECONDS,
        retries={"max_attempts": 1, "mode": "standard"},
    )
    client: ConverseClient = boto3.client("bedrock-runtime", region_name=region_name, config=config)
    return client


# -- request: our types to the wire ---------------------------------------------------------------


def _json(value: JsonValue) -> JsonValue:
    """A deep copy as plain dicts and lists, so the wire request shares nothing with ours."""
    if isinstance(value, Mapping):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    return value


def _block(block: ContentBlock) -> dict[str, Any] | None:
    match block:
        case TextBlock(text):
            # Converse rejects a blank text block; there is nothing to say, so say nothing.
            return {"text": text} if text.strip() else None
        case ToolUseBlock(id, name, tool_input):
            return {"toolUse": {"toolUseId": id, "name": name, "input": _json(dict(tool_input))}}
        case ToolResultBlock(tool_use_id, content, is_error):
            return {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": content}],
                    "status": "error" if is_error else "success",
                }
            }


def _message(message: Message) -> dict[str, Any]:
    content = [b for b in (_block(b) for b in message.content) if b is not None]
    if not content:
        raise ValueError(f"a {message.role} message has no content Bedrock would accept")
    return {"role": message.role, "content": content}


def _tool_spec(spec: ToolSpec) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": {"json": _json(dict(spec.input_schema))},
        }
    }


def _tool_choice(choice: ToolChoice) -> dict[str, Any]:
    if choice.mode == "tool":
        return {"tool": {"name": choice.name}}
    return {choice.mode: {}}


def build_request(model_id: str, request: ModelRequest) -> dict[str, Any]:
    """The keyword arguments for `client.converse`. Builds new objects; never edits `request`."""
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": [_message(m) for m in request.messages],
    }
    if request.system.strip():
        kwargs["system"] = [{"text": request.system}]
    if request.max_tokens is not None:
        kwargs["inferenceConfig"] = {"maxTokens": request.max_tokens}
    if request.tools:
        tool_config: dict[str, Any] = {"tools": [_tool_spec(t) for t in request.tools]}
        if request.tool_choice is not None:
            tool_config["toolChoice"] = _tool_choice(request.tool_choice)
        kwargs["toolConfig"] = tool_config
    return kwargs


# -- response: the wire to our types --------------------------------------------------------------


def _count(usage: Mapping[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MalformedModelResponse(f"Bedrock reported {key} as {value!r}")
    return value


def parse_response(wire: Mapping[str, Any]) -> ModelResponse:
    """A Converse response as a `ModelResponse`.

    `stopReason` is passed through exactly as Bedrock sent it. Content blocks this library has no
    type for (reasoning, guard content) are left out of `content`; they carry nothing the loop acts
    on.
    """
    stop_reason = wire.get("stopReason")
    if not isinstance(stop_reason, str):
        raise MalformedModelResponse("the Bedrock response has no stopReason")
    message = wire.get("output", {}).get("message", {})
    blocks: list[ContentBlock] = []
    for raw in message.get("content", ()):
        if "text" in raw:
            blocks.append(TextBlock(str(raw["text"])))
        elif "toolUse" in raw:
            use = raw["toolUse"]
            tool_input = use.get("input", {})
            if not isinstance(tool_input, dict):
                raise MalformedModelResponse("a toolUse block's input is not an object")
            blocks.append(ToolUseBlock(str(use["toolUseId"]), str(use["name"]), tool_input))
    usage = wire.get("usage", {})
    return ModelResponse(
        tuple(blocks),
        stop_reason,
        Usage(_count(usage, "inputTokens"), _count(usage, "outputTokens")),
    )


class BedrockProvider(ModelProvider):
    """Calls Bedrock's Converse API. Pass `client` to stub it; otherwise one is made lazily."""

    def __init__(
        self,
        model_id: str,
        *,
        client: ConverseClient | None = None,
        region_name: str | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("a model id is required")
        self._model_id = model_id
        self._region_name = region_name
        self._client = client

    def _get_client(self) -> ConverseClient:
        # Built on first use, not at construction, so importing or configuring this class needs
        # neither boto3 credentials nor the network. A race between two first calls makes two
        # clients and keeps one, which is harmless.
        if self._client is None:
            self._client = make_client(self._region_name)
        return self._client

    async def converse(self, request: ModelRequest) -> ModelResponse:
        kwargs = build_request(self._model_id, request)
        client = self._get_client()
        wire = await asyncio.to_thread(client.converse, **kwargs)
        return parse_response(wire)


__all__ = [
    "BedrockProvider",
    "ConverseClient",
    "build_request",
    "make_client",
    "parse_response",
]
