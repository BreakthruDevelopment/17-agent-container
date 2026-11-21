"""Assemble a configured agent from a YAML manifest, step by step (Builder, not Factory).

The product is always an `AgentLoop`. Stages fill in the model, system prompt, tools,
limits and permissions; `build()` walks each tool stack and refuses an `effect: write`
tool that has no `PermissionGate` anywhere in it. A consumer registers tools by passing
instances; the manifest names which of those the agent may call. The library ships no
business tools of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Self, TextIO

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_container.core.budget import Budget
from agent_container.core.errors import AgentContainerError
from agent_container.core.events import EventBus
from agent_container.core.log import JsonLinesSubscriber
from agent_container.core.loop import AgentLoop, LoopObserver
from agent_container.core.provider import ModelProvider
from agent_container.core.tool import (
    TOOL_NAME_PATTERN,
    AnyTool,
    Identity,
    has_write_gate,
    innermost,
    writes,
)

# Match `Budget.__init__` defaults. Duplicated because Budget does not export them as
# constants, and this slice must not edit `core/budget.py`.
_DEFAULT_MAX_TURNS = 10
_DEFAULT_MAX_TOKENS = 100_000
_DEFAULT_MAX_SECONDS = 120.0


class ManifestError(AgentContainerError, ValueError):
    """The YAML manifest is missing a field, has an unknown one, or failed validation."""


class UngatedWriteError(AgentContainerError, ValueError):
    """An `effect: write` tool has no `PermissionGate` anywhere in its stack."""


class Limits(BaseModel):
    """The three budget ceilings. Same names and defaults as `Budget`."""

    model_config = ConfigDict(extra="forbid")

    max_turns: int = Field(default=_DEFAULT_MAX_TURNS, ge=0)
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, ge=0)
    max_seconds: float = Field(default=_DEFAULT_MAX_SECONDS, ge=0.0)


class Manifest(BaseModel):
    """The YAML schema: model, system prompt, tool names, limits, permissions."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    # None: the key was omitted, so every registered tool is included. An explicit
    # empty list is an allowlist of nothing.
    tools: list[str] | None = None
    limits: Limits = Field(default_factory=Limits)
    permissions: list[str] = Field(default_factory=list)

    @field_validator("model", "system_prompt")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("tools")
    @classmethod
    def tool_names_are_identifiers(cls, names: list[str] | None) -> list[str] | None:
        if names is None:
            return None
        seen: set[str] = set()
        for name in names:
            if not TOOL_NAME_PATTERN.match(name):
                raise ValueError(f"invalid tool name {name!r}")
            if name in seen:
                raise ValueError(f"duplicate tool name {name!r}")
            seen.add(name)
        return names


def load_manifest(text: str) -> Manifest:
    """Parse a YAML document into a `Manifest`. Values are not echoed in the error."""
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest is not valid YAML: {exc}") from exc
    if loaded is None:
        raise ManifestError("manifest is empty")
    if not isinstance(loaded, dict):
        raise ManifestError("manifest must be a mapping")
    return manifest_from_mapping(loaded)


def manifest_from_mapping(data: Mapping[str, object]) -> Manifest:
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'manifest'}: {e['msg']}"
            for e in exc.errors(include_url=False, include_input=False)
        )
        raise ManifestError(f"invalid manifest: {problems}") from exc


def require_write_gate(tool: AnyTool) -> None:
    """Walk the decorator stack; raise if a write has no gate anywhere in it.

    The innermost tool's `effect` is what the spec says to read. `writes()` is also
    consulted, because a wrapper that claims `WRITE` over a read leaf is still a write
    stack (the same rule the loop enforces). A gate wrapped by `Cached` still counts.
    """
    leaf = innermost(tool)
    if writes(tool) and not has_write_gate(tool):
        raise UngatedWriteError(
            f"tool {tool.name!r} writes (innermost effect {leaf.effect!r}) but has no "
            "permission gate in its stack; wrap it in a PermissionGate before giving it "
            "to AgentBuilder"
        )


class AgentBuilder:
    """Staged assembly of one agent. Builder, not Factory: one product, filled in step by step."""

    def __init__(self) -> None:
        self._model: str | None = None
        self._system_prompt: str | None = None
        self._limits = Limits()
        self._permissions: frozenset[str] = frozenset()
        self._subject = "anonymous"
        self._identity: Identity | None = None
        self._tools: dict[str, AnyTool] = {}
        self._manifest_tool_names: list[str] | None = None
        self._provider: ModelProvider | None = None
        self._bus = EventBus()
        self._observers: list[LoopObserver] = [self._bus]
        self._fence_nonce: str | None = None

    @classmethod
    def from_yaml(cls, text: str) -> AgentBuilder:
        return cls().with_manifest(text)

    @classmethod
    def from_yaml_path(cls, path: Path | str) -> AgentBuilder:
        return cls().with_manifest_path(path)

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    def with_manifest(self, text: str) -> Self:
        return self._apply_manifest(load_manifest(text))

    def with_manifest_path(self, path: Path | str) -> Self:
        return self.with_manifest(Path(path).read_text(encoding="utf-8"))

    def _apply_manifest(self, manifest: Manifest) -> Self:
        self._model = manifest.model
        self._system_prompt = manifest.system_prompt
        self._limits = manifest.limits
        self._permissions = frozenset(manifest.permissions)
        self._manifest_tool_names = None if manifest.tools is None else list(manifest.tools)
        return self

    def with_model(self, model: str) -> Self:
        if not model.strip():
            raise ManifestError("model must not be blank")
        self._model = model
        return self

    def with_system_prompt(self, prompt: str) -> Self:
        if not prompt.strip():
            raise ManifestError("system_prompt must not be blank")
        self._system_prompt = prompt
        return self

    def with_limits(
        self,
        *,
        max_turns: int | None = None,
        max_tokens: int | None = None,
        max_seconds: float | None = None,
    ) -> Self:
        try:
            self._limits = Limits(
                max_turns=self._limits.max_turns if max_turns is None else max_turns,
                max_tokens=self._limits.max_tokens if max_tokens is None else max_tokens,
                max_seconds=self._limits.max_seconds if max_seconds is None else max_seconds,
            )
        except ValidationError as exc:
            raise ManifestError("invalid limits") from exc
        return self

    def with_permissions(self, permissions: Iterable[str], *, subject: str | None = None) -> Self:
        self._permissions = frozenset(permissions)
        if subject is not None:
            self._subject = subject
        self._identity = None
        return self

    def with_identity(self, identity: Identity) -> Self:
        self._identity = identity
        return self

    def with_tool(self, tool: AnyTool) -> Self:
        if not TOOL_NAME_PATTERN.match(tool.name):
            raise ValueError(f"invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"two tools are named {tool.name!r}")
        self._tools[tool.name] = tool
        return self

    def with_tools(self, tools: Iterable[AnyTool]) -> Self:
        for tool in tools:
            self.with_tool(tool)
        return self

    def with_provider(self, provider: ModelProvider) -> Self:
        self._provider = provider
        return self

    def with_event_bus(self, bus: EventBus) -> Self:
        """Replace the bus created with the builder. The previous bus is dropped."""
        self._observers = [observer for observer in self._observers if observer is not self._bus]
        self._bus = bus
        self._observers.insert(0, bus)
        return self

    def with_subscriber(self, subscriber: LoopObserver) -> Self:
        self._bus.subscribe(subscriber)
        return self

    def with_json_log(self, stream: TextIO | None = None) -> Self:
        """Subscribe the JSON-lines logger to this builder's event bus."""
        self._bus.subscribe(JsonLinesSubscriber(stream=stream))
        return self

    def with_observer(self, observer: LoopObserver) -> Self:
        """A loop observer besides the bus (the bus is always one)."""
        if observer is not self._bus and observer not in self._observers:
            self._observers.append(observer)
        return self

    def with_fence_nonce(self, nonce: str) -> Self:
        self._fence_nonce = nonce
        return self

    def budget(self) -> Budget:
        """A fresh `Budget` from the current limits. One per run; do not reuse it."""
        return Budget(
            max_turns=self._limits.max_turns,
            max_tokens=self._limits.max_tokens,
            max_seconds=self._limits.max_seconds,
        )

    def build(self) -> AgentLoop:
        if self._provider is None:
            raise ManifestError("a provider is required; call with_provider before build")
        if self._system_prompt is None:
            raise ManifestError(
                "a system prompt is required; set it on the manifest or with_system_prompt"
            )
        tools = self._selected_tools()
        for tool in tools:
            require_write_gate(tool)
        identity = self._identity or Identity(self._subject, self._permissions)
        return AgentLoop(
            self._provider,
            tools,
            system_prompt=self._system_prompt,
            identity=identity,
            observers=tuple(self._observers),
            fence_nonce=self._fence_nonce,
        )

    def _selected_tools(self) -> list[AnyTool]:
        if self._manifest_tool_names is None:
            return list(self._tools.values())
        selected: list[AnyTool] = []
        missing: list[str] = []
        for name in self._manifest_tool_names:
            tool = self._tools.get(name)
            if tool is None:
                missing.append(name)
            else:
                selected.append(tool)
        if missing:
            raise ManifestError(
                "manifest names tools that were not registered: " + ", ".join(missing)
            )
        return selected


__all__ = [
    "AgentBuilder",
    "Limits",
    "Manifest",
    "ManifestError",
    "UngatedWriteError",
    "load_manifest",
    "manifest_from_mapping",
    "require_write_gate",
]
