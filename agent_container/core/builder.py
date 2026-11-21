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
from typing import Self, TextIO, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import ErrorDetails

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
                # No value in the message: it reaches the caller through `msg`. See
                # `_describe_error`.
                raise ValueError("is not a valid tool name")
            if name in seen:
                raise ValueError("is a duplicate tool name")
            seen.add(name)
        return names


# Every field name the manifest may legitimately contain. A `loc` segment outside this set came
# from the DOCUMENT, not from the schema -- with `extra="forbid"` an unknown key becomes a loc
# segment -- so it is redacted. Unlike pydantic's `input_value`, `loc` is not truncated at all:
# a 20000-character key was measured reaching the message verbatim.
_KNOWN_FIELDS: frozenset[str] = frozenset(Manifest.model_fields) | frozenset(Limits.model_fields)

# Fixed text per pydantic error code, so the message never depends on `msg`. `msg` is partly
# input-derived -- a `field_validator` that interpolates the value puts it straight in, which is
# what ours did -- and a future validator must not be able to reopen that hole by being careless.
_ERROR_TEXT: dict[str, str] = {
    "missing": "is required",
    "extra_forbidden": "is not a known field",
    "string_type": "must be a string",
    "list_type": "must be a list",
    "dict_type": "must be a mapping",
    "int_type": "must be an integer",
    "int_parsing": "must be an integer",
    "float_type": "must be a number",
    "float_parsing": "must be a number",
    "bool_type": "must be a boolean",
    "greater_than_equal": "is below the permitted minimum",
    "greater_than": "is below the permitted minimum",
    "less_than_equal": "is above the permitted maximum",
    "less_than": "is above the permitted maximum",
    "too_long": "is too long",
    "too_short": "is too short",
    "string_too_short": "must not be empty",
    "model_type": "must be a mapping",
    "int_from_float": "must be a whole number",
    "invalid_key": "is not a usable key",
}
# Codes not in the table fall back to the raw pydantic code. That is safe -- the codes are a
# closed literal set in pydantic-core, never input-derived -- but it reads as jargon, so the
# four above were added after measuring which codes this schema can actually produce
# (`model: ""` reported `string_too_short`, and a numeric key `invalid_key`). Entries for codes
# the current schema cannot reach are kept deliberately, so adding a field does not regress
# the wording.


def _list_valued_fields(*models: type[BaseModel]) -> frozenset[str]:
    """Fields whose value is a list, so an integer `loc` segment under them is an index.

    Derived from the annotations rather than written out, so adding a list field to the schema
    cannot silently turn its indices into `<redacted>` -- or, far worse, the reverse.
    """
    found: set[str] = set()
    for model in models:
        for name, field in model.model_fields.items():
            candidates = (field.annotation, *get_args(field.annotation))
            if any(get_origin(c) is list for c in candidates):
                found.add(name)
    return frozenset(found)


# An integer `loc` segment is a list INDEX only directly inside one of these. Pydantic keeps a
# mapping KEY as an int whenever it fits an i64, and nothing in `loc` distinguishes the two, so
# treating every int as an index let an unknown numeric key through verbatim -- measured with
# `4111111111111111: x`, which rendered in full, as did a 19-digit key, a negative one, and
# `0xDEADBEEF` (as 3735928559). Larger ints and all floats pydantic stringifies, so those were
# already caught by the `_KNOWN_FIELDS` arm.
_INDEXED_FIELDS: frozenset[str] = _list_valued_fields(Manifest, Limits)


def _describe_error(error: ErrorDetails) -> str:
    """Render one pydantic error with nothing in it that came from the document.

    Every part of the obvious one-liner is input-derived and `include_input=False` covers none
    of it: `loc` carries unknown KEYS -- as strings, and as ints when they look like numbers --
    and `msg` carries whatever a validator interpolated.
    """
    parts: list[str] = []
    for position, part in enumerate(error["loc"]):
        if isinstance(part, int):
            parent = error["loc"][position - 1] if position else None
            parts.append(str(part) if parent in _INDEXED_FIELDS else "<redacted>")
        else:
            parts.append(part if part in _KNOWN_FIELDS else "<redacted>")
    where = ".".join(parts)
    code = error["type"]
    if code == "value_error":
        # One of our own validators. Their messages are value-free by construction (see
        # `tool_names_are_identifiers`), but we still do not pass `msg` through: strip
        # pydantic's "Value error, " prefix only if what is left is in the known set.
        detail = error["msg"].removeprefix("Value error, ")
        what = detail if detail in _VALUE_ERRORS else "is not valid"
    else:
        what = _ERROR_TEXT.get(code, code)
    return f"{where or 'manifest'}: {what}"


# The exact, value-free messages our own validators raise. Anything else is reported generically,
# so a validator added later cannot leak by interpolating a value into its ValueError.
_VALUE_ERRORS: frozenset[str] = frozenset(
    {"must not be blank", "is not a valid tool name", "is a duplicate tool name"}
)


def load_manifest(text: str) -> Manifest:
    """Parse a YAML document into a `Manifest`. Values are not echoed in the error.

    Both paths leaked, by different routes. The validation path passes `include_input=False`,
    which sanitizes the string we extract and does nothing to the `ValidationError` itself, so
    chaining it with `from exc` re-attached the input; `str(ValidationError)` renders
    `input_value=` with the offending value (pydantic truncates to roughly 25 head + 25 tail
    characters, so anything shorter than that is verbatim). On this path
    `str(YAMLError)` embeds a source excerpt -- `Mark.__str__` calls `get_snippet()`, up to 75
    characters of the buffer -- so a malformed manifest leaked whatever sat near the error: a
    `system_prompt`, a credential pasted into one. Reading `.line`/`.column` off the mark
    instead of stringifying it sidesteps the snippet.

    `problem` is safe for whole VALUES, which is what the promise covers, not for every byte of
    the document: several of PyYAML's problem strings repeat a YAML identifier -- `duplicate tag
    handle %r`, `found undefined alias %r`, `found undefined tag handle %r`, `could not
    determine a constructor for the tag %r`. Those are structural tokens. The scanner can also
    echo a single character from inside a quoted scalar (`expected escape sequence of 2
    hexadecimal numbers, but found 'S'`), which is a character, not a value. The one message
    that echoes a whole value, `expected the empty value, but found %r`, lives in
    `FullConstructor` and cannot be reached from `safe_load`. Reading only `.problem` also keeps
    `found duplicate anchor %r` out, since that lands in `.context`.

    And `safe_load` does not raise only `YAMLError`. `SafeConstructor`'s scalar conversions
    raise plain builtins with the value interpolated -- `!!int SECRET` gives `ValueError:
    invalid literal for int() with base 10: 'SECRET'`, `!!bool` a `KeyError`, `!!timestamp` an
    `AttributeError` -- so catching `YAMLError` alone let exactly the opening case through. The
    second handler catches those, names only the exception type, and suppresses the cause so the
    value cannot resurface in a traceback.
    """
    # Both messages are built inside the handlers and raised OUTSIDE them. Raising in place
    # keeps the original as `__context__` even with `from None`, and the original is the thing
    # that carries the value -- a snippet for a YAMLError, the scalar itself for a builtin. PR
    # #12 found the same shape in `ToolCallChange.apply()`: redacted message, secret in the
    # chain. `__suppress_context__` alone would hide it from the default traceback while
    # leaving it readable on the object.
    failure: str | None = None
    loaded: object = None
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None) or "could not be parsed"
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        failure = f"{problem}{where}"
    except Exception as exc:
        # `SafeConstructor`'s scalar conversions raise plain builtins with the value in the
        # message, so `except YAMLError` alone let the opening case straight through.
        failure = f"could not be parsed ({type(exc).__name__})"
    if failure is not None:
        raise ManifestError(f"manifest is not valid YAML: {failure}")
    if loaded is None:
        raise ManifestError("manifest is empty")
    if not isinstance(loaded, dict):
        raise ManifestError("manifest must be a mapping")
    return manifest_from_mapping(loaded)


def manifest_from_mapping(data: Mapping[str, object]) -> Manifest:
    invalid: str | None = None
    try:
        return Manifest.model_validate(data)
    except ValidationError as exc:
        # Build the message in here and raise OUTSIDE the handler. `from exc` would attach the
        # ValidationError as `__cause__`, and `include_input=False` only sanitizes the string
        # extracted above -- `str(exc)` still renders `input_value=<the manifest>`. Raising in
        # here would set `__context__` to the same object even with `from None`.
        invalid = "; ".join(
            _describe_error(e) for e in exc.errors(include_url=False, include_input=False)
        )
    raise ManifestError(f"invalid manifest: {invalid}")


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
            return self
        except ValidationError:
            # Raised outside for the reason given in `load_manifest`: chaining re-attaches the
            # rejected values. The message carries none, so there is nothing to extract.
            pass
        raise ManifestError("invalid limits")

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
            # The missing names came from the manifest and are by definition not in the
            # registry, so they are document content. The REGISTERED names are the caller's
            # own and safe to list -- and more useful for fixing the manifest anyway.
            available = ", ".join(sorted(self._tools)) or "none"
            raise ManifestError(
                f"manifest names {len(missing)} tool(s) that were not registered; "
                f"registered tools: {available}"
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
