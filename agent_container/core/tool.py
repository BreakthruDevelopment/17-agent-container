"""The tool contract: what a tool is, what it returns, and how a stack of wrappers is read.

A tool is wrapped by composition. `Retrying`, `Cached`, `Redacting`, `Metered` (17.4) and
`PermissionGate` each hold another tool, are themselves a `Tool`, and expose what they hold through
`Tool.wrapped`. `walk_stack` and `innermost` follow that chain, which is what lets the builder
(17.5) find a write tool's real `effect`, and a gate, however many decorators sit in between.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast

from pydantic import BaseModel, JsonValue, ValidationError


class Effect(StrEnum):
    """Whether a tool only looks or changes something. Required: the gate check reads it."""

    READ = "read"
    WRITE = "write"


# A tool name is an identifier. One definition, used by the loop at dispatch and by the provider
# contract suite -- two copies drifted once already, and a test can only pin the copy it imports.
# `\Z`, not `$`: in Python `$` also matches before a trailing newline, so `"echo\n"` would pass.
TOOL_NAME_PATTERN = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


class ErrorCategory(StrEnum):
    """Why a tool failed, so the model can tell a retry from a dead end (exam 2.2)."""

    TRANSIENT = "transient"  # might work if tried again
    VALIDATION = "validation"  # the input was wrong; fix it
    PERMISSION = "permission"  # not allowed, now or ever for this caller
    BUSINESS = "business"  # a rule said no; explain it, do not retry
    INTERNAL = "internal"  # the tool broke; nothing the caller did, and no rule to explain


class StagedChange(ABC):
    """A recorded write waiting for a person to approve it (Command).

    The core defines only the contract, because `ToolResult` and `RunResult` carry staged changes
    and so must be able to name the type. `PermissionGate` (17.4) supplies the implementation that
    replays a tool call.
    """

    @property
    @abstractmethod
    def description(self) -> str:
        """What this change would do, in words a person reads before approving.

        **It is the record of what was shown at approval, so it must not move.** Fixed when the
        change is staged, and the same string before and after `apply()`, whatever the applied
        call turns out to do: an interceptor beneath a gate may legitimately clamp or normalise
        an argument, and the execution then rightly follows the hook -- but an audit trail that
        silently agrees with whatever happened is worse than none, because it reads as
        corroboration. Render it once, from a snapshot; never recompose it from state the applied
        call can reach. `StagedChangeContract` in `agent_container.testing` checks this, and apps
        18 and 19 both depend on an approval record surviving.
        """

    @abstractmethod
    async def apply(self) -> None:
        """Carry the change out. Called only after approval."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool returns. A failure is a result, not an exception, so the model can react."""

    content: str
    is_error: bool = False
    error_category: ErrorCategory | None = None
    is_retryable: bool = False
    staged: StagedChange | None = None

    def __post_init__(self) -> None:
        # Normalise the category to a member, once, here -- so every `ToolResult` in the system
        # carries one however it was built: directly, through `error()`, or through
        # `dataclasses.replace` (which `Cached` and `Redacting` use). A plain wire value such as
        # `"transient"` is *equal* to its member but is not it, so `is` and `match` over the enum
        # take the wrong branch -- here, and in the three apps that import this library and can
        # match on a category this one handed them. An unknown value is a `ValueError`, not a
        # default, exactly as `Policy(policy)` in `gate.py` already is.
        #
        # `raw: object` so the `isinstance` narrowing is real to the type checker: the field is
        # annotated `ErrorCategory | None`, and under `warn_unreachable` a check against the
        # declared type reads as dead code. The runtime values are what this guards.
        raw: object = self.error_category
        if raw is not None and not isinstance(raw, ErrorCategory):
            if not isinstance(raw, str):
                # Neither a member nor a wire value: there is nothing honest to coerce it to, and
                # storing it would put a non-category where every reader expects one. The message
                # names the field, never the value.
                raise ValueError("error_category must be an ErrorCategory")
            # frozen=True, slots=True: this is the sanctioned way to normalise in __post_init__.
            object.__setattr__(self, "error_category", ErrorCategory(raw))
        if self.is_error and self.error_category is None:
            raise ValueError("an error result needs an error_category")
        if not self.is_error and (self.error_category is not None or self.is_retryable):
            raise ValueError("only an error result carries an error_category or is_retryable")

    @classmethod
    def ok(cls, content: str, *, staged: StagedChange | None = None) -> ToolResult:
        return cls(content, staged=staged)

    @classmethod
    def error(
        cls,
        category: ErrorCategory,
        message: str,
        *,
        retryable: bool | None = None,
    ) -> ToolResult:
        """A failed result. `retryable` defaults to True only for transient errors."""
        # Coerced here as well as in `__post_init__`, because the default below is computed from
        # the *parameter*, before the dataclass exists. The signature says `ErrorCategory`, and
        # that stays the contract; this only stops an untyped caller's `"transient"` from
        # defaulting to not-retryable, which is the wrong answer for a retryable error.
        category = ErrorCategory(category)
        if retryable is None:
            retryable = category is ErrorCategory.TRANSIENT
        return cls(message, is_error=True, error_category=category, is_retryable=retryable)


@dataclass(frozen=True, slots=True)
class Identity:
    """Who a run acts for. Decorators that remember results (`Cached`) must key on it, or one
    caller's decision is served to another."""

    subject: str
    permissions: frozenset[str] = frozenset()

    @classmethod
    def anonymous(cls) -> Identity:
        return cls("anonymous")


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool is told about the call it is serving."""

    run_id: str
    turn: int
    tool_use_id: str
    identity: Identity = field(default_factory=Identity.anonymous)


class AnyTool(Protocol):
    """A tool with its input type erased: the view the loop, decorators and builder hold.

    `Tool[A]` needs the argument type to be precise for the author, but a registry holds tools of
    many argument types, and a generic `Tool[A]` is invariant in `A`. Every `Tool[A]` satisfies
    this.
    """

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def effect(self) -> Effect: ...
    @property
    def input_model(self) -> type[BaseModel]: ...
    @property
    def wrapped(self) -> AnyTool | None: ...
    def input_schema(self) -> Mapping[str, JsonValue]: ...
    async def invoke(self, raw: Mapping[str, JsonValue], ctx: ToolContext) -> ToolResult: ...


class Tool[A: BaseModel](ABC):
    """Base for every tool and every tool decorator. `A` is the typed input model.

    A leaf tool sets `name`, `description`, `effect` and `input_model` (class attributes are fine)
    and implements `run`. A decorator takes the tool it wraps, forwards the four to it, returns it
    from `wrapped` and delegates `run`. Nothing else is needed, and nothing inherits from more than
    this one class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name the model calls it by: letters, digits, `_`, `-`; up to 64 characters."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the tool does and when to use it, written for the model."""

    @property
    @abstractmethod
    def effect(self) -> Effect:
        """`read` or `write`. A decorator returns the effect of the tool it wraps."""

    @property
    @abstractmethod
    def input_model(self) -> type[A]:
        """The typed, validated shape of the arguments."""

    @property
    def wrapped(self) -> AnyTool | None:
        """The tool this one wraps, or `None` for a leaf. Decorators override it."""
        return None

    @abstractmethod
    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        """Do the work on validated arguments. Return failures as results, do not raise."""

    def input_schema(self) -> Mapping[str, JsonValue]:
        # model_json_schema() is typed dict[str, Any]; a JSON Schema is JSON by definition, so the
        # cast only names that.
        return cast(dict[str, JsonValue], self.input_model.model_json_schema())

    async def invoke(self, raw: Mapping[str, JsonValue], ctx: ToolContext) -> ToolResult:
        """Validate `raw` against `input_model`, then `run`. What the loop calls.

        Invalid input comes back as a `validation` result naming the fields, never echoing the
        values.
        """
        try:
            args = self.input_model.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult.error(ErrorCategory.VALIDATION, f"invalid arguments: {problems}")
        return await self.run(args, ctx)


def walk_stack(tool: AnyTool) -> Iterator[AnyTool]:
    """Yield `tool`, then what it wraps, then what that wraps, down to the leaf."""
    seen: set[int] = set()
    current: AnyTool | None = tool
    while current is not None:
        if id(current) in seen:
            raise ValueError(f"a {type(current).__name__} in this tool stack wraps itself")
        seen.add(id(current))
        yield current
        current = current.wrapped


def innermost(tool: AnyTool) -> AnyTool:
    """The leaf tool at the bottom of a decorator stack."""
    *_, leaf = walk_stack(tool)
    return leaf


# The gate contract. `AGENTS.md` makes it a *construction* error for an `effect: write` tool to
# reach a loop without a `PermissionGate` anywhere in its stack, and construction lives in 17.2
# while the gate itself is 17.4's. So the core defines what a gate must advertise -- the same split
# as `StagedChange` above -- and 17.2 enforces it. 17.4's `PermissionGate` sets `gates_writes`.
GATE_MARKER = "gates_writes"


class WriteGate(Protocol):
    """What a permission gate advertises. 17.4's `PermissionGate` satisfies this."""

    @property
    def gates_writes(self) -> bool: ...


def writes(tool: AnyTool) -> bool:
    """Whether anything in this stack claims to write.

    Deliberately `any`, not `innermost(...).effect`: a decorator that reported `READ` over a write
    leaf would otherwise slip a write tool past the gate check. A stack is a write stack if any
    member says so.

    `==`, not `is`: `Effect` is a `StrEnum`, so a tool whose `effect` is the plain string "write"
    compares equal but is not identical. `is` would call such a tool a read and let it past the
    gate -- a security check failing open. `==` fails closed and costs nothing. (`mypy --strict`
    and the provider contract suite each catch the plain string too, but neither runs at the
    moment this decision is made.)
    """
    return any(member.effect == Effect.WRITE for member in walk_stack(tool))


def has_write_gate(tool: AnyTool) -> bool:
    """Whether some member of this stack advertises itself as a permission gate.

    Read by attribute rather than `isinstance`: a `runtime_checkable` Protocol with a non-method
    member cannot be used with `isinstance`, and importing 17.4's concrete gate here would invert
    the dependency. `is True` so a truthy stub cannot pass for a gate.

    A gate must also *wrap* something. Without that, a write tool could set `gates_writes = True`
    on itself and approve its own writes; a decorator always has something beneath it, and a leaf
    never does. This catches the accidental case and nothing more: an author set on evading it can
    return a throwaway tool from `wrapped`, or more cheaply just declare `Effect.READ`, which no
    check here has ever claimed to catch. The gate check guards mistakes, not the tool's author.

    The question it answers is "is a gate somewhere in this stack", not "is every write gated". A
    decorator that declares `Effect.WRITE`, performs writes of its own, and holds a gated stack
    beneath that it never calls satisfies it. That is the same evasion, and `Cached` relies on this
    check, so a decorator that writes must delegate to what it wraps.
    """
    return any(
        getattr(member, GATE_MARKER, False) is True and member.wrapped is not None
        for member in walk_stack(tool)
    )
