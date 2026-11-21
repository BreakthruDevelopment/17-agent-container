"""The permission gate (Proxy) and the staged change it hands back (Command)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from agent_container.core.decorators import Decorator, limit_description
from agent_container.core.errors import StagedChangeError
from agent_container.core.tool import (
    ErrorCategory,
    StagedChange,
    Tool,
    ToolContext,
    ToolResult,
    writes,
)


class Policy(StrEnum):
    READ = "read"  # run the tool now; only for a tool that does not write
    STAGED_WRITE = "staged_write"  # record the call as a StagedChange; the default for a write
    DENY = "deny"  # refuse every call


class ToolCallChange[A: BaseModel](StagedChange):
    """A recorded tool call. `apply()` runs it, once, after a person approves.

    Single use, and the attempt is what is counted, not the success: the flag is set before the
    tool runs, so a call that fails, or is cancelled partway, cannot be applied again. A write
    that may have half happened is not safe to repeat blindly; a person decides what to do next.
    A failed tool raises `StagedChangeError` so the approver sees it instead of a silent no-op.
    """

    def __init__(self, tool: Tool[A], args: A, ctx: ToolContext) -> None:
        self._tool = tool
        self._args = args
        self._ctx = ctx
        self._attempted = False

    @property
    def full_description(self) -> str:
        """Everything, unshortened. A wrapper that redacts reads this, then shortens."""
        shown = self._args.model_dump_json()
        return f"{self._tool.name}: {self._tool.description} (arguments: {shown})"

    @property
    def description(self) -> str:
        return limit_description(self.full_description)

    async def apply(self) -> None:
        if self._attempted:
            raise StagedChangeError("this change has already been applied or attempted")
        self._attempted = True
        result = await self._tool.run(self._args, self._ctx)
        if result.is_error:
            raise StagedChangeError(
                f"{self._tool.name} failed when applied ({result.error_category}): {result.content}"
            )


class PermissionGate[A: BaseModel](Decorator[A]):
    """Stand in front of a tool and decide, from its policy and the caller, what happens.

    `gates_writes` is what `has_write_gate` looks for, anywhere in the stack, so
    `Cached(PermissionGate(write))` is valid. Policy `STAGED_WRITE` requires a tool that writes and
    `READ` one that does not: a gate that runs a write immediately, or stages a read, is a
    mistake, refused at construction. `requires`, when set, is a permission the acting identity
    must hold; without it the call is denied, whatever the policy.

    What a staged call returns to the model is a fixed library string. The arguments appear only
    in the staged change's description, which a person reads.
    """

    gates_writes = True

    def __init__(
        self,
        inner: Tool[A],
        policy: Policy | str = Policy.STAGED_WRITE,
        *,
        requires: str | None = None,
    ) -> None:
        super().__init__(inner)
        self.policy = Policy(policy)  # an unknown policy is a ValueError, not a default
        self.requires = requires
        if self.policy == Policy.READ and writes(inner):
            raise ValueError(f"policy 'read' cannot gate {inner.name!r}: it writes")
        if self.policy == Policy.STAGED_WRITE and not writes(inner):
            raise ValueError(f"policy 'staged_write' needs a writing tool; {inner.name!r} reads")

    async def run(self, args: A, ctx: ToolContext) -> ToolResult:
        if self.policy == Policy.DENY or (
            self.requires is not None and self.requires not in ctx.identity.permissions
        ):
            return ToolResult.error(
                ErrorCategory.PERMISSION, "this caller is not permitted to use this tool"
            )
        if self.policy == Policy.READ:
            return await self._inner.run(args, ctx)
        change = ToolCallChange(self._inner, args, ctx)
        return ToolResult.ok(
            f"change staged for approval; nothing has been done yet ({self.name})",
            staged=change,
        )
