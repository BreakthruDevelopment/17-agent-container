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
    walk_stack,
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
        # A snapshot, deep: what a person reads in `description` must be what `apply()` runs. The
        # caller's model is mutable and may be reused to stage the next call (a batch), or be
        # changed by an interceptor's `after` hook once staging has returned; a shared reference
        # would let any of that rewrite an operation already shown for approval. Deep, because a
        # nested list or model is the same hazard one level down.
        self._args = args.model_copy(deep=True)
        self._ctx = ctx
        self._attempted = False
        # Rendered once, now: this string is the record of what the approver was shown, and a
        # record that moves is worse than none, because it reads as corroboration. Recomposing it
        # on every read made it a live view of `self._args`, and `apply()` hands that very object
        # to the tool -- so an interceptor's `before` hook beneath the gate (which holds the
        # tool's own authority, and may legitimately clamp or normalise an argument) rewrote the
        # audit text to match what ran. Described 5000, ran 100, read afterwards: 100. The
        # execution was right; the approval was the thing that got lost.
        self._full_description = (
            f"{self._tool.name}: {self._tool.description} "
            f"(arguments: {self._args.model_dump_json()})"
        )
        # Shortened once, now, for the same reason and in the same shape as `_RedactedStaged`:
        # `limit_description` reads the module-level `MAX_DESCRIPTION_CHARS` at call time, so
        # shortening on every read left this last step of the record live even though the text
        # beneath it was frozen. Measured: a 334-character staged record read back as
        # `'w: Write. (arguments...'` under a patched cap, while the same change behind
        # `Redacting` held still. Not reachable from the applied call -- `ToolContext` carries
        # `run_id`, `turn`, `tool_use_id` and `identity`, so neither a tool nor an interceptor
        # beneath the gate can reach a module global -- so this is the letter of
        # `StagedChange.description`'s contract rather than a live defect. Frozen anyway: the
        # contract is that the string does not move, and apps 18, 19 and 20 should not each have
        # to re-derive that nothing can reach it today.
        self._description = limit_description(self._full_description)

    @property
    def full_description(self) -> str:
        """Everything, unshortened. A wrapper that redacts reads this, then shortens.

        Fixed when the change was staged; see `__init__`.
        """
        return self._full_description

    @property
    def description(self) -> str:
        """The record of what the approver was shown. Fixed when the change was staged."""
        return self._description

    async def apply(self) -> None:
        if self._attempted:
            raise StagedChangeError("this change has already been applied or attempted")
        self._attempted = True
        result = await self._tool.run(self._args, self._ctx)
        if result.staged is not None:
            # A result that only stages another change was not executed. Reporting success here
            # would lose an approval silently, and `_attempted` is already set.
            raise StagedChangeError(
                f"{self._tool.name} was not executed: applying it produced another staged "
                "change instead of running. This approval is spent; nothing was done."
            )
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
        if self.policy == Policy.STAGED_WRITE and any(
            getattr(member, "policy", None) == Policy.STAGED_WRITE for member in walk_stack(inner)
        ):
            # Applying the outer change would only stage the inner one: an approval that does
            # nothing. One staging gate per stack.
            raise ValueError(f"{inner.name!r} is already behind a staging gate; do not nest them")
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
