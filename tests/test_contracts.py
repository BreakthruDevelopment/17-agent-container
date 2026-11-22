"""Run the shared contract suites against everything in this slice.

17.3 and 17.4 do the same with their own implementations: subclass, supply a factory, done.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest
from pydantic import BaseModel, JsonValue

from agent_container.core import AnyTool, ModelProvider, ModelResponse, StagedChange
from agent_container.core.decorators import Redacting, _RedactedStaged
from agent_container.testing import (
    ModelProviderContract,
    OverlapProbe,
    StagedChangeContract,
    ToolContract,
)
from tests.support import (
    DeleteTool,
    EchoTool,
    PassThrough,
    ProbedReplay,
    staged_pay_behind_a_clamp,
)


class TestReplayProviderContract(ModelProviderContract):
    def make_provider(
        self, replies: Sequence[ModelResponse], probe: OverlapProbe | None = None
    ) -> ModelProvider:
        return ProbedReplay(replies, probe)


class TestLeafToolContract(ToolContract):
    def make_tool(self) -> AnyTool:
        return EchoTool()

    def valid_input(self) -> Mapping[str, JsonValue]:
        return {"text": "hi"}

    def invalid_input(self) -> Mapping[str, JsonValue]:
        return {"text": 3}


class TestDecoratedToolContract(ToolContract):
    """Three decorators deep: still a tool, still the same effect, still walkable."""

    def make_tool(self) -> AnyTool:
        return PassThrough(PassThrough(PassThrough(DeleteTool())))

    def valid_input(self) -> Mapping[str, JsonValue]:
        return {"text": "hi"}

    def invalid_input(self) -> Mapping[str, JsonValue]:
        return {}


class TestToolCallChangeContract(StagedChangeContract):
    """`PermissionGate`'s own staged change, with a clamping hook beneath the gate."""

    def make_change(self) -> tuple[StagedChange, Callable[[], Sequence[object]]]:
        change, pay = staged_pay_behind_a_clamp()
        return change, lambda: pay.paid


class TestRedactedStagedContract(StagedChangeContract):
    """The same change seen through `Redacting`, which re-renders the text on every read.

    It reads `full_description` from the change beneath it, so it is frozen only if that one is.
    """

    def make_change(self) -> tuple[StagedChange, Callable[[], Sequence[object]]]:
        change, pay = staged_pay_behind_a_clamp()
        redacting = Redacting(EchoTool(), [r"secret-\d+"])
        return _RedactedStaged(change, redacting.redact), lambda: pay.paid


# -- The contract suite must itself be able to fail -----------------------------------------------
# A contract suite that passes against a broken implementation is worse than no suite, because
# every downstream consumer inherits the false assurance. Apps 18 and 19 will subclass
# `StagedChangeContract` for their own staged changes, so these two cases pin that it bites. The
# first version of the suite asked the subclass to *declare* that its arguments get rewritten; both
# cases below passed it green.


class _ThawedArgs(BaseModel):
    amount: int


class _Thawed(StagedChange):
    """Not frozen at all: the description is recomposed live, and `apply()` reaches no tool."""

    def __init__(self) -> None:
        self.args = _ThawedArgs(amount=5000)

    @property
    def description(self) -> str:
        return f"pay: {self.args.amount}"

    async def apply(self) -> None:
        self.args.amount = 100


class _ThawedCase(StagedChangeContract):
    """Reports no applied call, because there is none."""

    def make_change(self) -> tuple[StagedChange, Callable[[], Sequence[object]]]:
        return _Thawed(), lambda: []


class _UnrewrittenCase(StagedChangeContract):
    """A real staged change, but the hook beneath leaves the arguments alone."""

    def make_change(self) -> tuple[StagedChange, Callable[[], Sequence[object]]]:
        change, pay = staged_pay_behind_a_clamp(amount=5000, clamp_to=5000)
        return change, lambda: pay.paid


def test_the_contract_rejects_a_change_whose_apply_reaches_nothing() -> None:
    with pytest.raises(AssertionError, match="never reached the tool"):
        _ThawedCase().test_applying_does_not_rewrite_the_approval_record()


def test_the_contract_rejects_a_hook_that_does_not_rewrite_the_arguments() -> None:
    with pytest.raises(AssertionError, match="already reports"):
        _UnrewrittenCase().test_applying_does_not_rewrite_the_approval_record()
