"""Run the shared contract suites against everything in this slice.

17.3 and 17.4 do the same with their own implementations: subclass, supply a factory, done.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from pydantic import BaseModel, JsonValue

from agent_container.core import AnyTool, ModelProvider, ModelResponse, StagedChange
from agent_container.core.decorators import Redacting, _RedactedStaged
from agent_container.testing import (
    ModelProviderContract,
    OverlapProbe,
    StagedCase,
    StagedChangeContract,
    ToolContract,
)
from tests.support import (
    DeleteTool,
    EchoTool,
    PassThrough,
    ProbedReplay,
    pay_record_as_it_would_read,
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

    def make_change(self) -> StagedCase:
        change, pay = staged_pay_behind_a_clamp()
        return StagedCase(
            change,
            lambda: pay.paid,
            approved_value="5000",
            rewritten_value="100",
            recomposed=lambda: pay_record_as_it_would_read(pay),
        )


class TestRedactedStagedContract(StagedChangeContract):
    """The same change seen through `Redacting`, which re-renders the text on every read.

    It reads `full_description` from the change beneath it, so it is frozen only if that one is.
    """

    def make_change(self) -> StagedCase:
        change, pay = staged_pay_behind_a_clamp()
        redacting = Redacting(EchoTool(), [r"secret-\d+"])
        return StagedCase(
            _RedactedStaged(change, redacting.redact),
            lambda: pay.paid,
            approved_value="5000",
            rewritten_value="100",
            recomposed=lambda: pay_record_as_it_would_read(pay, redacting.redact),
        )


# -- The contract suite must itself be able to fail -----------------------------------------------
# A contract suite that passes against a broken implementation is worse than no suite, because
# every downstream consumer inherits the false assurance. Apps 18 and 19 will subclass
# `StagedChangeContract` for their own staged changes, so the cases below pin that it bites. Each
# one defeated an earlier version of the suite; the message each now fails with is asserted, so a
# future change that makes one fail for an unrelated reason is not mistaken for coverage.


class _ThawedArgs(BaseModel):
    amount: int


class _Thawed(StagedChange):
    """Round 1's witness: the description is recomposed live, and `apply()` does nothing at all.

    Not frozen -- `thaw()` moves the record -- yet the declaration-based suite passed it green,
    because a no-op `apply()` cannot move anything *during* apply.
    """

    def __init__(self) -> None:
        self.args = _ThawedArgs(amount=5000)

    @property
    def description(self) -> str:
        return f"pay: {self.args.amount}"

    async def apply(self) -> None:
        return None

    def thaw(self) -> None:
        self.args.amount = 100


class _ThawedCase(StagedChangeContract):
    """Reports no applied call, because there is none."""

    def make_change(self) -> StagedCase:
        thawed = _Thawed()
        return StagedCase(
            thawed,
            lambda: [],
            approved_value="5000",
            rewritten_value="100",
            recomposed=lambda: thawed.description,
        )


class _UnrewrittenCase(StagedChangeContract):
    """A real staged change, but the hook beneath leaves the arguments alone."""

    def make_change(self) -> StagedCase:
        change, pay = staged_pay_behind_a_clamp(amount=5000, clamp_to=5000)
        return StagedCase(
            change,
            lambda: pay.paid,
            approved_value="5000",
            rewritten_value="100",
            recomposed=lambda: pay_record_as_it_would_read(pay),
        )


class _ModelProbeCase(StagedChangeContract):
    """Defeated the one-sided guard: the probe reports the arguments model, not the scalar.

    `str(ClampArgs(amount=5000))` is `'amount=5000'`, which is not a substring of
    `'... (arguments: {"amount":5000})'`, so an "applied value absent from the record" check alone
    passed -- with the hook rewriting nothing. This probe shape is the idiom already used
    elsewhere in `tests/support.py`, so it is not a contrived one.
    """

    def make_change(self) -> StagedCase:
        change, pay = staged_pay_behind_a_clamp(amount=5000, clamp_to=5000)
        return StagedCase(
            change,
            lambda: list(pay.received),
            approved_value="5000",
            rewritten_value="100",
            recomposed=lambda: pay_record_as_it_would_read(pay),
        )


class _TruncatedCase(StagedChangeContract):
    """Defeated the one-sided guard: the arguments fall off the end of the description.

    A tool description long enough to push `full_description` past `MAX_DESCRIPTION_CHARS` means
    `limit_description` cuts the arguments away, so "applied value absent from the record" was
    trivially true and the equality then compared two identical truncated prefixes.
    """

    def make_change(self) -> StagedCase:
        change, pay = staged_pay_behind_a_clamp(long_description=True)
        return StagedCase(
            change,
            lambda: pay.paid,
            approved_value="5000",
            rewritten_value="100",
            recomposed=lambda: pay_record_as_it_would_read(pay),
        )


def test_the_contract_rejects_a_change_whose_apply_reaches_nothing() -> None:
    """Round 1's witness. Not frozen, and the declaration-based suite passed it green."""
    thawed = _Thawed()
    before = thawed.description
    thawed.thaw()
    assert thawed.description != before, "this witness is supposed to be unfrozen"

    with pytest.raises(AssertionError, match="never reached the tool"):
        _ThawedCase().test_applying_does_not_rewrite_the_approval_record()


def test_the_contract_rejects_a_hook_that_does_not_rewrite_the_arguments() -> None:
    with pytest.raises(AssertionError, match="not the rewritten value"):
        _UnrewrittenCase().test_applying_does_not_rewrite_the_approval_record()


def test_the_contract_rejects_a_probe_too_coarse_to_show_the_rewrite() -> None:
    with pytest.raises(AssertionError, match="not the rewritten value"):
        _ModelProbeCase().test_applying_does_not_rewrite_the_approval_record()


def test_the_contract_rejects_a_description_that_truncated_the_arguments_away() -> None:
    with pytest.raises(AssertionError, match="does not report the approved value"):
        _TruncatedCase().test_applying_does_not_rewrite_the_approval_record()


class _InvisibleFieldArgs(BaseModel):
    shown: int = 5000
    hidden: str = "aaa"


class _InvisibleFieldChange(StagedChange):
    """Round 3's witness: the applied call rewrites a field the record never displays.

    The description is recomposed live, so this is not frozen -- `thaw()` moves it. But the
    rewritten field is absent from the rendered record, so all four substring relations hold with
    honest values, and the freeze assertion passed because a live description had nothing to
    show. Reachable for real: a `Field(exclude=True)` argument does this through the actual gate,
    and an approval prompt that renders a human summary rather than full JSON makes most
    arguments invisible -- at which point the suite's own constraints push the author towards
    exactly this configuration.
    """

    def __init__(self) -> None:
        self.args = _InvisibleFieldArgs()
        self.received: list[str] = []

    @property
    def description(self) -> str:
        return f'pay: Pay a supplier. (arguments: {{"shown":{self.args.shown}}})'

    async def apply(self) -> None:
        self.args.hidden = "zzz"
        self.received.append(self.args.hidden)

    def thaw(self) -> None:
        self.args.shown = 100


class _InvisibleFieldCase(StagedChangeContract):
    def make_change(self) -> StagedCase:
        change = _InvisibleFieldChange()
        return StagedCase(
            change,
            lambda: change.received,
            approved_value="5000",
            rewritten_value="zzz",
            recomposed=lambda: change.description,
        )


def test_the_contract_rejects_a_rewrite_the_record_never_displays() -> None:
    """The four substring relations all hold here, with honest values. Only `recomposed` sees it."""
    change = _InvisibleFieldChange()
    before = change.description
    change.thaw()
    assert change.description != before, "this witness is supposed to be unfrozen"

    with pytest.raises(AssertionError, match="never displays"):
        _InvisibleFieldCase().test_applying_does_not_rewrite_the_approval_record()
