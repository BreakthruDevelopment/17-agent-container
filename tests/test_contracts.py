"""Run the shared contract suites against everything in this slice.

17.3 and 17.4 do the same with their own implementations: subclass, supply a factory, done.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import JsonValue

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

    def make_change(self) -> StagedChange:
        change, _ = staged_pay_behind_a_clamp()
        return change

    def rewrites_arguments(self) -> bool:
        return True


class TestRedactedStagedContract(StagedChangeContract):
    """The same change seen through `Redacting`, which re-renders the text on every read.

    It reads `full_description` from the change beneath it, so it is frozen only if that one is.
    """

    def make_change(self) -> StagedChange:
        change, _ = staged_pay_behind_a_clamp()
        return _RedactedStaged(change, Redacting(EchoTool(), [r"secret-\d+"]).redact)

    def rewrites_arguments(self) -> bool:
        return True
