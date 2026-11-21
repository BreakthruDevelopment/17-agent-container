"""AgentBuilder: YAML manifest, staged assembly, and the construction-time gate rule."""

from __future__ import annotations

import io
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent_container.core import (
    AgentBuilder,
    AgentLoop,
    AnyTool,
    Budget,
    Cached,
    EndReason,
    EventBus,
    Identity,
    LoopEvent,
    ManifestError,
    Metered,
    Metrics,
    ModelResponse,
    PermissionGate,
    Retrying,
    UngatedWriteError,
    load_manifest,
)
from tests.support import (
    DeleteTool,
    EchoTool,
    LyingDecorator,
    PassThrough,
    StringEffectTool,
    WriteClaimingDecorator,
    scripted,
    tool_call,
    use_tool,
)

NONCE = "testnonce0123456"

MANIFEST = """\
model: test-model
system_prompt: You are a test agent.
tools:
  - echo
limits:
  max_turns: 4
  max_tokens: 1000
  max_seconds: 30
permissions:
  - invoices:read
"""


def _builder(*tools: AnyTool) -> AgentBuilder:
    builder = (
        AgentBuilder()
        .with_model("test-model")
        .with_system_prompt("You are a test agent.")
        .with_provider(scripted(ModelResponse.of("ok")))
        .with_fence_nonce(NONCE)
    )
    for tool in tools:
        builder.with_tool(tool)
    return builder


def test_the_manifest_schema_is_model_prompt_tools_limits_permissions() -> None:
    loaded = load_manifest(MANIFEST)
    assert loaded.model == "test-model"
    assert loaded.system_prompt == "You are a test agent."
    assert loaded.tools == ["echo"]
    assert (loaded.limits.max_turns, loaded.limits.max_tokens, loaded.limits.max_seconds) == (
        4,
        1000,
        30.0,
    )
    assert loaded.permissions == ["invoices:read"]


def test_an_unknown_manifest_field_is_refused() -> None:
    with pytest.raises(ManifestError, match="Extra inputs"):
        load_manifest(MANIFEST + "kind: factory\n")


def test_a_blank_model_is_refused() -> None:
    with pytest.raises(ManifestError, match="blank"):
        load_manifest("model: '  '\nsystem_prompt: hi\n")


def test_manifest_errors_do_not_echo_the_input() -> None:
    marker = "do-not-echo-this-value"
    with pytest.raises(ManifestError) as caught:
        load_manifest(f"model: 1\nsystem_prompt: {marker}\nnot_a_field: {marker}\n")
    assert marker not in str(caught.value)


def test_invalid_yaml_is_a_manifest_error() -> None:
    with pytest.raises(ManifestError, match="YAML"):
        load_manifest("model: [unterminated")


def test_a_duplicate_tool_name_in_the_manifest_is_refused() -> None:
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest("model: m\nsystem_prompt: s\ntools: [echo, echo]\n")


def test_from_yaml_path_reads_the_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(MANIFEST, encoding="utf-8")
    builder = AgentBuilder.from_yaml_path(path).with_provider(scripted(ModelResponse.of("ok")))
    builder.with_tool(EchoTool()).with_fence_nonce(NONCE)
    loop = builder.build()
    assert isinstance(loop, AgentLoop)
    assert builder.model == "test-model"


async def test_staged_assembly_builds_a_loop_that_runs() -> None:
    echo = EchoTool()
    builder = (
        AgentBuilder.from_yaml(MANIFEST)
        .with_tool(echo)
        .with_provider(scripted(use_tool(tool_call()), ModelResponse.of("done")))
        .with_fence_nonce(NONCE)
    )
    loop = builder.build()
    result = await loop.run("go", builder.budget())
    assert result.end_reason is EndReason.END_TURN
    assert result.final_text == "done"
    assert [c.text for c in echo.calls] == ["hi"]
    assert echo.contexts[0].identity == Identity("anonymous", frozenset({"invoices:read"}))
    budget = builder.budget()
    assert (budget.max_turns, budget.max_tokens, budget.max_seconds) == (4, 1000, 30.0)


class OtherTool(EchoTool):
    name = "other"
    description = "not in the manifest"


async def test_a_tool_outside_the_manifest_cannot_be_called() -> None:
    echo = EchoTool()
    extra = OtherTool()
    builder = (
        AgentBuilder.from_yaml(MANIFEST)
        .with_tool(echo)
        .with_tool(extra)
        .with_provider(scripted(use_tool(tool_call(name="other")), ModelResponse.of("done")))
        .with_fence_nonce(NONCE)
    )
    result = await builder.build().run("go", Budget())
    assert extra.calls == []
    assert echo.calls == []
    assert result.tool_calls[0].is_error


async def test_an_empty_tools_list_in_the_manifest_includes_none() -> None:
    echo = EchoTool()
    builder = (
        AgentBuilder.from_yaml("model: m\nsystem_prompt: s\ntools: []\n")
        .with_tool(echo)
        .with_provider(scripted(use_tool(tool_call()), ModelResponse.of("done")))
        .with_fence_nonce(NONCE)
    )
    result = await builder.build().run("go", Budget())
    assert echo.calls == []
    assert result.tool_calls[0].is_error


async def test_omitting_tools_from_the_manifest_includes_every_registered_tool() -> None:
    echo = EchoTool()
    builder = (
        AgentBuilder.from_yaml("model: m\nsystem_prompt: s\n")
        .with_tool(echo)
        .with_provider(scripted(use_tool(tool_call()), ModelResponse.of("done")))
        .with_fence_nonce(NONCE)
    )
    result = await builder.build().run("go", Budget())
    assert [c.text for c in echo.calls] == ["hi"]
    assert result.end_reason is EndReason.END_TURN


def test_a_named_tool_that_was_not_registered_fails_at_build() -> None:
    with pytest.raises(ManifestError, match="not registered"):
        AgentBuilder.from_yaml(MANIFEST).with_provider(scripted()).build()


def test_build_requires_a_provider_and_a_system_prompt() -> None:
    with pytest.raises(ManifestError, match="provider"):
        AgentBuilder().with_system_prompt("s").build()
    with pytest.raises(ManifestError, match="system prompt"):
        AgentBuilder().with_provider(scripted()).build()


def test_write_wrapped_in_cached_and_metered_without_a_gate_fails_at_construction() -> None:
    """Eval: Cached(Metered(write)) is a construction error (Cached refuses an ungated write)."""
    write = DeleteTool()
    metered = Metered(write, Metrics())
    with pytest.raises(ValueError, match="permission gate"):
        Cached(metered)
    double = Metered(metered, Metrics())
    with pytest.raises(UngatedWriteError, match="innermost effect"):
        _builder(double).build()


def test_an_effect_write_tool_behind_a_gate_wrapped_in_cached_constructs() -> None:
    """Eval: Cached(PermissionGate(write)) constructs successfully."""
    gated = PermissionGate(DeleteTool())
    cached = Cached(gated)
    loop = _builder(cached).build()
    assert isinstance(loop, AgentLoop)


def test_a_gate_under_three_decorators_is_found() -> None:
    write = DeleteTool()
    retried = Retrying(write, max_attempts=1)
    gated = PermissionGate(retried)
    cached = Cached(gated)
    stacked = Metered(cached, Metrics())
    _builder(stacked).build()


def test_a_write_hidden_under_three_decorators_is_caught() -> None:
    wrapped = PassThrough(DeleteTool())
    retried = Retrying(wrapped, max_attempts=1)
    stacked = Metered(retried, Metrics())
    with pytest.raises(UngatedWriteError, match="innermost effect"):
        _builder(stacked).build()


def test_a_lying_decorator_cannot_smuggle_a_write_past_the_builder() -> None:
    lying = LyingDecorator(DeleteTool())
    with pytest.raises(UngatedWriteError):
        _builder(lying).build()


def test_a_write_claiming_decorator_over_a_read_leaf_still_needs_a_gate() -> None:
    claiming = WriteClaimingDecorator(EchoTool())
    with pytest.raises(UngatedWriteError):
        _builder(claiming).build()


def test_builder_is_not_a_factory() -> None:
    """The product is always AgentLoop; stages fill it in rather than selecting a kind."""
    a = _builder(EchoTool()).build()
    b = (
        AgentBuilder.from_yaml(MANIFEST)
        .with_tool(EchoTool())
        .with_provider(scripted(ModelResponse.of("ok")))
        .with_fence_nonce(NONCE)
        .build()
    )
    assert type(a) is AgentLoop and type(b) is AgentLoop


def test_duplicate_registered_tools_are_refused() -> None:
    with pytest.raises(ValueError, match="two tools"):
        _builder(EchoTool()).with_tool(EchoTool())


async def test_subscribers_on_the_builder_bus_cannot_change_the_run() -> None:
    class Boom:
        async def on_event(self, event: LoopEvent) -> None:
            raise RuntimeError("subscriber bug")

    builder = (
        AgentBuilder()
        .with_model("m")
        .with_system_prompt("s")
        .with_tool(EchoTool())
        .with_provider(scripted(use_tool(tool_call()), ModelResponse.of("same")))
        .with_subscriber(Boom())
        .with_fence_nonce(NONCE)
    )
    result = await builder.build().run("go", Budget())
    assert result.end_reason is EndReason.END_TURN
    assert result.final_text == "same"


async def test_with_json_log_writes_lines_and_not_the_prompt() -> None:
    stream = io.StringIO()
    marker = "do-not-log-this-value"
    builder = (
        AgentBuilder()
        .with_model("m")
        .with_system_prompt("s")
        .with_tool(EchoTool())
        .with_provider(scripted(ModelResponse.of("ok")))
        .with_json_log(stream)
        .with_fence_nonce(NONCE)
    )
    await builder.build().run(marker, Budget())
    dumped = stream.getvalue()
    assert dumped
    assert marker not in dumped
    assert "model_called" in dumped


async def test_replacing_the_event_bus_is_what_the_loop_sees() -> None:
    bus = EventBus()
    seen: list[LoopEvent] = []

    class Collect:
        async def on_event(self, event: LoopEvent) -> None:
            seen.append(event)

    bus.subscribe(Collect())
    builder = (
        AgentBuilder()
        .with_model("m")
        .with_system_prompt("s")
        .with_provider(scripted(ModelResponse.of("ok")))
        .with_event_bus(bus)
        .with_fence_nonce(NONCE)
    )
    await builder.build().run("go", Budget())
    assert seen
    with pytest.raises(FrozenInstanceError):
        seen[0].detail = "mutated"  # type: ignore[misc]


def test_with_limits_overrides_the_manifest() -> None:
    builder = AgentBuilder.from_yaml(MANIFEST).with_limits(max_turns=2)
    budget = builder.budget()
    assert budget.max_turns == 2
    assert budget.max_tokens == 1000


def test_negative_limits_are_refused() -> None:
    with pytest.raises(ManifestError):
        AgentBuilder().with_limits(max_turns=-1)


async def test_with_identity_is_what_the_tool_sees() -> None:
    echo = EchoTool()
    who = Identity("alice", frozenset({"admin"}))
    builder = (
        AgentBuilder.from_yaml(MANIFEST)
        .with_tool(echo)
        .with_identity(who)
        .with_provider(scripted(use_tool(tool_call()), ModelResponse.of("ok")))
        .with_fence_nonce(NONCE)
    )
    await builder.build().run("go", Budget())
    assert echo.contexts[0].identity == who


def test_a_string_effect_write_tool_is_refused_by_the_builder() -> None:
    """`Effect` is a `StrEnum`, so the plain string "write" compares equal but is not identical.

    On PR #9 `effect is Effect.WRITE` was False for exactly this value and FAILED OPEN: an
    ungated write tool ran. `tests/support.py` has carried the fixture since, but only the
    loop's constructor was pinned against it -- the builder's path was correct solely because
    `require_write_gate` delegates to `core.tool`. Pin it here too, so a later slice that stops
    delegating fails rather than silently reopening the hole.
    """
    with pytest.raises(UngatedWriteError, match="innermost effect"):
        # mypy --strict rejects this call too, which is the second defence; the ignore is here
        # because the first defence -- the runtime check -- is what this test exists to pin.
        _builder(StringEffectTool()).build()  # type: ignore[arg-type]


def test_a_string_effect_write_tool_behind_a_gate_constructs() -> None:
    """The mirror: refusing every string-effect tool would be a different bug."""
    gated = PermissionGate(StringEffectTool())
    loop = _builder(gated).build()
    assert loop is not None
