"""AgentBuilder: YAML manifest, staged assembly, and the construction-time gate rule."""

from __future__ import annotations

import ast
import io
import traceback
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
from agent_container.core.builder import _INDEXED_FIELDS
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
    with pytest.raises(ManifestError, match="is not a known field"):
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


def test_a_yaml_syntax_error_does_not_echo_the_manifest() -> None:
    """`load_manifest` promises values are not echoed. `str(YAMLError)` embeds a source excerpt.

    A malformed manifest leaked whatever sat near the error -- here a system prompt. The
    message keeps the parser's own problem description and position. `problem` is safe for
    whole VALUES -- what the promise covers -- though a few of PyYAML's problem strings do
    repeat a YAML identifier, and the scanner echoes one character from inside a quoted scalar.
    """
    marker = "MARKER-do-not-echo-abc123"
    with pytest.raises(ManifestError) as caught:
        load_manifest(f'system_prompt: "{marker}" trailing-token\n')
    message = str(caught.value)
    assert marker not in message, f"manifest text leaked into the error: {message!r}"
    assert "line" in message and "column" in message, message
    # The chain too: a YAMLError stringifies to a snippet of the document.
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "document",
    [
        "system_prompt: !!int {marker}",
        "system_prompt: !!float {marker}",
        "system_prompt: !!bool {marker}",
        "system_prompt: !!timestamp {marker}",
    ],
)
def test_a_tagged_scalar_does_not_echo_the_manifest(document: str) -> None:
    """`safe_load` does not raise only `YAMLError`.

    `SafeConstructor`'s scalar conversions raise plain builtins with the value interpolated,
    so catching `YAMLError` alone let the exact case the docstring opens with straight through:
    a `system_prompt` with a credential in it, returned in a `ValueError`.
    """
    marker = "MARKER-do-not-echo-abc123"
    with pytest.raises(ManifestError) as caught:
        load_manifest(document.format(marker=marker) + "\n")
    message = str(caught.value)
    assert marker not in message, f"manifest text leaked: {message!r}"
    assert marker.lower() not in message.lower(), f"manifest text leaked: {message!r}"
    # The cause is suppressed too, or the value returns in the traceback (PR #12's finding).
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "document",
    [
        'system_prompt: "{marker}"',  # model missing -> "Field required"
        'model: [1, 2]\nsystem_prompt: "{marker}"',  # wrong type
        'model: m\nsystem_prompt: "{marker}"\ntools: not-a-list',
    ],
)
def test_a_validation_error_does_not_echo_the_manifest(document: str) -> None:
    """The ordinary error path leaked too, and by a different route than the YAML one.

    `include_input=False` sanitizes the string we extract from `exc.errors()` and does nothing
    to the `ValidationError`, so `from exc` re-attached the whole input: `str(exc)` renders
    `input_value={'system_prompt': '<credential>'}`. Pydantic truncates to roughly 25 head and
    25 tail characters, so a short secret comes back verbatim. This is not a hostile-document
    path -- any missing field or typo reaches it.
    """
    marker = "MARKER-do-not-echo-abc123"
    with pytest.raises(ManifestError) as caught:
        load_manifest(document.format(marker=marker) + "\n")
    message = str(caught.value)
    assert marker not in message, f"manifest text leaked: {message!r}"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert marker not in rendered, "manifest text leaked into the rendered traceback"


def test_invalid_limits_do_not_echo_the_values() -> None:
    """`with_limits` had the same `from exc` shape, with the rejected numbers in the chain."""
    with pytest.raises(ManifestError) as caught:
        AgentBuilder().with_limits(max_turns=-99_777_333)
    assert "99_777_333".replace("_", "") not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


# Every position a secret can occupy in a manifest. `loc` carries unknown KEYS and `msg` carries
# whatever a validator interpolated, and `include_input=False` covers neither -- so round 3 of
# this leak was in `str(ManifestError)` itself, reaching any caller's log with no traceback
# render needed. Unlike `input_value`, `loc` is not truncated at all: a 20000-character key was
# measured arriving verbatim.
_MARKER = "MARKERsecretabc123"
_LONG_KEY = "Z" * 20_000


@pytest.mark.parametrize(
    ("label", "document"),
    [
        ("unknown key", f"model: m\nsystem_prompt: p\n{_MARKER}: 1\n"),
        ("unknown key, untruncated", f"model: m\nsystem_prompt: p\n? >-\n  {_LONG_KEY}\n: 1\n"),
        ("bad tool name", f'model: m\nsystem_prompt: p\ntools: ["{_MARKER} bad!"]\n'),
        ("duplicate tool name", f'model: m\nsystem_prompt: p\ntools: ["{_MARKER}", "{_MARKER}"]\n'),
        ("unknown nested key", f"model: m\nsystem_prompt: p\nlimits: {{{_MARKER}: 2}}\n"),
        ("missing field", f'system_prompt: "{_MARKER}"\n'),
        ("wrong type", f'model: [1, 2]\nsystem_prompt: "{_MARKER}"\n'),
    ],
)
def test_the_error_message_itself_does_not_echo_the_manifest(label: str, document: str) -> None:
    with pytest.raises(ManifestError) as caught:
        load_manifest(document)
    message = str(caught.value)
    assert _MARKER not in message, f"{label}: leaked into the message: {message!r}"
    assert _LONG_KEY[:200] not in message, f"{label}: leaked an unknown key"
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert _MARKER not in rendered, f"{label}: leaked into the rendered traceback"
    # Bounded: the pre-fix message for the long-key case measured 20050 characters.
    assert len(message) < 500, f"{label}: message is {len(message)} characters"


def test_our_own_validators_carry_no_values() -> None:
    """A validator that interpolates the value puts it straight into `msg`, which ours did.

    `_describe_error` no longer passes `msg` through, so a careless validator added later
    cannot reopen the hole -- but the validators are kept value-free as well, because their
    `ValueError` is visible to anyone who validates a `Manifest` directly.

    Parsed rather than grepped. The first version of this test searched a positional window
    for `raise ValueError(f...`, which held every validator only by accident and caught just
    one of the eight ways to get a value into a string: `"x" + value`, `.format()`, `%`,
    `str(value)`, implicit concatenation and a trailing comment all walked past it.
    """
    tree = ast.parse(Path("agent_container/core/builder.py").read_text())
    validators = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            (isinstance(d, ast.Call) and getattr(d.func, "id", None) == "field_validator")
            or getattr(d, "id", None) == "field_validator"
            for d in node.decorator_list
        )
    ]
    # Names, not a count: this also fails if a validator is renamed or dropped, rather than
    # quietly checking fewer functions than it thinks.
    assert {v.name for v in validators} == {"not_blank", "tool_names_are_identifiers"}, (
        f"validators changed: {sorted(v.name for v in validators)}"
    )
    for validator in validators:
        for node in ast.walk(validator):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            raised = node.exc
            if not (
                isinstance(raised, ast.Call) and getattr(raised.func, "id", None) == "ValueError"
            ):
                continue
            assert len(raised.args) == 1, f"{validator.name}: expected one argument"
            argument = raised.args[0]
            assert isinstance(argument, ast.Constant) and isinstance(argument.value, str), (
                f"{validator.name} builds its message at runtime "
                f"({type(argument).__name__}), so a value can reach it"
            )


def test_every_list_valued_field_is_indexable() -> None:
    """An integer `loc` segment is an index only under a list field; elsewhere it is a KEY.

    Pydantic keeps a mapping key as an int whenever it fits an i64, and `loc` does not say
    which is which -- so `4111111111111111: x` rendered verbatim while every test passed. The
    set is derived from the annotations; this pins what that derivation currently finds, so
    adding a list field to the schema fails here rather than silently redacting its indices.
    """
    assert frozenset({"tools", "permissions"}) == _INDEXED_FIELDS


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("card-shaped", "4111111111111111"),
        ("19 digits", "1234567890123456789"),
        ("negative", "-4111111111111111"),
        ("hex", "0xDEADBEEF"),
        ("boolean", "true"),
    ],
)
def test_a_numeric_unknown_key_is_redacted(label: str, key: str) -> None:
    """These arrive in `loc` as `int`, not `str`, so `_KNOWN_FIELDS` never sees them."""
    with pytest.raises(ManifestError) as caught:
        load_manifest(f"model: m\nsystem_prompt: p\n{key}: x\n")
    message = str(caught.value)
    assert key not in message, f"{label}: numeric key leaked: {message!r}"
    assert key.lstrip("-0x") not in message, f"{label}: numeric key leaked: {message!r}"
    assert "<redacted>" in message, message


def test_a_numeric_key_nested_under_a_known_field_is_redacted() -> None:
    with pytest.raises(ManifestError) as caught:
        load_manifest("model: m\nsystem_prompt: p\nlimits: {4111111111111111: x}\n")
    assert "4111111111111111" not in str(caught.value), str(caught.value)
    assert "limits.<redacted>" in str(caught.value), str(caught.value)


@pytest.mark.parametrize("field", ["tools", "permissions"])
def test_a_real_list_index_is_still_reported(field: str) -> None:
    """The redaction must not cost the diagnostic the int branch existed for."""
    with pytest.raises(ManifestError) as caught:
        load_manifest(f"model: m\nsystem_prompt: p\n{field}: [1]\n")
    assert f"{field}.0" in str(caught.value), str(caught.value)


def test_unregistered_tool_names_are_not_echoed() -> None:
    """The missing names are document content by definition; the registry's are the caller's."""
    with pytest.raises(ManifestError) as caught:
        AgentBuilder().with_manifest(
            'model: m\nsystem_prompt: p\ntools: ["nosuchtool"]\n'
        ).with_provider(scripted()).with_tools([EchoTool()]).build()
    assert "nosuchtool" not in str(caught.value), str(caught.value)
    assert "registered tools" in str(caught.value), str(caught.value)
