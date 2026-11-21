"""Exceptions the core raises. Everything else the loop reports as data in a `RunResult`."""


class AgentContainerError(Exception):
    """Base class for errors raised by the core."""


class UnrecognizedStopReason(AgentContainerError, ValueError):
    """A model reply carried a stop reason the loop has no explicit handling for.

    Raised rather than defaulted: continuing, or ending, on a value nobody decided about is how a
    loop ends up deciding from the text instead.
    """

    def __init__(self, value: str) -> None:
        super().__init__(f"unrecognized stop reason: {value!r}")
        self.value = value


class MalformedModelResponse(AgentContainerError, ValueError):
    """A reply contradicted itself, for example `tool_use` with no tool call in it."""


class InputTooLarge(AgentContainerError, ValueError):
    """The user message is empty or longer than the 8,000 character cap."""


class BudgetAlreadyStarted(AgentContainerError, RuntimeError):
    """A `Budget` was passed to a second run. A budget is created once per run."""


class ReplayExhausted(AgentContainerError, RuntimeError):
    """A `ReplayProvider` was asked for more replies than it was given."""


class ReplayFormatError(AgentContainerError, ValueError):
    """A replay fixture is not in the recorded-reply format."""


class StagedChangeError(AgentContainerError, RuntimeError):
    """A staged change could not be applied: it was applied before, or the tool failed."""
