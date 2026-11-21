"""JSON-lines run logger.

One logger, six fields: run id, turn, tool, outcome, tokens, duration. A subscriber
on the event bus writes one line per event. It never logs secrets, credentials, or
prompts: only the fields the bus already carries, never `detail` (a stop reason is
safe today; a future caller stuffing a prompt into it would not be).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TextIO, TypedDict

from agent_container.core.loop import LoopEvent

EVENTS_LOGGER = "agent_container.events"

# The schema. Tests pin this tuple so a field cannot be added silently.
LOG_FIELDS: tuple[str, ...] = ("run_id", "turn", "tool", "outcome", "tokens", "duration")


class LogLine(TypedDict):
    run_id: str
    turn: int
    tool: str | None
    outcome: str
    tokens: int
    duration: float


def format_log_line(payload: LogLine) -> str:
    """One JSON object, keys in `LOG_FIELDS` order, nothing else."""
    ordered = {
        "run_id": payload["run_id"],
        "turn": payload["turn"],
        "tool": payload["tool"],
        "outcome": payload["outcome"],
        "tokens": payload["tokens"],
        "duration": payload["duration"],
    }
    return json.dumps(ordered, separators=(",", ":"))


class JsonLinesFormatter(logging.Formatter):
    """Write the log record's message and nothing else: the subscriber already JSON-encoded it."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def configure_json_lines_logging(
    stream: TextIO | None = None,
    *,
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Attach a message-only handler so each info record is one JSON line.

    Safe to call more than once for the same stream: a second call does not add another
    handler. `propagate` is turned off so a root formatter cannot wrap the line in a
    timestamp and break JSON-lines.
    """
    log = logger or logging.getLogger(EVENTS_LOGGER)
    target = sys.stderr if stream is None else stream
    for handler in log.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is target:
            log.setLevel(logging.INFO)
            log.propagate = False
            return log
    attached = logging.StreamHandler(target)
    attached.setFormatter(JsonLinesFormatter())
    log.addHandler(attached)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


class JsonLinesSubscriber:
    """A bus subscriber that writes one JSON line per event.

    `duration` is seconds since this subscriber first saw the run, measured with `clock`.
    The loop does not put a duration on the event; this is the honest figure we can
    produce without editing `core/loop.py`. Bounded to `max_runs` concurrent run ids so
    a long-lived process cannot grow the map without bound.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_runs: int = 4096,
    ) -> None:
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        self._stream = stream
        self._logger = logger or logging.getLogger(EVENTS_LOGGER)
        self._clock = clock
        self._max_runs = max_runs
        self._started: OrderedDict[str, float] = OrderedDict()

    def _line_for(self, event: LoopEvent) -> str:
        now = self._clock()
        started = self._started.get(event.run_id)
        if started is None:
            self._started[event.run_id] = now
            duration = 0.0
            while len(self._started) > self._max_runs:
                self._started.popitem(last=False)
        else:
            duration = max(0.0, now - started)
            self._started.move_to_end(event.run_id)
        payload: LogLine = {
            "run_id": event.run_id,
            "turn": event.turn,
            "tool": event.tool,
            "outcome": event.kind.value,
            "tokens": event.input_tokens + event.output_tokens,
            "duration": duration,
        }
        return format_log_line(payload)

    async def on_event(self, event: LoopEvent) -> None:
        line = self._line_for(event)
        if self._stream is not None:
            self._stream.write(line + "\n")
            self._stream.flush()
            return
        # `%s` so a JSON payload containing `%` cannot become a formatting error, and
        # cannot pull extra args off the logging call.
        self._logger.info("%s", line)


__all__ = [
    "EVENTS_LOGGER",
    "LOG_FIELDS",
    "JsonLinesFormatter",
    "JsonLinesSubscriber",
    "LogLine",
    "configure_json_lines_logging",
    "format_log_line",
]
