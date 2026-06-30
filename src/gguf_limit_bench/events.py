"""Context-local event sink for streaming live evaluation events.

Pure evaluation code calls :func:`emit`; if the engine has installed a sink via
:func:`set_event_sink`, the event is delivered (e.g. appended to ``live.jsonl``).
With no sink installed, :func:`emit` is a silent no-op, so the library and unit
tests stay fully decoupled from any UI. A sink that raises never breaks the run.

A ``contextvars.ContextVar`` is used (not a global) so the sink is scoped to the
call that installed it. The evaluation stack is synchronous and single-threaded
within the engine process, so the sink propagates down the call chain.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Iterator

EventSink = Callable[[str, dict], None]

_sink: contextvars.ContextVar[EventSink | None] = contextvars.ContextVar(
    "gguf_event_sink", default=None
)


def emit(event_type: str, data: dict) -> None:
    """Deliver an event to the installed sink, if any. Never raises."""
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(event_type, data)
    except Exception:  # noqa: BLE001 - a broken sink must never break evaluation
        pass


@contextlib.contextmanager
def set_event_sink(sink: EventSink) -> Iterator[None]:
    """Install *sink* for the duration of the context, then restore the previous."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def current_sink() -> EventSink | None:
    return _sink.get()
