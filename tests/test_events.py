"""Tests for the context-local event sink used to stream live eval events.

The pure evaluation code (pack_runner) emits via this sink; the engine installs a
sink that appends to live.jsonl. With no sink installed, emit is a silent no-op so
unit tests and library use stay decoupled from any UI.
"""

from __future__ import annotations

from gguf_limit_bench import events


def test_emit_is_noop_without_sink():
    events.emit("question_started", {"q_id": "q1"})  # must not raise


def test_emit_routes_to_installed_sink():
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append((t, d))):
        events.emit("question_started", {"q_id": "q1"})
    assert seen == [("question_started", {"q_id": "q1"})]


def test_sink_is_restored_after_context():
    seen: list = []
    with events.set_event_sink(lambda t, d: seen.append(t)):
        events.emit("a", {})
    events.emit("b", {})  # outside the context -> no sink
    assert seen == ["a"]


def test_broken_sink_never_breaks_evaluation():
    def bad(t, d):
        raise RuntimeError("ui exploded")

    with events.set_event_sink(bad):
        events.emit("a", {})  # must be swallowed
