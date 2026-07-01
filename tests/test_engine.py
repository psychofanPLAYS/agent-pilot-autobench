"""Tests for the sequential engine runner.

The engine reads run-spec.json, runs each model sequentially via an injected
``run_model`` callback (no GPU in tests), maintains the status heartbeat, obeys
control.json between models, and always releases its lock.
"""

from __future__ import annotations

import pytest

from gguf_limit_bench import engine, run_dir


def _spec(tmp_path, models):
    run_dir.write_spec(tmp_path, {"models": models, "mode": "librarian_bench", "options": {"k": 1}})


def test_engine_runs_models_sequentially(tmp_path):
    _spec(tmp_path, ["a", "b"])
    seen = []

    def fake_run(model, options, emit):
        seen.append((model, options["k"]))
        return tmp_path

    engine.run_engine(tmp_path, fake_run)

    assert seen == [("a", 1), ("b", 1)]
    assert run_dir.read_status(tmp_path)["phase"] == "complete"


def test_engine_honors_stop_between_models(tmp_path):
    _spec(tmp_path, ["a", "b"])
    seen = []

    def fake_run(model, options, emit):
        seen.append(model)
        run_dir.write_control(tmp_path, "stop")  # ask to stop after the first
        return tmp_path

    engine.run_engine(tmp_path, fake_run)

    assert seen == ["a"]  # second model never started
    assert run_dir.read_status(tmp_path)["phase"] == "stopped"


def test_engine_marks_failed_on_exception(tmp_path):
    _spec(tmp_path, ["a"])

    def boom(model, options, emit):
        raise RuntimeError("llama exploded")

    with pytest.raises(RuntimeError):
        engine.run_engine(tmp_path, boom)

    assert run_dir.read_status(tmp_path)["phase"] == "failed"


def test_engine_emits_model_lifecycle_events(tmp_path):
    _spec(tmp_path, ["a"])

    def fake_run(model, options, emit):
        emit("question_scored", {"q_id": "q1", "score": 1.0})
        return tmp_path

    engine.run_engine(tmp_path, fake_run)

    live = (tmp_path / run_dir.LIVE_FILE).read_text(encoding="utf-8")
    assert "model_started" in live
    assert "question_scored" in live
    assert "model_finished" in live


def test_engine_appends_running_score_as_questions_land(tmp_path):
    import json

    _spec(tmp_path, ["a"])

    def fake_run(model, options, emit):
        emit("question_scored", {"q_id": "q1", "correct": True})
        emit("question_scored", {"q_id": "q2", "correct": False})
        return tmp_path

    engine.run_engine(tmp_path, fake_run)

    rows = [
        json.loads(line)
        for line in (tmp_path / run_dir.LIVE_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    running = [r for r in rows if r["type"] == "running_score"]
    assert running[-1]["data"] == {"answered": 2, "correct": 1, "quality_0_100": 50.0}


def test_engine_installs_event_sink_for_pure_emit(tmp_path):
    from gguf_limit_bench import events

    _spec(tmp_path, ["a"])

    def fake_run(model, options, emit):
        # pure evaluation code (e.g. pack_runner) emits via the contextvar sink,
        # NOT the emit param; the engine must route it to live.jsonl
        events.emit("question_started", {"q_id": "q9"})
        return tmp_path

    engine.run_engine(tmp_path, fake_run)

    live = (tmp_path / run_dir.LIVE_FILE).read_text(encoding="utf-8")
    assert "question_started" in live and "q9" in live


def test_engine_releases_lock(tmp_path):
    _spec(tmp_path, ["a"])
    engine.run_engine(tmp_path, lambda m, o, e: tmp_path)
    # lock is free again -> a fresh acquire succeeds
    assert run_dir.acquire_lock(tmp_path, pid=12345) is True


def test_engine_heartbeat_refreshes_status_during_a_long_model(tmp_path, monkeypatch):
    """A slow model with no events must still get a fresh status heartbeat, so the
    web UI never mistakes a live run for a crashed one on refresh/reattach."""
    import time

    _spec(tmp_path, ["a"])
    writes: list = []
    real_write = run_dir.write_status

    def spy(*args, **kwargs):
        writes.append(kwargs.get("phase"))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(engine.run_dir, "write_status", spy)

    def fake_run(model, options, emit):
        # block (emitting nothing) until the heartbeat thread has ticked a few times
        deadline = time.time() + 2.0
        while len(writes) < 3 and time.time() < deadline:
            time.sleep(0.02)
        return tmp_path

    engine.run_engine(tmp_path, fake_run, heartbeat_seconds=0.05)

    # initial status + heartbeat tick(s) during the model + final write
    assert len(writes) >= 3
    assert run_dir.read_status(tmp_path)["phase"] == "complete"
