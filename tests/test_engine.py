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


def test_engine_releases_lock(tmp_path):
    _spec(tmp_path, ["a"])
    engine.run_engine(tmp_path, lambda m, o, e: tmp_path)
    # lock is free again -> a fresh acquire succeeds
    assert run_dir.acquire_lock(tmp_path, pid=12345) is True
