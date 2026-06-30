"""Tests for the run-directory contract: spec, status/heartbeat, control, lock.

The run directory is the only seam between the thin web UI and the detached
engine. These functions are the read/write surface for both sides.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from gguf_limit_bench import run_dir


def test_spec_roundtrip(tmp_path):
    spec = {"models": ["a.gguf", "b.gguf"], "mode": "librarian_bench", "options": {"budget_minutes": 5}}
    run_dir.write_spec(tmp_path, spec)
    assert run_dir.read_spec(tmp_path) == spec


def test_status_roundtrip_includes_heartbeat(tmp_path):
    run_dir.write_status(
        tmp_path, phase="running", model="a", model_index=1, model_total=2, pid=999
    )
    status = run_dir.read_status(tmp_path)
    assert status["phase"] == "running"
    assert status["model"] == "a"
    assert status["model_index"] == 1
    assert status["model_total"] == 2
    assert status["pid"] == 999
    assert "alive_at" in status


def test_read_status_missing_returns_empty(tmp_path):
    assert run_dir.read_status(tmp_path) == {}


def test_control_defaults_to_run(tmp_path):
    assert run_dir.read_control(tmp_path)["action"] == "run"


def test_control_write_then_read(tmp_path):
    run_dir.write_control(tmp_path, "stop")
    control = run_dir.read_control(tmp_path)
    assert control["action"] == "stop"
    assert "requested_at" in control


def test_engine_is_alive_fresh_vs_stale():
    now = datetime(2026, 1, 1, 12, 0, 0)
    fresh = {"alive_at": (now - timedelta(seconds=3)).isoformat()}
    stale = {"alive_at": (now - timedelta(seconds=30)).isoformat()}
    assert run_dir.engine_is_alive(fresh, now=now) is True
    assert run_dir.engine_is_alive(stale, now=now) is False


def test_engine_is_alive_no_status():
    assert run_dir.engine_is_alive({}, now=datetime(2026, 1, 1)) is False


def test_lock_is_exclusive(tmp_path):
    assert run_dir.acquire_lock(tmp_path, pid=1) is True
    assert run_dir.acquire_lock(tmp_path, pid=2) is False
    run_dir.release_lock(tmp_path)
    assert run_dir.acquire_lock(tmp_path, pid=3) is True


def test_atomic_writes_leave_no_temp(tmp_path):
    run_dir.write_spec(tmp_path, {"models": [], "mode": "x", "options": {}})
    run_dir.write_status(tmp_path, phase="running")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
