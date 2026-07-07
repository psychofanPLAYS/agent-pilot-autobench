"""Tests for cross-platform process-group isolation and kill-tree helpers.

These guard the #1 hygiene gap: a llama-server (GPU-holding) child must never be
orphaned when the engine stops or crashes. The child is spawned in its own
process group so the whole tree can be killed deterministically.
"""

from __future__ import annotations

from gguf_limit_bench import server_probe


def test_process_group_kwargs_windows(monkeypatch):
    monkeypatch.setattr(server_probe, "_is_windows", lambda: True)
    kwargs = server_probe.process_group_kwargs()
    assert kwargs == {"creationflags": server_probe._CREATE_NEW_PROCESS_GROUP}


def test_process_group_kwargs_posix(monkeypatch):
    monkeypatch.setattr(server_probe, "_is_windows", lambda: False)
    assert server_probe.process_group_kwargs() == {"start_new_session": True}


def test_detached_process_kwargs_windows_requests_job_breakaway(monkeypatch):
    monkeypatch.setattr(server_probe, "_is_windows", lambda: True)
    kwargs = server_probe.detached_process_kwargs()
    flags = kwargs["creationflags"]
    assert flags & server_probe._CREATE_NEW_PROCESS_GROUP
    assert flags & server_probe._CREATE_BREAKAWAY_FROM_JOB
    assert flags & server_probe._DETACHED_PROCESS


def test_detached_process_kwargs_posix(monkeypatch):
    monkeypatch.setattr(server_probe, "_is_windows", lambda: False)
    assert server_probe.detached_process_kwargs() == {"start_new_session": True}


def test_kill_process_tree_noop_when_already_exited(monkeypatch):
    calls: list = []
    monkeypatch.setattr(server_probe.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    class Dead:
        pid = 111

        def poll(self):
            return 0  # already exited

    server_probe.kill_process_tree(Dead())
    assert calls == []  # must not try to kill an already-dead process


def test_kill_process_tree_windows_uses_taskkill_tree(monkeypatch):
    monkeypatch.setattr(server_probe, "_is_windows", lambda: True)
    calls: list = []
    monkeypatch.setattr(server_probe.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    class Live:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    server_probe.kill_process_tree(Live())
    flat = " ".join(str(c[0]) for c in calls)
    assert "taskkill" in flat
    assert "/T" in flat and "/F" in flat
    assert "4321" in flat


def test_kill_pid_tree_windows_uses_taskkill_tree(monkeypatch):
    monkeypatch.setattr(server_probe, "_is_windows", lambda: True)
    calls: list = []
    monkeypatch.setattr(server_probe.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    server_probe.kill_pid_tree(7654)
    flat = " ".join(str(c[0]) for c in calls)
    assert "taskkill" in flat
    assert "/T" in flat and "/F" in flat
    assert "7654" in flat


def test_stop_process_delegates_to_kill_tree(monkeypatch):
    seen: list = []

    class Live:
        pid = 7

        def poll(self):
            return None

    monkeypatch.setattr(server_probe, "kill_process_tree", lambda p: seen.append(p))
    server_probe._stop_process(Live())
    assert len(seen) == 1
