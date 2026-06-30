# Cockpit Phase 1 — hygiene backbone implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans or
> subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the benchmark engine a properly isolated, externally-controllable
process with impeccable lifecycle hygiene, so the web UI can become a thin client
that launches/controls/renders without ever running evaluation in-process.

**Architecture:** A new `engine` process reads a run directory's `run-spec.json`,
runs models sequentially (wrapping the existing `_run_one_autoresearch`), writes a
`status.json` heartbeat + appends to `live.jsonl`, and obeys `control.json`
(stop/abort). llama-server is always spawned in its own process group and killed
as a tree. The web server spawns this engine detached instead of using a daemon
thread.

**Tech stack:** Python 3.13, stdlib `subprocess`/`signal`/`atexit`, Typer CLI,
pytest. Target OS Windows 11 (cross-platform).

Spec: `docs/superpowers/specs/2026-06-30-inflight-cockpit-design.md` §3, §5.

---

## File structure

- `src/gguf_limit_bench/server_probe.py` (modify) — add `process_group_kwargs()`
  and `kill_process_tree()`; route `_stop_process` and the 3 Popen sites through
  them.
- `src/gguf_limit_bench/server_session.py`, `simple_bench_runner.py` (modify) —
  use `process_group_kwargs()` at their Popen sites.
- `src/gguf_limit_bench/run_dir.py` (create) — run-directory contract: spec,
  status/heartbeat, control, PID lock; pure + filesystem.
- `src/gguf_limit_bench/engine.py` (create) — sequential engine runner +
  signal/atexit cleanup; takes an injectable `run_model` callback.
- `src/gguf_limit_bench/cli.py` (modify) — add `engine` command (`--run-dir`).
- `src/gguf_limit_bench/webui.py` (modify) — `start_run` writes spec + spawns
  engine detached; stop/abort write control; `state_payload` reads status; drop
  the daemon-thread runner.
- Tests under `tests/` mirroring each module.

---

## Task 1: Process-group isolation + kill-tree

**Files:** Modify `src/gguf_limit_bench/server_probe.py`; Test
`tests/test_process_tree.py`.

- [ ] **Step 1 — failing tests**

```python
import subprocess, sys
from gguf_limit_bench import server_probe

def test_process_group_kwargs_windows(monkeypatch):
    monkeypatch.setattr(server_probe.os, "name", "nt")
    kw = server_probe.process_group_kwargs()
    assert kw == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

def test_process_group_kwargs_posix(monkeypatch):
    monkeypatch.setattr(server_probe.os, "name", "posix")
    assert server_probe.process_group_kwargs() == {"start_new_session": True}

def test_kill_process_tree_windows_uses_taskkill(monkeypatch):
    monkeypatch.setattr(server_probe.os, "name", "nt")
    calls = []
    monkeypatch.setattr(server_probe.subprocess, "run",
                        lambda *a, **k: calls.append((a, k)))
    class P:
        pid = 4321
        def poll(self): return None
        def wait(self, timeout=None): return 0
    server_probe.kill_process_tree(P())
    assert any("taskkill" in str(c[0]) for c in calls)
    assert any("4321" in str(c[0]) for c in calls)
```

- [ ] **Step 2 — run, expect FAIL** (`pytest tests/test_process_tree.py -v`):
  AttributeError: no `process_group_kwargs` / `kill_process_tree`.

- [ ] **Step 3 — implement** in `server_probe.py` (near `_stop_process`):

```python
import os

def process_group_kwargs() -> dict:
    """Popen kwargs that put the child in its own process group/session."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}

def kill_process_tree(process: "subprocess.Popen") -> None:
    """Terminate a process and all its children. Best-effort, cross-platform."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, PermissionError):
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
```

Add `import signal` if absent. Then make `_stop_process` delegate:

```python
def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    kill_process_tree(process)
```

- [ ] **Step 4 — apply group kwargs at the 3 Popen sites**: in
  `server_probe.py:~159`, `server_session.py:~93`, `simple_bench_runner.py:~95`
  add `**process_group_kwargs()` to the `subprocess.Popen(...)` call (import the
  helper in the latter two from `server_probe`).

- [ ] **Step 5 — run tests + full suite** (`pytest tests/test_process_tree.py
  -v` PASS; then `pytest -q`). Expected: green.

- [ ] **Step 6 — commit** `feat(hygiene): process-group isolation + kill-tree for llama-server`.

---

## Task 2: Run-directory contract (`run_dir.py`)

**Files:** Create `src/gguf_limit_bench/run_dir.py`; Test `tests/test_run_dir.py`.

Responsibilities: write/read `run-spec.json`, `status.json` (with `alive_at`),
`control.json`; acquire/release a single-writer lock (`engine.lock` + pid);
`engine_is_alive(status, now, stale_seconds=10)`.

- [ ] **Step 1 — failing tests**

```python
from datetime import datetime, timedelta
from gguf_limit_bench import run_dir

def test_spec_roundtrip(tmp_path):
    spec = {"models": ["a.gguf"], "mode": "librarian_bench", "options": {}}
    run_dir.write_spec(tmp_path, spec)
    assert run_dir.read_spec(tmp_path) == spec

def test_status_roundtrip(tmp_path):
    run_dir.write_status(tmp_path, phase="running", model="a", model_index=1,
                         model_total=2, pid=999)
    s = run_dir.read_status(tmp_path)
    assert s["phase"] == "running" and s["pid"] == 999 and "alive_at" in s

def test_control_default_and_set(tmp_path):
    assert run_dir.read_control(tmp_path)["action"] == "run"
    run_dir.write_control(tmp_path, "stop")
    assert run_dir.read_control(tmp_path)["action"] == "stop"

def test_engine_alive_staleness():
    now = datetime(2026,1,1,12,0,0)
    fresh = {"alive_at": now.isoformat()}
    stale = {"alive_at": (now - timedelta(seconds=30)).isoformat()}
    assert run_dir.engine_is_alive(fresh, now=now)
    assert not run_dir.engine_is_alive(stale, now=now)

def test_lock_is_exclusive(tmp_path):
    assert run_dir.acquire_lock(tmp_path, pid=1) is True
    assert run_dir.acquire_lock(tmp_path, pid=2) is False
    run_dir.release_lock(tmp_path)
    assert run_dir.acquire_lock(tmp_path, pid=3) is True
```

- [ ] **Step 2 — run, expect FAIL** (module missing).

- [ ] **Step 3 — implement** `run_dir.py` with atomic writes (write temp +
  `os.replace`), ISO timestamps via an injectable `now` (default
  `datetime.now`), and a lockfile that stores the pid; `acquire_lock` returns
  False if the lock exists and its pid is still alive, True (and overwrites) if
  stale/absent. `engine_is_alive(status, now, stale_seconds=10)` parses
  `alive_at` and compares.

- [ ] **Step 4 — run tests + suite.** Expected: green.

- [ ] **Step 5 — commit** `feat(engine): run-directory contract (spec/status/control/lock)`.

---

## Task 3: Engine runner (`engine.py`)

**Files:** Create `src/gguf_limit_bench/engine.py`; Test `tests/test_engine.py`.

`run_engine(run_dir_path, run_model, *, poll=...)`: read spec; acquire lock;
write `status(running)`; for each model sequentially — append `model_started` to
`live.jsonl`, call `run_model(model, options, emit)`, append `model_finished`;
between models check `read_control` — if `stop`/`abort`, append `stopped`, break;
on finish write `status(complete)`. `register_cleanup(get_processes)` installs
SIGINT/SIGTERM/atexit handlers that `kill_process_tree` each tracked process and
`release_lock`. Tests inject a fake `run_model` (no GPU).

- [ ] **Step 1 — failing tests**

```python
from gguf_limit_bench import engine, run_dir

def test_engine_runs_models_sequentially(tmp_path):
    run_dir.write_spec(tmp_path, {"models": ["a", "b"], "mode": "x", "options": {}})
    seen = []
    def fake_run(model, options, emit): seen.append(model); return tmp_path
    engine.run_engine(tmp_path, fake_run)
    assert seen == ["a", "b"]
    assert run_dir.read_status(tmp_path)["phase"] == "complete"

def test_engine_honors_stop(tmp_path):
    run_dir.write_spec(tmp_path, {"models": ["a", "b"], "mode": "x", "options": {}})
    def fake_run(model, options, emit):
        run_dir.write_control(tmp_path, "stop")  # request stop after first
        return tmp_path
    engine.run_engine(tmp_path, fake_run)
    s = run_dir.read_status(tmp_path)
    assert s["phase"] in ("stopped", "complete")
    # only the first model ran
```

- [ ] **Step 2 — FAIL** (module missing).
- [ ] **Step 3 — implement** `engine.py` per above; `emit` appends to
  `live.jsonl` via `run_dir`; refresh `status.alive_at` before each model.
- [ ] **Step 4 — run tests + suite.** Green.
- [ ] **Step 5 — commit** `feat(engine): sequential runner with control + cleanup`.

---

## Task 4: CLI `engine` command

**Files:** Modify `src/gguf_limit_bench/cli.py`; Test `tests/test_cli_engine.py`
(smoke: `--help` lists `engine`; invoking with a spec dir + stubbed runner
writes status complete — use monkeypatch on the real model runner).

- [ ] Add `@app.command("engine") def engine_cmd(run_dir: Path, ...)` that builds
  the real `run_model` (wrapping `_run_one_autoresearch` with paths from config)
  and calls `engine.run_engine`. Resolve llama paths/runs-root from config/spec.
- [ ] Tests + suite green. Commit `feat(cli): engine command (detached run entry)`.

---

## Task 5: Web server spawns engine detached + control + reattach

**Files:** Modify `src/gguf_limit_bench/webui.py`; Test `tests/test_webui.py`
(extend existing).

- [ ] `start_run`: validate selection; create run dir; `write_spec`; spawn
  `pilotbench engine --run-dir <dir>` via `subprocess.Popen(..., **process_group_kwargs())`
  detached; store `active_run_dir`. Remove `_run_models` daemon-thread evaluation
  (keep a thin `WebRunState` for status mirroring).
- [ ] `request_stop_after_current` → `write_control(active_run_dir, "stop")`; add
  abort → `write_control(..., "abort")`.
- [ ] `state_payload`: if `active_run_dir` set, read `status.json` +
  tail `live.jsonl`; expose to UI. On startup/reattach, detect a live run dir via
  fresh status and resume display.
- [ ] Tests: `start_run` writes spec + calls Popen (monkeypatched) with group
  kwargs; stop writes control.json; reattach reads fresh status. Suite green.
- [ ] Commit `feat(webui): thin client launches detached engine + file control`.

---

## Done-when
- `pytest -q` green; `ruff` clean.
- A real run launched from the web UI runs in a separate process; closing the
  browser does not stop it; `stop` finishes the current item; `abort` kills the
  llama-server tree with no orphan (verify via Task Manager / `nvidia-smi`).
