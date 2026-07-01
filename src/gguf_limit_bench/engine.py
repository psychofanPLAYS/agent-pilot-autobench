"""Sequential benchmark engine — the detached process that does all the work.

It reads ``run-spec.json`` from a run directory, runs each model in order via an
injected ``run_model`` callback, keeps ``status.json`` fresh, mirrors lifecycle
and per-question events into ``live.jsonl``, and obeys ``control.json`` (a
``stop``/``abort`` request halts the queue after the current model). The web UI
never imports this; it launches it as a separate process and reads the run
directory.

Testing keeps ``run_model`` injectable so the engine logic is exercised without a
GPU. The real callback (wired in the CLI) spawns llama-server and runs the
autoresearch + champion eval.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Callable, Protocol

from gguf_limit_bench import events, run_dir


class EventSink(Protocol):
    def __call__(self, event_type: str, data: dict) -> None: ...


# run_model(model, options, emit) -> receipt path
RunModel = Callable[[str, dict, EventSink], object]

_STOP_ACTIONS = ("stop", "abort")

# How often the heartbeat thread re-stamps status.json so a long single-model run
# never looks stale to the web UI's liveness check (a browser refresh / reattach
# must never mark a live run crashed). Kept ≤ run_dir.engine_is_alive stale window.
HEARTBEAT_SECONDS = 2.0


def _model_label(model: object) -> str:
    """Human-friendly name for status/events, for str or {path,...} model items."""
    if isinstance(model, dict):
        return Path(str(model.get("path", ""))).name or str(model)
    return str(model)


def run_engine(
    run_dir_path: Path,
    run_model: RunModel,
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
    """Run every model in the spec sequentially, honouring stop/abort control.

    A background heartbeat thread re-stamps ``status.json`` every
    ``heartbeat_seconds`` from the latest phase/model fields, so a long single
    model run keeps a fresh liveness signal even when no events fire for a while
    (e.g. during a slow autoresearch attempt or model load)."""
    run_dir_path = Path(run_dir_path)
    spec = run_dir.read_spec(run_dir_path)
    models = list(spec.get("models", []))
    options = dict(spec.get("options", {}))
    pid = os.getpid()

    tally = {"answered": 0, "correct": 0}
    # Latest status fields; the heartbeat thread re-stamps alive_at from these.
    hb_state = {"phase": "running", "model": None, "model_index": None,
                "model_total": len(models)}
    hb_lock = threading.Lock()
    hb_stop = threading.Event()

    def write_status(**changes: object) -> None:
        with hb_lock:
            hb_state.update(changes)
            snapshot = dict(hb_state)
        run_dir.write_status(run_dir_path, pid=pid, **snapshot)

    def _heartbeat() -> None:
        while not hb_stop.wait(heartbeat_seconds):
            with hb_lock:
                snapshot = dict(hb_state)
            run_dir.write_status(run_dir_path, pid=pid, **snapshot)

    def emit(event_type: str, data: dict) -> None:
        run_dir.append_event(run_dir_path, event_type, data)
        if event_type == "question_scored":
            tally["answered"] += 1
            if data.get("correct"):
                tally["correct"] += 1
            answered = tally["answered"]
            quality = round(100.0 * tally["correct"] / answered, 1) if answered else 0.0
            run_dir.append_event(
                run_dir_path,
                "running_score",
                {"answered": answered, "correct": tally["correct"], "quality_0_100": quality},
            )

    run_dir.acquire_lock(run_dir_path, pid)
    write_status(phase="running")
    heartbeat = threading.Thread(target=_heartbeat, name="engine-heartbeat", daemon=True)
    heartbeat.start()
    stopped = False
    try:
      with events.set_event_sink(emit):
        for index, model in enumerate(models, start=1):
            label = _model_label(model)
            if run_dir.read_control(run_dir_path)["action"] in _STOP_ACTIONS:
                emit("run_stopped", {"reason": "control", "before_model": label})
                stopped = True
                break
            write_status(phase="running", model=label, model_index=index)
            emit("model_started", {"model": label, "index": index, "total": len(models)})
            try:
                run_model(model, options, emit)
            except BaseException as exc:  # noqa: BLE001 - record then re-raise
                emit("model_failed", {"model": label, "error": str(exc)})
                write_status(phase="failed", model=label, model_index=index)
                raise
            emit("model_finished", {"model": label, "index": index})

        final_phase = "stopped" if stopped else "complete"
        write_status(phase=final_phase)
        emit("run_finished", {"phase": final_phase})
    finally:
        hb_stop.set()
        heartbeat.join(timeout=2.0)
        run_dir.release_lock(run_dir_path)
