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
from typing import Callable, Protocol

from gguf_limit_bench import run_dir


class EventSink(Protocol):
    def __call__(self, event_type: str, data: dict) -> None: ...


# run_model(model, options, emit) -> receipt path
RunModel = Callable[[str, dict, EventSink], object]

_STOP_ACTIONS = ("stop", "abort")


def run_engine(run_dir_path: Path, run_model: RunModel) -> None:
    """Run every model in the spec sequentially, honouring stop/abort control."""
    run_dir_path = Path(run_dir_path)
    spec = run_dir.read_spec(run_dir_path)
    models = list(spec.get("models", []))
    options = dict(spec.get("options", {}))
    pid = os.getpid()

    def emit(event_type: str, data: dict) -> None:
        run_dir.append_event(run_dir_path, event_type, data)

    run_dir.acquire_lock(run_dir_path, pid)
    run_dir.write_status(
        run_dir_path, phase="running", model_total=len(models), pid=pid
    )
    stopped = False
    try:
        for index, model in enumerate(models, start=1):
            if run_dir.read_control(run_dir_path)["action"] in _STOP_ACTIONS:
                emit("run_stopped", {"reason": "control", "before_model": model})
                stopped = True
                break
            run_dir.write_status(
                run_dir_path,
                phase="running",
                model=model,
                model_index=index,
                model_total=len(models),
                pid=pid,
            )
            emit("model_started", {"model": model, "index": index, "total": len(models)})
            try:
                run_model(model, options, emit)
            except BaseException as exc:  # noqa: BLE001 - record then re-raise
                emit("model_failed", {"model": model, "error": str(exc)})
                run_dir.write_status(
                    run_dir_path,
                    phase="failed",
                    model=model,
                    model_index=index,
                    model_total=len(models),
                    pid=pid,
                )
                raise
            emit("model_finished", {"model": model, "index": index})

        final_phase = "stopped" if stopped else "complete"
        run_dir.write_status(
            run_dir_path, phase=final_phase, model_total=len(models), pid=pid
        )
        emit("run_finished", {"phase": final_phase})
    finally:
        run_dir.release_lock(run_dir_path)
