"""Run-directory contract — the only seam between the thin web UI and the engine.

The web UI writes ``run-spec.json`` (the launch instruction) and ``control.json``
(stop/abort), and reads ``status.json`` (heartbeat) plus ``live.jsonl`` (event
stream). The detached engine does the reverse. All writes are atomic so a reader
never observes a half-written file.

No business logic lives here: it is pure read/write of well-known filenames.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

SPEC_FILE = "run-spec.json"
STATUS_FILE = "status.json"
CONTROL_FILE = "control.json"
LIVE_FILE = "live.jsonl"
LOCK_FILE = "engine.lock"

VALID_ACTIONS = ("run", "stop", "abort")


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# --- run spec ---------------------------------------------------------------


def write_spec(run_dir: Path, spec: dict) -> None:
    _write_atomic(Path(run_dir) / SPEC_FILE, json.dumps(spec, ensure_ascii=True, indent=2))


def read_spec(run_dir: Path) -> dict:
    return _read_json(Path(run_dir) / SPEC_FILE)


# --- status / heartbeat -----------------------------------------------------


def write_status(
    run_dir: Path,
    *,
    phase: str,
    model: str | None = None,
    model_index: int | None = None,
    model_total: int | None = None,
    question_index: int | None = None,
    question_total: int | None = None,
    pid: int | None = None,
    now: datetime | None = None,
) -> None:
    status = {
        "phase": phase,
        "model": model,
        "model_index": model_index,
        "model_total": model_total,
        "question_index": question_index,
        "question_total": question_total,
        "pid": pid,
        "alive_at": _now(now).isoformat(timespec="seconds"),
    }
    _write_atomic(Path(run_dir) / STATUS_FILE, json.dumps(status, ensure_ascii=True, indent=2))


def read_status(run_dir: Path) -> dict:
    return _read_json(Path(run_dir) / STATUS_FILE)


def engine_is_alive(status: dict, *, now: datetime, stale_seconds: float = 10.0) -> bool:
    """True if the engine heartbeat in *status* is newer than *stale_seconds*."""
    alive_at = status.get("alive_at")
    if not alive_at:
        return False
    try:
        stamp = datetime.fromisoformat(str(alive_at))
    except ValueError:
        return False
    return (now - stamp).total_seconds() <= stale_seconds


# --- control (stop / abort) -------------------------------------------------


def write_control(run_dir: Path, action: str, *, now: datetime | None = None) -> None:
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown control action: {action!r}")
    payload = {"action": action, "requested_at": _now(now).isoformat(timespec="seconds")}
    _write_atomic(Path(run_dir) / CONTROL_FILE, json.dumps(payload, ensure_ascii=True))


def read_control(run_dir: Path) -> dict:
    control = _read_json(Path(run_dir) / CONTROL_FILE)
    if "action" not in control:
        return {"action": "run"}
    return control


# --- single-writer lock -----------------------------------------------------


def acquire_lock(run_dir: Path, pid: int) -> bool:
    """Create an exclusive engine lock. Returns False if one already exists."""
    lock = Path(run_dir) / LOCK_FILE
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": pid}))
    return True


def release_lock(run_dir: Path) -> None:
    lock = Path(run_dir) / LOCK_FILE
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


# --- live event stream ------------------------------------------------------


def append_event(
    run_dir: Path, event_type: str, data: dict, *, now: datetime | None = None
) -> None:
    """Append one ``{time,type,data}`` record to ``live.jsonl`` (engine side)."""
    record = {
        "time": _now(now).isoformat(timespec="seconds"),
        "type": event_type,
        "data": data,
    }
    path = Path(run_dir) / LIVE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
