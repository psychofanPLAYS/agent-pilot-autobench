"""Replay a recorded ``live.jsonl`` into a run directory at a realistic cadence.

This drives the cockpit with no GPU: it re-appends each recorded event and keeps
``status.json`` fresh, deriving model/question progress from the events. Used for
visual verification of the in-flight cockpit and as a demo.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Callable

from gguf_limit_bench import run_dir


def replay(
    run_dir_path: Path,
    source_path: Path,
    *,
    delay: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    on_step: Callable[[], None] | None = None,
) -> None:
    """Re-stream the events in *source_path* into *run_dir_path*/live.jsonl."""
    run_dir_path = Path(run_dir_path)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    model = None
    model_index = model_total = None
    question_index = question_total = None
    run_dir.write_status(run_dir_path, phase="running", pid=pid)

    lines = Path(source_path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_dir.append_event(run_dir_path, record.get("type", "event"), record.get("data", {}))
        data = record.get("data", {}) if isinstance(record.get("data"), dict) else {}
        if record.get("type") == "model_started":
            model = data.get("model", model)
            model_index = data.get("index", model_index)
            model_total = data.get("total", model_total)
        elif record.get("type") == "question_started":
            question_index = data.get("index", question_index)
            question_total = data.get("total", question_total)
        run_dir.write_status(
            run_dir_path,
            phase="running",
            model=model,
            model_index=model_index,
            model_total=model_total,
            question_index=question_index,
            question_total=question_total,
            pid=pid,
        )
        if on_step is not None:
            on_step()
        if delay:
            sleep(delay)

    run_dir.write_status(
        run_dir_path,
        phase="complete",
        model=model,
        model_index=model_index,
        model_total=model_total,
        pid=pid,
    )
