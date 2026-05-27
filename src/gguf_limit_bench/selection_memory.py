from __future__ import annotations

import json
from pathlib import Path

from gguf_limit_bench.discovery import ModelInfo


def selection_memory_path(db_path: Path = Path("_db/agentpilot.sqlite")) -> Path:
    return db_path.parent / "last-selection.json"


def load_last_selected_paths(path: Path | None = None) -> set[Path]:
    path = path or selection_memory_path()
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = payload.get("selected_model_paths", [])
    if not isinstance(values, list):
        return set()
    return {Path(str(value)) for value in values}


def save_last_selected_models(models: list[ModelInfo], path: Path | None = None) -> None:
    path = path or selection_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_model_paths": [str(model.path) for model in models],
        "selected_model_names": [model.name for model in models],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
