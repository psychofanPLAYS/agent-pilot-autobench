from pathlib import Path

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.selection_memory import load_last_selected_paths, save_last_selected_models


def test_selection_memory_round_trips_selected_model_paths(tmp_path):
    path = tmp_path / "_db" / "last-selection.json"
    models = [
        ModelInfo(path=Path("models/a.gguf"), name="a.gguf", family="qwen"),
        ModelInfo(path=Path("models/b.gguf"), name="b.gguf", family="qwen"),
    ]

    save_last_selected_models(models, path)

    assert load_last_selected_paths(path) == {Path("models/a.gguf"), Path("models/b.gguf")}
