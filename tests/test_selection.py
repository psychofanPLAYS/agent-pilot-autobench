from pathlib import Path

from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.selection import SelectionState


def test_selection_state_toggles_one_model_and_selects_all():
    models = [
        ModelInfo(path=Path("a.gguf"), name="a", family="qwen"),
        ModelInfo(path=Path("b.gguf"), name="b", family="llama"),
    ]
    state = SelectionState(models)

    state.toggle(0)
    assert state.selected_paths() == [Path("a.gguf")]

    state.select_all()
    assert state.selected_paths() == [Path("a.gguf"), Path("b.gguf")]

    state.clear()
    assert state.selected_paths() == []

