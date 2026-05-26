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


def test_selection_state_keeps_selected_paths_when_models_are_reordered():
    models = [
        ModelInfo(path=Path("small.gguf"), name="small", size_bytes=10),
        ModelInfo(path=Path("large.gguf"), name="large", size_bytes=30),
    ]
    state = SelectionState(models)
    state.toggle(0)

    reordered = list(reversed(models))
    state.replace_models(reordered)

    assert state.selected_paths() == [Path("small.gguf")]
    assert state.is_selected(1) is True
    assert state.is_selected(0) is False
