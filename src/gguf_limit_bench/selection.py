from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.discovery import ModelInfo


class SelectionState:
    def __init__(self, models: list[ModelInfo]) -> None:
        self.models = models
        self._selected: set[Path] = set()

    def toggle(self, index: int) -> None:
        path = self.models[index].path
        if path in self._selected:
            self._selected.remove(path)
        else:
            self._selected.add(path)

    def select_all(self) -> None:
        self._selected = {model.path for model in self.models}

    def clear(self) -> None:
        self._selected.clear()

    def is_selected(self, index: int) -> bool:
        return self.models[index].path in self._selected

    def selected_models(self) -> list[ModelInfo]:
        return [model for model in self.models if model.path in self._selected]

    def selected_paths(self) -> list[Path]:
        return [model.path for model in self.selected_models()]

    def replace_models(self, models: list[ModelInfo]) -> None:
        self.models = models
        valid_paths = {model.path for model in models}
        self._selected = {path for path in self._selected if path in valid_paths}
