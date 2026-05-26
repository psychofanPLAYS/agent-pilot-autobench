from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.discovery import ModelInfo


class SelectionState:
    def __init__(self, models: list[ModelInfo]) -> None:
        self.models = models
        self._selected: set[int] = set()

    def toggle(self, index: int) -> None:
        if index in self._selected:
            self._selected.remove(index)
        else:
            self._selected.add(index)

    def select_all(self) -> None:
        self._selected = set(range(len(self.models)))

    def clear(self) -> None:
        self._selected.clear()

    def is_selected(self, index: int) -> bool:
        return index in self._selected

    def selected_models(self) -> list[ModelInfo]:
        return [model for index, model in enumerate(self.models) if index in self._selected]

    def selected_paths(self) -> list[Path]:
        return [model.path for model in self.selected_models()]

