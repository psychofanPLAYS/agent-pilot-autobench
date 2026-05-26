from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.selection import SelectionState


class BenchTui(App):
    CSS = """
    DataTable { height: 1fr; }
    #status { height: 3; padding: 1; }
    """
    BINDINGS = [
        ("space", "toggle_model", "Toggle"),
        ("a", "select_all", "Select all"),
        ("c", "clear", "Clear"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.models: list[ModelInfo] = []
        self.selection = SelectionState([])

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Loading GGUF models...", id="status")
            yield DataTable(id="models")
        yield Footer()

    def on_mount(self) -> None:
        self.models = discover_models([self.root])
        self.selection = SelectionState(self.models)
        table = self.query_one("#models", DataTable)
        table.cursor_type = "row"
        table.add_columns("Sel", "Family", "Params", "Quant", "Vision", "Model")
        self._refresh_table()

    def action_toggle_model(self) -> None:
        table = self.query_one("#models", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.models):
            return
        self.selection.toggle(table.cursor_row)
        self._refresh_table(keep_row=table.cursor_row)

    def action_select_all(self) -> None:
        self.selection.select_all()
        self._refresh_table()

    def action_clear(self) -> None:
        self.selection.clear()
        self._refresh_table()

    def _refresh_table(self, keep_row: int = 0) -> None:
        table = self.query_one("#models", DataTable)
        table.clear()
        for index, model in enumerate(self.models):
            table.add_row(
                "x" if self.selection.is_selected(index) else "",
                model.family,
                model.parameters,
                model.quant,
                "yes" if model.has_vision else "",
                model.name,
            )
        selected = len(self.selection.selected_models())
        self.query_one("#status", Static).update(
            f"{len(self.models)} models found. {selected} selected. "
            "Space toggles, A selects all, C clears. Use CLI autoresearch for unattended runs."
        )
        if self.models:
            table.move_cursor(row=min(keep_row, len(self.models) - 1))
