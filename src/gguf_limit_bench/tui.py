from __future__ import annotations

from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.run_config import PRESETS, RunConfig
from gguf_limit_bench.selection import SelectionState


RunModelCallback = Callable[[ModelInfo], Path]


class BenchTui(App):
    CSS = """
    Screen { background: #0f1117; color: #d6deeb; }
    Header, Footer { background: #151923; color: #d6deeb; }
    Static { color: #d6deeb; }
    DataTable { height: 1fr; }
    #menu { height: 7; padding: 1; border: solid #293241; }
    #status { height: 3; padding: 1; }
    #dashboard { height: 8; padding: 1; border: solid #293241; }
    """
    BINDINGS = [
        ("space", "toggle_model", "Toggle"),
        ("enter", "run_selected", "Start selected"),
        ("r", "run_selected", "Run selected"),
        ("s", "cycle_sort", "Sort"),
        ("a", "select_all", "Select all"),
        ("c", "clear", "Clear"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, root: Path, run_model: RunModelCallback | None = None) -> None:
        super().__init__()
        self.root = root
        self.run_model = run_model
        self.models: list[ModelInfo] = []
        self.models_to_run: list[ModelInfo] = []
        self.ran_inside_tui = False
        self.selection = SelectionState([])
        self.sort_modes = ("size", "name", "family")
        self.sort_mode_index = 0
        self.run_config = RunConfig.from_preset("normal")

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(self._menu_text(), id="menu")
            yield Static("Loading GGUF models...", id="status")
            yield DataTable(id="models")
            yield Static(self._dashboard_text("Waiting for model selection."), id="dashboard")
            yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Footer()

    def on_mount(self) -> None:
        self.models = discover_models([self.root])
        self.selection = SelectionState(self.models)
        table = self.query_one("#models", DataTable)
        table.cursor_type = "row"
        table.add_columns("Sel", "Family", "Params", "Quant", "GB", "Vision", "Model")
        self._refresh_table()

    def action_toggle_model(self) -> None:
        table = self.query_one("#models", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.models):
            return
        self.selection.toggle(table.cursor_row)
        self._refresh_table(keep_row=table.cursor_row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_run_selected()

    def action_select_all(self) -> None:
        self.selection.select_all()
        self._refresh_table()

    def action_clear(self) -> None:
        self.selection.clear()
        self._refresh_table()

    def action_cycle_sort(self) -> None:
        self.sort_mode_index = (self.sort_mode_index + 1) % len(self.sort_modes)
        self.models = self._sorted_models(self.models)
        self.selection.replace_models(self.models)
        self._refresh_table()

    def action_run_selected(self) -> None:
        selected_models = self.selection.selected_models()
        if not selected_models:
            self.query_one("#status", Static).update(
                f"{len(self.models)} models found. 0 selected. Select at least one model with Space, then press Enter."
            )
            return
        self.models_to_run = selected_models
        if self.run_model is None:
            self.query_one("#dashboard", Static).update(
                self._dashboard_text(
                    f"Starting {len(selected_models)} model(s) with {self.run_config.preset_id} preset."
                )
            )
            self.exit()
            return
        self.ran_inside_tui = True
        self.run_worker(
            lambda: self._run_models_inside_tui(selected_models), thread=True, exclusive=True
        )

    def _run_models_inside_tui(self, selected_models: list[ModelInfo]) -> None:
        for index, model in enumerate(selected_models, start=1):
            self.call_from_thread(
                self._update_run_dashboard,
                f"Running {index}/{len(selected_models)}: {model.name}",
                int((index - 1) / len(selected_models) * 100),
            )
            receipt_path = self.run_model(model) if self.run_model is not None else None
            self.call_from_thread(
                self._update_run_dashboard,
                f"Finished {model.name}. Receipt: {receipt_path}",
                int(index / len(selected_models) * 100),
            )
        self.call_from_thread(self.exit)

    def _refresh_table(self, keep_row: int = 0) -> None:
        table = self.query_one("#models", DataTable)
        table.clear()
        for index, model in enumerate(self.models):
            table.add_row(
                "x" if self.selection.is_selected(index) else "",
                model.family,
                model.parameters,
                model.quant,
                f"{model.size_gb:.2f}",
                "yes" if model.has_vision else "",
                model.name,
            )
        selected = len(self.selection.selected_models())
        self.query_one("#status", Static).update(
            f"{len(self.models)} models found. {selected} selected. "
            f"Sort: {self.sort_modes[self.sort_mode_index]}. "
            "Space selects, Enter/R runs selected, S cycles sort, A selects all, C clears, Q quits."
        )
        if self.models:
            table.move_cursor(row=min(keep_row, len(self.models) - 1))

    def _sorted_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        mode = self.sort_modes[self.sort_mode_index]
        if mode == "name":
            return sorted(models, key=lambda model: model.name.lower())
        if mode == "family":
            return sorted(
                models, key=lambda model: (model.family, -model.size_bytes, model.name.lower())
            )
        return sorted(models, key=lambda model: (-model.size_bytes, model.name.lower()))

    def _menu_text(self) -> str:
        lines = ["Beginner presets:"]
        for preset in PRESETS.values():
            marker = "*" if preset.id == self.run_config.preset_id else " "
            lines.append(f"{marker} {preset.label}: {preset.description}")
        lines.append(
            "Advanced defaults: target TTFT <10s, generation >=20 tok/s, full GPU offload, no swap."
        )
        return "\n".join(lines)

    def _dashboard_text(self, message: str) -> str:
        return (
            "[blue]Run dashboard[/blue]\n"
            f"{message}\n"
            f"Preset: {self.run_config.preset_id} | Budget: {self.run_config.budget_minutes} min/model | "
            f"Packs: {', '.join(self.run_config.packs)}"
        )

    def _update_run_dashboard(self, message: str, progress: int) -> None:
        self.query_one("#dashboard", Static).update(self._dashboard_text(message))
        self.query_one("#progress", ProgressBar).update(progress=progress)
