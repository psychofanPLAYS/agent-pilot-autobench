from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.evaluation_mode import EvaluationMode
from gguf_limit_bench.reports import write_leaderboard
from gguf_limit_bench.run_history import truncated_previous_runs_text
from gguf_limit_bench.run_config import PRESETS, RunConfig
from gguf_limit_bench.selection import SelectionState
from gguf_limit_bench.selection_memory import load_last_selected_paths, save_last_selected_models
from gguf_limit_bench.telemetry import TelemetrySnapshot, sample_telemetry


RunModelCallback = Callable[[ModelInfo], Path]


BANNER = r"""
                              pilotBENCHY
       _ __      __  ____  ______   ____  ______  _   __  ______  __  ____  __
      (_) /___  / /_/ __ )/ ____/  / __ )/ ____/ / | / / / ____/ / / / / / / /
     / / / __ \/ __/ __  / __/    / __  / __/   /  |/ / / /     / /_/ / /_/ /
    / / / /_/ / /_/ /_/ / /___   / /_/ / /___  / /|  / / /___  / __  / __  /
   /_/_/ .___/\__/_____/_____/  /_____/_____/ /_/ |_/  \____/ /_/ /_/_/ /_/
      /_/
"""


@dataclass
class TelemetryStats:
    count: int = 0
    cpu_sum: float = 0.0
    ram_sum: float = 0.0
    gpu_sum: float = 0.0
    cpu_min: float | None = None
    cpu_max: float | None = None
    ram_min: float | None = None
    ram_max: float | None = None
    gpu_min: float | None = None
    gpu_max: float | None = None

    def add(self, snapshot: TelemetrySnapshot) -> None:
        self.count += 1
        self.cpu_sum += snapshot.cpu_used_percent
        self.ram_sum += snapshot.ram_used_percent
        gpu = float(snapshot.gpu_util_percent or 0)
        self.gpu_sum += gpu
        self.cpu_min = _min_value(self.cpu_min, snapshot.cpu_used_percent)
        self.cpu_max = _max_value(self.cpu_max, snapshot.cpu_used_percent)
        self.ram_min = _min_value(self.ram_min, snapshot.ram_used_percent)
        self.ram_max = _max_value(self.ram_max, snapshot.ram_used_percent)
        self.gpu_min = _min_value(self.gpu_min, gpu)
        self.gpu_max = _max_value(self.gpu_max, gpu)

    def avg_cpu(self) -> float:
        return self.cpu_sum / self.count if self.count else 0.0

    def avg_ram(self) -> float:
        return self.ram_sum / self.count if self.count else 0.0

    def avg_gpu(self) -> float:
        return self.gpu_sum / self.count if self.count else 0.0


class BenchTui(App):
    CSS = """
    Screen { background: #0f1117; color: #d6deeb; }
    Header, Footer { background: #151923; color: #d6deeb; }
    Static { color: #d6deeb; }
    DataTable { height: 2fr; }
    #banner { height: 8; padding: 0 1; color: #f2c94c; }
    #telemetry { height: 5; padding: 1; border: solid #3a4a5c; }
    #menu { height: 7; padding: 1; border: solid #293241; }
    #status { height: 3; padding: 1; }
    #history_box { height: 10; border: solid #293241; }
    #history { padding: 1; }
    #dashboard { height: 7; padding: 1; border: solid #293241; }
    """
    BINDINGS = [
        ("space", "toggle_model", "Toggle"),
        ("enter", "run_selected", "Start selected"),
        ("r", "run_selected", "Run selected"),
        ("b", "abort_after_current", "Abort after current"),
        ("escape", "cancel", "Cancel"),
        ("s", "cycle_sort", "Sort"),
        ("a", "select_all", "Select all"),
        ("c", "clear", "Clear"),
        ("m", "toggle_evaluation", "Mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        root: Path,
        run_model: RunModelCallback | None = None,
        runs_root: Path = Path("_runs"),
    ) -> None:
        super().__init__()
        self.root = root
        self.runs_root = runs_root
        self.run_model = run_model
        self.models: list[ModelInfo] = []
        self.models_to_run: list[ModelInfo] = []
        self.ran_inside_tui = False
        self.phase = "selecting"
        self.abort_after_current = False
        self.selection = SelectionState([])
        self.sort_modes = ("size", "name", "family")
        self.sort_mode_index = 0
        self.run_config = RunConfig.from_preset("normal")
        self.evaluation_mode = EvaluationMode.BENCHMARK
        self.telemetry_stats = TelemetryStats()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(BANNER, id="banner")
            yield Static("Sampling hardware telemetry...", id="telemetry")
            yield Static(self._menu_text(), id="menu")
            yield Static("Loading GGUF models...", id="status")
            yield DataTable(id="models")
            with VerticalScroll(id="history_box"):
                yield Static(self._history_text(), id="history")
            yield Static(self._dashboard_text("Waiting for model selection."), id="dashboard")
            yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Footer()

    def on_mount(self) -> None:
        self.models = discover_models([self.root])
        self.selection = SelectionState(self.models)
        self._restore_last_selection()
        table = self.query_one("#models", DataTable)
        table.cursor_type = "row"
        table.add_columns("Sel", "Family", "Params", "Quant", "GB", "Vision", "Model")
        self._refresh_table()
        self._refresh_telemetry()
        self.set_interval(5, self._refresh_telemetry)

    def action_toggle_model(self) -> None:
        if self.phase != "selecting":
            return
        table = self.query_one("#models", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self.models):
            return
        self.selection.toggle(table.cursor_row)
        self._refresh_table(keep_row=table.cursor_row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.action_run_selected()

    def action_select_all(self) -> None:
        if self.phase != "selecting":
            return
        self.selection.select_all()
        self._refresh_table()

    def action_clear(self) -> None:
        if self.phase != "selecting":
            return
        self.selection.clear()
        self._refresh_table()

    def action_cycle_sort(self) -> None:
        if self.phase != "selecting":
            return
        self.sort_mode_index = (self.sort_mode_index + 1) % len(self.sort_modes)
        self.models = self._sorted_models(self.models)
        self.selection.replace_models(self.models)
        self._refresh_table()

    def action_toggle_evaluation(self) -> None:
        if self.phase != "selecting":
            return
        self.evaluation_mode = (
            EvaluationMode.SPEED_SCOUT
            if self.evaluation_mode is EvaluationMode.BENCHMARK
            else EvaluationMode.BENCHMARK
        )
        if self.is_running:
            self._refresh_table()

    def action_cancel(self) -> None:
        if self.phase == "selecting":
            self.models_to_run = []
            self.exit()
            return
        self.abort_after_current = True
        self.query_one("#status", Static).update(
            "Abort requested. The current test will finish, then the queue will stop."
        )

    def action_abort_after_current(self) -> None:
        self.action_cancel()

    def action_run_selected(self) -> None:
        if self.phase != "selecting":
            return
        selected_models = self.selection.selected_models()
        if not selected_models:
            self.query_one("#status", Static).update(
                f"{len(self.models)} models found. 0 selected. Select at least one model with Space, then press Enter."
            )
            return
        self.models_to_run = selected_models
        save_last_selected_models(selected_models)
        if self.run_model is None:
            self.query_one("#dashboard", Static).update(
                self._dashboard_text(
                    f"Starting {len(selected_models)} model(s) with {self.run_config.preset_id} preset."
                )
            )
            self.exit()
            return
        self.ran_inside_tui = True
        self.phase = "testing"
        self.query_one("#status", Static).update(
            f"Testing {len(selected_models)} selected model(s). B or Esc stops after current test."
        )
        self.query_one("#history", Static).update(self._history_text())
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
            self.call_from_thread(self._refresh_history)
            if self.abort_after_current:
                self.call_from_thread(
                    self._update_run_dashboard,
                    "Stopped after the current test by user request.",
                    int(index / len(selected_models) * 100),
                )
                break
        self._show_champion()
        self.phase = "finished"
        self.call_from_thread(self.exit)

    def _show_champion(self) -> None:
        board = write_leaderboard(self.runs_root)
        name = board.champion.model_name if board.entries else None
        score = board.champion.score if board.entries else None
        self.call_from_thread(
            self.query_one("#dashboard", Static).update,
            self._dashboard_text(format_champion_line(name, score)),
        )

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
        mode_label = (
            "benchmark (asks questions)"
            if self.evaluation_mode is EvaluationMode.BENCHMARK
            else "speed scout (no questions)"
        )
        self.query_one("#status", Static).update(
            f"{len(self.models)} models found. {selected} selected. "
            f"Sort: {self.sort_modes[self.sort_mode_index]}. Mode: {mode_label}. "
            "Space selects, Enter/R runs, S sort, A all, C clear, M mode, Esc cancel."
        )
        if self.models:
            table.move_cursor(row=min(keep_row, len(self.models) - 1))

    def _restore_last_selection(self) -> None:
        remembered = load_last_selected_paths()
        if not remembered:
            return
        for index, model in enumerate(self.models):
            if model.path in remembered:
                self.selection.toggle(index)

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

    def _history_text(self) -> str:
        return truncated_previous_runs_text(self.runs_root)

    def _refresh_history(self) -> None:
        self.query_one("#history", Static).update(self._history_text())

    def _refresh_telemetry(self) -> None:
        snapshot = sample_telemetry()
        self.telemetry_stats.add(snapshot)
        self.query_one("#telemetry", Static).update(_telemetry_text(snapshot, self.telemetry_stats))


def format_champion_line(model_name: str | None, score: float | None) -> str:
    if model_name is None or score is None:
        return "Champion: not decided yet"
    return f"Champion: {model_name} ({score:.2f})"


def _telemetry_text(snapshot: TelemetrySnapshot, stats: TelemetryStats) -> str:
    gpu_memory = "n/a"
    if snapshot.gpu_used_mb is not None and snapshot.gpu_total_mb is not None:
        gpu_memory = f"{snapshot.gpu_used_mb}/{snapshot.gpu_total_mb} MB"
    return (
        "Hardware monitor\n"
        f"CPU now {snapshot.cpu_used_percent:.0f}% | min/avg/max "
        f"{stats.cpu_min:.0f}/{stats.avg_cpu():.0f}/{stats.cpu_max:.0f}%    "
        f"RAM now {snapshot.ram_used_percent:.0f}% | min/avg/max "
        f"{stats.ram_min:.0f}/{stats.avg_ram():.0f}/{stats.ram_max:.0f}%\n"
        f"GPU now {snapshot.gpu_util_percent if snapshot.gpu_util_percent is not None else 'n/a'}% "
        f"| min/avg/max {stats.gpu_min:.0f}/{stats.avg_gpu():.0f}/{stats.gpu_max:.0f}%    "
        f"VRAM {gpu_memory}    Swap {snapshot.swap_used_percent:.0f}%    "
        f"Disk R/W {snapshot.disk_read_mb:.0f}/{snapshot.disk_write_mb:.0f} MB"
    )


def _min_value(current: float | None, value: float) -> float:
    return value if current is None else min(current, value)


def _max_value(current: float | None, value: float) -> float:
    return value if current is None else max(current, value)
