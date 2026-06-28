from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Callable

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, ProgressBar, Static

from gguf_limit_bench.discovery import ModelInfo, discover_models
from gguf_limit_bench.evaluation_mode import EvaluationMode
from gguf_limit_bench.modes import DEFAULT_RUN_MODE, next_mode
from gguf_limit_bench.reports import write_leaderboard
from gguf_limit_bench.run_history import truncated_previous_runs_text
from gguf_limit_bench.run_config import PRESETS, RunConfig
from gguf_limit_bench.selection import SelectionState
from gguf_limit_bench.selection_memory import load_last_selected_paths, save_last_selected_models
import gguf_limit_bench.state_db as state_db
from gguf_limit_bench.telemetry import TelemetrySnapshot, sample_telemetry


RunModelCallback = Callable[[ModelInfo], Path]

DEFAULT_HISTORY_RATIO = 0.18
HISTORY_RATIO_STEP = 0.04
MIN_HISTORY_RATIO = 0.10
MAX_HISTORY_RATIO = 0.36
MIN_HISTORY_HEIGHT = 3
RESERVED_HEIGHT_WITH_MIN_TABLE = 20
NARROW_TABLE_WIDTH = 90
MEDIUM_TABLE_WIDTH = 125


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
    # Layout is sized to fit a standard 80x24 terminal: the model table flexes
    # (1fr) and takes whatever space the fixed panels leave, while recent runs
    # scales with terminal height and can be resized by the user.
    CSS = """
    Screen { background: #0f1117; color: #d6deeb; }
    Header, Footer { background: #151923; color: #d6deeb; }
    Static { color: #d6deeb; }
    DataTable { height: 1fr; min-height: 4; }
    #banner { height: 1; color: #f2c94c; text-style: bold; }
    #telemetry { height: 1; color: #8fa3bf; }
    #status { height: 2; }
    #details { height: 2; color: #9fb4d1; }
    #history_box { min-height: 3; }
    #history { color: #8fa3bf; }
    #dashboard { height: 6; padding: 0 1; border: solid #293241; }
    #progress { height: 1; }
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
        ("m", "cycle_mode", "Mode"),
        ("[", "shrink_history", "Recent runs smaller"),
        ("]", "grow_history", "Recent runs larger"),
        ("0", "reset_history", "Reset panels"),
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
        self.run_mode = DEFAULT_RUN_MODE
        self.telemetry_stats = TelemetryStats()
        self.history_ratio = DEFAULT_HISTORY_RATIO
        self._table_layout: tuple[str, ...] = ()

    @property
    def evaluation_mode(self) -> EvaluationMode:
        return self.run_mode.evaluation

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Agent Pilot — local GGUF + llama.cpp agent benchmark", id="banner")
            yield Static("Sampling hardware telemetry...", id="telemetry")
            yield Static("Loading GGUF models...", id="status")
            yield DataTable(id="models")
            yield Static("Cursor: no model highlighted yet.", id="details")
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
        self._refresh_table()
        self._refresh_telemetry()
        self._apply_history_layout()
        self.set_interval(5, self._refresh_telemetry)
        # Live run progress: while a benchmark is running, poll the active run dir
        # so the user actually sees questions being asked instead of a frozen screen.
        self.set_interval(1.5, self._refresh_run_progress)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_history_layout(event.size.height)

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

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        event.stop()
        self._refresh_details(event.cursor_row)

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

    def action_cycle_mode(self) -> None:
        if self.phase != "selecting":
            return
        self.run_mode = next_mode(self.run_mode)
        if self.is_running:
            self._refresh_table()

    def action_grow_history(self) -> None:
        self._resize_history(HISTORY_RATIO_STEP)

    def action_shrink_history(self) -> None:
        self._resize_history(-HISTORY_RATIO_STEP)

    def action_reset_history(self) -> None:
        self.history_ratio = DEFAULT_HISTORY_RATIO
        self._apply_history_layout()
        if self.phase == "selecting":
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
        message = format_champion_line(name, score)

        # Append per-pack scoreboard from the newest results.json, if present.
        extra_lines: list[str] = []
        try:
            results_json = _newest_results_json(self.runs_root)
            if results_json is not None:
                payload = json.loads(results_json.read_text(encoding="utf-8"))
                packs = payload.get("packs", [])
                extra_lines.append(format_scoreboard(packs))
                model_key = payload.get("model", "")
                db_path = self.runs_root / "state.db"
                if model_key and db_path.exists():
                    with sqlite3.connect(db_path) as conn:
                        for pack in packs:
                            pack_id = pack.get("pack_id", "")
                            if pack_id:
                                stats = state_db.lifetime_pack_stats(conn, model_key, pack_id)
                                if stats["seen"]:
                                    extra_lines.append(format_lifetime_line(pack_id, stats))
        except Exception:  # noqa: BLE001 — don't crash the TUI
            pass

        if extra_lines:
            message = message + "\n" + "\n".join(extra_lines)

        self.call_from_thread(
            self.query_one("#dashboard", Static).update,
            self._dashboard_text(message),
        )

    def _refresh_table(self, keep_row: int = 0) -> None:
        table = self.query_one("#models", DataTable)
        layout = self._ensure_table_columns(table, self.size.width or 80)
        table.clear()
        for index, model in enumerate(self.models):
            table.add_row(*self._row_cells(model, index, layout))
        selected = len(self.selection.selected_models())
        resize_hint = (
            "brackets resize recent runs"
            if (self.size.width or 80) < MEDIUM_TABLE_WIDTH
            else "brackets resize recent runs panel"
        )
        self.query_one("#status", Static).update(
            f"{len(self.models)} models found. {selected} selected. "
            f"Sort: {self.sort_modes[self.sort_mode_index]}. Mode: {self.run_mode.label}. "
            f"Space select, Enter/R run, S sort, M mode, {resize_hint}."
        )
        self._refresh_details(table.cursor_row)
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
            f"Mode: {self.run_mode.label} — {self.run_mode.description}\n"
            f"Budget: {self.run_mode.budget_minutes} min/model (press M to change mode)"
        )

    def _update_run_dashboard(self, message: str, progress: int) -> None:
        self.query_one("#dashboard", Static).update(self._dashboard_text(message))
        self.query_one("#progress", ProgressBar).update(progress=progress)

    def _refresh_run_progress(self) -> None:
        if self.phase != "testing":
            return
        status = active_run_status(self.runs_root)
        if status:
            try:
                self.query_one("#dashboard", Static).update(self._dashboard_text(status))
            except Exception:  # noqa: BLE001 - UI refresh must never crash the run
                pass

    def _history_text(self) -> str:
        return truncated_previous_runs_text(self.runs_root)

    def _refresh_history(self) -> None:
        self.query_one("#history", Static).update(self._history_text())

    def _refresh_telemetry(self) -> None:
        snapshot = sample_telemetry()
        self.telemetry_stats.add(snapshot)
        self.query_one("#telemetry", Static).update(_telemetry_text(snapshot, self.telemetry_stats))

    def _refresh_details(self, row: int | None = None) -> None:
        if row is None or row < 0 or row >= len(self.models):
            text = "Cursor: no model highlighted yet."
        else:
            model = self.models[row]
            selected = "selected" if self.selection.is_selected(row) else "not selected"
            text = (
                f"Cursor: {model.name}\n"
                f"{model.parameters} | {model.quant} | {model.size_gb:.2f} GB | {selected}"
            )
        self.query_one("#details", Static).update(text)

    def _table_layout_for_width(self, width: int) -> tuple[str, ...]:
        if width < NARROW_TABLE_WIDTH:
            return ("selected", "size", "model")
        if width < MEDIUM_TABLE_WIDTH:
            return ("selected", "parameters", "quant", "size", "model")
        return ("selected", "family", "parameters", "quant", "size", "vision", "model")

    def _ensure_table_columns(self, table: DataTable, width: int) -> tuple[str, ...]:
        layout = self._table_layout_for_width(width)
        if layout == self._table_layout:
            return layout
        table.clear(columns=True)
        for key, label, column_width in _column_specs(layout, width):
            table.add_column(label, key=key, width=column_width)
        self._table_layout = layout
        return layout

    def _row_cells(self, model: ModelInfo, index: int, layout: tuple[str, ...]) -> tuple[str, ...]:
        values = {
            "selected": "x" if self.selection.is_selected(index) else "",
            "family": model.family,
            "parameters": model.parameters,
            "quant": model.quant,
            "size": f"{model.size_gb:.2f}",
            "vision": "yes" if model.has_vision else "",
            "model": model.name,
        }
        return tuple(values[key] for key in layout)

    def _resize_history(self, delta: float) -> None:
        self.history_ratio = max(
            MIN_HISTORY_RATIO,
            min(MAX_HISTORY_RATIO, self.history_ratio + delta),
        )
        self._apply_history_layout()
        if self.phase == "selecting":
            self._refresh_table()

    def _apply_history_layout(self, screen_height: int | None = None) -> None:
        history_box = self.query_one("#history_box", VerticalScroll)
        total_height = screen_height or self.size.height or 24
        max_history_height = max(
            MIN_HISTORY_HEIGHT,
            total_height - RESERVED_HEIGHT_WITH_MIN_TABLE,
        )
        target_height = max(
            MIN_HISTORY_HEIGHT,
            min(max_history_height, round(total_height * self.history_ratio)),
        )
        history_box.styles.height = target_height
        history_box.refresh(layout=True)


def active_run_status(runs_root: Path) -> str | None:
    """Build a one-line live-progress string from the most recent run dir.

    Reads the active run's ``events.jsonl`` for the current profile and the
    newest ``transcript.jsonl`` for how many questions have been asked/scored,
    so the cockpit can show real progress instead of a frozen screen. Returns
    None when there is no run dir yet; never raises.
    """
    try:
        dirs = [p for p in runs_root.glob("*") if p.is_dir()]
    except OSError:
        return None
    if not dirs:
        return None
    newest = max(dirs, key=lambda p: p.stat().st_mtime)

    profile: str | None = None
    events = newest / "events.jsonl"
    if events.exists():
        try:
            for line in events.read_text(encoding="utf-8").splitlines():
                data = json.loads(line).get("data", {})
                settings = data.get("settings")
                if isinstance(settings, dict) and settings.get("profile_name"):
                    profile = settings["profile_name"]
                elif data.get("profile_name"):
                    profile = data["profile_name"]
        except (OSError, ValueError):
            pass

    asked = correct = 0
    last_pred: str | None = None
    transcripts = sorted(
        newest.glob("**/transcript.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if transcripts:
        try:
            for line in transcripts[0].read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                asked += 1
                if row.get("correct") or row.get("outcome") == "correct":
                    correct += 1
                last_pred = row.get("predicted_answer", last_pred)
        except (OSError, ValueError):
            pass

    parts = [f"Running: {newest.name[:44]}"]
    if profile:
        parts.append(f"profile {profile}")
    if asked:
        tail = f"asked {asked}Q · {correct} correct"
        if last_pred:
            tail += f" · last={last_pred}"
        parts.append(tail)
    else:
        parts.append("launching server / warming up…")
    return "  ·  ".join(parts)


def _newest_results_json(runs_root: Path) -> Path | None:
    """Return the results.json from the most recently modified run dir, or None."""
    candidates = sorted(
        runs_root.glob("*/results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def format_champion_line(model_name: str | None, score: float | None) -> str:
    if model_name is None or score is None:
        return "Champion: not decided yet"
    return f"Champion: {model_name} ({score:.2f})"


def format_scoreboard(per_pack: list[dict]) -> str:
    """Format a one-line scoreboard from per-pack result dicts.

    Each dict must contain: pack_id, correct, asked, incomplete.
    Returns e.g. ``"simple-bench 2/5 · easy-gotcha 4/5 (1 incomplete)"``.
    Returns ``"no packs scored"`` for an empty list.
    """
    if not per_pack:
        return "no packs scored"
    parts = [f"{p['pack_id']} {p['correct']}/{p['asked']}" for p in per_pack]
    total_incomplete = sum(int(p.get("incomplete", 0)) for p in per_pack)
    line = " · ".join(parts)
    if total_incomplete > 0:
        line += f" ({total_incomplete} incomplete)"
    return line


def format_lifetime_line(pack_id: str, stats: dict) -> str:
    """Format a one-line lifetime stats string for a single pack.

    *stats* must contain: seen, correct, accuracy (float 0–1).
    Returns e.g. ``"lifetime: easy-gotcha 50 seen · 31 correct (62%)"``.
    """
    seen = int(stats.get("seen", 0))
    correct = int(stats.get("correct", 0))
    pct = int(round(float(stats.get("accuracy", 0.0)) * 100))
    return f"lifetime: {pack_id} {seen} seen · {correct} correct ({pct}%)"


def _telemetry_text(snapshot: TelemetrySnapshot, stats: TelemetryStats) -> str:
    gpu_memory = "n/a"
    if snapshot.gpu_used_mb is not None and snapshot.gpu_total_mb is not None:
        gpu_memory = f"{snapshot.gpu_used_mb}/{snapshot.gpu_total_mb}MB"
    gpu_util = snapshot.gpu_util_percent if snapshot.gpu_util_percent is not None else "n/a"
    # Single compact line so the panel fits in one terminal row.
    return (
        f"CPU {snapshot.cpu_used_percent:.0f}%  "
        f"RAM {snapshot.ram_used_percent:.0f}%  "
        f"GPU {gpu_util}%  VRAM {gpu_memory}  "
        f"Swap {snapshot.swap_used_percent:.0f}%"
    )


def _column_specs(layout: tuple[str, ...], table_width: int) -> list[tuple[str, str, int | None]]:
    fixed_widths = {
        "selected": ("Sel", 3),
        "family": ("Family", 9),
        "parameters": ("Params", 9),
        "quant": ("Quant", 10),
        "size": ("GB", 7),
        "vision": ("Vision", 7),
    }
    fixed_total = sum(fixed_widths[key][1] for key in layout if key != "model")
    # DataTable adds padding and separators; leave a little slack so compact
    # layouts do not create a horizontal scroll just from chrome.
    model_width = max(24, table_width - fixed_total - (3 * len(layout)))
    specs: list[tuple[str, str, int | None]] = []
    for key in layout:
        if key == "model":
            specs.append((key, "Model", model_width))
        else:
            label, width = fixed_widths[key]
            specs.append((key, label, width))
    return specs


def _min_value(current: float | None, value: float) -> float:
    return value if current is None else min(current, value)


def _max_value(current: float | None, value: float) -> float:
    return value if current is None else max(current, value)
