import asyncio

from textual.widgets import DataTable, Static

from gguf_limit_bench.evaluation_mode import EvaluationMode
from gguf_limit_bench.modes import RUN_MODES
from gguf_limit_bench.tui import BenchTui, format_champion_line


def test_tui_loads_models_and_supports_select_all(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Small-Q4_K_M.gguf").write_bytes(b"1" * 10)
    (model_dir / "Qwen3-Large-Q4_K_M.gguf").write_bytes(b"1" * 30)

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#models", DataTable)
            status = app.query_one("#status", Static)

            assert table.row_count == 2
            assert [model.name for model in app.models] == [
                "Qwen3-Large-Q4_K_M.gguf",
                "Qwen3-Small-Q4_K_M.gguf",
            ]
            assert "2 models found. 0 selected." in str(status.render())
            assert "GB" in [str(column.label) for column in table.columns.values()]

            await pilot.press("a")
            await pilot.pause()

            assert len(app.selection.selected_models()) == 2
            assert "2 selected" in str(status.render())

    asyncio.run(run_tui_check())


def test_tui_continue_requires_a_selection_then_exits_with_selected_models(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one("#status", Static)

            await pilot.press("enter")
            await pilot.pause()

            assert app.models_to_run == []
            assert "Select at least one model" in str(status.render())

            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause()

            assert [model.name for model in app.models_to_run] == ["Qwen3-Test-Q4_K_M.gguf"]

    asyncio.run(run_tui_check())


def test_tui_r_key_also_starts_selected_models(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.press("r")
            await pilot.pause()

            assert [model.name for model in app.models_to_run] == ["Qwen3-Test-Q4_K_M.gguf"]

    asyncio.run(run_tui_check())


def test_tui_can_run_selected_models_inside_the_app(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")
    called: list[str] = []

    def fake_run_model(model):
        called.append(model.name)
        return tmp_path / "runs" / model.name

    async def run_tui_check():
        app = BenchTui(root=model_dir, run_model=fake_run_model)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("space")
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert app.ran_inside_tui is True
            assert called == ["Qwen3-Test-Q4_K_M.gguf"]

    asyncio.run(run_tui_check())


def test_tui_s_key_cycles_sort_modes_and_keeps_selection(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "b-small-Q4_K_M.gguf").write_bytes(b"1" * 10)
    (model_dir / "a-large-Q4_K_M.gguf").write_bytes(b"1" * 30)

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("space")
            selected_before = app.selection.selected_paths()

            await pilot.press("s")
            await pilot.pause()

            assert [model.name for model in app.models] == [
                "a-large-Q4_K_M.gguf",
                "b-small-Q4_K_M.gguf",
            ]
            assert app.selection.selected_paths() == selected_before

    asyncio.run(run_tui_check())


def test_tui_defaults_to_find_best_settings_mode(tmp_path):
    app = BenchTui(root=tmp_path, runs_root=tmp_path)
    assert app.run_mode.id == "best_settings"
    assert app.evaluation_mode is EvaluationMode.BENCHMARK


def test_tui_m_key_cycles_run_mode(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.run_mode.id == "best_settings"

            await pilot.press("m")
            await pilot.pause()
            assert app.run_mode.id == "flag_effect"

            # Cycle through to the quick mode; its evaluation is the speed scout.
            for _ in range(len(RUN_MODES)):
                if app.run_mode.id == "quick":
                    break
                await pilot.press("m")
                await pilot.pause()
            assert app.run_mode.id == "quick"
            assert app.evaluation_mode is EvaluationMode.SPEED_SCOUT

    asyncio.run(run_tui_check())


def test_champion_line_formats():
    assert format_champion_line("QwenX", 950.0) == "Champion: QwenX (950.00)"
    assert format_champion_line(None, None) == "Champion: not decided yet"
