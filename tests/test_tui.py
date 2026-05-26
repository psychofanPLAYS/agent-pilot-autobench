import asyncio

from textual.widgets import DataTable, Static

from gguf_limit_bench.tui import BenchTui


def test_tui_loads_models_and_supports_select_all(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "Qwen3-Test-Q4_K_M.gguf").write_bytes(b"fake")

    async def run_tui_check():
        app = BenchTui(root=model_dir)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one("#models", DataTable)
            status = app.query_one("#status", Static)

            assert table.row_count == 1
            assert "1 models found. 0 selected." in str(status.render())

            await pilot.press("a")
            await pilot.pause()

            assert len(app.selection.selected_models()) == 1
            assert "1 selected" in str(status.render())

    asyncio.run(run_tui_check())
