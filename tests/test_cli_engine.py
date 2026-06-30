"""Smoke tests for the `engine` CLI command (the detached run entry point)."""

from __future__ import annotations

from typer.testing import CliRunner

import gguf_limit_bench.cli as cli_mod
from gguf_limit_bench import run_dir
from gguf_limit_bench.cli import app

runner = CliRunner()


def test_engine_command_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "engine" in result.output


def test_engine_command_runs_spec_models(tmp_path, monkeypatch):
    rd = tmp_path / "run1"
    rd.mkdir()
    run_dir.write_spec(
        rd,
        {"models": ["m1.gguf"], "mode": "librarian_bench", "options": {"budget_minutes": 1}},
    )

    captured: dict = {}

    class FakeReceipt:
        def __init__(self, path):
            self.path = path

    def fake_run(**kwargs):
        captured.update(kwargs)
        return FakeReceipt(rd)

    monkeypatch.setattr(cli_mod, "_run_one_autoresearch", fake_run)

    result = runner.invoke(app, ["engine", "--run-dir", str(rd)])

    assert result.exit_code == 0, result.output
    assert run_dir.read_status(rd)["phase"] == "complete"
    # librarian_bench mode must wire the librarian champion packs
    assert captured.get("champion_pack_ids")
    assert str(captured.get("model")).endswith("m1.gguf")
