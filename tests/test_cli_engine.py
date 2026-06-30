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


def _run_engine_capturing(tmp_path, monkeypatch, spec):
    """Invoke `engine` against a spec, capturing the kwargs run_model receives."""
    rd = tmp_path / "run"
    rd.mkdir()
    run_dir.write_spec(rd, spec)

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
    return captured


def test_engine_prefers_spec_llama_server_over_config(tmp_path, monkeypatch):
    spec_server = tmp_path / "real" / "llama-server.exe"
    captured = _run_engine_capturing(
        tmp_path,
        monkeypatch,
        {
            "models": ["m1.gguf"],
            "mode": "librarian_bench",
            "options": {"budget_minutes": 1},
            "paths": {"llama_server": str(spec_server)},
        },
    )
    assert str(captured.get("llama_server")) == str(spec_server)


def test_engine_falls_back_to_config_when_spec_path_null(tmp_path, monkeypatch):
    from gguf_limit_bench.config import load_config

    default_server = load_config().paths.llama_server
    captured = _run_engine_capturing(
        tmp_path,
        monkeypatch,
        {
            "models": ["m1.gguf"],
            "mode": "librarian_bench",
            "options": {"budget_minutes": 1},
            "paths": {"llama_server": None},
        },
    )
    assert str(captured.get("llama_server")) == str(default_server)


def test_engine_runs_unchanged_without_paths_block(tmp_path, monkeypatch):
    from gguf_limit_bench.config import load_config

    default_server = load_config().paths.llama_server
    captured = _run_engine_capturing(
        tmp_path,
        monkeypatch,
        {"models": ["m1.gguf"], "mode": "librarian_bench", "options": {"budget_minutes": 1}},
    )
    # A spec with no "paths" key behaves exactly as before: config defaults win.
    assert str(captured.get("llama_server")) == str(default_server)
