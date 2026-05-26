import json
from pathlib import Path

from typer.testing import CliRunner

from gguf_limit_bench.autoresearch import AttemptResult
from gguf_limit_bench.cli import DEFAULT_MODEL_ROOTS, app
from gguf_limit_bench.discovery import ModelInfo


runner = CliRunner()


class FakeAttemptRunner:
    def __init__(self, llama_bench: Path, model: Path, timeout_seconds: int = 300) -> None:
        self.model = model

    def __call__(self, settings):
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=55.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="unknown",
            stdout="{}",
            stderr="",
            returncode=0,
        )


def test_autoresearch_command_writes_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    model = tmp_path / "Qwen3-Test-Q4_K_M.gguf"
    model.write_bytes(b"fake")

    result = runner.invoke(
        app,
        [
            "autoresearch",
            "--model",
            str(model),
            "--runs-root",
            str(tmp_path / "runs"),
            "--budget-minutes",
            "1",
            "--max-attempts",
            "1",
        ],
    )

    assert result.exit_code == 0
    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.name != "learning"]
    assert len(run_dirs) == 1
    assert json.loads((run_dirs[0] / "best-settings.json").read_text(encoding="utf-8"))
    assert (tmp_path / "runs" / "learning" / "optuna.sqlite3").exists()


def test_autoresearch_all_qwen_only_skips_non_qwen_models(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    qwen = ModelInfo(path=tmp_path / "Qwen3-Test-Q4_K_M.gguf", name="qwen", family="qwen")
    llama = ModelInfo(path=tmp_path / "Llama-Test-Q4_K_M.gguf", name="llama", family="llama")
    monkeypatch.setattr("gguf_limit_bench.cli.discover_models", lambda roots: [qwen, llama])

    result = runner.invoke(
        app,
        [
            "autoresearch-all",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(tmp_path / "runs"),
            "--qwen-only",
            "--budget-minutes",
            "1",
            "--max-attempts",
            "1",
        ],
    )

    assert result.exit_code == 0
    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.name != "learning"]
    assert len(run_dirs) == 1
    assert (tmp_path / "runs" / "learning" / "optuna.sqlite3").exists()


def test_autoresearch_all_qwen_35b_only_skips_27b_models(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    qwen_35b = ModelInfo(
        path=tmp_path / "Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
        name="qwen-35b",
        family="qwen",
        parameters="35B-A3B",
        has_mtp=True,
    )
    qwen_27b = ModelInfo(
        path=tmp_path / "Qwen3.6-27B-Q4_K_M.gguf",
        name="qwen-27b",
        family="qwen",
        parameters="27B",
    )
    monkeypatch.setattr("gguf_limit_bench.cli.discover_models", lambda roots: [qwen_35b, qwen_27b])

    result = runner.invoke(
        app,
        [
            "autoresearch-all",
            "--runs-root",
            str(tmp_path / "runs"),
            "--qwen-35b-only",
            "--budget-minutes",
            "1",
            "--max-attempts",
            "1",
            "--no-learning",
            "--no-workflow-eval",
        ],
    )

    assert result.exit_code == 0
    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.name != "learning"]
    assert len(run_dirs) == 1
    assert "35B" in json.loads((run_dirs[0] / "best-settings.json").read_text(encoding="utf-8"))["model"]


def test_default_model_roots_include_lm_studio_folder():
    assert Path("G:/AI/models") in DEFAULT_MODEL_ROOTS
    assert Path("G:/AI/models/LM_Studio-gguf") in DEFAULT_MODEL_ROOTS


def test_autoresearch_all_honors_total_budget_and_finish_early(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    models = [
        ModelInfo(path=tmp_path / "Qwen3-A-Q4_K_M.gguf", name="qwen-a", family="qwen"),
        ModelInfo(path=tmp_path / "Qwen3-B-Q4_K_M.gguf", name="qwen-b", family="qwen"),
    ]
    monkeypatch.setattr("gguf_limit_bench.cli.discover_models", lambda roots: models)

    result = runner.invoke(
        app,
        [
            "autoresearch-all",
            "--runs-root",
            str(tmp_path / "runs"),
            "--total-budget-minutes",
            "1",
            "--budget-minutes",
            "1",
            "--finish-early-on",
            "--target-score",
            "1",
            "--max-attempts",
            "1",
            "--no-learning",
            "--no-workflow-eval",
        ],
    )

    assert result.exit_code == 0
    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.name != "learning"]
    assert len(run_dirs) == 1
