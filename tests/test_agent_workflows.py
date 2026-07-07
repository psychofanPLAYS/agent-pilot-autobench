import json
from pathlib import Path

from typer.testing import CliRunner

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchLoop
from gguf_limit_bench.cli import app
from gguf_limit_bench.discovery import ModelInfo


runner = CliRunner()


class FastAttemptRunner:
    def __init__(self, llama_bench: Path, model: Path, timeout_seconds: int = 300) -> None:
        self.model = model

    def __call__(self, settings):
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=51.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="unknown",
            stdout="{}",
            stderr="",
            returncode=0,
        )


def test_workflow_eval_receipts_are_small_and_deterministic(tmp_path):
    def fake_runner(settings):
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=64.0,
            prompt_tokens_per_second=800.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="unknown",
            stdout='{"ok": true}',
            stderr="",
            returncode=0,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Agent-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (receipt.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    receipt_files = set(path.name for path in receipt.path.iterdir())
    assert {
        "best-settings.json",
        "events.jsonl",
        "recovery.json",
        "summary.md",
    }.issubset(receipt_files)
    assert {"itemized-report.md", "report.html", "report.json"}.issubset(receipt_files)
    assert Path(best["model"]) == Path("G:/AI/models/Qwen3-Agent-Q4_K_M.gguf")
    assert best["settings"] == {
        "context_size": 4096,
        "parallel": 1,
        "gpu_layers": 99,
        "batch_size": 2048,
        "ubatch_size": 512,
        "flash_attention": True,
        "kv_unified": True,
    }
    assert best["result"]["stdout"] == '{"ok": true}'
    assert best["result"]["stderr"] == ""
    assert [event["type"] for event in events] == [
        "autoresearch_started",
        "autoresearch_attempt_started",
        "autoresearch_attempt_finished",
        "autoresearch_finished",
    ]
    assert (receipt.path / "events.jsonl").stat().st_size < 20_000
    assert (receipt.path / "best-settings.json").stat().st_size < 30_000


def test_default_model_root_is_repo_relative(monkeypatch, tmp_path):
    seen_roots = []

    def fake_discover(roots):
        seen_roots.extend(Path(root) for root in roots)
        return []

    monkeypatch.setattr("gguf_limit_bench.cli.discover_models", fake_discover)

    result = runner.invoke(app, ["survey", "--json-out"])

    assert result.exit_code == 0
    # Relative config paths are anchored to the folder holding _CONFIG.toml
    # (or stay cwd-relative when no config file exists).
    from gguf_limit_bench.config import find_config_path

    found = find_config_path()
    expected = Path("_models") if found is None else found.parent / "_models"
    assert seen_roots == [expected]


def test_bulk_autoresearch_supports_total_budget_and_finish_early(tmp_path, monkeypatch):
    qwen_a = ModelInfo(path=tmp_path / "Qwen3-A-Q4_K_M.gguf", name="qwen-a", family="qwen")
    qwen_b = ModelInfo(path=tmp_path / "Qwen3-B-Q4_K_M.gguf", name="qwen-b", family="qwen")
    monkeypatch.setattr("gguf_limit_bench.cli.discover_models", lambda roots: [qwen_a, qwen_b])
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FastAttemptRunner)

    result = runner.invoke(
        app,
        [
            "autoresearch-all",
            "--speed-scout",
            "--runs-root",
            str(tmp_path / "runs"),
            "--total-budget-minutes",
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
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
    assert len(run_dirs) == 1
