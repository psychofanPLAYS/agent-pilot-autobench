import json
from pathlib import Path
import sys

from typer.testing import CliRunner

from gguf_limit_bench.autoresearch import AttemptResult
from gguf_limit_bench.cli import app
from gguf_limit_bench.config import DEFAULT_MODEL_ROOTS
from gguf_limit_bench.discovery import ModelInfo
from gguf_limit_bench.doctor import DoctorCheck, DoctorReport


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
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
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
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
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
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
    assert len(run_dirs) == 1
    assert (
        "35B"
        in json.loads((run_dirs[0] / "best-settings.json").read_text(encoding="utf-8"))["model"]
    )


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
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
    assert len(run_dirs) == 1


def test_start_command_check_only_prints_beginner_next_step(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path=str(tmp_path),
                    detail="directory exists",
                )
            ]
        ),
    )

    result = runner.invoke(app, ["start", "--root", str(tmp_path), "--check-only"])

    assert result.exit_code == 0
    assert "Everything looks ready" in result.output
    assert "Remove --check-only to open the picker" in result.output


def test_start_command_opens_tui_after_ready_check(tmp_path, monkeypatch):
    opened_roots: list[Path] = []
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path=str(tmp_path),
                    detail="directory exists",
                )
            ]
        ),
    )

    class FakeBenchTui:
        models_to_run: list[ModelInfo] = []

        def __init__(self, root: Path, **kwargs) -> None:
            opened_roots.append(root)

        def run(self) -> None:
            pass

    monkeypatch.setattr("gguf_limit_bench.cli.BenchTui", FakeBenchTui)

    result = runner.invoke(app, ["start", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert opened_roots == [tmp_path]
    assert "Opening the model picker" in result.output
    assert "No models selected" in result.output


def test_start_command_runs_selected_models_from_tui(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path=str(tmp_path),
                    detail="directory exists",
                )
            ]
        ),
    )
    selected = ModelInfo(
        path=tmp_path / "Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
        name="Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
        family="qwen",
        parameters="35B-A3B",
        has_mtp=True,
    )
    runs: list[dict] = []

    class FakeBenchTui:
        def __init__(self, root: Path, **kwargs) -> None:
            self.models_to_run = [selected]

        def run(self) -> None:
            pass

    def fake_run_one_autoresearch(**kwargs):
        runs.append(kwargs)
        receipt = tmp_path / "runs" / "fake"
        receipt.mkdir(parents=True)
        return type("Receipt", (), {"path": receipt})()

    monkeypatch.setattr("gguf_limit_bench.cli.BenchTui", FakeBenchTui)
    monkeypatch.setattr("gguf_limit_bench.cli._run_one_autoresearch", fake_run_one_autoresearch)

    result = runner.invoke(
        app,
        [
            "start",
            "--root",
            str(tmp_path),
            "--runs-root",
            str(tmp_path / "runs"),
            "--budget-minutes",
            "1",
            "--max-attempts",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Starting research loop for 1 selected model" in result.output
    assert runs[0]["model"] == selected.path
    assert runs[0]["enable_mtp"] is True


def test_start_command_uses_preset_budget_when_budget_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path=str(tmp_path),
                    detail="directory exists",
                )
            ]
        ),
    )
    selected = ModelInfo(path=tmp_path / "Qwen3.gguf", name="Qwen3.gguf", family="qwen")
    runs: list[dict] = []

    class FakeBenchTui:
        def __init__(self, root: Path, **kwargs) -> None:
            self.models_to_run = [selected]

        def run(self) -> None:
            pass

    def fake_run_one_autoresearch(**kwargs):
        runs.append(kwargs)
        receipt = tmp_path / "runs" / "fake"
        receipt.mkdir(parents=True)
        return type("Receipt", (), {"path": receipt})()

    monkeypatch.setattr("gguf_limit_bench.cli.BenchTui", FakeBenchTui)
    monkeypatch.setattr("gguf_limit_bench.cli._run_one_autoresearch", fake_run_one_autoresearch)

    result = runner.invoke(app, ["start", "--root", str(tmp_path), "--preset", "deep"])

    assert result.exit_code == 0
    assert runs[0]["budget_seconds"] == 20 * 60


def test_start_command_exits_when_required_check_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="llama-bench",
                    status="missing",
                    path=str(tmp_path / "llama-bench.exe"),
                    detail="file was not found",
                )
            ]
        ),
    )

    result = runner.invoke(app, ["start", "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "Something is missing" in result.output
    assert "Run this first" in result.output


def test_global_start_flag_check_only_uses_beginner_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path=str(tmp_path),
                    detail="directory exists",
                )
            ]
        ),
    )

    result = runner.invoke(app, ["--start", "--check-only"])

    assert result.exit_code == 0
    assert "Everything looks ready" in result.output
    assert "Remove --check-only to open the picker" in result.output


def test_global_start_flag_opens_tui(monkeypatch):
    opened_roots: list[Path] = []
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path="G:/AI/models",
                    detail="directory exists",
                )
            ]
        ),
    )

    class FakeBenchTui:
        models_to_run: list[ModelInfo] = []

        def __init__(self, root: Path, **kwargs) -> None:
            opened_roots.append(root)

        def run(self) -> None:
            pass

    monkeypatch.setattr("gguf_limit_bench.cli.BenchTui", FakeBenchTui)

    result = runner.invoke(app, ["--start"])

    assert result.exit_code == 0
    assert opened_roots == [Path("G:/AI/models")]


def test_first_run_command_checks_paths_and_prepares_local_state(tmp_path, monkeypatch):
    db_path = tmp_path / "db" / "agentpilot.sqlite"
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="model root",
                    status="ok",
                    path=str(tmp_path),
                    detail="directory exists",
                )
            ]
        ),
    )

    result = runner.invoke(
        app,
        [
            "first-run",
            "--root",
            str(tmp_path),
            "--db-path",
            str(db_path),
            "--runs-root",
            str(runs_root),
            "--shim-dir",
            str(tmp_path / "bin"),
            "--skip-env-sync",
        ],
    )

    assert result.exit_code == 0
    assert db_path.exists()
    assert (runs_root / "leaderboard.md").exists()
    assert (tmp_path / "bin" / "agent-autobench.bat").exists()
    assert (tmp_path / "bin" / "apb.bat").exists()
    assert "First-time installer" in result.output
    assert "First-time setup is ready" in result.output
    assert "agent-autobench --start" in result.output


def test_first_run_json_out_is_agent_friendly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gguf_limit_bench.cli.build_doctor_report",
        lambda **kwargs: DoctorReport(
            checks=[
                DoctorCheck(
                    name="llama-bench",
                    status="missing",
                    path=str(tmp_path / "llama-bench.exe"),
                    detail="file was not found",
                )
            ]
        ),
    )

    result = runner.invoke(
        app,
        [
            "first-run",
            "--root",
            str(tmp_path),
            "--db-path",
            str(tmp_path / "db.sqlite"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--shim-dir",
            str(tmp_path / "bin"),
            "--skip-env-sync",
            "--json-out",
        ],
    )

    assert result.exit_code == 1
    assert "\x1b[" not in result.output
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert payload["install_ready"] is True
    assert payload["next_command"] == "agent-autobench doctor"
    assert any(step["name"] == "agent-autobench command" for step in payload["install_steps"])
    assert payload["checks"][0]["name"] == "llama-bench"


def test_results_command_prints_latest_champion(tmp_path):
    run = tmp_path / "runs" / "20260526-test"
    run.mkdir(parents=True)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 0, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "unknown",
                },
                "score": 51.0,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["results", "--runs-root", str(tmp_path / "runs")])

    assert result.exit_code == 0
    assert "Champion: Winner.gguf" in result.output
    assert (tmp_path / "runs" / "leaderboard.md").exists()
    assert (tmp_path / "runs" / "champion.json").exists()
    assert (tmp_path / "runs" / "results.html").exists()
    assert "HTML report" in result.output


def test_serve_probe_command_prints_real_serving_metrics(tmp_path, monkeypatch):
    class FakeServingProbeResult:
        ok = True
        ttft_ms = 321.0
        tokens_per_second = 27.5
        warm_ttft_ms = 111.0
        warm_tokens_per_second = 30.0
        warmup_penalty_ms = 210.0
        server_ready_ms = 1000.0
        cold_start_to_first_token_ms = 1321.0
        ttft_samples_ms = [321.0, 120.0, 102.0]
        tokens_cached_samples = [0, 7, 7]
        tokens_evaluated_samples = [7, 0, 0]
        generated_tokens = 12
        output_chars = 48
        failure = "none"
        stderr_tail = ""

        def to_dict(self):
            return {
                "ok": self.ok,
                "ttft_ms": self.ttft_ms,
                "tokens_per_second": self.tokens_per_second,
                "warm_ttft_ms": self.warm_ttft_ms,
                "warmup_penalty_ms": self.warmup_penalty_ms,
                "server_ready_ms": self.server_ready_ms,
                "cold_start_to_first_token_ms": self.cold_start_to_first_token_ms,
                "generated_tokens": self.generated_tokens,
                "output_chars": self.output_chars,
                "failure": self.failure,
            }

    calls: list[dict] = []

    def fake_probe(**kwargs):
        calls.append(kwargs)
        return FakeServingProbeResult()

    monkeypatch.setattr("gguf_limit_bench.cli.probe_llama_server_ttft", fake_probe)
    model = tmp_path / "Qwen3-Test-Q4_K_M.gguf"
    model.write_bytes(b"fake")

    result = runner.invoke(app, ["serve-probe", "--model", str(model), "--context-size", "8192"])

    assert result.exit_code == 0
    assert calls[0]["model"] == model
    assert calls[0]["settings"].context_size == 8192
    assert calls[0]["samples"] == 0
    assert calls[0]["cache_prompt"] is True
    assert "Cold TTFT: 321 ms" in result.output
    assert "Warm TTFT: 111 ms" in result.output
    assert "Warmup penalty: 210 ms" in result.output
    assert "Serving speed: 27.50 tok/s" in result.output


def test_packs_command_lists_builtin_benchmark_packs():
    result = runner.invoke(app, ["packs"])

    assert result.exit_code == 0
    assert "hermes-pilot" in result.output
    assert "context-limit" in result.output


def test_benchmark_suite_command_runs_plan_and_writes_ledgers(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 32768,
                "tasks": [
                    {
                        "id": "general",
                        "phase": "general",
                        "harness": "lm-evaluation-harness",
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'score': 0.6}))",
                        ],
                    },
                    {
                        "id": "agentic",
                        "phase": "agentic",
                        "harness": "inspect-ai",
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'score': 0.8}))",
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "benchmark-suite",
            "--plan",
            str(plan_path),
            "--runs-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert "agent_bench_score: 0.700000" in result.output
    assert (tmp_path / "runs" / "benchmark-suite.tsv").exists()
    assert (tmp_path / "runs" / "agentic-suite.tsv").exists()


def test_benchmark_suite_template_writes_real_harness_plan(tmp_path):
    output = tmp_path / "benchmark-suite.plan.json"

    result = runner.invoke(
        app,
        [
            "benchmark-suite-template",
            "--output",
            str(output),
            "--model",
            "qwen-local",
            "--base-url",
            "http://127.0.0.1:8080/v1",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "Benchmark-suite plan written" in result.output
    assert payload["tasks"][0]["command"][4:7] == ["lm-eval", "run", "--model"]
    assert payload["tasks"][1]["command"][4:7] == ["inspect", "eval", "path/to/inspect_task.py"]
