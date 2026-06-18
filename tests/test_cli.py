import json
from pathlib import Path
import sys

import pytest
from typer.testing import CliRunner

from gguf_limit_bench.autoresearch import AttemptResult, PerplexityResult
from gguf_limit_bench.cli import app
from gguf_limit_bench.config import (
    BenchmarkSettings,
    DEFAULT_MODEL_ROOTS,
    PathSettings,
    PilotbenchConfig,
)
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


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--budget-minutes", "0"),
        ("--max-attempts", "-1"),
        ("--flag-context-size", "0"),
        ("--simple-bench-max-tokens", "0"),
    ],
)
def test_autoresearch_rejects_non_positive_numeric_options(monkeypatch, option, value):
    def fail_run(*args, **kwargs):
        raise AssertionError("invalid CLI input must fail before autoresearch starts")

    monkeypatch.setattr("gguf_limit_bench.cli._run_one_autoresearch", fail_run)

    result = runner.invoke(
        app,
        ["autoresearch", "--model", "model.gguf", option, value],
    )

    assert result.exit_code == 2
    assert "Invalid value" in result.output


@pytest.mark.parametrize("argument", ["--host=0.0.0.0", "--port", "--model=other.gguf"])
def test_autoresearch_rejects_extra_args_that_override_managed_server_fields(monkeypatch, argument):
    def fail_run(*args, **kwargs):
        raise AssertionError("unsafe extra args must fail before autoresearch starts")

    monkeypatch.setattr("gguf_limit_bench.cli._run_one_autoresearch", fail_run)

    result = runner.invoke(
        app,
        [
            "autoresearch",
            "--model",
            "model.gguf",
            "--flag-ladder",
            f"--llama-server-extra-arg={argument}",
        ],
        terminal_width=200,
    )

    assert result.exit_code == 2


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
            "--no-workflow-eval",
        ],
    )

    assert result.exit_code == 0
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
    assert len(run_dirs) == 1
    assert json.loads((run_dirs[0] / "best-settings.json").read_text(encoding="utf-8"))
    assert (tmp_path / "runs" / "learning" / "optuna.sqlite3").exists()


def test_autoresearch_command_accepts_context_ladder(tmp_path, monkeypatch):
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
            "--no-learning",
            "--no-ttft-probe",
            "--no-workflow-eval",
            "--context-ladder",
            "4096",
            "--context-ladder",
            "8192",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
    assert (
        json.loads((run_dirs[0] / "context-profile.json").read_text(encoding="utf-8"))["rows"][1][
            "context_size"
        ]
        == 8192
    )


def test_autoresearch_command_accepts_perplexity_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)

    class FakePerplexityRunner:
        def __init__(self, **kwargs) -> None:
            pass

        def __call__(self, settings):
            return PerplexityResult(
                ok=True,
                perplexity=6.0 + settings.context_size / 4096,
                stdout="Final estimate: PPL = 7.0",
                stderr="",
                returncode=0,
            )

    monkeypatch.setattr("gguf_limit_bench.cli.LlamaPerplexityRunner", FakePerplexityRunner)
    model = tmp_path / "Qwen3-Test-Q4_K_M.gguf"
    model.write_bytes(b"fake")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello local benchmark world\n", encoding="utf-8")
    llama_perplexity = tmp_path / "llama-perplexity.exe"
    llama_perplexity.write_text("fake", encoding="utf-8")

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
            "--no-learning",
            "--no-ttft-probe",
            "--no-workflow-eval",
            "--llama-perplexity",
            str(llama_perplexity),
            "--perplexity-corpus",
            str(corpus),
            "--perplexity-context",
            "4096",
            "--perplexity-context",
            "8192",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = [
        path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "learning"
    ]
    profile = json.loads((run_dirs[0] / "perplexity-profile.json").read_text(encoding="utf-8"))
    assert [row["context_size"] for row in profile["rows"]] == [4096, 8192]
    assert "Perplexity profile" in result.output


def test_autoresearch_command_accepts_benchmark_suite_plan(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    model = tmp_path / "Qwen3-Test-Q4_K_M.gguf"
    model.write_bytes(b"fake")
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 4096,
                "tasks": [
                    {
                        "id": "general",
                        "phase": "general",
                        "harness": "fake-general",
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'score': 0.6}))",
                        ],
                    },
                    {
                        "id": "agentic",
                        "phase": "agentic",
                        "harness": "fake-agentic",
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
            "autoresearch",
            "--model",
            str(model),
            "--runs-root",
            str(tmp_path / "runs"),
            "--budget-minutes",
            "1",
            "--max-attempts",
            "1",
            "--no-learning",
            "--no-ttft-probe",
            "--no-workflow-eval",
            "--benchmark-suite-plan",
            str(plan_path),
        ],
    )

    run_dirs = [
        path
        for path in (tmp_path / "runs").iterdir()
        if path.is_dir() and path.name != "learning" and "benchmark-suite" not in path.name
    ]
    best = json.loads((run_dirs[0] / "best-settings.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert best["score"] == 0.7
    assert best["result"]["agent_bench_score"] == 0.7
    assert (tmp_path / "runs" / "agent-bench-score.tsv").exists()
    assert "agent_bench_score" in (tmp_path / "runs" / "autoresearch-attempts.tsv").read_text(
        encoding="utf-8"
    )


def test_autoresearch_flag_ladder_dry_run_writes_plan_without_runner(tmp_path, monkeypatch):
    def fail_runner(*args, **kwargs):
        raise AssertionError("dry run should not create a benchmark runner")

    monkeypatch.setattr("gguf_limit_bench.cli.LlamaServerSimpleBenchAttemptRunner", fail_runner)
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
            "--llama-server",
            str(tmp_path / "llama-server.exe"),
            "--flag-ladder",
            "--dry-run",
            "--flag-context-size",
            "8192",
            "--parallel-max",
            "4",
            "--llama-server-extra-arg=--dry",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = list((tmp_path / "runs").iterdir())
    plan = json.loads((run_dirs[0] / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["context_size"] == 8192
    assert plan["profiles"][0]["name"] == "L0-baseline"
    assert "--dry" in plan["profiles"][0]["command"]


def test_autoresearch_dry_run_requires_flag_ladder(tmp_path, monkeypatch):
    def fail_runner(*args, **kwargs):
        raise AssertionError("invalid dry run must not create a benchmark runner")

    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", fail_runner)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    result = runner.invoke(
        app,
        [
            "autoresearch",
            "--model",
            str(model),
            "--runs-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--dry-run requires --flag-ladder" in result.output


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
            "--no-workflow-eval",
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


def test_default_model_roots_are_repo_relative():
    assert DEFAULT_MODEL_ROOTS == (Path("_models"),)


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


def test_first_run_option_sets_up_then_starts_from_config(tmp_path, monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "gguf_limit_bench.cli.load_config",
        lambda: PilotbenchConfig(
            paths=PathSettings(
                model_roots=(tmp_path,),
                llama_bench=tmp_path / "llama-bench.exe",
                llama_cli=tmp_path / "llama-cli.exe",
                llama_server=tmp_path / "llama-server.exe",
                llama_perplexity=tmp_path / "llama-perplexity.exe",
                runs_root=tmp_path / "_runs",
            ),
            benchmark=BenchmarkSettings(parallel_max=2, default_preset="deep"),
        ),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli._setup_app",
        lambda **kwargs: calls.append(("setup", kwargs)),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli._start_app",
        lambda **kwargs: calls.append(("start", kwargs)),
    )

    result = runner.invoke(app, ["--first-run"])

    assert result.exit_code == 0
    assert [name for name, _ in calls] == ["setup", "start"]
    assert calls[0][1]["install_command"] is True
    assert calls[0][1]["add_to_path"] is True
    assert calls[1][1]["preset"] == "deep"
    assert calls[1][1]["parallel_max"] == 2


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
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "local-model",
                "tasks": [
                    {"id": "general", "phase": "general", "command": [sys.executable, "-V"]},
                    {"id": "agentic", "phase": "agentic", "command": [sys.executable, "-V"]},
                ],
            }
        ),
        encoding="utf-8",
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
            "--benchmark-suite-plan",
            str(plan_path),
        ],
    )

    assert result.exit_code == 0
    assert "Starting research loop for 1 selected model" in result.output
    assert runs[0]["model"] == selected.path
    assert runs[0]["enable_mtp"] is True
    assert runs[0]["benchmark_suite_plan"] == plan_path


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
                    path="_models",
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
    assert opened_roots == [Path("_models")]


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
            "setup",
            "--root",
            str(tmp_path),
            "--db-path",
            str(db_path),
            "--runs-root",
            str(runs_root),
            "--shim-dir",
            str(tmp_path / "bin"),
            "--skip-env-sync",
            "--no-add-to-path",
        ],
    )

    assert result.exit_code == 0
    assert db_path.exists()
    assert (runs_root / "leaderboard.md").exists()
    assert (tmp_path / "bin" / "agent-autobench.bat").exists()
    assert (tmp_path / "bin" / "apb.bat").exists()
    assert "Setup wizard" in result.output
    assert "Setup is ready" in result.output
    assert "agent-autobench --start" in result.output


def test_setup_command_uses_resolved_runs_root_when_not_overridden(tmp_path, monkeypatch):
    expected_runs_root = tmp_path / "resolved-runs"

    monkeypatch.setattr(
        "gguf_limit_bench.cli.load_config",
        lambda: PilotbenchConfig(
            paths=PathSettings(
                model_roots=(tmp_path,),
                llama_bench=tmp_path / "llama-bench.exe",
                llama_cli=tmp_path / "llama-cli.exe",
                llama_server=tmp_path / "llama-server.exe",
                runs_root=expected_runs_root,
            ),
            benchmark=BenchmarkSettings(parallel_max=1, default_preset="quick"),
        ),
    )
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
            "setup",
            "--db-path",
            str(tmp_path / "db.sqlite"),
            "--shim-dir",
            str(tmp_path / "bin"),
            "--skip-env-sync",
            "--no-add-to-path",
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    assert (expected_runs_root / "leaderboard.md").exists()
    assert json.loads(result.output)["runs_root"] == str(expected_runs_root)


def test_setup_json_out_is_agent_friendly(tmp_path, monkeypatch):
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
            "setup",
            "--root",
            str(tmp_path),
            "--db-path",
            str(tmp_path / "db.sqlite"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--shim-dir",
            str(tmp_path / "bin"),
            "--skip-env-sync",
            "--no-add-to-path",
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


def test_results_command_can_open_browser_report(tmp_path, monkeypatch):
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
    opened: list[str] = []
    monkeypatch.setattr("gguf_limit_bench.cli.webbrowser.open", lambda url: opened.append(url))

    result = runner.invoke(
        app,
        ["results", "--runs-root", str(tmp_path / "runs"), "--open-browser"],
    )

    assert result.exit_code == 0
    assert opened == [(tmp_path / "runs" / "results.html").resolve().as_uri()]
    assert "Opened browser report" in result.output


def test_results_command_can_delegate_to_local_report_server(tmp_path, monkeypatch):
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
    served: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        "gguf_limit_bench.cli._serve_report_directory",
        lambda directory, port: served.append((directory, port)),
    )

    result = runner.invoke(
        app,
        ["results", "--runs-root", str(tmp_path / "runs"), "--serve", "--port", "8765"],
    )

    assert result.exit_code == 0
    assert served == [((tmp_path / "runs").resolve(), 8765)]
    assert "http://127.0.0.1:8765/results.html" in result.output


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
    assert payload["tasks"][0]["commands"][0][:5] == [
        "uvx",
        "--from",
        "lm-eval",
        "lm-eval",
        "run",
    ]
    assert payload["tasks"][0]["commands"][0][5:7] == ["--model", "local-chat-completions"]
    assert payload["tasks"][0]["commands"][1][4:8] == [
        "python",
        "-m",
        "gguf_limit_bench.score_extract",
        "--root",
    ]
    assert payload["tasks"][1]["commands"][0][4:7] == [
        "inspect",
        "eval",
        "benchmarks/inspect_tasks/json_repair.py",
    ]
    assert payload["tasks"][1]["commands"][1][4:8] == [
        "python",
        "-m",
        "gguf_limit_bench.inspect_score",
        "--log-dir",
    ]


def test_benchmark_suite_plans_lists_bundled_plans():
    result = runner.invoke(app, ["benchmark-suite-plans", "--json-out"])

    assert result.exit_code == 0
    plans = json.loads(result.output)
    normalized_plans = [path.replace("\\", "/") for path in plans]
    assert any(
        path.endswith("benchmarks/plans/local-openai-smoke.plan.json") for path in normalized_plans
    )
    assert any(
        path.endswith("benchmarks/plans/local-bfcl-smoke.plan.json") for path in normalized_plans
    )
