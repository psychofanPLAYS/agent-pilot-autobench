import json
from pathlib import Path
import re
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
from gguf_limit_bench.programs import MIN_SERIOUS_CONTEXT_SIZE


runner = CliRunner()


def _path_arg(path: Path) -> str:
    return path.as_posix()


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
        ("--flag-context-size", "4096"),
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


def test_qe_format_command_writes_fresh_session_receipt(tmp_path, monkeypatch):
    calls = []

    def fake_run_qe_format_suite(**kwargs):
        calls.append(kwargs)
        kwargs["out_dir"].mkdir(parents=True)
        (kwargs["out_dir"] / "qe-format-summary.json").write_text(
            json.dumps({"score": 0.75, "attempts": 20}),
            encoding="utf-8",
        )
        return {
            "score": 0.75,
            "format_rate": 0.8,
            "direct_answer_rate": 0.1,
            "attempts": 20,
        }

    monkeypatch.setattr("gguf_limit_bench.cli.run_qe_format_suite", fake_run_qe_format_suite)

    result = runner.invoke(
        app,
        [
            "qe-format",
            "--model",
            "qwen3.5-qe-2b",
            "--base-url",
            "http://127.0.0.1:8081",
            "--runs-root",
            str(tmp_path),
            "--repeats",
            "10",
            "--timeout-seconds",
            "77",
            "--max-tokens",
            "96",
            "--temperature",
            "0.1",
            "--top-p",
            "0.8",
            "--min-p",
            "0.02",
            "--repeat-penalty",
            "1.05",
            "--dry-multiplier",
            "0.6",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["model"] == "qwen3.5-qe-2b"
    assert calls[0]["base_url"] == "http://127.0.0.1:8081"
    assert calls[0]["repeats"] == 10
    assert calls[0]["timeout_seconds"] == 77
    assert calls[0]["answer_max_tokens"] == 96
    assert calls[0]["sampling"] == {
        "temperature": 0.1,
        "top_p": 0.8,
        "min_p": 0.02,
        "repeat_penalty": 1.05,
        "dry_multiplier": 0.6,
    }
    assert calls[0]["out_dir"].parent == tmp_path
    assert "QE format receipt:" in result.output
    assert "Score: 0.750000" in result.output


def test_qe_results_command_prints_hard_recommendation(tmp_path):
    run = tmp_path / "runs" / "qe-good"
    run.mkdir(parents=True)
    (run / "qe-format-summary.json").write_text(
        json.dumps(
            {
                "model": "qwen-qe",
                "score": 0.95,
                "format_rate": 0.94,
                "direct_answer_rate": 0.0,
                "attempts": 50,
                "median_tps": 180.0,
                "median_ttft_ms": 125.0,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["qe-results", "--runs-root", str(tmp_path / "runs")])

    assert result.exit_code == 0, result.output
    assert "QE champion: qwen-qe" in result.output
    assert "Action: PROMOTE_QE_PROFILE" in result.output
    assert "Leaderboard written:" in result.output
    assert (tmp_path / "runs" / "qe-format-leaderboard.md").exists()


def test_qe_results_command_labels_retest_as_top_candidate_not_champion(tmp_path):
    run = tmp_path / "runs" / "qe-retest"
    run.mkdir(parents=True)
    (run / "qe-format-summary.json").write_text(
        json.dumps(
            {
                "model": "qwen-qe",
                "score": 0.78,
                "format_rate": 0.78,
                "direct_answer_rate": 0.0,
                "attempts": 50,
                "median_tps": 180.0,
                "median_ttft_ms": 125.0,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["qe-results", "--runs-root", str(tmp_path / "runs")])

    assert result.exit_code == 0, result.output
    assert "QE top candidate: qwen-qe" in result.output
    assert "QE champion:" not in result.output
    assert "Action: RETEST_QE_PROFILE" in result.output


def test_flag_recommendations_command_writes_operator_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.detect_gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    model = tmp_path / "gemma-4-26B-A4B-it-Q4_K_M.gguf"
    model.touch()
    llama_server = tmp_path / "llama-server.exe"
    llama_server.touch()

    result = runner.invoke(
        app,
        [
            "flag-recommendations",
            "--model",
            str(model),
            "--llama-server",
            str(llama_server),
            "--output-dir",
            str(tmp_path / "runs"),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Flag recommendations written:" in result.output
    assert "Standard: ctx 131072" in result.output
    assert "Long agent: ctx 200000" in result.output
    assert (tmp_path / "runs" / "flag-recommendations.md").exists()
    payload = json.loads((tmp_path / "runs" / "flag-recommendations.json").read_text())
    assert payload["profiles"][0]["command"][:3] == [
        str(llama_server),
        "--model",
        str(model),
    ]


def test_deployment_readiness_command_prints_gate_result(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "model_name": "Winner.gguf",
                "lane_type": "chat_agent",
                "profiles": [{"id": "standard", "label": "Standard", "context_size": 131072}],
            }
        ),
        encoding="utf-8",
    )
    run = runs_root / "winner-run"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 131072, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 44.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.81,
                    "serving_ttft_ms": 500.0,
                    "serving_tokens_per_second": 32.0,
                },
                "score": 0.81,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["deployment-readiness", "--runs-root", str(runs_root)])

    assert result.exit_code == 0, result.output
    assert "Deployment readiness: PROMOTE_DEPLOYMENT_PROFILE" in result.output
    assert "Recommended profile: standard" in result.output
    assert (runs_root / "deployment-readiness.md").exists()


def test_hard_recommendations_command_prints_single_operator_summary(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "model_name": "Winner.gguf",
                "lane_type": "chat_agent",
                "profiles": [{"id": "standard", "label": "Standard", "context_size": 131072}],
            }
        ),
        encoding="utf-8",
    )
    run = runs_root / "speed-only"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 131072, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "serving_ttft_ms": 500.0,
                    "serving_tokens_per_second": 36.0,
                },
                "score": 42.0,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["hard-recommendations", "--runs-root", str(runs_root)])

    assert result.exit_code == 0, result.output
    assert "Hard recommendations: RETEST" in result.output
    assert "Operator verdict: NOT_USABLE_YET" in result.output
    assert "No deployable recommendation exists." in result.output
    assert "Scored candidates: 0/1" in result.output
    assert "Performance prediction: LAB_ONLY_SPEED_PROOF (high risk)" in result.output
    assert "Deployment expectation: do_not_deploy" in result.output
    assert "Proven recommendations: 0" in result.output
    assert "Settings candidates:" in result.output
    assert "#1 standard | SYSTEMS_ONLY | needs_agent_score | ctx=131072" in result.output
    assert "Proof commands: " in result.output
    assert "Model gate: RETEST" in result.output
    assert "Deployment gate: RETEST_DEPLOYMENT" in result.output
    assert "Context gate: WAITING_FOR_DEPLOYMENT | required=131072" in result.output
    assert (
        "Resource gate: WAITING_FOR_DEPLOYMENT | "
        "required=same-run resource telemetry for the promoted settings receipt"
    ) in " ".join(result.output.split())
    assert "Candidate readiness: not_recommendable (0/100)" in result.output
    output_words = " ".join(result.output.split())
    assert (
        "Candidate performance: quality=unmeasured speed=interactive context=long_agentic"
        in output_words
    )
    assert "Candidate rankings:" in result.output
    assert "#1 Winner.gguf" in result.output
    assert "gaps=agent_quality, benchmark_suite" in result.output
    assert "Repeatability: single_run (1 run)" in result.output
    assert "Proof command (model/model_plan):" in result.output
    assert "Proof command (model/model_score):" in result.output
    assert "Proof runbook:" in result.output
    assert "1. [model/model_plan] pending -> benchmark-suite.plan.json" in output_words
    payload = json.loads((runs_root / "hard-recommendations.json").read_text(encoding="utf-8"))
    assert payload["proof_runbook"][1]["proves"] == (
        f"{_path_arg(runs_root)}/<suite-run>/suite-verdict.json"
    )
    assert (
        f"apb benchmark-suite --plan benchmark-suite.plan.json --runs-root {_path_arg(runs_root)}"
    ) in result.output
    assert "Proof command (refresh/refresh_hard_recommendations):" in result.output
    assert (runs_root / "hard-recommendations.md").exists()


def test_hard_recommendations_command_preserves_required_context_in_refresh_command(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "model_name": "Winner.gguf",
                "lane_type": "chat_agent",
                "profiles": [
                    {"id": "standard", "label": "Standard", "context_size": 131072},
                    {"id": "long_agent", "label": "Long agent", "context_size": 200000},
                ],
            }
        ),
        encoding="utf-8",
    )
    run = runs_root / "standard-proof"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 131072, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.82,
                    "benchmark_suite_ok": True,
                    "benchmark_suite_general_score": 0.82,
                    "benchmark_suite_agentic_score": 0.82,
                    "serving_ttft_ms": 420.0,
                    "serving_tokens_per_second": 38.0,
                },
                "score": 0.82,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "hard-recommendations",
            "--runs-root",
            str(runs_root),
            "--required-context",
            "200000",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert payload["context_gate"]["action"] == "RETEST_CONTEXT"
    assert commands["refresh_hard_recommendations"].endswith("--required-context 200000")


def test_hard_recommendations_command_accepts_target_model_scope(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Qwopus.gguf",
                "model_name": "Qwopus.gguf",
                "lane_type": "chat_agent",
                "profiles": [{"id": "standard", "label": "Standard", "context_size": 131072}],
            }
        ),
        encoding="utf-8",
    )
    run = runs_root / "old-qwopus-speed"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Qwopus.gguf",
                "settings": {"context_size": 4096, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "serving_ttft_ms": 500.0,
                    "serving_tokens_per_second": 36.0,
                },
                "score": 42.0,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "hard-recommendations",
            "--runs-root",
            str(runs_root),
            "--target-model",
            "Gemma-4-26B",
            "--target-model-path",
            "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
        ],
    )

    assert result.exit_code == 0, result.output
    normalized_output = result.output.replace("\\", "/")
    assert "Target scope: Gemma-4-26B | NO_TARGET_EVIDENCE | matched 0, ignored 1" in result.output
    assert "Scored candidates: 0/0" in result.output
    assert (
        'apb flag-recommendations --model "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf" '
        f"--output-dir {_path_arg(runs_root)}"
    ) in normalized_output
    payload = json.loads((runs_root / "hard-recommendations.json").read_text(encoding="utf-8"))
    assert payload["target_scope"]["target_model"] == "Gemma-4-26B"
    assert payload["target_scope"]["target_model_path"].replace("\\", "/") == (
        "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf"
    )
    assert payload["candidate_rankings"] == []


def test_tui_command_forwards_target_scope_and_required_context(tmp_path, monkeypatch):
    captured = {}

    class FakeBenchTui:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models_to_run = []
            self.ran_inside_tui = False
            self.run_mode = type(
                "RunMode",
                (),
                {
                    "budget_minutes": 5,
                    "context_ladder": None,
                    "evaluation": None,
                },
            )()

        def run(self):
            return None

    monkeypatch.setattr("gguf_limit_bench.cli.BenchTui", FakeBenchTui)
    monkeypatch.setattr("gguf_limit_bench.cli._run_tui_selection", lambda **kwargs: None)
    model_root = tmp_path / "models"
    model_root.mkdir()

    result = runner.invoke(
        app,
        [
            "tui",
            "--root",
            str(model_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--target-model",
            "Gemma-4-26B",
            "--target-model-path",
            "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
            "--required-context",
            "200000",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["target_model"] == "Gemma-4-26B"
    assert captured["target_model_path"].replace("\\", "/") == (
        "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf"
    )
    assert captured["required_context"] == 200000


def test_deployment_proof_command_runs_selected_profile(tmp_path, monkeypatch):
    calls = []
    receipt_path = tmp_path / "runs" / "deployment-proof-receipt"
    receipt_path.mkdir(parents=True)
    (receipt_path / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
                "settings": {"profile_name": "long_agent", "context_size": 200000},
            }
        ),
        encoding="utf-8",
    )
    refreshed = []

    def fake_run_deployment_proof(**kwargs):
        calls.append(kwargs)
        return type("Receipt", (), {"path": receipt_path})()

    def fake_write_deployment_readiness(runs_root):
        refreshed.append(("readiness", runs_root))
        path = runs_root / "deployment-readiness.json"
        path.write_text(
            json.dumps(
                {
                    "action": "PROMOTE_DEPLOYMENT_PROFILE",
                    "recommended_profile_id": "standard",
                }
            ),
            encoding="utf-8",
        )
        return type(
            "Outputs",
            (),
            {"json_path": path, "markdown_path": runs_root / "deployment-readiness.md"},
        )()

    def fake_write_hard_recommendations(runs_root, **kwargs):
        refreshed.append(("hard", runs_root, kwargs))
        path = runs_root / "hard-recommendations.json"
        payload = {
            "overall_action": "PROMOTE_READY_STACK",
            "hard_recommendations": [{"type": "settings_profile"}],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return type(
            "Outputs",
            (),
            {
                "json_path": path,
                "markdown_path": runs_root / "hard-recommendations.md",
                "payload": payload,
            },
        )()

    monkeypatch.setattr("gguf_limit_bench.cli.run_deployment_proof", fake_run_deployment_proof)
    monkeypatch.setattr(
        "gguf_limit_bench.cli.write_deployment_readiness", fake_write_deployment_readiness
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli.write_hard_recommendations", fake_write_hard_recommendations
    )

    result = runner.invoke(
        app,
        [
            "deployment-proof",
            "--runs-root",
            str(tmp_path / "runs"),
            "--profile",
            "standard",
            "--flag-recommendations",
            str(tmp_path / "runs" / "flag-recommendations.json"),
            "--benchmark-suite-plan",
            str(tmp_path / "benchmark-suite.plan.json"),
            "--budget-minutes",
            "17",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["runs_root"] == tmp_path / "runs"
    assert calls[0]["profile_id"] == "standard"
    assert calls[0]["flag_recommendations_path"] == tmp_path / "runs" / "flag-recommendations.json"
    assert calls[0]["benchmark_suite_plan"] == tmp_path / "benchmark-suite.plan.json"
    assert calls[0]["budget_seconds"] == 17 * 60
    assert calls[0]["simple_bench_max_tokens"] == 8192
    assert "Deployment proof receipt:" in result.output
    assert str(receipt_path) in result.output
    assert refreshed == [
        ("readiness", tmp_path / "runs"),
        (
            "hard",
            tmp_path / "runs",
            {
                "target_model_path": "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
                "required_context": 200000,
            },
        ),
    ]
    assert "Deployment readiness: PROMOTE_DEPLOYMENT_PROFILE" in result.output
    assert "Hard recommendations: PROMOTE_READY_STACK" in result.output
    assert "Proven recommendations: 1" in result.output


def test_autoresearch_command_writes_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    model = tmp_path / "Qwen3-Test-Q4_K_M.gguf"
    model.write_bytes(b"fake")

    result = runner.invoke(
        app,
        [
            "autoresearch",
            "--speed-scout",
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
            "--speed-scout",
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
            str(MIN_SERIOUS_CONTEXT_SIZE),
            "--parallel-max",
            "4",
            "--llama-server-extra-arg=--dry",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = list((tmp_path / "runs").iterdir())
    plan = json.loads((run_dirs[0] / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["context_size"] == MIN_SERIOUS_CONTEXT_SIZE
    assert plan["profiles"][0]["name"] == "Lmin-stripped"
    assert "--dry" in plan["profiles"][0]["command"]


def test_autoresearch_flag_ladder_dry_run_defaults_to_16k_context(tmp_path, monkeypatch):
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
            "--parallel-max",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dirs = list((tmp_path / "runs").iterdir())
    plan = json.loads((run_dirs[0] / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    assert plan["context_size"] == MIN_SERIOUS_CONTEXT_SIZE
    assert all(
        profile["settings"]["context_size"] >= MIN_SERIOUS_CONTEXT_SIZE
        for profile in plan["profiles"]
    )


def test_autoresearch_dry_run_rejects_speed_scout(tmp_path, monkeypatch):
    def fail_runner(*args, **kwargs):
        raise AssertionError("invalid dry run must not create a benchmark runner")

    monkeypatch.setattr("gguf_limit_bench.cli.LlamaServerSimpleBenchAttemptRunner", fail_runner)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake")

    # Dry-run plans the flag ladder, which only exists in benchmark mode. Asking
    # for the synthetic speed scout and a dry run at the same time is incoherent.
    result = runner.invoke(
        app,
        [
            "autoresearch",
            "--model",
            str(model),
            "--runs-root",
            str(tmp_path / "runs"),
            "--dry-run",
            "--speed-scout",
        ],
    )

    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)


def test_autoresearch_all_qwen_only_skips_non_qwen_models(tmp_path, monkeypatch):
    monkeypatch.setattr("gguf_limit_bench.cli.LlamaBenchAttemptRunner", FakeAttemptRunner)
    qwen = ModelInfo(path=tmp_path / "Qwen3-Test-Q4_K_M.gguf", name="qwen", family="qwen")
    llama = ModelInfo(path=tmp_path / "Llama-Test-Q4_K_M.gguf", name="llama", family="llama")
    monkeypatch.setattr("gguf_limit_bench.cli.discover_models", lambda roots: [qwen, llama])

    result = runner.invoke(
        app,
        [
            "autoresearch-all",
            "--speed-scout",
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
            "--speed-scout",
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


def test_start_command_opens_webui_after_ready_check(tmp_path, monkeypatch):
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

    def fake_serve_webui(**kwargs) -> str:
        opened_roots.append(kwargs["root"])
        return "http://127.0.0.1:9999/"

    monkeypatch.setattr("gguf_limit_bench.cli.serve_webui", fake_serve_webui)

    result = runner.invoke(app, ["start", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert opened_roots == [tmp_path]
    assert "Opening the browser cockpit" in result.output


def test_engine_command_wires_librarian_packs_mtp_budget_and_plan(tmp_path, monkeypatch):
    # The per-model evaluation logic that used to live in the web run_model
    # callback now lives in the detached `engine` command. The web side is thin.
    from gguf_limit_bench import run_dir

    rd = tmp_path / "run"
    rd.mkdir()
    model_path = tmp_path / "Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    run_dir.write_spec(
        rd,
        {
            "models": [{"path": str(model_path), "has_mtp": True}],
            "mode": "librarian_bench",
            "options": {
                "budget_minutes": 7,
                "forced_server_args": ["--jinja"],
                "benchmark_suite_plan": str(plan_path),
            },
        },
    )
    runs: list[dict] = []

    def fake_run_one_autoresearch(**kwargs):
        runs.append(kwargs)
        receipt = tmp_path / "runs" / "fake"
        receipt.mkdir(parents=True, exist_ok=True)
        return type("Receipt", (), {"path": receipt})()

    monkeypatch.setattr("gguf_limit_bench.cli._run_one_autoresearch", fake_run_one_autoresearch)

    result = runner.invoke(app, ["engine", "--run-dir", str(rd)])

    assert result.exit_code == 0, result.output
    assert runs[0]["model"] == model_path
    assert runs[0]["enable_mtp"] is True
    assert runs[0]["budget_seconds"] == 7 * 60
    assert "--jinja" in runs[0]["forced_server_args"]
    assert runs[0]["benchmark_suite_plan"] == plan_path
    assert "librarian-gate" in runs[0]["champion_pack_ids"]


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


def test_global_start_flag_opens_webui(monkeypatch):
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

    def fake_serve_webui(**kwargs) -> str:
        opened_roots.append(kwargs["root"])
        return "http://127.0.0.1:9999/"

    monkeypatch.setattr("gguf_limit_bench.cli.serve_webui", fake_serve_webui)

    result = runner.invoke(app, ["--start"])

    assert result.exit_code == 0
    # Relative config paths are anchored to the folder holding _CONFIG.toml
    # (or stay cwd-relative when no config file exists).
    from gguf_limit_bench.config import find_config_path

    found = find_config_path()
    expected = Path("_models") if found is None else found.parent / "_models"
    assert opened_roots == [expected]


def test_bare_apb_launches_without_setup_when_already_installed(monkeypatch):
    """Once installed, plain `apb` (no command, no flags) just opens the app."""
    calls: list[str] = []
    monkeypatch.setattr("gguf_limit_bench.cli.is_setup_complete", lambda _root: True)
    monkeypatch.setattr(
        "gguf_limit_bench.cli._setup_app",
        lambda **kwargs: calls.append("setup"),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli._start_app",
        lambda **kwargs: calls.append("start"),
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    # No flag soup, no setup: it went straight to launching.
    assert calls == ["start"]
    assert "Usage:" not in result.output


def test_bare_apb_self_installs_on_first_run_then_launches(monkeypatch):
    """First-ever `apb` detects it is not set up, runs setup, then launches."""
    calls: list[str] = []
    monkeypatch.setattr("gguf_limit_bench.cli.is_setup_complete", lambda _root: False)
    monkeypatch.setattr(
        "gguf_limit_bench.cli._setup_app",
        lambda **kwargs: calls.append("setup"),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli._start_app",
        lambda **kwargs: calls.append("start"),
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert calls == ["setup", "start"]
    assert "First run detected" in result.output


def test_vram_plan_reports_fitting_contexts(tmp_path, monkeypatch):
    from gguf_limit_bench.gguf_metadata import ModelArch
    from gguf_limit_bench.vram import VramInfo

    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 1024)

    monkeypatch.setattr(
        "gguf_limit_bench.cli.read_model_arch",
        lambda _path: ModelArch(
            architecture="gemma4",
            n_layers=35,
            n_heads=8,
            n_heads_kv=1,
            embedding_length=1536,
            key_length=512,
            value_length=512,
            train_context_length=131072,
        ),
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli.detect_vram_mb",
        lambda: VramInfo(total_mb=24564, free_mb=22000),
    )

    result = runner.invoke(
        app,
        ["vram-plan", "--model", str(model), "--kv-bits", "8", "--json-out"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["architecture"] == "gemma4"
    assert payload["max_fitting_context"] == 262144
    assert payload["vram_total_mb"] == 24564


def test_context_limit_climbs_and_recovers_from_oom(tmp_path, monkeypatch):
    from gguf_limit_bench.server_probe import ServingProbeResult

    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")

    monkeypatch.setattr(
        "gguf_limit_bench.cli.load_config",
        lambda: PilotbenchConfig(
            paths=PathSettings(model_roots=(tmp_path,), llama_server=server),
        ),
    )
    # No VRAM guard interference in this test.
    monkeypatch.setattr("gguf_limit_bench.cli.read_model_arch", lambda _p: None)
    monkeypatch.setattr("gguf_limit_bench.cli.detect_vram_mb", lambda: None)

    def fake_probe(*, settings, **kwargs):
        if settings.context_size > 65_536:
            return ServingProbeResult(
                ok=False,
                ttft_ms=None,
                tokens_per_second=0.0,
                output_chars=0,
                generated_tokens=0,
                failure="oom",
                stderr_tail="cudaMalloc failed: out of memory",
            )
        return ServingProbeResult(
            ok=True,
            ttft_ms=10.0,
            tokens_per_second=80.0,
            output_chars=10,
            generated_tokens=8,
            failure="none",
        )

    monkeypatch.setattr("gguf_limit_bench.cli.probe_llama_server_ttft", fake_probe)

    result = runner.invoke(
        app,
        ["context-limit", "--model", str(model), "--no-refine", "--json-out"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kv_cache_type"] == "q8_0"  # q8_0 is the default
    assert payload["max_context"] == 65_536
    assert payload["hit_oom"] is True


def test_vram_plan_errors_when_metadata_unreadable(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("gguf_limit_bench.cli.read_model_arch", lambda _path: None)

    result = runner.invoke(app, ["vram-plan", "--model", str(model)])

    assert result.exit_code == 1
    assert "Could not read GGUF" in result.output


def test_autoconfigure_detects_and_persists_missing_paths(tmp_path):
    from gguf_limit_bench.cli import _autoconfigure_paths

    models = tmp_path / "AI" / "models"
    models.mkdir(parents=True)
    server = tmp_path / "llama" / "llama-server.exe"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"")

    # Config points at non-existent default paths (fresh machine).
    config = PilotbenchConfig(
        paths=PathSettings(
            model_roots=(tmp_path / "_models",),
            llama_bench=tmp_path / "_llama" / "llama-bench.exe",
            llama_cli=tmp_path / "_llama" / "llama-cli.exe",
            llama_server=tmp_path / "_llama" / "llama-server.exe",
            llama_perplexity=tmp_path / "_llama" / "llama-perplexity.exe",
        )
    )
    persisted: dict[str, str] = {}

    new_config, steps = _autoconfigure_paths(
        config,
        detect_models=lambda: [models],
        detect_binaries=lambda: {"llama-server": server},
        persist=lambda values: persisted.update(values) or "step",
    )

    assert persisted["PILOTBENCH_MODEL_ROOTS"] == str(models)
    assert persisted["PILOTBENCH_LLAMA_SERVER"] == str(server)
    assert len(steps) == 1
    # new_config came from a real reload; it is a valid config object.
    assert new_config is not None


def test_autoconfigure_leaves_existing_paths_untouched(tmp_path):
    from gguf_limit_bench.cli import _autoconfigure_paths

    # Everything already resolves: detection must not run, nothing persisted.
    (tmp_path / "models").mkdir()
    for name in ("llama-bench.exe", "llama-cli.exe", "llama-server.exe", "llama-perplexity.exe"):
        (tmp_path / name).write_bytes(b"")
    config = PilotbenchConfig(
        paths=PathSettings(
            model_roots=(tmp_path / "models",),
            llama_bench=tmp_path / "llama-bench.exe",
            llama_cli=tmp_path / "llama-cli.exe",
            llama_server=tmp_path / "llama-server.exe",
            llama_perplexity=tmp_path / "llama-perplexity.exe",
        )
    )

    def _boom():
        raise AssertionError("detection should not run when paths already exist")

    new_config, steps = _autoconfigure_paths(
        config,
        detect_models=_boom,
        detect_binaries=_boom,
        persist=lambda values: (_ for _ in ()).throw(AssertionError("must not persist")),
    )

    assert steps == []
    assert new_config is config


def test_bare_apb_with_subcommand_does_not_trigger_setup_or_launch(tmp_path, monkeypatch):
    """A subcommand (e.g. `apb results`) must not be hijacked by the launcher."""
    calls: list[str] = []
    monkeypatch.setattr(
        "gguf_limit_bench.cli.is_setup_complete",
        lambda _root: calls.append("checked") or True,
    )
    monkeypatch.setattr(
        "gguf_limit_bench.cli._start_app",
        lambda **kwargs: calls.append("start"),
    )

    # Point at a tmp runs-root so the real `results` command can't pollute the repo.
    result = runner.invoke(app, ["results", "--runs-root", str(tmp_path / "runs")])

    assert result.exit_code == 0
    # The callback returned early: neither the install check nor launch ran.
    assert calls == []


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
    assert "Next command: apb" in result.output


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
    assert payload["next_command"] == "apb doctor"
    assert any(step["name"] == "agent-autobench command" for step in payload["install_steps"])
    assert payload["checks"][0]["name"] == "llama-bench"


def test_results_command_prints_unproven_result_as_top_candidate(tmp_path):
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
    assert "Top candidate: Winner.gguf" in result.output
    assert "Champion: Winner.gguf" not in result.output
    assert (tmp_path / "runs" / "leaderboard.md").exists()
    assert (tmp_path / "runs" / "champion.json").exists()
    assert (tmp_path / "runs" / "verdict.json").exists()
    assert (tmp_path / "runs" / "verdict.md").exists()
    assert (tmp_path / "runs" / "hard-recommendations.json").exists()
    assert (tmp_path / "runs" / "hard-recommendations.md").exists()
    assert (tmp_path / "runs" / "report-audit.json").exists()
    assert (tmp_path / "runs" / "report-audit.md").exists()
    assert (tmp_path / "runs" / "results.html").exists()
    assert "Verdict: RETEST" in result.output
    assert "Report audit: warning (1 warning(s))" in result.output
    assert "Audit warning: missing_agent_quality in 20260526-test" in result.output
    assert "Predicted quality: unmeasured" in result.output
    assert "Recommendation class: needs_agent_benchmark" in result.output
    assert "Candidate readiness: not_recommendable (0/100)" in result.output
    assert "Candidate rankings:" in result.output
    assert "#1 Winner.gguf" in result.output
    assert "Repeatability: single_run (1 run)" in result.output
    assert "Proof runbook:" in result.output
    assert "1. [model/model_plan] pending -> benchmark-suite.plan.json" in " ".join(
        result.output.split()
    )
    assert "Hard recommendations:" in result.output
    assert "HTML report" in result.output


def test_results_command_target_scope_does_not_show_unrelated_top_candidate(tmp_path):
    run = tmp_path / "runs" / "old-qwopus-speed"
    run.mkdir(parents=True)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Qwopus.gguf",
                "settings": {"context_size": 4096, "parallel": 1, "gpu_layers": 99},
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

    result = runner.invoke(
        app,
        [
            "results",
            "--runs-root",
            str(tmp_path / "runs"),
            "--target-model",
            "Gemma-4-26B",
            "--target-model-path",
            "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
        ],
    )

    assert result.exit_code == 0, result.output
    normalized_output = result.output.replace("\\", "/")
    assert "No benchmark receipts found for target model: Gemma-4-26B" in result.output
    assert "Top candidate: Qwopus.gguf" not in result.output
    assert "Target scope: Gemma-4-26B | NO_TARGET_EVIDENCE | matched 0, ignored 1" in result.output
    assert (
        'apb flag-recommendations --model "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf" '
        f"--output-dir {_path_arg(tmp_path / 'runs')}"
    ) in normalized_output
    payload = json.loads(
        (tmp_path / "runs" / "hard-recommendations.json").read_text(encoding="utf-8")
    )
    assert payload["target_scope"]["status"] == "NO_TARGET_EVIDENCE"


def test_results_command_prints_score_summary_before_verdict_for_suite_backed_result(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "model_name": "Winner.gguf",
                "profiles": [{"id": "standard", "label": "Standard", "context_size": 131072}],
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "runs" / "20260526-suite-backed"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 131072, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.82,
                    "benchmark_suite_ok": True,
                    "benchmark_suite_general_score": 0.78,
                    "benchmark_suite_agentic_score": 0.86,
                    "serving_ttft_ms": 420.0,
                    "serving_tokens_per_second": 38.0,
                },
                "score": 0.82,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["results", "--runs-root", str(tmp_path / "runs")])

    assert result.exit_code == 0, result.output
    assert result.output.index("Benchmark scores:") < result.output.index("Verdict: PROMOTE")
    assert "Score contract: agent_bench_score" in result.output
    assert "Agent bench score: 0.820000" in result.output
    assert "General score: 0.780000" in result.output
    assert "Agentic score: 0.860000" in result.output
    assert "Generation speed: 42.00 tok/s" in result.output
    assert "Serving speed: 38.00 tok/s" in result.output
    assert "Settings candidates:" in result.output
    assert "#1 standard | PROVEN | recommended | ctx=131072" in result.output


def test_results_json_out_includes_verdict_and_audit_for_unproven_candidate(tmp_path):
    run = tmp_path / "runs" / "20260526-speed-only"
    run.mkdir(parents=True)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 262144, "parallel": 1, "gpu_layers": 99},
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

    result = runner.invoke(
        app,
        ["results", "--runs-root", str(tmp_path / "runs"), "--json-out"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result_label"] == "top_candidate"
    assert payload["top_candidate"]["model_name"] == "Winner.gguf"
    assert payload["verdict"]["action"] == "RETEST"
    assert payload["verdict"]["prediction"]["recommendation"] == "needs_agent_benchmark"
    assert payload["report_audit"]["status"] == "warning"
    assert payload["report_audit"]["warnings"][0]["code"] == "missing_agent_quality"
    assert payload["decision_packet"]["candidate_assessment"]["readiness"] == "not_recommendable"
    assert payload["decision_packet"]["operator_verdict"]["status"] == "NOT_USABLE_YET"
    assert payload["decision_packet"]["score_evidence"]["scored_candidate_count"] == 0
    assert payload["decision_packet"]["performance_prediction"]["status"] == "LAB_ONLY_SPEED_PROOF"
    assert (
        payload["decision_packet"]["performance_prediction"]["deployment_expectation"]
        == "do_not_deploy"
    )
    assert payload["decision_packet"]["candidate_rankings"][0]["model"] == "Winner.gguf"
    assert payload["decision_packet"]["settings_candidates"] == []
    assert payload["decision_packet"]["repeatability"]["confidence"] == "single_run"
    assert payload["decision_packet"]["repeatability"]["run_count"] == 1
    assert payload["decision_packet"]["context_gate"]["action"] == "WAITING_FOR_DEPLOYMENT"
    assert payload["decision_packet"]["resource_gate"]["action"] == "WAITING_FOR_DEPLOYMENT"
    assert payload["decision_packet"]["stability_gate"]["action"] == "WAITING_FOR_PROMOTED_STACK"
    assert payload["decision_packet"]["proven_components"] == []
    assert payload["decision_packet"]["proof_runbook"][0]["id"] == "model_plan"
    assert payload["decision_packet"]["proof_runbook"][0]["proves"] == "benchmark-suite.plan.json"
    assert payload["decision_packet"]["proof_commands"][0]["id"] == "model_plan"
    assert payload["artifacts"]["hard_recommendations"].endswith("hard-recommendations.md")
    assert (tmp_path / "runs" / "hard-recommendations.md").exists()


def test_results_json_out_preserves_required_context_in_decision_packet(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "model_name": "Winner.gguf",
                "lane_type": "chat_agent",
                "profiles": [
                    {"id": "standard", "label": "Standard", "context_size": 131072},
                    {"id": "long_agent", "label": "Long agent", "context_size": 200000},
                ],
            }
        ),
        encoding="utf-8",
    )
    run = runs_root / "standard-proof"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 131072, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.82,
                    "benchmark_suite_ok": True,
                    "benchmark_suite_general_score": 0.82,
                    "benchmark_suite_agentic_score": 0.82,
                    "serving_ttft_ms": 420.0,
                    "serving_tokens_per_second": 38.0,
                },
                "score": 0.82,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "results",
            "--runs-root",
            str(runs_root),
            "--required-context",
            "200000",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    decision = payload["decision_packet"]
    assert decision["context_gate"]["action"] == "RETEST_CONTEXT"
    assert decision["context_gate"]["required_context"] == 200000
    assert decision["context_gate"]["profile_id"] == "long_agent"
    candidates = {item["profile_id"]: item for item in decision["settings_candidates"]}
    assert candidates["standard"]["decision"] == "baseline_below_required_context"
    assert candidates["long_agent"]["decision"] == "next_to_test"


def test_results_json_out_uses_neutral_model_key_for_promoted_result(tmp_path):
    run = tmp_path / "runs" / "20260526-promoted"
    run.mkdir(parents=True)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 131072, "parallel": 1, "gpu_layers": 99},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.82,
                    "benchmark_suite_ok": True,
                    "benchmark_suite_general_score": 0.8,
                    "benchmark_suite_agentic_score": 0.84,
                    "serving_ttft_ms": 420.0,
                    "serving_tokens_per_second": 38.0,
                },
                "score": 0.82,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["results", "--runs-root", str(tmp_path / "runs"), "--json-out"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result_label"] == "recommended_model"
    assert payload["model"]["model_name"] == "Winner.gguf"
    assert payload["top_candidate"]["model_name"] == "Winner.gguf"
    assert payload["verdict"]["action"] == "PROMOTE"


def test_export_profile_refuses_unproven_deployment_profile(tmp_path):
    run = tmp_path / "runs" / "20260526-speed-only"
    run.mkdir(parents=True)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 262144, "parallel": 1, "gpu_layers": 99},
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

    result = runner.invoke(
        app,
        [
            "export-profile",
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "champions"),
        ],
    )

    assert result.exit_code == 1
    assert "Deployment readiness:" in result.output
    assert "Refusing to export unproven champion" in result.output
    assert not (tmp_path / "champions" / "champion_hermes_pilot.ps1").exists()


def test_export_profile_allow_unproven_keeps_manual_escape_hatch(tmp_path):
    run = tmp_path / "runs" / "20260526-speed-only"
    run.mkdir(parents=True)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "settings": {"context_size": 262144, "parallel": 1, "gpu_layers": 99},
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

    result = runner.invoke(
        app,
        [
            "export-profile",
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "champions"),
            "--allow-unproven",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "WARNING: exporting an unproven champion" in result.output
    assert (tmp_path / "champions" / "champion_hermes_pilot.ps1").exists()


def test_results_command_prints_all_audit_warnings(tmp_path):
    for run_id in ("20260526-speed-a", "20260526-speed-b"):
        run = tmp_path / "runs" / run_id
        run.mkdir(parents=True)
        (run / "best-settings.json").write_text(
            json.dumps(
                {
                    "model": f"G:/AI/models/{run_id}.gguf",
                    "settings": {"context_size": 262144, "parallel": 1, "gpu_layers": 99},
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
    assert "Report audit: warning (2 warning(s))" in result.output
    assert "Audit warning: missing_agent_quality in 20260526-speed-a" in result.output
    assert "Audit warning: missing_agent_quality in 20260526-speed-b" in result.output
    assert re.search(r"report-audit\s*\.md", result.output)


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


def test_export_plan_command_prints_resolved_plan(tmp_path):
    run = tmp_path / "runs" / "20260629-qwen"
    run.mkdir(parents=True)
    (run / "resolved-plan.json").write_text(
        json.dumps({"schema_version": 1, "program": "autoresearch", "model": "qwen.gguf"}),
        encoding="utf-8",
    )
    (run / "command.txt").write_text(
        "agent-autobench autoresearch --model qwen.gguf\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["export-plan", "--run", str(run), "--json-out"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["program"] == "autoresearch"
    assert payload["model"] == "qwen.gguf"


def test_export_plan_command_copies_resolved_plan(tmp_path):
    run = tmp_path / "runs" / "20260629-qwen"
    run.mkdir(parents=True)
    (run / "resolved-plan.json").write_text(
        json.dumps({"schema_version": 1, "program": "speed"}),
        encoding="utf-8",
    )
    output = tmp_path / "exports" / "plan.json"

    result = runner.invoke(app, ["export-plan", "--run", str(run), "--output", str(output)])

    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["program"] == "speed"


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

    result = runner.invoke(
        app,
        [
            "serve-probe",
            "--model",
            str(model),
            "--context-size",
            "8192",
            "--runs-root",
            str(tmp_path / "_runs"),
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["model"] == model
    assert calls[0]["settings"].context_size == 16_384
    assert calls[0]["settings"].kv_unified is True
    assert calls[0]["settings"].cache_type_k == "q8_0"
    assert calls[0]["settings"].cache_type_v == "q8_0"
    assert "--jinja" in calls[0]["settings"].extra_server_args
    assert "Context bumped to 16k" in result.output
    assert calls[0]["samples"] == 0
    assert calls[0]["cache_prompt"] is True
    assert "Cold TTFT: 321 ms" in result.output
    assert "Warm TTFT: 111 ms" in result.output
    assert "Warmup penalty: 210 ms" in result.output
    assert "Serving speed: 27.50 tok/s" in result.output


def test_serve_probe_command_writes_speed_program_receipt(tmp_path, monkeypatch):
    class FakeServingProbeResult:
        ok = True
        ttft_ms = 250.0
        tokens_per_second = 88.5
        warm_ttft_ms = None
        warm_tokens_per_second = None
        warmup_penalty_ms = None
        server_ready_ms = 900.0
        cold_start_to_first_token_ms = 1150.0
        ttft_samples_ms = [250.0]
        tokens_cached_samples = [0]
        tokens_evaluated_samples = [128]
        generated_tokens = 420
        output_chars = 2100
        failure = "none"
        stderr_tail = ""

        def to_dict(self):
            return {
                "ok": self.ok,
                "ttft_ms": self.ttft_ms,
                "tokens_per_second": self.tokens_per_second,
                "generated_tokens": self.generated_tokens,
                "output_chars": self.output_chars,
                "failure": self.failure,
            }

    monkeypatch.setattr(
        "gguf_limit_bench.cli.probe_llama_server_ttft",
        lambda **_kwargs: FakeServingProbeResult(),
    )
    model = tmp_path / "Qwen3-Test-Q8_0.gguf"
    model.write_bytes(b"fake")

    result = runner.invoke(
        app,
        [
            "serve-probe",
            "--model",
            str(model),
            "--runs-root",
            str(tmp_path / "runs"),
            "--context-size",
            "16384",
            "--max-tokens",
            "512",
        ],
    )

    assert result.exit_code == 0
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    payload = json.loads((run_dirs[0] / "speed-probe.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dirs[0] / "resolved-plan.json").read_text(encoding="utf-8"))
    status = json.loads((run_dirs[0] / "status.json").read_text(encoding="utf-8"))
    assert payload["program"] == "speed"
    assert payload["model"] == str(model)
    assert payload["settings"]["context_size"] == 16_384
    assert payload["settings"]["cache_type_k"] == "q8_0"
    assert payload["result"]["tokens_per_second"] == 88.5
    assert "500 word poem" in payload["prompt"]
    assert resolved["program"] == "speed"
    assert resolved["requested_context_size"] == 16384
    assert resolved["commands"][0]["argv"][:3] == [
        "agent-autobench",
        "serve-probe",
        "--model",
    ]
    assert "agent-autobench serve-probe" in (run_dirs[0] / "command.txt").read_text(
        encoding="utf-8"
    )
    assert status["status"] == "finished"
    assert "Receipt:" in result.output


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
    assert "Verdict: PROMOTE" in result.output
    assert "agent_bench_score: 0.700000" in result.output
    assert (tmp_path / "runs" / "benchmark-suite.tsv").exists()
    assert (tmp_path / "runs" / "agentic-suite.tsv").exists()


def test_benchmark_suite_template_defaults_to_local_librarian_plan(tmp_path):
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
    assert payload["context"] == 131072
    assert payload["settings"]["context_target"] == "required_context_131072"
    assert payload["settings"]["score_contract"] == "agent_bench_score"
    assert payload["settings"]["plan_kind"] == "local_librarian_template"
    assert payload["settings"]["base_url"] == "http://127.0.0.1:8080"
    assert "--jinja" in payload["settings"]["extra_server_args"]
    assert payload["settings"]["answer_max_tokens"] == 8192
    assert payload["tasks"][0]["id"] == "local_librarian_general"
    assert payload["tasks"][0]["harness"] == "librarian-suite"
    assert payload["tasks"][0]["commands"][0][9:11] == ["--base-url", "{base_url}"]
    assert payload["tasks"][0]["commands"][0][15:17] == ["--settings-json", "{settings_json}"]
    assert payload["tasks"][0]["commands"][0][:5] == [
        "uv",
        "run",
        "--extra",
        "dev",
        "python",
    ]
    assert "gguf_limit_bench.librarian_suite" in payload["tasks"][0]["commands"][0]
    assert "librarian-query" in payload["tasks"][1]["commands"][0]


def test_benchmark_suite_template_can_write_external_harness_plan(tmp_path):
    output = tmp_path / "benchmark-suite.plan.json"

    result = runner.invoke(
        app,
        [
            "benchmark-suite-template",
            "--template-kind",
            "external",
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
    assert payload["tasks"][0]["commands"][0][:5] == [
        "uvx",
        "--from",
        "lm-eval",
        "lm-eval",
        "run",
    ]
    assert "base_url={base_url}/chat/completions" in payload["tasks"][0]["commands"][0][8]
    assert payload["tasks"][1]["commands"][0][4:7] == [
        "inspect",
        "eval",
        "benchmarks/inspect_tasks/json_repair.py",
    ]
    assert "{base_url}" in payload["tasks"][1]["commands"][0]


def test_benchmark_suite_plans_lists_bundled_plans():
    result = runner.invoke(app, ["benchmark-suite-plans", "--json-out"])

    assert result.exit_code == 0
    plans = json.loads(result.output)
    normalized_plans = [plan["path"].replace("\\", "/") for plan in plans]
    assert all("context" in plan for plan in plans)
    assert any(
        path.endswith("benchmarks/plans/local-openai-smoke.plan.json") for path in normalized_plans
    )
    assert any(
        path.endswith("benchmarks/plans/local-bfcl-smoke.plan.json") for path in normalized_plans
    )
    gemma = next(
        plan
        for plan in plans
        if plan["filename"] == "wiki-librarian-gemma4-26b-a4b-thinking.plan.json"
    )
    assert gemma["context"] == 131072
    assert gemma["context_target"] == "required_context_131072"
    assert gemma["score_contract"] == "agent_bench_score"


def test_flight_plans_command_lists_beginner_contract():
    result = runner.invoke(app, ["flight-plans", "--json-out"])

    assert result.exit_code == 0
    plans = json.loads(result.output)
    librarian = next(plan for plan in plans if plan["id"] == "librarian_benchmark")
    assert librarian["recommended"] is True
    assert librarian["mode_id"] == "librarian_bench"
    assert librarian["evidence_class"] == "recommendation"
    assert librarian["score_contract"] == "agent_bench_score"
    assert librarian["budget_minutes"] == 30
    assert librarian["suggested_benchmark_suite_plans"][0]["filename"].endswith(".plan.json")


def test_flight_plans_command_human_output_marks_recommended_plan():
    result = runner.invoke(app, ["flight-plans"])

    assert result.exit_code == 0
    assert "pilotBENCHY Flight Plans" in result.output
    assert "Librarian benchmark" in result.output
    assert "Recommended" in result.output
    assert "Score" in result.output
    assert "agent_bench_score" in result.output
    assert "speed_only" in result.output
    assert "yes" in result.output


def test_run_one_autoresearch_benchmark_mode_uses_simplebench_runner(monkeypatch, tmp_path):
    import gguf_limit_bench.cli as cli
    from gguf_limit_bench.evaluation_mode import EvaluationMode

    captured = {}

    class FakeLoop:
        def __init__(self, **kwargs):
            captured["candidate_sequence"] = kwargs.get("candidate_sequence")
            captured["resolved_plan"] = kwargs.get("resolved_plan")

        def run(self):
            class R:
                path = tmp_path

            return R()

    monkeypatch.setattr(cli, "AutoresearchLoop", FakeLoop)
    cli._run_one_autoresearch(
        model=tmp_path / "m.gguf",
        llama_bench=tmp_path / "llama-bench",
        llama_cli=tmp_path / "llama-cli",
        llama_server=tmp_path / "llama-server",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=1,
        max_attempts=1,
        learning=True,
        workflow_eval=False,
        ttft_probe=False,
        evaluation=EvaluationMode.BENCHMARK,
        run_mode_id="librarian_bench",
        flight_plan_id="librarian_benchmark",
    )
    # Benchmark mode must route through the flag-ladder question engine, which
    # supplies a candidate_sequence (the synthetic speed scout supplies none).
    assert captured["candidate_sequence"] is not None
    assert captured["resolved_plan"]["mode_id"] == "librarian_bench"
    assert captured["resolved_plan"]["flight_plan_id"] == "librarian_benchmark"
    assert "--temp" in captured["candidate_sequence"][0].extra_server_args
    assert "--top-p" in captured["candidate_sequence"][0].extra_server_args


def test_run_one_autoresearch_can_opt_out_of_hf_sampler_defaults(monkeypatch, tmp_path):
    import gguf_limit_bench.cli as cli
    from gguf_limit_bench.evaluation_mode import EvaluationMode

    captured = {}

    class FakeLoop:
        def __init__(self, **kwargs):
            captured["candidate_sequence"] = kwargs.get("candidate_sequence")

        def run(self):
            class R:
                path = tmp_path

            return R()

    monkeypatch.setattr(cli, "AutoresearchLoop", FakeLoop)
    cli._run_one_autoresearch(
        model=Path("G:/AI/models/Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"),
        llama_bench=tmp_path / "llama-bench",
        llama_cli=tmp_path / "llama-cli",
        llama_server=tmp_path / "llama-server",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=1,
        max_attempts=1,
        learning=False,
        workflow_eval=False,
        ttft_probe=False,
        evaluation=EvaluationMode.BENCHMARK,
        sampler_policy="runtime_defaults",
    )

    assert "--temp" not in captured["candidate_sequence"][0].extra_server_args


def test_run_one_autoresearch_speed_scout_uses_no_candidate_sequence(monkeypatch, tmp_path):
    import gguf_limit_bench.cli as cli
    from gguf_limit_bench.evaluation_mode import EvaluationMode

    captured = {}

    class FakeLoop:
        def __init__(self, **kwargs):
            captured["candidate_sequence"] = kwargs.get("candidate_sequence")

        def run(self):
            class R:
                path = tmp_path

            return R()

    monkeypatch.setattr(cli, "AutoresearchLoop", FakeLoop)
    cli._run_one_autoresearch(
        model=tmp_path / "m.gguf",
        llama_bench=tmp_path / "llama-bench",
        llama_cli=tmp_path / "llama-cli",
        llama_server=tmp_path / "llama-server",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=1,
        max_attempts=1,
        learning=False,
        workflow_eval=False,
        ttft_probe=False,
        evaluation=EvaluationMode.SPEED_SCOUT,
    )
    assert captured["candidate_sequence"] is None


def test_forced_server_args_apply_to_every_flag_ladder_profile(tmp_path):
    import gguf_limit_bench.cli as cli
    from gguf_limit_bench.evaluation_mode import EvaluationMode

    model = tmp_path / "model with spaces % & ^.gguf"
    model.write_bytes(b"fake")
    receipt = cli._run_one_autoresearch(
        model=model,
        llama_bench=tmp_path / "llama-bench.exe",
        llama_cli=tmp_path / "llama-cli.exe",
        llama_server=tmp_path / "llama-server.exe",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=2,
        max_attempts=None,
        learning=False,
        workflow_eval=False,
        ttft_probe=False,
        evaluation=EvaluationMode.BENCHMARK,
        flag_ladder=True,
        dry_run=True,
        forced_server_args=("--no-mmap",),
    )
    plan = json.loads((receipt.path / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    resolved = json.loads((receipt.path / "resolved-plan.json").read_text(encoding="utf-8"))
    status = json.loads((receipt.path / "status.json").read_text(encoding="utf-8"))
    commanded = [p for p in plan["profiles"] if p.get("command")]
    assert commanded, "dry-run plan should contain commands"
    for profile in commanded:
        assert "--no-mmap" in profile["command"]
    assert resolved["dry_run"] is True
    assert resolved["flag_ladder"] is True
    assert resolved["commands"][0]["argv"][3] == str(model)
    assert "agent-autobench autoresearch" in (receipt.path / "command.txt").read_text(
        encoding="utf-8"
    )
    assert status["status"] == "finished"


def test_effective_forced_args_keep_standard_flags_and_template_choice(monkeypatch):
    import gguf_limit_bench.cli as cli

    monkeypatch.setattr(cli, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 4090")

    args = cli._effective_forced_server_args(
        ("--jinja", "--chat-template-file", "G:/templates/qwen.jinja")
    )

    assert "--flash-attn" in args
    assert "--kv-unified" in args
    assert "--cache-type-k" in args
    assert "--cache-type-v" in args
    assert "--jinja" in args
    assert "--chat-template-file" in args
    assert "G:/templates/qwen.jinja" in args


def test_standard_forced_args_do_not_duplicate_managed_flags_in_plan(tmp_path, monkeypatch):
    import gguf_limit_bench.cli as cli
    from gguf_limit_bench.evaluation_mode import EvaluationMode

    monkeypatch.setattr(cli, "detect_gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"fake")

    receipt = cli._run_one_autoresearch(
        model=model,
        llama_bench=tmp_path / "llama-bench.exe",
        llama_cli=tmp_path / "llama-cli.exe",
        llama_server=tmp_path / "llama-server.exe",
        runs_root=tmp_path,
        budget_seconds=60,
        parallel_max=2,
        max_attempts=None,
        learning=False,
        workflow_eval=False,
        ttft_probe=False,
        evaluation=EvaluationMode.BENCHMARK,
        flag_ladder=True,
        dry_run=True,
        forced_server_args=cli._effective_forced_server_args(),
    )

    plan = json.loads((receipt.path / "flag-ladder-plan.json").read_text(encoding="utf-8"))
    command = next(p["command"] for p in plan["profiles"] if p["name"] == "L6-q8-kv")
    assert command.count("--flash-attn") == 1
    assert command.count("--gpu-layers") == 1
    assert command.count("--kv-unified") == 1
    assert command.count("--cache-type-k") == 1
    assert command.count("--cache-type-v") == 1
