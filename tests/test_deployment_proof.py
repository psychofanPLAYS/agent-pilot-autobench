import json
from pathlib import Path
import sys

from gguf_limit_bench.autoresearch import AttemptResult
from gguf_limit_bench.deployment_proof import BenchmarkSuitePreflightError, run_deployment_proof
from gguf_limit_bench.deployment_readiness import write_deployment_readiness


def _write_flag_recommendations(root):
    (root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Winner.gguf",
                "model_name": "Winner.gguf",
                "profiles": [
                    {
                        "id": "standard",
                        "label": "Standard",
                        "context_size": 131072,
                        "settings": {
                            "profile_name": "standard",
                            "context_size": 131072,
                            "parallel": 1,
                            "gpu_layers": 99,
                            "batch_size": 2048,
                            "ubatch_size": 512,
                            "flash_attention": True,
                            "kv_unified": True,
                            "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0",
                            "extra_server_args": ["--jinja"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_benchmark_suite_plan(path):
    score_command = [
        sys.executable,
        "-c",
        "import json; print(json.dumps({'score': 0.82}))",
    ]
    path.write_text(
        json.dumps(
            {
                "model": "local-model",
                "context": 131072,
                "tasks": [
                    {
                        "id": "general",
                        "phase": "general",
                        "harness": "fake-general",
                        "command": score_command,
                    },
                    {
                        "id": "agentic",
                        "phase": "agentic",
                        "harness": "fake-agentic",
                        "command": score_command,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_runtime_settings_benchmark_suite_plan(path):
    score_command = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "settings=json.loads(sys.argv[1]); "
            "assert settings['gguf_model_path'].endswith('Winner.gguf'); "
            "assert '--jinja' in settings['extra_server_args']; "
            "assert settings['context_size'] == 131072; "
            "print(json.dumps({'score': 0.66}))"
        ),
        "{settings_json}",
    ]
    path.write_text(
        json.dumps(
            {
                "model": "template-label",
                "context": 131072,
                "tasks": [
                    {
                        "id": "general",
                        "phase": "general",
                        "harness": "fake-general",
                        "command": score_command,
                    },
                    {
                        "id": "agentic",
                        "phase": "agentic",
                        "harness": "fake-agentic",
                        "command": score_command,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_deployment_proof_runs_exact_flag_profile_and_proves_readiness(tmp_path):
    _write_flag_recommendations(tmp_path)
    plan_path = tmp_path / "benchmark-suite.plan.json"
    _write_benchmark_suite_plan(plan_path)
    seen_settings = []

    def fake_attempt_runner(settings):
        seen_settings.append(settings)
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=41.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=430.0,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
            serving_ttft_ms=430.0,
            serving_tokens_per_second=37.0,
            flag_profile=settings.profile_name,
        )

    receipt = run_deployment_proof(
        runs_root=tmp_path,
        profile_id="standard",
        benchmark_suite_plan=plan_path,
        attempt_runner=fake_attempt_runner,
    )

    assert seen_settings[0].profile_name == "standard"
    assert seen_settings[0].context_size == 131072
    assert seen_settings[0].cache_type_k == "q8_0"
    assert seen_settings[0].cache_type_v == "q8_0"
    assert seen_settings[0].extra_server_args == ("--jinja",)
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))
    resolved_plan = json.loads((receipt.path / "resolved-plan.json").read_text(encoding="utf-8"))
    assert Path(best["model"]).name == "Winner.gguf"
    assert best["settings"]["profile_name"] == "standard"
    assert best["result"]["benchmark_suite_ok"] is True
    assert best["result"]["agent_bench_score"] == 0.82
    assert best["result"]["serving_ttft_ms"] == 430.0
    assert best["promotion_eligible"] is True
    assert resolved_plan["program"] == "deployment-proof"
    assert resolved_plan["simple_bench_max_tokens"] == 8192
    assert resolved_plan["selected_profile"]["id"] == "standard"
    assert resolved_plan["selected_profile"]["settings"]["cache_type_k"] == "q8_0"

    readiness = write_deployment_readiness(tmp_path)
    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "PROMOTE_DEPLOYMENT_PROFILE"
    assert payload["recommended_profile_id"] == "standard"


def test_deployment_proof_benchmark_suite_uses_runtime_model_and_settings(tmp_path):
    _write_flag_recommendations(tmp_path)
    plan_path = tmp_path / "benchmark-suite.plan.json"
    _write_runtime_settings_benchmark_suite_plan(plan_path)

    def fake_attempt_runner(settings):
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=41.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=430.0,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
            serving_ttft_ms=430.0,
            serving_tokens_per_second=37.0,
            flag_profile=settings.profile_name,
        )

    receipt = run_deployment_proof(
        runs_root=tmp_path,
        profile_id="standard",
        benchmark_suite_plan=plan_path,
        attempt_runner=fake_attempt_runner,
    )

    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))
    suite_summary = json.loads(
        Path(best["result"]["benchmark_suite_receipt"], "suite-summary.json").read_text(
            encoding="utf-8"
        )
    )
    suite_plan = json.loads(
        Path(best["result"]["benchmark_suite_receipt"], "suite-plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert suite_summary["model"].replace("\\", "/") == "G:/AI/models/Winner.gguf"
    assert suite_plan["model"].replace("\\", "/") == "G:/AI/models/Winner.gguf"
    assert best["result"]["benchmark_suite_ok"] is True
    assert best["result"]["agent_bench_score"] == 0.66


def test_deployment_proof_owned_server_keeps_benchmark_suite_inside_runner(tmp_path, monkeypatch):
    _write_flag_recommendations(tmp_path)
    plan_path = tmp_path / "benchmark-suite.plan.json"
    _write_benchmark_suite_plan(plan_path)
    runner_kwargs = {}
    loop_kwargs = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            runner_kwargs.update(kwargs)

    class FakeLoop:
        def __init__(self, **kwargs):
            loop_kwargs.update(kwargs)

        def run(self):
            return object()

    monkeypatch.setattr(
        "gguf_limit_bench.deployment_proof.LlamaServerSimpleBenchAttemptRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        "gguf_limit_bench.deployment_proof.AutoresearchLoop",
        FakeLoop,
    )

    run_deployment_proof(
        runs_root=tmp_path,
        profile_id="standard",
        benchmark_suite_plan=plan_path,
        llama_server=tmp_path / "llama-server.exe",
    )

    assert runner_kwargs["benchmark_suite_plan"].model == "local-model"
    assert runner_kwargs["runs_root"] == tmp_path
    assert loop_kwargs["benchmark_suite_plan"] is None


def test_deployment_proof_fails_preflight_before_launching_runner(tmp_path):
    _write_flag_recommendations(tmp_path)
    plan_path = tmp_path / "benchmark-suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "local-model",
                "context": 131072,
                "tasks": [
                    {
                        "id": "missing",
                        "phase": "general",
                        "harness": "lm-evaluation-harness",
                        "command": ["definitely-not-a-real-benchmark-command"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        run_deployment_proof(
            runs_root=tmp_path,
            profile_id="standard",
            benchmark_suite_plan=plan_path,
            llama_server=tmp_path / "llama-server.exe",
        )
    except BenchmarkSuitePreflightError as exc:
        assert "benchmark-suite preflight failed" in str(exc)
        assert exc.receipt_path.endswith("benchmark-suite-preflight.json")
    else:
        raise AssertionError("expected preflight failure")

    preflight = json.loads((tmp_path / "benchmark-suite-preflight.json").read_text())
    assert preflight["status"] == "HARNESS_MISSING"
    assert preflight["issues"][0]["task_id"] == "missing"
