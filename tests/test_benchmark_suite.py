import json
from pathlib import Path
import sys

from gguf_limit_bench.benchmark_suite import (
    BenchmarkSuitePlan,
    preflight_benchmark_suite,
    run_benchmark_suite,
)


def _score_command(score: float) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import json; print(json.dumps({{'score': {score}}}))",
    ]


def _weak_librarian_score_command() -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import json, pathlib, sys; "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
            "'librarian_bench_score': 1.0, 'agent_bench_score': None, "
            "'score': 1.0, 'recommendation_grade': False, "
            "'agent_quality_gate': 'weak_sample'}))"
        ),
        "{task_dir}/score.json",
    ]


def test_benchmark_suite_writes_general_agentic_and_agent_score_ledgers(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 32768,
                "settings": {"parallel": 4, "kv_unified": True},
                "tasks": [
                    {
                        "id": "arc_easy_smoke",
                        "phase": "general",
                        "harness": "lm-evaluation-harness",
                        "command": _score_command(0.70),
                    },
                    {
                        "id": "inspect_tool_smoke",
                        "phase": "agentic",
                        "harness": "inspect-ai",
                        "command": _score_command(0.90),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    assert suite_run.ok is True
    assert suite_run.agent_bench_score == 0.80
    verdict = json.loads(
        (tmp_path / "runs" / suite_run.run_id / "suite-verdict.json").read_text(encoding="utf-8")
    )
    verdict_md = (tmp_path / "runs" / suite_run.run_id / "suite-verdict.md").read_text(
        encoding="utf-8"
    )
    assert verdict["action"] == "PROMOTE"
    assert verdict["agent_bench_score"] == 0.80
    assert "Action: `PROMOTE`" in verdict_md
    assert (tmp_path / "runs" / "benchmark-suite.tsv").exists()
    assert (tmp_path / "runs" / "agentic-suite.tsv").exists()
    assert (tmp_path / "runs" / "agent-bench-score.tsv").exists()
    assert "lm-evaluation-harness" in (tmp_path / "runs" / "benchmark-suite.tsv").read_text(
        encoding="utf-8"
    )
    assert "inspect-ai" in (tmp_path / "runs" / "agentic-suite.tsv").read_text(encoding="utf-8")
    event_types = [
        json.loads(line)["type"]
        for line in (tmp_path / "runs" / suite_run.run_id / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert event_types == [
        "benchmark_suite_started",
        "benchmark_suite_task_started",
        "benchmark_suite_command_started",
        "benchmark_suite_command_finished",
        "benchmark_suite_task_finished",
        "benchmark_suite_task_started",
        "benchmark_suite_command_started",
        "benchmark_suite_command_finished",
        "benchmark_suite_task_finished",
        "benchmark_suite_finished",
    ]


def test_benchmark_suite_rejects_weak_librarian_score_file(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "gemma-local",
                "context": 131072,
                "tasks": [
                    {
                        "id": "weak_librarian_general",
                        "phase": "general",
                        "harness": "librarian-suite",
                        "command": _weak_librarian_score_command(),
                        "score_file": "{task_dir}/score.json",
                    },
                    {
                        "id": "weak_librarian_agentic",
                        "phase": "agentic",
                        "harness": "librarian-suite",
                        "command": _weak_librarian_score_command(),
                        "score_file": "{task_dir}/score.json",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    assert suite_run.ok is False
    assert suite_run.agent_bench_score == 0.0
    assert suite_run.results[0].score is None
    assert suite_run.results[0].failure_class == "no_score"


def test_benchmark_suite_reads_librarian_agent_score_from_task_dir_score_file(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "gemma-local",
                "context": 131072,
                "tasks": [
                    {
                        "id": "recommendation_grade_librarian",
                        "phase": "general",
                        "harness": "librarian-suite",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, pathlib, sys; "
                                "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
                                "'librarian_bench_score': 0.64, "
                                "'agent_bench_score': 0.64, "
                                "'recommendation_grade': True}))"
                            ),
                            "{task_dir}/score.json",
                        ],
                        "score_file": "{task_dir}/score.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=Path("runs"),
    )

    assert suite_run.results[0].score == 0.64
    assert suite_run.results[0].failure_class == "none"


def test_benchmark_suite_records_harness_missing_as_failed_evidence(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 4096,
                "tasks": [
                    {
                        "id": "missing_harness",
                        "phase": "general",
                        "harness": "lm-evaluation-harness",
                        "command": ["definitely-not-a-real-benchmark-command"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    assert suite_run.ok is False
    assert suite_run.results[0].failure_class == "harness_missing"
    assert "harness_missing" in (tmp_path / "runs" / "benchmark-suite.tsv").read_text(
        encoding="utf-8"
    )


def test_benchmark_suite_preflight_catches_missing_harness_before_run(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 4096,
                "tasks": [
                    {
                        "id": "missing_harness",
                        "phase": "general",
                        "harness": "lm-evaluation-harness",
                        "command": ["definitely-not-a-real-benchmark-command"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    preflight = preflight_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
        plan_path=plan_path,
    )

    payload = json.loads((tmp_path / "runs" / "benchmark-suite-preflight.json").read_text())
    assert preflight.ok is False
    assert preflight.status == "HARNESS_MISSING"
    assert preflight.issues[0].failure_class == "harness_missing"
    assert payload["issues"][0]["executable"] == "definitely-not-a-real-benchmark-command"
    assert payload["plan_path"] == str(plan_path)
    assert "HARNESS_MISSING" in (tmp_path / "runs" / "benchmark-suite-preflight.md").read_text(
        encoding="utf-8"
    )


def test_benchmark_suite_preflight_rejects_blank_required_base_url(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "gemma-local",
                "context": 131072,
                "settings": {},
                "tasks": [
                    {
                        "id": "needs_endpoint",
                        "phase": "general",
                        "harness": "fake",
                        "command": [
                            sys.executable,
                            "-c",
                            "print('ok')",
                            "--base-url",
                            "{base_url}",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    preflight = preflight_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
        plan_path=plan_path,
    )

    assert preflight.ok is False
    assert preflight.status == "INVALID_PLAN"
    assert preflight.issues[0].failure_class == "invalid_plan"
    assert "base_url" in preflight.issues[0].detail


def test_benchmark_suite_preflight_rejects_missing_uv_fallback_module(tmp_path, monkeypatch):
    import gguf_limit_bench.benchmark_suite as benchmark_suite

    monkeypatch.setattr(benchmark_suite.shutil, "which", lambda command: None)
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "gemma-local",
                "context": 131072,
                "tasks": [
                    {
                        "id": "missing_module",
                        "phase": "general",
                        "harness": "repo-local",
                        "command": [
                            "uv",
                            "run",
                            "--extra",
                            "dev",
                            "python",
                            "-m",
                            "definitely_missing_benchmark_module",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    preflight = preflight_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
        plan_path=plan_path,
    )

    assert preflight.ok is False
    assert preflight.status == "HARNESS_MISSING"
    assert preflight.issues[0].failure_class == "harness_missing"
    assert "definitely_missing_benchmark_module" in preflight.issues[0].detail


def test_benchmark_suite_preflight_rejects_librarian_without_runtime_jinja_settings(
    tmp_path, monkeypatch
):
    import gguf_limit_bench.benchmark_suite as benchmark_suite

    monkeypatch.setattr(benchmark_suite.shutil, "which", lambda command: None)
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "G:/AI/models/Gemma.gguf",
                "context": 200000,
                "settings": {"base_url": "http://127.0.0.1:8080"},
                "tasks": [
                    {
                        "id": "local_librarian_general",
                        "phase": "general",
                        "harness": "librarian-suite",
                        "command": [
                            "uv",
                            "run",
                            "--extra",
                            "dev",
                            "python",
                            "-m",
                            "gguf_limit_bench.librarian_suite",
                            "--model",
                            "{model}",
                            "--base-url",
                            "{base_url}",
                            "--settings-json",
                            '{"plan_kind":"local_librarian_template"}',
                            "--pack",
                            "librarian-gate",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    preflight = preflight_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
        plan_path=plan_path,
    )

    assert preflight.ok is False
    assert preflight.status == "INVALID_PLAN"
    assert preflight.issues[0].failure_class == "invalid_plan"
    assert "--jinja" in preflight.issues[0].detail


def test_benchmark_suite_falls_back_to_venv_python_for_uv_run_python_when_uv_missing(
    tmp_path, monkeypatch
):
    import gguf_limit_bench.benchmark_suite as benchmark_suite

    monkeypatch.setattr(benchmark_suite.shutil, "which", lambda command: None)
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "gemma-local",
                "context": 131072,
                "tasks": [
                    {
                        "id": "local_librarian_without_uv",
                        "phase": "general",
                        "harness": "librarian-suite",
                        "command": [
                            "uv",
                            "run",
                            "--extra",
                            "dev",
                            "python",
                            "-c",
                            "import json; print(json.dumps({'agent_bench_score': 0.42}))",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    command_receipt = json.loads(
        (
            tmp_path / "runs" / suite_run.run_id / "local_librarian_without_uv" / "command.json"
        ).read_text(encoding="utf-8")
    )
    recorded_command = command_receipt["commands"][0]
    assert suite_run.results[0].ok is True
    assert suite_run.results[0].score == 0.42
    assert recorded_command[:2] == [sys.executable, "-c"]


def test_benchmark_suite_expands_runtime_base_url_from_settings(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "gemma-local",
                "context": 131072,
                "settings": {"base_url": "http://127.0.0.1:64951"},
                "tasks": [
                    {
                        "id": "base_url_task",
                        "phase": "general",
                        "harness": "fake",
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'score': 0.5}))",
                            "--base-url",
                            "{base_url}",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    command_receipt = json.loads(
        (tmp_path / "runs" / suite_run.run_id / "base_url_task" / "command.json").read_text(
            encoding="utf-8"
        )
    )
    assert command_receipt["commands"][0][-1] == "http://127.0.0.1:64951"


def test_benchmark_suite_task_can_run_generate_then_evaluate_commands(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    score_file = "{task_dir}/score.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 32768,
                "tasks": [
                    {
                        "id": "bfcl_two_step",
                        "phase": "agentic",
                        "harness": "bfcl",
                        "commands": [
                            [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; Path(r'{task_dir}/responses.json').write_text('{}')",
                            ],
                            [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; Path(r'{task_dir}/score.json').write_text('{\"score\": 0.77}')",
                            ],
                        ],
                        "score_file": score_file,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    assert suite_run.results[0].ok is True
    assert suite_run.results[0].score == 0.77
    assert (tmp_path / "runs" / "agentic-suite.tsv").exists()


def test_benchmark_suite_task_expands_environment_without_leaking_secret(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    score_file = "{task_dir}/score.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 4096,
                "tasks": [
                    {
                        "id": "env_task",
                        "phase": "general",
                        "harness": "custom",
                        "env": {
                            "LOCAL_SERVER_ENDPOINT": "http://127.0.0.1",
                            "LOCAL_SERVER_PORT": "8080",
                            "SECRET_TOKEN": "do-not-write-this",
                        },
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "import json, os; "
                                "print(json.dumps({'score': 1.0, "
                                "'endpoint': os.environ['LOCAL_SERVER_ENDPOINT']}))"
                            ),
                        ],
                        "score_file": score_file,
                    },
                    {
                        "id": "agentic",
                        "phase": "agentic",
                        "harness": "custom",
                        "command": _score_command(1.0),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )
    command_receipt = json.loads(
        (tmp_path / "runs" / suite_run.run_id / "env_task" / "command-1.json").read_text(
            encoding="utf-8"
        )
    )

    assert suite_run.results[0].ok is True
    assert suite_run.results[0].score == 1.0
    assert command_receipt["env"]["LOCAL_SERVER_PORT"] == "8080"
    assert command_receipt["env"]["SECRET_TOKEN"] == "<redacted>"


def test_benchmark_suite_does_not_count_boolean_score_values(tmp_path):
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "qwen-local",
                "context": 4096,
                "tasks": [
                    {
                        "id": "boolean_score",
                        "phase": "general",
                        "harness": "custom",
                        "command": [
                            sys.executable,
                            "-c",
                            "import json; print(json.dumps({'score': True}))",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite_run = run_benchmark_suite(
        BenchmarkSuitePlan.from_path(plan_path),
        runs_root=tmp_path / "runs",
    )

    assert suite_run.results[0].ok is False
    assert suite_run.results[0].score is None
    assert suite_run.results[0].failure_class == "no_score"
