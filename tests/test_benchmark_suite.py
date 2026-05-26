import json
import sys

from gguf_limit_bench.benchmark_suite import BenchmarkSuitePlan, run_benchmark_suite


def _score_command(score: float) -> list[str]:
    return [
        sys.executable,
        "-c",
        f"import json; print(json.dumps({{'score': {score}}}))",
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
    assert (tmp_path / "runs" / "benchmark-suite.tsv").exists()
    assert (tmp_path / "runs" / "agentic-suite.tsv").exists()
    assert (tmp_path / "runs" / "agent-bench-score.tsv").exists()
    assert "lm-evaluation-harness" in (tmp_path / "runs" / "benchmark-suite.tsv").read_text(
        encoding="utf-8"
    )
    assert "inspect-ai" in (tmp_path / "runs" / "agentic-suite.tsv").read_text(encoding="utf-8")


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
        ((tmp_path / "runs" / suite_run.run_id / "env_task" / "command-1.json")).read_text(
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
