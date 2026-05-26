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
