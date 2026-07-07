from pathlib import Path

from gguf_limit_bench.benchmark_suite import BenchmarkSuitePlan


def test_bundled_benchmark_suite_plans_are_valid_and_complete():
    plan_paths = sorted(Path("benchmarks/plans").glob("*.plan.json"))

    assert {path.name for path in plan_paths} >= {
        "local-openai-smoke.plan.json",
        "local-bfcl-smoke.plan.json",
        "external-agentic-heavy.plan.json",
        "wiki-librarian-gemma4-26b-a4b-thinking.plan.json",
        "wiki-librarian-qwen3-moe-thinking.plan.json",
    }
    for path in plan_paths:
        plan = BenchmarkSuitePlan.from_path(path)
        phases = {task.phase for task in plan.tasks}
        text = path.read_text(encoding="utf-8")

        assert phases == {"general", "agentic"}
        assert "path/to" not in text
        assert "{task_dir}" in text
        assert plan.settings["score_contract"] == "agent_bench_score"
