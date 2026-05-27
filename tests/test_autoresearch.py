import json
from pathlib import Path
import sys
import time

from gguf_limit_bench.autoresearch import (
    AttemptResult,
    AutoresearchLoop,
    AutoresearchSettings,
    PerplexityResult,
    build_autoresearch_llama_bench_command,
    build_llama_perplexity_command,
    parse_llama_bench_jsonl,
    parse_llama_perplexity_output,
)
from gguf_limit_bench.benchmark_suite import BenchmarkSuitePlan


def test_parse_llama_bench_jsonl_extracts_generation_speed_and_context():
    output = "\n".join(
        [
            json.dumps({"n_prompt": 512, "n_gen": 0, "avg_ts": 930.0}),
            json.dumps({"n_prompt": 0, "n_gen": 128, "n_depth": 8192, "avg_ts": 61.5}),
        ]
    )

    result = parse_llama_bench_jsonl(output, returncode=0)

    assert result.ok is True
    assert result.generation_tokens_per_second == 61.5
    assert result.prompt_tokens_per_second == 930.0
    assert result.context_size == 8192
    assert result.failure == "none"


def test_parse_llama_bench_jsonl_uses_decode_only_row_not_mixed_prompt_generation():
    output = "\n".join(
        [
            json.dumps({"n_prompt": 512, "n_gen": 0, "avg_ts": 1286.674301}),
            json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": 128.792227}),
            json.dumps({"n_prompt": 128, "n_gen": 32, "avg_ts": 475.030187}),
        ]
    )

    result = parse_llama_bench_jsonl(output, returncode=0)

    assert result.ok is True
    assert result.prompt_tokens_per_second == 1286.674301
    assert result.generation_tokens_per_second == 128.792227


def test_autoresearch_llama_bench_command_uses_low_burn_probe():
    command = build_autoresearch_llama_bench_command(
        llama_bench=Path("llama-bench.exe"),
        model=Path("model.gguf"),
        settings=AutoresearchSettings(),
    )

    assert "-pg" in command
    assert "128,32" in command
    assert "--no-warmup" in command


def test_llama_perplexity_command_uses_model_corpus_and_context():
    command = build_llama_perplexity_command(
        llama_perplexity=Path("llama-perplexity.exe"),
        model=Path("model.gguf"),
        corpus=Path("corpus.txt"),
        settings=AutoresearchSettings(context_size=8192, batch_size=1024),
    )

    assert command[:3] == ["llama-perplexity.exe", "--model", "model.gguf"]
    assert ["--file", "corpus.txt"] == command[3:5]
    assert ["--ctx-size", "8192"] in [command[index : index + 2] for index in range(len(command))]


def test_parse_llama_perplexity_output_extracts_final_ppl():
    result = parse_llama_perplexity_output(
        stdout="partial\nFinal estimate: PPL = 6.1234 +/- 0.02",
        stderr="",
        returncode=0,
    )

    assert result.ok is True
    assert result.perplexity == 6.1234


def test_autoresearch_loop_keeps_only_better_setting_and_writes_receipts(tmp_path):
    seen: list[AutoresearchSettings] = []
    speeds = {
        "baseline": 40.0,
        "context": 42.0,
        "batch": 38.0,
    }

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        seen.append(settings)
        if len(seen) == 1:
            speed = speeds["baseline"]
        elif settings.context_size > seen[0].context_size:
            speed = speeds["context"]
        else:
            speed = speeds["batch"]
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=speed,
            prompt_tokens_per_second=800.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        parallel_max=4,
        max_attempts=3,
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (receipt.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert all(settings.kv_unified for settings in seen)
    assert best["settings"]["context_size"] == 8192
    assert best["result"]["generation_tokens_per_second"] == 42.0
    assert [event["type"] for event in events].count("autoresearch_attempt_finished") == 3
    assert (receipt.path / "summary.md").exists()
    assert (receipt.path / "itemized-report.md").exists()
    assert (receipt.path / "report.html").exists()
    assert (receipt.path / "report.json").exists()
    report = json.loads((receipt.path / "report.json").read_text(encoding="utf-8"))
    metrics = {metric["metric"]: metric for metric in report["metric_statuses"]}
    assert metrics["generation_tps"]["status"] == "measured"
    assert metrics["max_total_usable_context"]["status"] == "estimated"
    assert metrics["tps_falloff_with_context"]["status"] == "measured"
    assert metrics["perplexity_falloff"]["status"] == "not_measured"
    assert "Plain-English Takeaway" in (receipt.path / "summary.md").read_text(encoding="utf-8")
    assert (receipt.path / "recovery.json").exists()
    ledger = tmp_path / "autoresearch-results.tsv"
    assert ledger.exists()
    assert (
        "run_id\tmodel\tscore\tstatus\tcontext\tgeneration_tps\tprompt_tps\t"
        "serving_ttft_ms\tserving_warm_ttft_ms\tserving_warmup_penalty_ms\t"
        "serving_server_ready_ms\tserving_cold_start_to_first_token_ms\t"
        "serving_tps\tagent_bench_score\tbenchmark_suite_general_score\t"
        "benchmark_suite_agentic_score\tbenchmark_suite_status\t"
        "benchmark_suite_receipt\tbenchmark_suite_failure\treceipt\tdescription"
    ) in ledger.read_text(encoding="utf-8")
    attempts_ledger = tmp_path / "autoresearch-attempts.tsv"
    assert attempts_ledger.exists()
    attempts_text = attempts_ledger.read_text(encoding="utf-8")
    assert "run_id\tattempt\tbranch\tcommit\tdirty\tmodel\tdecision\tscore" in attempts_text
    assert "\tkeep\t" in attempts_text
    assert "\tdiscard\t" in attempts_text


def test_autoresearch_loop_uses_agent_bench_score_when_suite_plan_is_provided(tmp_path):
    seen: list[AutoresearchSettings] = []

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        seen.append(settings)
        speed = 100.0 if settings.context_size == 4096 else 10.0
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=speed,
            prompt_tokens_per_second=800.0,
            ttft_ms=100.0,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
        )

    score_command = [
        sys.executable,
        "-c",
        (
            "import json; "
            "score=0.4 if int('{context}') == 4096 else 0.9; "
            "print(json.dumps({'score': score}))"
        ),
    ]
    plan_path = tmp_path / "suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "will-be-overridden",
                "context": 0,
                "tasks": [
                    {
                        "id": "general_score",
                        "phase": "general",
                        "harness": "fake-general",
                        "command": score_command,
                    },
                    {
                        "id": "agentic_score",
                        "phase": "agentic",
                        "harness": "fake-agentic",
                        "command": score_command,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=2,
        benchmark_suite_plan=BenchmarkSuitePlan.from_path(plan_path),
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))
    attempts_text = (tmp_path / "autoresearch-attempts.tsv").read_text(encoding="utf-8")
    results_text = (tmp_path / "autoresearch-results.tsv").read_text(encoding="utf-8")
    suite_dirs = sorted(
        path for path in tmp_path.iterdir() if path.is_dir() and "benchmark-suite" in path.name
    )
    suite_plan = json.loads((suite_dirs[-1] / "suite-plan.json").read_text(encoding="utf-8"))

    assert [settings.context_size for settings in seen] == [4096, 8192]
    assert best["settings"]["context_size"] == 8192
    assert best["score"] == 0.9
    assert best["result"]["agent_bench_score"] == 0.9
    assert best["result"]["benchmark_suite_ok"] is True
    assert suite_plan["model"] == "will-be-overridden"
    assert suite_plan["settings"]["gguf_model_path"].endswith("Qwen3-Test-Q4_K_M.gguf")
    assert "agent_bench_score" in attempts_text
    assert "\t0.400000\t0.400000\t0.400000\tpass\t" in attempts_text
    assert "\t0.900000\t0.900000\t0.900000\tpass\t" in results_text


def test_autoresearch_loop_caps_benchmark_suite_to_remaining_budget(tmp_path):
    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=50.0,
            prompt_tokens_per_second=800.0,
            ttft_ms=100.0,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
        )

    slow_command = [
        sys.executable,
        "-c",
        "import json, time; time.sleep(5); print(json.dumps({'score': 1.0}))",
    ]
    plan_path = tmp_path / "slow-suite.plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "model": "local-model",
                "context": 4096,
                "tasks": [
                    {
                        "id": "slow_general",
                        "phase": "general",
                        "harness": "fake-general",
                        "command": slow_command,
                        "timeout_seconds": 30,
                    },
                    {
                        "id": "slow_agentic",
                        "phase": "agentic",
                        "harness": "fake-agentic",
                        "command": slow_command,
                        "timeout_seconds": 30,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=1,
        max_attempts=1,
        benchmark_suite_plan=BenchmarkSuitePlan.from_path(plan_path),
    )

    started = time.monotonic()
    receipt = loop.run()
    elapsed = time.monotonic() - started
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))

    assert elapsed < 3.0
    assert best["result"]["benchmark_suite_ok"] is False
    assert "slow_general:timeout" in best["result"]["benchmark_suite_failure"]


def test_autoresearch_attempt_ledger_records_crash_decision(tmp_path):
    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=False,
            generation_tokens_per_second=0.0,
            prompt_tokens_per_second=0.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="model_load",
            stdout="",
            stderr="failed to load model",
            returncode=1,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Bad.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
    )

    loop.run()
    attempts_text = (tmp_path / "autoresearch-attempts.tsv").read_text(encoding="utf-8")

    assert "\tcrash\t" in attempts_text
    assert "\tfailed\t" in attempts_text


def test_attempt_score_rewards_serving_tps_and_penalizes_real_ttft():
    result = AttemptResult(
        ok=True,
        generation_tokens_per_second=50.0,
        prompt_tokens_per_second=800.0,
        ttft_ms=None,
        context_size=4096,
        failure="none",
        stdout="",
        stderr="",
        returncode=0,
        serving_ttft_ms=500.0,
        serving_tokens_per_second=40.0,
    )

    assert result.score() == 62.5


def test_attempt_score_penalizes_missing_serving_ttft():
    measured = AttemptResult(
        ok=True,
        generation_tokens_per_second=50.0,
        prompt_tokens_per_second=800.0,
        ttft_ms=None,
        context_size=4096,
        failure="none",
        stdout="",
        stderr="",
        returncode=0,
        serving_ttft_ms=500.0,
        serving_tokens_per_second=40.0,
    )
    missing = AttemptResult(
        ok=True,
        generation_tokens_per_second=50.0,
        prompt_tokens_per_second=800.0,
        ttft_ms=None,
        context_size=4096,
        failure="none",
        stdout="",
        stderr="",
        returncode=0,
    )

    assert missing.score() < measured.score()


def test_autoresearch_loop_writes_chartable_serving_metrics_tsv(tmp_path):
    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=50.0,
            prompt_tokens_per_second=800.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
            serving_ttft_ms=100.0,
            serving_tokens_per_second=40.0,
            serving_server_ready_ms=1000.0,
            serving_cold_start_to_first_token_ms=1100.0,
            serving_question_results=[
                {
                    "question_index": 1,
                    "question_id": "latency_definition",
                    "is_cold": True,
                    "ttft_ms": 100.0,
                    "tokens_per_second": 39.0,
                    "generated_tokens": 16,
                    "output_chars": 80,
                    "tokens_cached": 154,
                    "tokens_evaluated": 139,
                },
                {
                    "question_index": 2,
                    "question_id": "tool_plan",
                    "is_cold": False,
                    "ttft_ms": 50.0,
                    "tokens_per_second": 41.0,
                    "generated_tokens": 16,
                    "output_chars": 82,
                    "tokens_cached": 154,
                    "tokens_evaluated": 139,
                },
            ],
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
    )

    loop.run()
    ledger = (tmp_path / "serving-metrics.tsv").read_text(encoding="utf-8")

    assert "run_id\tmodel\tcontext\tquestion_index\tquestion_id" in ledger
    assert "latency_definition" in ledger
    assert "tool_plan" in ledger
    assert "\t4096\t1\t" in ledger


def test_autoresearch_loop_recovers_from_oom_and_continues(tmp_path):
    attempts = 0

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            return AttemptResult(
                ok=False,
                generation_tokens_per_second=0.0,
                prompt_tokens_per_second=0.0,
                ttft_ms=None,
                context_size=settings.context_size,
                failure="gpu_oom",
                stdout="",
                stderr="CUDA error: out of memory",
                returncode=1,
            )
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=50.0 + attempts,
            prompt_tokens_per_second=700.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="{}",
            stderr="",
            returncode=0,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=3,
    )

    receipt = loop.run()
    recovery = json.loads((receipt.path / "recovery.json").read_text(encoding="utf-8"))

    assert attempts == 3
    assert recovery["status"] == "finished"
    assert recovery["detail"] == "none"


def test_autoresearch_loop_preserves_last_failed_attempt_when_no_attempt_succeeds(tmp_path):
    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=False,
            generation_tokens_per_second=0.0,
            prompt_tokens_per_second=0.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="model_load",
            stdout="",
            stderr="failed to load model",
            returncode=1,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Bad.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))

    assert best["result"]["failure"] == "model_load"
    assert best["result"]["stderr"] == "failed to load model"
    assert best["settings"]["context_size"] == 4096


def test_autoresearch_loop_records_failed_learner_settings_as_best_when_all_fail(tmp_path):
    class FakeLearner:
        def suggest(self):
            return type(
                "Suggestion",
                (),
                {
                    "trial_id": 1,
                    "settings": AutoresearchSettings(context_size=65536, gpu_layers=40),
                },
            )()

        def tell(self, suggestion, result: AttemptResult) -> None:
            pass

        def best(self):
            return {"score": -10000.0, "settings": {"context_size": 65536}, "storage": "fake"}

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=False,
            generation_tokens_per_second=0.0,
            prompt_tokens_per_second=0.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="model_load",
            stdout="",
            stderr="failed to load model",
            returncode=1,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Bad.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
        learner=FakeLearner(),
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))

    assert best["settings"]["context_size"] == 65536
    assert best["result"]["context_size"] == 65536


def test_autoresearch_loop_handles_zero_attempts_with_learning_enabled(tmp_path):
    class FakeLearner:
        def suggest(self):
            raise AssertionError("No attempts should be requested")

        def tell(self, suggestion, result: AttemptResult) -> None:
            raise AssertionError("No attempts should be reported")

        def best(self):
            return None

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=lambda settings: AttemptResult(
            ok=True,
            generation_tokens_per_second=1.0,
            prompt_tokens_per_second=1.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="",
            stderr="",
            returncode=0,
        ),
        budget_seconds=60,
        max_attempts=0,
        learner=FakeLearner(),
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))

    assert best["result"]["failure"] == "no_successful_attempt"
    assert best["learner_best"] is None


def test_autoresearch_loop_uses_learning_suggestions_and_reports_scores(tmp_path):
    class FakeLearner:
        def __init__(self) -> None:
            self.reported: list[tuple[object, AttemptResult]] = []

        def suggest(self):
            return type(
                "Suggestion",
                (),
                {
                    "trial_id": 123,
                    "settings": AutoresearchSettings(context_size=8192, parallel=3),
                },
            )()

        def tell(self, suggestion, result: AttemptResult) -> None:
            self.reported.append((suggestion, result))

        def best(self):
            return {"score": 88.0, "settings": {"context_size": 8192, "kv_unified": True}}

    learner = FakeLearner()
    seen: list[AutoresearchSettings] = []

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        seen.append(settings)
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=64.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="",
            stderr="",
            returncode=0,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
        learner=learner,
    )

    receipt = loop.run()
    best = json.loads((receipt.path / "best-settings.json").read_text(encoding="utf-8"))

    assert seen == [AutoresearchSettings(context_size=8192, parallel=3)]
    assert learner.reported[0][0].trial_id == 123
    assert learner.reported[0][1].generation_tokens_per_second == 64.0
    assert best["learner_best"]["score"] == 88.0


def test_autoresearch_loop_writes_context_profile_from_fixed_ladder(tmp_path):
    seen: list[AutoresearchSettings] = []

    def fake_runner(settings: AutoresearchSettings) -> AttemptResult:
        seen.append(settings)
        speed_by_context = {4096: 90.0, 8192: 72.0, 16384: 45.0}
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=speed_by_context[settings.context_size],
            prompt_tokens_per_second=900.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="",
            stderr="",
            returncode=0,
            serving_ttft_ms=200.0 + settings.context_size / 4096,
            serving_warm_ttft_ms=100.0 + settings.context_size / 4096,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_runner,
        budget_seconds=60,
        max_attempts=1,
        context_ladder=(4096, 8192, 16384),
    )

    receipt = loop.run()
    profile = json.loads((receipt.path / "context-profile.json").read_text(encoding="utf-8"))
    report = json.loads((receipt.path / "report.json").read_text(encoding="utf-8"))
    metrics = {metric["metric"]: metric for metric in report["metric_statuses"]}

    assert [settings.context_size for settings in seen] == [4096, 4096, 8192, 16384]
    assert [row["context_size"] for row in profile["rows"]] == [4096, 8192, 16384]
    assert profile["rows"][0]["tps_retention_vs_baseline"] == 1.0
    assert profile["rows"][1]["tps_retention_vs_baseline"] == 0.8
    assert profile["rows"][2]["tps_retention_vs_baseline"] == 0.5
    assert "context_size" in (receipt.path / "context-profile.tsv").read_text(encoding="utf-8")
    assert "Context Profile" in (receipt.path / "context-profile.md").read_text(encoding="utf-8")
    assert metrics["tps_falloff_with_context"]["status"] == "measured"


def test_autoresearch_loop_writes_perplexity_profile_when_runner_is_provided(tmp_path):
    seen_contexts: list[int] = []

    def fake_attempt_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=80.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="none",
            stdout="",
            stderr="",
            returncode=0,
        )

    def fake_perplexity_runner(settings: AutoresearchSettings) -> PerplexityResult:
        seen_contexts.append(settings.context_size)
        return PerplexityResult(
            ok=True,
            perplexity={4096: 6.0, 8192: 6.3, 16384: 7.2}[settings.context_size],
            stderr="",
            stdout="Final estimate: PPL = 6.0",
            returncode=0,
        )

    loop = AutoresearchLoop(
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        runs_root=tmp_path,
        attempt_runner=fake_attempt_runner,
        budget_seconds=60,
        max_attempts=1,
        perplexity_runner=fake_perplexity_runner,
        perplexity_contexts=(4096, 8192, 16384),
    )

    receipt = loop.run()
    profile = json.loads((receipt.path / "perplexity-profile.json").read_text(encoding="utf-8"))
    report = json.loads((receipt.path / "report.json").read_text(encoding="utf-8"))
    metrics = {metric["metric"]: metric for metric in report["metric_statuses"]}

    assert seen_contexts == [4096, 8192, 16384]
    assert [row["perplexity"] for row in profile["rows"]] == [6.0, 6.3, 7.2]
    assert profile["rows"][1]["perplexity_delta_vs_baseline"] == 0.3
    assert "perplexity" in (receipt.path / "perplexity-profile.tsv").read_text(encoding="utf-8")
    assert "Perplexity Profile" in (receipt.path / "perplexity-profile.md").read_text(
        encoding="utf-8"
    )
    assert metrics["perplexity_falloff"]["status"] == "measured"
