import json
from pathlib import Path

from gguf_limit_bench.autoresearch import (
    AttemptResult,
    AutoresearchLoop,
    AutoresearchSettings,
    build_autoresearch_llama_bench_command,
    parse_llama_bench_jsonl,
)


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
    assert "Plain-English Takeaway" in (receipt.path / "summary.md").read_text(encoding="utf-8")
    assert (receipt.path / "recovery.json").exists()
    ledger = tmp_path / "autoresearch-results.tsv"
    assert ledger.exists()
    assert (
        "run_id\tmodel\tscore\tstatus\tcontext\tgeneration_tps\tprompt_tps\t"
        "serving_ttft_ms\tserving_warm_ttft_ms\tserving_warmup_penalty_ms\t"
        "serving_server_ready_ms\tserving_cold_start_to_first_token_ms\t"
        "serving_tps\treceipt\tdescription"
    ) in ledger.read_text(encoding="utf-8")
    attempts_ledger = tmp_path / "autoresearch-attempts.tsv"
    assert attempts_ledger.exists()
    attempts_text = attempts_ledger.read_text(encoding="utf-8")
    assert "run_id\tattempt\tbranch\tcommit\tdirty\tmodel\tdecision\tscore" in attempts_text
    assert "\tkeep\t" in attempts_text
    assert "\tdiscard\t" in attempts_text


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
