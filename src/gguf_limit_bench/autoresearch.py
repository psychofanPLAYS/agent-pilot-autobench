from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from gguf_limit_bench.benchmark_suite import (
    BenchmarkSuitePlan,
    BenchmarkSuiteRun,
    run_benchmark_suite,
)
from gguf_limit_bench.evidence import evidence_status, normalize_success_failure
from gguf_limit_bench.receipts import RunReceipt
from gguf_limit_bench.telemetry import classify_failure, sample_telemetry


@dataclass(frozen=True)
class AutoresearchSettings:
    context_size: int = 4096
    parallel: int = 1
    gpu_layers: int = 99
    batch_size: int = 2048
    ubatch_size: int = 512
    flash_attention: bool = True
    kv_unified: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AttemptResult:
    ok: bool
    generation_tokens_per_second: float
    prompt_tokens_per_second: float
    ttft_ms: float | None
    context_size: int
    failure: str
    stdout: str
    stderr: str
    returncode: int
    workflow_score: float = 0.0
    workflow_results: list[dict] = field(default_factory=list)
    serving_ttft_ms: float | None = None
    serving_tokens_per_second: float | None = None
    serving_warm_ttft_ms: float | None = None
    serving_warmup_penalty_ms: float | None = None
    serving_server_ready_ms: float | None = None
    serving_cold_start_to_first_token_ms: float | None = None
    serving_question_results: list[dict] = field(default_factory=list)
    serving_failure: str | None = None
    agent_bench_score: float | None = None
    benchmark_suite_general_score: float | None = None
    benchmark_suite_agentic_score: float | None = None
    benchmark_suite_ok: bool | None = None
    benchmark_suite_receipt: str | None = None
    benchmark_suite_failure: str | None = None

    def score(self) -> float:
        if not self.ok:
            return -10_000.0
        if self.agent_bench_score is not None:
            return self.agent_bench_score
        context_bonus = min(self.context_size, 131_072) / 4096.0
        measured_ttft = self.serving_ttft_ms if self.serving_ttft_ms is not None else self.ttft_ms
        ttft_penalty = (measured_ttft if measured_ttft is not None else 10_000.0) / 1000.0
        serving_speed_bonus = (self.serving_tokens_per_second or 0.0) / 10.0
        return (
            self.generation_tokens_per_second
            + self.prompt_tokens_per_second / 100.0
            + context_bonus
            + self.workflow_score
            + serving_speed_bonus
            - ttft_penalty
        )

    def to_dict(self) -> dict:
        return asdict(self)


AttemptRunner = Callable[[AutoresearchSettings], AttemptResult]


class LlamaBenchAttemptRunner:
    def __init__(self, llama_bench: Path, model: Path, timeout_seconds: int = 300) -> None:
        self.llama_bench = llama_bench
        self.model = model
        self.timeout_seconds = timeout_seconds

    def __call__(self, settings: AutoresearchSettings) -> AttemptResult:
        command = build_autoresearch_llama_bench_command(
            llama_bench=self.llama_bench,
            model=self.model,
            settings=settings,
        )
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else "benchmark timed out"
            return AttemptResult(
                ok=False,
                generation_tokens_per_second=0.0,
                prompt_tokens_per_second=0.0,
                ttft_ms=None,
                context_size=settings.context_size,
                failure="timeout",
                stdout=stdout,
                stderr=stderr,
                returncode=124,
            )
        return parse_llama_bench_jsonl(
            completed.stdout,
            returncode=completed.returncode,
            stderr=completed.stderr,
            fallback_context=settings.context_size,
        )


class AutoresearchLoop:
    def __init__(
        self,
        model: Path,
        runs_root: Path,
        attempt_runner: AttemptRunner,
        budget_seconds: int,
        parallel_max: int = 4,
        max_attempts: int | None = None,
        learner: Any | None = None,
        benchmark_suite_plan: BenchmarkSuitePlan | None = None,
    ) -> None:
        self.model = model
        self.runs_root = runs_root
        self.attempt_runner = attempt_runner
        self.budget_seconds = budget_seconds
        self.parallel_max = parallel_max
        self.max_attempts = max_attempts
        self.learner = learner
        self.benchmark_suite_plan = benchmark_suite_plan

    def run(self) -> RunReceipt:
        receipt = RunReceipt.create(self.runs_root, slug=_safe_slug(self.model.stem))
        receipt.event(
            "autoresearch_started",
            {
                "model": str(self.model),
                "budget_seconds": self.budget_seconds,
                "parallel_max": self.parallel_max,
            },
        )
        receipt.mark_recovery(step="autoresearch", status="running")

        best_settings = AutoresearchSettings()
        best_result: AttemptResult | None = None
        last_settings = best_settings
        last_result: AttemptResult | None = None
        failures: list[dict] = []
        started = time.monotonic()
        attempt_index = 0

        while time.monotonic() - started < self.budget_seconds:
            if self.max_attempts is not None and attempt_index >= self.max_attempts:
                break
            suggestion = self.learner.suggest() if self.learner is not None else None
            settings = (
                suggestion.settings
                if suggestion is not None
                else self._candidate(best_settings, attempt_index)
            )
            last_settings = settings
            attempt_index += 1
            receipt.event(
                "autoresearch_attempt_started",
                {
                    "attempt": attempt_index,
                    "settings": settings.to_dict(),
                    "learner_trial_id": getattr(suggestion, "trial_id", None),
                    "telemetry": sample_telemetry().to_dict(),
                },
            )
            result = self.attempt_runner(settings)
            if result.ok and self.benchmark_suite_plan is not None:
                remaining_seconds = self.budget_seconds - (time.monotonic() - started)
                result = self._with_benchmark_suite(result, settings, remaining_seconds)
            last_result = result
            if suggestion is not None:
                self.learner.tell(suggestion, result)
            decision = _decision_for_attempt(result, best_result)
            if not result.ok:
                failures.append(
                    {
                        "attempt": attempt_index,
                        "settings": settings.to_dict(),
                        "failure": result.failure,
                    }
                )
            elif best_result is None or result.score() > best_result.score():
                best_settings = settings
                best_result = result

            receipt.event(
                "autoresearch_attempt_finished",
                {
                    "attempt": attempt_index,
                    "settings": settings.to_dict(),
                    "learner_trial_id": getattr(suggestion, "trial_id", None),
                    "result": result.to_dict(),
                    "score": result.score(),
                    "decision": decision,
                    "accepted": best_result == result,
                    "telemetry": sample_telemetry().to_dict(),
                },
            )
            _append_attempts_tsv(
                self.runs_root,
                receipt.path,
                self.model,
                attempt_index,
                settings,
                result,
                decision,
            )
            receipt.mark_recovery(
                step=f"attempt:{attempt_index}",
                status="finished" if result.ok else "failed",
                detail=result.failure,
            )

        if best_result is None:
            best_settings = last_settings
            best_result = last_result or AttemptResult(
                ok=False,
                generation_tokens_per_second=0.0,
                prompt_tokens_per_second=0.0,
                ttft_ms=None,
                context_size=best_settings.context_size,
                failure="no_successful_attempt",
                stdout="",
                stderr="",
                returncode=1,
            )

        receipt.write_json(
            "best-settings.json",
            {
                "model": str(self.model),
                "settings": best_settings.to_dict(),
                "result": best_result.to_dict(),
                "score": best_result.score(),
                "status": _status_for_result(best_result),
                "learner_best": self.learner.best() if self.learner is not None else None,
            },
        )
        if self.learner is not None:
            receipt.write_json("learning.json", self.learner.best() or {})
        if best_result.workflow_results:
            receipt.write_json(
                "workflow-results.json",
                {"score": best_result.workflow_score, "tasks": best_result.workflow_results},
            )
        receipt.write_json("recovery.json", {"status": "finished", "detail": best_result.failure})
        receipt.write_summary(
            _summary_lines(
                self.model,
                best_settings,
                best_result,
                failures,
                self.learner.best() if self.learner is not None else None,
            )
        )
        receipt.event(
            "autoresearch_finished",
            {"settings": best_settings.to_dict(), "result": best_result.to_dict()},
        )
        _append_results_tsv(self.runs_root, receipt.path, self.model, best_settings, best_result)
        _append_serving_metrics_tsv(
            self.runs_root,
            receipt.path,
            self.model,
            best_settings,
            best_result,
        )
        return receipt

    def _candidate(self, best: AutoresearchSettings, attempt_index: int) -> AutoresearchSettings:
        if attempt_index == 0:
            return best
        mutation = (attempt_index - 1) % 4
        if mutation == 0:
            next_context = max(4096, best.context_size * 2)
            return replace(best, context_size=min(next_context, 131_072), kv_unified=True)
        if mutation == 1:
            return replace(best, batch_size=max(512, best.batch_size // 2), kv_unified=True)
        if mutation == 2:
            return replace(
                best,
                parallel=min(self.parallel_max, best.parallel + 1),
                kv_unified=True,
            )
        return replace(best, ubatch_size=max(128, best.ubatch_size // 2), kv_unified=True)

    def _with_benchmark_suite(
        self,
        result: AttemptResult,
        settings: AutoresearchSettings,
        timeout_seconds: float,
    ) -> AttemptResult:
        assert self.benchmark_suite_plan is not None
        plan = BenchmarkSuitePlan(
            model=self.benchmark_suite_plan.model,
            context=settings.context_size,
            settings={
                **self.benchmark_suite_plan.settings,
                **settings.to_dict(),
                "gguf_model_path": str(self.model),
                "score_contract": "agent_bench_score",
            },
            tasks=self.benchmark_suite_plan.tasks,
        )
        suite_run = run_benchmark_suite(plan, self.runs_root, timeout_seconds=timeout_seconds)
        if not suite_run.ok:
            return replace(
                result,
                ok=False,
                failure="benchmark_suite_failed",
                agent_bench_score=suite_run.agent_bench_score,
                benchmark_suite_general_score=suite_run.general_score,
                benchmark_suite_agentic_score=suite_run.agentic_score,
                benchmark_suite_ok=False,
                benchmark_suite_receipt=suite_run.receipt_path,
                benchmark_suite_failure=_benchmark_suite_failure(suite_run),
            )
        return replace(
            result,
            agent_bench_score=suite_run.agent_bench_score,
            benchmark_suite_general_score=suite_run.general_score,
            benchmark_suite_agentic_score=suite_run.agentic_score,
            benchmark_suite_ok=True,
            benchmark_suite_receipt=suite_run.receipt_path,
            benchmark_suite_failure=None,
        )


def build_autoresearch_llama_bench_command(
    llama_bench: Path,
    model: Path,
    settings: AutoresearchSettings,
) -> list[str]:
    command = [
        str(llama_bench),
        "--model",
        str(model),
        "-o",
        "jsonl",
        "-r",
        "1",
        "-pg",
        "128,32",
        "-ngl",
        str(settings.gpu_layers),
        "-fa",
        "1" if settings.flash_attention else "0",
        "-b",
        str(settings.batch_size),
        "-ub",
        str(settings.ubatch_size),
        "--no-warmup",
    ]
    if settings.context_size:
        command.extend(["-d", str(settings.context_size)])
    return command


def parse_llama_bench_jsonl(
    stdout: str,
    returncode: int,
    stderr: str = "",
    fallback_context: int = 0,
) -> AttemptResult:
    prompt_speed = 0.0
    generation_speed = 0.0
    context_size = fallback_context
    for line in stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        speed = float(row.get("avg_ts") or row.get("tokens_per_second") or 0.0)
        n_gen = int(row.get("n_gen") or row.get("n_generated") or 0)
        n_prompt = int(row.get("n_prompt") or 0)
        context_size = int(row.get("n_depth") or row.get("n_ctx") or context_size)
        if n_gen > 0 and n_prompt == 0:
            generation_speed = max(generation_speed, speed)
        elif n_prompt > 0 and n_gen == 0:
            prompt_speed = max(prompt_speed, speed)

    failure = classify_failure(stderr + "\n" + stdout)
    if returncode == 0 and failure == "unknown" and generation_speed <= 0.0:
        failure = "no_decode_row"
    ok = returncode == 0 and generation_speed > 0.0
    return AttemptResult(
        ok=ok,
        generation_tokens_per_second=generation_speed,
        prompt_tokens_per_second=prompt_speed,
        ttft_ms=None,
        context_size=context_size,
        failure=normalize_success_failure(ok, failure),
        stdout=stdout[-8000:],
        stderr=stderr[-8000:],
        returncode=returncode,
    )


def _summary_lines(
    model: Path,
    settings: AutoresearchSettings,
    result: AttemptResult,
    failures: list[dict],
    learner_best: dict | None = None,
) -> list[str]:
    plain_takeaway = _plain_english_takeaway(result)
    lines = [
        f"# {model.name}",
        "",
        "## Plain-English Takeaway",
        "",
        f"- {plain_takeaway}",
        f"- Score: `{result.score():.2f}`. Higher is better; failed attempts get a very low score.",
        "",
        "## Best Settings",
        "",
        f"- Context: `{settings.context_size}`",
        f"- Parallel: `{settings.parallel}`",
        f"- GPU layers: `{settings.gpu_layers}`",
        f"- Batch / ubatch: `{settings.batch_size}` / `{settings.ubatch_size}`",
        f"- Flash attention: `{settings.flash_attention}`",
        f"- Unified KV cache: `{settings.kv_unified}`",
        "",
        "## Best Result",
        "",
        f"- Generation tokens/sec: `{result.generation_tokens_per_second}`",
        f"- Prompt tokens/sec: `{result.prompt_tokens_per_second}`",
        f"- Workflow score: `{result.workflow_score}`",
        f"- Agent bench score: `{result.agent_bench_score}`",
        f"- Benchmark suite general score: `{result.benchmark_suite_general_score}`",
        f"- Benchmark suite agentic score: `{result.benchmark_suite_agentic_score}`",
        f"- Benchmark suite status: `{_benchmark_suite_status(result)}`",
        f"- Benchmark suite receipt: `{result.benchmark_suite_receipt or 'none'}`",
        f"- Benchmark suite failure: `{result.benchmark_suite_failure or 'none'}`",
        f"- Serving cold TTFT ms: `{result.serving_ttft_ms}`",
        f"- Serving warm TTFT ms: `{result.serving_warm_ttft_ms}`",
        f"- Serving warmup penalty ms: `{result.serving_warmup_penalty_ms}`",
        f"- Serving server-ready ms: `{result.serving_server_ready_ms}`",
        f"- Serving server-start to first-token ms: `{result.serving_cold_start_to_first_token_ms}`",
        f"- Serving tokens/sec: `{result.serving_tokens_per_second}`",
        f"- Serving question count: `{len(result.serving_question_results)}`",
        f"- Serving probe failure: `{result.serving_failure or 'none'}`",
        f"- Failure class: `{result.failure}`",
        f"- Status: `{_status_for_result(result)}`",
        "",
        "## Recovery",
        "",
        f"- Failed attempts recorded: `{len(failures)}`",
        "- Full attempt log: `events.jsonl`",
    ]
    if learner_best is not None:
        lines.extend(
            [
                "",
                "## Learning",
                "",
                f"- Best learned score: `{learner_best['score']}`",
                f"- Learning storage: `{learner_best.get('storage', 'unknown')}`",
            ]
        )
    return lines


def _plain_english_takeaway(result: AttemptResult) -> str:
    if not result.ok:
        if result.benchmark_suite_ok is False:
            return (
                "The speed probe loaded, but the required benchmark suite failed. "
                "Do not treat this as production-ready evidence."
            )
        if result.failure == "gpu_oom":
            return (
                "This setting ran out of GPU memory. Try a smaller context, "
                "lower batch size, or fewer GPU layers."
            )
        if result.failure == "model_load":
            return (
                "The model did not load. Check the model file and llama.cpp "
                "compatibility before retesting."
            )
        if result.failure == "timeout":
            return (
                "The run took too long. Use a smaller preset or a shorter model "
                "list for the next pass."
            )
        return (
            "No useful benchmark result was produced yet. The receipt still records "
            "what failed so the next run can avoid repeating it."
        )
    if result.generation_tokens_per_second >= 20.0:
        if result.serving_ttft_ms is None:
            return (
                "This is llama-bench speed evidence only. Real serving TTFT is not "
                "proven for this setting yet."
            )
        if result.context_size <= 0:
            return (
                "This has real serving TTFT plus speed evidence, but no useful agent "
                "context target was proven."
            )
        if result.benchmark_suite_ok:
            return (
                "This setting has speed evidence plus a passing general and agentic "
                "benchmark suite. Compare it by agent_bench_score, not raw tokens/sec alone."
            )
        if result.workflow_score <= 0:
            return "This is context and speed evidence, but workflow usefulness is still unproven."
        return "This has useful speed, but agent readiness depends on stronger workflow evidence."
    return (
        "This loaded, but it is below the current useful-pilot speed target. "
        "Retest only if quality or context is especially valuable."
    )


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)[:80]


def _status_for_result(result: AttemptResult) -> str:
    return evidence_status(
        ok=result.ok,
        failure=result.failure,
        generation_tps=result.generation_tokens_per_second,
        context_size=result.context_size,
        workflow_score=result.workflow_score,
        workflow_results=result.workflow_results,
        serving_ttft_ms=result.serving_ttft_ms,
    ).value


def _decision_for_attempt(
    result: AttemptResult,
    previous_best: AttemptResult | None,
) -> str:
    if not result.ok:
        return "crash"
    if previous_best is None:
        return "keep"
    return "keep" if result.score() > previous_best.score() else "discard"


def _benchmark_suite_status(result: AttemptResult) -> str:
    if result.benchmark_suite_ok is True:
        return "pass"
    if result.benchmark_suite_ok is False:
        return "fail"
    return "not_run"


def _benchmark_suite_failure(suite_run: BenchmarkSuiteRun) -> str:
    failures = [
        f"{item.id}:{item.failure_class}"
        for item in suite_run.results
        if not item.ok or item.failure_class != "none"
    ]
    return ";".join(failures) if failures else "benchmark_suite_failed"


def _append_attempts_tsv(
    runs_root: Path,
    receipt_path: Path,
    model: Path,
    attempt_index: int,
    settings: AutoresearchSettings,
    result: AttemptResult,
    decision: str,
) -> None:
    ledger = runs_root / "autoresearch-attempts.tsv"
    if not ledger.exists():
        ledger.write_text(
            (
                "run_id\tattempt\tbranch\tcommit\tdirty\tmodel\tdecision\tscore\t"
                "evidence_status\tcontext\t"
                "generation_tps\tprompt_tps\tserving_ttft_ms\tserving_warm_ttft_ms\t"
                "serving_tps\tagent_bench_score\tbenchmark_suite_general_score\t"
                "benchmark_suite_agentic_score\tbenchmark_suite_status\t"
                "benchmark_suite_receipt\tbenchmark_suite_failure\tsettings_json\t"
                "receipt\tdescription\n"
            ),
            encoding="utf-8",
        )
    description = _plain_english_takeaway(result).replace("\t", " ").replace("\n", " ")
    settings_json = json.dumps(settings.to_dict(), sort_keys=True, separators=(",", ":"))
    git_metadata = _git_metadata(Path.cwd())
    line = "\t".join(
        [
            receipt_path.name,
            str(attempt_index),
            git_metadata["branch"],
            git_metadata["commit"],
            git_metadata["dirty"],
            model.name,
            decision,
            f"{result.score():.6f}",
            _status_for_result(result),
            str(settings.context_size),
            f"{result.generation_tokens_per_second:.6f}",
            f"{result.prompt_tokens_per_second:.6f}",
            "" if result.serving_ttft_ms is None else f"{result.serving_ttft_ms:.6f}",
            "" if result.serving_warm_ttft_ms is None else f"{result.serving_warm_ttft_ms:.6f}",
            (
                ""
                if result.serving_tokens_per_second is None
                else f"{result.serving_tokens_per_second:.6f}"
            ),
            _tsv_float(result.agent_bench_score),
            _tsv_float(result.benchmark_suite_general_score),
            _tsv_float(result.benchmark_suite_agentic_score),
            _benchmark_suite_status(result),
            result.benchmark_suite_receipt or "",
            result.benchmark_suite_failure or "",
            settings_json,
            str(receipt_path),
            description,
        ]
    )
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line + "\n")


def _git_metadata(cwd: Path) -> dict[str, str]:
    def run_git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
        if completed.returncode != 0:
            return "unknown"
        return completed.stdout.strip() or "unknown"

    branch = run_git("branch", "--show-current")
    commit = run_git("rev-parse", "--short=12", "HEAD")
    dirty = "unknown"
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    else:
        if completed.returncode == 0:
            dirty = "yes" if completed.stdout.strip() else "no"
    return {"branch": branch, "commit": commit, "dirty": dirty}


def _append_results_tsv(
    runs_root: Path,
    receipt_path: Path,
    model: Path,
    settings: AutoresearchSettings,
    result: AttemptResult,
) -> None:
    ledger = runs_root / "autoresearch-results.tsv"
    if not ledger.exists():
        ledger.write_text(
            (
                "run_id\tmodel\tscore\tstatus\tcontext\tgeneration_tps\tprompt_tps\t"
                "serving_ttft_ms\tserving_warm_ttft_ms\tserving_warmup_penalty_ms\t"
                "serving_server_ready_ms\tserving_cold_start_to_first_token_ms\t"
                "serving_tps\tagent_bench_score\tbenchmark_suite_general_score\t"
                "benchmark_suite_agentic_score\tbenchmark_suite_status\t"
                "benchmark_suite_receipt\tbenchmark_suite_failure\treceipt\tdescription\n"
            ),
            encoding="utf-8",
        )
    status = _status_for_result(result)
    description = _plain_english_takeaway(result).replace("\t", " ").replace("\n", " ")
    line = "\t".join(
        [
            receipt_path.name,
            model.name,
            f"{result.score():.6f}",
            status,
            str(settings.context_size),
            f"{result.generation_tokens_per_second:.6f}",
            f"{result.prompt_tokens_per_second:.6f}",
            "" if result.serving_ttft_ms is None else f"{result.serving_ttft_ms:.6f}",
            ("" if result.serving_warm_ttft_ms is None else f"{result.serving_warm_ttft_ms:.6f}"),
            (
                ""
                if result.serving_warmup_penalty_ms is None
                else f"{result.serving_warmup_penalty_ms:.6f}"
            ),
            (
                ""
                if result.serving_server_ready_ms is None
                else f"{result.serving_server_ready_ms:.6f}"
            ),
            (
                ""
                if result.serving_cold_start_to_first_token_ms is None
                else f"{result.serving_cold_start_to_first_token_ms:.6f}"
            ),
            (
                ""
                if result.serving_tokens_per_second is None
                else f"{result.serving_tokens_per_second:.6f}"
            ),
            _tsv_float(result.agent_bench_score),
            _tsv_float(result.benchmark_suite_general_score),
            _tsv_float(result.benchmark_suite_agentic_score),
            _benchmark_suite_status(result),
            result.benchmark_suite_receipt or "",
            result.benchmark_suite_failure or "",
            str(receipt_path),
            description,
        ]
    )
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line + "\n")


def _append_serving_metrics_tsv(
    runs_root: Path,
    receipt_path: Path,
    model: Path,
    settings: AutoresearchSettings,
    result: AttemptResult,
) -> None:
    if not result.serving_question_results:
        return
    ledger = runs_root / "serving-metrics.tsv"
    if not ledger.exists():
        ledger.write_text(
            (
                "run_id\tmodel\tcontext\tquestion_index\tquestion_id\tis_cold\t"
                "ttft_ms\ttokens_per_second\tgenerated_tokens\toutput_chars\t"
                "tokens_cached\ttokens_evaluated\tserver_ready_ms\t"
                "cold_start_to_first_token_ms\treceipt\n"
            ),
            encoding="utf-8",
        )
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        for question in result.serving_question_results:
            handle.write(
                "\t".join(
                    [
                        receipt_path.name,
                        model.name,
                        str(settings.context_size),
                        str(question.get("question_index", "")),
                        str(question.get("question_id", "")),
                        str(question.get("is_cold", "")),
                        _tsv_float(question.get("ttft_ms")),
                        _tsv_float(question.get("tokens_per_second")),
                        str(question.get("generated_tokens", "")),
                        str(question.get("output_chars", "")),
                        (
                            ""
                            if question.get("tokens_cached") is None
                            else str(question["tokens_cached"])
                        ),
                        (
                            ""
                            if question.get("tokens_evaluated") is None
                            else str(question["tokens_evaluated"])
                        ),
                        _tsv_float(result.serving_server_ready_ms),
                        _tsv_float(result.serving_cold_start_to_first_token_ms),
                        str(receipt_path),
                    ]
                )
                + "\n"
            )


def _tsv_float(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"
