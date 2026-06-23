from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable
import warnings

from gguf_limit_bench.benchmark_suite import (
    BenchmarkSuitePlan,
    BenchmarkSuiteRun,
    run_benchmark_suite,
)
from gguf_limit_bench.evidence import evidence_status, normalize_success_failure
from gguf_limit_bench.receipts import RunReceipt
from gguf_limit_bench.run_report import write_itemized_run_report
from gguf_limit_bench.telemetry import classify_failure, sample_telemetry


BASE_SETTING_FIELDS = (
    "context_size",
    "parallel",
    "gpu_layers",
    "batch_size",
    "ubatch_size",
    "flash_attention",
    "kv_unified",
)
EXTRA_SETTING_DEFAULTS = {
    "profile_name": "baseline",
    "cont_batching": True,
    "cache_ram_mb": None,
    "cache_reuse": None,
    "cache_idle_slots": False,
    "ctx_checkpoints": None,
    "checkpoint_min_step": None,
    "cache_type_k": None,
    "cache_type_v": None,
    "threads": None,
    "threads_batch": None,
    "spec_type": None,
    "spec_draft_n_max": None,
    "spec_draft_n_min": None,
    "spec_draft_p_min": None,
    "extra_server_args": (),
}


@dataclass(frozen=True)
class AutoresearchSettings:
    profile_name: str = "baseline"
    context_size: int = 4096
    parallel: int = 1
    gpu_layers: int = 99
    batch_size: int = 2048
    ubatch_size: int = 512
    flash_attention: bool = True
    kv_unified: bool = True
    cont_batching: bool = True
    cache_ram_mb: int | None = None
    cache_reuse: int | None = None
    cache_idle_slots: bool = False
    ctx_checkpoints: int | None = None
    checkpoint_min_step: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    threads: int | None = None
    threads_batch: int | None = None
    spec_type: str | None = None
    spec_draft_n_max: int | None = None
    spec_draft_n_min: int | None = None
    spec_draft_p_min: float | None = None
    extra_server_args: tuple[str, ...] = ()
    draft_max: int | None = None
    draft_min: int | None = None
    draft_p_min: float | None = None

    def __post_init__(self) -> None:
        deprecated = [
            name
            for name in ("draft_max", "draft_min", "draft_p_min")
            if getattr(self, name) is not None
        ]
        if not deprecated:
            return
        warnings.warn(
            f"{', '.join(deprecated)} are deprecated; use native spec_draft_* settings",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.spec_type is None:
            object.__setattr__(self, "spec_type", "draft-mtp")
        if self.spec_draft_n_max is None and self.draft_max is not None:
            object.__setattr__(self, "spec_draft_n_max", self.draft_max)
        if self.spec_draft_n_min is None and self.draft_min is not None:
            object.__setattr__(self, "spec_draft_n_min", self.draft_min)
        if self.spec_draft_p_min is None and self.draft_p_min is not None:
            object.__setattr__(self, "spec_draft_p_min", self.draft_p_min)

    def to_dict(self) -> dict:
        payload = {field_name: getattr(self, field_name) for field_name in BASE_SETTING_FIELDS}
        for field_name, default in EXTRA_SETTING_DEFAULTS.items():
            value = getattr(self, field_name)
            if value != default:
                payload[field_name] = value
        return payload


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
    flag_profile: str | None = None
    launch_command: list[str] = field(default_factory=list)
    simple_bench_score: float | None = None
    simple_bench_accuracy: float | None = None
    simple_bench_receipt: str | None = None
    simple_bench_failure: str | None = None

    def score(self) -> float:
        if not self.ok:
            return -10_000.0
        if self.simple_bench_score is not None:
            return self.simple_bench_score
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


@dataclass(frozen=True)
class PerplexityResult:
    ok: bool
    perplexity: float | None
    stdout: str
    stderr: str
    returncode: int
    failure: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


PerplexityRunner = Callable[[AutoresearchSettings], PerplexityResult]


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


class LlamaPerplexityRunner:
    def __init__(
        self,
        llama_perplexity: Path,
        model: Path,
        corpus: Path,
        timeout_seconds: int = 600,
    ) -> None:
        self.llama_perplexity = llama_perplexity
        self.model = model
        self.corpus = corpus
        self.timeout_seconds = timeout_seconds

    def __call__(self, settings: AutoresearchSettings) -> PerplexityResult:
        command = build_llama_perplexity_command(
            llama_perplexity=self.llama_perplexity,
            model=self.model,
            corpus=self.corpus,
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
            stderr = exc.stderr if isinstance(exc.stderr, str) else "perplexity timed out"
            return PerplexityResult(
                ok=False,
                perplexity=None,
                stdout=stdout[-8000:],
                stderr=stderr[-8000:],
                returncode=124,
                failure="timeout",
            )
        return parse_llama_perplexity_output(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
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
        context_ladder: tuple[int, ...] | None = None,
        perplexity_runner: PerplexityRunner | None = None,
        perplexity_contexts: tuple[int, ...] | None = None,
        candidate_sequence: tuple[AutoresearchSettings, ...] | None = None,
        skipped_profiles: tuple[dict, ...] = (),
        round_seconds: int | None = None,
    ) -> None:
        self.model = model
        self.runs_root = runs_root
        self.attempt_runner = attempt_runner
        self.budget_seconds = budget_seconds
        # Karpathy round cap: when set, no single attempt may exceed this, so the
        # loop keeps the fixed-budget-per-round cadence instead of letting one
        # profile eat the whole session.
        self.round_seconds = round_seconds
        self.parallel_max = parallel_max
        self.max_attempts = max_attempts
        self.learner = learner
        self.benchmark_suite_plan = benchmark_suite_plan
        self.context_ladder = context_ladder
        self.perplexity_runner = perplexity_runner
        self.perplexity_contexts = perplexity_contexts
        self.candidate_sequence = candidate_sequence
        self.skipped_profiles = skipped_profiles

    def run(self) -> RunReceipt:
        receipt = RunReceipt.create(self.runs_root, slug=_safe_slug(self.model.stem))
        attach_receipt = getattr(self.attempt_runner, "set_receipt_path", None)
        if callable(attach_receipt):
            attach_receipt(receipt.path)
        receipt.event(
            "autoresearch_started",
            {
                "model": str(self.model),
                "budget_seconds": self.budget_seconds,
                "parallel_max": self.parallel_max,
                "candidate_sequence": [
                    settings.to_dict() for settings in self.candidate_sequence or ()
                ],
            },
        )
        if self.skipped_profiles:
            receipt.event(
                "flag_ladder_profiles_skipped",
                {"profiles": list(self.skipped_profiles)},
            )
        receipt.mark_recovery(step="autoresearch", status="running")

        best_settings = AutoresearchSettings()
        best_result: AttemptResult | None = None
        last_settings = best_settings
        last_result: AttemptResult | None = None
        failures: list[dict] = []
        attempt_records: list[tuple[AutoresearchSettings, AttemptResult]] = []
        started = time.monotonic()
        attempt_index = 0

        while time.monotonic() - started < self.budget_seconds:
            if self.max_attempts is not None and attempt_index >= self.max_attempts:
                break
            in_ladder = self.candidate_sequence is not None and attempt_index < len(
                self.candidate_sequence
            )
            # The ordered ladder runs first. Without a learner the run ends when the
            # ladder is exhausted; with a learner it then keeps searching (Optuna)
            # until the time budget, which is what makes an overnight run converge.
            if not in_ladder and self.candidate_sequence is not None and self.learner is None:
                break
            if in_ladder and self.candidate_sequence is not None:
                suggestion = None
                settings = self.candidate_sequence[attempt_index]
            else:
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
            remaining_seconds = self.budget_seconds - (time.monotonic() - started)
            attempt_seconds = remaining_seconds
            if self.round_seconds is not None:
                attempt_seconds = min(attempt_seconds, self.round_seconds)
            set_timeout = getattr(self.attempt_runner, "set_timeout_seconds", None)
            if callable(set_timeout):
                set_timeout(max(1, math.ceil(attempt_seconds)))
            result = self.attempt_runner(settings)
            if result.ok and self.benchmark_suite_plan is not None:
                remaining_seconds = self.budget_seconds - (time.monotonic() - started)
                result = self._with_benchmark_suite(result, settings, remaining_seconds)
            last_result = result
            attempt_records.append((settings, result))
            if suggestion is not None and self.learner is not None:
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

        planned_profiles = (
            len(self.candidate_sequence) if self.candidate_sequence is not None else None
        )
        ladder_complete = planned_profiles is None or len(attempt_records) >= planned_profiles

        receipt.write_json(
            "best-settings.json",
            {
                "model": str(self.model),
                "settings": best_settings.to_dict(),
                "result": best_result.to_dict(),
                "score": best_result.score(),
                "status": _status_for_result(best_result) if ladder_complete else "partial",
                "promotion_eligible": ladder_complete,
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
        if self.context_ladder:
            self._write_context_profile(receipt, best_settings)
        if self.perplexity_runner is not None and self.perplexity_contexts:
            self._write_perplexity_profile(receipt, best_settings)
        if self.candidate_sequence is not None:
            assert planned_profiles is not None
            _write_flag_ladder_comparison(
                receipt=receipt,
                model=self.model,
                attempts=attempt_records,
                champion_profile=best_settings.profile_name if ladder_complete else None,
                provisional_best_profile=best_settings.profile_name,
                planned_profiles=planned_profiles,
            )
        write_itemized_run_report(receipt.path)
        return receipt

    def _candidate(self, best: AutoresearchSettings, attempt_index: int) -> AutoresearchSettings:
        if self.candidate_sequence is not None:
            return self.candidate_sequence[attempt_index]
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

    def _write_context_profile(
        self, receipt: RunReceipt, best_settings: AutoresearchSettings
    ) -> None:
        assert self.context_ladder is not None
        rows: list[dict] = []
        baseline_tps: float | None = None
        receipt.event(
            "context_ladder_started",
            {"contexts": list(self.context_ladder), "base_settings": best_settings.to_dict()},
        )
        for context_size in self.context_ladder:
            settings = replace(best_settings, context_size=int(context_size), kv_unified=True)
            result = self.attempt_runner(settings)
            if result.ok and result.generation_tokens_per_second > 0 and baseline_tps is None:
                baseline_tps = result.generation_tokens_per_second
            retention = (
                result.generation_tokens_per_second / baseline_tps
                if result.ok and baseline_tps
                else None
            )
            row = {
                "context_size": settings.context_size,
                "ok": result.ok,
                "generation_tps": result.generation_tokens_per_second,
                "prompt_tps": result.prompt_tokens_per_second,
                "tps_retention_vs_baseline": retention,
                "cold_ttft_ms": result.serving_ttft_ms,
                "warm_ttft_ms": result.serving_warm_ttft_ms,
                "serving_tps": result.serving_tokens_per_second,
                "failure": result.failure,
                "settings": settings.to_dict(),
            }
            rows.append(row)
            receipt.event(
                "context_ladder_attempt_finished",
                {"settings": settings.to_dict(), "result": result.to_dict(), "row": row},
            )

        payload = {
            "run_id": receipt.path.name,
            "model": str(self.model),
            "base_settings": best_settings.to_dict(),
            "rows": rows,
        }
        receipt.write_json("context-profile.json", payload)
        (receipt.path / "context-profile.tsv").write_text(
            _context_profile_tsv(rows), encoding="utf-8"
        )
        (receipt.path / "context-profile.md").write_text(
            _context_profile_markdown(payload), encoding="utf-8"
        )
        receipt.event("context_ladder_finished", {"rows": rows})

    def _write_perplexity_profile(
        self, receipt: RunReceipt, best_settings: AutoresearchSettings
    ) -> None:
        assert self.perplexity_runner is not None
        assert self.perplexity_contexts is not None
        rows: list[dict] = []
        baseline_perplexity: float | None = None
        receipt.event(
            "perplexity_profile_started",
            {"contexts": list(self.perplexity_contexts), "base_settings": best_settings.to_dict()},
        )
        for context_size in self.perplexity_contexts:
            settings = replace(best_settings, context_size=int(context_size), kv_unified=True)
            result = self.perplexity_runner(settings)
            if result.ok and result.perplexity is not None and baseline_perplexity is None:
                baseline_perplexity = result.perplexity
            delta = (
                round(result.perplexity - baseline_perplexity, 6)
                if result.ok and result.perplexity is not None and baseline_perplexity is not None
                else None
            )
            row = {
                "context_size": settings.context_size,
                "ok": result.ok,
                "perplexity": result.perplexity,
                "perplexity_delta_vs_baseline": delta,
                "failure": result.failure,
                "settings": settings.to_dict(),
            }
            rows.append(row)
            receipt.event(
                "perplexity_profile_attempt_finished",
                {"settings": settings.to_dict(), "result": result.to_dict(), "row": row},
            )

        payload = {
            "run_id": receipt.path.name,
            "model": str(self.model),
            "base_settings": best_settings.to_dict(),
            "rows": rows,
        }
        receipt.write_json("perplexity-profile.json", payload)
        (receipt.path / "perplexity-profile.tsv").write_text(
            _perplexity_profile_tsv(rows), encoding="utf-8"
        )
        (receipt.path / "perplexity-profile.md").write_text(
            _perplexity_profile_markdown(payload), encoding="utf-8"
        )
        receipt.event("perplexity_profile_finished", {"rows": rows})


def _context_profile_tsv(rows: list[dict]) -> str:
    header = (
        "context_size\tok\tgeneration_tps\tprompt_tps\t"
        "tps_retention_vs_baseline\tcold_ttft_ms\twarm_ttft_ms\t"
        "serving_tps\tfailure"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["context_size"]),
                    str(row["ok"]).lower(),
                    _tsv_float(row["generation_tps"]),
                    _tsv_float(row["prompt_tps"]),
                    _tsv_float(row["tps_retention_vs_baseline"]),
                    _tsv_float(row["cold_ttft_ms"]),
                    _tsv_float(row["warm_ttft_ms"]),
                    _tsv_float(row["serving_tps"]),
                    str(row["failure"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _context_profile_markdown(payload: dict) -> str:
    lines = [
        f"# Context Profile: {Path(payload['model']).name}",
        "",
        "Fixed-context ladder using the best-known settings from this run.",
        "",
        "| Context | OK | Gen TPS | Prompt TPS | TPS Retention | Cold TTFT | Warm TTFT | Serving TPS | Failure |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['context_size']} | {str(row['ok']).lower()} | "
            f"{_md_float(row['generation_tps'])} | {_md_float(row['prompt_tps'])} | "
            f"{_md_float(row['tps_retention_vs_baseline'])} | {_md_float(row['cold_ttft_ms'])} | "
            f"{_md_float(row['warm_ttft_ms'])} | {_md_float(row['serving_tps'])} | "
            f"{row['failure']} |"
        )
    return "\n".join(lines) + "\n"


def _perplexity_profile_tsv(rows: list[dict]) -> str:
    header = "context_size\tok\tperplexity\tperplexity_delta_vs_baseline\tfailure"
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["context_size"]),
                    str(row["ok"]).lower(),
                    _tsv_float(row["perplexity"]),
                    _tsv_float(row["perplexity_delta_vs_baseline"]),
                    str(row["failure"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _perplexity_profile_markdown(payload: dict) -> str:
    lines = [
        f"# Perplexity Profile: {Path(payload['model']).name}",
        "",
        "Fixed-context perplexity ladder using the best-known settings from this run.",
        "",
        "| Context | OK | Perplexity | Delta vs baseline | Failure |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['context_size']} | {str(row['ok']).lower()} | "
            f"{_md_float(row['perplexity'])} | "
            f"{_md_float(row['perplexity_delta_vs_baseline'])} | "
            f"{row['failure']} |"
        )
    return "\n".join(lines) + "\n"


def _md_float(value) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _write_flag_ladder_comparison(
    *,
    receipt: RunReceipt,
    model: Path,
    attempts: list[tuple[AutoresearchSettings, AttemptResult]],
    champion_profile: str | None,
    provisional_best_profile: str,
    planned_profiles: int,
) -> None:
    baseline_tps = next(
        (
            result.generation_tokens_per_second
            for settings, result in attempts
            if settings.profile_name == "L0-baseline" and result.ok
        ),
        None,
    )
    rows = []
    for settings, result in attempts:
        tps = result.generation_tokens_per_second if result.ok else None
        slowdown_percent = (
            ((baseline_tps - tps) / baseline_tps) * 100.0
            if baseline_tps and tps is not None
            else None
        )
        warning_count = _attempt_warning_count(result.simple_bench_receipt)
        rows.append(
            {
                "profile": settings.profile_name,
                "ok": result.ok,
                "champion": settings.profile_name == champion_profile,
                "score": result.simple_bench_score,
                "accuracy": result.simple_bench_accuracy,
                "median_tps": tps,
                "slowdown_vs_baseline_percent": slowdown_percent,
                "median_ttft_ms": result.ttft_ms,
                "warning_count": warning_count,
                "failure": result.failure,
                "settings": settings.to_dict(),
                "command": result.launch_command,
                "receipt": result.simple_bench_receipt,
            }
        )
    payload = {
        "model": str(model),
        "status": "complete" if len(attempts) >= planned_profiles else "partial",
        "planned_profiles": planned_profiles,
        "completed_profiles": len(attempts),
        "champion_profile": champion_profile,
        "provisional_best_profile": provisional_best_profile,
        "baseline_profile": "L0-baseline",
        "baseline_tps": baseline_tps,
        "rows": rows,
    }
    receipt.write_json("flag-ladder-results.json", payload)
    (receipt.path / "flag-ladder-results.tsv").write_text(_flag_ladder_tsv(rows), encoding="utf-8")
    (receipt.path / "flag-ladder-results.md").write_text(
        _flag_ladder_markdown(payload), encoding="utf-8"
    )


def _attempt_warning_count(receipt_path: str | None) -> int | None:
    if not receipt_path:
        return None
    summary_path = Path(receipt_path) / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("warning_count")
    return int(value) if value is not None else None


def _flag_ladder_tsv(rows: list[dict]) -> str:
    header = (
        "profile\tok\tchampion\tscore\taccuracy\tmedian_tps\t"
        "slowdown_vs_baseline_percent\tmedian_ttft_ms\twarning_count\tfailure\treceipt"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row["profile"],
                    str(row["ok"]).lower(),
                    str(row["champion"]).lower(),
                    _tsv_float(row["score"]),
                    _tsv_float(row["accuracy"]),
                    _tsv_float(row["median_tps"]),
                    _tsv_float(row["slowdown_vs_baseline_percent"]),
                    _tsv_float(row["median_ttft_ms"]),
                    "" if row["warning_count"] is None else str(row["warning_count"]),
                    str(row["failure"]),
                    str(row["receipt"] or ""),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _flag_ladder_markdown(payload: dict) -> str:
    result_label = (
        f"Champion: `{payload['champion_profile']}`"
        if payload["champion_profile"] is not None
        else f"Provisional best: `{payload['provisional_best_profile']}` (partial ladder)"
    )
    lines = [
        f"# Flag Ladder Results: {Path(payload['model']).name}",
        "",
        (
            f"Status: `{payload['status']}` "
            f"({payload['completed_profiles']}/{payload['planned_profiles']} profiles)"
        ),
        "",
        result_label,
        "",
        "Positive slowdown means slower than L0; negative means faster.",
        "",
        "| Profile | OK | Champion | Accuracy | TPS | Slowdown vs L0 | TTFT ms | Warnings | Failure |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['profile']} | {str(row['ok']).lower()} | "
            f"{str(row['champion']).lower()} | {_md_float(row['accuracy'])} | "
            f"{_md_float(row['median_tps'])} | "
            f"{_md_float(row['slowdown_vs_baseline_percent'])}% | "
            f"{_md_float(row['median_ttft_ms'])} | "
            f"{row['warning_count'] if row['warning_count'] is not None else 'n/a'} | "
            f"{row['failure']} |"
        )
    return "\n".join(lines) + "\n"


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


def build_llama_perplexity_command(
    llama_perplexity: Path,
    model: Path,
    corpus: Path,
    settings: AutoresearchSettings,
) -> list[str]:
    return [
        str(llama_perplexity),
        "--model",
        str(model),
        "--file",
        str(corpus),
        "--ctx-size",
        str(settings.context_size),
        "--batch-size",
        str(settings.batch_size),
        "--ubatch-size",
        str(settings.ubatch_size),
        "--n-gpu-layers",
        str(settings.gpu_layers),
        "--flash-attn",
        "on" if settings.flash_attention else "off",
    ]


def parse_llama_perplexity_output(stdout: str, stderr: str, returncode: int) -> PerplexityResult:
    combined = stdout + "\n" + stderr
    matches = re.findall(r"(?:PPL|perplexity)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", combined, re.I)
    perplexity = float(matches[-1]) if matches else None
    ok = returncode == 0 and perplexity is not None
    return PerplexityResult(
        ok=ok,
        perplexity=perplexity,
        stdout=stdout[-8000:],
        stderr=stderr[-8000:],
        returncode=returncode,
        failure="none" if ok else "no_perplexity",
    )


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
        f"- Profile: `{settings.profile_name}`",
        f"- Parallel: `{settings.parallel}`",
        f"- GPU layers: `{settings.gpu_layers}`",
        f"- Batch / ubatch: `{settings.batch_size}` / `{settings.ubatch_size}`",
        f"- Flash attention: `{settings.flash_attention}`",
        f"- Unified KV cache: `{settings.kv_unified}`",
        f"- Extra llama-server args: `{list(settings.extra_server_args)}`",
        "",
        "## Best Result",
        "",
        f"- Generation tokens/sec: `{result.generation_tokens_per_second}`",
        f"- Prompt tokens/sec: `{result.prompt_tokens_per_second}`",
        f"- Flag profile: `{result.flag_profile or settings.profile_name}`",
        f"- SimpleBench score: `{result.simple_bench_score}`",
        f"- SimpleBench accuracy: `{result.simple_bench_accuracy}`",
        f"- SimpleBench receipt: `{result.simple_bench_receipt or 'none'}`",
        f"- SimpleBench failure: `{result.simple_bench_failure or 'none'}`",
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
        "- Itemized report: `itemized-report.md`",
        "- Browser report: `report.html`",
        "- Machine report: `report.json`",
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
