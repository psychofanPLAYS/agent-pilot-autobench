from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable

from gguf_limit_bench.receipts import RunReceipt
from gguf_limit_bench.run_config import RunStatus
from gguf_limit_bench.telemetry import classify_failure, sample_telemetry


@dataclass(frozen=True)
class AutoresearchSettings:
    context_size: int = 0
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

    def score(self) -> float:
        if not self.ok:
            return -10_000.0
        context_bonus = min(self.context_size, 131_072) / 4096.0
        ttft_penalty = (self.ttft_ms or 0.0) / 1000.0
        return (
            self.generation_tokens_per_second
            + self.prompt_tokens_per_second / 100.0
            + context_bonus
            + self.workflow_score
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
    ) -> None:
        self.model = model
        self.runs_root = runs_root
        self.attempt_runner = attempt_runner
        self.budget_seconds = budget_seconds
        self.parallel_max = parallel_max
        self.max_attempts = max_attempts
        self.learner = learner

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
            settings = suggestion.settings if suggestion is not None else self._candidate(best_settings, attempt_index)
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
            last_result = result
            if suggestion is not None:
                self.learner.tell(suggestion, result)
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
                    "accepted": best_result == result,
                    "telemetry": sample_telemetry().to_dict(),
                },
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
        return receipt

    def _candidate(self, best: AutoresearchSettings, attempt_index: int) -> AutoresearchSettings:
        if attempt_index == 0:
            return best
        mutation = (attempt_index - 1) % 4
        if mutation == 0:
            next_context = 4096 if best.context_size == 0 else best.context_size * 2
            return replace(best, context_size=min(next_context, 131_072), kv_unified=True)
        if mutation == 1:
            return replace(best, batch_size=max(512, best.batch_size // 2), kv_unified=True)
        if mutation == 2:
            return replace(best, parallel=min(self.parallel_max, best.parallel + 1), kv_unified=True)
        return replace(best, ubatch_size=max(128, best.ubatch_size // 2), kv_unified=True)


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
        if n_gen > 0:
            generation_speed = max(generation_speed, speed)
        elif n_prompt > 0:
            prompt_speed = max(prompt_speed, speed)

    failure = classify_failure(stderr + "\n" + stdout)
    return AttemptResult(
        ok=returncode == 0 and generation_speed > 0.0,
        generation_tokens_per_second=generation_speed,
        prompt_tokens_per_second=prompt_speed,
        ttft_ms=None,
        context_size=context_size,
        failure=failure,
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
    lines = [
        f"# {model.name}",
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


def _safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)[:80]


def _status_for_result(result: AttemptResult) -> str:
    if not result.ok:
        if result.failure in {"model_load", "gpu_oom", "memory_allocation", "crash"}:
            return RunStatus.FAILED.value
        return RunStatus.PARTIAL.value
    if result.generation_tokens_per_second < 20.0:
        return RunStatus.CHAMPION_RETEST_NEEDED.value
    return RunStatus.CANDIDATE.value
