from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import re
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings
from gguf_limit_bench.benchmark_suite import (
    BenchmarkSuitePlan,
    BenchmarkSuiteRun,
    run_benchmark_suite,
)
from gguf_limit_bench.hf_recommended_settings import sampling_payload_from_server_args
from gguf_limit_bench.server_probe import (
    _free_port,
    _stop_process,
    _wait_until_ready,
    build_llama_server_command,
    iter_llama_completion_stream_events,
    process_group_kwargs,
)
from gguf_limit_bench.simple_bench import (
    DEFAULT_SIMPLE_BENCH_PATH,
    DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
    SimpleBenchQuestion,
    SimpleBenchQuestionResult,
    combine_simple_bench_results,
    extract_final_answer,
    load_simple_bench_questions,
    load_simple_bench_system_prompt,
    simple_bench_prompt,
)
from gguf_limit_bench.telemetry import classify_failure
from gguf_limit_bench.telemetry import sample_telemetry


@dataclass(frozen=True)
class CompletionMeasurement:
    ok: bool
    response: str
    ttft_ms: float | None
    tokens_per_second: float
    generated_tokens: int
    output_chars: int
    prompt_tokens_per_second: float = 0.0
    failure: str = "none"


class LlamaServerSimpleBenchAttemptRunner:
    def __init__(
        self,
        *,
        llama_server: Path,
        model: Path,
        benchmark_path: Path = DEFAULT_SIMPLE_BENCH_PATH,
        system_prompt_path: Path = DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
        timeout_seconds: int = 600,
        max_tokens: int = 4096,
        host: str = "127.0.0.1",
        port: int | None = None,
        benchmark_suite_plan: BenchmarkSuitePlan | None = None,
        runs_root: Path | None = None,
    ) -> None:
        self.llama_server = llama_server
        self.model = model
        self.benchmark_path = benchmark_path
        self.system_prompt_path = system_prompt_path
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.host = host
        self.port = port
        self.benchmark_suite_plan = benchmark_suite_plan
        self.runs_root = runs_root
        self.receipt_path: Path | None = None
        self.questions = load_simple_bench_questions(benchmark_path)
        self.system_prompt = load_simple_bench_system_prompt(system_prompt_path)

    def set_receipt_path(self, receipt_path: Path) -> None:
        self.receipt_path = receipt_path

    def set_timeout_seconds(self, timeout_seconds: int) -> None:
        self.timeout_seconds = max(1, timeout_seconds)

    def __call__(self, settings: AutoresearchSettings) -> AttemptResult:
        started = time.perf_counter()
        deadline = time.monotonic() + self.timeout_seconds
        port = self.port or _free_port()
        command = build_llama_server_command(
            llama_server=self.llama_server,
            model=self.model,
            settings=settings,
            host=self.host,
            port=port,
        )
        attempt_dir = self._attempt_dir(settings)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _write_launch_receipt(attempt_dir, command)
        stdout_log = (attempt_dir / "server.stdout.log").open("w", encoding="utf-8")
        stderr_log = (attempt_dir / "server.stderr.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_log,
                stderr=stderr_log,
                text=True,
                **process_group_kwargs(),
            )
        except OSError as exc:
            stdout_log.close()
            stderr_log.close()
            return self._failed_result(
                settings=settings,
                command=command,
                attempt_dir=attempt_dir,
                failure=f"server_start_error: {exc}",
                returncode=1,
                stdout="",
                stderr=str(exc),
            )

        base_url = f"http://{self.host}:{port}"
        question_results: list[SimpleBenchQuestionResult] = []
        partial_failure: str | None = None
        stdout = ""
        try:
            ready_at = _wait_until_ready(base_url, process, timeout_seconds=self.timeout_seconds)
            _append_receipt_event(
                self.receipt_path,
                "llama_server_ready",
                {"base_url": base_url, "telemetry": sample_telemetry().to_dict()},
            )
            sampling = sampling_payload_from_server_args(settings.extra_server_args)
            _warmup_server(
                base_url,
                self.system_prompt,
                timeout_seconds=self.timeout_seconds,
                sampling=sampling,
            )
            _append_receipt_event(
                self.receipt_path,
                "llama_server_warmup_finished",
                {"base_url": base_url, "telemetry": sample_telemetry().to_dict()},
            )
            for index, question in enumerate(self.questions, start=1):
                measurement = measure_simple_bench_completion(
                    base_url=base_url,
                    question=question,
                    system_prompt=self.system_prompt,
                    max_tokens=self.max_tokens,
                    timeout_seconds=_remaining_timeout_seconds(deadline),
                    sampling=sampling,
                )
                predicted = extract_final_answer(measurement.response)
                question_result = SimpleBenchQuestionResult(
                    question_id=question.question_id,
                    expected_answer=question.answer,
                    predicted_answer=predicted,
                    correct=predicted == question.answer,
                    ttft_ms=measurement.ttft_ms,
                    tokens_per_second=measurement.tokens_per_second,
                    generated_tokens=measurement.generated_tokens,
                    output_chars=measurement.output_chars,
                    prompt_chars=len(simple_bench_prompt(self.system_prompt, question)),
                    response=measurement.response,
                    prompt_tokens_per_second=measurement.prompt_tokens_per_second,
                    failure=measurement.failure,
                    outcome=(
                        "correct"
                        if predicted == question.answer
                        else ("incomplete" if predicted is None else "wrong")
                    ),
                )
                question_results.append(question_result)
                _append_jsonl(attempt_dir / "transcript.jsonl", question_result.to_dict())
                if not measurement.ok:
                    break
            batch = combine_simple_bench_results(question_results)
            server_ready_ms = (ready_at - started) * 1000.0
            returncode = process.poll()
            result = AttemptResult(
                ok=batch.ok,
                generation_tokens_per_second=batch.median_tps,
                prompt_tokens_per_second=batch.median_prompt_tps,
                ttft_ms=batch.median_ttft_ms,
                context_size=settings.context_size,
                failure="none" if batch.ok else batch.failure,
                stdout="",
                stderr="",
                returncode=returncode if returncode is not None else 0,
                serving_ttft_ms=batch.median_ttft_ms,
                serving_tokens_per_second=batch.median_tps,
                serving_server_ready_ms=server_ready_ms,
                serving_question_results=[result.to_dict() for result in question_results],
                flag_profile=settings.profile_name,
                launch_command=command,
                simple_bench_score=batch.score,
                simple_bench_accuracy=batch.accuracy,
                simple_bench_receipt=str(attempt_dir),
                simple_bench_failure=None if batch.ok else batch.failure,
                completed_questions=len(question_results),
                attempted_questions=len(self.questions),
            )
            _append_receipt_event(
                self.receipt_path,
                "simple_bench_finished",
                {
                    "base_url": base_url,
                    "ok": batch.ok,
                    "telemetry": sample_telemetry().to_dict(),
                },
            )
            if batch.ok and self.benchmark_suite_plan is not None:
                result = self._with_benchmark_suite(
                    result=result,
                    settings=settings,
                    base_url=base_url,
                    timeout_seconds=max(0.0, deadline - time.monotonic()),
                )
            return result
        except TimeoutError as exc:
            stdout, stderr = _flush_and_read_logs(
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                attempt_dir=attempt_dir,
            )
            if question_results:
                partial_failure = "budget_exhausted_partial"
                batch = combine_simple_bench_results(question_results)
                return AttemptResult(
                    ok=False,
                    generation_tokens_per_second=batch.median_tps,
                    prompt_tokens_per_second=batch.median_prompt_tps,
                    ttft_ms=batch.median_ttft_ms,
                    context_size=settings.context_size,
                    failure=partial_failure,
                    stdout=stdout[-8000:],
                    stderr=f"{stderr}\n{exc}"[-8000:],
                    returncode=process.returncode or 124,
                    serving_ttft_ms=batch.median_ttft_ms,
                    serving_tokens_per_second=batch.median_tps,
                    serving_question_results=[result.to_dict() for result in question_results],
                    flag_profile=settings.profile_name,
                    launch_command=command,
                    simple_bench_score=batch.score,
                    simple_bench_accuracy=batch.accuracy,
                    simple_bench_receipt=str(attempt_dir),
                    simple_bench_failure=partial_failure,
                    completed_questions=len(question_results),
                    attempted_questions=len(self.questions),
                )
            failure = str(exc)
            return self._failed_result(
                settings=settings,
                command=command,
                attempt_dir=attempt_dir,
                failure=failure,
                returncode=process.returncode or 124,
                stdout=stdout,
                stderr=f"{stderr}\n{exc}",
            )
        except (OSError, URLError) as exc:
            stdout, stderr = _flush_and_read_logs(
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                attempt_dir=attempt_dir,
            )
            failure = f"server_error: {exc}"
            return self._failed_result(
                settings=settings,
                command=command,
                attempt_dir=attempt_dir,
                failure=failure,
                returncode=process.returncode or 1,
                stdout=stdout,
                stderr=f"{stderr}\n{exc}",
            )
        finally:
            _append_receipt_event(
                self.receipt_path,
                "llama_server_stopping",
                {"base_url": base_url, "telemetry": sample_telemetry().to_dict()},
            )
            _stop_process(process)
            stdout_log.close()
            stderr_log.close()
            warning_count = _write_short_logs(
                attempt_dir=attempt_dir,
                settings=settings,
                returncode=process.returncode,
            )
            if question_results:
                summary = combine_simple_bench_results(question_results).to_dict()
                summary["settings"] = settings.to_dict()
                summary["command"] = command
                summary["status"] = "partial" if partial_failure else "complete"
                summary["failure"] = partial_failure or summary.get("failure", "none")
                summary["completed_questions"] = len(question_results)
                summary["attempted_questions"] = len(self.questions)
                summary["warning_count"] = warning_count
                summary["warnings_log"] = str(attempt_dir / "warnings.log")
                summary["server_tail_log"] = str(attempt_dir / "server-tail.log")
                (attempt_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
            else:
                _add_short_log_metadata(attempt_dir, warning_count)

    def _attempt_dir(self, settings: AutoresearchSettings) -> Path:
        root = self.receipt_path or Path("_runs")
        safe_profile = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in settings.profile_name
        )
        return root / f"simplebench-{safe_profile}"

    def _failed_result(
        self,
        *,
        settings: AutoresearchSettings,
        command: list[str],
        attempt_dir: Path,
        failure: str,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> AttemptResult:
        classified = classify_failure(f"{failure}\n{stdout}\n{stderr}")
        normalized_failure = classified if classified != "unknown" else failure
        summary = {
            "ok": False,
            "failure": normalized_failure,
            "settings": settings.to_dict(),
            "command": command,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
        (attempt_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return AttemptResult(
            ok=False,
            generation_tokens_per_second=0.0,
            prompt_tokens_per_second=0.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure=normalized_failure,
            stdout=stdout[-8000:],
            stderr=stderr[-8000:],
            returncode=returncode,
            flag_profile=settings.profile_name,
            launch_command=command,
            simple_bench_score=None,
            simple_bench_accuracy=None,
            simple_bench_receipt=str(attempt_dir),
            simple_bench_failure=normalized_failure,
            completed_questions=0,
            attempted_questions=len(self.questions),
        )

    def _with_benchmark_suite(
        self,
        *,
        result: AttemptResult,
        settings: AutoresearchSettings,
        base_url: str,
        timeout_seconds: float,
    ) -> AttemptResult:
        assert self.benchmark_suite_plan is not None
        runs_root = self.runs_root or self.receipt_path or Path("_runs")
        if timeout_seconds <= 0:
            return replace(
                result,
                ok=False,
                failure="benchmark_suite_failed",
                benchmark_suite_ok=False,
                benchmark_suite_failure="benchmark_suite_budget_exhausted",
            )
        suite_settings = {
            **self.benchmark_suite_plan.settings,
            **settings.to_dict(),
            "gguf_model_path": str(self.model),
            "score_contract": "agent_bench_score",
            "base_url": base_url,
        }
        plan = BenchmarkSuitePlan(
            model=str(self.model),
            context=settings.context_size,
            settings=suite_settings,
            tasks=self.benchmark_suite_plan.tasks,
        )
        _append_receipt_event(
            self.receipt_path,
            "benchmark_suite_started_on_owned_server",
            {"base_url": base_url, "telemetry": sample_telemetry().to_dict()},
        )
        suite_run = run_benchmark_suite(plan, runs_root, timeout_seconds=timeout_seconds)
        _append_receipt_event(
            self.receipt_path,
            "benchmark_suite_finished_on_owned_server",
            {
                "base_url": base_url,
                "ok": suite_run.ok,
                "telemetry": sample_telemetry().to_dict(),
            },
        )
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


def _warmup_server(
    base_url: str,
    system_prompt: str,
    timeout_seconds: int,
    sampling: dict[str, object] | None = None,
) -> None:
    """Best-effort warmup so the first scored question is not penalized by cold
    CUDA kernels and allocator warmup. Any failure is ignored on purpose."""
    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Warmup only. Reply with OK."},
            ],
            "stream": False,
            "max_tokens": 8,
            **(sampling or {"temperature": 1.0}),
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=min(max(1, timeout_seconds), 120)) as response:
            response.read()
    except (OSError, URLError):
        return


def measure_simple_bench_completion(
    *,
    base_url: str,
    question: SimpleBenchQuestion,
    system_prompt: str,
    max_tokens: int,
    timeout_seconds: int,
    sampling: dict[str, object] | None = None,
) -> CompletionMeasurement:
    body: dict[str, object] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.prompt},
        ],
        "stream": True,
        **(sampling or {"temperature": 1.0}),
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    else:
        # n_predict = -1: let a reasoning model think to its own stop, bounded
        # only by the request timeout (see UNLIMITED_THINKING in pack_runner).
        body["n_predict"] = -1
    payload = json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    generated_tokens = 0
    fallback_chunks = 0
    content_parts: list[str] = []
    server_tokens_per_second: float | None = None
    server_prompt_tokens_per_second: float | None = None
    usage_completion_tokens: int | None = None
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for event in iter_llama_completion_stream_events(response):
                timings = event.get("timings")
                if isinstance(timings, dict) and timings.get("predicted_per_second") is not None:
                    server_tokens_per_second = float(timings["predicted_per_second"])
                if isinstance(timings, dict) and timings.get("prompt_per_second") is not None:
                    server_prompt_tokens_per_second = float(timings["prompt_per_second"])
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    usage_completion_tokens = int(usage["completion_tokens"])
                content_parts_for_event: list[str] = []
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    for field_name in ("reasoning_content", "content"):
                        value = delta.get(field_name, "")
                        if isinstance(value, str):
                            content_parts_for_event.append(value)
                content = "".join(content_parts_for_event)
                token_count = _event_token_count(event)
                if token_count <= 0 and not content:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                generated_tokens += token_count
                fallback_chunks += 1
                content_parts.append(content)
    except (OSError, URLError) as exc:
        return CompletionMeasurement(
            ok=False,
            response="",
            ttft_ms=None,
            tokens_per_second=0.0,
            generated_tokens=0,
            output_chars=0,
            failure=f"request_error: {exc}",
        )
    finished = time.perf_counter()
    response_text = "".join(content_parts)
    if first_token_at is None:
        return CompletionMeasurement(
            ok=False,
            response=response_text,
            ttft_ms=None,
            tokens_per_second=0.0,
            generated_tokens=0,
            output_chars=len(response_text),
            failure="no_streamed_token",
        )
    measured_tokens = usage_completion_tokens or generated_tokens or fallback_chunks
    generation_seconds = max(finished - first_token_at, 0.001)
    return CompletionMeasurement(
        ok=True,
        response=response_text,
        ttft_ms=(first_token_at - started) * 1000.0,
        tokens_per_second=server_tokens_per_second or measured_tokens / generation_seconds,
        generated_tokens=measured_tokens,
        output_chars=len(response_text),
        prompt_tokens_per_second=server_prompt_tokens_per_second or 0.0,
    )


def _event_token_count(event: dict) -> int:
    tokens = event.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return 0


def _benchmark_suite_failure(suite_run: BenchmarkSuiteRun) -> str:
    failures = [
        f"{item.id}:{item.failure_class}"
        for item in suite_run.results
        if not item.ok or item.failure_class != "none"
    ]
    return ";".join(failures) if failures else "benchmark_suite_failed"


def _remaining_timeout_seconds(deadline: float) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("SimpleBench attempt budget exhausted")
    return max(1, math.ceil(remaining))


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _append_receipt_event(receipt_path: Path | None, event_type: str, data: dict) -> None:
    if receipt_path is None:
        return
    receipt_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "type": event_type,
        "data": data,
    }
    _append_jsonl(receipt_path / "events.jsonl", payload)


def _write_short_logs(
    *,
    attempt_dir: Path,
    settings: AutoresearchSettings,
    returncode: int | None,
) -> int:
    stdout_lines = _read_log_lines(attempt_dir / "server.stdout.log")
    stderr_lines = _read_log_lines(attempt_dir / "server.stderr.log")
    combined = stdout_lines + stderr_lines
    warning_pattern = re.compile(
        r"(?:\b(?:warn(?:ing)?|error|failed|failure|invalid|unsupported|unknown argument|"
        r"out of memory|oom|exception|abort)\b|^\S+\s+[WE]\s+)",
        flags=re.IGNORECASE,
    )
    warning_lines = [line for line in combined if warning_pattern.search(line)][-80:]
    warning_text = warning_lines or ["No warning or error lines detected."]
    (attempt_dir / "warnings.log").write_text("\n".join(warning_text) + "\n", encoding="utf-8")

    tail_header = [
        f"profile={settings.profile_name}",
        f"returncode={returncode}",
        f"warning_count={len(warning_lines)}",
        "",
    ]
    tail_lines = combined[-80:] or ["No server log lines captured."]
    (attempt_dir / "server-tail.log").write_text(
        "\n".join(tail_header + tail_lines) + "\n",
        encoding="utf-8",
    )
    return len(warning_lines)


def _read_log_lines(path: Path) -> list[str]:
    try:
        return [
            line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line
        ]
    except OSError:
        return []


def _flush_and_read_logs(
    *,
    stdout_log,
    stderr_log,
    attempt_dir: Path,
) -> tuple[str, str]:
    stdout_log.flush()
    stderr_log.flush()
    stdout_path = attempt_dir / "server.stdout.log"
    stderr_path = attempt_dir / "server.stderr.log"
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return stdout, stderr


def _add_short_log_metadata(attempt_dir: Path, warning_count: int) -> None:
    summary_path = attempt_dir / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    payload["warning_count"] = warning_count
    payload["warnings_log"] = str(attempt_dir / "warnings.log")
    payload["server_tail_log"] = str(attempt_dir / "server-tail.log")
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def _write_launch_receipt(attempt_dir: Path, command: list[str]) -> None:
    """Store exact argv as data, never as a double-clickable command script."""
    (attempt_dir / "launch-command.json").write_text(
        json.dumps(command, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
