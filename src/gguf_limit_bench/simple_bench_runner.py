from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings
from gguf_limit_bench.server_probe import (
    _free_port,
    _stop_process,
    _wait_until_ready,
    build_llama_server_command,
    iter_llama_completion_stream_events,
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


@dataclass(frozen=True)
class CompletionMeasurement:
    ok: bool
    response: str
    ttft_ms: float | None
    tokens_per_second: float
    generated_tokens: int
    output_chars: int
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
    ) -> None:
        self.llama_server = llama_server
        self.model = model
        self.benchmark_path = benchmark_path
        self.system_prompt_path = system_prompt_path
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.host = host
        self.port = port
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
        stdout = ""
        try:
            ready_at = _wait_until_ready(base_url, process, timeout_seconds=self.timeout_seconds)
            for index, question in enumerate(self.questions, start=1):
                measurement = measure_simple_bench_completion(
                    base_url=base_url,
                    question=question,
                    system_prompt=self.system_prompt,
                    max_tokens=self.max_tokens,
                    timeout_seconds=_remaining_timeout_seconds(deadline),
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
                    failure=measurement.failure,
                )
                question_results.append(question_result)
                _append_jsonl(attempt_dir / "transcript.jsonl", question_result.to_dict())
                if not measurement.ok:
                    break
            batch = combine_simple_bench_results(question_results)
            server_ready_ms = (ready_at - started) * 1000.0
            returncode = process.poll()
            return AttemptResult(
                ok=batch.ok,
                generation_tokens_per_second=batch.median_tps,
                prompt_tokens_per_second=0.0,
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
            )
        except TimeoutError as exc:
            stdout, stderr = _flush_and_read_logs(
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                attempt_dir=attempt_dir,
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
        )


def measure_simple_bench_completion(
    *,
    base_url: str,
    question: SimpleBenchQuestion,
    system_prompt: str,
    max_tokens: int,
    timeout_seconds: int,
) -> CompletionMeasurement:
    payload = json.dumps(
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question.prompt},
            ],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode("utf-8")
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
    usage_completion_tokens: int | None = None
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for event in iter_llama_completion_stream_events(response):
                timings = event.get("timings")
                if isinstance(timings, dict) and timings.get("predicted_per_second") is not None:
                    server_tokens_per_second = float(timings["predicted_per_second"])
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
    )


def _event_token_count(event: dict) -> int:
    tokens = event.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return 0


def _remaining_timeout_seconds(deadline: float) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("SimpleBench attempt budget exhausted")
    return max(1, math.ceil(remaining))


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
        r"\b(?:warn(?:ing)?|error|failed|failure|invalid|unsupported|unknown argument|"
        r"out of memory|oom|exception|abort)\b",
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
