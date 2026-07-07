from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import BinaryIO
from urllib.error import URLError
from urllib.request import Request, urlopen

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.oom import is_oom_failure

_CREATE_NEW_PROCESS_GROUP = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))


@dataclass(frozen=True)
class ServingProbePrompt:
    id: str
    prompt: str


_AGENT_PROMPT_PREFIX = """You are a local coding assistant running on a workstation.
Use this request as an agent-latency benchmark. The prompt intentionally includes
planning context, tool context, and user intent so time-to-first-token reflects a
real first question better than a tiny toy prompt.

Project: local GGUF benchmark cockpit.
Goal: choose settings that make a local model useful for coding-agent work.
Important metrics: time to first token, generation tokens per second, prompt-cache
reuse, context length, and whether the model stays responsive after the first
question.
"""


DEFAULT_AGENT_TTFT_QUESTIONS = (
    ServingProbePrompt(
        id="latency_definition",
        prompt=_AGENT_PROMPT_PREFIX
        + "\nTask: Reply with exactly one concise sentence explaining why cold TTFT and warm TTFT should be measured separately.\n",
    ),
    ServingProbePrompt(
        id="tool_plan",
        prompt=_AGENT_PROMPT_PREFIX
        + "\nTask: Reply with a three-step plan for safely testing a local model behind a coding agent.\n",
    ),
    ServingProbePrompt(
        id="json_receipt",
        prompt=_AGENT_PROMPT_PREFIX
        + '\nTask: Return compact JSON with keys "metric", "risk", and "next_check" for a TTFT benchmark receipt.\n',
    ),
    ServingProbePrompt(
        id="failure_triage",
        prompt=_AGENT_PROMPT_PREFIX
        + "\nTask: Name the most likely cause when first-token latency is high but later turns are fast.\n",
    ),
    ServingProbePrompt(
        id="deployment_choice",
        prompt=_AGENT_PROMPT_PREFIX
        + "\nTask: In one sentence, choose between a fast 4K context and a slower 64K context for an agent pilot, and say why.\n",
    ),
)


DEFAULT_AGENT_TTFT_PROMPT = DEFAULT_AGENT_TTFT_QUESTIONS[0].prompt
HIGH_CONTEXT_QUESTION_THRESHOLD = 32_768


@dataclass(frozen=True)
class ServingProbeResult:
    ok: bool
    ttft_ms: float | None
    tokens_per_second: float
    output_chars: int
    generated_tokens: int
    failure: str
    warm_ttft_ms: float | None = None
    warm_tokens_per_second: float | None = None
    warmup_penalty_ms: float | None = None
    server_ready_ms: float | None = None
    cold_start_to_first_token_ms: float | None = None
    request_count: int = 1
    ttft_samples_ms: list[float] = field(default_factory=list)
    tokens_per_second_samples: list[float] = field(default_factory=list)
    tokens_cached_samples: list[int | None] = field(default_factory=list)
    tokens_evaluated_samples: list[int | None] = field(default_factory=list)
    question_results: list[dict] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_llama_server_command(
    *,
    llama_server: Path,
    model: Path,
    settings: AutoresearchSettings,
    host: str,
    port: int,
) -> list[str]:
    from gguf_limit_bench.flag_ladder import llama_server_args_for_settings

    extra_options = {
        arg.split("=", 1)[0] for arg in settings.extra_server_args if arg.startswith("-")
    }
    command = [
        str(llama_server),
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(settings.context_size or 4096),
        "--batch-size",
        str(settings.batch_size),
        "--ubatch-size",
        str(settings.ubatch_size),
        "--parallel",
        str(settings.parallel),
        "--metrics",
        "--slots",
        "--no-webui",
    ]
    if "--gpu-layers" not in extra_options:
        command.extend(["--gpu-layers", str(settings.gpu_layers)])
    if "--flash-attn" not in extra_options:
        command.extend(["--flash-attn", "on" if settings.flash_attention else "off"])
    command.extend(llama_server_args_for_settings(settings))
    return command


def probe_llama_server_ttft(
    *,
    llama_server: Path,
    model: Path,
    settings: AutoresearchSettings,
    prompt: str = DEFAULT_AGENT_TTFT_PROMPT,
    max_tokens: int = 64,
    samples: int = 0,
    cache_prompt: bool = True,
    timeout_seconds: int = 180,
) -> ServingProbeResult:
    host = "127.0.0.1"
    port = _free_port()
    command = build_llama_server_command(
        llama_server=llama_server,
        model=model,
        settings=settings,
        host=host,
        port=port,
    )
    server_started = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **process_group_kwargs(),
        )
    except OSError as exc:
        return ServingProbeResult(
            ok=False,
            ttft_ms=None,
            tokens_per_second=0.0,
            output_chars=0,
            generated_tokens=0,
            failure=f"server_start_error: {exc}",
        )
    base_url = f"http://{host}:{port}"
    try:
        ready_at = _wait_until_ready(base_url, process, timeout_seconds=timeout_seconds)
        measurements: list[ServingProbeResult] = []
        prompts = serving_prompts_for_context(
            context_size=settings.context_size or 4096,
            fallback_prompt=prompt,
            samples=samples,
        )
        for index, serving_prompt in enumerate(prompts, start=1):
            measurement = measure_llama_completion_stream_ttft(
                base_url=base_url,
                prompt=serving_prompt.prompt,
                max_tokens=max_tokens,
                cache_prompt=cache_prompt,
                question_id=serving_prompt.id,
                question_index=index,
                timeout_seconds=timeout_seconds,
            )
            if not measurement.ok:
                return measurement
            measurements.append(measurement)
        result = _combine_measurements(measurements)
        server_ready_ms = (ready_at - server_started) * 1000.0
        cold_start_to_first_token_ms = (
            server_ready_ms + result.ttft_ms if result.ttft_ms is not None else None
        )
        return replace(
            result,
            server_ready_ms=server_ready_ms,
            cold_start_to_first_token_ms=cold_start_to_first_token_ms,
        )
    except TimeoutError as exc:
        return _failed_probe("timeout", str(exc), process)
    except OSError as exc:
        return _failed_probe("server_error", str(exc), process)
    finally:
        _stop_process(process)


def measure_llama_completion_stream_ttft(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int,
    cache_prompt: bool = True,
    question_id: str = "single_prompt",
    question_index: int = 1,
    timeout_seconds: int,
) -> ServingProbeResult:
    payload = json.dumps(
        {
            "prompt": prompt,
            "stream": True,
            "n_predict": max_tokens,
            "temperature": 0,
            "cache_prompt": cache_prompt,
            "return_tokens": True,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    output_chars = 0
    generated_tokens = 0
    fallback_chunks = 0
    server_tokens_per_second: float | None = None
    tokens_cached: int | None = None
    tokens_evaluated: int | None = None
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for event in iter_llama_completion_stream_events(response):
                tokens_cached = _optional_int(event.get("tokens_cached"), tokens_cached)
                tokens_evaluated = _optional_int(event.get("tokens_evaluated"), tokens_evaluated)
                timings = event.get("timings")
                if isinstance(timings, dict):
                    predicted_per_second = timings.get("predicted_per_second")
                    if predicted_per_second is not None:
                        server_tokens_per_second = float(predicted_per_second)
                token_count = _event_token_count(event)
                content = event.get("content")
                if not content:
                    content = ""
                has_token = token_count > 0 or bool(content)
                if not has_token:
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                generated_tokens += token_count
                fallback_chunks += 1
                output_chars += len(content) if isinstance(content, str) else 0
    except URLError as exc:
        return ServingProbeResult(
            ok=False,
            ttft_ms=None,
            tokens_per_second=0.0,
            output_chars=0,
            generated_tokens=0,
            failure=f"request_error: {exc}",
        )

    finished = time.perf_counter()
    if first_token_at is None:
        return ServingProbeResult(
            ok=False,
            ttft_ms=None,
            tokens_per_second=0.0,
            output_chars=0,
            generated_tokens=0,
            failure="no_streamed_token",
        )
    generation_seconds = max(finished - first_token_at, 0.001)
    measured_tokens = generated_tokens or fallback_chunks
    ttft_ms = (first_token_at - started) * 1000.0
    tokens_per_second = server_tokens_per_second or measured_tokens / generation_seconds
    return ServingProbeResult(
        ok=True,
        ttft_ms=ttft_ms,
        tokens_per_second=tokens_per_second,
        output_chars=output_chars,
        generated_tokens=measured_tokens,
        failure="none",
        warm_ttft_ms=None,
        warm_tokens_per_second=None,
        warmup_penalty_ms=None,
        request_count=1,
        ttft_samples_ms=[ttft_ms],
        tokens_per_second_samples=[tokens_per_second],
        tokens_cached_samples=[tokens_cached],
        tokens_evaluated_samples=[tokens_evaluated],
        question_results=[
            {
                "question_index": question_index,
                "question_id": question_id,
                "prompt_chars": len(prompt),
                "ttft_ms": ttft_ms,
                "tokens_per_second": tokens_per_second,
                "generated_tokens": measured_tokens,
                "output_chars": output_chars,
                "tokens_cached": tokens_cached,
                "tokens_evaluated": tokens_evaluated,
                "is_cold": question_index == 1,
            }
        ],
    )


def serving_prompts_for_context(
    *,
    context_size: int,
    fallback_prompt: str,
    samples: int,
) -> list[ServingProbePrompt]:
    if fallback_prompt != DEFAULT_AGENT_TTFT_PROMPT:
        return [
            ServingProbePrompt(id="custom_prompt", prompt=fallback_prompt)
            for _ in range(max(1, samples))
        ]
    question_count = max(_question_count_for_context(context_size), samples)
    prompts: list[ServingProbePrompt] = []
    while len(prompts) < question_count:
        prompts.extend(DEFAULT_AGENT_TTFT_QUESTIONS)
    return prompts[:question_count]


def _question_count_for_context(context_size: int) -> int:
    if context_size >= HIGH_CONTEXT_QUESTION_THRESHOLD:
        return 5
    if context_size >= 16_384:
        return 3
    if context_size >= 8_192:
        return 2
    return 1


def iter_llama_completion_stream_events(stream: BinaryIO):
    for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def iter_openai_stream_content(stream: BinaryIO):
    for raw_line in stream:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in payload.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                yield content


def _event_token_count(event: dict) -> int:
    tokens = event.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return 0


def _optional_int(value, fallback: int | None) -> int | None:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _combine_measurements(measurements: list[ServingProbeResult]) -> ServingProbeResult:
    first = measurements[0]
    warm_measurements = measurements[1:]
    warm_ttft = _mean([item.ttft_ms for item in warm_measurements if item.ttft_ms is not None])
    warm_tps = _mean(_reliable_serving_tps(warm_measurements))
    warmup_penalty = None
    if first.ttft_ms is not None and warm_ttft is not None:
        warmup_penalty = max(0.0, first.ttft_ms - warm_ttft)
    return ServingProbeResult(
        ok=True,
        ttft_ms=first.ttft_ms,
        tokens_per_second=_mean(_reliable_serving_tps(measurements)) or 0.0,
        output_chars=sum(item.output_chars for item in measurements),
        generated_tokens=sum(item.generated_tokens for item in measurements),
        failure="none",
        warm_ttft_ms=warm_ttft,
        warm_tokens_per_second=warm_tps,
        warmup_penalty_ms=warmup_penalty,
        request_count=len(measurements),
        ttft_samples_ms=[item.ttft_ms for item in measurements if item.ttft_ms is not None],
        tokens_per_second_samples=[item.tokens_per_second for item in measurements],
        tokens_cached_samples=[
            item.tokens_cached_samples[-1] if item.tokens_cached_samples else None
            for item in measurements
        ],
        tokens_evaluated_samples=[
            item.tokens_evaluated_samples[-1] if item.tokens_evaluated_samples else None
            for item in measurements
        ],
        question_results=[question for item in measurements for question in item.question_results],
    )


def _reliable_serving_tps(measurements: list[ServingProbeResult]) -> list[float]:
    return [
        item.tokens_per_second
        for item in measurements
        if item.generated_tokens >= 8 and item.output_chars > 0
    ]


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _wait_until_ready(base_url: str, process: subprocess.Popen, timeout_seconds: int) -> float:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise TimeoutError(f"llama-server exited early with code {process.returncode}")
        try:
            with urlopen(f"{base_url}/health", timeout=2) as response:
                if 200 <= response.status < 500:
                    return time.perf_counter()
        except (OSError, URLError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become ready: {last_error}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def process_group_kwargs() -> dict:
    """Popen kwargs that place the child in its own process group / session.

    A llama-server launched this way can be killed as a tree without taking down
    the parent, and never lingers as an orphan after a hard kill of the engine.
    """
    if _is_windows():
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(process: subprocess.Popen) -> None:
    """Terminate *process* and every child it spawned. Best-effort, cross-platform.

    On Windows this uses ``taskkill /T /F`` so GPU-holding grandchildren are
    reaped; on POSIX it signals the whole process group. Safe to call on a
    process that has already exited (no-op).
    """
    if process.poll() is not None:
        return
    if _is_windows():
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        if callable(killpg) and callable(getpgid):
            killpg(getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, PermissionError):
        pass
    except subprocess.TimeoutExpired:
        try:
            killpg = getattr(os, "killpg", None)
            getpgid = getattr(os, "getpgid", None)
            if callable(killpg) and callable(getpgid):
                killpg(getpgid(process.pid), getattr(signal, "SIGKILL", signal.SIGTERM))
            else:
                process.kill()
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _stop_process(process: subprocess.Popen) -> None:
    kill_process_tree(process)


def _is_windows() -> bool:
    return os.name == "nt"


def _failed_probe(failure: str, detail: str, process: subprocess.Popen) -> ServingProbeResult:
    stdout = ""
    stderr = detail
    if process.poll() is not None:
        try:
            out, err = process.communicate(timeout=2)
            stdout = out or ""
            stderr = err or detail
        except subprocess.TimeoutExpired:
            pass
    # An out-of-memory crash exits early and looks like a "timeout" to the ready
    # wait; relabel it so callers can recognise it and back off the context size.
    if is_oom_failure(stderr, process.returncode):
        failure = "oom"
    return ServingProbeResult(
        ok=False,
        ttft_ms=None,
        tokens_per_second=0.0,
        output_chars=0,
        generated_tokens=0,
        failure=failure,
        stdout_tail=stdout[-2000:],
        stderr_tail=stderr[-2000:],
    )
