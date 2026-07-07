from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated long llama-server completions and write a stability receipt."
    )
    parser.add_argument("--llama-server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ctx-size", required=True, type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--chat-template-file", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    host = "127.0.0.1"
    port = free_port()
    command = [
        str(args.llama_server),
        "--model",
        str(args.model),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(args.ctx_size),
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--parallel",
        "1",
        "--metrics",
        "--slots",
        "--no-webui",
        "--gpu-layers",
        "99",
    ]
    for item in args.server_arg:
        command.extend(split_arg(item))
    if args.chat_template_file:
        command.extend(["--jinja", "--chat-template-file", str(args.chat_template_file)])

    (args.out / "launch-command.json").write_text(
        json.dumps(command, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stdout_log = (args.out / "server.stdout.log").open("w", encoding="utf-8")
    stderr_log = (args.out / "server.stderr.log").open("w", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=stdout_log, stderr=stderr_log, text=True)
    base_url = f"http://{host}:{port}"
    results: list[dict] = []
    status = "failed"
    failure = "unknown"
    try:
        ready_at = wait_until_ready(base_url, process, args.timeout_seconds)
        for index in range(1, args.repeat + 1):
            result = run_chat_completion(
                base_url=base_url,
                request_index=index,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
            )
            results.append(result)
            write_jsonl(args.out / "requests.jsonl", result)
            if not result["ok"]:
                failure = str(result["failure"])
                break
        else:
            status = "passed"
            failure = "none"
        receipt = {
            "status": status,
            "failure": failure,
            "model": str(args.model),
            "ctx_size": args.ctx_size,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "server_ready_ms": (ready_at - started) * 1000.0,
            "results": results,
        }
        return 0 if status == "passed" else 1
    except TimeoutError as exc:
        receipt = {
            "status": "failed",
            "failure": f"timeout: {exc}",
            "model": str(args.model),
            "ctx_size": args.ctx_size,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "results": results,
        }
        return 124
    finally:
        stop_process(process)
        stdout_log.close()
        stderr_log.close()
        stdout_tail = read_tail(args.out / "server.stdout.log")
        stderr_tail = read_tail(args.out / "server.stderr.log")
        receipt["returncode"] = process.returncode
        receipt["stdout_tail"] = stdout_tail
        receipt["stderr_tail"] = stderr_tail
        receipt["cuda_illegal_memory"] = "illegal memory access" in stderr_tail.lower()
        (args.out / "summary.json").write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.out / "server-tail.log").write_text(stderr_tail + "\n", encoding="utf-8")


def run_chat_completion(
    *,
    base_url: str,
    request_index: int,
    max_tokens: int,
    timeout_seconds: int,
) -> dict:
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a stability probe. Continue writing dense technical notes "
                    "until the generation limit stops you. Do not summarize early."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Probe {request_index}: write a long numbered incident analysis with "
                    "many concrete observations, hypotheses, tests, and rollback steps. "
                    "Keep going until the token budget ends."
                ),
            },
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0,
        "ignore_eos": True,
    }
    payload = json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0
    chars = 0
    usage_completion_tokens: int | None = None
    predicted_per_second: float | None = None
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for event in iter_sse(response):
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                chunks += 1
                usage = event.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    usage_completion_tokens = int(usage["completion_tokens"])
                timings = event.get("timings")
                if isinstance(timings, dict) and timings.get("predicted_per_second") is not None:
                    predicted_per_second = float(timings["predicted_per_second"])
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    for field in ("reasoning_content", "content"):
                        value = delta.get(field)
                        if isinstance(value, str):
                            chars += len(value)
    except (OSError, URLError) as exc:
        return {
            "request_index": request_index,
            "ok": False,
            "failure": f"request_error: {exc}",
            "chunks": chunks,
            "output_chars": chars,
            "completion_tokens": usage_completion_tokens,
        }
    finished = time.perf_counter()
    if first_token_at is None:
        return {
            "request_index": request_index,
            "ok": False,
            "failure": "no_streamed_token",
            "chunks": chunks,
            "output_chars": chars,
            "completion_tokens": usage_completion_tokens,
        }
    measured_tokens = usage_completion_tokens or chunks
    generation_seconds = max(finished - first_token_at, 0.001)
    return {
        "request_index": request_index,
        "ok": True,
        "failure": "none",
        "ttft_ms": (first_token_at - started) * 1000.0,
        "tokens_per_second": predicted_per_second or measured_tokens / generation_seconds,
        "completion_tokens": measured_tokens,
        "chunks": chunks,
        "output_chars": chars,
        "elapsed_seconds": finished - started,
    }


def split_arg(value: str) -> list[str]:
    if "=" in value:
        flag, raw = value.split("=", 1)
        return [flag, raw]
    return [value]


def iter_sse(response):
    for raw_line in response:
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


def wait_until_ready(base_url: str, process: subprocess.Popen, timeout_seconds: int) -> float:
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


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_tail(path: Path, limit: int = 16000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


if __name__ == "__main__":
    sys.exit(main())
