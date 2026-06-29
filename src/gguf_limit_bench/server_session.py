"""server_session: context manager that launches a llama-server and yields base_url.

This module provides :func:`llama_server_session`, which starts a
``llama-server`` subprocess for the given model and
:class:`~gguf_limit_bench.autoresearch.AutoresearchSettings`, waits until the
HTTP API is ready, and guarantees teardown on exit.

It is used by :mod:`gguf_limit_bench.champion_eval` to avoid duplicating the
launch / ready-wait / teardown logic that already lives in
:class:`~gguf_limit_bench.simple_bench_runner.LlamaServerSimpleBenchAttemptRunner`.

The existing runner and all callers that depend on it are **not modified**.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from gguf_limit_bench.autoresearch import AutoresearchSettings
from gguf_limit_bench.server_probe import (
    _free_port,
    _stop_process,
    _wait_until_ready,
    build_llama_server_command,
)


@contextmanager
def llama_server_session(
    *,
    llama_server: Path,
    model: Path,
    settings: AutoresearchSettings,
    host: str = "127.0.0.1",
    port: int | None = None,
    log_dir: Path | None = None,
    timeout_seconds: int = 600,
) -> Generator[str, None, None]:
    """Context manager that starts llama-server and yields the base URL.

    Parameters
    ----------
    llama_server:
        Path to the llama-server executable.
    model:
        Path to the GGUF model file.
    settings:
        Launch parameters (context, GPU layers, etc.).
    host:
        Bind host (default ``"127.0.0.1"``).
    port:
        Port to bind; if *None* a free port is chosen automatically.
    log_dir:
        Directory for server stdout/stderr logs.  When *None* the process
        stdout/stderr are suppressed (sent to ``subprocess.DEVNULL``).
    timeout_seconds:
        How long to wait for the server to become ready.

    Yields
    ------
    str
        The base URL of the running server, e.g. ``"http://127.0.0.1:8080"``.

    Raises
    ------
    OSError
        If the server process cannot be started.
    TimeoutError
        If the server does not become ready within *timeout_seconds*.
    """
    actual_port = port or _free_port()
    command = build_llama_server_command(
        llama_server=llama_server,
        model=model,
        settings=settings,
        host=host,
        port=actual_port,
    )

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_file = (log_dir / "champion-server.stdout.log").open("w", encoding="utf-8")
        stderr_file = (log_dir / "champion-server.stderr.log").open("w", encoding="utf-8")
    else:
        stdout_file = subprocess.DEVNULL  # type: ignore[assignment]
        stderr_file = subprocess.DEVNULL  # type: ignore[assignment]

    process = subprocess.Popen(
        command,
        stdout=stdout_file,
        stderr=stderr_file,
        text=log_dir is not None,
    )
    base_url = f"http://{host}:{actual_port}"
    try:
        _wait_until_ready(base_url, process, timeout_seconds=timeout_seconds)
        yield base_url
    finally:
        _stop_process(process)
        if log_dir is not None:
            stdout_file.close()  # type: ignore[union-attr]
            stderr_file.close()  # type: ignore[union-attr]
