from __future__ import annotations

from typing import Iterable

from gguf_limit_bench.run_config import RunStatus


MIN_AGENT_CONTEXT = 32_768
MIN_AGENT_WORKFLOW_TASKS = 4
MIN_USEFUL_GENERATION_TPS = 20.0


def normalize_success_failure(ok: bool, failure: str) -> str:
    if ok and failure == "unknown":
        return "none"
    return failure


def evidence_status(
    *,
    ok: bool,
    failure: str,
    generation_tps: float,
    context_size: int,
    workflow_score: float,
    workflow_results: Iterable[dict] | None,
    serving_ttft_ms: float | None = None,
) -> RunStatus:
    failure = normalize_success_failure(ok, failure)
    if not ok:
        if failure in {"model_load", "gpu_oom", "memory_allocation", "crash"}:
            return RunStatus.FAILED
        return RunStatus.PARTIAL
    if generation_tps < MIN_USEFUL_GENERATION_TPS:
        return RunStatus.SLOW
    if context_size <= 0:
        if serving_ttft_ms is not None:
            return RunStatus.SERVING_MEASURED
        return RunStatus.SPEED_ONLY
    if context_size < MIN_AGENT_CONTEXT:
        return RunStatus.CONTEXT_UNPROVEN

    workflow_results = list(workflow_results or [])
    if not workflow_results:
        return RunStatus.WORKFLOW_UNPROVEN
    passed_count = sum(1 for task in workflow_results if task.get("passed") is True)
    if passed_count < MIN_AGENT_WORKFLOW_TASKS or workflow_score < MIN_AGENT_WORKFLOW_TASKS:
        return RunStatus.WORKFLOW_WEAK
    return RunStatus.WORKFLOW_SMOKE


def display_status(status: str) -> str:
    return status.upper().replace("_", " ")
