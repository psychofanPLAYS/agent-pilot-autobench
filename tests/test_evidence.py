from gguf_limit_bench.evidence import evidence_status
from gguf_limit_bench.run_config import RunStatus


def test_current_status_vocabulary_has_no_agent_ready_shortcut():
    statuses = {status.value for status in RunStatus}

    assert "agent_ready" not in statuses
    assert "serving_measured" in statuses
    assert "workflow_smoke" in statuses


def test_serving_ttft_without_context_is_serving_measured():
    status = evidence_status(
        ok=True,
        failure="none",
        generation_tps=80.0,
        context_size=0,
        workflow_score=0.0,
        workflow_results=[],
        serving_ttft_ms=120.0,
    )

    assert status == RunStatus.SERVING_MEASURED


def test_complete_local_workflow_evidence_is_only_workflow_smoke():
    status = evidence_status(
        ok=True,
        failure="none",
        generation_tps=80.0,
        context_size=65_536,
        workflow_score=4.0,
        workflow_results=[
            {"name": "tool_choice", "passed": True},
            {"name": "safe_plan", "passed": True},
            {"name": "json_repair", "passed": True},
            {"name": "command_safety", "passed": True},
        ],
    )

    assert status == RunStatus.WORKFLOW_SMOKE
