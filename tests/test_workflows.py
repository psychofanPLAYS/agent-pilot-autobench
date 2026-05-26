import json
from pathlib import Path

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings
from gguf_limit_bench.workflows import (
    WorkflowAugmentedAttemptRunner,
    WorkflowEvaluator,
    build_llama_cli_command,
    default_workflow_tasks,
    evaluate_workflow_output,
)


def test_build_llama_cli_command_uses_low_token_agent_eval_flags():
    task = default_workflow_tasks()[0]
    command = build_llama_cli_command(
        llama_cli=Path("G:/AI/llamaCPP-server/_internal/runtime/llama.cpp/llama-cli.exe"),
        model=Path("G:/AI/models/Qwen3-Test-Q4_K_M.gguf"),
        settings=AutoresearchSettings(context_size=4096, parallel=2),
        task=task,
    )

    assert "--no-display-prompt" in command
    assert "--simple-io" in command
    assert "--reasoning-budget" in command
    assert "--json-schema" in command
    assert "0" in command
    assert "--n-predict" in command
    assert str(task.max_output_tokens) in command
    assert "--parallel" in command
    assert "2" in command


def test_build_llama_cli_command_can_enable_mtp_draft_probe():
    command = build_llama_cli_command(
        llama_cli=Path("llama-cli.exe"),
        model=Path("Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"),
        settings=AutoresearchSettings(),
        task=default_workflow_tasks()[0],
        enable_mtp=True,
        mtp_draft_max=16,
    )

    assert "--draft-max" in command
    assert "16" in command


def test_evaluate_workflow_output_scores_valid_agent_json():
    task = default_workflow_tasks()[0]
    output = 'notes before\n{"action":"inspect_receipts","reason":"Need benchmark evidence."}\n'

    result = evaluate_workflow_output(task, output, returncode=0, stderr="")

    assert result["passed"] is True
    assert result["score"] > 0
    assert result["parsed"]["action"] == "inspect_receipts"


def test_workflow_augmented_runner_adds_real_world_score_without_breaking_bench_result():
    def bench_runner(settings: AutoresearchSettings) -> AttemptResult:
        return AttemptResult(
            ok=True,
            generation_tokens_per_second=50.0,
            prompt_tokens_per_second=900.0,
            ttft_ms=None,
            context_size=settings.context_size,
            failure="unknown",
            stdout="{}",
            stderr="",
            returncode=0,
        )

    class FakeEvaluator:
        def run(self, settings: AutoresearchSettings):
            return {"score": 2.5, "tasks": [{"name": "tool_choice", "passed": True}]}

    runner = WorkflowAugmentedAttemptRunner(bench_runner=bench_runner, evaluator=FakeEvaluator())
    result = runner(AutoresearchSettings())

    assert result.ok is True
    assert result.workflow_score == 2.5
    assert result.workflow_results[0]["name"] == "tool_choice"


def test_workflow_evaluator_writes_small_json_receipt(tmp_path, monkeypatch):
    task = default_workflow_tasks()[0]

    def fake_run(command, capture_output, check, text, timeout):
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"action": "inspect_receipts", "reason": "Check evidence."}),
                "stderr": "",
            },
        )()

    monkeypatch.setattr("subprocess.run", fake_run)
    evaluator = WorkflowEvaluator(
        llama_cli=Path("llama-cli.exe"),
        model=Path("model.gguf"),
        tasks=[task],
        receipt_path=tmp_path / "workflow-results.json",
        timeout_seconds=10,
    )

    result = evaluator.run(AutoresearchSettings())

    assert result["score"] > 0
    payload = json.loads((tmp_path / "workflow-results.json").read_text(encoding="utf-8"))
    assert payload["tasks"][0]["passed"] is True


def test_workflow_evaluator_retries_without_schema_when_llama_grammar_fails(tmp_path, monkeypatch):
    task = default_workflow_tasks()[0]
    calls = []

    def fake_run(command, capture_output, check, text, timeout):
        calls.append(command)
        if len(calls) == 1:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "Error: Failed to initialize samplers: Unexpected empty grammar stack",
                    "stderr": "",
                },
            )()
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"action": "inspect_receipts", "reason": "Need evidence."}),
                "stderr": "",
            },
        )()

    monkeypatch.setattr("subprocess.run", fake_run)
    evaluator = WorkflowEvaluator(
        llama_cli=Path("llama-cli.exe"),
        model=Path("model.gguf"),
        tasks=[task],
        receipt_path=tmp_path / "workflow-results.json",
        timeout_seconds=10,
    )

    result = evaluator.run(AutoresearchSettings())

    assert result["tasks"][0]["passed"] is True
    assert result["tasks"][0]["retry"] == "without_json_schema"
    assert "--json-schema" in calls[0]
    assert "--json-schema" not in calls[1]
