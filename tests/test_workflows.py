import json
from pathlib import Path

import pytest

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings
from gguf_limit_bench.runtime_capabilities import parse_llama_help
from gguf_limit_bench import workflows
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
    capabilities = parse_llama_help("version: b9596", "--spec-type VALUES\n--spec-draft-n-max N")
    command = build_llama_cli_command(
        llama_cli=Path("llama-cli.exe"),
        model=Path("Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"),
        settings=AutoresearchSettings(),
        task=default_workflow_tasks()[0],
        enable_mtp=True,
        mtp_spec_draft_n_max=3,
        runtime_capabilities=capabilities,
    )

    assert command[-4:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    assert "--draft-max" not in command
    assert "--draft-min" not in command


def test_build_llama_cli_command_rejects_mtp_draft_max_above_four():
    with pytest.raises(ValueError, match="between 1 and 4"):
        build_llama_cli_command(
            llama_cli=Path("llama-cli.exe"),
            model=Path("Qwen-MTP.gguf"),
            settings=AutoresearchSettings(),
            task=default_workflow_tasks()[0],
            enable_mtp=True,
            mtp_spec_draft_n_max=5,
        )


def test_deprecated_workflow_mtp_keyword_emits_native_flags():
    capabilities = parse_llama_help("version: b9596", "--spec-type VALUES\n--spec-draft-n-max N")
    with pytest.warns(DeprecationWarning, match="mtp_draft_max"):
        command = build_llama_cli_command(
            llama_cli=Path("llama-cli.exe"),
            model=Path("Qwen-MTP.gguf"),
            settings=AutoresearchSettings(),
            task=default_workflow_tasks()[0],
            enable_mtp=True,
            mtp_draft_max=3,
            runtime_capabilities=capabilities,
        )

    assert command[-4:] == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    assert "--draft-max" not in command


def test_workflow_evaluator_accepts_deprecated_mtp_keyword_once():
    capabilities = parse_llama_help("version: b9596", "--spec-type VALUES\n--spec-draft-n-max N")
    with pytest.warns(DeprecationWarning, match="mtp_draft_max"):
        evaluator = WorkflowEvaluator(
            llama_cli=Path("llama-cli.exe"),
            model=Path("Qwen-MTP.gguf"),
            enable_mtp=True,
            mtp_draft_max=2,
            capability_collector=lambda path: capabilities,
        )

    assert evaluator.mtp_spec_draft_n_max == 2


def test_workflow_mtp_draft_max_must_be_between_one_and_four():
    capabilities = parse_llama_help("version: b9596", "--spec-type VALUES\n--spec-draft-n-max N")
    for invalid in (0, 5):
        with pytest.raises(ValueError, match="between 1 and 4"):
            build_llama_cli_command(
                llama_cli=Path("llama-cli.exe"),
                model=Path("Qwen-MTP.gguf"),
                settings=AutoresearchSettings(),
                task=default_workflow_tasks()[0],
                enable_mtp=True,
                mtp_spec_draft_n_max=invalid,
                runtime_capabilities=capabilities,
            )


def test_workflow_evaluator_skips_mtp_when_cli_capabilities_are_unknown(tmp_path, monkeypatch):
    capabilities = parse_llama_help("version: b9596", "--model FNAME")
    monkeypatch.setattr(
        workflows.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported MTP must not launch llama-cli")
        ),
    )
    evaluator = WorkflowEvaluator(
        llama_cli=Path("llama-cli.exe"),
        model=Path("Qwen-MTP.gguf"),
        enable_mtp=True,
        capability_collector=lambda path: capabilities,
        receipt_path=tmp_path / "workflow.json",
    )

    result = evaluator.run(AutoresearchSettings())

    assert result["score"] == 0
    assert result["tasks"][0]["failure"] == "mtp_runtime_unsupported"
    assert json.loads((tmp_path / "workflow.json").read_text(encoding="utf-8")) == result


def test_workflow_evaluator_collects_cli_capabilities_once(monkeypatch):
    capabilities = parse_llama_help("version: b9596", "--spec-type VALUES\n--spec-draft-n-max N")
    collected = []
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": '{"action":"inspect_receipts","reason":"Need evidence"}',
                "stderr": "",
            },
        )()

    monkeypatch.setattr(workflows.subprocess, "run", fake_run)
    tasks = [default_workflow_tasks()[0], default_workflow_tasks()[0]]
    evaluator = WorkflowEvaluator(
        llama_cli=Path("llama-cli.exe"),
        model=Path("Qwen-MTP.gguf"),
        tasks=tasks,
        enable_mtp=True,
        capability_collector=lambda path: collected.append(path) or capabilities,
    )

    evaluator.run(AutoresearchSettings())

    assert collected == [Path("llama-cli.exe")]
    assert len(commands) == 2
    assert all("--spec-type" in command for command in commands)


def test_evaluate_workflow_output_scores_valid_agent_json():
    task = default_workflow_tasks()[0]
    output = 'notes before\n{"action":"inspect_receipts","reason":"Need benchmark evidence."}\n'

    result = evaluate_workflow_output(task, output, returncode=0, stderr="")

    assert result["passed"] is True
    assert result["score"] > 0
    assert result["parsed"]["action"] == "inspect_receipts"
    assert result["evidence_level"] == "smoke"


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
    assert result["evidence_level"] == "smoke"
    payload = json.loads((tmp_path / "workflow-results.json").read_text(encoding="utf-8"))
    assert payload["tasks"][0]["passed"] is True
    assert payload["evidence_level"] == "smoke"


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
