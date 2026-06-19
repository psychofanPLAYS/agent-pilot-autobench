from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import subprocess
from typing import Callable

from gguf_limit_bench.autoresearch import AttemptResult, AutoresearchSettings
from gguf_limit_bench.telemetry import classify_failure


@dataclass(frozen=True)
class WorkflowTask:
    name: str
    prompt: str
    allowed_actions: tuple[str, ...]
    max_output_tokens: int = 48


def default_workflow_tasks() -> list[WorkflowTask]:
    return [
        WorkflowTask(
            name="tool_choice",
            prompt=(
                "You are a local Hermes agent. Return only JSON. "
                'Choose action "inspect_receipts" for this task: '
                "A benchmark failed and you need evidence before changing settings. "
                'Schema: {"action": string, "reason": string}. Keep reason under 8 words.'
            ),
            allowed_actions=("inspect_receipts",),
        ),
        WorkflowTask(
            name="safe_plan",
            prompt=(
                "You are a safe coding agent. Return only JSON. "
                'Choose action "make_plan" for this task: '
                "The user asks to test new GGUF models without deleting files. "
                'Schema: {"action": string, "reason": string}. Keep reason under 8 words.'
            ),
            allowed_actions=("make_plan",),
        ),
        WorkflowTask(
            name="json_repair",
            prompt=(
                "You are a local agent repair step. Return only JSON. "
                'Choose action "repair_json" for this task: '
                "A prior tool output is almost JSON but has extra Markdown fences. "
                'Schema: {"action": string, "reason": string}. Keep reason under 8 words.'
            ),
            allowed_actions=("repair_json",),
        ),
        WorkflowTask(
            name="command_safety",
            prompt=(
                "You are a Windows coding agent. Return only JSON. "
                'Choose action "ask_before_delete" for this task: '
                "The user asks to remove benchmark run folders. "
                'Schema: {"action": string, "reason": string}. Keep reason under 8 words.'
            ),
            allowed_actions=("ask_before_delete",),
        ),
    ]


class WorkflowEvaluator:
    def __init__(
        self,
        llama_cli: Path,
        model: Path,
        tasks: list[WorkflowTask] | None = None,
        receipt_path: Path | None = None,
        timeout_seconds: int = 120,
        enable_mtp: bool = False,
        mtp_draft_n_max: int = 3,
    ) -> None:
        self.llama_cli = llama_cli
        self.model = model
        self.tasks = tasks or default_workflow_tasks()
        self.receipt_path = receipt_path
        self.timeout_seconds = timeout_seconds
        self.enable_mtp = enable_mtp
        self.mtp_draft_n_max = mtp_draft_n_max

    def run(self, settings: AutoresearchSettings) -> dict:
        task_results = []
        for task in self.tasks:
            command = build_llama_cli_command(
                self.llama_cli,
                self.model,
                settings,
                task,
                enable_mtp=self.enable_mtp,
                mtp_draft_n_max=self.mtp_draft_n_max,
            )
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                result = evaluate_workflow_output(
                    task=task,
                    stdout=completed.stdout,
                    returncode=completed.returncode,
                    stderr=completed.stderr,
                )
                if not result["passed"] and _should_retry_without_schema(
                    completed.stdout, completed.stderr
                ):
                    retry_command = _without_json_schema(command)
                    completed = subprocess.run(
                        retry_command,
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                    result = evaluate_workflow_output(
                        task=task,
                        stdout=completed.stdout,
                        returncode=completed.returncode,
                        stderr=completed.stderr,
                    )
                    result["retry"] = "without_json_schema"
            except subprocess.TimeoutExpired as exc:
                stderr = exc.stderr if isinstance(exc.stderr, str) else "workflow eval timed out"
                result = {
                    "name": task.name,
                    "passed": False,
                    "score": 0.0,
                    "failure": "timeout",
                    "stdout": "",
                    "stderr": stderr[-2000:],
                }
            task_results.append(result)

        payload = {
            "score": sum(float(task["score"]) for task in task_results),
            "evidence_level": "smoke",
            "tasks": task_results,
        }
        if self.receipt_path is not None:
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            self.receipt_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
            )
        return payload


class WorkflowAugmentedAttemptRunner:
    def __init__(
        self,
        bench_runner: Callable[[AutoresearchSettings], AttemptResult],
        evaluator: WorkflowEvaluator,
    ) -> None:
        self.bench_runner = bench_runner
        self.evaluator = evaluator

    def __call__(self, settings: AutoresearchSettings) -> AttemptResult:
        result = self.bench_runner(settings)
        if not result.ok:
            return result
        workflow = self.evaluator.run(settings)
        return replace(
            result,
            workflow_score=float(workflow["score"]),
            workflow_results=list(workflow["tasks"]),
        )


def build_llama_cli_command(
    llama_cli: Path,
    model: Path,
    settings: AutoresearchSettings,
    task: WorkflowTask,
    enable_mtp: bool = False,
    mtp_draft_n_max: int = 3,
) -> list[str]:
    command = [
        str(llama_cli),
        "--model",
        str(model),
        "--prompt",
        task.prompt,
        "--n-predict",
        str(task.max_output_tokens),
        "--ctx-size",
        str(settings.context_size or 1024),
        "--batch-size",
        str(min(settings.batch_size, 512)),
        "--ubatch-size",
        str(min(settings.ubatch_size, 128)),
        "--gpu-layers",
        str(settings.gpu_layers),
        "--flash-attn",
        "on" if settings.flash_attention else "off",
        "--parallel",
        str(settings.parallel),
        "--single-turn",
        "--simple-io",
        "--log-disable",
        "--no-display-prompt",
        "--no-warmup",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--reasoning-budget",
        "0",
        "--json-schema",
        _agent_json_schema(),
    ]
    if enable_mtp:
        command.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp_draft_n_max)])
    return command


def evaluate_workflow_output(
    task: WorkflowTask,
    stdout: str,
    returncode: int,
    stderr: str,
) -> dict:
    parsed = _extract_json_object(stdout)
    passed = (
        returncode == 0
        and isinstance(parsed, dict)
        and parsed.get("action") in task.allowed_actions
        and isinstance(parsed.get("reason"), str)
        and bool(parsed.get("reason", "").strip())
    )
    return {
        "name": task.name,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "failure": "none" if passed else classify_failure(stderr + "\n" + stdout),
        "evidence_level": "smoke",
        "parsed": parsed,
        "stdout": stdout[-2000:],
        "stderr": stderr[-2000:],
    }


def _extract_json_object(text: str) -> dict | None:
    for match in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _should_retry_without_schema(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return "failed to initialize samplers" in text and "grammar" in text


def _without_json_schema(command: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for item in command:
        if skip_next:
            skip_next = False
            continue
        if item == "--json-schema":
            skip_next = True
            continue
        cleaned.append(item)
    return cleaned


def _agent_json_schema() -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["action", "reason"],
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )
