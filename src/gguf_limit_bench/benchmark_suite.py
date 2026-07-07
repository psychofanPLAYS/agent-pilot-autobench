from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import importlib.util
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, TypeGuard


GENERAL_LEDGER = "benchmark-suite.tsv"
AGENTIC_LEDGER = "agentic-suite.tsv"
AGENT_SCORE_LEDGER = "agent-bench-score.tsv"
SCORE_KEYS = (
    "agent_bench_score",
    "score",
    "accuracy",
    "overall_accuracy",
    "acc",
    "pass_rate",
    "exact_match",
)


@dataclass(frozen=True)
class BenchmarkSuiteTask:
    id: str
    phase: str
    harness: str
    commands: tuple[tuple[str, ...], ...]
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 600
    min_score: float | None = None
    score_file: str | None = None


@dataclass(frozen=True)
class BenchmarkSuitePlan:
    model: str
    context: int
    settings: dict[str, Any] = field(default_factory=dict)
    tasks: tuple[BenchmarkSuiteTask, ...] = field(default_factory=tuple)

    @classmethod
    def from_path(cls, path: Path) -> "BenchmarkSuitePlan":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks = tuple(_task_from_dict(item) for item in payload.get("tasks", []))
        if not tasks:
            raise ValueError("Benchmark suite plan must contain at least one task.")
        phases = {task.phase for task in tasks}
        invalid = sorted(phases - {"general", "agentic"})
        if invalid:
            raise ValueError(f"Unsupported benchmark suite phase: {', '.join(invalid)}")
        return cls(
            model=str(payload["model"]),
            context=int(payload.get("context", 0)),
            settings=dict(payload.get("settings", {})),
            tasks=tasks,
        )


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    id: str
    phase: str
    harness: str
    ok: bool
    score: float | None
    pass_fail: str
    runtime_seconds: float
    failure_class: str
    receipt_path: str
    stdout_tail: str
    stderr_tail: str
    commands: tuple[tuple[str, ...], ...]
    tool_validity: str = "not_applicable"


@dataclass(frozen=True)
class BenchmarkSuiteRun:
    run_id: str
    receipt_path: str
    model: str
    context: int
    settings: dict[str, Any]
    agent_bench_score: float
    general_score: float | None
    agentic_score: float | None
    results: tuple[BenchmarkSuiteResult, ...]

    @property
    def ok(self) -> bool:
        phases = {result.phase for result in self.results}
        return (
            "general" in phases
            and "agentic" in phases
            and all(result.ok for result in self.results)
            and self.general_score is not None
            and self.agentic_score is not None
        )


@dataclass(frozen=True)
class BenchmarkSuitePreflightIssue:
    task_id: str
    phase: str
    harness: str
    command_index: int
    executable: str
    failure_class: str
    detail: str


@dataclass(frozen=True)
class BenchmarkSuitePreflight:
    ok: bool
    status: str
    model: str
    context: int
    issue_count: int
    issues: tuple[BenchmarkSuitePreflightIssue, ...]
    receipt_path: str
    next_action: str
    plan_path: str | None = None
    plan_fingerprint: str | None = None


def benchmark_suite_run_to_dict(run: BenchmarkSuiteRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "receipt_path": run.receipt_path,
        "model": run.model,
        "context": run.context,
        "settings": run.settings,
        "agent_bench_score": run.agent_bench_score,
        "general_score": run.general_score,
        "agentic_score": run.agentic_score,
        "ok": run.ok,
        "results": [asdict(result) for result in run.results],
    }


def benchmark_suite_preflight_to_dict(preflight: BenchmarkSuitePreflight) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": preflight.ok,
        "status": preflight.status,
        "model": preflight.model,
        "context": preflight.context,
        "issue_count": preflight.issue_count,
        "issues": [asdict(issue) for issue in preflight.issues],
        "receipt_path": preflight.receipt_path,
        "next_action": preflight.next_action,
        "plan_path": preflight.plan_path,
        "plan_fingerprint": preflight.plan_fingerprint,
    }


def preflight_benchmark_suite(
    plan: BenchmarkSuitePlan,
    runs_root: Path,
    *,
    plan_path: Path | None = None,
) -> BenchmarkSuitePreflight:
    runs_root.mkdir(parents=True, exist_ok=True)
    issues: list[BenchmarkSuitePreflightIssue] = []
    for task in plan.tasks:
        for command_index, command in enumerate(task.commands, start=1):
            issues.extend(_preflight_command_shape(task, command_index, command, plan))
            resolved = _resolve_local_command(command)
            executable = resolved[0]
            if not _command_available(executable):
                issues.append(
                    BenchmarkSuitePreflightIssue(
                        task_id=task.id,
                        phase=task.phase,
                        harness=task.harness,
                        command_index=command_index,
                        executable=executable,
                        failure_class="harness_missing",
                        detail=f"Executable `{executable}` was not found on PATH.",
                    )
                )
                continue
            module_issue = _preflight_python_module(task, command_index, resolved)
            if module_issue is not None:
                issues.append(module_issue)
    plan_fingerprint = _plan_fingerprint(plan_path, plan)
    status = "PASS" if not issues else "HARNESS_MISSING"
    if any(issue.failure_class == "invalid_plan" for issue in issues):
        status = "INVALID_PLAN"
    next_action = (
        "Benchmark-suite command preflight passed; run the suite against the live model."
        if not issues
        else _preflight_next_action(issues)
    )
    receipt_path = runs_root / "benchmark-suite-preflight.json"
    preflight = BenchmarkSuitePreflight(
        ok=not issues,
        status=status,
        model=plan.model,
        context=plan.context,
        issue_count=len(issues),
        issues=tuple(issues),
        receipt_path=str(receipt_path),
        next_action=next_action,
        plan_path=None if plan_path is None else str(plan_path),
        plan_fingerprint=plan_fingerprint,
    )
    payload = benchmark_suite_preflight_to_dict(preflight)
    receipt_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    (runs_root / "benchmark-suite-preflight.md").write_text(
        _preflight_markdown(payload),
        encoding="utf-8",
    )
    return preflight


def run_benchmark_suite(
    plan: BenchmarkSuitePlan,
    runs_root: Path,
    timeout_seconds: float | None = None,
) -> BenchmarkSuiteRun:
    runs_root.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
    receipt_path = _new_receipt_dir(runs_root)
    receipt_path.mkdir(parents=True, exist_ok=False)
    _suite_event(
        receipt_path,
        "benchmark_suite_started",
        {
            "model": plan.model,
            "context": plan.context,
            "tasks": [task.id for task in plan.tasks],
        },
    )
    (receipt_path / "suite-plan.json").write_text(
        json.dumps(
            {
                "model": plan.model,
                "context": plan.context,
                "settings": plan.settings,
                "tasks": [asdict(task) for task in plan.tasks],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    results: list[BenchmarkSuiteResult] = []
    for task in plan.tasks:
        _suite_event(
            receipt_path,
            "benchmark_suite_task_started",
            {"task": task.id, "phase": task.phase, "harness": task.harness},
        )
        result = _run_task(
            task,
            plan=plan,
            runs_root=runs_root,
            receipt_path=receipt_path,
            deadline=deadline,
        )
        results.append(result)
        _suite_event(
            receipt_path,
            "benchmark_suite_task_finished",
            {
                "task": task.id,
                "phase": task.phase,
                "harness": task.harness,
                "ok": result.ok,
                "score": result.score,
                "failure_class": result.failure_class,
                "runtime_seconds": round(result.runtime_seconds, 3),
            },
        )
        _append_phase_ledger(runs_root, plan, result)

    suite_run = BenchmarkSuiteRun(
        run_id=receipt_path.name,
        receipt_path=str(receipt_path),
        model=plan.model,
        context=plan.context,
        settings=plan.settings,
        general_score=_phase_average(results, "general"),
        agentic_score=_phase_average(results, "agentic"),
        agent_bench_score=_agent_bench_score(results),
        results=tuple(results),
    )
    _append_agent_score_ledger(runs_root, suite_run)
    (receipt_path / "suite-summary.json").write_text(
        json.dumps(benchmark_suite_run_to_dict(suite_run), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    _write_suite_verdict(receipt_path, suite_run)
    _suite_event(
        receipt_path,
        "benchmark_suite_finished",
        {
            "ok": suite_run.ok,
            "agent_bench_score": suite_run.agent_bench_score,
            "general_score": suite_run.general_score,
            "agentic_score": suite_run.agentic_score,
        },
    )
    return suite_run


def _command_available(executable: str) -> bool:
    path = Path(executable)
    if path.is_absolute() or path.parent != Path("."):
        return path.exists()
    return shutil.which(executable) is not None


def _preflight_next_action(issues: list[BenchmarkSuitePreflightIssue]) -> str:
    invalid = [issue for issue in issues if issue.failure_class == "invalid_plan"]
    if invalid:
        details = "; ".join(issue.detail for issue in invalid[:3])
        return (
            f"Fix benchmark-suite plan wiring before deployment proof: {details}. "
            "Regenerate the plan or edit it, then rerun benchmark-suite-preflight."
        )
    missing = ", ".join(sorted({issue.executable for issue in issues}))
    harnesses = ", ".join(sorted({issue.harness for issue in issues}))
    return (
        f"Install or expose missing benchmark harness executable(s): {missing}. "
        f"Affected harnesses: {harnesses}. Rerun benchmark-suite-preflight before deployment proof."
    )


def _preflight_command_shape(
    task: BenchmarkSuiteTask,
    command_index: int,
    command: tuple[str, ...],
    plan: BenchmarkSuitePlan,
) -> list[BenchmarkSuitePreflightIssue]:
    issues: list[BenchmarkSuitePreflightIssue] = []
    joined = "\n".join(command)
    if "{base_url}" in joined and not str(plan.settings.get("base_url") or "").strip():
        issues.append(
            BenchmarkSuitePreflightIssue(
                task_id=task.id,
                phase=task.phase,
                harness=task.harness,
                command_index=command_index,
                executable=command[0],
                failure_class="invalid_plan",
                detail="Command uses {base_url}, but plan.settings.base_url is blank.",
            )
        )
    if task.harness == "librarian-suite":
        issues.extend(_preflight_librarian_command(task, command_index, command, plan))
    return issues


def _preflight_librarian_command(
    task: BenchmarkSuiteTask,
    command_index: int,
    command: tuple[str, ...],
    plan: BenchmarkSuitePlan,
) -> list[BenchmarkSuitePreflightIssue]:
    issues: list[BenchmarkSuitePreflightIssue] = []
    settings_arg = _value_after(command, "--settings-json")
    if settings_arg is None:
        issues.append(
            _invalid_plan_issue(
                task,
                command_index,
                command[0],
                "librarian-suite command must pass --settings-json.",
            )
        )
    else:
        expanded_settings = _expand_token(
            settings_arg,
            plan=plan,
            runs_root=None,
            receipt_path=None,
            task_dir=None,
        )
        try:
            payload = json.loads(expanded_settings)
        except json.JSONDecodeError:
            payload = {}
        args = tuple(str(item) for item in payload.get("extra_server_args", ()))
        if "--jinja" not in args:
            issues.append(
                _invalid_plan_issue(
                    task,
                    command_index,
                    command[0],
                    (
                        "librarian-suite settings_json must include runtime "
                        "extra_server_args with --jinja."
                    ),
                )
            )
    for pack_id in _values_after_all(command, "--pack"):
        try:
            from gguf_limit_bench.packs import load_pack

            load_pack(pack_id)
        except Exception as exc:  # pragma: no cover - defensive receipt detail
            issues.append(
                _invalid_plan_issue(
                    task,
                    command_index,
                    command[0],
                    f"librarian-suite pack `{pack_id}` could not be loaded: {exc}",
                )
            )
    return issues


def _preflight_python_module(
    task: BenchmarkSuiteTask,
    command_index: int,
    command: tuple[str, ...],
) -> BenchmarkSuitePreflightIssue | None:
    if not command:
        return None
    executable = Path(command[0]).name.lower()
    if executable not in {Path(sys.executable).name.lower(), "python", "python.exe"}:
        return None
    try:
        module_index = command.index("-m")
    except ValueError:
        return None
    if module_index + 1 >= len(command):
        return _invalid_plan_issue(
            task,
            command_index,
            command[0],
            "Python command uses -m without a module name.",
        )
    module_name = command[module_index + 1]
    if importlib.util.find_spec(module_name) is None:
        return BenchmarkSuitePreflightIssue(
            task_id=task.id,
            phase=task.phase,
            harness=task.harness,
            command_index=command_index,
            executable=command[0],
            failure_class="harness_missing",
            detail=f"Python module `{module_name}` is not importable in the current environment.",
        )
    return None


def _invalid_plan_issue(
    task: BenchmarkSuiteTask,
    command_index: int,
    executable: str,
    detail: str,
) -> BenchmarkSuitePreflightIssue:
    return BenchmarkSuitePreflightIssue(
        task_id=task.id,
        phase=task.phase,
        harness=task.harness,
        command_index=command_index,
        executable=executable,
        failure_class="invalid_plan",
        detail=detail,
    )


def _value_after(args: tuple[str, ...], option: str) -> str | None:
    values = _values_after_all(args, option)
    return values[0] if values else None


def _values_after_all(args: tuple[str, ...], option: str) -> list[str]:
    values: list[str] = []
    for index, arg in enumerate(args):
        if arg == option and index + 1 < len(args):
            values.append(args[index + 1])
        elif arg.startswith(f"{option}="):
            values.append(arg.removeprefix(f"{option}="))
    return values


def _plan_fingerprint(plan_path: Path | None, plan: BenchmarkSuitePlan) -> str:
    if plan_path is not None:
        try:
            return sha256(plan_path.read_bytes()).hexdigest()
        except OSError:
            pass
    payload = {
        "model": plan.model,
        "context": plan.context,
        "settings": plan.settings,
        "tasks": [asdict(task) for task in plan.tasks],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _preflight_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Suite Preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Model: `{payload['model']}`",
        f"- Context: `{payload['context']}`",
        f"- Issues: `{payload['issue_count']}`",
        f"- Next action: {payload['next_action']}",
        "",
        "| Task | Phase | Harness | Command | Executable | Failure | Detail |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for issue in payload.get("issues", []):
        lines.append(
            f"| `{issue['task_id']}` | `{issue['phase']}` | `{issue['harness']}` | "
            f"{issue['command_index']} | `{issue['executable']}` | "
            f"`{issue['failure_class']}` | {issue['detail']} |"
        )
    lines.append("")
    return "\n".join(lines)


def suite_verdict(suite_run: BenchmarkSuiteRun) -> dict[str, Any]:
    if suite_run.ok:
        action = "PROMOTE"
        confidence = "high"
        summary = (
            "This benchmark-suite run passed the required general and agentic phases. "
            "It is valid recommendation evidence for this model/settings profile."
        )
        next_run = "Compare challenger models or settings with the same benchmark-suite plan."
    else:
        action = "REJECT"
        confidence = "high"
        summary = (
            "This benchmark-suite run did not pass. Do not promote this model/settings "
            "profile from speed, fit, or partial evidence."
        )
        next_run = "Inspect failed task receipts, fix the cause, then rerun the same plan."
    return {
        "action": action,
        "confidence": confidence,
        "model": suite_run.model,
        "context": suite_run.context,
        "agent_bench_score": suite_run.agent_bench_score,
        "general_score": suite_run.general_score,
        "agentic_score": suite_run.agentic_score,
        "suite_ok": suite_run.ok,
        "summary": summary,
        "next_run": next_run,
        "receipt_path": suite_run.receipt_path,
    }


def _write_suite_verdict(receipt_path: Path, suite_run: BenchmarkSuiteRun) -> None:
    verdict = suite_verdict(suite_run)
    (receipt_path / "suite-verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (receipt_path / "suite-verdict.md").write_text(
        _suite_verdict_markdown(verdict),
        encoding="utf-8",
    )


def _suite_verdict_markdown(verdict: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Suite Verdict",
        "",
        f"- Action: `{verdict['action']}`",
        f"- Confidence: `{verdict['confidence']}`",
        f"- Model: `{verdict['model']}`",
        f"- Context: `{verdict['context']}`",
        f"- Agent bench score: `{verdict['agent_bench_score']:.6f}`",
        f"- General score: `{_fmt_optional_score(verdict['general_score'])}`",
        f"- Agentic score: `{_fmt_optional_score(verdict['agentic_score'])}`",
        "",
        "## Why",
        "",
        str(verdict["summary"]),
        "",
        "## Next Run",
        "",
        str(verdict["next_run"]),
        "",
    ]
    return "\n".join(lines)


def _fmt_optional_score(value: Any) -> str:
    return "unmeasured" if value is None else f"{float(value):.6f}"


def _task_from_dict(payload: dict[str, Any]) -> BenchmarkSuiteTask:
    commands = _commands_from_payload(payload)
    min_score_value = payload.get("min_score")
    return BenchmarkSuiteTask(
        id=str(payload["id"]),
        phase=str(payload["phase"]),
        harness=str(payload.get("harness", payload["phase"])),
        commands=commands,
        env={str(key): str(value) for key, value in dict(payload.get("env", {})).items()},
        timeout_seconds=int(payload.get("timeout_seconds", 600)),
        min_score=None if min_score_value is None else float(min_score_value),
        score_file=None if payload.get("score_file") is None else str(payload.get("score_file")),
    )


def _commands_from_payload(payload: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    if "commands" in payload:
        commands = payload["commands"]
        if not isinstance(commands, list) or not commands:
            raise ValueError("Benchmark suite task commands must be a non-empty list.")
        parsed: list[tuple[str, ...]] = []
        for command in commands:
            if not isinstance(command, list) or not command:
                raise ValueError("Each benchmark suite command must be a non-empty list.")
            parsed.append(tuple(str(part) for part in command))
        return tuple(parsed)
    command = payload.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError("Benchmark suite task command must be a non-empty list.")
    return (tuple(str(part) for part in command),)


def _run_task(
    task: BenchmarkSuiteTask,
    *,
    plan: BenchmarkSuitePlan,
    runs_root: Path,
    receipt_path: Path,
    deadline: float | None = None,
) -> BenchmarkSuiteResult:
    task_dir = receipt_path / _safe_id(task.id)
    task_dir.mkdir(parents=True, exist_ok=False)
    commands = tuple(
        _resolve_local_command(
            tuple(
                _expand_token(
                    part,
                    plan=plan,
                    runs_root=runs_root,
                    receipt_path=receipt_path,
                    task_dir=task_dir,
                )
                for part in command
            )
        )
        for command in task.commands
    )
    task_env = {
        key: _expand_token(
            value,
            plan=plan,
            runs_root=runs_root,
            receipt_path=receipt_path,
            task_dir=task_dir,
        )
        for key, value in task.env.items()
    }
    started = time.monotonic()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    returncode = 0
    try:
        for command_index, command in enumerate(commands, start=1):
            command_timeout = _remaining_timeout(deadline, task.timeout_seconds)
            _suite_event(
                receipt_path,
                "benchmark_suite_command_started",
                {
                    "task": task.id,
                    "command_index": command_index,
                    "command": _display_command(command),
                    "timeout_seconds": round(command_timeout, 3),
                },
            )
            completed = subprocess.run(
                list(command),
                capture_output=True,
                check=False,
                text=True,
                timeout=command_timeout,
                env={**os.environ, **task_env} if task_env else None,
            )
            stdout_parts.append(completed.stdout or "")
            stderr_parts.append(completed.stderr or "")
            returncode = completed.returncode
            _suite_event(
                receipt_path,
                "benchmark_suite_command_finished",
                {
                    "task": task.id,
                    "command_index": command_index,
                    "returncode": returncode,
                },
            )
            (task_dir / f"command-{command_index}.json").write_text(
                json.dumps(
                    {
                        "command": list(command),
                        "env": _redacted_env(task_env),
                        "returncode": returncode,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if returncode != 0:
                break
        runtime = time.monotonic() - started
        stdout = "\n".join(stdout_parts)
        stderr = "\n".join(stderr_parts)
        score = _score_from_task_output(task, stdout=stdout, task_dir=task_dir)
        ok = returncode == 0 and score is not None
        pass_fail = (
            "pass"
            if returncode == 0 and score is not None and _passes_threshold(score, task.min_score)
            else "fail"
        )
        failure_class = _failure_class(
            returncode=returncode,
            score=score,
            pass_fail=pass_fail,
            stderr=stderr,
        )
    except FileNotFoundError as exc:
        runtime = time.monotonic() - started
        stdout = ""
        stderr = str(exc)
        returncode = 127
        score = None
        ok = False
        pass_fail = "fail"
        failure_class = "harness_missing"
    except subprocess.TimeoutExpired as exc:
        runtime = time.monotonic() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "benchmark suite task timed out"
        returncode = 124
        score = None
        ok = False
        pass_fail = "fail"
        failure_class = "timeout"

    (task_dir / "command.json").write_text(
        json.dumps(
            {
                "commands": [list(command) for command in commands],
                "env": _redacted_env(task_env),
                "returncode": returncode,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    (task_dir / "stdout.txt").write_text(stdout[-20_000:], encoding="utf-8")
    (task_dir / "stderr.txt").write_text(stderr[-20_000:], encoding="utf-8")
    result = BenchmarkSuiteResult(
        id=task.id,
        phase=task.phase,
        harness=task.harness,
        ok=ok and pass_fail == "pass",
        score=score,
        pass_fail=pass_fail,
        runtime_seconds=runtime,
        failure_class=failure_class,
        receipt_path=str(task_dir),
        stdout_tail=stdout[-2000:],
        stderr_tail=stderr[-2000:],
        commands=commands,
        tool_validity="pass"
        if task.phase == "agentic" and ok and pass_fail == "pass"
        else "not_applicable",
    )
    (task_dir / "result.json").write_text(
        json.dumps(asdict(result), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return result


def _suite_event(receipt_path: Path, event_type: str, data: dict[str, Any]) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "type": event_type,
        "data": data,
    }
    with (receipt_path / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _display_command(command: tuple[str, ...]) -> list[str]:
    return [Path(part).name if index == 0 else part for index, part in enumerate(command)]


def _resolve_local_command(command: tuple[str, ...]) -> tuple[str, ...]:
    """Run in-repo Python benchmark helpers even when `uv` is absent from PATH."""
    if len(command) < 4:
        return command
    executable = Path(command[0]).name.lower()
    if executable not in {"uv", "uv.exe"} or command[1] != "run" or shutil.which(command[0]):
        return command
    python_index = _uv_run_python_index(command)
    if python_index is None:
        return command
    return (sys.executable, *command[python_index + 1 :])


def _uv_run_python_index(command: tuple[str, ...]) -> int | None:
    options_with_values = {
        "--active",
        "--all-extras",
        "--extra",
        "--group",
        "--no-dev",
        "--with",
        "--with-editable",
        "--with-requirements",
        "--isolated",
        "--python",
        "--project",
        "--directory",
        "--env-file",
    }
    index = 2
    while index < len(command):
        token = command[index]
        if token == "python":
            return index
        if token in options_with_values and index + 1 < len(command):
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return None
    return None


def _remaining_timeout(deadline: float | None, task_timeout_seconds: int) -> float:
    if deadline is None:
        return float(task_timeout_seconds)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise subprocess.TimeoutExpired(cmd="<benchmark-suite-budget>", timeout=0)
    return max(0.001, min(float(task_timeout_seconds), remaining))


def _score_from_task_output(
    task: BenchmarkSuiteTask,
    *,
    stdout: str,
    task_dir: Path,
) -> float | None:
    if task.score_file is not None:
        score_path = Path(
            _expand_token(
                task.score_file,
                plan=None,
                runs_root=None,
                receipt_path=None,
                task_dir=task_dir,
            )
        )
        if not score_path.is_absolute() and not score_path.exists():
            score_path = task_dir / score_path
        if score_path.exists():
            score = _score_from_json_text(score_path.read_text(encoding="utf-8"), task=task)
            if score is not None:
                return score
    for line in reversed(stdout.splitlines()):
        score = _score_from_json_text(line, task=task)
        if score is not None:
            return score
    return _score_from_json_text(stdout, task=task)


def _redacted_env(env: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        if any(secret in key.upper() for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def _score_from_json_text(text: str, task: BenchmarkSuiteTask | None = None) -> float | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if task is not None and task.harness == "librarian-suite" and isinstance(payload, dict):
        score = payload.get("agent_bench_score")
        return float(score) if _is_number(score) else None
    return _find_score(payload)


def _find_score(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key in SCORE_KEYS:
            value = payload.get(key)
            if _is_number(value):
                return float(value)
        for value in payload.values():
            score = _find_score(value)
            if score is not None:
                return score
    if isinstance(payload, list):
        scores = [score for item in payload if (score := _find_score(item)) is not None]
        if scores:
            return sum(scores) / len(scores)
    return None


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _passes_threshold(score: float, min_score: float | None) -> bool:
    return min_score is None or score >= min_score


def _failure_class(
    *,
    returncode: int,
    score: float | None,
    pass_fail: str,
    stderr: str,
) -> str:
    if returncode != 0:
        return "crash"
    if score is None:
        return "no_score"
    if pass_fail != "pass":
        return "below_threshold"
    if "tool" in stderr.lower() and "invalid" in stderr.lower():
        return "tool_invalid"
    return "none"


def _append_phase_ledger(
    runs_root: Path,
    plan: BenchmarkSuitePlan,
    result: BenchmarkSuiteResult,
) -> None:
    ledger = runs_root / (GENERAL_LEDGER if result.phase == "general" else AGENTIC_LEDGER)
    if not ledger.exists():
        if result.phase == "general":
            ledger.write_text(
                (
                    "run_id\tmodel\tcontext\tsettings_json\tbenchmark_id\tharness\t"
                    "score\tpass_fail\truntime_seconds\treceipt\tfailure_class\n"
                ),
                encoding="utf-8",
            )
        else:
            ledger.write_text(
                (
                    "run_id\tmodel\tcontext\tsettings_json\ttask_id\tharness\t"
                    "score\tpass_fail\truntime_seconds\tlatency_seconds\ttool_validity\t"
                    "receipt\tfailure_class\n"
                ),
                encoding="utf-8",
            )
    settings_json = json.dumps(plan.settings, sort_keys=True, separators=(",", ":"))
    score = "" if result.score is None else f"{result.score:.6f}"
    if result.phase == "general":
        line = "\t".join(
            [
                Path(result.receipt_path).parent.name,
                plan.model,
                str(plan.context),
                settings_json,
                result.id,
                result.harness,
                score,
                result.pass_fail,
                f"{result.runtime_seconds:.6f}",
                result.receipt_path,
                result.failure_class,
            ]
        )
    else:
        line = "\t".join(
            [
                Path(result.receipt_path).parent.name,
                plan.model,
                str(plan.context),
                settings_json,
                result.id,
                result.harness,
                score,
                result.pass_fail,
                f"{result.runtime_seconds:.6f}",
                f"{result.runtime_seconds:.6f}",
                result.tool_validity,
                result.receipt_path,
                result.failure_class,
            ]
        )
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line.replace("\n", " ").replace("\t\t", "\t\t") + "\n")


def _append_agent_score_ledger(runs_root: Path, suite_run: BenchmarkSuiteRun) -> None:
    ledger = runs_root / AGENT_SCORE_LEDGER
    if not ledger.exists():
        ledger.write_text(
            (
                "run_id\tmodel\tcontext\tagent_bench_score\tgeneral_score\tagentic_score\t"
                "pass_fail\treceipt\n"
            ),
            encoding="utf-8",
        )
    line = "\t".join(
        [
            suite_run.run_id,
            suite_run.model,
            str(suite_run.context),
            f"{suite_run.agent_bench_score:.6f}",
            "" if suite_run.general_score is None else f"{suite_run.general_score:.6f}",
            "" if suite_run.agentic_score is None else f"{suite_run.agentic_score:.6f}",
            "pass" if suite_run.ok else "fail",
            suite_run.receipt_path,
        ]
    )
    with ledger.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line + "\n")


def _agent_bench_score(results: list[BenchmarkSuiteResult]) -> float:
    general = _phase_average(results, "general")
    agentic = _phase_average(results, "agentic")
    if general is None or agentic is None:
        return 0.0
    return (general + agentic) / 2.0


def _phase_average(results: list[BenchmarkSuiteResult], phase: str) -> float | None:
    scores = [
        result.score
        for result in results
        if result.phase == phase and result.ok and result.score is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _expand_token(
    value: str,
    *,
    plan: BenchmarkSuitePlan | None,
    runs_root: Path | None,
    receipt_path: Path | None,
    task_dir: Path | None,
) -> str:
    replacements = {
        "model": "" if plan is None else plan.model,
        "context": "" if plan is None else str(plan.context),
        "base_url": "" if plan is None else str(plan.settings.get("base_url", "")),
        "settings_json": ""
        if plan is None
        else json.dumps(
            plan.settings,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "runs_root": "" if runs_root is None else str(runs_root),
        "receipt_dir": "" if receipt_path is None else str(receipt_path),
        "task_dir": "" if task_dir is None else str(task_dir),
    }
    for key, replacement in replacements.items():
        value = value.replace("{" + key + "}", replacement)
    return value


def _new_receipt_dir(runs_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs_root / f"{stamp}-benchmark-suite"
    if not base.exists():
        return base
    index = 2
    while (candidate := runs_root / f"{stamp}-benchmark-suite-{index}").exists():
        index += 1
    return candidate


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return safe[:80] or "task"
