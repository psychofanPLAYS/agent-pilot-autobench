from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


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


def run_benchmark_suite(
    plan: BenchmarkSuitePlan,
    runs_root: Path,
    timeout_seconds: float | None = None,
) -> BenchmarkSuiteRun:
    runs_root.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
    receipt_path = _new_receipt_dir(runs_root)
    receipt_path.mkdir(parents=True, exist_ok=False)
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
        result = _run_task(
            task,
            plan=plan,
            runs_root=runs_root,
            receipt_path=receipt_path,
            deadline=deadline,
        )
        results.append(result)
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
    return suite_run


def _task_from_dict(payload: dict[str, Any]) -> BenchmarkSuiteTask:
    commands = _commands_from_payload(payload)
    return BenchmarkSuiteTask(
        id=str(payload["id"]),
        phase=str(payload["phase"]),
        harness=str(payload.get("harness", payload["phase"])),
        commands=commands,
        env={str(key): str(value) for key, value in dict(payload.get("env", {})).items()},
        timeout_seconds=int(payload.get("timeout_seconds", 600)),
        min_score=(None if payload.get("min_score") is None else float(payload.get("min_score"))),
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
        pass_fail = "pass" if ok and _passes_threshold(score, task.min_score) else "fail"
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
        if not score_path.is_absolute():
            score_path = task_dir / score_path
        if score_path.exists():
            score = _score_from_json_text(score_path.read_text(encoding="utf-8"))
            if score is not None:
                return score
    for line in reversed(stdout.splitlines()):
        score = _score_from_json_text(line)
        if score is not None:
            return score
    return _score_from_json_text(stdout)


def _redacted_env(env: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in env.items():
        if any(secret in key.upper() for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def _score_from_json_text(text: str) -> float | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
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


def _is_number(value: Any) -> bool:
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
