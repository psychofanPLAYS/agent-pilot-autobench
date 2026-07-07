from __future__ import annotations

from pathlib import Path
import json
import math
import subprocess
from typing import Any

from gguf_limit_bench.autoresearch import (
    AttemptRunner,
    AutoresearchLoop,
    AutoresearchSettings,
)
from gguf_limit_bench.benchmark_suite import BenchmarkSuitePlan, preflight_benchmark_suite
from gguf_limit_bench.receipts import RunReceipt
from gguf_limit_bench.simple_bench import (
    DEFAULT_SIMPLE_BENCH_PATH,
    DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
)
from gguf_limit_bench.simple_bench_runner import LlamaServerSimpleBenchAttemptRunner


DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS = 8192


class BenchmarkSuitePreflightError(ValueError):
    def __init__(self, message: str, *, receipt_path: str) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path


def run_deployment_proof(
    *,
    runs_root: Path,
    profile_id: str = "standard",
    flag_recommendations_path: Path | None = None,
    benchmark_suite_plan: Path | None = None,
    llama_server: Path | None = None,
    simple_bench: Path = DEFAULT_SIMPLE_BENCH_PATH,
    simple_bench_system_prompt: Path = DEFAULT_SIMPLE_BENCH_SYSTEM_PROMPT,
    budget_seconds: int = 1800,
    simple_bench_max_tokens: int = DEFAULT_DEPLOYMENT_SIMPLE_BENCH_MAX_TOKENS,
    attempt_runner: AttemptRunner | None = None,
) -> RunReceipt:
    """Run one exact profile from flag-recommendations.json and write normal APB receipts."""
    flag_path = flag_recommendations_path or runs_root / "flag-recommendations.json"
    flag_payload = _load_flag_recommendations(flag_path)
    profile = _profile_by_id(flag_payload, profile_id)
    model = Path(str(flag_payload.get("model") or ""))
    if not str(model):
        raise ValueError("flag-recommendations.json does not contain a model path.")
    settings = _settings_from_profile(profile)
    resolved_suite = (
        BenchmarkSuitePlan.from_path(benchmark_suite_plan)
        if benchmark_suite_plan is not None
        else None
    )
    if resolved_suite is not None:
        preflight_suite = _suite_plan_for_profile_preflight(
            resolved_suite,
            model=model,
            settings=settings,
        )
        preflight = preflight_benchmark_suite(
            preflight_suite,
            runs_root,
            plan_path=benchmark_suite_plan,
        )
        if not preflight.ok:
            raise BenchmarkSuitePreflightError(
                f"benchmark-suite preflight failed: {preflight.next_action}",
                receipt_path=preflight.receipt_path,
            )
    owns_server_runner = attempt_runner is None
    runner = attempt_runner
    if runner is None:
        if llama_server is None:
            raise ValueError("llama_server is required when no attempt_runner is provided.")
        runner = LlamaServerSimpleBenchAttemptRunner(
            llama_server=llama_server,
            model=model,
            benchmark_path=simple_bench,
            system_prompt_path=simple_bench_system_prompt,
            timeout_seconds=max(60, budget_seconds),
            max_tokens=simple_bench_max_tokens,
            benchmark_suite_plan=resolved_suite,
            runs_root=runs_root,
        )
    command = _deployment_proof_command_record(
        profile_id=profile_id,
        runs_root=runs_root,
        flag_recommendations_path=flag_path,
        benchmark_suite_plan=benchmark_suite_plan,
        budget_seconds=budget_seconds,
        simple_bench_max_tokens=simple_bench_max_tokens,
    )
    loop = AutoresearchLoop(
        model=model,
        runs_root=runs_root,
        attempt_runner=runner,
        budget_seconds=budget_seconds,
        parallel_max=int(profile.get("parallel") or settings.parallel or 1),
        max_attempts=1,
        learner=None,
        benchmark_suite_plan=None if owns_server_runner else resolved_suite,
        candidate_sequence=(settings,),
        resolved_plan={
            "schema_version": 1,
            "program": "deployment-proof",
            "model": str(model),
            "profile_id": profile_id,
            "flag_recommendations": str(flag_path),
            "benchmark_suite_plan": (
                None if benchmark_suite_plan is None else str(benchmark_suite_plan)
            ),
            "budget_seconds": budget_seconds,
            "simple_bench_max_tokens": simple_bench_max_tokens,
            "settings": settings.to_dict(),
            "selected_profile": profile,
        },
        commands=(command,),
    )
    return loop.run()


def _suite_plan_for_profile_preflight(
    plan: BenchmarkSuitePlan,
    *,
    model: Path,
    settings: AutoresearchSettings,
) -> BenchmarkSuitePlan:
    suite_settings = {
        **plan.settings,
        **settings.to_dict(),
        "gguf_model_path": str(model),
        "score_contract": "agent_bench_score",
    }
    if "base_url" not in suite_settings:
        suite_settings["base_url"] = str(plan.settings.get("base_url") or "")
    return BenchmarkSuitePlan(
        model=str(model),
        context=settings.context_size,
        settings=suite_settings,
        tasks=plan.tasks,
    )


def _load_flag_recommendations(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"flag recommendations not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"flag recommendations are invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("flag recommendations must be a JSON object.")
    return payload


def _profile_by_id(payload: dict[str, Any], profile_id: str) -> dict[str, Any]:
    for profile in payload.get("profiles", []):
        if isinstance(profile, dict) and profile.get("id") == profile_id:
            return profile
    raise ValueError(f"profile `{profile_id}` was not found in flag recommendations.")


def _settings_from_profile(profile: dict[str, Any]) -> AutoresearchSettings:
    settings_payload = profile.get("settings")
    if not isinstance(settings_payload, dict):
        raise ValueError(f"profile `{profile.get('id')}` does not contain settings.")
    allowed = set(AutoresearchSettings.__dataclass_fields__)
    values = {key: value for key, value in settings_payload.items() if key in allowed}
    if "extra_server_args" in values:
        values["extra_server_args"] = tuple(str(arg) for arg in values["extra_server_args"])
    return AutoresearchSettings(**values)


def _deployment_proof_command_record(
    *,
    profile_id: str,
    runs_root: Path,
    flag_recommendations_path: Path,
    benchmark_suite_plan: Path | None,
    budget_seconds: int,
    simple_bench_max_tokens: int,
) -> dict[str, Any]:
    argv = [
        "apb",
        "deployment-proof",
        "--profile",
        profile_id,
        "--runs-root",
        str(runs_root),
        "--flag-recommendations",
        str(flag_recommendations_path),
        "--budget-minutes",
        str(max(1, math.ceil(budget_seconds / 60))),
        "--simple-bench-max-tokens",
        str(simple_bench_max_tokens),
    ]
    if benchmark_suite_plan is not None:
        argv.extend(["--benchmark-suite-plan", str(benchmark_suite_plan)])
    return {
        "argv": argv,
        "display_command": subprocess.list2cmdline(argv),
        "cwd": str(Path.cwd()),
    }
