from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from gguf_limit_bench.modes import mode_by_id


DEFAULT_FLIGHT_PLAN_ID = "librarian_benchmark"


@dataclass(frozen=True)
class FlightPlan:
    id: str
    label: str
    description: str
    mode_id: str
    budget_minutes: int
    evidence_goal: str
    evidence_class: str
    score_contract: str
    workflow: tuple[str, ...]
    start_label: str
    recommended: bool = False
    advanced: bool = False
    default_benchmark_suite_plan: str | None = None
    suggested_benchmark_suite_plans: tuple[str, ...] = ()

    def to_payload(self, project_root: Path | None = None) -> dict:
        payload = asdict(self)
        payload["workflow"] = list(self.workflow)
        payload["suggested_benchmark_suite_plans"] = [
            _plan_reference(filename, project_root)
            for filename in self.suggested_benchmark_suite_plans
        ]
        payload["default_benchmark_suite_plan"] = _plan_payload_path(
            self.default_benchmark_suite_plan, project_root
        )
        return payload


FLIGHT_PLANS: tuple[FlightPlan, ...] = (
    FlightPlan(
        id="librarian_benchmark",
        label="Librarian benchmark",
        description="Recommended agent-memory test for local models; one model runs, two compare.",
        mode_id="librarian_bench",
        budget_minutes=30,
        evidence_goal="Preflight-gated librarian score, per-pack matrix, receipts, and winner.",
        evidence_class="recommendation",
        score_contract="agent_bench_score",
        workflow=("preflight", "librarian-packs", "bias-checks", "report"),
        start_label="Start librarian benchmark",
        recommended=True,
        suggested_benchmark_suite_plans=(
            "wiki-librarian-gemma4-26b-a4b-thinking.plan.json",
            "wiki-librarian-qwen3-moe-thinking.plan.json",
        ),
    ),
    FlightPlan(
        id="find_best_settings",
        label="Find best settings",
        description="Recommended general run: prove fit, speed, quality, and useful flags.",
        mode_id="best_settings",
        budget_minutes=30,
        evidence_goal="Actionable settings with accuracy-first scoring and report links.",
        evidence_class="recommendation",
        score_contract="simple_bench_score",
        workflow=("preflight", "fit", "speed", "intelligence", "flag-ablation", "report"),
        start_label="Find best settings",
    ),
    FlightPlan(
        id="overnight_campaign",
        label="Overnight campaign",
        description="Long run for serious comparisons when the machine can work unattended.",
        mode_id="deep",
        budget_minutes=60,
        evidence_goal="Deep campaign receipts across fit, speed, quality, flags, and context.",
        evidence_class="recommendation",
        score_contract="agent_bench_score",
        workflow=(
            "preflight",
            "fit",
            "speed",
            "intelligence",
            "flag-ablation",
            "long-context-dropoff",
            "report",
        ),
        start_label="Start overnight campaign",
        advanced=True,
    ),
    FlightPlan(
        id="quick_check",
        label="Quick check",
        description="Fast proof that the model loads and produces a speed receipt.",
        mode_id="quick",
        budget_minutes=5,
        evidence_goal="Load success, rough throughput, and a receipt path.",
        evidence_class="speed_only",
        score_contract="none",
        workflow=("preflight", "speed"),
        start_label="Start quick check",
        advanced=True,
    ),
)


def all_flight_plans() -> tuple[FlightPlan, ...]:
    return FLIGHT_PLANS


def flight_plan_by_id(flight_plan_id: str) -> FlightPlan:
    for plan in FLIGHT_PLANS:
        if plan.id == flight_plan_id:
            return plan
    raise KeyError(f"unknown flight plan: {flight_plan_id}")


def default_flight_plan() -> FlightPlan:
    return flight_plan_by_id(DEFAULT_FLIGHT_PLAN_ID)


def flight_plan_payloads(project_root: Path | None = None) -> list[dict]:
    return [plan.to_payload(project_root=project_root) for plan in FLIGHT_PLANS]


def validate_flight_plan_modes() -> None:
    for plan in FLIGHT_PLANS:
        mode_by_id(plan.mode_id)


def _plan_payload_path(filename: str | None, project_root: Path | None) -> str | None:
    if filename is None:
        return None
    if project_root is None:
        return filename
    return str(project_root / "benchmarks" / "plans" / filename)


def _plan_reference(filename: str, project_root: Path | None) -> dict[str, str]:
    return {
        "filename": filename,
        "path": _plan_payload_path(filename, project_root) or filename,
    }
