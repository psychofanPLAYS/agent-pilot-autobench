from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RunStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    PARTIAL = "partial"
    NEEDS_MORE_TIME = "needs_more_time"
    FAILED = "failed"
    REJECTED = "rejected"
    CANDIDATE = "candidate"
    CHAMPION_RETEST_NEEDED = "champion_retest_needed"
    SLOW = "slow"
    SPEED_ONLY = "speed_only"
    SERVING_MEASURED = "serving_measured"
    CONTEXT_UNPROVEN = "context_unproven"
    WORKFLOW_UNPROVEN = "workflow_unproven"
    WORKFLOW_WEAK = "workflow_weak"
    WORKFLOW_SMOKE = "workflow_smoke"


@dataclass(frozen=True)
class RunPreset:
    id: str
    label: str
    description: str
    budget_minutes: int
    max_extra_minutes: int
    total_session_cap_minutes: int | None
    max_attempts: int | None
    packs: tuple[str, ...]
    adaptive: bool


PRESETS: dict[str, RunPreset] = {
    "quick": RunPreset(
        id="quick",
        label="Quick Scout",
        description="Does it load and look fast?",
        budget_minutes=5,
        max_extra_minutes=0,
        total_session_cap_minutes=None,
        max_attempts=1,
        packs=("load-smoke", "speed"),
        adaptive=False,
    ),
    "normal": RunPreset(
        id="normal",
        label="Normal",
        description="Good default test.",
        budget_minutes=10,
        max_extra_minutes=3,
        total_session_cap_minutes=None,
        max_attempts=3,
        packs=("load-smoke", "speed", "json-discipline"),
        adaptive=True,
    ),
    "deep": RunPreset(
        id="deep",
        label="Deep Pilot",
        description="Serious agent pilot test.",
        budget_minutes=20,
        max_extra_minutes=10,
        total_session_cap_minutes=None,
        max_attempts=None,
        packs=("hermes-pilot", "context-limit", "tool-calling", "coding-smoke"),
        adaptive=True,
    ),
    "overnight": RunPreset(
        id="overnight",
        label="Overnight",
        description="Let it research while you sleep.",
        budget_minutes=20,
        max_extra_minutes=20,
        total_session_cap_minutes=480,
        max_attempts=None,
        packs=("all",),
        adaptive=True,
    ),
}


@dataclass(frozen=True)
class RunConfig:
    preset_id: str
    budget_minutes: int
    max_extra_minutes: int
    total_session_cap_minutes: int | None
    max_attempts: int | None
    packs: tuple[str, ...]
    adaptive: bool
    min_ttft_target_ms: int = 10_000
    min_generation_tps: float = 20.0
    require_full_gpu_offload: bool = True
    require_no_swap: bool = True
    status: RunStatus = RunStatus.PENDING
    extension_reasons: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_preset(cls, preset_id: str) -> "RunConfig":
        preset = PRESETS[preset_id]
        return cls(
            preset_id=preset.id,
            budget_minutes=preset.budget_minutes,
            max_extra_minutes=preset.max_extra_minutes,
            total_session_cap_minutes=preset.total_session_cap_minutes,
            max_attempts=preset.max_attempts,
            packs=preset.packs,
            adaptive=preset.adaptive,
        )

    def should_allow_extension(self, unfinished_required_pack: bool, healthy: bool) -> bool:
        return self.adaptive and self.max_extra_minutes > 0 and unfinished_required_pack and healthy
