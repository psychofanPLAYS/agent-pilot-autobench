from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlannerSuggestion:
    source_model: str
    action: str
    reason: str
    next_settings: dict[str, Any]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class PlannerDecision:
    accepted: bool
    reason: str


def validate_planner_suggestion(suggestion: PlannerSuggestion) -> PlannerDecision:
    if suggestion.action == "promote_self":
        return PlannerDecision(False, "Self-promotion requires objective metrics and external receipt evidence.")
    if suggestion.action not in {"try_next_settings", "skip_zone", "needs_retest"}:
        return PlannerDecision(False, f"Unsupported planner action: {suggestion.action}")
    if not suggestion.evidence.get("receipt_path"):
        return PlannerDecision(False, "Planner suggestions need a receipt path as evidence.")
    if any(key.lower() in {"delete", "remove", "format", "reset"} for key in suggestion.next_settings):
        return PlannerDecision(False, "Planner suggestions cannot request unsafe system or file actions.")
    return PlannerDecision(True, "Accepted as a measured next-experiment proposal.")
