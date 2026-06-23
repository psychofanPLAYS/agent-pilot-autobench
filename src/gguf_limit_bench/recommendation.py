"""Turn autoresearch attempts into a flag recommendation.

This is the buyer's answer: instead of a leaderboard of profiles, produce
"use these settings, here is the tradeoff and why." It adapts measured
AttemptResults (including honest *partial* results) into Pareto candidates,
computes the non-dominated frontier, and picks a recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass

from gguf_limit_bench.autoresearch import AttemptResult, _is_partial_result
from gguf_limit_bench.pareto import Objectives, ParetoCandidate, pareto_front, recommend

# Accuracy-first, speed-second — matches the existing simple_bench_score policy.
DEFAULT_OBJECTIVES: Objectives = (("accuracy", "max"), ("tps", "max"))


@dataclass(frozen=True)
class Recommendation:
    recommended: ParetoCandidate | None
    frontier: tuple[ParetoCandidate, ...]
    considered: int
    total: int
    objectives: Objectives
    rationale: str


def _usable(result: AttemptResult) -> bool:
    return result.ok or _is_partial_result(result)


def attempts_to_candidates(attempts: list[AttemptResult]) -> list[ParetoCandidate]:
    """Map ok and partial attempts to Pareto candidates (crashes are dropped)."""
    candidates: list[ParetoCandidate] = []
    for result in attempts:
        if not _usable(result):
            continue
        tps = result.serving_tokens_per_second or result.generation_tokens_per_second or 0.0
        accuracy = result.simple_bench_accuracy if result.simple_bench_accuracy is not None else 0.0
        ttft = result.serving_ttft_ms if result.serving_ttft_ms is not None else result.ttft_ms
        partial = _is_partial_result(result)
        metrics: dict[str, float] = {
            "accuracy": accuracy,
            "tps": tps,
            "prompt_tps": result.prompt_tokens_per_second or 0.0,
            "context": float(result.context_size),
        }
        if ttft is not None:
            metrics["ttft_ms"] = ttft
        candidates.append(
            ParetoCandidate(
                label=result.flag_profile or "unknown",
                metrics=metrics,
                payload={
                    "partial": partial,
                    "context_size": result.context_size,
                    "accuracy": accuracy,
                    "tps": tps,
                    "command": result.launch_command,
                },
            )
        )
    return candidates


def recommend_settings(
    attempts: list[AttemptResult],
    *,
    objectives: Objectives = DEFAULT_OBJECTIVES,
    weights: dict[str, float] | None = None,
) -> Recommendation:
    """Compute the Pareto frontier over attempts and pick a recommendation."""
    candidates = attempts_to_candidates(attempts)
    front = pareto_front(candidates, objectives)
    chosen = recommend(candidates, objectives, weights=weights)
    return Recommendation(
        recommended=chosen,
        frontier=tuple(front),
        considered=len(candidates),
        total=len(attempts),
        objectives=objectives,
        rationale=_build_rationale(chosen, front, len(candidates), len(attempts)),
    )


def _build_rationale(
    chosen: ParetoCandidate | None,
    front: list[ParetoCandidate],
    considered: int,
    total: int,
) -> str:
    if chosen is None:
        return (
            f"Tested {total} profile(s); none produced usable evidence "
            "(all crashed or hit OOM). No recommendation."
        )
    accuracy = chosen.payload.get("accuracy", 0.0)
    tps = chosen.payload.get("tps", 0.0)
    ctx = int(chosen.payload.get("context_size", 0))
    partial_note = " (partial evidence)" if chosen.payload.get("partial") else ""
    lead = (
        f"Tested {total} profile(s); {considered} produced usable evidence. "
        f"Recommended `{chosen.label}`{partial_note}: "
        f"accuracy {accuracy:.0%}, {tps:.0f} tok/s at {ctx // 1024}k context. "
        f"{len(front)} profile(s) sit on the Pareto frontier."
    )
    if len(front) > 1:
        alts = ", ".join(f"`{c.label}`" for c in front if c.label != chosen.label)
        lead += f" Frontier alternatives (different speed/quality tradeoff): {alts}."
    return lead


def render_recommendation_markdown(rec: Recommendation) -> str:
    """Render the recommendation as a markdown block for the report."""
    lines = ["## Recommended settings", "", rec.rationale, ""]
    if rec.recommended is not None and rec.frontier:
        lines.append("| profile | accuracy | tok/s | context | partial |")
        lines.append("|---|---|---|---|---|")
        for cand in rec.frontier:
            star = " ⭐" if cand.label == rec.recommended.label else ""
            acc = cand.payload.get("accuracy", 0.0)
            tps = cand.payload.get("tps", 0.0)
            ctx = int(cand.payload.get("context_size", 0)) // 1024
            partial = "yes" if cand.payload.get("partial") else "no"
            lines.append(f"| `{cand.label}`{star} | {acc:.0%} | {tps:.0f} | {ctx}k | {partial} |")
    return "\n".join(lines) + "\n"
