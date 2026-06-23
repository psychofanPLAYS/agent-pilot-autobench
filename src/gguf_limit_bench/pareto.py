"""Pareto-frontier recommender.

Turns a set of measured profiles into a decision: compute the non-dominated
frontier over multiple objectives (e.g. generation tok/s up, accuracy up,
TTFT down, VRAM down), then pick one by weighted preference. This is the layer
that makes the benchmark worth money -- "use these settings, here is the
tradeoff" instead of a raw leaderboard.

Pure functions; no dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# An objective is (metric_name, direction) with direction in {"max", "min"}.
Objective = tuple[str, str]
Objectives = tuple[Objective, ...]


@dataclass(frozen=True)
class ParetoCandidate:
    label: str
    metrics: dict[str, float]
    payload: dict = field(default_factory=dict)


def _validate_direction(direction: str) -> None:
    if direction not in ("max", "min"):
        raise ValueError(f"objective direction must be 'max' or 'min', got {direction!r}")


def dominates(a: ParetoCandidate, b: ParetoCandidate, objectives: Objectives) -> bool:
    """True iff ``a`` is at least as good as ``b`` on every objective and strictly
    better on at least one (Pareto domination)."""
    at_least_as_good = True
    strictly_better = False
    for name, direction in objectives:
        _validate_direction(direction)
        av = a.metrics[name]
        bv = b.metrics[name]
        if direction == "max":
            ge, gt = av >= bv, av > bv
        else:
            ge, gt = av <= bv, av < bv
        if not ge:
            at_least_as_good = False
        if gt:
            strictly_better = True
    return at_least_as_good and strictly_better


def pareto_front(
    candidates: list[ParetoCandidate], objectives: Objectives
) -> list[ParetoCandidate]:
    """Return the candidates not dominated by any other (order preserved)."""
    front: list[ParetoCandidate] = []
    for candidate in candidates:
        if not any(
            dominates(other, candidate, objectives)
            for other in candidates
            if other is not candidate
        ):
            front.append(candidate)
    return front


def _normalized(value: float, lo: float, hi: float, direction: str) -> float:
    if hi <= lo:
        return 0.0
    span = hi - lo
    return (value - lo) / span if direction == "max" else (hi - value) / span


def recommend(
    candidates: list[ParetoCandidate],
    objectives: Objectives,
    *,
    weights: dict[str, float] | None = None,
) -> ParetoCandidate | None:
    """Pick a recommendation from the Pareto frontier by weighted, min-max
    normalized objective score. Defaults to equal weights. Ties go to the
    earliest candidate. Returns None when there are no candidates."""
    front = pareto_front(candidates, objectives)
    if not front:
        return None

    bounds: dict[str, tuple[float, float]] = {}
    for name, _direction in objectives:
        values = [c.metrics[name] for c in front]
        bounds[name] = (min(values), max(values))

    def score(candidate: ParetoCandidate) -> float:
        total = 0.0
        for name, direction in objectives:
            _validate_direction(direction)
            weight = 1.0 if weights is None else weights.get(name, 0.0)
            lo, hi = bounds[name]
            total += weight * _normalized(candidate.metrics[name], lo, hi, direction)
        return total

    best = front[0]
    best_score = score(best)
    for candidate in front[1:]:
        candidate_score = score(candidate)
        if candidate_score > best_score:
            best, best_score = candidate, candidate_score
    return best
