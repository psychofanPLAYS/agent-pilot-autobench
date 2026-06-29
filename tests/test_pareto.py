"""Pareto-frontier recommender: turn a set of measured profiles into a decision.

A benchmark is only worth money if it ends in "use these settings, here's the
tradeoff" rather than a raw leaderboard. These pure functions compute the
non-dominated frontier over multiple objectives (speed, accuracy, context,
VRAM, ...) and pick a recommendation by weighted preference.
"""

from __future__ import annotations

import pytest

from gguf_limit_bench.pareto import (
    ParetoCandidate,
    dominates,
    pareto_front,
    recommend,
)

# objective spec: (name, direction) with direction in {"max", "min"}
SPEED_QUALITY = (("tps", "max"), ("accuracy", "max"))
SPEED_LATENCY = (("tps", "max"), ("ttft_ms", "min"))


def _c(label, **metrics):
    return ParetoCandidate(label=label, metrics=metrics)


def test_dominates_when_better_on_all_and_strict_on_one():
    a = _c("a", tps=100, ttft_ms=50)
    b = _c("b", tps=90, ttft_ms=60)
    assert dominates(a, b, SPEED_LATENCY) is True
    assert dominates(b, a, SPEED_LATENCY) is False


def test_equal_candidates_do_not_dominate_each_other():
    a = _c("a", tps=100, ttft_ms=50)
    b = _c("b", tps=100, ttft_ms=50)
    assert dominates(a, b, SPEED_LATENCY) is False
    assert dominates(b, a, SPEED_LATENCY) is False


def test_tradeoff_means_neither_dominates():
    a = _c("a", tps=100, ttft_ms=60)  # faster gen, slower first token
    b = _c("b", tps=90, ttft_ms=50)  # slower gen, faster first token
    assert dominates(a, b, SPEED_LATENCY) is False
    assert dominates(b, a, SPEED_LATENCY) is False


def test_pareto_front_drops_dominated_candidates():
    a = _c("a", tps=100, accuracy=0.5)  # frontier
    b = _c("b", tps=50, accuracy=0.9)  # frontier (tradeoff)
    c = _c("c", tps=40, accuracy=0.4)  # dominated by both
    front = pareto_front([a, b, c], SPEED_QUALITY)
    labels = {cand.label for cand in front}
    assert labels == {"a", "b"}


def test_pareto_front_single_candidate():
    a = _c("a", tps=10, accuracy=0.1)
    assert [c.label for c in pareto_front([a], SPEED_QUALITY)] == ["a"]


def test_pareto_front_empty():
    assert pareto_front([], SPEED_QUALITY) == []


def test_recommend_follows_weights():
    a = _c("a", tps=100, accuracy=0.5)
    b = _c("b", tps=50, accuracy=0.9)
    # weight all on speed -> pick the fast one
    assert recommend([a, b], SPEED_QUALITY, weights={"tps": 1.0, "accuracy": 0.0}).label == "a"
    # weight all on accuracy -> pick the accurate one
    assert recommend([a, b], SPEED_QUALITY, weights={"tps": 0.0, "accuracy": 1.0}).label == "b"


def test_recommend_only_considers_frontier():
    a = _c("a", tps=100, accuracy=0.9)  # dominates c
    c = _c("c", tps=10, accuracy=0.1)
    # even weighting everything on accuracy, c is dominated so a still wins
    assert recommend([a, c], SPEED_QUALITY, weights={"tps": 0.0, "accuracy": 1.0}).label == "a"


def test_recommend_default_equal_weights():
    a = _c("a", tps=100, accuracy=0.5)
    b = _c("b", tps=50, accuracy=0.9)
    # min-max normalized, equal weight: a=(1.0+0.0)/2, b=(0.0+1.0)/2 -> tie -> first
    assert recommend([a, b], SPEED_QUALITY).label == "a"


def test_recommend_empty_is_none():
    assert recommend([], SPEED_QUALITY) is None


def test_recommend_carries_payload():
    a = ParetoCandidate(label="a", metrics={"tps": 100, "accuracy": 0.9}, payload={"profile": "L6"})
    rec = recommend([a], SPEED_QUALITY)
    assert rec.payload["profile"] == "L6"


def test_dominates_raises_on_unknown_direction():
    a = _c("a", tps=1)
    b = _c("b", tps=2)
    with pytest.raises(ValueError):
        dominates(a, b, (("tps", "sideways"),))
