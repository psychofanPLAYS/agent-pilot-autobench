"""Serving-performance metric helpers.

Definitions follow the vLLM ``bench serve`` / GuideLLM methodology so our
numbers are comparable to the rest of the field, but they are computed natively
from already-collected timing data (no extra dependency, fully offline):

- **TTFT** time to first token (collected upstream, summarized here)
- **TPOT** time per output token, excluding the first token
- **ITL**  inter-token latency, the gap between consecutive tokens
- **E2EL** end-to-end latency
- percentiles (p50/p90/p99) and **goodput** = fraction of requests meeting SLOs

These upgrade the speed program from a single-stream tok/s number to a real
serving profile and enable the concurrency / request-rate programs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RequestRecord:
    """One request's latency metrics (any field may be unknown/None)."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2el_ms: float | None = None


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' method). ``p`` in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence is undefined")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def tpot_ms(*, generation_ms: float, generated_tokens: int) -> float | None:
    """Time per output token, excluding the first token.

    Returns None when fewer than two tokens were produced (no inter-token gap).
    """
    if generated_tokens < 2:
        return None
    return generation_ms / (generated_tokens - 1)


def inter_token_latencies_ms(token_timestamps: list[float]) -> list[float]:
    """Gaps (ms) between consecutive token arrival timestamps (seconds)."""
    return [
        (token_timestamps[i + 1] - token_timestamps[i]) * 1000.0
        for i in range(len(token_timestamps) - 1)
    ]


def e2el_ms(*, started: float, finished: float) -> float:
    """End-to-end latency in milliseconds."""
    return (finished - started) * 1000.0


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    """Return p50/p90/p99/mean/min/max for a latency sample (None if empty)."""
    if not values:
        return {key: None for key in ("p50", "p90", "p99", "mean", "min", "max")}
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def goodput_fraction(
    records: list[RequestRecord],
    *,
    ttft_ms_slo: float | None = None,
    tpot_ms_slo: float | None = None,
    e2el_ms_slo: float | None = None,
) -> float:
    """Fraction of requests meeting every provided SLO threshold.

    A record meets an SLO only if its metric is known (not None) and within the
    threshold. With no SLOs provided, every record passes. Empty input -> 0.0.
    """
    if not records:
        return 0.0
    slos = (
        (ttft_ms_slo, "ttft_ms"),
        (tpot_ms_slo, "tpot_ms"),
        (e2el_ms_slo, "e2el_ms"),
    )
    met = 0
    for record in records:
        ok = True
        for threshold, field_name in slos:
            if threshold is None:
                continue
            value = getattr(record, field_name)
            if value is None or value > threshold:
                ok = False
                break
        if ok:
            met += 1
    return met / len(records)
