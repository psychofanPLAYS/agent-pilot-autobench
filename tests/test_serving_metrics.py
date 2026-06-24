"""Serving-performance metric helpers (vLLM/GuideLLM definitions, computed natively).

These are pure functions over already-collected timing data, so they need no
server. They upgrade the speed program from single-stream tok/s to a real
serving profile: TTFT, TPOT, inter-token latency, end-to-end latency,
percentiles, and goodput-under-SLO.
"""

from __future__ import annotations

import math

import pytest

from gguf_limit_bench.serving_metrics import (
    RequestRecord,
    e2el_ms,
    goodput_fraction,
    inter_token_latencies_ms,
    percentile,
    summarize_latencies,
    tpot_ms,
)


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 90) == pytest.approx(3.7)


def test_percentile_single_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 99) == 42.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_tpot_excludes_first_token():
    # 900 ms of decode over 10 tokens => 9 inter-token gaps => 100 ms/token
    assert tpot_ms(generation_ms=900.0, generated_tokens=10) == pytest.approx(100.0)


def test_tpot_needs_at_least_two_tokens():
    assert tpot_ms(generation_ms=900.0, generated_tokens=1) is None
    assert tpot_ms(generation_ms=900.0, generated_tokens=0) is None


def test_inter_token_latencies_are_consecutive_gaps_in_ms():
    # token arrival timestamps in seconds
    assert inter_token_latencies_ms([0.0, 0.1, 0.25]) == pytest.approx([100.0, 150.0])


def test_inter_token_latencies_too_few_tokens():
    assert inter_token_latencies_ms([0.0]) == []
    assert inter_token_latencies_ms([]) == []


def test_e2el_ms_is_total_wall_clock():
    assert e2el_ms(started=1.0, finished=2.5) == pytest.approx(1500.0)


def test_summarize_latencies_reports_p50_p90_p99():
    values = [float(i) for i in range(1, 101)]  # 1..100
    summary = summarize_latencies(values)
    assert summary["p50"] == pytest.approx(50.5)
    assert summary["p90"] == pytest.approx(90.1)
    assert summary["p99"] == pytest.approx(99.01)
    assert summary["mean"] == pytest.approx(50.5)
    assert summary["max"] == 100.0


def test_summarize_latencies_empty_is_all_none():
    summary = summarize_latencies([])
    assert summary["p50"] is None and summary["p99"] is None and summary["mean"] is None


def test_goodput_fraction_counts_records_meeting_all_slos():
    records = [
        RequestRecord(ttft_ms=100.0, tpot_ms=40.0, e2el_ms=2000.0),  # meets
        RequestRecord(ttft_ms=300.0, tpot_ms=40.0, e2el_ms=2000.0),  # fails ttft
        RequestRecord(ttft_ms=120.0, tpot_ms=80.0, e2el_ms=2000.0),  # fails tpot
    ]
    frac = goodput_fraction(records, ttft_ms_slo=200.0, tpot_ms_slo=50.0)
    assert frac == pytest.approx(1 / 3)


def test_goodput_fraction_no_slo_means_all_pass():
    records = [RequestRecord(ttft_ms=999.0, tpot_ms=999.0, e2el_ms=9999.0)]
    assert goodput_fraction(records) == 1.0


def test_goodput_fraction_missing_metric_fails_that_slo():
    records = [RequestRecord(ttft_ms=None, tpot_ms=10.0, e2el_ms=100.0)]
    # ttft SLO requested but record has no ttft -> does not meet
    assert goodput_fraction(records, ttft_ms_slo=200.0) == 0.0


def test_goodput_fraction_empty_is_zero():
    assert goodput_fraction([], ttft_ms_slo=200.0) == 0.0


def test_request_record_is_frozen():
    rec = RequestRecord(ttft_ms=1.0, tpot_ms=2.0, e2el_ms=3.0)
    with pytest.raises((AttributeError, TypeError)):
        rec.ttft_ms = 5.0  # type: ignore[misc]
    assert not math.isnan(rec.ttft_ms)
