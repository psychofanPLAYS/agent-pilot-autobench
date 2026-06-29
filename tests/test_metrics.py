import json as _json
from pathlib import Path

from gguf_limit_bench.metrics import (
    SCHEMA_VERSION,
    AgentIndex,
    SampleStats,
    agent_index,
    aggregate_samples,
    build_metrics_record,
    inter_token_latency_ms,
    quality_per_gb,
    tokens_per_gb,
    tokens_per_joule,
    total_time_for_100_tokens_s,
    write_metrics,
)

# Every intended signal at 1.0 -> index 100, full coverage, gate passed.
FULL_SIGNALS = {
    "librarian-gate": 1.0,
    "librarian-triage": 1.0,
    "librarian-write-entry": 1.0,
    "suite_agentic": 1.0,
    "librarian-rerank": 1.0,
    "librarian-compress": 1.0,
    "librarian-dedupe": 1.0,
    "librarian-contradiction": 1.0,
    "suite_general": 1.0,
}


# --- aggregate_samples -------------------------------------------------------


def test_aggregate_three_samples_is_replicated():
    stats = aggregate_samples([10.0, 12.0, 14.0])
    assert stats.n == 3
    assert stats.median == 12.0
    # Linear-interpolation quantiles (numpy default): P25=11, P75=13.
    assert stats.iqr_low == 11.0
    assert stats.iqr_high == 13.0
    assert stats.status == "replicated"


def test_aggregate_single_sample_is_unreplicated():
    stats = aggregate_samples([7.5])
    assert stats == SampleStats(n=1, median=7.5, iqr_low=7.5, iqr_high=7.5, status="unreplicated")


def test_aggregate_empty_is_empty_status():
    stats = aggregate_samples([])
    assert stats.n == 0
    assert stats.status == "empty"
    assert stats.median == 0.0


# --- agent_index -------------------------------------------------------------


def test_full_signals_score_100():
    idx = agent_index(FULL_SIGNALS)
    assert isinstance(idx, AgentIndex)
    assert round(idx.value, 6) == 100.0
    assert idx.gate_passed is True
    assert idx.capped is False
    assert round(idx.coverage, 6) == 1.0


def test_category_weighting_is_applied():
    signals = {
        "librarian-gate": 1.0,
        "suite_agentic": 1.0,
        "librarian-triage": 1.0,
        "librarian-write-entry": 1.0,
    }
    idx = agent_index(signals)
    assert round(idx.value, 6) == 100.0
    assert round(idx.coverage, 6) == round(4 / 9, 6)
    assert idx.category_subscores["agentic_execution"] == 1.0
    assert idx.category_subscores["retrieval_synthesis"] is None


def test_failed_gate_caps_index():
    signals = dict(FULL_SIGNALS, **{"librarian-gate": 0.0})
    idx = agent_index(signals)
    assert idx.gate_passed is False
    assert idx.capped is True
    assert idx.value <= 40.0


def test_unmeasured_gate_does_not_cap():
    signals = {k: v for k, v in FULL_SIGNALS.items() if k != "librarian-gate"}
    idx = agent_index(signals)
    assert idx.gate_passed is True
    assert idx.capped is False


def test_confidence_band_from_samples():
    samples = {k: [v, v, v] for k, v in FULL_SIGNALS.items()}
    samples["suite_agentic"] = [0.6, 0.8, 1.0]
    idx = agent_index(FULL_SIGNALS, samples=samples)
    assert idx.ci_low is not None and idx.ci_high is not None
    assert idx.ci_low <= idx.value <= idx.ci_high


# --- efficiency + speed derivations -----------------------------------------


def test_tokens_per_gb():
    assert tokens_per_gb(60.0, 8.0) == 7.5
    assert tokens_per_gb(60.0, 0.0) is None


def test_quality_per_gb():
    assert quality_per_gb(80.0, 8.0) == 10.0
    assert quality_per_gb(80.0, 0.0) is None


def test_tokens_per_joule():
    assert tokens_per_joule(1000.0, 250.0) == 4.0
    assert tokens_per_joule(1000.0, 0.0) is None
    assert tokens_per_joule(1000.0, None) is None


def test_total_time_for_100_tokens():
    assert total_time_for_100_tokens_s(0.5, 50.0) == 2.5
    assert total_time_for_100_tokens_s(0.5, 0.0) is None


def test_inter_token_latency_ms():
    assert inter_token_latency_ms(50.0) == 20.0
    assert inter_token_latency_ms(0.0) is None


# --- record builder + writer -------------------------------------------------


def test_build_record_is_json_serializable_and_versioned():
    record = build_metrics_record(
        model_fingerprint={"name": "qwen3.5-7b", "quant": "Q4_K_M", "size_gb": 4.3},
        machine_fingerprint={"gpu": "RTX 4090", "vram_total_mb": 24564},
        run_id="20260629-abc",
        timestamp="2026-06-29T12:00:00Z",
        tool_versions={"pilotbench": "0.1.0", "llama_cpp": "b1234"},
        agent_index=agent_index(FULL_SIGNALS),
        speed={"generation_tps": {"median": 60.0}},
        efficiency={"tokens_per_gb": 7.5},
    )
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["run_id"] == "20260629-abc"
    assert record["agent_index"]["value"] == 100.0
    _json.dumps(record)


def test_write_metrics_round_trips(tmp_path: Path):
    record = build_metrics_record(
        model_fingerprint={"name": "m"},
        machine_fingerprint={"gpu": "g"},
        run_id="r",
        timestamp="t",
        tool_versions={},
        agent_index=agent_index(FULL_SIGNALS),
        speed={},
        efficiency={},
    )
    out = write_metrics(tmp_path, record)
    assert out == tmp_path / "metrics.json"
    assert _json.loads(out.read_text(encoding="utf-8"))["run_id"] == "r"
