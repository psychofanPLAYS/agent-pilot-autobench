"""Sync-ready metrics: Agent Index, sample aggregation, efficiency, record writer.

This module is pure computation (no I/O except the final writers). It consumes
data already present in receipts (``results.json`` pack accuracies,
``best-settings.json`` speed, an optional ``energy.json`` sidecar) and produces a
normalized, schema-versioned ``metrics.json`` that is identical whether stored
locally or POSTed to a future sync service.

Design: see docs/superpowers/specs/2026-06-29-world-class-results-page-design.md §3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Mapping, Sequence

SCHEMA_VERSION = 1
GATE_SIGNAL = "librarian-gate"
GATE_THRESHOLD = 0.5
GATE_FAIL_CAP = 40.0

# Category-weighted composite (mid-2026 AA Intelligence Index v4.1 pattern, mapped
# to agent-worker tasks). Weights sum to 1.0; the gate is a must-pass prerequisite,
# not a weighted term.
DEFAULT_CATEGORY_WEIGHTS = {
    "agentic_execution": 0.38,
    "retrieval_synthesis": 0.30,
    "reasoning_correctness": 0.32,
}
CATEGORY_SIGNALS = {
    "agentic_execution": ("suite_agentic", "librarian-triage", "librarian-write-entry"),
    "retrieval_synthesis": ("librarian-rerank", "librarian-compress", "librarian-dedupe"),
    "reasoning_correctness": ("librarian-contradiction", "suite_general"),
}

_INTENDED_SIGNALS = tuple(sig for sigs in CATEGORY_SIGNALS.values() for sig in sigs) + (
    GATE_SIGNAL,
)

_SUITE_SIGNAL_KEYS = {
    "benchmark_suite_general_score": "suite_general",
    "benchmark_suite_agentic_score": "suite_agentic",
}


# ---------------------------------------------------------------------------
# Sample aggregation (standardized across ALL metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleStats:
    n: int
    median: float
    iqr_low: float
    iqr_high: float
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def aggregate_samples(values: Sequence[float]) -> SampleStats:
    """Median + IQR over K samples. K>=3 -> 'replicated', else 'unreplicated'."""
    cleaned = [float(v) for v in values]
    if not cleaned:
        return SampleStats(n=0, median=0.0, iqr_low=0.0, iqr_high=0.0, status="empty")
    ordered = sorted(cleaned)
    return SampleStats(
        n=len(ordered),
        median=statistics.median(ordered),
        iqr_low=_quantile(ordered, 0.25),
        iqr_high=_quantile(ordered, 0.75),
        status="replicated" if len(ordered) >= 3 else "unreplicated",
    )


# ---------------------------------------------------------------------------
# Agent Index (category-weighted composite, gate must-pass, coverage, CI band)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentIndex:
    value: float
    gate_passed: bool
    capped: bool
    coverage: float
    category_subscores: dict[str, float | None]
    components: dict[str, float]
    ci_low: float | None
    ci_high: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def _index_value(
    signals: Mapping[str, float], weights: Mapping[str, float]
) -> tuple[float, dict[str, float | None]]:
    subscores: dict[str, float | None] = {}
    weighted_sum = 0.0
    active_weight = 0.0
    for category, members in CATEGORY_SIGNALS.items():
        present = [float(signals[m]) for m in members if m in signals]
        if present:
            sub = sum(present) / len(present)
            subscores[category] = sub
            weighted_sum += weights[category] * sub
            active_weight += weights[category]
        else:
            subscores[category] = None
    raw = 100.0 * (weighted_sum / active_weight) if active_weight > 0 else 0.0
    return raw, subscores


def agent_index(
    signals: Mapping[str, float],
    *,
    samples: Mapping[str, Sequence[float]] | None = None,
    weights: Mapping[str, float] = DEFAULT_CATEGORY_WEIGHTS,
    gate_threshold: float = GATE_THRESHOLD,
) -> AgentIndex:
    """Composite 0-100 agent-quality score. Hardware-independent and poolable.

    ``signals`` maps a signal id (pack id or ``suite_*``) to a 0..1 accuracy.
    ``samples`` (optional) maps the same ids to repeated raw values for a CI band.
    """
    quality_signals = {k: float(v) for k, v in signals.items() if k != GATE_SIGNAL}

    # When repeated samples exist for a signal, its central value is the sample
    # median and the CI band is its IQR. Deriving value and CI from the same base
    # guarantees ci_low <= value <= ci_high regardless of any point value passed.
    central = dict(quality_signals)
    lows = dict(quality_signals)
    highs = dict(quality_signals)
    if samples:
        for sig, vals in samples.items():
            if sig == GATE_SIGNAL:
                continue
            stats = aggregate_samples(vals)
            central[sig] = stats.median
            lows[sig] = stats.iqr_low
            highs[sig] = stats.iqr_high

    raw, subscores = _index_value(central, weights)

    gate_value = signals.get(GATE_SIGNAL)
    gate_passed = gate_value is None or float(gate_value) >= gate_threshold
    capped = gate_value is not None and float(gate_value) < gate_threshold
    value = min(raw, GATE_FAIL_CAP) if capped else raw

    measured = sum(1 for s in _INTENDED_SIGNALS if s in signals)
    coverage = measured / len(_INTENDED_SIGNALS)

    ci_low: float | None = None
    ci_high: float | None = None
    if samples:
        low_raw, _ = _index_value(lows, weights)
        high_raw, _ = _index_value(highs, weights)
        ci_low = min(low_raw, GATE_FAIL_CAP) if capped else low_raw
        ci_high = min(high_raw, GATE_FAIL_CAP) if capped else high_raw

    components = dict(central)
    if gate_value is not None:
        components[GATE_SIGNAL] = float(gate_value)

    return AgentIndex(
        value=value,
        gate_passed=gate_passed,
        capped=capped,
        coverage=coverage,
        category_subscores=subscores,
        components=components,
        ci_low=ci_low,
        ci_high=ci_high,
    )


# ---------------------------------------------------------------------------
# Efficiency + standardized speed derivations
# ---------------------------------------------------------------------------


def tokens_per_gb(tps: float, vram_gb: float) -> float | None:
    return None if not vram_gb else float(tps) / float(vram_gb)


def quality_per_gb(index_value: float, vram_gb: float) -> float | None:
    return None if not vram_gb else float(index_value) / float(vram_gb)


def tokens_per_joule(total_tokens: float, energy_joules: float | None) -> float | None:
    if not energy_joules:
        return None
    return float(total_tokens) / float(energy_joules)


def total_time_for_100_tokens_s(ttft_s: float, output_tps: float) -> float | None:
    if not output_tps:
        return None
    return float(ttft_s) + 100.0 / float(output_tps)


def inter_token_latency_ms(output_tps: float) -> float | None:
    if not output_tps:
        return None
    return 1000.0 / float(output_tps)


# ---------------------------------------------------------------------------
# Record builder + writer
# ---------------------------------------------------------------------------


def build_metrics_record(
    *,
    model_fingerprint: dict,
    machine_fingerprint: dict,
    run_id: str,
    timestamp: str,
    tool_versions: dict,
    agent_index: AgentIndex,
    speed: dict,
    efficiency: dict,
) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "timestamp": str(timestamp),
        "model": dict(model_fingerprint),
        "machine": dict(machine_fingerprint),
        "tool_versions": dict(tool_versions),
        "agent_index": agent_index.to_dict(),
        "speed": dict(speed),
        "efficiency": dict(efficiency),
    }
    json.dumps(record)  # fail fast on non-serializable input
    return record


def write_metrics(receipt_dir: Path, record: dict) -> Path:
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    out = receipt_dir / "metrics.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Receipt -> metrics.json entry point
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _signals_from_results(results: dict) -> dict[str, float]:
    signals: dict[str, float] = {}
    for pack in results.get("packs", []) or []:
        if not isinstance(pack, dict) or pack.get("status") != "scored":
            continue
        pack_id, accuracy = pack.get("pack_id"), pack.get("accuracy")
        if pack_id is None or accuracy is None:
            continue
        try:
            signals[str(pack_id)] = float(accuracy)
        except (TypeError, ValueError):
            continue
    return signals


def write_run_metrics(receipt_dir: Path) -> Path:
    """Read a receipt's existing artifacts and write its ``metrics.json``."""
    receipt_dir = Path(receipt_dir)
    best = _read_json(receipt_dir / "best-settings.json") or {}
    results = _read_json(receipt_dir / "results.json") or {}
    energy = _read_json(receipt_dir / "energy.json") or {}
    result = best.get("result", {}) if isinstance(best.get("result"), dict) else {}

    signals = _signals_from_results(results)
    for src_key, sig in _SUITE_SIGNAL_KEYS.items():
        if result.get(src_key) is not None:
            signals[sig] = float(result[src_key])

    idx = agent_index(signals)

    gen_tps = float(result.get("generation_tokens_per_second") or 0.0)
    ttft_ms = result.get("serving_ttft_ms")
    speed = {
        "generation_tps": aggregate_samples([gen_tps]).to_dict() if gen_tps else {},
        "prompt_tps": aggregate_samples(
            [float(result.get("prompt_tokens_per_second") or 0.0)]
        ).to_dict(),
        "ttft_cold_ms": {"median": float(ttft_ms)} if ttft_ms is not None else {},
        "total_time_100_s": total_time_for_100_tokens_s(
            float(ttft_ms) / 1000.0 if ttft_ms is not None else 0.0, gen_tps
        ),
        "inter_token_latency_ms": inter_token_latency_ms(gen_tps),
    }

    vram_gb = (energy.get("peak_vram_mb") or 0) / 1024.0
    total_tokens = float(energy.get("total_tokens") or 0.0)
    efficiency = {
        "peak_vram_gb": vram_gb or None,
        "tokens_per_gb": tokens_per_gb(gen_tps, vram_gb),
        "quality_per_gb": quality_per_gb(idx.value, vram_gb),
        "tokens_per_joule": tokens_per_joule(total_tokens, energy.get("energy_joules")),
    }

    record = build_metrics_record(
        model_fingerprint={
            "name": Path(str(best.get("model", ""))).name,
            "settings": best.get("settings", {}),
        },
        machine_fingerprint={"gpu": energy.get("gpu_name", "unknown")},
        run_id=receipt_dir.name,
        timestamp=str(best.get("timestamp", "")),
        tool_versions={"status": best.get("status", "")},
        agent_index=idx,
        speed=speed,
        efficiency=efficiency,
    )
    return write_metrics(receipt_dir, record)
