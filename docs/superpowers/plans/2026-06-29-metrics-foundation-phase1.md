# Metrics Foundation (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the sync-ready metrics foundation — a pure `metrics.py` that computes the category-weighted **Agent Index** (with gate must-pass, coverage, and confidence band), standardized K-sample median+IQR aggregation for all metrics, efficiency derivations (tok/s-per-GB, tokens/Joule), and a normalized `metrics.json` writer — plus a telemetry peak-VRAM/energy sampler, wired into receipt generation.

**Architecture:** A new pure module `src/gguf_limit_bench/metrics.py` holds all computation (no I/O except the final writer). It consumes data already present in receipts (`results.json` pack accuracies, `best-settings.json` speed, a new energy sidecar). `telemetry.py` gains a `PeakEnergySampler` that tracks peak VRAM and integrates power into Joules over a generation window, with injectable clock/sampler for tests. `run_report.py` calls a single `write_run_metrics(receipt_dir)` entry point. Charts/pages (Phase 2) consume `metrics.json` later — not in this plan.

**Tech Stack:** Python 3.11+, stdlib `statistics`, dataclasses, pytest. No new dependencies (`nvidia-ml-py`/`nvidia-smi` already present).

**Spec:** `docs/superpowers/specs/2026-06-29-world-class-results-page-design.md` (§3 metrics model, §5 architecture, §6 testing).

---

## File Structure

- **Create** `src/gguf_limit_bench/metrics.py` — pure metrics: `aggregate_samples`, `agent_index`, efficiency functions, record builder/writer, `write_run_metrics`.
- **Create** `tests/test_metrics.py` — unit tests for every pure function.
- **Modify** `src/gguf_limit_bench/telemetry.py` — add `PeakEnergySampler`.
- **Modify** `tests/test_telemetry.py` (create if absent) — sampler tests.
- **Modify** `src/gguf_limit_bench/run_report.py` — call `write_run_metrics(receipt_path)` from `write_itemized_run_report`.
- **Modify** `tests/test_run_report.py` (create if absent) — integration test that `metrics.json` is written.

Shared constants/signatures (defined in Task 1, referenced everywhere):

```python
SCHEMA_VERSION = 1
GATE_SIGNAL = "librarian-gate"
GATE_THRESHOLD = 0.5
GATE_FAIL_CAP = 40.0
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
```

---

## Task 1: Module scaffold + `aggregate_samples` (K-sample median/IQR)

**Files:**
- Create: `src/gguf_limit_bench/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from gguf_limit_bench.metrics import SampleStats, aggregate_samples


def test_aggregate_three_samples_is_replicated():
    stats = aggregate_samples([10.0, 12.0, 14.0])
    assert stats.n == 3
    assert stats.median == 12.0
    assert stats.iqr_low == 10.0
    assert stats.iqr_high == 14.0
    assert stats.status == "replicated"


def test_aggregate_single_sample_is_unreplicated():
    stats = aggregate_samples([7.5])
    assert stats == SampleStats(n=1, median=7.5, iqr_low=7.5, iqr_high=7.5, status="unreplicated")


def test_aggregate_empty_is_empty_status():
    stats = aggregate_samples([])
    assert stats.n == 0
    assert stats.status == "empty"
    assert stats.median == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'aggregate_samples'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gguf_limit_bench/metrics.py
"""Sync-ready metrics: Agent Index, sample aggregation, efficiency, record writer."""

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
    cleaned = [float(v) for v in values]
    if not cleaned:
        return SampleStats(n=0, median=0.0, iqr_low=0.0, iqr_high=0.0, status="empty")
    ordered = sorted(cleaned)
    median = statistics.median(ordered)
    status = "replicated" if len(ordered) >= 3 else "unreplicated"
    return SampleStats(
        n=len(ordered),
        median=median,
        iqr_low=_quantile(ordered, 0.25),
        iqr_high=_quantile(ordered, 0.75),
        status=status,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): K-sample median/IQR aggregation with replication status"
```

---

## Task 2: `agent_index` — category-weighted composite with gate + coverage

**Files:**
- Modify: `src/gguf_limit_bench/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py  (append)
from gguf_limit_bench.metrics import AgentIndex, agent_index

# Every intended signal at 1.0 -> index 100, full coverage, gate passed.
FULL_SIGNALS = {
    "librarian-gate": 1.0,
    "librarian-triage": 1.0, "librarian-write-entry": 1.0, "suite_agentic": 1.0,
    "librarian-rerank": 1.0, "librarian-compress": 1.0, "librarian-dedupe": 1.0,
    "librarian-contradiction": 1.0, "suite_general": 1.0,
}


def test_full_signals_score_100():
    idx = agent_index(FULL_SIGNALS)
    assert round(idx.value, 6) == 100.0
    assert idx.gate_passed is True
    assert idx.capped is False
    assert round(idx.coverage, 6) == 1.0


def test_category_weighting_is_applied():
    # Only agentic-execution signals present, all 1.0 -> renormalized to that one
    # category -> 100. Coverage reflects 3 of 9 intended signals.
    signals = {"librarian-gate": 1.0, "suite_agentic": 1.0,
               "librarian-triage": 1.0, "librarian-write-entry": 1.0}
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
    assert idx.gate_passed is True   # absent gate = not a failure
    assert idx.capped is False


def test_confidence_band_from_samples():
    samples = {k: [v, v, v] for k, v in FULL_SIGNALS.items()}
    samples["suite_agentic"] = [0.6, 0.8, 1.0]
    idx = agent_index(FULL_SIGNALS, samples=samples)
    assert idx.ci_low is not None and idx.ci_high is not None
    assert idx.ci_low <= idx.value <= idx.ci_high
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -k agent_index -v` (and the new tests by name)
Expected: FAIL with `ImportError: cannot import name 'agent_index'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gguf_limit_bench/metrics.py  (append)

_INTENDED_SIGNALS = tuple(
    sig for sigs in CATEGORY_SIGNALS.values() for sig in sigs
) + (GATE_SIGNAL,)


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
    quality_signals = {k: float(v) for k, v in signals.items() if k != GATE_SIGNAL}
    raw, subscores = _index_value(quality_signals, weights)

    gate_value = signals.get(GATE_SIGNAL)
    gate_passed = gate_value is None or float(gate_value) >= gate_threshold
    capped = gate_value is not None and float(gate_value) < gate_threshold
    value = min(raw, GATE_FAIL_CAP) if capped else raw

    measured = sum(1 for s in _INTENDED_SIGNALS if s in signals)
    coverage = measured / len(_INTENDED_SIGNALS)

    ci_low: float | None = None
    ci_high: float | None = None
    if samples:
        lows: dict[str, float] = {}
        highs: dict[str, float] = {}
        for sig, vals in samples.items():
            if sig == GATE_SIGNAL:
                continue
            stats = aggregate_samples(vals)
            lows[sig] = stats.iqr_low
            highs[sig] = stats.iqr_high
        low_raw, _ = _index_value({**quality_signals, **lows}, weights)
        high_raw, _ = _index_value({**quality_signals, **highs}, weights)
        ci_low = min(low_raw, GATE_FAIL_CAP) if capped else low_raw
        ci_high = min(high_raw, GATE_FAIL_CAP) if capped else high_raw

    return AgentIndex(
        value=value,
        gate_passed=gate_passed,
        capped=capped,
        coverage=coverage,
        category_subscores=subscores,
        components={k: float(v) for k, v in signals.items()},
        ci_low=ci_low,
        ci_high=ci_high,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (all tests including Task 1).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): category-weighted Agent Index with gate, coverage, CI band"
```

---

## Task 3: Efficiency + standardized speed derivations

**Files:**
- Modify: `src/gguf_limit_bench/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py  (append)
from gguf_limit_bench.metrics import (
    inter_token_latency_ms,
    quality_per_gb,
    tokens_per_gb,
    tokens_per_joule,
    total_time_for_100_tokens_s,
)


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
    # ttft 0.5s + 100 tokens / 50 tps = 0.5 + 2.0 = 2.5s
    assert total_time_for_100_tokens_s(0.5, 50.0) == 2.5
    assert total_time_for_100_tokens_s(0.5, 0.0) is None


def test_inter_token_latency_ms():
    assert inter_token_latency_ms(50.0) == 20.0
    assert inter_token_latency_ms(0.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -k "per_gb or joule or token" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gguf_limit_bench/metrics.py  (append)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): efficiency + standardized speed derivations"
```

---

## Task 4: Metrics record builder + writer

**Files:**
- Modify: `src/gguf_limit_bench/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py  (append)
import json as _json
from pathlib import Path

from gguf_limit_bench.metrics import SCHEMA_VERSION, build_metrics_record, write_metrics


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
    _json.dumps(record)  # must not raise


def test_write_metrics_round_trips(tmp_path: Path):
    record = build_metrics_record(
        model_fingerprint={"name": "m"}, machine_fingerprint={"gpu": "g"},
        run_id="r", timestamp="t", tool_versions={},
        agent_index=agent_index(FULL_SIGNALS), speed={}, efficiency={},
    )
    out = write_metrics(tmp_path, record)
    assert out == tmp_path / "metrics.json"
    assert _json.loads(out.read_text(encoding="utf-8"))["run_id"] == "r"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -k "record or write_metrics" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gguf_limit_bench/metrics.py  (append)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): sync-ready metrics record builder and writer"
```

---

## Task 5: `PeakEnergySampler` in telemetry (peak VRAM + Joules)

**Files:**
- Modify: `src/gguf_limit_bench/telemetry.py`
- Test: `tests/test_telemetry.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_telemetry.py
from gguf_limit_bench.telemetry import PeakEnergySampler, TelemetrySnapshot


def _snap(used, power):
    return TelemetrySnapshot(
        ram_available_mb=0, ram_used_percent=0.0,
        gpu_used_mb=used, gpu_total_mb=24000, gpu_power_watts=power,
    )


def test_peak_vram_tracks_max():
    times = iter([0.0, 1.0, 2.0])
    snaps = iter([_snap(1000, 100.0), _snap(3000, 100.0), _snap(2000, 100.0)])
    sampler = PeakEnergySampler(sampler=lambda: next(snaps), clock=lambda: next(times))
    sampler.sample(); sampler.sample(); sampler.sample()
    assert sampler.peak_vram_mb == 3000


def test_energy_integrates_power_over_time():
    # 100W held across two 1s intervals -> ~200 J (trapezoidal).
    times = iter([0.0, 1.0, 2.0])
    snaps = iter([_snap(1000, 100.0), _snap(1000, 100.0), _snap(1000, 100.0)])
    sampler = PeakEnergySampler(sampler=lambda: next(snaps), clock=lambda: next(times))
    sampler.sample(); sampler.sample(); sampler.sample()
    assert round(sampler.energy_joules, 3) == 200.0
    assert round(sampler.duration_s, 3) == 2.0


def test_energy_none_when_power_unavailable():
    times = iter([0.0, 1.0])
    snaps = iter([_snap(1000, None), _snap(1000, None)])
    sampler = PeakEnergySampler(sampler=lambda: next(snaps), clock=lambda: next(times))
    sampler.sample(); sampler.sample()
    assert sampler.energy_joules is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_telemetry.py -v`
Expected: FAIL with `ImportError: cannot import name 'PeakEnergySampler'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gguf_limit_bench/telemetry.py  (append; add `import time` to the imports at top)

class PeakEnergySampler:
    """Track peak VRAM and integrate GPU power into Joules over a window.

    Inject ``sampler`` and ``clock`` for tests; defaults read real telemetry.
    """

    def __init__(self, sampler=sample_telemetry, clock=time.monotonic) -> None:
        self._sampler = sampler
        self._clock = clock
        self._peak_vram_mb: int | None = None
        self._energy_joules = 0.0
        self._saw_power = False
        self._last_time: float | None = None
        self._last_power: float | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None

    def sample(self) -> None:
        snap = self._sampler()
        now = self._clock()
        if self._start_time is None:
            self._start_time = now
        self._end_time = now
        if snap.gpu_used_mb is not None:
            self._peak_vram_mb = (
                snap.gpu_used_mb
                if self._peak_vram_mb is None
                else max(self._peak_vram_mb, snap.gpu_used_mb)
            )
        power = snap.gpu_power_watts
        if power is not None:
            self._saw_power = True
            if self._last_time is not None and self._last_power is not None:
                dt = now - self._last_time
                self._energy_joules += (power + self._last_power) / 2.0 * dt
            self._last_time = now
            self._last_power = power

    @property
    def peak_vram_mb(self) -> int | None:
        return self._peak_vram_mb

    @property
    def energy_joules(self) -> float | None:
        return self._energy_joules if self._saw_power else None

    @property
    def duration_s(self) -> float:
        if self._start_time is None or self._end_time is None:
            return 0.0
        return self._end_time - self._start_time
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_telemetry.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/gguf_limit_bench/telemetry.py tests/test_telemetry.py
git commit -m "feat(telemetry): PeakEnergySampler for peak VRAM and Joules"
```

---

## Task 6: Wire `write_run_metrics` into receipt generation

**Files:**
- Modify: `src/gguf_limit_bench/metrics.py` (add `write_run_metrics`)
- Modify: `src/gguf_limit_bench/run_report.py:28-51` (call it from `write_itemized_run_report`)
- Test: `tests/test_run_report.py` (create)

`write_run_metrics` reads receipt files that already exist (`best-settings.json`, optional `results.json`, optional `energy.json` sidecar written by the run loop) and produces `metrics.json`. It reuses the pack-reading shape from `reports.py::_load_agent_quality` (packs with `status == "scored"`, `pack_id`, `accuracy`), and maps `librarian-*` accuracies + suite scores into Agent Index signals.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_report.py
import json
from pathlib import Path

from gguf_limit_bench.metrics import write_run_metrics


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_run_metrics_builds_metrics_json(tmp_path: Path):
    _write(tmp_path / "best-settings.json", {
        "model": "C:/models/qwen3.5-7b-Q4_K_M.gguf",
        "status": "workflow_smoke",
        "result": {
            "generation_tokens_per_second": 60.0,
            "prompt_tokens_per_second": 900.0,
            "serving_ttft_ms": 500.0,
            "benchmark_suite_general_score": 0.8,
            "benchmark_suite_agentic_score": 0.7,
        },
        "settings": {"context_size": 8192},
    })
    _write(tmp_path / "results.json", {
        "packs": [
            {"pack_id": "librarian-gate", "status": "scored", "accuracy": 1.0},
            {"pack_id": "librarian-triage", "status": "scored", "accuracy": 0.9},
        ],
    })
    out = write_run_metrics(tmp_path)
    assert out == tmp_path / "metrics.json"
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["agent_index"]["gate_passed"] is True
    assert 0.0 < record["agent_index"]["value"] <= 100.0
    assert record["speed"]["generation_tps"]["median"] == 60.0


def test_write_run_metrics_without_results_json(tmp_path: Path):
    _write(tmp_path / "best-settings.json", {
        "model": "m.gguf", "status": "speed_only",
        "result": {"generation_tokens_per_second": 42.0},
        "settings": {},
    })
    record = json.loads(write_run_metrics(tmp_path).read_text(encoding="utf-8"))
    # No quality signals -> coverage 0, value 0, but speed still recorded.
    assert record["agent_index"]["coverage"] == 0.0
    assert record["speed"]["generation_tps"]["median"] == 42.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'write_run_metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gguf_limit_bench/metrics.py  (append)

_SUITE_SIGNAL_KEYS = {
    "benchmark_suite_general_score": "suite_general",
    "benchmark_suite_agentic_score": "suite_agentic",
}


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
        model_fingerprint={"name": Path(str(best.get("model", ""))).name,
                           "settings": best.get("settings", {})},
        machine_fingerprint={"gpu": energy.get("gpu_name", "unknown")},
        run_id=receipt_dir.name,
        timestamp=str(best.get("timestamp", "")),
        tool_versions={"status": best.get("status", "")},
        agent_index=idx,
        speed=speed,
        efficiency=efficiency,
    )
    return write_metrics(receipt_dir, record)


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_report.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Call it from receipt generation**

In `src/gguf_limit_bench/run_report.py`, inside `write_itemized_run_report` (after the existing `report.html` write at line ~51), add:

```python
    from gguf_limit_bench.metrics import write_run_metrics
    write_run_metrics(receipt_path)
```

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/test_metrics.py tests/test_telemetry.py tests/test_run_report.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add src/gguf_limit_bench/metrics.py src/gguf_limit_bench/run_report.py tests/test_run_report.py
git commit -m "feat(metrics): write metrics.json per receipt from run report"
```

---

## Final verification

- [ ] Run the whole test suite: `pytest -q` — expected: all pass, no regressions.
- [ ] Run `ruff check src/gguf_limit_bench/metrics.py src/gguf_limit_bench/telemetry.py` — expected: clean.
- [ ] Confirm `metrics.json` appears in a fresh receipt after a run (or by calling `write_run_metrics` on an existing `_runs/<id>` dir).

## Out of scope (Phase 2 — separate plan)

- `charts.py` + vendored Chart.js, upgraded `results.html` & `report.html` diagrams, trend maps.
- Wiring `PeakEnergySampler` into the live generation loop to emit `energy.json` (this plan computes from it and reads the sidecar; the emit-site wiring lands with Phase 2's run-loop touch, or as a small follow-up). Until then `metrics.json` records efficiency as `null` gracefully.
- **K-repeat execution:** Phase 1 ships the aggregation machinery (`aggregate_samples`, CI band from samples). Actually *running* each scored signal K=3 times lives in the run/scoring loop and lands when that loop is touched (with the energy emit-site). Until then records are single-sample / `unreplicated` but already store the median+IQR shape, so no schema change is needed when K-repeat turns on.
- Phase 3 online sync.
