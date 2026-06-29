import json
from pathlib import Path

from gguf_limit_bench.metrics import write_run_metrics


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_run_metrics_builds_metrics_json(tmp_path: Path):
    _write(
        tmp_path / "best-settings.json",
        {
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
        },
    )
    _write(
        tmp_path / "results.json",
        {
            "packs": [
                {"pack_id": "librarian-gate", "status": "scored", "accuracy": 1.0},
                {"pack_id": "librarian-triage", "status": "scored", "accuracy": 0.9},
            ],
        },
    )
    out = write_run_metrics(tmp_path)
    assert out == tmp_path / "metrics.json"
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["agent_index"]["gate_passed"] is True
    assert 0.0 < record["agent_index"]["value"] <= 100.0
    assert record["speed"]["generation_tps"]["median"] == 60.0


def test_write_run_metrics_without_results_json(tmp_path: Path):
    _write(
        tmp_path / "best-settings.json",
        {
            "model": "m.gguf",
            "status": "speed_only",
            "result": {"generation_tokens_per_second": 42.0},
            "settings": {},
        },
    )
    record = json.loads(write_run_metrics(tmp_path).read_text(encoding="utf-8"))
    assert record["agent_index"]["coverage"] == 0.0
    assert record["speed"]["generation_tps"]["median"] == 42.0
