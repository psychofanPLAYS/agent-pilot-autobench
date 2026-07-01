import json
from pathlib import Path

from gguf_limit_bench.metrics import write_run_metrics
from gguf_limit_bench.run_report import write_itemized_run_report


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


def test_run_report_html_includes_charts(tmp_path: Path):
    _write(
        tmp_path / "best-settings.json",
        {
            "model": "C:/models/qwen3.5-7b.gguf",
            "status": "workflow_smoke",
            "score": 0.86,
            "result": {"ok": True, "generation_tokens_per_second": 60.0, "failure": "none"},
            "settings": {"context_size": 16384},
        },
    )
    _write(
        tmp_path / "context-profile.json",
        {
            "rows": [
                {"context_size": 4096, "generation_tps": 60.0, "cold_ttft_ms": 400, "ok": True},
                {"context_size": 8192, "generation_tps": 50.0, "cold_ttft_ms": 500, "ok": True},
            ]
        },
    )
    _write(
        tmp_path / "results.json",
        {
            "packs": [
                {
                    "pack_id": "librarian-gate",
                    "status": "scored",
                    "accuracy": 1.0,
                    "correct": 4,
                    "wrong": 0,
                    "incomplete": 0,
                },
                {
                    "pack_id": "librarian-triage",
                    "status": "scored",
                    "accuracy": 0.5,
                    "correct": 2,
                    "wrong": 2,
                    "incomplete": 0,
                },
            ]
        },
    )

    write_itemized_run_report(tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "Chart.js v4" in html
    assert "renderChart(" in html
    assert 'id="r-context"' in html
    assert 'id="r-outcomes"' in html
    assert 'id="r-packs"' in html
