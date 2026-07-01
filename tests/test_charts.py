import json

from gguf_limit_bench import charts

MODELS = [
    {
        "name": "qwen3.5-7b",
        "agent_index": 86.0,
        "gen_tps": 63.0,
        "prompt_tps": 900.0,
        "serving_tps": 40.0,
        "cold_ttft": 500.0,
        "vram_gb": 6.0,
        "pack_scores": {"librarian-gate": 1.0, "librarian-triage": 0.9, "librarian-rerank": 0.8},
        "family": "qwen",
    },
    {
        "name": "gemma3-12b",
        "agent_index": 81.0,
        "gen_tps": 34.0,
        "prompt_tps": 700.0,
        "serving_tps": None,
        "cold_ttft": 800.0,
        "vram_gb": 9.0,
        "pack_scores": {"librarian-gate": 1.0, "librarian-triage": 0.7},
        "family": "gemma",
    },
]
PACK_IDS = ["librarian-gate", "librarian-triage", "librarian-rerank"]


def _is_json(obj) -> None:
    json.dumps(obj)  # raises if any non-serializable value slipped in


def test_runtime_inlines_library_and_helper():
    runtime = charts.chartjs_runtime()
    assert "Chart.js v4" in runtime
    assert "renderChart" in runtime
    assert runtime.count("<script>") == 2


def test_render_chart_emits_canvas_and_init():
    html = charts.render_chart("c1", {"type": "bar", "data": {}}, height=200)
    assert 'id="c1"' in html
    assert "renderChart(" in html
    assert "200px" in html


def test_quality_vs_speed_is_bubble_and_serializable():
    cfg = charts.quality_vs_speed_config(MODELS)
    assert cfg["type"] == "bubble"
    points = cfg["data"]["datasets"][0]["data"]
    assert len(points) == 2
    assert points[0]["x"] == 63.0 and points[0]["y"] == 86.0
    _is_json(cfg)


def test_quality_vs_speed_skips_models_without_index_or_speed():
    models = MODELS + [{"name": "x", "agent_index": None, "gen_tps": 10.0}]
    cfg = charts.quality_vs_speed_config(models)
    assert len(cfg["data"]["datasets"][0]["data"]) == 2


def test_agent_index_bar_sorted_desc():
    cfg = charts.agent_index_bar_config(MODELS)
    assert cfg["data"]["labels"] == ["qwen3.5-7b", "gemma3-12b"]
    assert cfg["data"]["datasets"][0]["data"] == [86.0, 81.0]
    _is_json(cfg)


def test_pack_radar_one_dataset_per_model():
    cfg = charts.pack_radar_config(MODELS, PACK_IDS)
    assert cfg["type"] == "radar"
    assert len(cfg["data"]["datasets"]) == 2
    assert cfg["data"]["labels"] == ["gate", "triage", "rerank"]
    # gemma has no rerank score -> None placeholder keeps axis alignment
    assert cfg["data"]["datasets"][1]["data"][2] is None
    _is_json(cfg)


def test_speed_bars_has_three_series():
    cfg = charts.speed_bars_config(MODELS)
    labels = [d["label"] for d in cfg["data"]["datasets"]]
    assert labels == ["Generation tok/s", "Prompt tok/s", "Serving tok/s"]
    _is_json(cfg)


def test_efficiency_bars_none_without_vram():
    assert charts.efficiency_bars_config([{"name": "a", "agent_index": 50.0}]) is None
    cfg = charts.efficiency_bars_config(MODELS)
    assert cfg is not None
    _is_json(cfg)


def test_index_trend_needs_two_points():
    assert charts.index_trend_config([{"label": "r1", "index": 80.0}]) is None
    cfg = charts.index_trend_config([{"label": "r1", "index": 80.0}, {"label": "r2", "index": 85.0}])
    assert cfg["type"] == "line"
    _is_json(cfg)


def test_context_scaling_two_rows_with_dual_axis():
    rows = [
        {"context_size": 4096, "generation_tps": 60.0, "cold_ttft_ms": 400.0},
        {"context_size": 8192, "generation_tps": 45.0, "cold_ttft_ms": 600.0},
    ]
    cfg = charts.context_scaling_config(rows)
    assert cfg is not None
    assert "y1" in cfg["options"]["scales"]
    assert charts.context_scaling_config(rows[:1]) is None
    _is_json(cfg)


def test_attempts_progression_orders_by_attempt():
    cfg = charts.attempts_progression_config(
        [{"attempt": 2, "score": 1.0}, {"attempt": 1, "score": 0.5}]
    )
    assert cfg["data"]["datasets"][0]["data"] == [0.5, 1.0]
    assert charts.attempts_progression_config([]) is None


def test_outcome_doughnut_none_when_empty():
    assert charts.outcome_doughnut_config(0, 0, 0) is None
    cfg = charts.outcome_doughnut_config(8, 1, 1)
    assert cfg["data"]["datasets"][0]["data"] == [8, 1, 1]
    _is_json(cfg)


def test_pack_accuracy_bars():
    cfg = charts.pack_accuracy_bars_config(
        [{"pack_id": "librarian-gate", "accuracy": 1.0}, {"pack_id": "librarian-triage", "accuracy": 0.5}]
    )
    assert cfg["data"]["labels"] == ["gate", "triage"]
    assert cfg["data"]["datasets"][0]["data"] == [100.0, 50.0]
    assert charts.pack_accuracy_bars_config([]) is None
    _is_json(cfg)
