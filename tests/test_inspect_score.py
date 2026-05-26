import json

from gguf_limit_bench.inspect_score import extract_inspect_score


def test_extract_inspect_score_reads_accuracy_metric_from_json_logs(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "json_repair.json").write_text(
        json.dumps({"results": {"scores": [{"metrics": {"accuracy": {"value": 0.75}}}]}}),
        encoding="utf-8",
    )

    assert extract_inspect_score(log_dir) == 0.75
