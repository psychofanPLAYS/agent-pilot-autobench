import json

from gguf_limit_bench.qe_results import build_qe_leaderboard, write_qe_leaderboard


def _write_qe_summary(run_dir, **overrides):
    run_dir.mkdir(parents=True)
    payload = {
        "model": "qwen-qe-2b",
        "score": 0.88,
        "format_rate": 0.76,
        "direct_answer_rate": 0.0,
        "attempts": 50,
        "answer_max_tokens": 128,
        "sampling": {"temperature": 0.1},
        "median_tps": 200.0,
        "median_ttft_ms": 140.0,
        "resources": {"end": {"gpu_used_mb": 3600}, "delta": {"gpu_util_percent": 12}},
    }
    payload.update(overrides)
    (run_dir / "qe-format-summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_qe_leaderboard_retests_fast_profile_when_format_rate_is_below_gate(tmp_path):
    _write_qe_summary(tmp_path / "runs" / "20260706-qe-fast")

    board = build_qe_leaderboard(tmp_path / "runs")

    assert board.champion is not None
    assert board.champion.action == "RETEST_QE_PROFILE"
    assert board.champion.format_rate == 0.76
    assert board.champion.direct_answer_rate == 0.0
    assert "format rate is below" in board.champion.recommendation
    assert board.champion.resource_summary["end_gpu_used_mb"] == 3600
    assert board.champion.resource_summary["delta_gpu_util_percent"] == 12


def test_qe_leaderboard_promotes_only_clean_high_format_runs(tmp_path):
    _write_qe_summary(
        tmp_path / "runs" / "bad-direct",
        score=0.98,
        format_rate=0.99,
        direct_answer_rate=0.02,
        median_tps=300.0,
    )
    _write_qe_summary(
        tmp_path / "runs" / "best-clean",
        score=0.95,
        format_rate=0.94,
        direct_answer_rate=0.0,
        median_tps=180.0,
    )

    board = build_qe_leaderboard(tmp_path / "runs")

    assert board.champion is not None
    assert board.champion.run_id == "best-clean"
    assert board.champion.action == "PROMOTE_QE_PROFILE"
    assert board.entries[1].action == "REJECT_QE_PROFILE"


def test_write_qe_leaderboard_persists_json_and_markdown(tmp_path):
    _write_qe_summary(tmp_path / "runs" / "candidate")

    board = write_qe_leaderboard(tmp_path / "runs")

    payload = json.loads((tmp_path / "runs" / "qe-format-leaderboard.json").read_text())
    markdown = (tmp_path / "runs" / "qe-format-leaderboard.md").read_text(encoding="utf-8")
    assert payload["champion"]["run_id"] == board.champion.run_id
    assert "QE Format Leaderboard" in markdown
    assert "Top candidate: `qwen-qe-2b`" in markdown
    assert "Champion: `qwen-qe-2b`" not in markdown
    assert "RETEST_QE_PROFILE" in markdown
