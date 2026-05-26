import json

from gguf_limit_bench.reports import build_leaderboard, write_leaderboard


def _write_run(root, name, score, generation, failure="unknown", context=0):
    run = root / name
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": f"G:/AI/models/{name}.gguf",
                "settings": {
                    "context_size": context,
                    "parallel": 1,
                    "gpu_layers": 99,
                    "batch_size": 2048,
                    "ubatch_size": 512,
                    "flash_attention": True,
                    "kv_unified": True,
                },
                "result": {
                    "ok": failure == "unknown",
                    "generation_tokens_per_second": generation,
                    "prompt_tokens_per_second": 900.0,
                    "ttft_ms": None,
                    "context_size": context,
                    "failure": failure,
                    "stdout": "",
                    "stderr": "",
                    "returncode": 0 if failure == "unknown" else 1,
                    "workflow_score": 0.0,
                    "workflow_results": [],
                },
                "score": score,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_build_leaderboard_ranks_successes_and_explains_context_zero(tmp_path):
    _write_run(tmp_path, "slow", 10.0, 10.0)
    _write_run(tmp_path, "fast", 50.0, 50.0)
    _write_run(tmp_path, "broken", -10000.0, 0.0, failure="model_load")

    leaderboard = build_leaderboard(tmp_path)

    assert leaderboard.entries[0].model_name == "fast.gguf"
    assert leaderboard.entries[0].status == "PASS"
    assert leaderboard.entries[0].context_label == "default/unset"
    assert leaderboard.entries[-1].status == "LOAD FAIL"


def test_write_leaderboard_writes_markdown_and_champion_json(tmp_path):
    _write_run(tmp_path, "winner", 99.0, 90.0)

    leaderboard = write_leaderboard(tmp_path)

    assert (tmp_path / "leaderboard.md").exists()
    assert (tmp_path / "champion.json").exists()
    champion = json.loads((tmp_path / "champion.json").read_text(encoding="utf-8"))
    assert champion["model_name"] == "winner.gguf"
    assert leaderboard.champion.model_name == "winner.gguf"


def test_write_leaderboard_handles_missing_runs_folder(tmp_path):
    runs_root = tmp_path / "missing-runs"

    leaderboard = write_leaderboard(runs_root)

    assert leaderboard.entries == []
    assert (runs_root / "leaderboard.md").exists()
