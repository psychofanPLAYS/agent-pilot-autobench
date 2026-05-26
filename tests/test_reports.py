import json

from gguf_limit_bench.reports import build_leaderboard, write_leaderboard


def _write_run(
    root,
    name,
    score,
    generation,
    failure="none",
    context=0,
    workflow_score=0.0,
    workflow_results=None,
    serving_ttft_ms=None,
    serving_warm_ttft_ms=None,
    serving_warmup_penalty_ms=None,
    serving_server_ready_ms=None,
    serving_cold_start_to_first_token_ms=None,
    serving_tokens_per_second=None,
    agent_bench_score=None,
    benchmark_suite_ok=None,
    benchmark_suite_general_score=None,
    benchmark_suite_agentic_score=None,
    benchmark_suite_receipt=None,
    benchmark_suite_failure=None,
):
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
                    "ok": failure in {"none", "unknown"},
                    "generation_tokens_per_second": generation,
                    "prompt_tokens_per_second": 900.0,
                    "ttft_ms": None,
                    "context_size": context,
                    "failure": failure,
                    "stdout": "",
                    "stderr": "",
                    "returncode": 0 if failure == "unknown" else 1,
                    "workflow_score": workflow_score,
                    "workflow_results": workflow_results or [],
                    "serving_ttft_ms": serving_ttft_ms,
                    "serving_warm_ttft_ms": serving_warm_ttft_ms,
                    "serving_warmup_penalty_ms": serving_warmup_penalty_ms,
                    "serving_server_ready_ms": serving_server_ready_ms,
                    "serving_cold_start_to_first_token_ms": serving_cold_start_to_first_token_ms,
                    "serving_tokens_per_second": serving_tokens_per_second,
                    "agent_bench_score": agent_bench_score,
                    "benchmark_suite_ok": benchmark_suite_ok,
                    "benchmark_suite_general_score": benchmark_suite_general_score,
                    "benchmark_suite_agentic_score": benchmark_suite_agentic_score,
                    "benchmark_suite_receipt": benchmark_suite_receipt,
                    "benchmark_suite_failure": benchmark_suite_failure,
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
    assert leaderboard.entries[0].status == "SPEED ONLY"
    assert leaderboard.entries[0].context_label == "unset (speed-only)"
    assert leaderboard.entries[-1].status == "LOAD FAIL"


def test_leaderboard_marks_serving_measured_when_ttft_exists_without_context(tmp_path):
    _write_run(tmp_path, "served-no-context", 50.0, 40.0, serving_ttft_ms=250.0)

    entry = build_leaderboard(tmp_path).entries[0]

    assert entry.status == "SERVING MEASURED"


def test_leaderboard_marks_context_and_workflow_evidence_separately(tmp_path):
    _write_run(tmp_path, "speed-only", 90.0, 80.0, context=0)
    _write_run(tmp_path, "context-only", 100.0, 80.0, context=65536)
    _write_run(
        tmp_path,
        "workflow-weak",
        110.0,
        80.0,
        context=65536,
        workflow_score=2.0,
        workflow_results=[
            {"name": "tool_choice", "passed": True},
            {"name": "safe_plan", "passed": True},
        ],
    )
    _write_run(
        tmp_path,
        "workflow-smoke",
        120.0,
        80.0,
        context=65536,
        workflow_score=4.0,
        workflow_results=[
            {"name": "tool_choice", "passed": True},
            {"name": "safe_plan", "passed": True},
            {"name": "json_repair", "passed": True},
            {"name": "command_safety", "passed": True},
        ],
    )

    entries = {entry.run_id: entry for entry in build_leaderboard(tmp_path).entries}

    assert entries["speed-only"].status == "SPEED ONLY"
    assert entries["context-only"].status == "WORKFLOW UNPROVEN"
    assert entries["workflow-weak"].status == "WORKFLOW WEAK"
    assert entries["workflow-smoke"].status == "WORKFLOW SMOKE"


def test_suite_backed_leaderboard_uses_agent_bench_score_and_status(tmp_path):
    _write_run(tmp_path, "speedy-scout", 500.0, 500.0)
    _write_run(
        tmp_path,
        "suite-backed",
        0.7,
        20.0,
        context=32768,
        agent_bench_score=0.7,
        benchmark_suite_ok=True,
        benchmark_suite_general_score=0.6,
        benchmark_suite_agentic_score=0.8,
        benchmark_suite_receipt="runs/suite-receipt",
    )

    leaderboard = write_leaderboard(tmp_path)
    champion = leaderboard.champion
    markdown = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")

    assert champion.model_name == "suite-backed.gguf"
    assert champion.score == 0.7
    suite_entry = next(entry for entry in leaderboard.entries if entry.run_id == "suite-backed")
    assert suite_entry.score == 0.7
    assert suite_entry.status == "BENCHMARK SUITE"
    assert suite_entry.agent_bench_score == 0.7
    assert suite_entry.benchmark_suite_status == "pass"
    assert "Agent bench score" in markdown
    assert "BENCHMARK SUITE" in markdown


def test_write_leaderboard_writes_markdown_and_champion_json(tmp_path):
    _write_run(tmp_path, "winner", 99.0, 90.0)

    leaderboard = write_leaderboard(tmp_path)

    assert (tmp_path / "leaderboard.md").exists()
    assert (tmp_path / "champion.json").exists()
    assert (tmp_path / "results.html").exists()
    champion = json.loads((tmp_path / "champion.json").read_text(encoding="utf-8"))
    assert champion["model_name"] == "winner.gguf"
    assert leaderboard.champion.model_name == "winner.gguf"


def test_results_html_is_actionable_and_beautiful_enough_to_open(tmp_path):
    _write_run(
        tmp_path,
        "winner",
        99.0,
        90.0,
        context=135936,
        workflow_score=4.0,
        workflow_results=[
            {"name": "tool_choice", "passed": True},
            {"name": "safe_plan", "passed": True},
            {"name": "json_repair", "passed": True},
            {"name": "command_safety", "passed": True},
        ],
    )
    _write_run(tmp_path, "broken", -10000.0, 0.0, failure="model_load")

    write_leaderboard(tmp_path)

    html = (tmp_path / "results.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "Agent Pilot Autobench Results" in html
    assert "Plain-English takeaway" in html
    assert "winner.gguf" in html
    assert "What to do next" in html
    assert "agent-autobench export-profile" in html
    assert "LOAD FAIL" in html
    assert "Evidence" in html


def test_write_leaderboard_handles_missing_runs_folder(tmp_path):
    runs_root = tmp_path / "missing-runs"

    leaderboard = write_leaderboard(runs_root)

    assert leaderboard.entries == []
    assert (runs_root / "leaderboard.md").exists()


def test_leaderboard_markdown_starts_with_plain_english_takeaway(tmp_path):
    _write_run(tmp_path, "winner", 99.0, 90.0)

    write_leaderboard(tmp_path)

    markdown = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert "## Plain-English Takeaway" in markdown
    assert "Best measured model" in markdown


def test_build_leaderboard_recomputes_stale_llama_bench_speed_from_raw_stdout(tmp_path):
    inflated = _write_run(tmp_path, "inflated", 487.90, 475.03)
    payload = json.loads((inflated / "best-settings.json").read_text(encoding="utf-8"))
    payload["result"]["stdout"] = "\n".join(
        [
            json.dumps({"n_prompt": 512, "n_gen": 0, "avg_ts": 1286.674301}),
            json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": 128.792227}),
            json.dumps({"n_prompt": 128, "n_gen": 32, "avg_ts": 475.030187}),
        ]
    )
    (inflated / "best-settings.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_run(tmp_path, "steady", 200.0, 190.0)

    leaderboard = build_leaderboard(tmp_path)

    inflated_entry = next(entry for entry in leaderboard.entries if entry.run_id == "inflated")
    assert inflated_entry.generation_tps == 128.792227
    assert inflated_entry.score < 200.0
    assert leaderboard.champion.model_name == "steady.gguf"


def test_successful_unknown_failure_is_normalized_to_none(tmp_path):
    _write_run(tmp_path, "legacy-success", 60.0, 40.0, failure="unknown")

    entry = build_leaderboard(tmp_path).entries[0]

    assert entry.failure == "none"


def test_leaderboard_surfaces_real_serving_ttft_and_tps(tmp_path):
    _write_run(
        tmp_path,
        "served",
        60.0,
        40.0,
        serving_ttft_ms=750.0,
        serving_warm_ttft_ms=250.0,
        serving_warmup_penalty_ms=500.0,
        serving_tokens_per_second=30.0,
    )

    leaderboard = write_leaderboard(tmp_path)
    entry = leaderboard.entries[0]
    markdown = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")

    assert entry.serving_ttft_ms == 750.0
    assert entry.serving_warm_ttft_ms == 250.0
    assert entry.serving_warmup_penalty_ms == 500.0
    assert entry.serving_tps == 30.0
    assert "750 ms" in markdown
    assert "250 ms" in markdown
    assert "500 ms" in markdown
    assert "30.00 tok/s" in markdown
