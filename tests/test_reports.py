import json

from gguf_limit_bench.reports import (
    build_report_audit,
    build_leaderboard,
    build_model_comparison,
    build_verdict,
    write_leaderboard,
)


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
    promotion_eligible=True,
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
                "promotion_eligible": promotion_eligible,
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


def test_partial_ladder_is_not_eligible_for_global_champion(tmp_path):
    _write_run(tmp_path, "complete", 10.0, 10.0)
    _write_run(tmp_path, "partial-fast", 999.0, 999.0, promotion_eligible=False)

    leaderboard = write_leaderboard(tmp_path)

    assert [entry.model_name for entry in leaderboard.entries] == ["complete.gguf"]
    champion = json.loads((tmp_path / "champion.json").read_text(encoding="utf-8"))
    assert champion["model_name"] == "complete.gguf"


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


def test_verdict_promotes_only_suite_backed_agent_quality(tmp_path):
    _write_run(
        tmp_path,
        "suite-backed",
        0.82,
        55.0,
        context=131072,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=51.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        benchmark_suite_general_score=0.78,
        benchmark_suite_agentic_score=0.86,
    )

    verdict = build_verdict(build_leaderboard(tmp_path))

    assert verdict.action == "PROMOTE"
    assert verdict.confidence == "high"
    assert verdict.champion_model == "suite-backed.gguf"
    assert verdict.agent_quality_score == 0.82
    assert verdict.expected_generation_tps == 55.0
    assert verdict.expected_serving_tps == 51.0
    assert verdict.context_label == "131072"
    assert verdict.prediction["quality"] == "strong"
    assert verdict.prediction["speed"] == "interactive"
    assert verdict.prediction["context"] == "long_agentic"
    assert "suite-backed" in verdict.summary.lower()


def test_verdict_promotes_librarian_agent_quality_without_suite_flag(tmp_path):
    run = _write_run(
        tmp_path,
        "librarian-backed",
        0.68,
        38.0,
        context=98304,
        serving_ttft_ms=900.0,
        serving_tokens_per_second=34.0,
    )
    (run / "results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "status": "scored",
                        "asked": 10,
                        "correct": 8,
                        "incomplete": 0,
                        "accuracy": 0.80,
                    },
                    {
                        "pack_id": "librarian-rerank",
                        "status": "scored",
                        "asked": 10,
                        "correct": 6,
                        "incomplete": 0,
                        "accuracy": 0.60,
                    },
                    {
                        "pack_id": "librarian-compress",
                        "status": "scored",
                        "asked": 10,
                        "correct": 7,
                        "incomplete": 0,
                        "accuracy": 0.70,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    verdict = build_verdict(build_leaderboard(tmp_path))

    assert verdict.action == "PROMOTE"
    assert verdict.confidence == "medium"
    assert verdict.agent_quality_score == 0.70
    assert verdict.prediction["quality"] == "usable"
    assert verdict.prediction["speed"] == "interactive"
    assert "librarian" in verdict.summary.lower()


def test_verdict_retests_tiny_librarian_sample_as_weak_evidence(tmp_path):
    run = _write_run(tmp_path, "tiny-librarian-sample", 0.95, 44.0, context=131072)
    (run / "results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "status": "scored",
                        "asked": 4,
                        "correct": 4,
                        "incomplete": 0,
                        "accuracy": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    leaderboard = build_leaderboard(tmp_path)
    verdict = build_verdict(leaderboard)
    audit = build_report_audit(leaderboard)

    assert leaderboard.champion.agent_bench_score is None
    assert verdict.action == "RETEST"
    assert verdict.confidence == "low"
    assert verdict.prediction["quality"] == "unmeasured"
    assert audit.status == "warning"
    assert audit.warnings[0]["code"] == "weak_agent_quality"
    assert "3 scored packs" in audit.warnings[0]["message"]


def test_model_comparison_ranks_recommendation_grade_librarian_over_tiny_sample(tmp_path):
    weak = _write_run(tmp_path, "tiny-perfect", 0.99, 60.0, context=131072)
    weak.joinpath("results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "status": "scored",
                        "asked": 4,
                        "correct": 4,
                        "incomplete": 0,
                        "accuracy": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    broad = _write_run(tmp_path, "broad-usable", 0.70, 40.0, context=131072)
    broad.joinpath("results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "status": "scored",
                        "asked": 10,
                        "correct": 7,
                        "incomplete": 0,
                        "accuracy": 0.70,
                    },
                    {
                        "pack_id": "librarian-rerank",
                        "status": "scored",
                        "asked": 10,
                        "correct": 7,
                        "incomplete": 0,
                        "accuracy": 0.70,
                    },
                    {
                        "pack_id": "librarian-compress",
                        "status": "scored",
                        "asked": 10,
                        "correct": 7,
                        "incomplete": 0,
                        "accuracy": 0.70,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    comparison = build_model_comparison(build_leaderboard(tmp_path))

    assert comparison.entries[0].model_name == "broad-usable.gguf"
    assert comparison.entries[0].agent_bench_score == 0.70
    assert comparison.entries[1].model_name == "tiny-perfect.gguf"
    assert comparison.entries[1].agent_bench_score is None


def test_verdict_rejects_speed_only_as_not_enough_evidence(tmp_path):
    _write_run(tmp_path, "fast-fit-only", 120.0, 120.0, context=262144)

    verdict = build_verdict(build_leaderboard(tmp_path))

    assert verdict.action == "RETEST"
    assert verdict.confidence == "low"
    assert verdict.agent_quality_score is None
    assert verdict.prediction["quality"] == "unmeasured"
    assert verdict.prediction["recommendation"] == "needs_agent_benchmark"
    assert "benchmark-suite" in verdict.next_run
    assert "not an intelligence result" in verdict.summary.lower()


def test_report_audit_flags_promotion_eligible_speed_only_receipts(tmp_path):
    _write_run(tmp_path, "fast-fit-only", 120.0, 120.0, context=262144)

    audit = build_report_audit(build_leaderboard(tmp_path))

    assert audit.status == "warning"
    assert audit.warning_count == 1
    assert audit.warnings[0]["code"] == "missing_agent_quality"
    assert audit.warnings[0]["run_id"] == "fast-fit-only"
    assert "Speed/context evidence is not enough" in audit.warnings[0]["message"]


def test_report_audit_accepts_suite_and_librarian_quality(tmp_path):
    _write_run(
        tmp_path,
        "suite-backed",
        0.82,
        55.0,
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        benchmark_suite_general_score=0.78,
        benchmark_suite_agentic_score=0.86,
    )
    run = _write_run(tmp_path, "librarian-backed", 0.68, 38.0, context=98304)
    (run / "results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "status": "scored",
                        "asked": 10,
                        "correct": 8,
                        "incomplete": 0,
                        "accuracy": 0.80,
                    },
                    {
                        "pack_id": "librarian-rerank",
                        "status": "scored",
                        "asked": 10,
                        "correct": 6,
                        "incomplete": 0,
                        "accuracy": 0.60,
                    },
                    {
                        "pack_id": "librarian-compress",
                        "status": "scored",
                        "asked": 10,
                        "correct": 7,
                        "incomplete": 0,
                        "accuracy": 0.70,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = build_report_audit(build_leaderboard(tmp_path))

    assert audit.status == "pass"
    assert audit.warning_count == 0


def test_results_html_and_leaderboard_lead_with_benchmark_scores(tmp_path):
    _write_run(
        tmp_path,
        "suite-backed",
        0.82,
        55.0,
        context=131072,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        benchmark_suite_general_score=0.78,
        benchmark_suite_agentic_score=0.86,
    )

    write_leaderboard(tmp_path)

    html = (tmp_path / "results.html").read_text(encoding="utf-8")
    markdown = (tmp_path / "leaderboard.md").read_text(encoding="utf-8")
    assert html.index("Benchmark scores") < html.index("Verdict: PROMOTE")
    assert "Agent bench score" in html
    assert "0.8200" in html
    assert "General score" in html
    assert "0.7800" in html
    assert "Agentic score" in html
    assert "0.8600" in html
    assert "Report Audit" in html
    assert markdown.index("## Benchmark Scores") < markdown.index("## Verdict")
    assert "- Score contract: `agent_bench_score`" in markdown
    assert "- Agent bench score: `0.8200`" in markdown
    assert "- General score: `0.7800`" in markdown
    assert "- Agentic score: `0.8600`" in markdown
    assert "## Verdict" in markdown
    assert "Action: `PROMOTE`" in markdown
    assert "## Recommended Model" in markdown
    assert "## Champion" not in markdown
    assert "## Report Audit" in markdown


def test_write_leaderboard_writes_verdict_artifacts(tmp_path):
    _write_run(
        tmp_path,
        "suite-backed",
        0.75,
        44.0,
        context=65536,
        agent_bench_score=0.75,
        benchmark_suite_ok=True,
        benchmark_suite_general_score=0.70,
        benchmark_suite_agentic_score=0.80,
    )

    write_leaderboard(tmp_path)

    verdict_json = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    verdict_md = (tmp_path / "verdict.md").read_text(encoding="utf-8")
    assert verdict_json["action"] == "PROMOTE"
    assert verdict_json["champion_model"] == "suite-backed.gguf"
    assert verdict_json["prediction"]["quality"] == "strong"
    assert "# pilotBENCHY Verdict" in verdict_md
    assert "Action: `PROMOTE`" in verdict_md


def test_write_leaderboard_writes_report_audit_artifacts(tmp_path):
    _write_run(tmp_path, "fast-fit-only", 120.0, 120.0, context=262144)

    write_leaderboard(tmp_path)

    audit_json = json.loads((tmp_path / "report-audit.json").read_text(encoding="utf-8"))
    audit_md = (tmp_path / "report-audit.md").read_text(encoding="utf-8")
    assert audit_json["status"] == "warning"
    assert audit_json["warning_count"] == 1
    assert "missing_agent_quality" in audit_md


def test_write_leaderboard_writes_markdown_and_champion_json(tmp_path):
    _write_run(tmp_path, "winner", 99.0, 90.0)

    leaderboard = write_leaderboard(tmp_path)

    assert (tmp_path / "leaderboard.md").exists()
    assert (tmp_path / "model-comparison.md").exists()
    assert (tmp_path / "model-comparison.json").exists()
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
    assert "pilotBENCHY Results" in html
    assert "Plain-English takeaway" in html
    assert "winner.gguf" in html
    assert "What to do next" in html
    assert "agent-autobench deployment-readiness" in html
    assert "agent-autobench export-profile" in html
    assert "export is only a deployment" in html
    assert "Candidate settings" in html
    assert "Winning settings" not in html
    assert "LOAD FAIL" in html
    assert "Evidence" in html
    assert "Model comparison" in html
    assert "_runs\\model-comparison.md" in html
    assert "model comparison view" in html
    assert "per-model winner view" not in html
    # Must not carry the old Agent Pilot browser branding.
    assert "Agent Pilot Autobench Results" not in html
    assert "Gemma vs Qwen" not in html


def test_write_leaderboard_handles_missing_runs_folder(tmp_path):
    runs_root = tmp_path / "missing-runs"

    leaderboard = write_leaderboard(runs_root)

    assert leaderboard.entries == []
    assert (runs_root / "leaderboard.md").exists()
    assert json.loads((runs_root / "model-comparison.json").read_text(encoding="utf-8")) == []


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


def test_model_comparison_groups_repeated_runs_by_model_path(tmp_path):
    first = _write_run(tmp_path, "qwen-first", 20.0, 20.0, context=32768)
    payload = json.loads((first / "best-settings.json").read_text(encoding="utf-8"))
    payload["model"] = "G:/AI/models/Qwen-Agent.gguf"
    (first / "best-settings.json").write_text(json.dumps(payload), encoding="utf-8")
    second = _write_run(
        tmp_path,
        "qwen-second",
        40.0,
        40.0,
        context=65536,
        serving_ttft_ms=300.0,
        serving_tokens_per_second=35.0,
    )
    payload = json.loads((second / "best-settings.json").read_text(encoding="utf-8"))
    payload["model"] = "G:/AI/models/Qwen-Agent.gguf"
    (second / "best-settings.json").write_text(json.dumps(payload), encoding="utf-8")
    _write_run(tmp_path, "mistral", 30.0, 30.0, context=32768)

    comparison = build_model_comparison(build_leaderboard(tmp_path))

    qwen = next(entry for entry in comparison.entries if entry.model_name == "Qwen-Agent.gguf")
    assert qwen.run_count == 2
    assert qwen.best_run_id == "qwen-second"
    assert qwen.best_context_label == "65536"
    assert qwen.serving_tps == 35.0


def test_model_comparison_picks_best_agent_score_for_same_model(tmp_path):
    low = _write_run(
        tmp_path,
        "same-model-low-suite",
        0.60,
        55.0,
        agent_bench_score=0.60,
        benchmark_suite_ok=True,
        benchmark_suite_general_score=0.55,
        benchmark_suite_agentic_score=0.65,
    )
    high = _write_run(
        tmp_path,
        "same-model-high-librarian",
        0.90,
        40.0,
        agent_bench_score=0.90,
    )
    low_payload = json.loads((low / "best-settings.json").read_text(encoding="utf-8"))
    high_payload = json.loads((high / "best-settings.json").read_text(encoding="utf-8"))
    high_payload["model"] = low_payload["model"]
    (high / "best-settings.json").write_text(json.dumps(high_payload), encoding="utf-8")

    comparison = build_model_comparison(build_leaderboard(tmp_path))

    assert len(comparison.entries) == 1
    assert comparison.entries[0].best_run_id == "same-model-high-librarian"
    assert comparison.entries[0].agent_bench_score == 0.90


def test_results_html_does_not_call_ranked_unscored_candidate_fastest(tmp_path):
    _write_run(tmp_path, "fast-speed-only", 80.0, 80.0, context=32768)
    _write_run(
        tmp_path,
        "slower-workflow-weak",
        40.0,
        40.0,
        context=65536,
        workflow_score=2.0,
        workflow_results=[{"name": "tool_choice", "passed": True}],
    )

    write_leaderboard(tmp_path)

    html = (tmp_path / "results.html").read_text(encoding="utf-8")
    assert "fastest measured model so far" not in html
    assert "highest-ranked unscored candidate so far" in html


def test_write_leaderboard_writes_model_level_comparison_report(tmp_path):
    _write_run(tmp_path, "winner", 99.0, 90.0, serving_ttft_ms=250.0)

    write_leaderboard(tmp_path)

    markdown = (tmp_path / "model-comparison.md").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "model-comparison.json").read_text(encoding="utf-8"))
    assert "pilotBENCHY Model Comparison" in markdown
    assert "Agent Pilot Model Comparison" not in markdown
    assert "winner.gguf" in markdown
    assert "per-model recommendation" in markdown or "Keep iterating" in markdown
    assert payload[0]["model_name"] == "winner.gguf"
    assert payload[0]["run_count"] == 1
    assert payload[0]["browser_report_path"].endswith("report.html")


def test_results_json_yields_agent_quality_and_pack_scores(tmp_path):
    run = _write_run(tmp_path, "librarian-model", 50.0, 40.0, context=32768)
    (run / "results.json").write_text(
        json.dumps(
            {
                "model": "G:/AI/models/librarian-model.gguf",
                "selection_mode": "rolling",
                "sample_size": 6,
                "gpu": "rtx",
                "recommended_flags": [],
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "tier": "core",
                        "status": "scored",
                        "failure_class": "",
                        "asked": 4,
                        "correct": 3,
                        "wrong": 1,
                        "incomplete": 0,
                        "accuracy": 0.75,
                        "median_tps": 30.0,
                        "median_ttft_ms": 200.0,
                        "questions": [],
                    },
                    {
                        "pack_id": "librarian-dedupe",
                        "tier": "core",
                        "status": "scored",
                        "failure_class": "",
                        "asked": 4,
                        "correct": 1,
                        "wrong": 3,
                        "incomplete": 0,
                        "accuracy": 0.25,
                        "median_tps": 28.0,
                        "median_ttft_ms": 210.0,
                        "questions": [],
                    },
                    {
                        "pack_id": "librarian-compress",
                        "tier": "core",
                        "status": "preflight_fail",
                        "failure_class": "preflight",
                        "asked": 0,
                        "correct": 0,
                        "wrong": 0,
                        "incomplete": 0,
                        "accuracy": 0.0,
                        "median_tps": 0.0,
                        "median_ttft_ms": None,
                        "questions": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    leaderboard = build_leaderboard(tmp_path)
    entry = next(e for e in leaderboard.entries if e.run_id == "librarian-model")

    # Canonical score is weighted over scored attempts, matching librarian_suite.
    assert entry.librarian_score == 0.5
    assert entry.scored_pack_count == 2
    assert entry.pack_scores == {"librarian-gate": 0.75, "librarian-dedupe": 0.25}
    # Raw librarian score is preserved, but weak samples do not become recommendation-grade.
    assert entry.agent_bench_score is None

    comparison = build_model_comparison(leaderboard)
    model = next(e for e in comparison.entries if e.model_name == "librarian-model.gguf")
    assert model.librarian_score == 0.5
    assert model.agent_bench_score is None
    assert model.pack_scores == {"librarian-gate": 0.75, "librarian-dedupe": 0.25}
    assert model.scored_pack_count == 2


def test_results_html_renders_agent_quality_matrix(tmp_path):
    run = _write_run(tmp_path, "librarian-model", 50.0, 40.0, context=32768)
    (run / "results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "librarian-gate",
                        "status": "scored",
                        "asked": 10,
                        "correct": 9,
                        "incomplete": 0,
                        "accuracy": 0.9,
                    },
                    {
                        "pack_id": "librarian-rerank",
                        "status": "scored",
                        "asked": 10,
                        "correct": 4,
                        "incomplete": 0,
                        "accuracy": 0.4,
                    },
                    {
                        "pack_id": "librarian-compress",
                        "status": "scored",
                        "asked": 10,
                        "correct": 8,
                        "incomplete": 0,
                        "accuracy": 0.8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    write_leaderboard(tmp_path)
    html = (tmp_path / "results.html").read_text(encoding="utf-8")
    markdown = (tmp_path / "model-comparison.md").read_text(encoding="utf-8")

    assert "Best model by agent quality" in html
    assert "gate" in html and "rerank" in html
    assert "90%" in html
    assert "Eligible agent score" in markdown
    assert "90%" in markdown


def test_results_html_includes_interactive_charts(tmp_path):
    run = _write_run(
        tmp_path,
        "librarian-model",
        50.0,
        40.0,
        context=32768,
        benchmark_suite_general_score=0.8,
        benchmark_suite_agentic_score=0.7,
    )
    (run / "results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {"pack_id": "librarian-gate", "status": "scored", "accuracy": 1.0},
                    {"pack_id": "librarian-triage", "status": "scored", "accuracy": 0.8},
                    {"pack_id": "librarian-rerank", "status": "scored", "accuracy": 0.6},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(
        json.dumps({"efficiency": {"peak_vram_gb": 6.0}}), encoding="utf-8"
    )

    write_leaderboard(tmp_path)
    html = (tmp_path / "results.html").read_text(encoding="utf-8")

    # Inlined Chart.js runtime + the helper that draws each canvas.
    assert "Chart.js v4" in html
    assert "renderChart(" in html
    # The signature frontier scatter, ranking, radar, and efficiency canvases.
    for canvas_id in ("c-frontier", "c-index", "c-radar", "c-eff"):
        assert f'id="{canvas_id}"' in html
    # KPI strip with the standardized Agent Index headline.
    assert "Agent Index" in html


def test_results_html_renders_index_trend_across_runs(tmp_path):
    for i, (name, acc) in enumerate(
        [("20260621-a", 0.6), ("20260622-b", 0.9)], start=1
    ):
        run = _write_run(tmp_path, name, 50.0 + i, 40.0 + i, context=8192)
        (run / "results.json").write_text(
            json.dumps(
                {
                    "packs": [
                        {"pack_id": "librarian-gate", "status": "scored", "accuracy": 1.0},
                        {"pack_id": "librarian-triage", "status": "scored", "accuracy": acc},
                    ]
                }
            ),
            encoding="utf-8",
        )

    write_leaderboard(tmp_path)
    html = (tmp_path / "results.html").read_text(encoding="utf-8")
    assert 'id="c-trend"' in html
    assert "Agent Index over time" in html


def test_librarian_suite_summary_is_used_when_results_json_missing(tmp_path):
    run = _write_run(tmp_path, "suite-model", 50.0, 40.0, context=32768)
    (run / "librarian-suite-summary.json").write_text(
        json.dumps(
            {
                "librarian_bench_score": 0.6,
                "agent_bench_score": 0.6,
                "accuracy": 0.6,
                "status": "scored",
                "packs": [
                    {"pack_id": "librarian-triage", "status": "scored", "accuracy": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )

    entry = next(e for e in build_leaderboard(tmp_path).entries if e.run_id == "suite-model")
    assert entry.librarian_score == 0.6
    assert entry.pack_scores == {"librarian-triage": 0.6}


def test_agent_quality_uses_librarian_suite_weighted_score_not_pack_mean(tmp_path):
    run = _write_run(tmp_path, "uneven-packs", 50.0, 40.0, context=32768)
    (run / "results.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "pack_id": "large-pack",
                        "status": "scored",
                        "asked": 9,
                        "correct": 9,
                        "incomplete": 0,
                        "accuracy": 1.0,
                    },
                    {
                        "pack_id": "tiny-pack",
                        "status": "scored",
                        "asked": 1,
                        "correct": 0,
                        "incomplete": 0,
                        "accuracy": 0.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    entry = next(e for e in build_leaderboard(tmp_path).entries if e.run_id == "uneven-packs")

    assert entry.librarian_score == 0.9
    assert entry.agent_bench_score is None


def test_leaderboard_excludes_non_generative_models(tmp_path):
    # A real LLM and a stale query-expansion receipt both present.
    _write_run(tmp_path, "Qwen3.5-4B-Q8_0", 120.0, 120.0)
    _write_run(tmp_path, "qmd-query-expansion-qwen3.5-2B.Q8_0", 999.0, 999.0)
    leaderboard = build_leaderboard(tmp_path)
    names = [e.model_name for e in leaderboard.entries]
    assert all("query-expansion" not in n for n in names)
    assert any("Qwen3.5-4B" in n for n in names)
