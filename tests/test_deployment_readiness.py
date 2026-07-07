import json

from gguf_limit_bench.deployment_readiness import write_deployment_readiness


def _write_flag_recs(root, model="G:/AI/models/Qwen3.6-27B-MTP-Q5_K_M.gguf"):
    (root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": model,
                "model_name": model.rsplit("/", 1)[-1],
                "lane_type": "chat_agent",
                "profiles": [
                    {"id": "standard", "label": "Standard", "context_size": 131072},
                    {"id": "long_agent", "label": "Long agent", "context_size": 200000},
                    {"id": "over_the_top", "label": "Over-the-top", "context_size": 262144},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_run(
    root,
    name,
    *,
    model="G:/AI/models/Qwen3.6-27B-MTP-Q5_K_M.gguf",
    context=131072,
    agent_bench_score=None,
    benchmark_suite_ok=None,
    benchmark_suite_failure="",
    failure="none",
    serving_ttft_ms=None,
    serving_tokens_per_second=None,
    settings=None,
    status=None,
):
    run = root / name
    run.mkdir(parents=True)
    settings_payload = {"context_size": context, "parallel": 1, "gpu_layers": 99}
    if settings:
        settings_payload.update(settings)
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": model,
                "status": status,
                "settings": settings_payload,
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": failure,
                    "agent_bench_score": agent_bench_score,
                    "benchmark_suite_ok": benchmark_suite_ok,
                    "benchmark_suite_failure": benchmark_suite_failure,
                    "serving_ttft_ms": serving_ttft_ms,
                    "serving_tokens_per_second": serving_tokens_per_second,
                },
                "score": agent_bench_score if agent_bench_score is not None else 42.0,
            }
        ),
        encoding="utf-8",
    )
    return run


def test_deployment_readiness_refuses_unproven_flag_recommendations(tmp_path):
    _write_flag_recs(tmp_path)

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "RETEST_DEPLOYMENT"
    assert payload["profiles"][0]["status"] == "UNPROVEN"
    assert "No matching scored receipt" in payload["profiles"][0]["reason"]
    assert "profile `standard`" in payload["next_run"]


def test_deployment_readiness_promotes_only_profiles_with_score_context_and_serving(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "standard-proven",
        context=131072,
        agent_bench_score=0.82,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    events = tmp_path / "standard-proven" / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "autoresearch_attempt_started",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 12000,
                                "gpu_total_mb": 24564,
                                "gpu_util_percent": 30,
                                "gpu_power_watts": 210.0,
                                "ram_available_mb": 64000,
                                "ram_used_percent": 48.0,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "autoresearch_attempt_finished",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 17200,
                                "gpu_total_mb": 24564,
                                "gpu_util_percent": 88,
                                "gpu_power_watts": 355.5,
                                "ram_available_mb": 51000,
                                "ram_used_percent": 58.0,
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    statuses = {profile["id"]: profile["status"] for profile in payload["profiles"]}
    evidence = payload["profiles"][0]["evidence"]
    assert payload["action"] == "PROMOTE_DEPLOYMENT_PROFILE"
    assert payload["recommended_profile_id"] == "standard"
    assert statuses["standard"] == "PROVEN"
    assert statuses["long_agent"] == "UNPROVEN"
    assert evidence["resource_summary"] == {
        "max_gpu_used_mb": 17200,
        "gpu_total_mb": 24564,
        "max_gpu_util_percent": 88,
        "max_gpu_power_watts": 355.5,
        "max_ram_used_percent": 58.0,
        "min_ram_available_mb": 51000,
    }
    assert "Deployment Readiness" in readiness.markdown_path.read_text(encoding="utf-8")
    assert "17200/24564 MB" in readiness.markdown_path.read_text(encoding="utf-8")


def test_deployment_readiness_blocks_promotion_on_critical_runtime_warning(tmp_path):
    _write_flag_recs(tmp_path)
    run = _write_run(
        tmp_path,
        "standard-template-warning",
        context=131072,
        agent_bench_score=0.82,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    warning_dir = run / "simplebench-standard"
    warning_dir.mkdir()
    (warning_dir / "warnings.log").write_text(
        "18.26.395.388 W common_chat_try_specialized_template: "
        "detected an outdated gemma4 chat template\n",
        encoding="utf-8",
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    standard = next(profile for profile in payload["profiles"] if profile["id"] == "standard")
    evidence = standard["evidence"]
    assert payload["action"] == "RETEST_DEPLOYMENT"
    assert payload["recommended_profile_id"] is None
    assert standard["status"] == "RUNTIME_WARNING"
    assert "critical warnings" in standard["reason"]
    assert evidence["runtime_warnings"]["warning_count"] == 1
    assert "outdated gemma4" in evidence["runtime_warnings"]["critical"][0]
    assert "1 runtime warning(s)" in readiness.markdown_path.read_text(encoding="utf-8")


def test_deployment_readiness_keeps_quality_winner_but_uses_best_matching_resource_evidence(
    tmp_path,
):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "quality-winner-stale-resource",
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    (tmp_path / "quality-winner-stale-resource" / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "autoresearch_attempt_started",
                "data": {
                    "telemetry": {
                        "gpu_used_mb": 1311,
                        "gpu_total_mb": 24564,
                        "gpu_util_percent": 15,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_run(
        tmp_path,
        "lower-score-live-resource",
        context=131072,
        agent_bench_score=0.74,
        benchmark_suite_ok=True,
        serving_ttft_ms=460.0,
        serving_tokens_per_second=37.0,
    )
    (tmp_path / "lower-score-live-resource" / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "llama_server_ready",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 20221,
                                "gpu_total_mb": 24564,
                                "gpu_util_percent": 84,
                                "ram_used_percent": 95.5,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "benchmark_suite_finished_on_owned_server",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 20254,
                                "gpu_total_mb": 24564,
                                "gpu_util_percent": 86,
                                "gpu_power_watts": 253.28,
                                "ram_used_percent": 96.7,
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    standard = next(profile for profile in payload["profiles"] if profile["id"] == "standard")
    evidence = standard["evidence"]
    assert payload["recommended_profile_id"] == "standard"
    assert payload["recommendation_basis"]["run_id"] == "quality-winner-stale-resource"
    assert evidence["run_id"] == "quality-winner-stale-resource"
    assert evidence["resource_run_id"] == "lower-score-live-resource"
    assert evidence["resource_summary"]["max_gpu_used_mb"] == 20254
    assert evidence["resource_summary"]["max_gpu_util_percent"] == 86
    assert evidence["resource_summary"]["max_ram_used_percent"] == 96.7
    assert "20254/24564 MB" in readiness.markdown_path.read_text(encoding="utf-8")


def test_deployment_readiness_surfaces_failed_profile_attempt_with_resource_evidence(tmp_path):
    _write_flag_recs(tmp_path)
    run = _write_run(
        tmp_path,
        "long-agent-200k-suite-failed-partial",
        context=200000,
        benchmark_suite_ok=False,
        benchmark_suite_failure="gsm8k_cot_zeroshot_smoke:harness_missing",
        failure="benchmark_suite_failed",
        serving_ttft_ms=159.3,
        serving_tokens_per_second=156.4,
        settings={"profile_name": "long_agent"},
        status="partial",
    )
    (run / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "llama_server_ready",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 20868,
                                "gpu_total_mb": 24564,
                                "gpu_util_percent": 61,
                                "ram_used_percent": 92.1,
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "benchmark_suite_finished_on_owned_server",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 20898,
                                "gpu_total_mb": 24564,
                                "gpu_util_percent": 87,
                                "gpu_power_watts": 272.26,
                                "ram_available_mb": 2500,
                                "ram_used_percent": 92.3,
                            }
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    standard = next(profile for profile in payload["profiles"] if profile["id"] == "standard")
    long_agent = next(profile for profile in payload["profiles"] if profile["id"] == "long_agent")
    evidence = long_agent["evidence"]
    assert payload["action"] == "RETEST_DEPLOYMENT"
    assert standard["status"] == "UNPROVEN"
    assert long_agent["status"] == "FAILED_PROOF"
    assert (
        long_agent["reason"]
        == "Matching receipt fit and served, but failed the benchmark-suite proof."
    )
    assert evidence["context"] == 200000
    assert evidence["serving_tps"] == 156.4
    assert evidence["failure"] == "benchmark_suite_failed"
    assert evidence["benchmark_suite_failure"] == "gsm8k_cot_zeroshot_smoke:harness_missing"
    assert evidence["resource_summary"] == {
        "max_gpu_used_mb": 20898,
        "gpu_total_mb": 24564,
        "max_gpu_util_percent": 87,
        "max_gpu_power_watts": 272.26,
        "max_ram_used_percent": 92.3,
        "min_ram_available_mb": 2500,
    }
    assert "`FAILED_PROOF`" in readiness.markdown_path.read_text(encoding="utf-8")


def test_deployment_readiness_next_run_fixes_failed_profile_after_baseline_is_proven(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "standard-proven",
        context=131072,
        agent_bench_score=0.81,
        benchmark_suite_ok=True,
        serving_ttft_ms=430.0,
        serving_tokens_per_second=38.0,
        settings={"profile_name": "standard"},
    )
    _write_run(
        tmp_path,
        "long-agent-200k-suite-failed-partial",
        context=200000,
        benchmark_suite_ok=False,
        benchmark_suite_failure="gsm8k_cot_zeroshot_smoke:harness_missing",
        failure="benchmark_suite_failed",
        serving_ttft_ms=159.3,
        serving_tokens_per_second=156.4,
        settings={"profile_name": "long_agent"},
        status="partial",
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    assert payload["recommended_profile_id"] == "standard"
    assert payload["next_run"] == (
        "Fix `long_agent` proof failure "
        "(gsm8k_cot_zeroshot_smoke:harness_missing) and rerun the profile."
    )


def test_deployment_readiness_recommends_best_proven_profile_by_evidence(tmp_path):
    model = "G:/AI/models/Qwen3.6-27B-MTP-Q5_K_M.gguf"
    (tmp_path / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": model,
                "model_name": model.rsplit("/", 1)[-1],
                "lane_type": "chat_agent",
                "profiles": [
                    {
                        "id": "standard",
                        "label": "Standard",
                        "context_size": 131072,
                        "settings": {"profile_name": "standard", "context_size": 131072},
                    },
                    {
                        "id": "long_agent",
                        "label": "Long agent",
                        "context_size": 200000,
                        "settings": {"profile_name": "long_agent", "context_size": 200000},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_run(
        tmp_path,
        "standard-proven",
        context=131072,
        agent_bench_score=0.81,
        benchmark_suite_ok=True,
        serving_ttft_ms=430.0,
        serving_tokens_per_second=38.0,
    )
    (tmp_path / "standard-proven" / "best-settings.json").write_text(
        json.dumps(
            {
                "model": model,
                "settings": {"profile_name": "standard", "context_size": 131072},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.81,
                    "benchmark_suite_ok": True,
                    "serving_ttft_ms": 430.0,
                    "serving_tokens_per_second": 38.0,
                },
                "score": 0.81,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "long-agent-proven").mkdir()
    (tmp_path / "long-agent-proven" / "best-settings.json").write_text(
        json.dumps(
            {
                "model": model,
                "settings": {"profile_name": "long_agent", "context_size": 200000},
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 36.0,
                    "prompt_tokens_per_second": 820.0,
                    "failure": "none",
                    "agent_bench_score": 0.86,
                    "benchmark_suite_ok": True,
                    "serving_ttft_ms": 520.0,
                    "serving_tokens_per_second": 31.0,
                },
                "score": 0.86,
            }
        ),
        encoding="utf-8",
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    profiles = {profile["id"]: profile for profile in payload["profiles"]}
    markdown = readiness.markdown_path.read_text(encoding="utf-8")
    assert payload["action"] == "PROMOTE_DEPLOYMENT_PROFILE"
    assert payload["recommended_profile_id"] == "long_agent"
    assert payload["recommendation_basis"]["run_id"] == "long-agent-proven"
    assert payload["recommendation_basis"]["agent_bench_score"] == 0.86
    assert payload["recommendation_basis"]["context"] == 200000
    assert (
        profiles["long_agent"]["recommendation_score"]
        > profiles["standard"]["recommendation_score"]
    )
    assert "Recommendation Basis" in markdown
    assert "`long-agent-proven`" in markdown


def test_deployment_readiness_keeps_speed_only_context_as_systems_only(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "speed-only-long-context",
        context=262144,
        agent_bench_score=None,
        serving_ttft_ms=900.0,
        serving_tokens_per_second=20.0,
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    statuses = {profile["id"]: profile["status"] for profile in payload["profiles"]}
    assert payload["action"] == "RETEST_DEPLOYMENT"
    assert statuses["standard"] == "SYSTEMS_ONLY"
    assert statuses["over_the_top"] == "SYSTEMS_ONLY"


def test_deployment_readiness_rejects_suite_failed_receipts_even_with_scores(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "suite-failed",
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=False,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    statuses = {profile["id"]: profile["status"] for profile in payload["profiles"]}
    assert payload["action"] == "RETEST_DEPLOYMENT"
    assert statuses["standard"] == "REJECTED"
    assert "failed benchmark-suite" in payload["profiles"][0]["reason"]


def test_deployment_readiness_requires_exact_profile_settings_when_recorded(tmp_path):
    model = "G:/AI/models/Qwen3.6-27B-MTP-Q5_K_M.gguf"
    (tmp_path / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": model,
                "model_name": model.rsplit("/", 1)[-1],
                "lane_type": "chat_agent",
                "profiles": [
                    {
                        "id": "standard",
                        "label": "Standard",
                        "context_size": 131072,
                        "settings": {
                            "profile_name": "standard",
                            "context_size": 131072,
                            "parallel": 1,
                            "gpu_layers": 99,
                            "batch_size": 2048,
                            "ubatch_size": 512,
                            "flash_attention": True,
                            "kv_unified": True,
                            "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0",
                            "extra_server_args": ["--jinja"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run = tmp_path / "wrong-profile-receipt"
    run.mkdir()
    (run / "best-settings.json").write_text(
        json.dumps(
            {
                "model": model,
                "settings": {
                    "profile_name": "not-standard",
                    "context_size": 131072,
                    "parallel": 1,
                    "gpu_layers": 99,
                    "batch_size": 2048,
                    "ubatch_size": 512,
                    "flash_attention": True,
                    "kv_unified": True,
                    "cache_type_k": "q4_0",
                    "cache_type_v": "q4_0",
                    "extra_server_args": ["--jinja"],
                },
                "result": {
                    "ok": True,
                    "generation_tokens_per_second": 42.0,
                    "prompt_tokens_per_second": 900.0,
                    "failure": "none",
                    "agent_bench_score": 0.82,
                    "benchmark_suite_ok": True,
                    "serving_ttft_ms": 420.0,
                    "serving_tokens_per_second": 38.0,
                },
                "score": 0.82,
            }
        ),
        encoding="utf-8",
    )

    readiness = write_deployment_readiness(tmp_path)

    payload = json.loads(readiness.json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "RETEST_DEPLOYMENT"
    assert payload["profiles"][0]["status"] == "UNPROVEN"
    assert "exact selected profile" in payload["profiles"][0]["reason"]
