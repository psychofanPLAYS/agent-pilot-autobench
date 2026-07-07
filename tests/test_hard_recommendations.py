import json
from pathlib import Path

from gguf_limit_bench.hard_recommendations import write_hard_recommendations


def _write_flag_recs(root, model="G:/AI/models/Winner.gguf"):
    (root / "flag-recommendations.json").write_text(
        json.dumps(
            {
                "model": model,
                "model_name": model.rsplit("/", 1)[-1],
                "lane_type": "chat_agent",
                "profiles": [
                    {"id": "standard", "label": "Standard", "context_size": 131072},
                    {"id": "long_agent", "label": "Long agent", "context_size": 200000},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_run(
    root,
    name,
    *,
    model="G:/AI/models/Winner.gguf",
    context=131072,
    generation_tokens_per_second=42.0,
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
                    "generation_tokens_per_second": generation_tokens_per_second,
                    "prompt_tokens_per_second": 900.0,
                    "failure": failure,
                    "agent_bench_score": agent_bench_score,
                    "benchmark_suite_ok": benchmark_suite_ok,
                    "benchmark_suite_general_score": agent_bench_score,
                    "benchmark_suite_agentic_score": agent_bench_score,
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


def _write_qe_summary(root, *, score=0.78, format_rate=0.78):
    run = root / "qe-candidate"
    run.mkdir(parents=True)
    (run / "qe-format-summary.json").write_text(
        json.dumps(
            {
                "model": "qwen-qe-2b",
                "score": score,
                "format_rate": format_rate,
                "direct_answer_rate": 0.0,
                "attempts": 50,
                "median_tps": 200.0,
                "median_ttft_ms": 140.0,
            }
        ),
        encoding="utf-8",
    )


def _write_benchmark_suite_preflight_failure(root):
    (root / "benchmark-suite-preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": False,
                "status": "HARNESS_MISSING",
                "model": "G:/AI/models/Winner.gguf",
                "context": 200000,
                "issue_count": 1,
                "issues": [
                    {
                        "task_id": "gsm8k_cot_zeroshot_smoke",
                        "phase": "general",
                        "harness": "lm-evaluation-harness",
                        "command_index": 1,
                        "executable": "uvx",
                        "failure_class": "harness_missing",
                        "detail": "Executable `uvx` was not found on PATH.",
                    }
                ],
                "receipt_path": str(root / "benchmark-suite-preflight.json"),
                "next_action": (
                    "Install or expose missing benchmark harness executable(s): uvx. "
                    "Affected harnesses: lm-evaluation-harness. "
                    "Rerun benchmark-suite-preflight before deployment proof."
                ),
            }
        ),
        encoding="utf-8",
    )


def _write_benchmark_suite_preflight_pass(root):
    (root / "benchmark-suite-preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "PASS",
                "model": "G:/AI/models/Winner.gguf",
                "context": 200000,
                "issue_count": 0,
                "issues": [],
                "receipt_path": str(root / "benchmark-suite-preflight.json"),
                "next_action": (
                    "Benchmark-suite command preflight passed; run the suite against the live model."
                ),
            }
        ),
        encoding="utf-8",
    )


def _path_arg(path: Path) -> str:
    return path.as_posix()


def test_hard_recommendations_refuses_speed_only_and_lists_next_actions(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "speed-only",
        context=262144,
        serving_ttft_ms=900.0,
        serving_tokens_per_second=20.0,
    )
    _write_qe_summary(tmp_path)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    markdown = outputs.markdown_path.read_text(encoding="utf-8")
    assert payload["overall_action"] == "RETEST"
    assert payload["operator_verdict"] == {
        "status": "NOT_USABLE_YET",
        "headline": "No deployable recommendation exists.",
        "why": (
            "The best receipt is still missing score-backed model proof, deployment proof, "
            "QE proof, or stability proof."
        ),
        "next_command": (
            "apb benchmark-suite-template --output benchmark-suite.plan.json "
            '--model "G:/AI/models/Winner.gguf" --base-url http://127.0.0.1:8080/v1 '
            "--context 262144"
        ),
    }
    assert payload["score_evidence"] == {
        "candidate_count": 1,
        "scored_candidate_count": 0,
        "proven_recommendation_count": 0,
        "proven_component_count": 0,
        "top_agent_quality_score": None,
        "top_general_score": None,
        "top_agentic_score": None,
        "top_generation_tps": 42.0,
        "top_serving_tps": 20.0,
    }
    assert payload["performance_prediction"] == {
        "status": "LAB_ONLY_SPEED_PROOF",
        "risk": "high",
        "deployment_expectation": "do_not_deploy",
        "expected_user_experience": (
            "Likely responsive token streaming, but agent quality is unmeasured. "
            "Use only for lab probing until benchmark-suite or librarian evidence exists."
        ),
        "quality_basis": "No agent-quality score is present.",
        "speed_basis": "serving 20.00 tok/s; generation 42.00 tok/s; class slow_interactive",
        "context_basis": "262144 context; class long_agentic",
        "missing_basis": "model, deployment, context, resource, qe",
    }
    assert payload["hard_recommendations"] == []
    assert payload["settings_candidates"][0] == {
        "rank": 1,
        "profile_id": "standard",
        "label": "Standard",
        "context_size": 131072,
        "status": "SYSTEMS_ONLY",
        "decision": "needs_agent_score",
        "recommendation_score": 69.7667,
        "source_model": "Winner.gguf",
        "target_model": "Winner.gguf",
        "evidence": {
            "run_id": "speed-only",
            "receipt_path": str(tmp_path / "speed-only"),
            "context": 262144,
            "agent_bench_score": None,
            "serving_ttft_ms": 900.0,
            "serving_tps": 20.0,
            "generation_tps": 42.0,
            "status": "WORKFLOW UNPROVEN",
            "resource_summary": {},
        },
        "reason": "Matching receipt has context/serving evidence but no agent-quality score.",
        "next_action": "Rerun profile `standard` with benchmark-suite or librarian score evidence.",
    }
    assert payload["settings_candidates"][1]["profile_id"] == "long_agent"
    assert payload["settings_candidates"][1]["decision"] == "needs_agent_score"
    assert payload["model_gate"]["action"] == "RETEST"
    assert payload["deployment_gate"]["action"] == "RETEST_DEPLOYMENT"
    assert payload["qe_gate"]["action"] == "RETEST_QE_PROFILE"
    assert payload["candidate_assessment"] == {
        "model": "Winner.gguf",
        "run_id": "speed-only",
        "readiness": "not_recommendable",
        "readiness_score": 0,
        "confidence": "low",
        "known_performance": {
            "agent_quality_score": None,
            "general_score": None,
            "agentic_score": None,
            "generation_tps": 42.0,
            "serving_tps": 20.0,
            "cold_ttft_ms": 900.0,
            "warm_ttft_ms": None,
            "context": 262144,
            "quality": "unmeasured",
            "speed": "slow_interactive",
            "context_class": "long_agentic",
            "recommendation_class": "needs_agent_benchmark",
        },
        "missing_evidence": [
            {
                "gate": "model",
                "status": "RETEST",
                "required": "agent-quality score from benchmark-suite or recommendation-grade librarian evidence",
                "next_action": (
                    "Run a benchmark-suite plan or librarian-bench mode so pilotBENCHY "
                    "can produce agent_bench_score, general score, and agentic score."
                ),
            },
            {
                "gate": "deployment",
                "status": "RETEST_DEPLOYMENT",
                "required": "matching scored receipt at the selected context plus serving telemetry",
                "next_action": (
                    "Run a scored librarian/benchmark-suite receipt at 131072 context "
                    "for profile `standard` with serving telemetry enabled."
                ),
            },
            {
                "gate": "context",
                "status": "WAITING_FOR_DEPLOYMENT",
                "required": "deployment proof at required context >= 131072",
                "next_action": (
                    "Run deployment proof for profile `standard` at 131072 context "
                    "with score and serving/resource telemetry."
                ),
            },
            {
                "gate": "resource",
                "status": "WAITING_FOR_DEPLOYMENT",
                "required": "same-run resource telemetry for the promoted settings receipt",
                "next_action": "First promote a deployment profile with score and serving evidence.",
            },
            {
                "gate": "qe",
                "status": "RETEST_QE_PROFILE",
                "required": "fresh-session QE score >= 0.90, format rate >= 0.90, direct answers == 0",
                "next_action": "Improve prompt/template/canonicalizer and rerun qe-format before deployment.",
            },
        ],
    }
    assert any("benchmark-suite" in action for action in payload["next_actions"])
    assert any("profile `standard`" in action for action in payload["next_actions"])
    assert any("qe-format" in action for action in payload["next_actions"])
    assert not any(
        "Repeat the promoted model/settings" in action for action in payload["next_actions"]
    )
    assert payload["stability_gate"]["action"] == "WAITING_FOR_PROMOTED_STACK"
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    runbook = payload["proof_runbook"]
    assert [step["step"] for step in runbook] == [1, 2, 3, 4, 5, 6, 7]
    assert runbook[0] == {
        "step": 1,
        "id": "model_plan",
        "gate": "model",
        "status": "pending",
        "command": commands["model_plan"],
        "proves": "benchmark-suite.plan.json",
        "success_condition": "Plan file exists and names the target model/context.",
        "next": "Edit the generated plan if needed; the next proof command runs it.",
    }
    assert runbook[1]["id"] == "model_score"
    assert runbook[1]["proves"] == f"{_path_arg(tmp_path)}/<suite-run>/suite-verdict.json"
    assert runbook[1]["success_condition"] == "suite-verdict action is PROMOTE."
    preflight_step = next(step for step in runbook if step["id"] == "benchmark_suite_preflight")
    assert preflight_step["proves"] == f"{_path_arg(tmp_path)}/benchmark-suite-preflight.json"
    assert preflight_step["success_condition"] == "benchmark-suite preflight status is PASS."
    assert runbook[-1]["id"] == "refresh_hard_recommendations"
    assert runbook[-1]["success_condition"] == (
        "hard-recommendations overall_action is PROMOTE_READY_STACK or an intentional "
        "PROMOTE_PARTIAL."
    )
    assert "model_plan" in commands
    assert "benchmark-suite-template" in commands["model_plan"]
    assert "--context 262144" in commands["model_plan"]
    assert "model_score" in commands
    assert commands["model_score"] == (
        f"apb benchmark-suite --plan benchmark-suite.plan.json --runs-root {_path_arg(tmp_path)}"
    )
    assert commands["benchmark_suite_preflight"] == (
        f"apb benchmark-suite-preflight --plan benchmark-suite.plan.json "
        f"--runs-root {_path_arg(tmp_path)}"
    )
    assert "deployment_flags" in commands
    assert commands["deployment_flags"] == (
        'apb flag-recommendations --model "G:/AI/models/Winner.gguf" '
        f"--output-dir {_path_arg(tmp_path)}"
    )
    assert "deployment_proof" in commands
    assert commands["deployment_proof"] == (
        f"apb deployment-proof --profile standard --runs-root {_path_arg(tmp_path)} "
        f"--flag-recommendations {_path_arg(tmp_path / 'flag-recommendations.json')} "
        "--benchmark-suite-plan benchmark-suite.plan.json --budget-minutes 30"
    )
    assert "qe_format" in commands
    assert "qe-format" in commands["qe_format"]
    assert "PORT" not in commands["qe_format"]
    assert "--base-url http://127.0.0.1:8080" in commands["qe_format"]
    assert "## Proof Commands" in markdown
    assert "## Proof Runbook" in markdown
    assert "| 1 | `model` | `model_plan` | `pending` | `benchmark-suite.plan.json` |" in markdown
    assert "### model/model_score" in markdown
    assert "## Candidate Assessment" in markdown
    assert "## Operator Verdict" in markdown
    assert "Status: `NOT_USABLE_YET`" in markdown
    assert "## Score Evidence" in markdown
    assert "## Performance Prediction" in markdown
    assert "## Settings Candidates" in markdown
    assert (
        "| 1 | `standard` | 131072 | `SYSTEMS_ONLY` | `needs_agent_score` | `69.7667` | `speed-only @ 262144` |"
        in markdown
    )
    assert "Status: `LAB_ONLY_SPEED_PROOF`" in markdown
    assert "Deployment expectation: `do_not_deploy`" in markdown
    assert "Scored candidates: `0/1`" in markdown
    assert "Readiness: `not_recommendable` (`0/100`)" in markdown
    assert "`serving_tps`: `20.0000`" in markdown
    assert "- `model`: RETEST" in markdown
    assert "apb hard-recommendations --runs-root" in markdown
    assert "No hard recommendations are proven yet." in markdown


def test_hard_recommendations_promotes_only_proven_model_settings_and_qe(tmp_path):
    _write_flag_recs(tmp_path)
    run = _write_run(
        tmp_path,
        "suite-proven",
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "autoresearch_attempt_finished",
                "data": {
                    "telemetry": {
                        "gpu_used_mb": 17100,
                        "gpu_total_mb": 24564,
                        "gpu_util_percent": 91,
                        "ram_available_mb": 50000,
                        "ram_used_percent": 59.0,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    component_types = {item["type"] for item in payload["proven_components"]}
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert payload["operator_verdict"]["status"] == "PARTIAL_NOT_PRODUCTION"
    assert payload["operator_verdict"]["next_command"].startswith(
        "apb deployment-proof --profile standard"
    )
    assert payload["score_evidence"]["candidate_count"] == 1
    assert payload["score_evidence"]["scored_candidate_count"] == 1
    assert payload["score_evidence"]["proven_recommendation_count"] == 0
    assert payload["score_evidence"]["proven_component_count"] == 3
    assert payload["score_evidence"]["top_agent_quality_score"] == 0.82
    assert payload["performance_prediction"]["status"] == "PARTIAL_STACK"
    assert payload["performance_prediction"]["risk"] == "medium"
    assert payload["performance_prediction"]["deployment_expectation"] == "lab_candidate"
    assert "stability" in payload["performance_prediction"]["missing_basis"]
    assert payload["stability_gate"]["action"] == "RETEST_STABILITY"
    assert (
        payload["stability_gate"]["required"]
        == "at least 3 comparable receipts with repeatable measured metrics"
    )
    assert any("Repeat the promoted model/settings" in action for action in payload["next_actions"])
    assert payload["hard_recommendations"] == []
    assert component_types == {"model", "settings_profile", "qe_profile"}
    assert payload["settings_candidates"][0]["profile_id"] == "standard"
    assert payload["settings_candidates"][0]["status"] == "PROVEN"
    assert payload["settings_candidates"][0]["decision"] == "recommended"
    assert (
        payload["settings_candidates"][0]["evidence"]["resource_summary"]["max_gpu_used_mb"]
        == 17100
    )
    assert payload["settings_candidates"][1]["profile_id"] == "long_agent"
    assert payload["settings_candidates"][1]["decision"] == "next_to_test"
    assert payload["model_gate"]["action"] == "PROMOTE"
    assert payload["deployment_gate"]["recommended_profile_id"] == "standard"
    settings = next(
        item for item in payload["proven_components"] if item["type"] == "settings_profile"
    )
    assert settings["evidence"]["resource_summary"]["max_gpu_used_mb"] == 17100
    assert settings["evidence"]["resource_summary"]["gpu_total_mb"] == 24564
    assert payload["qe_gate"]["action"] == "PROMOTE_QE_PROFILE"
    assert payload["scorecard"]["quality"] == "strong"
    assert payload["scorecard"]["speed"] == "interactive"
    assert payload["scorecard"]["context"] == "long_agentic"
    assert payload["candidate_assessment"]["readiness"] == "near_ready"
    assert payload["candidate_assessment"]["readiness_score"] == 90
    assert [item["gate"] for item in payload["candidate_assessment"]["missing_evidence"]] == [
        "stability"
    ]
    assert payload["proof_runbook"][0]["id"] == "stability_repeat"
    assert payload["proof_commands"][0]["id"] == "stability_repeat"
    assert "## Resource Evidence" in outputs.markdown_path.read_text(encoding="utf-8")
    assert "17100/24564 MB" in outputs.markdown_path.read_text(encoding="utf-8")
    assert "No hard recommendations are proven yet." in outputs.markdown_path.read_text(
        encoding="utf-8"
    )


def test_hard_recommendations_blocks_settings_candidate_on_critical_runtime_warning(tmp_path):
    _write_flag_recs(tmp_path)
    run = _write_run(
        tmp_path,
        "template-warning",
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=430.0,
        serving_tokens_per_second=38.0,
        settings={"profile_name": "standard"},
    )
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "llama_server_ready",
                "data": {
                    "telemetry": {
                        "gpu_used_mb": 17100,
                        "gpu_total_mb": 24564,
                        "gpu_util_percent": 82,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    warning_dir = run / "simplebench-standard"
    warning_dir.mkdir()
    (warning_dir / "warnings.log").write_text(
        "18.26.395.388 W common_chat_try_specialized_template: "
        "detected an outdated gemma4 chat template\n",
        encoding="utf-8",
    )
    _write_qe_summary(tmp_path, score=0.94, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    standard = next(
        item for item in payload["settings_candidates"] if item["profile_id"] == "standard"
    )
    component_types = {item["type"] for item in payload["proven_components"]}
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert payload["deployment_gate"]["action"] == "RETEST_DEPLOYMENT"
    assert payload["deployment_gate"]["profiles"][0]["status"] == "RUNTIME_WARNING"
    assert payload["runtime_warning_gate"]["action"] == "RETEST_RUNTIME_WARNINGS"
    assert "outdated gemma4" in payload["runtime_warning_gate"]["critical"][0]
    assert standard["status"] == "RUNTIME_WARNING"
    assert standard["decision"] == "fix_runtime_warning"
    assert "outdated gemma4" in standard["next_action"]
    assert any("critical runtime warnings" in action for action in payload["next_actions"])
    assert "settings_profile" not in component_types
    assert {"model", "qe_profile"}.issubset(component_types)


def test_hard_recommendations_ready_stack_requires_repeatable_stability(tmp_path):
    _write_flag_recs(tmp_path)
    for index, score in enumerate([0.80, 0.81, 0.82], start=1):
        run = _write_run(
            tmp_path,
            f"suite-proven-{index}",
            context=131072,
            generation_tokens_per_second=40.0 + index,
            agent_bench_score=score,
            benchmark_suite_ok=True,
            serving_ttft_ms=410.0 + index,
            serving_tokens_per_second=37.0 + index,
        )
        (run / "events.jsonl").write_text(
            json.dumps(
                {
                    "type": "autoresearch_attempt_finished",
                    "data": {
                        "telemetry": {
                            "gpu_used_mb": 17100,
                            "gpu_total_mb": 24564,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    assert payload["repeatability"]["confidence"] == "repeatable"
    assert payload["stability_gate"]["action"] == "PROMOTE_STABILITY"
    assert payload["overall_action"] == "PROMOTE_READY_STACK"
    assert payload["operator_verdict"]["status"] == "READY_TO_USE"
    assert {item["type"] for item in payload["hard_recommendations"]} == {
        "model",
        "settings_profile",
        "qe_profile",
    }
    assert {item["type"] for item in payload["proven_components"]} == {
        "model",
        "settings_profile",
        "qe_profile",
    }
    assert payload["operator_verdict"]["next_command"] is None
    assert payload["score_evidence"]["candidate_count"] == 1
    assert payload["score_evidence"]["scored_candidate_count"] == 1
    assert payload["score_evidence"]["proven_recommendation_count"] == 3
    assert payload["score_evidence"]["proven_component_count"] == 3
    assert payload["performance_prediction"]["status"] == "READY_AGENT_STACK"
    assert payload["performance_prediction"]["risk"] == "low"
    assert payload["performance_prediction"]["deployment_expectation"] == "deployable"
    assert payload["candidate_assessment"]["readiness"] == "ready_stack"
    assert payload["candidate_assessment"]["readiness_score"] == 100
    assert payload["candidate_assessment"]["missing_evidence"] == []
    assert payload["proof_commands"] == []


def test_hard_recommendations_stability_requires_same_context_and_settings_stack(tmp_path):
    _write_flag_recs(tmp_path)
    for name, context, score in [
        ("standard-128k", 131072, 0.82),
        ("long-agent-200k", 200000, 0.81),
        ("over-the-top-262k", 262144, 0.80),
    ]:
        run = _write_run(
            tmp_path,
            name,
            context=context,
            agent_bench_score=score,
            benchmark_suite_ok=True,
            serving_ttft_ms=420.0,
            serving_tokens_per_second=38.0,
        )
        (run / "events.jsonl").write_text(
            json.dumps(
                {
                    "type": "autoresearch_attempt_finished",
                    "data": {
                        "telemetry": {
                            "gpu_used_mb": 17100,
                            "gpu_total_mb": 24564,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    assert payload["model_gate"]["champion_run_id"] == "standard-128k"
    assert payload["repeatability"]["run_count"] == 1
    assert payload["repeatability"]["confidence"] == "single_run"
    assert payload["stability_gate"]["action"] == "RETEST_STABILITY"
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert payload["hard_recommendations"] == []
    assert [item["gate"] for item in payload["candidate_assessment"]["missing_evidence"]] == [
        "stability"
    ]
    assert "current comparable receipts: 1" in payload["stability_gate"]["next_run"]


def test_hard_recommendations_stability_ignores_non_recommendation_grade_repeats(tmp_path):
    _write_flag_recs(tmp_path)
    scored_run = _write_run(
        tmp_path,
        "suite-proven",
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    (scored_run / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "benchmark_suite_finished_on_owned_server",
                "data": {
                    "telemetry": {
                        "gpu_used_mb": 17100,
                        "gpu_total_mb": 24564,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_run(
        tmp_path,
        "speed-repeat-a",
        context=131072,
        serving_ttft_ms=421.0,
        serving_tokens_per_second=38.0,
    )
    _write_run(
        tmp_path,
        "speed-repeat-b",
        context=131072,
        serving_ttft_ms=422.0,
        serving_tokens_per_second=38.0,
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    assert payload["model_gate"]["champion_run_id"] == "suite-proven"
    assert payload["repeatability"]["run_count"] == 1
    assert payload["repeatability"]["confidence"] == "single_run"
    assert payload["stability_gate"]["action"] == "RETEST_STABILITY"
    assert "speed-repeat-a" not in payload["repeatability"]["receipt_paths"]
    assert "speed-repeat-b" not in payload["repeatability"]["receipt_paths"]


def test_hard_recommendations_ready_stack_requires_same_run_resource_proof(tmp_path):
    _write_flag_recs(tmp_path)
    for index, score in enumerate([0.82, 0.81, 0.80], start=1):
        run = _write_run(
            tmp_path,
            f"standard-repeat-{index}",
            context=131072,
            generation_tokens_per_second=40.0 + index,
            agent_bench_score=score,
            benchmark_suite_ok=True,
            serving_ttft_ms=410.0 + index,
            serving_tokens_per_second=37.0 + index,
        )
        if index == 2:
            (run / "events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "benchmark_suite_finished_on_owned_server",
                        "data": {
                            "telemetry": {
                                "gpu_used_mb": 17100,
                                "gpu_total_mb": 24564,
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    missing = {item["gate"]: item for item in payload["candidate_assessment"]["missing_evidence"]}
    assert payload["model_gate"]["champion_run_id"] == "standard-repeat-1"
    assert payload["repeatability"]["confidence"] == "repeatable"
    assert payload["stability_gate"]["action"] == "PROMOTE_STABILITY"
    assert payload["resource_gate"] == {
        "action": "RETEST_RESOURCE",
        "required": "same-run resource telemetry for the promoted settings receipt",
        "run_id": "standard-repeat-1",
        "resource_run_id": "standard-repeat-2",
        "next_run": (
            "Rerun deployment proof for profile `standard` with resource telemetry "
            "captured in the same receipt as the promoted score."
        ),
    }
    assert missing["resource"]["status"] == "RETEST_RESOURCE"
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert payload["hard_recommendations"] == []
    assert payload["performance_prediction"]["missing_basis"] == "resource"


def test_hard_recommendations_stability_repeat_uses_promoted_profile(tmp_path):
    _write_flag_recs(tmp_path)
    run = _write_run(
        tmp_path,
        "long-agent-proven",
        context=200000,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    (run / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "benchmark_suite_finished_on_owned_server",
                "data": {
                    "telemetry": {
                        "gpu_used_mb": 17100,
                        "gpu_total_mb": 24564,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path, required_context=200000)

    payload = outputs.payload
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert payload["deployment_gate"]["recommended_profile_id"] == "long_agent"
    assert payload["stability_gate"]["action"] == "RETEST_STABILITY"
    assert commands["stability_repeat"].startswith("apb deployment-proof --profile long_agent ")


def test_hard_recommendations_lists_ranked_candidates_with_evidence_gaps(tmp_path):
    _write_flag_recs(tmp_path, model="G:/AI/models/Beta.gguf")
    _write_run(
        tmp_path,
        "alpha-speed-only",
        model="G:/AI/models/Alpha.gguf",
        context=262144,
        serving_ttft_ms=900.0,
        serving_tokens_per_second=20.0,
    )
    _write_run(
        tmp_path,
        "beta-suite-proven",
        model="G:/AI/models/Beta.gguf",
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    _write_qe_summary(tmp_path)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    rankings = payload["candidate_rankings"]
    markdown = outputs.markdown_path.read_text(encoding="utf-8")
    assert [item["model"] for item in rankings] == ["Beta.gguf", "Alpha.gguf"]
    assert rankings[0]["rank"] == 1
    assert rankings[0]["run_id"] == "beta-suite-proven"
    assert rankings[0]["agent_quality_score"] == 0.82
    assert rankings[0]["prediction"] == {
        "quality": "strong",
        "speed": "interactive",
        "context": "long_agentic",
        "recommendation": "score_backed_candidate",
    }
    assert rankings[0]["evidence_gaps"] == []
    assert rankings[1]["run_id"] == "alpha-speed-only"
    assert rankings[1]["prediction"]["quality"] == "unmeasured"
    assert rankings[1]["prediction"]["speed"] == "slow_interactive"
    assert rankings[1]["prediction"]["context"] == "long_agentic"
    assert rankings[1]["prediction"]["recommendation"] == "needs_agent_benchmark"
    assert rankings[1]["evidence_gaps"] == ["agent_quality", "benchmark_suite"]
    assert "## Candidate Rankings" in markdown
    assert (
        "| 1 | `Beta.gguf` | `beta-suite-proven` | `BENCHMARK SUITE` | `0.8200` | `strong` | `interactive` | `long_agentic` | `none` |"
        in markdown
    )
    assert (
        "| 2 | `Alpha.gguf` | `alpha-speed-only` | `WORKFLOW UNPROVEN` | `unmeasured` | `unmeasured` | `slow_interactive` | `long_agentic` | `agent_quality, benchmark_suite` |"
        in markdown
    )


def test_hard_recommendations_ranks_best_distinct_model_candidates(tmp_path):
    _write_flag_recs(tmp_path, model="G:/AI/models/Beta.gguf")
    _write_run(
        tmp_path,
        "beta-good",
        model="G:/AI/models/Beta.gguf",
        context=131072,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    _write_run(
        tmp_path,
        "beta-duplicate",
        model="G:/AI/models/Beta.gguf",
        context=131072,
        agent_bench_score=0.81,
        benchmark_suite_ok=True,
        serving_ttft_ms=450.0,
        serving_tokens_per_second=32.0,
    )
    _write_run(
        tmp_path,
        "alpha-speed-only",
        model="G:/AI/models/Alpha.gguf",
        context=262144,
        serving_ttft_ms=900.0,
        serving_tokens_per_second=20.0,
    )
    _write_qe_summary(tmp_path)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    rankings = payload["candidate_rankings"]
    assert [item["model"] for item in rankings] == ["Beta.gguf", "Alpha.gguf"]
    assert [item["rank"] for item in rankings] == [1, 2]
    assert rankings[0]["run_id"] == "beta-good"
    assert "beta-duplicate" not in outputs.markdown_path.read_text(encoding="utf-8")


def test_hard_recommendations_reports_repeatability_for_top_candidate(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "winner-a",
        generation_tokens_per_second=40.0,
        serving_ttft_ms=410.0,
        serving_tokens_per_second=36.0,
        agent_bench_score=0.80,
        benchmark_suite_ok=True,
    )
    _write_run(
        tmp_path,
        "winner-b",
        generation_tokens_per_second=42.0,
        serving_ttft_ms=430.0,
        serving_tokens_per_second=38.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
    )
    _write_run(
        tmp_path,
        "winner-c",
        generation_tokens_per_second=41.0,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=37.0,
        agent_bench_score=0.81,
        benchmark_suite_ok=True,
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    repeatability = payload["repeatability"]
    assert repeatability["model"] == "Winner.gguf"
    assert repeatability["run_count"] == 3
    assert repeatability["confidence"] == "repeatable"
    assert repeatability["score"]["min"] == 0.8
    assert repeatability["score"]["max"] == 0.82
    assert repeatability["generation_tps"]["min"] == 40.0
    assert repeatability["generation_tps"]["max"] == 42.0
    assert repeatability["cold_ttft_ms"]["min"] == 410.0
    assert repeatability["cold_ttft_ms"]["max"] == 430.0
    markdown = outputs.markdown_path.read_text(encoding="utf-8")
    assert "## Repeatability" in markdown
    assert "Confidence: `repeatable`" in markdown
    assert "Runs: `3`" in markdown


def test_hard_recommendations_writes_json_atomically(tmp_path, monkeypatch):
    _write_flag_recs(tmp_path)
    _write_run(tmp_path, "speed-only", context=131072)
    target = tmp_path / "hard-recommendations.json"
    target.write_text('{"schema_version":0}\n', encoding="utf-8")
    original_write_text = Path.write_text

    def fail_on_direct_json_write(self, data, *args, **kwargs):
        if self == target:
            original_write_text(self, "", encoding=kwargs.get("encoding", "utf-8"))
            raise OSError("simulated interrupted direct write")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_on_direct_json_write)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_hard_recommendations_scores_at_modern_agent_context_floor(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "short-context-speed-only",
        context=4096,
        serving_ttft_ms=350.0,
        serving_tokens_per_second=45.0,
    )
    _write_qe_summary(tmp_path)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert "--context 131072" in commands["model_plan"]
    assert "--profile standard" in commands["deployment_proof"]


def test_hard_recommendations_required_context_blocks_ready_stack_until_profile_is_proven(
    tmp_path,
):
    _write_flag_recs(tmp_path)
    for index, score in enumerate([0.80, 0.81, 0.82], start=1):
        _write_run(
            tmp_path,
            f"standard-128k-{index}",
            context=131072,
            generation_tokens_per_second=40.0 + index,
            agent_bench_score=score,
            benchmark_suite_ok=True,
            serving_ttft_ms=410.0 + index,
            serving_tokens_per_second=37.0 + index,
        )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path, required_context=200000)

    payload = outputs.payload
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    missing = {item["gate"]: item for item in payload["candidate_assessment"]["missing_evidence"]}
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert payload["operator_verdict"]["status"] == "PARTIAL_NOT_PRODUCTION"
    assert payload["hard_recommendations"] == []
    assert {item["type"] for item in payload["proven_components"]} == {
        "model",
        "settings_profile",
        "qe_profile",
    }
    candidates = {item["profile_id"]: item for item in payload["settings_candidates"]}
    assert payload["settings_candidates"][0]["profile_id"] == "long_agent"
    assert candidates["long_agent"]["decision"] == "next_to_test"
    assert candidates["standard"]["decision"] == "baseline_below_required_context"
    assert candidates["standard"]["next_action"] == (
        "This profile is proven only below the required context; run `long_agent` next."
    )
    assert payload["context_gate"] == {
        "action": "RETEST_CONTEXT",
        "required_context": 200000,
        "proven_context": 131072,
        "profile_id": "long_agent",
        "next_run": (
            "Run deployment proof for profile `long_agent` at 200000 context "
            "with score and serving/resource telemetry."
        ),
    }
    assert missing["context"]["status"] == "RETEST_CONTEXT"
    assert "context" in payload["performance_prediction"]["missing_basis"]
    assert commands["context_plan"] == (
        "apb benchmark-suite-template --output benchmark-suite-200000.plan.json "
        '--model "G:/AI/models/Winner.gguf" --base-url http://127.0.0.1:8080/v1 '
        "--context 200000"
    )
    assert commands["deployment_proof"] == (
        f"apb deployment-proof --profile long_agent --runs-root {_path_arg(tmp_path)} "
        f"--flag-recommendations {_path_arg(tmp_path / 'flag-recommendations.json')} "
        "--benchmark-suite-plan benchmark-suite-200000.plan.json --budget-minutes 30"
    )
    assert "Required context: `200000`" in outputs.markdown_path.read_text(encoding="utf-8")


def test_hard_recommendations_keeps_failed_required_context_proof_actionable(tmp_path):
    _write_flag_recs(tmp_path)
    flag_path = tmp_path / "flag-recommendations.json"
    flags = json.loads(flag_path.read_text(encoding="utf-8"))
    flags["profiles"].append(
        {"id": "over_the_top", "label": "Over-the-top", "context_size": 262144}
    )
    flag_path.write_text(json.dumps(flags), encoding="utf-8")
    _write_run(
        tmp_path,
        "standard-128k-proven",
        context=131072,
        generation_tokens_per_second=44.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=412.0,
        serving_tokens_per_second=39.0,
    )
    failed = _write_run(
        tmp_path,
        "long-agent-200k-partial",
        context=200000,
        generation_tokens_per_second=156.4,
        benchmark_suite_ok=False,
        benchmark_suite_failure="gsm8k_cot_zeroshot_smoke:harness_missing",
        failure="benchmark_suite_failed",
        serving_ttft_ms=159.3,
        serving_tokens_per_second=156.4,
        settings={"profile_name": "long_agent"},
        status="partial",
    )
    (failed / "events.jsonl").write_text(
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
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path, required_context=200000)

    payload = outputs.payload
    candidates = {item["profile_id"]: item for item in payload["settings_candidates"]}
    long_agent = candidates["long_agent"]
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert payload["context_gate"]["action"] == "RETEST_CONTEXT"
    assert payload["context_gate"]["next_run"] == (
        "Fix `long_agent` proof failure "
        "(gsm8k_cot_zeroshot_smoke:harness_missing) and rerun the profile."
    )
    assert payload["settings_candidates"][0]["profile_id"] == "long_agent"
    assert long_agent["status"] == "FAILED_PROOF"
    assert long_agent["decision"] == "fix_failed_proof"
    assert long_agent["evidence"]["context"] == 200000
    assert long_agent["evidence"]["serving_tps"] == 156.4
    assert long_agent["evidence"]["resource_summary"]["max_gpu_used_mb"] == 20898
    assert long_agent["next_action"] == (
        "Fix `long_agent` proof failure "
        "(gsm8k_cot_zeroshot_smoke:harness_missing) and rerun the profile."
    )
    assert candidates["over_the_top"]["decision"] == "next_to_test"
    assert candidates["standard"]["decision"] == "baseline_below_required_context"
    assert "FAILED_PROOF" in outputs.markdown_path.read_text(encoding="utf-8")


def test_hard_recommendations_surfaces_benchmark_suite_preflight_blocker(tmp_path):
    _write_flag_recs(tmp_path)
    _write_benchmark_suite_preflight_failure(tmp_path)
    _write_run(
        tmp_path,
        "standard-128k-proven",
        context=131072,
        generation_tokens_per_second=44.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=412.0,
        serving_tokens_per_second=39.0,
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path, required_context=200000)

    payload = outputs.payload
    missing = {item["gate"]: item for item in payload["candidate_assessment"]["missing_evidence"]}
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert payload["benchmark_suite_preflight"]["status"] == "HARNESS_MISSING"
    assert payload["next_actions"][0].startswith(
        "Install or expose missing benchmark harness executable(s): uvx."
    )
    assert missing["benchmark_suite_preflight"]["status"] == "HARNESS_MISSING"
    assert commands["benchmark_suite_preflight"] == (
        f"apb benchmark-suite-preflight --plan benchmark-suite-200000.plan.json "
        f"--runs-root {_path_arg(tmp_path)}"
    )
    assert "Benchmark-suite preflight: `HARNESS_MISSING`" in outputs.markdown_path.read_text(
        encoding="utf-8"
    )


def test_hard_recommendations_reruns_failed_context_profile_after_preflight_passes(tmp_path):
    _write_flag_recs(tmp_path)
    _write_benchmark_suite_preflight_pass(tmp_path)
    _write_run(
        tmp_path,
        "standard-128k-proven",
        context=131072,
        generation_tokens_per_second=44.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=412.0,
        serving_tokens_per_second=39.0,
    )
    _write_run(
        tmp_path,
        "long-agent-200k-old-harness-failed",
        context=200000,
        generation_tokens_per_second=156.4,
        benchmark_suite_ok=False,
        benchmark_suite_failure="gsm8k_cot_zeroshot_smoke:harness_missing",
        failure="benchmark_suite_failed",
        serving_ttft_ms=159.3,
        serving_tokens_per_second=156.4,
        settings={"profile_name": "long_agent"},
        status="partial",
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path, required_context=200000)

    payload = outputs.payload
    candidates = {item["profile_id"]: item for item in payload["settings_candidates"]}
    assert payload["benchmark_suite_preflight"]["status"] == "PASS"
    assert payload["context_gate"]["next_run"] == (
        "Rerun `long_agent` now that benchmark-suite preflight passes; "
        "the previous proof failed against an older unavailable harness plan."
    )
    assert candidates["long_agent"]["next_action"] == (
        "Rerun profile `long_agent` now that benchmark-suite preflight passes; "
        "the previous proof used an unavailable harness plan."
    )
    assert payload["next_actions"][0] == payload["context_gate"]["next_run"]


def test_hard_recommendations_rejects_stale_preflight_from_other_model_or_context(tmp_path):
    _write_flag_recs(tmp_path)
    (tmp_path / "benchmark-suite-preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "PASS",
                "model": "G:/AI/models/Other.gguf",
                "context": 131072,
                "issue_count": 0,
                "issues": [],
                "receipt_path": str(tmp_path / "benchmark-suite-preflight.json"),
                "next_action": (
                    "Benchmark-suite command preflight passed; run the suite against the live model."
                ),
            }
        ),
        encoding="utf-8",
    )
    _write_run(
        tmp_path,
        "standard-128k-proven",
        context=131072,
        generation_tokens_per_second=44.0,
        agent_bench_score=0.82,
        benchmark_suite_ok=True,
        serving_ttft_ms=412.0,
        serving_tokens_per_second=39.0,
    )
    _write_run(
        tmp_path,
        "long-agent-200k-old-harness-failed",
        context=200000,
        generation_tokens_per_second=156.4,
        benchmark_suite_ok=False,
        benchmark_suite_failure="gsm8k_cot_zeroshot_smoke:harness_missing",
        failure="benchmark_suite_failed",
        serving_ttft_ms=159.3,
        serving_tokens_per_second=156.4,
        settings={"profile_name": "long_agent"},
        status="partial",
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(
        tmp_path,
        target_model="Winner.gguf",
        required_context=200000,
    )

    payload = outputs.payload
    missing = {item["gate"]: item for item in payload["candidate_assessment"]["missing_evidence"]}
    assert payload["benchmark_suite_preflight"]["status"] == "STALE"
    assert payload["benchmark_suite_preflight"]["original_status"] == "PASS"
    assert "Other.gguf" in payload["benchmark_suite_preflight"]["stale_reasons"][1]
    assert missing["benchmark_suite_preflight"]["status"] == "STALE"
    assert payload["next_actions"][0].startswith(
        "Regenerate benchmark-suite preflight for this exact model/context/plan"
    )
    assert payload["context_gate"]["next_run"] == (
        "Fix `long_agent` proof failure "
        "(gsm8k_cot_zeroshot_smoke:harness_missing) and rerun the profile."
    )


def test_hard_recommendations_refuses_ready_stack_when_model_and_settings_differ(tmp_path):
    _write_flag_recs(tmp_path, model="G:/AI/models/Alpha.gguf")
    _write_run(
        tmp_path,
        "alpha-settings-proven",
        model="G:/AI/models/Alpha.gguf",
        context=131072,
        agent_bench_score=0.81,
        benchmark_suite_ok=True,
        serving_ttft_ms=420.0,
        serving_tokens_per_second=38.0,
    )
    _write_run(
        tmp_path,
        "beta-model-wins",
        model="G:/AI/models/Beta.gguf",
        context=131072,
        agent_bench_score=0.92,
        benchmark_suite_ok=True,
        serving_ttft_ms=390.0,
        serving_tokens_per_second=40.0,
    )
    _write_qe_summary(tmp_path, score=0.95, format_rate=0.94)

    outputs = write_hard_recommendations(tmp_path)

    payload = json.loads(outputs.json_path.read_text(encoding="utf-8"))
    recommendation_types = {item["type"] for item in payload["hard_recommendations"]}
    component_types = {item["type"] for item in payload["proven_components"]}
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert payload["model_gate"]["champion_model"] == "Beta.gguf"
    assert payload["deployment_gate"]["model_name"] == "Alpha.gguf"
    assert payload["deployment_gate"]["action"] == "PROMOTE_DEPLOYMENT_PROFILE"
    assert payload["overall_action"] == "PROMOTE_PARTIAL"
    assert "settings_profile" not in recommendation_types
    assert payload["hard_recommendations"] == []
    assert "settings_profile" not in component_types
    assert {"model", "qe_profile"}.issubset(component_types)
    assert any(
        "flag-recommendations for `Beta.gguf`" in action for action in payload["next_actions"]
    )
    assert commands["deployment_flags"] == (
        'apb flag-recommendations --model "G:/AI/models/Beta.gguf" '
        f"--output-dir {_path_arg(tmp_path)}"
    )


def test_hard_recommendations_can_scope_to_intended_target_model(tmp_path):
    _write_flag_recs(tmp_path, model="G:/AI/models/Qwopus.gguf")
    _write_run(
        tmp_path,
        "old-qwopus-speed",
        model="G:/AI/models/Qwopus.gguf",
        context=4096,
        serving_ttft_ms=100.0,
        serving_tokens_per_second=120.0,
    )
    _write_qe_summary(tmp_path)

    outputs = write_hard_recommendations(
        tmp_path,
        target_model="Gemma-4-26B",
        target_model_path="G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
    )

    payload = outputs.payload
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert payload["target_scope"] == {
        "target_model": "Gemma-4-26B",
        "target_model_path": "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf",
        "status": "NO_TARGET_EVIDENCE",
        "matched_receipt_count": 0,
        "ignored_receipt_count": 1,
    }
    assert payload["model_gate"]["action"] == "NO_EVIDENCE"
    assert payload["candidate_assessment"]["model"] == "Gemma-4-26B"
    assert payload["score_evidence"]["candidate_count"] == 0
    assert payload["candidate_rankings"] == []
    assert payload["settings_candidates"][0]["status"] == "STALE_MODEL"
    assert payload["settings_candidates"][0]["target_model"] == "Gemma-4-26B"
    assert payload["settings_candidates"][0]["next_action"] == (
        "Run apb flag-recommendations for `Gemma-4-26B` before testing profile `standard`."
    )
    assert commands["model_plan"] == (
        "apb benchmark-suite-template --output benchmark-suite.plan.json "
        '--model "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf" --base-url http://127.0.0.1:8080/v1 '
        "--context 131072"
    )
    model_plan = next(
        command for command in payload["proof_commands"] if command["id"] == "model_plan"
    )
    assert model_plan["context_target"] == "required_context_131072"
    assert model_plan["context_size"] == "131072"
    assert commands["deployment_flags"] == (
        'apb flag-recommendations --model "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf" '
        f"--output-dir {_path_arg(tmp_path)}"
    )
    assert commands["refresh_hard_recommendations"] == (
        f"apb hard-recommendations --runs-root {_path_arg(tmp_path)} "
        '--target-model "Gemma-4-26B" '
        '--target-model-path "G:/AI/models/Gemma-4-26B-A4B-Q8_0.gguf"'
    )
    assert commands["deployment_proof"] == (
        f"apb deployment-proof --profile standard --runs-root {_path_arg(tmp_path)} "
        f"--flag-recommendations {_path_arg(tmp_path / 'flag-recommendations.json')} "
        "--benchmark-suite-plan benchmark-suite.plan.json --budget-minutes 30"
    )


def test_hard_recommendations_uses_required_context_for_model_plan_proof(tmp_path):
    _write_flag_recs(tmp_path)
    _write_run(
        tmp_path,
        "old-qwopus-speed",
        model="G:/AI/models/Qwopus.gguf",
        context=131072,
    )

    outputs = write_hard_recommendations(
        tmp_path,
        target_model="Gemma-4-26B",
        required_context=200000,
    )

    model_plan = next(
        command for command in outputs.payload["proof_commands"] if command["id"] == "model_plan"
    )
    assert model_plan["command"].endswith("--context 200000")
    assert model_plan["context_size"] == "200000"
    assert model_plan["context_target"] == "required_context_200000"


def test_hard_recommendations_marks_settings_candidates_stale_when_flags_are_for_other_model(
    tmp_path,
):
    _write_flag_recs(tmp_path, model="G:/AI/models/Alpha.gguf")
    _write_run(
        tmp_path,
        "beta-speed-only",
        model="G:/AI/models/Beta.gguf",
        context=4096,
        serving_ttft_ms=100.0,
        serving_tokens_per_second=120.0,
    )
    _write_qe_summary(tmp_path)

    outputs = write_hard_recommendations(tmp_path)

    payload = outputs.payload
    markdown = outputs.markdown_path.read_text(encoding="utf-8")
    deployment_missing = next(
        item
        for item in payload["candidate_assessment"]["missing_evidence"]
        if item["gate"] == "deployment"
    )
    commands = {command["id"]: command["command"] for command in payload["proof_commands"]}
    assert payload["deployment_gate"]["model_name"] == "Alpha.gguf"
    assert payload["model_gate"]["champion_model"] == "Beta.gguf"
    assert deployment_missing["status"] == "MODEL_MISMATCH"
    assert deployment_missing["next_action"] == "Run apb flag-recommendations for `Beta.gguf`."
    assert payload["next_actions"] == [
        (
            "Run a benchmark-suite plan or librarian-bench mode so pilotBENCHY can "
            "produce agent_bench_score, general score, and agentic score."
        ),
        "Run apb flag-recommendations for `Beta.gguf`.",
        "Improve prompt/template/canonicalizer and rerun qe-format before deployment.",
    ]
    assert payload["settings_candidates"][0]["status"] == "STALE_MODEL"
    assert payload["settings_candidates"][0]["decision"] == "regenerate_for_top_candidate"
    assert payload["settings_candidates"][0]["source_model"] == "Alpha.gguf"
    assert payload["settings_candidates"][0]["target_model"] == "Beta.gguf"
    assert payload["settings_candidates"][0]["next_action"] == (
        "Run apb flag-recommendations for `Beta.gguf` before testing profile `standard`."
    )
    assert "STALE_MODEL" in markdown
    assert "regenerate_for_top_candidate" in markdown
    assert commands["deployment_flags"] == (
        'apb flag-recommendations --model "G:/AI/models/Beta.gguf" '
        f"--output-dir {_path_arg(tmp_path)}"
    )
