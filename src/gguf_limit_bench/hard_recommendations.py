from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from gguf_limit_bench.deployment_readiness import build_deployment_readiness
from gguf_limit_bench.flag_recommendations import STANDARD_AGENT_CONTEXT
from gguf_limit_bench.qe_results import build_qe_leaderboard
from gguf_limit_bench.reports import build_leaderboard, build_report_audit, build_verdict


@dataclass(frozen=True)
class HardRecommendationOutputs:
    json_path: Path
    markdown_path: Path
    payload: dict[str, Any]


def write_hard_recommendations(
    runs_root: Path,
    *,
    target_model: str | None = None,
    target_model_path: str | None = None,
    required_context: int | None = None,
) -> HardRecommendationOutputs:
    runs_root.mkdir(parents=True, exist_ok=True)
    payload = build_hard_recommendations(
        runs_root,
        target_model=target_model,
        target_model_path=target_model_path,
        required_context=required_context,
    )
    json_path = runs_root / "hard-recommendations.json"
    markdown_path = runs_root / "hard-recommendations.md"
    _atomic_write_text(
        json_path,
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
    )
    _atomic_write_text(markdown_path, _markdown(payload))
    return HardRecommendationOutputs(
        json_path=json_path,
        markdown_path=markdown_path,
        payload=payload,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_hard_recommendations(
    runs_root: Path,
    *,
    target_model: str | None = None,
    target_model_path: str | None = None,
    required_context: int | None = None,
) -> dict[str, Any]:
    target_model = target_model or (Path(target_model_path).name if target_model_path else None)
    required_context = max(_int_value(required_context), STANDARD_AGENT_CONTEXT)
    all_leaderboard = build_leaderboard(runs_root)
    leaderboard = _scope_leaderboard(all_leaderboard, target_model)
    target_scope = _target_scope(
        target_model,
        all_leaderboard,
        leaderboard,
        target_model_path=target_model_path,
    )
    verdict = build_verdict(leaderboard)
    audit = build_report_audit(leaderboard)
    deployment = build_deployment_readiness(runs_root)
    benchmark_suite_preflight = _benchmark_suite_preflight(
        runs_root,
        target_model=target_model,
        required_context=required_context,
    )
    qe_gate = _qe_gate(runs_root)
    deployment_matches_model = _deployment_matches_verdict(
        deployment,
        verdict,
        target_model=target_model,
    )
    deployment_promoted = (
        deployment["action"] == "PROMOTE_DEPLOYMENT_PROFILE" and deployment_matches_model
    )
    context_gate = _context_gate(
        deployment,
        deployment_promoted=deployment_promoted,
        required_context=required_context,
        benchmark_suite_preflight=benchmark_suite_preflight,
    )
    resource_gate = _resource_gate(
        deployment,
        deployment_promoted=deployment_promoted,
        context_gate=context_gate,
    )
    runtime_warning_gate = _runtime_warning_gate(deployment)
    top_candidate_model_path = leaderboard.champion.model_path if leaderboard.entries else None
    top_candidate_context = _top_candidate_context(leaderboard)
    repeatability = _repeatability_summary(leaderboard)
    base_stack_promoted = (
        verdict.action == "PROMOTE"
        and deployment_promoted
        and context_gate["action"] == "PROMOTE_CONTEXT"
        and qe_gate["action"] == "PROMOTE_QE_PROFILE"
    )
    stability_gate = _stability_gate(repeatability, base_stack_promoted=base_stack_promoted)

    proven_components: list[dict[str, Any]] = []
    if verdict.action == "PROMOTE":
        proven_components.append(
            {
                "type": "model",
                "label": verdict.champion_model,
                "run_id": verdict.champion_run_id,
                "agent_quality_score": verdict.agent_quality_score,
                "general_score": verdict.general_score,
                "agentic_score": verdict.agentic_score,
                "prediction": verdict.prediction,
                "receipt_path": verdict.receipt_path,
            }
        )
    if deployment_promoted:
        profile = _recommended_profile_payload(deployment)
        proven_components.append(
            {
                "type": "settings_profile",
                "label": profile.get("label") if profile else deployment["recommended_profile_id"],
                "profile_id": deployment["recommended_profile_id"],
                "context_size": profile.get("context_size") if profile else None,
                "evidence": profile.get("evidence") if profile else None,
            }
        )
    if qe_gate["action"] == "PROMOTE_QE_PROFILE":
        proven_components.append(
            {
                "type": "qe_profile",
                "label": qe_gate["model"],
                "score": qe_gate["score"],
                "format_rate": qe_gate["format_rate"],
                "direct_answer_rate": qe_gate["direct_answer_rate"],
                "receipt_path": qe_gate["receipt_path"],
            }
        )

    next_actions = _next_actions(
        verdict,
        deployment,
        qe_gate,
        deployment_matches_model,
        stability_gate,
        context_gate,
        resource_gate,
        runtime_warning_gate,
        benchmark_suite_preflight,
        target_model=target_model,
    )
    proof_commands = _proof_commands(
        verdict,
        deployment,
        qe_gate,
        stability_gate,
        next_actions,
        context_gate,
        resource_gate,
        benchmark_suite_preflight,
        top_candidate_model_path=top_candidate_model_path,
        top_candidate_context=top_candidate_context,
        target_model=target_model,
        target_model_path=target_model_path,
        required_context=required_context,
        runs_root=runs_root,
    )
    overall_action = _overall_action(
        model_promoted=verdict.action == "PROMOTE",
        deployment_promoted=(
            deployment["action"] == "PROMOTE_DEPLOYMENT_PROFILE" and deployment_matches_model
        ),
        qe_promoted=qe_gate["action"] == "PROMOTE_QE_PROFILE",
        stability_promoted=stability_gate["action"] == "PROMOTE_STABILITY",
        context_promoted=context_gate["action"] == "PROMOTE_CONTEXT",
        resource_promoted=resource_gate["action"] == "PROMOTE_RESOURCE",
        proven_components=proven_components,
    )
    hard_recommendations = proven_components if overall_action == "PROMOTE_READY_STACK" else []
    candidate_rankings = _candidate_rankings(leaderboard)
    candidate_assessment = _candidate_assessment(
        verdict,
        deployment,
        qe_gate,
        stability_gate,
        deployment_promoted=deployment_promoted,
        deployment_matches_model=deployment_matches_model,
        context_gate=context_gate,
        resource_gate=resource_gate,
        runtime_warning_gate=runtime_warning_gate,
        benchmark_suite_preflight=benchmark_suite_preflight,
        target_model=target_model,
    )
    score_evidence = _score_evidence(
        candidate_rankings,
        hard_recommendations,
        proven_components,
    )
    performance_prediction = _performance_prediction(
        candidate_assessment,
        overall_action=overall_action,
    )
    settings_candidates = _settings_candidates(
        deployment,
        verdict,
        deployment_matches_model=deployment_matches_model,
        target_model=target_model,
        context_gate=context_gate,
        benchmark_suite_preflight=benchmark_suite_preflight,
    )
    return {
        "schema_version": 1,
        "target_scope": target_scope,
        "overall_action": overall_action,
        "operator_verdict": _operator_verdict(
            overall_action=overall_action,
            candidate_assessment=candidate_assessment,
            proof_commands=proof_commands,
            proven_components=proven_components,
        ),
        "summary": _summary(hard_recommendations, proven_components),
        "hard_recommendations": hard_recommendations,
        "proven_components": proven_components,
        "score_evidence": score_evidence,
        "performance_prediction": performance_prediction,
        "settings_candidates": settings_candidates,
        "candidate_rankings": candidate_rankings,
        "repeatability": repeatability,
        "stability_gate": stability_gate,
        "context_gate": context_gate,
        "resource_gate": resource_gate,
        "runtime_warning_gate": runtime_warning_gate,
        "candidate_assessment": candidate_assessment,
        "scorecard": {
            "quality": verdict.prediction["quality"],
            "speed": verdict.prediction["speed"],
            "context": verdict.prediction["context"],
            "recommendation": verdict.prediction["recommendation"],
            "agent_quality_score": verdict.agent_quality_score,
            "expected_generation_tps": verdict.expected_generation_tps,
            "expected_serving_tps": verdict.expected_serving_tps,
            "expected_cold_ttft_ms": verdict.expected_cold_ttft_ms,
            "expected_warm_ttft_ms": verdict.expected_warm_ttft_ms,
        },
        "model_gate": asdict(verdict),
        "deployment_gate": deployment,
        "benchmark_suite_preflight": benchmark_suite_preflight,
        "qe_gate": qe_gate,
        "report_audit": asdict(audit),
        "next_actions": next_actions,
        "proof_commands": proof_commands,
        "proof_runbook": _proof_runbook(proof_commands, runs_root=runs_root),
    }


def _scope_leaderboard(leaderboard, target_model: str | None):
    if not target_model:
        return leaderboard
    return type(leaderboard)(
        entries=[
            entry
            for entry in leaderboard.entries
            if _entry_matches_target_model(entry, target_model)
        ]
    )


def _target_scope(
    target_model: str | None,
    all_leaderboard,
    scoped_leaderboard,
    *,
    target_model_path: str | None = None,
) -> dict[str, Any]:
    if not target_model:
        scope = {
            "target_model": None,
            "status": "UNSCOPED",
            "matched_receipt_count": len(all_leaderboard.entries),
            "ignored_receipt_count": 0,
        }
        if target_model_path:
            scope["target_model_path"] = target_model_path
        return scope
    matched = len(scoped_leaderboard.entries)
    ignored = max(0, len(all_leaderboard.entries) - matched)
    scope = {
        "target_model": target_model,
        "status": "SCOPED" if matched else "NO_TARGET_EVIDENCE",
        "matched_receipt_count": matched,
        "ignored_receipt_count": ignored,
    }
    if target_model_path:
        scope["target_model_path"] = target_model_path
    return scope


def _entry_matches_target_model(entry, target_model: str) -> bool:
    target = _model_match_key(target_model)
    if not target:
        return True
    names = [entry.model_name, Path(entry.model_path).name, entry.model_path]
    return any(_model_keys_match(_model_match_key(name), target) for name in names if name)


def _model_keys_match(candidate: str, target: str) -> bool:
    if not candidate or not target:
        return False
    return candidate == target or target in candidate or candidate in target


def _model_match_key(value: object) -> str:
    text = str(value or "").lower().replace("\\", "/").rsplit("/", 1)[-1]
    if text.endswith(".gguf"):
        text = text[:-5]
    for char in ("_", ".", " "):
        text = text.replace(char, "-")
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-")


def _qe_gate(runs_root: Path) -> dict[str, Any]:
    leaderboard = build_qe_leaderboard(runs_root)
    champion = leaderboard.champion
    if champion is None:
        return {
            "action": "NO_QE_EVIDENCE",
            "model": None,
            "score": None,
            "format_rate": None,
            "direct_answer_rate": None,
            "attempts": 0,
            "receipt_path": None,
            "recommendation": "No QE fresh-session receipts found.",
            "next_run": "Run apb qe-format --model MODEL --base-url http://127.0.0.1:PORT.",
        }
    return asdict(champion)


def _benchmark_suite_preflight(
    runs_root: Path,
    *,
    target_model: str | None = None,
    required_context: int | None = None,
) -> dict[str, Any]:
    path = runs_root / "benchmark-suite-preflight.json"
    if not path.is_file():
        return {
            "status": "UNKNOWN",
            "ok": None,
            "issue_count": 0,
            "issues": [],
            "receipt_path": None,
            "next_action": "Run apb benchmark-suite-preflight --plan PLAN before deployment proof.",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "INVALID",
            "ok": False,
            "issue_count": 0,
            "issues": [],
            "receipt_path": str(path),
            "next_action": "Regenerate benchmark-suite preflight; the current receipt is invalid.",
        }
    if not isinstance(payload, dict):
        return {
            "status": "INVALID",
            "ok": False,
            "issue_count": 0,
            "issues": [],
            "receipt_path": str(path),
            "next_action": "Regenerate benchmark-suite preflight; the current receipt is invalid.",
        }
    payload.setdefault("receipt_path", str(path))
    payload.setdefault(
        "next_action",
        "Run apb benchmark-suite-preflight --plan PLAN before deployment proof.",
    )
    stale_reasons = _benchmark_preflight_stale_reasons(
        payload,
        target_model=target_model,
        required_context=required_context,
    )
    if stale_reasons:
        payload = dict(payload)
        payload["original_status"] = payload.get("status")
        payload["status"] = "STALE"
        payload["ok"] = False
        payload["stale_reasons"] = stale_reasons
        payload["next_action"] = (
            "Regenerate benchmark-suite preflight for this exact model/context/plan: "
            + "; ".join(stale_reasons)
        )
    return payload


def _benchmark_preflight_stale_reasons(
    payload: dict[str, Any],
    *,
    target_model: str | None,
    required_context: int | None,
) -> list[str]:
    reasons: list[str] = []
    if required_context is not None:
        receipt_context = _int_value(payload.get("context"))
        if receipt_context != _int_value(required_context):
            reasons.append(
                f"receipt context {receipt_context or 'unknown'} != required {required_context}"
            )
    if target_model:
        receipt_model = str(payload.get("model") or "")
        if not receipt_model or not _model_keys_match(
            _model_match_key(receipt_model),
            _model_match_key(target_model),
        ):
            reasons.append(
                f"receipt model `{receipt_model or 'unknown'}` does not match `{target_model}`"
            )
    return reasons


def _benchmark_preflight_blocks(preflight: dict[str, Any]) -> bool:
    return str(preflight.get("status") or "") in {"HARNESS_MISSING", "INVALID_PLAN", "STALE"}


def _recommended_profile_payload(deployment: dict[str, Any]) -> dict[str, Any] | None:
    recommended_id = deployment.get("recommended_profile_id")
    for profile in deployment.get("profiles", []):
        if profile.get("id") == recommended_id:
            return profile
    return None


def _deployment_matches_verdict(
    deployment: dict[str, Any],
    verdict,
    *,
    target_model: str | None = None,
) -> bool:
    champion_model = verdict.champion_model or target_model
    if champion_model is None:
        return False
    deployment_model_name = str(deployment.get("model_name") or "")
    deployment_model_path = str(deployment.get("model") or "")
    target_key = _model_match_key(champion_model)
    return any(
        _model_keys_match(_model_match_key(candidate), target_key)
        for candidate in (deployment_model_name, Path(deployment_model_path).name)
        if candidate
    )


def _top_candidate_context(leaderboard) -> int:
    if not leaderboard.entries:
        return STANDARD_AGENT_CONTEXT
    value = leaderboard.champion.settings.get("context_size")
    try:
        context = int(value)
    except (TypeError, ValueError):
        return STANDARD_AGENT_CONTEXT
    return max(context, STANDARD_AGENT_CONTEXT)


def _context_gate(
    deployment: dict[str, Any],
    *,
    deployment_promoted: bool,
    required_context: int,
    benchmark_suite_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = _recommended_profile_payload(deployment)
    evidence = profile.get("evidence") if isinstance(profile, dict) else None
    proven_context = _int_value(
        (evidence or {}).get("context") if isinstance(evidence, dict) else None
    )
    if deployment_promoted and proven_context >= required_context:
        return {
            "action": "PROMOTE_CONTEXT",
            "required_context": required_context,
            "proven_context": proven_context,
            "profile_id": profile.get("id") if isinstance(profile, dict) else None,
            "next_run": "Required deployment context is already proven.",
        }
    target_profile = _profile_for_required_context(deployment, required_context)
    profile_id = str(target_profile.get("id") or "standard") if target_profile else "standard"
    profile_context = _int_value(
        target_profile.get("context_size") if isinstance(target_profile, dict) else required_context
    )
    action = "RETEST_CONTEXT" if deployment_promoted else "WAITING_FOR_DEPLOYMENT"
    next_run = (
        _failed_profile_next_run(target_profile, benchmark_suite_preflight)
        if isinstance(target_profile, dict)
        and str(target_profile.get("status") or "") == "FAILED_PROOF"
        else (
            f"Run deployment proof for profile `{profile_id}` at {profile_context} context "
            "with score and serving/resource telemetry."
        )
    )
    return {
        "action": action,
        "required_context": required_context,
        "proven_context": proven_context or None,
        "profile_id": profile_id,
        "next_run": next_run,
    }


def _failed_profile_next_run(
    profile: dict[str, Any], benchmark_suite_preflight: dict[str, Any] | None = None
) -> str:
    profile_id = str(profile.get("id") or "the required profile")
    evidence = profile.get("evidence") if isinstance(profile.get("evidence"), dict) else {}
    failure = evidence.get("benchmark_suite_failure") or evidence.get("failure") or "failed proof"
    if (
        "harness_missing" in str(failure)
        and (benchmark_suite_preflight or {}).get("status") == "PASS"
    ):
        return (
            f"Rerun `{profile_id}` now that benchmark-suite preflight passes; "
            "the previous proof failed against an older unavailable harness plan."
        )
    return f"Fix `{profile_id}` proof failure ({failure}) and rerun the profile."


def _profile_for_required_context(
    deployment: dict[str, Any], required_context: int
) -> dict[str, Any] | None:
    profiles = [
        profile
        for profile in deployment.get("profiles", [])
        if isinstance(profile, dict) and _int_value(profile.get("context_size")) >= required_context
    ]
    if profiles:
        return sorted(profiles, key=lambda item: _int_value(item.get("context_size")))[0]
    profiles = [profile for profile in deployment.get("profiles", []) if isinstance(profile, dict)]
    if profiles:
        return sorted(
            profiles,
            key=lambda item: _int_value(item.get("context_size")),
            reverse=True,
        )[0]
    return None


def _resource_gate(
    deployment: dict[str, Any],
    *,
    deployment_promoted: bool,
    context_gate: dict[str, Any],
) -> dict[str, Any]:
    if not deployment_promoted:
        return {
            "action": "WAITING_FOR_DEPLOYMENT",
            "required": "same-run resource telemetry for the promoted settings receipt",
            "run_id": None,
            "resource_run_id": None,
            "next_run": "First promote a deployment profile with score and serving evidence.",
        }
    if context_gate["action"] != "PROMOTE_CONTEXT":
        return {
            "action": "WAITING_FOR_CONTEXT",
            "required": "same-run resource telemetry for the required context profile",
            "run_id": None,
            "resource_run_id": None,
            "next_run": str(context_gate["next_run"]),
        }
    profile = _recommended_profile_payload(deployment)
    evidence = profile.get("evidence") if isinstance(profile, dict) else None
    evidence = evidence if isinstance(evidence, dict) else {}
    run_id = evidence.get("run_id")
    resource_run_id = evidence.get("resource_run_id")
    resource_summary = evidence.get("resource_summary")
    if resource_summary and resource_run_id == run_id:
        return {
            "action": "PROMOTE_RESOURCE",
            "required": "same-run resource telemetry for the promoted settings receipt",
            "run_id": run_id,
            "resource_run_id": resource_run_id,
            "next_run": "Resource proof is satisfied for the promoted settings receipt.",
        }
    profile_id = deployment.get("recommended_profile_id") or (
        profile.get("id") if isinstance(profile, dict) else "standard"
    )
    return {
        "action": "RETEST_RESOURCE",
        "required": "same-run resource telemetry for the promoted settings receipt",
        "run_id": run_id,
        "resource_run_id": resource_run_id,
        "next_run": (
            f"Rerun deployment proof for profile `{profile_id}` with resource telemetry "
            "captured in the same receipt as the promoted score."
        ),
    }


def _runtime_warning_gate(deployment: dict[str, Any]) -> dict[str, Any]:
    critical: list[str] = []
    profiles: list[str] = []
    for profile in deployment.get("profiles", []):
        if not isinstance(profile, dict):
            continue
        evidence = profile.get("evidence")
        if not isinstance(evidence, dict):
            continue
        warnings = evidence.get("runtime_warnings")
        if not isinstance(warnings, dict):
            continue
        samples = warnings.get("critical")
        if not isinstance(samples, list) or not samples:
            continue
        profiles.append(str(profile.get("id") or "unknown"))
        critical.extend(str(sample) for sample in samples)
    if critical:
        profile_label = ", ".join(profiles)
        return {
            "action": "RETEST_RUNTIME_WARNINGS",
            "required": "no critical runtime warnings in promoted model/settings proof",
            "profiles": profiles,
            "critical": critical[:5],
            "next_run": (
                f"Fix critical runtime warnings for profile(s) {profile_label} and rerun "
                "deployment proof before promotion."
            ),
        }
    return {
        "action": "PASS_RUNTIME_WARNINGS",
        "required": "no critical runtime warnings in promoted model/settings proof",
        "profiles": [],
        "critical": [],
        "next_run": "Runtime-warning gate is satisfied.",
    }


def _overall_action(
    *,
    model_promoted: bool,
    deployment_promoted: bool,
    qe_promoted: bool,
    stability_promoted: bool,
    context_promoted: bool,
    resource_promoted: bool,
    proven_components: list[dict[str, Any]],
) -> str:
    if (
        model_promoted
        and deployment_promoted
        and context_promoted
        and resource_promoted
        and qe_promoted
        and stability_promoted
    ):
        return "PROMOTE_READY_STACK"
    if proven_components:
        return "PROMOTE_PARTIAL"
    return "RETEST"


def _candidate_assessment(
    verdict,
    deployment: dict[str, Any],
    qe_gate: dict[str, Any],
    stability_gate: dict[str, str],
    *,
    deployment_promoted: bool,
    deployment_matches_model: bool,
    context_gate: dict[str, Any],
    resource_gate: dict[str, Any],
    runtime_warning_gate: dict[str, Any],
    benchmark_suite_preflight: dict[str, Any],
    target_model: str | None = None,
) -> dict[str, Any]:
    model_promoted = verdict.action == "PROMOTE"
    qe_promoted = qe_gate["action"] == "PROMOTE_QE_PROFILE"
    stability_promoted = stability_gate["action"] == "PROMOTE_STABILITY"
    readiness_score = _readiness_score(
        model_promoted=model_promoted,
        deployment_promoted=deployment_promoted,
        context_promoted=context_gate["action"] == "PROMOTE_CONTEXT",
        resource_promoted=resource_gate["action"] == "PROMOTE_RESOURCE",
        qe_promoted=qe_promoted,
        stability_promoted=stability_promoted,
    )
    return {
        "model": verdict.champion_model or target_model,
        "run_id": verdict.champion_run_id,
        "readiness": _readiness_label(readiness_score),
        "readiness_score": readiness_score,
        "confidence": verdict.confidence,
        "known_performance": {
            "agent_quality_score": verdict.agent_quality_score,
            "general_score": verdict.general_score,
            "agentic_score": verdict.agentic_score,
            "generation_tps": verdict.expected_generation_tps,
            "serving_tps": verdict.expected_serving_tps,
            "cold_ttft_ms": verdict.expected_cold_ttft_ms,
            "warm_ttft_ms": verdict.expected_warm_ttft_ms,
            "context": _context_value(verdict.context_label),
            "quality": verdict.prediction["quality"],
            "speed": verdict.prediction["speed"],
            "context_class": verdict.prediction["context"],
            "recommendation_class": verdict.prediction["recommendation"],
        },
        "missing_evidence": _missing_evidence(
            verdict,
            deployment,
            qe_gate,
            stability_gate,
            deployment_promoted=deployment_promoted,
            deployment_matches_model=deployment_matches_model,
            context_gate=context_gate,
            resource_gate=resource_gate,
            runtime_warning_gate=runtime_warning_gate,
            benchmark_suite_preflight=benchmark_suite_preflight,
            target_model=target_model,
        ),
    }


def _readiness_score(
    *,
    model_promoted: bool,
    deployment_promoted: bool,
    context_promoted: bool,
    resource_promoted: bool,
    qe_promoted: bool,
    stability_promoted: bool,
) -> int:
    score = 0
    if model_promoted:
        score += 30
    if deployment_promoted:
        score += 20
    if context_promoted:
        score += 10
    if resource_promoted:
        score += 10
    if qe_promoted:
        score += 20
    if stability_promoted:
        score += 10
    return score


def _readiness_label(score: int) -> str:
    if score >= 100:
        return "ready_stack"
    if score >= 75:
        return "near_ready"
    if score >= 40:
        return "partial_candidate"
    return "not_recommendable"


def _missing_evidence(
    verdict,
    deployment: dict[str, Any],
    qe_gate: dict[str, Any],
    stability_gate: dict[str, str],
    *,
    deployment_promoted: bool,
    deployment_matches_model: bool,
    context_gate: dict[str, Any],
    resource_gate: dict[str, Any],
    runtime_warning_gate: dict[str, Any],
    benchmark_suite_preflight: dict[str, Any],
    target_model: str | None = None,
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    if _benchmark_preflight_blocks(benchmark_suite_preflight):
        missing.append(
            {
                "gate": "benchmark_suite_preflight",
                "status": str(benchmark_suite_preflight["status"]),
                "required": "all benchmark-suite command executables available before GPU proof",
                "next_action": str(benchmark_suite_preflight["next_action"]),
            }
        )
    if verdict.action != "PROMOTE":
        missing.append(
            {
                "gate": "model",
                "status": verdict.action,
                "required": (
                    "agent-quality score from benchmark-suite or recommendation-grade "
                    "librarian evidence"
                ),
                "next_action": verdict.next_run,
            }
        )
    if not deployment_promoted:
        status = (
            "MODEL_MISMATCH"
            if _deployment_model_mismatch(
                deployment,
                verdict,
                deployment_matches_model,
                target_model=target_model,
            )
            else str(deployment["action"])
        )
        model_label = verdict.champion_model or target_model
        next_action = (
            f"Run apb flag-recommendations for `{model_label}`."
            if status == "MODEL_MISMATCH"
            else str(deployment["next_run"])
        )
        missing.append(
            {
                "gate": "deployment",
                "status": status,
                "required": (
                    "matching scored receipt at the selected context plus serving telemetry"
                ),
                "next_action": next_action,
            }
        )
    if context_gate["action"] != "PROMOTE_CONTEXT":
        missing.append(
            {
                "gate": "context",
                "status": str(context_gate["action"]),
                "required": (
                    f"deployment proof at required context >= {context_gate['required_context']}"
                ),
                "next_action": str(context_gate["next_run"]),
            }
        )
    if resource_gate["action"] != "PROMOTE_RESOURCE":
        missing.append(
            {
                "gate": "resource",
                "status": str(resource_gate["action"]),
                "required": str(resource_gate["required"]),
                "next_action": str(resource_gate["next_run"]),
            }
        )
    if runtime_warning_gate["action"] != "PASS_RUNTIME_WARNINGS":
        missing.append(
            {
                "gate": "runtime_warnings",
                "status": str(runtime_warning_gate["action"]),
                "required": str(runtime_warning_gate["required"]),
                "next_action": str(runtime_warning_gate["next_run"]),
            }
        )
    if qe_gate["action"] != "PROMOTE_QE_PROFILE":
        missing.append(
            {
                "gate": "qe",
                "status": str(qe_gate["action"]),
                "required": (
                    "fresh-session QE score >= 0.90, format rate >= 0.90, direct answers == 0"
                ),
                "next_action": str(qe_gate["next_run"]),
            }
        )
    if stability_gate["action"] == "RETEST_STABILITY":
        missing.append(
            {
                "gate": "stability",
                "status": stability_gate["action"],
                "required": stability_gate["required"],
                "next_action": stability_gate["next_run"],
            }
        )
    return missing


def _context_value(context_label: str | None) -> int | None:
    if context_label is None:
        return None
    try:
        return int(context_label)
    except ValueError:
        return None


def _summary(
    hard_recommendations: list[dict[str, Any]],
    proven_components: list[dict[str, Any]],
) -> str:
    if not hard_recommendations:
        if proven_components:
            return (
                f"{len(proven_components)} component(s) are proven, but no full hard "
                "recommendation is deployable yet."
            )
        return "No hard recommendations are proven yet."
    return f"{len(hard_recommendations)} hard recommendation(s) are proven by current receipts."


def _candidate_rankings(leaderboard, limit: int = 5) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for entry in leaderboard.entries:
        candidate_key = _candidate_model_key(entry)
        if candidate_key in seen_models:
            continue
        seen_models.add(candidate_key)
        rank = len(rankings) + 1
        rankings.append(
            {
                "rank": rank,
                "model": entry.model_name,
                "model_path": entry.model_path,
                "run_id": entry.run_id,
                "status": entry.status,
                "score": entry.score,
                "agent_quality_score": entry.agent_bench_score,
                "general_score": entry.benchmark_suite_general_score,
                "agentic_score": entry.benchmark_suite_agentic_score,
                "context": _context_value(entry.context_label),
                "generation_tps": entry.generation_tps,
                "serving_tps": entry.serving_tps,
                "cold_ttft_ms": entry.serving_ttft_ms,
                "prediction": {
                    "quality": _quality_label(entry.agent_bench_score),
                    "speed": _speed_label(entry.serving_tps or entry.generation_tps),
                    "context": _context_label(entry.context_label),
                    "recommendation": _candidate_recommendation(entry),
                },
                "evidence_gaps": _candidate_evidence_gaps(entry),
                "receipt_path": entry.receipt_path,
            }
        )
        if len(rankings) >= limit:
            break
    return rankings


def _settings_candidates(
    deployment: dict[str, Any],
    verdict,
    *,
    deployment_matches_model: bool,
    context_gate: dict[str, Any],
    benchmark_suite_preflight: dict[str, Any],
    target_model: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    stale_model = _deployment_model_mismatch(
        deployment,
        verdict,
        deployment_matches_model,
        target_model=target_model,
    )
    source_model = str(
        deployment.get("model_name") or Path(str(deployment.get("model") or "")).name
    )
    target_model = verdict.champion_model or target_model
    for profile in deployment.get("profiles", []):
        if not isinstance(profile, dict):
            continue
        status = str(profile.get("status") or "UNKNOWN")
        profile_id = str(profile.get("id") or "unknown")
        display_status = "STALE_MODEL" if stale_model else status
        below_required_context = (
            not stale_model
            and display_status == "PROVEN"
            and context_gate["action"] == "RETEST_CONTEXT"
            and _int_value(profile.get("context_size"))
            < _int_value(context_gate.get("required_context"))
        )
        decision = (
            "baseline_below_required_context"
            if below_required_context
            else _settings_candidate_decision(display_status)
        )
        candidates.append(
            {
                "rank": 0,
                "profile_id": profile_id,
                "label": str(profile.get("label") or profile_id),
                "context_size": _int_value(profile.get("context_size")),
                "status": display_status,
                "decision": decision,
                "recommendation_score": profile.get("recommendation_score"),
                "evidence": None if stale_model else profile.get("evidence"),
                "source_model": source_model or None,
                "target_model": target_model,
                "reason": (
                    f"Flag recommendations are for {source_model}, not current top candidate {target_model}."
                    if stale_model
                    else str(profile.get("reason") or "")
                ),
                "next_action": _settings_candidate_next_action(
                    profile_id,
                    profile,
                    status=display_status,
                    target_model=target_model,
                    context_gate=context_gate if below_required_context else None,
                    benchmark_suite_preflight=benchmark_suite_preflight,
                ),
            }
        )
    ranked = sorted(candidates, key=_settings_candidate_sort_key)
    for index, candidate in enumerate(ranked, start=1):
        candidate["rank"] = index
    return ranked[:limit]


def _settings_candidate_decision(status: str) -> str:
    if status == "STALE_MODEL":
        return "regenerate_for_top_candidate"
    if status == "PROVEN":
        return "recommended"
    if status == "REJECTED":
        return "rejected"
    if status == "RUNTIME_WARNING":
        return "fix_runtime_warning"
    if status == "FAILED_PROOF":
        return "fix_failed_proof"
    if status == "QUALITY_ONLY":
        return "needs_serving_proof"
    if status == "SYSTEMS_ONLY":
        return "needs_agent_score"
    if status == "UNPROVEN":
        return "next_to_test"
    return "blocked"


def _settings_candidate_next_action(
    profile_id: str,
    profile: dict[str, Any],
    *,
    status: str | None = None,
    target_model: str | None = None,
    context_gate: dict[str, Any] | None = None,
    benchmark_suite_preflight: dict[str, Any] | None = None,
) -> str:
    status = status or str(profile.get("status") or "UNKNOWN")
    context = _int_value(profile.get("context_size"))
    if status == "STALE_MODEL":
        model = target_model or "the current top candidate"
        return f"Run apb flag-recommendations for `{model}` before testing profile `{profile_id}`."
    if context_gate is not None:
        return (
            "This profile is proven only below the required context; run "
            f"`{context_gate.get('profile_id') or 'the required profile'}` next."
        )
    if status == "PROVEN":
        return "Keep as a deployable settings profile; repeat stability proof before production."
    if status == "REJECTED":
        return (
            f"Do not use profile `{profile_id}` until a newer scored receipt replaces the failure."
        )
    if status == "RUNTIME_WARNING":
        evidence = profile.get("evidence") if isinstance(profile.get("evidence"), dict) else {}
        warnings = evidence.get("runtime_warnings") if isinstance(evidence, dict) else {}
        critical = warnings.get("critical") if isinstance(warnings, dict) else None
        sample = f" ({critical[0]})" if isinstance(critical, list) and critical else ""
        return f"Fix runtime warnings for profile `{profile_id}`{sample} and rerun the proof."
    if status == "FAILED_PROOF":
        evidence = profile.get("evidence") if isinstance(profile.get("evidence"), dict) else {}
        failure = (
            evidence.get("benchmark_suite_failure") or evidence.get("failure") or "failed proof"
        )
        if (
            "harness_missing" in str(failure)
            and (benchmark_suite_preflight or {}).get("status") == "PASS"
        ):
            return (
                f"Rerun profile `{profile_id}` now that benchmark-suite preflight passes; "
                "the previous proof used an unavailable harness plan."
            )
        return f"Fix `{profile_id}` proof failure ({failure}) and rerun the profile."
    if status == "QUALITY_ONLY":
        return f"Rerun profile `{profile_id}` with serving/resource telemetry enabled."
    if status == "SYSTEMS_ONLY":
        return f"Rerun profile `{profile_id}` with benchmark-suite or librarian score evidence."
    if status == "UNPROVEN":
        return (
            f"Run deployment proof for profile `{profile_id}` at {context} context with "
            "score and serving/resource telemetry."
        )
    return f"Refresh deployment readiness for profile `{profile_id}`."


def _settings_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float, int]:
    decision_priority = {
        "recommended": 0,
        "fix_failed_proof": 1,
        "fix_runtime_warning": 2,
        "next_to_test": 3,
        "needs_serving_proof": 4,
        "baseline_below_required_context": 5,
        "needs_agent_score": 6,
        "regenerate_for_top_candidate": 7,
        "blocked": 8,
        "rejected": 9,
    }
    profile_priority = {"standard": 0, "long_agent": 1, "over_the_top": 2, "bare_minimum": 3}
    score = _float_value(candidate.get("recommendation_score"))
    score_sort = -(score or 0.0) if candidate.get("decision") == "recommended" else 0.0
    return (
        decision_priority.get(str(candidate.get("decision")), 99),
        score_sort,
        profile_priority.get(str(candidate.get("profile_id")), 99),
        -int(candidate.get("context_size") or 0),
    )


def _score_evidence(
    candidate_rankings: list[dict[str, Any]],
    hard_recommendations: list[dict[str, Any]],
    proven_components: list[dict[str, Any]],
) -> dict[str, Any]:
    top = candidate_rankings[0] if candidate_rankings else {}
    return {
        "candidate_count": len(candidate_rankings),
        "scored_candidate_count": sum(
            1 for item in candidate_rankings if item.get("agent_quality_score") is not None
        ),
        "proven_recommendation_count": len(hard_recommendations),
        "top_agent_quality_score": top.get("agent_quality_score"),
        "top_general_score": top.get("general_score"),
        "top_agentic_score": top.get("agentic_score"),
        "top_generation_tps": top.get("generation_tps"),
        "top_serving_tps": top.get("serving_tps"),
        "proven_component_count": len(proven_components),
    }


def _operator_verdict(
    *,
    overall_action: str,
    candidate_assessment: dict[str, Any],
    proof_commands: list[dict[str, str]],
    proven_components: list[dict[str, Any]],
) -> dict[str, str | None]:
    next_command = proof_commands[0]["command"] if proof_commands else None
    if overall_action == "PROMOTE_READY_STACK":
        return {
            "status": "READY_TO_USE",
            "headline": "A deployable recommendation is proven.",
            "why": (
                "Model quality, deployment settings, QE behavior, and stability all passed "
                "the current gates."
            ),
            "next_command": None,
        }
    if proven_components:
        missing = (
            ", ".join(item["gate"] for item in candidate_assessment.get("missing_evidence", []))
            or "stack"
        )
        return {
            "status": "PARTIAL_NOT_PRODUCTION",
            "headline": "Some evidence is proven, but the stack is not production-ready.",
            "why": f"Missing required proof: {missing}.",
            "next_command": next_command,
        }
    return {
        "status": "NOT_USABLE_YET",
        "headline": "No deployable recommendation exists.",
        "why": (
            "The best receipt is still missing score-backed model proof, deployment proof, "
            "QE proof, or stability proof."
        ),
        "next_command": next_command,
    }


def _performance_prediction(
    candidate_assessment: dict[str, Any], *, overall_action: str
) -> dict[str, str]:
    performance = candidate_assessment.get("known_performance") or {}
    missing = candidate_assessment.get("missing_evidence") or []
    missing_gates = [
        str(item.get("gate")) for item in missing if isinstance(item, dict) and item.get("gate")
    ]
    agent_score = performance.get("agent_quality_score")
    quality = str(performance.get("quality") or "unmeasured")
    speed = str(performance.get("speed") or "unmeasured")
    context_class = str(performance.get("context_class") or "unmeasured")
    generation_tps = _number_label(performance.get("generation_tps"), digits=2)
    serving_tps = _number_label(performance.get("serving_tps"), digits=2)
    context = performance.get("context")
    if overall_action == "PROMOTE_READY_STACK":
        status = "READY_AGENT_STACK"
        risk = "low"
        deployment_expectation = "deployable"
        expected = (
            "Score-backed model quality, deployment proof, QE proof, and repeatability are "
            "all present. This is the current deployable local stack."
        )
    elif agent_score is not None:
        status = "PARTIAL_STACK"
        risk = "medium"
        deployment_expectation = "lab_candidate"
        expected = (
            "Agent quality is measured, but the missing gates still block production use. "
            "Treat this as a lab candidate until the remaining proof is collected."
        )
    else:
        status = "LAB_ONLY_SPEED_PROOF"
        risk = "high"
        deployment_expectation = "do_not_deploy"
        expected = (
            "Likely responsive token streaming, but agent quality is unmeasured. "
            "Use only for lab probing until benchmark-suite or librarian evidence exists."
        )
    return {
        "status": status,
        "risk": risk,
        "deployment_expectation": deployment_expectation,
        "expected_user_experience": expected,
        "quality_basis": (
            f"agent-quality score {_number_label(agent_score, digits=4)}; class {quality}"
            if agent_score is not None
            else "No agent-quality score is present."
        ),
        "speed_basis": (
            f"serving {serving_tps} tok/s; generation {generation_tps} tok/s; class {speed}"
        ),
        "context_basis": f"{context if context is not None else 'unmeasured'} context; class {context_class}",
        "missing_basis": ", ".join(missing_gates) if missing_gates else "none",
    }


def _number_label(value: object, *, digits: int) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "unmeasured"


def _candidate_model_key(entry) -> str:
    return str(entry.model_path or entry.model_name).casefold()


def _candidate_stack_key(entry) -> str:
    settings = entry.settings if isinstance(entry.settings, dict) else {}
    stable_settings = {
        "model": _candidate_model_key(entry),
        "context_size": _context_value(entry.context_label) or settings.get("context_size"),
        "profile_name": settings.get("profile_name"),
        "parallel": settings.get("parallel"),
        "gpu_layers": settings.get("gpu_layers"),
        "batch_size": settings.get("batch_size"),
        "ubatch_size": settings.get("ubatch_size"),
        "flash_attention": settings.get("flash_attention"),
        "kv_unified": settings.get("kv_unified"),
        "cache_type_k": settings.get("cache_type_k"),
        "cache_type_v": settings.get("cache_type_v"),
        "extra_server_args": settings.get("extra_server_args"),
    }
    return json.dumps(stable_settings, sort_keys=True, default=str)


def _repeatability_evidence_key(entry) -> str:
    if entry.agent_bench_score is not None:
        return f"agent_quality:{entry.benchmark_suite_status}:{entry.agent_quality_gate}"
    return f"unscored:{entry.status}"


def _repeatability_summary(leaderboard) -> dict[str, Any]:
    if not leaderboard.entries:
        return {
            "model": None,
            "run_count": 0,
            "confidence": "no_evidence",
            "summary": "No receipts are available for repeatability analysis.",
        }
    champion = leaderboard.champion
    stack_key = _candidate_stack_key(champion)
    evidence_key = _repeatability_evidence_key(champion)
    entries = [
        entry
        for entry in leaderboard.entries
        if _candidate_stack_key(entry) == stack_key
        and _repeatability_evidence_key(entry) == evidence_key
    ]
    score = _metric_range([entry.score for entry in entries])
    generation_tps = _metric_range([entry.generation_tps for entry in entries])
    serving_tps = _metric_range(
        [entry.serving_tps for entry in entries if entry.serving_tps is not None]
    )
    cold_ttft_ms = _metric_range(
        [entry.serving_ttft_ms for entry in entries if entry.serving_ttft_ms is not None]
    )
    confidence = _repeatability_confidence(
        run_count=len(entries),
        ranges=[score, generation_tps, serving_tps, cold_ttft_ms],
    )
    return {
        "model": champion.model_name,
        "model_path": champion.model_path,
        "stack_fingerprint": stack_key,
        "evidence_fingerprint": evidence_key,
        "run_count": len(entries),
        "confidence": confidence,
        "score": score,
        "generation_tps": generation_tps,
        "serving_tps": serving_tps,
        "cold_ttft_ms": cold_ttft_ms,
        "summary": _repeatability_summary_text(confidence, len(entries)),
        "receipt_paths": [entry.receipt_path for entry in entries],
    }


def _metric_range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "spread_pct": None}
    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)
    spread_pct = None if mean == 0 else (maximum - minimum) / abs(mean)
    return {
        "min": round(minimum, 6),
        "max": round(maximum, 6),
        "mean": round(mean, 6),
        "spread_pct": None if spread_pct is None else round(spread_pct, 6),
    }


def _repeatability_confidence(*, run_count: int, ranges: list[dict[str, float | None]]) -> str:
    if run_count <= 0:
        return "no_evidence"
    if run_count == 1:
        return "single_run"
    if run_count < 3:
        return "limited"
    spreads = [float(item["spread_pct"]) for item in ranges if item.get("spread_pct") is not None]
    if not spreads:
        return "limited"
    return "repeatable" if max(spreads) <= 0.15 else "variable"


def _repeatability_summary_text(confidence: str, run_count: int) -> str:
    if confidence == "repeatable":
        return f"{run_count} comparable receipts show low spread across measured metrics."
    if confidence == "variable":
        return f"{run_count} comparable receipts exist, but measured metrics vary too much."
    if confidence == "limited":
        return f"Only {run_count} comparable receipts exist; repeatability is weak."
    if confidence == "single_run":
        return "Only one receipt exists for this model; repeatability is unproven."
    return "No receipts are available for repeatability analysis."


def _stability_gate(repeatability: dict[str, Any], *, base_stack_promoted: bool) -> dict[str, str]:
    if not base_stack_promoted:
        return {
            "action": "WAITING_FOR_PROMOTED_STACK",
            "confidence": str(repeatability.get("confidence") or "no_evidence"),
            "required": "model, deployment, and QE gates must pass before stability can promote",
            "next_run": "First promote the model, deployment profile, and QE gates.",
        }
    confidence = str(repeatability.get("confidence") or "no_evidence")
    if confidence == "repeatable":
        return {
            "action": "PROMOTE_STABILITY",
            "confidence": confidence,
            "required": "at least 3 comparable receipts with repeatable measured metrics",
            "next_run": "Stability proof is satisfied for the current top model.",
        }
    run_count = int(repeatability.get("run_count") or 0)
    return {
        "action": "RETEST_STABILITY",
        "confidence": confidence,
        "required": "at least 3 comparable receipts with repeatable measured metrics",
        "next_run": (
            "Repeat the promoted model/settings until there are at least 3 comparable "
            f"receipts with repeatable measured metrics; current comparable receipts: {run_count}."
        ),
    }


def _quality_label(score: float | None) -> str:
    if score is None:
        return "unmeasured"
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "usable"
    return "weak"


def _speed_label(tps: float | None) -> str:
    if tps is None or tps <= 0.0:
        return "unmeasured"
    if tps >= 25.0:
        return "interactive"
    if tps >= 10.0:
        return "slow_interactive"
    return "batch_only"


def _context_label(context_label: str) -> str:
    context = _context_value(context_label)
    if context is None:
        return "unmeasured"
    if context >= 131_072:
        return "long_agentic"
    if context >= 65_536:
        return "agentic"
    if context >= 32_768:
        return "basic_agentic"
    return "short"


def _candidate_recommendation(entry) -> str:
    if entry.status == "SUITE FAILED" or entry.benchmark_suite_status == "fail":
        return "do_not_use"
    if entry.agent_bench_score is not None:
        return "score_backed_candidate"
    return "needs_agent_benchmark"


def _candidate_evidence_gaps(entry) -> list[str]:
    gaps: list[str] = []
    if entry.agent_bench_score is None:
        gaps.append("agent_quality")
    if entry.benchmark_suite_status != "pass":
        gaps.append("benchmark_suite")
    if entry.serving_ttft_ms is None and entry.serving_tps is None:
        gaps.append("serving")
    context = _context_value(entry.context_label) or 0
    if context < STANDARD_AGENT_CONTEXT:
        gaps.append("long_context")
    return gaps


def _int_value(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _next_actions(
    verdict,
    deployment: dict[str, Any],
    qe_gate: dict[str, Any],
    deployment_matches_model: bool,
    stability_gate: dict[str, str],
    context_gate: dict[str, Any],
    resource_gate: dict[str, Any],
    runtime_warning_gate: dict[str, Any],
    benchmark_suite_preflight: dict[str, Any],
    *,
    target_model: str | None = None,
) -> list[str]:
    actions: list[str] = []
    if _benchmark_preflight_blocks(benchmark_suite_preflight):
        actions.append(str(benchmark_suite_preflight["next_action"]))
    if verdict.action != "PROMOTE":
        actions.append(verdict.next_run)
    if _deployment_model_mismatch(
        deployment,
        verdict,
        deployment_matches_model,
        target_model=target_model,
    ):
        model_label = verdict.champion_model or target_model
        actions.append(f"Run apb flag-recommendations for `{model_label}`.")
    elif deployment["action"] != "PROMOTE_DEPLOYMENT_PROFILE":
        actions.append(str(deployment["next_run"]))
    if context_gate["action"] == "RETEST_CONTEXT":
        actions.append(str(context_gate["next_run"]))
    if resource_gate["action"] == "RETEST_RESOURCE":
        actions.append(str(resource_gate["next_run"]))
    if runtime_warning_gate["action"] != "PASS_RUNTIME_WARNINGS":
        actions.append(str(runtime_warning_gate["next_run"]))
    if qe_gate["action"] != "PROMOTE_QE_PROFILE":
        actions.append(str(qe_gate["next_run"]))
    if stability_gate["action"] == "RETEST_STABILITY":
        actions.append(stability_gate["next_run"])
    return _dedupe(actions)


def _deployment_model_mismatch(
    deployment: dict[str, Any],
    verdict,
    deployment_matches_model: bool,
    *,
    target_model: str | None = None,
) -> bool:
    model_label = verdict.champion_model or target_model
    if deployment_matches_model or model_label is None:
        return False
    return bool(deployment.get("model") or deployment.get("model_name"))


def _proof_commands(
    verdict,
    deployment: dict[str, Any],
    qe_gate: dict[str, Any],
    stability_gate: dict[str, str],
    next_actions: list[str],
    context_gate: dict[str, Any],
    resource_gate: dict[str, Any],
    benchmark_suite_preflight: dict[str, Any],
    *,
    top_candidate_model_path: str | None,
    top_candidate_context: int,
    target_model: str | None = None,
    target_model_path: str | None = None,
    required_context: int | None = None,
    runs_root: Path = Path("_runs"),
) -> list[dict[str, str]]:
    if not next_actions:
        return []
    commands: list[dict[str, str]] = []
    runs_root_arg = _command_path_arg(runs_root)
    flag_recommendations_arg = _command_path_arg(runs_root / "flag-recommendations.json")
    model_label = verdict.champion_model or target_model or "MODEL"
    model_argument = target_model_path or top_candidate_model_path or target_model or model_label
    plan_context = max(top_candidate_context, _int_value(required_context))
    benchmark_suite_plan = "benchmark-suite.plan.json"
    if verdict.action != "PROMOTE":
        commands.append(
            {
                "id": "model_plan",
                "gate": "model",
                "purpose": "Create an editable score-backed benchmark-suite plan for the target model.",
                "command": (
                    "apb benchmark-suite-template --output benchmark-suite.plan.json "
                    f'--model "{model_argument}" --base-url http://127.0.0.1:8080/v1 '
                    f"--context {plan_context}"
                ),
                "context_size": str(plan_context),
                "context_target": f"required_context_{plan_context}",
                "then": "Edit the generated plan if needed; the next proof command runs it.",
            }
        )
        commands.append(
            {
                "id": "model_score",
                "gate": "model",
                "purpose": "Run the scored benchmark-suite plan and write recommendation-grade receipts.",
                "command": (
                    "apb benchmark-suite --plan benchmark-suite.plan.json "
                    f"--runs-root {runs_root_arg}"
                ),
                "then": "Require a passing suite verdict before treating the model as recommended.",
            }
        )
    elif context_gate["action"] == "RETEST_CONTEXT":
        benchmark_suite_plan = f"benchmark-suite-{plan_context}.plan.json"
        commands.append(
            {
                "id": "context_plan",
                "gate": "context",
                "purpose": "Create a score-backed benchmark-suite plan at the required context.",
                "command": (
                    f"apb benchmark-suite-template --output {benchmark_suite_plan} "
                    f'--model "{model_argument}" --base-url http://127.0.0.1:8080/v1 '
                    f"--context {plan_context}"
                ),
                "context_size": str(plan_context),
                "context_target": f"required_context_{plan_context}",
                "then": "Use this plan in the required-context deployment proof.",
            }
        )
    model_mismatch = deployment.get("model_name") != model_label
    deployment_needs_proof = (
        deployment["action"] != "PROMOTE_DEPLOYMENT_PROFILE"
        or context_gate["action"] == "RETEST_CONTEXT"
        or resource_gate["action"] == "RETEST_RESOURCE"
        or (deployment["action"] == "PROMOTE_DEPLOYMENT_PROFILE" and model_mismatch)
    )
    if deployment_needs_proof:
        commands.append(
            {
                "id": "benchmark_suite_preflight",
                "gate": "benchmark_suite_preflight",
                "purpose": "Check benchmark-suite harness commands before launching the model.",
                "command": (
                    f"apb benchmark-suite-preflight --plan {benchmark_suite_plan} "
                    f"--runs-root {runs_root_arg}"
                ),
                "then": "Only run deployment proof after this preflight returns PASS.",
            }
        )
        model_path = str(deployment.get("model") or "<MODEL.gguf>")
        if model_mismatch:
            model_path = (
                top_candidate_model_path
                or target_model_path
                or target_model
                or f"<path-to-{model_label}>"
            )
        if deployment["action"] != "PROMOTE_DEPLOYMENT_PROFILE" or model_mismatch:
            commands.append(
                {
                    "id": "deployment_flags",
                    "gate": "deployment",
                    "purpose": "Generate llama.cpp flag profiles for the model that needs deployment proof.",
                    "command": (
                        f'apb flag-recommendations --model "{model_path}" '
                        f"--output-dir {runs_root_arg}"
                    ),
                    "then": (
                        "Run the recommended profile, collect serving telemetry, then rerun "
                        f"`apb deployment-readiness --runs-root {runs_root_arg}`."
                    ),
                }
            )
        profile_id = (
            "standard"
            if model_mismatch and model_label != "MODEL"
            else (
                str(context_gate.get("profile_id") or "")
                if context_gate["action"] == "RETEST_CONTEXT"
                else str((deployment.get("recommended_profile_id") or "standard"))
                if resource_gate["action"] == "RETEST_RESOURCE"
                else (_next_deployment_profile_id(deployment) or "standard")
            )
        )
        profile_id = profile_id or "standard"
        gate = (
            "context"
            if context_gate["action"] == "RETEST_CONTEXT" and not model_mismatch
            else "resource"
            if resource_gate["action"] == "RETEST_RESOURCE" and not model_mismatch
            else "deployment"
        )
        then = (
            f"Run `apb hard-recommendations --runs-root {runs_root_arg}` and require PROMOTE_CONTEXT."
            if gate == "context"
            else f"Run `apb hard-recommendations --runs-root {runs_root_arg}` and require PROMOTE_RESOURCE."
            if gate == "resource"
            else (
                f"Run `apb deployment-readiness --runs-root {runs_root_arg}` "
                "and require PROMOTE_DEPLOYMENT_PROFILE."
            )
        )
        commands.append(
            {
                "id": "deployment_proof",
                "gate": gate,
                "purpose": (
                    "Run the selected flag profile and collect scored serving proof. "
                    "This is the receipt-producing step; flag generation alone is not proof."
                ),
                "command": (
                    f"apb deployment-proof --profile {profile_id} --runs-root {runs_root_arg} "
                    f"--flag-recommendations {flag_recommendations_arg} "
                    f"--benchmark-suite-plan {benchmark_suite_plan} --budget-minutes 30"
                ),
                "then": then,
            }
        )
    if qe_gate["action"] != "PROMOTE_QE_PROFILE":
        qe_model = qe_gate.get("model") or "QE_MODEL"
        commands.append(
            {
                "id": "qe_format",
                "gate": "qe",
                "purpose": "Retest the query-expansion lane in fresh sessions.",
                "command": (
                    f"apb qe-format --model {qe_model} --base-url http://127.0.0.1:8080 "
                    f"--runs-root {runs_root_arg} --repeats 10 --max-tokens 128"
                ),
                "then": (
                    f"Run `apb qe-results --runs-root {runs_root_arg}` and require "
                    "PROMOTE_QE_PROFILE before deployment."
                ),
            }
        )
    if stability_gate["action"] == "RETEST_STABILITY" and verdict.action == "PROMOTE":
        profile_id = str(deployment.get("recommended_profile_id") or "standard")
        commands.append(
            {
                "id": "stability_repeat",
                "gate": "stability",
                "purpose": "Repeat the promoted model/settings until the stability gate is satisfied.",
                "command": (
                    f"apb deployment-proof --profile {profile_id} --runs-root {runs_root_arg} "
                    f"--flag-recommendations {flag_recommendations_arg} "
                    f"--benchmark-suite-plan {benchmark_suite_plan} --budget-minutes 30"
                ),
                "then": (
                    f"Rerun `apb hard-recommendations --runs-root {runs_root_arg}` "
                    "and require PROMOTE_STABILITY."
                ),
            }
        )
    commands.append(
        {
            "id": "refresh_hard_recommendations",
            "gate": "refresh",
            "purpose": "Rebuild the consolidated recommendation after new receipts are present.",
            "command": _refresh_hard_recommendations_command(
                target_model,
                target_model_path=target_model_path,
                required_context=required_context,
                runs_root=runs_root,
            ),
            "then": "Only deploy when this report returns PROMOTE_READY_STACK or an intentional partial promotion.",
        }
    )
    return commands


def _refresh_hard_recommendations_command(
    target_model: str | None,
    *,
    target_model_path: str | None = None,
    required_context: int | None = None,
    runs_root: Path = Path("_runs"),
) -> str:
    command = f"apb hard-recommendations --runs-root {_command_path_arg(runs_root)}"
    if target_model:
        command += f' --target-model "{target_model}"'
    if target_model_path:
        command += f' --target-model-path "{target_model_path}"'
    if required_context and required_context != STANDARD_AGENT_CONTEXT:
        command += f" --required-context {required_context}"
    return command


def _proof_runbook(
    commands: list[dict[str, str]], *, runs_root: Path = Path("_runs")
) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "id": command["id"],
            "gate": command["gate"],
            "status": "pending",
            "command": command["command"],
            "proves": _proof_artifact(command["id"], runs_root=runs_root),
            "success_condition": _proof_success_condition(command["id"]),
            "next": command["then"],
        }
        for index, command in enumerate(commands, start=1)
    ]


def _proof_artifact(command_id: str, *, runs_root: Path = Path("_runs")) -> str:
    runs_root_label = _path_label(runs_root)
    artifacts = {
        "model_plan": "benchmark-suite.plan.json",
        "context_plan": "benchmark-suite-<required-context>.plan.json",
        "benchmark_suite_preflight": f"{runs_root_label}/benchmark-suite-preflight.json",
        "model_score": f"{runs_root_label}/<suite-run>/suite-verdict.json",
        "deployment_flags": f"{runs_root_label}/flag-recommendations.json",
        "deployment_proof": f"{runs_root_label}/<deployment-proof-run>/best-settings.json",
        "qe_format": f"{runs_root_label}/<qe-format-run>/qe-format-summary.json",
        "stability_repeat": f"{runs_root_label}/<deployment-proof-run>/best-settings.json",
        "refresh_hard_recommendations": f"{runs_root_label}/hard-recommendations.json",
    }
    return artifacts.get(command_id, f"{runs_root_label}/<receipt>")


def _path_label(path: Path) -> str:
    return Path(path).as_posix()


def _command_path_arg(path: Path) -> str:
    label = _path_label(path)
    return f'"{label}"' if any(char.isspace() for char in label) else label


def _proof_success_condition(command_id: str) -> str:
    conditions = {
        "model_plan": "Plan file exists and names the target model/context.",
        "context_plan": "Required-context plan file exists and names the target model/context.",
        "benchmark_suite_preflight": "benchmark-suite preflight status is PASS.",
        "model_score": "suite-verdict action is PROMOTE.",
        "deployment_flags": "flag-recommendations.json contains profiles for the target model.",
        "deployment_proof": (
            "deployment-readiness action is PROMOTE_DEPLOYMENT_PROFILE for the same model."
        ),
        "qe_format": "qe-results action is PROMOTE_QE_PROFILE.",
        "stability_repeat": "stability_gate action is PROMOTE_STABILITY.",
        "refresh_hard_recommendations": (
            "hard-recommendations overall_action is PROMOTE_READY_STACK or an intentional "
            "PROMOTE_PARTIAL."
        ),
    }
    return conditions.get(command_id, "Receipt exists and the related gate is promoted.")


def _next_deployment_profile_id(deployment: dict[str, Any]) -> str | None:
    priority = {"standard": 0, "long_agent": 1, "over_the_top": 2, "bare_minimum": 3}
    profiles = [
        profile
        for profile in deployment.get("profiles", [])
        if isinstance(profile, dict) and profile.get("status") != "PROVEN"
    ]
    if not profiles:
        return None
    profile = sorted(profiles, key=lambda item: priority.get(str(item.get("id")), 99))[0]
    return str(profile.get("id") or "standard")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Hard Recommendations",
        "",
        f"- Action: `{payload['overall_action']}`",
        f"- Summary: {payload['summary']}",
        f"- Proven recommendations: `{len(payload['hard_recommendations'])}`",
        f"- Proven components: `{len(payload.get('proven_components') or [])}`",
        "",
        "## Operator Verdict",
        "",
        f"- Status: `{payload['operator_verdict']['status']}`",
        f"- Headline: {payload['operator_verdict']['headline']}",
        f"- Why: {payload['operator_verdict']['why']}",
        f"- Next command: `{payload['operator_verdict']['next_command'] or 'none'}`",
        "",
        "## Score Evidence",
        "",
        (
            f"- Scored candidates: `{payload['score_evidence']['scored_candidate_count']}/"
            f"{payload['score_evidence']['candidate_count']}`"
        ),
        f"- Proven recommendations: `{payload['score_evidence']['proven_recommendation_count']}`",
        f"- Proven components: `{payload['score_evidence'].get('proven_component_count', 0)}`",
        f"- Top agent score: `{_fmt(payload['score_evidence']['top_agent_quality_score'])}`",
        f"- Top general score: `{_fmt(payload['score_evidence']['top_general_score'])}`",
        f"- Top agentic score: `{_fmt(payload['score_evidence']['top_agentic_score'])}`",
        f"- Top generation: `{_fmt(payload['score_evidence']['top_generation_tps'])}` tok/s",
        f"- Top serving: `{_fmt(payload['score_evidence']['top_serving_tps'])}` tok/s",
        "",
        "## Performance Prediction",
        "",
        f"- Status: `{payload['performance_prediction']['status']}`",
        f"- Risk: `{payload['performance_prediction']['risk']}`",
        f"- Deployment expectation: `{payload['performance_prediction']['deployment_expectation']}`",
        f"- Expected user experience: {payload['performance_prediction']['expected_user_experience']}",
        f"- Quality basis: {payload['performance_prediction']['quality_basis']}",
        f"- Speed basis: {payload['performance_prediction']['speed_basis']}",
        f"- Context basis: {payload['performance_prediction']['context_basis']}",
        f"- Missing basis: {payload['performance_prediction']['missing_basis']}",
        "",
        "## Settings Candidates",
        "",
        "| Rank | Profile | Context | Status | Decision | Score | Evidence | Reason |",
        "| ---: | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    if payload["settings_candidates"]:
        for item in payload["settings_candidates"]:
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else None
            evidence_label = (
                f"{evidence.get('run_id')} @ {evidence.get('context')}" if evidence else "none"
            )
            lines.append(
                f"| {item['rank']} | `{item['profile_id']}` | {item['context_size']} | "
                f"`{item['status']}` | `{item['decision']}` | "
                f"`{_fmt(item.get('recommendation_score'))}` | `{evidence_label}` | "
                f"{item.get('reason') or ''} |"
            )
    else:
        lines.append(
            "| - | none | 0 | `NO_FLAG_RECOMMENDATIONS` | `blocked` | "
            "`unmeasured` | `none` | Run flag recommendations first. |"
        )
    lines.extend(
        [
            "",
            "## Scorecard",
            "",
            f"- Quality: `{payload['scorecard']['quality']}`",
            f"- Speed: `{payload['scorecard']['speed']}`",
            f"- Context: `{payload['scorecard']['context']}`",
            f"- Recommendation class: `{payload['scorecard']['recommendation']}`",
            f"- Agent quality score: `{_fmt(payload['scorecard']['agent_quality_score'])}`",
            f"- Expected generation: `{_fmt(payload['scorecard']['expected_generation_tps'])}` tok/s",
            f"- Expected serving: `{_fmt(payload['scorecard']['expected_serving_tps'])}` tok/s",
            "",
            "## Repeatability",
            "",
            f"- Model: `{payload['repeatability']['model'] or 'none'}`",
            f"- Runs: `{payload['repeatability']['run_count']}`",
            f"- Confidence: `{payload['repeatability']['confidence']}`",
            f"- Summary: {payload['repeatability']['summary']}",
            f"- Score range: `{_metric_label(payload['repeatability'].get('score'))}`",
            f"- Generation range: `{_metric_label(payload['repeatability'].get('generation_tps'))}` tok/s",
            f"- Serving range: `{_metric_label(payload['repeatability'].get('serving_tps'))}` tok/s",
            f"- Cold TTFT range: `{_metric_label(payload['repeatability'].get('cold_ttft_ms'))}` ms",
            "",
            "## Candidate Assessment",
            "",
            f"- Candidate: `{payload['candidate_assessment']['model'] or 'none'}`",
            f"- Run: `{payload['candidate_assessment']['run_id'] or 'none'}`",
            (
                f"- Readiness: `{payload['candidate_assessment']['readiness']}` "
                f"(`{payload['candidate_assessment']['readiness_score']}/100`)"
            ),
            f"- Confidence: `{payload['candidate_assessment']['confidence']}`",
            "",
            "### Known Performance",
            "",
        ]
    )
    for key, value in payload["candidate_assessment"]["known_performance"].items():
        lines.append(f"- `{key}`: `{_fmt(value)}`")
    lines.extend(
        [
            "",
            "### Missing Evidence",
            "",
        ]
    )
    if payload["candidate_assessment"]["missing_evidence"]:
        for item in payload["candidate_assessment"]["missing_evidence"]:
            lines.append(
                f"- `{item['gate']}`: {item['status']} - {item['required']} "
                f"Next: {item['next_action']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidate Rankings",
            "",
            "| Rank | Model | Run | Status | Agent score | Quality | Speed | Context | Gaps |",
            "| ---: | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    if payload["candidate_rankings"]:
        for item in payload["candidate_rankings"]:
            prediction = item.get("prediction") or {}
            gaps = ", ".join(item.get("evidence_gaps") or []) or "none"
            lines.append(
                f"| {item['rank']} | `{item['model']}` | `{item['run_id']}` | "
                f"`{item['status']}` | `{_fmt(item.get('agent_quality_score'))}` | "
                f"`{prediction.get('quality', 'unmeasured')}` | "
                f"`{prediction.get('speed', 'unmeasured')}` | "
                f"`{prediction.get('context', 'unmeasured')}` | `{gaps}` |"
            )
    else:
        lines.append(
            "| - | none | none | none | `unmeasured` | `unmeasured` | `unmeasured` | `unmeasured` | `no_receipts` |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Model gate: `{payload['model_gate']['action']}`",
            f"- Deployment gate: `{payload['deployment_gate']['action']}`",
            f"- Context gate: `{payload['context_gate']['action']}`",
            f"- Required context: `{payload['context_gate']['required_context']}`",
            f"- Proven context: `{payload['context_gate'].get('proven_context') or 'none'}`",
            f"- Resource gate: `{payload['resource_gate']['action']}`",
            f"- Benchmark-suite preflight: `{payload['benchmark_suite_preflight']['status']}`",
            f"- QE gate: `{payload['qe_gate']['action']}`",
            f"- Stability gate: `{payload['stability_gate']['action']}`",
            "",
            "## Proven Recommendations",
            "",
        ]
    )
    if payload["hard_recommendations"]:
        for recommendation in payload["hard_recommendations"]:
            label = recommendation.get("label") or recommendation.get("profile_id") or "unknown"
            lines.append(f"- `{recommendation['type']}`: `{label}`")
    else:
        lines.append("No hard recommendations are proven yet.")
    lines.extend(["", "## Proven Components", ""])
    proven_components = payload.get("proven_components") or []
    if proven_components:
        for component in proven_components:
            label = component.get("label") or component.get("profile_id") or "unknown"
            lines.append(f"- `{component['type']}`: `{label}`")
    else:
        lines.append("No partial components are proven yet.")
    resource_lines = _resource_evidence_lines(payload["hard_recommendations"] or proven_components)
    if resource_lines:
        lines.extend(["", "## Resource Evidence", ""])
        lines.extend(resource_lines)
    lines.extend(["", "## Next Actions", ""])
    if payload["next_actions"]:
        lines.extend(f"- {action}" for action in payload["next_actions"])
    else:
        lines.append("- Retest challengers before replacing the current recommendations.")
    lines.extend(
        [
            "",
            "## Proof Runbook",
            "",
            "| Step | Gate | ID | Status | Proves | Success condition |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    if payload["proof_runbook"]:
        for step in payload["proof_runbook"]:
            lines.append(
                f"| {step['step']} | `{step['gate']}` | `{step['id']}` | "
                f"`{step['status']}` | `{step['proves']}` | {step['success_condition']} |"
            )
    else:
        lines.append("| - | none | none | done | none | No proof steps are pending. |")
    lines.extend(["", "## Proof Commands", ""])
    if payload["proof_commands"]:
        for command in payload["proof_commands"]:
            command_id = command.get("id", command["gate"])
            lines.extend(
                [
                    f"### {command['gate']}/{command_id}",
                    "",
                    command["purpose"],
                    "",
                    "```powershell",
                    command["command"],
                    "```",
                    "",
                    command["then"],
                    "",
                ]
            )
    else:
        lines.append("No proof commands are pending.")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "unmeasured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_label(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("min") is None or metric.get("max") is None:
        return "unmeasured"
    return (
        f"{_fmt(metric.get('min'))}-{_fmt(metric.get('max'))} "
        f"(mean {_fmt(metric.get('mean'))}, spread {_fmt(metric.get('spread_pct'))})"
    )


def _resource_evidence_lines(recommendations: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for recommendation in recommendations:
        evidence = recommendation.get("evidence")
        if not isinstance(evidence, dict):
            continue
        summary = evidence.get("resource_summary")
        if not isinstance(summary, dict) or not summary:
            continue
        label = recommendation.get("label") or recommendation.get("profile_id") or "profile"
        lines.append(f"- `{label}`: {_resource_label(summary)}")
    return lines


def _resource_label(summary: dict[str, Any]) -> str:
    gpu_used = summary.get("max_gpu_used_mb")
    gpu_total = summary.get("gpu_total_mb")
    gpu = (
        f"{gpu_used}/{gpu_total} MB VRAM"
        if gpu_used is not None and gpu_total is not None
        else "VRAM unmeasured"
    )
    parts = [gpu]
    if summary.get("max_gpu_util_percent") is not None:
        parts.append(f"GPU {summary['max_gpu_util_percent']}%")
    if summary.get("max_gpu_power_watts") is not None:
        parts.append(f"{summary['max_gpu_power_watts']} W")
    if summary.get("max_ram_used_percent") is not None:
        parts.append(f"RAM {summary['max_ram_used_percent']}%")
    if summary.get("min_ram_available_mb") is not None:
        parts.append(f"{summary['min_ram_available_mb']} MB RAM free min")
    return ", ".join(parts)
