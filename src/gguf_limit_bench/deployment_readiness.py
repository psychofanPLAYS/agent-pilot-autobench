from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gguf_limit_bench.reports import LeaderboardEntry, build_leaderboard


@dataclass(frozen=True)
class DeploymentReadinessOutputs:
    json_path: Path
    markdown_path: Path


def write_deployment_readiness(runs_root: Path) -> DeploymentReadinessOutputs:
    runs_root.mkdir(parents=True, exist_ok=True)
    payload = build_deployment_readiness(runs_root)
    json_path = runs_root / "deployment-readiness.json"
    markdown_path = runs_root / "deployment-readiness.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return DeploymentReadinessOutputs(json_path=json_path, markdown_path=markdown_path)


def build_deployment_readiness(runs_root: Path) -> dict[str, Any]:
    flag_path = runs_root / "flag-recommendations.json"
    if not flag_path.is_file():
        return _empty_payload("Run apb flag-recommendations for the target model first.")
    try:
        flag_payload = json.loads(flag_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_payload("flag-recommendations.json is missing or invalid.")
    leaderboard = build_leaderboard(runs_root)
    profiles = [
        _profile_readiness(profile, flag_payload, leaderboard.entries, runs_root)
        for profile in flag_payload.get("profiles", [])
        if isinstance(profile, dict)
    ]
    proven = [profile for profile in profiles if profile["status"] == "PROVEN"]
    recommended = _recommended_profile(proven)
    action = "PROMOTE_DEPLOYMENT_PROFILE" if recommended is not None else "RETEST_DEPLOYMENT"
    summary = (
        f"Profile `{recommended['id']}` is deployable from scored context and serving evidence."
        if recommended is not None
        else "No deployment profile has enough score, context, and serving evidence yet."
    )
    return {
        "schema_version": 1,
        "action": action,
        "summary": summary,
        "recommended_profile_id": None if recommended is None else recommended["id"],
        "recommendation_basis": _recommendation_basis(recommended),
        "model": flag_payload.get("model"),
        "model_name": flag_payload.get("model_name"),
        "lane_type": flag_payload.get("lane_type"),
        "profiles": profiles,
        "next_run": _next_run(profiles),
    }


def _profile_readiness(
    profile: dict[str, Any],
    flag_payload: dict[str, Any],
    entries: list[LeaderboardEntry],
    runs_root: Path,
) -> dict[str, Any]:
    context_size = _int(profile.get("context_size"))
    context_candidates = [
        entry
        for entry in entries
        if _same_model(entry, flag_payload) and _context_int(entry.context_label) >= context_size
    ]
    candidates = [
        entry for entry in context_candidates if _matches_recorded_profile_settings(entry, profile)
    ]
    if not candidates:
        failed_attempt = _failed_profile_attempt(profile, flag_payload, runs_root)
        if failed_attempt is not None:
            return _profile_payload_from_evidence(
                profile,
                "FAILED_PROOF",
                "Matching receipt fit and served, but failed the benchmark-suite proof.",
                failed_attempt,
            )
        reason = (
            "Matching receipts exist, but none prove the exact selected profile settings."
            if context_candidates and isinstance(profile.get("settings"), dict)
            else "No matching scored receipt proves this model at the requested context."
        )
        return _profile_payload(
            profile,
            "UNPROVEN",
            reason,
            None,
        )
    best = max(candidates, key=_candidate_key)
    resource_source = _best_resource_source(candidates)
    if best.status == "SUITE FAILED" or best.benchmark_suite_status == "fail":
        return _profile_payload(
            profile,
            "REJECTED",
            "Matching receipt failed benchmark-suite; it cannot prove deployment readiness.",
            best,
            resource_source,
        )
    has_quality = best.agent_bench_score is not None
    has_serving = best.serving_ttft_ms is not None or best.serving_tps is not None
    if has_quality and has_serving:
        return _profile_payload(
            profile,
            "PROVEN",
            "Matching receipt has agent-quality score plus serving timing evidence.",
            best,
            resource_source,
        )
    if has_quality:
        return _profile_payload(
            profile,
            "QUALITY_ONLY",
            "Matching receipt has agent-quality score but lacks serving/resource timing evidence.",
            best,
            resource_source,
        )
    return _profile_payload(
        profile,
        "SYSTEMS_ONLY",
        "Matching receipt has context/serving evidence but no agent-quality score.",
        best,
        resource_source,
    )


def _profile_payload(
    profile: dict[str, Any],
    status: str,
    reason: str,
    evidence: LeaderboardEntry | None,
    resource_source: tuple[LeaderboardEntry, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidence_payload = None
    if evidence is not None:
        resource_entry, resource_summary = (
            resource_source
            if resource_source is not None
            else (evidence, _resource_summary(Path(evidence.receipt_path)))
        )
        evidence_payload = {
            "run_id": evidence.run_id,
            "receipt_path": evidence.receipt_path,
            "context": _context_int(evidence.context_label),
            "agent_bench_score": evidence.agent_bench_score,
            "serving_ttft_ms": evidence.serving_ttft_ms,
            "serving_tps": evidence.serving_tps,
            "generation_tps": evidence.generation_tps,
            "status": evidence.status,
            "resource_summary": resource_summary,
        }
        runtime_warnings = _runtime_warnings(Path(evidence.receipt_path))
        if runtime_warnings:
            evidence_payload["runtime_warnings"] = runtime_warnings
        if resource_summary:
            evidence_payload["resource_run_id"] = resource_entry.run_id
            evidence_payload["resource_receipt_path"] = resource_entry.receipt_path
        if status == "PROVEN" and _runtime_warnings_block_promotion(runtime_warnings):
            status = "RUNTIME_WARNING"
            reason = (
                "Matching receipt has score and serving evidence, but runtime emitted "
                "critical warnings that must be fixed before deployment promotion."
            )
    return _profile_payload_from_evidence(profile, status, reason, evidence_payload)


def _profile_payload_from_evidence(
    profile: dict[str, Any],
    status: str,
    reason: str,
    evidence_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "id": str(profile.get("id") or "unknown"),
        "label": str(profile.get("label") or profile.get("id") or "unknown"),
        "context_size": _int(profile.get("context_size")),
        "status": status,
        "reason": reason,
        "evidence": evidence_payload,
    }
    payload["recommendation_score"] = _profile_recommendation_score(payload)
    return payload


def _failed_profile_attempt(
    profile: dict[str, Any], flag_payload: dict[str, Any], runs_root: Path
) -> dict[str, Any] | None:
    attempts: list[dict[str, Any]] = []
    context_size = _int(profile.get("context_size"))
    for best_path in runs_root.glob("*/best-settings.json"):
        try:
            payload = json.loads(best_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        settings = payload.get("settings")
        result = payload.get("result")
        if not isinstance(settings, dict) or not isinstance(result, dict):
            continue
        if not _same_model_payload(str(payload.get("model") or ""), flag_payload):
            continue
        if _int(settings.get("context_size")) < context_size:
            continue
        if not _matches_settings_dict(settings, profile):
            continue
        if result.get("benchmark_suite_ok") is not False and not _has_real_failure(result):
            continue
        attempts.append(_failed_attempt_payload(best_path.parent, settings, result))
    if not attempts:
        return None
    return max(
        attempts,
        key=lambda item: (
            _number(item.get("serving_tps")) or _number(item.get("generation_tps")) or 0,
            _number(item.get("context")) or 0,
        ),
    )


def _failed_attempt_payload(
    receipt_path: Path, settings: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    resource_summary = _resource_summary(receipt_path)
    payload: dict[str, Any] = {
        "run_id": receipt_path.name,
        "receipt_path": str(receipt_path),
        "context": _int(settings.get("context_size")),
        "agent_bench_score": _number(result.get("agent_bench_score")),
        "serving_ttft_ms": _number(result.get("serving_ttft_ms")),
        "serving_tps": _number(result.get("serving_tokens_per_second")),
        "generation_tps": _number(result.get("generation_tokens_per_second")),
        "status": "SUITE FAILED" if result.get("benchmark_suite_ok") is False else "FAILED",
        "failure": str(result.get("failure") or "unknown"),
        "benchmark_suite_failure": str(result.get("benchmark_suite_failure") or ""),
        "resource_summary": resource_summary,
    }
    runtime_warnings = _runtime_warnings(receipt_path)
    if runtime_warnings:
        payload["runtime_warnings"] = runtime_warnings
    if resource_summary:
        payload["resource_run_id"] = receipt_path.name
        payload["resource_receipt_path"] = str(receipt_path)
    return payload


def _has_real_failure(result: dict[str, Any]) -> bool:
    failure = str(result.get("failure") or "").strip().lower()
    benchmark_failure = str(result.get("benchmark_suite_failure") or "").strip().lower()
    return failure not in {"", "none", "null"} or benchmark_failure not in {"", "none", "null"}


def _recommended_profile(proven: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not proven:
        return None
    priority = {"long_agent": 0, "over_the_top": 1, "standard": 2, "bare_minimum": 3}
    return sorted(
        proven,
        key=lambda profile: (
            float(profile.get("recommendation_score") or 0.0),
            _int(profile.get("context_size")),
            -priority.get(str(profile["id"]), 99),
        ),
        reverse=True,
    )[0]


def _profile_recommendation_score(profile: dict[str, Any]) -> float:
    evidence = profile.get("evidence") or {}
    quality = _number(evidence.get("agent_bench_score")) or 0.0
    context = _number(evidence.get("context")) or _number(profile.get("context_size")) or 0.0
    serving_tps = (
        _number(evidence.get("serving_tps")) or _number(evidence.get("generation_tps")) or 0.0
    )
    ttft_ms = _number(evidence.get("serving_ttft_ms")) or 0.0
    quality_points = float(quality) * 100.0
    context_points = min(float(context), 262_144.0) / 4096.0
    speed_points = min(float(serving_tps), 60.0) / 3.0
    ttft_penalty = min(float(ttft_ms), 5000.0) / 1000.0
    return round(quality_points + context_points + speed_points - ttft_penalty, 4)


def _recommendation_basis(recommended: dict[str, Any] | None) -> dict[str, Any] | None:
    if recommended is None:
        return None
    evidence = recommended.get("evidence") or {}
    return {
        "profile_id": recommended["id"],
        "run_id": evidence.get("run_id"),
        "receipt_path": evidence.get("receipt_path"),
        "agent_bench_score": evidence.get("agent_bench_score"),
        "context": evidence.get("context"),
        "serving_tps": evidence.get("serving_tps"),
        "serving_ttft_ms": evidence.get("serving_ttft_ms"),
        "recommendation_score": recommended.get("recommendation_score"),
    }


def _next_run(profiles: list[dict[str, Any]]) -> str:
    if not profiles:
        return "Run apb flag-recommendations for the target model."
    priority = {"standard": 0, "long_agent": 1, "over_the_top": 2, "bare_minimum": 3}
    for profile in sorted(profiles, key=lambda item: priority.get(str(item["id"]), 99)):
        if profile["status"] != "PROVEN":
            if profile["status"] == "FAILED_PROOF":
                raw_evidence = profile.get("evidence")
                evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
                failure = (
                    evidence.get("benchmark_suite_failure")
                    or evidence.get("failure")
                    or "failed proof"
                )
                return f"Fix `{profile['id']}` proof failure ({failure}) and rerun the profile."
            return (
                f"Run a scored librarian/benchmark-suite receipt at {profile['context_size']} "
                f"context for profile `{profile['id']}` with serving telemetry enabled."
            )
    return (
        "Retest challengers with the same score/context/serving gate before replacing this profile."
    )


def _same_model(entry: LeaderboardEntry, flag_payload: dict[str, Any]) -> bool:
    return _same_model_payload(entry.model_path, flag_payload, model_name=entry.model_name)


def _same_model_payload(
    model_path: str, flag_payload: dict[str, Any], *, model_name: str | None = None
) -> bool:
    model = str(flag_payload.get("model") or "")
    expected_name = str(flag_payload.get("model_name") or Path(model).name)
    actual_name = model_name or Path(model_path).name
    return model_path == model or actual_name == expected_name


def _matches_recorded_profile_settings(entry: LeaderboardEntry, profile: dict[str, Any]) -> bool:
    return _matches_settings_dict(entry.settings, profile)


def _matches_settings_dict(settings: dict[str, Any], profile: dict[str, Any]) -> bool:
    profile_settings = profile.get("settings")
    if not isinstance(profile_settings, dict):
        recorded_profile_name = str(settings.get("profile_name") or "")
        if recorded_profile_name:
            return recorded_profile_name == str(profile.get("id") or "")
        return True
    expected_profile_name = str(profile_settings.get("profile_name") or profile.get("id") or "")
    if expected_profile_name and str(settings.get("profile_name") or "") != expected_profile_name:
        return False
    for key, expected in profile_settings.items():
        if _normalized_setting(settings.get(key)) != _normalized_setting(expected):
            return False
    return True


def _normalized_setting(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalized_setting(item) for item in value]
    if isinstance(value, list):
        return [_normalized_setting(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalized_setting(item) for key, item in sorted(value.items())}
    return value


def _candidate_key(entry: LeaderboardEntry) -> tuple[float, int, float]:
    return (
        entry.agent_bench_score if entry.agent_bench_score is not None else -1.0,
        _context_int(entry.context_label),
        entry.serving_tps or entry.generation_tps,
    )


def _best_resource_source(
    candidates: list[LeaderboardEntry],
) -> tuple[LeaderboardEntry, dict[str, Any]] | None:
    measured = [
        (entry, _resource_summary(Path(entry.receipt_path)))
        for entry in candidates
        if Path(entry.receipt_path).is_dir()
    ]
    measured = [(entry, summary) for entry, summary in measured if summary]
    if not measured:
        return None
    return max(
        measured,
        key=lambda item: (
            _number(item[1].get("max_gpu_used_mb")) or 0,
            _number(item[1].get("max_gpu_util_percent")) or 0,
            _number(item[1].get("max_ram_used_percent")) or 0,
        ),
    )


def _resource_summary(receipt_path: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    events_path = receipt_path / "events.jsonl"
    if not events_path.is_file():
        return {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        telemetry = dict(dict(event.get("data") or {}).get("telemetry") or {})
        if telemetry:
            samples.append(telemetry)
    if not samples:
        return {}
    summary: dict[str, Any] = {}
    _set_max(summary, "max_gpu_used_mb", samples, "gpu_used_mb")
    _set_first_present(summary, "gpu_total_mb", samples, "gpu_total_mb")
    _set_max(summary, "max_gpu_util_percent", samples, "gpu_util_percent")
    _set_max(summary, "max_gpu_power_watts", samples, "gpu_power_watts")
    _set_max(summary, "max_ram_used_percent", samples, "ram_used_percent")
    _set_min(summary, "min_ram_available_mb", samples, "ram_available_mb")
    return summary


def _runtime_warnings(receipt_path: Path) -> dict[str, Any]:
    warning_paths = [
        receipt_path / "warnings.log",
        *sorted(receipt_path.glob("simplebench-*/warnings.log")),
    ]
    logs: list[dict[str, Any]] = []
    total_count = 0
    for warning_path in warning_paths:
        if not warning_path.is_file():
            continue
        lines = [
            line
            for line in warning_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line and "No warning or error lines detected." not in line
        ]
        summary_count = _warning_count_from_summary(warning_path.parent / "summary.json")
        warning_count = max(len(lines), summary_count or 0)
        if warning_count <= 0:
            continue
        total_count += warning_count
        logs.append(
            {
                "path": str(warning_path),
                "warning_count": warning_count,
                "sample": lines[:3],
            }
        )
    if not total_count:
        return {}
    critical = _critical_runtime_warning_samples(logs)
    payload: dict[str, Any] = {"warning_count": total_count, "logs": logs}
    if critical:
        payload["critical"] = critical
    return payload


def _runtime_warnings_block_promotion(warnings: dict[str, Any]) -> bool:
    return bool(warnings.get("critical")) if warnings else False


def _critical_runtime_warning_samples(logs: list[dict[str, Any]]) -> list[str]:
    critical: list[str] = []
    patterns = (
        "detected an outdated gemma4 chat template",
        "cuda error",
        "cuda error:",
        "cuda failure",
        "illegal memory access",
        "out of memory",
        "unknown argument",
    )
    for log in logs:
        for line in log.get("sample") or []:
            normalized = str(line).lower()
            if any(pattern in normalized for pattern in patterns):
                critical.append(str(line))
    return critical[:5]


def _warning_count_from_summary(summary_path: Path) -> int | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("warning_count")
    return int(value) if isinstance(value, int) and value >= 0 else None


def _set_max(
    summary: dict[str, Any], target_key: str, samples: list[dict[str, Any]], source_key: str
) -> None:
    values = [_number(sample.get(source_key)) for sample in samples]
    present = [value for value in values if value is not None]
    if present:
        summary[target_key] = max(present)


def _set_min(
    summary: dict[str, Any], target_key: str, samples: list[dict[str, Any]], source_key: str
) -> None:
    values = [_number(sample.get(source_key)) for sample in samples]
    present = [value for value in values if value is not None]
    if present:
        summary[target_key] = min(present)


def _set_first_present(
    summary: dict[str, Any], target_key: str, samples: list[dict[str, Any]], source_key: str
) -> None:
    for sample in samples:
        value = _number(sample.get(source_key))
        if value is not None:
            summary[target_key] = value
            return


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _context_int(label: str) -> int:
    try:
        return int(label)
    except (TypeError, ValueError):
        return 0


def _int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if not isinstance(value, int | float | str | bytes | bytearray):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _empty_payload(summary: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "action": "NO_FLAG_RECOMMENDATIONS",
        "summary": summary,
        "recommended_profile_id": None,
        "recommendation_basis": None,
        "model": None,
        "model_name": None,
        "lane_type": None,
        "profiles": [],
        "next_run": "Run apb flag-recommendations, then rerun deployment-readiness.",
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Deployment Readiness",
        "",
        f"- Action: `{payload['action']}`",
        f"- Recommended profile: `{payload['recommended_profile_id'] or 'none'}`",
        f"- Summary: {payload['summary']}",
        f"- Next run: {payload['next_run']}",
        "",
        "## Recommendation Basis",
        "",
        _recommendation_basis_markdown(payload.get("recommendation_basis")),
        "",
        "| Profile | Context | Status | Evidence | Resources | Warnings | Reason |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for profile in payload.get("profiles", []):
        evidence = profile.get("evidence") or {}
        evidence_label = (
            f"{evidence.get('run_id')} @ {evidence.get('context')}" if evidence else "none"
        )
        resource_label = _resource_label(dict(evidence.get("resource_summary") or {}))
        warning_label = _warning_label(dict(evidence.get("runtime_warnings") or {}))
        lines.append(
            f"| `{profile['id']}` | {profile['context_size']} | `{profile['status']}` | "
            f"`{evidence_label}` | `{resource_label}` | `{warning_label}` | {profile['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _recommendation_basis_markdown(basis: Any) -> str:
    if not isinstance(basis, dict) or not basis:
        return "No profile is recommended yet."
    return (
        f"Profile `{basis.get('profile_id')}` is recommended from receipt "
        f"`{basis.get('run_id')}` with score `{basis.get('recommendation_score')}`, "
        f"agent score `{basis.get('agent_bench_score')}`, context `{basis.get('context')}`, "
        f"and serving `{basis.get('serving_tps')}` tok/s."
    )


def _resource_label(summary: dict[str, Any]) -> str:
    if not summary:
        return "unmeasured"
    gpu_used = summary.get("max_gpu_used_mb")
    gpu_total = summary.get("gpu_total_mb")
    gpu = (
        f"{gpu_used}/{gpu_total} MB"
        if gpu_used is not None and gpu_total is not None
        else "gpu n/a"
    )
    util = summary.get("max_gpu_util_percent")
    power = summary.get("max_gpu_power_watts")
    ram = summary.get("max_ram_used_percent")
    parts = [gpu]
    if util is not None:
        parts.append(f"GPU {util}%")
    if power is not None:
        parts.append(f"{power} W")
    if ram is not None:
        parts.append(f"RAM {ram}%")
    return ", ".join(parts)


def _warning_label(warnings: dict[str, Any]) -> str:
    count = _int(warnings.get("warning_count"))
    if count <= 0:
        return "none"
    return f"{count} runtime warning(s)"
