from __future__ import annotations

from dataclasses import asdict, dataclass, field
from html import escape
import json
from pathlib import Path

from gguf_limit_bench import charts
from gguf_limit_bench.agent_quality import (
    MIN_LIBRARIAN_RECOMMENDATION_ATTEMPTS,
    MIN_LIBRARIAN_RECOMMENDATION_PACKS,
    is_recommendation_grade_librarian_sample,
    librarian_agent_quality_gate,
)
from gguf_limit_bench.autoresearch import parse_llama_bench_jsonl
from gguf_limit_bench.discovery import is_non_generative_gguf
from gguf_limit_bench.evidence import display_status, evidence_status, normalize_success_failure
from gguf_limit_bench.metrics import agent_index as _agent_index


@dataclass(frozen=True)
class LeaderboardEntry:
    run_id: str
    model_name: str
    model_path: str
    score: float
    status: str
    context_label: str
    generation_tps: float
    prompt_tps: float
    serving_ttft_ms: float | None
    serving_warm_ttft_ms: float | None
    serving_warmup_penalty_ms: float | None
    serving_server_ready_ms: float | None
    serving_cold_start_to_first_token_ms: float | None
    serving_tps: float | None
    serving_failure: str | None
    agent_bench_score: float | None
    benchmark_suite_general_score: float | None
    benchmark_suite_agentic_score: float | None
    benchmark_suite_status: str
    benchmark_suite_receipt: str | None
    benchmark_suite_failure: str | None
    failure: str
    settings: dict
    receipt_path: str
    librarian_score: float | None = None
    pack_scores: dict[str, float] = field(default_factory=dict)
    scored_pack_count: int = 0
    scored_attempt_count: int = 0
    agent_quality_gate: str = "missing"


@dataclass(frozen=True)
class Leaderboard:
    entries: list[LeaderboardEntry]

    @property
    def champion(self) -> LeaderboardEntry:
        return self.entries[0]


@dataclass(frozen=True)
class ModelComparisonEntry:
    model_name: str
    model_path: str
    run_count: int
    best_run_id: str
    best_score: float
    best_status: str
    best_context_label: str
    generation_tps: float
    prompt_tps: float
    cold_ttft_ms: float | None
    warm_ttft_ms: float | None
    serving_tps: float | None
    agent_bench_score: float | None
    benchmark_suite_status: str
    best_receipt_path: str
    itemized_report_path: str
    browser_report_path: str
    recommendation: str
    librarian_score: float | None = None
    pack_scores: dict[str, float] = field(default_factory=dict)
    scored_pack_count: int = 0


@dataclass(frozen=True)
class ModelComparison:
    entries: list[ModelComparisonEntry]


@dataclass(frozen=True)
class Verdict:
    action: str
    confidence: str
    champion_model: str | None
    champion_run_id: str | None
    summary: str
    next_run: str
    agent_quality_score: float | None
    general_score: float | None
    agentic_score: float | None
    expected_generation_tps: float | None
    expected_prompt_tps: float | None
    expected_serving_tps: float | None
    expected_cold_ttft_ms: float | None
    expected_warm_ttft_ms: float | None
    context_label: str | None
    evidence_status: str | None
    receipt_path: str | None
    prediction: dict[str, str]


@dataclass(frozen=True)
class ReportAudit:
    status: str
    warning_count: int
    warnings: list[dict[str, str]]


@dataclass(frozen=True)
class AgentQuality:
    raw_score: float | None
    eligible_score: float | None
    pack_scores: dict[str, float]
    scored_pack_count: int
    scored_attempt_count: int
    completion_rate: float
    gate: str


def build_leaderboard(runs_root: Path) -> Leaderboard:
    entries: list[LeaderboardEntry] = []
    for best_path in sorted(runs_root.glob("*/best-settings.json")):
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        if (
            payload.get("promotion_eligible") is False
            or str(payload.get("status", "")).lower() == "partial"
        ):
            continue
        result = _normalized_result(payload)
        settings = payload.get("settings", {})
        model_path = str(payload.get("model", ""))
        # Never let a non-LLM (embedding/reranker/query-expansion/etc.) become a
        # champion via a stale historical receipt — they should never be benchmarked.
        if model_path and is_non_generative_gguf(Path(model_path)):
            continue
        ok = bool(result.get("ok", False))
        failure = normalize_success_failure(ok, str(result.get("failure", "unknown")))
        context = int(settings.get("context_size") or 0)
        score = _normalized_score(payload, result)
        agent_quality = _load_agent_quality(best_path.parent)
        agent_bench_score = _float_or_none(result.get("agent_bench_score"))
        if agent_bench_score is None:
            agent_bench_score = agent_quality.eligible_score
        if failure == "model_load":
            status = "LOAD FAIL"
        elif result.get("benchmark_suite_ok") is True:
            status = "BENCHMARK SUITE"
        elif result.get("benchmark_suite_ok") is False:
            status = "SUITE FAILED"
        else:
            status = display_status(
                evidence_status(
                    ok=ok,
                    failure=failure,
                    generation_tps=float(result.get("generation_tokens_per_second") or 0.0),
                    context_size=context,
                    workflow_score=float(result.get("workflow_score") or 0.0),
                    workflow_results=result.get("workflow_results") or [],
                    serving_ttft_ms=(
                        float(result["serving_ttft_ms"])
                        if result.get("serving_ttft_ms") is not None
                        else None
                    ),
                ).value
            )
        entries.append(
            LeaderboardEntry(
                run_id=best_path.parent.name,
                model_name=Path(model_path).name,
                model_path=model_path,
                score=score,
                status=status,
                context_label="unset (speed-only)" if context == 0 else str(context),
                generation_tps=float(result.get("generation_tokens_per_second") or 0.0),
                prompt_tps=float(result.get("prompt_tokens_per_second") or 0.0),
                serving_ttft_ms=(
                    float(result["serving_ttft_ms"])
                    if result.get("serving_ttft_ms") is not None
                    else None
                ),
                serving_warm_ttft_ms=(
                    float(result["serving_warm_ttft_ms"])
                    if result.get("serving_warm_ttft_ms") is not None
                    else None
                ),
                serving_warmup_penalty_ms=(
                    float(result["serving_warmup_penalty_ms"])
                    if result.get("serving_warmup_penalty_ms") is not None
                    else None
                ),
                serving_server_ready_ms=(
                    float(result["serving_server_ready_ms"])
                    if result.get("serving_server_ready_ms") is not None
                    else None
                ),
                serving_cold_start_to_first_token_ms=(
                    float(result["serving_cold_start_to_first_token_ms"])
                    if result.get("serving_cold_start_to_first_token_ms") is not None
                    else None
                ),
                serving_tps=(
                    float(result["serving_tokens_per_second"])
                    if result.get("serving_tokens_per_second") is not None
                    else None
                ),
                serving_failure=result.get("serving_failure"),
                agent_bench_score=agent_bench_score,
                benchmark_suite_general_score=_float_or_none(
                    result.get("benchmark_suite_general_score")
                ),
                benchmark_suite_agentic_score=_float_or_none(
                    result.get("benchmark_suite_agentic_score")
                ),
                benchmark_suite_status=_benchmark_suite_status(result),
                benchmark_suite_receipt=result.get("benchmark_suite_receipt"),
                benchmark_suite_failure=result.get("benchmark_suite_failure"),
                failure=failure,
                settings=settings,
                receipt_path=str(best_path.parent),
                librarian_score=agent_quality.raw_score,
                pack_scores=agent_quality.pack_scores,
                scored_pack_count=agent_quality.scored_pack_count,
                scored_attempt_count=agent_quality.scored_attempt_count,
                agent_quality_gate=agent_quality.gate,
            )
        )
    return Leaderboard(entries=sorted(entries, key=_leaderboard_rank_key, reverse=True))


def build_report_audit(leaderboard: Leaderboard) -> ReportAudit:
    warnings: list[dict[str, str]] = []
    for entry in leaderboard.entries:
        if entry.status in {"LOAD FAIL", "SUITE FAILED"}:
            continue
        if entry.agent_bench_score is not None:
            continue
        if entry.librarian_score is not None:
            warnings.append(
                {
                    "code": "weak_agent_quality",
                    "run_id": entry.run_id,
                    "model": entry.model_name,
                    "status": entry.status,
                    "receipt_path": entry.receipt_path,
                    "message": (
                        "Librarian evidence exists but is too small for a hard recommendation. "
                        f"Need at least {MIN_LIBRARIAN_RECOMMENDATION_PACKS} scored packs and "
                        f"{MIN_LIBRARIAN_RECOMMENDATION_ATTEMPTS} scored attempts; got "
                        f"{entry.scored_pack_count} pack(s) and {entry.scored_attempt_count} attempt(s)."
                    ),
                }
            )
            continue
        warnings.append(
            {
                "code": "missing_agent_quality",
                "run_id": entry.run_id,
                "model": entry.model_name,
                "status": entry.status,
                "receipt_path": entry.receipt_path,
                "message": (
                    "Speed/context evidence is not enough for a hard recommendation. "
                    "Run a score-backed benchmark before promoting this receipt."
                ),
            }
        )
    return ReportAudit(
        status="pass" if not warnings else "warning",
        warning_count=len(warnings),
        warnings=warnings,
    )


def build_verdict(leaderboard: Leaderboard) -> Verdict:
    if not leaderboard.entries:
        return Verdict(
            action="NO_EVIDENCE",
            confidence="none",
            champion_model=None,
            champion_run_id=None,
            summary=(
                "No usable benchmark receipts exist yet. pilotBENCHY cannot recommend a model "
                "until at least one scored run is recorded."
            ),
            next_run="Run a benchmark-suite or librarian-bench flight plan for the target model.",
            agent_quality_score=None,
            general_score=None,
            agentic_score=None,
            expected_generation_tps=None,
            expected_prompt_tps=None,
            expected_serving_tps=None,
            expected_cold_ttft_ms=None,
            expected_warm_ttft_ms=None,
            context_label=None,
            evidence_status=None,
            receipt_path=None,
            prediction={
                "quality": "unmeasured",
                "speed": "unmeasured",
                "context": "unmeasured",
                "recommendation": "run_agent_benchmark",
            },
        )

    champion = leaderboard.champion
    if champion.benchmark_suite_status == "pass" and champion.agent_bench_score is not None:
        action = "PROMOTE"
        confidence = "high"
        summary = (
            f"{champion.model_name} is the current recommendation because it has a measured "
            f"agent-quality score of {_format_score(champion.agent_bench_score)} plus a passing "
            "benchmark-suite run."
        )
        next_run = (
            "Use this as the current champion, then rerun the same suite against any challenger "
            "model or settings profile before replacing it."
        )
    elif champion.librarian_score is not None and champion.agent_bench_score is not None:
        action = "PROMOTE"
        confidence = "medium"
        summary = (
            f"{champion.model_name} is the current recommendation from librarian/agent-quality "
            f"evidence. It scored {_format_score(champion.agent_bench_score)} across "
            f"{champion.scored_pack_count} scored pack(s), but it has not run the full "
            "general+agentic benchmark-suite gate yet."
        )
        next_run = (
            "Use this as a current working recommendation, then run the benchmark-suite gate before "
            "treating it as final production evidence."
        )
    elif champion.benchmark_suite_status == "fail":
        action = "REJECT"
        confidence = "high"
        summary = (
            f"{champion.model_name} loaded, but the required benchmark-suite failed. "
            "Do not deploy this settings profile from speed evidence."
        )
        next_run = (
            "Inspect the benchmark-suite failure receipt, fix the failed task or template issue, "
            "then rerun the same plan."
        )
    else:
        action = "RETEST"
        confidence = "low"
        summary = (
            f"{champion.model_name} is only the best systems result so far. Fit, context, or "
            "raw speed is not an intelligence result, so this is not enough evidence to "
            "recommend the model for agent work."
        )
        next_run = (
            "Run a benchmark-suite plan or librarian-bench mode so pilotBENCHY can produce "
            "agent_bench_score, general score, and agentic score."
        )

    return Verdict(
        action=action,
        confidence=confidence,
        champion_model=champion.model_name,
        champion_run_id=champion.run_id,
        summary=summary,
        next_run=next_run,
        agent_quality_score=champion.agent_bench_score,
        general_score=champion.benchmark_suite_general_score,
        agentic_score=champion.benchmark_suite_agentic_score,
        expected_generation_tps=champion.generation_tps,
        expected_prompt_tps=champion.prompt_tps,
        expected_serving_tps=champion.serving_tps,
        expected_cold_ttft_ms=champion.serving_ttft_ms,
        expected_warm_ttft_ms=champion.serving_warm_ttft_ms,
        context_label=champion.context_label,
        evidence_status=champion.status,
        receipt_path=champion.receipt_path,
        prediction=_prediction_for_entry(champion, action),
    )


def score_summary_for_entry(entry: LeaderboardEntry) -> dict[str, object]:
    return {
        "model": entry.model_name,
        "status": entry.status,
        "score_contract": _score_contract(entry),
        "raw_score": entry.score,
        "agent_bench_score": entry.agent_bench_score,
        "general_score": entry.benchmark_suite_general_score,
        "agentic_score": entry.benchmark_suite_agentic_score,
        "generation_tps": entry.generation_tps,
        "prompt_tps": entry.prompt_tps,
        "serving_tps": entry.serving_tps,
        "cold_ttft_ms": entry.serving_ttft_ms,
        "warm_ttft_ms": entry.serving_warm_ttft_ms,
        "context": _context_int(entry.context_label),
        "context_label": entry.context_label,
    }


def _score_contract(entry: LeaderboardEntry) -> str:
    if entry.agent_bench_score is not None:
        return "agent_bench_score"
    if entry.librarian_score is not None:
        return "librarian_bench_score_unpromoted"
    return "speed_or_fit_only"


def _context_int(context_label: str) -> int | None:
    try:
        return int(context_label)
    except ValueError:
        return None


def _prediction_for_entry(entry: LeaderboardEntry, action: str) -> dict[str, str]:
    return {
        "quality": _quality_prediction(entry.agent_bench_score),
        "speed": _speed_prediction(entry.serving_tps or entry.generation_tps),
        "context": _context_prediction(entry.context_label),
        "recommendation": _recommendation_prediction(action, entry.agent_bench_score),
    }


def _quality_prediction(score: float | None) -> str:
    if score is None:
        return "unmeasured"
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "usable"
    return "weak"


def _speed_prediction(tps: float | None) -> str:
    if tps is None or tps <= 0.0:
        return "unmeasured"
    if tps >= 25.0:
        return "interactive"
    if tps >= 10.0:
        return "slow_interactive"
    return "batch_only"


def _context_prediction(context_label: str) -> str:
    try:
        context = int(context_label)
    except ValueError:
        return "unmeasured"
    if context >= 131_072:
        return "long_agentic"
    if context >= 65_536:
        return "agentic"
    if context >= 32_768:
        return "basic_agentic"
    return "short"


def _recommendation_prediction(action: str, score: float | None) -> str:
    if action == "REJECT":
        return "do_not_use"
    if score is None:
        return "needs_agent_benchmark"
    if action == "PROMOTE":
        return "use_current_champion"
    return "needs_agent_benchmark"


def _load_agent_quality(receipt_dir: Path) -> AgentQuality:
    """Aggregate librarian/agent per-pack accuracy from a receipt directory.

    Reads ``results.json`` (champion_eval output) if present, otherwise falls back
    to ``librarian-suite-summary.json``. The headline score uses the same contract
    as ``librarian_suite``: scored-attempt accuracy multiplied by completion rate.
    Per-pack scores remain pack accuracies for the matrix view.
    """
    payload: dict | None = None
    for name in ("results.json", "librarian-suite-summary.json"):
        candidate = receipt_dir / name
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None
            continue
        if isinstance(payload, dict):
            break
        payload = None
    if not isinstance(payload, dict):
        return AgentQuality(None, None, {}, 0, 0, 0.0, "missing")

    pack_scores: dict[str, float] = {}
    asked_total = 0
    correct_total = 0
    incomplete_total = 0
    for pack in payload.get("packs", []) or []:
        if not isinstance(pack, dict):
            continue
        if str(pack.get("status", "")) != "scored":
            continue
        pack_id = pack.get("pack_id")
        accuracy = pack.get("accuracy")
        if pack_id is None or accuracy is None:
            continue
        try:
            pack_scores[str(pack_id)] = float(accuracy)
            asked = int(pack.get("asked") or 0)
            asked_total += asked
            correct_total += int(pack.get("correct") or 0)
            incomplete_total += int(pack.get("incomplete") or 0)
        except (TypeError, ValueError):
            continue

    if not pack_scores:
        return AgentQuality(None, None, {}, 0, 0, 0.0, "missing")
    completion_rate = (asked_total - incomplete_total) / asked_total if asked_total else 0.0
    top_level_score = _float_or_none(payload.get("librarian_bench_score"))
    if top_level_score is not None:
        librarian_score = top_level_score
    elif asked_total > 0:
        accuracy = correct_total / asked_total
        librarian_score = accuracy * completion_rate
    else:
        librarian_score = sum(pack_scores.values()) / len(pack_scores)
    eligible = is_recommendation_grade_librarian_sample(
        scored_pack_count=len(pack_scores),
        scored_attempt_count=asked_total,
    )
    gate = librarian_agent_quality_gate(
        scored_pack_count=len(pack_scores),
        scored_attempt_count=asked_total,
    )
    return AgentQuality(
        raw_score=librarian_score,
        eligible_score=librarian_score if eligible else None,
        pack_scores=pack_scores,
        scored_pack_count=len(pack_scores),
        scored_attempt_count=asked_total,
        completion_rate=completion_rate,
        gate=gate,
    )


def build_model_comparison(leaderboard: Leaderboard) -> ModelComparison:
    groups: dict[tuple[str, str], list[LeaderboardEntry]] = {}
    for entry in leaderboard.entries:
        groups.setdefault((entry.model_name, entry.model_path), []).append(entry)

    comparison_entries: list[ModelComparisonEntry] = []
    for (model_name, model_path), runs in groups.items():
        ranked_runs = sorted(runs, key=_model_run_rank_key, reverse=True)
        best = ranked_runs[0]
        receipt = Path(best.receipt_path)
        comparison_entries.append(
            ModelComparisonEntry(
                model_name=model_name,
                model_path=model_path,
                run_count=len(runs),
                best_run_id=best.run_id,
                best_score=best.score,
                best_status=best.status,
                best_context_label=best.context_label,
                generation_tps=best.generation_tps,
                prompt_tps=best.prompt_tps,
                cold_ttft_ms=best.serving_ttft_ms,
                warm_ttft_ms=best.serving_warm_ttft_ms,
                serving_tps=best.serving_tps,
                agent_bench_score=best.agent_bench_score,
                benchmark_suite_status=best.benchmark_suite_status,
                best_receipt_path=best.receipt_path,
                itemized_report_path=str(receipt / "itemized-report.md"),
                browser_report_path=str(receipt / "report.html"),
                recommendation=_model_recommendation(best, len(runs)),
                librarian_score=best.librarian_score,
                pack_scores=dict(best.pack_scores),
                scored_pack_count=best.scored_pack_count,
            )
        )
    return ModelComparison(
        entries=sorted(comparison_entries, key=_model_comparison_rank_key, reverse=True)
    )


def _model_run_rank_key(entry: LeaderboardEntry) -> tuple[float, int, float, int]:
    agent_quality = entry.agent_bench_score if entry.agent_bench_score is not None else -1.0
    return (
        agent_quality,
        _status_rank(entry.status),
        entry.score,
        1,
    )


def _model_comparison_rank_key(entry: ModelComparisonEntry) -> tuple[float, int, float, int]:
    # Rank by agent-quality first (so the comparison leads with quality, not speed),
    # then fall back to evidence tier + speed score for models without recommendation-grade
    # librarian evidence. Raw tiny-sample librarian scores remain visible but do not rank.
    agent_quality = entry.agent_bench_score if entry.agent_bench_score is not None else -1.0
    return (
        agent_quality,
        _status_rank(entry.best_status),
        entry.best_score,
        entry.run_count,
    )


def _status_rank(status: str) -> int:
    return {
        "BENCHMARK SUITE": 700,
        "WORKFLOW SMOKE": 600,
        "WORKFLOW WEAK": 500,
        "WORKFLOW UNPROVEN": 400,
        "CONTEXT UNPROVEN": 300,
        "SERVING MEASURED": 250,
        "SPEED ONLY": 200,
        "SLOW": 100,
        "SUITE FAILED": 50,
        "LOAD FAIL": 0,
    }.get(status, 0)


def _leaderboard_rank_key(entry: LeaderboardEntry) -> tuple[int, float]:
    evidence_rank = {
        "BENCHMARK SUITE": 700,
        "WORKFLOW SMOKE": 600,
        "WORKFLOW WEAK": 500,
        "WORKFLOW UNPROVEN": 400,
        "CONTEXT UNPROVEN": 300,
        "SERVING MEASURED": 250,
        "SPEED ONLY": 200,
        "SLOW": 100,
        "SUITE FAILED": 50,
        "LOAD FAIL": 0,
    }.get(entry.status, 0)
    return evidence_rank, entry.score


def _normalized_result(payload: dict) -> dict:
    result = dict(payload.get("result", {}))
    stdout = str(result.get("stdout") or "")
    if not stdout.strip():
        return result

    parsed = parse_llama_bench_jsonl(
        stdout=stdout,
        returncode=int(result.get("returncode") or 0),
        stderr=str(result.get("stderr") or ""),
        fallback_context=int(result.get("context_size") or 0),
    )
    if parsed.generation_tokens_per_second <= 0.0 and parsed.prompt_tokens_per_second <= 0.0:
        return result

    result.update(
        {
            "ok": parsed.ok,
            "generation_tokens_per_second": parsed.generation_tokens_per_second,
            "prompt_tokens_per_second": parsed.prompt_tokens_per_second,
            "context_size": parsed.context_size,
            "failure": parsed.failure,
        }
    )
    return result


def _normalized_score(payload: dict, result: dict) -> float:
    if not result.get("ok", False):
        return -10_000.0
    if result.get("agent_bench_score") is not None:
        return float(result["agent_bench_score"])
    context_size = int(
        result.get("context_size") or payload.get("settings", {}).get("context_size") or 0
    )
    context_bonus = min(context_size, 131_072) / 4096.0
    measured_ttft = result.get("serving_ttft_ms")
    if measured_ttft is None:
        measured_ttft = result.get("ttft_ms")
    ttft_penalty = float(measured_ttft if measured_ttft is not None else 10_000.0) / 1000.0
    serving_speed_bonus = float(result.get("serving_tokens_per_second") or 0.0) / 10.0
    return (
        float(result.get("generation_tokens_per_second") or 0.0)
        + float(result.get("prompt_tokens_per_second") or 0.0) / 100.0
        + context_bonus
        + float(result.get("workflow_score") or 0.0)
        + serving_speed_bonus
        - ttft_penalty
    )


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def _benchmark_suite_status(result: dict) -> str:
    if result.get("benchmark_suite_ok") is True:
        return "pass"
    if result.get("benchmark_suite_ok") is False:
        return "fail"
    return "not_run"


def write_leaderboard(runs_root: Path) -> Leaderboard:
    runs_root.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard(runs_root)
    if not leaderboard.entries:
        (runs_root / "leaderboard.md").write_text(
            "# pilotBENCHY Leaderboard\n\nNo runs found.\n",
            encoding="utf-8",
        )
        _write_report_audit(runs_root, build_report_audit(leaderboard))
        _write_verdict(runs_root, build_verdict(leaderboard))
        _write_empty_model_comparison(runs_root)
        (runs_root / "results.html").write_text(_empty_html(), encoding="utf-8")
        return leaderboard
    model_comparison = build_model_comparison(leaderboard)
    _write_report_audit(runs_root, build_report_audit(leaderboard))
    _write_verdict(runs_root, build_verdict(leaderboard))
    (runs_root / "leaderboard.md").write_text(_leaderboard_markdown(leaderboard), encoding="utf-8")
    (runs_root / "model-comparison.md").write_text(
        _model_comparison_markdown(model_comparison), encoding="utf-8"
    )
    (runs_root / "model-comparison.json").write_text(
        json.dumps(
            [asdict(entry) for entry in model_comparison.entries], ensure_ascii=True, indent=2
        ),
        encoding="utf-8",
    )
    (runs_root / "results.html").write_text(_leaderboard_html(leaderboard), encoding="utf-8")
    (runs_root / "champion.json").write_text(
        json.dumps(asdict(leaderboard.champion), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return leaderboard


def _write_report_audit(runs_root: Path, audit: ReportAudit) -> None:
    (runs_root / "report-audit.json").write_text(
        json.dumps(asdict(audit), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (runs_root / "report-audit.md").write_text(_report_audit_markdown(audit), encoding="utf-8")


def _report_audit_markdown(audit: ReportAudit) -> str:
    lines = [
        "# pilotBENCHY Report Audit",
        "",
        f"- Status: `{audit.status}`",
        f"- Warnings: `{audit.warning_count}`",
        "",
    ]
    if not audit.warnings:
        lines.extend(["No report audit warnings.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "| Code | Run | Model | Status | Message |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for warning in audit.warnings:
        lines.append(
            "| "
            f"`{warning['code']}` | `{warning['run_id']}` | `{warning['model']}` | "
            f"`{warning['status']}` | {warning['message']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_verdict(runs_root: Path, verdict: Verdict) -> None:
    (runs_root / "verdict.json").write_text(
        json.dumps(asdict(verdict), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (runs_root / "verdict.md").write_text(_verdict_markdown(verdict), encoding="utf-8")


def _verdict_markdown(verdict: Verdict) -> str:
    lines = [
        "# pilotBENCHY Verdict",
        "",
        f"- Action: `{verdict.action}`",
        f"- Confidence: `{verdict.confidence}`",
        f"- Model: `{verdict.champion_model or 'none'}`",
        f"- Evidence: `{verdict.evidence_status or 'none'}`",
        f"- Agent quality: `{_format_score(verdict.agent_quality_score)}`",
        f"- General score: `{_format_score(verdict.general_score)}`",
        f"- Agentic score: `{_format_score(verdict.agentic_score)}`",
        f"- Expected generation: `{_format_tps(verdict.expected_generation_tps)}`",
        f"- Expected serving: `{_format_tps(verdict.expected_serving_tps)}`",
        f"- Cold TTFT: `{_format_ms(verdict.expected_cold_ttft_ms)}`",
        f"- Warm TTFT: `{_format_ms(verdict.expected_warm_ttft_ms)}`",
        f"- Context: `{verdict.context_label or 'unmeasured'}`",
        f"- Predicted quality: `{verdict.prediction['quality']}`",
        f"- Predicted speed: `{verdict.prediction['speed']}`",
        f"- Predicted context: `{verdict.prediction['context']}`",
        f"- Recommendation class: `{verdict.prediction['recommendation']}`",
        "",
        "## Why",
        "",
        verdict.summary,
        "",
        "## Next Run",
        "",
        verdict.next_run,
        "",
    ]
    if verdict.receipt_path:
        lines.extend(["## Receipt", "", f"`{verdict.receipt_path}`", ""])
    return "\n".join(lines)


def _leaderboard_markdown(leaderboard: Leaderboard) -> str:
    champion = leaderboard.champion
    verdict = build_verdict(leaderboard)
    audit = build_report_audit(leaderboard)
    score_summary = score_summary_for_entry(champion)
    result_heading = "Recommended Model" if verdict.action == "PROMOTE" else "Top Candidate"
    lines = [
        "# pilotBENCHY Leaderboard",
        "",
        "## Benchmark Scores",
        "",
        f"- Score contract: `{score_summary['score_contract']}`",
        f"- Agent bench score: `{_format_score(champion.agent_bench_score)}`",
        f"- General score: `{_format_score(champion.benchmark_suite_general_score)}`",
        f"- Agentic score: `{_format_score(champion.benchmark_suite_agentic_score)}`",
        f"- Raw score: `{champion.score:.4f}`",
        f"- Generation speed: `{champion.generation_tps:.2f}` tok/s",
        f"- Serving speed: `{_format_tps(champion.serving_tps)}`",
        f"- Context: `{champion.context_label}`",
        "",
        "## Verdict",
        "",
        f"- Action: `{verdict.action}`",
        f"- Confidence: `{verdict.confidence}`",
        f"- Summary: {verdict.summary}",
        f"- Next run: {verdict.next_run}",
        "",
        "## Report Audit",
        "",
        f"- Status: `{audit.status}`",
        f"- Warnings: `{audit.warning_count}`",
        "",
        "## Plain-English Takeaway",
        "",
        f"- Best measured model: `{champion.model_name}`",
        f"- Result: {_plain_english_status(champion)}",
        f"- Proof folder: `{champion.receipt_path}`",
        "",
        f"## {result_heading}",
        "",
        f"- Model: `{champion.model_name}`",
        f"- Score: `{champion.score:.2f}`",
        f"- Status: `{champion.status}`",
        f"- Context: `{champion.context_label}`",
        f"- Generation: `{champion.generation_tps:.2f}` tok/s",
        f"- Prompt: `{champion.prompt_tps:.2f}` tok/s",
        f"- Cold TTFT: `{_format_ms(champion.serving_ttft_ms)}`",
        f"- Warm TTFT: `{_format_ms(champion.serving_warm_ttft_ms)}`",
        f"- Warmup penalty: `{_format_ms(champion.serving_warmup_penalty_ms)}`",
        f"- Server ready: `{_format_ms(champion.serving_server_ready_ms)}`",
        f"- Server start to first token: `{_format_ms(champion.serving_cold_start_to_first_token_ms)}`",
        f"- Serving generation: `{_format_tps(champion.serving_tps)}`",
        f"- Agent bench score: `{_format_score(champion.agent_bench_score)}`",
        f"- Benchmark suite: `{champion.benchmark_suite_status}`",
        f"- Benchmark suite general: `{_format_score(champion.benchmark_suite_general_score)}`",
        f"- Benchmark suite agentic: `{_format_score(champion.benchmark_suite_agentic_score)}`",
        f"- Evidence: `{champion.status}`",
        "",
        "## Runs",
        "",
        "| Rank | Status | Score | Agent bench | Suite | Bench gen tok/s | Prompt tok/s | Cold TTFT | Warm TTFT | Warmup | Serve tok/s | Context | Model |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, entry in enumerate(leaderboard.entries, start=1):
        lines.append(
            f"| {rank} | {entry.status} | {entry.score:.2f} | "
            f"{_format_score(entry.agent_bench_score)} | {entry.benchmark_suite_status} | "
            f"{entry.generation_tps:.2f} | {entry.prompt_tps:.2f} | "
            f"{_format_ms(entry.serving_ttft_ms)} | {_format_ms(entry.serving_warm_ttft_ms)} | "
            f"{_format_ms(entry.serving_warmup_penalty_ms)} | {_format_tps(entry.serving_tps)} | "
            f"{entry.context_label} | `{entry.model_name}` |"
        )
    if audit.warnings:
        lines.extend(["", "## Audit Warnings", ""])
        for warning in audit.warnings:
            lines.append(f"- `{warning['code']}` in `{warning['run_id']}`: {warning['message']}")
    lines.append("")
    return "\n".join(lines)


def _write_empty_model_comparison(runs_root: Path) -> None:
    (runs_root / "model-comparison.md").write_text(
        "# pilotBENCHY Model Comparison\n\nNo model runs found yet.\n",
        encoding="utf-8",
    )
    (runs_root / "model-comparison.json").write_text("[]\n", encoding="utf-8")


def _ordered_pack_ids(comparison: ModelComparison) -> list[str]:
    """Stable, ordered list of every librarian pack id seen across the comparison.

    Known librarian packs are listed first in a canonical order; any extra pack ids
    that appear in the data are appended in first-seen order.
    """
    preferred = [
        "librarian-gate",
        "librarian-dedupe",
        "librarian-compress",
        "librarian-query",
        "librarian-rerank",
        "librarian-contradiction",
        "librarian-triage",
        "librarian-write-entry",
    ]
    seen: list[str] = []
    for entry in comparison.entries:
        for pack_id in entry.pack_scores:
            if pack_id not in seen:
                seen.append(pack_id)
    ordered = [pid for pid in preferred if pid in seen]
    ordered.extend(pid for pid in seen if pid not in preferred)
    return ordered


def _short_pack_label(pack_id: str) -> str:
    return pack_id[len("librarian-") :] if pack_id.startswith("librarian-") else pack_id


def _model_comparison_markdown(comparison: ModelComparison) -> str:
    pack_ids = _ordered_pack_ids(comparison)
    lines = [
        "# pilotBENCHY Model Comparison",
        "",
        "This is the model-level view. It groups repeated runs by model so pilotBENCHY can "
        "compare best-known settings per model instead of treating every receipt folder as a "
        "separate universe. Models are ranked by recommendation-grade agent_bench_score "
        "first, then by speed evidence. Per-pack columns remain raw inspection metrics.",
        "",
    ]
    pack_header = "".join(f" {_short_pack_label(pid)} |" for pid in pack_ids)
    pack_sep = "".join(" ---: |" for _ in pack_ids)
    lines.append(
        "| Rank | Model | Eligible agent score |" + pack_header + " Runs | Best status | "
        "Gen TPS | Cold TTFT | Serving TPS | Suite | Best receipt |"
    )
    lines.append("|---:|---|---:|" + pack_sep + "---:|---|---:|---:|---:|---|---|")
    for rank, entry in enumerate(comparison.entries, start=1):
        pack_cells = "".join(f" {_format_pct(entry.pack_scores.get(pid))} |" for pid in pack_ids)
        lines.append(
            f"| {rank} | `{entry.model_name}` | {_format_score(entry.agent_bench_score)} |"
            + pack_cells
            + f" {entry.run_count} | {entry.best_status} | "
            f"{entry.generation_tps:.2f} | {_format_ms(entry.cold_ttft_ms)} | "
            f"{_format_tps(entry.serving_tps)} | {entry.benchmark_suite_status} | "
            f"`{entry.best_receipt_path}` |"
        )
    lines.extend(["", "## Recommendations", ""])
    for entry in comparison.entries:
        lines.append(f"- `{entry.model_name}`: {entry.recommendation}")
    return "\n".join(lines) + "\n"


def _empty_html() -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            "  <title>pilotBENCHY Results</title>",
            "  <style>",
            _html_css(),
            "  </style>",
            "</head>",
            "<body>",
            '  <main class="shell">',
            '    <section class="hero">',
            '      <p class="eyebrow">No runs yet</p>',
            "      <h1>pilotBENCHY Results</h1>",
            "      <p>Run a benchmark first, then refresh this report.</p>",
            "    </section>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )


def _best_by_agent_quality(comparison: ModelComparison) -> ModelComparisonEntry | None:
    scored = [e for e in comparison.entries if e.agent_bench_score is not None]
    if not scored:
        return None
    return max(scored, key=lambda e: e.agent_bench_score or 0.0)


def _agent_verdict(comparison: ModelComparison) -> str:
    best = _best_by_agent_quality(comparison)
    if best is None:
        return (
            "No agent-quality scores yet. Run a librarian benchmark to measure how well each "
            "model actually performs the agent tasks, not just how fast it generates tokens."
        )
    runner_up = next(
        (
            e
            for e in comparison.entries
            if e.agent_bench_score is not None and e.model_name != best.model_name
        ),
        None,
    )
    verdict = (
        f"{best.model_name} leads on agent quality with "
        f"{_format_pct(best.agent_bench_score)} accuracy across "
        f"{best.scored_pack_count} librarian pack(s)."
    )
    if runner_up is not None:
        verdict += (
            f" Next best is {runner_up.model_name} at {_format_pct(runner_up.agent_bench_score)}."
        )
    return verdict


def _agent_index_for(
    pack_scores: dict, suite_general: float | None, suite_agentic: float | None
) -> float | None:
    """Standardized Agent Index from a model's pack accuracies + suite scores.

    Returns None when no category-level quality signal was measured (so the model
    is not plotted as a misleading zero).
    """
    signals: dict[str, float] = dict(pack_scores)
    if suite_general is not None:
        signals["suite_general"] = float(suite_general)
    if suite_agentic is not None:
        signals["suite_agentic"] = float(suite_agentic)
    if not signals:
        return None
    idx = _agent_index(signals)
    if all(value is None for value in idx.category_subscores.values()):
        return None
    return idx.value


def _peak_vram_gb(receipt_path: str) -> float | None:
    metrics_path = Path(receipt_path) / "metrics.json"
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    efficiency = data.get("efficiency") or {}
    value = efficiency.get("peak_vram_gb")
    return float(value) if value else None


def _dashboard_models(leaderboard: Leaderboard, comparison: ModelComparison) -> list[dict]:
    by_run = {entry.run_id: entry for entry in leaderboard.entries}
    models: list[dict] = []
    for entry in comparison.entries:
        best = by_run.get(entry.best_run_id)
        suite_general = best.benchmark_suite_general_score if best else None
        suite_agentic = best.benchmark_suite_agentic_score if best else None
        models.append(
            {
                "name": entry.model_name,
                "agent_index": _agent_index_for(entry.pack_scores, suite_general, suite_agentic),
                "gen_tps": entry.generation_tps,
                "prompt_tps": entry.prompt_tps,
                "serving_tps": entry.serving_tps,
                "cold_ttft": entry.cold_ttft_ms,
                "vram_gb": _peak_vram_gb(entry.best_receipt_path),
                "pack_scores": dict(entry.pack_scores),
                "family": "",
            }
        )
    return models


def _index_history(leaderboard: Leaderboard) -> list[dict]:
    """Best-so-far Agent Index across runs over time (a progress trend).

    Runs are ordered by their timestamped ``run_id``; the series is the running
    maximum Agent Index, so the line shows how benchmarking more models improves
    the best known agent-quality result.
    """
    points: list[dict] = []
    best = 0.0
    seen = False
    for entry in sorted(leaderboard.entries, key=lambda e: e.run_id):
        idx = _agent_index_for(
            entry.pack_scores,
            entry.benchmark_suite_general_score,
            entry.benchmark_suite_agentic_score,
        )
        if idx is None:
            continue
        seen = True
        best = max(best, idx)
        points.append({"label": entry.run_id, "index": best})
    return points if seen else []


def _chart_card(title: str, description: str, chart_html: str, *, wide: bool = False) -> str:
    cls = "chart-card wide" if wide else "chart-card"
    return (
        f'<div class="{cls}"><h3>{escape(title)}</h3>'
        f'<p class="receipt">{escape(description)}</p>{chart_html}</div>'
    )


def _kpi_strip(models: list[dict], leaderboard: Leaderboard) -> str:
    scored = [m for m in models if m.get("agent_index") is not None]
    champ = max(scored, key=lambda m: m["agent_index"]) if scored else None
    fastest = max(models, key=lambda m: m.get("gen_tps") or 0.0) if models else None
    lead = champ or fastest
    index_label = f"{champ['agent_index']:.0f}" if champ else "—"
    speed_label = f"{lead['gen_tps']:.0f} tok/s" if lead and lead.get("gen_tps") else "—"
    eff_rows = [m for m in models if m.get("vram_gb") and m.get("agent_index") is not None]
    if eff_rows:
        best_eff = max(eff_rows, key=lambda m: m["agent_index"] / m["vram_gb"])
        eff_label = f"{best_eff['agent_index'] / best_eff['vram_gb']:.1f}/GB"
    else:
        eff_label = "—"
    cards = [
        ("Champion", lead["name"] if lead else "—"),
        ("Agent Index", index_label),
        ("Gen speed", speed_label),
        ("Best efficiency", eff_label),
        ("Models · runs", f"{len(models)} · {len(leaderboard.entries)}"),
    ]
    items = "".join(
        f'<div class="kpi"><div class="k">{escape(label)}</div>'
        f'<div class="v">{escape(str(value))}</div></div>'
        for label, value in cards
    )
    return f'<section class="kpis">{items}</section>'


def _charts_section(models: list[dict], pack_ids: list[str], history: list[dict]) -> str:
    blocks: list[str] = []
    frontier = charts.quality_vs_speed_config(models)
    if frontier["data"]["datasets"][0]["data"]:
        blocks.append(
            _chart_card(
                "Quality vs speed",
                "The frontier — up and to the right is better. Bubble size = VRAM.",
                charts.render_chart("c-frontier", frontier, height=360),
                wide=True,
            )
        )
    index_bar = charts.agent_index_bar_config(models)
    if index_bar["data"]["labels"]:
        height = max(180, 40 * len(index_bar["data"]["labels"]) + 40)
        blocks.append(
            _chart_card(
                "Agent Index ranking",
                "Composite agent-quality score (0–100), red→green.",
                charts.render_chart("c-index", index_bar, height=height),
            )
        )
    radar = charts.pack_radar_config(models, pack_ids)
    if radar["data"]["datasets"]:
        blocks.append(
            _chart_card(
                "Capability profile",
                "Per-pack accuracy shape for each model.",
                charts.render_chart("c-radar", radar, height=360),
            )
        )
    if any(m.get("gen_tps") for m in models):
        blocks.append(
            _chart_card(
                "Speed breakdown",
                "Generation, prompt, and serving throughput per model.",
                charts.render_chart("c-speed", charts.speed_bars_config(models), height=320),
            )
        )
    efficiency = charts.efficiency_bars_config(models)
    if efficiency is not None:
        height = max(180, 40 * len(efficiency["data"]["labels"]) + 40)
        blocks.append(
            _chart_card(
                "Efficiency frontier",
                "Agent Index per GB of VRAM — the local-hardware advantage.",
                charts.render_chart("c-eff", efficiency, height=height),
            )
        )
    trend = charts.index_trend_config(history)
    if trend is not None:
        blocks.append(
            _chart_card(
                "Agent Index over time",
                "Best agent-quality result so far, across runs as they accumulate.",
                charts.render_chart("c-trend", trend, height=300),
                wide=True,
            )
        )
    if not blocks:
        return ""
    return (
        '<section class="panel"><h2>Visual overview</h2>'
        '<div class="chart-grid">' + "".join(blocks) + "</div></section>"
    )


def _leaderboard_html(leaderboard: Leaderboard) -> str:
    champion = leaderboard.champion
    model_comparison = build_model_comparison(leaderboard)
    verdict_report = build_verdict(leaderboard)
    audit = build_report_audit(leaderboard)
    score_summary = score_summary_for_entry(champion)
    pack_ids = _ordered_pack_ids(model_comparison)
    dashboard_models = _dashboard_models(leaderboard, model_comparison)
    kpi_strip = _kpi_strip(dashboard_models, leaderboard)
    charts_section = _charts_section(
        dashboard_models, pack_ids, _index_history(leaderboard)
    )
    chart_runtime = charts.chartjs_runtime() if charts_section else ""
    rows = "\n".join(
        _html_row(rank, entry) for rank, entry in enumerate(leaderboard.entries, start=1)
    )
    model_rows = "\n".join(
        _model_html_row(rank, entry, pack_ids)
        for rank, entry in enumerate(model_comparison.entries, start=1)
    )
    pack_headers = "".join(
        f'<th class="pack">{escape(_short_pack_label(pid))}</th>' for pid in pack_ids
    )
    quality_best = _best_by_agent_quality(model_comparison)
    if quality_best is not None:
        hero_eyebrow = "Best model by agent quality"
        hero_lede = (
            f"{quality_best.model_name} scores {_format_pct(quality_best.agent_bench_score)} "
            "on agent-quality tasks."
        )
    else:
        hero_eyebrow = "Highest-ranked unscored candidate"
        hero_lede = f"{champion.model_name} is the highest-ranked unscored candidate so far."
    verdict = _agent_verdict(model_comparison)
    settings_heading = (
        "Recommended settings" if verdict_report.action == "PROMOTE" else "Candidate settings"
    )
    settings = "\n".join(
        f"<li><span>{escape(str(key))}</span><strong>{escape(str(value))}</strong></li>"
        for key, value in sorted(champion.settings.items())
    )
    audit_warnings = "\n".join(
        "<li>"
        f"<code>{escape(warning['code'])}</code> in "
        f"<code>{escape(warning['run_id'])}</code>: "
        f"{escape(warning['message'])}"
        "</li>"
        for warning in audit.warnings
    )
    audit_body = (
        f"<ul>{audit_warnings}</ul>" if audit_warnings else "<p>No report audit warnings.</p>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>pilotBENCHY Results</title>
  <style>
{_html_css()}
  </style>
</head>
<body>
  {chart_runtime}
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">{escape(hero_eyebrow)}</p>
      <h1>pilotBENCHY Results</h1>
      <p class="lede">{escape(hero_lede)}</p>
      <p class="verdict">{escape(verdict)}</p>
      <div class="verdict-box">
        <h2>Benchmark scores</h2>
        <p><strong>Score contract: {escape(str(score_summary["score_contract"]))}</strong></p>
        <ul class="score-list">
          <li><span>Agent bench score</span><strong>{escape(_format_score(champion.agent_bench_score))}</strong></li>
          <li><span>General score</span><strong>{escape(_format_score(champion.benchmark_suite_general_score))}</strong></li>
          <li><span>Agentic score</span><strong>{escape(_format_score(champion.benchmark_suite_agentic_score))}</strong></li>
          <li><span>Raw score</span><strong>{champion.score:.4f}</strong></li>
          <li><span>Generation</span><strong>{champion.generation_tps:.2f} tok/s</strong></li>
          <li><span>Serving</span><strong>{escape(_format_tps(champion.serving_tps))}</strong></li>
          <li><span>Context</span><strong>{escape(champion.context_label)}</strong></li>
        </ul>
      </div>
      <div class="verdict-box">
        <h2>Verdict: {escape(verdict_report.action)}</h2>
        <p><strong>{escape(verdict_report.confidence)} confidence</strong></p>
        <p>{escape(verdict_report.summary)}</p>
        <p class="receipt">Next run: {escape(verdict_report.next_run)}</p>
      </div>
      <div class="verdict-box">
        <h2>Report Audit</h2>
        <p><strong>{escape(audit.status)}</strong> ({audit.warning_count} warning(s))</p>
        {audit_body}
      </div>
    </section>
    {kpi_strip}
    {charts_section}
    <section class="panel">
      <h2>Model comparison</h2>
      <p class="receipt">
        Ranked by recommendation-grade agent_bench_score, with raw per-pack scores and
        secondary speed metrics. Pack cells use a red-to-green scale by accuracy.
      </p>
      <div class="table-wrap">
        <table class="matrix">
          <thead>
            <tr>
              <th>Rank</th><th>Model</th><th>Eligible agent score</th>
              {pack_headers}
              <th>Gen tok/s</th><th>Cold TTFT</th>
            </tr>
          </thead>
          <tbody>
            {model_rows}
          </tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Plain-English takeaway</h2>
      <p>{escape(_plain_english_status(champion))}</p>
      <p class="receipt">Proof folder: <code>{escape(champion.receipt_path)}</code></p>
      <p class="receipt">Evidence: <code>{escape(champion.status)}</code></p>
    </section>
    <section class="panel">
      <h2>What to do next</h2>
      <ol>
        <li>Open <code>_runs\\leaderboard.md</code> when you want the compact Markdown version.</li>
        <li>Open <code>_runs\\model-comparison.md</code> when you want the model comparison view.</li>
        <li>
          Run <code>agent-autobench deployment-readiness</code> before
          <code>agent-autobench export-profile</code>; export is only a deployment
          recommendation after the readiness gate promotes a profile.
        </li>
        <li>Use the receipt path below when an AI agent needs to inspect the proof.</li>
      </ol>
      <p class="receipt">Receipt: <code>{escape(champion.receipt_path)}</code></p>
    </section>
    <section class="panel">
      <h2>{escape(settings_heading)}</h2>
      <ul class="settings">
        {settings}
      </ul>
    </section>
    <section class="panel">
      <h2>All runs</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th><th>Status</th><th>Score</th><th>Generation</th>
              <th>Agent Bench</th><th>Suite</th><th>Prompt</th><th>Cold TTFT</th><th>Warm TTFT</th><th>Warmup</th><th>Serving</th><th>Context</th><th>Model</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def _html_row(rank: int, entry: LeaderboardEntry) -> str:
    status_class = "pass" if entry.status in {"WORKFLOW SMOKE", "BENCHMARK SUITE"} else "fail"
    return (
        f'<tr class="{status_class}">'
        f"<td>{rank}</td>"
        f"<td>{escape(entry.status)}</td>"
        f"<td>{entry.score:.2f}</td>"
        f"<td>{entry.generation_tps:.2f}</td>"
        f"<td>{escape(_format_score(entry.agent_bench_score))}</td>"
        f"<td>{escape(entry.benchmark_suite_status)}</td>"
        f"<td>{entry.prompt_tps:.2f}</td>"
        f"<td>{escape(_format_ms(entry.serving_ttft_ms))}</td>"
        f"<td>{escape(_format_ms(entry.serving_warm_ttft_ms))}</td>"
        f"<td>{escape(_format_ms(entry.serving_warmup_penalty_ms))}</td>"
        f"<td>{escape(_format_tps(entry.serving_tps))}</td>"
        f"<td>{escape(entry.context_label)}</td>"
        f"<td><code>{escape(entry.model_name)}</code></td>"
        "</tr>"
    )


def _accuracy_color(value: float) -> str:
    """Red (0.0) -> amber (0.5) -> green (1.0) background for a 0..1 accuracy."""
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        # red -> amber
        ratio = value / 0.5
        r, g, b = 220, int(70 + ratio * (170 - 70)), 70
    else:
        # amber -> green
        ratio = (value - 0.5) / 0.5
        r, g, b = (
            int(220 - ratio * (220 - 60)),
            int(170 + ratio * (200 - 170)),
            int(70 + ratio * 20),
        )
    return f"rgba({r}, {g}, {b}, 0.28)"


def _pack_cell(value: float | None) -> str:
    if value is None:
        return '<td class="pack empty">—</td>'
    color = _accuracy_color(value)
    return f'<td class="pack" style="background:{color}">{_format_pct(value)}</td>'


def _model_html_row(rank: int, entry: ModelComparisonEntry, pack_ids: list[str]) -> str:
    status_class = "pass" if entry.best_status in {"WORKFLOW SMOKE", "BENCHMARK SUITE"} else "fail"
    pack_cells = "".join(_pack_cell(entry.pack_scores.get(pid)) for pid in pack_ids)
    agent_cell = (
        f'<td class="agent" style="background:{_accuracy_color(entry.agent_bench_score)}">'
        f"{_format_pct(entry.agent_bench_score)}</td>"
        if entry.agent_bench_score is not None
        else '<td class="agent empty">—</td>'
    )
    return (
        f'<tr class="{status_class}">'
        f"<td>{rank}</td>"
        f"<td><code>{escape(entry.model_name)}</code></td>"
        f"{agent_cell}"
        f"{pack_cells}"
        f"<td>{entry.generation_tps:.2f}</td>"
        f"<td>{escape(_format_ms(entry.cold_ttft_ms))}</td>"
        "</tr>"
    )


def _model_recommendation(best: LeaderboardEntry, run_count: int) -> str:
    if best.status == "BENCHMARK SUITE":
        return "Suite-backed candidate. Use this as the current per-model recommendation."
    if best.agent_bench_score is not None and best.librarian_score is not None:
        return (
            "Recommendation-grade librarian evidence. Use as a working recommendation, then confirm "
            "with the benchmark-suite gate before final promotion."
        )
    if best.status == "WORKFLOW SMOKE":
        return "Promising local candidate. Run the benchmark-suite phase before final promotion."
    if best.status in {"WORKFLOW UNPROVEN", "SERVING MEASURED", "SPEED ONLY"}:
        return (
            f"{run_count} run(s) recorded. Keep iterating with context and workflow checks before "
            "treating this model as proven."
        )
    if best.status == "LOAD FAIL":
        return (
            "Best run still failed to load. Fix model path, VRAM fit, or llama.cpp settings first."
        )
    return "Keep the receipt, but collect stronger evidence before ranking this model highly."


def _plain_english_status(entry: LeaderboardEntry) -> str:
    if entry.status == "BENCHMARK SUITE":
        return (
            f"{entry.model_name} is the current suite-backed candidate. "
            f"Its comparable agent_bench_score is {_format_score(entry.agent_bench_score)} "
            "from the required general and agentic benchmark-suite phase."
        )
    if entry.status == "SUITE FAILED":
        return (
            f"{entry.model_name} loaded, but its required benchmark suite failed. "
            "Keep the receipt for debugging, but do not treat it as production-ready."
        )
    if entry.status == "WORKFLOW SMOKE":
        return (
            f"{entry.model_name} is the current best workflow-smoke candidate. "
            f"It generated {entry.generation_tps:.2f} tokens/sec and passed local smoke checks, "
            "but still needs the full benchmark-suite phase before production use."
        )
    if entry.status == "LOAD FAIL":
        return (
            f"{entry.model_name} is not ready yet because it failed to load. "
            "Check the model file, llama.cpp build, and VRAM fit before using it."
        )
    if entry.status == "SPEED ONLY":
        if entry.serving_ttft_ms is None:
            return (
                f"{entry.model_name} is llama-bench speed evidence only. It generated "
                f"{entry.generation_tps:.2f} tokens/sec, but real serving TTFT was not measured."
            )
        return (
            f"{entry.model_name} is speed evidence only. It generated "
            f"{entry.generation_tps:.2f} tokens/sec, but context and workflow readiness "
            "were not proven."
        )
    if entry.status == "WORKFLOW UNPROVEN":
        return (
            f"{entry.model_name} has speed/context evidence, but no workflow tasks proved "
            "agent usefulness yet."
        )
    if entry.status == "WORKFLOW WEAK":
        return (
            f"{entry.model_name} passed only smoke-level workflow checks. Treat it as a "
            "candidate for retest, not deployment."
        )
    return (
        f"{entry.model_name} produced a weak or incomplete result. "
        "Keep the receipt, but do not treat it as the production recommendation yet."
    )


def _format_ms(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.0f} ms"


def _format_tps(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.2f} tok/s"


def _format_score(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.4f}"


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _html_css() -> str:
    return """
    :root {
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #151b23;
      --panel-strong: #1f2937;
      --text: #f5f7fb;
      --muted: #a7b0c0;
      --line: #303846;
      --green: #3ddc84;
      --red: #ff6b6b;
      --gold: #f2c94c;
      --blue: #65b7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 16px/1.5 "Segoe UI", system-ui, sans-serif;
    }
    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }
    .hero, .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 24px;
      margin-bottom: 16px;
    }
    .hero {
      background: linear-gradient(135deg, #151b23 0%, #192235 55%, #20271e 100%);
    }
    .eyebrow {
      margin: 0 0 8px;
      color: var(--gold);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
      font-size: 0.78rem;
    }
    h1, h2 { margin: 0 0 12px; line-height: 1.15; }
    h1 { font-size: 2.4rem; }
    h2 { font-size: 1.25rem; }
    .lede, .receipt, li { color: var(--muted); }
    .lede { font-size: 1.15rem; color: var(--text); margin: 0 0 6px; }
    .verdict {
      margin: 12px 0 0;
      padding: 12px 14px;
      border-left: 3px solid var(--green);
      background: rgba(61, 220, 132, 0.08);
      border-radius: 0 8px 8px 0;
      color: var(--text);
    }
    code {
      color: var(--text);
      background: #0b0f14;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 6px;
    }
    .score-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .score-grid div, .settings li, .score-list li {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    span { display: block; color: var(--muted); font-size: 0.86rem; }
    strong { display: block; margin-top: 4px; color: var(--text); font-size: 1.15rem; }
    .settings {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .score-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      list-style: none;
      padding: 0;
      margin: 12px 0 0;
    }
    .table-wrap { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow-wrap: anywhere;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 0.84rem; }
    tr.pass td:first-child { color: var(--green); }
    tr.fail td:first-child { color: var(--red); }
    table.matrix th, table.matrix td { white-space: nowrap; }
    table.matrix tbody tr:hover { background: rgba(101, 183, 255, 0.06); }
    th.pack, td.pack, td.agent {
      text-align: center;
      font-variant-numeric: tabular-nums;
    }
    td.agent { font-weight: 700; color: var(--text); }
    td.pack { color: var(--text); }
    td.pack.empty, td.agent.empty { color: var(--muted); background: transparent; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .kpi {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }
    .kpi .k {
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .kpi .v { color: var(--text); font-size: 1.5rem; font-weight: 800; margin-top: 4px; }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 12px;
    }
    .chart-card {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .chart-card.wide { grid-column: 1 / -1; }
    .chart-card h3 { margin: 0 0 4px; font-size: 1.0rem; }
    .chart-card .receipt { margin: 0 0 12px; font-size: 0.84rem; }
    .chart-box { width: 100%; }
    @media (max-width: 720px) {
      h1 { font-size: 1.8rem; }
      .hero, .panel { padding: 18px; }
      table { font-size: 0.88rem; }
      .chart-grid { grid-template-columns: 1fr; }
    }
    """
