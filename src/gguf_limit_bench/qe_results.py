from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


MIN_QE_RECOMMENDATION_ATTEMPTS = 30
MIN_QE_PROMOTE_FORMAT_RATE = 0.90
MIN_QE_PROMOTE_SCORE = 0.90


@dataclass(frozen=True)
class QeResultEntry:
    run_id: str
    receipt_path: str
    model: str
    action: str
    recommendation: str
    next_run: str
    score: float
    format_rate: float
    direct_answer_rate: float
    attempts: int
    answer_max_tokens: int | None
    sampling: dict[str, Any]
    median_tps: float | None
    median_ttft_ms: float | None
    resource_summary: dict[str, Any]


@dataclass(frozen=True)
class QeLeaderboard:
    entries: list[QeResultEntry]

    @property
    def champion(self) -> QeResultEntry | None:
        return self.entries[0] if self.entries else None


def build_qe_leaderboard(runs_root: Path) -> QeLeaderboard:
    """Build a recommendation board from QE fresh-session receipts."""
    entries: list[QeResultEntry] = []
    for summary_path in sorted(runs_root.glob("*/qe-format-summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append(_entry_from_summary(summary_path.parent, payload))
    return QeLeaderboard(entries=sorted(entries, key=_rank_key, reverse=True))


def write_qe_leaderboard(runs_root: Path) -> QeLeaderboard:
    runs_root.mkdir(parents=True, exist_ok=True)
    leaderboard = build_qe_leaderboard(runs_root)
    (runs_root / "qe-format-leaderboard.json").write_text(
        json.dumps(_leaderboard_payload(leaderboard), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (runs_root / "qe-format-leaderboard.md").write_text(
        _leaderboard_markdown(leaderboard),
        encoding="utf-8",
    )
    return leaderboard


def _entry_from_summary(run_dir: Path, payload: dict[str, Any]) -> QeResultEntry:
    score = _float(payload.get("score"))
    format_rate = _float(payload.get("format_rate"))
    direct_answer_rate = _float(payload.get("direct_answer_rate"))
    attempts = int(payload.get("attempts") or 0)
    action, recommendation, next_run = _recommendation(
        score=score,
        format_rate=format_rate,
        direct_answer_rate=direct_answer_rate,
        attempts=attempts,
    )
    return QeResultEntry(
        run_id=run_dir.name,
        receipt_path=str(run_dir),
        model=str(payload.get("model") or "unknown"),
        action=action,
        recommendation=recommendation,
        next_run=next_run,
        score=score,
        format_rate=format_rate,
        direct_answer_rate=direct_answer_rate,
        attempts=attempts,
        answer_max_tokens=_int_or_none(payload.get("answer_max_tokens")),
        sampling=dict(payload.get("sampling") or {}),
        median_tps=_float_or_none(payload.get("median_tps")),
        median_ttft_ms=_float_or_none(payload.get("median_ttft_ms")),
        resource_summary=_resource_summary(dict(payload.get("resources") or {})),
    )


def _recommendation(
    *, score: float, format_rate: float, direct_answer_rate: float, attempts: int
) -> tuple[str, str, str]:
    if direct_answer_rate > 0.0:
        return (
            "REJECT_QE_PROFILE",
            "QE profile answered user questions. Do not deploy it as query expansion.",
            "Tighten the system prompt/template or try another QE model, then rerun qe-format.",
        )
    if attempts < MIN_QE_RECOMMENDATION_ATTEMPTS:
        return (
            "RETEST_QE_PROFILE",
            (
                "QE sample is too small for a hard recommendation. "
                f"Need at least {MIN_QE_RECOMMENDATION_ATTEMPTS} fresh attempts."
            ),
            "Rerun qe-format with --repeats high enough to reach at least 30 attempts.",
        )
    if format_rate >= MIN_QE_PROMOTE_FORMAT_RATE and score >= MIN_QE_PROMOTE_SCORE:
        return (
            "PROMOTE_QE_PROFILE",
            "QE profile produced clean retrieval payloads without direct answers.",
            "Use this QE profile as current champion, then retest after prompt or llama.cpp changes.",
        )
    return (
        "RETEST_QE_PROFILE",
        (
            "QE profile avoided direct answers, but format rate is below the production gate "
            f"of {MIN_QE_PROMOTE_FORMAT_RATE:.0%}."
        ),
        "Improve prompt/template/canonicalizer and rerun qe-format before deployment.",
    )


def _rank_key(entry: QeResultEntry) -> tuple[int, float, float, float, int, float]:
    promoted = 1 if entry.action == "PROMOTE_QE_PROFILE" else 0
    no_direct_answer = 1 if entry.direct_answer_rate == 0.0 else 0
    return (
        promoted,
        no_direct_answer,
        entry.format_rate,
        entry.score,
        entry.attempts,
        entry.median_tps or 0.0,
    )


def _leaderboard_payload(leaderboard: QeLeaderboard) -> dict[str, Any]:
    return {
        "score_contract": "qe_format_leaderboard ranks fresh-session QE format receipts",
        "promotion_gate": {
            "min_attempts": MIN_QE_RECOMMENDATION_ATTEMPTS,
            "min_format_rate": MIN_QE_PROMOTE_FORMAT_RATE,
            "min_score": MIN_QE_PROMOTE_SCORE,
            "direct_answer_rate": 0.0,
        },
        "champion": asdict(leaderboard.champion) if leaderboard.champion else None,
        "entries": [asdict(entry) for entry in leaderboard.entries],
    }


def _leaderboard_markdown(leaderboard: QeLeaderboard) -> str:
    lines = [
        "# QE Format Leaderboard",
        "",
        (
            "Promotion gate: at least "
            f"{MIN_QE_RECOMMENDATION_ATTEMPTS} fresh attempts, "
            f"{MIN_QE_PROMOTE_FORMAT_RATE:.0%} format rate, "
            f"{MIN_QE_PROMOTE_SCORE:.0%} score, and zero direct answers."
        ),
        "",
    ]
    champion = leaderboard.champion
    if champion is None:
        lines.extend(
            [
                "No QE format receipts found yet.",
                "",
                "Next run: `apb qe-format --model MODEL --base-url http://127.0.0.1:PORT`",
                "",
            ]
        )
        return "\n".join(lines)
    winner_label = "Champion" if champion.action == "PROMOTE_QE_PROFILE" else "Top candidate"
    lines.extend(
        [
            f"{winner_label}: `{champion.model}`",
            f"Action: `{champion.action}`",
            f"Recommendation: {champion.recommendation}",
            f"Next run: {champion.next_run}",
            "",
            "| Rank | Action | Score | Format | Direct answers | TPS | TTFT | Attempts | Model | Receipt |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for rank, entry in enumerate(leaderboard.entries, start=1):
        lines.append(
            f"| {rank} | `{entry.action}` | {entry.score:.3f} | "
            f"{entry.format_rate:.0%} | {entry.direct_answer_rate:.0%} | "
            f"{_fmt_optional(entry.median_tps)} | {_fmt_optional(entry.median_ttft_ms)} | "
            f"{entry.attempts} | `{entry.model}` | `{entry.receipt_path}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _resource_summary(resources: dict[str, Any]) -> dict[str, Any]:
    end = dict(resources.get("end") or {})
    delta = dict(resources.get("delta") or {})
    keys = (
        "gpu_used_mb",
        "gpu_used_memory_mb",
        "gpu_total_mb",
        "gpu_util_percent",
        "gpu_utilization_pct",
        "gpu_power_watts",
        "ram_available_mb",
        "ram_used_mb",
        "ram_used_percent",
        "process_rss_mb",
        "cpu_used_percent",
        "swap_used_percent",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        if key in end:
            summary[f"end_{key}"] = end[key]
        if key in delta:
            summary[f"delta_{key}"] = delta[key]
    return summary


def _float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if not isinstance(value, int | float | str | bytes | bytearray):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return _float(value)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"
