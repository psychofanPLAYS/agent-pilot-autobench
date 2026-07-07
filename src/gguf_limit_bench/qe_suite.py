"""Fresh-session query-expansion format probe.

This suite is intentionally narrower than the librarian MC packs: it measures
whether a QE lane emits usable retrieval payloads over repeated fresh sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable

from gguf_limit_bench.pack_runner import _chat
from gguf_limit_bench.qe_format import (
    QeFormatAssessment,
    assess_qe_response,
    summarize_qe_assessments,
)
from gguf_limit_bench.telemetry import sample_telemetry


DEFAULT_REPEATS = 10
DEFAULT_ANSWER_MAX_TOKENS = 128
QE_SYSTEM_PROMPT = (
    "You are a query expansion model. Do not answer the user's question. "
    "Return exactly one LEX line with 1-3 comma-separated search terms and "
    "exactly one HYDE line with a synthetic document to retrieve."
)


@dataclass(frozen=True)
class QeCase:
    case_id: str
    user_question: str


ChatFn = Callable[..., tuple[str, float | None, float, float, int]]
TelemetrySampler = Callable[[], object]


DEFAULT_QE_CASES: tuple[QeCase, ...] = (
    QeCase("template-reasoning", "Why did the Qwen helper stop showing reasoning?"),
    QeCase("runtime-flags", "What should I inspect before trusting llama.cpp flags?"),
    QeCase("context-expectations", "Find the note about 262k context expectations."),
    QeCase("data-hygiene", "Which notes say bad benchmark data must not corrupt learned data?"),
    QeCase("interface-split", "Show why the web UI and TUI should share the same runners."),
)


def run_qe_format_suite(
    *,
    model: str,
    base_url: str,
    out_dir: Path,
    cases: Iterable[QeCase] = DEFAULT_QE_CASES,
    repeats: int = DEFAULT_REPEATS,
    answer_max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
    timeout_seconds: int = 600,
    sampling: dict[str, object] | None = None,
    chat: ChatFn = _chat,
    telemetry_sampler: TelemetrySampler = sample_telemetry,
) -> dict[str, object]:
    """Run QE cases in fresh sessions and write deterministic receipts."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if answer_max_tokens < 1:
        raise ValueError("answer_max_tokens must be at least 1.")
    case_list = tuple(cases)
    if not case_list:
        raise ValueError("at least one QE case is required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    resource_start = _telemetry_dict(telemetry_sampler())
    attempts: list[dict[str, object]] = []
    assessments: list[QeFormatAssessment] = []
    for case in case_list:
        for repeat in range(1, repeats + 1):
            response, ttft_ms, tps, prompt_tps, generated_tokens = chat(
                base_url=base_url,
                system_prompt=QE_SYSTEM_PROMPT,
                user_content=_case_prompt(case),
                max_tokens=answer_max_tokens,
                timeout_seconds=timeout_seconds,
                sampling=sampling,
            )
            assessment = assess_qe_response(response)
            assessments.append(assessment)
            attempts.append(
                {
                    "case_id": case.case_id,
                    "repeat": repeat,
                    "prompt": _case_prompt(case),
                    "response": response,
                    "format_ok": assessment.format_ok,
                    "lex_terms": list(assessment.lex_terms),
                    "hyde": assessment.hyde,
                    "answered_question": assessment.answered_question,
                    "issues": list(assessment.issues),
                    "score": assessment.score,
                    "ttft_ms": ttft_ms,
                    "tokens_per_second": tps,
                    "prompt_tokens_per_second": prompt_tps,
                    "generated_tokens": generated_tokens,
                }
            )
    resource_end = _telemetry_dict(telemetry_sampler())

    summary = _summary_payload(
        model=model,
        base_url=base_url,
        cases=case_list,
        repeats=repeats,
        answer_max_tokens=answer_max_tokens,
        sampling=sampling or {},
        resources={
            "start": resource_start,
            "end": resource_end,
            "delta": _telemetry_delta(resource_start, resource_end),
        },
        assessments=assessments,
        attempts=attempts,
    )
    _write_receipts(out_dir, summary, attempts)
    return summary


def _case_prompt(case: QeCase) -> str:
    return (
        f"User question:\n{case.user_question}\n\n"
        "Return only:\n"
        "LEX: term one, term two, term three\n"
        "HYDE: synthetic document that would satisfy the retrieval need"
    )


def _summary_payload(
    *,
    model: str,
    base_url: str,
    cases: tuple[QeCase, ...],
    repeats: int,
    answer_max_tokens: int,
    sampling: dict[str, object],
    resources: dict[str, object],
    assessments: list[QeFormatAssessment],
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    summary = summarize_qe_assessments(assessments)
    attempts_count = int(summary["attempts"])
    direct_answer_count = int(summary["direct_answer_count"])
    return {
        "model": model,
        "base_url": base_url,
        "suite": "qe-format",
        "status": "scored",
        "case_count": len(cases),
        "repeats": repeats,
        "answer_max_tokens": answer_max_tokens,
        "sampling": sampling,
        "resources": resources,
        "attempts": attempts_count,
        "valid": summary["valid"],
        "format_rate": summary["format_rate"],
        "direct_answer_count": direct_answer_count,
        "direct_answer_rate": direct_answer_count / attempts_count if attempts_count else 0.0,
        "issue_counts": summary["issue_counts"],
        "score": summary["score"],
        "score_contract": "qe_format_score = mean deterministic LEX/HYDE format score",
        "median_tps": _median(
            [
                float(attempt["tokens_per_second"])
                for attempt in attempts
                if float(attempt["tokens_per_second"]) > 0.0
            ]
        )
        or 0.0,
        "median_prompt_tps": _median(
            [
                float(attempt["prompt_tokens_per_second"])
                for attempt in attempts
                if float(attempt["prompt_tokens_per_second"]) > 0.0
            ]
        )
        or 0.0,
        "median_ttft_ms": _median(
            [float(attempt["ttft_ms"]) for attempt in attempts if attempt["ttft_ms"] is not None]
        ),
    }


def _write_receipts(
    out_dir: Path, summary: dict[str, object], attempts: list[dict[str, object]]
) -> None:
    (out_dir / "qe-format-summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "qe-format-attempts.json").write_text(
        json.dumps(attempts, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "qe-format-summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# QE Format Suite: {summary['model']}",
            "",
            f"- Score: `{_fmt_float(summary['score'])}`",
            f"- Format rate: `{_fmt_float(summary['format_rate'])}`",
            f"- Direct-answer rate: `{_fmt_float(summary['direct_answer_rate'])}`",
            f"- Attempts: `{summary['attempts']}`",
            f"- Repeats per case: `{summary['repeats']}`",
            f"- Median generation speed: `{_fmt_float(summary['median_tps'])}` tok/s",
            f"- Median prompt speed: `{_fmt_float(summary['median_prompt_tps'])}` tok/s",
            f"- Median TTFT: `{summary['median_ttft_ms'] if summary['median_ttft_ms'] is not None else 'n/a'}` ms",
            "",
            "## Issues",
            "",
            *[
                f"- `{issue}`: `{count}`"
                for issue, count in sorted(dict(summary["issue_counts"]).items())
            ],
            "",
        ]
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt_float(value: object) -> str:
    return f"{float(value):.6f}"


def _telemetry_dict(snapshot: object) -> dict[str, object]:
    to_dict = getattr(snapshot, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _telemetry_delta(start: dict[str, object], end: dict[str, object]) -> dict[str, object]:
    keys = sorted(set(start) | set(end))
    delta: dict[str, object] = {}
    for key in keys:
        before = start.get(key)
        after = end.get(key)
        if isinstance(before, int | float) and isinstance(after, int | float):
            delta[key] = after - before
    return delta
