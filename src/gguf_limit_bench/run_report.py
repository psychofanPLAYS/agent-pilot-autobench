from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from html import escape
import json
from pathlib import Path

from gguf_limit_bench import charts


@dataclass(frozen=True)
class AttemptReport:
    attempt: int
    decision: str
    score: float
    status: str
    context_size: int
    generation_tps: float
    prompt_tps: float
    cold_ttft_ms: float | None
    warm_ttft_ms: float | None
    serving_tps: float | None
    questions: str
    simple_bench_score_fraction: str
    simple_bench_accuracy: float | None
    failure: str
    settings: dict


def write_itemized_run_report(receipt_path: Path) -> None:
    best = json.loads((receipt_path / "best-settings.json").read_text(encoding="utf-8"))
    attempts = _attempts_from_events(receipt_path / "events.jsonl")
    context_profile_rows = _context_profile_rows(receipt_path / "context-profile.json")
    perplexity_profile_rows = _profile_rows(receipt_path / "perplexity-profile.json")
    payload = {
        "run_id": receipt_path.name,
        "model": best.get("model", ""),
        "best_settings": best.get("settings", {}),
        "score": best.get("score"),
        "status": best.get("status"),
        "best_result": _compact_result(best.get("result", {})),
        "attempts": [asdict(attempt) for attempt in attempts],
        "context_scaling": _context_scaling(attempts, context_profile_rows),
        "metric_statuses": _metric_statuses(
            best.get("result", {}), attempts, context_profile_rows, perplexity_profile_rows
        ),
        "recommendation": _recommendation(best, attempts),
        "quality": _results_quality(receipt_path / "results.json"),
    }
    (receipt_path / "report.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (receipt_path / "itemized-report.md").write_text(_markdown(payload), encoding="utf-8")
    (receipt_path / "report.html").write_text(_html(payload), encoding="utf-8")
    # Sync-ready metrics record (Agent Index, speed, efficiency) from this receipt.
    from gguf_limit_bench.metrics import write_run_metrics

    write_run_metrics(receipt_path)


def _results_quality(results_path: Path) -> dict:
    """Per-pack accuracy + aggregate question outcomes from a run's results.json.

    Returns ``{"packs": [...], "correct": int, "wrong": int, "incomplete": int}``;
    empty lists/zeros when the receipt has no scored librarian results.
    """
    empty = {"packs": [], "correct": 0, "wrong": 0, "incomplete": 0}
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    packs: list[dict] = []
    correct = wrong = incomplete = 0
    for pack in payload.get("packs", []) or []:
        if not isinstance(pack, dict) or pack.get("status") != "scored":
            continue
        if pack.get("accuracy") is not None and pack.get("pack_id"):
            packs.append({"pack_id": str(pack["pack_id"]), "accuracy": float(pack["accuracy"])})
        correct += int(pack.get("correct") or 0)
        wrong += int(pack.get("wrong") or 0)
        incomplete += int(pack.get("incomplete") or 0)
    return {"packs": packs, "correct": correct, "wrong": wrong, "incomplete": incomplete}


def _attempts_from_events(events_path: Path) -> list[AttemptReport]:
    attempts: list[AttemptReport] = []
    if not events_path.exists():
        return attempts
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "autoresearch_attempt_finished":
            continue
        data = event.get("data", {})
        result = data.get("result", {})
        settings = data.get("settings", {})
        attempt = AttemptReport(
            attempt=int(data.get("attempt") or 0),
            decision=str(data.get("decision") or ""),
            score=float(data.get("score") or 0.0),
            status=str(result.get("failure") or "unknown"),
            context_size=int(settings.get("context_size") or result.get("context_size") or 0),
            generation_tps=float(result.get("generation_tokens_per_second") or 0.0),
            prompt_tps=float(result.get("prompt_tokens_per_second") or 0.0),
            cold_ttft_ms=_float_or_none(result.get("serving_ttft_ms")),
            warm_ttft_ms=_float_or_none(result.get("serving_warm_ttft_ms")),
            serving_tps=_float_or_none(result.get("serving_tokens_per_second")),
            questions=_question_count_label(
                result.get("completed_questions"), result.get("attempted_questions")
            ),
            simple_bench_score_fraction=_score_fraction(
                result.get("simple_bench_accuracy"), result.get("completed_questions")
            ),
            simple_bench_accuracy=_float_or_none(result.get("simple_bench_accuracy")),
            failure=str(result.get("failure") or "unknown"),
            settings=settings,
        )
        attempts.append(_hydrate_attempt_from_simple_bench_summary(attempt, result))
    return attempts


def _hydrate_attempt_from_simple_bench_summary(
    attempt: AttemptReport, result: dict
) -> AttemptReport:
    summary = _simple_bench_summary(result.get("simple_bench_receipt"))
    if summary is None:
        return attempt

    completed = int(summary.get("total") or 0)
    attempted = int(summary.get("attempted_questions") or completed)
    if completed <= 0:
        return attempt

    decision = "partial" if attempt.decision == "crash" else attempt.decision
    return replace(
        attempt,
        decision=decision,
        generation_tps=_fallback_float(attempt.generation_tps, summary.get("median_tps")),
        prompt_tps=_fallback_float(attempt.prompt_tps, summary.get("median_prompt_tps")),
        cold_ttft_ms=_fallback_optional_float(attempt.cold_ttft_ms, summary.get("median_ttft_ms")),
        questions=_question_count_label(completed, attempted),
        simple_bench_score_fraction=_score_fraction(summary.get("accuracy"), completed),
        simple_bench_accuracy=_fallback_optional_float(
            attempt.simple_bench_accuracy, summary.get("accuracy")
        ),
    )


def _simple_bench_summary(receipt_path: str | None) -> dict | None:
    if not receipt_path:
        return None
    summary_path = Path(receipt_path) / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _question_count_label(completed, attempted) -> str:
    completed_count = int(completed or 0)
    attempted_count = int(attempted or completed_count)
    return f"{completed_count}/{attempted_count}" if attempted_count else "n/a"


def _score_fraction(accuracy, total) -> str:
    if accuracy is None or not total:
        return "n/a"
    total_count = int(total)
    correct = round(float(accuracy) * total_count)
    return f"{correct}/{total_count}"


def _fallback_float(current: float, fallback) -> float:
    if current > 0:
        return current
    if fallback is None:
        return current
    return float(fallback)


def _fallback_optional_float(current: float | None, fallback) -> float | None:
    if current is not None:
        return current
    if fallback is None:
        return None
    return float(fallback)


def _context_profile_rows(profile_path: Path) -> list[dict]:
    return _profile_rows(profile_path)


def _profile_rows(profile_path: Path) -> list[dict]:
    if not profile_path.exists():
        return []
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _compact_result(result: dict) -> dict:
    keys = [
        "ok",
        "generation_tokens_per_second",
        "prompt_tokens_per_second",
        "context_size",
        "serving_ttft_ms",
        "serving_warm_ttft_ms",
        "serving_tokens_per_second",
        "agent_bench_score",
        "benchmark_suite_ok",
        "failure",
    ]
    return {key: result.get(key) for key in keys}


def _context_scaling(attempts: list[AttemptReport], context_profile_rows: list[dict]) -> list[dict]:
    if context_profile_rows:
        return [
            {
                "context_size": int(row.get("context_size") or 0),
                "generation_tps": float(row.get("generation_tps") or 0.0),
                "tps_retention_vs_baseline": float(row.get("tps_retention_vs_baseline") or 0.0),
                "cold_ttft_ms": _float_or_none(row.get("cold_ttft_ms")),
                "warm_ttft_ms": _float_or_none(row.get("warm_ttft_ms")),
            }
            for row in context_profile_rows
        ]
    ordered = sorted(attempts, key=lambda attempt: attempt.context_size)
    if not ordered:
        return []
    baseline = next((attempt for attempt in ordered if attempt.generation_tps > 0), ordered[0])
    baseline_tps = baseline.generation_tps or 1.0
    return [
        {
            "context_size": attempt.context_size,
            "generation_tps": attempt.generation_tps,
            "tps_retention_vs_baseline": attempt.generation_tps / baseline_tps,
            "cold_ttft_ms": attempt.cold_ttft_ms,
            "warm_ttft_ms": attempt.warm_ttft_ms,
        }
        for attempt in ordered
    ]


def _metric_statuses(
    result: dict,
    attempts: list[AttemptReport],
    context_profile_rows: list[dict],
    perplexity_profile_rows: list[dict],
) -> list[dict]:
    successful_contexts = [
        attempt.context_size
        for attempt in attempts
        if attempt.failure == "none" and attempt.context_size > 0 and attempt.generation_tps > 0
    ]
    successful_contexts.extend(
        int(row.get("context_size") or 0)
        for row in context_profile_rows
        if row.get("ok") is True
        and int(row.get("context_size") or 0) > 0
        and float(row.get("generation_tps") or 0.0) > 0
    )
    distinct_contexts = sorted(set(successful_contexts))
    best_context = max(distinct_contexts) if distinct_contexts else None
    successful_perplexity_rows = [
        row
        for row in perplexity_profile_rows
        if row.get("ok") is True and row.get("perplexity") is not None
    ]
    return [
        _metric_status(
            "cold_ttft_ms",
            result.get("serving_ttft_ms"),
            "measured" if result.get("serving_ttft_ms") is not None else "not_measured",
            "Time from starting the server request path to first token.",
        ),
        _metric_status(
            "warm_ttft_ms",
            result.get("serving_warm_ttft_ms"),
            "measured" if result.get("serving_warm_ttft_ms") is not None else "not_measured",
            "Time to first token after the server is already warm.",
        ),
        _metric_status(
            "generation_tps",
            result.get("generation_tokens_per_second"),
            "measured"
            if float(result.get("generation_tokens_per_second") or 0.0) > 0
            else "not_measured",
            "Decode speed from llama-bench or the current attempt runner.",
        ),
        _metric_status(
            "prompt_tps",
            result.get("prompt_tokens_per_second"),
            "measured"
            if float(result.get("prompt_tokens_per_second") or 0.0) > 0
            else "not_measured",
            "Prompt ingestion speed from llama-bench or the current attempt runner.",
        ),
        _metric_status(
            "max_total_usable_context",
            best_context,
            "estimated" if best_context is not None else "not_measured",
            "Largest successful context observed in this receipt. Needs a context ladder for proof.",
        ),
        _metric_status(
            "tps_falloff_with_context",
            None,
            "measured" if len(distinct_contexts) > 1 else "needs_context_ladder",
            "Requires repeated successful attempts at different context sizes.",
        ),
        _metric_status(
            "perplexity_falloff",
            None,
            "measured" if len(successful_perplexity_rows) > 1 else "not_measured",
            "Quality drift over the fixed-context perplexity ladder.",
        ),
        _metric_status(
            "agent_bench_score",
            result.get("agent_bench_score"),
            "measured" if result.get("agent_bench_score") is not None else "not_measured",
            "End-user usefulness score from the benchmark-suite phase.",
        ),
    ]


def _metric_status(name: str, value, status: str, note: str) -> dict:
    return {"metric": name, "status": status, "value": value, "note": note}


def _recommendation(best: dict, attempts: list[AttemptReport]) -> str:
    result = best.get("result", {})
    if not attempts:
        return "No completed attempts were recorded. Rerun with a longer budget or inspect events.jsonl."
    if best.get("status") in {"workflow_unproven", "speed_only", "serving_measured"}:
        return (
            "This is useful systems evidence, but it is not a full agent-readiness result yet. "
            "Run with a benchmark-suite plan before promoting the model."
        )
    if result.get("benchmark_suite_ok") is True:
        return "This run has suite-backed evidence. Compare it by agent_bench_score and receipt details."
    return "Keep this receipt for comparison, then continue iterative testing."


def _markdown(payload: dict) -> str:
    attempts = payload["attempts"]
    lines = [
        f"# Itemized Report: {Path(payload['model']).name}",
        "",
        "## Verdict",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Score: `{payload.get('score')}`",
        f"- Recommendation: {payload['recommendation']}",
        "",
        "## Attempts",
        "",
        "| Attempt | Decision | Context | Gen TPS | Cold TTFT | Warm TTFT | Serving TPS | Questions | Score | Accuracy | Failure |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for attempt in attempts:
        lines.append(
            "| "
            f"{attempt['attempt']} | {attempt['decision']} | {attempt['context_size']} | "
            f"{attempt['generation_tps']:.2f} | {_fmt(attempt['cold_ttft_ms'])} | "
            f"{_fmt(attempt['warm_ttft_ms'])} | {_fmt(attempt['serving_tps'])} | "
            f"{attempt['questions']} | {attempt['simple_bench_score_fraction']} | "
            f"{_fmt(attempt['simple_bench_accuracy'])} | "
            f"{attempt['failure']} |"
        )
    lines.extend(
        [
            "",
            "## Best Settings",
            "",
            "```json",
            json.dumps(payload["best_settings"], ensure_ascii=True, indent=2),
            "```",
            "",
            "## Metric Coverage",
            "",
            "| Metric | Status | Value | Note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for metric in payload["metric_statuses"]:
        lines.append(
            f"| `{metric['metric']}` | `{metric['status']}` | "
            f"`{_fmt(metric['value'])}` | {metric['note']} |"
        )
    lines.extend(
        [
            "",
            "## Context Scaling",
            "",
            "| Context | Gen TPS | Retention vs baseline | Cold TTFT | Warm TTFT |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["context_scaling"]:
        lines.append(
            "| "
            f"{row['context_size']} | {row['generation_tps']:.2f} | "
            f"{row['tps_retention_vs_baseline']:.2f} | {_fmt(row['cold_ttft_ms'])} | "
            f"{_fmt(row['warm_ttft_ms'])} |"
        )
    return "\n".join(lines) + "\n"


def _html(payload: dict) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{row['attempt']}</td><td>{escape(row['decision'])}</td>"
        f"<td>{row['context_size']}</td><td>{row['generation_tps']:.2f}</td>"
        f"<td>{escape(_fmt(row['cold_ttft_ms']))}</td>"
        f"<td>{escape(_fmt(row['warm_ttft_ms']))}</td>"
        f"<td>{escape(_fmt(row['serving_tps']))}</td>"
        f"<td>{escape(row['questions'])}</td>"
        f"<td>{escape(row['simple_bench_score_fraction'])}</td>"
        f"<td>{escape(_fmt(row['simple_bench_accuracy']))}</td>"
        f"<td>{escape(row['failure'])}</td>"
        "</tr>"
        for row in payload["attempts"]
    )
    settings = escape(json.dumps(payload["best_settings"], ensure_ascii=True, indent=2))
    charts_html, chart_runtime = _charts_block(payload)
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{escape(row['metric'])}</td><td>{escape(row['status'])}</td>"
        f"<td>{escape(_fmt(row['value']))}</td><td>{escape(row['note'])}</td>"
        "</tr>"
        for row in payload["metric_statuses"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Agent Pilot Itemized Report</title>
  <style>
    body {{ margin: 0; background: #101418; color: #e8edf2; font: 15px/1.5 Segoe UI, sans-serif; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .panel {{ border: 1px solid #314150; background: #17202a; border-radius: 8px; padding: 18px; margin: 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #314150; padding: 8px; text-align: left; }}
    code, pre {{ background: #0c1116; border: 1px solid #314150; border-radius: 6px; padding: 10px; overflow: auto; }}
    .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .chart-card {{ border: 1px solid #314150; background: #0f1620; border-radius: 8px; padding: 14px; }}
    .chart-card.wide {{ grid-column: 1 / -1; }}
    .chart-card h3 {{ margin: 0 0 4px; font-size: 0.98rem; }}
    .chart-card p {{ margin: 0 0 10px; color: #9aa8b7; font-size: 0.82rem; }}
    @media (max-width: 720px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  {chart_runtime}
  <main>
    <h1>Agent Pilot Itemized Report</h1>
    <section class="panel">
      <h2>{escape(Path(payload["model"]).name)}</h2>
      <p>Status: <strong>{escape(str(payload.get("status")))}</strong></p>
      <p>Score: <strong>{escape(str(payload.get("score")))}</strong></p>
      <p>{escape(payload["recommendation"])}</p>
    </section>
    {charts_html}
    <section class="panel">
      <h2>Attempts</h2>
      <table>
        <thead><tr><th>#</th><th>Decision</th><th>Context</th><th>Gen TPS</th><th>Cold TTFT</th><th>Warm TTFT</th><th>Serving TPS</th><th>Questions</th><th>Score</th><th>Accuracy</th><th>Failure</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Best Settings</h2>
      <pre>{settings}</pre>
    </section>
    <section class="panel">
      <h2>Metric Coverage</h2>
      <table>
        <thead><tr><th>Metric</th><th>Status</th><th>Value</th><th>Note</th></tr></thead>
        <tbody>{metric_rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _run_chart_card(title: str, description: str, chart_html: str, *, wide: bool = False) -> str:
    cls = "chart-card wide" if wide else "chart-card"
    return (
        f'<div class="{cls}"><h3>{escape(title)}</h3>'
        f"<p>{escape(description)}</p>{chart_html}</div>"
    )


def _charts_block(payload: dict) -> tuple[str, str]:
    """Return ``(charts_html, chart_runtime)`` for the per-run report.

    ``chart_runtime`` is empty (and no charts emitted) when nothing is plottable,
    so a bare speed-only receipt does not pay for the inlined library.
    """
    quality = payload.get("quality") or {}
    blocks: list[str] = []

    context = charts.context_scaling_config(payload.get("context_scaling") or [])
    if context is not None:
        blocks.append(
            _run_chart_card(
                "Context scaling",
                "Throughput (and cold TTFT) as the context window grows.",
                charts.render_chart("r-context", context, height=320),
                wide=True,
            )
        )
    attempts = charts.attempts_progression_config(payload.get("attempts") or [])
    if attempts is not None:
        blocks.append(
            _run_chart_card(
                "Optimizer progression",
                "Score across autoresearch attempts.",
                charts.render_chart("r-attempts", attempts, height=300),
            )
        )
    doughnut = charts.outcome_doughnut_config(
        quality.get("correct", 0), quality.get("wrong", 0), quality.get("incomplete", 0)
    )
    if doughnut is not None:
        blocks.append(
            _run_chart_card(
                "Question outcomes",
                "Correct, wrong, and incomplete across scored packs.",
                charts.render_chart("r-outcomes", doughnut, height=300),
            )
        )
    pack_bars = charts.pack_accuracy_bars_config(quality.get("packs") or [])
    if pack_bars is not None:
        blocks.append(
            _run_chart_card(
                "Per-pack accuracy",
                "Accuracy on each librarian pack for this run.",
                charts.render_chart("r-packs", pack_bars, height=300),
            )
        )
    if not blocks:
        return "", ""
    section = '<section class="panel"><h2>Visual overview</h2><div class="chart-grid">' + (
        "".join(blocks)
    ) + "</div></section>"
    return section, charts.chartjs_runtime()


def _float_or_none(value) -> float | None:
    return None if value is None else float(value)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"
