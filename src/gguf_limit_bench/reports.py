from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path

from gguf_limit_bench.autoresearch import parse_llama_bench_jsonl
from gguf_limit_bench.discovery import is_non_generative_gguf
from gguf_limit_bench.evidence import display_status, evidence_status, normalize_success_failure


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


@dataclass(frozen=True)
class ModelComparison:
    entries: list[ModelComparisonEntry]


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
                agent_bench_score=_float_or_none(result.get("agent_bench_score")),
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
            )
        )
    return Leaderboard(entries=sorted(entries, key=_leaderboard_rank_key, reverse=True))


def build_model_comparison(leaderboard: Leaderboard) -> ModelComparison:
    groups: dict[tuple[str, str], list[LeaderboardEntry]] = {}
    for entry in leaderboard.entries:
        groups.setdefault((entry.model_name, entry.model_path), []).append(entry)

    comparison_entries: list[ModelComparisonEntry] = []
    for (model_name, model_path), runs in groups.items():
        ranked_runs = sorted(runs, key=_leaderboard_rank_key, reverse=True)
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
            )
        )
    return ModelComparison(
        entries=sorted(comparison_entries, key=_model_comparison_rank_key, reverse=True)
    )


def _model_comparison_rank_key(entry: ModelComparisonEntry) -> tuple[int, float, int]:
    return (
        {
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
        }.get(entry.best_status, 0),
        entry.best_score,
        entry.run_count,
    )


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
            "# Agent Pilot Autobench Leaderboard\n\nNo runs found.\n",
            encoding="utf-8",
        )
        _write_empty_model_comparison(runs_root)
        (runs_root / "results.html").write_text(_empty_html(), encoding="utf-8")
        return leaderboard
    model_comparison = build_model_comparison(leaderboard)
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


def _leaderboard_markdown(leaderboard: Leaderboard) -> str:
    champion = leaderboard.champion
    lines = [
        "# Agent Pilot Autobench Leaderboard",
        "",
        "## Plain-English Takeaway",
        "",
        f"- Best measured model: `{champion.model_name}`",
        f"- Result: {_plain_english_status(champion)}",
        f"- Proof folder: `{champion.receipt_path}`",
        "",
        "## Champion",
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
    lines.append("")
    return "\n".join(lines)


def _write_empty_model_comparison(runs_root: Path) -> None:
    (runs_root / "model-comparison.md").write_text(
        "# pilotBENCHY Model Comparison\n\nNo model runs found yet.\n",
        encoding="utf-8",
    )
    (runs_root / "model-comparison.json").write_text("[]\n", encoding="utf-8")


def _model_comparison_markdown(comparison: ModelComparison) -> str:
    lines = [
        "# pilotBENCHY Model Comparison",
        "",
        "This is the model-level view. It groups repeated runs by model so pilotBENCHY can "
        "compare best-known settings per model instead of treating every receipt folder as a "
        "separate universe.",
        "",
        "| Rank | Model | Runs | Best status | Best score | Context | Gen TPS | Prompt TPS | Cold TTFT | Warm TTFT | Serving TPS | Suite | Best receipt |",
        "|---:|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, entry in enumerate(comparison.entries, start=1):
        lines.append(
            f"| {rank} | `{entry.model_name}` | {entry.run_count} | {entry.best_status} | "
            f"{entry.best_score:.2f} | {entry.best_context_label} | "
            f"{entry.generation_tps:.2f} | {entry.prompt_tps:.2f} | "
            f"{_format_ms(entry.cold_ttft_ms)} | {_format_ms(entry.warm_ttft_ms)} | "
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
            "  <title>Agent Pilot Autobench Results</title>",
            "  <style>",
            _html_css(),
            "  </style>",
            "</head>",
            "<body>",
            '  <main class="shell">',
            '    <section class="hero">',
            '      <p class="eyebrow">No runs yet</p>',
            "      <h1>Agent Pilot Autobench Results</h1>",
            "      <p>Run a benchmark first, then refresh this report.</p>",
            "    </section>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )


def _leaderboard_html(leaderboard: Leaderboard) -> str:
    champion = leaderboard.champion
    model_comparison = build_model_comparison(leaderboard)
    rows = "\n".join(
        _html_row(rank, entry) for rank, entry in enumerate(leaderboard.entries, start=1)
    )
    model_rows = "\n".join(
        _model_html_row(rank, entry) for rank, entry in enumerate(model_comparison.entries, start=1)
    )
    settings = "\n".join(
        f"<li><span>{escape(str(key))}</span><strong>{escape(str(value))}</strong></li>"
        for key, value in sorted(champion.settings.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Agent Pilot Autobench Results</title>
  <style>
{_html_css()}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">Current champion</p>
      <h1>Agent Pilot Autobench Results</h1>
      <p class="lede">{escape(champion.model_name)} is the current measured winner.</p>
      <div class="score-grid">
        <div><span>Score</span><strong>{champion.score:.2f}</strong></div>
        <div><span>Status</span><strong>{escape(champion.status)}</strong></div>
        <div><span>Generation</span><strong>{champion.generation_tps:.2f} tok/s</strong></div>
        <div><span>Prompt</span><strong>{champion.prompt_tps:.2f} tok/s</strong></div>
        <div><span>Cold TTFT</span><strong>{escape(_format_ms(champion.serving_ttft_ms))}</strong></div>
        <div><span>Warm TTFT</span><strong>{escape(_format_ms(champion.serving_warm_ttft_ms))}</strong></div>
        <div><span>Warmup Penalty</span><strong>{escape(_format_ms(champion.serving_warmup_penalty_ms))}</strong></div>
        <div><span>Server Ready</span><strong>{escape(_format_ms(champion.serving_server_ready_ms))}</strong></div>
        <div><span>Start To First Token</span><strong>{escape(_format_ms(champion.serving_cold_start_to_first_token_ms))}</strong></div>
        <div><span>Serving</span><strong>{escape(_format_tps(champion.serving_tps))}</strong></div>
        <div><span>Agent Bench</span><strong>{escape(_format_score(champion.agent_bench_score))}</strong></div>
        <div><span>Suite</span><strong>{escape(champion.benchmark_suite_status)}</strong></div>
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
        <li>Open <code>_runs\\model-comparison.md</code> when you want the per-model winner view.</li>
        <li>
          Run <code>agent-autobench export-profile</code> to create a localhost-safe
          server profile.
        </li>
        <li>Use the receipt path below when an AI agent needs to inspect the proof.</li>
      </ol>
      <p class="receipt">Receipt: <code>{escape(champion.receipt_path)}</code></p>
    </section>
    <section class="panel">
      <h2>Winning settings</h2>
      <ul class="settings">
        {settings}
      </ul>
    </section>
    <section class="panel">
      <h2>Best models on this hardware</h2>
      <p class="receipt">Grouped by model name and path, ranked by the best proven run for each model.</p>
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Model</th><th>Runs</th><th>Status</th><th>Score</th>
            <th>Context</th><th>Gen TPS</th><th>Cold TTFT</th><th>Serving</th><th>Receipt</th>
          </tr>
        </thead>
        <tbody>
          {model_rows}
        </tbody>
      </table>
    </section>
    <section class="panel">
      <h2>All runs</h2>
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


def _model_html_row(rank: int, entry: ModelComparisonEntry) -> str:
    status_class = "pass" if entry.best_status in {"WORKFLOW SMOKE", "BENCHMARK SUITE"} else "fail"
    return (
        f'<tr class="{status_class}">'
        f"<td>{rank}</td>"
        f"<td><code>{escape(entry.model_name)}</code></td>"
        f"<td>{entry.run_count}</td>"
        f"<td>{escape(entry.best_status)}</td>"
        f"<td>{entry.best_score:.2f}</td>"
        f"<td>{escape(entry.best_context_label)}</td>"
        f"<td>{entry.generation_tps:.2f}</td>"
        f"<td>{escape(_format_ms(entry.cold_ttft_ms))}</td>"
        f"<td>{escape(_format_tps(entry.serving_tps))}</td>"
        f"<td><code>{escape(entry.best_receipt_path)}</code></td>"
        "</tr>"
    )


def _model_recommendation(best: LeaderboardEntry, run_count: int) -> str:
    if best.status == "BENCHMARK SUITE":
        return "Suite-backed candidate. Use this as the current per-model champion."
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
        "Keep the receipt, but do not treat it as the production champion yet."
    )


def _format_ms(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.0f} ms"


def _format_tps(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.2f} tok/s"


def _format_score(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.4f}"


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
    .score-grid div, .settings li {
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
    @media (max-width: 720px) {
      h1 { font-size: 1.8rem; }
      .hero, .panel { padding: 18px; }
      table { font-size: 0.88rem; }
    }
    """
