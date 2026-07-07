from __future__ import annotations

from pathlib import Path

from gguf_limit_bench.reports import (
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard,
    build_verdict,
)


def truncated_previous_runs_text(runs_root: Path, limit: int = 8) -> str:
    """Return compact run history text for the TUI without dumping full receipts."""
    leaderboard = build_leaderboard(runs_root)
    if not leaderboard.entries and runs_root.name == "_runs":
        legacy_root = Path("runs")
        if legacy_root.exists():
            leaderboard = build_leaderboard(legacy_root)
    if not leaderboard.entries:
        return "Previous runs\nNo receipts yet. Run a model to create the first comparable receipt."
    return _history_text(leaderboard, limit=limit)


def _history_text(leaderboard: Leaderboard, limit: int) -> str:
    lines = ["Previous runs", "Rank  Status            Score     Gen TPS   Agent     Model"]
    for rank, entry in enumerate(leaderboard.entries[:limit], start=1):
        lines.append(_entry_line(rank, entry))
    remaining = len(leaderboard.entries) - limit
    if remaining > 0:
        lines.append(
            f"... {remaining} older run(s) hidden. Open _runs/results.html for the full report."
        )
    champion = leaderboard.champion
    verdict = build_verdict(leaderboard)
    result_label = "Recommended model" if verdict.action == "PROMOTE" else "Top candidate"
    lines.extend(
        [
            "",
            f"{result_label}: {champion.model_name}",
            f"Proof: {champion.receipt_path}",
            f"Meaning: {champion.status} | context {champion.context_label}",
        ]
    )
    return "\n".join(lines)


def _entry_line(rank: int, entry: LeaderboardEntry) -> str:
    agent_score = "n/a" if entry.agent_bench_score is None else f"{entry.agent_bench_score:.2f}"
    model = entry.model_name
    if len(model) > 46:
        model = f"{model[:43]}..."
    return (
        f"{rank:>4}  {entry.status:<16}  {entry.score:>7.2f}  "
        f"{entry.generation_tps:>7.2f}  {agent_score:>7}  {model}"
    )
