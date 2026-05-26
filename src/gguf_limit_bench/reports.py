from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


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
    failure: str
    settings: dict
    receipt_path: str


@dataclass(frozen=True)
class Leaderboard:
    entries: list[LeaderboardEntry]

    @property
    def champion(self) -> LeaderboardEntry:
        return self.entries[0]


def build_leaderboard(runs_root: Path) -> Leaderboard:
    entries: list[LeaderboardEntry] = []
    for best_path in sorted(runs_root.glob("*/best-settings.json")):
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        result = payload.get("result", {})
        settings = payload.get("settings", {})
        model_path = str(payload.get("model", ""))
        failure = str(result.get("failure", "unknown"))
        context = int(settings.get("context_size") or 0)
        entries.append(
            LeaderboardEntry(
                run_id=best_path.parent.name,
                model_name=Path(model_path).name,
                model_path=model_path,
                score=float(payload.get("score", -10_000.0)),
                status=_status_for(failure, float(payload.get("score", -10_000.0))),
                context_label="default/unset" if context == 0 else str(context),
                generation_tps=float(result.get("generation_tokens_per_second") or 0.0),
                prompt_tps=float(result.get("prompt_tokens_per_second") or 0.0),
                failure=failure,
                settings=settings,
                receipt_path=str(best_path.parent),
            )
        )
    return Leaderboard(entries=sorted(entries, key=lambda entry: entry.score, reverse=True))


def write_leaderboard(runs_root: Path) -> Leaderboard:
    runs_root.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard(runs_root)
    if not leaderboard.entries:
        (runs_root / "leaderboard.md").write_text("# Agent Pilot Autobench Leaderboard\n\nNo runs found.\n", encoding="utf-8")
        return leaderboard
    (runs_root / "leaderboard.md").write_text(_leaderboard_markdown(leaderboard), encoding="utf-8")
    (runs_root / "champion.json").write_text(
        json.dumps(asdict(leaderboard.champion), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return leaderboard


def _status_for(failure: str, score: float) -> str:
    if failure == "model_load":
        return "LOAD FAIL"
    if score <= -10_000:
        return "FAIL"
    return "PASS"


def _leaderboard_markdown(leaderboard: Leaderboard) -> str:
    champion = leaderboard.champion
    lines = [
        "# Agent Pilot Autobench Leaderboard",
        "",
        "## Champion",
        "",
        f"- Model: `{champion.model_name}`",
        f"- Score: `{champion.score:.2f}`",
        f"- Status: `{champion.status}`",
        f"- Context: `{champion.context_label}`",
        f"- Generation: `{champion.generation_tps:.2f}` tok/s",
        f"- Prompt: `{champion.prompt_tps:.2f}` tok/s",
        "",
        "## Runs",
        "",
        "| Rank | Status | Score | Gen tok/s | Prompt tok/s | Context | Model |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for rank, entry in enumerate(leaderboard.entries, start=1):
        lines.append(
            f"| {rank} | {entry.status} | {entry.score:.2f} | "
            f"{entry.generation_tps:.2f} | {entry.prompt_tps:.2f} | "
            f"{entry.context_label} | `{entry.model_name}` |"
        )
    lines.append("")
    return "\n".join(lines)
