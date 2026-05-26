from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
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
        (runs_root / "results.html").write_text(_empty_html(), encoding="utf-8")
        return leaderboard
    (runs_root / "leaderboard.md").write_text(_leaderboard_markdown(leaderboard), encoding="utf-8")
    (runs_root / "results.html").write_text(_leaderboard_html(leaderboard), encoding="utf-8")
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
            "    <section class=\"hero\">",
            "      <p class=\"eyebrow\">No runs yet</p>",
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
    rows = "\n".join(_html_row(rank, entry) for rank, entry in enumerate(leaderboard.entries, start=1))
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
      </div>
    </section>
    <section class="panel">
      <h2>What to do next</h2>
      <ol>
        <li>Open <code>runs\\leaderboard.md</code> when you want the compact Markdown version.</li>
        <li>Run <code>agent-autobench export-profile</code> to create a localhost-safe server profile.</li>
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
      <h2>All runs</h2>
      <table>
        <thead>
          <tr><th>Rank</th><th>Status</th><th>Score</th><th>Generation</th><th>Prompt</th><th>Context</th><th>Model</th></tr>
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
    status_class = "pass" if entry.status == "PASS" else "fail"
    return (
        f'<tr class="{status_class}">'
        f"<td>{rank}</td>"
        f"<td>{escape(entry.status)}</td>"
        f"<td>{entry.score:.2f}</td>"
        f"<td>{entry.generation_tps:.2f}</td>"
        f"<td>{entry.prompt_tps:.2f}</td>"
        f"<td>{escape(entry.context_label)}</td>"
        f"<td><code>{escape(entry.model_name)}</code></td>"
        "</tr>"
    )


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
