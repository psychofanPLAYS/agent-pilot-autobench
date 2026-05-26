from __future__ import annotations

from pathlib import Path
import sqlite3


CORE_TABLES = {
    "models",
    "machines",
    "backend_builds",
    "experiments",
    "runs",
    "run_settings",
    "metrics_perf",
    "metrics_gpu",
    "metrics_quality",
    "metrics_context",
    "metrics_tool_use",
    "errors",
    "artifacts",
    "champions",
    "learned_rules",
    "planner_proposals",
    "skipped_runs",
}


def init_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        for table in sorted(CORE_TABLES):
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    payload_json TEXT NOT NULL DEFAULT '{{}}'
                )
                """
            )
        connection.commit()
