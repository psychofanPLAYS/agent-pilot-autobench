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


def _init_connection(connection: sqlite3.Connection) -> None:
    """Create all tables (core + extended) on an already-open connection."""
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
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS question_attempts (
            id      INTEGER PRIMARY KEY,
            model_key   TEXT NOT NULL,
            pack_id     TEXT NOT NULL,
            question_id TEXT NOT NULL,
            outcome     TEXT NOT NULL,
            ts          TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS selection_cursor (
            model_key   TEXT NOT NULL,
            pack_id     TEXT NOT NULL,
            cursor      INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (model_key, pack_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_context_limit (
            model_key     TEXT NOT NULL,
            kv_cache_type TEXT NOT NULL,
            max_context   INTEGER NOT NULL,
            hit_oom       INTEGER NOT NULL DEFAULT 0,
            ts            TEXT NOT NULL,
            PRIMARY KEY (model_key, kv_cache_type)
        )
        """
    )
    connection.commit()


def init_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        _init_connection(connection)


# ---------------------------------------------------------------------------
# question_attempts helpers
# ---------------------------------------------------------------------------


def record_question_attempt(
    conn: sqlite3.Connection,
    model_key: str,
    pack_id: str,
    question_id: str,
    outcome: str,
    ts: str,
) -> None:
    """Insert one row into question_attempts."""
    conn.execute(
        "INSERT INTO question_attempts (model_key, pack_id, question_id, outcome, ts)"
        " VALUES (?, ?, ?, ?, ?)",
        (model_key, pack_id, question_id, outcome, ts),
    )
    conn.commit()


def lifetime_pack_stats(
    conn: sqlite3.Connection,
    model_key: str,
    pack_id: str,
) -> dict[str, object]:
    """Return lifetime statistics for a (model_key, pack_id) pair.

    Keys
    ----
    seen      – count of DISTINCT question_ids attempted
    correct   – count of distinct question_ids whose MOST RECENT outcome == "correct"
    accuracy  – correct / seen, or 0.0 when seen == 0
    last_seen – max ts across all attempts, or None
    """
    # Count of distinct questions attempted
    row_seen = conn.execute(
        "SELECT COUNT(DISTINCT question_id) FROM question_attempts"
        " WHERE model_key=? AND pack_id=?",
        (model_key, pack_id),
    ).fetchone()
    seen: int = row_seen[0] if row_seen else 0

    # Most-recent outcome per question_id (use max ts as a proxy for "latest")
    row_correct = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT question_id,
                   outcome,
                   ROW_NUMBER() OVER (
                       PARTITION BY question_id
                       ORDER BY ts DESC
                   ) AS rn
            FROM question_attempts
            WHERE model_key=? AND pack_id=?
        ) latest
        WHERE rn = 1 AND outcome = 'correct'
        """,
        (model_key, pack_id),
    ).fetchone()
    correct: int = row_correct[0] if row_correct else 0

    row_last = conn.execute(
        "SELECT MAX(ts) FROM question_attempts WHERE model_key=? AND pack_id=?",
        (model_key, pack_id),
    ).fetchone()
    last_seen: str | None = row_last[0] if row_last else None

    accuracy = (correct / seen) if seen > 0 else 0.0

    return {
        "seen": seen,
        "correct": correct,
        "accuracy": accuracy,
        "last_seen": last_seen,
    }


# ---------------------------------------------------------------------------
# selection_cursor helpers
# ---------------------------------------------------------------------------


def get_selection_cursor(
    conn: sqlite3.Connection,
    model_key: str,
    pack_id: str,
) -> int:
    """Return the stored cursor position, defaulting to 0 if absent."""
    row = conn.execute(
        "SELECT cursor FROM selection_cursor WHERE model_key=? AND pack_id=?",
        (model_key, pack_id),
    ).fetchone()
    return int(row[0]) if row else 0


def set_selection_cursor(
    conn: sqlite3.Connection,
    model_key: str,
    pack_id: str,
    cursor: int,
) -> None:
    """Upsert the cursor position for a (model_key, pack_id) pair."""
    conn.execute(
        """
        INSERT INTO selection_cursor (model_key, pack_id, cursor)
        VALUES (?, ?, ?)
        ON CONFLICT(model_key, pack_id) DO UPDATE SET cursor=excluded.cursor
        """,
        (model_key, pack_id, cursor),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# model_context_limit helpers — remember a model's usable context across runs
# ---------------------------------------------------------------------------


def record_context_limit(
    conn: sqlite3.Connection,
    model_key: str,
    kv_cache_type: str,
    max_context: int,
    hit_oom: bool,
    ts: str,
) -> None:
    """Remember the largest context a model served, keeping the best seen.

    Stored per (model, kv cache type) so a future session can warm-start the
    context ladder instead of rediscovering the ceiling from scratch.
    """
    conn.execute(
        """
        INSERT INTO model_context_limit (model_key, kv_cache_type, max_context, hit_oom, ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(model_key, kv_cache_type) DO UPDATE SET
            max_context = MAX(model_context_limit.max_context, excluded.max_context),
            hit_oom = excluded.hit_oom,
            ts = excluded.ts
        """,
        (model_key, kv_cache_type, int(max_context), 1 if hit_oom else 0, ts),
    )
    conn.commit()


def get_context_limit(
    conn: sqlite3.Connection,
    model_key: str,
    kv_cache_type: str,
) -> dict[str, object] | None:
    """Return the remembered context limit for a model, or None if never run."""
    row = conn.execute(
        """
        SELECT max_context, hit_oom, ts FROM model_context_limit
        WHERE model_key=? AND kv_cache_type=?
        """,
        (model_key, kv_cache_type),
    ).fetchone()
    if not row:
        return None
    return {"max_context": int(row[0]), "hit_oom": bool(row[1]), "ts": row[2]}
