import sqlite3

import pytest

from gguf_limit_bench.state_db import (
    CORE_TABLES,
    get_selection_cursor,
    init_state_db,
    lifetime_pack_stats,
    record_question_attempt,
    set_selection_cursor,
)


def test_init_state_db_creates_research_memory_tables(tmp_path):
    db_path = tmp_path / "db" / "agentpilot.sqlite"

    init_state_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert CORE_TABLES <= tables


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_conn():
    """In-memory SQLite connection with state_db schema initialised."""
    conn = sqlite3.connect(":memory:")
    # Re-use the same init path: pass a Path that triggers in-memory creation
    # via our helper that accepts a connection directly.
    from gguf_limit_bench.state_db import _init_connection

    _init_connection(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# question_attempts + lifetime_pack_stats
# ---------------------------------------------------------------------------


def test_new_tables_created_by_init(tmp_path):
    db_path = tmp_path / "pilot.sqlite"
    init_state_db(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "question_attempts" in tables
    assert "selection_cursor" in tables


def test_record_question_attempt_inserts_row(mem_conn):
    record_question_attempt(
        mem_conn,
        model_key="model-a",
        pack_id="pack-1",
        question_id="q1",
        outcome="correct",
        ts="2024-01-01T00:00:00",
    )
    rows = mem_conn.execute("SELECT * FROM question_attempts").fetchall()
    assert len(rows) == 1


def test_lifetime_pack_stats_seen_and_correct(mem_conn):
    # q1: wrong then correct (latest = correct)
    record_question_attempt(mem_conn, "m", "p", "q1", "wrong", "2024-01-01T01:00:00")
    record_question_attempt(mem_conn, "m", "p", "q1", "correct", "2024-01-01T02:00:00")
    # q2: correct once
    record_question_attempt(mem_conn, "m", "p", "q2", "correct", "2024-01-01T03:00:00")

    stats = lifetime_pack_stats(mem_conn, model_key="m", pack_id="p")

    assert stats["seen"] == 2
    assert stats["correct"] == 2
    assert stats["accuracy"] == 1.0


def test_lifetime_pack_stats_last_seen(mem_conn):
    record_question_attempt(mem_conn, "m", "p", "q1", "correct", "2024-01-01T01:00:00")
    record_question_attempt(mem_conn, "m", "p", "q2", "wrong", "2024-01-02T00:00:00")

    stats = lifetime_pack_stats(mem_conn, model_key="m", pack_id="p")
    assert stats["last_seen"] == "2024-01-02T00:00:00"


def test_lifetime_pack_stats_partial_correct(mem_conn):
    # q1: latest wrong; q2: correct
    record_question_attempt(mem_conn, "m", "p", "q1", "correct", "2024-01-01T01:00:00")
    record_question_attempt(mem_conn, "m", "p", "q1", "wrong", "2024-01-01T02:00:00")
    record_question_attempt(mem_conn, "m", "p", "q2", "correct", "2024-01-01T03:00:00")

    stats = lifetime_pack_stats(mem_conn, model_key="m", pack_id="p")

    assert stats["seen"] == 2
    assert stats["correct"] == 1
    assert stats["accuracy"] == pytest.approx(0.5)


def test_lifetime_pack_stats_empty(mem_conn):
    stats = lifetime_pack_stats(mem_conn, model_key="m", pack_id="p")
    assert stats["seen"] == 0
    assert stats["correct"] == 0
    assert stats["accuracy"] == 0.0
    assert stats["last_seen"] is None


def test_lifetime_pack_stats_isolation(mem_conn):
    """Stats for one (model, pack) pair don't bleed into another."""
    record_question_attempt(mem_conn, "m1", "pack-a", "q1", "correct", "2024-01-01T00:00:00")
    record_question_attempt(mem_conn, "m2", "pack-b", "q1", "correct", "2024-01-01T00:00:00")

    stats_a = lifetime_pack_stats(mem_conn, model_key="m1", pack_id="pack-a")
    stats_b = lifetime_pack_stats(mem_conn, model_key="m2", pack_id="pack-b")
    stats_none = lifetime_pack_stats(mem_conn, model_key="m1", pack_id="pack-b")

    assert stats_a["seen"] == 1
    assert stats_b["seen"] == 1
    assert stats_none["seen"] == 0


# ---------------------------------------------------------------------------
# selection_cursor
# ---------------------------------------------------------------------------


def test_get_selection_cursor_default_zero(mem_conn):
    assert get_selection_cursor(mem_conn, model_key="m", pack_id="p") == 0


def test_set_and_get_selection_cursor(mem_conn):
    set_selection_cursor(mem_conn, model_key="m", pack_id="p", cursor=5)
    assert get_selection_cursor(mem_conn, model_key="m", pack_id="p") == 5


def test_set_selection_cursor_upsert(mem_conn):
    set_selection_cursor(mem_conn, model_key="m", pack_id="p", cursor=3)
    set_selection_cursor(mem_conn, model_key="m", pack_id="p", cursor=7)
    assert get_selection_cursor(mem_conn, model_key="m", pack_id="p") == 7


def test_selection_cursor_isolation(mem_conn):
    set_selection_cursor(mem_conn, model_key="m1", pack_id="p", cursor=10)
    assert get_selection_cursor(mem_conn, model_key="m2", pack_id="p") == 0
