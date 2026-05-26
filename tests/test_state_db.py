import sqlite3

from gguf_limit_bench.state_db import CORE_TABLES, init_state_db


def test_init_state_db_creates_research_memory_tables(tmp_path):
    db_path = tmp_path / "db" / "agentpilot.sqlite"

    init_state_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert CORE_TABLES <= tables
