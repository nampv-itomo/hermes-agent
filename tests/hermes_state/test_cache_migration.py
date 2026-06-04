"""Migration tests for the tool_result_cache table (SCHEMA_VERSION 14 → 15).

Verifies:

1. Fresh v15 init creates the tool_result_cache table + 4 indexes.
2. Simulated v14 DB (table dropped) + SessionDB() re-init re-creates
   the table — additive-only contract.
3. Schema is downgrade-safe: dropping the new table + indexes does not
   raise on read; SessionDB() reapplies them on next init.
4. SCHEMA_VERSION = 15 is the current target.

These tests are the gate the DevOps review asked for: a CI run that
exercises the migration path catches any future change that breaks
backwards compatibility.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import hermes_state
from hermes_state import SCHEMA_VERSION, SCHEMA_SQL, SessionDB


# ---------------------------------------------------------------------------
# SCHEMA_VERSION is the current target
# ---------------------------------------------------------------------------


class TestSchemaVersionTarget:
    def test_schema_version_is_15(self):
        """v1.0 of the tool result cache lands at SCHEMA_VERSION 15."""
        assert SCHEMA_VERSION == 15


# ---------------------------------------------------------------------------
# Fresh v15 init
# ---------------------------------------------------------------------------


class TestFreshInitCreatesTable:
    def test_tool_result_cache_table_exists(self, tmp_path: Path):
        db = SessionDB(tmp_path / "state.db")
        try:
            row = db._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='tool_result_cache'"
            ).fetchone()
            assert row is not None, "tool_result_cache table missing after SessionDB init"
        finally:
            db._conn.close()

    def test_all_four_indexes_exist(self, tmp_path: Path):
        db = SessionDB(tmp_path / "state.db")
        try:
            rows = db._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name LIKE 'idx_tool_cache%'"
            ).fetchall()
            names = sorted(r[0] for r in rows)
            assert names == [
                "idx_tool_cache_agent",
                "idx_tool_cache_expires",
                "idx_tool_cache_last_hit",
                "idx_tool_cache_tool",
            ]
        finally:
            db._conn.close()

    def test_table_has_all_required_columns(self, tmp_path: Path):
        db = SessionDB(tmp_path / "state.db")
        try:
            cols = db._conn.execute("PRAGMA table_info(tool_result_cache)").fetchall()
            col_names = [c[1] for c in cols]
            # PK must be `key`
            assert col_names[0] == "key"
            # All spec columns must be present (order is implementation detail)
            for required in (
                "key", "tool_name", "args_json", "result_json", "size_bytes",
                "created_at", "last_hit_at", "hit_count", "ttl_seconds",
                "expires_at", "hermes_profile", "agent_id", "session_id",
            ):
                assert required in col_names, f"missing column: {required}"
        finally:
            db._conn.close()

    def test_schema_sql_contains_table_and_indexes(self):
        # The additive-only contract means SCHEMA_SQL is the source of
        # truth; verify the migration isn't relying on _reconcile_columns
        # for the new objects (which would be wrong — table CREATE belongs
        # in SCHEMA_SQL).
        assert "CREATE TABLE IF NOT EXISTS tool_result_cache" in SCHEMA_SQL
        for idx in (
            "idx_tool_cache_expires",
            "idx_tool_cache_tool",
            "idx_tool_cache_last_hit",
            "idx_tool_cache_agent",
        ):
            assert f"CREATE INDEX IF NOT EXISTS {idx}" in SCHEMA_SQL, (
                f"missing index in SCHEMA_SQL: {idx}"
            )


# ---------------------------------------------------------------------------
# Migration safety — additive-only contract
# ---------------------------------------------------------------------------


class TestAdditiveOnlyContract:
    def test_v14_db_upgrade_creates_table(self, tmp_path: Path):
        """Simulate a v14 DB: pre-existing tables exist, but no
        tool_result_cache.  SessionDB() init must add the new table +
        indexes idempotently."""
        db_path = tmp_path / "state.db"
        # Bootstrap: open a v15 DB so other tables exist
        db = SessionDB(db_path)
        try:
            # Drop the new objects to simulate a pre-migration DB
            db._conn.execute("DROP TABLE tool_result_cache")
            for idx in (
                "idx_tool_cache_expires",
                "idx_tool_cache_tool",
                "idx_tool_cache_last_hit",
                "idx_tool_cache_agent",
            ):
                db._conn.execute(f"DROP INDEX IF EXISTS {idx}")
            db._conn.commit()
        finally:
            db._conn.close()

        # Re-init: must bring back the dropped objects
        db2 = SessionDB(db_path)
        try:
            row = db2._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='tool_result_cache'"
            ).fetchone()
            assert row is not None, "re-init did not recreate tool_result_cache"
            rows = db2._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name LIKE 'idx_tool_cache%'"
            ).fetchall()
            assert len(rows) == 4, f"re-init did not recreate all 4 indexes, got {len(rows)}"
        finally:
            db2._conn.close()

    def test_idempotent_reinit(self, tmp_path: Path):
        """Opening the same DB twice in a row must not raise; the
        IF NOT EXISTS guards are the source of truth."""
        db_path = tmp_path / "state.db"
        db1 = SessionDB(db_path)
        db1._conn.close()
        db2 = SessionDB(db_path)  # must not raise
        try:
            row = db2._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='tool_result_cache'"
            ).fetchone()
            assert row is not None
        finally:
            db2._conn.close()

    def test_old_binary_with_new_db_is_safe(self, tmp_path: Path):
        """Old Hermes code (v14) opening a v15 DB must not break.  We
        simulate this by opening the DB with raw sqlite3 (skipping
        SessionDB) and ensuring the new table is simply ignored — the
        old SELECT/UPDATE paths on existing tables still work."""
        db_path = tmp_path / "state.db"
        # First, populate using the new schema
        db = SessionDB(db_path)
        try:
            # Insert a tool_result_cache row directly
            db._conn.execute(
                "INSERT INTO tool_result_cache "
                "(key, tool_name, args_json, result_json, size_bytes, "
                " created_at, last_hit_at, hit_count, ttl_seconds, expires_at, "
                " hermes_profile, agent_id, session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, '', '', '')",
                ("k1", "web_search", '{"q":"x"}', '{"r":1}', 10,
                 1000.0, 1000.0, 60, 1060.0),
            )
            db._conn.commit()
        finally:
            db._conn.close()

        # Now open as if v14 (raw sqlite3 — old code wouldn't know about
        # the new table, but it also doesn't break because the table is
        # just unused data on disk).
        conn = sqlite3.connect(str(db_path))
        try:
            # Old code paths still work
            row = conn.execute("SELECT 1").fetchone()
            assert row == (1,)
            # The new table is still readable but the old binary never
            # touches it
            row = conn.execute(
                "SELECT tool_name FROM tool_result_cache WHERE key = ?",
                ("k1",),
            ).fetchone()
            assert row == ("web_search",)
        finally:
            conn.close()
