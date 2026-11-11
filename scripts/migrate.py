"""
scripts/migrate.py
------------------
Applies all pending SQL migrations in the migrations/ directory in order.

Each migration file is a plain .sql file named NNN_description.sql.
Applied migrations are tracked in a schema_migrations table so the
runner is safe to re-run; already-applied migrations are skipped.

Usage:
    uv run python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from memory.db import get_connection

# Root of the project (parent of scripts/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"

# DDL for the migrations-tracking table.
CREATE_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT        PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def get_applied_versions(conn: psycopg.Connection) -> set[str]:
    """Return the set of migration versions already applied."""
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        return {row[0] for row in cur.fetchall()}


def apply_migration(conn: psycopg.Connection, path: Path) -> None:
    """Execute a single migration file and record it in schema_migrations."""
    version = path.stem          # e.g. "0001_initial_schema"
    sql = path.read_text(encoding="utf-8")

    # Each migration file manages its own BEGIN / COMMIT, so we run it
    # outside of an implicit transaction (autocommit=True for the session).
    # We re-open the connection in autocommit mode for this migration, then
    # record the version in a separate autocommit statement.
    with conn.cursor() as cur:
        # psycopg3 executes multi-statement strings correctly via execute().
        cur.execute(sql)          # migration file already has BEGIN/COMMIT
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s) "
            "ON CONFLICT (version) DO NOTHING",
            (version,),
        )
    conn.commit()


def main() -> None:
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migration_files:
        print("No migration files found in migrations/.")
        sys.exit(0)

    print(f"Found {len(migration_files)} migration file(s).")

    with get_connection() as conn:
        # Ensure the tracking table exists.
        with conn.cursor() as cur:
            cur.execute(CREATE_TRACKING_TABLE_SQL)
        conn.commit()

        applied = get_applied_versions(conn)

        pending = [f for f in migration_files if f.stem not in applied]
        if not pending:
            print("✅  All migrations already applied — nothing to do.")
            return

        for migration_path in pending:
            print(f"  Applying {migration_path.name} …", end=" ", flush=True)
            apply_migration(conn, migration_path)
            print("done.")

    print(f"\n✅  {len(pending)} migration(s) applied successfully.")


if __name__ == "__main__":
    main()

