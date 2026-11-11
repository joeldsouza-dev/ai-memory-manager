"""
scripts/test_db.py
------------------
Quick connectivity test – runs `SELECT version()` and prints the result.

Usage:
    uv run python scripts/test_db.py
"""

from memory.db import get_connection


def main() -> None:
    print("Connecting to PostgreSQL …")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            version = row[0] if row else "unknown"
    print(f"✅  Connected successfully!\n\nPostgreSQL version:\n  {version}")


if __name__ == "__main__":
    main()

