"""
tests/conftest.py
-----------------
Shared pytest fixtures: database connection, teardown, and test-entity helpers.

Each test runs inside its own database transaction which is rolled back at the
end — so tests are isolated without needing to truncate tables.
"""

from __future__ import annotations

import uuid
from typing import Generator

import psycopg
import pytest

from memory.db import get_connection


# ---------------------------------------------------------------------------
# Connection fixture — one connection per test, rolled back after
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn() -> Generator[psycopg.Connection, None, None]:
    """Yield an open psycopg3 connection.

    The connection is NOT closed between statements so the test can observe
    uncommitted state.  After the test, any uncommitted work is rolled back.
    """
    c = get_connection()
    yield c
    c.rollback()
    c.close()


# ---------------------------------------------------------------------------
# Entity-creation helpers (all within the test's connection/transaction)
# ---------------------------------------------------------------------------


def make_user(conn: psycopg.Connection, name: str = "Test User") -> uuid.UUID:
    """Insert a user and return its UUID."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (name) VALUES (%s) RETURNING id", (name,)
        )
        return cur.fetchone()[0]


def make_project(
    conn: psycopg.Connection,
    user_id: uuid.UUID,
    name: str = "Test Project",
    instructions: str | None = None,
) -> uuid.UUID:
    """Insert a project and return its UUID."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO projects (user_id, name, instructions) VALUES (%s, %s, %s) RETURNING id",
            (str(user_id), name, instructions),
        )
        return cur.fetchone()[0]


def make_conversation(
    conn: psycopg.Connection,
    user_id: uuid.UUID,
    project_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Insert a conversation and return its UUID."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO conversations (user_id, project_id) VALUES (%s, %s) RETURNING id",
            (str(user_id), str(project_id) if project_id else None),
        )
        return cur.fetchone()[0]


def make_message(
    conn: psycopg.Connection,
    conversation_id: uuid.UUID,
    content: str = "Hello",
    role: str = "user",
) -> uuid.UUID:
    """Insert a message and return its UUID."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) "
            "VALUES (%s, %s, %s) RETURNING id",
            (str(conversation_id), role, content),
        )
        return cur.fetchone()[0]
