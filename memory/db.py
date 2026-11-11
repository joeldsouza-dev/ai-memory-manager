"""
memory/db.py
------------
Reusable PostgreSQL connection helper.

Configuration is loaded from environment variables (see .env.example).
Call `get_connection()` to obtain a psycopg3 connection.
"""

import os

import psycopg
from dotenv import load_dotenv

# Load variables from a .env file if one is present (safe to call multiple times).
load_dotenv()


def get_connection() -> psycopg.Connection:
    """Return an open psycopg3 connection using environment variables.

    Required environment variables
    --------------------------------
    DB_HOST     – PostgreSQL host        (default: localhost)
    DB_PORT     – PostgreSQL port        (default: 5432)
    DB_NAME     – Database name          (default: ai_memory)
    DB_USER     – Database user          (default: postgres)
    DB_PASSWORD – Database password      (required)
    """
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "ai_memory")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise ValueError(
            "DB_PASSWORD environment variable is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )

    conn = psycopg.connect(
        host=host,
        port=int(port),
        dbname=dbname,
        user=user,
        password=password,
    )
    return conn

