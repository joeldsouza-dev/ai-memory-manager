"""
memory/jobs.py
--------------
Job creation, claiming, processing, failure handling, and crash recovery.

Key Architectural Guarantees
----------------------------
1. Transaction Boundary:
   Fetching the source message is committed immediately so the connection
   is completely IDLE during the extractor (LLM) execution. No database locks
   or snapshots remain open while waiting for external APIs.

2. Savepoint-Guarded Idempotency:
   Candidate identity (extraction_id) is derived deterministically from candidate
   attributes (memory_type, content, project_id) rather than list position.
   Insertion into `job_memories` uses savepoints so duplicate candidates are
   skipped cleanly without rolling back the entire job or duplicating memories.

3. Structured Retry & Exponential Backoff:
   Failures are categorized as terminal (e.g. TerminalError, ValueError) or
   retryable. Retryable jobs are scheduled for redelivery via `next_retry_at`
   with exponential backoff up to `max_retries`.

4. Scoped Memory Persistence & Project Ownership:
   Supports both global (project_id=None) and project-scoped memories. Project
   ownership is verified against the database before persistence.

5. Database-Level Ownership Integrity:
   Composite foreign keys on `memory_jobs`:
     (message_id, conversation_id) -> messages(id, conversation_id)
     (conversation_id, user_id)    -> conversations(id, user_id)
   guarantee that a job's user owns the conversation containing the source message.

6. Fencing Token & Lease Recovery:
   Every claim assigns a fresh `lease_token`. Job completion requires matching
   the current lease_token, preventing stale/expired workers from committing after
   another worker has reclaimed the job.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import psycopg

from memory.db import get_connection
from memory.models import (
    FencingTokenError,
    MemoryCandidate,
    MemoryJob,
    Message,
    ProjectOwnershipError,
    RetryableError,
    TerminalError,
)

# UUID v5 namespace for deterministic candidate extraction IDs
_EXTRACTION_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

DEFAULT_LEASE_SECONDS = 300  # 5 minutes
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_BACKOFF_SECONDS = 1.0


# ---------------------------------------------------------------------------
# Internal Sentinel Exception
# ---------------------------------------------------------------------------


class _CandidateConflict(Exception):
    """Internal sentinel raised to trigger savepoint rollback on duplicate insert."""
    pass


# ---------------------------------------------------------------------------
# Extraction ID Helper
# ---------------------------------------------------------------------------


def make_extraction_id(
    job_id: uuid.UUID,
    candidate_or_content: MemoryCandidate | str,
    memory_type: str = "general",
    project_id: Optional[uuid.UUID] = None,
) -> uuid.UUID:
    """Generate a deterministic UUID v5 for a candidate.

    Derived from (job_id, memory_type, project_id, normalized_content).
    This guarantees that reordering candidates across retries resolves to the
    exact same extraction_id without relying on list positions.
    """
    if isinstance(candidate_or_content, MemoryCandidate):
        c = candidate_or_content
        content_key = c.content.strip().lower()
        type_key = c.memory_type.strip().lower()
        project_key = str(c.project_id) if c.project_id else "global"
    else:
        content_key = str(candidate_or_content).strip().lower()
        type_key = str(memory_type).strip().lower()
        project_key = str(project_id) if project_id else "global"

    seed = f"{job_id}:{type_key}:{project_key}:{content_key}"
    return uuid.uuid5(_EXTRACTION_NS, seed)


def is_terminal_error(exc: Exception) -> bool:
    """Return True if the failure is terminal and should not be retried."""
    if isinstance(exc, (TerminalError, ValueError)):
        return True
    return False


# ---------------------------------------------------------------------------
# Job Creation
# ---------------------------------------------------------------------------


def create_job(
    conn: psycopg.Connection,
    message_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: Optional[uuid.UUID] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> MemoryJob:
    """Insert a new pending memory_job row.

    Must be called with an active connection/transaction. If conversation_id
    is not provided, it is looked up from the message to satisfy composite FKs.
    Composite foreign keys enforce that user_id owns the conversation.
    """
    with conn.cursor() as cur:
        if conversation_id is None:
            cur.execute(
                "SELECT conversation_id FROM messages WHERE id = %s",
                (str(message_id),),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Message {message_id} not found.")
            conversation_id = row[0]

        cur.execute(
            """
            INSERT INTO memory_jobs (message_id, conversation_id, user_id, max_retries)
            VALUES (%s, %s, %s, %s)
            RETURNING id, message_id, conversation_id, user_id, status, lease_token,
                      locked_at, last_error, retry_count, max_retries, next_retry_at,
                      created_at, updated_at
            """,
            (str(message_id), str(conversation_id), str(user_id), max_retries),
        )
        job_row = cur.fetchone()

    return _row_to_job(job_row)


# ---------------------------------------------------------------------------
# Job Claiming & Queue Handling
# ---------------------------------------------------------------------------


def claim_job(
    conn: psycopg.Connection,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    job_id: Optional[uuid.UUID] = None,
) -> Optional[MemoryJob]:
    """Atomically claim one eligible job using SELECT ... FOR UPDATE SKIP LOCKED.

    Eligible jobs:
      1. New pending jobs: status = 'pending'
      2. Retryable failed jobs: status = 'failed' AND retry_count < max_retries AND next_retry_at <= NOW()
      3. Expired worker leases: status = 'processing' AND locked_at < NOW() - lease_seconds

    Assigns a fresh lease_token (fencing token) on every claim.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=lease_seconds)
    now = datetime.now(tz=timezone.utc)

    where_clauses = [
        "status = 'pending'",
        "(status = 'failed' AND retry_count < max_retries AND next_retry_at IS NOT NULL AND next_retry_at <= %s)",
        "(status = 'processing' AND locked_at < %s)",
    ]
    query_condition = f"({' OR '.join(where_clauses)})"
    params: list = [now, cutoff]

    if job_id is not None:
        query_condition = f"id = %s AND {query_condition}"
        params.insert(0, str(job_id))

    sql = f"""
        UPDATE memory_jobs
        SET status      = 'processing',
            locked_at   = NOW(),
            lease_token = gen_random_uuid(),
            updated_at  = NOW()
        WHERE id = (
            SELECT id FROM memory_jobs
            WHERE {query_condition}
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, message_id, conversation_id, user_id, status, lease_token,
                  locked_at, last_error, retry_count, max_retries, next_retry_at,
                  created_at, updated_at
    """

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        row = cur.fetchone()

    conn.commit()
    return _row_to_job(row) if row else None


def renew_lease(
    conn: psycopg.Connection,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
) -> bool:
    """Renew worker lease for a job currently in processing status.

    Returns True if successfully renewed; False if the job was reclaimed
    by another worker or has already completed.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE memory_jobs
            SET locked_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
              AND lease_token = %s
              AND status = 'processing'
            """,
            (str(job_id), str(lease_token)),
        )
        renewed = cur.rowcount > 0

    conn.commit()
    return renewed


# ---------------------------------------------------------------------------
# Job Processing Pipeline
# ---------------------------------------------------------------------------


def process_job(
    job: MemoryJob,
    extractor: Callable[[Message], list[MemoryCandidate]],
    conn: Optional[psycopg.Connection] = None,
    base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
) -> list[uuid.UUID]:
    """Process a claimed job end-to-end with strict transaction boundaries.

    Flow:
      1. Fetch original message and commit read transaction immediately.
      2. Invoke extractor outside any DB transaction.
      3. Compute candidate extraction_ids deterministically (independent of list order).
      4. Validate candidates and project ownership.
      5. Open single atomic write transaction:
         - Insert candidate memories with savepoints on duplicate detection.
         - Update job to 'completed' with fencing token validation.
         - Commit.
      6. On failure: record last_error, increment retry_count, schedule exponential
         backoff if retryable, and re-raise.
    """
    _conn = conn or get_connection()
    should_close = conn is None
    try:
        return _run(job, extractor, _conn)
    except Exception as exc:
        _fail_job(
            conn=_conn,
            job=job,
            exc=exc,
            base_backoff_seconds=base_backoff_seconds,
        )
        raise
    finally:
        if should_close:
            _conn.close()


def _run(
    job: MemoryJob,
    extractor: Callable[[Message], list[MemoryCandidate]],
    conn: psycopg.Connection,
) -> list[uuid.UUID]:
    # ── Step 1: Fetch source message and commit read transaction ────────────
    message = _fetch_message(conn, job.message_id)

    # In psycopg, SELECT creates an implicit transaction when autocommit=False.
    # We commit it here so the connection is completely IDLE before the LLM call.
    conn.commit()
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        conn.rollback()

    # ── Step 2: Extraction runs OUTSIDE any database transaction ────────────
    candidates = extractor(message)

    # ── Step 3: Assign deterministic extraction_ids ──────────────────────────
    # Derived from candidate attributes, independent of list order.
    for candidate in candidates:
        if candidate.extraction_id is None:
            candidate.extraction_id = make_extraction_id(job.id, candidate)

    # ── Step 4: Validate candidates & project ownership ─────────────────────
    _validate_candidates(conn, job, candidates)

    # ── Step 5: Atomic write transaction with fencing token check ────────────
    persisted_ids: list[uuid.UUID] = []
    with conn.transaction():
        for candidate in candidates:
            mem_id = _insert_candidate(conn, job, message, candidate)
            if mem_id is not None and mem_id not in persisted_ids:
                persisted_ids.append(mem_id)

        # Fencing check: ensure current worker still owns the valid lease_token
        _complete_job(conn, job.id, job.lease_token)

    return persisted_ids


# ---------------------------------------------------------------------------
# Private Database Operations
# ---------------------------------------------------------------------------


def _fetch_message(conn: psycopg.Connection, message_id: uuid.UUID) -> Message:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, conversation_id, role, content, created_at FROM messages WHERE id = %s",
            (str(message_id),),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"Message {message_id} not found.")
    return Message(
        id=row[0],
        conversation_id=row[1],
        role=row[2],
        content=row[3],
        created_at=row[4],
    )


def _validate_candidates(
    conn: psycopg.Connection,
    job: MemoryJob,
    candidates: list[MemoryCandidate],
) -> None:
    """Validate structure and verify project ownership for project-scoped candidates."""
    valid_types = {"semantic", "episodic", "procedural"}
    for i, c in enumerate(candidates):
        if c.memory_type not in valid_types:
            raise ValueError(
                f"Candidate {i}: invalid memory_type {c.memory_type!r}. "
                f"Must be one of {valid_types}."
            )
        if not c.content or not c.content.strip():
            raise ValueError(f"Candidate {i}: content must not be empty.")
        if not (0.0 <= c.importance <= 1.0):
            raise ValueError(
                f"Candidate {i}: importance {c.importance} out of [0, 1]."
            )
        if not (0.0 <= c.confidence <= 1.0):
            raise ValueError(
                f"Candidate {i}: confidence {c.confidence} out of [0, 1]."
            )

        # Validate project ownership if candidate specifies a project
        if c.project_id is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM projects WHERE id = %s AND user_id = %s",
                    (str(c.project_id), str(job.user_id)),
                )
                if cur.fetchone() is None:
                    raise ProjectOwnershipError(
                        f"Project {c.project_id} does not exist or is not owned by user {job.user_id}."
                    )


def _insert_candidate(
    conn: psycopg.Connection,
    job: MemoryJob,
    message: Message,
    candidate: MemoryCandidate,
) -> Optional[uuid.UUID]:
    """Persist a memory candidate using a savepoint to guard against duplicates.

    Returns the memory UUID, or the existing memory UUID if already persisted.
    """
    # Quick check: already recorded in job_memories for this job?
    with conn.cursor() as cur:
        cur.execute(
            "SELECT memory_id FROM job_memories WHERE job_id = %s AND extraction_id = %s",
            (str(job.id), str(candidate.extraction_id)),
        )
        existing = cur.fetchone()
        if existing:
            return existing[0]

    # Savepoint block: insert memory then link in job_memories.
    # If a conflict occurs in job_memories, rollback to savepoint discards the memory.
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories
                        (user_id, project_id, source_conversation_id, source_message_id,
                         memory_type, category, content, importance, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        str(job.user_id),
                        str(candidate.project_id) if candidate.project_id else None,
                        str(message.conversation_id),
                        str(message.id),
                        candidate.memory_type,
                        candidate.category,
                        candidate.content,
                        candidate.importance,
                        candidate.confidence,
                    ),
                )
                memory_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO job_memories (job_id, memory_id, extraction_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (job_id, extraction_id) DO NOTHING
                    """,
                    (str(job.id), str(memory_id), str(candidate.extraction_id)),
                )

                if cur.rowcount == 0:
                    # Duplicate extraction_id raced or already present; roll back memory insertion
                    raise _CandidateConflict()

                return memory_id
    except _CandidateConflict:
        # Retrieve the winning memory_id from job_memories
        with conn.cursor() as cur:
            cur.execute(
                "SELECT memory_id FROM job_memories WHERE job_id = %s AND extraction_id = %s",
                (str(job.id), str(candidate.extraction_id)),
            )
            row = cur.fetchone()
            return row[0] if row else None


def _complete_job(
    conn: psycopg.Connection,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
) -> None:
    """Mark job completed. Validates fencing token to reject stale workers."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE memory_jobs
            SET status     = 'completed',
                locked_at  = NULL,
                updated_at = NOW()
            WHERE id = %s
              AND lease_token = %s
              AND status IN ('processing', 'pending')
            """,
            (str(job_id), str(lease_token)),
        )
        if cur.rowcount == 0:
            raise FencingTokenError(
                f"Cannot complete job {job_id}: lease expired or reclaimed by another worker."
            )


def _fail_job(
    conn: psycopg.Connection,
    job: MemoryJob,
    exc: Exception,
    base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
) -> None:
    """Record job failure and schedule exponential backoff if retryable."""
    is_terminal = is_terminal_error(exc)
    new_retry_count = job.retry_count + 1

    if is_terminal or new_retry_count >= job.max_retries:
        next_retry = None
    else:
        # Exponential backoff: base * 2^(retry_count - 1)
        delay = base_backoff_seconds * (2 ** (new_retry_count - 1))
        next_retry = datetime.now(tz=timezone.utc) + timedelta(seconds=delay)

    error_msg = str(exc)[:2000]

    try:
        if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
            conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memory_jobs
                SET status        = 'failed',
                    locked_at     = NULL,
                    last_error    = %s,
                    retry_count   = %s,
                    next_retry_at = %s,
                    updated_at    = NOW()
                WHERE id = %s
                  AND lease_token = %s
                """,
                (error_msg, new_retry_count, next_retry, str(job.id), str(job.lease_token)),
            )
        conn.commit()
    except Exception:
        pass


def _row_to_job(row: tuple) -> MemoryJob:
    return MemoryJob(
        id=row[0],
        message_id=row[1],
        conversation_id=row[2],
        user_id=row[3],
        status=row[4],
        lease_token=row[5],
        locked_at=row[6],
        last_error=row[7],
        retry_count=row[8],
        max_retries=row[9],
        next_retry_at=row[10],
        created_at=row[11],
        updated_at=row[12],
    )
