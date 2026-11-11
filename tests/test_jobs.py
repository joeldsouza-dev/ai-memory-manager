"""
tests/test_jobs.py
------------------
Comprehensive tests for asynchronous memory job processing:
1. One message creates exactly one job (UNIQUE message_id).
2. One job produces multiple memory records.
3. Zero candidates completes a job successfully.
4. Failure on second insert rolls back the entire transaction.
5. Job completion and memory insertion are atomic.
6. Idempotency with reordered extraction candidates.
7. Retrying a job doesn't duplicate memories (savepoint-guarded).
8. Transaction boundary: connection is completely IDLE during extraction.
9. Failed retryable jobs are rescheduled with exponential backoff.
10. Terminal failures are not retried.
11. Max retries exhaustion stops further claims.
12. Memory scope: global and project-scoped memories.
13. Project ownership validation rejects unauthorized projects.
14. Database integrity: composite FK rejects memory jobs for mismatched users.
15. Fencing token prevents stale/expired worker from completing a reclaimed job.
16. Worker lease renewal works while processing.
17. Expired worker lease allows job reclamation.
18. Two concurrent workers cannot claim the same active job simultaneously.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import patch

import psycopg
import pytest

from memory.db import get_connection
from memory.jobs import (
    DEFAULT_LEASE_SECONDS,
    claim_job,
    create_job,
    make_extraction_id,
    process_job,
    renew_lease,
)
from memory.models import (
    FencingTokenError,
    MemoryCandidate,
    MemoryJob,
    Message,
    ProjectOwnershipError,
    TerminalError,
)
from tests.conftest import make_conversation, make_message, make_project, make_user


# ---------------------------------------------------------------------------
# Extractor Helpers
# ---------------------------------------------------------------------------


def extractor_with(*candidates: MemoryCandidate):
    def _extractor(message: Message) -> list[MemoryCandidate]:
        # Return shallow copies so extraction_id can be assigned independently
        return [
            MemoryCandidate(
                memory_type=c.memory_type,
                content=c.content,
                extraction_id=c.extraction_id,
                category=c.category,
                project_id=c.project_id,
                importance=c.importance,
                confidence=c.confidence,
            )
            for c in candidates
        ]
    return _extractor


def null_extractor(message: Message) -> list[MemoryCandidate]:
    return []


def make_candidate(
    content: str,
    memory_type: str = "semantic",
    project_id: Optional[uuid.UUID] = None,
    importance: float = 0.5,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=memory_type,
        content=content,
        project_id=project_id,
        importance=importance,
    )


# ---------------------------------------------------------------------------
# 1. One message creates one job
# ---------------------------------------------------------------------------


def test_one_message_one_job(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "I prefer Python")

    job = create_job(conn, msg_id, user_id)

    assert job.message_id == msg_id
    assert job.conversation_id == conv_id
    assert job.user_id == user_id
    assert job.status == "pending"
    assert job.retry_count == 0
    assert job.lease_token is not None

    with pytest.raises(psycopg.errors.UniqueViolation):
        create_job(conn, msg_id, user_id)


# ---------------------------------------------------------------------------
# 2. One job produces multiple memory records
# ---------------------------------------------------------------------------


def test_one_job_multiple_memories(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "I prefer Python and hints before solutions")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    extractor = extractor_with(
        make_candidate("User prefers Python for coding.", "semantic"),
        make_candidate("User prefers hints before complete solutions.", "procedural"),
    )

    memory_ids = process_job(job, extractor, conn)

    assert len(memory_ids) == 2

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM memories WHERE id = ANY(%s)",
            ([str(m) for m in memory_ids],),
        )
        assert len(cur.fetchall()) == 2

        cur.execute(
            "SELECT memory_id FROM job_memories WHERE job_id = %s",
            (str(job.id),),
        )
        linked = {r[0] for r in cur.fetchall()}
    assert {str(m) for m in memory_ids} == {str(m) for m in linked}

    with conn.cursor() as cur:
        cur.execute("SELECT status FROM memory_jobs WHERE id = %s", (str(job.id),))
        assert cur.fetchone()[0] == "completed"


# ---------------------------------------------------------------------------
# 3. Zero candidates completes successfully
# ---------------------------------------------------------------------------


def test_zero_candidates_completes_job(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "ok")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    memory_ids = process_job(job, null_extractor, conn)
    assert memory_ids == []

    with conn.cursor() as cur:
        cur.execute("SELECT status FROM memory_jobs WHERE id = %s", (str(job.id),))
        assert cur.fetchone()[0] == "completed"


# ---------------------------------------------------------------------------
# 4. Failure on second insert rolls back the first
# ---------------------------------------------------------------------------


def test_second_insert_failure_rolls_back_first(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "I love Python and coffee")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    extractor = extractor_with(
        make_candidate("User loves Python."),
        make_candidate("User loves coffee."),
    )

    import memory.jobs as jobs_module
    original_insert = jobs_module._insert_candidate
    calls = []

    def _patched_insert(c, job_, msg, candidate):
        calls.append(candidate.content)
        if len(calls) == 2:
            raise RuntimeError("Simulated failure on second insert")
        return original_insert(c, job_, msg, candidate)

    with patch.object(jobs_module, "_insert_candidate", side_effect=_patched_insert):
        with pytest.raises(RuntimeError, match="Simulated failure on second insert"):
            process_job(job, extractor, conn)

    conn.rollback()

    # Neither memory must exist in DB
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM memories WHERE source_message_id = %s",
            (str(msg_id),),
        )
        assert cur.fetchone()[0] == 0

        cur.execute(
            "SELECT status, retry_count FROM memory_jobs WHERE id = %s",
            (str(job.id),),
        )
        row = cur.fetchone()
    assert row[0] == "failed"
    assert row[1] == 1


# ---------------------------------------------------------------------------
# 5. Job completion and memory insertion are atomic
# ---------------------------------------------------------------------------


def test_job_completion_is_atomic(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Atomic test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    import memory.jobs as jobs_module

    def _broken_complete(c, job_id, lease_token):
        raise RuntimeError("Simulated completion error")

    with patch.object(jobs_module, "_complete_job", side_effect=_broken_complete):
        with pytest.raises(RuntimeError, match="Simulated completion error"):
            process_job(job, extractor_with(make_candidate("Atomic memory")), conn)

    conn.rollback()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM memories WHERE source_message_id = %s",
            (str(msg_id),),
        )
        assert cur.fetchone()[0] == 0

        cur.execute(
            "SELECT status FROM memory_jobs WHERE id = %s",
            (str(job.id),),
        )
        assert cur.fetchone()[0] == "failed"


# ---------------------------------------------------------------------------
# 6. Idempotency: Reordered extraction candidates resolve to same identity
# ---------------------------------------------------------------------------


def test_reordered_extraction_candidates_idempotency(conn: psycopg.Connection) -> None:
    """Extraction candidates returned in different orders across retries
    must resolve to the exact same extraction_ids and not duplicate memories."""
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Preferences message")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    cand1 = make_candidate("User prefers dark mode.", "semantic")
    cand2 = make_candidate("User prefers tab size 4.", "procedural")

    # Order 1: [cand1, cand2]
    ext_id1_order1 = make_extraction_id(job.id, cand1)
    ext_id2_order1 = make_extraction_id(job.id, cand2)

    # Order 2: [cand2, cand1]
    ext_id1_order2 = make_extraction_id(job.id, cand1)
    ext_id2_order2 = make_extraction_id(job.id, cand2)

    assert ext_id1_order1 == ext_id1_order2
    assert ext_id2_order1 == ext_id2_order2
    assert ext_id1_order1 != ext_id2_order1

    # First attempt: succeed
    extractor1 = extractor_with(cand1, cand2)
    mem_ids1 = process_job(job, extractor1, conn)
    assert len(mem_ids1) == 2

    # Reset job to processing with same lease to simulate retry with reordered candidates
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memory_jobs SET status = 'processing' WHERE id = %s",
            (str(job.id),),
        )
    conn.commit()

    extractor2 = extractor_with(cand2, cand1)  # reversed!
    mem_ids2 = process_job(job, extractor2, conn)

    # Must not duplicate any memories in database
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM memories WHERE source_message_id = %s",
            (str(msg_id),),
        )
        assert cur.fetchone()[0] == 2

        cur.execute(
            "SELECT COUNT(*) FROM job_memories WHERE job_id = %s",
            (str(job.id),),
        )
        assert cur.fetchone()[0] == 2


# ---------------------------------------------------------------------------
# 7. Retrying a job doesn't duplicate memories (savepoint-guarded)
# ---------------------------------------------------------------------------


def test_retry_does_not_duplicate_memories(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "retry test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    candidate = make_candidate("User likes retries.", "episodic")

    # Process job completely
    mem_ids = process_job(job, extractor_with(candidate), conn)
    assert len(mem_ids) == 1

    # Re-run process_job on same job
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memory_jobs SET status = 'processing' WHERE id = %s",
            (str(job.id),),
        )
    conn.commit()

    mem_ids2 = process_job(job, extractor_with(candidate), conn)
    assert len(mem_ids2) == 1
    assert mem_ids2[0] == mem_ids[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM memories WHERE source_message_id = %s",
            (str(msg_id),),
        )
        assert cur.fetchone()[0] == 1


# ---------------------------------------------------------------------------
# 8. Transaction boundary: Connection is completely IDLE during extraction
# ---------------------------------------------------------------------------


def test_transaction_boundary_idle_during_extraction(conn: psycopg.Connection) -> None:
    """Verifies that no database transaction remains open when the extractor runs."""
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Transaction boundary test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    observed_status = []

    def checking_extractor(message: Message) -> list[MemoryCandidate]:
        # Inspect psycopg connection transaction status during extraction
        observed_status.append(conn.info.transaction_status)
        return [make_candidate("Extracted memory")]

    process_job(job, checking_extractor, conn)

    assert len(observed_status) == 1
    # psycopg.pq.TransactionStatus.IDLE is 0
    assert observed_status[0] == psycopg.pq.TransactionStatus.IDLE


# ---------------------------------------------------------------------------
# 9. Failed retryable jobs are rescheduled with exponential backoff
# ---------------------------------------------------------------------------


def test_retry_mechanism_exponential_backoff(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Backoff test")
    job = create_job(conn, msg_id, user_id, max_retries=3)
    conn.commit()

    def failing_extractor(message: Message):
        raise ConnectionResetError("Transient network failure")

    with pytest.raises(ConnectionResetError):
        process_job(job, failing_extractor, conn, base_backoff_seconds=2.0)

    # Check that status is failed and next_retry_at is scheduled in the future
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, retry_count, next_retry_at FROM memory_jobs WHERE id = %s",
            (str(job.id),),
        )
        status, retries, next_retry = cur.fetchone()

    assert status == "failed"
    assert retries == 1
    assert next_retry is not None
    now = datetime.now(tz=timezone.utc)
    # retry 1 with base 2.0s -> delay ~ 2.0 * (2^0) = 2.0s
    assert next_retry > now
    diff = (next_retry - now).total_seconds()
    assert 0 < diff <= 3.0

    # Simulate arrival of next_retry_at: update next_retry_at to past
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memory_jobs SET next_retry_at = NOW() - INTERVAL '1 second' WHERE id = %s",
            (str(job.id),),
        )
    conn.commit()

    # claim_job should now reclaim this failed job for retry!
    reclaimed = claim_job(conn, job_id=job.id)
    assert reclaimed is not None
    assert reclaimed.id == job.id
    assert reclaimed.status == "processing"
    assert reclaimed.retry_count == 1


# ---------------------------------------------------------------------------
# 10. Terminal failures are not retried
# ---------------------------------------------------------------------------


def test_terminal_failure_not_retried(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Terminal test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    def terminal_extractor(message: Message):
        raise TerminalError("Irrecoverable format error")

    with pytest.raises(TerminalError):
        process_job(job, terminal_extractor, conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, retry_count, next_retry_at FROM memory_jobs WHERE id = %s",
            (str(job.id),),
        )
        status, retries, next_retry = cur.fetchone()

    assert status == "failed"
    assert retries == 1
    assert next_retry is None  # Never scheduled for retry!

    # Attempting to claim must return None
    reclaimed = claim_job(conn, job_id=job.id)
    assert reclaimed is None


# ---------------------------------------------------------------------------
# 11. Max retries exhaustion stops further claims
# ---------------------------------------------------------------------------


def test_max_retries_exceeded(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Max retry test")
    job = create_job(conn, msg_id, user_id, max_retries=2)
    conn.commit()

    def transient_extractor(message: Message):
        raise RuntimeError("Transient error")

    # Attempt 1: retry_count becomes 1 (< max 2) -> scheduled
    with pytest.raises(RuntimeError):
        process_job(job, transient_extractor, conn)

    job.retry_count = 1
    # Attempt 2: retry_count becomes 2 (>= max 2) -> exhausted
    with pytest.raises(RuntimeError):
        process_job(job, transient_extractor, conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, retry_count, next_retry_at FROM memory_jobs WHERE id = %s",
            (str(job.id),),
        )
        status, retries, next_retry = cur.fetchone()

    assert status == "failed"
    assert retries == 2
    assert next_retry is None

    # Cannot be claimed
    assert claim_job(conn, job_id=job.id) is None


# ---------------------------------------------------------------------------
# 12. Memory scope: Global and project-scoped memories
# ---------------------------------------------------------------------------


def test_memory_scope_global_and_project(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    project_id = make_project(conn, user_id, "Personal Assistant Project")
    conv_id = make_conversation(conn, user_id, project_id=project_id)
    msg_id = make_message(conn, conv_id, "Scope test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    extractor = extractor_with(
        make_candidate("Global preference", "semantic", project_id=None),
        make_candidate("Project instruction", "procedural", project_id=project_id),
    )

    mem_ids = process_job(job, extractor, conn)
    assert len(mem_ids) == 2

    with conn.cursor() as cur:
        cur.execute(
            "SELECT content, project_id FROM memories WHERE id = ANY(%s) ORDER BY content",
            ([str(m) for m in mem_ids],),
        )
        rows = cur.fetchall()

    assert rows[0][0] == "Global preference"
    assert rows[0][1] is None

    assert rows[1][0] == "Project instruction"
    assert rows[1][1] == project_id


# ---------------------------------------------------------------------------
# 13. Project ownership validation rejects unauthorized projects
# ---------------------------------------------------------------------------


def test_project_ownership_validation(conn: psycopg.Connection) -> None:
    user_alice = make_user(conn, "Alice")
    user_bob = make_user(conn, "Bob")
    bob_project = make_project(conn, user_bob, "Bob's Secret Project")

    conv_id = make_conversation(conn, user_alice)
    msg_id = make_message(conn, conv_id, "Alice message")
    job = create_job(conn, msg_id, user_alice)
    conn.commit()

    # Alice's candidate improperly references Bob's project
    extractor = extractor_with(
        make_candidate("Unauthorized project memory", project_id=bob_project)
    )

    with pytest.raises(ProjectOwnershipError):
        process_job(job, extractor, conn)

    # ProjectOwnershipError is a TerminalError -> no retry scheduled
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, next_retry_at FROM memory_jobs WHERE id = %s",
            (str(job.id),),
        )
        row = cur.fetchone()
    assert row[0] == "failed"
    assert row[1] is None


# ---------------------------------------------------------------------------
# 14. Database integrity: Composite FK rejects mismatched conversation user
# ---------------------------------------------------------------------------


def test_database_integrity_composite_fk_rejects_mismatched_user(
    conn: psycopg.Connection,
) -> None:
    user_alice = make_user(conn, "Alice")
    user_bob = make_user(conn, "Bob")
    alice_conv = make_conversation(conn, user_alice)
    alice_msg = make_message(conn, alice_conv, "Alice text")

    # Bob attempts to create a job pointing to Alice's message
    # Composite FK (conversation_id, user_id) -> conversations(id, user_id) must reject this!
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        create_job(conn, alice_msg, user_bob)


# ---------------------------------------------------------------------------
# 15. Fencing token prevents stale/expired worker from completing reclaimed job
# ---------------------------------------------------------------------------


def test_fencing_token_prevents_stale_worker_completion(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Fencing token test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    # Worker 1 claims job
    worker1_job = claim_job(conn, job_id=job.id)
    assert worker1_job is not None

    # Simulate worker 1 lease expiring: update locked_at into the past
    expired = datetime.now(tz=timezone.utc) - timedelta(seconds=DEFAULT_LEASE_SECONDS + 60)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memory_jobs SET locked_at = %s WHERE id = %s",
            (expired, str(job.id)),
        )
    conn.commit()

    # Worker 2 reclaims job -> receives a NEW lease_token
    worker2_job = claim_job(conn, job_id=job.id)
    assert worker2_job is not None
    assert worker2_job.lease_token != worker1_job.lease_token

    # Stale Worker 1 attempts to process and complete using its old lease_token
    with pytest.raises(FencingTokenError, match="lease expired or reclaimed"):
        process_job(worker1_job, extractor_with(make_candidate("Stale memory")), conn)

    conn.rollback()

    # Verify no memory from Worker 1 was persisted
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM memories WHERE source_message_id = %s",
            (str(msg_id),),
        )
        assert cur.fetchone()[0] == 0


# ---------------------------------------------------------------------------
# 16. Worker lease renewal works while processing
# ---------------------------------------------------------------------------


def test_lease_renewal(conn: psycopg.Connection) -> None:
    user_id = make_user(conn)
    conv_id = make_conversation(conn, user_id)
    msg_id = make_message(conn, conv_id, "Lease renewal test")
    job = create_job(conn, msg_id, user_id)
    conn.commit()

    claimed = claim_job(conn, job_id=job.id)
    assert claimed is not None

    # Renew lease with valid lease_token
    assert renew_lease(conn, claimed.id, claimed.lease_token) is True

    # Renew lease with fake/old lease_token must fail
    fake_token = uuid.uuid4()
    assert renew_lease(conn, claimed.id, fake_token) is False


# ---------------------------------------------------------------------------
# 17. Expired worker lease allows job reclamation
# ---------------------------------------------------------------------------


def test_expired_lease_job_is_reclaimable() -> None:
    conn = get_connection()
    try:
        user_id = make_user(conn)
        conv_id = make_conversation(conn, user_id)
        msg_id = make_message(conn, conv_id, "Stale claim test")
        job = create_job(conn, msg_id, user_id)

        expired_time = datetime.now(tz=timezone.utc) - timedelta(seconds=DEFAULT_LEASE_SECONDS + 60)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE memory_jobs SET status = 'processing', locked_at = %s WHERE id = %s",
                (expired_time, str(job.id)),
            )
        conn.commit()

        reclaimed = claim_job(conn, lease_seconds=DEFAULT_LEASE_SECONDS, job_id=job.id)
        assert reclaimed is not None
        assert reclaimed.id == job.id
        assert reclaimed.status == "processing"
        assert reclaimed.locked_at is not None
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (str(user_id),))
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# 18. Two concurrent workers cannot claim the same active job
# ---------------------------------------------------------------------------


def test_two_workers_cannot_claim_same_job() -> None:
    setup_conn = get_connection()
    user_id = make_user(setup_conn)
    conv_id = make_conversation(setup_conn, user_id)
    msg_id = make_message(setup_conn, conv_id, "Concurrent claim test")
    job = create_job(setup_conn, msg_id, user_id)
    setup_conn.commit()
    setup_conn.close()

    results: list[Optional[MemoryJob]] = []
    barrier = threading.Barrier(2)
    lock = threading.Lock()

    def worker():
        c = get_connection()
        try:
            barrier.wait()
            claimed = claim_job(c, job_id=job.id)
            with lock:
                results.append(claimed)
        finally:
            c.close()

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    claimed = [r for r in results if r is not None]
    skipped = [r for r in results if r is None]

    assert len(claimed) == 1, f"Expected 1 claim, got {len(claimed)}"
    assert len(skipped) == 1, f"Expected 1 skip, got {len(skipped)}"
    assert claimed[0].id == job.id

    cleanup_conn = get_connection()
    with cleanup_conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (str(user_id),))
    cleanup_conn.commit()
    cleanup_conn.close()
