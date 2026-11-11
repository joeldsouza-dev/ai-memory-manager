-- =============================================================================
-- Migration: 0003_memory_jobs.sql
-- Description: Adds the memory_jobs and job_memories tables to support
--              atomic, idempotent, crash-safe memory extraction job processing.
--
-- New tables:
--   memory_jobs  – One job per user message (UNIQUE message_id).
--                  Tracks status, lease, retries, and errors.
--   job_memories – Association table between a job and the memories it
--                  produced. UNIQUE(job_id, extraction_id) is the
--                  idempotency key: retrying a crashed job cannot create
--                  duplicate memory records.
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- memory_jobs
-- ---------------------------------------------------------------------------
-- Status lifecycle:
--   pending  → (worker claims) → processing → completed
--                                           ↘ failed
--
-- Lease / crash recovery:
--   locked_at is set when a worker claims the job.
--   A job in 'processing' state whose locked_at is older than the lease
--   duration (e.g. 5 minutes) is considered abandoned and can be reclaimed.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_jobs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The message that triggered this extraction job.
    -- UNIQUE ensures exactly one job per message.
    message_id      UUID        NOT NULL UNIQUE
                                REFERENCES messages(id) ON DELETE CASCADE,

    -- The user who owns this message (denormalised for fast queue queries).
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending', 'processing', 'completed', 'failed')),

    -- Worker lease: set when a worker claims the job, cleared on completion.
    -- Jobs in 'processing' with locked_at < NOW() - lease_duration are stale.
    locked_at       TIMESTAMPTZ,

    -- Error message from the last failed attempt.
    last_error      TEXT,

    -- Total number of processing attempts made.
    retry_count     INT         NOT NULL DEFAULT 0,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  memory_jobs            IS 'One extraction job per user message. Tracks status and worker lease.';
COMMENT ON COLUMN memory_jobs.message_id IS 'The message that triggered extraction. UNIQUE enforces one job per message.';
COMMENT ON COLUMN memory_jobs.locked_at  IS 'UTC time when a worker claimed this job. NULL = unclaimed.';
COMMENT ON COLUMN memory_jobs.status     IS 'pending | processing | completed | failed';

CREATE INDEX IF NOT EXISTS memory_jobs_status_idx     ON memory_jobs (status);
CREATE INDEX IF NOT EXISTS memory_jobs_user_id_idx    ON memory_jobs (user_id);
CREATE INDEX IF NOT EXISTS memory_jobs_message_id_idx ON memory_jobs (message_id);

-- Partial index for the queue-claim query: only unlocked pending jobs.
CREATE INDEX IF NOT EXISTS memory_jobs_queue_idx
    ON memory_jobs (created_at)
    WHERE status = 'pending';

-- Partial index for the lease-reclaim query: processing jobs with a lock.
CREATE INDEX IF NOT EXISTS memory_jobs_stale_idx
    ON memory_jobs (locked_at)
    WHERE status = 'processing' AND locked_at IS NOT NULL;


-- ---------------------------------------------------------------------------
-- job_memories
-- ---------------------------------------------------------------------------
-- Association table linking a job to the memories it produced.
--
-- Idempotency design:
--   extraction_id is a deterministic UUID (v5 namespace derived from
--   job_id + candidate index) assigned during extraction — before the
--   transaction opens. UNIQUE(job_id, extraction_id) prevents the same
--   candidate from being inserted twice, even if the worker crashes and
--   restarts, because ON CONFLICT DO NOTHING silently skips duplicates.
--
-- Why not compare content strings?
--   Content can legitimately repeat across different extractions.
--   A content-based dedup would incorrectly suppress valid new memories.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS job_memories (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    job_id          UUID        NOT NULL REFERENCES memory_jobs(id) ON DELETE CASCADE,
    memory_id       UUID        NOT NULL REFERENCES memories(id)    ON DELETE CASCADE,

    -- Deterministic per-candidate key (UUID v5: namespace=job_id, name=str(index)).
    -- The UNIQUE constraint is the idempotency guard.
    extraction_id   UUID        NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT job_memories_job_extraction_key UNIQUE (job_id, extraction_id)
);

COMMENT ON TABLE  job_memories                IS 'Links a memory_job to the memory records it produced. UNIQUE(job_id, extraction_id) ensures idempotency on retry.';
COMMENT ON COLUMN job_memories.extraction_id  IS 'Deterministic UUID (v5) derived from job_id + candidate index. Guards against duplicate inserts on retry.';

CREATE INDEX IF NOT EXISTS job_memories_job_id_idx    ON job_memories (job_id);
CREATE INDEX IF NOT EXISTS job_memories_memory_id_idx ON job_memories (memory_id);


COMMIT;

