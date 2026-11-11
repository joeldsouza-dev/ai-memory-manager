-- =============================================================================
-- Migration: 0004_job_integrity_and_recovery.sql
-- Description: Strengthens database integrity for memory_jobs and adds
--              support for fencing tokens, lease renewal, and exponential backoff.
--
-- 1. Adds UNIQUE (id, user_id) on conversations to enable composite ownership FKs.
-- 2. Adds conversation_id to memory_jobs with composite foreign keys:
--      (message_id, conversation_id) -> messages(id, conversation_id)
--      (conversation_id, user_id)    -> conversations(id, user_id)
--    Guarantees that a memory job's user owns the conversation of the source message.
-- 3. Adds lease_token (fencing token) to prevent stale workers from completing jobs.
-- 4. Adds max_retries and next_retry_at for exponential backoff retry scheduling.
-- 5. Drops redundant indexes.
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. conversations: unique constraint for composite FK reference
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'conversations_id_user_id_key'
    ) THEN
        ALTER TABLE conversations
            ADD CONSTRAINT conversations_id_user_id_key UNIQUE (id, user_id);
    END IF;
END $$;


-- ---------------------------------------------------------------------------
-- 2. memory_jobs: add conversation_id and composite FKs
-- ---------------------------------------------------------------------------

ALTER TABLE memory_jobs
    ADD COLUMN IF NOT EXISTS conversation_id UUID;

-- Backfill conversation_id for any existing rows from messages
UPDATE memory_jobs mj
SET conversation_id = m.conversation_id
FROM messages m
WHERE mj.message_id = m.id
  AND mj.conversation_id IS NULL;

ALTER TABLE memory_jobs
    ALTER COLUMN conversation_id SET NOT NULL;

-- Composite FK 1: validates that message_id belongs to conversation_id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'memory_jobs_message_conversation_fk'
    ) THEN
        ALTER TABLE memory_jobs
            ADD CONSTRAINT memory_jobs_message_conversation_fk
                FOREIGN KEY (message_id, conversation_id)
                REFERENCES messages(id, conversation_id)
                ON DELETE CASCADE;
    END IF;
END $$;

-- Composite FK 2: validates that conversation_id belongs to user_id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'memory_jobs_conversation_user_fk'
    ) THEN
        ALTER TABLE memory_jobs
            ADD CONSTRAINT memory_jobs_conversation_user_fk
                FOREIGN KEY (conversation_id, user_id)
                REFERENCES conversations(id, user_id)
                ON DELETE CASCADE;
    END IF;
END $$;


-- ---------------------------------------------------------------------------
-- 3. memory_jobs: fencing token and retry fields
-- ---------------------------------------------------------------------------

ALTER TABLE memory_jobs
    ADD COLUMN IF NOT EXISTS lease_token UUID DEFAULT gen_random_uuid(),
    ADD COLUMN IF NOT EXISTS max_retries INT NOT NULL DEFAULT 3,
    ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ;

-- Backfill lease_token for existing jobs where it might be null
UPDATE memory_jobs
SET lease_token = gen_random_uuid()
WHERE lease_token IS NULL;

ALTER TABLE memory_jobs
    ALTER COLUMN lease_token SET NOT NULL;


-- ---------------------------------------------------------------------------
-- 4. Remove redundant indexes
-- ---------------------------------------------------------------------------

-- memory_jobs_message_id_idx is redundant with memory_jobs_message_id_key (UNIQUE on message_id)
DROP INDEX IF EXISTS memory_jobs_message_id_idx;

-- job_memories_job_id_idx is redundant with job_memories_job_extraction_key (UNIQUE on (job_id, extraction_id))
DROP INDEX IF EXISTS job_memories_job_id_idx;

-- conversations_project_id_idx is redundant with conversations_project_id_key (UNIQUE on project_id)
DROP INDEX IF EXISTS conversations_project_id_idx;


-- ---------------------------------------------------------------------------
-- 5. Add index for retry queue scheduling
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS memory_jobs_retry_queue_idx
    ON memory_jobs (next_retry_at)
    WHERE status = 'failed' AND next_retry_at IS NOT NULL;


COMMIT;

