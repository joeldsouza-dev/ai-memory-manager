-- =============================================================================
-- Migration: 0002_fix_conversation_project_cascade.sql
-- Description: Fix the conversations → projects composite FK to cascade on
--              project deletion, so deleting a project also removes its
--              conversation (and downstream messages/summaries).
--
-- Root cause: PostgreSQL does not support ON DELETE CASCADE on a composite
-- foreign key created inline; the original FK was created with NO ACTION.
-- This patch drops and re-creates the constraint with CASCADE.
-- =============================================================================

BEGIN;

ALTER TABLE conversations
    DROP CONSTRAINT conversations_project_ownership_fk;

ALTER TABLE conversations
    ADD CONSTRAINT conversations_project_ownership_fk
        FOREIGN KEY (project_id, user_id)
        REFERENCES projects(id, user_id)
        ON DELETE CASCADE;

COMMIT;

