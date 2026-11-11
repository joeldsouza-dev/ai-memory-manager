-- =============================================================================
-- Migration: 0001_initial_schema.sql
-- Description: Initial schema for the AI Memory Engine.
--
-- Tables created (in dependency order):
--   1. users
--   2. projects
--   3. conversations
--   4. messages
--   5. conversation_summaries
--   6. memories
--
-- Safe to re-run: all DDL uses IF NOT EXISTS / IF NOT EXISTS guards.
-- Does NOT drop or recreate the database.
-- =============================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- Extension
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- 1. users
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  users            IS 'Human users of the AI memory engine.';
COMMENT ON COLUMN users.id         IS 'Globally unique user identifier (UUID v4).';
COMMENT ON COLUMN users.name       IS 'Display name of the user.';
COMMENT ON COLUMN users.created_at IS 'UTC timestamp when the account was created.';


-- ---------------------------------------------------------------------------
-- 2. projects
-- ---------------------------------------------------------------------------
-- A user can own many projects.
-- The composite UNIQUE (id, user_id) is intentionally redundant with the PK
-- but is required so that downstream tables can reference it as a composite
-- foreign key to enforce ownership without joining back to this table.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS projects (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name         VARCHAR(200) NOT NULL,
    instructions TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Composite unique: enables composite FK ownership checks in child tables.
    CONSTRAINT projects_id_user_id_key UNIQUE (id, user_id)
);

COMMENT ON TABLE  projects              IS 'Projects owned by a user. Each project has exactly one persistent conversation.';
COMMENT ON COLUMN projects.id           IS 'Globally unique project identifier.';
COMMENT ON COLUMN projects.user_id      IS 'Owner of this project.';
COMMENT ON COLUMN projects.name         IS 'Human-readable project name.';
COMMENT ON COLUMN projects.instructions IS 'Optional system-level instructions for this project context.';

CREATE INDEX IF NOT EXISTS projects_user_id_idx ON projects (user_id);


-- ---------------------------------------------------------------------------
-- 3. conversations
-- ---------------------------------------------------------------------------
-- Two kinds of conversations:
--   • Project conversation  : project_id IS NOT NULL, UNIQUE(project_id) → 1 per project.
--   • Standalone conversation: project_id IS NULL, no limit per user.
--
-- Ownership invariant: a project conversation must belong to the same user
-- who owns the project. Enforced via the composite FK:
--   (project_id, user_id) → projects(id, user_id)
-- PostgreSQL skips FK enforcement when ANY column in the referencing side is
-- NULL, so standalone conversations (project_id = NULL) are unaffected.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversations (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id UUID,                -- NULL = standalone conversation
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Exactly one conversation per project.
    CONSTRAINT conversations_project_id_key UNIQUE (project_id),

    -- When project_id IS NOT NULL, the conversation user must own that project.
    -- NULL project_id bypasses this check (PostgreSQL composite FK semantics).
    -- CASCADE: deleting a project removes its conversation (and downstream
    -- messages/summaries via their own CASCADE chains).
    CONSTRAINT conversations_project_ownership_fk
        FOREIGN KEY (project_id, user_id) REFERENCES projects(id, user_id)
        ON DELETE CASCADE
);

COMMENT ON TABLE  conversations            IS 'Conversation threads. Project conversations are 1-to-1 with a project; standalone ones have project_id = NULL.';
COMMENT ON COLUMN conversations.user_id    IS 'User who owns this conversation (must match project owner when project_id is set).';
COMMENT ON COLUMN conversations.project_id IS 'The project this conversation belongs to. NULL for standalone conversations.';

CREATE INDEX IF NOT EXISTS conversations_user_id_idx    ON conversations (user_id);
CREATE INDEX IF NOT EXISTS conversations_project_id_idx ON conversations (project_id) WHERE project_id IS NOT NULL;


-- ---------------------------------------------------------------------------
-- 4. messages
-- ---------------------------------------------------------------------------
-- Every message belongs to a conversation.
-- UNIQUE(id, conversation_id) allows the memories table to validate that a
-- source message actually belongs to a given source conversation via a
-- composite foreign key, without an extra join.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS messages (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Enables composite FK from memories for source-message validation.
    CONSTRAINT messages_id_conversation_id_key UNIQUE (id, conversation_id)
);

COMMENT ON TABLE  messages                IS 'Individual messages within a conversation.';
COMMENT ON COLUMN messages.conversation_id IS 'The conversation this message belongs to.';
COMMENT ON COLUMN messages.role            IS 'Speaker role: user | assistant | system.';
COMMENT ON COLUMN messages.content         IS 'Raw message text.';

CREATE INDEX IF NOT EXISTS messages_conversation_id_idx ON messages (conversation_id);


-- ---------------------------------------------------------------------------
-- 5. conversation_summaries
-- ---------------------------------------------------------------------------
-- Stores the latest compressed summary for a conversation (not the full
-- history). One row per conversation, updated in-place as summarisation runs.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id                         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id            UUID        NOT NULL UNIQUE
                                           REFERENCES conversations(id) ON DELETE CASCADE,
    summary                    TEXT        NOT NULL,
    last_summarized_message_id UUID        REFERENCES messages(id) ON DELETE SET NULL,
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  conversation_summaries                           IS 'Latest compressed summary for each conversation. Upserted, never appended.';
COMMENT ON COLUMN conversation_summaries.last_summarized_message_id IS 'The most recent message included in the current summary. NULL if unknown.';


-- ---------------------------------------------------------------------------
-- 6. memories
-- ---------------------------------------------------------------------------
-- Memories are extracted insights stored permanently.
-- Scope:
--   • Global (project_id = NULL)  : visible across all projects for the user.
--   • Project-scoped (project_id IS NOT NULL): specific to one project.
--
-- Ownership: when project_id IS NOT NULL, the composite FK
--   (project_id, user_id) → projects(id, user_id)
-- ensures the memory belongs to the same user who owns the project.
-- NULL project_id bypasses the FK (PostgreSQL composite FK semantics),
-- so global memories are NEVER touched by cascading project deletions.
--
-- Source references:
--   • source_conversation_id: the conversation the memory was extracted from.
--   • source_message_id: the specific triggering message (optional).
-- A composite FK (source_message_id, source_conversation_id) → messages
-- guarantees the message actually belongs to that conversation.
-- A CHECK prevents a dangling source_message_id with no conversation.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memories (
    id                     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- NULL = global user memory; non-NULL = project-scoped memory.
    project_id             UUID,

    -- Optional: conversation from which this memory was extracted.
    source_conversation_id UUID         REFERENCES conversations(id) ON DELETE SET NULL,

    -- Optional: specific message that triggered extraction.
    -- Must belong to source_conversation_id when both are set.
    source_message_id      UUID,

    memory_type            VARCHAR(30)  NOT NULL
                           CHECK (memory_type IN ('semantic', 'episodic', 'procedural')),

    category               VARCHAR(100),
    content                TEXT         NOT NULL,
    embedding              VECTOR(384),                     -- populated separately

    importance             FLOAT        NOT NULL DEFAULT 0.5
                           CHECK (importance >= 0 AND importance <= 1),

    confidence             FLOAT        NOT NULL DEFAULT 1.0
                           CHECK (confidence >= 0 AND confidence <= 1),

    status                 VARCHAR(20)  NOT NULL DEFAULT 'active'
                           CHECK (status IN ('active', 'superseded', 'archived')),

    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- If source_message_id is set, source_conversation_id must also be set.
    CONSTRAINT memories_source_ref_check
        CHECK (
            source_message_id IS NULL
            OR source_conversation_id IS NOT NULL
        ),

    -- Project-scoped memories: enforce same-user ownership.
    -- Skipped automatically by PostgreSQL when project_id IS NULL,
    -- so global memories are never affected by cascading project deletes.
    CONSTRAINT memories_project_ownership_fk
        FOREIGN KEY (project_id, user_id)
        REFERENCES projects(id, user_id)
        ON DELETE CASCADE,

    -- Validate source message belongs to source conversation.
    -- Skipped when source_message_id IS NULL (no specific message recorded).
    CONSTRAINT memories_source_message_fk
        FOREIGN KEY (source_message_id, source_conversation_id)
        REFERENCES messages(id, conversation_id)
        ON DELETE SET NULL
);

COMMENT ON TABLE  memories                      IS 'Extracted, persistent memories for a user, optionally scoped to a project.';
COMMENT ON COLUMN memories.project_id           IS 'NULL = global memory; UUID = project-scoped memory.';
COMMENT ON COLUMN memories.source_conversation_id IS 'Conversation from which this memory was extracted. NULL if unknown.';
COMMENT ON COLUMN memories.source_message_id    IS 'Specific message that triggered extraction. Must belong to source_conversation_id.';
COMMENT ON COLUMN memories.memory_type          IS 'semantic | episodic | procedural';
COMMENT ON COLUMN memories.embedding            IS '384-dim vector from all-MiniLM-L6-v2. NULL until generated.';
COMMENT ON COLUMN memories.importance           IS 'Subjective relevance score [0, 1]. Higher = more important.';
COMMENT ON COLUMN memories.confidence           IS 'Extraction confidence [0, 1]. Lower = uncertain or inferred.';
COMMENT ON COLUMN memories.status               IS 'active | superseded | archived';

-- Lookup indexes
CREATE INDEX IF NOT EXISTS memories_user_id_idx             ON memories (user_id);
CREATE INDEX IF NOT EXISTS memories_project_id_idx          ON memories (project_id) WHERE project_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS memories_source_conversation_idx  ON memories (source_conversation_id) WHERE source_conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS memories_memory_type_idx         ON memories (memory_type);
CREATE INDEX IF NOT EXISTS memories_status_idx              ON memories (status);

-- HNSW index for approximate nearest-neighbour cosine similarity search.
-- Built at schema creation time; HNSW (unlike IVFFlat) does not require
-- pre-existing rows and remains accurate as new embeddings are inserted.
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
    ON memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


COMMIT;
