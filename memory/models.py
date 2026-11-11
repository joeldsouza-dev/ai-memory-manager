"""
memory/models.py
----------------
Plain dataclasses for every domain entity.

These are simple data-holders — no ORM magic, no business logic.
They are used as the return type from store functions and as the
input type for the job processor.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Domain entities (read from DB)
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: uuid.UUID
    name: str
    created_at: datetime


@dataclass
class Project:
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    instructions: Optional[str]
    created_at: datetime


@dataclass
class Conversation:
    id: uuid.UUID
    user_id: uuid.UUID
    project_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime


@dataclass
class Message:
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str          # 'user' | 'assistant' | 'system'
    content: str
    created_at: datetime


@dataclass
class Memory:
    id: uuid.UUID
    user_id: uuid.UUID
    project_id: Optional[uuid.UUID]
    source_conversation_id: Optional[uuid.UUID]
    source_message_id: Optional[uuid.UUID]
    memory_type: str   # 'semantic' | 'episodic' | 'procedural'
    category: Optional[str]
    content: str
    importance: float
    confidence: float
    status: str        # 'active' | 'superseded' | 'archived'
    created_at: datetime
    updated_at: datetime


@dataclass
class MemoryJob:
    id: uuid.UUID
    message_id: uuid.UUID
    conversation_id: uuid.UUID
    user_id: uuid.UUID
    status: str        # 'pending' | 'processing' | 'completed' | 'failed'
    lease_token: uuid.UUID
    locked_at: Optional[datetime]
    last_error: Optional[str]
    retry_count: int
    max_retries: int
    next_retry_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Input types for job processing
# ---------------------------------------------------------------------------


@dataclass
class MemoryCandidate:
    """A single memory candidate produced by the extractor.

    extraction_id can be assigned explicitly or left None to be computed
    deterministically from (job_id, memory_type, content, project_id).
    This guarantees idempotency regardless of the order in which candidates
    are returned by the extractor.
    """

    memory_type: str                         # 'semantic' | 'episodic' | 'procedural'
    content: str
    extraction_id: Optional[uuid.UUID] = None  # deterministic candidate key
    category: Optional[str] = None
    project_id: Optional[uuid.UUID] = None    # None = global; UUID = project-scoped
    importance: float = 0.5
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TerminalError(Exception):
    """Terminal failure that should not be retried."""
    pass


class RetryableError(Exception):
    """Explicitly retryable failure."""
    pass


class FencingTokenError(Exception):
    """Raised when worker lease token is expired or reclaimed by another worker."""
    pass


class ProjectOwnershipError(TerminalError):
    """Raised when project does not exist or is not owned by the specified user."""
    pass
