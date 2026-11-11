"""
memory/schemas.py
-----------------
Pydantic schemas for LLM-based memory extraction output validation.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class MemoryCandidate(BaseModel):
    """Structured memory candidate extracted from a conversation message."""

    model_config = ConfigDict(extra="forbid")

    memory_type: Literal["semantic", "episodic", "procedural"] = Field(
        description="Type of memory: semantic (facts/preferences), episodic (events/experiences), procedural (assistant behavior instructions)."
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Clear, non-empty distilled memory text.",
    )
    scope: Literal["global", "project"] = Field(
        default="global",
        description="Memory scope: global across all projects, or project-specific.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score in the extraction accuracy (0.0 to 1.0).",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated long-term importance/relevance score (0.0 to 1.0).",
    )


class ExtractionResult(BaseModel):
    """Complete structured output from the LLM memory extraction pipeline."""

    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryCandidate] = Field(
        default_factory=list,
        description="List of extracted memory candidates. Empty if no persistent memory exists.",
    )
    clarification_required: bool = Field(
        default=False,
        description="Set to true if user preference change is ambiguous or conflicting.",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Clarification question to present to the user if clarification_required is true.",
    )

