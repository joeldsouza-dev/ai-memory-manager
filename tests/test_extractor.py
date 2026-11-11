"""
tests/test_extractor.py
-----------------------
Tests for Memory Extractor V1:
1. Multiple memory candidates.
2. Zero memory candidates.
3. All three memory types (semantic, episodic, procedural).
4. Temporary instruction vs. persistent preference.
5. Global versus project scope resolution.
6. Ambiguous preference clarification handling.
7. Invalid JSON response raises ExtractionParseError.
8. Invalid memory type raises ExtractionValidationError.
9. Confidence and importance out of range raise ExtractionValidationError.
10. Provider API failure raises LLMAPIError.
11. Integration with background job processor (process_job).
"""

from __future__ import annotations

import json
import uuid

import psycopg
import pytest

from memory.extractor import (
    ExtractionParseError,
    ExtractionValidationError,
    MemoryExtractor,
)
from memory.jobs import claim_job, create_job, process_job
from memory.models import Message
from memory.providers import LLMAPIError, MockLLMProvider
from tests.conftest import make_conversation, make_message, make_project, make_user


# ---------------------------------------------------------------------------
# Test 1: Multiple memory candidates
# ---------------------------------------------------------------------------


def test_extract_multiple_memory_candidates() -> None:
    canned = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "User prefers Python for coding.",
                "scope": "global",
                "confidence": 0.95,
                "importance": 0.8,
            },
            {
                "memory_type": "procedural",
                "content": "User prefers hints before complete solutions.",
                "scope": "global",
                "confidence": 0.9,
                "importance": 0.7,
            },
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="I prefer Python for coding and I like hints before complete solutions."
    )

    assert len(result.memories) == 2
    assert result.clarification_required is False
    assert result.memories[0].memory_type == "semantic"
    assert result.memories[0].content == "User prefers Python for coding."
    assert result.memories[1].memory_type == "procedural"
    assert result.memories[1].content == "User prefers hints before complete solutions."


# ---------------------------------------------------------------------------
# Test 2: Zero memory candidates
# ---------------------------------------------------------------------------


def test_extract_zero_memory_candidates() -> None:
    canned = {
        "memories": [],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(message="Hello, how is the weather today?")

    assert result.memories == []
    assert result.clarification_required is False
    assert result.clarification_question is None


# ---------------------------------------------------------------------------
# Test 3: All three memory types (semantic, episodic, procedural)
# ---------------------------------------------------------------------------


def test_extract_all_three_memory_types() -> None:
    canned = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "User is a backend engineer specializing in distributed systems.",
                "scope": "global",
                "confidence": 0.95,
                "importance": 0.85,
            },
            {
                "memory_type": "episodic",
                "content": "User completed the database migration to PostgreSQL 17 on Monday.",
                "scope": "global",
                "confidence": 0.9,
                "importance": 0.6,
            },
            {
                "memory_type": "procedural",
                "content": "Assistant must provide type annotations in all code snippets.",
                "scope": "global",
                "confidence": 0.95,
                "importance": 0.75,
            },
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="I am a backend engineer. On Monday I migrated to Postgres 17. Always include type annotations in code."
    )

    types = [m.memory_type for m in result.memories]
    assert types == ["semantic", "episodic", "procedural"]


# ---------------------------------------------------------------------------
# Test 4: Temporary instruction vs persistent preference
# ---------------------------------------------------------------------------


def test_temporary_instruction_vs_persistent_preference() -> None:
    # Scenario A: Temporary task instruction -> no persistent memory
    canned_temp = {
        "memories": [],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider_temp = MockLLMProvider(responses=[json.dumps(canned_temp)])
    extractor_temp = MemoryExtractor(provider=provider_temp)

    result_temp = extractor_temp.extract(message="Give this example in Java.")
    assert len(result_temp.memories) == 0

    # Scenario B: Persistent preference -> semantic memory
    canned_persistent = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "User prefers Python for coding.",
                "scope": "global",
                "confidence": 0.95,
                "importance": 0.8,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider_perm = MockLLMProvider(responses=[json.dumps(canned_persistent)])
    extractor_perm = MemoryExtractor(provider=provider_perm)

    result_perm = extractor_perm.extract(message="I prefer Python for coding.")
    assert len(result_perm.memories) == 1
    assert result_perm.memories[0].content == "User prefers Python for coding."


# ---------------------------------------------------------------------------
# Test 5: Global versus project scope resolution
# ---------------------------------------------------------------------------


def test_global_vs_project_scope() -> None:
    active_project_id = uuid.uuid4()
    canned = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "User prefers dark mode editor theme.",
                "scope": "global",
                "confidence": 0.95,
                "importance": 0.5,
            },
            {
                "memory_type": "semantic",
                "content": "For this DSA project, always use Java.",
                "scope": "project",
                "confidence": 0.95,
                "importance": 0.85,
            },
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="I prefer dark mode. For this DSA project, always use Java.",
        project_id=active_project_id,
    )

    assert result.memories[0].scope == "global"
    assert result.memories[1].scope == "project"

    # Test resolution through job extractor callable
    job_extractor = extractor.create_job_extractor(conn=None)
    dummy_msg = Message(
        id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        role="user",
        content="test",
        created_at=None,  # type: ignore
    )

    # When no DB connection provided, global remains None
    provider.queue_response(json.dumps(canned))
    persisted = job_extractor(dummy_msg)
    assert len(persisted) == 2
    assert persisted[0].project_id is None
    assert persisted[1].project_id is None  # dummy msg has no conv row to resolve


# ---------------------------------------------------------------------------
# Test 6: Ambiguous preference clarification
# ---------------------------------------------------------------------------


def test_ambiguous_preference_clarification() -> None:
    canned = {
        "memories": [],
        "clarification_required": True,
        "clarification_question": "You previously preferred Python. Would you like to switch your primary language to Rust or keep Python as default?",
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="Actually, maybe I should switch everything to Rust instead of Python?",
        existing_summaries="User previously stated: prefers Python for all coding tasks.",
    )

    assert result.clarification_required is True
    assert result.clarification_question is not None
    assert "Rust" in result.clarification_question
    assert result.memories == []

    # Verify that job extractor produces 0 candidates when clarification is required
    provider.queue_response(json.dumps(canned))
    dummy_msg = Message(
        id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        role="user",
        content="switch to Rust?",
        created_at=None,  # type: ignore
    )
    job_ext = extractor.create_job_extractor()
    assert job_ext(dummy_msg) == []


# ---------------------------------------------------------------------------
# Test 7: Invalid JSON raises ExtractionParseError
# ---------------------------------------------------------------------------


def test_invalid_json_raises_extraction_parse_error() -> None:
    provider = MockLLMProvider(responses=["{ not valid json at all ..."])
    extractor = MemoryExtractor(provider=provider)

    with pytest.raises(ExtractionParseError, match="invalid JSON"):
        extractor.extract(message="Any message")


# ---------------------------------------------------------------------------
# Test 8: Invalid memory type raises ExtractionValidationError
# ---------------------------------------------------------------------------


def test_invalid_memory_type_raises_validation_error() -> None:
    canned = {
        "memories": [
            {
                "memory_type": "short_term",  # Invalid! Only semantic, episodic, procedural allowed
                "content": "Some fact",
                "scope": "global",
                "confidence": 0.9,
                "importance": 0.5,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    with pytest.raises(ExtractionValidationError, match="Input should be 'semantic', 'episodic' or 'procedural'"):
        extractor.extract(message="Testing invalid type")


# ---------------------------------------------------------------------------
# Test 9: Confidence and importance out of range raise ExtractionValidationError
# ---------------------------------------------------------------------------


def test_confidence_and_importance_out_of_range() -> None:
    # Confidence > 1.0
    canned_high_conf = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "Fact",
                "scope": "global",
                "confidence": 1.5,
                "importance": 0.5,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned_high_conf)])
    extractor = MemoryExtractor(provider=provider)

    with pytest.raises(ExtractionValidationError, match="less than or equal to 1"):
        extractor.extract(message="Test out of range")

    # Importance < 0.0
    canned_low_imp = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "Fact",
                "scope": "global",
                "confidence": 0.8,
                "importance": -0.2,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider.queue_response(json.dumps(canned_low_imp))

    with pytest.raises(ExtractionValidationError, match="greater than or equal to 0"):
        extractor.extract(message="Test negative importance")


# ---------------------------------------------------------------------------
# Test 10: API failure raises LLMAPIError
# ---------------------------------------------------------------------------


def test_api_failure_raises_llm_api_error() -> None:
    provider = MockLLMProvider(
        responses=[LLMAPIError("503 Service Unavailable: Rate limited or server busy")]
    )
    extractor = MemoryExtractor(provider=provider)

    with pytest.raises(LLMAPIError, match="503 Service Unavailable"):
        extractor.extract(message="Test API failure")


# ---------------------------------------------------------------------------
# Test 11: End-to-end integration with background job processor
# ---------------------------------------------------------------------------


def test_integration_with_background_job_processor(conn: psycopg.Connection) -> None:
    """Verifies that MemoryExtractor integrates seamlessly with process_job()
    and persists memories to the PostgreSQL database with proper project scoping."""
    user_id = make_user(conn, "Extractor Integration User")
    project_id = make_project(conn, user_id, "DSA Project")
    conv_id = make_conversation(conn, user_id, project_id=project_id)
    msg_id = make_message(
        conn,
        conv_id,
        "For this DSA project, always use Java. Also, I solved Two Sum yesterday.",
    )

    job = create_job(conn, msg_id, user_id)
    conn.commit()

    canned_llm_output = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "For this DSA project, always use Java.",
                "scope": "project",
                "confidence": 0.95,
                "importance": 0.9,
            },
            {
                "memory_type": "episodic",
                "content": "User solved Two Sum yesterday.",
                "scope": "global",
                "confidence": 0.9,
                "importance": 0.5,
            },
        ],
        "clarification_required": False,
        "clarification_question": None,
    }

    provider = MockLLMProvider(responses=[json.dumps(canned_llm_output)])
    extractor = MemoryExtractor(provider=provider)

    # Claim job
    claimed_job = claim_job(conn, job_id=job.id)
    assert claimed_job is not None

    # Process job using injected extractor
    job_extractor_fn = extractor.create_job_extractor(conn=conn)
    persisted_ids = process_job(
        job=claimed_job,
        extractor=job_extractor_fn,
        conn=conn,
    )

    assert len(persisted_ids) == 2

    # Verify rows in database
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, memory_type, project_id
            FROM memories
            WHERE id = ANY(%s)
            ORDER BY memory_type
            """,
            ([str(m) for m in persisted_ids],),
        )
        rows = cur.fetchall()

    # episodic memory is global -> project_id is None
    assert rows[0][0] == "User solved Two Sum yesterday."
    assert rows[0][1] == "episodic"
    assert rows[0][2] is None

    # semantic memory is project-scoped -> project_id matches DSA Project
    assert rows[1][0] == "For this DSA project, always use Java."
    assert rows[1][1] == "semantic"
    assert rows[1][2] == project_id

    # Verify job is completed
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM memory_jobs WHERE id = %s", (str(job.id),))
        assert cur.fetchone()[0] == "completed"


# ===========================================================================
# Targeted Refinement Scenarios
# ===========================================================================


def test_scenario_global_python_request_java_for_one_solution() -> None:
    """Scenario 1: Existing global preference is Python; user requests Java for one solution.
    Expected: No persistent memory stored, no clarification asked."""
    canned = {
        "memories": [],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="Can you give me this solution in Java?",
        existing_summaries=["User prefers Python for general programming."],
    )

    assert result.memories == []
    assert result.clarification_required is False
    assert result.clarification_question is None


def test_scenario_user_requests_cpp_for_one_example() -> None:
    """Scenario 2: User requests C++ for one example.
    Expected: No persistent memory stored."""
    canned = {
        "memories": [],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="Explain this concept using C++ for this example."
    )

    assert result.memories == []
    assert result.clarification_required is False


def test_scenario_from_now_on_use_java() -> None:
    """Scenario 3: User says 'From now on, use Java'.
    Expected: Global semantic memory."""
    canned = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "From now on, use Java for all coding examples.",
                "scope": "global",
                "confidence": 0.95,
                "importance": 0.85,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="From now on, use Java for all my coding examples."
    )

    assert len(result.memories) == 1
    assert result.memories[0].memory_type == "semantic"
    assert result.memories[0].scope == "global"
    assert "Java" in result.memories[0].content


def test_scenario_for_this_dsa_project_use_cpp() -> None:
    """Scenario 4: User says 'For this DSA project, use C++'.
    Expected: Project-scoped semantic memory."""
    dsa_project_id = uuid.uuid4()
    canned = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "For this DSA project, always use C++.",
                "scope": "project",
                "confidence": 0.95,
                "importance": 0.9,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="For this DSA project, always use C++.",
        project_id=dsa_project_id,
    )

    assert len(result.memories) == 1
    assert result.memories[0].memory_type == "semantic"
    assert result.memories[0].scope == "project"


def test_scenario_i_think_java_is_better_clarification() -> None:
    """Scenario 5: User says 'I think Java is better' or 'Maybe I should switch to Java'.
    Expected: Clarification required, no persistent memory update."""
    canned = {
        "memories": [],
        "clarification_required": True,
        "clarification_question": "You previously had a preference for Python. Are you considering switching your default language to Java, or is this just an observation?",
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="I think Java is better.",
        existing_summaries=["User generally prefers Python for programming."],
    )

    assert result.clarification_required is True
    assert result.clarification_question is not None
    assert result.memories == []


def test_scenario_normal_technical_question_no_memory() -> None:
    """Scenario 6: User asks a normal technical question.
    Expected: No unnecessary memory created."""
    canned = {
        "memories": [],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider = MockLLMProvider(responses=[json.dumps(canned)])
    extractor = MemoryExtractor(provider=provider)

    result = extractor.extract(
        message="What is the time complexity of quicksort in the worst case vs average case?"
    )

    assert result.memories == []
    assert result.clarification_required is False


def test_scenario_project_with_global_preference() -> None:
    """Scenario 7: User in a project with a global Python preference:
    - Ad-hoc Java request -> No memory.
    - Explicit persistent project preference -> Project-scoped memory only."""
    active_project_id = uuid.uuid4()

    # Part A: Ad-hoc Java request in project
    canned_adhoc = {
        "memories": [],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider_a = MockLLMProvider(responses=[json.dumps(canned_adhoc)])
    extractor_a = MemoryExtractor(provider=provider_a)

    result_a = extractor_a.extract(
        message="Can you write this helper function in Java?",
        project_id=active_project_id,
        existing_summaries=["User prefers Python for general programming."],
    )
    assert result_a.memories == []
    assert result_a.clarification_required is False

    # Part B: Explicit persistent project preference
    canned_project = {
        "memories": [
            {
                "memory_type": "semantic",
                "content": "For this backend project, use Java for all implementations.",
                "scope": "project",
                "confidence": 0.95,
                "importance": 0.85,
            }
        ],
        "clarification_required": False,
        "clarification_question": None,
    }
    provider_b = MockLLMProvider(responses=[json.dumps(canned_project)])
    extractor_b = MemoryExtractor(provider=provider_b)

    result_b = extractor_b.extract(
        message="For this backend project, use Java for all implementations.",
        project_id=active_project_id,
        existing_summaries=["User prefers Python for general programming."],
    )
    assert len(result_b.memories) == 1
    assert result_b.memories[0].scope == "project"
    assert result_b.memories[0].memory_type == "semantic"


def test_temporary_instructions_never_generate_memory_candidates() -> None:
    """Test that various temporary instructions never produce MemoryCandidate objects
    when passed through create_job_extractor."""
    temporary_phrases = [
        "Give me this solution in Java.",
        "Explain this concept using C++.",
        "For this example, use JavaScript.",
        "Can you show me how to implement this in Go?",
        "Fix line 10 in Python.",
        "Show a quick snippet in Rust.",
    ]

    for phrase in temporary_phrases:
        canned = {
            "memories": [],
            "clarification_required": False,
            "clarification_question": None,
        }
        provider = MockLLMProvider(responses=[json.dumps(canned)])
        extractor = MemoryExtractor(provider=provider)
        job_ext = extractor.create_job_extractor(conn=None)

        msg = Message(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="user",
            content=phrase,
            created_at=None,  # type: ignore
        )

        candidates = job_ext(msg)
        assert candidates == [], f"Phrase '{phrase}' produced unexpected memory candidate!"


