"""Metadata-safe reflection violation serialization."""

from __future__ import annotations

from typing import Any

from contract_question_agent.schemas import ReflectionViolation

REFLECTION_METADATA_FIELD_MAX_LENGTH = 500
REFLECTION_METADATA_TRUNCATION_SUFFIX = "... [truncated]"
_TRUNCATED_REFLECTION_VIOLATION_FIELDS = (
    "thesis",
    "problem",
    "rewrite_guidance",
)


def reflection_violation_metadata(violation: ReflectionViolation) -> dict[str, Any]:
    payload = violation.model_dump(mode="json")
    for field in _TRUNCATED_REFLECTION_VIOLATION_FIELDS:
        payload[field] = _truncate_metadata_string(payload[field])
    return payload


def _truncate_metadata_string(value: str) -> str:
    if len(value) <= REFLECTION_METADATA_FIELD_MAX_LENGTH:
        return value
    content_length = (
        REFLECTION_METADATA_FIELD_MAX_LENGTH
        - len(REFLECTION_METADATA_TRUNCATION_SUFFIX)
    )
    return f"{value[:content_length]}{REFLECTION_METADATA_TRUNCATION_SUFFIX}"
