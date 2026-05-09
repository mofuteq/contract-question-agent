"""Rule-based safety checks for generated verification-question outputs."""

from __future__ import annotations

from typing import Any

from contract_question_agent.schemas import VerificationQuestionOutput


SAFETY_DISCLAIMER = (
    "This output is for decision-support only and does not provide legal advice. "
    "Discuss these questions with a qualified professional before relying on them."
)

BANNED_PHRASES: tuple[str, ...] = (
    "you should sign",
    "you should not sign",
    "this is illegal",
    "this is legal",
    "this clause is enforceable",
    "this clause is unenforceable",
    "you are legally required",
    "this guarantees",
)


def find_banned_phrases(value: Any) -> list[str]:
    """Return banned phrases found anywhere in a JSON-like value."""
    text = _flatten_text(value).lower()
    return [phrase for phrase in BANNED_PHRASES if phrase in text]


def apply_safety_check(output: VerificationQuestionOutput) -> VerificationQuestionOutput:
    """Annotate a structured output with deterministic safety status."""
    payload = output.model_dump(exclude={"evidence_text"})
    found = find_banned_phrases(payload)
    warnings = [f"banned phrase found: {phrase}" for phrase in found]
    return output.model_copy(
        update={
            "safety_disclaimer": SAFETY_DISCLAIMER,
            "safety_status": "failed" if warnings else "passed",
            "safety_warnings": warnings,
        }
    )


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return "\n".join(_flatten_text(item) for item in value)
    return ""
