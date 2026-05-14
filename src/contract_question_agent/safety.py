"""Rule-based safety checks for generated verification-question outputs."""

from __future__ import annotations

import unicodedata
from typing import Any

from contract_question_agent.schemas import (
    LegalReviewQuestion,
    SelectedReviewLens,
    VerificationQuestion,
    VerificationQuestionOutput,
)


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
    output = normalize_plain_string_fields(output)
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


def normalize_plain_string_fields(
    output: VerificationQuestionOutput,
) -> VerificationQuestionOutput:
    """Apply NFKC normalization and remove leading Markdown markers."""
    return output.model_copy(
        update={
            "selected_review_lenses": [
                SelectedReviewLens(
                    label=_normalize_plain_string(item.label),
                    source=item.source,
                    reason=_normalize_plain_string(item.reason),
                )
                for item in output.selected_review_lenses
            ],
            "unknowns": [_normalize_plain_string(item) for item in output.unknowns],
            "decision_risks": [
                _normalize_plain_string(item) for item in output.decision_risks
            ],
            "legal_review_questions": [
                LegalReviewQuestion(
                    question=item.question,
                    reason=_normalize_plain_string(item.reason),
                )
                for item in output.legal_review_questions
            ],
            "verification_questions": [
                VerificationQuestion(
                    question=item.question,
                    why_it_matters=_normalize_plain_string(item.why_it_matters),
                )
                for item in output.verification_questions
            ],
        }
    )


def _normalize_plain_string(value: str) -> str:
    stripped = unicodedata.normalize("NFKC", value).lstrip()
    while stripped[:1] in {">", "-", "*"}:
        stripped = stripped[1:].lstrip()
    return stripped


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return "\n".join(_flatten_text(item) for item in value)
    return ""
