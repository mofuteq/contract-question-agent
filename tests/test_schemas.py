from __future__ import annotations

import pytest
from pydantic import ValidationError

from contract_question_agent.schemas import (
    LegalReviewQuestion,
    ReflectionResult,
    ReflectionViolation,
    SelectedReviewLens,
    VerificationQuestion,
    VerificationQuestionOutput,
)


def test_verification_question_output_validates_minimal_shape():
    output = VerificationQuestionOutput(
        contract_id="C1",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete for one year.",
        unknowns=["Scope of restricted business is unclear."],
        decision_risks=["May affect post-contract options."],
        legal_review_questions=[
            LegalReviewQuestion(question="What law applies?", reason="Law affects review.")
        ],
        verification_questions=[
            VerificationQuestion(
                question="Which activities are covered?",
                why_it_matters="The scope affects practical impact.",
            )
        ],
        suggested_next_step="Discuss with a qualified professional.",
        safety_disclaimer="disclaimer",
        safety_status="passed",
        safety_warnings=[],
        model_name="fake-model",
    )

    assert output.contract_id == "C1"
    assert output.selected_review_lenses == []


def test_verification_question_output_accepts_selected_review_lenses():
    output = VerificationQuestionOutput(
        contract_id="C1",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete for one year.",
        selected_review_lenses=[
            SelectedReviewLens(
                label="Time period",
                source="mcp_clause_review_hints",
                reason="The clause states a one-year restriction.",
            )
        ],
        unknowns=[],
        decision_risks=[],
        legal_review_questions=[],
        verification_questions=[],
        suggested_next_step="Discuss with a qualified professional.",
        safety_disclaimer="disclaimer",
        safety_status="passed",
        safety_warnings=[],
        model_name="fake-model",
    )

    assert output.selected_review_lenses[0].label == "Time period"
    assert output.selected_review_lenses[0].source == "mcp_clause_review_hints"


def test_reflection_schema_accepts_passed_output():
    result = ReflectionResult(status="passed")

    assert result.status == "passed"
    assert result.violations == []
    assert result.regeneration_guidance == ""


def test_reflection_schema_accepts_failed_output():
    result = ReflectionResult(
        status="failed",
        violations=[
            ReflectionViolation(
                thesis="generate verification questions, not legal answers",
                problem="The output answers the issue.",
                rewrite_guidance="Rewrite as questions for professional review.",
            )
        ],
        regeneration_guidance="Ask questions instead of answering the legal issue.",
    )

    assert result.status == "failed"
    assert result.violations[0].thesis == (
        "generate verification questions, not legal answers"
    )


def test_verification_question_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        VerificationQuestionOutput(
            contract_id="C1",
            clause_type="Non-Compete",
            evidence_text="text",
            unknowns=[],
            decision_risks=[],
            legal_review_questions=[],
            verification_questions=[],
            suggested_next_step="next",
            safety_disclaimer="disclaimer",
            safety_status="passed",
            safety_warnings=[],
            model_name="fake-model",
            behavioral_lenses=[],
        )
