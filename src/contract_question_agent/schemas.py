"""Pydantic schemas for v0.2 verification-question generation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from contract_question_agent.cuad_loader import ClauseSpanRecord


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerificationQuestion(_StrictModel):
    question: str
    why_it_matters: str


class LegalReviewQuestion(_StrictModel):
    question: str
    reason: str


class VerificationQuestionOutput(_StrictModel):
    contract_id: str
    clause_type: str
    evidence_text: str
    unknowns: list[str]
    decision_risks: list[str]
    legal_review_questions: list[LegalReviewQuestion]
    verification_questions: list[VerificationQuestion]
    suggested_next_step: str
    safety_disclaimer: str
    safety_status: str
    safety_warnings: list[str]
    model_name: str


class GenerateQuestionsRequest(_StrictModel):
    input_path: Path
    output_path: Path
    metadata_path: Path
    log_path: Path
    run_id: str
    created_at: str
    clause_type: str | None = None
    contract_id: str | None = None
    limit: NonNegativeInt | None = None
    offset: NonNegativeInt = 0
    model_name: str
    dry_run: bool = False


class LoadedClauseSpans(_StrictModel):
    request: GenerateQuestionsRequest
    records: list[ClauseSpanRecord]
    rows_read: NonNegativeInt


class FilteredClauseSpans(_StrictModel):
    request: GenerateQuestionsRequest
    records: list[ClauseSpanRecord]
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt


class GeneratedQuestions(_StrictModel):
    request: GenerateQuestionsRequest
    outputs: list[VerificationQuestionOutput]
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_generated: NonNegativeInt


class SafetyCheckedQuestions(_StrictModel):
    request: GenerateQuestionsRequest
    outputs: list[VerificationQuestionOutput]
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_generated: NonNegativeInt
    safety_failed_count: NonNegativeInt


class WrittenQuestions(_StrictModel):
    output_path: Path
    metadata_path: Path
    log_path: Path
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_generated: NonNegativeInt
    safety_failed_count: NonNegativeInt
    rows_written: int
    outputs: list[VerificationQuestionOutput] = Field(default_factory=list)


class RunMetadata(_StrictModel):
    run_id: str
    created_at: str
    input_path: Path
    output_path: Path
    metadata_path: Path
    log_path: Path
    clause_type: str | None
    contract_id: str | None
    limit: NonNegativeInt | None
    offset: NonNegativeInt
    model_name: str
    dry_run: bool
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_generated: NonNegativeInt
    safety_failed_count: NonNegativeInt
    rows_written: NonNegativeInt


class QuestionModelClient(Protocol):
    model_name: str

    async def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        """Generate one structured output for one CUAD clause span."""
