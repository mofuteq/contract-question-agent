"""Pydantic schemas for v0.2 verification-question generation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

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


class SelectedReviewLens(_StrictModel):
    label: str
    source: str
    reason: str


class VerificationQuestionOutput(_StrictModel):
    contract_id: str
    clause_type: str
    evidence_text: str
    selected_review_lenses: list[SelectedReviewLens] = Field(default_factory=list)
    unknowns: list[str]
    decision_risks: list[str]
    legal_review_questions: list[LegalReviewQuestion]
    verification_questions: list[VerificationQuestion]
    suggested_next_step: str
    safety_disclaimer: str
    safety_status: str
    safety_warnings: list[str]
    model_name: str


class ReflectionViolation(_StrictModel):
    thesis: str
    problem: str
    rewrite_guidance: str


class ReflectionResult(_StrictModel):
    status: Literal["passed", "failed"]
    violations: list[ReflectionViolation] = Field(default_factory=list)
    regeneration_guidance: str = ""


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
    regeneration_count: NonNegativeInt = 0
    regeneration_guidance: str = ""


class GeneratedQuestions(_StrictModel):
    request: GenerateQuestionsRequest
    records: list[ClauseSpanRecord] = Field(default_factory=list)
    outputs: list[VerificationQuestionOutput]
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_generated: NonNegativeInt
    regeneration_count: NonNegativeInt = 0
    regeneration_guidance: str = ""


class ReflectedQuestions(_StrictModel):
    request: GenerateQuestionsRequest
    records: list[ClauseSpanRecord]
    outputs: list[VerificationQuestionOutput]
    reflection_results: list[ReflectionResult]
    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_generated: NonNegativeInt
    regeneration_count: NonNegativeInt = 0
    regeneration_guidance: str = ""
    regeneration_requested: bool = False


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

    async def generate(
        self,
        record: ClauseSpanRecord,
        *,
        regeneration_guidance: str | None = None,
    ) -> VerificationQuestionOutput:
        """Generate one structured output for one CUAD clause span."""

    async def reflect(self, output: VerificationQuestionOutput) -> ReflectionResult:
        """Evaluate one structured output against the skill thesis."""
