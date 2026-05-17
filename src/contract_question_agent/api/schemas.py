"""HTTP schemas for the FastAPI workflow adapter."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from contract_question_agent.cli_generate_questions import DEFAULT_OUTPUT_DIR
from contract_question_agent.schemas import VerificationQuestionOutput


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_ApiModel):
    status: str = "ok"


class GenerateVerificationQuestionsRequest(_ApiModel):
    input_path: Path = Field(
        ...,
        description="Local JSONL path containing CUAD clause span records.",
    )
    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR,
        description="Local parent directory for this workflow run's artifacts.",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional deterministic run id. Generated when omitted.",
    )
    clause_type: str | None = None
    contract_id: str | None = None
    limit: NonNegativeInt | None = None
    offset: NonNegativeInt = 0
    model: str | None = Field(
        default=None,
        description="Optional OpenRouter model override.",
    )
    dry_run: bool = Field(
        default=False,
        description="Use the deterministic offline model client.",
    )


class GenerateVerificationQuestionsResponse(_ApiModel):
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
    rows_in_scope: NonNegativeInt = 0
    rows_out_of_scope: NonNegativeInt = 0
    rows_generated: NonNegativeInt
    scope_status_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    out_of_scope_reasons: dict[str, NonNegativeInt] = Field(default_factory=dict)
    safety_failed_count: NonNegativeInt
    rows_written: NonNegativeInt
    outputs: list[VerificationQuestionOutput] = Field(default_factory=list)
