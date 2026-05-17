"""HTTP schemas for the FastAPI workflow adapter."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from contract_question_agent.schemas import VerificationQuestionOutput


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_ApiModel):
    status: str = "ok"


class RunRequest(_ApiModel):
    contract_id: str | None = Field(
        default=None,
        description="Optional contract identifier. Generated run id is used when omitted.",
    )
    clause_type: str = Field(
        ...,
        description="Clause type for this single clause payload.",
    )
    evidence_text: str = Field(
        ...,
        description="Clause evidence text to generate verification questions from.",
    )
    mcp_hints_enabled: bool = Field(
        default=True,
        description=(
            "Reserved API flag for enabling deterministic MCP hints. "
            "The current workflow configuration remains authoritative."
        ),
    )
    model_name: str | None = Field(
        default=None,
        description="Optional model override.",
    )
    dry_run: bool = Field(
        default=False,
        description="Use the deterministic offline model client.",
    )


class RunResponse(_ApiModel):
    run_id: str
    created_at: str

    rows_read: NonNegativeInt
    rows_filtered: NonNegativeInt
    rows_in_scope: NonNegativeInt
    rows_out_of_scope: NonNegativeInt
    rows_generated: NonNegativeInt
    rows_written: NonNegativeInt

    scope_status_counts: dict[str, NonNegativeInt] = Field(default_factory=dict)
    out_of_scope_reasons: dict[str, NonNegativeInt] = Field(default_factory=dict)

    selected_review_lenses: list[dict] = Field(default_factory=list)

    safety_failed_count: NonNegativeInt
    safety_status: str | None = None

    verification_questions: list[VerificationQuestionOutput] = Field(
        default_factory=list
    )

    model_name: str
    dry_run: bool

    output_path: Path
    metadata_path: Path
    log_path: Path
