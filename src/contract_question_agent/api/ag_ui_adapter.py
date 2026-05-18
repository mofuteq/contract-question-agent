"""Safe AG-UI snapshots derived from API workflow responses."""

from __future__ import annotations

from typing import Any

from contract_question_agent.api.schemas import RunResponse
from contract_question_agent.schemas import VerificationQuestionOutput


def sanitize_verification_questions(
    outputs: list[VerificationQuestionOutput],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in outputs:
        data = output.model_dump(mode="json")
        data.pop("evidence_text", None)
        rows.append(data)
    return rows


def run_response_to_snapshot(response: RunResponse) -> dict[str, Any]:
    return {
        "run_id": response.run_id,
        "created_at": response.created_at,
        "rows_read": response.rows_read,
        "rows_filtered": response.rows_filtered,
        "rows_in_scope": response.rows_in_scope,
        "rows_out_of_scope": response.rows_out_of_scope,
        "rows_generated": response.rows_generated,
        "rows_written": response.rows_written,
        "scope_status_counts": response.scope_status_counts,
        "out_of_scope_reasons": response.out_of_scope_reasons,
        "selected_review_lenses": response.selected_review_lenses,
        "safety_failed_count": response.safety_failed_count,
        "safety_status": response.safety_status,
        "verification_questions": sanitize_verification_questions(
            response.verification_questions
        ),
        "model_name": response.model_name,
        "dry_run": response.dry_run,
        "output_path": str(response.output_path),
        "metadata_path": str(response.metadata_path),
        "log_path": str(response.log_path),
    }
