"""WRITE_OUTPUT node logic."""

from __future__ import annotations

import json
from pathlib import Path

from contract_question_agent import tracing
from contract_question_agent.schemas import (
    RunMetadata,
    SafetyCheckedQuestions,
    WrittenQuestions,
)


def write_output_node(state: SafetyCheckedQuestions) -> WrittenQuestions:
    write_verification_questions_jsonl(state.request.output_path, state.outputs)
    metadata = RunMetadata(
        run_id=state.request.run_id,
        created_at=state.request.created_at,
        input_path=state.request.input_path,
        output_path=state.request.output_path,
        metadata_path=state.request.metadata_path,
        log_path=state.request.log_path,
        clause_type=state.request.clause_type,
        contract_id=state.request.contract_id,
        limit=state.request.limit,
        offset=state.request.offset,
        model_name=state.request.model_name,
        dry_run=state.request.dry_run,
        rows_read=state.rows_read,
        rows_filtered=state.rows_filtered,
        rows_generated=state.rows_generated,
        safety_failed_count=state.safety_failed_count,
        rows_written=len(state.outputs),
        tracing_enabled=tracing.is_active(),
        langfuse_trace_id=tracing.get_current_trace_id(),
        langfuse_trace_url=tracing.get_current_trace_url(),
        langfuse_environment=tracing.get_tracing_environment(),
    )
    write_run_metadata_json(state.request.metadata_path, metadata)
    return WrittenQuestions(
        output_path=state.request.output_path,
        metadata_path=state.request.metadata_path,
        log_path=state.request.log_path,
        rows_read=state.rows_read,
        rows_filtered=state.rows_filtered,
        rows_generated=state.rows_generated,
        safety_failed_count=state.safety_failed_count,
        rows_written=len(state.outputs),
        outputs=state.outputs,
    )


def write_verification_questions_jsonl(path: Path, outputs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def write_run_metadata_json(path: Path, metadata: RunMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
