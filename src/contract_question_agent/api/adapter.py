"""Thin HTTP-to-workflow adapter.

FastAPI owns only transport concerns here. LangGraph still owns workflow state
transitions, and the configured model client still owns node-internal model
calls and deterministic MCP context retrieval.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from contract_question_agent.api.schemas import (
    RunRequest,
    RunResponse,
)
from contract_question_agent.cli_generate_questions import (
    DryRunQuestionClient,
    LOG_FILENAME,
    METADATA_FILENAME,
    OUTPUT_FILENAME,
)
from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterQuestionClient,
)
from contract_question_agent.schemas import GenerateQuestionsRequest, WrittenQuestions
from contract_question_agent.workflows import run_workflow_async

logger = logging.getLogger(__name__)

API_OUTPUT_DIR = Path("data/cuad/api-runs")
INPUT_FILENAME = "input_clause_spans.jsonl"


def make_api_run_id() -> str:
    return str(uuid4())


async def run_workflow_from_api_request(
    api_request: RunRequest,
    *,
    run_id: str | None = None,
) -> RunResponse:
    """Adapt one HTTP clause payload into the existing JSONL workflow."""
    run_id = run_id or make_api_run_id()
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run_dir = API_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    input_path = run_dir / INPUT_FILENAME
    output_path = run_dir / OUTPUT_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    log_path = run_dir / LOG_FILENAME

    record = ClauseSpanRecord(
        contract_id=api_request.contract_id or run_id,
        source_file="api-request",
        clause_type=api_request.clause_type,
        evidence_text=api_request.evidence_text,
        start_char=0,
        end_char=len(api_request.evidence_text),
        label_present=True,
    )
    input_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    model_name = (
        api_request.model_name
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    )
    model_client = _build_model_client(model_name, dry_run=api_request.dry_run)
    workflow_request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id=run_id,
        created_at=created_at,
        clause_type=api_request.clause_type,
        contract_id=api_request.contract_id or run_id,
        limit=1,
        offset=0,
        model_name=model_name,
        dry_run=api_request.dry_run,
    )

    with _run_file_logging(log_path):
        _log_run_start(workflow_request)
        result = await run_workflow_async(workflow_request, model_client=model_client)
        _log_run_result(result)

    selected_review_lenses = [
        lens.model_dump()
        for output in result.outputs
        for lens in output.selected_review_lenses
    ]
    safety_status = result.outputs[0].safety_status if result.outputs else None

    return RunResponse(
        run_id=run_id,
        created_at=created_at,
        rows_read=result.rows_read,
        rows_filtered=result.rows_filtered,
        rows_in_scope=result.rows_in_scope,
        rows_out_of_scope=result.rows_out_of_scope,
        rows_generated=result.rows_generated,
        rows_written=result.rows_written,
        scope_status_counts=result.scope_status_counts,
        out_of_scope_reasons=result.out_of_scope_reasons,
        selected_review_lenses=selected_review_lenses,
        safety_failed_count=result.safety_failed_count,
        safety_status=safety_status,
        verification_questions=result.outputs,
        model_name=workflow_request.model_name,
        dry_run=workflow_request.dry_run,
        output_path=result.output_path,
        metadata_path=result.metadata_path,
        log_path=result.log_path,
    )


def _build_model_client(model_name: str, *, dry_run: bool):
    if dry_run:
        return DryRunQuestionClient(model_name)
    return OpenRouterQuestionClient(model_name=model_name)


@contextmanager
def _run_file_logging(log_path: Path) -> Iterator[None]:
    package_logger = logging.getLogger("contract_question_agent")
    previous_level = package_logger.level
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        handler.close()
        package_logger.setLevel(previous_level)


def _log_run_start(request: GenerateQuestionsRequest) -> None:
    logger.info("run_id=%s", request.run_id)
    logger.info("run_dir=%s", request.output_path.parent)
    logger.info("input_path=%s", request.input_path)
    logger.info("output_path=%s", request.output_path)
    logger.info("metadata_path=%s", request.metadata_path)
    logger.info("log_path=%s", request.log_path)
    logger.info("model_name=%s", request.model_name)
    logger.info("dry_run=%s", request.dry_run)
    logger.info(
        "filters clause_type=%s contract_id=%s limit=%s offset=%s",
        request.clause_type,
        request.contract_id,
        request.limit,
        request.offset,
    )


def _log_run_result(result: WrittenQuestions) -> None:
    logger.info("rows_read=%s", result.rows_read)
    logger.info("rows_filtered=%s", result.rows_filtered)
    logger.info("rows_in_scope=%s", result.rows_in_scope)
    logger.info("rows_out_of_scope=%s", result.rows_out_of_scope)
    logger.info("scope_status_counts=%s", result.scope_status_counts)
    logger.info("out_of_scope_reasons=%s", result.out_of_scope_reasons)
    logger.info("rows_generated=%s", result.rows_generated)
    logger.info("safety_failed_count=%s", result.safety_failed_count)
    logger.info("rows_written=%s", result.rows_written)
