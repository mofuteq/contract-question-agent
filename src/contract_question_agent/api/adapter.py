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

from contract_question_agent.api.schemas import (
    GenerateVerificationQuestionsRequest,
    GenerateVerificationQuestionsResponse,
)
from contract_question_agent.cli_generate_questions import (
    DryRunQuestionClient,
    LOG_FILENAME,
    METADATA_FILENAME,
    OUTPUT_FILENAME,
    make_run_id,
)
from contract_question_agent.model_client import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterQuestionClient,
)
from contract_question_agent.schemas import GenerateQuestionsRequest, WrittenQuestions
from contract_question_agent.workflows import run_workflow_async

logger = logging.getLogger(__name__)


class RunAlreadyExistsError(ValueError):
    """Raised when a requested run id would overwrite local run artifacts."""


async def run_verification_question_workflow(
    api_request: GenerateVerificationQuestionsRequest,
) -> GenerateVerificationQuestionsResponse:
    """Build the existing workflow request and return its completed result."""
    run_id = api_request.run_id or make_run_id()
    run_dir = api_request.output_dir / run_id
    if run_dir.exists():
        raise RunAlreadyExistsError(f"Run directory already exists: {run_dir}")

    model_name = (
        api_request.model
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    )
    model_client = _build_model_client(model_name, dry_run=api_request.dry_run)
    output_path = run_dir / OUTPUT_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    log_path = run_dir / LOG_FILENAME
    workflow_request = GenerateQuestionsRequest(
        input_path=api_request.input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id=run_id,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        clause_type=api_request.clause_type,
        contract_id=api_request.contract_id,
        limit=api_request.limit,
        offset=api_request.offset,
        model_name=model_name,
        dry_run=api_request.dry_run,
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    with _run_file_logging(log_path):
        _log_run_start(workflow_request)
        result = await run_workflow_async(workflow_request, model_client=model_client)
        _log_run_result(result)

    return GenerateVerificationQuestionsResponse(
        run_id=workflow_request.run_id,
        created_at=workflow_request.created_at,
        input_path=workflow_request.input_path,
        output_path=result.output_path,
        metadata_path=result.metadata_path,
        log_path=result.log_path,
        clause_type=workflow_request.clause_type,
        contract_id=workflow_request.contract_id,
        limit=workflow_request.limit,
        offset=workflow_request.offset,
        model_name=workflow_request.model_name,
        dry_run=workflow_request.dry_run,
        rows_read=result.rows_read,
        rows_filtered=result.rows_filtered,
        rows_in_scope=result.rows_in_scope,
        rows_out_of_scope=result.rows_out_of_scope,
        rows_generated=result.rows_generated,
        scope_status_counts=result.scope_status_counts,
        out_of_scope_reasons=result.out_of_scope_reasons,
        safety_failed_count=result.safety_failed_count,
        rows_written=result.rows_written,
        outputs=result.outputs,
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
