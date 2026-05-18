"""FastAPI application boundary for contract-question-agent."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from contract_question_agent.api.adapter import (
    make_api_run_id,
    run_workflow_from_api_request,
)
from contract_question_agent.api.ag_ui_adapter import run_response_to_snapshot
from contract_question_agent.api.ag_ui_schemas import (
    AG_UI_RUN_ERROR,
    AG_UI_RUN_FINISHED,
    AG_UI_RUN_STARTED,
    AG_UI_STATE_SNAPSHOT,
    AG_UI_STEP_FINISHED,
    AG_UI_STEP_STARTED,
    sse_event,
)
from contract_question_agent.api.schemas import (
    HealthResponse,
    RunRequest,
    RunResponse,
)


def create_app(*, load_env: bool = True) -> FastAPI:
    if load_env:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    application = FastAPI(
        title="contract-question-agent",
        version="0.1.0",
        description=(
            "Thin HTTP adapter for generating clause-grounded verification "
            "questions with the existing LangGraph workflow."
        ),
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.post(
        "/runs",
        response_model=RunResponse,
        status_code=status.HTTP_200_OK,
    )
    async def create_run(request: RunRequest) -> RunResponse:
        try:
            return await run_workflow_from_api_request(request)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    @application.post("/ag-ui/runs")
    async def create_ag_ui_run(request: RunRequest) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[str]:
            run_id = make_api_run_id()
            thread_id = run_id

            yield sse_event(
                AG_UI_RUN_STARTED,
                {
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "input": {
                        "contract_id": request.contract_id,
                        "clause_type": request.clause_type,
                        "dry_run": request.dry_run,
                        "mcp_hints_enabled": request.mcp_hints_enabled,
                        "has_evidence_text": bool(request.evidence_text),
                    },
                },
            )
            yield sse_event(AG_UI_STEP_STARTED, {"step_name": "workflow"})

            try:
                response = await run_workflow_from_api_request(request, run_id=run_id)
                snapshot = run_response_to_snapshot(response)

                yield sse_event(AG_UI_STEP_FINISHED, {"step_name": "workflow"})
                yield sse_event(AG_UI_STATE_SNAPSHOT, {"snapshot": snapshot})
                yield sse_event(
                    AG_UI_RUN_FINISHED,
                    {
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "result": snapshot,
                        "outcome": {"type": "success"},
                    },
                )
            except Exception as exc:
                yield sse_event(
                    AG_UI_RUN_ERROR,
                    {
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "message": str(exc),
                        "code": exc.__class__.__name__,
                    },
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return application


app = create_app()
