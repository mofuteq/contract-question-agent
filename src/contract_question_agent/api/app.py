"""FastAPI application boundary for contract-question-agent."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

from contract_question_agent.api.adapter import run_workflow_from_api_request
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

    return application


app = create_app()
