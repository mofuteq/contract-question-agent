"""OpenRouter client for one-call structured question generation."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

warnings.filterwarnings("ignore", message=r".*is experimental.*")

from agent_framework import Agent
from agent_framework_openai import OpenAIChatClient

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent.schemas import VerificationQuestionOutput


DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "Generate verification questions for a contract clause. Do not provide legal "
    "advice. Do not decide whether the clause is legal, enforceable, acceptable, "
    "or whether the user should sign. Generate questions a user can discuss with "
    "a qualified legal professional. Stay grounded in the given clause text. "
    "Return structured output only."
)


class OpenRouterQuestionClient:
    """OpenRouter generation client backed by a Microsoft Agent Framework Agent."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        agent: Agent | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        self.call_count = 0
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required unless --dry-run is set."
            )
        self.agent = agent or OpenAIChatClient(
            model=self.model_name,
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        ).as_agent(
            id="openrouter-verification-question-agent",
            name="OpenRouter verification question agent",
            instructions=SYSTEM_PROMPT,
        )

    async def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        self.call_count += 1
        response = await self.agent.run(
            json.dumps(
                {
                    "contract_id": record.contract_id,
                    "clause_type": record.clause_type,
                    "evidence_text": record.evidence_text,
                },
                ensure_ascii=True,
            ),
            options={"response_format": VerificationQuestionOutput},
        )
        return _coerce_agent_response(response, self.model_name)


def _coerce_agent_response(response: Any, model_name: str) -> VerificationQuestionOutput:
    value = _safe_getattr(response, "value")
    if isinstance(value, VerificationQuestionOutput):
        return _with_generation_defaults(value, model_name)
    if value is not None:
        return _with_generation_defaults(
            VerificationQuestionOutput.model_validate(value),
            model_name,
        )

    raw_value = _extract_raw_structured_value(response)
    if raw_value is not None:
        return _with_generation_defaults(
            VerificationQuestionOutput.model_validate(raw_value),
            model_name,
        )

    text = _safe_getattr(response, "text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "OpenRouter agent response did not contain structured output. "
            f"{_safe_response_summary(response)}"
        )
    parsed = json.loads(text)
    return _with_generation_defaults(
        VerificationQuestionOutput.model_validate(parsed),
        model_name,
    )


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _extract_raw_structured_value(response: Any) -> Any | None:
    raw = _safe_getattr(response, "raw_representation")
    for candidate in _iter_raw_candidates(raw):
        if isinstance(candidate, VerificationQuestionOutput):
            return candidate
        if isinstance(candidate, dict) and _looks_like_question_output(candidate):
            return candidate
    return None


def _iter_raw_candidates(raw: Any):
    if raw is None:
        return
    for attr in ("value", "parsed", "output_parsed"):
        value = _safe_getattr(raw, attr)
        if value is not None:
            yield value
    nested = _safe_getattr(raw, "raw_representation")
    if nested is not None and nested is not raw:
        yield from _iter_raw_candidates(nested)
    choices = _safe_getattr(raw, "choices")
    if choices:
        for choice in choices:
            message = _safe_getattr(choice, "message")
            if message is not None:
                yield from _iter_raw_candidates(message)


def _looks_like_question_output(value: dict) -> bool:
    return {
        "contract_id",
        "clause_type",
        "evidence_text",
        "verification_questions",
    }.issubset(value)


def _safe_response_summary(response: Any) -> str:
    raw = _safe_getattr(response, "raw_representation")
    messages = _safe_getattr(response, "messages") or []
    finish_reason = _safe_getattr(response, "finish_reason")
    return (
        f"response_type={type(response).__name__} "
        f"raw_type={type(raw).__name__ if raw is not None else None} "
        f"message_count={len(messages) if hasattr(messages, '__len__') else None} "
        f"finish_reason={finish_reason!r}"
    )


def _with_generation_defaults(
    output: VerificationQuestionOutput,
    model_name: str,
) -> VerificationQuestionOutput:
    return output.model_copy(
        update={
            "safety_disclaimer": output.safety_disclaimer or SAFETY_DISCLAIMER,
            "safety_status": output.safety_status or "unchecked",
            "safety_warnings": output.safety_warnings or [],
            "model_name": output.model_name or model_name,
        }
    )
