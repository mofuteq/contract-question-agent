"""OpenRouter client for one-call structured question generation."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent.schemas import VerificationQuestionOutput


DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-pro"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "Generate verification questions for a contract clause. Do not provide legal "
    "advice. Do not decide whether the clause is legal, enforceable, acceptable, "
    "or whether the user should sign. Generate questions a user can discuss with "
    "a qualified legal professional. Stay grounded in the given clause text. "
    "Return structured output only."
)


class OpenRouterQuestionClient:
    """Minimal OpenRouter chat-completions client using structured JSON output."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = DEFAULT_OPENROUTER_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required unless --dry-run is set."
            )

    def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "contract_id": record.contract_id,
                            "clause_type": record.clause_type,
                            "evidence_text": record.evidence_text,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "verification_question_output",
                    "strict": True,
                    "schema": VerificationQuestionOutput.model_json_schema(),
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        content = _extract_content(response.json())
        parsed = json.loads(content)
        parsed.setdefault("safety_disclaimer", SAFETY_DISCLAIMER)
        parsed.setdefault("safety_status", "unchecked")
        parsed.setdefault("safety_warnings", [])
        parsed.setdefault("model_name", self.model_name)
        return VerificationQuestionOutput.model_validate(parsed)


def _extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenRouter response did not contain message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter response message content was empty")
    return content
