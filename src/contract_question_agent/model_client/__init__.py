"""Model clients for verification-question generation."""

from contract_question_agent.model_client.openrouter import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterQuestionClient,
)

__all__ = ["DEFAULT_OPENROUTER_MODEL", "OpenRouterQuestionClient"]
