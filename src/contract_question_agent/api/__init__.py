"""FastAPI adapter for the contract verification-question workflow."""

from contract_question_agent.api.app import app, create_app

__all__ = ["app", "create_app"]
