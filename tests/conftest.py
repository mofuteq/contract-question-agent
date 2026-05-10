from __future__ import annotations

import pytest

from contract_question_agent import tracing


@pytest.fixture(autouse=True)
def reset_langfuse_tracing(monkeypatch):
    for name in [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_TRACING_ENVIRONMENT",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tracing, "_CLIENT", None)
    monkeypatch.setattr(tracing, "_ENABLED", None)
