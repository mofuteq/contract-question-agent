from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import openrouter
from contract_question_agent.model_client.openrouter import (
    OpenRouterQuestionClient,
    extract_usage_details,
)
from contract_question_agent.schemas import VerificationQuestionOutput


class FakeAgent:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def run(self, message, *, options):
        self.calls.append((message, options))
        return self.response


def _span() -> ClauseSpanRecord:
    return ClauseSpanRecord(
        contract_id="C1",
        source_file="C1.txt",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete.",
        start_char=0,
        end_char=26,
        label_present=True,
    )


def _output() -> VerificationQuestionOutput:
    return VerificationQuestionOutput(
        contract_id="C1",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete.",
        unknowns=[],
        decision_risks=[],
        legal_review_questions=[],
        verification_questions=[],
        suggested_next_step="Discuss with a qualified professional.",
        safety_disclaimer="",
        safety_status="unchecked",
        safety_warnings=[],
        model_name="",
    )


def test_openrouter_client_uses_agent_response_format_with_pydantic_value():
    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
    )

    output = asyncio.run(client.generate(_span()))

    assert output.model_name == "test-model"
    assert client.call_count == 1
    assert output.safety_disclaimer
    assert agent.calls[0][1] == {"response_format": VerificationQuestionOutput}
    assert json.loads(agent.calls[0][0]) == {
        "contract_id": "C1",
        "clause_type": "Non-Compete",
        "evidence_text": "Employee will not compete.",
    }


def test_openrouter_client_parses_raw_json_text():
    payload = _output().model_copy(update={"model_name": "raw-model"}).model_dump_json()
    agent = FakeAgent(SimpleNamespace(value=None, text=payload))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
    )

    output = asyncio.run(client.generate(_span()))

    assert output.contract_id == "C1"
    assert output.model_name == "raw-model"


def test_openrouter_client_validates_dict_like_response_value():
    payload = _output().model_copy(update={"model_name": "dict-model"}).model_dump()
    agent = FakeAgent(SimpleNamespace(value=payload, text=""))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
    )

    output = asyncio.run(client.generate(_span()))

    assert output.contract_id == "C1"
    assert output.model_name == "dict-model"


def test_openrouter_client_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))

    client = OpenRouterQuestionClient(model_name="test-model", agent=agent)

    assert client.api_key == "env-key"


def test_openrouter_client_reads_model_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")
    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))

    client = OpenRouterQuestionClient(agent=agent)

    assert client.model_name == "env-model"


def test_openrouter_client_traces_generation_usage_without_evidence(monkeypatch):
    response = SimpleNamespace(
        value=_output(),
        text="",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        ),
    )
    agent = FakeAgent(response)
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
    )
    span_events: list[dict] = []
    generation_updates: list[dict] = []

    @contextmanager
    def fake_span(name, *, input=None, metadata=None, as_type="span"):
        span_events.append(
            {
                "name": name,
                "input": input,
                "metadata": metadata,
                "as_type": as_type,
            }
        )
        yield

    monkeypatch.setattr(openrouter.tracing, "span", fake_span)
    monkeypatch.setattr(
        openrouter.tracing,
        "update_current_generation",
        lambda **kwargs: generation_updates.append(kwargs),
    )

    output = asyncio.run(client.generate(_span()))

    assert output.contract_id == "C1"
    assert span_events == [
        {
            "name": "openrouter-verification-question-agent",
            "input": {
                "contract_id": "C1",
                "clause_type": "Non-Compete",
                "evidence_char_count": 26,
            },
            "metadata": {
                "contract_id": "C1",
                "clause_type": "Non-Compete",
                "provider": "openrouter",
                "runtime": "microsoft-agent-framework",
            },
            "as_type": "generation",
        }
    ]
    assert generation_updates == [
        {
            "model": "test-model",
            "output": {
                "contract_id": "C1",
                "clause_type": "Non-Compete",
                "unknown_count": 0,
                "decision_risk_count": 0,
                "legal_review_question_count": 0,
                "verification_question_count": 0,
                "safety_status": "unchecked",
                "model_name": "test-model",
            },
            "usage_details": {"input": 11, "output": 7, "total": 18},
        }
    ]
    trace_payload = json.dumps({"spans": span_events, "updates": generation_updates})
    assert "Employee will not compete" not in trace_payload
    assert "evidence_text" not in trace_payload


def test_extract_usage_details_maps_openai_style_usage():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
        )
    )

    assert extract_usage_details(response) == {
        "input": 12,
        "output": 8,
        "total": 20,
    }


def test_extract_usage_details_maps_maf_style_usage():
    response = SimpleNamespace(
        metadata={
            "inputUsage": 13,
            "outputUsage": 9,
            "totalUsage": 22,
        }
    )

    assert extract_usage_details(response) == {
        "input": 13,
        "output": 9,
        "total": 22,
    }


def test_extract_usage_details_returns_none_when_usage_missing():
    assert extract_usage_details(SimpleNamespace(value=_output(), text="")) is None
