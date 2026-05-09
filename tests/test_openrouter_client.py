from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client.openrouter import OpenRouterQuestionClient
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


def test_openrouter_client_reads_api_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=dotenv-key\n", encoding="utf-8")
    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))

    client = OpenRouterQuestionClient(model_name="test-model", agent=agent)

    assert client.api_key == "dotenv-key"
