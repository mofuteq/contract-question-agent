from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

from contract_question_agent.clause_hints.schemas import ClauseReviewHints
from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import openrouter
from contract_question_agent.model_client.openrouter import (
    MCPHintsLookupResult,
    OpenRouterQuestionClient,
    extract_usage_details,
    render_system_prompt,
)
from contract_question_agent.schemas import VerificationQuestionOutput


class FakeAgent:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def run(self, message, **kwargs):
        self.calls.append((message, kwargs))
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


def _output_with_selected_lens() -> VerificationQuestionOutput:
    return VerificationQuestionOutput.model_validate(
        _output().model_dump()
        | {
            "selected_review_lenses": [
                {
                    "label": "Time period",
                    "source": "mcp_clause_review_hints",
                    "reason": "The clause states a one-year restriction.",
                }
            ]
        },
    )


def _hints() -> ClauseReviewHints:
    return ClauseReviewHints(
        clause_type="Non-Compete",
        risk_lens="Review scope, duration, and exceptions as practical lenses.",
        common_unknowns=["Which activities are covered?", "How long does it last?"],
        question_categories=["Scope", "Timing"],
        review_hints=["Compare terms against the clause text."],
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
    assert agent.calls[0][1] == {
        "options": {
            "instructions": openrouter.SYSTEM_PROMPT,
            "response_format": VerificationQuestionOutput,
        }
    }
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


def test_openrouter_client_loads_system_prompt_from_jinja_template(monkeypatch):
    created_agents = []

    class FakeOpenAIChatClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def as_agent(self, **kwargs):
            created_agents.append(kwargs)
            return FakeAgent(SimpleNamespace(value=_output(), text=""))

    monkeypatch.setattr(openrouter, "OpenAIChatClient", FakeOpenAIChatClient)

    client = OpenRouterQuestionClient(api_key="test-key", model_name="test-model")

    assert client.agent is not None
    assert created_agents[0]["instructions"] == openrouter.SYSTEM_PROMPT
    assert "Generate verification questions for a contract clause" in openrouter.SYSTEM_PROMPT
    assert "Return structured output only." in openrouter.SYSTEM_PROMPT


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
                "mcp_hints_enabled": False,
                "mcp_hints_lookup_attempted": False,
                "mcp_hints_found": False,
                "mcp_tool_name": openrouter.MCP_HINTS_TOOL_NAME,
                "common_unknowns_count": 0,
                "question_categories_count": 0,
                "review_hints_count": 0,
            },
            "as_type": "generation",
        }
    ]
    assert generation_updates == [
        {
            "model": "test-model",
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": openrouter.SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": {
                            "contract_id": "C1",
                            "clause_type": "Non-Compete",
                            "evidence_char_count": 26,
                        },
                    },
                ],
            },
            "output": {
                "contract_id": "C1",
                "clause_type": "Non-Compete",
                "selected_review_lens_count": 0,
                "unknown_count": 0,
                "decision_risk_count": 0,
                "legal_review_question_count": 0,
                "verification_question_count": 0,
                "safety_status": "unchecked",
                "model_name": "test-model",
            },
            "metadata": {
                "contract_id": "C1",
                "clause_type": "Non-Compete",
                "provider": "openrouter",
                "runtime": "microsoft-agent-framework",
                "system_prompt_template": "verification_question_system.j2",
                "mcp_hints_enabled": False,
                "mcp_hints_lookup_attempted": False,
                "mcp_hints_found": False,
                "mcp_tool_name": openrouter.MCP_HINTS_TOOL_NAME,
                "common_unknowns_count": 0,
                "question_categories_count": 0,
                "review_hints_count": 0,
            },
            "usage_details": {"input": 11, "output": 7, "total": 18},
        }
    ]
    trace_payload = json.dumps({"spans": span_events, "updates": generation_updates})
    assert "Employee will not compete" not in trace_payload
    assert "evidence_text" not in trace_payload


def test_openrouter_client_does_not_use_mcp_when_flag_unset(monkeypatch):
    monkeypatch.delenv(openrouter.USE_MCP_HINTS_ENV, raising=False)

    async def fail_if_lookup_is_called(clause_type):
        raise AssertionError("MCP lookup should not run when MCP hints are off.")

    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
        mcp_hints_lookup=fail_if_lookup_is_called,
    )

    asyncio.run(client.generate(_span()))

    assert "tools" not in agent.calls[0][1]


def test_openrouter_client_does_not_use_mcp_when_flag_false(monkeypatch):
    monkeypatch.setenv(openrouter.USE_MCP_HINTS_ENV, "false")

    async def fail_if_lookup_is_called(clause_type):
        raise AssertionError("MCP lookup should not run when MCP hints are false.")

    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
        mcp_hints_lookup=fail_if_lookup_is_called,
    )

    asyncio.run(client.generate(_span()))

    assert client.use_mcp_hints is False
    assert "tools" not in agent.calls[0][1]


def test_openrouter_client_looks_up_mcp_hints_when_flag_true(monkeypatch):
    monkeypatch.setenv(openrouter.USE_MCP_HINTS_ENV, "true")
    lookup_calls = []

    async def fake_lookup(clause_type):
        lookup_calls.append(clause_type)
        return MCPHintsLookupResult(attempted=True, found=True, hints=_hints())

    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
        mcp_hints_lookup=fake_lookup,
    )

    asyncio.run(client.generate(_span()))

    assert client.use_mcp_hints is True
    assert lookup_calls == ["Non-Compete"]
    assert "tools" not in agent.calls[0][1]
    instructions = agent.calls[0][1]["options"]["instructions"]
    assert "Clause review hint candidates were retrieved from a tool." in instructions
    assert "Select only hints that are relevant to the given clause text." in instructions
    assert "Ignore hints that are not supported by or useful for the clause." in instructions
    assert "Populate selected_review_lenses before generating verification questions." in instructions
    assert "source='mcp_clause_review_hints'" in instructions
    assert "Review scope, duration, and exceptions as practical lenses." in instructions


def test_system_prompt_omits_candidate_hints_when_not_found_or_disabled():
    assert "Clause review hint candidates were retrieved from a tool." not in openrouter.SYSTEM_PROMPT
    assert (
        "Clause review hint candidates were retrieved from a tool."
        not in render_system_prompt(None)
    )


def test_system_prompt_renders_candidate_hints_when_found():
    prompt = render_system_prompt(_hints())

    assert "Clause review hint candidates were retrieved from a tool." in prompt
    assert "Use them as candidate lenses, not conclusions." in prompt
    assert "Select only hints that are relevant to the given clause text." in prompt
    assert "Ignore hints that are not supported by or useful for the clause." in prompt
    assert "Report the selected lenses in selected_review_lenses." in prompt
    assert "source='mcp_clause_review_hints'" in prompt
    assert "Keep selected lens reasons short and grounded in the clause text." in prompt
    assert "Do not treat selected lenses as legal conclusions." in prompt
    assert "Risk lens:" in prompt
    assert "Review scope, duration, and exceptions as practical lenses." in prompt
    assert "Common unknowns:" in prompt
    assert "Which activities are covered?" in prompt
    assert "Question categories:" in prompt
    assert "Scope" in prompt
    assert "Review hints:" in prompt
    assert "Compare terms against the clause text." in prompt


def test_openrouter_client_traces_mcp_lookup_metadata_without_evidence(monkeypatch):
    monkeypatch.setenv(openrouter.USE_MCP_HINTS_ENV, "true")

    async def fake_lookup(clause_type):
        return MCPHintsLookupResult(attempted=True, found=True, hints=_hints())

    agent = FakeAgent(SimpleNamespace(value=_output(), text=""))
    generation_updates: list[dict] = []
    span_events: list[dict] = []

    @contextmanager
    def fake_span(name, *, input=None, metadata=None, as_type="span"):
        span_events.append({"metadata": metadata})
        yield

    monkeypatch.setattr(openrouter.tracing, "span", fake_span)
    monkeypatch.setattr(
        openrouter.tracing,
        "update_current_generation",
        lambda **kwargs: generation_updates.append(kwargs),
    )
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
        mcp_hints_lookup=fake_lookup,
    )

    asyncio.run(client.generate(_span()))

    metadata = generation_updates[0]["metadata"]
    assert span_events[0]["metadata"]["mcp_hints_lookup_attempted"] is True
    assert metadata["mcp_hints_enabled"] is True
    assert metadata["mcp_hints_lookup_attempted"] is True
    assert metadata["mcp_hints_found"] is True
    assert metadata["mcp_tool_name"] == "lookup_clause_review_hints"
    assert metadata["clause_type"] == "Non-Compete"
    assert metadata["common_unknowns_count"] == 2
    assert metadata["question_categories_count"] == 2
    assert metadata["review_hints_count"] == 1
    assert "evidence_text" not in json.dumps(
        {"spans": span_events, "updates": generation_updates}
    )


def test_openrouter_client_accepts_selected_review_lenses_from_agent_response():
    agent = FakeAgent(SimpleNamespace(value=_output_with_selected_lens(), text=""))
    client = OpenRouterQuestionClient(
        api_key="test-key",
        model_name="test-model",
        agent=agent,
    )

    output = asyncio.run(client.generate(_span()))

    assert len(output.selected_review_lenses) == 1
    assert output.selected_review_lenses[0].label == "Time period"
    assert output.selected_review_lenses[0].source == "mcp_clause_review_hints"


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
