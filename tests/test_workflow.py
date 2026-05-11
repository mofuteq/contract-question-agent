from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType
from contextlib import contextmanager
from pathlib import Path

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent.schemas import (
    GenerateQuestionsRequest,
    LegalReviewQuestion,
    VerificationQuestion,
    VerificationQuestionOutput,
    WrittenQuestions,
)
from contract_question_agent.workflows import workflow
from contract_question_agent.workflows import tracing
from contract_question_agent.workflows import run_workflow, run_workflow_async
from contract_question_agent.workflows.nodes.filter_records import filter_clause_spans


class FakeQuestionClient:
    model_name = "fake-model"

    def __init__(self, *, unsafe: bool = False) -> None:
        self.call_count = 0
        self.unsafe = unsafe

    async def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        self.call_count += 1
        return VerificationQuestionOutput(
            contract_id=record.contract_id,
            clause_type=record.clause_type,
            evidence_text=record.evidence_text,
            unknowns=["You should sign." if self.unsafe else "Unknown business context."],
            decision_risks=["Potential operational impact."],
            legal_review_questions=[
                LegalReviewQuestion(
                    question=f"Review {record.clause_type}?",
                    reason="Professional context is needed.",
                )
            ],
            verification_questions=[
                VerificationQuestion(
                    question="What facts should be verified?",
                    why_it_matters="Facts ground the review.",
                )
            ],
            suggested_next_step="Discuss with a qualified professional.",
            safety_disclaimer="",
            safety_status="unchecked",
            safety_warnings=[],
            model_name=self.model_name,
        )


def _span(
    contract_id: str,
    clause_type: str,
    evidence_text: str = "Clause evidence.",
) -> ClauseSpanRecord:
    return ClauseSpanRecord(
        contract_id=contract_id,
        source_file=f"{contract_id}.txt",
        clause_type=clause_type,
        evidence_text=evidence_text,
        start_char=0,
        end_char=len(evidence_text),
        label_present=True,
    )


def _write_spans(path: Path, records: list[ClauseSpanRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def test_filter_records_is_deterministic_for_cli_args():
    records = [
        _span("C1", "Non-Compete"),
        _span("C2", "Non-Compete"),
        _span("C2", "Governing Law"),
        _span("C3", "Non-Compete"),
    ]

    assert filter_clause_spans(records, clause_type="Non-Compete", offset=1, limit=2) == [
        records[1],
        records[3],
    ]
    assert filter_clause_spans(records, contract_id="C2") == [records[1], records[2]]
    assert filter_clause_spans(
        records,
        clause_type="Governing Law",
        contract_id="C2",
        limit=1,
    ) == [records[2]]


def test_run_workflow_limit_one_calls_generate_once_and_writes_one_row(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    metadata_path = tmp_path / "run_metadata.json"
    log_path = tmp_path / "run.log"
    _write_spans(
        input_path,
        [
            _span("C1", "Non-Compete", "Do not compete."),
            _span("C2", "Governing Law", "Delaware law."),
            _span("C3", "Non-Compete", "Do not solicit customers."),
        ],
    )
    request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id="workflow-test",
        created_at="2026-05-10T12:00:00+09:00",
        clause_type="Non-Compete",
        limit=1,
        offset=1,
        model_name="fake-model",
        dry_run=True,
    )

    fake_client = FakeQuestionClient()
    result = run_workflow(request, model_client=fake_client)

    assert result.rows_written == 1
    assert result.metadata_path == metadata_path
    assert result.rows_read == 3
    assert result.rows_filtered == 1
    assert result.rows_generated == 1
    assert result.safety_failed_count == 0
    assert fake_client.call_count == 1
    rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0].contract_id == "C3"
    assert rows[0].safety_status == "passed"
    assert rows[0].safety_disclaimer == SAFETY_DISCLAIMER


def test_run_workflow_limit_three_calls_generate_three_times(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    metadata_path = tmp_path / "run_metadata.json"
    log_path = tmp_path / "run.log"
    _write_spans(
        input_path,
        [
            _span("C1", "Non-Compete", "Do not compete."),
            _span("C2", "Non-Compete", "Do not solicit customers."),
            _span("C3", "Non-Compete", "Do not use confidential information."),
            _span("C4", "Governing Law", "Delaware law."),
        ],
    )
    request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id="workflow-test",
        created_at="2026-05-10T12:00:00+09:00",
        clause_type="Non-Compete",
        limit=3,
        model_name="fake-model",
        dry_run=True,
    )

    fake_client = FakeQuestionClient()
    result = run_workflow(request, model_client=fake_client)

    assert result.rows_written == 3
    assert result.rows_read == 4
    assert result.rows_filtered == 3
    assert result.rows_generated == 3
    assert result.safety_failed_count == 0
    assert fake_client.call_count == 3
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 3


def test_run_workflow_metadata_records_safety_failure_count(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    metadata_path = tmp_path / "run_metadata.json"
    log_path = tmp_path / "run.log"
    _write_spans(input_path, [_span("C1", "Non-Compete", "Do not compete.")])
    request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id="workflow-test",
        created_at="2026-05-10T12:00:00+09:00",
        clause_type="Non-Compete",
        limit=1,
        model_name="fake-model",
        dry_run=True,
    )

    result = run_workflow(request, model_client=FakeQuestionClient(unsafe=True))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.safety_failed_count == 1
    assert metadata["rows_read"] == 1
    assert metadata["rows_filtered"] == 1
    assert metadata["rows_generated"] == 1
    assert metadata["safety_failed_count"] == 1
    assert metadata["rows_written"] == 1
    assert rows[0].safety_status == "failed"


def test_run_workflow_traces_langgraph_state_transitions(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    metadata_path = tmp_path / "run_metadata.json"
    log_path = tmp_path / "run.log"
    _write_spans(input_path, [_span("C1", "Non-Compete", "Do not compete.")])
    request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id="workflow-trace-test",
        created_at="2026-05-10T12:00:00+09:00",
        clause_type="Non-Compete",
        limit=1,
        model_name="fake-model",
        dry_run=True,
    )
    events: list[dict] = []
    active: list[dict] = []
    sessions: list[dict] = []

    @contextmanager
    def fake_span(name, *, input=None, metadata=None, as_type="span"):
        event = {
            "name": name,
            "input": input,
            "metadata": metadata,
            "as_type": as_type,
        }
        events.append(event)
        active.append(event)
        try:
            yield
        finally:
            active.pop()

    def fake_update_current_span(**kwargs):
        active[-1].update(kwargs)

    @contextmanager
    def fake_session(session_id, *, trace_name=None, tags=None):
        sessions.append(
            {"session_id": session_id, "trace_name": trace_name, "tags": tags}
        )
        yield

    monkeypatch.setattr(tracing, "span", fake_span)
    monkeypatch.setattr(tracing, "session", fake_session)
    monkeypatch.setattr(tracing, "get_langgraph_callbacks", lambda **kwargs: [])
    monkeypatch.setattr(tracing, "update_current_span", fake_update_current_span)
    monkeypatch.setattr(tracing, "flush", lambda: None)

    run_workflow(request, model_client=FakeQuestionClient())

    assert sessions == [
        {
            "session_id": "workflow-trace-test",
            "trace_name": "contract-question-agent-v0.3",
            "tags": ["contract-question-agent", "v0.3"],
        }
    ]
    assert events[0]["metadata"]["session_id"] == "workflow-trace-test"
    transition_events = [
        event for event in events if event["name"] != "contract-question-agent-v0.3"
    ]
    assert [event["name"] for event in transition_events] == [
        "LOAD_CLAUSE_SPANS",
        "FILTER_RECORDS",
        "GENERATE_MINIMAL_QUESTIONS",
        "SAFETY_CHECK",
        "WRITE_OUTPUT",
    ]
    assert transition_events[0]["metadata"] == {
        "node": "LOAD_CLAUSE_SPANS",
        "input_state": "GenerateQuestionsRequest",
        "next_node": "FILTER_RECORDS",
        "output_state": "LoadedClauseSpans",
    }
    assert transition_events[0]["input"]["state"] == "GenerateQuestionsRequest"
    assert transition_events[0]["output"]["state"] == "LoadedClauseSpans"
    assert transition_events[0]["output"]["rows_read"] == 1
    assert transition_events[-1]["metadata"]["next_node"] == "END"
    assert transition_events[-1]["output"]["rows_written"] == 1
    assert "evidence_text" not in json.dumps(transition_events)


def test_langfuse_session_id_is_ascii_and_under_limit():
    assert tracing.normalize_session_id("run-123") == "run-123"
    assert tracing.normalize_session_id("実行-123") == "-123"
    assert tracing.normalize_session_id("x" * 250) == "x" * 200
    assert tracing.normalize_session_id("実行") == "contract-question-agent-run"


def test_langgraph_callbacks_can_be_disabled_with_env_flag(monkeypatch):
    monkeypatch.setenv("LANGFUSE_LANGGRAPH_CALLBACK_ENABLED", "false")
    monkeypatch.setattr(tracing, "_ENABLED", True)

    assert (
        tracing.get_langgraph_callbacks(
            session_id="run-123",
            trace_name="contract-question-agent-v0.3",
        )
        == []
    )


def test_langgraph_callbacks_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LANGFUSE_LANGGRAPH_CALLBACK_ENABLED", raising=False)
    monkeypatch.setattr(tracing, "_ENABLED", True)

    assert (
        tracing.get_langgraph_callbacks(
            session_id="run-123",
            trace_name="contract-question-agent-v0.3",
        )
        == []
    )


def test_langgraph_callbacks_are_disabled_without_langfuse_credentials(monkeypatch):
    monkeypatch.setenv("LANGFUSE_LANGGRAPH_CALLBACK_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(tracing, "_ENABLED", None)

    assert (
        tracing.get_langgraph_callbacks(
            session_id="run-123",
            trace_name="contract-question-agent-v0.3",
        )
        == []
    )


def test_langgraph_callbacks_use_current_trace_context(monkeypatch):
    callback_calls: list[dict] = []

    class FakeCallbackHandler:
        def __init__(self, **kwargs):
            callback_calls.append(kwargs)

    class FakeClient:
        def get_current_trace_id(self):
            return "trace-123"

    langfuse_module = ModuleType("langfuse")
    langfuse_langchain_module = ModuleType("langfuse.langchain")
    langfuse_langchain_module.CallbackHandler = FakeCallbackHandler

    monkeypatch.setenv("LANGFUSE_LANGGRAPH_CALLBACK_ENABLED", "true")
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "get_client", lambda: FakeClient())
    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langfuse_langchain_module)

    callbacks = tracing.get_langgraph_callbacks(
        session_id="run-123",
        trace_name="contract-question-agent-v0.3",
        tags=["contract-question-agent", "v0.3"],
    )

    assert len(callbacks) == 1
    assert callback_calls == [{"trace_context": {"trace_id": "trace-123"}}]


def test_langgraph_callbacks_return_empty_without_active_trace(monkeypatch):
    callback_calls: list[dict] = []

    class FakeCallbackHandler:
        def __init__(self, **kwargs):
            callback_calls.append(kwargs)

    class FakeClient:
        def get_current_trace_id(self):
            return None

    langfuse_module = ModuleType("langfuse")
    langfuse_langchain_module = ModuleType("langfuse.langchain")
    langfuse_langchain_module.CallbackHandler = FakeCallbackHandler

    monkeypatch.setenv("LANGFUSE_LANGGRAPH_CALLBACK_ENABLED", "true")
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "get_client", lambda: FakeClient())
    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langfuse_langchain_module)

    assert (
        tracing.get_langgraph_callbacks(
            session_id="run-123",
            trace_name="contract-question-agent-v0.3",
        )
        == []
    )
    assert callback_calls == []


def test_run_workflow_passes_configurable_run_context_to_langgraph(
    tmp_path,
    monkeypatch,
):
    request = GenerateQuestionsRequest(
        input_path=tmp_path / "clause_spans.jsonl",
        output_path=tmp_path / "verification_questions.jsonl",
        metadata_path=tmp_path / "run_metadata.json",
        log_path=tmp_path / "run.log",
        run_id="workflow-config-test",
        created_at="2026-05-10T12:00:00+09:00",
        model_name="fake-model",
        dry_run=True,
    )
    written = WrittenQuestions(
        output_path=request.output_path,
        metadata_path=request.metadata_path,
        log_path=request.log_path,
        rows_read=0,
        rows_filtered=0,
        rows_generated=0,
        safety_failed_count=0,
        rows_written=0,
    )
    calls: list[dict] = []

    class FakeGraph:
        async def ainvoke(self, state, *, config=None):
            calls.append({"state": state, "config": config})
            return {"value": written}

    monkeypatch.setattr(workflow, "build_workflow", lambda *, model_client: FakeGraph())
    monkeypatch.setattr(tracing, "get_langgraph_callbacks", lambda **kwargs: [])

    result = asyncio.run(
        run_workflow_async(request, model_client=FakeQuestionClient())
    )

    assert result == written
    assert calls == [
        {
            "state": {"value": request},
            "config": {
                "configurable": {
                    "thread_id": "workflow-config-test",
                    "run_id": "workflow-config-test",
                    "session_id": "workflow-config-test",
                    "model_name": "fake-model",
                    "dry_run": True,
                },
                "metadata": {
                    "langfuse_session_id": "workflow-config-test",
                    "langfuse_tags": ["contract-question-agent", "v0.3"],
                    "run_id": "workflow-config-test",
                },
                "run_name": "contract-question-agent-v0.3",
                "tags": ["contract-question-agent", "v0.3"],
            },
        }
    ]


def test_run_workflow_passes_langgraph_callbacks_when_enabled(
    tmp_path,
    monkeypatch,
):
    request = GenerateQuestionsRequest(
        input_path=tmp_path / "clause_spans.jsonl",
        output_path=tmp_path / "verification_questions.jsonl",
        metadata_path=tmp_path / "run_metadata.json",
        log_path=tmp_path / "run.log",
        run_id="workflow-callback-test",
        created_at="2026-05-10T12:00:00+09:00",
        model_name="fake-model",
        dry_run=True,
    )
    written = WrittenQuestions(
        output_path=request.output_path,
        metadata_path=request.metadata_path,
        log_path=request.log_path,
        rows_read=0,
        rows_filtered=0,
        rows_generated=0,
        safety_failed_count=0,
        rows_written=0,
    )
    callback = object()
    calls: list[dict] = []
    span_events: list[dict] = []
    active_spans: list[str] = []

    class FakeGraph:
        async def ainvoke(self, state, *, config=None):
            calls.append({"state": state, "config": config})
            return {"value": written}

    @contextmanager
    def fake_span(name, *, input=None, metadata=None, as_type="span"):
        span_events.append({"name": name, "input": input, "metadata": metadata})
        active_spans.append(name)
        try:
            yield
        finally:
            active_spans.pop()

    def fake_get_langgraph_callbacks(**kwargs):
        assert active_spans == ["contract-question-agent-v0.3"]
        return [callback]

    monkeypatch.setattr(workflow, "build_workflow", lambda *, model_client: FakeGraph())
    monkeypatch.setattr(tracing, "span", fake_span)
    monkeypatch.setattr(
        tracing,
        "get_langgraph_callbacks",
        fake_get_langgraph_callbacks,
    )

    result = asyncio.run(
        run_workflow_async(request, model_client=FakeQuestionClient())
    )

    assert result == written
    assert [event["name"] for event in span_events] == ["contract-question-agent-v0.3"]
    assert calls[0]["config"] == {
        "configurable": {
            "thread_id": "workflow-callback-test",
            "run_id": "workflow-callback-test",
            "session_id": "workflow-callback-test",
            "model_name": "fake-model",
            "dry_run": True,
        },
        "metadata": {
            "langfuse_session_id": "workflow-callback-test",
            "langfuse_tags": ["contract-question-agent", "v0.3"],
            "run_id": "workflow-callback-test",
        },
        "run_name": "contract-question-agent-v0.3",
        "tags": ["contract-question-agent", "v0.3"],
        "callbacks": [callback],
    }


def test_state_transition_still_traces_in_callback_mode(monkeypatch):
    span_events: list[str] = []

    @contextmanager
    def fake_span(name, *, input=None, metadata=None, as_type="span"):
        span_events.append(name)
        yield

    monkeypatch.setenv("LANGFUSE_LANGGRAPH_CALLBACK_ENABLED", "true")
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "span", fake_span)

    with tracing.state_transition(
        "LOAD_CLAUSE_SPANS",
        input_state=object(),
        next_node="FILTER_RECORDS",
    ) as record_output:
        record_output(object())

    assert span_events == ["LOAD_CLAUSE_SPANS"]
