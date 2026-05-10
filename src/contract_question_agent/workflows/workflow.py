"""LangGraph orchestration for verification-question generation.

LangGraph owns workflow state transitions and business-readable tracing here.
Node-internal LLM calling remains delegated to the configured model client,
including the existing MAF-backed OpenRouter client path.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import TypedDict, cast

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
)

from langgraph.graph import END, StateGraph

from contract_question_agent.schemas import (
    FilteredClauseSpans,
    GenerateQuestionsRequest,
    GeneratedQuestions,
    LoadedClauseSpans,
    QuestionModelClient,
    SafetyCheckedQuestions,
    WrittenQuestions,
)
from contract_question_agent.workflows import nodes
from contract_question_agent.workflows import tracing

logger = logging.getLogger(__name__)

LOAD_CLAUSE_SPANS = "LOAD_CLAUSE_SPANS"
FILTER_RECORDS = "FILTER_RECORDS"
GENERATE_MINIMAL_QUESTIONS = "GENERATE_MINIMAL_QUESTIONS"
SAFETY_CHECK = "SAFETY_CHECK"
WRITE_OUTPUT = "WRITE_OUTPUT"


class WorkflowGraphState(TypedDict):
    """LangGraph carrier for the framework-independent workflow state."""

    value: (
        GenerateQuestionsRequest
        | LoadedClauseSpans
        | FilteredClauseSpans
        | GeneratedQuestions
        | SafetyCheckedQuestions
        | WrittenQuestions
    )


def run_workflow(
    request: GenerateQuestionsRequest,
    *,
    model_client: QuestionModelClient,
) -> WrittenQuestions:
    """Run LOAD -> FILTER -> GENERATE -> SAFETY -> WRITE."""
    return asyncio.run(run_workflow_async(request, model_client=model_client))


async def run_workflow_async(
    request: GenerateQuestionsRequest,
    *,
    model_client: QuestionModelClient,
) -> WrittenQuestions:
    """Async workflow entrypoint, useful for tests and embedding."""
    graph = build_workflow(model_client=model_client)
    session_id = tracing.normalize_session_id(request.run_id)
    trace_name = "contract-question-agent-v0.3"
    trace_tags = ["contract-question-agent", "v0.3"]
    with tracing.session(
        session_id,
        trace_name=trace_name,
        tags=trace_tags,
    ):
        langgraph_callbacks = tracing.get_langgraph_callbacks(
            session_id=session_id,
            trace_name=trace_name,
            tags=trace_tags,
        )
        if langgraph_callbacks:
            result = await graph.ainvoke(
                {"value": request},
                config=_graph_config(
                    request,
                    session_id=session_id,
                    trace_name=trace_name,
                    tags=trace_tags,
                    callbacks=langgraph_callbacks,
                ),
            )
            output = _validate_workflow_output(result)
            tracing.flush()
            return output

        with tracing.span(
            trace_name,
            input=_request_summary(request),
            metadata={
                "run_id": request.run_id,
                "session_id": session_id,
                "dry_run": request.dry_run,
            },
        ):
            result = await graph.ainvoke(
                {"value": request},
                config=_graph_config(
                    request,
                    session_id=session_id,
                    trace_name=trace_name,
                    tags=trace_tags,
                ),
            )
            output = _validate_workflow_output(result)
            tracing.update_current_span(output=_written_summary(output))
            tracing.flush()
            return output


def build_workflow(*, model_client: QuestionModelClient):
    """Build a deterministic LangGraph with business-readable node names."""
    graph = StateGraph(WorkflowGraphState)

    async def load_node(state: WorkflowGraphState) -> WorkflowGraphState:
        request = cast(GenerateQuestionsRequest, state["value"])
        with tracing.state_transition(
            LOAD_CLAUSE_SPANS,
            input_state=request,
            next_node=FILTER_RECORDS,
        ) as record_output:
            loaded = nodes.load_clause_spans_node(request)
            logger.info("rows_read=%s", loaded.rows_read)
            record_output(loaded)
            return {"value": loaded}

    async def filter_node(state: WorkflowGraphState) -> WorkflowGraphState:
        loaded = cast(LoadedClauseSpans, state["value"])
        with tracing.state_transition(
            FILTER_RECORDS,
            input_state=loaded,
            next_node=GENERATE_MINIMAL_QUESTIONS,
        ) as record_output:
            filtered = nodes.filter_records_node(loaded)
            logger.info("rows_filtered=%s", filtered.rows_filtered)
            record_output(filtered)
            return {"value": filtered}

    async def generate_node(state: WorkflowGraphState) -> WorkflowGraphState:
        filtered = cast(FilteredClauseSpans, state["value"])
        with tracing.state_transition(
            GENERATE_MINIMAL_QUESTIONS,
            input_state=filtered,
            next_node=SAFETY_CHECK,
        ) as record_output:
            generated = await nodes.generate_minimal_questions_node(
                filtered,
                model_client,
            )
            logger.info("rows_generated=%s", generated.rows_generated)
            record_output(generated)
            return {"value": generated}

    async def safety_node(state: WorkflowGraphState) -> WorkflowGraphState:
        generated = cast(GeneratedQuestions, state["value"])
        with tracing.state_transition(
            SAFETY_CHECK,
            input_state=generated,
            next_node=WRITE_OUTPUT,
        ) as record_output:
            checked = nodes.safety_check_node(generated)
            logger.info("safety_failed_count=%s", checked.safety_failed_count)
            record_output(checked)
            return {"value": checked}

    async def write_node(state: WorkflowGraphState) -> WorkflowGraphState:
        checked = cast(SafetyCheckedQuestions, state["value"])
        with tracing.state_transition(
            WRITE_OUTPUT,
            input_state=checked,
            next_node="END",
        ) as record_output:
            written = nodes.write_output_node(checked)
            logger.info("rows_written=%s", written.rows_written)
            record_output(written)
            return {"value": written}

    graph.add_node(LOAD_CLAUSE_SPANS, load_node)
    graph.add_node(FILTER_RECORDS, filter_node)
    graph.add_node(GENERATE_MINIMAL_QUESTIONS, generate_node)
    graph.add_node(SAFETY_CHECK, safety_node)
    graph.add_node(WRITE_OUTPUT, write_node)

    graph.set_entry_point(LOAD_CLAUSE_SPANS)
    graph.add_edge(LOAD_CLAUSE_SPANS, FILTER_RECORDS)
    graph.add_edge(FILTER_RECORDS, GENERATE_MINIMAL_QUESTIONS)
    graph.add_edge(GENERATE_MINIMAL_QUESTIONS, SAFETY_CHECK)
    graph.add_edge(SAFETY_CHECK, WRITE_OUTPUT)
    graph.add_edge(WRITE_OUTPUT, END)
    return graph.compile()


def _validate_workflow_output(result: WorkflowGraphState) -> WrittenQuestions:
    output = result["value"]
    if not isinstance(output, WrittenQuestions):
        raise RuntimeError("LangGraph workflow did not produce a WrittenQuestions output.")
    return output


def _request_summary(request: GenerateQuestionsRequest) -> dict[str, object]:
    return {
        "run_id": request.run_id,
        "input_path": str(request.input_path),
        "output_path": str(request.output_path),
        "clause_type": request.clause_type,
        "contract_id": request.contract_id,
        "limit": request.limit,
        "offset": request.offset,
        "model_name": request.model_name,
        "dry_run": request.dry_run,
    }


def _graph_config(
    request: GenerateQuestionsRequest,
    *,
    session_id: str,
    trace_name: str = "contract-question-agent-v0.3",
    tags: list[str] | None = None,
    callbacks: list[object] | None = None,
) -> dict[str, object]:
    trace_tags = tags or ["contract-question-agent", "v0.3"]
    config: dict[str, object] = {
        "configurable": {
            "thread_id": session_id,
            "run_id": request.run_id,
            "session_id": session_id,
            "model_name": request.model_name,
            "dry_run": request.dry_run,
        },
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": trace_tags,
            "run_id": request.run_id,
        },
        "run_name": trace_name,
        "tags": trace_tags,
    }
    if callbacks:
        # Manual spans remain the canonical business trace. The LangGraph
        # callback owns workflow-level tracing only in explicit callback mode
        # so Langfuse can render the Agent Graph without duplicate trees.
        config["callbacks"] = callbacks
    return config


def _written_summary(written: WrittenQuestions) -> dict[str, object]:
    return {
        "output_path": str(written.output_path),
        "metadata_path": str(written.metadata_path),
        "log_path": str(written.log_path),
        "rows_read": written.rows_read,
        "rows_filtered": written.rows_filtered,
        "rows_generated": written.rows_generated,
        "safety_failed_count": written.safety_failed_count,
        "rows_written": written.rows_written,
    }
