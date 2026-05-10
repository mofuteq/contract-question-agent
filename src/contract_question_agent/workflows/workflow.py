"""Workflow assembly for the v0.2 Microsoft Agent Framework graph."""

from __future__ import annotations

import asyncio
import logging
from typing import Never

from agent_framework import FunctionExecutor, WorkflowBuilder, WorkflowContext

from contract_question_agent import tracing
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

logger = logging.getLogger(__name__)


def run_workflow(
    request: GenerateQuestionsRequest,
    *,
    model_client: QuestionModelClient,
) -> WrittenQuestions:
    """Run LOAD -> FILTER -> GENERATE -> SAFETY -> WRITE -> DONE."""
    workflow = build_workflow(model_client=model_client)
    result = asyncio.run(workflow.run(request))
    outputs = result.get_outputs()
    if len(outputs) != 1 or not isinstance(outputs[0], WrittenQuestions):
        raise RuntimeError("Microsoft workflow did not produce one DONE output.")
    return outputs[0]


def build_workflow(*, model_client: QuestionModelClient):
    @tracing.observe(name="LOAD_CLAUSE_SPANS", as_type="span")
    async def load_node(
        request: GenerateQuestionsRequest,
        ctx: WorkflowContext[LoadedClauseSpans, Never],
    ) -> None:
        state = nodes.load_clause_spans_node(request)
        tracing.update_current_generation(metadata={"rows_read": state.rows_read})
        logger.info("rows_read=%s", state.rows_read)
        await ctx.send_message(state)

    @tracing.observe(name="FILTER_RECORDS", as_type="span")
    async def filter_node_func(
        state: LoadedClauseSpans,
        ctx: WorkflowContext[FilteredClauseSpans, Never],
    ) -> None:
        filtered = nodes.filter_records_node(state)
        tracing.update_current_generation(
            metadata={
                "rows_read": filtered.rows_read,
                "rows_filtered": filtered.rows_filtered,
            }
        )
        logger.info("rows_filtered=%s", filtered.rows_filtered)
        await ctx.send_message(filtered)

    @tracing.observe(name="GENERATE_MINIMAL_QUESTIONS", as_type="span")
    async def generate_node(
        state: FilteredClauseSpans,
        ctx: WorkflowContext[GeneratedQuestions, Never],
    ) -> None:
        generated = await nodes.generate_minimal_questions_node(state, model_client)
        tracing.update_current_generation(
            metadata={
                "rows_filtered": generated.rows_filtered,
                "rows_generated": generated.rows_generated,
                "model_name": generated.request.model_name,
                "dry_run": generated.request.dry_run,
            }
        )
        logger.info("rows_generated=%s", generated.rows_generated)
        await ctx.send_message(generated)

    @tracing.observe(name="SAFETY_CHECK", as_type="span")
    async def safety_node(
        state: GeneratedQuestions,
        ctx: WorkflowContext[SafetyCheckedQuestions, Never],
    ) -> None:
        checked = nodes.safety_check_node(state)
        tracing.update_current_generation(
            metadata={
                "rows_generated": checked.rows_generated,
                "safety_failed_count": checked.safety_failed_count,
            }
        )
        logger.info("safety_failed_count=%s", checked.safety_failed_count)
        await ctx.send_message(checked)

    @tracing.observe(name="WRITE_OUTPUT", as_type="span")
    async def write_node(
        state: SafetyCheckedQuestions,
        ctx: WorkflowContext[WrittenQuestions, Never],
    ) -> None:
        written = nodes.write_output_node(state)
        tracing.update_current_generation(
            metadata={
                "rows_written": written.rows_written,
                "output_path": written.output_path,
                "metadata_path": written.metadata_path,
                "log_path": written.log_path,
            }
        )
        await ctx.send_message(written)

    async def done_node(
        state: WrittenQuestions,
        ctx: WorkflowContext[Never, WrittenQuestions],
    ) -> None:
        await ctx.yield_output(state)

    load = FunctionExecutor(load_node, id="LOAD_CLAUSE_SPANS")
    filter_node = FunctionExecutor(filter_node_func, id="FILTER_RECORDS")
    generate = FunctionExecutor(generate_node, id="GENERATE_MINIMAL_QUESTIONS")
    safety = FunctionExecutor(safety_node, id="SAFETY_CHECK")
    write = FunctionExecutor(write_node, id="WRITE_OUTPUT")
    done = FunctionExecutor(done_node, id="DONE")

    return (
        WorkflowBuilder(
            start_executor=load,
            name="contract-question-agent-v0.2-minimal-e2e",
            description="Linear CUAD clause span to verification question JSONL workflow.",
            output_executors=[done],
        )
        .add_edge(load, filter_node)
        .add_edge(filter_node, generate)
        .add_edge(generate, safety)
        .add_edge(safety, write)
        .add_edge(write, done)
        .build()
    )
