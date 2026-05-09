"""Workflow assembly for the v0.2 Microsoft Agent Framework graph."""

from __future__ import annotations

import asyncio
from typing import Never

from agent_framework import FunctionExecutor, WorkflowBuilder, WorkflowContext

from contract_question_agent.schemas import (
    FilteredClauseSpans,
    GenerateQuestionsRequest,
    GeneratedQuestions,
    QuestionModelClient,
    WrittenQuestions,
)
from contract_question_agent.workflows import nodes


def run_linear_workflow(
    request: GenerateQuestionsRequest,
    *,
    model_client: QuestionModelClient,
) -> WrittenQuestions:
    """Run LOAD -> FILTER -> GENERATE -> SAFETY -> WRITE -> DONE."""
    workflow = build_linear_workflow(model_client=model_client)
    result = asyncio.run(workflow.run(request))
    outputs = result.get_outputs()
    if len(outputs) != 1 or not isinstance(outputs[0], WrittenQuestions):
        raise RuntimeError("Microsoft linear workflow did not produce one DONE output.")
    return outputs[0]


def build_linear_workflow(*, model_client: QuestionModelClient):
    async def generate_node(
        state: FilteredClauseSpans,
        ctx: WorkflowContext[GeneratedQuestions, Never],
    ) -> None:
        await nodes.generate_minimal_questions(state, ctx, model_client)

    load = FunctionExecutor(nodes.load_clause_spans, id="LOAD_CLAUSE_SPANS")
    filter_node = FunctionExecutor(nodes.filter_records, id="FILTER_RECORDS")
    generate = FunctionExecutor(generate_node, id="GENERATE_MINIMAL_QUESTIONS")
    safety = FunctionExecutor(nodes.safety_check, id="SAFETY_CHECK")
    write = FunctionExecutor(nodes.write_output, id="WRITE_OUTPUT")
    done = FunctionExecutor(nodes.done, id="DONE")

    return (
        WorkflowBuilder(
            start_executor=load,
            name="contract-question-agent-v0.2-poor-e2e",
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
