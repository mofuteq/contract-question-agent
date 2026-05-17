"""GENERATE_MINIMAL_QUESTIONS node logic."""

from __future__ import annotations

from contract_question_agent.schemas import (
    GeneratedQuestions,
    QuestionModelClient,
    ReflectedQuestions,
    ScopedClauseSpans,
)


async def generate_minimal_questions_node(
    state: ScopedClauseSpans | ReflectedQuestions,
    model_client: QuestionModelClient,
) -> GeneratedQuestions:
    outputs = [
        await model_client.generate(
            record,
            regeneration_guidance=state.regeneration_guidance or None,
        )
        for record in state.records
    ]
    return GeneratedQuestions(
        request=state.request,
        records=state.records,
        outputs=outputs,
        rows_read=state.rows_read,
        rows_filtered=state.rows_filtered,
        rows_in_scope=state.rows_in_scope,
        rows_out_of_scope=state.rows_out_of_scope,
        rows_generated=len(outputs),
        scope_status_counts=state.scope_status_counts,
        out_of_scope_reasons=state.out_of_scope_reasons,
        regeneration_count=state.regeneration_count,
        regeneration_guidance=state.regeneration_guidance,
    )
