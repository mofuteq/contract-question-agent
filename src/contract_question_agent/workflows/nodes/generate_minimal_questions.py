"""GENERATE_MINIMAL_QUESTIONS node logic."""

from __future__ import annotations

from contract_question_agent.schemas import (
    FilteredClauseSpans,
    GeneratedQuestions,
    QuestionModelClient,
    ReflectedQuestions,
)


async def generate_minimal_questions_node(
    state: FilteredClauseSpans | ReflectedQuestions,
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
        rows_generated=len(outputs),
        regeneration_count=state.regeneration_count,
        regeneration_guidance=state.regeneration_guidance,
    )
