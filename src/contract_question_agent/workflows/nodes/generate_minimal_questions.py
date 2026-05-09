"""GENERATE_MINIMAL_QUESTIONS node logic."""

from __future__ import annotations

from contract_question_agent.schemas import (
    FilteredClauseSpans,
    GeneratedQuestions,
    QuestionModelClient,
)


async def generate_minimal_questions_node(
    state: FilteredClauseSpans,
    model_client: QuestionModelClient,
) -> GeneratedQuestions:
    outputs = [await model_client.generate(record) for record in state.records]
    return GeneratedQuestions(request=state.request, outputs=outputs)
