"""REFLECT_AGAINST_SKILL_THESIS node logic."""

from __future__ import annotations

from contract_question_agent.schemas import (
    GeneratedQuestions,
    QuestionModelClient,
    ReflectedQuestions,
    ReflectionResult,
)

MAX_REGENERATION_COUNT = 1


async def reflect_against_skill_thesis_node(
    state: GeneratedQuestions,
    model_client: QuestionModelClient,
) -> ReflectedQuestions:
    reflection_results = [await model_client.reflect(output) for output in state.outputs]
    failed_results = [
        result for result in reflection_results if result.status == "failed"
    ]
    regeneration_requested = (
        bool(failed_results) and state.regeneration_count < MAX_REGENERATION_COUNT
    )
    regeneration_guidance = (
        _combined_regeneration_guidance(failed_results)
        if regeneration_requested
        else ""
    )
    return ReflectedQuestions(
        request=state.request,
        records=state.records,
        outputs=state.outputs,
        reflection_results=reflection_results,
        rows_read=state.rows_read,
        rows_filtered=state.rows_filtered,
        rows_in_scope=state.rows_in_scope,
        rows_out_of_scope=state.rows_out_of_scope,
        rows_generated=state.rows_generated,
        scope_status_counts=state.scope_status_counts,
        out_of_scope_reasons=state.out_of_scope_reasons,
        regeneration_count=(
            state.regeneration_count + 1
            if regeneration_requested
            else state.regeneration_count
        ),
        regeneration_guidance=regeneration_guidance,
        regeneration_requested=regeneration_requested,
    )


def _combined_regeneration_guidance(results: list[ReflectionResult]) -> str:
    guidance: list[str] = []
    for result in results:
        if result.regeneration_guidance:
            guidance.append(result.regeneration_guidance)
        for violation in result.violations:
            guidance.append(
                f"{violation.thesis}: {violation.problem} "
                f"Rewrite guidance: {violation.rewrite_guidance}"
            )
    return " ".join(guidance)
