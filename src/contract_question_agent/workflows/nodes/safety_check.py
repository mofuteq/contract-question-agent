"""SAFETY_CHECK node logic."""

from __future__ import annotations

from contract_question_agent.safety import apply_safety_check
from contract_question_agent.schemas import (
    GeneratedQuestions,
    ReflectedQuestions,
    SafetyCheckedQuestions,
)


def safety_check_node(state: GeneratedQuestions | ReflectedQuestions) -> SafetyCheckedQuestions:
    outputs = [apply_safety_check(output) for output in state.outputs]
    safety_failed_count = sum(
        1 for output in outputs if output.safety_status == "failed"
    )
    return SafetyCheckedQuestions(
        request=state.request,
        outputs=outputs,
        rows_read=state.rows_read,
        rows_filtered=state.rows_filtered,
        rows_in_scope=state.rows_in_scope,
        rows_out_of_scope=state.rows_out_of_scope,
        rows_generated=state.rows_generated,
        scope_status_counts=state.scope_status_counts,
        out_of_scope_reasons=state.out_of_scope_reasons,
        safety_failed_count=safety_failed_count,
    )
