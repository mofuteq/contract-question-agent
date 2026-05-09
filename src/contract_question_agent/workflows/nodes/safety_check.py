"""SAFETY_CHECK node logic."""

from __future__ import annotations

from contract_question_agent.safety import apply_safety_check
from contract_question_agent.schemas import GeneratedQuestions, SafetyCheckedQuestions


def safety_check_node(state: GeneratedQuestions) -> SafetyCheckedQuestions:
    outputs = [apply_safety_check(output) for output in state.outputs]
    return SafetyCheckedQuestions(request=state.request, outputs=outputs)
