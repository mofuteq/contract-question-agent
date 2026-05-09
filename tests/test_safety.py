from __future__ import annotations

from contract_question_agent.safety import SAFETY_DISCLAIMER, apply_safety_check
from contract_question_agent.schemas import VerificationQuestionOutput


def _output(text: str) -> VerificationQuestionOutput:
    return VerificationQuestionOutput(
        contract_id="C1",
        clause_type="Non-Compete",
        evidence_text="Clause text.",
        unknowns=[text],
        decision_risks=[],
        legal_review_questions=[],
        verification_questions=[],
        suggested_next_step="Discuss with a professional.",
        safety_disclaimer="",
        safety_status="unchecked",
        safety_warnings=[],
        model_name="fake-model",
    )


def _output_with_evidence(evidence_text: str) -> VerificationQuestionOutput:
    output = _output("Ask what this clause means in context.")
    return output.model_copy(update={"evidence_text": evidence_text})


def test_safety_check_passes_when_no_banned_phrase():
    checked = apply_safety_check(_output("Ask what this clause means in context."))

    assert checked.safety_status == "passed"
    assert checked.safety_warnings == []
    assert checked.safety_disclaimer == SAFETY_DISCLAIMER


def test_safety_check_flags_generated_banned_phrase_case_insensitively():
    checked = apply_safety_check(_output("YOU SHOULD SIGN."))

    assert checked.safety_status == "failed"
    assert checked.safety_warnings == ["banned phrase found: you should sign"]
    assert checked.safety_disclaimer == SAFETY_DISCLAIMER


def test_safety_check_excludes_evidence_text():
    checked = apply_safety_check(_output_with_evidence("You should sign."))

    assert checked.safety_status == "passed"
    assert checked.safety_warnings == []
