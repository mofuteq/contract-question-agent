from __future__ import annotations

import pytest

from contract_question_agent.skills.loader import load_skill_text


def test_load_contract_verification_questions_skill_text():
    skill_text = load_skill_text("contract_verification_questions")

    assert "Contract Verification Question Skill" in skill_text
    assert "MCP provides candidate review lenses" in skill_text
    assert "Graceful Degradation" in skill_text


def test_load_unknown_skill_raises_value_error():
    with pytest.raises(ValueError, match="Unknown skill name: unknown"):
        load_skill_text("unknown")
