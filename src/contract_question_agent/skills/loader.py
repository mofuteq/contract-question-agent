"""Load packaged task skill documentation for prompt context."""

from __future__ import annotations

from importlib.resources import files


CONTRACT_VERIFICATION_QUESTIONS_SKILL = "contract_verification_questions"
CONTRACT_VERIFICATION_QUESTIONS_SKILL_PATH = (
    "src/contract_question_agent/skills/contract_verification_questions/skill.md"
)


def load_skill_text(skill_name: str) -> str:
    if skill_name != CONTRACT_VERIFICATION_QUESTIONS_SKILL:
        raise ValueError(f"Unknown skill name: {skill_name}")

    skill_file = (
        files("contract_question_agent.skills.contract_verification_questions")
        / "skill.md"
    )
    return skill_file.read_text(encoding="utf-8").strip()
