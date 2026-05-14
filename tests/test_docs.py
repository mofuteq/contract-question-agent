from __future__ import annotations

from pathlib import Path


def test_mcp_docs_describe_deterministic_candidate_lens_retrieval():
    docs = Path("docs/mcp.md").read_text(encoding="utf-8")

    assert "provides candidate review lenses" in docs
    assert "deterministically retrieves hints when enabled" in docs
    assert "injects candidates into the system prompt" in docs
    assert "selects relevant lenses and generates verification questions" in docs
    assert "not autonomous model tool calling" in docs
    assert "node-internal context retrieval" in docs


def test_contract_verification_question_skill_docs_exist():
    skill_doc = Path(
        "src/contract_question_agent/skills/contract_verification_questions/skill.md"
    )

    assert skill_doc.exists()

    docs = skill_doc.read_text(encoding="utf-8")

    assert "MCP provides candidate review lenses" in docs
    assert "selected_review_lenses" in docs
    assert "mcp_clause_review_hints" in docs
    assert "Do not provide legal advice" in docs
    assert "Graceful Degradation" in docs
