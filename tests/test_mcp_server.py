from __future__ import annotations

import asyncio
import json
import re

from contract_question_agent.mcp.server import (
    format_clause_review_hints_response,
    mcp,
)


def test_mcp_server_exposes_exactly_one_tool():
    tools = asyncio.run(mcp.list_tools())

    assert [tool.name for tool in tools] == ["lookup_clause_review_hints"]


def test_known_clause_type_returns_found_true():
    response = format_clause_review_hints_response("Non-Compete")

    assert response["found"] is True
    assert response["clause_type"] == "Non-Compete"
    assert response["risk_lens"]
    assert response["common_unknowns"]
    assert response["question_categories"]
    assert response["review_hints"]


def test_unknown_clause_type_returns_found_false():
    response = format_clause_review_hints_response("Governing Law")

    assert response == {
        "found": False,
        "clause_type": "Governing Law",
        "risk_lens": None,
        "common_unknowns": [],
        "question_categories": [],
        "review_hints": [],
    }


def test_mcp_outputs_contain_no_legal_judgment_wording():
    forbidden_patterns = [
        r"\blegal\b",
        r"\billegal\b",
        r"\benforceable\b",
        r"\bunenforceable\b",
        r"\bvalid\b",
        r"\binvalid\b",
        r"\bsign\b",
        r"\bdo not sign\b",
    ]
    responses = [
        format_clause_review_hints_response("Non-Compete"),
        format_clause_review_hints_response("Change of Control"),
        format_clause_review_hints_response("Assignment"),
    ]
    response_text = json.dumps(responses, ensure_ascii=False).lower()

    assert all(
        re.search(pattern, response_text) is None for pattern in forbidden_patterns
    )
