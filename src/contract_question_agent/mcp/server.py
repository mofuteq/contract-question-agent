"""Standalone MCP server exposing clause review hints."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from contract_question_agent.clause_hints.catalog import (
    lookup_clause_review_hints as lookup_catalog_hints,
)

mcp = FastMCP("contract-question-agent-clause-hints", json_response=True)


def format_clause_review_hints_response(clause_type: str) -> dict[str, Any]:
    hints = lookup_catalog_hints(clause_type)
    if hints is None:
        return {
            "found": False,
            "clause_type": clause_type,
            "risk_lens": None,
            "common_unknowns": [],
            "question_categories": [],
            "review_hints": [],
        }

    return {"found": True, **hints.model_dump()}


@mcp.tool()
def lookup_clause_review_hints(clause_type: str) -> dict[str, Any]:
    return format_clause_review_hints_response(clause_type)


if __name__ == "__main__":
    mcp.run()
