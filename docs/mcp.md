# Standalone MCP Clause Review Hints Server

This MCP server exposes a single tool that returns generic, clause-type-specific review hints for supported contract clause types. It is intended as a standalone boundary for future clause review context without changing the current generation workflow.

## Run

```bash
uv run python -m contract_question_agent.mcp.server
```

## Inspect Locally

For development inspection with the official MCP CLI:

```bash
uv run mcp dev src/contract_question_agent/mcp/server.py
```

## Current Tool

The server exposes exactly one tool:

- `lookup_clause_review_hints(clause_type: str)`

Supported clause types:

- `Non-Compete`
- `Change of Control`
- `Assignment`

Unknown clause types return `found: false` with empty hint lists.

## Enable During Generation

MCP hints are off by default. To retrieve generic hints from the local MCP tool before the existing OpenRouter/MAF generation call:

```bash
CONTRACT_QUESTION_USE_MCP_HINTS=true uv run contract-question-generate ...
```

MCP access is node-internal retrieval for the existing MAF generation path, not workflow orchestration. The application performs the lookup deterministically when enabled, then passes found hints into the system prompt as candidate review lenses. The model should select only relevant hints and ignore unsupported or irrelevant ones. The LangGraph workflow topology is unchanged.

## What This Does Not Do Yet

This PR does not add retry or fallback loops, Docker setup, separate containers, or HTTP MCP transport. This server does not change output JSONL rows, run metadata, or the unified Langfuse trace tree.
