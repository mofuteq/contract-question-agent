# Standalone MCP Clause Review Hints Server

This MCP server exposes a single tool that returns generic, clause-type-specific candidate review lenses for supported contract clause types. It is intended as a standalone boundary for clause review context without changing the current generation workflow.

## Run

```bash
uv run python -m contract_question_agent.mcp.server
```

## Inspect Locally

For development inspection with the official MCP CLI:

```bash
uv run mcp dev src/contract_question_agent/mcp/server.py
```

## Current Design

```text
MCP:
  provides candidate review lenses

Application:
  deterministically retrieves hints when enabled

Jinja:
  injects candidates into the system prompt

LLM:
  selects relevant lenses and generates verification questions

Langfuse:
  records lookup attempted/found and hint counts
```

This is not autonomous model tool calling. The model does not decide when
to call the MCP server, and the MCP server does not orchestrate the
workflow. MCP is node-internal context retrieval: when explicitly enabled,
the application performs one deterministic lookup before the existing
OpenRouter/MAF generation call and passes any found candidates into the
Jinja-rendered system prompt.

The LangGraph topology remains unchanged. The retrieved candidates are
advisory review lenses, not legal conclusions, and the model is instructed
to select only lenses grounded in the clause text.

Task-level behavior is documented in
`src/contract_question_agent/skills/contract_verification_questions/skill.md`.

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

MCP access is node-internal retrieval for the existing MAF generation path,
not workflow orchestration. The application performs the lookup
deterministically when enabled, then passes found hints into the system
prompt as candidate review lenses. The model should select only relevant
hints and ignore unsupported or irrelevant ones.

## What This Does Not Do Yet

This design does not add retry or fallback loops, Docker setup, separate
containers, HTTP MCP transport, or autonomous model tool calling.
