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
