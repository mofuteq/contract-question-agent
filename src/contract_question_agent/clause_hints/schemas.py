"""Schemas for clause review hints."""

from __future__ import annotations

from pydantic import BaseModel


class ClauseReviewHints(BaseModel):
    clause_type: str
    risk_lens: str
    common_unknowns: list[str]
    question_categories: list[str]
    review_hints: list[str]
