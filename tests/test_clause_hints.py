from __future__ import annotations

from contract_question_agent.clause_hints.catalog import lookup_clause_review_hints


def test_non_compete_returns_hints():
    hints = lookup_clause_review_hints("Non-Compete")

    assert hints is not None
    assert hints.clause_type == "Non-Compete"
    assert hints.risk_lens
    assert hints.common_unknowns
    assert hints.question_categories
    assert hints.review_hints


def test_change_of_control_returns_hints():
    hints = lookup_clause_review_hints("Change of Control")

    assert hints is not None
    assert hints.clause_type == "Change of Control"
    assert hints.risk_lens
    assert hints.common_unknowns
    assert hints.question_categories
    assert hints.review_hints


def test_assignment_returns_hints():
    hints = lookup_clause_review_hints("Assignment")

    assert hints is not None
    assert hints.clause_type == "Assignment"
    assert hints.risk_lens
    assert hints.common_unknowns
    assert hints.question_categories
    assert hints.review_hints


def test_unknown_clause_type_returns_none():
    assert lookup_clause_review_hints("Governing Law") is None
