"""Local catalog of generic clause-type review hints."""

from __future__ import annotations

from contract_question_agent.clause_hints.schemas import ClauseReviewHints


_CATALOG: dict[str, ClauseReviewHints] = {
    "Non-Compete": ClauseReviewHints(
        clause_type="Non-Compete",
        risk_lens=(
            "Review the practical scope of restricted activities, duration, geography, "
            "covered parties, and exceptions."
        ),
        common_unknowns=[
            "Which activities or business lines are covered by the restriction.",
            "When the restriction starts and how long it lasts.",
            "Which locations, customers, affiliates, or roles are included.",
            "Whether carve-outs apply for existing work, passive holdings, or consent.",
        ],
        question_categories=[
            "Restricted activities",
            "Time period",
            "Geographic or customer scope",
            "Exceptions and approval process",
        ],
        review_hints=[
            "Check whether key terms are defined consistently across the agreement.",
            "Compare the restriction against the person or entity's expected activities.",
            "Look for notice, waiver, or consent mechanics tied to the restriction.",
            "Identify related clauses that may expand or narrow the same obligation.",
        ],
    ),
    "Change of Control": ClauseReviewHints(
        clause_type="Change of Control",
        risk_lens=(
            "Review triggers, required notices or consents, timing, and consequences "
            "when ownership or control changes."
        ),
        common_unknowns=[
            "Which transactions count as a change of control.",
            "Whether indirect ownership changes or parent-level events are included.",
            "What notice, approval, or reporting steps are required.",
            "What happens to fees, termination rights, or obligations after the event.",
        ],
        question_categories=[
            "Trigger definition",
            "Notice and consent",
            "Timing requirements",
            "Post-event consequences",
        ],
        review_hints=[
            "Check whether control thresholds are numeric, event-based, or undefined.",
            "Compare this clause with assignment, termination, and notice provisions.",
            "Identify whether consequences are automatic or require a party action.",
            "Look for cure periods, exceptions, or pre-approved transaction types.",
        ],
    ),
    "Assignment": ClauseReviewHints(
        clause_type="Assignment",
        risk_lens=(
            "Review who may transfer rights or obligations, consent requirements, "
            "permitted transfers, and continuing responsibilities."
        ),
        common_unknowns=[
            "Which rights, duties, or agreements may be transferred.",
            "Whether consent is required before a transfer.",
            "Whether affiliates, successors, or restructuring transactions are treated differently.",
            "Whether the original party remains responsible after transfer.",
        ],
        question_categories=[
            "Transfer scope",
            "Consent mechanics",
            "Permitted transfers",
            "Continuing responsibility",
        ],
        review_hints=[
            "Check whether assignment is limited to rights, obligations, or both.",
            "Compare permitted transfer language with change-of-control language.",
            "Look for notice timing and required form of consent.",
            "Identify whether delegation, subcontracting, or novation is addressed separately.",
        ],
    ),
}


def lookup_clause_review_hints(clause_type: str) -> ClauseReviewHints | None:
    return _CATALOG.get(clause_type)
