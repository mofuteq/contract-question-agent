"""IS_IN_SCOPE node logic."""

from __future__ import annotations

from collections import Counter

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.schemas import (
    FilteredClauseSpans,
    ScopeCheckResult,
    ScopedClauseSpans,
)

MIN_USEFUL_EVIDENCE_CHARS = 10


def is_record_in_scope(record: ClauseSpanRecord) -> ScopeCheckResult:
    evidence_text = record.evidence_text.strip()
    clause_type = record.clause_type.strip()
    if not evidence_text:
        return ScopeCheckResult(status="out_of_scope", reason="empty_evidence_text")
    if not clause_type:
        return ScopeCheckResult(status="out_of_scope", reason="empty_clause_type")
    if len(evidence_text) < MIN_USEFUL_EVIDENCE_CHARS:
        return ScopeCheckResult(status="out_of_scope", reason="evidence_text_too_short")
    return ScopeCheckResult(status="in_scope")


def is_in_scope_node(state: FilteredClauseSpans) -> ScopedClauseSpans:
    scoped_records: list[ClauseSpanRecord] = []
    scope_results: list[ScopeCheckResult] = []
    for record in state.records:
        result = is_record_in_scope(record)
        scope_results.append(result)
        if result.status == "in_scope":
            scoped_records.append(record)

    status_counts = Counter(result.status for result in scope_results)
    out_of_scope_reasons = Counter(
        result.reason
        for result in scope_results
        if result.status == "out_of_scope" and result.reason
    )
    return ScopedClauseSpans(
        request=state.request,
        records=scoped_records,
        scope_results=scope_results,
        rows_read=state.rows_read,
        rows_filtered=state.rows_filtered,
        rows_in_scope=len(scoped_records),
        rows_out_of_scope=status_counts["out_of_scope"],
        scope_status_counts=dict(status_counts),
        out_of_scope_reasons=dict(out_of_scope_reasons),
        regeneration_count=state.regeneration_count,
        regeneration_guidance=state.regeneration_guidance,
    )
