"""FILTER_RECORDS node logic."""

from __future__ import annotations

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.schemas import FilteredClauseSpans, LoadedClauseSpans


def filter_records_node(state: LoadedClauseSpans) -> FilteredClauseSpans:
    records = filter_clause_spans(
        state.records,
        clause_type=state.request.clause_type,
        contract_id=state.request.contract_id,
        limit=state.request.limit,
        offset=state.request.offset,
    )
    return FilteredClauseSpans(
        request=state.request,
        records=records,
        rows_read=state.rows_read,
        rows_filtered=len(records),
    )


def filter_clause_spans(
    records: list[ClauseSpanRecord],
    *,
    clause_type: str | None = None,
    contract_id: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[ClauseSpanRecord]:
    filtered = [
        record
        for record in records
        if (clause_type is None or record.clause_type == clause_type)
        and (contract_id is None or record.contract_id == contract_id)
    ]
    if offset:
        filtered = filtered[offset:]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered
