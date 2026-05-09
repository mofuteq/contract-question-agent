"""LOAD_CLAUSE_SPANS node logic."""

from __future__ import annotations

from pathlib import Path

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.schemas import GenerateQuestionsRequest, LoadedClauseSpans


def load_clause_spans_node(request: GenerateQuestionsRequest) -> LoadedClauseSpans:
    records = read_clause_spans_jsonl(request.input_path)
    return LoadedClauseSpans(
        request=request,
        records=records,
        rows_read=len(records),
    )


def read_clause_spans_jsonl(path: Path) -> list[ClauseSpanRecord]:
    records: list[ClauseSpanRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ClauseSpanRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid clause span JSONL at line {line_number}") from exc
    return records
