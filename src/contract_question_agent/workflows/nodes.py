"""Node internals for the v0.2 Microsoft Agent Framework workflow."""

from __future__ import annotations

import json
from pathlib import Path

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import apply_safety_check
from contract_question_agent.schemas import (
    FilteredClauseSpans,
    GenerateQuestionsRequest,
    GeneratedQuestions,
    LoadedClauseSpans,
    QuestionModelClient,
    SafetyCheckedQuestions,
    WrittenQuestions,
)


def load_clause_spans(request: GenerateQuestionsRequest) -> LoadedClauseSpans:
    records = read_clause_spans_jsonl(request.input_path)
    return LoadedClauseSpans(request=request, records=records)


def filter_records(state: LoadedClauseSpans) -> FilteredClauseSpans:
    records = filter_clause_spans(
        state.records,
        clause_type=state.request.clause_type,
        contract_id=state.request.contract_id,
        limit=state.request.limit,
        offset=state.request.offset,
    )
    return FilteredClauseSpans(request=state.request, records=records)


async def generate_minimal_questions(
    state: FilteredClauseSpans,
    model_client: QuestionModelClient,
) -> GeneratedQuestions:
    outputs = [await model_client.generate(record) for record in state.records]
    return GeneratedQuestions(request=state.request, outputs=outputs)


def safety_check(state: GeneratedQuestions) -> SafetyCheckedQuestions:
    outputs = [apply_safety_check(output) for output in state.outputs]
    return SafetyCheckedQuestions(request=state.request, outputs=outputs)


def write_output(state: SafetyCheckedQuestions) -> WrittenQuestions:
    write_verification_questions_jsonl(state.request.output_path, state.outputs)
    return WrittenQuestions(
        output_path=state.request.output_path,
        rows_written=len(state.outputs),
        outputs=state.outputs,
    )


def done(state: WrittenQuestions) -> WrittenQuestions:
    return state


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


def write_verification_questions_jsonl(path: Path, outputs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
