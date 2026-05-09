from __future__ import annotations

import json
from pathlib import Path

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent.schemas import (
    GenerateQuestionsRequest,
    LegalReviewQuestion,
    VerificationQuestion,
    VerificationQuestionOutput,
)
from contract_question_agent.workflows import run_workflow
from contract_question_agent.workflows.nodes.filter_records import filter_clause_spans


class FakeQuestionClient:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        self.call_count += 1
        return VerificationQuestionOutput(
            contract_id=record.contract_id,
            clause_type=record.clause_type,
            evidence_text=record.evidence_text,
            unknowns=["Unknown business context."],
            decision_risks=["Potential operational impact."],
            legal_review_questions=[
                LegalReviewQuestion(
                    question=f"Review {record.clause_type}?",
                    reason="Professional context is needed.",
                )
            ],
            verification_questions=[
                VerificationQuestion(
                    question="What facts should be verified?",
                    why_it_matters="Facts ground the review.",
                )
            ],
            suggested_next_step="Discuss with a qualified professional.",
            safety_disclaimer="",
            safety_status="unchecked",
            safety_warnings=[],
            model_name=self.model_name,
        )


def _span(
    contract_id: str,
    clause_type: str,
    evidence_text: str = "Clause evidence.",
) -> ClauseSpanRecord:
    return ClauseSpanRecord(
        contract_id=contract_id,
        source_file=f"{contract_id}.txt",
        clause_type=clause_type,
        evidence_text=evidence_text,
        start_char=0,
        end_char=len(evidence_text),
        label_present=True,
    )


def _write_spans(path: Path, records: list[ClauseSpanRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def test_filter_records_is_deterministic_for_cli_args():
    records = [
        _span("C1", "Non-Compete"),
        _span("C2", "Non-Compete"),
        _span("C2", "Governing Law"),
        _span("C3", "Non-Compete"),
    ]

    assert filter_clause_spans(records, clause_type="Non-Compete", offset=1, limit=2) == [
        records[1],
        records[3],
    ]
    assert filter_clause_spans(records, contract_id="C2") == [records[1], records[2]]
    assert filter_clause_spans(
        records,
        clause_type="Governing Law",
        contract_id="C2",
        limit=1,
    ) == [records[2]]


def test_run_workflow_limit_one_calls_generate_once_and_writes_one_row(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    _write_spans(
        input_path,
        [
            _span("C1", "Non-Compete", "Do not compete."),
            _span("C2", "Governing Law", "Delaware law."),
            _span("C3", "Non-Compete", "Do not solicit customers."),
        ],
    )
    request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        clause_type="Non-Compete",
        limit=1,
        offset=1,
        model_name="fake-model",
        dry_run=True,
    )

    fake_client = FakeQuestionClient()
    result = run_workflow(request, model_client=fake_client)

    assert result.rows_written == 1
    assert fake_client.call_count == 1
    rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0].contract_id == "C3"
    assert rows[0].safety_status == "passed"
    assert rows[0].safety_disclaimer == SAFETY_DISCLAIMER


def test_run_workflow_limit_three_calls_generate_three_times(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    _write_spans(
        input_path,
        [
            _span("C1", "Non-Compete", "Do not compete."),
            _span("C2", "Non-Compete", "Do not solicit customers."),
            _span("C3", "Non-Compete", "Do not use confidential information."),
            _span("C4", "Governing Law", "Delaware law."),
        ],
    )
    request = GenerateQuestionsRequest(
        input_path=input_path,
        output_path=output_path,
        clause_type="Non-Compete",
        limit=3,
        model_name="fake-model",
        dry_run=True,
    )

    fake_client = FakeQuestionClient()
    result = run_workflow(request, model_client=fake_client)

    assert result.rows_written == 3
    assert fake_client.call_count == 3
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 3
