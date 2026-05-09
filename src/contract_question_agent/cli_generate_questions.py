"""CLI for v0.2 poor E2E verification-question generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterQuestionClient,
)
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent.schemas import (
    GenerateQuestionsRequest,
    LegalReviewQuestion,
    VerificationQuestion,
    VerificationQuestionOutput,
)
from contract_question_agent.workflows import run_linear_workflow


class DryRunQuestionClient:
    """Deterministic offline model replacement for wiring tests and demos."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    async def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        return VerificationQuestionOutput(
            contract_id=record.contract_id,
            clause_type=record.clause_type,
            evidence_text=record.evidence_text,
            unknowns=["The business context for this clause is not provided."],
            decision_risks=[
                "The clause may affect obligations or negotiation priorities."
            ],
            legal_review_questions=[
                LegalReviewQuestion(
                    question=(
                        f"What should be reviewed about this {record.clause_type} "
                        "clause?"
                    ),
                    reason="The answer depends on the full agreement and applicable law.",
                )
            ],
            verification_questions=[
                VerificationQuestion(
                    question=(
                        "What facts or documents would help verify the practical "
                        "effect of this clause?"
                    ),
                    why_it_matters=(
                        "Grounding review in facts helps a professional assess the "
                        "clause in context."
                    ),
                )
            ],
            suggested_next_step=(
                "Discuss these clause-grounded questions with a qualified legal professional."
            ),
            safety_disclaimer=SAFETY_DISCLAIMER,
            safety_status="unchecked",
            safety_warnings=[],
            model_name=self.model_name,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate v0.2 verification questions from CUAD clause spans."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clause-type")
    parser.add_argument("--contract-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    request = GenerateQuestionsRequest(
        input_path=args.input,
        output_path=args.output,
        clause_type=args.clause_type,
        contract_id=args.contract_id,
        limit=args.limit,
        offset=args.offset,
        model_name=args.model or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        model_client = DryRunQuestionClient(request.model_name)
    else:
        try:
            model_client = OpenRouterQuestionClient(model_name=request.model_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    result = run_linear_workflow(request, model_client=model_client)
    print(f"Wrote {result.rows_written} rows to {result.output_path}")


if __name__ == "__main__":
    main()
