"""CLI for v0.2 minimal E2E verification-question generation."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterQuestionClient,
)
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent import tracing
from contract_question_agent.schemas import (
    GenerateQuestionsRequest,
    LegalReviewQuestion,
    VerificationQuestion,
    VerificationQuestionOutput,
)
from contract_question_agent.workflows import run_workflow


DEFAULT_OUTPUT_DIR = Path("data/cuad/runs")
OUTPUT_FILENAME = "verification_questions.jsonl"
METADATA_FILENAME = "run_metadata.json"
LOG_FILENAME = "run.log"

logger = logging.getLogger(__name__)


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--clause-type")
    parser.add_argument("--contract-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    args = build_parser().parse_args(argv)
    run_id = args.run_id or make_run_id()
    run_dir = args.output_dir / run_id
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}")
    output_path = run_dir / OUTPUT_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    log_path = run_dir / LOG_FILENAME
    model_name = args.model or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
    tracing.configure_maf_otel_if_enabled()
    if args.dry_run:
        model_client = DryRunQuestionClient(model_name)
    else:
        try:
            model_client = OpenRouterQuestionClient(model_name=model_name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    run_dir.mkdir(parents=True)
    configure_logging(log_path, verbose=args.verbose)
    logger.info("run_id=%s", run_id)
    logger.info("run_dir=%s", run_dir)
    logger.info("input_path=%s", args.input)
    logger.info("output_path=%s", output_path)
    logger.info("metadata_path=%s", metadata_path)
    logger.info("log_path=%s", log_path)
    logger.info("model_name=%s", model_name)
    logger.info("dry_run=%s", args.dry_run)
    logger.info(
        "filters clause_type=%s contract_id=%s limit=%s offset=%s",
        args.clause_type,
        args.contract_id,
        args.limit,
        args.offset,
    )
    request = GenerateQuestionsRequest(
        input_path=args.input,
        output_path=output_path,
        metadata_path=metadata_path,
        log_path=log_path,
        run_id=run_id,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        clause_type=args.clause_type,
        contract_id=args.contract_id,
        limit=args.limit,
        offset=args.offset,
        model_name=model_name,
        dry_run=args.dry_run,
    )
    trace_metadata = {
        "app": "contract-question-agent",
        "run_id": run_id,
        "input_path": args.input,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "log_path": log_path,
        "model_name": model_name,
        "dry_run": args.dry_run,
        "clause_type": args.clause_type,
        "contract_id": args.contract_id,
        "limit": args.limit,
        "offset": args.offset,
    }
    run_generator = tracing.observe(
        name="contract-question-generate",
        as_type="span",
    )(_run_generator)
    try:
        result = run_generator(request, model_client, trace_metadata)
    finally:
        tracing.flush()
    print(f"Wrote {result.rows_written} rows to {result.output_path}")


def _run_generator(
    request: GenerateQuestionsRequest,
    model_client,
    trace_metadata: dict,
):
    tracing.update_current_trace(
        name="contract-question-generate",
        session_id=request.run_id,
        metadata=trace_metadata,
        tags=["contract-question-agent", "v0.3"],
    )
    logger.info("tracing_enabled=%s", tracing.is_active())
    logger.info("langfuse_trace_id=%s", tracing.get_current_trace_id())
    logger.info("langfuse_trace_url=%s", tracing.get_current_trace_url())
    logger.info(
        "langfuse_environment=%s",
        tracing.get_tracing_environment(),
    )
    result = run_workflow(request, model_client=model_client)
    logger.info("rows_read=%s", result.rows_read)
    logger.info("rows_filtered=%s", result.rows_filtered)
    logger.info("rows_generated=%s", result.rows_generated)
    logger.info("safety_failed_count=%s", result.safety_failed_count)
    logger.info("rows_written=%s", result.rows_written)
    return result


def make_run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def configure_logging(log_path: Path, *, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.WARNING)
    logging.getLogger("contract_question_agent").setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)


if __name__ == "__main__":
    main()
