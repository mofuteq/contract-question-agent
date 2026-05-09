"""WRITE_OUTPUT node logic."""

from __future__ import annotations

import json
from pathlib import Path

from contract_question_agent.schemas import SafetyCheckedQuestions, WrittenQuestions


def write_output_node(state: SafetyCheckedQuestions) -> WrittenQuestions:
    write_verification_questions_jsonl(state.request.output_path, state.outputs)
    return WrittenQuestions(
        output_path=state.request.output_path,
        rows_written=len(state.outputs),
        outputs=state.outputs,
    )


def write_verification_questions_jsonl(path: Path, outputs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
