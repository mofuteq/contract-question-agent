from __future__ import annotations

import json
from pathlib import Path

from contract_question_agent.cli_generate_questions import main
from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import DEFAULT_OPENROUTER_MODEL
from contract_question_agent.schemas import VerificationQuestionOutput


def _span(
    contract_id: str = "C1",
    clause_type: str = "Non-Compete",
) -> ClauseSpanRecord:
    evidence_text = "Employee will not compete."
    return ClauseSpanRecord(
        contract_id=contract_id,
        source_file=f"{contract_id}.txt",
        clause_type=clause_type,
        evidence_text=evidence_text,
        start_char=0,
        end_char=len(evidence_text),
        label_present=True,
    )


def _write_input(path: Path, records: list[ClauseSpanRecord]) -> None:
    path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )


def _run_cli(
    input_path: Path,
    output_dir: Path,
    *,
    run_id: str = "test-run",
    extra_args: list[str] | None = None,
) -> Path:
    args = [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    ]
    if extra_args:
        args.extend(extra_args)
    main(args)
    return output_dir / run_id


def _read_one_output(run_dir: Path) -> VerificationQuestionOutput:
    return VerificationQuestionOutput.model_validate_json(
        (run_dir / "verification_questions.jsonl").read_text(encoding="utf-8").strip()
    )


def test_cli_dry_run_writes_run_directory_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(
        input_path,
        [
            _span("C1", "Non-Compete"),
            _span("C2", "Governing Law"),
        ],
    )

    def fail_if_network_client_is_built(*args, **kwargs):
        raise AssertionError("OpenRouter client should not be used in dry-run mode.")

    monkeypatch.setattr(
        "contract_question_agent.cli_generate_questions.OpenRouterQuestionClient",
        fail_if_network_client_is_built,
    )

    run_dir = _run_cli(
        input_path,
        output_dir,
        extra_args=["--clause-type", "Non-Compete", "--limit", "1", "--dry-run"],
    )

    output_path = run_dir / "verification_questions.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0].contract_id == "C1"
    assert rows[0].safety_status == "passed"
    assert metadata["run_id"] == "test-run"
    assert metadata["input_path"] == str(input_path)
    assert metadata["output_path"] == str(output_path)
    assert metadata["metadata_path"] == str(metadata_path)
    assert metadata["clause_type"] == "Non-Compete"
    assert metadata["contract_id"] is None
    assert metadata["limit"] == 1
    assert metadata["offset"] == 0
    assert metadata["model_name"] == DEFAULT_OPENROUTER_MODEL
    assert metadata["dry_run"] is True
    assert metadata["rows_written"] == 1
    assert "T" in metadata["created_at"]


def test_cli_fails_when_run_directory_exists(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    run_dir = output_dir / "already-there"
    _write_input(input_path, [_span()])
    run_dir.mkdir(parents=True)

    try:
        _run_cli(
            input_path,
            output_dir,
            run_id="already-there",
            extra_args=["--dry-run"],
        )
    except SystemExit as exc:
        assert str(exc) == f"Run directory already exists: {run_dir}"
    else:
        raise AssertionError("Expected SystemExit for an existing run directory.")


def test_cli_without_dry_run_requires_openrouter_key(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    try:
        _run_cli(input_path, output_dir, extra_args=[])
    except SystemExit as exc:
        assert str(exc) == "OPENROUTER_API_KEY is required unless --dry-run is set."
    else:
        raise AssertionError("Expected SystemExit for missing OPENROUTER_API_KEY.")
    assert not (output_dir / "test-run").exists()


def test_cli_dry_run_uses_openrouter_model_env(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")

    run_dir = _run_cli(input_path, output_dir, extra_args=["--dry-run"])

    assert _read_one_output(run_dir).model_name == "env-model"


def test_cli_dry_run_uses_dotenv_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])
    (tmp_path / ".env").write_text("OPENROUTER_MODEL=dotenv-model\n", encoding="utf-8")

    run_dir = _run_cli(input_path, output_dir, extra_args=["--dry-run"])

    assert _read_one_output(run_dir).model_name == "dotenv-model"


def test_cli_env_model_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])
    (tmp_path / ".env").write_text(
        "OPENROUTER_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    run_dir = _run_cli(input_path, output_dir, extra_args=["--dry-run"])

    assert _read_one_output(run_dir).model_name == "env-model"


def test_cli_model_arg_takes_precedence_over_env_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])
    (tmp_path / ".env").write_text(
        "OPENROUTER_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    run_dir = _run_cli(
        input_path,
        output_dir,
        extra_args=["--model", "cli-model", "--dry-run"],
    )

    assert _read_one_output(run_dir).model_name == "cli-model"


def test_default_openrouter_model_is_gemini_3_flash_preview():
    assert DEFAULT_OPENROUTER_MODEL == "google/gemini-3-flash-preview"
