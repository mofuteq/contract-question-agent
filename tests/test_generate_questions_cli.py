from __future__ import annotations

import json

from contract_question_agent.cli_generate_questions import main
from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import DEFAULT_OPENROUTER_MODEL
from contract_question_agent.schemas import VerificationQuestionOutput


def test_cli_dry_run_writes_jsonl_without_network(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    records = [
        ClauseSpanRecord(
            contract_id="C1",
            source_file="C1.txt",
            clause_type="Non-Compete",
            evidence_text="Employee will not compete.",
            start_char=0,
            end_char=26,
            label_present=True,
        ),
        ClauseSpanRecord(
            contract_id="C2",
            source_file="C2.txt",
            clause_type="Governing Law",
            evidence_text="Delaware law applies.",
            start_char=0,
            end_char=21,
            label_present=True,
        ),
    ]
    input_path.write_text(
        "".join(f"{record.model_dump_json()}\n" for record in records),
        encoding="utf-8",
    )

    def fail_if_network_client_is_built(*args, **kwargs):
        raise AssertionError("OpenRouter client should not be used in dry-run mode.")

    monkeypatch.setattr(
        "contract_question_agent.cli_generate_questions.OpenRouterQuestionClient",
        fail_if_network_client_is_built,
    )

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--clause-type",
            "Non-Compete",
            "--limit",
            "1",
            "--dry-run",
        ]
    )

    rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0].contract_id == "C1"
    assert rows[0].safety_status == "passed"


def test_cli_without_dry_run_requires_openrouter_key(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    input_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    try:
        main(["--input", str(input_path), "--output", str(output_path)])
    except SystemExit as exc:
        assert str(exc) == "OPENROUTER_API_KEY is required unless --dry-run is set."
    else:
        raise AssertionError("Expected SystemExit for missing OPENROUTER_API_KEY.")


def test_cli_dry_run_uses_openrouter_model_env(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    record = ClauseSpanRecord(
        contract_id="C1",
        source_file="C1.txt",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete.",
        start_char=0,
        end_char=26,
        label_present=True,
    )
    input_path.write_text(f"{record.model_dump_json()}\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    row = VerificationQuestionOutput.model_validate_json(
        output_path.read_text(encoding="utf-8").strip()
    )
    assert row.model_name == "env-model"


def test_cli_dry_run_uses_dotenv_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    record = ClauseSpanRecord(
        contract_id="C1",
        source_file="C1.txt",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete.",
        start_char=0,
        end_char=26,
        label_present=True,
    )
    input_path.write_text(f"{record.model_dump_json()}\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_MODEL=dotenv-model\n", encoding="utf-8")

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    row = VerificationQuestionOutput.model_validate_json(
        output_path.read_text(encoding="utf-8").strip()
    )
    assert row.model_name == "dotenv-model"


def test_cli_env_model_takes_precedence_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    record = ClauseSpanRecord(
        contract_id="C1",
        source_file="C1.txt",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete.",
        start_char=0,
        end_char=26,
        label_present=True,
    )
    input_path.write_text(f"{record.model_dump_json()}\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENROUTER_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    row = VerificationQuestionOutput.model_validate_json(
        output_path.read_text(encoding="utf-8").strip()
    )
    assert row.model_name == "env-model"


def test_cli_model_arg_takes_precedence_over_env_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "env-model")
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_path = tmp_path / "verification_questions.jsonl"
    record = ClauseSpanRecord(
        contract_id="C1",
        source_file="C1.txt",
        clause_type="Non-Compete",
        evidence_text="Employee will not compete.",
        start_char=0,
        end_char=26,
        label_present=True,
    )
    input_path.write_text(f"{record.model_dump_json()}\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENROUTER_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            "cli-model",
            "--dry-run",
        ]
    )

    row = VerificationQuestionOutput.model_validate_json(
        output_path.read_text(encoding="utf-8").strip()
    )
    assert row.model_name == "cli-model"


def test_default_openrouter_model_is_gemini_3_flash_preview():
    assert DEFAULT_OPENROUTER_MODEL == "google/gemini-3-flash-preview"
