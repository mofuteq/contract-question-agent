from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from contract_question_agent.api.app import create_app
from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.model_client import DEFAULT_OPENROUTER_MODEL
from contract_question_agent.schemas import VerificationQuestionOutput


def _span(
    contract_id: str = "C1",
    clause_type: str = "Non-Compete",
) -> ClauseSpanRecord:
    evidence_text = "Employee will not compete for one year after termination."
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


def test_health_endpoint_returns_ok():
    client = TestClient(create_app(load_env=False))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_dry_run_executes_existing_workflow_without_network(tmp_path, monkeypatch):
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
        "contract_question_agent.api.adapter.OpenRouterQuestionClient",
        fail_if_network_client_is_built,
    )
    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/verification-questions",
        json={
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "run_id": "api-test-run",
            "clause_type": "Non-Compete",
            "limit": 1,
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    run_dir = output_dir / "api-test-run"
    output_path = run_dir / "verification_questions.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    log_path = run_dir / "run.log"
    assert payload["run_id"] == "api-test-run"
    assert payload["input_path"] == str(input_path)
    assert payload["output_path"] == str(output_path)
    assert payload["metadata_path"] == str(metadata_path)
    assert payload["log_path"] == str(log_path)
    assert payload["model_name"] == DEFAULT_OPENROUTER_MODEL
    assert payload["dry_run"] is True
    assert payload["rows_read"] == 2
    assert payload["rows_filtered"] == 1
    assert payload["rows_in_scope"] == 1
    assert payload["rows_out_of_scope"] == 0
    assert payload["rows_generated"] == 1
    assert payload["safety_failed_count"] == 0
    assert payload["rows_written"] == 1
    assert payload["outputs"][0]["contract_id"] == "C1"
    assert payload["outputs"][0]["safety_status"] == "passed"

    output_rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8")
    assert len(output_rows) == 1
    assert metadata["run_id"] == "api-test-run"
    assert metadata["rows_written"] == 1
    assert "run_id=api-test-run" in log_text
    assert "rows_written=1" in log_text
    assert "OPENROUTER_API_KEY" not in log_text
    assert "Employee will not compete" not in log_text


def test_api_returns_conflict_when_run_directory_exists(tmp_path):
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])
    (output_dir / "already-there").mkdir(parents=True)
    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/verification-questions",
        json={
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "run_id": "already-there",
            "dry_run": True,
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"Run directory already exists: {output_dir / 'already-there'}"
    }


def test_api_without_dry_run_requires_openrouter_key(tmp_path, monkeypatch):
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [])
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/verification-questions",
        json={
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "run_id": "missing-key",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "OPENROUTER_API_KEY is required unless --dry-run is set."
    }
    assert not (output_dir / "missing-key").exists()
