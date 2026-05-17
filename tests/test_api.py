from __future__ import annotations

import json

from fastapi.testclient import TestClient

from contract_question_agent.api.app import create_app


def test_health_endpoint_returns_ok():
    client = TestClient(create_app(load_env=False))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_runs_dry_run_generates_questions_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )

    def fail_if_network_client_is_built(*args, **kwargs):
        raise AssertionError("OpenRouter client should not be used in dry-run mode.")

    monkeypatch.setattr(
        "contract_question_agent.api.adapter.OpenRouterQuestionClient",
        fail_if_network_client_is_built,
    )

    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "Employee will not compete for one year after termination.",
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["run_id"]
    assert payload["rows_read"] == 1
    assert payload["rows_filtered"] == 1
    assert payload["rows_in_scope"] == 1
    assert payload["rows_out_of_scope"] == 0
    assert payload["rows_generated"] == 1
    assert payload["rows_written"] == 1
    assert payload["dry_run"] is True
    assert payload["verification_questions"]
    assert payload["verification_questions"][0]["contract_id"] == "demo-contract"
    assert payload["safety_status"] == "passed"
    assert "selected_review_lenses" in payload
    assert "input_path" not in payload

    run_dir = tmp_path / "api-runs" / payload["run_id"]
    input_path = run_dir / "input_clause_spans.jsonl"
    output_path = run_dir / "verification_questions.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    log_path = run_dir / "run.log"
    assert payload["output_path"] == str(output_path)
    assert payload["metadata_path"] == str(metadata_path)
    assert payload["log_path"] == str(log_path)
    assert input_path.exists()
    assert output_path.exists()
    assert metadata_path.exists()
    assert log_path.exists()

    input_rows = input_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8")
    assert len(input_rows) == 1
    assert json.loads(input_rows[0])["contract_id"] == "demo-contract"
    assert metadata["run_id"] == payload["run_id"]
    assert metadata["rows_written"] == 1
    assert "run_id=" in log_text
    assert "rows_written=1" in log_text
    assert "OPENROUTER_API_KEY" not in log_text
    assert "Employee will not compete" not in log_text


def test_post_runs_rejects_public_local_path_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )
    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "Employee will not compete for one year after termination.",
            "input_path": "data/cuad/processed/clause_spans.jsonl",
            "output_dir": "data/cuad/runs",
            "dry_run": True,
        },
    )

    assert response.status_code == 422


def test_post_runs_empty_evidence_is_out_of_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )

    def fail_if_network_client_is_built(*args, **kwargs):
        raise AssertionError("OpenRouter client should not be used in dry-run mode.")

    monkeypatch.setattr(
        "contract_question_agent.api.adapter.OpenRouterQuestionClient",
        fail_if_network_client_is_built,
    )

    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "",
            "dry_run": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["rows_read"] == 1
    assert payload["rows_filtered"] == 1
    assert payload["rows_in_scope"] == 0
    assert payload["rows_out_of_scope"] == 1
    assert payload["rows_generated"] == 0
    assert payload["rows_written"] == 0
    assert payload["verification_questions"] == []
    assert payload["safety_status"] is None


def test_post_runs_without_dry_run_requires_openrouter_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "Employee will not compete for one year after termination.",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "OPENROUTER_API_KEY is required unless --dry-run is set."
    }


def test_verification_questions_endpoint_is_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )
    client = TestClient(create_app(load_env=False))

    response = client.post(
        "/verification-questions",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "Employee will not compete for one year after termination.",
            "dry_run": True,
        },
    )

    assert response.status_code == 404
