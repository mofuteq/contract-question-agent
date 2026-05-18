from __future__ import annotations

from fastapi.testclient import TestClient

from contract_question_agent.api.app import create_app


def test_ag_ui_runs_streams_run_events(tmp_path, monkeypatch):
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

    with client.stream(
        "POST",
        "/ag-ui/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "Employee will not compete for one year after termination.",
            "dry_run": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: RUN_STARTED" in body
    assert "event: STEP_STARTED" in body
    assert "event: STEP_FINISHED" in body
    assert "event: STATE_SNAPSHOT" in body
    assert "event: RUN_FINISHED" in body
    assert "event: RUN_ERROR" not in body


def test_ag_ui_run_snapshot_does_not_expose_evidence_text(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )

    client = TestClient(create_app(load_env=False))
    evidence_text = "Employee will not compete for one year after termination."

    with client.stream(
        "POST",
        "/ag-ui/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": evidence_text,
            "dry_run": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert evidence_text not in body


def test_ag_ui_empty_evidence_streams_out_of_scope_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )

    client = TestClient(create_app(load_env=False))

    with client.stream(
        "POST",
        "/ag-ui/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "",
            "dry_run": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert "event: RUN_FINISHED" in body
    assert '"rows_in_scope": 0' in body
    assert '"rows_out_of_scope": 1' in body


def test_ag_ui_missing_openrouter_key_streams_run_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "contract_question_agent.api.adapter.API_OUTPUT_DIR",
        tmp_path / "api-runs",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    client = TestClient(create_app(load_env=False))

    with client.stream(
        "POST",
        "/ag-ui/runs",
        json={
            "contract_id": "demo-contract",
            "clause_type": "Non-Compete",
            "evidence_text": "Employee will not compete for one year after termination.",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: RUN_ERROR" in body
    assert "OPENROUTER_API_KEY is required unless --dry-run is set." in body
