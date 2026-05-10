from __future__ import annotations

import json
import inspect
from functools import wraps
from pathlib import Path

from contract_question_agent import tracing
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
    log_path = run_dir / "run.log"
    rows = [
        VerificationQuestionOutput.model_validate(json.loads(line))
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log_text = log_path.read_text(encoding="utf-8")

    assert len(rows) == 1
    assert rows[0].contract_id == "C1"
    assert rows[0].safety_status == "passed"
    assert metadata["run_id"] == "test-run"
    assert metadata["input_path"] == str(input_path)
    assert metadata["output_path"] == str(output_path)
    assert metadata["metadata_path"] == str(metadata_path)
    assert metadata["log_path"] == str(log_path)
    assert metadata["clause_type"] == "Non-Compete"
    assert metadata["contract_id"] is None
    assert metadata["limit"] == 1
    assert metadata["offset"] == 0
    assert metadata["model_name"] == DEFAULT_OPENROUTER_MODEL
    assert metadata["dry_run"] is True
    assert metadata["rows_read"] == 2
    assert metadata["rows_filtered"] == 1
    assert metadata["rows_generated"] == 1
    assert metadata["safety_failed_count"] == 0
    assert metadata["rows_written"] == 1
    assert metadata["tracing_enabled"] is False
    assert metadata["langfuse_trace_id"] is None
    assert metadata["langfuse_trace_url"] is None
    assert "T" in metadata["created_at"]
    assert "run_id=test-run" in log_text
    assert "tracing_enabled=False" in log_text
    assert "rows_read=2" in log_text
    assert "rows_filtered=1" in log_text
    assert "rows_generated=1" in log_text
    assert "safety_failed_count=0" in log_text
    assert "rows_written=1" in log_text
    assert "OPENROUTER_API_KEY" not in log_text
    assert "Employee will not compete" not in log_text
    assert "What should be reviewed about this Non-Compete clause?" not in log_text
    assert "Grounding review in facts" not in log_text


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


def test_cli_noops_tracing_without_langfuse_env_vars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])

    run_dir = _run_cli(input_path, output_dir, extra_args=["--dry-run"])

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert metadata["tracing_enabled"] is False
    assert metadata["langfuse_trace_id"] is None
    assert metadata["langfuse_trace_url"] is None
    assert "tracing_enabled=False" in log_text


def test_cli_noops_tracing_when_langfuse_client_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret")
    monkeypatch.setattr(tracing, "get_client", lambda: None)
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])

    run_dir = _run_cli(input_path, output_dir, extra_args=["--dry-run"])

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert tracing.is_configured() is True
    assert tracing.is_active() is False
    assert metadata["tracing_enabled"] is False
    assert metadata["langfuse_trace_id"] is None
    assert metadata["langfuse_trace_url"] is None
    assert "tracing_enabled=False" in log_text


def test_cli_records_fake_langfuse_trace_and_node_spans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "test")
    input_path = tmp_path / "clause_spans.jsonl"
    output_dir = tmp_path / "runs"
    _write_input(input_path, [_span()])
    fake_client = FakeLangfuseClient()
    monkeypatch.setattr(tracing, "_CLIENT", fake_client)
    monkeypatch.setattr(tracing, "observe", fake_observe(fake_client))

    run_dir = _run_cli(input_path, output_dir, extra_args=["--dry-run"])

    span_names = [event[1] for event in fake_client.events if event[0] == "enter"]
    assert span_names == [
        "contract-question-generate",
        "LOAD_CLAUSE_SPANS",
        "FILTER_RECORDS",
        "GENERATE_MINIMAL_QUESTIONS",
        "SAFETY_CHECK",
        "WRITE_OUTPUT",
    ]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["tracing_enabled"] is True
    assert metadata["langfuse_trace_id"] == "trace-123"
    assert metadata["langfuse_trace_url"] == "https://langfuse.test/trace-123"
    assert metadata["langfuse_environment"] == "test"
    assert fake_client.trace_metadata["session_id"] == "test-run"
    assert fake_client.trace_metadata["metadata"]["run_id"] == "test-run"
    updated_metadata = {
        event[1]: event[2] for event in fake_client.events if event[0] == "update"
    }
    assert updated_metadata["LOAD_CLAUSE_SPANS"] == {"rows_read": 1}
    assert updated_metadata["FILTER_RECORDS"] == {
        "rows_read": 1,
        "rows_filtered": 1,
    }
    assert updated_metadata["GENERATE_MINIMAL_QUESTIONS"] == {
        "rows_filtered": 1,
        "rows_generated": 1,
        "model_name": DEFAULT_OPENROUTER_MODEL,
        "dry_run": True,
    }
    assert updated_metadata["SAFETY_CHECK"] == {
        "rows_generated": 1,
        "safety_failed_count": 0,
    }
    assert updated_metadata["WRITE_OUTPUT"] == {
        "rows_written": 1,
        "output_path": str(run_dir / "verification_questions.jsonl"),
        "metadata_path": str(run_dir / "run_metadata.json"),
        "log_path": str(run_dir / "run.log"),
    }
    assert fake_client.flushed is True


class FakeLangfuseClient:
    trace_id = "trace-123"

    def __init__(self) -> None:
        self.events = []
        self.flushed = False

    def get_current_trace_id(self):
        return self.trace_id

    def get_trace_url(self, *, trace_id):
        return f"https://langfuse.test/{trace_id}"

    def update_current_generation(self, **kwargs):
        metadata = kwargs.get("metadata", {})
        self.events.append(("update", self.events[-1][1], metadata))

    def update_current_trace(self, **kwargs):
        self.trace_metadata = kwargs

    def flush(self):
        self.flushed = True


def fake_observe(fake_client: FakeLangfuseClient):
    def _observe(*, name=None, as_type=None):
        assert as_type == "span"

        def _decorator(func):
            @wraps(func)
            def _wrapper(*args, **kwargs):
                fake_client.events.append(("enter", name, {}))
                try:
                    return func(*args, **kwargs)
                finally:
                    fake_client.events.append(("exit", name, None))

            @wraps(func)
            async def _async_wrapper(*args, **kwargs):
                fake_client.events.append(("enter", name, {}))
                try:
                    return await func(*args, **kwargs)
                finally:
                    fake_client.events.append(("exit", name, None))

            if inspect.iscoroutinefunction(func):
                return _async_wrapper
            return _wrapper

        return _decorator

    return _observe
