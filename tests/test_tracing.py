from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from contract_question_agent import tracing


def test_maf_otel_setup_skips_without_langfuse_env(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "agent_framework.observability",
        SimpleNamespace(
            configure_otel_providers=lambda **kwargs: calls.append(("configure", kwargs)),
            enable_instrumentation=lambda **kwargs: calls.append(("enable", kwargs)),
        ),
    )

    tracing.configure_maf_otel_if_enabled()

    assert calls == []


def test_maf_otel_setup_configures_once_with_safe_defaults(monkeypatch):
    calls = []
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.test")
    monkeypatch.setitem(
        sys.modules,
        "agent_framework.observability",
        SimpleNamespace(
            configure_otel_providers=lambda **kwargs: calls.append(("configure", kwargs)),
            enable_instrumentation=lambda **kwargs: calls.append(("enable", kwargs)),
        ),
    )

    tracing.configure_maf_otel_if_enabled()
    tracing.configure_maf_otel_if_enabled()

    assert calls == [
        (
            "configure",
            {"enable_sensitive_data": False, "enable_console_exporters": False},
        ),
        ("enable", {"enable_sensitive_data": False}),
    ]
    assert (
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "https://langfuse.test/api/public/otel/v1/traces"
    )
    assert os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"
    assert os.environ["OTEL_SERVICE_NAME"] == "contract-question-agent"
    assert "Authorization=Basic " in os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"]
    assert "x-langfuse-ingestion-version=4" in os.environ[
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
    ]
    assert os.environ.get("ENABLE_SENSITIVE_DATA") is None


def test_maf_otel_setup_failure_is_noop(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret")

    def fail_configure(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "agent_framework.observability",
        SimpleNamespace(
            configure_otel_providers=fail_configure,
            enable_instrumentation=lambda **kwargs: None,
        ),
    )

    tracing.configure_maf_otel_if_enabled()
