from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from opentelemetry import context

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
    assert tracing.is_active() is False


def test_maf_otel_setup_configures_once_with_safe_defaults(monkeypatch):
    calls = []
    processors = []
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
    monkeypatch.setattr(
        "opentelemetry.trace.get_tracer_provider",
        lambda: SimpleNamespace(
            add_span_processor=lambda processor: processors.append(processor),
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
    assert (
        os.environ["OTEL_RESOURCE_ATTRIBUTES"]
        == "deployment.environment.name=local"
    )
    assert "Authorization=Basic " in os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"]
    assert "x-langfuse-ingestion-version=4" in os.environ[
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
    ]
    assert os.environ.get("ENABLE_SENSITIVE_DATA") is None
    assert tracing.is_active() is True
    assert len(processors) == 1


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
    assert tracing.is_active() is False


def test_trace_run_session_copies_session_attributes_to_spans():
    processor = tracing._SessionAttributeSpanProcessor()
    span = SimpleNamespace(attributes={})
    span.set_attribute = lambda key, value: span.attributes.__setitem__(key, value)

    with tracing.trace_run_session("test-run", trace_name="contract-question-generate"):
        processor.on_start(span, parent_context=context.get_current())

    assert span.attributes == {
        "langfuse.session.id": "test-run",
        "session.id": "test-run",
        "langfuse.trace.name": "contract-question-generate",
    }
