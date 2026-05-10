"""Optional tracing helpers for Langfuse via Microsoft Agent Framework OTel.

Langfuse credentials are read from environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL` (default: https://cloud.langfuse.com)
- `LANGFUSE_HOST` (accepted as a backward-compatible fallback)
- `LANGFUSE_TRACING_ENVIRONMENT` (default: local)

This module only reads `os.environ`; loading `.env` is the responsibility of
the CLI entry point. It does not create manual Langfuse SDK observations. MAF
OpenTelemetry spans are the source of workflow, agent, chat, and token traces.
"""

import logging
import os
from base64 import b64encode

logger = logging.getLogger(__name__)

_CONFIGURED: bool | None = None
_MAF_OTEL_CONFIGURED = False
DEFAULT_LANGFUSE_ENVIRONMENT = "local"
DEFAULT_LANGFUSE_BASE_URL = "https://cloud.langfuse.com"


def is_configured() -> bool:
    """Return True when Langfuse public+secret keys are configured."""
    global _CONFIGURED
    if _CONFIGURED is None:
        public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        _CONFIGURED = bool(public and secret)
    return _CONFIGURED


def is_active() -> bool:
    """Return True when MAF OpenTelemetry was configured successfully."""
    return _MAF_OTEL_CONFIGURED


def is_enabled() -> bool:
    """Backward-compatible alias for active tracing."""
    return is_active()


def configure_maf_otel_if_enabled() -> None:
    """Configure Microsoft Agent Framework OpenTelemetry for Langfuse.

    This best-effort path can capture framework/provider GenAI spans when the
    MAF client emits them. It never enables sensitive data capture.
    """
    global _MAF_OTEL_CONFIGURED
    if _MAF_OTEL_CONFIGURED or not is_configured():
        return

    try:
        _configure_langfuse_otel_env()
        from agent_framework.observability import (  # type: ignore[import-not-found]
            configure_otel_providers,
            enable_instrumentation,
        )

        configure_otel_providers(
            enable_sensitive_data=False,
            enable_console_exporters=False,
        )
        enable_instrumentation(enable_sensitive_data=False)
        _MAF_OTEL_CONFIGURED = True
    except Exception as err:
        logger.warning("MAF OpenTelemetry setup failed: %s", err)


def get_current_trace_id() -> str | None:
    """Return None because trace identity is owned by the OTel backend."""
    return None


def get_current_trace_url() -> str | None:
    """Return None because trace identity is owned by the OTel backend."""
    return None


def get_tracing_environment() -> str:
    """Return the Langfuse tracing environment name."""
    return os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or DEFAULT_LANGFUSE_ENVIRONMENT


def flush() -> None:
    """Flush hook kept for CLI safety; MAF OTel handles exporter lifecycle."""
    return None


__all__ = [
    "flush",
    "configure_maf_otel_if_enabled",
    "get_current_trace_id",
    "get_current_trace_url",
    "get_tracing_environment",
    "is_active",
    "is_configured",
    "is_enabled",
]


def _configure_langfuse_otel_env() -> None:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return

    base_url = _get_langfuse_base_url().rstrip("/")
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"{base_url}/api/public/otel/v1/traces",
    )
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    os.environ.setdefault("OTEL_SERVICE_NAME", "contract-question-agent")
    _set_otel_resource_attribute(
        "deployment.environment.name",
        get_tracing_environment(),
    )
    if not os.getenv("OTEL_EXPORTER_OTLP_HEADERS") and not os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
    ):
        auth = b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
        os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = (
            f"Authorization=Basic {auth},x-langfuse-ingestion-version=4"
        )


def _get_langfuse_base_url() -> str:
    return os.getenv("LANGFUSE_BASE_URL") or os.getenv(
        "LANGFUSE_HOST",
        DEFAULT_LANGFUSE_BASE_URL,
    )


def _set_otel_resource_attribute(key: str, value: str) -> None:
    existing = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "").strip()
    parts = [part.strip() for part in existing.split(",") if part.strip()]
    if any(part.split("=", 1)[0].strip() == key for part in parts):
        return
    parts.append(f"{key}={value}")
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(parts)
