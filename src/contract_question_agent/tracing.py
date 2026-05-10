"""Optional Langfuse tracing helpers.

Provides `observe`, `update_current_generation`, `update_current_trace`, and
`flush`, which become no-ops when Langfuse credentials are not configured.

Credentials are read from environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL` (default: https://cloud.langfuse.com)
- `LANGFUSE_HOST` (accepted as a backward-compatible fallback)
- `LANGFUSE_TRACING_ENVIRONMENT` (default: local)

This module only reads `os.environ`; loading `.env` is the responsibility of
the CLI entry point.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_CONFIGURED: bool | None = None
_CLIENT: Any = None
_CLIENT_INIT_FAILED = False
DEFAULT_LANGFUSE_ENVIRONMENT = "local"

_SENSITIVE_METADATA_KEYS = {
    "api_key",
    "evidence_text",
    "generated_questions",
    "legal_review_questions",
    "model_output",
    "openrouter_api_key",
    "raw_contract_content",
    "secret_key",
    "verification_questions",
}


def is_configured() -> bool:
    """Return True when Langfuse public+secret keys are configured."""
    global _CONFIGURED
    if _CONFIGURED is None:
        public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        _CONFIGURED = bool(public and secret)
    return _CONFIGURED


def is_active() -> bool:
    """Return True when a usable Langfuse client is available."""
    return get_client() is not None


def is_enabled() -> bool:
    """Backward-compatible alias for active tracing."""
    return is_active()


def get_client() -> Any:
    """Return a cached Langfuse client or None when disabled/unavailable.

    Trace/observation updates use the cached client when available.
    """
    global _CLIENT, _CLIENT_INIT_FAILED
    if _CLIENT is not None:
        return _CLIENT
    if _CLIENT_INIT_FAILED or not is_configured():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
            "LANGFUSE_HOST",
            "https://cloud.langfuse.com",
        )
        _CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=host,
            environment=get_tracing_environment(),
        )
        logger.info("Langfuse tracing enabled host=%s", host)
        return _CLIENT
    except Exception as err:
        _CLIENT_INIT_FAILED = True
        logger.warning("Langfuse initialization failed: %s", err)
        return None


def _get_context() -> Any:
    """Return the cached Langfuse client or None when disabled/unavailable."""
    return get_client()


def observe(
    *,
    name: str | None = None,
    as_type: str | None = None,
) -> Callable[[F], F]:
    """Return the Langfuse `@observe` decorator, or a no-op when disabled."""
    if not is_active():
        def _passthrough(func: F) -> F:
            return func
        return _passthrough
    try:
        from langfuse import observe as _langfuse_observe  # type: ignore[import-not-found]
    except Exception as err:
        logger.warning("Langfuse observe import failed: %s", err)
        def _passthrough(func: F) -> F:
            return func
        return _passthrough

    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if as_type is not None:
        kwargs["as_type"] = as_type
    kwargs["capture_input"] = False
    kwargs["capture_output"] = False
    return _langfuse_observe(**kwargs)


def update_current_generation(**kwargs: Any) -> None:
    """Update the active generation/span with compact, safe metadata."""
    ctx = _get_context()
    if ctx is None:
        return
    kwargs = _safe_kwargs(kwargs)
    try:
        ctx.update_current_generation(**kwargs)
    except Exception as err:
        logger.debug("update_current_generation failed: %s", err)
        try:
            ctx.update_current_span(**kwargs)
        except Exception as span_err:
            logger.debug("update_current_span failed: %s", span_err)


def update_current_trace(**kwargs: Any) -> None:
    """Attach metadata (user_id, session_id, tags, ...) to the active trace."""
    ctx = _get_context()
    if ctx is None:
        return
    kwargs = _safe_kwargs(kwargs)
    try:
        ctx.update_current_trace(**kwargs)
    except Exception as err:
        logger.debug("update_current_trace failed: %s", err)


def get_current_trace_id() -> str | None:
    """Return the active trace id when Langfuse provides one."""
    client = get_client()
    if client is None:
        return None
    try:
        trace_id = client.get_current_trace_id()
    except Exception as err:
        logger.debug("get_current_trace_id failed: %s", err)
        return None
    return str(trace_id) if trace_id else None


def get_current_trace_url() -> str | None:
    """Return the active trace URL when Langfuse provides one."""
    client = get_client()
    if client is None:
        return None
    try:
        trace_url = client.get_trace_url(trace_id=get_current_trace_id())
    except Exception as err:
        logger.debug("get_current_trace_url failed: %s", err)
        return None
    return str(trace_url) if trace_url else None


def get_tracing_environment() -> str:
    """Return the Langfuse tracing environment name."""
    return os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or DEFAULT_LANGFUSE_ENVIRONMENT


def flush() -> None:
    """Flush pending Langfuse events; safe when disabled."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as err:
        logger.debug("Langfuse flush failed: %s", err)


__all__ = [
    "flush",
    "get_client",
    "get_current_trace_id",
    "get_current_trace_url",
    "get_tracing_environment",
    "is_active",
    "is_configured",
    "is_enabled",
    "observe",
    "update_current_generation",
    "update_current_trace",
]


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if "metadata" not in kwargs or kwargs["metadata"] is None:
        return kwargs
    return {**kwargs, "metadata": _safe_metadata(kwargs["metadata"])}


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if key_text.lower() in _SENSITIVE_METADATA_KEYS:
            continue
        safe[key_text] = _safe_metadata_value(value)
    return safe


def _safe_metadata_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
        return [_safe_metadata_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return _safe_metadata(value)
    return str(value)
