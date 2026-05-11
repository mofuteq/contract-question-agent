"""Optional Langfuse tracing helpers.

Provides `observe`, `span`, `update_current_generation`, `update_current_span`,
and `flush`, which become no-ops when Langfuse credentials are not configured.

Credentials are read from environment variables:
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (default: https://cloud.langfuse.com)

This module only reads `os.environ`; loading `.env` is the responsibility of
the application entry points.
"""

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_ENABLED: bool | None = None
_CLIENT: Any = None


def is_enabled() -> bool:
    """Return True when Langfuse public+secret keys are configured."""
    global _ENABLED
    if _ENABLED is None:
        public = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        _ENABLED = bool(public and secret)
    return _ENABLED


def get_client() -> Any:
    """Return a cached Langfuse client or None when disabled/unavailable.

    Used by all tracing helpers so missing credentials remain a no-op.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        _CLIENT = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=host,
        )
        logger.info("Langfuse tracing enabled host=%s", host)
        return _CLIENT
    except Exception as err:
        logger.warning("Langfuse initialization failed: %s", err)
        return None


def observe(
    *,
    name: str | None = None,
    as_type: str | None = None,
) -> Callable[[F], F]:
    """Return the Langfuse `@observe` decorator, or a no-op when disabled."""
    if not is_enabled():
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
    return _langfuse_observe(**kwargs)


def span(
    name: str,
    *,
    input: Any | None = None,
    metadata: Any | None = None,
    as_type: str = "span",
    trace_context: dict[str, str] | None = None,
) -> Any:
    """Return a Langfuse observation context manager, or a no-op context."""
    client = get_client()
    if client is None:
        return nullcontext()
    try:
        return client.start_as_current_observation(
            trace_context=trace_context,
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata,
        )
    except Exception as err:
        logger.debug("Langfuse span start failed: %s", err)
        return nullcontext()


def session(
    session_id: str,
    *,
    trace_name: str | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Propagate one Langfuse session id across all observations in a run."""
    if not is_enabled():
        return nullcontext()
    try:
        from langfuse import propagate_attributes  # type: ignore[import-not-found]

        return propagate_attributes(
            session_id=normalize_session_id(session_id),
            trace_name=trace_name,
            tags=tags,
        )
    except Exception as err:
        logger.debug("Langfuse session propagation failed: %s", err)
        return nullcontext()


def normalize_session_id(value: str) -> str:
    """Return a Langfuse-safe ASCII session id no longer than 200 chars."""
    normalized = value.encode("ascii", errors="ignore").decode("ascii").strip()
    if not normalized:
        normalized = "contract-question-agent-run"
    return normalized[:200]


def get_current_trace_id() -> str | None:
    """Return the active Langfuse trace id when available."""
    client = get_client()
    if client is None or not hasattr(client, "get_current_trace_id"):
        return None
    try:
        return client.get_current_trace_id()
    except Exception as err:
        logger.debug("Langfuse current trace id lookup failed: %s", err)
        return None


def get_current_observation_id() -> str | None:
    """Return the active Langfuse observation id when available."""
    client = get_client()
    if client is None or not hasattr(client, "get_current_observation_id"):
        return None
    try:
        return client.get_current_observation_id()
    except Exception as err:
        logger.debug("Langfuse current observation id lookup failed: %s", err)
        return None


def get_langgraph_callbacks(
    *,
    session_id: str,
    trace_name: str,
    tags: list[str] | None = None,
) -> list[Any]:
    """Return optional Langfuse LangGraph callbacks for graph visualization."""
    if not is_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler  # type: ignore[import-not-found]

        trace_id = get_current_trace_id()
        if trace_id:
            trace_context = {"trace_id": trace_id}
            observation_id = get_current_observation_id()
            if observation_id:
                trace_context["parent_span_id"] = observation_id
            return [CallbackHandler(trace_context=trace_context)]
        logger.debug(
            "Langfuse LangGraph callback skipped because no active trace id was found "
            "session_id=%s trace_name=%s tags=%s",
            session_id,
            trace_name,
            tags,
        )
        return []
    except Exception as err:
        logger.warning("Langfuse LangGraph callback initialization failed: %s", err)
        logger.debug(
            "Langfuse LangGraph callback context session_id=%s trace_name=%s tags=%s",
            session_id,
            trace_name,
            tags,
        )
        return []


@contextmanager
def state_transition(
    node_name: str,
    *,
    input_state: Any,
    next_node: str | None = None,
    config: Any | None = None,
) -> Any:
    """Trace one business-node state transition in the LangGraph workflow."""
    transition = {
        "node": node_name,
        "input_state": _state_name(input_state),
        "next_node": next_node,
    }
    trace_context = _langgraph_node_trace_context(config)
    with span(
        node_name,
        input=summarize_state(input_state),
        metadata=transition,
        trace_context=trace_context,
    ):
        def _record_output(output_state: Any) -> None:
            update_current_span(
                output=summarize_state(output_state),
                metadata=transition | {"output_state": _state_name(output_state)},
            )

        yield _record_output


def _langgraph_node_trace_context(config: Any | None) -> dict[str, str] | None:
    """Return the active LangGraph node observation as Langfuse trace context.

    The LangGraph CallbackHandler creates node CHAIN observations before the node
    function runs. LangGraph passes the active callback manager into node config;
    its parent_run_id is the current node run id. When available, use the matching
    Langfuse observation as the explicit parent for manual business-node spans so
    exports nest as: callback node CHAIN -> manual node SPAN -> generation.
    """
    if not isinstance(config, dict):
        return None
    callback_manager = config.get("callbacks")
    parent_run_id = getattr(callback_manager, "parent_run_id", None)
    if parent_run_id is None:
        return None

    handlers = list(getattr(callback_manager, "handlers", []) or [])
    handlers.extend(getattr(callback_manager, "inheritable_handlers", []) or [])
    seen: set[int] = set()
    for handler in handlers:
        handler_id = id(handler)
        if handler_id in seen:
            continue
        seen.add(handler_id)
        runs = getattr(handler, "_runs", None)
        if not isinstance(runs, dict) or parent_run_id not in runs:
            continue
        observation = runs[parent_run_id]
        trace_id = getattr(observation, "trace_id", None)
        observation_id = getattr(observation, "id", None)
        if isinstance(trace_id, str) and isinstance(observation_id, str):
            return {
                "trace_id": trace_id,
                "parent_span_id": observation_id,
            }
    return None


def summarize_state(state: Any) -> dict[str, Any]:
    """Return a compact, non-evidence workflow state summary for tracing."""
    if isinstance(state, BaseModel):
        data = state.model_dump(mode="json")
        summary: dict[str, Any] = {"state": state.__class__.__name__}
        request = data.get("request")
        if isinstance(request, dict):
            summary["request"] = _request_summary(request)
        else:
            summary.update(_request_summary(data))
        for key in (
            "rows_read",
            "rows_filtered",
            "rows_generated",
            "safety_failed_count",
            "rows_written",
        ):
            if key in data:
                summary[key] = data[key]
        if "records" in data and isinstance(data["records"], list):
            summary["record_count"] = len(data["records"])
        if "outputs" in data and isinstance(data["outputs"], list):
            summary["output_count"] = len(data["outputs"])
        for key in ("output_path", "metadata_path", "log_path"):
            if key in data:
                summary[key] = data[key]
        return summary
    return {"state": _state_name(state)}


def _request_summary(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "input_path",
        "output_path",
        "metadata_path",
        "log_path",
        "clause_type",
        "contract_id",
        "limit",
        "offset",
        "model_name",
        "dry_run",
    )
    return {key: data[key] for key in keys if key in data}


def _state_name(state: Any) -> str:
    return state.__class__.__name__


def update_current_generation(**kwargs: Any) -> None:
    """Update the active generation span with model/usage/input/output info."""
    client = get_client()
    if client is None:
        return
    try:
        client.update_current_generation(**kwargs)
    except Exception as err:
        logger.debug("update_current_generation failed: %s", err)


def update_current_span(**kwargs: Any) -> None:
    """Update the active Langfuse span with output/metadata/status details."""
    client = get_client()
    if client is None:
        return
    try:
        client.update_current_span(**kwargs)
    except Exception as err:
        logger.debug("update_current_span failed: %s", err)


def update_current_trace(**kwargs: Any) -> None:
    """Backward-compatible alias for trace-level metadata updates."""
    update_current_span(**kwargs)


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
    "get_current_observation_id",
    "get_current_trace_id",
    "get_langgraph_callbacks",
    "is_enabled",
    "observe",
    "normalize_session_id",
    "session",
    "span",
    "state_transition",
    "summarize_state",
    "update_current_generation",
    "update_current_span",
    "update_current_trace",
]
