"""Minimal AG-UI-compatible event helpers."""

from __future__ import annotations

import json
from typing import Any

AG_UI_RUN_STARTED = "RUN_STARTED"
AG_UI_STEP_STARTED = "STEP_STARTED"
AG_UI_STEP_FINISHED = "STEP_FINISHED"
AG_UI_STATE_SNAPSHOT = "STATE_SNAPSHOT"
AG_UI_RUN_FINISHED = "RUN_FINISHED"
AG_UI_RUN_ERROR = "RUN_ERROR"
AG_UI_CUSTOM = "CUSTOM"


def sse_event(event_type: str, payload: dict[str, Any]) -> str:
    body = {"type": event_type, **payload}
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(body, ensure_ascii=False, default=str)}\n\n"
    )
