"""Small SSE client helpers for the Streamlit run viewer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import requests

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/ag-ui/runs"


def remove_evidence_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_evidence_text(inner)
            for key, inner in value.items()
            if key != "evidence_text"
        }
    if isinstance(value, list):
        return [remove_evidence_text(item) for item in value]
    return value


def parse_sse_lines(lines: Iterable[str | None]) -> Iterator[tuple[str, dict[str, Any]]]:
    event_type: str | None = None
    data_lines: list[str] = []

    for raw_line in lines:
        if raw_line is None:
            continue

        line = raw_line.strip()
        if not line:
            if event_type and data_lines:
                yield event_type, json.loads("\n".join(data_lines))
            event_type = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if event_type and data_lines:
        yield event_type, json.loads("\n".join(data_lines))


def iter_sse_events(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 120,
) -> Iterator[tuple[str, dict[str, Any]]]:
    with requests.post(url, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        yield from parse_sse_lines(response.iter_lines(decode_unicode=True))
