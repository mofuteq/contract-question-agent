from __future__ import annotations

from viewer.sse_client import parse_sse_lines, remove_evidence_text


def test_remove_evidence_text_recursively():
    payload = {
        "evidence_text": "secret clause text",
        "verification_questions": [
            {
                "question": "What should be checked?",
                "evidence_text": "secret nested text",
            }
        ],
    }

    sanitized = remove_evidence_text(payload)

    assert "evidence_text" not in sanitized
    assert "evidence_text" not in sanitized["verification_questions"][0]


def test_parse_sse_lines_yields_events():
    lines = [
        "event: RUN_STARTED",
        'data: {"type":"RUN_STARTED","run_id":"run-1"}',
        "",
        "event: STATE_SNAPSHOT",
        'data: {"type":"STATE_SNAPSHOT","snapshot":{"rows_written":1}}',
        "",
    ]

    events = list(parse_sse_lines(lines))

    assert events == [
        ("RUN_STARTED", {"type": "RUN_STARTED", "run_id": "run-1"}),
        (
            "STATE_SNAPSHOT",
            {"type": "STATE_SNAPSHOT", "snapshot": {"rows_written": 1}},
        ),
    ]
