"""Minimal Streamlit viewer for AG-UI run events."""

from __future__ import annotations

from typing import Any

import streamlit as st

from viewer.sse_client import (
    DEFAULT_BACKEND_URL,
    iter_sse_events,
    remove_evidence_text,
)

DEFAULT_CONTRACT_ID = "demo-contract"
DEFAULT_CLAUSE_TYPE = "Non-Compete"
DEFAULT_EVIDENCE_TEXT = "Employee will not compete for one year after termination."
METRIC_FIELDS = [
    "rows_read",
    "rows_filtered",
    "rows_in_scope",
    "rows_out_of_scope",
    "rows_generated",
    "rows_written",
]


def init_session_state() -> None:
    st.session_state.setdefault("viewer_status", "idle")
    st.session_state.setdefault("viewer_events", [])
    st.session_state.setdefault("viewer_snapshot", {})
    st.session_state.setdefault("viewer_error", "")


def event_row(event_type: str, data: dict[str, Any]) -> dict[str, str]:
    if event_type in {"STEP_STARTED", "STEP_FINISHED"}:
        detail = str(data.get("step_name", ""))
    elif event_type == "RUN_STARTED":
        detail = str(data.get("run_id", ""))
    elif event_type == "RUN_ERROR":
        detail = str(data.get("message", ""))
    elif event_type == "RUN_FINISHED":
        detail = str(data.get("outcome", {}).get("type", ""))
    elif event_type == "STATE_SNAPSHOT":
        snapshot = data.get("snapshot", {})
        detail = f"rows_written={snapshot.get('rows_written', '')}"
    else:
        detail = ""

    return {"event": event_type, "detail": detail}


def render_status(status: str, error_message: str) -> None:
    st.metric("Current status", status)
    if status == "error" and error_message:
        st.error(error_message)


def render_event_timeline(events: list[dict[str, Any]]) -> None:
    st.subheader("Event timeline")
    if not events:
        st.caption("No events yet.")
        return

    rows = [event_row(event["type"], event["data"]) for event in events]
    render_table(rows)


def render_table(rows: list[dict[str, Any]]) -> None:
    try:
        st.dataframe(rows, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def render_metrics(snapshot: dict[str, Any]) -> None:
    st.subheader("Metrics")
    columns = st.columns(3)
    for index, field in enumerate(METRIC_FIELDS):
        with columns[index % len(columns)]:
            st.metric(field, snapshot.get(field, 0))


def render_selected_review_lenses(snapshot: dict[str, Any]) -> None:
    st.subheader("Selected review lenses")
    lenses = snapshot.get("selected_review_lenses") or []
    if lenses:
        render_table(lenses)
    else:
        st.caption("No selected review lenses.")


def render_verification_questions(snapshot: dict[str, Any]) -> None:
    st.subheader("Verification questions")
    outputs = snapshot.get("verification_questions") or []
    if not outputs:
        st.caption("No verification questions.")
        return

    for output_index, output in enumerate(outputs, start=1):
        st.markdown(f"**Output {output_index}: {output.get('clause_type', '')}**")
        for question_index, question in enumerate(
            output.get("verification_questions", []),
            start=1,
        ):
            question_text = question.get("question", "")
            why_it_matters = question.get("why_it_matters", "")
            st.markdown(f"{question_index}. {question_text}")
            if why_it_matters:
                st.caption(why_it_matters)


def render_safety_status(snapshot: dict[str, Any]) -> None:
    st.subheader("Safety")
    st.write(snapshot.get("safety_status") or "not_applicable")


def render_artifact_paths(snapshot: dict[str, Any]) -> None:
    st.subheader("Artifact paths")
    paths = [
        {"artifact": "output_path", "path": snapshot.get("output_path", "")},
        {"artifact": "metadata_path", "path": snapshot.get("metadata_path", "")},
        {"artifact": "log_path", "path": snapshot.get("log_path", "")},
    ]
    render_table(paths)


def render_snapshot(snapshot: dict[str, Any]) -> None:
    if not snapshot:
        return

    safe_snapshot = remove_evidence_text(snapshot)
    render_metrics(safe_snapshot)
    render_selected_review_lenses(safe_snapshot)
    render_safety_status(safe_snapshot)
    render_verification_questions(safe_snapshot)
    render_artifact_paths(safe_snapshot)
    with st.expander("Raw snapshot JSON"):
        st.json(safe_snapshot)


def update_placeholders(
    *,
    status_placeholder: st.delta_generator.DeltaGenerator,
    timeline_placeholder: st.delta_generator.DeltaGenerator,
    snapshot_placeholder: st.delta_generator.DeltaGenerator,
    status: str,
    events: list[dict[str, Any]],
    snapshot: dict[str, Any],
    error_message: str,
) -> None:
    with status_placeholder.container():
        render_status(status, error_message)
    with timeline_placeholder.container():
        render_event_timeline(events)
    with snapshot_placeholder.container():
        render_snapshot(snapshot)


def main() -> None:
    st.set_page_config(page_title="AG-UI Run Viewer", layout="wide")
    init_session_state()

    st.title("AG-UI Run Viewer")
    st.caption("Minimal run/output viewer. Not legal advice.")

    with st.sidebar:
        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
        dry_run = st.checkbox("Dry run", value=True)

    with st.form("run_form"):
        contract_id = st.text_input("Contract ID", value=DEFAULT_CONTRACT_ID)
        clause_type = st.text_input("Clause type", value=DEFAULT_CLAUSE_TYPE)
        evidence_text = st.text_area(
            "Evidence text",
            value=DEFAULT_EVIDENCE_TEXT,
            height=140,
        )
        submitted = st.form_submit_button("Run")

    status_placeholder = st.empty()
    timeline_placeholder = st.empty()
    snapshot_placeholder = st.empty()

    if submitted:
        payload = {
            "contract_id": contract_id,
            "clause_type": clause_type,
            "evidence_text": evidence_text,
            "dry_run": dry_run,
        }
        events: list[dict[str, Any]] = []
        snapshot: dict[str, Any] = {}
        status = "running"
        error_message = ""
        update_placeholders(
            status_placeholder=status_placeholder,
            timeline_placeholder=timeline_placeholder,
            snapshot_placeholder=snapshot_placeholder,
            status=status,
            events=events,
            snapshot=snapshot,
            error_message=error_message,
        )

        try:
            for event_type, data in iter_sse_events(backend_url, payload):
                safe_data = remove_evidence_text(data)
                events.append({"type": event_type, "data": safe_data})

                if event_type == "STATE_SNAPSHOT":
                    snapshot = remove_evidence_text(safe_data.get("snapshot", {}))
                elif event_type == "RUN_FINISHED":
                    snapshot = remove_evidence_text(safe_data.get("result", {}))
                    status = "finished"
                elif event_type == "RUN_ERROR":
                    error_message = str(safe_data.get("message", "Unknown error"))
                    status = "error"

                update_placeholders(
                    status_placeholder=status_placeholder,
                    timeline_placeholder=timeline_placeholder,
                    snapshot_placeholder=snapshot_placeholder,
                    status=status,
                    events=events,
                    snapshot=snapshot,
                    error_message=error_message,
                )

                if status == "error":
                    break
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            update_placeholders(
                status_placeholder=status_placeholder,
                timeline_placeholder=timeline_placeholder,
                snapshot_placeholder=snapshot_placeholder,
                status=status,
                events=events,
                snapshot=snapshot,
                error_message=error_message,
            )

        st.session_state.viewer_status = status
        st.session_state.viewer_events = events
        st.session_state.viewer_snapshot = snapshot
        st.session_state.viewer_error = error_message
    else:
        update_placeholders(
            status_placeholder=status_placeholder,
            timeline_placeholder=timeline_placeholder,
            snapshot_placeholder=snapshot_placeholder,
            status=st.session_state.viewer_status,
            events=st.session_state.viewer_events,
            snapshot=st.session_state.viewer_snapshot,
            error_message=st.session_state.viewer_error,
        )


if __name__ == "__main__":
    main()
