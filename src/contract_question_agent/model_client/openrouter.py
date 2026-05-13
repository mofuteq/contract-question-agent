"""OpenRouter client for one-call structured question generation."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

warnings.filterwarnings("ignore", message=r".*is experimental.*")

from agent_framework import Agent, MCPStdioTool
from agent_framework_openai import OpenAIChatClient
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import SAFETY_DISCLAIMER
from contract_question_agent.schemas import VerificationQuestionOutput
from contract_question_agent.workflows import tracing


DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT_TEMPLATE = "verification_question_system.j2"
USE_MCP_HINTS_ENV = "CONTRACT_QUESTION_USE_MCP_HINTS"
MCP_HINTS_TOOL_NAME = "contract-question-agent-clause-hints"
MCP_HINTS_SERVER_ARGS = [
    "run",
    "python",
    "-m",
    "contract_question_agent.mcp.server",
]


_PROMPT_ENV = Environment(
    loader=PackageLoader("contract_question_agent", "prompts"),
    autoescape=select_autoescape(default=False),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)
SYSTEM_PROMPT = _PROMPT_ENV.get_template(SYSTEM_PROMPT_TEMPLATE).render().strip()


class OpenRouterQuestionClient:
    """OpenRouter generation client backed by a Microsoft Agent Framework Agent."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        agent: Agent | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        self.use_mcp_hints = _env_flag_enabled(os.getenv(USE_MCP_HINTS_ENV))
        self.call_count = 0
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required unless --dry-run is set."
            )
        self.agent = agent or OpenAIChatClient(
            model=self.model_name,
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        ).as_agent(
            id="openrouter-verification-question-agent",
            name="OpenRouter verification question agent",
            instructions=SYSTEM_PROMPT,
        )

    async def generate(self, record: ClauseSpanRecord) -> VerificationQuestionOutput:
        self.call_count += 1
        request_payload = {
            "contract_id": record.contract_id,
            "clause_type": record.clause_type,
            "evidence_text": record.evidence_text,
        }
        mcp_hints_tool = _build_mcp_hints_tool() if self.use_mcp_hints else None
        span_metadata: dict[str, Any] = {
            "contract_id": record.contract_id,
            "clause_type": record.clause_type,
            "provider": "openrouter",
            "runtime": "microsoft-agent-framework",
            "mcp_hints_enabled": self.use_mcp_hints,
        }
        if mcp_hints_tool is not None:
            span_metadata["mcp_tool_name"] = MCP_HINTS_TOOL_NAME
        with tracing.span(
            "openrouter-verification-question-agent",
            input=_generation_input_summary(record),
            metadata=span_metadata,
            as_type="generation",
        ):
            run_kwargs: dict[str, Any] = {
                "options": {"response_format": VerificationQuestionOutput},
            }
            if mcp_hints_tool is not None:
                run_kwargs["tools"] = mcp_hints_tool
            response = await self.agent.run(
                json.dumps(request_payload, ensure_ascii=True),
                **run_kwargs,
            )
            output = _coerce_agent_response(response, self.model_name)
            mcp_tool_calls = extract_mcp_tool_calls(response)
            update_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "input": _generation_langfuse_input(record),
                "output": _generation_output_summary(output),
                "metadata": _generation_metadata(
                    record,
                    self.use_mcp_hints,
                    mcp_tool_calls=mcp_tool_calls,
                ),
            }
            usage_details = extract_usage_details(response)
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            tracing.update_current_generation(**update_kwargs)
            return output


def extract_mcp_tool_calls(response: Any) -> list[str]:
    """Return MCP tool names observed in a MAF response, without raw args/results."""
    calls: list[str] = []
    _collect_mcp_tool_calls(response, calls=calls, seen=set())
    return calls


def _collect_mcp_tool_calls(value: Any, *, calls: list[str], seen: set[int]) -> None:
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, dict):
        _collect_mcp_tool_call_from_mapping(value, calls)
        for key in (
            "messages",
            "contents",
            "content",
            "items",
            "raw_representation",
            "additional_properties",
            "metadata",
        ):
            _collect_mcp_tool_calls(value.get(key), calls=calls, seen=seen)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_mcp_tool_calls(item, calls=calls, seen=seen)
        return

    _collect_mcp_tool_call_from_object(value, calls)
    for key in (
        "messages",
        "contents",
        "content",
        "items",
        "raw_representation",
        "additional_properties",
        "metadata",
    ):
        _collect_mcp_tool_calls(getattr(value, key, None), calls=calls, seen=seen)


def _collect_mcp_tool_call_from_mapping(
    mapping: dict[str, Any],
    calls: list[str],
) -> None:
    content_type = mapping.get("type")
    if content_type not in {"mcp_server_tool_call", "function_call"}:
        return
    tool_name = mapping.get("tool_name") or mapping.get("name")
    if isinstance(tool_name, str) and _is_clause_hints_tool_name(tool_name):
        calls.append(tool_name)


def _collect_mcp_tool_call_from_object(value: Any, calls: list[str]) -> None:
    content_type = getattr(value, "type", None)
    if content_type not in {"mcp_server_tool_call", "function_call"}:
        return
    tool_name = getattr(value, "tool_name", None) or getattr(value, "name", None)
    if isinstance(tool_name, str) and _is_clause_hints_tool_name(tool_name):
        calls.append(tool_name)


def _is_clause_hints_tool_name(tool_name: str) -> bool:
    return tool_name == "lookup_clause_review_hints" or tool_name.endswith(
        "lookup_clause_review_hints"
    )


def _env_flag_enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _build_mcp_hints_tool() -> MCPStdioTool:
    return MCPStdioTool(
        MCP_HINTS_TOOL_NAME,
        command="uv",
        args=MCP_HINTS_SERVER_ARGS,
        allowed_tools=["lookup_clause_review_hints"],
    )


def extract_usage_details(response: Any) -> dict[str, int] | None:
    """Extract Langfuse usage details from common MAF/OpenAI response shapes."""
    return _extract_usage_details(response, seen=set())


def _extract_usage_details(response: Any, *, seen: set[int]) -> dict[str, int] | None:
    if response is None:
        return None
    response_id = id(response)
    if response_id in seen:
        return None
    seen.add(response_id)

    direct = _usage_from_mapping(_object_mapping(response))
    if direct is not None:
        return direct

    for key in (
        "usage",
        "usage_details",
        "metadata",
        "additional_properties",
        "raw_response",
        "response",
    ):
        nested = _get_value(response, key)
        if nested is None:
            continue
        usage_details = _extract_usage_details(nested, seen=seen)
        if usage_details is not None:
            return usage_details
    return None


def _usage_from_mapping(mapping: dict[str, Any]) -> dict[str, int] | None:
    key_sets = (
        ("prompt_tokens", "completion_tokens", "total_tokens"),
        ("inputUsage", "outputUsage", "totalUsage"),
        ("input_tokens", "output_tokens", "total_tokens"),
        ("input", "output", "total"),
    )
    for input_key, output_key, total_key in key_sets:
        input_tokens = _coerce_token_count(mapping.get(input_key))
        output_tokens = _coerce_token_count(mapping.get(output_key))
        total_tokens = _coerce_token_count(mapping.get(total_key))
        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue
        usage_details: dict[str, int] = {}
        if input_tokens is not None:
            usage_details["input"] = input_tokens
        if output_tokens is not None:
            usage_details["output"] = output_tokens
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if total_tokens is not None:
            usage_details["total"] = total_tokens
        return usage_details or None
    return None


def _object_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    mapping: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "inputUsage",
        "outputUsage",
        "totalUsage",
        "input_tokens",
        "output_tokens",
        "input",
        "output",
        "total",
    ):
        attr = getattr(value, key, None)
        if attr is not None:
            mapping[key] = attr
    return mapping


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _coerce_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _generation_input_summary(record: ClauseSpanRecord) -> dict[str, Any]:
    return {
        "contract_id": record.contract_id,
        "clause_type": record.clause_type,
        "evidence_char_count": len(record.evidence_text),
    }


def _generation_langfuse_input(record: ClauseSpanRecord) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _generation_input_summary(record),
            },
        ],
    }


def _generation_metadata(
    record: ClauseSpanRecord,
    mcp_hints_enabled: bool,
    *,
    mcp_tool_calls: list[str] | None = None,
) -> dict[str, Any]:
    mcp_tool_calls = mcp_tool_calls or []
    metadata: dict[str, Any] = {
        "contract_id": record.contract_id,
        "clause_type": record.clause_type,
        "provider": "openrouter",
        "runtime": "microsoft-agent-framework",
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
        "mcp_hints_enabled": mcp_hints_enabled,
        "mcp_tool_call_observed": bool(mcp_tool_calls),
        "mcp_tool_call_count": len(mcp_tool_calls),
    }
    if mcp_hints_enabled:
        metadata["mcp_tool_name"] = MCP_HINTS_TOOL_NAME
    if mcp_tool_calls:
        metadata["mcp_tool_calls"] = mcp_tool_calls
    return metadata


def _generation_output_summary(
    output: VerificationQuestionOutput,
) -> dict[str, Any]:
    return {
        "contract_id": output.contract_id,
        "clause_type": output.clause_type,
        "unknown_count": len(output.unknowns),
        "decision_risk_count": len(output.decision_risks),
        "legal_review_question_count": len(output.legal_review_questions),
        "verification_question_count": len(output.verification_questions),
        "safety_status": output.safety_status,
        "model_name": output.model_name,
    }


def _coerce_agent_response(response: Any, model_name: str) -> VerificationQuestionOutput:
    value = getattr(response, "value", None)
    if isinstance(value, VerificationQuestionOutput):
        return _with_generation_defaults(value, model_name)
    if value is not None:
        return _with_generation_defaults(
            VerificationQuestionOutput.model_validate(value),
            model_name,
        )

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OpenRouter agent response did not contain structured output.")
    parsed = json.loads(text)
    return _with_generation_defaults(
        VerificationQuestionOutput.model_validate(parsed),
        model_name,
    )


def _with_generation_defaults(
    output: VerificationQuestionOutput,
    model_name: str,
) -> VerificationQuestionOutput:
    return output.model_copy(
        update={
            "safety_disclaimer": output.safety_disclaimer or SAFETY_DISCLAIMER,
            "safety_status": output.safety_status or "unchecked",
            "safety_warnings": output.safety_warnings or [],
            "model_name": output.model_name or model_name,
        }
    )
