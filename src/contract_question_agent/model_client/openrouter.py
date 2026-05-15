"""OpenRouter client for one-call structured question generation."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

warnings.filterwarnings("ignore", message=r".*is experimental.*")

from agent_framework import Agent
from agent_framework_openai import OpenAIChatClient
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from contract_question_agent.clause_hints.schemas import ClauseReviewHints
from contract_question_agent.cuad_loader import ClauseSpanRecord
from contract_question_agent.safety import SAFETY_DISCLAIMER, normalize_plain_string_fields
from contract_question_agent.schemas import VerificationQuestionOutput
from contract_question_agent.schemas import ReflectionResult
from contract_question_agent.skills.loader import (
    CONTRACT_VERIFICATION_QUESTIONS_SKILL,
    CONTRACT_VERIFICATION_QUESTIONS_SKILL_PATH,
    load_skill_text,
)
from contract_question_agent.workflows import tracing


DEFAULT_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SYSTEM_PROMPT_TEMPLATE = "verification_question_system.j2"
REFLECTION_PROMPT_TEMPLATE = "reflection_system.j2"
GENERATION_TEMPERATURE = 0.0
GENERATION_TOP_P = 0.6
GENERATION_SEED = 42
USE_MCP_HINTS_ENV = "CONTRACT_QUESTION_USE_MCP_HINTS"
MCP_HINTS_TOOL_NAME = "lookup_clause_review_hints"
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
_SYSTEM_PROMPT_TEMPLATE = _PROMPT_ENV.get_template(SYSTEM_PROMPT_TEMPLATE)
_REFLECTION_PROMPT_TEMPLATE = _PROMPT_ENV.get_template(REFLECTION_PROMPT_TEMPLATE)
DEFAULT_SKILL_NAME = CONTRACT_VERIFICATION_QUESTIONS_SKILL
DEFAULT_SKILL_PATH = CONTRACT_VERIFICATION_QUESTIONS_SKILL_PATH
DEFAULT_SKILL_TEXT = load_skill_text(DEFAULT_SKILL_NAME)
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.render(
    skill_text=DEFAULT_SKILL_TEXT,
    clause_review_hints=None,
    regeneration_guidance=None,
).strip()
REFLECTION_SYSTEM_PROMPT = _REFLECTION_PROMPT_TEMPLATE.render().strip()


@dataclass(frozen=True)
class MCPHintsLookupResult:
    attempted: bool = False
    found: bool = False
    hints: ClauseReviewHints | None = None
    error_type: str | None = None


class OpenRouterQuestionClient:
    """OpenRouter generation client backed by a Microsoft Agent Framework Agent."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        agent: Agent | None = None,
        mcp_hints_lookup: Callable[[str], Awaitable[MCPHintsLookupResult]] | None = None,
        skill_text: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL
        self.use_mcp_hints = _env_flag_enabled(os.getenv(USE_MCP_HINTS_ENV))
        self._mcp_hints_lookup = mcp_hints_lookup or lookup_clause_review_hints_via_mcp
        self.skill_text = skill_text if skill_text is not None else DEFAULT_SKILL_TEXT
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

    async def generate(
        self,
        record: ClauseSpanRecord,
        *,
        regeneration_guidance: str | None = None,
    ) -> VerificationQuestionOutput:
        self.call_count += 1
        request_payload = {
            "contract_id": record.contract_id,
            "clause_type": record.clause_type,
            "evidence_text": record.evidence_text,
        }
        mcp_hints_result = (
            await self._mcp_hints_lookup(record.clause_type)
            if self.use_mcp_hints
            else MCPHintsLookupResult()
        )
        system_prompt = render_system_prompt(
            skill_text=self.skill_text,
            clause_review_hints=mcp_hints_result.hints,
            regeneration_guidance=regeneration_guidance,
        )
        span_metadata: dict[str, Any] = {
            "contract_id": record.contract_id,
            "clause_type": record.clause_type,
            "provider": "openrouter",
            "runtime": "microsoft-agent-framework",
            **_skill_metadata(),
            **_generation_params_metadata(),
            **_mcp_hints_metadata(mcp_hints_result, self.use_mcp_hints),
        }
        with tracing.span(
            "openrouter-verification-question-agent",
            input=_generation_input_summary(record),
            metadata=span_metadata,
            as_type="generation",
        ):
            run_kwargs: dict[str, Any] = {
                "options": {
                    "instructions": system_prompt,
                    "response_format": VerificationQuestionOutput,
                    "temperature": GENERATION_TEMPERATURE,
                    "top_p": GENERATION_TOP_P,
                    "seed": GENERATION_SEED,
                },
            }
            response = await self.agent.run(
                json.dumps(request_payload, ensure_ascii=True),
                **run_kwargs,
            )
            output = _coerce_agent_response(response, self.model_name)
            update_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "input": _generation_langfuse_input(record, system_prompt),
                "output": _generation_output_summary(output),
                "metadata": _generation_metadata(
                    record,
                    self.use_mcp_hints,
                    mcp_hints_result=mcp_hints_result,
                ),
            }
            usage_details = extract_usage_details(response)
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            tracing.update_current_generation(**update_kwargs)
            return output

    async def reflect(self, output: VerificationQuestionOutput) -> ReflectionResult:
        span_metadata: dict[str, Any] = {
            "contract_id": output.contract_id,
            "clause_type": output.clause_type,
            "provider": "openrouter",
            "runtime": "microsoft-agent-framework",
            "system_prompt_template": REFLECTION_PROMPT_TEMPLATE,
            **_generation_params_metadata(),
        }
        reflection_input = _reflection_input(output)
        with tracing.span(
            "openrouter-skill-thesis-reflector",
            input=reflection_input,
            metadata=span_metadata,
            as_type="generation",
        ):
            response = await self.agent.run(
                json.dumps(reflection_input, ensure_ascii=True),
                options={
                    "instructions": REFLECTION_SYSTEM_PROMPT,
                    "response_format": ReflectionResult,
                    "temperature": GENERATION_TEMPERATURE,
                    "top_p": GENERATION_TOP_P,
                    "seed": GENERATION_SEED,
                },
            )
            result = _coerce_reflection_response(response)
            update_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": reflection_input},
                    ]
                },
                "output": _reflection_output_summary(result),
                "metadata": span_metadata
                | {
                    "reflection_status": result.status,
                    "reflection_violation_count": len(result.violations),
                    "violations": [
                        violation.model_dump(mode="json")
                        for violation in result.violations
                    ],
                },
            }
            usage_details = extract_usage_details(response)
            if usage_details is not None:
                update_kwargs["usage_details"] = usage_details
            tracing.update_current_generation(**update_kwargs)
            return result


async def lookup_clause_review_hints_via_mcp(
    clause_type: str,
) -> MCPHintsLookupResult:
    try:
        params = StdioServerParameters(command="uv", args=MCP_HINTS_SERVER_ARGS)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    MCP_HINTS_TOOL_NAME,
                    {"clause_type": clause_type},
                )
        payload = _mcp_tool_result_payload(result)
        if not payload.get("found"):
            return MCPHintsLookupResult(attempted=True, found=False)
        return MCPHintsLookupResult(
            attempted=True,
            found=True,
            hints=ClauseReviewHints.model_validate(payload),
        )
    except Exception as err:
        return MCPHintsLookupResult(
            attempted=True,
            found=False,
            error_type=err.__class__.__name__,
        )


def _mcp_tool_result_payload(result: Any) -> dict[str, Any]:
    structured_content = getattr(result, "structuredContent", None)
    if isinstance(structured_content, dict):
        return structured_content
    structured_content = getattr(result, "structured_content", None)
    if isinstance(structured_content, dict):
        return structured_content

    content = getattr(result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _env_flag_enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


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


def render_system_prompt(
    skill_text: str | None = None,
    clause_review_hints: ClauseReviewHints | None = None,
    regeneration_guidance: str | None = None,
) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.render(
        skill_text=DEFAULT_SKILL_TEXT if skill_text is None else skill_text,
        clause_review_hints=clause_review_hints,
        regeneration_guidance=regeneration_guidance,
    ).strip()


def _generation_langfuse_input(
    record: ClauseSpanRecord,
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
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
    mcp_hints_result: MCPHintsLookupResult | None = None,
) -> dict[str, Any]:
    return {
        "contract_id": record.contract_id,
        "clause_type": record.clause_type,
        "provider": "openrouter",
        "runtime": "microsoft-agent-framework",
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
        **_skill_metadata(),
        **_generation_params_metadata(),
        **_mcp_hints_metadata(
            mcp_hints_result or MCPHintsLookupResult(),
            mcp_hints_enabled,
        ),
    }


def _skill_metadata() -> dict[str, str]:
    return {
        "skill_name": DEFAULT_SKILL_NAME,
        "skill_path": DEFAULT_SKILL_PATH,
    }


def _generation_params_metadata() -> dict[str, float | int]:
    return {
        "temperature": GENERATION_TEMPERATURE,
        "top_p": GENERATION_TOP_P,
        "seed": GENERATION_SEED,
    }


def _mcp_hints_metadata(
    result: MCPHintsLookupResult,
    mcp_hints_enabled: bool,
) -> dict[str, Any]:
    hints = result.hints
    metadata: dict[str, Any] = {
        "mcp_hints_enabled": mcp_hints_enabled,
        "mcp_hints_lookup_attempted": result.attempted,
        "mcp_hints_found": result.found,
        "mcp_tool_name": MCP_HINTS_TOOL_NAME,
        "common_unknowns_count": len(hints.common_unknowns) if hints is not None else 0,
        "question_categories_count": (
            len(hints.question_categories) if hints is not None else 0
        ),
        "review_hints_count": len(hints.review_hints) if hints is not None else 0,
    }
    if result.error_type is not None:
        metadata["mcp_hints_error_type"] = result.error_type
    return metadata


def _generation_output_summary(
    output: VerificationQuestionOutput,
) -> dict[str, Any]:
    return {
        "contract_id": output.contract_id,
        "clause_type": output.clause_type,
        "selected_review_lens_count": len(output.selected_review_lenses),
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


def _coerce_reflection_response(response: Any) -> ReflectionResult:
    value = getattr(response, "value", None)
    if isinstance(value, ReflectionResult):
        return value
    if value is not None:
        return ReflectionResult.model_validate(value)

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OpenRouter reflector response did not contain structured output.")
    return ReflectionResult.model_validate(json.loads(text))


def _with_generation_defaults(
    output: VerificationQuestionOutput,
    model_name: str,
) -> VerificationQuestionOutput:
    output = normalize_plain_string_fields(output)
    return output.model_copy(
        update={
            "safety_disclaimer": output.safety_disclaimer or SAFETY_DISCLAIMER,
            "safety_status": output.safety_status or "unchecked",
            "safety_warnings": output.safety_warnings or [],
            "model_name": output.model_name or model_name,
        }
    )


def _reflection_input(output: VerificationQuestionOutput) -> dict[str, Any]:
    return {
        "clause_type": output.clause_type,
        "selected_review_lenses": [
            item.model_dump(mode="json") for item in output.selected_review_lenses
        ],
        "unknowns": output.unknowns,
        "decision_risks": output.decision_risks,
        "legal_review_questions": [
            item.model_dump(mode="json") for item in output.legal_review_questions
        ],
        "verification_questions": [
            item.model_dump(mode="json") for item in output.verification_questions
        ],
        "suggested_next_step": output.suggested_next_step,
        "safety_status": output.safety_status,
    }


def _reflection_output_summary(result: ReflectionResult) -> dict[str, Any]:
    return {
        "reflection_status": result.status,
        "reflection_violation_count": len(result.violations),
        "regeneration_guidance_present": bool(result.regeneration_guidance),
    }
