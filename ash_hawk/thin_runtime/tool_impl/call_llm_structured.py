from __future__ import annotations

import asyncio
from pathlib import Path

import pydantic as pd

from ash_hawk.thin_runtime.llm_client import call_model_structured
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    FailureToolContext,
    RuntimeToolContext,
    ToolExecutionPayload,
)


class StructuredLLMResponse(pd.BaseModel):
    response: str


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    response = _structured_response(call)
    explanations, concepts = _extract_signal_lists(response)
    allowed_targets = call.context.workspace.allowed_target_files
    preferred_tool = "mutate_agent_files" if allowed_targets else ""
    return (
        True,
        ToolExecutionPayload(
            runtime_updates=RuntimeToolContext(preferred_tool=preferred_tool),
            failure_updates=FailureToolContext(explanations=explanations, concepts=concepts),
            audit_updates=AuditToolContext(llm_calls=["call_llm_structured"]),
        ),
        response,
        [],
    )


def _structured_response(call: ToolCall) -> str:
    model_result = call_model_structured(
        StructuredLLMResponse,
        system_prompt="Return a concise structured response for the current runtime context.",
        user_prompt=(
            f"Goal: {call.goal_id}\n"
            f"Agent text:\n{call.agent_text or ''}\n\n"
            f"Context:\n{call.context.model_dump(exclude_none=True)}\n\n"
            'Return JSON: {"response": "text"}'
        ),
    )
    if model_result is not None:
        return model_result.response

    try:
        from bolt_merlin.agent.execute import execute
    except ImportError:
        return "LLM unavailable"

    prompt = (
        "Provide a concise diagnosis and at least two hypotheses for the current eval improvement loop.\n\n"
        f"Scenario summary:\n{call.context.workspace.scenario_summary or 'No scenario summary available'}\n\n"
        f"Failure explanations:\n{' '.join(call.context.failure.explanations) or 'No failure explanations'}\n\n"
        f"Concepts:\n{' '.join(call.context.failure.concepts) or 'No concepts'}\n\n"
        "Return plain text in this exact shape:\n"
        "Diagnosis: <one paragraph>\n"
        "Hypotheses:\n- <hypothesis 1>\n- <hypothesis 2>"
    )
    result = asyncio.run(
        execute(
            prompt=prompt,
            agent_name="coding_agent",
            working_dir=Path(call.context.workspace.workdir or str(Path.cwd())),
            config_path=None,
            trace=False,
        )
    )
    if hasattr(result, "response") and isinstance(result.response, str) and result.response.strip():
        return result.response
    return "LLM unavailable"


def _extract_signal_lists(response: str) -> tuple[list[str], list[str]]:
    explanations: list[str] = []
    concepts: list[str] = []
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.startswith("Diagnosis:"):
            diagnosis = stripped.removeprefix("Diagnosis:").strip()
            if diagnosis:
                explanations.append(diagnosis)
        elif stripped.startswith("-"):
            hypothesis = stripped.removeprefix("-").strip()
            if hypothesis:
                concepts.append(hypothesis)
    return explanations[:3], concepts[:5]


COMMAND = ToolCommand(
    name="call_llm_structured",
    summary="Call an LLM for structured output.",
    when_to_use=["When this exact capability is needed"],
    when_not_to_use=["When required inputs are missing"],
    input_schema=basic_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["none"],
    risk_level="low",
    timeout_seconds=30,
    completion_criteria=[
        "Output matches declared schema",
        "Execution stays within timeout and permission bounds",
        "Errors are explicit and actionable when failure occurs",
    ],
    escalation_rules=[
        "Escalate when confidence in output validity is low",
        "Escalate when the request exceeds tool permissions",
        "Escalate when repeated retries produce the same failure",
    ],
    executor=_execute,
)


def run(call: ToolCall) -> ToolResult:
    return COMMAND.run(call)
