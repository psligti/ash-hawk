from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pydantic as pd

from ash_hawk.thin_runtime.llm_client import call_model_structured
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._native_tooling import workspace_relative_string
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    FailureToolContext,
    RankedHypothesis,
    RuntimeToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
    WorkspaceToolContext,
)


class StructuredHypothesis(pd.BaseModel):
    name: str
    score: float = pd.Field(ge=0.0, le=1.0)
    rationale: str = ""
    target_files: list[str] = pd.Field(default_factory=list)
    ideal_outcome: str = ""


class StructuredLLMResponse(pd.BaseModel):
    diagnosis: str
    blocker: str = ""
    ideal_outcome: str = ""
    hypotheses: list[StructuredHypothesis] = pd.Field(default_factory=list)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    response = _structured_response(call)
    explanations, concepts = _extract_signal_lists(response)
    ranked_hypotheses = _ranked_hypotheses(response)
    allowed_targets = _allowed_targets(response, call)
    preferred_tool = _preferred_tool(call, ranked_hypotheses, allowed_targets)
    return (
        True,
        ToolExecutionPayload(
            runtime_updates=RuntimeToolContext(preferred_tool=preferred_tool),
            workspace_updates=WorkspaceToolContext(allowed_target_files=allowed_targets),
            failure_updates=FailureToolContext(
                explanations=explanations,
                concepts=concepts,
                ranked_hypotheses=ranked_hypotheses,
            ),
            audit_updates=AuditToolContext(
                llm_calls=["call_llm_structured"],
                run_summary={
                    "diagnosis": response.diagnosis,
                    "blocker": response.blocker,
                    "ideal_outcome": response.ideal_outcome,
                    "allowed_targets": ", ".join(allowed_targets),
                },
            ),
        ),
        _render_response_message(response),
        [],
    )


def _preferred_tool(
    call: ToolCall,
    ranked_hypotheses: list[RankedHypothesis],
    allowed_targets: list[str],
) -> str:
    if not (ranked_hypotheses or allowed_targets):
        return ""
    active_skills = set(call.context.runtime.active_skills)
    if call.context.runtime.active_agent == "improver" or "improvement-loop" in active_skills:
        return "delegate_task"
    return "mutate_agent_files"


def _structured_response(call: ToolCall) -> StructuredLLMResponse:
    focus_files = _focus_files(call)
    model_result = call_model_structured(
        StructuredLLMResponse,
        system_prompt=(
            "You are the diagnosis and hypothesis planner for an eval-improvement loop. "
            "Given the runtime context, return one concise diagnosis plus up to three small, "
            "targeted hypotheses. Each hypothesis must name the primary durable files to change. "
            "Treat prior successful mutation patterns as survivor genes to preserve or extend. "
            "Treat recent flat or regressive mutations as dead ends to avoid repeating."
        ),
        user_prompt=(
            f"Goal: {call.goal_id}\n"
            f"Agent text:\n{call.agent_text or ''}\n\n"
            f"Focus files:\n{', '.join(focus_files) or 'None provided'}\n\n"
            f"Prior mutation memory:\n{_recent_learning(call)}\n\n"
            f"Context:\n{call.context.model_dump(exclude_none=True)}\n\n"
            "Return JSON with this exact schema:\n"
            '{"diagnosis": "text", "blocker": "text", "ideal_outcome": "text", '
            '"hypotheses": ['
            '{"name": "text", "score": 0.0, "rationale": "text", '
            '"target_files": ["path/to/file"], "ideal_outcome": "text"}]}'
        ),
        working_dir=Path(call.context.workspace.workdir or str(Path.cwd())),
    )
    if model_result is not None:
        return model_result

    try:
        from bolt_merlin.agent.execute import execute
    except ImportError:
        return StructuredLLMResponse(diagnosis="LLM unavailable")

    prompt = (
        "Provide a diagnosis-driven mutation plan for the current eval improvement loop.\n\n"
        f"Scenario summary:\n{call.context.workspace.scenario_summary or 'No scenario summary available'}\n\n"
        f"Focus files:\n{', '.join(focus_files) or 'None provided'}\n\n"
        f"Failure family:\n{call.context.failure.failure_family or 'No failure family'}\n\n"
        f"Failure explanations:\n{' '.join(call.context.failure.explanations) or 'No failure explanations'}\n\n"
        f"Current allowed targets:\n{', '.join(call.context.workspace.allowed_target_files) or 'None'}\n\n"
        f"Scenario required files:\n{', '.join(call.context.workspace.scenario_required_files) or 'None'}\n\n"
        f"Prior mutation memory:\n{_recent_learning(call)}\n\n"
        "Return JSON with this exact shape:\n"
        '{"diagnosis": "text", "blocker": "text", "ideal_outcome": "text", '
        '"hypotheses": ['
        '{"name": "text", "score": 0.0, "rationale": "text", '
        '"target_files": ["path/to/file"], "ideal_outcome": "text"}]}'
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
        try:
            return StructuredLLMResponse.model_validate(json.loads(result.response))
        except (json.JSONDecodeError, pd.ValidationError):
            return StructuredLLMResponse(diagnosis="LLM unavailable")
    return StructuredLLMResponse(diagnosis="LLM unavailable")


def _extract_signal_lists(response: StructuredLLMResponse) -> tuple[list[str], list[str]]:
    explanations: list[str] = []
    if response.diagnosis.strip():
        explanations.append(response.diagnosis.strip())
    if response.blocker.strip():
        explanations.append(response.blocker.strip())
    concepts = [
        hypothesis.name.strip() for hypothesis in response.hypotheses if hypothesis.name.strip()
    ]
    return explanations[:3], concepts[:5]


def _ranked_hypotheses(response: StructuredLLMResponse) -> list[RankedHypothesis]:
    return [
        RankedHypothesis(
            name=hypothesis.name.strip(),
            score=hypothesis.score,
            rationale=hypothesis.rationale.strip(),
            target_files=[path.strip() for path in hypothesis.target_files if path.strip()],
            ideal_outcome=hypothesis.ideal_outcome.strip(),
        )
        for hypothesis in response.hypotheses
        if hypothesis.name.strip()
    ]


def _allowed_targets(response: StructuredLLMResponse, call: ToolCall) -> list[str]:
    workspace_root = Path(call.context.workspace.repo_root or call.context.workspace.workdir or ".")
    ordered: list[str] = []
    for hypothesis in response.hypotheses:
        for path in hypothesis.target_files:
            cleaned = workspace_relative_string(path, workspace_root)
            if cleaned and cleaned not in ordered:
                ordered.append(cleaned)
    for path in call.context.workspace.allowed_target_files:
        normalized = workspace_relative_string(path, workspace_root)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    for path in _focus_files(call):
        normalized = workspace_relative_string(path, workspace_root)
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered[:8]


def _focus_files(call: ToolCall) -> list[str]:
    raw_focus = call.tool_args.get("focus_files")
    if not isinstance(raw_focus, list):
        return []
    ordered: list[str] = []
    for item in raw_focus:
        if isinstance(item, str) and item.strip() and item not in ordered:
            ordered.append(item)
    return ordered[:8]


def _render_response_message(response: StructuredLLMResponse) -> str:
    lines = [f"Diagnosis: {response.diagnosis}"]
    if response.blocker.strip():
        lines.append(f"Blocker: {response.blocker}")
    if response.ideal_outcome.strip():
        lines.append(f"Ideal outcome: {response.ideal_outcome}")
    for hypothesis in response.hypotheses[:3]:
        lines.append(
            f"- {hypothesis.name} (score {hypothesis.score:.2f})"
            + (f" targets: {', '.join(hypothesis.target_files)}" if hypothesis.target_files else "")
        )
    return "\n".join(lines)


def _recent_learning(call: ToolCall) -> str:
    lines: list[str] = []
    working_memory = call.context.memory.working_memory
    active_hypotheses = working_memory.get("active_hypotheses")
    if isinstance(active_hypotheses, list) and active_hypotheses:
        lines.append("- Active survivor candidates:")
        for item in active_hypotheses[:3]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip() or "unnamed"
            score = item.get("score")
            targets = item.get("target_files")
            target_text = ", ".join(targets[:3]) if isinstance(targets, list) else ""
            score_text = f" score={float(score):.2f}" if isinstance(score, int | float) else ""
            target_fragment = f" targets={target_text}" if target_text else ""
            lines.append(f"  - {name}{score_text}{target_fragment}")
    last_result = working_memory.get("last_result")
    if isinstance(last_result, dict):
        status = str(last_result.get("status", "")).strip() or "unknown"
        delta = last_result.get("score_delta")
        target_files = last_result.get("target_files")
        target_text = ", ".join(target_files[:3]) if isinstance(target_files, list) else ""
        delta_text = f" delta={float(delta):+.4f}" if isinstance(delta, int | float) else ""
        target_fragment = f" targets={target_text}" if target_text else ""
        lines.append(f"- Last validation outcome: {status}{delta_text}{target_fragment}")

    session_memory = call.context.memory.session_memory
    validations = session_memory.get("validations")
    if isinstance(validations, list) and validations:
        lines.append("- Recent validation history:")
        for item in validations[-3:]:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).strip() or "unknown"
            delta = item.get("score_delta")
            hypothesis_name = str(item.get("hypothesis", "")).strip() or "unnamed"
            delta_text = f" delta={float(delta):+.4f}" if isinstance(delta, int | float) else ""
            lines.append(f"  - {hypothesis_name}: {status}{delta_text}")

    semantic_memory = call.context.memory.semantic_memory
    rules = semantic_memory.get("rules")
    if isinstance(rules, list) and rules:
        structured_rules = [item for item in rules if isinstance(item, dict)]
        if structured_rules:
            lines.append("- Durable survivor rules:")
            for item in structured_rules[-3:]:
                if item.get("entry_type") not in {"survivor_gene", "rejected_gene"}:
                    continue
                status = str(item.get("status", "")).strip() or "unknown"
                hypothesis_name = str(item.get("hypothesis", "")).strip() or "unnamed"
                delta = item.get("score_delta")
                delta_text = f" delta={float(delta):+.4f}" if isinstance(delta, int | float) else ""
                lines.append(f"  - {hypothesis_name}: {status}{delta_text}")

    return "\n".join(lines) or "No prior mutation memory recorded yet."


COMMAND = ToolCommand(
    name="call_llm_structured",
    summary="Call an LLM for structured output.",
    when_to_use=["When this exact capability is needed"],
    when_not_to_use=["When required inputs are missing"],
    input_schema=basic_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="focus_files",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Optional file paths to prioritize in the diagnosis",
            )
        ],
        required=[],
    ),
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
