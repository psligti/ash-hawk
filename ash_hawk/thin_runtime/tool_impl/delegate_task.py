from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    DelegationRequest,
    RuntimeToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
)


def _normalize_string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    delegated_agent = call.tool_args.get("agent_name")
    if not isinstance(delegated_agent, str) or not delegated_agent.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Delegation requires a target agent_name",
            ["Missing required field: agent_name"],
        )

    description = call.tool_args.get("description")
    if not isinstance(description, str) or not description.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Delegation requires a non-empty description",
            ["Missing required field: description"],
        )

    active_skills = set(call.context.runtime.active_skills)
    if "improvement-loop" in active_skills:
        baseline_summary = call.context.evaluation.baseline_summary
        baseline_ready = (
            baseline_summary.status == "completed" or baseline_summary.score is not None
        )
        actionable_targets = (
            list(call.context.workspace.allowed_target_files)
            or list(call.context.workspace.scenario_required_files)
            or list(call.context.workspace.actionable_files)
        )
        if not baseline_ready:
            return (
                False,
                ToolExecutionPayload(),
                "Delegation requires a completed baseline evaluation first",
                ["Missing precondition: baseline evaluation not completed"],
            )
        if not actionable_targets:
            return (
                False,
                ToolExecutionPayload(),
                "Delegation requires at least one mutation-ready target file",
                ["Missing precondition: actionable mutation target not identified"],
            )

    requested_skills = _normalize_string_list(call.tool_args.get("requested_skills"))
    requested_tools = _normalize_string_list(call.tool_args.get("requested_tools"))
    delegated_goal_id = f"{call.goal_id}:{delegated_agent.strip()}"

    payload = ToolExecutionPayload(
        delegation=DelegationRequest(
            agent_name=delegated_agent.strip(),
            requested_skills=requested_skills,
            requested_tools=requested_tools,
            goal_id=delegated_goal_id,
            description=description.strip(),
        ),
        runtime_updates=RuntimeToolContext(
            lead_agent=call.context.runtime.lead_agent,
            active_agent=str(call.context.runtime.active_agent or ""),
            delegated_to=delegated_agent.strip(),
        ),
        audit_updates=AuditToolContext(
            delegation_requests=[
                f"delegate:{delegated_agent.strip()}:{','.join(requested_tools) or 'agent-default-tools'}"
            ]
        ),
    )
    return (
        True,
        payload,
        f"Delegating to {delegated_agent.strip()} for goal {delegated_goal_id}",
        [],
    )


COMMAND = ToolCommand(
    name="delegate_task",
    summary="Delegate a bounded sub-task to another thin runtime agent.",
    when_to_use=[
        "When another agent is better suited for a bounded sub-task",
        "When the current agent should remain orchestration-focused",
    ],
    when_not_to_use=[
        "When the current agent can complete the work directly without quality loss",
        "When delegation boundaries are unclear",
    ],
    input_schema=delegation_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="agent_name",
                type=SchemaFieldType.STRING,
                description="Target delegated agent name",
                required=True,
            ),
            ToolFieldSpec(
                name="description",
                type=SchemaFieldType.STRING,
                description="Concrete delegated task description",
                required=True,
            ),
            ToolFieldSpec(
                name="requested_skills",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Optional skills to activate for delegated agent",
            ),
            ToolFieldSpec(
                name="requested_tools",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Optional tool allowlist to constrain delegated run",
            ),
        ],
        required=["agent_name", "description"],
    ),
    output_schema=standard_output_schema(),
    side_effects=["runtime_state", "audit"],
    risk_level="medium",
    timeout_seconds=10,
    completion_criteria=[
        "Delegation payload includes target agent and bounded description",
        "Requested skills/tools are normalized and deduplicated",
    ],
    escalation_rules=[
        "Escalate when target agent is missing",
        "Escalate when delegation description is empty",
    ],
    executor=_execute,
)


def run(call: ToolCall) -> ToolResult:
    return COMMAND.run(call)
