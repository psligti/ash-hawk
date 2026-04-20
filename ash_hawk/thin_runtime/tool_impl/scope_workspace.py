from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._workspace_targets import rank_workspace_targets
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    allowed = _candidate_targets(call)[:8]
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(allowed_target_files=allowed),
        audit_updates=AuditToolContext(
            run_summary={
                "allowed_target_file_count": str(len(allowed)),
                "allowed_target_files": ", ".join(allowed),
                "total_changed_files": str(len(call.context.workspace.changed_files)),
            },
            tool_usage=["scope_workspace"],
        ),
    )
    preview_text = ", ".join(allowed) if allowed else "no scoped targets"
    return True, payload, f"Scoped workspace to {len(allowed)} targets: {preview_text}", []


def _candidate_targets(call: ToolCall) -> list[str]:
    workspace = call.context.workspace
    required_files = list(workspace.scenario_required_files)
    candidates = list(workspace.changed_files)

    ranked = rank_workspace_targets(candidates)
    ordered: list[str] = []
    for path in required_files + ranked:
        if path not in ordered:
            ordered.append(path)
    return ordered


COMMAND = ToolCommand(
    name="scope_workspace",
    summary="Scope workspace.",
    when_to_use=["When workspace state must be inspected or updated"],
    when_not_to_use=["When no workspace context exists"],
    input_schema=context_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["filesystem"],
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
