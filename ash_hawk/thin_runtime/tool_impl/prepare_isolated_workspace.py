from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._isolated_workspace import create_isolated_workspace
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    primary_root = Path(call.context.workspace.workdir or str(Path.cwd())).resolve()
    target_files = _target_files(call)
    if not target_files:
        return (
            False,
            ToolExecutionPayload(),
            "No mutation target files are available for isolated execution",
            ["no_target_files"],
        )

    snapshot = create_isolated_workspace(
        primary_root=primary_root,
        target_files=target_files,
        scenario_path=call.context.workspace.scenario_path,
        scenario_targets=list(call.context.workspace.scenario_targets),
        scenario_required_files=list(call.context.workspace.scenario_required_files),
        agent_config=call.context.workspace.agent_config,
    )
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(
            isolated_workspace=True,
            isolated_workspace_path=str(snapshot.isolated_root),
            primary_workdir=str(snapshot.primary_root),
            workdir=str(snapshot.isolated_root),
            source_scenario_path=snapshot.source_scenario_path,
            scenario_path=snapshot.isolated_scenario_path,
            changed_files=snapshot.copied_files,
        ),
        audit_updates=AuditToolContext(
            artifacts=[f"isolated-workspace:{snapshot.isolated_root}"],
            sync_events=["isolated_workspace_prepared"],
            run_summary={
                "isolated_workspace_path": str(snapshot.isolated_root),
                "copied_file_count": str(len(snapshot.copied_files)),
            },
        ),
    )
    return (
        True,
        payload,
        f"Prepared isolated workspace at {snapshot.isolated_root} with {len(snapshot.copied_files)} copied file(s)",
        [],
    )


def _target_files(call: ToolCall) -> list[str]:
    ordered: list[str] = []
    for candidate in list(call.context.workspace.allowed_target_files):
        cleaned = candidate.strip()
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return ordered[:5]


COMMAND = ToolCommand(
    name="prepare_isolated_workspace",
    summary="Prepare isolated workspace.",
    when_to_use=["When workspace state must be inspected or updated"],
    when_not_to_use=["When no workspace context exists"],
    input_schema=context_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["filesystem"],
    risk_level="medium",
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
