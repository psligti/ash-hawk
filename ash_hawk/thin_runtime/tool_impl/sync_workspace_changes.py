from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._evaluation_signals import collect_evaluation_signals
from ash_hawk.thin_runtime.tool_impl._isolated_workspace import (
    cleanup_isolated_workspace,
    sync_isolated_changes,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    RuntimeToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    isolated_path_raw = call.context.workspace.isolated_workspace_path
    primary_root_raw = (
        call.context.workspace.primary_workdir
        or call.context.workspace.repo_root
        or call.context.workspace.workdir
    )
    mutated_files = list(call.context.workspace.mutated_files)
    signals = collect_evaluation_signals(call)
    if not isolated_path_raw or not primary_root_raw:
        return (
            False,
            ToolExecutionPayload(),
            "No isolated workspace is available to sync",
            ["missing_isolated_workspace"],
        )
    if not mutated_files:
        return (
            False,
            ToolExecutionPayload(),
            "No mutated files are available to sync",
            ["missing_mutated_files"],
        )
    if signals.aggregate_passed is not True or signals.score_regressed:
        return (
            False,
            ToolExecutionPayload(),
            "Refusing to sync candidate changes before validation passes cleanly",
            ["validation_not_clean"],
        )

    isolated_root = Path(isolated_path_raw).resolve()
    primary_root = Path(primary_root_raw).resolve()
    synced_files = sync_isolated_changes(
        primary_root=primary_root,
        isolated_root=isolated_root,
        relative_paths=mutated_files,
    )
    cleanup_isolated_workspace(isolated_root)
    payload = ToolExecutionPayload(
        runtime_updates=RuntimeToolContext(
            stop_reason="validated candidate synced back to primary workspace"
        ),
        workspace_updates=WorkspaceToolContext(
            isolated_workspace=False,
            isolated_workspace_path="",
            changed_files=synced_files,
            workdir=str(primary_root),
            scenario_path=call.context.workspace.source_scenario_path
            or call.context.workspace.scenario_path,
            source_scenario_path="",
        ),
        audit_updates=AuditToolContext(
            sync_events=["synced_back"],
            run_summary={"synced_file_count": str(len(synced_files))},
            diff_report={"files": len(synced_files)},
        ),
        stop=True,
    )
    return (
        True,
        payload,
        f"Synced {len(synced_files)} validated file(s) back to the primary workspace",
        [],
    )


COMMAND = ToolCommand(
    name="sync_workspace_changes",
    summary="Sync workspace changes.",
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
