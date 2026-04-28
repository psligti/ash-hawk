from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._native_tooling import (
    is_forbidden_path,
    resolve_path,
    workspace_relative_string,
)
from ash_hawk.thin_runtime.tool_impl._workspace_targets import rank_workspace_targets
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    requested_targets = call.tool_args.get("target_files")
    if not isinstance(requested_targets, list):
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: target_files",
            ["missing_target_files"],
        )
    workspace_root = Path(call.context.workspace.repo_root or call.context.workspace.workdir or ".")
    allowed: list[str] = []
    seen: set[str] = set()
    for item in requested_targets:
        if not isinstance(item, str) or not item.strip():
            continue
        resolved = resolve_path(item, workspace_root)
        normalized = workspace_relative_string(resolved, workspace_root)
        if normalized is None:
            continue
        if normalized in seen:
            continue
        if is_forbidden_path(resolved):
            continue
        seen.add(normalized)
        allowed.append(normalized)
        if len(allowed) >= 8:
            break
    if not allowed:
        return (
            False,
            ToolExecutionPayload(),
            "Tool argument target_files must include at least one file path",
            ["empty_target_files"],
        )
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
    diagnosis_targets = [
        path
        for hypothesis in call.context.failure.ranked_hypotheses
        for path in hypothesis.target_files
        if path
    ]
    required_files = list(workspace.scenario_required_files)
    candidates = list(workspace.changed_files)

    ranked = rank_workspace_targets(candidates)
    ordered: list[str] = []
    for path in diagnosis_targets + required_files + ranked:
        if path not in ordered:
            ordered.append(path)
    return ordered


COMMAND = ToolCommand(
    name="scope_workspace",
    summary="Scope workspace.",
    when_to_use=["When workspace targets have been identified and should be narrowed explicitly"],
    when_not_to_use=["When you do not yet know which files should remain in scope"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="target_files",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.STRING,
                description="Ordered list of file paths to keep in active scope",
                required=True,
            )
        ],
        required=["target_files"],
    ),
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
