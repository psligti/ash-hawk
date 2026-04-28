from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._native_tooling import (
    check_forbidden_path_error,
    check_workspace_path_error,
    display_path,
    format_verification_block,
    format_write_output,
    human_bytes,
    resolve_path,
    suggest_glob_for_missing_file,
    verify_edit_applied,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    raw_file_path = call.tool_args.get("file_path")
    content = call.tool_args.get("content")
    if not isinstance(raw_file_path, str) or not raw_file_path.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: file_path",
            ["missing_file_path"],
        )
    if not isinstance(content, str):
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: content",
            ["missing_content"],
        )

    resolved_file_path, workspace_err = check_workspace_path_error(
        resolve_path(raw_file_path.strip(), workdir), workdir, "write"
    )
    if workspace_err or resolved_file_path is None:
        detail = workspace_err or "Access denied: path validation failed."
        msg = f"❌ **write**: {detail[2:] if detail.startswith('❌ ') else detail}"
        return False, ToolExecutionPayload(), msg, [msg]
    file_path = resolved_file_path
    forbidden_err = check_forbidden_path_error(file_path, "write")
    if forbidden_err:
        msg = f"❌ **write**: {forbidden_err[2:] if forbidden_err.startswith('❌ ') else forbidden_err}"
        return False, ToolExecutionPayload(), msg, [msg]
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        msg = f"❌ **write**: Failed to write {file_path}: {exc}"
        return False, ToolExecutionPayload(), msg, [msg]

    output = format_write_output(file_path, len(content.encode("utf-8")))
    if file_path.parent == workdir:
        output += (
            f"\n⚠️ WARNING: Wrote to workspace root: {file_path.name}. If the target file should be in a subdirectory,"
            f" use `glob('**/{file_path.name}')` to find the correct path first."
        )
    output += format_verification_block(file_path, content, verify_edit_applied(file_path, content))
    target = display_path(file_path, workdir)
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(mutated_files=[target]),
        audit_updates=AuditToolContext(tool_usage=["write"]),
    )
    return True, payload, output, []


COMMAND = ToolCommand(
    name="write",
    summary="Write content to a file.",
    when_to_use=["When a complete file write is the intended atomic action"],
    when_not_to_use=["When you do not know the exact file path yet"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="file_path",
                type=SchemaFieldType.STRING,
                description="Absolute or workspace-relative file path",
                required=True,
            ),
            ToolFieldSpec(
                name="content",
                type=SchemaFieldType.STRING,
                description="Full content to write",
                required=True,
            ),
        ],
        required=["file_path", "content"],
    ),
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
