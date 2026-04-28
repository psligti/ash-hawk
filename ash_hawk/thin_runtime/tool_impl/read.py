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
    format_error,
    format_read_output,
    resolve_path,
    suggest_glob_for_missing_file,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    requested_path = call.tool_args.get("file_path") or call.tool_args.get("path")
    if not isinstance(requested_path, str) or not requested_path.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: file_path",
            ["missing_file_path"],
        )
    resolved_file_path, workspace_err = check_workspace_path_error(
        resolve_path(requested_path.strip(), workdir), workdir, "read"
    )
    if workspace_err or resolved_file_path is None:
        payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
        message = workspace_err or "❌ Access denied: path validation failed."
        return False, payload, message, [message]
    file_path = resolved_file_path

    forbidden_err = check_forbidden_path_error(file_path, "read")
    if forbidden_err:
        payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
        return False, payload, forbidden_err, [forbidden_err]

    if not file_path.exists() or not file_path.is_file():
        msg = format_error(
            "read",
            f"File not found: {file_path}.{suggest_glob_for_missing_file(file_path, workdir)}",
        )
        payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
        return False, payload, msg, [msg]

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # noqa: BLE001
        msg = format_error("read", f"Failed to read {file_path}: {exc}")
        payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
        return False, payload, msg, [msg]

    offset = call.tool_args.get("offset", 1)
    limit = call.tool_args.get("limit", len(lines))
    if not isinstance(offset, int) or offset < 1:
        offset = 1
    if not isinstance(limit, int) or limit < 1:
        limit = len(lines)
    selected = lines[offset - 1 : offset - 1 + limit]
    numbered = [f"{index + offset}: {line}" for index, line in enumerate(selected)]
    output = format_read_output(
        file_path, "\n".join(numbered), offset=offset, total_lines=len(lines)
    )
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
    return True, payload, output, []


COMMAND = ToolCommand(
    name="read",
    summary="Read file contents with line numbers.",
    when_to_use=["When exact file contents are needed"],
    when_not_to_use=["When you do not know the exact file path yet"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="file_path",
                type=SchemaFieldType.STRING,
                description="Absolute or workspace-relative file path to read",
                required=True,
            ),
            ToolFieldSpec(
                name="offset",
                type=SchemaFieldType.INTEGER,
                description="1-indexed starting line number",
            ),
            ToolFieldSpec(
                name="limit",
                type=SchemaFieldType.INTEGER,
                description="Maximum number of lines to read",
            ),
        ],
        required=["file_path"],
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
