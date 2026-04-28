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
    format_edit_output,
    format_verification_block,
    mini_diff,
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
    old_string = call.tool_args.get("old_string")
    new_string = call.tool_args.get("new_string")
    if not isinstance(raw_file_path, str) or not raw_file_path.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: file_path",
            ["missing_file_path"],
        )
    if not isinstance(old_string, str):
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: old_string",
            ["missing_old_string"],
        )
    if not isinstance(new_string, str):
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: new_string",
            ["missing_new_string"],
        )

    resolved_file_path, workspace_err = check_workspace_path_error(
        resolve_path(raw_file_path.strip(), workdir), workdir, "edit"
    )
    if workspace_err or resolved_file_path is None:
        msg = workspace_err or "❌ Access denied: path validation failed."
        return False, ToolExecutionPayload(), msg, [msg]
    file_path = resolved_file_path
    forbidden_err = check_forbidden_path_error(file_path, "edit")
    if forbidden_err:
        return False, ToolExecutionPayload(), forbidden_err, [forbidden_err]
    if not file_path.exists():
        msg = f"❌ File not found: {file_path}.{suggest_glob_for_missing_file(file_path, workdir)}"
        return False, ToolExecutionPayload(), msg, [msg]
    if not old_string:
        msg = "❌ Edit FAILED — old_string is empty. An empty old_string matches nothing."
        return False, ToolExecutionPayload(), msg, [msg]

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        msg = f"❌ Failed to read {file_path}: {exc}"
        return False, ToolExecutionPayload(), msg, [msg]

    count = content.count(old_string)
    if count == 0:
        msg = (
            f"❌ old_string not found in {file_path}. The edit FAILED — the file was NOT modified. "
            "Re-read the file to get the exact current content, then retry with an exact match."
        )
        return False, ToolExecutionPayload(), msg, [msg]
    if count > 1:
        msg = (
            f"❌ old_string found {count} times in {file_path} — must be unique. The edit FAILED — the file was NOT modified. "
            "Include more surrounding context to make the match unique."
        )
        return False, ToolExecutionPayload(), msg, [msg]

    line_num: int | None = None
    first_old_line = old_string.splitlines()[0] if old_string.splitlines() else old_string
    for index, line in enumerate(content.splitlines(), 1):
        if first_old_line in line:
            line_num = index
            break

    new_content = content.replace(old_string, new_string)
    if new_content == content:
        msg = (
            f"❌ Edit produced no diff for {file_path}. old_string and new_string result in identical file content. "
            "The file was NOT modified. If the replacement is intentionally different, check for hidden whitespace or encoding differences."
        )
        return False, ToolExecutionPayload(), msg, [msg]

    try:
        file_path.write_text(new_content, encoding="utf-8")
        verify_content = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        msg = f"❌ Failed to edit {file_path}: {exc}"
        return False, ToolExecutionPayload(), msg, [msg]

    if new_string not in verify_content:
        msg = (
            f"❌ Edit verification failed for {file_path}. The file was written but the expected content was not found on re-read. "
            "This may indicate a filesystem issue or encoding problem."
        )
        return False, ToolExecutionPayload(), msg, [msg]

    output = format_edit_output(
        file_path, old_string.splitlines(), new_string.splitlines(), line_num=line_num
    )
    output += format_verification_block(
        file_path, new_string, verify_edit_applied(file_path, new_string)
    )
    target = display_path(file_path, workdir)
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(mutated_files=[target]),
        audit_updates=AuditToolContext(tool_usage=["edit"]),
    )
    return True, payload, output, []


COMMAND = ToolCommand(
    name="edit",
    summary="Replace an exact string in a file.",
    when_to_use=["When an exact in-place file edit is the intended atomic action"],
    when_not_to_use=["When you do not know the exact file path and exact old string yet"],
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
                name="old_string",
                type=SchemaFieldType.STRING,
                description="Exact string to replace",
                required=True,
            ),
            ToolFieldSpec(
                name="new_string",
                type=SchemaFieldType.STRING,
                description="Replacement string",
                required=True,
            ),
        ],
        required=["file_path", "old_string", "new_string"],
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
