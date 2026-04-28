from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._native_tooling import (
    check_workspace_path_error,
    format_error,
    format_glob_output,
    is_dangerous_glob,
    is_forbidden_glob_pattern,
    is_forbidden_path,
    resolve_path,
    safe_walk,
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
    pattern = call.tool_args.get("pattern")
    search_path = call.tool_args.get("path")
    if not isinstance(pattern, str) or not pattern.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: pattern",
            ["missing_pattern"],
        )
    base = (
        resolve_path(search_path, workdir)
        if isinstance(search_path, str) and search_path
        else workdir
    )
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["glob"]))
    resolved_base, workspace_err = check_workspace_path_error(base, workdir, "glob")
    if workspace_err or resolved_base is None:
        detail = (workspace_err or "❌ Access denied: path validation failed.").removeprefix("❌ ")
        msg = format_error("glob", detail)
        return False, payload, msg, [msg]
    base = resolved_base
    if not base.exists() or not base.is_dir():
        msg = format_error("glob", f"Directory not found: {base}")
        return False, payload, msg, [msg]
    if is_dangerous_glob(pattern):
        msg = format_error(
            "glob",
            f"Pattern '{pattern}' is too broad and may cause performance issues. Use a more specific pattern.",
        )
        return False, payload, msg, [msg]
    forbidden = is_forbidden_glob_pattern(pattern)
    if forbidden:
        msg = format_error(
            "glob",
            f"Access denied: pattern '{pattern}' matches forbidden pattern '{forbidden}'. Do not access evaluation infrastructure.",
        )
        return False, payload, msg, [msg]
    try:
        all_files = safe_walk(base)
        matches = sorted(
            str(path)
            for path in all_files
            if path.match(pattern) and path.is_file() and not is_forbidden_path(path)
        )
    except Exception as exc:  # noqa: BLE001
        msg = format_error("glob", f"Glob failed: {exc}")
        return False, payload, msg, [msg]
    return True, payload, format_glob_output(pattern, base, matches), []


COMMAND = ToolCommand(
    name="glob",
    summary="Find files matching a glob pattern.",
    when_to_use=["When file discovery by pattern is needed"],
    when_not_to_use=["When file content search is needed instead"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="pattern",
                type=SchemaFieldType.STRING,
                description="Glob pattern to match, e.g. '**/*.py'",
                required=True,
            ),
            ToolFieldSpec(
                name="path",
                type=SchemaFieldType.STRING,
                description="Optional base directory to search in",
            ),
        ],
        required=["pattern"],
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
