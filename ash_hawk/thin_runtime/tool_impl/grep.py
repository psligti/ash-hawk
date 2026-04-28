from __future__ import annotations

import re
from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._native_tooling import (
    MAX_GREP_FILE_SIZE,
    MAX_GREP_RESULTS,
    check_workspace_path_error,
    format_error,
    format_grep_output,
    is_forbidden_path,
    matches_include,
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
    include = call.tool_args.get("include", "*")
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
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["grep"]))
    resolved_base, workspace_err = check_workspace_path_error(base, workdir, "grep")
    if workspace_err or resolved_base is None:
        detail = (workspace_err or "❌ Access denied: path validation failed.").removeprefix("❌ ")
        msg = format_error("grep", detail)
        return False, payload, msg, [msg]
    base = resolved_base
    if not base.exists() or not base.is_dir():
        msg = format_error("grep", f"Directory not found: {base}")
        return False, payload, msg, [msg]
    forbidden = is_forbidden_path(base)
    if forbidden:
        msg = format_error(
            "grep",
            f"Access denied: path '{base}' matches forbidden pattern '{forbidden}'. Do not access evaluation infrastructure.",
        )
        return False, payload, msg, [msg]
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        msg = format_error("grep", f"Invalid regex: {exc}")
        return False, payload, msg, [msg]
    try:
        results: list[str] = []
        truncated = False
        for file_path in sorted(safe_walk(base)):
            if is_forbidden_path(file_path):
                continue
            if (
                isinstance(include, str)
                and include != "*"
                and not matches_include(file_path, include)
            ):
                continue
            if file_path.stat().st_size > MAX_GREP_FILE_SIZE:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeError):
                continue
            for index, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{file_path}:{index}: {line.strip()}")
                    if len(results) >= MAX_GREP_RESULTS:
                        truncated = True
                        break
            if truncated:
                break
    except Exception as exc:  # noqa: BLE001
        msg = format_error("grep", f"Grep failed: {exc}")
        return False, payload, msg, [msg]
    return True, payload, format_grep_output(pattern, base, results, truncated=truncated), []


COMMAND = ToolCommand(
    name="grep",
    summary="Search file contents using regex.",
    when_to_use=["When regex/text search is needed across files"],
    when_not_to_use=["When file name matching alone is enough"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="pattern",
                type=SchemaFieldType.STRING,
                description="Regex pattern to search for",
                required=True,
            ),
            ToolFieldSpec(
                name="include",
                type=SchemaFieldType.STRING,
                description="Optional include glob like '*.py'",
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
