from __future__ import annotations

from pathlib import Path

import yaml

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    workspace_files = (
        sorted(
            str(path.relative_to(workdir))
            for path in workdir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
        if workdir.exists()
        else []
    )
    preview = workspace_files[:5]
    scenario_targets, scenario_required_files, scenario_summary = load_scenario_brief(
        call.context.workspace.scenario_path,
        workdir,
    )
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(
            workdir=str(workdir.resolve()) if workdir.exists() else str(workdir),
            repo_root=str(workdir.resolve()) if workdir.exists() else str(workdir),
            changed_files=[],
            scenario_path=call.context.workspace.scenario_path,
            agent_config=call.context.workspace.agent_config,
            scenario_targets=scenario_targets,
            scenario_required_files=scenario_required_files,
            scenario_summary=scenario_summary,
        ),
        audit_updates=AuditToolContext(
            run_summary={
                "changed_file_count": str(len(workspace_files)),
                "workspace_file_count": str(len(workspace_files)),
                "changed_files_preview": ", ".join(preview),
                "scenario_path": call.context.workspace.scenario_path or "",
                "agent_config": call.context.workspace.agent_config or "",
                "scenario_target_count": str(len(scenario_targets)),
                "scenario_required_file_count": str(len(scenario_required_files)),
                "scenario_required_files": ", ".join(scenario_required_files),
                "scenario_summary": scenario_summary or "",
            },
            tool_usage=["load_workspace_state"],
        ),
    )
    preview_text = ", ".join(preview) if preview else "no top-level files"
    return (
        True,
        payload,
        f"Loaded workspace state ({len(workspace_files)} files: {preview_text})",
        [],
    )


def load_scenario_brief(
    scenario_path: str | None, workdir: Path
) -> tuple[list[str], list[str], str | None]:
    if not scenario_path:
        return [], [], None
    path = Path(scenario_path)
    if not path.is_absolute():
        path = (workdir / path).resolve()
    if not path.exists():
        return [], [], None

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return [], [], None

    if isinstance(data.get("scenarios"), list):
        scenario_targets: list[str] = []
        scenario_required_files: list[str] = []
        fragments: list[str] = []
        for item in data["scenarios"]:
            if not isinstance(item, dict):
                continue
            raw_scenario = item.get("scenario")
            if isinstance(raw_scenario, str) and raw_scenario.strip():
                resolved_scenario = (path.parent / raw_scenario).resolve()
                scenario_targets.append(str(resolved_scenario))
                if resolved_scenario.exists():
                    nested_data = yaml.safe_load(resolved_scenario.read_text(encoding="utf-8"))
                    for required in _extract_required_files(nested_data):
                        if required not in scenario_required_files:
                            scenario_required_files.append(required)
            focus = item.get("focus")
            if isinstance(focus, list):
                focus_text = ", ".join(str(entry) for entry in focus if str(entry).strip())
                if focus_text:
                    fragments.append(focus_text)
        summary = str(data.get("description", "")).strip()
        if fragments:
            summary = f"{summary} Focus: {'; '.join(fragments)}".strip()
        return scenario_targets, scenario_required_files, summary or None

    description = str(data.get("description", "")).strip()
    intent = (
        str(data.get("inputs", {}).get("intent", "")).strip()
        if isinstance(data.get("inputs"), dict)
        else ""
    )
    summary = " ".join(part for part in [description, intent] if part)
    return [str(path.resolve())], _extract_required_files(data), summary or None


def _extract_required_files(data: object) -> list[str]:
    if not isinstance(data, dict):
        return []
    graders = data.get("graders")
    if not isinstance(graders, list):
        return []
    required_files: list[str] = []
    for grader in graders:
        if not isinstance(grader, dict):
            continue
        config = grader.get("config")
        if not isinstance(config, dict):
            continue
        entries = config.get("required_file_changes")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if isinstance(path, str) and path.strip() and path not in required_files:
                required_files.append(path)
    return required_files


COMMAND = ToolCommand(
    name="load_workspace_state",
    summary="Load workspace state.",
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
