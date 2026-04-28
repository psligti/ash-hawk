from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl import (
    bash,
    call_llm_structured,
    glob,
    grep,
    mutate_agent_files,
    read,
    scope_workspace,
)


def test_catalog_exposes_tool_owned_dk_input_schemas() -> None:
    assert read.COMMAND.dk_inputs.required == ["file_path"]
    assert glob.COMMAND.dk_inputs.required == ["pattern"]
    assert grep.COMMAND.dk_inputs.required == ["pattern"]
    assert bash.COMMAND.dk_inputs.required == ["command"]
    assert scope_workspace.COMMAND.dk_inputs.required == ["target_files"]
    assert mutate_agent_files.COMMAND.dk_inputs.required == ["target_file"]
    assert call_llm_structured.COMMAND.dk_inputs.required == []


def test_read_requires_explicit_file_path() -> None:
    result = read.run(ToolCall(tool_name="read", goal_id="goal"))
    assert result.success is False
    assert result.error == "Missing required field: file_path"


def test_glob_requires_explicit_pattern() -> None:
    result = glob.run(ToolCall(tool_name="glob", goal_id="goal"))
    assert result.success is False
    assert result.error == "Missing required field: pattern"


def test_grep_requires_explicit_pattern() -> None:
    result = grep.run(ToolCall(tool_name="grep", goal_id="goal"))
    assert result.success is False
    assert result.error == "Missing required field: pattern"


def test_bash_requires_explicit_command() -> None:
    result = bash.run(ToolCall(tool_name="bash", goal_id="goal"))
    assert result.success is False
    assert result.error == "Missing required field: command"


def test_scope_workspace_requires_explicit_target_files() -> None:
    result = scope_workspace.run(ToolCall(tool_name="scope_workspace", goal_id="goal"))
    assert result.success is False
    assert result.error == "Missing required field: target_files"


def test_mutate_requires_explicit_target_file() -> None:
    result = mutate_agent_files.run(ToolCall(tool_name="mutate_agent_files", goal_id="goal"))
    assert result.success is False
    assert result.error == "Missing required field: target_file"
