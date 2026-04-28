from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl import (
    bash,
    edit,
    glob,
    grep,
    mutate_agent_files,
    read,
    test,
    todoread,
    todowrite,
    write,
)
from ash_hawk.thin_runtime.tool_types import ToolCallContext, WorkspaceToolContext


def _call(tool_name: str, tmp_path: Path, **tool_args: object) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        goal_id="goal-1",
        tool_args=tool_args,
        context=ToolCallContext(workspace=WorkspaceToolContext(workdir=str(tmp_path))),
    )


def test_native_read_reads_file_with_line_numbers(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = read.run(_call("read", tmp_path, file_path=str(file_path), offset=2, limit=1))

    assert result.success is True
    assert "📄 **sample.txt**" in result.payload.message
    assert "2: beta" in result.payload.message


def test_native_read_blocks_forbidden_scenario_file(tmp_path: Path) -> None:
    forbidden = tmp_path / "example.scenario.yaml"
    forbidden.write_text("id: scenario\n", encoding="utf-8")

    result = read.run(_call("read", tmp_path, file_path=str(forbidden)))

    assert result.success is False
    assert ".scenario.yaml" in (result.error or "")


def test_native_edit_replaces_exact_string(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("value = 1\n", encoding="utf-8")

    result = edit.run(
        _call(
            "edit",
            tmp_path,
            file_path=str(file_path),
            old_string="value = 1",
            new_string="value = 2",
        )
    )

    assert result.success is True
    assert file_path.read_text(encoding="utf-8") == "value = 2\n"
    assert "POST-EDIT VERIFICATION" in result.payload.message


def test_native_edit_blocks_forbidden_workspace_file(tmp_path: Path) -> None:
    forbidden = tmp_path / ".dawn-kestrel" / "agent_config.yaml"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("name: test\n", encoding="utf-8")

    result = edit.run(
        _call(
            "edit",
            tmp_path,
            file_path=str(forbidden),
            old_string="name: test",
            new_string="name: changed",
        )
    )

    assert result.success is False
    assert ".dawn-kestrel" in (result.error or "")


def test_native_write_writes_content(tmp_path: Path) -> None:
    file_path = tmp_path / "new.txt"

    result = write.run(_call("write", tmp_path, file_path=str(file_path), content="hello\n"))

    assert result.success is True
    assert file_path.read_text(encoding="utf-8") == "hello\n"
    assert "✅ Wrote **new.txt**" in result.payload.message


def test_native_write_blocks_forbidden_workspace_file(tmp_path: Path) -> None:
    forbidden = tmp_path / ".ash-hawk" / "state.json"

    result = write.run(_call("write", tmp_path, file_path=str(forbidden), content="{}\n"))

    assert result.success is False
    assert ".ash-hawk" in (result.error or "")


def test_native_bash_blocks_forbidden_command(tmp_path: Path) -> None:
    result = bash.run(_call("bash", tmp_path, command="cat example.scenario.yaml"))

    assert result.success is False
    assert "Access denied" in (result.error or "")


def test_native_read_blocks_absolute_path_escape(tmp_path: Path) -> None:
    result = read.run(_call("read", tmp_path, file_path="/etc/passwd"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_write_blocks_parent_traversal_escape(tmp_path: Path) -> None:
    result = write.run(_call("write", tmp_path, file_path="../../tmp/evil.txt", content="owned\n"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_edit_blocks_absolute_path_escape(tmp_path: Path) -> None:
    result = edit.run(
        _call(
            "edit",
            tmp_path,
            file_path="/etc/hosts",
            old_string="127.0.0.1",
            new_string="127.0.0.2",
        )
    )

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_glob_blocks_escape_base_path(tmp_path: Path) -> None:
    result = glob.run(_call("glob", tmp_path, pattern="*.py", path="../../"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_grep_blocks_absolute_search_path_escape(tmp_path: Path) -> None:
    result = grep.run(_call("grep", tmp_path, pattern="root", path="/etc"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_bash_blocks_workdir_escape(tmp_path: Path) -> None:
    result = bash.run(_call("bash", tmp_path, command="pwd", workdir="../../"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_test_blocks_path_escape(tmp_path: Path) -> None:
    result = test.run(_call("test", tmp_path, path="/"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_todoread_blocks_file_override_escape(tmp_path: Path) -> None:
    result = todoread.run(_call("todoread", tmp_path, file_path="/etc/passwd"))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_todowrite_blocks_file_override_escape(tmp_path: Path) -> None:
    result = todowrite.run(_call("todowrite", tmp_path, file_path="../../tmp/todos.md", todos=[]))

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_mutate_agent_files_blocks_target_path_escape(tmp_path: Path) -> None:
    result = mutate_agent_files.run(
        _call("mutate_agent_files", tmp_path, target_file="../../etc/passwd")
    )

    assert result.success is False
    assert "outside the workspace root" in (result.error or "")


def test_native_todo_roundtrip_uses_native_storage(tmp_path: Path) -> None:
    write_result = todowrite.run(
        _call(
            "todowrite",
            tmp_path,
            todos=[
                {"content": "investigate", "status": "in_progress", "priority": "high"},
                {"content": "verify", "status": "pending", "priority": "medium"},
            ],
        )
    )
    read_result = todoread.run(_call("todoread", tmp_path))

    assert write_result.success is True
    assert read_result.success is True
    assert "investigate" in read_result.payload.message
    assert ".ash-hawk" not in (read_result.error or "")
