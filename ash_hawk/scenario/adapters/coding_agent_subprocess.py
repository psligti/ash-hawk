from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Coroutine, TypeVar

from ash_hawk.scenario.models import JSONValue, ScenarioAdapterResult, ScenarioTraceEvent
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ArtifactEvent,
    DiffEvent,
    ToolCallEvent,
    ToolResultEvent,
)

_T = TypeVar("_T")


def _run_async(func: Callable[..., Coroutine[Any, Any, _T]], *args: Any, **kwargs: Any) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(func(*args, **kwargs))

    result_container: dict[str, _T] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_container["result"] = asyncio.run(func(*args, **kwargs))
        except BaseException as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container["result"]


async def _run_command(command: str, cwd: Path) -> dict[str, Any]:
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    return {
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "exit_code": process.returncode if process.returncode is not None else 1,
    }


def _run_command_sync(command: str, cwd: Path) -> dict[str, Any]:
    return _run_async(_run_command, command, cwd)


def _require_success(result: dict[str, Any], command: str) -> None:
    if result.get("exit_code") != 0:
        stderr = result.get("stderr", "")
        raise RuntimeError(f"Command failed ({command}): {stderr}")


def _count_changed_files(status_output: str) -> int:
    paths: set[str] = set()
    for line in status_output.splitlines():
        if not line:
            continue
        path = line[3:] if len(line) > 3 else ""
        if "->" in path:
            path = path.split("->", 1)[1]
        path = path.strip()
        if path:
            paths.add(path)
    return len(paths)


def _count_added_lines(numstat_output: str) -> int:
    total = 0
    for line in numstat_output.splitlines():
        if not line:
            continue
        added = line.split("\t", 1)[0]
        if added.isdigit():
            total += int(added)
    return total


class CodingAgentSubprocessAdapter:
    name: str = "coding_agent_subprocess"

    def run_scenario(
        self,
        scenario: dict[str, JSONValue],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, JSONValue],
    ) -> ScenarioAdapterResult:
        del tooling_harness, budgets

        trace_events: list[dict[str, JSONValue]] = []

        inputs_raw = scenario.get("inputs", {})
        if not isinstance(inputs_raw, dict):
            raise ValueError("Scenario inputs must be a mapping")
        inputs = inputs_raw
        repo_fixture_raw = inputs.get("repo_fixture")
        if not isinstance(repo_fixture_raw, str) or not repo_fixture_raw.strip():
            raise ValueError("Scenario inputs must include a repo_fixture path")

        fixture_path = Path(repo_fixture_raw)
        if not fixture_path.is_absolute():
            fixture_path = (workdir / fixture_path).resolve()
        if not fixture_path.exists():
            raise ValueError(f"Repo fixture not found: {fixture_path}")

        sut_raw = scenario.get("sut", {})
        if not isinstance(sut_raw, dict):
            raise ValueError("Scenario SUT must be a mapping")

        sut: dict[str, Any] = dict(sut_raw)
        sut_config_raw = sut.get("config", {})
        if not isinstance(sut_config_raw, dict):
            raise ValueError("Scenario SUT config must be a mapping")
        sut_config: dict[str, Any] = {str(key): value for key, value in sut_config_raw.items()}

        command_raw = sut_config.get("command")
        if not isinstance(command_raw, str) or not command_raw.strip():
            raise ValueError("Scenario SUT config must include command")
        command: str = command_raw

        verify_commands_raw = sut_config.get("verify_commands")
        verify_commands: list[str] = []
        if isinstance(verify_commands_raw, list):
            verify_commands = [cmd for cmd in verify_commands_raw if isinstance(cmd, str)]

        with TemporaryDirectory(dir=workdir, prefix="coding-agent-") as temp_dir:
            repo_dir = Path(temp_dir) / "repo"
            shutil.copytree(fixture_path, repo_dir)

            if not (repo_dir / ".git").exists():
                _require_success(_run_command_sync("git init", repo_dir), "git init")
                _require_success(_run_command_sync("git add .", repo_dir), "git add")
                commit_cmd = (
                    'git -c user.name="Ash Hawk" '
                    '-c user.email="ash-hawk@example.com" '
                    'commit -m "Init fixture"'
                )
                _require_success(_run_command_sync(commit_cmd, repo_dir), commit_cmd)

            trace_events.append(
                ToolCallEvent.create(
                    ts=DEFAULT_TRACE_TS,
                    data={"tool": "subprocess", "input": {"command": command}},
                ).model_dump()
            )
            command_result = _run_command_sync(command, repo_dir)
            trace_events.append(
                ToolResultEvent.create(
                    ts=DEFAULT_TRACE_TS,
                    data={"tool": "subprocess", "result": command_result},
                ).model_dump()
            )

            for verify_command_str in verify_commands:
                if not verify_command_str.strip():
                    continue
                trace_events.append(
                    ToolCallEvent.create(
                        ts=DEFAULT_TRACE_TS,
                        data={"tool": "subprocess", "input": {"command": verify_command_str}},
                    ).model_dump()
                )
                verify_result = _run_command_sync(verify_command_str, repo_dir)
                trace_events.append(
                    ToolResultEvent.create(
                        ts=DEFAULT_TRACE_TS,
                        data={"tool": "subprocess", "result": verify_result},
                    ).model_dump()
                )

            diff_output = _run_command_sync("git diff", repo_dir)["stdout"]
            status_output = _run_command_sync("git status --porcelain", repo_dir)["stdout"]
            numstat_output = _run_command_sync("git diff --numstat", repo_dir)["stdout"]

            diff_event = DiffEvent.create(
                ts=DEFAULT_TRACE_TS,
                data={
                    "patch_text": diff_output,
                    "changed_files": _count_changed_files(status_output),
                    "added_lines": _count_added_lines(numstat_output),
                },
            )
            trace_events.append(diff_event.model_dump())

            artifacts = {
                "diff.patch": diff_output,
                "stdout.txt": command_result.get("stdout", ""),
                "stderr.txt": command_result.get("stderr", ""),
            }

            for artifact_key in artifacts.keys():
                trace_events.append(
                    ArtifactEvent.create(
                        ts=DEFAULT_TRACE_TS,
                        data={"artifact_key": artifact_key},
                    ).model_dump()
                )

            return ScenarioAdapterResult(
                final_output=command_result.get("stdout", ""),
                trace_events=[ScenarioTraceEvent.model_validate(event) for event in trace_events],
                artifacts=artifacts,
            )
