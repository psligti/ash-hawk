from __future__ import annotations

import shlex
import sys
from pathlib import Path

from ash_hawk.scenario.adapters.coding_agent_subprocess import CodingAgentSubprocessAdapter


def test_coding_agent_subprocess_adapter_diff_and_verify() -> None:
    adapter = CodingAgentSubprocessAdapter()
    repo_root = Path(__file__).resolve().parents[2]
    repo_fixture = "examples/fixtures/repos/coding-agent-smoke"

    python_exe = shlex.quote(sys.executable)
    command = f"{python_exe} apply_fix.py"
    verify_command = f"{python_exe} -c \"print('verify')\""

    scenario = {
        "id": "coding-agent-subprocess",
        "sut": {
            "type": "coding_agent",
            "adapter": "coding_agent_subprocess",
            "config": {"command": command, "verify_commands": [verify_command]},
        },
        "inputs": {"repo_fixture": repo_fixture},
        "tools": {},
        "expectations": {},
        "budgets": {},
    }

    result = adapter.run_scenario(
        scenario=scenario,
        workdir=repo_root,
        tooling_harness={},
        budgets={},
    )

    assert result.final_output == ""
    assert set(result.artifacts.keys()) == {"diff.patch", "stdout.txt", "stderr.txt"}
    assert "version=2" in result.artifacts["diff.patch"]

    trace_events = [e.model_dump() for e in result.trace_events]
    diff_events = [e for e in trace_events if e.get("event_type") == "DiffEvent"]
    assert diff_events
    assert diff_events[0]["data"]["changed_files"] == 1
    assert diff_events[0]["data"]["added_lines"] == 1

    tool_calls = [e for e in trace_events if e.get("event_type") == "ToolCallEvent"]
    tool_results = [e for e in trace_events if e.get("event_type") == "ToolResultEvent"]

    assert any(call["data"]["input"]["command"] == command for call in tool_calls)
    assert any(call["data"]["input"]["command"] == verify_command for call in tool_calls)
    assert any(res["data"]["result"]["exit_code"] == 0 for res in tool_results)

    artifact_events = [e for e in trace_events if e.get("event_type") == "ArtifactEvent"]
    artifact_keys = {event["data"]["artifact_key"] for event in artifact_events}
    assert {"diff.patch", "stdout.txt", "stderr.txt"}.issubset(artifact_keys)
