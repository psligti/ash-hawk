"""Tests for the SDK adapter trace event generation."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from ash_hawk.scenario.adapters.sdk_dawn_kestrel import SdkDawnKestrelAdapter
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTranscript


def test_sdk_adapter_produces_policy_decision_events():
    """Test SDK adapter produces PolicyDecisionEvent in trace when tools are called."""
    adapter = SdkDawnKestrelAdapter()

    # Mock the DawnKestrelAgentRunner to return deterministic output
    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Done"},
        ],
        tool_calls=[
            {
                "name": "bash",
                "arguments": {"command": "echo test"},
            }
        ],
        agent_response="Done",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    # Create a tooling harness with a mock call function
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    def mock_tool_call(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool_calls_log.append((tool_name, tool_input))
        return {"stdout": "test output", "exit_code": 0}

    tooling_harness = {
        "policy": {
            "allowed_tools": ["bash"],
            "denied_tools": [],
        },
        "call": mock_tool_call,
    }

    scenario = {
        "id": "test-sdk-scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": ["bash"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    # Patch the DawnKestrelAgentRunner to return our mock transcript
    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        MockRunner.return_value = mock_runner_instance

        final_output, trace_events, artifacts, _ = adapter.run_scenario(
            scenario, workdir, tooling_harness, budgets
        )

    # Verify final output
    assert final_output == "Done"

    # Verify artifacts are empty
    assert artifacts == {}

    # Verify tool was called
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0][0] == "bash"

    # Verify trace events contain PolicyDecisionEvent
    policy_events = [e for e in trace_events if e["event_type"] == "PolicyDecisionEvent"]
    assert len(policy_events) >= 1

    # Verify the policy decision event has expected structure
    policy_event = policy_events[0]
    assert policy_event["data"]["tool_name"] == "bash"
    assert policy_event["data"]["allowed"] is True

    # Verify no RejectionEvent since tool was allowed
    rejection_events = [e for e in trace_events if e["event_type"] == "RejectionEvent"]
    assert len(rejection_events) == 0


def test_sdk_adapter_produces_rejection_events_for_denied_tools():
    """Test SDK adapter produces RejectionEvent when tools are denied by policy."""
    adapter = SdkDawnKestrelAdapter()

    # Mock the DawnKestrelAgentRunner to return deterministic output with a denied tool
    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Attempted denied tool"},
        ],
        tool_calls=[
            {
                "name": "dangerous_tool",
                "arguments": {"action": "delete"},
            }
        ],
        agent_response="Attempted denied tool",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    # Create a tooling harness that denies the dangerous_tool
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    def mock_tool_call(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool_calls_log.append((tool_name, tool_input))
        return {"stdout": "test output", "exit_code": 0}

    tooling_harness = {
        "policy": {
            "allowed_tools": ["bash"],
            "denied_tools": ["dangerous_tool"],
        },
        "call": mock_tool_call,
    }

    scenario = {
        "id": "test-sdk-denied-scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": ["bash"], "denied_tools": ["dangerous_tool"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    # Patch the DawnKestrelAgentRunner to return our mock transcript
    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        MockRunner.return_value = mock_runner_instance

        final_output, trace_events, artifacts, _ = adapter.run_scenario(
            scenario, workdir, tooling_harness, budgets
        )

    # Verify final output
    assert final_output == "Attempted denied tool"

    # Verify artifacts are empty
    assert artifacts == {}

    # Verify tool was NOT called since it was denied
    assert len(tool_calls_log) == 0

    # Verify trace events contain PolicyDecisionEvent
    policy_events = [e for e in trace_events if e["event_type"] == "PolicyDecisionEvent"]
    assert len(policy_events) >= 1

    # Verify the policy decision event shows tool was denied
    policy_event = policy_events[0]
    assert policy_event["data"]["tool_name"] == "dangerous_tool"
    assert policy_event["data"]["allowed"] is False

    # Verify RejectionEvent was created
    rejection_events = [e for e in trace_events if e["event_type"] == "RejectionEvent"]
    assert len(rejection_events) == 1

    # Verify the rejection event has expected structure
    rejection_event = rejection_events[0]
    assert rejection_event["data"]["tool_name"] == "dangerous_tool"


def test_sdk_adapter_no_tool_calls_no_policy_events():
    """Test SDK adapter produces no PolicyDecisionEvents when no tools are called."""
    adapter = SdkDawnKestrelAdapter()

    # Mock the DawnKestrelAgentRunner to return transcript with no tool calls
    mock_transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Direct response"},
        ],
        tool_calls=[],
        agent_response="Direct response",
    )
    mock_outcome = EvalOutcome(status=EvalStatus.COMPLETED)

    tooling_harness = {
        "policy": {
            "allowed_tools": [],
            "denied_tools": [],
        },
        "call": lambda name, inp: {"result": "ok"},
    }

    scenario = {
        "id": "test-sdk-no-tools-scenario",
        "sut": {
            "type": "coding_agent",
            "adapter": "sdk_dawn_kestrel",
            "config": {
                "provider": "test_provider",
                "model": "test_model",
            },
        },
        "tools": {"allowed_tools": []},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    budgets = {}

    # Patch the DawnKestrelAgentRunner to return our mock transcript
    with patch("ash_hawk.scenario.adapters.sdk_dawn_kestrel.DawnKestrelAgentRunner") as MockRunner:
        mock_runner_instance = MagicMock()
        mock_runner_instance.run = AsyncMock(return_value=(mock_transcript, mock_outcome))
        MockRunner.return_value = mock_runner_instance

        final_output, trace_events, artifacts, _ = adapter.run_scenario(
            scenario, workdir, tooling_harness, budgets
        )

    # Verify final output
    assert final_output == "Direct response"

    # Verify no PolicyDecisionEvents were created
    policy_events = [e for e in trace_events if e["event_type"] == "PolicyDecisionEvent"]
    assert len(policy_events) == 0

    # Verify no RejectionEvents were created
    rejection_events = [e for e in trace_events if e["event_type"] == "RejectionEvent"]
    assert len(rejection_events) == 0
