"""Tests for the mock adapter."""

from pathlib import Path

from ash_hawk.scenario.adapters.mock_adapter import MockAdapter
from ash_hawk.scenario.registry import get_default_adapter_registry
from ash_hawk.scenario.tooling import ToolingHarness


def test_mock_adapter_basic():
    """Test mock adapter emits correct trace events."""
    adapter = MockAdapter()

    scenario = {
        "id": "test-scenario",
        "sut": {"type": "coding_agent", "adapter": "mock_adapter"},
        "tools": {"allowed_tools": ["bash"]},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    tooling_harness = ToolingHarness(mode="mock", root=workdir)
    budgets = {}

    final_output, trace_events, artifacts, _ = adapter.run_scenario(
        scenario, workdir, tooling_harness, budgets
    )

    # Verify final output
    assert final_output == "OK"

    # Verify artifacts are empty
    assert artifacts == {}

    # Verify trace events contain expected events
    assert len(trace_events) >= 3  # At least user message, tool call/result, final message

    # Check for ModelMessageEvent for user
    user_msg_events = [e for e in trace_events if e["event_type"] == "ModelMessageEvent"]
    assert len(user_msg_events) >= 1
    assert user_msg_events[0]["data"]["role"] == "user"

    # Check for ToolCallEvent
    tool_call_events = [e for e in trace_events if e["event_type"] == "ToolCallEvent"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["data"]["tool"] == "bash"

    # Check for ToolResultEvent
    tool_result_events = [e for e in trace_events if e["event_type"] == "ToolResultEvent"]
    assert len(tool_result_events) == 1
    assert tool_result_events[0]["data"]["tool"] == "bash"
    assert "stdout" in tool_result_events[0]["data"]["result"]

    # Check for final ModelMessageEvent
    final_msg_events = [e for e in trace_events if e["event_type"] == "ModelMessageEvent"]
    assert len(final_msg_events) == 2
    assert final_msg_events[1]["data"]["role"] == "assistant"
    assert final_msg_events[1]["data"]["content"] == "OK"


def test_mock_adapter_with_expectations():
    """Test mock adapter emits VerificationEvent when expectations are set."""
    adapter = MockAdapter()

    scenario = {
        "id": "test-scenario",
        "sut": {"type": "coding_agent", "adapter": "mock_adapter"},
        "tools": {"allowed_tools": []},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {"must_events": ["some_event"]},
        "budgets": {},
    }

    workdir = Path("/tmp")
    tooling_harness = ToolingHarness(mode="mock", root=workdir)
    budgets = {}

    final_output, trace_events, artifacts, _ = adapter.run_scenario(
        scenario, workdir, tooling_harness, budgets
    )

    # Verify final output
    assert final_output == "OK"

    # Check for VerificationEvent
    verification_events = [e for e in trace_events if e["event_type"] == "VerificationEvent"]
    assert len(verification_events) == 1
    assert verification_events[0]["data"]["pass"] is True


def test_mock_adapter_without_bash():
    """Test mock adapter without bash tool doesn't emit tool events."""
    adapter = MockAdapter()

    scenario = {
        "id": "test-scenario",
        "sut": {"type": "coding_agent", "adapter": "mock_adapter"},
        "tools": {"allowed_tools": []},
        "inputs": {"prompt": "Test prompt"},
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    tooling_harness = ToolingHarness(mode="mock", root=workdir)
    budgets = {}

    final_output, trace_events, artifacts, _ = adapter.run_scenario(
        scenario, workdir, tooling_harness, budgets
    )

    # Verify final output
    assert final_output == "OK"

    # Check that no tool events are emitted
    tool_call_events = [e for e in trace_events if e["event_type"] == "ToolCallEvent"]
    tool_result_events = [e for e in trace_events if e["event_type"] == "ToolResultEvent"]

    assert len(tool_call_events) == 0
    assert len(tool_result_events) == 0


def test_mock_adapter_registered():
    """Test that mock_adapter is registered in the default registry."""
    registry = get_default_adapter_registry()

    # Check that mock_adapter is registered
    assert "mock_adapter" in registry
    adapter = registry.get("mock_adapter")
    assert adapter is not None
    assert adapter.name == "mock_adapter"


def test_mock_adapter_with_extra_mock_tool_calls():
    adapter = MockAdapter()

    scenario = {
        "id": "test-scenario-extra",
        "sut": {"type": "coding_agent", "adapter": "mock_adapter"},
        "tools": {"allowed_tools": []},
        "inputs": {
            "prompt": "Test prompt",
            "mock_tool_calls": [
                {
                    "tool": "note-lark_memory_search",
                    "input": {"query": "prefs"},
                }
            ],
        },
        "expectations": {},
        "budgets": {},
    }

    workdir = Path("/tmp")
    tooling_harness = ToolingHarness(mode="mock", root=workdir)
    tooling_harness.register_mock(
        "note-lark_memory_search",
        {"query": "prefs"},
        {"items": []},
    )

    final_output, trace_events, artifacts, _ = adapter.run_scenario(
        scenario, workdir, tooling_harness, {}
    )

    assert final_output == "OK"
    assert artifacts == {}

    tool_call_events = [e for e in trace_events if e["event_type"] == "ToolCallEvent"]
    tool_result_events = [e for e in trace_events if e["event_type"] == "ToolResultEvent"]
    assert len(tool_call_events) == 1
    assert len(tool_result_events) == 1
    assert tool_call_events[0]["data"]["tool"] == "note-lark_memory_search"
