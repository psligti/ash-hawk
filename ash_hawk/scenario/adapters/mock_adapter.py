# type-hygiene: skip-file
"""Mock adapter for deterministic unit/smoke tests.

This adapter emits a deterministic trace sequence for testing purposes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ash_hawk.scenario.models import ScenarioAdapterResult, ScenarioTraceEvent
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ModelMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    VerificationEvent,
)


class MockAdapter:
    """Mock adapter that emits deterministic trace events for testing.

    This adapter is designed for unit and smoke tests. It emits a predictable
    sequence of trace events:
    1. ModelMessageEvent for user message
    2. ToolCallEvent + ToolResultEvent (if bash is in allowed_tools)
    3. VerificationEvent (if expectations are configured)
    4. ModelMessageEvent for final output

    Returns:
        - final_output: Always returns "OK"
        - trace_events: List of trace event dicts
        - artifacts: Always empty dict
    """

    name: str = "mock_adapter"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: Any,
        budgets: dict[str, Any],
    ) -> ScenarioAdapterResult:
        """Execute mock scenario and return deterministic results.

        Args:
            scenario: Scenario configuration dict (from ScenarioV1.model_dump())
            workdir: Working directory (unused in mock)
            tooling_harness: ToolingHarness instance for tool calls
            budgets: Budget configuration (unused in mock)

        Returns:
            Tuple of (final_output, trace_events, artifacts, outcome)
        """
        trace_events: list[dict[str, Any]] = []

        # 1. Emit user message event
        user_msg = ModelMessageEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={"role": "user", "content": scenario.get("inputs", {}).get("prompt", "Hello")},
        )
        trace_events.append(user_msg.model_dump())

        # 2. Check if bash tool is allowed and emit tool events
        tools_config = scenario.get("tools", {})
        allowed_tools = tools_config.get("allowed_tools", [])

        if "bash" in allowed_tools:
            # Emit ToolCallEvent
            tool_call = ToolCallEvent.create(
                ts=DEFAULT_TRACE_TS,
                data={"tool": "bash", "input": {"command": "echo 'mock'"}},
            )
            trace_events.append(tool_call.model_dump())

            # Use tooling harness if available, otherwise use mock result
            try:
                # Register mock if tooling harness has register_mock method
                if hasattr(tooling_harness, "register_mock"):
                    tooling_harness.register_mock(
                        "bash",
                        {"command": "echo 'mock'"},
                        {"stdout": "mock\n", "stderr": "", "exit_code": 0},
                    )

                # Call the tool through harness
                if hasattr(tooling_harness, "call"):
                    result = tooling_harness.call("bash", {"command": "echo 'mock'"})
                else:
                    result = {"stdout": "mock\n", "stderr": "", "exit_code": 0}
            except Exception:
                # Fallback to mock result if anything fails
                result = {"stdout": "mock\n", "stderr": "", "exit_code": 0}

            # Emit ToolResultEvent
            tool_result = ToolResultEvent.create(
                ts=DEFAULT_TRACE_TS,
                data={"tool": "bash", "result": result},
            )
            trace_events.append(tool_result.model_dump())

        extra_calls = scenario.get("inputs", {}).get("mock_tool_calls", [])
        if isinstance(extra_calls, list):
            for item in extra_calls:
                if not isinstance(item, dict):
                    continue
                tool_name = item.get("tool")
                tool_input = item.get("input", {})
                if not isinstance(tool_name, str) or not tool_name.strip():
                    continue
                if not isinstance(tool_input, dict):
                    tool_input = {}

                call_event = ToolCallEvent.create(
                    ts=DEFAULT_TRACE_TS,
                    data={"tool": tool_name, "input": tool_input},
                )
                trace_events.append(call_event.model_dump())

                try:
                    if hasattr(tooling_harness, "call"):
                        result = tooling_harness.call(tool_name, tool_input)
                    else:
                        result = {"ok": True}
                except Exception:
                    result = {"ok": False}

                result_event = ToolResultEvent.create(
                    ts=DEFAULT_TRACE_TS,
                    data={"tool": tool_name, "result": result},
                )
                trace_events.append(result_event.model_dump())

        # 3. Emit VerificationEvent if expectations are configured
        expectations = scenario.get("expectations", {})
        has_expectations = (
            expectations.get("must_events")
            or expectations.get("must_not_events")
            or expectations.get("ordering_rules")
            or expectations.get("diff_assertions")
            or expectations.get("output_assertions")
        )

        if has_expectations:
            verification = VerificationEvent.create(
                ts=DEFAULT_TRACE_TS,
                data={"pass": True, "message": "Mock verification passed"},
            )
            trace_events.append(verification.model_dump())

        # 4. Emit final output message
        final_content = scenario.get("inputs", {}).get(
            "expected_output",
            "OK",
        )
        if not isinstance(final_content, str):
            final_content = str(final_content)
        final_msg = ModelMessageEvent.create(
            ts=DEFAULT_TRACE_TS,
            data={"role": "assistant", "content": final_content},
        )
        trace_events.append(final_msg.model_dump())

        # Return final output, trace events, empty artifacts, and None outcome
        return ScenarioAdapterResult(
            final_output=final_content,
            trace_events=[ScenarioTraceEvent.model_validate(event) for event in trace_events],
            artifacts={},
        )
