from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

import pydantic as pd

from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.models import ScenarioV1
from ash_hawk.scenario.registry import ScenarioAdapterRegistry, get_default_adapter_registry
from ash_hawk.scenario.tooling import ToolingHarness
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ArtifactEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ash_hawk.types import EvalOutcome, EvalTask, EvalTranscript, FailureMode, ToolSurfacePolicy


class ToolingHarnessRecorder:
    def __init__(self, harness: ToolingHarness) -> None:
        self._harness = harness
        self.events: list[dict[str, Any]] = []

    @property
    def harness(self) -> ToolingHarness:
        return self._harness

    def call(self, tool_name: str, tool_input: Any) -> dict[str, Any]:
        call_event = ToolCallEvent.create(
            DEFAULT_TRACE_TS,
            {"name": tool_name, "arguments": tool_input},
        )
        self.events.append(call_event.model_dump())
        result = self._harness.call(tool_name, tool_input)
        result_event = ToolResultEvent.create(
            DEFAULT_TRACE_TS,
            {"tool_name": tool_name, "result": result},
        )
        self.events.append(result_event.model_dump())
        return result


class ScenarioAgentRunner:
    def __init__(
        self,
        adapter_registry: ScenarioAdapterRegistry | None = None,
        tooling_mode: Literal["mock", "record", "replay"] = "record",
        artifacts_root: Path | None = None,
        injector: Any | None = None,
    ) -> None:
        self._adapter_registry = adapter_registry or get_default_adapter_registry()
        self._tooling_mode = tooling_mode
        self._artifacts_root = artifacts_root
        self._injector = injector

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        start_time = time.time()

        try:
            scenario = self._parse_scenario(task)
        except pd.ValidationError as exc:
            return self._failure_transcript(
                FailureMode.VALIDATION_ERROR,
                f"Scenario validation failed: {exc}",
                start_time,
            )
        except ValueError as exc:
            return self._failure_transcript(
                FailureMode.VALIDATION_ERROR,
                str(exc),
                start_time,
            )

        adapter = self._adapter_registry.get(scenario.sut.adapter)
        if adapter is None:
            return self._failure_transcript(
                FailureMode.AGENT_ERROR,
                f"Scenario adapter not found: {scenario.sut.adapter}",
                start_time,
            )

        tool_policy = self._build_policy(scenario, policy_enforcer)
        tooling_root = self._tooling_root(task)
        tooling_harness = ToolingHarness(mode=self._tooling_mode, root=tooling_root)
        self._register_tool_mocks(tooling_harness, scenario.tools.mocks)
        self._apply_fault_injection(tooling_harness, scenario.tools.fault_injection)

        tooling_recorder = ToolingHarnessRecorder(tooling_harness)
        tooling_context = {
            "call": tooling_recorder.call,
            "harness": tooling_harness,
            "mode": tooling_harness.mode,
            "policy": tool_policy.model_dump(),
            "injector": self._injector,
        }

        adapter_messages: list[dict[str, Any]] = []
        adapter_tool_calls: list[dict[str, Any]] = []
        final_output = None
        trace_events: list[dict[str, Any]] = []
        artifacts: dict[str, Any] = {}
        adapter_outcome = None
        try:
            adapter_result = await asyncio.to_thread(
                adapter.run_scenario,
                scenario.model_dump(),
                tooling_root,
                tooling_context,
                scenario.budgets.model_dump(),
            )
            # Adapter returns 4-6 values: (output, events, artifacts, outcome, messages?, tool_calls?)
            result_len = len(adapter_result)
            if result_len >= 4:
                final_output = adapter_result[0]
                trace_events = adapter_result[1]
                artifacts = adapter_result[2]
                adapter_outcome = adapter_result[3]
            else:
                # Backward compat for adapters returning 3 values
                final_output = adapter_result[0]
                trace_events = adapter_result[1]
                artifacts = adapter_result[2]
                adapter_outcome = None
            if result_len >= 6:
                adapter_messages = adapter_result[4]
                adapter_tool_calls = adapter_result[5]
        except Exception as exc:
            return self._failure_transcript(
                FailureMode.AGENT_ERROR,
                f"Scenario execution failed: {exc}",
                start_time,
                trace_events=tooling_recorder.events,
            )

        # If adapter returned a failure outcome, propagate it
        if adapter_outcome is not None and adapter_outcome.failure_mode is not None:
            duration = time.time() - start_time
            error_msg = adapter_outcome.error_message or "Agent returned failure"
            transcript = EvalTranscript(
                messages=adapter_messages,
                tool_calls=adapter_tool_calls,
                error_trace=error_msg,
                duration_seconds=duration,
                trace_events=self._normalize_trace_events(trace_events),
            )
            return transcript, adapter_outcome

        artifact_events = self._persist_artifacts(artifacts, config)
        combined_trace_events = (
            self._normalize_trace_events(trace_events) + tooling_recorder.events + artifact_events
        )

        duration = time.time() - start_time
        transcript = EvalTranscript(
            messages=adapter_messages,
            tool_calls=adapter_tool_calls,
            agent_response=final_output,
            duration_seconds=duration,
            trace_events=combined_trace_events,
        )

        # Use adapter outcome if available, otherwise success
        outcome = adapter_outcome if adapter_outcome is not None else EvalOutcome.success()
        return transcript, outcome

    def _parse_scenario(self, task: EvalTask) -> ScenarioV1:
        if not isinstance(task.input, dict):
            raise ValueError("Scenario task input must be a dict")
        scenario_raw = task.input.get("scenario")
        if not isinstance(scenario_raw, dict):
            raise ValueError("Scenario task input must include 'scenario' mapping")
        return ScenarioV1.model_validate(scenario_raw)

    def _tooling_root(self, task: EvalTask) -> Path:
        if isinstance(task.input, dict):
            root_raw = task.input.get("scenario_root")
            if isinstance(root_raw, str) and root_raw.strip() != "":
                return Path(root_raw)
        return Path.cwd()

    def _build_policy(
        self,
        scenario: ScenarioV1,
        policy_enforcer: PolicyEnforcer,
    ) -> ToolSurfacePolicy:
        timeout_seconds = (
            scenario.budgets.max_time_seconds
            if scenario.budgets.max_time_seconds is not None
            else policy_enforcer.policy.timeout_seconds
        )
        return ToolSurfacePolicy(
            allowed_tools=list(scenario.tools.allowed_tools),
            max_tool_calls=scenario.budgets.max_tool_calls,
            token_budget=scenario.budgets.max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _register_tool_mocks(
        self,
        harness: ToolingHarness,
        mocks: dict[str, Any],
    ) -> None:
        for tool_name, entries in mocks.items():
            if isinstance(entries, list):
                for entry in entries:
                    self._register_tool_mock_entry(harness, tool_name, entry)
            else:
                self._register_tool_mock_entry(harness, tool_name, entries)

    def _register_tool_mock_entry(
        self,
        harness: ToolingHarness,
        tool_name: str,
        entry: Any,
    ) -> None:
        input_value: Any = {}
        result_value: Any = {}

        if isinstance(entry, dict):
            input_value = entry.get("input", {})
            if "result" in entry:
                result_value = entry.get("result")
            else:
                result_value = entry
        else:
            result_value = entry

        if not isinstance(result_value, dict):
            result_value = {"value": result_value}

        harness.register_mock(tool_name, input_value, result_value)

    def _apply_fault_injection(self, harness: ToolingHarness, config: dict[str, Any]) -> None:
        timeouts = config.get("timeouts") or config.get("timeout_tools")
        if isinstance(timeouts, dict):
            for tool_name, count in timeouts.items():
                if isinstance(count, int) and count > 0:
                    for _ in range(count):
                        harness.inject_timeout(str(tool_name))
        elif isinstance(timeouts, list):
            for tool_name in timeouts:
                harness.inject_timeout(str(tool_name))

        malformed = config.get("malformed")
        if isinstance(malformed, list):
            for tool_name in malformed:
                harness.inject_malformed(str(tool_name))

    def _persist_artifacts(
        self,
        artifacts: Any,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if artifacts is None:
            return []

        suite_id = str(config.get("suite_id", "unknown-suite"))
        run_id = str(config.get("run_id", "unknown-run"))
        trial_id = str(config.get("trial_id", "unknown-trial"))

        artifacts_root = self._artifacts_root or Path(".ash-hawk")
        artifact_dir = artifacts_root / suite_id / "runs" / run_id / "artifacts" / trial_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = artifact_dir / "artifacts.json"
        serialized = json.dumps(artifacts, indent=2, default=repr)
        artifact_path.write_text(serialized, encoding="utf-8")

        event = ArtifactEvent.create(
            DEFAULT_TRACE_TS,
            {"path": str(artifact_path), "artifact_key": "artifacts"},
        )
        return [event.model_dump()]

    def _normalize_trace_events(self, trace_events: Any) -> list[dict[str, Any]]:
        if not isinstance(trace_events, list):
            return []

        normalized: list[dict[str, Any]] = []
        for event in trace_events:
            if isinstance(event, dict):
                normalized.append(event)
            elif hasattr(event, "model_dump"):
                normalized.append(event.model_dump())
            else:
                normalized.append({"event": str(event)})
        return normalized

    def _failure_transcript(
        self,
        failure_mode: FailureMode,
        message: str,
        start_time: float,
        trace_events: list[dict[str, Any]] | None = None,
    ) -> tuple[EvalTranscript, EvalOutcome]:
        duration = time.time() - start_time
        transcript = EvalTranscript(
            error_trace=message,
            duration_seconds=duration,
            trace_events=trace_events or [],
        )
        outcome = EvalOutcome.failure(failure_mode, message)
        return transcript, outcome


__all__ = ["ScenarioAgentRunner"]
