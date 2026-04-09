from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

import pydantic as pd

logger = logging.getLogger(__name__)

from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.models import (
    JSONValue,
    ScenarioAdapterResult,
    ScenarioTraceEvent,
    ScenarioV1,
    parse_scenario_tool_call,
)
from ash_hawk.scenario.registry import ScenarioAdapterRegistry, get_default_adapter_registry
from ash_hawk.scenario.tooling import ToolingHarness
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    EVENT_TYPE_MODEL_MESSAGE,
    EVENT_TYPE_TOOL_CALL,
    ArtifactEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalTask,
    EvalTranscript,
    FailureMode,
    ToolPermission,
    ToolSurfacePolicy,
)


class ToolingHarnessRecorder:
    def __init__(self, harness: ToolingHarness) -> None:
        self._harness = harness
        self.events: list[dict[str, JSONValue]] = []

    @property
    def harness(self) -> ToolingHarness:
        return self._harness

    def call(self, tool_name: str, tool_input: object) -> dict[str, JSONValue]:
        call_event = ToolCallEvent.create(
            DEFAULT_TRACE_TS,
            {"name": tool_name, "arguments": tool_input},
        )
        self.events.append(call_event.model_dump(mode="json"))
        result = self._harness.call(tool_name, tool_input)
        result_event = ToolResultEvent.create(
            DEFAULT_TRACE_TS,
            {"tool_name": tool_name, "result": result},
        )
        self.events.append(result_event.model_dump(mode="json"))
        if isinstance(result, dict):
            return {str(key): self._to_json_value(value) for key, value in result.items()}
        return {"value": self._to_json_value(result)}

    def _to_json_value(self, value: object) -> JSONValue:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, dict):
            return {str(key): self._to_json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_json_value(item) for item in value]
        return str(value)


class ScenarioAgentRunner:
    def __init__(
        self,
        adapter_registry: ScenarioAdapterRegistry | None = None,
        tooling_mode: Literal["mock", "record", "replay"] = "record",
        artifacts_root: Path | None = None,
        injector: object | None = None,
        scenario_timeout_seconds: float | None = None,
        agent_path: Path | None = None,
        adapter_override: str | None = None,
    ) -> None:
        self._adapter_registry = adapter_registry or get_default_adapter_registry()
        self._tooling_mode = tooling_mode
        self._artifacts_root = artifacts_root
        self._injector = injector
        self._scenario_timeout_seconds = scenario_timeout_seconds
        self._agent_path = agent_path
        self._adapter_override = adapter_override

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, object],
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

        adapter_name = self._adapter_override or scenario.sut.adapter
        adapter = self._adapter_registry.get(adapter_name)
        if adapter is None:
            return self._failure_transcript(
                FailureMode.AGENT_ERROR,
                f"Scenario adapter not found: {adapter_name}",
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
        if self._agent_path is not None:
            tooling_context["agent_path"] = str(self._agent_path)

        try:
            async_run = getattr(adapter, "async_run_scenario", None)
            if callable(async_run):
                raw_adapter_result = await async_run(
                    scenario.model_dump(),
                    tooling_root,
                    tooling_context,
                    scenario.budgets.model_dump(),
                )
            else:
                raw_adapter_result = await asyncio.to_thread(
                    adapter.run_scenario,
                    scenario.model_dump(),
                    tooling_root,
                    tooling_context,
                    scenario.budgets.model_dump(),
                )
            adapter_result = self._normalize_adapter_result(raw_adapter_result)
        except Exception as exc:
            return self._failure_transcript(
                FailureMode.AGENT_ERROR,
                f"Scenario execution failed: {exc}",
                start_time,
                trace_events=tooling_recorder.events,
            )

        adapter_messages = [
            message.model_dump(mode="json") for message in adapter_result.extract_messages()
        ]
        adapter_tool_calls = [
            tool_call.model_dump(mode="json") for tool_call in adapter_result.extract_tool_calls()
        ]
        normalized_trace_events = [
            event.model_dump(mode="json") for event in adapter_result.trace_events
        ]
        combined_raw_events = normalized_trace_events + tooling_recorder.events

        if not adapter_messages or not adapter_tool_calls:
            inferred_messages, inferred_tool_calls = self._infer_transcript_content(
                combined_raw_events
            )
            if not adapter_messages:
                logger.warning(
                    "ScenarioAgentRunner: adapter returned empty messages, "
                    "inferring %d messages from %d trace events",
                    len(inferred_messages),
                    len(combined_raw_events),
                )
                adapter_messages = inferred_messages
            if not adapter_tool_calls:
                logger.warning(
                    "ScenarioAgentRunner: adapter returned empty tool_calls, "
                    "inferring %d tool calls from %d trace events",
                    len(inferred_tool_calls),
                    len(combined_raw_events),
                )
                adapter_tool_calls = inferred_tool_calls

        # If adapter returned a failure outcome, propagate it
        if adapter_result.outcome.failure_mode is not None:
            duration = time.time() - start_time
            error_msg = adapter_result.outcome.error_message or "Agent returned failure"
            transcript = EvalTranscript(
                messages=adapter_messages,
                tool_calls=adapter_tool_calls,
                error_trace=error_msg,
                duration_seconds=duration,
                trace_events=normalized_trace_events + tooling_recorder.events,
            )
            return transcript, adapter_result.outcome

        artifact_events = self._persist_artifacts(adapter_result.artifacts, config)
        combined_trace_events = normalized_trace_events + tooling_recorder.events + artifact_events

        duration = time.time() - start_time
        transcript = EvalTranscript(
            messages=adapter_messages,
            tool_calls=adapter_tool_calls,
            agent_response=self._normalize_agent_response(adapter_result.final_output),
            duration_seconds=duration,
            trace_events=combined_trace_events,
        )

        outcome = adapter_result.outcome
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
        timeout_seconds = self._scenario_timeout_seconds
        if timeout_seconds is None:
            timeout_seconds = (
                scenario.budgets.max_time_seconds
                if scenario.budgets.max_time_seconds is not None
                else policy_enforcer.policy.timeout_seconds
            )
        return ToolSurfacePolicy(
            default_permission=ToolPermission.ALLOW,
            max_tool_calls=scenario.budgets.max_tool_calls,
            token_budget=scenario.budgets.max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _register_tool_mocks(
        self,
        harness: ToolingHarness,
        mocks: dict[str, JSONValue],
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
        entry: JSONValue,
    ) -> None:
        input_value: dict[str, JSONValue] = {}
        result_value: JSONValue = {}

        if isinstance(entry, dict):
            maybe_input = entry.get("input", {})
            if isinstance(maybe_input, dict):
                input_value = {str(key): value for key, value in maybe_input.items()}
            if "result" in entry:
                result_value = entry.get("result")
            else:
                result_value = entry
        else:
            result_value = entry

        if not isinstance(result_value, dict):
            result_value = {"value": result_value}

        harness.register_mock(tool_name, input_value, result_value)

    def _apply_fault_injection(self, harness: ToolingHarness, config: dict[str, JSONValue]) -> None:
        timeouts = config.get("timeouts") or config.get("timeout_tools")
        if isinstance(timeouts, dict):
            for tool_name, count in timeouts.items():
                if isinstance(count, int) and count > 0:
                    for _ in range(count):
                        harness.inject_timeout(str(tool_name))
        elif isinstance(timeouts, list):
            for timeout_tool in timeouts:
                harness.inject_timeout(str(timeout_tool))

        malformed = config.get("malformed")
        if isinstance(malformed, list):
            for malformed_tool in malformed:
                harness.inject_malformed(str(malformed_tool))

    def _persist_artifacts(
        self,
        artifacts: dict[str, JSONValue],
        config: dict[str, object],
    ) -> list[dict[str, JSONValue]]:
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

    def _normalize_trace_events(self, trace_events: list[object]) -> list[ScenarioTraceEvent]:
        normalized: list[ScenarioTraceEvent] = []
        for event in trace_events:
            if isinstance(event, ScenarioTraceEvent):
                normalized.append(event)
                continue
            if isinstance(event, dict):
                try:
                    normalized.append(ScenarioTraceEvent.model_validate(event))
                except pd.ValidationError:
                    continue
                continue
            if hasattr(event, "model_dump"):
                dumped = event.model_dump(mode="json")
                if isinstance(dumped, dict):
                    try:
                        normalized.append(ScenarioTraceEvent.model_validate(dumped))
                    except pd.ValidationError:
                        continue
        return normalized

    def _infer_transcript_content(
        self,
        trace_events: list[dict[str, JSONValue]],
    ) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
        messages: list[dict[str, JSONValue]] = []
        tool_calls: list[dict[str, JSONValue]] = []

        for event in trace_events:
            event_type = event.get("event_type")
            data = event.get("data")
            if not isinstance(data, dict):
                continue

            if event_type == EVENT_TYPE_MODEL_MESSAGE:
                role = data.get("role")
                content = data.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    messages.append({"role": role, "content": content})

            if event_type == EVENT_TYPE_TOOL_CALL:
                tool_name = data.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    tool_name = data.get("tool")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    continue

                arguments = data.get("arguments")
                if arguments is None:
                    arguments = data.get("input")
                if arguments is None:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}

                tool_calls.append({"name": tool_name, "arguments": arguments})

        return messages, tool_calls

    def _failure_transcript(
        self,
        failure_mode: FailureMode,
        message: str,
        start_time: float,
        trace_events: list[dict[str, JSONValue]] | None = None,
    ) -> tuple[EvalTranscript, EvalOutcome]:
        duration = time.time() - start_time
        transcript = EvalTranscript(
            error_trace=message,
            duration_seconds=duration,
            trace_events=trace_events or [],
        )
        outcome = EvalOutcome.failure(failure_mode, message)
        return transcript, outcome

    def _normalize_adapter_result(self, adapter_result: object) -> ScenarioAdapterResult:
        if isinstance(adapter_result, ScenarioAdapterResult):
            return adapter_result
        if not isinstance(adapter_result, tuple):
            raise ValueError("Adapter must return ScenarioAdapterResult or legacy tuple")

        result_len = len(adapter_result)
        if result_len < 3:
            raise ValueError(
                "Adapter legacy tuple must include at least output, trace_events, artifacts"
            )

        final_output = self._normalize_agent_response(adapter_result[0])
        trace_events_raw = adapter_result[1]
        trace_events = self._normalize_trace_events(
            trace_events_raw if isinstance(trace_events_raw, list) else []
        )

        artifacts_raw = adapter_result[2]
        artifacts: dict[str, JSONValue] = {}
        if isinstance(artifacts_raw, dict):
            artifacts = {
                str(key): self._to_json_value(value) for key, value in artifacts_raw.items()
            }

        outcome = EvalOutcome.success()
        if result_len >= 4:
            outcome_raw = adapter_result[3]
            if isinstance(outcome_raw, EvalOutcome):
                outcome = outcome_raw
            elif isinstance(outcome_raw, dict):
                outcome = EvalOutcome.model_validate(outcome_raw)

        if result_len >= 5 and isinstance(adapter_result[4], list):
            for message_raw in adapter_result[4]:
                if isinstance(message_raw, dict):
                    role = message_raw.get("role")
                    content = message_raw.get("content")
                    if isinstance(role, str) and isinstance(content, str):
                        trace_events.append(
                            ScenarioTraceEvent(
                                event_type="ModelMessageEvent",
                                ts=DEFAULT_TRACE_TS,
                                data={"role": role, "content": content},
                            )
                        )

        if result_len >= 6 and isinstance(adapter_result[5], list):
            for tool_call_raw in adapter_result[5]:
                parsed_tool_call = parse_scenario_tool_call(tool_call_raw)
                if parsed_tool_call is not None:
                    trace_events.append(
                        ScenarioTraceEvent(
                            event_type="ToolCallEvent",
                            ts=DEFAULT_TRACE_TS,
                            data={
                                "name": parsed_tool_call.name,
                                "arguments": parsed_tool_call.arguments,
                            },
                        )
                    )

        return ScenarioAdapterResult(
            final_output=final_output,
            trace_events=trace_events,
            artifacts=artifacts,
            outcome=outcome,
        )

    def _to_json_value(self, value: object) -> JSONValue:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, dict):
            return {str(key): self._to_json_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_json_value(item) for item in value]
        return str(value)

    def _normalize_agent_response(self, value: object) -> str | dict[str, object] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        return str(value)


__all__ = ["ScenarioAgentRunner"]
