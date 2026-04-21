# type-hygiene: skip-file
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ash_hawk.agents.source_workspace import (
    detect_agent_config_path,
    detect_package_name,
    import_package_from_agent_path,
)
from ash_hawk.scenario.adapter_utils import extract_prompt, run_async
from ash_hawk.scenario.models import (
    JSONValue,
    ScenarioAdapterResult,
    ScenarioTraceEvent,
)
from ash_hawk.scenario.tool_event_preview import tool_event_preview
from ash_hawk.types import EvalOutcome, FailureMode


class BoltMerlinScenarioAdapter:
    """Scenario adapter that calls the real bolt-merlin coding agent.

    Uses ``bolt_merlin.agent.execute.execute()`` which loads the coding agent
    system prompt, tools (read, edit, write, bash, glob, grep, todoread,
    todowrite, test), and runs the full agent loop — rather than delegating
    to a generic ``DawnKestrelAgentRunner``.
    """

    name: str = "bolt_merlin"

    @classmethod
    def agent_source_path(cls) -> Path | None:
        try:
            import bolt_merlin.agent as agent_pkg

            agent_file = agent_pkg.__file__
            if agent_file is None:
                return None
            return Path(agent_file).parent
        except ImportError:
            return None

    def run_scenario(
        self,
        scenario: dict[str, JSONValue],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, JSONValue],
    ) -> ScenarioAdapterResult:
        return run_async(
            self.async_run_scenario,
            scenario,
            workdir,
            tooling_harness,
            budgets,
        )

    async def async_run_scenario(
        self,
        scenario: dict[str, JSONValue],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, JSONValue],
    ) -> ScenarioAdapterResult:
        package_name = "bolt_merlin"
        agent_path_value: Path | None = None
        config_path_value: Path | None = None
        raw_agent_path = tooling_harness.get("agent_path")
        if isinstance(raw_agent_path, str) and raw_agent_path.strip():
            agent_path_value = Path(raw_agent_path)
            detected_package = detect_package_name(agent_path_value)
            if detected_package is not None:
                package_name = detected_package
            config_path_value = detect_agent_config_path(agent_path_value)

        try:
            with import_package_from_agent_path(package_name, agent_path_value):
                from bolt_merlin.agent.execute import execute
        except ImportError as exc:
            return ScenarioAdapterResult(
                final_output=None,
                trace_events=[],
                artifacts={},
                outcome=EvalOutcome.failure(
                    FailureMode.AGENT_ERROR,
                    error_message=f"bolt-merlin agent unavailable: {exc}",
                ),
            )

        inputs_raw = scenario.get("inputs")
        prompt = extract_prompt(inputs_raw) if isinstance(inputs_raw, dict) else ""
        if not prompt:
            return ScenarioAdapterResult(
                final_output=None,
                trace_events=[],
                artifacts={},
                outcome=EvalOutcome.failure(
                    FailureMode.VALIDATION_ERROR,
                    error_message="No prompt found in scenario inputs",
                ),
            )

        captured_events: list[Any] = []
        event_callback = tooling_harness.get("event_callback")

        def on_event(event: Any) -> None:
            captured_events.append(event)
            if callable(event_callback):
                payload = _build_live_event_payload(event)
                if payload is not None:
                    event_callback(payload)

        with import_package_from_agent_path(package_name, agent_path_value):
            result: Any = await execute(
                prompt=prompt,
                trace=False,
                on_event=on_event,
                config_path=config_path_value,
                working_dir=workdir.resolve(),
            )

        if hasattr(result, "error_type"):
            return ScenarioAdapterResult(
                final_output=None,
                trace_events=_build_trace_events(captured_events, prompt),
                artifacts={},
                outcome=EvalOutcome.failure(
                    FailureMode.AGENT_ERROR,
                    error_message=f"[{result.error_type}] {result.message}",
                ),
            )

        return ScenarioAdapterResult(
            final_output=result.response,
            trace_events=_build_trace_events(captured_events, prompt, result.response),
            artifacts={
                "session_id": result.session_id,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "duration_ms": result.duration_ms,
            },
            outcome=EvalOutcome.success(),
        )


def _build_trace_events(
    events: list[Any],
    prompt: str | None = None,
    final_response: str | None = None,
) -> list[ScenarioTraceEvent]:
    trace_events: list[ScenarioTraceEvent] = []

    if prompt:
        trace_events.append(
            ScenarioTraceEvent(
                event_type="ModelMessageEvent",
                ts=datetime.now(UTC).isoformat(),
                data={"role": "user", "content": prompt},
            )
        )

    for event in events:
        event_dict: dict[str, Any] = event.to_dict()
        event_type = str(event_dict.pop("event_type", "unknown"))
        timestamp_raw = event_dict.pop("timestamp", 0.0)
        if isinstance(timestamp_raw, int | float) and timestamp_raw > 0:
            ts = datetime.fromtimestamp(timestamp_raw, tz=UTC).isoformat()
        else:
            ts = datetime.now(UTC).isoformat()

        if event_type == "llm_call":
            text = event_dict.get("text", "")
            if text:
                trace_events.append(
                    ScenarioTraceEvent(
                        event_type="ModelMessageEvent",
                        ts=ts,
                        data={"role": "assistant", "content": text},
                    )
                )
        elif event_type == "tool_call":
            tool_name = event_dict.get("tool_name", "")
            tool_input = event_dict.get("tool_input", {})
            if tool_name:
                trace_events.append(
                    ScenarioTraceEvent(
                        event_type="ToolCallEvent",
                        ts=ts,
                        data={"name": tool_name, "arguments": tool_input},
                    )
                )
        else:
            trace_events.append(ScenarioTraceEvent(event_type=event_type, ts=ts, data=event_dict))

    if final_response:
        has_final = any(
            e.event_type == "ModelMessageEvent"
            and isinstance(e.data.get("content"), str)
            and cast(str, e.data["content"]) == final_response
            for e in trace_events
        )
        if not has_final:
            trace_events.append(
                ScenarioTraceEvent(
                    event_type="ModelMessageEvent",
                    ts=datetime.now(UTC).isoformat(),
                    data={"role": "assistant", "content": final_response},
                )
            )

    return trace_events


def _build_live_event_payload(event: Any) -> dict[str, object] | None:
    if type(event).__name__ != "ToolExecutionEvent":
        return None
    tool_name = str(getattr(event, "tool_name", "") or "")
    if not tool_name:
        return None
    error = getattr(event, "error", None)
    return {
        "tool": tool_name,
        "event_type": "tool_result",
        "success": error is None,
        "preview": tool_event_preview(
            tool_name, getattr(event, "tool_input", None), _live_event_result(event)
        ),
        "error": str(error) if error is not None else None,
    }


def _live_event_result(event: Any) -> dict[str, object]:
    payload: dict[str, object] = {}
    for attr in ("result", "output", "response", "stdout", "message"):
        value = getattr(event, attr, None)
        if value is not None:
            payload[attr] = value
    return payload


__all__ = ["BoltMerlinScenarioAdapter"]
