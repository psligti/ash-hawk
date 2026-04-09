# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import os
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterator, TypeVar, cast

from ash_hawk.agents.source_workspace import (
    detect_agent_config_path,
    detect_package_name,
    import_package_from_agent_path,
)
from ash_hawk.scenario.models import (
    JSONValue,
    ScenarioAdapterResult,
    ScenarioTraceEvent,
)
from ash_hawk.types import EvalOutcome, FailureMode

_T = TypeVar("_T")


def _run_coroutine_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:  # nosec B110
            pass
        asyncio.set_event_loop(None)
        loop.close()


def _run_async(func: Callable[..., Coroutine[Any, Any, _T]], *args: Any, **kwargs: Any) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coroutine_sync(func(*args, **kwargs))

    result_container: dict[str, _T] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_container["result"] = _run_coroutine_sync(func(*args, **kwargs))
        except BaseException as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_container:
        raise error_container["error"]

    return result_container["result"]


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _extract_prompt(scenario: dict[str, JSONValue]) -> str:
    inputs_raw = scenario.get("inputs")
    if not isinstance(inputs_raw, dict):
        return ""
    for key in ("prompt", "user_message", "message", "input"):
        value = inputs_raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


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

            return Path(agent_pkg.__file__).parent
        except ImportError:
            return None

    def run_scenario(
        self,
        scenario: dict[str, JSONValue],
        workdir: Path,
        tooling_harness: dict[str, object],
        budgets: dict[str, JSONValue],
    ) -> ScenarioAdapterResult:
        return _run_async(
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
        if isinstance(tooling_harness, dict):
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

        prompt = _extract_prompt(scenario)
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

        def on_event(event: Any) -> None:
            captured_events.append(event)

        with _cwd(workdir.resolve()):
            with import_package_from_agent_path(package_name, agent_path_value):
                result: Any = await execute(
                    prompt=prompt,
                    trace=False,
                    on_event=on_event,
                    config_path=config_path_value,
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


__all__ = ["BoltMerlinScenarioAdapter"]
