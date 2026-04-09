# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

from ash_hawk.agents import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.models import (
    ScenarioAdapterResult,
    ScenarioTraceEvent,
    parse_scenario_tool_call,
)
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    PolicyDecisionEvent,
    RejectionEvent,
)
from ash_hawk.scenario.trace_normalizer import normalize_eval_transcript
from ash_hawk.types import EvalTask, ToolSurfacePolicy

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


def _normalize_inputs(inputs_raw: Any) -> dict[str, Any]:
    if isinstance(inputs_raw, dict):
        return dict(inputs_raw)
    if inputs_raw is None:
        return {}
    return {"prompt": str(inputs_raw)}


def _extract_prompt(inputs: dict[str, Any]) -> str:
    for key in ("prompt", "user_message", "message", "input"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _resolve_provider_model(
    config: dict[str, Any], agent_path: str | None = None
) -> tuple[str, str]:
    provider = config.get("provider") or config.get("provider_id")
    model = config.get("model")

    if isinstance(provider, str) and provider.strip() and isinstance(model, str) and model.strip():
        return provider.strip(), model.strip()

    if agent_path is not None:
        agent_dir = Path(agent_path)
        for candidate in [
            agent_dir / "agent_config.yaml",
            agent_dir.parent / ".dawn-kestrel" / "agent_config.yaml",
        ]:
            if candidate.exists():
                try:
                    import yaml

                    data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                    runtime = data.get("runtime", {})
                    if not isinstance(provider, str) or not provider.strip():
                        provider = runtime.get("provider")
                    if not isinstance(model, str) or not model.strip():
                        model = runtime.get("model")
                    if (
                        isinstance(provider, str)
                        and provider.strip()
                        and isinstance(model, str)
                        and model.strip()
                    ):
                        return provider.strip(), model.strip()
                except Exception:
                    pass  # nosec B110 — config parse is best-effort

    try:
        from dawn_kestrel.base.config import load_agent_config
    except ImportError as exc:
        raise ValueError("Provider/model not configured and dawn-kestrel is unavailable") from exc

    dk_config = load_agent_config()

    if not isinstance(provider, str) or not provider.strip():
        provider = dk_config.get("runtime.provider") or os.environ.get("DAWN_KESTREL_PROVIDER")

    if not isinstance(model, str) or not model.strip():
        model = dk_config.get("runtime.model") or os.environ.get("DAWN_KESTREL_MODEL")

    if (
        not isinstance(provider, str)
        or not provider.strip()
        or not isinstance(model, str)
        or not model.strip()
    ):
        raise ValueError("Could not resolve provider/model for dawn-kestrel runner")

    return provider.strip(), model.strip()


def _normalize_tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


class SdkDawnKestrelAdapter:
    name: str = "sdk_dawn_kestrel"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: Any,
        budgets: dict[str, Any],
    ) -> ScenarioAdapterResult:
        """Sync backward-compatible wrapper around async_run_scenario."""
        return _run_async(
            self.async_run_scenario,
            scenario,
            workdir,
            tooling_harness,
            budgets,
        )

    async def async_run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: Any,
        budgets: dict[str, Any],
    ) -> ScenarioAdapterResult:
        """Execute scenario by calling the agent runner directly on the current event loop.

        This avoids creating a new event loop per trial, which breaks the
        dawn-kestrel SDK's ProviderBus singleton (its background asyncio.Task
        workers become stale when their originating loop is closed).
        """
        trace_events: list[dict[str, Any]] = []
        artifacts: dict[str, Any] = {}

        inputs = _normalize_inputs(scenario.get("inputs"))
        prompt = _extract_prompt(inputs)
        if prompt and "prompt" not in inputs:
            inputs["prompt"] = prompt

        task = EvalTask(
            id=str(scenario.get("id", "scenario")),
            description=str(scenario.get("description", "")),
            input=inputs,
        )

        sut_raw = scenario.get("sut", {})
        sut = dict(sut_raw) if isinstance(sut_raw, dict) else {}
        sut_config_raw = sut.get("config", {})
        sut_config = dict(sut_config_raw) if isinstance(sut_config_raw, dict) else {}

        agent_path_value = None
        if isinstance(tooling_harness, dict):
            agent_path_value = tooling_harness.get("agent_path")

        provider, model = _resolve_provider_model(sut_config, agent_path=agent_path_value)
        runner_kwargs_raw = sut_config.get("runner_kwargs", {})
        runner_kwargs = dict(runner_kwargs_raw) if isinstance(runner_kwargs_raw, dict) else {}

        runner = DawnKestrelAgentRunner(provider=provider, model=model, **runner_kwargs)

        injector = None
        if isinstance(tooling_harness, dict):
            injector = tooling_harness.get("injector")

        skill_name = sut_config.get("skill_name")
        if injector is not None and skill_name is not None:
            injector.current_skill_name = skill_name

        if injector is not None and hasattr(runner, "set_lesson_injector"):
            runner.set_lesson_injector(injector)

        run_config_raw = sut_config.get("run_config", {})
        run_config = dict(run_config_raw) if isinstance(run_config_raw, dict) else {}

        if "temperature" in sut_config and "temperature" not in run_config:
            run_config["temperature"] = sut_config.get("temperature")
        if "max_tokens" in sut_config and "max_tokens" not in run_config:
            run_config["max_tokens"] = sut_config.get("max_tokens")
        if "max_tokens" not in run_config and isinstance(budgets, dict):
            budget_tokens = budgets.get("max_tokens")
            if isinstance(budget_tokens, int):
                run_config["max_tokens"] = budget_tokens

        policy_mode = sut_config.get("policy_mode")
        if isinstance(policy_mode, str) and policy_mode.strip() and "policy_mode" not in run_config:
            run_config["policy_mode"] = policy_mode.strip()

        run_config["workdir"] = str(workdir.resolve())

        if agent_path_value is not None:
            run_config["agent_path"] = agent_path_value

        policy_payload: dict[str, Any] = {}
        tooling_call: Callable[[str, Any], dict[str, Any]] | None = None
        if isinstance(tooling_harness, dict):
            policy_raw = tooling_harness.get("policy")
            if isinstance(policy_raw, dict):
                policy_payload = policy_raw
            tooling_call_value = tooling_harness.get("call")
            if callable(tooling_call_value):
                tooling_call = tooling_call_value

        workdir_str = str(workdir.resolve())
        if "allowed_roots" not in policy_payload:
            policy_payload["allowed_roots"] = []
        if workdir_str not in policy_payload["allowed_roots"]:
            policy_payload["allowed_roots"].append(workdir_str)

        tool_policy = ToolSurfacePolicy.model_validate(policy_payload)
        policy_enforcer = PolicyEnforcer(tool_policy)

        # Direct await — no thread, no new event loop.
        # The caller's event loop is the same one the dawn-kestrel
        # ProviderBus workers live on, so they stay alive across calls.
        transcript, outcome = await runner.run(
            task=task,
            policy_enforcer=policy_enforcer,
            config=run_config,
        )

        normalized_events = normalize_eval_transcript(transcript)
        has_policy_trace_events = False
        for event in normalized_events:
            if event.event_type in {"PolicyDecisionEvent", "RejectionEvent"}:
                has_policy_trace_events = True
            trace_events.append(event.model_dump())

        for tool_call in transcript.tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_name = tool_call.get("name") or tool_call.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            tool_input = tool_call.get("arguments")
            if tool_input is None:
                tool_input = tool_call.get("input")
            if tool_input is None:
                tool_input = tool_call.get("args")

            normalized_tool_input = _normalize_tool_input(tool_input)
            policy_result = policy_enforcer.check_tool(
                tool_name,
                normalized_tool_input,
            )
            if policy_result.allowed:
                count_result = policy_enforcer.increment_tool_count()
                if not count_result.allowed:
                    policy_result = count_result

            if not has_policy_trace_events:
                trace_events.append(
                    PolicyDecisionEvent.create(
                        ts=DEFAULT_TRACE_TS,
                        data={
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                            "allowed": policy_result.allowed,
                            "failure_mode": (
                                policy_result.failure_mode.value
                                if policy_result.failure_mode is not None
                                else None
                            ),
                            "reason": policy_result.reason,
                        },
                    ).model_dump()
                )

            if not policy_result.allowed:
                if not has_policy_trace_events:
                    trace_events.append(
                        RejectionEvent.create(
                            ts=DEFAULT_TRACE_TS,
                            data={
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                                "failure_mode": (
                                    policy_result.failure_mode.value
                                    if policy_result.failure_mode is not None
                                    else None
                                ),
                                "reason": policy_result.reason,
                            },
                        ).model_dump()
                    )
                continue

            if tooling_call is None:
                raise ValueError("Tooling harness is missing callable 'call' for tool execution")

            tooling_call(tool_name, normalized_tool_input)

        final_output = transcript.agent_response
        if final_output is None:
            final_output = transcript.error_trace

        return ScenarioAdapterResult(
            final_output=final_output,
            trace_events=[ScenarioTraceEvent.model_validate(event) for event in trace_events],
            artifacts=artifacts,
            outcome=outcome,
        )


__all__ = ["SdkDawnKestrelAdapter"]
