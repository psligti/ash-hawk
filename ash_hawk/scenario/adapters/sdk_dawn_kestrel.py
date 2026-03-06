from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

from ash_hawk.agents import DawnKestrelAgentRunner
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    EVENT_TYPE_TOOL_CALL,
    EVENT_TYPE_TOOL_RESULT,
    PolicyDecisionEvent,
    RejectionEvent,
)
from ash_hawk.scenario.trace_normalizer import normalize_eval_transcript
from ash_hawk.types import EvalTask, ToolSurfacePolicy

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


def _resolve_provider_model(config: dict[str, Any]) -> tuple[str, str]:
    provider = config.get("provider") or config.get("provider_id")
    model = config.get("model")

    if isinstance(provider, str) and provider.strip() and isinstance(model, str) and model.strip():
        return provider.strip(), model.strip()

    try:
        from dawn_kestrel.core.settings import get_settings
    except ImportError as exc:
        raise ValueError("Provider/model not configured and dawn-kestrel is unavailable") from exc

    settings = get_settings()
    default_account = settings.get_default_account()

    if not isinstance(provider, str) or not provider.strip():
        if default_account is not None:
            provider = str(default_account.provider_id.value)
        else:
            provider = str(settings.get_default_provider().value)

    if not isinstance(model, str) or not model.strip():
        default_model = settings.get_default_model(provider)
        if isinstance(default_model, str) and default_model.strip():
            model = default_model

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
    ) -> tuple[str | dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
        del workdir

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

        provider, model = _resolve_provider_model(sut_config)
        runner_kwargs_raw = sut_config.get("runner_kwargs", {})
        runner_kwargs = dict(runner_kwargs_raw) if isinstance(runner_kwargs_raw, dict) else {}

        runner = DawnKestrelAgentRunner(provider=provider, model=model, **runner_kwargs)

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

        policy_payload: dict[str, Any] = {}
        tooling_call: Callable[[str, Any], dict[str, Any]] | None = None
        if isinstance(tooling_harness, dict):
            policy_raw = tooling_harness.get("policy")
            if isinstance(policy_raw, dict):
                policy_payload = policy_raw
            tooling_call_value = tooling_harness.get("call")
            if callable(tooling_call_value):
                tooling_call = tooling_call_value

        tool_policy = ToolSurfacePolicy.model_validate(policy_payload)
        policy_enforcer = PolicyEnforcer(tool_policy)

        transcript, outcome = _run_async(
            runner.run,
            task=task,
            policy_enforcer=policy_enforcer,
            config=run_config,
        )
        del outcome

        normalized_events = normalize_eval_transcript(transcript)
        for event in normalized_events:
            if event.event_type in {EVENT_TYPE_TOOL_CALL, EVENT_TYPE_TOOL_RESULT}:
                continue
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

        return final_output, trace_events, artifacts


__all__ = ["SdkDawnKestrelAdapter"]
