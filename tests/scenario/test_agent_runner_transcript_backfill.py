from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.agent_runner import ScenarioAgentRunner
from ash_hawk.scenario.registry import ScenarioAdapterRegistry
from ash_hawk.types import EvalTask, ToolPermission, ToolSurfacePolicy


@pytest.mark.asyncio
async def test_agent_runner_backfills_messages_and_tool_calls_from_trace_events() -> None:
    task = _build_task("mock_adapter")

    policy = ToolSurfacePolicy(allowed_tools=["bash"])
    policy_enforcer = PolicyEnforcer(policy)

    runner = ScenarioAgentRunner()
    transcript, outcome = await runner.run(task, policy_enforcer, config={})

    assert outcome.failure_mode is None
    assert transcript.messages
    assert transcript.messages[0]["role"] == "user"
    assert transcript.messages[-1]["role"] == "assistant"
    assert transcript.tool_calls
    assert transcript.tool_calls[0]["name"] == "bash"


class FiveTupleAdapter:
    name = "five_tuple_adapter"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], Any, list[dict[str, Any]]]:
        del scenario, workdir, tooling_harness, budgets
        return (
            "ok",
            [],
            {},
            None,
            [{"role": "assistant", "content": "from-five"}],
        )


class SixTupleAdapter:
    name = "six_tuple_adapter"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[
        str, list[dict[str, Any]], dict[str, Any], Any, list[dict[str, Any]], list[dict[str, Any]]
    ]:
        del scenario, workdir, tooling_harness, budgets
        return (
            "ok",
            [],
            {},
            None,
            [{"role": "assistant", "content": "from-six"}],
            [{"name": "read", "arguments": {"path": "README.md"}}],
        )


class SixTupleAliasAdapter:
    name = "six_tuple_alias_adapter"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[
        str, list[dict[str, Any]], dict[str, Any], Any, list[dict[str, Any]], list[dict[str, Any]]
    ]:
        del scenario, workdir, tooling_harness, budgets
        return (
            "ok",
            [],
            {},
            None,
            [{"role": "assistant", "content": "from-six-alias"}],
            [{"tool": "read", "input": {"path": "README.md"}}],
        )


@pytest.mark.asyncio
async def test_agent_runner_consumes_five_tuple_messages() -> None:
    registry = ScenarioAdapterRegistry()
    registry.register(FiveTupleAdapter())
    runner = ScenarioAgentRunner(adapter_registry=registry)

    transcript, outcome = await runner.run(
        _build_task("five_tuple_adapter"),
        PolicyEnforcer(ToolSurfacePolicy()),
        config={},
    )

    assert outcome.failure_mode is None
    assert transcript.messages == [{"role": "assistant", "content": "from-five"}]


@pytest.mark.asyncio
async def test_agent_runner_consumes_six_tuple_messages_and_tool_calls() -> None:
    registry = ScenarioAdapterRegistry()
    registry.register(SixTupleAdapter())
    runner = ScenarioAgentRunner(adapter_registry=registry)

    transcript, outcome = await runner.run(
        _build_task("six_tuple_adapter"),
        PolicyEnforcer(ToolSurfacePolicy()),
        config={},
    )

    assert outcome.failure_mode is None
    assert transcript.messages == [{"role": "assistant", "content": "from-six"}]
    assert transcript.tool_calls == [{"name": "read", "arguments": {"path": "README.md"}}]


@pytest.mark.asyncio
async def test_agent_runner_normalizes_six_tuple_tool_call_aliases() -> None:
    registry = ScenarioAdapterRegistry()
    registry.register(SixTupleAliasAdapter())
    runner = ScenarioAgentRunner(adapter_registry=registry)

    transcript, outcome = await runner.run(
        _build_task("six_tuple_alias_adapter"),
        PolicyEnforcer(ToolSurfacePolicy()),
        config={},
    )

    assert outcome.failure_mode is None
    assert transcript.messages == [{"role": "assistant", "content": "from-six-alias"}]
    assert transcript.tool_calls == [{"name": "read", "arguments": {"path": "README.md"}}]


def test_agent_runner_policy_uses_timeout_override() -> None:
    runner = ScenarioAgentRunner(scenario_timeout_seconds=600.0)
    scenario = runner._parse_scenario(_build_task("mock_adapter"))
    policy = runner._build_policy(scenario, PolicyEnforcer(ToolSurfacePolicy(timeout_seconds=30.0)))
    assert policy.timeout_seconds == 600.0
    assert policy.default_permission == ToolPermission.ALLOW
    assert policy.allowed_tools == []


def _build_task(adapter_name: str) -> EvalTask:
    return EvalTask(
        id=f"task-{adapter_name}",
        description="Agent runner tuple compatibility",
        input={
            "scenario": {
                "schema_version": "v1",
                "id": f"scenario-{adapter_name}",
                "description": "Tuple compatibility scenario",
                "sut": {"type": "coding_agent", "adapter": adapter_name, "config": {}},
                "inputs": {"prompt": "List one command"},
                "tools": {
                    "allowed_tools": ["bash"],
                    "mocks": {},
                    "fault_injection": {},
                },
                "budgets": {
                    "max_steps": 3,
                    "max_tool_calls": 5,
                    "max_tokens": 100,
                    "max_time_seconds": 10.0,
                },
                "expectations": {
                    "must_events": [],
                    "must_not_events": [],
                    "ordering_rules": [],
                    "diff_assertions": [],
                    "output_assertions": [],
                },
                "graders": [],
            },
            "scenario_root": str(Path.cwd()),
        },
    )
