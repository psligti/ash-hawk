from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.models import ContextSnapshot, ThinRuntimeExecutionResult, ToolResult
from ash_hawk.thin_runtime.tool_types import DelegationRequest, ToolExecutionPayload


def test_execute_delegation_uses_delegated_agent_iteration_budget() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    delegated_agent = harness.agents.get("executor")
    context = ContextSnapshot(
        goal={"max_iterations": 8},
        runtime={"active_agent": "improver", "lead_agent": "improver", "max_iterations": 8},
    )
    delegation_result = ToolResult(
        tool_name="delegate_task",
        success=True,
        payload=ToolExecutionPayload(
            delegation=DelegationRequest(
                agent_name="executor",
                goal_id="goal-parent:executor",
                description="Apply the scoped mutation",
            )
        ),
    )

    with patch.object(harness.runner, "run") as run_mock:
        run_mock.return_value = ThinRuntimeExecutionResult(
            run_id="child-run",
            goal=RuntimeGoal(
                goal_id="goal-parent:executor",
                description="Apply the scoped mutation",
                max_iterations=delegated_agent.budgets["max_iterations"],
            ),
            agent=delegated_agent,
            artifact_dir="/tmp/child-run",
            success=True,
        )

        execute_delegation = getattr(harness.runner, "_execute_delegation")
        execute_delegation(delegation_result, context=context, depth=0)

    delegated_goal = run_mock.call_args.args[0]
    assert isinstance(delegated_goal, RuntimeGoal)
    assert delegated_goal.max_iterations == delegated_agent.budgets["max_iterations"]
    assert delegated_goal.max_iterations == 6


def test_execute_delegation_falls_back_to_parent_iteration_budget() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    delegated_agent = harness.agents.get("executor")
    original_budgets = dict(delegated_agent.budgets)
    delegated_agent.budgets = {}
    context = ContextSnapshot(
        goal={"max_iterations": 8},
        runtime={"active_agent": "improver", "lead_agent": "improver", "max_iterations": 8},
    )
    delegation_result = ToolResult(
        tool_name="delegate_task",
        success=True,
        payload=ToolExecutionPayload(
            delegation=DelegationRequest(
                agent_name="executor",
                goal_id="goal-parent:executor",
                description="Apply the scoped mutation",
            )
        ),
    )

    try:
        with patch.object(harness.runner, "run") as run_mock:
            run_mock.return_value = ThinRuntimeExecutionResult(
                run_id="child-run",
                goal=RuntimeGoal(
                    goal_id="goal-parent:executor",
                    description="Apply the scoped mutation",
                    max_iterations=8,
                ),
                agent=delegated_agent,
                artifact_dir="/tmp/child-run",
                success=True,
            )

            execute_delegation = getattr(harness.runner, "_execute_delegation")
            execute_delegation(delegation_result, context=context, depth=0)

        delegated_goal = run_mock.call_args.args[0]
        assert isinstance(delegated_goal, RuntimeGoal)
        assert delegated_goal.max_iterations == 8
    finally:
        delegated_agent.budgets = original_budgets


def test_execute_delegation_uses_minimum_iteration_budget_floor() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    delegated_agent = harness.agents.get("executor")
    original_budgets = dict(delegated_agent.budgets)
    delegated_agent.budgets = {}
    context = ContextSnapshot(
        runtime={"active_agent": "improver", "lead_agent": "improver"},
    )
    delegation_result = ToolResult(
        tool_name="delegate_task",
        success=True,
        payload=ToolExecutionPayload(
            delegation=DelegationRequest(
                agent_name="executor",
                goal_id="goal-parent:executor",
                description="Apply the scoped mutation",
            )
        ),
    )

    try:
        with patch.object(harness.runner, "run") as run_mock:
            run_mock.return_value = ThinRuntimeExecutionResult(
                run_id="child-run",
                goal=RuntimeGoal(
                    goal_id="goal-parent:executor",
                    description="Apply the scoped mutation",
                    max_iterations=1,
                ),
                agent=delegated_agent,
                artifact_dir="/tmp/child-run",
                success=True,
            )

            execute_delegation = getattr(harness.runner, "_execute_delegation")
            execute_delegation(delegation_result, context=context, depth=0)

        delegated_goal = run_mock.call_args.args[0]
        assert isinstance(delegated_goal, RuntimeGoal)
        assert delegated_goal.max_iterations == 1
    finally:
        delegated_agent.budgets = original_budgets
