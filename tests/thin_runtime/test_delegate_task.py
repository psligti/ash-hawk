from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from dawn_kestrel.provider.llm_client import LLMResponse
from dawn_kestrel.provider.provider_types import TokenUsage

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl import delegate_task
from ash_hawk.thin_runtime.tool_types import (
    RuntimeToolContext,
    ToolCallContext,
)


class _ChooseToolClient:
    provider_id = "test"

    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        options: object = None,
    ) -> LLMResponse:
        del messages
        del options
        tool_defs = tools or []
        selected = ""
        for tool_def in tool_defs:
            function = tool_def.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    selected = name
                    break
        if not selected:
            return LLMResponse(
                text="No tool call",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )
        return LLMResponse(
            text=f"Choosing {selected}",
            usage=TokenUsage(input=0, output=0, reasoning=0),
            finish_reason="tool_use",
            cost=Decimal("0"),
            tool_calls=[
                {
                    "id": f"call-{selected}",
                    "type": "function",
                    "function": {"name": selected, "arguments": "{}"},
                    "tool": selected,
                    "input": {},
                }
            ],
        )


def test_delegate_task_emits_delegation_payload() -> None:
    call = ToolCall(
        tool_name="delegate_task",
        goal_id="goal-123",
        tool_args={
            "agent_name": "executor",
            "description": "Apply mutation to target file",
            "requested_skills": ["workspace-governance", "workspace-governance"],
            "requested_tools": ["mutate_agent_files", "diff_workspace_changes"],
        },
        context=ToolCallContext(
            runtime=RuntimeToolContext(active_agent="improver", lead_agent="improver")
        ),
    )

    result = delegate_task.run(call)

    assert result.success is True
    assert result.payload.delegation is not None
    delegation = result.payload.delegation
    assert delegation.agent_name == "executor"
    assert delegation.requested_skills == ["workspace-governance"]
    assert delegation.requested_tools == ["mutate_agent_files", "diff_workspace_changes"]
    assert delegation.goal_id == "goal-123:executor"


def test_requested_tools_constrain_runtime_tool_surface() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    harness.runner.set_client_factory(_ChooseToolClient)

    result = harness.runner.run(
        RuntimeGoal(
            goal_id="goal-constraint",
            description="Constrain executor tools",
            max_iterations=1,
        ),
        agent_name="executor",
        requested_skills=None,
        requested_tools=["load_workspace_state"],
        tool_execution_order=None,
    )

    assert result.selected_tool_names
    assert result.selected_tool_names[0] == "load_workspace_state"


def test_requested_tools_fallback_when_requested_set_has_no_matches() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    harness.runner.set_client_factory(_ChooseToolClient)

    result = harness.runner.run(
        RuntimeGoal(
            goal_id="goal-fallback",
            description="Fallback when requested tools mismatch",
            max_iterations=1,
        ),
        agent_name="executor",
        requested_skills=None,
        requested_tools=["shell", "read_file"],
        tool_execution_order=None,
    )

    assert result.selected_tool_names
    assert result.selected_tool_names[0] == "load_workspace_state"
    active_tools = result.context.tool.get("active_tools", [])
    assert isinstance(active_tools, list)
    assert "load_workspace_state" in active_tools


def test_delegation_result_summary_includes_child_tool_outcomes() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    harness.runner.set_client_factory(_ChooseToolClient)

    delegated_result = harness.runner.run(
        RuntimeGoal(goal_id="goal-child", description="Child run for summary", max_iterations=1),
        agent_name="executor",
        requested_skills=None,
        requested_tools=["load_workspace_state"],
        tool_execution_order=None,
    )

    summarize_delegation = getattr(harness.runner, "_delegation_result_summary")
    summary = summarize_delegation(delegated_result)
    assert "executor" in summary
    assert "load_workspace_state" in summary
