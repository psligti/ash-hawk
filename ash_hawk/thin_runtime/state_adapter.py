from __future__ import annotations

# type-hygiene: skip-file  # dynamic DK/native bridge state is intentionally structural
from typing import Any

from dawn_kestrel.tools.framework import ToolResult as DkToolResult

from ash_hawk.thin_runtime.models import ToolResult


class ThinRuntimeStateAdapter:
    def __init__(self, *, runner: Any, state: Any, to_dk_result: Any) -> None:
        self._runner = runner
        self._state = state
        self._to_dk_result = to_dk_result

    def apply_tool_result(
        self,
        *,
        tool_name: str,
        matching_skills: list[Any],
        result: ToolResult,
    ) -> DkToolResult:
        self._runner.native_after_tool(
            goal=self._state.goal,
            agent=self._state.agent,
            tool_name=tool_name,
            matching_skills=matching_skills,
            result=result,
            context=self._state.context,
            available_contexts=self._state.available_contexts,
        )

        def checkpoint_result() -> DkToolResult:
            return DkToolResult(
                output=self._runner.native_checkpoint_output(
                    tool_name=tool_name,
                    result=result,
                    context=self._state.context,
                ),
                error=result.error,
                metadata={"tool_name": result.tool_name, "success": result.success},
            )

        if not result.success:
            self._state.terminated = True
            self._state.terminal_success = False
            self._state.terminal_reason = result.error
            self._runner.native_refresh_runtime_view(
                goal=self._state.goal,
                agent=self._state.agent,
                active_skills=self._state.active_skills,
                candidate_skills=self._state.candidate_skills,
                context=self._state.context,
            )
            return checkpoint_result()

        delegation_record, delegated_context = self._runner.native_execute_delegation(
            result,
            context=self._state.context,
            depth=self._state.depth,
        )
        if delegation_record is not None:
            self._state.delegations.append(delegation_record)
            self._state.context.audit.setdefault("delegations", []).append(
                delegation_record.model_dump()
            )
        if delegated_context is not None:
            self._runner.native_merge_seed_context(self._state.context, delegated_context)
        self._state.available_contexts = self._runner.native_available_contexts_from_snapshot(
            self._state.context
        )
        self._state.context.runtime["available_contexts"] = sorted(self._state.available_contexts)
        self._runner.native_update_iteration_state(
            self._state.context,
            self._state.goal,
            self._state.agent,
            self._state.selected_tool_names,
        )
        self._runner.native_refresh_runtime_view(
            goal=self._state.goal,
            agent=self._state.agent,
            active_skills=self._state.active_skills,
            candidate_skills=self._state.candidate_skills,
            context=self._state.context,
        )

        if result.payload.stop:
            stop_reason = result.payload.runtime_updates.stop_reason or "Tool requested stop"
            self._state.terminated = True
            self._state.terminal_success = True
            self._state.terminal_reason = stop_reason
            self._runner.hooks.emit(
                "on_stop_condition",
                {
                    "goal_id": self._state.goal.goal_id,
                    "agent": self._state.agent.name,
                    "reason": stop_reason,
                },
            )
            return checkpoint_result()

        write_error = self._runner.native_write_matching_skill_memory(
            self._state.agent,
            matching_skills,
            tool_name,
            len(self._state.tool_results),
        )
        if write_error is not None:
            self._state.terminated = True
            self._state.terminal_success = False
            self._state.terminal_reason = write_error
            return DkToolResult(output=write_error, error=write_error)

        if self._runner.native_should_reset_tool_cycle(
            goal=self._state.goal,
            agent=self._state.agent,
            tool_name=tool_name,
            completion_tools=set(self._state.agent.iteration_completion_tools),
            selected_tool_names=self._state.selected_tool_names,
        ):
            self._state.remaining_tools = list(self._state.tool_execution_order or [])

        return checkpoint_result()
