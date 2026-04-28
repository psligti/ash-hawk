from __future__ import annotations

from types import SimpleNamespace

from dawn_kestrel.tools.framework import ToolResult as DkToolResult

from ash_hawk.thin_runtime.models import ContextSnapshot, RuntimeGoal, ToolResult
from ash_hawk.thin_runtime.state_adapter import ThinRuntimeStateAdapter
from ash_hawk.thin_runtime.tool_types import ToolExecutionPayload


def test_state_adapter_passes_state_depth_to_native_delegation() -> None:
    captured: dict[str, object] = {}

    def _emit(_name: str, _payload: dict[str, object]) -> None:
        return None

    def _to_dk_result(_result: ToolResult) -> DkToolResult:
        return DkToolResult(output="ok", error=None)

    class _Runner:
        def __init__(self) -> None:
            self.hooks = SimpleNamespace(emit=_emit)

        def native_after_tool(self, **_kwargs: object) -> None:
            return None

        def native_execute_delegation(
            self,
            _result: ToolResult,
            *,
            context: ContextSnapshot,
            depth: int,
        ) -> tuple[None, None]:
            del context
            captured["depth"] = depth
            return None, None

        def native_merge_seed_context(
            self,
            _context: ContextSnapshot,
            _seed: ContextSnapshot,
        ) -> None:
            return None

        def native_available_contexts_from_snapshot(
            self,
            _context: ContextSnapshot,
        ) -> set[str]:
            return {"goal_context"}

        def native_update_iteration_state(
            self,
            _context: ContextSnapshot,
            _goal: RuntimeGoal,
            _agent: object,
            _selected_tool_names: list[str],
        ) -> None:
            return None

        def native_refresh_runtime_view(
            self,
            *,
            goal: RuntimeGoal,
            agent: object,
            active_skills: list[object],
            candidate_skills: list[object],
            context: ContextSnapshot,
        ) -> None:
            del goal, agent, active_skills, candidate_skills, context
            return None

        def native_checkpoint_output(
            self,
            *,
            tool_name: str,
            result: ToolResult,
            context: ContextSnapshot,
        ) -> str:
            del tool_name, result, context
            return "ok"

        def native_write_matching_skill_memory(
            self,
            _agent: object,
            _matching_skills: list[object],
            _tool_name: str,
            _tool_count: int,
        ) -> None:
            return None

        def native_should_reset_tool_cycle(self, **_kwargs: object) -> bool:
            return False

    state = SimpleNamespace(
        goal=RuntimeGoal(goal_id="goal-depth", description="depth"),
        agent=SimpleNamespace(name="coordinator", iteration_completion_tools=[]),
        context=ContextSnapshot(runtime={"available_contexts": []}),
        available_contexts=set(),
        depth=2,
        terminated=False,
        terminal_success=True,
        terminal_reason=None,
        delegations=[],
        selected_tool_names=[],
        tool_results=[],
        active_skills=[],
        candidate_skills=[],
        tool_execution_order=None,
        remaining_tools=[],
    )

    adapter = ThinRuntimeStateAdapter(
        runner=_Runner(),
        state=state,
        to_dk_result=_to_dk_result,
    )

    result = ToolResult(
        tool_name="load_workspace_state", success=True, payload=ToolExecutionPayload()
    )
    adapter.apply_tool_result(tool_name="load_workspace_state", matching_skills=[], result=result)

    assert captured["depth"] == 2
