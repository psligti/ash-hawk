from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import uuid4

from ash_hawk.thin_runtime.agent_text import build_agent_text, build_live_brief
from ash_hawk.thin_runtime.agents import AgentRegistry
from ash_hawk.thin_runtime.context import RuntimeContextAssembler
from ash_hawk.thin_runtime.dream_state import DEFERRED_SCOPES, DreamStateConsolidator
from ash_hawk.thin_runtime.hooks import HookDispatcher
from ash_hawk.thin_runtime.live_eval import stream_observed_events
from ash_hawk.thin_runtime.memory import ThinRuntimeMemoryManager
from ash_hawk.thin_runtime.models import (
    AgentSpec,
    ContextSnapshot,
    DelegationRecord,
    RuntimeGoal,
    SkillSpec,
    ThinRuntimeExecutionResult,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from ash_hawk.thin_runtime.persistence import ThinRuntimePersistence
from ash_hawk.thin_runtime.planner import PlannerDecision, build_tool_contract_view, plan_next_tool
from ash_hawk.thin_runtime.skills import SkillRegistry
from ash_hawk.thin_runtime.tool_types import (
    AcceptanceStatus,
    AuditRunResult,
    AuditToolContext,
    ClaimAuditStatus,
    EvaluationToolContext,
    FailureBucketCount,
    FailureCluster,
    FailureToolContext,
    MemoryToolContext,
    Phase2GateStatus,
    Phase2Metrics,
    RankedHypothesis,
    RuntimeToolContext,
    ScoreSummary,
    ToolCallContext,
    ToolContractView,
    ToolExecutionPayload,
    ToolStateContext,
    TraceRecord,
    TranscriptRecord,
    VerificationStatus,
    WorkspaceToolContext,
)
from ash_hawk.thin_runtime.tools import ToolRegistry
from ash_hawk.types import ToolSurfacePolicy


class AgenticLoopRunner:
    def __init__(
        self,
        *,
        agents: AgentRegistry,
        skills: SkillRegistry,
        tools: ToolRegistry,
        hooks: HookDispatcher,
        memory: ThinRuntimeMemoryManager,
        persistence: ThinRuntimePersistence,
        context: RuntimeContextAssembler,
        workdir: Path,
        policy: ToolSurfacePolicy,
    ) -> None:
        self.agents = agents
        self.skills = skills
        self.tools = tools
        self.hooks = hooks
        self.memory = memory
        self.persistence = persistence
        self.dream_state = DreamStateConsolidator(memory=memory, persistence=persistence)
        self.context = context
        self.workdir = workdir
        self.policy = policy
        self._decision_trace_buffer: list[str] = []

    def _refresh_runtime_view(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        skills: list[SkillSpec],
        tools: list[ToolSpec],
        context: ContextSnapshot,
        available_skills: list[SkillSpec] | None = None,
        include_skill_instructions: bool = True,
    ) -> None:
        memory_snapshot = self.memory.snapshot()
        current_workdir_raw = context.workspace.get("workdir")
        current_workdir = (
            Path(str(current_workdir_raw)).resolve()
            if isinstance(current_workdir_raw, str) and current_workdir_raw.strip()
            else self.workdir
        )
        self.context.refresh(
            snapshot=context,
            goal=goal,
            agent=agent,
            skills=skills,
            tools=tools,
            memory_snapshot=memory_snapshot,
            workdir=current_workdir,
            available_skills=available_skills,
        )
        context.runtime["live_brief"] = build_live_brief(context)
        context.runtime["agent_text"] = build_agent_text(
            goal,
            agent,
            skills,
            tools=tools,
            context_snapshot=context,
            memory_snapshot=memory_snapshot,
            include_skill_instructions=include_skill_instructions,
        )

    def run(
        self,
        goal: RuntimeGoal,
        *,
        agent_name: str,
        requested_skills: list[str] | None,
        requested_tools: list[str] | None = None,
        tool_execution_order: list[str] | None = None,
        scenario_path: str | None = None,
        seed_context: ContextSnapshot | None = None,
        depth: int = 0,
    ) -> ThinRuntimeExecutionResult:
        run_id = f"{goal.goal_id}-{uuid4().hex[:8]}"
        agent = self.agents.get(agent_name)
        active_skills = self._resolve_active_skills(agent, requested_skills)
        candidate_skills = self._resolve_candidate_skills(agent, requested_skills)
        active_tools = self._resolve_active_tools(
            agent,
            active_skills,
            candidate_skills,
            requested_tools=requested_tools,
        )
        context = self.context.assemble(
            goal=goal,
            agent=agent,
            skills=active_skills,
            tools=active_tools,
            memory_snapshot=self.memory.snapshot(),
            workdir=self.workdir,
            available_skills=candidate_skills,
        )
        if scenario_path is not None:
            context.workspace["scenario_path"] = scenario_path
        if tool_execution_order is not None:
            context.runtime["explicit_order_override"] = True
        if seed_context is not None:
            self._merge_seed_context(context, seed_context)
            context.runtime["available_contexts"] = sorted(
                self._available_contexts_from_snapshot(context)
            )
        self._refresh_runtime_view(
            goal=goal,
            agent=agent,
            skills=active_skills,
            tools=active_tools,
            context=context,
            available_skills=candidate_skills,
        )
        return self._run_loop(
            run_id=run_id,
            goal=goal,
            agent=agent,
            active_skills=active_skills,
            candidate_skills=candidate_skills,
            active_tools=active_tools,
            context=context,
            tool_execution_order=tool_execution_order,
            requested_tools=requested_tools,
            depth=depth,
        )

    def _run_loop(
        self,
        *,
        run_id: str,
        goal: RuntimeGoal,
        agent: AgentSpec,
        active_skills: list[SkillSpec],
        candidate_skills: list[SkillSpec],
        active_tools: list[ToolSpec],
        context: ContextSnapshot,
        tool_execution_order: list[str] | None,
        requested_tools: list[str] | None,
        depth: int,
    ) -> ThinRuntimeExecutionResult:
        tool_results: list[ToolResult] = []
        delegations: list[DelegationRecord] = []
        selected_tool_names: list[str] = []
        available_contexts_raw = context.runtime.get("available_contexts")
        if isinstance(available_contexts_raw, list) and available_contexts_raw:
            available_contexts = {item for item in available_contexts_raw if isinstance(item, str)}
        else:
            available_contexts = self._available_contexts_from_snapshot(context)
        success = True
        error: str | None = None
        self._decision_trace_buffer = []

        self.hooks.emit(
            "before_run",
            {
                "goal_id": goal.goal_id,
                "description": goal.description,
                "agent": agent.name,
                "skills": [skill.name for skill in active_skills],
                "max_iterations": goal.max_iterations,
            },
        )
        self.hooks.emit("before_agent", {"goal_id": goal.goal_id, "agent": agent.name})
        context.runtime["run_id"] = run_id
        self._record_agent_event(
            goal=goal,
            agent=agent,
            event_type="enter",
            success=True,
            run_id=run_id,
        )
        self._read_agent_memory(agent)
        self._refresh_context_from_memory(context, goal_id=goal.goal_id, run_id=run_id)
        self._update_iteration_state(context, goal, agent, selected_tool_names)
        self._refresh_runtime_view(
            goal=goal,
            agent=agent,
            skills=active_skills,
            tools=active_tools,
            context=context,
            available_skills=candidate_skills,
        )
        context.runtime["available_contexts"] = sorted(available_contexts)

        touched_skill_reads: set[str] = set()
        remaining_tools = list(tool_execution_order or [])
        completion_tools = set(agent.iteration_completion_tools)

        while True:
            tool_budget_error = self._check_tool_budget(goal, agent, selected_tool_names)
            if tool_budget_error is not None:
                success = False
                error = tool_budget_error
                break

            stop_error = self._check_iteration_stop(goal, agent, selected_tool_names)
            if stop_error is not None:
                success = False
                error = stop_error
                break

            active_tools = self._resolve_active_tools(
                agent,
                active_skills,
                candidate_skills,
                requested_tools=requested_tools,
            )
            context.tool["active_tools"] = [tool.name for tool in active_tools]
            context.tool["available_tools_snapshot"] = [tool.name for tool in active_tools]
            context.tool["resolved_toolset"] = [tool.name for tool in active_tools]
            context.tool["tool_contracts"] = [
                build_tool_contract_view(tool).model_dump(exclude_none=True)
                for tool in active_tools
            ]
            self._refresh_runtime_view(
                goal=goal,
                agent=agent,
                skills=active_skills,
                tools=active_tools,
                context=context,
                available_skills=candidate_skills,
            )

            decision = self._select_next_tool(
                goal=goal,
                agent=agent,
                active_skills=active_skills,
                candidate_skills=candidate_skills,
                active_tools=active_tools,
                remaining_tools=remaining_tools,
                tool_execution_order=tool_execution_order,
                context=context,
                available_contexts=available_contexts,
                tool_results=tool_results,
            )
            tool_name = decision.selected_tool
            if decision.activate_skills:
                active_skills = self._activate_skills(
                    active_skills=active_skills,
                    candidate_skills=candidate_skills,
                    requested_skill_names=decision.activate_skills,
                )
                context.runtime["active_skills"] = [skill.name for skill in active_skills]
            if tool_name is None:
                failure_family = context.failure.get("failure_family")
                completed_loops = self._current_iteration_count(agent, selected_tool_names)
                mutated_files = context.workspace.get("mutated_files", [])
                if agent.iteration_budget_mode == "loop" and (
                    isinstance(failure_family, str)
                    and failure_family.strip()
                    or completed_loops == 0
                    or not isinstance(mutated_files, list)
                    or not mutated_files
                ):
                    error = "No eligible tools available for a complete diagnosis-mutation-re-evaluation loop"
                    success = False
                    self.hooks.emit(
                        "on_stop_condition",
                        {"goal_id": goal.goal_id, "agent": agent.name, "reason": error},
                    )
                else:
                    if selected_tool_names:
                        success = True
                        error = None
                        self.hooks.emit(
                            "on_stop_condition",
                            {
                                "goal_id": goal.goal_id,
                                "agent": agent.name,
                                "reason": "No additional eligible tools available",
                            },
                        )
                    else:
                        error = "No eligible tools available for current contexts"
                        success = False
                        self.hooks.emit(
                            "on_stop_condition",
                            {"goal_id": goal.goal_id, "agent": agent.name, "reason": error},
                        )
                break

            if tool_execution_order is not None and tool_name in remaining_tools:
                remaining_tools.remove(tool_name)
            selected_tool_names.append(tool_name)
            matching_skills = self._matching_skills(active_skills, tool_name, available_contexts)

            self._read_matching_skill_memory(
                goal,
                agent,
                matching_skills,
                touched_skill_reads,
                run_id=run_id,
            )
            result = self._invoke_tool(
                goal=goal,
                agent=agent,
                tool_name=tool_name,
                matching_skills=matching_skills,
                context=context,
                available_contexts=available_contexts,
                remaining_tools=remaining_tools,
                selected_tool_names=selected_tool_names,
            )
            tool_results.append(result)
            self._after_tool(
                goal=goal,
                agent=agent,
                tool_name=tool_name,
                matching_skills=matching_skills,
                result=result,
                context=context,
                available_contexts=available_contexts,
            )

            if not result.success:
                success = False
                error = result.error
                break

            delegation_record, delegated_context = self._execute_delegation(
                result, context=context, depth=depth
            )
            if delegation_record is not None:
                delegations.append(delegation_record)
                context.audit.setdefault("delegations", []).append(delegation_record.model_dump())
            if delegated_context is not None:
                self._merge_seed_context(context, delegated_context)
            available_contexts = self._available_contexts_from_snapshot(context)
            context.runtime["available_contexts"] = sorted(available_contexts)
            self._update_iteration_state(context, goal, agent, selected_tool_names)

            if result.payload.stop:
                stop_reason = result.payload.runtime_updates.stop_reason
                self.hooks.emit(
                    "on_stop_condition",
                    {"goal_id": goal.goal_id, "agent": agent.name, "reason": stop_reason},
                )
                break

            error = self._write_matching_skill_memory(
                agent, matching_skills, tool_name, len(tool_results)
            )
            if error is not None:
                success = False
                break

            if self._should_reset_tool_cycle(
                goal=goal,
                agent=agent,
                tool_name=tool_name,
                completion_tools=completion_tools,
                selected_tool_names=selected_tool_names,
            ):
                remaining_tools = list(tool_execution_order or [])

        if (
            success
            and agent.iteration_budget_mode == "loop"
            and any(skill.name == "improvement-loop" for skill in active_skills)
            and self._current_iteration_count(agent, selected_tool_names) == 0
        ):
            success = False
            error = "No completed improvement loops recorded"
            self.hooks.emit(
                "on_stop_condition",
                {"goal_id": goal.goal_id, "agent": agent.name, "reason": error},
            )
        self.hooks.emit(
            "after_agent", {"goal_id": goal.goal_id, "agent": agent.name, "success": success}
        )
        self._record_agent_event(
            goal=goal,
            agent=agent,
            event_type="exit",
            success=success,
            error=error,
            run_id=run_id,
        )
        self.hooks.emit(
            "after_run",
            {"goal_id": goal.goal_id, "agent": agent.name, "tool_count": len(tool_results)},
        )

        self.context.refresh(
            snapshot=context,
            goal=goal,
            agent=agent,
            skills=active_skills,
            tools=active_tools,
            memory_snapshot=self.memory.snapshot(),
            workdir=self.workdir,
            available_skills=candidate_skills,
        )

        execution = ThinRuntimeExecutionResult(
            run_id=run_id,
            goal=goal,
            agent=agent,
            active_skills=active_skills,
            tools=active_tools,
            context=context,
            tool_results=tool_results,
            emitted_hooks=self.hooks.emitted(),
            delegations=delegations,
            memory_snapshot=self.memory.snapshot(),
            artifact_dir=str(self.persistence.run_dir(run_id)),
            available_contexts=sorted(available_contexts),
            selected_tool_names=selected_tool_names,
            success=success,
            error=error,
        )
        self._persist_and_dream(execution)
        self.persistence.persist_execution(execution)
        return execution

    def _check_iteration_stop(
        self, goal: RuntimeGoal, agent: AgentSpec, selected_tool_names: list[str]
    ) -> str | None:
        if self._current_iteration_count(agent, selected_tool_names) < goal.max_iterations:
            return None
        error = f"Reached max iterations: {goal.max_iterations}"
        self.hooks.emit(
            "on_stop_condition", {"goal_id": goal.goal_id, "agent": agent.name, "reason": error}
        )
        return error

    def _check_tool_budget(
        self,
        goal: RuntimeGoal,
        agent: AgentSpec,
        selected_tool_names: list[str],
    ) -> str | None:
        max_tool_calls = agent.budgets.get("max_tool_calls")
        if not isinstance(max_tool_calls, int) or len(selected_tool_names) < max_tool_calls:
            return None
        error = f"Reached max tool calls: {max_tool_calls}"
        self.hooks.emit(
            "on_stop_condition",
            {"goal_id": goal.goal_id, "agent": agent.name, "reason": error},
        )
        return error

    def _current_iteration_count(self, agent: AgentSpec, selected_tool_names: list[str]) -> int:
        if agent.iteration_budget_mode != "loop":
            return len(selected_tool_names)
        completion_tools = set(agent.iteration_completion_tools)
        if not completion_tools:
            return len(selected_tool_names)
        return sum(1 for tool_name in selected_tool_names if tool_name in completion_tools)

    def _tool_preconditions_met(self, tool_name: str, context: ContextSnapshot) -> bool:
        if not self._phase_allows_tool(tool_name, context):
            return False
        if tool_name == "run_baseline_eval":
            return not self._baseline_ready(context)
        if tool_name == "detect_agent_config":
            return not self._agent_config_ready(context)
        if tool_name == "mutate_agent_files":
            return self._mutation_candidates_available(context)
        if tool_name == "run_eval_repeated":
            mutated_files = context.workspace.get("mutated_files", [])
            return isinstance(mutated_files, list) and len(mutated_files) > 0
        if tool_name == "delegate_task":
            return self._delegation_ready(context)
        return True

    def _phase_allows_tool(self, tool_name: str, context: ContextSnapshot) -> bool:
        if context.runtime.get("explicit_order_override") is True:
            return True
        active_agent = context.runtime.get("active_agent") or context.runtime.get("lead_agent")
        active_skills = context.runtime.get("active_skills", [])
        if (
            active_agent == "improver"
            and isinstance(active_skills, list)
            and "improvement-loop" in active_skills
        ):
            return self._improver_phase_allows_tool(tool_name, context)
        if active_agent == "executor":
            return self._executor_phase_allows_tool(tool_name, context)
        return True

    def _improver_phase_allows_tool(self, tool_name: str, context: ContextSnapshot) -> bool:
        if not self._has_executed_tool(context, "load_workspace_state"):
            return tool_name == "load_workspace_state"
        if not self._agent_config_ready(context):
            return tool_name == "detect_agent_config"
        if not self._baseline_ready(context):
            return tool_name == "run_baseline_eval"
        if self._sync_back_pending(context):
            return tool_name == "sync_workspace_changes"
        if self._candidate_validation_pending(context):
            return tool_name == "run_eval_repeated"
        if self._candidate_rejected(context) or not self._mutation_candidates_available(context):
            return tool_name == "call_llm_structured"
        return tool_name == "delegate_task"

    def _executor_phase_allows_tool(self, tool_name: str, context: ContextSnapshot) -> bool:
        if not self._mutation_candidates_available(context):
            return True
        isolated_workspace_path = context.workspace.get("isolated_workspace_path")
        if not isinstance(isolated_workspace_path, str) or not isolated_workspace_path.strip():
            return tool_name == "prepare_isolated_workspace"
        mutated_files = context.workspace.get("mutated_files", [])
        if not isinstance(mutated_files, list) or not mutated_files:
            return tool_name == "mutate_agent_files"
        return False

    def _candidate_validation_pending(self, context: ContextSnapshot) -> bool:
        mutated_files = context.workspace.get("mutated_files", [])
        isolated_workspace_path = context.workspace.get("isolated_workspace_path")
        if not isinstance(mutated_files, list) or not mutated_files:
            return False
        if not isinstance(isolated_workspace_path, str) or not isolated_workspace_path.strip():
            return False
        sync_events = set(self._sync_events(context))
        return "candidate_validated" not in sync_events and "candidate_rejected" not in sync_events

    def _sync_back_pending(self, context: ContextSnapshot) -> bool:
        sync_events = set(self._sync_events(context))
        return "candidate_validated" in sync_events and "synced_back" not in sync_events

    def _candidate_rejected(self, context: ContextSnapshot) -> bool:
        sync_events = set(self._sync_events(context))
        mutated_files = context.workspace.get("mutated_files", [])
        return "candidate_rejected" in sync_events and (
            not isinstance(mutated_files, list) or not mutated_files
        )

    def _sync_events(self, context: ContextSnapshot) -> list[str]:
        raw = context.audit.get("sync_events", [])
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, str)]

    def _has_executed_tool(self, context: ContextSnapshot, tool_name: str) -> bool:
        raw_tool_results = context.audit.get("tool_results", [])
        if not isinstance(raw_tool_results, list):
            return False
        for item in raw_tool_results:
            if isinstance(item, dict) and item.get("tool") == tool_name:
                return True
        return False

    def _baseline_ready(self, context: ContextSnapshot) -> bool:
        baseline_summary = context.evaluation.get("baseline_summary")
        if not isinstance(baseline_summary, dict):
            return False
        status = baseline_summary.get("status")
        score = baseline_summary.get("score")
        return status == "completed" or score is not None

    def _agent_config_ready(self, context: ContextSnapshot) -> bool:
        required_keys = ("agent_config", "source_root", "package_name")
        return all(
            isinstance(context.workspace.get(key), str) and str(context.workspace.get(key)).strip()
            for key in required_keys
        )

    def _mutation_candidates_available(self, context: ContextSnapshot) -> bool:
        raw_hypotheses = context.failure.get("ranked_hypotheses")
        if isinstance(raw_hypotheses, list):
            for hypothesis in raw_hypotheses:
                if isinstance(hypothesis, dict) and hypothesis.get("target_files"):
                    return True
        for key in ("scenario_required_files", "allowed_target_files", "actionable_files"):
            value = context.workspace.get(key, [])
            if isinstance(value, list) and len(value) > 0:
                return True
        return False

    def _delegation_ready(self, context: ContextSnapshot) -> bool:
        active_skills = context.runtime.get("active_skills", [])
        if not isinstance(active_skills, list) or "improvement-loop" not in active_skills:
            return True
        if not self._baseline_ready(context):
            return False
        return self._mutation_candidates_available(context)

    def _update_iteration_state(
        self,
        context: ContextSnapshot,
        goal: RuntimeGoal,
        agent: AgentSpec,
        selected_tool_names: list[str],
    ) -> None:
        completed_iterations = self._current_iteration_count(agent, selected_tool_names)
        context.runtime["iteration_budget_mode"] = agent.iteration_budget_mode
        context.runtime["completed_iterations"] = completed_iterations
        context.runtime["remaining_iterations"] = max(goal.max_iterations - completed_iterations, 0)

    def _should_reset_tool_cycle(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        tool_name: str,
        completion_tools: set[str],
        selected_tool_names: list[str],
    ) -> bool:
        if agent.iteration_budget_mode != "loop":
            return False
        if tool_name not in completion_tools:
            return False
        return self._current_iteration_count(agent, selected_tool_names) < goal.max_iterations

    def _matching_skills(
        self, active_skills: list[SkillSpec], tool_name: str, available_contexts: set[str]
    ) -> list[SkillSpec]:
        return [
            skill
            for skill in active_skills
            if tool_name in skill.tool_names
            and set(skill.input_contexts).issubset(available_contexts)
        ]

    def _read_matching_skill_memory(
        self,
        goal: RuntimeGoal,
        agent: AgentSpec,
        matching_skills: list[SkillSpec],
        touched_skill_reads: set[str],
        *,
        run_id: str | None = None,
    ) -> None:
        for skill in matching_skills:
            if skill.name in touched_skill_reads:
                continue
            self.hooks.emit("before_skill", {"agent": agent.name, "skill": skill.name})
            self._record_skill_event(
                goal=goal,
                agent=agent,
                skill=skill,
                event_type="enter",
                success=True,
                run_id=run_id,
            )
            self._read_skill_memory(agent, skill)
            touched_skill_reads.add(skill.name)

    def _invoke_tool(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        tool_name: str,
        matching_skills: list[SkillSpec],
        context: ContextSnapshot,
        available_contexts: set[str],
        remaining_tools: list[str],
        selected_tool_names: list[str],
        tool_args: dict[str, object] | None = None,
    ) -> ToolResult:
        completed_iterations = self._current_iteration_count(agent, selected_tool_names)
        self.hooks.emit(
            "before_tool",
            {
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "tool": tool_name,
                "skills": [skill.name for skill in matching_skills],
                "tool_args": tool_args or {},
                "iterations": completed_iterations,
                "tool_call_count": len(selected_tool_names),
                "max_iterations": goal.max_iterations,
                "remaining_tools": list(remaining_tools),
            },
        )
        self._record_tool_event(
            goal=goal,
            agent=agent,
            tool_name=tool_name,
            event_type="enter",
            success=True,
            error=None,
            skills=matching_skills,
            run_id=(value if isinstance((value := context.runtime.get("run_id")), str) else None),
        )

        def _emit_observed_event(payload: dict[str, object]) -> None:
            self.hooks.emit(
                "on_observed_event",
                {
                    "goal_id": goal.goal_id,
                    "agent": agent.name,
                    "parent_tool": tool_name,
                    **payload,
                },
            )

        with stream_observed_events(_emit_observed_event):
            return self.tools.invoke(
                ToolCall(
                    tool_name=tool_name,
                    goal_id=goal.goal_id,
                    tool_args=tool_args or {},
                    caller_agent=agent.name,
                    caller_skill=matching_skills[0].name if matching_skills else None,
                    agent_text=str(context.runtime.get("agent_text", "")) or None,
                    skills=[skill.name for skill in matching_skills],
                    remaining_tools=remaining_tools,
                    available_contexts=sorted(available_contexts),
                    iterations=completed_iterations,
                    tool_call_count=len(selected_tool_names),
                    max_iterations=goal.max_iterations,
                    context=self._tool_call_context(
                        context, active_tools=[tool.name for tool in self.tools.list_tools()]
                    ),
                ),
                self.policy,
                available_contexts=available_contexts,
            )

    def _after_tool(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        tool_name: str,
        matching_skills: list[SkillSpec],
        result: ToolResult,
        context: ContextSnapshot,
        available_contexts: set[str],
    ) -> None:
        self._record_tool_side_effects(context, agent, matching_skills, result)
        self._update_available_contexts(
            available_contexts, self.tools.get(tool_name), matching_skills
        )
        current_run_id = context.runtime.get("run_id")
        self._refresh_context_from_memory(
            context,
            goal_id=goal.goal_id,
            run_id=current_run_id if isinstance(current_run_id, str) else None,
        )
        active_skill_names = context.runtime.get("active_skills", [])
        active_skill_specs = [
            self.skills.get(name)
            for name in active_skill_names
            if isinstance(name, str) and name.strip()
        ]
        active_tools = self._resolve_active_tools(agent, active_skill_specs, active_skill_specs)
        self._refresh_runtime_view(
            goal=goal,
            agent=agent,
            skills=active_skill_specs,
            tools=active_tools,
            context=context,
            available_skills=active_skill_specs,
        )
        context.runtime["available_contexts"] = sorted(available_contexts)
        self.memory.append(
            "artifact_memory",
            "events",
            {"agent": agent.name, "tool": tool_name, "success": result.success},
        )
        self.hooks.emit(
            "after_tool",
            {
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "tool": tool_name,
                "success": result.success,
                "skills": [skill.name for skill in matching_skills],
                "message": result.payload.message,
                "error": result.error,
                "failure_signals": list(result.payload.failure_updates.explanations),
                "tool_usage": list(result.payload.audit_updates.tool_usage),
                "llm_calls": list(result.payload.audit_updates.llm_calls),
                "deterministic_tool_calls": list(
                    result.payload.audit_updates.deterministic_tool_calls
                ),
                "agent_invocations": list(result.payload.audit_updates.agent_invocations),
                "artifacts": list(result.payload.audit_updates.artifacts),
                "events": [
                    event.model_dump(exclude_none=True)
                    for event in result.payload.audit_updates.events
                ],
                "run_summary": dict(result.payload.audit_updates.run_summary),
                "diff_report": dict(result.payload.audit_updates.diff_report),
                "run_result": result.payload.audit_updates.run_result.model_dump(exclude_none=True),
            },
        )
        self._record_tool_event(
            goal=goal,
            agent=agent,
            tool_name=tool_name,
            event_type="exit",
            success=result.success,
            error=result.error,
            skills=matching_skills,
            run_id=(value if isinstance((value := context.runtime.get("run_id")), str) else None),
        )
        for skill in matching_skills:
            self.hooks.emit(
                "after_skill",
                {"agent": agent.name, "skill": skill.name, "success": result.success},
            )
            self._record_skill_event(
                goal=goal,
                agent=agent,
                skill=skill,
                event_type="exit",
                success=result.success,
                error=result.error,
                run_id=(
                    value if isinstance((value := context.runtime.get("run_id")), str) else None
                ),
            )
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": context.runtime.get("run_id"),
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "tool": tool_name,
                "success": result.success,
                "error": result.error,
            },
            actor=agent.name,
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": context.runtime.get("run_id"),
                "goal_id": goal.goal_id,
                "speaker": agent.name,
                "type": "tool_call",
                "tool": tool_name,
                "message": f"{agent.name} executed {tool_name}",
                "success": result.success,
            },
            actor=agent.name,
        )
        if not result.success:
            self.hooks.emit(
                "on_policy_decision",
                {
                    "goal_id": goal.goal_id,
                    "agent": agent.name,
                    "tool": tool_name,
                    "error": result.error,
                },
            )

    def _write_matching_skill_memory(
        self,
        agent: AgentSpec,
        matching_skills: list[SkillSpec],
        tool_name: str,
        tool_count: int,
    ) -> str | None:
        for skill in matching_skills:
            try:
                self._write_skill_memory(agent, skill, tool_name, tool_count)
            except ValueError as exc:
                return str(exc)
        return None

    def _record_tool_selection_decision(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        decision: PlannerDecision,
        run_id: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "goal_id": goal.goal_id,
            "agent": agent.name,
            "tool": decision.selected_tool,
            "reason": decision.rationale,
            "source": decision.source,
            "considered_tools": decision.considered_tools,
            "confidence": decision.confidence,
            "activate_skills": decision.activate_skills,
            "reason_model_authored": decision.reason_model_authored,
        }
        self.hooks.emit("on_policy_decision", payload)
        context_message = (
            f"selected {decision.selected_tool or 'none'} via {decision.source}: "
            f"{decision.rationale}"
        )
        existing_decisions = getattr(self, "_decision_trace_buffer", None)
        if not isinstance(existing_decisions, list):
            self._decision_trace_buffer = []
            existing_decisions = self._decision_trace_buffer
        existing_decisions.append(context_message)
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "tool": decision.selected_tool,
                "event_type": "policy_decision",
                "phase": decision.source,
                "rationale": decision.rationale,
                "success": decision.selected_tool is not None,
            },
            actor=agent.name,
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "speaker": agent.name,
                "type": "policy_decision",
                "tool": decision.selected_tool,
                "message": context_message,
                "success": decision.selected_tool is not None,
            },
            actor=agent.name,
        )

    def _record_agent_event(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        event_type: str,
        success: bool,
        error: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "event_type": f"agent_{event_type}",
                "success": success,
                "error": error,
            },
            actor=agent.name,
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "speaker": agent.name,
                "type": f"agent_{event_type}",
                "message": f"{agent.name} {event_type}",
                "success": success,
            },
            actor=agent.name,
        )

    def _record_skill_event(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        skill: SkillSpec,
        event_type: str,
        success: bool,
        error: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "skill": skill.name,
                "event_type": f"skill_{event_type}",
                "success": success,
                "error": error,
            },
            actor=agent.name,
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "speaker": agent.name,
                "type": f"skill_{event_type}",
                "skill": skill.name,
                "message": f"{agent.name} {event_type} skill {skill.name}",
                "success": success,
            },
            actor=agent.name,
        )

    def _record_tool_event(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        tool_name: str,
        event_type: str,
        success: bool,
        error: str | None,
        skills: list[SkillSpec],
        run_id: str | None = None,
    ) -> None:
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "tool": tool_name,
                "skill": skills[0].name if skills else None,
                "event_type": f"tool_{event_type}",
                "success": success,
                "error": error,
            },
            actor=agent.name,
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": run_id,
                "goal_id": goal.goal_id,
                "speaker": agent.name,
                "type": f"tool_{event_type}",
                "tool": tool_name,
                "skill": skills[0].name if skills else None,
                "message": f"{agent.name} {event_type} {tool_name}",
                "success": success,
            },
            actor=agent.name,
        )

    def _select_next_tool(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        active_skills: list[SkillSpec],
        candidate_skills: list[SkillSpec],
        active_tools: list[ToolSpec],
        remaining_tools: list[str],
        tool_execution_order: list[str] | None,
        context: ContextSnapshot,
        available_contexts: set[str],
        tool_results: list[ToolResult],
    ) -> PlannerDecision:
        del available_contexts

        context.tool["recent_tool_history"] = self._planner_tool_history(tool_results)
        context.tool["last_tool"] = tool_results[-1].tool_name if tool_results else None

        tool_surface = active_tools
        if tool_execution_order is not None:
            tool_surface = [tool for tool in active_tools if tool.name in remaining_tools]
        candidate_tools = [
            tool for tool in tool_surface if self._tool_preconditions_met(tool.name, context)
        ]
        decision = plan_next_tool(
            goal=goal,
            agent=agent,
            active_skills=active_skills,
            candidate_skills=candidate_skills,
            candidate_tools=candidate_tools,
            context=context,
            tool_execution_order=tool_execution_order,
        )
        if decision.selected_tool is None and decision.source.startswith("invalid_model"):
            decision = plan_next_tool(
                goal=goal,
                agent=agent,
                active_skills=active_skills,
                candidate_skills=candidate_skills,
                candidate_tools=candidate_tools,
                context=context,
                tool_execution_order=tool_execution_order,
                feedback=decision.rationale,
            )
        if decision.selected_tool is None and tool_execution_order is None and candidate_tools:
            decision = plan_next_tool(
                goal=goal,
                agent=agent,
                active_skills=active_skills,
                candidate_skills=candidate_skills,
                candidate_tools=candidate_tools,
                context=context,
                tool_execution_order=tool_execution_order,
                feedback=(
                    "Choose exactly one next tool from the allowed tool surface. Return null only "
                    "if no safe tool exists."
                ),
            )
        context.tool["policy_decisions"] = list(context.tool.get("policy_decisions", [])) + [
            f"{decision.source}: {decision.rationale}"
        ]
        context.tool["context_sections"] = [
            "goal",
            "runtime",
            "workspace",
            "evaluation",
            "failure",
            "memory",
            "tool",
            "audit",
        ]
        context.tool["policy_checked"] = True
        context.tool["context_assembled"] = True
        context.runtime["last_decision"] = decision.rationale
        if decision.selected_tool is not None:
            context.runtime["preferred_tool"] = decision.selected_tool
        self._record_tool_selection_decision(
            goal=goal,
            agent=agent,
            decision=decision,
            run_id=(value if isinstance((value := context.runtime.get("run_id")), str) else None),
        )
        return decision

    def _planner_tool_history(self, tool_results: list[ToolResult]) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for result in tool_results[-5:]:
            payload = result.payload
            history.append(
                {
                    "tool": result.tool_name,
                    "success": result.success,
                    "error": result.error,
                    "message": payload.message,
                    "workspace_updates": sorted(
                        payload.workspace_updates.model_dump(
                            exclude_none=True,
                            exclude_defaults=True,
                        ).keys()
                    ),
                    "evaluation_updates": sorted(
                        payload.evaluation_updates.model_dump(
                            exclude_none=True,
                            exclude_defaults=True,
                        ).keys()
                    ),
                    "failure_updates": sorted(
                        payload.failure_updates.model_dump(
                            exclude_none=True,
                            exclude_defaults=True,
                        ).keys()
                    ),
                    "runtime_updates": sorted(
                        payload.runtime_updates.model_dump(
                            exclude_none=True,
                            exclude_defaults=True,
                        ).keys()
                    ),
                }
            )
        return history

    def _resolve_active_skills(
        self, agent: AgentSpec, requested_skills: list[str] | None
    ) -> list[SkillSpec]:
        skill_names = requested_skills or agent.skill_names
        return [self.skills.get(name) for name in skill_names]

    def _resolve_candidate_skills(
        self, agent: AgentSpec, requested_skills: list[str] | None
    ) -> list[SkillSpec]:
        candidate_names: list[str] = []
        for name in (requested_skills or []) + agent.skill_names + agent.available_skills:
            if name not in candidate_names:
                candidate_names.append(name)
        return [self.skills.get(name) for name in candidate_names]

    def _resolve_active_tools(
        self,
        agent: AgentSpec,
        active_skills: list[SkillSpec],
        candidate_skills: list[SkillSpec] | None = None,
        requested_tools: list[str] | None = None,
    ) -> list[ToolSpec]:
        tool_names: list[str] = []
        for tool_name in agent.available_tools:
            if tool_name not in tool_names:
                tool_names.append(tool_name)
        skill_source = candidate_skills or active_skills
        for skill in skill_source:
            for tool_name in skill.tool_names:
                if tool_name not in tool_names:
                    tool_names.append(tool_name)
        resolved = self.tools.resolve_allowed(tool_names, self.policy)
        if not requested_tools:
            return resolved
        requested = {name for name in requested_tools if name}
        if not requested:
            return resolved
        filtered = [tool for tool in resolved if tool.name in requested]
        if filtered:
            return filtered
        return resolved

    def _activate_skills(
        self,
        *,
        active_skills: list[SkillSpec],
        candidate_skills: list[SkillSpec],
        requested_skill_names: list[str],
    ) -> list[SkillSpec]:
        active_names = {skill.name for skill in active_skills}
        ordered = list(active_skills)
        for skill in candidate_skills:
            if skill.name in requested_skill_names and skill.name not in active_names:
                ordered.append(skill)
                active_names.add(skill.name)
        return ordered

    def _read_agent_memory(self, agent: AgentSpec) -> None:
        for scope_name in agent.memory_read_scopes:
            self.hooks.emit("before_memory_read", {"agent": agent.name, "scope": scope_name})
            self.memory.read_scope(scope_name)
            self.hooks.emit("after_memory_read", {"agent": agent.name, "scope": scope_name})

    def _read_skill_memory(self, agent: AgentSpec, skill: SkillSpec) -> None:
        for scope_name in skill.memory_read_scopes:
            self.hooks.emit(
                "before_memory_read",
                {"agent": agent.name, "scope": scope_name, "skill": skill.name},
            )
            self.memory.read_scope(scope_name)
            self.hooks.emit(
                "after_memory_read", {"agent": agent.name, "scope": scope_name, "skill": skill.name}
            )

    def _write_skill_memory(
        self, agent: AgentSpec, skill: SkillSpec, tool_name: str, tool_count: int
    ) -> None:
        for scope_name in skill.memory_write_scopes:
            self.hooks.emit(
                "before_memory_write",
                {"agent": agent.name, "scope": scope_name, "skill": skill.name},
            )
            entry = {
                "agent": agent.name,
                "skill": skill.name,
                "tool": tool_name,
                "tool_count": tool_count,
            }
            deferred_targets = self._deferred_memory_targets(
                agent=agent,
                skill=skill,
                tool_name=tool_name,
                tool_count=tool_count,
                entry=entry,
                scope_name=scope_name,
            )
            if scope_name in DEFERRED_SCOPES:
                if not self.memory.can_write_scope(scope_name, agent.name):
                    raise ValueError(
                        f"Actor '{agent.name}' cannot write memory scope '{scope_name}'"
                    )
                for key, value in deferred_targets:
                    self.memory.append(
                        "session_memory",
                        "dream_queue",
                        {"scope": scope_name, "key": key, "value": value},
                        actor=agent.name,
                    )
            else:
                self.memory.append(
                    scope_name,
                    "entries",
                    entry,
                    actor=agent.name,
                )
            self.hooks.emit(
                "after_memory_write",
                {"agent": agent.name, "scope": scope_name, "skill": skill.name},
            )

    def _deferred_memory_targets(
        self,
        *,
        agent: AgentSpec,
        skill: SkillSpec,
        tool_name: str,
        tool_count: int,
        entry: Mapping[str, object],
        scope_name: str,
    ) -> list[tuple[str, object]]:
        summary = f"{agent.name} used {skill.name} with {tool_name} at step {tool_count}."
        canonical_key = {
            "episodic_memory": "episodes",
            "semantic_memory": "rules",
            "personal_memory": "preferences",
            "artifact_memory": "artifacts",
        }.get(scope_name, "entries")
        canonical_value: object = summary if canonical_key != "entries" else entry
        return [(canonical_key, canonical_value), ("entries", entry)]

    def _record_tool_side_effects(
        self,
        context: ContextSnapshot,
        agent: AgentSpec,
        matching_skills: list[SkillSpec],
        result: ToolResult,
    ) -> None:
        audit_results = context.audit.setdefault("tool_results", [])
        if isinstance(audit_results, list):
            cast(list[dict[str, object]], audit_results).append(
                {
                    "agent": agent.name,
                    "skills": [skill.name for skill in matching_skills],
                    "tool": result.tool_name,
                    "success": result.success,
                    "error": result.error,
                    "payload": result.payload.model_dump(exclude_none=True, exclude_defaults=True),
                }
            )
        context.runtime["active_agent"] = agent.name
        context.tool["last_tool"] = result.tool_name
        self._merge_context_updates(context, result.payload)

    def _execute_delegation(
        self, result: ToolResult, *, context: ContextSnapshot, depth: int
    ) -> tuple[DelegationRecord | None, ContextSnapshot | None]:
        delegation = result.payload.delegation
        if delegation is None:
            return None, None
        if not self._delegation_ready(context):
            return (
                DelegationRecord(
                    agent_name=delegation.agent_name,
                    goal_id=delegation.goal_id,
                    selected_tool_names=[],
                    success=False,
                    error="Delegation requires a completed baseline and mutation-ready targets",
                ),
                None,
            )
        self.hooks.emit(
            "before_delegation",
            {
                "agent": context.runtime.get("active_agent"),
                "delegated_agent": delegation.agent_name,
                "goal_id": delegation.goal_id,
            },
        )
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": context.runtime.get("run_id"),
                "goal_id": delegation.goal_id,
                "agent": context.runtime.get("active_agent"),
                "event_type": "delegation_enter",
                "rationale": delegation.description,
                "success": True,
            },
            actor=str(context.runtime.get("active_agent") or "coordinator"),
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": context.runtime.get("run_id"),
                "goal_id": delegation.goal_id,
                "speaker": str(context.runtime.get("active_agent") or "coordinator"),
                "type": "delegation_enter",
                "message": delegation.description,
                "success": True,
            },
            actor=str(context.runtime.get("active_agent") or "coordinator"),
        )
        context.audit.setdefault("agent_invocations", []).append(delegation.agent_name)
        if depth >= 2:
            return (
                DelegationRecord(
                    agent_name=delegation.agent_name,
                    goal_id=delegation.goal_id,
                    selected_tool_names=[],
                    success=False,
                    error="Delegation depth limit reached",
                ),
                None,
            )
        agent_name = delegation.agent_name
        goal_id = delegation.goal_id
        description = delegation.description
        requested_skills = delegation.requested_skills
        requested_tools = delegation.requested_tools
        delegated_agent = self.agents.get(agent_name)
        delegated_result = self.run(
            RuntimeGoal(
                goal_id=goal_id,
                description=description,
                max_iterations=self._delegated_goal_max_iterations(
                    delegated_agent=delegated_agent,
                    parent_context=context,
                ),
            ),
            agent_name=agent_name,
            requested_skills=requested_skills or None,
            requested_tools=requested_tools or None,
            tool_execution_order=None,
            scenario_path=context.workspace.get("scenario_path"),
            seed_context=self._delegation_seed_context(context),
            depth=depth + 1,
        )
        summary = self._delegation_result_summary(delegated_result)
        context.runtime["last_delegation_summary"] = summary
        context.audit.setdefault("delegation_summaries", []).append(
            {
                "agent_name": delegated_result.agent.name,
                "goal_id": goal_id,
                "success": delegated_result.success,
                "error": delegated_result.error,
                "summary": summary,
            }
        )
        self.hooks.emit(
            "after_delegation",
            {
                "agent": context.runtime.get("active_agent"),
                "delegated_agent": delegated_result.agent.name,
                "goal_id": goal_id,
                "success": delegated_result.success,
            },
        )
        self.memory.append(
            "session_memory",
            "traces",
            {
                "run_id": context.runtime.get("run_id"),
                "goal_id": goal_id,
                "agent": context.runtime.get("active_agent"),
                "event_type": "delegation_exit",
                "success": delegated_result.success,
                "error": delegated_result.error,
            },
            actor=str(context.runtime.get("active_agent") or "coordinator"),
        )
        self.memory.append(
            "session_memory",
            "transcripts",
            {
                "run_id": context.runtime.get("run_id"),
                "goal_id": goal_id,
                "speaker": str(context.runtime.get("active_agent") or "coordinator"),
                "type": "delegation_exit",
                "message": f"delegation to {delegated_result.agent.name} completed",
                "success": delegated_result.success,
            },
            actor=str(context.runtime.get("active_agent") or "coordinator"),
        )
        return (
            DelegationRecord(
                agent_name=delegated_result.agent.name,
                goal_id=goal_id,
                selected_tool_names=delegated_result.selected_tool_names,
                success=delegated_result.success,
                error=delegated_result.error,
                summary=summary,
            ),
            delegated_result.context,
        )

    def _delegation_seed_context(self, context: ContextSnapshot) -> ContextSnapshot:
        seed = context.model_copy(deep=True)
        seed.workspace["mutated_files"] = []
        seed.workspace["isolated_workspace"] = False
        seed.workspace["isolated_workspace_path"] = ""
        seed.workspace["scenario_path"] = context.workspace.get(
            "source_scenario_path"
        ) or context.workspace.get("scenario_path")
        seed.workspace["source_scenario_path"] = ""
        seed.audit["sync_events"] = []
        return seed

    def _delegation_result_summary(self, delegated_result: ThinRuntimeExecutionResult) -> str:
        if not delegated_result.tool_results:
            if delegated_result.error:
                return f"{delegated_result.agent.name}: failed ({delegated_result.error})"
            status = "succeeded" if delegated_result.success else "failed"
            return f"{delegated_result.agent.name}: {status}"
        previews: list[str] = []
        for tool_result in delegated_result.tool_results[:3]:
            status = "ok" if tool_result.success else "failed"
            detail = (tool_result.payload.message or tool_result.error or "").strip()
            if detail:
                detail = detail.replace("\n", " ")
                if len(detail) > 80:
                    detail = f"{detail[:77]}..."
                previews.append(f"{tool_result.tool_name}:{status} ({detail})")
            else:
                previews.append(f"{tool_result.tool_name}:{status}")
        if len(delegated_result.tool_results) > 3:
            previews.append(f"+{len(delegated_result.tool_results) - 3} more")
        status_text = "succeeded" if delegated_result.success else "failed"
        base = f"{delegated_result.agent.name} {status_text}; " + "; ".join(previews)
        if len(base) > 320:
            return f"{base[:317]}..."
        return base

    def _delegated_goal_max_iterations(
        self,
        *,
        delegated_agent: AgentSpec,
        parent_context: ContextSnapshot,
    ) -> int:
        configured_budget = delegated_agent.budgets.get("max_iterations")
        if isinstance(configured_budget, int) and configured_budget >= 1:
            return configured_budget

        parent_goal_budget = parent_context.goal.get("max_iterations")
        if isinstance(parent_goal_budget, int) and parent_goal_budget >= 1:
            return parent_goal_budget

        runtime_budget = parent_context.runtime.get("max_iterations")
        if isinstance(runtime_budget, int) and runtime_budget >= 1:
            return runtime_budget

        return 1

    def _initialize_available_contexts(self) -> set[str]:
        return {
            "goal_context",
            "runtime_context",
            "workspace_context",
            "memory_context",
            "tool_context",
            "audit_context",
        }

    def _available_contexts_from_snapshot(self, context: ContextSnapshot) -> set[str]:
        available = self._initialize_available_contexts()
        if self._has_meaningful_context(context.evaluation):
            available.add("evaluation_context")
        if self._has_meaningful_context(context.failure):
            available.add("failure_context")
        return available

    def _has_meaningful_context(self, values: dict[str, object]) -> bool:
        for value in values.values():
            if isinstance(value, dict):
                if any(nested not in (None, {}, [], "") for nested in value.values()):
                    return True
            elif isinstance(value, list):
                if value:
                    return True
            elif value not in (None, ""):
                return True
        return False

    def _refresh_context_from_memory(
        self,
        context: ContextSnapshot,
        *,
        goal_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        snapshot = self.memory.snapshot()
        context.memory["working_snapshot"] = snapshot.get("working_memory", {})
        context.memory["episodic"] = snapshot.get("episodic_memory", {})
        context.memory["semantic"] = snapshot.get("semantic_memory", {})
        context.memory["personal"] = snapshot.get("personal_memory", {})
        artifact_memory = snapshot.get("artifact_memory", {})
        session_memory = snapshot.get("session_memory", {})
        traces = session_memory.get("traces", [])
        transcripts = session_memory.get("transcripts", [])
        filtered_traces = _filter_records_for_run(traces, goal_id=goal_id, run_id=run_id)
        filtered_transcripts = _filter_records_for_run(
            transcripts,
            goal_id=goal_id,
            run_id=run_id,
        )
        context.memory["session"] = _scoped_session_memory(
            session_memory,
            goal_id=goal_id,
            run_id=run_id,
            traces=filtered_traces,
            transcripts=filtered_transcripts,
        )
        context.audit["events"] = filtered_traces[-25:]
        context.audit["artifacts"] = artifact_memory.get("artifacts", [])
        context.audit["transcripts"] = filtered_transcripts[-12:]
        context.audit["decision_trace"] = list(getattr(self, "_decision_trace_buffer", []))[-5:]

    def _merge_context_updates(
        self, context: ContextSnapshot, payload: ToolExecutionPayload
    ) -> None:
        context.runtime.update(
            payload.runtime_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )
        context.workspace.update(
            payload.workspace_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )
        context.evaluation.update(
            payload.evaluation_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )
        context.failure.update(
            payload.failure_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )
        context.memory.update(
            payload.memory_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )
        context.tool.update(
            payload.tool_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )
        context.audit.update(
            payload.audit_updates.model_dump(exclude_none=True, exclude_defaults=True)
        )

    def _merge_seed_context(self, context: ContextSnapshot, seed: ContextSnapshot) -> None:
        workspace_seed = {
            key: value
            for key, value in seed.workspace.items()
            if key
            in {
                "allowed_target_files",
                "changed_files",
                "mutated_files",
                "isolated_workspace",
                "isolated_workspace_path",
                "primary_workdir",
                "scenario_path",
                "source_scenario_path",
                "open_python_repl_sessions",
            }
        }
        audit_seed = {
            key: value
            for key, value in seed.audit.items()
            if key in {"artifacts", "diff_report", "run_summary", "validation_tools", "sync_events"}
        }
        evaluation_seed = dict(seed.evaluation)
        parent_baseline = context.evaluation.get("baseline_summary", {})
        if isinstance(parent_baseline, dict) and any(
            parent_baseline.get(key) not in (None, "") for key in ("score", "status", "tool")
        ):
            evaluation_seed.pop("baseline_summary", None)
        for target, source in (
            (context.workspace, workspace_seed),
            (context.evaluation, evaluation_seed),
            (context.failure, seed.failure),
            (context.audit, audit_seed),
        ):
            target.update(
                {key: value for key, value in source.items() if value not in (None, {}, [], "")}
            )

    def _tool_call_context(
        self, context: ContextSnapshot, *, active_tools: list[str]
    ) -> ToolCallContext:
        raw_failure_buckets = context.failure.get("failure_buckets", {})
        failure_buckets: list[FailureBucketCount]
        if isinstance(raw_failure_buckets, dict):
            failure_buckets = [
                FailureBucketCount(bucket=str(key), count=value)
                for key, value in raw_failure_buckets.items()
                if isinstance(value, int)
            ]
        elif isinstance(raw_failure_buckets, list):
            failure_buckets = [
                FailureBucketCount.model_validate(item)
                for item in raw_failure_buckets
                if isinstance(item, dict)
            ]
        else:
            failure_buckets = []
        clustered_failures = [
            FailureCluster.model_validate(item)
            for item in context.failure.get("clustered_failures", [])
            if isinstance(item, dict)
        ]
        ranked_hypotheses = [
            RankedHypothesis.model_validate(item)
            for item in context.failure.get("ranked_hypotheses", [])
            if isinstance(item, dict)
        ]
        return ToolCallContext(
            runtime=RuntimeToolContext.model_validate(context.runtime),
            workspace=WorkspaceToolContext.model_validate(context.workspace),
            evaluation=EvaluationToolContext(
                baseline_summary=ScoreSummary.model_validate(
                    context.evaluation.get("baseline_summary", {})
                ),
                targeted_validation_summary=ScoreSummary.model_validate(
                    context.evaluation.get("targeted_validation_summary", {})
                ),
                integrity_summary=ScoreSummary.model_validate(
                    context.evaluation.get("integrity_summary", {})
                ),
                last_eval_summary=ScoreSummary.model_validate(
                    context.evaluation.get("last_eval_summary", {})
                ),
                repeat_eval_summary=ScoreSummary.model_validate(
                    context.evaluation.get("repeat_eval_summary", {})
                ),
                recent_eval_summaries=[
                    item
                    for item in context.evaluation.get("recent_eval_summaries", [])
                    if isinstance(item, str)
                ],
                regressions=[
                    item
                    for item in context.evaluation.get("regressions", [])
                    if isinstance(item, str)
                ],
                eval_manifest_path=(
                    value
                    if isinstance((value := context.evaluation.get("eval_manifest_path")), str)
                    and value.strip()
                    else None
                ),
                eval_manifest_hash=(
                    value
                    if isinstance((value := context.evaluation.get("eval_manifest_hash")), str)
                    and value.strip()
                    else None
                ),
                aggregated_score=(
                    float(value)
                    if isinstance(
                        (value := context.evaluation.get("aggregated_score")), int | float
                    )
                    else None
                ),
                acceptance=AcceptanceStatus.model_validate(
                    context.evaluation.get("acceptance", {})
                ),
                verification=VerificationStatus.model_validate(
                    context.evaluation.get("verification", {})
                ),
                claim_audit=ClaimAuditStatus.model_validate(
                    context.evaluation.get("claim_audit", {})
                ),
                phase2_metrics=Phase2Metrics.model_validate(
                    context.evaluation.get("phase2_metrics", {})
                ),
                phase2_gate=Phase2GateStatus.model_validate(
                    context.evaluation.get("phase2_gate", {})
                ),
            ),
            failure=FailureToolContext(
                suspicious_reviews=[
                    item
                    for item in context.failure.get("suspicious_reviews", [])
                    if isinstance(item, str)
                ],
                failure_buckets=failure_buckets,
                explanations=[
                    item
                    for item in context.failure.get("explanations", [])
                    if isinstance(item, str)
                ],
                failure_family=(
                    value
                    if isinstance((value := context.failure.get("failure_family")), str)
                    else None
                ),
                clustered_failures=clustered_failures,
                ranked_hypotheses=ranked_hypotheses,
                concepts=[
                    item for item in context.failure.get("concepts", []) if isinstance(item, str)
                ],
                diagnosis_mode=(
                    value
                    if isinstance((value := context.failure.get("diagnosis_mode")), str)
                    else None
                ),
                top_hypothesis=(
                    value
                    if isinstance((value := context.failure.get("top_hypothesis")), str)
                    else None
                ),
                diagnosed_issues=[
                    item
                    for item in context.failure.get("diagnosed_issues", [])
                    if isinstance(item, str)
                ],
            ),
            memory=MemoryToolContext(
                run_state=RuntimeToolContext.model_validate(context.memory.get("run_state", {})),
                working_memory=(
                    value
                    if isinstance((value := context.memory.get("working_snapshot")), dict)
                    else {}
                ),
                session_memory=(
                    value if isinstance((value := context.memory.get("session")), dict) else {}
                ),
                episodic_memory=(
                    value if isinstance((value := context.memory.get("episodic")), dict) else {}
                ),
                semantic_memory=(
                    value if isinstance((value := context.memory.get("semantic")), dict) else {}
                ),
                personal_memory=(
                    value if isinstance((value := context.memory.get("personal")), dict) else {}
                ),
                working_snapshot_loaded=bool(context.memory.get("working_snapshot")) or None,
                session_loaded=bool(context.memory.get("session")) or None,
                episodic_loaded=bool(context.memory.get("episodic")) or None,
                semantic_loaded=bool(context.memory.get("semantic")) or None,
                personal_loaded=bool(context.memory.get("personal")) or None,
                search_results=[
                    item
                    for item in context.memory.get("search_results", [])
                    if isinstance(item, str)
                ],
                last_memory_entry_written=True if context.memory.get("last_memory_entry") else None,
                episode_recorded=(
                    value
                    if isinstance((value := context.memory.get("episode_recorded")), bool)
                    else None
                ),
                lesson_saved=(
                    value
                    if isinstance((value := context.memory.get("lesson_saved")), bool)
                    else None
                ),
                lessons_consolidated=(
                    value
                    if isinstance((value := context.memory.get("lessons_consolidated")), bool)
                    else None
                ),
                calibration_factor=(
                    float(value)
                    if isinstance((value := context.memory.get("calibration_factor")), int | float)
                    else None
                ),
                skip_hypothesis=(
                    value
                    if isinstance((value := context.memory.get("skip_hypothesis")), bool)
                    else None
                ),
                backfilled=(
                    value if isinstance((value := context.memory.get("backfilled")), bool) else None
                ),
                snapshot_saved=True if context.memory.get("snapshot_state") else None,
                resumed=True if context.memory.get("resumed_state") else None,
            ),
            tool=ToolStateContext(
                active_tools=active_tools,
                available_tool_summaries=[
                    item
                    for item in context.tool.get("available_tool_summaries", [])
                    if isinstance(item, str)
                ],
                tool_contracts=[
                    ToolContractView.model_validate(item)
                    for item in context.tool.get("tool_contracts", [])
                    if isinstance(item, dict)
                ],
                policy_decisions=[
                    item
                    for item in context.tool.get("policy_decisions", [])
                    if isinstance(item, str)
                ],
                registered_mcp_tools=[
                    item
                    for item in context.tool.get("registered_mcp_tools", [])
                    if isinstance(item, str)
                ],
                last_tool=(
                    value if isinstance((value := context.tool.get("last_tool")), str) else None
                ),
                available_tools_snapshot=[
                    item
                    for item in context.tool.get("available_tools_snapshot", [])
                    if isinstance(item, str)
                ],
                resolved_toolset=[
                    item
                    for item in context.tool.get("resolved_toolset", [])
                    if isinstance(item, str)
                ],
                policy_checked=(
                    value
                    if isinstance((value := context.tool.get("policy_checked")), bool)
                    else None
                ),
                context_assembled=(
                    value
                    if isinstance((value := context.tool.get("context_assembled")), bool)
                    else None
                ),
                context_sections=[
                    item
                    for item in context.tool.get("context_sections", [])
                    if isinstance(item, str)
                ],
            ),
            audit=AuditToolContext(
                events=[
                    TraceRecord.model_validate(item)
                    for item in context.audit.get("events", [])
                    if isinstance(item, dict)
                ],
                artifacts=[
                    item for item in context.audit.get("artifacts", []) if isinstance(item, str)
                ],
                transcripts=[
                    TranscriptRecord.model_validate(item)
                    for item in context.audit.get("transcripts", [])
                    if isinstance(item, dict)
                ],
                progress_artifacts=[
                    item
                    for item in context.audit.get("progress_artifacts", [])
                    if isinstance(item, str)
                ],
                artifact_index=[
                    item
                    for item in context.audit.get("artifact_index", [])
                    if isinstance(item, str)
                ],
                validation_tools=[
                    item
                    for item in context.audit.get("validation_tools", [])
                    if isinstance(item, str)
                ],
                iteration_logs=[
                    item
                    for item in context.audit.get("iteration_logs", [])
                    if isinstance(item, str)
                ],
                decision_trace=[
                    item
                    for item in context.audit.get("decision_trace", [])
                    if isinstance(item, str)
                ],
                sync_events=[
                    item for item in context.audit.get("sync_events", []) if isinstance(item, str)
                ],
                commit_events=[
                    item for item in context.audit.get("commit_events", []) if isinstance(item, str)
                ],
                agent_invocations=[
                    item
                    for item in context.audit.get("agent_invocations", [])
                    if isinstance(item, str)
                ],
                llm_calls=[
                    item for item in context.audit.get("llm_calls", []) if isinstance(item, str)
                ],
                deterministic_tool_calls=[
                    item
                    for item in context.audit.get("deterministic_tool_calls", [])
                    if isinstance(item, str)
                ],
                tool_usage=[
                    item for item in context.audit.get("tool_usage", []) if isinstance(item, str)
                ],
                delegation_requests=[
                    item
                    for item in context.audit.get("delegation_requests", [])
                    if isinstance(item, str)
                ],
                run_summary=(
                    value if isinstance((value := context.audit.get("run_summary")), dict) else {}
                ),
                diff_report=(
                    value if isinstance((value := context.audit.get("diff_report")), dict) else {}
                ),
                run_result=AuditRunResult.model_validate(context.audit.get("run_result", {})),
            ),
        )

    def _update_available_contexts(
        self, available_contexts: set[str], tool: ToolSpec, matching_skills: list[SkillSpec]
    ) -> None:
        available_contexts.update(tool.produces_contexts)
        for skill in matching_skills:
            available_contexts.update(skill.output_contexts)

    def _persist_and_dream(self, execution: ThinRuntimeExecutionResult) -> None:
        self._stage_execution_artifacts_for_memory(execution)
        snapshot = self.memory.snapshot()
        execution.memory_snapshot = snapshot
        session_memory = snapshot.get("session_memory", {})
        dream_queue = session_memory.get("dream_queue", [])
        if isinstance(dream_queue, list):
            staged_items = cast(list[object], dream_queue)
            self.persistence.save_dream_queue(
                [cast(dict[str, object], item) for item in staged_items if isinstance(item, dict)]
            )
        self.persistence.save_session_snapshot(snapshot)
        dream_payload: dict[str, object] = {
            "run_id": execution.run_id,
            "applied": 0,
            "scopes": [],
            "error": None,
        }
        try:
            dream_result = self.dream_state.run()
            dream_payload.update(dream_result)
        except Exception as exc:
            dream_payload["error"] = str(exc)
        finally:
            self.hooks.emit("after_dream_state", dream_payload)
            execution.memory_snapshot = self.memory.snapshot()
            execution.emitted_hooks = self.hooks.emitted()

    def _stage_execution_artifacts_for_memory(self, execution: ThinRuntimeExecutionResult) -> None:
        artifact_paths = [
            execution.artifact_dir,
            str(self.persistence.execution_file(execution.run_id)),
            str(self.persistence.summary_file(execution.run_id)),
        ]
        audit_artifacts = execution.context.audit.setdefault("artifacts", [])
        if isinstance(audit_artifacts, list):
            typed_audit_artifacts = cast(list[object], audit_artifacts)
            for path in artifact_paths:
                if path not in typed_audit_artifacts:
                    typed_audit_artifacts.append(path)

        for path in artifact_paths:
            self._append_unique_deferred_memory(
                scope_name="artifact_memory",
                key="artifacts",
                value=path,
            )

        run_status = "succeeded" if execution.success else "failed"
        run_episode = (
            f"Thin runtime run {execution.run_id} {run_status}. "
            f"Artifacts stored at {execution.artifact_dir}."
        )
        self._append_unique_deferred_memory(
            scope_name="episodic_memory",
            key="episodes",
            value=run_episode,
        )

        for result in execution.tool_results:
            run_result = result.payload.audit_updates.run_result
            if not run_result.run_id:
                continue
            score_fragment = (
                f" with score {run_result.aggregate_score:.2f}"
                if isinstance(run_result.aggregate_score, int | float)
                else ""
            )
            outcome_fragment = (
                "passed"
                if run_result.aggregate_passed is True
                else "failed"
                if run_result.aggregate_passed is False
                else "completed"
            )
            tool_episode = f"Tool {result.tool_name} observed run {run_result.run_id}{score_fragment} and {outcome_fragment}."
            self._append_unique_deferred_memory(
                scope_name="episodic_memory",
                key="episodes",
                value=tool_episode,
            )

    def _append_unique_deferred_memory(self, *, scope_name: str, key: str, value: object) -> None:
        session_memory = self.memory.read_scope("session_memory")
        raw_queue = session_memory.get("dream_queue", [])
        queue = cast(list[object], raw_queue) if isinstance(raw_queue, list) else []
        entry = {"scope": scope_name, "key": key, "value": value}
        if entry in queue:
            return
        self.memory.append("session_memory", "dream_queue", entry)


def _filter_records_for_run(
    records: object,
    *,
    goal_id: str | None,
    run_id: str | None,
) -> list[dict[str, object]]:
    if not isinstance(records, list):
        return []
    filtered: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if run_id is not None:
            if item.get("run_id") == run_id:
                filtered.append(item)
            continue
        if goal_id is not None and item.get("goal_id") == goal_id:
            filtered.append(item)
    return filtered


def _scoped_session_memory(
    session_memory: object,
    *,
    goal_id: str | None,
    run_id: str | None,
    traces: list[dict[str, object]],
    transcripts: list[dict[str, object]],
) -> dict[str, object]:
    if not isinstance(session_memory, dict):
        return {"traces": traces[-25:], "transcripts": transcripts[-12:]}
    scoped = {
        key: value
        for key, value in session_memory.items()
        if key not in {"traces", "transcripts", "delegations", "retries", "validations"}
    }
    scoped["traces"] = traces[-25:]
    scoped["transcripts"] = transcripts[-12:]
    for key in ("delegations", "retries", "validations"):
        scoped[key] = _filter_records_for_run(
            session_memory.get(key, []),
            goal_id=goal_id,
            run_id=run_id,
        )
    return scoped
