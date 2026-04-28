from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence

# type-hygiene: skip-file  # dynamic DK/native bridge types are intentionally structural
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from dawn_kestrel.agent.loop import run_agent
from dawn_kestrel.agent.types import AgentResult, LoopConfig
from dawn_kestrel.provider.llm_client import LLMClient, LLMResponse
from dawn_kestrel.provider.provider_types import TokenUsage
from dawn_kestrel.skills import SkillRegistry as DkSkillRegistry
from dawn_kestrel.tools.framework import ToolContext as DkToolContext
from dawn_kestrel.tools.framework import ToolResult as DkToolResult

from ash_hawk.dawn_kestrel_skills import prepare_skill_runtime
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ModelMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)
from ash_hawk.thin_runtime.agent_text import build_agent_text, build_live_checkpoint
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
from ash_hawk.thin_runtime.planner import PlannerDecision, build_tool_contract_view
from ash_hawk.thin_runtime.runner import AgenticLoopRunner

_RUNTIME_MANAGED_INPUTS = {
    "goal_id",
    "remaining_tools",
    "available_contexts",
    "agent_text",
    "iterations",
    "tool_call_count",
    "max_iterations",
    "retry_count",
    "context",
}

ToolDef = dict[str, Any]
ToolDefList = list[ToolDef]


@dataclass
class _NativeRunState:
    goal: RuntimeGoal
    agent: AgentSpec
    active_skills: list[SkillSpec]
    candidate_skills: list[SkillSpec]
    context: ContextSnapshot
    available_contexts: set[str]
    depth: int = 0
    selected_tool_names: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    delegations: list[DelegationRecord] = field(default_factory=list)
    touched_skill_reads: set[str] = field(default_factory=set)
    tool_execution_order: list[str] | None = None
    requested_tools: list[str] | None = None
    remaining_tools: list[str] = field(default_factory=list)
    terminated: bool = False
    terminal_success: bool = True
    terminal_reason: str | None = None


class _ThinRuntimeClient:
    def __init__(
        self,
        *,
        base_client: LLMClient | None,
        runner: DkNativeLoopRunner,
        state: _NativeRunState,
        registry: _DynamicThinRuntimeToolRegistry,
    ) -> None:
        self._base_client = base_client
        self._runner = runner
        self._state = state
        self._registry = registry
        self.provider_id = getattr(base_client, "provider_id", "thin-runtime")

    async def complete(
        self,
        messages: list[ToolDef],
        tools: ToolDefList | None = None,
        options: Any = None,
    ) -> LLMResponse:
        del options

        if self._state.terminated:
            return LLMResponse(
                text=self._state.terminal_reason or "Thin runtime execution terminated.",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )

        dynamic_tools = self._merge_dynamic_tools(tools)

        if self._state.tool_execution_order is not None:
            explicit_tools = self._registry.list_tools()
            response = self._explicit_order_response(explicit_tools)
            self._record_decision(response, explicit_tools, source="explicit_order")
            return response

        if self._base_client is None:
            response = LLMResponse(
                text="No model client is available for tool selection.",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )
            return response

        response = await self._base_client.complete(messages, tools=dynamic_tools)
        self._record_decision(response, dynamic_tools, source="dk_tool_calls")
        return response

    def _explicit_order_response(self, tools: ToolDefList) -> LLMResponse:
        if not tools:
            return LLMResponse(
                text=self._state.terminal_reason or "No tools available.",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )
        function = tools[0].get("function", {})
        tool_name = function.get("name", "") if isinstance(function, dict) else ""
        tool_call = {
            "id": f"explicit-{len(self._state.selected_tool_names) + 1}",
            "type": "function",
            "function": {"name": tool_name, "arguments": "{}"},
            "tool": tool_name,
            "input": {},
        }
        return LLMResponse(
            text="Selected from explicit tool execution order override.",
            usage=TokenUsage(input=0, output=0, reasoning=0),
            finish_reason="tool_use",
            cost=Decimal("0"),
            tool_calls=[tool_call],
        )

    def _record_decision(
        self,
        response: LLMResponse,
        tool_defs: ToolDefList,
        *,
        source: str,
    ) -> None:
        tool_calls = response.tool_calls or []
        if not tool_calls:
            return
        tool_name = _tool_name_from_call(tool_calls[0])
        response_text = response.text.strip()
        rationale = (
            response_text or f"Model emitted tool call(s): {', '.join(_tool_names(tool_calls))}"
        )
        decision = PlannerDecision(
            selected_tool=tool_name,
            source=source,
            rationale=rationale,
            considered_tools=_tool_names_from_defs(tool_defs),
            reason_model_authored=(source == "dk_tool_calls" and bool(response_text)),
        )
        self._runner.record_native_tool_decision(
            goal=self._state.goal,
            agent=self._state.agent,
            decision=decision,
        )
        self._state.context.runtime["last_decision"] = rationale
        if tool_name is not None:
            self._state.context.runtime["preferred_tool"] = tool_name

    def _merge_dynamic_tools(self, additional_tools: ToolDefList | None) -> ToolDefList:
        dynamic_tools = list(self._registry.list_tools())
        existing_names = set(_tool_names_from_defs(dynamic_tools))
        for tool_def in additional_tools or []:
            name = _tool_name_from_def(tool_def)
            if name is None or name in existing_names or name in self._registry.get_all():
                continue
            dynamic_tools.append(tool_def)
            existing_names.add(name)
        return dynamic_tools


class _ThinRuntimeDkTool:
    def __init__(
        self,
        *,
        spec: ToolSpec,
        runner: DkNativeLoopRunner,
        state: _NativeRunState,
        registry: _DynamicThinRuntimeToolRegistry,
    ) -> None:
        self.id = spec.name
        self.description = spec.summary or spec.description
        self._spec = spec
        self._runner = runner
        self._state = state
        self._registry = registry

    def parameters(self) -> dict[str, object]:
        properties: dict[str, object] = {}
        required: list[str] = []
        for input_field in self._spec.inputs.properties:
            if input_field.name in _RUNTIME_MANAGED_INPUTS:
                continue
            property_schema: dict[str, object] = {
                "type": input_field.type.value,
                "description": input_field.description,
            }
            if input_field.type.value == "array" and input_field.item_type is not None:
                property_schema["items"] = {"type": input_field.item_type.value}
            properties[input_field.name] = property_schema
            if input_field.required:
                required.append(input_field.name)
        schema: dict[str, object] = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        return schema

    async def execute(self, args: dict[str, Any], ctx: DkToolContext) -> DkToolResult:
        return await asyncio.to_thread(self._execute_sync, args, ctx)

    def _execute_sync(self, args: dict[str, Any], ctx: DkToolContext) -> DkToolResult:
        if self._state.terminated:
            return DkToolResult(
                output=self._state.terminal_reason or "Execution already terminated.",
                error=self._state.terminal_reason,
            )

        tool_name = self.id
        candidate_tool_names = [tool.name for tool in self._registry.current_candidate_tools()]
        if tool_name not in candidate_tool_names:
            return DkToolResult(
                output=f"Tool {tool_name} is not available for the current thin-runtime state.",
                error=f"tool_unavailable: {tool_name}",
            )

        if (
            self._state.tool_execution_order is not None
            and tool_name in self._state.remaining_tools
        ):
            self._state.remaining_tools.remove(tool_name)
        self._state.selected_tool_names.append(tool_name)

        matching_skills = self._runner.matching_skills(
            self._state.candidate_skills,
            tool_name,
            self._state.available_contexts,
        )
        if matching_skills:
            self._state.active_skills = self._runner.activate_skills(
                active_skills=self._state.active_skills,
                candidate_skills=self._state.candidate_skills,
                requested_skill_names=[skill.name for skill in matching_skills],
            )
            self._state.context.runtime["active_skills"] = [
                skill.name for skill in self._state.active_skills
            ]
        self._runner.read_matching_skill_memory(
            self._state.goal,
            self._state.agent,
            matching_skills,
            self._state.touched_skill_reads,
        )

        result = self._runner.invoke_tool(
            goal=self._state.goal,
            agent=self._state.agent,
            tool_name=tool_name,
            matching_skills=matching_skills,
            context=self._state.context,
            available_contexts=self._state.available_contexts,
            remaining_tools=self._state.remaining_tools,
            selected_tool_names=self._state.selected_tool_names,
            tool_args=args,
        )
        self._state.tool_results.append(result)
        return self._runner.apply_dk_tool_result(
            state=self._state,
            tool_name=tool_name,
            matching_skills=matching_skills,
            result=result,
        )


class _DynamicThinRuntimeToolRegistry:
    def __init__(
        self,
        *,
        runner: DkNativeLoopRunner,
        state: _NativeRunState,
        tools: list[ToolSpec],
    ) -> None:
        self._runner = runner
        self._state = state
        self._tools: dict[str, _ThinRuntimeDkTool] = {}
        self.tool_metadata: dict[str, ToolDef] = {}
        for tool in tools:
            self.register(_ThinRuntimeDkTool(spec=tool, runner=runner, state=state, registry=self))

    def register(self, tool: _ThinRuntimeDkTool) -> None:
        self._tools[tool.id] = tool

    def current_candidate_tools(self) -> list[ToolSpec]:
        if self._state.terminated:
            return []

        tool_budget_error = self._runner.check_tool_budget(
            self._state.goal,
            self._state.agent,
            self._state.selected_tool_names,
        )
        if tool_budget_error is not None:
            self._terminate(tool_budget_error, success=False)
            return []

        stop_error = self._runner.check_iteration_stop(
            self._state.goal,
            self._state.agent,
            self._state.selected_tool_names,
        )
        if stop_error is not None:
            self._terminate(stop_error, success=False)
            return []

        active_tools = self._runner.resolve_active_tools_for_runtime(
            self._state.agent,
            self._state.active_skills,
            self._state.candidate_skills,
            requested_tools=self._state.requested_tools,
        )
        self._state.context.tool["active_tools"] = [tool.name for tool in active_tools]
        self._state.context.tool["available_tools_snapshot"] = [tool.name for tool in active_tools]
        self._state.context.tool["resolved_toolset"] = [tool.name for tool in active_tools]
        self._state.context.tool["tool_contracts"] = [
            build_tool_contract_view(tool).model_dump(exclude_none=True) for tool in active_tools
        ]
        tool_surface = active_tools
        if self._state.tool_execution_order is not None:
            active_by_name = {tool.name: tool for tool in active_tools}
            tool_surface = []
            for tool_name in self._state.remaining_tools:
                tool = active_by_name.get(tool_name)
                if tool is None and self._state.requested_tools is None:
                    resolved = self._runner.tools.resolve_allowed([tool_name], self._runner.policy)
                    if resolved:
                        tool = resolved[0]
                if tool is not None:
                    tool_surface.append(tool)
        candidate_tools = [
            tool
            for tool in tool_surface
            if self._runner.tool_preconditions_met(tool.name, self._state.context)
        ]
        if candidate_tools:
            return candidate_tools
        reason, success = self._no_candidate_outcome()
        self._terminate(reason, success=success)
        return []

    def list_tools(self) -> ToolDefList:
        tool_defs: ToolDefList = []
        for tool in self.current_candidate_tools():
            registered = self._tools.get(tool.name)
            if registered is None:
                continue
            tool_defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": registered.id,
                        "description": registered.description,
                        "parameters": registered.parameters(),
                    },
                }
            )
        return tool_defs

    def get_all(self) -> dict[str, _ThinRuntimeDkTool]:
        return dict(self._tools)

    async def execute(self, tool_call: ToolDef, ctx: DkToolContext | None) -> DkToolResult:
        resolved_ctx = DkToolContext(
            session_id=ctx.session_id if ctx is not None else self._state.goal.goal_id,
            working_dir=self._runner.workdir,
            metadata={
                "goal_id": self._state.goal.goal_id,
                "agent": self._state.agent.name,
            },
        )
        tool_name = _tool_name_from_call(tool_call)
        if tool_name is None:
            return DkToolResult(output="Unknown tool", error="unknown_tool")
        tool = self._tools.get(tool_name)
        if tool is None:
            return DkToolResult(
                output=f"Unknown tool: {tool_name}", error=f"unknown_tool: {tool_name}"
            )
        raw_input = tool_call.get("input", tool_call.get("arguments", {}))
        tool_input = raw_input if isinstance(raw_input, dict) else {}
        return await tool.execute(
            tool_input,
            resolved_ctx,
        )

    def _no_candidate_outcome(self) -> tuple[str, bool]:
        failure_family = self._state.context.failure.get("failure_family")
        completed_loops = self._runner.current_iteration_count(
            self._state.agent,
            self._state.selected_tool_names,
        )
        mutated_files = self._state.context.workspace.get("mutated_files", [])
        if self._state.agent.iteration_budget_mode == "loop" and (
            isinstance(failure_family, str)
            and failure_family.strip()
            or completed_loops == 0
            or not isinstance(mutated_files, list)
            or not mutated_files
        ):
            return (
                "No eligible tools available for a complete diagnosis-mutation-re-evaluation loop",
                False,
            )
        if self._state.selected_tool_names:
            return "No additional eligible tools available", True
        return "No eligible tools available for current contexts", False

    def _terminate(self, reason: str, *, success: bool) -> None:
        if self._state.terminated:
            return
        self._state.terminated = True
        self._state.terminal_success = success
        self._state.terminal_reason = reason
        self._runner.hooks.emit(
            "on_stop_condition",
            {
                "goal_id": self._state.goal.goal_id,
                "agent": self._state.agent.name,
                "reason": reason,
            },
        )


class DkNativeLoopRunner(AgenticLoopRunner):
    def __init__(
        self,
        *args: Any,
        client_factory: Any | None = None,
        dk_skill_registry: DkSkillRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._client_factory = client_factory
        self._client: LLMClient | None = None
        self._dk_skill_registry = dk_skill_registry

    def set_client_factory(self, factory: Any) -> None:
        self._client_factory = factory
        self._client = None

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
        self._current_native_run_id = run_id
        self._record_agent_event(
            goal=goal,
            agent=agent,
            event_type="enter",
            success=True,
            run_id=run_id,
        )
        self._read_agent_memory(agent)
        self._refresh_context_from_memory(context, goal_id=goal.goal_id, run_id=run_id)
        available_contexts_raw = context.runtime.get("available_contexts")
        if isinstance(available_contexts_raw, list) and available_contexts_raw:
            available_contexts = {item for item in available_contexts_raw if isinstance(item, str)}
        else:
            available_contexts = self._available_contexts_from_snapshot(context)
        self._update_iteration_state(context, goal, agent, [])
        self._refresh_runtime_view(
            goal=goal,
            agent=agent,
            skills=active_skills,
            tools=self._resolve_active_tools(agent, active_skills, candidate_skills),
            context=context,
            available_skills=candidate_skills,
            include_skill_instructions=False,
        )
        context.runtime["available_contexts"] = sorted(available_contexts)

        state = _NativeRunState(
            goal=goal,
            agent=agent,
            active_skills=active_skills,
            candidate_skills=candidate_skills,
            context=context,
            available_contexts=available_contexts,
            depth=depth,
            requested_tools=list(requested_tools) if requested_tools else None,
            tool_execution_order=list(tool_execution_order) if tool_execution_order else None,
            remaining_tools=list(tool_execution_order or []),
        )
        registry = _DynamicThinRuntimeToolRegistry(
            runner=self,
            state=state,
            tools=self.tools.list_tools(),
        )
        base_client = None if tool_execution_order is not None else self._build_client()
        client = _ThinRuntimeClient(
            base_client=base_client, runner=self, state=state, registry=registry
        )
        loop_config = LoopConfig(max_iterations=self._loop_iterations(goal, active_tools))

        async def _run_with_skills() -> AgentResult:
            skill_runtime, preloaded_messages = await prepare_skill_runtime(
                registry=self._dk_skill_registry,
                preactivate=[skill.name for skill in active_skills],
            )
            if skill_runtime is not None:
                for skill_name in skill_runtime.activated_names():
                    self._emit_skill_activation(
                        goal=goal,
                        agent=agent,
                        skill_name=skill_name,
                        source="preloaded",
                        available=True,
                        message="skill activated",
                    )

            def _on_dk_event(event: object) -> None:
                event_type = getattr(event, "event_type", "")
                if event_type != "skill_activation":
                    return
                skill_name = str(getattr(event, "skill_name", "")).strip()
                if not skill_name:
                    return
                self._emit_skill_activation(
                    goal=goal,
                    agent=agent,
                    skill_name=skill_name,
                    source="dawn-kestrel",
                    available=bool(getattr(event, "available", False)),
                    message=str(getattr(event, "message", "")).strip() or "skill activated",
                )

            loop_config.on_event = _on_dk_event
            messages = [
                *preloaded_messages,
                *self._initial_messages(goal, agent, active_skills, context),
            ]
            return await run_agent(
                client=cast(Any, client),
                messages=messages,
                tools=cast(Any, registry),
                config=loop_config,
                skills=skill_runtime,
            )

        result = asyncio.run(_run_with_skills())
        execution = self._finalize_run(
            run_id=run_id,
            state=state,
            active_tools=active_tools,
            result=result,
        )
        self._persist_and_dream(execution)
        self.persistence.persist_execution(execution)
        return execution

    def _emit_skill_activation(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        skill_name: str,
        source: str,
        available: bool,
        message: str,
    ) -> None:
        self.hooks.emit(
            "before_skill",
            {
                "goal_id": goal.goal_id,
                "agent": agent.name,
                "skill": skill_name,
                "source": source,
                "available": available,
                "message": message,
            },
        )

    def _build_client(self) -> LLMClient:
        if self._client_factory is not None:
            return self._client_factory()
        if self._client is not None:
            return self._client
        from dawn_kestrel.base.config import get_config_api_key, load_agent_config

        config = load_agent_config()
        provider = (
            config.get("runtime.provider") or os.environ.get("DAWN_KESTREL_PROVIDER") or "anthropic"
        )
        model = (
            config.get("runtime.model")
            or os.environ.get("DAWN_KESTREL_MODEL")
            or "claude-sonnet-4-20250514"
        )
        api_key = get_config_api_key(provider) or None
        timeout_value = os.getenv("ASH_HAWK_LLM_TIMEOUT_SECONDS")
        timeout_seconds = float(timeout_value) if timeout_value is not None else 600.0
        self._client = LLMClient(
            provider_id=provider,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return self._client

    def record_native_tool_decision(
        self,
        *,
        goal: RuntimeGoal,
        agent: AgentSpec,
        decision: PlannerDecision,
    ) -> None:
        self._record_tool_selection_decision(
            goal=goal,
            agent=agent,
            decision=decision,
            run_id=(
                value
                if isinstance((value := getattr(self, "_current_native_run_id", None)), str)
                else None
            ),
        )

    def matching_skills(
        self,
        active_skills: list[SkillSpec],
        tool_name: str,
        available_contexts: set[str],
    ) -> list[SkillSpec]:
        return self._matching_skills(active_skills, tool_name, available_contexts)

    def activate_skills(
        self,
        *,
        active_skills: list[SkillSpec],
        candidate_skills: list[SkillSpec],
        requested_skill_names: list[str],
    ) -> list[SkillSpec]:
        return self._activate_skills(
            active_skills=active_skills,
            candidate_skills=candidate_skills,
            requested_skill_names=requested_skill_names,
        )

    def read_matching_skill_memory(
        self,
        goal: RuntimeGoal,
        agent: AgentSpec,
        matching_skills: list[SkillSpec],
        touched_skill_reads: set[str],
    ) -> None:
        self._read_matching_skill_memory(
            goal,
            agent,
            matching_skills,
            touched_skill_reads,
            run_id=self._current_run_id(),
        )

    def invoke_tool(
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
        tool_args: dict[str, Any] | None = None,
    ) -> ToolResult:
        return self._invoke_tool(
            goal=goal,
            agent=agent,
            tool_name=tool_name,
            matching_skills=matching_skills,
            context=context,
            available_contexts=available_contexts,
            remaining_tools=remaining_tools,
            selected_tool_names=selected_tool_names,
            tool_args=tool_args,
        )

    def apply_dk_tool_result(
        self,
        *,
        state: _NativeRunState,
        tool_name: str,
        matching_skills: list[SkillSpec],
        result: ToolResult,
    ) -> DkToolResult:
        self._after_tool(
            goal=state.goal,
            agent=state.agent,
            tool_name=tool_name,
            matching_skills=matching_skills,
            result=result,
            context=state.context,
            available_contexts=state.available_contexts,
        )

        def checkpoint_result() -> DkToolResult:
            return DkToolResult(
                output=self._checkpoint_output(
                    tool_name=tool_name,
                    result=result,
                    context=state.context,
                ),
                error=result.error,
                metadata={"tool_name": result.tool_name, "success": result.success},
            )

        if not result.success:
            state.terminated = True
            state.terminal_success = False
            state.terminal_reason = result.error
            self._refresh_native_runtime_view(state)
            return checkpoint_result()

        delegation_record, delegated_context = self._execute_delegation(
            result,
            context=state.context,
            depth=state.depth,
        )
        if delegation_record is not None:
            state.delegations.append(delegation_record)
            state.context.audit.setdefault("delegations", []).append(delegation_record.model_dump())
        if delegated_context is not None:
            self._merge_seed_context(state.context, delegated_context)
        state.available_contexts = self._available_contexts_from_snapshot(state.context)
        state.context.runtime["available_contexts"] = sorted(state.available_contexts)
        self._update_iteration_state(
            state.context,
            state.goal,
            state.agent,
            state.selected_tool_names,
        )
        self._refresh_native_runtime_view(state)

        if result.payload.stop:
            stop_reason = result.payload.runtime_updates.stop_reason or "Tool requested stop"
            state.terminated = True
            state.terminal_success = True
            state.terminal_reason = stop_reason
            self.hooks.emit(
                "on_stop_condition",
                {
                    "goal_id": state.goal.goal_id,
                    "agent": state.agent.name,
                    "reason": stop_reason,
                },
            )
            return checkpoint_result()

        write_error = self._write_matching_skill_memory(
            state.agent,
            matching_skills,
            tool_name,
            len(state.tool_results),
        )
        if write_error is not None:
            state.terminated = True
            state.terminal_success = False
            state.terminal_reason = write_error
            return DkToolResult(output=write_error, error=write_error)

        if self._should_reset_tool_cycle(
            goal=state.goal,
            agent=state.agent,
            tool_name=tool_name,
            completion_tools=set(state.agent.iteration_completion_tools),
            selected_tool_names=state.selected_tool_names,
        ):
            state.remaining_tools = list(state.tool_execution_order or [])

        return checkpoint_result()

    def check_tool_budget(
        self,
        goal: RuntimeGoal,
        agent: AgentSpec,
        selected_tool_names: list[str],
    ) -> str | None:
        return self._check_tool_budget(goal, agent, selected_tool_names)

    def check_iteration_stop(
        self,
        goal: RuntimeGoal,
        agent: AgentSpec,
        selected_tool_names: list[str],
    ) -> str | None:
        return self._check_iteration_stop(goal, agent, selected_tool_names)

    def resolve_active_tools_for_runtime(
        self,
        agent: AgentSpec,
        active_skills: list[SkillSpec],
        candidate_skills: list[SkillSpec] | None = None,
        requested_tools: list[str] | None = None,
    ) -> list[ToolSpec]:
        return self._resolve_active_tools(
            agent,
            active_skills,
            candidate_skills,
            requested_tools=requested_tools,
        )

    def tool_preconditions_met(self, tool_name: str, context: ContextSnapshot) -> bool:
        return self._tool_preconditions_met(tool_name, context)

    def current_iteration_count(
        self,
        agent: AgentSpec,
        selected_tool_names: list[str],
    ) -> int:
        return self._current_iteration_count(agent, selected_tool_names)

    def _refresh_native_runtime_view(self, state: _NativeRunState) -> None:
        tools = self._resolve_active_tools(
            state.agent,
            state.active_skills,
            state.candidate_skills,
            requested_tools=state.requested_tools,
        )
        self._refresh_runtime_view(
            goal=state.goal,
            agent=state.agent,
            skills=state.active_skills,
            tools=tools,
            context=state.context,
            available_skills=state.candidate_skills,
            include_skill_instructions=False,
        )

    def _checkpoint_output(
        self,
        *,
        tool_name: str,
        result: ToolResult,
        context: ContextSnapshot,
    ) -> str:
        primary_output = result.payload.message.strip() or result.error or f"{tool_name} completed"
        checkpoint = build_live_checkpoint(tool_name, context)
        return f"{primary_output}\n\n{checkpoint}".strip()

    def _current_run_id(self) -> str | None:
        value = getattr(self, "_current_native_run_id", None)
        return value if isinstance(value, str) else None

    def _initial_messages(
        self,
        goal: RuntimeGoal,
        agent: AgentSpec,
        active_skills: list[SkillSpec],
        context: ContextSnapshot,
    ) -> list[ToolDef]:
        system_text = build_agent_text(
            goal,
            agent,
            active_skills,
            context_snapshot=context,
            memory_snapshot=self.memory.snapshot(),
            include_skill_instructions=False,
        )
        user_text = (
            "Use the available registered tools directly. Do not invent tools or ask for a planner. "
            "Call tools when you need more information or need to change state. When the task is complete, "
            "respond normally with no tool calls. The tool API already describes the registered tools, so "
            "do not restate tool contracts in your reasoning. Use the protected skill content for stable "
            "instructions and use the live context updates after each tool call as your working brief."
        )
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

    def _loop_iterations(self, goal: RuntimeGoal, active_tools: list[ToolSpec]) -> int:
        tool_count = max(len(active_tools), 1)
        return max(goal.max_iterations * max(tool_count, 3), 12)

    def _finalize_run(
        self,
        *,
        run_id: str,
        state: _NativeRunState,
        active_tools: list[ToolSpec],
        result: AgentResult,
    ) -> ThinRuntimeExecutionResult:
        if result.session is not None:
            state.context.audit["dk_session_messages"] = list(result.session.messages)
            state.context.audit["dk_session_events"] = result.session.transcript()
            state.context.audit["trace_events"] = [
                event.model_dump()
                for event in self._dk_trace_events(result.session.messages, result.session.events)
            ]

        success = True
        error: str | None = None
        if state.terminated:
            success = state.terminal_success
            error = None if state.terminal_success else state.terminal_reason
        elif result.error is not None:
            success = False
            error = result.error
        elif result.truncated:
            success = False
            error = f"Reached max iterations: {state.goal.max_iterations}"
            self.hooks.emit(
                "on_stop_condition",
                {"goal_id": state.goal.goal_id, "agent": state.agent.name, "reason": error},
            )
        elif (
            state.agent.iteration_budget_mode == "loop"
            and "improvement-loop" in [skill.name for skill in state.active_skills]
            and self._current_iteration_count(state.agent, state.selected_tool_names) == 0
        ):
            success = False
            error = "No completed improvement loops recorded"
            self.hooks.emit(
                "on_stop_condition",
                {"goal_id": state.goal.goal_id, "agent": state.agent.name, "reason": error},
            )

        self._refresh_runtime_view(
            goal=state.goal,
            agent=state.agent,
            skills=state.active_skills,
            tools=active_tools,
            context=state.context,
            available_skills=state.candidate_skills,
            include_skill_instructions=False,
        )

        self.hooks.emit(
            "after_agent",
            {"goal_id": state.goal.goal_id, "agent": state.agent.name, "success": success},
        )
        self._record_agent_event(
            goal=state.goal,
            agent=state.agent,
            event_type="exit",
            success=success,
            error=error,
            run_id=(
                value if isinstance((value := state.context.runtime.get("run_id")), str) else None
            ),
        )
        self.hooks.emit(
            "after_run",
            {
                "goal_id": state.goal.goal_id,
                "agent": state.agent.name,
                "tool_count": len(state.tool_results),
            },
        )
        return ThinRuntimeExecutionResult(
            run_id=run_id,
            goal=state.goal,
            agent=state.agent,
            active_skills=state.active_skills,
            tools=active_tools,
            context=state.context,
            tool_results=state.tool_results,
            emitted_hooks=self.hooks.emitted(),
            delegations=state.delegations,
            memory_snapshot=self.memory.snapshot(),
            artifact_dir=str(self.persistence.run_dir(run_id)),
            available_contexts=sorted(state.available_contexts),
            selected_tool_names=state.selected_tool_names,
            success=success,
            error=error,
        )

    def _dk_trace_events(self, messages: ToolDefList, events: Sequence[object]) -> list[TraceEvent]:
        trace_events: list[TraceEvent] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str):
                trace_events.append(
                    ModelMessageEvent.create(DEFAULT_TRACE_TS, {"role": role, "content": content})
                )
        for event in events:
            event_type = getattr(event, "event_type", "")
            if event_type == "tool_call":
                tool_name = getattr(event, "tool_name", "")
                tool_input = getattr(event, "tool_input", {})
                trace_events.append(
                    ToolCallEvent.create(
                        DEFAULT_TRACE_TS,
                        {"name": tool_name, "arguments": tool_input},
                    )
                )
                trace_events.append(
                    ToolResultEvent.create(
                        DEFAULT_TRACE_TS,
                        {
                            "tool_name": tool_name,
                            "result": getattr(event, "tool_result", ""),
                            "error": getattr(event, "error", None),
                        },
                    )
                )
        return trace_events


def _tool_name_from_call(tool_call: ToolDef) -> str | None:
    tool_name = tool_call.get("tool") or tool_call.get("name")
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return None


def _tool_name_from_def(tool_def: ToolDef) -> str | None:
    function = tool_def.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return None


def _tool_names(tool_calls: ToolDefList) -> list[str]:
    names: list[str] = []
    for tool_call in tool_calls:
        tool_name = _tool_name_from_call(tool_call)
        if tool_name is not None:
            names.append(tool_name)
    return names


def _tool_names_from_defs(tool_defs: ToolDefList) -> list[str]:
    names: list[str] = []
    for tool_def in tool_defs:
        tool_name = _tool_name_from_def(tool_def)
        if tool_name is not None:
            names.append(tool_name)
    return names


def _dk_tool_result_from_thin_result(result: ToolResult) -> DkToolResult:
    payload = result.payload
    output = payload.message or ""
    if not output:
        output = result.error or f"{result.tool_name} completed"
    return DkToolResult(
        output=output,
        error=result.error,
        metadata={"tool_name": result.tool_name, "success": result.success},
    )
