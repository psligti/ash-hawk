from __future__ import annotations

import pydantic as pd

from ash_hawk.thin_runtime.models import (
    AgentSpec,
    ContextSnapshot,
    RuntimeGoal,
    SkillSpec,
    ToolSpec,
)


class ToolSelectionDecision(pd.BaseModel):
    selected_tool: str | None = None
    source: str
    rationale: str
    considered_tools: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


def select_tool_via_policy(
    *,
    goal: RuntimeGoal,
    agent: AgentSpec,
    active_skills: list[SkillSpec],
    eligible_tools: list[ToolSpec],
    context: ContextSnapshot,
    tool_execution_order: list[str] | None,
) -> ToolSelectionDecision:
    del goal
    del agent
    del active_skills

    eligible_names = [tool.name for tool in eligible_tools]
    if not eligible_names:
        return ToolSelectionDecision(
            selected_tool=None,
            source="no_eligible_tools",
            rationale="No eligible tools available for the current context.",
            considered_tools=[],
        )

    if tool_execution_order is not None:
        for tool_name in tool_execution_order:
            if tool_name in eligible_names:
                return ToolSelectionDecision(
                    selected_tool=tool_name,
                    source="explicit_order",
                    rationale="Selected from explicit tool execution order override.",
                    considered_tools=eligible_names,
                )
        return ToolSelectionDecision(
            selected_tool=None,
            source="explicit_order_miss",
            rationale="Explicit tool execution order contained no eligible tools.",
            considered_tools=eligible_names,
        )

    if _should_force_repeat_eval(context) and "run_eval_repeated" in eligible_names:
        return ToolSelectionDecision(
            selected_tool="run_eval_repeated",
            source="loop_followthrough",
            rationale="A mutation already happened, so the next action must be loop-closing re-evaluation.",
            considered_tools=eligible_names,
        )

    if _needs_structured_diagnosis(context) and "call_llm_structured" in eligible_names:
        return ToolSelectionDecision(
            selected_tool="call_llm_structured",
            source="diagnosis_required",
            rationale="Failure evidence exists but no ranked hypothesis does, so diagnosis must happen before mutation.",
            considered_tools=eligible_names,
        )

    if _has_ranked_targets(context) and "mutate_agent_files" in eligible_names:
        return ToolSelectionDecision(
            selected_tool="mutate_agent_files",
            source="hypothesis_to_mutation",
            rationale="A ranked diagnosis target exists, so the next action should be one targeted mutation.",
            considered_tools=eligible_names,
        )

    preferred_tool = context.runtime.get("preferred_tool")
    if isinstance(preferred_tool, str) and preferred_tool in eligible_names:
        return ToolSelectionDecision(
            selected_tool=preferred_tool,
            source="context_preference",
            rationale="Used preferred_tool from runtime context.",
            considered_tools=eligible_names,
        )

    return ToolSelectionDecision(
        selected_tool=None,
        source="guardrail_clear",
        rationale="No guardrail forced a specific tool; defer to runtime ordering.",
        considered_tools=eligible_names,
    )


def _has_active_failure(context: ContextSnapshot) -> bool:
    failure_family = context.failure.get("failure_family")
    if isinstance(failure_family, str) and failure_family.strip():
        return True
    explanations = context.failure.get("explanations")
    return isinstance(explanations, list) and bool(explanations)


def _needs_structured_diagnosis(context: ContextSnapshot) -> bool:
    ranked = context.failure.get("ranked_hypotheses")
    mutated = context.workspace.get("mutated_files")
    return _has_active_failure(context) and not ranked and not mutated


def _has_ranked_targets(context: ContextSnapshot) -> bool:
    raw_hypotheses = context.failure.get("ranked_hypotheses")
    if not isinstance(raw_hypotheses, list):
        return False
    for hypothesis in raw_hypotheses:
        if isinstance(hypothesis, dict) and hypothesis.get("target_files"):
            return True
    return False


def _should_force_repeat_eval(context: ContextSnapshot) -> bool:
    mutated_files = context.workspace.get("mutated_files")
    return isinstance(mutated_files, list) and bool(mutated_files)
