# type-hygiene: skip-file
from __future__ import annotations

from typing import Any

import pydantic as pd

from ash_hawk.thin_runtime.llm_client import call_model_structured
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


def _format_memory_context(memory: dict[str, Any]) -> str:
    semantic_memory = memory.get("semantic", {})
    personal_memory = memory.get("personal", {})
    episodic_memory = memory.get("episodic", {})
    session_memory = memory.get("session", {})

    def _limited_list(raw: Any, *, limit: int) -> list[str]:
        if not isinstance(raw, list):
            return []
        values = [str(item).strip() for item in raw if str(item).strip()]
        return values[:limit]

    summary = {
        "preferences": _limited_list(personal_memory.get("preferences"), limit=5),
        "rules": _limited_list(semantic_memory.get("rules"), limit=5),
        "boosts": _limited_list(semantic_memory.get("boosts"), limit=5),
        "penalties": _limited_list(semantic_memory.get("penalties"), limit=5),
        "episodes": _limited_list(episodic_memory.get("episodes"), limit=3),
        "recent_search_results": _limited_list(memory.get("search_results"), limit=5),
        "recent_transcripts": _limited_list(session_memory.get("transcripts"), limit=3),
    }
    return str({key: value for key, value in summary.items() if value})


def select_tool_via_policy(
    *,
    goal: RuntimeGoal,
    agent: AgentSpec,
    active_skills: list[SkillSpec],
    eligible_tools: list[ToolSpec],
    context: ContextSnapshot,
    tool_execution_order: list[str] | None,
) -> ToolSelectionDecision:
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

    preferred_tool = context.runtime.get("preferred_tool")
    if isinstance(preferred_tool, str) and preferred_tool in eligible_names:
        return ToolSelectionDecision(
            selected_tool=preferred_tool,
            source="context_preference",
            rationale="Used preferred_tool from runtime context.",
            considered_tools=eligible_names,
        )

    tool_descriptions = [
        f"- {tool.name}: {tool.summary} | when_to_use={tool.when_to_use} | when_not_to_use={tool.when_not_to_use}"
        for tool in eligible_tools
    ]
    model_result = call_model_structured(
        ToolSelectionDecision,
        system_prompt=(
            "You are the explicit tool-selection policy for an agentic runtime. "
            "Choose the next tool from the provided eligible tools, or decide that no tool should be selected. "
            "Base the decision on the agent text, active skills, failure/evaluation context, and the declared when_to_use/when_not_to_use guidance."
        ),
        user_prompt=(
            f"Goal: {goal.description}\n"
            f"Agent: {agent.name}\n"
            f"Agent text:\n{context.runtime.get('agent_text', '')}\n\n"
            f"Active skills: {[skill.name for skill in active_skills]}\n"
            f"Workspace context: {context.workspace}\n"
            f"Failure context: {context.failure}\n"
            f"Evaluation context: {context.evaluation}\n"
            f"Memory context: {_format_memory_context(context.memory)}\n"
            f"Eligible tools:\n{chr(10).join(tool_descriptions)}\n\n"
            'Return JSON: {"selected_tool": "tool_name or null", "source": "model_policy", "rationale": "why", "considered_tools": ["tool1", "tool2"]}'
        ),
    )
    if model_result is not None and model_result.selected_tool in eligible_names:
        return model_result.model_copy(update={"considered_tools": eligible_names})

    return ToolSelectionDecision(
        selected_tool=None,
        source="policy_unavailable",
        rationale="No explicit policy decision was produced.",
        considered_tools=eligible_names,
    )
