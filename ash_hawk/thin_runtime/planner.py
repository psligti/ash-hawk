from __future__ import annotations

import json
from pathlib import Path

import pydantic as pd

from ash_hawk.thin_runtime.llm_client import call_model_structured
from ash_hawk.thin_runtime.models import (
    AgentSpec,
    ContextSnapshot,
    RuntimeGoal,
    SkillSpec,
    ToolSpec,
)
from ash_hawk.thin_runtime.tool_types import ToolContractView

MAX_SKILL_INSTRUCTIONS_ACTIVE = 2_000
MAX_SKILL_INSTRUCTIONS_CANDIDATE = 600
MAX_TEXT_CHARS = 1_000
MAX_LIST_ITEMS = 5


class PlannerDecision(pd.BaseModel):
    selected_tool: str | None = None
    activate_skills: list[str] = pd.Field(default_factory=list)
    source: str
    rationale: str
    considered_tools: list[str] = pd.Field(default_factory=list)
    confidence: float | None = pd.Field(default=None, ge=0.0, le=1.0)
    reason_model_authored: bool = pd.Field(default=False)

    model_config = pd.ConfigDict(extra="forbid")


class _PlannerResponse(pd.BaseModel):
    selected_tool: str | None = None
    activate_skills: list[str] = pd.Field(default_factory=list)
    rationale: str
    confidence: float | None = pd.Field(default=None, ge=0.0, le=1.0)

    model_config = pd.ConfigDict(extra="forbid")


def plan_next_tool(
    *,
    goal: RuntimeGoal,
    agent: AgentSpec,
    active_skills: list[SkillSpec],
    candidate_skills: list[SkillSpec],
    candidate_tools: list[ToolSpec],
    context: ContextSnapshot,
    tool_execution_order: list[str] | None,
    feedback: str | None = None,
) -> PlannerDecision:
    candidate_names = [tool.name for tool in candidate_tools]
    if not candidate_names:
        return PlannerDecision(
            selected_tool=None,
            source="no_candidate_tools",
            rationale="No allowed tools are available for the current thin-runtime state.",
            considered_tools=[],
            reason_model_authored=False,
        )

    if tool_execution_order is not None:
        for tool_name in tool_execution_order:
            if tool_name in candidate_names:
                return PlannerDecision(
                    selected_tool=tool_name,
                    source="explicit_order",
                    rationale="Selected from explicit tool execution order override.",
                    considered_tools=candidate_names,
                    reason_model_authored=False,
                )
        return PlannerDecision(
            selected_tool=None,
            source="explicit_order_miss",
            rationale="Explicit tool execution order contained no candidate tools.",
            considered_tools=candidate_names,
            reason_model_authored=False,
        )

    response = call_model_structured(
        _PlannerResponse,
        system_prompt=(
            "You are the thin-runtime planner. Choose the next tool to execute from the allowed "
            "tool surface. Use the provided skill context to decide what reasoning or execution "
            "frame is relevant right now. Return only one next tool, the skills that should be "
            "active for that step, and a concise rationale."
        ),
        user_prompt=build_planner_prompt(
            goal=goal,
            agent=agent,
            active_skills=active_skills,
            candidate_skills=candidate_skills,
            candidate_tools=candidate_tools,
            context=context,
            feedback=feedback,
        ),
        working_dir=_planner_workdir(context),
    )
    if response is None:
        return PlannerDecision(
            selected_tool=None,
            source="model_unavailable",
            rationale="The model planner was unavailable and no model-selected next tool was produced.",
            considered_tools=candidate_names,
            reason_model_authored=False,
        )

    if response.selected_tool not in candidate_names:
        return PlannerDecision(
            selected_tool=None,
            source="invalid_model_selection",
            rationale=(
                f"The model selected '{response.selected_tool}', which is not in the current allowed "
                "tool surface."
            ),
            considered_tools=candidate_names,
            confidence=response.confidence,
            reason_model_authored=False,
        )

    candidate_skill_names = {skill.name for skill in candidate_skills}
    invalid_skills = [
        name for name in response.activate_skills if name not in candidate_skill_names
    ]
    if invalid_skills:
        return PlannerDecision(
            selected_tool=None,
            source="invalid_model_skills",
            rationale=(
                "The model requested unavailable skills: " + ", ".join(sorted(invalid_skills))
            ),
            considered_tools=candidate_names,
            confidence=response.confidence,
            reason_model_authored=False,
        )

    return PlannerDecision(
        selected_tool=response.selected_tool,
        activate_skills=response.activate_skills,
        source="model_planner",
        rationale=response.rationale,
        considered_tools=candidate_names,
        confidence=response.confidence,
        reason_model_authored=True,
    )


def build_planner_prompt(
    *,
    goal: RuntimeGoal,
    agent: AgentSpec,
    active_skills: list[SkillSpec],
    candidate_skills: list[SkillSpec],
    candidate_tools: list[ToolSpec],
    context: ContextSnapshot,
    feedback: str | None,
) -> str:
    active_skill_names = {item.name for item in active_skills}
    skill_payload = [
        {
            "name": skill.name,
            "description": skill.description,
            "when_to_use": skill.when_to_use,
            "when_not_to_use": skill.when_not_to_use,
            "tool_names": skill.tool_names,
            "input_contexts": skill.input_contexts,
            "output_contexts": skill.output_contexts,
            "instructions_excerpt": _truncate_text(
                skill.instructions_markdown,
                (
                    MAX_SKILL_INSTRUCTIONS_ACTIVE
                    if skill.name in active_skill_names
                    else MAX_SKILL_INSTRUCTIONS_CANDIDATE
                ),
            ),
            "active_now": skill.name in active_skill_names,
        }
        for skill in candidate_skills
    ]
    tool_payload = [
        build_tool_contract_view(tool).model_dump(exclude_none=True) for tool in candidate_tools
    ]
    sections = [
        f"Goal ID: {goal.goal_id}",
        f"Goal: {goal.description}",
        f"Agent: {agent.name}",
        f"Active skills: {', '.join(skill.name for skill in active_skills) or 'none'}",
        "Choose the single best next tool from the allowed tool surface below.",
        "If a currently inactive skill is needed, include it in activate_skills.",
        "Allowed tool contracts:",
        json.dumps(tool_payload, indent=2, sort_keys=True),
        "Candidate skills:",
        json.dumps(skill_payload, indent=2, sort_keys=True),
        "Current context:",
        json.dumps(_compact_context_for_planner(context), indent=2, sort_keys=True),
    ]
    if feedback:
        sections.extend(["Previous planning feedback:", feedback])
    sections.extend(
        [
            "Return JSON with this exact shape:",
            '{"selected_tool": "tool_name", "activate_skills": ["skill-name"], "rationale": "why this is the next best step", "confidence": 0.0}',
        ]
    )
    return "\n\n".join(sections)


def build_tool_contract_view(tool: ToolSpec) -> ToolContractView:
    return ToolContractView(
        name=tool.name,
        summary=tool.summary or tool.description,
        when_to_use=tool.when_to_use,
        when_not_to_use=tool.when_not_to_use,
        inputs=tool.inputs,
        outputs=tool.outputs,
        side_effects=tool.side_effects,
        risk_level=tool.risk_level,
        timeout_seconds=tool.timeout_seconds,
        completion_criteria=tool.completion_criteria,
        escalation_rules=tool.escalation_rules,
    )


def _planner_workdir(context: ContextSnapshot) -> Path | None:
    raw_workdir = context.workspace.get("workdir")
    if isinstance(raw_workdir, str) and raw_workdir.strip():
        return Path(raw_workdir)
    return None


def _compact_context_for_planner(context: ContextSnapshot) -> dict[str, object]:
    return {
        "goal": _compact_mapping(
            context.goal,
            keys=["goal_id", "description", "target_score", "max_iterations"],
        ),
        "runtime": _compact_mapping(
            context.runtime,
            keys=[
                "lead_agent",
                "active_skills",
                "active_skill_summaries",
                "phase",
                "recent_steps",
                "latest_evidence",
                "constraints",
                "max_iterations",
                "completed_iterations",
                "remaining_iterations",
                "goal_intent",
                "progress_summary",
                "next_pressure",
                "preferred_tool",
                "last_decision",
                "stop_reason",
            ],
        ),
        "workspace": _compact_mapping(
            context.workspace,
            keys=[
                "workdir",
                "source_root",
                "allowed_target_files",
                "changed_files",
                "mutated_files",
                "scenario_path",
                "scenario_required_files",
                "scenario_targets",
                "scenario_summary",
                "actionable_files",
                "reference_files",
                "blocked_files",
                "file_summaries",
                "open_python_repl_sessions",
                "package_name",
                "agent_config",
            ],
        ),
        "evaluation": _compact_mapping(
            context.evaluation,
            keys=[
                "baseline_summary",
                "last_eval_summary",
                "repeat_eval_summary",
                "recent_eval_summaries",
                "aggregated_score",
                "regressions",
                "acceptance",
                "verification",
            ],
        ),
        "failure": {
            **_compact_mapping(
                context.failure,
                keys=[
                    "failure_family",
                    "explanations",
                    "diagnosed_issues",
                    "top_hypothesis",
                    "suspicious_reviews",
                ],
            ),
            "failed_trials_count": _count_value(context.failure.get("failed_trials")),
            "ranked_hypotheses": _summarize_hypotheses(context.failure.get("ranked_hypotheses")),
        },
        "memory": _compact_memory_section(context.memory),
        "tool": {
            **_compact_mapping(
                context.tool,
                keys=[
                    "active_tools",
                    "registered_mcp_tools",
                    "policy_decisions",
                    "recent_tool_history",
                    "last_tool",
                ],
            ),
            "tool_contracts_count": _count_value(context.tool.get("tool_contracts")),
        },
        "audit": {
            "artifact_count": _count_value(context.audit.get("artifacts")),
            "artifact_index": _summarize_value(context.audit.get("artifact_index", [])),
            "decision_trace": _summarize_value(context.audit.get("decision_trace", [])),
        },
    }


def _compact_memory_section(memory_section: dict[str, object]) -> dict[str, object]:
    working_snapshot = _mapping_value(memory_section.get("working_snapshot"))
    session_memory = _mapping_value(memory_section.get("session"))
    semantic_memory = _mapping_value(memory_section.get("semantic"))
    personal_memory = _mapping_value(memory_section.get("personal"))
    episodic_memory = _mapping_value(memory_section.get("episodic"))
    return {
        "working_snapshot": _compact_mapping(
            working_snapshot,
            keys=["active_hypotheses", "phase_status", "last_result"],
        ),
        "session_counts": {
            "delegations": _count_value(session_memory.get("delegations")),
            "retries": _count_value(session_memory.get("retries")),
            "validations": _count_value(session_memory.get("validations")),
            "traces": _count_value(session_memory.get("traces")),
            "transcripts": _count_value(session_memory.get("transcripts")),
            "dream_queue": _count_value(session_memory.get("dream_queue")),
        },
        "semantic_memory": {
            "rules_count": _count_value(semantic_memory.get("rules")),
            "boosts_count": _count_value(semantic_memory.get("boosts")),
            "penalties_count": _count_value(semantic_memory.get("penalties")),
        },
        "personal_preferences_count": _count_value(personal_memory.get("preferences")),
        "episode_count": _count_value(episodic_memory.get("episodes")),
    }


def _compact_mapping(source: dict[str, object], *, keys: list[str]) -> dict[str, object]:
    compact: dict[str, object] = {}
    for key in keys:
        if key not in source:
            continue
        compact[key] = _summarize_value(source[key])
    return compact


def _summarize_value(value: object) -> object:
    if isinstance(value, str):
        return _truncate_text(value, MAX_TEXT_CHARS)
    if isinstance(value, list):
        items = [_summarize_value(item) for item in value[:MAX_LIST_ITEMS]]
        if len(value) > MAX_LIST_ITEMS:
            items.append(f"... +{len(value) - MAX_LIST_ITEMS} more")
        return items
    if isinstance(value, dict):
        dict_items: list[tuple[object, object]] = list(value.items())[:MAX_LIST_ITEMS]
        compact = {str(key): _summarize_value(item) for key, item in dict_items}
        if len(value) > MAX_LIST_ITEMS:
            compact["__truncated__"] = f"+{len(value) - MAX_LIST_ITEMS} more keys"
        return compact
    return value


def _summarize_hypotheses(raw_hypotheses: object) -> list[object]:
    if not isinstance(raw_hypotheses, list):
        return []
    summarized: list[object] = []
    for item in raw_hypotheses[:MAX_LIST_ITEMS]:
        if isinstance(item, dict):
            summarized.append(
                {
                    "name": _summarize_value(item.get("name")),
                    "score": item.get("score"),
                    "target_files": _summarize_value(item.get("target_files", [])),
                }
            )
        else:
            summarized.append(_summarize_value(item))
    if len(raw_hypotheses) > MAX_LIST_ITEMS:
        summarized.append(f"... +{len(raw_hypotheses) - MAX_LIST_ITEMS} more")
    return summarized


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [truncated {len(text) - limit} chars]"


def _mapping_value(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): cast_value for key, cast_value in value.items()}
    return {}


def _count_value(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    return 0
