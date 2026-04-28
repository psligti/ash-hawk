# type-hygiene: skip-file
from __future__ import annotations

from typing import Any

from ash_hawk.thin_runtime.models import (
    AgentSpec,
    ContextSnapshot,
    RuntimeGoal,
    SkillSpec,
    ToolSpec,
)


def _normalize_memory_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _build_memory_sections(memory_snapshot: dict[str, dict[str, Any]]) -> list[str]:
    semantic_memory = memory_snapshot.get("semantic_memory", {})
    personal_memory = memory_snapshot.get("personal_memory", {})
    episodic_memory = memory_snapshot.get("episodic_memory", {})

    semantic_rules = _normalize_memory_items(semantic_memory.get("rules"))
    semantic_boosts = _normalize_memory_items(semantic_memory.get("boosts"))
    semantic_penalties = _normalize_memory_items(semantic_memory.get("penalties"))
    personal_preferences = _normalize_memory_items(personal_memory.get("preferences"))
    episodic_entries = _normalize_memory_items(episodic_memory.get("episodes"))

    if not any(
        [
            semantic_rules,
            semantic_boosts,
            semantic_penalties,
            personal_preferences,
            episodic_entries,
        ]
    ):
        return []

    sections = ["Memory guidance:"]
    if personal_preferences:
        sections.append("User preferences:")
        sections.extend(f"- {item}" for item in personal_preferences[:5])
    if semantic_rules:
        sections.append("Learned rules:")
        sections.extend(f"- {item}" for item in semantic_rules[:5])
    if semantic_boosts:
        sections.append("Behaviors to reinforce:")
        sections.extend(f"- {item}" for item in semantic_boosts[:5])
    if semantic_penalties:
        sections.append("Behaviors to avoid:")
        sections.extend(f"- {item}" for item in semantic_penalties[:5])
    if episodic_entries:
        sections.append("Relevant past episodes:")
        sections.extend(f"- {item}" for item in episodic_entries[:3])
    return sections


def build_agent_text(
    goal: RuntimeGoal,
    agent: AgentSpec,
    skills: list[SkillSpec],
    *,
    tools: list[ToolSpec] | None = None,
    context_snapshot: ContextSnapshot | None = None,
    memory_snapshot: dict[str, dict[str, Any]] | None = None,
    include_skill_instructions: bool = True,
) -> str:
    memory_sections = _build_memory_sections(memory_snapshot or {})
    sections: list[str] = [
        f"Goal: {goal.description}",
        f"Agent: {agent.name}",
    ]
    if context_snapshot is not None:
        sections.extend(_build_context_sections(context_snapshot))
    if agent.instructions_markdown:
        sections.extend(["Agent instructions:", agent.instructions_markdown])
    if skills:
        sections.append(f"Active skill names: {', '.join(skill.name for skill in skills)}")
    del tools
    if include_skill_instructions:
        sections.append("Active skills:")
        for skill in skills:
            sections.append(f"## Skill: {skill.name}")
            if skill.description:
                sections.append(skill.description)
            if skill.instructions_markdown:
                sections.append(skill.instructions_markdown)
    sections.extend(
        [
            *memory_sections,
        ]
    )
    return "\n".join(sections)


def build_live_brief(context_snapshot: ContextSnapshot) -> str:
    return "\n".join(_live_brief_lines(context_snapshot))


def build_live_checkpoint(tool_name: str, context_snapshot: ContextSnapshot) -> str:
    lines = [f"Live context update after {tool_name}:"]
    lines.extend(_live_brief_lines(context_snapshot, include_header=False))
    return "\n".join(lines)


def _build_context_sections(context_snapshot: ContextSnapshot) -> list[str]:
    return _live_brief_lines(context_snapshot)


def _live_brief_lines(
    context_snapshot: ContextSnapshot,
    *,
    include_header: bool = True,
) -> list[str]:
    sections: list[str] = ["Current run context:"] if include_header else []
    goal_intent = context_snapshot.runtime.get("goal_intent")
    if isinstance(goal_intent, str) and goal_intent.strip():
        sections.extend(["Objective:", f"- {goal_intent.strip()}"])

    phase = context_snapshot.runtime.get("phase")
    if isinstance(phase, str) and phase.strip():
        sections.extend(["Phase:", f"- {phase.strip()}"])

    progress_summary = context_snapshot.runtime.get("progress_summary")
    if isinstance(progress_summary, str) and progress_summary.strip():
        sections.extend(["Progress:", f"- {progress_summary.strip()}"])

    active_skills = _normalize_memory_items(context_snapshot.runtime.get("active_skills"))
    if active_skills:
        sections.append("Active skills:")
        sections.extend(f"- {item}" for item in active_skills[:6])

    recent_steps = _normalize_memory_items(context_snapshot.runtime.get("recent_steps"))
    if recent_steps:
        sections.append("Recent steps:")
        sections.extend(f"- {item}" for item in recent_steps[:3])

    latest_evidence = _normalize_memory_items(context_snapshot.runtime.get("latest_evidence"))
    if latest_evidence:
        sections.append("Latest evidence:")
        sections.extend(f"- {item}" for item in latest_evidence[:4])

    actionable_files = _normalize_memory_items(context_snapshot.workspace.get("actionable_files"))
    if actionable_files:
        sections.append("Actionable files:")
        sections.extend(f"- {item}" for item in actionable_files[:5])

    reference_files = _normalize_memory_items(context_snapshot.workspace.get("reference_files"))
    if reference_files:
        sections.append("Reference files:")
        sections.extend(f"- {item}" for item in reference_files[:4])

    blocked_files = _normalize_memory_items(context_snapshot.workspace.get("blocked_files"))
    if blocked_files:
        sections.append("Blocked files:")
        sections.extend(f"- {item}" for item in blocked_files[:4])

    constraints = _normalize_memory_items(context_snapshot.runtime.get("constraints"))
    if constraints:
        sections.append("Constraints:")
        sections.extend(f"- {item}" for item in constraints[:3])

    repl_sessions = _normalize_memory_items(
        context_snapshot.workspace.get("open_python_repl_sessions")
    )
    if repl_sessions:
        sections.append("Open python REPL sessions:")
        sections.extend(f"- {item}" for item in repl_sessions[:4])

    next_pressure = context_snapshot.runtime.get("next_pressure")
    if isinstance(next_pressure, str) and next_pressure.strip():
        sections.extend(["Next pressure:", f"- {next_pressure.strip()}"])

    artifact_index = _normalize_memory_items(context_snapshot.audit.get("artifact_index"))
    if artifact_index:
        sections.append("Artifact refs:")
        sections.extend(f"- {item}" for item in artifact_index[:6])

    return sections
