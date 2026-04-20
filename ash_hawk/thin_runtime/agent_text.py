# type-hygiene: skip-file
from __future__ import annotations

from typing import Any

from ash_hawk.thin_runtime.models import AgentSpec, RuntimeGoal, SkillSpec


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
    memory_snapshot: dict[str, dict[str, Any]] | None = None,
) -> str:
    memory_sections = _build_memory_sections(memory_snapshot or {})
    sections: list[str] = [
        f"Goal: {goal.description}",
        f"Agent: {agent.name}",
    ]
    if agent.instructions_markdown:
        sections.extend(["Agent instructions:", agent.instructions_markdown])
    if skills:
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
