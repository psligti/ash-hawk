"""Lesson injection service for applying curated lessons to agent execution."""

from __future__ import annotations

from typing import Any

from ash_hawk.contracts import CuratedLesson
from ash_hawk.contracts.lesson_payloads import (
    HarnessLessonPayload,
    PolicyLessonPayload,
    SkillLessonPayload,
    ToolLessonPayload,
    parse_lesson_payload,
)
from ash_hawk.services.lesson_service import LessonService


class LessonInjector:
    """Injects curated lessons into agent execution context.

    Loads lessons for a target agent and transforms them into
    runtime modifications: prompt augmentations, tool overrides,
    policy rules, and harness adjustments.
    """

    def __init__(self, lesson_service: LessonService | None = None) -> None:
        self._service = lesson_service or LessonService()

    def inject_into_prompt(self, agent_id: str, base_prompt: str) -> str:
        """Inject skill and policy lessons into agent prompt.

        Args:
            agent_id: The agent to load lessons for.
            base_prompt: The original prompt text.

        Returns:
            Augmented prompt with learned lessons appended.
        """
        lessons = self._service.get_lessons_for_agent(agent_id)
        skill_lessons = [l for l in lessons if l.lesson_type == "skill"]
        policy_lessons = [l for l in lessons if l.lesson_type == "policy"]

        additions: list[str] = []

        for lesson in skill_lessons:
            payload = parse_lesson_payload("skill", lesson.lesson_payload)
            if isinstance(payload, SkillLessonPayload):
                additions.extend(payload.instruction_additions)

        for lesson in policy_lessons:
            payload = parse_lesson_payload("policy", lesson.lesson_payload)
            if isinstance(payload, PolicyLessonPayload):
                if payload.enabled:
                    rule_desc = f"Rule '{payload.rule_name}' ({payload.rule_type})"
                    if payload.condition:
                        rule_desc += f" when {payload.condition}"
                    if payload.action:
                        rule_desc += f" then {payload.action}"
                    additions.append(rule_desc)

        if not additions:
            return base_prompt

        lessons_section = "\n## Learned Lessons\n\n" + "\n".join(
            f"- {addition}" for addition in additions
        )
        return base_prompt + lessons_section

    def get_tool_overrides(self, agent_id: str) -> dict[str, dict[str, Any]]:
        """Get tool parameter overrides from lessons.

        Args:
            agent_id: The agent to load lessons for.

        Returns:
            Dict mapping tool_id to override configuration.
        """
        lessons = self._service.get_lessons_for_agent(agent_id)
        tool_lessons = [l for l in lessons if l.lesson_type == "tool"]

        overrides: dict[str, dict[str, Any]] = {}
        for lesson in tool_lessons:
            payload = parse_lesson_payload("tool", lesson.lesson_payload)
            if isinstance(payload, ToolLessonPayload):
                overrides[payload.tool_id] = {
                    "defaults": payload.parameter_defaults,
                    "hints": payload.usage_hints,
                    "timeout": payload.timeout_override,
                    "preconditions": payload.preconditions,
                }

        return overrides

    def get_policy_rules(self, agent_id: str) -> list[dict[str, Any]]:
        """Get policy rules from lessons for Vox Jay integration.

        Args:
            agent_id: The agent to load lessons for.

        Returns:
            List of policy rule configurations.
        """
        lessons = self._service.get_lessons_for_agent(agent_id)
        policy_lessons = [l for l in lessons if l.lesson_type == "policy"]

        rules: list[dict[str, Any]] = []
        for lesson in policy_lessons:
            payload = parse_lesson_payload("policy", lesson.lesson_payload)
            if isinstance(payload, PolicyLessonPayload) and payload.enabled:
                rules.append(
                    {
                        "name": payload.rule_name,
                        "type": payload.rule_type,
                        "condition": payload.condition,
                        "action": payload.action,
                        "priority": payload.priority,
                        "source_lesson": lesson.lesson_id,
                    }
                )

        rules.sort(key=lambda r: r["priority"], reverse=True)
        return rules

    def get_harness_adjustments(self, agent_id: str) -> dict[str, Any]:
        """Get harness adjustments from lessons.

        Args:
            agent_id: The agent to load lessons for.

        Returns:
            Dict with grader_adjustments, fixture_overrides, timeout_adjustments.
        """
        lessons = self._service.get_lessons_for_agent(agent_id)
        harness_lessons = [l for l in lessons if l.lesson_type == "harness"]

        combined: dict[str, Any] = {
            "grader_adjustments": {},
            "fixture_overrides": {},
            "timeout_adjustments": {},
            "parallelism": None,
        }

        for lesson in harness_lessons:
            payload = parse_lesson_payload("harness", lesson.lesson_payload)
            if isinstance(payload, HarnessLessonPayload):
                combined["grader_adjustments"].update(payload.grader_adjustments)
                combined["fixture_overrides"].update(payload.fixture_overrides)
                combined["timeout_adjustments"].update(payload.timeout_adjustments)
                if payload.parallelism_override is not None:
                    combined["parallelism"] = payload.parallelism_override

        return combined

    def get_all_lessons(self, agent_id: str) -> list[CuratedLesson]:
        """Get all active lessons for an agent.

        Args:
            agent_id: The agent to load lessons for.

        Returns:
            List of all active CuratedLessons.
        """
        return self._service.get_lessons_for_agent(agent_id)

    def has_lessons(self, agent_id: str) -> bool:
        """Check if an agent has any active lessons.

        Args:
            agent_id: The agent to check.

        Returns:
            True if agent has at least one active lesson.
        """
        return len(self._service.get_lessons_for_agent(agent_id)) > 0
