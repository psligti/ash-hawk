"""Tests for LessonInjector service."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson
from ash_hawk.services.lesson_injector import LessonInjector
from ash_hawk.services.lesson_service import LessonService


@pytest.fixture
def lesson_service() -> LessonService:
    return LessonService()


@pytest.fixture
def lesson_injector(lesson_service: LessonService) -> LessonInjector:
    return LessonInjector(lesson_service)


@pytest.fixture
def skill_lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-skill-1",
        source_proposal_id="prop-1",
        applies_to_agents=["test-agent"],
        lesson_type="skill",
        title="Prefer LSP over grep",
        description="Use LSP tools for symbol lookups",
        lesson_payload={
            "skill_name": "symbol-lookup",
            "instruction_additions": [
                "Prefer LSP tools (goto_definition, find_references) over grep for symbol lookups",
                "Use ast_grep for structural code matching",
            ],
        },
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def policy_lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-policy-1",
        source_proposal_id="prop-2",
        applies_to_agents=["test-agent"],
        lesson_type="policy",
        title="Boost high-confidence passes",
        description="Prioritize tasks with high prediction confidence",
        lesson_payload={
            "rule_name": "boost-high-confidence",
            "rule_type": "ranking",
            "condition": {"predicted_score": {">=": 0.9}},
            "action": {"boost": 0.2},
            "priority": 80,
        },
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def tool_lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-tool-1",
        source_proposal_id="prop-3",
        applies_to_agents=["test-agent"],
        lesson_type="tool",
        title="Increase bash timeout",
        description="Bash commands need more time",
        lesson_payload={
            "tool_id": "bash",
            "parameter_defaults": {"timeout": 60},
            "usage_hints": ["Use for build commands", "Avoid for quick checks"],
            "timeout_override": 120,
        },
        created_at=datetime.now(UTC),
    )


class TestLessonInjectorInjectIntoPrompt:
    def test_no_lessons_returns_unchanged(self, lesson_injector: LessonInjector) -> None:
        prompt = "Original prompt"
        result = lesson_injector.inject_into_prompt("unknown-agent", prompt)
        assert result == prompt

    def test_skill_lesson_appends_instructions(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        skill_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(skill_lesson)
        prompt = "Complete the task"
        result = lesson_injector.inject_into_prompt("test-agent", prompt)
        assert result.startswith(prompt)
        assert "Prefer LSP tools" in result
        assert "## Learned Lessons" in result

    def test_policy_lesson_appends_rules(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        policy_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(policy_lesson)
        prompt = "Complete the task"
        result = lesson_injector.inject_into_prompt("test-agent", prompt)
        assert "boost-high-confidence" in result
        assert "ranking" in result

    def test_multiple_lessons_combined(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        skill_lesson: CuratedLesson,
        policy_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(skill_lesson)
        lesson_service._store.store(policy_lesson)
        prompt = "Complete the task"
        result = lesson_injector.inject_into_prompt("test-agent", prompt)
        assert "Prefer LSP tools" in result
        assert "boost-high-confidence" in result


class TestLessonInjectorGetToolOverrides:
    def test_no_lessons_returns_empty(self, lesson_injector: LessonInjector) -> None:
        result = lesson_injector.get_tool_overrides("unknown-agent")
        assert result == {}

    def test_tool_lesson_returns_overrides(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        tool_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(tool_lesson)
        result = lesson_injector.get_tool_overrides("test-agent")
        assert "bash" in result
        assert result["bash"]["defaults"] == {"timeout": 60}
        assert result["bash"]["timeout"] == 120
        assert "Use for build commands" in result["bash"]["hints"]


class TestLessonInjectorGetPolicyRules:
    def test_no_lessons_returns_empty(self, lesson_injector: LessonInjector) -> None:
        result = lesson_injector.get_policy_rules("unknown-agent")
        assert result == []

    def test_policy_lesson_returns_rules(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        policy_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(policy_lesson)
        result = lesson_injector.get_policy_rules("test-agent")
        assert len(result) == 1
        assert result[0]["name"] == "boost-high-confidence"
        assert result[0]["type"] == "ranking"
        assert result[0]["priority"] == 80

    def test_rules_sorted_by_priority(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
    ) -> None:
        lesson1 = CuratedLesson(
            lesson_id="lesson-p1",
            source_proposal_id="prop-1",
            applies_to_agents=["test-agent"],
            lesson_type="policy",
            title="Low priority",
            description="",
            lesson_payload={
                "rule_name": "low-priority",
                "rule_type": "engagement",
                "priority": 20,
            },
            created_at=datetime.now(UTC),
        )
        lesson2 = CuratedLesson(
            lesson_id="lesson-p2",
            source_proposal_id="prop-2",
            applies_to_agents=["test-agent"],
            lesson_type="policy",
            title="High priority",
            description="",
            lesson_payload={
                "rule_name": "high-priority",
                "rule_type": "engagement",
                "priority": 90,
            },
            created_at=datetime.now(UTC),
        )
        lesson_service._store.store(lesson1)
        lesson_service._store.store(lesson2)
        result = lesson_injector.get_policy_rules("test-agent")
        assert result[0]["name"] == "high-priority"
        assert result[1]["name"] == "low-priority"

    def test_disabled_rules_excluded(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
    ) -> None:
        lesson = CuratedLesson(
            lesson_id="lesson-disabled",
            source_proposal_id="prop-1",
            applies_to_agents=["test-agent"],
            lesson_type="policy",
            title="Disabled rule",
            description="",
            lesson_payload={
                "rule_name": "disabled-rule",
                "rule_type": "engagement",
                "priority": 50,
                "enabled": False,
            },
            created_at=datetime.now(UTC),
        )
        lesson_service._store.store(lesson)
        result = lesson_injector.get_policy_rules("test-agent")
        assert result == []


class TestLessonInjectorHasLessons:
    def test_no_lessons_returns_false(self, lesson_injector: LessonInjector) -> None:
        assert lesson_injector.has_lessons("unknown-agent") is False

    def test_with_lessons_returns_true(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        skill_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(skill_lesson)
        assert lesson_injector.has_lessons("test-agent") is True


class TestLessonInjectorGetAllLessons:
    def test_returns_all_lessons_for_agent(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        skill_lesson: CuratedLesson,
        tool_lesson: CuratedLesson,
    ) -> None:
        lesson_service._store.store(skill_lesson)
        lesson_service._store.store(tool_lesson)
        lessons = lesson_injector.get_all_lessons("test-agent")
        assert len(lessons) == 2
