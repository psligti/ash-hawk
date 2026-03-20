"""Tests for lesson payload schemas."""

from __future__ import annotations

import pytest

from ash_hawk.contracts.lesson_payloads import (
    EvalLessonPayload,
    HarnessLessonPayload,
    PolicyLessonPayload,
    SkillLessonPayload,
    ToolLessonPayload,
    parse_lesson_payload,
)


class TestPolicyLessonPayload:
    def test_create_minimal(self) -> None:
        payload = PolicyLessonPayload(
            rule_name="test-rule",
            rule_type="engagement",
        )
        assert payload.rule_name == "test-rule"
        assert payload.rule_type == "engagement"
        assert payload.condition == {}
        assert payload.action == {}
        assert payload.priority == 50
        assert payload.enabled is True

    def test_create_full(self) -> None:
        payload = PolicyLessonPayload(
            rule_name="high-priority-rule",
            rule_type="ranking",
            condition={"score_threshold": 0.8},
            action={"boost": 0.2},
            priority=90,
            enabled=False,
        )
        assert payload.rule_name == "high-priority-rule"
        assert payload.rule_type == "ranking"
        assert payload.condition == {"score_threshold": 0.8}
        assert payload.action == {"boost": 0.2}
        assert payload.priority == 90
        assert payload.enabled is False

    def test_priority_bounds(self) -> None:
        with pytest.raises(Exception):
            PolicyLessonPayload(rule_name="test", rule_type="engagement", priority=-1)
        with pytest.raises(Exception):
            PolicyLessonPayload(rule_name="test", rule_type="engagement", priority=101)


class TestSkillLessonPayload:
    def test_create_minimal(self) -> None:
        payload = SkillLessonPayload(skill_name="test-skill")
        assert payload.skill_name == "test-skill"
        assert payload.instruction_additions == []
        assert payload.instruction_removals == []
        assert payload.examples == []

    def test_create_with_instructions(self) -> None:
        payload = SkillLessonPayload(
            skill_name="code-review",
            instruction_additions=["Always check for type hints", "Run linter before commit"],
            instruction_removals=["Old pattern"],
        )
        assert len(payload.instruction_additions) == 2
        assert len(payload.instruction_removals) == 1


class TestToolLessonPayload:
    def test_create_minimal(self) -> None:
        payload = ToolLessonPayload(tool_id="read")
        assert payload.tool_id == "read"
        assert payload.parameter_defaults == {}
        assert payload.usage_hints == []
        assert payload.timeout_override is None

    def test_create_with_overrides(self) -> None:
        payload = ToolLessonPayload(
            tool_id="bash",
            parameter_defaults={"timeout": 30},
            usage_hints=["Use for single commands only"],
            timeout_override=60,
        )
        assert payload.tool_id == "bash"
        assert payload.parameter_defaults == {"timeout": 30}
        assert payload.timeout_override == 60


class TestHarnessLessonPayload:
    def test_create_minimal(self) -> None:
        payload = HarnessLessonPayload(suite_id="test-suite")
        assert payload.suite_id == "test-suite"
        assert payload.grader_adjustments == {}
        assert payload.parallelism_override is None

    def test_create_with_adjustments(self) -> None:
        payload = HarnessLessonPayload(
            suite_id="eval-suite",
            grader_adjustments={"llm_judge": {"weight": 0.8}},
            timeout_adjustments={"task-1": 120},
            parallelism_override=4,
        )
        assert payload.suite_id == "eval-suite"
        assert payload.parallelism_override == 4


class TestEvalLessonPayload:
    def test_create_minimal(self) -> None:
        payload = EvalLessonPayload(eval_id="test-eval")
        assert payload.eval_id == "test-eval"
        assert payload.rubric_additions == []
        assert payload.threshold_adjustments == {}

    def test_create_with_additions(self) -> None:
        payload = EvalLessonPayload(
            eval_id="eval-1",
            rubric_additions=[{"criterion": "code quality", "weight": 0.3}],
            threshold_adjustments={"pass_threshold": 0.75},
        )
        assert len(payload.rubric_additions) == 1
        assert payload.threshold_adjustments == {"pass_threshold": 0.75}


class TestParseLessonPayload:
    def test_parse_policy_payload(self) -> None:
        result = parse_lesson_payload(
            "policy",
            {"rule_name": "test", "rule_type": "engagement"},
        )
        assert isinstance(result, PolicyLessonPayload)
        assert result.rule_name == "test"

    def test_parse_skill_payload(self) -> None:
        result = parse_lesson_payload(
            "skill",
            {"skill_name": "testing", "instruction_additions": ["Always test"]},
        )
        assert isinstance(result, SkillLessonPayload)
        assert result.skill_name == "testing"

    def test_parse_tool_payload(self) -> None:
        result = parse_lesson_payload(
            "tool",
            {"tool_id": "grep", "timeout_override": 30},
        )
        assert isinstance(result, ToolLessonPayload)
        assert result.tool_id == "grep"

    def test_parse_harness_payload(self) -> None:
        result = parse_lesson_payload(
            "harness",
            {"suite_id": "suite-1"},
        )
        assert isinstance(result, HarnessLessonPayload)

    def test_parse_harness_payload_without_suite_id(self) -> None:
        result = parse_lesson_payload(
            "harness",
            {"timeout_adjustments": {"security": 120}},
        )
        assert isinstance(result, HarnessLessonPayload)
        assert result.suite_id is None

    def test_parse_eval_payload(self) -> None:
        result = parse_lesson_payload(
            "eval",
            {"eval_id": "eval-1"},
        )
        assert isinstance(result, EvalLessonPayload)

    def test_parse_unknown_type_returns_none(self) -> None:
        result = parse_lesson_payload("unknown", {"foo": "bar"})
        assert result is None

    def test_parse_invalid_payload_returns_none(self) -> None:
        result = parse_lesson_payload(
            "policy",
            {"invalid": "data"},
        )
        assert result is None
