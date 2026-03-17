from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pydantic as pd

if TYPE_CHECKING:
    from ash_hawk.contracts import CuratedLesson


class ApplicabilityContext(pd.BaseModel):
    agent_id: str = pd.Field(description="Agent ID to check applicability for")
    task_type: str | None = pd.Field(default=None, description="Type of task being executed")
    tool_context: list[str] = pd.Field(
        default_factory=list, description="Tools available in context"
    )
    error_patterns: list[str] = pd.Field(
        default_factory=list, description="Recent error patterns observed"
    )
    experiment_id: str | None = pd.Field(default=None, description="Current experiment ID")
    strategy_focus: list[str] = pd.Field(
        default_factory=list, description="Strategies being focused on"
    )
    metadata: dict[str, Any] = pd.Field(default_factory=dict, description="Additional context")

    model_config = pd.ConfigDict(extra="allow")


class RuleEvaluationResult(pd.BaseModel):
    rule: str = pd.Field(description="The rule that was evaluated")
    passed: bool = pd.Field(description="Whether the rule passed")
    reason: str = pd.Field(description="Why the rule passed or failed")

    model_config = pd.ConfigDict(extra="forbid")


@dataclass
class ApplicabilityResult:
    lesson_id: str
    applicable: bool
    rule_results: list[RuleEvaluationResult]
    match_score: float
    matched_rules: list[str]
    failed_rules: list[str]


class RuleParser:
    CONDITION_PATTERN = re.compile(r"^(agent|task|tool|error|experiment|strategy|meta):(.+)$")
    OPERATOR_PATTERN = re.compile(r"^([^=<>!~]+)(=|!=|~=|>|<|>=|<=)(.+)$")

    def parse(self, rule: str) -> tuple[str, str, Any] | tuple[str, str]:
        match = self.CONDITION_PATTERN.match(rule.strip())
        if not match:
            return ("unknown", rule)

        condition_type = match.group(1)
        condition_value = match.group(2)

        op_match = self.OPERATOR_PATTERN.match(condition_value)
        if op_match:
            field = op_match.group(1).strip()
            operator = op_match.group(2)
            value = op_match.group(3).strip().strip("\"'")
            return (condition_type, field, operator, value)

        return (condition_type, condition_value)


class ApplicabilityEvaluator:
    def __init__(self) -> None:
        self._parser = RuleParser()

    def evaluate_lesson(
        self,
        lesson: CuratedLesson,
        context: ApplicabilityContext,
    ) -> ApplicabilityResult:
        if not lesson.is_active():
            return ApplicabilityResult(
                lesson_id=lesson.lesson_id,
                applicable=False,
                rule_results=[],
                match_score=0.0,
                matched_rules=[],
                failed_rules=["lesson_not_active"],
            )

        if lesson.applies_to_agents and context.agent_id not in lesson.applies_to_agents:
            return ApplicabilityResult(
                lesson_id=lesson.lesson_id,
                applicable=False,
                rule_results=[],
                match_score=0.0,
                matched_rules=[],
                failed_rules=["agent_not_in_applies_to"],
            )

        if lesson.experiment_id is not None:
            if context.experiment_id != lesson.experiment_id:
                return ApplicabilityResult(
                    lesson_id=lesson.lesson_id,
                    applicable=False,
                    rule_results=[],
                    match_score=0.0,
                    matched_rules=[],
                    failed_rules=["experiment_mismatch"],
                )

        if not lesson.applicability_rules:
            return ApplicabilityResult(
                lesson_id=lesson.lesson_id,
                applicable=True,
                rule_results=[],
                match_score=1.0,
                matched_rules=["no_rules"],
                failed_rules=[],
            )

        rule_results = []
        for rule in lesson.applicability_rules:
            result = self._evaluate_rule(rule, context)
            rule_results.append(result)

        passed = all(r.passed for r in rule_results)
        matched = [r.rule for r in rule_results if r.passed]
        failed = [r.rule for r in rule_results if not r.passed]

        match_score = len(matched) / len(rule_results) if rule_results else 1.0

        return ApplicabilityResult(
            lesson_id=lesson.lesson_id,
            applicable=passed,
            rule_results=rule_results,
            match_score=match_score,
            matched_rules=matched,
            failed_rules=failed,
        )

    def _evaluate_rule(self, rule: str, context: ApplicabilityContext) -> RuleEvaluationResult:
        parsed = self._parser.parse(rule)

        if parsed[0] == "unknown":
            return RuleEvaluationResult(
                rule=rule,
                passed=True,
                reason="Unknown rule format, passing by default",
            )

        condition_type = parsed[0]

        if condition_type == "agent":
            return self._evaluate_agent_rule(rule, parsed[1], context)
        elif condition_type == "task":
            return self._evaluate_task_rule(rule, parsed[1], context)
        elif condition_type == "tool":
            return self._evaluate_tool_rule(rule, parsed[1], context)
        elif condition_type == "error":
            return self._evaluate_error_rule(rule, parsed[1], context)
        elif condition_type == "experiment":
            return self._evaluate_experiment_rule(rule, parsed[1], context)
        elif condition_type == "strategy":
            return self._evaluate_strategy_rule(rule, parsed[1], context)
        elif condition_type == "meta":
            return self._evaluate_meta_rule(rule, parsed[1], context)

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason=f"Unknown condition type: {condition_type}",
        )

    def _evaluate_agent_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if isinstance(condition, str):
            passed = context.agent_id == condition or condition in context.agent_id
            return RuleEvaluationResult(
                rule=rule,
                passed=passed,
                reason=f"Agent {context.agent_id} {'matches' if passed else 'does not match'} {condition}",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Complex agent rule passed",
        )

    def _evaluate_task_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if context.task_type is None:
            return RuleEvaluationResult(
                rule=rule,
                passed=False,
                reason="No task type in context",
            )

        if isinstance(condition, str):
            passed = condition.lower() in context.task_type.lower()
            return RuleEvaluationResult(
                rule=rule,
                passed=passed,
                reason=f"Task type {context.task_type} {'matches' if passed else 'does not match'} {condition}",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Task rule evaluation passed",
        )

    def _evaluate_tool_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if isinstance(condition, str):
            passed = condition in context.tool_context
            return RuleEvaluationResult(
                rule=rule,
                passed=passed,
                reason=f"Tool {condition} {'is' if passed else 'is not'} in context",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Tool rule evaluation passed",
        )

    def _evaluate_error_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if isinstance(condition, str):
            pattern = condition.lower()
            for error in context.error_patterns:
                if pattern in error.lower():
                    return RuleEvaluationResult(
                        rule=rule,
                        passed=True,
                        reason=f"Error pattern '{pattern}' found in '{error}'",
                    )
            return RuleEvaluationResult(
                rule=rule,
                passed=False,
                reason=f"Error pattern '{pattern}' not found in recent errors",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Error rule evaluation passed",
        )

    def _evaluate_experiment_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if isinstance(condition, str):
            passed = context.experiment_id == condition
            return RuleEvaluationResult(
                rule=rule,
                passed=passed,
                reason=f"Experiment {context.experiment_id or 'none'} {'matches' if passed else 'does not match'} {condition}",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Experiment rule evaluation passed",
        )

    def _evaluate_strategy_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if isinstance(condition, str):
            passed = condition in context.strategy_focus
            return RuleEvaluationResult(
                rule=rule,
                passed=passed,
                reason=f"Strategy '{condition}' {'is' if passed else 'is not'} in focus",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Strategy rule evaluation passed",
        )

    def _evaluate_meta_rule(
        self,
        rule: str,
        condition: str | tuple,
        context: ApplicabilityContext,
    ) -> RuleEvaluationResult:
        if isinstance(condition, tuple) and len(condition) >= 4:
            _, field, operator, value = condition
            actual = context.metadata.get(field)

            if actual is None:
                return RuleEvaluationResult(
                    rule=rule,
                    passed=False,
                    reason=f"Metadata field '{field}' not found",
                )

            passed = self._compare_values(actual, operator, value)
            return RuleEvaluationResult(
                rule=rule,
                passed=passed,
                reason=f"Metadata {field}={actual} {operator} {value}: {'passed' if passed else 'failed'}",
            )

        return RuleEvaluationResult(
            rule=rule,
            passed=True,
            reason="Meta rule evaluation passed",
        )

    def _compare_values(self, actual: Any, operator: str, expected: str) -> bool:
        try:
            if operator == "=":
                return str(actual) == expected
            elif operator == "!=":
                return str(actual) != expected
            elif operator == "~=":
                return expected.lower() in str(actual).lower()
            elif operator == ">":
                return float(actual) > float(expected)
            elif operator == "<":
                return float(actual) < float(expected)
            elif operator == ">=":
                return float(actual) >= float(expected)
            elif operator == "<=":
                return float(actual) <= float(expected)
        except (ValueError, TypeError):
            return False
        return False


class LessonApplicabilityEngine:
    def __init__(self) -> None:
        self._evaluator = ApplicabilityEvaluator()

    def filter_applicable_lessons(
        self,
        lessons: list[CuratedLesson],
        context: ApplicabilityContext,
    ) -> list[tuple[CuratedLesson, ApplicabilityResult]]:
        results = []
        for lesson in lessons:
            result = self._evaluator.evaluate_lesson(lesson, context)
            if result.applicable:
                results.append((lesson, result))
        return results

    def get_best_matching_lessons(
        self,
        lessons: list[CuratedLesson],
        context: ApplicabilityContext,
        min_score: float = 0.5,
        limit: int = 10,
    ) -> list[tuple[CuratedLesson, ApplicabilityResult]]:
        scored = []
        for lesson in lessons:
            result = self._evaluator.evaluate_lesson(lesson, context)
            if result.match_score >= min_score:
                scored.append((lesson, result))

        scored.sort(key=lambda x: x[1].match_score, reverse=True)
        return scored[:limit]

    def explain_applicability(
        self,
        lesson: CuratedLesson,
        context: ApplicabilityContext,
    ) -> str:
        result = self._evaluator.evaluate_lesson(lesson, context)

        lines = [
            f"Lesson: {lesson.lesson_id}",
            f"Applicable: {result.applicable}",
            f"Match Score: {result.match_score:.2f}",
        ]

        if result.matched_rules:
            lines.append(f"Matched Rules: {', '.join(result.matched_rules)}")

        if result.failed_rules:
            lines.append(f"Failed Rules: {', '.join(result.failed_rules)}")

        for rr in result.rule_results:
            status = "✓" if rr.passed else "✗"
            lines.append(f"  {status} {rr.rule}: {rr.reason}")

        return "\n".join(lines)


__all__ = [
    "ApplicabilityContext",
    "ApplicabilityEvaluator",
    "ApplicabilityResult",
    "LessonApplicabilityEngine",
    "RuleEvaluationResult",
    "RuleParser",
]
