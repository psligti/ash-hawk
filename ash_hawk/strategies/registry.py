"""Strategy registry for hierarchical improvement focus areas."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Strategy(StrEnum):
    """Top-level improvement strategies.

    Each strategy represents a major area of agent improvement focus.
    Lessons, proposals, and evals can be tagged with these strategies
    for targeted improvement and parallel experimentation.
    """

    POLICY_QUALITY = "policy-quality"
    SKILL_QUALITY = "skill-quality"
    TOOL_QUALITY = "tool-quality"
    HARNESS_QUALITY = "harness-quality"
    EVAL_QUALITY = "eval-quality"
    AGENT_BEHAVIOR = "agent-behavior"


class SubStrategy(StrEnum):
    """Sub-strategies for granular improvement focus.

    Each sub-strategy belongs to a parent strategy and represents
    a specific aspect that can be improved independently.
    """

    ENGAGEMENT_POLICY = "engagement-policy"
    RANKING_POLICY = "ranking-policy"
    STRATEGY_BUDGET = "strategy-budget"
    TOOL_ACCESS = "tool-access"

    INSTRUCTION_CLARITY = "instruction-clarity"
    EXAMPLE_QUALITY = "example-quality"
    CONTEXT_RELEVANCE = "context-relevance"
    VOICE_TONE = "voice-tone"
    PLAYBOOK_ADHERENCE = "playbook-adherence"
    CLASSIFICATION_ACCURACY = "classification-accuracy"
    FALSE_POSITIVE_RATE = "false-positive-rate"

    TOOL_EFFICIENCY = "tool-efficiency"
    TOOL_SELECTION = "tool-selection"
    ERROR_RECOVERY = "error-recovery"
    RETRY_BEHAVIOR = "retry-behavior"
    REPO_INSPECTION = "repo-inspection"

    GRADER_CALIBRATION = "grader-calibration"
    TIMEOUT_TUNING = "timeout-tuning"
    FIXTURE_DESIGN = "fixture-design"

    RUBRIC_PRECISION = "rubric-precision"
    TEST_COVERAGE = "test-coverage"
    EVAL_FALSE_POSITIVE_RATE = "eval-false-positive-rate"

    EVIDENCE_QUALITY = "evidence-quality"
    TASK_COMPLETION = "task-completion"
    CHANGE_PRECISION = "change-precision"
    SAFETY_QUALITY = "safety-quality"
    ENGAGEMENT_PROXY = "engagement-proxy"


STRATEGY_HIERARCHY: dict[Strategy, list[SubStrategy]] = {
    Strategy.POLICY_QUALITY: [
        SubStrategy.ENGAGEMENT_POLICY,
        SubStrategy.RANKING_POLICY,
        SubStrategy.STRATEGY_BUDGET,
        SubStrategy.TOOL_ACCESS,
    ],
    Strategy.SKILL_QUALITY: [
        SubStrategy.INSTRUCTION_CLARITY,
        SubStrategy.EXAMPLE_QUALITY,
        SubStrategy.CONTEXT_RELEVANCE,
        SubStrategy.VOICE_TONE,
        SubStrategy.PLAYBOOK_ADHERENCE,
    ],
    Strategy.TOOL_QUALITY: [
        SubStrategy.TOOL_EFFICIENCY,
        SubStrategy.TOOL_SELECTION,
        SubStrategy.ERROR_RECOVERY,
        SubStrategy.RETRY_BEHAVIOR,
        SubStrategy.REPO_INSPECTION,
    ],
    Strategy.HARNESS_QUALITY: [
        SubStrategy.GRADER_CALIBRATION,
        SubStrategy.TIMEOUT_TUNING,
        SubStrategy.FIXTURE_DESIGN,
    ],
    Strategy.EVAL_QUALITY: [
        SubStrategy.RUBRIC_PRECISION,
        SubStrategy.TEST_COVERAGE,
        SubStrategy.EVAL_FALSE_POSITIVE_RATE,
    ],
    Strategy.AGENT_BEHAVIOR: [
        SubStrategy.EVIDENCE_QUALITY,
        SubStrategy.TASK_COMPLETION,
        SubStrategy.CHANGE_PRECISION,
        SubStrategy.SAFETY_QUALITY,
        SubStrategy.ENGAGEMENT_PROXY,
    ],
}


def get_parent_strategy(sub_strategy: SubStrategy) -> Strategy | None:
    """Get the parent strategy for a sub-strategy."""
    for strategy, subs in STRATEGY_HIERARCHY.items():
        if sub_strategy in subs:
            return strategy
    return None


def get_sub_strategies(strategy: Strategy) -> list[SubStrategy]:
    """Get all sub-strategies for a strategy."""
    return STRATEGY_HIERARCHY.get(strategy, [])


def validate_strategy_pair(strategy: Strategy, sub_strategy: SubStrategy) -> bool:
    """Validate that a sub-strategy belongs to a strategy."""
    return sub_strategy in STRATEGY_HIERARCHY.get(strategy, [])


STRATEGY_TO_LESSON_TYPE: dict[Strategy, list[str]] = {
    Strategy.POLICY_QUALITY: ["policy"],
    Strategy.SKILL_QUALITY: ["skill"],
    Strategy.TOOL_QUALITY: ["tool"],
    Strategy.HARNESS_QUALITY: ["harness"],
    Strategy.EVAL_QUALITY: ["eval", "harness"],
    Strategy.AGENT_BEHAVIOR: ["policy", "skill", "tool"],
}


def get_compatible_lesson_types(strategy: Strategy) -> list[str]:
    """Get lesson types compatible with a strategy."""
    return STRATEGY_TO_LESSON_TYPE.get(strategy, [])
