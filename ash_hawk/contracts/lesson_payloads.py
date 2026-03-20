"""Typed payload schemas for curated lessons by improvement type."""

from __future__ import annotations

from typing import Any, Literal

import pydantic as pd


class PolicyLessonPayload(pd.BaseModel):
    """Lesson payload for policy engine improvements.

    Used to inject new rules or modify existing policy behavior
    for Vox Jay policy adapters (engagement, ranking, strategy, budget).

    Attributes:
        rule_name: Unique identifier for this policy rule.
        rule_type: Category of policy (engagement, ranking, strategy, budget).
        condition: When this rule should apply (e.g., score thresholds, patterns).
        action: What to do when condition is met (e.g., boost score, demote).
        priority: Execution priority (higher = applied first).
        enabled: Whether this rule is active.
    """

    rule_name: str = pd.Field(
        description="Unique identifier for this policy rule",
    )
    rule_type: Literal["engagement", "ranking", "strategy", "budget", "tool"] = pd.Field(
        description="Category of policy rule",
    )
    condition: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="When this rule should apply",
    )
    action: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="What to do when condition is met",
    )
    priority: int = pd.Field(
        default=50,
        ge=0,
        le=100,
        description="Execution priority (higher = applied first)",
    )
    enabled: bool = pd.Field(
        default=True,
        description="Whether this rule is active",
    )

    model_config = pd.ConfigDict(extra="forbid")


class SkillLessonPayload(pd.BaseModel):
    """Lesson payload for skill/instruction improvements.

    Used to inject new instructions, remove outdated guidance,
    or add examples to agent system prompts.

    Attributes:
        skill_name: Name of the skill or behavior being modified.
        instruction_additions: New instructions to inject into prompt.
        instruction_removals: Pattern fragments to remove from prompt.
        examples: New few-shot examples to add.
        context_triggers: Keywords that activate this skill.
    """

    skill_name: str = pd.Field(
        description="Name of the skill or behavior being modified",
    )
    instruction_additions: list[str] = pd.Field(
        default_factory=list,
        description="New instructions to inject into prompt",
    )
    instruction_removals: list[str] = pd.Field(
        default_factory=list,
        description="Pattern fragments to remove from prompt",
    )
    examples: list[dict[str, str]] = pd.Field(
        default_factory=list,
        description="New few-shot examples to add",
    )
    context_triggers: list[str] = pd.Field(
        default_factory=list,
        description="Keywords that activate this skill",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ToolLessonPayload(pd.BaseModel):
    """Lesson payload for tool usage improvements.

    Used to adjust tool parameters, add usage hints, or
    override timeouts for specific tools.

    Attributes:
        tool_id: Identifier of the tool being modified.
        parameter_defaults: Default parameter values to apply.
        usage_hints: Guidance strings shown before tool use.
        timeout_override: Custom timeout in seconds (None = no override).
        preconditions: Conditions that must be met before using tool.
        postconditions: Expected outcomes after tool use.
    """

    tool_id: str = pd.Field(
        description="Identifier of the tool being modified",
    )
    parameter_defaults: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Default parameter values to apply",
    )
    usage_hints: list[str] = pd.Field(
        default_factory=list,
        description="Guidance strings shown before tool use",
    )
    timeout_override: int | None = pd.Field(
        default=None,
        ge=1,
        description="Custom timeout in seconds",
    )
    preconditions: list[str] = pd.Field(
        default_factory=list,
        description="Conditions that must be met before using tool",
    )
    postconditions: list[str] = pd.Field(
        default_factory=list,
        description="Expected outcomes after tool use",
    )

    model_config = pd.ConfigDict(extra="forbid")


class HarnessLessonPayload(pd.BaseModel):
    """Lesson payload for eval harness improvements.

    Used to adjust grader weights, thresholds, timeouts,
    or fixture configurations for evaluation suites.

    Attributes:
        suite_id: Target evaluation suite identifier.
        grader_adjustments: Weight/threshold changes per grader.
        fixture_overrides: Custom fixture values.
        timeout_adjustments: Timeout changes per task or global.
        parallelism_override: Custom parallelism setting.
    """

    suite_id: str | None = pd.Field(
        default=None,
        description="Target evaluation suite identifier (None applies globally)",
    )
    grader_adjustments: dict[str, dict[str, float | int | str]] = pd.Field(
        default_factory=dict,
        description="Weight/threshold changes per grader",
    )
    fixture_overrides: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Custom fixture values",
    )
    timeout_adjustments: dict[str, int] = pd.Field(
        default_factory=dict,
        description="Timeout changes per task or global",
    )
    parallelism_override: int | None = pd.Field(
        default=None,
        ge=1,
        description="Custom parallelism setting",
    )

    model_config = pd.ConfigDict(extra="forbid")


class EvalLessonPayload(pd.BaseModel):
    """Lesson payload for evaluation suite improvements.

    Used to add new test cases, adjust rubrics, or
    modify evaluation criteria.

    Attributes:
        eval_id: Target evaluation identifier.
        rubric_additions: New rubric criteria to add.
        test_case_additions: New test cases to include.
        threshold_adjustments: Pass/fail threshold changes.
        weight_adjustments: Task weight changes.
    """

    eval_id: str = pd.Field(
        description="Target evaluation identifier",
    )
    rubric_additions: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="New rubric criteria to add",
    )
    test_case_additions: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="New test cases to include",
    )
    threshold_adjustments: dict[str, float] = pd.Field(
        default_factory=dict,
        description="Pass/fail threshold changes",
    )
    weight_adjustments: dict[str, float] = pd.Field(
        default_factory=dict,
        description="Task weight changes",
    )

    model_config = pd.ConfigDict(extra="forbid")


# Type alias for all payload types
LessonPayload = (
    PolicyLessonPayload
    | SkillLessonPayload
    | ToolLessonPayload
    | HarnessLessonPayload
    | EvalLessonPayload
)


def parse_lesson_payload(
    lesson_type: str,
    payload_dict: dict[str, Any],
) -> LessonPayload | None:
    """Parse a lesson payload dict into the appropriate typed model.

    Args:
        lesson_type: The type of lesson (policy, skill, tool, harness, eval).
        payload_dict: Raw payload dictionary.

    Returns:
        Typed payload model, or None if type is unknown.
    """
    parsers: dict[str, type[LessonPayload]] = {
        "policy": PolicyLessonPayload,
        "skill": SkillLessonPayload,
        "tool": ToolLessonPayload,
        "harness": HarnessLessonPayload,
        "eval": EvalLessonPayload,
    }

    parser_class = parsers.get(lesson_type)
    if parser_class is None:
        return None

    try:
        return parser_class(**payload_dict)
    except pd.ValidationError:
        return None
