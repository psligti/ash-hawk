"""Data structures for improvement target selection.

This module provides the core data types for the improvement agent
to decide WHERE improvements should happen based on the control level
hierarchy: Agent (highest) > Skill (medium) > Tool (lowest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from ash_hawk.strategies import Strategy


class ControlLevel(IntEnum):
    """Control level for improvement targets.

    Higher values indicate more control over agent behavior.
    """

    TOOL = 1
    SKILL = 2
    AGENT = 3


CONTROL_LEVEL_TO_LESSON_TYPE: dict[ControlLevel, str] = {
    ControlLevel.AGENT: "policy",
    ControlLevel.SKILL: "skill",
    ControlLevel.TOOL: "tool",
}

CONTROL_LEVEL_TO_STRATEGY: dict[ControlLevel, Strategy] = {
    ControlLevel.AGENT: Strategy.POLICY_QUALITY,
    ControlLevel.SKILL: Strategy.SKILL_QUALITY,
    ControlLevel.TOOL: Strategy.TOOL_QUALITY,
}


@dataclass
class Finding:
    """Represents a finding from review analysis.

    Attributes:
        category: Category of the finding (tool_selection, instruction_clarity, etc.).
        severity: Severity level (low, medium, high, critical).
        description: Human-readable description.
        evidence: Evidence supporting the finding.
        affected_runs: Number of runs affected.
        suggested_control_level: Suggested control level for improvement.
    """

    category: str
    severity: str
    description: str
    evidence: list[str] = field(default_factory=list[str])
    affected_runs: int = 1
    suggested_control_level: ControlLevel | None = None


@dataclass
class ReviewMetrics:
    """Metrics from review analysis.

    Attributes:
        mean_score: Average score across trials.
        pass_rate: Fraction of trials passing.
        tool_efficiency: Ratio of useful to total tool calls.
        error_rate: Fraction of trials with errors.
        timeout_rate: Fraction of trials that timed out.
        calibration_error: ECE if available.
    """

    mean_score: float = 0.0
    pass_rate: float = 0.0
    tool_efficiency: float = 1.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    calibration_error: float | None = None


__all__ = [
    "ControlLevel",
    "Finding",
    "ReviewMetrics",
    "CONTROL_LEVEL_TO_LESSON_TYPE",
    "CONTROL_LEVEL_TO_STRATEGY",
]
