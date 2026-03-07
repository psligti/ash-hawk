"""Types for gap scorecard coverage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Requirement:
    """A coverage requirement to check against eval suite."""

    req_id: str
    dimension: str
    priority: str
    description: str
    agents: tuple[str, ...]
    keyword_any: tuple[str, ...] = ()
    required_graders: tuple[str, ...] = ()
    minimum_matches: int = 1


@dataclass
class RequirementCoverage:
    """Coverage status for a single requirement."""

    requirement: Requirement
    covered: bool
    matched_tasks: list[str]
    coverage_ratio: float = 0.0


@dataclass
class AgentDepth:
    """Task count depth for a single agent."""

    agent: str
    task_count: int
    target_count: int

    @property
    def gap(self) -> int:
        return max(self.target_count - self.task_count, 0)

    @property
    def score(self) -> float:
        if self.target_count <= 0:
            return 1.0
        return min(self.task_count / self.target_count, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "task_count": self.task_count,
            "target_count": self.target_count,
            "gap": self.gap,
            "score": self.score,
        }


@dataclass
class GapScorecard:
    """Complete gap analysis scorecard for an eval suite."""

    suite_id: str
    generated_at: str
    overall_score: float
    dimension_scores: dict[str, float]
    requirement_coverage: list[RequirementCoverage]
    agent_depth: list[AgentDepth]
    blueprint_recommendations: list[str]
    total_tasks: int
    total_requirements: int
    covered_requirements: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "generated_at": self.generated_at,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "requirement_coverage": [
                {
                    "req_id": rc.requirement.req_id,
                    "dimension": rc.requirement.dimension,
                    "priority": rc.requirement.priority,
                    "covered": rc.covered,
                    "matched_tasks": rc.matched_tasks,
                    "coverage_ratio": rc.coverage_ratio,
                }
                for rc in self.requirement_coverage
            ],
            "agent_depth": [ad.to_dict() for ad in self.agent_depth],
            "blueprint_recommendations": self.blueprint_recommendations,
            "total_tasks": self.total_tasks,
            "total_requirements": self.total_requirements,
            "covered_requirements": self.covered_requirements,
        }


@dataclass
class GapDiff:
    """Difference between two scorecards for tracking progress."""

    suite_id: str
    overall_score_delta: float
    dimension_deltas: dict[str, float]
    agent_depth_deltas: dict[str, int]
    new_covered_requirements: list[str]
    regression_requirements: list[str]
    comparison_time: str


PRIORITY_WEIGHTS: dict[str, int] = {
    "critical": 3,
    "high": 2,
    "medium": 1,
    "low": 0,
}


DEFAULT_AGENT_TARGETS: dict[str, int] = {
    "security": 10,
    "architecture": 8,
    "documentation": 6,
    "unit_tests": 6,
    "linting": 6,
    "performance": 6,
    "general": 8,
    "explore": 5,
}


__all__ = [
    "Requirement",
    "RequirementCoverage",
    "AgentDepth",
    "GapScorecard",
    "GapDiff",
    "PRIORITY_WEIGHTS",
    "DEFAULT_AGENT_TARGETS",
]
