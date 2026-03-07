"""Gap scorecard generator for eval coverage analysis."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ash_hawk.reporting.scorecard_types import (
    DEFAULT_AGENT_TARGETS,
    PRIORITY_WEIGHTS,
    AgentDepth,
    GapDiff,
    GapScorecard,
    Requirement,
    RequirementCoverage,
)


class GapScorecardGenerator:
    """Generator for gap analysis scorecards."""

    DEFAULT_REQUIREMENTS: list[Requirement] = [
        Requirement(
            req_id="SEC-001",
            dimension="security_depth",
            priority="critical",
            description="SQL injection detection tests",
            agents=("security",),
            keyword_any=("sql", "injection", "query"),
            required_graders=("llm_judge",),
            minimum_matches=2,
        ),
        Requirement(
            req_id="SEC-002",
            dimension="security_depth",
            priority="critical",
            description="XSS vulnerability detection",
            agents=("security",),
            keyword_any=("xss", "cross-site", "script"),
            required_graders=("llm_judge",),
            minimum_matches=2,
        ),
        Requirement(
            req_id="SEC-003",
            dimension="security_depth",
            priority="critical",
            description="Hardcoded secrets detection",
            agents=("security",),
            keyword_any=("secret", "password", "api_key", "token"),
            required_graders=("llm_judge",),
            minimum_matches=2,
        ),
        Requirement(
            req_id="ARCH-001",
            dimension="architecture_depth",
            priority="high",
            description="Boundary violation detection",
            agents=("architecture",),
            keyword_any=("boundary", "layer", "violation"),
            required_graders=("llm_judge",),
            minimum_matches=1,
        ),
        Requirement(
            req_id="ARCH-002",
            dimension="architecture_depth",
            priority="high",
            description="Circular dependency detection",
            agents=("architecture",),
            keyword_any=("circular", "dependency", "import"),
            required_graders=("llm_judge",),
            minimum_matches=1,
        ),
        Requirement(
            req_id="DELEG-001",
            dimension="delegation_quality",
            priority="high",
            description="Subagent delegation tests",
            agents=("general", "orchestrator"),
            keyword_any=("delegate", "subagent", "task("),
            required_graders=("llm_judge", "delegation"),
            minimum_matches=2,
        ),
        Requirement(
            req_id="TOOL-001",
            dimension="tool_behavior",
            priority="medium",
            description="Tool call verification tests",
            agents=(),
            required_graders=("tool_call",),
            minimum_matches=3,
        ),
        Requirement(
            req_id="PROMPT-001",
            dimension="prompt_robustness",
            priority="medium",
            description="Vague/ambiguous prompt handling",
            agents=(),
            keyword_any=("vague", "ambiguous", "clarif"),
            required_graders=("llm_judge",),
            minimum_matches=2,
        ),
        Requirement(
            req_id="RUNTIME-001",
            dimension="runtime_reliability",
            priority="medium",
            description="Timeout handling tests",
            agents=(),
            keyword_any=("timeout", "deadline", "hang"),
            required_graders=("transcript",),
            minimum_matches=1,
        ),
    ]

    def __init__(
        self,
        requirements: list[Requirement] | None = None,
        agent_targets: dict[str, int] | None = None,
    ) -> None:
        self._requirements = requirements or self.DEFAULT_REQUIREMENTS
        self._agent_targets = agent_targets or DEFAULT_AGENT_TARGETS

    def analyze_suite(self, suite: dict[str, Any] | Any) -> GapScorecard:
        """Analyze an eval suite for coverage gaps.

        Args:
            suite: EvalSuite object or dict with 'id' and 'tasks'

        Returns:
            GapScorecard with coverage analysis
        """
        if isinstance(suite, dict):
            suite_id = suite.get("id", "unknown-suite")
            tasks = suite.get("tasks", [])
        else:
            suite_id = getattr(suite, "id", "unknown-suite")
            tasks = getattr(suite, "tasks", [])

        # Analyze requirement coverage
        requirement_coverage = self._analyze_requirements(tasks)

        # Analyze agent depth
        agent_depth = self._analyze_agent_depth(tasks)

        # Compute dimension scores
        dimension_scores = self._compute_dimension_scores(requirement_coverage)

        # Compute overall score
        overall_score = self._compute_overall_score(requirement_coverage)

        # Generate recommendations
        recommendations = self._generate_recommendations(requirement_coverage, agent_depth)

        return GapScorecard(
            suite_id=suite_id,
            generated_at=datetime.now(UTC).isoformat(),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            requirement_coverage=requirement_coverage,
            agent_depth=agent_depth,
            blueprint_recommendations=recommendations,
            total_tasks=len(tasks),
            total_requirements=len(self._requirements),
            covered_requirements=sum(1 for rc in requirement_coverage if rc.covered),
        )

    def _analyze_requirements(self, tasks: list[Any]) -> list[RequirementCoverage]:
        """Check which requirements are covered by tasks."""
        coverage = []

        for req in self._requirements:
            matched_tasks = self._find_matching_tasks(tasks, req)
            covered = len(matched_tasks) >= req.minimum_matches
            ratio = min(len(matched_tasks) / max(req.minimum_matches, 1), 1.0)

            coverage.append(
                RequirementCoverage(
                    requirement=req,
                    covered=covered,
                    matched_tasks=matched_tasks,
                    coverage_ratio=ratio,
                )
            )

        return coverage

    def _find_matching_tasks(self, tasks: list[Any], req: Requirement) -> list[str]:
        """Find tasks that match a requirement."""
        matched = []

        for task in tasks:
            if isinstance(task, dict):
                task_id = task.get("id", "unknown")
                description = task.get("description", "").lower()
                tags = [t.lower() for t in task.get("tags", [])]
                agents = [task.get("input", {}).get("agent", "")]
                grader_types = [g.get("grader_type", "") for g in task.get("grader_specs", [])]
            else:
                task_id = getattr(task, "id", "unknown")
                description = getattr(task, "description", "").lower()
                tags = [t.lower() for t in getattr(task, "tags", [])]
                task_input = getattr(task, "input", {})
                agents = [task_input.get("agent", "")] if isinstance(task_input, dict) else []
                grader_specs = getattr(task, "grader_specs", [])
                grader_types = [getattr(g, "grader_type", "") for g in grader_specs]

            # Check agent match
            if req.agents:
                if not any(a in agents for a in req.agents):
                    continue

            # Check keyword match
            if req.keyword_any:
                search_text = description + " " + " ".join(tags)
                if not any(kw.lower() in search_text for kw in req.keyword_any):
                    continue

            # Check grader match
            if req.required_graders:
                if not any(gt in grader_types for gt in req.required_graders):
                    continue

            matched.append(task_id)

        return matched

    def _analyze_agent_depth(self, tasks: list[Any]) -> list[AgentDepth]:
        """Count tasks per agent."""
        agent_counts: dict[str, int] = {}

        for task in tasks:
            if isinstance(task, dict):
                agent = task.get("input", {}).get("agent", "general")
            else:
                task_input = getattr(task, "input", {})
                agent = (
                    task_input.get("agent", "general")
                    if isinstance(task_input, dict)
                    else "general"
                )

            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        depth = []
        for agent, target in self._agent_targets.items():
            count = agent_counts.get(agent, 0)
            depth.append(AgentDepth(agent=agent, task_count=count, target_count=target))

        return depth

    def _compute_dimension_scores(self, coverage: list[RequirementCoverage]) -> dict[str, float]:
        """Compute scores per dimension."""
        dimension_items: dict[str, list[RequirementCoverage]] = {}

        for rc in coverage:
            dim = rc.requirement.dimension
            if dim not in dimension_items:
                dimension_items[dim] = []
            dimension_items[dim].append(rc)

        scores = {}
        for dim, items in dimension_items.items():
            total_weight = sum(PRIORITY_WEIGHTS.get(rc.requirement.priority, 1) for rc in items)
            covered_weight = sum(
                PRIORITY_WEIGHTS.get(rc.requirement.priority, 1) * rc.coverage_ratio for rc in items
            )
            scores[dim] = covered_weight / total_weight if total_weight > 0 else 0.0

        return scores

    def _compute_overall_score(self, coverage: list[RequirementCoverage]) -> float:
        """Compute overall coverage score."""
        total_weight = sum(PRIORITY_WEIGHTS.get(rc.requirement.priority, 1) for rc in coverage)
        covered_weight = sum(
            PRIORITY_WEIGHTS.get(rc.requirement.priority, 1) * rc.coverage_ratio for rc in coverage
        )
        return covered_weight / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(
        self,
        coverage: list[RequirementCoverage],
        agent_depth: list[AgentDepth],
    ) -> list[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []

        # Uncovered requirements
        for rc in coverage:
            if not rc.covered:
                priority = rc.requirement.priority.upper()
                recommendations.append(f"[{priority}] Add tests for: {rc.requirement.description}")

        # Agent gaps
        for ad in agent_depth:
            if ad.gap > 0:
                recommendations.append(
                    f"[GAP] Add {ad.gap} more tasks for agent '{ad.agent}' (current: {ad.task_count}, target: {ad.target_count})"
                )

        return recommendations

    def compare_baseline(self, current: GapScorecard, baseline: GapScorecard) -> GapDiff:
        """Compare current scorecard against baseline."""
        return GapDiff(
            suite_id=current.suite_id,
            overall_score_delta=current.overall_score - baseline.overall_score,
            dimension_deltas={
                dim: current.dimension_scores.get(dim, 0) - baseline.dimension_scores.get(dim, 0)
                for dim in set(current.dimension_scores) | set(baseline.dimension_scores)
            },
            agent_depth_deltas={
                ad.agent: ad.task_count
                - next((bad.task_count for bad in baseline.agent_depth if bad.agent == ad.agent), 0)
                for ad in current.agent_depth
            },
            new_covered_requirements=[
                rc.requirement.req_id
                for rc in current.requirement_coverage
                if rc.covered
                and not any(
                    brc.requirement.req_id == rc.requirement.req_id and brc.covered
                    for brc in baseline.requirement_coverage
                )
            ],
            regression_requirements=[
                rc.requirement.req_id
                for rc in current.requirement_coverage
                if not rc.covered
                and any(
                    brc.requirement.req_id == rc.requirement.req_id and brc.covered
                    for brc in baseline.requirement_coverage
                )
            ],
            comparison_time=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def to_markdown(scorecard: GapScorecard) -> str:
        """Convert scorecard to markdown format."""
        lines = [
            f"# Gap Scorecard: {scorecard.suite_id}",
            "",
            f"**Generated**: {scorecard.generated_at}",
            f"**Overall Score**: {scorecard.overall_score:.2%}",
            f"**Requirements**: {scorecard.covered_requirements}/{scorecard.total_requirements} covered",
            "",
            "## Dimension Scores",
            "",
        ]

        for dim, score in sorted(scorecard.dimension_scores.items()):
            status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
            lines.append(f"- {status} **{dim}**: {score:.2%}")

        lines.extend(["", "## Agent Depth", ""])

        for ad in scorecard.agent_depth:
            status = "✅" if ad.score >= 1.0 else "⚠️" if ad.score >= 0.5 else "❌"
            lines.append(f"- {status} **{ad.agent}**: {ad.task_count}/{ad.target_count} tasks")

        if scorecard.blueprint_recommendations:
            lines.extend(["", "## Recommendations", ""])
            for rec in scorecard.blueprint_recommendations[:10]:
                lines.append(f"- {rec}")

        return "\n".join(lines)

    @staticmethod
    def to_json(scorecard: GapScorecard) -> dict[str, Any]:
        """Convert scorecard to JSON-compatible dict."""
        return scorecard.to_dict()


def load_scorecard(path: Path) -> GapScorecard:
    """Load a scorecard from JSON file."""
    data = json.loads(path.read_text())
    return GapScorecard(
        suite_id=data["suite_id"],
        generated_at=data["generated_at"],
        overall_score=data["overall_score"],
        dimension_scores=data["dimension_scores"],
        requirement_coverage=[
            RequirementCoverage(
                requirement=Requirement(
                    req_id=rc["req_id"],
                    dimension=rc["dimension"],
                    priority=rc["priority"],
                    description=rc.get("description", ""),
                    agents=tuple(rc.get("agents", ())),
                ),
                covered=rc["covered"],
                matched_tasks=rc["matched_tasks"],
                coverage_ratio=rc.get("coverage_ratio", 0.0),
            )
            for rc in data.get("requirement_coverage", [])
        ],
        agent_depth=[
            AgentDepth(
                agent=ad["agent"],
                task_count=ad["task_count"],
                target_count=ad["target_count"],
            )
            for ad in data.get("agent_depth", [])
        ],
        blueprint_recommendations=data.get("blueprint_recommendations", []),
        total_tasks=data.get("total_tasks", 0),
        total_requirements=data.get("total_requirements", 0),
        covered_requirements=data.get("covered_requirements", 0),
    )


__all__ = ["GapScorecardGenerator", "load_scorecard"]
