"""Tests for gap scorecard module."""

from __future__ import annotations

import pytest

from ash_hawk.reporting.gap_scorecard import GapScorecardGenerator
from ash_hawk.reporting.scorecard_types import (
    AgentDepth,
    GapScorecard,
    Requirement,
    RequirementCoverage,
)


class TestGapScorecardGenerator:
    """Tests for GapScorecardGenerator."""

    def test_analyze_empty_suite(self) -> None:
        generator = GapScorecardGenerator()
        suite = {"id": "test-suite", "tasks": []}
        scorecard = generator.analyze_suite(suite)

        assert scorecard.suite_id == "test-suite"
        assert scorecard.total_tasks == 0
        assert scorecard.overall_score == 0.0

    def test_analyze_suite_with_tasks(self) -> None:
        generator = GapScorecardGenerator()
        suite = {
            "id": "test-suite",
            "tasks": [
                {
                    "id": "task-1",
                    "description": "Test SQL injection detection",
                    "tags": ["security"],
                    "input": {"agent": "security"},
                    "grader_specs": [{"grader_type": "llm_judge"}],
                },
                {
                    "id": "task-2",
                    "description": "Check for XSS vulnerability",
                    "tags": ["security"],
                    "input": {"agent": "security"},
                    "grader_specs": [{"grader_type": "llm_judge"}],
                },
            ],
        }
        scorecard = generator.analyze_suite(suite)

        assert scorecard.total_tasks == 2
        assert scorecard.agent_depth[0].agent == "security"
        assert scorecard.agent_depth[0].task_count == 2

    def test_dimension_scores_computed(self) -> None:
        generator = GapScorecardGenerator()
        suite = {
            "id": "test-suite",
            "tasks": [
                {
                    "id": "task-1",
                    "description": "SQL injection test",
                    "input": {"agent": "security"},
                    "grader_specs": [{"grader_type": "llm_judge"}],
                }
            ],
        }
        scorecard = generator.analyze_suite(suite)

        assert "security_depth" in scorecard.dimension_scores

    def test_blueprint_recommendations(self) -> None:
        generator = GapScorecardGenerator()
        suite = {"id": "test-suite", "tasks": []}
        scorecard = generator.analyze_suite(suite)

        assert len(scorecard.blueprint_recommendations) > 0

    def test_to_markdown(self) -> None:
        generator = GapScorecardGenerator()
        suite = {"id": "test-suite", "tasks": []}
        scorecard = generator.analyze_suite(suite)

        md = GapScorecardGenerator.to_markdown(scorecard)
        assert "# Gap Scorecard" in md
        assert "test-suite" in md

    def test_to_json(self) -> None:
        generator = GapScorecardGenerator()
        suite = {"id": "test-suite", "tasks": []}
        scorecard = generator.analyze_suite(suite)

        data = GapScorecardGenerator.to_json(scorecard)
        assert data["suite_id"] == "test-suite"
        assert "overall_score" in data

    def test_compare_baseline(self) -> None:
        generator = GapScorecardGenerator()

        current = generator.analyze_suite(
            {
                "id": "suite",
                "tasks": [
                    {
                        "id": "t1",
                        "description": "SQL injection",
                        "input": {"agent": "security"},
                        "grader_specs": [{"grader_type": "llm_judge"}],
                    }
                ],
            }
        )

        baseline = generator.analyze_suite({"id": "suite", "tasks": []})

        diff = generator.compare_baseline(current, baseline)
        assert diff.overall_score_delta > 0


class TestRequirementCoverage:
    """Tests for RequirementCoverage dataclass."""

    def test_covered_when_matched(self) -> None:
        req = Requirement(
            req_id="TEST-001",
            dimension="test",
            priority="high",
            description="Test requirement",
            agents=(),
            minimum_matches=1,
        )
        coverage = RequirementCoverage(
            requirement=req,
            covered=True,
            matched_tasks=["task-1"],
            coverage_ratio=1.0,
        )
        assert coverage.covered is True


class TestAgentDepth:
    """Tests for AgentDepth dataclass."""

    def test_gap_calculation(self) -> None:
        depth = AgentDepth(agent="security", task_count=5, target_count=10)
        assert depth.gap == 5

    def test_no_gap_when_exceeded(self) -> None:
        depth = AgentDepth(agent="security", task_count=15, target_count=10)
        assert depth.gap == 0

    def test_score_calculation(self) -> None:
        depth = AgentDepth(agent="security", task_count=5, target_count=10)
        assert depth.score == 0.5

    def test_score_capped_at_one(self) -> None:
        depth = AgentDepth(agent="security", task_count=15, target_count=10)
        assert depth.score == 1.0
