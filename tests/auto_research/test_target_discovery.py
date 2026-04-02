"""Tests for TargetDiscovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ash_hawk.auto_research.target_discovery import (
    AGENT_SEARCH_DIRS,
    POLICY_SEARCH_DIRS,
    SKILL_SEARCH_DIRS,
    TOOL_SEARCH_DIRS,
    TargetDiscovery,
    _compute_low_score_threshold,
)
from ash_hawk.auto_research.types import ImprovementTarget, TargetType
from ash_hawk.services.dawn_kestrel_injector import DawnKestrelInjector


def _make_injector() -> DawnKestrelInjector:
    mock_injector = MagicMock(spec=DawnKestrelInjector)
    mock_injector.get_agent_path.return_value = Path(".dawn-kestrel/agents/test-agent")
    mock_injector.get_policy_path.return_value = Path(".dawn-kestrel/policies/test-policy")
    mock_injector.get_skill_path.return_value = Path(".dawn-kestrel/skills/test-skill")
    mock_injector.get_tool_path.return_value = Path(".dawn-kestrel/tools/test-tool")
    return mock_injector


def _make_target(
    target_type: TargetType,
    name: str,
    discovered_path: Path,
) -> ImprovementTarget:
    mock_injector = _make_injector()
    return ImprovementTarget(
        target_type=target_type,
        name=name,
        discovered_path=discovered_path,
        injector=mock_injector,
    )


@pytest.fixture
def discovery() -> TargetDiscovery:
    return TargetDiscovery(project_root=Path.cwd())


class TestDiscoverAllTargets:
    """Tests for discover_all_targets."""

    def test_discovers_skills_in_standard_dirs(self, discovery: TargetDiscovery) -> None:
        with patch.object(
            discovery,
            "_discover_skills",
            return_value=[
                _make_target(
                    TargetType.SKILL, "test-skill", Path(".dawn-kestrel/skills/test-skill")
                ),
                _make_target(
                    TargetType.SKILL, "another-skill", Path(".opencode/skills/another-skill")
                ),
            ],
        ):
            targets = discovery.discover_all_targets()

            assert len(targets) >= 2
            skill_targets = [t for t in targets if t.target_type == TargetType.SKILL]
            assert len(skill_targets) >= 2

    def test_discovers_tools_in_standard_dirs(self, discovery: TargetDiscovery) -> None:
        with patch.object(
            discovery,
            "_discover_tools",
            return_value=[
                _make_target(TargetType.TOOL, "test-tool", Path(".dawn-kestrel/tools/test-tool")),
            ],
        ):
            targets = discovery.discover_all_targets()

            assert len(targets) >= 1
            tool_targets = [t for t in targets if t.target_type == TargetType.TOOL]
            assert len(tool_targets) >= 1

    def test_discovers_agents_in_standard_dirs(self, discovery: TargetDiscovery) -> None:
        with patch.object(
            discovery,
            "_discover_agents",
            return_value=[
                _make_target(
                    TargetType.AGENT, "test-agent", Path(".dawn-kestrel/agents/test-agent")
                ),
            ],
        ):
            targets = discovery.discover_all_targets()

            assert len(targets) >= 1
            agent_targets = [t for t in targets if t.target_type == TargetType.AGENT]
            assert len(agent_targets) >= 1

    def test_discovers_policies_in_standard_dirs(self, discovery: TargetDiscovery) -> None:
        with patch.object(
            discovery,
            "_discover_policies",
            return_value=[
                _make_target(
                    TargetType.POLICY,
                    "default-policy",
                    Path(".dawn-kestrel/policies/default-policy"),
                ),
            ],
        ):
            targets = discovery.discover_all_targets()

            assert len(targets) >= 1
            policy_targets = [t for t in targets if t.target_type == TargetType.POLICY]
            assert len(policy_targets) >= 1


class TestRankTargetsByImpact:
    """Tests for rank_targets_by_impact."""

    def test_ranks_by_low_score_frequency(self, discovery: TargetDiscovery) -> None:
        targets = [
            _make_target(TargetType.SKILL, "skill-a", Path("skills/skill-a.md")),
            _make_target(TargetType.SKILL, "skill-b", Path("skills/skill-b.md")),
            _make_target(TargetType.SKILL, "skill-c", Path("skills/skill-c.md")),
        ]
        for target in targets:
            target.priority = 20
        scenario_results = {
            "scenario-1": 0.6,
            "scenario-2": 0.55,
            "scenario-3": 0.52,
            "scenario-4": 0.5,
            "scenario-5": 0.45,
            "scenario-6": 0.4,
            "scenario-7": 0.35,
            "scenario-8": 0.25,
            "scenario-9": 0.15,
        }
        ranked = discovery.rank_targets_by_impact(targets, scenario_results)

        assert ranked[0].name == "skill-a"
        assert ranked[1].name == "skill-b"
        assert ranked[2].name == "skill-c"

        assert ranked[0].priority == 20
        assert ranked[1].priority == 20
        assert ranked[2].priority == 20

    def test_handles_empty_scenario_results(self, discovery: TargetDiscovery) -> None:
        targets = [
            _make_target(TargetType.SKILL, "skill-a", Path("skills/skill-a.md")),
        ]
        targets[0].priority = 20
        ranked = discovery.rank_targets_by_impact(targets, {})
        assert ranked[0].name == "skill-a"
        assert ranked[0].priority == 20


class TestLowScoreThreshold:
    """Tests for _compute_low_score_threshold."""

    def test_returns_25th_percentile_for_non_uniform_distribution(
        self, discovery: TargetDiscovery
    ) -> None:
        scores = {f"s{i}": 0.3 + i * 0.05 for i in range(10)}
        threshold = _compute_low_score_threshold(scores)
        assert 0.3 <= threshold <= 0.55

    def test_handles_empty_dict(self) -> None:
        threshold = _compute_low_score_threshold({})
        assert threshold == 0.0
