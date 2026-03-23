from pathlib import Path

import pytest

from ash_hawk.auto_research.discovery import filter_targets_by_type
from ash_hawk.auto_research.types import ImprovementType


class TestFilterTargetsByType:
    def test_filter_by_skill(self) -> None:
        targets = [
            Path("skills/delegation.md"),
            Path("tools/code-review.md"),
            Path("policies/workflow.md"),
        ]
        result = filter_targets_by_type(targets, [ImprovementType.SKILL])
        assert result == [Path("skills/delegation.md")]

    def test_filter_by_multiple_types(self) -> None:
        targets = [
            Path("skills/delegation.md"),
            Path("tools/code-review.md"),
            Path("policies/workflow.md"),
            Path(".opencode/skills/test.md"),
        ]
        result = filter_targets_by_type(targets, [ImprovementType.SKILL, ImprovementType.TOOL])
        assert Path("skills/delegation.md") in result
        assert Path("tools/code-review.md") in result
        assert Path(".opencode/skills/test.md") in result
        assert Path("policies/workflow.md") not in result

    def test_filter_empty_types_returns_all(self) -> None:
        targets = [
            Path("skills/delegation.md"),
            Path("tools/code-review.md"),
        ]
        result = filter_targets_by_type(targets, [])
        assert result == targets

    def test_filter_opencode_paths(self) -> None:
        targets = [
            Path(".opencode/skills/delegation.md"),
            Path(".opencode/policies/workflow.md"),
            Path(".claude/SKILL.md"),
        ]
        result = filter_targets_by_type(targets, [ImprovementType.SKILL])
        assert Path(".opencode/skills/delegation.md") in result
        assert Path(".claude/SKILL.md") in result
        assert Path(".opencode/policies/workflow.md") not in result

    def test_filter_agent_type(self) -> None:
        targets = [
            Path("agents/bolt-merlin.md"),
            Path("skills/delegation.md"),
            Path(".opencode/agents/build.md"),
        ]
        result = filter_targets_by_type(targets, [ImprovementType.AGENT])
        assert Path("agents/bolt-merlin.md") in result
        assert Path(".opencode/agents/build.md") in result
        assert Path("skills/delegation.md") not in result
