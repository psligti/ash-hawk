from pathlib import Path

import pytest

from ash_hawk.auto_research.discovery import _discover_by_patterns  # noqa: PLC2701
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

    def test_filter_tool_prompts(self) -> None:
        targets = [
            Path("dawn_kestrel/tools/prompts/bash.txt"),
            Path("dawn_kestrel/tools/prompts/read.txt"),
            Path("skills/delegation.md"),
        ]
        result = filter_targets_by_type(targets, [ImprovementType.TOOL])
        assert Path("dawn_kestrel/tools/prompts/bash.txt") in result
        assert Path("dawn_kestrel/tools/prompts/read.txt") in result
        assert Path("skills/delegation.md") not in result


class TestDiscoverByPatterns:
    def test_parent_relative_pattern(self, tmp_path: Path) -> None:
        sibling = tmp_path / "sibling-repo"
        sibling.mkdir()
        tools_dir = sibling / "tools"
        tools_dir.mkdir()
        tool_file = tools_dir / "bash.txt"
        tool_file.write_text("Bash tool prompt")

        project = tmp_path / "my-project"
        project.mkdir()

        result = _discover_by_patterns(project, ["../sibling-repo/tools/*.txt"])

        assert tool_file in result

    def test_relative_pattern(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_file = skills_dir / "delegation.md"
        skill_file.write_text("Delegation skill")

        result = _discover_by_patterns(tmp_path, ["skills/**/*.md"])

        assert skill_file in result

    def test_multiple_patterns(self, tmp_path: Path) -> None:
        sibling = tmp_path.parent / "sdk"
        sibling.mkdir()
        sdk_tools = sibling / "tools" / "prompts"
        sdk_tools.mkdir(parents=True)
        sdk_tool = sdk_tools / "bash.txt"
        sdk_tool.write_text("Bash prompt")

        local_skills = tmp_path / "skills"
        local_skills.mkdir()
        local_skill = local_skills / "test.md"
        local_skill.write_text("Test skill")

        result = _discover_by_patterns(
            tmp_path,
            ["skills/**/*.md", "../sdk/tools/prompts/*.txt"],
        )

        assert local_skill in result
        assert sdk_tool in result
