"""Tests for DawnKestrelInjector."""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.agents.dawn_kestrel_injector import DawnKestrelInjector


class TestDawnKestrelInjector:
    def test_save_and_get_agent_content(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_agent_content("test-agent", "# Test Agent\n\nTest agent instructions.")
        agent_path = injector.get_agent_path("test-agent")
        assert agent_path == tmp_path / ".dawn-kestrel/agents/test-agent/AGENT.md"
        assert agent_path.exists()
        assert agent_path.read_text() == "# Test Agent\n\nTest agent instructions."

    def test_save_and_get_skill_content(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_skill_content("test-skill", "# Test Skill\n\nTest skill instructions.")
        skill_path = injector.get_skill_path("test-skill")
        assert skill_path == tmp_path / ".dawn-kestrel/skills/test-skill/SKILL.md"
        assert skill_path.exists()
        assert skill_path.read_text() == "# Test Skill\n\nTest skill instructions."

    def test_save_and_get_tool_content(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_tool_content("grep", "# Test Tool\n\nTest tool instructions.")
        tool_path = injector.get_tool_path("grep")
        assert tool_path == tmp_path / ".dawn-kestrel/tools/grep/TOOL.md"
        assert tool_path.exists()
        assert tool_path.read_text() == "# Test Tool\n\nTest tool instructions."

    def test_inject_into_prompt_no_files(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        result = injector.inject_into_prompt("test-agent", "Hello world")
        assert result == "Hello world"

    def test_inject_into_prompt_with_skill(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_skill_content("test-skill", "# Test Skill\n\nTest skill instructions.")
        result = injector.inject_into_prompt("test-agent", "Hello world", skills=["test-skill"])
        assert "## Skill: test-skill" in result
        assert "\n\n---" in result

    def test_inject_into_prompt_with_tool(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_tool_content("grep", "# Test Tool\n\nTest tool instructions.")
        result = injector.inject_into_prompt("test-agent", "Hello world", tools=["grep"])
        assert "## Tool: grep" in result
        assert "\n\n---" in result

    def test_inject_into_prompt_with_missing_skill_and_tool(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        result = injector.inject_into_prompt(
            "test-agent",
            "Hello world",
            skills=["missing-skill"],
            tools=["missing-tool"],
        )
        assert result == "Hello world"

    def test_roundtrip_save_agent_and_inject(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_agent_content("test-agent", "# Save")
        content = injector.get_agent_content("test-agent")
        assert content == "# Save"

    def test_roundtrip_save_skill_and_inject(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_skill_content("test-skill", "# Save")
        content = injector.get_skill_content("test-skill")
        assert content == "# Save"

    def test_cache_invalidation(self, tmp_path: Path) -> None:
        injector = DawnKestrelInjector(project_root=tmp_path)
        injector.save_agent_content("test-agent", "# v1")
        _ = injector.get_agent_content("test-agent")
        injector.save_agent_content("test-agent", "# v2")
        content = injector.get_agent_content("test-agent")
        assert content == "# v2"
