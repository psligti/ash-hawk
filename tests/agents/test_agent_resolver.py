"""Tests for agent_resolver module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ash_hawk.agents.agent_resolver import (
    AgentResolution,
    AgentResolutionError,
    resolve_agent_path,
)


class TestResolveAgentPath:
    """Test resolve_agent_path function."""

    def test_direct_path_exists(self, tmp_path: Path) -> None:
        """Direct path that exists resolves as cli_path."""
        agent_dir = tmp_path / "my-agent"
        agent_dir.mkdir()
        (agent_dir / "AGENT.md").write_text("# Agent")

        result = resolve_agent_path(str(agent_dir), tmp_path)
        assert result.path == agent_dir.resolve()
        assert result.name == "my-agent"
        assert result.resolved_from == "cli_path"

    def test_direct_path_relative(self, tmp_path: Path) -> None:
        """Relative path is resolved against workdir."""
        agent_dir = tmp_path / "agents" / "bot"
        agent_dir.mkdir(parents=True)
        (agent_dir / "AGENT.md").write_text("# Bot")

        result = resolve_agent_path("agents/bot", tmp_path)
        assert result.path == agent_dir.resolve()
        assert result.resolved_from == "cli_path"

    def test_direct_agent_package_path_uses_parent_package_name(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "bolt_merlin" / "agent"
        agent_dir.mkdir(parents=True)
        (tmp_path / "bolt_merlin" / "__init__.py").write_text("", encoding="utf-8")
        (agent_dir / "__init__.py").write_text("", encoding="utf-8")

        result = resolve_agent_path(str(agent_dir), tmp_path)

        assert result.path == agent_dir.resolve()
        assert result.name == "bolt_merlin"
        assert result.resolved_from == "cli_path"

    def test_direct_path_symlink(self, tmp_path: Path) -> None:
        """Symlink is resolved to its real path."""
        real_dir = tmp_path / "real-agent"
        real_dir.mkdir()
        (real_dir / "AGENT.md").write_text("# Agent")

        link = tmp_path / "linked-agent"
        link.symlink_to(real_dir)

        result = resolve_agent_path(str(link), tmp_path)
        assert result.path == real_dir.resolve()
        assert result.resolved_from == "cli_path"

    def test_name_lookup_dawn_kestrel(self, tmp_path: Path) -> None:
        """Agent name found in .dawn-kestrel/agents/ directory."""
        agent_dir = tmp_path / ".dawn-kestrel" / "agents" / "my-agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "AGENT.md").write_text("# Agent")

        result = resolve_agent_path("my-agent", tmp_path)
        assert result.path == agent_dir.resolve()
        assert result.name == "my-agent"
        assert result.resolved_from == "name_lookup"

    def test_name_lookup_opencode(self, tmp_path: Path) -> None:
        """Agent name found as .opencode/agent/<name>.md file."""
        opencode_dir = tmp_path / ".opencode" / "agent"
        opencode_dir.mkdir(parents=True)
        (opencode_dir / "my-agent.md").write_text("# Agent config")

        result = resolve_agent_path("my-agent", tmp_path)
        assert result.name == "my-agent"
        assert result.resolved_from == "name_lookup"
        assert result.path == (opencode_dir / "my-agent").resolve()

    def test_path_wins_over_name(self, tmp_path: Path) -> None:
        """When both a direct path and name lookup exist, direct path wins."""
        direct_dir = tmp_path / "my-agent"
        direct_dir.mkdir()
        (direct_dir / "AGENT.md").write_text("# Direct")

        name_dir = tmp_path / ".dawn-kestrel" / "agents" / "my-agent"
        name_dir.mkdir(parents=True)
        (name_dir / "AGENT.md").write_text("# Named")

        result = resolve_agent_path("my-agent", tmp_path)
        assert result.resolved_from == "cli_path"
        assert result.path == direct_dir.resolve()

    def test_neither_resolves(self, tmp_path: Path) -> None:
        """Raises AgentResolutionError when nothing matches."""
        with pytest.raises(AgentResolutionError, match="Cannot resolve agent"):
            resolve_agent_path("nonexistent-agent", tmp_path)

    def test_name_with_underscores(self, tmp_path: Path) -> None:
        """Agent name with underscores resolves in .dawn-kestrel."""
        agent_dir = tmp_path / ".dawn-kestrel" / "agents" / "bolt_merlin"
        agent_dir.mkdir(parents=True)
        (agent_dir / "AGENT.md").write_text("# Bolt Merlin")

        result = resolve_agent_path("bolt_merlin", tmp_path)
        assert result.path == agent_dir.resolve()
        assert result.name == "bolt_merlin"
        assert result.resolved_from == "name_lookup"

    def test_agent_resolution_is_frozen(self) -> None:
        """AgentResolution is frozen dataclass; attribute assignment raises."""
        resolution = AgentResolution(
            path=Path("/tmp/agent"),
            name="test",
            resolved_from="cli_path",
        )
        with pytest.raises(FrozenInstanceError):
            resolution.name = "changed"  # type: ignore[misc]
