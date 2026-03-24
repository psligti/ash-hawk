"""Prompt capture module for saving agent, skill, and tool prompts.

This module provides functionality to capture and persist prompts from
the improvement agent system to version-controlled markdown files.

Storage structure:
    .dawn-kestrel/
    ├── agent.md              # Agent system prompt
    ├── skills/
    │   ├── {skill_name}.md   # Skill definitions
    │   └── ...
    └── tools/
        ├── {tool_name}.md    # Tool descriptions
        └── ...
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

logger = logging.getLogger(__name__)

DEFAULT_DAWN_KESTREL_DIR = Path(".dawn-kestrel")
AGENT_PROMPT_FILE = "agent.md"
SKILLS_DIR = "skills"
TOOLS_DIR = "tools"


class CapturedPrompt:
    """Represents a captured prompt with metadata.

    Attributes:
        name: Name of the prompt (agent, skill name, or tool name).
        prompt_type: Type of prompt (agent, skill, tool).
        content: The actual prompt content.
        metadata: Additional metadata (version, hash, etc.).
        captured_at: When the prompt was captured.
        path: Path where the prompt is stored.
    """

    def __init__(
        self,
        name: str,
        prompt_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> None:
        self.name = name
        self.prompt_type = prompt_type
        self.content = content
        self.metadata = metadata or {}
        self.captured_at = datetime.now(UTC)
        self.path = path
        self.content_hash = self._compute_hash(content)

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for versioning."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    def to_markdown(self) -> str:
        """Convert captured prompt to markdown with frontmatter.

        Returns:
            Markdown string with YAML frontmatter and content.
        """
        frontmatter = {
            "name": self.name,
            "type": self.prompt_type,
            "version": self.metadata.get("version", "1.0.0"),
            "content_hash": self.content_hash,
            "captured_at": self.captured_at.isoformat(),
            **self.metadata,
        }
        frontmatter.pop("name", None)
        frontmatter["name"] = self.name

        frontmatter_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        return f"---\n{frontmatter_str}---\n\n{self.content}"

    def __repr__(self) -> str:
        return (
            f"CapturedPrompt(name={self.name!r}, type={self.prompt_type!r}, "
            f"hash={self.content_hash[:8]}...)"
        )


class PromptCapture:
    """Captures and persists prompts from agent, skills, and tools.

    This class provides methods to save prompts to the .dawn-kestrel
    directory structure for version control and tracking.

    Attributes:
        base_dir: Base directory for storing prompts (default: .dawn-kestrel).
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_DAWN_KESTREL_DIR
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        (self.base_dir / SKILLS_DIR).mkdir(parents=True, exist_ok=True)
        (self.base_dir / TOOLS_DIR).mkdir(parents=True, exist_ok=True)

    def save_agent_prompt(
        self,
        agent_name: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> CapturedPrompt:
        """Save agent system prompt to .dawn-kestrel/agent.md.

        Args:
            agent_name: Name of the agent.
            prompt: The full system prompt text.
            metadata: Optional metadata (version, model, etc.).

        Returns:
            CapturedPrompt with path and metadata.
        """
        enriched_metadata = {
            "agent_name": agent_name,
            **(metadata or {}),
        }

        captured = CapturedPrompt(
            name=agent_name,
            prompt_type="agent",
            content=prompt,
            metadata=enriched_metadata,
        )

        path = self.base_dir / AGENT_PROMPT_FILE
        path.write_text(captured.to_markdown(), encoding="utf-8")
        captured.path = path

        logger.info(f"Saved agent prompt to {path} (hash: {captured.content_hash})")
        return captured

    def save_skill_prompt(
        self,
        skill_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> CapturedPrompt:
        """Save skill prompt to .dawn-kestrel/skills/{skill_name}.md.

        Args:
            skill_name: Name of the skill.
            content: The skill content/instructions.
            metadata: Optional metadata (version, triggers, etc.).

        Returns:
            CapturedPrompt with path and metadata.
        """
        captured = CapturedPrompt(
            name=skill_name,
            prompt_type="skill",
            content=content,
            metadata=metadata,
        )

        path = self.base_dir / SKILLS_DIR / f"{skill_name}.md"
        path.write_text(captured.to_markdown(), encoding="utf-8")
        captured.path = path

        logger.info(f"Saved skill prompt to {path} (hash: {captured.content_hash})")
        return captured

    def save_tool_prompt(
        self,
        tool_name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CapturedPrompt:
        """Save tool description to .dawn-kestrel/tools/{tool_name}.md.

        Args:
            tool_name: Name of the tool.
            description: Tool description/instructions.
            parameters: JSON schema for tool parameters.
            metadata: Optional metadata (version, timeout, etc.).

        Returns:
            CapturedPrompt with path and metadata.
        """
        enriched_metadata = {
            "tool_name": tool_name,
            **(metadata or {}),
        }

        full_content = description
        if parameters:
            params_yaml = yaml.dump(parameters, default_flow_style=False)
            full_content += f"\n\n## Parameters\n\n```yaml\n{params_yaml}```\n"

        captured = CapturedPrompt(
            name=tool_name,
            prompt_type="tool",
            content=full_content,
            metadata=enriched_metadata,
        )

        path = self.base_dir / TOOLS_DIR / f"{tool_name}.md"
        path.write_text(captured.to_markdown(), encoding="utf-8")
        captured.path = path

        logger.info(f"Saved tool prompt to {path} (hash: {captured.content_hash})")
        return captured

    def load_agent_prompt(self) -> CapturedPrompt | None:
        """Load agent prompt from .dawn-kestrel/agent.md.

        Returns:
            CapturedPrompt if file exists, None otherwise.
        """
        path = self.base_dir / AGENT_PROMPT_FILE
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        return self._parse_markdown(content, path)

    def load_skill_prompt(self, skill_name: str) -> CapturedPrompt | None:
        """Load skill prompt from .dawn-kestrel/skills/{skill_name}.md.

        Args:
            skill_name: Name of the skill to load.

        Returns:
            CapturedPrompt if file exists, None otherwise.
        """
        path = self.base_dir / SKILLS_DIR / f"{skill_name}.md"
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        return self._parse_markdown(content, path)

    def load_tool_prompt(self, tool_name: str) -> CapturedPrompt | None:
        """Load tool prompt from .dawn-kestrel/tools/{tool_name}.md.

        Args:
            tool_name: Name of the tool to load.

        Returns:
            CapturedPrompt if file exists, None otherwise.
        """
        path = self.base_dir / TOOLS_DIR / f"{tool_name}.md"
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        return self._parse_markdown(content, path)

    def list_skills(self) -> list[str]:
        """List all saved skill names.

        Returns:
            List of skill names (without .md extension).
        """
        skills_dir = self.base_dir / SKILLS_DIR
        if not skills_dir.exists():
            return []

        return [p.stem for p in skills_dir.glob("*.md")]

    def list_tools(self) -> list[str]:
        """List all saved tool names.

        Returns:
            List of tool names (without .md extension).
        """
        tools_dir = self.base_dir / TOOLS_DIR
        if not tools_dir.exists():
            return []

        return [p.stem for p in tools_dir.glob("*.md")]

    def get_prompt_history(
        self,
        prompt_type: str,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get history of prompt changes (from git if available).

        Args:
            prompt_type: Type of prompt (agent, skill, tool).
            name: Name of skill/tool (required for non-agent types).

        Returns:
            List of historical versions with hashes and timestamps.
        """
        if prompt_type == "agent":
            path = self.base_dir / AGENT_PROMPT_FILE
        elif prompt_type == "skill" and name:
            path = self.base_dir / SKILLS_DIR / f"{name}.md"
        elif prompt_type == "tool" and name:
            path = self.base_dir / TOOLS_DIR / f"{name}.md"
        else:
            return []

        if not path.exists():
            return []

        current = self._parse_markdown(path.read_text(encoding="utf-8"), path)
        if not current:
            return []

        return [
            {
                "hash": current.content_hash,
                "version": current.metadata.get("version", "unknown"),
                "captured_at": current.captured_at.isoformat(),
                "path": str(path),
            }
        ]

    def _parse_markdown(self, content: str, path: Path) -> CapturedPrompt | None:
        """Parse a markdown file with YAML frontmatter.

        Args:
            content: Full file content.
            path: Path to the file.

        Returns:
            CapturedPrompt if parsing succeeds, None otherwise.
        """
        if not content.startswith("---"):
            return CapturedPrompt(
                name=path.stem,
                prompt_type="unknown",
                content=content,
                path=path,
            )

        try:
            parts = content.split("---", 2)
            if len(parts) < 3:
                return None

            frontmatter_str = parts[1].strip()
            body = parts[2].strip()

            loaded = yaml.safe_load(frontmatter_str)
            if not isinstance(loaded, dict):
                return None
            metadata = cast(dict[str, Any], loaded)

            name: str = metadata.pop("name", path.stem)
            prompt_type: str = metadata.pop("type", "unknown")

            return CapturedPrompt(
                name=name,
                prompt_type=prompt_type,
                content=body,
                metadata=metadata,
                path=path,
            )
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter from {path}: {e}")
            return None

    def capture_from_bolt_merlin(
        self,
        agent_config: Any,
        skill_registry: Any,
        tool_registry: Any,
    ) -> dict[str, CapturedPrompt]:
        """Capture prompts from bolt-merlin style registries.

        This method provides compatibility with the bolt-merlin registry
        pattern for extracting agent prompts, skill content, and tool
        descriptions.

        Args:
            agent_config: AgentConfig object with agent.prompt attribute.
            skill_registry: Skill registry with get_all() method.
            tool_registry: Tool registry with get_all() method.

        Returns:
            Dict mapping capture keys to CapturedPrompt objects.
        """
        captured: dict[str, CapturedPrompt] = {}

        if hasattr(agent_config, "agent"):
            agent = agent_config.agent
            if hasattr(agent, "prompt"):
                captured["agent"] = self.save_agent_prompt(
                    agent_name=getattr(agent, "name", "default"),
                    prompt=agent.prompt,
                    metadata={
                        "model": getattr(agent, "options", {}).get("model"),
                        "temperature": getattr(agent, "temperature", None),
                    },
                )

        if hasattr(skill_registry, "get_all"):
            skills = skill_registry.get_all()
            for skill_name, skill in skills.items():
                if hasattr(skill, "content"):
                    captured[f"skill:{skill_name}"] = self.save_skill_prompt(
                        skill_name=skill_name,
                        content=skill.content,
                        metadata={
                            "description": getattr(skill, "description", ""),
                            "location": getattr(skill, "location", ""),
                        },
                    )

        if hasattr(tool_registry, "get_all"):
            tools = tool_registry.get_all()
            for tool_name, tool in tools.items():
                if hasattr(tool, "description"):
                    captured[f"tool:{tool_name}"] = self.save_tool_prompt(
                        tool_name=tool_name,
                        description=tool.description,
                        parameters=tool.parameters() if hasattr(tool, "parameters") else None,
                    )

        return captured


__all__ = [
    "CapturedPrompt",
    "PromptCapture",
    "DEFAULT_DAWN_KESTREL_DIR",
]
