"""Dawn-Kestrel file injector for loading agent/skill/tool content from disk.

This injector reads improved content from .dawn-kestrel/ directory structure
and injects it into agent prompts at runtime, closing the loop between
auto-research improvements and evaluation runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical paths for dawn-kestrel content
DAWN_KESTREL_DIR = Path(".dawn-kestrel")

AGENT_PATH_TEMPLATE = DAWN_KESTREL_DIR / "agents" / "{name}" / "AGENT.md"
SKILL_PATH_TEMPLATE = DAWN_KESTREL_DIR / "skills" / "{name}" / "SKILL.md"
TOOL_PATH_TEMPLATE = DAWN_KESTREL_DIR / "tools" / "{name}" / "TOOL.md"


class DawnKestrelInjector:
    """Injects content from .dawn-kestrel/ files into agent prompts.

    Reads agent, skill, and tool definitions from the canonical
    .dawn-kestrel directory structure and appends them to prompts.

    Directory structure:
        .dawn-kestrel/
        ├── agents/
        │   └── {agent_name}/
        │       └── AGENT.md
        ├── skills/
        │   └── {skill_name}/
        │       └── SKILL.md
        └── tools/
            └── {tool_name}/
                └── TOOL.md

    Usage:
        injector = DawnKestrelInjector()
        augmented = injector.inject_into_prompt("my-agent", base_prompt)
    """

    def __init__(
        self,
        project_root: Path | None = None,
        inject_agent: bool = True,
        inject_skills: bool = True,
        inject_tools: bool = True,
    ) -> None:
        """Initialize the injector.

        Args:
            project_root: Root directory to search for .dawn-kestrel/.
                          Defaults to current working directory.
            inject_agent: Whether to inject agent content.
            inject_skills: Whether to inject skill content.
            inject_tools: Whether to inject tool content.
        """
        self._project_root = project_root or Path.cwd()
        self._inject_agent = inject_agent
        self._inject_skills = inject_skills
        self._inject_tools = inject_tools
        self._cache: dict[str, str] = {}
        self._cache_valid: dict[str, bool] = {}

    @property
    def project_root(self) -> Path:
        return self._project_root

    def get_agent_path(self, name: str) -> Path:
        return self._resolve_path(AGENT_PATH_TEMPLATE, name)

    def get_skill_path(self, name: str) -> Path:
        return self._resolve_path(SKILL_PATH_TEMPLATE, name)

    def get_tool_path(self, name: str) -> Path:
        return self._resolve_path(TOOL_PATH_TEMPLATE, name)

    def _resolve_path(self, template: Path, name: str) -> Path:
        return self._project_root / str(template).format(name=name)

    def _read_file(self, path: Path) -> str | None:
        """Read file content with caching."""
        cache_key = str(path)
        if cache_key in self._cache and self._cache_valid.get(cache_key, False):
            return self._cache[cache_key]

        if not path.exists():
            self._cache_valid[cache_key] = False
            return None

        try:
            content = path.read_text(encoding="utf-8").strip()
            self._cache[cache_key] = content
            self._cache_valid[cache_key] = True
            return content if content else None
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            self._cache_valid[cache_key] = False
            return None

    def invalidate_cache(self, path: Path | None = None) -> None:
        """Invalidate cache for a specific path or all paths.

        Args:
            path: Specific path to invalidate, or None for all.
        """
        if path is not None:
            cache_key = str(path)
            self._cache_valid[cache_key] = False
        else:
            self._cache_valid.clear()

    def get_agent_content(self, agent_name: str) -> str | None:
        """Get agent definition content.

        Args:
            agent_name: Name of the agent.

        Returns:
            Agent content or None if not found.
        """
        if not self._inject_agent:
            return None
        path = self._resolve_path(AGENT_PATH_TEMPLATE, agent_name)
        return self._read_file(path)

    def get_skill_content(self, skill_name: str) -> str | None:
        """Get skill definition content.

        Args:
            skill_name: Name of the skill.

        Returns:
            Skill content or None if not found.
        """
        if not self._inject_skills:
            return None
        path = self._resolve_path(SKILL_PATH_TEMPLATE, skill_name)
        return self._read_file(path)

    def get_tool_content(self, tool_name: str) -> str | None:
        """Get tool definition content.

        Args:
            tool_name: Name of the tool.

        Returns:
            Tool content or None if not found.
        """
        if not self._inject_tools:
            return None
        path = self._resolve_path(TOOL_PATH_TEMPLATE, tool_name)
        return self._read_file(path)

    def inject_into_prompt(
        self,
        agent_id: str,
        base_prompt: str,
        skills: list[str] | None = None,
        tools: list[str] | None = None,
    ) -> str:
        """Inject .dawn-kestrel/ content into agent prompt.

        Args:
            agent_id: The agent identifier (used for agent content lookup).
            base_prompt: The original prompt text.
            skills: List of skill names to inject (optional).
            tools: List of tool names to inject (optional).

        Returns:
            Augmented prompt with injected content.
        """
        sections: list[str] = []

        # Inject agent content
        agent_content = self.get_agent_content(agent_id)
        if agent_content:
            sections.append(f"## Agent Instructions ({agent_id})\n\n{agent_content}")

        # Inject skill content
        if skills:
            for skill_name in skills:
                skill_content = self.get_skill_content(skill_name)
                if skill_content:
                    sections.append(f"## Skill: {skill_name}\n\n{skill_content}")

        # Inject tool content
        if tools:
            for tool_name in tools:
                tool_content = self.get_tool_content(tool_name)
                if tool_content:
                    sections.append(f"## Tool: {tool_name}\n\n{tool_content}")

        if not sections:
            return base_prompt

        injected_section = "\n\n---\n\n" + "\n\n".join(sections)
        return base_prompt + injected_section

    def save_agent_content(self, agent_name: str, content: str) -> Path:
        """Save agent content to the canonical path.

        Args:
            agent_name: Name of the agent.
            content: Content to save.

        Returns:
            Path where content was saved.
        """
        path = self._resolve_path(AGENT_PATH_TEMPLATE, agent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        logger.info(f"Saved agent content to {path}")
        return path

    def save_skill_content(self, skill_name: str, content: str) -> Path:
        """Save skill content to the canonical path.

        Args:
            skill_name: Name of the skill.
            content: Content to save.

        Returns:
            Path where content was saved.
        """
        path = self._resolve_path(SKILL_PATH_TEMPLATE, skill_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        logger.info(f"Saved skill content to {path}")
        return path

    def save_tool_content(self, tool_name: str, content: str) -> Path:
        """Save tool content to the canonical path.

        Args:
            tool_name: Name of the tool.
            content: Content to save.

        Returns:
            Path where content was saved.
        """
        path = self._resolve_path(TOOL_PATH_TEMPLATE, tool_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        logger.info(f"Saved tool content to {path}")
        return path

    def discover_skill_name(self, scenario_tools: list[str] | None = None) -> str | None:
        """Discover skill name from scenario or defaults.

        Args:
            scenario_tools: List of allowed tools from scenario.

        Returns:
            Discovered skill name or None.
        """
        # Try to find an existing skill file
        skills_dir = self._project_root / DAWN_KESTREL_DIR / "skills"
        if skills_dir.exists():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        return skill_dir.name

        # Default to "default" if we have scenario tools
        if scenario_tools:
            return "default"

        return None

    def discover_tool_names(self, allowed_tools: list[str] | None = None) -> list[str]:
        """Discover tool names from allowed tools or existing files.

        Args:
            allowed_tools: List of allowed tools from scenario.

        Returns:
            List of tool names to inject.
        """
        if not allowed_tools:
            return []

        tool_names: list[str] = []

        for tool in allowed_tools:
            tool_path = self._resolve_path(TOOL_PATH_TEMPLATE, tool)
            if tool_path.exists():
                tool_names.append(tool)

        return tool_names


__all__ = [
    "DawnKestrelInjector",
    "DAWN_KESTREL_DIR",
    "AGENT_PATH_TEMPLATE",
    "SKILL_PATH_TEMPLATE",
    "TOOL_PATH_TEMPLATE",
]
