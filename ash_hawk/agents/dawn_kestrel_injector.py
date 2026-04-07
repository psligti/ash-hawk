from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

DAWN_KESTREL_DIR = Path(".dawn-kestrel")

AGENT_PATH_TEMPLATE = DAWN_KESTREL_DIR / "agents" / "{name}" / "AGENT.md"
SKILL_PATH_TEMPLATE = DAWN_KESTREL_DIR / "skills" / "{name}" / "SKILL.md"
TOOL_PATH_TEMPLATE = DAWN_KESTREL_DIR / "tools" / "{name}" / "TOOL.md"
POLICY_PATH_TEMPLATE = DAWN_KESTREL_DIR / "policies" / "{name}" / "POLICY.md"

if TYPE_CHECKING:
    from dawn_kestrel.agents.context import (
        BaseContextStrategy,
        ContextBundle,
        ScenarioInfo,
    )


class DawnKestrelInjector:
    def __init__(
        self,
        project_root: Path | None = None,
        inject_agent: bool = True,
        inject_skills: bool = True,
        inject_tools: bool = True,
        strategy: BaseContextStrategy | None = None,
        current_skill_name: str | None = None,
    ) -> None:
        self._project_root = project_root or Path.cwd()
        self._inject_agent = inject_agent
        self._inject_skills = inject_skills
        self._inject_tools = inject_tools
        self._strategy = strategy
        self._current_skill_name = current_skill_name
        self._cache: dict[str, str] = {}
        self._cache_valid: dict[str, bool] = {}

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def strategy(self) -> BaseContextStrategy | None:
        return self._strategy

    @strategy.setter
    def strategy(self, value: BaseContextStrategy | None) -> None:
        self._strategy = value
        self.invalidate_cache()

    @property
    def current_skill_name(self) -> str | None:
        return self._current_skill_name

    @current_skill_name.setter
    def current_skill_name(self, value: str | None) -> None:
        self._current_skill_name = value

    def get_agent_path(self, name: str) -> Path:
        return self._resolve_path(AGENT_PATH_TEMPLATE, name)

    def get_skill_path(self, name: str) -> Path:
        return self._resolve_path(SKILL_PATH_TEMPLATE, name)

    def get_tool_path(self, name: str) -> Path:
        return self._resolve_path(TOOL_PATH_TEMPLATE, name)

    def get_policy_path(self, name: str) -> Path:
        return self._resolve_path(POLICY_PATH_TEMPLATE, name)

    def _resolve_path(self, template: Path, name: str) -> Path:
        return self._project_root / str(template).format(name=name)

    def _read_file(self, path: Path) -> str | None:
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
        if path is not None:
            cache_key = str(path)
            self._cache_valid[cache_key] = False
        else:
            self._cache_valid.clear()

    def get_agent_content(self, agent_name: str) -> str | None:
        if not self._inject_agent:
            return None
        path = self._resolve_path(AGENT_PATH_TEMPLATE, agent_name)
        return self._read_file(path)

    def get_skill_content(self, skill_name: str) -> str | None:
        if not self._inject_skills:
            return None
        path = self._resolve_path(SKILL_PATH_TEMPLATE, skill_name)
        return self._read_file(path)

    def get_tool_content(self, tool_name: str) -> str | None:
        if not self._inject_tools:
            return None
        path = self._resolve_path(TOOL_PATH_TEMPLATE, tool_name)
        return self._read_file(path)

    def get_policy_content(self, policy_name: str) -> str | None:
        path = self._resolve_path(POLICY_PATH_TEMPLATE, policy_name)
        return self._read_file(path)

    def build_context(
        self,
        agent_id: str,
        scenario: ScenarioInfo | None = None,
    ) -> ContextBundle | None:
        if self._strategy is None:
            return None

        result = self._strategy.build_context(
            agent_id=agent_id,
            scenario=scenario,
            project_root=self._project_root,
        )

        if result.is_ok():
            return result.unwrap()

        logger.warning(f"Strategy build_context failed: {result.error}")
        return None

    def inject_into_prompt(
        self,
        agent_id: str,
        base_prompt: str,
        skills: list[str] | None = None,
        tools: list[str] | None = None,
        scenario: ScenarioInfo | None = None,
        use_strategy: bool = True,
    ) -> str:
        if use_strategy and self._strategy:
            bundle = self.build_context(agent_id, scenario)
            if bundle:
                injected_prompt = bundle.inject_into_prompt(base_prompt)
                if isinstance(injected_prompt, str):
                    return injected_prompt
                return str(injected_prompt)

        sections: list[str] = []

        agent_content = self.get_agent_content(agent_id)
        if agent_content:
            sections.append(f"## Agent Instructions ({agent_id})\n\n{agent_content}")

        if skills:
            for skill_name in skills:
                skill_content = self.get_skill_content(skill_name)
                if skill_content:
                    sections.append(f"## Skill: {skill_name}\n\n{skill_content}")

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
        path = self._resolve_path(AGENT_PATH_TEMPLATE, agent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        logger.info(f"Saved agent content to {path}")
        return path

    def save_skill_content(self, skill_name: str, content: str) -> Path:
        path = self._resolve_path(SKILL_PATH_TEMPLATE, skill_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        self._current_skill_name = skill_name
        logger.info(f"Saved skill content to {path}")
        return path

    def save_policy_content(self, policy_name: str, content: str) -> Path:
        path = self._resolve_path(POLICY_PATH_TEMPLATE, policy_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        logger.info(f"Saved policy content to {path}")
        return path

    def delete_skill_content(self, skill_name: str) -> bool:
        """Delete a skill directory and its contents.

        Args:
            skill_name: Name of the skill to delete.

        Returns:
            True if deleted, False if it didn't exist.
        """
        path = self._resolve_path(SKILL_PATH_TEMPLATE, skill_name)
        skill_dir = path.parent
        if skill_dir.exists() and skill_dir.is_dir():
            shutil.rmtree(skill_dir)
            self.invalidate_cache(path)
            logger.info(f"Deleted skill directory: {skill_dir}")
            return True
        return False

    def save_tool_content(self, tool_name: str, content: str) -> Path:
        path = self._resolve_path(TOOL_PATH_TEMPLATE, tool_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.invalidate_cache(path)
        logger.info(f"Saved tool content to {path}")
        return path

    def delete_agent_content(self, agent_name: str) -> bool:
        path = self._resolve_path(AGENT_PATH_TEMPLATE, agent_name)
        agent_dir = path.parent
        if agent_dir.exists() and agent_dir.is_dir():
            shutil.rmtree(agent_dir)
            self.invalidate_cache(path)
            logger.info(f"Deleted agent directory: {agent_dir}")
            return True
        return False

    def delete_tool_content(self, tool_name: str) -> bool:
        path = self._resolve_path(TOOL_PATH_TEMPLATE, tool_name)
        tool_dir = path.parent
        if tool_dir.exists() and tool_dir.is_dir():
            shutil.rmtree(tool_dir)
            self.invalidate_cache(path)
            logger.info(f"Deleted tool directory: {tool_dir}")
            return True
        return False

    def delete_policy_content(self, policy_name: str) -> bool:
        path = self._resolve_path(POLICY_PATH_TEMPLATE, policy_name)
        policy_dir = path.parent
        if policy_dir.exists() and policy_dir.is_dir():
            shutil.rmtree(policy_dir)
            self.invalidate_cache(path)
            logger.info(f"Deleted policy directory: {policy_dir}")
            return True
        return False

    def discover_skill_name(self, scenario_tools: list[str] | None = None) -> str | None:
        if self._current_skill_name is not None:
            return self._current_skill_name

        skills_dir = self._project_root / DAWN_KESTREL_DIR / "skills"
        if skills_dir.exists():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        return skill_dir.name

        if scenario_tools:
            return "default"

        return None

    def discover_tool_names(self, allowed_tools: list[str] | None = None) -> list[str]:
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
    "POLICY_PATH_TEMPLATE",
]
