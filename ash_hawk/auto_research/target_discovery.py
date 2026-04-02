from __future__ import annotations

import logging
from pathlib import Path

from ash_hawk.auto_research.types import ImprovementTarget, TargetType
from ash_hawk.services.dawn_kestrel_injector import (
    DAWN_KESTREL_DIR,
    DawnKestrelInjector,
)

logger = logging.getLogger(__name__)

SKILL_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "skills",
    Path(".opencode/skills"),
    Path(".claude/skills"),
]

TOOL_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "tools",
    Path(".opencode/tools"),
    Path(".claude/tools"),
]

AGENT_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "agents",
    Path(".opencode/agents"),
    Path(".claude/agents"),
]

POLICY_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "policies",
    Path(".opencode/policies"),
    Path(".claude/policies"),
]

_TYPE_PRIORITY: dict[TargetType, int] = {
    TargetType.AGENT: 30,
    TargetType.POLICY: 25,
    TargetType.SKILL: 20,
    TargetType.TOOL: 10,
}


class TargetDiscovery:
    SEARCH_DIRS: dict[TargetType, list[Path]] = {
        TargetType.SKILL: SKILL_SEARCH_DIRS,
        TargetType.TOOL: TOOL_SEARCH_DIRS,
        TargetType.AGENT: AGENT_SEARCH_DIRS,
        TargetType.POLICY: POLICY_SEARCH_DIRS,
    }

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def discover_all_targets(self) -> list[ImprovementTarget]:
        targets: list[ImprovementTarget] = []
        targets.extend(self._discover_agents())
        targets.extend(self._discover_policies())
        targets.extend(self._discover_skills())
        targets.extend(self._discover_tools())

        for target in targets:
            target.priority = _TYPE_PRIORITY.get(target.target_type, 0)

        targets.sort(key=lambda t: t.priority, reverse=True)
        return targets

    def rank_targets_by_impact(
        self,
        targets: list[ImprovementTarget],
        scenario_results: dict[str, float],
    ) -> list[ImprovementTarget]:
        if not scenario_results:
            return targets

        low_score_threshold = _compute_low_score_threshold(scenario_results)

        low_scoring_ids = {
            sid for sid, score in scenario_results.items() if score <= low_score_threshold
        }

        target_impact: dict[str, float] = {}
        for target in targets:
            frequency = _count_target_in_scenarios(target, low_scoring_ids)
            base_priority = _TYPE_PRIORITY.get(target.target_type, 0)
            target_impact[target.name] = base_priority + (frequency * 10)

        for target in targets:
            target.priority = int(target_impact.get(target.name, 0))

        targets.sort(key=lambda t: t.priority, reverse=True)
        return targets

    def _discover_skills(self) -> list[ImprovementTarget]:
        targets: list[ImprovementTarget] = []
        for search_dir in SKILL_SEARCH_DIRS:
            skills_path = self._project_root / search_dir
            if not (skills_path.exists() and skills_path.is_dir()):
                continue
            for skill_dir in sorted(skills_path.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.exists():
                    continue
                name = self._infer_name_from_path(skill_file)
                injector = DawnKestrelInjector(project_root=self._project_root)
                injector.current_skill_name = name
                targets.append(
                    ImprovementTarget(
                        target_type=TargetType.SKILL,
                        name=name,
                        discovered_path=skill_file,
                        injector=injector,
                    )
                )
        return targets

    def _discover_tools(self) -> list[ImprovementTarget]:
        targets: list[ImprovementTarget] = []
        for search_dir in TOOL_SEARCH_DIRS:
            tools_path = self._project_root / search_dir
            if not (tools_path.exists() and tools_path.is_dir()):
                continue
            for tool_dir in sorted(tools_path.iterdir()):
                if not tool_dir.is_dir():
                    continue
                tool_file = tool_dir / "TOOL.md"
                if not tool_file.exists():
                    continue
                name = self._infer_name_from_path(tool_file)
                injector = DawnKestrelInjector(project_root=self._project_root)
                targets.append(
                    ImprovementTarget(
                        target_type=TargetType.TOOL,
                        name=name,
                        discovered_path=tool_file,
                        injector=injector,
                    )
                )
        return targets

    def _discover_agents(self) -> list[ImprovementTarget]:
        targets: list[ImprovementTarget] = []
        for search_dir in AGENT_SEARCH_DIRS:
            agents_path = self._project_root / search_dir
            if not (agents_path.exists() and agents_path.is_dir()):
                continue
            for agent_dir in sorted(agents_path.iterdir()):
                if not agent_dir.is_dir():
                    continue
                agent_file = agent_dir / "AGENT.md"
                if not agent_file.exists():
                    continue
                name = self._infer_name_from_path(agent_file)
                injector = DawnKestrelInjector(project_root=self._project_root)
                targets.append(
                    ImprovementTarget(
                        target_type=TargetType.AGENT,
                        name=name,
                        discovered_path=agent_file,
                        injector=injector,
                    )
                )
        return targets

    def _discover_policies(self) -> list[ImprovementTarget]:
        targets: list[ImprovementTarget] = []
        for search_dir in POLICY_SEARCH_DIRS:
            policies_path = self._project_root / search_dir
            if not (policies_path.exists() and policies_path.is_dir()):
                continue
            for policy_dir in sorted(policies_path.iterdir()):
                if not policy_dir.is_dir():
                    continue
                policy_file = policy_dir / "POLICY.md"
                if not policy_file.exists():
                    continue
                name = self._infer_name_from_path(policy_file)
                injector = DawnKestrelInjector(project_root=self._project_root)
                targets.append(
                    ImprovementTarget(
                        target_type=TargetType.POLICY,
                        name=name,
                        discovered_path=policy_file,
                        injector=injector,
                    )
                )
        return targets

    def _infer_name_from_path(self, path: Path) -> str:
        stem = path.stem
        if stem.upper() in ("SKILL", "TOOL", "AGENT", "POLICY"):
            return path.parent.name
        return stem


def _compute_low_score_threshold(scenario_results: dict[str, float]) -> float:
    """25th percentile cutoff for low-scoring scenarios."""
    scores = sorted(scenario_results.values())
    if not scores:
        return 0.0
    idx = max(0, len(scores) // 4 - 1)
    return scores[idx]


def _count_target_in_scenarios(
    target: ImprovementTarget,
    scenario_ids: set[str],
) -> int:
    count = 0
    target_lower = target.name.lower()
    for sid in scenario_ids:
        if target_lower in sid.lower():
            count += 1
    for dep in target.dependencies:
        dep_lower = dep.lower()
        for sid in scenario_ids:
            if dep_lower in sid.lower():
                count += 1
    return count


__all__ = [
    "AGENT_SEARCH_DIRS",
    "POLICY_SEARCH_DIRS",
    "SKILL_SEARCH_DIRS",
    "TOOL_SEARCH_DIRS",
    "TargetDiscovery",
]
