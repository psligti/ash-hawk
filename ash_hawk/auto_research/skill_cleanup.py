"""Cleanup for auto-research skill accumulation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ash_hawk.auto_research.types import (
    CleanupResult,
    CycleResult,
    EnhancedCycleResult,
    IterationResult,
)
from ash_hawk.services.dawn_kestrel_injector import (
    DAWN_KESTREL_DIR,
    DawnKestrelInjector,
)

logger = logging.getLogger(__name__)


@dataclass
class CleanupConfig:
    enabled: bool = True
    remove_unused: bool = True
    remove_negative_impact: bool = True
    min_positive_delta: float = 0.001
    preserve_promoted_lessons: bool = True
    dry_run: bool = False


class SkillCleaner:
    """Identifies and removes low-value skills created during auto-research cycles.

    A skill is considered low-value if:
    1. It was created during the cycle (not a baseline skill)
    2. It never appeared in an applied iteration with positive delta
    3. OR it had negative overall impact (skill-level regression)

    Args:
        project_root: Root directory of the project being evaluated.
        config: Cleanup configuration options.
    """

    def __init__(
        self,
        project_root: Path,
        config: CleanupConfig | None = None,
    ) -> None:
        self._project_root = project_root
        self._config = config or CleanupConfig()
        self._injector = DawnKestrelInjector(project_root=project_root)

    def get_baseline_skills(self) -> set[str]:
        skills_dir = self._project_root / DAWN_KESTREL_DIR / "skills"
        if not skills_dir.exists():
            return set()

        baseline = set()
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    baseline.add(skill_dir.name)

        return baseline

    def extract_skills_from_iteration(
        self,
        iteration: IterationResult,
    ) -> set[str]:
        if not iteration.improvement_text:
            return set()

        text = iteration.improvement_text
        skills: set[str] = set()

        skill_pattern = r"(?:skill|Skill):\s*([a-zA-Z0-9_-]+)"
        for match in re.finditer(skill_pattern, text):
            skills.add(match.group(1).strip())

        created_pattern = r"(?:created|new)\s+(?:skill|Skill)\s+([a-zA-Z0-9_-]+)"
        for match in re.finditer(created_pattern, text):
            skills.add(match.group(1).strip())

        path_pattern = r"\.dawn-kestrel/skills/([a-zA-Z0-9_-]+)"
        for match in re.finditer(path_pattern, text):
            skills.add(match.group(1).strip())

        return skills

    def identify_new_skills(
        self,
        baseline_skills: set[str],
        cycle_result: CycleResult,
    ) -> set[str]:
        new_skills: set[str] = set()

        for iteration in cycle_result.iterations:
            iteration_skills = self.extract_skills_from_iteration(iteration)
            for skill in iteration_skills:
                if skill not in baseline_skills:
                    new_skills.add(skill)

        return new_skills

    def evaluate_skill_effectiveness(
        self,
        skill_name: str,
        cycle_result: CycleResult,
    ) -> tuple[bool, str]:
        appeared_in_positive_iteration = False
        for iteration in cycle_result.applied_iterations:
            iteration_skills = self.extract_skills_from_iteration(iteration)
            if skill_name in iteration_skills and iteration.delta > 0:
                appeared_in_positive_iteration = True
                break

        if appeared_in_positive_iteration:
            return True, "Contributed to positive delta in iteration"

        caused_regression = False
        for iteration in cycle_result.iterations:
            iteration_skills = self.extract_skills_from_iteration(iteration)
            if skill_name in iteration_skills and iteration.delta < 0:
                caused_regression = True
                break

        if caused_regression:
            return False, "Associated with regression in iteration"

        return False, "Never contributed to applied iteration"

    def cleanup_cycle_result(
        self,
        cycle_result: CycleResult,
        baseline_skills: set[str] | None = None,
    ) -> CleanupResult:
        result = CleanupResult()

        if baseline_skills is None:
            baseline_skills = self.get_baseline_skills()

        new_skills = self.identify_new_skills(baseline_skills, cycle_result)

        if not new_skills:
            result.completed_at = datetime.now(UTC)
            return result

        for skill_name in new_skills:
            is_effective, reason = self.evaluate_skill_effectiveness(skill_name, cycle_result)

            if is_effective:
                result.kept_skills.append(skill_name)
                logger.debug(f"Keeping skill {skill_name}: {reason}")
            else:
                should_remove = False

                if self._config.remove_unused and "Never contributed" in reason:
                    should_remove = True
                elif self._config.remove_negative_impact and "regression" in reason:
                    should_remove = True

                if should_remove:
                    if self._config.dry_run:
                        logger.info(f"[DRY RUN] Would delete skill: {skill_name}")
                        result.cleaned_skills.append(skill_name)
                    else:
                        try:
                            deleted = self._injector.delete_skill_content(skill_name)
                            if deleted:
                                result.cleaned_skills.append(skill_name)
                                logger.info(f"Deleted low-value skill: {skill_name} ({reason})")
                            else:
                                result.kept_skills.append(skill_name)
                                logger.debug(f"Skill {skill_name} not found, skipping")
                        except Exception as e:
                            error_msg = f"Failed to delete skill {skill_name}: {e}"
                            result.errors.append(error_msg)
                            logger.error(error_msg)
                else:
                    result.kept_skills.append(skill_name)

        result.completed_at = datetime.now(UTC)
        return result

    def cleanup_enhanced_result(
        self,
        enhanced_result: EnhancedCycleResult,
        baseline_skills: set[str] | None = None,
    ) -> CleanupResult:
        aggregated = CleanupResult()

        if baseline_skills is None:
            baseline_skills = self.get_baseline_skills()

        if not self._config.enabled:
            aggregated.completed_at = datetime.now(UTC)
            return aggregated

        for target_name, cycle_result in enhanced_result.target_results.items():
            target_cleanup = self.cleanup_cycle_result(cycle_result, baseline_skills)

            aggregated.cleaned_skills.extend(target_cleanup.cleaned_skills)
            aggregated.kept_skills.extend(target_cleanup.kept_skills)
            aggregated.errors.extend(target_cleanup.errors)

            for skill in target_cleanup.kept_skills:
                baseline_skills.add(skill)

        aggregated.cleaned_skills = list(set(aggregated.cleaned_skills))
        aggregated.kept_skills = list(set(aggregated.kept_skills))
        aggregated.errors = list(set(aggregated.errors))

        aggregated.completed_at = datetime.now(UTC)

        if aggregated.cleaned_skills:
            logger.info(
                f"Cleanup complete: removed {len(aggregated.cleaned_skills)} skills, "
                f"kept {len(aggregated.kept_skills)} skills"
            )

        return aggregated


__all__ = [
    "CleanupConfig",
    "CleanupResult",
    "SkillCleaner",
]
