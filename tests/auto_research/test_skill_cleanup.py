"""Tests for SkillCleaner."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash_hawk.auto_research.skill_cleanup import CleanupConfig, SkillCleaner
from ash_hawk.auto_research.types import (
    CleanupResult,
    CycleResult,
    CycleStatus,
    EnhancedCycleResult,
    IterationResult,
    TargetType,
)


@pytest.fixture
def temp_project_root(tmp_path: Path) -> Path:
    skills_dir = tmp_path / ".dawn-kestrel" / "skills"
    skills_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def baseline_skill(temp_project_root: Path) -> Path:
    skill_dir = temp_project_root / ".dawn-kestrel" / "skills" / "baseline-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "# Baseline Skill\n\nThis skill existed before the cycle.", encoding="utf-8"
    )
    return skill_dir


@pytest.fixture
def cleaner(temp_project_root: Path) -> SkillCleaner:
    return SkillCleaner(
        project_root=temp_project_root,
        config=CleanupConfig(
            enabled=True,
            remove_unused=True,
            remove_negative_impact=True,
        ),
    )


class TestSkillCleaner:
    """Test SkillCleaner class."""

    def test_get_baseline_skills_empty(self, cleaner: SkillCleaner) -> None:
        result = cleaner.get_baseline_skills()
        assert result == set()

    def test_get_baseline_skills_with_existing(
        self,
        cleaner: SkillCleaner,
        baseline_skill: Path,
    ) -> None:
        result = cleaner.get_baseline_skills()
        assert "baseline-skill" in result

    def test_extract_skills_from_iteration_empty(self, cleaner: SkillCleaner) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.6,
            improvement_text="",
            applied=True,
        )
        result = cleaner.extract_skills_from_iteration(iteration)
        assert result == set()

    def test_extract_skills_from_iteration_with_skill_pattern(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.6,
            improvement_text="Created skill: new-optimization-skill for better performance",
            applied=True,
        )
        result = cleaner.extract_skills_from_iteration(iteration)
        assert "new-optimization-skill" in result

    def test_extract_skills_from_iteration_with_created_pattern(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.6,
            improvement_text="Created new skill error-handling-improvement",
            applied=True,
        )
        result = cleaner.extract_skills_from_iteration(iteration)
        assert "error-handling-improvement" in result

    def test_extract_skills_from_iteration_with_path_reference(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.6,
            improvement_text="Updated .dawn-kestrel/skills/file-operations/SKILL.md",
            applied=True,
        )
        result = cleaner.extract_skills_from_iteration(iteration)
        assert "file-operations" in result

    def test_identify_new_skills_none(self, cleaner: SkillCleaner) -> None:
        cycle_result = CycleResult(
            agent_name="test-agent",
            target_path=".dawn-kestrel/skills/test-skill",
            target_type=TargetType.SKILL,
        )
        baseline = {"existing-skill"}
        result = cleaner.identify_new_skills(baseline, cycle_result)
        assert result == set()

    def test_identify_new_skills_with_new(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.6,
            improvement_text="Created skill: brand-new-skill",
            applied=True,
        )
        cycle_result = CycleResult(
            agent_name="test-agent",
            target_path=".dawn-kestrel/skills/test-skill",
            target_type=TargetType.SKILL,
            iterations=[iteration],
        )
        baseline = {"existing-skill"}
        result = cleaner.identify_new_skills(baseline, cycle_result)
        assert "brand-new-skill" in result
        assert "existing-skill" not in result

    def test_evaluate_skill_effectiveness_positive_delta(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.7,
            improvement_text="Created skill: effective-skill",
            applied=True,
        )
        cycle_result = CycleResult(
            agent_name="test-agent",
            target_path=".dawn-kestrel/skills/test-skill",
            target_type=TargetType.SKILL,
            iterations=[iteration],
        )
        is_effective, reason = cleaner.evaluate_skill_effectiveness("effective-skill", cycle_result)
        assert is_effective is True
        assert "positive delta" in reason.lower()

    def test_evaluate_skill_effectiveness_never_contributed(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.6,
            improvement_text="Some improvement without skill reference",
            applied=True,
        )
        cycle_result = CycleResult(
            agent_name="test-agent",
            target_path=".dawn-kestrel/skills/test-skill",
            target_type=TargetType.SKILL,
            iterations=[iteration],
        )
        is_effective, reason = cleaner.evaluate_skill_effectiveness("unused-skill", cycle_result)
        assert is_effective is False
        assert "Never contributed" in reason

    def test_cleanup_cycle_result_dry_run(
        self,
        temp_project_root: Path,
    ) -> None:
        dry_cleaner = SkillCleaner(
            project_root=temp_project_root,
            config=CleanupConfig(
                enabled=True,
                remove_unused=True,
                remove_negative_impact=True,
                dry_run=True,
            ),
        )
        skill_creation_iteration = IterationResult(
            iteration_num=1,
            score_before=0.5,
            score_after=0.5,
            improvement_text="Created skill: dry-run-test-skill",
            applied=False,
        )
        successful_iteration = IterationResult(
            iteration_num=2,
            score_before=0.5,
            score_after=0.7,
            improvement_text="Made improvements without referencing the skill",
            applied=True,
        )
        cycle_result = CycleResult(
            agent_name="test-agent",
            target_path=".dawn-kestrel/skills/test-skill",
            target_type=TargetType.SKILL,
            iterations=[skill_creation_iteration, successful_iteration],
        )
        baseline: set[str] = set()
        result = dry_cleaner.cleanup_cycle_result(cycle_result, baseline)
        assert "dry-run-test-skill" in result.cleaned_skills

    def test_cleanup_enhanced_result_disabled(
        self,
        cleaner: SkillCleaner,
    ) -> None:
        disabled_cleaner = SkillCleaner(
            project_root=cleaner._project_root,
            config=CleanupConfig(enabled=False),
        )
        enhanced_result = EnhancedCycleResult(
            agent_name="test-agent",
            target_results={},
        )
        result = disabled_cleaner.cleanup_enhanced_result(enhanced_result)
        assert result.cleaned_skills == []
        assert result.kept_skills == []


class TestCleanupResult:
    """Test CleanupResult dataclass."""

    def test_default_values(self) -> None:
        result = CleanupResult()
        assert result.cleaned_skills == []
        assert result.kept_skills == []
        assert result.errors == []
        assert result.started_at is not None
        assert result.completed_at is None

    def test_total_processed(self) -> None:
        result = CleanupResult(
            cleaned_skills=["skill1", "skill2"],
            kept_skills=["skill3"],
        )
        assert result.total_processed == 3

    def test_duration_seconds(self) -> None:
        result = CleanupResult(
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            completed_at=datetime(2024, 1, 1, 12, 0, 5, tzinfo=UTC),
        )
        assert result.duration_seconds == 5.0


class TestCleanupConfig:
    """Test CleanupConfig dataclass."""

    def test_default_values(self) -> None:
        config = CleanupConfig()
        assert config.enabled is True
        assert config.remove_unused is True
        assert config.remove_negative_impact is True
        assert config.min_positive_delta == 0.001
        assert config.preserve_promoted_lessons is True
        assert config.dry_run is False

    def test_custom_values(self) -> None:
        config = CleanupConfig(
            enabled=False,
            remove_unused=False,
            remove_negative_impact=False,
            min_positive_delta=0.01,
            preserve_promoted_lessons=False,
            dry_run=True,
        )
        assert config.enabled is False
        assert config.remove_unused is False
        assert config.remove_negative_impact is False
        assert config.min_positive_delta == 0.01
        assert config.preserve_promoted_lessons is False
        assert config.dry_run is True
