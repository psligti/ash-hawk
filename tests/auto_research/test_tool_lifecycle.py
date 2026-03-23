"""Tests for ToolLifecycleManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.auto_research.tool_lifecycle import (
    DAWN_KESTREL_OVERLAY_DIR,
    TYPE_TO_SUBDIR,
    ToolLifecycleManager,
)
from ash_hawk.auto_research.types import ImprovementType, ToolCopyLifecycle


@pytest.fixture
def temp_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "test-repo"
    repo_root.mkdir()
    return repo_root


@pytest.fixture
def original_skill(temp_repo_root: Path) -> Path:
    skills_dir = temp_repo_root / "skills"
    skills_dir.mkdir()
    skill_file = skills_dir / "delegation.md"
    skill_file.write_text("# Delegation Skill\n\nOriginal content.")
    return skill_file


@pytest.fixture
def lifecycle_manager(temp_repo_root: Path) -> ToolLifecycleManager:
    return ToolLifecycleManager(repo_root=temp_repo_root)


class TestToolLifecycleManagerCreate:
    def test_create_working_copy_skill(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)

        assert lifecycle.original_path == original_skill.resolve()
        assert lifecycle.target_type == ImprovementType.SKILL
        assert lifecycle.created_this_cycle is True
        assert lifecycle.existed_before is False

        expected_copy = lifecycle_manager.overlay_root / "skills" / original_skill.name
        assert lifecycle.copy_path == expected_copy
        assert lifecycle.copy_path.exists()
        assert lifecycle.copy_path.read_text() == original_skill.read_text()

    def test_create_working_copy_uses_existing(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle1 = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        assert lifecycle1.created_this_cycle is True

        lifecycle2 = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        assert lifecycle2.created_this_cycle is False
        assert lifecycle2.existed_before is True
        assert lifecycle1.copy_path == lifecycle2.copy_path

    def test_create_working_copy_no_original(
        self,
        lifecycle_manager: ToolLifecycleManager,
        temp_repo_root: Path,
    ) -> None:
        nonexistent = temp_repo_root / "skills" / "new-skill.md"

        lifecycle = lifecycle_manager.create_working_copy(nonexistent, ImprovementType.SKILL)

        assert lifecycle.created_this_cycle is True
        assert lifecycle.copy_path.exists()
        assert lifecycle.copy_path.read_text() == ""

    def test_type_to_subdir_mapping(self) -> None:
        assert TYPE_TO_SUBDIR[ImprovementType.SKILL] == "skills"
        assert TYPE_TO_SUBDIR[ImprovementType.POLICY] == "policies"
        assert TYPE_TO_SUBDIR[ImprovementType.TOOL] == "tools"
        assert TYPE_TO_SUBDIR[ImprovementType.AGENT] == "agents"


class TestToolLifecycleManagerGet:
    def test_get_working_copy_exists(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)

        copy_path = lifecycle_manager.get_working_copy(original_skill)

        assert copy_path is not None
        assert copy_path.exists()

    def test_get_working_copy_not_tracked(
        self,
        lifecycle_manager: ToolLifecycleManager,
        temp_repo_root: Path,
    ) -> None:
        untracked = temp_repo_root / "skills" / "untracked.md"

        copy_path = lifecycle_manager.get_working_copy(untracked)

        assert copy_path is None

    def test_get_all_lifecycles(
        self,
        lifecycle_manager: ToolLifecycleManager,
        temp_repo_root: Path,
        original_skill: Path,
    ) -> None:
        policy_dir = temp_repo_root / "policies"
        policy_dir.mkdir()
        policy_file = policy_dir / "test-policy.md"
        policy_file.write_text("# Policy")

        lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        lifecycle_manager.create_working_copy(policy_file, ImprovementType.POLICY)

        lifecycles = lifecycle_manager.get_all_lifecycles()

        assert len(lifecycles) == 2
        types = {lc.target_type for lc in lifecycles}
        assert ImprovementType.SKILL in types
        assert ImprovementType.POLICY in types


class TestToolLifecycleManagerCleanupSuccess:
    def test_cleanup_success_keeps_created_copy(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        assert lifecycle.created_this_cycle is True

        lifecycle_manager.cleanup_success(original_skill)

        assert lifecycle.copy_path.exists()


class TestToolLifecycleManagerCleanupFailed:
    def test_cleanup_failed_deletes_created_copy(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        assert lifecycle.created_this_cycle is True
        assert lifecycle.copy_path.exists()

        lifecycle_manager.cleanup_failed(original_skill)

        assert not lifecycle.copy_path.exists()

    def test_cleanup_failed_keeps_preexisting_copy(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        lifecycle_manager._lifecycles[original_skill.resolve()].created_this_cycle = False
        lifecycle_manager._lifecycles[original_skill.resolve()].existed_before = True

        lifecycle_manager.cleanup_failed(original_skill)

        copy_path = lifecycle_manager.get_working_copy(original_skill)
        assert copy_path is not None
        assert copy_path.exists()

    def test_cleanup_failed_removes_empty_parent_dir(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)

        lifecycle_manager.cleanup_failed(original_skill)

        assert not lifecycle.copy_path.exists()
        parent = lifecycle.copy_path.parent
        assert not parent.exists()


class TestToolLifecycleManagerCleanupAllFailed:
    def test_cleanup_all_failed_deletes_all_created(
        self,
        lifecycle_manager: ToolLifecycleManager,
        temp_repo_root: Path,
        original_skill: Path,
    ) -> None:
        policy_dir = temp_repo_root / "policies"
        policy_dir.mkdir()
        policy_file = policy_dir / "test-policy.md"
        policy_file.write_text("# Policy")

        lc1 = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)
        lc2 = lifecycle_manager.create_working_copy(policy_file, ImprovementType.POLICY)

        assert lc1.copy_path.exists()
        assert lc2.copy_path.exists()

        lifecycle_manager.cleanup_all_failed()

        assert not lc1.copy_path.exists()
        assert not lc2.copy_path.exists()

    def test_cleanup_all_failed_removes_empty_overlay_dir(
        self,
        lifecycle_manager: ToolLifecycleManager,
        original_skill: Path,
    ) -> None:
        lifecycle = lifecycle_manager.create_working_copy(original_skill, ImprovementType.SKILL)

        lifecycle_manager.cleanup_all_failed()

        assert not lifecycle.copy_path.exists()
        assert not lifecycle_manager.overlay_root.exists()


class TestToolCopyLifecycle:
    def test_lifecycle_dataclass_fields(self, original_skill: Path, temp_repo_root: Path) -> None:
        copy_path = temp_repo_root / ".dawn-kestrel" / "skills" / "delegation.md"

        lifecycle = ToolCopyLifecycle(
            original_path=original_skill,
            copy_path=copy_path,
            target_type=ImprovementType.SKILL,
            existed_before=False,
            created_this_cycle=True,
        )

        assert lifecycle.original_path == original_skill
        assert lifecycle.copy_path == copy_path
        assert lifecycle.target_type == ImprovementType.SKILL
        assert lifecycle.existed_before is False
        assert lifecycle.created_this_cycle is True
