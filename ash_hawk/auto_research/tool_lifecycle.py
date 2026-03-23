"""Tool lifecycle management for auto-research improvement cycles.

Creates and manages working copies of tools in .dawn-kestrel overlay directory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ash_hawk.auto_research.types import (
    ImprovementType,
    ToolCopyLifecycle,
)

logger = logging.getLogger(__name__)

DAWN_KESTREL_OVERLAY_DIR = ".dawn-kestrel"

TYPE_TO_SUBDIR: dict[ImprovementType, str] = {
    ImprovementType.SKILL: "skills",
    ImprovementType.POLICY: "policies",
    ImprovementType.TOOL: "tools",
    ImprovementType.AGENT: "agents",
}


class ToolLifecycleManager:
    """Manages working copies of tools in .dawn-kestrel overlay.

    Creates copies of tools being improved, tracks which were newly created,
    and handles cleanup based on improvement success/failure.
    """

    def __init__(self, repo_root: Path | None = None):
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.overlay_root = self.repo_root / DAWN_KESTREL_OVERLAY_DIR
        self._lifecycles: dict[Path, ToolCopyLifecycle] = {}

    def create_working_copy(
        self,
        original_path: Path,
        target_type: ImprovementType,
    ) -> ToolCopyLifecycle:
        """Create a working copy in .dawn-kestrel overlay.

        Args:
            original_path: Path to the original tool file.
            target_type: Type of tool (skill, policy, tool, agent).

        Returns:
            ToolCopyLifecycle tracking the copy state.
        """
        original_path = original_path.resolve()
        subdir = TYPE_TO_SUBDIR.get(target_type, "tools")
        relative_name = original_path.name
        copy_path = self.overlay_root / subdir / relative_name

        existed_before = copy_path.exists()
        created_this_cycle = False

        if not existed_before:
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            if original_path.exists():
                shutil.copy2(original_path, copy_path)
                logger.info(f"Created working copy: {copy_path}")
            else:
                copy_path.touch()
                logger.info(f"Created new working copy (no original): {copy_path}")
            created_this_cycle = True

        lifecycle = ToolCopyLifecycle(
            original_path=original_path,
            copy_path=copy_path,
            target_type=target_type,
            existed_before=existed_before,
            created_this_cycle=created_this_cycle,
        )
        self._lifecycles[original_path] = lifecycle
        return lifecycle

    def get_working_copy(self, original_path: Path) -> Path | None:
        """Get the working copy path for an original tool.

        Args:
            original_path: Path to the original tool file.

        Returns:
            Working copy path if it exists, None otherwise.
        """
        original_path = original_path.resolve()
        lifecycle = self._lifecycles.get(original_path)
        return lifecycle.copy_path if lifecycle else None

    def get_all_lifecycles(self) -> list[ToolCopyLifecycle]:
        """Get all tracked lifecycles."""
        return list(self._lifecycles.values())

    def cleanup_success(self, original_path: Path) -> None:
        """Clean up after successful improvement (keep working copy).

        Args:
            original_path: Path to the original tool file.
        """
        lifecycle = self._lifecycles.get(original_path.resolve())
        if not lifecycle:
            return

        if lifecycle.created_this_cycle:
            logger.info(f"Keeping working copy (improvement succeeded): {lifecycle.copy_path}")

    def cleanup_failed(self, original_path: Path) -> None:
        """Clean up after failed improvement (delete working copy if newly created).

        Args:
            original_path: Path to the original tool file.
        """
        lifecycle = self._lifecycles.get(original_path.resolve())
        if not lifecycle:
            return

        if lifecycle.created_this_cycle and lifecycle.copy_path.exists():
            lifecycle.copy_path.unlink()
            logger.info(f"Deleted working copy (improvement failed): {lifecycle.copy_path}")

            parent = lifecycle.copy_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                logger.debug(f"Removed empty directory: {parent}")

    def cleanup_all_failed(self) -> None:
        """Clean up all working copies created this cycle (used on error/cancellation)."""
        for lifecycle in self._lifecycles.values():
            if lifecycle.created_this_cycle and lifecycle.copy_path.exists():
                lifecycle.copy_path.unlink()
                logger.info(f"Deleted working copy (cycle error): {lifecycle.copy_path}")

        for subdir in TYPE_TO_SUBDIR.values():
            type_dir = self.overlay_root / subdir
            if type_dir.exists() and not any(type_dir.iterdir()):
                type_dir.rmdir()
                logger.debug(f"Removed empty directory: {type_dir}")

        if self.overlay_root.exists() and not any(self.overlay_root.iterdir()):
            self.overlay_root.rmdir()
            logger.debug(f"Removed empty overlay directory: {self.overlay_root}")


__all__ = ["ToolLifecycleManager", "DAWN_KESTREL_OVERLAY_DIR", "TYPE_TO_SUBDIR"]
