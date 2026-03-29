"""Diff application with backup and validation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ApplyResult:
    success: bool
    file_path: Path
    backup_path: Path | None = None
    error: str | None = None
    lines_added: int = 0
    lines_removed: int = 0


@dataclass
class DiffApplier:
    backup_suffix: str = ".bak"

    async def apply(
        self,
        file_path: Path,
        diff: str,
        dry_run: bool = False,
        backup: bool = True,
    ) -> ApplyResult:
        if dry_run:
            return self._validate_diff(file_path, diff)

        if not file_path.exists():
            return ApplyResult(
                success=False,
                file_path=file_path,
                error=f"File not found: {file_path}",
            )

        original = file_path.read_text(encoding="utf-8")

        if backup:
            backup_path = file_path.with_suffix(self.backup_suffix)
            shutil.copy2(file_path, backup_path)
        else:
            backup_path = None

        try:
            result = self._apply_patch(original, diff)
            improved = result["content"]

            self._write_file(file_path, improved)

            return ApplyResult(
                success=True,
                file_path=file_path,
                backup_path=backup_path,
                lines_added=result["lines_added"],
                lines_removed=result["lines_removed"],
            )

        except Exception as e:
            return ApplyResult(
                success=False,
                file_path=file_path,
                backup_path=backup_path,
                error=str(e),
            )

    def _write_file(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def _validate_diff(self, file_path: Path, diff: str) -> ApplyResult:
        lines = diff.splitlines()
        lines_added = 0
        lines_removed = 0

        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                lines_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                lines_removed += 1

        return ApplyResult(
            success=True,
            file_path=file_path,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

    def _apply_patch(self, original: str, diff: str) -> dict[str, Any]:
        from ash_hawk.improvement.differ import DiffGenerator

        original_lines = original.splitlines(keepends=True)
        generator = DiffGenerator()
        hunks = generator.parse_diff(diff)

        result_lines = list(original_lines)

        for hunk in reversed(hunks):
            old_start = hunk.get("old_start", 1) - 1
            old_count = hunk.get("old_count", 0)
            new_lines = hunk.get("lines", [])

            replacement: list[str] = []
            for line in new_lines:
                if line.startswith("+"):
                    replacement.append(line[1:])
                elif line.startswith(" "):
                    replacement.append(line[1:])
                elif not line.startswith("-"):
                    replacement.append(line)

            if old_start < len(result_lines):
                del result_lines[old_start : old_start + old_count]
                for i, new_line in enumerate(replacement):
                    result_lines.insert(old_start + i, new_line)

        content = "".join(result_lines)
        lines_added = sum(
            1
            for hunk in hunks
            for added_line in hunk.get("lines", [])
            if added_line.startswith("+")
        )
        lines_removed = sum(
            1
            for hunk in hunks
            for removed_line in hunk.get("lines", [])
            if removed_line.startswith("-")
        )

        return {
            "content": content,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        }


__all__ = ["ApplyResult", "DiffApplier"]
