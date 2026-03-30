"""Tests for DiffApplier."""

from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.improvement.applier import ApplyResult, DiffApplier


@pytest.mark.asyncio
async def test_apply_valid_diff(tmp_path: Path) -> None:
    target = tmp_path / "agent.md"
    target.write_text("old line 1\nold line 2\nold line 3\n", encoding="utf-8")

    diff = """--- a/agent.md
+++ b/agent.md
@@ -1,3 +1,3 @@
 old line 1
-old line 2
+new line 2
 old line 3"""

    applier = DiffApplier()
    result = await applier.apply(target, diff)

    assert result.success is True
    assert result.file_path == target
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert result.lines_added == 1
    assert result.lines_removed == 1

    content = target.read_text(encoding="utf-8")
    assert "new line 2" in content
    assert "old line 2" not in content


@pytest.mark.asyncio
async def test_apply_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.md"

    applier = DiffApplier()
    result = await applier.apply(missing, "some diff")

    assert result.success is False
    assert "not found" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_apply_dry_run_does_not_modify(tmp_path: Path) -> None:
    target = tmp_path / "agent.md"
    original = "old content\n"
    target.write_text(original, encoding="utf-8")

    diff = """--- a/agent.md
+++ b/agent.md
@@ -1,1 +1,1 @@
-old content
+new content"""

    applier = DiffApplier()
    result = await applier.apply(target, diff, dry_run=True)

    assert result.success is True
    assert result.lines_added == 1
    assert result.lines_removed == 1
    assert target.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_apply_creates_backup(tmp_path: Path) -> None:
    target = tmp_path / "file.md"
    target.write_text("original\n", encoding="utf-8")

    diff = """--- a/file.md
+++ b/file.md
@@ -1,1 +1,1 @@
-original
+modified"""

    applier = DiffApplier(backup_suffix=".bak")
    result = await applier.apply(target, diff)

    assert result.success is True
    assert result.backup_path == target.with_suffix(".bak")
    assert result.backup_path.read_text(encoding="utf-8") == "original\n"


@pytest.mark.asyncio
async def test_apply_without_backup(tmp_path: Path) -> None:
    target = tmp_path / "file.md"
    target.write_text("original\n", encoding="utf-8")

    diff = """--- a/file.md
+++ b/file.md
@@ -1,1 +1,1 @@
-original
+modified"""

    applier = DiffApplier()
    result = await applier.apply(target, diff, backup=False)

    assert result.success is True
    assert result.backup_path is None
