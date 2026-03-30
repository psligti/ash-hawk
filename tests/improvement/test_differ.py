"""Tests for DiffGenerator."""

from __future__ import annotations

from pathlib import Path

from ash_hawk.improvement.differ import DiffGenerator


def test_generate_returns_empty_for_identical_content() -> None:
    gen = DiffGenerator()
    result = gen.generate(Path("test.md"), "hello\n", "hello\n")
    assert result == ""


def test_generate_produces_unified_diff() -> None:
    gen = DiffGenerator()
    result = gen.generate(Path("test.md"), "old line\n", "new line\n")
    assert "--- a/test.md" in result
    assert "+++ b/test.md" in result
    assert "-old line" in result
    assert "+new line" in result


def test_generate_multi_combines_diffs() -> None:
    gen = DiffGenerator()
    files = [
        (Path("a.md"), "old a\n", "new a\n"),
        (Path("b.md"), "old b\n", "new b\n"),
    ]
    result = gen.generate_multi(files)
    assert "-old a" in result
    assert "-old b" in result
    assert "+new a" in result
    assert "+new b" in result


def test_generate_multi_skips_identical_files() -> None:
    gen = DiffGenerator()
    files = [
        (Path("same.md"), "unchanged\n", "unchanged\n"),
        (Path("changed.md"), "old\n", "new\n"),
    ]
    result = gen.generate_multi(files)
    assert "same.md" not in result
    assert "changed.md" in result


def test_parse_diff_extracts_hunks() -> None:
    gen = DiffGenerator()
    diff = """--- a/test.md
+++ b/test.md
@@ -1,2 +1,2 @@
 old line 1
-old line 2
+new line 2
 new line 3"""
    hunks = gen.parse_diff(diff)
    assert len(hunks) == 1
    assert hunks[0]["old_start"] == 1
    assert hunks[0]["old_count"] == 2
    assert hunks[0]["new_start"] == 1
    assert hunks[0]["new_count"] == 2
    assert len(hunks[0]["lines"]) == 4


def test_parse_diff_multiple_hunks() -> None:
    gen = DiffGenerator()
    diff = """--- a/file.md
+++ b/file.md
@@ -1,1 +1,1 @@
-old
+new
@@ -5,1 +5,1 @@
-old2
+new2"""
    hunks = gen.parse_diff(diff)
    assert len(hunks) == 2
    assert hunks[0]["old_start"] == 1
    assert hunks[1]["old_start"] == 5


def test_parse_hunk_header_without_count() -> None:
    gen = DiffGenerator()
    start, count = gen._parse_hunk_header("42")
    assert start == 42
    assert count == 1


def test_parse_hunk_header_with_count() -> None:
    gen = DiffGenerator()
    start, count = gen._parse_hunk_header("10,5")
    assert start == 10
    assert count == 5
