"""Unified diff generation."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class DiffGenerator:
    context_lines: int = 3

    def generate(
        self,
        file_path: Path,
        original: str,
        improved: str,
    ) -> str:
        original_lines = original.splitlines(keepends=True)
        improved_lines = improved.splitlines(keepends=True)

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                improved_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=self.context_lines,
            )
        )

        if not diff_lines:
            return ""

        return "".join(diff_lines)

    def generate_multi(
        self,
        files: list[tuple[Path, str, str]],
    ) -> str:
        diffs: list[str] = []

        for file_path, original, improved in files:
            diff = self.generate(file_path, original, improved)
            if diff:
                diffs.append(diff)

        return "\n".join(diffs)

    def _parse_hunk_header(self, header: str) -> tuple[int, int]:
        cleaned = header.lstrip("-+")
        if "," in cleaned:
            start, count = cleaned.split(",", 1)
            return int(start), int(count)
        return int(cleaned), 1

    def parse_diff(self, diff: str) -> list[dict[str, Any]]:
        hunks: list[dict[str, Any]] = []
        current_file: str | None = None
        current_hunk: dict[str, Any] | None = None
        hunk_lines: list[str] = []

        for line in diff.splitlines():
            if line.startswith("--- "):
                if current_hunk and hunk_lines:
                    current_hunk["lines"] = hunk_lines
                    hunks.append(current_hunk)
                    hunk_lines = []
                current_file = line[4:].strip()
                current_hunk = None
            elif line.startswith("+++ "):
                pass
            elif line.startswith("@@ "):
                if current_hunk and hunk_lines:
                    current_hunk["lines"] = hunk_lines
                    hunks.append(current_hunk)
                    hunk_lines = []
                parts = line.split()
                old_start, old_count = self._parse_hunk_header(parts[1])
                new_start, new_count = self._parse_hunk_header(parts[2])
                current_hunk = {
                    "file": current_file,
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                }
            elif current_hunk is not None:
                if line.startswith(("+", "-", " ")):
                    hunk_lines.append(line)

        if current_hunk and hunk_lines:
            current_hunk["lines"] = hunk_lines
            hunks.append(current_hunk)

        return hunks


__all__ = ["DiffGenerator"]
