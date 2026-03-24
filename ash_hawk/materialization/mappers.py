from __future__ import annotations

import json
import re
from pathlib import Path

from ash_hawk.contracts import CuratedLesson
from ash_hawk.materialization.base import PayloadMapper
from ash_hawk.materialization.types import FileFormat, PatchKind, PatchOperation


class MarkdownPayloadMapper(PayloadMapper):
    """Map lessons to markdown patch operations."""

    def can_map(self, lesson: CuratedLesson, target_format: FileFormat) -> bool:
        if target_format != FileFormat.MARKDOWN:
            return False
        return lesson.lesson_type in {"skill", "policy"}

    def map(
        self,
        lesson: CuratedLesson,
        repo_root: Path,
        target_format: FileFormat,
    ) -> list[PatchOperation]:
        target_path = self._resolve_target_path(lesson, repo_root)
        if target_path is None:
            return []

        marker = f"<!-- ash-hawk-lesson:{lesson.lesson_id} -->"
        section = self._render_section(lesson, marker)

        return [
            PatchOperation(
                kind=PatchKind.APPEND_SECTION,
                path=str(target_path.relative_to(repo_root)),
                format=FileFormat.MARKDOWN,
                content=section,
                marker=marker,
                section_title=lesson.title,
            )
        ]

    def _resolve_target_path(self, lesson: CuratedLesson, repo_root: Path) -> Path | None:
        surface = lesson.lesson_payload.get("target_surface", "")
        if not surface:
            surface = " ".join(lesson.applies_to_agents)
        surface = surface.lower()

        if "policy" in surface or lesson.lesson_type == "policy":
            return repo_root / "policy.md"
        if "voice" in surface or "tone" in surface:
            return repo_root / "voice.md"
        if "strategy" in surface or "playbook" in surface:
            return repo_root / "strategy.md"
        if "readme" in surface:
            return repo_root / "README.md"

        slug = re.sub(r"[^a-z0-9_]+", "_", surface).strip("_")
        if slug:
            return repo_root / f"{slug}.md"
        return None

    def _render_section(self, lesson: CuratedLesson, marker: str) -> str:
        lines = [
            marker,
            f"## {lesson.title}",
            f"- lesson_id: `{lesson.lesson_id}`",
            f"- lesson_type: `{lesson.lesson_type}`",
        ]
        if lesson.strategy is not None:
            lines.append(f"- strategy: `{lesson.strategy.value}`")
        if lesson.sub_strategies:
            values = ", ".join(f"`{sub}`" for sub in lesson.sub_strategies)
            lines.append(f"- sub_strategies: {values}")

        lines.append("")
        lines.append(lesson.description)
        if lesson.lesson_payload:
            payload_text = json.dumps(lesson.lesson_payload, indent=2, sort_keys=True, default=str)
            lines.extend(["", "```json", payload_text, "```"])
        return "\n".join(lines)


class PythonPayloadMapper(PayloadMapper):
    """Map lessons to Python patch operations."""

    def can_map(self, lesson: CuratedLesson, target_format: FileFormat) -> bool:
        if target_format != FileFormat.PYTHON:
            return False
        return lesson.lesson_type in {"skill", "tool", "harness"}

    def map(
        self,
        lesson: CuratedLesson,
        repo_root: Path,
        target_format: FileFormat,
    ) -> list[PatchOperation]:
        payload = lesson.lesson_payload
        patches: list[PatchOperation] = []

        target_file = payload.get("target_file")
        if target_file:
            path = self._resolve_python_path(target_file, repo_root)
            if path:
                patches.extend(self._map_to_python_ops(lesson, path, repo_root))

        return patches

    def _resolve_python_path(self, target: str, repo_root: Path) -> Path | None:
        if target.startswith("/"):
            return None
        candidate = repo_root / target
        if candidate.suffix != ".py":
            candidate = candidate.with_suffix(".py")
        return candidate

    def _map_to_python_ops(
        self,
        lesson: CuratedLesson,
        target: Path,
        repo_root: Path,
    ) -> list[PatchOperation]:
        payload = lesson.lesson_payload
        patches: list[PatchOperation] = []

        kind = payload.get("patch_kind", "append_section")
        rel_path = str(target.relative_to(repo_root))

        if kind == "append_section":
            content = self._render_python_section(lesson)
            patches.append(
                PatchOperation(
                    kind=PatchKind.APPEND_SECTION,
                    path=rel_path,
                    format=FileFormat.PYTHON,
                    content=content,
                    marker=f"# ash-hawk-lesson:{lesson.lesson_id}",
                    section_title=lesson.title,
                )
            )
        elif kind == "prepend_docstring" and "docstring" in payload:
            patches.append(
                PatchOperation(
                    kind=PatchKind.PREPEND_SECTION,
                    path=rel_path,
                    format=FileFormat.PYTHON,
                    content=payload["docstring"],
                    marker=f"# ash-hawk-lesson:{lesson.lesson_id}",
                    section_title=lesson.title,
                )
            )
        elif kind == "ast_transform" and "pattern" in payload and "replacement" in payload:
            patches.append(
                PatchOperation(
                    kind=PatchKind.AST_TRANSFORM,
                    path=rel_path,
                    format=FileFormat.PYTHON,
                    ast_pattern=payload["pattern"],
                    ast_replacement=payload["replacement"],
                    marker=f"# ash-hawk-lesson:{lesson.lesson_id}",
                )
            )

        return patches

    def _render_python_section(self, lesson: CuratedLesson) -> str:
        lines = [
            f"# ash-hawk-lesson:{lesson.lesson_id}",
            "",
            f"# {lesson.title}",
            f"# lesson_type: {lesson.lesson_type}",
            f"# source: {lesson.source_proposal_id}",
            "",
        ]

        description = lesson.description.strip()
        for line in description.splitlines():
            lines.append(f"# {line}")

        lines.append("")

        payload = lesson.lesson_payload
        if payload.get("code_snippet"):
            lines.append(payload["code_snippet"])

        return "\n".join(lines)
