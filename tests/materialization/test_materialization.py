from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest

from ash_hawk.contracts import CuratedLesson
from ash_hawk.materialization import (
    FileFormat,
    GitRepoBackend,
    LessonMaterializer,
    MarkdownPayloadMapper,
    MaterializationConfig,
    PatchKind,
    PatchOperation,
    ProjectRegistry,
    PythonPayloadMapper,
)
from ash_hawk.strategies import Strategy, SubStrategy


def _lesson(
    lesson_id: str,
    lesson_type: Literal["policy", "skill", "tool", "harness", "eval"],
    title: str,
    description: str,
    *,
    payload: dict[str, Any] | None = None,
    strategy: Strategy | None = None,
    sub_strategies: list[SubStrategy] | None = None,
    applies_to_agents: list[str] | None = None,
) -> CuratedLesson:
    return CuratedLesson(
        lesson_id=lesson_id,
        source_proposal_id=f"proposal-{lesson_id}",
        applies_to_agents=applies_to_agents or ["test-agent"],
        lesson_type=lesson_type,
        title=title,
        description=description,
        lesson_payload=payload or {},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
        strategy=strategy,
        sub_strategies=sub_strategies or [],
    )


class TestMarkdownPayloadMapper:
    def test_can_map_skill_lesson_to_markdown(self) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson("lesson-1", "skill", "Voice lesson", "Description")
        assert mapper.can_map(lesson, FileFormat.MARKDOWN)

    def test_can_map_policy_lesson_to_markdown(self) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson("lesson-2", "policy", "Policy lesson", "Description")
        assert mapper.can_map(lesson, FileFormat.MARKDOWN)

    def test_cannot_map_tool_lesson_to_markdown(self) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson("lesson-3", "tool", "Tool lesson", "Description")
        assert not mapper.can_map(lesson, FileFormat.MARKDOWN)

    def test_cannot_map_to_python_format(self) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson("lesson-4", "skill", "Skill lesson", "Description")
        assert not mapper.can_map(lesson, FileFormat.PYTHON)

    def test_map_policy_lesson_to_policy_md(self, tmp_path: Path) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson(
            "lesson-policy",
            "policy",
            "Policy update",
            "Update policy constraints.",
            payload={"target_surface": "agent policy file"},
        )

        patches = mapper.map(lesson, tmp_path, FileFormat.MARKDOWN)

        assert len(patches) == 1
        assert patches[0].kind == PatchKind.APPEND_SECTION
        assert patches[0].path == "policy.md"
        assert patches[0].content is not None
        assert "lesson-policy" in patches[0].content

    def test_map_voice_lesson_to_voice_md(self, tmp_path: Path) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson(
            "lesson-voice",
            "skill",
            "Voice tone update",
            "Adjust voice guidelines.",
            payload={"target_surface": "voice and tone"},
        )

        patches = mapper.map(lesson, tmp_path, FileFormat.MARKDOWN)

        assert len(patches) == 1
        assert patches[0].path == "voice.md"

    def test_map_strategy_lesson_to_strategy_md(self, tmp_path: Path) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson(
            "lesson-strategy",
            "skill",
            "Strategy playbook",
            "Update playbook.",
            payload={"target_surface": "strategy playbook"},
        )

        patches = mapper.map(lesson, tmp_path, FileFormat.MARKDOWN)

        assert len(patches) == 1
        assert patches[0].path == "strategy.md"

    def test_map_includes_marker_for_idempotency(self, tmp_path: Path) -> None:
        mapper = MarkdownPayloadMapper()
        lesson = _lesson("lesson-marker", "skill", "Marker test", "Test markers.")

        patches = mapper.map(lesson, tmp_path, FileFormat.MARKDOWN)

        assert patches[0].content is not None
        assert "<!-- ash-hawk-lesson:lesson-marker -->" in patches[0].content


class TestPythonPayloadMapper:
    def test_can_map_skill_lesson_to_python(self) -> None:
        mapper = PythonPayloadMapper()
        lesson = _lesson("lesson-py", "skill", "Python skill", "Description")
        assert mapper.can_map(lesson, FileFormat.PYTHON)

    def test_can_map_tool_lesson_to_python(self) -> None:
        mapper = PythonPayloadMapper()
        lesson = _lesson("lesson-tool", "tool", "Tool lesson", "Description")
        assert mapper.can_map(lesson, FileFormat.PYTHON)

    def test_cannot_map_policy_lesson_to_python(self) -> None:
        mapper = PythonPayloadMapper()
        lesson = _lesson("lesson-policy", "policy", "Policy lesson", "Description")
        assert not mapper.can_map(lesson, FileFormat.PYTHON)

    def test_map_with_target_file_creates_patch(self, tmp_path: Path) -> None:
        mapper = PythonPayloadMapper()
        lesson = _lesson(
            "lesson-target",
            "skill",
            "Python patch",
            "Add code snippet.",
            payload={"target_file": "utils/helper.py", "code_snippet": "def foo(): pass"},
        )

        patches = mapper.map(lesson, tmp_path, FileFormat.PYTHON)

        assert len(patches) == 1
        assert patches[0].path == "utils/helper.py"

    def test_map_without_target_file_returns_empty(self, tmp_path: Path) -> None:
        mapper = PythonPayloadMapper()
        lesson = _lesson(
            "lesson-no-target",
            "skill",
            "No target",
            "Missing target file.",
            payload={},
        )

        patches = mapper.map(lesson, tmp_path, FileFormat.PYTHON)

        assert len(patches) == 0


class TestProjectRegistry:
    def test_register_and_get_target(self) -> None:
        registry = ProjectRegistry()
        registry.register(
            "test-agent",
            "/path/to/repo",
            default_format=FileFormat.MARKDOWN,
        )

        target = registry.get("test-agent")

        assert target is not None
        assert target.agent_id == "test-agent"
        assert target.default_format == FileFormat.MARKDOWN

    def test_get_nonexistent_target_returns_none(self) -> None:
        registry = ProjectRegistry()
        assert registry.get("nonexistent") is None

    def test_get_config_creates_materialization_config(self) -> None:
        registry = ProjectRegistry()
        registry.register(
            "configured-agent",
            "/repo/path",
            default_format=FileFormat.PYTHON,
        )

        config = registry.get_config(
            "configured-agent",
            run_lint=True,
            run_types=False,
        )

        assert config is not None
        assert config.agent_id == "configured-agent"
        assert config.repo_root == "/repo/path"
        assert config.run_lint is True
        assert config.run_types is False

    def test_list_agents(self) -> None:
        registry = ProjectRegistry()
        registry.register("agent-a", "/path/a")
        registry.register("agent-b", "/path/b")

        agents = registry.list_agents()

        assert set(agents) == {"agent-a", "agent-b"}


class TestGitRepoBackend:
    @pytest.mark.asyncio
    async def test_apply_append_section_creates_file(self, tmp_path: Path) -> None:
        backend = GitRepoBackend()
        config = MaterializationConfig(
            repo_root=str(tmp_path),
            agent_id="test-agent",
            run_lint=False,
            run_types=False,
        )
        patch = PatchOperation(
            kind=PatchKind.APPEND_SECTION,
            path="test.md",
            format=FileFormat.MARKDOWN,
            content="# Test Section\n\nContent here.",
            marker="<!-- test-marker -->",
        )

        result = await backend.apply([patch], config)

        assert result.error is None
        assert "test.md" in result.files_modified
        assert (tmp_path / "test.md").exists()
        content = (tmp_path / "test.md").read_text()
        assert "test-marker" in content

    @pytest.mark.asyncio
    async def test_apply_create_file_creates_new_file(self, tmp_path: Path) -> None:
        backend = GitRepoBackend()
        config = MaterializationConfig(
            repo_root=str(tmp_path),
            agent_id="test-agent",
            run_lint=False,
            run_types=False,
        )
        patch = PatchOperation(
            kind=PatchKind.CREATE_FILE,
            path="new_file.py",
            format=FileFormat.PYTHON,
            content='# New file\n"""Docstring."""\n',
            marker="# test-marker",
        )

        result = await backend.apply([patch], config)

        assert result.error is None
        assert "new_file.py" in result.files_modified
        assert (tmp_path / "new_file.py").exists()

    @pytest.mark.asyncio
    async def test_apply_skips_existing_marker(self, tmp_path: Path) -> None:
        backend = GitRepoBackend()
        config = MaterializationConfig(
            repo_root=str(tmp_path),
            agent_id="test-agent",
            run_lint=False,
            run_types=False,
        )
        existing_file = tmp_path / "existing.md"
        existing_file.write_text("<!-- existing-marker -->\n\nExisting content.")

        patch = PatchOperation(
            kind=PatchKind.APPEND_SECTION,
            path="existing.md",
            format=FileFormat.MARKDOWN,
            content="New content.",
            marker="<!-- existing-marker -->",
        )

        result = await backend.apply([patch], config)

        assert "existing.md" not in result.files_modified
        content = existing_file.read_text()
        assert "New content." not in content


class TestLessonMaterializer:
    @pytest.mark.asyncio
    async def test_materialize_creates_files(self, tmp_path: Path) -> None:
        registry = ProjectRegistry()
        registry.register(
            "test-agent",
            str(tmp_path),
            default_format=FileFormat.MARKDOWN,
        )
        backend = GitRepoBackend()
        materializer = LessonMaterializer(
            backend=backend,
            mappers=[MarkdownPayloadMapper()],
            registry=registry,
        )
        lesson = _lesson(
            "lesson-1",
            "skill",
            "Test lesson",
            "Description",
            payload={"target_surface": "policy"},
        )
        config = registry.get_config("test-agent", run_lint=False, run_types=False)
        assert config is not None

        result = await materializer.materialize(lesson, config)

        assert result.error is None
        assert result.lesson_id == "lesson-1"
        assert len(result.files_modified) > 0

    @pytest.mark.asyncio
    async def test_materialize_batch_processes_multiple_lessons(self, tmp_path: Path) -> None:
        registry = ProjectRegistry()
        registry.register(
            "batch-agent",
            str(tmp_path),
            default_format=FileFormat.MARKDOWN,
        )
        backend = GitRepoBackend()
        materializer = LessonMaterializer(
            backend=backend,
            mappers=[MarkdownPayloadMapper()],
            registry=registry,
        )
        lessons = [
            _lesson("batch-1", "skill", "Lesson 1", "Desc", payload={"target_surface": "policy"}),
            _lesson("batch-2", "skill", "Lesson 2", "Desc", payload={"target_surface": "voice"}),
        ]
        config = registry.get_config("batch-agent", run_lint=False, run_types=False)
        assert config is not None

        results = await materializer.materialize_batch(lessons, config)

        assert len(results) == 2
        assert results[0].lesson_id == "batch-1"
        assert results[1].lesson_id == "batch-2"
